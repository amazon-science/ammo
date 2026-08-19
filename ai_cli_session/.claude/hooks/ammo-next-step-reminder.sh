#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PostToolUse hook — inject concise next-step reminder for AMMO orchestrator,
# plus stage-specific Socratic reasoning chains that force the orchestrator to
# DERIVE its next decision rather than autopilot through it.
#
# Fires after state-mutating tool calls (Bash|Write|Edit). Looks at current
# state.json (and the cached previous snapshot) and emits guidance via
# hookSpecificOutput.additionalContext. Non-blocking (always exit 0).
#
# Skipped for subagents (agentName in transcript or CLAUDE_SUBAGENT=1).
# Throttled to once per 15s per session — reduced from 30s so the Socratic
# nudges land at decision time rather than after the next action is taken.
# Terminal-status nudges bypass the throttle (always fire on irreversible
# transitions to campaign_complete / campaign_exhausted).
#
# Message generation (the stage-ladder reminder + the 9 Socratic edge nudges +
# the threshold warning) is delegated to the python state engine
# (skills/ammo/scripts/ammo_state.py next-step). The hook keeps only the
# orchestration shell: lead gating, the worktree cwd-drift warning, state.json
# discovery, and the throttle + terminal-transition bypass. The engine
# implements the first-fire fix: with no PREV_STATE snapshot, NO Socratic edge
# nudges fire (an edge is undefined without a baseline) — only the stage-ladder
# reminder.
#
# PREV_STATE (/tmp/ammo-state-prev-<SID>.json) is READ here and WRITTEN by
# ammo-state-validate.sh — the blocking validator owns the ordering contract.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Skip subagents ──
# Delegated to the shared _ammo_is_lead helper — same precedence as
# ammo-stop-guard so the two hooks never disagree about who the lead is.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"
if ! _ammo_is_lead "$INPUT"; then
    exit 0
fi

# ── Worktree cwd drift warning ──
# The orchestrator is supposed to drive the campaign from the main repo.
# When its cwd is inside .claude/worktrees/<basename>/, any python/git/state
# writes land in an isolated op worktree and never reach the main branch.
# Warn once per (session_id, worktree basename) BEFORE running the normal
# reminder so the orchestrator sees the drift instead of stage guidance.
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null) || true
case "$CWD" in
    */.claude/worktrees/*)
        WT_BASENAME=$(echo "$CWD" | sed -E 's#.*/\.claude/worktrees/([^/]+).*#\1#')
        _SID_FOR_WT="${CLAUDE_SESSION_ID:-$(echo "$INPUT" | jq -r '.session_id // "default"' 2>/dev/null)}"
        WT_MARKER="/tmp/ammo-worktree-warned-${_SID_FOR_WT}-${WT_BASENAME}"
        if [ ! -f "$WT_MARKER" ]; then
            touch "$WT_MARKER"
            jq -c -n --arg msg "AMMO CWD DRIFT: Orchestrator cwd is inside .claude/worktrees/${WT_BASENAME}/. Operations (state.json edits, git commits, python runs) from here stay in the isolated op worktree and will NOT reach the main branch. cd back to the main repo before resuming campaign-level work; worktrees are for Stage 4-5 parallel-track agents only." \
                '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:$msg}}'
            exit 0
        fi
        ;;
esac

SESSION_ID="${CLAUDE_SESSION_ID:-$(echo "$INPUT" | jq -r '.session_id // "default"' 2>/dev/null)}"

# ── Resolve the state engine (hook-relative; the test harness copies neither
#    scripts nor schemas into its tmp project dir, so hook-relative resolution
#    is REQUIRED). Advisory hook → fail-open if python3 / engine is missing. ──
ENGINE="$HELPER_DIR/../skills/ammo/scripts/ammo_state.py"
HOOK_PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -x "$HOOK_PROJ/.venv/bin/python" ]; then
    ENGINE_PY="$HOOK_PROJ/.venv/bin/python"
else
    ENGINE_PY="python3"
fi
if ! command -v "$ENGINE_PY" &>/dev/null; then exit 0; fi
[ -f "$ENGINE" ] || exit 0

# ── Find state.json (BEFORE throttle so we can detect terminal-status bypass) ──
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE=""
ARTIFACT_DIR=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] || continue
    STATE_FILE="$d/state.json"
    ARTIFACT_DIR="$d"
    break
done
[ -z "$STATE_FILE" ] && exit 0

# ── PREV_STATE: READ ONLY here ──
# ammo-state-validate.sh (the BLOCKING validator) owns this snapshot: it reads
# the file as its pre-write baseline, then overwrites it after the engine runs.
# This advisory hook only reads it. Writing it here made a blocking tier/scope
# gate's grandfathering baseline depend on an advisory hook's side effect and on
# PostToolUse array order.
PREV_STATE="${AMMO_REMINDER_STATE_DIR:-/tmp}/ammo-state-prev-${SESSION_ID}.json"

# ── Determine if this is a TERMINAL transition (bypasses throttle) ──
# Ask the engine: terminal-transition detection requires PREV_STATE to exist
# AND prev status to differ (first-fire fix — no edge without a baseline).
IS_TERMINAL_TRANSITION=$("$ENGINE_PY" "$ENGINE" next-step --state "$STATE_FILE" --prev "$PREV_STATE" --print-terminal 2>/dev/null) || IS_TERMINAL_TRANSITION=0
[ "$IS_TERMINAL_TRANSITION" = "1" ] || IS_TERMINAL_TRANSITION=0

# ── Throttle (15s per session). Terminal transitions bypass. ──
MARKER="/tmp/ammo-reminder-last-${SESSION_ID}"
NOW=$(date +%s)
if [ "$IS_TERMINAL_TRANSITION" -eq 0 ]; then
    if [ -f "$MARKER" ]; then
        LAST=$(stat -c %Y "$MARKER" 2>/dev/null || stat -f %m "$MARKER" 2>/dev/null || echo 0)
        if [ $(( NOW - LAST )) -lt 15 ]; then
            exit 0
        fi
    fi
fi
touch "$MARKER"

# ── Generate the message via the engine (stage-ladder reminder + Socratic edges
#    + threshold warning). The engine emits the compact hookSpecificOutput JSON
#    on a non-empty message, or nothing when there is nothing to say. ──
MSG_JSON=$("$ENGINE_PY" "$ENGINE" next-step --state "$STATE_FILE" --prev "$PREV_STATE" --emit hook 2>/dev/null) || MSG_JSON=""

# No PREV_STATE write here — ammo-state-validate.sh refreshes it.

if [ -n "$MSG_JSON" ]; then
    echo "$MSG_JSON"
fi
exit 0
