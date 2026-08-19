#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PostToolUse hook — validates state.json against .claude/schemas/state.schema.json.
#
# Fires on Write|Edit (file_path match) AND on Bash (command-string detection).
# Delegates schema + cross-field validation to scripts/ammo_state.py (validate
# --emit hook). Blocks via decision:block on violations.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || true

# --- Bash detection path ---
# When fired on Bash, tool_input has .command not .file_path.
# Detect state.json writes by inspecting the command string.
if [ -z "$FILE_PATH" ]; then
    # Skip if the Bash command failed (write likely didn't complete)
    TOOL_STATUS=$(echo "$INPUT" | jq -r '.tool_response.status // "success"' 2>/dev/null) || true
    [ "$TOOL_STATUS" = "error" ] && exit 0

    COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null) || true
    [ -z "$COMMAND" ] && exit 0

    # Fast filter: bail unless the command names state.json, or names a
    # first-party writer that resolves the path itself from --artifact-dir.
    case "$COMMAND" in
        *state.json*) ;;
        *'reconcile_track_state.py'*) ;;
        *) exit 0;;
    esac

    # Bail if CLAUDE_PROJECT_DIR is not set (can't locate state.json reliably)
    [ -z "${CLAUDE_PROJECT_DIR:-}" ] && exit 0

    # Exclude read-only commands (cat, grep, jq without redirect, head, tail, etc.)
    # Only proceed if the command looks like a write (>, mv, tee, open(...,'w'), json.dump)
    LOOKS_LIKE_WRITE=false
    case "$COMMAND" in
        *'> '*state.json*|*'>>'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'mv '*state.json*|*'mv '*'state.json'*) LOOKS_LIKE_WRITE=true;;
        *'tee '*state.json*) LOOKS_LIKE_WRITE=true;;
        *"open("*state.json*"'w'"*) LOOKS_LIKE_WRITE=true;;
        *"open("*state.json*'"w"'*) LOOKS_LIKE_WRITE=true;;
        *'open('*state.json*'\"w\"'*) LOOKS_LIKE_WRITE=true;;
        *'json.dump'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'json.dump'*'state.json'*) LOOKS_LIKE_WRITE=true;;
        *'write_text'*state.json*|*'write_bytes'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'sed -i'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'cp '*state.json*) LOOKS_LIKE_WRITE=true;;
        # Idioms below match anywhere in the command (the *state.json* fast
        # filter above already guarantees the command touches state.json).
        # Ordered patterns like *'json.dump'*state.json* miss the common
        # path-in-a-variable form (path='...state.json' BEFORE json.dump(s,tmp));
        # a live campaign wrote invalid state through exactly that gap via
        # tempfile.NamedTemporaryFile + os.replace.
        *'json.dump'*) LOOKS_LIKE_WRITE=true;;
        *'os.replace('*|*'os.rename('*) LOOKS_LIKE_WRITE=true;;
        *'NamedTemporaryFile'*|*'mkstemp'*) LOOKS_LIKE_WRITE=true;;
        *'shutil.move'*|*'shutil.copy'*) LOOKS_LIKE_WRITE=true;;
        *'write_text'*|*'write_bytes'*) LOOKS_LIKE_WRITE=true;;
        *'ammo_state.py'*' set '*|*'ammo_state.py'*' advance'*|*'ammo_state.py'*' enrich'*|*'ammo_state.py'*' backfill'*) LOOKS_LIKE_WRITE=true;;
        # reconcile_track_state.py --write is the canonical per-track lead write
        # path (parallel-tracks.md § Track State Reconciliation). It reached
        # state.json completely unvalidated until this arm existed; the Codex
        # twin (post_tool_use_guard.py:_first_party_state_paths) already had it.
        *'reconcile_track_state.py'*'--write'*) LOOKS_LIKE_WRITE=true;;
    esac
    [ "$LOOKS_LIKE_WRITE" = "false" ] && exit 0

    # Extract the actual target path from the command.
    # Look for kernel_opt_artifacts/*/state.json patterns in the command text.
    # Allow path chars including those inside quotes (Python open('path/state.json','w'))
    EXTRACTED=$(echo "$COMMAND" | grep -oE '[A-Za-z0-9_./-]*kernel_opt_artifacts/[A-Za-z0-9_./-]+/state\.json' | head -1) || true

    # reconcile_track_state.py takes --artifact-dir, not a state.json path, so
    # derive state.json from the dir (matching the Codex twin's
    # `_resolve_payload_path(...) / "state.json"`). Both `--artifact-dir X` and
    # `--artifact-dir=X` spellings.
    if [ -z "$EXTRACTED" ]; then
        case "$COMMAND" in
            *'reconcile_track_state.py'*)
                _AD=$(echo "$COMMAND" | grep -oE -- '--artifact-dir[= ]+[A-Za-z0-9_./-]+' | head -1 | sed -E 's/^--artifact-dir[= ]+//') || true
                [ -n "$_AD" ] && EXTRACTED="${_AD%/}/state.json"
                ;;
        esac
    fi

    if [ -n "$EXTRACTED" ]; then
        # Resolve relative path against CLAUDE_PROJECT_DIR
        if [[ "$EXTRACTED" = /* ]]; then
            FILE_PATH="$EXTRACTED"
        else
            FILE_PATH="$CLAUDE_PROJECT_DIR/$EXTRACTED"
        fi
    else
        # Command mentions state.json + looks like a write but path doesn't contain
        # kernel_opt_artifacts — this is an unrelated state.json write. Bail.
        exit 0
    fi

    [ -f "$FILE_PATH" ] || exit 0
else
    # --- Write/Edit path (original logic) ---
    case "$FILE_PATH" in
        */kernel_opt_artifacts/*/state.json) ;;
        *) exit 0;;
    esac
    [ -f "$FILE_PATH" ] || exit 0
fi

# Walk up from state.json to find the schema.
# Stop at .git boundary to avoid picking up unrelated schemas.
DIR=$(dirname "$FILE_PATH")
SCHEMA=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -f "$DIR/.claude/schemas/state.schema.json" ]; then
        SCHEMA="$DIR/.claude/schemas/state.schema.json"
        break
    fi
    PARENT=$(dirname "$DIR")
    [ "$PARENT" = "$DIR" ] && break
    # Stop at git root — don't walk above the worktree
    [ -d "$DIR/.git" ] && break
    DIR="$PARENT"
done

[ -z "$SCHEMA" ] && exit 0

# Delegate ALL validation (schema + cross-field Stage-6 guard + audit gates +
# new-round-start gate) to the python state engine. The engine emits compact
# block JSON on a violation (silent on pass) and always exits 0; we echo its
# output so the PostToolUse hook surfaces the decision:block to Claude Code.
#
# Resolve the engine relative to THIS hook's own dir (not CWD) — the test
# harness runs the hook from a tmp project dir, so a CWD-relative path would
# miss it.
ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../skills/ammo/scripts/ammo_state.py"

# Interpreter: prefer the session worktree venv (guaranteed to have jsonschema)
# over bare python3 — a jsonschema-less system python3 would degrade every
# validation into a fail-closed block (the "VALIDATION FOOTGUN" from live runs).
HOOK_PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -x "$HOOK_PROJ/.venv/bin/python" ]; then
    ENGINE_PY="$HOOK_PROJ/.venv/bin/python"
else
    ENGINE_PY="python3"
fi

# P4#6 FAIL-CLOSED POINT (DEFAULT-ON): if python3 or the engine is unavailable,
# validation CANNOT run. Emit a decision:block so the PostToolUse hook surfaces
# it (exit-code is NOT the enforcement mechanism — the block JSON is). Escape
# hatch AMMO_VALIDATE_FAIL_OPEN=1 reverts to the legacy fail-open for degraded
# environments. NOTE: this is the python3/engine-missing branch only — the
# missing-schema-FILE fast-bail at the `[ -z "$SCHEMA" ] && exit 0` line above
# stays fail-open (deliberate vanished-schema carve-out).
if ! command -v "$ENGINE_PY" &>/dev/null || [ ! -f "$ENGINE" ]; then
    if [ "${AMMO_VALIDATE_FAIL_OPEN:-}" = "1" ]; then exit 0; fi
    REASON="state.json validation could not run (python3 or the AMMO state engine is unavailable) — restore the session .venv/python3 or set AMMO_VALIDATE_FAIL_OPEN=1 to bypass (degraded)."
    jq -cn --arg r "$REASON" '{"decision":"block","reason":$r,"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":$r}}'
    exit 0
fi

# ── PREV_STATE snapshot: THIS hook owns it ──
# ORDERING CONTRACT (one sentence): this blocking validator reads
# /tmp/ammo-state-prev-<SID>.json as the PRE-write baseline, then OVERWRITES it
# with the post-write state after the engine has run.
#
# The snapshot write used to live in ammo-next-step-reminder.sh, an ADVISORY
# hook. That made a blocking tier/scope gate's grandfathering baseline depend on
# an advisory hook's side effect and on the order of the PostToolUse array. The
# blocking hook now owns both halves, so the contract holds no matter what else
# is registered. The Codex twin keeps the same read-then-write order
# (post_tool_use_guard.py:693-700 reads, then writes after validate).
#
# The gates enforce only NEW or MODIFIED scoreboard/candidate entries. An absent
# or stale snapshot makes the engine grandfather every pre-existing entry (never
# a false block); enforcement of a NEW bad write works once the snapshot exists.
# AMMO_REMINDER_STATE_DIR overrides the snapshot root (same env name the Codex
# twin uses). Test harnesses set it so runs are isolated from each other and
# from a live session's /tmp state.
SESSION_ID="${CLAUDE_SESSION_ID:-$(echo "$INPUT" | jq -r '.session_id // "default"' 2>/dev/null)}"
PREV_STATE="${AMMO_REMINDER_STATE_DIR:-/tmp}/ammo-state-prev-${SESSION_ID}.json"
PREV_ARGS=()
[ -f "$PREV_STATE" ] && PREV_ARGS=(--prev "$PREV_STATE")

# Engine present: delegate. When the jsonschema LIB is missing or the state
# JSON won't parse, the engine itself emits decision:block (fail-closed,
# default-on, honoring AMMO_VALIDATE_FAIL_OPEN). --fail-closed documents intent
# (default is already on). We echo the engine's stdout so the block surfaces.
OUT=$("$ENGINE_PY" "$ENGINE" validate --state "$FILE_PATH" --schema "$SCHEMA" "${PREV_ARGS[@]}" --emit hook --fail-closed 2>/dev/null) || true

# Refresh the snapshot AFTER validate, so at validate time it still held the
# pre-write state. Written atomically (tmp + mv) so a concurrent reader never
# sees a half-copied file. Best-effort: a snapshot failure must not turn into a
# block, it only makes the NEXT validate grandfather more entries.
if [ -f "$FILE_PATH" ]; then
    cp "$FILE_PATH" "${PREV_STATE}.tmp.$$" 2>/dev/null \
        && mv -f "${PREV_STATE}.tmp.$$" "$PREV_STATE" 2>/dev/null \
        || rm -f "${PREV_STATE}.tmp.$$" 2>/dev/null || true
fi

[ -n "$OUT" ] && echo "$OUT"
exit 0
