#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse/Edit|Write|MultiEdit|NotebookEdit — BLOCK subagents from editing
# orchestrator-owned skill/agent definition files.
#
# Protected paths (substring match):
#   .claude/skills/ammo/references/
#   .claude/skills/ammo/orchestration/
#   .claude/agents/
#   .claude/skills/ammo/scripts/     — the Mechanical Authorities
#   .claude/schemas/                 — the schema a subagent's work is judged by
#
# audit-invariants.md § Mechanical Authorities names scripts/ and schemas/ as
# the two locations that decide whether a track passes. A subagent that can
# rewrite the state engine, the validation-gate verifier, or state.schema.json
# can pass itself, so both are now on the protected set. The team lead is not
# a subagent and stays exempt, so a real fix still lands in one turn.
#
# _ammo_shared/ needs no entry: scripts/render_ammo_variants.py renders it INTO
# the two variant trees, and cli_tool_manager.py copies only
# ai_cli_session/.claude (line ~323) or ai_cli_session/.codex (line ~412) into a
# session worktree. _ammo_shared/ never reaches a session, so no agent can edit
# it from inside one.
#
# Orchestrator (team-lead) is exempt — it can modify these files freely.
# Subagents get a deny with a message telling them to surface the change upstream.
set -euo pipefail
trap 'exit 0' ERR

command -v jq >/dev/null 2>&1 || exit 0

INPUT=$(cat)

# Extract file path (Edit/Write/MultiEdit/NotebookEdit all use file_path).
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""' 2>/dev/null) || exit 0
[ -z "$FILE_PATH" ] && exit 0

# Only gate paths under protected directories.
case "$FILE_PATH" in
    *".claude/skills/ammo/references/"*) ;;
    *".claude/skills/ammo/orchestration/"*) ;;
    *".claude/skills/ammo/scripts/"*) ;;
    *".claude/schemas/"*) ;;
    *".claude/agents/"*) ;;
    *) exit 0 ;;
esac

# Orchestrator check — source the shared helper.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"

if _ammo_is_lead "$INPUT"; then
    exit 0
fi

# Subagent → deny.
jq -nc --arg path "$FILE_PATH" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("Subagents may not edit \($path). Files under .claude/skills/ammo/references/, .claude/skills/ammo/orchestration/, .claude/skills/ammo/scripts/, .claude/schemas/, and .claude/agents/ are orchestrator-owned. scripts/ and schemas/ are the Mechanical Authorities that decide whether your own work passes, so editing them from inside a track is never the fix. Surface the proposed change to the orchestrator via SendMessage instead.")
  }
}'
exit 0
