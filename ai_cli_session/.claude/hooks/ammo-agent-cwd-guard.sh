#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse/Agent — BLOCK subagent spawn when orchestrator CWD has drifted.
# Subagents inherit the parent's cwd; relative paths in the spawn prompt resolve
# against it. The original bug: orchestrator cd'd into a track worktree to inspect
# a diff, then spawned subagents from there — they all got the wrong cwd.
# Now blocks (deny) so the orchestrator must cd back before spawning.
set -euo pipefail

INPUT="$(cat)"

# Inside a subagent (agent_type is set) — not our target. The bug is
# orchestrator-only; impl-champions legitimately spawn from track worktrees.
AGENT_TYPE=$(printf '%s' "$INPUT" | jq -r '.agent_type // ""' 2>/dev/null) || exit 0
[ -n "$AGENT_TYPE" ] && exit 0

CWD=$(printf '%s' "$INPUT" | jq -r '.cwd // empty')
ROOT="${CLAUDE_PROJECT_DIR:-}"

# Outside an AMMO session — no opinion.
[ -z "$ROOT" ] && exit 0
[ -z "$CWD" ] && exit 0

# Normalize trailing slashes to avoid false positives.
ROOT="${ROOT%/}"
CWD="${CWD%/}"

# Cwd matches session root — silent allow.
[ "$CWD" = "$ROOT" ] && exit 0

# Drift — block until orchestrator returns to session root.
jq -nc --arg cwd "$CWD" --arg root "$ROOT" \
  '{hookSpecificOutput:{
     hookEventName:"PreToolUse",
     permissionDecision:"deny",
     permissionDecisionReason:
       ("CWD drifted to \($cwd). Run: cd \($root)")
   }}'
exit 0
