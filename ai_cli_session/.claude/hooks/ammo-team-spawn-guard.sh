#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — enforce that team-member spawns carry an addressable name
# under the CC 2.1.179 implicit-team model.
#
# CC 2.1.179 model (ELF-verified):
#   - TeamCreate / TeamDelete are REMOVED. There is no team-CRUD tool to gate on.
#   - CC auto-forms ONE implicit team per session (name = session-<first8ofSessionId>),
#     gated on CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1. The team is created lazily when
#     the first named teammate is spawned via the Agent tool. The main session is lead.
#   - Team membership now derives PURELY from spawning with a `name`. The `name` param
#     makes the teammate addressable via SendMessage({to: name}).
#   - The Agent `team_name` param is DEPRECATED & IGNORED ("the session has a single
#     implicit team"). It is neither required nor checked here.
#
# Member types (form/join the implicit team): ammo-champion, ammo-impl-champion,
# ammo-transcript-monitor. Every other subagent_type is either a one-shot subagent or a
# named, addressable helper (e.g. the eval pipeline's general-purpose causal-analyzer /
# transcript-grader) — both are legitimate under the implicit-team model.
#
# Matcher: Agent (Task is dead in CC 2.1.121+; see probes/PROBE_FINDINGS.md).
#
# Behavior:
#   - Runs inside a subagent (.agent_type set) → ALLOW unconditionally
#     (spawn gating is an orchestrator-only concern; a champion spawning a
#     delegate/investigator as a subagent must not be blocked).
#   - Member type (ammo-champion / ammo-impl-champion / ammo-transcript-monitor) with an
#     EMPTY `name` → DENY (a member with no name cannot be addressed via SendMessage).
#   - Member type WITH a non-empty `name` → ALLOW (it forms/joins the implicit team).
#     team_name is neither required nor checked; no team-directory check.
#   - Any other subagent_type → ALLOW (named subagents are addressable and legitimate;
#     the name just makes them reachable via SendMessage).
#
# Deny format: {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#              "permissionDecision":"deny","permissionDecisionReason":"..."}}
# Always exits 0 (the JSON drives the decision). Fail-open on any error.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq >/dev/null 2>&1; then
    exit 0
fi

INPUT=$(cat)

# Inside-subagent short-circuit: .agent_type at top level is only populated
# when this hook fires from a subagent process. On the orchestrator it is
# empty. Spawn gating is an orchestrator-only concern.
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // ""' 2>/dev/null) || exit 0
if [ -n "$AGENT_TYPE" ]; then
    exit 0
fi

SUBAGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // ""' 2>/dev/null) || exit 0
NAME=$(echo "$INPUT" | jq -r '.tool_input.name // ""' 2>/dev/null) || exit 0

emit_deny() {
    local reason="$1"
    jq -c -n --arg reason "$reason" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $reason
        }
    }'
    exit 0
}

case "$SUBAGENT_TYPE" in
    ammo-champion|ammo-impl-champion|ammo-transcript-monitor)
        # Team-member types: must carry a non-empty `name` so they are addressable
        # via SendMessage({to: name}) and can receive the shutdown_request handshake.
        # Under the implicit-team model the name is what forms/joins the team; no
        # team_name and no team-directory are required.
        if [ -z "$NAME" ]; then
            emit_deny "AMMO spawn-guard: $SUBAGENT_TYPE must spawn with a non-empty tool_input.name so it is addressable via SendMessage({to: <name>}) and can receive the shutdown_request handshake. Add a descriptive name (e.g. 'champion-1', 'impl-op001', 'monitor-champion-1') and re-spawn. Do NOT pass team_name — the session has a single implicit team and team_name is ignored."
        fi
        exit 0
        ;;
    *)
        # Every other type is a one-shot subagent or a named, addressable helper.
        # Under the implicit-team model a named spawn is legitimate (the name just
        # makes it reachable via SendMessage), so we ALLOW unconditionally.
        exit 0
        ;;
esac
