#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PostToolUse hook — stamps audit.{stage}.started_at + cycle when the
# orchestrator dispatches an ammo-auditor.
#
# Matcher: Agent (configured in settings.local.json).
#
# Why PostToolUse and not PreToolUse: PreToolUse fires before the permission
# decision, so a DENIED spawn would still stamp a start time for an audit that
# never ran. Why not SubagentStart: its payload carries no prompt, and the
# prompt is where the stage/round/cycle live.
#
# Why the lead does not write these fields: started_at and cycle are the
# provenance record that an auditor was really dispatched. Only passed_at stays
# lead-written, after verdict review. ammo_state.py's schema-4.2 backstop
# rejects a gate with passed_at and no started_at, so this hook is the sole
# writer of the start stamp.
#
# Behavior:
#   - Not the lead (_ammo_is_lead says no) → exit 0. Only the orchestrator
#     dispatches audits; a champion that quotes the dispatch format in its own
#     prompt must never stamp. The shared predicate covers .agent_type,
#     CLAUDE_SUBAGENT=1, the transcript agentName and the team member lists —
#     a tmux teammate carries none of the first two.
#   - tool_response reports an error → exit 0. PostToolUse fires for failed
#     tool calls too, and a spawn that never launched must not attest that an
#     auditor started (the same reason PreToolUse was rejected).
#   - tool_input.subagent_type != "ammo-auditor" → exit 0.
#   - Parses task / artifact_dir / stage / round / cycle out of
#     tool_input.prompt (the canonical dispatch block in
#     skills/ammo/orchestration/audit-protocol.md). Leading whitespace and
#     inline `#` comments are tolerated. Any missing or invalid field → exit 0.
#   - run_in_background is NOT inspected: a foreground dispatch stamps too.
#
# Entirely FAIL-OPEN and silent: no stdout, no blocking JSON, every path exits 0.
# The stamp is visibility, never a gate on the spawn itself.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq >/dev/null 2>&1; then exit 0; fi
if ! command -v python3 >/dev/null 2>&1; then exit 0; fi

INPUT=$(cat)

# Lead-only short-circuit. .agent_type alone is not enough: a tmux teammate
# carries no .agent_type and no CLAUDE_SUBAGENT, so the shared predicate is the
# only check that covers every rung (see _ammo_is_lead.sh L1-L4b). Same idiom
# ammo-monitor-reminder.sh uses on this matcher.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"
if ! _ammo_is_lead "$INPUT"; then exit 0; fi

# A spawn the runtime rejected must not attest that an auditor started.
TOOL_STATUS=$(jq -r '.tool_response.status // "success"' <<<"$INPUT" 2>/dev/null) || exit 0
if [ "$TOOL_STATUS" = "error" ]; then exit 0; fi
SPAWN_ERR=$(jq -r '
    def resp: (.tool_response // .toolResponse // {}) | if type == "object" then . else {} end;
    [resp.error, resp.is_error, resp.isError, .is_error, .isError]
    | map(select(. != null and . != false and . != ""))
    | length' <<<"$INPUT" 2>/dev/null) || exit 0
if [ "${SPAWN_ERR:-0}" != "0" ]; then exit 0; fi

SUBAGENT_TYPE=$(jq -r '.tool_input.subagent_type // ""' <<<"$INPUT" 2>/dev/null) || exit 0
if [ "$SUBAGENT_TYPE" != "ammo-auditor" ]; then exit 0; fi

PROMPT=$(jq -r '.tool_input.prompt // ""' <<<"$INPUT" 2>/dev/null) || exit 0
if [ -z "$PROMPT" ]; then exit 0; fi

# Read one `key: value` line out of the dispatch block. First match wins.
# Strips leading indentation, an inline `# comment`, and trailing spaces.
# No `head` in the pipeline on purpose: under `set -o pipefail` a SIGPIPE from
# an early-closing reader turns into exit 141, which the ERR trap would swallow
# as a silent no-stamp. The first line is taken with parameter expansion.
dispatch_field() {
    local all first
    all=$(printf '%s\n' "$PROMPT" | sed -n -E "s/^[[:space:]]*$1:[[:space:]]*(.*)\$/\\1/p") || return 1
    first=${all%%$'\n'*}
    printf '%s' "$first" | sed -E 's/[[:space:]]*#.*$//; s/[[:space:]]+$//'
}

TASK=$(dispatch_field task) || exit 0
if [ "$TASK" != "audit_gate" ]; then exit 0; fi

ARTIFACT_DIR=$(dispatch_field artifact_dir) || exit 0
STAGE=$(dispatch_field stage) || exit 0
ROUND=$(dispatch_field round) || exit 0
CYCLE=$(dispatch_field cycle) || exit 0

# stage must be one of the four dispatchable gates (ammo_state.py
# AUDIT_GATE_STAGES). stage_6 / stage_7 are pre-consolidation schema leftovers
# and are never dispatched.
case "$STAGE" in
    stage_1|stage_2|stage_45|stage_67) ;;
    *) exit 0 ;;
esac

# round and cycle are 1-based counters.
if ! printf '%s' "$ROUND" | grep -qE '^[1-9][0-9]*$'; then exit 0; fi
if ! printf '%s' "$CYCLE" | grep -qE '^[1-9][0-9]*$'; then exit 0; fi

if [ -z "$ARTIFACT_DIR" ] || [ ! -d "$ARTIFACT_DIR" ]; then exit 0; fi

# Resolve the state engine relative to THIS hook's own dir, the same way
# ammo-state-validate.sh does. Never walk up from artifact_dir: artifact_dir
# arrives in the spawn prompt and the artifact tree is writable by the campaign
# agent, so a walk-up lets prompt content choose the code this hook executes.
ENGINE="$HELPER_DIR/../skills/ammo/scripts/ammo_state.py"
if [ ! -f "$ENGINE" ]; then exit 0; fi

# Swallow the exit code: the engine already exits 0 on both fail-open no-ops
# (round absent, round with no audit key), and a real error must not surface
# here — the stamp is advisory.
python3 "$ENGINE" audit-started \
    --artifact-dir "$ARTIFACT_DIR" \
    --stage "$STAGE" \
    --round "$ROUND" \
    --cycle "$CYCLE" >/dev/null 2>&1 || true

exit 0
