#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Stop hook — AMMO orchestrator continuation nudge.
#
# Fires for the ORCHESTRATOR (not teammates) when campaign artifacts exist
# and work remains. Exits silently when:
#   - No state.json in kernel_opt_artifacts/
#   - Campaign is terminal (complete/exhausted) AND REPORT.md exists
#
# Nudges at:
#   - Stage 7 (active): proceed to next round or set terminal status + 7b
#   - Stage 7 (terminal, no report): spawn report subagent (closes the 7b gap)
#   - Stage 7b: spawn report subagent
#
# Teammates are excluded via the agent_type JSON field (present for subagents,
# absent for the lead) with a fallback to session_id / team-config check.
#
# Uses file-based one-shot circuit breaker (keyed by session_id):
#   1st stop attempt: create marker file, nudge with stage-specific prompt
#   2nd stop attempt: marker file exists → allow through
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Skip for subagents/teammates ──
# Delegated to the shared _ammo_is_lead helper. Contract: returns 0 for the
# lead orchestrator (or solo orchestrator fail-open), 1 for teammates /
# subagents. See _ammo_is_lead.sh for precedence details.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"
if ! _ammo_is_lead "$INPUT"; then
    exit 0
fi

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# ── Circuit breaker ──
MARKER="/tmp/ammo-stop-nudged-${SESSION_ID}"
if [ -f "$MARKER" ]; then
    rm -f "$MARKER"
    exit 0
fi

# ── Find campaign ──
STATE_FILE=""
ARTIFACT_DIR=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] || continue
    STATE_FILE="$d/state.json"
    ARTIFACT_DIR="$d"
    break
done
[ -z "$STATE_FILE" ] && exit 0  # no campaign artifacts

# ── Fully done check: terminal status + report exists → nothing left to do ──
STATUS=$(jq -r '.campaign.status // empty' "$STATE_FILE" 2>/dev/null)
if [[ "$STATUS" != "active" ]] && [[ -f "${ARTIFACT_DIR}REPORT.md" ]]; then
    exit 0
fi

# ── Paused check: user explicitly paused → allow stop ──
if [[ "$STATUS" = "paused" ]]; then
    exit 0
fi

STAGE=$(jq -r '.campaign.current_stage // "unknown"' "$STATE_FILE" 2>/dev/null)
ROUND=$(jq -r '.campaign.current_round // 1' "$STATE_FILE" 2>/dev/null)

# Stale-state reminders live in ammo-prompt-reminder.sh (UserPromptSubmit).
# Stop hooks cannot inject non-blocking context (hookSpecificOutput is
# silently dropped, and decision:block forces continuation).

# ── Stage-specific nudge ──
# Only nudge at stages where the orchestrator should keep going.
NUDGE=""
case "$STAGE" in
    7_campaign_eval*)
        if [ "$STATUS" = "active" ]; then
            NUDGE="You are at Stage 7 (Campaign Evaluation). Do NOT stop or ask the user.
Execute mechanical threshold check: read f (top bottleneck share) from profiling data, compare to min_e2e_improvement_pct in state.json.
- If f >= threshold: update state to next round and continue (mining on new baseline if SHIP, pivot technology if EXHAUSTED).
- If f < threshold: set campaign status to campaign_complete or campaign_exhausted, then IMMEDIATELY proceed to Stage 7b (spawn report subagent)."
        else
            # Terminal status but no REPORT.md (we passed the fully-done check above)
            NUDGE="Campaign is $STATUS but REPORT.md has not been generated.
Spawn the report subagent NOW: read .claude/skills/ammo/report/SKILL.md and spawn a
general-purpose subagent to generate REPORT.md in ${ARTIFACT_DIR}."
        fi
        ;;
    7b_report*|*report_gen*)
        NUDGE="You are at Stage 7b (Report Generation). Spawn the report subagent now:
Read .claude/skills/ammo/report/SKILL.md and spawn a general-purpose subagent to generate REPORT.md.
Do NOT stop without spawning the report subagent."
        ;;
esac

[ -z "$NUDGE" ] && exit 0

touch "$MARKER"

cat >&2 <<EOF
AMMO: Campaign at stage $STAGE, status $STATUS, round $ROUND.

$NUDGE
EOF
exit 2
