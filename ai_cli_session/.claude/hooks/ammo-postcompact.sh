#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# SessionStart hook for AMMO orchestrator
# Injects resume context after compaction

# emit_context: print one hookSpecificOutput document, or the empty document
# when jq failed. Never print partial JSON; the harness never sees a half doc.
emit_context() {
    local doc
    if doc=$(jq -n "$@" 2>/dev/null) && [ -n "$doc" ]; then
        printf '%s\n' "$doc"
    else
        printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":""}}'
    fi
}

# Clean up stale GPU reservation warning flags from previous sessions.
# Preserve the current session's flag so post-compaction restarts don't
# re-trigger the one-shot CVD warning.
if [ -n "${CLAUDE_SESSION_ID:-}" ]; then
    for _f in /tmp/ammo_gpu_res/.warned_*; do
        [ -f "$_f" ] || continue
        [ "$_f" = "/tmp/ammo_gpu_res/.warned_${CLAUDE_SESSION_ID}" ] && continue
        rm -f "$_f"
    done
else
    rm -f /tmp/ammo_gpu_res/.warned_* 2>/dev/null || true
fi

CHECKPOINT_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "compaction_checkpoint.json" 2>/dev/null | head -1)

if [ -n "$CHECKPOINT_FILES" ]; then
    STATE_DIR=$(dirname "$CHECKPOINT_FILES")
    STATE_FILE="$STATE_DIR/state.json"

    MODEL=$(jq -r '.model // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    STAGE=$(jq -r '.stage // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    TEAM_NAME=$(jq -r '.team_name // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    DEBATE_TEAM=$(jq -r '.debate_team // ""' "$CHECKPOINT_FILES" 2>/dev/null)
    TRACK_COUNT=$(jq -r '.track_count // 0' "$CHECKPOINT_FILES" 2>/dev/null)
    CAMPAIGN_ROUND=$(jq -r '.campaign_round // 0' "$CHECKPOINT_FILES" 2>/dev/null)
    CAMPAIGN_STATUS=$(jq -r '.campaign_status // ""' "$CHECKPOINT_FILES" 2>/dev/null)
    CUMULATIVE_SPEEDUP=$(jq -r '.cumulative_speedup // 1.0' "$CHECKPOINT_FILES" 2>/dev/null)

    # Read current state for more context (v2 shape)
    if [ -f "$STATE_FILE" ]; then
        CURRENT_STATUS=$(jq -r '.campaign.status // "unknown"' "$STATE_FILE" 2>/dev/null)
        ROUND_SUMMARY=$(jq -r '
          (.campaign.current_round) as $cr |
          .campaign.rounds[$cr - 1].round_summary // ""
        ' "$STATE_FILE" 2>/dev/null)
    else
        CURRENT_STATUS="unknown"
        ROUND_SUMMARY=""
    fi

    # Build additional resume context for v2 features.
    # EXTRA_CONTEXT holds real newlines; jq --arg escapes them for JSON.
    EXTRA_CONTEXT=""
    if [ -n "$DEBATE_TEAM" ] && [ "$DEBATE_TEAM" != "" ]; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}
6. **Team active ($DEBATE_TEAM)**: Check campaign.rounds[current_round-1] for debate/champion status."
    fi
    if [ "$TRACK_COUNT" -gt 0 ] 2>/dev/null; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}
7. **Parallel tracks ($TRACK_COUNT active)**: Check campaign.rounds[current_round-1].parallel_tracks.tracks in state.json for worktree paths and GPU assignments."
    fi
    if [ -n "$CAMPAIGN_STATUS" ] && [ "$CAMPAIGN_STATUS" != "" ]; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}
8. **Campaign loop active**: Round $CAMPAIGN_ROUND | Status: $CAMPAIGN_STATUS | Cumulative speedup: ${CUMULATIVE_SPEEDUP}x. See Campaign Loop section in SKILL.md."
    fi

    # Clean up checkpoint
    rm -f "$CHECKPOINT_FILES"

    # Build with jq (never shell interpolation): round_summary and other state
    # values hold agent prose, so a newline, quote or backslash would make an
    # interpolated heredoc invalid JSON and silently drop the whole briefing.
    emit_context \
        --arg state_file "$STATE_FILE" \
        --arg extra "$EXTRA_CONTEXT" \
        --arg model "$MODEL" \
        --arg stage "$STAGE" \
        --arg status "$CURRENT_STATUS" \
        --arg summary "$ROUND_SUMMARY" \
        '{
          hookSpecificOutput: {
            hookEventName: "SessionStart",
            additionalContext: (
              "# Session Resumed After Compaction\n\n## You Are The AMMO Lead Orchestrator\n\n"
              + "This session was compacted while orchestrating an AMMO optimization.\n\n"
              + "### Immediate Actions\n\n"
              + "1. **Read the skill**: `.claude/skills/ammo/SKILL.md`\n"
              + "2. **Load state**: `cat " + $state_file + "`\n"
              + "3. **Resume current stage** — spawn subagents as needed"
              + $extra + "\n\n"
              + "### Model: " + $model + " | Stage: " + $stage + " | Status: " + $status + "\n"
              + "### Round summary: " + $summary + "\n\n"
              + "You are the LEAD — scaffold, delegate, gate. Do not implement directly."
            )
          }
        }'

else
    # No checkpoint — check for active state
    STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "state.json" 2>/dev/null | head -1)

    if [ -n "$STATE_FILES" ]; then
        MODEL=$(jq -r '.target.model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
        STAGE=$(jq -r '.campaign.current_stage // "unknown"' "$STATE_FILES" 2>/dev/null)

        emit_context \
            --arg state_file "$STATE_FILES" \
            --arg model "$MODEL" \
            --arg stage "$STAGE" \
            '{
              hookSpecificOutput: {
                hookEventName: "SessionStart",
                additionalContext: (
                  "# AMMO Optimization Detected\n\n"
                  + "Existing optimization state at: `" + $state_file + "`\n"
                  + "Model: " + $model + " | Stage: " + $stage + "\n\n"
                  + "If continuing this optimization:\n"
                  + "- Read the skill: `.claude/skills/ammo/SKILL.md`\n"
                  + "- You are the lead orchestrator — scaffold, delegate, gate. Do not implement directly."
                )
              }
            }'
    else
        cat << 'EMPTY_EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": ""
  }
}
EMPTY_EOF
    fi
fi
