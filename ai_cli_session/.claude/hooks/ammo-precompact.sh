#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreCompact hook for AMMO orchestrator
# Saves orchestration context from kernel_opt_artifacts/*/state.json
# Reads v2 state.json shape (round-centric).

STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "state.json" 2>/dev/null | head -1)

if [ -n "$STATE_FILES" ]; then
    STATE_DIR=$(dirname "$STATE_FILES")

    # Canonical v2 jq pattern: CR = campaign.current_round (1-based),
    # IDX = CR - 1 (0-based array index). rounds[IDX] is always present.
    CR=$(jq -r '.campaign.current_round // 1' "$STATE_FILES" 2>/dev/null)
    IDX=$(( CR - 1 ))

    MODEL=$(jq -r '.target.model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
    STAGE=$(jq -r '.campaign.current_stage // "unknown"' "$STATE_FILES" 2>/dev/null)
    STATUS=$(jq -r '.campaign.status // "unknown"' "$STATE_FILES" 2>/dev/null)
    TEAM_NAME=$(jq -r ".campaign.rounds[$IDX].team_name // \"unknown\"" "$STATE_FILES" 2>/dev/null)
    DEBATE_TEAM=$(jq -r ".campaign.rounds[$IDX].team_name // \"\"" "$STATE_FILES" 2>/dev/null)
    TRACK_COUNT=$(jq -r "(.campaign.rounds[$IDX].parallel_tracks.tracks // []) | length" "$STATE_FILES" 2>/dev/null)
    CAMPAIGN_ROUND=$CR
    CAMPAIGN_STATUS="$STATUS"
    CUMULATIVE_SPEEDUP=$(jq -r '.campaign.cumulative_speedup_vs_round1 // 1.0' "$STATE_FILES" 2>/dev/null)

    # Create checkpoint for restoration.
    # Build with jq (never shell interpolation): any value may hold a newline,
    # a quote or a backslash, which would make an interpolated heredoc invalid
    # JSON and silently break the SessionStart resume briefing. Numbers pass as
    # strings and are coerced by `tonumber?`, so a failed jq read cannot emit a
    # bare `,`. Write to a temp file and rename, so a reader never sees a
    # partial checkpoint.
    CHECKPOINT_FILE="$STATE_DIR/compaction_checkpoint.json"
    if [ -L "$CHECKPOINT_FILE" ]; then
        # Refuse to write through a symlink.
        exit 0
    fi
    CHECKPOINT_TMP="$STATE_DIR/.compaction_checkpoint.json.$$.tmp"
    if jq -n \
        --arg timestamp "$(date -Iseconds)" \
        --arg model "$MODEL" \
        --arg stage "$STAGE" \
        --arg status "$STATUS" \
        --arg team_name "$TEAM_NAME" \
        --arg debate_team "$DEBATE_TEAM" \
        --arg track_count "$TRACK_COUNT" \
        --arg campaign_round "$CAMPAIGN_ROUND" \
        --arg campaign_status "$CAMPAIGN_STATUS" \
        --arg cumulative_speedup "$CUMULATIVE_SPEEDUP" \
        --arg state_file "$STATE_FILES" \
        '{
          checkpoint_type: "pre_compaction",
          timestamp: $timestamp,
          model: $model,
          stage: $stage,
          status: $status,
          team_name: $team_name,
          debate_team: $debate_team,
          track_count: ($track_count | tonumber? // 0),
          campaign_round: ($campaign_round | tonumber? // 0),
          campaign_status: $campaign_status,
          cumulative_speedup: ($cumulative_speedup | tonumber? // 1.0),
          state_file: $state_file,
          skill_path: ".claude/skills/ammo/SKILL.md"
        }' > "$CHECKPOINT_TMP" 2>/dev/null; then
        mv -f "$CHECKPOINT_TMP" "$CHECKPOINT_FILE"
    else
        rm -f "$CHECKPOINT_TMP"
    fi

    # PreCompact only supports exit codes (0=allow, 2=block), not hookSpecificOutput.
    # Context injection happens in the SessionStart hook (ammo-postcompact.sh).
    exit 0

else
    # No AMMO state — allow compaction
    exit 0
fi
