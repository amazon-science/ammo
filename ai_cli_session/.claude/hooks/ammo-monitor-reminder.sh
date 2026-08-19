#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PostToolUse hook — reminds AMMO orchestrator to spawn transcript-monitor
# after creating ammo-champion or ammo-impl-champion agents.
#
# Matcher: Agent (configured in settings.local.json)
# Only fires for orchestrator sessions (no agentName in transcript).
# Behavior: injects additionalContext, never blocks. Fail-open on any error.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Extract fields from hook input (separate calls to avoid @tsv tab-collapse)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // ""' 2>/dev/null) || true
AGENT_NAME=$(echo "$INPUT" | jq -r '.tool_input.name // ""' 2>/dev/null) || true
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // ""' 2>/dev/null) || true

# Only fire for impl-champion agent types (debate champions no longer get monitors)
case "$AGENT_TYPE" in
    ammo-impl-champion) ;;
    *) exit 0;;
esac

# Only fire for the lead orchestrator. Delegated to the shared
# _ammo_is_lead helper so post-compaction agentName=team-lead entries
# in the transcript don't silence the reminder. Champions/monitors still
# get suppressed via the helper's L3 (transcript agentName != team-lead).
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"
if ! _ammo_is_lead "$INPUT"; then
    exit 0
fi

# This reminder is advisory only. The blocking owner of monitor pairing is the
# state engine cohort check (ammo_state.py), which rejects a selected track that
# lacks durable implementer/monitor pairing evidence.

# Build reminder — works with or without a name
if [ -n "$AGENT_NAME" ]; then
    MONITOR_NAME="monitor-$AGENT_NAME"
    MSG="You just spawned $AGENT_NAME (type: $AGENT_TYPE). You MUST now spawn a corresponding transcript-monitor agent named $MONITOR_NAME with agentType=ammo-transcript-monitor to monitor this champion."
else
    MSG="You just spawned an unnamed $AGENT_TYPE agent. You MUST now spawn a corresponding transcript-monitor (agentType=ammo-transcript-monitor) to monitor this champion. NOTE: The champion was spawned without a name — consider re-spawning with a name parameter for proper team coordination."
fi

jq -c -n --arg msg "AMMO MONITOR REMINDER: $MSG Do this before spawning any other agents or doing other work." '
{
    hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: $msg
    }
}
'
exit 0
