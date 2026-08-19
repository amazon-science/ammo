#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Stop hook — blocks auditor from ending turn after Phase 1 without completing Phase 2.
#
# Logic: if a Phase 2 sentinel exists (inject-audit-phase2.sh fired) but the
# verdict file doesn't contain a "## Phase 2" section, block with instructions.
# Only fires for auditor subagents (agent_type present in input).
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Only fire for subagents (auditors are always spawned as subagents)
AGENT_TYPE=$(jq -r '.agent_type // empty' <<<"$INPUT")
[[ -z "$AGENT_TYPE" ]] && exit 0

SESSION_ID=$(jq -r '.session_id // "unknown"' <<<"$INPUT")

# ── SENTINEL NAME CONTRACT (shared with inject-audit-phase2.sh) ──
# The sentinel path is EXACTLY:
#
#   ${SENTINEL_PREFIX}${SESSION_ID}__${AGENT_ID:-noagent}__${VERDICT_BASE}
#
# with VERDICT_BASE = basename of the verdict file without its .md suffix.
# VERDICT_BASE is therefore everything after the LAST `__`. The DOUBLE
# underscore is load-bearing: session ids and agent ids both contain single
# underscores. This guard used to strip only "${SESSION_ID}_", so VERDICT_BASE
# kept the agent-id prefix and the find below never matched — the Stop block
# could never fire (inert since fcf1ea2).
#
# Any change to this expression must land in BOTH hooks in the same edit;
# test-ammo-audit-phase2.sh asserts the pair agrees.
SENTINEL_PREFIX="/tmp/ammo_audit_phase2_injected_"

# Check if any Phase 2 sentinel exists for this session
SENTINEL_MATCH=$(find /tmp -maxdepth 1 -name "$(basename "$SENTINEL_PREFIX")${SESSION_ID}__*" 2>/dev/null | head -1)
[[ -z "$SENTINEL_MATCH" ]] && exit 0

# Recover VERDICT_BASE: the trailing field after the last `__`.
VERDICT_BASE="${SENTINEL_MATCH##*__}"
[[ -z "$VERDICT_BASE" ]] && exit 0

# Find the actual verdict file by searching kernel_opt_artifacts
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
VERDICT_FILE=$(find "$PROJECT_DIR"/kernel_opt_artifacts -path "*/audits/${VERDICT_BASE}.md" 2>/dev/null | head -1)
[[ -z "$VERDICT_FILE" ]] && exit 0

# Check if Phase 2 section exists in the verdict file
if grep -q "^## Phase 2" "$VERDICT_FILE" 2>/dev/null; then
    exit 0
fi

# Phase 2 not written — block
jq -c -n '{
  "decision": "block",
  "reason": "Phase 2 (Checklist Verification) not yet written. Read audit-invariants.md and complete the checklist section before finishing."
}'
