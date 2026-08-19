#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# _ammo_is_lead.sh — source-able helper for AMMO hooks.
#
# Consolidates 4 divergent lead-detection paths (stop-guard, next-step-reminder,
# monitor-reminder, msg-check) into a single predicate.
#
# Usage:
#   HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$HELPER_DIR/_ammo_is_lead.sh"
#   if ! _ammo_is_lead "$INPUT"; then
#       exit 0
#   fi
#
# Contract:
#   - $1 = hook input JSON string (not consumed from stdin)
#   - return 0 = IS lead (orchestrator / team-lead)
#   - return 1 = NOT lead (subagent, teammate)
#   - never writes stdout (hooks reserve stdout for JSON)
#   - never writes stderr on the happy path (audit-grade silent)
#   - never calls `exit` (would kill the sourcing hook)
#   - idempotent — safe to source multiple times
#
# Precedence (first match wins):
#   L1  .agent_type non-empty               → NOT lead
#   L2  CLAUDE_SUBAGENT=1 env                → NOT lead
#   L3  transcript head -5 agentName:
#           == "team-lead"                   → IS  lead  (post-compaction)
#           != "team-lead" (non-empty)       → NOT lead
#   L4  any team config leadSessionId match  → IS  lead
#   L4b any team config members[].sessionId
#           match (and no L4 match)          → NOT lead
#       team configs exist but none names me → falls through to L5
#   L5  no positive subagent signal          → IS  lead  (default)
#   L6  any error                            → IS  lead  (fail-open)
#
# Why an L4 miss falls THROUGH instead of returning NOT lead:
# subagents are positively identified by L1 (.agent_type), L2
# (CLAUDE_SUBAGENT=1) and L4b (config names my session as a member); a config
# that names my session NOWHERE identifies nobody. Team configs are S3-synced
# and survive across sessions, and the pinned CC lead transcript carries no
# agentName, so ONE stale config with a dead leadSessionId used to demote the
# real lead in every hook that sources this helper. A stale config is the
# common case; an unlabeled subagent that also evades L1, L2 and L4b is not.
#
# All command substitutions capture stdout into variables so nothing can
# leak to the caller's fd 1 (which hooks reserve for JSON). Every pipe
# end also carries a 2>/dev/null to prevent jq / head diagnostics from
# appearing on the caller's fd 2 when a pathological input is passed.

# Guard: define only once so repeated sourcing (e.g. when multiple hooks
# run in one shell) is a no-op.
if ! declare -F _ammo_is_lead >/dev/null 2>&1; then

_ammo_is_lead() {
    # All errors → fail-open (return 0). We never want the helper to
    # silence the lead orchestrator because of a parsing hiccup.
    local _input="${1:-}"

    # jq must be available; without it fail-open. stdout+stderr silenced.
    command -v jq >/dev/null 2>&1 || return 0

    # ── L1: .agent_type non-empty → NOT lead ──
    local _agent_type=""
    _agent_type=$(printf '%s' "$_input" 2>/dev/null | jq -r '.agent_type // empty' 2>/dev/null) || _agent_type=""
    if [ -n "$_agent_type" ]; then
        return 1
    fi

    # ── L2: CLAUDE_SUBAGENT env → NOT lead ──
    if [ "${CLAUDE_SUBAGENT:-0}" = "1" ]; then
        return 1
    fi

    # ── L3: transcript head -5 agentName ──
    local _transcript=""
    _transcript=$(printf '%s' "$_input" 2>/dev/null | jq -r '.transcript_path // empty' 2>/dev/null) || _transcript=""
    if [ -n "$_transcript" ] && [ -f "$_transcript" ]; then
        local _agent_name=""
        _agent_name=$(head -5 "$_transcript" 2>/dev/null | \
            jq -rs 'first(.[] | select(.agentName) | .agentName) // ""' 2>/dev/null) || _agent_name=""
        if [ "$_agent_name" = "team-lead" ]; then
            return 0
        fi
        if [ -n "$_agent_name" ]; then
            return 1
        fi
    fi

    # ── L4/L4b/L5: team config scan ──
    local _session_id=""
    _session_id=$(printf '%s' "$_input" 2>/dev/null | jq -r '.session_id // empty' 2>/dev/null) || _session_id=""

    local _teams_root="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"

    # Fast-path: no teams dir at all → L5 fail-open to lead.
    [ -d "$_teams_root" ] || return 0

    local _tdir _cfg _lead_sid _member_hit
    local _matched=0
    local _member_matched=0

    # Safe glob: enable nullglob so non-matches don't iterate over the
    # literal glob string. Save/restore shell options so we don't perturb
    # the caller's environment.
    local _restore_nullglob=0
    if ! shopt -q nullglob 2>/dev/null; then
        if shopt -s nullglob 2>/dev/null; then
            _restore_nullglob=1
        fi
    fi

    for _tdir in "$_teams_root"/*/; do
        _cfg="$_tdir/config.json"
        [ -f "$_cfg" ] || continue
        if [ -n "$_session_id" ]; then
            _lead_sid=$(jq -r '.leadSessionId // empty' "$_cfg" 2>/dev/null) || _lead_sid=""
            if [ "$_lead_sid" = "$_session_id" ]; then
                _matched=1
                break
            fi
            # L4b: config names me as a member → positive teammate signal.
            _member_hit=$(jq -r --arg sid "$_session_id" \
                'if any((.members // [])[]?; (.sessionId // "") == $sid) then "1" else "" end' \
                "$_cfg" 2>/dev/null) || _member_hit=""
            if [ "$_member_hit" = "1" ]; then
                _member_matched=1
            fi
        fi
    done

    if [ "$_restore_nullglob" = "1" ]; then
        shopt -u nullglob 2>/dev/null || true
    fi

    if [ "$_matched" = "1" ]; then
        return 0
    fi

    # ── L4b: a config names me as a member → NOT lead ──
    if [ "$_member_matched" = "1" ]; then
        return 1
    fi

    # ── L5: no positive subagent signal → IS lead ──
    # Reached when team configs exist but none names this session (typically a
    # stale S3-synced config) AND L1/L2/L3/L4b all declined to identify a
    # subagent.
    return 0
}

fi  # declare -F guard
