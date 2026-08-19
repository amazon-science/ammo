#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# _ammo_msg_identity.sh — source-able helper shared by the AMMO message hooks
# (ammo-msg-check.sh reader, ammo-msg-mirror.sh sender mirror).
#
# Why this file exists: on Claude Code 2.1.202 every agent's REPL polls its
# own team inbox file every ~1s and PRUNES delivered messages EVEN WHILE the
# agent is busy inside a tool call. Measured live (2026-08-13, real AMMO
# session container): 321ms persistence for team-lead, 392ms for a tmux
# champion mid-90s-sleep; the reader hook saw a non-empty inbox 0 times in 14
# fires. The inbox file therefore cannot carry mid-turn messages. The mirror
# hook captures each successful SendMessage at the SENDER and appends it to a
# side channel that only these hooks touch; the reader injects from that
# channel. Both sides must resolve identity and channel paths the same way —
# that shared logic lives here.
#
# Usage:
#   HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$HELPER_DIR/_ammo_msg_identity.sh"
#   ammo_msg_resolve_identity "$INPUT"
#
# ammo_msg_resolve_identity sets (always defined, possibly empty):
#   AGENT_NAME            mailbox name ("team-lead" for the orchestrator)
#   TEAM_NAME             team the agent belongs to
#   RESOLVED_TRANSCRIPT   transcript path after EnterWorktree repair
#   TRANSCRIPT_IS_OWN     1 when RESOLVED_TRANSCRIPT is the agent's own
#   TEAMS_ROOT            ${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams
#
# Identity precedence (unchanged from the pre-split ammo-msg-check.sh):
#   1. agentName/teamName in the transcript head (tmux teammates)
#   2. agent-<id>.meta.json sidecar (CC 2.1.19x in-process teammates)
#   3. first-message "You are champion-N in the AMMO" regex (legacy)
#   4. hook-input .agent_type carrying an eligible teammate NAME
#   5. _ammo_is_lead + team-config leadSessionId scan → team-lead
#      (newest config wins: AMMO rounds reuse the lead session id, so several
#      configs can match; the current round's config is the newest)
#   6. member scan across team configs when TEAM_NAME is still empty
#
# Contract: functions never exit and never write stdout; callers own those
# decisions. Callers may define dbg() for debug lines; a no-op fallback is
# provided. Safe to source multiple times.

if ! declare -F ammo_msg_resolve_identity >/dev/null 2>&1; then

# Callers normally define dbg() before sourcing; keep a no-op fallback so a
# helper call can never crash a hook that forgot.
if ! declare -F dbg >/dev/null 2>&1; then
    dbg() { :; }
fi

_AMMO_MSG_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_AMMO_MSG_HELPER_DIR/_ammo_is_lead.sh" 2>/dev/null || true

# Strip anything that could escape a path component. Applied to every value
# that becomes a directory or file name — recipient names arrive from
# model-authored tool input, so "../../etc/x" must not steer a write.
ammo_msg_sanitize() {
    local s="${1:-}"
    s="${s//[^A-Za-z0-9._@-]/_}"
    s="${s//../_}"
    printf '%s' "${s:0:96}"
}

# Team names additionally get Claude Code's own dot→dash mangling so both
# hooks land on one canonical directory regardless of which form they saw.
ammo_msg_sanitize_team() {
    local t="${1:-}"
    t=$(printf '%s' "$t" | tr '.' '-')
    ammo_msg_sanitize "$t"
}

# Root of the side channel. Per-uid so two users never share a directory.
# Sessions in one server container share /tmp; channels are namespaced by
# team below the root (team names are unique per lead session and per round).
ammo_msg_mirror_root() {
    printf '%s' "${AMMO_MSG_MIRROR_DIR:-/tmp/ammo-msg-mirror-$(id -u 2>/dev/null || echo 0)}"
}

# Create the root 0700 and REFUSE to use it unless we own it and it is a real
# directory. The path is predictable, so another local user could pre-create
# it wide open and read or forge messages; a later chmod would fail with
# EPERM. Validate instead of assuming. Prints the root on success.
ammo_msg_ensure_root() {
    local root owner mode
    root=$(ammo_msg_mirror_root)
    [ -n "$root" ] || return 1
    # install -d applies the mode at creation, closing the create-then-chmod gap.
    install -d -m 700 "$root" 2>/dev/null || mkdir -p "$root" 2>/dev/null || return 1
    [ -L "$root" ] && return 1
    [ -d "$root" ] || return 1
    owner=$(stat -Lc '%u' "$root" 2>/dev/null) || return 1
    [ "$owner" = "$(id -u 2>/dev/null)" ] || return 1
    mode=$(stat -Lc '%a' "$root" 2>/dev/null) || return 1
    case "$mode" in
        700) ;;
        *) chmod 700 "$root" 2>/dev/null || return 1 ;;
    esac
    printf '%s' "$root"
}

# Compose the channel file path for a (team, agent) pair. Pure string; does
# not create or validate anything.
ammo_msg_channel_file() {
    local team agent
    team=$(ammo_msg_sanitize_team "${1:-}")
    agent=$(ammo_msg_sanitize "${2:-}")
    { [ -n "$team" ] && [ -n "$agent" ]; } || return 1
    printf '%s/%s/%s.jsonl' "$(ammo_msg_mirror_root)" "$team" "$agent"
}

# True only when path exists, is a regular file, is not a symlink, and we own
# it. Used before trusting any channel file, so a planted file cannot steer
# behaviour.
ammo_msg_own_file() {
    local p="${1:-}"
    [ -n "$p" ] || return 1
    [ -L "$p" ] && return 1
    [ -f "$p" ] || return 1
    [ -O "$p" ] || return 1
    return 0
}

ammo_msg_resolve_identity() {
    local _input="${1:-}"
    AGENT_NAME=""
    TEAM_NAME=""
    RESOLVED_TRANSCRIPT=""
    TRANSCRIPT_IS_OWN=0
    TEAMS_ROOT="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"

    command -v jq >/dev/null 2>&1 || return 0

    local TRANSCRIPT_PATH AGENT_TYPE SESSION_ID
    TRANSCRIPT_PATH=$(printf '%s' "$_input" | jq -r '.transcript_path // ""' 2>/dev/null) || TRANSCRIPT_PATH=""
    AGENT_TYPE=$(printf '%s' "$_input" | jq -r '.agent_type // ""' 2>/dev/null) || AGENT_TYPE=""
    SESSION_ID=$(printf '%s' "$_input" | jq -r '.session_id // ""' 2>/dev/null) || SESSION_ID=""

    RESOLVED_TRANSCRIPT="$TRANSCRIPT_PATH"

    # ── EnterWorktree breaks transcript_path — try parent ──
    if [ -n "$TRANSCRIPT_PATH" ] && [ ! -f "$TRANSCRIPT_PATH" ]; then
        local PARENT_DIR PARENT_TRANSCRIPT
        PARENT_DIR=$(echo "$(dirname "$TRANSCRIPT_PATH")" | sed 's/--claude-worktrees-[^/]*//')
        PARENT_TRANSCRIPT="$PARENT_DIR/$(basename "$TRANSCRIPT_PATH")"
        if [ -f "$PARENT_TRANSCRIPT" ]; then
            dbg "worktree-fix: using parent transcript: $PARENT_TRANSCRIPT"
            RESOLVED_TRANSCRIPT="$PARENT_TRANSCRIPT"
        fi
    fi

    if [ -n "$RESOLVED_TRANSCRIPT" ] && [ -f "$RESOLVED_TRANSCRIPT" ]; then
        local result
        result=$(head -5 "$RESOLVED_TRANSCRIPT" 2>/dev/null | jq -rs '
            first(.[] | select(.agentName) | [.agentName, .teamName // ""] | @tsv)
        ' 2>/dev/null) || true
        if [ -n "$result" ]; then
            IFS=$'\t' read -r AGENT_NAME TEAM_NAME <<< "$result"
            TRANSCRIPT_IS_OWN=1
        fi

        # ── Fallback A: meta.json sidecar (CC 2.1.19x in-process teammates) ──
        # 2.1.19x writes agentName:null in teammate transcripts; the spawned
        # name lives in the agent-<id>.meta.json sidecar as .name (named
        # spawns) or .agentType (typed spawns), with .teamName alongside.
        if [ -z "$AGENT_NAME" ]; then
            local META_FILE meta_result META_NAME META_TEAM
            META_FILE="${RESOLVED_TRANSCRIPT%.jsonl}.meta.json"
            if [ -f "$META_FILE" ]; then
                meta_result=$(jq -r '
                    [(.name // .agentType // ""), (.teamName // "")] | @tsv
                ' "$META_FILE" 2>/dev/null) || true
                if [ -n "$meta_result" ]; then
                    IFS=$'\t' read -r META_NAME META_TEAM <<< "$meta_result"
                    if [ -n "$META_NAME" ]; then
                        AGENT_NAME="$META_NAME"
                        [ -z "$TEAM_NAME" ] && TEAM_NAME="$META_TEAM"
                        TRANSCRIPT_IS_OWN=1
                        dbg "meta.json sidecar: agent=$AGENT_NAME team=$TEAM_NAME"
                    fi
                fi
            fi
        fi

        # ── Fallback B: legacy content regex (pre-2.1.19x tmux teammates) ──
        if [ -z "$AGENT_NAME" ]; then
            local FALLBACK
            FALLBACK=$(head -1 "$RESOLVED_TRANSCRIPT" 2>/dev/null | jq -r '
                .message.content // "" |
                capture("You are (?<name>(champion|impl-champion)-\\S+) in the AMMO") |
                .name
            ' 2>/dev/null) || true
            if [ -n "$FALLBACK" ]; then
                AGENT_NAME="$FALLBACK"
                TRANSCRIPT_IS_OWN=1
                dbg "content-based fallback: agent=$AGENT_NAME"
            fi
        fi
    fi

    # ── Fallback C: hook-input agent_type carries the teammate NAME ──
    # CC 2.1.198 sets .agent_type to the spawned teammate's name (e.g.
    # "champion-2") for in-process teammates. Accept it only when it matches
    # an eligible name pattern so plain subagent types (ammo-delegate, ...)
    # never slip through.
    if [ -z "$AGENT_NAME" ] && [ -n "$AGENT_TYPE" ]; then
        case "$AGENT_TYPE" in
            champion-*|impl-champion-*|monitor-champion-*|monitor-impl-champion-*)
                AGENT_NAME="$AGENT_TYPE"
                dbg "agent_type fallback: agent=$AGENT_NAME"
                ;;
        esac
    fi

    # ── Orchestrator detection via leadSessionId (via helper) ──
    # The orchestrator (team-lead) has no agentName in its transcript. Use the
    # shared _ammo_is_lead helper as the gating predicate. If it says "is
    # lead", scan team configs for those whose leadSessionId matches
    # SESSION_ID and adopt the NEWEST match: AMMO rounds reuse the lead
    # session id, so round-1 and round-2 configs can both match and only the
    # newest names the live team.
    if [ -z "$AGENT_NAME" ] && [ -n "$SESSION_ID" ]; then
        if declare -F _ammo_is_lead >/dev/null 2>&1 && _ammo_is_lead "$_input"; then
            local _candidate_team="" _best_mtime=-1 tdir lead_sid _mt _tname
            for tdir in "$TEAMS_ROOT"/*/; do
                [ -f "$tdir/config.json" ] || continue
                lead_sid=$(jq -r '.leadSessionId // empty' "$tdir/config.json" 2>/dev/null) || continue
                if [ "$lead_sid" = "$SESSION_ID" ]; then
                    _mt=$(stat -c %Y "$tdir/config.json" 2>/dev/null) || _mt=0
                    if [ "$_mt" -gt "$_best_mtime" ] 2>/dev/null; then
                        _tname=$(jq -r '.name // empty' "$tdir/config.json" 2>/dev/null) || _tname=""
                        [ -z "$_tname" ] && _tname=$(basename "${tdir%/}")
                        _candidate_team="$_tname"
                        _best_mtime="$_mt"
                    fi
                fi
            done
            if [ -n "$_candidate_team" ]; then
                AGENT_NAME="team-lead"
                TEAM_NAME="$_candidate_team"
                dbg "orchestrator via helper + team-config match → team=$TEAM_NAME"
            fi
        fi
    fi

    # ── Broader TEAM_NAME fallback ──
    # If we have an AGENT_NAME but no TEAM_NAME (e.g. transcript lacks
    # teamName), scan all team configs for a member with that name.
    if [ -n "$AGENT_NAME" ] && [ -z "$TEAM_NAME" ] && [ -d "$TEAMS_ROOT" ]; then
        local tdir2
        for tdir2 in "$TEAMS_ROOT"/*/; do
            [ -f "$tdir2/config.json" ] || continue
            if jq -e --arg n "$AGENT_NAME" '.members[]? | select(.name == $n)' "$tdir2/config.json" >/dev/null 2>&1; then
                TEAM_NAME=$(jq -r '.name // empty' "$tdir2/config.json" 2>/dev/null) || true
                [ -z "$TEAM_NAME" ] && TEAM_NAME=$(basename "${tdir2%/}")
                dbg "broader team fallback: agent=$AGENT_NAME → team=$TEAM_NAME"
                break
            fi
        done
    fi

    return 0
}

fi  # declare -F guard
