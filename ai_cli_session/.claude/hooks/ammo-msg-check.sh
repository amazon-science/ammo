#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — AMMO mid-turn message injection (non-blocking).
#
# Problem: SendMessage delivery is queued between turns. Champions are
# deaf during long tool execution chains (GPU benchmarks, E2E sweeps).
#
# Approach: Inject undelivered messages as additionalContext BEFORE each
# tool call. Never deny — champions keep working with full awareness.
#
# Sources (merged, deduped):
#   1. The real team inbox <teams>/<team>/inboxes/<agent>.json. On CC
#      2.1.202 this is a dead source: every agent's REPL polls its own
#      inbox ~1s and prunes delivered messages even while the agent is busy
#      inside a tool call (measured live 2026-08-13: 321-392ms persistence,
#      0 non-empty sightings in 14 hook fires). Kept for older CC versions
#      and as a harmless secondary source.
#   2. The sender-side mirror channel written by ammo-msg-mirror.sh
#      (PostToolUse on SendMessage). The poller never touches it, so it is
#      the source that actually works on 2.1.202.
#
# Dedup: Sidecar file tracks last-injected timestamp. Only new messages
# since last injection get injected. Includes "ignore if delivered again"
# context so turn-end delivery of the same messages doesn't confuse the
# agent. The read-decide-advance sequence runs under a file lock so
# parallel tool calls cannot double-inject the same batch.
#
# Cleanup: Sidecar deleted when all messages have been delivered
# (msg_ts <= delivery_ts in transcript).
#
# Identity: shared with the mirror hook — see _ammo_msg_identity.sh.
# Targets: champion-*, impl-champion-*, team-lead (orchestrator),
#          monitor-champion-*, monitor-impl-champion-*.
# Applies to all tool calls. Fail-open (exit 0 on any error).
set -euo pipefail
trap 'exit 0' ERR

DEBUG="${AMMO_MSG_CHECK_DEBUG:-}"
DLOG="/tmp/ammo-msg-check-debug.log"
dbg() { [ -n "$DEBUG" ] && echo "[$(date +%T)] $*" >> "$DLOG" || true; }

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Parse hook input ──
# Query each field separately — bash `read` with tab IFS collapses
# consecutive tabs (tab is a whitespace class member), which silently
# shifts fields when a middle field (e.g. agent_type) is empty.
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null) || TOOL_NAME=""
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // ""' 2>/dev/null) || TRANSCRIPT_PATH=""
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // ""' 2>/dev/null) || AGENT_TYPE=""
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // ""' 2>/dev/null) || SESSION_ID=""
dbg "tool=$TOOL_NAME agent_type=$AGENT_TYPE session_id=$SESSION_ID transcript=$TRANSCRIPT_PATH"

[ -z "$TOOL_NAME" ] && exit 0

# ── Identity (shared with ammo-msg-mirror.sh) ──
# Sets AGENT_NAME, TEAM_NAME, RESOLVED_TRANSCRIPT, TRANSCRIPT_IS_OWN,
# TEAMS_ROOT. See _ammo_msg_identity.sh for the precedence order.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HELPER_DIR/_ammo_msg_identity.sh" 2>/dev/null || exit 0
ammo_msg_resolve_identity "$INPUT"

dbg "agent=$AGENT_NAME team=$TEAM_NAME"
[ -z "$AGENT_NAME" ] && exit 0

# ── Resolve the agent's OWN transcript for delivery detection ──
# CC 2.1.19x passes the LEAD session's transcript_path in hook input for
# in-process teammates. Delivery detection must scan the teammate's own
# transcript (<lead>/subagents/agent-<id>.jsonl) — the lead's transcript
# contains <teammate-message> entries delivered TO THE LEAD, and keying the
# cutoff on those can wrongly mark this agent's inbox as delivered.
# Locate it via the meta.json sidecars (name for named spawns, agentType for
# typed spawns); newest match wins (respawns create fresh transcripts).
# If no own transcript is found, use NO delivery cutoff rather than the
# lead's: on 2.1.19x CC drains the inbox after turn-boundary delivery, so
# every inbox message is undelivered by construction, while the lead's later
# deliveries would wrongly suppress injection.
DELIVERY_TRANSCRIPT="$RESOLVED_TRANSCRIPT"
if [ "$AGENT_NAME" != "team-lead" ] && [ "$TRANSCRIPT_IS_OWN" != "1" ]; then
    case "$(basename "$RESOLVED_TRANSCRIPT" 2>/dev/null)" in
        agent-*.jsonl) ;;  # already the agent's own transcript
        *)
            DELIVERY_TRANSCRIPT=""
            SUB_DIR="${RESOLVED_TRANSCRIPT%.jsonl}/subagents"
            if [ -d "$SUB_DIR" ]; then
                own=$(ls -t "$SUB_DIR"/agent-*.meta.json 2>/dev/null | while read -r mf; do
                    mname=$(jq -r '.name // .agentType // empty' "$mf" 2>/dev/null) || continue
                    if [ "$mname" = "$AGENT_NAME" ] && [ -f "${mf%.meta.json}.jsonl" ]; then
                        echo "${mf%.meta.json}.jsonl"
                        break
                    fi
                done) || own=""
                if [ -n "$own" ] && [ -f "$own" ]; then
                    DELIVERY_TRANSCRIPT="$own"
                    dbg "delivery transcript → own subagent transcript: $own"
                else
                    dbg "own transcript not found — no delivery cutoff (inbox drains on 2.1.19x)"
                fi
            fi
            ;;
    esac
fi

# ── Eligibility filter ──
# Eligible: champion-*, impl-champion-*, team-lead, monitor-champion-*,
# monitor-impl-champion-*. Anything else (verifier-1, researcher, etc.)
# is out of scope and skipped.
case "$AGENT_NAME" in
    champion-*|impl-champion-*|team-lead|monitor-champion-*|monitor-impl-champion-*) ;;
    *) dbg "skip: agentName $AGENT_NAME not in eligible set"; exit 0;;
esac

# ── Locate the two message sources ──
# Claude Code sanitizes team dir names (dots → dashes), so try both.
TEAM_DIR="$TEAMS_ROOT/$TEAM_NAME"
if [ ! -d "$TEAM_DIR" ]; then
    SANITIZED=$(echo "$TEAM_NAME" | tr '.' '-')
    TEAM_DIR="$TEAMS_ROOT/$SANITIZED"
    dbg "team dir sanitized: $SANITIZED"
fi
MY_INBOX="$TEAM_DIR/inboxes/$AGENT_NAME.json"

MIRROR_FILE=$(ammo_msg_channel_file "$TEAM_NAME" "$AGENT_NAME" 2>/dev/null) || MIRROR_FILE=""
HAVE_MIRROR=0
if [ -n "$MIRROR_FILE" ] && [ -s "$MIRROR_FILE" ] && ammo_msg_own_file "$MIRROR_FILE"; then
    HAVE_MIRROR=1
fi
if [ ! -f "$MY_INBOX" ] && [ "$HAVE_MIRROR" != "1" ]; then
    exit 0
fi

# ── Merge inbox + mirror into one message array ──
# Streamed (no --argjson) so a large channel cannot hit the kernel argv
# limit. Dedup by (from, text): the same message can sit in both sources
# for a moment, and their timestamps differ (CC stamp vs mirror stamp).
MERGED=$( {
    if [ -f "$MY_INBOX" ]; then
        jq -c 'if type == "array" then .[] else empty end' "$MY_INBOX" 2>/dev/null || true
    fi
    if [ "$HAVE_MIRROR" = "1" ]; then
        jq -c -R 'try fromjson | select(type == "object")' "$MIRROR_FILE" 2>/dev/null || true
    fi
    true
} | jq -c -s 'unique_by([(.from // ""), ((.text // "") | tostring)]) | sort_by(.timestamp // "")' 2>/dev/null) || MERGED='[]'
[ -n "$MERGED" ] || MERGED='[]'

# ── Sidecar file: tracks last-injected timestamp ──
# Scoped by team name: teammate names (champion-1, ...) repeat across
# concurrent sessions on a shared /tmp; the implicit team name
# (session-<sid8>) is unique per lead session and prevents cross-session
# dedup pollution.
SIDECAR="/tmp/ammo-msg-injected-${TEAM_NAME:+${TEAM_NAME}-}${AGENT_NAME}.ts"

# ── Self-identity forms for the .from filter ──
# CC writes .from inconsistently across versions: plain name ("champion-1"),
# qualified agentId ("champion-1@session-abc"), or the raw transcript hash id
# ("achampion-1-c601798..."). Exclude all three so self-sent messages never
# re-inject.
SELF_ID="$AGENT_NAME"
[ -n "$TEAM_NAME" ] && SELF_ID="$AGENT_NAME@$TEAM_NAME"
RAW_ID=""
case "$(basename "${DELIVERY_TRANSCRIPT:-}" 2>/dev/null)" in
    agent-*.jsonl) RAW_ID=$(basename "$DELIVERY_TRANSCRIPT" .jsonl); RAW_ID="${RAW_ID#agent-}";;
esac

# ── Get newest non-self message timestamp ──
LAST_INBOX_TS=$(printf '%s' "$MERGED" | jq -r --arg me "$AGENT_NAME" --arg meid "$SELF_ID" --arg meraw "$RAW_ID" '
    [.[] | select(.from != $me and .from != $meid and ($meraw == "" or .from != $meraw)) | .timestamp // empty]
    | map(select(. != null and . != ""))
    | if length == 0 then "" else max end
' 2>/dev/null) || exit 0

# No non-self messages → nothing to inject
[ -z "$LAST_INBOX_TS" ] && exit 0
dbg "last_inbox_ts=$LAST_INBOX_TS"

# ── Get last delivery timestamp from the agent's own transcript ──
# Bounded tail: transcripts grow to hundreds of MB over a long session and
# this hook runs before every tool call. A delivery older than the last 4MB
# is older than anything the cutoff logic needs. Tool results and meta
# entries can quote the tag text verbatim (grep output, file reads), so
# require a real user-turn entry: string content, no toolUseResult, not meta.
LAST_DELIVERY_TS=""
if [ -n "${DELIVERY_TRANSCRIPT:-}" ] && [ -f "$DELIVERY_TRANSCRIPT" ]; then
    LAST_DELIVERY_TS=$(tail -c 4000000 "$DELIVERY_TRANSCRIPT" 2>/dev/null | jq -R -r 'try fromjson
        | select(.type == "user")
        | select(has("toolUseResult") | not)
        | select(.isMeta != true)
        | select(.message.content | type == "string")
        | select(.message.content | test("<teammate-message teammate_id="))
        | .timestamp // empty
    ' 2>/dev/null | tail -1) || true
fi

dbg "last_delivery_ts=${LAST_DELIVERY_TS:-(none)}"

# ── If all messages delivered, clean up sidecar and exit ──
if [ -n "$LAST_DELIVERY_TS" ]; then
    if [ "$LAST_INBOX_TS" \< "$LAST_DELIVERY_TS" ] || [ "$LAST_INBOX_TS" = "$LAST_DELIVERY_TS" ]; then
        dbg "all delivered: inbox_ts=$LAST_INBOX_TS <= delivery_ts=$LAST_DELIVERY_TS — cleaning sidecar"
        rm -f "$SIDECAR" 2>/dev/null || true
        exit 0
    fi
fi

# ── Cursor read + injection build + cursor write under ONE lock ──
# Parallel tool calls fire this hook concurrently; without the lock every
# call reads the same stale cursor and injects the same batch (measured 8/8
# duplicate injections on the global twin of this hook before the fix).
# Early exits below release the lock via fd close. Fail-open when flock is
# unavailable.
exec 8>"$SIDECAR.lock" 2>/dev/null || true
flock -w 5 8 2>/dev/null || true

# ── Check sidecar: skip if we already injected up to this point ──
LAST_INJECTED_TS=""
if [ -f "$SIDECAR" ]; then
    LAST_INJECTED_TS=$(cat "$SIDECAR" 2>/dev/null) || LAST_INJECTED_TS=""
fi

if [ -n "$LAST_INJECTED_TS" ] && [ "$LAST_INBOX_TS" = "$LAST_INJECTED_TS" ]; then
    dbg "already injected up to $LAST_INJECTED_TS — skipping"
    exit 0
fi
if [ -n "$LAST_INJECTED_TS" ] && [ "$LAST_INBOX_TS" \< "$LAST_INJECTED_TS" ]; then
    dbg "inbox_ts=$LAST_INBOX_TS < injected_ts=$LAST_INJECTED_TS — skipping"
    exit 0
fi

# ── Determine cutoff: inject messages newer than max(delivery_ts, injected_ts) ──
CUTOFF=""
if [ -n "$LAST_DELIVERY_TS" ] && [ -n "$LAST_INJECTED_TS" ]; then
    if [ "$LAST_DELIVERY_TS" \> "$LAST_INJECTED_TS" ]; then
        CUTOFF="$LAST_DELIVERY_TS"
    else
        CUTOFF="$LAST_INJECTED_TS"
    fi
elif [ -n "$LAST_DELIVERY_TS" ]; then
    CUTOFF="$LAST_DELIVERY_TS"
elif [ -n "$LAST_INJECTED_TS" ]; then
    CUTOFF="$LAST_INJECTED_TS"
fi

dbg "cutoff=${CUTOFF:-(none)} (max of delivery=$LAST_DELIVERY_TS, injected=$LAST_INJECTED_TS)"

# ── Build injection output ──
# neutralize: message text is model-authored; a literal
# "</injected-teammate-message>" or "<system-reminder>" inside it would
# break out of the wrapper and let a teammate forge system-level context.
# Only the handful of structural tag names are escaped — generic code like
# "a<b" or "List<int>" passes through untouched.
# Budget: keep the NEWEST messages within ~24KB of blocks; older overflow
# is omitted (it still arrives natively at the turn boundary) and counted.
INJECT_OUTPUT=$(printf '%s' "$MERGED" | jq -c --arg me "$AGENT_NAME" --arg meid "$SELF_ID" --arg meraw "$RAW_ID" --arg cutoff "${CUTOFF:-}" '
    def neutralize: tostring
        | gsub("<(?=/?(?:injected-teammate-message|teammate-message|system-reminder|system|human|assistant)(?:[>\\s/]|$))"; "&lt;");
    def fromlabel: (.from // "unknown") | tostring | gsub("[\"<>]"; "_");
    [.[] | select(.from != $me and .from != $meid and ($meraw == "" or .from != $meraw))
         | if $cutoff == "" then . else select((.timestamp // "") > $cutoff) end
    ] as $undelivered |
    if ($undelivered | length) == 0 then empty else
        ($undelivered | map(
            "<injected-teammate-message from=\"\(fromlabel)\" ts=\"\(.timestamp // "unknown")\">\n\((.text // "") | neutralize)\n</injected-teammate-message>"
        )) as $blocks |
        24000 as $max |
        (reduce ($blocks | reverse)[] as $b ({keep: [], len: 0, stop: false};
            if .stop then .
            else (($b | length) + 2) as $l |
                if (.len + $l) <= $max then {keep: (.keep + [$b]), len: (.len + $l), stop: false}
                else .stop = true end
            end
        ) | .keep | reverse) as $kept0 |
        (if ($kept0 | length) == 0 then $blocks[-1:] else $kept0 end) as $kept |
        (($blocks | length) - ($kept | length)) as $omitted |
        ($kept | length) as $count |
        ((if $omitted > 0 then "\($omitted) older message(s) omitted for size — they still arrive at your turn boundary. " else "" end)
         + "\($count) mid-turn teammate message(s) injected below. These are being delivered early so you have full context while working. When these same messages arrive again at your turn boundary, IGNORE the duplicates — you already have the content.\n\n"
         + ($kept | join("\n\n"))) as $context |
        {hookSpecificOutput: {
            hookEventName: "PreToolUse",
            additionalContext: $context
        }}
    end
' 2>/dev/null) || exit 0

# ── If nothing to inject (all filtered by cutoff), exit ──
[ -z "$INJECT_OUTPUT" ] && { dbg "no new messages after cutoff"; exit 0; }

# ── Update sidecar with newest injected timestamp, then release the lock ──
echo "$LAST_INBOX_TS" > "$SIDECAR" 2>/dev/null || true
dbg "injected: sidecar updated to $LAST_INBOX_TS"
flock -u 8 2>/dev/null || true

# ── Emit injection (non-blocking — no permissionDecision) ──
echo "$INJECT_OUTPUT"
exit 0
