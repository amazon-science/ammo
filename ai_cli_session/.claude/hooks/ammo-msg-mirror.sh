#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# ammo-msg-mirror.sh — PostToolUse hook (matcher: SendMessage), sender-side
# message mirror.
#
# Why: ammo-msg-check.sh injects undelivered teammate messages mid-turn, but
# on Claude Code 2.1.202 the file it reads is always empty. Every agent's
# REPL polls its own team inbox every ~1s and prunes delivered messages EVEN
# WHILE the agent is busy inside a tool call. Measured live (2026-08-13, real
# AMMO session container): 321ms persistence for team-lead, 392ms for a tmux
# champion mid-90s-sleep; the reader saw a non-empty inbox 0 times in 14
# fires. Queued messages surface only at the recipient's next turn boundary,
# so a champion inside a 90-minute benchmark call is deaf for 90 minutes.
#
# Fix: capture the message at the SENDER, where PostToolUse sees it plainly
# in tool_input, and append it to a side channel that only these hooks touch.
# The recipient's ammo-msg-check.sh merges this channel with the real inbox.
# This races nothing — the poller never sees the side channel.
#
# Channel: ${AMMO_MSG_MIRROR_DIR:-/tmp/ammo-msg-mirror-<uid>}/<team>/<to>.jsonl
# Keyed by team, not session: agent names (champion-1, ...) repeat across
# AMMO rounds and across sessions sharing one container /tmp; the team name
# is unique per lead session and per round.
#
# Fail-open: any error exits 0. A PostToolUse hook that errors must not
# disturb the tool result.
set -euo pipefail
trap 'exit 0' ERR

DEBUG="${AMMO_MSG_CHECK_DEBUG:-}"
DLOG="/tmp/ammo-msg-check-debug.log"
dbg() { [ -n "$DEBUG" ] && echo "[$(date +%T)] mirror: $*" >> "$DLOG" || true; }

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null) || exit 0
[ "$TOOL_NAME" = "SendMessage" ] || exit 0

TO=$(echo "$INPUT" | jq -r '.tool_input.to // ""' 2>/dev/null) || exit 0
[ -n "$TO" ] || exit 0

# Only mirror a message the real SendMessage accepted. A failed send (unknown
# recipient) must not be injected anywhere, or the recipient would see a
# message Claude Code never queued for it.
OK=$(echo "$INPUT" | jq -r '
    if (.tool_response | type) == "object" then
        (if .tool_response.success == false then "no" else "yes" end)
    elif (.tool_response | type) == "string" then
        (.tool_response | if test("\"success\"\\s*:\\s*false") then "no" else "yes" end)
    else "yes" end' 2>/dev/null) || OK="yes"
[ "$OK" = "yes" ] || exit 0

# Message body: SendMessage accepts a string OR a protocol-frame object. Drop
# the object form — those are shutdown/plan-approval/permission frames that
# Claude Code routes itself, and injecting them early would invite an agent
# to answer a frame out of band. JSON frames sent as strings (idle
# notifications etc.) are control-plane noise; mid-turn they are pure
# distraction.
BODY=$(echo "$INPUT" | jq -r 'if (.tool_input.message | type) == "string" then .tool_input.message else "" end' 2>/dev/null) || exit 0
[ -n "$BODY" ] || exit 0
case "$BODY" in
    '{'*'"type"'*) exit 0 ;;
esac

SUMMARY=$(echo "$INPUT" | jq -r '.tool_input.summary // ""' 2>/dev/null) || SUMMARY=""
# msg_id (when the tool response carries one) lets the reader dedupe the
# mirror copy against the real-inbox copy exactly.
MSG_ID=$(echo "$INPUT" | jq -r '
    if (.tool_response | type) == "object" then (.tool_response.msg_id // .tool_response.messageId // "") else "" end
' 2>/dev/null) || MSG_ID=""

HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HELPER_DIR/_ammo_msg_identity.sh" 2>/dev/null || exit 0
ammo_msg_resolve_identity "$INPUT"

FROM="$AGENT_NAME"
[ -n "$FROM" ] || FROM="unknown"
# Without a team there is no channel to route into. The recipient resolves
# the same team from its own side, so a mirror write under a guessed name
# would never be read.
[ -n "$TEAM_NAME" ] || { dbg "no team resolved — not mirroring ($FROM -> $TO)"; exit 0; }

# Every file we create holds model-authored text: keep it owner-only.
umask 077

ROOT=$(ammo_msg_ensure_root) || { dbg "channel root unusable, not mirroring"; exit 0; }
CH_DIR="$ROOT/$(ammo_msg_sanitize_team "$TEAM_NAME")"
install -d -m 700 "$CH_DIR" 2>/dev/null || mkdir -p "$CH_DIR" 2>/dev/null || exit 0
TO_SAFE=$(ammo_msg_sanitize "$TO")
[ -n "$TO_SAFE" ] || exit 0
OUT="$CH_DIR/$TO_SAFE.jsonl"
# A planted symlink here would redirect the append; a planted foreign file
# would poison the channel. Either way, do not write. -L is checked on its
# own because -e follows links and reports false for a DANGLING symlink.
if [ -L "$OUT" ] || { [ -e "$OUT" ] && ! ammo_msg_own_file "$OUT"; }; then
    dbg "refusing to write non-owned $OUT"
    exit 0
fi

# Bounded TTL sweep so finished teams do not accumulate forever on the
# container's shared /tmp. Anything older than the reader's injection window
# is dead weight on disk.
MAX_AGE_S="${AMMO_MSG_MIRROR_MAX_AGE_S:-7200}"
case "$MAX_AGE_S" in ''|*[!0-9]*) MAX_AGE_S=7200 ;; esac
SWEEP_MIN=$(( MAX_AGE_S / 60 ))
[ "$SWEEP_MIN" -lt 1 ] && SWEEP_MIN=1
find "$ROOT" -mindepth 1 -maxdepth 2 -mmin "+$SWEEP_MIN" -delete 2>/dev/null || true
install -d -m 700 "$CH_DIR" 2>/dev/null || true

TS=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)
LINE=$(jq -cn --arg from "$FROM" --arg to "$TO_SAFE" --arg text "${BODY:0:4000}" \
              --arg summary "${SUMMARY:0:200}" --arg ts "$TS" --arg mid "$MSG_ID" \
        '{from:$from,to:$to,text:$text,summary:$summary,timestamp:$ts}
         + (if $mid != "" then {msg_id:$mid} else {} end)' 2>/dev/null) || exit 0

# Keep the file bounded: a runaway chat must not fill /tmp. 512KB, trimmed to
# the most recent half when exceeded. Trim and append under ONE lock, via a
# per-process temp name: trimming outside the lock loses any message a
# concurrent sender commits during the read-rewrite window, and a shared trim
# temp lets two trimmers interleave into a file of NUL bytes.
MAXB=524288
{
    flock -w 5 9 2>/dev/null || true
    if [ -f "$OUT" ]; then
        sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
        if [ "$sz" -gt "$MAXB" ] 2>/dev/null; then
            TRIM="$OUT.trim.$$"
            if tail -c $((MAXB / 2)) "$OUT" 2>/dev/null | tail -n +2 > "$TRIM" 2>/dev/null; then
                mv "$TRIM" "$OUT" 2>/dev/null || rm -f "$TRIM" 2>/dev/null || true
            else
                rm -f "$TRIM" 2>/dev/null || true
            fi
        fi
    fi
    printf '%s\n' "$LINE" >> "$OUT" 2>/dev/null || true
} 9>"$CH_DIR/.lock" 2>/dev/null || printf '%s\n' "$LINE" >> "$OUT" 2>/dev/null || true

dbg "mirrored: $FROM -> $TO_SAFE team=$TEAM_NAME bytes=${#BODY}"
exit 0
