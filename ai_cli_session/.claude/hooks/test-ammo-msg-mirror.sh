#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-msg-mirror.sh (PostToolUse hook — sender-side mirror)
# Run: bash .claude/hooks/test-ammo-msg-mirror.sh
#
# The mirror captures successful SendMessage calls into
# $AMMO_MSG_MIRROR_DIR/<team>/<recipient>.jsonl so ammo-msg-check.sh can
# inject them mid-turn. On CC 2.1.202 the real inbox is drained ~1s after
# every send (even while the recipient is busy), so this channel is the only
# mid-turn source that works.

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-msg-mirror.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
export AMMO_MSG_MIRROR_DIR="$TMPDIR/mirror-root"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

ok()  { TOTAL=$((TOTAL + 1)); PASS=$((PASS + 1)); echo "  PASS [$TOTAL]: $1"; }
bad() { TOTAL=$((TOTAL + 1)); FAIL=$((FAIL + 1)); echo "  FAIL [$TOTAL]: $1"; shift; for l in "$@"; do echo "        $l"; done; }

make_team_config() {
    local team_dir="$1" team_name="$2" lead_sid="${3:-lead-session}"
    mkdir -p "$team_dir/inboxes"
    echo "{\"name\":\"$team_name\",\"leadSessionId\":\"$lead_sid\",\"members\":[{\"name\":\"champion-1\",\"agentType\":\"ammo-champion\",\"agentId\":\"champion-1@$team_name\"}]}" > "$team_dir/config.json"
}

make_champion_transcript() {
    local file="$1" team="$2" agent="$3"
    echo '{"type":"permission-mode","sessionId":"test-session"}' > "$file"
    echo "{\"teamName\":\"$team\",\"agentName\":\"$agent\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"},\"sessionId\":\"test-session\"}" >> "$file"
}

# run_mirror <to> <message-json-fragment or string> <success> [extra jq args...]
# Builds a champion-1 sender input and pipes it through the hook.
run_mirror() {
    local to="$1" msg="$2" success="$3" transcript="${4:-$TRANSCRIPT}" sid="${5:-test-session}"
    jq -cn --arg to "$to" --arg msg "$msg" --argjson ok "$success" \
        --arg t "$transcript" --arg s "$sid" \
        '{tool_name:"SendMessage",session_id:$s,transcript_path:$t,
          tool_input:{to:$to,message:$msg,summary:"test summary"},
          tool_response:{success:$ok,msg_id:"mid-123"}}' \
        | env HOME="$TMPDIR" bash "$HOOK" >/dev/null 2>&1 || true
}

reset() { rm -rf "$AMMO_MSG_MIRROR_DIR"; }

TEAM_DIR="$TMPDIR/.claude/teams/test-team"
make_team_config "$TEAM_DIR" "test-team"
TRANSCRIPT="$TMPDIR/transcript.jsonl"
make_champion_transcript "$TRANSCRIPT" "test-team" "champion-1"
CH="$AMMO_MSG_MIRROR_DIR/test-team"

echo "== Basic mirroring =="

reset; run_mirror "impl-champion-op001" "abort the sweep, baseline is wrong" true
if [ -s "$CH/impl-champion-op001.jsonl" ] \
   && jq -e 'select(.from == "champion-1" and .to == "impl-champion-op001" and .text == "abort the sweep, baseline is wrong" and .msg_id == "mid-123" and (.timestamp | length) > 0)' \
        "$CH/impl-champion-op001.jsonl" >/dev/null 2>&1; then
    ok "successful send mirrored with from/to/text/msg_id/timestamp"
else
    bad "successful send mirrored" "file: $(cat "$CH/impl-champion-op001.jsonl" 2>/dev/null || echo '(missing)')"
fi

reset; run_mirror "ghost" "unreachable recipient" false
if [ ! -e "$CH/ghost.jsonl" ]; then ok "failed send (success:false) not mirrored"; else bad "failed send was mirrored"; fi

reset
jq -cn --arg t "$TRANSCRIPT" \
    '{tool_name:"SendMessage",session_id:"test-session",transcript_path:$t,
      tool_input:{to:"champion-1",message:{type:"shutdown_request",requestId:"r1"}},
      tool_response:{success:true}}' \
    | env HOME="$TMPDIR" bash "$HOOK" >/dev/null 2>&1 || true
if [ ! -e "$CH/champion-1.jsonl" ]; then ok "protocol-frame object message not mirrored"; else bad "protocol frame object was mirrored"; fi

reset; run_mirror "champion-1" '{"type":"idle_notification","from":"x"}' true
if [ ! -e "$CH/champion-1.jsonl" ]; then ok "protocol-frame string message not mirrored"; else bad "protocol frame string was mirrored"; fi

echo ""
echo "== Path safety =="

reset; run_mirror '../../../tmp/ammo-mirror-pwned' "escape attempt" true
if [ ! -e /tmp/ammo-mirror-pwned.jsonl ] && ! find "$AMMO_MSG_MIRROR_DIR/.." -maxdepth 1 -name 'ammo-mirror-pwned*' 2>/dev/null | grep -q .; then
    ok "path traversal in recipient contained"
else
    bad "path traversal escaped the channel root"
fi
rm -f /tmp/ammo-mirror-pwned.jsonl 2>/dev/null || true

reset
mkdir -p "$CH"
ln -s "$TMPDIR/steal-target" "$CH/champion-1.jsonl"
run_mirror "champion-1" "should not follow symlink" true
if [ ! -e "$TMPDIR/steal-target" ]; then ok "symlinked channel file refused"; else bad "append followed a planted symlink"; fi

echo ""
echo "== Content bounds =="

reset; run_mirror "worker" "$(head -c 9000 /dev/zero | tr '\0' 'A')" true
LEN=$(jq -r '.text | length' "$CH/worker.jsonl" 2>/dev/null) || LEN=0
if [ "$LEN" = "4000" ]; then ok "oversized body capped at 4000 chars"; else bad "body cap wrong: $LEN"; fi

reset; ( export AMMO_MSG_MIRROR_MAX_AGE_S="2h"; run_mirror "worker" "still delivered" true )
if [ -s "$CH/worker.jsonl" ]; then ok "non-numeric max-age does not drop the message"; else bad "bad max-age aborted the mirror"; fi

reset
mkdir -p "$CH"
# Seed a >512KB channel; the trim must run under the lock and keep the tail.
# (yes|head dies with SIGPIPE under pipefail — guard the pipeline status)
{ yes '{"from":"old","to":"worker","text":"padding line","timestamp":"2026-04-03T17:00:00.000Z"}' | head -c 600000 > "$CH/worker.jsonl"; } 2>/dev/null || true
run_mirror "worker" "TRIM_SURVIVOR" true
SZ=$(stat -c%s "$CH/worker.jsonl" 2>/dev/null || echo 0)
if [ "$SZ" -lt 524288 ] && tail -1 "$CH/worker.jsonl" | grep -q TRIM_SURVIVOR; then
    ok "oversize channel trimmed, new message appended last"
else
    bad "trim failed: size=$SZ, tail=$(tail -c 120 "$CH/worker.jsonl" 2>/dev/null)"
fi

reset
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    ( run_mirror "worker" "PARALLEL_$i" true ) &
done
wait
N=$(grep -c PARALLEL_ "$CH/worker.jsonl" 2>/dev/null) || N=0
if [ "$N" = "12" ]; then ok "12 parallel sends → 12 lines (no lost appends)"; else bad "parallel appends lost messages: $N/12"; fi

echo ""
echo "== Sender identity =="

# Lead sender: no agentName in transcript; session_id matches a team config's
# leadSessionId → from must resolve to team-lead.
reset
LEAD_TEAM_DIR="$TMPDIR/.claude/teams/lead-team"
make_team_config "$LEAD_TEAM_DIR" "lead-team" "lead-sid-777"
LEAD_T="$TMPDIR/lead-transcript.jsonl"
echo '{"type":"user","message":{"role":"user","content":"go"},"sessionId":"lead-sid-777"}' > "$LEAD_T"
run_mirror "champion-1" "dispatch from the top" true "$LEAD_T" "lead-sid-777"
FROMV=$(jq -r '.from' "$AMMO_MSG_MIRROR_DIR/lead-team/champion-1.jsonl" 2>/dev/null) || FROMV=""
if [ "$FROMV" = "team-lead" ]; then ok "lead sender resolved via leadSessionId → from=team-lead"; else bad "lead sender resolution wrong: from=$FROMV"; fi

# Unresolvable team → nothing written anywhere.
reset
NOTEAM_T="$TMPDIR/noteam.jsonl"
echo '{"type":"user","message":{"role":"user","content":"hi"},"sessionId":"orphan-sid"}' > "$NOTEAM_T"
NOTEAM_HOME=$(mktemp -d)
jq -cn --arg t "$NOTEAM_T" \
    '{tool_name:"SendMessage",session_id:"orphan-sid",transcript_path:$t,
      tool_input:{to:"champion-1",message:"lost"},tool_response:{success:true}}' \
    | env HOME="$NOTEAM_HOME" bash "$HOOK" >/dev/null 2>&1 || true
if [ ! -d "$AMMO_MSG_MIRROR_DIR" ] || ! find "$AMMO_MSG_MIRROR_DIR" -name '*.jsonl' 2>/dev/null | grep -q .; then
    ok "unresolvable team → not mirrored"
else
    bad "message mirrored without a resolvable team"
fi
rm -rf "$NOTEAM_HOME"

# Dotted team name lands in the dot→dash channel dir on BOTH sides.
reset
DOT_TEAM_DIR="$TMPDIR/.claude/teams/ammo-round-1-qwen3-5-4b"
mkdir -p "$DOT_TEAM_DIR/inboxes"
echo '{"name":"ammo-round-1-qwen3.5-4b","leadSessionId":"lead-session","members":[]}' > "$DOT_TEAM_DIR/config.json"
DOT_T="$TMPDIR/dot-transcript.jsonl"
make_champion_transcript "$DOT_T" "ammo-round-1-qwen3.5-4b" "champion-1"
run_mirror "impl-champion-op001" "dot team routing" true "$DOT_T"
if [ -s "$AMMO_MSG_MIRROR_DIR/ammo-round-1-qwen3-5-4b/impl-champion-op001.jsonl" ]; then
    ok "dotted team name → dot→dash channel dir"
else
    bad "dotted team channel dir wrong" "$(find "$AMMO_MSG_MIRROR_DIR" -type f 2>/dev/null)"
fi

echo ""
echo "== Fail-open =="

for bad_input in '{}' 'not json' '{"tool_name":"Bash"}' '{"tool_name":"SendMessage"}'; do
    rc=0
    printf '%s' "$bad_input" | env HOME="$TMPDIR" bash "$HOOK" >/dev/null 2>&1 || rc=$?
    if [ "$rc" = "0" ]; then ok "fail-open on input: $bad_input"; else bad "non-zero exit ($rc) on: $bad_input"; fi
done

echo ""
echo "== settings.local.json wiring =="
SETTINGS="$(cd "$(dirname "$0")/../" && pwd)/settings.local.json"
WIRED=$(jq -r '
    .hooks.PostToolUse
    | map(select(.hooks[]?.command // "" | contains("ammo-msg-mirror.sh")))
    | .[0].matcher // ""
' "$SETTINGS" 2>/dev/null) || WIRED=""
if [ "$WIRED" = "SendMessage" ]; then
    ok "settings.local.json mirror matcher = SendMessage"
else
    bad "mirror hook not wired (matcher=$WIRED)"
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
