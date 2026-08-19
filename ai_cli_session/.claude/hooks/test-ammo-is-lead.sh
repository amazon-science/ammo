#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for _ammo_is_lead.sh (source-able bash helper).
#
# Covers the 5-layer lead detection contract:
#   L1: .agent_type non-empty  → NOT lead
#   L2: CLAUDE_SUBAGENT=1 env  → NOT lead
#   L3: transcript agentName in first 5 lines:
#         "team-lead"  → IS lead (post-compaction)
#         anything else → NOT lead
#   L4: teams/*/config.json leadSessionId == session_id → IS lead;
#         teams exist but no match → falls through to L5
#   L5: no positive subagent signal → IS lead (solo orchestrator / stale config)
#   L6: any error → IS lead (fail-open)
#
# Also verifies:
#   - function writes nothing to stdout
#   - function does not exit the caller
#   - sourcing twice is idempotent
#
# Run: bash .claude/hooks/test-ammo-is-lead.sh
set -euo pipefail

# AMMO_TEST_HOOK lets a reviewer aim this harness at an ARCHIVED copy of the
# helper to see which cases the archived version fails (red->green evidence for
# the stale-team-config demotion cases 09/09b).
HELPER="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/_ammo_is_lead.sh}"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# Runner script — reads $AMMO_TEST_INPUT env var, sources helper, invokes it.
# Writes "<rc> <stdout_bytes>" so parent can assert both. Sanity-checks that
# the function did not exit the caller (the trailing `echo` only runs if
# the helper returned rather than exit'd).
RUNNER="$TMPDIR/runner.sh"
cat > "$RUNNER" <<'RUNNER_EOF'
#!/bin/bash
set +e  # helper errors must not propagate
source "$HELPER_PATH"
# Capture stdout — if the helper writes anything there, test fails.
tmp=$(mktemp)
_ammo_is_lead "$AMMO_TEST_INPUT" >"$tmp" 2>/dev/null
rc=$?
bytes=$(wc -c <"$tmp" | tr -d ' ')
rm -f "$tmp"
echo "$rc $bytes DID_NOT_EXIT"
RUNNER_EOF
chmod +x "$RUNNER"

run_case() {
    local name="$1" expected="$2" input="$3"
    shift 3
    local extra_env=("$@")
    TOTAL=$((TOTAL + 1))

    local out
    out=$(env -i HOME="$TMPDIR" PATH="$PATH" HELPER_PATH="$HELPER" \
        AMMO_TEST_INPUT="$input" "${extra_env[@]}" bash "$RUNNER" 2>&1) || true
    local got="${out%% *}"
    local rest="${out#* }"
    local bytes="${rest%% *}"
    local marker="${out##* }"

    local pass=true
    if [ "$got" != "$expected" ]; then pass=false; fi
    if [ "$bytes" != "0" ]; then pass=false; fi
    if [ "$marker" != "DID_NOT_EXIT" ]; then pass=false; fi

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $name (rc=$got, stdout_bytes=$bytes)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name"
        echo "        expected rc=$expected, stdout_bytes=0, DID_NOT_EXIT"
        echo "        got:          $out"
        FAIL=$((FAIL + 1))
    fi
}

# Place a team-config fixture under $HOME/.claude/teams/<team>/config.json.
# $3.. = optional member sessionIds (L4b fixtures).
write_team_config() {
    local team_name="$1" lead_sid="$2"
    shift 2
    local dir="$TMPDIR/.claude/teams/$team_name"
    local members="" sid
    for sid in "$@"; do
        [ -n "$members" ] && members="$members,"
        members="$members{\"name\":\"m-$sid\",\"sessionId\":\"$sid\"}"
    done
    mkdir -p "$dir"
    cat > "$dir/config.json" <<EOF
{"name":"$team_name","leadSessionId":"$lead_sid","members":[$members]}
EOF
}

ORCH_T="$TMPDIR/orch_transcript.jsonl"
echo '{"type":"user","message":{"role":"user","content":"hi"}}' > "$ORCH_T"

CHAMP_T="$TMPDIR/champ_transcript.jsonl"
echo '{"type":"permission-mode"}' > "$CHAMP_T"
echo '{"agentName":"champion-1","type":"user"}' >> "$CHAMP_T"

TL_T="$TMPDIR/teamlead_transcript.jsonl"
echo '{"type":"permission-mode"}' > "$TL_T"
echo '{"agentName":"team-lead","type":"user"}' >> "$TL_T"

MON_T="$TMPDIR/monitor_transcript.jsonl"
echo '{"type":"permission-mode"}' > "$MON_T"
echo '{"agentName":"monitor-1","type":"user"}' >> "$MON_T"

# agentName on line 6 (outside head -5 window)
L6_T="$TMPDIR/line6_transcript.jsonl"
for i in 1 2 3 4 5; do echo '{"type":"system","subtype":"compact"}' >> "$L6_T"; done
echo '{"agentName":"champion-1","type":"user"}' >> "$L6_T"

MISSING_T="$TMPDIR/nonexistent_transcript.jsonl"

echo "== L1: agent_type non-empty → NOT lead =="
run_case "01 agent_type=ammo-champion → NOT lead" 1 \
    '{"agent_type":"ammo-champion","session_id":"s1"}'

run_case "02 agent_type=general-purpose → NOT lead" 1 \
    '{"agent_type":"general-purpose","session_id":"s1"}'

echo ""; echo "== L2: CLAUDE_SUBAGENT env → NOT lead =="
run_case "03 CLAUDE_SUBAGENT=1 → NOT lead" 1 \
    '{"session_id":"s1"}' \
    CLAUDE_SUBAGENT=1

echo ""; echo "== L3: transcript agentName semantics =="
run_case "04 agentName=team-lead → IS lead (post-compaction)" 0 \
    "{\"transcript_path\":\"$TL_T\",\"session_id\":\"s1\"}"

run_case "05 agentName=champion-1 → NOT lead" 1 \
    "{\"transcript_path\":\"$CHAMP_T\",\"session_id\":\"s1\"}"

run_case "06 agentName=monitor-1 → NOT lead" 1 \
    "{\"transcript_path\":\"$MON_T\",\"session_id\":\"s1\"}"

run_case "07 agentName on line 6 (outside window), no teams → IS lead" 0 \
    "{\"transcript_path\":\"$L6_T\",\"session_id\":\"s1\"}"

echo ""; echo "== L4: team-config leadSessionId scan =="
# Setup a team config whose leadSessionId matches s1
write_team_config "round1-team" "s1"
run_case "08 teams/*/leadSessionId matches session_id → IS lead" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"s1\"}"

# Bypass regression: a team config whose leadSessionId names a DEAD session
# used to demote the real lead. Team configs are S3-synced and outlive the
# session that wrote them, and the pinned CC lead transcript carries no
# agentName, so L3 declines and L4 was the deciding rung. A config miss
# identifies nobody, so it now falls through to L5 (IS lead). Subagents are
# still caught by L1 (.agent_type) and L2 (CLAUDE_SUBAGENT=1) — see cases 15
# and 09c below. This case is exit 1 (NOT lead) on the pre-fix helper.
run_case "09 team-config exists but no leadSessionId match → IS lead (stale config)" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"nomatch\"}"

run_case "09b stale config + no transcript at all → IS lead" 0 \
    "{\"session_id\":\"nomatch\"}"

run_case "09c stale config + agent_type set → still NOT lead (L1 wins)" 1 \
    "{\"agent_type\":\"ammo-delegate\",\"session_id\":\"nomatch\"}"

run_case "09d stale config + CLAUDE_SUBAGENT=1 → still NOT lead (L2 wins)" 1 \
    "{\"session_id\":\"nomatch\"}" CLAUDE_SUBAGENT=1

run_case "09e stale config + champion transcript → still NOT lead (L3 wins)" 1 \
    "{\"transcript_path\":\"$CHAMP_T\",\"session_id\":\"nomatch\"}"

echo ""; echo "== L4b: team-config members[].sessionId scan =="
# A tmux teammate carries no .agent_type, no CLAUDE_SUBAGENT and no agentName,
# so members[] is the only rung that can demote it. Without L4b it reaches L5
# and is misread as the lead.
rm -rf "$TMPDIR/.claude/teams"
write_team_config "round2-team" "lead-sid" "mate-1" "mate-2"

run_case "17 members[].sessionId match, no other signal → NOT lead" 1 \
    "{\"session_id\":\"mate-1\"}"

run_case "17b members[] match + orchestrator transcript → NOT lead" 1 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"mate-2\"}"

run_case "17c lead of the same config → IS lead (L4 beats L4b)" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"lead-sid\"}"

run_case "17d session in neither leadSessionId nor members[] → IS lead" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"nomatch\"}"

# Wipe teams dir between cases
rm -rf "$TMPDIR/.claude/teams"

echo ""; echo "== L5: no teams → IS lead (solo orchestrator) =="
run_case "10 no teams dir, orchestrator transcript → IS lead" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"s1\"}"

mkdir -p "$TMPDIR/.claude/teams"  # empty teams dir
run_case "11 empty teams dir → IS lead" 0 \
    "{\"transcript_path\":\"$ORCH_T\",\"session_id\":\"s1\"}"
rmdir "$TMPDIR/.claude/teams"

echo ""; echo "== L6: fail-open on errors =="
run_case "12 malformed JSON input → IS lead (fail-open)" 0 \
    "NOT_JSON_AT_ALL"

run_case "13 missing transcript file → IS lead (fail-open, no teams)" 0 \
    "{\"transcript_path\":\"$MISSING_T\",\"session_id\":\"s1\"}"

run_case "14 no session_id, no transcript, no teams → IS lead" 0 \
    "{}"

echo ""; echo "== Precedence: agent_type beats team-config match =="
write_team_config "round2-team" "s2"
run_case "15 agent_type wins over matching leadSessionId → NOT lead" 1 \
    "{\"agent_type\":\"ammo-delegate\",\"session_id\":\"s2\",\"transcript_path\":\"$ORCH_T\"}"

echo ""; echo "== Precedence: transcript agentName beats team-config =="
# Even though leadSessionId may not match session_id in team config, the
# transcript says we're a champion → NOT lead. Champions must never be
# treated as lead because team config is a weaker signal than the actual
# transcript agentName (L3 runs before L4 in the precedence order).
run_case "16 transcript agentName=champion beats leadSessionId match → NOT lead" 1 \
    "{\"transcript_path\":\"$CHAMP_T\",\"session_id\":\"s2\"}"

echo ""; echo "== Stress: scan 100 team configs, still no stdout leak =="
STRESS_HOME=$(mktemp -d)
for i in $(seq 1 100); do
    mkdir -p "$STRESS_HOME/.claude/teams/stress-team-$i"
    echo "{\"name\":\"stress-team-$i\",\"leadSessionId\":\"sid-$i\",\"members\":[]}" \
        > "$STRESS_HOME/.claude/teams/stress-team-$i/config.json"
done
TOTAL=$((TOTAL + 1))
stdout_dump=$(env -i HOME="$STRESS_HOME" PATH="$PATH" HELPER_PATH="$HELPER" bash -c '
    source "$HELPER_PATH"
    _ammo_is_lead "{\"session_id\":\"sid-50\"}"
    echo "rc=$?" >&2
' 2>/tmp/is-lead-stress-stderr.log)
stdout_bytes=$(echo -n "$stdout_dump" | wc -c)
rc_line=$(grep "^rc=" /tmp/is-lead-stress-stderr.log | tail -1)
if [ "$stdout_bytes" = "0" ] && [ "$rc_line" = "rc=0" ]; then
    echo "  PASS [$TOTAL]: 100-team scan, sid-50 match → IS lead, stdout=0"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: 100-team stress (stdout=$stdout_bytes bytes, rc_line=$rc_line)"
    echo "        stdout: [$stdout_dump]"
    FAIL=$((FAIL + 1))
fi
rm -rf "$STRESS_HOME" /tmp/is-lead-stress-stderr.log

echo ""; echo "== Stderr audit on happy paths =="
# On inputs where no error is expected, helper must also stay silent on
# stderr (audit-grade). Ensures no jq noise leaks through. Uses a separate
# HOME so it doesn't race with the stress test's teams-dir writes.
TOTAL=$((TOTAL + 1))
STDERR_HOME=$(mktemp -d)
STDERR_LOG=$(mktemp)
env -i HOME="$STDERR_HOME" PATH="$PATH" HELPER_PATH="$HELPER" bash -c '
    source "$HELPER_PATH"
    for i in 1 2 3 4 5; do
        _ammo_is_lead "{}" >/dev/null 2>&1 || true
    done
' 1>/dev/null 2>"$STDERR_LOG"
stderr_bytes=$(wc -c <"$STDERR_LOG" | tr -d ' ')
if [ "$stderr_bytes" = "0" ]; then
    echo "  PASS [$TOTAL]: 5 invocations with clean input → stderr is silent"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: stderr leaked $stderr_bytes bytes"
    echo "        stderr: [$(cat "$STDERR_LOG")]"
    FAIL=$((FAIL + 1))
fi
rm -rf "$STDERR_HOME" "$STDERR_LOG"

echo ""; echo "== Idempotency: sourcing helper twice works =="
TOTAL=$((TOTAL + 1))
# Verify (a) file exists, (b) function defined after each source, (c) both
# sources return 0. This is strict enough to RED when the helper is missing.
out=$(env -i HOME="$TMPDIR" PATH="$PATH" HELPER_PATH="$HELPER" bash -c '
    [ -f "$HELPER_PATH" ] || { echo "missing-file"; exit 0; }
    source "$HELPER_PATH" || { echo "source1-failed"; exit 0; }
    type _ammo_is_lead >/dev/null 2>&1 || { echo "not-defined-after-first"; exit 0; }
    source "$HELPER_PATH" || { echo "source2-failed"; exit 0; }
    type _ammo_is_lead >/dev/null 2>&1 || { echo "not-defined-after-second"; exit 0; }
    if _ammo_is_lead "{}" >/dev/null 2>&1; then echo "ok:lead"; else echo "ok:notlead"; fi
' 2>/dev/null || echo "FAIL")
case "$out" in
    ok:lead|ok:notlead)
        echo "  PASS [$TOTAL]: sourcing helper twice is idempotent ($out)"
        PASS=$((PASS + 1))
        ;;
    *)
        echo "  FAIL [$TOTAL]: double-source broke (got: $out)"
        FAIL=$((FAIL + 1))
        ;;
esac

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
