#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-monitor-reminder.sh (PostToolUse hook)
# Run: bash .claude/hooks/test-ammo-monitor-reminder.sh
#
# Tests all enforcement paths:
#   1. Agent type filtering (only ammo-champion / ammo-impl-champion fire)
#   2. Session identity gate (orchestrator vs champion/monitor sessions)
#   3. Name handling (named, unnamed, special chars)
#   4. JSON output validity
#   5. Fail-open behavior
#   6. Edge cases (transcript window, empty files, etc.)
#
set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-monitor-reminder.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" "$TMPDIR/hook-stderr" "$TMPDIR/hook-stdout" 2>/dev/null || true; }
trap cleanup EXIT

# Isolation: pin AMMO_GPU_RES_DIR to a throwaway dir for every hook invocation
# so no hook write can reach the real reservation dir of a live session.
DEFAULT_GPU_RES_DIR="$TMPDIR/gpures-default"
mkdir -p "$DEFAULT_GPU_RES_DIR"

# ─────────────────────────────────────────────
# make_orchestrator_transcript: no agentName anywhere in first 5 lines
# ─────────────────────────────────────────────
make_orchestrator_transcript() {
    local file="$1"
    echo '{"type":"user","message":{"role":"user","content":"hello"},"sessionId":"orch-session"}' > "$file"
}

# ─────────────────────────────────────────────
# make_champion_transcript: agentName present on line 2 (permission-mode prefix on line 1)
# ─────────────────────────────────────────────
make_champion_transcript() {
    local file="$1" agent_name="${2:-champion-1}"
    echo '{"type":"permission-mode","sessionId":"test-session"}' > "$file"
    echo "{\"agentName\":\"$agent_name\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"},\"sessionId\":\"test-session\"}" >> "$file"
}

# ─────────────────────────────────────────────
# run_test: core test runner
#   $1 = test name
#   $2 = expected exit code
#   $3 = JSON input
#   $4 = (optional) string that must appear in stdout
#   $5 = (optional) string that must NOT appear in stdout
#   $6 = (optional) "KNOWN_BUG" — marks test as documenting a known bug; failure is shown but not counted
# ─────────────────────────────────────────────
run_test() {
    local test_name="$1" expected_exit="$2" json_input="$3"
    local check_output="${4:-}" forbid_output="${5:-}" known_bug="${6:-}"
    local actual_exit=0
    TOTAL=$((TOTAL + 1))

    echo "$json_input" | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$DEFAULT_GPU_RES_DIR" bash "$HOOK" > "$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?

    local pass=true

    # Exit code check
    [ "$actual_exit" -ne "$expected_exit" ] && pass=false

    # Required output check
    if [ -n "$check_output" ]; then
        if ! grep -qF "$check_output" "$TMPDIR/hook-stdout" 2>/dev/null; then
            pass=false
        fi
    fi

    # Forbidden output check
    if [ -n "$forbid_output" ]; then
        if grep -qF "$forbid_output" "$TMPDIR/hook-stdout" 2>/dev/null; then
            pass=false
        fi
    fi

    # When output is expected, validate JSON
    if [ -n "$check_output" ] && [ -s "$TMPDIR/hook-stdout" ]; then
        if ! jq . "$TMPDIR/hook-stdout" >/dev/null 2>&1; then
            echo "  WARN: stdout is not valid JSON!"
            pass=false
        fi
    fi

    local bug_label=""
    [ -n "$known_bug" ] && bug_label=" [KNOWN BUG]"

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)$bug_label"
        PASS=$((PASS + 1))
    else
        if [ -n "$known_bug" ]; then
            # Known bug: print but don't count as failure
            echo "  XFAIL[$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit) — KNOWN BUG: $known_bug"
        else
            echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
            [ -n "$check_output" ]   && echo "        required output : $check_output"
            [ -n "$forbid_output" ]  && echo "        forbidden output: $forbid_output"
            echo "        stdout: $(head -3 "$TMPDIR/hook-stdout" 2>/dev/null || echo '(none)')"
            echo "        stderr: $(head -3 "$TMPDIR/hook-stderr" 2>/dev/null || echo '(none)')"
            FAIL=$((FAIL + 1))
        fi
    fi
}

# ─────────────────────────────────────────────
# check_json_field: verify a jq expression evaluates to expected value on stdout
# ─────────────────────────────────────────────
check_json_field() {
    local test_name="$1" json_input="$2" jq_expr="$3" expected_value="$4"
    TOTAL=$((TOTAL + 1))

    local actual_value
    actual_value=$(echo "$json_input" | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$DEFAULT_GPU_RES_DIR" bash "$HOOK" 2>/dev/null | jq -r "$jq_expr" 2>/dev/null || echo "JQ_FAILED")

    if [ "$actual_value" = "$expected_value" ]; then
        echo "  PASS [$TOTAL]: $test_name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name"
        echo "        jq: $jq_expr"
        echo "        expected: $expected_value"
        echo "        got     : $actual_value"
        FAIL=$((FAIL + 1))
    fi
}

ORCH_T="$TMPDIR/orch_transcript.jsonl"
CHAMP_T="$TMPDIR/champion_transcript.jsonl"

make_orchestrator_transcript "$ORCH_T"
make_champion_transcript "$CHAMP_T" "champion-1"

# ════════════════════════════════════════════
echo "== Agent type filtering =="
# ════════════════════════════════════════════

run_test "ammo-champion → no output (debate monitors removed)" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

run_test "ammo-impl-champion → injects reminder" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    "additionalContext"

run_test "ammo-delegate → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-delegate\",\"name\":\"delegate-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

run_test "ammo-transcript-monitor → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-transcript-monitor\",\"name\":\"monitor-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

run_test "general-purpose → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"general-purpose\",\"name\":\"gp-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

run_test "empty subagent_type → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"\",\"name\":\"some-agent\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== Session identity gate =="
# ════════════════════════════════════════════

run_test "Orchestrator session (no agentName in transcript), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

run_test "Champion session (agentName=champion-1), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$CHAMP_T\"}" \
    "" "additionalContext"

MONITOR_T="$TMPDIR/monitor_transcript.jsonl"
make_champion_transcript "$MONITOR_T" "monitor-1"
run_test "Monitor session, ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-3\"},\"transcript_path\":\"$MONITOR_T\"}" \
    "" "additionalContext"

# T3a: Post-compaction — the lead orchestrator's transcript gains an
# agentName=team-lead entry after context compaction. The old inline
# "if any agentName present → suppress" logic silenced the nudge in this
# case, which is a regression. The helper must treat agentName=team-lead
# as the lead, so the hook still fires.
TL_T="$TMPDIR/teamlead_transcript.jsonl"
echo '{"type":"permission-mode"}' > "$TL_T"
echo '{"agentName":"team-lead","type":"user"}' >> "$TL_T"
run_test "Lead transcript (agentName=team-lead), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$TL_T\"}" \
    "" "additionalContext"

run_test "No transcript_path, ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"}}" \
    "" "additionalContext"

run_test "Missing transcript file, ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"/nonexistent/path/transcript.jsonl\"}" \
    "" "additionalContext"

EMPTY_T="$TMPDIR/empty_transcript.jsonl"
: > "$EMPTY_T"
run_test "Empty transcript file, ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$EMPTY_T\"}" \
    "" "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== Name handling =="
# ════════════════════════════════════════════

check_json_field "impl-champion-op003 → reminder mentions monitor-impl-champion-op003 (name handling)" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("monitor-impl-champion-op003")' \
    "true"

check_json_field "impl-champion-op003 → reminder mentions monitor-impl-champion-op003 (impl type)" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("monitor-impl-champion-op003")' \
    "true"

# Named agent: correct agent name in reminder (not transcript path)
check_json_field "Named impl-champion → agentType=ammo-transcript-monitor in reminder" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("ammo-transcript-monitor")' \
    "true"

# Unnamed impl-champion (empty name) — fires with "unnamed" warning
run_test "Impl-champion with empty name → fires with unnamed warning" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"\"},\"transcript_path\":\"$ORCH_T\"}" \
    "unnamed"

run_test "Impl-champion with missing name key → fires with unnamed warning" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\"},\"transcript_path\":\"$ORCH_T\"}" \
    "unnamed"

# Unnamed impl-champion in champion session — session gate correctly suppresses
CHAMP_GATE_T="$TMPDIR/champ_for_gate.jsonl"
make_champion_transcript "$CHAMP_GATE_T" "existing-champion"
run_test "Unnamed impl-champion in champion session → suppressed by session gate" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\"},\"transcript_path\":\"$CHAMP_GATE_T\"}" \
    "" "additionalContext"

# Unnamed impl-champion with no transcript_path — fires with unnamed warning (fail-open)
run_test "Unnamed impl-champion, no transcript → fires with unnamed warning" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"\"}}" \
    "unnamed"

# Special characters in name — jq --arg handles them safely (use impl-champion)
run_test "Name with double-quotes → valid JSON output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"name-with-\\\"quotes\\\"\"},\"transcript_path\":\"$ORCH_T\"}" \
    "monitor-name-with"

run_test "Name with backslash → valid JSON output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"op\\\\003\"},\"transcript_path\":\"$ORCH_T\"}" \
    "additionalContext"

run_test "Name with HTML special chars → valid JSON output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"agent<tag>&amp;\"},\"transcript_path\":\"$ORCH_T\"}" \
    "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== JSON output validity =="
# ════════════════════════════════════════════

# Validate structure: top-level key must be hookSpecificOutput (use impl-champion since debate champions no longer fire)
check_json_field "Output has hookSpecificOutput key" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    'has("hookSpecificOutput")' \
    "true"

check_json_field "hookSpecificOutput has hookEventName=PostToolUse" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.hookEventName' \
    "PostToolUse"

check_json_field "hookSpecificOutput has additionalContext field" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput | has("additionalContext")' \
    "true"

check_json_field "additionalContext starts with AMMO MONITOR REMINDER" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | startswith("AMMO MONITOR REMINDER")' \
    "true"

check_json_field "impl-champion output has PostToolUse hookEventName" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.hookEventName' \
    "PostToolUse"

# ════════════════════════════════════════════
echo ""; echo "== Fail-open behavior =="
# ════════════════════════════════════════════

run_test "Empty JSON input → exit 0 (no crash)" 0 '{}' "" "additionalContext"

run_test "Malformed JSON input → exit 0 (no crash)" 0 'NOT JSON AT ALL' "" "additionalContext"

run_test "Missing tool_input field → exit 0" 0 \
    '{"transcript_path":"/nonexistent.jsonl"}' "" "additionalContext"

run_test "null tool_input → exit 0" 0 \
    '{"tool_input":null}' "" "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== Transcript window edge cases =="
# ════════════════════════════════════════════

# agentName on line 1 (no permission-mode prefix - direct subagent)
AGENT_L1_T="$TMPDIR/agent_line1.jsonl"
echo '{"agentName":"champion-1","type":"user","sessionId":"s1"}' > "$AGENT_L1_T"
run_test "agentName on line 1 (no permission-mode) → suppressed" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$AGENT_L1_T\"}" \
    "" "additionalContext"

# agentName on line 2 (permission-mode on line 1 - standard subagent format)
AGENT_L2_T="$TMPDIR/agent_line2.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$AGENT_L2_T"
echo '{"agentName":"champion-1","type":"user","sessionId":"s1"}' >> "$AGENT_L2_T"
run_test "agentName on line 2 (with permission-mode prefix) → suppressed" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$AGENT_L2_T\"}" \
    "" "additionalContext"

# agentName on line 5 (boundary - still within head -5 window)
AGENT_L5_T="$TMPDIR/agent_line5.jsonl"
for i in 1 2 3 4; do echo '{"type":"system","subtype":"compact"}' >> "$AGENT_L5_T"; done
echo '{"agentName":"champion-1","type":"user","sessionId":"s1"}' >> "$AGENT_L5_T"
run_test "agentName on line 5 (boundary of head -5 window) → suppressed" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$AGENT_L5_T\"}" \
    "" "additionalContext"

# agentName on line 6 (outside head -5 window - treated as orchestrator)
AGENT_L6_T="$TMPDIR/agent_line6.jsonl"
for i in 1 2 3 4 5; do echo '{"type":"system","subtype":"compact"}' >> "$AGENT_L6_T"; done
echo '{"agentName":"champion-1","type":"user","sessionId":"s1"}' >> "$AGENT_L6_T"
run_test "agentName on line 6 (outside head -5 window), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$AGENT_L6_T\"}" \
    "" "additionalContext"

# 5 lines of non-agent system entries → no agentName found → fires as orchestrator
SYSTEM_ONLY_T="$TMPDIR/system_only.jsonl"
for i in 1 2 3 4 5; do
    echo '{"type":"system","subtype":"compact","sessionId":"s1"}' >> "$SYSTEM_ONLY_T"
done
run_test "5 non-agent system lines (no agentName), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$SYSTEM_ONLY_T\"}" \
    "" "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== JSON output is compact (single line) =="
# ════════════════════════════════════════════

run_test "Output is compact JSON (single line)" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    "hookSpecificOutput"

# Verify it's exactly one line
TOTAL=$((TOTAL + 1))
LINE_COUNT=$(echo "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op001\"},\"transcript_path\":\"$ORCH_T\"}" \
    | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$DEFAULT_GPU_RES_DIR" bash "$HOOK" 2>/dev/null | wc -l)
if [ "$LINE_COUNT" -eq 1 ]; then
    echo "  PASS [$TOTAL]: Output is exactly 1 line (compact JSON)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Expected 1 line, got $LINE_COUNT lines"
    FAIL=$((FAIL + 1))
fi

# ════════════════════════════════════════════
echo ""; echo "== Non-champion agents produce no stdout =="
# ════════════════════════════════════════════

TOTAL=$((TOTAL + 1))
STDOUT_BYTES=$(echo "{\"tool_input\":{\"subagent_type\":\"ammo-delegate\",\"name\":\"d1\"},\"transcript_path\":\"$ORCH_T\"}" \
    | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$DEFAULT_GPU_RES_DIR" bash "$HOOK" 2>/dev/null | wc -c)
if [ "$STDOUT_BYTES" -eq 0 ]; then
    echo "  PASS [$TOTAL]: ammo-delegate produces empty stdout"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: ammo-delegate wrote $STDOUT_BYTES bytes to stdout (expected 0)"
    FAIL=$((FAIL + 1))
fi

TOTAL=$((TOTAL + 1))
STDOUT_BYTES=$(echo "{\"tool_input\":{\"subagent_type\":\"ammo-transcript-monitor\",\"name\":\"m1\"},\"transcript_path\":\"$ORCH_T\"}" \
    | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$DEFAULT_GPU_RES_DIR" bash "$HOOK" 2>/dev/null | wc -c)
if [ "$STDOUT_BYTES" -eq 0 ]; then
    echo "  PASS [$TOTAL]: ammo-transcript-monitor produces empty stdout"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: ammo-transcript-monitor wrote $STDOUT_BYTES bytes to stdout (expected 0)"
    FAIL=$((FAIL + 1))
fi

# ════════════════════════════════════════════
echo ""; echo "== Reminder content spot-checks =="
# ════════════════════════════════════════════

check_json_field "Reminder mentions agent name (impl-champion)" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("impl-champion-op003")' \
    "true"

check_json_field "Reminder mentions agent type (impl-champion)" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("ammo-impl-champion")' \
    "true"

check_json_field "Reminder instructs to spawn before other work (impl-champion)" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("before spawning any other agents")' \
    "true"

check_json_field "impl-champion reminder mentions agent name" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("impl-champion-op003")' \
    "true"

check_json_field "impl-champion reminder mentions impl type" \
    "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-champion-op003\"},\"transcript_path\":\"$ORCH_T\"}" \
    '.hookSpecificOutput.additionalContext | contains("ammo-impl-champion")' \
    "true"

# ════════════════════════════════════════════
echo ""; echo "== Corrupt/noisy transcript resilience =="
# ════════════════════════════════════════════

CORRUPT_T="$TMPDIR/corrupt.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$CORRUPT_T"
printf '%20000s\n' '' >> "$CORRUPT_T"   # 20k-space corrupt line
echo '{"type":"user","message":{"role":"user","content":"hello"}}' >> "$CORRUPT_T"
# No agentName line → should fire (treated as orchestrator after jq handles corrupt line)
run_test "Corrupt line in transcript (no agentName), ammo-champion → no output" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-1\"},\"transcript_path\":\"$CORRUPT_T\"}" \
    "" "additionalContext"

CORRUPT_AGENT_T="$TMPDIR/corrupt_with_agent.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$CORRUPT_AGENT_T"
echo '{"agentName":"champion-1","type":"user","sessionId":"s1"}' >> "$CORRUPT_AGENT_T"
printf '%20000s\n' '' >> "$CORRUPT_AGENT_T"
# agentName IS present on line 2 — should suppress
run_test "Corrupt line after agentName entry → suppressed" 0 \
    "{\"tool_input\":{\"subagent_type\":\"ammo-champion\",\"name\":\"champion-2\"},\"transcript_path\":\"$CORRUPT_AGENT_T\"}" \
    "" "additionalContext"

# ════════════════════════════════════════════
echo ""; echo "== Hook writes nothing to the reservation dir =="
# ════════════════════════════════════════════
# The reminder is advisory only: it must not drop marker files. The blocking
# owner of monitor pairing is the ammo_state.py cohort check.

SENT_DIR="$TMPDIR/sentinel-gpures"
rm -rf "$SENT_DIR"; mkdir -p "$SENT_DIR"
TOTAL=$((TOTAL + 1))
echo "{\"tool_input\":{\"subagent_type\":\"ammo-impl-champion\",\"name\":\"impl-pair-1\"},\"transcript_path\":\"$ORCH_T\"}" \
    | env HOME="$TMPDIR" AMMO_GPU_RES_DIR="$SENT_DIR" CLAUDE_SESSION_ID="msid" bash "$HOOK" >/dev/null 2>&1 || true
if [ -z "$(ls -A "$SENT_DIR" 2>/dev/null)" ]; then
    echo "  PASS [$TOTAL]: impl-champion spawn → reservation dir untouched"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: hook wrote to $SENT_DIR: $(ls -A "$SENT_DIR")"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
