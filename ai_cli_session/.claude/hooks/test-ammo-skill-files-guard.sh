#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-skill-files-guard.sh (PreToolUse Edit|Write|MultiEdit|
# NotebookEdit). Run: bash .claude/hooks/test-ammo-skill-files-guard.sh
#
# Contract: a SUBAGENT may not edit orchestrator-owned skill/agent definition
# files; the team LEAD may. The hook denies via a permissionDecision:"deny" JSON
# body on stdout and always exits 0 (PreToolUse deny convention for this hook —
# distinct from the exit-2 convention used by the two Bash guards).
#
# Protected set:
#   .claude/skills/ammo/references/     .claude/skills/ammo/scripts/
#   .claude/skills/ammo/orchestration/  .claude/schemas/
#   .claude/agents/
#
# scripts/ and schemas/ are the audit-invariants.md "Mechanical Authorities":
# the state engine, the gate verifier, and the schema a track is judged by. A
# subagent that can rewrite them can pass itself. Those cases are ALLOW on the
# pre-fix hook; aim this harness at an archived copy with AMMO_TEST_HOOK to see
# it fail them.
set -euo pipefail

HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-skill-files-guard.sh}"
HELPER_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# An archived hook copy still sources _ammo_is_lead.sh from ITS own dir, so
# stage the archived hook next to the live helper when AMMO_TEST_HOOK is set.
if [ -n "${AMMO_TEST_HOOK:-}" ]; then
    mkdir -p "$TMPDIR/hooks"
    cp "$AMMO_TEST_HOOK" "$TMPDIR/hooks/ammo-skill-files-guard.sh"
    cp "$HELPER_DIR/_ammo_is_lead.sh" "$TMPDIR/hooks/_ammo_is_lead.sh"
    HOOK="$TMPDIR/hooks/ammo-skill-files-guard.sh"
fi

# run_test:
#   $1 name
#   $2 expected verdict: DENY | ALLOW
#   $3 file_path
#   $4 identity: subagent | lead
run_test() {
    local name="$1" expect="$2" path="$3" identity="$4"
    TOTAL=$((TOTAL + 1))

    local payload
    if [ "$identity" = "subagent" ]; then
        payload=$(jq -nc --arg p "$path" \
            '{tool_name:"Edit",session_id:"s1",agent_type:"ammo-impl-champion",tool_input:{file_path:$p}}')
    else
        payload=$(jq -nc --arg p "$path" \
            '{tool_name:"Edit",session_id:"s1",tool_input:{file_path:$p}}')
    fi

    local rc=0
    # HOME points at an empty dir so the lead case has no teams/ to scan.
    echo "$payload" | env HOME="$TMPDIR" bash "$HOOK" \
        >"$TMPDIR/out" 2>"$TMPDIR/err" || rc=$?

    local got="ALLOW"
    if grep -q '"permissionDecision": *"deny"' "$TMPDIR/out" 2>/dev/null; then
        got="DENY"
    fi

    local ok=true
    [ "$got" = "$expect" ] || ok=false
    # The hook must always exit 0 — the JSON body is the enforcement channel.
    [ "$rc" -eq 0 ] || ok=false
    # A deny body must be valid JSON, or Claude Code ignores it.
    if [ "$got" = "DENY" ]; then
        jq . "$TMPDIR/out" >/dev/null 2>&1 || ok=false
    fi

    if [ "$ok" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected=$expect got=$got exit=$rc)"
        echo "        path: $path"
        echo "        out:  $(head -c 200 "$TMPDIR/out" 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

W="/w/.claude"

echo "== Subagent DENY: the pre-existing protected set =="
run_test "references/ md → DENY" DENY "$W/skills/ammo/references/audit-invariants.md" subagent
run_test "orchestration/ md → DENY" DENY "$W/skills/ammo/orchestration/parallel-tracks.md" subagent
run_test "agents/ md → DENY" DENY "$W/agents/ammo-impl-champion.md" subagent

echo ""; echo "== Subagent DENY: Mechanical Authorities (new) =="
# Every case in this block is ALLOW on the pre-fix hook.
run_test "scripts/ammo_state.py → DENY (state engine)" DENY \
    "$W/skills/ammo/scripts/ammo_state.py" subagent
run_test "scripts/verify_validation_gates.py → DENY (gate verifier)" DENY \
    "$W/skills/ammo/scripts/verify_validation_gates.py" subagent
run_test "scripts/reconcile_track_state.py → DENY" DENY \
    "$W/skills/ammo/scripts/reconcile_track_state.py" subagent
run_test "schemas/state.schema.json → DENY (the schema it is judged by)" DENY \
    "$W/schemas/state.schema.json" subagent
run_test "absolute worktree path under scripts/ → DENY" DENY \
    "/data/sessions/x/worktree/.claude/skills/ammo/scripts/ammo_state.py" subagent
run_test "relative path under schemas/ → DENY" DENY \
    ".claude/schemas/state.schema.json" subagent

echo ""; echo "== Lead ALLOW: the same protected paths =="
run_test "lead edits scripts/ammo_state.py → ALLOW" ALLOW \
    "$W/skills/ammo/scripts/ammo_state.py" lead
run_test "lead edits schemas/state.schema.json → ALLOW" ALLOW \
    "$W/schemas/state.schema.json" lead
run_test "lead edits references/ md → ALLOW" ALLOW \
    "$W/skills/ammo/references/audit-invariants.md" lead

echo ""; echo "== ALLOW: paths outside the protected set =="
run_test "subagent edits vllm source → ALLOW" ALLOW \
    "/w/vllm/vllm/attention/layer.py" subagent
run_test "subagent edits its own artifacts → ALLOW" ALLOW \
    "/w/kernel_opt_artifacts/tgt/rounds/1/notes.md" subagent
run_test "subagent edits state.json → ALLOW (state-validate owns that)" ALLOW \
    "/w/kernel_opt_artifacts/tgt/state.json" subagent
run_test "subagent edits tests under skills/ammo/tests → ALLOW" ALLOW \
    "$W/skills/ammo/tests/test_ammo_state.py" subagent
# A near-miss must not match: schemas/ outside .claude is not the authority.
run_test "subagent edits vllm/schemas/x.json → ALLOW (not .claude/schemas)" ALLOW \
    "/w/vllm/schemas/x.json" subagent

echo ""; echo "== Fail-open =="
TOTAL=$((TOTAL + 1))
rc=0
echo '{"tool_name":"Edit","tool_input":{}}' | bash "$HOOK" >"$TMPDIR/out" 2>&1 || rc=$?
if [ "$rc" -eq 0 ] && [ ! -s "$TMPDIR/out" ]; then
    echo "  PASS [$TOTAL]: no file_path → ALLOW, silent"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: no file_path → expected silent exit 0, got exit=$rc"
    FAIL=$((FAIL + 1))
fi

TOTAL=$((TOTAL + 1))
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env printf dirname basename test rm; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
rc=0
echo '{"tool_name":"Edit","tool_input":{"file_path":"/w/.claude/schemas/state.schema.json"}}' \
    | env PATH="$MIN_PATH" bash "$HOOK" >/dev/null 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
    echo "  PASS [$TOTAL]: jq unavailable → ALLOW (fail-open)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: jq unavailable → expected exit 0, got $rc"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
