#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-team-spawn-guard.sh (PreToolUse hook, matcher=Agent)
# Run: bash .claude/hooks/test-ammo-team-spawn-guard.sh
#
# CC 2.1.179 implicit-team model:
#   - TeamCreate / TeamDelete are removed; there is no team-CRUD tool and no
#     per-round team directory to gate on. CC auto-forms one implicit team per
#     session; membership derives purely from spawning with a `name`.
#   - The hook DENIES an Agent spawn of a member type (ammo-champion /
#     ammo-impl-champion / ammo-transcript-monitor) only when it has an EMPTY
#     `name` (a member with no name cannot be addressed via SendMessage).
#   - A member type WITH a name → ALLOW (regardless of any team dir, no team_name).
#   - Every OTHER subagent_type → ALLOW (named subagents are addressable and
#     legitimate, e.g. the eval pipeline's general-purpose causal-analyzer /
#     transcript-grader).
#   - The `team_name` Agent param is deprecated/ignored and is neither required
#     nor checked.
#
# Key fields consumed from hook stdin (probe-confirmed):
#   .tool_input.subagent_type    — e.g. "ammo-champion"
#   .tool_input.name             — e.g. "champion-1" (addressable teammate handle)
#   .agent_type                  — empty on orchestrator, populated in subagents
#
# Deny is emitted as:
#   {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#    "permissionDecision":"deny","permissionDecisionReason":"..."}}
# with exit 0 (Claude Code parses the JSON for the decision).

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-team-spawn-guard.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() {
    rm -rf "$TMPDIR" "$TMPDIR/hook-stdout" "$TMPDIR/hook-stderr" 2>/dev/null || true
}
trap cleanup EXIT

# run_test:
#   $1 name
#   $2 expected_exit (0 always — deny is a JSON output, not a non-zero exit)
#   $3 expected decision: "deny" | "allow" | "no_output"
#   $4 json input
#   $5 (optional) expected substring of permissionDecisionReason (for deny)
#   Extra env: TEST_HOME, TEST_CONFIG_DIR set before call
run_test() {
    local name="$1" expected_exit="$2" expected_decision="$3"
    local json_input="$4" expected_reason="${5:-}"
    local actual_exit=0
    TOTAL=$((TOTAL + 1))

    local home_dir="${TEST_HOME:-$TMPDIR/home-default}"
    mkdir -p "$home_dir"
    local env_args=(env HOME="$home_dir")
    if [ -n "${TEST_CONFIG_DIR:-}" ]; then
        env_args+=(CLAUDE_CONFIG_DIR="$TEST_CONFIG_DIR")
    fi
    if [ -n "${TEST_PATH_OVERRIDE:-}" ]; then
        env_args+=(PATH="$TEST_PATH_OVERRIDE")
    fi

    echo "$json_input" | "${env_args[@]}" bash "$HOOK" \
        > "$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?

    local pass=true
    if [ "$actual_exit" -ne "$expected_exit" ]; then
        pass=false
    fi

    case "$expected_decision" in
        deny)
            if ! grep -q '"permissionDecision"[[:space:]]*:[[:space:]]*"deny"' "$TMPDIR/hook-stdout" 2>/dev/null; then
                pass=false
            fi
            if [ -s "$TMPDIR/hook-stdout" ] && ! jq . "$TMPDIR/hook-stdout" >/dev/null 2>&1; then
                echo "  WARN: stdout is not valid JSON"
                pass=false
            fi
            if [ -n "$expected_reason" ]; then
                if ! grep -qF "$expected_reason" "$TMPDIR/hook-stdout" 2>/dev/null; then
                    pass=false
                fi
            fi
            ;;
        allow)
            # Allow = exit 0 AND no deny decision (empty or allow JSON both ok)
            if grep -q '"permissionDecision"[[:space:]]*:[[:space:]]*"deny"' "$TMPDIR/hook-stdout" 2>/dev/null; then
                pass=false
            fi
            ;;
        no_output)
            if [ -s "$TMPDIR/hook-stdout" ]; then
                pass=false
            fi
            ;;
    esac

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected_exit=$expected_exit, got=$actual_exit, want=$expected_decision)"
        echo "        stdout: $(head -3 "$TMPDIR/hook-stdout" 2>/dev/null || echo '(none)')"
        echo "        stderr: $(head -3 "$TMPDIR/hook-stderr" 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

# ══════════════════════════════════════════════
echo "== T1-T3: Member types WITH a name → ALLOW (no team_name needed) =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t1" TEST_CONFIG_DIR=""
run_test "T1 ammo-champion with name (no team_name) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1"}}'

run_test "T2 ammo-impl-champion with name (no team_name) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"impl-op001"}}'

run_test "T3 ammo-transcript-monitor with name (no team_name) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-transcript-monitor","name":"monitor-champion-1"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T4-T5: Member types with a name → ALLOW regardless of any team dir =="
# ══════════════════════════════════════════════

# No team-directory scaffolding: the dir model is gone. A name is sufficient.
TEST_HOME="$TMPDIR/t4" TEST_CONFIG_DIR=""
run_test "T4 ammo-champion with name → ALLOW (no team dir exists)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1"}}'

TEST_HOME="$TMPDIR/t5" TEST_CONFIG_DIR=""
run_test "T5 ammo-impl-champion with name → ALLOW (no team dir exists)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"impl-1"}}'

run_test "T5 ammo-transcript-monitor with name → ALLOW (no team dir exists)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-transcript-monitor","name":"monitor-impl-1"}}'

# Deprecated team_name still present → ALLOW (it is ignored, not gated).
run_test "T5 ammo-champion with name + (ignored) team_name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1","team_name":"ammo-round-1-qwen-h100"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T5b: Member type with EMPTY name → DENY (cannot be addressed) =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t5b" TEST_CONFIG_DIR=""
run_test "T5b ammo-champion with empty name → DENY (needs name)" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":""}}' \
    "non-empty tool_input.name"

run_test "T5b ammo-impl-champion with no name field → DENY (needs name)" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion"}}' \
    "non-empty tool_input.name"

# A leftover team_name does NOT satisfy the requirement — a name is still required.
run_test "T5b ammo-champion with team_name but empty name → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","team_name":"ammo-round-2-new"}}' \
    "non-empty tool_input.name"

# ══════════════════════════════════════════════
echo ""; echo "== T6: Out-of-scope (non-member) subagent types → ALLOW =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t6" TEST_CONFIG_DIR=""
run_test "T6 ammo-researcher with name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-researcher","name":"baseline-research"}}'

run_test "T6 general-purpose with name → ALLOW (named eval teammate is legitimate)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"general-purpose","name":"causal-analyzer"}}'

run_test "T6 general-purpose with name=transcript-grader → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"general-purpose","name":"transcript-grader"}}'

run_test "T6 ammo-delegate with name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-delegate","name":"dl"}}'

run_test "T6 ammo-delegate without name → ALLOW (one-shot subagent)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-delegate"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T7: Inside subagent (agent_type set) → ALLOW =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t7" TEST_CONFIG_DIR=""
# Even for a gated subagent_type, we MUST allow because the hook is running
# inside a subagent, not on the orchestrator thread.
run_test "T7 agent_type=ammo-delegate top-level → ALLOW (not orchestrator)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c"},"agent_type":"ammo-delegate"}'

run_test "T7 agent_type=ammo-researcher top-level → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"i"},"agent_type":"ammo-researcher"}'

# Inside a subagent, even an empty-name member type is ALLOWED (gate is off).
run_test "T7 agent_type set + member type with empty name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":""},"agent_type":"ammo-delegate"}'

# ══════════════════════════════════════════════
echo ""; echo "== T9: Missing tool_input.subagent_type → ALLOW (fail-open) =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t9" TEST_CONFIG_DIR=""
run_test "T9 no tool_input.subagent_type but a name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"name":"c"}}'

run_test "T9 empty tool_input → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{}}'

run_test "T9 no tool_input at all → ALLOW" 0 allow \
    '{"tool_name":"Agent"}'

# ══════════════════════════════════════════════
echo ""; echo "== T10: team_name is ignored — never gates a member spawn =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t10" TEST_CONFIG_DIR=""
# team_name with dots, no dir anywhere — under the implicit-team model this is
# irrelevant; the name is what matters → ALLOW.
run_test "T10 ammo-champion with name + dotted team_name (no dir) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-1-qwen3.5-4b"}}'

run_test "T10b ammo-champion with name + arbitrary team_name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-9-zzz.9"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T11: CLAUDE_CONFIG_DIR set is irrelevant — name still gates =="
# ══════════════════════════════════════════════

T11_HOME="$TMPDIR/t11home"
T11_CFG="$TMPDIR/t11cfg"
mkdir -p "$T11_HOME" "$T11_CFG"
TEST_HOME="$T11_HOME" TEST_CONFIG_DIR="$T11_CFG"
run_test "T11 CLAUDE_CONFIG_DIR set + named member → ALLOW (no dir check)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-1-config-override"}}'

TEST_HOME="$T11_HOME" TEST_CONFIG_DIR="$T11_CFG"
run_test "T11 CLAUDE_CONFIG_DIR set + member with empty name → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"","team_name":"ammo-round-2"}}' \
    "non-empty tool_input.name"

# ══════════════════════════════════════════════
echo ""; echo "== T12: jq unavailable → ALLOW (fail-open) =="
# ══════════════════════════════════════════════

# Build a minbin dir that symlinks every PATH tool EXCEPT jq. The hook's
# `command -v jq` must return nothing and the hook must exit 0 silently.
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env sed grep tr dirname basename mktemp stat head tail awk printf test which rm ls mkdir touch cut date; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
TEST_HOME="$TMPDIR/t12" TEST_CONFIG_DIR="" TEST_PATH_OVERRIDE="$MIN_PATH"
run_test "T12 jq unavailable → ALLOW (fail-open)" 0 no_output \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c"}}'
unset TEST_PATH_OVERRIDE

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open edge cases =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/fedge" TEST_CONFIG_DIR=""
run_test "Empty JSON {} → ALLOW (no subagent_type)" 0 allow '{}'
run_test "Non-Agent tool_name → ALLOW" 0 allow \
    '{"tool_name":"Bash","tool_input":{"command":"ls"}}'
run_test "tool_input is null → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":null}'

# ══════════════════════════════════════════════
echo ""; echo "== Round-transition scenario: named members across rounds → ALLOW =="
# ══════════════════════════════════════════════

# No team dirs, no team_name needed — successive rounds just spawn named members
# into the session's single implicit team.
TEST_HOME="$TMPDIR/rt" TEST_CONFIG_DIR=""
run_test "Round 1 named champion → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c"}}'
run_test "Round 2 named impl-champion → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"i"}}'
run_test "Round 3 champion with empty name → DENY (needs name)" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":""}}' \
    "non-empty tool_input.name"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
