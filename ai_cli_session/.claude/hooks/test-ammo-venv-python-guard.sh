#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-venv-python-guard.sh (PreToolUse hook, matcher=Bash)
# Run: bash .claude/hooks/test-ammo-venv-python-guard.sh
#
# The hook blocks bare `python` / `python3` invocations of the vllm-dependent
# sweep/profiling scripts (run_vllm_bench_latency_sweep.py, ncu_sanity_driver.py)
# because system python has no vllm. It denies via `exit 2` + stderr.
#
# The script name is matched with ANY path prefix, or none — same semantics as
# the Codex twin (_SWEEP_SCRIPTS_RE in .codex/hooks/pre_tool_use_guard.py). A
# path-prefix-only match let `python ncu_sanity_driver.py` through on Claude
# while Codex blocked it.
#
# ALLOWED: a `.venv/bin/python` prefix, a `source .venv/bin/activate &&` prefix,
# and management scripts that do not import vllm (gpu_reservation.py,
# new_target.py).
#
# AMMO_TEST_HOOK lets a reviewer aim this harness at an ARCHIVED copy of the
# hook to see which cases the archived version fails.

set -euo pipefail

HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-venv-python-guard.sh}"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# run_test:
#   $1 name
#   $2 expected_exit  (2 = DENY, 0 = ALLOW)
#   $3 command string (placed in .tool_input.command)
#   $4 (optional) expected substring of stderr (deny only)
run_test() {
    local name="$1" expected_exit="$2" command="$3" expected_err="${4:-}"
    local actual_exit=0
    TOTAL=$((TOTAL + 1))

    local payload
    payload=$(jq -n -c --arg cmd "$command" '{tool_name:"Bash",tool_input:{command:$cmd}}')

    echo "$payload" | bash "$HOOK" \
        > "$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?

    local pass=true
    [ "$actual_exit" -ne "$expected_exit" ] && pass=false
    if [ -n "$expected_err" ]; then
        grep -qF "$expected_err" "$TMPDIR/hook-stderr" 2>/dev/null || pass=false
    fi

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected_exit=$expected_exit, got=$actual_exit)"
        echo "        cmd: $command"
        echo "        stderr: $(head -3 "$TMPDIR/hook-stderr" 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

SKILL_DIR=".claude/skills/ammo/scripts"

# ══════════════════════════════════════════════
echo "== DENY: bare python, full skill path (exit 2) =="
# ══════════════════════════════════════════════
run_test "1 python + full path sweep runner → DENY" 2 \
    "python $SKILL_DIR/run_vllm_bench_latency_sweep.py --slot base" ".venv/bin/python"
run_test "2 python3 + full path ncu driver → DENY" 2 \
    "python3 $SKILL_DIR/ncu_sanity_driver.py --target-json target.json --batch-size 8" ".venv/bin/python"

# ══════════════════════════════════════════════
echo ""; echo "== DENY: bare filename, no path prefix (Codex parity) =="
# ══════════════════════════════════════════════
run_test "3 python + bare ncu_sanity_driver.py → DENY" 2 \
    "python ncu_sanity_driver.py --target-json target.json --batch-size 8" ".venv/bin/python"
run_test "4 python + bare run_vllm_bench_latency_sweep.py → DENY" 2 \
    "python run_vllm_bench_latency_sweep.py --slot base" ".venv/bin/python"

# ══════════════════════════════════════════════
echo ""; echo "== DENY: relative ./ path =="
# ══════════════════════════════════════════════
run_test "5 python ./ncu_sanity_driver.py → DENY" 2 \
    "python ./ncu_sanity_driver.py --batch-size 8" ".venv/bin/python"
run_test "6 python ./scripts/run_vllm_bench_latency_sweep.py → DENY" 2 \
    "python ./scripts/run_vllm_bench_latency_sweep.py --slot base" ".venv/bin/python"

# ══════════════════════════════════════════════
echo ""; echo "== ALLOW: .venv/bin/python prefix (exit 0) =="
# ══════════════════════════════════════════════
run_test "7 .venv/bin/python + full path sweep runner → ALLOW" 0 \
    ".venv/bin/python $SKILL_DIR/run_vllm_bench_latency_sweep.py --slot base"
run_test "8 .venv/bin/python + bare ncu driver → ALLOW" 0 \
    ".venv/bin/python ncu_sanity_driver.py --batch-size 8"
run_test "9 absolute .venv/bin/python → ALLOW" 0 \
    "/data/sessions/s1/worktree/.venv/bin/python $SKILL_DIR/ncu_sanity_driver.py --batch-size 8"

# ══════════════════════════════════════════════
echo ""; echo "== ALLOW: source .venv/bin/activate prefix =="
# ══════════════════════════════════════════════
run_test "10 activate && python + full path → ALLOW" 0 \
    "source .venv/bin/activate && python $SKILL_DIR/run_vllm_bench_latency_sweep.py --slot base"
run_test "11 activate && python + bare filename → ALLOW" 0 \
    "source .venv/bin/activate && python ncu_sanity_driver.py --batch-size 8"

# ══════════════════════════════════════════════
echo ""; echo "== ALLOW: management scripts and unrelated commands =="
# ══════════════════════════════════════════════
run_test "12 python gpu_reservation.py reserve → ALLOW (no vllm import)" 0 \
    "python $SKILL_DIR/gpu_reservation.py reserve --num-gpus 1 --session-id t1"
run_test "13 python + bare gpu_reservation.py → ALLOW" 0 \
    "python gpu_reservation.py status"
run_test "14 python new_target.py → ALLOW" 0 \
    "python $SKILL_DIR/new_target.py --model-id x"
run_test "15 ls → ALLOW" 0 "ls -la"
run_test "15a python3.12 + bare ncu driver → DENY (versioned interpreter)" 2 \
    "python3.12 ncu_sanity_driver.py --batch-size 8"
run_test "15b .venv/bin/python3.12 + bare ncu driver → ALLOW" 0 \
    ".venv/bin/python3.12 ncu_sanity_driver.py --batch-size 8"
run_test "16 cat ncu_sanity_driver.py → ALLOW (not a python invocation)" 0 \
    "cat ncu_sanity_driver.py"
run_test "17 bash ncu_sanity.sh wrapper → ALLOW (wrapper picks the venv python)" 0 \
    "CSV_OUT=ncu/s.csv TARGET_JSON=target.json BATCH_SIZE=8 ./ncu_sanity.sh flash_fwd"

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open: empty / missing command =="
# ══════════════════════════════════════════════
TOTAL=$((TOTAL + 1))
empty_exit=0
echo '{"tool_name":"Bash","tool_input":{}}' | bash "$HOOK" >/dev/null 2>&1 || empty_exit=$?
if [ "$empty_exit" -eq 0 ]; then
    echo "  PASS [$TOTAL]: 18 missing command → ALLOW (fail-open)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: 18 missing command → expected exit 0, got $empty_exit"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
