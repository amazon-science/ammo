#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# AMMO test runner — runs the full local test surface for the skill template:
#   1. the pytest suites under skills/ammo/tests/, skills/ammo/eval/tests/, and
#      skills/ammo/eval/causal/tests/
#   2. every bash hook harness under hooks/test-*.sh
#
# Each child is judged by its EXIT CODE (each harness exits non-zero on any
# failure), NOT by parsing its summary line — harnesses use different summary
# formats ("Results: N passed ..." vs "N / N passed"), so exit code is the only
# uniform signal. The runner prints a per-child line + an aggregate footer and
# exits 1 if ANY child failed.
#
# Usage:  bash .claude/skills/ammo/tests/run_all.sh
set -uo pipefail

# ── Resolve BASE (the .claude dir) from this script's own location ──
#    THIS = .../.claude/skills/ammo/tests/run_all.sh  →  BASE = .../.claude
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTEST_DIR="$BASE/skills/ammo/tests"
EVAL_TEST_DIR="$BASE/skills/ammo/eval/tests"
CAUSAL_TEST_DIR="$BASE/skills/ammo/eval/causal/tests"
HOOKS_DIR="$BASE/hooks"

FAILED=0
TOTAL=0
declare -a FAILED_NAMES=()

run_child() {
    local label="$1"; shift
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "──────────────────────────────────────────────"
    echo ">>> $label"
    echo "──────────────────────────────────────────────"
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "<<< FAILED ($label) — exit $rc"
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$label")
    else
        echo "<<< OK ($label)"
    fi
}

# ── 1. pytest suite (worktree venv python — never system python3) ──
PYTHON="$BASE/../.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "FAILED: worktree .venv/bin/python not found" >&2
    exit 1
fi
run_child "pytest: skills/ammo/tests" "$PYTHON" -m pytest "$PYTEST_DIR" -q
run_child "pytest: skills/ammo/eval/tests" "$PYTHON" -m pytest "$EVAL_TEST_DIR" -q
run_child "pytest: skills/ammo/eval/causal/tests" "$PYTHON" -m pytest "$CAUSAL_TEST_DIR" -q

# ── 2. every hook harness ──
shopt -s nullglob
for harness in "$HOOKS_DIR"/test-*.sh; do
    run_child "hook: $(basename "$harness")" bash "$harness"
done
shopt -u nullglob

# ── Aggregate footer ──
echo ""
echo "================================================"
echo "AMMO run_all: $((TOTAL - FAILED))/$TOTAL child suites passed"
if [ "$FAILED" -gt 0 ]; then
    echo "FAILED suites:"
    for n in "${FAILED_NAMES[@]}"; do
        echo "  - $n"
    done
    echo "================================================"
    exit 1
fi
echo "All suites green."
echo "================================================"
exit 0
