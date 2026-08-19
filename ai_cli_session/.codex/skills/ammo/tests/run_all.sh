#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# AMMO Codex mechanical test runner. Each child is judged by exit code; output
# text is never parsed as a pass signal.
#
# Usage: bash .codex/skills/ammo/tests/run_all.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTEST_DIR="$BASE/skills/ammo/tests"
EVAL_TEST_DIR="$BASE/skills/ammo/eval/tests"
CAUSAL_TEST_DIR="$BASE/skills/ammo/eval/causal/tests"
HOOK_TEST="$PYTEST_DIR/test_hook_semantics.py"

FAILED=0
TOTAL=0
declare -a FAILED_NAMES=()

run_child() {
    local label="$1"
    shift
    TOTAL=$((TOTAL + 1))
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAILED: $label (exit $rc)"
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$label")
    else
        echo "OK: $label"
    fi
}

PYTHON="$BASE/../.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "FAILED: worktree .venv/bin/python not found" >&2
    exit 1
fi
if [ ! -f "$HOOK_TEST" ]; then
    echo "FAILED: required Codex hook suite missing: test_hook_semantics.py" >&2
    exit 1
fi

run_child "pytest: skills/ammo/tests" "$PYTHON" -m pytest "$PYTEST_DIR" -q
run_child "pytest: skills/ammo/eval/tests" "$PYTHON" -m pytest "$EVAL_TEST_DIR" -q
run_child "pytest: skills/ammo/eval/causal/tests" "$PYTHON" -m pytest "$CAUSAL_TEST_DIR" -q

echo "AMMO run_all: $((TOTAL - FAILED))/$TOTAL child suites passed"
if [ "$FAILED" -gt 0 ]; then
    printf 'FAILED suites:\n'
    printf '  - %s\n' "${FAILED_NAMES[@]}"
    exit 1
fi
echo "All suites green."
