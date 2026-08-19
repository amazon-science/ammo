#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-pip-guard.sh (PreToolUse hook, matcher=Bash)
# Run: bash .claude/hooks/test-ammo-pip-guard.sh
#
# The hook is an always-on, session-wide DENY of package-install commands:
#   pip install / pip3 install / uv pip install / python[3] -m pip install
#   (and the uninstall variants).
# It blocks via `exit 2` + stderr pointing at the .venv policy (the established
# PreToolUse Bash deny convention here — see ammo-venv-python-guard.sh:37,
# ammo-pretool-guard.sh:113). Read-only pip forms (list/show/freeze/--version)
# are ALLOWED. Escape hatch: AMMO_ALLOW_PIP=1 → ALLOW. Fail-open: missing jq.
#
# This guard is DISTINCT from:
#   - ammo-pretool-guard.sh   (wrong-tree one-shot venv block; campaign-scoped)
#   - ammo-venv-python-guard.sh (bare-python sweep-script block)
# It fires unconditionally on every Bash invocation.

set -euo pipefail

# AMMO_TEST_HOOK lets a reviewer aim this harness at an ARCHIVED copy of the
# hook to see which cases the archived version fails (red->green evidence for
# the flag/path/wrapper bypass block below).
HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-pip-guard.sh}"
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
#   Env knobs honored: TEST_ALLOW_PIP, TEST_PATH_OVERRIDE
run_test() {
    local name="$1" expected_exit="$2" command="$3" expected_err="${4:-}"
    local actual_exit=0
    TOTAL=$((TOTAL + 1))

    local payload
    payload=$(jq -n -c --arg cmd "$command" '{tool_name:"Bash",tool_input:{command:$cmd}}')

    local env_args=(env)
    if [ -n "${TEST_ALLOW_PIP:-}" ]; then
        env_args+=(AMMO_ALLOW_PIP="$TEST_ALLOW_PIP")
    fi
    if [ -n "${TEST_PATH_OVERRIDE:-}" ]; then
        env_args+=(PATH="$TEST_PATH_OVERRIDE")
    fi

    echo "$payload" | "${env_args[@]}" bash "$HOOK" \
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

# ══════════════════════════════════════════════
echo "== DENY: install/uninstall forms (exit 2) =="
# ══════════════════════════════════════════════
run_test "1 pip install foo → DENY (exit 2, stderr names .venv)" 2 "pip install foo" ".venv"
run_test "2 pip3 install numpy → DENY" 2 "pip3 install numpy" ".venv"
run_test "3 uv pip install -e . → DENY" 2 "uv pip install -e ." ".venv"
run_test "4 python -m pip install x → DENY" 2 "python -m pip install x" ".venv"
run_test "5 python3 -m pip install x → DENY" 2 "python3 -m pip install x" ".venv"
run_test "6 pip uninstall foo → DENY" 2 "pip uninstall foo" ".venv"
run_test "7 uv pip uninstall foo → DENY" 2 "uv pip uninstall foo" ".venv"
# Embedded in a larger command (env prefix + chaining) still denies.
run_test "8 cd /x && pip install foo → DENY (matches anywhere)" 2 "cd /x && pip install foo" ".venv"
run_test "9 VLLM_USE_PRECOMPILED=1 uv pip install -e . → DENY" 2 "VLLM_USE_PRECOMPILED=1 uv pip install -e ." ".venv"

# ══════════════════════════════════════════════
echo ""; echo "== ALLOW: read-only pip forms (exit 0) =="
# ══════════════════════════════════════════════
run_test "10 pip list → ALLOW" 0 "pip list"
run_test "11 pip show numpy → ALLOW" 0 "pip show numpy"
run_test "12 pip freeze → ALLOW" 0 "pip freeze"
run_test "13 pip --version → ALLOW" 0 "pip --version"

# ══════════════════════════════════════════════
echo ""; echo "== ALLOW: unrelated commands (exit 0) =="
# ══════════════════════════════════════════════
run_test "14 ls → ALLOW" 0 "ls -la"
run_test "15 python script.py → ALLOW (no pip install)" 0 "python script.py"
run_test "16 echo pipeline → ALLOW (substring 'pip' not an install)" 0 "echo running pipeline"
run_test "17 grep install file → ALLOW (no pip invoker)" 0 "grep install requirements.txt"
# Quoted/argument-position mentions must NOT match — the invoker has to sit at
# command position (regression tests for the substring-over-block fix).
run_test "17b grep \"pip install\" → ALLOW (quoted mention, not a command)" 0 "grep \"pip install\" file.py"
run_test "17c echo '...pip install...' → ALLOW (quoted mention)" 0 "echo \"do not pip install things\""
run_test "17d rg 'uv pip install' → ALLOW (search pattern)" 0 "rg \"uv pip install\" docs/"
run_test "17e git log --grep='pip install' → ALLOW (grep arg)" 0 "git log --grep=\"pip install\""

# ══════════════════════════════════════════════
echo ""; echo "== Bypass: flag between invoker and verb, path-qualified, wrapped, nested =="
# ══════════════════════════════════════════════
# The pre-fix regex required the verb to sit IMMEDIATELY after a literal
# invoker name, so every case here returned exit 0. The classifier tokenizes
# the command and inspects each segment at command position instead.
run_test "22 pip -q install → DENY (flag between invoker and verb)" 2 "pip -q install numpy" ".venv"
run_test "23 pip3 --quiet install → DENY" 2 "pip3 --quiet install numpy" ".venv"
run_test "24 python -m pip -q install → DENY" 2 "python -m pip -q install numpy" ".venv"
run_test "25 uv pip -q install → DENY" 2 "uv pip -q install numpy" ".venv"
run_test "26 /usr/bin/pip install → DENY (path-qualified invoker)" 2 "/usr/bin/pip install numpy" ".venv"
run_test "27 .venv/bin/pip install -e . → DENY (path-qualified)" 2 ".venv/bin/pip install -e ." ".venv"
run_test "28 /opt/py/bin/python3 -m pip install → DENY (path-qualified interpreter)" 2 \
    "/opt/py/bin/python3 -m pip install numpy" ".venv"
run_test "29 sudo pip install → DENY (wrapper)" 2 "sudo pip install numpy" ".venv"
run_test "30 env FOO=1 uv pip install → DENY (env wrapper)" 2 "env FOO=1 uv pip install -e ." ".venv"
run_test "31 bash -c \"pip install\" → DENY (nested shell)" 2 'bash -c "pip install numpy"' ".venv"
run_test "32 sh -c 'python -m pip install' → DENY (nested shell)" 2 \
    "sh -c 'python -m pip install numpy'" ".venv"
run_test "33 bash -lc \"cd /w && uv pip install -e .\" → DENY (nested, chained)" 2 \
    'bash -lc "cd /w && uv pip install -e ."' ".venv"
run_test "34 pip -q uninstall → DENY (uninstall variant with flag)" 2 "pip -q uninstall -y numpy" ".venv"
# The widened detection must not start matching read-only or unrelated forms.
run_test "35 pip -q list → ALLOW (flag + read-only verb)" 0 "pip -q list"
run_test "36 pip config get global.index-url → ALLOW" 0 "pip config get global.index-url"
run_test "37 pipx run cowsay → ALLOW (not a pip invoker)" 0 "pipx run cowsay hi"
run_test "38 apt install (non-python) → ALLOW (out of this guard's scope)" 0 "apt install jq"

# ══════════════════════════════════════════════
echo ""; echo "== Escape hatch: AMMO_ALLOW_PIP=1 =="
# ══════════════════════════════════════════════
TEST_ALLOW_PIP="1"
run_test "18 AMMO_ALLOW_PIP=1 + pip install foo → ALLOW (bypass)" 0 "pip install foo"
run_test "19 AMMO_ALLOW_PIP=1 + uv pip install -e . → ALLOW" 0 "uv pip install -e ."
TEST_ALLOW_PIP=""

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open: jq unavailable =="
# ══════════════════════════════════════════════
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env sed grep tr dirname basename mktemp head tail awk printf test which rm ls mkdir touch cut; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
TEST_PATH_OVERRIDE="$MIN_PATH"
# With jq gone the harness can't build the payload via jq, so pass a literal.
TOTAL=$((TOTAL + 1))
nojq_exit=0
echo '{"tool_name":"Bash","tool_input":{"command":"pip install foo"}}' \
    | env PATH="$MIN_PATH" bash "$HOOK" >/dev/null 2>&1 || nojq_exit=$?
if [ "$nojq_exit" -eq 0 ]; then
    echo "  PASS [$TOTAL]: 20 jq unavailable + pip install → ALLOW (fail-open)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: 20 jq unavailable → expected exit 0, got $nojq_exit"
    FAIL=$((FAIL + 1))
fi
TEST_PATH_OVERRIDE=""

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open: empty / missing command =="
# ══════════════════════════════════════════════
TOTAL=$((TOTAL + 1))
empty_exit=0
echo '{"tool_name":"Bash","tool_input":{}}' | bash "$HOOK" >/dev/null 2>&1 || empty_exit=$?
if [ "$empty_exit" -eq 0 ]; then
    echo "  PASS [$TOTAL]: 21 missing command → ALLOW (fail-open)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: 21 missing command → expected exit 0, got $empty_exit"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
