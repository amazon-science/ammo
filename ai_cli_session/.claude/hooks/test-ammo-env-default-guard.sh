#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-env-default-guard.sh (PreToolUse Edit|Write).
# Run: bash .claude/hooks/test-ammo-env-default-guard.sh
#
# Contract: block an envs.py edit that registers a CAMPAIGN VLLM_* gating flag
# with a truthy default (exit 2 + stderr naming the flag); allow default-off
# spellings, infra-legit allowlisted names, platform prefixes, comments, and
# non-envs.py files.
#
# The pre-fix hook keyed on VLLM_OP\d+ only. impl-track-rules.md § Env Flag
# Naming (PR-Ready) FORBIDS that shape, so every conforming flag name sailed
# through. Aim this harness at an archived copy with AMMO_TEST_HOOK to see the
# "conforming name" and "getenv idiom" blocks fail there.
set -euo pipefail

HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-env-default-guard.sh}"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# run_test:
#   $1 name  $2 expected_exit (2=BLOCK 0=ALLOW)  $3 content
#   $4 (optional) file_path, default an envs.py
#   $5 (optional) expected stderr substring
run_test() {
    local name="$1" expected="$2" content="$3"
    local path="${4:-/workspace/vllm/vllm/envs.py}" want_err="${5:-}"
    local rc=0
    TOTAL=$((TOTAL + 1))

    jq -nc --arg p "$path" --arg c "$content" \
        '{session_id:"s1",hook_event_name:"PreToolUse",tool_name:"Edit",tool_input:{file_path:$p,old_string:"# old",new_string:$c}}' \
        | bash "$HOOK" >"$TMPDIR/out" 2>"$TMPDIR/err" || rc=$?

    local ok=true
    [ "$rc" -eq "$expected" ] || ok=false
    if [ -n "$want_err" ]; then
        grep -qF "$want_err" "$TMPDIR/err" 2>/dev/null || ok=false
    fi
    # A block must never write to stdout — PreToolUse stdout is the JSON channel.
    [ -s "$TMPDIR/out" ] && ok=false

    if [ "$ok" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected_exit=$expected got=$rc)"
        echo "        content: $content"
        echo "        stderr:  $(head -2 "$TMPDIR/err" 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

echo "== BLOCK: legacy VLLM_OP<n> shape (pre-existing coverage) =="
run_test "1 VLLM_OP001: bool = True" 2 'VLLM_OP001: bool = True' "" "BLOCKED"
run_test "2 VLLM_OP002: bool = 1" 2 'VLLM_OP002: bool = 1' "" "BLOCKED"
run_test '3 "VLLM_OP004", True' 2 '    "VLLM_OP004", True' "" "BLOCKED"

echo ""; echo "== BLOCK: conforming VLLM_<SCOPE>_<MECHANISM> names (new) =="
# Every case in this block is ALLOW (exit 0) on the pre-fix hook.
run_test "4 VLLM_MOE_TWO_STREAM: bool = True" 2 \
    'VLLM_MOE_TWO_STREAM: bool = True' "" "VLLM_MOE_TWO_STREAM"
run_test "5 VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100: bool = True" 2 \
    'VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100: bool = True' "" "BLOCKED"
run_test "6 VLLM_GDN_BF16_STATE = 1" 2 'VLLM_GDN_BF16_STATE = 1' "" "BLOCKED"
run_test "7 comma-True dict entry, conforming name" 2 \
    '    "VLLM_QWEN3_LMHEAD_W8A16", True' "" "VLLM_QWEN3_LMHEAD_W8A16"
run_test "8 default=True kwarg, conforming name" 2 \
    '    register("VLLM_TRITON_SKINNY_GEMM", default=True)' "" "BLOCKED"

echo ""; echo "== BLOCK: os.getenv / os.environ.get truthy-default idioms (new) =="
run_test "9 bool(int(os.getenv(KEY,\"1\")))" 2 \
    '    "VLLM_QWEN3_5_FP8_MLP_GATE_UP_SM89": lambda: bool(int(os.getenv("VLLM_QWEN3_5_FP8_MLP_GATE_UP_SM89", "1"))),' \
    "" "VLLM_QWEN3_5_FP8_MLP_GATE_UP_SM89"
run_test "10 os.environ.get(KEY,\"1\") == \"1\"" 2 \
    '    "VLLM_ATTN_GATED_RMS_NORM_FUSION": lambda: os.environ.get("VLLM_ATTN_GATED_RMS_NORM_FUSION", "1") == "1",' \
    "" "VLLM_ATTN_GATED_RMS_NORM_FUSION"
run_test "11 os.getenv(KEY,\"1\") on the legacy shape too" 2 \
    '    "VLLM_OP019": lambda: bool(int(os.getenv("VLLM_OP019", "1"))),' "" "BLOCKED"

echo ""; echo "== ALLOW: default-off spellings =="
run_test "12 bool = False" 0 'VLLM_MOE_TWO_STREAM: bool = False'
run_test "13 bool(os.getenv(KEY,\"0\") == \"1\")" 0 \
    '    "VLLM_MOE_TWO_STREAM": lambda: bool(os.getenv("VLLM_MOE_TWO_STREAM", "0") == "1"),'
run_test "14 bool(int(os.getenv(KEY,\"0\")))" 0 \
    '    "VLLM_MOE_TWO_STREAM": lambda: bool(int(os.getenv("VLLM_MOE_TWO_STREAM", "0"))),'
run_test "15 == \"1\" comparison must not read as = \"1\" default" 0 \
    '    "VLLM_OP001": lambda: bool(os.getenv("VLLM_OP001", "0") == "1"),'
run_test "16 = 0" 0 'VLLM_MOE_TWO_STREAM: bool = 0'

echo ""; echo "== ALLOW: allowlisted infra names and platform prefixes =="
run_test "17 VLLM_USE_DEEP_GEMM (upstream, defaults on)" 0 \
    '    "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),'
run_test "18 VLLM_USE_V1" 0 '    "VLLM_USE_V1": lambda: bool(int(os.getenv("VLLM_USE_V1", "1"))),'
run_test "19 VLLM_SKIP_P2P_CHECK" 0 \
    '    "VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",'
run_test "20 VLLM_USE_PRECOMPILED (provisioning)" 0 'VLLM_USE_PRECOMPILED = 1'
run_test "21 VLLM_ROCM_FP8_PADDING (ROCm prefix)" 0 \
    '    "VLLM_ROCM_FP8_PADDING": lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),'
run_test "22 VLLM_CPU_INT4_W4A8 (CPU prefix)" 0 \
    '    "VLLM_CPU_INT4_W4A8": lambda: bool(int(os.getenv("VLLM_CPU_INT4_W4A8", "1"))),'
run_test "23 VLLM_XPU_ENABLE_XPU_GRAPH (XPU prefix)" 0 'VLLM_XPU_ENABLE_XPU_GRAPH: bool = True'
run_test "24 VLLM_RAY_DP_PACK_STRATEGY (RAY prefix)" 0 'VLLM_RAY_DP_PACK_STRATEGY = 1'

echo ""; echo "== ALLOW: comments, scope, empty =="
run_test "25 whole-line comment" 0 '# VLLM_MOE_TWO_STREAM: bool = True  # turned off below'
run_test "26 indented comment" 0 '    # VLLM_MOE_TWO_STREAM: bool = True'
run_test "27 trailing comment beside a default-off line" 0 \
    'VLLM_MOE_TWO_STREAM: bool = False  # was VLLM_MOE_TWO_STREAM = True'
run_test "28 non-envs.py file" 0 'VLLM_MOE_TWO_STREAM = True' "/workspace/vllm/vllm/config.py"
run_test "29 config.py named envs_py is not envs.py" 0 \
    'VLLM_MOE_TWO_STREAM = True' "/workspace/vllm/vllm/my_envs_py.py"
run_test "30 empty new_string (deletion)" 0 ''
run_test "31 short name below the {3,} floor" 0 'VLLM_X = True'

echo ""; echo "== Fail-open: jq unavailable =="
TOTAL=$((TOTAL + 1))
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env printf grep sed sort dirname basename test rm; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
rc=0
echo '{"tool_name":"Edit","tool_input":{"file_path":"/w/vllm/vllm/envs.py","new_string":"VLLM_MOE_TWO_STREAM: bool = True"}}' \
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
