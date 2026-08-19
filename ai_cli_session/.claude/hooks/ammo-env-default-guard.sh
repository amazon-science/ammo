#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — AMMO env var default-off enforcement.
#
# Fires on Edit and Write tool calls. Blocks edits to envs.py that register a
# campaign VLLM_* gating flag with a truthy default. This prevents cross-track
# contamination where a prior round's gating flag silently activates stale
# optimizations during subsequent sweeps.
#
# Convention: a campaign gating flag must default to False/0/"0". The sweep
# harness enables it explicitly via opt_env in target.json.
#
# NAME-AGNOSTIC (was VLLM_OP\d+ only). impl-track-rules.md § Env Flag Naming
# (PR-Ready) FORBIDS the VLLM_OP<n> shape and mandates descriptive
# VLLM_<SCOPE>_<MECHANISM>[_<ARCH>] names, so the old regex fired only on names
# a Stage 4-5 audit already treats as BLOCKING and missed every conforming one.
# Live campaign artifacts carry both shapes (VLLM_OP002_FUSED_GDN_EPILOGUE next
# to VLLM_QWEN3_5_FP8_MLP_GATE_UP_SM89), so the match is now
# VLLM_[A-Z0-9_]{3,} minus an explicit allowlist.
#
# ALLOWLIST — infra-legit names that are NOT campaign gating flags and DO
# legitimately default truthy. Two rules:
#   1. Platform/infra prefixes (ROCM/XPU/CPU/TPU/XLA/ZENTORCH/RAY/LOG/...): an
#      NVIDIA campaign never gates on these, and upstream defaults several on.
#   2. An explicit name list, seeded by scanning the real vLLM envs.py in the
#      session image for `getenv(KEY, "1"|True)` registrations, plus the
#      infra/provisioning keys that appear in campaign artifacts
#      (VLLM_USE_PRECOMPILED, VLLM_CACHE_ROOT, VLLM_ATTENTION_BACKEND, ...).
# Anything not on the list counts as a campaign flag. That direction is the safe
# one: a false block costs the champion one turn and names the flag, while a
# false allow silently corrupts an A/B arm.
#
# Exit 0 = allow, Exit 2 = block (stderr fed back to agent).
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Extract file path from Edit or Write tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || true
[ -z "$FILE_PATH" ] && exit 0

# Only inspect edits to envs.py files
case "$FILE_PATH" in
    */envs.py) ;;
    *) exit 0 ;;
esac

# Extract the content being written
# Edit tool: new_string field; Write tool: content field
NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty' 2>/dev/null) || true
[ -z "$NEW_CONTENT" ] && exit 0

# Non-NVIDIA platform families and the infra prefixes — never campaign flags.
ALLOW_PREFIX_RE='^VLLM_(ROCM|XPU|CPU|TPU|XLA|ZENTORCH|NEURON|HPU|CI|TEST|DOCKER|SERVER|API|LOG|LOGGING|USAGE|RAY|NIXL|MOONCAKE|PLUGIN|TRACE|TARGET|MAIN|SYSTEM|PROCESS|CONFIG|CACHE|ASSETS)_'

# Explicit infra-legit names. Seeded from the truthy-default registrations in
# the real vLLM envs.py plus the provisioning keys seen in campaign artifacts.
ALLOW_NAMES='
VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE
VLLM_ALLOW_INSECURE_SERIALIZATION
VLLM_ALLOW_LONG_MAX_MODEL_LEN
VLLM_ALLOW_RUNTIME_LORA_UPDATING
VLLM_ALLREDUCE_USE_SYMM_MEM
VLLM_ATTENTION_BACKEND
VLLM_BATCH_INVARIANT
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
VLLM_COMPILE
VLLM_CONFIGURE_LOGGING
VLLM_DISABLE_COMPILE_CACHE
VLLM_DP_RANK
VLLM_DP_SIZE
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING
VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
VLLM_ENABLE_PREGRAD_PASSES
VLLM_ENABLE_V1_MULTIPROCESSING
VLLM_ENFORCE_STRICT_TOOL_CALLING
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
VLLM_KV_CACHE_LAYOUT
VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES
VLLM_MEDIA_URL_ALLOW_REDIRECTS
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
VLLM_MLA_DISABLE
VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD
VLLM_NO_USAGE_STATS
VLLM_SKIP_P2P_CHECK
VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY
VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
VLLM_TORCH_COMPILE_LEVEL
VLLM_USE_AOT_COMPILE
VLLM_USE_BYTECODE_HOOK
VLLM_USE_DEEP_GEMM
VLLM_USE_DEEP_GEMM_E8M0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES
VLLM_USE_FUSED_MOE_GROUPED_TOPK
VLLM_USE_LAYERNAME
VLLM_USE_PRECOMPILED
VLLM_USE_STANDALONE_COMPILE
VLLM_USE_V1
VLLM_WORKER_MULTIPROC_METHOD
'

# One truthy-default registration per matched LINE. Names are then extracted
# from that line and filtered by the allowlist, so the block message can say
# WHICH flag is at fault. The last alternative is the getenv/environ.get idiom
# the Codex twin already carried.
#
# `(?<![=!<>])=(?!=)` restricts the assignment alternatives to a SINGLE `=`:
# the canonical default-OFF spelling `os.getenv(KEY,"0") == "1"` ends in `== "1"`
# and a bare `=\s*"1"` matched it, blocking the correct code.
_ASSIGN='(?<![=!<>])=(?!=)'
TRUTHY_RE="(?:VLLM_[A-Z0-9_]{3,}[^\n#]*(?::\s*bool\s*)?${_ASSIGN}\s*(?:True|1\b|[\"\x27]1[\"\x27])|VLLM_[A-Z0-9_]{3,}[^\n#]*,\s*True\b|VLLM_[A-Z0-9_]{3,}[^\n#]*default\s*${_ASSIGN}\s*(?:True|1\b|[\"\x27]1[\"\x27])|default\s*${_ASSIGN}\s*(?:True|1\b|[\"\x27]1[\"\x27])[^\n#]*VLLM_[A-Z0-9_]{3,}|os\.(?:environ\.get|getenv)\(\s*[\"\x27]VLLM_[A-Z0-9_]{3,}[\"\x27]\s*,\s*[\"\x27]?(?:1|True)[\"\x27]?\s*\))"

# Strip comments BEFORE matching: a name mentioned in prose is documentation,
# not a registration, and `# VLLM_X = True` in a diff that turns a flag OFF
# must not read as turning it on. Only a `#` at line start or after whitespace
# opens a comment, so `"a#b"` inside a string literal survives.
CONTENT_NO_COMMENTS=$(printf '%s\n' "$NEW_CONTENT" | sed -E 's/(^|[[:space:]])#.*$//')

OFFENDERS=""
while IFS= read -r _line; do
    [ -z "$_line" ] && continue
    while IFS= read -r _name; do
        [ -z "$_name" ] && continue
        echo "$_name" | grep -qP "$ALLOW_PREFIX_RE" && continue
        printf '%s\n' "$ALLOW_NAMES" | grep -qxF "$_name" && continue
        case "$OFFENDERS" in
            *" $_name "*) ;;
            *) OFFENDERS="$OFFENDERS $_name " ;;
        esac
    done < <(printf '%s\n' "$_line" | grep -oP 'VLLM_[A-Z0-9_]{3,}' | sort -u)
done < <(printf '%s\n' "$CONTENT_NO_COMMENTS" | grep -P "$TRUTHY_RE" || true)

if [ -n "${OFFENDERS// /}" ]; then
    {
        echo "BLOCKED: campaign VLLM_* env var registered with a truthy default."
        echo ""
        echo "Flag(s):$OFFENDERS"
        echo ""
        cat <<'EOF'
Cross-track contamination prevention requires every campaign gating flag to
default to False (or 0/"0"). The E2E sweep harness enables them explicitly via
opt_env in target.json; a truthy default activates the path precisely when the
key is ABSENT, which is exactly the baseline arm.

Fix: change the default to False (or 0/"0"). Examples:
  VLLM_MOE_TWO_STREAM: bool = False
  "VLLM_MOE_TWO_STREAM": lambda: bool(int(os.getenv("VLLM_MOE_TWO_STREAM", "0"))),

If this flag is upstream vLLM infrastructure rather than a campaign gating flag
(and so legitimately defaults on), add its name to ALLOW_NAMES in this hook and
say why in the same edit.

See: integration-logic.md § Promotion and Finalization
     impl-track-rules.md § Env Flag Naming (PR-Ready)
EOF
    } >&2
    exit 2
fi

exit 0
