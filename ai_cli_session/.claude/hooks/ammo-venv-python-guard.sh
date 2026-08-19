#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — Block bare `python` / `python3` invocations of AMMO sweep
# scripts. The session .venv has vLLM editable-installed; system python does not.
# Only targets sweep/profiling scripts that import vllm — NOT management scripts
# like gpu_reservation.py, new_target.py which don't need vllm.
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Only fire on commands invoking vllm-dependent sweep/profiling scripts.
# These are the scripts that import vllm and MUST run under .venv/bin/python.
# The script name is matched with any path prefix, or none — a bare
# `python ncu_sanity_driver.py` from the scripts dir is the same mistake as the
# full-path form. This mirrors the Codex guard (_SWEEP_SCRIPTS_RE).
SWEEP_SCRIPTS="(?:run_vllm_bench_latency_sweep|ncu_sanity_driver)\.py"
# Interpreter alternation matches versioned forms (python3.12) too — keep it
# identical to the Codex twin (_SWEEP_SCRIPTS_RE in pre_tool_use_guard.py).
PYTHON_RE="python(?:\d+(?:\.\d+)?)?"
if ! echo "$COMMAND" | grep -qP "${PYTHON_RE}\s+\S*${SWEEP_SCRIPTS}"; then
    exit 0
fi

# Allow: .venv/bin/python prefix (relative or absolute)
if echo "$COMMAND" | grep -qP "\.venv/bin/${PYTHON_RE}\s+\S*${SWEEP_SCRIPTS}"; then
    exit 0
fi

# Allow: source .venv/bin/activate && ... earlier in the command
if echo "$COMMAND" | grep -qP 'source\s+\S*\.venv/bin/activate\s*&&'; then
    exit 0
fi

cat >&2 <<'EOF'
BLOCKED: Use .venv/bin/python for sweep scripts (system python has no vllm)

Fix — use:
    .venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py ...
EOF
exit 2
