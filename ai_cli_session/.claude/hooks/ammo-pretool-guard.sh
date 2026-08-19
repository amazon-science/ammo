#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — AMMO campaign production-parity reminders and GPU pool guard.
#
# N1/N4 warnings: Warns (but does NOT block) when a Bash command contains
# patterns that violate AMMO non-negotiables.
#
# GPU pool guard: Blocks ONCE per session if a GPU command lacks the
# reservation pattern (CVD=$(python gpu_reservation.py reserve ...)).
# Subsequent commands without the pattern are allowed through.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
# Try both field names (Claude Code docs ambiguous: tool_input vs input)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Determine if an AMMO campaign is active
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
_CAMPAIGN_ACTIVE=false
ls "$PROJECT_DIR"/kernel_opt_artifacts/*/state.json &>/dev/null && _CAMPAIGN_ACTIVE=true

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY="$HOOK_DIR/../skills/ammo/scripts/hook_cmd_classify.py"

# _is_inspection_only — true only when EVERY segment of $COMMAND is
# inspection-class. The old test was an ANCHORED regex on the start of the
# command string, so a single inspection prefix disabled the rest of this hook:
# `cat notes.md && vllm bench latency ...` took the fast path, silencing both
# hard blocks below (worktree venv + GPU pool). One python3 exec per hook call.
#
# Fail-CLOSED on a missing classifier or interpreter: no fast path, so the
# command falls through to the guard logic. That is the safe direction — the
# guards themselves stay one-shot and advisory-first.
_is_inspection_only() {
    [ -f "$CLASSIFY" ] || return 1
    command -v "$CLASSIFY_PY" &>/dev/null || return 1
    "$CLASSIFY_PY" "$CLASSIFY" --mode readonly "$1" >/dev/null 2>&1
}

if [ -x "$HOOK_DIR/../../.venv/bin/python" ]; then
    CLASSIFY_PY="$HOOK_DIR/../../.venv/bin/python"
else
    CLASSIFY_PY="python3"
fi

if [ "$_CAMPAIGN_ACTIVE" = "true" ]; then
    # Suppress noisy warnings on read-only / inspection commands
    if _is_inspection_only "$COMMAND"; then
        exit 0
    fi

    # N1: Production parity reminders
    if echo "$COMMAND" | grep -qP 'TORCH_COMPILE_DISABLE\s*=\s*1'; then
        echo "AMMO REMINDER: TORCH_COMPILE_DISABLE=1 detected. AMMO non-negotiable N1 requires production parity (CUDA graphs + torch.compile). If this is a false positive (e.g., documentation or search), ignore this warning." >&2
    fi
    if echo "$COMMAND" | grep -qP '(--|=)enforce[_-]eager'; then
        echo "AMMO REMINDER: --enforce-eager detected. AMMO non-negotiable N1 requires CUDA graphs to be enabled. If this is a false positive (e.g., documentation or search), ignore this warning." >&2
    fi
    if echo "$COMMAND" | grep -qP 'VLLM_TORCH_COMPILE_LEVEL\s*=\s*[01](\s|$|")'; then
        echo "AMMO REMINDER: VLLM_TORCH_COMPILE_LEVEL < 2 detected. AMMO non-negotiable N1 requires torch.compile level ≥ 2. If this is a false positive (e.g., documentation or search), ignore this warning." >&2
    fi

    # N4: Sweep script mandate reminder
    if echo "$COMMAND" | grep -qP 'vllm\s+bench\s+latency' && \
       ! echo "$COMMAND" | grep -q 'run_vllm_bench_latency_sweep'; then
        echo "AMMO REMINDER: Raw 'vllm bench latency' detected. AMMO non-negotiable N4 requires using the sweep script:" >&2
        echo "  python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir>" >&2
        echo "If this is a false positive (e.g., documentation or search), ignore this warning." >&2
    fi

    # N5: VLLM_OP* env contamination warning (warn-only, not blocking)
    # Catches ad-hoc commands like `VLLM_OP001=1 python benchmark.py`.
    # The sweep script's _sanitize_vllm_op_env() is the hard gate.
    if echo "$COMMAND" | grep -qP 'VLLM_OP\d+=\s*(1|True|true)\b' && \
       ! echo "$COMMAND" | grep -q 'run_vllm_bench_latency_sweep'; then
        echo "AMMO REMINDER: VLLM_OP* env var set to True/1 detected in command." >&2
        echo "  Cross-track contamination prevention: VLLM_OP* flags must default to off." >&2
        echo "  The sweep script sanitizes VLLM_OP* from inherited env automatically." >&2
        echo "  If setting for a specific benchmark, use opt_env in target.json instead." >&2
    fi
fi

# ── Worktree venv activation guard ──
# When the agent's cwd is inside .claude/worktrees/<name>/, any python/pip/pytest
# invocation MUST run under that worktree's .venv. The main repo's .venv has an
# editable-install .pth that resolves `import vllm` to the MAIN worktree's
# source tree — silently invalidating all profiling/benchmarks for the
# isolated op worktree. One-shot block per session; trust agent judgment after.
# Runs before the GPU-pool guard so a venv violation on a GPU command surfaces
# the more fundamental issue (wrong source tree) rather than being masked.
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
case "$CWD" in
    */.claude/worktrees/*)
        WORKTREE_ROOT=$(echo "$CWD" | sed -E 's#(.*/\.claude/worktrees/[^/]+).*#\1#')
        EXPECTED_VENV="$WORKTREE_ROOT/.venv"

        if echo "$COMMAND" | grep -qP '\b(python3?|pytest|pip|uv)\b'; then
            _VENV_OK=false
            echo "$COMMAND" | grep -qF "source $EXPECTED_VENV/bin/activate" && _VENV_OK=true
            echo "$COMMAND" | grep -qF "$EXPECTED_VENV/bin/python"          && _VENV_OK=true
            echo "$COMMAND" | grep -qF "$EXPECTED_VENV/bin/pytest"          && _VENV_OK=true
            echo "$COMMAND" | grep -qF "$EXPECTED_VENV/bin/pip"             && _VENV_OK=true
            # Exempt: bare `import vllm` check (used to verify activation worked)
            if echo "$COMMAND" | grep -qP '^\s*python3?\s+-c\s+["\x27]import\s+vllm["\x27;\s]*(print\(vllm\.__file__\))?\s*["\x27]?\s*$'; then
                _VENV_OK=true
            fi

            _SID_RAW="${CLAUDE_SESSION_ID:-}"
            if [ "$_VENV_OK" = "false" ] && [ -n "$_SID_RAW" ]; then
                _GRD="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
                mkdir -p "$_GRD"
                WARNED_VENV_FLAG="$_GRD/.warned_venv_${_SID_RAW}"
                if [ ! -f "$WARNED_VENV_FLAG" ]; then
                    touch "$WARNED_VENV_FLAG"
                    cat >&2 <<EOF
AMMO WORKTREE VENV: Python command in op worktree without activating the worktree-local venv.

You are in: $CWD
Expected venv: $EXPECTED_VENV

The main repo's .venv has an editable-install .pth that resolves 'import vllm'
to the MAIN worktree's source tree — NOT your isolated edits. Running profiling
or benchmarks under the wrong venv silently invalidates the results.

Fix: prefix your command with
  source $EXPECTED_VENV/bin/activate &&

Or invoke python by absolute path: $EXPECTED_VENV/bin/python ...

Verify once after activation:
  python -c "import vllm; print(vllm.__file__)"
  # The path MUST contain '/.claude/worktrees/'

(This block fires only once per session.)
EOF
                    exit 2
                fi
            fi
        fi
        ;;
esac

# ── GPU Pool Pattern Guard ──
# Detect GPU-heavy commands that DON'T use the reservation pattern.
# One-shot warning per session (same mechanism as ammo-stop-guard.sh).

# Detect if this command is likely GPU-heavy (conservative patterns)
IS_GPU_CMD=false
if echo "$COMMAND" | grep -qP '\b(nsys|ncu)\b' || \
   echo "$COMMAND" | grep -qP 'nvidia-smi\s+--query-compute'; then
    IS_GPU_CMD=true
elif echo "$COMMAND" | grep -qP '\b(vllm|torchrun)\b'; then
    IS_GPU_CMD=true
elif echo "$COMMAND" | grep -qP '\b(python3?|pytest)\b' && \
     echo "$COMMAND" | grep -qiP '(torch|cuda|triton|vllm|benchmark|kernel|gpu)'; then
    # Exemption: bare import checks
    if echo "$COMMAND" | grep -qP '^\s*python3?\s+-c\s+["\x27]import\s+(vllm|torch)["\x27]\s*$'; then
        IS_GPU_CMD=false
    else
        IS_GPU_CMD=true
    fi
fi

if [ "$IS_GPU_CMD" = "false" ]; then
    exit 0
fi

# If command uses the reservation pattern, allow through
if echo "$COMMAND" | grep -q 'gpu_reservation.py reserve'; then
    exit 0
fi

# Explicit CUDA_VISIBLE_DEVICES=<digits> WITHOUT reservation is a bypass —
# fall through to the one-shot warning so agents learn to use the reservation pattern.

# If command has CUDA_VISIBLE_DEVICES="" (explicit no-GPU), allow through
if echo "$COMMAND" | grep -qP "CUDA_VISIBLE_DEVICES=(\"\"|\x27\x27)"; then
    exit 0
fi

# GPU command without reservation pattern — one-shot warning
SESSION_ID="${CLAUDE_SESSION_ID:-}"
if [ -z "$SESSION_ID" ]; then
    exit 0  # No session ID — fail-open
fi

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
WARNED_FLAG="${GPU_RES_DIR}/.warned_${SESSION_ID}"
mkdir -p "$GPU_RES_DIR"

if [ ! -f "$WARNED_FLAG" ]; then
    touch "$WARNED_FLAG"
    cat >&2 <<EOF
AMMO GPU POOL: GPU command detected without reservation.

Use the GPU pool pattern to acquire GPUs before running GPU commands:

  CVD=\$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=\$CVD <your_command>

Or set CUDA_VISIBLE_DEVICES="" if no GPU is needed.
(This warning fires only once per session.)
EOF
    exit 2
else
    exit 0  # Already warned — trust agent judgment
fi

exit 0
