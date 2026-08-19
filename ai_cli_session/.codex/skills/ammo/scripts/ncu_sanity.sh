#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Optional claim-driven targeted NCU probe — fast 3-counter capture, bounded runtime.
# Run only when a specific unresolved hardware-counter claim cannot be answered
# from the existing nsys trace, roofline math, or hardware specifications.
#
# Why this shape:
#
#   `ncu --set <X>` expands to 15-20 metric groups, each requiring one
#   application-replay pass under CUDA graphs. Combined with vLLM's
#   3-5 min cold start per pass, the default `--set` easily blows past
#   30 minutes of wall-clock for a sanity check that only needs to answer
#   "is SM utilization absurdly low? is DRAM BW way off peak? is the
#   instruction count right-shaped?". Three HW counters cover that.
#
#   `--replay-mode application` is required under CUDA graphs — kernel
#   replay can't re-launch individual kernels from inside a captured
#   graph. Application replay re-runs the whole ncu_sanity_driver.py,
#   which is exactly what the driver is designed for: a tiny warmup +
#   bounded iter budget keeps each pass cheap.
#
#   `--launch-count 30` caps per-pass captured launches. With the driver's
#   default warmup=3 iters=10, and a moderate decoder-layer count, this
#   lands well within the bounded budget without forcing ncu to record
#   hundreds of launches per pass.
#
#   `--cache-control none --clock-control none` avoid forcing clock/cache
#   state changes that slow each pass and aren't useful for a sanity check.
#
#   `--profile-from-start off` defers capture until the driver calls
#   cudaProfilerStart(), keeping the cold-start + warmup launches off the
#   captured count entirely.
#
#   `timeout --kill-after=10 600` is the last-ditch guard: if something
#   goes wrong (driver hangs, ncu lockup, CUDA-graph edge case) the whole
#   capture dies at 10 minutes instead of eating the session's GPU budget.
#
# Required env:
#   CSV_OUT     — path to write ncu CSV output
#   TARGET_JSON — path to target.json for the session's model
#
# Kernel filters are passed as positional args after the env vars. The wrapper
# joins them into a single `--kernel-name regex:<a>|<b>|<c>` expression — ncu
# 2025.4+ rejects repeated `--kernel-name` flags with "option cannot be
# specified more than once", so a single regex alternation is the portable
# form. Args are treated as regex alternatives; prefer plain alphanumeric +
# underscore substrings (CUTLASS mangled names with `<`, `>`, `(`, `)` must be
# regex-escaped by the caller if matched literally):
#
#   CSV_OUT=ncu/sanity.csv TARGET_JSON=target.json BATCH_SIZE=8 \
#     ./ncu_sanity.sh s161616gemm flash_fwd rsqrt

set -euo pipefail

: "${CSV_OUT:?CSV_OUT env var required (output path for ncu CSV)}"
: "${TARGET_JSON:?TARGET_JSON env var required (path to target.json)}"
: "${BATCH_SIZE:?BATCH_SIZE env var required (decode bucket to capture)}"

# Default tunables — override via env if needed.
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
LAUNCH_COUNT="${LAUNCH_COUNT:-30}"
NCU_TIMEOUT_S="${NCU_TIMEOUT_S:-600}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"  # short OL keeps each pass cheap; fine for counters

# The driver imports vllm, so it must run under the session worktree venv.
# Prefer ./.venv/bin/python; fall back to `python` when no venv is present.
if [ -x "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"
else
    PY="python"
fi

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <kernel_substring> [<kernel_substring> ...]" >&2
    exit 2
fi

# ncu 2025.4+ rejects repeated `--kernel-name`. Join all positional args into
# a single regex alternation. Args must be regex-safe — this is fine for plain
# alphanumeric substrings (the usual case). If a caller needs literal matching
# of CUTLASS metachars, escape them before passing in.
KERNEL_REGEX="$1"
shift
for k in "$@"; do
    KERNEL_REGEX+="|$k"
done
KERNEL_FLAGS=(--kernel-name "regex:${KERNEL_REGEX}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/ncu_sanity_driver.py"

mkdir -p "$(dirname "$CSV_OUT")"

# Three HW counters: SM throughput %, DRAM throughput %, kernel duration.
# These provide bounded evidence for an explicitly named utilization/bandwidth
# claim and cost one application-replay pass each (3 total).
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum"

echo "[ncu_sanity] CSV_OUT=$CSV_OUT"
echo "[ncu_sanity] target=$TARGET_JSON bs=$BATCH_SIZE warmup=$WARMUP iters=$ITERS"
echo "[ncu_sanity] filters: regex:${KERNEL_REGEX}"
echo "[ncu_sanity] launch_count=$LAUNCH_COUNT timeout=${NCU_TIMEOUT_S}s"

timeout --kill-after=10 "$NCU_TIMEOUT_S" \
    ncu --replay-mode application \
        --set none \
        --metrics "$METRICS" \
        --cache-control none \
        --clock-control none \
        --profile-from-start off \
        --launch-count "$LAUNCH_COUNT" \
        --target-processes all \
        --csv --log-file "$CSV_OUT" \
        "${KERNEL_FLAGS[@]}" \
        -- \
        "$PY" "$DRIVER" \
            --target-json "$TARGET_JSON" \
            --batch-size "$BATCH_SIZE" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --output-len "$OUTPUT_LEN"

echo "[ncu_sanity] done → $CSV_OUT"
