# E2E Latency Benchmark Guide (vLLM) for Kernel Optimizations

Read this before you measure any end-to-end latency number. It tells you how to run the vLLM E2E benchmarks and how to read what they print. Use them to show that a kernel optimization improves *real* inference latency under **production parity**:
- CUDA graphs enabled (or the exact mode used in production)
- torch.compile enabled (or the exact mode used in production)
- same TP/EP topology and serving knobs

Default gates and required reporting are not in this file. They live in `references/validation-defaults.md`.

## Contents
- Tool Selection (which invocation each stage uses)
- Quickstart (baseline vs optimized)
- Workload selection (decode-heavy vs prefill-heavy)
- Batch-size sweep
- Parity checklist (must match baseline)
- Interpreting output + speedup math
- Troubleshooting
- Recording results

## Tool Selection

Pick the invocation that matches your stage.

| Stage | Tool | Why |
|-------|------|-----|
| Stage 1 (baseline) | `run_vllm_bench_latency_sweep.py --round {N} --slot baseline --labels baseline --capture-golden-refs` | Clean E2E timing — **no profiling flags allowed** (guard enforced). Authoritative numbers for speedup calculations. |
| Stage 1 (profiling) | `run_vllm_bench_latency_sweep.py --round {N} --slot profiling --nsys-profile --nsys-mode node --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800` | Bounded selected-step nsys attribution capture. The sweep resolves `--nsys-trace` from `target.hardware` (`cuda-sw` on Blackwell). E2E numbers here are profiler-contaminated and not used for comparisons. See `nsys-profiling-guide.md`. |
| Stage 5 (validation) | `run_vllm_bench_latency_sweep.py --round {N} --slot opt/{op_id} --labels opt --baseline-from $STAGE1_DIR --fresh-cache` | GPU-locked optimized-only sweep compared against the Stage 1 baseline (never re-run a baseline from the worktree — § E2E Baseline Reuse in `validation-defaults.md`) |
| Stage 6 (integration) | `run_vllm_bench_latency_sweep.py --round {N} --slot integration --labels opt --baseline-from $STAGE1_DIR --fresh-cache` (round >= 2 adds `--baseline-from-arm opt`) | Gate-quality combined measurement vs the Stage 1 baseline when two or more tracks pass; a single unchanged passer uses the short-circuit. EXHAUSTED rounds skip it. Round >= 2 MUST import the promoted arm, or the delta is cumulative, not incremental — see § Combined Validation Workflow in `orchestration/integration-logic.md`. |

For all measurements reported in `validation_results.md` or used for profiling, use the sweep script.

## Quickstart

Run the sweep script from the worktree root, and always pass the artifact directory explicitly:

```bash
# Stage 1a: Clean E2E baseline (no profiling — authoritative timing)
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round 1 --slot baseline --labels baseline \
  --capture-golden-refs

# Stage 1b: Profiling traces (separate invocation — overhead doesn't pollute baseline)
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round 1 --slot profiling --labels baseline \
  --nsys-profile --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 --nsys-timeout-s 1800

# Stage 5: Per-track validation sweep (opt-only vs Stage 1 baseline, isolated cache)
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {N} --slot opt/{op_id} \
  --labels opt --baseline-from {artifact_dir}/rounds/{N}/sweeps/baseline --fresh-cache

# Stage 6: Integration sweep (--fresh-cache for gate-quality measurement)
# This replaces the former T16 re-profile step.
# Round >= 2 MUST append `--baseline-from-arm opt` to the command below: the
# comparator is then the promoted arm, not that directory's stale pre-SHIP
# baseline arm. Without it the measured delta is cumulative, not incremental.
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {N} --slot integration \
  --labels opt --baseline-from {artifact_dir}/rounds/{N}/sweeps/baseline --fresh-cache

# Post-SHIP: Golden-refs capture after environment promotion
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {N} --slot golden_capture \
  --labels baseline --capture-golden-refs
```

Do not put the workload on the command line. The sweep script reads model, workload, and env config from `target.json`, so you never need `--model`, `--dtype`, `--batch-size`, or the rest.

### Fresh-cache isolation (`--fresh-cache`, v3.1)

This flag keeps a sweep from inheriting warm compile caches
from a previous sweep. It allocates `{out_root}/cache/{sweep_id}/`
and injects `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` into the child
env. The cache is removed on success. First launch of N pays full
compile (~5 min for large models); launches 2..N hit the warm
in-sweep cache.

## Workload selection

### Decode-heavy (recommended for decode-bucket optimizations)

Benchmark decode-heavy when the optimization targets **decode buckets** (small `M` per step). Many optimization fast-paths are tuned for those buckets, and a decode-heavy benchmark makes that visible:

- `--input-len 64` (short prefill)
- `--output-len 512` (long decode)
- Sweep your decode bucket `--batch-size` set

### Prefill-heavy (optional)

Run a second, prefill-heavy benchmark if you claim a prefill win — large input length, and usually a smaller output length. Keep those results separate from the decode-heavy results.

## Batch-size sweep

Sweep the **same bucket set** you profiled in Stage 1 and plan to enable in Stage 6.

Let the sweep script do the multi-bucket runs; it loads the model once per label. Pass `--round {N} --slot {SLOT}` so results land in the canonical round-scoped layout:

```bash
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {CR} --slot baseline
```

`{SLOT}` is one of: `baseline` (Stage 1), `opt/{op_id}` (Stage 5 per-track), `integration` (Stage 6 combined sweep), `golden_capture` (post-SHIP golden-ref refresh).

The script takes the workload config from `target.json`. Both formats work: the flat
format (`input_len`, `output_len`, `batch_sizes`), and `workload_matrix` for
multi-dimensional `(input_len x output_len x batch_size)` sweeps.

Per-bucket nsys profiles need a **separate short profiling invocation** (`--slot profiling`), run after the clean baseline. `--slot baseline` plus any profiler flag is a hard error, so profiling overhead can never contaminate authoritative timing. The E2E numbers from a profiling invocation are themselves contaminated by nsys overhead and are NOT authoritative. For the exact command, the architecture trace backend (Blackwell `cuda-sw`), selected-step capture semantics, and the bounded `--nsys-output-len 2` fallback, see `nsys-profiling-guide.md § Required Stage 1 Commands`.

## Parity checklist (must match baseline)

Your numbers are not trustworthy until every item below holds.

- Same model weights + same revision/commit.
- Same dtype/quantization (FP8 formats and scale shapes matter).
- Same TP/EP topology and identical routing/dispatch mode.
- Same CUDA graphs mode and torch.compile mode.
- Same scheduler knobs that affect bucketing (e.g., max batched tokens / chunked prefill).
- Confirm **optimized path actually executed**, with one of:
  - enablement log line, or
  - instrumentation counter, or
  - an unmistakable kernel name in Nsight Systems.

### Debug-only: run eager

Use eager mode only to debug correctness or functional issues. Never use it for a production-parity perf claim.

```bash
vllm bench latency --enforce-eager ...
```

## Interpreting output

`vllm bench latency` prints iteration latencies. The value you want is usually **Avg latency**.

Compute speedup and improvement:

```python
baseline_s = 10.95
opt_s = 10.20

speedup = baseline_s / opt_s
improvement_pct = (baseline_s - opt_s) / baseline_s * 100
```

If your measured E2E improvement is small, sanity-check the component share with `references/e2e-delta-math.md`.

## Troubleshooting

### Optimized path not activating
- Verify the enable flag is set (env var / config).
- Verify a compiled specialization exists for your `(dtype, TP/EP, bucket set)`.
- Verify your bucket guard matches the validated envelope.
- Use Nsight Systems to confirm which kernels run under the captured graph.

### "No E2E win" even though microbench is faster
Common causes:
- The target component is a small fraction of end-to-end (`f` small), so the expected E2E gain is bounded (see `references/e2e-delta-math.md`).
- Graph breaks or unexpected fallbacks (different kernels between baseline and optimized runs).
- Another bottleneck dominates (attention/KV/cache/scheduler).

### High variance / poor reproducibility
- Increase iterations.
- Isolate the GPU: no other jobs, and no power or thermal limit.
- Use a consistent warmup protocol.

## Recording results

Write the Stage 5 per-track results to `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validation_results.md` with:
- full repro commands (baseline + optimized) and env vars
- the bucket set and capture/compile settings
- baseline vs optimized tables (speedup + improvement)
- evidence that the optimized path executed

The Stage 6 integration record is `{artifact_dir}/rounds/{CR}/sweeps/integration/e2e_latency_results.json`, written by the sweep script. `references/artifact-layout.md` owns every artifact path.

Use `references/validation-defaults.md` for the minimum reporting template and default gates.
