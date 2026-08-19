# Nsight Systems Profiling Guide

This guide owns how AMMO captures GPU traces and how it reads them. Read it before you run a profiler in Stage 1, and again before you write a bottleneck claim in Stage 2. It gives the exact commands, the capture semantics, and the list of what counts as valid evidence.

## The Default Workflow

Stage 2 has one default profiling workflow:

1. Run a clean E2E baseline sweep with no profiler flags.
2. Run a short Nsight Systems node capture. The sweep resolves the CUDA trace backend for your hardware.
3. Mine the nsys report for kernel ranking, launch chains, per-rank/per-device skew, CUDA memcpy/NVLink activity, and timing shares.
4. Run targeted Nsight Compute only for physical-ceiling claims such as occupancy, achieved bandwidth counters, or SM utilization.

Do not use profiler-contaminated latency as the official E2E baseline.

## Required Stage 1 Commands

Run two separate sweeps, in this order.

Clean baseline, used for all speedup math:

```bash
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot baseline \
  --labels baseline \
  --capture-golden-refs
```

Bounded selected-step nsys capture, used for bottleneck attribution on every platform:

```bash
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot profiling \
  --labels baseline \
  --nsys-profile \
  --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 \
  --nsys-timeout-s 1800
```

`--slot baseline` plus any profiler flag is invalid; the sweep script enforces this guard.

The trace backend is resolved by the sweep script — see § Trace Backend. Do not pass `--nsys-trace` unless you must override it.

Keep `--nsys-mode node` for the default Stage 2 path because node mode is the ranking source for CUDA graph workloads.

Outputs:

```text
{artifact_dir}/rounds/{N}/sweeps/baseline/e2e_latency_results.json
{artifact_dir}/rounds/{N}/profiling/nsys/*.nsys-rep
```

This produces one `.nsys-rep` per selected depth and batch bucket. The trace
sidecar records both the synthetic capture shape and source shape:
`source_input_len`, `source_output_len`, `capture_output_step`,
`capture_window_output_len`, and `capture_target_output_len`.

Use the bounded fallback `--nsys-output-len 2 --nsys-num-iters 1` only if
selected-step capture fails; document that depth coverage was omitted.

## Selected-Step Capture

`--nsys-capture-output-steps` accepts comma-separated integers and percentages.
`--nsys-capture-output-steps 2,50%,100%` resolves against each workload bucket's
`output_len`. Percentages resolve against `--nsys-output-len` when supplied,
otherwise against each bucket's workload `output_len`. `--nsys-output-len`
remains a horizon override for percentage resolution. Duplicate resolved steps
are removed in stable order.

The sweep then profiles short shape-equivalent windows that land on a genuine full-batch decode step, not a chunked-prefill chunk. For target step `k`, the sweep captures decode depth `input_len + k` (invariant of the window) by running `input_len + k - w_eff`, `output_len = w_eff`, and vLLM's CUDA profiler captures only the final worker step.

`w_eff` is an effective capture window that the script floors child-wide to clear chunked prefill: `w_eff = max(--nsys-capture-window-output-len, max over (bucket, step) of ceil((input_len + step - requested_window) * batch_size / 16384) + 6)`, where `requested_window = --nsys-capture-window-output-len`. The `ceil(...)` term is the number of chunked-prefill worker-steps (chunk = 16384 tokens) the capture must arm past, evaluated at the shifted prompt length `input_len + step - requested_window` (longest at the deepest requested step).

`--nsys-capture-window-output-len` (default 2) is therefore only a LOWER BOUND — the script auto-raises it as needed (e.g. 2 → 11 for an 8192-token prompt at batch 8, where the deepest step 512 gives `ceil((8192+512-2)*8/16384)+6 = 5+6 = 11`) and you cannot force the effective window below the floor. You normally do not set it.

A step shallower than the effective capture window is still captured — the
script shifts `input_len` (`il_eff = input_len + step - w_eff`) so the trace
lands at the requested decode depth. The only step that is dropped is one where
`il_eff < 1` (the context is too short to host a steady-state decode); that
bucket/step logs a loud WARNING and the remaining steps continue. To recover a
dropped step, use a larger `input_len` or a shallower step — lowering
`--nsys-capture-window-output-len` will not help, because it is only a lower
bound that the script auto-raises to clear chunked prefill.

## Trace Backend

`_default_nsys_trace()` in the sweep script owns this rule: `--nsys-trace` defaults to `cuda-sw` when `target.hardware` names Blackwell (B200/B300/GB200/GB300, SM100/SM120), else `cuda`. The sweep prints the resolved value and its source. The script is the only place this rule lives; there is no hand-maintained hardware matrix to read or keep current.

On Blackwell, the hardware event system for CUDA tracing interacts poorly with CUDA graph replay, causing long warmup/profiling stalls followed by RPC or NCCL watchdog timeouts. This affects all workloads (not just MoE or TP>1). `cuda-sw` still captures CUDA API/software activity and graph node attribution while avoiding that hardware-tracing path. On a non-Blackwell target, override to `cuda-sw` only after logs match that replay/collective timeout failure.

## When a Capture Fails

If the command fails:

- Check the supervisor log under `rounds/{N}/sweeps/profiling/logs/`.
- Check the printed `nsys_trace:` line. It reports the resolved backend and whether it came from `target.hardware` or an explicit flag.
- Use the bounded fallback (`--nsys-output-len 2 --nsys-num-iters 1`) only when selected-step capture itself fails — see § Required Stage 1 Commands.
- Increase `--nsys-timeout-s` only if logs show useful forward progress.
- Reduce the profiled bucket set only as a last resort, and document the omitted buckets.

## Stage 2 Mining

Mine the existing nsys reports. Do not re-run the sweep just to analyze traces.

Minimum analysis:

- Export nsys reports to SQLite or stats tables.
- Rank kernels by total GPU time and count.
- Group kernel chains in timestamp order, not architecture order.
- Map top kernels to source paths or generated backends when possible.
- For TP > 1, compare all rank/device reports and report per-rank skew.
- Separate compute kernels from communication kernels and memcpy/P2P traffic.
- Compute `f_decode`, `decode_share_of_e2e`, and `f_e2e` using measured Stage 1 data.

Useful commands:

```bash
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep

nsys stats --force-export=true --report cuda_gpu_trace \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output {artifact_dir}/rounds/{N}/profiling/nsys/baseline.sqlite \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep
```

Report approximate trace timings honestly. A value such as `~74 us` is valid when it comes from the nsys trace.

## Optional Graph Diagnostics

`--nsys-mode graph` is optional diagnostic enrichment. Use it only when node-mode results leave a specific open question about graph structure, launch grouping, CUDA graph replay, or metadata available only in graph view.

If node and graph mode disagree, use node-mode timing for bottleneck ranking (§ Required Stage 1 Commands) and explain why graph mode was captured.

Graph diagnostic command:

```bash
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot profiling \
  --labels baseline \
  --nsys-profile \
  --nsys-mode graph \
  --nsys-output-len 2 \
  --nsys-num-iters 1 \
  --nsys-timeout-s 1800
```

Graph diagnostics use the same hardware-derived trace backend as node mode.

## Targeted NCU

The physical-ceiling gate is disjunctive: back the claim with targeted Nsight Compute, or with an explicitly cited hardware spec plus the math. Either one alone is enough. Nsys alone is not.

These count as physical-ceiling claims:

- Occupancy or achieved occupancy.
- SM utilization or tensor-core utilization.
- Achieved memory bandwidth from hardware counters.
- Register pressure, shared-memory pressure, or stall-reason claims.
- "Kernel X can improve by at most Y%" where Y comes from a hardware ceiling.

Keep NCU narrow. Profile only the top kernels identified by nsys, with representative bucket sizes. Store results under:

```text
{artifact_dir}/rounds/{N}/profiling/ncu/
```

Stage 2 may rank bottlenecks from nsys without NCU.

## Valid Stage 2 Evidence

Acceptable:

- `rounds/{N}/profiling/nsys/*.nsys-rep`
- nsys stats/export tables derived from those reports
- sweep JSON from `rounds/{N}/sweeps/baseline/`
- targeted NCU CSV/report for hardware-counter claims
- source-code mapping used only to explain what a measured kernel is

Not acceptable:

- Profiler-run latency as official E2E timing.
- Architecture-inferred kernel chains without trace timestamps.
- Occupancy/bandwidth-counter claims without targeted NCU.
- Graph-mode timing as the primary bottleneck ranking source.

## Report Checklist

`rounds/{N}/mining/bottleneck_analysis.md` must include:

- Exact Stage 1 baseline and profiling commands.
- Artifact paths for every trace used.
- Top kernels/components by measured time.
- Per-rank/per-device comparison for TP/DP runs.
- `f_decode`, `decode_share_of_e2e`, and `f_e2e` tables.
- Technology Landscape entries for the top components.
- Any NCU-backed physical-ceiling claims with paths to the raw NCU artifacts.
