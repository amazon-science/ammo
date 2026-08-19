# E2E Delta Math (Stop‑Condition)

A big microbench speedup can still give a tiny end-to-end gain. The multiplier you put into the Amdahl formula must be the kernel's contribution to the **end‑to‑end wall clock**, not its share inside a single phase. This document defines that multiplier (`f_e2e`), shows how to derive it from sweep + profiling outputs, and gives the single boundary equation champions project with. Read it before you write any projected E2E number.

## f_e2e: The Correct Amdahl Multiplier

Let:
- `T_e2e` = total end-to-end wall time of one request (prefill + decode).
- `T_component` = wall-clock time spent in the target component **across the whole request**.
- `f_e2e = T_component / T_e2e` — the component's share of E2E wall time.
- `s` = component speedup factor (`T_component_old / T_component_new`, so `s ≥ 1` is faster).

Then the standard Amdahl form for E2E improvement is:

```text
E2E_improvement_fraction = f_e2e × (1 - 1/s)
E2E_improvement_pct = 100 × E2E_improvement_fraction
```

### Why `f_decode` is wrong for prediction

`f_decode` (component share within decode-step GPU time) is a **decode diagnostic**. Rank campaign opportunity by addressable production E2E impact — `f_e2e × removable_fraction` — and use `f_e2e` as the Amdahl multiplier. Two reasons:

1. Decode time is only a fraction of E2E (`decode_share_of_e2e < 1` whenever prefill > 0).
2. Decode wall-time is not all kernel-busy time (`decode_busy < 1` whenever inter-kernel overhead exists).

Using `f_decode` directly over-projects E2E gains by `1 / (decode_busy × decode_share_of_e2e)`. That is a **2.1×** over-projection in the worked example below, and it reaches **5–50×** on prefill-heavy or low-busy workloads.

`f_decode` stays in the ranking table as a diagnostic column; `f_e2e` is the column champions plug into Amdahl.

## Conversion Formula

```
f_e2e = f_decode × decode_busy × decode_share_of_e2e
```

| Variable | Definition | Data source |
|----------|-----------|-------------|
| `f_decode` | Component's share of decode-step GPU time | nsys trace; ranking metric only |
| `decode_busy` | `decode_kernel_time / decode_wall_time`. Fraction of decode wall-time that is actual GPU kernel work (vs scheduling, launch gaps). | Tier-0: `merged_busy_per_step × n_decode_steps / decode_avg_s`. Union overlapping kernel intervals (`merged_busy`); never sum concurrent kernels, and scale the per-step profiled window up to the full `decode_avg_s` before dividing. Tier-1: `Σ all_kernel_dur / (avg_latency × num_profiled_iters)` (sweep-level aggregate; **never** mix Chrome-trace timestamps with `RequestOutput.metrics` `time.monotonic` clocks). |
| `decode_share_of_e2e` | `decode_avg_s / avg_s`. Fraction of the authoritative E2E wall that is decode. | `e2e_latency_results.json` — never read the stored key (Denominator rule below). If phase metrics are unavailable, `OL / (IL + OL)` is a disclosed heuristic, not a timing bound. |

**Denominator rule.** Both factors denominate the authoritative E2E wall (`avg_s`). Never normalize a share to `prefill_avg + decode_avg`: that sum is a per-request-mean sub-interval, it discards the admission/ramp residual — real wall time, ~40% of the wall on prefill-heavy workloads — and it can attribute more E2E to decode kernels than decode physically occupies. `mine_trace.py` re-derives both shares from `avg_s` and `decode_avg_s` at point of use and WARNs when the stored key disagrees; never read `decode_share_of_e2e` from JSON. `decode_busy` likewise denominates the decode region's own wall taken from the **same** trace as its merged-busy numerator — never a per-step figure divided by a window from a different run or shape.

Consumers (`verify_validation_gates.py`, `generate_validation_report.py`, `ammo_state.py`) derive `decode_avg_s / avg_s` at point of use and never trust the stored key, because a persisted baseline outlives the emitter that wrote it.

**Ceiling and closure are transcription checks, not accuracy checks.** Under the mandated formulas the four-slice closure is an algebraic identity (`(avg−dec)/avg + busy·(dec/avg) + (1−busy)·(dec/avg) ≡ 1` for any inputs) and the ceiling reduces to `busy ≤ 1`. Both hold exactly for correct and badly wrong numbers alike. Run them — a violation is arithmetic proof of a denominator or transcription slip — but never cite passing them as evidence the measurement is right.

**Provenance is what validates the numbers.** Read the derivation, not just the identities:

- `decode_wall_s` and the ramp residual come from `avg_s` / `decode_avg_s` in the clean profiler-free sweep, never from a profiled run's inflated latency.
- `decode_busy`: the numerator is merged (union) busy over the decode region; the denominator is that region's wall in the same trace. Publish per-rank values — a spread wider than a few points means a wrong region boundary or a leaked warmup gap.
- `inter_kernel_share` must be corroborated by *measured* idle (summed true gaps from the merged sweep), not only by `(1−busy)·decode_share`. Disagreement means the busy figure is wrong.
- State the capture's shape. A reduced `--nsys-output-len` capture has a short, ragged decode tail: its prefill region is still valid (prefill depends on ISL and batch, not OSL), but it cannot supply decode step counts or decode shares.
- Segment phases by a per-step marker plus a phase-distinguishing kernel signature (e.g. attention `gridZ` at chunk size vs decode size), not by wall-clock gap size. Resolve the marker to a single kernel ID — name-substring matches can inflate the step count by an integer factor.

The researcher computes `decode_busy` and `decode_share_of_e2e` once per BS in Stage 2 and writes both into the `## Workload Dilution` table in `bottleneck_analysis.md`. Champions read those values; they do not re-derive them.

## Worked Example — Qwen3.6-27B-FP8, H100, BS=8

From the motivating campaign (see Appendix A of the design spec):

| Quantity | Value | Source |
|----------|-------|--------|
| `f_decode` (DeepGEMM gate_up) | 0.066 | nsys decode-region kernel time |
| `decode_busy` | 0.57 | `9.1 s / 15.9 s` from `## Workload Dilution` |
| `decode_share_of_e2e` | 0.82 | `15.9 s / 19.4 s` from `RequestOutput.metrics` |

```
f_e2e = 0.066 × 0.57 × 0.82 = 0.031
```

A 1.18× kernel speedup (the BW-utilization-limited ceiling for this DeepGEMM shape) projects:

```
E2E_improvement = 0.031 × (1 - 1/1.18) = 0.031 × 0.153 = 0.005  (0.47%)
```

…which may be below `min_e2e_improvement_pct` depending on the configured threshold. A naïve `f_decode`-based projection gives `0.066 × 0.153 = 1.0%`, over-projecting by **2.1×** and giving the champion false confidence the proposal is shippable. After conversion, the same evidence correctly classifies the proposal as marginal. Always compare against the campaign's configured `min_e2e_improvement_pct` (see `references/validation-defaults.md` for default).

## Conversion Is Always Required

Always convert, and use the measured `f_e2e` published for the exact workload bucket. Never decide that dilution is "small enough" to substitute `f_decode`; if the values are close, the conversion simply produces a close answer. When phase timing is missing, label any fallback heuristic and keep the resulting uncertainty or bound in the evidence scope.

## Prefill-Active Components — Why f_e2e Is a Lower Bound

The `f_e2e = f_decode × decode_busy × decode_share_of_e2e` formula handles **decode-only** kernels — kernels whose contribution to E2E is entirely through the decode phase. Some kernels run during BOTH prefill and decode (e.g., attention, embedding, RMSNorm). For these:

- The published `f_e2e` is a **lower bound** on the kernel's true E2E contribution.
- Actual gains from speeding up the kernel may *exceed* the projection because the kernel also accelerates prefill.
- The researcher flags such components with `prefill-active? = Yes` in the Top Components table; champions targeting them should treat their projection as conservative and note the upside in their proposal.

The decode-only bound may justify implementation when it clears the campaign floor; shipping still requires clean Gate 5.3b validation.

## Four-Slice Composition Model

Total E2E wall-time decomposes into four exclusive slices:

```
total_e2e = prefill_time + decode_kernel_time + decode_inter_kernel_time + other
```

Equivalently, in the dimensionless shares the researcher publishes:

```
1.0 = prefill_share_of_e2e + (decode_busy × decode_share_of_e2e) + inter_kernel_share + other_share
        prefill              decode_kernel                          decode_inter_kernel  other (small)
```

Where:

- `prefill_share_of_e2e` = `(total_e2e_s - decode_avg_s) / total_e2e_s` — the prefill/ramp region wall. Never `prefill_avg_s / total_e2e_s`: per-request `prefill_avg_s` (`first_token_ts - scheduled_ts`) excludes admission wait and the batch-fill ramp under chunked prefill, understating the slice several-fold on prefill-heavy workloads.
- `decode_busy × decode_share_of_e2e` — the slice that is actual decode-step kernel execution (sum of all `f_e2e` values for decode-only kernels equals this slice, modulo measurement noise).
- `inter_kernel_share = (1 - decode_busy) × decode_share_of_e2e` — decode wall time spent NOT in kernels (scheduling, launch gaps, host-side overhead).
- `other` — small residual (warmup, KV management, cleanup); typically < 0.02.

Each mechanism attacks a different slice; the slice decides which scope is EV-eligible (§ Pre-Implementation Magnitude — Slice Split).

**Window rule.** All three factors of a phase row (`f_phase`, `phase_busy`, `phase_share_of_e2e`) must use the **same** step window. A transient excluded from one side must be excluded from the other — the span from the wall, the kernel from the busy numerator; either alone is a silent ~10% error. Publish the step range and transient treatment beside the table, and verify the stated formula reproduces the published number.

**Family rule.** Define component families by **enumerated symbol lists**, not substring rules (prefix matchers under-count 15-30%). Completeness requires an **exhaustive partition** — every symbol assigned to one family, families summing to the window total with a stated residual; until one exists, family rows are lower bounds. Reconcile instance counts against per-step cadence (k× expected rate = k call sites). Publish the in-CUDA-graph fraction of any launch-overhead row. Derive secondary tables from the partition or delete them — never hand-maintain two tables over one window.

## Projection Authority — One Boundary Equation

There is one projection equation. Project the delta at the declared production boundary, occurrence-weighted, over baseline E2E:

```
E2E_improvement = (Δ_at_declared_boundary × occurrence_count) / baseline_E2E
```

**Non-overlap rule.** Never sum separately-measured async intervals: CPU API duration, GPU gaps, memcpy, and kernel time do NOT add. If the work you are crediting overlaps, enlarge the boundary to enclose it and do ONE A/B across the enlarged boundary — never add two independent measurements.

**Evidence ladder.** Only `bound`/`production_boundary`/`clean_e2e`-scope magnitude enters a projection; `proxy` magnitude never does (authority: `references/debate-rules.md` § Evidence-Scope Ladder).

**Frame limits.** Optimizations that change occurrence count (e.g. speculative decode) or realize benefit only under load (scheduler / throughput / memory-footprint) are OUT of this frame — instrument them with a phase / throughput measurement, not this equation.

## Pre-Implementation Magnitude — Slice Split

The EV-eligible pre-impl magnitude depends on which slice the mechanism attacks:

- **Kernel-work mechanisms** (kernel replacement, fusion, custom kernel, layout transform): the CUDA-graphed kernel A/B microbenchmark is EV-eligible only when it satisfies `debate-rules.md` § Production Baseline Provenance. A kernel harness that fails dispatch/identity confirmation is `proxy`, not `production_boundary`, and is EV-ineligible.
- **Host / dispatch / inter-kernel mechanisms**: the ordinary EV-eligible pre-impl magnitude is the device-work-removed zero-cost `bound`. Host-gap / dispatch savings become creditable only from an actual-runner boundary A/B. A host-only idea with no removable device work may receive one explicitly contingent implementation spike when a measured production host-slice ceiling meets or exceeds the campaign floor and feasibility is grounded. The ceiling is not expected magnitude and supplies no EV. The spike is the track's first Gate 5.2 execution and uses the normal Gate 5.2 artifacts/state fields; if its boundary remains unchanged, passing it satisfies Gate 5.2 and no duplicate A/B is run. Use a boundary-appropriate clock—host monotonic timestamps or NVTX+Nsys for CPU/dispatch intervals, CUDA events only for GPU intervals. Compare the occurrence-weighted E2E-equivalent improvement from the boundary equation above with the campaign floor, not the interval-local percentage. Stop immediately unless it meets or exceeds the floor; run no expensive clean E2E sweep first.

The micro-experiment must exercise the actual production code path; scheduler / preamble / collective mocks are insufficient.

## Inductor Baseline Parity (Gate 5.2 → E2E Translation)

Gate 5.2 measures the declared production boundary, including the production
compile/fusion behavior. An isolated CUDA-graph harness is binding only when
trace and dispatch evidence prove that its baseline and optimized boundaries
are identical to the compiled production boundary. Otherwise the isolated
result is `proxy` evidence: measure the compiled fused chain or enlarge the
boundary to an actual-runner interval before using its magnitude.

Reason: vLLM's Inductor pass pipeline (§ Which Passes to Check) can fuse the target chain into one op at compile time, so an unfused Gate 5.2 baseline may measure against code that never runs in production.

### How to Check for Inductor Fusion

Before projecting E2E from Gate 5.2 kernel speedup, verify the ACTUAL per-kernel time of the baseline chain inside the compiled CUDA graph:

1. Open the Stage 1 nsys trace (`rounds/{N}/profiling/nsys/baseline_bs{BS}.nsys-rep`)
2. Grep for your target kernel names in the decode window
3. If the unfused chain (e.g., separate `rms_norm` + `dynamic_scaled_fp8_quant`) does NOT appear — but a single fused kernel does (e.g., `rms_norm_dynamic_per_token_quant`) — then Inductor has already fused your target

### Corrected Projection

When Inductor has fused the baseline:

```
effective_speedup = inductor_fused_kernel_time / your_kernel_time   (from nsys)
E2E_improvement = f_e2e × (1 - 1/effective_speedup)
```

An isolated unfused proxy must not be substituted:

```
E2E_improvement = f_e2e × (1 - 1/proxy_unfused_speedup)   ← not Gate 5.2
```

Compare the corrected production-boundary projection with the campaign's configured floor; there is no universal kernel-speedup cutoff.

### Which Passes to Check

Enumerate the passes in the installed vLLM; do not trust a roster in this file:

```bash
grep -rn "class .*FusionPass" vllm/compilation/
```

Read each matched pass's pattern. Overlap classes to look for: norm+quant,
activation+quant, rope+KV-cache write, and collective+norm. If your target
chain matches a live pattern, your Gate 5.2 baseline is likely already fused in
production.

## Practical Usage in Phase 0 / Phase 4

1. Read `f_decode`, `decode_busy`, `decode_share_of_e2e`, `inter_kernel_share`, `prefill_share_of_e2e` from the `## Workload Dilution` table in `bottleneck_analysis.md`.
2. If the researcher already published `f_e2e` for your component (which they do for decode-only candidates in the Top Components table), use it directly — no conversion needed.
3. Otherwise compute `f_e2e` per the formula above.
4. Project with the one boundary equation (§ Projection Authority).
5. Solve for the kernel speedup `s` (or analogous variable) needed to clear the `min_e2e_improvement_pct` threshold.
6. If the required `s` is implausibly large given the BW/compute ceiling, switch to:
   - a different target where `f_e2e` is larger, OR
   - document the limitation and stop the proposal.

## Crossover Prediction from Kernel Data

When an optimization improves some batch sizes but regresses others, predict the crossover batch size from kernel-level measurements. That way you do not run expensive E2E benchmarks at every intermediate BS.

### Per-BS Delta Math

The standard formula generalizes to per-BS:

```
predicted_e2e_improvement(BS) = f_e2e(BS) × (1 - T_kernel_opt(BS) / T_kernel_base(BS))
```

Where `f_e2e(BS)` is the per-BS row in the `## Workload Dilution` table. **Use `f_e2e(BS)`, not `f_decode(BS)`** — see Conversion Is Always Required above.

### Why f Varies with Batch Size

The component share is NOT constant across batch sizes:
- At BS=1, `decode_busy` is often higher (smaller launch gaps relative to kernel work) and a kernel may be 8% of E2E (`f_e2e = 0.08`).
- At BS=32, the same kernel may be 3% (`f_e2e = 0.03`) because (a) the kernel scales superlinearly, (b) `decode_busy` may drop as more launches fit into a step, (c) decode_share may shift if prefill scales differently.

Extract per-BS `f_e2e` from the per-bucket `## Workload Dilution` rows. For an intermediate BS that was not profiled, interpolate linearly between the two rows that BRACKET it — the nearest profiled BS below and the nearest profiled BS above. This is approximate but sufficient for crossover prediction.

### Finding the Crossover

```
crossover_bs = max(BS) where predicted_e2e_improvement(BS) >= noise_tolerance_pct / 100
```

### When Kernel Prediction Is Unreliable

If the warm-cache vs cold-cache kernel speedup ratio exceeds 1.5× at any probed BS, the kernel prediction may not reflect production behavior (where L2 cache pressure from the full model pipeline dominates). In this case:
- For narrow BS ranges (< 15 values): fall back to E2E binary search.
- For wide BS ranges (>= 15 values): gate to exact PASS batch sizes only (skip probing).

See `references/crossover-probing.md` for the full probing protocol.
