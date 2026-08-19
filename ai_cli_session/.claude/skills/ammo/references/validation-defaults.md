# Validation Defaults and Reporting (Validation Stage)

Use this as the **default** guidance for each Validation Stage (Stage 5) track artifact: `{artifact_dir}/rounds/{N}/tracks/{op_id}/validation_results.md`.

This file decides what counts as valid Stage 5 evidence. It fixes what you measure against, the conditions the measurement must run under, the five gates (5.1a, 5.1b, 5.2, 5.3a, 5.3b), how per-batch-size numbers become one track verdict, and what each track report must contain. Read it before you run a gate, and again before you write a verdict. The last section says when a campaign may stop — and which reasons never count.

## What is in this file

- **Dual Baseline Requirement (NON-NEGOTIABLE)**
- **Production Parity Requirement (NON-NEGOTIABLE)**
- Default correctness tolerances (starting points)
- Default kernel perf gate (Stage 5.2)
- Default end-to-end gate (Stage 5.3)
- **Minimum E2E improvement threshold** (campaign-wide viability gate)
- Required reporting checklist for `validation_results.md`
- **Invalid Reasons to Stop** (exhaustion-driven campaign contract)

---

## Dual Baseline Requirement (NON-NEGOTIABLE)

**BLOCKING**: Validation MUST compare against vLLM's actual production kernels, not naive PyTorch.

Both arms of the comparison — correctness and performance — use the production path. A win over a strawman is not a win.

### Correctness Baseline

Call the production entry point that the target component really dispatches to. Read that symbol out of the trace and out of the installed vLLM source. Do not name it from memory, because the export set changes between versions.

### Performance Baseline
```python
# REQUIRED: Measure vLLM's actual kernel time
# NOT: naive PyTorch loops or reimplementations

# Use the same production call path that vLLM uses for the target component
```

### INVALID Baselines (DO NOT USE for any target component)
```python
# WRONG - naive PyTorch loops:
for expert_idx in range(num_experts):
    expert_out = torch.matmul(x_expert, weights[expert_idx])
    output.index_add_(0, indices, expert_out)

# WRONG - manual per-expert GEMM:
for e in range(E):
    mask = (expert_ids == e)
    out[mask] = F.linear(x[mask], w[e])
```

**Verification**: Review `validation_results.md` for each track — the champion documents baseline provenance there (Stage 1 vLLM production kernel, not re-run from worktree). Cross-reference against the baselines captured under `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json`.

---

## E2E Baseline Reuse Requirement (NON-NEGOTIABLE)

**BLOCKING**: The champion MUST use Stage 1 baseline numbers for all E2E latency comparisons. NEVER re-run a baseline from the worktree.

### Source of Truth

| Data | Location (`{CR}` = `campaign.current_round`) | Captured by |
|------|----------|-------------|
| Per-BS E2E latency | `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json` | Stage 1 clean baseline sweep on the session base branch |
| Summary table | `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline E2E latency" | Stage 1 profiler |
| Kernel breakdown | `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline Truth Snapshot" | Stage 1 profiler |

### Why a worktree baseline is contaminated

Worktrees contain optimized code. A "baseline" arm started from a worktree can run the
optimized path instead — for example when the editable install resolves to the worktree
tree. Both arms then execute optimized code, and the real improvement sinks into noise.
This happened in practice: a real 5.5% improvement measured as 0.075%, because baseline
and optimized both ran the optimized code.

### Procedure

1. Read the Stage 1 baseline from
   `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json`.
2. Run ONLY the optimized benchmark from the worktree
   (`--labels opt --baseline-from $STAGE1_DIR`, with the enable flag set).
3. Compare the optimized latency against the Stage 1 latency using the
   following field-resolution rule:
   - Read the flat `avg_latency` / `avg_s` field of the per-bucket entry.
   - Prefer `aggregate.mean_latency` when the entry carries it. Only archived
     artifacts have that block; the sweep no longer writes it.
4. In `validation_results.md`, cite: "Baseline source: Stage 1 (not re-run)"
   and note whether comparison used `aggregate.mean_latency` or `avg_latency`.

### Latency Field Resolution

Every sweep is a single launch. Stage-2 evidence requires only that
`e2e_latency_results.json` exists. Use the helper below to read latency out of
either schema. Its `aggregate` branch serves archived artifacts only:

```python
# Field-resolution helper — use in champion scripts and docs.
def _latency_seconds(entry: dict) -> float | None:
    agg = entry.get("aggregate")   # legacy artifacts only
    if isinstance(agg, dict) and isinstance(agg.get("mean_latency"), (int, float)):
        return float(agg["mean_latency"])
    for k in ("avg_latency", "avg_s"):
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None
```

Significance comes free with the launch. Every sweep row (single-launch included)
carries a `significance` block: a within-launch Welch test on the per-iteration
`latencies` arrays of baseline vs opt (~num_iters samples each).
`significance.significant: false` means the delta is smaller than ~2 standard
errors, and the per-BS verdict logic blocks PASS on such rows automatically. Do
NOT re-run the sweep to "get significance": the within-launch test is the default
significance evidence.

### Sweep Script Guidance

For Stage 5 timing slots, run `scripts/run_vllm_bench_latency_sweep.py` with
`--labels opt --baseline-from $STAGE1_DIR` — optimized-only, compared against
the imported Stage 1 baseline. Do NOT run a baseline arm from the worktree:
it may execute the optimized code path and contaminate the comparison. The
Stage 6 integration sweep runs from the promoted session mainline (not a
track worktree), where a fresh paired measurement is safe; in round >= 2 that
sweep MUST also pass `--baseline-from-arm opt` so the comparator is the
promoted arm and not the directory's stale pre-SHIP baseline arm, which would
make the delta cumulative instead of incremental.

For gate sweeps, also pass:
- `--fresh-cache` (isolates vLLM/Triton compile caches under
  `{out_root}/cache/{sweep_id}` so the previous sweep's partially-warm
  cache does not skew measurements; cache is removed on success)

---

## Production Parity Requirement (NON-NEGOTIABLE)

**BLOCKING**: All measurements MUST use production-equivalent settings.

### Required Environment

Use the exact frozen production environment in `target.json` for both baseline
and optimized arms. Compile and graph modes are target properties, not constants
owned by this guide. The optimized arm may add only the candidate's declared
mechanism-specific flags; all unrelated keys and values must remain identical.

### Forbidden Settings
```bash
# DO NOT USE these in validation:
export TORCH_COMPILE_DISABLE=1     # FORBIDDEN unless the frozen target requires it
--enforce-eager                    # FORBIDDEN unless the frozen target requires it
```

### Benchmark Requirements
```python
# Benchmark script MUST NOT contain:
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # FORBIDDEN
enforce_eager=True                          # FORBIDDEN

# Benchmark script must not override the frozen target environment.
```

### GPU Isolation Requirement (NON-NEGOTIABLE)

**BLOCKING**: Benchmark results are INVALID if collected under GPU contention.

- Only one GPU benchmark process may run at a time on a given set of GPUs
- Before starting any benchmark, verify GPU is idle: `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader`
- **Validation (Stages 5-6)**: Use `scripts/run_vllm_bench_latency_sweep.py` for all
  E2E measurements — it holds a system-wide GPU lock to prevent concurrent runs
- **Profiling (Stage 1)**: Use a separate `run_vllm_bench_latency_sweep.py --slot profiling --nsys-profile --nsys-mode node` invocation. The sweep resolves `--nsys-trace` from `target.hardware` (`cuda-sw` on Blackwell). See `nsys-profiling-guide.md` § Trace Backend.
- If contention is detected mid-benchmark: STOP, report to lead, and re-run after GPU is clear

**Why**: During the OLMo-3-7B verification run, concurrent GPU benchmarks inflated latencies
by ~80% (1.37s → 2.48s) and caused OOM errors on a 44 GiB L40S GPU.

### Kernel-Level Benchmark Requirements (NON-NEGOTIABLE)

For kernel-level (isolated) benchmarks comparing Triton vs CUDA C++:

**REQUIRED**: Capture kernel times under CUDA graphs
```python
# Option A: Use torch.cuda.make_graphed_callables
graphed_baseline = torch.cuda.make_graphed_callables(baseline_fn, (inputs,))
graphed_optimized = torch.cuda.make_graphed_callables(optimized_fn, (inputs,))

# Option B: Manual graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    baseline_out = baseline_fn(*inputs)
g.replay()  # Timed iterations
```

**WHY**: Launch overhead differences between Triton (many small ops) and CUDA C++
(single kernel) are ~100-200 µs. CUDA graphs eliminate this, enabling fair comparison.

**INVALID**: Timing with torch.cuda.Event alone without CUDA graph capture
```python
# WRONG - unfair comparison due to launch overhead:
start.record()
baseline_out = fused_experts(...)  # Triton launch overhead: ~50-100 µs
end.record()
```

**Verification**: The champion MUST confirm CUDA graph usage in their benchmark scripts before running Gate 5.2. Benchmarks without CUDA graph capture are a Stage 5.2 FAIL — the `production_parity` invariant is violated.

### Cold-Cache Requirement for Bandwidth-Bound Kernels

When a load-bearing claim depends on bandwidth or cache residency, report both warm-cache and representative pipeline/cold-cache evidence.

- **Warm-cache**: Standard CUDA-graphed loop (100+ iterations on same tensors)
- **Cold-cache**: Use L2-busting methodology — chained distinct data totaling > 2.5x L2 cache size *(hardware-anchored sizing; debate scoring only, NOT impl ship gate)* between measurements, or use distinct random tensors per iteration

**Why**: Tight CUDA graph loops on small tensors keep data in L2 cache, inflating speedups for BW-bound kernels. In production, the full model pipeline (N layers x per-layer state) typically exceeds L2, forcing DRAM access.

Pick a working set that represents the measured production pipeline, and say why it
is representative. When cache behavior is not load-bearing, cache experiments are
diagnostic only — they are not a universal ship gate.

---

## Default correctness tolerances (starting points)

These are *starting points*, not universal truths.

### Gate 5.1a: Synthetic kernel tests

- **FP32**: `atol=1e-3`, `rtol=1e-3`
- **BF16/FP16**: `atol=1e-2`, `rtol=1e-2`
- **FP8 / block-quant**: **must be model-specific — there is no valid generic placeholder.**
  - Copy tolerances from the model's actual vLLM tests (`tests/quantization/`, `tests/models/`), or derive them by measuring the baseline kernel's own output spread across two runs and setting tolerance to ~2× that spread.
  - Do NOT invent loose bounds to make a comparison pass: an `atol` on the order of the activation magnitude (e.g. hundreds for logits) accepts almost any output and turns Gate 5.1a into a no-op. If you cannot justify a tolerance from model tests or baseline-spread measurement, rely on Gate 5.1b (E2E GSM8K accuracy) as the correctness signal and record the 5.1a tolerance as "not established".

Also require:
- no NaNs/Infs
- shape/stride parity
- deterministic indexing for routing and pair ordering (when required by baseline)

### Gate 5.1b: E2E Greedy Decode Correctness (HARD GATE)

**Gate 5.1b is a hard gate.** Correctness failure blocks the track from proceeding to latency benchmarks.

The champion owns it and runs it through the sweep script. The result is deterministic, so no validator is involved. The script's Phase 1 runs GSM8K greedy decode with `logprobs=5` and compares optimized outputs against golden refs captured in Stage 1.

**Invocation**:
- Stage 1 (capture): `--capture-golden-refs` → saves `json/golden_refs.json`
- Stage 5 (verify): use the dedicated `--slot opt_correctness/{op_id}` with `--labels opt --verify-correctness --baseline-from $STAGE1_DIR`. Its durable result is `rounds/{N}/sweeps/opt_correctness/{op_id}/json/correctness_verdict.json`; never overwrite it with profiling or official timing output.

**Gate logic**: `opt_gsm8k_accuracy >= baseline_gsm8k_accuracy - (tolerance_pct / 100)`

- **n = 1319** questions by default (configurable via `--correctness-num-questions`; bundled as `data/gsm8k_full.json` for offline/sandboxed AMMO sessions)
- **tolerance_pct = 1.0pp** by default (configurable via `--correctness-tolerance-pct`). Allows ~13 questions of noise at N=1319. Set to `0.0` for strict `opt >= baseline` comparison.
- **Percentage comparison** — allows question-level churn (different questions correct) as long as aggregate accuracy stays within tolerance of baseline
- Token-level data is computed and logged as diagnostics only
- Single gate for all tracks regardless of lossless/lossy classification
- **max_tokens = 1024**

**Exit codes**: 3 = correctness FAIL (opt_accuracy < baseline_accuracy - tolerance), 4 = infrastructure error (retry). Two checks raise exit code 4 instead of a verdict:

- **Metadata mismatch check**: golden refs `num_questions` or `max_tokens` differs from the verification run → exit code 4 (infrastructure error). `tolerance_pct` is recorded in golden metadata for traceability but is NOT enforced on mismatch, so agents may tune tolerance per campaign without re-capturing golden refs.
- **Baseline accuracy floor**: `baseline_correct_count == 0` while `num_questions > 0` → exit code 4 (infrastructure error — the model cannot solve any GSM8K questions).

**Self-consistency check** (Stage 1 only): Golden ref capture runs prompts twice to verify greedy decode is deterministic. If non-deterministic, metadata records `deterministic: false`.

**Classification scope**: Lossless/lossy classification does NOT affect Gate 5.1b. Classification determines Gate 5.1a tolerances (BF16 vs FP8).

## Default kernel perf gate (Stage 5.2)

### Gate 5.2 (mechanism impact)

Gate 5.2 = mechanism impact at the declared production boundary; it always runs. The boundary varies by mechanism: kernel → production kernel A/B; fusion → the compiled fused chain; dispatch/graph → the actual-runner boundary interval; overlap → the enclosing phase. For fully-overlapped work the boundary degenerates to the whole phase, where Gate 5.2 coincides with the phase-significance path — that coincidence is the signal that no cheap proxy exists.

Measure at that boundary. For kernel and fusion mechanisms, measure **GPU kernel time** under CUDA graphs for the same bucket set as Stage 1. For dispatch/graph/overlap mechanisms the measured quantity is the boundary interval named above (actual-runner boundary interval / enclosing phase), NOT kernel time — the boundary also names the instrument.

Gate 5.2 requires a reproducible positive delta at the declared production
boundary for at least one target bucket and an explanation of any negative
bucket. It does not reuse E2E verdict tiers or impose a universal percentage
floor. Gate 5.3b alone classifies shipping, regression, and gating. The one
campaign-wide floor is `campaign.config.min_e2e_improvement_pct`, which applies
uniformly to both categories (§ Minimum E2E Improvement Threshold).

Reporting requirements:
- baseline vs optimized per-bucket table (µs + speedup)
- if possible, per-stage breakdown (routing/prepare/W1/act/quant/W2/reduce)
- claim-appropriate profiler sanity when the conclusion depends on hardware counters or launch-resource details

### Inductor Baseline Parity

Gate 5.2 never promotes an uncompiled/raw-function comparison into a binding
result. An isolated CUDA-graph harness is admissible only when trace and
dispatch evidence prove that it reproduces the compiled production boundary.
If Inductor fuses or rewrites that boundary, measure the compiled fused chain
or enlarge to the actual-runner interval; retain the isolated result as
proxy-only evidence.

See `references/e2e-delta-math.md` § Inductor Baseline Parity (Gate 5.2 → E2E Translation) for the check and corrected formula.

## Default end-to-end gate (Stage 5.3)

### Gate 5.3a: Production Activation Proof (NON-NEGOTIABLE)

Gate 5.3a proves the optimized mechanism really runs at its declared production boundary, under the frozen CUDA-graph/compile configuration. Name the mechanism-specific activation signature before you profile: replacement-kernel identity, fused launch-chain change, dispatch/graph path marker, or overlap/timeline ordering. The proof must match that claim. Kernel-name presence alone is not a universal gate.

Nsys is normally sufficient for path, launch, graph, and timeline claims. Use NCU or another detailed profiler only when a load-bearing claim needs counters or launch-resource details. Run profiling separately from the clean E2E sweep; the standard Nsys capture is:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {N} --slot opt_profiling/{op_id} --labels opt \
  --nsys-profile --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1
```

Verify the declared activation signature from the trace and any actual-runner instrumentation. For a kernel replacement this normally includes the expected production-dispatch kernel identity; for fusion, dispatch, graph, or overlap work it may instead be a changed launch chain, path marker, graph node, or timeline relationship.

The durable profiling result lives under `rounds/{N}/sweeps/opt_profiling/{op_id}/`; it never shares the correctness or authoritative timing slot.

- **If the declared activation signature is observed**: PASS. Proceed to Gate 5.3b.
- **If the signature is absent or ambiguous**: FAIL. Do not use Gate 5.3b as evidence until activation is established.
- **Latency numbers from profiled runs are INVALID** (profiler overhead). Only the trace data matters.

Cost: ~85s (4B/L40S), ~4.5 min (70B/8xH100).

### Gate 5.3b: E2E Measurement Sweep

Run clean E2E under identical knobs and capture/compile settings in dedicated `--slot opt/{op_id}` with `--labels opt --baseline-from $STAGE1_DIR` (optimized-only from the worktree; the baseline is the imported Stage 1 measurement — see § E2E Baseline Reuse Requirement). The authoritative result is `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json`. Record the opt-arm identity and the Stage 1 baseline source in `evidence.json`. No profiling flags are permitted. **Only runs after Gate 5.1b and Gate 5.3a pass.**

Default iteration counts:
- **Profiling** (Stage 1): `--num-iters 1` (keep traces small)
- **Validation** (Stage 5): Use `num_iters` from `target.json` (default: 10 via `new_target.py`)

### Per-BS Tiered Verdict

Thresholds from `target.json` gating block (defaults: `noise_tolerance_pct: 0.5`, `catastrophic_regression_pct: 5.0`, `min_e2e_improvement_pct: 0.5`). Computed by `scripts/generate_validation_report.py` — do not hand-classify.

| Speedup | Verdict | Meaning |
|---------|---------|---------|
| >= 1 + max(min_e2e_improvement_pct, noise_tolerance_pct)/100, AND significant | `PASS` | Improvement at this batch size clears both the campaign floor and the noise band |
| >= (1.0 - noise_tolerance_pct/100) | `NOISE` | Within measurement noise (or below the PASS floor, or not statistically significant) — treated as neutral |
| >= (1.0 - catastrophic_regression_pct/100) | `REGRESSED` | Material regression, gating required |
| < (1.0 - catastrophic_regression_pct/100) | `CATASTROPHIC` | Too large to gate, track fails |

**Significance requirement**: sweep rows carry a `significance` block (within-launch Welch test on the ~num_iters per-iteration latencies of baseline vs opt — see § Latency Field Resolution). `significance.significant: false` blocks PASS regardless of the speedup magnitude; the row classifies NOISE. Legacy rows without the field fall back to the floor check alone. This costs zero extra launches — the per-iteration data is already collected in every single-launch sweep.

### Per-BS Verdicts and Track-Level Fallback Ladder

| Per-BS Results | Track Verdict | Action |
|---------------|--------------|--------|
| All PASS or NOISE (at least one PASS) | `PASS` | Ship directly |
| Any CATASTROPHIC | candidate for `FAIL` | Apply this section's fallback ladder before authoring FAIL |
| Some PASS + some REGRESSED | `GATING_REQUIRED` | Champion implements gating (see below); on success: `GATED_PASS` |
| All REGRESSED/NOISE (no PASS) | candidate for `FAIL` | Apply this section's fallback ladder before authoring FAIL |

The canonical track-level fallback ladder is `PASS → GATED_PASS → GATING_REQUIRED → RETRY_WITH_CONTINGENCY → FAIL`. This table names the per-BS inputs; the ladder names the verdict rungs. `FAIL` is authored only after every applicable rung is exhausted. Gating implementation details live in `impl-track-rules.md`; this table does not authorize terminal failure on its own.

### Phase Decomposition (flags are triage, never a verdict input)

`generate_validation_report.py` renders a per-BS phase table alongside the verdict table, computed from fields the sweep **already harvests** (`prefill_avg_s` / `decode_avg_s` / `decode_share_of_e2e` from `RequestOutput.metrics`, plus derived `tpot_s` / `otps`). Zero extra benchmark runs, zero extra model loads. These are batch-mode phase decompositions of the SAME fixed-batch measurement — **not** serving TTFT/ITL under load (no queueing or arrival dynamics exist in this instrument).

Read the three flags below as triage only. Phase-level Welch significance is a separate quantity, computed separately, and it IS a verdict input for the § DILUTED_PASS Ship Path — and for nothing else.

Three flags, all informational — **none of them changes a per-BS or track verdict**:

| Flag | Condition | What the consumer does |
|------|-----------|------------------------|
| `DILUTED-WIN` | Decode-slice improvement ≥ `max(min_e2e_improvement_pct, noise_tolerance_pct)` while the e2e verdict is below PASS, AND the e2e delta matches the phase expectation `share × decode_Δ + (1−share) × prefill_Δ` | The mechanism worked; the small e2e number is Amdahl dilution — the instrument working correctly. If the track fails on the e2e floor, record `fail_reason` as "diluted (Amdahl-consistent)", NOT "ineffective" — this distinction feeds `exhausted_technologies[]` decisions (a diluted mechanism may be worth re-proposing on a workload/component where its `f_e2e` is larger; an ineffective one is not). It does NOT rescue the ship on its own — it rescues the ship only when the stricter, significance-gated conditions in § DILUTED_PASS Ship Path (below) also hold. |
| `PHASE-REGRESSION` | e2e verdict is PASS but prefill or decode regressed beyond `noise_tolerance_pct` | A net win masking a phase-level regression; a different ISL/OSL mix could flip the sign. The § Decision section MUST address it (explain why the regression is acceptable for the target workload, or add gating) before SHIP. The auditor checks this at S45. |
| `INCONSISTENT` | Actual e2e delta disagrees with the phase-decomposition expectation (beyond noise/50% relative) | Wall-clock latency and `RequestOutput.metrics` timestamps disagree — measurement integrity problem. Investigate before trusting either number. |

Caveat: no significance test applies to the phase SCALARS in this table (`prefill_avg_s` / `decode_avg_s` / `decode_share_of_e2e` and the derived `tpot_s` / `otps`), because they are per-request means with no per-iteration distribution. That is why the three flags above are triage signals, not statistical verdicts, and why none of them changes a verdict. The sweep ADDITIONALLY emits per-iteration phase-mean arrays (`prefill_iter_means_s` / `decode_iter_means_s`, one batch-mean per iteration). Phase-level Welch IS computed on those arrays (`_row_phase_significance` → `row["phase_significance"]`). Exactly one gate consumes that phase Welch result — the § DILUTED_PASS Ship Path below — and nothing else does. It is never an input to the three informational flags or to the ordinary per-BS/track verdict ladder.

### DILUTED_PASS Ship Path

`DILUTED-WIN` (above) is informational and never rescues a track on its own. A track that the verdict ladder resolved to `FAIL` **because every BS landed in the NOISE band** (real decode win, but Amdahl-diluted to a ~1.00x e2e) can nonetheless SHIP as a passing terminal track — but only when the sweep JSON carries enough evidence to satisfy ALL of the following stricter, significance-gated conditions. `generate_validation_report.py` computes every threshold below from sweep-produced JSON only. The champion's only levers are which workload to sweep and what to write in prose — never the arithmetic.

**Ship conditions (0-6, ALL required, checked at EVERY swept BS row):**

0. **Sample floor.** The sweep ran with `bench.num_iters >= 30` — mechanically, `phase_significance.decode.n_baseline >= 30` AND `phase_significance.decode.n_opt >= 30` at every row. A run at the general `num_iters=10` default disqualifies the whole track. (The floor mitigates the fixed `|t|>=2.0` Welch liberalness at low df and the BS-dependent variance shrinkage of `decode_iter_means_s`; it is a mitigation, not a proof of parity.)
1. **Decode delta Welch-significant.** `phase_significance.decode.significant is True` at every row. A row missing the decode Welch key (Tier-B/C fallback, or a length-mismatch guard failure) disqualifies the whole track — fail-closed. Champions cannot cherry-pick which BS carry phase evidence.
2. **Decode improvement clears its own floor, in E2E-EQUIVALENT terms.** `decode_improvement_pct × decode_share_of_e2e >= 2 × noise_tolerance_pct` at every row. The share-weighting (same basis as the Amdahl-consistency check) states the floor in e2e-relative units, closing the workload-choice (ISL/OSL) lever a flat decode-relative floor would leave open.
3. **Prefill NOT regressed beyond noise at ANY BS, significance-gated.** For every row: if `prefill_improvement_pct < -noise_tolerance_pct` AND `phase_significance.prefill.significant is not False` (i.e. significant OR unknown), the track is vetoed. A non-significant negative wobble does not veto; a missing prefill Welch result still vetoes (fail-safe on the blocking direction).
4. **E2E speedup >= 1.0 at EVERY BS.** `row["speedup"] >= 1.0` everywhere — deliberately stricter than "not REGRESSED" (the NOISE band permits `speedup < 1.0`). No net regression anywhere.
5. **Amdahl-consistent at every row.** `_phase_flags`'s `amdahl_consistent` must be `True` (not `None`, not `False`) for every candidate row.
6. **Gates 5.1a/5.1b unchanged.** Correctness is still mandatory and gates BEFORE the E2E sweep — DILUTED_PASS adds no new correctness surface; a track that failed 5.1a/5.1b never reaches this path.

**Marker schema.** DILUTED_PASS introduces **no new status enum value**. A qualifying track keeps `status == "PASS"` and gains one optional field `diluted: true` on the track object, plus an `e2e_gate.diluted_pass` evidence block in `validation_summary.json`. The `ammo_state.py` cross-field rule enforces `diluted: true ⟹ status == "PASS"` (a structurally nonsensical `diluted:true` + non-PASS combination is blocked at validation).

**DILUTED_PASS is discovered by the report generator after a sweep runs — never a champion's projected verdict.** It is a validation-stage OUTCOME, checked only when the ladder already produced `FAIL`; it is not a proposal target, and EV ranking (debate) never sees it (see `debate-scoring-rubric.md`).

**Cumulative-accounting firewall.** A DILUTED_PASS ship contributes its MEASURED e2e (~1.00x) to `cumulative_speedup_vs_round1` — never its TPOT gain. The TPOT gain is reported on its own line (`round.diluted_tracks[].tpot_improvement_pct`), never folded into cumulative speedup.

### GATING_REQUIRED Workflow

When the track verdict is `GATING_REQUIRED`, the champion runs the gating workflow (feasibility check → crossover probing → gating mechanism → re-validation → `GATED_PASS`/`FAIL`, one attempt per track). The canonical step-by-step definition lives at `impl-track-rules.md § GATING_REQUIRED Workflow` — follow it there.

Target batch sizes are defined in `target.json`. Use `references/e2e-delta-math.md` to set realistic expectations for E2E delta given component share `f`.

## Minimum E2E Improvement Threshold

Every optimization candidate must clear a minimum expected E2E improvement to be worth pursuing. One threshold does this job for all of them, in place of per-optimization ad-hoc criteria.

**Default**: `campaign.config.min_e2e_improvement_pct: 0.5` in `state.json` (scaffolded by `new_target.py` from `schemas/state.schema.json`). This is the single authoritative documentation of the default value — all other files reference this section rather than hardcoding a number. The default equals `noise_tolerance_pct` by design: the minimum "win" must never sit inside the declared noise band. If you raise the noise tolerance, raise this floor with it.

### Where It's Checked

| Decision Point | Check | Basis |
|---------------|-------|-------|
| **Pre-debate (campaign stop)** | `top_addressable_e2e_pct < threshold` | Largest measured `f_e2e × removable_fraction` across non-overlapping slices |
| **Post-debate (candidate gate)** | `max(e2e_projections) < threshold` | Champion's projected E2E gain from the debate-scoring formula |
| **Post-validation (plain PASS)** | Per-BS `PASS` requires `speedup ≥ 1 + max(threshold, noise_tolerance_pct)/100` AND within-launch significance | § Per-BS Tiered Verdict — enforced by `generate_validation_report.py` |
| **Post-validation (GATED_PASS)** | At least one BS shows E2E improvement ≥ threshold | Per-BS verdict system handles regression classification |

**Host/inter-kernel-slice note**: the pre-debate stop check includes the largest physically removable host/inter-kernel slice without adding overlapping intervals (see `references/e2e-delta-math.md`). The same `min_e2e_improvement_pct` applies uniformly across mechanisms.

**Pre-debate/campaign-stop math**: `top_addressable_e2e_pct = 100 × max(f_e2e × removable_fraction)` across non-overlapping slices. If that value is below the threshold, no measured physical mechanism can clear the floor. Raw component share, decode-only share, and a proxy estimate are not substitutes.

**Label a ceiling achievable or upper-bound.** A Speed-of-Light figure (`1 / max(pipe utilization)` from NCU) is an upper bound on speedup, not a forecast — it is unreachable when the same report shows the kernel latency-bound at low occupancy. Using one as `removable_fraction` is legitimate but must be disclosed as an upper bound, naming the pipes *not* measured; if a proxy harness produced the counters, state what it reconstructs rather than replays (identical input tensors across launches inflate cache hit rates).

**Post-debate math lives in `references/debate-scoring-rubric.md`**, not here. Do not restate the projection formula in validation-stage prose.

**GATED_PASS rule**: An optimization that benefits some batch sizes but regresses others is still worth pursuing if at least one BS shows E2E improvement ≥ `campaign.config.min_e2e_improvement_pct`.

## Required reporting checklist for `{artifact_dir}/rounds/{N}/tracks/{op_id}/validation_results.md`

Include:

1) **Repro commands**
- exact commands for baseline and optimized runs
- env vars and flags that affect dispatch / CUDA graphs / torch.compile / quant

2) **Environment**
- GPU model + driver/CUDA
- vLLM commit or version
- model id + quant format
- TP/EP topology

3) **Correctness**
- tolerance used + rationale
- max/mean absolute error (and any outliers)
- special-case tests for top_k>1 (overlap / reduction)

4) **Kernel perf (production parity)**
- bucket set and capture mode
- baseline vs optimized per-bucket µs table
- Gate 5.3a claim-appropriate profiler evidence proving the declared production activation signature

5) **E2E latency**
- baseline vs optimized per-bucket table
- variance notes (iters, warmup, noise sources)
- connection to component share `f` (if improvement is small)
- Per-BS verdict table (PASS/NOISE/REGRESSED/CATASTROPHIC) for each tested batch size
- Phase-decomposition table (auto-rendered by `generate_validation_report.py` when the sweep JSON carries phase fields) + resolution of any `PHASE-REGRESSION` flag in § Decision

6) **Decision**
- ship / restrict envelope / pivot route / stop
- Stage 6 enablement guard proposal (what exactly will be enabled, where, and how to roll back)
- If GATED_PASS: dispatch mechanism type, env var name, dispatch condition, crossover_threshold_bs, pre-gating and post-gating per-BS E2E tables

---

## Invalid Reasons to Stop

The campaign stop condition is purely mechanical: `bottleneck_mining.top_addressable_e2e_pct < campaign.config.min_e2e_improvement_pct`. The orchestrator has ZERO discretion.

**Design intent (do not "fix"):** the threshold is deliberately permissive — it will rarely fire on its own, and that is the point. Campaigns are exhaustion-driven: they end when every viable avenue has been tried (accumulated `exhausted_technologies[]` starving candidate generation), not when a heuristic decides further rounds look unpromising. A future maintainer noticing that "the stop condition almost never triggers" is observing intended behavior.

The following are NOT valid reasons to stop or ask the user:

- **"The round's chosen technology class (e.g., Triton) didn't work"** → let the next round's debate pick a different technology per `references/technology-selection.md`
- **"The remaining bottleneck is near its physical ceiling"** → the addressable share `f_e2e × removable_fraction` already encodes the ceiling; if `top_addressable_e2e_pct >= min_e2e_improvement_pct`, the math says there's room. This governs not-stopping only — an upper-bound ceiling still makes the addressable share optimistic, and the debate must weigh that (§ above)
- **"A new round is unlikely to find better candidates"** → the orchestrator cannot predict debate outcomes
- **"The campaign has been running for many rounds"** → round count is not a stop criterion
- **"Implementation complexity is increasing"** → complexity is scored in debate, not a campaign-level gate

SKILL.md's Stage 7 contract codifies the same rule from the orchestrator side, and the Stop hook (`ammo-stop-guard.sh`) enforces it mechanically — it blocks session end while `campaign.status == "active"`.
