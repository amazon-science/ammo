# Implementation Track Rules

These are the agent-facing rules for a Stages 4-5 parallel implementation track. The impl-champion owns the track and must follow every rule here. Read it before you edit a source file, and again before you record a gate result. It covers who may change code, when a rebuild is required, the fixed gate sequence, the one-attempt recovery path when per-batch-size results are mixed, and how the env flag you register reaches a vLLM maintainer.

## Worktree Build Rules

The champion compiles. The kernel correctness & speedup checks run against the committed, compiled code.

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, CuTeDSL kernels, configs) | Edit, test, commit. **No rebuild.** | Immediate (CuTeDSL JIT-compiles on first import — cache under `$CUTE_DSL_CACHE_DIR` or `/tmp/{user}/cutlass_python_cache`) |
| **C++ kernel** (csrc/ changes, including CUTLASS C++ templates) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- Write your kernel-gate artifacts to `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validator_tests/`, where `{CR}` is `campaign.current_round`. The `validator_tests/` dirname is kept as a historical label because downstream consumers read from this exact path: the state.json merge, the dashboard tabs, and the eval scorer.
- These gates run before the E2E sweep, so they need no GPU coordination.

## Write the Kernel Gates From the Plan

Derive your kernel correctness tests and benchmarks from the **optimization plan and debate summary**, not from the code you wrote. Test what the optimization SHOULD do (the plan) rather than only what it DOES do (the implementation). Hold yourself to the plan's intent: keep assertions tight, and report benchmark numbers straight.

## Champion-Owned Validation

The champion owns all Stage 5 validation, running the kernel-level gates:

```
Kernel-Level:
  Gate 5.1a: Kernel correctness tests (champion-authored, from the plan)
  Gate 5.2: Kernel speedup benchmark under production parity (boundary: `references/validation-defaults.md` § Gate 5.2)

E2E-Level (Champion):
  Gate 5.3a: opt_profiling/{op_id} (production activation; separate profiling invocation)
  Gate 5.1b: opt_correctness/{op_id} (clean production correctness)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Writes final validation_results.md with evidence chain
```

For host/dispatch/inter-kernel mechanisms, run the actual-runner boundary A/B as a cheap early gate before the E2E sweep. An ordinary track must confirm its projected magnitude. A `contingent_host_spike` has no projected host-saving magnitude, so its first obligation is instead the minimum reversible production-boundary spike, recorded as its first Gate 5.2 run; it stops unless the occurrence-weighted E2E-equivalent improvement meets or exceeds `campaign.config.min_e2e_improvement_pct` (`references/e2e-delta-math.md` § Pre-Implementation Magnitude — Slice Split).

Run Gate 5.1b and Gate 5.3a as separate sweep invocations. Profiler overhead and trace-buffer growth can invalidate or exhaust a full correctness run, and the sweep script rejects the combined invocation.

## When a Kernel Gate Fails

When a kernel correctness or speedup check fails:

1. Champion records the gate failure details
2. Champion diagnoses root cause
3. Champion fixes implementation, recompiles if needed
4. Champion completes the Self-Validation Gate checklist (root cause reasoning, smoke test, fix-attempt counter)
5. Champion commits and re-runs the kernel gates (no E2E sweep until 5.1a PASSes)

Never "fix" a failure by loosening your own test. The gate exists to catch a broken kernel before the expensive E2E sweep. A weakened assertion only defers the failure to Gate 5.1b (GSM8K) or 5.3b (latency), where it surfaces anyway — and by then you have lost the kernel-level diagnosis you would have had here.

**DILUTED_PASS is not a target you implement toward.** A track whose e2e verdict lands in FAIL-by-all-NOISE may still SHIP as diluted (`status=PASS`, `diluted=true`) when `generate_validation_report.py` finds all 7 significance-gated conditions hold (`references/validation-defaults.md` § DILUTED_PASS Ship Path). That is a validation-stage OUTCOME discovered after the sweep runs — never something to design a mechanism toward. Do not build a decode-only optimization and count on dilution to rescue the ship. Do not argue for a small projected e2e delta on that basis either: EV ranking uses projected e2e only (`references/debate-scoring-rubric.md`).

## Track State Reconciliation

A failure is not complete until `state.json` is reconciled against the on-disk gate result files, and the orchestrator does that reconciliation. An infrastructure blocker follows the same rule: it is not recorded until the orchestrator reconciles the track entry.

The champion never edits `state.json` directly. It writes the gate artifacts and `rounds/{CR}/tracks/{op_id}/validation_results.md`, and the orchestrator merges them into `campaign.rounds[$IDX].parallel_tracks.tracks[op_id]` at each checkpoint (see `orchestration/parallel-tracks.md` § Reconciliation and Cohort Barrier):

- Gate 5.1a: `rounds/{CR}/tracks/{op_id}/validator_tests/gate_5_1a_results.json`
- Gate 5.2: `rounds/{CR}/tracks/{op_id}/validator_tests/gate_5_2_results.json`
- Gate 5.3b (E2E sweep): `rounds/{CR}/sweeps/opt/{op_id}/e2e_latency_results.json`

The reconciled track entry must carry a `status` and `verdict` that agree with these gate files (`verdict` is `null` for `GPU_BLOCKED`). `GPU_BLOCKED` is a lead-triage blocker, not a terminal pass/fail verdict, so Stage 6 must not count it as complete. Never leave a failed or blocked track at `IN_PROGRESS` with stale verdict fields.

## GATING_REQUIRED Workflow

> **This is the canonical definition.** Other files reference this section.

When per-BS verdicts show mixed results (some PASS + some REGRESSED), the track enters GATING_REQUIRED:

1. Sweep reports per-BS verdict table showing mixed results
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion runs crossover probing benchmarks itself
4. Champion runs the kernel sweep + E2E confirmation per `crossover-probing.md`
5. Champion implements gating mechanism per § Batch-Size Dispatch Mechanisms below (decision tree)
6. Champion registers env var in `vllm/envs.py`, defaulting off (`=0`). **The flag name is the PR-facing public name of the optimization** — derive it from the mechanism per § Env Flag Naming (PR-Ready) below, never from the internal `op_id`.
7. Champion commits gated implementation
8. Champion re-runs the kernel gates on the gated kernel (correctness & speedup)
9. Champion re-runs separate activation profiling, clean correctness, and clean E2E on gated code (5.3a, 5.1b, 5.3b) — all BS must be PASS or NOISE
10. If both kernel re-validation and sweep pass: verdict = `GATED_PASS`. If either fails: verdict = `FAIL`.

One gating attempt per track — no nested gating.

## Env Flag Naming (PR-Ready)

> **This is the canonical naming rule.** Other files reference this section.

Every `VLLM_*` env flag you register in `vllm/envs.py` is **not an internal label** — this holds for a GATED_PASS dispatch gate and for any opt-in optimization. Three routes carry the name outward: it ships verbatim in the `vllm/envs.py` diff, it gets promoted into `target.json:bench.baseline_env` on SHIP, and it is copied straight into the PR's enable instructions (the PR workflow reads `baseline_env` keys verbatim — it cannot rename a flag without rewriting your merged code). So the name a reviewer reads in the PR is the name you type here. A name that doesn't communicate what the optimization does reads as low-effort and gets the PR bounced.

**Never let the internal `op_id` become the flag name.** An `op_id` (`op007`, `OP-003`) is a campaign tracking handle for wiring up agents, tracks, and artifact dirs. It is meaningless to a vLLM maintainer.

**Convention** — describe the optimization, not its tracking number:

```
VLLM_<SCOPE>_<MECHANISM>[_<ARCH>]
```

- `<SCOPE>` — the model family or subsystem the flag governs (`NEMOTRON3`, `MAMBA2`, `MOE`, `ATTN`). Use the model family when the optimization is checkpoint-specific; use the subsystem when it's general.
- `<MECHANISM>` — what the kernel/dispatch actually does (`FP8_PREFILL_GEMM`, `GATED_RMS_NORM_FUSION`, `TWO_STREAM`, `SSD_FUSED_STATE`).
- `<ARCH>` — optional hardware tag when the path is architecture-gated (`SM100`, `SM90`).

**Example 1 (good):**
Optimization: a CUTLASS SM100 FP8 dense GEMM for the prefill path of Nemotron-3.
Flag: `VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100`

**Example 2 (bad — rejected):**
Same optimization, named off the tracking id.
Flag: `VLLM_OP004` — a maintainer cannot tell what it does, what it touches, or whether it's safe to enable. This is exactly the leak that gets a PR bounced on naming.

**Rule of thumb**: if you deleted the campaign's `state.json`, would the flag name still tell a stranger what the optimization does? If not, rename it before you commit. A flag whose name is just `VLLM_OP<n>` / `VLLM_OPT<n>` (the op_id with a `VLLM_` prefix) is never acceptable — the Stage 4-5 audit treats it as a BLOCKING finding.

## Batch-Size Dispatch Mechanisms (for GATED_PASS Optimizations)

A `GATED_PASS` optimization needs a dispatch mechanism so it activates only at the beneficial batch sizes. Pick the variant that fits the dispatch site's context, using the decision tree below.

### Decision Tree

```
1. Is the dispatch site inside a fullgraph-compiled region (torch.compile fullgraph=True)?
   YES -> Use torch.cond() (Variant 1)

2. Is the dispatch site inside a custom op, layer forward(), or CUDA-graphed path?
   YES -> Use Python if/else on M dimension (Variant 2)
         Sub-decision: Is the threshold architectural or empirical?
           ARCHITECTURAL (e.g., kernel's BLOCK_M determines max M) -> hardcode threshold
           EMPIRICAL (from crossover probing) -> use probed threshold with conservative bias

3. Is the dispatch site at module init time or platform level?
   YES -> Use init-time function pointer selection (Variant 3)
```

### Variant 1: torch.cond (Fullgraph-Compiled Paths)

Use this when `torch.compile(fullgraph=True)` traces through the dispatch site. A standard Python `if` on tensor shape causes graph breaks under fullgraph mode.

```python
# Reference: `torch.cond` in the fp8 GEMM dispatch (quantization/utils/fp8_utils.py)
condition = input.shape[0] < crossover_threshold
return torch.cond(
    condition,
    optimized_fn,    # Active for BS below threshold
    baseline_fn,     # Fallback for BS above threshold
    (input, weight, *other_args),
)
```

Both branches must return same-shape tensors.

### Variant 2: Python if/else (CUDA-Graphed / Layer Forward Paths)

> In the dispatch templates below, `{OP_NAME}` is a **descriptive mechanism name** (e.g. `MOE_TWO_STREAM`, `FP8_PREFILL_GEMM_SM100`) — the public, PR-facing flag name, NOT the internal `op_id`. `VLLM_OP003`-style names leak the tracking handle and are a BLOCKING Stage 4-5 audit finding. See § Env Flag Naming (PR-Ready) above.

Use this for code paths captured by CUDA graphs. The Python conditional is evaluated at graph capture time and frozen -- no runtime cost during replay. Each batch-size bucket captures a separate graph with the correct branch.

```python
# Two-level dispatch: env var enables, M-check selects

# Level 1: Env var gate (checked once at init)
if envs.VLLM_{OP_NAME}:
    gemm_fn = gated_optimized_gemm
else:
    gemm_fn = default_gemm

# Level 2: Runtime M-check (frozen in CUDA graph per bucket)
def gated_optimized_gemm(layer, x, weight, bias=None):
    M = x.numel() // x.shape[-1]
    if M <= crossover_threshold:  # Beneficial range
        return optimized_kernel(x, weight)
    return default_kernel(x, weight, bias)  # Baseline fallback
```

### Variant 3: Init-Time Function Pointer (Module Load)

Use this for dispatch at module or platform level. It costs nothing per call.

```python
# Reference: `dispatch_unquantized_gemm` in vllm/model_executor/layers/utils.py
def dispatch_kernel():
    if envs.VLLM_{OP_NAME} and M_typical <= crossover_threshold:
        return optimized_fn
    return default_fn
```

### Priority Dispatch Chain (Overlapping Call Sites)

Use this for the rare case where 2+ gated optimizations share a call site:

```python
AMMO_DISPATCH_CHAIN = [
    # (condition, kernel_fn, name) -- first match wins
    (lambda M: 2 <= M <= 16, fused_qkv_fn, "op012"),
    (lambda M: 2 <= M <= 32, selective_fn, "op007"),
]

def ammo_dispatch(layer, x, weight, bias=None):
    M = x.numel() // x.shape[-1]
    for condition, kernel_fn, name in AMMO_DISPATCH_CHAIN:
        if condition(M):
            return kernel_fn(layer, x, weight, bias)
    return default_fn(layer, x, weight, bias)
```

## Stage 1 Baseline Reuse

The rule and the baseline artifact paths live in `validation-defaults.md`
§ E2E Baseline Reuse Requirement.

## Track Constraints

These constraints apply to the champion, including its kernel gates:

1. **All batch sizes.** Test every batch size in target.json. No exceptions. No cherry-picking.
2. **Production parity.** Per `references/validation-defaults.md` § Production Parity Requirement. NEVER use `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, or `VLLM_TORCH_COMPILE_LEVEL=0` to simplify the production comparator.
3. **vLLM baseline.** Per `references/validation-defaults.md` § Dual Baseline Requirement — never naive PyTorch.

## References

- `validation-defaults.md` — verdict thresholds (noise_tolerance_pct, catastrophic_regression_pct) and per-BS classification logic
- `crossover-probing.md` — crossover probing protocol for GATING_REQUIRED tracks
- `gpu-pool.md` — GPU reservation pattern
- `cudagraph-safety.md` — CUDA graph capture checklist
