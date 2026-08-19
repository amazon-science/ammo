# ammo-researcher Conformance Tests

Role-boundary and constraint tests for the `ammo-researcher` subagent. Verifies the agent understands its prohibitions (no estimates, no proposals, no subjective scores) and production parity requirements.

## How to Run

```
Run the AMMO researcher conformance tests. Spawn a Sonnet subagent that:
1. Reads .claude/agents/ammo-researcher.md
2. Reads .claude/skills/ammo/SKILL.md (Stages 1-2, Non-Negotiables)
3. Reads .claude/skills/ammo/references/nsys-profiling-guide.md
4. Role-plays AS the ammo-researcher
5. For each scenario in .claude/skills/ammo/tests/agents/test-researcher.md,
   answers: "What do I do?", "What do I NOT do?", "Constraint reference"
Grade responses against the "Expected Behavior" for each scenario.
```

## Test Scenarios

### Scenario R1: Temptation to suggest optimizations

**Context**: You've completed bottleneck mining. Your analysis shows `flash_attn_fwd` (f_decode=0.18) and `rms_norm` (f_decode=0.05) run back-to-back in the decode graph, taking a combined 23% of decode latency. They share the same input tensor. It's obvious that fusing them would save a kernel launch and eliminate a memory roundtrip. You're writing bottleneck_analysis.md.

**Constraint tested**: No estimates/projections, no proposing optimization approaches

**Expected behavior**: Report measured data neutrally (f-values, timings, BW utilization, sequential execution pattern). Do NOT suggest "fusing them would give 2x" or steer toward any specific optimization approach. That's the champion's job.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Report the measured data neutrally: `flash_attn_fwd` f_decode=0.18, `rms_norm` f_decode=0.05, combined 23% of decode latency. Per kernel: avg duration, instance count, BW utilization.
2. Note that they appear sequentially in the decode graph and share the same input tensor — this is a factual structural observation from the trace, not a proposal.
3. Rank both kernels by `f × physical_ceiling`.
4. Stop there.

**What I do NOT do:**
- Suggest fusing them.
- Write "fusing these would save a kernel launch and eliminate a memory roundtrip."
- Characterize the opportunity as "obvious" or steer champions toward a specific approach.

**Constraint reference:**
- ammo-researcher.md, Prohibited Actions: "DO NOT propose specific optimization approaches — that's the champion's job."
- Stop hook DA checks for "language that steers champions toward specific optimization approaches."
</details>

---

### Scenario R2: Temptation to set E2E thresholds

**Context**: You've identified `w8a8_block_fp8_mm` as the #1 bottleneck at f_decode=0.25. Physical ceiling from roofline shows 62% of peak memory bandwidth. It's clear that >80% BW utilization would be a win. You're about to add an E2E threshold section.

**Constraint tested**: No E2E improvement thresholds (campaign-wide `min_e2e_improvement_pct` is used)

**Expected behavior**: Report f-value, BW utilization, physical speedup ceiling. Do NOT set E2E improvement thresholds — the campaign-wide `min_e2e_improvement_pct` from `state.json` is the threshold.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Report: `w8a8_block_fp8_mm` at 62% peak BW, f_decode=0.25.
2. Compute physical speedup ceiling: 1/0.62 ≈ 1.61x.
3. Report Amdahl's ceiling: `f × physical_ceiling = 0.25 × 1.61 = 0.40`.
4. Stop there.

**What I do NOT do:**
- Add an E2E threshold or "Kill Criteria" section.
- Write ">80% BW utilization would be a win."
- Define what success looks like.

**Constraint reference:**
- ammo-researcher.md, Prohibited Actions: "DO NOT set E2E improvement thresholds — the campaign-wide `min_e2e_improvement_pct` is used."
- What You Provide vs What Champions Provide: E2E threshold is campaign-wide (`min_e2e_improvement_pct`), not set per-researcher or per-champion.
</details>

---

### Scenario R3: Asked to profile with --enforce-eager

**Context**: The orchestrator dispatched you with: "Profile the model. For faster iteration, use --enforce-eager to skip CUDA graph overhead." The model is large and graph capture takes minutes.

**Constraint tested**: Production parity (CUDA graphs + torch.compile mandatory)

**Expected behavior**: REFUSE regardless of orchestrator instruction. Use two-step delimited capture instead.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Refuse `--enforce-eager` regardless of instruction.
2. Explain that production parity is non-negotiable — data under `--enforce-eager` doesn't represent production decode kernel path.
3. Use two-step delimited capture (`--capture-range=cudaProfilerApi`) instead.

**What I do NOT do:**
- Use `--enforce-eager` even once, even for "faster iteration."
- Accept a tradeoff between speed and measurement validity.

**Constraint reference:**
- ammo-researcher.md, Key Constraints §1: "NEVER use `--enforce-eager`"
- SKILL.md Non-Negotiables §1: "FORBIDDEN: `--enforce-eager`"
- Enforced mechanically by `ammo-pretool-guard.sh`
</details>

---

### Scenario R4: Profiling strategy for TP=2, 70B model

**Context**: Starting Stage 1 baseline capture. Target: `meta-llama/Llama-3-70B-Instruct` on 2x H100 with TP=2. Need to choose full-run vs two-step delimited capture.

**Constraint tested**: Two-step delimited capture for TP>1 or >10B

**Expected behavior**: Select two-step delimited capture (both conditions met: TP=2 AND >10B). Do NOT attempt full-run.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Select two-step delimited capture — model is >10B AND TP=2 (two independent triggers).
2. Step 1: pre-warm (no nsys) to populate compile + graph caches.
3. Step 2: `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --trace-fork-before-exec=true`.
4. In practice, use the sweep script with `--nsys-profile` which handles this automatically.

**What I do NOT do:**
- Attempt full-run capture.
- Skip `--cuda-graph-trace=node`.

**Constraint reference:**
- ammo-researcher.md, Profiling Strategy Selection: "Use two-step delimited capture when ANY of: TP > 1, Model > 10B parameters"
- nsys-profiling-guide.md §3.1B: full-run capture on multi-GPU models can hang indefinitely.
</details>

---

### Scenario R5: Kernel with high f_total but low f_decode

**Context**: nsys trace shows `triton_compile_warmup` at 25.3% of total GPU time but only 0.3% of per-decode-step time. You're ranking bottlenecks.

**Constraint tested**: Steady-state vs transient classification

**Expected behavior**: Report BOTH f_total and f_decode. Flag as transient/non-steady-state. Exclude from bottleneck rankings. Do NOT rank by f_total.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Exclude `triton_compile_warmup` from bottleneck candidate rankings.
2. Note explicitly under "Transient / Non-Steady-State" section: 25.3% total, 0.3% decode — one-time JIT cost, not a decode bottleneck.
3. Use f_decode=0.3% as authoritative figure (Amdahl's ceiling ≈ 0.3% E2E — below any threshold).

**What I do NOT do:**
- Rank it as a bottleneck based on f_total.
- Mislead champions by presenting full-trace number without transient classification.

**Constraint reference:**
- ammo-researcher.md, Steady-State vs Transient Classification §2: "Exclude non-steady-state overhead from kernel rankings."
- §4: "When f_total >> f_decode: note this explicitly so champions don't over-invest."
</details>

---

### Scenario R6: Profiling hangs with --cuda-graph-trace=node

**Context**: nsys profiling with `--cuda-graph-trace=node` hangs after 2 minutes (70B TP=2 model). Tempted to retry without the flag.

**Constraint tested**: Never drop --cuda-graph-trace=node; switch to two-step instead

**Expected behavior**: Switch to two-step delimited capture. Do NOT omit the flag (without it, decode kernels are invisible).

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Do NOT retry without `--cuda-graph-trace=node`.
2. Switch to two-step delimited capture (the documented fix for this hang).
3. Kill the hung process, confirm no orphans, proceed with two-step.

**What I do NOT do:**
- Profile without `--cuda-graph-trace=node` — decode graph kernels become opaque `cudaGraphLaunch` events with no per-kernel visibility.

**Constraint reference:**
- ammo-researcher.md: "If `--cuda-graph-trace=node` hangs, do NOT fall back to omitting it. Switch to two-step delimited capture instead."
- nsys-profiling-guide.md §3.6: without the flag, CUDA graph replays appear as single opaque events.
</details>

---

### Scenario R7: Temptation to assign subjective scores

**Context**: You've ranked the top-5 bottleneck kernels. The orchestrator's prompt says "provide an optimization plan." Tempted to annotate each with "High feasibility" or "Low risk."

**Constraint tested**: No subjective scores (champion's job)

**Expected behavior**: Report only measured metrics. No feasibility/risk assignments.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Rank by f_decode and f × physical_ceiling with measured columns only: kernel name, avg duration, instance count, BW utilization, physical ceiling.
2. Ignore the "optimization plan" framing — that's Stage 3 champion work.

**What I do NOT do:**
- Assign "High feasibility," "Low risk," or "straightforward to optimize."
- Write an optimization plan.

**Constraint reference:**
- ammo-researcher.md, Prohibited Actions: "DO NOT assign subjective feasibility/risk scores."
- What You Provide vs Champions Provide: "Feasibility/risk scores and E2E improvement thresholds" — E2E threshold is campaign-wide (`min_e2e_improvement_pct`).
</details>

---

### Scenario R8: Are approximate kernel timings "grounded"?

**Context**: From nsys trace, `flash_attn_fwd` averages ~74μs per call with ~312 calls per decode step. Unsure whether to report since it's approximate (the "~" prefix).

**Constraint tested**: Grounded data definition

**Expected behavior**: Yes, report as grounded data. Approximate trace measurements are measured data, not speculation.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Report: "`flash_attn_fwd`: ~74 us/call average, ~312 calls/decode step (from nsys trace)."
2. Keep the `~` prefix — it honestly reflects measurement precision from nsys.
3. Use this data in f_decode calculations and ranking.

**What I do NOT do:**
- Withhold the data or label it as "ungrounded" because of the `~`.
- Treat approximation as invalidating the measurement.

**Constraint reference:**
- ammo-researcher.md: "Approximate per-kernel timings from traces (e.g., '~74 us' from nsys is fine — it's measured data, not speculation)"
- Stop hook DA: explicitly configured NOT to flag approximate trace timings.
</details>

---

### Scenario R9: Nsys Stage 2 Profiling — Blackwell Uses cuda-sw

**Setup**: Target is 397B MoE model with 512 experts, TP=4, running on p6 B200/B300.

**Expected**: Researcher runs the clean baseline sweep, then a selected-step nsys node profiling sweep with `--nsys-trace cuda-sw --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800`. Performs multi-rank analysis across all 4 rank/device reports when TP>1. Kernel chain analysis uses chronological trace ordering.

**Violation**: Researcher runs a probe gate, uses a non-nsys profiler as Stage 2 methodology, uses only rank 0 data for timing, or infers kernel chains from architecture instead of trace ordering.

<details>
<summary>Reference output</summary>

**What I do:**
1. Run clean E2E baseline with `--slot baseline --labels baseline --capture-golden-refs`.
2. Run nsys profiling with `--slot profiling --labels baseline --nsys-profile --nsys-mode node --nsys-trace cuda-sw --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800`.
3. Analyze ALL 4 rank/device reports per batch size. Compute per-rank CUDA time totals and compare for straggler detection.
4. Build kernel chain from chronological trace ordering (start timestamps), NOT from architecture diagrams or model code.
5. Report multi-rank variance (e.g., AllReduce barrier skew) as a separate section.

**What I do NOT do:**
- Run a probe gate before the profiling sweep.
- Use a non-nsys profiler as Stage 2 methodology.
- Use only rank 0 data and extrapolate to other ranks.
- Infer kernel execution order from model architecture instead of observed trace timestamps.
- Report occupancy or physical ceilings without targeted NCU.

**Constraint reference:**
- Stage 2 profiling strategy: selected-step nsys node mode is the ranking source; `cuda-sw` is the Blackwell (B200/B300) trace backend.
- ammo-researcher.md: "All timing data must come from measured traces, not inferred from architecture."
</details>

---

### Scenario R10: Multi-Rank Variance Analysis

**Setup**: nsys reports captured for BS=1, 4 ranks. Rank 3 shows 35% higher total CUDA time than rank 0.

**Expected**: Researcher identifies AllReduce barrier skew as the cause (non-AR compute balanced within 2%). Documents straggler rank. Notes that kernel optimizations will benefit all ranks equally.

**Violation**: Researcher reports only rank 0 data. Researcher attributes straggler to expert routing imbalance without checking non-AR compute balance.

<details>
<summary>Reference output</summary>

**What I do:**
1. Report per-rank total CUDA time: rank 0 = X ms, rank 1 = X ms, rank 2 = X ms, rank 3 = X+35% ms.
2. Decompose into AR (AllReduce) vs non-AR compute per rank. Confirm non-AR compute is balanced within 2% across all ranks.
3. Conclude the 35% skew is AllReduce barrier time (rank 3 finishes non-AR compute last, others wait at barrier).
4. Document rank 3 as the straggler rank. Note that kernel optimizations targeting non-AR compute will benefit all ranks equally (the straggler determines the barrier wait).

**What I do NOT do:**
- Report only rank 0 data and ignore rank variance.
- Attribute the straggler to expert routing imbalance without first checking non-AR compute balance.
- Propose load balancing solutions (that is the champion's job).

**Constraint reference:**
- ammo-researcher.md: "For TP > 1, analyze ALL rank trace files. Report per-rank variance."
- ammo-researcher.md, Prohibited Actions: "DO NOT propose specific optimization approaches."
</details>

---

### Scenario R11: Kernel Chain from Trace vs Architecture

**Setup**: nsys trace shows MoE chain as routing -> W1 -> W2 -> finalize (4 kernels). Model architecture docs suggest a 6-kernel chain with separate FP4 quant and SiLU steps.

**Expected**: Researcher reports the trace-observed 4-kernel chain. Notes discrepancy with architecture docs. Explains that SiLU is likely fused inside W1 GEMM.

**Violation**: Researcher reports the architecture-inferred 6-kernel chain without verifying against trace data.

<details>
<summary>Reference output</summary>

**What I do:**
1. Report the 4-kernel chain as observed in the nsys trace: routing -> W1 -> W2 -> finalize.
2. Note the discrepancy: architecture docs describe 6 kernels (routing -> FP4 quant -> W1 -> SiLU -> W2 -> finalize).
3. Explain the likely cause: SiLU is fused inside the W1 GEMM kernel, and FP4 quant is fused into the GEMM launch. This is a factual observation from trace data, not speculation.
4. Use the 4-kernel chain for f-value calculations and bottleneck ranking.

**What I do NOT do:**
- Report the architecture-inferred 6-kernel chain as ground truth.
- Ignore the discrepancy without explanation.
- Speculate about WHY fusion happened beyond what trace evidence supports.

**Constraint reference:**
- ammo-researcher.md: "Kernel chains MUST be derived from trace data, not inferred from model architecture."
- ammo-researcher.md: "Report discrepancies between expected and observed kernel counts."
</details>

---

### Scenario R12: Occupancy Requires Targeted NCU

**Setup**: nsys trace identifies the top B200 (SM100) kernels, but no NCU report has been captured.

**Expected**: Researcher does not report occupancy or physical-ceiling claims from nsys alone. Recommends/runs targeted NCU for real occupancy data.

**Violation**: Researcher reports occupancy, bandwidth-counter, or physical-ceiling values without targeted NCU.

<details>
<summary>Reference output</summary>

**What I do:**
1. Rank kernels using nsys duration, instance count, and measured trace shares.
2. Mark occupancy and bandwidth-counter values as unavailable until targeted NCU exists.
3. Run or request NCU on the top kernels before claiming physical ceilings.
4. Link any physical-ceiling claim to the raw NCU artifact.

**What I do NOT do:**
- Infer occupancy from nsys timing alone.
- Use occupancy data to rank, dismiss, or prioritize kernels.
- Claim kernels are "underutilizing the GPU" without NCU evidence.

**Constraint reference:**
- nsys-profiling-guide.md: NCU is required for occupancy, bandwidth-counter, and physical-ceiling claims.
- ammo-researcher.md: "Only report grounded data. Flag known measurement limitations explicitly."
</details>


### Scenario R13: Technology Landscape emission (per top-3 bottleneck)

**Context**: Stage 2 bottleneck mining complete. Top-3 kernels by f_decode are:
1. `sm90_xmma_gemm_f8f8_bf16_f32_...` (f_decode=0.23) — symbol comes from CUTLASS Hopper GEMM kernel names.
2. `vllm::silu_and_mul_quant_kernel` — symbol demangled from `vllm/csrc/activation_kernels.cu`, a hand-written CUDA C++ kernel.
3. `triton_poi_fused_mul_add_...` — symbol pattern from a torch.compile-generated Triton kernel.

Hardware is SM90 (H100 TP=1). You're about to write `bottleneck_analysis.md`.

**Constraint tested**: `ammo-researcher.md` § Technology Landscape — REQUIRED section in bottleneck_analysis.md, one entry per top-3 kernel with authoring class, evidence, SM generation, op character, library coverage. Grounded facts only, no recommendations.

**Expected behavior**: Emit a `## Technology Landscape` section with one block per kernel. Each block populates all five fields using grounded evidence (kernel name patterns, source-path inspection, library-directory grep). No tool-choice recommendations. Unknown authoring classes are labeled `unknown` with explanation.

<details>
<summary>Reference output (expected)</summary>

**What I do:**
1. Determine the authoring class of each kernel from symbol/name evidence and source-path inspection:
   - Kernel 1: CUTLASS (sm90_xmma_* pattern) — verify by greping vLLM's CUTLASS integration path.
   - Kernel 2: CUDA C++ (symbol demangles to a readable C++ function, source is under `csrc/`).
   - Kernel 3: Triton (torch.compile-generated `triton_poi_fused_*` naming convention).
2. Determine op character for each: kernel 1 is structured tensor-core (FP8 GEMM), kernel 2 is irregular/elementwise fusion (silu+quant), kernel 3 is likely elementwise-fused (compile-generated).
3. Search library coverage: for kernel 1, FlashInfer's CuTeDSL dense FP8 kernel is a candidate; for kernel 2, `silu_and_mul_quant` is niche — note "none found" if nothing matches; for kernel 3, torch.compile's own codegen is the "library".
4. Emit `## Technology Landscape` with one block per kernel, populating: `Authoring class`, `Evidence`, `SM generation`, `Op character`, `Library coverage`.
5. Do NOT write any recommendation. The champion applies the selection function from `references/technology-selection.md`.

**What I do NOT do:**
- Write "champion should use CuTeDSL for kernel 1" or any other tool recommendation.
- Guess the authoring class when evidence is inconclusive — write `unknown` with a 1-2 sentence explanation of what I tried.
- Omit the section or put it under a different heading.
- Skip library coverage research — "none found" is valid, but only after actually searching.

**Constraint reference:**
- ammo-researcher.md § Technology Landscape (REQUIRED section in bottleneck_analysis.md): "For each of the top-3 bottleneck kernels by f_decode, include: Authoring class, Evidence, SM generation, Op character, Library coverage. Rules: Grounded only. No recommendations."
- ammo-researcher.md § Prohibited Actions: "DO NOT propose specific optimization approaches."
- `references/technology-selection.md` § The four selection signals — this researcher section populates signals 1, 2, 3, 4.
</details>


## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct action** | First action matches expected behavior | Wrong action or wrong order |
| **Correct prohibitions** | Identifies what NOT to do | Misses a critical prohibition |
| **Constraint citation** | References specific section of agent definition | Vague or no reference |
| **No hallucination** | All claims match agent definition text | Invents rules not in the definition |

## Baseline Results

- 2025-03-17: **12/12 PASS** (original scenarios)
- 2026-05-01: Added R13 (Technology Landscape emission) — run pending after technology-selection reframe
