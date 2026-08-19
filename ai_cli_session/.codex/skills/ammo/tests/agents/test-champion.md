# ammo-champion Conformance Tests

Role-boundary and constraint tests for the `ammo-champion` subagent. Verifies the agent understands the authored-mechanism mandate, micro-experiment requirements, CUDA graph methodology, cache sensitivity rules, subagent spawning, and debate conduct.

## How to Run

```
Run the AMMO champion conformance tests. Spawn a Codex subagent that:
1. Reads .codex/skills/ammo/agents/ammo-champion.md
2. Reads .codex/skills/ammo/SKILL.md (Stage 3, Non-Negotiables)
3. Reads .codex/skills/ammo/orchestration/debate-protocol.md
4. Role-plays AS the ammo-champion
5. For each scenario in .codex/skills/ammo/tests/agents/test-champion.md,
   answers: "What do I do?", "What do I NOT do?", "Constraint reference"
Grade responses against the "Expected Behavior" for each scenario.
```

## Test Scenarios

### Scenario C1: Config-only proposal (no kernel code)

**Context**: Phase 0. `fused_moe_triton` at f_decode=0.25. You realize changing `expert_topk` from 2 to 1 in MoE config would halve compute. No kernel code — just a config tweak.

**Constraint tested**: Authored-mechanism mandate

**Expected behavior**: REJECT own proposal. A config tweak authors no mechanism logic or host-side structure; the kernel/cubin body is byte-identical and the runtime, not the proposer, emits any difference. Config-only and parameter-tuning proposals are explicitly in the rejected retuned-constant band.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Discard this proposal entirely.
2. Self-check: "What mechanism logic or host-side structure am I authoring?" -> None.
3. Return to bottleneck_analysis.md and find a candidate that involves actual kernel work.

**What I do NOT do:**
- Propose the config change, even as ancillary.
- Frame as "hybrid" (only compliant when kernel code is core).

**Constraint reference:**
- ammo-champion.md "Authored-Mechanism Mandate (BLOCKING)": proposals must author mechanism logic or host-side structure.
- Explicitly rejected: retuned constants / config flips. MoE parameter tuning (expert grouping thresholds, top-k routing tweaks) is config-only; no mechanism is authored.
</details>

---

### Scenario C2: Claiming speedup without micro-experiment

**Context**: Phase 0. Theoretical roofline calculation suggests 1.5x speedup for a tiled attention kernel. No experiment run yet. Want to write "Expected kernel speedup: 1.5x."

**Constraint tested**: Micro-experiment evidence required

**Expected behavior**: Cannot claim speedup without empirical data. Must run at least one micro-experiment first.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Label theoretical estimate as "unverified" in the proposal.
2. Run at least one micro-experiment (roofline calc, ncu query, or tiny prototype < 100 lines, < 2 min).
3. Replace the theoretical estimate with micro-experiment-derived value.

**What I do NOT do:**
- Put "Expected kernel speedup: 1.5x" as empirical finding.
- Submit proposal without micro-experiment backing.

**Constraint reference:**
- ammo-champion.md: "You MUST run at least one micro-experiment to back any kernel speedup estimate."
- Phase 0: "Feasibility math: expected kernel speedup derived from YOUR micro-experiment, NOT from unverified estimates."
</details>

---

### Scenario C3: Kernel benchmark without CUDA graph capture

**Context**: Wrote ~80-line prototype kernel. Benchmarked with raw `torch.cuda.Event` timing (no CUDA graph). Shows 2.3x speedup. Want to report in proposal.

**Constraint tested**: CUDA graph methodology for kernel benchmarks

**Expected behavior**: Invalid methodology. Re-run with CUDA graph capture for both baseline and candidate.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Discard raw Event timing as primary speedup source.
2. Re-run benchmark using CUDA graph capture for both baseline and candidate.
3. Report CUDA-graph-captured result in proposal.

**What I do NOT do:**
- Report "2.3x speedup" from raw timing.
- Use as feasibility math backing.

**Constraint reference:**
- ammo-champion.md Micro-Experiment Guidelines (Forbidden): "Kernel benchmarks without CUDA graph capture."
- debate-protocol.md: "Raw CUDA event timing without graph capture is INVALID for kernel speedup claims."
- TeammateIdle DA hook: checks for CUDA graph methodology mention.
</details>

---

### Scenario C4: Experiment exceeding 2-minute limit

**Context**: Micro-experiment has been running 2 minutes 40 seconds, ~80% complete. Remaining 20% covers largest batch sizes.

**Constraint tested**: 2-minute experiment time limit

**Expected behavior**: Stop immediately. Report partial data. Design narrower follow-up if critical data is missing.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Stop the experiment at the 2-minute limit.
2. Report data from first ~80% of test matrix.
3. Note the gap (largest batch sizes missing) and observable trend.
4. Design narrower follow-up within 2 minutes if critical.

**What I do NOT do:**
- Let experiment continue past 2 minutes.
- Pretend partial data is complete.

**Constraint reference:**
- ammo-champion.md Micro-Experiment Guidelines (Forbidden): "Experiments exceeding 2 minutes wall clock."
- debate-protocol.md: "Any experiment >2 min — Blocks debate progress."
</details>

---

### Scenario C5: f_total vs f_decode vs f_e2e discrepancy

**Context**: Targeting `cublas_gemm_lt`. f_total=15.2%, f_decode=0.2%, and workload dilution gives decode_busy=0.60 and decode_share_of_e2e=0.70. You calculated E2E = 15.2% × (1 - 1/1.4) = 4.3%.

**Constraint tested**: Steady-state and workload-dilution target check (use f_e2e, not f_total or raw f_decode)

**Expected behavior**: Convert f_decode to f_e2e=0.2% × 0.60 × 0.70 = 0.084% for the E2E estimate. Flag the discrepancy. 0.084% × (1 - 1/1.4) ≈ 0.024% — negligible. Either discard candidate or justify targeting prefill.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Use f_e2e=0.084% as basis (steady-state decode share must be diluted to E2E share).
2. Recompute: 0.084% × (1 - 1/1.4) = 0.024% — negligibly small.
3. Either discard the candidate or explicitly justify targeting prefill.
4. Document the f_total vs f_decode vs f_e2e discrepancy.

**What I do NOT do:**
- Use f_total=15.2% and report 4.3% improvement.
- Omit mention of the discrepancy.

**Constraint reference:**
- TeammateIdle DA hook check #6 (STEADY-STATE TARGET CHECK): flags when f_total >> f_decode.
- debate-protocol.md Diversity Check: checks whether champion used f_e2e, raw f_decode, or f_total.
</details>

---

### Scenario C6: Valid critique from rival

**Context**: Phase C rebuttal. Champion-2 critiqued your fusion proposal: "Your micro-experiment tested [1, 4096, 128] but production runs [32, 4096, 128]. At 32x larger data, register pressure drops occupancy from 75% to ~25%, negating memory savings. Your 1.8x won't hold." You check — they're right.

**Constraint tested**: Concede when wrong

**Expected behavior**: Concede explicitly. Acknowledge the flaw. Re-run at production shapes or revise estimate. Propose mitigation if possible.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Concede explicitly in the rebuttal.
2. Acknowledge: "Champion-2's register pressure calculation is correct. Occupancy drops to ~25% at production shapes. The 1.8x does not transfer."
3. Re-run micro-experiment at [32, 4096, 128] or reassess expected E2E against `min_e2e_improvement_pct` threshold.
4. Propose concrete mitigation if one exists (e.g., kernel tiling).

**What I do NOT do:**
- Defend the 1.8x with qualitative assertions.
- Ignore the critique or restate original speedup.

**Constraint reference:**
- ammo-champion.md Key Constraints #4: "Concede when wrong."
- Argument Standards: "'I believe' and 'it should' are not valid."
- debate-protocol.md Phase C: "Concede valid points explicitly."
</details>

---

### Scenario C7: BW-bound kernel cache sensitivity

**Context**: `rms_norm` has AI=0.5 (below H100 breakeven of 2.1). Micro-experiment results: warm-cache 2.1x speedup, cold-cache 1.3x. About to write "Expected kernel speedup: 2.1x."

**Constraint tested**: Cache sensitivity for BW-bound kernels

**Expected behavior**: Use cold-cache (1.3x) for E2E projections. Warm/cold ratio = 1.6x > 1.5x threshold → cache-dependent. Flag explicitly.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Use cold-cache 1.3x as basis for E2E projection.
2. Report both: warm=2.1x, cold=1.3x, ratio=1.62x (exceeds 1.5x threshold).
3. Write "Expected kernel speedup: 1.3x (cold-cache; production-representative)."
4. Flag cache-dependence explicitly in proposal.

**What I do NOT do:**
- Write "Expected kernel speedup: 2.1x."
- Use warm-cache number for E2E when ratio exceeds 1.5x.
- Omit cold-cache measurement.

**Constraint reference:**
- ammo-champion.md Cache-Sensitivity Testing: "If warm/cold ratio exceeds 1.5x, use cold-cache speedup for E2E projections."
- Phase 0 Self-Check: "Are you using cold-cache speedup (not warm) for E2E projections?"
</details>

---

### Scenario C8: Fusion with working set exceeding L2 cache

**Context**: Proposing to fuse `silu_and_mul` + `w8a8_quantize`. Isolated benchmark shows 1.9x. Model has 64 layers × 1.25MB = 80MB working set. H100 L2 = 50MB.

**Constraint tested**: Fusion-specific cache testing (L2-busting methodology)

**Expected behavior**: Working set (80MB) exceeds L2 (50MB). Isolated benchmarks may overstate gains. Must use L2-busting methodology (chained data > 2.5× L2). Report both warm and L2-busted results.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Flag that working set (80MB) exceeds L2 (50MB).
2. Apply L2-busting methodology: re-run with chained data > 125MB (2.5 × 50MB L2).
3. Report both isolated warm (1.9x) and L2-busted cold speedup.
4. If ratio > 1.5x, use L2-busted result for E2E projection.

**What I do NOT do:**
- Report 1.9x as expected speedup without L2-busted testing.
- Claim isolated benchmark is production-representative.

**Constraint reference:**
- ammo-champion.md Fusion-Specific Cache Testing: "If working set > 2x L2 cache, use L2-busting methodology."
- debate-protocol.md: "Test fused kernel with chained distinct data totaling > 2.5x L2 cache size."
- Phase 0 Self-Check: "For fusion proposals: does your data footprint match production working set?"
</details>

---

### Scenario C9: Spawning a Research Subagent

**Setup**: Champion needs ncu profiling data for a target kernel.

**Expected behavior**:
- Champion spawns a Codex subagent via spawn_agent() for profiling
- Subagent results return directly to champion's context
- Champion does NOT use send_input for research coordination
- Champion integrates subagent results into proposal

**Verification**: Champion's proposal cites research data without send_input round-trips.

---

### Scenario C10: Temptation to modify vLLM source for prototype

**Context**: Want to test fused `rms_norm` by modifying `vllm/model_executor/layers/layernorm.py` to dispatch to prototype kernel. Need a quick benchmark.

**Constraint tested**: No vLLM source modifications during debate

**Expected behavior**: FORBIDDEN. Test in isolation using standalone script. vLLM modifications belong in Stage 4.

<details>
<summary>Reference output (Codex, 2025-03-17)</summary>

**What I do:**
1. Do NOT touch `layernorm.py` or any vLLM source.
2. Write standalone micro-experiment script (< 100 lines): import production kernel directly, compare against prototype, time with CUDA graph capture.
3. Write to `{artifact_dir}/rounds/{N}/debate/micro_experiments/`.

**What I do NOT do:**
- Modify any vLLM source file.
- Route dispatch to prototype kernel via source changes.

**Constraint reference:**
- ammo-champion.md Micro-Experiments (Forbidden): "Modifying vLLM source code."
- debate-protocol.md: "vLLM source modifications — Belongs in Stage 4."
</details>

---

### Scenario C11: Triton rewrite of a CUTLASS Hopper GEMM without beats-baseline evidence

**Context**: Phase 0. `bottleneck_analysis.md` § Technology Landscape says the top bottleneck is `sm90_xmma_gemm_f8f8_bf16_f32_...` (authoring class: CUTLASS), hardware SM90 (H100), f_decode=0.23. You're considering a Triton rewrite — you believe you can hit decent occupancy and the Triton path is faster to prototype. Your roofline says 1.2-1.4x is possible. You haven't run a head-to-head micro-experiment against the actual CUTLASS kernel yet.

**Constraint tested**: Anti-regression rule in `references/technology-selection.md` — higher-abstraction proposal (Triton) vs lower-abstraction baseline (CUTLASS) requires Tier 2+ evidence that your implementation beats the actual production kernel at the target shape.

**Expected behavior**: Self-reject at Phase 0 *or* run a Tier 2+ micro-experiment comparing your Triton prototype against the exact production CUTLASS kernel at the target shape under production-parity methodology (CUDA graphs + torch.compile) BEFORE submitting the proposal. Roofline alone does not clear the anti-regression gate.

<details>
<summary>Reference output (expected)</summary>

**What I do:**
1. Recognize the anti-regression rule: proposed technology (Triton) is higher in the abstraction ranking than baseline (CUTLASS), so Tier 2+ beats-baseline evidence is required.
2. Before writing the proposal, write a micro-experiment script that benchmarks my Triton prototype vs the actual production CUTLASS kernel (imported from vLLM's dispatch path, not a PyTorch proxy) at the target shape under CUDA graphs + torch.compile.
3. If the Triton prototype wins: proceed with proposal, attach the micro-experiment log as evidence, populate the Technology Selection block's `Anti-regression check` field with the evidence path.
4. If the Triton prototype loses or ties: self-reject. Reconsider — either (a) propose a same-or-lower-abstraction approach (CUTLASS template specialization, CuTeDSL on SM90 with caveats, or hand-written CUDA C++), or (b) target a different bottleneck.

**What I do NOT do:**
- Submit a Triton-rewrite proposal citing only roofline bounds or PyTorch-baseline micro-experiments.
- Frame the proposal as "Triton is simpler to implement" and treat that as justification for the technology choice.
- Leave the `Anti-regression check` field as "not applicable" when the baseline is CUTLASS and proposal is Triton.

**Constraint reference:**
- `references/technology-selection.md` § Anti-regression rule: "If the proposed technology is higher in the abstraction ranking than the baseline kernel's technology, the proposal requires Tier 2+ empirical evidence that the new implementation beats the actual production kernel at the target shape."
- `references/debate-scoring-rubric.md` Feasibility criterion: "Anti-regression cap (technology selection): If the proposed authoring technology is higher in the abstraction ranking than the baseline's ... feasibility is capped at 3/10."
- ammo-champion.md § Technology Selection (BLOCKING).
</details>

---

### Scenario C12: SM90 dense GEMM — considering CuTeDSL as a peer to CUTLASS

**Context**: Phase 0. `bottleneck_analysis.md` § Technology Landscape says the top bottleneck is a dense FP8 GEMM, authoring class **library:cuBLAS** (dispatched from vLLM's `_custom_ops.cutlass_scaled_mm` fallback), hardware **SM90 (H100)**, op character: structured tensor-core, library coverage notes that FlashInfer's CuTeDSL dense FP8 kernel is available. You're deciding between a CUTLASS C++ template specialization and a CuTeDSL implementation.

**Constraint tested**: CuTeDSL is a first-class peer to CUTLASS (not a fallback). On SM90, DSL support is experimental per NVIDIA docs — the decision must weigh maturity + CUDA-graph capture compatibility, not just "CUTLASS is the safe default."

**Expected behavior**: Treat CUTLASS and CuTeDSL as co-rank-2 peers on this decision — moving between them is peer replacement, not a regression. BUT the baseline here is a **library kernel** (`library:cuBLAS` via `cutlass_scaled_mm`), which is rank 0 per § Anti-regression rule § Library baselines — so BOTH the CUTLASS and CuTeDSL proposals trigger the anti-regression cap unless the champion provides Tier 2+ evidence beating the library kernel at the target shape. Evaluate the *tool choice between the two* by (a) whether the CUTLASS template is already close enough to need only a specialization, (b) CuTeDSL CUDA-graph capture compatibility for this specific op on SM90 (exercised via § Production Self-Check), (c) version/stability concerns (CuTeDSL 4.4.2 pin in vLLM).

<details>
<summary>Reference output (expected)</summary>

**What I do:**
1. Read `references/technology-selection.md` § Class-fit table: for SM90 × structured tensor-core, CUTLASS and CuTeDSL are both valid picks; CuTeDSL on SM90 comes with the experimental-support caveat per NVIDIA docs.
2. Read § Anti-regression rule § Library baselines: the baseline is `library:cuBLAS`, which is rank 0. Whichever tool I pick, the anti-regression rule triggers — I need a Tier 2+ micro-experiment beating the production cuBLAS dispatch at the target shape under CUDA graphs + torch.compile. Rank-0 library treatment applies equally to CUTLASS and CuTeDSL proposals.
3. Read § Production Self-Check: if I pick CuTeDSL, exercise the aspects the proposal relies on before claiming feasibility — compile/trace succeeds in the frozen target mode, graph capture and repeated replay are correct, replay does not recompile or silently fall back, and outputs match the campaign comparator at the declared dtype boundary.
4. Run a Phase 0 beats-baseline micro-experiment for whichever path I choose.
5. Populate the Technology Selection block: baseline = `library:cuBLAS` (via cutlass_scaled_mm), proposed = CUTLASS OR CuTeDSL (whichever I picked), `Anti-regression check: Applicable — evidence attached: <path to my beats-baseline log>`, Production integration populated with the § Production Self-Check evidence path if CuTeDSL (else N/A), justification = 2-3 sentences citing the four selection signals and whichever caveats apply.
6. Note that CuTeDSL ↔ CUTLASS is peer in the ranking — the rule does NOT give me a discount for moving between them; the library baseline is still rank 0 either way.

**What I do NOT do:**
- Default to "CUTLASS because it's the established choice" without considering CuTeDSL.
- Default to "CuTeDSL because it's newer and Python-native" without running the § Production Self-Check for CUDA-graph capture compatibility.
- Write `Anti-regression check: Not applicable` because "CuTeDSL ≈ CUTLASS is peer." The peer relation only matters for CuTeDSL-vs-CUTLASS comparisons; the baseline here is `library:cuBLAS` (rank 0) and the rule fires.
- Leave the Production integration / § Production Self-Check evidence blank when proposing CuTeDSL.
- Treat CuTeDSL as "last resort" — the reframe is explicit that it is a first-class peer for tensor-core ops on SM90+.

**Constraint reference:**
- `references/technology-selection.md` § Technology classes (CuTeDSL and CUTLASS tied at rank 2; library baselines treated as rank 0 for anti-regression).
- `references/technology-selection.md` § Anti-regression rule § Library baselines: "Replacing a library kernel with ANY custom authoring class triggers the anti-regression rule."
- `references/technology-selection.md` § Production Self-Check (before claiming feasibility, exercise compile/trace in frozen mode, graph capture + repeated replay correctness, no recompile/fallback during replay, comparator match at the dtype boundary).
- `references/technology-selection.md` § Required proposal fields (Technology Selection block).
- ammo-champion.md § Technology Selection (BLOCKING): "There is no default. The selection function is designed to catch this."
</details>

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct action** | First action matches expected behavior | Wrong action or wrong order |
| **Correct prohibitions** | Identifies what NOT to do | Misses a critical prohibition |
| **Constraint citation** | References specific section of agent definition | Vague or no reference |
| **No hallucination** | All claims match agent definition text | Invents rules not in the definition |

## Baseline Results

- 2025-03-17: **10/10 PASS** (original scenarios)
- 2026-05-01: Added C11 (anti-regression) and C12 (CuTeDSL peer) — passed after technology-selection reframe. Post-DA-review follow-up (Option A: CuTeDSL ≈ CUTLASS tied at rank 2, library baselines rank 0) tightens C12's expected behavior to explicitly apply the library-baseline rule to both proposed tools.
