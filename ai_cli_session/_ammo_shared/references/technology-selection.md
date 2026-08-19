# Technology Selection

This reference governs how a proposal justifies the mechanism it will implement.
Read it before you write a proposal. It deliberately gives no permanent
technology ranking — compiler, library, and hardware support change faster than
AMMO's guidance — so every rule here tests capability instead of rank.

## How to Choose a Mechanism

Pick the lowest-risk mechanism that can express your algorithm and has a
credible path to beating the **actual production implementation** on the frozen
target. Technology is a reasoned design choice, not a deterministic lookup.

Evaluate these five facts:

1. **Production baseline.** Identify the dispatched implementation from trace
   symbols and source. A framework reference or a plausible kernel name is not
   enough.
2. **Required hardware behavior.** State which features or resource controls the
   mechanism needs, and confirm them on the deployed GPU when they are material.
3. **Operation fit.** Say which class the work belongs to — structured
   tensor-core, irregular/control-flow, host scheduling, communication, layout,
   or another — and say why the mechanism expresses that class safely.
4. **Existing coverage.** Search current production libraries and the source
   tree. Prefer extending a mature implementation when that preserves its tuned
   core, and justify any rewrite against that baseline.
5. **Production integration.** Show how your path takes part in the target's
   compile, graph, dtype, dispatch, fallback, and rollback contracts.

Kernel DSLs, CUDA/C++, library extensions, compiler passes, and Python/C++ host
structure are all valid mechanisms. Name yours with the most precise current
name. The schema, not this prose, owns any machine enum.

## Beats-Baseline Rule

A replacement that gives up capabilities a mature production kernel, library, or
runtime path already has is a high-risk rewrite. It needs production-boundary
empirical evidence that it beats that actual baseline at the target shape. A
proxy, an eager-only harness, a framework reference, or a roofline bound does
not satisfy this rule.

If you do not have that evidence, disclose the gap honestly: the candidate stays
eligible, but only under the probability ceiling in `debate-scoring-rubric.md`.
A declaration that contradicts itself fails eligibility instead.

Extending the existing production mechanism without replacing its tuned core is
not automatically a high-risk rewrite, but it still needs ordinary feasibility
evidence.

This rule is capability-based. Do not infer it from a timeless ordering such as
"Triton above CUTLASS", and do not infer it from language choice alone.

## Production Self-Check

Exercise the aspects your proposal relies on before you claim feasibility:

- compile/trace succeeds in the frozen target mode;
- graph capture and repeated replay are correct when the production path uses
  graphs;
- replay does not recompile or silently choose the fallback;
- outputs match the campaign comparator at the declared dtype boundary;
- the optimized and original paths activate only in their intended domains.

Match the profiler to the claim. Nsys normally establishes dispatch, ordering,
occurrence, and activation. Reach for counters or another profiler only when the
claim depends on hardware quantities Nsys does not expose.

## Required Proposal Block

Every Phase 0 proposal includes:

```markdown
## Technology Selection
- Baseline technology: <measured production implementation + evidence>
- Proposed technology: <kernel, library, compiler, or host mechanism>
- Hardware requirements: <required features/resources and target support>
- Operation fit: <why this mechanism fits the work>
- Existing coverage: <nearest mature implementation and search evidence>
- Production integration: <compile/graph/dispatch/fallback implications>
- Justification: <why this is the best current risk/performance tradeoff>
- Beats-baseline check: <not triggered, evidence path, or honest evidence gap>
```

Missing, empty, or contradictory fields fail the Phase 0 gate. The lead checks
that the evidence pointers resolve, but the lead does not substitute a canned
technology preference for the champion's technical reasoning.

## Exhaustion Across Rounds

Read the structured `exhausted_technologies` entries that match the current
component, shape, failure mode, and expiry. A prior failure is evidence, not a
universal ban: you may reuse a technology when the algorithm, mechanism, target,
or invalidating evidence changed materially and your proposal says how.

The lead's soft diversity check requires the next debate to show progress
instead of repeating the same failed bet with no new reasoning. It does not force a
technology ladder, and it does not suppress a component whose other mechanisms
stay viable.

## Authorities

- eligibility and probability ceilings: `debate-scoring-rubric.md`
- evidence scope and experiment limits: `debate-rules.md`
- graph/compile contracts: `cudagraph-safety.md` and
  `torch-compile-contract.md`
- production-path and fallback rules: `impl-track-rules.md`
