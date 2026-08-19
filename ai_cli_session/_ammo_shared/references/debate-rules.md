# Debate Evidence Rules

Every Stage 3 observation must answer two questions: how much can we trust
this number (its tier), and which part of production does it cover (its
scope)? A favorite tool or a tidy experiment recipe is not evidence by itself.

## Execution Confidence

The tier says how the number was produced, and it caps the P-score.

| Tier | Meaning | Minimum durable evidence | P-score cap |
|---|---|---|---|
| `tier_1` | analysis or unexecuted bound | cited calculation/source | 3/10 |
| `tier_2` | measured execution | reproducible script/command, log, target identity | 7/10 |
| `tier_3` | measured hardware claim | tier-2 evidence plus claim-relevant profiler output | no cap |

The state validator enforces `p_score <= cap(tier)`. A script with no
execution log is tier 1. Claim tier 3 only when candidate success depends on a
hardware quantity the profiler measures; an NCU run added only to lift the
tier strengthens nothing.

## Evidence-Scope Ladder

The scope says which production boundary the number covers. Every magnitude
declares exactly one schema scope.

| Scope | Boundary | May size pre-implementation EV? |
|---|---|---|
| `bound` | conservative zero-cost removal from baseline trace; host gaps remain non-recoverable | yes |
| `proxy` | standalone, reconstructed, synthetic, or identity-unverified harness | no |
| `production_boundary` | A/B at a declared boundary in the actual production path | yes |
| `clean_e2e` | clean production sweep | final ship signal |

A `proxy` can show feasibility or an upper bound, never production E2E
magnitude. An open objection that the measurement does not cover the projected
boundary makes the candidate `EV=0` — not a confidence deduction. Only the
lead or the critic who raised the objection may close it, and only with a
cited measurement of the declared boundary.

## Claim-Driven Experiments

Run the smallest bounded experiment that settles a load-bearing uncertainty.
Permitted: arithmetic/roofline analysis, source or ISA inspection, layout
analysis, a small kernel prototype, a production-boundary benchmark,
claim-appropriate profiler capture. Stage 3 never modifies vLLM source,
downloads model weights, or runs a clean full-model ship benchmark.

Hold the conditions the claim depends on: production API and dispatch,
shapes/dtypes/layouts, compile/graph mode, working set, and enclosing pipeline
costs. Time GPU intervals with CUDA events; time host dispatch with host
timestamps or NVTX+Nsys. A cache or multi-layer experiment is required exactly
when cache residency or repeated pipeline behavior is load-bearing. Report the
buckets that argue against you as well as the favorable ones.

If no experiment is needed or affordable, declare the evidence gap (explain why
the gap exists & provide evidence; critic can block) and keep the tier and scope
ceiling that already applies. Do not do implementation-grade work just to fill a
template.

## Durable Evidence

Every experiment actually run leaves artifacts under
`rounds/{CR}/debate/micro_experiments/`:

- the runnable script or the exact command;
- a log with device/runtime identity, inputs, method, iterations, units, and
  result;
- profiler output, only when the claim depends on it.

The proposal cites these artifacts and says which values are measured, which
are bounds, and which are assumptions. Missing execution evidence makes the
claim tier 1, however production-like the code looks.

## Production Baseline Provenance

A production-boundary kernel claim must call the same production entry point,
with the shapes, dtypes, strides, layouts, and graph/compile mode that decide
dispatch. Bind the claim to Stage 2 with profiler evidence that fits the
claim: Nsys normally proves kernel/path identity, occurrence, ordering, and
launch context; NCU or other counters only for claims about registers,
occupancy, achieved bandwidth, or cache behavior.

Investigate any material identity or throughput discrepancy; never explain it
away as noise. A harness whose production dispatch you cannot confirm stays
`proxy`.

## Projection and Dismissal

Project with `e2e-delta-math.md` and the measured addressable share
`f_e2e × removable_fraction`. `f_decode` is diagnostic only.

Do not dismiss an opportunity because one proxy failed, a utilization number
sounds high, or a technology previously failed. Dismiss only when the
admissible bound cannot clear the campaign floor, or the evidence closes the
relevant mechanism space. Record what that conclusion covers in structured
exhaustion memory. An independent critique must challenge any dismissal that
materially changes candidate coverage.

## Authorities

- projection and slice admissibility: `e2e-delta-math.md`
- probability and selection: `debate-scoring-rubric.md`
- technology risk: `technology-selection.md`
- GPU ownership: `gpu-pool.md`
