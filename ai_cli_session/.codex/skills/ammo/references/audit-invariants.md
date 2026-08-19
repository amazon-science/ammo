# Audit Invariants

This is the Phase 2 coverage floor delivered to `ammo-auditor` only after its
independent Phase 1 reconstruction. It names semantic properties, not a frozen
row count. Mechanical validators own schema, enum, arithmetic, completeness,
and transition checks; verify their inputs, then challenge the meaning,
provenance, and unexplained discrepancies.

Apply **Pre-Check**, the assigned stage section, and **Cross-Artifact Checks**.
Record a non-applicable property with its reason. `BLOCKING` means the
stage claim is unsafe; `HIGH` means investigate and explain before acceptance.
Severity must scale with campaign impact: a finding that cannot change the
stage verdict or a shipped number is not `BLOCKING`, and a finding with no
material impact is reported as a note, not a repair-cycle trigger. The lead
rejects a `BLOCKING` rating that names no campaign-impact mechanism.
Phase 2 cannot erase a Phase 1 blocker.

## Pre-Check

- **Primary artifacts exist (BLOCKING).** Resolve canonical round-scoped paths
  from `state.json` and `artifact-layout.md`. A state pointer or rendered view is
  not evidence when its primary file is absent.
- **Mechanical validation is green (BLOCKING).** Run the applicable schema,
  transition, evidence, and validation-gate commands. Failed sweep legs, null or
  non-finite metrics, placeholders, partial buckets, or state/report disagreement
  invalidate the affected claim.
- **Identity and provenance agree (BLOCKING).** Target, round, `op_id`, source
  tree/patch, environment, workload, topology, and artifact timestamps describe
  the same run. Do not silently combine adjacent runs.
- **Frozen target is honored (BLOCKING).** Model revision, dtype, TP/DP/EP,
  compile/graph mode, batch/workload matrix, serving knobs, and comparator match
  the campaign contract unless an explicit authorized override exists.
- **Claims are cited (HIGH).** Non-trivial architecture, dispatch, phase, and
  performance assertions resolve to source, model config, trace, or runner data.

## After Stage 1 — `T_AUDIT_S1`

- **Clean timing is authoritative (BLOCKING).** Baseline timing comes from the
  profiler-free `baseline` slot with successful legs and the frozen workload.
  Profiling or golden-capture latency is never substituted.
- **Profiling is separate and representative (BLOCKING).** Nsys captures use the
  same production path and topology, cover the claimed phases/steps, and are
  clearly marked non-timing evidence. Rank/device coverage is sufficient for the
  topology.
- **Baseline values are physically and statistically plausible (HIGH).** Check
  units, iteration aggregation, variance/outliers, and scaling across workload
  buckets. Investigate implausible values rather than enforcing a universal
  latency constant.
- **Golden comparator is usable (BLOCKING).** The capture is complete for the
  declared correctness task and bound to the same baseline source/environment.
- **Constraints reflect evidence (BLOCKING).** Architecture facts match the
  actual model config; production dispatch, correctness invariants, and workload
  characterization match source and traces.

## After Stage 2 — `T_AUDIT_S2`

- **Phase accounting inputs are canonical (BLOCKING).** `scripts/mine_trace.py`
  owns the arithmetic and fails loud, so verify its inputs instead of
  recomputing `decode_busy`, the shares, closure, or per-row `f_e2e`:
  - `rounds/{N}/mining/mined.json` exists with schema `mine_trace/1`, and the
    pasted `tables.md` numbers in `bottleneck_analysis.md` match it. A
    hand-edited table is a blocker.
  - `config_echo` names the canonical round-scoped traces, arm, and
    `e2e_results`, and its `provenance` records the kernel table and nsys
    versions actually used. No FATAL is suppressed and every `warnings[]` entry
    is addressed in the artifact prose.
  - The delimiter policy is defensible: `separation_ratio` is not marginal and
    `first_kernel_after` fits a steady-state step boundary. State
    the capture's `--nsys-output-len` — a reduced-OSL capture can ground prefill
    attribution but not decode step counts or decode shares.
  - The family partition is semantically right: labels match their symbols,
    `partition_coverage` is complete, and any `residual_pct` is small and
    disclosed. `step_count_source` is the strongest rung the artifacts support.
- **Opportunity accounting uses `f_e2e` (BLOCKING).** Top components are ranked
  by production E2E share, not decode-only share. Their total does not exceed the
  measured phase budget; prefill-active and inter-kernel slices are not erased.
- **Bottlenecks map to production source (BLOCKING).** Kernel aliases, occurrence
  counts, phase membership, dispatch paths, and TP-rank behavior support each
  named bottleneck. Trace order overrides architecture intuition.
- **Bounds are physical, not candidate promises (HIGH).** Removable-work and
  hardware ceilings state their boundary and assumptions. Host gaps are not
  counted as recoverable without production-boundary evidence.
- **Mining remains proposal-neutral (HIGH).** The artifact supplies facts and
  bounds; it does not pre-select a mechanism or inflate attainable speedup.

## After Stages 4-5 — `T_AUDIT_S45`

- **Every selected track is terminal and reconciled (BLOCKING).** Typed state,
  `evidence.json`, generated summary, human verdict, monitor record, source
  commit, and gate artifacts identify the same track and outcome.
- **Correctness covers the production path (BLOCKING).** Applicable component
  tests and the binding full-model comparator pass at the declared precision.
  Lossy work includes its required task-quality evidence. A component test may be
  skipped only with a valid mechanism-specific reason; full-model correctness
  may never be skipped.
- **The optimized mechanism actually runs (BLOCKING).** Claim-appropriate
  profiler artifacts prove the expected production path, buckets, and fallback.
  Nsys is normally sufficient for activation/ordering; counters are required
  only for counter-dependent claims.
- **Clean E2E is paired and complete (BLOCKING).** Baseline and optimized arms
  use the same source identity where required, environment, venv, launch,
  topology, workload, cache policy, and profiler-free timing method. Apply the
  canonical per-bucket verdicts and significance rules.
- **Projection and observation are reconcilable (HIGH).** Recompute the
  occurrence-weighted, non-overlapping production-boundary estimate using
  `f_e2e`. A material gap requires a causal explanation; proxy magnitude cannot
  retroactively justify an E2E claim.
- **Gating is proved, not inferred (BLOCKING when used).** The implemented
  boundary, exact eligible set/threshold, fallback, rollback control, compile
  behavior, and pre/post results support `GATED_PASS`. Non-monotonic results use
  exact-set gating.
- **Diluted ship is independently re-derived (BLOCKING when used).** Recompute
  every current `validation-defaults.md` DILUTED_PASS condition from raw sweep
  JSON. The generated `diluted_pass` block, `diluted:true`, PASS status, phase
  significance, and campaign accounting must agree. Do not double-count its
  phase win as another E2E contribution.
- **Monitor findings are resolved (HIGH/BLOCKING by consequence).** Continuous
  coverage spans the implementer's lifetime, and unresolved accuracy, scope,
  provenance, or activation findings remain visible to the final audit.

## After Stage 6-7 — `T_AUDIT_S67`

For EXHAUSTED, skip SHIP-only properties but verify the terminal integration
record and structured exhaustion memory.

- **The measured composition is the shipped composition (BLOCKING).** Selected
  commits/patches, any independently reviewed conflict resolution, final source
  tree, accepted environment, and integration evidence describe one exact set.
  Failed, exploratory, or unselected flags are not promoted.
- **Combined validation is production-valid (BLOCKING).** Full-model
  correctness, clean paired E2E, all required buckets, and separate activation
  profiling pass for the exact composition. A single-track short-circuit is
  valid only when its accepted patch and environment are unchanged.
- **No interaction is hidden (BLOCKING).** Merge residue, dispatch ordering,
  graph/compile behavior, return shapes, flag collisions, gating interactions,
  and original fallbacks are reviewed. Individual PASS results do not prove a
  multi-track composition.
- **Bookkeeping derives from primary results (BLOCKING).** Integration status,
  selected/shipped identities, dilution annotations, commit, and environment
  agree across state and artifacts. Cumulative improvement is the direct Round-1
  baseline/latest-integrated ratio, never multiplied round-local gains.
- **Baseline continuity is sound (BLOCKING).** Round 1 remains immutable. After
  SHIP, fresh golden references are captured from the promoted mainline and
  accepted environment for the next round.
- **Continuation belongs to Stage 7 (BLOCKING).** Stage 6 finalized the current
  round only. Threshold comparison, terminal token, mining invalidation, and any
  next-round creation follow the Stage 7 state/transition authorities.
- **Exhaustion is specific (HIGH).** Failed component/technology/shape/failure
  tuples retain evidence and expiry. They inform the next debate without acting
  as a blanket technology ban.

## Cross-Artifact Checks

- **Causality.** Stage timestamps, artifact mtimes, source commits, profiler
  captures, and verdict writes form a plausible order. Later evidence cannot
  validate earlier code without explicit rebinding.
- **Same-run pairing.** Flag different sessions, worktrees, venvs, cache modes,
  or stale baseline reuse. Require a paired rerun when the difference can explain
  the measured effect.
- **No contaminated authority.** Profiler, warmup, debug, partial, failed, and
  archived runs remain diagnostic and cannot supply clean timing or final
  correctness.
- **Round-to-round continuity.** Shipped identities and accepted environment are
  monotonic and complete; regressions or drift are investigated from direct
  measurements, not smoothed away.
- **Independent reconstruction wins.** When a renderer, state summary, delegate,
  or report disagrees with primary evidence, preserve the disagreement, fix the
  producer, and re-audit. Do not choose the convenient copy.

## Mechanical Authorities

- schema and state transitions: `.codex/schemas/state.schema.json`,
  `scripts/transitions.json`, and `scripts/ammo_state.py`
- structured validation: `scripts/reconcile_track_state.py` and
  `scripts/verify_validation_gates.py`
- verdict arithmetic and special paths: `validation-defaults.md`
- E2E projection: `e2e-delta-math.md`
- canonical paths and evidence precedence: `artifact-layout.md`
