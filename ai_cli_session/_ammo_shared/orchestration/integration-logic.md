# Stage 6: Integration Validation

Stage 6 answers one question: which accepted track results can ship together?
It then validates the exact composition you select and promotes only that
composition into the campaign baseline. It never reopens a failed track and
never creates the next round. Read this before you decide anything in Stage 6.

Read the current round from `state.json`. Reconcile every track against its
primary Gate 5 artifacts before you decide. Every track must be terminal under
the state schema, and `T_AUDIT_S45` must have passed. The schema owns field
shapes and status enums. `scripts/transitions.json` owns legal transitions.
`references/artifact-layout.md` defines artifact locations.

## Four Branches

Accepted means `PASS` or `GATED_PASS`. Rejected means `FAIL`. `diluted:true` is
evidence about measured impact, not a separate integration status. Use
`references/validation-defaults.md` for verdict semantics.

1. **No accepted track:** finalize the round as EXHAUSTED. Record why each track
   was rejected, run the Stage 6-7 audit, and enter Stage 7.
2. **Exactly one unchanged accepted track:** use the single-pass short-circuit.
   Its Stage 5 production correctness, activation, and clean E2E result stay
   authoritative only if the patch and the declared environment are
   byte-for-byte unchanged.
3. **Multiple compatible accepted tracks:** put their exact validated commits
   and environment settings on an integration branch, then run the full
   integration validation below. Per-track evidence cannot establish the
   behavior or the performance of a composition.
4. **Overlap, gating interaction, merge conflict, or ambiguous compatibility:**
   judge compatibility and dominance from evidence. File overlap is a warning,
   not proof of incompatibility; disjoint files are not proof of independence.
   If a conflict or a semantic interaction needs code resolution, spawn
   `ammo-resolver` under `.claude/agents/ammo-resolver.md`, then obtain the
   required independent merge review. Validate the resulting exact subset as a
   composition. If the defensible result is one unchanged track, fall back to
   the single-pass short-circuit; if none remains, exhaust the round.

Do not select by a copied PASS/GATED/dilution matrix, and do not select by a
single best number. Judge each candidate subset on correctness, production
activation, gating behavior, interaction risk, regressions across the frozen
workload, and clean E2E evidence. One subset dominates another only when its
evidence supports the same required behavior, shows no worse accepted-workload
result, and shows a better campaign objective. When the evidence does not
establish dominance, validate the plausible composition rather than guessing.

## Single-Track Short-Circuit

When exactly one accepted track remains, and its patch and environment have not
changed since Stage 5, it ships on its Stage 5 evidence — integration would add
no new technical information. Preserve its PASS versus GATED_PASS semantics and promote
the result it already validated. Materialize that evidence in the canonical
integration slot when the mechanical pre-ship checks require it there; do not
rerun merely to rename it. Any source, build, dispatch, or environment change
disables the short-circuit and requires the combined workflow.

## Combined Validation Workflow

The integration branch must contain exactly the selected, previously validated
track changes plus any independently reviewed conflict resolution. Record the
selected track identities and immutable commit/patch provenance before running
validation. Any later source, build, dispatch, or environment change invalidates
the result and requires validation again.

For every multi-track composition:

1. Resolve merge and semantic interactions without weakening any selected
   track's accepted boundary, fallback, or rollback control. Ambiguous or
   conflicting resolutions use `ammo-resolver` and a fresh independent reviewer;
   after the bounded review loop, return unresolved evidence to the lead.
2. Run full-model correctness against the current round's frozen golden
   comparator, not only component tests.
3. Run a clean, profiler-free production E2E measurement across the frozen
   batch-size/workload matrix, compared against the Stage 1 baseline
   (`--labels opt --baseline-from`), which imports by literal filename
   `{baseline_label}_{tag}.json`. A round >= 2 integration sweep MUST import the
   promoted arm (`--baseline-from-arm opt`), or assert the comparator's `avg_s`
   and env against the promoted mainline before any timing leg runs; an
   unasserted pre-SHIP comparator makes the delta cumulative, not incremental.
   Apply the thresholds and per-bucket rules in
   `references/validation-defaults.md`.
4. Capture separate claim-appropriate profiler evidence in the
   `integration_profiling` slot. It must prove that every selected optimized
   path and gated boundary actually activates after composition. Profiled
   latency is not timing authority.
5. Write the coherent result to the current round's `integration` state and
   artifact slots. Never combine metrics from different source trees or
   environments into one verdict.

If a composition fails, use its evidence to remove or repair the implicated
track or interaction, then validate the new exact subset. A track that passed on
its own may still serve as the unchanged single-track fallback. Never ship an
extra track because its individual result looked harmless, except for the narrow
diluted-track carve-out below. Otherwise every shipped multi-track set must be
the set measured in the passing integration run.

### Diluted-track carve-out when composition loses

This is an explicit exception to the exact-set rule above. When a measured
multi-track composition loses to the best individual track, a track that would
otherwise be dropped may still ship alongside that best track only when all of
the following hold:

- the dropped track is a distinct mechanism/component and is recorded
  `diluted:true`;
- its current Stage-5 per-batch-size evidence contains no regression, re-read
  from that track's `validation_summary.json` rather than a stale state marker;
  and
- it lands as a second, independently cherry-picked commit alongside, not
  instead of or folded invisibly into, the best individual track.

Record the diluted track as shipped with `diluted:true` and record why the
combined objective selected the best individual result. This exception keeps a
non-regressing diluted mechanism without pretending that its effect was
additive. For this carve-out, cumulative speedup remains derived from the
combined artifact's measured production E2E for the shipped pair. Never replace
that authority with the best individual's result plus an assumed diluted-track
contribution, and never add, multiply, or otherwise credit unmeasured
additivity.

## Pre-SHIP Mechanical Checks

Before merging to the session mainline, run the existing fail-closed mechanical
checks for conflict residue, successful sweep legs, schema-valid state, and
per-workload regressions. These checks enforce recorded invariants; the lead's
technical review still owns compatibility and the final evidence judgment.

## Promotion and Finalization

After a passing integration decision:

1. Merge or cherry-pick the exact validated patch set onto the session
   mainline and record its final commit identity.
2. Promote the environment contract. The new baseline environment is **the
   prior baseline environment plus only the selected accepted flags and values
   used by the passing validation**. Clear the experimental environment. Do not
   promote flags from rejected, unselected, superseded, or exploratory tracks,
   and do not retain unrelated shell-inherited experimental keys. Environment
   naming and isolation are owned by `references/impl-track-rules.md` and the
   sweep runner.
3. Capture fresh golden references from the promoted mainline in the current
   round's `golden_capture` slot. These become the next round's correctness
   comparator; they do not replace the current round's original evidence.
4. Record the accepted track identities, integration outcome, metrics, commit,
   promoted environment, dilution annotations where applicable, and current
   round finalization fields in schema-valid state. Finalize the current round
   in place.
5. Run `T_AUDIT_S67` according to `orchestration/audit-protocol.md`. It covers
   the exact integration, environment promotion, golden capture, shipped
   bookkeeping, and campaign-evaluation inputs. A rejected change must be
   removed and the promoted scope rebuilt and remeasured before re-audit;
   resolve consequential findings before advancing.
6. Enter Stage 7. **Only Stage 7** may evaluate the stop condition, increment
   `campaign.current_round`, or create the next-round state and artifacts.

For EXHAUSTED, record the terminal integration/round outcome in place and run
`T_AUDIT_S67` before Stage 7; there is no environment promotion or golden
capture.

Compute cumulative improvement directly from the original Round-1 production
baseline and the latest accepted integrated production latency. Never multiply
round-local speedups. Stage 7 continuation, terminal-token choice, all-diluted
round handling, and mining invalidation are owned by `SKILL.md` under
`Stage 7: Campaign Evaluation`.

Until `T_AUDIT_S67` passes, current-round `shipped` identities are a provisional
promotion record, not a reportable final ship. A Stage-67 rejection immediately
removes the implicated code, environment keys, current integration/shipped and
global shipped identities before remeasurement; record the rejection in
`dropped`, `integration.final_decision`, and the round summary. Preserve the
isolated Stage-5 verdict as history—Stage 67 rejects promotion/composition, not
the fact that the earlier isolated gate passed.

## Authority Pointers

- State shapes and enums: `.claude/schemas/state.schema.json`
- Legal transitions and terminal sets: `scripts/transitions.json`
- Track verdicts, correctness, E2E, activation, and gating:
  `references/validation-defaults.md`
- Environment flags and fallback contracts: `references/impl-track-rules.md`
- Canonical integration, profiling, and golden paths:
  `references/artifact-layout.md`
- Conflict-resolution role: `.claude/agents/ammo-resolver.md`
- Stage 6-7 audit: `orchestration/audit-protocol.md` and
  `references/audit-invariants.md`
- Stage 7 continuation: `SKILL.md`
