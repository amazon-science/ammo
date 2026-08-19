# Stage 3: Adversarial Debate Protocol

Read this file when you run Stage 3 as the lead. Stage 3 takes the grounded evidence from Stage 2, has champions propose candidates independently, cross-examines those candidates, and ends in a typed implementation handoff. Debate is mandatory; there is no fast track. This file owns choreography only — who acts, in what order, and which file each act writes.

The policy lives elsewhere. Apply each authority below where it applies, and never copy its policy into a debate artifact:

- evidence scope and experiments: `references/debate-rules.md`;
- projection math: `references/e2e-delta-math.md`;
- eligibility and ranking: `references/debate-scoring-rubric.md`;
- technology and category declarations: `references/technology-selection.md` and `references/optimization-categories.md`;
- paths and typed state: `references/artifact-layout.md` and `.claude/schemas/state.schema.json`;
- agent messaging and lifecycle: `references/champion-common-patterns.md` and `references/shutdown-protocol.md`.

Two shorthands run through the paths below. `{CR}` means `campaign.current_round`; `{N}` means the debate sub-round.

## Cohort and Independence

Spawn **2-4 `ammo-champion`** agents. Give each one a stable, round-qualified task name. Hand every champion the same full Stage 2 profile and the same workload-dilution summary. You may add a distinct analysis lens to steer where a champion looks first, but never assign or reserve a component to anyone.

Each champion analyzes the full profile on its own and submits 2-3 ranked candidates. Champions do not share drafts and do not begin critique before you complete the Phase 0 barrier. Debate champions have no transcript monitors.

## Phase 0: Independent Proposals

Champions run in parallel. Each one atomically publishes exactly one file:

```text
rounds/{CR}/debate/proposals/{champion_id}_proposal.md
```

That file holds 2-3 independently eligible candidates, plus the declarations and evidence the authorities above require. Wait for every proposal you expect before you start the eligibility and diversity barrier.

### Eligibility, revision, and elimination

Gate each candidate against the four structural gates owned by `references/debate-scoring-rubric.md`: authored mechanism, Technology Selection, Precision Classification, and Category block. Read those gates as written; do not restate or reinterpret them.

- Reject a candidate with an explicit revise request. The champion then increments the proposal's `revision`, atomically republishes the same path, and sends `{type: "proposal_revised", champion_id, revision}`.
- A later revision invalidates your earlier verdict until you re-gate it.
- A candidate that does not produce a compliant revision is eliminated and receives no `op_id`.
- Only the gate-final revision may enter critique.

Then run the diversity and exhaustion checks owned by the scoring and technology-selection authorities. Overlapping ideas may stay in the debate; portfolio concentration waits for winner selection, because the evidence is richest there.

### Stable identities and eligibility record

Give every survivor a stable `OP-NNN` identity plus a short display name: 2-4 plain words that describe the mechanism (for example, "fused rmsnorm quant"). The `op_id` is the key in state, paths, and prompts; the name is how prose and summaries refer to the work. Record both, with the proposal path and candidate section, the accepted revision, the four gate outcomes, and the evidence pointer, in the lead-owned artifact:

```text
rounds/{CR}/debate/eligibility_verdicts.md
```

That mapping stays fixed through every sub-round and through winner selection. Scoring trusts the recorded verdict, with one exception: the critique-surfaced re-derivation defined by `references/debate-scoring-rubric.md`. Append every such re-derivation to this same artifact.

## Critique Assignment

Fix the critique assignment once eligibility and diversity settle the survivor set. It covers all minted `op_id`s and it is cross-owner: a candidate is never critiqued by the champion who proposed it. A simple ring works when adjacent owners differ:

```text
OP-001 -> OP-002 -> ... -> OP-N -> OP-001
```

When candidate counts make a cross-owner ring impossible, write an explicit cross-owner map instead: every candidate still receives one independent critique, and one owner may author several critiques. Reuse the assignment while membership stays stable.

If fewer than two distinct champion owners survive eligibility, ask the remaining Phase 0 cohort for bounded replacement proposals and re-gate them. If you cannot produce two independent owners without inventing evidence, block selection rather than allow self-critique.

Send one authoritative `critique_assignment` per directed edge. Each one carries `critic_op_id`, the target `op_id`, the target path/section, and the output path. The same `critic_op_id` may appear on several edges when owner counts are unequal. This broadcast is the only start signal for Phase B; champions do not infer assignments from directory contents.

## Debate Sub-Rounds

One sub-round is the minimum and five is the cap.

- **Sub-round 1:** the Phase 0 proposal section is the argument of record, so run Phase B critique and then Phase C rebuttal. There is no duplicate Phase A.
- **Sub-rounds 2-5:** Phase A argument, then Phase B critique, then Phase C rebuttal.

All sub-round artifacts live under `rounds/{CR}/debate/round_{N}/`. Publish each complete artifact through a temporary sibling and an atomic rename. Appearance of the final path means the artifact is complete.

### Phase A: Argument, sub-rounds 2+

Each surviving candidate addresses its live objections in:

```text
rounds/{CR}/debate/round_{N}/{op_id}_argument.md
```

Run corrected or additional evidence when a live measurement dispute requires it, under `references/debate-rules.md`. An argument must respond to prior criticism rather than repeat the proposal.

### Phase B: Critique

After each `critique_assignment`, its owner waits for one thing only — that target's argument of record:

- sub-round 1: the assigned gate-final proposal path and candidate section;
- sub-rounds 2+: the assigned `{target_op_id}_argument.md`.

It then writes:

```text
rounds/{CR}/debate/round_{N}/{critic_op_id}_critique_{target_op_id}.md
```

Critique the target's feasibility, evidence scope and provenance, projection, hardware accounting, production/compile safety, precision, and risks. Do not wait for unrelated candidates.

### Phase C: Rebuttal

Each candidate waits for the exact incoming critique path you supplied, and for nothing else. It then writes:

```text
rounds/{CR}/debate/round_{N}/{op_id}_rebuttal.md
```

The rebuttal must answer with evidence, explicitly concede valid points, or give a concrete mitigation. It ends with one op-scoped declaration:

```markdown
## Open Items Declaration
- [UNADDRESSED_CRITIQUE] <still-unanswered criticism>
- [NEW_EVIDENCE] <new claim not yet cross-examined>
```

Omit a line that does not apply. Use `- [NONE]` only when neither category remains. `[NONE]` never erases an independently recorded boundary-scope objection.

## Convergence Criteria

Wait for one rebuttal and one declaration per surviving `op_id`, then update a live open-item ledger from the full record.

- A declared item stays live until a later argument, critique, or rebuttal addresses it and the record shows closure.
- If no live items remain, go to selection.
- After sub-round 1, one live item is enough to require sub-round 2; this keeps the first cross-examination of open claims mandatory.
- After each complete sub-round 2+, stop early on either convergence condition. Clear winner: the top 2-3 candidates have no unaddressed critiques and every other candidate has conceded material weaknesses. Stagnation: the new arguments substantially repeat the prior round with no new evidence and no new counter-arguments.
- If neither round-2+ convergence condition holds, live items remain, and `{N} < 5`, run another sub-round.
- At early convergence or at the five-round cap, stop producing rounds and carry every unresolved item into scoring. Stopping never records an item as closed.

A champion may rebut a boundary-scope objection, but **only the lead or the critic who raised it may close it**, and only with a cited artifact showing that the declared production boundary was actually measured. Any such objection still live at scoring makes that candidate `EV=0` under `references/debate-rules.md`. Every other unresolved critique receives the scoring treatment defined by `references/debate-scoring-rubric.md`.

## Barriers and Liveness

Only two global barriers exist:

1. all Phase 0 proposals are present before eligibility, diversity, minting, and the ring broadcast;
2. all surviving rebuttals are present before the end-of-round open-item tally.

Argument-to-critique and critique-to-rebuttal dependencies are pairwise, so they may run concurrently. Track the exact set of artifacts you expect. If an artifact misses its phase deadline, ping its owner once with the expected path; after a short grace period, eliminate the stalled candidate and repair every affected cross-owner assignment before you continue. Never leave a survivor without an incoming critique, and never weaken a barrier to avoid the repair.

## Winner Selection

After convergence or the round cap, read the complete debate record and apply `references/debate-scoring-rubric.md` in this order:

1. resolve structural eligibility, including any critique-triggered re-derivation;
2. enforce evidence-scope blocks and score only admissible magnitude;
3. rank eligible candidates by expected value;
4. apply the portfolio concentration rule and select **2-3 winners**.

Write both `selected_winners` and the complete typed `selected_candidates` entries to the current round in `state.json`. Each winner's entry must include its `proposal_file`, track assignment, score breakdown, evidence scope, Stage 4 validation obligations, and cited evidence, as the schema and the scoring authority require. The `op_id`s must agree across both fields. This typed state is the authoritative Stage 4 handoff. The canonical gates plus the typed Stage 4 validation obligations are the complete blocking validation contract; proposal prose cannot add another gate.

Then write `rounds/{CR}/debate/summary.md` yourself. You are its only author; champions have no write path to it. Cover each winner (name and `op_id`, mechanism in one sentence, score and EV, obligations, cited evidence) and one short paragraph on why it beat the field. Take every number from `state.json` — if the summary and `state.json` disagree, `state.json` wins and you correct the summary.

## Handoff

Before Stage 4, confirm two things: the selected candidate records resolve to their gate-final proposals, and every live objection is represented in scoring or in a validation obligation. Once the winner is selected, enumerate the CURRENT ROSTER and send `SendMessage(to=<name>, message={"type": "shutdown_request"})` to every debate-round teammate on it — champions and any rostered delegates or monitors, completed and idle members included, so their roster entries clear. The bounded-wait rule of `references/shutdown-protocol.md` applies. Then create one isolated implementation track per winner under the Stage 4-5 protocol.
