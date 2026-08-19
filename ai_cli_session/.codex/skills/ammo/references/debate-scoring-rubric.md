# Debate Scoring Rubric

The lead applies this rubric at Winner Selection, after the Stage 3 adversarial debate. Scoring has two layers: an **eligibility layer** (structural gates + probability ceilings) decides who **can** win, and an **expected-value ranking** (`EV = P(success) × projected E2E delta`) decides who **does** win. Provability and impact multiply, never add, and impact enters linearly and uncapped — no score saturation for large deltas.

## What you read for each candidate

Read every debate artifact across all sub-rounds. Each candidate has one **argument-of-record** per sub-round:

- **Sub-round 1** — the candidate's section in `rounds/{CR}/debate/proposals/{champion_id}_proposal.md`, resolved through the `op_id → proposal section` mapping in `rounds/{CR}/debate/eligibility_verdicts.md`. There is no sub-round-1 `{op_id}_argument.md`, because Phase A is skipped in sub-round 1 (`orchestration/debate-protocol.md` § Debate Sub-Rounds).
- **Sub-rounds 2+** — `rounds/{CR}/debate/round_{N}/{op_id}_argument.md`.

Also read every sub-round's critiques (`{op_id}_critique_*.md`) and rebuttals (`{op_id}_rebuttal.md`), including sub-round 1's.

## Lossy Classification Rule

Classify each proposal before you score it, by the dtype boundary rule: if the optimization introduces a precision reduction at ANY point in the dataflow — an output dtype with fewer mantissa/exponent bits than its input — the proposal is **lossy**; otherwise it is **lossless**.

| Scenario | Classification | Rationale |
|----------|---------------|-----------|
| BF16 activations quantized to FP8 before fused GEMM | **Lossy** | New precision reduction introduced |
| Fusing two GEMMs on an already-FP8 model | **Lossless** | No new precision reduction — model was already quantized |
| FP32 accumulator → BF16 output (same as baseline) | **Lossless** | Accumulator precision matches baseline — no new truncation |
| Switching from FP32 to BF16 accumulator for larger tiles | **Lossy** | Accumulator precision reduced vs. baseline for performance |
| BF16 weights cast to FP8 for faster tensor core MMA | **Lossy** | Weight precision reduced for performance |
| INT4 dequant fused with GEMM (model already INT4) | **Lossless** | No new quantization — just fusing the existing dequant step |

The champion self-declares `lossless` or `lossy` in the Phase 0 proposal, citing this rule. The field is required (`debate-protocol.md` § Eligibility, revision, and elimination).

## Scoring Is On Content, Not Volume

Score evidence and reasoning, never prose length, section count, or visible diligence. A compact proposal that clears every criterion scores exactly as well as a verbose one, and ties prefer the compact one. Critique deductions count whether each material critique was *addressed*, not how many words addressed it: a one-row concession or one-sentence counter fully addresses a critique. The probability ceilings are cleared by the *presence of the required evidence* — a methodology checklist line, both warm/cold numbers, a baseline-provenance pointer — not by narrative around it. Never penalize a proposal for being short when the required evidence is present. Champions write per `references/writing-style.md`; hold them to evidence, not word count.

## Layer 1: Eligibility (structural gates)

The gates check what the proposal *is*, not how good it is.

**Evaluated once, trusted at scoring (CUT-4).** The lead runs the four Phase-0 gates (Gate 1 Authored-Mechanism, Gate 2 Technology-Selection block, Gate 3 Precision, Gate 4 Category) under `orchestration/debate-protocol.md` § Eligibility, revision, and elimination, and records the per-`op_id` verdict in `rounds/{CR}/debate/eligibility_verdicts.md`. Debate rounds write separate argument/critique/rebuttal files and never amend the proposal, so the text scoring reads is byte-identical to what the gate passed. Scoring trusts a recorded all-PASS verdict and does not re-derive the gates.

**Exception — critique-surfaced Gate-1/Gate-2 evidence.** When a Phase B critique, a Phase C rebuttal/concession, or (sub-rounds 2+) a Phase A argument surfaces evidence about mechanism authorship (Gate 1) or Technology-Selection truthfulness (Gate 2) that the Phase-0 gate never saw, scoring MUST re-derive that gate against the full record (proposal + arguments + critiques + rebuttals). The re-derivation is elimination-capable, not a P-score deduction — this closes the path where a flag-flip that slipped a thin Phase-0 self-report rides an uncapped measured delta to a win:

- **Gate 1**: the "authored mechanism" turns out to be a retuned constant or config flip (byte-identical kernel/cubin body — e.g. a hand-described loop that is functionally `num_stages=2→3`) → eliminated regardless of EV, before EV is computed.
- **Gate 2**: a Technology Selection field was false (e.g. `Baseline technology: unknown` for a known library kernel, or an anti-regression claim contradicted by evidence) → if the gate now fails (absent/empty/internally-contradictory fill), eliminated.

Append the outcome — op_id, triggering artifact, revised verdict — to `eligibility_verdicts.md` (lead-owned). A candidate with zero Gate-1/Gate-2-relevant content across all sub-rounds keeps its recorded verdict.

Gate 3 (Precision) and Gate 4 (Category) are never re-run as elimination gates. Gate 3's scoring-time signal is the −2 deduction for undisclosed precision reduction (§ deductions) plus the lossless/lossy declaration feeding § Lossy E2E Impact Scoring. Gate 4's is proxy-scope EV-ineligibility and the scope-objection EV=0 (§ Projection-integrity corrections; `references/debate-rules.md` § Evidence-Scope Ladder).

## Layer 2: Expected-Value Ranking

Every ordinary eligible candidate gets one score:

```
EV_pct = P(success) × projected_e2e_delta_pct
```

A `contingent_host_spike` has no admissible host-saving magnitude yet, so it receives no fabricated EV; its selection basis and first stop gate are defined under Winner Selection Rules. DILUTED_PASS is a validation-stage OUTCOME, never a proposal target: EV ranking uses `projected_e2e_improvement_pct` only, and arguing that "decode dilution will still ship it" does not raise a projection.

### The probability term: P(success)

Score evidence quality 0-10 (§ Scoring Scale); `P(success) = score / 10`. Build the score in order: base assessment, then ceilings, then deductions.

**Base assessment** — how well-evidenced is the claim that this optimization will ship? Weigh admissible measured boundaries, conservative bounds, source/ISA analysis, grounded profiling, correct math, and hardware-specific effects. Unsupported attainable-speedup claims score low; a disclosed evidence gap keeps its ceiling. **Per-BS f-values required**: champions report `f_e2e` for each target batch size, with `f_decode` diagnostic only.

**Probability ceilings** — a ceiling caps the score, it does not eliminate; multiple ceilings combine as `min(...)`, not additive:

| Ceiling | Trigger |
|---------|---------|
| **3/10** (P ≤ 0.3) | **Theoretical-only**: NO empirical kernel benchmark (only roofline, ISA inspection, or `ncu --query-metrics`), regardless of theoretical quality |
| **3/10** (P ≤ 0.3) | **Baseline provenance**: a measured baseline materially diverges from Stage 2 for the same shape and the production dispatch/identity discrepancy remains unresolved. |
| **3/10** (P ≤ 0.3) | **High-risk replacement**: the proposal gives up capabilities present in a mature production kernel/library/runtime path without production-boundary beats-baseline evidence. Apply the capability-based rule in `technology-selection.md`; do not infer risk from a static language/DSL ranking. |
| **5/10** (P ≤ 0.5) | **Methodology**: micro-experiment methodology would be INVALID under validation-defaults.md (no CUDA graph capture, eager mode) |
| **5/10** (P ≤ 0.5) | **Cache audit**: a load-bearing cache-residency claim lacks the warm/cold evidence needed to support it |
| **5/10** (P ≤ 0.5) | **Unresolved measurement dispute**: see § Handling Conflicting Experimental Data |
| **6/10** (P ≤ 0.6) | **Pipeline accounting**: a repeated optimization reports isolated speedup without the occurrence-weighted production boundary required by `references/e2e-delta-math.md` |

An internally contradictory anti-regression fill (e.g. "not applicable" when the ranking comparison says the rule applies) fails Gate 2 at Layer 1 and eliminates; the ceiling is only for candidates that filled the block honestly but lack the evidence. Honest disclosure competes at low P; fabricated compliance is eliminated.

**Deductions** (applied after ceilings; each lowers the P-score):

| Deduction | Trigger |
|-----------|---------|
| −2 | Each unaddressed material critique from another champion (conceded + mitigated critiques are neutral) |
| −2 | Undisclosed precision reduction revealed by a critique on a "lossless" proposal |
| −2 | Fusion proposal where test data < 25% of production pipeline working set AND warm/cold > 1.5x |
| −2 | Ranking with a phase-only share or omitting measured `f_e2e` in the per-BS table |
| −1 to −2 | Implementation complexity risk: large CUDA/Triton diff, many files modified, CUDA graph safety risk, regression likelihood. High complexity is a lower probability of shipping, not a separate score |

### Scoring Scale (the P-score)

Evidence quality scores 0-10; `P(success) = score / 10`.

| Score | Meaning |
|-------|---------|
| 9-10 | Strong evidence, no material gaps — near-certain to survive validation |
| 7-8 | Solid evidence with minor gaps |
| 5-6 | Adequate evidence but notable uncertainties |
| 3-4 | Weak evidence, major gaps or unaddressed critiques — a real but long-shot bet |
| 0-2 | Insufficient evidence or fatally flawed |

A low P-score is not elimination: a 3/10 theoretical-only proposal projecting a large delta is a legitimate long-shot entry — that is why the terms multiply instead of add.

### The magnitude term: projected E2E delta

Project with **`f_e2e`, NOT `f_decode`**, using the one boundary equation (`references/e2e-delta-math.md` § Projection Authority). Per-BS projections REQUIRED; the EV uses the projected delta on the best gatable BS range — BS-dependent regressions are acceptable if gatable. The delta enters the EV linearly and uncapped.

Only `bound`-, `production_boundary`-, and `clean_e2e`-scope magnitude enters EV; `proxy`-scope magnitude is EV-ineligible (feasibility / upper-bound only). An unresolved scope objection is a hard EV=0 (`references/debate-rules.md` § Evidence-Scope Ladder). A newly selected candidate whose `score_breakdown.evidence_scope == "proxy"` is a state-validator gate violation (`scripts/ammo_state.py`).

### Projection-integrity corrections

The lead corrects the delta, not the points:

- **`f_decode` without conversion**: when a workload-dilution red flag fires, recompute with `f_e2e = f_decode × decode_busy × decode_share_of_e2e` and apply −1 (see `references/e2e-delta-math.md`).
- **Unaccounted integration overhead**: correct partition/structural costs downward before EV.
- **Cache-dependent speedups**: warm/cold > 1.5x → project from the cold-cache number.

### Lossy E2E Impact Scoring

A lossy proposal's E2E delta uses the standard projection — `effective_E2E = 1 + f_e2e × (1 - 1/s_T)`, where `s_T` is the total kernel speedup (see `references/e2e-delta-math.md`). Score the accuracy and numerics risk where it can be measured: the precision-reduction deduction above and Gate 5.1a correctness at validation.

The champion must decompose `s_T = s_L × s_Q` — the **lossless component** (`s_L`: fusion, tiling, memory layout, scheduling) times the **quantization component** (`s_Q`: dtype reduction enabling faster MMA, reduced BW) — and back the split with micro-experiment evidence (e.g. running the fused kernel at original vs. reduced precision). The split makes the quant claim auditable: a champion who cannot show the quant component actually delivers `s_Q` has an unsupported claim, scored down in the base assessment. The E2E delta still projects from `s_T` directly.

## Winner Selection Rules

0. **Gate precedence**: Layer 1 resolves and eliminates BEFORE any EV is computed — by the trusted recorded verdict or the elimination-capable re-derivation, per § Layer 1 — so a large uncapped delta can never buy a structurally ineligible candidate a win. The probability ceilings are P-score rules that apply only to candidates that cleared both gates.
1. **Minimum threshold**: eliminate candidates whose admissible `projected_e2e_improvement_pct` is below `campaign.config.min_e2e_improvement_pct`. The sole exception is at most one `contingent_host_spike` under `e2e-delta-math.md`: a measured production host-slice ceiling meets or exceeds the floor but no admissible host-saving magnitude exists yet. Record the ceiling and feasibility basis, never count the ceiling as projection or EV, and make the actual-runner boundary A/B its first Gate 5.2 stop gate.
2. **Rank by EV**: order ordinary survivors by `EV_pct` descending and select **2-3 winners** top-down, subject to the concentration rule; the optional contingent host spike may occupy one slot on its measured ceiling and feasibility, not fabricated EV. Prefer 3 if ≥3 GPUs are available for parallel tracks.
3. **Portfolio concentration rule**: never select two candidates with the same `(component, mechanism-category, technology-class)` tuple — that is the same bet twice, and the second slot is wasted regardless of its EV. Two candidates on the **same component** via **different mechanisms or different authoring classes** are allowed (and often desirable when that component's EV dominates) provided they are **algorithmically independent** — different scheduling, tiling, or fusion strategy, not the same algorithm wrapped in different tooling. Reject the pairing when the two Kernel Code Scope descriptions are identical except for the technology field. When a duplicate-tuple candidate is skipped, the next-ranked non-duplicate takes the slot.

## Handling Conflicting Experimental Data

When two champions present contradictory micro-experiment results for the same kernel/shape with a materially large discrepancy, scoring of the disputed claim is blocked until resolved (advisory — scoring-only, NOT a ship/retract gate):

1. **Standardized tiebreaker**: run a CUDA-graphed benchmark with agreed methodology.
2. **Methodology disclosure**: both champions disclose exact measurement code; the one using production-parity methodology (CUDA graphs + torch.compile) takes precedence.
3. **Unresolved**: cap the disputed claim's P-score at 5/10 (scoring-only advisory ceiling).

The lead MUST NOT advance a candidate to Stage 4 with unresolved material-magnitude measurement discrepancies.

## Output

**Authoritative cross-agent contract**: write structured winner data to `state.json.campaign.rounds[N-1].debate.selected_candidates` as an **array** (one entry per winner — debate selects 2-3 per round). Each entry has:

- `op_id`
- `name` — the short display name minted with the `op_id` (2-4 plain words that describe the mechanism)
- `selection_mode` (`ordinary` or `contingent_host_spike`); contingent mode also records `host_slice_ceiling_pct`, while `score_breakdown.expected_e2e_pct`, `weighted_total`, and `projected_e2e_improvement_pct` remain zero
- `track_assignment` (enum: `lossless`, `quant`, `structural`)
- `score_breakdown` (feasibility, evidence_tier, expected_e2e_pct, weighted_total) — under EV scoring: `feasibility` = the P-score (0-10), `expected_e2e_pct` = the projected E2E delta (after any lead corrections), `weighted_total` = `EV_pct` (`feasibility/10 × expected_e2e_pct`; the field name is retained for schema compatibility); and REQUIRED `evidence_scope` (`bound`|`proxy`|`production_boundary`|`clean_e2e`) — populate it on every selected candidate; a newly selected `proxy` is a validator gate violation.
- `stage_4_validation_obligations` (schema-owned enum); contingent mode includes `production_boundary_spike` as its first implementation gate
- `cited_evidence` (array of `file:line` or artifact paths)

`op_id` values here must match the entries in `selected_winners` (the list of chosen op_id strings). Each Stage 4 impl-champion reads the entry matching its assigned `op_id`. Never hand-author prose that substitutes for these typed fields.

The canonical gates plus `stage_4_validation_obligations` are the complete blocking validation contract. Proposal prose can motivate an experiment, but naming NCU, Nsys, or another tool there does not create a new terminal gate; the required evidence follows the claim and the canonical gate authority.

**Summary (humans only)**: after the typed state is written, the lead authors `{artifact_dir}/rounds/{CR}/debate/summary.md` from `state.json` and the debate record. Champions never write it. If `summary.md` and `state.json` disagree, `state.json` wins; the lead corrects the summary.
