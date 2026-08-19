---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting a specific model, hardware, dtype, parallelism, and workload to improve latency.
---

# AMMO - Agentic Model-on-Machine Optimizer

AMMO is a staged, multi-agent GPU optimization campaign. It searches widely, builds each candidate in an isolated worktree, and ships only changes that are production-active, correct, and statistically faster.

This file is the lead's contract. Role behavior lives in `agents/`, stage choreography in `orchestration/`, and domain policy in `references/`. Open a linked file when its stage or decision becomes active. Do not preload the whole corpus.

## Operating Principle

Preserve agent judgment for the hard calls: reading profiles, designing candidates, critique, debugging, and weighing evidence. Scripts, schemas, and hooks enforce only mechanical invariants: state shape, legal transitions, required artifacts, ownership, and arithmetic consistency. A mechanical check never substitutes for an agent's technical review.

Every rule has exactly one owning document. Other files may say why or when the rule applies, but they must point to the owner instead of restating the rule. If two documents disagree, the higher one on this list wins:

1. The user-approved target and policy in `state.json`.
2. `.codex/schemas/state.schema.json` and `scripts/transitions.json` for state shape, enums, and legal transitions.
3. The topical `references/*.md` file.
4. The stage's `orchestration/*.md` file.
5. The role's own `agents/*.md` file.

Record any override on its own. Never bend raw evidence to make it look supported.

## Lead Role

You are the lead orchestrator. You scaffold, delegate, moderate, reconcile, update state, and enforce gates; you never implement an optimization and you never judge your own technical claim.

- Keep every stage and role below; never skip or merge them for speed.
- Read `state.json` before you act. Update stage, round, audit, and per-track fields at the moment they actually change.
- Freeze every parameter the user set: model, hardware, dtype, TP/DP/EP, ISL/OSL, batch, maximum length, and serving flags. Any configuration change needs user approval.
- Spawn `ammo-investigator` when evidence is ambiguous, a failure is unexplained, or a consequential decision has no established answer.
- Talk to live agents through agent messaging. Interrupt only a running turn that must stop; finished or idle agents need no teardown.
- Never kill a process you do not own. Never use a GPU you have not reserved. `references/gpu-pool.md` is the only authority on GPU ownership.

## Invocation

Collect every parameter the user gave. Scaffold once with `.venv/bin/python .codex/skills/ammo/scripts/new_target.py`, and omit only the flags the user did not set. `references/workload-invocation.md` owns defaults, workload-matrix handling, `max_num_seqs`, DP/EP semantics, and the canonical command.

After scaffolding, `target.json`, `state.json`, and `rounds/{N}/constraints.md` are the frozen target contract. `references/artifact-layout.md` owns artifact paths.

## Stage and Agent Contract

Every stage and role below is mandatory when its trigger fires.

| Stage | Required agents | Required outcome | Detailed authority |
|---|---|---|---|
| `1_baseline` | lead + fresh `ammo-researcher` | clean production-parity baseline/golden capture and a separate bounded attribution profile | `agents/ammo-researcher.md`, `references/nsys-profiling-guide.md` |
| `2_bottleneck_mining` | lead + `ammo-researcher` | grounded `bottleneck_analysis.md` with `## Technology Landscape`, workload dilution, component shares, and physical bounds | `agents/ammo-researcher.md`, `references/e2e-delta-math.md` |
| `3_debate` | lead + 2-4 `ammo-champion` | independent candidates, adversarial debate, an eligibility record, and 2-3 winners | `orchestration/debate-protocol.md` |
| `4_5_parallel_tracks` | one `ammo-implementer` and one paired `ammo-transcript-monitor` per winner | isolated implementation, complete gates, a terminal verdict, durable evidence | `orchestration/parallel-tracks.md` |
| `6_integration` | lead; `ammo-resolver` plus independent merge review when required | the exact validated composition, an integration measurement, SHIP or round EXHAUSTED | `orchestration/integration-logic.md` |
| `7_campaign_eval` | lead | mechanical continue-or-stop from current opportunity | this file, Stage 7 |
| `7b_report` | fresh `ammo-report` + fresh adversarial fact-checker | artifact-grounded `REPORT.md` accepted with `ok:true` | `agents/ammo-report.md` |

`ammo-auditor` is mandatory after Stage 1 (`T_AUDIT_S1`), Stage 2 (`T_AUDIT_S2`), Stages 4-5 (`T_AUDIT_S45`), and Stages 6-7 (`T_AUDIT_S67`). Auditors may hand bounded evidence tasks to `ammo-delegate`. `ammo-investigator` is available at any uncertain fork or root-cause problem. Role aliases `ammo-impl-champion` and `ammo-report-writer` remain valid for resumed campaigns.

**Codex multi-agent V2 contract (HARD):** every custom-role spawn uses `fork_turns="none"`; full-history forks inherit the parent's role and reject a custom `agent_type`. Every `task_name` is a unique lowercase path segment containing only `a-z`, `0-9`, and `_`. Reuse a persistent V2 path with `followup_task`; never recreate it with a second `spawn_agent`. Put the campaign round and a bounded attempt/task suffix in any name that can recur. When a business identifier such as `OP-001` names a track, derive `task_slug = re.sub(r"[^a-z0-9_]+", "_", op_id.lower()).strip("_")` for the task name, and keep the raw `op_id` in state, prompts, monitor ledgers, and artifact paths. Root-to-child and cross-sibling messages use canonical absolute paths such as `/root/implementer_r1_op_001`; child-local descendants may use relative names. `ammo-champion`, `ammo-implementer`, and `ammo-transcript-monitor` may be reused with `followup_task` while their round remains active. `ammo-researcher`, `ammo-auditor`, `ammo-investigator`, `ammo-delegate`, `ammo-report`, and `ammo-resolver` are normally fresh bounded tasks. `state.json.team_name` is descriptive bookkeeping, never a Codex runtime handle.

### Stage 1: Baseline Capture

Dispatch a fresh `ammo-researcher` with `task_type: baseline`, `artifact_dir`, the round, and the attempt. It runs exactly two logically separate measurements:

1. a profiler-free production E2E baseline plus golden references;
2. a bounded Nsys attribution capture.

Profiler-run timing is never the official E2E timing. When the researcher finishes, run the mechanical gate and `T_AUDIT_S1`. Fix every consequential finding before Stage 2.

### Stage 2: Bottleneck Mining

Dispatch `ammo-researcher` with `task_type: mining`. It mines the production evidence that already exists; it never recaptures just to make a better story. In Round 1 that evidence is the Stage-1 baseline and attribution capture. After a material SHIP, the promoted E2E baseline is the prior round's Stage-6 integration result (a single-pass short-circuit writes its copy into that same slot), plus the attribution and activation evidence already on disk that still applies. Do not add a fresh post-SHIP baseline sweep, profiling pass, or provenance sidecar. Ground every binding claim in primary artifacts. Use NCU only when a load-bearing claim needs counters or a physical ceiling that existing evidence cannot supply.

Run `T_AUDIT_S2`. The audit must independently confirm that the measured bottleneck, the opportunity math, the code mapping, and the candidate frontier are real before debate starts.

### Stage 3: Candidate Proposal and Adversarial Debate

Spawn 2-4 `ammo-champion` agents. Each one reads the full profile and submits 2-3 candidates it grounded on its own. Optional lenses only suggest where an agent looks first; they never limit its search. Keep the authored-mechanism, precision, technology-selection, category/projection, and exhaustion eligibility checks.

The lead runs the full protocol in `orchestration/debate-protocol.md`: Phase 0 proposals, eligibility and diversity checks, the fixed critique ring, critique, rebuttal, open-item handling, and winner selection. Debate is mandatory. A scope objection may be closed only by the lead or critic, and only with cited evidence that the declared production boundary was actually measured. An owner cannot close its own objection.

Only production-boundary magnitude may enter expected-value ranking. Proxy or synthetic evidence can show feasibility or a bound; it can never supply production E2E magnitude. `references/debate-rules.md` owns the evidence ladder and `references/e2e-delta-math.md` owns projection.

### Stages 4-5: Parallel Tracks

For each winner, the lead creates a distinct branch/worktree, records its stable identity in state, then spawns one `ammo-implementer` and one read-only `ammo-transcript-monitor`. The monitor reviews methodology continuously and runs its own collision checks; it does not implement and does not author the track verdict.

Keep all five binding gates — 5.1a, 5.1b, 5.2, 5.3a, 5.3b — as defined in `references/validation-defaults.md`.

`references/validation-defaults.md` owns the exact comparator, tolerances, activation proof, production-parity rules, and verdict semantics. Before a track may report `FAIL`, the implementer must exhaust the applicable fixes and the canonical fallback path in `references/impl-track-rules.md`. Route unexplained failures to `ammo-investigator`. `DILUTED_PASS` is discovered only after validation; it stays canonical status `PASS` with `diluted:true` and contributes measured E2E only.

Do not enter Stage 6 until every track is terminal: `PASS`, `GATED_PASS`, or `FAIL`. Reconcile state against the primary gate artifacts before that transition. Once all tracks are terminal, run the projection-accuracy diagnostics and `T_AUDIT_S45`.

### Stage 6: Integration Validation

Follow `orchestration/integration-logic.md` and keep the whole decision matrix:

- No passer: the round is EXHAUSTED.
- One unchanged passer: the single-track short-circuit is permitted.
- Multiple compatible passers: compose them, then rerun full-model correctness and a clean E2E.
- Overlap, gating, or merge ambiguity: use the prescribed dominance logic, or spawn `ammo-resolver` plus an independent merge review.

On SHIP: run the pre-ship mechanical checks, integrate the exact validated change, promote its environment contract, clear the experimental environment, refresh the golden references after promotion, and run `T_AUDIT_S67`. The canonical Stage-6 integration result becomes the new production baseline — a combined integration sweep supplies it when several tracks pass, and a single-pass short-circuit copies the validated Stage-5 result into the same integration slot. The former separate T16 re-profile stays eliminated. Compute cumulative gain as a direct ratio — original Round-1 baseline latency divided by current integrated latency — and never multiply round deltas.

### Stage 7: Campaign Evaluation

Stage 7 is autonomous and mechanical: stop iff the current production baseline's `bottleneck_mining.top_addressable_e2e_pct` — the largest measured `f_e2e × removable_fraction` across non-overlapping slices — is below `min_e2e_improvement_pct`; otherwise continue. A material SHIP may never use its pre-promotion mining for this decision.

When the stop condition fires, the terminal token records how the round ended: use `campaign_complete` after a SHIP integration status (`single_pass`, `combined`, or `gated_pass`) and `campaign_exhausted` after an EXHAUSTED/failed integration. Never pick between them by narrative preference. Before you set the terminal token (and before Stage 7b), enumerate the active-round mapping and release every remaining agent per `references/shutdown-protocol.md`.

- After SHIP, if the change is material, create the next round at `2_bottleneck_mining` and mine the promoted prior-round Stage-6 integration/single-pass evidence. Do not capture a fresh baseline, a fresh profile, or a promoted-commit provenance sidecar. If the newly mined addressable impact is below the floor, terminate as `campaign_complete` from the audited SHIP outcome. Otherwise run the normal Stage-2 gate and `T_AUDIT_S2`, then enter `3_debate`.
- After EXHAUSTED, reuse the mining that is still valid, append structured `exhausted_technologies`, and enter `3_debate` with a genuine mechanism pivot.
- If mining was proven wrong, set boolean `mining_invalidated:true` with a reason and re-enter Stage 2.
- If every shipped track is `diluted:true`, reuse valid mining as for EXHAUSTED, because the measured baseline did not materially move.
- Let every track in the current cohort reach a terminal state before you start the next round.

Do not stop for qualitative "good enough," time, difficulty, or token reasons. `references/validation-defaults.md` owns the invalid-stop list and the minimum-improvement policy.

### Stage 7b: Report and Fact Check

At `campaign_complete` or `campaign_exhausted`, spawn a fresh `ammo-report`. It must build every factual claim from campaign artifacts, disclose missing or failed evidence, and get `ok:true` from a fresh adversarial fact-checker before returning the report.

## Audit Gates

The lead follows `orchestration/audit-protocol.md`. `references/audit-invariants.md` owns the stage-specific questions, and they reach the auditor at the prescribed time.

Every new round entry must carry `"audit": {}` from creation, so the mandatory audit gates cannot fail open.

For each of `T_AUDIT_S1`, `T_AUDIT_S2`, `T_AUDIT_S45`, and `T_AUDIT_S67`:

1. Spawn a fresh `ammo-auditor`. Do not preload the Phase 2 checklist.
2. The auditor first writes `## Phase 1 - Independent Reconstruction` from primary evidence and critically weighs any delegate findings.
3. The hook then delivers `## Phase 2 - Checklist Verification`.
4. The lead fixes consequential findings and re-audits. It never edits evidence to manufacture a pass.
5. After the bounded review loop, apply `audit-protocol.md` § Repair and Re-audit: Stage 45 may quarantine an attributable track; Stage 67 removes and remeasures the implicated promoted code; shared or uncertain scope pauses.

Auditors and delegates never write campaign evidence. Independent audit is the semantic gate; hook validation is only its mechanical backstop.

## State Management

`.codex/schemas/state.schema.json` owns the closed state shape and enums. `scripts/transitions.json` owns legal transitions. Do not restate or invent enum values in prose. Keep these stage values exactly:

`1_baseline` -> `2_bottleneck_mining` -> `3_debate` -> `4_5_parallel_tracks` -> `6_integration` -> `7_campaign_eval` -> `7b_report`.

At each real transition:

- set the current stage and its matching `started_at`/`completed_at` timestamps;
- write per-track gate, verdict, and metric fields when their primary artifacts land;
- write the typed `selected_candidates` contract and debate enrichment at selection;
- write mining enrichment from the structured `f_e2e` table before advancing;
- keep provenance, attempts, audit records, exhaustion memory, shipped identities, the environment contract, and the direct cumulative speedup.

Use `scripts/ammo_state.py` for state operations and validation. Do not cite retired shell-hook names. Frontend fields consume campaign truth; they are never a second policy source.

## Pause and Resume

Pause only when the user asks or the audit escalation path requires it. A pause keeps state and artifacts; it does not turn unfinished tracks into failures.

On resume:

1. Read this file, `state.json`, and the active round's primary artifacts.
2. Confirm `campaign.status`, `campaign.current_round`, and `campaign.current_stage`.
3. Inspect GPU reservations and clear only stale reservations owned by the crashed session.
4. Reconcile debate or track state with on-disk artifacts; never infer completion from narrative alone.
5. Reuse stable agent/task identities where possible and resume from the last verified gate.
6. If the campaign is terminal and `REPORT.md` is absent, run Stage 7b.

Repair malformed or legacy state through the schema/backfill path in `scripts/ammo_state.py`. Never rebuild it from guesses. Full artifact reconciliation lives in `orchestration/parallel-tracks.md`.

## Failure and Escalation

- Ordinary technical failure: investigate, fix it if the evidence says it is fixable, and continue the same stage.
- Unexplained correctness, activation, measurement, or integration failure: spawn `ammo-investigator` with the primary artifacts and one concrete decision question.
- Critical evidence or safety failure: halt the current stage, preserve the evidence, and resolve it before you advance.
- Escalate to the user only for missing authority or an unrecoverable conflict, never an ordinary engineering fork.

Record campaign-scoped blockers under `{artifact_dir}/blockers/`.

## Authority Index

Read only what the active task requires.

| Topic | Authority |
|---|---|
| Workload/configuration | `references/workload-invocation.md` |
| Artifact paths and authority | `references/artifact-layout.md` |
| GPU ownership/isolation | `references/gpu-pool.md` |
| Profiling | `references/nsys-profiling-guide.md` |
| Opportunity/projection math | `references/e2e-delta-math.md` |
| Evidence scope and micro-experiments | `references/debate-rules.md` |
| Candidate scoring | `references/debate-scoring-rubric.md` |
| Technology and category | `references/technology-selection.md`, `references/optimization-categories.md` |
| Validation and verdicts | `references/validation-defaults.md` |
| Implementation fallback/gating | `references/impl-track-rules.md` |
| torch.compile/CUDA graphs | `references/torch-compile-contract.md`, `references/cudagraph-safety.md` |
| Audit checklist | `references/audit-invariants.md` |
| Messaging/delegation | `references/champion-common-patterns.md` |
| Shutdown | `references/shutdown-protocol.md` |
| Debate choreography | `orchestration/debate-protocol.md` |
| Track choreography | `orchestration/parallel-tracks.md` |
| Integration | `orchestration/integration-logic.md` |
| Audits | `orchestration/audit-protocol.md` |

## Active Helpers

- `scripts/new_target.py`: scaffold target and state.
- `scripts/run_vllm_bench_latency_sweep.py`: authoritative baseline, opt, profiling, correctness, and integration runner.
- `scripts/check_projection_accuracy.py`: append `## Projection Accuracy` diagnostics.
- `scripts/gpu_reservation.py status --table` / `scripts/gpu_reservation.py force-clear --all --session-id <crashed_session_id>`: inspect or clear owned stale reservations.
- `scripts/ammo_state.py`: validate, update, transition, and backfill campaign state.

Run Python helpers through the active worktree `.venv/bin/python`. Never modify a helper just to get past a failed semantic gate.
