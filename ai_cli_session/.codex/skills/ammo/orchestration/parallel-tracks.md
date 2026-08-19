# Stages 4-5: Parallel Track Orchestration

Each selected debate winner becomes one track: one branch, one worktree, one
implementer, and one monitor paired to that implementer. This file owns cohort
identity, dispatch, pairing, reconciliation, and the Stage 6 barrier. Read it
before you spawn the first track agent, and again before you try to leave
Stage 5.

The rest lives elsewhere. Implementation behavior belongs to
`agents/ammo-implementer.md`. Continuous review belongs to
`agents/ammo-transcript-monitor.md`. Gate and verdict policy belongs to
`references/impl-track-rules.md` and `references/validation-defaults.md`.

## One Worktree and One Agent Pair per Track

The lead must do all three steps below for every selected raw `op_id`.

1. Create a distinct branch and worktree with
   `scripts/create_worktree_with_build.sh`, capture its returned absolute
   worktree path, and persist that identity.
2. Record the raw `op_id` to branch/worktree mapping, the planned canonical
   agent identities, and the round-scoped
   observation/offsets/terminal-summary paths before dispatch. Record the exact
   `implementer_rollout_id` immediately after the implementer is indexed. Bind
   all of these fields before monitor dispatch and before Stage-4 entry.
3. Spawn exactly one `ammo-implementer` and one continuously paired
   `ammo-transcript-monitor`, both with `fork_turns="none"`.

Raw `op_id` values stay the keys in `state.json` and in artifact paths. Derive
lowercase task slugs only for stable runtime names such as
`implementer_r{round}_{task_slug}` and `monitor_r{round}_{task_slug}`. Address
agents by their canonical absolute task paths. A descriptive round label is
metadata, not a runtime team handle.

Spawn from the session worktree root so children inherit the correct cwd. Pass
the captured absolute worktree path to the implementer, which uses its local
venv. GPU ownership and world size are governed by `references/gpu-pool.md`.
Worktree build and source-ownership rules are governed by
`references/impl-track-rules.md`.

## What to Send the Implementer

Supply these fields. Do not copy the implementer role into the prompt.

- `op_id`, round number, artifact directory, absolute worktree path, and
  `worktree_branch`;
- the typed selected-candidate entry and its `proposal_file`;
- the round's mining artifact, constraints, frozen target, and baseline
  artifact locations;
- precision classification and the campaign's configured thresholds;
- TP/DP, GPU count, and the injected agent-scoped GPU owner;
- pointers to `agents/ammo-implementer.md`, `references/artifact-layout.md`,
  `references/gpu-pool.md`, `references/impl-track-rules.md`, and
  `references/validation-defaults.md`.

The typed selected-candidate entry is the implementation mandate. Rendered
debate summaries are views of it, never a replacement for it.

The implementer owns source changes, tests, commits, track gate artifacts, and
`validation_results.md`. The lead does not implement, does not author the track
verdict, and does not join the validation loop.

## Resolve the Transcript Before You Dispatch the Monitor

Resolve the implementer's exact rollout after it is indexed. Use
`scripts/resolve_codex_transcript.py` with the campaign's root `codex_thread_id`
and the implementer's canonical agent path. Do not make a monitor scan
directories, choose by modification time, or guess which resumed rollout is
authoritative.

Bind the queue-routing identity exactly as
`campaign_session_id = state["codex_thread_id"]`. This is the root Codex UUID.
It is never `state["session_id"]`, an AMMO server session id, or a child rollout
id. Use this same value for transcript resolution, monitor dispatch, and every
monitor queue record's `target_session_id`.

## What to Send the Monitor

Supply the monitor with:

- campaign_session_id: `{campaign_session_id}` (exactly
  `state["codex_thread_id"]`, the root Codex UUID)
- target_rollout_id: `{target_rollout_id}`
- target_agent: `{canonical_implementer_path}`
- target_transcript_path: `{target_transcript_path}`
- collision_candidate_paths: `{bounded_candidate_paths}`
- artifact_dir: `{artifact_dir}`
- round_number: `{round_number}`
- op_id: `{op_id}`
- worktree_branch: `{worktree_branch}`
- output_dir: `{artifact_dir}/rounds/{round_number}/tracks/{op_id}/monitor_audits`
- monitor_queue_path: `{artifact_dir}/monitor_interventions.jsonl`
- monitor identity and implementer task name
- precision classification
- pointers to `agents/ammo-transcript-monitor.md` and its topical authorities

The monitor is read-only with respect to track and campaign evidence. It covers
the implementer's full active lifetime without a break. It writes its own
durable observation log, transcript offsets, and permitted queue records as
defined by `agents/ammo-transcript-monitor.md`.

## Collisions and Monitor Handoff

Each monitor independently reads the active round's track mapping to detect a
shared branch or a duplicate live rollout.

On a collision the lead resolves it, identifies the authoritative rollout, and
interrupts only a stale writer that is still running.

A stale transcript or a monitor safety bound starts an acknowledged handoff. The
current monitor keeps reduced-cadence coverage until a replacement validates the
same target and assumes coverage, or until the implementer ends. Neither
staleness nor timeout is track completion.

## Evidence to Leave Behind, and Who Writes It

`references/artifact-layout.md` defines the canonical paths and primary-artifact
precedence. In those round-scoped locations, each track must leave at minimum
its current commit/diff, validation report, applicable gate artifacts,
activation evidence, clean correctness and timing outputs, and monitor audit
evidence. Failed and blocked attempts retain their evidence too.

Ownership is exclusive:

- the implementer writes source, tests, commits, track evidence, and its
  verdict report;
- the monitor writes only its observation/offset/queue artifacts;
- the lead is the sole writer of campaign track state and reconciliation;
- delegates and auditors remain within their own read/write contracts.

Monitor interventions are recorded in durable plaintext before delivery. The
lead acknowledges and resolves escalated queue records. It does not rewrite
track evidence to close them.

## Mechanical Validation Handoff

Scaffold the track's `evidence.json` with `scripts/create_evidence_template.py`
before implementation. The implementer fills it from primary artifacts and keeps
`validation_results.md` human-authored.

After a track's evidence is complete, run
`scripts/generate_validation_report.py` with `--output-md` directed to
`rounds/{N}/tracks/{op_id}/_scratch/generated_validation.md`. That emits the
machine-readable `validation_summary.json` and does not overwrite the
implementer-owned verdict. The lead then runs
`scripts/reconcile_track_state.py --write` for that track.

After all tracks are reconciled, run `scripts/verify_validation_gates.py` once
for the cohort with `--json-output rounds/{N}/validation_gate_report.json`.

These machine artifacts are durable inputs to `T_AUDIT_S45`. They do not replace
the implementer's judgment, the continuous monitor review, or the independent
audit.

## Reconciliation and Cohort Barrier

The lead reconciles each raw `op_id` entry atomically as primary artifacts land,
not only after the stage ends. Merge observed fields and invent no placeholders.
Preserve evidence provenance. Repair any disagreement between state and the
primary files before you use that track in a decision. State shape,
status/verdict enums, and legal transitions are owned by
`.codex/schemas/state.schema.json` and `scripts/transitions.json`.

A blocker that still needs lead triage is not a terminal track.

The selected winner ids, the typed selected-candidate ids, and the track keys
must match exactly. Each track must also retain its implementer/monitor pairing
record.

Do not begin Stage 6, re-profile, or open the next campaign round until every
selected track has reached a schema-defined terminal state and all
implementer/monitor coverage has been reconciled. This all-tracks-terminal
barrier applies even when one track finishes or appears shippable early.

Once the barrier is satisfied:

1. Re-read every track's primary evidence and reconcile final state.
2. Resolve or acknowledge all consequential monitor findings.
3. Run projection-accuracy diagnostics and `T_AUDIT_S45`.
4. Advance only after the audit and mechanical transition checks pass.

Early per-track audit evidence gathering is optional and may overlap a slow
sibling, as defined by `orchestration/audit-protocol.md`. It neither completes
`T_AUDIT_S45` nor relaxes the barrier.

## Shutdown and Cleanup

After the cohort is reconciled, interrupt only implementer or monitor turns that
are still running; completed or idle agents need no close action.

Remove isolated worktrees only after Stage 6 no longer needs them, or after a
track is formally abandoned. Follow `references/shutdown-protocol.md` for
release and cleanup semantics.
