# Stages 4-5: Parallel Track Orchestration

Each selected debate winner becomes one track: one branch, one worktree, one
implementation champion, and one monitor paired to that champion. This file
owns cohort identity, dispatch, pairing, reconciliation, and the Stage 6
barrier. Read it before you spawn the first track agent, and again before you
try to leave Stage 5.

The rest lives elsewhere. Implementation behavior belongs to
`.claude/agents/ammo-impl-champion.md`. Continuous review belongs to
`.claude/agents/ammo-transcript-monitor.md`. Gate and verdict policy belongs to
`references/impl-track-rules.md` and `references/validation-defaults.md`.

## One Worktree and One Agent Pair per Track

The lead must do all three steps below for every selected raw `op_id`.

1. Spawn the track's `ammo-impl-champion` with the Agent tool. Its
   `isolation: worktree` frontmatter plus the `WorktreeCreate` hook
   (`worktree-create-with-build.sh`) create the distinct branch and worktree,
   with Python isolation and a per-worktree `.venv`. Capture the worktree's
   absolute path and branch, then persist that identity.
2. Record the raw `op_id` to branch/worktree mapping, the planned agent names,
   and the round-scoped observation/offsets/terminal-summary paths before
   monitor dispatch and before Stage-4 entry. Record the champion's exact agent
   name (`implementer_agent`) and agent id (`implementer_rollout_id`, from its
   `agent-*.meta.json` sidecar) as soon as it is spawned.
3. Spawn exactly one `ammo-impl-champion` and one continuously paired
   `ammo-transcript-monitor` per track, each with a `name`. Both join the
   session's single implicit team, so there is no TeamCreate and you must not
   pass `team_name`.

Raw `op_id` values stay the keys in `state.json` and in artifact paths. Use
stable runtime names such as `impl-champion-{op_id}` and
`monitor-impl-champion-{op_id}`, and address agents by `name` via SendMessage.
The descriptive round label in `state.json.campaign.rounds[$IDX].team_name` is
metadata, not a CC team handle.

Spawn from the session worktree root (`$CLAUDE_PROJECT_DIR`) so children inherit
the correct cwd; if you inspected a track worktree, `cd` back before the next
`Agent(...)`. Pass the captured absolute worktree path to the champion, which
uses its worktree-local venv. GPU ownership and world size are governed by
`references/gpu-pool.md`. Worktree build and source-ownership rules are governed
by `references/impl-track-rules.md`.

## What to Send the Champion

Supply these fields. Do not copy the champion role into the prompt.

- `op_id`, round number, artifact directory, absolute worktree path, and
  `worktree_branch`;
- the typed selected-candidate entry and its `proposal_file`;
- the round's mining artifact, constraints, frozen target, and baseline
  artifact locations (Stage 1 baseline: per-BS JSONs under
  `rounds/{N}/sweeps/baseline/`, summary in `rounds/{N}/constraints.md` — DO
  NOT re-run);
- precision classification and the campaign's configured thresholds;
- TP/DP, GPU count, and the reservation session-id to use;
- pointers to `.claude/agents/ammo-impl-champion.md`,
  `references/artifact-layout.md`, `references/gpu-pool.md`,
  `references/impl-track-rules.md`, and `references/validation-defaults.md`.

The typed selected-candidate entry is the implementation mandate. Rendered
debate summaries are views of it, never a replacement for it.

The champion owns source changes, tests, commits, track gate artifacts, and
`validation_results.md`. The lead does not implement, does not author the track
verdict, and does not join the validation loop.

## What to Send the Monitor

Monitors read the champion's session transcript JSONL under the projects
directory: `os.path.join(os.environ.get("CLAUDE_CONFIG_DIR",
os.path.expanduser("~/.claude")), "projects", cwd.replace("/", "-"))`. Supply
the monitor with:

- target agent name: `impl-champion-{op_id}`
- `target_rollout_id`: the champion's agent id (must match state
  `implementer_rollout_id`; the monitor echoes it in offsets and summary)
- projects dir (computed as above)
- artifact_dir, round_number, `op_id`, `worktree_branch`
- output_dir: `{artifact_dir}/rounds/{N}/tracks/{op_id}/monitor_audits`
- precision classification
- pointers to `.claude/agents/ammo-transcript-monitor.md` and its topical
  authorities

The monitor is read-only with respect to track and campaign evidence. It covers
the champion's full active lifetime without a break. It writes its own durable
observation log and transcript offsets under `monitor_audits/`. It records each
intervention durably first, then delivers it to the champion (and escalations to
the lead) via SendMessage.

## Collisions and Monitor Handoff

Each monitor independently reads
`campaign.rounds[$IDX].parallel_tracks.tracks` on every poll to detect a shared
branch or a duplicate live agent. This cross-track check is read-only.

On a collision the lead identifies the authoritative agent and stops only a
stale writer that is still running (SendMessage `shutdown_request`). Never stop
the authoritative one.

A stale transcript or a monitor safety bound starts an acknowledged handoff. The
current monitor keeps reduced-cadence coverage until a replacement validates the
same target and assumes coverage, or until the champion ends. Neither staleness
nor timeout is track completion.

## Evidence to Leave Behind, and Who Writes It

`references/artifact-layout.md` defines the canonical paths and primary-artifact
precedence. In those round-scoped locations, each track must leave at minimum
its current commit/diff, validation report, applicable gate artifacts,
activation evidence, clean correctness and timing outputs, and monitor audit
evidence. Failed and blocked attempts retain their evidence too.

Ownership is exclusive:

- the champion writes source, tests, commits, track evidence, and its verdict
  report;
- the monitor writes only its observation/offset artifacts and durable
  intervention records;
- the lead is the sole writer of campaign track state and reconciliation;
- delegates and auditors remain within their own read/write contracts.

Monitor interventions are recorded in durable plaintext before delivery. The
lead acknowledges and resolves escalated findings. It does not rewrite track
evidence to close them.

## Mechanical Validation Handoff

Scaffold the track's `evidence.json` with `scripts/create_evidence_template.py`
before implementation. The champion fills it from primary artifacts and keeps
`validation_results.md` human-authored.

After a track's evidence is complete, run
`scripts/generate_validation_report.py` with `--output-md` directed to
`rounds/{N}/tracks/{op_id}/_scratch/generated_validation.md`. That emits the
machine-readable `validation_summary.json` and does not overwrite the
champion-owned verdict. The lead then runs
`scripts/reconcile_track_state.py --write` for that track.

After all tracks are reconciled, run `scripts/verify_validation_gates.py` once
for the cohort with `--json-output rounds/{N}/validation_gate_report.json`.

These machine artifacts are durable inputs to `T_AUDIT_S45`. They do not
replace the champion's judgment, the continuous monitor review, or the
independent audit.

## Reconciliation and Cohort Barrier

The lead reconciles each raw `op_id` entry atomically as primary artifacts land,
not only after the stage ends. Merge observed fields and invent no placeholders
(atomic `.tmp` + `os.replace`; keys not yet observed stay absent or null).
Preserve evidence provenance. Repair any disagreement between state and the
primary files before you use that track in a decision. State shape,
status/verdict enums, and legal transitions are owned by
`.claude/schemas/state.schema.json` and `scripts/transitions.json`.

A blocker that still needs lead triage is not a terminal track.

The selected winner ids, the typed selected-candidate ids, and the track keys
must match exactly. Each track must also retain its champion/monitor pairing
record.

Do not begin Stage 6, re-profile, or open the next campaign round until every
selected track has reached a schema-defined terminal state and all
champion/monitor coverage has been reconciled. This all-tracks-terminal barrier
applies even when one track finishes or appears shippable early.

Once the barrier is satisfied:

1. Re-read every track's primary evidence and reconcile final state.
2. Resolve or acknowledge all consequential monitor findings.
3. Run projection-accuracy diagnostics and `T_AUDIT_S45`.
4. Advance only after the audit and mechanical transition checks pass.

Early per-track audit evidence gathering is optional and may overlap a slow
sibling, as defined by `orchestration/audit-protocol.md`. It neither completes
`T_AUDIT_S45` nor relaxes the barrier.

## Shutdown and Cleanup

Drain the round's teammates after the cohort is reconciled. Send
`SendMessage(to=<name>, message={"type": "shutdown_request"})` to each champion
and each monitor, and confirm each `shutdown_approved` before moving on, subject
to the bounded 5-minute wait of `references/shutdown-protocol.md` (an idle
transcript counts as drained).

There is no TeamDelete: the session's implicit team has no agent-callable
teardown, and its shared dirs are auto-reclaimed at session exit.

Remove isolated worktrees (`git worktree remove {worktree_path} --force`, failed
tracks included) only after Stage 6 no longer needs them, or after a track is
formally abandoned. Follow `references/shutdown-protocol.md` for release and
cleanup semantics.
