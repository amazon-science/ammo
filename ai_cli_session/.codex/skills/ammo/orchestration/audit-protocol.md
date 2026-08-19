# Audit Protocol (T_AUDIT)

Every audit is a fresh, independent sanity gate. The auditor first rebuilds what
the stage really did from primary evidence, and reads the institutional checklist
only after that. A stage may advance only after its audit produces a genuine
final `PASS`. This file owns dispatch,
triggers, remediation, and state recording; the lead applies it at every gate.

Two other files own the rest, so do not restate the auditor role here. The
auditor's reasoning, evidence, severity, and output contract live in
`agents/ammo-auditor.md`. The stage checklists live in
`references/audit-invariants.md`, and the hook delivers them only after the
auditor writes Phase 1.

## Mandatory Triggers

Spawn an auditor at every trigger below. No outcome ever earns an auto-pass.

| Trigger | Fire after | PASS state key |
|---|---|---|
| `T_AUDIT_S1` | Stage 1 baseline capture | `audit.stage_1.passed_at` |
| `T_AUDIT_S2` | Stage 2 bottleneck mining (schema v4.1+) | `audit.stage_2.passed_at` |
| `T_AUDIT_S45` | every Stage 4-5 track is terminal (`PASS`, `GATED_PASS`, or `FAIL`) | `audit.stage_45.passed_at` |
| `T_AUDIT_S67` | SHIP: merge, environment promotion, and golden capture are complete; EXHAUSTED: integration outcome is recorded | `audit.stage_67.passed_at` |

Stage 3 gets no audit of its own, because its cross-critique, rebuttal, and
open-item protocol is already adversarial. This single exception weakens none of
the four triggers above.

Leave transition machinery to `scripts/ammo_state.py`. It is the sole authority
for transition enforcement, new-round requirements, resumed-schema handling, and
legacy `stage_6` / `stage_7` backfill. Do not restate those branches here.

## Dispatch

Spawn a new `ammo-auditor` instance for every gate and for every re-audit. Never
reuse an auditor. Never hand it an earlier audit's conclusions as facts.

```python
spawn_agent(
    task_name=f"audit_{stage}_{round}_c{cycle}",
    agent_type="ammo-auditor",
    fork_turns="none",
    message=f"""
    task: audit_gate
    artifact_dir: {artifact_dir}
    stage: {stage}       # stage_1 | stage_2 | stage_45 | stage_67
    round: {round}
    cycle: {cycle}
    """,
)
```

Keep the message that thin. The auditor uses the exact supplied artifact directory
and resolves canonical paths from it. Its authority and workflow come from
`agents/ammo-auditor.md`; do not copy them into the dispatch prompt. What
holds in every audit:

- the auditor writes its assigned verdict, but cannot modify source, campaign
  artifacts, or `state.json`;
- Phase 1 is an independent reconstruction, so the auditor must not read
  `references/audit-invariants.md` before it writes Phase 1;
- the hook then injects the Phase 2 checklist, and Phase 2 cannot erase or
  downgrade a Phase 1 blocker;
- every evidence question beyond a direct state lookup or arithmetic is fanned
  out to a fresh `ammo-delegate` instance, whose answers are evidence for the
  auditor to challenge, not verdicts.

The lead may continue unrelated work while an audit reads evidence, but must not
perform the gated transition.

## Started-At Stamping

`audit.{stage}.started_at` and `.cycle` are stamped mechanically, not by the
lead. `.codex/hooks/post_tool_use_guard.py` reads this dispatch message the
moment the `ammo-auditor` spawn succeeds and writes both fields — this is why
the message format above is load-bearing. `passed_at` stays lead-written,
after the lead verifies the final PASS verdict. For `schema_version >= 4.2`,
state validation rejects a gate with `passed_at` set and `started_at` missing;
backfill it with
`ammo_state.py audit-started --artifact-dir DIR --stage {stage} --round {round} --cycle {cycle}`.

The same rule covers the pre-consolidation keys `audit.stage_6` and
`audit.stage_7`, which the S67 gate still accepts in place of `stage_67`.
`audit-started` cannot address them, so migrate the verdict to `audit.stage_67`
and stamp that gate.

## Optional Early S45 Evidence

You may overlap evidence collection with a slow sibling track. When one track
becomes terminal while a sibling is still running, the lead may dispatch one
read-only `ammo-delegate` to gather that track's per-track checklist evidence
into:

`rounds/{N}/audits/stage_45_partial_{op_id}.md`

Overlapped collection is all this buys. The partial is not a verdict, never
advances the gate, and contains no round-level checks. `T_AUDIT_S45` still fires
once, with a fresh auditor, only after all tracks are terminal. That auditor must
critically recheck every partial, redispatch weak evidence, and gather all
round-level evidence fresh. Before any S45 re-audit, delete every partial for the
round and regather it, because a repair may have changed shared inputs.

## Verdict Handling

The verdict file, not an auditor message, is authoritative for the audit result:

- `PASS`: no `BLOCKING` or `HIGH` finding remains. Only now may the lead record
  PASS metadata and continue.
- `BLOCKED`: at least one `BLOCKING` finding remains. Apply a bounded repair and
  re-audit.
- `NEEDS_INVESTIGATION`: consequential uncertainty remains. Dispatch a bounded
  investigator, resolve its severity, and re-audit; do not advance or stamp
  PASS.

A PASS is genuine only when the verdict contains the independent Phase 1, the
hook-delivered Phase 2, reconciliation against primary evidence, and an explicit
final `Overall: PASS`. This file defines no separate attestation record: PASS
provenance is the complete verdict plus the state record that points to it.

After verifying that final PASS, the lead records both fields in the current
round:

```text
audit.{stage}.passed_at = <ISO-8601 UTC>
audit.{stage}.verdict_file = "rounds/{N}/audits/{verdict_filename}"
```

Never stamp either field from a partial file, an auditor message, a Phase 1-only
file, or a verdict that still contains `BLOCKING` or `HIGH` findings.

## Repair and Re-audit

Send each finding to the narrowest capable fix owner. Pick that owner from the
finding's evidence, category, and scope. The auditor never repairs its own
finding, and the lead never repairs a stage deliverable inline. Mining-artifact
findings go to `ammo-researcher`, which owns the derivation chain and regenerates
tables from primary data instead of hand-patching values. After a repair, spawn a
fresh auditor with the same stage and round; it must verify the repair from
primary evidence and look for collateral damage.

An attributable Stage-67 track rejection needs the track's own implementer.
Immediately resume or spawn that track's `ammo-implementer` in an isolated
repair worktree to remove its code and environment contract. Use `ammo-resolver`
only when reconstructing the remaining composition introduces a merge/semantic
conflict. The lead owns state and promotion bookkeeping but does not edit source.
Preserve the track's historical Stage-5 verdict, and record the promotion
rejection in `dropped`, `integration.final_decision`, and the round summary
before re-audit.

Allow at most three audit-fix cycles at a gate. If blockers remain, keep the
narrowest safe scope, and keep no rejected code:

- At Stage 45, if every blocker is attributable to named `op_id` values,
  quarantine those tracks as `FAIL` with an `audit_escalation` reason, record a
  track-scoped `campaign.auditor_escalation`, and audit the surviving scope.
- At Stage 67, code and environment are already promoted, so a track-scoped
  blocker requires removing the implicated change and environment contract,
  rerunning integration correctness, E2E, activation, and golden capture, then
  auditing that reconstructed scope. Merely marking the track `FAIL` is never
  sufficient. If safe attribution or reconstruction is unavailable, pause as
  campaign-scoped.
- If any blocker affects shared baseline, environment, workflow, decision,
  integration, or campaign evidence, set `campaign.status = "paused"` and record
  a campaign-scoped `campaign.auditor_escalation` with a resumable task.
- If scope is uncertain, treat it as campaign-scoped.

The post-quarantine audit may stamp PASS only if the remaining scope genuinely
passes. If it exposes a campaign-scoped blocker, pause instead. Auditor
escalation is the sole orchestrator-initiated pause allowed without a user pause
request.

## Artifact Paths

All new verdicts are round-scoped:

```text
rounds/{N}/audits/stage_1.md
rounds/{N}/audits/stage_2.md
rounds/{N}/audits/stage_45.md
rounds/{N}/audits/stage_67.md
rounds/{N}/audits/{stage}_cycle_{C}.md
rounds/{N}/audits/stage_45_partial_{op_id}.md   # optional evidence, never verdict
```

The first audit uses the unadorned filename; re-audits add `_cycle_{C}`. The
auditor writes only its assigned cycle file. After verifying a final passing
cycle, the lead copies that complete verdict to canonical `stage_67.md` and
records both paths/provenance in state or the round summary. See
`references/artifact-layout.md` for the full tree. Historical path and state-key
normalization belongs to `scripts/ammo_state.py`, not this protocol.

## References

- `agents/ammo-auditor.md` — independent reconstruction, delegate fanout,
  severity, and verdict contract
- `agents/ammo-delegate.md` — bounded evidence helper
- `references/audit-invariants.md` — hook-delivered Phase 2 checklist
- `references/artifact-layout.md` — canonical artifact locations
- `scripts/ammo_state.py` — state validation, compatibility, and transitions
- `.codex/hooks/post_tool_use_guard.py` — Phase 2 delivery, audit reminders, and
  `started_at`/`cycle` stamping on auditor dispatch
