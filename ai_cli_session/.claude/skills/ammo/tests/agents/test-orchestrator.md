# Orchestrator (Lead) Conformance Tests

Workflow conformance tests for the AMMO lead orchestrator. Verifies the agent correctly follows the campaign workflow: current-round track monitoring, resume after interruption, campaign evaluation, integration decisions, role boundaries, and non-negotiable violation detection.

All state snippets below use the **v2 round-centric shape**
(`ai_cli_session/.claude/schemas/state.schema.json`, `$id = ammo.campaign.state/v2`).
Only the fields needed to disambiguate a scenario are shown; every scenario
also assumes the bootstrap-required fields (`target`, `session_id`,
`gpu_resources`, `campaign.config`, all stage sub-objects on each round entry)
are present with canonical defaults.

Key v2 reminders when reading these scenarios:

- Top-level `stage`, `parallel_tracks`, `integration`, `debate`, `summary`, and
  `stage_timestamps` no longer exist. Stage lives at `campaign.current_stage`;
  per-round state lives under `campaign.rounds[N-1]`.
- Track status enum is `{IN_PROGRESS, PASS, GATING_REQUIRED, GATED_PASS, FAIL,
  GPU_BLOCKED}` (not `PASSED`/`FAILED`). Per-track failure reason is
  `tracks[op].fail_reason`.
- Thresholds live under `campaign.config` (e.g.
  `campaign.config.min_e2e_improvement_pct`).

Claude runtime note (CC 2.1.179 implicit-team model): every session has ONE implicit team, auto-formed when the first named teammate is spawned (there is no `TeamCreate`/`TeamDelete` — both tools were removed in 2.1.179). A "round team" is therefore a logical cohort of named teammates (champions/impl-champions/monitors) living in the session's implicit team; the round `team_name` in state.json is an AMMO descriptive state-label, NOT a Claude team handle. Teammates join by being spawned with a `name`. Teardown is by SendMessage `shutdown_request` → confirm `shutdown_approved`; the implicit team itself is auto-reclaimed at session exit. Debate champions are shut down with `shutdown_request` after selection; the rest of the round cohort persists across debate and implementation until Stage 4-5 tracks are resolved.

## How to Run

```
Run the AMMO orchestrator conformance tests. Spawn Sonnet subagents that:
1. Read these files first:
   - .claude/skills/ammo/SKILL.md
   - .claude/skills/ammo/orchestration/parallel-tracks.md
   - .claude/skills/ammo/orchestration/integration-logic.md
   - .claude/skills/ammo/orchestration/debate-protocol.md
2. Role-play AS the lead orchestrator (scaffolds, gates — never implements)
3. For each scenario, receive a state.json snapshot and context, then answer:
   - "Next actions (in order):"
   - "Must NOT do:"
   - "Skill reference:"

Run in 4 parallel batches:
- Batch A: Scenarios 1a-1c (Current-Round Track Monitoring)
- Batch B: Scenarios 2a-2c (Resume After Interruption)
- Batch C: Scenarios 3a-3c, 4a-4c (Campaign Eval + Integration)
- Batch D: Scenarios 5a-5b, 6a-6d (Role Boundaries + Violation Detection)

Grade each response against the "Expected Behavior" column.
```

## Test Scenarios

### Category 1: Current-Round Track Monitoring

**Scenario 1a: Stage 4-5, Round 2, tracks still running**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 2,
    "current_stage": "4_5_parallel_tracks",
    "rounds": [
      { "round_id": 1, "status": "completed", "...": "..." },
      {
        "round_id": 2,
        "status": "IN_PROGRESS",
        "team_name": "ammo-round-2-llama70b-h100",
        "debate": {
          "started_at": "...", "completed_at": "...",
          "selected_winners": ["op001", "op002"]
        },
        "parallel_tracks": {
          "started_at": "...", "completed_at": null,
          "tracks": {
            "op001": { "status": "IN_PROGRESS" },
            "op002": { "status": "IN_PROGRESS" }
          }
        },
        "integration": { "started_at": null, "completed_at": null, "status": "pending" },
        "campaign_eval": { "started_at": null, "completed_at": null }
      }
    ]
  }
}
```
Context: Two impl tracks are running (impl-champion agents) in the round team.

Expected behavior: Monitor current-round implementation tracks only. Do not create a round-3 entry or spawn debate champions during Stage 4-5. Advance to Stage 6 only after all current-round tracks are terminal and audit gates allow.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Continue actively monitoring the two current-round impl tracks (op001 and op002) on `campaign.rounds[1]`.
2. As each impl-champion returns, run its compilation gate (T9) in the track worktree.
3. Update `campaign.rounds[1].parallel_tracks.tracks[op_id]` after each track has validated evidence (T10).
4. Run the Stage 4-5 validation gate report and record the result under the active round.
5. When all current-round tracks are terminal and audit gates allow, drain the round cohort (`ammo-round-2-llama70b-h100`): SendMessage `shutdown_request` to each remaining named teammate and confirm `shutdown_approved` (no `TeamDelete` in 2.1.179; the implicit team auto-cleans at session exit).
6. Set `campaign.current_stage = "6_integration"` only after there are no non-terminal current-round tracks.

**Must NOT do:**
- Append a future-round entry during Stage 4-5.
- Spawn debate champions while current-round implementation tracks are active.
- Implement anything yourself.
- Go idle while either current-round impl track is still running.
- Advance to Stage 6 while any current-round track is `IN_PROGRESS`, `GATING_REQUIRED`, or `GPU_BLOCKED`.

**Skill reference:**
- `SKILL.md` § Stages 4-5: "Do not advance to Stage 6 until all current-round implementation tracks are terminal."
- `orchestration/parallel-tracks.md`: `GPU_BLOCKED` is a lead-triage blocker, not terminal Stage 6 accounting.
</details>

---

**Scenario 1b: Stage 4-5, all current-round tracks terminal**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 1,
    "current_stage": "4_5_parallel_tracks",
    "rounds": [
      {
        "round_id": 1,
        "status": "IN_PROGRESS",
        "team_name": "ammo-round-1-llama70b-h100",
        "debate": { "selected_winners": ["op001"] },
        "parallel_tracks": {
          "started_at": "...", "completed_at": "...",
          "tracks": { "op001": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.12 } }
        },
        "integration": { "status": "pending" }
      }
    ]
  }
}
```
Context: First round of the campaign. The only impl track has returned and passed.

Expected behavior: Gate the completed track, close the active round agents/monitors, and proceed to Stage 6.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Run or confirm the compilation gate (T9) for op001 in its worktree.
2. Read `{artifact_dir}/rounds/{CR}/tracks/op001/validation_results.md` and ensure `campaign.rounds[0].parallel_tracks.tracks.op001` carries structured status/metrics (T10).
3. Run the Stage 4-5 validation gate report.
4. Satisfy T11 (all tracks terminal), then drain the round cohort (`ammo-round-1-llama70b-h100`): SendMessage `shutdown_request` to each remaining named teammate, confirm `shutdown_approved` (no `TeamDelete` in 2.1.179; implicit team auto-cleans at session exit).
5. Advance to Stage 6 by setting `campaign.current_stage = "6_integration"`.

**Must NOT do:**
- Start the next round's debate before Stage 6/7 campaign evaluation.
- Leave active impl-champion/monitor agents open after all current-round tracks are terminal.
- Implement anything yourself.

**Skill reference:**
- `SKILL.md` § Stages 4-5: "Proceed to integration only when `overall_status == \"PASS\"`."
- `SKILL.md` § Stage 6: "Only Stage-5-passing candidates may integrate."
</details>

---

**Scenario 1c: Current-round track is GPU_BLOCKED**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 3,
    "current_stage": "4_5_parallel_tracks",
    "rounds": [
      { "round_id": 1, "status": "completed", "...": "..." },
      { "round_id": 2, "status": "completed", "...": "..." },
      {
        "round_id": 3,
        "status": "IN_PROGRESS",
        "team_name": "ammo-round-3-llama70b-h100",
        "debate": { "selected_winners": ["op003", "op004"] },
        "parallel_tracks": {
          "tracks": {
            "op003": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.06 },
            "op004": { "status": "GPU_BLOCKED", "verdict": null, "fail_reason": "gpu_unavailable" }
          }
        }
      }
    ]
  }
}
```
Context: One current-round track passed. The other could not complete because GPUs were unavailable.

Expected behavior: Treat `GPU_BLOCKED` as non-terminal. Perform explicit lead triage before retrying, closing, or marking the round exhausted. Do not advance to Stage 6 yet.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Confirm op003's evidence and gate result are recorded.
2. Inspect op004's worktree/logs to determine whether the GPU block is transient, retryable, or requires abandoning the track.
3. If retryable, keep op004 non-terminal and retry under explicit lead control.
4. If not retryable, write a terminal `FAIL` with evidence and `fail_reason` before Stage 6 accounting.
5. Run the Stage 4-5 validation gate report only after every current-round track is terminal.
6. Advance to Stage 6 only after op004 is no longer `GPU_BLOCKED`.

**Must NOT do:**
- Count `GPU_BLOCKED` as terminal.
- Advance to Stage 6 while op004 is `GPU_BLOCKED`.
- Report candidate success for a track that never produced compliant evidence.
- Start future-round work before the current round is resolved.

**Skill reference:**
- `SKILL.md` § Stages 4-5: "`GPU_BLOCKED` requires explicit lead triage before retrying, closing, or marking the round exhausted."
- `orchestration/parallel-tracks.md`: `GPU_BLOCKED` is non-terminal and must not advance to Stage 6.
</details>

---

### Category 2: Resume After Interruption

**Scenario 2a: Resume into Stage 4-5 with current-round tracks active**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 2,
    "current_stage": "4_5_parallel_tracks",
    "rounds": [
      { "round_id": 1, "status": "completed", "...": "..." },
      {
        "round_id": 2,
        "status": "IN_PROGRESS",
        "team_name": "ammo-round-2-llama70b-h100",
        "parallel_tracks": {
          "tracks": {
            "op001": { "status": "IN_PROGRESS" },
            "op002": { "status": "IN_PROGRESS" }
          }
        }
      },
    ]
  }
}
```
Context: Resuming after compaction. Session was interrupted while round-2 impl tracks were running.

Expected behavior: Read SKILL.md + state.json, inspect active track artifacts, and resume monitoring or reconcile completed current-round tracks. Do not spawn debate champions or create a next-round entry.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json` to confirm the full state (op IDs, artifact directory, which round entries exist).
2. Check whether the two round-2 impl tracks are actually still running by inspecting their worktrees and artifact files.
3. **If impl tracks have already completed**: run compilation gate (T9) and update state.json (T10).
4. **If impl tracks are still in-flight**: resume monitoring, do not re-spawn them.
5. Run the Stage 4-5 validation gate report after all current-round tracks are terminal.
6. Close every active round agent/monitor pair and move to Stage 6 only after current-round terminal statuses and audit gates allow.

**Must NOT do:**
- Re-spawn impl agents without verifying they are not already complete or still running.
- Create a next-round entry while resuming Stage 4-5.
- Spawn debate champions during current-round implementation.
- Go idle while current-round impl tracks are unresolved.

**Skill reference:**
- SKILL.md § Stages 4-5: "On resume, inspect only the active round's implementation tracks and artifact files."
- SKILL.md § Stages 4-5: "Do not advance to Stage 6 until all current-round implementation tracks are terminal."
</details>

---

**Scenario 2b: Resume into Stage 3, debate team gone**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 1,
    "current_stage": "3_debate",
    "rounds": [
      {
        "round_id": 1,
        "status": "IN_PROGRESS",
        "team_name": "ammo-round-1-llama70b-h100",
        "debate": {
          "started_at": "...",
          "completed_at": null,
          "candidates": [],
          "rounds_completed": 0,
          "selected_winners": []
        }
      }
    ]
  }
}
```
Context: Resuming after interruption. A round cohort was spawned but no proposals exist. Teammates may be lost.

Expected behavior: Drain any stale/orphaned round teammates via SendMessage `shutdown_request` (confirm `shutdown_approved`) before re-spawning the logical round cohort. Re-create from scratch by spawning fresh named champions. Do NOT skip debate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json`. Also check `{artifact_dir}/debate/` for any proposal files.
2. Attempt to contact the round cohort via SendMessage to determine if any named teammates are still alive.
3. Confirm the cohort is lost: candidates empty, rounds_completed 0, no debate files.
4. Drain any stale/orphaned round teammates via SendMessage `shutdown_request` (confirm `shutdown_approved`) before re-spawning the round cohort.
5. Re-run Stage 3 from scratch: spawn fresh named champions directly (the first named spawn re-forms the session's implicit team; no `TeamCreate`), restart Phase 0.
6. Update `campaign.rounds[0].team_name` with the round's state-label (descriptive only; not a Claude team handle).
7. Moderate the debate through completion.

**Must NOT do:**
- Fabricate proposals from `bottleneck_analysis.md` directly as the lead.
- Skip the debate and proceed to Stage 4.
- Assume the old team is still alive.

**Skill reference:**
- SKILL.md § Pause and Resume, step 4: "Reconcile debate or track state with on-disk artifacts; never infer completion from narrative alone."
- SKILL.md § Stage 3: "Debate is always mandatory."
</details>

---

**Scenario 2c: Resume into Stage 7, SHIP decision made but no re-profile**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 1,
    "current_stage": "7_campaign_eval",
    "cumulative_speedup_vs_round1": 1.12,
    "shipped_optimizations": [
      { "op_id": "op001", "round": 1, "classification": "lossless" }
    ],
    "rounds": [
      {
        "round_id": 1,
        "status": "SHIPPED",
        "integration": { "status": "combined", "final_decision": { "action": "ship_combined", "total_e2e_speedup": 1.12 } },
        "campaign_eval": { "started_at": "...", "completed_at": null },
        "shipped": ["op001"]
      }
    ]
  }
}
```
Context: Resuming. A candidate shipped in round 1 but re-profiling hasn't happened yet.

Expected behavior: Trigger re-profiling on patched codebase, then bottleneck mining, then mechanical threshold check. Do NOT use stale data.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json` to confirm full campaign state.
2. Confirm ship decision is recorded on `campaign.rounds[0]` and in `campaign.shipped_optimizations`.
3. Execute T16: trigger re-profiling — invoke `ammo-researcher` subagent for baseline capture on the patched codebase.
4. After re-profile: execute T17 — bottleneck mining on the new baseline (updated `bottleneck_analysis.md`); record `rounds[0].bottleneck_mining.top_bottleneck_share_pct` (or the next-round equivalent if moving on).
5. Execute T18 (mechanical threshold check):
   - If below `campaign.config.min_e2e_improvement_pct`: set `campaign.status = "campaign_complete"`; spawn report subagent; done.
   - If above: increment `campaign.current_round`, append a new `campaign.rounds[...]` entry, set `current_stage = "3_debate"`.

**Must NOT do:**
- Skip re-profiling — SKILL.md explicitly requires it after SHIP.
- Check the mechanical threshold against the old `bottleneck_analysis.md`.
- Spawn the report subagent before confirming `campaign.status = "campaign_complete"` or `campaign_exhausted`.

**Skill reference:**
- SKILL.md § Stage 7: "After SHIP, if the change is material, create the next round at `2_bottleneck_mining` and mine the promoted prior-round Stage-6 integration/single-pass evidence."
</details>

---

### Category 3: Campaign Evaluation

**Scenario 3a: SHIP, top bottleneck below threshold after re-profile**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 2,
    "current_stage": "7_campaign_eval",
    "config": { "min_e2e_improvement_pct": 3 },
    "cumulative_speedup_vs_round1": 1.25,
    "shipped_optimizations": [
      { "op_id": "op001", "round": 1, "classification": "lossless" },
      { "op_id": "op003", "round": 2, "classification": "lossless" }
    ],
    "rounds": [
      { "round_id": 1, "status": "SHIPPED", "shipped": ["op001"] },
      {
        "round_id": 2,
        "status": "SHIPPED",
        "shipped": ["op003"],
        "bottleneck_mining": { "top_bottleneck_share_pct": 2.1 }
      }
    ]
  }
}
```
Context: Re-profiling done. New top bottleneck = 2.1% of decode latency (below 3% threshold).

Expected behavior: Set `campaign.status = "campaign_complete"`. Spawn report subagent in background. Do NOT start a new round. Leave `campaign.current_stage` at `7_campaign_eval` (or transition to `7b_report` when the report subagent starts).

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Ensure round 2 results are fully recorded on `campaign.rounds[1]`.
2. Update `campaign.shipped_optimizations` and `campaign.cumulative_speedup_vs_round1`.
3. Confirm 2.1% < 3% threshold (`campaign.config.min_e2e_improvement_pct`).
4. Set `campaign.status = "campaign_complete"`.
5. Run gate T19.
6. Set `campaign.current_stage = "7b_report"` and spawn the report generation subagent in background (T20).
7. Declare campaign done. Do not block on the report subagent.

**Must NOT do:**
- Proceed to a new debate round.
- Wait for the report subagent to finish.
- Start another round after the threshold check says the campaign is complete.

**Skill reference:**
- SKILL.md § Campaign Stop Condition: "If f < threshold... stop."
- Campaign State Transitions: `active → (threshold met after ship) → campaign_complete`
</details>

---

**Scenario 3b: EXHAUSTED, top bottleneck above threshold**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 2,
    "current_stage": "7_campaign_eval",
    "config": { "min_e2e_improvement_pct": 3 },
    "rounds": [
      { "round_id": 1, "status": "SHIPPED", "shipped": ["op001"] },
      {
        "round_id": 2,
        "status": "EXHAUSTED",
        "shipped": [],
        "bottleneck_mining": { "top_bottleneck_share_pct": 8.5 }
      }
    ]
  }
}
```
Context: Round 2 had no passing candidates. EXISTING profiling (round 2's bottleneck mining) shows top bottleneck at 8.5%.

Expected behavior: No re-profile (nothing shipped). 8.5% > 3% → campaign continues. New debate from existing data. Do NOT set `campaign_exhausted`.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Ensure the failed round 2 is recorded as `rounds[1].status = "EXHAUSTED"`.
2. Mechanical threshold check against EXISTING profiling data: 8.5% > 3% → campaign continues.
3. Run gate T19.
4. Increment `campaign.current_round` to 3. Append a new `campaign.rounds[2]` entry with `round_id: 3` and initialized stage sub-objects.
5. Start new debate from existing bottleneck data (skip re-profiling, skip Stage 2). Set `campaign.current_stage = "3_debate"`.

**Must NOT do:**
- Trigger re-profiling — nothing shipped.
- Set `campaign.status = "campaign_exhausted"` — threshold not met.
- Skip debate for the new round.

**Skill reference:**
- SKILL.md § Campaign Stop Condition: "After EXHAUSTED: Check threshold against EXISTING profiling data (no re-profile needed)."
</details>

---

**Scenario 3c: SHIP, top bottleneck above threshold starts a fresh next round**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 2,
    "current_stage": "7_campaign_eval",
    "rounds": [
      { "round_id": 1, "status": "SHIPPED", "shipped": ["op001"] },
      { "round_id": 2, "status": "SHIPPED", "shipped": ["op002"],
        "bottleneck_mining": { "top_bottleneck_share_pct": 7.0 } }
    ]
  }
}
```
Context: Round 2 shipped. Post-SHIP mining on the new baseline reports top bottleneck 7% (above threshold).

Expected behavior: Start round 3 from the new baseline mining results and run a fresh Stage 3 debate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Ensure round 2 shipped results are recorded on `rounds[1]`.
2. Confirm 7% > `campaign.config.min_e2e_improvement_pct` → campaign continues.
3. Run gate T19.
4. Advance `campaign.current_round` to 3.
5. Append a fresh `campaign.rounds[2]` entry with `round_id: 3` and initialized stage sub-objects.
6. Set `campaign.current_stage = "3_debate"` and spawn fresh champions for the new round.

**Must NOT do:**
- Reuse winners from a future round that has not started.
- Skip Stage 3 for round 3.
- Re-profile again — already done.

**Skill reference:**
- SKILL.md § Stage 7: "`f >= threshold`: continue unconditionally. Start the next round with a prior-failure note and require champions to run technology selection."
- SKILL.md § Stage 3: "Debate is always mandatory."
</details>

---

### Category 4: Integration

**Scenario 4a: Two tracks pass, different components**

State (excerpt):
```json
{
  "campaign": {
    "status": "active",
    "current_round": 1,
    "current_stage": "6_integration",
    "rounds": [
      {
        "round_id": 1,
        "parallel_tracks": {
          "tracks": {
            "op001": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.12,
                       "description": "vllm/attention/backends/flash_attn.py" },
            "op002": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.08,
                       "description": "csrc/quantization/gptq_marlin.cu" }
          }
        },
        "integration": { "status": "pending" }
      }
    ]
  }
}
```

Expected behavior: Cherry-pick both to integration branch, re-run correctness + E2E. Ship combined if better than best individual.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Run conflict detection: check file overlap between the two tracks' reported files_changed. Confirm disjoint.
2. Create integration branch from main.
3. Cherry-pick both passing tracks.
4. Run correctness tests.
5. Run combined E2E benchmark using sweep script.
6. Evaluate: combined E2E >= max(1.12, 1.08) → ship combined; else ship best individual.
7. Update `campaign.rounds[0].integration` (status, final_decision, combined_e2e_result).
8. Set `campaign.current_stage = "7_campaign_eval"`.

**Must NOT do:**
- Skip the combined E2E re-run.
- Pick just one without attempting combination.

**Skill reference:**
- integration-logic.md Decision Matrix: "Multiple pass, different components → Cherry-pick both."
</details>

---

**Scenario 4b: Two tracks pass, same component**

State (excerpt):
```json
{
  "campaign": {
    "current_stage": "6_integration",
    "rounds": [
      {
        "round_id": 1,
        "parallel_tracks": {
          "tracks": {
            "op001": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.15,
                       "description": "vllm/attention/backends/flash_attn.py + csrc/attention/flash_attn_kernel.cu" },
            "op002": { "status": "PASS", "verdict": "PASS", "e2e_speedup": 1.08,
                       "description": "vllm/attention/backends/flash_attn.py" }
          }
        },
        "integration": { "status": "pending" }
      }
    ]
  }
}
```

Expected behavior: Overlapping files → pick best E2E → op001 (1.15x). No combination attempt.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Conflict detection: `flash_attn.py` in both → overlap.
2. Pick op001 (1.15x > 1.08x).
3. Update `campaign.rounds[0].integration.status = "single_pass"` and record `final_decision`.
4. Set `campaign.current_stage = "7_campaign_eval"`.

**Must NOT do:**
- Attempt cherry-pick combination with overlapping files.
- Pick op002 (inferior E2E).

**Skill reference:**
- integration-logic.md: "Multiple pass, same component → Pick the candidate with the best E2E speedup."
</details>

---

**Scenario 4c: Zero tracks pass**

State (excerpt):
```json
{
  "campaign": {
    "current_stage": "6_integration",
    "rounds": [
      {
        "round_id": 1,
        "parallel_tracks": {
          "tracks": {
            "op001": { "status": "FAIL", "verdict": "FAIL", "fail_reason": "correctness regression" },
            "op002": { "status": "FAIL", "verdict": "FAIL", "fail_reason": "negative E2E impact" }
          }
        },
        "integration": { "status": "pending" }
      }
    ]
  }
}
```

Expected behavior: Round EXHAUSTED (not campaign-level). Move to Stage 7 for threshold check.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Update `campaign.rounds[0].integration.status = "exhausted"` and set `rounds[0].status = "EXHAUSTED"`.
2. Set `campaign.current_stage = "7_campaign_eval"`.
3. In Stage 7: mechanical threshold check on EXISTING profiling data (round 1's `bottleneck_mining.top_bottleneck_share_pct`).

**Must NOT do:**
- Trigger re-profiling (nothing shipped).
- Set `campaign.status = "campaign_exhausted"` yet — that's Stage 7's decision.
- Attempt to salvage failed candidates.

**Skill reference:**
- SKILL.md § Stage 6: "If none pass: round EXHAUSTED (not campaign-level — campaign evaluates in Stage 7)."
</details>

---

## Scenario 4d: GATED_PASS Track Integration

### Context
Stage 6 integration on `campaign.rounds[N-1]`. Two tracks completed:
- op001: `status: "PASS"`, `verdict: "PASS"`, `e2e_speedup: 1.12`, modifies `vllm/attention/backends/flash_attn.py`
- op003: `status: "GATED_PASS"`, `verdict: "GATED_PASS"`, `e2e_speedup: 1.025`, `gating: {env_var: "VLLM_OP003", crossover_threshold_bs: 16, regressing_bs: [32]}`, modifies `vllm/model_executor/layers/fused_moe/fused_moe.py`

Cherry-pick of op003 produces a merge conflict in `vllm/envs.py` (both tracks register new env vars).

### Expected Behavior
1. Orchestrator detects merge conflict on GATED_PASS track.
2. Spawns resolver agent (`ammo-resolver.md`) with conflicting files + both tracks' gating metadata.
3. Spawns DA reviewer to verify resolver's merge.
4. Does NOT simply pick best E2E and discard the other.
5. Records `resolver_invoked: true` and the resolver outcome on `campaign.rounds[N-1].integration`.

### Anti-Patterns (FAIL if observed)
- Treating merge conflict as "overlapping components" and picking best E2E.
- Skipping the resolver agent and resolving the conflict directly.
- Ignoring the GATED_PASS track's gating metadata during merge.

---

### Category 5: Role Boundaries

**Scenario 5a: Temptation to implement directly**

Context: Stage 4-5. An impl-champion returned with "CUDA kernel compilation failed — missing shared memory declaration in fused_attn.cu." You can see the bug — it's a one-line fix.

Expected behavior: Do NOT fix the kernel. Re-dispatch a new impl-champion with the error context.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Do NOT touch the code. Re-spawn a new ammo-impl-champion into the same worktree, providing the error message and context so the impl-champion can apply the fix itself.
2. Confirm other parallel tracks are still progressing.
3. When the impl-champion returns, run the compilation gate (T9).

**Must NOT do:**
- Edit `csrc/attention/fused_attn.cu` yourself, even for a one-line fix. The prohibition is unconditional: "Do not write kernel code (CUDA or Triton) yourself."

**Skill reference:**
- SKILL.md § Lead Role, "Prohibited": "Do not write kernel code (CUDA or Triton) yourself."
</details>

---

**Scenario 5b: Temptation to skip debate**

Context: Stage 2 complete. bottleneck_analysis.md shows flash_attn_fwd at 35% of decode latency — massively dominant. Next kernel at 4%. Obvious what to optimize.

Expected behavior: Full debate mandatory. Minimum 1 round (full A/B/C). Conditional 2nd round if any champion declares open items after Phase C. NEVER skip debate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Proceed with full Stage 3 debate. Spawn the logical round agent cohort (`ammo-round-{round_id}-{model_short}-{hardware}`), spawn 2-4 champions (no monitors for debate), broadcast Phase 0.
2. After round 1 Phase C, check champion open-items declarations. If any declare open items → round 2.
3. Run at least 1 full debate round (A/B/C + open items declaration).
4. Write summary.md, select winners, shut down debate champions via `shutdown_request`. Round team persists for Stages 4-5.

**Must NOT do:**
- Skip the debate.
- Unilaterally declare flash_attn_fwd the winner.
- Treat "obvious" dominance as a fast-track exception.
- shut down debate champions via `shutdown_request` after debate — the round team persists for implementation agents in Stages 4-5.

**Skill reference:**
- debate-protocol.md § "Debate is Always Mandatory": "There is no fast-track exception."
- SKILL.md § Stage 3: "After selection: Shut down debate champions via `shutdown_request`... The round team persists."
</details>

---

### Category 6: Non-Negotiable Violation Detection

**Scenario 6a: Researcher used `--enforce-eager` in profiling**

Context: ammo-researcher returned from Stage 1. Commands included `vllm bench latency --model meta-llama/Llama-3-70B --enforce-eager --batch-size 1,4,16`. Results look clean.

Expected behavior: FAIL the gate. Reject all results. Re-dispatch researcher with explicit violation callout. Do NOT advance to Stage 2.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. The researcher's Stop hook (DA) catches the production-parity violation (`--enforce-eager` is in its check #3) and returns `{ok: false}`. The researcher must re-run.
2. Document the blocker.
3. Re-spawn the ammo-researcher (task_type: baseline) with explicit instructions: re-run WITHOUT `--enforce-eager`. CUDA graphs + torch.compile must be active.
4. After successful return, run the mechanical state checks and audit on compliant results.

**Must NOT do:**
- Pass the gate because "the constraints.md looks clean." The measurement conditions are what matter.
- Proceed to Stage 2 on an `--enforce-eager` baseline.

**Skill reference:**
- references/validation-defaults.md § Forbidden Settings: "`--enforce-eager` # FORBIDDEN unless the frozen target requires it"
- "These are NOT advisory. Violation blocks stage progression."
</details>

---

**Scenario 6b: Impl-champion used raw `vllm bench latency` instead of sweep script**

Context: Impl-champion returned `verdict: "PASS"`. validation_results.md shows: `Command: vllm bench latency --model meta-llama/Llama-3-70B --batch-size 1 --num-iters 50`. Results look good — 12.7% improvement.

Expected behavior: FAIL the track. Raw `vllm bench latency` is FORBIDDEN. Re-dispatch with sweep script mandate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the Stage 4-5 gate for this track. E2E validation is non-compliant.
2. Document the compliance failure.
3. Re-spawn the ammo-impl-champion: E2E benchmark must use `run_vllm_bench_latency_sweep.py`. Forbid raw `vllm bench latency`.
4. Re-gate after re-run.

**Must NOT do:**
- Accept the PASS verdict because the 12.7% improvement looks good.
- Rationalize that raw invocations and the sweep script produce equivalent results.

**Skill reference:**
- references/e2e-latency-guide.md § Tool Selection: "For all measurements reported in `validation_results.md` or used for profiling, use the sweep script."
</details>

---

**Scenario 6c: Researcher set `TORCH_COMPILE_DISABLE=1`**

Context: ammo-researcher returned from Stage 1. Profiling command included `TORCH_COMPILE_DISABLE=1`. Researcher explained: "Disabled torch.compile to get cleaner nsys traces."

Expected behavior: FAIL the gate. Reject "cleaner traces" rationale. Re-dispatch without the flag.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the Stage 1 gate (T3). Profiling data is invalid — captured without torch.compile.
2. Document the blocker. Note the researcher's rationale and why it is not a valid exception.
3. Re-spawn researcher WITHOUT `TORCH_COMPILE_DISABLE=1`.
4. Re-gate on compliant profiles.

**Must NOT do:**
- Accept the "cleaner traces" rationale as a valid tradeoff.
- Proceed to Stage 2 on torch.compile-disabled profiles.

**Skill reference:**
- references/validation-defaults.md § Forbidden Settings: "`export TORCH_COMPILE_DISABLE=1` # FORBIDDEN unless the frozen target requires it"
- "These are NOT advisory. Violation blocks stage progression."
</details>

---

**Scenario 6d: Impl-champion's E2E validation used `VLLM_TORCH_COMPILE_LEVEL=0`**

Context: Impl-champion returned `verdict: "PASS"`. validation_results.md notes: `Environment: VLLM_TORCH_COMPILE_LEVEL=0 (to isolate kernel improvement from compile effects)`. E2E: 1.174x speedup.

Expected behavior: FAIL the track. Reject "pure kernel improvement" framing. Re-dispatch for compliant E2E.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the gate. `VLLM_TORCH_COMPILE_LEVEL=0` is explicitly forbidden.
2. Reject the "pure kernel improvement" framing: "There is no such category in the AMMO validation protocol — only production-parity E2E results count."
3. Mark the track FAIL in `campaign.rounds[N-1].parallel_tracks.tracks[op]` (status FAIL, verdict FAIL, fail_reason set).
4. Re-spawn impl-champion to re-run ONLY the E2E validation with the correct environment.
5. Re-gate with compliant results.

**Must NOT do:**
- Accept the result with a caveat.
- Try to mathematically adjust the result.
- Allow this track to proceed to Stage 6.

**Skill reference:**
- references/impl-track-rules.md § Track Constraints, Production parity: "NEVER use `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, or `VLLM_TORCH_COMPILE_LEVEL=0` to simplify the production comparator."
- parallel-tracks.md § Pass Criteria: "production parity" is an explicit pass criterion.
</details>

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct next action** | First action matches expected behavior | Wrong first action or wrong order |
| **Correct prohibitions** | Identifies what NOT to do | Misses a critical prohibition |
| **Skill citation** | References the specific section | Vague or no reference |
| **No hallucination** | All claims match the skill text | Invents rules not in the skill |

A scenario **passes** if all four criteria are met. The test suite **passes** if all 19 scenarios pass.

## Baseline Results

| Category | Scenarios | Count |
|----------|-----------|-------|
| Current-Round Track Monitoring | 1a, 1b, 1c | 3 |
| Resume After Interruption | 2a, 2b, 2c | 3 |
| Campaign Evaluation | 3a, 3b, 3c | 3 |
| Integration | 4a, 4b, 4c, 4d | 4 |
| Role Boundaries | 5a, 5b | 2 |
| Violation Detection | 6a, 6b, 6c, 6d | 4 |
| **Total** | | **19** |
