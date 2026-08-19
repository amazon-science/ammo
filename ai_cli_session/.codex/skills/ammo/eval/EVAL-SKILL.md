---
name: ammo-eval
description: Post-mortem evaluation for AMMO GPU kernel optimization campaigns. Scores completed campaign artifacts on a 5-dimension scorecard (E2E speedup, gates, debate quality, efficiency, transcript quality), archives versioned results, and generates a trend dashboard. Use after an /ammo campaign finishes to measure skill performance. Triggers on requests to evaluate, score, or measure ammo skill performance.
---

# AMMO Eval — Post-Mortem Campaign Evaluation

Score a completed AMMO campaign, archive the result, and generate a performance dashboard.

## When to Use

After an `/ammo` campaign completes (or is interrupted with results), invoke this skill to:
1. Parse all campaign artifacts into structured metrics
2. Score the run on a 5-dimension scorecard
3. Optionally grade transcript quality via LLM agent
4. Archive the scored run into a versioned repository
5. Rebuild aggregates and generate an HTML dashboard

## Arguments

The user provides:
- `--session-id` (required): exact root Codex thread UUID for the campaign (`threads.id` in `$CODEX_HOME/state_5.sqlite`). This is the only required argument — everything else can be auto-detected.
- `--artifact-dir` (auto-detected): Path to the campaign's artifact directory. **Auto-detection**: scan `kernel_opt_artifacts/*/state.json` for a matching `codex_thread_id` field (legacy fallback: `session_id`). If not found, scan every rollout listed in parsed `rollout_files` for writes to `kernel_opt_artifacts/` and infer the directory from the path.
- `--description` (optional, default: auto-generated from target info): Human-readable description of what skill changes were made
- `--skip-transcript-grading` (optional): Skip the LLM grader for speed
- `--skip-deep-analysis` (optional): Skip the causal LLM deep dive for speed
- `--repository` (optional, default `~/.codex/ammo-eval`): Eval repository path

### Minimal Invocation

The user can simply say: `Run /ammo-eval for session ba11c39a-...`

The orchestrator resolves everything:
1. Open `$CODEX_HOME/state_5.sqlite` read-only and resolve the exact `threads.id`; never guess by title, cwd, or filename.
2. Recursively traverse `thread_spawn_edges(parent_thread_id, child_thread_id, status)` and use each joined `threads.rollout_path` under `$CODEX_HOME/sessions/...`.
3. Find artifact dir by scanning `kernel_opt_artifacts/*/state.json` for matching `codex_thread_id` (legacy fallback: `session_id`).
4. If no match exists in state.json, inspect the resolved rollout set for writes to `kernel_opt_artifacts/` and extract the directory.
5. Auto-generate description from `target.json` (model + hardware + dtype).
6. Run the full pipeline (Steps 0-8) including causal analysis unless `--skip-*` flags are given.

Current Codex subagent rollouts can contain a fork-history replay. Always use `parse_session_logs.py`; its ownership-boundary filter prevents replayed parent events from being counted again. The historical `~/.codex/projects/.../<session-id>.jsonl` layout remains a fallback only through explicit `--session-dir` or automatic fallback when no current state entry exists.

## Pipeline

Run these scripts in sequence from `.codex/skills/ammo/eval/scripts/`:

### Step 0: Parse Session Logs
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/parse_session_logs.py \
  --session-id <SESSION_ID> \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_session_data.json
```

Resolves the root and recursive subagent graph from the read-only Codex state database, then extracts lifecycle, `threads.tokens_used`, tool counts, task durations, and stage timing from all owned rollout records. The output records every source in `rollout_files`, preserves edge status and agent path/role metadata, and replaces the older `campaign.agent_costs` fallback — no manual recording by the lead is needed.

### Step 1: Parse Artifacts
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/parse_artifacts.py \
  --artifact-dir <ARTIFACT_DIR> \
  --session-data /tmp/ammo_eval_session_data.json \
  --output /tmp/ammo_eval_snapshot.json
```

The `--session-data` flag provides session-log-derived timing and cost data. When omitted, falls back to the per-round stage timestamps on each `campaign.rounds[*].{stage}.{started_at,completed_at}` entry in state.json.

### Step 2: Snapshot Changes
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/snapshot_changes.py \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_changes_snapshot
```

This captures a full record of all code produced during the campaign:
- Git diffs (patches) from every campaign worktree across all `campaign.rounds[*].parallel_tracks.tracks` in `state.json`
- Session-member worktrees that shared the same session ID
- A complete copy of the artifact directory (minus large binary profiling files)
- Skill diff from the main branch
- A manifest summarizing worktree metadata, merge status, and changed files

The output directory is passed to `archive_run.py` in Step 5.

### Step 3: Score Campaign
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/score_campaign.py \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scorecard.json \
  --report /tmp/ammo_eval_report.md
```

### Step 3a: Prepare Eval Subagents

Codex does not provide a persistent Claude-style team object. Spawn independent eval subagents when needed and keep their agent IDs in the parent thread until the eval is complete.

```
spawn_agent(task_name="eval_causal_analyzer", fork_turns="none", message="<causal analyzer prompt>")
spawn_agent(task_name="eval_transcript_grader", fork_turns="none", message="<transcript grader prompt>")
```

Completed or idle eval subagents need no teardown call and remain addressable for follow-up through `followup_task` or `send_message`. Use `interrupt_agent` only for an agent that is still running and must be stopped.

### Step 3b: Extract Events + Build Causal DAG

```bash
.venv/bin/python .codex/skills/ammo/eval/causal/extract_events.py \
  --session-data /tmp/ammo_eval_session_data.json \
  --output /tmp/ammo_eval_events.json

.venv/bin/python .codex/skills/ammo/eval/causal/build_dag.py \
  --events /tmp/ammo_eval_events.json \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_causal_dag.json
```

For current Codex data, `extract_events.py` consumes all paths in `session_data.rollout_files` and understands `response_item/function_call`, the `response_item/custom_tool_call` outer `exec` wrapper, tool outputs, authoritative `event_msg/patch_apply_end` file changes, `event_msg/sub_agent_activity`, and `event_msg/task_complete`. `--session-jsonl` remains available only for legacy single-file input.

### Step 3c: Score Nodes + Detect Anomalies

```bash
.venv/bin/python .codex/skills/ammo/eval/causal/score_nodes.py \
  --dag /tmp/ammo_eval_causal_dag.json \
  --events /tmp/ammo_eval_events.json \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scored_dag.json \
  --anomalies /tmp/ammo_eval_anomalies.json
```

### Step 3d: LLM Deep Dive (Optional)

If `--skip-deep-analysis` is NOT set, spawn a Codex subagent:

```
spawn_agent(task_name="eval_causal_analyzer", fork_turns="none", message="""
Read the causal analyzer rubric at .codex/skills/ammo/eval/causal/agents/causal_analyzer.md,
then analyze anomalies at /tmp/ammo_eval_anomalies.json
with events at /tmp/ammo_eval_events.json
and artifacts at <ARTIFACT_DIR>.
CITATION REQUIREMENT: Every finding must include an "evidence" array with
{file_path, line_start, line_end, claim, quoted_content} for each factual claim.
Read the cited lines and include verbatim excerpts. Findings without citations
will be flagged by the validator.
Write deep_analysis.json to /tmp/ammo_eval_deep_analysis.json.
""")
```

### Step 3e: Generate Post-Mortem

```bash
.venv/bin/python .codex/skills/ammo/eval/causal/generate_postmortem.py \
  --scored-dag /tmp/ammo_eval_scored_dag.json \
  --deep-analysis /tmp/ammo_eval_deep_analysis.json \
  --events /tmp/ammo_eval_events.json \
  --output-dag /tmp/ammo_eval_causal_dag_final.json \
  --output-narrative /tmp/ammo_eval_postmortem.md \
  --output-viz /tmp/ammo_eval_causal_viz.html
```

### Step 3f: Cross-Version Diff (if previous version exists in archive)

```bash
.venv/bin/python .codex/skills/ammo/eval/causal/diff_versions.py \
  --current-dag /tmp/ammo_eval_causal_dag_final.json \
  --previous-dag ~/.codex/ammo-eval/versions/<prev>/runs/<target>/run_1/causal_dag.json \
  --output /tmp/ammo_eval_version_diff.json \
  [--deep --regression-report /tmp/ammo_eval_regression_report.md]
```

### Step 4: Transcript Grading (Optional)

If `--skip-transcript-grading` is NOT set, spawn a Codex subagent:

```
spawn_agent(task_name="eval_transcript_grader", fork_turns="none", message="""
Read the grader rubric at .codex/skills/ammo/eval/agents/transcript_grader.md,
then evaluate the campaign artifacts at <ARTIFACT_DIR>.
CITATION REQUIREMENT: Every deduction (wasted_retries, hallucinated_data,
off_track_reasoning, anti_patterns) must include an "evidence" array with
{file_path, line_start, line_end, claim, quoted_content} for each factual claim.
Read the cited lines and include verbatim excerpts. Findings without citations
will be flagged by the validator.
Write transcript_grading.json to /tmp/ammo_eval_transcript_grading.json.
""")
```

Steps 3d and 4 can run in parallel since they are independent.

After both agents complete, validate their citations:
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/verify_citations.py \
  --input /tmp/ammo_eval_deep_analysis.json \
  --artifact-dir <ARTIFACT_DIR>

.venv/bin/python .codex/skills/ammo/eval/scripts/verify_citations.py \
  --input /tmp/ammo_eval_transcript_grading.json \
  --artifact-dir <ARTIFACT_DIR>
```

If either agent has broken citations (>20% failure rate), send a message asking them to re-source the broken references before proceeding. If the failure rate is low (<20%), flag the broken citations in the final report but continue.

Then re-score with transcript quality:
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/score_campaign.py \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scorecard.json \
  --report /tmp/ammo_eval_report.md \
  --enrich-from /tmp/ammo_eval_transcript_grading.json
```

### Step 5: Archive
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/archive_run.py \
  --scorecard /tmp/ammo_eval_scorecard.json \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --description "<DESCRIPTION>" \
  --transcript-grading /tmp/ammo_eval_transcript_grading.json \
  --changes-snapshot /tmp/ammo_eval_changes_snapshot \
  --causal-dag /tmp/ammo_eval_causal_dag_final.json \
  --postmortem-narrative /tmp/ammo_eval_postmortem.md \
  --causal-viz /tmp/ammo_eval_causal_viz.html
```

The `--changes-snapshot` flag stores the full changes snapshot (worktree patches, artifact copies, manifest) alongside the scored run in the archive. This ensures you can always reconstruct what code was produced, even after cleanup.

The `--causal-dag`, `--postmortem-narrative`, and `--causal-viz` flags are optional. When provided, these causal engine outputs are stored alongside the scorecard in the archive run directory.

### Step 6: Aggregate & Dashboard
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/aggregate_versions.py \
  --repository ~/.codex/ammo-eval

.venv/bin/python .codex/skills/ammo/eval/scripts/generate_dashboard.py \
  --repository ~/.codex/ammo-eval
```

### Step 7: Present Results

1. Read and display `/tmp/ammo_eval_report.md` to the user
2. Tell the user: "Dashboard generated at `~/.codex/ammo-eval/dashboard.html`"
3. If there are previous versions in the repository, highlight the delta vs the most recent previous version
4. **Auto-rank against prior runs**: Query the eval repository (`index.json`) for all prior runs with the same target slug. If any exist, display a comparative ranking:
   ```
   This run's +X% E2E improvement ranks Nth of M runs for {target_slug}.
   Best: +Y% ({version_id}) | Median: +Z% | Worst: +W% ({version_id})
   ```
   Include the full ranked table if 3+ prior runs exist.
5. **Auto-offer deeper comparison**: After presenting results, tell the user: "There are N prior runs for this target. Want me to dig deeper into what drove the differences?" This is an open-ended offer — the orchestrator uses judgment on what to investigate based on user questions. Keep the causal-analyzer and transcript-grader subagent IDs available until follow-up work is done.

### Step 8: Cleanup

After presenting results, ask the user whether to clean up campaign artifacts. Ask directly and wait for the user's selection before deleting files:

**Question**: "The eval run has been archived with a full snapshot. Which artifacts should I clean up?"

**Options**:
1. **Campaign worktrees** — Remove all worktrees listed in the changes snapshot manifest via `git worktree remove <path> --force` and delete their tracking branches
2. **Artifact directory** — Remove the `kernel_opt_artifacts/<campaign_dir>/` directory
3. **Temp files** — Remove `/tmp/ammo_eval_*.json`, `/tmp/ammo_eval_*.md`, and `/tmp/ammo_eval_changes_snapshot/`
4. **Running eval work** — Interrupt a causal-analyzer or transcript-grader only if it is still running and the user explicitly wants that work stopped

If the user selects nothing (or cancels), skip cleanup entirely. For each selected category, execute the cleanup and report what was removed.

**Subagent availability**: Completed or idle eval subagents need no teardown and remain addressable through their retained IDs. Use `followup_task` or `send_message` for later analysis. Use `interrupt_agent` only for still-running work selected in option 4.

Cleanup commands:
```bash
# Worktrees (for each path in manifest.worktrees[*].path):
git worktree remove <path> --force
git branch -D <branch>   # only for local-only branches, skip remote-tracking

# Artifact directory:
rm -rf <ARTIFACT_DIR>

# Temp files:
rm -rf /tmp/ammo_eval_snapshot.json /tmp/ammo_eval_scorecard.json \
       /tmp/ammo_eval_report.md /tmp/ammo_eval_transcript_grading.json \
       /tmp/ammo_eval_changes_snapshot \
       /tmp/ammo_eval_events.json /tmp/ammo_eval_causal_dag.json \
       /tmp/ammo_eval_scored_dag.json /tmp/ammo_eval_anomalies.json \
       /tmp/ammo_eval_deep_analysis.json /tmp/ammo_eval_causal_dag_final.json \
       /tmp/ammo_eval_postmortem.md /tmp/ammo_eval_causal_viz.html \
       /tmp/ammo_eval_version_diff.json /tmp/ammo_eval_regression_report.md

# Eval subagents (only if user selected option 4):
# interrupt_agent for each retained eval subagent that is still running;
# completed or idle eval subagents need no teardown call
```

## Citation Protocol (ALL agents + orchestrator)

Every factual claim in the eval pipeline — whether from the causal-analyzer, transcript-grader, or the orchestrator presenting results — must be backed by a structured citation to a specific file and line range. This prevents the pattern where agents make plausible-sounding claims that cannot be traced to source data.

### Why this matters

In prior eval runs, agents produced findings that read convincingly but required a second pass to source exact evidence. The cost of that second pass was significant (extra agent invocations, context window usage). By requiring citations upfront, agents are forced to verify their claims at analysis time rather than producing unsupported narratives. This is the eval pipeline's own version of the AMMO "evidence-demanding review" principle.

### Citation format (structured JSON)

Every finding, deduction, or claim must include an `evidence` array of citation objects:

```json
{
  "evidence": [
    {
      "file_path": "debate/campaign_round_2/micro_experiments/champion1_pershape_bw.log",
      "line_start": 1,
      "line_end": 12,
      "claim": "All GEMM shapes achieve 80-83% BW in isolation",
      "quoted_content": "gate_up_proj [2560, 18432]: 82.3% BW\ndown_proj [18432, 2560]: 80.5% BW\n..."
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `file_path` | yes | Path relative to artifact dir (or absolute for JSONL/temp files) |
| `line_start` | yes | First line number (1-indexed) |
| `line_end` | yes | Last line number (inclusive) |
| `claim` | yes | The factual statement this citation supports |
| `quoted_content` | yes | Verbatim excerpt from the cited lines (truncated to ~200 chars if needed) |

### Where citations are required

| Agent / Role | What needs citations |
|---|---|
| **causal-analyzer** | Every `root_cause`, `causal_chain` step, and `counterfactual` must cite the source artifact |
| **transcript-grader** | Every `wasted_retries` entry, `hallucinated_data` entry, `off_track_reasoning` entry, and `anti_patterns` finding |
| **orchestrator (Step 7)** | Every claim in the results presentation — speedup numbers, round outcomes, ranking comparisons |

### Verification

After each agent writes its output JSON, the orchestrator runs the citation validator:

```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/verify_citations.py \
  --input /tmp/ammo_eval_deep_analysis.json \
  --artifact-dir <ARTIFACT_DIR> \
  [--strict]  # fail on broken citations instead of just flagging
```

The validator checks:
1. Each cited `file_path` exists
2. The cited `line_start:line_end` range is within the file
3. The `quoted_content` fuzzy-matches the actual content at those lines (>70% similarity)

Findings with broken citations are flagged with `"citation_verified": false` in the output. In `--strict` mode, the orchestrator asks the agent to re-source broken citations before proceeding.

## Scoring Dimensions

| Dimension | Weight | What it Measures |
|-----------|--------|------------------|
| E2E Outcome | 40% | Cumulative speedup (verified-only if unverified lossy ops exist), shipped optimization count |
| Gate Pass Rates | 15% | First-attempt pass rate across all verification gates (unverified lossy shipped ops count as expected failures) |
| Debate Quality | 15% | Proposal grounding, micro-experiment backing, filtering |
| Campaign Efficiency | 15% | Rounds to completion, failure rate, convergence |
| Transcript Quality | 15% | LLM-graded: wasted retries, hallucinated data, off-track reasoning, delegation causality |

When transcript grading is skipped, the remaining 4 dimensions redistribute proportionally (E2E→47%, others→18%/18%/17%).

When Codex subagents are used, the transcript grader additionally scores:
- **Subagent causality bonus** (+0.5 per verified causal chain, max +1.5): delegated research that demonstrably improved proposals
- **Subagent failures** (-1.0 each, max -2.0): wrong delegated data that went uncaught
- **Subagent efficiency** (-0.25 each, max -1.0): redundant delegated work
- **Subagent utilization failures** (-0.5 each, max -1.5): champions ignoring correct delegated data

### Accuracy Verification Adjustment

Sessions that shipped **lossy** optimizations (FP8, INT4, etc.) without Gate 5.1b (GSM8K accuracy) verification are penalized:

- **E2E Outcome**: Uses verified-only cumulative speedup — lossy ops that were never accuracy-gated are excluded entirely. Verified-only speedup is computed per-round when decomposition is available, otherwise approximated as `1 + (raw_speedup - 1) * (verified_ops / total_ops)`.
- **Gate Pass Rates**: Each unverified lossy shipped op counts as an expected gate failure, lowering the pass rate.

Classification uses each track's `classification` field at `campaign.rounds[*].parallel_tracks.tracks[op_id].classification`, falling back to op_id name pattern matching (`fp8`, `int4`, `int8`, `w8a16`, `w4a16`, `quantiz`, `awq`, `gptq`). Lossless ops are always considered verified. The `_ensure_accuracy_verification()` backfill in `score_campaign.py` handles old snapshots that lack the `accuracy_verification` field.

To retroactively apply scoring changes to all archived runs:
```bash
.venv/bin/python .codex/skills/ammo/eval/scripts/rescore_archived.py \
  --repository ~/.codex/ammo-eval        # re-score all
  [--target qwen3-5-4b_l40s_bf16_tp1]     # or filter by target
  [--dry-run]                              # preview without writing
```
Then regenerate aggregates and dashboard (Steps 6a/6b).

**Philosophy**: Speedup is king. Guardrail violations are scored and reported but never automatically fail the eval.

## Repository Structure

```
~/.codex/ammo-eval/
  index.json                     # Master index with leaderboards
  versions/{version_id}/
    meta.json                    # Git commit, description, skill diff
    runs/{target_slug}/
      run_1/scorecard.json       # Scored run
      run_1/report.md
      run_1/changes_snapshot/    # Full code snapshot
        manifest.json            #   Worktree metadata, merge status
        patches/                 #   Git diffs for each worktree
          agent-xxx.patch
          agent-xxx_uncommitted.patch
        campaign_artifacts/      #   Copy of kernel_opt_artifacts dir
        skill.patch              #   Skill diff vs main
      aggregate.json             # Mean/stddev across runs
    summary.json                 # Cross-target summary
  dashboard.html                 # Static HTML dashboard
```

## Reference Targets

The standard evaluation targets (run each after a skill change):

1. **MoE**: Qwen3-30B-A3B on L40S (fp8, tp=1)
2. **Dense**: Llama-3.1-8B on L40S (fp8, tp=1)

Run 2-3 campaigns per target per skill revision (sequential) to handle non-determinism. The aggregation script computes mean/stddev across runs.

## Resume After Interruption

If the eval pipeline is interrupted, check which files exist:
- `/tmp/ammo_eval_snapshot.json` → Step 1 complete, continue from Step 2
- `/tmp/ammo_eval_changes_snapshot/manifest.json` → Step 2 complete, continue from Step 3
- `/tmp/ammo_eval_scorecard.json` → Step 3 complete, continue from Step 4/5
- Check repository for existing runs → Step 5 may already be done

## Dashboard Views

The HTML dashboard has 4 tabs:
1. **Trend**: Line chart of score and speedup across skill versions
2. **Leaderboard**: Best version per target by E2E speedup
3. **Change Correlation**: Delta vs previous version (green/red)
4. **Run Detail**: Drill-down into individual scorecards
