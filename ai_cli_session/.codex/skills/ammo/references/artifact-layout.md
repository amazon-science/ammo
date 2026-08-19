# Artifact Layout (V2) — Path Resolution Reference

This file says where every AMMO campaign artifact lives. Read it before you write any file under `kernel_opt_artifacts/`, and read it again whenever two candidate files could answer the same question. It is the single source of truth for campaign artifact paths: all agent docs, orchestration docs, and scripts point here, and if another document contradicts a path here, this file wins.

## Table of Contents

1. [Root Layout](#root-layout)
2. [Round Directory Tree](#round-directory-tree)
3. [Path Resolution Rules](#path-resolution-rules)
4. [Who Writes What](#who-writes-what)
5. [Authoritative vs Diagnostic Artifacts](#authoritative-vs-diagnostic-artifacts)
6. [Which File Wins](#which-file-wins)
7. [Prohibited Patterns](#prohibited-patterns)
8. [Layout Detection (for scripts)](#layout-detection-for-scripts)
9. [Hook Enforcement](#hook-enforcement)

---

## Root Layout

Only these entries live at the campaign root. Everything else is round-scoped.

```
kernel_opt_artifacts/{target}/
├── state.json              # Campaign state (orchestrator-only writes)
├── state.json.lock         # flock file — every state.json writer serializes on it
├── target.json             # Workload + bench config (mutated on SHIP)
├── REPORT.md               # Terminal deliverable (Stage 7b)
├── report_assets/          # Charts + generators for REPORT.md
│   ├── *.png
│   └── gen_*.py
├── rounds/                 # Round-scoped hierarchy (1-indexed)
│   └── {N}/
│       └── (see Round Directory Tree)
└── blockers/               # Escalation artifacts (cross-round)
    └── {stage}_{date}.md
```

`{target}` = `{model}_{hardware}_{dtype}_tp{tp}` (e.g., `deepseek-v4-flash_B200_fp8_tp1`).

### Root-Level Files

| File | Writer | Notes |
|------|--------|-------|
| `state.json` | Orchestrator only | Schema: `.codex/schemas/state.schema.json` |
| `state.json.lock` | `ammo_state.py` + `reconcile_track_state.py` | Empty flock file, never deleted. Holds no campaign data |
| `target.json` | `new_target.py` + orchestrator (env promotion on SHIP) | Never written by sub-agents |
| `REPORT.md` | `ammo-report` (terminal only) | Authoritative final deliverable |
| `report_assets/` | `ammo-report` | 5 required PNGs + gen scripts |
| `blockers/{stage}_{date}.md` | Orchestrator on escalation | Cross-round (NOT round-scoped) |

> **Note:** `state.json` plus the path conventions in this file are the only sources of artifact metadata. Per-artifact `.metrics.json` sidecars are no longer used. The frontend lists files with `GET /api/campaigns/{id}/tree` and reads metrics from `state.json` (for example `rounds[N-1].bottleneck_mining` and `tracks[op_id].gate_5_2_metrics`).

---

## Round Directory Tree

Every round `N` (1-indexed) contains the full lifecycle for that optimization round. Read the comments in the tree — each one is a rule about what the file is for and whether you may trust its numbers.

```
rounds/{N}/
├── constraints.md                       # Baseline constraints for this round
│
├── profiling/
│   ├── nsys/                            # Stage 1 nsys node traces
│   │   ├── baseline_bs{BS}.nsys-rep
│   │   ├── opt/{op_id}/                 # Stage 5 activation traces
│   │   └── integration/                 # Stage 6 combined activation traces
│   └── ncu/                             # Targeted hardware-counter data
│   │   ├── sanity.csv                   # Top-3 kernel metrics (OL=8 driver)
│   │   └── sanity_results.md            # Researcher narrative
│
├── sweeps/
│   ├── baseline/                        # Stage 1 E2E baseline (AUTHORITATIVE)
│   │   ├── json/                        # golden_refs.json, baseline_bs{BS}.json
│   │   ├── logs/                        # Per-bucket + supervisor logs
│   │   ├── status/                      # Heartbeat files
│   │   └── e2e_latency_results.{json,md}
│   ├── opt/{op_id}/                     # Stage 5 per-track opt sweep (AUTHORITATIVE)
│   │   ├── json/                        # correctness_verdict.json, opt_outputs.json, opt_bs{BS}.json
│   │   ├── logs/
│   │   ├── status/
│   │   └── e2e_latency_results.{json,md}
│   ├── opt_correctness/{op_id}/         # Gate 5.1b clean full-model correctness
│   │   └── (same sub-structure)
│   ├── opt_profiling/{op_id}/           # Gate 5.3a production activation capture
│   │   └── (same sub-structure; latency is contaminated)
│   ├── integration/                     # Stage 6 result; promoted baseline on SHIP
│   │   └── (same sub-structure)
│   ├── integration_profiling/           # Stage 6 combined activation capture
│   │   └── (same sub-structure; latency is contaminated)
│   ├── golden_capture/                  # Post-SHIP golden-refs for next round
│   │   └── json/golden_refs.json
│   └── post_ship_profiling/             # Optional post-SHIP re-attribution capture
│       └── (same sub-structure; latency is contaminated)
│
├── mining/
│   ├── mine_config.json                 # Researcher JUDGMENT input to mine_trace.py
│   ├── mined.json                       # mine_trace.py output (schema mine_trace/1)
│   ├── tables.md                        # DERIVED — mine_trace.py; pasted verbatim below
│   └── bottleneck_analysis.md
│
├── debate/
│   ├── eligibility_verdicts.md          # Lead-owned — CUT-4 four-gate verdict + gated_revision (write-once-then-append)
│   ├── proposals/                       # {champion_id}_proposal.md
│   ├── round_{D}/                       # Debate rounds (D=1,2,...; NOT campaign rounds)
│   │   └── {op_id}_{argument|critique_{target}|rebuttal}.md
│   ├── micro_experiments/               # Champion feasibility scripts + logs
│   └── summary.md                       # DERIVED — lead-authored from state.json; state.json wins
│
├── tracks/
│   └── {op_id}/
│       ├── validation_results.md        # Champion's final verdict (AUTHORITATIVE)
│       ├── evidence.json                # Structured primary-evidence index
│       ├── validation_summary.json      # Generated machine summary
│       ├── diff.patch                   # Track diff vs merge base (refreshed per gate commit)
│       ├── runtime_pkg.patch            # Only when a materialized runtime was edited
│       ├── validator_tests/             # Impl-champion kernel correctness & speedup artifacts
│       │   ├── test_correctness.py
│       │   ├── bench_gate_5_2.py
│       │   ├── gate_5_1a_results.json
│       │   └── gate_5_2_results.json
│       ├── monitor_audits/              # Transcript monitor (Stages 4-5)
│       │   ├── {monitor_id}_observations.md
│       │   ├── transcript_offsets.json
│       │   └── {monitor_id}_summary.json
│       └── _scratch/                    # Non-authoritative iteration artifacts
│           └── *.md, *.py, *.log
│
├── audits/
│   ├── stage_1.md                       # T_AUDIT_S1 verdict
│   ├── stage_2.md                       # T_AUDIT_S2 verdict
│   ├── stage_45_partial_{op_id}.md      # Optional early per-track evidence (pre-T_AUDIT_S45); NOT a verdict
│   ├── stage_45.md                      # T_AUDIT_S45 verdict
│   ├── stage_67.md                      # T_AUDIT_S67 verdict
│   └── stage_{stage}_cycle_{C}.md       # Re-audit on BLOCKED (C=2,3)
│
├── validation_gate_report.json          # Cohort mechanical gate result
│
└── _archive/                            # Superseded sweep runs (auto-managed)
    └── {slot}_{timestamp}/              # e.g., baseline_2026-05-05T181212Z/
```

---

## Path Resolution Rules

These rules fix the output path for each writer, so nobody has to invent one.

### Sweep Script (`run_vllm_bench_latency_sweep.py`)

Name the sweep output with `--round` and `--slot`. Never name the directory yourself.

```bash
.venv/bin/python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir {artifact_dir} \
    --round {N} --slot {SLOT} \
    [--labels baseline|opt] [other flags...]
```

| `--slot` value | Resolves to | Used by |
|----------------|-------------|---------|
| `baseline` | `rounds/{N}/sweeps/baseline/` | Stage 1 clean E2E (profiling flags forbidden) |
| `profiling` | `rounds/{N}/sweeps/profiling/` | Stage 1 profiling (latency here is contaminated) |
| `profiling_prefill` | `rounds/{N}/sweeps/profiling_prefill/` | Stage 1 prefill-inclusive capture at reduced `--nsys-output-len`; latency contaminated, and the shortened OSL means any decode-shape claim must be cross-checked against a production-depth capture |
| `opt/{op_id}` | `rounds/{N}/sweeps/opt/{op_id}/` | Stage 5 per-track opt |
| `opt_correctness/{op_id}` | `rounds/{N}/sweeps/opt_correctness/{op_id}/` | Gate 5.1b clean full-model correctness |
| `opt_profiling/{op_id}` | `rounds/{N}/sweeps/opt_profiling/{op_id}/` | Gate 5.3a activation evidence; never timing authority |
| `integration` | `rounds/{N}/sweeps/integration/` | Stage 6 combined sweep or copied single-pass result; promoted baseline for the next material-SHIP round |
| `integration_profiling` | `rounds/{N}/sweeps/integration_profiling/` | Stage 6 combined activation proof; never timing authority |
| `golden_capture` | `rounds/{N}/sweeps/golden_capture/` | Post-SHIP golden-refs |
| `post_ship_profiling` | `rounds/{N}/sweeps/post_ship_profiling/` | Optional post-SHIP re-attribution; traces route to `profiling/nsys/post_ship/`; never timing authority |

**Resolution logic:**
1. `--round N --slot SLOT` → `out_root = {artifact_dir}/rounds/{N}/sweeps/{SLOT}/`
2. `--round N` only → fail (slot required)
3. Neither → read `state.json.campaign.current_round` + require `--slot`
4. `--out-name`: **removed**. Hard error with guidance pointing to `--round`/`--slot`.

**Archive behavior:** When the target `out_root` already exists and is non-empty, the script moves the existing contents to `rounds/{N}/_archive/{slot}_{timestamp}/` (`{timestamp}` is ISO-8601 UTC). Active slots NEVER carry timestamps in their names.

### Per-Bucket nsys Traces (sweep `--nsys-profile`)

Traces from `--nsys-profile` go to **`rounds/{N}/profiling/nsys/`** — NOT inside the sweep output dir. The sweep script extracts `--round` and writes the traces to that sibling profiling dir.

Legacy campaigns may still contain `rounds/{N}/profiling/probe/` or `rounds/{N}/profiling/torch_profile/`. Readers may keep compatibility for those paths, but new campaigns do not scaffold or recommend them.

### Monitor Logs (Impl-Stage Only)

An impl-stage monitor MUST write to the **monitored entity's `monitor_audits/` subdirectory**.

| Monitor target | Output path |
|---------------|-------------|
| Impl-champion `{op_id}` | `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md` |

Monitors receive `round_number` and `output_dir` in their dispatch prompt. They MUST NOT write to the campaign root. Debate champions do not have monitors.

### Champion Scratch Files

Every intermediate artifact — drafts, debug scripts, iteration logs — MUST go in `rounds/{N}/tracks/{op_id}/_scratch/`. The ONLY authoritative file at the track root is `validation_results.md`.

### Projection Accuracy (`check_projection_accuracy.py`)

This script appends a `## Projection Accuracy` section to `rounds/{N}/tracks/{op_id}/validation_results.md` — NOT a campaign-root file. Call it with `--round N --track-id {op_id}`.

---

## Who Writes What

Each table below lists what one agent writes, and when.

### Orchestrator
| Writes to | When |
|-----------|------|
| `state.json` | Every stage transition, gate result, track update |
| `target.json` | SHIP env promotion |
| `rounds/{N}/debate/summary.md` (lead-authored) | After debate winner selection |
| `rounds/{N}/debate/eligibility_verdicts.md` | At op_id minting — CUT-4 four-gate verdict + `gated_revision` per surviving op_id; appended on any critique-triggered Gate-1/Gate-2 re-derivation (write-once-then-append; lead-owned) |
| `blockers/{stage}_{date}.md` | On escalation |

### ammo-researcher
| Writes to | When |
|-----------|------|
| `rounds/{N}/sweeps/baseline/*` | T=1 (via sweep script) |
| `rounds/{N}/profiling/nsys/*.nsys-rep` | T=1 (via sweep `--nsys-profile`) |
| `rounds/{N}/constraints.md` | T=2 |
| `rounds/{N}/mining/bottleneck_analysis.md` | T=4 |
| `rounds/{N}/profiling/ncu/sanity.csv` | T=6 (targeted, only for physical-ceiling claims) |

### ammo-champion (debate)
| Writes to | When |
|-----------|------|
| `rounds/{N}/debate/proposals/{champion_id}_proposal.md` | Phase 0 |
| `rounds/{N}/debate/round_{D}/{op_id}_argument.md` | Phase A (debate sub-rounds 2+ only — sub-round 1 has no Phase A; the Phase 0 proposal is its argument-of-record) |
| `rounds/{N}/debate/round_{D}/{op_id}_critique_{target}.md` | Phase B |
| `rounds/{N}/debate/round_{D}/{op_id}_rebuttal.md` | Phase C |
| `rounds/{N}/debate/micro_experiments/{champion_id}_*.py` | Phase 0 (optional) |

### ammo-implementer
| Writes to | When |
|-----------|------|
| `rounds/{N}/tracks/{op_id}/validation_results.md` | After all gates |
| `rounds/{N}/tracks/{op_id}/validator_tests/*` | Kernel correctness & speedup tests |
| `rounds/{N}/tracks/{op_id}/_scratch/*` | During iteration |
| `rounds/{N}/sweeps/opt_profiling/{op_id}/*` | Via sweep script (Gate 5.3a) |
| `rounds/{N}/sweeps/opt_correctness/{op_id}/*` | Via sweep script (Gate 5.1b) |
| `rounds/{N}/sweeps/opt/{op_id}/*` | Via sweep script (clean Gate 5.3b timing) |

### ammo-transcript-monitor (impl-stage only)
| Writes to | When |
|-----------|------|
| `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md` | Impl monitoring |
| `rounds/{N}/tracks/{op_id}/monitor_audits/transcript_offsets.json` | Impl monitor target/offset binding |
| `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_summary.json` | Terminal monitor coverage record |

### ammo-auditor
| Writes to | When |
|-----------|------|
| `rounds/{N}/audits/stage_1.md` | T_AUDIT_S1 |
| `rounds/{N}/audits/stage_2.md` | T_AUDIT_S2 |
| `rounds/{N}/audits/stage_45.md` | T_AUDIT_S45 |
| `rounds/{N}/audits/stage_67.md` | T_AUDIT_S67 |
| `rounds/{N}/audits/stage_45_partial_{op_id}.md` | Optional — early per-track evidence, written by an `ammo-delegate` dispatched directly by the orchestrator (not by `ammo-auditor` itself) as soon as that track reaches terminal status, ahead of the full T_AUDIT_S45 gate. Consumed (not produced) by the real auditor at T_AUDIT_S45. See `orchestration/audit-protocol.md` § Optional Early S45 Evidence. |

### ammo-report
| Writes to | When |
|-----------|------|
| `REPORT.md` | Terminal only |
| `report_assets/*.png` | Terminal only |
| `report_assets/gen_*.py` | Terminal only |

---

## Authoritative vs Diagnostic Artifacts

An authoritative file is an input to a later stage, so treat it as a contract. A diagnostic file only informs a reader.

### Authoritative — downstream stages read these

| Path | Consumer | Gate |
|------|----------|------|
| `state.json` | All agents, all stages | — |
| `rounds/{N}/sweeps/baseline/e2e_latency_results.json` | Stage-1 mining and later invalidated-mining reuse until a material SHIP supersedes it | Stage 2 |
| `rounds/{N}/sweeps/baseline/json/golden_refs.json` | Stage 5.1b correctness | Gate 5.1b |
| `rounds/{N}/sweeps/opt_correctness/{op_id}/json/correctness_verdict.json` | Orchestrator, auditor | Gate 5.1b |
| `rounds/{N}/sweeps/opt_profiling/{op_id}/` | Orchestrator, auditor | Gate 5.3a activation only |
| `rounds/{N}/mining/mine_config.json` | `mine_trace.py` (researcher judgment input) | Stage 2 |
| `rounds/{N}/mining/mined.json` | Auditor, `bottleneck_analysis.md` tables | Stage 2 |
| `rounds/{N}/mining/tables.md` | Researcher, pasted verbatim (DERIVED) | Stage 2 |
| `rounds/{N}/mining/bottleneck_analysis.md` | Debate champions, routing | Stage 2 |
| `rounds/{N}/debate/summary.md` | Report writer (DERIVED) | — |
| `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json` | Orchestrator, integration | Gate 5.3b |
| `rounds/{N}/tracks/{op_id}/validation_results.md` | Orchestrator, auditor, report writer | T9 |
| `rounds/{N}/tracks/{op_id}/evidence.json` | `verify_validation_gates.py`, `reconcile_track_state.py` | Gate 5.x |
| `rounds/{N}/tracks/{op_id}/validation_summary.json` | `reconcile_track_state.py` (DERIVED) | T9 |
| `rounds/{N}/tracks/{op_id}/diff.patch` | Review, resume, integration, PR extraction | Gate 5.x |
| `rounds/{N}/tracks/{op_id}/runtime_pkg.patch` | Integration (only when a materialized runtime was edited) | Gate 5.x |
| `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_1a_results.json` | Orchestrator state merge | Gate 5.1a |
| `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_2_results.json` | Orchestrator state merge | Gate 5.2 |
| `rounds/{N}/sweeps/integration/e2e_latency_results.json` | Campaign eval, cumulative speedup, and round N+1 mining after a material SHIP | Stage 6 / next-round Stage 2 |
| `rounds/{N}/sweeps/golden_capture/json/golden_refs.json` | Next round's Stage 5.1b | Post-SHIP |
| `rounds/{N}/constraints.md` | Debate champions (current round) | — |
| `rounds/{N}/audits/stage_{1\|2\|45\|67}.md` | State transitions and downstream stages | T_AUDIT |
| `rounds/{N}/audits/stage_{stage}_cycle_{C}.md` | Re-audit verdict after BLOCKED (C=2,3) | T_AUDIT |
| `rounds/{N}/validation_gate_report.json` | Orchestrator, auditor (cohort mechanical result) | Gate 5.x |

### Diagnostic — informational, not pipeline-consumed

The Confusion Risk column says how easily a reader mistakes the file for an authoritative one.

| Path | Purpose | Confusion Risk |
|------|---------|----------------|
| `rounds/{N}/profiling/ncu/sanity.csv` | HW counter sanity | MEDIUM — latency column is NOT baseline |
| `rounds/{N}/debate/micro_experiments/*` | Feasibility evidence | LOW |
| `rounds/{N}/tracks/{op_id}/_scratch/*` | Champion iteration | MEDIUM — drafts look like finals |
| `rounds/{N}/tracks/{op_id}/monitor_audits/*` | DA enforcement (impl-stage) | LOW |
| `rounds/{N}/audits/stage_45_partial_{op_id}.md` | Early per-track evidence | MEDIUM — not a verdict |
| `rounds/{N}/sweeps/post_ship_profiling/` | Post-SHIP re-attribution | HIGH — latency is contaminated |
| `rounds/{N}/_archive/{slot}_{timestamp}/` | Superseded runs | LOW (quarantined) |

---

## Which File Wins

Use these answers when more than one candidate file exists.

1. **"Which is the baseline E2E?"** → In Round 1, `rounds/1/sweeps/baseline/e2e_latency_results.json`. After a material SHIP in round N, `rounds/{N}/sweeps/integration/e2e_latency_results.json` is the promoted baseline for round N+1 mining, including when it was populated by the single-pass short-circuit. Do not require a new round-N+1 baseline file. NEVER use profiler timing or an `_archive/` dir.

2. **"Which profiling evidence feeds Stage 2?"** → In Round 1, `rounds/1/profiling/nsys/baseline_bs{BS}.nsys-rep` or the matching `baseline_profile*.nsys-rep` companion path. After a material SHIP, use those still-applicable attribution traces together with the shipped track's activation evidence and any Stage-6 integration activation capture. T16 remains eliminated: there is no mandatory fresh post-SHIP trace or provenance sidecar. Legacy probe directories are compatibility-only and are not Stage 2 ranking inputs.

3. **"Which is the authoritative track verdict?"** → `rounds/{N}/tracks/{op_id}/validation_results.md`. NEVER anything in `_scratch/`.

4. **"Which opt sweep is authoritative?"** → `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json`. Only one per track per round (superseded runs archived).

5. **"What are the golden refs for correctness?"** → For the CURRENT round's opt sweeps: `rounds/{N}/sweeps/baseline/json/golden_refs.json`. For round N+1 after SHIP: `rounds/{N}/sweeps/golden_capture/json/golden_refs.json`.

6. **"Is this file authoritative or derived?"** → `debate/summary.md` is DERIVED (lead-authored from `state.json`; champions never write it). If it disagrees with `state.json.campaign.rounds[N-1].debate.selected_candidates`, state.json wins and the lead corrects the summary.

---

## Prohibited Patterns

Never create any of these. Each item names the replacement or the reason.

1. **Writing to campaign root** (except `state.json`, `state.json.lock`, `target.json`, `REPORT.md`, `report_assets/`)
2. **Timestamped directory names in active slots** (only `_archive/` may carry timestamps)
3. **Ad-hoc `--out-name`** on sweep script (use `--round` + `--slot`)
4. **`monitor_log_*` at campaign root** — must be under `monitor_audits/`
5. **`validation_results_DRAFT.md`** or similar at track root — use `_scratch/`
6. **Nested `kernel_opt_artifacts/` paths** — monitors must receive absolute `output_dir`
7. **`e2e_latency_opt*`, `e2e_latency_combined/`** at campaign root — use semantic slots
8. **`investigation/`, `runs/`, `monitoring/`** directories at root — removed from scaffold
9. **Writing nsys traces into sweep output** — traces go to `profiling/nsys/`, not `sweeps/*/nsys/`
10. **`debate/campaign_round_{N}/` nesting** — replaced by top-level `rounds/{N}/debate/`
11. **`audits/stage_{N}_round_{M}.md` at campaign root** — replaced by `rounds/{M}/audits/stage_{N}.md`
12. **`bottleneck_analysis.md` at campaign root** — replaced by `rounds/{N}/mining/bottleneck_analysis.md`
13. **`constraints.md` at campaign root** — replaced by `rounds/{N}/constraints.md`

---

## Layout Detection (for scripts)

A script that must serve both v1 (legacy flat) and v2 (round-scoped) layouts uses one filesystem check:

```python
def _is_v2_layout(artifact_dir: Path) -> bool:
    return (artifact_dir / "rounds").is_dir()
```

This check is independent of `state.json["schema_version"]`. The schema version remains `"4.1"`; layout is detected by filesystem presence of `rounds/`.

When `_is_v2_layout(artifact_dir)` is `True`, scripts MUST emit v2 paths. When `False`, scripts MAY fall back to legacy paths (typically only relevant for old campaigns started before this spec).

---

## Hook Enforcement

The PostToolUse guard in `.codex/hooks/post_tool_use_guard.py` emits a non-blocking warning when files are created outside the allowed patterns. The tree and path-resolution rules above remain the documentation authority.

Op-id pattern: `[A-Za-z0-9_-]+` (supports `OP-001`, `op007`, etc.).
