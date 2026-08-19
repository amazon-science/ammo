# AMMO Codex Port

This README describes the actual Codex AMMO runtime under `ai_cli_session/.codex`.
It intentionally covers only the campaign workflow and hook system. Claude AMMO
is the shared-policy source of truth; the Codex skill, agents, hooks, schemas, and
scripts in this directory add only Codex CLI/runtime adapters.

Canonical entrypoint: `.codex/skills/ammo/SKILL.md`.
Runtime agents: `.codex/agents/*.toml`.
Prompt/reference bodies: `.codex/skills/ammo/agents/*.md`.
Hooks: `.codex/hooks.json` and `.codex/hooks/*.py`.
Project Codex config: `.codex/config.toml`.

## 2. Campaign Workflow

AMMO is a round-based vLLM kernel optimization campaign. The lead Codex session
scaffolds, delegates, reconciles state, and gates transitions; it must not write
kernel implementation code itself. Mutable campaign state is stored in
`kernel_opt_artifacts/.../state.json`, and mutable round artifacts live under
`rounds/{N}/`.

```text
Stage 0: Scaffold
  lead -> new_target.py
  output: target.json, state.json

Round 1 (and only an explicitly required later recapture):
  Stage 1: Baseline capture
    lead spawns ammo-researcher
    researcher runs two separate sweeps:
      1. clean official baseline: --round {N} --slot baseline
      2. profiling capture:       --round {N} --slot profiling --nsys-profile
    output: rounds/{N}/constraints.md
            rounds/{N}/sweeps/baseline/e2e_latency_results.json
            rounds/{N}/profiling/...
    gate: T_AUDIT_S1

Round N Stage 2: Bottleneck mining
    lead spawns ammo-researcher
    researcher mines the exact resolved incumbent traces
    output: rounds/{N}/mining/bottleneck_analysis.md
    gate: T_AUDIT_S2 for schema v4.1+

  Stage 3: Adversarial debate
    lead spawns 2-4 ammo-champion agents over the full profile
    each champion develops 2-3 ranked, micro-experiment-backed candidates
    Phase A: independent proposals
    Phase B: cross-critique
    Phase C: rebuttal + explicit open items
    lead applies EV ranking and portfolio concentration at selection time
    output: rounds/{N}/debate/proposals/
            rounds/{N}/debate/round_{K}/
            campaign.rounds[N-1].debate.selected_candidates
            rounds/{N}/debate/summary.md rendered from state
    no T_AUDIT after Stage 3; the debate structure is the adversarial check

  Stages 4-5: Parallel implementation tracks
    lead creates one .codex/worktrees/ammo-track-{op_id}/ per selected candidate
    lead spawns one ammo-implementer per track
    lead also spawns one ammo-transcript-monitor per implementer
    implementer writes code in its worktree
    implementer self-runs Gates 5.1a and 5.2 inline (own correctness test + CUDA-graph bench)
    implementer runs separate sibling sweeps for Gates 5.1b, 5.3a, and 5.3b
    output: rounds/{N}/tracks/{op_id}/evidence.json
            rounds/{N}/tracks/{op_id}/validation_results.md
            rounds/{N}/tracks/{op_id}/validator_tests/
            rounds/{N}/sweeps/opt_correctness/{op_id}/
            rounds/{N}/sweeps/opt_profiling/{op_id}/
            rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json
    gate: Claude-defined inline/E2E validation gates + T_AUDIT_S45

  Stage 6: Integration and incumbent protection
    exactly one passing track:
      copy Stage 5 sweep results into rounds/{N}/sweeps/integration/
      set integration.status to single_pass or gated_pass
      do not run a redundant combined sweep
    multiple passing tracks:
      cherry-pick compatible tracks
      run integration sweep with --fresh-cache
      spawn ammo-resolver for qualifying GATED_PASS merge conflicts
    no passing tracks:
      set integration.status to exhausted
    output: integration fields under campaign.rounds[N-1].integration
            rounds/{N}/sweeps/integration/e2e_latency_results.json when applicable
    gate: pre-SHIP mechanical checks + T_AUDIT_S67

  Stage 7: Campaign evaluation
    no user prompt
    mining source, stop condition, and its single input are owned by
      SKILL.md Stage 7 and references/validation-defaults.md
      (Minimum E2E Improvement Threshold, Invalid Reasons to Stop)

  Stage 7b: Report
    lead spawns ammo-report
    report must pass a different-agent adversarial review
    output: REPORT.md, optional maintenance_decision.md and artifact_bundle.json,
            and report assets
```

### State And Artifacts

The active round is always `campaign.rounds[campaign.current_round - 1]`.
`campaign.current_stage` uses these canonical values:

```text
1_baseline -> 2_bottleneck_mining -> 3_debate -> 4_5_parallel_tracks -> 6_integration -> 7_campaign_eval -> 7b_report
```

Terminal campaign outcomes belong in `campaign.status`, not `current_stage`.
`new_target.py` seeds round 1 and scaffolds
`campaign.config.min_e2e_improvement_pct`; its default value is documented once
in `references/validation-defaults.md` (Minimum E2E Improvement Threshold).

Canonical V2 artifact paths include:

```text
kernel_opt_artifacts/{target}/
  state.json
  target.json
  REPORT.md
  report_assets/
  blockers/
  rounds/{N}/
    constraints.md
    profiling/{probe,nsys,ncu,torch_profile}/
    profiling/nsys/post_ship/
    sweeps/baseline/
    sweeps/opt_correctness/{op_id}/
    sweeps/opt_profiling/{op_id}/
    sweeps/opt/{op_id}/
    sweeps/integration/
    sweeps/golden_capture/
    mining/bottleneck_analysis.md
    debate/
      proposals/
      round_{K}/
      monitor_audits/
      summary.md
    tracks/{op_id}/
      evidence.json
      validation_results.md
      validator_tests/
      monitor_audits/
    audits/
```

Markdown is a rendered view where structured JSON exists. `state.json` and
round-scoped gate JSONs win over narrative summaries.

### Agent Roles

Codex multi-agent V2 has no persistent Claude-style round-team object. The lead
records canonical absolute task paths in notes or artifacts and uses current
`spawn_agent`, `send_message`,
`followup_task`, `wait_agent`, and `interrupt_agent` operations. Completed or
idle agents need no close handshake; interrupt only work that is still running
and must stop.

The spawnable contract surface is `.codex/agents/*.toml`. Those TOMLs are
runtime bootstraps: they load `.codex/skills/ammo/SKILL.md` and the matching
`.codex/skills/ammo/agents/*.md` file, and they must not add independent AMMO
policy. The behavior surface is the skill plus agent Markdown.

| Agent | Actual Codex role |
| --- | --- |
| `ammo-researcher` | Runs Stage 1 baseline/profiling and Stage 2 mining. It reports measured facts, not proposals or feasibility scores. |
| `ammo-champion` | Reads the full Stage 2 profile, develops 2-3 ranked candidates with micro-experiment evidence, and participates in cross-critique/rebuttal. There is no claim phase or preassigned component territory. |
| `ammo-implementer` | Implements one selected candidate in an explicit `.codex/worktrees/ammo-track-{op_id}/` worktree. It owns code changes and E2E evidence, and self-runs Gates 5.1a + 5.2 inline (writes its own correctness test + CUDA-graph speedup bench into the track's `validator_tests/`). |
| `ammo-transcript-monitor` | Watches one Stage 4-5 implementer transcript and writes interventions to `monitor_interventions.jsonl`; hooks deliver those interventions back to the campaign session. |
| `ammo-auditor` | Fresh-spawned at T_AUDIT gates. It writes Phase 1 first; the PostToolUse hook then injects Phase 2 checklist instructions. |
| `ammo-delegate` | Bounded helper for evidence gathering, used by champions, implementers, and auditors. |
| `ammo-investigator` | Targeted decision-support/root-cause investigator for ambiguous forks and stuck states. |
| `ammo-resolver` | Resolves Stage 6 integration conflicts, especially GATED_PASS dispatch conflicts, then the lead requests validation/DA review. |
| `ammo-report` | Writes the final external-facing report; a different agent reviews its claims against source artifacts. |

Legacy Claude names remain spawnable because prompts and dependencies may still
refer to exact historical names:

| Legacy name | Codex behavior |
| --- | --- |
| `ammo-impl-champion` | Native TOML alias that reads `agents/ammo-implementer.md`. |
| `ammo-report-writer` | Native TOML alias that reads `agents/ammo-report.md`. |

> **Note — kernel-level validator removed.** The `ammo-validator` subagent (and its
> legacy `ammo-impl-validator` TOML alias) ran Gates 5.1a + 5.2 as an independent,
> adversarial author of the correctness test and speedup bench. It has been removed:
> `ammo-implementer` now self-runs both gates inline. This trades away kernel-level
> adversarial independence for the per-track blocking wait it cost; the surviving
> independent-of-the-kernel-author signals are the E2E sweep's Gate 5.1b (GSM8K) and
> Gate 5.3b (latency), the transcript monitor, and the Stage-4/5 audit.

The transcript monitor is mandatory for every Stage 4-5 `ammo-implementer` or
`ammo-impl-champion`. Debate champions do not get transcript monitors: Phase B
cross-critique, Phase C rebuttal, and explicit open items provide that stage's
quality control. A monitor never replaces the Claude-defined inline/E2E gates
or the Stage-4/5 audit.

The project template enables multi-agent V2 and hook support in `.codex/config.toml`.
V2 is selected when a fresh root session is created; resuming a V1 root keeps
that root on V1. Custom AMMO roles always spawn with `fork_turns="none"`, and
their unique round-qualified lowercase task-path slug is distinct from raw
`OP-001` artifact identity.
The session config selects the model/provider; individual AMMO TOMLs deliberately
omit `model` so they inherit that selection. Role TOMLs only set reasoning effort
where the role needs an override.

### Audit Gates

Audit gates are part of the workflow, not optional review. The lead spawns a
fresh `ammo-auditor` for each gate with only stage and round context. The lead
does not include `references/audit-invariants.md` in the spawn prompt.

| Gate | Trigger | State field on PASS |
| --- | --- | --- |
| `T_AUDIT_S1` | Stage 1 baseline capture complete | `rounds[N-1].audit.stage_1.passed_at` |
| `T_AUDIT_S2` | Stage 2 bottleneck mining complete for v4.1+ campaigns | `rounds[N-1].audit.stage_2.passed_at` |
| `T_AUDIT_S45` | All current-round implementation tracks are terminal | `rounds[N-1].audit.stage_45.passed_at` |
| `T_AUDIT_S67` | SHIP after merge/env promotion/golden refs, or EXHAUSTED after integration status is set | `rounds[N-1].audit.stage_67.passed_at` |

The auditor's Phase 2 checklist is hook-delivered after Phase 1. If the hook has
fired, the Stop hook blocks auditor completion until the verdict file contains
a `## Phase 2` section.

## 5. The Hook System

Codex AMMO hooks are Python runtime guards. `.codex/hooks.json` wires:

| Event | Script | Matchers |
| --- | --- | --- |
| `SessionStart` | `hooks/session_start.py` | `startup|resume` |
| `PreCompact` | `hooks/pre_compact.py` | `manual|auto` |
| `PreToolUse` | `hooks/pre_tool_use_guard.py` | `Bash`, edits, and MCP calls (native `spawn_agent` is not a supported PreToolUse target) |
| `PostToolUse` | `hooks/post_tool_use_guard.py` | `Bash`, edits, and MCP calls |
| `SubagentStart` | `hooks/post_tool_use_guard.py` | child lifecycle start, cwd/monitor-order warning, and implementer-monitor ledger binding |
| `SubagentStop` | `hooks/post_tool_use_guard.py` | child lifecycle stop |
| `Stop` | `hooks/stop_gate_guard.py` | session stop attempts |

The hooks search upward from the current working directory and also check the
`ai_cli_session/.codex/hooks/` template path. That lets the same hooks run from
the session root, nested repository paths, or copied session worktrees.

### 5.1 PreToolUse: Intercept, Block, Or Warn Before A Tool Runs

`pre_tool_use_guard.py` is the front-door guard. It allows static inspection
commands quickly, then enforces AMMO-specific safety when a command or active
artifact directory indicates AMMO context.

Hard blocks:

- Child/subagent edits to orchestrator-owned AMMO docs/contracts under
  `.codex/skills/ammo/references/`, `.codex/skills/ammo/orchestration/`,
  `.codex/skills/ammo/agents/`, and `.codex/agents/`.
- `VLLM_OP*` feature flags in `envs.py` defaulting on. AMMO-introduced flags
  must default off and be enabled explicitly through the benchmark environment.
- Python, pytest, pip, and uv commands inside `.codex/worktrees/...` that do not
  use that worktree's `.venv`.
- Use of `.claude/skills/ammo` or `.claude/worktrees` in Codex AMMO commands.
- Profiled sweep output being used as official optimized/integration timing.
- Pending monitor interventions with `ack_required` before the next intercepted
  non-inspection tool call.

Native `spawn_agent` calls are not PreToolUse/PostToolUse targets in Codex.
The guaranteed `SubagentStart` hook checks inherited cwd and owed-monitor
ordering, records the lifecycle obligation, and injects blocking context into
an out-of-order child. That child must return without campaign work until the
parent corrects the issue and explicitly resumes it.

Warnings/additional context:

- Raw `vllm bench latency` is not official AMMO evidence; use
  `run_vllm_bench_latency_sweep.py`.
- Commands that disable production parity (`TORCH_COMPILE_DISABLE=1`,
  `VLLM_TORCH_COMPILE_LEVEL=0/1`, `--enforce-eager`, or
  `--disable-cuda-graph`) are flagged unless explicitly allowed.
- GPU-heavy commands without `gpu_reservation.py` or explicit no-GPU intent are
  blocked once per session with reservation guidance.

### 5.2 PostToolUse and Subagent Lifecycle: Inject Context, Validate State, Release Reservations

`post_tool_use_guard.py` reviews the result after intercepted tools. It is both
an injector and a validator.

It injects:

- A required `SubagentStart` monitor reminder when `ammo-implementer` or legacy
  `ammo-impl-champion` starts.
- Pending `WARNING`, `CRITICAL`, and `HARD_GATE` transcript-monitor
  interventions after AMMO writes or non-static AMMO commands.
- Auditor Phase 2 instructions after an `ammo-auditor` writes a
  `rounds/{N}/audits/stage_*.md` verdict file that lacks `## Phase 2`.
- A one-line confirmation after a successful `ammo-auditor` spawn, when the
  hook stamps `audit.{stage}.started_at` and `.cycle` from the dispatch
  message. The stamp runs on the spawn tool call only; `SubagentStart` gets the
  child's identity, not the dispatch message. It is fail-open: a spawn the hook
  cannot parse gets no stamp and no message.
- Artifact-layout warnings for writes outside the canonical V2 tree.
- Stage-specific guidance after the Stage 2 gate, after sweep runs, and during
  Stage 6 single-passer vs multi-passer integration.

It validates and can block:

- `state.json` against `.codex/schemas/state.schema.json`.
- Audit-gated stage transitions, including T_AUDIT_S1 before Stage 2,
  T_AUDIT_S2 before Stage 3 for v4.1+, T_AUDIT_S45 before Stage 6, and
  T_AUDIT_S67 before Stage 7/new-round work.
- Stage 6 entry while any current-round track is non-terminal.

It also releases GPU reservations for observed `gpu_reservation.py reserve`
commands and uses native subagent lifecycle identity for child cleanup. It does
not age-reap child reservations because reservation age is not a liveness
signal for long-running benchmark agents.

### 5.3 Stop: Block Premature Exit And Clean Up

`stop_gate_guard.py` runs when Codex attempts to end the root turn/session. It
finds the active AMMO artifact directory and nudges instead of blocking:
Stage 7/7b receives a continuation nudge, and a terminal campaign without
`REPORT.md` receives a report-generation nudge (repeat stop attempts re-nudge;
this is intentional). Paused campaigns and subagent Stop payloads are allowed.
Trusted server-session and Codex-thread identity mismatches remain blocking.
On allowed stop the hook runs
`gpu_reservation.py release-session --include-children` for the active session.
`AMMO_ALLOW_STOP=1` skips the nudge and performs cleanup.

Auditor Phase 2 is enforced by the PostToolUse/SubagentStop lifecycle path,
which mirrors Claude's agent-local audit hook without turning root Stop into an
additional campaign-policy validator.

### 5.4 PreCompact + SessionStart: Atomic Resume Checkpoints

`pre_compact.py` handles Codex 0.144.1's native `PreCompact` event for both
`manual` and `auto` triggers. For a non-terminal AMMO campaign it safely resolves
the campaign-local `state.json` and atomically writes
`compaction_checkpoint.json`. The checkpoint records the canonical state path,
current round, stage, campaign/round statuses, and active track identities. It
does not copy derived cumulative-speedup values or obsolete Claude team fields.
No AMMO state, or only terminal AMMO state, leaves unrelated compaction alone;
an identified active campaign blocks compaction only when its checkpoint cannot
be written safely.

`session_start.py` consumes that checkpoint on `startup|resume`, injects a resume
packet telling the lead to read `.codex/skills/ammo/SKILL.md`, reload the
referenced `state.json`, and resume the current stage, then removes the consumed
checkpoint. If it only finds `state.json`, it injects a lighter "AMMO
Optimization Detected" context block.

Codex also exposes `PostCompact`, but AMMO intentionally does not register it:
the current PostCompact command output is universal status/message metadata and
does not provide SessionStart's `additionalContext` channel. Durable
`PreCompact` state plus the existing `SessionStart` resume packet is therefore
the supported context-restoration path.

### 5.5 Monitor Queue Delivery

Transcript monitors append JSONL records to `monitor_interventions.jsonl`.
Records target the real active Codex session id, not a champion name or op id.
`common.py` discovers the campaign-level queue and nested track queues, filters
open records for the active session, and exposes them to PreToolUse/PostToolUse.

Acknowledgement uses:

```bash
python .codex/skills/ammo/scripts/monitor_queue_ack.py \
  --session-id <real-session-id> \
  --queue <monitor_interventions.jsonl> \
  --record-id <record-id> \
  --note "<what changed or why rebutted>"
```

Static inspection remains allowed so the lead can read evidence before
acknowledging. Non-inspection tool calls are blocked while acknowledged-required
records remain open.

### 5.6 What Hooks Are Not

Hooks are guardrails, not the whole AMMO policy. Claude-defined gates,
`state.json`, round-scoped artifacts, and the configured agents remain the
source of truth for campaign decisions. If a hook warning and structured
evidence disagree, resolve the discrepancy by reading the relevant gate,
schema, and artifact JSON rather than treating the warning as final.
