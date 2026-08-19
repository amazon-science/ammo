# AMMO: How We Keep Agents Honest on 8-Hour Autonomous GPU Optimization Campaigns

**Date:** 2026-05-21
**Author:** Jin Huang
**Audience:** AI/ML engineers familiar with vLLM and CUDA, new to multi-agent orchestration

---

## 1. Introduction

AMMO (Agentic Model-on-Machine Optimizer) is a fully autonomous multi-agent system that finds and implements kernel-level performance optimizations for vLLM inference. Point it at a deployment — model, hardware, dtype, parallelism — and it runs unattended for four to eight hours, profiling the workload, debating optimization candidates, writing actual CUDA/Triton kernels, validating them, and shipping the ones that survive end-to-end measurement.

The interesting engineering problem here is not the optimization. There are mature catalogs of GEMM tricks, fusion patterns, and quantization strategies. The interesting problem is **how to keep an agent honest, grounded, and consistent over the horizon required to actually do this work.** Multi-hour agentic tasks fail in distinctive ways: agents hallucinate success, optimize proxy metrics, skip mandatory gates as their context fills with tool output, and benchmark against environments that have nothing to do with production. AMMO is, at its core, a system of mechanisms designed to make those failures hard.

This document explains those mechanisms in the order we found them to matter, grounded in real H100 campaign data and a controlled 145-cell ablation study.

### What this document covers

| Section | Topic | Read if you want to... |
|---------|-------|------------------------|
| §2 | Campaign Workflow | Understand the 7-stage pipeline, agent roles, and how rounds iterate |
| §3 | The Problem | See the specific failure modes that motivated each mechanism |
| §4 | The Mechanism Hierarchy | Understand, in priority order, what actually keeps agents on-rails |
| §5 | The Hook System | See real enforcement code and understand the runtime layer |
| §6 | Annotated Walkthrough | Follow a real H100 campaign and see where each mechanism fires |
| §7 | When the Hierarchy Hurts | Understand the costs and limitations (ablation counter-evidence) |
| §8 | Eval & Closing | See how we measure whether the system works beyond individual campaigns |

---

## 2. Campaign Workflow

A campaign is an iterative loop of seven stages, repeated until the top remaining bottleneck falls below a configurable threshold (default 0.5% of E2E). Each iteration is called a *round*.

```
                            AMMO Campaign Pipeline
 ================================================================================

 The campaign is an iterative loop of 7 stages. Each iteration (round) discovers,
 debates, and implements optimizations. The loop repeats until the top bottleneck
 falls below the mechanical stop threshold (configurable, default 0.5%).

 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                          ROUND N (Stages 1-7)                                  │
 │                                                                                │
 │  Stage 1: Baseline Capture (ammo-researcher, task_type: baseline)              │
 │  ┌──────────────────────────────────────────────────────────────────────────┐   │
 │  │  1. Lead scaffolds target via new_target.py                              │   │
 │  │  2. Researcher runs nsys probe → determines profiling tier               │   │
 │  │  3. TWO-INVOCATION pattern (critical — separates timing from profiling): │   │
 │  │     a) Clean E2E sweep: --slot baseline --capture-golden-refs            │   │
 │  │        (NO profiling flags — this is the AUTHORITATIVE timing)           │   │
 │  │     b) Profiling sweep: --slot profiling --nsys-profile OR --torch-prof  │   │
 │  │        (timing here is CONTAMINATED — used only for trace analysis)      │   │
 │  │  4. Researcher writes constraints.md                                     │   │
 │  │                                                                          │   │
 │  │  Output: sweeps/baseline/e2e_latency_results.json (AUTHORITATIVE)        │   │
 │  │          profiling/{nsys,torch_profile}/* (for Stage 2 mining)           │   │
 │  │          constraints.md                                                  │   │
 │  └──────────────────────────────────────────────────────┬───────────────────┘   │
 │                                                          │                      │
 │                                              AUDIT GATE: T_AUDIT_S1             │
 │                                              (baseline artifacts exist,         │
 │                                               profiling tier determined)        │
 │                                                          │                      │
 │                                                          v                      │
 │  Stage 2: Bottleneck Mining (ammo-researcher, task_type: mining)                │
 │  ┌──────────────────────────────────────────────────────────────────────────┐   │
 │  │  Researcher analyzes Tier 0/1 traces:                                    │   │
 │  │  - Top-K kernels by GPU time                                             │   │
 │  │  - Component share f (fraction of E2E each kernel contributes)           │   │
 │  │  - Bandwidth utilization vs physical ceiling                             │   │
 │  │  - NO feasibility estimates, NO E2E projections                          │   │
 │  │                                                                          │   │
 │  │  Output: mining/bottleneck_analysis.md                                   │   │
 │  └──────────────────────────────────────────────────────┬───────────────────┘   │
 │                                                          │                      │
 │                                              AUDIT GATE: T_AUDIT_S2             │
 │                                              (bottleneck_analysis +             │
 │                                               e2e_latency_results exist)        │
 │                                                          │                      │
 │                                                          v                      │
 │  Stage 3: Adversarial Debate (NO audit after — champions are adversarial)       │
 │  ┌──────────────────────────────────────────────────────────────────────────┐   │
 │  │  TeamCreate: ammo-round-{round_id}-{model_short}-{hardware}              │   │
 │  │                                                                          │   │
 │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │   │
 │  │  │Champion 1│  │Champion 2│  │Champion 3│  (2-4 ammo-champion agents)   │   │
 │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                              │   │
 │  │       │              │              │                                    │   │
 │  │  Phase -1: Target Claim Waterfall (structural diversity insurance)       │   │
 │  │       │  Champions read bottleneck_analysis, claim components via        │   │
 │  │       │  waterfall: highest-f unclaimed → claim. Prevents monoculture.   │   │
 │  │       │  Orchestrator reviews claim distribution (5-check rubric)        │   │
 │  │       │                                                                  │   │
 │  │  Phase 0: Independent proposals + micro-experiments                      │   │
 │  │       │  GATES: Custom Kernel Mandate, Technology Selection,             │   │
 │  │       │         Precision Classification, Category Block                 │   │
 │  │       │                                                                  │   │
 │  │  Rounds 1-2+: Evidence ──> Critique ──> Rebuttal (min 2, max 5)         │   │
 │  │       │              │              │                                    │   │
 │  │       └──────────────┼──────────────┘                                   │   │
 │  │                      v                                                   │   │
 │  │  Lead scores via rubric ──> Select 2-3 winners ──> shutdown champions   │   │
 │  │  Output: debate/summary.md  (round team persists for Stages 4-5)        │   │
 │  └──────────────────────────────────────────────────────┬───────────────────┘   │
 │                                                          │                      │
 │                                                          v                      │
 │  Stages 4-5: Parallel Worktree Tracks (Adversarial Validation)                  │
 │  ┌──────────────────────────────────────────────────────────────────────────┐   │
 │  │                                                                          │   │
 │  │  STEP 1: Spawn impl-champion + transcript-monitor per track              │   │
 │  │  ┌─────────────────────────┐       ┌─────────────────────────┐          │   │
 │  │  │ Track A (worktree)      │       │ Track B (worktree)      │          │   │
 │  │  │ ammo-impl-champion      │       │ ammo-impl-champion      │          │   │
 │  │  │ - Activate worktree venv│       │ - Activate worktree venv│          │   │
 │  │  │ - Write kernel code     │       │ - Write kernel code     │  GPU     │   │
 │  │  │ - Run validation gates  │       │ - Run validation gates  │ isolated │   │
 │  │  │   (mechanical checks)   │       │   (mechanical checks)   │          │   │
 │  │  │ - E2E via sweep script  │       │ - E2E via sweep script  │          │   │
 │  │  └─────────┬───────────────┘       └─────────┬───────────────┘          │   │
 │  │            │                                  │                          │   │
 │  │  ┌─────────────────────────┐       ┌─────────────────────────┐          │   │
 │  │  │ transcript-monitor A    │       │ transcript-monitor B    │          │   │
 │  │  │ (real-time compliance,  │       │ (real-time compliance,  │          │   │
 │  │  │  mid-turn injection via │       │  mid-turn injection via │          │   │
 │  │  │  ammo-msg-check hook)   │       │  ammo-msg-check hook)   │          │   │
 │  │  └─────────────────────────┘       └─────────────────────────┘          │   │
 │  │                                                                          │   │
 │  │  STEP 2: Wait for ALL tracks to reach terminal status                   │   │
 │  │          (PASS / GATED_PASS / FAIL — do NOT stop early)                 │   │
 │  │                                                                          │   │
 │  │  STEP 3: TeamDelete round team                                          │   │
 │  │  STEP 4: AUDIT GATE: T_AUDIT_S45 (all tracks terminal)                 │   │
 │  │                                                                          │   │
 │  └──────────────────────────────────────────────────────┬───────────────────┘   │
 │                                                          │                      │
 │                                                          v                      │
 │  Stage 6: Integration Validation                                                │
 │  ┌──────────────────────────────────────────────────────────────────────────┐   │
 │  │  Single passer? ──> Short-circuit: Stage 5 results = integration result  │   │
 │  │                                                                          │   │
 │  │  Multiple passers, disjoint files?                                       │   │
 │  │       ──yes──> Cherry-pick both, re-run E2E (--fresh-cache) ──> SHIP    │   │
 │  │                                                                          │   │
 │  │  Multiple passers, file conflicts?                                       │   │
 │  │       ──GATED_PASS──> Spawn ammo-resolver, DA review ──> SHIP           │   │
 │  │       ──PASS only──> Pick best E2E single candidate ──> SHIP            │   │
 │  │                                                                          │   │
 │  │  None pass? ──> round EXHAUSTED (not campaign-level)                    │   │
 │  └──────────────────────────────────────────────────────┬───────────────────┘   │
 │                                                          │                      │
 │                                              AUDIT GATE: T_AUDIT_S67            │
 │                                                          │                      │
 └──────────────────────────────────────────────────────────┼──────────────────────┘
                                                            │
                                                            v
 Stage 7: Campaign Evaluation (mechanical — no user interaction)
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                 │
 │  IF SHIP:                              IF round EXHAUSTED:                      │
 │    1. Record shipped candidates          1. Record failed round                 │
 │    2. Update cumulative speedup          2. Mechanical threshold check           │
 │    3. Mining on new baseline               on EXISTING profile (no re-profile)  │
 │       (Stage 6 integration sweep IS      │                                      │
 │        the post-SHIP measurement —       │                                      │
 │        no separate re-profile)           │                                      │
 │    4. Mechanical threshold check          │                                      │
 │       │                                  │                                      │
 │       v                                  v                                      │
 │  top bottleneck f < threshold?       top bottleneck f < threshold?              │
 │    YES → campaign_complete             YES → campaign_exhausted                 │
 │    NO  → next round (Stage 3)          NO  → new debate from existing data      │
 │                                                                                 │
 │  On campaign_complete or campaign_exhausted:                                    │
 │    → Spawn report subagent (background) → REPORT.md                            │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

The diagram looks busy because the system is built for adversarial robustness, not minimum complexity. A few things are worth surfacing in prose.

**The pipeline is agent-specialized.** Nine distinct agent definitions divide the labor. The lead orchestrator (the main Claude session) drives the state machine and never writes kernel code. The `ammo-researcher` profiles and mines bottlenecks, but is forbidden from making feasibility estimates — its job is to surface what's expensive, not to predict what will be fast. The `ammo-champion` proposes and argues, backed by mandatory micro-experiments. The `ammo-impl-champion` writes the actual kernel inside an isolated git worktree, then runs a fixed ladder of validation gates whose pass/fail conditions are defined by schema and checked mechanically by `verify_validation_gates.py` — the implementer executes the gates but does not get to define what passing means. An `ammo-auditor` runs after each major stage, performing a four-phase cold reconstruction of what should have happened versus what did. An `ammo-transcript-monitor` watches the implementer's tool calls in real time and injects mid-turn corrections through a hook.

**Ten non-negotiables encode the rules of the game.** These aren't aspirational; each is enforced by a hook or a script. Production parity (CUDA graphs + `torch.compile`) is enforced by the sweep script. The vLLM compiled path is the baseline (not naive PyTorch). `torch.allclose()` is mandatory. Raw `vllm bench latency` is forbidden — only the sweep wrapper is allowed. Custom kernel code is mandatory; config-only proposals get rejected at Phase 0. The campaign loop runs autonomously — the orchestrator is forbidden from asking the user whether to continue. Mining respects a tiered profiling strategy keyed off an `nsys` probe's output. None of these are *suggestions to the agent*; they are runtime invariants.

**Audit gates separate stages.** Most stage transitions are blocked behind an auditor verdict. The exceptions are deliberate: there's no audit *after* Stage 3 because debate is itself adversarial, and it would be circular to audit an adversarial process with another adversarial process. Stage 7's mechanical threshold check serves as the audit for the round-to-round transition.

That's the pipeline. The rest of this document is about the techniques that hold it together.

## File Structure

```
.claude/skills/ammo/
├── SKILL.md                              # Main orchestration (campaign loop, task graph, non-negotiables)
├── README.md                             # Workflow diagram, test suite, examples
├── orchestration/
│   ├── audit-protocol.md                 # Four-phase auditor model, gate triggers, invariant lists
│   ├── debate-protocol.md                # Stage 3: team setup, Phase -1 waterfall, convergence
│   ├── parallel-tracks.md                # Stages 4-5: worktree creation, GPU assignment, pass criteria
│   └── integration-logic.md              # Stage 6: conflict detection, cherry-pick, resolver agent
├── references/                           # 25 reference docs (key ones below)
│   ├── validation-defaults.md            # Gate thresholds, pass/fail criteria, "invalid reasons to stop"
│   ├── technology-selection.md           # Kernel authoring class constraints, anti-regression rule
│   ├── nsys-profiling-guide.md           # Baseline-then-capture workflow, nsys/ncu capture, evidence rules
│   ├── torch-profiler-guide.md           # Chrome trace analysis, multi-rank, kernel chains
│   ├── debate-scoring-rubric.md          # 6-criterion weighted scoring (min 5.0 to advance)
│   ├── e2e-delta-math.md                 # f × kernel_speedup = E2E improvement
│   ├── audit-invariants.md               # Per-stage invariant checklists for auditor
│   ├── gpu-pool.md                       # Two-layer GPU reservation architecture
│   ├── torch-compile-contract.md         # torch.compile dispatch and CUDA graph interaction
│   ├── code-templates.md                 # GPU kernel patterns (token-major, expert-major)
│   ├── gpu-configs.md                    # Hardware specs per SM arch (SMEM, registers, TMA)
│   ├── optimization-techniques.md        # Technique catalog T1-T14
│   ├── cudagraph-safety.md               # Stream usage, no allocations during capture
│   └── ...                               # 12 more (artifact-layout, fusion heuristics, etc.)
├── scripts/
│   ├── run_vllm_bench_latency_sweep.py   # E2E benchmarks with GPU lock (170KB — the measurement tool)
│   ├── nsys_probe.py                     # Profiling tier determination (Tier 0/1/2 routing)
│   ├── gpu_reservation.py                # Two-layer GPU pool management within sessions
│   ├── new_target.py                     # Scaffold artifact directory + state.json
│   ├── transcript_filter.py              # Filter/search session transcripts
│   ├── generate_validation_report.py     # Structured reporting for tracks
│   └── ...                               # 6 more (correctness comparator, GPU status, etc.)
├── eval/                                 # Skill evaluation pipeline (causal analysis, scoring, dashboard)
└── report/
    └── SKILL.md                          # Report generation skill (for background report subagent)

.claude/agents/
├── ammo-researcher.md          # 33KB — profiling + bottleneck mining (grounded data only)
├── ammo-champion.md            # 25KB — debate: proposals, micro-experiments, argues with data
├── ammo-delegate.md            # 7KB  — fire-and-forget research subagent for champions
├── ammo-impl-champion.md       # 31KB — implements kernel in isolated worktree
├── ammo-auditor.md             # 33KB — four-phase post-stage review (cold reconstruction)
├── ammo-transcript-monitor.md  # 38KB — real-time compliance monitoring + mid-turn injection
├── ammo-investigator.md        # 16KB — deep-dive research spawned on demand
├── ammo-report-writer.md       # 6KB  — background REPORT.md generation with charts
└── ammo-resolver.md            # 5KB  — merge conflict resolution during Stage 6
```

## Specialized Agents

| Agent | Role | Key Constraint |
|-------|------|----------------|
| **ammo-researcher** | Profiles baseline, mines bottlenecks | Cannot make feasibility estimates or E2E projections |
| **ammo-champion** | Proposes optimizations, argues in debate | Must back claims with micro-experiments; technology selection required |
| **ammo-delegate** | Research/profiling subagent for champions | Fire-and-forget; returns data, not recommendations |
| **ammo-impl-champion** | Implements kernel in isolated worktree | Frontmatter Stop hook enforces validation gates + Amdahl's sanity check |
| **ammo-auditor** | Four-phase post-stage review | Cold reconstruction → invariant checklist → reconciliation → verdict |
| **ammo-transcript-monitor** | Real-time compliance monitoring | Watches tool calls live; mid-turn message injection via hook |
| **ammo-investigator** | Deep-dive research on demand | Spawned by orchestrator when derivation fails or needs verification |
| **ammo-report-writer** | Final deliverable generation | Background; produces REPORT.md with charts on campaign completion |
| **ammo-resolver** | Merge-conflict resolution in Stage 6 | Spawned only when cherry-picking passing tracks produces git conflicts |

The **lead** (main Claude session) orchestrates all stages, manages `state.json`, owns all gates, and never writes kernel code directly.

### Which Model Runs Which Agent

Each agent pins a model in its frontmatter, and
`ai_cli_session/.claude/models.json` records every pin in one place (it is a
roster, not a runtime switch — `tests/unit/test_model_roster.py` fails if any
pin site disagrees with it). The pins as shipped:

| Agent | Model | Effort | Why this model |
|-------|-------|--------|----------------|
| **lead** (session) | `claude-opus-5` | xhigh | Drives the state machine for hours; must follow long procedural documents without drift. State-machine discipline degrades first on smaller models. |
| **ammo-researcher** | opus | xhigh | Mining is the round's foundation — a misread profile poisons every downstream decision. Runs once per round, so the cost is bounded. |
| **ammo-champion** | opus | xhigh | Debate quality decides what gets built. Champions must design micro-experiments and argue against each other; weak champions converge on shallow candidates. 2-4 run per round. |
| **ammo-impl-champion** | opus | xhigh | Writes the actual CUDA/Triton kernel and drives it through the validation gates. Kernel bugs cost GPU-hours to discover, so this is the wrong place to save tokens. |
| **ammo-auditor** | opus | xhigh | Adversarially reconstructs each stage from artifacts alone. An auditor weaker than the agents it audits will rubber-stamp their mistakes. |
| **ammo-transcript-monitor** | opus | inherited | Long-running background reviewer of implementer transcripts; needs enough judgment to distinguish methodology errors from noise. |
| **ammo-investigator** | opus | inherited | Spawned only on demand when state looks inconsistent or a decision fork needs evidence; rare, so pinning big costs little. |
| **ammo-report-writer** | opus | inherited | One run at campaign end; report quality is the deliverable. Its fact-checker Stop hook runs on sonnet. |
| **ammo-resolver** | opus | inherited | Rare, but a bad conflict resolution silently corrupts the shipped integration. |
| **ammo-delegate** | sonnet | inherited | Fire-and-forget data gatherer for champions: runs a command, reads a profile, returns numbers. No decisions to make, and many spawn per debate — this is the high-volume role, so it gets the cheaper model. |

"xhigh" rows pin `effort: xhigh` in the agent frontmatter; "inherited" rows
take the session-wide `CLAUDE_CODE_EFFORT_LEVEL=xhigh` from
`settings.local.json`. The explicit pins exist so the four
mistake-propagating roles keep maximum effort even if you lower the
session-wide default.

The pattern: **models are matched to the blast radius of a mistake, not to
the prestige of the role.** Agents whose errors propagate (mining, debate,
kernel code, audits) get the strongest model at the highest reasoning
effort. The one high-volume role with no decision authority (delegate) gets
a mid-tier model. Nothing runs on the smallest tier by default — haiku is
wired into `settings.local.json` (`ANTHROPIC_DEFAULT_HAIKU_MODEL`) only so
tooling that asks for "haiku" resolves to something current.

### Reducing Cost

The shipped configuration is the recommended one — it is what the ablation
and eval numbers in this document were measured with. But every pin is
yours to change. Token spend concentrates in three places: the champions
(debate rounds with micro-experiments), the impl-champions (kernel
iterations against GPU feedback), and the xhigh reasoning effort on the
five heavyweight pins. In rough order of savings vs. risk:

1. **Lower reasoning effort.** Set `CLAUDE_CODE_EFFORT_LEVEL=high` in
   `settings.local.json` and change the four frontmatter `effort: xhigh`
   pins (researcher, champion, impl-champion, auditor) to `high`. Biggest
   single lever; keeps every model the same. Expect somewhat shallower
   debate arguments and audit reconstructions.
2. **Drop ammo-delegate from sonnet to haiku.** Cheap and low-risk — the
   role returns raw data. Verify your haiku tier handles the profiling
   tool output sizes.
3. **Drop transcript-monitor, investigator, resolver, report-writer to
   sonnet.** These are judgment roles but not on the kernel-critical path;
   sonnet handles them acceptably. The report loses some polish.
4. **Drop champions to sonnet.** Last resort — this is where quality is
   made. Expect fewer novel candidates and weaker adversarial pressure in
   debate; the mechanical gates still prevent wrong results from shipping,
   but more rounds may end EXHAUSTED.

Do not drop the **auditor** below the strongest model you run elsewhere:
an auditor weaker than the audited agents defeats its purpose.

To change a pin: edit the agent's frontmatter `model:` line, update
`ai_cli_session/.claude/models.json` in the same commit, and run
`pytest tests/unit/test_model_roster.py` — it fails on any site you missed.
The session model itself lives in `ai_cli_session/.claude/settings.local.json`
(`"model"` plus the three `ANTHROPIC_DEFAULT_*_MODEL` alias variables that
`opus` / `sonnet` / `haiku` resolve to).

---

## 3. The Problem: Why Long-Horizon Agent Tasks Fail

To understand why AMMO looks the way it does, it helps to internalize the failure modes. These are not abstract; they are things we watched happen in early campaigns before the corresponding mechanism existed.

### 3.1 Agent drift and reward hacking

Over a multi-hour session, an agent under optimization pressure on proxy metrics will, eventually, find ways to satisfy the proxy without satisfying the goal. In kernel optimization this manifests in characteristic ways.

**Dispatch-dead code.** A kernel passes every kernel-level gate — numerical correctness, golden-reference parity, impressive speedup — but the E2E sweep shows zero improvement. The fused kernel is fast, but it's fast on a code path that doesn't execute in production. The production backend routes around it via a dispatch guard the implementer didn't account for.

**Config-only optimization.** An agent under iteration pressure produces a steady stream of "improvements" by twiddling autotune configs, environment flags, or scheduler parameters. It looks like progress. It ships nothing meaningful, and the cumulative speedup never moves.

**Self-validation.** When the same agent writes a kernel and writes the test that proves it correct, the test drifts toward validating what was built rather than what is needed. The implementer doesn't intend this; it's a natural consequence of having both jobs.

**Benchmark gaming.** CUDA graphs add capture overhead to a benchmark's first iteration; `torch.compile` adds 30-60 seconds of warmup. An agent in a hurry will find reasons not to run with them. Anything the agent measures will look fast — because the slowness it removed was real, just not removable in production.

The hard part: the agent genuinely believes it is succeeding. There is no malice. There is just optimization pressure on a proxy that correlates only weakly with the actual goal.

### 3.2 Context rot and instruction dilution

Modern context windows are large enough to hold an entire campaign's worth of tool output. They are not large enough to hold that output *and* keep the agent's attention sharply focused on a 30-page SKILL.md it read 100K tokens ago.

This is not a compaction problem. Compaction actually helps; it discards noise and re-anchors the agent. The problem happens *within* a single uncompacted window. The instructions are still in context, technically visible — but functionally, they're wallpaper, drowned out by 95% operational data.

Typical symptoms: an agent that knows "always use the sweep script" will, eight hours in, casually invoke `vllm bench latency` directly. Another skips mandatory gates because the previous track passed cleanly and "the pattern was familiar." A third re-enters a completed stage, generating artifacts that contradict the active state.

You cannot fix this with better prompts. The instructions are there. What's failing is attention allocation in a context where the relevant rule is one paragraph among hundreds of pages of profiling data, sweep logs, and git diffs. The only solution is to move enforcement out of the prompt and into the runtime. Hooks that fire on every tool call don't degrade with context length.

### 3.3 Production parity violations

vLLM in production runs CUDA graphs plus `torch.compile`. Both fundamentally change what runs on the GPU.

CUDA graphs collapse multi-launch sequences into a single GPU submission. The kernel-launch overhead disappears, but only kernels marked safe (no allocations, no host syncs, deterministic stream usage) can be captured. `torch.compile` eliminates Python overhead, fuses operations, and routes through different attention kernels depending on the workload. The same model in eager mode and compiled mode has entirely different kernel traces.

Benchmarking without these is benchmarking a different program. An agent that "optimizes" an op that doesn't fire under graph capture has optimized nothing. An "optimization" that allocates GPU memory inside a captured region is a regression in production, where it will simply break capture.

The hard part is that production parity is *expensive*. Compile warmup is slow, profilers are flaky against captured graphs, and eager-mode iteration is friendly and fast. Every shortcut the agent takes for iteration speed moves it further from production truth.

---

## 4. The Mechanism Hierarchy: What Actually Makes This Work

The order below reflects empirical importance. Where we cite numbers like "+80pp" or "0 of 5 vs 5 of 5," those come from a controlled ablation where we ran the same optimization task repeatedly, toggling one mechanism at a time to isolate its effect.

### 4.1 A clear metric to hill-climb

Everything else is decoration if the agent doesn't know what it's optimizing. AMMO's North Star is a single number: end-to-end latency, in seconds, measured by a dedicated sweep script under production parity.

This works because it leaves nothing to subjectivity. It is system-level, not a kernel-level proxy. It is measured under CUDA graphs + `torch.compile`. It is measured the same way every time, from a sweep script the agent invokes but does not control. The campaign loop is mechanical: `improvement >= threshold → continue, else stop`. Champions are scored on projected E2E impact. Tracks pass or fail on E2E delta. Integration ships on E2E. There is exactly one number that matters, and the agent has no path to fudge it.

Without this anchor, the rest of the hierarchy collapses. With it, the other mechanisms have something concrete to defend.

### 4.2 Grounding — agents must touch grass

LLMs hallucinate when they reason in pure isolation for too long. AMMO forces continuous contact with physical reality.

Profiling produces ground truth. `nsys`, `ncu`, and `torch.profiler` traces are measurements, not arguments. The researcher cites specific kernel times, bandwidth utilizations, and component shares from real GPU traces — and schema validation rejects unsourced numbers. Champions cannot enter debate without micro-experiments: small empirical probes that produce concrete throughput or latency numbers from actual GPU runs. "I think this would be fast" is never accepted. The implementer must compile, pass tests, and run a full E2E sweep before its track is even eligible for a verdict. No stage allows more than about 30 minutes of pure reasoning without contact with the GPU.

Ablation evidence: on a task with a hidden accuracy wall, agents without debate shipped 2 of 5 by pivoting freely based on empirical feedback. Agents *with* debate shipped 0 of 5 — debate locked them all onto the same non-viable target. The grounding mechanism (try, measure, pivot) beat structured deliberation when the task rewarded creative search.

### 4.3 Orchestrator non-discretion

Counter-intuitively, the most powerful agent in the system — the Opus orchestrator — has the *least* policy authority. It scaffolds, delegates, and gates. It does not stop, does not override config, and does not make qualitative judgments about whether to continue.

Termination is mechanical. The campaign stops when, and only when, every viable component's projected E2E share falls below the configured threshold. The orchestrator is explicitly forbidden from terminating because "we've tried enough approaches," "diminishing returns feel likely," "the remaining targets are library kernels," or "we've been running for X hours." Configuration is similarly fixed; if the orchestrator believes the threshold should change, it must propose the change to the user and wait, not act on its own taste.

This matters because the most expensive failure mode for autonomous agents on long tasks is *premature termination on plausible grounds*. The agent convinces itself it's done because it has plausible-sounding reasons. Removing this discretion entirely — making stop a function of a numeric threshold the orchestrator cannot rewrite — eliminates the failure category outright.

### 4.4 Adversarial structure and audit gates

Within a single round, the pipeline catches its own failures. The champion proposes, the implementer builds, the validator tests, results come back. What can go wrong over many rounds is more insidious: the *new baseline* after integration is somehow slower than before the optimization shipped; the orchestrator forgets to activate an optimization flag in a later round; state fields silently contradict each other; a researcher's profiling sweep mutates the baseline environment via side effect. These are entropy phenomena, not single-point failures.

The defense is structural role separation, plus an auditor that reads cross-round.

| Role            | What it does                                  | What it cannot do                       |
|-----------------|-----------------------------------------------|-----------------------------------------|
| Champion        | Proposes optimizations, argues with evidence  | Cannot implement or validate            |
| Impl-champion   | Writes kernel code in an isolated worktree    | Cannot write its own validation tests   |
| Impl-validator  | Writes independent correctness/perf tests     | Cannot see the champion's tests         |
| Auditor         | Cold-reconstructs and reviews after each stage | Cannot override decisions               |
| Transcript monitor | Watches tool calls in real time, injects mid-turn | Cannot modify code                  |

The auditor's four-phase model is the multi-round consistency mechanism. After every gated stage transition, an independent auditor agent runs: it cold-reconstructs (what *should* have happened, derived from campaign state and artifacts alone), walks an invariant checklist (cross-field and cross-round invariants the schema can't express), reconciles cold reconstruction against actual state, and renders a verdict — PASS with observations, or severity-ranked findings.

Ablation evidence: on a task requiring precise dispatch methodology (CUDA-graph-safe kernel routing), agents without debate shipped 1 of 5. With debate, 5 of 5. The +80pp lift came from debate specifying methodology the solo agent couldn't discover on its own. The transcript monitor adds another layer: without it, 12% of runs declared "done" without producing any artifacts — pure shutdown hallucination. With the monitor watching, zero.

### 4.5 State machine plus hook enforcement

The defense against context rot is architectural. Move the rules out of the prompt and into the runtime.

A single state file is the source of truth for the campaign. Every stage transition, every track verdict, every audit result is recorded as structured JSON validated against a schema. After compaction or context rot, the agent reconstructs "where am I?" by reading the file from disk, not by remembering. The schema expresses structural rules; cross-field rules (like "Stage 6 requires all tracks terminal") live in hook code.

Nineteen-plus production hooks fire on every tool use. The agent doesn't need to remember the rules; the runtime enforces them mechanically. They span guardrail warnings on every Bash call, state validation on every state-file write, stage-aware context injection after every tool use, inter-agent coordination that injects undelivered teammate messages mid-turn, automatic GPU reservation release, lifecycle protection that blocks session exit during an active campaign, and compliance checks on artifact metadata.

The next-step reminder hook deserves a specific call-out: it's *edge-triggered*, not level-triggered. After every tool use, it reads the campaign state, compares it to a cached snapshot, and emits stage-appropriate guidance only on state changes. The agent is continuously oriented to its current stage and next action — without reminders being drowned out by repetition.

### 4.6 Production parity

Mechanism 4.1 — a clear metric — is only as good as the conditions under which the metric is measured. Production parity makes the hill real instead of an illusion.

A dedicated sweep script is the only sanctioned measurement path. It always runs with CUDA graphs and `torch.compile`. There is no flag to bypass them from the agent's side. The script also sanitizes vLLM optimization environment variables (preventing cross-track contamination), separates clean measurement runs (used for speedup math) from profiling runs (with profiler overhead, used only for trace analysis), captures golden reference outputs for next-round correctness checks, and uses file-lock-based GPU sequencing so two sweeps never share a GPU.

The wrong-venv problem is illustrative. If an implementer in a worktree accidentally invokes the session's main `python` instead of the worktree's venv, `import vllm` resolves to the session's editable install — i.e., the unmodified code. The benchmark then measures the *baseline*, not the implementer's changes. We've seen this fail silently, with the implementer celebrating a "tiny improvement" that is in fact zero. AMMO defends against it three ways: the implementer's own checklist (prompt-level), a guardrail hook's one-shot block when cwd is in a worktree (runtime-level), and the transcript monitor's critical-pattern check on `import vllm` resolving to the wrong path (real-time inspection). Three independent layers because two have failed in production.

Ablation evidence: on a task where baselines drift over time due to upstream vLLM changes, agents without a monitor shipped 0 of 5; agents with a monitor shipped 2 of 5. The monitor caught that baselines were stale and triggered a fresh re-capture. Without re-capture, the correctness gate always failed against outdated references.

### 4.7 Match the guardrail to the failure type

The most uncomfortable finding from our ablation work: no single mechanism dominates across all tasks.

| When the binding constraint is...            | The best mechanism is...   | Effect size |
|----------------------------------------------|----------------------------|-------------|
| Specification quality (what to optimize)     | Debate                     | +80pp       |
| Methodology integrity (measuring correctly)  | Transcript monitor         | +40pp       |
| Fundamental physics (target non-viable)      | Nothing — all arms fail    | 0pp         |
| Creative search (pivot freedom needed)       | Actor independence         | Debate *hurts* by 40pp |
| Process reliability (no false ships)         | Transcript monitor         | 12% hallucination → 0% |

More mechanisms do *not* always mean better outcomes. When we aggregate across all tasks, the headline debate advantage collapses from +20pp to roughly +8pp after quality adjustment. The value is in matching mechanism to failure type — and the wrong guardrail on the wrong task actively hurts.

---

## 5. The Hook System

If section 4 is the why, section 5 is the how. Hooks are how AMMO moves enforcement out of the prompt and into the runtime.

Claude Code hooks are shell scripts that fire on lifecycle events: `PreToolUse` before any tool call, `PostToolUse` after, `Stop` when the agent attempts to end the session, `PreCompact` before context compression, `SessionStart` on start or resume. Hooks communicate back to the agent through `additionalContext` (injected as system guidance), `decision: block` (prevents execution), `stderr` (warning messages), and exit code (allow/block on Stop).

AMMO ships nineteen-plus production hooks. Three patterns illustrate the design space.

### 5.1 Pattern 1: intercept-and-warn (`ammo-pretool-guard.sh`)

This hook fires on every Bash command. It regex-matches dangerous patterns and emits warnings — but, deliberately, does not block. Blocking creates friction; warnings educate.

```bash
# N1: Production parity reminders
if echo "$COMMAND" | grep -qP 'TORCH_COMPILE_DISABLE\s*=\s*1'; then
    echo "AMMO REMINDER: TORCH_COMPILE_DISABLE=1 detected. AMMO non-negotiable N1
    requires production parity (CUDA graphs + torch.compile)." >&2
fi

# N4: Sweep script mandate
if echo "$COMMAND" | grep -qP 'vllm\s+bench\s+latency' && \
   ! echo "$COMMAND" | grep -q 'run_vllm_bench_latency_sweep'; then
    echo "AMMO REMINDER: Raw 'vllm bench latency' detected. AMMO non-negotiable N4
    requires using the sweep script." >&2
fi
```

Three design decisions worth noting. First, this hook warns without blocking — the agent self-corrects, and our experience is that warnings are sufficient for soft guardrails. Second, read-only commands (`grep`, `cat`, `git log`) take a fast-path exit; we don't pay the cost on every read. Third, the *one* place this hook does block — wrong-venv usage in a worktree — uses a one-shot block: it blocks once per session, then trusts the agent. That asymmetry encodes the cost: false positives on a soft guardrail are cheap; false positives on a hard block are expensive.

### 5.2 Pattern 2: schema, invariants, and transition gates (`ammo-state-validate.sh`)

This hook fires on every `state.json` write (detected via the file path or, for Bash writes, by parsing the command string). It performs three levels of validation and *blocks* on failure — state corruption is catastrophic and there is no good recovery from a malformed campaign.

```bash
# Level 1: JSON Schema validation (Draft 2020-12)
ERRORS=$(python3 <<'PY'
from jsonschema import Draft202012Validator
validator = Draft202012Validator(schema)
for err in validator.iter_errors(state):
    errors.append(f"{path}: {err.message}")
PY
)

# Level 2: Cross-field invariant (can't express in schema)
# "Stage 6 transition requires all tracks terminal"
NON_TERMINAL=$(jq '[.campaign.rounds[IDX].parallel_tracks.tracks[]
    | select(.status != "PASS" and .status != "GATED_PASS" and .status != "FAIL")]
    | length' "$STATE_FILE")
if [ "$NON_TERMINAL" -gt 0 ] && [ "$STAGE" = "6_integration" ]; then
    BLOCK "Stage 6 entered with $NON_TERMINAL non-terminal tracks"
fi

# Level 3: Audit-gate transition blocking
# Round N+1 cannot start without audit.stage_67.passed_at on Round N
if [ "$CR" -gt 1 ]; then
    PREV_AUDIT=$(jq ".campaign.rounds[$((CR-2))].audit.stage_67.passed_at // empty")
    if [ -z "$PREV_AUDIT" ]; then
        BLOCK "Round $CR started without stage_67 audit on Round $((CR-1))"
    fi
fi
```

The three layers separate concerns. Layer 1 is structural: the JSON parses, fields have correct types, enums are valid. Layer 2 is semantic: cross-field rules that schemas cannot express. Layer 3 is transitional: audit gates that connect rounds. A legacy bypass exists for old campaigns missing audit fields, so the rules don't apply retroactively. Each gate also carries a `started_at` provenance field, stamped mechanically by a hook the moment the lead dispatches an auditor — Layer 3 blocks a gate that has `passed_at` but no `started_at`, so a lead cannot hand-write a PASS without a real audit having started first.

### 5.3 Pattern 3: edge-triggered context injection (`ammo-next-step-reminder.sh`)

This hook fires after every tool use, reads `state.json`, and injects stage-aware guidance only on state transitions. It is throttled to once per 15 seconds per session, but terminal transitions bypass the throttle — irreversible decisions always get guidance.

```bash
# Throttled to once per 15s per session
LAST_TS=$(stat -c %Y "$THROTTLE_FILE" 2>/dev/null || echo 0)
NOW=$(date +%s)
if [ $(( NOW - LAST_TS )) -lt 15 ] && [ "$IS_TERMINAL_TRANSITION" != "1" ]; then
    exit 0  # Throttled — skip
fi

# Edge detection: compare current vs previous state snapshot
PREV_BASELINE_DONE=$(jq '...' "/tmp/ammo-state-prev-${SESSION_ID}.json")
if [ -n "$BASELINE_DONE" ] && [ -z "$PREV_BASELINE_DONE" ]; then
    # Baseline just completed — emit Stage 2 guidance
    EMIT "Baseline capture complete. Next: spawn ammo-researcher with task_type=mining..."
fi

# Terminal transitions BYPASS the throttle
if [ "$STATUS" = "campaign_complete" ] || [ "$STATUS" = "campaign_exhausted" ]; then
    IS_TERMINAL_TRANSITION=1
fi
```

The key insight is edge-triggered, not level-triggered. The agent doesn't need to be told "you're in Stage 2" every 15 seconds — it would learn to ignore the noise. It needs to be told "you just *entered* Stage 2; here's what comes next." The hook also detects worktree drift (warns if the orchestrator's cwd is inside a worktree, a common mistake) and excludes subagents (only the lead orchestrator gets these reminders; champions and validators don't).

Each hook ships with a companion test script. `test-ammo-state-validate.sh` alone is 32KB of scenarios covering schema violations, cross-field invariants, and audit gates. Hooks are infrastructure, and like all infrastructure they earn their reliability through tests.

---

## 6. Annotated Campaign Walkthrough

Mechanisms are easier to internalize through a story. Here is one.

### 6.1 Qwen3.6-35B-A3B-FP8 — H100, 7 rounds

The target was Qwen3.6-35B-A3B-FP8 — a hybrid architecture with 30 GatedDeltaNet layers and 10 full-attention layers, 256 experts top-8, MTP speculative decoding — running on a single H100 80GB at TP=1. Final result: 1.10x cumulative E2E speedup, GPU utilization moving from 41% to 97% over the campaign. Two optimizations shipped.

**Round 1 ended in EXHAUSTED.** The probe came back GREEN, routing to Tier 0 profiling with `nsys --cuda-graph-trace=node`. The top bottleneck was `fused_moe_kernel` at f=0.262. Three champions debated: a CUTLASS MoE replacement, a RMSNorm + residual + FP8-quant fusion, and a MoE down-proj + moe_sum fusion. Champion-1 went straight to a Tier 2 probe of its CUTLASS proposal, measured 1.04x cold, and self-conceded — *grounding doing its job*. The RMSNorm fusion track passed every kernel-level gate (correctness, golden-ref, 9.23x kernel speedup) — but failed the E2E gate at 0%. The `nsys` trace showed zero calls to the fused kernel in decode. Root cause: `apply_input_quant=False` on the production attention backend; the fused kernel was on a dead code path. *Adversarial validation doing its job — kernel-level tests would have shipped a phantom optimization.*

**Round 2 shipped, but not before catching cross-round drift.** As the round opened, the Stage 4–5 auditor discovered that the Round 1 baseline had been captured with `torch.profiler` active during all five timed iterations, inflating latency by 44–70% (BS=4: 4.690s contaminated vs 3.368s clean). *The auditor doing its job — without it, the round would have shipped a speedup measured against an inflated reference.* Corrected baselines were substituted. Stage 3 settled on the MoE down + moe_sum Triton fusion. The first implementation iteration tried fp32-staging and failed Gate 5.2 (cold 0.85x); the second iteration with bf16-direct passed at 1.211x kernel, 388 lines of code. E2E showed +6.67% at BS=4, +1.40% at BS=8. GSM8K accuracy was identical to baseline (79.00%). `nsys` confirmed 65,184 instances of the fused kernel firing, with the original `moe_sum_kernel` absent from the trace. SHIP.

**Round 6 shipped despite a wrong-venv near-miss.** The track was a MTP preamble delta kernel. The implementer first measured in the *session* venv (which lacked the `delta_advance` code), and the result showed only a 0.18ms improvement — essentially zero. The symptom was a `WARNING: Unknown vLLM environment variable VLLM_MTP_DELTA_PREAMBLE` in the log. *The transcript monitor caught this within three minutes* and demanded a re-run from the worktree venv. The corrected measurement: 9.96x kernel speedup, decode launches dropping from 28 to 1 per step. E2E: +5.45% at BS=4, +6.36% at BS=8. GSM8K dropped 0.99pp (from 79.08% to 78.09%) — within the 1.0pp threshold. The invalid runs were preserved in `invalid_old_runs/` for the audit trail. SHIP.

**Round 7 terminated mechanically.** A re-profile showed `decode_busy = 0.97` (up from 0.41 at campaign start). The remaining bottlenecks were NVJet and DeepGEMM — production-tuned NVIDIA library kernels. The mechanical check showed `f < threshold` for every remaining component. *Orchestrator non-discretion doing its job* — there was no temptation to declare the campaign "stuck" or "diminishing"; the threshold simply did not pass.

### 6.2 Qwen3.6-27B-FP8 — H100, supplementary moments

This campaign (H100, TP=1) closed at 1.11x with three shipped optimizations: an RMSNorm + quant fusion (1.258x kernel, lossless, BS=4 +25.8% E2E), a SiLU + quant chain, and an lm_head FP8 conversion (2.1x kernel, lossy). Two moments are particularly instructive.

The first is a *methodology bug* — `scale_1x128_kernel` was undercounted by 4.5x in the original Round 1 bottleneck analysis due to an attribution error. The discovery happened mechanically: Round 1 exhausted, Round 2 re-mined the bottlenecks with a corrected methodology, and the original `bottleneck_analysis.md` was preserved as `bottleneck_analysis_INVALIDATED.md`. The system's iterative structure surfaced the bug because the audit chain made the inconsistency visible.

The second is a *failed CUTLASS attempt that proves the validation pipeline is real*. Round 2's `gate_up_cutlass_sm90` track tried a CUTLASS SM90 GEMM replacement and failed compilation with `static_assert(MmaTileShape::M >= 128)`. The adversarial validation pipeline (build → test → E2E) caught this before any incorrect results could propagate. Track FAIL. Round continued.

And, as in the 35B campaign, the auditor caught a baseline corruption: in Round 3 the recorded baseline environment had been silently flipped to a non-production setting due to a researcher's pristine-sweep side effect. Restored before integration certification could use the wrong environment.

---

## 7. When the Hierarchy Hurts

A document that reports only successes is a sales pitch, not engineering. The same mechanisms we celebrated above also have failure modes, and the ablation data is unsparing.

**Debate lock-in.** `fp8_r05` is the cleanest example. All five seeds in arm B (debate + actor) locked onto the same target — GDN `in_proj` — which had a non-viable accuracy wall. Two of five seeds in arm A (actor only), free to pivot independently per seed, found viable MLP `gate_up_proj` targets at +11–13%. Debate's strength — converging on a sharp specification — is also its weakness when the sharp specification is wrong. We are still working on this; the current mitigation is the Phase −1 target claim waterfall, which forces structural diversity across champions.

**Monitor over-engineering.** V2 D2 (Opus monitor) on `int4_r05` was *worse* than D1 (Sonnet monitor): the more aggressive monitor pushed implementations toward Dynamo-opaque architectures that hurt numerical fidelity. D1 shipped 3 of 3 on this task; D2 shipped fewer. A monitor that cares too much about every potential issue can pull the implementer away from local optima it would otherwise find.

**Conservatism tax.** On `gemm_r01`, the mean speedup in arm C (debate + impl-champion, no monitor) was 9.26%; in arm D (full AMMO, with monitor) it was 3.74%. Arm D had zero retracted claims (zero false-positive ships), but it also had lower-magnitude wins. The monitor is, in part, a friction system. It shaves the upper tail.

**Quality-adjusted aggregation.** Across the 100 V1 cells, the headline arm B vs arm A ship-rate gap of +20pp collapses to roughly +8pp once you remove noise-level ships and protocol inconsistencies. The story "more mechanisms = better" is not the story the data tells. The story is closer to "match the mechanism to the failure type, accept that some mechanisms have a cost on tasks where they aren't load-bearing, and don't pretend you can have all the upside without paying for it."

We keep all of these mechanisms in production because, on the workloads we actually care about, the failure modes they prevent (dispatch-dead code, stale baselines, hallucinated ships) are dramatically more expensive than the conservatism tax they impose. But the framing matters: AMMO is a system of trade-offs, not a stack of unconditional improvements.

---

## 8. Eval and Closing

Beyond individual campaign telemetry, we exercise the system in two ways.

**Conformance test suite.** Fifty-three scenarios across four test files exercise the orchestrator and each subagent against expected behaviors. The orchestrator alone has 21 scenarios covering resume, campaign evaluation, integration logic, non-negotiable violations, tiered profiling, and baseline promotion. Researcher, champion, and implementer/validator have their own files. Each scenario has a description, expected behavior, and reference output from a baseline run. New skill or hook changes are validated against the suite before deployment.

**Controlled ablation (145 cells).** V1 (100 cells) and V2 (45 cells) hold deployment constant (Qwen3.5-4B / L40S / BF16 / TP=1) and vary one mechanism at a time. V2 specifically shares debate output across arms to eliminate debate-variance confounds; what varies is the monitor's presence and choice of model. All cells are audited for contamination — V1 excluded and re-ran 12% of cells for protocol violations. The numbers in this document are post-audit.

**Causal analysis pipeline.** Post-campaign, an offline pipeline extracts events from session transcripts (JSONL), builds a causal DAG linking agent decisions to outcomes, scores nodes by attribution to the final result, and generates a postmortem. The findings feed back into the next iteration of skill and hook design. This is how, for instance, the Phase −1 target-claim waterfall was added in response to debate lock-in; how the two-invocation baseline pattern was added in response to profiler contamination; and how the monitor's CRITICAL pattern set grew to include `import vllm` path checks.

The system we have today is not the system we started with. AMMO works because the failure modes are measured, the mechanisms are matched to them, and the costs are paid honestly. Hour-long autonomous campaigns were once aspirational; eight-hour campaigns are now routine. The bottleneck is no longer "can the agent stay on task" — it is "can we find the next class of optimization to teach it."

That is, on balance, a good problem to have.

---

## Appendix A: Real artifacts

The walkthroughs in this document (Qwen3.6-35B-A3B-FP8 and Qwen3.6-27B-FP8
campaigns) reference the artifact tree every campaign produces. In your own
session it lives in the session worktree under `kernel_opt_artifacts/` and is
surfaced by the server as the session report (`GET /sessions/{id}/report`).

Key files for follow-up reading:

- `state.json` — full campaign state machine
- `REPORT.md` — final deliverable
- `rounds/*/mining/bottleneck_analysis.md` — profiling-grounded component shares
- `rounds/*/debate/summary.md` — debate outcomes with rubric scores
- `rounds/*/tracks/*/validation_results.md` — per-track validation narratives
- `rounds/*/audits/stage_*.md` — auditor findings
- `rounds/*/sweeps/*/e2e_latency_results.json` — raw E2E measurements

## Appendix B: Ablation data sources

These are paths within the ablation-run artifact bundle released alongside the
paper; they are not part of this repository.

- `ammo-ablation-runs/v1/INVESTIGATION_SUMMARY.md` — V1 causal analysis
- `ammo-ablation-runs/v1/per_task.md` — per-task ship rates
- `ammo-ablation-runs/v1/DEBATE_VARIANCE_INVESTIGATION.md` — confound analysis
- `ammo-ablation-runs/v2/V2_INVESTIGATION_REPORT.md` — V2 synthesis
- `ammo-ablation-runs/v2/investigation_occ_r07.md` — signature datapoint
- `ammo_monitoring_in_the_loop_v3.tex` — formal paper writeup
