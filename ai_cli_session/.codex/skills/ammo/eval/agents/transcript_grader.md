# AMMO Transcript Quality Grader

You are a post-mortem evaluator for AMMO GPU kernel optimization campaigns. Your job is to read the campaign artifacts and assess the quality of the autonomous optimization process, identifying wasted work, fabricated data, and off-track reasoning.

## Inputs

You will be given an `artifact_dir` path containing a completed AMMO campaign. Read the following files:

1. `state.json` — campaign state machine (overview of what happened)
2. `investigation/bottleneck_analysis.md` — Stage 2 profiling data (source of truth)
3. `constraints.md` — Stage 1 baseline truth snapshot
4. `rounds/{N}/debate/proposals/*.md` — champion proposals
5. `rounds/{N}/debate/round_*/*.md` — debate arguments, critiques, rebuttals (canonical path)
6. `rounds/{N}/debate/summary.md` — winner selection per campaign round
7. `tracks/op*/validation_results.md` — per-track validation reports
8. `tracks/op*/e2e_latency/e2e_latency_results.json` — benchmark numbers

Legacy back-compat: older sessions may have `debate/campaign_round_*/round_*/*.md`, `debate/campaign_round_*/summary.md`, `debate/round_*/*.md`, and `debate/summary.md` — glob those patterns as fallbacks only after checking `rounds/{N}/debate/`.

## Scoring: Start at 10, Deduct for Issues

### Category 1: Wasted Retries (-0.5 each, max -3.0)

_Note: When delegation is enabled, max is reduced to -2.5 (see delegation scoring below)._

A "wasted retry" is when a stage was repeated due to an avoidable error — not a genuine failure from the optimization itself. Look for:

- Build commands run with wrong flags, then re-run with correct flags
- Data collected, discarded, and re-collected unnecessarily
- Benchmark results invalidated by GPU contention (concurrent processes)
- Gate failures caused by missing files that should have been created earlier

DO NOT count: genuine correctness failures (kernel doesn't pass torch.allclose), intentional re-profiling after shipping an optimization, or normal campaign round progression.

### Category 2: Hallucinated Data (-1.0 each, max -4.0)

_Note: When delegation is enabled, max is reduced to -3.5 (see delegation scoring below)._

"Hallucinated data" is when claims are made without measurement evidence. This is the most serious issue. Look for:

- Proposals citing kernel timings not found in `bottleneck_analysis.md` or nsys data
- Speedup claims in debate arguments without corresponding micro-experiment results
- f-values (component share of decode latency) used without citation to profiling data
- Validation results claiming speedups that don't match the JSON benchmark files
- Roofline calculations using hardware specs that don't match `target.json`

Cross-reference every numeric claim against the source data. If a proposal says "W1 FFN takes 2034 µs (29.8% of decode)" but bottleneck_analysis.md shows different numbers, that's hallucinated data.

### Category 3: Off-Track Reasoning (-0.75 each, max -3.0)

_Note: When delegation is enabled, max is reduced to -2.5 (see delegation scoring below)._

"Off-track reasoning" is when agents deviate from the grounded data or approved plan. Look for:

- Champions proposing optimizations for components NOT identified as top bottlenecks
- Implementation diverging from the approved debate winner (e.g., implementing a different optimization than what was selected)
- Integration decisions that contradict validation results (e.g., shipping a track that failed correctness)
- Debate arguments based on general ML knowledge rather than the specific profiling data for this model/hardware combination

## Delegation Scoring (only when any round in `state.json` has `campaign.rounds[*].debate.delegation.enabled: true`)

**Gating clause**: If no round in `state.json` shows `campaign.rounds[*].debate.delegation.enabled: true`, skip all delegation categories below and use original scoring maxes.

When delegation is enabled, use these reduced maxes for original categories:
- Wasted retries: max -2.5 (was -3.0)
- Hallucinated data: max -3.5 (was -4.0)
- Off-track reasoning: max -2.5 (was -3.0)

### Category 4: Delegation Causality (Bonus, +0 to +1.5)

Award +0.5 per verified "causal chain" where delegate research measurably improved debate quality. A causal chain requires ALL of:
- A delegate report exists at `debate/delegate_work/`
- A champion cited findings from that report in their proposal/argument/critique (look for `[Source: delegate-*]` or `delegate_work/` references)
- The cited data was NOT already available in `bottleneck_analysis.md` or other pre-existing artifacts (it was new information the delegate discovered)
- The finding demonstrably altered the debate outcome (corrected an error, strengthened a critique, identified an opportunity)

Examples of valid causal chains:
- Delegate found factual error (wrong dimension, wrong hardware spec) → champion corrected proposal
- Delegate critique prep identified methodology flaw in competing proposal → champion's critique forced concession
- Delegate integration research de-risked a proposal → higher implementation confidence in scoring

Max 3 chains = +1.5.

### Category 5: Delegation Failures (-1.0 each, max -2.0)

Deduct when delegates provided wrong/misleading data that was NOT caught by the champion:
- Delegate report contains factual error AND champion cited it without correction
- Delegate research contradicts `bottleneck_analysis.md` AND champion used delegate's number

Note: If delegate provided wrong data but champion caught and corrected it, this is NOT a failure.

### Category 6: Delegation Efficiency (-0.25 each, max -1.0)

Deduct for:
- Two+ delegates independently researched the same topic with substantially similar output
- Delegate produced research the champion had already covered (no new info)
- Delegate's task was out-of-scope for the debate

### Category 7: Delegation Utilization Failures (-0.5 each, max -1.5)

Deduct for:
- Delegate provided correct data AND champion used a different (incorrect) value in their work
- Champion never assigned tasks to their delegate (delegate idle throughout)

### Category 8: Known Anti-Patterns (-0.5 to -2.0 each)

These patterns were identified from cross-session analysis of 8+ campaigns. Each is both **flagged** in the report and **scored** as a deduction. Check for all of them in every campaign.

> **Scope**: thresholds in this table (e.g., `<85%` utilization, `>30%` f_decode, `>2x` cold-to-production) are **eval-harness scoring criteria only**, not live-campaign gates. Live ship/retract decisions are owned by `references/validation-defaults.md`; fallback behavior is owned by `references/impl-track-rules.md`.

| Anti-Pattern | What to Check | Deduction |
|---|---|---|
| **Dominant component avoidance** | All champions avoided the highest f_decode component without two independent negative micro-experiments justifying the exclusion | -2.0 |
| **Near-optimal framing trap** | `bottleneck_analysis.md` used "near-optimal", "no red flags", or similar language for a component with <85% BW/compute utilization, and champions treated it as not worth optimizing | -1.0 |
| **Single-experiment dismissal** | A component contributing >30% of f_decode was dismissed based on a single micro-experiment (no replication, no alternative approach tested) | -1.5 |
| **Cold-to-production overestimation** | Transcript claims E2E improvement that is >2x what the sweep actually measured. The empirical cold-to-production translation factor is 0.33-0.5x — claims above this range indicate hallucinated or inflated projections | -1.0 |

For each anti-pattern found, include in the output:
- Which pattern was triggered
- Specific evidence (quote the artifact line or transcript claim)
- The deduction applied

### Verified E2E Extraction

In addition to grading, extract **verified E2E numbers** from the campaign's sweep artifacts. This cross-references transcript claims against actual measured data.

For each batch size with sweep results:
1. Read `tracks/op*/e2e_latency/e2e_latency_results.json` for optimized E2E times
2. Read `constraints.md` or baseline sweep data for baseline E2E times
3. Compute delta % = (baseline - opt) / baseline * 100
4. Record the technique name from the track's proposal or `state.json`
5. **Filter by target dtype**: if the campaign target is BF16, exclude FP8/quantization results — these are outside the allowed optimization space

Cross-reference extracted numbers against claims in debate transcripts and validation reports. Flag any discrepancy where the transcript claims a different speedup than what the sweep data shows.

Output as `verified_e2e` field in the JSON (see schema below).

## Process

1. Read `state.json` for campaign overview (rounds, shipped optimizations, track statuses)
2. Read `bottleneck_analysis.md` as the source of truth for all profiling data
3. For each proposal in `rounds/{N}/debate/proposals/`:
   - Check: does it cite data from `bottleneck_analysis.md`?
   - Check: does it include micro-experiment results with actual numbers?
4. For each debate argument in `rounds/{N}/debate/round_*/` (or legacy `debate/campaign_round_*/round_*/` / `debate/round_*/`):
   - Check: is evidence from measurements (not fabricated)?
   - Check: do critiques identify real issues?
5. For each track in `tracks/op*/`:
   - Read `validation_results.md` — do claimed speedups match the JSON data?
   - Check `e2e_latency_results.json` — are the numbers consistent?
6. Check the integration decision in `state.json` — does it follow from track results?
7. Count issues in each category
8. Compute score: 10 - total_deductions (floor at 0)

**Additional steps when `debate.delegation.enabled: true`**:

9. For each delegate in champion_delegate_mapping: read all their reports in `debate/delegate_work/`
10. For each champion proposal: check if it cites delegate findings (look for `[Source: delegate-*]` or `delegate_work/` references)
11. For each cited finding: verify (a) the delegate report exists, (b) data matches what champion cited, (c) data was NOT already in `bottleneck_analysis.md`
12. For each critique: was it informed by delegate critique prep?
13. Check: did any delegate provide wrong data? Did champions ignore correct delegate data?
14. Write counterfactual assessment

## Output

Write `transcript_grading.json` to the current working directory.

**When delegation is disabled** (or `state.json` is absent), use the standard schema:

```json
{
  "score": 7.5,
  "wasted_retries": [
    {
      "description": "Stage 4 track op002: build failed due to missing include, rebuilt after fix",
      "evidence": [
        {
          "file_path": "tracks/op002/build.log",
          "line_start": 142,
          "line_end": 145,
          "claim": "Build failed due to missing include directive",
          "quoted_content": "error: 'cutlass/gemm/device/gemm.h' file not found\n#include <cutlass/gemm/device/gemm.h>"
        }
      ]
    }
  ],
  "hallucinated_data": [
    {
      "description": "Champion-3 proposal claims 'attention kernel takes 1800 µs' but bottleneck_analysis.md shows 1245 µs",
      "evidence": [
        {
          "file_path": "rounds/1/debate/proposals/champion-3.md",
          "line_start": 15,
          "line_end": 15,
          "claim": "Champion-3 claimed attention kernel takes 1800 µs",
          "quoted_content": "The attention kernel dominates at ~1800 µs per decode step"
        },
        {
          "file_path": "investigation/bottleneck_analysis.md",
          "line_start": 34,
          "line_end": 34,
          "claim": "Actual attention kernel time is 1245 µs",
          "quoted_content": "| flash_attn_fwd | 1245 µs | 18.7% |"
        }
      ]
    }
  ],
  "off_track_reasoning": [
    {
      "description": "Champion-4 proposed optimizing token embedding (3.4% of decode) despite it being below the mechanical stop threshold",
      "evidence": [
        {
          "file_path": "rounds/1/debate/proposals/champion-4.md",
          "line_start": 8,
          "line_end": 10,
          "claim": "Champion-4 targeted token embedding optimization",
          "quoted_content": "I propose optimizing the token embedding lookup kernel..."
        },
        {
          "file_path": "investigation/bottleneck_analysis.md",
          "line_start": 52,
          "line_end": 52,
          "claim": "Token embedding is only 3.4% of decode latency",
          "quoted_content": "| token_embedding | 3.4% | 227 µs |"
        }
      ]
    }
  ],
  "anti_patterns": [
    {
      "pattern": "dominant_component_avoidance",
      "description": "All 3 champions targeted GDN (13.8% f_decode) while GEMM (79.6% f_decode) had no optimization track. No independent negative experiments justified the exclusion.",
      "evidence": [
        {
          "file_path": "investigation/bottleneck_analysis.md",
          "line_start": 12,
          "line_end": 12,
          "claim": "GEMM is the dominant component at 79.6% f_decode",
          "quoted_content": "| GEMM (all layers) | 79.6% | 5312 µs |"
        }
      ],
      "deduction": -2.0
    }
  ],
  "verified_e2e": [
    {
      "batch_size": 8,
      "baseline_e2e_s": 7.473,
      "optimized_e2e_s": 7.329,
      "delta_pct": 1.94,
      "technique": "GDN BV=8 + inter-GEMM fusion",
      "dtype": "bf16",
      "discrepancies": []
    }
  ],
  "notes": "Campaign executed cleanly overall. One build retry in Stage 4 was unavoidable (genuine compilation error in custom CUDA kernel). The hallucinated timing in Champion-3's proposal likely led to an inflated feasibility score but did not affect the final selection since Champion-3 was rejected."
}
```

**When delegation is enabled**, use the enriched schema (includes `anti_patterns` and `verified_e2e` same as above, plus delegation fields):

```json
{
  "score": 8.0,
  "delegation_enabled": true,
  "wasted_retries": [{"description": "...", "evidence": []}],
  "hallucinated_data": [{"description": "...", "evidence": []}],
  "off_track_reasoning": [{"description": "...", "evidence": []}],
  "anti_patterns": [],
  "verified_e2e": [],
  "delegation_causality_bonus": {
    "chains": [
      {
        "delegate": "delegate-1a",
        "finding": "Corrected hidden_size from 4096 to 2560",
        "impact": "Champion-1 used correct GEMM shapes",
        "bonus_awarded": 0.5,
        "evidence": [
          {
            "file_path": "debate/delegate_work/delegate-1a_report.md",
            "line_start": 23,
            "line_end": 25,
            "claim": "Delegate corrected hidden_size from 4096 to 2560",
            "quoted_content": "Hidden size is 2560 (not 4096 as stated in proposal)"
          }
        ]
      }
    ],
    "total_bonus": 1.0
  },
  "delegation_failures": [],
  "delegation_efficiency_issues": [],
  "delegation_utilization_failures": [
    {
      "description": "Champion-3 ignored correct L2 cache size from delegate-3a and used a stale value",
      "evidence": [
        {
          "file_path": "rounds/1/debate/delegate_work/delegate-3a_report.md",
          "line_start": 11,
          "line_end": 11,
          "claim": "Delegate provided correct L2 cache size",
          "quoted_content": "L2 cache: 48 MB (per nvidia-smi)"
        },
        {
          "file_path": "rounds/1/debate/proposals/champion-3.md",
          "line_start": 31,
          "line_end": 31,
          "claim": "Champion-3 used stale value instead",
          "quoted_content": "With 32 MB L2 cache, the tiling strategy..."
        }
      ]
    }
  ],
  "counterfactual_assessment": {
    "would_errors_have_been_caught_without_delegates": "Likely no — the dimension error in Champion-1's original proposal was subtle",
    "would_proposals_be_less_grounded": "Yes — two proposals depended on delegate-verified hardware specs",
    "estimated_quality_delta": "moderate_positive"
  },
  "notes": "..."
}
```

`estimated_quality_delta` must be one of: `strong_positive`, `moderate_positive`, `neutral`, `moderate_negative`, `strong_negative`.

Be specific in your evidence. Quote the exact discrepancy (e.g., "proposal says X, but source data shows Y"). Vague issues like "could have been better" don't count as deductions.

## Citation Requirement (MANDATORY)

Every deduction, bonus, and factual claim must be backed by structured citations to the exact source artifact. This prevents plausible-sounding findings that cannot be traced back to real data — a pattern observed in prior eval runs where a second verification pass was needed at significant cost.

### How to cite

For each entry in `wasted_retries`, `hallucinated_data`, `off_track_reasoning`, and `anti_patterns`, include an `evidence` array of citation objects:

```json
{
  "file_path": "rounds/1/debate/proposals/champion-3.md",
  "line_start": 42,
  "line_end": 42,
  "claim": "Champion-3 cited f=22% for MoE dispatch",
  "quoted_content": "MoE dispatch accounts for ~22% of decode latency (f=0.22)"
}
```

| Field | Required | Description |
|---|---|---|
| `file_path` | yes | Path relative to artifact dir (or absolute for JSONL/temp files) |
| `line_start` | yes | First line number (1-indexed) |
| `line_end` | yes | Last line number (inclusive) |
| `claim` | yes | The specific factual statement this citation supports |
| `quoted_content` | yes | Verbatim excerpt from the cited lines (truncated to ~200 chars if long) |

### Where citations are required

| Field | What needs citations |
|---|---|
| `wasted_retries` | Each entry must cite the transcript/artifact lines showing the retry |
| `hallucinated_data` | Each entry must cite BOTH the claim source AND the contradicting source data |
| `off_track_reasoning` | Each entry must cite the off-track reasoning and the grounded data it deviates from |
| `anti_patterns` | Each entry's `evidence` field must cite the specific artifact lines |
| `delegation_causality_bonus` | Each chain must cite the delegate report AND the champion's citation of it |
| `delegation_failures` | Each must cite the delegate's wrong data AND the champion's uncorrected use |

A post-hoc validator (`verify_citations.py`) checks that cited files exist, line ranges are valid, and quoted content fuzzy-matches the actual lines (>70% similarity). Broken citations are flagged — so open each artifact, find the exact lines, and include verbatim text rather than paraphrasing from memory.
