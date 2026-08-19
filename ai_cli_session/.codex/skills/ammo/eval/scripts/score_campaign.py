#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Score a parsed AMMO campaign snapshot into a weighted scorecard.

Reads an artifacts_snapshot.json (from parse_artifacts.py) and produces:
  - scorecard.json: structured scoring with 5 dimensions
  - report.md: human-readable markdown summary

Scoring dimensions:
  1. E2E Outcome (40%)    — cumulative speedup, shipped count
  2. Gate Pass Rates (15%) — pass/fail rate
  3. Debate Quality (15%)  — grounding, micro-experiments, filtering
  4. Campaign Efficiency (15%) — rounds, failure rate, convergence
  5. Transcript Quality (15%) — LLM-graded (optional, from transcript_grading.json)

When transcript quality is unavailable, weights redistribute proportionally.

Usage:
  python score_campaign.py --snapshot artifacts_snapshot.json
  python score_campaign.py --snapshot snapshot.json --output scorecard.json --report report.md
  python score_campaign.py --snapshot snapshot.json --enrich-from transcript_grading.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scoring constants (tunable)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "e2e_outcome": 0.40,
    "gate_pass_rates": 0.15,
    "debate_quality": 0.15,
    "campaign_efficiency": 0.15,
    "transcript_quality": 0.15,
}

# E2E speedup tier thresholds
E2E_TIERS = [
    (1.20, 10.0),
    (1.10, 8.0),
    (1.05, 6.0),
    (1.02, 4.0),
    (1.00, 2.0),
]
E2E_EXHAUSTED_DISCOUNT = 0.6  # Exhausted campaigns with shipped opts get 60% of tier score
E2E_EXHAUSTED_NO_SHIP_SCORE = 0.0  # Exhausted with nothing shipped = 0
E2E_SHIP_BONUS_PER_EXTRA = 0.5
E2E_SHIP_BONUS_MAX = 1.5

# Gate pass rate thresholds
GATE_TIERS = [
    (1.00, 10.0),
    (0.80, 7.0),
    (0.50, 4.0),
    (0.00, 1.0),
]

# Campaign efficiency: rounds-to-completion scoring
ROUNDS_TIERS = [
    (1, 10.0),
    (2, 8.0),
    (3, 6.0),
]
ROUNDS_DEFAULT = 4.0
CONVERGENCE_BONUS = 2.0


# ---------------------------------------------------------------------------
# Dimension Scorers
# ---------------------------------------------------------------------------

def _tier_score(value: float, tiers: List[Tuple[float, float]]) -> float:
    """Map a value to a score using tier thresholds (descending)."""
    for threshold, score in tiers:
        if value >= threshold:
            return score
    return tiers[-1][1] if tiers else 0.0


def score_e2e(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Score E2E outcome dimension.

    When unverified lossy ops exist, score is based on verified-only cumulative
    speedup (excluding lossy ops that were never accuracy-gated by 5.1b).
    """
    campaign = snapshot.get("campaign") or {}
    status = campaign.get("status")
    cumulative = campaign.get("cumulative_speedup_vs_round1", 1.0)
    shipped = campaign.get("shipped_optimizations_count", 0)
    rounds = campaign.get("rounds", [])

    # Check accuracy verification — use verified-only speedup if unverified lossy exist
    acc_ver = snapshot.get("accuracy_verification") or {}
    has_unverified_lossy = acc_ver.get("has_unverified_lossy", False)
    verified_cumulative = acc_ver.get("verified_cumulative_speedup")
    verified_estimated = acc_ver.get("verified_cumulative_estimated", False)
    lossy_unverified_count = acc_ver.get("lossy_unverified", 0)

    # Diluted ships (status=PASS + diluted=true) contribute ~1.00x e2e; discount
    # them from the ship-bonus so they never earn the extra-ship diversity bonus.
    #
    # The `diluted` marker lives canonically on the TRACK object
    # (rounds[].parallel_tracks.tracks[].diluted), which parse_artifacts surfaces
    # into snapshot["tracks"]. Production does not reliably copy it onto the
    # campaign.shipped_optimizations[] entries, so counting only entry.diluted
    # would leave diluted_ship_count == 0 in the real pipeline. We therefore
    # take the UNION: a shipped op counts as diluted if its own entry carries
    # diluted:true OR it matches a diluted track by op_id. This keeps the
    # firewall live end-to-end without depending on a producer that may not
    # write the entry-level marker.
    shipped_opt_entries = campaign.get("shipped_optimization_ids", []) or []
    diluted_track_ids = {
        t.get("op_id")
        for t in (snapshot.get("tracks") or [])
        if isinstance(t, dict) and t.get("diluted") is True and t.get("op_id")
    }
    diluted_ship_count = sum(
        1 for e in shipped_opt_entries
        if isinstance(e, dict) and (
            e.get("diluted") is True or e.get("op_id") in diluted_track_ids
        )
    )

    # Effective speedup for scoring: verified-only when unverified lossy exist
    effective_cumulative = cumulative
    adjustment_applied = False
    if has_unverified_lossy and verified_cumulative is not None:
        effective_cumulative = verified_cumulative
        adjustment_applied = True

    # Effective shipped count: only count verified ops for the base-score/tier
    # gate. Diluted ships are REAL, correctness-gated ships whose measured
    # ~1.00x e2e legitimately enters cumulative accounting — they must NOT be
    # subtracted here (doing so would zero the base E2E score for a
    # diluted-only campaign, misreporting a genuine ship as "shipped nothing").
    # Only unverified-lossy ops (never accuracy-gated by 5.1b) drop out of the
    # base-score gate.
    effective_shipped = shipped
    if adjustment_applied:
        effective_shipped = max(0, shipped - lossy_unverified_count)

    if effective_shipped == 0:
        base_score = E2E_EXHAUSTED_NO_SHIP_SCORE
        tier = "exhausted_no_ship"
    elif status == "campaign_exhausted":
        raw_tier = _tier_score(effective_cumulative, E2E_TIERS)
        base_score = round(raw_tier * E2E_EXHAUSTED_DISCOUNT, 2)
        tier = f"exhausted_shipped (>={effective_cumulative:.2f}x × {E2E_EXHAUSTED_DISCOUNT})"
    else:
        base_score = _tier_score(effective_cumulative, E2E_TIERS)
        tier = f">={effective_cumulative:.2f}x"

    # Diversity ship-bonus: diluted ships ARE discounted OUT of the bonus (a
    # narrow decode-only dilution win must not harvest the "+0.5 per extra
    # real speedup" cap), but only out of the bonus — never out of the
    # base-score gate above.
    bonus_shipped = max(0, effective_shipped - diluted_ship_count)
    extra_ships = max(0, bonus_shipped - 1)
    bonus = min(extra_ships * E2E_SHIP_BONUS_PER_EXTRA, E2E_SHIP_BONUS_MAX)
    score = min(10.0, base_score + bonus)

    per_round = []
    for r in rounds:
        if isinstance(r, dict):
            per_round.append({
                "round_id": r.get("round_id"),
                "e2e_speedup": r.get("round_e2e_speedup"),
                "cumulative_after": r.get("cumulative_speedup_after"),
            })

    sub_scores: Dict[str, Any] = {
        "cumulative_speedup": cumulative,
        "shipped_count": shipped,
        "speedup_tier": tier,
        "base_score": base_score,
        "ship_bonus": bonus,
        "diluted_ship_count": diluted_ship_count,
        "per_round_speedups": per_round,
    }

    if adjustment_applied:
        sub_scores["accuracy_adjustment"] = {
            "applied": True,
            "raw_cumulative": cumulative,
            "verified_cumulative": verified_cumulative,
            "verified_estimated": verified_estimated,
            "lossy_unverified_excluded": lossy_unverified_count,
            "effective_shipped": effective_shipped,
        }

    return {
        "score": round(score, 2),
        "sub_scores": sub_scores,
    }


def score_gates(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Score gate pass rates dimension (pure pass/fail, enriched with gate-type sub-scores).

    Unverified lossy shipped ops are counted as expected 5.1b failures using
    the empirical failure rate (UNVERIFIED_LOSSY_EXPECTED_FAIL_RATE).
    """
    gates = snapshot.get("gates") or {}
    total = 0
    passed = 0

    # Phase 1
    p1 = gates.get("phase1_baseline", {})
    if p1.get("status") != "UNKNOWN":
        total += 1
        if p1.get("status") == "PASS":
            passed += 1

    # Per-track validation gates — also collect gate-type breakdown
    gate_type_stats: Dict[str, Dict[str, int]] = {}  # type -> {total, passed, failed}
    per_track_detail: List[Dict[str, Any]] = []

    for track in gates.get("validation_gates", []):
        total += 1
        track_passed = track.get("status") in ("PASSED", "GATED_PASS") or \
                       track.get("verdict") in ("PASS", "GATED_PASS")
        if track_passed:
            passed += 1

        gate_type = track.get("gate_failure_type")
        if gate_type:
            if gate_type not in gate_type_stats:
                gate_type_stats[gate_type] = {"total": 0, "passed": 0, "failed": 0}
            gate_type_stats[gate_type]["total"] += 1
            gate_type_stats[gate_type]["failed"] += 1
        elif track_passed:
            # Track passed — no failure type, count as passed in "all"
            pass

        per_track_detail.append({
            "track_id": track.get("track_id"),
            "passed": track_passed,
            "gate_failure_type": gate_type,
            "classification": track.get("classification"),
            "accuracy": track.get("accuracy"),
        })

    # Integration
    integration = gates.get("integration", {})
    if integration.get("status") not in (None, "pending"):
        total += 1
        if integration.get("status") in ("validated", "single_pass", "combined", "gated_pass"):
            passed += 1

    # Unverified lossy penalty: count unverified lossy shipped ops as expected failures
    acc_ver = snapshot.get("accuracy_verification") or {}
    lossy_unverified = acc_ver.get("lossy_unverified", 0)
    unverified_penalty_applied = False
    expected_failures = 0
    if lossy_unverified > 0:
        expected_failures = lossy_unverified  # count each as a failed gate
        total += lossy_unverified
        # Do not add to passed — they are expected failures
        unverified_penalty_applied = True

    rate = passed / total if total > 0 else 1.0
    score = _tier_score(rate, GATE_TIERS)

    # Accuracy gate analysis (from enriched parse_artifacts data)
    accuracy_analysis = gates.get("accuracy_gate_analysis")
    near_miss_count = 0
    if accuracy_analysis:
        near_miss_count = len(accuracy_analysis.get("near_misses", []))

    sub_scores: Dict[str, Any] = {
        "phase1_passed": p1.get("status") == "PASS",
        "pass_rate": round(rate, 3),
        "total_gates_checked": total,
        "total_passed": passed,
        "gate_type_breakdown": gate_type_stats if gate_type_stats else None,
        "accuracy_near_misses": near_miss_count,
        "per_track": per_track_detail if per_track_detail else None,
        "accuracy_gate_analysis": accuracy_analysis,
    }

    if unverified_penalty_applied:
        sub_scores["unverified_lossy_penalty"] = {
            "applied": True,
            "lossy_unverified_count": lossy_unverified,
            "expected_failures_added": expected_failures,
            "explanation": (
                f"{lossy_unverified} shipped lossy op(s) were never accuracy-verified "
                f"by Gate 5.1b. Each is counted as an expected gate failure."
            ),
        }

    return {
        "score": round(score, 2),
        "sub_scores": sub_scores,
    }


def score_debate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Score debate quality dimension."""
    debate = snapshot.get("debate") or {}
    total_proposals = debate.get("total_proposals", 0)

    # Grounding rate (0-3 points)
    grounding_rate = debate.get("grounding_rate")
    grounding_pts = (grounding_rate * 3.0) if grounding_rate is not None else 1.5

    # Micro-experiment rate (0-3 points)
    micro_rate = debate.get("micro_experiment_rate")
    micro_pts = (micro_rate * 3.0) if micro_rate is not None else 1.5

    # Candidate filtering correctness (0-2 points)
    # Check if rejected candidates scored lower than winners
    all_scores = debate.get("all_candidate_scores") or []
    if len(all_scores) >= 2:
        # Assume sorted by selection: winners first, then rejected
        # A crude heuristic: scores should be roughly descending
        filtering_correct = all(
            all_scores[i] >= all_scores[i + 1] - 0.5
            for i in range(min(2, len(all_scores) - 1))
        ) if len(all_scores) > 1 else True
        filtering_pts = 2.0 if filtering_correct else 0.5
    else:
        filtering_pts = 1.0  # Not enough data

    # Component diversity (0-2 points)
    # Check if proposals targeted different components
    rounds_data = debate.get("per_campaign_round", [])
    has_diversity = any(
        r.get("proposals_count", 0) >= 2 for r in rounds_data
    )
    diversity_pts = 2.0 if has_diversity else 1.0

    score = min(10.0, grounding_pts + micro_pts + filtering_pts + diversity_pts)

    return {
        "score": round(score, 2),
        "sub_scores": {
            "total_proposals": total_proposals,
            "grounding_rate": grounding_rate,
            "grounding_points": round(grounding_pts, 2),
            "micro_experiment_rate": micro_rate,
            "micro_experiment_points": round(micro_pts, 2),
            "filtering_correct": filtering_pts >= 1.5,
            "filtering_points": round(filtering_pts, 2),
            "component_diversity": has_diversity,
            "diversity_points": round(diversity_pts, 2),
        },
    }


def score_efficiency(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Score campaign efficiency dimension."""
    campaign = snapshot.get("campaign") or {}
    total_rounds = campaign.get("total_rounds", 1)
    status = campaign.get("status")
    rounds_data = campaign.get("rounds", [])

    # Rounds to completion
    rounds_score = ROUNDS_DEFAULT
    for max_rounds, s in ROUNDS_TIERS:
        if total_rounds <= max_rounds:
            rounds_score = s
            break

    # Failed implementation ratio
    total_tracks = 0
    failed_tracks = 0
    for r in rounds_data:
        if not isinstance(r, dict):
            continue
        total_tracks += r.get("candidates_selected", 0)
        failed_tracks += r.get("candidates_failed", 0)

    fail_ratio = failed_tracks / total_tracks if total_tracks > 0 else 0.0
    # Penalty: 0% failed = 0 penalty, 50%+ failed = -4
    fail_penalty = min(4.0, fail_ratio * 8.0)

    # Convergence bonus
    converged = status == "campaign_complete"
    convergence = CONVERGENCE_BONUS if converged else 0.0

    score = max(0.0, min(10.0, rounds_score - fail_penalty + convergence))

    return {
        "score": round(score, 2),
        "sub_scores": {
            "rounds_to_completion": total_rounds,
            "rounds_base_score": rounds_score,
            "total_impl_tracks": total_tracks,
            "failed_impl_tracks": failed_tracks,
            "failed_implementation_ratio": round(fail_ratio, 3),
            "fail_penalty": round(fail_penalty, 2),
            "campaign_converged": converged,
            "convergence_bonus": convergence,
        },
    }


def score_transcript(grading: Dict[str, Any]) -> Dict[str, Any]:
    """Score transcript quality from LLM grader output."""
    if not grading:
        return {"score": None, "sub_scores": {"graded": False}}

    delegation_enabled = grading.get("delegation_enabled", False)

    score = grading.get("score")
    if score is None:
        score = 10.0
        wasted = len(grading.get("wasted_retries", []))
        hallucinated = len(grading.get("hallucinated_data", []))
        off_track = len(grading.get("off_track_reasoning", []))

        if delegation_enabled:
            # Reduced maxes for original categories when delegation is enabled
            score -= min(2.5, wasted * 0.5)
            score -= min(3.5, hallucinated * 1.0)
            score -= min(2.5, off_track * 0.75)
        else:
            score -= min(3.0, wasted * 0.5)
            score -= min(4.0, hallucinated * 1.0)
            score -= min(3.0, off_track * 0.75)

        if delegation_enabled:
            # Delegation failures (-1.0 each, max -2.0)
            deleg_failures = len(grading.get("delegation_failures", []))
            score -= min(2.0, deleg_failures * 1.0)

            # Delegation efficiency issues (-0.25 each, max -1.0)
            deleg_efficiency = len(grading.get("delegation_efficiency_issues", []))
            score -= min(1.0, deleg_efficiency * 0.25)

            # Delegation utilization failures (-0.5 each, max -1.5)
            deleg_utilization = len(grading.get("delegation_utilization_failures", []))
            score -= min(1.5, deleg_utilization * 0.5)

            # Causality bonus (+0.5 per chain, max +1.5)
            causality = grading.get("delegation_causality_bonus") or {}
            bonus = causality.get("total_bonus", 0.0)
            if bonus == 0.0:
                chains = causality.get("chains", [])
                bonus = sum(c.get("bonus_awarded", 0.5) for c in chains)
            bonus = min(1.5, bonus)
            score += bonus

        score = max(0.0, min(10.0, score))

    sub_scores: Dict[str, Any] = {
        "graded": True,
        "wasted_retries": len(grading.get("wasted_retries", [])),
        "hallucinated_data_instances": len(grading.get("hallucinated_data", [])),
        "off_track_reasoning_instances": len(grading.get("off_track_reasoning", [])),
        "grader_notes": grading.get("notes"),
    }

    if delegation_enabled:
        causality = grading.get("delegation_causality_bonus") or {}
        bonus = causality.get("total_bonus", 0.0)
        if bonus == 0.0:
            chains = causality.get("chains", [])
            bonus = sum(c.get("bonus_awarded", 0.5) for c in chains)
        bonus = min(1.5, bonus)

        sub_scores["delegation_enabled"] = True
        sub_scores["delegation_causality_chains"] = len(causality.get("chains", []))
        sub_scores["delegation_causality_bonus"] = round(bonus, 2)
        sub_scores["delegation_failures"] = len(grading.get("delegation_failures", []))
        sub_scores["delegation_efficiency_issues"] = len(grading.get("delegation_efficiency_issues", []))
        sub_scores["delegation_utilization_failures"] = len(grading.get("delegation_utilization_failures", []))
        counterfactual = grading.get("counterfactual_assessment")
        if counterfactual:
            sub_scores["counterfactual_quality_delta"] = counterfactual.get("estimated_quality_delta")

    return {
        "score": round(score, 2),
        "sub_scores": sub_scores,
    }


# ---------------------------------------------------------------------------
# Composite Scorer
# ---------------------------------------------------------------------------

def _ensure_accuracy_verification(snapshot: Dict[str, Any]) -> None:
    """Backfill accuracy_verification into snapshot if missing.

    Old snapshots don't have this field. We reconstruct it from campaign
    shipped_optimization_ids using the same logic as parse_artifacts.py.
    """
    if snapshot.get("accuracy_verification") is not None:
        return

    campaign = snapshot.get("campaign") or {}
    shipped_ops = campaign.get("shipped_optimization_ids", [])
    tracks = snapshot.get("tracks") or []

    if not shipped_ops:
        return

    import re
    lossy_pattern = re.compile(
        r"fp8|int4|int8|w8a16|w4a16|quantiz|awq|gptq|squeezellm", re.IGNORECASE
    )

    ops_detail = []
    lossless_count = 0
    lossy_count = 0
    lossy_verified = 0
    lossy_unverified = 0

    for op in shipped_ops:
        op_id = op.get("op_id") if isinstance(op, dict) else str(op)
        op_round = op.get("round") if isinstance(op, dict) else None

        # Try tracks for classification
        classification = None
        for t in tracks:
            if t.get("op_id") == op_id and t.get("classification") in ("lossy", "lossless"):
                classification = t["classification"]
                break
        if classification is None:
            classification = "lossy" if lossy_pattern.search(op_id) else "lossless"

        # Check accuracy verification: lossy ops verified only if track shows
        # passing verdict with explicit 'lossy' classification
        verified = True
        if classification == "lossy":
            verified = False
            for t in tracks:
                if t.get("op_id") == op_id:
                    if t.get("verdict") in ("PASS", "GATED_PASS") and t.get("classification") == "lossy":
                        verified = True
                        break

        if classification == "lossless":
            lossless_count += 1
        else:
            lossy_count += 1
            if verified:
                lossy_verified += 1
            else:
                lossy_unverified += 1

        ops_detail.append({
            "op_id": op_id,
            "round": op_round,
            "classification": classification,
            "accuracy_verified": verified,
        })

    # Compute verified-only cumulative speedup
    total_cumulative = campaign.get("cumulative_speedup_vs_round1", 1.0)
    total_shipped = len(ops_detail)
    verified_count = sum(1 for op in ops_detail if op.get("accuracy_verified", False))
    ratio = verified_count / total_shipped if total_shipped > 0 else 1.0
    verified_cumulative = round(1.0 + (total_cumulative - 1.0) * ratio, 4)

    snapshot["accuracy_verification"] = {
        "shipped_ops_detail": ops_detail,
        "total_shipped": total_shipped,
        "lossless_shipped": lossless_count,
        "lossy_shipped": lossy_count,
        "lossy_verified": lossy_verified,
        "lossy_unverified": lossy_unverified,
        "has_unverified_lossy": lossy_unverified > 0,
        "verified_cumulative_speedup": verified_cumulative,
        "verified_cumulative_estimated": True,
    }


def compute_scorecard(
    snapshot: Dict[str, Any],
    transcript_grading: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the full scorecard from a parsed snapshot."""
    _ensure_accuracy_verification(snapshot)
    e2e = score_e2e(snapshot)
    gates = score_gates(snapshot)
    debate = score_debate(snapshot)
    efficiency = score_efficiency(snapshot)
    transcript = score_transcript(transcript_grading or {})

    has_transcript = transcript["score"] is not None
    weights = dict(WEIGHTS)
    if not has_transcript:
        # Redistribute transcript weight proportionally
        transcript_w = weights.pop("transcript_quality")
        total_remaining = sum(weights.values())
        for k in weights:
            weights[k] = weights[k] / total_remaining

    dimensions = {}
    overall = 0.0

    for dim_name, scorer_result, w_key in [
        ("e2e_outcome", e2e, "e2e_outcome"),
        ("gate_pass_rates", gates, "gate_pass_rates"),
        ("debate_quality", debate, "debate_quality"),
        ("campaign_efficiency", efficiency, "campaign_efficiency"),
        ("transcript_quality", transcript, "transcript_quality"),
    ]:
        w = weights.get(w_key, 0.0)
        s = scorer_result["score"]
        contrib = (s * w) if s is not None else 0.0
        overall += contrib
        dimensions[dim_name] = {
            "score": s,
            "weight": round(w, 3),
            "weighted_contribution": round(contrib, 3),
            "sub_scores": scorer_result["sub_scores"],
        }

    # Also compute score without transcript for comparison
    weights_no_t = dict(WEIGHTS)
    weights_no_t.pop("transcript_quality")
    total_no_t = sum(weights_no_t.values())
    overall_no_t = sum(
        dimensions[k]["score"] * (weights_no_t[k] / total_no_t)
        for k in weights_no_t
        if dimensions[k]["score"] is not None
    )

    # Raw metrics summary
    campaign = snapshot.get("campaign") or {}
    raw_metrics = {
        "cumulative_speedup_vs_round1": campaign.get("cumulative_speedup_vs_round1"),
        "shipped_optimizations": campaign.get("shipped_optimizations_count"),
        "total_rounds": campaign.get("total_rounds"),
        "total_proposals": (snapshot.get("debate") or {}).get("total_proposals"),
        "total_tracks": sum(
            r.get("candidates_selected", 0) for r in campaign.get("rounds", [])
            if isinstance(r, dict)
        ),
        "tracks_passed": sum(
            r.get("candidates_shipped", 0) for r in campaign.get("rounds", [])
            if isinstance(r, dict)
        ),
        "tracks_failed": sum(
            r.get("candidates_failed", 0) for r in campaign.get("rounds", [])
            if isinstance(r, dict)
        ),
        "mean_batch_speedup": (snapshot.get("e2e_results") or {}).get("mean_speedup"),
        "campaign_status": campaign.get("status"),
    }

    # Timing metrics (not scored — for cross-run comparison)
    stage_timestamps = snapshot.get("stage_timestamps")
    timing_metrics = None
    if stage_timestamps and stage_timestamps.get("stages"):
        per_stage = {}
        for s in stage_timestamps["stages"]:
            if s.get("duration_seconds") is not None:
                per_stage[s["stage"]] = s["duration_seconds"]
        timing_metrics = {
            "per_stage_seconds": per_stage,
            "total_tracked_seconds": stage_timestamps.get("total_tracked_seconds"),
        }

    # Delegation metrics (not scored — for cross-run comparison)
    delegation_data = snapshot.get("delegation")
    delegation_metrics = None
    if delegation_data and delegation_data.get("enabled"):
        delegation_metrics = {
            "enabled": True,
            "delegates_per_champion": delegation_data.get("delegates_per_champion"),
            "total_delegates_spawned": delegation_data.get("total_delegates_spawned"),
            "delegate_artifacts_count": delegation_data.get("delegate_artifacts_count"),
            "delegate_citation_rate": delegation_data.get("delegate_citation_rate"),
            "delegate_task_results": delegation_data.get("delegate_task_results"),
        }
    elif delegation_data:
        delegation_metrics = {"enabled": False}

    # Agent cost metrics (not scored — for cross-run comparison)
    agent_costs_data = snapshot.get("agent_costs")
    agent_cost_metrics = None
    if agent_costs_data:
        agent_cost_metrics = {
            "total_tokens": agent_costs_data.get("total_tokens"),
            "total_duration_ms": agent_costs_data.get("total_duration_ms"),
            "total_agent_invocations": agent_costs_data.get("total_agent_invocations"),
            "by_role": agent_costs_data.get("by_role"),
        }

    # Accuracy verification summary
    acc_ver = snapshot.get("accuracy_verification") or {}
    acc_ver_summary = None
    if acc_ver:
        acc_ver_summary = {
            "has_unverified_lossy": acc_ver.get("has_unverified_lossy", False),
            "lossy_shipped": acc_ver.get("lossy_shipped", 0),
            "lossy_unverified": acc_ver.get("lossy_unverified", 0),
            "lossy_verified": acc_ver.get("lossy_verified", 0),
            "lossless_shipped": acc_ver.get("lossless_shipped", 0),
            "verified_cumulative_speedup": acc_ver.get("verified_cumulative_speedup"),
            "verified_cumulative_estimated": acc_ver.get("verified_cumulative_estimated"),
            "ops_detail": acc_ver.get("shipped_ops_detail"),
        }

    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": round(overall, 2),
        "overall_score_without_transcript": round(overall_no_t, 2),
        "dimensions": dimensions,
        "raw_metrics": raw_metrics,
        "accuracy_verification": acc_ver_summary,
        "timing": timing_metrics,
        "delegation": delegation_metrics,
        "agent_costs": agent_cost_metrics,
        "metadata": {
            "artifact_dir": snapshot.get("artifact_dir"),
            "target": snapshot.get("target"),
        },
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(scorecard: Dict[str, Any], snapshot: Dict[str, Any]) -> str:
    """Generate a markdown report from the scorecard."""
    lines = []
    lines.append("# AMMO Eval Report")
    lines.append("")

    target = scorecard.get("metadata", {}).get("target") or {}
    lines.append(f"**Target**: {target.get('model_id', 'unknown')} on {target.get('hardware', 'unknown')} ({target.get('dtype', '?')}, tp={target.get('tp', '?')})")
    lines.append(f"**Scored**: {scorecard.get('scored_at', 'unknown')}")
    lines.append("")

    # Overall
    lines.append(f"## Overall Score: {scorecard.get('overall_score', '?')} / 10.0")
    lines.append("")

    # Dimensions table
    lines.append("| Dimension | Score | Weight | Contribution |")
    lines.append("|---|---|---|---|")
    for dim_name, dim_data in scorecard.get("dimensions", {}).items():
        label = dim_name.replace("_", " ").title()
        score = dim_data.get("score")
        score_str = f"{score}" if score is not None else "N/A"
        lines.append(f"| {label} | {score_str} | {dim_data.get('weight', 0):.0%} | {dim_data.get('weighted_contribution', 0):.2f} |")
    lines.append("")

    # E2E Outcome detail
    e2e_sub = scorecard.get("dimensions", {}).get("e2e_outcome", {}).get("sub_scores", {})
    lines.append("## E2E Outcome")
    lines.append("")
    lines.append(f"- Cumulative speedup: **{e2e_sub.get('cumulative_speedup', '?')}x** across {scorecard.get('raw_metrics', {}).get('total_rounds', '?')} round(s)")
    lines.append(f"- Shipped optimizations: {e2e_sub.get('shipped_count', 0)}")
    lines.append(f"- Campaign status: {scorecard.get('raw_metrics', {}).get('campaign_status', 'unknown')}")
    lines.append("")

    # Per-round breakdown
    per_round = e2e_sub.get("per_round_speedups", [])
    if per_round:
        lines.append("| Round | E2E Speedup | Cumulative |")
        lines.append("|---|---|---|")
        for r in per_round:
            lines.append(f"| {r.get('round_id', '?')} | {r.get('e2e_speedup') or '?'}x | {r.get('cumulative_after') or '?'}x |")
        lines.append("")

    # Accuracy verification adjustment
    acc_adj = e2e_sub.get("accuracy_adjustment")
    if acc_adj and acc_adj.get("applied"):
        lines.append("### Accuracy Verification Adjustment")
        lines.append("")
        lines.append(f"- Raw cumulative speedup: {acc_adj['raw_cumulative']:.3f}x")
        lines.append(f"- Verified-only cumulative: {acc_adj['verified_cumulative']:.3f}x{' (estimated)' if acc_adj.get('verified_estimated') else ''}")
        lines.append(f"- Lossy ops excluded (no 5.1b gate): {acc_adj['lossy_unverified_excluded']}")
        lines.append(f"- Effective shipped count: {acc_adj['effective_shipped']}")
        lines.append("")
        lines.append("> **Note**: This session ran without the GSM8K accuracy gate (5.1b).")
        lines.append("> Lossy optimizations (FP8/INT4) that shipped without accuracy verification")
        lines.append("> are excluded from the E2E score. The gate pass rate is also penalized.")
        lines.append("")

    # E2E per-batch results
    # TODO: For GATED_PASS tracks, add per-BS verdict columns to this table
    # (e.g., "Verdict" column showing PASS/NOISE/GATED_OFF per batch size,
    # and a "Gating" column indicating whether the optimization is active).
    # This would make it immediately visible which batch sizes benefit and
    # which are gated off. Deferred to a future iteration.
    e2e_results = snapshot.get("e2e_results")
    if e2e_results and e2e_results.get("per_batch"):
        lines.append("### Per-Batch E2E Results")
        lines.append("")
        lines.append("| Batch Size | Baseline (s) | Optimized (s) | Speedup | Improvement |")
        lines.append("|---|---|---|---|---|")
        for row in e2e_results["per_batch"]:
            bs = row.get("batch_size", "?")
            b = row.get("baseline_avg_s")
            o = row.get("optimized_avg_s")
            s = row.get("speedup")
            i = row.get("improvement_pct")
            lines.append(f"| {bs} | {f'{b:.4f}' if b else '?'} | {f'{o:.4f}' if o else '?'} | {f'{s:.3f}x' if s else '?'} | {f'{i:.1f}%' if i else '?'} |")
        lines.append("")

    # Gate Pass Rates detail
    gate_sub = scorecard.get("dimensions", {}).get("gate_pass_rates", {}).get("sub_scores", {})
    lines.append("## Gate Pass Rates")
    lines.append("")
    lines.append(f"- Pass rate: {gate_sub.get('pass_rate', '?'):.0%}" if isinstance(gate_sub.get('pass_rate'), float) else f"- Pass rate: {gate_sub.get('pass_rate', '?')}")
    lines.append(f"- Gates checked: {gate_sub.get('total_gates_checked', 0)}, passed: {gate_sub.get('total_passed', 0)}")
    lines.append("")

    # Gate type breakdown
    gate_type_bd = gate_sub.get("gate_type_breakdown")
    if gate_type_bd:
        lines.append("### Failures by Gate Type")
        lines.append("")
        lines.append("| Gate | Failures | Description |")
        lines.append("|---|---|---|")
        gate_descriptions = {
            "5.1a": "Kernel correctness (torch.allclose)",
            "5.1b": "GSM8K accuracy (opt >= baseline - tolerance)",
            "5.2": "Kernel speedup / kill gate",
            "5.3": "E2E speedup threshold",
        }
        for gt in ["5.1a", "5.1b", "5.2", "5.3"]:
            if gt in gate_type_bd:
                lines.append(f"| {gt} | {gate_type_bd[gt].get('failed', 0)} | {gate_descriptions.get(gt, '')} |")
        lines.append("")

    # Accuracy gate analysis (5.1b detail)
    acc_analysis = gate_sub.get("accuracy_gate_analysis")
    if acc_analysis:
        lines.append("### Accuracy Gate Analysis (5.1b)")
        lines.append("")
        all_margins = acc_analysis.get("all_accuracy_margins", [])
        near_misses = acc_analysis.get("near_misses", [])
        lines.append(f"- Total 5.1b failures: {acc_analysis.get('total_5_1b_failures', 0)}")
        lines.append(f"- Near-misses (gap <= 1.5%): {len(near_misses)}")
        lines.append("")
        if all_margins:
            lines.append("| Track | Classification | Opt Accuracy | Baseline | Gap | Questions Delta | Near-Miss |")
            lines.append("|---|---|---|---|---|---|---|")
            for m in all_margins:
                opt = m.get("opt_accuracy_pct")
                base = m.get("baseline_accuracy_pct")
                gap = m.get("accuracy_gap_pct")
                qd = m.get("questions_delta")
                nm = "YES" if m.get("near_miss") else "no"
                lines.append(
                    f"| {m.get('track_id', '?')} | {m.get('classification', '?')} "
                    f"| {f'{opt:.1f}%' if opt is not None else '?'} "
                    f"| {f'{base:.1f}%' if base is not None else '?'} "
                    f"| {f'{gap:.1f}%' if gap is not None else '?'} "
                    f"| {qd if qd is not None else '?'} "
                    f"| {nm} |"
                )
            lines.append("")

    # Per-track gate detail
    per_track_gates = gate_sub.get("per_track")
    if per_track_gates:
        lines.append("### Per-Track Gate Results")
        lines.append("")
        lines.append("| Track | Passed | Failure Type | Classification |")
        lines.append("|---|---|---|---|")
        for t in per_track_gates:
            passed_str = "PASS" if t.get("passed") else "FAIL"
            ft = t.get("gate_failure_type") or "-"
            cls = t.get("classification") or "-"
            lines.append(f"| {t.get('track_id', '?')} | {passed_str} | {ft} | {cls} |")
        lines.append("")

    # Unverified lossy penalty
    gate_sub_penalty = gate_sub.get("unverified_lossy_penalty")
    if gate_sub_penalty and gate_sub_penalty.get("applied"):
        lines.append("### Unverified Lossy Ops Penalty")
        lines.append("")
        lines.append(f"- Unverified lossy shipped ops: {gate_sub_penalty['lossy_unverified_count']}")
        lines.append(f"- Expected gate failures added: {gate_sub_penalty['expected_failures_added']}")
        lines.append("")

    # Debate Quality detail
    debate_sub = scorecard.get("dimensions", {}).get("debate_quality", {}).get("sub_scores", {})
    lines.append("## Debate Quality")
    lines.append("")
    lines.append(f"- Proposals: {debate_sub.get('total_proposals', 0)}")
    gr = debate_sub.get('grounding_rate')
    lines.append(f"- Grounding rate: {f'{gr:.0%}' if gr is not None else 'N/A'} ({debate_sub.get('grounding_points', 0)}/3 pts)")
    mr = debate_sub.get('micro_experiment_rate')
    lines.append(f"- Micro-experiment rate: {f'{mr:.0%}' if mr is not None else 'N/A'} ({debate_sub.get('micro_experiment_points', 0)}/3 pts)")
    lines.append(f"- Filtering correct: {debate_sub.get('filtering_correct', '?')} ({debate_sub.get('filtering_points', 0)}/2 pts)")
    lines.append(f"- Component diversity: {debate_sub.get('component_diversity', '?')} ({debate_sub.get('diversity_points', 0)}/2 pts)")
    lines.append("")

    # Campaign Efficiency detail
    eff_sub = scorecard.get("dimensions", {}).get("campaign_efficiency", {}).get("sub_scores", {})
    lines.append("## Campaign Efficiency")
    lines.append("")
    lines.append(f"- Rounds to completion: {eff_sub.get('rounds_to_completion', '?')}")
    lines.append(f"- Failed implementation ratio: {eff_sub.get('failed_implementation_ratio', 0):.0%} ({eff_sub.get('failed_impl_tracks', 0)}/{eff_sub.get('total_impl_tracks', 0)})")
    lines.append(f"- Campaign converged: {eff_sub.get('campaign_converged', '?')}")
    lines.append("")

    # Transcript Quality (if available)
    t_dim = scorecard.get("dimensions", {}).get("transcript_quality", {})
    if t_dim.get("score") is not None:
        t_sub = t_dim.get("sub_scores", {})
        lines.append("## Transcript Quality")
        lines.append("")
        lines.append(f"- Score: {t_dim.get('score')}/10")
        lines.append(f"- Wasted retries: {t_sub.get('wasted_retries', 0)}")
        lines.append(f"- Hallucinated data: {t_sub.get('hallucinated_data_instances', 0)}")
        lines.append(f"- Off-track reasoning: {t_sub.get('off_track_reasoning_instances', 0)}")
        if t_sub.get("delegation_enabled"):
            lines.append("")
            lines.append("### Delegation Analysis")
            lines.append("")
            bonus = t_sub.get("delegation_causality_bonus", 0)
            chains = t_sub.get("delegation_causality_chains", 0)
            lines.append(f"- Causality bonus: +{bonus} ({chains} verified chain(s))")
            failures = t_sub.get("delegation_failures", 0)
            if failures:
                lines.append(f"- Delegation failures (uncaught errors): {failures} (-{min(2.0, failures * 1.0):.1f} pts)")
            efficiency = t_sub.get("delegation_efficiency_issues", 0)
            if efficiency:
                lines.append(f"- Delegation efficiency issues: {efficiency} (-{min(1.0, efficiency * 0.25):.2f} pts)")
            utilization = t_sub.get("delegation_utilization_failures", 0)
            if utilization:
                lines.append(f"- Delegation utilization failures: {utilization} (-{min(1.5, utilization * 0.5):.1f} pts)")
            delta = t_sub.get("counterfactual_quality_delta")
            if delta:
                lines.append(f"- Counterfactual quality delta: {delta}")
        if t_sub.get("grader_notes"):
            lines.append(f"- Notes: {t_sub['grader_notes']}")
        lines.append("")

    # Timing (not scored — for cross-run comparison)
    timing = scorecard.get("timing")
    if timing and timing.get("per_stage_seconds"):
        lines.append("## Stage Timing")
        lines.append("")
        lines.append("| Stage | Duration |")
        lines.append("|---|---|")
        for stage, secs in timing["per_stage_seconds"].items():
            mins = secs / 60
            label = stage.replace("_", " ").title()
            lines.append(f"| {label} | {mins:.1f} min ({secs:.0f}s) |")
        total = timing.get("total_tracked_seconds")
        if total:
            lines.append(f"| **Total** | **{total / 60:.1f} min** |")
        lines.append("")

    # Delegation (not scored — for cross-run comparison)
    deleg = scorecard.get("delegation")
    if deleg and deleg.get("enabled"):
        lines.append("## Delegation")
        lines.append("")
        lines.append(f"- Delegates per champion: {deleg.get('delegates_per_champion', '?')}")
        lines.append(f"- Total delegates spawned: {deleg.get('total_delegates_spawned', 0)}")
        lines.append(f"- Delegate artifacts produced: {deleg.get('delegate_artifacts_count', 0)}")
        cr = deleg.get('delegate_citation_rate')
        lines.append(f"- Proposals citing delegate work: {f'{cr:.0%}' if cr is not None else 'N/A'}")
        tasks = deleg.get("delegate_task_results", {})
        lines.append(f"- Delegate tasks: {tasks.get('completed', 0)} completed, {tasks.get('failed', 0)} failed, {tasks.get('timed_out', 0)} timed out")
        lines.append("")
    elif deleg and not deleg.get("enabled"):
        lines.append("## Delegation")
        lines.append("")
        lines.append("- Delegation: disabled (baseline run)")
        lines.append("")

    # Agent Costs (not scored — for cross-run comparison)
    costs = scorecard.get("agent_costs")
    if costs and costs.get("total_tokens"):
        lines.append("## Agent Costs")
        lines.append("")
        total_tokens = costs["total_tokens"]
        total_ms = costs.get("total_duration_ms", 0)
        lines.append(f"- Total tokens: {total_tokens:,}")
        lines.append(f"- Total agent time: {total_ms / 1000:.0f}s ({total_ms / 60000:.1f} min)")
        lines.append(f"- Agent invocations: {costs.get('total_agent_invocations', 0)}")
        by_role = costs.get("by_role", {})
        if by_role:
            lines.append("")
            lines.append("| Role | Invocations | Tokens | Time |")
            lines.append("|---|---|---|---|")
            for role, data in sorted(by_role.items()):
                tok = data.get("total_tokens", 0)
                ms = data.get("total_duration_ms", 0)
                cnt = data.get("count", 0)
                lines.append(f"| {role} | {cnt} | {tok:,} | {ms / 1000:.0f}s |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot", type=str, required=True,
        help="Path to artifacts_snapshot.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for scorecard.json (default: alongside snapshot)",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output path for report.md (default: alongside snapshot)",
    )
    parser.add_argument(
        "--enrich-from", type=str, default=None,
        help="Path to transcript_grading.json to incorporate transcript scores",
    )

    args = parser.parse_args()
    snapshot_path = Path(args.snapshot).expanduser().resolve()

    if not snapshot_path.exists():
        print(f"ERROR: Snapshot file not found: {snapshot_path}", file=sys.stderr)
        return 1

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    # Load transcript grading if provided
    transcript_grading = None
    if args.enrich_from:
        tg_path = Path(args.enrich_from).expanduser().resolve()
        if tg_path.exists():
            transcript_grading = json.loads(tg_path.read_text(encoding="utf-8"))

    scorecard = compute_scorecard(snapshot, transcript_grading)
    report_md = generate_report(scorecard, snapshot)

    # Determine output paths
    out_dir = snapshot_path.parent
    scorecard_path = Path(args.output) if args.output else out_dir / "scorecard.json"
    report_path = Path(args.report) if args.report else out_dir / "report.md"

    scorecard_path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Wrote: {scorecard_path}", file=sys.stderr)
    print(f"Wrote: {report_path}", file=sys.stderr)
    print(f"Overall score: {scorecard['overall_score']}/10.0", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
