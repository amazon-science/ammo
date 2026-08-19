#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Parse completed AMMO campaign artifacts into a structured JSON snapshot.

Reads all files from a completed campaign artifact directory and emits a flat,
structured JSON snapshot suitable for scoring and archival.

All inputs are optional — missing files yield null fields, never errors.

Inputs (searched in artifact_dir):
  state.json, target.json,

  Legacy (flat) layout:
    constraints.md,
    investigation/bottleneck_analysis.md,
    debate/summary.md, debate/proposals/*.md,
    debate/campaign_round_*/summary.md,
    tracks/op*/validation_results.md,
    tracks/op*/e2e_latency/e2e_latency_results.json,
    tracks/op*/validation_summary.json

  Artifact Layout V2 (self-describing rounds/{N}/...):
    rounds/*/constraints.md,
    rounds/*/debate/summary.md, rounds/*/debate/proposals/*.md,
    rounds/*/debate/round_*/*.md,
    rounds/*/sweeps/opt/*/e2e_latency_results.json

  Both layouts are searched and unioned; either alone is sufficient.

Output:
  artifacts_snapshot.json (to --output or stdout)

Usage:
  python parse_artifacts.py --artifact-dir /path/to/artifacts
  python parse_artifacts.py --artifact-dir /path/to/artifacts --output snapshot.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None if not found."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _load_text(path: Path) -> Optional[str]:
    """Load text file, returning None if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


_LOSSY_OP_PATTERN = re.compile(
    r"fp8|int4|int8|w8a16|w4a16|quantiz|awq|gptq|squeezellm", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# v2 state accessors
# ---------------------------------------------------------------------------

def _rounds(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return state.get("campaign", {}).get("rounds", []) or []


def _current_round(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current round dict, or an empty dict if missing."""
    rounds = _rounds(state)
    cr = state.get("campaign", {}).get("current_round")
    if isinstance(cr, int) and 1 <= cr <= len(rounds):
        rnd = rounds[cr - 1]
        if isinstance(rnd, dict):
            return rnd
    # Fall back to the last round if current_round is out of bounds
    for rnd in reversed(rounds):
        if isinstance(rnd, dict):
            return rnd
    return {}


def _current_tracks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current round's parallel_tracks.tracks map."""
    tracks = _current_round(state).get("parallel_tracks", {}).get("tracks", {})
    return tracks if isinstance(tracks, dict) else {}


def _current_integration(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current round's integration block."""
    integration = _current_round(state).get("integration", {})
    return integration if isinstance(integration, dict) else {}


def _all_tracks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Merge every round's parallel_tracks.tracks into a single {op_id: track} map.

    Later rounds overwrite earlier ones on duplicate op_ids (rare but possible).
    """
    merged: Dict[str, Any] = {}
    for rnd in _rounds(state):
        if not isinstance(rnd, dict):
            continue
        tracks = rnd.get("parallel_tracks", {}).get("tracks", {})
        if isinstance(tracks, dict):
            for op_id, track in tracks.items():
                if isinstance(track, dict):
                    merged[op_id] = track
    return merged


def _classify_op_lossy(
    op_id: str,
    tracks: List[Dict[str, Any]],
    parallel_tracks: Dict[str, Any],
) -> str:
    """Classify a shipped op as 'lossy' or 'lossless'.

    Priority: parallel_tracks classification > parsed tracks > name pattern.
    """
    pt = parallel_tracks.get(op_id, {})
    if isinstance(pt, dict) and pt.get("classification") in ("lossy", "lossless"):
        return pt["classification"]
    for t in tracks:
        if t.get("op_id") == op_id and t.get("classification") in ("lossy", "lossless"):
            return t["classification"]
    return "lossy" if _LOSSY_OP_PATTERN.search(op_id) else "lossless"


def _is_op_accuracy_verified(
    op_id: str,
    classification: str,
    tracks: List[Dict[str, Any]],
    parallel_tracks: Dict[str, Any],
) -> bool:
    """Check if a shipped op was accuracy-verified via Gate 5.1b.

    Lossless ops are always considered verified (no accuracy risk).
    Lossy ops are verified only if they appear in parallel_tracks with a
    passing verdict AND an explicit 'lossy' classification (indicating the
    5.1b gate infrastructure was active).
    """
    if classification == "lossless":
        return True
    pt = parallel_tracks.get(op_id, {})
    if isinstance(pt, dict):
        if pt.get("verdict") in ("PASS", "GATED_PASS") and pt.get("classification") == "lossy":
            return True
    for t in tracks:
        if t.get("op_id") == op_id:
            if t.get("verdict") in ("PASS", "GATED_PASS") and t.get("classification") == "lossy":
                return True
    return False


def _parse_accuracy_verification(
    state: Dict[str, Any],
    tracks: List[Dict[str, Any]],
    campaign_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze accuracy verification coverage for shipped optimizations.

    Returns a block describing which shipped ops are lossy, which were
    accuracy-verified, and a verified-only cumulative speedup estimate.
    """
    parallel_tracks = _all_tracks(state)
    shipped_ops = state.get("campaign", {}).get("shipped_optimizations", [])
    rounds = campaign_data.get("rounds", [])
    total_cumulative = campaign_data.get("cumulative_speedup_vs_round1", 1.0)

    ops_detail: List[Dict[str, Any]] = []
    lossless_count = 0
    lossy_count = 0
    lossy_verified = 0
    lossy_unverified = 0

    for op in shipped_ops:
        op_id = op.get("op_id") if isinstance(op, dict) else str(op)
        op_round = op.get("round") if isinstance(op, dict) else None
        classification = _classify_op_lossy(op_id, tracks, parallel_tracks)
        verified = _is_op_accuracy_verified(op_id, classification, tracks, parallel_tracks)

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
    verified_cumulative, estimated = _compute_verified_cumulative(
        ops_detail, rounds, total_cumulative
    )

    return {
        "shipped_ops_detail": ops_detail,
        "total_shipped": len(ops_detail),
        "lossless_shipped": lossless_count,
        "lossy_shipped": lossy_count,
        "lossy_verified": lossy_verified,
        "lossy_unverified": lossy_unverified,
        "has_unverified_lossy": lossy_unverified > 0,
        "verified_cumulative_speedup": verified_cumulative,
        "verified_cumulative_estimated": estimated,
    }


def _compute_verified_cumulative(
    ops_detail: List[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    total_cumulative: float,
) -> tuple:
    """Compute cumulative speedup from verified ops only.

    Returns (speedup, estimated) where estimated=True if approximated
    because per-round decomposition was unavailable.
    """
    # Build round -> ops mapping
    round_ops: Dict[Any, List[Dict[str, Any]]] = {}
    for op in ops_detail:
        r = op.get("round")
        if r is not None:
            round_ops.setdefault(r, []).append(op)

    # Build round_id -> round_e2e_speedup mapping
    round_speedups: Dict[Any, float] = {}
    for r in rounds:
        rid = r.get("round_id")
        spd = r.get("round_e2e_speedup")
        if rid is not None and isinstance(spd, (int, float)):
            round_speedups[rid] = spd

    if round_speedups and round_ops:
        # Per-round decomposition available — multiply rounds where ALL ops are verified
        verified_cumulative = 1.0
        for rid, ops in round_ops.items():
            if all(op.get("accuracy_verified", False) for op in ops):
                spd = round_speedups.get(rid)
                if spd is not None:
                    verified_cumulative *= spd
        return (round(verified_cumulative, 4), False)

    # Fall back to linear approximation
    total = len(ops_detail)
    verified = sum(1 for op in ops_detail if op.get("accuracy_verified", False))
    if total == 0:
        return (1.0, True)
    ratio = verified / total
    verified_cumulative = 1.0 + (total_cumulative - 1.0) * ratio
    return (round(verified_cumulative, 4), True)


def _classify_gate_failure(fail_reason: Optional[str], verdict: Optional[str],
                           correctness: Optional[str]) -> Optional[str]:
    """Classify which gate type caused a track failure.

    Returns one of: "5.1a", "5.1b", "5.2", "5.3", or None if passed/unknown.
    """
    if verdict in ("PASS", "GATED_PASS"):
        return None
    if not fail_reason:
        if correctness == "FAIL":
            return "5.1a"
        return None

    fr_lower = fail_reason.lower()
    # Gate 5.1b: GSM8K accuracy gate
    if any(kw in fr_lower for kw in ["5.1b", "gsm8k", "opt_accuracy", "accuracy",
                                      "baseline_accuracy", "question"]):
        return "5.1b"
    # Gate 5.1a: kernel-level correctness (allclose)
    if any(kw in fr_lower for kw in ["5.1a", "allclose", "kernel correctness",
                                      "bit-exact", "numerical"]):
        return "5.1a"
    # Gate 5.2: kernel speedup / kill gate
    if any(kw in fr_lower for kw in ["5.2", "kill gate", "kernel speedup",
                                      "bw wall", "bandwidth"]):
        return "5.2"
    # Gate 5.3: E2E speedup
    if any(kw in fr_lower for kw in ["5.3", "e2e improvement", "e2e speedup"]):
        return "5.3"
    # Fallback: correctness field
    if correctness == "FAIL":
        return "5.1b" if "accuracy" in fr_lower or "%" in fail_reason else "5.1a"
    return None


def _extract_accuracy_numbers(fail_reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract accuracy percentages and question counts from fail_reason strings.

    Handles three generations of the Gate 5.1b verdict string:
      Pattern 0 (new, tolerance-bearing, post 2026-05-01):
        "Gate 5.1b v2 FAIL: opt_accuracy 86.0% (1134/1319) < threshold 88.0%
         (baseline 89.0% - 1.0pp tolerance)"
      Pattern 1 (legacy strict):
        "Gate 5.1b v2 FAIL: opt_accuracy 89.0% (178/1319) < baseline_accuracy 89.5% (179/1319)"
      Pattern 2 (oldest free-form):
        "Gate 5.1b FAIL: 88.5% vs 89.0% (-1 question)"

    Pattern 0 is preferred when present; its `baseline_correct` is derived
    (round(opt_total * baseline_pct/100)) and flagged via
    `baseline_correct_approximate=True` because the new verdict string omits
    the exact A/B fraction for the baseline. Pattern 1 carries the exact
    baseline count and sets the flag to False.
    """
    if not fail_reason:
        return None

    result: Dict[str, Any] = {}

    # Pattern 0: "opt_accuracy X% (A/B) [<>]=? threshold Y% (baseline Z% - Tpp tolerance)"
    # Preferred — introduced alongside --correctness-tolerance-pct.
    m0 = re.search(
        r"opt_accuracy\s+(\d+\.?\d*)%\s*\((\d+)/(\d+)\)\s*[<>]=?\s*threshold\s+(\d+\.?\d*)%\s*\(baseline\s+(\d+\.?\d*)%\s*-\s*(\d+\.?\d*)pp\s*tolerance\)",
        fail_reason,
    )
    if m0:
        result["opt_accuracy_pct"] = float(m0.group(1))
        result["opt_correct"] = int(m0.group(2))
        result["opt_total"] = int(m0.group(3))
        result["threshold_pct"] = float(m0.group(4))
        result["baseline_accuracy_pct"] = float(m0.group(5))
        result["tolerance_pct"] = float(m0.group(6))
        # accuracy_gap is baseline - opt (same semantics as Pattern 1) —
        # NOT threshold-derived, so near-miss math stays consistent across
        # tolerance settings.
        result["accuracy_gap_pct"] = round(
            result["baseline_accuracy_pct"] - result["opt_accuracy_pct"], 2
        )
        # Pattern 0's verdict string omits the baseline A/B count; derive it.
        # baseline_correct is approximated via rounding.
        result["baseline_total"] = result["opt_total"]
        result["baseline_correct"] = round(
            result["opt_total"] * result["baseline_accuracy_pct"] / 100
        )
        result["baseline_correct_approximate"] = True
        result["questions_delta"] = result["opt_correct"] - result["baseline_correct"]
        return result

    # Pattern 1: "opt_accuracy X% (A/B) < baseline_accuracy Y% (C/D)"
    m = re.search(
        r"opt_accuracy\s+(\d+\.?\d*)%\s*\((\d+)/(\d+)\)\s*[<>]\s*baseline_accuracy\s+(\d+\.?\d*)%\s*\((\d+)/(\d+)\)",
        fail_reason,
    )
    if m:
        result["opt_accuracy_pct"] = float(m.group(1))
        result["opt_correct"] = int(m.group(2))
        result["opt_total"] = int(m.group(3))
        result["baseline_accuracy_pct"] = float(m.group(4))
        result["baseline_correct"] = int(m.group(5))
        result["baseline_total"] = int(m.group(6))
        result["accuracy_gap_pct"] = round(result["baseline_accuracy_pct"] - result["opt_accuracy_pct"], 2)
        result["questions_delta"] = result["opt_correct"] - result["baseline_correct"]
        result["baseline_correct_approximate"] = False
        return result

    # Pattern 2: "X% vs Y% (-Nq)" or "X% vs Y% (-N question)"
    m = re.search(r"(\d+\.?\d*)%\s*vs\s*(\d+\.?\d*)%", fail_reason)
    if m:
        result["opt_accuracy_pct"] = float(m.group(1))
        result["baseline_accuracy_pct"] = float(m.group(2))
        result["accuracy_gap_pct"] = round(result["baseline_accuracy_pct"] - result["opt_accuracy_pct"], 2)
        # Try to extract question delta
        qm = re.search(r"[(-](\d+)\s*(?:net\s+)?question", fail_reason)
        if qm:
            result["questions_delta"] = -int(qm.group(1))
        return result

    return None


# ---------------------------------------------------------------------------
# Target & Environment
# ---------------------------------------------------------------------------

def _parse_target(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    target = _load_json(artifact_dir / "target.json")
    if target is None:
        # Fall back to state.json target
        state = _load_json(artifact_dir / "state.json")
        if state and "target" in state:
            t = state["target"]
            return {
                "model_id": t.get("model_id"),
                "hardware": t.get("hardware"),
                "dtype": t.get("dtype"),
                "tp": t.get("tp"),
            }
        return None
    t = target.get("target", {})
    return {
        "model_id": t.get("model_id"),
        "hardware": _safe_get(_load_json(artifact_dir / "state.json"), "target", "hardware"),
        "dtype": t.get("dtype"),
        "tp": t.get("tp"),
    }


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def _parse_campaign(state: Dict[str, Any]) -> Dict[str, Any]:
    campaign = state.get("campaign", {})
    rounds_data = campaign.get("rounds", [])

    rounds = []
    for r in rounds_data:
        if not isinstance(r, dict):
            continue
        tracks = r.get("parallel_tracks", {}).get("tracks", {})
        if not isinstance(tracks, dict):
            tracks = {}
        shipped = r.get("shipped", [])
        failed_count = sum(
            1 for v in tracks.values()
            if isinstance(v, dict) and v.get("status") in ("FAIL", "FAILED")
        )
        integration = r.get("integration", {}) if isinstance(r.get("integration"), dict) else {}

        rounds.append({
            "round_id": r.get("round_id"),
            "top_bottleneck_share_pct": r.get("bottleneck_mining", {}).get("top_bottleneck_share_pct"),
            "candidates_proposed": len(tracks),
            "candidates_selected": _candidate_selection_count(integration, tracks),
            "candidates_shipped": len(shipped),
            "candidates_failed": failed_count,
            "round_e2e_speedup": _best_e2e_from_tracks(tracks),
            "cumulative_speedup_after": r.get("cumulative_speedup_after"),
        })

    shipped_opts = campaign.get("shipped_optimizations", [])
    return {
        "status": campaign.get("status"),
        "total_rounds": campaign.get("current_round", len(rounds)),
        "cumulative_speedup_vs_round1": campaign.get("cumulative_speedup_vs_round1", 1.0),
        "shipped_optimizations_count": len(shipped_opts),
        "shipped_optimization_ids": shipped_opts,
        "rounds": rounds,
    }


def _candidate_selection_count(integration: Dict[str, Any], tracks: Dict[str, Any]) -> int:
    selected = integration.get("selected_candidates")
    if selected is None:
        selected = integration.get("passing_candidates")
    if isinstance(selected, list):
        return len(selected)
    return len(tracks)


def _best_e2e_from_tracks(tracks: Dict[str, Any]) -> Optional[float]:
    """Extract the best e2e_speedup from a round's parallel_tracks.tracks dict."""
    best = None
    for v in tracks.values():
        if not isinstance(v, dict):
            continue
        status = v.get("status")
        verdict = v.get("verdict")
        passing = status in ("PASS", "GATED_PASS", "PASSED") or verdict in ("PASS", "GATED_PASS")
        if not passing:
            continue
        speedup = v.get("e2e_speedup")
        if isinstance(speedup, (int, float)):
            if best is None or speedup > best:
                best = speedup
    return best


# ---------------------------------------------------------------------------
# E2E Results
# ---------------------------------------------------------------------------

def _parse_e2e_results(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    """Find and parse e2e_latency_results.json from tracks or root.

    Search order (first non-empty wins):
      1. v2 layout — rounds/*/sweeps/opt/*/e2e_latency_results.json
      2. legacy   — tracks/*/e2e_latency/e2e_latency_results.json
      3. legacy   — e2e_latency/e2e_latency_results.json (root)
    """
    # v2 self-describing layout: rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json
    e2e_files = list(artifact_dir.glob("rounds/*/sweeps/opt/*/e2e_latency_results.json"))
    if not e2e_files:
        e2e_files = list(artifact_dir.glob("tracks/*/e2e_latency/e2e_latency_results.json"))
    if not e2e_files:
        e2e_files = list(artifact_dir.glob("e2e_latency/e2e_latency_results.json"))
    if not e2e_files:
        return None

    # Aggregate across all tracks (take the best shipped track's results)
    all_results = []
    for f in e2e_files:
        data = _load_json(f)
        if data and isinstance(data.get("results"), list):
            all_results.append(data)

    if not all_results:
        return None

    # Use the first file's results as representative (typically the shipped track)
    data = all_results[0]
    results = data.get("results", [])

    per_batch = []
    speedups = []
    for row in results:
        if not isinstance(row, dict):
            continue
        speedup = row.get("speedup")
        per_batch.append({
            "batch_size": row.get("batch_size"),
            "input_len": row.get("input_len"),
            "output_len": row.get("output_len"),
            "baseline_avg_s": _safe_get(row, data.get("bench", {}).get("baseline_label", "baseline"), "avg_s"),
            "optimized_avg_s": _safe_get(row, data.get("bench", {}).get("opt_label", "opt"), "avg_s"),
            "speedup": speedup,
            "improvement_pct": row.get("improvement_pct"),
        })
        if isinstance(speedup, (int, float)):
            speedups.append(speedup)

    return {
        "batch_sizes": [r["batch_size"] for r in per_batch if r.get("batch_size") is not None],
        "per_batch": per_batch,
        "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _parse_gates(artifact_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse gate pass/fail information from state.json and validation files."""
    # Phase 1 baseline gate — determined by constraints.md existence.
    # Both legacy (constraints.md at root) and v2 (rounds/*/constraints.md) count.
    has_constraints = (
        (artifact_dir / "constraints.md").exists()
        or any(artifact_dir.glob("rounds/*/constraints.md"))
    )
    phase1 = {
        "status": "PASS" if has_constraints else "UNKNOWN",
    }

    # Per-track validation gates (current round)
    tracks_state = _current_tracks(state)
    track_gates = []
    gate_type_counts: Dict[str, int] = {}  # e.g., {"5.1b": 3, "5.2": 1}
    accuracy_margins: List[Dict[str, Any]] = []  # per-track accuracy data

    for track_id, track_data in tracks_state.items():
        if not isinstance(track_data, dict):
            continue

        fail_reason = track_data.get("fail_reason")
        verdict = track_data.get("verdict")
        correctness = track_data.get("correctness")
        classification = track_data.get("classification")

        gate_type = _classify_gate_failure(fail_reason, verdict, correctness)
        accuracy_data = _extract_accuracy_numbers(fail_reason) if gate_type == "5.1b" else None

        gate_entry = {
            "track_id": track_id,
            "status": track_data.get("status", "UNKNOWN"),
            "verdict": verdict,
            "correctness": correctness,
            "classification": classification,
            "kernel_speedup": track_data.get("kernel_speedup"),
            "e2e_speedup": track_data.get("e2e_speedup"),
            "gating": track_data.get("gating"),
            "gate_failure_type": gate_type,
            "fail_reason": fail_reason,
            "accuracy": accuracy_data,
        }
        track_gates.append(gate_entry)

        # Aggregate gate type counts
        if gate_type:
            gate_type_counts[gate_type] = gate_type_counts.get(gate_type, 0) + 1

        # Collect accuracy margins for 5.1b failures
        if accuracy_data:
            accuracy_margins.append({
                "track_id": track_id,
                "classification": classification,
                "opt_accuracy_pct": accuracy_data.get("opt_accuracy_pct"),
                "baseline_accuracy_pct": accuracy_data.get("baseline_accuracy_pct"),
                "accuracy_gap_pct": accuracy_data.get("accuracy_gap_pct"),
                "questions_delta": accuracy_data.get("questions_delta"),
                # Near-miss threshold is constant at 1.5pp: slightly above the
                # 1.0pp default tolerance so runs that *just* missed the gate
                # (e.g. 1.1pp gap at tol=1.0) still show up for review, but
                # genuinely divergent runs don't drown the signal.
                "near_miss": (accuracy_data.get("accuracy_gap_pct") or 999) <= 1.5,
            })

    # Integration gate (current round)
    integration = _current_integration(state)

    return {
        "phase1_baseline": phase1,
        "validation_gates": track_gates,
        "integration": {
            "status": integration.get("status", "pending"),
            "final_decision": _safe_get(integration, "final_decision", "action"),
        },
        "gate_type_breakdown": gate_type_counts,
        "accuracy_gate_analysis": {
            "total_5_1b_failures": gate_type_counts.get("5.1b", 0),
            "near_misses": [m for m in accuracy_margins if m.get("near_miss")],
            "all_accuracy_margins": accuracy_margins,
        } if accuracy_margins else None,
    }


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

def _parse_debate(artifact_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse debate quality metrics from debate artifacts and state."""
    campaign = state.get("campaign", {})
    total_rounds = campaign.get("current_round", 1)
    rounds = _rounds(state)

    def _round_debate(round_idx: int) -> Dict[str, Any]:
        """Return the debate block for 1-based round_idx, or an empty dict."""
        if 1 <= round_idx <= len(rounds):
            rnd = rounds[round_idx - 1]
            if isinstance(rnd, dict):
                debate = rnd.get("debate", {})
                if isinstance(debate, dict):
                    return debate
        return {}

    def _proposal_dirs_for_round(round_idx: int) -> List[Path]:
        """Return all directories that may hold proposals for this round.

        Combines the legacy flat layout with the v2 self-describing layout
        (rounds/{N}/debate/proposals).
        """
        dirs: List[Path] = []
        # v2 self-describing layout
        dirs.append(artifact_dir / "rounds" / str(round_idx) / "debate" / "proposals")
        # legacy flat layouts
        if round_idx == 1:
            dirs.append(artifact_dir / "debate" / "proposals")
        else:
            dirs.append(artifact_dir / "debate" / f"campaign_round_{round_idx}" / "proposals")
        return dirs

    def _debate_base_dirs_for_round(round_idx: int) -> List[Path]:
        """Return all directories that may hold debate round_* subdirs / summary.md."""
        dirs: List[Path] = []
        # v2 self-describing layout
        dirs.append(artifact_dir / "rounds" / str(round_idx) / "debate")
        # legacy flat layouts
        if round_idx == 1:
            dirs.append(artifact_dir / "debate")
        else:
            dirs.append(artifact_dir / "debate" / f"campaign_round_{round_idx}")
        return dirs

    # Count proposals (union v2 + legacy across all rounds).
    proposal_files: List[Path] = []
    per_round_proposal_files: Dict[int, List[Path]] = {}
    for round_idx in range(1, total_rounds + 1):
        rfiles: List[Path] = []
        for pdir in _proposal_dirs_for_round(round_idx):
            if pdir.exists():
                rfiles.extend(pdir.glob("*.md"))
        per_round_proposal_files[round_idx] = rfiles
        proposal_files.extend(rfiles)

    total_proposals = len(proposal_files)

    # Check micro-experiment backing and grounding
    proposals_with_micro = 0
    proposals_with_grounding = 0
    for pf in proposal_files:
        content = _load_text(pf)
        if content is None:
            continue
        # Check for micro-experiment sections
        if re.search(r"micro.?experiment|prototype|benchmark|roofline|ncu", content, re.IGNORECASE):
            proposals_with_micro += 1
        # Check for grounded data references
        if re.search(r"bottleneck_analysis|nsys|profil|µs|microsec|\d+\.?\d*\s*(µs|us|ms)", content, re.IGNORECASE):
            proposals_with_grounding += 1

    # Parse candidate scores: prefer state.json structured contract
    # (debate.selected_candidates[].score_breakdown.weighted_total) for
    # post-refactor artifacts, fall back to summary.md regex for legacy ones.
    all_scores = []
    per_campaign_round = []

    for round_idx in range(1, total_rounds + 1):
        scores = _scores_from_state(_round_debate(round_idx))
        if not scores:
            # Try every base dir (v2 + legacy) for this round; first hit wins.
            for base in _debate_base_dirs_for_round(round_idx):
                scores = _extract_candidate_scores(base / "summary.md")
                if scores:
                    break
        if scores:
            all_scores.extend(scores)

        # Count debate round_* subdirs across v2 and legacy bases (deduped by name).
        debate_round_names = set()
        for base in _debate_base_dirs_for_round(round_idx):
            if base.exists():
                for d in base.iterdir():
                    if d.is_dir() and re.match(r"round_\d+", d.name):
                        debate_round_names.add(d.name)

        round_debate = _round_debate(round_idx)
        per_campaign_round.append({
            "campaign_round": round_idx,
            "champion_count": len(round_debate.get("candidates", []) or []),
            "proposals_count": len(per_round_proposal_files.get(round_idx, [])),
            "debate_rounds": len(debate_round_names),
            "winners_selected": len(round_debate.get("selected_winners", []) or []),
            "candidate_scores": scores if scores else None,
        })

    return {
        "campaign_rounds_with_debate": len([r for r in per_campaign_round if r.get("proposals_count", 0) > 0]),
        "total_proposals": total_proposals,
        "proposals_with_micro_experiments": proposals_with_micro,
        "proposals_with_grounded_data": proposals_with_grounding,
        "micro_experiment_rate": proposals_with_micro / total_proposals if total_proposals > 0 else None,
        "grounding_rate": proposals_with_grounding / total_proposals if total_proposals > 0 else None,
        "total_debate_rounds_across_campaign": sum(r.get("debate_rounds", 0) for r in per_campaign_round),
        "all_candidate_scores": all_scores if all_scores else None,
        "per_campaign_round": per_campaign_round,
    }


def _scores_from_state(round_debate: Dict[str, Any]) -> List[float]:
    """Read per-winner weighted_total from state.json's structured debate contract.

    Returns the scores of the selected winners for this round (typically 2-4
    per round). Empty list if the round predates the structured contract
    (state.schema.json PR #1c) — caller falls back to summary.md regex.
    """
    if not isinstance(round_debate, dict):
        return []
    selected = round_debate.get("selected_candidates") or []
    scores: List[float] = []
    for cand in selected:
        if not isinstance(cand, dict):
            continue
        breakdown = cand.get("score_breakdown") or {}
        total = breakdown.get("weighted_total")
        if isinstance(total, (int, float)):
            scores.append(float(total))
    return scores


def _extract_candidate_scores(summary_path: Path) -> List[float]:
    """Regex-extract per-candidate final scores from debate summary.md tables."""
    content = _load_text(summary_path)
    if content is None:
        return []

    scores = []
    # Match patterns like "**8.4/10**" or "8.4/10" or "| 8.4/10 |" in table rows
    # Also match patterns like "Total (weighted): 8.4/10"
    for match in re.finditer(r"(\d+\.?\d*)\s*/\s*10", content):
        try:
            score = float(match.group(1))
            if 0 <= score <= 10:
                scores.append(score)
        except ValueError:
            continue

    # Deduplicate: the summary table often has the same score multiple times
    # (per-criterion + final). Take the scores from the Final Score column.
    # Heuristic: look for "Final Score" or "Total" rows
    final_scores = []
    for line in content.split("\n"):
        if re.search(r"(Final\s+Score|Total|SELECTED|REJECTED)", line, re.IGNORECASE):
            line_scores = []
            for m in re.finditer(r"\*?\*?(\d+\.?\d*)\s*/\s*10\*?\*?", line):
                try:
                    s = float(m.group(1))
                    if 0 <= s <= 10:
                        line_scores.append(s)
                except ValueError:
                    continue
            if line_scores:
                final_scores.append(line_scores[-1])  # Last score on line is usually the total

    return final_scores if final_scores else scores


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def _parse_tracks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse per-track results from every round's parallel_tracks.tracks."""
    tracks = []
    for rnd in _rounds(state):
        if not isinstance(rnd, dict):
            continue
        round_id = rnd.get("round_id")
        tracks_state = rnd.get("parallel_tracks", {}).get("tracks", {})
        if not isinstance(tracks_state, dict):
            continue
        for track_id, track_data in tracks_state.items():
            if not isinstance(track_data, dict):
                continue
            tracks.append({
                "op_id": track_id,
                "campaign_round": round_id,
                "status": track_data.get("status", "UNKNOWN"),
                "verdict": track_data.get("verdict"),
                "correctness": track_data.get("correctness"),
                "classification": track_data.get("classification"),
                "diluted": track_data.get("diluted", False),
                "kernel_speedup": track_data.get("kernel_speedup"),
                "e2e_speedup": track_data.get("e2e_speedup"),
                "fail_reason": track_data.get("fail_reason"),
            })

    return tracks


# ---------------------------------------------------------------------------
# Stage Timestamps
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp, returning None on failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


# Map the flat output stage name to the round sub-object key in v2 state.json.
_STAGE_KEY_MAP: List[tuple] = [
    ("1_baseline", "baseline"),
    ("2_bottleneck_mining", "bottleneck_mining"),
    ("3_debate", "debate"),
    ("4_5_parallel_tracks", "parallel_tracks"),
    ("6_integration", "integration"),
    ("7_campaign_eval", "campaign_eval"),
]


def _parse_stage_timestamps(
    state: Dict[str, Any],
    session_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract stage timing data.

    If session_data is provided, use its stage_timestamps (ground-truth from session logs).
    Otherwise derive from per-round stage sub-objects (baseline, bottleneck_mining, ...)
    carried on each v2 round.
    """
    if session_data is not None:
        sd_ts = session_data.get("stage_timestamps")
        if sd_ts and isinstance(sd_ts, dict):
            # Convert session_data stage_timestamps to the standard output format
            # session_data has per-round keys like "round_1", "round_2"
            # We convert to the flat stages list for compatibility
            stages = []
            total_seconds = 0.0
            # Use round_1 as the primary source for backward-compatible stage list
            round_1 = sd_ts.get("round_1", {})
            stage_map = [
                ("1_baseline", "stage_1_baseline_start", "stage_2_bottleneck_end"),
                ("3_debate", "stage_3_debate_start", "stage_3_debate_end"),
                ("4_5_parallel_tracks", "stage_4_5_impl_start", "stage_4_5_impl_end"),
                ("6_integration", "stage_6_7_eval_start", "stage_6_7_eval_end"),
            ]
            for stage_name, start_key, end_key in stage_map:
                started_at = round_1.get(start_key)
                completed_at = round_1.get(end_key) if end_key else None
                started = _parse_iso(started_at)
                completed = _parse_iso(completed_at)
                duration_s = None
                if started and completed:
                    duration_s = (completed - started).total_seconds()
                    total_seconds += duration_s
                stages.append({
                    "stage": stage_name,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": round(duration_s, 1) if duration_s is not None else None,
                })
            return {
                "stages": stages,
                "total_tracked_seconds": round(total_seconds, 1) if total_seconds > 0 else None,
                "per_round": sd_ts,
                "source": "session_logs",
            }

    # Fall back to per-round stage sub-objects in v2 state.json.
    rounds = _rounds(state)
    if not rounds:
        return None

    def _stage_entry(rnd: Dict[str, Any], round_key: str) -> Dict[str, Any]:
        sub = rnd.get(round_key, {}) if isinstance(rnd, dict) else {}
        return sub if isinstance(sub, dict) else {}

    # Flat "stages" list: use the current round (fall back to the first populated round).
    current = _current_round(state) or (rounds[0] if isinstance(rounds[0], dict) else {})

    stages = []
    total_seconds = 0.0
    for stage_name, round_key in _STAGE_KEY_MAP:
        entry = _stage_entry(current, round_key)
        started_at = entry.get("started_at")
        completed_at = entry.get("completed_at")
        started = _parse_iso(started_at)
        completed = _parse_iso(completed_at)
        duration_s = None
        if started and completed:
            duration_s = (completed - started).total_seconds()
            total_seconds += duration_s
        stages.append({
            "stage": stage_name,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": round(duration_s, 1) if duration_s is not None else None,
        })

    # Per-round breakdown: {round_1: {stage: {started_at, completed_at, duration_seconds}}}
    per_round: Dict[str, Dict[str, Any]] = {}
    any_timestamps = False
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        rid = rnd.get("round_id")
        round_key = f"round_{rid}" if rid is not None else f"round_{len(per_round) + 1}"
        round_stages: Dict[str, Any] = {}
        for stage_name, sub_key in _STAGE_KEY_MAP:
            entry = _stage_entry(rnd, sub_key)
            started_at = entry.get("started_at")
            completed_at = entry.get("completed_at")
            if started_at or completed_at:
                any_timestamps = True
            started = _parse_iso(started_at)
            completed = _parse_iso(completed_at)
            duration_s = None
            if started and completed:
                duration_s = (completed - started).total_seconds()
            round_stages[stage_name] = {
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": round(duration_s, 1) if duration_s is not None else None,
            }
        per_round[round_key] = round_stages

    if not any_timestamps:
        return None

    return {
        "stages": stages,
        "total_tracked_seconds": round(total_seconds, 1) if total_seconds > 0 else None,
        "per_round": per_round,
        "source": "state_json",
    }


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

def _parse_delegation(artifact_dir: Path, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse delegation metrics from debate artifacts and state.json.

    Aggregates `debate.delegation` across every round — delegation is enabled
    if any round's debate carries an `enabled: true` flag.
    """
    delegation: Dict[str, Any] = {}
    enabled = False
    for rnd in _rounds(state):
        if not isinstance(rnd, dict):
            continue
        debate_state = rnd.get("debate", {}) if isinstance(rnd.get("debate"), dict) else {}
        d = debate_state.get("delegation", {}) if isinstance(debate_state.get("delegation"), dict) else {}
        if d.get("enabled"):
            enabled = True
        # Later rounds overwrite earlier ones on duplicate keys (last-wins for config;
        # mappings and results are round-scoped so merge instead).
        for key, val in d.items():
            if key in ("champion_delegate_mapping", "delegate_results") and isinstance(val, dict):
                merged = delegation.get(key, {})
                if not isinstance(merged, dict):
                    merged = {}
                merged.update(val)
                delegation[key] = merged
            else:
                delegation[key] = val

    if not enabled:
        return {"enabled": False}

    mapping = delegation.get("champion_delegate_mapping", {})
    delegate_results = delegation.get("delegate_results", {})

    # Count delegate work artifacts
    delegate_work_dir = artifact_dir / "debate" / "delegate_work"
    delegate_artifacts = list(delegate_work_dir.glob("*.md")) if delegate_work_dir.exists() else []
    delegate_scripts = list(delegate_work_dir.glob("*.py")) if delegate_work_dir.exists() else []

    # Check if proposals cite delegate work
    proposals_dir = artifact_dir / "debate" / "proposals"
    proposal_files = list(proposals_dir.glob("*.md")) if proposals_dir.exists() else []
    proposals_citing_delegates = 0
    for pf in proposal_files:
        content = _load_text(pf)
        if content and re.search(r"delegate[-_]?\d|delegate.work|Source:\s*delegate", content, re.IGNORECASE):
            proposals_citing_delegates += 1

    total_proposals = len(proposal_files)
    citation_rate = proposals_citing_delegates / total_proposals if total_proposals > 0 else None

    # Count delegate results by status
    completed = sum(1 for v in delegate_results.values()
                    if isinstance(v, dict) and v.get("status") == "completed")
    failed = sum(1 for v in delegate_results.values()
                 if isinstance(v, dict) and v.get("status") == "failed")
    timed_out = sum(1 for v in delegate_results.values()
                    if isinstance(v, dict) and v.get("status") == "timeout")

    total_delegates = sum(len(v) for v in mapping.values() if isinstance(v, list))

    return {
        "enabled": True,
        "delegates_per_champion": delegation.get("delegates_per_champion", 1),
        "total_delegates_spawned": total_delegates,
        "champion_delegate_mapping": mapping,
        "delegate_artifacts_count": len(delegate_artifacts),
        "delegate_scripts_count": len(delegate_scripts),
        "proposals_citing_delegates": proposals_citing_delegates,
        "total_proposals": total_proposals,
        "delegate_citation_rate": round(citation_rate, 3) if citation_rate is not None else None,
        "delegate_task_results": {
            "completed": completed,
            "failed": failed,
            "timed_out": timed_out,
        },
    }


# ---------------------------------------------------------------------------
# Agent Costs
# ---------------------------------------------------------------------------

def _parse_agent_costs(
    state: Dict[str, Any],
    session_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse agent cost data.

    If session_data is provided, use its agent_costs and cost_summary (ground-truth).
    Otherwise fall back to state.json agent_costs.
    """
    if session_data is not None:
        summary = session_data.get("cost_summary")
        agent_list = session_data.get("agent_costs")
        if summary and isinstance(summary, dict):
            result = dict(summary)
            result["source"] = "session_logs"
            if agent_list:
                result["agent_list"] = agent_list
            return result

    costs = state.get("campaign", {}).get("agent_costs")
    if not costs or not isinstance(costs, list):
        return None

    total_tokens = 0
    total_duration_ms = 0
    by_role: Dict[str, Dict[str, Any]] = {}

    for entry in costs:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "unknown")
        tokens = entry.get("total_tokens", 0) or 0
        duration = entry.get("duration_ms", 0) or 0
        total_tokens += tokens
        total_duration_ms += duration

        if role not in by_role:
            by_role[role] = {"count": 0, "total_tokens": 0, "total_duration_ms": 0}
        by_role[role]["count"] += 1
        by_role[role]["total_tokens"] += tokens
        by_role[role]["total_duration_ms"] += duration

    return {
        "total_agent_invocations": len(costs),
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms,
        "by_role": by_role,
    }


# ---------------------------------------------------------------------------
# Master Parser
# ---------------------------------------------------------------------------

def parse_campaign(
    artifact_dir: Path,
    session_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master parser. Returns the complete artifacts_snapshot dict.

    Args:
        artifact_dir: Path to the completed campaign artifact directory.
        session_data: Optional parsed session log data (from parse_session_logs.py).
            When provided, stage_timestamps and agent_costs use session log data
            instead of state.json.
    """
    state = _load_json(artifact_dir / "state.json") or {}

    target = _parse_target(artifact_dir)
    campaign = _parse_campaign(state)
    e2e_results = _parse_e2e_results(artifact_dir)
    gates = _parse_gates(artifact_dir, state)
    debate = _parse_debate(artifact_dir, state)
    tracks = _parse_tracks(state)

    stage_timestamps = _parse_stage_timestamps(state, session_data=session_data)
    delegation = _parse_delegation(artifact_dir, state)
    agent_costs = _parse_agent_costs(state, session_data=session_data)
    accuracy_verification = _parse_accuracy_verification(state, tracks, campaign)

    return {
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "target": target,
        "campaign": campaign,
        "e2e_results": e2e_results,
        "gates": gates,
        "debate": debate,
        "tracks": tracks,
        "stage_timestamps": stage_timestamps,
        "delegation": delegation,
        "agent_costs": agent_costs,
        "accuracy_verification": accuracy_verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact-dir", type=str, required=True,
        help="Path to completed campaign artifact directory",
    )
    parser.add_argument(
        "--session-data", type=str, default=None,
        help="Optional: path to session log data JSON (output of parse_session_logs.py). "
             "When provided, stage_timestamps and agent_costs use session log data "
             "instead of state.json.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for artifacts_snapshot.json (default: stdout)",
    )

    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    if not artifact_dir.exists():
        print(f"ERROR: Artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 1

    session_data: Optional[Dict[str, Any]] = None
    if args.session_data:
        session_data_path = Path(args.session_data).expanduser().resolve()
        if not session_data_path.exists():
            print(f"ERROR: Session data file does not exist: {session_data_path}", file=sys.stderr)
            return 1
        session_data = _load_json(session_data_path)
        if session_data is None:
            print(f"ERROR: Failed to parse session data JSON: {session_data_path}", file=sys.stderr)
            return 1
        print(f"Using session data from: {session_data_path}", file=sys.stderr)

    snapshot = parse_campaign(artifact_dir, session_data=session_data)
    json_str = json.dumps(snapshot, indent=2) + "\n"

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Wrote: {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
