#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Score each DAG node on 5 metrics and classify 5 anomaly types.

Reads a scored DAG (from build_dag.py), a structured events list (from
extract_events.py), and a snapshot JSON to produce:
  - A scored DAG with `metrics` and `anomaly` fields on each node.
  - A flat list of anomaly dicts with schema:
      {"node_id", "anomaly_type", "triggered_by", "downstream_impact"}

Usage:
    python score_nodes.py \\
        --dag /path/to/dag.json \\
        --events /path/to/events.json \\
        --snapshot /path/to/snapshot.json \\
        --output /path/to/scored_dag.json \\
        --anomalies /path/to/anomalies.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prediction keyword pattern for prediction_accuracy metric
# ---------------------------------------------------------------------------

_PREDICTION_RE = re.compile(r"estimat|expect|predict|~", re.IGNORECASE)

# Speedup value pattern: matches "1.05x" or "+5%" or "5.0%"
_SPEEDUP_VALUE_RE = re.compile(
    r"(\d+\.\d+)\s*x\b"        # e.g. 1.05x
    r"|\+(\d+\.?\d*)\s*%"      # e.g. +5%
    r"|(\d+\.?\d*)\s*%\s+(?:E2E|speedup|improvement)",  # e.g. 5% E2E
    re.IGNORECASE,
)

# Percentage-as-multiplier conversion threshold:
# values > 2.0 are treated as % gains → convert to multiplier (1 + v/100)
_PCT_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Pipeline ordering helper (mirrors build_dag._node_pipeline_pos)
# ---------------------------------------------------------------------------

def _node_pipeline_pos(node_id: str) -> int:
    _ORDER = ["bottleneck_mining", "baseline_capture", "debate"]
    if node_id in _ORDER:
        return _ORDER.index(node_id)
    if node_id.startswith("impl_"):
        return len(_ORDER)
    if node_id == "integration":
        return len(_ORDER) + 100
    if node_id == "campaign_eval":
        return len(_ORDER) + 200
    return len(_ORDER) + 50


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_data_grounding(
    node: Dict[str, Any],
    node_events: List[Dict[str, Any]],
    upstream_data_outputs: List[Dict[str, Any]],
    data_claims_extracted: int,
) -> Optional[float]:
    """Fraction of this node's data_output values that appear in upstream data_citation edges.

    data_citation edges go FROM upstream nodes TO this node. We count how many
    of this node's own data_output values were cited from upstream (i.e., how
    many of this node's values appeared upstream).

    Simplified: count how many data_output events in this node have a value
    that also appears in any upstream node's data_outputs.
    """
    if data_claims_extracted == 0:
        return 1.0

    # Collect this node's data output values
    my_values: List[Tuple[str, float]] = []
    for evt in node_events:
        if evt["event_type"] == "data_output":
            dt = evt["details"].get("datum_type", "")
            val = evt["details"].get("value", 0.0)
            my_values.append((dt, val))

    if not my_values:
        return 1.0

    # Tolerances from build_dag
    _TOLERANCES: Dict[str, float] = {
        "f_value": 0.005,
        "kernel_timing": 0.02,
        "speedup": 0.01,
        "memory_size": 0.01,
        "bandwidth": 0.02,
    }
    _DEFAULT_TOL = 0.01

    def _match(a: float, b: float, tol: float) -> bool:
        denom = max(abs(a), abs(b))
        if denom == 0:
            return a == b
        return abs(a - b) / denom < tol

    grounded = 0
    for dt, val in my_values:
        tol = _TOLERANCES.get(dt, _DEFAULT_TOL)
        for up_dt, up_val in upstream_data_outputs:
            if up_dt == dt and _match(val, up_val, tol):
                grounded += 1
                break  # count once per claim

    return grounded / data_claims_extracted


def _compute_output_influence(
    node_id: str,
    edges: List[Dict[str, Any]],
    all_nodes: List[Dict[str, Any]],
    this_node_start_idx: int,
) -> float:
    """Fraction of downstream nodes (higher event_range.start_idx) this node has edges to."""
    downstream_node_ids = set(
        n["id"] for n in all_nodes
        if n["event_range"]["start_idx"] > this_node_start_idx
    )
    if not downstream_node_ids:
        return 0.0

    # Count how many downstream nodes this node has at least one edge to
    targets_reached = set()
    for edge in edges:
        if edge.get("from") == node_id:
            to = edge.get("to")
            if to in downstream_node_ids:
                targets_reached.add(to)

    return min(1.0, len(targets_reached) / len(downstream_node_ids))


def _extract_speedup_value(text: str) -> Optional[float]:
    """Extract a speedup value from text near prediction keywords."""
    for m in _SPEEDUP_VALUE_RE.finditer(text):
        if m.group(1) is not None:
            # Nx form: already a multiplier (e.g. 1.05)
            return float(m.group(1))
        elif m.group(2) is not None:
            # +X% form: convert to multiplier
            pct = float(m.group(2))
            return 1.0 + pct / 100.0
        elif m.group(3) is not None:
            # X% E2E form: convert to multiplier
            pct = float(m.group(3))
            return 1.0 + pct / 100.0
    return None


def _compute_prediction_accuracy(
    node_id: str,
    node_events: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Optional[float]:
    """Accuracy of speedup predictions vs actual snapshot results.

    Scans data_output events with datum_type 'speedup' near prediction keywords.
    For impl_* nodes, looks up actual e2e_speedup in snapshot validation_gates.
    Returns min(actual/predicted, predicted/actual), or None if no prediction found.
    """
    # Only meaningful for impl nodes with a corresponding validation gate
    if not node_id.startswith("impl_"):
        return None

    # Extract op_id from node_id: "impl_op001" → "op001", "impl_track_1" → None
    op_id_match = re.match(r"impl_(.+)", node_id)
    if not op_id_match:
        return None
    op_id = op_id_match.group(1)

    # Look up actual speedup in snapshot
    validation_gates = snapshot.get("gates", {}).get("validation_gates", [])
    gate = next((g for g in validation_gates if g.get("track_id") == op_id), None)
    if gate is None:
        return None

    actual_speedup = gate.get("e2e_speedup")
    if actual_speedup is None:
        return None

    # actual_speedup may be a multiplier like 1.05 (from "1.05x") or a % gain
    # Normalize: if > _PCT_THRESHOLD, treat as percentage gain
    if actual_speedup > _PCT_THRESHOLD:
        actual_speedup = 1.0 + actual_speedup / 100.0

    # Scan node events for data_output events with speedup datum_type
    # Check if the surrounding text (raw_text) contains prediction keywords
    predicted_speedup = None
    for evt in node_events:
        if evt["event_type"] != "data_output":
            continue
        details = evt.get("details", {})
        if details.get("datum_type") != "speedup":
            continue
        raw_text = details.get("raw_text", "")
        # Check if this speedup datum appears near a prediction keyword
        # We check raw_text and also the text block that spawned it
        if _PREDICTION_RE.search(raw_text):
            val = details.get("value", 0.0)
            # Normalize: if > _PCT_THRESHOLD, it's a % gain
            if val > _PCT_THRESHOLD:
                predicted_speedup = 1.0 + val / 100.0
            else:
                predicted_speedup = val
            break

    if predicted_speedup is None or predicted_speedup <= 0:
        return None

    if actual_speedup <= 0:
        return None

    return min(actual_speedup / predicted_speedup, predicted_speedup / actual_speedup)


def _compute_retry_rate(node_events: List[Dict[str, Any]]) -> float:
    """Fraction of tool_call events that failed."""
    tool_calls = [e for e in node_events if e["event_type"] == "tool_call"]
    if not tool_calls:
        return 0.0
    failed = sum(
        1 for e in tool_calls
        if not e.get("details", {}).get("success", True)
    )
    return failed / len(tool_calls)


def _compute_transition_quality(upstream_avg: float, downstream_avg: float) -> float:
    """Quality drop from upstream to downstream.

    Positive = quality dropped (upstream was better than downstream).
    Negative = quality improved downstream.
    """
    return upstream_avg - downstream_avg


def _avg_non_null_metrics(metrics: Dict[str, Any], exclude: Optional[set] = None) -> Optional[float]:
    """Average of non-None metric values, optionally excluding keys."""
    if exclude is None:
        exclude = set()
    vals = [
        v for k, v in metrics.items()
        if k not in exclude and v is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _compute_transition_quality_for_node(
    node_metrics: Dict[str, Any],
    downstream_nodes_metrics: List[Dict[str, Any]],
) -> Optional[float]:
    """Compute transition_quality for a node given its metrics and downstream nodes' metrics.

    Returns None for leaf nodes (no downstream nodes).
    """
    if not downstream_nodes_metrics:
        return None

    _EXCLUDE = {"transition_quality"}

    upstream_avg = _avg_non_null_metrics(node_metrics, exclude=_EXCLUDE)
    if upstream_avg is None:
        return None

    downstream_avgs = [
        _avg_non_null_metrics(m, exclude=_EXCLUDE)
        for m in downstream_nodes_metrics
    ]
    downstream_avgs = [v for v in downstream_avgs if v is not None]
    if not downstream_avgs:
        return None

    downstream_avg = sum(downstream_avgs) / len(downstream_avgs)
    return _compute_transition_quality(upstream_avg, downstream_avg)


# ---------------------------------------------------------------------------
# Anomaly classification
# ---------------------------------------------------------------------------

def _classify_anomalies(
    node_id: str,
    metrics: Dict[str, Any],
    downstream_metrics: List[Dict[str, Any]],
    track_outcome: Optional[str],
) -> List[Dict[str, Any]]:
    """Classify anomalies for a single node based on its metrics.

    Args:
        node_id: The node's ID string.
        metrics: Dict with keys: data_grounding, output_influence,
            prediction_accuracy, retry_rate, transition_quality.
        downstream_metrics: List of metrics dicts for immediate downstream nodes.
        track_outcome: For impl nodes, the validation gate status
            ("PASSED", "FAILED", "GATED_PASS", etc.), or None.

    Returns:
        List of anomaly dicts, each with:
            {"node_id", "anomaly_type", "triggered_by", "downstream_impact"}
    """
    anomalies = []

    dg = metrics.get("data_grounding")
    oi = metrics.get("output_influence")
    pa = metrics.get("prediction_accuracy")
    rr = metrics.get("retry_rate")
    tq = metrics.get("transition_quality")

    # DATA_CORRUPTION: data_grounding < 0.5 AND any downstream has prediction_accuracy < 0.3
    if dg is not None and dg < 0.5:
        downstream_low_pa = [
            m for m in downstream_metrics
            if m.get("prediction_accuracy") is not None and m["prediction_accuracy"] < 0.3
        ]
        if downstream_low_pa:
            anomalies.append({
                "node_id": node_id,
                "anomaly_type": "DATA_CORRUPTION",
                "triggered_by": {"data_grounding": dg},
                "downstream_impact": [],  # filled by caller
            })

    # QUALITY_DROP: transition_quality > 0.3
    if tq is not None and tq > 0.3:
        anomalies.append({
            "node_id": node_id,
            "anomaly_type": "QUALITY_DROP",
            "triggered_by": {"transition_quality": tq},
            "downstream_impact": [],
        })

    # WASTED_EFFORT: output_influence < 0.3 AND retry_rate < 0.1
    if oi is not None and oi < 0.3 and rr is not None and rr < 0.1:
        anomalies.append({
            "node_id": node_id,
            "anomaly_type": "WASTED_EFFORT",
            "triggered_by": {"output_influence": oi, "retry_rate": rr},
            "downstream_impact": [],
        })

    # THRASHING: retry_rate > 0.3
    if rr is not None and rr > 0.3:
        anomalies.append({
            "node_id": node_id,
            "anomaly_type": "THRASHING",
            "triggered_by": {"retry_rate": rr},
            "downstream_impact": [],
        })

    # LUCKY_WIN: prediction_accuracy < 0.3 AND track_outcome in (PASSED, GATED_PASS)
    if pa is not None and pa < 0.3 and track_outcome in ("PASSED", "GATED_PASS"):
        anomalies.append({
            "node_id": node_id,
            "anomaly_type": "LUCKY_WIN",
            "triggered_by": {"prediction_accuracy": pa},
            "downstream_impact": [],
        })

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_nodes(
    dag: Dict[str, Any],
    events: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Score each DAG node on 5 metrics and classify anomalies.

    Args:
        dag: DAG dict from build_dag.build_dag().
        events: Event list from extract_events.extract_events().
        snapshot: Snapshot JSON dict (campaign results with validation_gates).

    Returns:
        Tuple of (scored_dag, anomalies_list).
        scored_dag: Deep copy of dag with `metrics` and `anomaly` keys on each node.
        anomalies_list: Flat list of anomaly dicts.
    """
    scored_dag = deepcopy(dag)
    nodes = scored_dag["nodes"]
    edges = scored_dag["edges"]

    # --- Build per-node event index ---
    # Each node has event_range.start_idx and event_range.end_idx
    # We'll use the range to slice events for that node.
    # But build_dag strips _events_indices, so we reconstruct from event_range
    # by checking event stage assignments. Instead, we rebuild the node→events
    # mapping by re-running the same grouping logic as build_dag but for scoring.

    # Map each event to a node by re-running stage→node mapping
    # Simplified: group by event_range overlap (start_idx <= idx <= end_idx)
    # This is correct because each event belongs to exactly one node's range.
    node_event_map: Dict[str, List[Dict[str, Any]]] = {n["id"]: [] for n in nodes}

    # Build a mapping from event index to node
    # Since event ranges can overlap (multiple stages → same node), we use
    # the node's event_range to assign events within that range.
    # For events that fall in multiple ranges, assign to the most specific (smallest range).
    idx_to_node: Dict[int, str] = {}

    # Sort nodes by range size ascending so smaller (more specific) ranges win
    sorted_nodes = sorted(
        nodes,
        key=lambda n: n["event_range"]["end_idx"] - n["event_range"]["start_idx"],
    )

    for n in sorted_nodes:
        start = n["event_range"]["start_idx"]
        end = n["event_range"]["end_idx"]
        for idx in range(start, end + 1):
            if idx < len(events):
                # Only assign if not already assigned by a more specific node,
                # but since we process smallest ranges last (reversed), we overwrite.
                # Actually we want smallest range to win: process smallest LAST.
                idx_to_node[idx] = n["id"]

    # Now build node_event_map
    for idx, node_id in idx_to_node.items():
        if idx < len(events):
            node_event_map[node_id].append(events[idx])

    # --- Build upstream data outputs per node ---
    # Collect all data_output values from nodes that are "upstream" (lower pipeline pos)
    node_data_outputs: Dict[str, List[Tuple[str, float]]] = {}
    for n in nodes:
        nid = n["id"]
        vals = []
        for evt in node_event_map.get(nid, []):
            if evt["event_type"] == "data_output":
                dt = evt["details"].get("datum_type", "")
                val = evt["details"].get("value", 0.0)
                vals.append((dt, val))
        node_data_outputs[nid] = vals

    # --- Build track outcome map from snapshot ---
    validation_gates = snapshot.get("gates", {}).get("validation_gates", [])
    gate_by_track: Dict[str, Dict[str, Any]] = {
        g["track_id"]: g for g in validation_gates if g.get("track_id")
    }

    # --- First pass: compute all metrics except transition_quality ---
    node_metrics: Dict[str, Dict[str, Any]] = {}

    for node in nodes:
        nid = node["id"]
        nevents = node_event_map.get(nid, [])
        start_idx = node["event_range"]["start_idx"]
        data_claims = node.get("data_claims_extracted", 0)

        # Collect upstream data outputs
        my_pos = _node_pipeline_pos(nid)
        upstream_vals: List[Tuple[str, float]] = []
        for other_node in nodes:
            other_id = other_node["id"]
            if other_id != nid and _node_pipeline_pos(other_id) < my_pos:
                upstream_vals.extend(node_data_outputs.get(other_id, []))

        dg = _compute_data_grounding(node, nevents, upstream_vals, data_claims)
        oi = _compute_output_influence(nid, edges, nodes, start_idx)
        pa = _compute_prediction_accuracy(nid, nevents, snapshot)
        rr = _compute_retry_rate(nevents)

        node_metrics[nid] = {
            "data_grounding": dg,
            "output_influence": oi,
            "prediction_accuracy": pa,
            "retry_rate": rr,
            "transition_quality": None,  # filled in second pass
        }

    # --- Second pass: compute transition_quality ---
    for node in nodes:
        nid = node["id"]
        my_pos = _node_pipeline_pos(nid)

        # Find immediate downstream nodes: nodes with the smallest pipeline pos > my_pos
        # "Immediate" = nodes that have a DAG edge from this node
        downstream_node_ids = set()
        for edge in edges:
            if edge.get("from") == nid:
                to = edge.get("to")
                if to and to in node_metrics:
                    downstream_node_ids.add(to)

        downstream_m = [node_metrics[did] for did in downstream_node_ids if did in node_metrics]

        tq = _compute_transition_quality_for_node(node_metrics[nid], downstream_m)
        node_metrics[nid]["transition_quality"] = tq

    # --- Apply metrics to nodes in scored_dag ---
    for node in scored_dag["nodes"]:
        nid = node["id"]
        node["metrics"] = node_metrics[nid]

    # --- Classify anomalies ---
    all_anomalies: List[Dict[str, Any]] = []

    for node in scored_dag["nodes"]:
        nid = node["id"]
        m = node_metrics[nid]

        # Determine track outcome for impl nodes
        track_outcome = None
        if nid.startswith("impl_"):
            op_id = nid[len("impl_"):]
            gate = gate_by_track.get(op_id)
            if gate:
                track_outcome = gate.get("status")

        # Get downstream metrics
        downstream_ids = set()
        for edge in edges:
            if edge.get("from") == nid and edge.get("to") in node_metrics:
                downstream_ids.add(edge["to"])
        downstream_m = [node_metrics[did] for did in downstream_ids]

        anomalies = _classify_anomalies(nid, m, downstream_m, track_outcome)

        # Fill downstream_impact with actual node ids
        for a in anomalies:
            a["downstream_impact"] = list(downstream_ids)

        node["anomaly"] = [a["anomaly_type"] for a in anomalies]
        all_anomalies.extend(anomalies)

    return scored_dag, all_anomalies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dag", required=True, help="Path to dag.json (from build_dag.py)")
    p.add_argument("--events", required=True, help="Path to events.json (from extract_events.py)")
    p.add_argument("--snapshot", required=True, help="Path to snapshot.json")
    p.add_argument("--output", required=True, help="Output path for scored_dag.json")
    p.add_argument("--anomalies", required=True, help="Output path for anomalies.json")

    args = p.parse_args()

    # Load inputs
    for path_str, label in [
        (args.dag, "dag"),
        (args.events, "events"),
        (args.snapshot, "snapshot"),
    ]:
        p_obj = Path(path_str)
        if not p_obj.exists():
            print(f"ERROR: {label} file not found: {p_obj}", file=sys.stderr)
            return 1

    try:
        dag = json.loads(Path(args.dag).read_text(encoding="utf-8"))
        events = json.loads(Path(args.events).read_text(encoding="utf-8"))
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading input files: {e}", file=sys.stderr)
        return 1

    scored_dag, anomalies = score_nodes(dag=dag, events=events, snapshot=snapshot)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(scored_dag, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote scored DAG to: {out_path}", file=sys.stderr)

    anom_path = Path(args.anomalies)
    anom_path.write_text(json.dumps(anomalies, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {len(anomalies)} anomalies to: {anom_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
