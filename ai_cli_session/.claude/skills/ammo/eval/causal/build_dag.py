#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Build a coarse stage-level causal DAG from an AMMO events list.

Reads the structured event log produced by extract_events.py and emits a
directed acyclic graph where nodes are AMMO pipeline stages and edges encode
four types of causal dependencies: file_dependency, message_dependency,
data_citation, and decision_gate.

Usage:
    python build_dag.py \\
        --events /path/to/events.json \\
        --artifact-dir /path/to/campaign/artifacts \\
        --output /path/to/dag.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stage → Node ID mapping
# ---------------------------------------------------------------------------

# Priority-ordered list of (substring, node_id_prefix) rules.
# The first match wins. Impl nodes use a special handler.
_STAGE_RULES: List[Tuple[str, str]] = [
    ("bottleneck", "bottleneck_mining"),
    ("mining", "bottleneck_mining"),
    # "baseline" maps to bottleneck_mining because AMMO's Stage 1 combines
    # baseline capture and bottleneck mining into a single researcher pass.
    ("baseline", "bottleneck_mining"),
    ("debate", "debate"),
    ("impl", "impl"),        # special: needs op_id suffix
    ("4_5", "impl"),         # special: needs op_id suffix
    ("integration", "integration"),
    ("eval", "campaign_eval"),
    ("6", "campaign_eval"),  # stage_6_7_eval
    ("7", "campaign_eval"),
]

# Tolerance by datum_type for data_citation matching
_CITATION_TOLERANCES: Dict[str, float] = {
    "f_value": 0.005,
    "kernel_timing": 0.02,
    "speedup": 0.01,
    "memory_size": 0.01,
    "bandwidth": 0.02,
}
_DEFAULT_TOLERANCE = 0.01

# Structural pipeline order used for decision_gate edges and disambiguation
_PIPELINE_ORDER = [
    "bottleneck_mining",
    "baseline_capture",
    "debate",
    # impl_* nodes come here (dynamic)
    "integration",
    "campaign_eval",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    """Return the filename portion of a path."""
    return os.path.basename(path) if path else ""


def _map_stage_to_node(stage_label: str, agent_name: Optional[str] = None) -> Optional[str]:
    """Map a stage label (e.g. 'round_1:implementation') to a node ID.

    Returns None for unrecognized or empty stages (those events are skipped).
    For impl stages, returns 'impl_{op_id}' where op_id is extracted from
    agent_name if available, otherwise 'impl_track_N'.
    """
    if not stage_label:
        return None

    label_lower = stage_label.lower()

    for fragment, node_prefix in _STAGE_RULES:
        if fragment in label_lower:
            if node_prefix == "impl":
                op_id = _extract_op_id(agent_name)
                return f"impl_{op_id}" if op_id else None  # resolved later
            return node_prefix

    return None


def _extract_op_id(agent_name: Optional[str]) -> Optional[str]:
    """Extract an op ID from an agent name like 'impl-champion-op001' → 'op001'."""
    if not agent_name:
        return None
    import re
    # Match op001, op1, op-001 patterns
    m = re.search(r"op[-_]?(\d+)", agent_name.lower())
    if m:
        return f"op{m.group(1).lstrip('0') or '0'}"
    return None


def _values_match(a: float, b: float, tolerance: float) -> bool:
    """Return True when |a - b| / max(|a|, |b|) < tolerance."""
    denom = max(abs(a), abs(b))
    if denom == 0:
        return a == b
    return abs(a - b) / denom < tolerance


# ---------------------------------------------------------------------------
# Core: group events into nodes
# ---------------------------------------------------------------------------

def _group_events_into_nodes(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group events by stage, map to node IDs, and build node metadata.

    Returns dict of node_id → node dict.
    """
    # First pass: determine the op_id counter for impl nodes without explicit ids
    impl_counter = 0
    # Map from stage_label → node_id (to stay consistent across events)
    stage_to_node: Dict[str, str] = {}

    # Pre-scan for agent_spawn events to get agent names per stage
    stage_to_agent: Dict[str, Dict[str, Any]] = {}
    for evt in events:
        stage = evt.get("stage")
        if not stage:
            continue
        if evt["event_type"] == "agent_spawn" and stage not in stage_to_agent:
            stage_to_agent[stage] = {
                "agent_id": evt.get("agent_id"),
                "agent_name": evt.get("agent_name"),
                "agent_role": evt.get("agent_role"),
            }

    # Determine node_id for each stage label
    for evt in events:
        stage = evt.get("stage")
        if not stage or stage in stage_to_node:
            continue

        agent_name = stage_to_agent.get(stage, {}).get("agent_name")
        node_id = _map_stage_to_node(stage, agent_name)

        if node_id == "impl_None" or node_id is None:
            # Try to find the impl node's agent name from spawns in this stage
            # Fall back to sequential naming
            if stage not in stage_to_node:
                impl_counter += 1
                node_id = f"impl_track_{impl_counter}"

        if node_id:
            stage_to_node[stage] = node_id

    # Build nodes dict; when multiple stages map to the same node_id, merge them
    nodes: Dict[str, Dict[str, Any]] = {}

    for idx, evt in enumerate(events):
        stage = evt.get("stage")
        if not stage or stage not in stage_to_node:
            continue

        node_id = stage_to_node[stage]
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "stage_labels": [],
                "inputs": [],
                "outputs": [],
                "data_claims_extracted": 0,
                "event_range": {"start_idx": idx, "end_idx": idx},
                "agent_id": "orchestrator",
                "agent_name": "orchestrator",
                "agent_role": "orchestrator",
                "_events_indices": [],
            }

        node = nodes[node_id]

        # Update stage_labels
        if stage not in node["stage_labels"]:
            node["stage_labels"].append(stage)

        # Update event range
        if idx < node["event_range"]["start_idx"]:
            node["event_range"]["start_idx"] = idx
        if idx > node["event_range"]["end_idx"]:
            node["event_range"]["end_idx"] = idx

        node["_events_indices"].append(idx)

        event_type = evt["event_type"]

        if event_type == "file_write":
            path = evt.get("details", {}).get("path", "")
            if path and path not in node["outputs"]:
                node["outputs"].append(path)

        elif event_type == "file_read":
            path = evt.get("details", {}).get("path", "")
            if path and path not in node["inputs"]:
                node["inputs"].append(path)

        elif event_type == "data_output":
            node["data_claims_extracted"] += 1

        elif event_type == "agent_spawn":
            # Use the first agent_spawn to set agent metadata
            if node["agent_id"] == "orchestrator":
                node["agent_id"] = evt.get("agent_id") or "orchestrator"
                node["agent_name"] = evt.get("agent_name") or "orchestrator"
                node["agent_role"] = evt.get("agent_role") or "orchestrator"

    return nodes


# ---------------------------------------------------------------------------
# Core: build edges
# ---------------------------------------------------------------------------

def _build_file_dependency_edges(
    nodes: Dict[str, Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each file_write in node A and file_read in node B where paths match
    and B's event is after A's event, emit a file_dependency edge."""
    edges = []
    seen: set = set()

    # Build lookup: filename → list of (node_id, event_idx, timestamp)
    # for writes and reads
    writes: Dict[str, List[Tuple[str, int, Optional[str]]]] = {}  # basename → list
    reads: Dict[str, List[Tuple[str, int, Optional[str]]]] = {}

    for node_id, node in nodes.items():
        indices = node["_events_indices"]
        for idx in indices:
            evt = events[idx]
            if evt["event_type"] == "file_write":
                path = evt.get("details", {}).get("path", "")
                bn = _basename(path)
                if bn:
                    writes.setdefault(bn, []).append((node_id, idx, evt.get("timestamp"), path))
            elif evt["event_type"] == "file_read":
                path = evt.get("details", {}).get("path", "")
                bn = _basename(path)
                if bn:
                    reads.setdefault(bn, []).append((node_id, idx, evt.get("timestamp"), path))

    # Match writes to reads where read is after write and different nodes
    for bn, write_list in writes.items():
        if bn not in reads:
            continue
        for w_node, w_idx, w_ts, w_path in write_list:
            for r_node, r_idx, r_ts, r_path in reads[bn]:
                if r_node == w_node:
                    continue
                if r_idx <= w_idx:
                    continue
                key = (w_node, r_node, bn)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "type": "file_dependency",
                    "from": w_node,
                    "to": r_node,
                    "artifact": bn,
                })

    return edges


def _build_message_dependency_edges(
    nodes: Dict[str, Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each send_message event, create edge from sender's node to receiver's node."""
    edges = []
    seen: set = set()

    # Build agent_id → node_id lookup
    agent_to_node: Dict[str, str] = {}
    for node_id, node in nodes.items():
        if node["agent_id"] and node["agent_id"] != "orchestrator":
            agent_to_node[node["agent_id"]] = node_id

    for node_id, node in nodes.items():
        for idx in node["_events_indices"]:
            evt = events[idx]
            if evt["event_type"] != "send_message":
                continue
            details = evt.get("details", {})
            from_agent = details.get("from_agent")
            to_agent = details.get("to_agent")

            from_node = agent_to_node.get(from_agent) if from_agent else node_id
            to_node = agent_to_node.get(to_agent) if to_agent else None

            if from_node and to_node and from_node != to_node:
                key = (from_node, to_node)
                if key not in seen:
                    seen.add(key)
                    edges.append({
                        "type": "message_dependency",
                        "from": from_node,
                        "to": to_node,
                    })

    return edges


def _build_data_citation_edges(
    nodes: Dict[str, Dict[str, Any]],
    events: List[Dict[str, Any]],
    pipeline_order: List[str],
) -> List[Dict[str, Any]]:
    """For each data_output in node B, match against data_outputs in upstream nodes.

    Disambiguation: when multiple upstream nodes match, prefer the one with
    the latest file_write event timestamp.
    """
    edges = []

    # Build pipeline position lookup
    def _pipeline_pos(node_id: str) -> int:
        for i, nid in enumerate(pipeline_order):
            if nid == node_id:
                return i
            if nid == "impl_*" or node_id.startswith("impl_"):
                # impl nodes sit between debate and integration
                return len(_PIPELINE_ORDER) - 1
        return len(pipeline_order)

    def _node_pipeline_pos(node_id: str) -> int:
        # Find the position of this node based on AMMO stage order
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

    # Collect all data_outputs per node with their event index and timestamp
    node_data_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for node_id, node in nodes.items():
        outputs = []
        for idx in node["_events_indices"]:
            evt = events[idx]
            if evt["event_type"] == "data_output":
                outputs.append({
                    "value": evt["details"]["value"],
                    "datum_type": evt["details"]["datum_type"],
                    "raw_text": evt["details"].get("raw_text", ""),
                    "event_idx": idx,
                    "timestamp": evt.get("timestamp"),
                })
        node_data_outputs[node_id] = outputs

    # Collect latest file_write timestamp per node (for disambiguation)
    node_latest_write_ts: Dict[str, Optional[str]] = {}
    for node_id, node in nodes.items():
        latest = None
        for idx in node["_events_indices"]:
            evt = events[idx]
            if evt["event_type"] == "file_write":
                ts = evt.get("timestamp")
                if ts and (latest is None or ts > latest):
                    latest = ts
        node_latest_write_ts[node_id] = latest

    seen_edges: set = set()

    for b_node_id, b_outputs in node_data_outputs.items():
        if not b_outputs:
            continue
        b_pos = _node_pipeline_pos(b_node_id)

        # Find upstream nodes (earlier in pipeline)
        upstream_nodes = [
            nid for nid in node_data_outputs
            if _node_pipeline_pos(nid) < b_pos
        ]

        for b_out in b_outputs:
            b_val = b_out["value"]
            b_type = b_out["datum_type"]
            tolerance = _CITATION_TOLERANCES.get(b_type, _DEFAULT_TOLERANCE)

            # Find all upstream matches
            matches: List[Tuple[str, float, Optional[str]]] = []  # (node_id, value, timestamp)
            for a_node_id in upstream_nodes:
                for a_out in node_data_outputs[a_node_id]:
                    if a_out["datum_type"] != b_type:
                        continue
                    a_val = a_out["value"]
                    if _values_match(a_val, b_val, tolerance):
                        matches.append((a_node_id, a_val, node_latest_write_ts.get(a_node_id)))

            if not matches:
                continue

            # Disambiguation: prefer the node with the latest file_write timestamp
            def _sort_key(m: Tuple[str, float, Optional[str]]) -> str:
                return m[2] or ""

            matches.sort(key=_sort_key, reverse=True)
            best_node = matches[0][0]
            matched_values = [str(m[1]) for m in matches]

            edge_key = (best_node, b_node_id, b_type, b_val)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append({
                "type": "data_citation",
                "from": best_node,
                "to": b_node_id,
                "data_values_transferred": matched_values,
                "datum_type": b_type,
            })

    return edges


def _build_decision_gate_edges(
    nodes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Hardcoded structural edges representing AMMO's decision gates.

    baseline_capture → bottleneck_mining → debate → each impl_* →
    integration (if exists) → campaign_eval (if exists)
    """
    edges = []
    node_ids = set(nodes.keys())

    impl_nodes = sorted(nid for nid in node_ids if nid.startswith("impl_"))

    def _add(from_id: str, to_id: str) -> None:
        if from_id in node_ids and to_id in node_ids:
            edges.append({
                "type": "decision_gate",
                "from": from_id,
                "to": to_id,
            })

    # Pipeline structural order
    if "baseline_capture" in node_ids and "bottleneck_mining" in node_ids:
        _add("baseline_capture", "bottleneck_mining")

    if "bottleneck_mining" in node_ids and "debate" in node_ids:
        _add("bottleneck_mining", "debate")

    # debate → each impl_*
    for impl_id in impl_nodes:
        _add("debate", impl_id)

    # each impl_* → integration or campaign_eval
    for impl_id in impl_nodes:
        if "integration" in node_ids:
            _add(impl_id, "integration")
        elif "campaign_eval" in node_ids:
            _add(impl_id, "campaign_eval")

    if "integration" in node_ids and "campaign_eval" in node_ids:
        _add("integration", "campaign_eval")

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dag(events: List[Dict[str, Any]], artifact_dir: str) -> Dict[str, Any]:
    """Build a coarse stage-level causal DAG from an events list.

    Args:
        events: Structured event list from extract_events.extract_events().
        artifact_dir: Path to the campaign artifact directory (for context).

    Returns:
        DAG dict with schema:
        {
            "version": "1.0",
            "campaign": {"artifact_dir": ...},
            "nodes": [...],
            "edges": [...],
            "summary": {"total_nodes": N, "total_edges": M, ...},
        }
    """
    # Group events into nodes
    nodes = _group_events_into_nodes(events)

    # Build pipeline order (dynamic, including impl_ nodes)
    impl_nodes = sorted(nid for nid in nodes if nid.startswith("impl_"))
    full_pipeline = [
        "baseline_capture",
        "bottleneck_mining",
        "debate",
        *impl_nodes,
        "integration",
        "campaign_eval",
    ]

    # Build all edge types
    edges: List[Dict[str, Any]] = []
    edges.extend(_build_file_dependency_edges(nodes, events))
    edges.extend(_build_message_dependency_edges(nodes, events))
    edges.extend(_build_data_citation_edges(nodes, events, full_pipeline))
    edges.extend(_build_decision_gate_edges(nodes))

    # Strip internal fields from nodes before serializing
    node_list = []
    for node in nodes.values():
        n = {k: v for k, v in node.items() if not k.startswith("_")}
        node_list.append(n)

    # Sort nodes by pipeline order
    def _node_sort_key(n: Dict[str, Any]) -> int:
        nid = n["id"]
        try:
            return full_pipeline.index(nid)
        except ValueError:
            return len(full_pipeline)

    node_list.sort(key=_node_sort_key)

    # Summary
    stage_type_counts: Dict[str, int] = {}
    for n in node_list:
        nid = n["id"]
        if nid == "baseline_capture":
            stage_type_counts["baseline_capture"] = stage_type_counts.get("baseline_capture", 0) + 1
        elif nid == "bottleneck_mining":
            stage_type_counts["bottleneck_mining"] = stage_type_counts.get("bottleneck_mining", 0) + 1
        elif nid == "debate":
            stage_type_counts["debate"] = stage_type_counts.get("debate", 0) + 1
        elif nid.startswith("impl_"):
            stage_type_counts["impl"] = stage_type_counts.get("impl", 0) + 1
        elif nid == "integration":
            stage_type_counts["integration"] = stage_type_counts.get("integration", 0) + 1
        elif nid == "campaign_eval":
            stage_type_counts["campaign_eval"] = stage_type_counts.get("campaign_eval", 0) + 1
        else:
            stage_type_counts["other"] = stage_type_counts.get("other", 0) + 1

    summary = {
        "total_nodes": len(node_list),
        "total_edges": len(edges),
        "stage_type_counts": stage_type_counts,
        "edge_type_counts": _count_by_type(edges, "type"),
    }

    return {
        "version": "1.0",
        "campaign": {"artifact_dir": str(artifact_dir)},
        "nodes": node_list,
        "edges": edges,
        "summary": summary,
    }


def _count_by_type(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        v = item.get(key, "unknown")
        counts[v] = counts.get(v, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--events", required=True, help="Path to events.json")
    p.add_argument(
        "--artifact-dir",
        required=True,
        help="Path to campaign artifact directory",
    )
    p.add_argument("--output", required=True, help="Output path for DAG JSON")

    args = p.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"ERROR: events file not found: {events_path}", file=sys.stderr)
        return 1

    try:
        events = json.loads(events_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading events: {e}", file=sys.stderr)
        return 1

    dag = build_dag(events=events, artifact_dir=args.artifact_dir)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(dag, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"Wrote DAG with {dag['summary']['total_nodes']} nodes and "
        f"{dag['summary']['total_edges']} edges to: {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
