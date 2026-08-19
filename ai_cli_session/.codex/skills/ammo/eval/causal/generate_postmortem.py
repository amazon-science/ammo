#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate a causal post-mortem narrative + interactive DAG visualization.

Reads a scored DAG (from score_nodes.py), an events list (from extract_events.py),
and an optional deep-analysis list (from a causal_analyzer agent) to produce:
  - A narrative Markdown report with critical path, anomalies, and recommendations.
  - An interactive HTML visualization of the DAG using dagre-d3.
  - An enriched DAG JSON with summary fields populated.

Usage:
    python generate_postmortem.py \\
        --scored-dag /path/to/scored_dag.json \\
        --events /path/to/events.json \\
        --output-dag /path/to/final_dag.json \\
        --output-narrative /path/to/postmortem.md \\
        --output-viz /path/to/viz.html \\
        [--deep-analysis /path/to/deep_analysis.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_CAUSAL_DIR = Path(__file__).parent
_TEMPLATES_DIR = _CAUSAL_DIR / "templates"


def _read_template(name: str) -> str:
    """Read a template file from the templates directory."""
    path = _TEMPLATES_DIR / name
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Critical path computation
# ---------------------------------------------------------------------------

def _node_pipeline_pos(node_id: str) -> int:
    """Return a coarse pipeline position for a node ID."""
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


def _compute_critical_path(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    """Compute critical path via DP on topologically sorted nodes.

    Node weight = output_influence metric (fallback 0.5 if absent).
    Critical path = sequence from first to last node maximizing cumulative weight.

    Returns a list of node IDs in order.
    """
    if not nodes:
        return []

    # Sort nodes by event_range.start_idx (topological order approximation)
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (n.get("event_range", {}).get("start_idx", 0), _node_pipeline_pos(n["id"])),
    )

    node_ids = [n["id"] for n in sorted_nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # Node weight from output_influence metric
    def _weight(n: Dict[str, Any]) -> float:
        metrics = n.get("metrics") or {}
        oi = metrics.get("output_influence")
        return float(oi) if oi is not None else 0.5

    # Build adjacency list from edges (forward only: from lower to higher index)
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in edges:
        frm = edge.get("from")
        to = edge.get("to")
        if frm and to and frm in node_index and to in node_index:
            if node_index[frm] < node_index[to]:
                adj[frm].append(to)

    # DP: dist[i] = max cumulative weight ending at node i
    # prev[i] = predecessor node_id on best path to i
    dist: Dict[str, float] = {nid: 0.0 for nid in node_ids}
    prev: Dict[str, Optional[str]] = {nid: None for nid in node_ids}

    for nid in node_ids:
        w = _weight(next(n for n in nodes if n["id"] == nid))
        # Initialize: path consisting of just this node
        if dist[nid] == 0.0:
            dist[nid] = w

        for neighbor in adj.get(nid, []):
            candidate = dist[nid] + _weight(next(n for n in nodes if n["id"] == neighbor))
            if candidate > dist[neighbor]:
                dist[neighbor] = candidate
                prev[neighbor] = nid

    # Find end node with max dist
    best_end = max(node_ids, key=lambda nid: dist[nid])

    # Reconstruct path
    path = []
    cur: Optional[str] = best_end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    # If path has only 1 node (all isolated), return all nodes sorted
    if len(path) < 2:
        return node_ids

    return path


# ---------------------------------------------------------------------------
# Merge deep analysis into DAG nodes
# ---------------------------------------------------------------------------

def _merge_deep_analysis(
    nodes: List[Dict[str, Any]],
    deep_analysis: Optional[List[Dict[str, Any]]],
) -> None:
    """Attach deep_analysis entries to matching nodes in-place."""
    if not deep_analysis:
        for node in nodes:
            node.setdefault("deep_analysis", None)
        return

    da_by_node: Dict[str, Dict[str, Any]] = {}
    for entry in deep_analysis:
        nid = entry.get("node_id")
        if nid:
            da_by_node[nid] = entry

    for node in nodes:
        nid = node["id"]
        if nid in da_by_node:
            node["deep_analysis"] = da_by_node[nid]
        else:
            node.setdefault("deep_analysis", None)


# ---------------------------------------------------------------------------
# Trajectory-changing decisions
# ---------------------------------------------------------------------------

def _find_trajectory_decisions(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find nodes that have anomalies and at least one outgoing edge."""
    nodes_with_outgoing = set(edge.get("from") for edge in edges if edge.get("from"))

    decisions = []
    for node in nodes:
        anomaly = node.get("anomaly") or []
        if anomaly and node["id"] in nodes_with_outgoing:
            decisions.append({
                "node_id": node["id"],
                "anomaly_types": anomaly,
                "outgoing_count": sum(1 for e in edges if e.get("from") == node["id"]),
            })
    return decisions


# ---------------------------------------------------------------------------
# Collect actionable fixes from deep_analysis
# ---------------------------------------------------------------------------

def _collect_fixes(nodes: List[Dict[str, Any]]) -> List[str]:
    """Gather unique actionable_fix strings from deep_analysis fields."""
    fixes = []
    seen = set()
    for node in nodes:
        da = node.get("deep_analysis")
        if not da:
            continue
        fix = da.get("actionable_fix")
        if fix and fix not in seen:
            seen.add(fix)
            fixes.append(f"[{node['id']}] {fix}")
    return fixes


# ---------------------------------------------------------------------------
# Narrative rendering
# ---------------------------------------------------------------------------

def _fmt_campaign_outcome(dag: Dict[str, Any]) -> str:
    """Format the campaign outcome section."""
    summary = dag.get("summary", {})
    campaign_info = dag.get("campaign", {})
    lines = []
    artifact_dir = campaign_info.get("artifact_dir", "unknown")
    lines.append(f"- **Artifact directory**: `{artifact_dir}`")
    lines.append(f"- **Total nodes**: {summary.get('total_nodes', '?')}")
    lines.append(f"- **Total edges**: {summary.get('total_edges', '?')}")
    lines.append(f"- **Critical path length**: {len(summary.get('critical_path', []))}")
    lines.append(f"- **Trajectory-changing decisions**: {len(summary.get('trajectory_changing_decisions', []))}")
    lines.append(f"- **Actionable fixes**: {len(summary.get('actionable_fixes', []))}")
    return "\n".join(lines)


def _fmt_critical_path(cp: List[str], nodes: List[Dict[str, Any]]) -> str:
    """Format the critical path as a numbered list."""
    if not cp:
        return "_No critical path computed._"
    node_map = {n["id"]: n for n in nodes}
    lines = []
    for i, nid in enumerate(cp, 1):
        node = node_map.get(nid, {})
        role = node.get("agent_role", "")
        stage_labels = node.get("stage_labels", [])
        stage = stage_labels[0] if stage_labels else ""
        oi = None
        metrics = node.get("metrics") or {}
        oi = metrics.get("output_influence")
        oi_str = f", influence={oi:.2f}" if oi is not None else ""
        lines.append(f"{i}. **{nid}** — role: {role or 'unknown'}, stage: {stage or 'unknown'}{oi_str}")
    return "\n".join(lines)


def _fmt_trajectory_decisions(decisions: List[Dict[str, Any]]) -> str:
    """Format trajectory-changing decisions."""
    if not decisions:
        return "_No trajectory-changing decisions detected._"
    lines = []
    for d in decisions:
        anomaly_str = ", ".join(d["anomaly_types"])
        lines.append(f"- **{d['node_id']}**: anomalies=[{anomaly_str}], outgoing_edges={d['outgoing_count']}")
    return "\n".join(lines)


def _fmt_anomalies(nodes: List[Dict[str, Any]]) -> str:
    """Format anomaly details for all nodes that have anomalies."""
    anomalous = [n for n in nodes if n.get("anomaly")]
    if not anomalous:
        return "_No anomalies detected._"

    sections = []
    for node in anomalous:
        nid = node["id"]
        anomaly_types = node.get("anomaly", [])
        da = node.get("deep_analysis")
        metrics = node.get("metrics") or {}

        lines = [f"### {nid}"]
        lines.append(f"- **Anomaly types**: {', '.join(anomaly_types)}")

        # Relevant metrics
        relevant = {k: v for k, v in metrics.items() if v is not None}
        if relevant:
            metric_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in relevant.items())
            lines.append(f"- **Metrics**: {metric_str}")

        if da:
            if da.get("root_cause"):
                lines.append(f"- **Root cause**: {da['root_cause']}")
            causal_chain = da.get("causal_chain", [])
            if causal_chain:
                lines.append(f"- **Causal chain**: {' → '.join(causal_chain)}")
            if da.get("counterfactual"):
                lines.append(f"- **Counterfactual**: {da['counterfactual']}")
            if da.get("actionable_fix"):
                lines.append(f"- **Fix**: {da['actionable_fix']}")
            if da.get("confidence") is not None:
                lines.append(f"- **Confidence**: {da['confidence'] * 100:.0f}%")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _fmt_recommendations(fixes: List[str]) -> str:
    """Format actionable fixes as a list."""
    if not fixes:
        return "_No actionable fixes identified. Provide deep_analysis to populate this section._"
    lines = [f"- {fix}" for fix in fixes]
    return "\n".join(lines)


def _fmt_cross_version_delta(dag: Dict[str, Any]) -> str:
    """Format cross-version comparison (placeholder if not provided)."""
    delta = dag.get("summary", {}).get("cross_version_delta")
    if not delta:
        return "_No previous version data available for comparison._"
    lines = []
    for k, v in delta.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def _render_narrative(dag: Dict[str, Any], nodes: List[Dict[str, Any]]) -> str:
    """Render the narrative Markdown by filling in template placeholders."""
    template = _read_template("narrative_template.md")

    summary = dag.get("summary", {})
    cp = summary.get("critical_path", [])
    traj = summary.get("trajectory_changing_decisions", [])
    fixes = summary.get("actionable_fixes", [])

    replacements = {
        "{{campaign_outcome}}": _fmt_campaign_outcome(dag),
        "{{critical_path}}": _fmt_critical_path(cp, nodes),
        "{{trajectory_decisions}}": _fmt_trajectory_decisions(traj),
        "{{anomalies}}": _fmt_anomalies(nodes),
        "{{recommendations}}": _fmt_recommendations(fixes),
        "{{cross_version_delta}}": _fmt_cross_version_delta(dag),
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


# ---------------------------------------------------------------------------
# Viz rendering
# ---------------------------------------------------------------------------

def _render_viz(dag: Dict[str, Any]) -> str:
    """Render the interactive HTML visualization."""
    template = _read_template("viz_template.html")
    dag_json = json.dumps(dag, indent=None, default=str)
    return template.replace("__DAG_JSON_PLACEHOLDER__", dag_json)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_postmortem(
    scored_dag: Dict[str, Any],
    events: List[Dict[str, Any]],
    deep_analysis: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Generate a complete causal post-mortem from a scored DAG.

    Args:
        scored_dag: Scored DAG dict from score_nodes.score_nodes().
        events: Event list from extract_events.extract_events().
        deep_analysis: Optional list of deep analysis dicts, each with keys:
            node_id, anomaly_type, root_cause, causal_chain, counterfactual,
            skill_attribution, actionable_fix, confidence.

    Returns:
        Dict with keys:
            "dag":       Enriched DAG dict with summary fields populated.
            "narrative": Markdown post-mortem string.
            "viz_html":  Interactive HTML visualization string.
    """
    dag = deepcopy(scored_dag)
    nodes = dag["nodes"]
    edges = dag["edges"]

    # 1. Merge deep analysis into nodes
    _merge_deep_analysis(nodes, deep_analysis)

    # 2. Compute critical path
    critical_path = _compute_critical_path(nodes, edges)

    # 3. Find trajectory-changing decisions
    trajectory_decisions = _find_trajectory_decisions(nodes, edges)

    # 4. Collect actionable fixes from deep analysis
    actionable_fixes = _collect_fixes(nodes)

    # 5. Populate summary fields
    summary = dag.setdefault("summary", {})
    summary["critical_path"] = critical_path
    summary["trajectory_changing_decisions"] = trajectory_decisions
    summary["actionable_fixes"] = actionable_fixes

    # 6. Render outputs
    narrative = _render_narrative(dag, nodes)
    viz_html = _render_viz(dag)

    return {
        "dag": dag,
        "narrative": narrative,
        "viz_html": viz_html,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scored-dag", required=True, help="Path to scored_dag.json (from score_nodes.py)")
    p.add_argument("--events", required=True, help="Path to events.json (from extract_events.py)")
    p.add_argument("--output-dag", required=True, help="Output path for final enriched dag.json")
    p.add_argument("--output-narrative", required=True, help="Output path for postmortem.md")
    p.add_argument("--output-viz", required=True, help="Output path for viz.html")
    p.add_argument(
        "--deep-analysis",
        default=None,
        help="Optional path to deep_analysis.json (from causal_analyzer agent)",
    )

    args = p.parse_args()

    # Validate inputs
    for path_str, label in [(args.scored_dag, "scored-dag"), (args.events, "events")]:
        if not Path(path_str).exists():
            print(f"ERROR: {label} file not found: {path_str}", file=sys.stderr)
            return 1

    try:
        scored_dag = json.loads(Path(args.scored_dag).read_text(encoding="utf-8"))
        events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading input files: {e}", file=sys.stderr)
        return 1

    deep_analysis = None
    if args.deep_analysis:
        da_path = Path(args.deep_analysis)
        if not da_path.exists():
            print(f"ERROR: deep-analysis file not found: {da_path}", file=sys.stderr)
            return 1
        try:
            deep_analysis = json.loads(da_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR reading deep-analysis: {e}", file=sys.stderr)
            return 1

    result = generate_postmortem(
        scored_dag=scored_dag,
        events=events,
        deep_analysis=deep_analysis,
    )

    # Write outputs
    Path(args.output_dag).write_text(
        json.dumps(result["dag"], indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Wrote enriched DAG to: {args.output_dag}", file=sys.stderr)

    Path(args.output_narrative).write_text(result["narrative"], encoding="utf-8")
    print(f"Wrote narrative to: {args.output_narrative}", file=sys.stderr)

    Path(args.output_viz).write_text(result["viz_html"], encoding="utf-8")
    print(f"Wrote viz to: {args.output_viz}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
