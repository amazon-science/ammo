#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Cross-version comparison of two causal DAGs.

Detects metric regressions, new/resolved anomalies, and edge count changes
between a current scored DAG and a previous scored DAG.

Usage:
    python diff_versions.py \\
        --current-dag /path/to/current_dag.json \\
        --previous-dag /path/to/previous_dag.json \\
        --output /path/to/diff.json \\
        [--deep] \\
        [--skill-diff /path/to/skill.patch] \\
        [--regression-report /path/to/report.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Anomaly extraction helpers
# ---------------------------------------------------------------------------

def _extract_anomaly_info(anomaly_field: Any) -> Optional[Dict[str, Any]]:
    """Normalize the 'anomaly' field from a DAG node into a canonical dict.

    score_nodes.py sets anomaly to a list of type strings: ["THRASHING", ...]
    Tests may inject a single dict: {"type": "DATA_CORRUPTION", "triggered_by": {...}}
    This function normalizes both into Optional[{"type": str, "triggered_by": ...}].

    If the field is None, empty, or False-y: returns None (no anomaly).
    If the field is a non-empty list of strings: returns {"type": first_type}.
    If the field is a dict with "type": returns the dict as-is.
    """
    if anomaly_field is None:
        return None

    if isinstance(anomaly_field, dict):
        if anomaly_field.get("type"):
            return anomaly_field
        return None

    if isinstance(anomaly_field, list):
        # Filter out None/empty entries
        types = [t for t in anomaly_field if t]
        if not types:
            return None
        return {"type": types[0], "triggered_by": None}

    return None


def _has_anomaly(node: Dict[str, Any]) -> bool:
    """Return True if this node has an active anomaly."""
    return _extract_anomaly_info(node.get("anomaly")) is not None


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------

def _compute_metric_deltas(
    current_node: Dict[str, Any],
    previous_node: Dict[str, Any],
) -> Dict[str, float]:
    """Compute per-metric deltas: current_value - previous_value.

    Skips metrics where either value is None.
    """
    _METRICS = [
        "data_grounding",
        "output_influence",
        "prediction_accuracy",
        "retry_rate",
        "transition_quality",
    ]
    deltas: Dict[str, float] = {}

    current_metrics = current_node.get("metrics", {}) or {}
    previous_metrics = previous_node.get("metrics", {}) or {}

    for metric in _METRICS:
        curr_val = current_metrics.get(metric)
        prev_val = previous_metrics.get(metric)
        if curr_val is not None and prev_val is not None:
            deltas[metric] = curr_val - prev_val

    return deltas


def _count_edges_by_type(edges: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count edges grouped by their 'type' field."""
    counts: Dict[str, int] = {}
    for edge in edges:
        edge_type = edge.get("type", "unknown")
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counts


def _count_anomalies_by_type(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count active anomalies by type across all nodes."""
    counts: Dict[str, int] = {}
    for node in nodes:
        info = _extract_anomaly_info(node.get("anomaly"))
        if info is not None:
            atype = info.get("type", "UNKNOWN")
            counts[atype] = counts.get(atype, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Skill-diff correlation helpers
# ---------------------------------------------------------------------------

# Map from section name → DAG nodes that the section influences
_SECTION_TO_NODES: Dict[str, List[str]] = {
    "researcher_guidance": ["baseline_capture", "bottleneck_mining"],
    "debate_guidance": ["debate"],
    "implementation_guidance": [],   # filled dynamically for impl_* nodes
    "delegation_guidance": ["debate"],
    "general": [],                   # filled dynamically to all nodes
}

# File-path keywords → section
_PATH_SECTION_RULES: List[Tuple[str, str]] = [
    ("researcher", "researcher_guidance"),
    ("bottleneck", "researcher_guidance"),
    ("champion", "debate_guidance"),
    ("debate", "debate_guidance"),
    ("impl", "implementation_guidance"),
    ("validator", "implementation_guidance"),
    ("delegate", "delegation_guidance"),
]


def _infer_section_from_path(file_path: str) -> Optional[str]:
    """Infer section from file path using keyword rules.

    Returns None if no rule matches (caller should check SKILL.md content).
    """
    lower = file_path.lower()
    for keyword, section in _PATH_SECTION_RULES:
        if keyword in lower:
            return section
    return None


def _infer_section_from_skill_md_context(hunk_header: str, hunk_lines: List[str]) -> str:
    """For SKILL.md hunks, infer section from surrounding ## headings.

    hunk_header is the @@ … @@ context string (may contain the function/heading name).
    hunk_lines are all lines in the hunk (context + added + removed).
    """
    # Look for a ## heading in the hunk lines themselves (most recent heading wins)
    current_heading: Optional[str] = None
    for line in hunk_lines:
        # Strip unified-diff leading char (space, +, -)
        text = line[1:] if line and line[0] in (" ", "+", "-") else line
        if text.startswith("## "):
            current_heading = text[3:].strip().lower()
    if current_heading is None:
        # Fall back to @@ context text
        current_heading = hunk_header.lower()

    if not current_heading:
        return "general"

    if any(k in current_heading for k in ("debate", "champion", "adversar")):
        return "debate_guidance"
    if any(k in current_heading for k in ("researcher", "bottleneck", "baseline", "mining")):
        return "researcher_guidance"
    if any(k in current_heading for k in ("impl", "implement", "validator", "validat")):
        return "implementation_guidance"
    if any(k in current_heading for k in ("delegat",)):
        return "delegation_guidance"
    return "general"


def _parse_skill_diff(patch_text: str) -> List[Dict[str, Any]]:
    """Parse a unified diff into a list of changed hunks.

    Each hunk dict:
    {
        "file": str,
        "section": str,
        "added_lines": list[str],
        "removed_lines": list[str],
    }
    """
    hunks: List[Dict[str, Any]] = []
    current_file: Optional[str] = None
    current_hunk_header: str = ""
    current_hunk_lines: List[str] = []
    added_lines: List[str] = []
    removed_lines: List[str] = []
    in_hunk = False

    def _flush_hunk() -> None:
        if current_file is None or not in_hunk:
            return
        # Determine section
        lower_path = (current_file or "").lower()
        if "skill.md" in lower_path or "skill.md" in Path(current_file).name.lower():
            section = _infer_section_from_skill_md_context(
                current_hunk_header, current_hunk_lines
            )
        else:
            section = _infer_section_from_path(current_file)
            if section is None:
                section = "general"
        hunks.append({
            "file": current_file,
            "section": section,
            "added_lines": list(added_lines),
            "removed_lines": list(removed_lines),
        })

    for raw_line in patch_text.splitlines():
        # New file header: "diff --git a/... b/..."
        if raw_line.startswith("diff --git "):
            _flush_hunk()
            in_hunk = False
            current_hunk_header = ""
            current_hunk_lines = []
            added_lines = []
            removed_lines = []
            # Extract b/ path
            m = re.search(r' b/(.+)$', raw_line)
            current_file = m.group(1) if m else None
            continue

        # +++ line also carries the filename (use as fallback)
        if raw_line.startswith("+++ "):
            path_part = raw_line[4:].strip()
            if path_part.startswith("b/"):
                path_part = path_part[2:]
            if path_part != "/dev/null":
                current_file = path_part
            continue

        if raw_line.startswith("--- "):
            continue

        # Hunk header: @@ -l,s +l,s @@ optional context
        if raw_line.startswith("@@ "):
            _flush_hunk()
            in_hunk = True
            current_hunk_header = ""
            current_hunk_lines = []
            added_lines = []
            removed_lines = []
            # Extract optional context after the second @@
            m = re.search(r'^@@[^@]+@@(.*)$', raw_line)
            current_hunk_header = m.group(1).strip() if m else ""
            continue

        if in_hunk:
            current_hunk_lines.append(raw_line)
            if raw_line.startswith("+"):
                added_lines.append(raw_line[1:])
            elif raw_line.startswith("-"):
                removed_lines.append(raw_line[1:])

    _flush_hunk()
    return hunks


def _map_skill_changes_to_nodes(
    skill_changes: List[Dict[str, Any]],
    all_node_ids: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Map each skill change section to the DAG node(s) it would affect.

    Returns: {node_id: [list of skill change dicts that could affect this node]}
    """
    # Collect impl_* and all node ids from the changes themselves + caller-provided list
    impl_nodes: List[str] = []
    all_nodes: List[str] = []
    if all_node_ids:
        all_nodes = list(all_node_ids)
        impl_nodes = [n for n in all_node_ids if n.startswith("impl_")]

    result: Dict[str, List[Dict[str, Any]]] = {}

    def _add(node_id: str, change: Dict[str, Any]) -> None:
        result.setdefault(node_id, []).append(change)

    for change in skill_changes:
        section = change.get("section", "general")

        if section == "researcher_guidance":
            for nid in ["baseline_capture", "bottleneck_mining"]:
                _add(nid, change)

        elif section == "debate_guidance":
            _add("debate", change)

        elif section == "implementation_guidance":
            # Affects all impl_* nodes
            if impl_nodes:
                for nid in impl_nodes:
                    _add(nid, change)
            else:
                # No node list provided — use a placeholder
                _add("impl_*", change)

        elif section == "delegation_guidance":
            _add("debate", change)

        else:  # "general"
            if all_nodes:
                for nid in all_nodes:
                    _add(nid, change)
            else:
                _add("*", change)

    return result


def _summarise_change(change: Dict[str, Any]) -> str:
    """Produce a short human-readable summary of a skill hunk."""
    added = [l.strip() for l in change.get("added_lines", []) if l.strip()]
    removed = [l.strip() for l in change.get("removed_lines", []) if l.strip()]
    parts: List[str] = []
    if added:
        parts.append(f"Added: {'; '.join(added[:3])}")
    if removed:
        parts.append(f"Removed: {'; '.join(removed[:3])}")
    if not parts:
        return "content changed"
    return " | ".join(parts)


def _correlate_changes(
    skill_node_map: Dict[str, List[Dict[str, Any]]],
    metric_deltas: Dict[str, Dict[str, float]],
    new_anomalies: List[Dict[str, Any]],
    resolved_anomalies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each node that has BOTH a skill change mapping AND a behavioral change,
    produce a hypothesis dict.

    Args:
        skill_node_map: {node_id: [skill changes]}, from _map_skill_changes_to_nodes
        metric_deltas: {node_id: {metric: delta}}, from diff_versions
        new_anomalies: list of {"node": ..., "type": ...}
        resolved_anomalies: list of {"node": ..., "type": ...}

    Returns: list of hypothesis dicts
    """
    _SIGNIFICANT_DELTA = 0.1

    # Build per-node behavioral change summaries
    new_anomaly_by_node: Dict[str, List[str]] = {}
    for a in new_anomalies:
        new_anomaly_by_node.setdefault(a["node"], []).append(a["type"])
    resolved_anomaly_by_node: Dict[str, List[str]] = {}
    for a in resolved_anomalies:
        resolved_anomaly_by_node.setdefault(a["node"], []).append(a["type"])

    # Expand wildcard mappings
    all_behaviour_nodes = (
        set(metric_deltas.keys())
        | set(new_anomaly_by_node.keys())
        | set(resolved_anomaly_by_node.keys())
    )

    expanded_map: Dict[str, List[Dict[str, Any]]] = {}
    for node_id, changes in skill_node_map.items():
        if node_id in ("*", "impl_*"):
            for bnode in all_behaviour_nodes:
                if node_id == "impl_*" and not bnode.startswith("impl_"):
                    continue
                expanded_map.setdefault(bnode, []).extend(changes)
        else:
            expanded_map.setdefault(node_id, []).extend(changes)

    hypotheses: List[Dict[str, Any]] = []
    for node_id in sorted(expanded_map.keys()):
        skill_changes = expanded_map[node_id]
        if not skill_changes:
            continue

        # Check if there is a significant behavioral change at this node
        node_metric_deltas = metric_deltas.get(node_id, {})
        significant_deltas = {
            m: d for m, d in node_metric_deltas.items()
            if abs(d) >= _SIGNIFICANT_DELTA
        }
        node_new_anomalies = new_anomaly_by_node.get(node_id, [])
        node_resolved_anomalies = resolved_anomaly_by_node.get(node_id, [])

        has_behavioral_change = (
            bool(significant_deltas)
            or bool(node_new_anomalies)
            or bool(node_resolved_anomalies)
        )
        if not has_behavioral_change:
            continue

        # Build behavioral_changes summary
        behavioral_changes: Dict[str, Any] = {}
        for m, d in significant_deltas.items():
            behavioral_changes[m] = round(d, 4)
        if node_new_anomalies:
            behavioral_changes["new_anomalies"] = node_new_anomalies
        if node_resolved_anomalies:
            behavioral_changes["resolved_anomalies"] = node_resolved_anomalies

        # Determine confidence
        if len(skill_changes) == 1 and (
            len(significant_deltas) + len(node_new_anomalies) + len(node_resolved_anomalies)
        ) == 1:
            confidence = "high"
        elif any(c.get("section") == "general" for c in skill_changes):
            confidence = "low"
        else:
            confidence = "medium"

        # Build hypothesis text
        sections_mentioned = sorted({c.get("section", "general") for c in skill_changes})
        delta_desc = ", ".join(
            f"{m} ({d:+.4f})" for m, d in sorted(significant_deltas.items())
        )
        anomaly_desc_parts: List[str] = []
        for atype in node_new_anomalies:
            anomaly_desc_parts.append(f"new {atype} anomaly")
        for atype in node_resolved_anomalies:
            anomaly_desc_parts.append(f"resolved {atype} anomaly")
        anomaly_desc = ", ".join(anomaly_desc_parts)

        evidence_parts: List[str] = []
        if delta_desc:
            evidence_parts.append(f"metric deltas: {delta_desc}")
        if anomaly_desc:
            evidence_parts.append(anomaly_desc)
        evidence_str = "; ".join(evidence_parts)

        change_summaries = [_summarise_change(c) for c in skill_changes[:3]]
        change_str = "; ".join(change_summaries)

        hypothesis = (
            f"Skill change to {', '.join(sections_mentioned)} ({change_str}) "
            f"correlates with behavioral change at {node_id} node: {evidence_str}. "
            f"Review whether the skill edit directly caused this shift."
        )

        # Recommended action
        if confidence == "high":
            recommended_action = (
                f"Directly investigate the {sections_mentioned[0]} change — "
                f"it is the sole candidate for the observed {evidence_str}."
            )
        elif any(a in ["DATA_CORRUPTION"] for a in node_new_anomalies):
            recommended_action = (
                "Audit data-grounding enforcement in the updated section and "
                "add a validation step to catch corruption early."
            )
        else:
            recommended_action = (
                f"Run an A/B comparison with and without the {sections_mentioned} "
                "skill changes to isolate the causal factor."
            )

        hypotheses.append({
            "node_id": node_id,
            "skill_changes": [
                {
                    "file": c.get("file", ""),
                    "section": c.get("section", "general"),
                    "summary": _summarise_change(c),
                }
                for c in skill_changes
            ],
            "behavioral_changes": behavioral_changes,
            "hypothesis": hypothesis,
            "confidence": confidence,
            "recommended_action": recommended_action,
        })

    return hypotheses


def _generate_regression_report(diff: Dict[str, Any]) -> str:
    """Format the diff data into a human-readable markdown regression report.

    When deep_investigation contains hypotheses (skill-diff mode), a richer
    report with Causal Hypotheses and Uncorrelated Changes sections is produced.
    """
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("# Causal DAG Regression Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Schema version: {diff.get('version', 'unknown')}")
    lines.append("")

    # Summary table
    new_anomalies = diff.get("new_anomalies", [])
    resolved_anomalies = diff.get("resolved_anomalies", [])
    metric_deltas = diff.get("metric_deltas", {})
    edge_changes = diff.get("edge_changes", {})

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- New anomalies: **{len(new_anomalies)}**")
    lines.append(f"- Resolved anomalies: **{len(resolved_anomalies)}**")
    lines.append(f"- Nodes with metric changes: **{len(metric_deltas)}**")
    lines.append(f"- Edge type changes: **{len(edge_changes)}**")
    lines.append("")

    # --- Rich skill-diff sections (only when deep_investigation has hypotheses) ---
    deep = diff.get("deep_investigation") or {}
    hypotheses = deep.get("hypotheses") if isinstance(deep, dict) else None

    if hypotheses is not None:
        # Skill Changes section
        skill_changes_parsed = deep.get("skill_changes_parsed", [])
        lines.append("## Skill Changes")
        lines.append("")
        if skill_changes_parsed:
            for hunk in skill_changes_parsed:
                added_n = len(hunk.get("added_lines", []))
                removed_n = len(hunk.get("removed_lines", []))
                lines.append(
                    f"- `{hunk['file']}`: {hunk['section']} "
                    f"— {added_n} lines added, {removed_n} lines removed"
                )
                # Key changes: first 3 added/removed lines
                key_lines: List[str] = []
                for l in hunk.get("added_lines", [])[:3]:
                    stripped = l.strip()
                    if stripped:
                        key_lines.append(f"+ {stripped}")
                for l in hunk.get("removed_lines", [])[:3]:
                    stripped = l.strip()
                    if stripped:
                        key_lines.append(f"- {stripped}")
                if key_lines:
                    lines.append(f"  - Key changes: {' | '.join(key_lines)}")
        else:
            lines.append("- No skill changes detected.")
        lines.append("")

        # Behavioral Changes section
        lines.append("## Behavioral Changes")
        lines.append("")
        has_behavioral = False
        for node_id, deltas in sorted(metric_deltas.items()):
            sig_deltas = {m: d for m, d in deltas.items() if abs(d) >= 0.1}
            node_new = [a["type"] for a in new_anomalies if a["node"] == node_id]
            node_resolved = [a["type"] for a in resolved_anomalies if a["node"] == node_id]
            if sig_deltas or node_new or node_resolved:
                has_behavioral = True
                parts: List[str] = []
                for m, d in sorted(sig_deltas.items()):
                    parts.append(f"{m} {d:+.4f}")
                for atype in node_new:
                    parts.append(f"new anomaly: {atype}")
                for atype in node_resolved:
                    parts.append(f"resolved: {atype}")
                lines.append(f"- `{node_id}`: {', '.join(parts)}")
        # Also nodes with anomalies but no metric deltas
        for a in new_anomalies:
            if a["node"] not in metric_deltas:
                has_behavioral = True
                lines.append(f"- `{a['node']}`: new anomaly: {a['type']}")
        for a in resolved_anomalies:
            if a["node"] not in metric_deltas:
                has_behavioral = True
                lines.append(f"- `{a['node']}`: resolved: {a['type']}")
        if not has_behavioral:
            lines.append("- No significant behavioral changes detected.")
        lines.append("")

        # Causal Hypotheses section
        lines.append("## Causal Hypotheses")
        lines.append("")
        if hypotheses:
            for hyp in hypotheses:
                lines.append(f"- **{hyp['node_id']}**: {hyp['hypothesis']}")
                for sc in hyp.get("skill_changes", []):
                    lines.append(f"  - Skill change: {sc['section']} in `{sc['file']}`")
                beh = hyp.get("behavioral_changes", {})
                beh_parts: List[str] = []
                for k, v in beh.items():
                    if isinstance(v, list):
                        beh_parts.append(f"{k}: {v}")
                    else:
                        beh_parts.append(f"{k}: {v:+.4f}" if isinstance(v, float) else f"{k}: {v}")
                if beh_parts:
                    lines.append(f"  - Behavioral evidence: {'; '.join(beh_parts)}")
                lines.append(f"  - Confidence: {hyp.get('confidence', 'unknown')}")
                lines.append(f"  - Recommended action: {hyp.get('recommended_action', '')}")
        else:
            lines.append("- No correlated changes found.")
        lines.append("")

        # Uncorrelated Changes section
        uncorr_skill = deep.get("uncorrelated_skill_changes", [])
        uncorr_behavioral = deep.get("uncorrelated_behavioral_changes", [])
        lines.append("## Uncorrelated Changes")
        lines.append("")
        if uncorr_skill:
            lines.append("Skill changes with no behavioral impact:")
            for item in uncorr_skill:
                lines.append(f"- `{item.get('file', '?')}` ({item.get('section', '?')})")
        else:
            lines.append("- Skill changes with no behavioral impact: none")
        if uncorr_behavioral:
            lines.append("Behavioral changes with no matching skill change (may be stochastic):")
            for node_id in uncorr_behavioral:
                lines.append(f"- `{node_id}`")
        else:
            lines.append("- Behavioral changes with no matching skill change: none")
        lines.append("")

    # --- Standard structural sections (always present) ---

    # New anomalies
    if new_anomalies:
        lines.append("## New Anomalies (Regressions)")
        lines.append("")
        lines.append("| Node | Anomaly Type |")
        lines.append("|------|-------------|")
        for a in new_anomalies:
            lines.append(f"| `{a['node']}` | `{a['type']}` |")
        lines.append("")

    # Resolved anomalies
    if resolved_anomalies:
        lines.append("## Resolved Anomalies (Improvements)")
        lines.append("")
        lines.append("| Node | Anomaly Type |")
        lines.append("|------|-------------|")
        for a in resolved_anomalies:
            lines.append(f"| `{a['node']}` | `{a['type']}` |")
        lines.append("")

    # Metric deltas
    if metric_deltas:
        lines.append("## Metric Deltas by Node")
        lines.append("")
        lines.append("Positive delta = current is higher than previous.")
        lines.append("")
        lines.append("| Node | Metric | Delta |")
        lines.append("|------|--------|-------|")
        for node_id, deltas in sorted(metric_deltas.items()):
            for metric, delta in sorted(deltas.items()):
                sign = "+" if delta > 0 else ""
                lines.append(f"| `{node_id}` | {metric} | {sign}{delta:.4f} |")
        lines.append("")

    # Edge changes
    if edge_changes:
        lines.append("## Edge Count Changes")
        lines.append("")
        lines.append("| Edge Type | Previous | Current | Delta |")
        lines.append("|-----------|----------|---------|-------|")
        for edge_type, counts in sorted(edge_changes.items()):
            prev_n = counts["previous"]
            curr_n = counts["current"]
            delta = counts["delta"]
            sign = "+" if delta > 0 else ""
            lines.append(f"| {edge_type} | {prev_n} | {curr_n} | {sign}{delta} |")
        lines.append("")

    # Anomaly distribution shift
    dist = diff.get("anomaly_distribution_shift", {})
    if dist:
        prev_dist = dist.get("previous", {})
        curr_dist = dist.get("current", {})
        all_types = sorted(set(list(prev_dist.keys()) + list(curr_dist.keys())))
        if all_types:
            lines.append("## Anomaly Distribution Shift")
            lines.append("")
            lines.append("| Anomaly Type | Previous Count | Current Count |")
            lines.append("|-------------|----------------|---------------|")
            for atype in all_types:
                prev_count = prev_dist.get(atype, 0)
                curr_count = curr_dist.get(atype, 0)
                lines.append(f"| `{atype}` | {prev_count} | {curr_count} |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_versions(
    current_dag: Dict[str, Any],
    previous_dag: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two scored causal DAGs and return a structured diff.

    Args:
        current_dag: The newer scored DAG (from score_nodes.py).
        previous_dag: The older scored DAG to compare against.

    Returns:
        Diff dict with schema:
        {
          "version": "1.0",
          "compared": {"current": {...}, "previous": {...}},
          "metric_deltas": {"node_id": {"metric": delta, ...}, ...},
          "new_anomalies": [{"node": "...", "type": "...", "not_in_previous": true}],
          "resolved_anomalies": [{"node": "...", "type": "...", "was_in_previous": true}],
          "edge_changes": {"type": {"previous": N, "current": M, "delta": M-N}},
          "anomaly_distribution_shift": {"previous": {...}, "current": {...}},
          "deep_investigation": null,
        }
    """
    # Index nodes by ID in both DAGs
    current_nodes_by_id: Dict[str, Dict[str, Any]] = {
        n["id"]: n for n in current_dag.get("nodes", [])
    }
    previous_nodes_by_id: Dict[str, Dict[str, Any]] = {
        n["id"]: n for n in previous_dag.get("nodes", [])
    }

    # Matched nodes (present in both DAGs)
    matched_node_ids = set(current_nodes_by_id.keys()) & set(previous_nodes_by_id.keys())

    # --- Metric deltas ---
    metric_deltas: Dict[str, Dict[str, float]] = {}
    for node_id in sorted(matched_node_ids):
        deltas = _compute_metric_deltas(
            current_nodes_by_id[node_id],
            previous_nodes_by_id[node_id],
        )
        if deltas:
            metric_deltas[node_id] = deltas

    # --- New anomalies: in current but not in previous ---
    new_anomalies: List[Dict[str, Any]] = []
    for node_id in sorted(matched_node_ids):
        curr_node = current_nodes_by_id[node_id]
        prev_node = previous_nodes_by_id[node_id]

        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        prev_info = _extract_anomaly_info(prev_node.get("anomaly"))

        if curr_info is not None and prev_info is None:
            new_anomalies.append({
                "node": node_id,
                "type": curr_info.get("type", "UNKNOWN"),
                "not_in_previous": True,
            })

    # Also detect nodes that are only in current (newly added nodes with anomalies)
    only_in_current = set(current_nodes_by_id.keys()) - set(previous_nodes_by_id.keys())
    for node_id in sorted(only_in_current):
        curr_node = current_nodes_by_id[node_id]
        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        if curr_info is not None:
            new_anomalies.append({
                "node": node_id,
                "type": curr_info.get("type", "UNKNOWN"),
                "not_in_previous": True,
            })

    # --- Resolved anomalies: in previous but not in current ---
    resolved_anomalies: List[Dict[str, Any]] = []
    for node_id in sorted(matched_node_ids):
        curr_node = current_nodes_by_id[node_id]
        prev_node = previous_nodes_by_id[node_id]

        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        prev_info = _extract_anomaly_info(prev_node.get("anomaly"))

        if prev_info is not None and curr_info is None:
            resolved_anomalies.append({
                "node": node_id,
                "type": prev_info.get("type", "UNKNOWN"),
                "was_in_previous": True,
            })

    # --- Edge count changes ---
    current_edge_counts = _count_edges_by_type(current_dag.get("edges", []))
    previous_edge_counts = _count_edges_by_type(previous_dag.get("edges", []))
    all_edge_types = set(list(current_edge_counts.keys()) + list(previous_edge_counts.keys()))

    edge_changes: Dict[str, Dict[str, int]] = {}
    for edge_type in sorted(all_edge_types):
        prev_count = previous_edge_counts.get(edge_type, 0)
        curr_count = current_edge_counts.get(edge_type, 0)
        delta = curr_count - prev_count
        if delta != 0:
            edge_changes[edge_type] = {
                "previous": prev_count,
                "current": curr_count,
                "delta": delta,
            }

    # --- Anomaly distribution shift ---
    previous_anomaly_dist = _count_anomalies_by_type(previous_dag.get("nodes", []))
    current_anomaly_dist = _count_anomalies_by_type(current_dag.get("nodes", []))

    # Build "compared" metadata
    compared = {
        "current": {
            "total_nodes": len(current_dag.get("nodes", [])),
            "total_edges": len(current_dag.get("edges", [])),
            "campaign": current_dag.get("campaign", {}),
        },
        "previous": {
            "total_nodes": len(previous_dag.get("nodes", [])),
            "total_edges": len(previous_dag.get("edges", [])),
            "campaign": previous_dag.get("campaign", {}),
        },
    }

    return {
        "version": "1.0",
        "compared": compared,
        "metric_deltas": metric_deltas,
        "new_anomalies": new_anomalies,
        "resolved_anomalies": resolved_anomalies,
        "edge_changes": edge_changes,
        "anomaly_distribution_shift": {
            "previous": previous_anomaly_dist,
            "current": current_anomaly_dist,
        },
        "deep_investigation": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--current-dag",
        required=True,
        help="Path to current (newer) scored DAG JSON",
    )
    p.add_argument(
        "--previous-dag",
        required=True,
        help="Path to previous (older) scored DAG JSON",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for diff JSON",
    )
    p.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="Enable deep investigation mode (generates regression report)",
    )
    p.add_argument(
        "--skill-diff",
        default=None,
        help="Optional path to a skill diff JSON for additional context",
    )
    p.add_argument(
        "--regression-report",
        default=None,
        help="Output path for markdown regression report (requires --deep)",
    )

    args = p.parse_args()

    # Load DAG files
    for path_str, label in [
        (args.current_dag, "current-dag"),
        (args.previous_dag, "previous-dag"),
    ]:
        if not Path(path_str).exists():
            print(f"ERROR: {label} file not found: {path_str}", file=sys.stderr)
            return 1

    try:
        current_dag = json.loads(Path(args.current_dag).read_text(encoding="utf-8"))
        previous_dag = json.loads(Path(args.previous_dag).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading input files: {e}", file=sys.stderr)
        return 1

    # Optionally load skill diff (as a unified diff / patch text)
    skill_patch_text: Optional[str] = None
    if args.skill_diff:
        skill_diff_path = Path(args.skill_diff)
        if not skill_diff_path.exists():
            print(f"WARNING: skill-diff file not found: {skill_diff_path}", file=sys.stderr)
        else:
            try:
                skill_patch_text = skill_diff_path.read_text(encoding="utf-8")
            except OSError as e:
                print(f"WARNING: could not read skill-diff: {e}", file=sys.stderr)

    # Run diff
    diff = diff_versions(current_dag=current_dag, previous_dag=previous_dag)

    # Deep mode
    if args.deep:
        if skill_patch_text is not None:
            # --- Full correlation analysis ---
            # Collect all DAG node IDs for wildcard expansion
            all_node_ids = [
                n["id"]
                for n in current_dag.get("nodes", [])
                if isinstance(n.get("id"), str)
            ]

            skill_changes = _parse_skill_diff(skill_patch_text)
            skill_node_map = _map_skill_changes_to_nodes(skill_changes, all_node_ids)
            hypotheses = _correlate_changes(
                skill_node_map,
                diff["metric_deltas"],
                diff["new_anomalies"],
                diff["resolved_anomalies"],
            )

            # Determine uncorrelated skill changes
            affected_nodes_from_skill = set(skill_node_map.keys()) - {"*", "impl_*"}
            behavioral_nodes = (
                set(diff["metric_deltas"].keys())
                | {a["node"] for a in diff["new_anomalies"]}
                | {a["node"] for a in diff["resolved_anomalies"]}
            )
            correlated_nodes = {h["node_id"] for h in hypotheses}

            # Skill changes that touched nodes with no behavioral change
            uncorrelated_skill: List[Dict[str, Any]] = []
            for node_id, changes in skill_node_map.items():
                if node_id in ("*", "impl_*"):
                    continue
                if node_id not in correlated_nodes:
                    for c in changes:
                        entry = {"file": c.get("file", ""), "section": c.get("section", "general")}
                        if entry not in uncorrelated_skill:
                            uncorrelated_skill.append(entry)

            # Behavioral changes with no matching skill change
            uncorrelated_behavioral = sorted(
                behavioral_nodes - correlated_nodes - affected_nodes_from_skill
            )

            diff["deep_investigation"] = {
                "mode": "skill_correlation",
                "skill_changes_parsed": skill_changes,
                "correlations_found": len(hypotheses),
                "hypotheses": hypotheses,
                "uncorrelated_skill_changes": uncorrelated_skill,
                "uncorrelated_behavioral_changes": uncorrelated_behavioral,
                # Also keep structural summary for compatibility
                "new_anomaly_count": len(diff["new_anomalies"]),
                "resolved_anomaly_count": len(diff["resolved_anomalies"]),
                "regressed_metrics": [
                    {"node": node_id, "metric": metric, "delta": delta}
                    for node_id, deltas in diff["metric_deltas"].items()
                    for metric, delta in deltas.items()
                    if delta < 0
                ],
                "improved_metrics": [
                    {"node": node_id, "metric": metric, "delta": delta}
                    for node_id, deltas in diff["metric_deltas"].items()
                    for metric, delta in deltas.items()
                    if delta > 0
                ],
            }
        else:
            # --- Structural-only deep mode (no skill diff) ---
            diff["deep_investigation"] = {
                "mode": "structural_diff",
                "new_anomaly_count": len(diff["new_anomalies"]),
                "resolved_anomaly_count": len(diff["resolved_anomalies"]),
                "regressed_metrics": [
                    {"node": node_id, "metric": metric, "delta": delta}
                    for node_id, deltas in diff["metric_deltas"].items()
                    for metric, delta in deltas.items()
                    if delta < 0
                ],
                "improved_metrics": [
                    {"node": node_id, "metric": metric, "delta": delta}
                    for node_id, deltas in diff["metric_deltas"].items()
                    for metric, delta in deltas.items()
                    if delta > 0
                ],
            }

    # Write diff JSON
    out_path = Path(args.output)
    try:
        out_path.write_text(
            json.dumps(diff, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        print(f"ERROR writing output: {e}", file=sys.stderr)
        return 1

    print(
        f"Wrote diff to: {out_path} "
        f"({len(diff['new_anomalies'])} new anomalies, "
        f"{len(diff['resolved_anomalies'])} resolved)",
        file=sys.stderr,
    )

    # Write regression report if requested
    if args.regression_report:
        report_text = _generate_regression_report(diff)
        report_path = Path(args.regression_report)
        try:
            report_path.write_text(report_text, encoding="utf-8")
        except OSError as e:
            print(f"ERROR writing regression report: {e}", file=sys.stderr)
            return 1
        print(f"Wrote regression report to: {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
