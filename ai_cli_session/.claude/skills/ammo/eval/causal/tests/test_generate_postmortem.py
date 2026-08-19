# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for generate_postmortem.py."""
import json
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_scored():
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes
    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(session_jsonl_path=FIXTURES / "sample_session.jsonl", session_data=session_data)
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, anomalies = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag, anomalies, events


def test_generates_narrative_md():
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    assert "# Causal Post-Mortem" in result["narrative"]
    assert "Critical Path" in result["narrative"]


def test_generates_viz_html():
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    assert "__DAG_DATA__" in result["viz_html"]
    assert "dagre" in result["viz_html"].lower() or "d3" in result["viz_html"].lower()


def test_merges_deep_analysis_into_dag():
    from generate_postmortem import generate_postmortem
    scored_dag, anomalies, events = _build_scored()
    if anomalies:
        deep = [{
            "node_id": anomalies[0]["node_id"],
            "anomaly_type": anomalies[0]["anomaly_type"],
            "root_cause": "test root cause",
            "causal_chain": ["step 1", "step 2"],
            "counterfactual": "test counterfactual",
            "skill_attribution": "agent_behavior",
            "actionable_fix": "test fix",
            "confidence": 0.8,
        }]
        result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=deep)
        node = next(n for n in result["dag"]["nodes"] if n["id"] == anomalies[0]["node_id"])
        assert node["deep_analysis"] is not None
        assert node["deep_analysis"]["root_cause"] == "test root cause"


def test_critical_path_in_summary():
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    cp = result["dag"]["summary"]["critical_path"]
    assert isinstance(cp, list)
    assert len(cp) >= 2
    node_ids = {n["id"] for n in result["dag"]["nodes"]}
    for nid in cp:
        assert nid in node_ids


def test_works_without_deep_analysis():
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    assert result["narrative"]
    assert result["viz_html"]
    assert result["dag"]
