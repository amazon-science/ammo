# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# test_build_dag.py
"""Tests for build_dag.py — events → causal DAG."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_events():
    from extract_events import extract_events
    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    return extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )


def _build(events=None):
    from build_dag import build_dag
    if events is None:
        events = _load_events()
    return build_dag(events=events, artifact_dir=str(FIXTURES))


def test_builds_coarse_dag_with_expected_nodes():
    dag = _build()
    node_ids = {n["id"] for n in dag["nodes"]}
    assert "bottleneck_mining" in node_ids
    assert "debate" in node_ids
    assert any(nid.startswith("impl_") for nid in node_ids)


def test_file_dependency_edges():
    dag = _build()
    file_edges = [e for e in dag["edges"] if e["type"] == "file_dependency"]
    mining_to_debate = [e for e in file_edges
                        if e["from"] == "bottleneck_mining" and e["to"] == "debate"]
    assert len(mining_to_debate) > 0
    assert any("bottleneck_analysis.md" in e.get("artifact", "") for e in mining_to_debate)


def test_data_citation_edges():
    dag = _build()
    citation_edges = [e for e in dag["edges"] if e["type"] == "data_citation"]
    assert len(citation_edges) > 0
    all_transferred = []
    for e in citation_edges:
        all_transferred.extend(e.get("data_values_transferred", []))
    assert any("29.8" in str(v) for v in all_transferred)


def test_decision_gate_edges():
    dag = _build()
    gate_edges = [e for e in dag["edges"] if e["type"] == "decision_gate"]
    froms = {e["from"] for e in gate_edges}
    tos = {e["to"] for e in gate_edges}
    assert "debate" in froms
    assert any(t.startswith("impl_") for t in tos)


def test_data_citation_disambiguation():
    dag = _build()
    citation_edges = [e for e in dag["edges"] if e["type"] == "data_citation"]
    for edge in citation_edges:
        if any("29.8" in str(v) for v in edge.get("data_values_transferred", [])):
            assert edge["from"] is not None


def test_nodes_have_data_claims_extracted():
    dag = _build()
    for node in dag["nodes"]:
        assert "data_claims_extracted" in node
        assert isinstance(node["data_claims_extracted"], int)


def test_dag_json_schema():
    dag = _build()
    assert "version" in dag
    assert "nodes" in dag
    assert "edges" in dag
    assert "summary" in dag
    assert "total_nodes" in dag["summary"]
