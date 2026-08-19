# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# test_score_nodes.py
"""Tests for score_nodes.py — node scoring + anomaly detection."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_scored_dag():
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes

    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, anomalies = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag, anomalies, events


def test_all_nodes_have_metrics():
    scored_dag, _, _ = _build_scored_dag()
    for node in scored_dag["nodes"]:
        m = node["metrics"]
        assert "data_grounding" in m
        assert "output_influence" in m
        assert "prediction_accuracy" in m
        assert "retry_rate" in m
        assert "transition_quality" in m


def test_data_grounding_metric():
    scored_dag, _, _ = _build_scored_dag()
    debate = next((n for n in scored_dag["nodes"] if n["id"] == "debate"), None)
    if debate:
        assert debate["metrics"]["data_grounding"] is not None


def test_prediction_accuracy_null_when_no_prediction():
    scored_dag, _, _ = _build_scored_dag()
    mining = next((n for n in scored_dag["nodes"] if n["id"] == "bottleneck_mining"), None)
    if mining:
        assert mining["metrics"]["prediction_accuracy"] is None


def test_transition_quality_sign_convention():
    from score_nodes import _compute_transition_quality
    tq = _compute_transition_quality(0.8, 0.4)
    assert tq == pytest.approx(0.4, abs=0.01)
    assert tq > 0.3


def test_thrashing_anomaly():
    from score_nodes import _classify_anomalies
    node_metrics = {
        "data_grounding": 0.8, "output_influence": 0.8,
        "prediction_accuracy": None, "retry_rate": 0.5,
        "transition_quality": 0.0,
    }
    anomalies = _classify_anomalies("test_node", node_metrics, downstream_metrics=[], track_outcome=None)
    types = [a["anomaly_type"] for a in anomalies]
    assert "THRASHING" in types


def test_lucky_win_requires_pass_outcome():
    from score_nodes import _classify_anomalies
    node_metrics = {
        "data_grounding": 0.8, "output_influence": 0.8,
        "prediction_accuracy": 0.1, "retry_rate": 0.0,
        "transition_quality": 0.0,
    }
    anomalies_pass = _classify_anomalies("impl_op001", node_metrics, downstream_metrics=[], track_outcome="PASSED")
    assert any(a["anomaly_type"] == "LUCKY_WIN" for a in anomalies_pass)
    anomalies_fail = _classify_anomalies("impl_op002", node_metrics, downstream_metrics=[], track_outcome="FAILED")
    assert not any(a["anomaly_type"] == "LUCKY_WIN" for a in anomalies_fail)


def test_anomalies_json_schema():
    _, anomalies, _ = _build_scored_dag()
    for a in anomalies:
        assert "node_id" in a
        assert "anomaly_type" in a
        assert "triggered_by" in a
        assert a["anomaly_type"] in ("DATA_CORRUPTION", "QUALITY_DROP", "WASTED_EFFORT", "THRASHING", "LUCKY_WIN")
