# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for AMMO track state reconciliation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconcile_track_state.py"
_spec = importlib.util.spec_from_file_location("reconcile_track_state", str(_SCRIPT))
reconcile_track_state = importlib.util.module_from_spec(_spec)
sys.modules["reconcile_track_state"] = reconcile_track_state
_spec.loader.exec_module(reconcile_track_state)


def _write_artifact(tmp_path: Path, evidence: dict) -> Path:
    artifact_dir = tmp_path / "artifacts"
    track_dir = artifact_dir / "rounds" / "1" / "tracks" / "op001"
    track_dir.mkdir(parents=True)
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "current_round": 1,
                    "rounds": [
                        {
                            "round_id": 1,
                            "parallel_tracks": {"tracks": {"op001": {"status": "IN_PROGRESS"}}},
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    (track_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return artifact_dir


def _component_pass_evidence(**extra: dict) -> dict:
    evidence = {
        "correctness": {"status": "PASS"},
        "kernel_bench": {"status": "PASS"},
        "e2e": {"status": "PASS"},
    }
    evidence.update(extra)
    return evidence


def _generated_diluted_evidence(**extra: dict) -> dict:
    evidence = {
        "verdict": "FAIL",
        "diluted": False,
        "correctness": {"status": "PASS"},
        "kernel_bench": {"status": "PASS"},
        "e2e": {
            "status": "PASS",
            "admissibility": {"status": "PASS"},
            "fastpath_proof": {"status": "PASS", "hits": 1},
        },
        "kill_criteria": {"e2e_threshold": {"status": "PASS"}},
    }
    evidence.update(extra)
    return evidence


def _enable_binding_diluted_gates(artifact_dir: Path, **track_updates: object) -> None:
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]
    track.update(
        {
            "correctness": True,
            "gate_5_1a": "PASS",
            "gate_5_2": "PASS",
            **track_updates,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")


def _write_diluted_summary(artifact_dir: Path, gate: dict | None = None) -> None:
    summary_path = artifact_dir / "rounds" / "1" / "tracks" / "op001" / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "e2e_gate": gate
                or {"track_verdict": "PASS", "diluted_pass": {"eligible": True}}
            }
        ),
        encoding="utf-8",
    )


def test_inferred_pass_requires_nonempty_kill_criteria(tmp_path):
    artifact_dir = _write_artifact(tmp_path, _component_pass_evidence(kill_criteria={}))

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001")
    except ValueError as exc:
        assert "track verdict" in str(exc)
    else:
        raise AssertionError("empty kill_criteria must not infer PASS")


def test_inferred_pass_requires_kill_criteria_field(tmp_path):
    artifact_dir = _write_artifact(tmp_path, _component_pass_evidence())

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001")
    except ValueError as exc:
        assert "track verdict" in str(exc)
    else:
        raise AssertionError("missing kill_criteria must not infer PASS")


def test_inferred_pass_with_kill_criteria_updates_round_scoped_state(tmp_path):
    artifact_dir = _write_artifact(
        tmp_path,
        _component_pass_evidence(kill_criteria={"e2e_threshold": {"status": "PASS"}}),
    )

    result = reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]

    assert result.verdict == "PASS"
    assert track["status"] == "PASS"
    assert track["evidence_path"] == "rounds/1/tracks/op001/evidence.json"
    assert track["kill_criteria_results"] == {"e2e_threshold": "PASS"}
    assert "completed_at" in track


def test_contingent_boundary_arithmetic_reconciles_to_track_state(tmp_path):
    boundary = {
        "baseline_duration_us": 100.0,
        "optimized_duration_us": 90.0,
        "occurrence_count": 1000,
        "baseline_e2e_us": 2_000_000.0,
        "e2e_equivalent_improvement_pct": 0.5,
        "campaign_floor_pct": 0.5,
        "meets_floor": True,
    }
    artifact_dir = _write_artifact(
        tmp_path,
        _component_pass_evidence(
            kernel_bench={"status": "PASS", "boundary_ab": boundary},
            kill_criteria={"e2e_threshold": {"status": "PASS"}},
        ),
    )

    reconcile_track_state.reconcile_track_state(
        artifact_dir, "op001", write=True
    )
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]
    assert track["gate_5_2_boundary"] == boundary


def test_diluted_pass_marker_reconciles_to_track_state(tmp_path):
    artifact_dir = _write_artifact(
        tmp_path,
        _component_pass_evidence(
            verdict="PASS",
            diluted=True,
            kill_criteria={"e2e_threshold": {"status": "PASS"}},
        ),
    )

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))

    assert state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]["diluted"] is True
    assert state["campaign"]["rounds"][0]["diluted_tracks"] == [
        {
            "op_id": "op001",
            "tpot_improvement_pct": None,
            "decode_share_of_e2e": None,
        }
    ]


def test_diluted_true_cannot_reconcile_with_fail_verdict(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "FAIL", "diluted": True})

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    except ValueError as exc:
        assert "requires verdict PASS" in str(exc)
    else:
        raise AssertionError("diluted=true must never persist on a FAIL track")


def test_explicit_false_clears_stale_diluted_marker(tmp_path):
    artifact_dir = _write_artifact(
        tmp_path,
        _component_pass_evidence(
            verdict="PASS",
            diluted=False,
            kill_criteria={"e2e_threshold": {"status": "PASS"}},
        ),
    )
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]["diluted"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]["diluted"] is False


def test_non_boolean_diluted_evidence_is_rejected(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "PASS", "diluted": "true"})

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    except ValueError as exc:
        assert "must be boolean" in str(exc)
    else:
        raise AssertionError("non-boolean diluted evidence must be rejected")


def test_generated_diluted_summary_overrides_template_false(tmp_path):
    artifact_dir = _write_artifact(tmp_path, _generated_diluted_evidence())
    _enable_binding_diluted_gates(artifact_dir)
    _write_diluted_summary(artifact_dir)

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))

    assert state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]["diluted"] is True


def test_generated_diluted_summary_overrides_only_e2e_floor_failure(tmp_path):
    artifact_dir = _write_artifact(tmp_path, _generated_diluted_evidence())
    _enable_binding_diluted_gates(artifact_dir)
    _write_diluted_summary(artifact_dir)

    result = reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]

    assert result.verdict == "PASS"
    assert track["status"] == "PASS"
    assert track["diluted"] is True


def test_generated_diluted_summary_cannot_waive_binding_vetoes(tmp_path):
    cases = (
        ({"correctness": {"status": "FAIL"}}, {}, "correctness"),
        ({"kernel_bench": {"status": "FAIL"}}, {}, "kernel"),
        ({"e2e": {"status": "FAIL", "admissibility": {"status": "FAIL"}, "fastpath_proof": {"status": "PASS", "hits": 1}}}, {}, "admissibility"),
        ({"e2e": {"status": "FAIL", "admissibility": {"status": "PASS"}, "fastpath_proof": {"status": "FAIL", "hits": 0}}}, {}, "fast-path"),
        ({"kill_criteria": {"threshold": {"status": "FAIL"}}}, {}, "kill"),
        ({}, {"gate_5_1a": "FAIL"}, "Gate 5.1a"),
        ({}, {"correctness": False}, "Gate 5.1b"),
    )
    for index, (evidence_updates, track_updates, expected) in enumerate(cases):
        artifact_dir = _write_artifact(
            tmp_path / str(index), _generated_diluted_evidence(**evidence_updates)
        )
        _enable_binding_diluted_gates(artifact_dir, **track_updates)
        _write_diluted_summary(artifact_dir)
        original = (artifact_dir / "state.json").read_text(encoding="utf-8")

        try:
            reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"generated diluted PASS must not waive {expected}")
        assert (artifact_dir / "state.json").read_text(encoding="utf-8") == original


def test_generated_diluted_summary_cannot_promote_untouched_template(tmp_path):
    artifact_dir = _write_artifact(
        tmp_path,
        {
            "verdict": "FAIL",
            "diluted": False,
            "correctness": {"status": "FAIL"},
            "kernel_bench": {"status": "FAIL"},
            "e2e": {
                "status": "FAIL",
                "admissibility": {"status": "FAIL"},
                "fastpath_proof": {"status": "FAIL", "hits": 0},
            },
            "kill_criteria": {},
        },
    )
    _write_diluted_summary(artifact_dir)

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    except ValueError as exc:
        assert "cannot waive binding validation gates" in str(exc)
    else:
        raise AssertionError("untouched evidence template must never reconcile to diluted PASS")


def test_generated_diluted_summary_requires_eligible_true_and_pass_verdict(tmp_path):
    gates = (
        {"track_verdict": "PASS", "diluted_pass": {"eligible": False}},
        {"track_verdict": "FAIL", "diluted_pass": {"eligible": True}},
        {"track_verdict": "PASS", "diluted_pass": True},
    )
    for index, gate in enumerate(gates):
        artifact_dir = _write_artifact(tmp_path / str(index), {"verdict": "PASS"})
        summary_path = artifact_dir / "rounds" / "1" / "tracks" / "op001" / "validation_summary.json"
        summary_path.write_text(json.dumps({"e2e_gate": gate}), encoding="utf-8")
        try:
            reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
        except ValueError as exc:
            assert "diluted_pass" in str(exc)
        else:
            raise AssertionError("contradictory generated diluted summary must fail closed")


def test_diluted_rollup_reconcile_preserves_populated_metrics(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "PASS", "diluted": True})
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["diluted_tracks"] = [
        {
            "op_id": "op001",
            "tpot_improvement_pct": 3.25,
            "decode_share_of_e2e": 0.8,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["campaign"]["rounds"][0]["diluted_tracks"] == [
        {
            "op_id": "op001",
            "tpot_improvement_pct": 3.25,
            "decode_share_of_e2e": 0.8,
        }
    ]


def test_generated_plain_summary_clears_stale_explicit_true(tmp_path):
    artifact_dir = _write_artifact(
        tmp_path,
        _component_pass_evidence(
            verdict="PASS",
            diluted=True,
            kill_criteria={"e2e_threshold": {"status": "PASS"}},
        ),
    )
    summary_path = artifact_dir / "rounds" / "1" / "tracks" / "op001" / "validation_summary.json"
    summary_path.write_text(
        json.dumps({"e2e_gate": {"track_verdict": "PASS"}}),
        encoding="utf-8",
    )

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))

    assert state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]["diluted"] is False


def test_gpu_blocked_reconciles_without_completed_at(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "GPU_BLOCKED"})

    result = reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]

    assert result.verdict == "GPU_BLOCKED"
    assert track["status"] == "GPU_BLOCKED"
    assert track["verdict"] is None
    assert "completed_at" not in track


def test_gpu_blocked_reconciliation_clears_stale_completed_at(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "GPU_BLOCKED"})
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]
    track["status"] = "PASS"
    track["completed_at"] = "2026-05-11T00:00:00Z"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]

    assert track["status"] == "GPU_BLOCKED"
    assert "completed_at" not in track


def test_v2_missing_current_round_evidence_does_not_use_legacy_track_path(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "PASS"})
    current = artifact_dir / "rounds" / "1" / "tracks" / "op001" / "evidence.json"
    legacy = artifact_dir / "tracks" / "op001" / "evidence.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
    current.unlink()

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001")
    except FileNotFoundError as exc:
        assert "rounds/1" in str(exc).replace("\\", "/")
    else:
        raise AssertionError("V2 reconciliation must not consume legacy-root evidence")


def test_true_legacy_reconciliation_keeps_flat_layout(tmp_path):
    artifact_dir = tmp_path / "legacy"
    track_dir = artifact_dir / "tracks" / "op001"
    track_dir.mkdir(parents=True)
    (artifact_dir / "state.json").write_text(
        json.dumps({"parallel_tracks": {"op001": {"status": "IN_PROGRESS"}}}),
        encoding="utf-8",
    )
    (track_dir / "evidence.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    result = reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))

    assert result.verdict == "PASS"
    assert state["parallel_tracks"]["op001"]["evidence_path"] == "tracks/op001/evidence.json"


def test_reconcile_rejects_track_id_path_traversal(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "PASS"})

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "../../../tracks/op", write=True)
    except ValueError as exc:
        assert "Invalid track id" in str(exc)
    else:
        raise AssertionError("track id traversal must be rejected")


def test_reconcile_rejects_boolean_current_round(tmp_path):
    artifact_dir = _write_artifact(tmp_path, {"verdict": "PASS"})
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    except ValueError as exc:
        assert "current-round track container" in str(exc)
    else:
        raise AssertionError("boolean current_round must fail closed")


def test_partial_v2_campaign_cannot_reconcile_flat_legacy_track(tmp_path):
    artifact_dir = tmp_path / "partial"
    track_dir = artifact_dir / "tracks" / "op001"
    track_dir.mkdir(parents=True)
    state_path = artifact_dir / "state.json"
    original = {
        "campaign": {"status": "active"},
        "parallel_tracks": {"op001": {"status": "IN_PROGRESS"}},
    }
    state_path.write_text(json.dumps(original), encoding="utf-8")
    (track_dir / "evidence.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    try:
        reconcile_track_state.reconcile_track_state(artifact_dir, "op001", write=True)
    except ValueError as exc:
        assert "current-round track container" in str(exc)
    else:
        raise AssertionError("partial V2 campaign must not reconcile legacy-root evidence")
    assert json.loads(state_path.read_text(encoding="utf-8")) == original
