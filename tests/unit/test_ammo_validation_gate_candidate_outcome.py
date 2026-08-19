# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_SCRIPT = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "scripts" / "verify_validation_gates.py"
RECONCILE_SCRIPT = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "scripts" / "reconcile_track_state.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _derive_e2e(baseline_avg: float, optimized_avg: float) -> tuple[float, float]:
    """Speedup and improvement_pct exactly as the gate recomputes them."""
    return baseline_avg / optimized_avg, (1.0 - optimized_avg / baseline_avg) * 100.0


def _classify_e2e(speedup: float, config: dict) -> str:
    """Per-batch verdict from the campaign gating thresholds."""
    if speedup >= 1.0 + max(config["min_e2e_improvement_pct"], config["noise_tolerance_pct"]) / 100.0:
        return "PASS"
    if speedup >= 1.0 - config["noise_tolerance_pct"] / 100.0:
        return "NOISE"
    if speedup >= 1.0 - config["catastrophic_regression_pct"] / 100.0:
        return "REGRESSED"
    return "CATASTROPHIC"


def _write_track_artifact(
    artifact_dir: Path,
    *,
    track_id: str = "op001",
    status: str = "FAIL",
    evidence_overrides: dict | None = None,
) -> None:
    track_dir = artifact_dir / "rounds" / "1" / "tracks" / track_id
    track_dir.mkdir(parents=True)
    baseline_dir = artifact_dir / "rounds" / "1" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    worktree = artifact_dir / "worktree"
    worktree.mkdir()

    state = {
        "campaign": {
            "current_round": 1,
            "config": {
                "min_e2e_improvement_pct": 0.5,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
            },
            "rounds": [
                {
                    "round_id": 1,
                    "parallel_tracks": {
                        "started_at": "2026-04-29T00:00:00Z",
                        "completed_at": "2026-04-29T01:00:00Z",
                        "tracks": {
                            track_id: {
                                "status": status,
                                "verdict": None if status == "GPU_BLOCKED" else status,
                                "branch": f"ammo/{track_id}",
                                "worktree_path": str(worktree),
                                "validation_results_path": f"rounds/1/tracks/{track_id}/validation_results.md",
                                "evidence_path": f"rounds/1/tracks/{track_id}/evidence.json",
                                "correctness": status in {"PASS", "GATED_PASS"},
                                "gate_5_1a": "PASS" if status in {"PASS", "GATED_PASS"} else "FAIL",
                                "gate_5_2": "PASS" if status in {"PASS", "GATED_PASS"} else "FAIL",
                                "kill_criteria_results": {
                                    "speedup": "PASS" if status in {"PASS", "GATED_PASS"} else "FAIL"
                                },
                            }
                        },
                    },
                }
            ],
        }
    }
    (artifact_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    fail_speedup, fail_improvement = _derive_e2e(10.0, 10.5)
    evidence = {
        "schema_version": 2,
        "track_id": track_id,
        "baseline_source": {"kind": "stage1", "citation": "Stage 1 baseline reused; not re-run."},
        "correctness": {
            "status": "FAIL",
            "method": "torch.allclose",
            "atol": 1e-3,
            "rtol": 1e-3,
            "max_abs_diff": 2.0,
            "nan_inf_check": True,
            "graph_replay_check": True,
        },
        "kernel_bench": {
            "status": "FAIL",
            "weighted_speedup": 0.95,
            "measured_under_cuda_graphs": True,
            "buckets": [{"name": "bs1", "speedup": 0.95}],
        },
        "e2e": {
            "status": "FAIL",
            "run_purpose": "official",
            "baseline_avg_s": 10.0,
            "optimized_avg_s": 10.5,
            "speedup": fail_speedup,
            "improvement_pct": fail_improvement,
            "admissibility": {"status": "PASS"},
            "fastpath_proof": {"status": "PASS", "hits": 1},
        },
        "kill_criteria": {
            "speedup": {"status": "FAIL", "source_run_purpose": "official"},
        },
        "amdahl": {
            "component_share_f": 0.5,
            "kernel_speedup": 0.95,
            "expected_e2e_pct": -2.6315789474,
            "actual_e2e_pct": -5.0,
        },
        "cross_track_contamination": {"status": "N/A", "note": "single track"},
    }
    if evidence_overrides:
        evidence.update(evidence_overrides)
    # The gate recomputes both ratios from the arm latencies, so the fixture must
    # never hand-author them.
    e2e_block = evidence.get("e2e")
    if isinstance(e2e_block, dict):
        e2e_block["speedup"], e2e_block["improvement_pct"] = _derive_e2e(
            e2e_block["baseline_avg_s"], e2e_block["optimized_avg_s"]
        )
    (track_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    (track_dir / "validation_results.md").write_text(
        f"# {track_id} validation results\n\nOverall verdict: {status}\n",
        encoding="utf-8",
    )
    # Canonical artifacts are written for every status: check_canonical_validation_
    # artifacts runs on every non-GPU_BLOCKED track, so a FAIL fixture that omits
    # them fails for the wrong reason.
    config = state["campaign"]["config"]
    (artifact_dir / "target.json").write_text(
        json.dumps({"workload": {"batch_sizes": [1]}, "gating": config}),
        encoding="utf-8",
    )
    correctness_path = (
        artifact_dir / "rounds" / "1" / "sweeps" / "opt_correctness"
        / track_id / "json" / "correctness_verdict.json"
    )
    correctness_path.parent.mkdir(parents=True)
    correctness_path.write_text(
        json.dumps(
            {
                "gate": "5.1b",
                "verdict": "PASS",
                "num_questions": 100,
                "baseline_accuracy": 0.8,
                "optimized_accuracy": 0.8,
                "accuracy_delta": 0.0,
                "baseline_correct_count": 80,
                "optimized_correct_count": 80,
                "questions_lost": [],
                "questions_gained": [],
                "tolerance_pct": 1.0,
                "threshold": 0.79,
            }
        ),
        encoding="utf-8",
    )
    e2e = evidence["e2e"]
    sweep_path = artifact_dir / "rounds" / "1" / "sweeps" / "opt" / track_id / "e2e_latency_results.json"
    sweep_path.parent.mkdir(parents=True)
    raw_dir = sweep_path.parent / "json"
    raw_dir.mkdir()
    for label, avg in (("baseline", e2e["baseline_avg_s"]), ("opt", e2e["optimized_avg_s"])):
        (raw_dir / f"{label}.json").write_text(
            json.dumps({"avg_latency": avg}), encoding="utf-8"
        )
        (raw_dir / f"{label}.runner.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
    baseline_root = artifact_dir / "rounds" / "1" / "sweeps" / "baseline"
    sweep_path.write_text(
        json.dumps(
            {
                "execution_mode": "inproc_sweep",
                "out_dir": str(sweep_path.parent),
                "target_json": str(artifact_dir / "target.json"),
                "baseline_source": str(baseline_root),
                "bench": {"baseline_label": "baseline", "opt_label": "opt"},
                "results": [
                    {
                        "batch_size": 1,
                        "speedup": e2e["speedup"],
                        "improvement_pct": e2e["improvement_pct"],
                        "baseline": {
                            "avg_s": e2e["baseline_avg_s"],
                            "ok": True,
                            "returncode": 0,
                            "output_json": "json/baseline.json",
                            "runner_json": "json/baseline.runner.json",
                        },
                        "opt": {
                            "avg_s": e2e["optimized_avg_s"],
                            "ok": True,
                            "returncode": 0,
                            "output_json": "json/opt.json",
                            "runner_json": "json/opt.runner.json",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verdict = _classify_e2e(e2e["speedup"], config)
    summary_path = track_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "track_id": track_id,
                "inputs": {"e2e_json": str(sweep_path)},
                "status": {"has_e2e": True, "has_track_evidence": True},
                "e2e_gate": {
                    "per_bs_verdicts": [
                        {"batch_size": 1, "speedup": e2e["speedup"], "verdict": verdict}
                    ],
                    "track_verdict": verdict if verdict == "PASS" else "FAIL",
                    "thresholds": {
                        "noise_tolerance_pct": config["noise_tolerance_pct"],
                        "catastrophic_regression_pct": config["catastrophic_regression_pct"],
                        "min_e2e_improvement_pct": config["min_e2e_improvement_pct"],
                        "pass_floor_speedup": 1.0
                        + max(
                            config["min_e2e_improvement_pct"],
                            config["noise_tolerance_pct"],
                        )
                        / 100.0,
                    },
                    "pass": verdict == "PASS",
                    "failing": [],
                },
            }
        ),
        encoding="utf-8",
    )


def test_selected_candidate_no_patch_fail_blocks_validation(tmp_path):
    _write_track_artifact(tmp_path)
    module = _load_module(VALIDATION_SCRIPT, "verify_validation_gates")

    report = module.verify_validation(tmp_path, "op001")

    assert report.overall_status == "BLOCKED"
    assert report.advance_to_stage6 is False
    assert any("no-patch FAIL" in blocker or "failed without changed files" in blocker for blocker in report.blockers)


def test_selected_candidate_fail_with_attempt_history_is_not_candidate_success(tmp_path):
    _write_track_artifact(tmp_path, evidence_overrides={"attempt_history": [{"summary": "tried fused path"}]})
    module = _load_module(VALIDATION_SCRIPT, "verify_validation_gates")

    report = module.verify_validation(tmp_path, "op001")

    assert report.blockers == []
    assert report.overall_status == "EVIDENCE_COMPLETE_NO_PASS"
    assert report.advance_to_stage6 is True
    assert "mark the round EXHAUSTED" in report.recommendation
    assert "do not report candidate success" in report.recommendation


def test_hard_infeasibility_override_does_not_allow_no_patch_fail(tmp_path):
    _write_track_artifact(
        tmp_path,
        evidence_overrides={
            "hard_infeasibility_override": {
                "acknowledged_by_orchestrator": True,
                "reviewed_by_monitor_or_da": True,
                "reason": "Required private kernel hook is absent in this target.",
                "why_debate_selection_is_invalid_or_superseded": "Dispatch packet assumed the hook existed.",
                "evidence": ["trace shows fallback path only"],
            }
        },
    )
    module = _load_module(VALIDATION_SCRIPT, "verify_validation_gates")

    report = module.verify_validation(tmp_path, "op001")

    assert report.overall_status == "BLOCKED"
    assert report.advance_to_stage6 is False
    assert any("failed without changed files" in blocker for blocker in report.blockers)


def test_validation_warnings_do_not_block_a_passing_candidate(tmp_path, monkeypatch):
    _write_track_artifact(
        tmp_path,
        status="PASS",
        evidence_overrides={
            "correctness": {
                "status": "PASS",
                "method": "torch.allclose",
                "atol": 1e-3,
                "rtol": 1e-3,
                "max_abs_diff": 0.0,
                "nan_inf_check": True,
                "graph_replay_check": True,
            },
            "kernel_bench": {
                "status": "PASS",
                "weighted_speedup": 1.2,
                "measured_under_cuda_graphs": True,
                "buckets": [{"name": "bs1", "speedup": 1.2}],
            },
            "e2e": {
                "status": "PASS",
                "run_purpose": "official",
                "baseline_avg_s": 10.0,
                "optimized_avg_s": 9.0,
                "admissibility": {"status": "PASS"},
                "fastpath_proof": {"status": "PASS", "hits": 1},
            },
            "kill_criteria": {
                "speedup": {"status": "PASS", "source_run_purpose": "official"},
            },
            "amdahl": {
                "component_share_f": 0.5,
                "kernel_speedup": 1.25,
                "expected_e2e_pct": 10.0,
                "actual_e2e_pct": 10.0,
            },
            "changed_files": ["vllm/model_executor/layers/fused.py"],
        },
    )
    module = _load_module(VALIDATION_SCRIPT, "verify_validation_gates")

    def warn_candidate_outcome(track, state):
        return module.GateResult("candidate_outcome", "WARN", "non-blocking warning", [], track.track_id)

    monkeypatch.setattr(module, "check_candidate_outcome", warn_candidate_outcome)
    report = module.verify_validation(tmp_path, "op001")

    assert report.warnings == ["op001:candidate_outcome: non-blocking warning"]
    assert report.overall_status == "PASS"
    assert report.advance_to_stage6 is True


def test_gpu_blocked_reconciles_and_verifies_cleanly(tmp_path):
    _write_track_artifact(tmp_path, status="IN_PROGRESS")
    (tmp_path / "rounds" / "1" / "tracks" / "op001" / "evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "track_id": "op001",
                "verdict": "GPU_BLOCKED",
                "gpu_blocked": {
                    "reason": "No reserved GPU was available for the official run.",
                    "evidence": ["nvidia-smi unavailable in reserved pool"],
                },
                "kill_criteria": {
                    "gpu_available": {"status": "FAIL", "source_run_purpose": "official"}
                },
            }
        ),
        encoding="utf-8",
    )
    reconcile = _load_module(RECONCILE_SCRIPT, "reconcile_track_state")
    validation = _load_module(VALIDATION_SCRIPT, "verify_validation_gates")

    reconcile.reconcile_track_state(tmp_path, "op001", write=True)
    report = validation.verify_validation(tmp_path, "op001")

    assert report.blockers == []
    assert report.track_outcomes == {"op001": "GPU_BLOCKED"}
    assert report.overall_status == "GPU_BLOCKED"
    assert report.advance_to_stage6 is False
    assert "Lead triage is required" in report.recommendation
