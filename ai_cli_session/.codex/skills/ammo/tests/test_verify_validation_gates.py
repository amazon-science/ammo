# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for current-round validation gate semantics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_validation_gates.py"
_spec = importlib.util.spec_from_file_location("verify_validation_gates", str(_SCRIPT))
verify_validation_gates = importlib.util.module_from_spec(_spec)
sys.modules["verify_validation_gates"] = verify_validation_gates
_spec.loader.exec_module(verify_validation_gates)


def _track(status: str = "PASS") -> dict:
    return {
        "status": status,
        "branch": "branch",
        "worktree_path": "/tmp/nonexistent-worktree",
        "validation_results_path": "unused_validation.md",
        "evidence_path": "unused_evidence.json",
        "correctness": True,
        "gate_5_1a": "PASS",
        "gate_5_2": "PASS",
    }


def _track_context(metadata: dict, evidence: dict) -> verify_validation_gates.TrackContext:
    return verify_validation_gates.TrackContext(
        track_id="op_test",
        metadata=metadata,
        validation_path=Path("unused_validation.md"),
        validation_text="",
        evidence_path=Path("unused_evidence.json"),
        evidence=evidence,
    )


def _passing_evidence() -> dict:
    return {
        "e2e": {"status": "PASS"},
        "kill_criteria": {},
    }


def _complete_correctness(status: str) -> dict:
    return {
        "status": status,
        "method": "torch.allclose",
        "atol": 1e-3,
        "rtol": 1e-3,
        "max_abs_diff": 1e-4,
        "nan_inf_check": status == "PASS",
        "graph_replay_check": status == "PASS",
    }


def _complete_kernel_bench(status: str) -> dict:
    return {
        "status": status,
        "weighted_speedup": 1.05,
        "measured_under_cuda_graphs": status == "PASS",
        "buckets": [{"batch_size": 1, "speedup": 1.05}],
    }


def _canonical_validation_fixture(tmp_path: Path, *, speedup: float = 1.02):
    artifact = tmp_path / "artifact"
    track_dir = artifact / "rounds" / "1" / "tracks" / "op_test"
    correctness_path = (
        artifact / "rounds" / "1" / "sweeps" / "opt_correctness"
        / "op_test" / "json" / "correctness_verdict.json"
    )
    sweep_path = artifact / "rounds" / "1" / "sweeps" / "opt" / "op_test" / "e2e_latency_results.json"
    baseline_dir = artifact / "rounds" / "1" / "sweeps" / "baseline"
    for directory in (track_dir, correctness_path.parent, sweep_path.parent, baseline_dir):
        directory.mkdir(parents=True, exist_ok=True)

    state = {
        "campaign": {
            "current_round": 1,
            "config": {
                "min_e2e_improvement_pct": 0.5,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
            },
            "rounds": [{}],
        }
    }
    target = {
        "workload": {"batch_sizes": [1]},
        "gating": dict(state["campaign"]["config"]),
    }
    (artifact / "target.json").write_text(json.dumps(target), encoding="utf-8")
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
                "message": "PASS",
                "diagnostics": {},
            }
        ),
        encoding="utf-8",
    )
    baseline_avg = 1.0
    optimized_avg = baseline_avg / speedup
    improvement = (1.0 - optimized_avg / baseline_avg) * 100.0
    raw_dir = sweep_path.parent / "json"
    raw_dir.mkdir()
    for label, avg in (("base", baseline_avg), ("candidate", optimized_avg)):
        (raw_dir / f"{label}.json").write_text(
            json.dumps({"avg_latency": avg}), encoding="utf-8"
        )
        (raw_dir / f"{label}.runner.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
    sweep_path.write_text(
        json.dumps(
            {
                "execution_mode": "inproc_sweep",
                "out_dir": str(sweep_path.parent),
                "target_json": str(artifact / "target.json"),
                "baseline_source": str(baseline_dir),
                "bench": {"baseline_label": "base", "opt_label": "candidate"},
                "results": [
                    {
                        "batch_size": 1,
                        "speedup": speedup,
                        "improvement_pct": improvement,
                        "base": {
                            "avg_s": baseline_avg,
                            "ok": True,
                            "returncode": 0,
                            "output_json": "json/base.json",
                            "runner_json": "json/base.runner.json",
                        },
                        "candidate": {
                            "avg_s": optimized_avg,
                            "ok": True,
                            "returncode": 0,
                            "output_json": "json/candidate.json",
                            "runner_json": "json/candidate.runner.json",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path = track_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(artifact),
                "track_id": "op_test",
                "inputs": {"e2e_json": str(sweep_path)},
                "status": {"has_e2e": True, "has_track_evidence": True},
                "e2e_gate": {
                    "per_bs_verdicts": [
                        {"batch_size": 1, "speedup": speedup, "verdict": "PASS"}
                    ],
                    "track_verdict": "PASS",
                    "thresholds": {
                        "noise_tolerance_pct": 0.5,
                        "catastrophic_regression_pct": 5.0,
                        "min_e2e_improvement_pct": 0.5,
                        "pass_floor_speedup": 1.005,
                    },
                    "pass": True,
                    "failing": [],
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = _track("PASS")
    metadata.update({"diluted": False, "per_bs_verdict": {"1": "PASS"}})
    evidence = {
        "e2e": {
            "status": "PASS",
            "run_purpose": "official",
            "baseline_avg_s": baseline_avg,
            "optimized_avg_s": optimized_avg,
            "speedup": speedup,
            "improvement_pct": improvement,
        }
    }
    track = verify_validation_gates.TrackContext(
        track_id="op_test",
        metadata=metadata,
        validation_path=track_dir / "validation_results.md",
        validation_text="Gate 5.1b tolerance 1.0pp",
        evidence_path=track_dir / "evidence.json",
        evidence=evidence,
    )
    return artifact, state, track, correctness_path, sweep_path, summary_path


def test_canonical_validation_artifacts_recompute_clean_pass(tmp_path):
    artifact, state, track, *_ = _canonical_validation_fixture(tmp_path)

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_canonical_pass_requires_gate_5_1b_artifact(tmp_path):
    artifact, state, track, correctness_path, *_ = _canonical_validation_fixture(tmp_path)
    correctness_path.unlink()

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("5.1b" in item for item in result.evidence)


def test_canonical_gate_rejects_tampered_correctness_and_e2e_arithmetic(tmp_path):
    artifact, state, track, correctness_path, sweep_path, _ = _canonical_validation_fixture(tmp_path)
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    correctness["optimized_correct_count"] = 60
    correctness_path.write_text(json.dumps(correctness), encoding="utf-8")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["results"][0]["speedup"] = 9.0
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("correct" in item or "speedup" in item for item in result.evidence)


def test_canonical_gate_rejects_stale_summary_source_and_failed_runner(tmp_path):
    artifact, state, track, _, sweep_path, summary_path = _canonical_validation_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["inputs"]["e2e_json"] = str(artifact / "rounds" / "0" / "stale.json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["results"][0]["candidate"]["ok"] = False
    sweep["results"][0]["candidate"]["returncode"] = 1
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("canonical current-round official sweep" in item or "did not succeed" in item for item in result.evidence)


def test_canonical_gate_rejects_unrelated_structured_evidence(tmp_path):
    artifact, state, track, *_ = _canonical_validation_fixture(tmp_path)
    track.evidence["e2e"]["speedup"] = 2.0

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("structured E2E evidence" in item for item in result.evidence)


def test_canonical_gate_rejects_nonfinite_and_symlinked_primary(tmp_path):
    artifact, state, track, correctness_path, sweep_path, _ = _canonical_validation_fixture(tmp_path)
    external = tmp_path / "external-correctness.json"
    external.write_text(correctness_path.read_text(encoding="utf-8"), encoding="utf-8")
    correctness_path.unlink()
    correctness_path.symlink_to(external)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["results"][0]["candidate"]["avg_s"] = float("nan")
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("symlink" in item or "invalid" in item for item in result.evidence)


def test_canonical_gate_rejects_forged_diluted_summary_without_phase_proof(tmp_path):
    artifact, state, track, _, sweep_path, summary_path = _canonical_validation_fixture(
        tmp_path, speedup=1.001
    )
    track.metadata["diluted"] = True
    track.metadata["per_bs_verdict"] = {"1": "NOISE"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["e2e_gate"]["per_bs_verdicts"][0]["verdict"] = "NOISE"
    summary["e2e_gate"]["track_verdict"] = "PASS"
    summary["e2e_gate"]["diluted_pass"] = {
        "eligible": True,
        "candidate_bs": [1],
        "decode_floor_pct": 1.0,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    assert "phase_significance" not in sweep["results"][0]

    result = verify_validation_gates.check_canonical_validation_artifacts(track, artifact, state)

    assert result.status == "FAIL"
    assert any("diluted PASS" in item for item in result.evidence)


def _multi_bs_fixture(tmp_path: Path, batch_speedups=((1, 1.02), (4, 1.03), (8, 1.04))):
    """Canonical fixture with more than one sweep row.

    Before the per-BS reconciliation, `len(canonical_rows) == 1` disabled the
    only primary-data comparison on exactly this shape, so a multi-BS campaign
    could declare any E2E numbers it liked.
    """
    artifact = tmp_path / "artifact"
    track_dir = artifact / "rounds" / "1" / "tracks" / "op_test"
    correctness_path = (
        artifact / "rounds" / "1" / "sweeps" / "opt_correctness"
        / "op_test" / "json" / "correctness_verdict.json"
    )
    sweep_path = (artifact / "rounds" / "1" / "sweeps" / "opt" / "op_test"
                  / "e2e_latency_results.json")
    baseline_dir = artifact / "rounds" / "1" / "sweeps" / "baseline"
    raw_dir = sweep_path.parent / "json"
    for directory in (track_dir, correctness_path.parent, sweep_path.parent,
                      baseline_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    state = {
        "campaign": {
            "current_round": 1,
            "config": {
                "min_e2e_improvement_pct": 0.5,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
            },
            "rounds": [{}],
        }
    }
    batches = [bs for bs, _ in batch_speedups]
    target = {
        "workload": {"batch_sizes": batches},
        "gating": dict(state["campaign"]["config"]),
    }
    (artifact / "target.json").write_text(json.dumps(target), encoding="utf-8")
    correctness_path.write_text(
        json.dumps({
            "gate": "5.1b", "verdict": "PASS", "num_questions": 100,
            "baseline_accuracy": 0.8, "optimized_accuracy": 0.8,
            "accuracy_delta": 0.0, "baseline_correct_count": 80,
            "optimized_correct_count": 80, "questions_lost": [],
            "questions_gained": [], "tolerance_pct": 1.0, "threshold": 0.79,
            "message": "PASS", "diagnostics": {},
        }),
        encoding="utf-8",
    )

    rows = []
    per_bs_evidence = []
    per_bs_verdicts = []
    for bs, speedup in batch_speedups:
        baseline_avg = float(bs)
        optimized_avg = baseline_avg / speedup
        improvement = (1.0 - optimized_avg / baseline_avg) * 100.0
        for label, avg in (("base", baseline_avg), ("candidate", optimized_avg)):
            (raw_dir / f"{label}_bs{bs}.json").write_text(
                json.dumps({"avg_latency": avg}), encoding="utf-8")
            (raw_dir / f"{label}_bs{bs}.runner.json").write_text(
                json.dumps({"ok": True}), encoding="utf-8")
        rows.append({
            "batch_size": bs,
            "speedup": speedup,
            "improvement_pct": improvement,
            "base": {"avg_s": baseline_avg, "ok": True, "returncode": 0,
                     "output_json": f"json/base_bs{bs}.json",
                     "runner_json": f"json/base_bs{bs}.runner.json"},
            "candidate": {"avg_s": optimized_avg, "ok": True, "returncode": 0,
                          "output_json": f"json/candidate_bs{bs}.json",
                          "runner_json": f"json/candidate_bs{bs}.runner.json"},
        })
        per_bs_evidence.append({
            "batch_size": bs, "baseline_avg_s": baseline_avg,
            "optimized_avg_s": optimized_avg, "speedup": speedup,
            "improvement_pct": improvement, "verdict": "PASS",
        })
        per_bs_verdicts.append({"batch_size": bs, "speedup": speedup,
                                "verdict": "PASS"})
    sweep_path.write_text(
        json.dumps({
            "execution_mode": "inproc_sweep",
            "out_dir": str(sweep_path.parent),
            "target_json": str(artifact / "target.json"),
            "baseline_source": str(baseline_dir),
            "bench": {"baseline_label": "base", "opt_label": "candidate"},
            "results": rows,
        }),
        encoding="utf-8",
    )
    summary_path = track_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps({
            "artifact_dir": str(artifact), "track_id": "op_test",
            "inputs": {"e2e_json": str(sweep_path)},
            "status": {"has_e2e": True, "has_track_evidence": True},
            "e2e_gate": {
                "per_bs_verdicts": per_bs_verdicts,
                "track_verdict": "PASS",
                "thresholds": {
                    "noise_tolerance_pct": 0.5,
                    "catastrophic_regression_pct": 5.0,
                    "min_e2e_improvement_pct": 0.5,
                    "pass_floor_speedup": 1.005,
                },
                "pass": True, "failing": [],
            },
        }),
        encoding="utf-8",
    )
    metadata = _track("PASS")
    metadata.update({"diluted": False,
                     "per_bs_verdict": {str(bs): "PASS" for bs in batches}})
    smallest = per_bs_evidence[0]
    evidence = {
        "e2e": {
            "status": "PASS", "run_purpose": "official",
            "baseline_avg_s": smallest["baseline_avg_s"],
            "optimized_avg_s": smallest["optimized_avg_s"],
            "speedup": smallest["speedup"],
            "improvement_pct": smallest["improvement_pct"],
        }
    }
    track = verify_validation_gates.TrackContext(
        track_id="op_test",
        metadata=metadata,
        validation_path=track_dir / "validation_results.md",
        validation_text="Gate 5.1b tolerance 1.0pp",
        evidence_path=track_dir / "evidence.json",
        evidence=evidence,
    )
    return artifact, state, track, sweep_path, per_bs_evidence


def test_multi_bs_scalar_e2e_reconciles_against_smallest_row(tmp_path):
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_multi_bs_forged_scalar_e2e_is_now_caught(tmp_path):
    """THE regression: the old len(canonical_rows)==1 guard let this pass."""
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    track.evidence["e2e"]["speedup"] = 3.5
    track.evidence["e2e"]["improvement_pct"] = 71.4

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("speedup disagrees with official sweep row bs=1" in item
               for item in result.evidence)


def test_multi_bs_scalar_e2e_taken_from_wrong_row_is_caught(tmp_path):
    """Scalars summarize the SMALLEST batch size; a larger row's numbers fail."""
    artifact, state, track, _, per_bs = _multi_bs_fixture(tmp_path)
    largest = per_bs[-1]
    track.evidence["e2e"].update({
        "baseline_avg_s": largest["baseline_avg_s"],
        "optimized_avg_s": largest["optimized_avg_s"],
        "speedup": largest["speedup"],
        "improvement_pct": largest["improvement_pct"],
    })

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("bs=1" in item for item in result.evidence)


def test_multi_bs_per_bs_evidence_array_reconciles_every_row(tmp_path):
    artifact, state, track, _, per_bs = _multi_bs_fixture(tmp_path)
    track.evidence["e2e"]["per_bs"] = per_bs

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_multi_bs_per_bs_evidence_row_tampering_is_caught(tmp_path):
    artifact, state, track, _, per_bs = _multi_bs_fixture(tmp_path)
    per_bs[-1]["optimized_avg_s"] = 0.001
    track.evidence["e2e"]["per_bs"] = per_bs

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("per_bs optimized_avg_s disagrees" in item
               for item in result.evidence)


def test_multi_bs_per_bs_evidence_must_cover_every_sweep_row(tmp_path):
    artifact, state, track, _, per_bs = _multi_bs_fixture(tmp_path)
    track.evidence["e2e"]["per_bs"] = per_bs[:-1]

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("does not cover the official sweep batch sizes" in item
               for item in result.evidence)


def test_absent_per_bs_evidence_array_stays_optional(tmp_path):
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    assert "per_bs" not in track.evidence["e2e"]

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_single_bs_reconciliation_still_fires(tmp_path):
    artifact, state, track, *_ = _canonical_validation_fixture(tmp_path)
    track.evidence["e2e"]["baseline_avg_s"] = 42.0

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("baseline_avg_s disagrees" in item for item in result.evidence)


def test_multi_bs_kernel_bench_buckets_must_cover_target_batches(tmp_path):
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    track.evidence["kernel_bench"] = {
        "status": "PASS", "weighted_speedup": 1.05,
        "measured_under_cuda_graphs": True,
        "buckets": [{"batch_size": 1, "speedup": 1.02}],
    }

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "FAIL"
    assert any("buckets do not cover target batch size(s) 4, 8" in item
               for item in result.evidence)


def test_multi_bs_kernel_bench_full_bucket_coverage_passes(tmp_path):
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    track.evidence["kernel_bench"] = {
        "status": "PASS", "weighted_speedup": 1.05,
        "measured_under_cuda_graphs": True,
        "buckets": [{"batch_size": bs, "speedup": 1.02} for bs in (1, 4, 8)],
    }

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_shape_only_kernel_bench_buckets_are_not_batch_checked(tmp_path):
    """A microbenchmark keyed on shapes alone is legitimate; do not false-fail."""
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    track.evidence["kernel_bench"] = {
        "status": "PASS", "weighted_speedup": 1.05,
        "measured_under_cuda_graphs": True,
        "buckets": [{"label": "gate_up_proj", "warm_speedup": 1.2},
                    {"label": "down_proj", "warm_speedup": 1.1}],
    }

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_skipped_kernel_bench_is_not_batch_checked(tmp_path):
    artifact, state, track, *_ = _multi_bs_fixture(tmp_path)
    track.evidence["kernel_bench"] = {
        "status": "SKIPPED", "reason": "component wall time is binding",
        "buckets": [{"batch_size": 1}],
    }

    result = verify_validation_gates.check_canonical_validation_artifacts(
        track, artifact, state)

    assert result.status == "PASS", result.evidence


def test_passing_track_cannot_treat_failed_correctness_or_kernel_as_complete_pass():
    metadata = _track("PASS")
    track = _track_context(
        metadata,
        {
            "correctness": _complete_correctness("FAIL"),
            "kernel_bench": _complete_kernel_bench("FAIL"),
        },
    )

    assert verify_validation_gates.check_correctness(track).status == "FAIL"
    assert verify_validation_gates.check_kernel_bench(track).status == "FAIL"


def test_passing_kernel_changing_track_requires_both_inline_gates_pass():
    metadata = _track("PASS")
    metadata["gate_5_1a"] = "FAIL"
    track = _track_context(metadata, {})

    result = verify_validation_gates.check_inline_kernel_gates(track)

    assert result.status == "FAIL"
    assert "Gate 5.1a and Gate 5.2 PASS" in result.message


def test_pure_inter_kernel_skips_inline_gates_but_keeps_binding_correctness():
    metadata = _track("PASS")
    metadata.update(
        {
            "gate_5_1a": "SKIPPED",
            "gate_5_1a_skip_reason": "pure inter-kernel scheduling change; no kernel authored",
            "gate_5_2": "SKIPPED",
            "gate_5_2_skip_reason": "component wall time is the binding metric",
            "correctness": True,
        }
    )
    track = _track_context(
        metadata,
        {
            "correctness": {
                "status": "SKIPPED",
                "reason": "pure inter-kernel scheduling change; no kernel authored",
            },
            "kernel_bench": {
                "status": "SKIPPED",
                "reason": "component wall time is the binding metric",
            },
            "e2e": {"status": "PASS"},
        },
    )

    assert verify_validation_gates.check_inline_kernel_gates(track).status == "PASS"
    assert verify_validation_gates.check_correctness(track).status == "PASS"
    assert verify_validation_gates.check_kernel_bench(track).status == "PASS"
    assert verify_validation_gates.check_candidate_outcome(track).status == "PASS"

    metadata["correctness"] = False
    assert verify_validation_gates.check_track_state(track).status == "FAIL"


def test_failed_track_complete_fail_evidence_is_not_mislabeled_incomplete():
    metadata = _track("FAIL")
    metadata.update({"correctness": False, "gate_5_1a": "FAIL", "gate_5_2": "FAIL"})
    track = _track_context(
        metadata,
        {
            "correctness": _complete_correctness("FAIL"),
            "kernel_bench": _complete_kernel_bench("FAIL"),
        },
    )

    assert verify_validation_gates.check_correctness(track).status == "PASS"
    assert verify_validation_gates.check_kernel_bench(track).status == "PASS"
    assert verify_validation_gates.check_inline_kernel_gates(track).status == "PASS"


def _valid_gating() -> dict:
    return {
        "mechanism": "env_guarded_dispatch",
        "env_var": "VLLM_OP123",
        "dispatch_condition": "batch_size >= crossover_threshold_bs",
        "crossover_threshold_bs": 8,
        "crossover_probing": {"status": "PASS", "batch_sizes": [1, 8, 32]},
        "pre_gating_results": {"bs1": "baseline path"},
        "post_gating_results": {
            "bs32": {"status": "PASS", "improvement_pct": 1.25},
        },
    }


def test_gated_pass_requires_verdict_and_gating_metadata():
    track = _track_context({"status": "GATED_PASS"}, _passing_evidence())

    result = verify_validation_gates.check_candidate_outcome(track)

    assert result.status == "FAIL"
    assert "metadata.verdict" in result.message


def test_gpu_blocked_track_state_report_does_not_call_it_terminal():
    track = _track_context(_track("GPU_BLOCKED"), {"verdict": "GPU_BLOCKED"})

    result = verify_validation_gates.check_track_state(track)

    assert result.status == "PASS"
    assert "recorded GPU blocker" in result.message
    assert "recorded blocker status: GPU_BLOCKED" in result.evidence
    assert "terminal status: GPU_BLOCKED" not in result.evidence


def test_gated_pass_requires_structured_gating_fields():
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": {"mechanism": "env_guarded_dispatch"},
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(track)

    assert result.status == "FAIL"
    assert "metadata.gating" in result.message
    assert "env_var" in result.evidence
    assert "post_gating_results" in result.evidence


def test_gated_pass_with_structured_gating_metadata_passes():
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": _valid_gating(),
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(
        track,
        {"campaign": {"config": {"min_e2e_improvement_pct": 1.0}}},
    )

    assert result.status == "PASS"


def test_gated_pass_requires_post_gating_improvement_at_threshold():
    gating = _valid_gating()
    gating["post_gating_results"] = {
        "bs8": {"status": "PASS", "improvement_pct": 0.9},
        "bs32": {"status": "NOISE", "improvement_pct": 0.8},
    }
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": gating,
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(
        track,
        {"campaign": {"config": {"min_e2e_improvement_pct": 1.0}}},
    )

    assert result.status == "FAIL"
    assert "post-gating batch" in result.message
    assert "threshold=1.000" in result.evidence


def test_gated_pass_malformed_legacy_fallback_remains_quarter_percent():
    gating = _valid_gating()
    gating["post_gating_results"] = {
        "bs8": {"status": "PASS", "improvement_pct": 0.4},
    }
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": gating,
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(track, {"campaign": {}})

    assert result.status == "PASS"


def test_gated_pass_malformed_legacy_below_quarter_percent_still_fails():
    gating = _valid_gating()
    gating["post_gating_results"] = {
        "bs8": {"status": "PASS", "improvement_pct": 0.2},
    }
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": gating,
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(track, {"campaign": {}})

    assert result.status == "FAIL"
    assert "threshold=0.250" in result.evidence


def test_gated_pass_rejects_any_failed_post_gating_batch():
    gating = _valid_gating()
    gating["post_gating_results"] = {
        "bs8": {"status": "PASS", "improvement_pct": 1.3},
        "bs32": {"status": "REGRESSED", "improvement_pct": -0.2},
    }
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": gating,
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(
        track,
        {"campaign": {"config": {"min_e2e_improvement_pct": 1.0}}},
    )

    assert result.status == "FAIL"
    assert "all reported batch sizes" in result.message
    assert any("REGRESSED" in item for item in result.evidence)


def test_gated_pass_rejects_failed_post_gating_batch_without_improvement_pct():
    gating = _valid_gating()
    gating["post_gating_results"] = {
        "bs8": {"status": "PASS", "improvement_pct": 1.3},
        "bs32": {"verdict": "REGRESSED", "speedup": 0.98},
    }
    track = _track_context(
        {
            "status": "GATED_PASS",
            "verdict": "GATED_PASS",
            "gating": gating,
        },
        _passing_evidence(),
    )

    result = verify_validation_gates.check_candidate_outcome(
        track,
        {"campaign": {"config": {"min_e2e_improvement_pct": 1.0}}},
    )

    assert result.status == "FAIL"
    assert any("REGRESSED" in item and "MISSING" in item for item in result.evidence)


def test_state_tracks_use_only_current_round():
    state = {
        "campaign": {
            "current_round": 2,
            "rounds": [
                {"round_id": 1, "parallel_tracks": {"tracks": {"op_old": _track("FAIL")}}},
                {"round_id": 2, "parallel_tracks": {"tracks": {"op_new": _track("PASS")}}},
            ],
        }
    }

    tracks = verify_validation_gates._state_tracks(state)

    assert set(tracks) == {"op_new"}


def test_cohort_verifier_rejects_missing_track_and_pairing(tmp_path):
    state = {
        "campaign": {
            "schema_version": "4.2",
            "current_round": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "debate": {
                        "selected_winners": ["op_new", "op_two"],
                        "selected_candidates": [
                            {"op_id": "op_new"}, {"op_id": "op_two"}
                        ],
                    },
                    "parallel_tracks": {"tracks": {}},
                }
            ],
        }
    }

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, None)
    except ValueError as exc:
        assert "cohort mismatch" in str(exc)
    else:
        raise AssertionError("selected winner without a track must fail")

    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op_new": _track("PASS"),
        "op_two": _track("FAIL"),
    }
    for op_id in ("op_new", "op_two"):
        metadata = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"][op_id]
        metadata["evidence_path"] = f"rounds/1/tracks/{op_id}/evidence.json"
        metadata["validation_results_path"] = (
            f"rounds/1/tracks/{op_id}/validation_results.md"
        )
    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, None)
    except ValueError as exc:
        assert "pairing evidence" in str(exc)
    else:
        raise AssertionError("selected track without monitor pairing must fail")

    for op_id in ("op_new", "op_two"):
        monitor_dir = tmp_path / "rounds" / "1" / "tracks" / op_id / "monitor_audits"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        (monitor_dir / "obs.md").write_text("observed\n", encoding="utf-8")
        (monitor_dir / "offsets.json").write_text(
            json.dumps(
                {
                    "target_agent": f"/root/impl_{op_id}",
                    "target_rollout_id": f"rollout_{op_id}",
                    "target_transcript_path": f"/{op_id}.jsonl",
                    "last_offset": 10,
                }
            ),
            encoding="utf-8",
        )
        (monitor_dir / "summary.json").write_text(
            json.dumps(
                {
                    "monitor_agent": f"/root/monitor_{op_id}",
                    "target_agent": f"/root/impl_{op_id}",
                    "target_rollout_id": f"rollout_{op_id}",
                    "coverage_status": "TRACK_COMPLETE",
                }
            ),
            encoding="utf-8",
        )
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"][op_id].update(
            {
                "implementer_agent": f"/root/impl_{op_id}",
                "implementer_rollout_id": f"rollout_{op_id}",
                "monitor_agent": f"/root/monitor_{op_id}",
                "monitor_evidence_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/obs.md"
                ),
                "monitor_offsets_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/offsets.json"
                ),
                "monitor_summary_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/summary.json"
                ),
            }
        )
    resolved = verify_validation_gates._resolve_tracks(tmp_path, state, None)
    assert {track.track_id for track in resolved} == {"op_new", "op_two"}


def test_contingent_boundary_recomputes_e2e_equivalent_improvement():
    metadata = _track("PASS")
    metadata.update(
        {
            "gate_5_2": "PASS",
            "gate_5_2_boundary": {
                "baseline_duration_us": 100.0,
                "optimized_duration_us": 90.0,
                "occurrence_count": 1000,
                "baseline_e2e_us": 2_000_000.0,
                "e2e_equivalent_improvement_pct": 0.5,
                "campaign_floor_pct": 0.5,
                "meets_floor": True,
            },
        }
    )
    track = _track_context(metadata, _passing_evidence())
    state = {
        "campaign": {
            "current_round": 1,
            "config": {"min_e2e_improvement_pct": 0.5},
            "rounds": [
                {
                    "debate": {
                        "selected_candidates": [
                            {
                                "op_id": track.track_id,
                                "selection_mode": "contingent_host_spike",
                            }
                        ]
                    }
                }
            ],
        }
    }
    assert verify_validation_gates.check_contingent_boundary(track, state).status == "PASS"
    track.metadata["gate_5_2_boundary"]["occurrence_count"] = 100
    assert verify_validation_gates.check_contingent_boundary(track, state).status == "FAIL"


def test_track_lookup_rejects_stale_previous_round_track(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "current_round": 2,
                    "rounds": [
                        {"round_id": 1, "parallel_tracks": {"tracks": {"op_old": _track("FAIL")}}},
                        {"round_id": 2, "parallel_tracks": {"tracks": {"op_new": _track("PASS")}}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, "op_old")
    except KeyError as exc:
        assert "op_old" in str(exc)
    else:
        raise AssertionError("stale previous-round track should not resolve")


def test_round_scoped_track_paths_are_default_for_current_round(tmp_path):
    state = {
        "campaign": {
            "current_round": 2,
            "rounds": [
                {"round_id": 1, "parallel_tracks": {"tracks": {}}},
                {"round_id": 2, "parallel_tracks": {"tracks": {"op_new": _track("PASS")}}},
            ],
        }
    }
    state["campaign"]["rounds"][1]["parallel_tracks"]["tracks"]["op_new"].pop("evidence_path")
    state["campaign"]["rounds"][1]["parallel_tracks"]["tracks"]["op_new"].pop("validation_results_path")

    tracks = verify_validation_gates._resolve_tracks(tmp_path, state, "op_new")

    assert tracks[0].evidence_path == (tmp_path / "rounds/2/tracks/op_new/evidence.json").resolve()
    assert tracks[0].validation_path == (
        tmp_path / "rounds/2/tracks/op_new/validation_results.md"
    ).resolve()


def test_flat_parallel_tracks_legacy_fallback_still_works():
    state = {"parallel_tracks": {"op_legacy": _track("PASS")}}

    assert set(verify_validation_gates._state_tracks(state)) == {"op_legacy"}


def test_malformed_round_centric_state_does_not_use_flat_fallback():
    state = {
        "campaign": {
            "current_round": 2,
            "rounds": [{"round_id": 1, "parallel_tracks": {"tracks": {"op_old": _track("FAIL")}}}],
        },
        "parallel_tracks": {"op_stale_flat": _track("PASS")},
    }

    assert verify_validation_gates._state_tracks(state) == {}


def test_partial_v2_campaign_does_not_use_flat_fallback(tmp_path):
    state = {
        "campaign": {"status": "active"},
        "parallel_tracks": {"op_stale_flat": _track("PASS")},
    }
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_results.md").write_text("legacy", encoding="utf-8")

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, None)
    except ValueError as exc:
        assert "Round-centric" in str(exc)
    else:
        raise AssertionError("partial V2 campaign must not consume flat-root evidence")


def test_v2_empty_current_round_does_not_fall_back_to_root_evidence(tmp_path):
    state = {
        "campaign": {
            "schema_version": "4.1",
            "current_round": 2,
            "rounds": [{"round_id": 1}, {"round_id": 2, "parallel_tracks": {"tracks": {}}}],
        }
    }
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_results.md").write_text("legacy", encoding="utf-8")

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, None)
    except ValueError as exc:
        assert "current round" in str(exc)
    else:
        raise AssertionError("V2 state must not consume flat-root evidence")


def test_v2_explicit_prior_round_evidence_path_is_rejected(tmp_path):
    state = {
        "campaign": {
            "schema_version": "4.1",
            "current_round": 2,
            "rounds": [
                {"round_id": 1},
                {
                    "round_id": 2,
                    "parallel_tracks": {
                        "tracks": {
                            "op_new": {
                                **_track("PASS"),
                                "evidence_path": "rounds/1/tracks/op_new/evidence.json",
                                "validation_results_path": "rounds/2/tracks/op_new/validation_results.md",
                            }
                        }
                    },
                },
            ],
        }
    }

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, "op_new")
    except ValueError as exc:
        assert "prior round" in str(exc)
    else:
        raise AssertionError("current-round verification must reject prior-round evidence")


def test_v2_explicit_root_or_cross_track_evidence_paths_are_rejected(tmp_path):
    traversal = tmp_path / "rounds" / "2" / "tracks" / "op_new" / ".." / ".." / ".." / ".." / "evidence.json"
    for evidence_path in (
        "evidence.json",
        "rounds/2/tracks/op_other/evidence.json",
        str(tmp_path.parent / "external-evidence.json"),
        str(traversal),
    ):
        state = {
            "campaign": {
                "schema_version": "4.1",
                "current_round": 2,
                "rounds": [
                    {"round_id": 1},
                    {
                        "round_id": 2,
                        "parallel_tracks": {
                            "tracks": {
                                "op_new": {
                                    **_track("PASS"),
                                    "evidence_path": evidence_path,
                                    "validation_results_path": "rounds/2/tracks/op_new/validation_results.md",
                                }
                            }
                        },
                    },
                ],
            }
        }

        try:
            verify_validation_gates._resolve_tracks(tmp_path, state, "op_new")
        except ValueError as exc:
            assert "current-round track directory" in str(exc)
        else:
            raise AssertionError(f"V2 path escaped current track: {evidence_path}")


def test_v2_symlinked_evidence_cannot_escape_current_track(tmp_path):
    current_dir = tmp_path / "rounds" / "2" / "tracks" / "op_new"
    prior_dir = tmp_path / "rounds" / "1" / "tracks" / "op_new"
    current_dir.mkdir(parents=True)
    prior_dir.mkdir(parents=True)
    prior_evidence = prior_dir / "evidence.json"
    prior_evidence.write_text("{}", encoding="utf-8")
    (current_dir / "evidence.json").symlink_to(prior_evidence)
    state = {
        "campaign": {
            "schema_version": "4.1",
            "current_round": 2,
            "rounds": [
                {"round_id": 1},
                {
                    "round_id": 2,
                    "parallel_tracks": {
                        "tracks": {
                            "op_new": {
                                **_track("PASS"),
                                "evidence_path": "rounds/2/tracks/op_new/evidence.json",
                                "validation_results_path": "rounds/2/tracks/op_new/validation_results.md",
                            }
                        }
                    },
                },
            ],
        }
    }

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, "op_new")
    except ValueError as exc:
        assert "prior round" in str(exc) or "current-round track directory" in str(exc)
    else:
        raise AssertionError("symlinked prior-round evidence must be rejected")


def test_cli_reports_stale_evidence_error_without_traceback(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "schema_version": "4.1",
                    "current_round": 2,
                    "rounds": [
                        {"round_id": 1},
                        {
                            "round_id": 2,
                            "parallel_tracks": {
                                "tracks": {
                                    "op_new": {
                                        **_track("PASS"),
                                        "evidence_path": "rounds/1/tracks/op_new/evidence.json",
                                        "validation_results_path": "rounds/2/tracks/op_new/validation_results.md",
                                    }
                                }
                            },
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path), "--track", "op_new"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "ERROR:" in result.stderr
    assert "prior round" in result.stderr
    assert "Traceback" not in result.stderr


def test_true_legacy_root_evidence_context_still_resolves(tmp_path):
    state = {"route_decision": {}, "parallel_tracks": {}}
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_results.md").write_text("legacy", encoding="utf-8")

    tracks = verify_validation_gates._resolve_tracks(tmp_path, state, None)

    assert len(tracks) == 1
    assert tracks[0].track_id == "root"
    assert tracks[0].evidence_path == tmp_path / "evidence.json"


def test_empty_state_cannot_fall_back_to_root_evidence(tmp_path):
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_results.md").write_text("legacy", encoding="utf-8")

    try:
        verify_validation_gates._resolve_tracks(tmp_path, {}, None)
    except ValueError as exc:
        assert "explicit legacy" in str(exc)
    else:
        raise AssertionError("empty state must not consume flat-root evidence")


def test_v2_rejects_track_id_path_traversal(tmp_path):
    malicious = "../../../tracks/op"
    state = {
        "campaign": {
            "schema_version": "4.1",
            "current_round": 2,
            "rounds": [
                {},
                {"parallel_tracks": {"tracks": {malicious: _track("PASS")}}},
            ],
        }
    }

    try:
        verify_validation_gates._resolve_tracks(tmp_path, state, malicious)
    except ValueError as exc:
        assert "Invalid track id" in str(exc)
    else:
        raise AssertionError("track id traversal must be rejected")


def test_gpu_blocked_is_not_terminal_track_status():
    assert "GPU_BLOCKED" not in verify_validation_gates.TERMINAL_TRACK_STATUSES


def test_gpu_blocked_outcome_blocks_stage6_even_with_failures():
    report = verify_validation_gates.VerificationReport(artifact_dir="artifact")
    report.track_outcomes = {"op_fail": "FAIL", "op_blocked": "GPU_BLOCKED"}

    verify_validation_gates.finalize_report_status(report)

    assert report.overall_status == "GPU_BLOCKED"
    assert report.advance_to_stage6 is False
    assert "Lead triage" in report.recommendation


def test_stage1_baseline_reuse_reads_round_scoped_current_round(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "rounds" / "2" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    track = _track_context(
        {"status": "PASS"},
        {"baseline_source": {"kind": "stage1", "citation": "Stage 1 baseline not re-run"}},
    )

    result = verify_validation_gates.check_stage1_baseline(track, tmp_path)

    assert result.status == "PASS"
    assert result.evidence == ["rounds/2/sweeps/baseline/json/baseline_bs1.json"]


def test_current_evidence_requires_stage1_reuse_provenance(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "rounds" / "2" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    opt_sweep = tmp_path / "rounds" / "2" / "sweeps" / "opt" / "OP-X" / "e2e_latency_results.json"
    opt_sweep.parent.mkdir(parents=True)
    opt_sweep.write_text("{}", encoding="utf-8")
    track = _track_context(
        {"status": "PASS"},
        {
            "schema_version": 3,
            "baseline_source": {
                "kind": "stage1",
                "citation": "Baseline source: Stage 1 (not re-run)",
                "sweep_result": str(opt_sweep.relative_to(tmp_path)),
            },
        },
    )

    result = verify_validation_gates.check_stage1_baseline(track, tmp_path)

    assert result.status == "PASS"
    assert result.name == "stage1_baseline_reuse"


def test_current_evidence_rejects_worktree_paired_ab(tmp_path):
    """Reverted contract: a worktree-run paired A/B is NOT accepted — the
    baseline arm can execute the optimized code path (editable install)."""
    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "rounds" / "2" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    opt_sweep = tmp_path / "rounds" / "2" / "sweeps" / "opt" / "OP-X" / "e2e_latency_results.json"
    opt_sweep.parent.mkdir(parents=True)
    opt_sweep.write_text("{}", encoding="utf-8")
    track = _track_context(
        {"status": "PASS"},
        {
            "schema_version": 3,
            "baseline_source": {
                "kind": "paired_ab",
                "citation": "adjacent matched arms",
                "sweep_result": str(opt_sweep.relative_to(tmp_path)),
            },
        },
    )

    result = verify_validation_gates.check_stage1_baseline(track, tmp_path)

    assert result.status == "FAIL"
    assert result.name == "stage1_baseline_reuse"


def test_current_template_passes_structured_and_stage1_provenance_checks(tmp_path):
    template_script = (
        Path(__file__).resolve().parents[1] / "scripts" / "create_evidence_template.py"
    )
    template_spec = importlib.util.spec_from_file_location(
        "create_evidence_template", str(template_script)
    )
    template_module = importlib.util.module_from_spec(template_spec)
    assert template_spec.loader is not None
    template_spec.loader.exec_module(template_module)

    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 1, "rounds": [{}]}}),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "rounds" / "1" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    opt_sweep = tmp_path / "rounds" / "1" / "sweeps" / "opt" / "op_test" / "e2e_latency_results.json"
    opt_sweep.parent.mkdir(parents=True)
    opt_sweep.write_text("{}", encoding="utf-8")
    evidence = template_module.template("op_test")
    evidence["baseline_source"].update(
        {
            "sweep_result": str(opt_sweep.relative_to(tmp_path)),
        }
    )
    track = _track_context({"status": "PASS"}, evidence)

    assert verify_validation_gates.check_structured_evidence(track).status == "PASS"
    assert verify_validation_gates.check_stage1_baseline(track, tmp_path).status == "PASS"


def test_current_evidence_requires_bound_opt_sweep_result(tmp_path):
    """Stage 1 reuse citation alone is not enough — the evidence must bind the
    actual opt sweep result file that was compared against the baseline."""
    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 1, "rounds": [{}]}}),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "rounds" / "1" / "sweeps" / "baseline" / "json"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    track = _track_context(
        {"status": "PASS"},
        {
            "schema_version": 3,
            "baseline_source": {
                "kind": "stage1",
                "citation": "Stage 1 baseline not re-run",
            },
        },
    )

    result = verify_validation_gates.check_stage1_baseline(track, tmp_path)

    assert result.status == "FAIL"
    assert result.name == "stage1_baseline_reuse"


def test_stage1_baseline_reuse_ignores_stale_legacy_runs_when_round_missing(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    legacy_dir = tmp_path / "runs"
    legacy_dir.mkdir()
    (legacy_dir / "baseline_bs1.json").write_text("{}", encoding="utf-8")
    track = _track_context(
        {"status": "PASS"},
        {"baseline_source": {"kind": "stage1", "citation": "Stage 1 baseline not re-run"}},
    )

    result = verify_validation_gates.check_stage1_baseline(track, tmp_path)

    assert result.status == "FAIL"
    assert any("rounds/2/sweeps/baseline/json/baseline_bs*.json" in item for item in result.evidence)
