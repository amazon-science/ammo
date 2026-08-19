# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
RED phase tests for S5: benchmark harness missing baselines.

Bug: Gate runs (Stage 5 validation) use --labels opt, which correctly runs
only the opt child. But the results JSON ends up with baseline.avg_s: null
because there's no mechanism to import Stage 1 baseline data.

Design: baseline is measured ONLY in Stage 1. Gate runs (Stage 5) must NOT
re-run baselines. Instead, the harness needs a --baseline-from flag to
import Stage 1 baseline artifacts into the gate run's output.

Tests cover:
1. --baseline-from imports Stage 1 baseline data into gate results
2. --baseline-from with missing/invalid path fails fast
3. --labels opt without --baseline-from emits warnings in results
4. --labels baseline (Stage 1) is unaffected by --baseline-from
"""

import json
import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# Add project root to path.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SWEEP_SCRIPT = (
    PROJECT_ROOT
    / "ai_cli_session"
    / ".claude"
    / "skills"
    / "ammo"
    / "scripts"
    / "run_vllm_bench_latency_sweep.py"
)


def _import_sweep():
    """Import the sweep script as a proper module (handles @dataclass)."""
    module_name = "run_vllm_bench_latency_sweep"
    spec = importlib.util.spec_from_file_location(module_name, str(SWEEP_SCRIPT))
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = sweep
    spec.loader.exec_module(sweep)
    return sweep


def _make_target_json(artifact_dir: Path) -> Path:
    """Create a minimal valid target.json for testing."""
    target = {
        "artifact_dir": str(artifact_dir),
        "target": {
            "model_id": "test-model/test",
            "dtype": "fp16",
            "tp": 1,
            "ep": 1,
            "max_model_len": 4096,
        },
        "workload": {
            "input_len": 64,
            "output_len": 128,
            "batch_sizes": [1],
            "num_iters": 2,
        },
        "bench": {
            "runner": "vllm_bench_latency",
            "vllm_cmd": "vllm",
            "extra_args": [],
            "baseline_extra_args": [],
            "opt_extra_args": ["--some-opt-flag"],
            "baseline_env": {},
            "opt_env": {"ENABLE_OPT": "1"},
            "baseline_label": "baseline",
            "opt_label": "opt",
            "fastpath_evidence": {
                "baseline": {"require_patterns": [], "forbid_patterns": []},
                "opt": {"require_patterns": [], "forbid_patterns": []},
            },
        },
    }
    target_path = artifact_dir / "target.json"
    target_path.write_text(json.dumps(target, indent=2))
    return target_path


def _fake_vllm_bench_json(avg_latency: float = 0.05) -> Dict[str, Any]:
    return {
        "avg_latency": avg_latency,
        "latencies": [avg_latency] * 2,
        "percentiles": {
            "10": avg_latency,
            "50": avg_latency,
            "90": avg_latency,
            "99": avg_latency,
        },
    }


def _fake_runner_json(ok: bool = True) -> Dict[str, Any]:
    return {
        "ok": ok,
        "label": "test",
        "batch_size": 1,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:01:00Z",
        "duration_s": 60.0,
    }


def _extract_child_label(cmd: List[str]) -> str | None:
    for i, arg in enumerate(cmd):
        if arg == "--_child-label" and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _extract_out_root(cmd: List[str]) -> Path | None:
    for i, arg in enumerate(cmd):
        if arg == "--_out-root" and i + 1 < len(cmd):
            return Path(cmd[i + 1])
    return None


def _make_capture_side_effect(child_labels_spawned: List[str]):
    """Side effect that captures child labels and writes opt artifacts only."""
    def side_effect(cmd, *, env, cwd, timeout_s, log_path, heartbeat_s, status_path=None):
        label = _extract_child_label(cmd)
        out_root = _extract_out_root(cmd)
        if label:
            child_labels_spawned.append(label)
        if out_root and label:
            json_dir = out_root / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            tag = "bs1"
            avg = 0.05 if label == "baseline" else 0.04
            (json_dir / f"{label}_{tag}.json").write_text(
                json.dumps(_fake_vllm_bench_json(avg))
            )
            (json_dir / f"{label}_{tag}.runner.json").write_text(
                json.dumps(_fake_runner_json(ok=True))
            )
        return {"ok": True, "returncode": 0}
    return side_effect


def _create_stage1_baseline(artifact_dir: Path, baseline_dir_name: str = "e2e_baseline") -> Path:
    """Create a fake Stage 1 baseline output directory with baseline artifacts."""
    baseline_dir = artifact_dir / baseline_dir_name
    json_dir = baseline_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # Write baseline artifacts for batch_size=1 (tag="bs1").
    (json_dir / "baseline_bs1.json").write_text(
        json.dumps(_fake_vllm_bench_json(0.05))
    )
    (json_dir / "baseline_bs1.runner.json").write_text(
        json.dumps(_fake_runner_json(ok=True))
    )
    return baseline_dir


def _run_sweep_main(sweep, artifact_dir: Path, labels: str, side_effect_fn,
                    extra_args: List[str] | None = None):
    """Run sweep.main() with patched args and _run_cmd_streaming."""
    test_argv = [
        "run_vllm_bench_latency_sweep.py",
        "--artifact-dir", str(artifact_dir),
        "--labels", labels,
        "--overwrite",
    ]
    if extra_args:
        test_argv.extend(extra_args)

    with (
        patch.object(sys, "argv", test_argv),
        patch.object(sweep, "_run_cmd_streaming", side_effect=side_effect_fn),
        patch("shutil.which", return_value="/usr/bin/vllm"),
    ):
        try:
            sweep.main()
        except SystemExit as e:
            if e.code and e.code != 0:
                raise


def _read_results(artifact_dir: Path) -> Dict[str, Any]:
    """Find and read the e2e_latency_results.json."""
    results_files = list(artifact_dir.rglob("e2e_latency_results.json"))
    assert results_files, "No e2e_latency_results.json found!"
    return json.loads(results_files[0].read_text())


@pytest.fixture(scope="module")
def sweep():
    return _import_sweep()


# ---------------------------------------------------------------------------
# Test 1: --baseline-from imports Stage 1 data into gate results
# ---------------------------------------------------------------------------

class TestBaselineFromImport:
    """When --labels opt and --baseline-from is set, baseline artifacts from
    Stage 1 should be imported so results have non-null baselines."""

    def test_baseline_from_produces_non_null_baselines(self, sweep, tmp_path):
        """Gate run with --baseline-from should have non-null baseline avg_s
        in the results JSON.

        Current: no --baseline-from flag exists → RED (argparse error).
        """
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        # Create Stage 1 baseline artifacts.
        baseline_dir = _create_stage1_baseline(artifact_dir)

        child_labels_spawned: List[str] = []
        side_effect = _make_capture_side_effect(child_labels_spawned)

        _run_sweep_main(
            sweep, artifact_dir, "opt", side_effect,
            extra_args=["--baseline-from", str(baseline_dir)],
        )

        results = _read_results(artifact_dir)

        # Only opt child should be spawned (baseline comes from import).
        assert child_labels_spawned == ["opt"], (
            f"Only opt child should be spawned with --baseline-from, "
            f"got: {child_labels_spawned}"
        )

        # Baseline data should be non-null (imported from Stage 1).
        for row in results["results"]:
            baseline_avg = row.get("baseline", {}).get("avg_s")
            assert baseline_avg is not None, (
                f"Row bs={row['batch_size']} has null baseline avg_s! "
                "--baseline-from should import Stage 1 baseline data."
            )
            assert isinstance(baseline_avg, float)

    def test_baseline_from_populates_speedup(self, sweep, tmp_path):
        """With imported baselines, speedup should be calculated."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)
        baseline_dir = _create_stage1_baseline(artifact_dir)

        child_labels_spawned: List[str] = []
        side_effect = _make_capture_side_effect(child_labels_spawned)

        _run_sweep_main(
            sweep, artifact_dir, "opt", side_effect,
            extra_args=["--baseline-from", str(baseline_dir)],
        )

        results = _read_results(artifact_dir)

        for row in results["results"]:
            assert "speedup" in row, (
                f"Row bs={row['batch_size']} missing speedup. "
                "With both baseline and opt data, speedup should be calculated."
            )
            assert row["speedup"] > 0

    def test_baseline_from_records_source_in_metadata(self, sweep, tmp_path):
        """Results should record where baseline data came from."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)
        baseline_dir = _create_stage1_baseline(artifact_dir)

        side_effect = _make_capture_side_effect([])

        _run_sweep_main(
            sweep, artifact_dir, "opt", side_effect,
            extra_args=["--baseline-from", str(baseline_dir)],
        )

        results = _read_results(artifact_dir)
        assert "baseline_source" in results, (
            "Results should record baseline_source metadata "
            "when --baseline-from is used."
        )
        assert results["baseline_source"] != "none"


# ---------------------------------------------------------------------------
# Test 2: --baseline-from with invalid path fails fast
# ---------------------------------------------------------------------------

class TestBaselineFromFailFast:
    """When --baseline-from points to a non-existent or empty dir, fail fast."""

    def test_nonexistent_baseline_from_path_fails(self, sweep, tmp_path):
        """--baseline-from pointing to nonexistent dir should raise SystemExit."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        side_effect = _make_capture_side_effect([])

        with pytest.raises(SystemExit) as exc_info:
            _run_sweep_main(
                sweep, artifact_dir, "opt", side_effect,
                extra_args=["--baseline-from", str(tmp_path / "nonexistent")],
            )

        # Must be a validation error (not argparse error code 2).
        err = exc_info.value.code
        assert err is not None and err != 0
        assert isinstance(err, str) and "baseline" in err.lower(), (
            f"Expected a baseline-related validation error, got: {err!r}"
        )

    def test_baseline_from_missing_json_files_fails(self, sweep, tmp_path):
        """--baseline-from dir exists but has no baseline JSON files → fail."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        # Create an empty baseline dir (no json/ subdirectory).
        empty_baseline_dir = artifact_dir / "e2e_baseline"
        empty_baseline_dir.mkdir()

        side_effect = _make_capture_side_effect([])

        with pytest.raises(SystemExit) as exc_info:
            _run_sweep_main(
                sweep, artifact_dir, "opt", side_effect,
                extra_args=["--baseline-from", str(empty_baseline_dir)],
            )

        err = exc_info.value.code
        assert err is not None and err != 0
        assert isinstance(err, str) and "baseline" in err.lower(), (
            f"Expected a baseline-related validation error, got: {err!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: --labels opt without --baseline-from emits warnings
# ---------------------------------------------------------------------------

class TestMissingBaselineWarnings:
    """When --labels opt is used without --baseline-from, results should
    include warnings about missing baseline data."""

    def test_labels_opt_without_baseline_from_has_warnings(self, sweep, tmp_path):
        """Results should contain a warnings field when no baseline source."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        side_effect = _make_capture_side_effect([])

        _run_sweep_main(sweep, artifact_dir, "opt", side_effect)

        results = _read_results(artifact_dir)

        assert "warnings" in results and results["warnings"], (
            "Results should contain warnings when --labels opt is used "
            "without --baseline-from. Currently null baselines are silent."
        )

    def test_labels_opt_without_baseline_from_has_baseline_source_none(self, sweep, tmp_path):
        """Results should record baseline_source: 'none'."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        side_effect = _make_capture_side_effect([])

        _run_sweep_main(sweep, artifact_dir, "opt", side_effect)

        results = _read_results(artifact_dir)

        assert results.get("baseline_source") == "none", (
            "Results should set baseline_source='none' when --baseline-from "
            "is not provided with --labels opt."
        )


# ---------------------------------------------------------------------------
# Test 4: --labels baseline (Stage 1) is unaffected
# ---------------------------------------------------------------------------

class TestStage1Unaffected:
    """--labels baseline (Stage 1 mode) must not be changed by the fix."""

    def test_labels_baseline_still_works(self, sweep, tmp_path):
        """Stage 1 (--labels baseline) should work exactly as before."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        child_labels_spawned: List[str] = []
        side_effect = _make_capture_side_effect(child_labels_spawned)

        _run_sweep_main(sweep, artifact_dir, "baseline", side_effect)

        assert child_labels_spawned == ["baseline"]

    def test_labels_baseline_opt_explicit_still_works(self, sweep, tmp_path):
        """--labels baseline,opt should spawn baseline then opt."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)

        child_labels_spawned: List[str] = []
        side_effect = _make_capture_side_effect(child_labels_spawned)

        _run_sweep_main(sweep, artifact_dir, "baseline,opt", side_effect)

        assert child_labels_spawned == ["baseline", "opt"]

    def test_labels_baseline_ignores_baseline_from(self, sweep, tmp_path):
        """--labels baseline should not be affected by --baseline-from."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        _make_target_json(artifact_dir)
        baseline_dir = _create_stage1_baseline(artifact_dir)

        child_labels_spawned: List[str] = []
        side_effect = _make_capture_side_effect(child_labels_spawned)

        _run_sweep_main(
            sweep, artifact_dir, "baseline", side_effect,
            extra_args=["--baseline-from", str(baseline_dir)],
        )

        # Should still just run baseline child normally.
        assert child_labels_spawned == ["baseline"]
