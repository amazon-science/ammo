#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for round/slot E2E lookup in generate_validation_report.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_GEN = _SCRIPTS_DIR / "generate_validation_report.py"
_EVIDENCE = _SCRIPTS_DIR / "create_evidence_template.py"
_AMMO_DIR = Path(__file__).resolve().parent.parent


def _e2e_payload(speedup: float = 1.05) -> str:
    return json.dumps({
        "model_id": "test/model",
        "tp": 1,
        "max_model_len": 4096,
        "workload": {"input_len": 64, "output_len": 64, "num_iters": 1},
        "results": [{
            "batch_size": 1,
            "speedup": speedup,
            "improvement_pct": (speedup - 1) * 100,
            "baseline": {
                "avg_s": 1.0,
                "cmd": ["python", "bench.py", "--label", "baseline"],
                "env_overrides": {"CUDA_VISIBLE_DEVICES": "0"},
            },
            "opt": {
                "avg_s": 1.0 / speedup,
                "cmd": ["python", "bench.py", "--label", "opt"],
                "env_overrides": {"CUDA_VISIBLE_DEVICES": "0"},
            },
        }],
        "bench": {"baseline_label": "baseline", "opt_label": "opt"},
    })


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_GEN), *args], capture_output=True, text=True)


def test_e2e_guide_tool_selection_preserves_stage1_and_stage6_control_shape():
    guide = (_AMMO_DIR / "references" / "e2e-latency-guide.md").read_text(encoding="utf-8")
    table = guide.split("## Tool Selection", 1)[1].split("For all measurements", 1)[0]

    assert "Stage 1 (baseline)" in table
    assert "--round {N} --slot baseline --labels baseline --capture-golden-refs" in table
    assert "Stage 6 (integration)" in table
    assert "--round {N} --slot integration --labels opt --baseline-from $STAGE1_DIR --fresh-cache" in table
    assert "two or more tracks pass" in table
    assert "single unchanged passer uses the short-circuit" in table

    recording = guide.split("## Recording results", 1)[1]
    assert "{artifact_dir}/rounds/{CR}/tracks/{op_id}/validation_results.md" in recording
    assert "{artifact_dir}/rounds/{CR}/sweeps/integration/e2e_latency_results.json" in recording
    assert "integration_validation.md" not in recording
    assert "{artifact_dir}/validation_results.md" not in recording


def test_validation_defaults_use_round_scoped_validation_results_path():
    defaults = (_AMMO_DIR / "references" / "validation-defaults.md").read_text(encoding="utf-8")

    assert "{artifact_dir}/rounds/{N}/tracks/{op_id}/validation_results.md" in defaults
    assert "{artifact_dir}/validation_results.md" not in defaults


def test_round_and_slot_resolves_e2e_path(tmp_path: Path):
    artifact = tmp_path / "v2_artifact"
    rd = artifact / "rounds" / "1" / "sweeps" / "baseline"
    rd.mkdir(parents=True)
    (rd / "e2e_latency_results.json").write_text(_e2e_payload(speedup=1.10), encoding="utf-8")

    result = _run(["--artifact-dir", str(artifact), "--round", "1", "--slot", "baseline"])

    assert result.returncode == 0, f"failed: {result.stdout} / {result.stderr}"
    out = (artifact / "validation_results.md").read_text(encoding="utf-8")
    assert "1.1" in out or "10%" in out
    assert "Repro (baseline example):" in out
    assert "CUDA_VISIBLE_DEVICES=0 python bench.py --label baseline" in out
    assert "Repro (optimized example):" in out
    assert "CUDA_VISIBLE_DEVICES=0 python bench.py --label opt" in out


def test_legacy_path_still_works(tmp_path: Path):
    artifact = tmp_path / "v1_artifact"
    e2e_dir = artifact / "e2e_latency"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "e2e_latency_results.json").write_text(_e2e_payload(speedup=1.20), encoding="utf-8")

    result = _run(["--artifact-dir", str(artifact)])

    assert result.returncode == 0, f"failed: {result.stdout} / {result.stderr}"
    out = (artifact / "validation_results.md").read_text(encoding="utf-8")
    assert "1.2" in out or "20%" in out


def test_artifact_dir_mode_reports_malformed_json_without_traceback(tmp_path: Path):
    artifact = tmp_path / "bad_artifact"
    e2e_dir = artifact / "e2e_latency"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "e2e_latency_results.json").write_text("{bad json", encoding="utf-8")

    result = _run(["--artifact-dir", str(artifact)])

    assert result.returncode != 0
    assert "Failed to parse JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_create_evidence_template_defaults_to_current_round_track_path(tmp_path: Path):
    artifact = tmp_path / "v2_artifact"
    (artifact / "rounds" / "2").mkdir(parents=True)
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_EVIDENCE),
            "--artifact-dir",
            str(artifact),
            "--track-id",
            "op007",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"failed: {result.stdout} / {result.stderr}"
    assert (artifact / "rounds" / "2" / "tracks" / "op007" / "evidence.json").exists()
    assert not (artifact / "tracks" / "op007" / "evidence.json").exists()


def test_artifact_dir_track_report_defaults_to_current_round_track_path(tmp_path: Path):
    artifact = tmp_path / "v2_artifact"
    track_dir = artifact / "rounds" / "2" / "tracks" / "op007"
    track_dir.mkdir(parents=True)
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    (track_dir / "evidence.json").write_text(
        json.dumps({"track_id": "op007", "baseline_source": {"citation": "stage1"}}),
        encoding="utf-8",
    )

    result = _run(["--artifact-dir", str(artifact), "--track", "op007"])

    assert result.returncode == 0, f"failed: {result.stdout} / {result.stderr}"
    assert (track_dir / "validation_results.md").exists()
    assert not (artifact / "tracks" / "op007" / "validation_results.md").exists()
