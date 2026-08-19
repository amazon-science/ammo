# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Round-freshness regressions for the structured evidence scaffolder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_evidence_template.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
    )


def test_v2_campaign_rejects_explicit_legacy_root_output(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"schema_version": "4.1", "current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )

    result = _run("--artifact-dir", str(artifact), "--track-id", "op001", "--legacy")

    assert result.returncode != 0
    assert "V2" in result.stderr or "round" in result.stderr
    assert not (artifact / "tracks" / "op001" / "evidence.json").exists()


def test_true_legacy_campaign_can_request_legacy_root_output(tmp_path: Path):
    artifact = tmp_path / "legacy"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"parallel_tracks": {"op001": {"status": "IN_PROGRESS"}}}),
        encoding="utf-8",
    )

    result = _run("--artifact-dir", str(artifact), "--track-id", "op001", "--legacy")

    assert result.returncode == 0, result.stderr
    assert (artifact / "tracks" / "op001" / "evidence.json").exists()


def test_partial_v2_campaign_cannot_fall_back_to_legacy_output(tmp_path: Path):
    artifact = tmp_path / "partial-v2"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps(
            {
                "campaign": {"status": "active"},
                "parallel_tracks": {"op001": {"status": "IN_PROGRESS"}},
            }
        ),
        encoding="utf-8",
    )

    result = _run("--artifact-dir", str(artifact), "--track-id", "op001", "--legacy")

    assert result.returncode != 0
    assert "Round-centric" in result.stderr
    assert not (artifact / "tracks" / "op001" / "evidence.json").exists()


def test_empty_existing_state_cannot_be_treated_as_legacy(tmp_path: Path):
    artifact = tmp_path / "empty"
    artifact.mkdir()
    (artifact / "state.json").write_text("{}", encoding="utf-8")

    result = _run("--artifact-dir", str(artifact), "--track-id", "op001", "--legacy")

    assert result.returncode != 0
    assert "explicit legacy" in result.stderr
    assert not (artifact / "tracks" / "op001" / "evidence.json").exists()


def test_v2_default_uses_state_current_round(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"schema_version": "4.1", "current_round": 3, "rounds": [{}, {}, {}]}}),
        encoding="utf-8",
    )

    result = _run("--artifact-dir", str(artifact), "--track-id", "op007")

    assert result.returncode == 0, result.stderr
    evidence_path = artifact / "rounds" / "3" / "tracks" / "op007" / "evidence.json"
    assert evidence_path.exists()
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["diluted"] is False


def test_v2_rejects_explicit_prior_round_output(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"schema_version": "4.1", "current_round": 3, "rounds": [{}, {}, {}]}}),
        encoding="utf-8",
    )

    result = _run(
        "--artifact-dir",
        str(artifact),
        "--track-id",
        "op007",
        "--round",
        "2",
    )

    assert result.returncode != 0
    assert "current_round is 3" in result.stderr
    assert not (artifact / "rounds" / "2" / "tracks" / "op007" / "evidence.json").exists()


def test_v2_rejects_output_alias_to_prior_round_or_flat_root(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"schema_version": "4.1", "current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    outputs = [
        artifact / "rounds" / "1" / "tracks" / "op007" / "evidence.json",
        artifact / "tracks" / "op007" / "evidence.json",
    ]

    for output in outputs:
        result = _run(
            "--artifact-dir",
            str(artifact),
            "--track-id",
            "op007",
            "--output",
            str(output),
        )
        assert result.returncode != 0
        assert "current-round track directory" in result.stderr
        assert not output.exists()


def test_output_only_discovers_v2_state_and_rejects_stale_path(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"schema_version": "4.1", "current_round": 2, "rounds": [{}, {}]}}),
        encoding="utf-8",
    )
    output = artifact / "rounds" / "1" / "tracks" / "op007" / "evidence.json"

    result = _run("--track-id", "op007", "--output", str(output))

    assert result.returncode != 0
    assert "current-round track directory" in result.stderr
    assert not output.exists()


def test_template_rejects_track_id_path_traversal(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    result = _run(
        "--artifact-dir",
        str(artifact),
        "--track-id",
        "../../../escape",
    )

    assert result.returncode != 0
    assert "Invalid track id" in result.stderr
    assert not (tmp_path / "escape" / "evidence.json").exists()
