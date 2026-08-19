#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for v2 round/track projection-accuracy output paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _make_v2_artifact(
    tmp_path: Path,
    *,
    round_num: int,
    op_id: str,
    improvement_pct: float = 4.0,
    projected_pct: float = 5.0,
) -> Path:
    artifact = tmp_path / "v2_artifact"
    round_dir = artifact / "rounds" / str(round_num)
    (round_dir / "tracks" / op_id).mkdir(parents=True)
    sweep_dir = round_dir / "sweeps" / "opt" / op_id
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "e2e_latency_results.json").write_text(
        json.dumps({"results": [{"batch_size": 1, "improvement_pct": improvement_pct}]}),
        encoding="utf-8",
    )
    state = {
        "schema_version": "4.1",
        "campaign": {
            "current_round": round_num,
            "rounds": [{} for _ in range(round_num)],
        },
    }
    state["campaign"]["rounds"][round_num - 1] = {
        "debate": {
            "selected_candidates": [
                {"op_id": op_id, "projected_e2e_improvement_pct": projected_pct},
            ],
        },
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return artifact


def _make_legacy_artifact(
    tmp_path: Path,
    *,
    op_id: str = "op001",
    improvement_pct: float = 4.0,
    projected_pct: float = 5.0,
) -> Path:
    artifact = tmp_path / "v1_artifact"
    artifact.mkdir(parents=True)
    (artifact / "e2e_latency").mkdir()
    (artifact / "e2e_latency" / "e2e_latency_results.json").write_text(
        json.dumps({"results": [{"batch_size": 1, "improvement_pct": improvement_pct}]}),
        encoding="utf-8",
    )
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "debate": {
                        "selected_candidates": [
                            {"op_id": op_id, "projected_e2e_improvement_pct": projected_pct},
                        ],
                    },
                },
            ],
        },
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return artifact


class TestProjectionAccuracyV2:
    def test_round_and_track_id_flags(self, tmp_path):
        from check_projection_accuracy import main

        artifact = _make_v2_artifact(tmp_path, round_num=2, op_id="op007")
        rc = main([
            "--artifact-dir",
            str(artifact),
            "--op-id",
            "op007",
            "--round",
            "2",
            "--track-id",
            "op007",
        ])
        assert rc == 0
        out = artifact / "rounds" / "2" / "tracks" / "op007" / "validation_results.md"
        assert out.is_file()
        assert "## Projection Accuracy" in out.read_text(encoding="utf-8")

    def test_sweep_dir_derived_from_round_slot(self, tmp_path):
        from check_projection_accuracy import main

        artifact = _make_v2_artifact(
            tmp_path,
            round_num=2,
            op_id="op007",
            improvement_pct=4.5,
            projected_pct=5.0,
        )
        rc = main([
            "--artifact-dir",
            str(artifact),
            "--op-id",
            "op007",
            "--round",
            "2",
            "--track-id",
            "op007",
        ])
        assert rc == 0
        out = artifact / "rounds" / "2" / "tracks" / "op007" / "validation_results.md"
        assert "+4.50%" in out.read_text(encoding="utf-8")

    def test_legacy_fallback_no_round_flag(self, tmp_path):
        from check_projection_accuracy import main

        artifact = _make_legacy_artifact(tmp_path)
        rc = main(["--artifact-dir", str(artifact), "--op-id", "op001"])
        assert rc == 0
        assert (artifact / "validation_results.md").is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
