# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for check_projection_accuracy.py (Phase 1D / RED).

Per spec §3.8 of 2026-05-11-ammo-f-e2e-and-expanded-mandate-design.md.

Script contract:
- Reads `projected_e2e_improvement_pct` from
    `{artifact_dir}/state.json` →
    `campaign.rounds[current_round-1].debate.selected_candidates[i]
        .projected_e2e_improvement_pct`
- Reads realized `improvement_pct` per-BS from
    `{artifact_dir}/{sweep_dir}/e2e_latency_results.json` →
    `results[].improvement_pct` (one row per batch_size)
- Per-BS comparison rules:
    * realized_pct < 0          → flag "PROJECTION MISMATCH — regression"
    * realized_pct ≤ 0.1 (and ≥ 0)→ flag if projected_pct > 1.0
    * else                       → flag if (projected/realized) > 2.0
                                   OR |projected - realized| > max(0.5, 2*abs(realized))
- Always appends `## Projection Accuracy` section to
    `{artifact_dir}/validation_results.md` (creates the file if absent).
- ALWAYS exit 0 (informational).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).parent.parent.parent
SCRIPT = (
    ROOT / "ai_cli_session" / ".claude" / "skills" / "ammo" / "scripts"
    / "check_projection_accuracy.py"
)


# ─── Fixture helpers ───────────────────────────────────────────────────────


def _make_artifact(
    tmp_path: Path,
    *,
    projected_pct: Optional[float],
    realized_per_bs: List[Dict[str, Any]],
    op_id: str = "op-1",
    sweep_dir_name: str = "e2e_latency",
    include_validation_md: bool = False,
) -> Path:
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "auto_x"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "campaign": {
            "schema_version": "4.1",
            "status": "active",
            "current_round": 1,
            "current_stage": "5_validation",
            "config": {},
            "rounds": [
                {
                    "round_id": 1,
                    "debate": {
                        "selected_candidates": [
                            {
                                "op_id": op_id,
                                "track_assignment": "lossless",
                                "projected_e2e_improvement_pct": projected_pct,
                            },
                            # noise — make sure we pick the right one
                            {
                                "op_id": "op-other",
                                "track_assignment": "lossless",
                                "projected_e2e_improvement_pct": 99.0,
                            },
                        ],
                    },
                }
            ],
        }
    }
    (artifact_dir / "state.json").write_text(json.dumps(state, indent=2))

    sweep_dir = artifact_dir / sweep_dir_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    e2e_payload = {
        "results": [
            {
                "batch_size": row.get("bs", row.get("batch_size")),
                "improvement_pct": row.get("realized_pct"),
            }
            for row in realized_per_bs
        ]
    }
    (sweep_dir / "e2e_latency_results.json").write_text(json.dumps(e2e_payload, indent=2))

    if include_validation_md:
        (artifact_dir / "validation_results.md").write_text(
            "# Validation Results\n\nExisting content.\n"
        )

    return artifact_dir


def _run(artifact_dir: Path, op_id: str = "op-1", sweep_dir: str = "e2e_latency") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            "--op-id",
            op_id,
            "--sweep-dir",
            sweep_dir,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _read_section(artifact_dir: Path) -> str:
    md = (artifact_dir / "validation_results.md").read_text()
    # Return everything from "## Projection Accuracy" onward
    idx = md.find("## Projection Accuracy")
    if idx == -1:
        return ""
    return md[idx:]


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestScriptExists:
    def test_script_file_exists(self):
        assert SCRIPT.exists(), f"Script not found: {SCRIPT}"


class TestAlwaysExitsZero:
    """Script must NEVER exit non-zero (informational only)."""

    def test_match_exit_zero(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 5.0}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0

    def test_mismatch_exit_zero(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=20.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 1.0}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0

    def test_missing_state_exit_zero(self, tmp_path):
        artifact_dir = tmp_path / "kernel_opt_artifacts" / "auto_x"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = _run(artifact_dir)
        assert result.returncode == 0

    def test_missing_sweep_results_exit_zero(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[],
        )
        # Remove the sweep results
        (artifact_dir / "e2e_latency" / "e2e_latency_results.json").unlink()
        result = _run(artifact_dir)
        assert result.returncode == 0


class TestNormalCaseMatch:
    """Projected ≈ realized → no flag, but section still appended."""

    def test_close_match_no_flag(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 4.5}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section
        # No "MISMATCH" / "FLAG" should appear for the matching row
        # (allow it elsewhere, but assert section text doesn't claim a flag)
        assert "PROJECTION MISMATCH" not in section
        assert "regression" not in section.lower()


class TestRegressionEdgeCase:
    """realized_pct < 0 → always flag as regression."""

    def test_negative_realized_flagged(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": -2.0}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "PROJECTION MISMATCH" in section
        assert "regression" in section.lower()

    def test_negative_realized_zero_projection_still_flagged(self, tmp_path):
        """Even projected=0 with realized=-1 should flag (regression that cleared gates)."""
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=0.0,
            realized_per_bs=[{"bs": 8, "realized_pct": -1.0}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "regression" in section.lower()


class TestNoiseEdgeCase:
    """0 ≤ realized_pct ≤ 0.1: flag if projected > 1.0."""

    def test_noise_realized_high_projection_flagged(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=10.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 0.05}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "PROJECTION MISMATCH" in section or "MISMATCH" in section.upper()

    def test_noise_realized_small_projection_not_flagged(self, tmp_path):
        """projected=0.5 (≤1.0), realized=0.05 → no flag."""
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=0.5,
            realized_per_bs=[{"bs": 8, "realized_pct": 0.05}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section
        # Specifically — the bs=8 row should not be flagged
        assert "PROJECTION MISMATCH" not in section


class TestRatioMismatch:
    """ratio > 2.0 OR |proj - realized| > max(0.5, 2*abs(realized))."""

    def test_ratio_above_2_flagged(self, tmp_path):
        # realized=2.0 (above noise cap of 0.1), projected=5.0 → ratio 2.5
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 2.0}],
        )
        result = _run(artifact_dir)
        section = _read_section(artifact_dir)
        # 5/2 = 2.5 > 2.0 OR |5-2|=3 > max(0.5, 4)=4 → flagged via ratio
        assert "PROJECTION MISMATCH" in section or "MISMATCH" in section.upper()

    def test_ratio_at_threshold_flagged(self, tmp_path):
        """ratio = 2.0 should NOT flag (strict >); 2.01 flags."""
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=4.02,
            realized_per_bs=[{"bs": 8, "realized_pct": 2.0}],
        )
        result = _run(artifact_dir)
        section = _read_section(artifact_dir)
        assert "PROJECTION MISMATCH" in section or "MISMATCH" in section.upper()

    def test_abs_diff_violation(self, tmp_path):
        """|proj - realized| > max(0.5, 2*abs(realized))."""
        # realized=0.2 (above 0.1 cap), projected=1.5 → diff=1.3 > max(0.5, 0.4)=0.5
        # ratio = 1.5/0.2 = 7.5 (also > 2) so this also flags via ratio
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=1.5,
            realized_per_bs=[{"bs": 8, "realized_pct": 0.2}],
        )
        result = _run(artifact_dir)
        section = _read_section(artifact_dir)
        assert "PROJECTION MISMATCH" in section or "MISMATCH" in section.upper()

    def test_close_diff_within_tolerance_not_flagged(self, tmp_path):
        # realized=10, projected=10.4 → ratio=1.04, diff=0.4 < max(0.5, 20)=20
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=10.4,
            realized_per_bs=[{"bs": 8, "realized_pct": 10.0}],
        )
        result = _run(artifact_dir)
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section
        assert "PROJECTION MISMATCH" not in section


class TestAppendsSection:
    """Always appends `## Projection Accuracy` section to validation_results.md."""

    def test_creates_validation_md_if_absent(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 4.5}],
        )
        # No validation_results.md yet
        assert not (artifact_dir / "validation_results.md").exists()
        result = _run(artifact_dir)
        assert result.returncode == 0
        assert (artifact_dir / "validation_results.md").exists()
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section

    def test_appends_to_existing_validation_md(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[{"bs": 8, "realized_pct": 4.5}],
            include_validation_md=True,
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        md = (artifact_dir / "validation_results.md").read_text()
        # Pre-existing content preserved
        assert "Existing content." in md
        # Section appended below
        assert "## Projection Accuracy" in md
        assert md.index("Existing content.") < md.index("## Projection Accuracy")

    def test_per_bs_breakdown_emitted(self, tmp_path):
        """Section should contain per-BS values (projected vs realized)."""
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,
            realized_per_bs=[
                {"bs": 8, "realized_pct": 4.5},
                {"bs": 32, "realized_pct": 1.0},
            ],
        )
        result = _run(artifact_dir)
        section = _read_section(artifact_dir)
        # Both batch sizes should be referenced
        assert "8" in section
        assert "32" in section


class TestOpIdSelection:
    def test_uses_correct_op_id(self, tmp_path):
        """When --op-id matches second candidate, uses its projected value."""
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=5.0,  # op-1's value
            realized_per_bs=[{"bs": 8, "realized_pct": 4.5}],
        )
        # op-other has projected=99.0 — using --op-id op-other should flag.
        result = _run(artifact_dir, op_id="op-other")
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section
        # 99 vs 4.5 → ratio ≫ 2 → flagged
        assert "MISMATCH" in section.upper()


class TestMissingProjection:
    """If projected_pct is null → record as 'no projection' but no crash."""

    def test_null_projected_handled(self, tmp_path):
        artifact_dir = _make_artifact(
            tmp_path,
            projected_pct=None,
            realized_per_bs=[{"bs": 8, "realized_pct": 4.5}],
        )
        result = _run(artifact_dir)
        assert result.returncode == 0
        section = _read_section(artifact_dir)
        assert "## Projection Accuracy" in section
