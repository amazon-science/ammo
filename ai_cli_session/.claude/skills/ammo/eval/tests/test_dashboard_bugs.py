#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests that reproduce dashboard bugs found in the eval pipeline.

Each test targets a specific bug:
- test_version_ordering_by_date: Bug 1 - versions sorted by created_at, not dir name
- test_slug_normalization: Bug 2 - bf16/bfloat16 treated as same target
- test_delta_computation_cross_target: Bug 3 - deltas null when adjacent versions differ
- test_delta_finds_same_target_predecessor: Bug 6 - delta should find most recent same-target version
- test_chart_canvas_not_destroyed: Bug 5 - Chart.js fallback preserves <canvas>
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from aggregate_versions import rebuild_index  # noqa: E402
from generate_dashboard import _collect_dashboard_data  # noqa: E402


def _make_scorecard(score: float, speedup: float, shipped: int) -> dict:
    """Create a minimal scorecard for testing."""
    return {
        "overall_score": score,
        "overall_score_without_transcript": score * 0.8,
        "dimensions": {
            "e2e_outcome": {"score": score * 0.4, "weight": 0.4, "weighted_contribution": score * 0.16},
            "gate_pass_rates": {"score": 5.0, "weight": 0.15, "weighted_contribution": 0.75},
            "debate_quality": {"score": 7.0, "weight": 0.15, "weighted_contribution": 1.05},
            "campaign_efficiency": {"score": 6.0, "weight": 0.15, "weighted_contribution": 0.9},
            "transcript_quality": {"score": 7.0, "weight": 0.15, "weighted_contribution": 1.05},
        },
        "raw_metrics": {
            "cumulative_speedup_vs_round1": speedup,
            "shipped_optimizations": shipped,
            "total_rounds": 1,
            "total_proposals": 3,
            "total_tracks": 3,
            "tracks_passed": shipped,
            "tracks_failed": 3 - shipped,
            "campaign_status": "converged",
        },
        "timing": {"total_tracked_seconds": 3600},
        "agent_costs": {"total_tokens": 100000, "total_duration_ms": 60000, "total_agent_invocations": 10},
    }


def _make_version(repo: Path, version_id: str, target_slug: str,
                  created_at: str, description: str,
                  score: float, speedup: float, shipped: int = 1):
    """Create a version directory with meta.json and one run."""
    vdir = repo / "versions" / version_id
    run_dir = vdir / "runs" / target_slug / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "version_id": version_id,
        "git_commit": version_id[:9],
        "description": description,
        "created_at": created_at,
    }
    (vdir / "meta.json").write_text(json.dumps(meta))

    sc = _make_scorecard(score, speedup, shipped)
    (run_dir / "scorecard.json").write_text(json.dumps(sc))


@pytest.fixture
def repo_with_misordered_versions(tmp_path):
    """Reproduces Bug 1+2+3: versions that sort alphabetically != chronologically,
    with inconsistent slug naming."""
    repo = tmp_path / "ammo-eval"
    repo.mkdir()

    # Version A: created Mar 10, high score, git hash starts with 'f'
    _make_version(repo, "faaac90ef_campaign-A", "qwen3-5-4b_l40s_bf16_tp1",
                  "2026-03-10T20:00:00+00:00", "Campaign A", 8.99, 1.262, 2)

    # Version B: created Mar 11, mid score, git hash starts with '0'
    _make_version(repo, "06f309c83_campaign-B", "qwen3-5-4b_l40s_bf16_tp1",
                  "2026-03-11T21:00:00+00:00", "Campaign B", 6.27, 1.06, 1)

    # Version C: created Mar 13, low score, different slug (bfloat16 vs bf16)
    _make_version(repo, "99a0fbed6_campaign-C", "qwen3-5-4b_l40s_bfloat16_tp1",
                  "2026-03-13T13:00:00+00:00", "Campaign C", 3.31, 1.018, 2)

    return repo


class TestVersionOrdering:
    """Bug 1: Versions must be sorted by created_at, not directory name."""

    def test_version_ordering_by_date(self, repo_with_misordered_versions):
        repo = repo_with_misordered_versions
        index = rebuild_index(repo)
        versions = index["versions"]

        dates = [v["created_at"] for v in versions]
        assert dates == sorted(dates), (
            f"Versions not sorted chronologically: {dates}"
        )

        # Specifically: Campaign A (Mar 10) should be first
        assert "faaac90ef" in versions[0]["version_id"]
        assert "06f309c83" in versions[1]["version_id"]
        assert "99a0fbed6" in versions[2]["version_id"]


class TestSlugNormalization:
    """Bug 2: bf16 and bfloat16 should be treated as the same target."""

    def test_slug_normalization(self, repo_with_misordered_versions):
        repo = repo_with_misordered_versions
        index = rebuild_index(repo)

        target_slugs = {t["slug"] for t in index["reference_targets"]}
        # Should be 1 target, not 2
        assert len(target_slugs) == 1, (
            f"Expected 1 normalized target, got {len(target_slugs)}: {target_slugs}"
        )

    def test_all_versions_visible_for_target(self, repo_with_misordered_versions):
        repo = repo_with_misordered_versions
        index = rebuild_index(repo)

        # Pick the single normalized target slug
        target_slugs = [t["slug"] for t in index["reference_targets"]]
        assert len(target_slugs) == 1
        slug = target_slugs[0]

        # All 3 versions should have data for this target
        versions_with_data = [
            v for v in index["versions"]
            if slug in v.get("summary", {})
        ]
        assert len(versions_with_data) == 3, (
            f"Expected 3 versions with data for '{slug}', got {len(versions_with_data)}"
        )


class TestDeltaComputation:
    """Bugs 3+6: Deltas should compare same-target versions correctly."""

    def test_delta_computation_cross_target(self, repo_with_misordered_versions):
        repo = repo_with_misordered_versions
        index = rebuild_index(repo)
        data = _collect_dashboard_data(repo)

        # After slug normalization and ordering, deltas should not be null
        # for versions that evaluate the same (normalized) target
        for v in data["versions"][1:]:
            deltas = v.get("deltas", {})
            for target, d in deltas.items():
                # At least one delta should be non-null
                assert d.get("delta_score") is not None or d.get("delta_speedup") is not None, (
                    f"Delta is null for version {v['version_id']} target {target}: {d}"
                )

    def test_delta_finds_same_target_predecessor(self):
        """When versions alternate targets, delta should find the most recent
        version that evaluated the SAME target, not just version i-1."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ammo-eval"
            repo.mkdir()

            # v1: target A
            _make_version(repo, "aaa_v1", "target_a",
                          "2026-03-01T00:00:00+00:00", "v1", 5.0, 1.05)
            # v2: target B (different target)
            _make_version(repo, "bbb_v2", "target_b",
                          "2026-03-02T00:00:00+00:00", "v2", 6.0, 1.10)
            # v3: target A again
            _make_version(repo, "ccc_v3", "target_a",
                          "2026-03-03T00:00:00+00:00", "v3", 7.0, 1.15)

            rebuild_index(repo)
            data = _collect_dashboard_data(repo)

            # v3's delta for target_a should compare against v1 (not v2)
            v3 = data["versions"][2]
            delta_a = v3.get("deltas", {}).get("target_a", {})
            assert delta_a.get("delta_score") is not None, (
                f"v3 should have a non-null delta for target_a vs v1: {delta_a}"
            )
            # Delta should be 7.0 - 5.0 = 2.0
            assert abs(delta_a["delta_score"] - 2.0) < 0.1, (
                f"Expected delta_score ~2.0, got {delta_a['delta_score']}"
            )


class TestChartCanvasPreservation:
    """Bug 5: The <canvas> element should not be destroyed by Chart.js fallback."""

    def test_chart_canvas_not_destroyed(self, repo_with_misordered_versions):
        """The dashboard template should not replace canvas innerHTML on fallback."""
        repo = repo_with_misordered_versions
        rebuild_index(repo)

        template_path = Path(__file__).parent.parent / "viewer" / "dashboard_template.html"
        template = template_path.read_text()

        # The renderTrendChart function should NOT replace parentElement.innerHTML
        # when Chart.js is unavailable. It should just return early.
        assert "parentElement.innerHTML" not in template or "return" in template.split("parentElement.innerHTML")[0].split("renderTrendChart")[-1], (
            "renderTrendChart should not destroy canvas parentElement innerHTML. "
            "Use early return instead of innerHTML replacement when Chart.js unavailable."
        )
