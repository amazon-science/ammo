#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for AMMO artifact parsing compatibility paths."""

from __future__ import annotations

import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from parse_artifacts import _candidate_selection_count, _parse_debate


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_round1_canonical_proposals_take_precedence_over_legacy(tmp_path):
    artifact_dir = tmp_path / "artifact"
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "debate": {
                        "candidates": [],
                        "selected_winners": [],
                    }
                }
            ],
        }
    }

    _write(
        artifact_dir / "debate" / "campaign_round_1" / "proposals" / "champion-1_proposal.md",
        "micro-experiment with bottleneck_analysis data: 12 us\n",
    )
    _write(
        artifact_dir / "debate" / "proposals" / "champion-2_proposal.md",
        "prototype benchmark from nsys profile: 24 us\n",
    )

    result = _parse_debate(artifact_dir, state)

    assert result["total_proposals"] == 1
    assert result["proposals_with_micro_experiments"] == 1
    assert result["proposals_with_grounded_data"] == 1
    assert result["per_campaign_round"][0]["proposals_count"] == 1


def test_duplicate_round1_canonical_and_legacy_proposal_is_deduped(tmp_path):
    artifact_dir = tmp_path / "artifact"
    state = {"campaign": {"current_round": 1, "rounds": [{"debate": {}}]}}
    content = "micro-experiment with bottleneck_analysis data: 12 us\n"

    _write(
        artifact_dir / "debate" / "campaign_round_1" / "proposals" / "champion-1_proposal.md",
        content,
    )
    _write(
        artifact_dir / "debate" / "proposals" / "champion-1_proposal.md",
        content,
    )

    result = _parse_debate(artifact_dir, state)

    assert result["total_proposals"] == 1
    assert result["per_campaign_round"][0]["proposals_count"] == 1


def test_selected_candidates_explicit_empty_means_zero_selected():
    tracks = {"op1": {}, "op2": {}}

    assert _candidate_selection_count({"selected_candidates": []}, tracks) == 0


def test_selected_candidates_nonempty_wins_over_track_fallback():
    tracks = {"op1": {}, "op2": {}}

    assert _candidate_selection_count({"selected_candidates": ["op1"]}, tracks) == 1


def test_missing_selected_candidates_falls_back_to_passing_candidates():
    tracks = {"op1": {}, "op2": {}}

    assert _candidate_selection_count({"passing_candidates": ["op2"]}, tracks) == 1


def test_missing_legacy_selection_fields_falls_back_to_tracks():
    tracks = {"op1": {}, "op2": {}}

    assert _candidate_selection_count({}, tracks) == 2
