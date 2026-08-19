#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""DILUTED_PASS ship-path consumer tests for the eval scripts (Step 6).

Covers three surfaces:
- parse_artifacts._parse_tracks passthrough of the new `diluted` flag (default False).
- score_campaign.score_e2e discounting diluted ships out of `effective_shipped`
  (so a diluted ~1.00x ship never earns the extra-ship bonus).
- The cumulative-accounting firewall: a diluted track's TPOT gain must never be
  substituted or blended into any cumulative-speedup-shaped output field, in either
  score_campaign or parse_artifacts.
"""

import json
import sys
from pathlib import Path

import pytest

# Add the eval/scripts dir to path so we can import the target modules by name.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from parse_artifacts import _parse_tracks, _parse_campaign  # noqa: E402
from score_campaign import score_e2e, score_gates  # noqa: E402


# ---------------------------------------------------------------------------
# (ii) parse_artifacts._parse_tracks passthrough
# ---------------------------------------------------------------------------

def test_parse_tracks_passes_through_diluted_flag():
    state = {"campaign": {"rounds": [{"round_id": 1, "parallel_tracks": {"tracks": {
        "op-A": {"status": "PASS", "diluted": True},
        "op-B": {"status": "PASS"},                 # no diluted key
    }}}]}}
    tracks = {t["op_id"]: t for t in _parse_tracks(state)}
    assert tracks["op-A"]["diluted"] is True
    assert tracks["op-B"]["diluted"] is False       # default when absent


# ---------------------------------------------------------------------------
# (iii) score_campaign.score_e2e ship-bonus discount
# ---------------------------------------------------------------------------

def _e2e_snapshot(shipped_ids, cumulative=1.10, status="active"):
    """Minimal snapshot for score_e2e: shipped_optimization_ids carry dict entries."""
    return {
        "campaign": {
            "status": status,
            "cumulative_speedup_vs_round1": cumulative,
            "shipped_optimizations_count": len(shipped_ids),
            "shipped_optimization_ids": shipped_ids,
            "rounds": [],
        },
    }


def test_diluted_ship_discounted_from_effective_shipped():
    # 3 shipped ids, 1 diluted -> bonus_shipped == 2 -> extra_ships == 1 -> bonus 0.5.
    # (effective_shipped stays 3 for the base-score gate; only the bonus is discounted.)
    shipped_ids = [
        {"op_id": "op-A", "round": 1, "diluted": True},
        {"op_id": "op-B", "round": 1},
        {"op_id": "op-C", "round": 2},
    ]
    result = score_e2e(_e2e_snapshot(shipped_ids))
    sub = result["sub_scores"]
    assert sub["diluted_ship_count"] == 1
    assert sub["ship_bonus"] == pytest.approx(0.5)   # (3 - 1 diluted - 1) * 0.5


def test_diluted_ship_count_derived_from_track_objects():
    # PIPELINE-LEVEL: the diluted marker lives on the TRACK object, and production
    # does NOT reliably copy it onto shipped_optimizations[] entries. The scorer
    # must still count the diluted ship by joining shipped op_ids against the
    # diluted tracks list (snapshot["tracks"]) — NOT only by reading entry.diluted.
    snapshot = {
        "campaign": {
            "status": "active",
            "cumulative_speedup_vs_round1": 1.10,
            "shipped_optimizations_count": 2,
            # NOTE: neither shipped entry carries a `diluted` marker (real pipeline shape).
            "shipped_optimization_ids": [
                {"op_id": "op-A", "round": 1},
                {"op_id": "op-B", "round": 1},
            ],
            "rounds": [],
        },
        # The diluted signal is only on the track objects, exactly as
        # parse_artifacts._parse_tracks surfaces it.
        "tracks": [
            {"op_id": "op-A", "diluted": True, "status": "PASS"},
            {"op_id": "op-B", "diluted": False, "status": "PASS"},
        ],
    }
    result = score_e2e(snapshot)
    sub = result["sub_scores"]
    assert sub["diluted_ship_count"] == 1               # derived from tracks, not entries
    assert sub["ship_bonus"] == pytest.approx(0.0)      # (2 - 1 diluted - 1) * 0.5 = 0


def test_no_diluted_ship_keeps_full_bonus():
    # Control: same 3 ships, none diluted -> effective_shipped == 3 -> extra 2 -> bonus 1.0.
    shipped_ids = [
        {"op_id": "op-A", "round": 1},
        {"op_id": "op-B", "round": 1},
        {"op_id": "op-C", "round": 2},
    ]
    result = score_e2e(_e2e_snapshot(shipped_ids))
    sub = result["sub_scores"]
    assert sub["diluted_ship_count"] == 0
    assert sub["ship_bonus"] == pytest.approx(1.0)   # (3 - 1) * 0.5


def test_all_diluted_ships_keep_base_score_but_no_bonus():
    # A diluted-only campaign at cumulative ~1.00x DID ship real, correctness-gated
    # opts whose measured ~1.00x e2e enters cumulative accounting (pinned constraint
    # 6c). It must score the base '>=1.00x' tier (2.0), NOT be zeroed to
    # 'exhausted_no_ship'. Diluted ships are only discounted out of the diversity
    # ship_bonus (0.0 here), never out of the base-score gate.
    shipped_ids = [
        {"op_id": "op-A", "round": 1, "diluted": True},
        {"op_id": "op-B", "round": 1, "diluted": True},
    ]
    result = score_e2e(_e2e_snapshot(shipped_ids, cumulative=1.00))
    sub = result["sub_scores"]
    assert sub["diluted_ship_count"] == 2
    assert sub["ship_bonus"] == pytest.approx(0.0)
    # Base score is the '>=1.00x' tier (2.0), NOT the exhausted_no_ship 0.0.
    assert sub["base_score"] == pytest.approx(2.0)
    assert sub["speedup_tier"] == ">=1.00x"
    assert result["score"] == pytest.approx(2.0)


def test_diluted_only_campaign_beats_zeroed_no_ship():
    # Regression guard for the score_campaign.py:126 bug: a diluted-only campaign
    # must NOT score the same as a campaign that shipped nothing.
    diluted_only = score_e2e(_e2e_snapshot(
        [{"op_id": "op-A", "round": 1, "diluted": True}], cumulative=1.00))
    assert diluted_only["score"] > 0.0
    assert diluted_only["sub_scores"]["speedup_tier"] == ">=1.00x"


# ---------------------------------------------------------------------------
# (iv) cumulative-accounting firewall
# ---------------------------------------------------------------------------

_CUMULATIVE_SHAPED_KEYS = (
    "cumulative_speedup",
    "cumulative_after",
    "verified_cumulative_speedup",
    "cumulative_speedup_vs_round1",
)


def _no_tpot_leak(obj, tpot_value):
    """Assert the large TPOT number never appears in any cumulative-shaped field."""
    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _CUMULATIVE_SHAPED_KEYS:
                    assert v != tpot_value, f"TPOT {tpot_value} leaked into {k}"
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    _walk(obj)


def test_tpot_never_substituted_into_cumulative_speedup_output():
    # Round carries a diluted track with a large TPOT gain; the measured e2e a diluted
    # ship contributes is ~1.00x. The 40.0 TPOT must appear NOWHERE in a
    # cumulative-speedup-shaped field.
    snapshot = {
        "campaign": {
            "status": "active",
            "cumulative_speedup_vs_round1": 1.00,
            "shipped_optimizations_count": 1,
            "shipped_optimization_ids": [
                {"op_id": "op-diluted", "round": 1, "diluted": True},
            ],
            "rounds": [
                {
                    "round_id": 1,
                    "round_e2e_speedup": 1.00,
                    "cumulative_speedup_after": 1.00,
                    "diluted_tracks": [
                        {"op_id": "op-diluted", "tpot_improvement_pct": 40.0,
                         "decode_share_of_e2e": 0.8},
                    ],
                },
            ],
        },
    }
    result = score_e2e(snapshot)
    assert result["sub_scores"]["cumulative_speedup"] == 1.00   # measured, not TPOT
    _no_tpot_leak(result, 40.0)


def test_parse_artifacts_cumulative_ignores_diluted_tpot():
    # Same firewall at the parse_artifacts layer: the parsed cumulative equals the
    # state's ~1.00x value verbatim; no TPOT-derived number replaces it.
    state = {
        "campaign": {
            "status": "active",
            "current_round": 1,
            "cumulative_speedup_vs_round1": 1.00,
            "shipped_optimizations": [
                {"op_id": "op-diluted", "round": 1, "diluted": True},
            ],
            "rounds": [
                {
                    "round_id": 1,
                    "cumulative_speedup_after": 1.00,
                    "diluted_tracks": [
                        {"op_id": "op-diluted", "tpot_improvement_pct": 40.0,
                         "decode_share_of_e2e": 0.8},
                    ],
                    "parallel_tracks": {"tracks": {
                        "op-diluted": {"status": "PASS", "diluted": True,
                                       "e2e_speedup": 1.00},
                    }},
                },
            ],
        },
    }
    parsed = _parse_campaign(state)
    assert parsed["cumulative_speedup_vs_round1"] == 1.00
    _no_tpot_leak(parsed, 40.0)


def test_lowercase_gated_pass_remains_a_passing_integration_gate():
    result = score_gates({"gates": {"integration": {"status": "gated_pass"}}})
    assert result["sub_scores"]["total_passed"] == 1

