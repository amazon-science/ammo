# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Zone B — Backend E2E Latency Normalizer.

Tests the `_normalize_e2e_latency` read-path normalizer that:
- Backfills `baseline.e2e_latency` from legacy `combined_e2e_result`
- Backfills `integration.e2e_latency_combined` from legacy sources
- Splits `kernel_speedup` dict into scalar + variants
- Flattens `e2e_speedup` object to scalar
- Strips `_s` suffix from latency entry keys
- Computes `campaign.cumulative_e2e_speedup` at read time

TDD: Written before implementation.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.campaign_data_service import _normalize_e2e_latency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(rounds=None, campaign_extras=None):
    """Build a minimal state dict with campaign.rounds."""
    state = {
        "campaign": {
            "status": "active",
            "current_round": 1,
            "rounds": rounds or [],
        }
    }
    if campaign_extras:
        state["campaign"].update(campaign_extras)
    return state


def _make_round(round_id=1, baseline=None, integration=None, parallel_tracks=None):
    """Build a minimal round dict."""
    rnd = {"round_id": round_id}
    if baseline is not None:
        rnd["baseline"] = baseline
    else:
        rnd["baseline"] = {}
    if integration is not None:
        rnd["integration"] = integration
    else:
        rnd["integration"] = {}
    if parallel_tracks is not None:
        rnd["parallel_tracks"] = parallel_tracks
    return rnd


# ---------------------------------------------------------------------------
# Test 19: Baseline e2e_latency backfill from legacy combined_e2e_result
# ---------------------------------------------------------------------------

class TestNormalizeBackfillsBaselineE2eLatency:
    """Test 19: _normalize_e2e_latency backfills baseline.e2e_latency from legacy."""

    def test_backfills_from_combined_e2e_result_with_per_bs_verdict(self):
        """When baseline.e2e_latency absent, combined_e2e_result.latency_baseline_s
        exists, and per_bs_verdict has keys -> uses smallest numeric key."""
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"1": "PASS", "4": "NOISE"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["e2e_latency"] == {"1": {"avg": 7.66, "p50": 7.66}}

    def test_backfills_with_fallback_bs_key_1(self):
        """When per_bs_verdict absent, falls back to '1' as default BS key."""
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["e2e_latency"] == {"1": {"avg": 7.66, "p50": 7.66}}

    def test_does_not_overwrite_existing_baseline_e2e_latency(self):
        """If baseline.e2e_latency already exists, NOT overwritten."""
        existing = {"128": {"avg": 8.0, "p50": 7.9}}
        rnd = _make_round(
            baseline={"e2e_latency": existing},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["e2e_latency"] == existing


# ---------------------------------------------------------------------------
# Test 20: Integration e2e_latency_combined backfill from legacy
# ---------------------------------------------------------------------------

class TestNormalizeBackfillsIntegrationE2eLatencyCombined:
    """Test 20: backfills integration.e2e_latency_combined from legacy sources."""

    def test_backfills_from_combined_e2e_result_opt_s(self):
        """When e2e_latency_combined absent, uses combined_e2e_result data."""
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "speedup_x": 1.18,
                    "latency_baseline_s": 7.66,
                    "opt_s": 6.5,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        integ = state["campaign"]["rounds"][0]["integration"]
        assert integ["e2e_latency_combined"] == {"1": {"avg": 6.5, "p50": 6.5}}

    def test_backfills_from_standalone_integration_opt_s(self):
        """When combined_e2e_result has no opt_s but integration.opt_s exists."""
        rnd = _make_round(
            baseline={},
            integration={
                "opt_s": 5.8,
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"4": "PASS", "1": "NOISE"},
                },
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        integ = state["campaign"]["rounds"][0]["integration"]
        # opt_s from integration level used; BS key from per_bs_verdict (smallest=1)
        assert integ["e2e_latency_combined"] == {"1": {"avg": 5.8, "p50": 5.8}}

    def test_backfills_bs_key_from_per_bs_verdict_smallest(self):
        """Uses smallest numeric key from per_bs_verdict."""
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "opt_s": 6.0,
                    "per_bs_verdict": {"128": "PASS", "256": "NOISE"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        integ = state["campaign"]["rounds"][0]["integration"]
        assert integ["e2e_latency_combined"] == {"128": {"avg": 6.0, "p50": 6.0}}


# ---------------------------------------------------------------------------
# Test 21: No-op when e2e_latency_combined already exists
# ---------------------------------------------------------------------------

class TestNormalizeNoOpWhenE2eLatencyCombinedExists:
    """Test 21: does NOT overwrite when new field already present."""

    def test_preserves_existing_e2e_latency_combined(self):
        existing = {"128": {"avg": 6.0, "p50": 5.9}}
        rnd = _make_round(
            baseline={},
            integration={
                "e2e_latency_combined": existing,
                "combined_e2e_result": {
                    "opt_s": 99.0,
                    "per_bs_verdict": {"1": "PASS"},
                },
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        integ = state["campaign"]["rounds"][0]["integration"]
        assert integ["e2e_latency_combined"] == existing


# ---------------------------------------------------------------------------
# Test 22: Handles null combined_e2e_result
# ---------------------------------------------------------------------------

class TestNormalizeHandlesNullCombinedE2eResult:
    """Test 22: combined_e2e_result: null -> no crash, e2e_latency_combined stays absent."""

    def test_null_combined_e2e_result(self):
        rnd = _make_round(
            baseline={},
            integration={"combined_e2e_result": None},
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        integ = state["campaign"]["rounds"][0]["integration"]
        assert integ.get("e2e_latency_combined") is None

    def test_missing_combined_e2e_result(self):
        rnd = _make_round(baseline={}, integration={})
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        integ = state["campaign"]["rounds"][0]["integration"]
        assert "e2e_latency_combined" not in integ


# ---------------------------------------------------------------------------
# Test 23: Walks all rounds
# ---------------------------------------------------------------------------

class TestNormalizeWalksAllRounds:
    """Test 23: Multi-round state -> each round normalized independently."""

    def test_three_rounds_each_normalized(self):
        rounds = []
        for i in range(1, 4):
            rounds.append(_make_round(
                round_id=i,
                baseline={},
                integration={
                    "combined_e2e_result": {
                        "latency_baseline_s": 7.0 + i,
                        "opt_s": 6.0 + i,
                        "per_bs_verdict": {"1": "PASS"},
                    }
                },
            ))
        state = _make_state(rounds=rounds)
        _normalize_e2e_latency(state)

        for i, rnd in enumerate(state["campaign"]["rounds"]):
            bl = rnd["baseline"]
            integ = rnd["integration"]
            assert bl["e2e_latency"] == {"1": {"avg": 8.0 + i, "p50": 8.0 + i}}
            assert integ["e2e_latency_combined"] == {"1": {"avg": 7.0 + i, "p50": 7.0 + i}}


# ---------------------------------------------------------------------------
# Test 24: Idempotent
# ---------------------------------------------------------------------------

class TestNormalizeIdempotent:
    """Test 24: Running _normalize_e2e_latency twice produces identical output."""

    def test_double_normalize_same_result(self):
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "opt_s": 6.5,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": {"cold_bs8": 1.42, "warm_bs1": 1.28},
                        "e2e_speedup": {"speedup_x": 1.18},
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        first_pass = copy.deepcopy(state)
        _normalize_e2e_latency(state)
        assert state == first_pass

    def test_cumulative_e2e_speedup_unchanged_on_second_call(self):
        """Task 4 — confirm campaign.cumulative_e2e_speedup is stable across
        repeated normalization calls. The L2 and L3 endpoints invoke the
        normalizer inline on each request, so a cached state dict may be
        normalized many times before it's evicted."""
        rnd = _make_round(
            baseline={"e2e_latency": {"1": {"avg": 1.0}}},
            integration={"e2e_latency_combined": {"1": {"avg": 0.7}}},
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        first_speedup = state["campaign"]["cumulative_e2e_speedup"]

        for _ in range(3):
            _normalize_e2e_latency(state)
            assert state["campaign"]["cumulative_e2e_speedup"] == first_speedup

    def test_kernel_speedup_remains_scalar_on_second_call(self):
        """Once kernel_speedup is flattened from dict→scalar, a second pass
        must NOT re-wrap it (would otherwise drop the variants on the third
        pass)."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": {"cold_bs8": 1.42, "warm_bs1": 1.28},
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert isinstance(track["kernel_speedup"], (int, float))
        first_scalar = track["kernel_speedup"]
        first_variants = copy.deepcopy(track.get("kernel_speedup_variants"))

        _normalize_e2e_latency(state)
        track2 = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert isinstance(track2["kernel_speedup"], (int, float))
        assert track2["kernel_speedup"] == first_scalar
        assert track2.get("kernel_speedup_variants") == first_variants

    def test_no_s_suffixed_keys_reappear_after_second_call(self):
        """`_s` suffix stripping must be permanent — running the normalizer
        a second time must NOT reintroduce avg_s/p50_s entries (e.g. by
        mistakenly re-running the legacy backfill from combined_e2e_result)."""
        rnd = _make_round(
            baseline={"e2e_latency": {"1": {"avg_s": 1.0, "p50_s": 1.0}}},
            integration={
                "e2e_latency_combined": {"1": {"avg_s": 0.8, "p50_s": 0.8}},
                "combined_e2e_result": {
                    "latency_baseline_s": 1.0,
                    "opt_s": 0.8,
                    "per_bs_verdict": {"1": "PASS"},
                },
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]["e2e_latency"]["1"]
        elc = state["campaign"]["rounds"][0]["integration"]["e2e_latency_combined"]["1"]
        for entry in (bl, elc):
            assert not any(k.endswith("_s") for k in entry), entry


# ---------------------------------------------------------------------------
# Test 25: kernel_speedup dict -> scalar + variants
# ---------------------------------------------------------------------------

class TestNormalizeKernelSpeedupSplit:
    """Test 25: kernel_speedup dict split into scalar + kernel_speedup_variants."""

    def test_dict_split_max_non_meta(self):
        """Dict -> max non-meta value becomes scalar, whole dict -> variants."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": {"cold_bs8": 1.42, "warm_bs1": 1.28},
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["kernel_speedup"] == 1.42
        assert track["kernel_speedup_variants"] == {"cold_bs8": 1.42, "warm_bs1": 1.28}

    def test_dict_excludes_meta_keys(self):
        """Meta keys (target, floor, ship_gate, threshold, ceiling) excluded from max."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": {
                            "target": 2.0,
                            "floor": 1.0,
                            "ship_gate": 1.5,
                            "cold_bs8": 1.42,
                            "warm_bs1": 1.28,
                        },
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["kernel_speedup"] == 1.42
        assert "target" in track["kernel_speedup_variants"]

    def test_scalar_unchanged(self):
        """kernel_speedup already a scalar -> no change."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": 1.35,
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["kernel_speedup"] == 1.35
        assert "kernel_speedup_variants" not in track

    def test_null_unchanged(self):
        """kernel_speedup: null -> no change."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": None,
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["kernel_speedup"] is None


# ---------------------------------------------------------------------------
# Test 26: e2e_speedup dict -> scalar
# ---------------------------------------------------------------------------

class TestNormalizeE2eSpeedupFlatten:
    """Test 26: e2e_speedup dict with speedup_x -> flattened to number."""

    def test_dict_with_speedup_x_flattened(self):
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "e2e_speedup": {"speedup_x": 1.18, "measured": {"bs1": 1.2}},
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["e2e_speedup"] == 1.18

    def test_null_unchanged(self):
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "e2e_speedup": None,
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["e2e_speedup"] is None

    def test_scalar_unchanged(self):
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "e2e_speedup": 1.18,
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        assert track["e2e_speedup"] == 1.18


# ---------------------------------------------------------------------------
# Test 27: passing_candidates e2e_speedup dict -> scalar
# ---------------------------------------------------------------------------

class TestNormalizePassingCandidatesE2eSpeedup:
    """Test 27: integration.passing_candidates[*].e2e_speedup dict -> flattened."""

    def test_passing_candidate_speedup_flattened(self):
        rnd = _make_round(
            integration={
                "passing_candidates": [
                    {"op_id": "track_a", "e2e_speedup": {"speedup_x": 1.2}},
                    {"op_id": "track_b", "e2e_speedup": 1.3},
                    {"op_id": "track_c", "e2e_speedup": None},
                ],
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        candidates = state["campaign"]["rounds"][0]["integration"]["passing_candidates"]
        assert candidates[0]["e2e_speedup"] == 1.2
        assert candidates[1]["e2e_speedup"] == 1.3
        assert candidates[2]["e2e_speedup"] is None


# ---------------------------------------------------------------------------
# Test 28: per_bs_verdict backfill from combined_e2e_result
# ---------------------------------------------------------------------------

class TestNormalizePerBsVerdictBackfill:
    """Test 28: per_bs_verdict from combined_e2e_result -> baseline.per_bs_verdict."""

    def test_backfills_per_bs_verdict_to_baseline(self):
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["per_bs_verdict"] == {"1": "PASS"}

    def test_does_not_overwrite_existing_per_bs_verdict(self):
        existing = {"128": "NOISE"}
        rnd = _make_round(
            baseline={"per_bs_verdict": existing},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["per_bs_verdict"] == existing


# ---------------------------------------------------------------------------
# Test 29: _s suffix strip
# ---------------------------------------------------------------------------

class TestNormalizeStripsSuffix:
    """Test 29: strips _s suffix from e2e_latency entry keys (defensive)."""

    def test_strips_s_suffix_from_baseline(self):
        rnd = _make_round(
            baseline={
                "e2e_latency": {
                    "128": {"avg_s": 7.66, "p50_s": 7.55},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        entry = state["campaign"]["rounds"][0]["baseline"]["e2e_latency"]["128"]
        assert entry == {"avg": 7.66, "p50": 7.55}

    def test_strips_s_suffix_from_integration(self):
        rnd = _make_round(
            integration={
                "e2e_latency_combined": {
                    "256": {"avg_s": 6.5, "p50_s": 6.4, "p90_s": 7.0},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        entry = state["campaign"]["rounds"][0]["integration"]["e2e_latency_combined"]["256"]
        assert entry == {"avg": 6.5, "p50": 6.4, "p90": 7.0}

    def test_no_double_strip(self):
        """Already correct keys (no _s) -> unchanged."""
        rnd = _make_round(
            baseline={
                "e2e_latency": {
                    "128": {"avg": 7.66, "p50": 7.55},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        entry = state["campaign"]["rounds"][0]["baseline"]["e2e_latency"]["128"]
        assert entry == {"avg": 7.66, "p50": 7.55}


# ---------------------------------------------------------------------------
# Test 29b: cumulative_e2e_speedup computed from maps
# ---------------------------------------------------------------------------

class TestNormalizerInjectsCumulativeSpeedup:
    """Test 29b: normalizer computes and injects campaign.cumulative_e2e_speedup."""

    def test_computes_from_round1_baseline_and_integration(self):
        """rounds[0].baseline.e2e_latency / integration.e2e_latency_combined."""
        rnd = _make_round(
            round_id=1,
            baseline={
                "e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}},
            },
            integration={
                "e2e_latency_combined": {"128": {"avg": 5.68, "p50": 5.6}},
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        cum = state["campaign"]["cumulative_e2e_speedup"]
        assert abs(cum - 7.66 / 5.68) < 0.001  # ~1.3486

    def test_uses_smallest_bs_key(self):
        """When multiple BS keys, uses smallest numeric."""
        rnd = _make_round(
            round_id=1,
            baseline={
                "e2e_latency": {
                    "128": {"avg": 7.66, "p50": 7.55},
                    "256": {"avg": 8.2, "p50": 8.1},
                },
            },
            integration={
                "e2e_latency_combined": {
                    "128": {"avg": 5.68, "p50": 5.6},
                    "256": {"avg": 6.5, "p50": 6.4},
                },
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        cum = state["campaign"]["cumulative_e2e_speedup"]
        # Uses BS 128: 7.66 / 5.68
        assert abs(cum - 7.66 / 5.68) < 0.001

    def test_multi_round_uses_latest_integration(self):
        """With multiple rounds, uses latest round's integration."""
        r1 = _make_round(
            round_id=1,
            baseline={"e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}}},
            integration={"e2e_latency_combined": {"128": {"avg": 6.5, "p50": 6.4}}},
        )
        r2 = _make_round(
            round_id=2,
            baseline={"e2e_latency": {"128": {"avg": 6.5, "p50": 6.4}}},
            integration={"e2e_latency_combined": {"128": {"avg": 5.0, "p50": 4.9}}},
        )
        state = _make_state(rounds=[r1, r2])
        _normalize_e2e_latency(state)

        cum = state["campaign"]["cumulative_e2e_speedup"]
        # anchor = r1 baseline 7.66; latest = r2 integration 5.0
        assert abs(cum - 7.66 / 5.0) < 0.001  # ~1.532


# ---------------------------------------------------------------------------
# Test 29c: cumulative_speedup legacy fallback
# ---------------------------------------------------------------------------

class TestNormalizerCumulativeSpeedupLegacyFallback:
    """Test 29c: normalizer backfills map THEN computes cumulative from it."""

    def test_legacy_combined_e2e_result_backfills_then_computes(self):
        """No baseline.e2e_latency initially, but combined_e2e_result has data."""
        rnd = _make_round(
            round_id=1,
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "opt_s": 5.68,
                    "per_bs_verdict": {"1": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        # Baseline should be backfilled
        bl = state["campaign"]["rounds"][0]["baseline"]
        assert bl["e2e_latency"] == {"1": {"avg": 7.66, "p50": 7.66}}

        # Integration should be backfilled
        integ = state["campaign"]["rounds"][0]["integration"]
        assert integ["e2e_latency_combined"] == {"1": {"avg": 5.68, "p50": 5.68}}

        # Cumulative speedup should be computed from backfilled values
        cum = state["campaign"]["cumulative_e2e_speedup"]
        assert abs(cum - 7.66 / 5.68) < 0.001


# ---------------------------------------------------------------------------
# Test 29d: cumulative_speedup with no integration returns 1.0
# ---------------------------------------------------------------------------

class TestNormalizerCumulativeSpeedupNoIntegration:
    """Test 29d: baseline but no integration result -> cumulative = 1.0."""

    def test_no_integration_data(self):
        rnd = _make_round(
            round_id=1,
            baseline={"e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}}},
            integration={},
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        assert state["campaign"]["cumulative_e2e_speedup"] == 1.0

    def test_no_baseline_data(self):
        """No baseline e2e_latency at all -> cumulative = 1.0."""
        rnd = _make_round(
            round_id=1,
            baseline={},
            integration={},
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        assert state["campaign"]["cumulative_e2e_speedup"] == 1.0


# ---------------------------------------------------------------------------
# Test 29e: normalizer overwrites stale stored cumulative
# ---------------------------------------------------------------------------

class TestNormalizerOverwritesStaleCumulative:
    """Test 29e: stored cumulative_speedup_vs_round1 overwritten by computed value."""

    def test_overwrites_stale_value(self):
        rnd = _make_round(
            round_id=1,
            baseline={"e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}}},
            integration={"e2e_latency_combined": {"128": {"avg": 5.68, "p50": 5.6}}},
        )
        state = _make_state(
            rounds=[rnd],
            campaign_extras={
                "cumulative_speedup_vs_round1": 1.2,  # stale value
                "cumulative_e2e_speedup": 1.2,  # stale value
            },
        )
        _normalize_e2e_latency(state)

        cum = state["campaign"]["cumulative_e2e_speedup"]
        # Should be recomputed: 7.66 / 5.68 ≈ 1.3486
        assert abs(cum - 7.66 / 5.68) < 0.001
        assert cum != 1.2


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestNormalizeEdgeCases:
    """Additional edge-case tests for robustness."""

    def test_state_without_campaign(self):
        """State with no campaign key -> no crash."""
        state = {"target": {"model_id": "test"}}
        _normalize_e2e_latency(state)
        # Should not crash or add campaign key
        assert "campaign" not in state or state.get("campaign") is None

    def test_campaign_without_rounds(self):
        """Campaign with no rounds key -> no crash."""
        state = {"campaign": {"status": "active"}}
        _normalize_e2e_latency(state)
        # No crash

    def test_round_with_no_baseline_or_integration(self):
        """Round with missing baseline/integration -> no crash."""
        state = _make_state(rounds=[{"round_id": 1}])
        _normalize_e2e_latency(state)
        # No crash

    def test_non_numeric_per_bs_verdict_keys_ignored_for_bs_selection(self):
        """Non-numeric keys in per_bs_verdict don't affect BS key selection."""
        rnd = _make_round(
            baseline={},
            integration={
                "combined_e2e_result": {
                    "latency_baseline_s": 7.66,
                    "per_bs_verdict": {"primary": "PASS", "128": "NOISE", "256": "PASS"},
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        bl = state["campaign"]["rounds"][0]["baseline"]
        # Should use "128" (smallest numeric), not "primary"
        assert bl["e2e_latency"] == {"128": {"avg": 7.66, "p50": 7.66}}

    def test_kernel_speedup_all_meta_keys_only(self):
        """kernel_speedup dict with ONLY meta keys -> no non-meta values to extract."""
        rnd = _make_round(
            parallel_tracks={
                "tracks": {
                    "track_a": {
                        "kernel_speedup": {"target": 2.0, "floor": 1.0},
                    }
                }
            },
        )
        state = _make_state(rounds=[rnd])
        _normalize_e2e_latency(state)

        track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["track_a"]
        # No non-meta values -> scalar becomes None, variants still stored
        assert track["kernel_speedup"] is None
        assert track["kernel_speedup_variants"] == {"target": 2.0, "floor": 1.0}
