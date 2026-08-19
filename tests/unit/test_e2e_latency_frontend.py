# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Zone C — Frontend data path rewrite for schema v4.0 e2e_latency.

Covers:
- Task 9: circuit-board.js baselineByRound direct read from baseline.e2e_latency map
- Task 10: _baselineTooltip full percentile table with verdict color
- Task 11: campaign-app.js _baselineHeroLatency() simplification
- Task 12: _kernelSpeedupScalar / _e2eSpeedupScalar post-normalizer simplification,
          client-side delta_pp computation, cumulative read simplification

Tests 30-37 from `.claude/plans/e2e-latency-schema-restructure.md`.

Pattern: drive the frontend modules via Node.js with a synthetic `state` object
and assert on the published CircuitBoard.* surface or on the computed
baselineByRound catalog.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
CIRCUIT_BOARD_JS = ROOT / "frontend" / "js" / "circuit-board.js"
CAMPAIGN_APP_JS  = ROOT / "frontend" / "js" / "campaign-app.js"


def run_js(script: str, timeout: int = 10) -> str:
    # circuit-board.js is large (~100KB) — pipe via stdin to avoid ARG_MAX.
    result = subprocess.run(
        ["node", "--input-type=commonjs", "-e", "eval(require('fs').readFileSync(0,'utf8'))"],
        input=script, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error:\n{result.stderr}")
    return result.stdout.strip()


# ────────────────────────────────────────────────────────────────────────────
# Helpers to invoke circuit-board.js internals (module-scoped functions are
# reachable via the published CircuitBoard global).
# ────────────────────────────────────────────────────────────────────────────

def _cb_harness(extra_js: str) -> str:
    """Run `extra_js` with circuit-board.js loaded and a stub DOM."""
    src = CIRCUIT_BOARD_JS.read_text()
    script = f"""
const window = {{}};
const document = {{
  addEventListener: () => {{}},
  createElement: () => ({{
    style: {{}}, className: '', innerHTML: '',
    appendChild: () => {{}}, children: [], setAttribute: () => {{}},
    addEventListener: () => {{}},
  }}),
}};
{src}
{extra_js}
"""
    return run_js(script)


def _enrich_state(state: dict) -> dict:
    """Run CircuitBoard.enrichFromCatalog(state) and return the mutated state."""
    script = f"""
const state = {json.dumps(state)};
if (!state._catalog) state._catalog = {{ entries: {{}} }};
CircuitBoard.enrichFromCatalog(state);
console.log(JSON.stringify(state._catalog));
"""
    out = _cb_harness(script)
    return json.loads(out)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

V4_BASELINE_MULTI_BS = {
    "campaign": {
        "rounds": [
            {
                "round_id": 1,
                "baseline": {
                    "e2e_latency": {
                        "128": {
                            "avg": 7.66, "p50": 7.55,
                            "p10": 7.20, "p25": 7.40,
                            "p75": 7.80, "p90": 8.00, "p99": 8.50,
                        },
                        "256": {"avg": 8.20, "p50": 8.10},
                    },
                    "per_bs_verdict": None,  # round 1 — no verdicts
                },
                "integration": {},
            },
        ],
        "cumulative_e2e_speedup": 1.0,
    },
}

V4_REPROFILE_REGRESSED = {
    "campaign": {
        "rounds": [
            {
                "round_id": 1,
                "baseline": {
                    "e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}},
                    "per_bs_verdict": None,
                },
                "integration": {
                    "e2e_latency_combined": {"128": {"avg": 6.42, "p50": 6.40}},
                    "per_bs_verdict": {"128": "PASS"},
                },
            },
            {
                "round_id": 2,
                "baseline": {
                    "e2e_latency": {
                        "128": {"avg": 6.42, "p50": 6.40},
                        "512": {"avg": 9.90, "p50": 9.80},
                    },
                    "per_bs_verdict": {"128": "PASS", "512": "REGRESSED"},
                },
                "integration": {},
            },
        ],
        "cumulative_e2e_speedup": 1.193,
    },
}

V3_LEGACY_COMPAT = {
    "campaign": {
        "rounds": [
            {
                "round_id": 1,
                "baseline": {},
                "integration": {
                    "combined_e2e_result": {
                        "latency_baseline_s": 7.66,
                        "per_bs_verdict": {"128": "PASS"},
                    },
                },
            },
        ],
        # v3 stored cumulative_speedup_vs_round1; normalizer migrates it in prod,
        # but the frontend should still fall back cleanly when only this field is present.
        "cumulative_speedup_vs_round1": 1.12,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Zone C — Task 9 tests: direct read of baseline.e2e_latency map into
# baselineByRound (replaces the 112-line cascade).
# ────────────────────────────────────────────────────────────────────────────

class TestBaselineByRoundFromMap:
    """Task 9 — baselineByRound populated from baseline.e2e_latency map."""

    # Test 30
    def test_baseline_hero_reads_smallest_bs_avg_from_map(self):
        catalog = _enrich_state(V4_BASELINE_MULTI_BS)
        by_round = catalog.get("baselineByRound", {})
        entry = by_round.get("1") or by_round.get(1)
        assert entry is not None, f"expected entry for round 1, got keys {list(by_round.keys())}"
        # smallest BS is 128 → avg 7.66 → 7660 ms
        assert entry["primaryBsLatencyMs"] == pytest.approx(7660.0, rel=1e-3)
        assert entry["primaryBs"] == 128
        assert entry["source"] == "baseline_e2e_latency"

    # Test 31
    def test_baseline_hero_fallback_to_combined_e2e_result(self):
        catalog = _enrich_state(V3_LEGACY_COMPAT)
        by_round = catalog.get("baselineByRound", {})
        entry = by_round.get("1") or by_round.get(1)
        assert entry is not None, "expected legacy-compat entry for round 1"
        # 7.66 → 7660 ms, source flagged as fallback
        assert entry["primaryBsLatencyMs"] == pytest.approx(7660.0, rel=1e-3)

    # Test 32
    def test_baseline_hero_prefers_e2e_latency_over_combined_e2e_result(self):
        # Both present — new map-based baseline must win.
        state = {
            "campaign": {
                "rounds": [{
                    "round_id": 1,
                    "baseline": {
                        "e2e_latency": {"128": {"avg": 7.66, "p50": 7.55}},
                        "per_bs_verdict": None,
                    },
                    "integration": {
                        "combined_e2e_result": {"latency_baseline_s": 9.00},
                    },
                }],
                "cumulative_e2e_speedup": 1.0,
            },
        }
        catalog = _enrich_state(state)
        entry = catalog["baselineByRound"].get("1") or catalog["baselineByRound"].get(1)
        assert entry["primaryBsLatencyMs"] == pytest.approx(7660.0, rel=1e-3)
        assert entry["source"] == "baseline_e2e_latency"

    def test_baseline_map_carries_full_percentile_entries(self):
        """batchSizes on the catalog entry preserves the full percentile map
        so the tooltip can render p10/p25/p50/p75/p90/p99 without re-walking state."""
        catalog = _enrich_state(V4_BASELINE_MULTI_BS)
        entry = catalog["baselineByRound"].get("1") or catalog["baselineByRound"].get(1)
        bs = entry["batchSizes"]
        # expect a map of BS key → percentile dict
        assert "128" in bs or 128 in bs, f"batchSizes missing 128 key: {bs}"
        e128 = bs.get("128") or bs.get(128)
        assert e128["avg"] == pytest.approx(7.66)
        assert e128["p90"] == pytest.approx(8.00)

    def test_baseline_map_carries_per_bs_verdict(self):
        catalog = _enrich_state(V4_REPROFILE_REGRESSED)
        entry_r2 = catalog["baselineByRound"].get("2") or catalog["baselineByRound"].get(2)
        pbv = entry_r2.get("perBsVerdict")
        assert pbv is not None, "perBsVerdict missing on round 2 re-profile"
        # JSON keys are strings
        assert pbv.get("128") == "PASS" or pbv.get(128) == "PASS"
        assert pbv.get("512") == "REGRESSED" or pbv.get(512) == "REGRESSED"


# ────────────────────────────────────────────────────────────────────────────
# Zone C — Task 10 tests: _baselineTooltip renders per-BS percentile table
# with verdict coloring on re-profile rounds.
# ────────────────────────────────────────────────────────────────────────────

class TestBaselineTooltipPercentileTable:
    """Task 10 — _baselineTooltip shows all BS rows with p50/p90 where present.

    _baselineTooltip is a closure inside renderCircuitBoard; we drive it by
    simulating its core behaviour through a pure helper that the
    implementation exposes on `CircuitBoard.baselineTooltipHtml`.
    """

    def _tooltip_html(self, state: dict, round_id: int) -> str:
        script = f"""
const state = {json.dumps(state)};
if (!state._catalog) state._catalog = {{ entries: {{}} }};
CircuitBoard.enrichFromCatalog(state);
const html = CircuitBoard.baselineTooltipHtml(state, {round_id});
console.log(html || '');
"""
        return _cb_harness(script)

    # Test 37
    def test_baseline_tooltip_shows_all_bs_percentiles(self):
        html = self._tooltip_html(V4_BASELINE_MULTI_BS, 1)
        assert "128" in html, f"tooltip missing BS 128: {html!r}"
        assert "256" in html, f"tooltip missing BS 256: {html!r}"
        # avg value for 128 (7.66s → 7660.0ms or 7.66s depending on formatter)
        assert "7.66" in html or "7660" in html, f"tooltip missing avg: {html!r}"
        assert "8.00" in html or "8000" in html, f"tooltip missing p90: {html!r}"

    def test_baseline_tooltip_verdict_color_on_regressed(self):
        html = self._tooltip_html(V4_REPROFILE_REGRESSED, 2)
        # REGRESSED row must be rendered with a distinguishable class / tag
        assert "REGRESSED" in html.upper() or "regress" in html.lower(), (
            f"tooltip missing REGRESSED verdict marker: {html!r}"
        )

    def test_baseline_tooltip_round_1_has_no_verdict_legend(self):
        """Round 1 has no `per_bs_verdict` → tooltip must NOT render the verdict legend."""
        html = self._tooltip_html(V4_BASELINE_MULTI_BS, 1)
        assert "REGRESSED" not in html.upper(), (
            f"round 1 tooltip must not advertise verdicts: {html!r}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Zone C — Task 11 tests: campaign-app.js _baselineHeroLatency() simplification
#
# _baselineHeroLatency is a method on the Alpine campaign-app data object; we
# test it by extracting the function source and running it in isolation with a
# mock `this` context.
# ────────────────────────────────────────────────────────────────────────────

class TestBaselineHeroLatencyOverview:
    """Task 11 — campaign-app.js reads baseline.e2e_latency map first."""

    def _run_overview_helper(self, state: dict, current_round: int) -> dict | None:
        """Extract `_baselineHeroLatency` from campaign-app.js and invoke it."""
        src = CAMPAIGN_APP_JS.read_text()
        # Regex-extract the method body so we can bind a clean `this` without
        # pulling in the whole 6k-line Alpine component.
        m = re.search(
            r"_baselineHeroLatency\s*\(\s*\)\s*\{(?P<body>.*?)\n\s{8}\},",
            src,
            re.DOTALL,
        )
        if not m:
            pytest.skip("_baselineHeroLatency method not found in campaign-app.js")
        body = m.group("body")
        # Replace internal helper calls the method depends on (matchRound, _catalogEntries)
        # with simple stubs because we only exercise the primary-source branch here.
        script = f"""
const state = {json.dumps(state)};
const self = {{
    campaignState: state,
    currentRound: {current_round},
    artifactCatalog: state._catalog || null,
}};
function matchRound(e, r) {{ return e.round === r; }}
function _catalogEntries(cat) {{
    if (!cat || !cat.entries) return [];
    return Object.entries(cat.entries).map(([k, v]) => ({{ path: k, ...v }}));
}}
const _baselineHeroLatency = function () {{{body}}};
const result = _baselineHeroLatency.call(self);
console.log(JSON.stringify(result));
"""
        out = run_js(script)
        return json.loads(out) if out and out != "null" else None

    # Test 36
    def test_campaign_app_baseline_hero_reads_new_field(self):
        result = self._run_overview_helper(V4_BASELINE_MULTI_BS, 1)
        assert result is not None
        # avg 7.66s → 7660.0 ms
        assert result["value"] == "7660.0"
        assert result["unit"] == "ms"
        assert "BS128" in result["label"] or "BS 128" in result["label"]

    # Test 35 (covered in part by the numeric assertion above)
    def test_campaign_app_falls_back_to_legacy_latency_baseline_s(self):
        result = self._run_overview_helper(V3_LEGACY_COMPAT, 1)
        assert result is not None
        assert result["value"] == "7660.0"

    def test_campaign_app_returns_null_when_no_data(self):
        empty_state = {"campaign": {"rounds": [{"round_id": 1, "baseline": {}, "integration": {}}]}}
        result = self._run_overview_helper(empty_state, 1)
        assert result is None


# ────────────────────────────────────────────────────────────────────────────
# Zone C — Task 12 tests: speedup helpers simplified + delta_pp computed client-side
# ────────────────────────────────────────────────────────────────────────────

class TestSpeedupHelpers:
    """Task 12 — post-normalizer, these helpers see scalar values and
    should return them directly (no object walk needed on hot path)."""

    def _call_scalar(self, fn: str, arg) -> float | None:
        script = f"""
const v = CircuitBoard.{fn}({json.dumps(arg)});
console.log(v === null || v === undefined ? 'null' : String(v));
"""
        out = _cb_harness(script)
        if out == "null":
            return None
        return float(out)

    # Test 33
    def test_kernel_speedup_scalar_reads_number_directly(self):
        v = self._call_scalar("kernelSpeedupScalar", 1.35)
        assert v == pytest.approx(1.35)

    def test_kernel_speedup_scalar_null_returns_null(self):
        assert self._call_scalar("kernelSpeedupScalar", None) is None

    def test_kernel_speedup_scalar_legacy_object_still_works(self):
        """Legacy pre-normalizer object shape must still yield a scalar
        (back-compat for states loaded before the normalizer ran)."""
        v = self._call_scalar(
            "kernelSpeedupScalar",
            {"cold_bs8": 1.42, "warm_bs8": 1.28, "target": 1.4},
        )
        assert v == pytest.approx(1.42)

    # Test 34
    def test_e2e_speedup_scalar_reads_number(self):
        v = self._call_scalar("e2eSpeedupScalar", 1.18)
        assert v == pytest.approx(1.18)

    def test_e2e_speedup_scalar_legacy_object_still_works(self):
        v = self._call_scalar("e2eSpeedupScalar", {"speedup_x": 1.18})
        assert v == pytest.approx(1.18)


class TestDeltaPpClientSide:
    """Task 12 — delta_pp is computed client-side from baseline + integration
    map entries (was stored on combined_e2e_result in v3).

    We expose this via a new helper `CircuitBoard.computeDeltaPpFromMaps(
    baselineMap, combinedMap)`.
    """

    def _delta(self, baseline_map, combined_map):
        script = f"""
const b = {json.dumps(baseline_map)};
const c = {json.dumps(combined_map)};
const v = CircuitBoard.computeDeltaPpFromMaps(b, c);
console.log(v === null || v === undefined ? 'null' : String(v));
"""
        out = _cb_harness(script)
        return None if out == "null" else float(out)

    # Test 35
    def test_delta_pp_computed_in_frontend(self):
        delta = self._delta(
            {"128": {"avg": 7.66, "p50": 7.55}},
            {"128": {"avg": 6.50, "p50": 6.40}},
        )
        assert delta is not None
        # (7.66 - 6.5) / 7.66 * 100 ≈ 15.14
        assert delta == pytest.approx(15.14, abs=0.05)

    def test_delta_pp_negative_when_regressed(self):
        delta = self._delta(
            {"128": {"avg": 7.66, "p50": 7.55}},
            {"128": {"avg": 8.20, "p50": 8.10}},
        )
        assert delta is not None
        assert delta < 0

    def test_delta_pp_picks_smallest_bs_by_default(self):
        delta = self._delta(
            {"128": {"avg": 7.66, "p50": 7.55}, "512": {"avg": 9.10, "p50": 9.00}},
            {"128": {"avg": 6.50, "p50": 6.40}, "512": {"avg": 8.20, "p50": 8.10}},
        )
        assert delta == pytest.approx(15.14, abs=0.05)

    def test_delta_pp_returns_null_on_missing_data(self):
        assert self._delta(None, {"128": {"avg": 6.5, "p50": 6.4}}) is None
        assert self._delta({}, {}) is None


class TestCumulativeReadSimplified:
    """Task 12 — after the normalizer always injects campaign.cumulative_e2e_speedup,
    the frontend's chained fallback (`|| cumulative_speedup_vs_round1 || 1.0`)
    is no longer needed.

    Grep-based assertion: the updated module must read through
    `campaign.cumulative_e2e_speedup ?? 1.0` (or `|| 1`) and must NOT retain
    the legacy field name `cumulative_speedup_vs_round1`.
    """

    def test_circuit_board_does_not_read_cumulative_speedup_vs_round1(self):
        src = CIRCUIT_BOARD_JS.read_text()
        # The legacy field name should not appear as a read; if it does,
        # either the fallback chain was not removed, or an agent prompt needs cleanup.
        # Task 12 explicitly calls for this fallback chain to be removed from
        # circuit-board.js lines 257, 897, 1002.
        matches = re.findall(r"cumulative_speedup_vs_round1", src)
        assert not matches, (
            "circuit-board.js still references legacy cumulative_speedup_vs_round1 "
            f"(found {len(matches)} occurrences); Task 12 requires removing them."
        )

    # Functions allowed to read the v3 field name. `_normalizeCumulativeSpeedup`
    # ports the server's `_normalize_speedup_field`; `_integrationStageData`
    # walks the documented `cumulative_speedup_after ?? v3 ?? legacy` chain.
    SANCTIONED_READERS = ("_normalizeCumulativeSpeedup", "_integrationStageData")

    @staticmethod
    def _function_span(src, name):
        """(start, end) character span of a JS function body, by brace depth."""
        match = re.search(rf"(?:function\s+)?{re.escape(name)}\s*\([^)]*\)\s*\{{", src)
        assert match, f"campaign-app.js must define {name}"
        depth, index = 0, match.end() - 1
        while index < len(src):
            if src[index] == "{":
                depth += 1
            elif src[index] == "}":
                depth -= 1
                if depth == 0:
                    return match.start(), index + 1
            index += 1
        raise AssertionError(f"unbalanced braces in {name}")

    def test_campaign_app_reads_cumulative_speedup_vs_round1_only_where_sanctioned(self):
        # Artifact-Layout-V2 Task 5 supersedes the original Task 12 constraint:
        # the server now keeps state.json verbatim (no `_normalize_speedup_field`
        # call), so the FE owns the v3 → legacy field fallback. Scope, not count:
        # a comment mentioning the field must not change the verdict, and a new
        # read outside the sanctioned readers must fail.
        # See .claude/plans/artifact-layout-v2-frontend.md §Task 5.
        src = CAMPAIGN_APP_JS.read_text()
        # Blank out comments so prose mentions cannot influence the result.
        # Newlines are kept so reported line numbers stay accurate.
        code = re.sub(
            r"/\*.*?\*/|//[^\n]*",
            lambda m: re.sub(r"[^\n]", " ", m.group(0)),
            src,
            flags=re.S,
        )

        spans = [self._function_span(code, name) for name in self.SANCTIONED_READERS]
        reads = list(re.finditer(r"cumulative_speedup_vs_round1", code))

        assert reads, (
            "Task 5 requires the FE to read cumulative_speedup_vs_round1 — "
            "no non-comment read found."
        )
        leaked = [
            code[: read.start()].count("\n") + 1
            for read in reads
            if not any(start <= read.start() < end for start, end in spans)
        ]
        assert not leaked, (
            "campaign-app.js reads cumulative_speedup_vs_round1 outside "
            f"{', '.join(self.SANCTIONED_READERS)} at line(s) {leaked}."
        )
