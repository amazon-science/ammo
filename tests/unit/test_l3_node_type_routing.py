# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for L3 node-type routing (Bug 3 fix).

Covers the `l3NodeType` getter on the Alpine `campaignApp` factory and the
`_trackOverviewData()` method. The bug: when `currentNode` is one of the bare
stage names — `baseline`, `mining`, `debate`, `integration` — the getter
returned `'track'` (because the only branch was `node.startsWith('stage-')`),
causing the L3 overview panel to render the TRACK hero with `N/A / N/A /
NOT STARTED` instead of the correct stage-specific hero.

The fix: `l3NodeType` (and `_buildL3CatalogData`) must recognise both the
`stage-N` and the bare-name forms and map them to the same node type.

Live repro (Playwright, 2026-04-21):
  URL  #campaigns/57a060da-.../1/debate
  Result — overview shows "N/A / N/A / NOT STARTED" instead of the debate
  champions/result hero. `_trackOverviewData()` returns a ghost "NOT STARTED"
  stub because `debate` is not present in `state.parallel_tracks`.
"""

import subprocess
import tempfile
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"


def run_js(script: str) -> str:
    """Write script to a temp file and run with `node`.

    The JS src is ~4000 lines — `node -e` hits the argv length limit on
    Linux, so we route through a temp file (same pattern as
    tests/unit/test_fe_memo_keys.py).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            ["node", path], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        return result.stdout.strip()
    finally:
        os.unlink(path)


def run_on_app(setup_js: str, expr: str) -> str:
    """Instantiate the campaignApp factory, run `setup_js` against the
    `app` object, then evaluate `expr` and print it (JSON-encoded).
    """
    js_src = CAMPAIGN_APP_JS.read_text()
    script = f"""
const app = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{ data: (name, fn) => {{ _data = fn(); }}, directive: () => {{}} }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}}, removeItem: () => {{}} }};
    const window = {{ addEventListener: () => {{}}, removeEventListener: () => {{}},
                      location: {{ hash: '' }} }};
    const document = {{ addEventListener: (event, cb) => {{
        if (event === 'alpine:init') _initCb = cb;
    }} }};
    {js_src}
    if (_initCb) _initCb();
    return _data;
}})();
{setup_js}
const __r = (() => {{ try {{ return {expr}; }} catch (e) {{ return {{__err: String(e)}}; }} }})();
console.log(JSON.stringify(__r));
"""
    return run_js(script)


# ---------------------------------------------------------------------------
# l3NodeType getter
# ---------------------------------------------------------------------------

class TestL3NodeTypeRouting:
    """l3NodeType must map `stage-N` AND bare stage names to the same type."""

    def test_stage_prefixed_debate(self):
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'stage-2';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "debate"

    def test_bare_debate_is_debate_not_track(self):
        """Bug 3 — `/1/debate` must resolve to 'debate', not 'track'."""
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'debate';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "debate"

    def test_bare_baseline(self):
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'baseline';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "baseline"

    def test_bare_mining(self):
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'mining';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "mining"

    def test_bare_integration(self):
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'integration';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "integration"

    def test_track_op_id_still_track(self):
        """Regression guard — op_id clicks remain typed 'track'."""
        out = run_on_app(
            "app.currentLevel = 3; app.currentNode = 'C1';",
            "app.l3NodeType",
        )
        assert json.loads(out) == "track"


# ---------------------------------------------------------------------------
# _trackOverviewData() — should NOT be called for bare stage-name nodes
# ---------------------------------------------------------------------------

class TestTrackOverviewDataDispatch:
    """_trackOverviewData() must return null for non-track L3 nodes.

    The template `<template x-if="l3NodeType === 'track' && _trackOverviewData()">`
    falls through for non-track nodes. For BARE stage names (debate, baseline,
    mining, integration), the fix ensures `l3NodeType` is NOT 'track', and
    `_trackOverviewData()` returns null on the guard clause.

    The live-state values come from session 57a060da state.json (2026-04-21).
    """

    # Snapshot of the live state (trimmed to what the getter reads).
    LIVE_STATE = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "parallel_tracks": {
                        "tracks": {
                            "C1": {
                                "classification": "lossless",
                                "correctness": True,
                                "kernel_speedup": 1.3416,
                                "kernel_speedup_warm": 1.3416,
                                "status": "IN_PROGRESS",
                                "verdict": None,
                            },
                            "C3": {
                                "classification": "lossless",
                                "correctness": True,
                                "kernel_speedup": 1.4665,
                                "e2e_speedup": 1.01213,
                                "status": "PASS",
                                "verdict": "PASS",
                            },
                        },
                    },
                },
            ],
            "shipped_optimizations": [],
        },
        "parallel_tracks": {
            "C1": {
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.3416,
                "kernel_speedup_warm": 1.3416,
                "status": "IN_PROGRESS",
                "verdict": None,
            },
            "C3": {
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.4665,
                "e2e_speedup": 1.01213,
                "status": "PASS",
                "verdict": "PASS",
            },
        },
        "stage": "4_5_parallel_tracks",
    }

    def _setup(self, node: str) -> str:
        state = json.dumps(self.LIVE_STATE)
        return (
            f"app.currentLevel = 3;"
            f"app.currentRound = 1;"
            f"app.currentNode = {json.dumps(node)};"
            f"app.campaignState = {state};"
        )

    def test_bare_debate_overview_is_null(self):
        """Bug 3 — debate node must NOT produce a bogus 'NOT STARTED' stub."""
        out = run_on_app(self._setup("debate"), "app._trackOverviewData()")
        assert json.loads(out) is None

    def test_track_C1_returns_live_kernel_speedup(self):
        """Regression guard — track data still loads from state.parallel_tracks."""
        out = run_on_app(self._setup("C1"), "app._trackOverviewData()")
        parsed = json.loads(out)
        assert parsed is not None
        assert parsed["opId"] == "C1"
        assert parsed["status"] == "IN_PROGRESS"
        assert parsed["kernel"] == 1.3416
        assert parsed["classification"] == "LOSSLESS"

    def test_track_C3_returns_live_values(self):
        out = run_on_app(self._setup("C3"), "app._trackOverviewData()")
        parsed = json.loads(out)
        assert parsed is not None
        assert parsed["opId"] == "C3"
        assert parsed["status"] == "PASS"
        assert parsed["kernel"] == 1.4665
        assert parsed["e2e"] == 1.01213

    def test_bare_baseline_overview_is_null(self):
        out = run_on_app(self._setup("baseline"), "app._trackOverviewData()")
        assert json.loads(out) is None

    def test_bare_mining_overview_is_null(self):
        out = run_on_app(self._setup("mining"), "app._trackOverviewData()")
        assert json.loads(out) is None

    def test_bare_integration_overview_is_null(self):
        out = run_on_app(self._setup("integration"), "app._trackOverviewData()")
        assert json.loads(out) is None


# ---------------------------------------------------------------------------
# L3 round switches and debate artifact scoping
# ---------------------------------------------------------------------------

class TestL3RoundSwitchAndDebateArtifacts:
    """Regression coverage for AMMO porting-gap L3 round/artifact bugs."""

    def test_activate_l3_reinitializes_sections_when_round_changes_same_node(self):
        setup = """
        globalThis.setTimeout = () => 0;
        app.currentLevel = 3;
        app.currentSessionId = 'campaign-a';
        app.currentRound = 1;
        app.currentNode = 'stage-2';
        app.campaignState = {
            campaign: {
                current_round: 4,
                current_stage: '3_debate',
                rounds: [{}, {}, {}, {}],
            },
        };
        app._triggerLevelFade = () => {};
        app.stopPolling = () => {};
        app.loadCampaignDetail = () => { throw new Error('unexpected reload'); };
        app._buildL3CatalogData = () => ({});
        app.__nextTickCalls = 0;
        app.__initCalls = 0;
        app.$nextTick = (cb) => { app.__nextTickCalls += 1; cb(); };
        app.initL3Sections = () => { app.__initCalls += 1; };
        app.startL3Tour = () => {};
        app._activateL3('campaign-a', 4, 'stage-2');
        """
        out = run_on_app(
            setup,
            "({ nextTick: app.__nextTickCalls, init: app.__initCalls, round: app.currentRound, node: app.currentNode })",
        )
        parsed = json.loads(out)
        assert parsed == {"nextTick": 1, "init": 1, "round": 4, "node": "stage-2"}

    def test_round4_debate_uses_canonical_directory_root(self):
        setup = """
        app.currentLevel = 3;
        app.currentRound = 4;
        app.currentNode = 'stage-2';
        """
        out = run_on_app(setup, "app._l3ArtifactRootSpecs()")
        assert json.loads(out) == [
            {"label": "DEBATE", "path": "rounds/4/debate", "type": "directory"}
        ]
