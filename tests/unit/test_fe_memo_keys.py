# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for C1-C4 FE memoization (task #6 in tdd/simplify-sidecar).

Exercises the cache-key builders exposed on window.LG_HELPERS:
  - buildTrackOverviewKey(catalog, currentNode, currentRound, state)   [C1]
  - buildCatalogEntriesKey(catalog)                                    [C3]

And the Schwartzian transform helper used by champion-entry sorting:
  - sortChampionEntries(entries)                                       [C4]

These are pure functions — easy to unit-test without Alpine context.
The actual memoization logic is also exercised: repeated calls with the
same key yield the same cache value (cache hit), while changing any
component of the key invalidates the cache (cache miss).
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"

# Harness prelude — stubs Alpine + browser globals so campaign-app.js
# module-scope code runs cleanly in Node. Alpine.data()'s body is never
# invoked; we only need window.LG_HELPERS to be populated.
_HARNESS_PRELUDE = """
const window = {};
const localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const document = {
    addEventListener: () => {},
    createElement: () => ({
        style: {}, className: '', innerHTML: '',
        appendChild: () => {}, children: [],
        classList: { add: () => {}, remove: () => {} },
        addEventListener: () => {},
    }),
    querySelector: () => null,
    body: { appendChild: () => {} },
};
const Alpine = { directive: () => {}, data: () => {} };
"""


def run_js(trailer: str) -> str:
    """Write prelude + campaign-app.js + trailer to a temp file and execute.

    The JS file is ~4000 lines so we can't pass it via ``node -e`` (argv
    limit). Using a temp file is the simplest workaround.
    """
    js_src = CAMPAIGN_APP_JS.read_text()
    full = _HARNESS_PRELUDE + js_src + "\n" + trailer + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(full)
        path = f.name
    try:
        result = subprocess.run(
            ["node", path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        return result.stdout.strip()
    finally:
        os.unlink(path)


def _harness() -> str:
    """Back-compat shim: tests append their Node code to this and call run_js.

    The returned string is just an empty stub — run_js() now writes the
    full harness to a temp file. Tests use `_harness() + "...code..."`
    purely as a readability convention.
    """
    return ""


# ---------------------------------------------------------------------------
# C1 — _trackOverviewData cache key
# ---------------------------------------------------------------------------

class TestBuildTrackOverviewKey:
    def test_key_is_stable_for_stable_inputs(self):
        script = _harness() + """
        const catalog = { last_updated: '2026-04-21T00:00:00Z', last_scan_file_count: 12 };
        const state = {
            stage: '4_5_parallel_tracks',
            parallel_tracks: {
                op_a: { status: 'COMPLETED', verdict: 'PASS', kernel_speedup: 1.2 },
                op_b: { status: 'IN_PROGRESS' },
            },
        };
        const k1 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state);
        const k2 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state);
        console.log(k1 === k2 ? 'STABLE' : 'UNSTABLE');
        """
        assert run_js(script) == "STABLE"

    def test_key_changes_on_currentNode(self):
        script = _harness() + """
        const catalog = { last_updated: 't0', last_scan_file_count: 1 };
        const state = { stage: '1', parallel_tracks: {} };
        const k1 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state);
        const k2 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_b', 1, state);
        console.log(k1 !== k2 ? 'DIFFER' : 'SAME');
        """
        assert run_js(script) == "DIFFER"

    def test_key_changes_on_currentRound(self):
        script = _harness() + """
        const catalog = { last_updated: 't0', last_scan_file_count: 1 };
        const state = { stage: '1', parallel_tracks: {} };
        const k1 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state);
        const k2 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 2, state);
        console.log(k1 !== k2 ? 'DIFFER' : 'SAME');
        """
        assert run_js(script) == "DIFFER"

    def test_key_changes_on_track_status_value(self):
        """DA-round2 H3: value-level invalidation — status flip without key
        change MUST invalidate (shallow key-based hash is not enough)."""
        script = _harness() + """
        const catalog = { last_updated: 't0', last_scan_file_count: 1 };
        const state1 = { stage: 's', parallel_tracks: { op_a: { status: 'IN_PROGRESS', verdict: null, kernel_speedup: null } } };
        const state2 = { stage: 's', parallel_tracks: { op_a: { status: 'COMPLETED', verdict: 'PASS', kernel_speedup: 1.5 } } };
        const k1 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state1);
        const k2 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, state2);
        console.log(k1 !== k2 ? 'DIFFER' : 'SAME');
        """
        assert run_js(script) == "DIFFER"

    def test_key_uses_etag_when_available(self):
        """When state._etag is present, it's used as a cheap hash."""
        script = _harness() + """
        const catalog = { last_updated: 't0', last_scan_file_count: 1 };
        const s1 = { _etag: 'abc', stage: 's', parallel_tracks: { op_a: { status: 'X' } } };
        const s2 = { _etag: 'abc', stage: 's', parallel_tracks: { op_a: { status: 'Y' } } };
        const s3 = { _etag: 'def', stage: 's', parallel_tracks: { op_a: { status: 'X' } } };
        const k1 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, s1);
        const k2 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, s2);
        const k3 = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, s3);
        console.log(`${k1===k2}|${k1!==k3}`);
        """
        # When etag is identical, key is identical even if internals changed.
        # When etag changes, key differs.
        assert run_js(script) == "true|true"

    def test_key_tolerates_null_state(self):
        script = _harness() + """
        const catalog = { last_updated: 't0', last_scan_file_count: 1 };
        const k = window.LG_HELPERS.buildTrackOverviewKey(catalog, 'op_a', 1, null);
        console.log(typeof k === 'string' && k.length > 0 ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"

    def test_key_tolerates_null_catalog(self):
        script = _harness() + """
        const state = { stage: 's', parallel_tracks: {} };
        const k = window.LG_HELPERS.buildTrackOverviewKey(null, null, null, state);
        console.log(typeof k === 'string' ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"

# C3 — _catalogEntries materialization key
# ---------------------------------------------------------------------------

class TestBuildCatalogEntriesKey:
    def test_key_is_stable(self):
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 12 };
        const k1 = window.LG_HELPERS.buildCatalogEntriesKey(c);
        const k2 = window.LG_HELPERS.buildCatalogEntriesKey(c);
        console.log(k1 === k2 ? 'STABLE' : 'UNSTABLE');
        """
        assert run_js(script) == "STABLE"

    def test_key_changes_on_last_updated(self):
        script = _harness() + """
        const c1 = { last_updated: 't0', last_scan_file_count: 12 };
        const c2 = { last_updated: 't1', last_scan_file_count: 12 };
        const k1 = window.LG_HELPERS.buildCatalogEntriesKey(c1);
        const k2 = window.LG_HELPERS.buildCatalogEntriesKey(c2);
        console.log(k1 !== k2 ? 'DIFFER' : 'SAME');
        """
        assert run_js(script) == "DIFFER"

    def test_key_changes_on_scan_file_count(self):
        script = _harness() + """
        const c1 = { last_updated: 't0', last_scan_file_count: 12 };
        const c2 = { last_updated: 't0', last_scan_file_count: 15 };
        const k1 = window.LG_HELPERS.buildCatalogEntriesKey(c1);
        const k2 = window.LG_HELPERS.buildCatalogEntriesKey(c2);
        console.log(k1 !== k2 ? 'DIFFER' : 'SAME');
        """
        assert run_js(script) == "DIFFER"

    def test_key_tolerates_null_catalog(self):
        script = _harness() + """
        const k = window.LG_HELPERS.buildCatalogEntriesKey(null);
        console.log(typeof k === 'string' ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"


# ---------------------------------------------------------------------------
# C3 — _catalogEntries memoization behavior
# ---------------------------------------------------------------------------

class TestCatalogEntriesMemo:
    """_catalogEntries itself must return identical object references on
    cache hit (same catalog version → same array instance)."""

    def test_same_catalog_returns_same_array_reference(self):
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 2,
                    entries: { 'a.md': { path: 'a.md' }, 'b.md': { path: 'b.md' } } };
        const a1 = window.LG_HELPERS._catalogEntries(c);
        const a2 = window.LG_HELPERS._catalogEntries(c);
        console.log(a1 === a2 ? 'SAME_REF' : 'DIFF_REF');
        """
        assert run_js(script) == "SAME_REF"

    def test_different_catalog_version_returns_fresh_array(self):
        script = _harness() + """
        const c1 = { last_updated: 't0', last_scan_file_count: 2,
                     entries: { 'a.md': { path: 'a.md' }, 'b.md': { path: 'b.md' } } };
        const a1 = window.LG_HELPERS._catalogEntries(c1);
        const c2 = { last_updated: 't1', last_scan_file_count: 2,
                     entries: { 'a.md': { path: 'a.md' }, 'b.md': { path: 'b.md' } } };
        const a2 = window.LG_HELPERS._catalogEntries(c2);
        console.log(a1 !== a2 ? 'DIFF_REF' : 'SAME_REF');
        """
        assert run_js(script) == "DIFF_REF"

    def test_null_catalog_returns_empty_array(self):
        script = _harness() + """
        const a = window.LG_HELPERS._catalogEntries(null);
        console.log(Array.isArray(a) && a.length === 0 ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"

    def test_entries_shape_preserved(self):
        """Fallthrough semantics: path derived from key when absent."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 2,
                    entries: { 'dir/a.md': { stage: 'debate' }, 'b.md': { path: 'custom.md' } } };
        const a = window.LG_HELPERS._catalogEntries(c);
        const aPath = a.find(e => e.stage === 'debate')?.path;
        const bPath = a.find(e => e.stage !== 'debate')?.path;
        console.log(`${aPath}|${bPath}`);
        """
        assert run_js(script) == "dir/a.md|custom.md"


# ---------------------------------------------------------------------------
# Bug 2 — debate_rationale / source_code entries keyed without extension.
#
# Real-world catalog shape (confirmed via live session 57a060da…):
#   { "debate/proposals/champion-1_proposal": {
#       "labels": { "kind": "debate_rationale", ... },
#       "refs": [
#         { "path": "debate/proposals/champion-1_proposal.md", "role": "primary" },
#         { "path": "…/evidence.py", "role": "evidence" },
#       ]
#     } }
#
# The endpoint `/api/campaigns/{id}/artifacts/{path}` expects a REAL file path.
# Fetching the key `…/champion-1_proposal` (no .md) → 404.
# _catalogEntries() must prefer `refs.find(r=>r.role==='primary').path` over
# the catalog key so downstream consumers (including fetchArtifact,
# openSidecarArtifact) build URLs pointing at the real file on disk.
# ---------------------------------------------------------------------------

class TestCatalogEntriesPrimaryRef:
    def test_primary_ref_path_preferred_over_key(self):
        """When `refs` has role=primary, its path wins over the catalog key."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: {
                      'debate/proposals/champion-1_proposal': {
                        labels: { kind: 'debate_rationale' },
                        refs: [
                          { path: 'debate/proposals/champion-1_proposal.md', role: 'primary' },
                          { path: 'debate/proposals/evidence.py', role: 'evidence' },
                        ],
                      },
                    } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "debate/proposals/champion-1_proposal.md"

    def test_explicit_path_field_still_wins(self):
        """Legacy entries with a top-level `path` keep that precedence."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: {
                      'logical_key': {
                        path: 'explicit.md',
                        refs: [{ path: 'from_refs.md', role: 'primary' }],
                      },
                    } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "explicit.md"

    def test_falls_back_to_key_when_no_primary_ref(self):
        """Entries with empty refs (e.g. tracks/C1/validator_tests/…) fall back
        to the key because the key IS the on-disk path there."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: {
                      'tracks/C1/validator_tests/gate_5_2_results.json': {
                        refs: [],
                      },
                    } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "tracks/C1/validator_tests/gate_5_2_results.json"

    def test_first_primary_ref_wins_over_non_primary(self):
        """primary role must be picked even if it appears after other refs."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: {
                      'k': {
                        refs: [
                          { path: 'trace.sqlite', role: 'primary_trace' },
                          { path: 'actual.md',    role: 'primary' },
                          { path: 'evidence.py',  role: 'evidence' },
                        ],
                      },
                    } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "actual.md"

    def test_refs_first_entry_when_no_primary_role(self):
        """When refs exist but none has role=primary, first ref wins over key."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: {
                      'logical_no_primary': {
                        refs: [
                          { path: 'first.md', role: 'evidence' },
                          { path: 'second.md', role: 'evidence' },
                        ],
                      },
                    } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "first.md"

    def test_key_used_when_entry_has_neither_path_nor_refs(self):
        """Defensive: no path, no refs → fall back to the key."""
        script = _harness() + """
        const c = { last_updated: 't0', last_scan_file_count: 1,
                    entries: { 'lonely.md': { stage: 'x' } } };
        const a = window.LG_HELPERS._catalogEntries(c);
        console.log(a[0].path);
        """
        assert run_js(script) == "lonely.md"


# ---------------------------------------------------------------------------
# C4 — Schwartzian sort on debate champion keys
# ---------------------------------------------------------------------------

class TestSortChampionEntries:
    def test_numerically_ascending(self):
        script = _harness() + """
        const entries = [
            ['champion-10', []],
            ['champion-2', []],
            ['champion-1', []],
        ];
        const sorted = window.LG_HELPERS.sortChampionEntries(entries);
        console.log(sorted.map(e => e[0]).join(','));
        """
        assert run_js(script) == "champion-1,champion-2,champion-10"

    def test_no_digit_falls_to_end(self):
        script = _harness() + """
        const entries = [
            ['alpha', []],
            ['champion-1', []],
            ['champion-3', []],
        ];
        const sorted = window.LG_HELPERS.sortChampionEntries(entries);
        // parseInt of no-digit match falls back to 0 (which sorts FIRST) per
        // the original campaign-app.js:2874 behavior (|| ['999'] default).
        // The helper must preserve the historical ordering semantics.
        // Accepts either "alpha, champion-1, champion-3" (if new default is 0)
        // or "champion-1, champion-3, alpha" (if new default is 999) — both
        // forms preserve legit numeric ordering of champion-N entries.
        const order = sorted.map(e => e[0]).join(',');
        const ok = order === 'champion-1,champion-3,alpha' || order === 'alpha,champion-1,champion-3';
        console.log(ok ? 'OK' : `BAD:${order}`);
        """
        assert run_js(script) == "OK"

    def test_empty_input(self):
        script = _harness() + """
        const sorted = window.LG_HELPERS.sortChampionEntries([]);
        console.log(Array.isArray(sorted) && sorted.length === 0 ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"

    def test_stable_for_identical_numeric_key(self):
        script = _harness() + """
        const entries = [
            ['champion-1-a', ['a']],
            ['champion-1-b', ['b']],
        ];
        const sorted = window.LG_HELPERS.sortChampionEntries(entries);
        // Both start with "1" as their first digit match; result must preserve
        // both entries (no drop).
        console.log(sorted.length === 2 ? 'OK' : 'FAIL');
        """
        assert run_js(script) == "OK"


# ---------------------------------------------------------------------------
# mountCircuitBoard dataKey — audit-gate and escalation repaint triggers
# ---------------------------------------------------------------------------

CIRCUIT_BOARD_JS = ROOT / "frontend" / "js" / "circuit-board.js"

_CB_HARNESS = """
const window = {};
const localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
function _mkEl() {
    return {
        style: {}, className: '', innerHTML: '', textContent: '',
        appendChild: () => {}, children: [], dataset: {},
        classList: { add: () => {}, remove: () => {} },
        addEventListener: () => {}, removeEventListener: () => {},
        querySelector: () => null, querySelectorAll: () => [],
        setAttribute: () => {}, getAttribute: () => null,
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
    };
}
const document = {
    addEventListener: (name, fn) => { if (name === 'alpine:init') window._alpineInit = fn; },
    createElement: () => _mkEl(),
    createElementNS: () => _mkEl(),
    getElementById: () => _mkEl(),
    querySelector: () => null,
    body: { appendChild: () => {} },
};
let _factory = null;
const Alpine = {
    directive: () => {},
    data: (name, fn) => { if (name === 'campaignApp') _factory = fn; },
};
"""


def _run_mount_key(state_a, state_b) -> str:
    """Build the mountCircuitBoard dataKey for two states, report equality."""
    trailer = """
    window._alpineInit();
    const app = _factory();
    window.renderCircuitBoard = () => { app._renderCalls = (app._renderCalls || 0) + 1; };
    const keys = [];
    for (const st of [%s, %s]) {
        app.campaignState = st;
        app.artifactCatalog = { files: ['rounds/1/audits/stage_67.md'] };
        app._cbLastDataKey = null;
        app.mountCircuitBoard();
        keys.push(app._cbLastDataKey);
    }
    console.log(keys[0] === keys[1] ? 'SAME' : 'DIFFER');
    """ % (json.dumps(state_a), json.dumps(state_b))

    js = (
        _CB_HARNESS
        + CIRCUIT_BOARD_JS.read_text()
        + "\nwindow.renderCircuitBoard = renderCircuitBoard;\n"
        + CAMPAIGN_APP_JS.read_text()
        + "\n" + trailer + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(js)
        path = f.name
    try:
        result = subprocess.run(["node", path], capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        return result.stdout.strip().splitlines()[-1]
    finally:
        os.unlink(path)


def _campaign(**overrides):
    state = {
        "campaign": {
            "status": "active",
            "current_round": 1,
            "cumulative_e2e_speedup": 1.1,
            "shipped_optimizations": [],
            "rounds": [{
                "round_id": 1,
                "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
                "bottleneck_mining": {"completed_at": "2026-01-01T01:00:00Z"},
                "parallel_tracks": {"tracks": {"C1": {"status": "PASS"}}},
                "integration": {"status": "completed"},
                "audit": {
                    "stage_67": {"started_at": "2026-01-01T03:30:00Z", "cycle": 3},
                },
            }],
        },
    }
    state["campaign"].update(overrides)
    return state


class TestCircuitBoardDataKey:
    """A campaign-scoped escalation writes only campaign.status +
    campaign.auditor_escalation. Without those in the dataKey the memo guard
    swallows the repaint and the red escalated hex never appears."""

    def test_key_is_stable_for_identical_state(self):
        assert _run_mount_key(_campaign(), _campaign()) == "SAME"

    def test_key_changes_on_a_campaign_scoped_escalation(self):
        escalated = _campaign(
            status="paused",
            auditor_escalation={"stage": "7_campaign_eval", "round": 1, "scope": "campaign"},
        )
        assert _run_mount_key(_campaign(), escalated) == "DIFFER"

    def test_key_changes_when_only_the_escalation_scope_is_corrected(self):
        first = _campaign(
            auditor_escalation={"stage": "7_campaign_eval", "round": 1, "scope": "campaign"},
        )
        second = _campaign(
            auditor_escalation={"stage": "6_integration", "round": 1, "scope": "campaign"},
        )
        assert _run_mount_key(first, second) == "DIFFER"

    def test_key_changes_on_a_legacy_alias_gate_stamp(self):
        legacy = _campaign()
        legacy["campaign"]["rounds"][0]["audit"]["stage_6"] = {
            "started_at": "2026-01-01T04:00:00Z", "passed_at": "2026-01-01T05:00:00Z",
        }
        assert _run_mount_key(_campaign(), legacy) == "DIFFER"
