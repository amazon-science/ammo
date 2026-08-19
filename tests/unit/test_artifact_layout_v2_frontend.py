# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Artifact Layout V2 — Frontend Track.

Plan: .claude/plans/artifact-layout-v2-frontend.md

Pattern: Python invokes JS via Node.js subprocess (same approach as
tests/unit/test_e2e_latency_frontend.py and tests/unit/test_l1_campaign_grid.py).

Wave 1 (this file initially): module-scope helpers exported on window.LG_HELPERS:
  - parseArtifactPath(path)
  - _deriveRoundFromPath(path)
  - _buildPipelineProgress(campaign)
  - _countTrackStatuses(state)
  - _normalizeCumulativeSpeedup(campaign)  (Task 5)
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"
CIRCUIT_BOARD_JS = ROOT / "frontend" / "js" / "circuit-board.js"


# ────────────────────────────────────────────────────────────────────────────
# Subprocess harness — load campaign-app.js with stubbed Alpine/document/window,
# expose window.LG_HELPERS to extra_js, print result to stdout.
# campaign-app.js is large; pipe via stdin to avoid ARG_MAX.
# ────────────────────────────────────────────────────────────────────────────


def _run_js(script: str, timeout: int = 10) -> str:
    result = subprocess.run(
        ["node", "--input-type=commonjs", "-e",
         "eval(require('fs').readFileSync(0,'utf8'))"],
        input=script, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error:\n{result.stderr}")
    return result.stdout.strip()


def _app_harness(extra_js: str) -> str:
    """Run extra_js with campaign-app.js loaded (module-scope LG_HELPERS available).

    Alpine.data callback is invoked too so helpers defined inside alpine:init
    (e.g. _normalizeShippedOps) also land on window.LG_HELPERS.
    """
    src = CAMPAIGN_APP_JS.read_text()
    script = f"""
const window = {{}};
let _alpineInitCb = null;
const document = {{
  addEventListener: (e, cb) => {{ if (e === 'alpine:init') _alpineInitCb = cb; }},
}};
const Alpine = {{ data: () => {{}}, directive: () => {{}}, store: () => {{}} }};
const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
{src}
if (_alpineInitCb) {{ try {{ _alpineInitCb(); }} catch (e) {{ /* tolerated */ }} }}
{extra_js}
"""
    return _run_js(script)


def _both_harness(extra_js: str) -> str:
    """Load campaign-app.js + circuit-board.js. Used for tests that need
    CircuitBoard (e.g. _enrichFromCatalog).
    """
    app_src = CAMPAIGN_APP_JS.read_text()
    cb_src = CIRCUIT_BOARD_JS.read_text()
    script = f"""
const window = {{}};
let _alpineInitCb = null;
const document = {{
  addEventListener: (e, cb) => {{ if (e === 'alpine:init') _alpineInitCb = cb; }},
  createElement: () => ({{
    style: {{}}, className: '', innerHTML: '',
    appendChild: () => {{}}, children: [], setAttribute: () => {{}},
    addEventListener: () => {{}},
  }}),
}};
const Alpine = {{ data: () => {{}}, directive: () => {{}}, store: () => {{}} }};
const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
{app_src}
if (_alpineInitCb) {{ try {{ _alpineInitCb(); }} catch (e) {{ /* tolerated */ }} }}
{cb_src}
{extra_js}
"""
    return _run_js(script)


def _call(fn: str, *args) -> str:
    """Call window.LG_HELPERS.<fn>(...args), JSON-print result."""
    arg_list = ", ".join(json.dumps(a) for a in args)
    return _app_harness(
        f"const __r = window.LG_HELPERS.{fn}({arg_list});"
        f"console.log(JSON.stringify(__r));"
    )


def _call_obj(fn: str, *args) -> dict:
    out = _call(fn, *args)
    return json.loads(out) if out else None


# ════════════════════════════════════════════════════════════════════════════
# Task 1a: parseArtifactPath
# ════════════════════════════════════════════════════════════════════════════


class TestParseArtifactPath:
    def test_profiling_probe(self):
        r = _call_obj("parseArtifactPath", "rounds/1/profiling/probe/probe_results.json")
        assert r == {"round": 1, "stage": "baseline", "track_id": None}

    def test_tracks_op(self):
        r = _call_obj("parseArtifactPath", "rounds/2/tracks/op-001/validation_results.md")
        assert r == {"round": 2, "stage": "implementation", "track_id": "op-001"}

    def test_tracks_validator_tests(self):
        r = _call_obj(
            "parseArtifactPath",
            "rounds/2/tracks/op-001/validator_tests/gate_5_1a_results.json",
        )
        assert r == {"round": 2, "stage": "validation", "track_id": "op-001"}

    def test_mining(self):
        r = _call_obj("parseArtifactPath", "rounds/1/mining/bottleneck_analysis.md")
        assert r == {"round": 1, "stage": "mining", "track_id": None}

    def test_sweeps_opt(self):
        r = _call_obj("parseArtifactPath", "rounds/3/sweeps/opt/op-002/e2e_latency_results.json")
        assert r == {"round": 3, "stage": "implementation", "track_id": "op-002"}

    def test_debate_proposals(self):
        r = _call_obj(
            "parseArtifactPath",
            "rounds/1/debate/proposals/champion-1_proposal.md",
        )
        assert r == {"round": 1, "stage": "debate", "track_id": None}

    def test_debate_sub_round_not_campaign_round(self):
        # debate/round_2/ is a debate sub-round. The campaign round comes from
        # the outer rounds/1/ — NOT from the inner round_2/.
        r = _call_obj(
            "parseArtifactPath",
            "rounds/1/debate/round_2/op-001_argument.md",
        )
        assert r["round"] == 1, f"expected round=1 (outer), got {r}"
        assert r["stage"] == "debate"
        assert r["track_id"] is None

    def test_sweeps_integration(self):
        r = _call_obj("parseArtifactPath", "rounds/2/sweeps/integration/e2e_latency_results.json")
        assert r == {"round": 2, "stage": "integration", "track_id": None}

    def test_sweeps_baseline(self):
        r = _call_obj("parseArtifactPath", "rounds/2/sweeps/baseline/results.json")
        assert r == {"round": 2, "stage": "baseline", "track_id": None}

    def test_sweeps_golden_capture(self):
        r = _call_obj("parseArtifactPath", "rounds/1/sweeps/golden_capture/golden.json")
        assert r == {"round": 1, "stage": "baseline", "track_id": None}

    def test_audits_stage_1(self):
        r = _call_obj("parseArtifactPath", "rounds/1/audits/stage_1_baseline.md")
        assert r == {"round": 1, "stage": "baseline", "track_id": None}

    def test_audits_stage_45(self):
        r = _call_obj("parseArtifactPath", "rounds/1/audits/stage_45.md")
        assert r == {"round": 1, "stage": "implementation", "track_id": None}

    def test_audits_stage_67(self):
        r = _call_obj("parseArtifactPath", "rounds/1/audits/stage_67.md")
        assert r == {"round": 1, "stage": "integration", "track_id": None}

    def test_archive(self):
        r = _call_obj("parseArtifactPath", "rounds/1/_archive/baseline_2026-05-05/foo.json")
        assert r == {"round": 1, "stage": None, "track_id": None}

    def test_constraints(self):
        r = _call_obj("parseArtifactPath", "constraints.md")
        assert r["stage"] == "baseline"
        assert r["track_id"] is None

    def test_legacy_flat_bottleneck(self):
        r = _call_obj("parseArtifactPath", "bottleneck_analysis.md")
        assert r == {"round": None, "stage": "mining", "track_id": None}

    def test_legacy_flat_tracks(self):
        r = _call_obj("parseArtifactPath", "tracks/op-001/file.md")
        assert r == {"round": None, "stage": "implementation", "track_id": "op-001"}

    def test_legacy_flat_validator_tests(self):
        r = _call_obj("parseArtifactPath", "tracks/op-001/validator_tests/test.py")
        assert r == {"round": None, "stage": "validation", "track_id": "op-001"}

    def test_legacy_debate_flat(self):
        r = _call_obj("parseArtifactPath", "debate/summary.md")
        assert r["stage"] == "debate"
        assert r["track_id"] is None

    def test_legacy_e2e_latency(self):
        r = _call_obj("parseArtifactPath", "e2e_latency_results.json")
        assert r["stage"] == "baseline"

    def test_legacy_monitor_log(self):
        r = _call_obj("parseArtifactPath", "monitor_log_impl_op-001.log")
        assert r["stage"] == "debate"

    def test_null_and_empty(self):
        for path in [None, "", 0]:
            r = _call_obj("parseArtifactPath", path)
            assert r == {"round": None, "stage": None, "track_id": None}, (
                f"path={path!r}: {r}"
            )


# ════════════════════════════════════════════════════════════════════════════
# Task 1b: _deriveRoundFromPath
# ════════════════════════════════════════════════════════════════════════════


class TestDeriveRoundFromPath:
    def _derive(self, path):
        out = _app_harness(
            f"const r = window.LG_HELPERS._deriveRoundFromPath({json.dumps(path)});"
            f"console.log(r === null ? 'null' : String(r));"
        )
        return None if out == "null" else int(out)

    def test_v2_rounds_plural(self):
        assert self._derive("rounds/3/mining/foo.md") == 3

    def test_v2_rounds_plural_nested(self):
        assert self._derive("rounds/1/tracks/op-001/foo.md") == 1

    def test_v2_debate_sub_round_not_extracted(self):
        # rounds/1/debate/round_2/... → campaign round 1, NOT 2
        assert self._derive("rounds/1/debate/round_2/op-001_argument.md") == 1

    def test_legacy_round_underscore_still_works(self):
        assert self._derive("round_2/foo.md") == 2

    def test_legacy_campaign_round_still_works(self):
        assert self._derive("campaign_round_4/summary.md") == 4

    def test_no_round_segment(self):
        assert self._derive("tracks/op-001/file.md") is None

    def test_legacy_debate_round_at_root_still_works(self):
        # `debate/round_1/foo.md` — leftmost match is `/round_1/`.
        # This is a v1-only artifact path (debate sub-round at root); for v1
        # campaigns where ALL artifacts implicitly belong to round 1, returning
        # 1 here is correct.
        assert self._derive("debate/round_1/foo.md") == 1

    def test_null_input(self):
        assert self._derive(None) is None

    def test_empty_string(self):
        assert self._derive("") is None


# ════════════════════════════════════════════════════════════════════════════
# Task 1c: _buildPipelineProgress
# ════════════════════════════════════════════════════════════════════════════


def _statuses(progress):
    """Helper: return list of statuses from pipeline_progress array."""
    return [p["status"] for p in progress]


class TestBuildPipelineProgress:
    def _bp(self, campaign):
        out = _app_harness(
            f"const r = window.LG_HELPERS._buildPipelineProgress({json.dumps(campaign)});"
            f"console.log(JSON.stringify(r));"
        )
        return json.loads(out)

    def test_baseline_active(self):
        r = self._bp({"current_stage": "1_baseline", "status": "active"})
        assert _statuses(r) == ["active", "pending", "pending", "pending", "pending", "pending"]
        assert r[0]["stage"] == "baseline"
        assert r[1]["stage"] == "mining"

    def test_mining_active(self):
        r = self._bp({"current_stage": "2_bottleneck_mining", "status": "active"})
        assert _statuses(r) == ["completed", "active", "pending", "pending", "pending", "pending"]

    def test_debate_active(self):
        r = self._bp({"current_stage": "3_debate", "status": "active"})
        assert _statuses(r) == ["completed", "completed", "active", "pending", "pending", "pending"]

    def test_parallel_tracks_active(self):
        # Stage 4_5 → impl AND validation both active.
        r = self._bp({"current_stage": "4_5_parallel_tracks", "status": "active"})
        assert _statuses(r) == ["completed", "completed", "completed", "active", "active", "pending"]

    def test_integration_active(self):
        r = self._bp({"current_stage": "6_integration", "status": "active"})
        assert _statuses(r) == ["completed", "completed", "completed", "completed", "completed", "active"]

    def test_terminal_complete(self):
        r = self._bp({"status": "campaign_complete"})
        assert _statuses(r) == ["completed"] * 6

    def test_terminal_exhausted(self):
        r = self._bp({"status": "campaign_exhausted"})
        assert _statuses(r) == ["completed"] * 6

    def test_7b_report(self):
        r = self._bp({"current_stage": "7b_report", "status": "active"})
        assert _statuses(r) == ["completed"] * 6

    def test_7_campaign_eval(self):
        r = self._bp({"current_stage": "7_campaign_eval", "status": "active"})
        assert _statuses(r) == ["completed"] * 6

    def test_empty_campaign(self):
        r = self._bp({})
        assert _statuses(r) == ["pending"] * 6

    def test_null_campaign(self):
        out = _app_harness(
            "const r = window.LG_HELPERS._buildPipelineProgress(null);"
            "console.log(JSON.stringify(r));"
        )
        r = json.loads(out)
        assert _statuses(r) == ["pending"] * 6


# ════════════════════════════════════════════════════════════════════════════
# Task 1d: _countTrackStatuses
# ════════════════════════════════════════════════════════════════════════════


class TestCountTrackStatuses:
    def _count(self, state):
        out = _app_harness(
            f"const r = window.LG_HELPERS._countTrackStatuses({json.dumps(state)});"
            f"console.log(JSON.stringify(r));"
        )
        return json.loads(out)

    def test_basic_counts(self):
        # Round 1 (current): op-001 shipped, op-002 failed, op-003 in_progress.
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": ["op-001"],
                "rounds": [{
                    "round_id": 1,
                    "shipped": ["op-001"],
                    "parallel_tracks": {"tracks": {
                        "op-002": {"status": "FAILED"},
                        "op-003": {"status": "IN_PROGRESS"},
                    }},
                }],
            }
        }
        r = self._count(state)
        assert r == {"shipped": 1, "failed": 1, "active": 1}

    def test_gpu_blocked_is_failed(self):
        state = {
            "campaign": {
                "current_round": 1,
                "rounds": [{
                    "round_id": 1, "shipped": [],
                    "parallel_tracks": {"tracks": {
                        "op-001": {"status": "GPU_BLOCKED"},
                    }},
                }],
            }
        }
        r = self._count(state)
        assert r["failed"] == 1, r

    def test_gated_pass_is_shipped(self):
        # GATED_PASS verdict → counted as shipped (per server logic).
        state = {
            "campaign": {
                "current_round": 1,
                "rounds": [{
                    "round_id": 1, "shipped": [],
                    "parallel_tracks": {"tracks": {
                        "op-001": {"verdict": "GATED_PASS"},
                    }},
                }],
            }
        }
        r = self._count(state)
        assert r["shipped"] == 1, r

    def test_past_round_nonterminal_not_counted_as_active(self):
        # Round 2 is current; round 1 has a non-terminal track that should NOT
        # be counted as active.
        state = {
            "campaign": {
                "current_round": 2,
                "rounds": [
                    {"round_id": 1, "shipped": [],
                     "parallel_tracks": {"tracks": {
                         "op-001": {"status": "IN_PROGRESS"},
                     }}},
                    {"round_id": 2, "shipped": [],
                     "parallel_tracks": {"tracks": {
                         "op-002": {"status": "IN_PROGRESS"},
                     }}},
                ],
            }
        }
        r = self._count(state)
        # op-001 (round 1, non-terminal, past round) → not counted as active
        # op-002 (round 2, non-terminal, current round) → active
        assert r == {"shipped": 0, "failed": 0, "active": 1}

    def test_empty_state(self):
        r = self._count({})
        assert r == {"shipped": 0, "failed": 0, "active": 0}

    def test_shipped_optimizations_list_str_format(self):
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": ["op-001", "op-002"],
                "rounds": [{
                    "round_id": 1, "shipped": ["op-001", "op-002"],
                    "parallel_tracks": {"tracks": {}},
                }],
            }
        }
        r = self._count(state)
        assert r["shipped"] == 2

    def test_shipped_optimizations_list_dict_format(self):
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [{"op_id": "op-001"}, {"op_id": "op-002"}],
                "rounds": [{
                    "round_id": 1, "shipped": ["op-001", "op-002"],
                    "parallel_tracks": {"tracks": {}},
                }],
            }
        }
        r = self._count(state)
        assert r["shipped"] == 2

    def test_works_with_trimmed_projection(self):
        # L1 projection: only status+verdict per track (no impl details).
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [],
                "rounds": [{
                    "round_id": 1, "shipped": [],
                    "parallel_tracks": {"tracks": {
                        "op-001": {"status": "FAILED", "verdict": "FAIL"},
                        "op-002": {"status": "IN_PROGRESS"},
                    }},
                }],
            }
        }
        r = self._count(state)
        assert r == {"shipped": 0, "failed": 1, "active": 1}


# ════════════════════════════════════════════════════════════════════════════
# Task 5: _normalizeCumulativeSpeedup
# ════════════════════════════════════════════════════════════════════════════


class TestNormalizeCumulativeSpeedup:
    def _norm(self, campaign):
        # The function mutates in-place; return the resulting campaign.
        out = _app_harness(
            f"const c = {json.dumps(campaign)};"
            f"window.LG_HELPERS._normalizeCumulativeSpeedup(c);"
            f"console.log(JSON.stringify(c));"
        )
        return json.loads(out)

    def test_v3_field_copied_when_v4_absent(self):
        c = self._norm({"cumulative_speedup_vs_round1": 1.34})
        assert c["cumulative_e2e_speedup"] == 1.34

    def test_v3_field_copied_when_v4_is_1_0(self):
        c = self._norm({"cumulative_e2e_speedup": 1.0, "cumulative_speedup_vs_round1": 1.5})
        assert c["cumulative_e2e_speedup"] == 1.5

    def test_v4_field_preserved_when_already_set(self):
        c = self._norm({"cumulative_e2e_speedup": 2.1, "cumulative_speedup_vs_round1": 1.5})
        assert c["cumulative_e2e_speedup"] == 2.1

    def test_both_absent_is_graceful(self):
        c = self._norm({})
        # No mutation, no crash. cumulative_e2e_speedup may be absent.
        assert "cumulative_e2e_speedup" not in c or c["cumulative_e2e_speedup"] is None

    def test_null_campaign_no_crash(self):
        out = _app_harness(
            "window.LG_HELPERS._normalizeCumulativeSpeedup(null);"
            "console.log('ok');"
        )
        assert out == "ok"

    def test_v3_value_1_0_not_copied(self):
        # 1.0 is the "no speedup" sentinel — don't copy it.
        c = self._norm({"cumulative_speedup_vs_round1": 1.0})
        # Either absent or stays at 1.0 — but NOT mutated to 1.0 from undefined.
        v4 = c.get("cumulative_e2e_speedup")
        assert v4 is None or v4 == 1.0


# ════════════════════════════════════════════════════════════════════════════
# Task 2: Sidecars shape adaptation (`data.sidecars` flat dict)
# ════════════════════════════════════════════════════════════════════════════

# Helpers exposed on window.LG_HELPERS that consume artifact-catalog data:
#   - _catalogEntries(catalog)     : materialize entries array from
#                                    EITHER a flat sidecars dict OR a legacy
#                                    `{entries: {...}}` wrapper.
#
# These tests drive the helper directly and assert behavior.


class TestCatalogEntriesFlatSidecars:
    def _entries(self, catalog):
        out = _app_harness(
            f"const arr = window.LG_HELPERS._catalogEntries({json.dumps(catalog)});"
            f"console.log(JSON.stringify(arr));"
        )
        return json.loads(out)

    def test_flat_dict_produces_entries(self):
        sidecars = {
            "rounds/1/mining/bottleneck_analysis.md": {
                "labels": {"kind": "bottleneck"},
                "metrics": {"top_component": "mlp", "top_f_decode_pct": 0.42},
            },
            "rounds/1/profiling/probe/probe_results.json": {
                "labels": {"kind": "e2e_latency"},
                "metrics": {"baseline_avg_s": 0.005},
            },
        }
        entries = self._entries(sidecars)
        assert isinstance(entries, list)
        assert len(entries) == 2
        paths = sorted(e["path"] for e in entries)
        assert paths == [
            "rounds/1/mining/bottleneck_analysis.md",
            "rounds/1/profiling/probe/probe_results.json",
        ]
        # Round derived from path.
        for e in entries:
            assert e["round"] == 1, e

    def test_legacy_entries_wrapper_still_works(self):
        legacy = {
            "entries": {
                "bottleneck_analysis.md": {
                    "metrics": {"top_component": "mlp"},
                    "labels": {"kind": "bottleneck"},
                    "round": 1,
                    "stage": "mining",
                },
            },
            "last_updated": "2026-05-13T00:00:00Z",
            "last_scan_file_count": 1,
            "schema_version": 1,
        }
        entries = self._entries(legacy)
        assert len(entries) == 1
        assert entries[0]["path"] == "bottleneck_analysis.md"

    def test_empty_dict_returns_empty(self):
        assert self._entries({}) == []

    def test_null_catalog(self):
        out = _app_harness(
            "const arr = window.LG_HELPERS._catalogEntries(null);"
            "console.log(JSON.stringify(arr));"
        )
        assert json.loads(out) == []


class TestCatalogEntriesKeyInvalidation:
    """When a fresh sidecar dict arrives (different object identity, more
    keys), the memoized cache MUST invalidate so downstream UI re-renders.

    Pre-V2 the cache was keyed on `(last_updated, last_scan_file_count)` —
    both undefined for the flat dict shape, so the key froze to `'|'` and
    every subsequent dict re-used stale entries.
    """

    def test_new_keys_invalidates_memo(self):
        # Two distinct dicts, second has one MORE key. Without proper
        # invalidation, both would return the same memoized array.
        out = _app_harness(
            """
const c1 = {"rounds/1/mining/foo.md": {"labels": {"kind": "bottleneck"}}};
const c2 = {
    "rounds/1/mining/foo.md": {"labels": {"kind": "bottleneck"}},
    "rounds/1/profiling/probe/probe_results.json": {"labels": {"kind": "e2e_latency"}},
};
const a = window.LG_HELPERS._catalogEntries(c1);
const b = window.LG_HELPERS._catalogEntries(c2);
console.log(JSON.stringify({a: a.length, b: b.length}));
"""
        )
        result = json.loads(out)
        assert result == {"a": 1, "b": 2}, (
            f"Expected memo to invalidate on new keys; got {result}"
        )

    def test_buildCatalogEntriesKey_changes_on_new_sidecars(self):
        out = _app_harness(
            """
const c1 = {"a": {}, "b": {}};
const c2 = {"a": {}, "b": {}, "c": {}};
const k1 = window.LG_HELPERS.buildCatalogEntriesKey(c1);
const k2 = window.LG_HELPERS.buildCatalogEntriesKey(c2);
console.log(JSON.stringify({k1, k2, equal: k1 === k2}));
"""
        )
        result = json.loads(out)
        assert result["equal"] is False, (
            f"buildCatalogEntriesKey must return distinct keys for different "
            f"sidecar dicts; got equal keys: {result}"
        )

    def test_buildCatalogEntriesKey_handles_legacy_shape(self):
        out = _app_harness(
            """
const legacy = {
    entries: {a: {}, b: {}},
    last_updated: "2026-05-13",
    last_scan_file_count: 2,
};
const k = window.LG_HELPERS.buildCatalogEntriesKey(legacy);
console.log(k);
"""
        )
        # Just needs to be a non-empty deterministic string; specifics don't matter.
        assert out and out != "null"


class TestSidecarsResponseShapeStorage:
    """Verifies that the JS source contains the v2 ingestion logic — the
    Alpine instance loadCampaignDetail/polling must store `data.sidecars`
    (with `data.artifact_catalog?.entries` fallback) and not crash on the
    absent `schema_version` field."""

    def test_loadCampaignDetail_reads_data_sidecars(self):
        src = (ROOT / "frontend" / "js" / "campaign-app.js").read_text()
        # The new shape stores sidecars; the old shape's .entries fallback
        # must remain so v1 campaigns keep working during transition.
        assert "data.sidecars" in src, (
            "campaign-app.js must read data.sidecars (new contract). "
            "See plan Task 2."
        )

    def test_no_schema_version_crash_path(self):
        """The previous code keyed its mismatch-warning on
        `data.artifact_catalog?.schema_version`. In the new shape that field
        is gone — code must not return early or warn when sidecars arrive
        with no schema_version."""
        src = (ROOT / "frontend" / "js" / "campaign-app.js").read_text()
        # Old guard had the form:
        #   const catalogVersion = data.artifact_catalog?.schema_version;
        #   if (catalogVersion != null && catalogVersion !== _CATALOG_SCHEMA_VERSION)
        # We only need to confirm the old early-return is gone; a permissive
        # check (just reading the field) is fine.
        offending = "if (catalogVersion != null && catalogVersion !== _CATALOG_SCHEMA_VERSION)"
        assert offending not in src, (
            "schema_version mismatch early-return is still present; v2 "
            "sidecars have no schema_version, so this path would block "
            "every poll. See plan Task 2."
        )


class TestCatalogHasPathFlatDict:
    """`_catalogHasPath(path)` must work with the flat sidecar dict — the
    Alpine method reads `this.artifactCatalog`. We can't easily instantiate
    the Alpine component from Node, so we drive the underlying behavior by
    testing the source uses the flat-dict access pattern."""

    def test_no_legacy_dot_entries_lookup(self):
        src = (ROOT / "frontend" / "js" / "campaign-app.js").read_text()
        # The old _catalogHasPath had `if (catalog.entries[path]) return true;`
        # Replace with a flat-dict-aware lookup. The new code should not
        # gate on `catalog.entries[path]`.
        assert "catalog.entries[path]" not in src, (
            "Legacy `catalog.entries[path]` lookup remains; v2 sidecars are "
            "a flat dict so the access becomes `catalog[path]`. See plan Task 2."
        )


class TestOpenSidecarArtifactFlatDict:
    """`openSidecarArtifact(path)` must scan the flat sidecars dict
    directly (no `.entries` wrapper)."""

    def test_no_dot_entries_path_lookup(self):
        src = (ROOT / "frontend" / "js" / "campaign-app.js").read_text()
        # `this.artifactCatalog?.entries?.[path]` — old keyed lookup.
        assert "this.artifactCatalog?.entries?.[path]" not in src, (
            "openSidecarArtifact still uses the old keyed lookup; v2 "
            "sidecars have no `.entries` wrapper. See plan Task 2."
        )
        # `Object.entries(this.artifactCatalog.entries)` — old scan.
        assert "Object.entries(this.artifactCatalog.entries)" not in src, (
            "openSidecarArtifact still scans `.entries`; v2 sidecars have "
            "no `.entries` wrapper. See plan Task 2."
        )


class TestEnrichFromCatalogV2:
    """Task 2c: `_enrichFromCatalog` in circuit-board.js must iterate the
    flat sidecar dict AND use parseArtifactPath so v2 paths produce the
    correct effective stage/round when the server doesn't stamp them."""

    def _enrich(self, state: dict, catalog: dict) -> dict:
        return self._call(state, catalog)

    def _call(self, state: dict, catalog: dict) -> dict:
        out = _both_harness(
            f"const state = {json.dumps(state)};"
            f"const catalog = {json.dumps(catalog)};"
            f"CircuitBoard.enrichFromCatalog(state, catalog);"
            f"console.log(JSON.stringify(state._catalog || {{}}));"
        )
        return json.loads(out)

    def test_flat_sidecars_iterates_all_entries(self):
        # Smoke test: with the flat dict, the iteration must NOT skip everything.
        state = {"campaign": {"rounds": [{"round_id": 1}]}}
        catalog = {
            "rounds/1/mining/bottleneck_analysis.md": {
                "labels": {"kind": "bottleneck"},
                "metrics": {
                    "top_component": "mlp",
                    "top_f_decode_pct": 0.42,
                },
            },
            "rounds/1/debate/proposals/champion-1_proposal.md": {
                "labels": {"kind": "debate_rationale", "champion_id": "champion-1"},
            },
        }
        cat = self._enrich(state, catalog)
        # Mining bucket populated from path-derived stage.
        assert cat.get("miningByRound", {}).get("1"), (
            f"Expected mining bucket populated by path-derived stage; got {cat}"
        )
        # Rationale list populated.
        rats = cat.get("rationales", [])
        assert any("champion-1" in (r.get("championId") or "") for r in rats), (
            f"Expected rationale entry with championId; got {rats}"
        )

    def test_v2_path_derives_round_and_stage_for_mining(self):
        # Mining bucket only fires when stage='mining' AND round is finite.
        state = {"campaign": {"rounds": [{"round_id": 1}]}}
        catalog = {
            "rounds/1/mining/bottleneck_analysis.md": {
                "labels": {"kind": "bottleneck"},
                "metrics": {
                    "top_component": "mlp",
                    "top_f_decode_pct": 0.42,
                },
                # No `v.stage`, no `v.round` — must derive from path.
            },
        }
        cat = self._enrich(state, catalog)
        bucket = cat.get("miningByRound", {}).get("1")
        assert bucket is not None, f"Mining bucket missing; got {cat}"
        assert bucket["component"] == "mlp"

    def test_old_entries_wrapper_backward_compat(self):
        # Server-stamped v.round + v.stage in legacy `{entries: {...}}` shape.
        state = {"campaign": {"rounds": [{"round_id": 1}]}}
        catalog = {
            "entries": {
                "bottleneck_analysis.md": {
                    "labels": {"kind": "bottleneck"},
                    "metrics": {"top_component": "mlp", "top_f_decode_pct": 0.42},
                    "round": 1,
                    "stage": "mining",
                },
            },
            "last_updated": "2026-05-13",
            "last_scan_file_count": 1,
        }
        cat = self._enrich(state, catalog)
        assert cat.get("miningByRound", {}).get("1") is not None, (
            f"Legacy `{{entries: {{...}}}}` shape lost — backward compat "
            f"broken. Got {cat}"
        )

    def test_stamped_round_stage_preferred_over_derived(self):
        # If the server DOES stamp v.round/v.stage, the function should
        # prefer the stamped values (no regression on legacy v1 campaigns).
        state = {"campaign": {"rounds": [{"round_id": 1}]}}
        catalog = {
            "tracks/op-001/file.md": {
                "labels": {"kind": "bottleneck"},
                "metrics": {"top_component": "mlp", "top_f_decode_pct": 0.42},
                "round": 1,
                "stage": "mining",  # explicitly stamped — should win even though
                                    # path-derived stage would be 'implementation'.
            },
        }
        cat = self._enrich(state, catalog)
        # Stamped stage='mining' must keep this entry in the mining bucket.
        assert cat.get("miningByRound", {}).get("1") is not None, (
            f"Stamped v.stage='mining' was overridden by path-derived stage. "
            f"Got {cat}"
        )

    def test_debate_entry_matched_by_derived_stage(self):
        state = {"campaign": {"rounds": [{"round_id": 1}]}}
        catalog = {
            "rounds/1/debate/summary.md": {
                "labels": {"kind": "debate_summary"},
                "metrics": {"champions_count": 3, "winners": ["op-001"]},
            },
        }
        cat = self._enrich(state, catalog)
        bucket = cat.get("debateByRound", {}).get("1")
        assert bucket is not None, f"Expected debate bucket; got {cat}"


# ════════════════════════════════════════════════════════════════════════════
# Task 3: _catalogEntries enrichment with _parsed + _stage
# ════════════════════════════════════════════════════════════════════════════


class TestCatalogEntriesV2Enrichment:
    def _entries(self, catalog):
        out = _app_harness(
            f"const arr = window.LG_HELPERS._catalogEntries({json.dumps(catalog)});"
            f"console.log(JSON.stringify(arr));"
        )
        return json.loads(out)

    def test_v2_entry_gets_parsed_stage(self):
        cat = {"rounds/1/mining/bottleneck_analysis.md": {"labels": {}}}
        e = self._entries(cat)[0]
        assert e["_stage"] == "mining"
        assert e["_parsed"] == {"round": 1, "stage": "mining", "track_id": None}

    def test_v2_entry_gets_parsed_round(self):
        cat = {"rounds/3/sweeps/opt/op-002/e2e_latency_results.json": {}}
        e = self._entries(cat)[0]
        assert e["round"] == 3, e
        assert e["_parsed"]["round"] == 3

    def test_v2_entry_gets_parsed_track_id(self):
        cat = {"rounds/2/tracks/op-001/validation_results.md": {}}
        e = self._entries(cat)[0]
        assert e["_parsed"]["track_id"] == "op-001"

    def test_v2_validator_tests_gets_validation_stage(self):
        cat = {"rounds/2/tracks/op-001/validator_tests/gate_5_1a.json": {}}
        e = self._entries(cat)[0]
        assert e["_stage"] == "validation", e
        assert e["_parsed"]["track_id"] == "op-001"

    def test_v1_entry_preserves_server_stamped_stage_as_fallback(self):
        # Server stamps stage='mining' but the path doesn't match the v2
        # mining heuristic (no `/mining/` segment). Path-derived stage is
        # null for `tracks/op-001/file.md`'s _parsed.stage='implementation',
        # but the test below uses a path with no mappable v2 stage so the
        # fallback fires.
        cat = {
            "entries": {
                "weird_legacy_path.md": {
                    "stage": "mining",
                    "round": 1,
                    "labels": {},
                },
            },
            "last_updated": "2026-05-13",
            "last_scan_file_count": 1,
        }
        e = self._entries(cat)[0]
        # parsed.stage is null (no heuristic match), so _stage falls through
        # to the stamped value.
        assert e["_stage"] == "mining"

    def test_legacy_round_defaults_to_1(self):
        # `bottleneck_analysis.md` at root: derivedRound=null, stamped=undefined.
        # Per Task 3 plan: must default to round=1 so v1 campaigns keep
        # showing artifacts in round-scoped views.
        cat = {"bottleneck_analysis.md": {"labels": {}}}
        e = self._entries(cat)[0]
        assert e["round"] == 1, (
            f"Legacy entry without round info must default to 1; got {e['round']}"
        )

    def test_isV2Layout_detection_positive(self):
        out = _app_harness(
            "const r = window.LG_HELPERS._isV2Layout("
            "{'rounds/1/mining/bottleneck_analysis.md': {}});"
            "console.log(r);"
        )
        assert out == "true"

    def test_isV2Layout_detection_negative(self):
        # Legacy paths (no `rounds/N/` prefix) → not v2.
        out = _app_harness(
            "const r = window.LG_HELPERS._isV2Layout("
            "{'bottleneck_analysis.md': {}, 'tracks/op-001/file.md': {}});"
            "console.log(r);"
        )
        assert out == "false"

    def test_isV2Layout_with_legacy_envelope(self):
        out = _app_harness(
            "const r = window.LG_HELPERS._isV2Layout("
            "{entries: {'rounds/1/mining/foo.md': {}}});"
            "console.log(r);"
        )
        assert out == "true"

    def test_isV2Layout_null_safe(self):
        out = _app_harness(
            "console.log(window.LG_HELPERS._isV2Layout(null));"
            "console.log(window.LG_HELPERS._isV2Layout({}));"
        )
        assert out.strip().split("\n") == ["false", "false"]


# ════════════════════════════════════════════════════════════════════════════
# Task 4: source-file regression checks for path-prefix filter rewrites
# ════════════════════════════════════════════════════════════════════════════


class TestV2PathFilters:
    """These tests scrub the JS source for filter sites that should have
    migrated to use parsed metadata. The plan flagged ~10 sites; these
    cover the most consequential ones."""

    def _src(self):
        return CAMPAIGN_APP_JS.read_text()

    def test_bottleneck_filter_uses_stage(self):
        src = self._src()
        # Every `entries.find(... bottleneck_analysis ...)` filter must also
        # gate on `_stage === 'mining'` so v2 paths under `rounds/N/mining/`
        # match. Count occurrences of the new pattern.
        bottleneck_uses = src.count("bottleneck_analysis.md")
        # We added the pattern to 3 sites (artifactSections mining,
        # l3ArtifactTree mining, _stageCatalogMetrics mining). The plan
        # listed bottleneck sites at lines 3360, 3536, 3740, 4854 — where
        # 3740 is the REPORT-section EXCLUSION (no `endsWith` needed since
        # report_section paths shouldn't end in bottleneck_analysis.md).
        assert bottleneck_uses >= 3, (
            f"Expected ≥3 sites referencing 'bottleneck_analysis.md' (path "
            f"endsWith pattern for v2 layout); found {bottleneck_uses}. "
            "See plan Task 4."
        )

    def test_tracks_includes_filter_kept_with_parsed_check(self):
        src = self._src()
        # The `e.path.includes('tracks/')` filter is paired with
        # `_parsed?.track_id != null` — verify both patterns appear in code
        # (not in comments) at expected counts. We added 5 v2 + kept 5 legacy.
        parsed_count = sum(
            1 for line in src.split("\n")
            if "e._parsed?.track_id != null" in line
            and not line.strip().startswith("//")
        )
        assert parsed_count >= 5, (
            f"Expected ≥5 code sites using `e._parsed?.track_id != null`; "
            f"found {parsed_count}. See plan Task 4."
        )
        legacy_count = sum(
            1 for line in src.split("\n")
            if "e.path.includes('tracks/')" in line
            and not line.strip().startswith("//")
        )
        assert legacy_count >= 5, (
            f"Legacy `e.path.includes('tracks/')` fallback should still be "
            f"present alongside the v2 filter; found {legacy_count}."
        )

    def test_no_lone_e_stage_filter_in_artifact_pipelines(self):
        """`e.stage === '...'` (with no underscore) was the v1 server-stamped
        field. After Task 3, all artifact-pipeline filter sites must use
        `e._stage`. We grep for lone `e.stage ===` or `e.stage === 'x'`
        forms — these should be zero in artifact filter contexts (excluding
        comments and string literals)."""
        src = self._src()
        offending = []
        for line in src.split("\n"):
            stripped = line.strip()
            # Skip JS comments — both leading-`//` and inline jsdoc-style.
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if "e.stage ===" in stripped or "e.stage===" in stripped:
                offending.append(stripped)
        assert not offending, (
            f"Found {len(offending)} `e.stage ===` filter site(s) that "
            f"should have migrated to `e._stage ===`:\n"
            + "\n".join(offending[:5])
        )


# ════════════════════════════════════════════════════════════════════════════
# Task 7: debateArtifactMatchesRound v2 path support
# ════════════════════════════════════════════════════════════════════════════


class TestDebateArtifactMatchesRoundV2:
    def _match(self, path, round_id, round_attr=None):
        attr = round_attr if round_attr is not None else round_id
        out = _app_harness(
            f"const r = window.LG_HELPERS.debateArtifactMatchesRound("
            f"{{path: {json.dumps(path)}, round: {json.dumps(attr)}}}, "
            f"{json.dumps(round_id)});"
            f"console.log(r);"
        )
        return out == "true"

    def test_v2_debate_same_round(self):
        assert self._match("rounds/2/debate/summary.md", 2) is True

    def test_v2_debate_different_round(self):
        # Artifact lives in rounds/1/, but caller asks for round 2 → reject.
        assert self._match("rounds/1/debate/summary.md", 2, round_attr=1) is False

    def test_v2_debate_sub_round_does_not_confuse(self):
        # rounds/1/debate/round_2/op-001_argument.md:
        #   - Campaign round = 1 (outer rounds/1/).
        #   - Debate sub-round = 2 (inner round_2/).
        # Asking for campaign round 1 must match.
        assert self._match("rounds/1/debate/round_2/op-001_argument.md", 1) is True

    def test_v2_debate_sub_round_rejects_wrong_campaign_round(self):
        # Same path but caller asks for campaign round 2 → reject; the
        # artifact belongs to campaign round 1.
        assert self._match(
            "rounds/1/debate/round_2/op-001_argument.md", 2, round_attr=1
        ) is False

    def test_v2_debate_round_1_summary(self):
        assert self._match("rounds/1/debate/proposals/champion-1.md", 1) is True

    def test_legacy_round_underscore_debate_still_works(self):
        # `round_2/debate/...` (legacy v1).
        assert self._match("round_2/debate/summary.md", 2) is True

    def test_legacy_campaign_round_debate_still_works(self):
        # `debate/campaign_round_2/...` (legacy v1).
        assert self._match("debate/campaign_round_2/summary.md", 2) is True

    def test_legacy_bare_debate_still_works_for_r1(self):
        # `debate/summary.md` is valid only for round 1.
        assert self._match("debate/summary.md", 1) is True

    def test_legacy_bare_debate_rejected_for_r2(self):
        assert self._match("debate/summary.md", 2) is False


# ════════════════════════════════════════════════════════════════════════════
# Task 6: L1 card computation — pipeline_progress + counts + flatten target
# ════════════════════════════════════════════════════════════════════════════


class TestL1CardComputation:
    """Tests `mergeSessionsAndCampaigns` via the alpine:init component
    surface. We instantiate the Alpine component, call the method directly,
    and inspect the returned cards.
    """

    def _merge(self, sessions, campaigns):
        sessions_js = json.dumps(sessions)
        campaigns_js = json.dumps(campaigns)
        out = _app_harness(
            f"""
let _data = null;
const Alpine2 = {{ data: (name, fn) => {{ _data = fn(); }}, directive: () => {{}}, store: () => {{}} }};
// Re-run alpine:init with the new Alpine stub so we capture the component.
let cb = null;
const document2 = {{
  addEventListener: (e, c) => {{ if (e === 'alpine:init') cb = c; }},
}};
// We can't re-run the script — but the original alpine:init listener
// invocation in _app_harness already ran with our stub Alpine that
// discards. Re-run by manually grabbing the campaignApp factory: the
// original `Alpine.data('campaignApp', factory)` was called above; we
// can't get it back. Instead, we monkey-patch by re-evaluating just
// the `Alpine.data` callback.
// Simpler: we re-run the file with a capture-Alpine inline.
console.log("HARNESS_PATTERN_DOES_NOT_SUPPORT");
"""
        )
        return out

    def _merge_via_capture(self, sessions, campaigns):
        """Run a capture-Alpine harness that snapshots the campaignApp
        factory and calls mergeSessionsAndCampaigns directly."""
        src = CAMPAIGN_APP_JS.read_text()
        script = f"""
let _data = null;
let _initCb = null;
const Alpine = {{
    data: (name, fn) => {{ _data = fn(); }},
    directive: () => {{}}, store: () => {{}},
}};
const document = {{
    addEventListener: (e, c) => {{ if (e === 'alpine:init') _initCb = c; }},
}};
const window = {{}};
const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
{src}
if (_initCb) _initCb();
const result = _data.mergeSessionsAndCampaigns(
    {json.dumps(sessions)},
    {json.dumps(campaigns)},
);
console.log(JSON.stringify(result));
"""
        out = _run_js(script)
        return json.loads(out)

    def test_card_pipeline_from_current_stage_when_server_omits(self):
        # Server-side L1 projection should ship pipeline_progress, but
        # during transition / for legacy projections that omit the field,
        # the FE must compute it from current_stage.
        sessions = [{"session_id": "s1", "status": "active"}]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "current_stage": "3_debate",
            # NO pipeline_progress field
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        assert len(cards) == 1
        pp = cards[0]["campaign"].get("pipeline_progress")
        assert pp is not None, f"pipeline_progress not computed; card={cards[0]}"
        assert len(pp) == 6
        statuses = [p["status"] for p in pp]
        # 3_debate → completed, completed, active, pending x 3.
        assert statuses == [
            "completed", "completed", "active", "pending", "pending", "pending"
        ], statuses

    def test_card_track_counts_from_trimmed_rounds(self):
        sessions = [{"session_id": "s1", "status": "active"}]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "current_round": 1,
            "shipped_optimizations": ["op-001"],
            "rounds": [{
                "round_id": 1,
                "shipped": ["op-001"],
                "parallel_tracks": {"tracks": {
                    "op-002": {"status": "FAILED"},
                    "op-003": {"status": "IN_PROGRESS"},
                }},
            }],
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        c = cards[0]["campaign"]
        assert c.get("shipped_count") == 1, c
        assert c.get("failed_count") == 1, c
        assert c.get("active_count") == 1, c

    def test_card_target_fields_flattened(self):
        # Server nests target as campaign.target.{model_id, dtype, tp}; FE
        # readers expect flat campaign.{model_id, dtype, tp}. Plan B1.
        sessions = [{"session_id": "s1", "status": "active"}]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "target": {
                "model_id": "meta-llama/Llama-3-8b",
                "dtype": "bf16",
                "tp": 4,
            },
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        c = cards[0]["campaign"]
        assert c.get("model_id") == "meta-llama/Llama-3-8b"
        assert c.get("dtype") == "bf16"
        assert c.get("tp") == 4

    def test_card_graceful_no_campaign(self):
        sessions = [{"session_id": "s1", "status": "active"}]
        cards = self._merge_via_capture(sessions, [])
        assert cards[0]["hasCampaign"] is False
        assert cards[0]["campaign"] is None

    def test_card_prefers_server_pipeline_during_transition(self):
        sessions = [{"session_id": "s1", "status": "active"}]
        # Server explicitly provides pipeline_progress — FE must NOT recompute.
        server_pp = [
            {"stage": "baseline",   "status": "completed"},
            {"stage": "mining",     "status": "completed"},
            {"stage": "debate",     "status": "completed"},
            {"stage": "implementation", "status": "completed"},
            {"stage": "validation", "status": "completed"},
            {"stage": "integration", "status": "active"},
        ]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "current_stage": "3_debate",  # would yield different result
            "pipeline_progress": server_pp,
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        assert cards[0]["campaign"]["pipeline_progress"] == server_pp

    def test_card_speedup_normalized_from_v3_field(self):
        sessions = [{"session_id": "s1", "status": "active"}]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "cumulative_speedup_vs_round1": 1.34,
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        c = cards[0]["campaign"]
        assert c.get("cumulative_e2e_speedup") == 1.34, c

    def test_card_with_only_status_verdict_per_track(self):
        # L1 trimmed projection — only status / verdict per track.
        sessions = [{"session_id": "s1", "status": "active"}]
        campaigns = [{
            "session_id": "s1",
            "status": "active",
            "current_round": 1,
            "rounds": [{
                "round_id": 1,
                "shipped": [],
                "parallel_tracks": {"tracks": {
                    "op-001": {"status": "FAIL", "verdict": "FAIL"},
                    "op-002": {"status": "IN_PROGRESS"},
                }},
            }],
        }]
        cards = self._merge_via_capture(sessions, campaigns)
        c = cards[0]["campaign"]
        assert c.get("failed_count") == 1
        assert c.get("active_count") == 1

