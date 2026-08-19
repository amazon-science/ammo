# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright E2E tests — Artifact Layout V2 frontend migration (FE-T8).

Verifies the complete L1 → L2 → L3 navigation flow with v2-structured server
responses (`data.sidecars` flat dict + L1 projection with `pipeline_progress`
+ `shipped_count`/`failed_count`/`active_count`) and confirms the FE-side
fallbacks engage when the v2 server fields are absent (transition compat).

Acceptance criteria (from `.claude/plans/artifact-layout-v2-frontend.md` §Task 8):
  - L1 grid renders campaign card with correct pipeline dots (server- or FE-computed)
  - L1 card shows correct shipped / failed / active counts
  - L2 circuit board navigation works with v2 state + sidecars shape
  - L3 mining node loads artifact from `rounds/1/mining/bottleneck_analysis.md`
  - L3 debate scoped to round (not confused by debate sub-rounds)
  - L3 validation column shows `validator_tests/*` artifacts
  - Sidecar-shape API response correctly populates `artifactCatalog`
  - Zero console errors during full navigation flow
  - Backward compat: legacy `artifact_catalog.entries` envelope still works

Pattern: API route interception (no live server beyond /ui shell). The
fixture `mock_api_routes` from conftest.py handles `/sessions`,
`/health`, and `/api/changelog`; we layer overrides for `/api/campaigns`
and `/api/campaign-data/{id}` here.

Two Alpine roots co-exist on /ui (`sessionApp` and `campaignApp`); we walk
[x-data]._x_dataStack to find the campaignApp frame, mirroring the helper
in test_sidecar_guards.py.
"""

from __future__ import annotations

import json
import time

import pytest
from playwright.sync_api import Page

from tests.playwright.conftest import (
    MOCK_HEALTH_RESPONSE,
    MOCK_AUTH_API_KEY,
)


SID = "fixture-v2-aaa111"

# Walk Alpine [x-data] stacks to find the campaignApp frame (the frame that
# owns `campaignState`). sessionApp lives on <body>, campaignApp on a deeper
# element; both must be reachable to hydrate /ui properly.
FIND_APP_JS = r"""(() => {
  for (const el of document.querySelectorAll('[x-data]')) {
    for (const frame of (el._x_dataStack || [])) {
      if (frame && Object.prototype.hasOwnProperty.call(frame, 'campaignState')) return frame;
    }
  }
  return null;
})()"""


# ──────────────────────────────────────────────────────────────────────────
# V2 mock fixtures
# ──────────────────────────────────────────────────────────────────────────

def _v2_state() -> dict:
    """Campaign state with two rounds: R1 fully complete, R2 in mining."""
    return {
        "target": {
            "model_id": "deepseek-ai/DeepSeek-R1-0528",
            "hardware": "H200",
            "dtype": "fp8",
            "tp": 4,
        },
        "campaign": {
            "status": "running",
            "current_round": 2,
            "cumulative_e2e_speedup": 1.34,
            "rounds": [
                {
                    "round_id": 1,
                    "stage": "6_integration",
                    "baseline": {
                        "e2e_latency": {
                            "128": {"avg": 7.66, "p50": 7.55, "p90": 8.00},
                        },
                        "per_bs_verdict": None,
                    },
                    "parallel_tracks": {
                        "tracks": {
                            "op-001": {"status": "shipped"},
                            "op-002": {"status": "shipped"},
                            "op-003": {"status": "failed"},
                        },
                    },
                    "integration": {
                        "e2e_latency_combined": {
                            "128": {"avg": 6.42, "p50": 6.40},
                        },
                        "per_bs_verdict": {"128": "PASS"},
                    },
                },
                {
                    "round_id": 2,
                    "stage": "2_bottleneck_mining",
                    "baseline": {
                        "e2e_latency": {
                            "128": {"avg": 6.42, "p50": 6.40},
                        },
                        "per_bs_verdict": {"128": "PASS"},
                    },
                    "parallel_tracks": {"tracks": {}},
                    "integration": {},
                },
            ],
            "shipped_optimizations": ["op-001", "op-002"],
        },
        "parallel_tracks": {},
        "stage": "2_bottleneck_mining",
    }


def _v2_sidecars() -> dict:
    """Flat path → entry dict — v2 layout under `rounds/{N}/...`.

    Includes carve-outs from the plan:
      - `_archive/` is excluded by parseArtifactPath (we omit it here)
      - validator_tests under tracks/{op}/validator_tests/ → stage='validation'
      - debate sub-rounds (debate/round_{D}/) → still scoped to top-level round
    """
    base = {
        "schema_version": 1,
        "emitter": "ammo-test",
        "emitted_at": "2026-05-13T00:00:00Z",
    }
    return {
        # ── Round 1 ─────────────────────────────────────────────────────
        "rounds/1/profiling/e2e_latency.json": {
            **base,
            "labels": {"kind": "e2e_latency", "full_benchmark": "true"},
            "metrics": {"avg_s": 7.66},
        },
        "rounds/1/mining/bottleneck_analysis.md": {
            **base,
            "labels": {"kind": "bottleneck_analysis"},
            "description": "Round 1 bottleneck mining — top component: GEMM",
        },
        "rounds/1/debate/summary.md": {
            **base,
            "labels": {"kind": "debate_summary"},
            "description": "R1 debate winners: op-001 + op-002",
        },
        "rounds/1/debate/round_1/rationale_op-001.md": {
            **base,
            "labels": {"kind": "debate_rationale", "op_id": "op-001"},
        },
        "rounds/1/debate/round_2/rationale_op-002.md": {
            **base,
            "labels": {"kind": "debate_rationale", "op_id": "op-002"},
        },
        "rounds/1/tracks/op-001/implementation/opt.py": {
            **base,
            "labels": {"kind": "source_code", "language": "python"},
            "refs": [{"path": "rounds/1/tracks/op-001/implementation/opt.py"}],
            "description": "op-001 fused GEMM",
        },
        "rounds/1/tracks/op-001/validator_tests/test_correctness.py": {
            **base,
            "labels": {"kind": "validator_test", "language": "python"},
            "description": "op-001 correctness test",
        },
        "rounds/1/tracks/op-002/implementation/opt.py": {
            **base,
            "labels": {"kind": "source_code", "language": "python"},
        },
        "rounds/1/sweeps/integration/combined.json": {
            **base,
            "labels": {"kind": "integration_combined"},
        },
        # ── Round 2 ─────────────────────────────────────────────────────
        "rounds/2/profiling/e2e_latency.json": {
            **base,
            "labels": {"kind": "e2e_latency", "full_benchmark": "true"},
            "metrics": {"avg_s": 6.42},
        },
        "rounds/2/mining/bottleneck_analysis.md": {
            **base,
            "labels": {"kind": "bottleneck_analysis"},
            "description": "Round 2 mining (skipped — reuses R1 profile)",
        },
    }


def _legacy_artifact_catalog() -> dict:
    """V3 envelope shape — `{schema_version, entries: {...}}`."""
    base = {
        "schema_version": 1,
        "emitter": "ammo-test",
        "emitted_at": "2026-05-13T00:00:00Z",
    }
    return {
        "schema_version": 1,
        "last_updated": "2026-05-13T00:00:00Z",
        "entries": {
            "bottleneck_analysis.md": {
                **base,
                "labels": {"kind": "bottleneck_analysis"},
            },
            "tracks/op-001/implementation/opt.py": {
                **base,
                "labels": {"kind": "source_code"},
            },
        },
    }


def _legacy_state_single_round() -> dict:
    """Minimal state for backward-compat test."""
    return {
        "target": {"model_id": "deepseek-ai/DeepSeek-R1-0528", "hardware": "A100",
                   "dtype": "fp8", "tp": 2},
        "campaign": {
            "status": "running", "current_round": 1,
            "cumulative_e2e_speedup": 1.0,
            "rounds": [{"round_id": 1, "stage": "2_bottleneck_mining",
                        "baseline": {}, "integration": {}, "parallel_tracks": {}}],
            "shipped_optimizations": [],
        },
        "parallel_tracks": {},
        "stage": "2_bottleneck_mining",
    }


def _sessions_payload() -> dict:
    return {"sessions": [{
        "session_id": SID, "status": "active", "cli_tool": "claude",
        "repo_name": "vllm", "gpu_ids": [0, 1, 2, 3],
        "terminal_url": f"/sessions/{SID}/terminal/",
        "terminal_ws_url": f"/sessions/{SID}/terminal/ws",
        "created_at": "2026-05-12T00:00:00",
        "last_accessed": int(time.time()), "owner_id": "mock-owner",
        "model_name": "deepseek-ai/DeepSeek-R1-0528", "dtype": "fp8",
        "ammo_version": "1.9.0",
    }]}


def _campaigns_payload(
    *,
    with_pipeline_progress: bool,
    with_counts: bool,
    include_rounds_for_count_fallback: bool = False,
) -> dict:
    """L1 projection. `with_pipeline_progress=False` exercises FE fallback.

    `include_rounds_for_count_fallback=True` ships the full `rounds` array
    inside the L1 overview so `_countTrackStatuses` has something to compute
    from when `with_counts=False` (simulates a server in transition that
    forwards rounds but hasn't yet pre-aggregated the counts).
    """
    overview: dict = {
        "session_id": SID,
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "hardware": "H200", "dtype": "fp8", "tp": 4,
        "status": "running", "current_round": 2,
        "cumulative_e2e_speedup": 1.34,
        "current_stage": "2_bottleneck_mining",
        "created_at": "2026-05-12T00:00:00",
        # B1: nested target so FE hoist logic exercised too.
        "target": {"model_id": "deepseek-ai/DeepSeek-R1-0528",
                   "hardware": "H200", "dtype": "fp8", "tp": 4},
    }
    if include_rounds_for_count_fallback:
        # Mirror state.campaign.rounds so _countTrackStatuses can bucket.
        overview["rounds"] = _v2_state()["campaign"]["rounds"]
        overview["shipped_optimizations"] = (
            _v2_state()["campaign"]["shipped_optimizations"]
        )
    if with_pipeline_progress:
        # 6-stage list, R2 has reached mining (idx 1 active).
        overview["pipeline_progress"] = [
            {"stage": "Baseline",      "status": "completed"},
            {"stage": "Mining",        "status": "active"},
            {"stage": "Debate",        "status": "pending"},
            {"stage": "Implementation","status": "pending"},
            {"stage": "Validation",    "status": "pending"},
            {"stage": "Integration",   "status": "pending"},
        ]
    if with_counts:
        overview["shipped_count"] = 2
        overview["failed_count"] = 1
        overview["active_count"] = 0
    return {"campaigns": [overview]}


# ──────────────────────────────────────────────────────────────────────────
# Route installation
# ──────────────────────────────────────────────────────────────────────────

def _install_routes(
    page: Page,
    *,
    state: dict,
    sidecars: dict | None,
    legacy_catalog: dict | None = None,
    with_pipeline_progress: bool = True,
    with_counts: bool = True,
    include_rounds_for_count_fallback: bool = False,
):
    """Mock /sessions, /api/campaigns, /api/campaign-data/{id}, /health."""
    sessions_payload = _sessions_payload()
    campaigns = _campaigns_payload(
        with_pipeline_progress=with_pipeline_progress,
        with_counts=with_counts,
        include_rounds_for_count_fallback=include_rounds_for_count_fallback,
    )

    # New v2 server response — nest sidecars when provided. Fall back to
    # the legacy envelope when sidecars=None (back-compat scenario).
    if sidecars is not None:
        campaign_data = {"state": state, "sidecars": sidecars}
    else:
        campaign_data = {"state": state, "artifact_catalog": legacy_catalog}

    def handler(route, request):
        url, m = request.url, request.method

        def ok(body):
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(body))

        if url.endswith("/health") or url.endswith("/health/"):
            ok(MOCK_HEALTH_RESPONSE); return
        if "/api/changelog" in url:
            ok({"version": "1.9.0", "entries": []}); return
        if "/sessions" in url or (url.rstrip("/").endswith("/sessions") and m == "GET"):
            ok(sessions_payload); return
        if f"/api/campaign-data/{SID}" in url:
            ok(campaign_data); return
        if "/api/campaigns" in url:
            tail = url.split("/api/campaigns", 1)[1]
            if tail in ("", "/"):
                ok(campaigns); return
            # Legacy /api/campaigns/{id} fallback — return state directly.
            route.fulfill(status=404, content_type="application/json",
                          body=json.dumps({"detail": "Not Found"})); return
        route.continue_()

    for pat in ("**/api/**", "**/sessions", "**/sessions/**", "**/health"):
        page.route(pat, handler)


def _prime(page: Page, server_url: str):
    """Navigate first to set origin, then dismiss tours / welcome popup."""
    page.goto(f"{server_url}/ui", wait_until="commit")
    for k, v in [("ammo_ui_theme", "lightgrid"),
                 ("ammo_lg_tour_completed", "true"),
                 ("ammo_lg_l1_deep_completed", "true"),
                 ("ammo_lg_l2_tour_completed", "true"),
                 ("ammo_lg_l3_tour_completed", "true"),
                 ("ammo_tour_completed", "true")]:
        page.evaluate(f"localStorage.setItem('{k}', '{v}')")


def _attach_console_collectors(page: Page) -> tuple[list[str], list[str]]:
    msgs: list[str] = []
    errors: list[str] = []

    def _on_console(msg):
        if msg.type == "error":
            errors.append(msg.text)
        msgs.append(msg.text)

    page.on("console", _on_console)
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    return msgs, errors


def _filter_real_errors(errors: list[str]) -> list[str]:
    """Drop noisy CSS / network-mock chatter; keep real JS exceptions."""
    keep: list[str] = []
    for e in errors:
        # Browsers log a console.error for every 4xx/5xx fetch; we mock 404
        # for the /api/campaigns/{id} fallback intentionally. Filter those.
        low = e.lower()
        if "404" in low and "/api/campaigns/" in low:
            continue
        if "failed to load resource" in low and "404" in low:
            continue
        keep.append(e)
    return keep


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.playwright
class TestArtifactLayoutV2:
    """E2E coverage for the artifact-layout-v2 frontend migration."""

    # ---------------------------------------------------------------- L1

    def test_l1_pipeline_dots_from_server_projection(self, page: Page, server_url: str):
        """Server-supplied `pipeline_progress` flows directly into stage dots."""
        msgs, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns", wait_until="networkidle")
        page.wait_for_timeout(1000)

        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const card = (d.allCards || []).find(c => c.session_id === '%s');
            if (!card) return {found: false};
            const els = d.cardPipelineElements ? d.cardPipelineElements(card) : [];
            const stages = els.filter(e => e.type === 'stage')
                              .map(e => ({label: e.label, status: e.status}));
            return {
                found: true,
                hasCampaign: card.hasCampaign,
                pipelineLen: (card.campaign && card.campaign.pipeline_progress || []).length,
                stages,
            };
        }""" % (FIND_APP_JS, SID))

        assert snap and snap["found"], "campaign card missing from L1 grid"
        assert snap["hasCampaign"] is True
        assert snap["pipelineLen"] == 6, f"expected 6 stages, got {snap}"
        # Server projection: Baseline complete, Mining active, rest pending.
        assert snap["stages"][0]["status"] == "completed"
        assert snap["stages"][1]["status"] == "active"
        for s in snap["stages"][2:]:
            assert s["status"] == "pending", snap

        # Zero unfiltered console errors.
        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    def test_l1_pipeline_dots_fallback_when_missing(self, page: Page, server_url: str):
        """Server omits `pipeline_progress` → FE `_buildPipelineProgress` fills in."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=False, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns", wait_until="networkidle")
        page.wait_for_timeout(1000)

        snap = page.evaluate("""() => {
            const d = %s;
            const card = (d.allCards || []).find(c => c.session_id === '%s');
            if (!card || !card.campaign) return null;
            return {
                pipeline: card.campaign.pipeline_progress || [],
            };
        }""" % (FIND_APP_JS, SID))

        assert snap is not None
        # FE fallback must produce a 6-element list.
        assert len(snap["pipeline"]) == 6, snap
        # current_round=2 + stage=2_bottleneck_mining → mining is active.
        statuses = [s["status"] for s in snap["pipeline"]]
        assert "active" in statuses, statuses
        assert statuses[0] == "completed", "baseline must be done by round 2"

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    def test_l1_track_counts_rendered(self, page: Page, server_url: str):
        """Server-supplied counts populate the L1 stat row dots."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns", wait_until="networkidle")
        page.wait_for_timeout(1000)

        snap = page.evaluate("""() => {
            const d = %s;
            const card = (d.allCards || []).find(c => c.session_id === '%s');
            if (!card || !card.campaign) return null;
            return {
                shipped: card.campaign.shipped_count,
                failed:  card.campaign.failed_count,
                active:  card.campaign.active_count,
            };
        }""" % (FIND_APP_JS, SID))

        assert snap == {"shipped": 2, "failed": 1, "active": 0}, snap

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    def test_l1_track_counts_fallback_from_state(self, page: Page, server_url: str):
        """When the L1 projection omits counts, `_countTrackStatuses` fills them in.

        Simulates a transitional server response that forwards `rounds` on
        the L1 overview but hasn't yet pre-aggregated `shipped_count` etc.
        """
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=False,
                        include_rounds_for_count_fallback=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns", wait_until="networkidle")
        page.wait_for_timeout(1000)

        snap = page.evaluate("""() => {
            const d = %s;
            const card = (d.allCards || []).find(c => c.session_id === '%s');
            if (!card || !card.campaign) return null;
            return {
                shipped: card.campaign.shipped_count,
                failed:  card.campaign.failed_count,
                active:  card.campaign.active_count,
            };
        }""" % (FIND_APP_JS, SID))

        assert snap is not None
        # _v2_state has 2 shipped + 1 failed in R1 parallel_tracks.tracks
        assert snap["shipped"] == 2, snap
        assert snap["failed"] == 1, snap

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    # ---------------------------------------------------------------- L2

    def test_l2_sidecars_shape_loads_catalog(self, page: Page, server_url: str):
        """Navigate to L2 → `data.sidecars` flows into `artifactCatalog`."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns/{SID}", wait_until="networkidle")
        page.wait_for_timeout(1500)

        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const cat = d.artifactCatalog;
            if (!cat) return {hasCatalog: false, level: d.currentLevel};
            // V2 sidecars is the FLAT dict — keys are paths directly. The
            // legacy envelope wraps them under .entries. Detect both.
            const isFlatDict = !cat.entries && typeof cat === 'object';
            const keys = Object.keys(isFlatDict ? cat : (cat.entries || {}));
            return {
                hasCatalog: true,
                level: d.currentLevel,
                isFlatDict,
                keysSample: keys.slice(0, 5).sort(),
                hasMiningR1: keys.includes('rounds/1/mining/bottleneck_analysis.md'),
                hasMiningR2: keys.includes('rounds/2/mining/bottleneck_analysis.md'),
            };
        }""" % FIND_APP_JS)

        assert snap and snap["hasCatalog"], f"catalog missing at L2: {snap}"
        assert snap["level"] == 2, snap
        assert snap["isFlatDict"] is True, "expected flat sidecars dict, not envelope"
        assert snap["hasMiningR1"], snap
        assert snap["hasMiningR2"], snap

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    # ---------------------------------------------------------------- L3

    def test_l3_mining_uses_v2_directory_root(self, page: Page, server_url: str):
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-1",
                  wait_until="networkidle")
        page.wait_for_timeout(1500)

        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const roots = d._l3ArtifactRootSpecs ? d._l3ArtifactRootSpecs() : [];
            return {
                level: d.currentLevel,
                node: d.currentNode,
                round: d.currentRound,
                paths: roots.map(s => s.path),
            };
        }""" % FIND_APP_JS)

        assert snap, "campaignApp not mounted on L3"
        assert snap["level"] == 3 and snap["round"] == 1, snap
        assert "rounds/1/mining" in snap["paths"], (
            f"v2 mining root missing from L3 browser: {snap}"
        )

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    def test_l3_debate_root_is_campaign_round_scoped(self, page: Page, server_url: str):
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-2",
                  wait_until="networkidle")
        page.wait_for_timeout(1500)

        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const roots = d._l3ArtifactRootSpecs ? d._l3ArtifactRootSpecs() : [];
            return {
                round: d.currentRound,
                paths: roots.map(s => s.path),
            };
        }""" % FIND_APP_JS)

        assert snap and snap["round"] == 1, snap
        assert snap["paths"] == ["rounds/1/debate"]

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    def test_l3_validation_column_shows_validator_tests(self, page: Page, server_url: str):
        """Validator tests under `tracks/{op}/validator_tests/` are stage='validation'."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        # stage-4 is Validation column in L3.
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-4",
                  wait_until="networkidle")
        page.wait_for_timeout(1500)

        # Check the parsed metadata via the catalog directly: the validator
        # test path must parse as stage='validation'.
        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const cat = d.artifactCatalog || {};
            const isFlat = !cat.entries;
            const path = 'rounds/1/tracks/op-001/validator_tests/test_correctness.py';
            const entry = isFlat ? cat[path] : (cat.entries || {})[path];
            // Use the FE helper to parse the path semantically.
            const helpers = (typeof window !== 'undefined') ? (window.LG_HELPERS || {}) : {};
            const parsed = helpers.parseArtifactPath ? helpers.parseArtifactPath(path) : null;
            return {
                level: d.currentLevel,
                hasEntry: !!entry,
                parsed,
            };
        }""" % FIND_APP_JS)

        assert snap, "campaignApp not mounted"
        assert snap["level"] == 3, snap
        assert snap["hasEntry"], "validator_tests sidecar missing from catalog"
        assert snap["parsed"] is not None, "LG_HELPERS.parseArtifactPath not exported"
        assert snap["parsed"]["stage"] == "validation", snap
        assert snap["parsed"]["track_id"] == "op-001", snap
        assert snap["parsed"]["round"] == 1, snap

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"

    # --------------------------------------------------------- Console gate

    def test_no_console_errors_during_v2_navigation(self, page: Page, server_url: str):
        """Full L1 → L2 → L3 walkthrough without a single JS exception."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page, state=_v2_state(), sidecars=_v2_sidecars(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)

        # L1
        page.goto(f"{server_url}/ui#campaigns", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L2
        page.goto(f"{server_url}/ui#campaigns/{SID}", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L3 baseline
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-0", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L3 mining
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-1", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L3 debate
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-2", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L3 implementation (R1)
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-3", wait_until="networkidle")
        page.wait_for_timeout(800)
        # L3 integration (R1)
        page.goto(f"{server_url}/ui#campaigns/{SID}/1/stage-5", wait_until="networkidle")
        page.wait_for_timeout(800)

        real = _filter_real_errors(errors)
        assert real == [], f"console errors during navigation: {real}"

    # --------------------------------------------------------- Back-compat

    def test_backward_compat_old_artifact_catalog_shape(self, page: Page, server_url: str):
        """Server still returns `artifact_catalog` envelope → FE keeps working."""
        _, errors = _attach_console_collectors(page)
        _install_routes(page,
                        state=_legacy_state_single_round(),
                        sidecars=None,
                        legacy_catalog=_legacy_artifact_catalog(),
                        with_pipeline_progress=True, with_counts=True)
        _prime(page, server_url)
        page.goto(f"{server_url}/ui#campaigns/{SID}", wait_until="networkidle")
        page.wait_for_timeout(1500)

        snap = page.evaluate("""() => {
            const d = %s;
            if (!d) return null;
            const cat = d.artifactCatalog;
            if (!cat) return {hasCatalog: false};
            // Legacy: envelope under .entries OR flattened to entries map.
            const isEnvelope = !!cat.entries;
            const isFlat = !cat.entries && typeof cat === 'object';
            const keys = Object.keys(isEnvelope ? cat.entries : cat);
            return {
                hasCatalog: true,
                level: d.currentLevel,
                isEnvelope,
                isFlat,
                keys: keys.sort(),
            };
        }""" % FIND_APP_JS)

        assert snap and snap["hasCatalog"], snap
        assert snap["level"] == 2, snap
        # In Task 2 the FE prefers `data.sidecars`; when absent, it falls
        # back to `data.artifact_catalog.entries` (flat) and only finally to
        # `data.artifact_catalog` (envelope). Either flat or envelope is OK
        # here as long as the legacy keys land in the catalog.
        assert "bottleneck_analysis.md" in snap["keys"], snap
        assert "tracks/op-001/implementation/opt.py" in snap["keys"], snap

        real = _filter_real_errors(errors)
        assert real == [], f"console errors: {real}"
