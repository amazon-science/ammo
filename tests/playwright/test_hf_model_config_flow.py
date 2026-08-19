# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for the HF model-config auto-detection flow.

Covers both the LIGHTGRID campaignApp and the Classic sessionApp after the
removal of the static `SUPPORTED_MODELS` preset list.  GPU info now comes
from `/health` (gpu/vllm blocks), and model selection drives an HF
`config.json` fetch via `/api/hf-model-config/{model_id}` that auto-fills
TP / DP / dtype on the create form.

Tests 22-29 from docs plan remove-static-models.md:

 22. LIGHTGRID HF select triggers /api/hf-model-config fetch.
 23. LIGHTGRID autofills TP / DP / dtype from the config response.
 24. LIGHTGRID shows a manual-TP/DP hint for gated models.
 25. LIGHTGRID renders no preset chips (legacy cmSupportedModels removed).
 26. Classic HF select triggers /api/hf-model-config + autofill.
 27. Classic renders no preset list & `supportedModels` is absent from state.
 28. LIGHTGRID loads GPU info from /health (never calls /api/supported-models).
 29. LIGHTGRID create modal defers GPU info display until /health resolves.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

MOCK_HEALTH_RESPONSE = {
    "status": "healthy",
    "gpu": {
        "type": "h200",
        "allowed_dtypes": ["fp8", "fp4", "bf16", "fp16"],
        "total_gpus": 8,
        "available_gpus": 8,
    },
    "vllm": {
        "docker_commit": "a" * 40,
        "version": "v0.20.0",
    },
    "gpu_manager": {"total_gpus": 8, "available_gpus": 8},
    "job_stats": {},
}


# Config fixtures keyed by model id.  Each entry is the JSON returned by the
# new /api/hf-model-config/{model_id} endpoint.
MOCK_HF_CONFIGS = {
    "deepseek-ai/DeepSeek-R1": {
        "model_id": "deepseek-ai/DeepSeek-R1",
        "is_moe": True,
        "suggested_tp": 8,
        "suggested_dp": 1,
        "suggested_dtype": "fp8",
        "reason": None,
        "config": {"hidden_size": 7168, "num_local_experts": 256},
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "is_moe": False,
        "suggested_tp": 4,
        "suggested_dp": 1,
        "suggested_dtype": "bf16",
        "reason": None,
        "config": {"hidden_size": 4096},
    },
    "google/gemma-2-9b-it": {
        "model_id": "google/gemma-2-9b-it",
        "is_moe": False,
        "suggested_tp": None,
        "suggested_dp": None,
        "suggested_dtype": None,
        "reason": "gated",
        "config": None,
    },
}


MOCK_HF_SEARCH_RESPONSE = {
    "models": [
        {
            "id": "deepseek-ai/DeepSeek-R1",
            "downloads": 5_000_000,
            "likes": 1000,
            "pipeline_tag": "text-generation",
            "tags": ["model_type:deepseek_v3"],
        },
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "downloads": 1_000_000,
            "likes": 500,
            "pipeline_tag": "text-generation",
            "tags": [],
        },
    ],
    "source": "huggingface",
}


def _make_handler(
    health_override: dict | None = None,
    config_override: dict | None = None,
    search_override: dict | None = None,
    *,
    health_delay_ms: int = 0,
    track: dict | None = None,
):
    """Produce a Playwright route handler with optional overrides and tracking.

    ``track`` is a mutable dict that the handler mutates in place to record
    which URLs were hit.  Tests assert on the contents afterwards.
    """

    def handler(route, request):
        url = request.url
        method = request.method
        if track is not None:
            track.setdefault("urls", []).append(url)

        # /health
        if url.endswith("/health"):
            payload = health_override or MOCK_HEALTH_RESPONSE
            if health_delay_ms > 0:
                # Playwright has no native delay mechanism in fulfill; simulate
                # slowness by waiting on the page's wall clock before fulfilling.
                import time
                time.sleep(health_delay_ms / 1000.0)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        # /api/hf-model-config/{model_id:path}
        if "/api/hf-model-config/" in url:
            model_id = url.split("/api/hf-model-config/", 1)[1]
            # HF model ids can contain an encoded slash; decode the common case.
            model_id = model_id.replace("%2F", "/").split("?", 1)[0]
            if track is not None:
                track.setdefault("config_fetches", []).append(model_id)
            payload = (config_override or {}).get(
                model_id, MOCK_HF_CONFIGS.get(model_id)
            )
            if payload is None:
                payload = {
                    "model_id": model_id,
                    "is_moe": False,
                    "suggested_tp": 1,
                    "suggested_dp": 1,
                    "suggested_dtype": "bf16",
                    "reason": None,
                    "config": {},
                }
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        # /api/hf-models (search)
        if "/api/hf-models" in url:
            payload = search_override or MOCK_HF_SEARCH_RESPONSE
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        # /api/supported-models → removed; return 404 to guarantee that any
        # accidental remaining call will fail loudly.
        if "/api/supported-models" in url or "/api/moe-models" in url:
            route.fulfill(
                status=404,
                content_type="application/json",
                body=json.dumps({"detail": "Not Found"}),
            )
            return

        # /sessions (list)
        if url.endswith("/sessions") or (
            "/sessions" in url and method == "GET"
        ):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"sessions": []}),
            )
            return

        # /api/changelog
        if "/api/changelog" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"version": "1.2.0", "entries": []}),
            )
            return

        # /api/campaigns
        if "/api/campaigns" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"campaigns": []}),
            )
            return

        # /health
        if url.endswith("/health"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"cluster": {}, "pods": []}),
            )
            return

        route.continue_()

    return handler


def _install(page: Page, **kwargs):
    handler = _make_handler(**kwargs)
    page.route("**/api/**", handler)
    page.route("**/sessions/**", handler)
    page.route("**/sessions", handler)
    page.route("**/health", handler)
    page.route("**/cluster/**", handler)


# ---------------------------------------------------------------------------
# Page fixtures — LIGHTGRID theme + Classic theme
# ---------------------------------------------------------------------------


@pytest.fixture
def lightgrid_page(context, server_url) -> Page:
    pg = context.new_page()
    _install(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


@pytest.fixture
def classic_page(context, server_url) -> Page:
    pg = context.new_page()
    _install(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_lg_modal(pg: Page):
    pg.evaluate(
        "() => window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        ").openCreateModal()"
    )
    expect(pg.locator(".lg-modal:has(.lg-modal-body-grid)")).to_be_visible()
    pg.wait_for_timeout(300)


def _lg_select_hf(pg: Page, model_id: str):
    """Invoke cmSelectHfModel directly with a synthetic HF result."""
    pg.evaluate(
        """async (id) => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            await d.cmSelectHfModel({id: id, pipeline_tag: 'text-generation', tags: []});
        }""",
        model_id,
    )
    pg.wait_for_timeout(200)


def _lg_data(pg: Page) -> dict:
    return pg.evaluate(
        "() => {"
        "  const d = window.Alpine.$data("
        "    document.querySelector('[x-data=\"campaignApp\"]')"
        "  );"
        "  return {"
        "    tp: d.cmForm.tp, dp: d.cmForm.dp, dtype: d.cmForm.dtype,"
        "    isMoe: d.cmIsMoe, gatedHint: !!d.cmGatedHint,"
        "    gpuInfo: d.cmGpuInfo,"
        "    gpuInfoLoaded: !!d.cmGpuInfoLoaded,"
        "    hasSupportedModels: 'cmSupportedModels' in d,"
        "  };"
        "}"
    )


def _classic_open_modal(pg: Page):
    pg.evaluate(
        "() => { const d = window.Alpine.$data("
        "  document.querySelector('[x-data=\"sessionApp()\"]')"
        "); d.showCreateModal = true; }"
    )
    pg.wait_for_timeout(200)


def _classic_select_hf(pg: Page, model_id: str):
    pg.evaluate(
        """async (id) => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="sessionApp()"]')
            );
            await d.selectHfModel({id: id, pipeline_tag: 'text-generation', tags: []});
        }""",
        model_id,
    )
    pg.wait_for_timeout(200)


def _classic_data(pg: Page) -> dict:
    return pg.evaluate(
        "() => {"
        "  const d = window.Alpine.$data("
        "    document.querySelector('[x-data=\"sessionApp()\"]')"
        "  );"
        "  return {"
        "    tp: d.createForm.tp, dp: d.createForm.dp, dtype: d.createForm.dtype,"
        "    gpuInfo: d.gpuInfo,"
        "    hasSupportedModels: 'supportedModels' in d,"
        "  };"
        "}"
    )


# ---------------------------------------------------------------------------
# Test 22 — LIGHTGRID HF select triggers config fetch
# ---------------------------------------------------------------------------


def test_lightgrid_hf_select_triggers_config_fetch(context, server_url):
    track: dict = {}
    pg = context.new_page()
    _install(pg, track=track)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _open_lg_modal(pg)
    _lg_select_hf(pg, "deepseek-ai/DeepSeek-R1")

    assert "deepseek-ai/DeepSeek-R1" in track.get("config_fetches", []), (
        f"Expected /api/hf-model-config/deepseek-ai/DeepSeek-R1 to be hit; "
        f"actual config_fetches={track.get('config_fetches', [])}"
    )
    state = _lg_data(pg)
    assert state["tp"] == 8
    assert state["dtype"] == "fp8"
    pg.close()


# ---------------------------------------------------------------------------
# Test 23 — LIGHTGRID autofills TP/DP/dtype/is_moe from config
# ---------------------------------------------------------------------------


def test_lightgrid_hf_config_autofills_tp_dp_dtype(lightgrid_page: Page):
    _open_lg_modal(lightgrid_page)
    _lg_select_hf(lightgrid_page, "meta-llama/Llama-3.1-8B-Instruct")

    state = _lg_data(lightgrid_page)
    assert state["tp"] == 4
    assert state["dp"] == 1
    assert state["dtype"] == "bf16"
    assert state["isMoe"] is False


# ---------------------------------------------------------------------------
# Test 24 — Gated model shows manual-TP/DP hint; TP/DP not overwritten
# ---------------------------------------------------------------------------


def test_lightgrid_hf_config_gated_shows_manual_message(lightgrid_page: Page):
    _open_lg_modal(lightgrid_page)
    # Record TP/DP before selection
    before = _lg_data(lightgrid_page)

    _lg_select_hf(lightgrid_page, "google/gemma-2-9b-it")

    after = _lg_data(lightgrid_page)
    assert after["gatedHint"] is True
    # TP and DP should NOT have been overwritten by a gated model.
    assert after["tp"] == before["tp"]
    assert after["dp"] == before["dp"]


# ---------------------------------------------------------------------------
# Test 25 — No preset chips rendered in LIGHTGRID create modal
# ---------------------------------------------------------------------------


def test_lightgrid_no_preset_chips_rendered(lightgrid_page: Page):
    _open_lg_modal(lightgrid_page)
    # The preset-chip row was keyed off `.lg-preset-chip` + `.lg-preset-row`.
    expect(lightgrid_page.locator(".lg-preset-chip")).to_have_count(0)
    expect(lightgrid_page.locator(".lg-preset-row")).to_have_count(0)

    state = _lg_data(lightgrid_page)
    # Alpine state no longer exposes the legacy preset list.
    assert state["hasSupportedModels"] is False


# ---------------------------------------------------------------------------
# Test 26 — Classic HF select triggers config fetch + autofill
# ---------------------------------------------------------------------------


def test_classic_hf_select_triggers_config_fetch(context, server_url):
    track: dict = {}
    pg = context.new_page()
    _install(pg, track=track)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _classic_open_modal(pg)
    _classic_select_hf(pg, "meta-llama/Llama-3.1-8B-Instruct")

    assert "meta-llama/Llama-3.1-8B-Instruct" in track.get("config_fetches", []), (
        f"Expected /api/hf-model-config hit; "
        f"got {track.get('config_fetches', [])}"
    )
    state = _classic_data(pg)
    assert state["tp"] == 4
    assert state["dtype"] == "bf16"
    pg.close()


# ---------------------------------------------------------------------------
# Test 27 — Classic modal has no preset list; `supportedModels` absent
# ---------------------------------------------------------------------------


def test_classic_no_preset_list_rendered(classic_page: Page):
    _classic_open_modal(classic_page)

    # No "Recommended Presets" heading, no "Matching Presets" heading.
    expect(
        classic_page.locator("text=Recommended Presets")
    ).to_have_count(0)
    expect(
        classic_page.locator("text=Matching Presets")
    ).to_have_count(0)

    state = _classic_data(classic_page)
    assert state["hasSupportedModels"] is False, (
        "sessionApp should no longer expose `supportedModels` state"
    )


# ---------------------------------------------------------------------------
# Test 28 — LIGHTGRID GPU info comes from /health (never /api/supported-models)
# ---------------------------------------------------------------------------


def test_lightgrid_gpu_info_from_health_not_supported_models(context, server_url):
    track: dict = {}
    pg = context.new_page()
    _install(pg, track=track)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _open_lg_modal(pg)
    state = _lg_data(pg)
    assert state["gpuInfo"]["type"] == "h200"
    assert "fp8" in state["gpuInfo"]["allowed_dtypes"]

    urls = track.get("urls", [])
    assert not any("/api/supported-models" in u for u in urls), (
        f"campaignApp must not call /api/supported-models. Offending URLs: "
        f"{[u for u in urls if '/api/supported-models' in u]}"
    )
    # /health must have been hit at least once.
    assert any(u.endswith("/health") for u in urls), (
        f"campaignApp must call /health. URLs seen: {urls[:20]}"
    )
    pg.close()


# ---------------------------------------------------------------------------
# Test 29 — Create modal defers / shows loading until /health resolves
# ---------------------------------------------------------------------------


def test_lightgrid_create_modal_defers_when_health_pending(context, server_url):
    """/health is deliberately held open until the test releases it.

    Before the release: `cmGpuInfoLoaded` is false.  After: it flips to true
    and `cmGpuInfo.type` carries the /health payload.

    We simulate the "pending /health" state by intercepting the request via
    `page.route`, storing the route object, and fulfilling it later in the
    test when we're ready.  This keeps the request authentically in flight
    without blocking the Playwright event loop.
    """
    pg = context.new_page()

    # Custom handler that defers /health by stashing the Route object in a
    # closure and fulfilling all other mocked endpoints immediately.
    pending_health_routes: list = []

    def handler(route, request):
        url = request.url
        if url.endswith("/health"):
            # Stash — will be fulfilled manually below.
            pending_health_routes.append(route)
            return
        # Reuse the default mock behaviour for everything else.
        default = _make_handler()
        default(route, request)

    pg.route("**/api/**", handler)
    pg.route("**/sessions/**", handler)
    pg.route("**/sessions", handler)
    pg.route("**/health", handler)
    pg.route("**/cluster/**", handler)

    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.goto(f"{server_url}/ui", wait_until="domcontentloaded")

    # Fire openCreateModal() without awaiting — /health is still held.
    pg.wait_for_timeout(400)
    pg.evaluate(
        "() => { const d = window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        "); d.openCreateModal(); }"
    )
    pg.wait_for_timeout(300)

    pending = pg.evaluate(
        "() => { const d = window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        "); return !!d.cmGpuInfoLoaded; }"
    )
    assert pending is False, (
        "cmGpuInfoLoaded should stay false while /health in flight"
    )

    # Release all pending /health responses.
    assert len(pending_health_routes) >= 1, (
        f"Expected at least 1 /health request in flight, "
        f"got {len(pending_health_routes)}"
    )
    for route in pending_health_routes:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(MOCK_HEALTH_RESPONSE),
        )
    pg.wait_for_timeout(400)

    resolved = pg.evaluate(
        "() => { const d = window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        "); return {loaded: !!d.cmGpuInfoLoaded, type: d.cmGpuInfo.type}; }"
    )
    assert resolved["loaded"] is True
    assert resolved["type"] == "h200"
    pg.close()
