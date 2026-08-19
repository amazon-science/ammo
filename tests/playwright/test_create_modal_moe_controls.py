# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for MoE-aware DP/EP parallelism controls in the LIGHTGRID
create-session modal.

Covers:
- MoE preset enables DP/EP pills; dense preset disables them.
- GPU requirement badge = TP × DP, amber→red when over server capacity.
- Prompt preview includes --data-parallel-size / --enable-expert-parallel
  only when the user opts in.
- Payload submitted to POST /sessions contains tp_size, dp_size, and
  gpu_count = tp * dp.
- MoE→Dense transition resets dp/ep to safe defaults (prevents stale
  MoE state leaking into a model that cannot use it).

All tests use mocked API routes; the tests exercise the LIGHTGRID UI path,
not the legacy classic modal. The LIGHTGRID create modal is opened by
calling ``openCreateModal()`` on the Alpine campaignApp root.
"""

from __future__ import annotations

import copy
import json

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Mock fixtures — /health (GPU info) + /api/hf-model-config/{id} per preset
# ---------------------------------------------------------------------------
#
# The static /api/supported-models preset list was removed. GPU info now
# lives on /health; per-model TP/DP/dtype suggestions come from
# /api/hf-model-config/{id} on demand (driven by cmSelectHfModel()).

MOCK_HEALTH_H200_8GPU = {
    "status": "healthy",
    "job_stats": {"total_jobs": 0},
    "gpu_manager": {"total_gpus": 8, "available_gpus": 8},
    "gpu": {
        "type": "h200",
        "allowed_dtypes": ["fp8", "fp4", "bf16", "fp16"],
        "total_gpus": 8,
        "available_gpus": 8,
    },
    "vllm": {"docker_commit": None, "version": None},
}

# Keyed by HF model id — drives the /api/hf-model-config/{id} mock.
MOCK_HF_CONFIGS: dict[str, dict] = {
    "deepseek-ai/DeepSeek-R1-0528": {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "is_moe": True,
        "suggested_tp": 8,
        "suggested_dp": 1,
        "suggested_dtype": "fp8",
        "reason": None,
        "config": {
            "hidden_size": 7168,
            "num_hidden_layers": 60,
            "num_local_experts": 256,
        },
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "is_moe": False,
        "suggested_tp": 1,
        "suggested_dp": 1,
        "suggested_dtype": "fp8",
        "reason": None,
        "config": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    },
}


def _install_routes(page: Page, health: dict | None = None):
    """Install minimal route handlers — /health + /api/hf-model-config/{id}
    + sessions list. Replaces the old /api/supported-models mock."""
    health_payload = health or MOCK_HEALTH_H200_8GPU

    def handler(route, request):
        url = request.url
        if url.endswith("/health") or url.endswith("/health/"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(health_payload),
            )
            return
        if "/api/hf-model-config/" in url:
            # Extract model_id from URL (everything after /api/hf-model-config/).
            tail = url.split("/api/hf-model-config/", 1)[1].split("?", 1)[0]
            # URL-decode basic %2F etc.
            from urllib.parse import unquote
            model_id = unquote(tail)
            cfg = MOCK_HF_CONFIGS.get(model_id, {
                "model_id": model_id,
                "is_moe": False,
                "suggested_tp": 1,
                "suggested_dp": 1,
                "suggested_dtype": "bf16",
                "reason": None,
                "config": {"hidden_size": 4096, "num_hidden_layers": 32},
            })
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(cfg),
            )
            return
        if url.endswith("/sessions") or (
            "/sessions" in url and request.method == "GET"
        ):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"sessions": []}),
            )
            return
        if "/api/changelog" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"version": "1.2.0", "entries": []}),
            )
            return
        route.continue_()

    page.route("**/api/**", handler)
    page.route("**/sessions/**", handler)
    page.route("**/sessions", handler)
    page.route("**/health", handler)


@pytest.fixture
def lightgrid_page(context, server_url) -> Page:
    """Page with LIGHTGRID theme active and create-modal-ready Alpine state."""
    pg = context.new_page()
    _install_routes(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    # Force LIGHTGRID theme; suppress tours/welcome.
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


def _open_modal_and_wait(pg: Page):
    """Invoke the Alpine helper directly to avoid brittle click resolution."""
    pg.evaluate(
        "() => window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        ").openCreateModal()"
    )
    expect(pg.locator(".lg-modal:has(.lg-modal-body-grid)")).to_be_visible()
    # Let the open animation/data load settle.
    pg.wait_for_timeout(300)


def _data(pg: Page) -> dict:
    """Snapshot the relevant Alpine state as a dict for assertions.

    `gpuTotal` is the legacy TP×DP floor (now an alias of `cmGpuMin()`);
    `gpuCount` is the decoupled pool value bound to the new stepper.
    """
    return pg.evaluate(
        "() => {"
        "  const d = window.Alpine.$data("
        "    document.querySelector('[x-data=\"campaignApp\"]')"
        "  );"
        "  return {"
        "    tp: d.cmForm.tp, dp: d.cmForm.dp, ep: d.cmForm.ep,"
        "    gpuCount: d.cmForm.gpuCount,"
        "    gpuMin: d.cmGpuMin(), gpuMax: d.cmGpuMax(),"
        "    isMoe: d.cmIsMoe, gpuTotal: d.cmGpuTotal(),"
        "    serverGpus: d.cmServerGpuCount(), over: d.cmGpuOverCapacity(),"
        "    prompt: d.cmGeneratePrompt(), canCreate: d.cmCanCreate()"
        "  };"
        "}"
    )


def _select_preset(pg: Page, preset_id: str):
    """Simulate selecting an HF model — replaces cmSelectPreset() which was
    removed along with the static preset list. Drives cmSelectHfModel() with
    the model id so the UI fetches /api/hf-model-config and auto-fills TP/DP/
    dtype. Waits on $nextTick so the awaited config fetch settles before
    assertions read state."""
    # Minimal "HF model" shape accepted by cmSelectHfModel — the handler only
    # reads model.id and (optionally) model.tags for the MoE fallback regex.
    pg.evaluate(
        """async (id) => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            await d.cmSelectHfModel({ id: id, tags: [] });
            await d.$nextTick();
            await d.$nextTick();
        }""",
        preset_id,
    )
    # Belt-and-suspenders: give the awaited fetch + watchers a beat to settle
    # so gpuCount clamps, etc., are applied before the test inspects state.
    pg.wait_for_timeout(150)


# ---------------------------------------------------------------------------
# T6.1 — MoE preset enables DP/EP
# ---------------------------------------------------------------------------


def test_moe_preset_enables_dp_ep_fields(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")

    # After the parallelism-grouping refactor, DP and EP are rows inside
    # the single `.lg-parallelism` block, keyed off `.lg-para-row` with
    # `.lg-para-disabled` when dense. Row filter uses the DP/EP key text.
    para = lightgrid_page.locator(
        ".lg-modal:has(.lg-modal-body-grid) .lg-parallelism"
    )
    dp_row = para.locator(".lg-para-row").filter(
        has=lightgrid_page.locator(".lg-para-key", has_text="DP")
    )
    ep_row = para.locator(".lg-para-row").filter(
        has=lightgrid_page.locator(".lg-para-key", has_text="EP")
    )
    expect(dp_row).to_be_visible()
    expect(ep_row).to_be_visible()
    expect(dp_row).not_to_have_class("lg-para-row lg-para-disabled")
    expect(ep_row).not_to_have_class("lg-para-row lg-para-disabled")
    # Pills should be keyboard-reachable when MoE.
    first_dp_pill = dp_row.locator(".lg-pill").first
    expect(first_dp_pill).to_have_attribute("tabindex", "0")

    state = _data(lightgrid_page)
    assert state["isMoe"] is True


# ---------------------------------------------------------------------------
# T6.2 — Dense preset disables DP/EP (no click attempt — DOM assert only)
# ---------------------------------------------------------------------------


def test_dense_preset_disables_dp_ep_fields(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "meta-llama/Llama-3.1-8B-Instruct")

    # Dense path: DP and EP rows carry `.lg-para-disabled`.
    para = lightgrid_page.locator(
        ".lg-modal:has(.lg-modal-body-grid) .lg-parallelism"
    )
    dp_row = para.locator(".lg-para-row.lg-para-disabled").filter(
        has=lightgrid_page.locator(".lg-para-key", has_text="DP")
    )
    ep_row = para.locator(".lg-para-row.lg-para-disabled").filter(
        has=lightgrid_page.locator(".lg-para-key", has_text="EP")
    )
    expect(dp_row).to_be_visible()
    expect(ep_row).to_be_visible()
    # Every pill in the disabled rows is removed from the tab order.
    for pill in dp_row.locator(".lg-pill").all():
        expect(pill).to_have_attribute("tabindex", "-1")
    for pill in ep_row.locator(".lg-pill").all():
        expect(pill).to_have_attribute("tabindex", "-1")

    state = _data(lightgrid_page)
    assert state["isMoe"] is False
    assert state["dp"] == 1
    assert state["ep"] is False


# ---------------------------------------------------------------------------
# T6.3 — GPU badge math: = TP × DP = N GPUs
# ---------------------------------------------------------------------------


def test_gpu_badge_reflects_tp_times_dp(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    # DeepSeek preset sets tp=8, dp=1 → 8 GPUs.
    badge = lightgrid_page.locator(".lg-modal:has(.lg-modal-body-grid) .lg-linked-tag").first
    expect(badge).to_contain_text("TP×DP = 8 GPUs")

    # Bump DP to 2 → 16 GPUs (still MoE, via data mutation).
    lightgrid_page.evaluate(
        "() => { window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')).cmForm.dp = 2; }"
    )
    expect(badge).to_contain_text("TP×DP = 16 GPUs")


# ---------------------------------------------------------------------------
# T6.4 — Over-capacity blocks submit + paints badge red
# ---------------------------------------------------------------------------


def test_over_capacity_blocks_submit_and_paints_red(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    # Force tp*dp > 8 (pod capacity from fixture). The $watch on cmForm.tp
    # and cmForm.dp calls cmClampGpuCount(), which bumps gpuCount from the
    # preset's 8 up to the new min of 32 — triggering over-capacity.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 8;
            d.cmForm.dp = 4;
            await d.$nextTick();
        }"""
    )
    state = _data(lightgrid_page)
    # gpuTotal is the TP×DP floor alias (cmGpuMin).
    assert state["gpuTotal"] == 32
    # gpuCount was clamped up to the new min by the $watch.
    assert state["gpuCount"] == 32
    assert state["serverGpus"] == 8
    assert state["over"] is True
    assert state["canCreate"] is False

    # The over-capacity class lives on the GPU Count badge (the 2nd
    # `.lg-linked-tag` in the modal) — the Parallelism badge shows the
    # floor only. Scope to the `.lg-gpu-count` field wrapper.
    gpu_count_badge = lightgrid_page.locator(
        ".lg-modal:has(.lg-modal-body-grid) .lg-gpu-count .lg-linked-tag"
    )
    expect(gpu_count_badge).to_have_class("lg-linked-tag over-capacity")


# ---------------------------------------------------------------------------
# T6.5 — Prompt flags appear only when dp>1 / ep==true
# ---------------------------------------------------------------------------


def test_prompt_excludes_flags_by_default(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    state = _data(lightgrid_page)
    assert state["prompt"].startswith("Use $ammo for model_id=deepseek-ai/DeepSeek-R1-0528")
    assert "--data-parallel-size" not in state["prompt"]
    assert "--enable-expert-parallel" not in state["prompt"]


def test_prompt_includes_flags_when_set(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    lightgrid_page.evaluate(
        "() => { const d = window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')); d.cmForm.dp = 2; d.cmForm.ep = true; }"
    )
    state = _data(lightgrid_page)
    assert state["prompt"].startswith("Use $ammo for model_id=deepseek-ai/DeepSeek-R1-0528")
    assert "--data-parallel-size 2" in state["prompt"]
    assert "--enable-expert-parallel" in state["prompt"]


# ---------------------------------------------------------------------------
# T6.6 — POST /sessions payload has tp_size, dp_size, gpu_count = tp*dp
# ---------------------------------------------------------------------------


def test_create_session_payload_has_tp_dp_and_gpu_count(
    context, server_url
):
    pg = context.new_page()
    _install_routes(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")

    captured = []

    def handler(route, request):
        url = request.url
        if request.method == "POST" and url.rstrip("/").endswith("/sessions"):
            captured.append(json.loads(request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "session_id": "test-moe-001",
                    "status": "active",
                    "cli_tool": "claude",
                    "repo_name": "vllm",
                    "gpu_ids": [0, 1, 2, 3],
                    "terminal_url": "/sessions/test-moe-001/terminal/",
                    "terminal_ws_url": "/sessions/test-moe-001/terminal/ws",
                    "created_at": "2026-04-27T00:00:00",
                    "last_accessed": 1777315832,
                    "owner_id": "mock",
                    "model_name": "deepseek-ai/DeepSeek-R1-0528",
                    "dtype": "fp8",
                }),
            )
            return
        if url.endswith("/health") or url.endswith("/health/"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_HEALTH_H200_8GPU),
            )
            return
        if "/api/hf-model-config/" in url:
            from urllib.parse import unquote
            tail = url.split("/api/hf-model-config/", 1)[1].split("?", 1)[0]
            model_id = unquote(tail)
            cfg = MOCK_HF_CONFIGS.get(model_id, {
                "model_id": model_id,
                "is_moe": False,
                "suggested_tp": 1,
                "suggested_dp": 1,
                "suggested_dtype": "bf16",
                "reason": None,
                "config": {"hidden_size": 4096, "num_hidden_layers": 32},
            })
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(cfg),
            )
            return
        if url.endswith("/sessions") or ("/sessions" in url and request.method == "GET"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"sessions": []}),
            )
            return
        if "/api/changelog" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"version": "x", "entries": []}))
            return
        route.continue_()

    pg.route("**/api/**", handler)
    pg.route("**/sessions/**", handler)
    pg.route("**/sessions", handler)
    pg.route("**/health", handler)
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _open_modal_and_wait(pg)
    _select_preset(pg, "deepseek-ai/DeepSeek-R1-0528")
    # tp=2, dp=2 → min 4 GPUs, under fixture capacity of 8. Post-decouple,
    # the preset sets gpuCount=cmGpuMin()=8 (DeepSeek default_tp=8), and
    # the $watch clamps to min on tp/dp change but never shrinks past the
    # user's value — so we must explicitly set gpuCount=4 for the payload
    # contract test. Use $nextTick to ensure the clamp settles.
    pg.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 2;
            d.cmForm.dp = 2;
            d.cmForm.ep = true;
            await d.$nextTick();
            d.cmForm.gpuCount = 4;
            await d.$nextTick();
        }"""
    )
    pg.evaluate(
        "() => window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        ").cmCreateSession()"
    )
    pg.wait_for_timeout(500)

    assert len(captured) >= 1, "expected at least one POST /sessions"
    payload = captured[0]
    assert payload["model_name"] == "deepseek-ai/DeepSeek-R1-0528"
    assert payload["tp_size"] == 2
    assert payload["dp_size"] == 2
    assert payload["gpu_count"] == 4
    assert payload["initial_prompt"].startswith("Use $ammo for model_id=deepseek-ai/DeepSeek-R1-0528")
    assert "--data-parallel-size 2" in payload["initial_prompt"]
    assert "--enable-expert-parallel" in payload["initial_prompt"]
    pg.close()


def test_create_session_payload_can_select_codex_harness(
    context, server_url
):
    pg = context.new_page()
    _install_routes(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")

    captured = []

    def handler(route, request):
        url = request.url
        if request.method == "POST" and url.rstrip("/").endswith("/sessions"):
            captured.append(json.loads(request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "session_id": "test-codex-001",
                    "status": "active",
                    "cli_tool": "codex",
                    "repo_name": "vllm",
                    "gpu_ids": [0],
                    "terminal_url": "/sessions/test-codex-001/terminal/",
                    "terminal_ws_url": "/sessions/test-codex-001/terminal/ws",
                    "created_at": "2026-04-27T00:00:00",
                    "last_accessed": 1777315832,
                    "owner_id": "mock",
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                    "dtype": "fp8",
                }),
            )
            return
        if url.endswith("/health") or url.endswith("/health/"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_HEALTH_H200_8GPU),
            )
            return
        if "/api/hf-model-config/" in url:
            from urllib.parse import unquote
            tail = url.split("/api/hf-model-config/", 1)[1].split("?", 1)[0]
            model_id = unquote(tail)
            cfg = MOCK_HF_CONFIGS.get(model_id, {
                "model_id": model_id,
                "is_moe": False,
                "suggested_tp": 1,
                "suggested_dp": 1,
                "suggested_dtype": "bf16",
                "reason": None,
                "config": {"hidden_size": 4096, "num_hidden_layers": 32},
            })
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(cfg),
            )
            return
        if url.endswith("/sessions") or ("/sessions" in url and request.method == "GET"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"sessions": []}),
            )
            return
        if "/api/changelog" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"version": "x", "entries": []}))
            return
        route.continue_()

    pg.route("**/api/**", handler)
    pg.route("**/sessions/**", handler)
    pg.route("**/sessions", handler)
    pg.route("**/health", handler)
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _open_modal_and_wait(pg)
    _select_preset(pg, "meta-llama/Llama-3.1-8B-Instruct")
    pg.evaluate(
        "() => { const d = window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')); d.cmForm.cliTool = 'codex'; }"
    )
    pg.evaluate(
        "() => window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        ").cmCreateSession()"
    )
    pg.wait_for_timeout(500)

    assert len(captured) >= 1, "expected at least one POST /sessions"
    assert captured[0]["cli_tool"] == "codex"
    pg.close()


# ---------------------------------------------------------------------------
# T6.7 — MoE→Dense HF transition resets dp/ep (no stale state leakage)
# ---------------------------------------------------------------------------


def test_moe_to_dense_hf_resets_dp_ep(lightgrid_page: Page):
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    # Simulate user picking DP=2, EP=on while on a MoE model.
    lightgrid_page.evaluate(
        "() => { const d = window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')); d.cmForm.dp = 2; d.cmForm.ep = true; }"
    )
    # Now the user picks a dense HF model (no MoE tag, no MoE name).
    lightgrid_page.evaluate(
        """() => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmSelectHfModel({
                id: 'meta-llama/Llama-3.1-70B-Instruct',
                tags: ['model_type:llama'],
            });
        }"""
    )
    state = _data(lightgrid_page)
    assert state["isMoe"] is False
    assert state["dp"] == 1
    assert state["ep"] is False


# ---------------------------------------------------------------------------
# T6.8 — cmDetectMoe: HF tags (structured) + regex fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ({"id": "some/Mixtral-8x7B-v0.1"}, True),                    # regex
        ({"id": "mistralai/Mistral-7B-Instruct-v0.3"}, False),       # neither
        ({"id": "x/y", "tags": ["model_type:qwen3_moe"]}, True),     # tags
        ({"id": "x/y", "tags": ["model_type:llama"]}, False),        # non-MoE tag
        ({"id": "x/y", "tags": []}, False),                          # empty tags
        ({"id": "foo/bar-moe-v1"}, True),                            # regex fallback
    ],
)
def test_cm_detect_moe(lightgrid_page: Page, model, expected):
    actual = lightgrid_page.evaluate(
        """(m) => window.Alpine.$data(
            document.querySelector('[x-data="campaignApp"]')
        ).cmDetectMoe(m)""",
        model,
    )
    assert actual is expected, f"cmDetectMoe({model!r}) expected {expected}"


# ===========================================================================
# gpu_count decouple — Phase 1 tests 10-15 from .claude/plans/gpu-decouple.md
# ===========================================================================


def test_gpu_count_stepper_defaults_to_tp_times_dp(lightgrid_page: Page):
    """Plan test #10 — DeepSeek preset (tp=8, dp=1) seeds gpuCount=8."""
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")
    state = _data(lightgrid_page)
    assert state["tp"] == 8
    assert state["dp"] == 1
    assert state["gpuCount"] == 8, (
        f"preset must seed gpuCount = cmGpuMin(), got {state['gpuCount']}"
    )
    # Stepper DOM reflects the bound value.
    stepper_value = lightgrid_page.locator(
        ".lg-modal:has(.lg-modal-body-grid) .lg-gpu-count .lg-stepper-value"
    )
    expect(stepper_value).to_have_value("8")


def test_gpu_count_stepper_increment_decrement(lightgrid_page: Page):
    """Plan test #11 — +/- buttons step within [min, max]; floor clamps."""
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "meta-llama/Llama-3.1-8B-Instruct")  # tp=1,dp=1
    state = _data(lightgrid_page)
    assert state["gpuCount"] == 1

    # Click "+" three times via Alpine helpers (avoids button resolution flake).
    for _ in range(3):
        lightgrid_page.evaluate(
            "() => window.Alpine.$data(document.querySelector"
            "('[x-data=\"campaignApp\"]')).cmIncGpuCount()"
        )
    assert _data(lightgrid_page)["gpuCount"] == 4

    # Click "-" twice → 2.
    for _ in range(2):
        lightgrid_page.evaluate(
            "() => window.Alpine.$data(document.querySelector"
            "('[x-data=\"campaignApp\"]')).cmDecGpuCount()"
        )
    assert _data(lightgrid_page)["gpuCount"] == 2

    # "-" until at floor (1), one more is a no-op.
    lightgrid_page.evaluate(
        "() => window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')).cmDecGpuCount()"
    )
    assert _data(lightgrid_page)["gpuCount"] == 1
    lightgrid_page.evaluate(
        "() => window.Alpine.$data(document.querySelector"
        "('[x-data=\"campaignApp\"]')).cmDecGpuCount()"
    )
    assert _data(lightgrid_page)["gpuCount"] == 1


def test_gpu_count_clamped_to_min_when_tp_dp_increased(lightgrid_page: Page):
    """Plan test #12 — $watch-driven clamp bumps gpuCount up to new floor."""
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "meta-llama/Llama-3.1-8B-Instruct")

    # Scenario A: below floor on dp bump → clamp up.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 1;
            d.cmForm.dp = 1;
            d.cmForm.gpuCount = 2;
            await d.$nextTick();
            d.cmForm.dp = 4;
            await d.$nextTick();
        }"""
    )
    assert _data(lightgrid_page)["gpuCount"] == 4

    # Scenario B: above floor preserved across tp bump.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 1;
            d.cmForm.dp = 1;
            d.cmForm.gpuCount = 8;
            await d.$nextTick();
            d.cmForm.tp = 2;
            await d.$nextTick();
        }"""
    )
    assert _data(lightgrid_page)["gpuCount"] == 8


def test_gpu_count_stepper_max_is_pod_capacity(lightgrid_page: Page):
    """Plan test #13 — + button stops at server capacity; over-cap gate flips."""
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "meta-llama/Llama-3.1-8B-Instruct")

    # Jump to the cap (server capacity=8 from fixture) and try to increment past.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.gpuCount = 8;
            await d.$nextTick();
            d.cmIncGpuCount();
            await d.$nextTick();
        }"""
    )
    state = _data(lightgrid_page)
    assert state["gpuCount"] == 8, "increment past cap must be a no-op"
    assert state["over"] is False
    assert state["canCreate"] is True

    # Force gpuCount=9 (bypass button) → over-capacity.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.gpuCount = 9;
            await d.$nextTick();
        }"""
    )
    state = _data(lightgrid_page)
    assert state["gpuCount"] == 9
    assert state["over"] is True
    assert state["canCreate"] is False


def test_create_session_payload_uses_user_selected_gpu_count(
    context, server_url
):
    """Plan test #14 — payload.gpu_count reflects the user's explicit choice."""
    pg = context.new_page()
    _install_routes(pg)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'lightgrid')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")

    captured = []

    def handler(route, request):
        url = request.url
        if request.method == "POST" and url.rstrip("/").endswith("/sessions"):
            captured.append(json.loads(request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "session_id": "test-decoupled-001",
                    "status": "active",
                    "cli_tool": "claude",
                    "repo_name": "vllm",
                    "gpu_ids": [0, 1, 2, 3, 4, 5],
                    "terminal_url": "/sessions/test-decoupled-001/terminal/",
                    "terminal_ws_url": "/sessions/test-decoupled-001/terminal/ws",
                    "created_at": "2026-05-05T00:00:00",
                    "last_accessed": 1777315832,
                    "owner_id": "mock",
                    "model_name": "deepseek-ai/DeepSeek-R1-0528",
                    "dtype": "fp8",
                }),
            )
            return
        if url.endswith("/health") or url.endswith("/health/"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_HEALTH_H200_8GPU),
            )
            return
        if "/api/hf-model-config/" in url:
            from urllib.parse import unquote
            tail = url.split("/api/hf-model-config/", 1)[1].split("?", 1)[0]
            model_id = unquote(tail)
            cfg = MOCK_HF_CONFIGS.get(model_id, {
                "model_id": model_id,
                "is_moe": False,
                "suggested_tp": 1,
                "suggested_dp": 1,
                "suggested_dtype": "bf16",
                "reason": None,
                "config": {"hidden_size": 4096, "num_hidden_layers": 32},
            })
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(cfg),
            )
            return
        if url.endswith("/sessions") or ("/sessions" in url and request.method == "GET"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"sessions": []}))
            return
        if "/api/changelog" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"version": "x", "entries": []}))
            return
        route.continue_()

    pg.route("**/api/**", handler)
    pg.route("**/sessions/**", handler)
    pg.route("**/sessions", handler)
    pg.route("**/health", handler)
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)

    _open_modal_and_wait(pg)
    _select_preset(pg, "deepseek-ai/DeepSeek-R1-0528")
    # User picks tp=2, dp=2 (min=4), gpuCount=6 — strictly > TP×DP, below server capacity.
    pg.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 2;
            d.cmForm.dp = 2;
            await d.$nextTick();
            d.cmForm.gpuCount = 6;
            await d.$nextTick();
        }"""
    )
    pg.evaluate(
        "() => window.Alpine.$data("
        "  document.querySelector('[x-data=\"campaignApp\"]')"
        ").cmCreateSession()"
    )
    pg.wait_for_timeout(500)

    assert len(captured) >= 1, "expected at least one POST /sessions"
    payload = captured[0]
    assert payload["tp_size"] == 2
    assert payload["dp_size"] == 2
    assert payload["gpu_count"] == 6, (
        f"payload.gpu_count must reflect user choice, got {payload['gpu_count']}"
    )
    pg.close()


def test_over_capacity_uses_gpu_count_not_tp_dp(lightgrid_page: Page):
    """Plan test #15 — TP×DP < cap but gpuCount > cap ⇒ over-capacity fires."""
    _open_modal_and_wait(lightgrid_page)
    _select_preset(lightgrid_page, "deepseek-ai/DeepSeek-R1-0528")

    # tp=2, dp=2 → min=4 (under cap); gpuCount=9 (over cap=8). The gate
    # must trigger off gpuCount, not the floor.
    lightgrid_page.evaluate(
        """async () => {
            const d = window.Alpine.$data(
                document.querySelector('[x-data="campaignApp"]')
            );
            d.cmForm.tp = 2;
            d.cmForm.dp = 2;
            await d.$nextTick();
            d.cmForm.gpuCount = 9;
            await d.$nextTick();
        }"""
    )
    state = _data(lightgrid_page)
    assert state["gpuMin"] == 4
    assert state["gpuCount"] == 9
    assert state["serverGpus"] == 8
    assert state["over"] is True
    assert state["canCreate"] is False

    # Red badge is on the GPU Count field.
    gpu_count_badge = lightgrid_page.locator(
        ".lg-modal:has(.lg-modal-body-grid) .lg-gpu-count .lg-linked-tag"
    )
    expect(gpu_count_badge).to_have_class("lg-linked-tag over-capacity")
