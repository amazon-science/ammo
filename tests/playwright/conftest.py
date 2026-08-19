# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright test fixtures for AMMO Sessions UI.

Provides browser setup, page fixtures, and API mocking helpers
for headless Chrome tests against the AMMO Sessions frontend.
"""

import os
import json
import pytest
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "playwright: Playwright browser tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def server_url():
    """Base URL of the running AMMO server (configurable via env var)."""
    return os.getenv("AMMO_SERVER_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def playwright_instance():
    """Session-scoped Playwright instance to avoid repeated startups."""
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def browser(playwright_instance) -> Browser:
    """Session-scoped headless Chromium browser."""
    browser = playwright_instance.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture
def context(browser) -> BrowserContext:
    """Fresh browser context per test (isolated cookies / storage)."""
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 800},
        ignore_https_errors=True,
    )
    yield ctx
    ctx.close()


@pytest.fixture
def page(context, server_url) -> Page:
    """New page pointed at the AMMO UI with Alpine.js fully initialised."""
    pg = context.new_page()
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    # Dismiss welcome popup and all tours so they don't block interactions
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")
    pg.reload(wait_until="networkidle")
    # Wait for Alpine.js to finish initialising the app
    pg.wait_for_function("() => !!document.querySelector('[x-data]')._x_dataStack")
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_SESSION_ACTIVE = {
    "session_id": "test-session-aaa111",
    "status": "active",
    "cli_tool": "claude",
    "repo_name": "vllm",
    "gpu_ids": [0],
    "terminal_url": "/sessions/test-session-aaa111/terminal/",
    "terminal_ws_url": "/sessions/test-session-aaa111/terminal/ws",
    "created_at": "2025-06-01T00:00:00",
    "last_accessed": 1748736000,
    "owner_id": "mock-owner",
    "model_name": "deepseek-ai/DeepSeek-R1-0528",
    "dtype": "fp8",
    "ammo_version": "1.2.0",
}

MOCK_SESSION_PAUSED = {
    **MOCK_SESSION_ACTIVE,
    "session_id": "test-session-bbb222",
    "status": "paused",
    "gpu_ids": [],
}

MOCK_SESSION_TERMINATED = {
    **MOCK_SESSION_ACTIVE,
    "session_id": "test-session-ccc333",
    "status": "terminated",
    "gpu_ids": [],
}

# The static /api/supported-models + /api/moe-models endpoints were removed.
# GPU info + vllm metadata now ship on /health; per-model TP/DP/dtype comes from
# /api/hf-model-config/{id} on demand.
MOCK_HEALTH_RESPONSE = {
    "status": "healthy",
    "job_stats": {"total_jobs": 0},
    "gpu_manager": {"total_gpus": 8, "available_gpus": 8},
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
}

MOCK_HF_MODEL_CONFIG_RESPONSE = {
    "model_id": "deepseek-ai/DeepSeek-R1",
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
}

MOCK_SESSION_ACTIVE_WITH_REPORT = {
    **MOCK_SESSION_ACTIVE,
    "has_report": True,
}

MOCK_SESSION_PAUSED_WITH_REPORT = {
    **MOCK_SESSION_PAUSED,
    "has_report": True,
}

MOCK_REPORT_MARKDOWN = """# Optimization Report

## Executive Summary

Target: Qwen3-30B-A3B on NVIDIA L40S with bf16, TP=1.

We profiled the model and identified GEMM operations as the primary bottleneck.

## Bottleneck Analysis

| Kernel | Time % | BW Util |
|--------|--------|---------|
| GEMM   | 85%    | 62%     |
| Attn   | 10%    | 78%     |

## Implementation & Results

```python
def optimized_gemm():
    pass
```

> **Lesson 1: Profile First**
>
> Always profile before optimizing.
"""

MOCK_REPORT_RESPONSE = {
    "session_id": "test-session-aaa111",
    "markdown": MOCK_REPORT_MARKDOWN,
    "report_path": "kernel_opt_artifacts/auto_qwen3_l40s/REPORT.md",
}

MOCK_SESSIONS_RESPONSE = {
    "sessions": [
        MOCK_SESSION_ACTIVE,
        MOCK_SESSION_PAUSED,
        MOCK_SESSION_TERMINATED,
    ]
}


def mock_api_routes(page: Page, overrides: dict | None = None):
    """Intercept API calls and return deterministic mock data.

    ``overrides`` is a dict mapping URL suffixes to (status, body) tuples.
    For example::

        mock_api_routes(page, {
            "/sessions": (503, {"error": "insufficient_gpus", ...}),
        })

    To mock 401 responses (e.g., for auth testing), use an override::

        mock_api_routes(page, {
            "/sessions": (401, {"detail": "Invalid or missing API key"}),
        })
    """
    overrides = overrides or {}

    def _handle(route, request):
        url = request.url

        # Check overrides first
        for suffix, (status, body) in overrides.items():
            if url.endswith(suffix):
                route.fulfill(
                    status=status,
                    content_type="application/json",
                    body=json.dumps(body),
                )
                return

        # Default mocks
        if url.endswith("/health") or url.endswith("/health/"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_HEALTH_RESPONSE),
            )
        elif "/api/hf-model-config/" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_HF_MODEL_CONFIG_RESPONSE),
            )
        elif url.endswith("/sessions"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_SESSIONS_RESPONSE),
            )
        elif "/sessions" in url and request.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_SESSIONS_RESPONSE),
            )
        elif "/api/changelog" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "version": "1.2.0",
                    "entries": [
                        {"version": "1.2.0", "date": "2026-04-03", "changes": ["Accuracy failure persistence", "Transcript monitor dual role"]},
                        {"version": "1.1.0", "date": "2026-04-02", "changes": ["Redesign Gate 5.1b"]},
                        {"version": "1.0.0", "date": "2026-03-31", "changes": ["Initial versioned release"]},
                    ],
                }),
            )
        else:
            route.continue_()

    page.route("**/api/**", _handle)
    page.route("**/sessions/**", _handle)
    page.route("**/sessions", _handle)
    # /health is not under /api/, but the frontend now reads gpu + vllm
    # metadata from it, so intercept that too.
    page.route("**/health", _handle)


@pytest.fixture
def mock_page(context, server_url) -> Page:
    """Page with ALL API routes mocked — fully deterministic, no server needed."""
    pg = context.new_page()

    # Set up mocks *before* navigation so the initial fetches are intercepted
    mock_api_routes(pg)

    # First load to establish origin, then dismiss tours and welcome popup via localStorage
    pg.goto(f"{server_url}/ui", wait_until="domcontentloaded")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")

    # Reload so Alpine picks up the tour-dismissed flag from localStorage
    pg.reload(wait_until="networkidle")
    # Give Alpine a moment to hydrate the mocked data
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


MOCK_AUTH_API_KEY = "test-playwright-api-key-12345"


@pytest.fixture
def authenticated_mock_page(context, server_url) -> Page:
    """Page with API routes mocked AND API key pre-populated in localStorage.

    Use this fixture when testing authenticated flows where the login modal
    should NOT appear. The API key is stored in localStorage before navigation
    so the UI picks it up on boot.
    """
    pg = context.new_page()

    # Navigate first to set the origin (localStorage requires same-origin)
    pg.goto(f"{server_url}/ui", wait_until="commit")
    pg.evaluate(
        f"() => localStorage.setItem('ammo_api_key', '{MOCK_AUTH_API_KEY}')"
    )
    # Dismiss welcome popup and all tours so they don't block interactions
    pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
    pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l1_deep_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l2_tour_completed', 'true')")
    pg.evaluate("localStorage.setItem('ammo_lg_l3_tour_completed', 'true')")

    # Now set up mocks that verify auth and reload the page
    mock_api_routes(pg)

    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()
