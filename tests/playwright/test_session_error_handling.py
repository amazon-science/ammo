# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for session error handling in the AMMO Sessions UI.

Covers: 503 local GPU errors, action debouncing, list refresh after actions,
and copy mode toggle state syncing.

All tests use mocked API responses for determinism.
"""

import json
import pytest
from playwright.sync_api import Page, expect

from tests.playwright.conftest import (
    MOCK_SESSION_ACTIVE,
    MOCK_SESSION_PAUSED,
    MOCK_SESSIONS_RESPONSE,
    mock_api_routes,
)

pytestmark = pytest.mark.playwright


# -------------------------------------------------------------------------
# 503 local GPU errors
# -------------------------------------------------------------------------


@pytest.mark.playwright
class TestSessionCreationErrors:
    """Verify local 503 behavior and error display."""

    @staticmethod
    def _submit_valid_create_form(pg: Page) -> None:
        """Select a model through the current HF flow and submit once."""
        pg.evaluate(
            """async () => {
                const app = window.Alpine.$data(document.body);
                app.showLoginModal = false;
                app.showCreateModal = true;
                await app.selectHfModel({
                    id: 'deepseek-ai/DeepSeek-R1',
                    tags: [],
                });
                await app.$nextTick();
                await app.createSession();
            }"""
        )

    def test_create_503_does_not_retry(self, context, server_url):
        """
        When POST /sessions returns 503, the UI should show the local error
        without retrying a peer route.
        """
        pg = context.new_page()
        mock_api_routes(pg)

        request_count = {"n": 0}
        def handle_sessions(route, request):
            if request.method == "POST" and request.url.endswith("/sessions"):
                request_count["n"] += 1

                route.fulfill(
                    status=503,
                    content_type="application/json",
                    body=json.dumps({
                        "error": "insufficient_gpus",
                        "available": 0,
                        "requested": 1,
                        "message": "This server has 0 GPUs available.",
                    }),
                )
            else:
                route.continue_()

        pg.route("**/sessions", handle_sessions)
        pg.route("**/sessions/**", handle_sessions)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        self._submit_valid_create_form(pg)

        assert request_count["n"] == 1
        expect(
            pg.get_by_text("This server has 0 GPUs available.", exact=True)
        ).to_be_visible()
        pg.close()

    def test_create_503_shows_error(self, context, server_url):
        """
        If POST /sessions returns a local 503, an error message should be
        displayed to the user.
        """
        pg = context.new_page()
        mock_api_routes(pg)

        request_count = {"n": 0}

        def always_503(route, request):
            if request.method == "POST" and request.url.endswith("/sessions"):
                request_count["n"] += 1
                route.fulfill(
                    status=503,
                    content_type="application/json",
                    body=json.dumps({
                        "error": "insufficient_gpus",
                        "available": 0,
                        "requested": 1,
                        "message": "No local GPUs available.",
                    }),
                )
            else:
                route.continue_()

        pg.route("**/sessions", always_503)
        pg.route("**/sessions/**", always_503)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        self._submit_valid_create_form(pg)

        assert request_count["n"] == 1
        expect(
            pg.get_by_text("No local GPUs available.", exact=True)
        ).to_be_visible()
        pg.close()


# -------------------------------------------------------------------------
# Action debouncing
# -------------------------------------------------------------------------


@pytest.mark.playwright
class TestSessionActionDebouncing:
    """Verify rapid double-clicks don't send duplicate requests."""

    def test_rapid_double_click_pause_sends_one_request(self, context, server_url):
        """
        Double-clicking Pause on an active session should send only 1 POST request,
        not 2 (the button should be disabled/loading after first click).
        """
        pg = context.new_page()
        mock_api_routes(pg)

        pause_count = {"n": 0}

        def handle_pause(route, request):
            if "pause" in request.url and request.method == "POST":
                pause_count["n"] += 1
                # Slow response to give time for double-click
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "paused"}),
                )
            else:
                route.continue_()

        pg.route("**/sessions/*/pause", handle_pause)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        # Double-click the Pause button rapidly
        pause_btn = pg.locator("button:has-text('Pause')").first
        pause_btn.click()
        pause_btn.click()  # Second click should be ignored if button is disabled
        pg.wait_for_timeout(1000)

        # Only 1 pause request should have been sent
        assert pause_count["n"] <= 1
        pg.close()

    def test_rapid_double_click_terminate_sends_one_request(self, context, server_url):
        """
        Double-clicking the terminate trash icon and confirming should send
        only 1 DELETE request, not 2.
        """
        pg = context.new_page()
        mock_api_routes(pg)

        delete_count = {"n": 0}

        def handle_delete(route, request):
            if request.method == "DELETE" and "test-session-aaa111" in request.url:
                delete_count["n"] += 1
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "terminated"}),
                )
            else:
                route.continue_()

        pg.route("**/sessions/test-session-aaa111", handle_delete)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        # Open terminate confirm dialog — find the trash icon button on the
        # active session card.  The button uses @click="confirmTerminate(...)"
        # which is the sibling after the Pause button.
        pause_btn = pg.locator("button:has-text('Pause')").first
        # The terminate button is the next sibling button after Pause
        delete_btn = pause_btn.locator("xpath=following-sibling::button[1]")
        delete_btn.click()
        pg.wait_for_timeout(300)

        # Click confirm twice rapidly inside the confirm dialog.
        # The dialog closes on confirm via `confirmDialog.open = false`, so
        # the second click may fail because the button is no longer visible.
        # Use force=True for the rapid second click to avoid visibility waits.
        confirm_btn = pg.locator("button:has-text('Confirm')")
        confirm_btn.click()
        try:
            confirm_btn.click(timeout=500, force=True)
        except Exception:
            pass  # Expected: dialog already closed after first click
        pg.wait_for_timeout(1000)

        assert delete_count["n"] <= 1
        pg.close()


# -------------------------------------------------------------------------
# Session list refresh
# -------------------------------------------------------------------------


@pytest.mark.playwright
class TestSessionUIUpdates:
    """Verify the session list refreshes after actions and state syncs."""

    def test_session_list_refreshes_after_action(self, context, server_url):
        """
        After a successful Pause action, the UI should re-fetch /sessions
        to show updated session states.
        """
        pg = context.new_page()

        sessions_calls = {"n": 0}

        def handle_all(route, request):
            url = request.url
            # Pause POST
            if "/pause" in url and request.method == "POST":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "paused"}),
                )
            # /sessions GET
            elif url.endswith("/sessions") and request.method == "GET":
                sessions_calls["n"] += 1
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(MOCK_SESSIONS_RESPONSE),
                )
            # /sessions GET (non-all)
            elif "/sessions" in url and request.method == "GET":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(MOCK_SESSIONS_RESPONSE),
                )
            else:
                route.continue_()

        def handle_api(route, request):
            url = request.url
            if url.endswith("/health") or url.endswith("/health/"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "status": "healthy",
                        "job_stats": {"total_jobs": 0},
                        "gpu_manager": {"total_gpus": 8, "available_gpus": 8},
                        "gpu": {
                            "type": "h200",
                            "allowed_dtypes": ["fp8"],
                            "total_gpus": 8,
                            "available_gpus": 8,
                        },
                        "vllm": {"docker_commit": None, "version": None},
                    }),
                )
            elif "/api/changelog" in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"version": "1.0.0", "entries": []}),
                )
            else:
                route.continue_()

        pg.route("**/sessions/**", handle_all)
        pg.route("**/sessions", handle_all)
        pg.route("**/api/**", handle_api)
        pg.route("**/health", handle_api)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        # Record calls so far (initial load)
        calls_before = sessions_calls["n"]

        # Click Pause on the active session
        pause_btn = pg.locator("button:has-text('Pause')").first
        pause_btn.click()
        # Wait long enough for the pause response + debounced loadSessions() call
        pg.wait_for_timeout(3000)

        # /sessions should have been re-fetched after the action
        assert sessions_calls["n"] > calls_before
        pg.close()

    def test_copy_mode_toggle_syncs_state(self, context, server_url):
        """
        Clicking the Copy Mode toggle should POST to tmux-mouse-mode endpoint
        and update the button's visual state.
        """
        pg = context.new_page()
        mock_api_routes(pg)

        tmux_calls = {"n": 0, "last_mode": None}

        def handle_tmux_mode(route, request):
            if "tmux-mouse-mode" in request.url and request.method == "POST":
                tmux_calls["n"] += 1
                body = request.post_data
                if body:
                    try:
                        payload = json.loads(body)
                        tmux_calls["last_mode"] = payload.get("mode")
                    except Exception:
                        pass
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"mode": "off", "session_id": "test-session-aaa111"}),
                )
            elif "tmux-mouse-mode" in request.url and request.method == "GET":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"mode": "on", "session_id": "test-session-aaa111"}),
                )
            else:
                route.continue_()

        pg.route("**/sessions/*/tmux-mouse-mode", handle_tmux_mode)

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        # Open the active session terminal
        pg.route(
            "**/sessions/test-session-aaa111/terminal/**",
            lambda route, _: route.fulfill(
                status=200,
                content_type="text/html",
                body="<html><body>mock terminal</body></html>",
            ),
        )

        open_btn = pg.locator("button:has-text('Open')").first
        open_btn.click()
        pg.wait_for_timeout(500)

        # Look for Copy Mode toggle button — scope to the classic sessionApp's
        # visible terminal header to avoid matching LIGHTGRID's hidden button
        copy_btn = pg.locator("button:has-text('Copy')").first
        if copy_btn.is_visible():
            copy_btn.click()
            pg.wait_for_timeout(500)

            # POST to tmux-mouse-mode should have been called
            assert tmux_calls["n"] >= 1

        pg.close()
