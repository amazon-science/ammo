# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for terminal interaction and Copy Mode toggle.

Covers: terminal iframe loading, Copy Mode button visibility,
Copy Mode toggle API calls, and Copy Mode banner display.
"""

import json
import pytest
from playwright.sync_api import Page, expect

from tests.playwright.conftest import (
    MOCK_SESSION_ACTIVE,
    MOCK_SESSIONS_RESPONSE,
    mock_api_routes,
)

pytestmark = pytest.mark.playwright


def _open_terminal(page: Page):
    """Helper: mock terminal route and click Open on the first active session."""
    page.route(
        "**/sessions/test-session-aaa111/terminal/**",
        lambda route, _: route.fulfill(
            status=200,
            content_type="text/html",
            body="<html><body>mock terminal</body></html>",
        ),
    )
    # Also mock the tmux mouse mode GET (sync on open)
    # Use route.fallback() for non-GET so POST requests fall through to test-specific routes
    page.route(
        "**/sessions/test-session-aaa111/tmux-mouse-mode",
        lambda route, req: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"mouse_mode": "on"}),
        ) if req.method == "GET" else route.fallback(),
    )
    page.locator("button:has-text('Open')").first.click()
    page.wait_for_timeout(500)


# -------------------------------------------------------------------------
# Terminal panel
# -------------------------------------------------------------------------


class TestTerminalPanel:
    """After clicking Open, a terminal panel appears in the main area."""

    def test_terminal_iframe_loads(self, mock_page: Page):
        _open_terminal(mock_page)
        iframe = mock_page.locator("iframe").first
        expect(iframe).to_be_visible()
        src = iframe.get_attribute("src")
        assert "/sessions/test-session-aaa111/terminal/" in src

    def test_terminal_header_shows_model(self, mock_page: Page):
        _open_terminal(mock_page)
        # The terminal header bar should show the model display
        header_text = mock_page.locator(
            "text=DeepSeek-R1-0528 (fp8)"
        )
        # There will be at least one instance (sidebar + terminal header)
        expect(header_text.first).to_be_visible()

    def test_close_terminal_button(self, mock_page: Page):
        _open_terminal(mock_page)
        # Close button is the X icon in the terminal header
        close_btn = mock_page.locator("button[title='Close terminal']")
        expect(close_btn).to_be_visible()

        close_btn.click()
        mock_page.wait_for_timeout(300)

        # Terminal iframe should be gone
        expect(mock_page.locator("iframe")).not_to_be_visible()

    def test_fullscreen_toggle(self, mock_page: Page):
        _open_terminal(mock_page)
        # The fullscreen button uses a dynamic Alpine.js :title binding;
        # wait for Alpine to evaluate it before querying by attribute.
        mock_page.wait_for_timeout(300)
        fs_btn = mock_page.locator("button[title='Fullscreen']")
        if fs_btn.count() == 0:
            # Fallback: find the fullscreen button by its sibling (close button)
            # It's the button right before button[title='Close terminal']
            fs_btn = mock_page.locator(
                "button:has(svg path[d*='M4 8V4'])"
            )
        expect(fs_btn.first).to_be_visible()


# -------------------------------------------------------------------------
# Copy Mode toggle
# -------------------------------------------------------------------------


class TestCopyMode:
    """Copy Mode button toggles tmux mouse mode via the API."""

    def test_copy_button_visible_in_terminal_header(self, mock_page: Page):
        _open_terminal(mock_page)
        copy_btn = mock_page.locator("button:has-text('Copy')").first
        expect(copy_btn).to_be_visible()

    def test_toggle_copy_mode_on_calls_api(self, mock_page: Page):
        """Clicking Copy sends POST to tmux-mouse-mode with mode=off."""
        captured = []

        def handle_post(route, request):
            if request.method == "POST" and "tmux-mouse-mode" in request.url:
                body = json.loads(request.post_data)
                captured.append(body)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "ok", "mouse_mode": "off"}),
                )
            else:
                route.continue_()

        mock_page.route("**/sessions/*/tmux-mouse-mode", handle_post)
        _open_terminal(mock_page)

        copy_btn = mock_page.locator("button:has-text('Copy')").first
        copy_btn.click()
        mock_page.wait_for_timeout(500)

        assert len(captured) >= 1
        # When copy mode is OFF (mouse is ON), clicking sends mode=off (turn mouse off -> copy mode on)
        assert captured[0]["mode"] == "off"

    def test_copy_mode_banner_appears(self, mock_page: Page):
        """When copy mode is toggled on, a warning banner appears."""
        # Mock the POST to return mouse_mode=off (copy mode ON)
        mock_page.route(
            "**/sessions/*/tmux-mouse-mode",
            lambda route, req: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "ok", "mouse_mode": "off"}),
            ) if req.method == "POST" else route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"mouse_mode": "on"}),
            ),
        )
        _open_terminal(mock_page)

        copy_btn = mock_page.locator("button:has-text('Copy')").first
        copy_btn.click()
        mock_page.wait_for_timeout(500)

        # Banner text should appear
        banner = mock_page.locator("text=Copy Mode:")
        expect(banner.first).to_be_visible()

    def test_copy_mode_button_changes_style(self, mock_page: Page):
        """When copy mode is on, the button should have the warning style class."""
        mock_page.route(
            "**/sessions/*/tmux-mouse-mode",
            lambda route, req: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "ok", "mouse_mode": "off"}),
            ) if req.method == "POST" else route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"mouse_mode": "on"}),
            ),
        )
        _open_terminal(mock_page)

        copy_btn = mock_page.locator("button:has-text('Copy')").first
        copy_btn.click()
        mock_page.wait_for_timeout(500)

        # After toggling, button text changes to "Copy Mode"
        copy_mode_btn = mock_page.locator("button:has-text('Copy Mode')").first
        expect(copy_mode_btn).to_be_visible()

    def test_toggle_copy_mode_off_restores_mouse(self, mock_page: Page):
        """Toggle copy mode on, then off again -> sends mode=on."""
        call_count = {"n": 0}

        def handle(route, request):
            if request.method == "POST" and "tmux-mouse-mode" in request.url:
                call_count["n"] += 1
                # First call: turn mouse off (copy on)
                # Second call: turn mouse on (copy off)
                if call_count["n"] == 1:
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"status": "ok", "mouse_mode": "off"}),
                    )
                else:
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"status": "ok", "mouse_mode": "on"}),
                    )
            elif request.method == "GET" and "tmux-mouse-mode" in request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"mouse_mode": "on"}),
                )
            else:
                route.continue_()

        mock_page.route("**/sessions/*/tmux-mouse-mode", handle)
        _open_terminal(mock_page)

        # First toggle: copy mode ON
        mock_page.locator("button:has-text('Copy')").first.click()
        mock_page.wait_for_timeout(500)

        # Second toggle: copy mode OFF
        mock_page.locator("button:has-text('Copy Mode')").first.click()
        mock_page.wait_for_timeout(500)

        assert call_count["n"] == 2


# -------------------------------------------------------------------------
# Two terminals side-by-side
# -------------------------------------------------------------------------


class TestSplitView:
    """Up to 2 terminals can be open simultaneously."""

    def test_two_terminals_side_by_side(self, context, server_url):
        """Opening two active sessions shows split view."""
        second_session = {
            **MOCK_SESSION_ACTIVE,
            "session_id": "test-session-ddd444",
            "model_name": "Qwen/Qwen3-Coder-480B",
            "dtype": "bf16",
        }
        multi_response = {
            "sessions": [
                MOCK_SESSION_ACTIVE,
                second_session,
            ]
        }

        pg = context.new_page()
        mock_api_routes(pg, {
            "/sessions": (200, multi_response),
        })

        # Mock terminal routes for both sessions
        pg.route(
            "**/sessions/test-session-aaa111/terminal/**",
            lambda route, _: route.fulfill(
                status=200,
                content_type="text/html",
                body="<html><body>terminal 1</body></html>",
            ),
        )
        pg.route(
            "**/sessions/test-session-ddd444/terminal/**",
            lambda route, _: route.fulfill(
                status=200,
                content_type="text/html",
                body="<html><body>terminal 2</body></html>",
            ),
        )
        pg.route(
            "**/sessions/*/tmux-mouse-mode",
            lambda route, _: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"mouse_mode": "on"}),
            ),
        )

        # Dismiss welcome popup and tours via localStorage
        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.evaluate("localStorage.setItem('ammo_ui_theme', 'classic')")
        pg.evaluate("localStorage.setItem('ammo_lg_tour_completed', 'true')")
        pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
        pg.reload(wait_until="networkidle")
        pg.wait_for_timeout(500)

        # Open first terminal
        open_btns = pg.locator("button:has-text('Open')")
        open_btns.nth(0).click()
        pg.wait_for_timeout(300)

        # Open second terminal
        open_btns.nth(1).click()
        pg.wait_for_timeout(300)

        # Both iframes should be visible
        iframes = pg.locator("iframe")
        assert iframes.count() == 2

        pg.close()
