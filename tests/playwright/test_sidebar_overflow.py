# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for sidebar overflow CSS fix (Tests 7-9).

Tests that the sidebar stays within the viewport and scrolls internally
rather than causing a page-level scrollbar.

CSS fix requires:
  - body: h-screen (not min-h-screen)
  - session list div: min-h-0 (so overflow-y-auto activates in flex column)
"""

import json
import pytest
from playwright.sync_api import Page

from tests.playwright.conftest import (
    mock_api_routes,
)

pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(i: int) -> dict:
    """Generate a deterministic mock session dict."""
    return {
        "session_id": f"mock-overflow-{i:03d}",
        "status": "active" if i % 2 == 0 else "paused",
        "cli_tool": "claude",
        "repo_name": "vllm",
        "branch": "main",
        "gpu_ids": [0] if i % 2 == 0 else [],
        "requested_gpu_count": 1,
        "created_at": 1234567890.0 + i,
        "last_accessed": 1234567900.0 + i,
        "inactivity_timeout_mins": 720,
        "model_name": f"Model-{i}",
        "dtype": "fp8",
        "has_report": False,
    }


@pytest.fixture
def many_sessions_page(context, server_url) -> Page:
    """Page loaded with 20 mock sessions to trigger sidebar overflow."""
    sessions = [_make_session(i) for i in range(20)]
    sessions_response = {"sessions": sessions, "total": len(sessions)}

    pg = context.new_page()
    mock_api_routes(pg, overrides={
        "/sessions": (200, sessions_response),
        "/sessions": (200, sessions_response),
    })
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Test 7: body uses h-screen
# ---------------------------------------------------------------------------

class TestBodyHeightClass:
    """Test 7: body tag must use h-screen, not min-h-screen."""

    def _body_class_tokens(self, page: Page) -> list:
        """Return body class as a list of individual tokens (space-split)."""
        return (page.locator("body").get_attribute("class") or "").split()

    def test_body_has_h_screen_class(self, mock_page: Page):
        """body must have h-screen (exact token) to pin height to viewport.

        Uses word-level (split) check — substring check would give a false positive
        because 'min-h-screen' contains 'h-screen' as a substring.
        """
        tokens = self._body_class_tokens(mock_page)
        assert "h-screen" in tokens, (
            f"body must have 'h-screen' class token to constrain height to viewport. "
            f"Got class tokens: {tokens}"
        )

    def test_body_does_not_have_min_h_screen(self, mock_page: Page):
        """body must NOT have min-h-screen (causes page to grow beyond viewport)."""
        tokens = self._body_class_tokens(mock_page)
        assert "min-h-screen" not in tokens, (
            f"body must NOT have 'min-h-screen' — it allows the page to grow "
            f"beyond viewport height. Got class tokens: {tokens}"
        )


# ---------------------------------------------------------------------------
# Test 8: session list has min-h-0
# ---------------------------------------------------------------------------

class TestSessionListMinHeight:
    """Test 8: session list div must have min-h-0 for overflow-y-auto to activate."""

    def test_session_list_has_min_h_0(self, mock_page: Page):
        """Session list div needs min-h-0 so overflow-y-auto works in a flex column.

        In a flex column, a child with overflow-y-auto won't actually scroll
        unless it also has min-h-0 (otherwise flexbox lets it grow unconstrained).
        """
        session_list = mock_page.locator(
            "aside.session-sidebar .flex-1.overflow-y-auto"
        )
        classes = session_list.get_attribute("class") or ""
        assert "min-h-0" in classes, (
            f"Session list div must have 'min-h-0' for overflow-y-auto to activate "
            f"inside the flex column. Got classes: {classes!r}"
        )


# ---------------------------------------------------------------------------
# Test 9: overflow behavior with many sessions
# ---------------------------------------------------------------------------

class TestSidebarOverflowBehavior:
    """Test 9: 20+ sessions → session list scrolls internally, body does not overflow."""

    def test_body_has_no_page_level_scrollbar(self, many_sessions_page: Page):
        """With 20+ sessions, the body should not overflow (no page-level scrollbar).

        scrollHeight == clientHeight means the page fits within the viewport.
        """
        overflow = many_sessions_page.evaluate("""() => {
            const body = document.body;
            return body.scrollHeight - body.clientHeight;
        }""")
        assert overflow == 0, (
            f"Body overflows by {overflow}px with 20+ sessions — "
            "page-level scrollbar present. Fix: use h-screen on body."
        )

    def test_session_list_scrolls_internally(self, many_sessions_page: Page):
        """With 20+ sessions, the session list div should have internal scroll.

        scrollHeight > clientHeight means the list has overflowing content
        that can be scrolled via overflow-y-auto.
        """
        overflow = many_sessions_page.evaluate("""() => {
            const list = document.querySelector(
                "aside.session-sidebar .flex-1.overflow-y-auto"
            );
            if (!list) return -1;
            return list.scrollHeight - list.clientHeight;
        }""")
        assert overflow > 0, (
            f"Session list scrollHeight - clientHeight = {overflow} — "
            "list should overflow internally with 20 sessions. "
            "Fix: add min-h-0 to session list div."
        )

    def test_body_does_not_grow_beyond_viewport_height(self, many_sessions_page: Page):
        """Verifier edge-case: body.offsetHeight must not exceed window.innerHeight.

        The current 'test_body_has_no_page_level_scrollbar' check
        (body.scrollHeight - body.clientHeight == 0) passes vacuously because
        the body grows to match its content — scrollHeight equals clientHeight,
        but both are 2500+ px while innerHeight is 800px.

        This test catches the real bug: body grows taller than the viewport,
        causing the window to scroll rather than the sidebar list.
        Fix: replace min-h-screen with h-screen on <body>.
        """
        result = many_sessions_page.evaluate("""() => {
            return {
                bodyOffsetHeight: document.body.offsetHeight,
                windowInnerHeight: window.innerHeight,
            };
        }""")
        assert result["bodyOffsetHeight"] <= result["windowInnerHeight"], (
            f"body.offsetHeight ({result['bodyOffsetHeight']}px) exceeds "
            f"window.innerHeight ({result['windowInnerHeight']}px) — "
            "body grows beyond viewport, causing page-level scroll. "
            "Fix: use h-screen (not min-h-screen) on <body>."
        )
