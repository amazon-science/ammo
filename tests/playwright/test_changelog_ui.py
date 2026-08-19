# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for the Changelog UI in the AMMO Sessions UI.

Covers: version badge visibility, changelog overlay open/close, TOC sidebar,
notification dot behaviour (seen/unseen), and per-session version chips.

All tests use mocked API responses for determinism (no live server required).
The /api/changelog mock and ammo_version session fields are provided by
conftest.mock_api_routes() after the Task #2 update.
"""

import re
import pytest
from playwright.sync_api import Page, expect

from tests.playwright.conftest import mock_api_routes

pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seen_version_page(context, server_url) -> Page:
    """Page where localStorage already has current version — no notification dot."""
    pg = context.new_page()
    mock_api_routes(pg)
    # Set version as seen BEFORE navigating so initApp() finds it immediately
    pg.goto(f"{server_url}/ui", wait_until="domcontentloaded")
    pg.evaluate("localStorage.setItem('ammo_last_seen_version', '1.2.0')")
    pg.evaluate("localStorage.setItem('ammo_tour_completed', 'true')")
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_changelog(page: Page) -> None:
    """Click the version badge to open the changelog overlay."""
    badge = page.locator("[data-testid='version-badge']")
    badge.click()
    page.wait_for_timeout(400)
    expect(page.locator("[data-testid='changelog-overlay']")).to_be_visible()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVersionBadge:
    """Version badge in the header shows current version."""

    def test_version_badge_visible(self, mock_page):
        """Version badge is visible and shows the v1.2.0 text."""
        badge = mock_page.locator("[data-testid='version-badge']")
        expect(badge).to_be_visible()
        badge_text = badge.inner_text()
        assert re.match(r"v\d+\.\d+\.\d+", badge_text.strip()), (
            f"Badge text '{badge_text}' does not match vN.N.N format"
        )

    def test_version_badge_opens_changelog(self, mock_page):
        """Clicking the version badge opens the changelog overlay with version headings."""
        _open_changelog(mock_page)
        overlay = mock_page.locator("[data-testid='changelog-overlay']")
        expect(overlay).to_be_visible()
        # Overlay should contain version headings from the mock data
        expect(overlay.locator("h2:has-text('1.2.0')")).to_be_visible()


class TestChangelogOverlay:
    """Changelog overlay opens, renders entries, and closes correctly."""

    def test_changelog_overlay_initially_hidden(self, mock_page):
        """Changelog overlay is NOT visible on initial page load (changelogOverlay.open starts false)."""
        overlay = mock_page.locator("[data-testid='changelog-overlay']")
        expect(overlay).to_be_hidden()

    def test_changelog_overlay_closes_on_escape(self, mock_page):
        """Press Escape key closes the changelog overlay."""
        _open_changelog(mock_page)
        mock_page.keyboard.press("Escape")
        mock_page.wait_for_timeout(300)
        overlay = mock_page.locator("[data-testid='changelog-overlay']")
        expect(overlay).to_be_hidden()

    def test_changelog_overlay_closes_on_backdrop_click(self, mock_page):
        """Clicking the backdrop closes the changelog overlay."""
        _open_changelog(mock_page)
        backdrop = mock_page.locator("[data-testid='changelog-backdrop']")
        backdrop.dispatch_event("click")
        mock_page.wait_for_timeout(300)
        overlay = mock_page.locator("[data-testid='changelog-overlay']")
        expect(overlay).to_be_hidden()

    def test_changelog_toc_sidebar(self, mock_page):
        """Changelog overlay has a TOC sidebar with version links for all entries."""
        _open_changelog(mock_page)
        toc = mock_page.locator("[data-testid='changelog-toc']")
        expect(toc).to_be_visible()
        expect(toc.locator("a:has-text('1.2.0')")).to_be_visible()
        expect(toc.locator("a:has-text('1.1.0')")).to_be_visible()
        expect(toc.locator("a:has-text('1.0.0')")).to_be_visible()


class TestNotificationDot:
    """Green notification dot appears when a new version has not been seen."""

    def test_unseen_version_notification_dot(self, mock_page):
        """Notification dot is visible when version has not been seen (fresh context = empty localStorage)."""
        dot = mock_page.locator("[data-testid='version-notification-dot']")
        expect(dot).to_be_visible()

        # Clicking the badge should mark the version as seen → dot disappears
        badge = mock_page.locator("[data-testid='version-badge']")
        badge.click()
        mock_page.wait_for_timeout(300)
        expect(dot).to_be_hidden()

    def test_seen_version_no_notification_dot(self, seen_version_page):
        """Notification dot is NOT visible when current version is already in localStorage."""
        dot = seen_version_page.locator("[data-testid='version-notification-dot']")
        expect(dot).to_be_hidden()


class TestSessionVersionChip:
    """Per-session version chip shows on session cards with ammo_version set."""

    def test_session_card_version_chip(self, mock_page):
        """Active session card shows the version chip with the session's ammo_version."""
        chip = mock_page.locator("[data-testid='session-version-chip']").first
        expect(chip).to_be_visible()
        chip_text = chip.inner_text()
        assert "1.2.0" in chip_text, (
            f"Expected '1.2.0' in chip text, got '{chip_text}'"
        )
