# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for the Report Viewer overlay in the AMMO Sessions UI.

Covers: report button visibility, overlay open/close, markdown rendering,
TOC navigation, and keyboard dismiss.

All tests use mocked API responses for determinism.
"""

import json
import pytest
from playwright.sync_api import Page, expect

from tests.playwright.conftest import (
    MOCK_SESSION_ACTIVE,
    MOCK_SESSION_ACTIVE_WITH_REPORT,
    MOCK_SESSION_PAUSED,
    MOCK_SESSION_PAUSED_WITH_REPORT,
    MOCK_SESSION_TERMINATED,
    MOCK_REPORT_RESPONSE,
    mock_api_routes,
)

pytestmark = pytest.mark.playwright


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def report_mock_page(context, server_url):
    """Page with sessions that have reports, and report endpoint mocked."""
    pg = context.new_page()

    sessions_with_report = {
        "sessions": [
            MOCK_SESSION_ACTIVE_WITH_REPORT,
            MOCK_SESSION_PAUSED_WITH_REPORT,
            MOCK_SESSION_TERMINATED,
        ]
    }

    mock_api_routes(pg, {
        "/sessions": (200, sessions_with_report),
        "/sessions": (200, sessions_with_report),
        f"/sessions/{MOCK_SESSION_ACTIVE_WITH_REPORT['session_id']}/report": (200, MOCK_REPORT_RESPONSE),
        f"/sessions/{MOCK_SESSION_PAUSED_WITH_REPORT['session_id']}/report": (200, MOCK_REPORT_RESPONSE),
    })

    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


@pytest.fixture
def no_report_mock_page(context, server_url):
    """Page with sessions that do NOT have reports."""
    pg = context.new_page()
    mock_api_routes(pg)
    pg.goto(f"{server_url}/ui", wait_until="networkidle")
    pg.wait_for_timeout(500)
    yield pg
    pg.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportButtonVisibility:
    """Report button appears only for sessions with has_report=True."""

    def test_report_button_visible_on_active_session_with_report(self, report_mock_page):
        # Find the active session card area, look for a button with text "Report"
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        expect(report_btn).to_be_visible()

    def test_report_button_hidden_on_active_session_without_report(self, no_report_mock_page):
        # x-show keeps elements in DOM but hidden; verify they're not visible
        report_btn = no_report_mock_page.locator("button:has-text('Report')").first
        expect(report_btn).not_to_be_visible()

    def test_report_button_visible_on_paused_session_with_report(self, report_mock_page):
        # Expand the paused section first
        paused_header = report_mock_page.locator("text=Paused").first
        paused_header.click()
        report_mock_page.wait_for_timeout(300)
        # Check for Report button in the paused section
        # There should be 2 report buttons total (active + paused)
        report_btns = report_mock_page.locator("button:has-text('Report')")
        expect(report_btns).to_have_count(2)

    def test_no_report_button_on_terminated_session(self, report_mock_page):
        # Terminated sessions never show Report button
        # The terminated mock doesn't have has_report=True
        # This is validated by total count being 2 (active + paused only)
        report_btns = report_mock_page.locator("button:has-text('Report')")
        # Only active session's report button is visible by default
        # (paused section collapsed), but terminated should never have one
        # Expand all sections and count
        paused_header = report_mock_page.locator("text=Paused").first
        paused_header.click()
        report_mock_page.wait_for_timeout(300)
        terminated_header = report_mock_page.locator("text=Terminated").first
        terminated_header.click()
        report_mock_page.wait_for_timeout(300)
        # Should still be exactly 2 (active + paused), not 3
        report_btns = report_mock_page.locator("button:has-text('Report')")
        expect(report_btns).to_have_count(2)


class TestReportOverlay:
    """Full-screen report overlay opens, renders, and closes correctly."""

    def test_clicking_report_opens_overlay(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        # The overlay should be visible
        overlay = report_mock_page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_visible()

    def test_overlay_renders_markdown_headings(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        # Check that h2 headings from the markdown are rendered
        heading = report_mock_page.locator("[data-testid='report-overlay'] h2:has-text('Executive Summary')")
        expect(heading).to_be_visible()

    def test_overlay_renders_tables(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        # Table with GEMM data
        table_cell = report_mock_page.locator("[data-testid='report-overlay'] td:has-text('GEMM')")
        expect(table_cell).to_be_visible()

    def test_overlay_renders_code_blocks(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        code_block = report_mock_page.locator("[data-testid='report-overlay'] code:has-text('optimized_gemm')")
        expect(code_block).to_be_visible()

    def test_overlay_renders_blockquote_callouts(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        blockquote = report_mock_page.locator("[data-testid='report-overlay'] blockquote:has-text('Lesson 1')")
        expect(blockquote).to_be_visible()

    def test_toc_sidebar_shows_headings(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        toc = report_mock_page.locator("[data-testid='report-toc']")
        expect(toc).to_be_visible()
        # TOC should contain the h2 headings
        toc_entry = toc.locator("a:has-text('Executive Summary')")
        expect(toc_entry).to_be_visible()

    def test_toc_links_are_clickable(self, report_mock_page):
        report_btn = report_mock_page.locator("button:has-text('Report')").first
        report_btn.click()
        report_mock_page.wait_for_timeout(500)
        # Verify TOC links exist and are clickable
        toc_link = report_mock_page.locator("[data-testid='report-toc'] a:has-text('Bottleneck Analysis')")
        expect(toc_link).to_be_visible()
        # Use dispatch_event to avoid Playwright waiting for navigation
        # (Alpine's @click.prevent stops the anchor's href navigation)
        toc_link.dispatch_event("click")
        report_mock_page.wait_for_timeout(200)
        # Overlay should still be open after TOC click
        overlay = report_mock_page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_visible()


class TestReportOverlayDismiss:
    """Overlay can be closed via multiple methods."""

    def _open_report(self, page):
        """Helper to open the report overlay."""
        report_btn = page.locator("button:has-text('Report')").first
        report_btn.click()
        page.wait_for_timeout(500)
        overlay = page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_visible()

    def test_close_button_closes_overlay(self, report_mock_page):
        self._open_report(report_mock_page)
        close_btn = report_mock_page.locator("[data-testid='report-close-btn']")
        # Use dispatch_event to avoid Playwright hanging on DOM mutations
        close_btn.dispatch_event("click")
        report_mock_page.wait_for_timeout(300)
        overlay = report_mock_page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_hidden()

    def test_escape_key_closes_overlay(self, report_mock_page):
        self._open_report(report_mock_page)
        report_mock_page.keyboard.press("Escape")
        report_mock_page.wait_for_timeout(300)
        overlay = report_mock_page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_hidden()

    def test_backdrop_click_closes_overlay(self, report_mock_page):
        self._open_report(report_mock_page)
        backdrop = report_mock_page.locator("[data-testid='report-backdrop']")
        # Use dispatch_event to avoid Playwright hanging on DOM mutations
        backdrop.dispatch_event("click")
        report_mock_page.wait_for_timeout(300)
        overlay = report_mock_page.locator("[data-testid='report-overlay']")
        expect(overlay).to_be_hidden()
