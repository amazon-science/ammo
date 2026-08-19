# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright visual tests for the L3 Artifact Viewer production port (C1–C9).

These tests run against a LIVE server + campaign session. They are driven by
the AMMO_SERVER_URL env var (default `http://localhost:18000`, matching
the Mac → B200 pod port-forward) and the TARGET_SESSION_ID env var (default
`998a3fe8-6fec-4867-b241-f650824aba02`).

Every test attaches a console-error collector and asserts zero console errors
after the artifact loads — per the TDD plan, console cleanliness is a gate.

Run (from Mac repo root):

    AMMO_SERVER_URL=http://localhost:18000 \\
        pytest tests/playwright/test_l3_artifact_viewer_visual.py -v --timeout=120

Per the L3 viewer plan:
  - C5.prep is a fail-loud gate: HEAD route must return 200 (405 == not deployed).
  - Screenshots land in ./screenshots/, min 2 KB each.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


TARGET_SESSION_ID = os.getenv(
    "TARGET_SESSION_ID", "998a3fe8-6fec-4867-b241-f650824aba02"
)
SCREENSHOT_DIR = Path(__file__).parent.parent.parent / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SCREENSHOT_BYTES = 2 * 1024  # 2 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goto_session_l3(page, server_url: str) -> None:
    """Navigate to the L3 artifact view of the target session."""
    url = f"{server_url}/ui#session/{TARGET_SESSION_ID}"
    page.goto(url, wait_until="networkidle")
    # Wait for Alpine.js to hydrate and campaignApp to boot.
    page.wait_for_function(
        "() => document.querySelector('[x-data]') && "
        "     document.querySelector('[x-data]')._x_dataStack"
    )
    # Wait for L3 rendering — if the hash bypasses L0/L1/L2 the viewer
    # should appear within ~10s.
    try:
        page.wait_for_selector(
            "text=/ARTIFACTS/i", timeout=10_000,
        )
    except Exception:
        pytest.skip(
            f"L3 viewer did not mount within 10s at {url}. "
            "Check that the session exists and the port-forward is live."
        )


def _console_errors(page) -> list[str]:
    """Attach a console-error collector to the page; return a shared list."""
    errors: list[str] = []

    def _collect(msg):
        if msg.type == "error":
            errors.append(msg.text)

    page.on("console", _collect)
    # Surface uncaught page errors too
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    return errors


def _save_screenshot(page, name: str) -> Path:
    out = SCREENSHOT_DIR / f"{name}.png"
    page.screenshot(path=str(out), full_page=False)
    assert out.stat().st_size >= MIN_SCREENSHOT_BYTES, (
        f"Screenshot {out} is {out.stat().st_size}B (< {MIN_SCREENSHOT_BYTES}B). "
        "Likely the page did not render."
    )
    return out


def _pick_tab_by_suffix(page, suffix: str):
    """Click an artifact tab whose path ends with the given suffix."""
    locator = page.locator(
        f".lg-artifact-tab-nested:has-text('{suffix}')"
    ).first
    locator.wait_for(state="visible", timeout=10_000)
    locator.click()
    # Let the artifact load — the meta-bar renderer label updates synchronously
    # once the fetch completes.
    page.wait_for_timeout(600)


# ---------------------------------------------------------------------------
# C5.prep — HEAD route fail-loud gate
# ---------------------------------------------------------------------------

class TestHeadRouteGateC5Prep:
    def test_head_route_returns_200(self, page, server_url):
        """C5.prep — HEAD /artifacts/state.json must return 200.

        405 means the backend HEAD route is not deployed. Abort with a clear
        message so the rest of the suite doesn't cascade-fail.
        """
        url = (
            f"{server_url}/api/campaigns/{TARGET_SESSION_ID}"
            f"/artifacts/state.json"
        )
        page.goto(server_url + "/ui", wait_until="domcontentloaded")
        status = page.evaluate(
            "(u) => fetch(u, {method: 'HEAD'}).then(r => r.status).catch(() => 0)",
            url,
        )
        assert status != 405, (
            f"HEAD {url} returned 405 — backend HEAD route not deployed. "
            "Abort: run the deploy recipe in .claude/plans/l3-artifact-viewer.md."
        )
        assert status == 200, f"HEAD {url} returned {status}, expected 200"


# ---------------------------------------------------------------------------
# C1 — visible-row parity
# ---------------------------------------------------------------------------

class TestArtifactRowParityC1:
    def test_sidebar_rows_match_flat_browser_model(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        sidebar_count = page.locator(".lg-artifact-browser-row").count()
        row_count = page.evaluate("""
            () => {
                const root = document.querySelector('[x-data]');
                if (!root || !root._x_dataStack) return -1;
                const app = root._x_dataStack[0];
                return app.l3ArtifactRows ? app.l3ArtifactRows().length : -1;
            }
        """)
        assert row_count >= 0, "Alpine component did not mount"
        assert sidebar_count == row_count, (
            f"row count parity broken: sidebar={sidebar_count} vs model={row_count}"
        )
        visible_loading = page.locator(
            ".lg-artifact-browser-status:visible", has_text="LOADING"
        ).count()
        assert visible_loading == 0, (
            f"artifact browser retained {visible_loading} stale LOADING badge(s)"
        )
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C2 — markdown tab
# ---------------------------------------------------------------------------

class TestMarkdownPaneC2:
    def test_markdown_rendered_with_headers_code_and_table(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        _pick_tab_by_suffix(page, ".md")
        # h1
        assert page.locator(".l3-prose h1").count() >= 1, "markdown missing <h1>"
        # syntax-highlighted <pre><code class="hljs">
        assert page.locator(".l3-prose pre code.hljs").count() >= 1, (
            "markdown code blocks not syntax-highlighted"
        )
        # at least one <table> from the markdown
        assert page.locator(".l3-prose table").count() >= 1, (
            "markdown table missing"
        )
        _save_screenshot(page, "l3-md")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C3 — JSON tree expand / collapse
# ---------------------------------------------------------------------------

class TestJsonTreeToggleC3:
    def test_json_tree_expand_collapse(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        _pick_tab_by_suffix(page, ".json")
        tree = page.locator(".json-tree").first
        tree.wait_for(state="visible", timeout=10_000)
        _save_screenshot(page, "l3-json-collapsed")

        # Expand the first expandable line
        first_expandable = page.locator(".jt-expandable").first
        first_expandable.wait_for(state="visible", timeout=5_000)
        # Count children before / after click
        before_hidden = page.locator(".jt-children.hidden").count()
        first_expandable.click()
        page.wait_for_timeout(120)
        after_hidden = page.locator(".jt-children.hidden").count()
        assert before_hidden != after_hidden, (
            f".jt-children.hidden did not toggle (before={before_hidden} "
            f"after={after_hidden})"
        )
        _save_screenshot(page, "l3-json-expanded")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C4 — code tab: hljs cyan keyword + gutter count
# ---------------------------------------------------------------------------

class TestCodePaneC4:
    def test_keyword_cyan_and_gutter_count(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        _pick_tab_by_suffix(page, ".py")
        kw = page.locator(".hljs-keyword").first
        kw.wait_for(state="visible", timeout=10_000)
        color = kw.evaluate("el => getComputedStyle(el).color")
        m = re.match(r"rgba?\(\s*(\d+)\D+(\d+)\D+(\d+)", color)
        assert m, f"could not parse hljs-keyword color: {color}"
        r, g, b = map(int, m.groups())
        # LIGHTGRID cyan ≈ #00f3ff (0, 243, 255) ± 5
        assert r <= 5 and abs(g - 243) <= 5 and abs(b - 255) <= 5, (
            f"hljs-keyword color {color} is not LIGHTGRID cyan"
        )

        # Gutter line-count must match file line-count.
        gutter_lines = page.locator(".code-gutter > div").count()
        file_lines = page.evaluate("""
            () => {
                const root = document.querySelector('[x-data]');
                const app  = root._x_dataStack[0];
                const s = app.activeSection;
                if (!s || typeof s.content !== 'string') return -1;
                return s.content.split('\\n').length;
            }
        """)
        assert file_lines > 0, "active section has no string content"
        assert gutter_lines == file_lines, (
            f"gutter count {gutter_lines} ≠ file lines {file_lines}"
        )
        _save_screenshot(page, "l3-code")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C5a — image inline (<5 MB)
# ---------------------------------------------------------------------------

class TestImageInlineC5a:
    def test_inline_image_is_rendered(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        # Pick any image tab — try png/jpg/svg in order.
        tab = None
        for suffix in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
            cand = page.locator(
                f".lg-artifact-tab-nested:has-text('{suffix}')"
            )
            if cand.count() > 0:
                tab = cand.first
                break
        if tab is None:
            pytest.skip("No image artifact present in target session")
        tab.click()
        page.wait_for_timeout(800)
        frame = page.locator(".image-frame img").first
        frame.wait_for(state="visible", timeout=10_000)
        meta = frame.evaluate(
            "el => ({ nw: el.naturalWidth, complete: el.complete })"
        )
        assert meta["complete"] is True
        assert meta["nw"] > 0, "inline image naturalWidth==0 (failed to load)"
        _save_screenshot(page, "l3-image")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C5b — oversize image falls back to download card
# ---------------------------------------------------------------------------

class TestImageOversizeFallbackC5b:
    def test_oversize_png_falls_back_to_download_card(self, page, server_url):
        errors = _console_errors(page)
        # Mock HEAD for any .png to return a 6 MB content-length.
        def _route(route, request):
            if request.method == "HEAD" and request.url.endswith(".png"):
                route.fulfill(
                    status=200,
                    headers={
                        "Content-Length": "6000000",
                        "Content-Type":   "image/png",
                    },
                    body="",
                )
                return
            route.continue_()

        page.route("**/api/campaigns/**/artifacts/**.png", _route)
        _goto_session_l3(page, server_url)
        tab = page.locator(".lg-artifact-tab-nested:has-text('.png')").first
        if tab.count() == 0:
            pytest.skip("No .png artifact in target session")
        tab.click()
        page.wait_for_timeout(800)
        # Download card must be visible; inline image must NOT be.
        assert page.locator(".l3-binary-card").count() >= 1, (
            "oversize image did not fall back to download card"
        )
        assert page.locator(".image-frame img").count() == 0, (
            "inline <img> rendered despite oversize gate"
        )
        _save_screenshot(page, "l3-image-oversize")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C6 — plaintext / .log
# ---------------------------------------------------------------------------

class TestPlaintextPaneC6:
    def test_log_view_populated_and_escaped(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        # Try .log then .txt
        tab = None
        for suffix in (".log", ".txt"):
            cand = page.locator(
                f".lg-artifact-tab-nested:has-text('{suffix}')"
            )
            if cand.count() > 0:
                tab = cand.first
                break
        if tab is None:
            pytest.skip("No plaintext artifact in target session")
        tab.click()
        page.wait_for_timeout(500)
        assert page.locator(".log-view").count() >= 1, ".log-view missing"
        raw = page.locator(".log-view").first.inner_html()
        # Must never contain an unescaped <script> tag.
        assert "<script>" not in raw.lower()
        _save_screenshot(page, "l3-plaintext")
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C7 — extension beats mime
# ---------------------------------------------------------------------------

class TestExtensionBeatsMimeC7:
    def test_md_with_octet_mime_still_renders_markdown(self, page, server_url):
        errors = _console_errors(page)

        def _route(route, request):
            if request.method == "GET" and request.url.endswith(".md"):
                # Force a misleading mime.
                route.fulfill(
                    status=200,
                    content_type="application/octet-stream",
                    body="# Markdown Wins\n\npath wins over mime\n",
                )
                return
            route.continue_()

        page.route("**/api/campaigns/**/artifacts/**.md", _route)
        _goto_session_l3(page, server_url)
        _pick_tab_by_suffix(page, ".md")
        assert page.locator(".l3-prose h1").count() >= 1, (
            "markdown didn't render under octet-stream mime (extension must win)"
        )
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# C8 — mockup-parity screenshots
# ---------------------------------------------------------------------------

class TestMockupParityC8:
    """C8 — all five renderer screenshots captured and > 2 KB."""

    SHOTS = ["l3-md", "l3-json-expanded", "l3-code", "l3-image", "l3-plaintext"]

    def test_all_screenshots_captured(self):
        """Runs LAST — depends on the other C tests having written screenshots."""
        missing = [
            s for s in self.SHOTS
            if not (SCREENSHOT_DIR / f"{s}.png").exists()
            or (SCREENSHOT_DIR / f"{s}.png").stat().st_size < MIN_SCREENSHOT_BYTES
        ]
        assert not missing, (
            f"Screenshot parity failed: {missing} missing or < "
            f"{MIN_SCREENSHOT_BYTES}B. Check C2/C3/C4/C5a/C6 test logs."
        )


# ---------------------------------------------------------------------------
# C9 — legacy L3 OVERVIEW card survives redesign
# ---------------------------------------------------------------------------

class TestLegacyL3OverviewC9:
    def test_l3_overview_card_opacity_reaches_one(self, page, server_url):
        errors = _console_errors(page)
        _goto_session_l3(page, server_url)
        sections = page.locator(".l3-section")
        assert sections.count() >= 1, "existing .l3-section elements missing"
        # OVERVIEW card is the first .l3-section — its reveal animation must
        # complete (opacity → 1).
        page.wait_for_timeout(1500)  # allow reveal animation to finish
        opacity = sections.first.evaluate(
            "el => getComputedStyle(el).opacity"
        )
        assert opacity == "1", (
            f"L3 OVERVIEW card opacity is {opacity!r}; animation didn't complete"
        )
        assert errors == [], f"Console errors: {errors}"
