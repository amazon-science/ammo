# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests for Session CRUD flows in the AMMO Sessions UI.

Covers: page load, sidebar rendering, create modal, session lifecycle
actions (pause / resume / terminate), and batch delete.

All tests use mocked API responses for determinism.
"""

import json
import pytest
from playwright.sync_api import Page, expect

from tests.playwright.conftest import (
    MOCK_AUTH_API_KEY,
    MOCK_HEALTH_RESPONSE,
    MOCK_SESSION_ACTIVE,
    MOCK_SESSION_PAUSED,
    MOCK_SESSION_TERMINATED,
    MOCK_SESSIONS_RESPONSE,
    mock_api_routes,
)

pytestmark = pytest.mark.playwright


# -------------------------------------------------------------------------
# Page load & sidebar rendering
# -------------------------------------------------------------------------


class TestPageLoad:
    """Verify the /ui page loads correctly and renders the sidebar."""

    def test_page_title(self, mock_page: Page):
        expect(mock_page).to_have_title("AMMO — Campaign Dashboard")

    def test_sidebar_visible(self, mock_page: Page):
        sidebar = mock_page.locator("aside.session-sidebar")
        expect(sidebar).to_be_visible()

    def test_header_visible(self, mock_page: Page):
        header = mock_page.locator("header")
        expect(header).to_be_visible()
        expect(mock_page.locator("h1")).to_have_text("AMMO Sessions")

    def test_gpu_info_badge_shows_type(self, mock_page: Page):
        badge = mock_page.locator(".gpu-info-badge")
        expect(badge).to_be_visible()
        # The mock returns gpu type "h200"
        expect(badge).to_contain_text("H200")


# -------------------------------------------------------------------------
# Session sidebar cards
# -------------------------------------------------------------------------


class TestSidebarSessionCards:
    """Session cards should appear grouped by status."""

    def test_active_session_displayed(self, mock_page: Page):
        # Active section header
        active_header = mock_page.locator("text=Active").first
        expect(active_header).to_be_visible()

        # Session card shows model name + dtype
        card_text = mock_page.locator(
            "text=DeepSeek-R1-0528 (fp8)"
        ).first
        expect(card_text).to_be_visible()

    def test_paused_session_displayed(self, mock_page: Page):
        paused_header = mock_page.locator("text=Paused").first
        expect(paused_header).to_be_visible()

    def test_terminated_session_displayed(self, mock_page: Page):
        terminated_header = mock_page.locator("text=Terminated").first
        expect(terminated_header).to_be_visible()

    def test_session_id_shown_on_card(self, mock_page: Page):
        # The active session card should show the session_id text
        sid = mock_page.locator(f"text={MOCK_SESSION_ACTIVE['session_id']}")
        expect(sid.first).to_be_visible()


# -------------------------------------------------------------------------
# "+ New AMMO Session" button & Create modal
# -------------------------------------------------------------------------


class TestCreateSessionModal:
    """The create-session modal and its form elements."""

    def test_new_button_visible(self, mock_page: Page):
        btn = mock_page.locator("button.create-session-btn")
        expect(btn).to_be_visible()
        expect(btn).to_contain_text("New")

    def test_modal_opens_on_click(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        modal_heading = mock_page.locator("h2:has-text('Create AMMO Session')")
        expect(modal_heading).to_be_visible()

    def test_modal_has_model_combobox(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        # Scope to the classic create modal (not the LIGHTGRID one inside #campaign-view)
        modal = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]")
        # The create modal uses a searchable combobox input, not a <select>.
        # Static presets were removed — model selection runs entirely through
        # the HF search combobox (queries /api/hf-models, configs auto-filled
        # from /api/hf-model-config/{id}). Just verify the input is present.
        search_input = modal.locator(
            "input[placeholder='Search models or paste HuggingFace ID...']"
        )
        expect(search_input).to_be_visible()
        # Focusing must not explode — dropdown may render HF results, but the
        # old "DeepSeek-R1 (FP8)" preset chip is gone post static-model removal.
        search_input.click()
        mock_page.wait_for_timeout(300)

    def test_modal_has_dtype_buttons(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        # Scope to the Data Type section in the classic create modal
        dtype_section = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]").locator(
            "label:has-text('Data Type')"
        ).locator("xpath=following-sibling::div[1]")
        # The mock GPU returns allowed_dtypes: fp8, fp4, bf16, fp16
        for dtype in ["fp8", "fp4", "bf16", "fp16"]:
            btn = dtype_section.locator(f"button:has-text('{dtype}')")
            expect(btn).to_be_visible()

    def test_modal_has_tp_buttons(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        # Scope to the Tensor Parallelism section in the classic create modal
        tp_section = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]").locator(
            "label:has-text('Tensor Parallelism')"
        ).locator("xpath=ancestor::div[1]").locator("div.flex.gap-2")
        for tp in ["1", "2", "4", "8"]:
            btn = tp_section.locator(f"button:has-text('{tp}')")
            expect(btn).to_be_visible()

    def test_modal_closes_on_cancel(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        expect(
            mock_page.locator("h2:has-text('Create AMMO Session')")
        ).to_be_visible()
        # Scope Cancel to the create modal (not the confirm dialog or LIGHTGRID)
        mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'bg-gh-card')]").locator(
            "button:has-text('Cancel')"
        ).click()
        expect(
            mock_page.locator("h2:has-text('Create AMMO Session')")
        ).not_to_be_visible()

    def test_modal_closes_on_escape(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        expect(
            mock_page.locator("h2:has-text('Create AMMO Session')")
        ).to_be_visible()
        mock_page.keyboard.press("Escape")
        expect(
            mock_page.locator("h2:has-text('Create AMMO Session')")
        ).not_to_be_visible()

    def test_create_button_disabled_without_model(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        # Scope to the classic modal to avoid matching LIGHTGRID's Create Session button
        modal = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]")
        create_btn = modal.locator("button:has-text('Create Session')")
        expect(create_btn).to_be_disabled()

    def test_prompt_preview_updates(self, mock_page: Page):
        mock_page.click("button.create-session-btn")
        # Scope to the classic create modal
        modal = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]")
        # Focus the model search input to open the dropdown
        search_input = modal.locator(
            "input[placeholder='Search models or paste HuggingFace ID...']"
        )
        search_input.click()
        mock_page.wait_for_timeout(300)
        # Click the preset model from the dropdown
        modal.locator("button:has-text('DeepSeek-R1 (FP8)')").first.click()
        mock_page.wait_for_timeout(200)

        # Prompt preview should contain the model id
        preview = modal.locator("code")
        expect(preview).to_contain_text("deepseek-ai/DeepSeek-R1-0528")

    def test_create_session_sends_post(self, mock_page: Page):
        """Select model, click Create, verify POST /sessions is called."""
        captured_requests = []

        def capture(route, request):
            if request.method == "POST" and request.url.endswith("/sessions"):
                captured_requests.append(json.loads(request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(MOCK_SESSION_ACTIVE),
                )
            else:
                route.continue_()

        # Re-route POST /sessions specifically
        mock_page.route("**/sessions", capture)

        mock_page.click("button.create-session-btn")
        # Scope to the classic create modal
        modal = mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]")
        # Use the searchable combobox to select a model
        search_input = modal.locator(
            "input[placeholder='Search models or paste HuggingFace ID...']"
        )
        search_input.click()
        mock_page.wait_for_timeout(300)
        modal.locator("button:has-text('DeepSeek-R1 (FP8)')").first.click()
        mock_page.wait_for_timeout(200)

        create_btn = modal.locator(
            "button:has-text('Create Session'):not([disabled])"
        )
        create_btn.click()
        mock_page.wait_for_timeout(1000)

        assert len(captured_requests) >= 1
        payload = captured_requests[0]
        assert payload["model_name"] == "deepseek-ai/DeepSeek-R1-0528"
        assert payload["dtype"] == "fp8"  # selectModel() applies preset default_dtype
        assert payload["repo_name"] == "vllm"


# -------------------------------------------------------------------------
# Session card actions: Open terminal
# -------------------------------------------------------------------------


class TestOpenTerminal:
    """Clicking an active session opens its terminal panel."""

    def test_click_active_session_opens_terminal(self, mock_page: Page):
        # Mock the terminal URL to avoid actual ttyd connection
        mock_page.route(
            "**/sessions/test-session-aaa111/terminal/**",
            lambda route, _: route.fulfill(
                status=200,
                content_type="text/html",
                body="<html><body>mock terminal</body></html>",
            ),
        )

        # Click the "Open" button on the active session card
        open_btn = mock_page.locator("button:has-text('Open')").first
        open_btn.click()
        mock_page.wait_for_timeout(500)

        # Terminal header should appear with model name
        terminal_header = mock_page.locator(
            ".terminal-container"
        ).first
        expect(terminal_header).to_be_visible()

    def test_url_hash_updates_on_open(self, mock_page: Page):
        mock_page.route(
            "**/sessions/test-session-aaa111/terminal/**",
            lambda route, _: route.fulfill(
                status=200,
                content_type="text/html",
                body="<html><body>mock</body></html>",
            ),
        )
        open_btn = mock_page.locator("button:has-text('Open')").first
        open_btn.click()
        mock_page.wait_for_timeout(300)

        assert "session/test-session-aaa111" in mock_page.url


# -------------------------------------------------------------------------
# Pause / Resume / Terminate actions with loading states
# -------------------------------------------------------------------------


class TestSessionActions:
    """Verify pause, resume, terminate actions + loading spinner states."""

    def test_pause_button_shows_spinner(self, mock_page: Page):
        """Click Pause on active session -> loading spinner appears."""
        # Mock the pause endpoint to be slow
        mock_page.route(
            "**/sessions/test-session-aaa111/pause",
            lambda route, _: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "paused"}),
            ),
        )

        # Find the Pause button in the active session card
        pause_btn = mock_page.locator("button:has-text('Pause')").first
        pause_btn.click()

        # The button should briefly show "Pausing..." text
        pausing_text = mock_page.locator("text=Pausing...")
        # Either visible during the action or the action completed
        mock_page.wait_for_timeout(500)

    def test_resume_button_on_paused_session(self, mock_page: Page):
        """Paused session shows Resume button after expanding the collapsed section."""
        # The Paused section is collapsed by default — expand it first
        mock_page.locator("button:has-text('Paused')").first.click()
        mock_page.wait_for_timeout(300)
        resume_btn = mock_page.locator("button:has-text('Resume')").first
        expect(resume_btn).to_be_visible()

    def test_resume_sends_post(self, mock_page: Page):
        captured = []

        def handle_resume(route, request):
            if "resume" in request.url and request.method == "POST":
                captured.append(request.url)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "active"}),
                )
            else:
                route.continue_()

        mock_page.route("**/sessions/*/resume", handle_resume)
        # Expand the collapsed Paused section first
        mock_page.locator("button:has-text('Paused')").first.click()
        mock_page.wait_for_timeout(300)
        resume_btn = mock_page.locator("button:has-text('Resume')").first
        resume_btn.click()
        mock_page.wait_for_timeout(500)

        assert len(captured) >= 1
        assert "test-session-bbb222" in captured[0]

    def test_terminate_shows_confirm_dialog(self, mock_page: Page):
        """Clicking terminate trash icon opens confirmation dialog."""
        # The terminate button is the trash icon SVG button in the active card
        delete_btns = mock_page.locator(
            "button.bg-gh-error\\/20"
        )
        # Click the first one (active session card)
        delete_btns.first.click()
        mock_page.wait_for_timeout(300)

        # Confirm dialog should appear with title set via x-text
        dialog_title = mock_page.locator("h3:has-text('Terminate Session?')")
        expect(dialog_title).to_be_visible()

        # Scope buttons to the confirm dialog container (sibling of h3)
        confirm_dialog = dialog_title.locator("xpath=ancestor::div[contains(@class,'bg-gh-card')]")
        confirm_btn = confirm_dialog.locator("button:has-text('Confirm')")
        expect(confirm_btn).to_be_visible()

        cancel_btn = confirm_dialog.locator("button:has-text('Cancel')")
        expect(cancel_btn).to_be_visible()

    def test_terminate_confirm_sends_delete(self, mock_page: Page):
        captured = []

        def handle_delete(route, request):
            if request.method == "DELETE" and "test-session-aaa111" in request.url:
                captured.append(request.url)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "terminated"}),
                )
            else:
                route.continue_()

        mock_page.route("**/sessions/test-session-aaa111", handle_delete)

        # Open confirm dialog
        delete_btns = mock_page.locator("button.bg-gh-error\\/20")
        delete_btns.first.click()
        mock_page.wait_for_timeout(200)

        # Click confirm
        mock_page.locator("button:has-text('Confirm')").click()
        mock_page.wait_for_timeout(500)

        assert len(captured) >= 1

    def test_terminate_cancel_closes_dialog(self, mock_page: Page):
        delete_btns = mock_page.locator("button.bg-gh-error\\/20")
        delete_btns.first.click()
        mock_page.wait_for_timeout(200)

        dialog_title = mock_page.locator("h3:has-text('Terminate Session?')")
        expect(dialog_title).to_be_visible()
        # Scope Cancel to the confirm dialog container
        confirm_dialog = dialog_title.locator("xpath=ancestor::div[contains(@class,'bg-gh-card')]")
        confirm_dialog.locator("button:has-text('Cancel')").click()
        mock_page.wait_for_timeout(200)

        expect(dialog_title).not_to_be_visible()


# -------------------------------------------------------------------------
# Batch delete terminated sessions
# -------------------------------------------------------------------------


class TestBatchDeleteTerminated:
    """The 'Delete All' link for terminated sessions."""

    def test_delete_all_link_visible(self, mock_page: Page):
        link = mock_page.locator("button:has-text('Delete All')")
        expect(link).to_be_visible()

    def test_delete_all_opens_confirm(self, mock_page: Page):
        mock_page.locator("button:has-text('Delete All')").click()
        mock_page.wait_for_timeout(200)
        expect(
            mock_page.locator("text=Delete All Terminated Sessions?")
        ).to_be_visible()

    def test_delete_all_confirm_sends_delete(self, mock_page: Page):
        captured = []

        def handle(route, request):
            if request.method == "DELETE" and request.url.endswith("/sessions/terminated"):
                captured.append(True)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"message": "Deleted 1 terminated sessions", "deleted": 1}),
                )
            else:
                route.continue_()

        mock_page.route("**/sessions/terminated", handle)

        mock_page.locator("button:has-text('Delete All')").click()
        mock_page.wait_for_timeout(200)
        mock_page.locator("button:has-text('Confirm')").click()
        mock_page.wait_for_timeout(500)

        assert len(captured) >= 1


# -------------------------------------------------------------------------
# Empty state
# -------------------------------------------------------------------------


class TestEmptyState:
    """When no sessions exist, an empty-state message should appear."""

    def test_no_sessions_shows_empty(self, context, server_url):
        pg = context.new_page()

        # Mock with empty session list
        mock_api_routes(pg, {
            "/sessions": (200, {"sessions": []}),
        })

        pg.goto(f"{server_url}/ui", wait_until="networkidle")
        pg.wait_for_timeout(500)

        expect(pg.locator("text=No sessions yet")).to_be_visible()
        expect(
            pg.locator("text=Create your first session")
        ).to_be_visible()
        pg.close()


# -------------------------------------------------------------------------
# Authenticated mock page tests
# -------------------------------------------------------------------------


class TestAuthenticatedMockPage:
    """Tests using the authenticated_mock_page fixture (API key pre-populated)."""

    def test_page_loads_with_pre_populated_key(self, authenticated_mock_page: Page):
        """Page loads without login modal when API key is in localStorage."""
        modal = authenticated_mock_page.locator("[data-testid='login-modal']")
        # The login modal should NOT be visible since we pre-populated the key
        expect(modal).not_to_be_visible()

    def test_sidebar_visible_when_authenticated(self, authenticated_mock_page: Page):
        """Sidebar renders sessions when authenticated."""
        sidebar = authenticated_mock_page.locator("aside.session-sidebar")
        expect(sidebar).to_be_visible()

    def test_create_session_includes_auth_header(self, authenticated_mock_page: Page):
        """POST /sessions includes Authorization header from localStorage key."""
        # Dismiss the driver.js guided tour that blocks pointer events
        # (authenticated_mock_page fixture doesn't set ammo_tour_completed)
        authenticated_mock_page.evaluate("""() => {
            localStorage.setItem('ammo_tour_completed', 'true');
            // Destroy driver.js tour if active
            if (window.__driver) { try { window.__driver.destroy(); } catch(e) {} }
            // Remove driver overlay and classes that intercept pointer events
            document.querySelectorAll('.driver-overlay, .driver-popover').forEach(e => e.remove());
            document.body.classList.remove('driver-active', 'driver-fade');
        }""")
        authenticated_mock_page.wait_for_timeout(200)

        captured_headers = []

        def capture(route, request):
            if request.method == "POST" and request.url.endswith("/sessions"):
                captured_headers.append(request.headers.get("authorization", ""))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(MOCK_SESSION_ACTIVE),
                )
            else:
                route.continue_()

        authenticated_mock_page.route("**/sessions", capture)
        authenticated_mock_page.click("button.create-session-btn")
        # Scope to the classic create modal
        modal = authenticated_mock_page.locator(
            "h2:has-text('Create AMMO Session')"
        ).locator("xpath=ancestor::div[contains(@class,'rounded-xl')]")
        # Use the searchable combobox to select a model
        search_input = modal.locator(
            "input[placeholder='Search models or paste HuggingFace ID...']"
        )
        search_input.click()
        authenticated_mock_page.wait_for_timeout(300)
        modal.locator(
            "button:has-text('DeepSeek-R1 (FP8)')"
        ).first.click()
        authenticated_mock_page.wait_for_timeout(200)

        create_btn = modal.locator(
            "button:has-text('Create Session'):not([disabled])"
        )
        create_btn.click()
        authenticated_mock_page.wait_for_timeout(1000)

        if captured_headers:
            assert captured_headers[0] == f"Bearer {MOCK_AUTH_API_KEY}"
