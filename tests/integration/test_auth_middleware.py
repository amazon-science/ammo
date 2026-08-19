# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for API key authentication middleware.

Tests verify that the auth middleware correctly gates protected endpoints
and leaves open endpoints accessible. Uses Starlette TestClient to test
against the real FastAPI app with AMMO_API_KEY set.

These tests complement the unit tests in tests/unit/test_api_key_middleware.py
by exercising the middleware against the actual endpoint handlers.
"""

import importlib
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_API_KEY = "integration-test-key-32bytes-minimum-len"


def _make_client(monkeypatch, api_key=TEST_API_KEY):
    """Create a Starlette TestClient with AMMO_API_KEY set."""
    if api_key is None:
        monkeypatch.delenv("AMMO_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AMMO_API_KEY", api_key)

    import app as app_module
    importlib.reload(app_module)

    from starlette.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAuthMiddlewareIntegration:
    """Test auth middleware against real endpoints."""

    def test_sessions_401_without_key(self, monkeypatch):
        """GET /sessions returns 401 when AMMO_API_KEY is set and no auth provided."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"

    def test_sessions_200_with_key(self, monkeypatch):
        """GET /sessions passes auth when correct key provided."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        # Should pass auth (not 401); may be 200 or 503 depending on session service state
        assert resp.status_code != 401

    def test_hf_model_config_requires_auth(self, monkeypatch):
        """GET /api/hf-model-config/{id} returns 401 without auth. Replaces the
        removed /api/moe-models + /api/supported-models auth checks — the new
        endpoint also sits under the /api/ prefix so it is auto-protected."""
        client = _make_client(monkeypatch)
        resp = client.get("/api/hf-model-config/meta-llama/Llama-3.1-8B")
        assert resp.status_code == 401

    def test_hf_model_config_passes_auth_with_key(self, monkeypatch):
        """GET /api/hf-model-config/{id} passes the auth middleware (status is
        NOT 401) when a valid Bearer key is supplied. The endpoint may still
        surface a non-401 error upstream (e.g. network error talking to HF);
        this test only asserts the middleware accepts the key."""
        client = _make_client(monkeypatch)
        resp = client.get(
            "/api/hf-model-config/meta-llama/Llama-3.1-8B",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code != 401

    def test_health_open_without_auth(self, monkeypatch):
        """GET /health remains open without auth."""
        client = _make_client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code != 401

    def test_health_with_key_still_open(self, monkeypatch):
        """GET /health accepts a valid key but does not require one."""
        client = _make_client(monkeypatch)
        resp = client.get("/health", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code != 401

    def test_health_always_open(self, monkeypatch):
        """GET /health never returns 401."""
        client = _make_client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code != 401
        # Health endpoint may return 200 or 500 depending on initialization state
        assert resp.status_code in (200, 500)

    def test_ui_html_always_open(self, monkeypatch):
        """GET /ui returns HTML without auth."""
        client = _make_client(monkeypatch)
        resp = client.get("/ui")
        assert resp.status_code != 401

    def test_sessions_requires_auth(self, monkeypatch):
        """GET /sessions returns 401 without auth."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions")
        assert resp.status_code == 401

    def test_sessions_200_with_key(self, monkeypatch):
        """GET /sessions passes auth with valid key."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        # Should pass auth; may be 200 or 503 if the local service is not initialized.
        assert resp.status_code != 401

    def test_session_limit_returns_429(self, monkeypatch):
        """Creating sessions beyond limit returns 429.

        This test verifies that the SessionLimitError raised by session_manager
        is properly translated to HTTP 429 by the endpoint handler.
        We mock session_manager.create_session to raise SessionLimitError.
        """
        from unittest.mock import AsyncMock

        client = _make_client(monkeypatch)
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}

        # Import the error class used by app.py (aliased as SessionLimitMgrError)
        from orchestration.session_manager import SessionLimitError

        # Patch the module-level session_manager in app
        import app as app_module
        original_sm = app_module.session_manager
        mock_sm = AsyncMock()
        mock_sm.create_session = AsyncMock(
            side_effect=SessionLimitError("Maximum 3 active sessions per client")
        )
        app_module.session_manager = mock_sm

        try:
            resp = client.post(
                "/sessions",
                json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
                headers=headers,
            )
        finally:
            app_module.session_manager = original_sm

        assert resp.status_code == 429
        data = resp.json()
        error_msg = data.get("detail", "") or data.get("error", "")
        assert "Maximum 3 active sessions" in error_msg

    def test_wrong_key_returns_401(self, monkeypatch):
        """Providing an incorrect key returns 401."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions", headers={"Authorization": "Bearer wrong-key-completely"})
        assert resp.status_code == 401

    def test_cookie_auth_works(self, monkeypatch):
        """Valid key via ammo_api_key cookie passes auth."""
        client = _make_client(monkeypatch)
        client.cookies.set("ammo_api_key", TEST_API_KEY)
        resp = client.get("/sessions")
        assert resp.status_code != 401

    def test_x_api_key_header_works(self, monkeypatch):
        """Valid key via X-API-Key header passes auth."""
        client = _make_client(monkeypatch)
        resp = client.get("/sessions", headers={"X-API-Key": TEST_API_KEY})
        assert resp.status_code != 401

    def test_query_param_token_works(self, monkeypatch):
        """Valid key via ?token=<key> query param passes auth."""
        client = _make_client(monkeypatch)
        resp = client.get(f"/sessions?token={TEST_API_KEY}")
        assert resp.status_code != 401
