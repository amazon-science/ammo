# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for API key authentication middleware.

Tests that AMMO_API_KEY environment variable gates session/UI endpoints
while keeping eval endpoints open.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_KEY = "test-secret-key-32bytes-minimum-length-00"
WRONG_KEY = "wrong-key-that-does-not-match-at-all-000"


def _make_client(monkeypatch, api_key=None):
    """Create a Starlette TestClient with AMMO_API_KEY set (or unset)."""
    import importlib

    if api_key is None:
        monkeypatch.delenv("AMMO_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AMMO_API_KEY", api_key)

    # Reload app module so module-level AMMO_API_KEY picks up the new env
    import app as app_module
    importlib.reload(app_module)

    from starlette.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# TestApiKeyMiddlewareDevMode
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiKeyMiddlewareDevMode:
    """When AMMO_API_KEY is unset or empty, all requests should pass through."""

    def test_unset_key_allows_protected_path(self, monkeypatch):
        """AMMO_API_KEY unset -> /sessions should NOT return 401."""
        client = _make_client(monkeypatch, api_key=None)
        resp = client.get("/sessions")
        assert resp.status_code != 401

    def test_empty_key_allows_protected_path(self, monkeypatch):
        """AMMO_API_KEY='' -> /sessions should NOT return 401."""
        client = _make_client(monkeypatch, api_key="")
        resp = client.get("/sessions")
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# TestApiKeyMiddlewareAuthEnabled
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiKeyMiddlewareAuthEnabled:
    """When AMMO_API_KEY is set, verify key extraction and validation."""

    def test_valid_bearer_token(self, monkeypatch):
        """Valid key via Authorization: Bearer <key> passes."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {VALID_KEY}"})
        assert resp.status_code != 401

    def test_valid_x_api_key_header(self, monkeypatch):
        """Valid key via X-API-Key header passes."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code != 401

    def test_valid_cookie(self, monkeypatch):
        """Valid key via ammo_api_key cookie passes."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        client.cookies.set("ammo_api_key", VALID_KEY)
        resp = client.get("/sessions")
        assert resp.status_code != 401

    def test_valid_query_param(self, monkeypatch):
        """Valid key via ?token=<key> query param passes."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get(f"/sessions?token={VALID_KEY}")
        assert resp.status_code != 401

    def test_invalid_key_returns_401(self, monkeypatch):
        """Invalid key returns 401 with expected detail."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {WRONG_KEY}"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"

    def test_missing_key_returns_401(self, monkeypatch):
        """No key on protected path returns 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"

    def test_empty_bearer_token_returns_401(self, monkeypatch):
        """Empty bearer token returns 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_whitespace_only_key_returns_401(self, monkeypatch):
        """Whitespace-only key returns 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/sessions", headers={"Authorization": "Bearer    "})
        assert resp.status_code == 401

    def test_partial_key_returns_401(self, monkeypatch):
        """Partial key match returns 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        partial = VALID_KEY[:20]
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {partial}"})
        assert resp.status_code == 401

    def test_bearer_priority_over_cookie(self, monkeypatch):
        """Wrong Bearer + right cookie -> 401 (Bearer checked first)."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        client.cookies.set("ammo_api_key", VALID_KEY)
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {WRONG_KEY}"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestApiKeyMiddlewareProtectedPaths
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiKeyMiddlewareProtectedPaths:
    """All these paths should return 401 without a valid key."""

    @pytest.mark.parametrize("path", [
        "/sessions",
        "/sessions/abc123",
        "/sessions/abc123/terminal/ws",
        "/sessions/abc123/terminal/",
        "/api/hf-model-config/meta-llama/Llama-3.1-8B",
        "/docs",
        "/redoc",
        "/openapi.json",
    ])
    def test_protected_path_returns_401(self, monkeypatch, path):
        """Protected path without key returns 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get(path)
        assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"


# ---------------------------------------------------------------------------
# TestApiKeyMiddlewareOpenPaths
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiKeyMiddlewareOpenPaths:
    """All these paths should pass without a key (no 401)."""

    @pytest.mark.parametrize("method,path", [
        ("GET", "/health"),
        ("GET", "/ui"),
        ("GET", "/api/changelog"),
    ])
    def test_open_path_does_not_return_401(self, monkeypatch, method, path):
        """Open path should not return 401 (may fail for other reasons)."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.request(method, path)
        assert resp.status_code != 401, f"Expected non-401 for {method} {path}, got {resp.status_code}"


# ---------------------------------------------------------------------------
# TestApiKeyMiddlewarePathEdgeCases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiKeyMiddlewarePathEdgeCases:
    """Edge cases for path matching logic."""

    def test_cluster_exact_is_not_a_protected_prefix(self, monkeypatch):
        """GET /cluster does not match the "/cluster/" prefix, so the middleware
        lets it through. The HTML dashboard route itself no longer exists, so the
        app answers 404 rather than 401."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/cluster")
        assert resp.status_code != 401
        assert resp.status_code == 404

    def test_health_is_open(self, monkeypatch):
        """GET /health remains open even when API auth is configured."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/health")
        assert resp.status_code != 401

    def test_clusterXYZ_is_open(self, monkeypatch):
        """GET /clusterXYZ is open (not a real route, no prefix match)."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/clusterXYZ")
        assert resp.status_code != 401

    def test_cluster_with_query_string_is_not_protected(self, monkeypatch):
        """GET /cluster?foo=bar is not auth-gated (path is /cluster, query ignored)."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/cluster?foo=bar")
        assert resp.status_code != 401

    def test_health_with_query_string_is_open(self, monkeypatch):
        """GET /health?foo=bar remains open."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/health?foo=bar")
        assert resp.status_code != 401

    def test_api_changelog_prefix_remains_protected(self, monkeypatch):
        """/api/changelogXYZ must stay protected — OPEN_EXACT_PATHS uses exact match, not prefix."""
        client = _make_client(monkeypatch, api_key=VALID_KEY)
        resp = client.get("/api/changelogXYZ")
        assert resp.status_code == 401, (
            f"Expected 401 for /api/changelogXYZ (not in OPEN_EXACT_PATHS), got {resp.status_code}"
        )
