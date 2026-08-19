# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the HEAD route on /api/campaigns/{session_id}/artifacts/{path}
(L3 artifact viewer — Task #4 / D group).

The HEAD route is required so the frontend can probe an artifact's size
(Content-Length) to decide whether to inline-render an image (<= 5 MB) or
fall back to a binary-download card. The GET handler must continue to work
unchanged (D5 regression).

Security invariants preserved:
  * `realpath` path-traversal check
  * session ownership gate (via session_manager.get_session(..., owner_id=...))
  * API-key middleware inheritance

All five tests must FAIL against the current code (no HEAD support, no
stat helpers on CampaignDataService) and PASS after Task #4 is implemented.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

# Add parent directories to path so `import app` resolves the server package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Valid UUID v4 strings — `get_client_id` validates UUID format.
CLIENT_OWNER_UUID = "00000000-0000-4000-a000-000000000011"
CLIENT_OTHER_UUID = "00000000-0000-4000-a000-000000000012"

VALID_API_KEY = "test-secret-key-32bytes-minimum-length-00"


def _make_session_info(session_id: str, worktree_path: str):
    """Minimal SessionInfo-like mock (only the fields the route reads)."""
    s = MagicMock()
    s.session_id = session_id
    s.worktree_path = worktree_path
    return s


@pytest.fixture(autouse=True)
def _disable_api_key_auth_by_default():
    """Disable the API-key middleware for tests that don't opt into it.

    Module-level `AMMO_API_KEY` is read once at import time, and other test
    modules (e.g. test_api_key_middleware.py) can reload it with a key set.
    Force it empty here so the HEAD tests exercise the route itself, not the
    auth gate — except the ownership test, which patches it back on.
    """
    import app as app_module
    original = app_module.AMMO_API_KEY
    app_module.AMMO_API_KEY = ""
    yield
    app_module.AMMO_API_KEY = original


# ---------------------------------------------------------------------------
# D1. HEAD returns Content-Length and empty body
# ---------------------------------------------------------------------------

def test_head_returns_content_length(tmp_path):
    """HEAD on a known-size artifact returns 200 + Content-Length + empty body."""
    import app as app_module

    # Build a real artifact layout on disk so the service resolves it.
    art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
    art_dir.mkdir(parents=True)
    (art_dir / "state.json").write_text('{"ok": true}')
    payload = "# Report\n\nLorem ipsum dolor sit amet.\n"
    (art_dir / "report.md").write_text(payload)
    expected_size = (art_dir / "report.md").stat().st_size
    assert expected_size == os.path.getsize(art_dir / "report.md")

    session = _make_session_info("sess-head-1", str(tmp_path))
    mock_sm = AsyncMock()
    mock_sm.get_session = AsyncMock(return_value=session)

    with patch("app.session_manager", mock_sm):
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.head(
            f"/api/campaigns/sess-head-1/artifacts/report.md",
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers.get("content-length") == str(expected_size)
    # HEAD response body must be empty per RFC 7231.
    assert resp.content == b""


# ---------------------------------------------------------------------------
# D2. HEAD on unknown path → 404, no stack leak
# ---------------------------------------------------------------------------

def test_head_404_unknown_path(tmp_path):
    """HEAD on a nonexistent artifact returns 404 with no traceback leakage."""
    import app as app_module

    art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
    art_dir.mkdir(parents=True)
    (art_dir / "state.json").write_text("{}")

    session = _make_session_info("sess-head-2", str(tmp_path))
    mock_sm = AsyncMock()
    mock_sm.get_session = AsyncMock(return_value=session)

    with patch("app.session_manager", mock_sm):
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.head(
            "/api/campaigns/sess-head-2/artifacts/does_not_exist.md",
        )

    assert resp.status_code == 404
    # Stack traces shouldn't leak into the response body (body is empty for HEAD
    # anyway, but we also check that the status was clean — never 500).
    assert resp.status_code != 500
    # HEAD body must be empty regardless of error.
    assert resp.content == b""


# ---------------------------------------------------------------------------
# D3. Path-traversal blocked
# ---------------------------------------------------------------------------

def test_head_path_traversal_blocked(tmp_path):
    """`..%2Fetc%2Fpasswd` (URL-encoded `../etc/passwd`) resolves outside the
    artifact dir and must return 404 — never leak real-filesystem contents."""
    import app as app_module

    # Two sibling dirs: legitimate artifact root and a "secret" dir above it.
    art_dir = tmp_path / "wt" / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
    art_dir.mkdir(parents=True)
    (art_dir / "state.json").write_text("{}")
    # Secret placed above the worktree root — same-prefix bypass attempt.
    secret = tmp_path / "secret.txt"
    secret.write_text("PWNED")

    session = _make_session_info("sess-head-3", str(tmp_path / "wt"))
    mock_sm = AsyncMock()
    mock_sm.get_session = AsyncMock(return_value=session)

    with patch("app.session_manager", mock_sm):
        client = TestClient(app_module.app, raise_server_exceptions=False)
        # URL-encoded traversal; FastAPI decodes and passes to the handler.
        resp = client.head(
            "/api/campaigns/sess-head-3/artifacts/..%2F..%2Fsecret.txt",
        )

    assert resp.status_code == 404, (
        f"path traversal must be blocked — got {resp.status_code}: {resp.text!r}"
    )
    # Defence in depth: the secret must NEVER appear in the response.
    assert b"PWNED" not in resp.content


# ---------------------------------------------------------------------------
# D4. Ownership enforced when API key is configured
# ---------------------------------------------------------------------------

def test_head_enforces_ownership(tmp_path):
    """With AMMO_API_KEY set, wrong client_id resolves to a None session → 404.

    The endpoint calls `session_manager.get_session(session_id, owner_id=client_id)`.
    When ownership check fails, the manager returns None, and the handler must
    surface a 404 — not leak the session's existence or artifact content.

    We patch `app.AMMO_API_KEY` directly rather than reloading the module,
    because reload resets module-level globals (including `session_manager`
    back to None) and races with the `patch("app.session_manager", ...)`
    context manager used below.
    """
    import app as app_module

    art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
    art_dir.mkdir(parents=True)
    (art_dir / "state.json").write_text("{}")
    (art_dir / "report.md").write_text("# Secret")

    # Mock: get_session returns None when owner_id doesn't match.
    async def _get_session(session_id, owner_id=None):
        if owner_id == CLIENT_OWNER_UUID:
            return _make_session_info(session_id, str(tmp_path))
        return None

    mock_sm = AsyncMock()
    mock_sm.get_session = AsyncMock(side_effect=_get_session)

    # The autouse _disable_api_key_auth_by_default fixture already reset
    # AMMO_API_KEY to "". Flip it on just for this test via attribute patch.
    original_key = app_module.AMMO_API_KEY
    app_module.AMMO_API_KEY = VALID_API_KEY
    try:
        with patch("app.session_manager", mock_sm):
            client = TestClient(app_module.app, raise_server_exceptions=False)

            # Wrong client ID → owner mismatch → 404.
            resp = client.head(
                "/api/campaigns/sess-head-4/artifacts/report.md",
                headers={
                    "X-API-Key": VALID_API_KEY,
                    "X-Client-ID": CLIENT_OTHER_UUID,
                },
            )
            assert resp.status_code == 404, (
                f"wrong owner must get 404 — got {resp.status_code}: {resp.text!r}"
            )

            # Sanity check: the correct owner gets 200 with the right size.
            resp_ok = client.head(
                "/api/campaigns/sess-head-4/artifacts/report.md",
                headers={
                    "X-API-Key": VALID_API_KEY,
                    "X-Client-ID": CLIENT_OWNER_UUID,
                },
            )
            assert resp_ok.status_code == 200
            expected_size = (art_dir / "report.md").stat().st_size
            assert resp_ok.headers.get("content-length") == str(expected_size)
    finally:
        app_module.AMMO_API_KEY = original_key


# ---------------------------------------------------------------------------
# D5. GET regression — HEAD route conversion must not break GET.
# ---------------------------------------------------------------------------

def test_get_still_works_for_head_capable_route(tmp_path):
    """After swapping @app.get → @app.api_route methods=[GET, HEAD], the GET
    path must continue to return the file body with the correct MIME type."""
    import app as app_module

    art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
    art_dir.mkdir(parents=True)
    (art_dir / "state.json").write_text("{}")
    body = "# My Report\n"
    (art_dir / "report.md").write_text(body)

    session = _make_session_info("sess-head-5", str(tmp_path))
    mock_sm = AsyncMock()
    mock_sm.get_session = AsyncMock(return_value=session)

    with patch("app.session_manager", mock_sm):
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.get(
            "/api/campaigns/sess-head-5/artifacts/report.md",
        )

    assert resp.status_code == 200, resp.text
    assert resp.text == body
    assert "text/markdown" in (resp.headers.get("content-type") or "")
