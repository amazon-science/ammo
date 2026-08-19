# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for download endpoint ownership validation.

Tests that prepare_download and download_session properly enforce
ownership checks so only the session owner can download archives.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import (
    reset_all_singletons,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
    mock_session_manager,
    make_session_state,
)
from shared.session_models import SessionStatus

# Valid UUID v4 strings for endpoint tests (get_client_id validates UUID format)
CLIENT_A_UUID = "00000000-0000-4000-a000-000000000001"
CLIENT_B_UUID = "00000000-0000-4000-a000-000000000002"
CLIENT_OWNER_UUID = "00000000-0000-4000-a000-000000000003"


# ============================================================================
# TestPrepareDownloadOwnership
# ============================================================================


@pytest.mark.unit
class TestPrepareDownloadOwnership:
    """Tests for ownership validation in SessionManager.prepare_download()."""

    @pytest.mark.asyncio
    async def test_own_session_succeeds(self, mock_session_manager):
        """Owner (client_id matches owner_id) can prepare download."""
        session_id = "sess-dl-own"
        owner = "client-A"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id=owner,
        )
        mock_session_manager._sessions[session_id] = state
        mock_session_manager.session_storage.enabled = True

        result = await mock_session_manager.prepare_download(
            session_id, owner_id=owner
        )

        assert result.archive_ready is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_other_users_session_raises(self, mock_session_manager):
        """Non-owner (client_id doesn't match) gets SessionError."""
        session_id = "sess-dl-other"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id="client-A",
        )
        mock_session_manager._sessions[session_id] = state
        mock_session_manager.session_storage.enabled = True

        from orchestration.session_manager import SessionError

        with pytest.raises(SessionError):
            await mock_session_manager.prepare_download(
                session_id, owner_id="client-B"
            )

    @pytest.mark.asyncio
    async def test_no_client_id_allows_access(self, mock_session_manager):
        """When owner_id is None (no client header), access is allowed (backward compat)."""
        session_id = "sess-dl-no-client"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id="client-A",
        )
        mock_session_manager._sessions[session_id] = state
        mock_session_manager.session_storage.enabled = True

        # owner_id=None means no X-Client-ID header was sent
        result = await mock_session_manager.prepare_download(
            session_id, owner_id=None
        )

        assert result.archive_ready is True

    @pytest.mark.asyncio
    async def test_cross_pod_session_checks_ownership_from_s3(
        self, mock_session_manager
    ):
        """
        Cross-pod session (not in local _sessions) loads metadata from S3
        and still enforces ownership.
        """
        session_id = "sess-cross-pod"
        mock_session_manager.session_storage.enabled = True

        # Session NOT in local _sessions (cross-pod scenario)
        assert session_id not in mock_session_manager._sessions

        # S3 returns metadata with owner_id = "client-A"
        s3_state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            owner_id="client-A",
        )
        mock_session_manager.session_storage.load_session_metadata = AsyncMock(
            return_value=s3_state
        )

        from orchestration.session_manager import SessionError

        # client-B should be rejected even for cross-pod sessions
        with pytest.raises(SessionError):
            await mock_session_manager.prepare_download(
                session_id, owner_id="client-B"
            )


@pytest.fixture
def mock_s3_enabled():
    """Create a MagicMock S3 storage with enabled=True and a default presigned URL."""
    s3 = MagicMock()
    s3.enabled = True
    s3.get_download_url = AsyncMock(
        return_value="https://s3.example.com/presigned-url"
    )
    return s3


# ============================================================================
# TestDownloadEndpointOwnership
# ============================================================================


@pytest.mark.unit
class TestDownloadEndpointOwnership:
    """Tests for ownership validation in the GET /sessions/{id}/download endpoint."""

    @pytest.fixture(autouse=True)
    def _disable_api_key_auth(self):
        """Disable API key middleware to prevent 401s from cross-test pollution.

        The APIKeyMiddleware reads AMMO_API_KEY at module level. Other test files
        (e.g. test_api_key_middleware.py) may reload the app module with a key set,
        which persists across the test suite. Patching it to empty ensures these
        endpoint tests exercise the download logic, not the auth middleware.
        """
        import app as app_module
        original = app_module.AMMO_API_KEY
        app_module.AMMO_API_KEY = ""
        yield
        app_module.AMMO_API_KEY = original

    @pytest.mark.asyncio
    async def test_own_session_returns_redirect(self, mock_session_manager, mock_s3_enabled):
        """Owner can download their session (gets presigned URL redirect)."""
        session_id = "sess-dl-redirect"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id=CLIENT_OWNER_UUID,
        )
        mock_session_manager._sessions[session_id] = state

        with patch("app.session_manager", mock_session_manager), \
             patch("app.session_s3_storage", mock_s3_enabled):
            from app import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                f"/sessions/{session_id}/download",
                headers={"X-Client-ID": CLIENT_OWNER_UUID},
                follow_redirects=False,
            )

        # Should redirect to the presigned URL
        assert response.status_code == 307

    @pytest.mark.asyncio
    async def test_other_users_session_returns_404(self, mock_session_manager, mock_s3_enabled):
        """Non-owner gets 404 when trying to download another user's session."""
        session_id = "sess-dl-blocked"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id=CLIENT_A_UUID,
        )
        mock_session_manager._sessions[session_id] = state

        with patch("app.session_manager", mock_session_manager), \
             patch("app.session_s3_storage", mock_s3_enabled):
            from app import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                f"/sessions/{session_id}/download",
                headers={"X-Client-ID": CLIENT_B_UUID},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_session_not_in_memory_returns_404(self, mock_session_manager, mock_s3_enabled):
        """Session not in local memory (cross-pod) returns 404 with guidance."""
        with patch("app.session_manager", mock_session_manager), \
             patch("app.session_s3_storage", mock_s3_enabled):
            from app import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/sessions/nonexistent-session/download",
                headers={"X-Client-ID": CLIENT_A_UUID},
            )

        assert response.status_code == 404
        assert "prepare-download" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_no_session_at_all_returns_404(self, mock_session_manager, mock_s3_enabled):
        """Completely unknown session ID returns 404."""
        with patch("app.session_manager", mock_session_manager), \
             patch("app.session_s3_storage", mock_s3_enabled):
            from app import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/sessions/does-not-exist/download",
                headers={"X-Client-ID": CLIENT_B_UUID},
            )

        assert response.status_code == 404
