# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for WebSocket and HTTP terminal proxy ownership validation.

Tests that authenticated users can only access their own sessions via
terminal WebSocket, HTTP proxy, and static asset proxy endpoints.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.session_manager import SessionError

# Valid UUID v4 values for testing (must match the UUID_PATTERN in app.py)
CLIENT_A = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
CLIENT_B = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"


# ============================================================================
# Helper: create a mock session with a given owner_id
# ============================================================================

def _make_mock_session(owner_id=CLIENT_A, status="active"):
    """Create a mock SessionInfo with the given owner_id."""
    from shared.session_models import SessionStatus
    session = MagicMock()
    session.owner_id = owner_id
    session.status = SessionStatus.ACTIVE if status == "active" else SessionStatus.PAUSED
    session.session_id = "test-session"
    return session


# ============================================================================
# WebSocket Terminal Proxy Ownership Validation
# ============================================================================

@pytest.mark.unit
class TestWebSocketOwnershipValidation:
    """Tests for ownership validation in terminal_websocket_proxy."""

    @pytest.fixture
    def mock_terminal_manager(self):
        """Create a mock TerminalManager with a running terminal."""
        tm = MagicMock()
        tm.get_terminal_port.return_value = 9001
        tm.is_terminal_running.return_value = True
        return tm

    @pytest.fixture
    def mock_session_manager_for_ownership(self):
        """Create a mock SessionManager that supports _validate_ownership."""
        sm = AsyncMock()
        sm.ensure_terminal_healthy = AsyncMock(return_value=9001)
        return sm

    @pytest.mark.asyncio
    async def test_websocket_own_session_proceeds(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """WebSocket to own session (client_id matches owner_id) proceeds to connection."""
        from app import terminal_websocket_proxy

        # _validate_ownership returns the session state (no error)
        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            return_value=_make_mock_session(owner_id=CLIENT_A)
        )

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.query_params = {"client_id": CLIENT_A}
        mock_ws.cookies = {}
        mock_ws.client_state = MagicMock()

        # Short-circuit by making websockets.connect raise
        import websockets
        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership), \
             patch("app.checkpoint_manager", None), \
             patch("app.websockets.connect", side_effect=ConnectionRefusedError("test")), \
             patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):
            await terminal_websocket_proxy(mock_ws, "test-session")

        # Connection was accepted (not closed with ownership error)
        mock_ws.accept.assert_called_once()
        mock_session_manager_for_ownership._validate_ownership.assert_called_once_with(
            "test-session", CLIENT_A
        )

    @pytest.mark.asyncio
    async def test_websocket_other_users_session_rejected(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """WebSocket to another user's session is rejected with close code 4403."""
        from app import terminal_websocket_proxy

        # _validate_ownership raises SessionError for wrong owner
        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            side_effect=SessionError("Session test-session not found")
        )

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.close = AsyncMock()
        mock_ws.query_params = {"client_id": CLIENT_B}
        mock_ws.cookies = {}

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership):
            await terminal_websocket_proxy(mock_ws, "test-session")

        # Should be closed with 4403 (forbidden)
        mock_ws.close.assert_called_once()
        close_code = mock_ws.close.call_args[1].get("code") or mock_ws.close.call_args[0][0]
        assert close_code == 4403

    @pytest.mark.asyncio
    async def test_websocket_legacy_session_accessible(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """Legacy session (owner_id=None) is accessible by any authenticated user."""
        from app import terminal_websocket_proxy

        # _validate_ownership returns session (legacy session, no owner restriction)
        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            return_value=_make_mock_session(owner_id=None)
        )

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.query_params = {"client_id": CLIENT_B}
        mock_ws.cookies = {}
        mock_ws.client_state = MagicMock()

        import websockets
        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership), \
             patch("app.checkpoint_manager", None), \
             patch("app.websockets.connect", side_effect=ConnectionRefusedError("test")), \
             patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):
            await terminal_websocket_proxy(mock_ws, "test-session")

        # Connection accepted, not rejected
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_no_client_id_proceeds(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """WebSocket without client_id passes None to _validate_ownership (backward compat)."""
        from app import terminal_websocket_proxy

        # _validate_ownership with None owner_id allows access
        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            return_value=_make_mock_session(owner_id=CLIENT_A)
        )

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.query_params = {}
        mock_ws.cookies = {}
        mock_ws.client_state = MagicMock()

        import websockets
        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership), \
             patch("app.checkpoint_manager", None), \
             patch("app.websockets.connect", side_effect=ConnectionRefusedError("test")), \
             patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):
            await terminal_websocket_proxy(mock_ws, "test-session")

        # Should have called _validate_ownership with None
        mock_session_manager_for_ownership._validate_ownership.assert_called_once_with(
            "test-session", None
        )
        mock_ws.accept.assert_called_once()


# ============================================================================
# HTTP Terminal Proxy Ownership Validation
# ============================================================================

@pytest.mark.unit
class TestTerminalHttpProxyOwnershipValidation:
    """Tests for ownership validation in terminal_http_proxy and terminal_static_proxy."""

    @pytest.fixture
    def mock_terminal_manager(self):
        """Create a mock TerminalManager with a running terminal."""
        tm = MagicMock()
        tm.get_terminal_port.return_value = 9001
        tm.is_terminal_running.return_value = True
        return tm

    @pytest.fixture
    def mock_session_manager_for_ownership(self):
        """Create a mock SessionManager."""
        sm = AsyncMock()
        sm.ensure_terminal_healthy = AsyncMock(return_value=9001)
        return sm

    @pytest.mark.asyncio
    async def test_http_proxy_own_session_ok(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """HTTP proxy to own session proxies content successfully."""
        from app import terminal_http_proxy

        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            return_value=_make_mock_session(owner_id=CLIENT_A)
        )

        mock_request = MagicMock()
        mock_request.cookies = {"ammo_client_id": CLIENT_A}
        mock_request.query_params = {}

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.text = "<html><head></head><body>ttyd</body></html>"
        mock_response.status_code = 200

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership), \
             patch("app.httpx.AsyncClient", return_value=mock_httpx_client):
            response = await terminal_http_proxy("test-session", mock_request)

        assert response.status_code == 200
        mock_session_manager_for_ownership._validate_ownership.assert_called_once_with(
            "test-session", CLIENT_A
        )

    @pytest.mark.asyncio
    async def test_http_proxy_other_users_session_404(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """HTTP proxy to another user's session returns 404."""
        from app import terminal_http_proxy
        from fastapi import HTTPException

        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            side_effect=SessionError("Session test-session not found")
        )

        mock_request = MagicMock()
        mock_request.cookies = {"ammo_client_id": CLIENT_B}
        mock_request.query_params = {}

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership):
            with pytest.raises(HTTPException) as exc_info:
                await terminal_http_proxy("test-session", mock_request)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_http_proxy_no_client_id_proceeds(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """HTTP proxy without client_id passes None to _validate_ownership."""
        from app import terminal_http_proxy

        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            return_value=_make_mock_session(owner_id=CLIENT_A)
        )

        mock_request = MagicMock()
        mock_request.cookies = {}
        mock_request.query_params = {}

        mock_response = MagicMock()
        mock_response.text = "<html><head></head><body>ttyd</body></html>"
        mock_response.status_code = 200

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_response)
        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
        mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership), \
             patch("app.httpx.AsyncClient", return_value=mock_httpx_client):
            response = await terminal_http_proxy("test-session", mock_request)

        mock_session_manager_for_ownership._validate_ownership.assert_called_once_with(
            "test-session", None
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_static_proxy_other_users_session_404(self, mock_terminal_manager, mock_session_manager_for_ownership):
        """Static asset proxy to another user's session returns 404."""
        from app import terminal_static_proxy
        from fastapi import HTTPException

        mock_session_manager_for_ownership._validate_ownership = MagicMock(
            side_effect=SessionError("Session test-session not found")
        )

        mock_request = MagicMock()
        mock_request.cookies = {"ammo_client_id": CLIENT_B}
        mock_request.query_params = {}

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager_for_ownership):
            with pytest.raises(HTTPException) as exc_info:
                await terminal_static_proxy("test-session", "auth_token.js", mock_request)

        assert exc_info.value.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
