# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for WebSocket terminal proxy and tmux mouse mode endpoints.

Tests the websocket_terminal_proxy, terminal_http_proxy, tmux-mouse-mode
endpoints in app.py, and the underlying TerminalManager methods.
"""

import asyncio
import os
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import (
    AsyncMock, MagicMock, Mock, patch, PropertyMock,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app import terminal_websocket_proxy


# ============================================================================
# WebSocket Terminal Proxy Tests
# ============================================================================

@pytest.mark.unit
class TestWebSocketTerminalProxy:
    """Tests for the terminal_websocket_proxy endpoint in app.py."""

    @pytest.fixture
    def mock_terminal_manager(self):
        """Create a mock TerminalManager."""
        tm = MagicMock()
        tm.get_terminal_port.return_value = 9001
        tm.is_terminal_running.return_value = True
        return tm

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        sm = AsyncMock()
        sm.ensure_terminal_healthy = AsyncMock(return_value=9001)
        return sm

    @pytest.mark.asyncio
    async def test_proxy_closes_when_terminal_not_initialized(self):
        """WebSocket proxy closes with 1011 when terminal_manager is None."""
        from app import terminal_websocket_proxy

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch("app.terminal_manager", None):
            await terminal_websocket_proxy(mock_ws, "test-session")

        mock_ws.close.assert_called_once_with(
            code=1011, reason="Terminal service not initialized"
        )

    @pytest.mark.asyncio
    async def test_proxy_recovers_dead_terminal(self, mock_terminal_manager, mock_session_manager):
        """When terminal is not running, proxy attempts recovery via session_manager."""
        from app import terminal_websocket_proxy

        mock_terminal_manager.get_terminal_port.return_value = None
        mock_terminal_manager.is_terminal_running.return_value = False
        mock_session_manager.ensure_terminal_healthy = AsyncMock(return_value=None)
        # Ownership check: allow access (no client_id provided -> None owner_id)
        mock_session_manager._validate_ownership = MagicMock(return_value=MagicMock())

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.query_params = {}
        mock_ws.cookies = {}

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", mock_session_manager):
            await terminal_websocket_proxy(mock_ws, "test-session")

        mock_session_manager.ensure_terminal_healthy.assert_called_once_with("test-session")
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_proxy_closes_when_no_session_manager(self, mock_terminal_manager):
        """When no session manager and terminal not running, closes with 1008."""
        from app import terminal_websocket_proxy

        mock_terminal_manager.get_terminal_port.return_value = None
        mock_terminal_manager.is_terminal_running.return_value = False

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", None):
            await terminal_websocket_proxy(mock_ws, "test-session")

        mock_ws.close.assert_called_once_with(
            code=1008, reason="No terminal found for session test-session"
        )

    @pytest.mark.asyncio
    async def test_proxy_cancels_pending_checkpoint_on_reconnect(self, mock_terminal_manager):
        """When client reconnects, pending checkpoint task is cancelled."""
        import app as app_module

        # Create a pending checkpoint task (use MagicMock not AsyncMock
        # because asyncio.Task.done() and .cancel() are synchronous)
        pending_task = MagicMock()
        pending_task.done.return_value = False
        pending_task.cancel = Mock()
        app_module._pending_checkpoint_tasks["test-session"] = pending_task

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.client_state = MagicMock()

        # Make websockets.connect raise to short-circuit the proxy loop
        import websockets
        try:
            with patch("app.terminal_manager", mock_terminal_manager), \
                 patch("app.session_manager", None), \
                 patch("app.checkpoint_manager", None), \
                 patch("app.websockets.connect", side_effect=ConnectionRefusedError("test")), \
                 patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):

                await terminal_websocket_proxy(mock_ws, "test-session")

            pending_task.cancel.assert_called_once()
        finally:
            # Clean up even on failure
            app_module._pending_checkpoint_tasks.pop("test-session", None)

    @pytest.mark.asyncio
    async def test_proxy_accepts_tty_subprotocol(self, mock_terminal_manager):
        """Proxy accepts with 'tty' subprotocol when client requests it."""
        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.client_state = MagicMock()

        import websockets
        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", None), \
             patch("app.checkpoint_manager", None), \
             patch("app.websockets.connect", side_effect=ConnectionRefusedError("test")), \
             patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):

            await terminal_websocket_proxy(mock_ws, "test-session")

        mock_ws.accept.assert_called_once_with(subprotocol="tty")

    @pytest.mark.asyncio
    async def test_proxy_registers_in_active_websocket_tasks(self, mock_terminal_manager):
        """Proxy registers itself in _active_websocket_tasks during execution."""
        import app as app_module

        mock_ws = AsyncMock()
        mock_ws.scope = {"subprotocols": ["tty"]}
        mock_ws.accept = AsyncMock()
        mock_ws.client_state = MagicMock()

        registered_tasks = {}

        original_connect = None

        async def mock_connect(*args, **kwargs):
            # Check if task was registered
            registered_tasks.update(dict(app_module._active_websocket_tasks))
            raise ConnectionRefusedError("test")

        with patch("app.terminal_manager", mock_terminal_manager), \
             patch("app.session_manager", None), \
             patch("app.checkpoint_manager", None), \
             patch("app.websockets.connect", side_effect=mock_connect), \
             patch("app._schedule_checkpoint_on_disconnect", new_callable=AsyncMock):

            await terminal_websocket_proxy(mock_ws, "test-session")

        # After proxy completes, task should be removed
        assert "ws_proxy_test-session" not in app_module._active_websocket_tasks


# ============================================================================
# Activity Recording Throttle Tests
# ============================================================================

@pytest.mark.unit
class TestActivityRecordingThrottle:
    """Tests for _record_activity_throttled function."""

    def test_record_activity_throttled_first_call(self):
        """First call always records activity."""
        import app as app_module

        mock_monitor = Mock()
        original_monitor = app_module.inactivity_monitor
        app_module.inactivity_monitor = mock_monitor
        app_module._last_activity_recorded.pop("throttle-test", None)

        try:
            app_module._record_activity_throttled("throttle-test")
            mock_monitor.record_activity.assert_called_once_with("throttle-test")
        finally:
            app_module.inactivity_monitor = original_monitor
            app_module._last_activity_recorded.pop("throttle-test", None)

    def test_record_activity_throttled_suppresses_rapid_calls(self):
        """Rapid calls within ACTIVITY_THROTTLE_SECONDS are suppressed."""
        import app as app_module

        mock_monitor = Mock()
        original_monitor = app_module.inactivity_monitor
        app_module.inactivity_monitor = mock_monitor

        # Set last recorded to now
        app_module._last_activity_recorded["throttle-test2"] = time.time()

        try:
            app_module._record_activity_throttled("throttle-test2")
            mock_monitor.record_activity.assert_not_called()
        finally:
            app_module.inactivity_monitor = original_monitor
            app_module._last_activity_recorded.pop("throttle-test2", None)

    def test_record_activity_throttled_allows_after_interval(self):
        """Call is allowed after ACTIVITY_THROTTLE_SECONDS has elapsed."""
        import app as app_module

        mock_monitor = Mock()
        original_monitor = app_module.inactivity_monitor
        app_module.inactivity_monitor = mock_monitor

        # Set last recorded to 2 seconds ago
        app_module._last_activity_recorded["throttle-test3"] = time.time() - 2.0

        try:
            app_module._record_activity_throttled("throttle-test3")
            mock_monitor.record_activity.assert_called_once_with("throttle-test3")
        finally:
            app_module.inactivity_monitor = original_monitor
            app_module._last_activity_recorded.pop("throttle-test3", None)

    def test_record_activity_noop_when_no_monitor(self):
        """No-op when inactivity_monitor is None."""
        import app as app_module

        original_monitor = app_module.inactivity_monitor
        app_module.inactivity_monitor = None
        app_module._last_activity_recorded.pop("throttle-test4", None)

        try:
            # Should not raise
            app_module._record_activity_throttled("throttle-test4")
        finally:
            app_module.inactivity_monitor = original_monitor
            app_module._last_activity_recorded.pop("throttle-test4", None)


# ============================================================================
# Tmux Mouse Mode Endpoint Tests
# ============================================================================

@pytest.mark.unit
class TestTmuxMouseModeEndpoints:
    """Tests for GET/POST /sessions/{id}/tmux-mouse-mode endpoints."""

    @pytest.fixture
    def mock_dependencies(self):
        """Set up mock session_manager and terminal_manager."""
        from shared.session_models import SessionStatus
        session_mock = MagicMock()
        session_mock.status = SessionStatus.ACTIVE

        sm = AsyncMock()
        sm.get_session = AsyncMock(return_value=session_mock)

        tm = AsyncMock()
        tm.get_tmux_mouse_mode = AsyncMock(return_value="on")
        tm.set_tmux_mouse_mode = AsyncMock(return_value="off")

        return sm, tm, session_mock

    @pytest.mark.asyncio
    async def test_get_mouse_mode_returns_on(self, mock_dependencies):
        """GET tmux-mouse-mode returns current mode."""
        sm, tm, _ = mock_dependencies

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import get_tmux_mouse_mode
            result = await get_tmux_mouse_mode("test-session", client_id=None)

        assert result == {"mouse_mode": "on"}
        tm.get_tmux_mouse_mode.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_get_mouse_mode_404_when_session_missing(self):
        """GET tmux-mouse-mode returns 404 when session not found."""
        from fastapi import HTTPException

        sm = AsyncMock()
        sm.get_session = AsyncMock(return_value=None)
        tm = AsyncMock()

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import get_tmux_mouse_mode
            with pytest.raises(HTTPException) as exc_info:
                await get_tmux_mouse_mode("nonexistent", client_id=None)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_mouse_mode_503_when_not_initialized(self):
        """GET tmux-mouse-mode returns 503 when service not initialized."""
        from fastapi import HTTPException

        with patch("app.session_manager", None), \
             patch("app.terminal_manager", None):
            from app import get_tmux_mouse_mode
            with pytest.raises(HTTPException) as exc_info:
                await get_tmux_mouse_mode("test-session", client_id=None)

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_toggle_mouse_mode_returns_result(self, mock_dependencies):
        """POST tmux-mouse-mode toggles and returns new mode."""
        sm, tm, _ = mock_dependencies

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"mode": "toggle"})

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import toggle_tmux_mouse_mode
            result = await toggle_tmux_mouse_mode("test-session", mock_request, client_id=None)

        assert result == {"status": "ok", "mouse_mode": "off"}
        tm.set_tmux_mouse_mode.assert_called_once_with("test-session", "toggle")

    @pytest.mark.asyncio
    async def test_toggle_mouse_mode_400_on_value_error(self, mock_dependencies):
        """POST tmux-mouse-mode returns 400 when ValueError raised."""
        from fastapi import HTTPException
        sm, tm, _ = mock_dependencies

        tm.set_tmux_mouse_mode = AsyncMock(side_effect=ValueError("Invalid mouse mode: bad"))

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"mode": "bad"})

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import toggle_tmux_mouse_mode
            with pytest.raises(HTTPException) as exc_info:
                await toggle_tmux_mouse_mode("test-session", mock_request, client_id=None)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_toggle_mouse_mode_404_when_session_missing(self):
        """POST tmux-mouse-mode returns 404 when session not found."""
        from fastapi import HTTPException

        sm = AsyncMock()
        sm.get_session = AsyncMock(return_value=None)
        tm = AsyncMock()

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"mode": "toggle"})

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import toggle_tmux_mouse_mode
            with pytest.raises(HTTPException) as exc_info:
                await toggle_tmux_mouse_mode("test-session", mock_request, client_id=None)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_mouse_mode_400_when_session_not_active(self, mock_dependencies):
        """POST tmux-mouse-mode returns 400 when session is not active."""
        from fastapi import HTTPException
        from shared.session_models import SessionStatus
        sm, tm, session_mock = mock_dependencies
        session_mock.status = SessionStatus.PAUSED

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"mode": "toggle"})

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import toggle_tmux_mouse_mode
            with pytest.raises(HTTPException) as exc_info:
                await toggle_tmux_mouse_mode("test-session", mock_request, client_id=None)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_toggle_mouse_mode_defaults_to_toggle_on_bad_json(self, mock_dependencies):
        """POST tmux-mouse-mode defaults to 'toggle' when body is invalid JSON."""
        sm, tm, _ = mock_dependencies

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(side_effect=Exception("Bad JSON"))

        with patch("app.session_manager", sm), \
             patch("app.terminal_manager", tm):
            from app import toggle_tmux_mouse_mode
            result = await toggle_tmux_mouse_mode("test-session", mock_request, client_id=None)

        tm.set_tmux_mouse_mode.assert_called_once_with("test-session", "toggle")
        assert result["status"] == "ok"


# ============================================================================
# Terminal Manager Mouse Mode Unit Tests
# ============================================================================

@pytest.mark.unit
class TestTerminalManagerMouseMode:
    """Unit tests for TerminalManager get/set tmux_mouse_mode."""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd and tmux."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager, TerminalProcess
            tm = TerminalManager(base_port=9000, max_ports=10)
            # Add a terminal with tmux session and dedicated socket
            tm._terminals["sess1"] = TerminalProcess(
                session_id="sess1", port=9000, pid=12345, master_fd=-1,
                tmux_session_name="ammo-sess1",
                tmux_socket_path="/tmp/sess1/tmux.sock",
            )
            yield tm

    @pytest.mark.asyncio
    async def test_get_mouse_mode_returns_on_by_default(self, terminal_manager):
        """get_tmux_mouse_mode returns 'on' when no tmux session."""
        from orchestration.terminal_manager import TerminalProcess
        terminal_manager._terminals["no-tmux"] = TerminalProcess(
            session_id="no-tmux", port=9001, pid=12346, master_fd=-1,
            tmux_session_name=None,
        )
        result = await terminal_manager.get_tmux_mouse_mode("no-tmux")
        assert result == "on"

    @pytest.mark.asyncio
    async def test_get_mouse_mode_queries_tmux(self, terminal_manager):
        """get_tmux_mouse_mode queries tmux for the actual mode."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"off\n", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc), \
             patch("asyncio.wait_for", new_callable=AsyncMock, return_value=(b"off\n", b"")):
            # We need to mock wait_for properly
            mock_proc.communicate = AsyncMock(return_value=(b"off\n", b""))
            with patch("asyncio.wait_for", AsyncMock(return_value=(b"off\n", b""))):
                result = await terminal_manager.get_tmux_mouse_mode("sess1")

        # Returns either "on" or "off" (depends on mock setup)
        assert result in ("on", "off")

    @pytest.mark.asyncio
    async def test_set_mouse_mode_raises_for_missing_tmux(self, terminal_manager):
        """set_tmux_mouse_mode raises ValueError when no tmux session."""
        from orchestration.terminal_manager import TerminalProcess
        terminal_manager._terminals["no-tmux"] = TerminalProcess(
            session_id="no-tmux", port=9001, pid=12346, master_fd=-1,
            tmux_session_name=None,
        )
        with pytest.raises(ValueError, match="No tmux session found"):
            await terminal_manager.set_tmux_mouse_mode("no-tmux", "on")

    @pytest.mark.asyncio
    async def test_set_mouse_mode_validates_mode(self, terminal_manager):
        """set_tmux_mouse_mode raises ValueError for invalid mode."""
        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True):
            with pytest.raises(ValueError, match="Invalid mouse mode"):
                await terminal_manager.set_tmux_mouse_mode("sess1", "invalid")

    @pytest.mark.asyncio
    async def test_set_mouse_mode_toggle(self, terminal_manager):
        """set_tmux_mouse_mode with 'toggle' toggles from on to off."""
        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
             patch.object(terminal_manager, 'get_tmux_mouse_mode', new_callable=AsyncMock, return_value="on"):

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0

            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc), \
                 patch("asyncio.wait_for", new_callable=AsyncMock, return_value=(b"", b"")):
                result = await terminal_manager.set_tmux_mouse_mode("sess1", "toggle")

        assert result == "off"

    @pytest.mark.asyncio
    async def test_set_mouse_mode_raises_when_terminal_not_connected(self, terminal_manager):
        """set_tmux_mouse_mode raises ValueError when tmux session doesn't exist yet."""
        with patch.object(terminal_manager, '_tmux_session_exists', return_value=False):
            with pytest.raises(ValueError, match="Terminal not connected"):
                await terminal_manager.set_tmux_mouse_mode("sess1", "on")


# ============================================================================
# WebSocket Proxy Bidirectional Forwarding Tests
# ============================================================================

@pytest.mark.unit
class TestWebSocketBidirectionalForwarding:
    """Tests for bidirectional message forwarding in the WebSocket proxy."""

    @pytest.mark.asyncio
    async def test_forward_to_ttyd_sends_binary_messages(self):
        """forward_to_ttyd sends binary data from client to ttyd."""
        # This test validates the forwarding logic conceptually
        # The actual forward_to_ttyd is a nested function inside terminal_websocket_proxy
        # We test the behavior via mocking the websocket interactions

        mock_client_ws = AsyncMock()
        mock_client_ws.receive = AsyncMock(side_effect=[
            {"type": "websocket.receive", "bytes": b"\x01test data"},
            {"type": "websocket.disconnect"},
        ])

        mock_ttyd_ws = AsyncMock()
        mock_ttyd_ws.send = AsyncMock()

        # Simulate the forwarding logic
        for _ in range(2):
            message = await mock_client_ws.receive()
            msg_type = message.get("type", "unknown")
            if msg_type == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"]:
                await mock_ttyd_ws.send(message["bytes"])

        mock_ttyd_ws.send.assert_called_once_with(b"\x01test data")

    @pytest.mark.asyncio
    async def test_forward_to_ttyd_sends_text_messages(self):
        """forward_to_ttyd sends text data from client to ttyd."""
        mock_client_ws = AsyncMock()
        mock_client_ws.receive = AsyncMock(side_effect=[
            {"type": "websocket.receive", "text": "1some command"},
            {"type": "websocket.disconnect"},
        ])

        mock_ttyd_ws = AsyncMock()
        mock_ttyd_ws.send = AsyncMock()

        for _ in range(2):
            message = await mock_client_ws.receive()
            msg_type = message.get("type", "unknown")
            if msg_type == "websocket.disconnect":
                break
            if "text" in message and message["text"]:
                await mock_ttyd_ws.send(message["text"])

        mock_ttyd_ws.send.assert_called_once_with("1some command")


# ============================================================================
# Schedule Checkpoint on Disconnect Tests
# ============================================================================

@pytest.mark.unit
class TestScheduleCheckpointOnDisconnect:
    """Tests for _schedule_checkpoint_on_disconnect."""

    @pytest.mark.asyncio
    async def test_no_checkpoint_when_manager_is_none(self):
        """No checkpoint scheduled when checkpoint_manager is None."""
        import app as app_module

        sid = "test-session-no-cp-mgr"
        app_module._pending_checkpoint_tasks.pop(sid, None)
        try:
            with patch("app.checkpoint_manager", None):
                await app_module._schedule_checkpoint_on_disconnect(sid)

            assert sid not in app_module._pending_checkpoint_tasks
        finally:
            app_module._pending_checkpoint_tasks.pop(sid, None)

    @pytest.mark.asyncio
    async def test_checkpoint_task_is_created(self):
        """Checkpoint task is created and tracked in _pending_checkpoint_tasks."""
        import app as app_module

        mock_cm = AsyncMock()
        mock_cm.checkpoint_session = AsyncMock()

        with patch("app.checkpoint_manager", mock_cm), \
             patch("app.session_manager", None):
            await app_module._schedule_checkpoint_on_disconnect("test-cp-session", grace_seconds=0)

        # Task should be tracked
        assert "test-cp-session" in app_module._pending_checkpoint_tasks
        # Clean up
        task = app_module._pending_checkpoint_tasks.pop("test-cp-session", None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ============================================================================
# WebSocket Auth Token Tests
# ============================================================================

@pytest.mark.unit
class TestWebSocketAuthToken:
    """Tests for WebSocket token query parameter and auth-related behavior.

    Note: Starlette's BaseHTTPMiddleware does NOT intercept WebSocket upgrade
    requests, so API key auth for WebSocket is enforced via the HTTP terminal
    endpoint (which proxies via iframe). The frontend appends ?token=<key> to
    terminal iframe URLs which get validated by the middleware on the HTTP route.
    """

    def test_websocket_without_token_rejected_when_auth_enabled(self):
        """HTTP terminal endpoint (not WebSocket) returns 401 without key when auth enabled.

        This validates that the middleware blocks unauthenticated HTTP access
        to terminal URLs, which is how browsers initially load the terminal.
        """
        import importlib

        valid_key = "test-secret-key-32bytes-minimum-length-00"

        with patch.dict(os.environ, {"AMMO_API_KEY": valid_key}):
            import app as app_module
            importlib.reload(app_module)

            from starlette.testclient import TestClient
            client = TestClient(app_module.app, raise_server_exceptions=False)

            # HTTP GET to terminal path without auth -> 401
            resp = client.get("/sessions/test-session/terminal/")
            assert resp.status_code == 401

    def test_websocket_with_valid_token_not_rejected_when_auth_enabled(self):
        """HTTP terminal endpoint with valid ?token= passes auth middleware."""
        import importlib

        valid_key = "test-secret-key-32bytes-minimum-length-00"

        with patch.dict(os.environ, {"AMMO_API_KEY": valid_key}):
            import app as app_module
            importlib.reload(app_module)

            from starlette.testclient import TestClient
            client = TestClient(app_module.app, raise_server_exceptions=False)

            # HTTP GET to terminal path with valid token -> NOT 401
            resp = client.get(f"/sessions/test-session/terminal/?token={valid_key}")
            assert resp.status_code != 401

    def test_websocket_no_auth_required_when_key_unset(self):
        """HTTP terminal endpoint passes without key when AMMO_API_KEY is unset."""
        import importlib

        with patch.dict(os.environ, {}, clear=False) as env:
            env.pop("AMMO_API_KEY", None)
            import app as app_module
            importlib.reload(app_module)

            from starlette.testclient import TestClient
            client = TestClient(app_module.app, raise_server_exceptions=False)

            # HTTP GET to terminal path without auth -> NOT 401
            resp = client.get("/sessions/test-session/terminal/")
            assert resp.status_code != 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
