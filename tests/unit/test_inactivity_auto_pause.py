# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for InactivityMonitor <-> SessionManager auto-pause.

Tests the integration between a real InactivityMonitor and a (partially)
mocked SessionManager - specifically what happens when the timeout callback
fires.
"""

import asyncio
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    make_create_request,
)
from shared.session_models import SessionStatus


# ============================================================================
# Helpers
# ============================================================================

def _make_monitor(timeout_mins=1, check_interval=1):
    from orchestration.inactivity_monitor import InactivityMonitor
    return InactivityMonitor(
        default_timeout_mins=timeout_mins,
        check_interval_seconds=check_interval,
    )


# ============================================================================
# Auto-Pause Integration
# ============================================================================

@pytest.mark.unit
class TestAutoPauseIntegration:
    """Tests that wire a real InactivityMonitor to a mock SessionManager."""

    @pytest.mark.asyncio
    async def test_auto_pause_calls_session_manager_pause(self):
        """When session times out, the pause callback (session_manager.pause_session) is invoked."""
        monitor = _make_monitor()
        pause_mock = AsyncMock()
        monitor.set_pause_callback(pause_mock)

        monitor.register_session("sess-a", timeout_mins=1)
        # Simulate session that has been idle for 2 minutes
        monitor._sessions["sess-a"].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        pause_mock.assert_called_once_with("sess-a")

    @pytest.mark.asyncio
    async def test_auto_pause_releases_gpus(self, mock_session_manager):
        """After auto-pause fires, GPU allocations for the session are cleared."""
        from shared.gpu_resource_manager import GPUAllocation

        session_id = "sess-gpu-test"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        mock_session_manager._sessions[session_id] = state

        # Manually register a GPU allocation so we can verify it is released.
        # Key format is "session:{session_id}:{gpu_id}" (see _allocation_key).
        gpu_manager = mock_session_manager.gpu_manager
        alloc_key = f"session:{session_id}:0"
        gpu_manager._allocations[alloc_key] = GPUAllocation(
            gpu_id=0,
            allocation_id=session_id,
            allocation_type="session",
            acquired_at=time.time(),
        )
        assert alloc_key in gpu_manager._allocations

        # Wire inactivity monitor to session_manager.pause_session
        monitor = _make_monitor()
        async def pause_callback(sid):
            await mock_session_manager.pause_session(sid)
        monitor.set_pause_callback(pause_callback)
        monitor.register_session(session_id, timeout_mins=1)
        monitor._sessions[session_id].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        # GPU allocation should have been released
        assert alloc_key not in gpu_manager._allocations

    @pytest.mark.asyncio
    async def test_auto_pause_stops_terminal(self, mock_session_manager):
        """When auto-pause fires, terminal_manager.stop_terminal is called."""
        session_id = "sess-terminal-test"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
        )
        # Give state a terminal_port so stop_terminal is invoked
        state.terminal_port = 9001
        mock_session_manager._sessions[session_id] = state

        monitor = _make_monitor()
        async def pause_callback(sid):
            await mock_session_manager.pause_session(sid)
        monitor.set_pause_callback(pause_callback)
        monitor.register_session(session_id, timeout_mins=1)
        monitor._sessions[session_id].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        mock_session_manager.terminal_manager.stop_terminal.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_auto_pause_unregisters_from_monitor(self):
        """After auto-pause, session is removed from monitor._sessions."""
        monitor = _make_monitor()
        pause_mock = AsyncMock()
        monitor.set_pause_callback(pause_mock)

        monitor.register_session("sess-b", timeout_mins=1)
        monitor._sessions["sess-b"].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        assert "sess-b" not in monitor._sessions


# ============================================================================
# Auto-Pause Resilience
# ============================================================================

@pytest.mark.unit
class TestAutoPauseResilience:
    """Tests for monitor resilience when pause callbacks fail or activity happens."""

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_stop_monitor(self):
        """If pause_session raises for one session, other sessions are still checked."""
        monitor = _make_monitor()

        call_order = []

        async def flaky_pause(sid):
            call_order.append(sid)
            if sid == "sess-fail":
                raise RuntimeError("pause failed")

        monitor.set_pause_callback(flaky_pause)

        monitor.register_session("sess-fail", timeout_mins=1)
        monitor.register_session("sess-ok", timeout_mins=1)
        monitor._sessions["sess-fail"].last_activity = time.time() - 120
        monitor._sessions["sess-ok"].last_activity = time.time() - 120

        # Should not raise despite sess-fail failing
        await monitor._check_inactive_sessions()

        # Both sessions should have had callback attempted
        assert "sess-fail" in call_order
        assert "sess-ok" in call_order

    @pytest.mark.asyncio
    async def test_websocket_activity_resets_timeout(self):
        """record_activity() before check prevents auto-pause."""
        monitor = _make_monitor()
        pause_mock = AsyncMock()
        monitor.set_pause_callback(pause_mock)

        monitor.register_session("sess-active", timeout_mins=1)
        # Set last_activity to nearly expired
        monitor._sessions["sess-active"].last_activity = time.time() - 55

        # Simulate user WebSocket activity
        monitor.record_activity("sess-active")

        await monitor._check_inactive_sessions()

        pause_mock.assert_not_called()
        assert "sess-active" in monitor._sessions

    @pytest.mark.asyncio
    async def test_multiple_sessions_different_timeouts(self):
        """Short-timeout session A pauses while long-timeout session B stays active."""
        monitor = _make_monitor()
        paused = []

        async def pause_cb(sid):
            paused.append(sid)

        monitor.set_pause_callback(pause_cb)

        monitor.register_session("sess-short", timeout_mins=1)
        monitor.register_session("sess-long", timeout_mins=30)

        # Short session expired 2 minutes ago
        monitor._sessions["sess-short"].last_activity = time.time() - 120
        # Long session active recently
        monitor._sessions["sess-long"].last_activity = time.time() - 60

        await monitor._check_inactive_sessions()

        assert "sess-short" in paused
        assert "sess-long" not in paused
        assert "sess-long" in monitor._sessions
        assert "sess-short" not in monitor._sessions

    @pytest.mark.asyncio
    async def test_manual_pause_then_no_auto_pause(self, mock_session_manager):
        """After manual pause, session is unregistered from monitor -> no double-pause."""
        session_id = "sess-manual"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        state.terminal_port = 9001
        mock_session_manager._sessions[session_id] = state

        monitor = _make_monitor()
        mock_session_manager.inactivity_monitor = monitor
        async def pause_callback(sid):
            await mock_session_manager.pause_session(sid)
        monitor.set_pause_callback(pause_callback)
        monitor.register_session(session_id, timeout_mins=1)
        # Session is nearly expired
        monitor._sessions[session_id].last_activity = time.time() - 55

        # Manual pause: unregisters from monitor
        await mock_session_manager.pause_session(session_id)

        # Now the session should NOT be in monitor._sessions
        assert session_id not in monitor._sessions

        # Even if we run a check, no double-pause
        pause_call_count = mock_session_manager.terminal_manager.stop_terminal.call_count
        await monitor._check_inactive_sessions()
        assert mock_session_manager.terminal_manager.stop_terminal.call_count == pause_call_count
