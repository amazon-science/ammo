# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for InactivityMonitor.

Tests the auto-pause functionality:
- record_activity() resets timer
- Timeout fires callback when no activity
- Activity before timeout prevents callback
- stop() cancels monitoring loop cleanly
"""

import asyncio
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.unit
class TestInactivityMonitorBasics:
    """Basic tests for InactivityMonitor initialization and registration."""

    @pytest.fixture
    def monitor(self):
        """Create an InactivityMonitor with short intervals for testing."""
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(
            default_timeout_mins=1,
            check_interval_seconds=1,
        )

    def test_init_defaults(self, monitor):
        """Monitor initializes with correct defaults."""
        assert monitor.default_timeout_mins == 1
        assert monitor.check_interval == 1
        assert monitor._running is False
        assert monitor._monitor_task is None
        assert len(monitor._sessions) == 0

    def test_register_session(self, monitor):
        """register_session adds session to monitoring."""
        monitor.register_session("sess1")
        assert "sess1" in monitor._sessions
        assert monitor._sessions["sess1"].timeout_mins == 1

    def test_register_session_custom_timeout(self, monitor):
        """register_session with custom timeout overrides default."""
        monitor.register_session("sess1", timeout_mins=5)
        assert monitor._sessions["sess1"].timeout_mins == 5

    def test_unregister_session(self, monitor):
        """unregister_session removes session from monitoring."""
        monitor.register_session("sess1")
        monitor.unregister_session("sess1")
        assert "sess1" not in monitor._sessions

    def test_unregister_nonexistent_session_is_noop(self, monitor):
        """unregister_session for unknown session does not raise."""
        monitor.unregister_session("nonexistent")  # Should not raise


@pytest.mark.unit
class TestRecordActivity:
    """Tests for record_activity() resetting the timer."""

    @pytest.fixture
    def monitor(self):
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(default_timeout_mins=1, check_interval_seconds=1)

    def test_record_activity_resets_timer(self, monitor):
        """record_activity() updates last_activity timestamp."""
        monitor.register_session("sess1")
        original_time = monitor._sessions["sess1"].last_activity

        # Advance time slightly
        time.sleep(0.01)
        monitor.record_activity("sess1")
        new_time = monitor._sessions["sess1"].last_activity

        assert new_time > original_time

    def test_record_activity_unknown_session_is_noop(self, monitor):
        """record_activity() for unknown session does not raise."""
        monitor.record_activity("nonexistent")  # Should not raise

    def test_record_activity_extends_timeout(self, monitor):
        """record_activity() prevents timeout by resetting last_activity."""
        monitor.register_session("sess1", timeout_mins=1)

        # Simulate activity
        monitor.record_activity("sess1")

        remaining = monitor.get_time_until_timeout("sess1")
        assert remaining is not None
        assert remaining > 50  # Should be close to full 60 seconds


@pytest.mark.unit
class TestTimeoutDetection:
    """Tests for timeout detection and callback firing."""

    @pytest.fixture
    def monitor(self):
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(default_timeout_mins=1, check_interval_seconds=1)

    def test_is_session_timed_out_false_when_active(self, monitor):
        """is_session_timed_out returns False for recently active session."""
        monitor.register_session("sess1")
        assert monitor.is_session_timed_out("sess1") is False

    def test_is_session_timed_out_true_when_expired(self, monitor):
        """is_session_timed_out returns True when timeout exceeded."""
        monitor.register_session("sess1", timeout_mins=1)
        # Manually set last_activity to 2 minutes ago
        monitor._sessions["sess1"].last_activity = time.time() - 120
        assert monitor.is_session_timed_out("sess1") is True

    def test_get_time_until_timeout_returns_none_for_unknown(self, monitor):
        """get_time_until_timeout returns None for unregistered session."""
        assert monitor.get_time_until_timeout("nonexistent") is None

    def test_get_time_until_timeout_returns_zero_when_expired(self, monitor):
        """get_time_until_timeout returns 0 when already expired."""
        monitor.register_session("sess1", timeout_mins=1)
        monitor._sessions["sess1"].last_activity = time.time() - 120
        remaining = monitor.get_time_until_timeout("sess1")
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_timeout_fires_callback(self, monitor):
        """When session times out, pause_callback is called."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)

        # Simulate expired session
        monitor._sessions["sess1"].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        callback.assert_called_once_with("sess1")

    @pytest.mark.asyncio
    async def test_timeout_unregisters_session(self, monitor):
        """After timeout, session is unregistered from monitoring."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)

        monitor._sessions["sess1"].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        assert "sess1" not in monitor._sessions

    @pytest.mark.asyncio
    async def test_no_callback_when_not_expired(self, monitor):
        """Callback is NOT fired when session has not timed out."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=30)

        await monitor._check_inactive_sessions()

        callback.assert_not_called()
        assert "sess1" in monitor._sessions

    @pytest.mark.asyncio
    async def test_activity_before_timeout_prevents_callback(self, monitor):
        """Recording activity before timeout prevents callback from firing."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)

        # Session was about to expire
        monitor._sessions["sess1"].last_activity = time.time() - 55

        # Record activity (resets timer)
        monitor.record_activity("sess1")

        await monitor._check_inactive_sessions()

        callback.assert_not_called()
        assert "sess1" in monitor._sessions

    @pytest.mark.asyncio
    async def test_timeout_without_callback_logs_warning(self, monitor):
        """When no callback is set, timeout logs a warning but doesn't crash."""
        monitor.register_session("sess1", timeout_mins=1)
        monitor._sessions["sess1"].last_activity = time.time() - 120

        # No callback set
        await monitor._check_inactive_sessions()

        # Session should still be unregistered
        assert "sess1" not in monitor._sessions

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_monitor(self, monitor):
        """Exception in callback is caught and does not crash the monitor."""
        callback = AsyncMock(side_effect=RuntimeError("callback failed"))
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)
        monitor._sessions["sess1"].last_activity = time.time() - 120

        # Should not raise
        await monitor._check_inactive_sessions()

        callback.assert_called_once_with("sess1")

    @pytest.mark.asyncio
    async def test_multiple_sessions_timeout_independently(self, monitor):
        """Multiple sessions can timeout independently."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)

        monitor.register_session("sess1", timeout_mins=1)
        monitor.register_session("sess2", timeout_mins=1)
        monitor.register_session("sess3", timeout_mins=30)

        # Only sess1 and sess2 are expired
        monitor._sessions["sess1"].last_activity = time.time() - 120
        monitor._sessions["sess2"].last_activity = time.time() - 120

        await monitor._check_inactive_sessions()

        assert callback.call_count == 2
        assert "sess3" in monitor._sessions


@pytest.mark.unit
class TestMonitorLifecycle:
    """Tests for start/stop lifecycle of the monitoring loop."""

    @pytest.fixture
    def monitor(self):
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(default_timeout_mins=1, check_interval_seconds=1)

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, monitor):
        """start() sets _running to True and creates task."""
        monitor.start()
        try:
            assert monitor._running is True
            assert monitor._monitor_task is not None
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, monitor):
        """Calling start() twice does not create duplicate tasks."""
        monitor.start()
        first_task = monitor._monitor_task
        monitor.start()
        second_task = monitor._monitor_task

        try:
            assert first_task is second_task
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_monitor_loop(self, monitor):
        """stop() cancels the monitoring loop cleanly."""
        monitor.start()
        assert monitor._running is True

        await monitor.stop()

        assert monitor._running is False
        assert monitor._monitor_task is not None  # Task exists but is done
        assert monitor._monitor_task.done()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, monitor):
        """Calling stop() when not running is a no-op."""
        await monitor.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_monitor_loop_checks_sessions(self, monitor):
        """Monitor loop calls _check_inactive_sessions periodically."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)
        monitor._sessions["sess1"].last_activity = time.time() - 120

        # Use very short check interval
        monitor.check_interval = 0.1
        monitor.start()

        # Wait for at least one check cycle
        await asyncio.sleep(0.3)

        await monitor.stop()

        # Callback should have been called
        callback.assert_called_once_with("sess1")


@pytest.mark.unit
class TestChildProcessTracking:
    """Tests for child process registration and cleanup."""

    @pytest.fixture
    def monitor(self):
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(default_timeout_mins=1, check_interval_seconds=1)

    def test_register_child_process(self, monitor):
        """register_child_process adds PID to session's child_pids."""
        monitor.register_session("sess1")
        monitor.register_child_process("sess1", 12345)
        assert 12345 in monitor._sessions["sess1"].child_pids

    def test_unregister_child_process(self, monitor):
        """unregister_child_process removes PID from session's child_pids."""
        monitor.register_session("sess1")
        monitor.register_child_process("sess1", 12345)
        monitor.unregister_child_process("sess1", 12345)
        assert 12345 not in monitor._sessions["sess1"].child_pids

    def test_unregister_nonexistent_pid_is_noop(self, monitor):
        """unregister_child_process for unknown PID does not raise."""
        monitor.register_session("sess1")
        monitor.unregister_child_process("sess1", 99999)  # Should not raise

    @pytest.mark.asyncio
    async def test_timeout_kills_child_processes(self, monitor):
        """On timeout, tracked child processes receive SIGTERM."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)
        monitor.register_child_process("sess1", 12345)
        monitor.register_child_process("sess1", 12346)
        monitor._sessions["sess1"].last_activity = time.time() - 120

        with patch("os.kill") as mock_kill:
            await monitor._check_inactive_sessions()

        # Should have sent SIGTERM to both child processes
        import signal
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        mock_kill.assert_any_call(12346, signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_kill_child_handles_process_not_found(self, monitor):
        """Killing already-dead child processes does not raise."""
        callback = AsyncMock()
        monitor.set_pause_callback(callback)
        monitor.register_session("sess1", timeout_mins=1)
        monitor.register_child_process("sess1", 99999)
        monitor._sessions["sess1"].last_activity = time.time() - 120

        with patch("os.kill", side_effect=ProcessLookupError):
            # Should not raise
            await monitor._check_inactive_sessions()

    def test_unregister_session_kills_child_processes(self, monitor):
        """unregister_session kills tracked child processes."""
        monitor.register_session("sess1")
        monitor.register_child_process("sess1", 12345)

        with patch("os.kill") as mock_kill:
            monitor.unregister_session("sess1")

        import signal
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)


@pytest.mark.unit
class TestSessionActivityInfo:
    """Tests for get_session_activity and get_all_activity."""

    @pytest.fixture
    def monitor(self):
        from orchestration.inactivity_monitor import InactivityMonitor
        return InactivityMonitor(default_timeout_mins=1, check_interval_seconds=1)

    def test_get_session_activity_returns_info(self, monitor):
        """get_session_activity returns activity dict for registered session."""
        monitor.register_session("sess1", timeout_mins=5)
        info = monitor.get_session_activity("sess1")
        assert info is not None
        assert info["session_id"] == "sess1"
        assert info["timeout_mins"] == 5
        assert "time_until_timeout_seconds" in info
        assert "child_pids" in info

    def test_get_session_activity_returns_none_for_unknown(self, monitor):
        """get_session_activity returns None for unregistered session."""
        assert monitor.get_session_activity("nonexistent") is None

    def test_get_all_activity(self, monitor):
        """get_all_activity returns info for all sessions."""
        monitor.register_session("sess1")
        monitor.register_session("sess2")
        all_info = monitor.get_all_activity()
        assert len(all_info) == 2
        assert "sess1" in all_info
        assert "sess2" in all_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
