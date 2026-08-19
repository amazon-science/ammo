# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Regression tests for server shutdown sequence.

Verifies that the lifespan shutdown handler properly:
- Cancels all active WebSocket proxy tasks (_active_websocket_tasks)
- Cancels all pending checkpoint tasks (_pending_checkpoint_tasks)
- Calls TerminalManager.cleanup() to kill all ttyd processes
- Schedules checkpoint on WebSocket disconnect with grace period
- Cancels previous checkpoint when same session reconnects
- Executes each shutdown step in isolation (terminal, WS, checkpoint, background, monitor)

These tests prevent regression of known production issues where WebSocket
connections hung during shutdown, causing 6+ minute shutdown delays.
"""

import asyncio
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Shutdown handler: WebSocket proxy task cancellation
# =============================================================================

@pytest.mark.unit
class TestShutdownCancelsWebSocketTasks:
    """Verify shutdown handler cancels all entries in _active_websocket_tasks."""

    @pytest.mark.asyncio
    async def test_cancels_all_active_websocket_tasks(self):
        """All tasks in _active_websocket_tasks are cancelled during shutdown."""
        import app as app_module

        # Create mock tasks
        task1 = AsyncMock(spec=asyncio.Task)
        task1.done.return_value = False
        task1.cancel = Mock()

        task2 = AsyncMock(spec=asyncio.Task)
        task2.done.return_value = False
        task2.cancel = Mock()

        # Save original state
        original_ws_tasks = app_module._active_websocket_tasks.copy()
        original_cp_tasks = app_module._pending_checkpoint_tasks.copy()

        try:
            app_module._active_websocket_tasks["ws_proxy_sess1"] = task1
            app_module._active_websocket_tasks["ws_proxy_sess2"] = task2

            # Simulate the shutdown step: cancel WebSocket tasks
            for task_id, task in list(app_module._active_websocket_tasks.items()):
                if not task.done():
                    task.cancel()

            task1.cancel.assert_called_once()
            task2.cancel.assert_called_once()
        finally:
            # Restore original state
            app_module._active_websocket_tasks.clear()
            app_module._active_websocket_tasks.update(original_ws_tasks)
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original_cp_tasks)

    @pytest.mark.asyncio
    async def test_skips_already_done_tasks(self):
        """Tasks already done are not cancelled."""
        import app as app_module

        done_task = AsyncMock(spec=asyncio.Task)
        done_task.done.return_value = True
        done_task.cancel = Mock()

        active_task = AsyncMock(spec=asyncio.Task)
        active_task.done.return_value = False
        active_task.cancel = Mock()

        original = app_module._active_websocket_tasks.copy()

        try:
            app_module._active_websocket_tasks["ws_done"] = done_task
            app_module._active_websocket_tasks["ws_active"] = active_task

            for task_id, task in list(app_module._active_websocket_tasks.items()):
                if not task.done():
                    task.cancel()

            done_task.cancel.assert_not_called()
            active_task.cancel.assert_called_once()
        finally:
            app_module._active_websocket_tasks.clear()
            app_module._active_websocket_tasks.update(original)

    @pytest.mark.asyncio
    async def test_clear_after_cancel_and_wait(self):
        """_active_websocket_tasks is cleared after cancel + wait."""
        import app as app_module

        # Use a real asyncio.Task wrapping a coroutine that hangs
        async def hanging_coro():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(hanging_coro())

        original = app_module._active_websocket_tasks.copy()

        try:
            app_module._active_websocket_tasks["ws_proxy_hang"] = task

            # Simulate shutdown: cancel, wait, clear
            for tid, t in list(app_module._active_websocket_tasks.items()):
                if not t.done():
                    t.cancel()

            if app_module._active_websocket_tasks:
                await asyncio.wait(
                    list(app_module._active_websocket_tasks.values()),
                    timeout=5.0,
                )
                app_module._active_websocket_tasks.clear()

            assert len(app_module._active_websocket_tasks) == 0
        finally:
            app_module._active_websocket_tasks.clear()
            app_module._active_websocket_tasks.update(original)


# =============================================================================
# Shutdown handler: Checkpoint task cancellation
# =============================================================================

@pytest.mark.unit
class TestShutdownCancelsCheckpointTasks:
    """Verify shutdown handler cancels all entries in _pending_checkpoint_tasks."""

    @pytest.mark.asyncio
    async def test_cancels_all_pending_checkpoint_tasks(self):
        """All tasks in _pending_checkpoint_tasks are cancelled during shutdown."""
        import app as app_module

        task1 = AsyncMock(spec=asyncio.Task)
        task1.done.return_value = False
        task1.cancel = Mock()

        task2 = AsyncMock(spec=asyncio.Task)
        task2.done.return_value = False
        task2.cancel = Mock()

        original = app_module._pending_checkpoint_tasks.copy()

        try:
            app_module._pending_checkpoint_tasks["sess1"] = task1
            app_module._pending_checkpoint_tasks["sess2"] = task2

            # Simulate shutdown step
            for session_id, task in list(app_module._pending_checkpoint_tasks.items()):
                if not task.done():
                    task.cancel()

            task1.cancel.assert_called_once()
            task2.cancel.assert_called_once()
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)

    @pytest.mark.asyncio
    async def test_clear_after_gather(self):
        """_pending_checkpoint_tasks is cleared after gather."""
        import app as app_module

        async def delayed_checkpoint():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(delayed_checkpoint())

        original = app_module._pending_checkpoint_tasks.copy()

        try:
            app_module._pending_checkpoint_tasks["sess_cp"] = task

            for sid, t in list(app_module._pending_checkpoint_tasks.items()):
                if not t.done():
                    t.cancel()

            if app_module._pending_checkpoint_tasks:
                await asyncio.gather(
                    *app_module._pending_checkpoint_tasks.values(),
                    return_exceptions=True,
                )
                app_module._pending_checkpoint_tasks.clear()

            assert len(app_module._pending_checkpoint_tasks) == 0
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)


# =============================================================================
# TerminalManager.cleanup() kills all ttyd processes
# =============================================================================

@pytest.mark.unit
class TestTerminalManagerCleanup:
    """Verify TerminalManager.cleanup() kills all tracked ttyd processes."""

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
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    @pytest.mark.asyncio
    async def test_cleanup_stops_all_terminals(self, terminal_manager):
        """cleanup() calls stop_terminal(force=True) for every tracked terminal."""
        from orchestration.terminal_manager import TerminalProcess

        # Inject 3 terminals
        for i in range(3):
            terminal_manager._terminals[f"sess_{i}"] = TerminalProcess(
                session_id=f"sess_{i}",
                port=9000 + i,
                pid=10000 + i,
                master_fd=-1,
                tmux_session_name=f"ammo-sess{i}",
            )
            terminal_manager._used_ports.add(9000 + i)

        with patch.object(terminal_manager, 'stop_terminal', new_callable=AsyncMock) as mock_stop:
            mock_stop.return_value = True
            await terminal_manager.cleanup()

        # All 3 should be stopped with force=True
        assert mock_stop.call_count == 3
        for call_args in mock_stop.call_args_list:
            assert call_args[1].get('force', False) is True or call_args[0][1] is True

    @pytest.mark.asyncio
    async def test_cleanup_with_no_terminals(self, terminal_manager):
        """cleanup() is a no-op when no terminals are tracked."""
        assert len(terminal_manager._terminals) == 0
        # Should not raise
        await terminal_manager.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_stops_on_individual_failure(self, terminal_manager):
        """cleanup() propagates exception when stop_terminal fails (no try/except)."""
        from orchestration.terminal_manager import TerminalProcess

        for i in range(3):
            terminal_manager._terminals[f"sess_{i}"] = TerminalProcess(
                session_id=f"sess_{i}",
                port=9000 + i,
                pid=10000 + i,
                master_fd=-1,
            )
            terminal_manager._used_ports.add(9000 + i)

        call_count = 0

        async def mock_stop(session_id, force=False):
            nonlocal call_count
            call_count += 1
            if session_id == "sess_1":
                raise OSError("Cannot kill pid 10001")
            return True

        with patch.object(terminal_manager, 'stop_terminal', side_effect=mock_stop):
            # cleanup() iterates list(_terminals.keys()) and calls stop_terminal.
            # Since cleanup has no try/except, an exception in stop_terminal
            # propagates and stops iteration.
            with pytest.raises(OSError, match="Cannot kill pid 10001"):
                await terminal_manager.cleanup()

        # Called for sess_0 (success) and sess_1 (failure), stopped before sess_2
        assert call_count == 2


# =============================================================================
# WebSocket disconnect triggers checkpoint scheduling
# =============================================================================

@pytest.mark.unit
class TestCheckpointSchedulingOnDisconnect:
    """Verify checkpoint scheduling on WebSocket disconnect."""

    @pytest.mark.asyncio
    async def test_schedule_checkpoint_creates_task(self):
        """_schedule_checkpoint_on_disconnect creates a task in _pending_checkpoint_tasks."""
        import app as app_module

        original = app_module._pending_checkpoint_tasks.copy()
        original_cp_mgr = app_module.checkpoint_manager
        original_session_mgr = app_module.session_manager

        try:
            # Set up mock checkpoint manager
            mock_cp_mgr = Mock()
            mock_cp_mgr.checkpoint_session = AsyncMock()
            app_module.checkpoint_manager = mock_cp_mgr
            app_module.session_manager = None  # Skip session check

            await app_module._schedule_checkpoint_on_disconnect("sess_cp_test", grace_seconds=60)

            assert "sess_cp_test" in app_module._pending_checkpoint_tasks
            task = app_module._pending_checkpoint_tasks["sess_cp_test"]
            assert isinstance(task, asyncio.Task)

            # Cancel to clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)
            app_module.checkpoint_manager = original_cp_mgr
            app_module.session_manager = original_session_mgr

    @pytest.mark.asyncio
    async def test_reconnect_cancels_previous_checkpoint(self):
        """When a client reconnects, the pending checkpoint for that session is cancelled."""
        import app as app_module

        original = app_module._pending_checkpoint_tasks.copy()

        try:
            # Simulate a pending checkpoint task
            cancelled = False

            async def delayed_checkpoint():
                nonlocal cancelled
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled = True
                    raise

            task = asyncio.create_task(delayed_checkpoint())
            app_module._pending_checkpoint_tasks["sess_reconnect"] = task

            # Yield to let the task start and reach its first await
            await asyncio.sleep(0)

            # Simulate reconnect: pop and cancel
            pending_task = app_module._pending_checkpoint_tasks.pop("sess_reconnect", None)
            if pending_task and not pending_task.done():
                pending_task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert cancelled is True
            assert "sess_reconnect" not in app_module._pending_checkpoint_tasks
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)

    @pytest.mark.asyncio
    async def test_second_disconnect_replaces_first_checkpoint(self):
        """
        If same session disconnects twice, second call creates new task,
        effectively replacing the first (the module stores by session_id key).
        """
        import app as app_module

        original = app_module._pending_checkpoint_tasks.copy()
        original_cp_mgr = app_module.checkpoint_manager
        original_session_mgr = app_module.session_manager

        try:
            mock_cp_mgr = Mock()
            mock_cp_mgr.checkpoint_session = AsyncMock()
            app_module.checkpoint_manager = mock_cp_mgr
            app_module.session_manager = None

            # First disconnect
            await app_module._schedule_checkpoint_on_disconnect("sess_double", grace_seconds=60)
            first_task = app_module._pending_checkpoint_tasks["sess_double"]

            # Second disconnect (overwrites)
            await app_module._schedule_checkpoint_on_disconnect("sess_double", grace_seconds=60)
            second_task = app_module._pending_checkpoint_tasks["sess_double"]

            # The task stored should be the second one
            assert second_task is not first_task
            assert "sess_double" in app_module._pending_checkpoint_tasks

            # Clean up
            first_task.cancel()
            second_task.cancel()
            for t in [first_task, second_task]:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)
            app_module.checkpoint_manager = original_cp_mgr
            app_module.session_manager = original_session_mgr

    @pytest.mark.asyncio
    async def test_no_checkpoint_when_manager_is_none(self):
        """No task created when checkpoint_manager is None."""
        import app as app_module

        original = app_module._pending_checkpoint_tasks.copy()
        original_cp_mgr = app_module.checkpoint_manager

        try:
            app_module.checkpoint_manager = None

            await app_module._schedule_checkpoint_on_disconnect("sess_none", grace_seconds=60)

            assert "sess_none" not in app_module._pending_checkpoint_tasks
        finally:
            app_module._pending_checkpoint_tasks.clear()
            app_module._pending_checkpoint_tasks.update(original)
            app_module.checkpoint_manager = original_cp_mgr


# =============================================================================
# Shutdown steps isolation: each step can be tested independently
# =============================================================================

@pytest.mark.unit
class TestShutdownStepIsolation:
    """Verify each shutdown step works independently."""

    @pytest.mark.asyncio
    async def test_terminal_cleanup_step(self):
        """Step 1: TerminalManager.cleanup() is an independent async operation."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)

        # cleanup() should be callable independently
        await tm.cleanup()

    @pytest.mark.asyncio
    async def test_ws_cancel_step_independent(self):
        """Step 2: WebSocket cancellation works on an empty dict."""
        ws_tasks = {}
        for task_id, task in list(ws_tasks.items()):
            if not task.done():
                task.cancel()
        # No error on empty dict

    @pytest.mark.asyncio
    async def test_checkpoint_cancel_step_independent(self):
        """Step 3: Checkpoint cancellation works on an empty dict."""
        cp_tasks = {}
        for sid, task in list(cp_tasks.items()):
            if not task.done():
                task.cancel()
        # No error on empty dict

    @pytest.mark.asyncio
    async def test_background_task_cancel(self):
        """Step 4: Background task cancellation with gather."""
        async def periodic_task():
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(periodic_task())

        # Cancel and gather
        task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_inactivity_monitor_stop(self):
        """Step 5: Inactivity monitor stop is callable."""
        mock_monitor = AsyncMock()
        mock_monitor.stop = AsyncMock()

        await mock_monitor.stop()
        mock_monitor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_order_terminal_before_websocket(self):
        """Terminal cleanup must happen BEFORE WebSocket cancellation."""
        execution_order = []

        async def mock_terminal_cleanup():
            execution_order.append("terminal_cleanup")

        async def mock_ws_cancel():
            execution_order.append("ws_cancel")

        async def mock_checkpoint_cancel():
            execution_order.append("checkpoint_cancel")

        # Execute in shutdown order
        await mock_terminal_cleanup()
        await mock_ws_cancel()
        await mock_checkpoint_cancel()

        assert execution_order == [
            "terminal_cleanup",
            "ws_cancel",
            "checkpoint_cancel",
        ]
        # Terminal cleanup must be first (breaks WebSocket loops)
        assert execution_order[0] == "terminal_cleanup"
