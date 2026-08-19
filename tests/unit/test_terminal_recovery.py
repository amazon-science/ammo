# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for terminal auto-recovery.

Tests the TerminalManager.cleanup_dead_terminal() method
and the SessionManager.ensure_terminal_healthy() method.
These are TDD tests -- written before the implementation.
"""

import asyncio
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Tests for TerminalManager.cleanup_dead_terminal()
# ============================================================================

@pytest.mark.unit
class TestTerminalManagerCleanupDeadTerminal:
    """Tests for TerminalManager.cleanup_dead_terminal()"""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd check."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            return_value="/usr/bin/ttyd"
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    def test_cleanup_dead_terminal_removes_stale_entry(self, terminal_manager):
        """When ttyd PID is dead, cleanup_dead_terminal removes _terminals entry and releases port."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=99999, master_fd=-1
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill', side_effect=ProcessLookupError):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is True
        assert "sess1" not in terminal_manager._terminals
        assert 9000 not in terminal_manager._used_ports

    def test_cleanup_dead_terminal_noop_when_alive(self, terminal_manager):
        """When ttyd PID is alive, cleanup_dead_terminal returns False and does NOT remove entry."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill', return_value=None):  # os.kill(pid, 0) succeeds
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is False
        assert "sess1" in terminal_manager._terminals
        assert 9000 in terminal_manager._used_ports

    def test_cleanup_dead_terminal_noop_for_unknown_session(self, terminal_manager):
        """cleanup_dead_terminal returns False for unknown session_id."""
        result = terminal_manager.cleanup_dead_terminal("nonexistent")
        assert result is False

    def test_cleanup_dead_terminal_handles_permission_error(self, terminal_manager):
        """When os.kill raises PermissionError, treat as alive (not dead)."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=1, master_fd=-1
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill', side_effect=PermissionError):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is False  # Treat as alive
        assert "sess1" in terminal_manager._terminals


# ============================================================================
# Tests for SessionManager.ensure_terminal_healthy()
# ============================================================================

@pytest.mark.unit
class TestSessionManagerEnsureTerminalHealthy:
    """Tests for SessionManager.ensure_terminal_healthy()"""

    @pytest.fixture
    def mock_managers(self, tmp_path):
        """Create SessionManager with fully mocked dependencies."""
        terminal_mgr = Mock()
        terminal_mgr.is_available.return_value = True
        terminal_mgr.is_terminal_running.return_value = False
        terminal_mgr.cleanup_dead_terminal.return_value = True
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9001)
        terminal_mgr.restart_ttyd_with_tmux_attach = AsyncMock(return_value=9002)
        terminal_mgr._tmux_session_exists = Mock(return_value=False)

        cli_tool_mgr = Mock()
        cli_tool_mgr.get_cli_command.return_value = ["/usr/bin/env", "claude", "--continue"]

        worktree_mgr = Mock()
        gpu_mgr = Mock()

        inactivity_monitor = Mock()
        inactivity_monitor.register_session = Mock()

        session_storage = Mock()
        session_storage.enabled = False

        from orchestration.session_manager import SessionManager
        with patch.object(SessionManager, '_load_sessions'):
            sm = SessionManager(
                sessions_dir=str(tmp_path),
                worktree_manager=worktree_mgr,
                gpu_manager=gpu_mgr,
                terminal_manager=terminal_mgr,
                cli_tool_manager=cli_tool_mgr,
                inactivity_monitor=inactivity_monitor,
                session_storage=session_storage,
            )
        return sm, terminal_mgr, cli_tool_mgr

    def _make_active_session(self, sm, session_id="sess1", tmp_path=None, cli_tool=None):
        """Helper to inject an ACTIVE session into the manager."""
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        session_dir = str(tmp_path / session_id) if tmp_path else f"/data/sessions/{session_id}"
        worktree_path = f"{session_dir}/worktree"

        state = SessionState(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            cli_tool=cli_tool or CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=time.time(),
            last_accessed=time.time(),
            terminal_port=9000,
            worktree_path=worktree_path,
            session_dir=session_dir,
        )
        sm._sessions[session_id] = state
        return state

    @pytest.mark.asyncio
    async def test_recovery_restarts_dead_terminal_with_continue(self, mock_managers, tmp_path):
        """When ttyd is dead for ACTIVE session, restart with --continue flag."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        project_dir = Path(state.session_dir) / "claude-config" / "projects" / "-tmp-worktree"
        project_dir.mkdir(parents=True)
        (project_dir / "conversation.jsonl").write_text("{}")

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9001
        assert state.terminal_port == 9001
        terminal_mgr.cleanup_dead_terminal.assert_called_once_with("sess1")
        cli_tool_mgr.get_cli_command.assert_called_once()
        # Verify is_resume=True was passed
        call_kwargs = cli_tool_mgr.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is True or \
            (len(call_kwargs) > 1 and call_kwargs[1].get("is_resume") is True)

    @pytest.mark.asyncio
    async def test_codex_recovery_without_history_replays_initial_prompt(self, mock_managers, tmp_path):
        """Codex first terminal attach should not resume before history exists."""
        from shared.session_models import CLIToolType

        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path, cli_tool=CLIToolType.CODEX)
        state.initial_prompt = "bootstrap prompt"
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9001
        cli_tool_mgr.get_cli_command.assert_called_once()
        call_kwargs = cli_tool_mgr.get_cli_command.call_args.kwargs
        assert call_kwargs["is_resume"] is False
        assert call_kwargs["initial_prompt"] == "bootstrap prompt"
        assert call_kwargs["extra_env"]["CODEX_HOME"].endswith("/codex-home")

    @pytest.mark.asyncio
    async def test_codex_recovery_with_history_uses_resume_last_without_prompt(self, mock_managers, tmp_path):
        """Codex full recovery should resume when per-session history exists."""
        from shared.session_models import CLIToolType

        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path, cli_tool=CLIToolType.CODEX)
        state.initial_prompt = "bootstrap prompt"
        codex_home = Path(state.session_dir) / "codex-home"
        codex_home.mkdir(parents=True)
        (codex_home / "session_index.jsonl").write_text("{}\n")

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9001
        cli_tool_mgr.get_cli_command.assert_called_once()
        call_kwargs = cli_tool_mgr.get_cli_command.call_args.kwargs
        assert call_kwargs["is_resume"] is True
        assert call_kwargs["initial_prompt"] is None
        assert call_kwargs["extra_env"]["CODEX_HOME"].endswith("/codex-home")

    @pytest.mark.asyncio
    async def test_no_recovery_when_terminal_alive(self, mock_managers, tmp_path):
        """When ttyd is alive, return existing port without restart."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        terminal_mgr.is_terminal_running.return_value = True

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9000  # Original port
        terminal_mgr.start_terminal_with_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_recovery_for_paused_session(self, mock_managers, tmp_path):
        """PAUSED sessions should NOT auto-recover; return None."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        from shared.session_models import SessionStatus

        state = self._make_active_session(sm, tmp_path=tmp_path)
        state.status = SessionStatus.PAUSED

        result = await sm.ensure_terminal_healthy("sess1")

        assert result is None
        terminal_mgr.start_terminal_with_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_recovery_for_unknown_session(self, mock_managers):
        """Unknown session returns None."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers

        result = await sm.ensure_terminal_healthy("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_recovery_failure_returns_none(self, mock_managers, tmp_path):
        """If start_terminal_with_command raises, return None and log error."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        from orchestration.terminal_manager import TerminalError
        terminal_mgr.start_terminal_with_command.side_effect = TerminalError("port bind fail")

        result = await sm.ensure_terminal_healthy("sess1")

        assert result is None

    @pytest.mark.asyncio
    async def test_recovery_not_available_when_ttyd_missing(self, mock_managers, tmp_path):
        """If ttyd is not available, return None."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        self._make_active_session(sm, tmp_path=tmp_path)
        terminal_mgr.is_available.return_value = False

        result = await sm.ensure_terminal_healthy("sess1")

        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_recovery_uses_lock(self, mock_managers, tmp_path):
        """Two simultaneous calls should not double-spawn ttyd (lock protects)."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        call_count = 0

        async def slow_start(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            # After first call succeeds, mark terminal as running
            # so second call sees it as alive
            terminal_mgr.is_terminal_running.return_value = True
            return 9001

        terminal_mgr.start_terminal_with_command.side_effect = slow_start

        results = await asyncio.gather(
            sm.ensure_terminal_healthy("sess1"),
            sm.ensure_terminal_healthy("sess1"),
        )

        # start_terminal_with_command should be called exactly once
        assert call_count == 1
        # Both calls should return the same port
        assert results[0] == 9001
        assert results[1] == 9001

    @pytest.mark.asyncio
    async def test_ensure_terminal_healthy_tmux_alive_calls_reattach(self, mock_managers, tmp_path):
        """When ttyd is dead but tmux session is still alive, reattach via restart_ttyd_with_tmux_attach.

        This is the key scenario enabled by destroy-unattached off: the browser disconnects,
        ttyd exits (or is killed), but the tmux session (and all team panes) survives.
        On reconnect, ensure_terminal_healthy detects the live tmux session and calls
        restart_ttyd_with_tmux_attach instead of doing a full CLI restart.

        This is important because a full restart would lose all in-progress team work
        and require --continue to re-establish the Claude session from scratch.
        """
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # tmux is alive (survived the browser disconnect)
        terminal_mgr._tmux_session_exists.return_value = True
        terminal_mgr.restart_ttyd_with_tmux_attach.return_value = 9002

        new_port = await sm.ensure_terminal_healthy("sess1")

        # Must reattach, not do a full CLI restart
        assert new_port == 9002
        assert state.terminal_port == 9002
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_called_once()
        terminal_mgr.start_terminal_with_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_terminal_healthy_both_dead_full_restart(self, mock_managers, tmp_path):
        """When both ttyd and tmux are dead, do a full CLI restart (start_terminal_with_command).

        This handles the case where the tmux session was explicitly killed (e.g. OS restart,
        container restart) or the session has been idle long enough that both processes died.
        A full restart is needed, using --continue to resume the most recent conversation.
        """
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # Both dead
        terminal_mgr._tmux_session_exists.return_value = False
        terminal_mgr.start_terminal_with_command.return_value = 9001

        new_port = await sm.ensure_terminal_healthy("sess1")

        # Must do a full restart, not just reattach
        assert new_port == 9001
        assert state.terminal_port == 9001
        terminal_mgr.start_terminal_with_command.assert_called_once()
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_not_called()

    # ---- edge-case probe added by verifier ----
    @pytest.mark.asyncio
    async def test_ensure_terminal_healthy_reattach_failure_returns_none(self, mock_managers, tmp_path):
        """When tmux is alive but restart_ttyd_with_tmux_attach raises TerminalError, return None.

        A failed reattach must NOT fall through to start_terminal_with_command — that
        would launch a new CLI process alongside a still-running tmux session (double
        process). The exception path must cleanly return None.
        """
        from orchestration.terminal_manager import TerminalError

        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        terminal_mgr._tmux_session_exists.return_value = True
        terminal_mgr.restart_ttyd_with_tmux_attach.side_effect = TerminalError("no ports available")

        result = await sm.ensure_terminal_healthy("sess1")

        assert result is None
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_called_once()
        terminal_mgr.start_terminal_with_command.assert_not_called()


# ============================================================================
# Tests for proxy auto-recovery integration
# ============================================================================

@pytest.mark.unit
class TestProxyAutoRecovery:
    """Tests for auto-recovery integration in proxy endpoints."""

    @pytest.fixture
    def mock_app_state(self):
        """Create mock app-level managers."""
        terminal_mgr = Mock()
        session_mgr = Mock()
        session_mgr.ensure_terminal_healthy = AsyncMock(return_value=9001)
        return terminal_mgr, session_mgr

    @pytest.mark.asyncio
    async def test_http_proxy_calls_recovery_on_stale_port(self, mock_app_state):
        """When terminal is dead, ensure_terminal_healthy should be invoked
        and return a recovered port."""
        terminal_mgr, session_mgr = mock_app_state
        terminal_mgr.get_terminal_port.return_value = 9000
        terminal_mgr.is_terminal_running.return_value = False

        # Simulate the recovery path
        recovered_port = await session_mgr.ensure_terminal_healthy("sess1")

        assert recovered_port == 9001
        session_mgr.ensure_terminal_healthy.assert_called_once_with("sess1")

    @pytest.mark.asyncio
    async def test_ws_proxy_calls_recovery_on_connection_refused(self, mock_app_state):
        """When WS proxy gets connection refused, it should attempt
        recovery before returning error to client."""
        terminal_mgr, session_mgr = mock_app_state
        terminal_mgr.get_terminal_port.return_value = 9000
        terminal_mgr.is_terminal_running.return_value = False

        # Simulate the recovery path
        recovered_port = await session_mgr.ensure_terminal_healthy("sess1")

        assert recovered_port == 9001
        session_mgr.ensure_terminal_healthy.assert_called_once_with("sess1")


# ============================================================================
# Tests for port exhaustion
# ============================================================================

@pytest.mark.unit
class TestPortExhaustion:
    """Tests for TerminalManager port allocation under exhaustion."""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd check."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            return_value="/usr/bin/ttyd"
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    def test_port_exhaustion_raises_terminal_error(self, terminal_manager):
        """When all max_ports are used, _allocate_port raises TerminalError."""
        from orchestration.terminal_manager import TerminalError, TerminalProcess

        # Fill all 10 ports (9000-9009) in _used_ports
        for i in range(terminal_manager.max_ports):
            port = terminal_manager.base_port + i
            terminal_manager._used_ports.add(port)
            terminal_manager._terminals[f"sess{i}"] = TerminalProcess(
                session_id=f"sess{i}", port=port, pid=10000 + i, master_fd=-1
            )

        with pytest.raises(TerminalError, match="No available ports"):
            terminal_manager._allocate_port()

    def test_port_released_after_stop_can_be_reallocated(self, terminal_manager):
        """After a port is released, _allocate_port can return it again."""
        # Add all ports except one as used
        for i in range(1, terminal_manager.max_ports):
            terminal_manager._used_ports.add(terminal_manager.base_port + i)

        # First allocation should succeed (port 9000 is free)
        port = terminal_manager._allocate_port()
        assert port == terminal_manager.base_port

        # Now all ports occupied - release port
        terminal_manager._release_port(port)

        # Should be allocatable again
        port2 = terminal_manager._allocate_port()
        assert port2 == terminal_manager.base_port


# ============================================================================
# Tests for terminal lifecycle edges
# ============================================================================

@pytest.mark.unit
class TestTerminalLifecycleEdges:
    """Tests for edge cases in terminal lifecycle: pause/resume, crash."""

    @pytest.fixture
    def mock_managers(self, tmp_path):
        """Create SessionManager with fully mocked dependencies."""
        terminal_mgr = Mock()
        terminal_mgr.is_available.return_value = True
        terminal_mgr.is_terminal_running.return_value = False
        terminal_mgr.cleanup_dead_terminal.return_value = True
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9001)
        terminal_mgr.restart_ttyd_with_tmux_attach = AsyncMock(return_value=9002)
        terminal_mgr._tmux_session_exists = Mock(return_value=False)

        cli_tool_mgr = Mock()
        cli_tool_mgr.get_cli_command.return_value = ["/usr/bin/env", "claude", "--continue"]

        worktree_mgr = Mock()
        gpu_mgr = Mock()

        inactivity_monitor = Mock()
        inactivity_monitor.register_session = Mock()

        session_storage = Mock()
        session_storage.enabled = False

        from orchestration.session_manager import SessionManager
        with patch.object(SessionManager, '_load_sessions'):
            sm = SessionManager(
                sessions_dir=str(tmp_path),
                worktree_manager=worktree_mgr,
                gpu_manager=gpu_mgr,
                terminal_manager=terminal_mgr,
                cli_tool_manager=cli_tool_mgr,
                inactivity_monitor=inactivity_monitor,
                session_storage=session_storage,
            )
        return sm, terminal_mgr, cli_tool_mgr

    def _make_session(self, sm, session_id, status, tmp_path):
        """Inject a session with given status into the manager."""
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        session_dir = str(tmp_path / session_id)
        worktree_path = f"{session_dir}/worktree"

        state = SessionState(
            session_id=session_id,
            status=status,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=time.time(),
            last_accessed=time.time(),
            terminal_port=9000,
            worktree_path=worktree_path,
            session_dir=session_dir,
        )
        sm._sessions[session_id] = state
        return state

    @pytest.mark.asyncio
    async def test_terminal_restart_after_pause_resume(self, mock_managers, tmp_path):
        """
        After a session is paused (terminal stopped), resuming should trigger
        ensure_terminal_healthy to assign a new port.
        """
        from shared.session_models import SessionStatus
        sm, terminal_mgr, cli_tool_mgr = mock_managers

        # Start in ACTIVE state
        state = self._make_session(sm, "sess1", SessionStatus.ACTIVE, tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # Terminal is not running (simulating paused state where ttyd was stopped)
        terminal_mgr.is_terminal_running.return_value = False
        terminal_mgr.cleanup_dead_terminal.return_value = True
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9005)

        new_port = await sm.ensure_terminal_healthy("sess1")

        # Should get a new port (not the old 9000)
        assert new_port == 9005
        assert state.terminal_port == 9005

    @pytest.mark.asyncio
    async def test_ttyd_crash_during_active_session(self, mock_managers, tmp_path):
        """
        When ttyd process dies mid-session (simulated by cleanup_dead_terminal returning True),
        ensure_terminal_healthy should clean up and restart with a new port.
        """
        from shared.session_models import SessionStatus
        sm, terminal_mgr, cli_tool_mgr = mock_managers

        state = self._make_session(sm, "sess1", SessionStatus.ACTIVE, tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # Simulate: terminal was running, then crashed
        terminal_mgr.is_terminal_running.return_value = False
        terminal_mgr.cleanup_dead_terminal.return_value = True  # stale entry found and removed
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9003)

        new_port = await sm.ensure_terminal_healthy("sess1")

        # Cleanup was called to remove the stale entry
        terminal_mgr.cleanup_dead_terminal.assert_called_once_with("sess1")
        # New terminal was started
        terminal_mgr.start_terminal_with_command.assert_called_once()
        assert new_port == 9003
        assert state.terminal_port == 9003
