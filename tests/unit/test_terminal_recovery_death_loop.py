# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the ttyd-alive/tmux-dead death loop fix.

When Claude Code exits, tmux destroys the pane (remain-on-exit off) and
the server exits (exit-empty on), but ttyd stays alive showing "[Process
exited]". Before the fix, cleanup_dead_terminal() saw ttyd alive and
returned False (no cleanup), causing start_terminal_with_command() to
early-return with the old port — an infinite recovery loop.

The fix adds tmux-awareness to cleanup_dead_terminal() and a defensive
health check to start_terminal_with_command()'s early-return path.
"""

import asyncio
import os
import signal
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.unit
class TestCleanupDeadTerminalTmuxAwareness:
    """Tests for tmux-aware cleanup_dead_terminal() fix."""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd/tmux checks."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            return_value="/usr/bin/ttyd"
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    def test_ttyd_alive_tmux_dead_kills_stale_ttyd(self, terminal_manager):
        """When ttyd PID is alive but tmux session is dead, kill stale ttyd and clean up."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0:
                return None  # ttyd alive
            # SIGKILL succeeds
            return None

        with patch('os.kill', side_effect=mock_kill), \
             patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
             patch('os.waitpid', return_value=(0, 0)):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is True
        assert "sess1" not in terminal_manager._terminals
        assert 9000 not in terminal_manager._used_ports
        # Verify ttyd was killed with SIGKILL
        assert (12345, signal.SIGKILL) in kill_calls

    def test_ttyd_alive_tmux_alive_no_cleanup(self, terminal_manager):
        """When both ttyd and tmux are alive, return False (no cleanup needed)."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill', return_value=None), \
             patch.object(terminal_manager, '_tmux_session_exists', return_value=True):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is False
        assert "sess1" in terminal_manager._terminals
        assert 9000 in terminal_manager._used_ports

    def test_no_tmux_name_skips_tmux_check(self, terminal_manager):
        """When terminal has no tmux_session_name, skip tmux check (backward compat)."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            # No tmux_session_name set (defaults to None)
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill', return_value=None):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        assert result is False
        assert "sess1" in terminal_manager._terminals

    def test_ttyd_kill_races_with_natural_exit(self, terminal_manager):
        """If ttyd dies between alive check and SIGKILL, handle ProcessLookupError gracefully."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        call_count = [0]

        def mock_kill(pid, sig):
            call_count[0] += 1
            if sig == 0:
                return None  # ttyd alive on first check
            # SIGKILL fails — process died between checks
            raise ProcessLookupError()

        with patch('os.kill', side_effect=mock_kill), \
             patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
             patch('os.waitpid', side_effect=ChildProcessError):
            result = terminal_manager.cleanup_dead_terminal("sess1")

        # Still cleaned up despite the race
        assert result is True
        assert "sess1" not in terminal_manager._terminals
        assert 9000 not in terminal_manager._used_ports


@pytest.mark.unit
class TestStartTerminalDefensiveCheck:
    """Tests for start_terminal_with_command() defensive health check."""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd/tmux checks."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            return_value="/usr/bin/ttyd"
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    @pytest.mark.asyncio
    async def test_stale_entry_cleaned_before_new_terminal(self, terminal_manager):
        """When session exists in _terminals but is unhealthy, clean up and spawn new."""
        from orchestration.terminal_manager import TerminalProcess

        # Set up stale entry (ttyd alive, tmux dead)
        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        # is_terminal_running returns False (tmux dead)
        # cleanup_dead_terminal will remove stale entry
        # Then start_terminal_with_command should allocate a new port and spawn

        mock_process = AsyncMock()
        mock_process.pid = 99999
        mock_process.returncode = None

        with patch.object(terminal_manager, 'is_terminal_running', return_value=False), \
             patch.object(terminal_manager, 'cleanup_dead_terminal', return_value=True) as mock_cleanup, \
             patch.object(terminal_manager, '_write_hardened_tmux_config'), \
             patch.object(terminal_manager, '_build_tmux_command', return_value=(
                 ["/bin/bash", "-c", "tmux new-session"], "/tmp/sess1/tmux.sock", "/tmp/sess1/tmux.conf"
             )), \
             patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process):

            # Simulate cleanup removing the entry
            def do_cleanup(sid):
                terminal_manager._terminals.pop(sid, None)
                terminal_manager._used_ports.discard(9000)
                return True
            mock_cleanup.side_effect = do_cleanup

            port = await terminal_manager.start_terminal_with_command(
                session_id="sess1",
                command=["/usr/bin/env", "node", "cli.js"],
                working_dir=Path("/tmp/worktree"),
                env={"HOME": "/home/session_user"},
                tmux_session_name="ammo-sess1",
            )

        # cleanup_dead_terminal was called for the stale entry
        mock_cleanup.assert_called_once_with("sess1")
        # A new terminal was spawned (not the old port 9000)
        assert port is not None
        assert "sess1" in terminal_manager._terminals
        # New entry should have the new PID
        assert terminal_manager._terminals["sess1"].pid == 99999

    @pytest.mark.asyncio
    async def test_healthy_entry_returns_existing_port(self, terminal_manager):
        """When session exists in _terminals and is healthy, return existing port."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1",
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        with patch.object(terminal_manager, 'is_terminal_running', return_value=True):
            port = await terminal_manager.start_terminal_with_command(
                session_id="sess1",
                command=["/usr/bin/env", "node", "cli.js"],
                working_dir=Path("/tmp/worktree"),
                env={"HOME": "/home/session_user"},
                tmux_session_name="ammo-sess1",
            )

        assert port == 9000
        # Original entry preserved
        assert terminal_manager._terminals["sess1"].pid == 12345


@pytest.mark.unit
class TestDeathLoopEndToEnd:
    """End-to-end test: the full recovery path after Claude Code exits."""

    @pytest.fixture
    def mock_managers(self, tmp_path):
        """Create SessionManager with mocked dependencies for recovery test."""
        from orchestration.terminal_manager import TerminalManager, TerminalProcess
        from orchestration.session_manager import SessionManager
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        terminal_mgr = Mock(spec=TerminalManager)
        terminal_mgr.is_available.return_value = True
        terminal_mgr.is_terminal_running.return_value = False
        terminal_mgr.cleanup_dead_terminal.return_value = True
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9001)
        terminal_mgr._tmux_session_exists = Mock(return_value=False)

        cli_mgr = Mock()
        cli_mgr.get_cli_command.return_value = ["/usr/bin/env", "node", "cli.js", "--continue"]

        worktree_mgr = Mock()
        gpu_mgr = Mock()
        inactivity_mgr = Mock()
        storage = Mock()
        storage.enabled = False

        sm = SessionManager(
            sessions_dir=str(tmp_path),
            worktree_manager=worktree_mgr,
            gpu_manager=gpu_mgr,
            terminal_manager=terminal_mgr,
            cli_tool_manager=cli_mgr,
            inactivity_monitor=inactivity_mgr,
            session_storage=storage,
        )

        # Create an ACTIVE session with terminal
        now = time.time()
        state = SessionState(
            session_id="sess1",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=now,
            last_accessed=now,
            worktree_path=str(tmp_path / "worktree"),
            session_dir=str(tmp_path / "session"),
            terminal_port=9000,
        )
        (tmp_path / "worktree").mkdir()
        (tmp_path / "session").mkdir()
        sm._sessions["sess1"] = state

        return sm, terminal_mgr, cli_mgr, state

    @pytest.mark.asyncio
    async def test_recovery_breaks_death_loop(self, mock_managers):
        """After Claude Code exits (tmux dead, ttyd alive), ensure_terminal_healthy
        properly cleans up the stale terminal and spawns a new one.

        Before fix: cleanup_dead_terminal returns False → start_terminal_with_command
        early-returns → old port returned → next check finds tmux still dead → loop.

        After fix: cleanup_dead_terminal detects tmux dead, kills stale ttyd,
        removes entry → start_terminal_with_command spawns new terminal → recovery works.
        """
        sm, terminal_mgr, cli_mgr, state = mock_managers

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9001
        assert state.terminal_port == 9001
        terminal_mgr.cleanup_dead_terminal.assert_called_once_with("sess1")
        terminal_mgr.start_terminal_with_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_uses_reattach_when_tmux_alive(self, mock_managers):
        """When tmux is alive but ttyd died, use restart_ttyd_with_tmux_attach
        (not full restart). This preserves the running Claude Code process."""
        sm, terminal_mgr, cli_mgr, state = mock_managers

        # tmux alive for this test
        terminal_mgr._tmux_session_exists.return_value = True
        terminal_mgr.restart_ttyd_with_tmux_attach = AsyncMock(return_value=9002)

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9002
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_called_once()
        # Full restart should NOT have been called
        terminal_mgr.start_terminal_with_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_recovery_doesnt_double_spawn(self, mock_managers):
        """Two concurrent recovery attempts should not both spawn terminals.
        The per-session lock ensures only one runs; the second sees recovery complete."""
        sm, terminal_mgr, cli_mgr, state = mock_managers

        # First call: terminal dead → recovery
        # Second call: after lock released, terminal now running
        call_count = [0]

        def is_running_after_first_recovery(sid):
            call_count[0] += 1
            # First two calls (initial check + lock re-check) return False
            # Third call (second task's lock re-check) returns True
            return call_count[0] > 2

        terminal_mgr.is_terminal_running.side_effect = is_running_after_first_recovery

        results = await asyncio.gather(
            sm.ensure_terminal_healthy("sess1"),
            sm.ensure_terminal_healthy("sess1"),
        )

        # At least one should have gotten a port
        assert any(r is not None for r in results)
        # start_terminal_with_command called at most once (no double-spawn)
        assert terminal_mgr.start_terminal_with_command.call_count <= 1
