# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for tmux intermediary in TerminalManager.

Tests tmux helper methods, tmux-wrapped start_terminal_with_command(),
tmux cleanup in stop_terminal(), restart_ttyd_with_tmux_attach(),
and tmux-aware ensure_terminal_healthy().

TDD tests -- written before the implementation.
"""

import asyncio
import pytest
import shlex
import sys
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Task 1: Tests for tmux helper methods in TerminalManager
# ============================================================================

@pytest.mark.unit
class TestTmuxHelperMethods:
    """Tests for tmux helper methods: _check_tmux, _tmux_session_exists,
    _kill_tmux_session, _build_tmux_command."""

    @pytest.fixture
    def terminal_manager(self):
        """Create a TerminalManager with mocked ttyd and tmux checks."""
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

    @pytest.fixture
    def terminal_manager_no_tmux(self):
        """Create a TerminalManager where tmux is NOT available."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": None,  # tmux not installed
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    # --- _check_tmux ---

    def test_check_tmux_returns_true_when_installed(self, terminal_manager):
        """_check_tmux returns True when shutil.which('tmux') finds it."""
        assert terminal_manager._tmux_available is True

    def test_check_tmux_returns_false_when_missing(self, terminal_manager_no_tmux):
        """_check_tmux returns False when shutil.which('tmux') returns None."""
        assert terminal_manager_no_tmux._tmux_available is False

    # --- _tmux_session_exists ---

    def test_tmux_session_exists_returns_true(self, terminal_manager):
        """_tmux_session_exists returns True when 'tmux has-session' returns 0."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = terminal_manager._tmux_session_exists("ammo-abc123")

        assert result is True
        mock_run.assert_called_once_with(
            ["tmux", "has-session", "-t", "ammo-abc123"],
            stdout=-3,  # subprocess.DEVNULL
            stderr=-3,
        )

    def test_tmux_session_exists_returns_false(self, terminal_manager):
        """_tmux_session_exists returns False when 'tmux has-session' returns non-zero."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)
            result = terminal_manager._tmux_session_exists("ammo-abc123")

        assert result is False

    def test_tmux_session_exists_returns_false_on_exception(self, terminal_manager):
        """_tmux_session_exists returns False when subprocess raises."""
        with patch('subprocess.run', side_effect=OSError("tmux not found")):
            result = terminal_manager._tmux_session_exists("ammo-abc123")

        assert result is False

    # --- _kill_tmux_session ---

    def test_kill_tmux_session_succeeds(self, terminal_manager):
        """_kill_tmux_session returns True when 'tmux kill-session' returns 0."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = terminal_manager._kill_tmux_session("ammo-abc123")

        assert result is True
        mock_run.assert_called_once_with(
            ["tmux", "kill-session", "-t", "ammo-abc123"],
            stdout=-3,
            stderr=-3,
        )

    def test_kill_tmux_session_noop_when_not_exists(self, terminal_manager):
        """_kill_tmux_session returns False when session doesn't exist (non-zero exit)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)
            result = terminal_manager._kill_tmux_session("ammo-abc123")

        assert result is False

    def test_kill_tmux_session_handles_exception(self, terminal_manager):
        """_kill_tmux_session returns False gracefully on exception."""
        with patch('subprocess.run', side_effect=OSError("tmux not found")):
            result = terminal_manager._kill_tmux_session("ammo-abc123")

        assert result is False

    # --- _build_tmux_command ---
    # Note: _build_tmux_command now returns (cmd, socket_path, config_path) tuple
    # and requires session_id parameter for dedicated socket/config paths.

    def test_build_tmux_command_new_session(self, terminal_manager):
        """_build_tmux_command with attach_only=False uses drop_privs.py wrapper and start.sh script."""
        inner_command = ["/usr/bin/env", "KEY=VALUE", "node", "cli.js", "prompt"]
        import os, shutil as sh
        session_id = "abc123-full-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, socket_path, config_path = terminal_manager._build_tmux_command(
                tmux_session_name="ammo-abc123",
                inner_command=inner_command,
                attach_only=False,
                session_id=session_id,
            )

            # Format: ["/bin/bash", "-c", "python3 /tmp/.../drop_privs.py '<tmux_cmd>'"]
            assert result[0] == "/bin/bash"
            assert result[1] == "-c"
            assert "drop_privs.py" in result[2]
            assert "new-session -A -s ammo-abc123" in result[2]
            # Inner command is written to a start.sh script
            assert "start.sh" in result[2]
            # Verify the script contains the inner command with proper quoting
            script_content = Path(f"/tmp/{session_id}/start.sh").read_text()
            for part in inner_command:
                assert part in script_content or shlex.quote(part) in script_content
            # Verify socket and config paths
            assert socket_path == f"/tmp/{session_id}/tmux.sock"
            assert config_path == f"/tmp/{session_id}/tmux.conf"
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_attach_only(self, terminal_manager):
        """_build_tmux_command with attach_only=True uses drop_privs.py wrapper."""
        import shutil as sh
        try:
            result, socket_path, _ = terminal_manager._build_tmux_command(
                tmux_session_name="ammo-abc123",
                inner_command=[],  # Not used for attach
                attach_only=True,
                session_id="abc123-full-id",
            )

            assert result[0] == "/bin/bash"
            assert result[1] == "-c"
            assert "drop_privs.py" in result[2]
            assert "attach -t ammo-abc123" in result[2]
        finally:
            sh.rmtree("/tmp/abc123-full-id", ignore_errors=True)

    def test_build_tmux_command_escapes_quotes_in_script(self, terminal_manager):
        """Inner command with single quotes is properly escaped in launcher script."""
        import os, shutil as sh
        inner_command = ["/usr/bin/env", "node", "cli.js", "hello 'world'"]
        session_id = "abc123-quotes-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, _, _ = terminal_manager._build_tmux_command(
                tmux_session_name="ammo-abc123",
                inner_command=inner_command,
                attach_only=False,
                session_id=session_id,
            )

            script_content = Path(f"/tmp/{session_id}/start.sh").read_text()
            assert "hello" in script_content
            assert "world" in script_content
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_escapes_spaces_in_script(self, terminal_manager):
        """Inner command args with spaces are properly quoted in launcher script."""
        import os, shutil as sh
        inner_command = ["/usr/bin/env", "node", "cli.js", "a prompt with spaces"]
        session_id = "abc123-spaces-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, _, _ = terminal_manager._build_tmux_command(
                tmux_session_name="ammo-abc123",
                inner_command=inner_command,
                attach_only=False,
                session_id=session_id,
            )

            script_content = Path(f"/tmp/{session_id}/start.sh").read_text()
            assert shlex.quote("a prompt with spaces") in script_content
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_mouse_in_config_not_command(self, terminal_manager):
        """Mouse mode is now set in config file, not in tmux command preamble."""
        import os, shutil as sh
        inner_command = ["/usr/bin/env", "node", "cli.js"]
        session_id = "abc123-mouse-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, _, _ = terminal_manager._build_tmux_command(
                tmux_session_name="ammo-abc123",
                inner_command=inner_command,
                attach_only=False,
                session_id=session_id,
            )
            assert "set-option" not in result[2], \
                "Mouse mode is now in config file, not in tmux command preamble"
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_brackets_properly_quoted_in_script(self, terminal_manager):
        """The launcher script properly quotes shell glob chars like brackets.

        The inner command is written to a start.sh script with shlex.quote()
        on each argument, so special characters like [1m] in model names are
        safely quoted in the script. tmux runs the script as a single executable
        path, avoiding multi-layer shell expansion.
        """
        import os, shutil as sh
        inner_cmd = [
            "/usr/bin/env", "FOO=bar", "/usr/bin/node", "cli.js",
            "--model", "claude-opus-5[1m]",
        ]
        session_id = "test-session-full-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, _, _ = terminal_manager._build_tmux_command(
                tmux_session_name="test-session",
                inner_command=inner_cmd,
                attach_only=False,
                session_id=session_id,
            )
            script_content = Path(f"/tmp/{session_id}/start.sh").read_text()
            assert "claude-opus-5[1m]" in script_content
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_quotes_parts_with_spaces_in_script(self, terminal_manager):
        """Parts with spaces (like initial_prompt) are properly quoted in launcher script."""
        import os, shutil as sh
        inner_cmd = ["/usr/bin/env", "/usr/bin/node", "cli.js", "a prompt with spaces"]
        session_id = "test-session-spaces-id"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            result, _, _ = terminal_manager._build_tmux_command(
                tmux_session_name="test-session",
                inner_command=inner_cmd,
                attach_only=False,
                session_id=session_id,
            )
            script_content = Path(f"/tmp/{session_id}/start.sh").read_text()
            assert shlex.quote("a prompt with spaces") in script_content, \
                "Parts with spaces must be quoted to preserve argument boundaries"
        finally:
            sh.rmtree(f"/tmp/{session_id}", ignore_errors=True)


# ============================================================================
# Task 2: Tests for tmux-wrapped start_terminal_with_command()
# ============================================================================

@pytest.mark.unit
class TestStartTerminalWithTmux:
    """Tests for start_terminal_with_command() with tmux wrapping."""

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

    @pytest.fixture
    def terminal_manager_no_tmux(self):
        """Create a TerminalManager where tmux is NOT available."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": None,
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            tm = TerminalManager(base_port=9000, max_ports=10)
            yield tm

    @pytest.mark.asyncio
    async def test_start_terminal_wraps_command_in_tmux(self, terminal_manager):
        """When tmux available and tmux_session_name provided, command should contain 'tmux new-session'."""
        import os, shutil
        session_id = "sess1"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                port = await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "node", "cli.js", "prompt"],
                    working_dir=Path("/tmp/workdir"),
                    env={"KEY": "VALUE"},
                    tmux_session_name="ammo-sess1abc12",
                )

                assert port == 9000
                # Verify the command passed to subprocess contains tmux wrapping
                call_args = mock_exec.call_args
                cmd_parts = call_args[0]  # positional args to create_subprocess_exec
                assert any("tmux" in str(p) for p in cmd_parts)
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_start_terminal_attaches_to_existing_tmux(self, terminal_manager):
        """When tmux session already exists, command should be 'tmux new-session -A' (creates-or-attaches)."""
        import os, shutil
        session_id = "sess1"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                port = await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "node", "cli.js", "prompt"],
                    working_dir=Path("/tmp/workdir"),
                    env={"KEY": "VALUE"},
                    tmux_session_name="ammo-sess1abc12",
                )

                assert port == 9000
                # _build_tmux_command uses 'new-session -A' which creates-or-attaches idempotently
                call_args = mock_exec.call_args
                cmd_parts = call_args[0]
                assert "new-session" in cmd_parts or any("new-session" in str(p) for p in cmd_parts)
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_start_terminal_falls_back_without_tmux(self, terminal_manager_no_tmux):
        """When tmux not available, command should be passed directly (backward compat)."""
        with patch.object(terminal_manager_no_tmux, '_is_port_available', return_value=True), \
             patch.object(terminal_manager_no_tmux, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            port = await terminal_manager_no_tmux.start_terminal_with_command(
                session_id="sess1",
                command=["/usr/bin/env", "node", "cli.js", "prompt"],
                working_dir=Path("/tmp/workdir"),
                env={"KEY": "VALUE"},
                tmux_session_name="ammo-sess1abc12",
            )

            assert port == 9000
            # Verify no tmux in the command
            call_args = mock_exec.call_args
            cmd_parts = call_args[0]
            cmd_str = " ".join(str(p) for p in cmd_parts)
            assert "tmux" not in cmd_str

    @pytest.mark.asyncio
    async def test_start_terminal_stores_tmux_session_name(self, terminal_manager):
        """The returned TerminalProcess has tmux_session_name set."""
        import os, shutil
        session_id = "sess1"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "node", "cli.js"],
                    working_dir=Path("/tmp/workdir"),
                    env={},
                    tmux_session_name="ammo-sess1abc12",
                )

                terminal = terminal_manager._terminals[session_id]
                assert terminal.tmux_session_name == "ammo-sess1abc12"
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_start_terminal_no_tmux_name_skips_wrapping(self, terminal_manager):
        """When tmux_session_name=None, no tmux wrapping even if tmux available."""
        with patch.object(terminal_manager, '_is_port_available', return_value=True), \
             patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            await terminal_manager.start_terminal_with_command(
                session_id="sess1",
                command=["/usr/bin/env", "node", "cli.js"],
                working_dir=Path("/tmp/workdir"),
                env={},
                # No tmux_session_name passed (default None)
            )

            # No tmux in command
            call_args = mock_exec.call_args
            cmd_parts = call_args[0]
            cmd_str = " ".join(str(p) for p in cmd_parts)
            assert "tmux" not in cmd_str

            # tmux_session_name should be None on the terminal
            terminal = terminal_manager._terminals["sess1"]
            assert terminal.tmux_session_name is None


# ============================================================================
# Task 3: Tests for tmux cleanup in stop_terminal()
# ============================================================================

@pytest.mark.unit
class TestStopTerminalTmuxCleanup:
    """Tests for tmux cleanup in stop_terminal()."""

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
    async def test_stop_terminal_kills_tmux_session(self, terminal_manager):
        """When tmux_session_name is set, _kill_tmux_session is called."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name="ammo-sess1abc12",
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill') as mock_kill, \
             patch.object(terminal_manager, '_kill_tmux_session', return_value=True) as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            # First os.kill is SIGTERM, second is check (raises ProcessLookupError)
            mock_kill.side_effect = [None, ProcessLookupError]

            result = await terminal_manager.stop_terminal("sess1")

        assert result is True
        mock_tmux_kill.assert_called_once_with("ammo-sess1abc12", socket_path=None)

    @pytest.mark.asyncio
    async def test_stop_terminal_skips_tmux_when_not_set(self, terminal_manager):
        """When tmux_session_name is None, no tmux cleanup."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name=None,
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill') as mock_kill, \
             patch.object(terminal_manager, '_kill_tmux_session') as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_kill.side_effect = [None, ProcessLookupError]

            result = await terminal_manager.stop_terminal("sess1")

        assert result is True
        mock_tmux_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_terminal_handles_tmux_kill_failure(self, terminal_manager):
        """tmux kill fails, still returns True (best effort)."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name="ammo-sess1abc12",
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill') as mock_kill, \
             patch.object(terminal_manager, '_kill_tmux_session', return_value=False) as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_kill.side_effect = [None, ProcessLookupError]

            result = await terminal_manager.stop_terminal("sess1")

        assert result is True
        mock_tmux_kill.assert_called_once_with("ammo-sess1abc12", socket_path=None)


# ============================================================================
# Task 4: Tests for restart_ttyd_with_tmux_attach()
# ============================================================================

@pytest.mark.unit
class TestRestartTtydWithTmuxAttach:
    """Tests for restart_ttyd_with_tmux_attach()."""

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
    async def test_restart_ttyd_creates_new_ttyd_with_attach(self, terminal_manager):
        """restart_ttyd_with_tmux_attach starts a new ttyd with 'tmux attach' command."""
        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
             patch.object(terminal_manager, '_is_port_available', return_value=True), \
             patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_process = AsyncMock()
            mock_process.pid = 54321
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            port = await terminal_manager.restart_ttyd_with_tmux_attach(
                session_id="sess1",
                tmux_session_name="ammo-sess1abc12",
                working_dir=Path("/tmp/workdir"),
                env={"KEY": "VALUE"},
            )

            assert isinstance(port, int)
            # Verify 'attach -t' is in the command (clipboard options precede 'attach')
            call_args = mock_exec.call_args
            cmd_parts = call_args[0]
            cmd_str = " ".join(str(p) for p in cmd_parts)
            assert "attach -t" in cmd_str

    @pytest.mark.asyncio
    async def test_restart_ttyd_allocates_new_port(self, terminal_manager):
        """restart_ttyd_with_tmux_attach gets a fresh port."""
        # Use up port 9000
        terminal_manager._used_ports.add(9000)

        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
             patch.object(terminal_manager, '_is_port_available',
                          side_effect=lambda p: p != 9000), \
             patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_process = AsyncMock()
            mock_process.pid = 54321
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            port = await terminal_manager.restart_ttyd_with_tmux_attach(
                session_id="sess1",
                tmux_session_name="ammo-sess1abc12",
                working_dir=Path("/tmp/workdir"),
                env={},
            )

            assert port != 9000  # Should get a different port

    @pytest.mark.asyncio
    async def test_restart_ttyd_fails_when_no_tmux_session(self, terminal_manager):
        """Raises TerminalError if tmux session does not exist."""
        from orchestration.terminal_manager import TerminalError

        with patch.object(terminal_manager, '_tmux_session_exists', return_value=False):
            with pytest.raises(TerminalError, match="tmux session.*does not exist"):
                await terminal_manager.restart_ttyd_with_tmux_attach(
                    session_id="sess1",
                    tmux_session_name="ammo-sess1abc12",
                    working_dir=Path("/tmp/workdir"),
                    env={},
                )

    @pytest.mark.asyncio
    async def test_restart_ttyd_stores_terminal_info(self, terminal_manager):
        """restart_ttyd_with_tmux_attach stores TerminalProcess with tmux_session_name."""
        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
             patch.object(terminal_manager, '_is_port_available', return_value=True), \
             patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            mock_process = AsyncMock()
            mock_process.pid = 54321
            mock_process.returncode = None
            mock_exec.return_value = mock_process

            await terminal_manager.restart_ttyd_with_tmux_attach(
                session_id="sess1",
                tmux_session_name="ammo-sess1abc12",
                working_dir=Path("/tmp/workdir"),
                env={},
            )

            terminal = terminal_manager._terminals["sess1"]
            assert terminal.tmux_session_name == "ammo-sess1abc12"
            assert terminal.pid == 54321


# ============================================================================
# Task 5: Tests for ensure_terminal_healthy() with tmux awareness
# ============================================================================

@pytest.mark.unit
class TestEnsureTerminalHealthyTmux:
    """Tests for tmux-aware ensure_terminal_healthy()."""

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

    def _make_active_session(self, sm, session_id="sess1", tmp_path=None):
        """Helper to inject an ACTIVE session into the manager."""
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        session_dir = str(tmp_path / session_id) if tmp_path else f"/data/sessions/{session_id}"
        worktree_path = f"{session_dir}/worktree"

        state = SessionState(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
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
    async def test_recovery_reattaches_when_tmux_alive_ttyd_dead(self, mock_managers, tmp_path):
        """ttyd dead, tmux alive -> restart ttyd with 'tmux attach' (no new CLI process)."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # tmux session IS alive
        terminal_mgr._tmux_session_exists.return_value = True

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9002  # From restart_ttyd_with_tmux_attach
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_called_once()
        # Should NOT have started a full new terminal
        terminal_mgr.start_terminal_with_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_recovery_creates_new_when_both_dead(self, mock_managers, tmp_path):
        """Both ttyd and tmux dead -> full restart with tmux new-session."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        Path(state.session_dir).mkdir(parents=True, exist_ok=True)

        # tmux session is NOT alive
        terminal_mgr._tmux_session_exists.return_value = False

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9001  # From start_terminal_with_command
        terminal_mgr.start_terminal_with_command.assert_called_once()
        # Should pass tmux_session_name
        call_kwargs = terminal_mgr.start_terminal_with_command.call_args.kwargs
        assert "tmux_session_name" in call_kwargs
        assert call_kwargs["tmux_session_name"].startswith("ammo-")

    @pytest.mark.asyncio
    async def test_recovery_noop_when_both_alive(self, mock_managers, tmp_path):
        """Both ttyd and tmux alive -> return existing port."""
        sm, terminal_mgr, cli_tool_mgr = mock_managers
        state = self._make_active_session(sm, tmp_path=tmp_path)
        terminal_mgr.is_terminal_running.return_value = True

        new_port = await sm.ensure_terminal_healthy("sess1")

        assert new_port == 9000  # Original port
        terminal_mgr.restart_ttyd_with_tmux_attach.assert_not_called()
        terminal_mgr.start_terminal_with_command.assert_not_called()


# ============================================================================
# Task 6: Tests for session lifecycle with tmux
# ============================================================================

@pytest.mark.unit
class TestSessionLifecycleTmux:
    """Tests for tmux integration in create/resume/pause/terminate."""

    @pytest.fixture
    def mock_managers(self, tmp_path):
        """Create SessionManager with fully mocked dependencies."""
        terminal_mgr = Mock()
        terminal_mgr.is_available.return_value = True
        terminal_mgr.is_terminal_running.return_value = True
        terminal_mgr.start_terminal_with_command = AsyncMock(return_value=9001)
        terminal_mgr.stop_terminal = AsyncMock(return_value=True)

        cli_tool_mgr = Mock()
        cli_tool_mgr.get_cli_command.return_value = ["/usr/bin/env", "claude", "prompt"]

        worktree_mgr = Mock()
        worktree_mgr.create_worktree = AsyncMock(return_value="/data/sessions/test/worktree")

        gpu_mgr = Mock()
        gpu_mgr.allocate_gpus.return_value = [0]

        inactivity_monitor = Mock()
        inactivity_monitor.register_session = Mock()
        inactivity_monitor.unregister_session = Mock()

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

    def test_create_session_passes_tmux_session_name(self, mock_managers):
        """create_session() passes tmux_session_name to terminal manager.

        Note: This is a design validation test. The actual create_session()
        call is complex (requires worktree, GPU, etc.), so we verify the
        integration pattern by checking that start_terminal_with_command
        accepts the tmux_session_name parameter.
        """
        from orchestration.terminal_manager import TerminalManager
        import inspect

        sig = inspect.signature(TerminalManager.start_terminal_with_command)
        assert "tmux_session_name" in sig.parameters

    def test_terminal_process_has_tmux_session_name_field(self):
        """TerminalProcess dataclass has tmux_session_name field."""
        from orchestration.terminal_manager import TerminalProcess

        tp = TerminalProcess(
            session_id="sess1", port=9000, pid=123, master_fd=-1,
            tmux_session_name="ammo-sess1abc12",
        )
        assert tp.tmux_session_name == "ammo-sess1abc12"

    def test_terminal_process_tmux_session_name_defaults_none(self):
        """TerminalProcess tmux_session_name defaults to None."""
        from orchestration.terminal_manager import TerminalProcess

        tp = TerminalProcess(
            session_id="sess1", port=9000, pid=123, master_fd=-1,
        )
        assert tp.tmux_session_name is None
