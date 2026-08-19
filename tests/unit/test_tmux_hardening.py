# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for tmux session hardening in TerminalManager.

Redesigned approach: dedicated tmux server per session via -S (socket)
and -f (config file), with no-prefix allowlist model.

Key principles:
- Each session gets its own tmux server via `-S socket_path`
- Hardened config is loaded at tmux startup via `-f config_path`
- Prefix changed to C-Space (frees C-b for application pass-through)
- All prefix bindings removed via `unbind-key -a -T prefix`
- Mouse ON by default in config
- No status bar
- destroy-unattached off (tmux survives browser disconnect)
- _apply_tmux_hardening() is a no-op (hardening is built into startup)
- Mouse mode toggle API uses dedicated socket via -S

TDD tests -- written before the implementation.
"""

import os
import subprocess
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# TestTmuxHardenedConfigWrite: config file creation (new allowlist model)
# ============================================================================

@pytest.mark.unit
class TestTmuxHardenedConfigWrite:
    """Tests for _write_hardened_tmux_config() -- new allowlist model."""

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

    def test_config_written_to_session_private_dir(self, terminal_manager, tmp_path):
        """Config file written to /tmp/{session_id}/tmux.conf."""
        session_id = "test-sess-abc123"
        with patch('orchestration.terminal_manager.Path') as mock_path_cls, \
             patch('orchestration.terminal_manager.os.chown') as mock_chown:
            mock_dir = Mock()
            mock_conf = Mock()
            mock_conf.__str__ = lambda self: f"/tmp/{session_id}/tmux.conf"

            def path_side_effect(*args):
                joined = "/".join(str(a) for a in args)
                if "tmux.conf" in joined:
                    return mock_conf
                return mock_dir

            mock_path_cls.side_effect = path_side_effect
            mock_dir.mkdir = Mock()
            mock_conf.parent = mock_dir
            mock_conf.write_text = Mock()

            terminal_manager._write_hardened_tmux_config(session_id)

            mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_conf.write_text.assert_called_once()
            mock_chown.assert_called_once_with(str(mock_dir), 1000, 1000)

    def test_config_unbinds_prefix_and_sets_prefix_to_c_space(self, terminal_manager, tmp_path):
        """Config sets prefix to C-Space (freeing C-b) and unbinds all prefix bindings.
        Must NOT use 'unbind-key -a -T root' which wipes mouse bindings."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "unbind-key -a -T prefix" in content
        assert "set -g prefix C-Space" in content
        # C-b should NOT appear as an active unbind target (no longer needed)
        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert not any("unbind-key -T root C-b" in l for l in active_lines), \
            "unbind-key -T root C-b is no longer needed — prefix is C-Space, not C-b"
        # Ensure "unbind-key -a -T root" is NOT used as an actual command (comments are OK)
        assert not any("unbind-key -a -T root" in l for l in active_lines), \
            "Must NOT unbind all root keys — that destroys mouse scroll bindings"

    def test_config_allows_ctrl_b_passthrough(self, terminal_manager, tmp_path):
        """Config sets prefix to C-Space so C-b is completely free and passes
        through to the application. Claude Code uses Ctrl+B to background tasks."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        # Prefix must NOT be C-b (must be something else like C-Space)
        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert not any("prefix C-b" in l for l in active_lines), \
            "Prefix must not be C-b — it must pass through to the application"
        assert any("prefix C-Space" in l for l in active_lines), \
            "Prefix must be set to C-Space to free C-b"
        # Must NOT bind C-b to anything
        assert not any(l.strip().startswith("bind-key") and "C-b" in l for l in active_lines), \
            "C-b must not be bound in any table — it must pass through to the application"

    def test_config_unbinds_all_prefix_keys(self, terminal_manager, tmp_path):
        """Config uses unbind-key -a -T prefix to remove ALL prefix bindings."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "unbind-key -a -T prefix" in content

    def test_config_disables_status_bar(self, terminal_manager, tmp_path):
        """Config sets status off to remove tmux status line."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "set -g status off" in content

    def test_config_disables_destroy_unattached(self, terminal_manager, tmp_path):
        """Config sets destroy-unattached off so tmux survives when browser disconnects.

        With destroy-unattached off, the tmux session (and all team panes) persist
        after ttyd's client process detaches. The terminal stays alive for reconnect.
        Explicit kill is done via _kill_tmux_session on pause/terminate.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "set -g destroy-unattached off" in content

    def test_config_destroy_unattached_off_not_on(self, terminal_manager, tmp_path):
        """Regression guard: config must NOT contain destroy-unattached on.

        destroy-unattached on was the old (broken) setting that caused agent teams
        to disappear whenever the browser disconnected. This test ensures we never
        regress back to that behavior.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert not any("set -g destroy-unattached on" in l for l in active_lines), (
            "REGRESSION: config must NOT contain 'set -g destroy-unattached on' — "
            "that setting kills the tmux session (and all team panes) whenever the "
            "browser disconnects. Use 'destroy-unattached off' instead."
        )

    def test_config_disables_remain_on_exit(self, terminal_manager, tmp_path):
        """Config sets remain-on-exit off so panes close when process exits."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "set -g remain-on-exit off" in content

    def test_config_enables_mouse(self, terminal_manager, tmp_path):
        """Config enables mouse mode by default for scroll support."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert "set -g mouse on" in content

    def test_config_sets_xterm_256color_terminal(self, terminal_manager, tmp_path):
        """Config overrides default-terminal to xterm-256color.

        tmux-256color (the default) is missing key capabilities like bce
        (background color erase) and ech (erase characters) that xterm.js
        supports. This mismatch causes garbled/overlapping text in apps
        like Claude Code that do frequent partial screen updates.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        assert 'set -g default-terminal "xterm-256color"' in content
        # Must NOT use tmux-256color (causes rendering artifacts)
        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert not any("tmux-256color" in l for l in active_lines), \
            "Must not set default-terminal to tmux-256color — it causes garbled rendering"

    def test_config_enables_exit_empty(self, terminal_manager, tmp_path):
        """Config sets exit-empty on so tmux server exits when last session ends.

        Without exit-empty, a tmux server can linger after all its sessions
        close, leaving orphaned server processes. With exit-empty on, the
        lifecycle chain is complete: process exits -> pane closes ->
        session ends -> tmux server exits.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        # Must be an active (non-commented) line
        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert any("set -g exit-empty on" in l for l in active_lines), (
            "Config must contain 'set -g exit-empty on' as an active line. "
            f"Active lines: {active_lines}"
        )

    def test_tmux_config_pane_lifecycle_complete(self, terminal_manager, tmp_path):
        """Config must have all three settings for correct pane lifecycle:
        - remain-on-exit off (pane closes when process exits)
        - exit-empty on (server exits when last session ends)
        - destroy-unattached off (session SURVIVES when browser disconnects)

        destroy-unattached off is critical: it prevents the tmux session and all
        team panes from being destroyed when the browser tab closes or ttyd detaches.
        Explicit session teardown is done via _kill_tmux_session on pause/terminate.

        Lifecycle: process exits -> pane closes -> window closes ->
        session closes -> tmux server exits. No orphaned tmux processes on clean exit.
        On browser disconnect: tmux session stays alive, ready for reconnect.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        required_settings = [
            "set -g remain-on-exit off",
            "set -g exit-empty on",
            "set -g destroy-unattached off",
        ]
        for setting in required_settings:
            assert any(setting in l for l in active_lines), (
                f"Config must contain '{setting}' as an active line for correct "
                f"pane lifecycle. Active lines: {active_lines}"
            )

    def test_config_returns_path(self, terminal_manager, tmp_path):
        """_write_hardened_tmux_config returns the Path to the config file."""
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            result = terminal_manager._write_hardened_tmux_config("sess")

        assert isinstance(result, Path)


# ============================================================================
# TestTmuxDedicatedSocket: tmux command uses -S socket and -f config
# ============================================================================

@pytest.mark.unit
class TestTmuxDedicatedSocket:
    """Tests for _build_tmux_command() using dedicated socket and config."""

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

    def _build_with_tmpdir(self, terminal_manager, session_id, **kwargs):
        """Helper to call _build_tmux_command with a real /tmp/{session_id} dir."""
        import os, shutil
        d = f"/tmp/{session_id}"
        os.makedirs(d, exist_ok=True)
        try:
            return terminal_manager._build_tmux_command(session_id=session_id, **kwargs)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_tmux_command_includes_socket_flag(self, terminal_manager):
        """tmux command includes -S with session-specific socket path."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        assert "-S" in cmd[2]
        assert "/tmp/test123-full-id/tmux.sock" in cmd[2]

    def test_tmux_command_includes_config_flag(self, terminal_manager):
        """tmux command includes -f with session-specific config path."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert "-f" in cmd[2]
        assert "/tmp/test123-full-id/tmux.conf" in cmd[2]

    def test_tmux_command_attach_includes_socket_flag(self, terminal_manager):
        """tmux attach command also uses -S with dedicated socket."""
        cmd, socket_path, config_path = terminal_manager._build_tmux_command(
            tmux_session_name="ammo-test123",
            inner_command=[],
            attach_only=True,
            session_id="test123-full-id",
        )
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        assert "-S" in cmd[2]
        assert "/tmp/test123-full-id/tmux.sock" in cmd[2]
        assert "attach" in cmd[2]

    def test_tmux_command_new_session(self, terminal_manager):
        """tmux new-session command includes -A -s flags."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert "new-session -A -s ammo-test123" in cmd[2]

    def test_tmux_command_no_preamble(self, terminal_manager):
        """New design does NOT have preamble set-option commands (mouse is in config)."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert "set-option" not in cmd[2]
        assert "bind m" not in cmd[2]

    def test_tmux_command_privilege_drop_wrapper(self, terminal_manager):
        """Command uses drop_privs.py wrapper for capability-aware privilege drop."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-c"
        assert "drop_privs.py" in cmd[2]

    def test_tmux_command_uses_start_script(self, terminal_manager):
        """New-session command references start.sh script instead of inline args."""
        cmd, socket_path, config_path = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert "start.sh" in cmd[2]

    def test_build_tmux_command_returns_socket_and_config_paths(self, terminal_manager):
        """_build_tmux_command returns (cmd, socket_path, config_path) tuple."""
        result = self._build_with_tmpdir(
            terminal_manager, "test123-full-id",
            tmux_session_name="ammo-test123",
            inner_command=["/usr/bin/env", "claude"],
            attach_only=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        cmd, socket_path, config_path = result
        assert isinstance(cmd, list)
        assert socket_path == "/tmp/test123-full-id/tmux.sock"
        assert config_path == "/tmp/test123-full-id/tmux.conf"


# ============================================================================
# TestApplyTmuxHardeningNoop: hardening is now built into startup
# ============================================================================

@pytest.mark.unit
class TestApplyTmuxHardeningNoop:
    """_apply_tmux_hardening() should be a no-op since hardening is built into startup."""

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

    def test_apply_tmux_hardening_is_noop(self, terminal_manager):
        """_apply_tmux_hardening() does nothing (no subprocess calls, no file writes)."""
        with patch('subprocess.run') as mock_run, \
             patch.object(terminal_manager, '_write_hardened_tmux_config') as mock_write:

            terminal_manager._apply_tmux_hardening("sess123")

            mock_run.assert_not_called()
            mock_write.assert_not_called()

    def test_apply_tmux_hardening_no_exception(self, terminal_manager):
        """_apply_tmux_hardening() returns without error regardless of input."""
        # Should not raise
        terminal_manager._apply_tmux_hardening("any-session-id")
        terminal_manager._apply_tmux_hardening("")


# ============================================================================
# TestTerminalProcessSocketPath: TerminalProcess stores socket path
# ============================================================================

@pytest.mark.unit
class TestTerminalProcessSocketPath:
    """TerminalProcess dataclass stores tmux_socket_path for mouse mode API."""

    def test_terminal_process_has_socket_path_field(self):
        """TerminalProcess has tmux_socket_path field."""
        from orchestration.terminal_manager import TerminalProcess

        tp = TerminalProcess(
            session_id="sess1", port=9000, pid=123, master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        assert tp.tmux_socket_path == "/tmp/sess1/tmux.sock"

    def test_terminal_process_socket_path_defaults_none(self):
        """TerminalProcess tmux_socket_path defaults to None."""
        from orchestration.terminal_manager import TerminalProcess

        tp = TerminalProcess(
            session_id="sess1", port=9000, pid=123, master_fd=-1,
        )
        assert tp.tmux_socket_path is None

    def test_terminal_process_tmux_session_name_defaults_none(self):
        """TerminalProcess tmux_session_name defaults to None (backward compat)."""
        from orchestration.terminal_manager import TerminalProcess

        tp = TerminalProcess(
            session_id="sess1", port=9000, pid=123, master_fd=-1,
        )
        assert tp.tmux_session_name is None


# ============================================================================
# TestMouseModeWithSocket: mouse mode API uses dedicated socket
# ============================================================================

@pytest.mark.unit
class TestMouseModeWithSocket:
    """Mouse mode toggle API must use -S socket_path for dedicated tmux server."""

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

    def _register_terminal(self, terminal_manager, session_id="sess1",
                           socket_path="/tmp/sess1/tmux.sock",
                           tmux_name="ammo-sess1"):
        """Helper to register a terminal with socket path."""
        from orchestration.terminal_manager import TerminalProcess
        terminal_manager._terminals[session_id] = TerminalProcess(
            session_id=session_id,
            port=9000,
            pid=12345,
            master_fd=-1,
            tmux_session_name=tmux_name,
            tmux_socket_path=socket_path,
        )

    @pytest.mark.asyncio
    async def test_get_mouse_mode_uses_socket(self, terminal_manager):
        """get_tmux_mouse_mode uses -S socket_path in tmux command."""
        self._register_terminal(terminal_manager)

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"on", b""))

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.wait_for', new_callable=AsyncMock) as mock_wait:
            mock_exec.return_value = mock_proc
            mock_wait.return_value = (b"on", b"")

            await terminal_manager.get_tmux_mouse_mode("sess1")

            # Verify -S socket_path is in the command
            call_args = mock_exec.call_args[0]
            assert "-S" in call_args
            socket_idx = list(call_args).index("-S")
            assert call_args[socket_idx + 1] == "/tmp/sess1/tmux.sock"

    @pytest.mark.asyncio
    async def test_set_mouse_mode_uses_socket(self, terminal_manager):
        """set_tmux_mouse_mode uses -S socket_path in tmux command."""
        self._register_terminal(terminal_manager)

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
             patch('asyncio.wait_for', new_callable=AsyncMock) as mock_wait:
            mock_exec.return_value = mock_proc
            mock_wait.return_value = (b"", b"")

            await terminal_manager.set_tmux_mouse_mode("sess1", "off")

            call_args = mock_exec.call_args[0]
            assert "-S" in call_args
            socket_idx = list(call_args).index("-S")
            assert call_args[socket_idx + 1] == "/tmp/sess1/tmux.sock"

    @pytest.mark.asyncio
    async def test_get_mouse_mode_without_socket_falls_back(self, terminal_manager):
        """When no socket path stored, get_tmux_mouse_mode defaults to 'on'."""
        from orchestration.terminal_manager import TerminalProcess
        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path=None,  # No socket path
        )

        result = await terminal_manager.get_tmux_mouse_mode("sess1")
        assert result == "on"

    def test_get_tmux_socket_path_returns_path(self, terminal_manager):
        """get_tmux_socket_path returns the stored socket path."""
        self._register_terminal(terminal_manager, socket_path="/tmp/sess1/tmux.sock")

        result = terminal_manager.get_tmux_socket_path("sess1")
        assert result == "/tmp/sess1/tmux.sock"

    def test_get_tmux_socket_path_returns_none_when_missing(self, terminal_manager):
        """get_tmux_socket_path returns None when session not found."""
        result = terminal_manager.get_tmux_socket_path("nonexistent")
        assert result is None


# ============================================================================
# TestTmuxSessionExistsWithSocket: _tmux_session_exists uses -S
# ============================================================================

@pytest.mark.unit
class TestTmuxSessionExistsWithSocket:
    """_tmux_session_exists and _kill_tmux_session use -S when socket_path is provided."""

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

    def test_session_exists_uses_socket(self, terminal_manager):
        """_tmux_session_exists includes -S socket_path when provided."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = terminal_manager._tmux_session_exists(
                "ammo-abc123", socket_path="/tmp/sess/tmux.sock"
            )

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "-S" in call_args
        assert "/tmp/sess/tmux.sock" in call_args

    def test_session_exists_without_socket_still_works(self, terminal_manager):
        """_tmux_session_exists works without socket_path (backward compat)."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = terminal_manager._tmux_session_exists("ammo-abc123")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "-S" not in call_args

    def test_kill_session_uses_socket(self, terminal_manager):
        """_kill_tmux_session includes -S socket_path when provided."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = terminal_manager._kill_tmux_session(
                "ammo-abc123", socket_path="/tmp/sess/tmux.sock"
            )

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "-S" in call_args
        assert "/tmp/sess/tmux.sock" in call_args


# ============================================================================
# TestStartTerminalStoresSocket: start_terminal_with_command stores socket
# ============================================================================

@pytest.mark.unit
class TestStartTerminalStoresSocket:
    """start_terminal_with_command stores socket path in TerminalProcess."""

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

    @pytest.mark.asyncio
    async def test_start_terminal_stores_socket_path(self, terminal_manager):
        """After start_terminal_with_command, TerminalProcess has tmux_socket_path set."""
        import os, shutil
        session_id = "sess1"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch.object(terminal_manager, '_write_hardened_tmux_config', return_value=Path("/tmp/sess1/tmux.conf")), \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "claude"],
                    working_dir=Path("/tmp/workdir"),
                    env={},
                    tmux_session_name="ammo-sess1abc12",
                )

                terminal = terminal_manager._terminals[session_id]
                assert terminal.tmux_socket_path is not None
                assert session_id in terminal.tmux_socket_path
                assert terminal.tmux_socket_path.endswith("tmux.sock")
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_start_terminal_without_tmux_no_socket(self, terminal_manager):
        """Without tmux_session_name, no socket path stored."""
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
                command=["/usr/bin/env", "claude"],
                working_dir=Path("/tmp/workdir"),
                env={},
            )

            terminal = terminal_manager._terminals["sess1"]
            assert terminal.tmux_socket_path is None


# ============================================================================
# TestStopTerminalSocketCleanup: stop_terminal uses socket for kill
# ============================================================================

@pytest.mark.unit
class TestStopTerminalSocketCleanup:
    """stop_terminal passes socket_path to _kill_tmux_session."""

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

    @pytest.mark.asyncio
    async def test_stop_terminal_passes_socket_to_kill(self, terminal_manager):
        """When stopping a terminal with socket_path, _kill_tmux_session gets the socket."""
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill') as mock_kill, \
             patch.object(terminal_manager, '_kill_tmux_session', return_value=True) as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_kill.side_effect = [None, ProcessLookupError]

            await terminal_manager.stop_terminal("sess1")

            mock_tmux_kill.assert_called_once_with(
                "ammo-sess1", socket_path="/tmp/sess1/tmux.sock"
            )

    @pytest.mark.asyncio
    async def test_stop_terminal_explicitly_kills_tmux(self, terminal_manager):
        """stop_terminal MUST explicitly call _kill_tmux_session on pause/terminate.

        With destroy-unattached off, the tmux session no longer auto-dies when the
        ttyd client detaches. Explicit _kill_tmux_session is the ONLY mechanism that
        tears down the session on user-initiated pause or terminate. This test locks
        in that invariant so no refactor accidentally drops the kill call.
        """
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess1"] = TerminalProcess(
            session_id="sess1", port=9000, pid=12345, master_fd=-1,
            tmux_session_name="ammo-sess1",
            tmux_socket_path="/tmp/sess1/tmux.sock",
        )
        terminal_manager._used_ports.add(9000)

        with patch('os.kill') as mock_kill, \
             patch.object(terminal_manager, '_kill_tmux_session', return_value=True) as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_kill.side_effect = [None, ProcessLookupError]

            result = await terminal_manager.stop_terminal("sess1")

            assert result is True
            assert mock_tmux_kill.called, (
                "stop_terminal must call _kill_tmux_session — with destroy-unattached off, "
                "the tmux session survives browser disconnect and will NOT clean itself up"
            )

    @pytest.mark.asyncio
    async def test_stop_terminal_kills_tmux_even_when_process_already_dead(self, terminal_manager):
        """_kill_tmux_session is called even when the ttyd process is already dead.

        With destroy-unattached off, a dead ttyd process leaves the tmux session
        alive (the session doesn't self-destruct on client detach). The
        ProcessLookupError path in stop_terminal must still explicitly kill the
        tmux session to avoid orphaned sessions accumulating across pauses.
        """
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess2"] = TerminalProcess(
            session_id="sess2", port=9001, pid=99999, master_fd=-1,
            tmux_session_name="ammo-sess2",
            tmux_socket_path="/tmp/sess2/tmux.sock",
        )
        terminal_manager._used_ports.add(9001)

        with patch('os.kill', side_effect=ProcessLookupError), \
             patch.object(terminal_manager, '_kill_tmux_session', return_value=True) as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            result = await terminal_manager.stop_terminal("sess2")

            assert result is True
            mock_tmux_kill.assert_called_once_with(
                "ammo-sess2", socket_path="/tmp/sess2/tmux.sock"
            )

    # ---- edge-case probe added by verifier ----
    @pytest.mark.asyncio
    async def test_stop_terminal_no_tmux_kill_when_no_session_name(self, terminal_manager):
        """When tmux_session_name is None (legacy/plain terminal), _kill_tmux_session is NOT called.

        The kill call is guarded by `if terminal.tmux_session_name:` — this test
        pins that guard so it can't be accidentally removed, which would cause
        _kill_tmux_session to be called with None and raise an error.
        """
        from orchestration.terminal_manager import TerminalProcess

        terminal_manager._terminals["sess3"] = TerminalProcess(
            session_id="sess3", port=9002, pid=77777, master_fd=-1,
            tmux_session_name=None,  # No tmux session
            tmux_socket_path=None,
        )
        terminal_manager._used_ports.add(9002)

        with patch('os.kill', side_effect=ProcessLookupError), \
             patch.object(terminal_manager, '_kill_tmux_session') as mock_tmux_kill, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            result = await terminal_manager.stop_terminal("sess3")

            assert result is True
            mock_tmux_kill.assert_not_called()


# ============================================================================
# TestTtydUidFlag: ttyd must run as session_user (UID 1000)
# ============================================================================

@pytest.mark.unit
class TestTtydUidFlag:
    """Verify ttyd is launched with --uid 1000 so terminals run as session_user."""

    def test_ttyd_command_includes_uid_flag(self):
        """terminal_manager.py source must include --uid in ttyd command."""
        content = (Path(__file__).parent.parent.parent / "orchestration" / "terminal_manager.py").read_text()
        assert '"--uid"' in content or "'--uid'" in content, \
            "ttyd command must include --uid flag to run as session_user"

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

    @pytest.mark.asyncio
    async def test_ttyd_subprocess_exec_includes_uid(self, terminal_manager):
        """The ttyd command passed to create_subprocess_exec must contain --uid 1000."""
        import os, shutil
        session_id = "sess-uid-test"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=False), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch.object(terminal_manager, '_write_hardened_tmux_config', return_value=Path("/tmp/test/tmux.conf")), \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "claude"],
                    working_dir=Path("/tmp/workdir"),
                    env={},
                    tmux_session_name="ammo-sessuid",
                )

                # Extract the args passed to create_subprocess_exec
                call_args = mock_exec.call_args[0]  # positional args tuple
                assert "--uid" in call_args, \
                    f"ttyd must be launched with --uid flag, got: {call_args}"
                uid_idx = call_args.index("--uid")
                assert call_args[uid_idx + 1] == "1000", \
                    f"--uid must be followed by '1000', got: {call_args[uid_idx + 1]}"
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)


# ============================================================================
# TestTtydSuSessionUser: ttyd child command must be wrapped with su session_user
# ============================================================================

@pytest.mark.unit
class TestTtydCapabilityAwarePrivilegeDrop:
    """ttyd child command must use capability-aware privilege drop (drop_privs.py)."""

    def test_build_tmux_command_uses_drop_privs_wrapper(self):
        """_build_tmux_command must produce a command using drop_privs.py wrapper."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            import os, shutil
            tm = TerminalManager(base_port=9000, max_ports=10)

            session_id = "test123-full-id"
            os.makedirs(f"/tmp/{session_id}", exist_ok=True)
            try:
                cmd, _, _ = tm._build_tmux_command(
                    "ammo-test123",
                    ["/usr/bin/env", "FOO=bar", "claude"],
                    attach_only=False,
                    session_id=session_id,
                )
                joined = " ".join(cmd)
                assert "drop_privs.py" in joined, \
                    f"_build_tmux_command must use drop_privs.py wrapper, got: {joined}"
            finally:
                shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_build_tmux_command_attach_uses_drop_privs_wrapper(self):
        """_build_tmux_command with attach_only=True must also use drop_privs.py wrapper."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            import shutil
            tm = TerminalManager(base_port=9000, max_ports=10)

            try:
                cmd, _, _ = tm._build_tmux_command(
                    "ammo-test123",
                    [],
                    attach_only=True,
                    session_id="test123-full-id",
                )
                joined = " ".join(cmd)
                assert "drop_privs.py" in joined, \
                    f"_build_tmux_command attach must use drop_privs.py wrapper, got: {joined}"
            finally:
                shutil.rmtree("/tmp/test123-full-id", ignore_errors=True)

    def test_home_env_set_to_session_user_home(self):
        """session_manager.py must set HOME=/home/session_user, not HOME=/root."""
        content = (Path(__file__).parent.parent.parent / "orchestration" / "session_manager.py").read_text()
        assert 'HOME": "/home/session_user"' in content or "HOME\": \"/home/session_user\"" in content, \
            "session_manager.py must set HOME=/home/session_user, not /root"
        assert 'HOME": "/root"' not in content, \
            "session_manager.py must NOT set HOME=/root (process runs as session_user)"


# ============================================================================
# TestDropPrivsCapsetInheritable: drop_privs.py must set inheritable caps
# ============================================================================

@pytest.mark.unit
class TestDropPrivsCapsetInheritable:
    """drop_privs.py script must set inheritable caps via capset before setuid."""

    def test_drop_privs_script_contains_capset_inheritable(self):
        """Read the generated drop_privs.py content and verify capset + inheritable."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            import shutil
            tm = TerminalManager(base_port=9000, max_ports=10)

            session_id = "test-capset-id"
            os.makedirs(f"/tmp/{session_id}", exist_ok=True)
            try:
                tm._build_tmux_command(
                    "ammo-test-capset",
                    ["/usr/bin/env", "claude"],
                    attach_only=False,
                    session_id=session_id,
                )
                script = Path(f"/tmp/{session_id}/drop_privs.py").read_text()
                assert "SYS_capset" in script, (
                    "drop_privs.py must use SYS_capset to set inheritable caps"
                )
                assert ".inheritable |=" in script, (
                    "drop_privs.py must set the inheritable bitmask"
                )
            finally:
                shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)

    def test_drop_privs_inheritable_before_setuid(self):
        """Inheritable caps must be set before setuid in drop_privs.py."""
        with patch(
            'orchestration.terminal_manager.shutil.which',
            side_effect=lambda name: {
                "ttyd": "/usr/bin/ttyd",
                "tmux": "/usr/bin/tmux",
            }.get(name),
        ):
            from orchestration.terminal_manager import TerminalManager
            import shutil
            tm = TerminalManager(base_port=9000, max_ports=10)

            session_id = "test-capset-order"
            os.makedirs(f"/tmp/{session_id}", exist_ok=True)
            try:
                tm._build_tmux_command(
                    "ammo-test-order",
                    ["/usr/bin/env", "claude"],
                    attach_only=False,
                    session_id=session_id,
                )
                script = Path(f"/tmp/{session_id}/drop_privs.py").read_text()
                inh_pos = script.find(".inheritable |=")
                uid_pos = script.find("os.setuid(1000)")
                assert inh_pos != -1 and uid_pos != -1, (
                    "drop_privs.py must contain both .inheritable and setuid"
                )
                assert inh_pos < uid_pos, (
                    "drop_privs.py must set inheritable caps BEFORE setuid "
                    f"(inheritable at {inh_pos}, setuid at {uid_pos})"
                )
            finally:
                shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)


# ============================================================================
# TestSetEnvironmentWithSocket: tmux set-environment uses -S
# ============================================================================

@pytest.mark.unit
class TestSetEnvironmentWithSocket:
    """tmux set-environment calls in start_terminal_with_command use -S socket."""

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

    @pytest.mark.asyncio
    async def test_set_environment_uses_socket_for_existing_session(self, terminal_manager):
        """When tmux session exists, set-environment commands use -S socket."""
        import os, shutil
        session_id = "sess1"
        os.makedirs(f"/tmp/{session_id}", exist_ok=True)
        try:
            with patch.object(terminal_manager, '_tmux_session_exists', return_value=True), \
                 patch.object(terminal_manager, '_is_port_available', return_value=True), \
                 patch.object(terminal_manager, '_is_port_in_use', return_value=True), \
                 patch.object(terminal_manager, '_write_hardened_tmux_config', return_value=Path("/tmp/sess1/tmux.conf")), \
                 patch('subprocess.run') as mock_run, \
                 patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec, \
                 patch('asyncio.sleep', new_callable=AsyncMock):

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None
                mock_exec.return_value = mock_process

                await terminal_manager.start_terminal_with_command(
                    session_id=session_id,
                    command=["/usr/bin/env", "claude"],
                    working_dir=Path("/tmp/workdir"),
                    env={"FOO": "bar"},
                    tmux_session_name="ammo-sess1",
                )

                # At least one subprocess.run call should have -S in it
                for c in mock_run.call_args_list:
                    cmd_list = c[0][0]
                    if "set-environment" in cmd_list:
                        assert "-S" in cmd_list, \
                            f"set-environment must use -S socket, got: {cmd_list}"
        finally:
            shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True)


# ============================================================================
# TestTmuxConfigDirOwnership: config dir must be owned by session_user
# ============================================================================

@pytest.mark.unit
class TestTmuxConfigDirOwnership:
    """The /tmp/{session_id}/ directory created by _write_hardened_tmux_config()
    must be owned by session_user (uid 1000, gid 1000) so that `su session_user`
    can create the tmux socket file inside it.

    Bug: The server runs as root, so mkdir() creates a root-owned directory.
    When tmux is launched via `su session_user -c 'tmux -S /tmp/{session_id}/tmux.sock ...'`,
    session_user cannot create the socket in a root-owned directory.

    Fix: chown the directory to session_user after creation.
    """

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

    def test_config_dir_chowned_to_session_user(self, terminal_manager, tmp_path):
        """After _write_hardened_tmux_config(), the config directory must be
        chowned to uid=1000, gid=1000 (session_user) so tmux can create its socket."""
        session_id = "test-chown-abc123"

        with patch('orchestration.terminal_manager.os.chown') as mock_chown:
            # Use a real tmp_path so mkdir actually works
            config_dir = tmp_path / session_id
            config_path = config_dir / "tmux.conf"

            with patch('orchestration.terminal_manager.Path', return_value=config_path):
                terminal_manager._write_hardened_tmux_config(session_id)

            # os.chown must be called on the config directory with uid=1000, gid=1000
            mock_chown.assert_called_once_with(str(config_dir), 1000, 1000)

    # ---- edge-case probe added by verifier ----
    def test_config_destroy_unattached_appears_exactly_once(self, terminal_manager, tmp_path):
        """destroy-unattached must appear exactly once in active config lines.

        Having two destroy-unattached lines (e.g. 'off' then 'on') would silently
        revert the fix because tmux uses the last value. This regression guard
        ensures no duplicate or conflicting setting sneaks in.
        """
        config_path = tmp_path / "tmux.conf"
        with patch('orchestration.terminal_manager.Path', return_value=config_path):
            config_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_manager._write_hardened_tmux_config("sess")
            content = config_path.read_text()

        active_lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        da_lines = [l for l in active_lines if "destroy-unattached" in l]
        assert len(da_lines) == 1, (
            f"Expected exactly 1 active destroy-unattached line, got {len(da_lines)}: {da_lines}"
        )

    def test_config_dir_chown_called_after_mkdir(self, terminal_manager, tmp_path):
        """os.chown must be called AFTER mkdir (directory must exist before chown)."""
        session_id = "test-order-abc123"
        call_order = []

        original_mkdir = Path.mkdir

        def tracking_mkdir(self_path, *args, **kwargs):
            call_order.append('mkdir')
            original_mkdir(self_path, *args, **kwargs)

        def tracking_chown(*args, **kwargs):
            call_order.append('chown')

        with patch('orchestration.terminal_manager.os.chown', side_effect=tracking_chown):
            config_dir = tmp_path / session_id
            config_path = config_dir / "tmux.conf"

            with patch('orchestration.terminal_manager.Path', return_value=config_path), \
                 patch.object(Path, 'mkdir', tracking_mkdir):
                terminal_manager._write_hardened_tmux_config(session_id)

            assert 'mkdir' in call_order, "mkdir must be called"
            assert 'chown' in call_order, "chown must be called"
            assert call_order.index('mkdir') < call_order.index('chown'), \
                f"mkdir must happen before chown, got order: {call_order}"
