# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for ttyd LIGHTGRID theme injection in TerminalManager.

TDD tests -- written before the implementation.

Tests verify that:
- _build_ttyd_theme_args() returns a list of strings
- The list contains --client-option flags
- The LIGHTGRID hex colors are present (#00f3ff cyan, #0a0a12 background, #e8e8f0 foreground, etc.)
- Theme args are injected BEFORE the -- separator in start_terminal_with_command
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.terminal_manager import TerminalManager


@pytest.fixture
def terminal_manager():
    """Create a TerminalManager with ttyd and tmux mocked as available."""
    with patch("shutil.which", return_value="/usr/bin/ttyd"):
        tm = TerminalManager(base_port=7680, max_ports=10)
        tm._ttyd_available = True
        tm._tmux_available = True
        return tm


class TestBuildTtydThemeArgs:
    """Tests for TerminalManager._build_ttyd_theme_args()."""

    def test_returns_list_of_strings(self, terminal_manager):
        """_build_ttyd_theme_args() must return a list of str."""
        result = terminal_manager._build_ttyd_theme_args()
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result), \
            "All elements must be strings"

    def test_contains_client_option_flags(self, terminal_manager):
        """Result must contain --client-option flags."""
        result = terminal_manager._build_ttyd_theme_args()
        assert "--client-option" in result

    def test_renderer_type_canvas(self, terminal_manager):
        """rendererType=canvas must be in the args."""
        result = terminal_manager._build_ttyd_theme_args()
        assert "rendererType=canvas" in result

    def test_theme_flag_present(self, terminal_manager):
        """A theme=... --client-option value must be present."""
        result = terminal_manager._build_ttyd_theme_args()
        theme_values = [
            v for i, v in enumerate(result)
            if i > 0 and result[i - 1] == "--client-option" and v.startswith("theme=")
        ]
        assert len(theme_values) == 1, \
            "Exactly one --client-option theme=... flag expected"

    def _parse_theme(self, terminal_manager) -> dict:
        """Helper: extract and parse the JSON theme dict from args."""
        result = terminal_manager._build_ttyd_theme_args()
        for i, v in enumerate(result):
            if result[i - 1] == "--client-option" and v.startswith("theme="):
                return json.loads(v[len("theme="):])
        pytest.fail("No theme= value found in args")

    def test_background_color(self, terminal_manager):
        """background must be #000000 (LIGHTGRID void)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("background") == "#000000"

    def test_foreground_color(self, terminal_manager):
        """foreground must be #c8c8d0 (LIGHTGRID text)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("foreground") == "#c8c8d0"

    def test_cursor_color(self, terminal_manager):
        """cursor must be #00f3ff (LIGHTGRID cyan)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("cursor") == "#00f3ff"

    def test_blue_ansi_color(self, terminal_manager):
        """ANSI blue must be #00f3ff (LIGHTGRID cyan)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("blue") == "#00f3ff"

    def test_magenta_ansi_color(self, terminal_manager):
        """ANSI magenta must be #ff00aa (LIGHTGRID magenta)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("magenta") == "#ff00aa"

    def test_green_ansi_color(self, terminal_manager):
        """ANSI green must be #00ffb2 (LIGHTGRID mint)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("green") == "#00ffb2"

    def test_red_ansi_color(self, terminal_manager):
        """ANSI red must be #ff3355 (LIGHTGRID red)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("red") == "#ff3355"

    def test_yellow_ansi_color(self, terminal_manager):
        """ANSI yellow must be #ffaa00 (LIGHTGRID amber)."""
        theme = self._parse_theme(terminal_manager)
        assert theme.get("yellow") == "#ffaa00"

    def test_theme_has_all_required_keys(self, terminal_manager):
        """Theme must include all 16 ANSI colors + cursor, background, foreground."""
        required_keys = {
            "background", "foreground", "cursor", "cursorAccent",
            "selectionBackground",
            "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
            "brightBlack", "brightRed", "brightGreen", "brightYellow",
            "brightBlue", "brightMagenta", "brightCyan", "brightWhite",
        }
        theme = self._parse_theme(terminal_manager)
        missing = required_keys - set(theme.keys())
        assert not missing, f"Theme missing keys: {missing}"


class TestThemeInjectedIntoTtydCommand:
    """Tests that _build_ttyd_theme_args() is injected BEFORE -- in start_terminal_with_command."""

    @pytest.mark.asyncio
    async def test_theme_args_before_separator(self, terminal_manager):
        """Theme --client-option flags appear before -- in the final ttyd command."""
        captured_cmd = []

        async def mock_subprocess(*args, **kwargs):
            captured_cmd.extend(args)  # args = ('ttyd', '--port', ...) — each element separate
            proc = MagicMock()
            proc.pid = 12345
            proc.returncode = None
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch.object(terminal_manager, "_is_port_in_use", return_value=True), \
             patch.object(terminal_manager, "_write_hardened_tmux_config"), \
             patch.object(terminal_manager, "_tmux_session_exists", return_value=False), \
             patch.object(terminal_manager, "_build_tmux_command",
                         return_value=(["/bin/bash", "-c", "tmux ..."], "/tmp/s/tmux.sock", "/tmp/s/tmux.conf")):
            await terminal_manager.start_terminal_with_command(
                session_id="test-session",
                command=["claude"],
                working_dir=Path("/tmp"),
                env={},
                tmux_session_name="test-tmux",
            )

        assert captured_cmd, "No command was captured"

        # Find the index of '--' separator
        try:
            sep_idx = captured_cmd.index("--")
        except ValueError:
            pytest.fail(f"'--' separator not found in command: {captured_cmd}")

        # '--client-option' must appear before '--'
        pre_sep = captured_cmd[:sep_idx]
        assert "--client-option" in pre_sep, \
            f"--client-option not found before '--' separator. Full cmd: {captured_cmd}"

        # 'rendererType=canvas' must appear before '--'
        assert "rendererType=canvas" in pre_sep, \
            f"rendererType=canvas not found before '--' separator. Full cmd: {captured_cmd}"


class TestThemeInjectedOnRestart:
    """Edge-case: LIGHTGRID theme must also be injected in restart_ttyd_with_tmux_attach.

    When ttyd dies and is restarted (browser reconnect), the LIGHTGRID theme
    must still be applied so users don't see a plain white terminal on reconnect.
    """

    @pytest.mark.asyncio
    async def test_theme_injected_on_tmux_reattach(self, terminal_manager):
        """Theme --client-option flags appear in restart_ttyd_with_tmux_attach command."""
        captured_cmd = []

        async def mock_subprocess(*args, **kwargs):
            captured_cmd.extend(args)
            proc = MagicMock()
            proc.pid = 99999
            proc.returncode = None
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess), \
             patch.object(terminal_manager, "_is_port_in_use", return_value=True), \
             patch.object(terminal_manager, "_tmux_session_exists", return_value=True), \
             patch.object(terminal_manager, "_build_tmux_command",
                         return_value=(["/bin/bash", "-c", "tmux attach ..."], "/tmp/s/tmux.sock", "/tmp/s/tmux.conf")):
            await terminal_manager.restart_ttyd_with_tmux_attach(
                session_id="restart-session",
                tmux_session_name="restart-tmux",
                working_dir=Path("/tmp"),
                env={},
            )

        assert captured_cmd, "No command was captured during restart"

        assert "--client-option" in captured_cmd, \
            f"--client-option not found in restart command: {captured_cmd}"
        assert "rendererType=canvas" in captured_cmd, \
            f"rendererType=canvas not injected in restart command: {captured_cmd}"
