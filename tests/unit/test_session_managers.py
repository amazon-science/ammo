# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for session manager components.
"""

import pytest
import sys
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
    CreateSessionRequest,
    SUPPORTED_REPOS,
)


@pytest.mark.unit
class TestWorktreeManager:
    """Test WorktreeManager functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        repos_dir = tempfile.mkdtemp()
        sessions_dir = tempfile.mkdtemp()
        yield repos_dir, sessions_dir
        shutil.rmtree(repos_dir, ignore_errors=True)
        shutil.rmtree(sessions_dir, ignore_errors=True)

    def test_worktree_manager_init(self, temp_dirs):
        """Test WorktreeManager initialization."""
        from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

        reset_worktree_manager()
        repos_dir, sessions_dir = temp_dirs

        manager = WorktreeManager(
            repos_dir=repos_dir,
            sessions_dir=sessions_dir,
        )

        assert manager.repos_dir == Path(repos_dir)
        assert manager.sessions_dir == Path(sessions_dir)
        assert manager.repos_dir.exists()
        assert manager.sessions_dir.exists()

    def test_get_repo_config(self, temp_dirs):
        """Test getting repository configuration."""
        from orchestration.worktree_manager import WorktreeManager, WorktreeError, reset_worktree_manager

        reset_worktree_manager()
        repos_dir, sessions_dir = temp_dirs
        manager = WorktreeManager(repos_dir=repos_dir, sessions_dir=sessions_dir)

        config = manager.get_repo_config("vllm")
        assert "url" in config
        assert "default_branch" in config

        with pytest.raises(WorktreeError):
            manager.get_repo_config("nonexistent")

    def test_get_session_paths(self, temp_dirs):
        """Test session path generation."""
        from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

        reset_worktree_manager()
        repos_dir, sessions_dir = temp_dirs
        manager = WorktreeManager(repos_dir=repos_dir, sessions_dir=sessions_dir)

        session_id = "test-session-123"

        session_dir = manager.get_session_dir(session_id)
        assert str(session_dir) == f"{sessions_dir}/{session_id}"

        worktree_path = manager.get_worktree_path(session_id)
        assert str(worktree_path) == f"{sessions_dir}/{session_id}/worktree"

        logs_dir = manager.get_logs_dir(session_id)
        assert str(logs_dir) == f"{sessions_dir}/{session_id}/logs"

    def test_create_session_dirs(self, temp_dirs):
        """Test session directory creation."""
        from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

        reset_worktree_manager()
        repos_dir, sessions_dir = temp_dirs
        manager = WorktreeManager(repos_dir=repos_dir, sessions_dir=sessions_dir)

        session_id = "test-dirs-123"
        paths = manager.create_session_dirs(session_id)

        assert paths["session_dir"].exists()
        assert paths["logs_dir"].exists()


@pytest.mark.unit
class TestInactivityMonitor:
    """Test InactivityMonitor functionality."""

    def test_monitor_init(self):
        """Test InactivityMonitor initialization."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(
            default_timeout_mins=30,
            check_interval_seconds=60,
        )

        assert monitor.default_timeout_mins == 30
        assert monitor.check_interval == 60

    def test_register_session(self):
        """Test registering a session for monitoring."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=30)
        monitor.register_session("session-1", timeout_mins=45)

        activity = monitor.get_session_activity("session-1")
        assert activity is not None
        assert activity["session_id"] == "session-1"
        assert activity["timeout_mins"] == 45

    def test_unregister_session(self):
        """Test unregistering a session."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=30)
        monitor.register_session("session-1")
        monitor.unregister_session("session-1")

        activity = monitor.get_session_activity("session-1")
        assert activity is None

    def test_record_activity(self):
        """Test recording activity."""
        import time
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=30)
        monitor.register_session("session-1")

        initial_activity = monitor.get_session_activity("session-1")
        initial_time = initial_activity["last_activity"]

        time.sleep(0.1)
        monitor.record_activity("session-1")

        updated_activity = monitor.get_session_activity("session-1")
        assert updated_activity["last_activity"] > initial_time

    def test_time_until_timeout(self):
        """Test time until timeout calculation."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=1)  # 1 minute
        monitor.register_session("session-1")

        remaining = monitor.get_time_until_timeout("session-1")
        assert remaining is not None
        assert 55 < remaining <= 60  # Should be close to 60 seconds

    def test_is_session_timed_out(self):
        """Test timeout detection."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=1)
        monitor.register_session("session-1")

        # Should not be timed out initially
        assert not monitor.is_session_timed_out("session-1")

    def test_register_child_process(self):
        """Test registering child processes."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=30)
        monitor.register_session("session-1")
        monitor.register_child_process("session-1", 12345)
        monitor.register_child_process("session-1", 12346)

        activity = monitor.get_session_activity("session-1")
        assert 12345 in activity["child_pids"]
        assert 12346 in activity["child_pids"]

    def test_unregister_child_process(self):
        """Test unregistering child processes."""
        from orchestration.inactivity_monitor import InactivityMonitor

        monitor = InactivityMonitor(default_timeout_mins=30)
        monitor.register_session("session-1")
        monitor.register_child_process("session-1", 12345)
        monitor.unregister_child_process("session-1", 12345)

        activity = monitor.get_session_activity("session-1")
        assert 12345 not in activity["child_pids"]


@pytest.mark.unit
class TestTerminalManager:
    """Test TerminalManager functionality."""

    def test_manager_init(self):
        """Test TerminalManager initialization."""
        from orchestration.terminal_manager import TerminalManager

        manager = TerminalManager(base_port=9000, max_ports=50)

        assert manager.base_port == 9000
        assert manager.max_ports == 50

    def test_port_allocation(self):
        """Test port allocation logic."""
        from orchestration.terminal_manager import TerminalManager

        manager = TerminalManager(base_port=19000, max_ports=5)

        # Simulate allocation
        manager._used_ports.add(19000)
        manager._used_ports.add(19001)

        # Next available should be 19002
        port = manager._allocate_port()
        assert port == 19002
        assert port in manager._used_ports

    def test_port_release(self):
        """Test port release."""
        from orchestration.terminal_manager import TerminalManager

        manager = TerminalManager(base_port=19000)
        manager._used_ports.add(19000)

        manager._release_port(19000)
        assert 19000 not in manager._used_ports

    def test_get_terminal_url(self):
        """Test terminal URL generation."""
        from orchestration.terminal_manager import TerminalManager, TerminalProcess

        manager = TerminalManager()
        manager._terminals["test-session"] = TerminalProcess(
            session_id="test-session",
            port=8001,
            pid=12345,
            master_fd=-1,
        )

        url = manager.get_terminal_url("test-session", host="example.com")
        assert url == "http://example.com:8001/"

    def test_get_terminal_url_not_found(self):
        """Test terminal URL when session not found."""
        from orchestration.terminal_manager import TerminalManager

        manager = TerminalManager()
        url = manager.get_terminal_url("nonexistent")
        assert url is None


@pytest.mark.unit
class TestCLIToolManager:
    """Test CLIToolManager functionality."""

    @pytest.fixture
    def temp_worktree(self):
        """Create temporary worktree directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def template_settings(self):
        """The server-side settings.local.json that setup_claude_workspace copies.

        Model IDs are read from the template, not hardcoded, so a model bump does
        not turn these tests red. The contract under test is "the workspace keeps
        the template's model config", not any one model id.
        """
        import json

        template = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "settings.local.json"
        )
        assert template.exists(), f"Missing model-config template: {template}"
        with open(template) as f:
            return json.load(f)

    def test_setup_claude_workspace(self, temp_worktree):
        """Test setting up Claude workspace."""
        from orchestration.cli_tool_manager import CLIToolManager

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session",
            gpu_ids=[0],
        )

        # Check .claude directory was created
        claude_dir = temp_worktree / ".claude"
        assert claude_dir.exists()

    def test_setup_claude_workspace_injects_cuda_visible_devices_into_settings_local(self, temp_worktree):
        """settings.local.json in session .claude/ dir must include CUDA_VISIBLE_DEVICES in env section.

        Root cause: Claude Code subagents spawned via Task tool read env vars from
        settings.local.json's "env" section, NOT from the process environment.
        Without CUDA_VISIBLE_DEVICES in settings.local.json, subagents lose GPU access.
        """
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session",
            gpu_ids=[2, 5],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists(), "settings.local.json must exist in session .claude/"

        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        assert "CUDA_VISIBLE_DEVICES" in env_section, \
            "settings.local.json must have CUDA_VISIBLE_DEVICES in env section"
        assert env_section["CUDA_VISIBLE_DEVICES"] == "2,5", \
            f"Expected '2,5', got '{env_section.get('CUDA_VISIBLE_DEVICES')}'"

    def test_setup_claude_workspace_no_gpus_sets_empty_cuda_visible_devices(self, temp_worktree):
        """When gpu_ids is empty, CUDA_VISIBLE_DEVICES should be empty string (not absent)."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-no-gpu",
            gpu_ids=[],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        assert "CUDA_VISIBLE_DEVICES" in env_section
        assert env_section["CUDA_VISIBLE_DEVICES"] == ""

    def test_setup_claude_workspace_preserves_existing_env_vars_in_settings_local(self, temp_worktree):
        """Injecting CUDA_VISIBLE_DEVICES must not remove existing env vars like
        CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-preserve",
            gpu_ids=[0],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        assert env_section.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS") == "1", \
            "Existing env vars must be preserved"
        assert env_section.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_cli_tool_config_env_vars_have_no_model_ids(self):
        """CLI_TOOL_CONFIGS env_vars must not contain hardcoded model IDs.
        Model configuration lives in settings.local.json and managed-settings.json."""
        from orchestration.cli_tool_manager import CLI_TOOL_CONFIGS

        env_vars = CLI_TOOL_CONFIGS[CLIToolType.CLAUDE].env_vars
        # Model env vars must NOT be present
        assert "ANTHROPIC_MODEL" not in env_vars
        assert "CLAUDE_MODEL" not in env_vars
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env_vars
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env_vars
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in env_vars
        # Operational env vars must still be present
        assert env_vars.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") == "1"
        assert env_vars.get("DISABLE_AUTOUPDATER") == "1"
        assert env_vars.get("DISABLE_COST_WARNINGS") == "1"

    def test_cli_tool_config_startup_flags_no_model_flag(self):
        """startup_flags must not contain --model. Model is set via settings files."""
        from orchestration.cli_tool_manager import CLI_TOOL_CONFIGS

        flags = CLI_TOOL_CONFIGS[CLIToolType.CLAUDE].startup_flags
        assert "--model" not in flags
        assert len(flags) == 0

    def test_cli_tool_config_claude_uses_native_binary(self):
        """CC 2.1.114+ ships as a native ELF binary, not node cli.js."""
        from orchestration.cli_tool_manager import CLI_TOOL_CONFIGS

        config = CLI_TOOL_CONFIGS[CLIToolType.CLAUDE]
        assert config.command == "/usr/bin/claude"
        assert "node" not in config.command
        assert not any("cli.js" in f for f in config.startup_flags)

    def test_setup_claude_workspace_does_not_inject_model_env_vars(self, temp_worktree, template_settings):
        """setup_claude_workspace must only inject CUDA_VISIBLE_DEVICES, not model env vars.
        Model config comes from the template settings.local.json."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-no-model-inject",
            gpu_ids=[3],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        # CUDA_VISIBLE_DEVICES must still be injected
        assert env_section["CUDA_VISIBLE_DEVICES"] == "3"
        # Model env vars must come from template, not be overwritten
        template_env = template_settings["env"]
        for key in (
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        ):
            assert env_section.get(key) == template_env[key], \
                f"{key} must be preserved verbatim from the template"
        # ANTHROPIC_MODEL must NOT be injected (not in template either)
        assert "ANTHROPIC_MODEL" not in env_section

    def test_settings_local_has_prompt_caching_and_model_defaults(self, temp_worktree, template_settings):
        """Template must include 1h prompt caching, xhigh effort, and the
        template's default Opus model."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-model-defaults",
            gpu_ids=[0],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        env = settings.get("env", {})
        assert env.get("ENABLE_PROMPT_CACHING_1H") == "1", \
            "1h prompt caching must be enabled"
        assert env.get("CLAUDE_CODE_EFFORT_LEVEL") == "xhigh", \
            "Effort level must mirror user env (xhigh)"
        expected_opus = template_settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"]
        assert env.get("ANTHROPIC_DEFAULT_OPUS_MODEL") == expected_opus, \
            "Default Opus model must match the template"
        assert settings.get("model") == template_settings["model"], \
            "Top-level model key must match the template"

    def test_settings_json_does_not_contain_hooks(self, temp_worktree):
        """settings.json must NOT contain hooks — hooks belong only in settings.local.json.

        Root cause: _create_sandboxed_settings() reads settings.local.json as template
        and writes hooks into settings.json. Since shutil.copytree also copies
        settings.local.json (with hooks), Claude Code merges both files additively,
        causing every hook to fire twice.
        """
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-hooks",
            gpu_ids=[0],
        )

        settings_path = temp_worktree / ".claude" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
            assert "hooks" not in settings, (
                "settings.json should not contain hooks — they belong only in settings.local.json"
            )

    def test_settings_local_json_still_contains_hooks(self, temp_worktree):
        """settings.local.json must retain hooks after workspace setup."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-hooks-local",
            gpu_ids=[0],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists()
        with open(settings_local_path) as f:
            settings = json.load(f)
        assert "hooks" in settings, "settings.local.json must contain hooks"
        assert "PreToolUse" in settings["hooks"], "Missing PreToolUse hooks"
        assert "Stop" in settings["hooks"], "Missing Stop hook"

    def test_no_hooks_in_both_settings_files(self, temp_worktree):
        """Hooks must not appear in both settings.json and settings.local.json."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-hooks-both",
            gpu_ids=[0],
        )

        settings_path = temp_worktree / ".claude" / "settings.json"
        settings_local_path = temp_worktree / ".claude" / "settings.local.json"

        has_hooks_in_json = False
        if settings_path.exists():
            with open(settings_path) as f:
                has_hooks_in_json = "hooks" in json.load(f)

        has_hooks_in_local = False
        if settings_local_path.exists():
            with open(settings_local_path) as f:
                has_hooks_in_local = "hooks" in json.load(f)

        assert not (has_hooks_in_json and has_hooks_in_local), (
            "Hooks present in BOTH settings.json and settings.local.json — "
            "Claude Code merges them additively, causing every hook to fire twice"
        )

    def test_settings_json_retains_deny_rules(self, temp_worktree):
        """settings.json must still contain session-specific deny rules after hooks removal."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-deny",
            gpu_ids=[0],
        )

        settings_path = temp_worktree / ".claude" / "settings.json"
        assert settings_path.exists()
        with open(settings_path) as f:
            settings = json.load(f)
        assert "permissions" in settings
        deny = settings["permissions"].get("deny", [])
        assert len(deny) > 0, "settings.json must have deny rules"
        additional = settings["permissions"].get("additionalDirectories", [])
        assert str(temp_worktree) in additional, "worktree path must be in additionalDirectories"

    def test_settings_json_does_not_duplicate_model_or_env(self, temp_worktree):
        """settings.json should not contain model/env/effortLevel — those belong in settings.local.json."""
        from orchestration.cli_tool_manager import CLIToolManager
        import json

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id="test-session-no-dup",
            gpu_ids=[0],
        )

        settings_path = temp_worktree / ".claude" / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
            assert "model" not in settings, "model should only be in settings.local.json"
            assert "env" not in settings, "env should only be in settings.local.json"
            assert "effortLevel" not in settings, "effortLevel should only be in settings.local.json"

    def test_get_cli_command_no_model_flag(self, template_settings):
        """get_cli_command must not include --model or model ID strings."""
        from orchestration.cli_tool_manager import CLIToolManager
        import unittest.mock

        manager = CLIToolManager()
        with unittest.mock.patch("shutil.which", return_value="/usr/bin/node"):
            cmd = manager.get_cli_command(CLIToolType.CLAUDE)

        assert "--model" not in cmd
        # No element may contain a template model ID. The ids come from the template
        # so a model bump cannot leave this check looking for a retired name.
        model_ids = {
            value
            for key, value in template_settings["env"].items()
            if key.startswith("ANTHROPIC_DEFAULT_") and key.endswith("_MODEL")
        }
        model_ids.add(template_settings["model"])
        for part in cmd:
            for model_str in model_ids:
                assert model_str not in part, \
                    f"Command part '{part}' contains model ID '{model_str}'"

    def test_get_cli_command_codex_uses_codex_binary_and_prompt(self):
        """Codex sessions must launch the Codex CLI with the initial prompt."""
        from orchestration.cli_tool_manager import CLIToolManager
        import unittest.mock

        manager = CLIToolManager()
        with unittest.mock.patch("shutil.which", return_value="/usr/bin/codex"):
            cmd = manager.get_cli_command(
                CLIToolType.CODEX,
                extra_env={"CODEX_HOME": "/data/sessions/s1/codex-home"},
                initial_prompt="Use ammo",
            )

        assert cmd[:2] == ["/usr/bin/env", "CODEX_HOME=/data/sessions/s1/codex-home"]
        assert "/usr/bin/codex" in cmd
        assert "Use ammo" == cmd[-1]
        assert "claude" not in " ".join(cmd)

    def test_get_cli_command_codex_resume_uses_resume_last(self):
        """Codex resume should use the interactive resume subcommand, not Claude --continue."""
        from orchestration.cli_tool_manager import CLIToolManager
        import unittest.mock

        manager = CLIToolManager()
        with unittest.mock.patch("shutil.which", return_value="/usr/bin/codex"):
            cmd = manager.get_cli_command(
                CLIToolType.CODEX,
                initial_prompt="Continue the campaign",
                is_resume=True,
            )

        codex_idx = cmd.index("/usr/bin/codex")
        assert cmd[codex_idx + 1:codex_idx + 3] == ["resume", "--last"]
        assert "--continue" not in cmd
        assert cmd[-1] == "Continue the campaign"


@pytest.mark.unit
class TestGPUReleasedAfterTerminalConfirmedDead:
    """
    Tests for Bug #8: GPUs released before terminal confirmed dead.

    `_release_gpus()` runs after `stop_terminal()`, but if stop_terminal() fails,
    GPUs are freed while CLI tool still running.

    Fix: Only release GPUs after terminal is confirmed stopped. If stop_terminal()
    fails, escalate to force-kill child processes, THEN release GPUs.
    The warning should become an error, and process cleanup should happen first.
    """

    def _make_session_manager(self, tmp_path):
        """Create a SessionManager with fully mocked dependencies."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager

        reset_worktree_manager()
        sessions_dir = str(tmp_path / "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        mock_worktree = MagicMock()
        mock_worktree.cleanup_session = MagicMock()
        mock_worktree.get_worktree_path = MagicMock(return_value=tmp_path / "wt")
        mock_worktree.get_logs_dir = MagicMock(return_value=tmp_path / "logs")
        mock_worktree.get_session_dir = MagicMock(return_value=tmp_path / "sd")

        mock_gpu = MagicMock()
        mock_gpu.get_gpu_count.return_value = 4
        mock_gpu.get_available_gpu_count.return_value = 4
        mock_gpu.release_gpus_for_session = MagicMock()

        mock_terminal = MagicMock()
        mock_terminal.stop_terminal = AsyncMock()

        mock_cli = MagicMock()
        mock_cli.stop_cli_tool = MagicMock()

        mock_inactivity = MagicMock()
        mock_inactivity.unregister_session = MagicMock()
        mock_inactivity.get_session_activity = MagicMock(return_value=None)

        mock_storage = MagicMock()
        mock_storage.enabled = False

        manager = SessionManager(
            sessions_dir=sessions_dir,
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        return manager, mock_gpu, mock_terminal

    def _make_active_session(self, session_manager, session_id="test-session"):
        """Create an active session state in the manager."""
        from shared.session_models import SessionState, SessionStatus, CLIToolType
        import time as _time

        state = SessionState(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            gpu_ids=[0],
            created_at=_time.time(),
            last_accessed=_time.time(),
            terminal_port=8001,
            worktree_path="/tmp/wt",
        )
        session_manager._sessions[session_id] = state
        return state

    @pytest.mark.asyncio
    async def test_gpus_released_after_terminal_stops_successfully(self, tmp_path):
        """
        Normal path: terminal stops successfully, THEN GPUs released.
        Verify ordering: stop_terminal called before release_gpus.
        """
        manager, mock_gpu, mock_terminal = self._make_session_manager(tmp_path)
        state = self._make_active_session(manager)

        call_order = []

        async def track_stop_terminal(session_id):
            call_order.append("stop_terminal")

        def track_release_gpus(session_id):
            call_order.append("release_gpus")

        mock_terminal.stop_terminal = track_stop_terminal
        mock_gpu.release_gpus_for_session = track_release_gpus

        await manager.terminate_session("test-session")

        assert "stop_terminal" in call_order
        assert "release_gpus" in call_order
        terminal_idx = call_order.index("stop_terminal")
        gpu_idx = call_order.index("release_gpus")
        assert terminal_idx < gpu_idx, (
            f"stop_terminal ({terminal_idx}) should come before release_gpus ({gpu_idx}). "
            f"Got order: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_gpus_released_only_after_force_kill_when_terminal_stop_fails(self, tmp_path):
        """
        Bug #8 fix: When stop_terminal() fails, GPUs must NOT be released until
        after force-killing child processes. The terminal failure should be logged
        as an error (not just a warning), and process cleanup (os.kill) should happen first.

        Current buggy code: stop_terminal() fails with warning, GPUs released immediately.
        Fixed code: stop_terminal() fails, force-kill child processes, THEN release GPUs.
        """
        manager, mock_gpu, mock_terminal = self._make_session_manager(tmp_path)
        state = self._make_active_session(manager)

        call_order = []

        async def failing_stop_terminal(session_id):
            call_order.append("stop_terminal_called")
            raise Exception("ttyd failed to stop")

        def track_release_gpus(session_id):
            call_order.append("release_gpus")

        mock_terminal.stop_terminal = failing_stop_terminal
        mock_gpu.release_gpus_for_session = track_release_gpus

        # Add a child PID to the session state so force-kill has something to kill
        state.cli_process_pid = 12345

        # Patch os.kill to track force-kills and os.getpgid
        import signal
        with patch('orchestration.session_manager.os.kill') as mock_kill, \
             patch('orchestration.session_manager.os.getpgid', return_value=9999):
            mock_kill.side_effect = lambda pid, sig: call_order.append(f"kill_{pid}")
            await manager.terminate_session("test-session")

        # GPUs MUST be released (eventually)
        assert "release_gpus" in call_order, "GPUs must be released even when stop_terminal fails"

        # release_gpus must come AFTER the terminal stop attempt
        assert call_order.index("release_gpus") > call_order.index("stop_terminal_called"), (
            "GPUs must be released after the terminal stop attempt, not before"
        )

        # CRITICAL: force-kill should have been attempted before GPU release
        # (Bug #8 fix: escalate to force-kill when stop_terminal fails)
        kill_events = [e for e in call_order if e.startswith("kill_")]
        assert len(kill_events) > 0, (
            f"When stop_terminal() fails, must force-kill child processes before releasing GPUs. "
            f"Got call_order: {call_order}. No kill_ events found."
        )
        # All kills must precede GPU release
        gpu_release_idx = call_order.index("release_gpus")
        for kill_event in kill_events:
            kill_idx = call_order.index(kill_event)
            assert kill_idx < gpu_release_idx, (
                f"Force-kill ({kill_event} at idx {kill_idx}) must precede GPU release (idx {gpu_release_idx})"
            )

    @pytest.mark.asyncio
    async def test_gpus_released_even_if_force_kill_also_fails(self, tmp_path):
        """
        Even if both stop_terminal() and force-kill fail, GPUs must eventually be released.
        We don't want to leak GPU allocations.
        """
        manager, mock_gpu, mock_terminal = self._make_session_manager(tmp_path)
        state = self._make_active_session(manager)

        released = []

        async def failing_stop_terminal(session_id):
            raise Exception("ttyd failed to stop")

        def track_release_gpus(session_id):
            released.append(True)

        mock_terminal.stop_terminal = failing_stop_terminal
        mock_gpu.release_gpus_for_session = track_release_gpus

        # Force kill also fails
        with patch('orchestration.session_manager.os.kill', side_effect=OSError("kill failed")):
            with patch('orchestration.session_manager.os.getpgid', side_effect=OSError("no process")):
                await manager.terminate_session("test-session")

        assert len(released) == 1, "GPUs must be released even when both terminal stop and force-kill fail"

    @pytest.mark.asyncio
    async def test_pause_session_gpus_released_after_terminal_confirmed_dead(self, tmp_path):
        """
        Same fix applies to pause_session: GPUs released only after terminal confirmed stopped.
        """
        manager, mock_gpu, mock_terminal = self._make_session_manager(tmp_path)
        state = self._make_active_session(manager)

        call_order = []

        async def track_stop_terminal(session_id):
            call_order.append("stop_terminal")

        def track_release_gpus(session_id):
            call_order.append("release_gpus")

        mock_terminal.stop_terminal = track_stop_terminal
        mock_gpu.release_gpus_for_session = track_release_gpus

        await manager.pause_session("test-session")

        assert "stop_terminal" in call_order
        assert "release_gpus" in call_order
        assert call_order.index("stop_terminal") < call_order.index("release_gpus"), (
            "In pause_session: stop_terminal should come before release_gpus"
        )

    @pytest.mark.asyncio
    async def test_pause_session_gpus_released_after_force_kill_when_terminal_stop_fails(self, tmp_path):
        """
        In pause_session, when stop_terminal() fails:
        - should log as error (not just warning)
        - force-kill child processes
        - THEN release GPUs
        """
        manager, mock_gpu, mock_terminal = self._make_session_manager(tmp_path)
        state = self._make_active_session(manager)

        call_order = []

        async def failing_stop_terminal(session_id):
            call_order.append("stop_terminal_called")
            raise Exception("ttyd failed to stop")

        def track_release_gpus(session_id):
            call_order.append("release_gpus")

        mock_terminal.stop_terminal = failing_stop_terminal
        mock_gpu.release_gpus_for_session = track_release_gpus

        # Add a child PID to the session state so force-kill has something to kill
        state.cli_process_pid = 12345

        with patch('orchestration.session_manager.os.kill') as mock_kill, \
             patch('orchestration.session_manager.os.getpgid', return_value=9999):
            mock_kill.side_effect = lambda pid, sig: call_order.append(f"kill_{pid}")
            await manager.pause_session("test-session")

        assert "release_gpus" in call_order, "GPUs must be released even when stop_terminal fails in pause"
        assert call_order.index("release_gpus") > call_order.index("stop_terminal_called"), (
            "GPUs must be released AFTER terminal stop attempt in pause_session"
        )

        # CRITICAL: force-kill should have been attempted
        kill_events = [e for e in call_order if e.startswith("kill_")]
        assert len(kill_events) > 0, (
            f"When stop_terminal() fails in pause_session, must force-kill child processes. "
            f"Got call_order: {call_order}"
        )
        # All kills must precede GPU release
        gpu_release_idx = call_order.index("release_gpus")
        for kill_event in kill_events:
            assert call_order.index(kill_event) < gpu_release_idx, (
                f"Force-kill must precede GPU release in pause_session"
            )


@pytest.mark.unit
class TestResumeForcesPausedOnS3Restore:
    """
    Bug B: Resume early-return when S3 state says "active".

    When a session is restored from S3, it may have status=ACTIVE (that was its
    state when paused). But it can't be genuinely active on this host since it was
    just restored. The resume code at line 770 checks `if status == ACTIVE: return`
    which causes an early return without starting a terminal.

    Fix: After S3 restore, force status to PAUSED so the resume logic proceeds.
    """

    def _make_session_manager(self, tmp_path):
        """Create a SessionManager with fully mocked dependencies for resume testing."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        import time as _time

        reset_worktree_manager()
        sessions_dir = str(tmp_path / "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        mock_worktree = MagicMock()
        mock_worktree.cleanup_session = MagicMock()
        mock_worktree.get_worktree_path = MagicMock(return_value=tmp_path / "wt")
        mock_worktree.get_logs_dir = MagicMock(return_value=tmp_path / "logs")
        mock_worktree.get_session_dir = MagicMock(return_value=tmp_path / "sd")
        mock_worktree.initialize_vllm_environment = AsyncMock(return_value={"timings": {}})

        mock_gpu = MagicMock()
        mock_gpu.get_gpu_count.return_value = 4
        mock_gpu.get_available_gpu_count.return_value = 4
        mock_gpu.release_gpus_for_session = MagicMock()
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(return_value=[0])

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = True
        mock_terminal.start_terminal_with_command = AsyncMock(return_value=8001)
        mock_terminal.stop_terminal = AsyncMock()

        mock_cli = MagicMock()
        mock_cli.stop_cli_tool = MagicMock()
        mock_cli.get_cli_command = MagicMock(return_value=["/usr/bin/env", "claude"])

        mock_inactivity = MagicMock()
        mock_inactivity.unregister_session = MagicMock()
        mock_inactivity.register_session = MagicMock()
        mock_inactivity.get_session_activity = MagicMock(return_value=None)

        mock_storage = MagicMock()
        mock_storage.enabled = True
        mock_storage.get_s3_last_modified = AsyncMock(return_value=None)

        manager = SessionManager(
            sessions_dir=sessions_dir,
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        return manager, mock_terminal, mock_storage, mock_gpu

    @pytest.mark.asyncio
    async def test_resume_forces_paused_when_restored_from_s3(self, tmp_path):
        """
        Sessions restored from S3 with status=ACTIVE must be forced to PAUSED
        before resume processing, to avoid the early-return at line 770.
        """
        import time as _time
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        manager, mock_terminal, mock_storage, mock_gpu = self._make_session_manager(tmp_path)

        # Create worktree directory so resume doesn't fail on missing path
        worktree_path = tmp_path / "sessions" / "s3-session" / "worktree"
        worktree_path.mkdir(parents=True)
        session_dir = worktree_path.parent

        # Simulate S3 returning a session with status=ACTIVE
        s3_state = SessionState(
            session_id="s3-session",
            status=SessionStatus.ACTIVE,  # This is the bug trigger
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=1,
        )
        mock_storage.restore_session_from_s3 = AsyncMock(return_value=s3_state)

        # Session is NOT in local memory (forces S3 restore)
        assert "s3-session" not in manager._sessions

        response = await manager.resume_session("s3-session")

        # The terminal should have been started (not early-returned)
        assert mock_terminal.start_terminal_with_command.called, (
            "Terminal must be started when resuming from S3, not early-returned. "
            "The status=ACTIVE from S3 should be forced to PAUSED before resume logic."
        )

        # Session should now be ACTIVE (set by resume logic after starting terminal)
        state = manager._sessions.get("s3-session")
        assert state is not None
        assert state.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_resume_local_active_session_returns_early(self, tmp_path):
        """Sessions that are genuinely active locally should still return early."""
        import time as _time
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        manager, mock_terminal, mock_storage, mock_gpu = self._make_session_manager(tmp_path)

        # Create a genuinely active local session
        worktree_path = tmp_path / "sessions" / "local-active" / "worktree"
        worktree_path.mkdir(parents=True)

        local_state = SessionState(
            session_id="local-active",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            terminal_port=8001,
            requested_gpu_count=0,
        )
        manager._sessions["local-active"] = local_state

        response = await manager.resume_session("local-active")

        # Should NOT start a new terminal (it's already active)
        assert not mock_terminal.start_terminal_with_command.called, (
            "Terminal should NOT be restarted for genuinely active local session"
        )

        # Response should indicate session is already active
        assert "already active" in response.message.lower() or response.status == "active"



@pytest.mark.unit
class TestLocalResumeMissingSession:
    """Local-only resume errors do not proxy to peer servers."""

    @pytest.mark.asyncio
    async def test_resume_returns_400_when_not_found_locally(self):
        """When a session is not local or restorable from S3, return 400."""
        import app as app_module
        from httpx import ASGITransport, AsyncClient

        orig_session_manager = app_module.session_manager
        orig_api_key = app_module.AMMO_API_KEY

        try:
            app_module.AMMO_API_KEY = ""
            mock_sm = MagicMock()
            mock_sm.resume_session = AsyncMock(
                side_effect=app_module.SessionMgrError("Session notexist not found")
            )
            app_module.session_manager = mock_sm

            async with AsyncClient(
                transport=ASGITransport(app=app_module.app),
                base_url="http://test"
            ) as test_client:
                resp = await test_client.post("/sessions/notexist/resume")

            assert resp.status_code == 400

        finally:
            app_module.AMMO_API_KEY = orig_api_key
            app_module.session_manager = orig_session_manager


@pytest.mark.unit
class TestLocalDelete:
    """Local-only DELETE behavior."""

    @pytest.mark.asyncio
    async def test_delete_session_locally_when_found(self):
        """When session exists locally, terminate normally without proxying."""
        import app as app_module
        from httpx import ASGITransport, AsyncClient

        orig_session_manager = app_module.session_manager
        orig_checkpoint_manager = app_module.checkpoint_manager
        orig_api_key = app_module.AMMO_API_KEY

        try:
            app_module.AMMO_API_KEY = ""
            mock_sm = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "session_id": "local123",
                "status": "terminated",
                "message": "Session terminated.",
            }
            mock_sm.terminate_session = AsyncMock(return_value=mock_response)
            app_module.session_manager = mock_sm
            app_module.checkpoint_manager = None

            async with AsyncClient(
                transport=ASGITransport(app=app_module.app),
                base_url="http://test",
            ) as test_client:
                resp = await test_client.delete("/sessions/local123")

            assert resp.status_code == 200, (
                f"Expected 200 for local terminate, got {resp.status_code}: {resp.text}"
            )
            data = resp.json()
            assert data.get("session_id") == "local123"
            assert data.get("status") == "terminated"
            mock_sm.terminate_session.assert_awaited_once()

        finally:
            app_module.AMMO_API_KEY = orig_api_key
            app_module.session_manager = orig_session_manager
            app_module.checkpoint_manager = orig_checkpoint_manager



@pytest.mark.unit
class TestTerminalZombieReaping:
    """
    Bug D: No waitpid -> zombie accumulation.

    stop_terminal sends SIGTERM/SIGKILL but never calls waitpid to reap
    the child process, causing zombie accumulation over time.
    """

    @pytest.mark.asyncio
    async def test_stop_terminal_reaps_child(self):
        """stop_terminal must call os.waitpid to prevent zombie accumulation."""
        from orchestration.terminal_manager import TerminalManager, TerminalProcess

        manager = TerminalManager(base_port=19000, max_ports=5)

        # Add a fake terminal entry
        manager._terminals["test-session"] = TerminalProcess(
            session_id="test-session",
            port=19000,
            pid=99999,
            master_fd=-1,
        )
        manager._used_ports.add(19000)

        with patch("os.kill") as mock_kill, \
             patch("os.waitpid") as mock_waitpid:
            # Make os.kill(pid, 0) raise ProcessLookupError to simulate dead process
            def kill_side_effect(pid, sig):
                if sig == 0:
                    raise ProcessLookupError("No such process")
            mock_kill.side_effect = kill_side_effect
            mock_waitpid.return_value = (99999, 0)

            await manager.stop_terminal("test-session")

            # waitpid should have been called to reap the zombie
            assert mock_waitpid.called, (
                "stop_terminal must call os.waitpid to reap child process and prevent zombies"
            )
            # Should be called with WNOHANG (non-blocking)
            waitpid_args = mock_waitpid.call_args
            assert waitpid_args[0][0] == 99999, "waitpid should be called with the terminal PID"
            assert waitpid_args[0][1] == os.WNOHANG, "waitpid should use WNOHANG flag"

    @pytest.mark.asyncio
    async def test_stop_terminal_handles_waitpid_failure(self):
        """stop_terminal must not fail if waitpid raises (e.g., child already reaped)."""
        from orchestration.terminal_manager import TerminalManager, TerminalProcess

        manager = TerminalManager(base_port=19000, max_ports=5)

        manager._terminals["test-session"] = TerminalProcess(
            session_id="test-session",
            port=19000,
            pid=99998,
            master_fd=-1,
        )
        manager._used_ports.add(19000)

        with patch("os.kill") as mock_kill, \
             patch("os.waitpid", side_effect=ChildProcessError("No child")):
            mock_kill.side_effect = lambda pid, sig: None if sig != 0 else (_ for _ in ()).throw(ProcessLookupError)

            result = await manager.stop_terminal("test-session")

            # Should still succeed even if waitpid fails
            assert result is True, "stop_terminal should succeed even if waitpid raises"
            # Port should be released
            assert 19000 not in manager._used_ports


@pytest.mark.unit
class TestTerminalStderrCapture:
    """
    Bug E: ttyd stderr suppressed.

    stderr=DEVNULL loses crash diagnostics. ttyd should write to PIPE so
    we can log stderr on failures.
    """

    @pytest.mark.asyncio
    async def test_start_terminal_does_not_suppress_stderr(self):
        """ttyd stderr must NOT be DEVNULL - it should be PIPE for diagnostics."""
        import asyncio as _asyncio
        from orchestration.terminal_manager import TerminalManager

        manager = TerminalManager(base_port=19000, max_ports=5)
        manager._ttyd_available = True
        manager._tmux_available = True

        captured_kwargs = {}

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured_kwargs.update(kwargs)
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.returncode = None
            return mock_proc

        # Patch on the module where it's looked up (orchestration.terminal_manager uses asyncio directly)
        with patch("orchestration.terminal_manager.asyncio.create_subprocess_exec",
                    side_effect=fake_create_subprocess_exec), \
             patch.object(manager, "_is_port_in_use", return_value=True):
            port = await manager.start_terminal_with_command(
                session_id="test-session",
                command=["/bin/bash"],
                working_dir=Path("/tmp"),
                env={"HOME": "/root"},
                tmux_session_name="ammo-test",
            )

        assert "stderr" in captured_kwargs, (
            "start_terminal_with_command must pass stderr kwarg to subprocess"
        )
        assert captured_kwargs["stderr"] != _asyncio.subprocess.DEVNULL, (
            "stderr must NOT be DEVNULL - ttyd diagnostics are lost. Use PIPE instead."
        )
        assert captured_kwargs["stderr"] == _asyncio.subprocess.PIPE, (
            "stderr should be PIPE to capture ttyd crash diagnostics"
        )


@pytest.mark.unit
class TestContinueFallback:
    """
    Bug F: --continue fallback.

    If claude-config/ has no conversation history, `claude --continue` exits
    immediately because there's nothing to continue. The resume logic should
    check for conversation data before using --continue.
    """

    def _make_session_manager(self, tmp_path):
        """Create a SessionManager with mocked dependencies for resume testing."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        import time as _time

        reset_worktree_manager()
        sessions_dir = str(tmp_path / "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        mock_worktree = MagicMock()
        mock_worktree.initialize_vllm_environment = AsyncMock(return_value={"timings": {}})

        mock_gpu = MagicMock()
        mock_gpu.get_gpu_count.return_value = 4
        mock_gpu.get_available_gpu_count.return_value = 4
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(return_value=[0])

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = True
        mock_terminal.start_terminal_with_command = AsyncMock(return_value=8001)

        mock_cli = MagicMock()
        mock_cli.get_cli_command = MagicMock(return_value=["/usr/bin/env", "claude"])

        mock_inactivity = MagicMock()
        mock_inactivity.register_session = MagicMock()

        mock_storage = MagicMock()
        mock_storage.enabled = False

        manager = SessionManager(
            sessions_dir=sessions_dir,
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        return manager, mock_cli

    @pytest.mark.asyncio
    async def test_resume_uses_continue_when_history_exists(self, tmp_path):
        """When claude-config has conversation history, is_resume=True."""
        import time as _time
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        manager, mock_cli = self._make_session_manager(tmp_path)

        # Create session directory structure with history
        worktree_path = tmp_path / "sessions" / "has-history" / "worktree"
        worktree_path.mkdir(parents=True)
        session_dir = worktree_path.parent

        claude_config_dir = session_dir / "claude-config"
        claude_config_dir.mkdir(parents=True)
        # Create a projects directory with conversation data (Claude Code format)
        projects_dir = claude_config_dir / "projects"
        project_dir = projects_dir / "-tmp-has-history-worktree"
        project_dir.mkdir(parents=True)
        (project_dir / "conversation.jsonl").write_text("{}")

        state = SessionState(
            session_id="has-history",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=1,
        )
        manager._sessions["has-history"] = state

        await manager.resume_session("has-history")

        # get_cli_command should have been called with is_resume=True
        mock_cli.get_cli_command.assert_called_once()
        call_kwargs = mock_cli.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is True or call_kwargs[1].get("is_resume") is True, (
            "When claude-config has conversation data, is_resume should be True"
        )

    @pytest.mark.asyncio
    async def test_resume_skips_continue_when_no_history(self, tmp_path):
        """When claude-config is empty (no projects dir), is_resume=False."""
        import time as _time
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        manager, mock_cli = self._make_session_manager(tmp_path)

        # Create session directory structure WITHOUT history
        worktree_path = tmp_path / "sessions" / "no-history" / "worktree"
        worktree_path.mkdir(parents=True)
        session_dir = worktree_path.parent

        claude_config_dir = session_dir / "claude-config"
        claude_config_dir.mkdir(parents=True)
        # Empty claude-config - no projects directory

        state = SessionState(
            session_id="no-history",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=1,
        )
        manager._sessions["no-history"] = state

        await manager.resume_session("no-history")

        # get_cli_command should have been called with is_resume=False
        mock_cli.get_cli_command.assert_called_once()
        call_kwargs = mock_cli.get_cli_command.call_args
        is_resume_value = call_kwargs.kwargs.get("is_resume", call_kwargs[1].get("is_resume"))
        assert is_resume_value is False, (
            f"When claude-config has no conversation data, is_resume should be False, got {is_resume_value}"
        )

    @pytest.mark.asyncio
    async def test_codex_resume_without_history_replays_stored_initial_prompt(self, tmp_path, monkeypatch):
        """Codex resume without history should preserve the original prompt."""
        import time as _time
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        manager, mock_cli = self._make_session_manager(tmp_path)

        worktree_path = tmp_path / "sessions" / "codex-no-history" / "worktree"
        worktree_path.mkdir(parents=True)
        session_dir = worktree_path.parent

        state = SessionState(
            session_id="codex-no-history",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CODEX,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
            initial_prompt="original codex prompt",
        )
        manager._sessions["codex-no-history"] = state

        await manager.resume_session("codex-no-history")

        mock_cli.get_cli_command.assert_called_once()
        call_kwargs = mock_cli.get_cli_command.call_args.kwargs
        assert call_kwargs["is_resume"] is False
        assert call_kwargs["initial_prompt"] == "original codex prompt"


@pytest.mark.unit
class TestCrossPodProjectDirRename:
    """
    Bug: Cross-pod session resume loses Claude conversation history.

    Claude Code stores conversation data in:
        claude-config/projects/{worktree-path-encoded}/
    where path encoding replaces '/' with '-'.

    After cross-host S3 restore, the worktree path can change
    (e.g., /data/sessions/ -> /local/sessions/), causing the encoded
    directory name to not match. Claude starts fresh instead of resuming.

    Fix: After S3 restore, detect and rename the project directory
    to match the new worktree path encoding.
    """

    def _make_session_manager(self, tmp_path):
        """Create a SessionManager with mocked dependencies for resume testing."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        import time as _time

        reset_worktree_manager()
        sessions_dir = str(tmp_path / "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        mock_worktree = MagicMock()
        mock_worktree.initialize_vllm_environment = AsyncMock(return_value={"timings": {}})

        mock_gpu = MagicMock()
        mock_gpu.get_gpu_count.return_value = 4
        mock_gpu.get_available_gpu_count.return_value = 4
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(return_value=[0])

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = True
        mock_terminal.start_terminal_with_command = AsyncMock(return_value=8001)

        mock_cli = MagicMock()
        mock_cli.get_cli_command = MagicMock(return_value=["/usr/bin/env", "claude"])

        mock_inactivity = MagicMock()
        mock_inactivity.register_session = MagicMock()

        mock_storage = MagicMock()
        mock_storage.enabled = True
        mock_storage.get_s3_last_modified = AsyncMock(return_value=None)

        manager = SessionManager(
            sessions_dir=sessions_dir,
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        return manager, mock_cli, mock_storage, mock_terminal

    def _setup_cross_pod_dirs(self, tmp_path, old_worktree_path, new_worktree_path):
        """
        Set up directory structure simulating a cross-host S3 restore.

        Creates:
        - session_dir with claude-config/projects/{old-path-encoded}/
        - worktree at new_worktree_path

        Returns:
            (session_dir, claude_config_dir, old_encoded_name, new_encoded_name)
        """
        session_dir = tmp_path / "sessions" / "cross-pod-session"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create worktree at the NEW path
        new_worktree = Path(new_worktree_path)
        new_worktree.mkdir(parents=True, exist_ok=True)

        # Claude config with project dir encoded from OLD path
        claude_config_dir = session_dir / "claude-config"
        projects_dir = claude_config_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)

        old_encoded = old_worktree_path.replace("/", "-")
        new_encoded = new_worktree_path.replace("/", "-")

        old_project_dir = projects_dir / old_encoded
        old_project_dir.mkdir(parents=True, exist_ok=True)
        # Add a conversation file to prove data is preserved
        (old_project_dir / "conversation.json").write_text('{"messages": ["hello"]}')

        return session_dir, claude_config_dir, old_encoded, new_encoded

    def _make_state(self, session_dir, worktree_path, session_id="cross-pod-session"):
        """Create a SessionState for testing."""
        import time as _time
        return SessionState(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=1,
        )

    def test_renames_project_dir_on_cross_pod_resume(self, tmp_path):
        """S3 restore with path mismatch -> dir renamed, file intact."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        old_path = "/data/sessions/cross-pod-session/worktree"
        new_path = str(tmp_path / "local" / "sessions" / "cross-pod-session" / "worktree")

        session_dir, claude_config_dir, old_encoded, new_encoded = \
            self._setup_cross_pod_dirs(tmp_path, old_path, new_path)

        state = self._make_state(session_dir, new_path)

        manager._fix_claude_project_dir_after_s3_restore(state)

        projects_dir = claude_config_dir / "projects"
        # Old dir should be gone
        assert not (projects_dir / old_encoded).exists(), \
            f"Old project dir should be renamed: {old_encoded}"
        # New dir should exist
        assert (projects_dir / new_encoded).exists(), \
            f"New project dir should exist: {new_encoded}"
        # Data should be preserved
        conv_file = projects_dir / new_encoded / "conversation.json"
        assert conv_file.exists(), "Conversation data must survive rename"
        assert "hello" in conv_file.read_text()

    def test_updates_claude_json_on_cross_pod_resume(self, tmp_path):
        """.claude.json projects key updated from old path to new path."""
        import json
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        old_path = "/data/sessions/cross-pod-session/worktree"
        new_path = str(tmp_path / "local" / "sessions" / "cross-pod-session" / "worktree")

        session_dir, claude_config_dir, old_encoded, new_encoded = \
            self._setup_cross_pod_dirs(tmp_path, old_path, new_path)

        # Create .claude.json with old path key
        claude_json = claude_config_dir / ".claude.json"
        claude_json.write_text(json.dumps({
            "theme": "dark",
            "hasCompletedOnboarding": True,
            "projects": {
                old_path: {
                    "hasTrustDialogAccepted": True,
                    "allowedTools": [],
                }
            }
        }, indent=2))

        state = self._make_state(session_dir, new_path)

        manager._fix_claude_project_dir_after_s3_restore(state)

        # Verify .claude.json was updated
        updated = json.loads(claude_json.read_text())
        assert old_path not in updated["projects"], \
            "Old path key should be removed from .claude.json projects"
        assert new_path in updated["projects"], \
            "New path key should be added to .claude.json projects"
        assert updated["projects"][new_path]["hasTrustDialogAccepted"] is True

    def test_skips_rename_when_project_dir_already_matches(self, tmp_path):
        """No-op when paths match (same-pod resume or paths unchanged)."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        worktree_path = str(tmp_path / "sessions" / "cross-pod-session" / "worktree")
        session_dir = tmp_path / "sessions" / "cross-pod-session"
        session_dir.mkdir(parents=True, exist_ok=True)
        Path(worktree_path).mkdir(parents=True, exist_ok=True)

        # Create projects dir with MATCHING encoding
        claude_config_dir = session_dir / "claude-config"
        projects_dir = claude_config_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        matching_encoded = worktree_path.replace("/", "-")
        matching_dir = projects_dir / matching_encoded
        matching_dir.mkdir(parents=True, exist_ok=True)
        (matching_dir / "data.json").write_text('{"ok": true}')

        state = self._make_state(session_dir, worktree_path)

        manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should still exist, unchanged
        assert matching_dir.exists()
        assert (matching_dir / "data.json").read_text() == '{"ok": true}'

    def test_skips_rename_when_no_project_dirs(self, tmp_path):
        """Empty projects dir -> no error."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        worktree_path = str(tmp_path / "sessions" / "cross-pod-session" / "worktree")
        session_dir = tmp_path / "sessions" / "cross-pod-session"
        session_dir.mkdir(parents=True, exist_ok=True)
        Path(worktree_path).mkdir(parents=True, exist_ok=True)

        # Create empty projects dir
        claude_config_dir = session_dir / "claude-config"
        projects_dir = claude_config_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)

        state = self._make_state(session_dir, worktree_path)

        # Should not raise
        manager._fix_claude_project_dir_after_s3_restore(state)

    def test_skips_rename_when_multiple_project_dirs(self, tmp_path):
        """>1 dirs -> warning logged, skip rename."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        worktree_path = str(tmp_path / "sessions" / "cross-pod-session" / "worktree")
        session_dir = tmp_path / "sessions" / "cross-pod-session"
        session_dir.mkdir(parents=True, exist_ok=True)
        Path(worktree_path).mkdir(parents=True, exist_ok=True)

        # Create multiple project dirs
        claude_config_dir = session_dir / "claude-config"
        projects_dir = claude_config_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        (projects_dir / "dir-one").mkdir()
        (projects_dir / "dir-two").mkdir()

        state = self._make_state(session_dir, worktree_path)

        # Should not raise, should skip
        manager._fix_claude_project_dir_after_s3_restore(state)

        # Both dirs should still exist (unchanged)
        assert (projects_dir / "dir-one").exists()
        assert (projects_dir / "dir-two").exists()

    @pytest.mark.asyncio
    async def test_skips_rename_when_not_restored_from_s3(self, tmp_path):
        """Same-pod resume -> helper is NOT called."""
        import time as _time

        manager, mock_cli, mock_storage, mock_terminal = self._make_session_manager(tmp_path)
        mock_storage.enabled = False  # No S3

        old_path = "/data/sessions/skip-s3-session/worktree"
        new_path = str(tmp_path / "sessions" / "skip-s3-session" / "worktree")

        session_dir = tmp_path / "sessions" / "skip-s3-session"
        session_dir.mkdir(parents=True, exist_ok=True)
        Path(new_path).mkdir(parents=True, exist_ok=True)

        # Set up a stale project dir that would be renamed IF the helper ran
        claude_config_dir = session_dir / "claude-config"
        projects_dir = claude_config_dir / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)
        old_encoded = old_path.replace("/", "-")
        stale_dir = projects_dir / old_encoded
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "data.json").write_text("{}")

        state = SessionState(
            session_id="skip-s3-session",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=_time.time(),
            last_accessed=_time.time(),
            worktree_path=new_path,
            session_dir=str(session_dir),
            requested_gpu_count=1,
        )
        manager._sessions["skip-s3-session"] = state

        with patch.object(manager, "_fix_claude_project_dir_after_s3_restore") as mock_fix:
            await manager.resume_session("skip-s3-session")
            mock_fix.assert_not_called()

    def test_handles_missing_claude_json_gracefully(self, tmp_path):
        """Dir renamed even without .claude.json present."""
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        old_path = "/data/sessions/cross-pod-session/worktree"
        new_path = str(tmp_path / "local" / "sessions" / "cross-pod-session" / "worktree")

        session_dir, claude_config_dir, old_encoded, new_encoded = \
            self._setup_cross_pod_dirs(tmp_path, old_path, new_path)

        # Ensure no .claude.json exists
        claude_json = claude_config_dir / ".claude.json"
        if claude_json.exists():
            claude_json.unlink()

        state = self._make_state(session_dir, new_path)

        # Should not raise
        manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should be renamed
        projects_dir = claude_config_dir / "projects"
        assert not (projects_dir / old_encoded).exists()
        assert (projects_dir / new_encoded).exists()

    def test_handles_claude_json_without_projects_key(self, tmp_path):
        """Dir renamed, json unchanged when no projects key."""
        import json
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager
        reset_worktree_manager()

        manager, _, _, _ = self._make_session_manager(tmp_path)

        old_path = "/data/sessions/cross-pod-session/worktree"
        new_path = str(tmp_path / "local" / "sessions" / "cross-pod-session" / "worktree")

        session_dir, claude_config_dir, old_encoded, new_encoded = \
            self._setup_cross_pod_dirs(tmp_path, old_path, new_path)

        # Create .claude.json WITHOUT projects key
        claude_json = claude_config_dir / ".claude.json"
        original_json = {"theme": "dark", "hasCompletedOnboarding": True}
        claude_json.write_text(json.dumps(original_json, indent=2))

        state = self._make_state(session_dir, new_path)

        manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should be renamed
        projects_dir = claude_config_dir / "projects"
        assert not (projects_dir / old_encoded).exists()
        assert (projects_dir / new_encoded).exists()

        # .claude.json should be unchanged (no projects key to update)
        updated = json.loads(claude_json.read_text())
        assert updated == original_json


@pytest.mark.unit
class TestCreateSessionTpDpStorage:
    """Verify SessionManager.create_session persists tp_size/dp_size from request."""

    def _make_manager(self, tmp_path, available_gpus=8):
        from orchestration.session_manager import SessionManager
        from orchestration.worktree_manager import reset_worktree_manager

        reset_worktree_manager()
        sessions_dir = str(tmp_path / "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        mock_worktree = MagicMock()
        mock_worktree.create_worktree = MagicMock(return_value=tmp_path / "wt")
        mock_worktree.get_session_dir = MagicMock(return_value=tmp_path / "sd")
        mock_worktree.get_logs_dir = MagicMock(return_value=tmp_path / "logs")
        mock_worktree.initialize_vllm_environment = AsyncMock(
            return_value={"timings": {}}
        )

        mock_gpu = MagicMock()
        mock_gpu.get_gpu_count.return_value = available_gpus
        mock_gpu.get_available_gpu_count.return_value = available_gpus
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(
            return_value=list(range(available_gpus))
        )

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = False  # skip terminal path

        mock_cli = MagicMock()
        mock_cli.setup_workspace = MagicMock()

        mock_inactivity = MagicMock()
        mock_inactivity.register_session = MagicMock()

        mock_storage = MagicMock()
        mock_storage.enabled = False

        manager = SessionManager(
            sessions_dir=sessions_dir,
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        return manager

    @pytest.mark.asyncio
    async def test_create_session_stores_tp_dp_from_request(self, tmp_path):
        """Explicit tp_size=4, dp_size=2 from request lands on SessionState."""
        manager = self._make_manager(tmp_path, available_gpus=8)

        request = CreateSessionRequest(gpu_count=8, tp_size=4, dp_size=2)
        await manager.create_session(request, owner_id="client-a")

        # Exactly one session should have been created
        assert len(manager._sessions) == 1
        state = next(iter(manager._sessions.values()))
        assert state.tp_size == 4
        assert state.dp_size == 2
        assert state.requested_gpu_count == 8

    @pytest.mark.asyncio
    async def test_create_session_legacy_tp_fallback(self, tmp_path):
        """Old clients (no tp_size/dp_size) → tp_size == gpu_count, dp_size == 1."""
        manager = self._make_manager(tmp_path, available_gpus=8)

        # Simulates an old client that only sends gpu_count
        request = CreateSessionRequest(gpu_count=8)
        await manager.create_session(request, owner_id="client-b")

        assert len(manager._sessions) == 1
        state = next(iter(manager._sessions.values()))
        assert state.tp_size == 8, (
            "Legacy requests (no explicit tp_size) must fall back to gpu_count"
        )
        assert state.dp_size == 1
