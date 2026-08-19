# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for CLIToolManager workspace setup — statusline config (Tests 1-2).

Design note: statusLine lives in settings.local.json (user-preference config, like
hooks/model/effortLevel), NOT in settings.json (sandboxed permissions file).
_create_sandboxed_settings() strips statusLine alongside hooks/model/effortLevel/env.
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.cli_tool_manager import CLIToolManager


@pytest.fixture
def manager():
    return CLIToolManager()


@pytest.fixture
def worktree(tmp_path):
    """Isolated temp dir to use as a fake git worktree."""
    wt = tmp_path / "worktree"
    wt.mkdir()
    return wt


@pytest.mark.unit
class TestStatuslineWorkspaceSetup:
    """Tests 1-2: statusline config written by setup_claude_workspace()."""

    def test_settings_local_json_contains_statusline(self, manager, worktree):
        """Test 1: settings.local.json has statusLine after setup_claude_workspace().

        statusLine belongs in settings.local.json (user-pref layer — model, hooks,
        statusLine), NOT settings.json (sandboxed permissions file). Claude Code
        reads both files additively; settings.local.json takes precedence.
        """
        manager.setup_claude_workspace(
            worktree_path=worktree,
            session_id="test-session-00000000",
            gpu_ids=[],
        )

        settings_local_path = worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists(), (
            "settings.local.json must be present in worktree after setup"
        )

        with open(settings_local_path) as f:
            settings = json.load(f)

        assert "statusLine" in settings, (
            "settings.local.json must contain 'statusLine' key"
        )
        sl = settings["statusLine"]
        assert sl.get("type") == "command", "statusLine.type must be 'command'"
        assert "ammo-statusline.sh" in sl.get("command", ""), (
            "statusLine.command must reference 'ammo-statusline.sh'"
        )

    def test_settings_local_json_statusline_uses_dynamic_project_dir(self, manager, worktree):
        """Verifier edge-case: statusLine.command must reference $CLAUDE_PROJECT_DIR.

        The statusline script must be invoked via $CLAUDE_PROJECT_DIR so it resolves
        correctly on any host (cross-host S3 restore, different worktree paths).
        A hardcoded absolute path would break when the worktree is on a different pod.
        """
        manager.setup_claude_workspace(
            worktree_path=worktree,
            session_id="test-session-00000000",
            gpu_ids=[],
        )

        settings_local_path = worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        command = settings.get("statusLine", {}).get("command", "")
        assert "$CLAUDE_PROJECT_DIR" in command, (
            "statusLine.command must use $CLAUDE_PROJECT_DIR (not a hardcoded path) "
            f"so it resolves correctly on any host. Got: {command!r}"
        )

    def test_settings_json_contains_statusline(self, manager, worktree):
        """Test 1b: settings.json (sandboxed) MUST have statusLine.

        statusLine is intentionally kept in settings.json (not stripped) so
        Claude Code can display session status. Unlike hooks/model/effortLevel/env,
        statusLine is a display-only config that needs to flow through.
        """
        manager.setup_claude_workspace(
            worktree_path=worktree,
            session_id="test-session-00000000",
            gpu_ids=[],
        )

        settings_path = worktree / ".claude" / "settings.json"
        assert settings_path.exists(), "settings.json must be created by setup_claude_workspace()"

        with open(settings_path) as f:
            settings = json.load(f)

        assert "statusLine" in settings, (
            "settings.json must contain 'statusLine' so Claude Code can display "
            "session status information"
        )

    def test_ammo_statusline_script_exists_and_is_executable(self, manager, worktree):
        """Test 2: ammo-statusline.sh exists in .claude/hooks/ and is executable (0o755).

        setup_claude_workspace() copies the .claude/ template then chmods every .sh
        in hooks/ to 0o755, so the script just needs to be present in the template.
        """
        manager.setup_claude_workspace(
            worktree_path=worktree,
            session_id="test-session-00000000",
            gpu_ids=[],
        )

        script_path = worktree / ".claude" / "hooks" / "ammo-statusline.sh"
        assert script_path.exists(), (
            "ammo-statusline.sh must exist at worktree/.claude/hooks/ammo-statusline.sh "
            "after setup_claude_workspace()"
        )

        mode = os.stat(script_path).st_mode
        assert mode & stat.S_IXUSR, "ammo-statusline.sh must be user-executable (0o755)"
        assert mode & stat.S_IXGRP, "ammo-statusline.sh must be group-executable (0o755)"
        assert mode & stat.S_IXOTH, "ammo-statusline.sh must be other-executable (0o755)"
