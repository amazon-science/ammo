# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Integration tests for AMMO session support."""
import pytest
import requests
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestHealthGpuInfo:
    """Tests that GPU info is exposed via /health after the static-model-selector
    removal. Replaces the deleted TestSupportedModelsEndpoint class — model
    selection now runs entirely through /api/hf-models (search) +
    /api/hf-model-config (auto-detection)."""

    def test_health_returns_gpu_block(self, server_url):
        """/health exposes a structured `gpu` block with type + allowed_dtypes."""
        response = requests.get(f"{server_url}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "gpu" in data, f"/health must include 'gpu' block. Got keys: {list(data.keys())}"
        gpu = data["gpu"]
        assert "type" in gpu, f"/health gpu block must include 'type'. Got keys: {list(gpu.keys())}"
        assert "allowed_dtypes" in gpu, (
            f"/health gpu block must include 'allowed_dtypes'. Got keys: {list(gpu.keys())}"
        )
        assert isinstance(gpu["allowed_dtypes"], list), (
            f"gpu.allowed_dtypes must be a list, got {type(gpu['allowed_dtypes']).__name__}"
        )

    def test_health_gpu_manager_preserved(self, server_url):
        """/health must include the `gpu_manager` block used by local GPU display."""
        response = requests.get(f"{server_url}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "gpu_manager" in data, (
            f"/health must preserve the 'gpu_manager' block for local GPU display. "
            f"Got keys: {list(data.keys())}"
        )


@pytest.mark.integration
class TestAmmoSessionSkillFiles:
    """Tests that AMMO skill files are correctly deployed to sessions."""

    def _resolve_path(self, relative_path):
        """Resolve path relative to repo root, trying multiple locations."""
        candidates = [
            Path(__file__).parent.parent.parent / relative_path,
            Path("/app") / relative_path,
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]  # Return first candidate for assertion error messages

    def test_managed_settings_has_team_tools(self):
        """Verify managed-settings.json includes team-related tools."""
        settings_path = self._resolve_path("ai_cli_session/managed-settings.json")

        with open(settings_path) as f:
            settings = json.load(f)

        allow_list = settings["permissions"]["allow"]
        required_tools = ["TeamCreate", "TeamDelete", "SendMessage", "TaskCreate", "TaskUpdate", "TaskList", "EnterWorktree"]
        for tool in required_tools:
            assert tool in allow_list, f"Missing tool in managed-settings: {tool}"

    def test_agent_files_exist(self):
        """Verify AMMO agent definitions exist in template."""
        agents_dir = self._resolve_path("ai_cli_session/.claude/agents")

        expected_agents = ["ammo-researcher.md", "ammo-champion.md", "ammo-impl-champion.md", "ammo-impl-validator.md", "ammo-delegate.md"]
        for agent_file in expected_agents:
            assert (agents_dir / agent_file).exists(), f"Missing agent file: {agent_file}"

    def test_hook_files_exist_and_executable(self):
        """Verify AMMO hook scripts exist and are executable."""
        hooks_dir = self._resolve_path("ai_cli_session/.claude/hooks")

        expected_hooks = [
            "ammo-gate-guard.sh", "ammo-precompact.sh", "ammo-postcompact.sh",
            "worktree-create-with-build.sh", "worktree-remove-cleanup.sh"
        ]
        for hook_file in expected_hooks:
            path = hooks_dir / hook_file
            assert path.exists(), f"Missing hook: {hook_file}"
            assert os.access(path, os.X_OK), f"Hook not executable: {hook_file}"

    def test_skill_files_are_campaign_workflow(self):
        """Verify AMMO skill has campaign workflow with debate protocol."""
        skill_dir = self._resolve_path("ai_cli_session/.claude/skills/ammo")

        skill_md = skill_dir / "SKILL.md"
        assert skill_md.exists()
        content = skill_md.read_text()

        # Campaign workflow indicators
        assert "Campaign Workflow" in content, "Should have Campaign Workflow"
        assert "debate" in content.lower(), "Should mention debate protocol"

        # Version B indicators should be absent
        assert "route-selection-decision-tree" not in content, "Should not reference MoE route selection"

    def test_settings_local_has_hooks(self):
        """Verify settings.local.json has hook configurations."""
        settings_path = self._resolve_path("ai_cli_session/.claude/settings.local.json")

        with open(settings_path) as f:
            settings = json.load(f)

        assert "hooks" in settings
        assert "WorktreeCreate" in settings["hooks"]
        assert "WorktreeRemove" in settings["hooks"]

    def test_settings_local_has_no_tmux_deny_rule(self):
        """settings.local.json must NOT deny Bash(tmux:*).

        Commit 30a6b3e removed the rule so agents can close stuck panes.
        Session isolation is owned by terminal_manager's per-session tmux
        server (-S socket) plus its hardened config (-f), covered by
        tests/unit/test_tmux_hardening.py.
        """
        settings_path = self._resolve_path("ai_cli_session/.claude/settings.local.json")

        with open(settings_path) as f:
            settings = json.load(f)

        deny_list = settings.get("permissions", {}).get("deny", [])
        assert "Bash(tmux:*)" not in deny_list
