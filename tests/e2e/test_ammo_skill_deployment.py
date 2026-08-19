# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for AMMO skill deployment into sessions.

Verifies that when a session is created, the AMMO skill files (hooks, agents,
settings, skill docs) are correctly deployed into the session worktree's
.claude/ directory.

These tests require:
- A running server (Docker container) at AMMO_SERVER_URL
- The Docker container named 'ammo-server'
"""

import pytest
import requests
import time
import os
import sys
import json
import uuid
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _auth_headers() -> dict:
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8000"
SESSION_CREATION_TIMEOUT = 300  # seconds (editable_install can take 200s+ under load)
ACTIVE_WAIT_TIMEOUT = 120  # seconds to wait for session to become active
CONTAINER_NAME = "ammo-server"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_headers() -> dict:
    """Return combined client-id + auth headers for session isolation."""
    return {"X-Client-ID": str(uuid.uuid4()), **_auth_headers()}


def _server_is_reachable(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_active(
    server_url: str, session_id: str, headers: dict = None, timeout: float = ACTIVE_WAIT_TIMEOUT
) -> dict:
    """Poll until session is active or timeout. Returns latest session data."""
    headers = headers or {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "active":
                return data
            if data.get("status") in ("failed", "terminated"):
                return data
        time.sleep(2)
    return {}


def _read_container_file(path: str) -> str:
    """Read a file from inside the Docker container."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "cat", path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _file_exists_in_container(path: str) -> bool:
    """Check if a file exists inside the Docker container."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "test", "-f", path],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Session fixture (shared across all tests in this module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="module")
def shared_session(server_url):
    """Create a single session and reuse it across all tests in this module.

    Session creation takes ~30s, so we share one session for all deployment
    verification tests. Yields (session_id, headers, worktree_claude_dir).
    """
    if not _server_is_reachable(server_url):
        pytest.skip("Server not reachable")

    headers = _client_headers()
    session_id = None

    try:
        resp = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "gpu_count": 0,
                "inactivity_timeout_mins": 60,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT,
        )
        if resp.status_code != 200:
            pytest.skip(f"Session creation failed ({resp.status_code}): {resp.text}")

        create_data = resp.json()
        session_id = create_data["session_id"]
        assert session_id

        # Wait for active
        session_data = _wait_for_active(server_url, session_id, headers=headers)
        if session_data.get("status") != "active":
            pytest.skip(
                f"Session did not become active: {session_data.get('status', 'unknown')}"
            )

        # The .claude/ dir inside the worktree
        worktree_claude_dir = f"/data/sessions/{session_id}/worktree/.claude"

        yield session_id, headers, worktree_claude_dir

    finally:
        if session_id:
            requests.delete(
                f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
class TestAmmoSkillDeployment:
    """Verify AMMO skill files are correctly deployed into session worktrees."""

    def test_settings_local_has_enter_worktree_hook(self, shared_session):
        """settings.local.json must have a PreToolUse hook with matcher EnterWorktree."""
        session_id, headers, claude_dir = shared_session

        content = _read_container_file(f"{claude_dir}/settings.local.json")
        assert content is not None, "settings.local.json not found in session worktree"

        settings = json.loads(content)
        hooks = settings.get("hooks", {})
        pre_tool_use = hooks.get("PreToolUse", [])

        matchers = [entry.get("matcher") for entry in pre_tool_use]
        assert "EnterWorktree" in matchers, (
            f"PreToolUse hooks should include EnterWorktree matcher, got matchers: {matchers}"
        )

    def test_settings_local_has_stop_hook(self, shared_session):
        """settings.local.json must have a Stop hook pointing to ammo-stop-guard.sh."""
        session_id, headers, claude_dir = shared_session

        content = _read_container_file(f"{claude_dir}/settings.local.json")
        assert content is not None, "settings.local.json not found in session worktree"

        settings = json.loads(content)
        hooks = settings.get("hooks", {})
        stop_hooks = hooks.get("Stop", [])

        assert len(stop_hooks) > 0, "No Stop hooks found in settings.local.json"

        # Check that at least one Stop hook references ammo-stop-guard.sh
        all_commands = []
        for entry in stop_hooks:
            for hook in entry.get("hooks", []):
                cmd = hook.get("command", "")
                all_commands.append(cmd)

        assert any("ammo-stop-guard.sh" in cmd for cmd in all_commands), (
            f"No Stop hook references ammo-stop-guard.sh, found commands: {all_commands}"
        )

    def test_ammo_impl_champion_deployed(self, shared_session):
        """agents/ammo-impl-champion.md must exist and contain key content."""
        session_id, headers, claude_dir = shared_session

        path = f"{claude_dir}/agents/ammo-impl-champion.md"
        assert _file_exists_in_container(path), f"File not found: {path}"

        content = _read_container_file(path)
        assert content is not None, f"Could not read {path}"

        assert "Worktree Isolation" in content, (
            "ammo-impl-champion.md should contain 'Worktree Isolation'"
        )
        assert "EnterWorktree" in content, (
            "ammo-impl-champion.md should contain 'EnterWorktree'"
        )

    def test_ammo_impl_validator_deployed(self, shared_session):
        """agents/ammo-impl-validator.md must exist and contain key content."""
        session_id, headers, claude_dir = shared_session

        path = f"{claude_dir}/agents/ammo-impl-validator.md"
        assert _file_exists_in_container(path), f"File not found: {path}"

        content = _read_container_file(path)
        assert content is not None, f"Could not read {path}"

        assert "Adversarial Verification" in content, (
            "ammo-impl-validator.md should contain 'Adversarial Verification'"
        )

    def test_worktree_hook_has_base_repo(self, shared_session):
        """hooks/worktree-create-with-build.sh must contain BASE_REPO logic."""
        session_id, headers, claude_dir = shared_session

        path = f"{claude_dir}/hooks/worktree-create-with-build.sh"
        assert _file_exists_in_container(path), f"File not found: {path}"

        content = _read_container_file(path)
        assert content is not None, f"Could not read {path}"

        assert "BASE_REPO" in content, (
            "worktree-create-with-build.sh should contain 'BASE_REPO'"
        )

    def test_stop_guard_has_stage_nudges(self, shared_session):
        """hooks/ammo-stop-guard.sh must contain stage-specific nudge logic."""
        session_id, headers, claude_dir = shared_session

        path = f"{claude_dir}/hooks/ammo-stop-guard.sh"
        assert _file_exists_in_container(path), f"File not found: {path}"

        content = _read_container_file(path)
        assert content is not None, f"Could not read {path}"

        assert "7_campaign_eval" in content, (
            "ammo-stop-guard.sh should contain '7_campaign_eval' stage nudge"
        )

    def test_skill_md_has_config_fidelity(self, shared_session):
        """skills/ammo/SKILL.md must contain Configuration Fidelity section."""
        session_id, headers, claude_dir = shared_session

        path = f"{claude_dir}/skills/ammo/SKILL.md"
        assert _file_exists_in_container(path), f"File not found: {path}"

        content = _read_container_file(path)
        assert content is not None, f"Could not read {path}"

        assert "Configuration Fidelity" in content, (
            "skills/ammo/SKILL.md should contain 'Configuration Fidelity'"
        )
