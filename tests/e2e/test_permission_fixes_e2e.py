# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for git worktree permission fix and tmux exit-empty config.

Verifies that the deployed container has:
1. /data/repos/vllm/.git/worktrees/ owned by session_user with sticky bit
2. session_user can write to .git/worktrees/
3. session_user can write to .claude/worktrees/ (flock file location)
4. tmux hardened config includes exit-empty on
5. tmux config has the full lifecycle chain (remain-on-exit, exit-empty, destroy-unattached)
6. worktree hook contains BASE_REPO logic (lock fix deployed)

These tests require:
- A running server (Docker container) at AMMO_SERVER_URL
- The Docker container named 'ammo-server'
"""

import pytest
import requests
import time
import os
import sys
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


def _exec_in_container(cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a shell command inside the Docker container."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "sh", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def _exec_as_session_user(cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a command inside the container as session_user (uid 1000)."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "su", "-s", "/bin/sh", "session_user", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


# ---------------------------------------------------------------------------
# Session fixture (shared across all tests in this module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="module")
def shared_session(server_url):
    """Create a single session and reuse it across all tests in this module.

    Session creation takes ~30s, so we share one session for all permission
    verification tests. Yields (session_id, headers).
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

        yield session_id, headers

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
class TestPermissionFixes:
    """Verify git worktree permissions and tmux config are correctly deployed."""

    def test_git_worktrees_dir_writable_by_session_user(self, shared_session):
        """The .git/worktrees/ dir must be owned by session_user with sticky bit."""
        session_id, headers = shared_session

        result = _exec_in_container(
            "stat -c '%U:%G %a' /data/repos/vllm/.git/worktrees/"
        )
        assert result.returncode == 0, (
            f"stat failed: {result.stderr}"
        )

        output = result.stdout.strip()
        # Expect something like "session_user:session_user 1777"
        # or "1000:1000 1777" if name resolution isn't available
        parts = output.split()
        assert len(parts) == 2, f"Unexpected stat output format: {output}"

        owner = parts[0]
        mode = parts[1]

        # Check owner is session_user (or 1000:1000)
        assert owner in ("session_user:session_user", "1000:1000"), (
            f"Expected owner session_user:session_user or 1000:1000, got: {owner}"
        )

        # Check mode includes sticky bit (1777) or at least world-writable with sticky
        assert mode.startswith("1") and mode.endswith("777"), (
            f"Expected mode 1777 (sticky + world-writable), got: {mode}"
        )

    def test_session_user_can_write_to_git_worktrees(self, shared_session):
        """session_user must be able to create and remove files in .git/worktrees/."""
        session_id, headers = shared_session

        result = _exec_as_session_user(
            "touch /data/repos/vllm/.git/worktrees/.e2e_write_test "
            "&& rm /data/repos/vllm/.git/worktrees/.e2e_write_test "
            "&& echo OK"
        )
        assert result.returncode == 0, (
            f"Write test failed (rc={result.returncode}): {result.stderr}"
        )
        assert "OK" in result.stdout, (
            f"Expected 'OK' in output, got: {result.stdout}"
        )

    def test_session_user_can_write_to_base_repo_claude_worktrees(self, shared_session):
        """session_user must be able to write to .claude/worktrees/ (flock location)."""
        session_id, headers = shared_session

        result = _exec_as_session_user(
            "touch /data/repos/vllm/.claude/worktrees/.e2e_lock_test "
            "&& rm /data/repos/vllm/.claude/worktrees/.e2e_lock_test "
            "&& echo OK"
        )
        assert result.returncode == 0, (
            f"Write test to .claude/worktrees/ failed (rc={result.returncode}): {result.stderr}"
        )
        assert "OK" in result.stdout, (
            f"Expected 'OK' in output, got: {result.stdout}"
        )

    def test_tmux_config_has_exit_empty(self, shared_session):
        """The tmux hardened config must include 'exit-empty on'."""
        session_id, headers = shared_session

        # Find the tmux.conf file created for the session
        result = _exec_in_container(
            "find /tmp -name tmux.conf 2>/dev/null | head -5"
        )
        assert result.returncode == 0, f"find failed: {result.stderr}"

        tmux_conf_paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        assert len(tmux_conf_paths) > 0, (
            "No tmux.conf files found in /tmp; session may not have started tmux"
        )

        # Read the first one (or find one matching our session)
        # Try to find one matching session_id prefix
        matching = [p for p in tmux_conf_paths if session_id[:8] in p]
        conf_path = matching[0] if matching else tmux_conf_paths[0]

        content = _read_container_file(conf_path)
        assert content is not None, f"Could not read tmux config at {conf_path}"

        assert "exit-empty on" in content, (
            f"tmux config at {conf_path} missing 'exit-empty on'. Content:\n{content}"
        )

    def test_tmux_config_has_full_lifecycle_chain(self, shared_session):
        """tmux config must have all three lifecycle settings for clean shutdown."""
        session_id, headers = shared_session

        # Find the tmux.conf file
        result = _exec_in_container(
            "find /tmp -name tmux.conf 2>/dev/null | head -5"
        )
        assert result.returncode == 0, f"find failed: {result.stderr}"

        tmux_conf_paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        assert len(tmux_conf_paths) > 0, "No tmux.conf files found"

        matching = [p for p in tmux_conf_paths if session_id[:8] in p]
        conf_path = matching[0] if matching else tmux_conf_paths[0]

        content = _read_container_file(conf_path)
        assert content is not None, f"Could not read tmux config at {conf_path}"

        required_settings = [
            "remain-on-exit off",
            "exit-empty on",
            "destroy-unattached on",
        ]
        for setting in required_settings:
            assert setting in content, (
                f"tmux config at {conf_path} missing '{setting}'. Content:\n{content}"
            )

    def test_worktree_hook_has_base_repo_in_container(self, shared_session):
        """The worktree hook in the session must contain BASE_REPO logic."""
        session_id, headers = shared_session

        hook_path = f"/data/sessions/{session_id}/worktree/.claude/hooks/worktree-create-with-build.sh"
        content = _read_container_file(hook_path)
        assert content is not None, (
            f"Hook file not found at {hook_path}"
        )

        assert "BASE_REPO" in content, (
            f"worktree-create-with-build.sh missing 'BASE_REPO' (lock fix not deployed). "
            f"First 200 chars: {content[:200]}"
        )
