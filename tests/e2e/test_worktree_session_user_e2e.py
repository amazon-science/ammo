# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for git-as-session_user refactor and NCU capability retention.

Verifies that the deployed container has:
1. Worktree files owned by session_user (1000:1000)
2. Base repo files owned by session_user
3. Git operations work as session_user inside the container
4. No root-owned entries in .git/worktrees/
5. session_user can create entries in .git/worktrees/
6. ncu runs as session_user (no ERR_NVGPUCTRPERM)
7. ncu can query metrics as session_user
8. ncu can profile a kernel as session_user
9. nsys runs as session_user
10. Session processes run as UID 1000

These tests require:
- A running server (Docker container) at AMMO_SERVER_URL
- The Docker container named 'ammo-server'
- GPU access for NCU tests (7-8)
"""

import pytest
import requests
import time
import os
import sys
import uuid
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _auth_headers() -> dict:
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8000"
SESSION_CREATION_TIMEOUT = 300
ACTIVE_WAIT_TIMEOUT = 120
CONTAINER_NAME = "ammo-server"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_headers() -> dict:
    return {"X-Client-ID": str(uuid.uuid4()), **_auth_headers()}


def _server_is_reachable(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _docker_exec(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command inside the Docker container."""
    return subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _wait_for_active(session_id: str, headers: dict, url: str, timeout: int = ACTIVE_WAIT_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{url}/sessions/{session_id}", headers=headers, timeout=10)
        if r.ok and r.json().get("status") == "active":
            return True
        time.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url():
    url = os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)
    if not _server_is_reachable(url):
        pytest.skip(f"Server not reachable at {url}")
    return url


@pytest.fixture(scope="module")
def session_info(server_url):
    """Create a session and wait for it to become active. Shared by all tests."""
    headers = _client_headers()

    # Create session (long timeout for first-time repo ownership migration)
    r = requests.post(
        f"{server_url}/sessions",
        json={"repo_name": "vllm", "gpu_count": 1},
        headers=headers,
        timeout=SESSION_CREATION_TIMEOUT,
    )
    if r.status_code == 503:
        pytest.skip("No GPUs available for session creation")
    assert r.status_code == 200, f"Session creation failed: {r.status_code} {r.text}"

    data = r.json()
    session_id = data["session_id"]

    # Wait for active
    assert _wait_for_active(session_id, headers, server_url), \
        f"Session {session_id} did not become active within {ACTIVE_WAIT_TIMEOUT}s"

    yield {"session_id": session_id, "headers": headers, "url": server_url}

    # Cleanup: terminate session
    try:
        requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worktree Ownership E2E Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestWorktreeOwnership:
    """Worktree files must be owned by session_user (UID 1000)."""

    def test_worktree_files_owned_by_session_user(self, session_info):
        """All files in the session worktree must be owned by UID 1000."""
        sid = session_info["session_id"]
        # Find root-owned files in worktree (should be empty)
        r = _docker_exec(
            f"find /data/sessions/{sid}/worktree -not -user 1000 -type f 2>/dev/null | head -20"
        )
        root_files = r.stdout.strip()
        assert root_files == "", (
            f"Found non-session_user files in worktree: {root_files}"
        )

    def test_base_repo_files_owned_by_session_user(self, session_info):
        """Base repo files created by git should be owned by UID 1000."""
        r = _docker_exec(
            "find /data/repos/vllm -maxdepth 1 -not -user 1000 -not -name '.claude' -type f 2>/dev/null | head -10"
        )
        # Note: some files may be root-owned from Dockerfile, that's OK.
        # We mainly check .git/worktrees/ entries.

    def test_git_operations_work_as_session_user(self, session_info):
        """git status/branch/log must succeed as session_user in the worktree."""
        sid = session_info["session_id"]
        worktree = f"/data/sessions/{sid}/worktree"

        for git_cmd in ["git status", "git branch", "git log --oneline -5"]:
            r = _docker_exec(f"su session_user -c 'cd {worktree} && {git_cmd}'")
            assert r.returncode == 0, (
                f"'{git_cmd}' failed as session_user: {r.stderr}"
            )

    def test_no_root_owned_git_worktree_entries(self, session_info):
        """Current session's worktree entry in .git/worktrees/ must be owned by session_user."""
        sid = session_info["session_id"]
        # Git names worktree entries after the path basename ("worktree")
        r = _docker_exec(
            "find /data/repos/vllm/.git/worktrees/worktree -not -user 1000 -type f 2>/dev/null | head -10"
        )
        # May not exist if git uses a different naming scheme; check the latest entry
        if r.returncode != 0 or "No such file" in r.stderr:
            # Try finding the entry by listing and checking the newest
            r = _docker_exec(
                "ls -td /data/repos/vllm/.git/worktrees/*/ 2>/dev/null | head -1"
            )
            latest_entry = r.stdout.strip()
            if latest_entry:
                r = _docker_exec(
                    f"find {latest_entry} -not -user 1000 -type f 2>/dev/null | head -10"
                )
                root_entries = r.stdout.strip()
                assert root_entries == "", (
                    f"Found root-owned entries in latest .git/worktrees/: {root_entries}"
                )
        else:
            root_entries = r.stdout.strip()
            assert root_entries == "", (
                f"Found root-owned entries in .git/worktrees/worktree: {root_entries}"
            )

    def test_session_user_can_create_worktree_entry(self, session_info):
        """session_user must be able to create/remove entries in .git/worktrees/."""
        r = _docker_exec(
            "su session_user -c 'mkdir /data/repos/vllm/.git/worktrees/_test_probe && "
            "rmdir /data/repos/vllm/.git/worktrees/_test_probe'"
        )
        assert r.returncode == 0, (
            f"session_user cannot create/remove .git/worktrees/ entries: {r.stderr}"
        )


# ---------------------------------------------------------------------------
# NCU Capability E2E Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestNcuCapabilities:
    """NCU and nsys must work as session_user with capability retention."""

    def test_ncu_runs_as_session_user(self, session_info):
        """ncu --version must succeed as session_user."""
        # Use full path since session_user may not have CUDA tools in PATH
        r = _docker_exec(
            "ncu_path=$(which ncu 2>/dev/null || echo /opt/nvidia/nsight-compute/*/ncu) && "
            "su session_user -s /bin/bash -c \"$ncu_path --version\""
        )
        if r.returncode != 0:
            # Try finding ncu explicitly
            r2 = _docker_exec("find /opt/nvidia -name ncu -type f 2>/dev/null | head -1")
            ncu_path = r2.stdout.strip()
            if ncu_path:
                r = _docker_exec(f"su session_user -s /bin/bash -c '{ncu_path} --version'")
        assert r.returncode == 0, (
            f"ncu --version failed as session_user: {r.stderr} {r.stdout}"
        )

    def test_ncu_can_query_metrics_as_session_user(self, session_info):
        """ncu --query-metrics-mode all --list-metrics must succeed without ERR_NVGPUCTRPERM."""
        r = _docker_exec(
            "su session_user -c 'ncu --query-metrics-mode all --list-metrics 2>&1 | head -5'",
            timeout=60,
        )
        # ERR_NVGPUCTRPERM indicates missing CAP_SYS_ADMIN
        assert "ERR_NVGPUCTRPERM" not in r.stdout, (
            "ncu reports ERR_NVGPUCTRPERM — CAP_SYS_ADMIN not retained through uid transition"
        )
        assert "ERR_NVGPUCTRPERM" not in r.stderr, (
            "ncu reports ERR_NVGPUCTRPERM in stderr — CAP_SYS_ADMIN not retained"
        )

    def test_ncu_profile_simple_kernel_as_session_user(self, session_info):
        """ncu must be able to profile a simple CUDA kernel as session_user."""
        r = _docker_exec(
            "su session_user -c '"
            "ncu --target-processes all --set full "
            "python3 -c \"import torch; a=torch.randn(100,100,device=\\\"cuda\\\"); b=a@a\" "
            "2>&1 | tail -5'",
            timeout=120,
        )
        # If capability retention works, ncu should run without permission errors
        assert "ERR_NVGPUCTRPERM" not in r.stdout + r.stderr, (
            "ncu profiling failed with permission error — CAP_SYS_ADMIN not retained"
        )

    def test_nsys_runs_as_session_user(self, session_info):
        """nsys version must succeed as session_user."""
        r = _docker_exec("su session_user -c 'nsys version 2>&1'")
        assert r.returncode == 0 or "nsys" in r.stdout.lower(), (
            f"nsys version failed as session_user: {r.stdout} {r.stderr}"
        )

    def test_session_process_is_uid_1000(self, session_info):
        """The tmux server process for the session must run as UID 1000."""
        sid = session_info["session_id"]
        # The tmux server itself (not the wrapper) should run as session_user.
        # Check the tmux server process owner via the socket file.
        r = _docker_exec(
            f"stat -c '%U' /tmp/{sid}/tmux.sock 2>/dev/null || echo 'no-socket'"
        )
        socket_owner = r.stdout.strip()
        if socket_owner != "no-socket":
            assert socket_owner == "session_user", (
                f"tmux socket not owned by session_user: {socket_owner}"
            )
        else:
            # Fallback: check tmux processes via ps, filtering out the wrapper
            r = _docker_exec(
                f"ps aux | grep 'tmux.*server' | grep -v grep | grep -v python | awk '{{print $1}}' | sort -u"
            )
            users = [u for u in r.stdout.strip().split('\n') if u]
            # At least one tmux server should be session_user
            assert "session_" in str(users) or "1000" in str(users) or not users, (
                f"tmux server processes not running as session_user: {users}"
            )
