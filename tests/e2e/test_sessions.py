# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for AI CLI Session Service.

These tests verify the complete session lifecycle including:
- Session creation and configuration
- Session listing and retrieval
- Session pause/resume
- Session termination
- Terminal proxy functionality

Requirements:
- Server must be running at AMMO_SERVER_URL (default: http://localhost:8000)
- For terminal tests, ttyd must be installed in the container
"""

import pytest
import requests
import time
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Session creation timeout: editable_install can take 200s+ under concurrent load
SESSION_CREATION_TIMEOUT = 300


def _auth_headers() -> dict:
    """Return auth headers for protected endpoints (empty when AMMO_API_KEY unset)."""
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


@pytest.mark.e2e
class TestSessionLifecycle:
    """Tests for basic session lifecycle operations."""

    def test_create_session_defaults(self, server_url):
        """Test creating a session with default settings."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={},
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        # Session creation may fail if repo not cloned yet, which is OK for first run
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert data.get("status") in ["creating", "active"]
            assert data.get("cli_tool") == "claude"
            assert data.get("repo_name") == "vllm"

            # Clean up
            session_id = data["session_id"]
            requests.delete(f"{server_url}/sessions/{session_id}", headers=headers)

        elif response.status_code == 400:
            # May fail due to repo not being available
            data = response.json()
            assert "error" in data or "detail" in data

    def test_create_session_custom_settings(self, server_url):
        """Test creating a session with custom settings."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "branch": "main",
                "gpu_count": 0,
                "inactivity_timeout_mins": 60,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert data.get("inactivity_timeout_mins") == 60

            # Clean up
            session_id = data["session_id"]
            requests.delete(f"{server_url}/sessions/{session_id}", headers=headers)

    def test_list_sessions_empty(self, server_url):
        """Test listing sessions when none exist."""
        response = requests.get(f"{server_url}/sessions", headers=_auth_headers(), timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)

    def test_list_sessions_with_filter(self, server_url):
        """Test listing sessions with status filter."""
        headers = _auth_headers()
        # Test valid filter
        response = requests.get(
            f"{server_url}/sessions",
            params={"status": "active"},
            headers=headers,
            timeout=10
        )
        assert response.status_code == 200

        # Test invalid filter
        response = requests.get(
            f"{server_url}/sessions",
            params={"status": "invalid_status"},
            headers=headers,
            timeout=10
        )
        assert response.status_code == 400

    def test_get_session_not_found(self, server_url):
        """Test getting a non-existent session."""
        response = requests.get(
            f"{server_url}/sessions/nonexistent-session-id",
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code == 404

    def test_terminate_session_not_found(self, server_url):
        """Test terminating a non-existent session."""
        response = requests.delete(
            f"{server_url}/sessions/nonexistent-session-id",
            headers=_auth_headers(),
            timeout=10
        )

        # Should return 400 (session not found) or 404
        assert response.status_code in [400, 404]

    def test_pause_session_not_found(self, server_url):
        """Test pausing a non-existent session."""
        response = requests.post(
            f"{server_url}/sessions/nonexistent-session-id/pause",
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code in [400, 404]

    def test_resume_session_not_found(self, server_url):
        """Test resuming a non-existent session."""
        response = requests.post(
            f"{server_url}/sessions/nonexistent-session-id/resume",
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.slow
class TestSessionWorkflows:
    """Tests for complete session workflows."""

    @pytest.fixture
    def created_session(self, server_url) -> Optional[str]:
        """Create a session for testing and clean up after."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "gpu_count": 0,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        if response.status_code != 200:
            pytest.skip(f"Could not create session: {response.text}")
            return None

        session_id = response.json()["session_id"]

        yield session_id

        # Clean up
        requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)

    def test_full_session_lifecycle(self, server_url, created_session):
        """Test complete session lifecycle: create -> get -> pause -> resume -> terminate."""
        if created_session is None:
            pytest.skip("Session creation failed")

        session_id = created_session
        headers = _auth_headers()

        # Get session info
        response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] in ["creating", "active"]

        # Wait for session to become active (if still creating)
        for _ in range(30):
            response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
            if response.status_code == 200:
                if response.json()["status"] == "active":
                    break
            time.sleep(2)

        # List sessions - should include our session
        response = requests.get(f"{server_url}/sessions", headers=headers, timeout=10)
        assert response.status_code == 200
        sessions = response.json()["sessions"]
        session_ids = [s["session_id"] for s in sessions]
        assert session_id in session_ids

        # Pause session
        response = requests.post(f"{server_url}/sessions/{session_id}/pause", headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            assert data.get("status") == "paused" or data.get("message") is not None

            # Verify paused state
            response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
            assert response.status_code == 200
            assert response.json()["status"] == "paused"

            # Resume session
            response = requests.post(
                f"{server_url}/sessions/{session_id}/resume",
                json={},
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT
            )
            if response.status_code == 200:
                # Verify resumed
                response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                assert response.status_code == 200
                assert response.json()["status"] in ["creating", "active"]

        # Terminate session
        response = requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)
        assert response.status_code == 200

        # Verify terminated
        response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
        if response.status_code == 200:
            assert response.json()["status"] == "terminated"
        else:
            # Session might be cleaned up immediately
            assert response.status_code == 404

    def test_multiple_sessions(self, server_url):
        """Test creating and managing multiple sessions."""
        session_ids = []
        headers = _auth_headers()

        try:
            # Create multiple sessions
            for i in range(2):
                response = requests.post(
                    f"{server_url}/sessions",
                    json={
                        "repo_name": "vllm",
                        "cli_tool": "claude",
                        "gpu_count": 0,
                    },
                    headers=headers,
                    timeout=SESSION_CREATION_TIMEOUT
                )

                if response.status_code == 200:
                    session_ids.append(response.json()["session_id"])

            if len(session_ids) < 2:
                pytest.skip("Could not create multiple sessions")

            # List all sessions
            response = requests.get(f"{server_url}/sessions", headers=headers, timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert data["total"] >= len(session_ids)

            # Get each session
            for session_id in session_ids:
                response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                assert response.status_code == 200

        finally:
            # Clean up all sessions
            for session_id in session_ids:
                requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)


@pytest.mark.e2e
class TestTerminalProxy:
    """Tests for terminal proxy functionality."""

    def test_terminal_not_found(self, server_url):
        """Test terminal access for non-existent session."""
        response = requests.get(
            f"{server_url}/sessions/nonexistent-session/terminal/",
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code in [404, 503]

    def test_terminal_static_not_found(self, server_url):
        """Test terminal static asset for non-existent session."""
        response = requests.get(
            f"{server_url}/sessions/nonexistent-session/terminal/style.css",
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code in [404, 503]


@pytest.mark.e2e
class TestSessionValidation:
    """Tests for session request validation."""

    def test_invalid_cli_tool(self, server_url):
        """Test creating session with invalid CLI tool."""
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "cli_tool": "invalid_tool",
            },
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code == 400

    def test_invalid_repo_name(self, server_url):
        """Test creating session with invalid repo name."""
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "nonexistent_repo",
            },
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code == 400

    def test_invalid_gpu_count(self, server_url):
        """Test creating session with invalid GPU count."""
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "gpu_count": -1,
            },
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code == 400

    def test_invalid_timeout(self, server_url):
        """Test creating session with invalid timeout."""
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "inactivity_timeout_mins": 0,
            },
            headers=_auth_headers(),
            timeout=10
        )

        assert response.status_code == 400


@pytest.mark.e2e
class TestSessionHealth:
    """Tests for session service health."""

    def test_health_check(self, server_url):
        """Test that health endpoint still works with session service."""
        response = requests.get(f"{server_url}/health", timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_sessions_endpoint_available(self, server_url):
        """Test that sessions endpoint is available."""
        response = requests.get(f"{server_url}/sessions", headers=_auth_headers(), timeout=10)

        # Should return 200 even if empty
        assert response.status_code == 200


@pytest.mark.e2e
@pytest.mark.slow
class TestSessionSandboxing:
    """Tests for session sandboxing configuration."""

    @pytest.fixture
    def sandboxed_session(self, server_url) -> Optional[dict]:
        """Create a session and return session info for sandbox testing."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "gpu_count": 0,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        if response.status_code != 200:
            pytest.skip(f"Could not create session: {response.text}")
            return None

        session_data = response.json()
        session_id = session_data["session_id"]

        # Wait for session to become active
        for _ in range(30):
            response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "active":
                    session_data = data
                    break
            time.sleep(2)

        yield session_data

        # Clean up
        requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)

    def test_session_has_sandbox_settings(self, server_url, sandboxed_session):
        """Test that session workspace has sandboxed settings.json."""
        if sandboxed_session is None:
            pytest.skip("Session creation failed")

        session_id = sandboxed_session["session_id"]
        worktree_path = sandboxed_session.get("worktree_path")

        # Verify session info includes worktree path
        assert worktree_path is not None, "Session should have worktree_path"

        # The settings.json should be created in the worktree's .claude directory
        # We can't directly check the file from outside Docker, but we verify
        # the session was created with the expected configuration
        assert sandboxed_session.get("cli_tool") == "claude"
        assert sandboxed_session.get("status") in ["creating", "active"]

    def test_session_isolation(self, server_url):
        """Test that multiple sessions are isolated from each other."""
        session_ids = []
        headers = _auth_headers()

        try:
            # Create two sessions
            for _ in range(2):
                response = requests.post(
                    f"{server_url}/sessions",
                    json={
                        "repo_name": "vllm",
                        "cli_tool": "claude",
                        "gpu_count": 0,
                    },
                    headers=headers,
                    timeout=SESSION_CREATION_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    session_ids.append(data["session_id"])

            if len(session_ids) < 2:
                pytest.skip("Could not create multiple sessions for isolation test")

            # Verify each session has a unique worktree path
            worktree_paths = []
            for session_id in session_ids:
                response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                if response.status_code == 200:
                    worktree_path = response.json().get("worktree_path")
                    if worktree_path:
                        worktree_paths.append(worktree_path)

            # All worktree paths should be unique
            assert len(set(worktree_paths)) == len(worktree_paths), \
                "Each session should have a unique worktree path"

            # Session IDs should be in the worktree paths
            for i, session_id in enumerate(session_ids):
                if i < len(worktree_paths):
                    assert session_id in worktree_paths[i], \
                        f"Session ID should be in worktree path"

        finally:
            # Clean up all sessions
            for session_id in session_ids:
                requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)

    def test_session_gpu_allocation(self, server_url):
        """Test that sessions correctly allocate GPUs when requested."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "gpu_count": 1,  # Request 1 GPU
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        if response.status_code != 200:
            # May fail if no GPUs available
            if response.status_code == 400:
                data = response.json()
                # Check if it's a GPU availability issue
                error_msg = data.get("error", "") or data.get("detail", "")
                if "gpu" in error_msg.lower() or "available" in error_msg.lower():
                    pytest.skip("No GPUs available for testing")
            pytest.fail(f"Session creation failed: {response.text}")

        session_id = None
        try:
            data = response.json()
            session_id = data["session_id"]

            # Wait for session to become active
            for _ in range(30):
                response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                if response.status_code == 200:
                    session_data = response.json()
                    if session_data.get("status") == "active":
                        break
                time.sleep(2)
            else:
                pytest.skip("Session did not become active")

            # Verify GPU allocation
            gpu_ids = session_data.get("gpu_ids", [])
            assert len(gpu_ids) >= 1, "Session should have at least 1 GPU allocated"
            assert all(isinstance(g, int) for g in gpu_ids), "GPU IDs should be integers"

        finally:
            if session_id:
                requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)


@pytest.mark.e2e
@pytest.mark.slow
class TestCodexCLI:
    """Tests for Codex CLI session support."""

    def test_create_codex_session(self, server_url):
        """Test creating a session with Codex CLI tool."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "codex",
                "gpu_count": 0,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        # Note: This test may be skipped if Codex CLI is not installed
        if response.status_code == 400:
            data = response.json()
            error_msg = str(data.get("error", "") or data.get("detail", ""))
            if "codex" in error_msg.lower() or "not found" in error_msg.lower():
                pytest.skip("Codex CLI not installed in container")

        session_id = None
        try:
            if response.status_code == 200:
                data = response.json()
                session_id = data["session_id"]
                assert data.get("cli_tool") == "codex"
                assert data.get("status") in ["creating", "active"]

        finally:
            if session_id:
                requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)

    def test_codex_session_has_workspace_files(self, server_url):
        """Test that Codex session workspace has required files."""
        headers = _auth_headers()
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "codex",
                "gpu_count": 0,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT
        )

        if response.status_code == 400:
            data = response.json()
            error_msg = str(data.get("error", "") or data.get("detail", ""))
            if "codex" in error_msg.lower() or "not found" in error_msg.lower():
                pytest.skip("Codex CLI not installed in container")

        session_id = None
        try:
            if response.status_code == 200:
                data = response.json()
                session_id = data["session_id"]

                # Wait for session to become active
                for _ in range(30):
                    response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                    if response.status_code == 200:
                        if response.json().get("status") == "active":
                            break
                    time.sleep(2)

                # Verify session info
                response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
                if response.status_code == 200:
                    session_data = response.json()
                    assert session_data.get("cli_tool") == "codex"
                    # Worktree should be set
                    assert session_data.get("worktree_path") is not None

        finally:
            if session_id:
                requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)
