# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for full session lifecycle.

Tests the complete flow: create -> verify terminal accessible -> pause -> resume -> terminate.

These tests require:
- A running server (Docker container or direct) at AMMO_SERVER_URL
- The vllm repository available for worktree creation

Skip conditions:
- Server not reachable
- Session creation fails (e.g., repo not cloned)
"""

import pytest
import requests
import time
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8000"
SESSION_CREATION_TIMEOUT = 300  # seconds (editable_install can take 200s+ under load)
ACTIVE_WAIT_TIMEOUT = 120  # seconds to wait for session to become active


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _client_headers() -> dict:
    """Return combined client-id + auth headers for session isolation."""
    return {"X-Client-ID": str(uuid.uuid4()), **_auth_headers()}


@pytest.fixture
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture
def headers():
    """Combined client-id and auth headers, regenerated per test."""
    return _client_headers()


def _server_is_reachable(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_active(server_url: str, session_id: str, headers: dict = None, timeout: float = ACTIVE_WAIT_TIMEOUT) -> dict:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.slow
class TestSessionFullLifecycle:
    """Full create -> terminal -> pause -> resume -> terminate."""

    @pytest.fixture(autouse=True)
    def _check_server(self, server_url):
        if not _server_is_reachable(server_url):
            pytest.skip("Server not reachable")

    def test_full_lifecycle(self, server_url, headers):
        """
        Complete session lifecycle:
        1. Create session (0 GPUs, to be safe in any env)
        2. Wait for active status
        3. Verify terminal URL is populated
        4. Verify terminal endpoint responds
        5. Pause session -> verify paused
        6. Resume session -> verify active again
        7. Terminate -> verify terminated
        """
        session_id = None
        try:
            # --- 1. Create ---
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
            assert create_data["cli_tool"] == "claude"
            assert create_data["repo_name"] == "vllm"

            # --- 2. Wait for active ---
            session_data = _wait_for_active(server_url, session_id, headers=headers)
            if session_data.get("status") != "active":
                pytest.skip(
                    f"Session did not become active: {session_data.get('status', 'unknown')}"
                )

            # --- 3. Verify terminal URL is set ---
            terminal_url = session_data.get("terminal_url")
            assert terminal_url is not None, "Active session should have a terminal_url"
            assert session_id in terminal_url

            # --- 4. Verify terminal endpoint responds ---
            term_resp = requests.get(
                f"{server_url}{terminal_url}",
                headers=headers,
                timeout=10,
                allow_redirects=True,
            )
            # 200 means ttyd is proxied; 502/503 if ttyd not ready yet
            assert term_resp.status_code in (200, 502, 503), (
                f"Terminal returned unexpected status {term_resp.status_code}"
            )

            # --- 5. Pause ---
            pause_resp = requests.post(
                f"{server_url}/sessions/{session_id}/pause",
                headers=headers,
                timeout=30,
            )
            assert pause_resp.status_code == 200
            pause_data = pause_resp.json()
            assert pause_data.get("status") == "paused"

            # Confirm via GET
            get_resp = requests.get(
                f"{server_url}/sessions/{session_id}", headers=headers, timeout=10
            )
            assert get_resp.status_code == 200
            assert get_resp.json()["status"] == "paused"
            # Paused sessions should NOT have terminal URLs
            assert get_resp.json().get("terminal_url") is None

            # --- 6. Resume ---
            resume_resp = requests.post(
                f"{server_url}/sessions/{session_id}/resume",
                json={},
                headers=headers,
                timeout=60,
            )
            assert resume_resp.status_code == 200
            resume_data = resume_resp.json()
            assert resume_data.get("status") in ("creating", "active")

            # Wait for active again
            session_data = _wait_for_active(server_url, session_id, headers=headers)
            if session_data.get("status") == "active":
                assert session_data.get("terminal_url") is not None

            # --- 7. Terminate ---
            term_resp = requests.delete(
                f"{server_url}/sessions/{session_id}",
                headers=headers,
                timeout=30,
            )
            assert term_resp.status_code == 200
            term_data = term_resp.json()
            assert term_data.get("status") == "terminated"

            # Verify final state
            final = requests.get(
                f"{server_url}/sessions/{session_id}", headers=headers, timeout=10
            )
            if final.status_code == 200:
                assert final.json()["status"] == "terminated"
            else:
                # Session may have been fully cleaned up
                assert final.status_code == 404

            session_id = None  # Prevent cleanup since we already terminated

        finally:
            if session_id:
                requests.delete(
                    f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
                )

    def test_session_appears_in_listing(self, server_url, headers):
        """Created session shows up in GET /sessions list."""
        session_id = None
        try:
            resp = requests.post(
                f"{server_url}/sessions",
                json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if resp.status_code != 200:
                pytest.skip("Session creation failed")

            session_id = resp.json()["session_id"]

            # Verify it appears in the list
            list_resp = requests.get(
                f"{server_url}/sessions", headers=headers, timeout=10
            )
            assert list_resp.status_code == 200
            ids = [s["session_id"] for s in list_resp.json()["sessions"]]
            assert session_id in ids

        finally:
            if session_id:
                requests.delete(
                    f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
                )

    def test_session_with_model_metadata(self, server_url, headers):
        """Session preserves model_name and dtype from creation request."""
        session_id = None
        try:
            resp = requests.post(
                f"{server_url}/sessions",
                json={
                    "repo_name": "vllm",
                    "cli_tool": "claude",
                    "gpu_count": 0,
                    "model_name": "deepseek-ai/DeepSeek-R1-0528",
                    "dtype": "fp8",
                },
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if resp.status_code != 200:
                pytest.skip("Session creation failed")

            data = resp.json()
            session_id = data["session_id"]
            assert data.get("model_name") == "deepseek-ai/DeepSeek-R1-0528"
            assert data.get("dtype") == "fp8"

            # Verify via GET
            get_resp = requests.get(
                f"{server_url}/sessions/{session_id}", headers=headers, timeout=10
            )
            if get_resp.status_code == 200:
                get_data = get_resp.json()
                assert get_data.get("model_name") == "deepseek-ai/DeepSeek-R1-0528"
                assert get_data.get("dtype") == "fp8"

        finally:
            if session_id:
                requests.delete(
                    f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
                )


@pytest.mark.e2e
class TestSessionErrorHandling:
    """Error handling edge cases for sessions."""

    @pytest.fixture(autouse=True)
    def _check_server(self, server_url):
        if not _server_is_reachable(server_url):
            pytest.skip("Server not reachable")

    def test_double_terminate(self, server_url, headers):
        """Terminating an already-terminated session returns 400 or is idempotent."""
        resp = requests.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT,
        )
        if resp.status_code != 200:
            pytest.skip("Session creation failed")

        session_id = resp.json()["session_id"]

        # First terminate
        r1 = requests.delete(
            f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
        )
        assert r1.status_code == 200

        # Second terminate -- may be 200 (idempotent) or 400 (already terminated)
        r2 = requests.delete(
            f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
        )
        assert r2.status_code in (200, 400, 404)

    def test_pause_terminated_session(self, server_url, headers):
        """Pausing a terminated session returns 400."""
        resp = requests.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT,
        )
        if resp.status_code != 200:
            pytest.skip("Session creation failed")

        session_id = resp.json()["session_id"]
        requests.delete(
            f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
        )

        # Try to pause terminated session
        r = requests.post(
            f"{server_url}/sessions/{session_id}/pause", headers=headers, timeout=10
        )
        assert r.status_code in (400, 404)
