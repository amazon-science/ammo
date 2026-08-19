# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for session lifecycle REST API endpoints.

Tests the complete session CRUD operations via the HTTP API:
- Session creation with various configurations
- Session listing and retrieval with client isolation
- Session pause/resume lifecycle
- Session termination and batch cleanup
- Download preparation and redirect

Requirements:
- Server must be running at AMMO_SERVER_URL (default: http://localhost:8000)
"""

import pytest
import pytest_asyncio
import httpx
import uuid
import time
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERVER_URL_ENV = "AMMO_SERVER_URL"
DEFAULT_SERVER_URL = "http://localhost:8000"


def _server_url() -> str:
    import os
    return os.getenv(SERVER_URL_ENV, DEFAULT_SERVER_URL)


def _unique_client_id() -> str:
    """Generate a valid UUID v4 client ID for session isolation."""
    return str(uuid.uuid4())


def _auth_headers() -> dict:
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server_url():
    return _server_url()


@pytest.fixture
def client_id():
    """Unique client ID per test to ensure isolation."""
    return _unique_client_id()


@pytest.fixture
def client_headers(client_id):
    return {"X-Client-ID": client_id, **_auth_headers()}


@pytest_asyncio.fixture
async def async_client():
    # Session creation can take 200s+ (editable_install under concurrent load)
    async with httpx.AsyncClient(timeout=300) as client:
        yield client


@pytest_asyncio.fixture
async def created_session(server_url, client_headers, async_client):
    """Create a session and clean up after the test.

    Yields (session_id, create_response_dict) or skips if creation fails.
    """
    resp = await async_client.post(
        f"{server_url}/sessions",
        json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
        headers=client_headers,
    )
    if resp.status_code != 200:
        pytest.skip(f"Cannot create session (status {resp.status_code}): {resp.text}")

    data = resp.json()
    session_id = data["session_id"]

    yield session_id, data

    # Cleanup -- best-effort
    await async_client.delete(
        f"{server_url}/sessions/{session_id}",
        headers=client_headers,
    )


# ---------------------------------------------------------------------------
# Tests: Session Creation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionCreate:
    """POST /sessions -- session creation."""

    @pytest.mark.asyncio
    async def test_create_session_returns_required_fields(
        self, server_url, client_headers, async_client
    ):
        """Creating a session returns session_id, status, and terminal info."""
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
            headers=client_headers,
        )
        if resp.status_code != 200:
            pytest.skip(f"Session creation not available: {resp.text}")

        data = resp.json()
        assert "session_id" in data
        assert data["status"] in ("creating", "active")
        assert data["cli_tool"] == "claude"
        assert data["repo_name"] == "vllm"
        # created_at / last_accessed must be numeric timestamps
        assert isinstance(data["created_at"], (int, float))
        assert isinstance(data["last_accessed"], (int, float))

        # Cleanup
        await async_client.delete(
            f"{server_url}/sessions/{data['session_id']}",
            headers=client_headers,
        )

    @pytest.mark.asyncio
    async def test_create_session_insufficient_gpus_returns_503(
        self, server_url, client_headers, async_client
    ):
        """Requesting more GPUs than available returns a local 503."""
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 9999},
            headers=client_headers,
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("error") == "insufficient_gpus"
        assert "available" in data
        assert "requested" in data
        assert data["requested"] == 9999
        assert "retry_suggested" not in data
        assert "pods_with_gpus" not in data

    @pytest.mark.asyncio
    async def test_create_session_invalid_cli_tool(
        self, server_url, client_headers, async_client
    ):
        """Invalid cli_tool returns 400 (Pydantic validation)."""
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"cli_tool": "not_a_tool"},
            headers=client_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_invalid_repo(
        self, server_url, client_headers, async_client
    ):
        """Unsupported repo_name returns 400."""
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"repo_name": "nonexistent_repo_xyz"},
            headers=client_headers,
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_session_with_gpu_count_exceeding_tp_times_dp(
        self, server_url, client_headers, async_client
    ):
        """Decoupled gpu_count: pool may exceed tp*dp (spare GPUs for parallel tracks).

        Plan: .claude/plans/gpu-decouple.md Phase 1 test #9 / Task 6.

        Sends gpu_count=2, tp_size=1, dp_size=1 — post-decouple this is valid
        (tp*dp=1 <= gpu_count=2). Accepts either 200 (pod had the GPUs) or
        503 (pod short on capacity). MUST NEVER return 422: that would mean
        the validator regressed back to strict-equality.
        """
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "claude",
                "gpu_count": 2,
                "tp_size": 1,
                "dp_size": 1,
            },
            headers=client_headers,
        )
        # Validator regression gate: 422 means the <= invariant broke.
        assert resp.status_code != 422, (
            f"gpu_count (2) > tp*dp (1) must NOT be rejected by validator; "
            f"got 422: {resp.text}"
        )
        assert resp.status_code in (200, 503), (
            f"expected 200 (created) or 503 (insufficient GPUs); got "
            f"{resp.status_code}: {resp.text}"
        )

        if resp.status_code == 200:
            session_id = resp.json()["session_id"]
            try:
                info_resp = await async_client.get(
                    f"{server_url}/sessions/{session_id}",
                    headers=client_headers,
                )
                assert info_resp.status_code == 200
                info = info_resp.json()
                # Pool holds the full gpu_count; model replica remains tp*dp.
                assert info["requested_gpu_count"] == 2, (
                    f"requested_gpu_count must be preserved = 2, got {info.get('requested_gpu_count')}"
                )
                assert info["tp_size"] == 1, (
                    f"tp_size must remain 1 (not collapsed to gpu_count), got {info.get('tp_size')}"
                )
                assert info["dp_size"] == 1
            finally:
                await async_client.delete(
                    f"{server_url}/sessions/{session_id}",
                    headers=client_headers,
                )


# ---------------------------------------------------------------------------
# Tests: Session Listing & Retrieval
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionList:
    """GET /sessions -- session listing with client isolation."""

    @pytest.mark.asyncio
    async def test_list_sessions_returns_valid_structure(
        self, server_url, client_headers, async_client
    ):
        """List endpoint returns sessions array and total count."""
        resp = await async_client.get(
            f"{server_url}/sessions", headers=client_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)
        assert isinstance(data["total"], int)

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_client_id(
        self, server_url, async_client
    ):
        """Sessions created with one client ID are not visible to another."""
        client_a = _unique_client_id()
        client_b = _unique_client_id()
        auth = _auth_headers()

        # Create a session as client A
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
            headers={"X-Client-ID": client_a, **auth},
        )
        if resp.status_code != 200:
            pytest.skip("Cannot create session")

        session_id = resp.json()["session_id"]

        try:
            # Client B should NOT see client A's session
            resp_b = await async_client.get(
                f"{server_url}/sessions",
                headers={"X-Client-ID": client_b, **auth},
            )
            assert resp_b.status_code == 200
            ids_b = [s["session_id"] for s in resp_b.json()["sessions"]]
            assert session_id not in ids_b

            # Client A should see its own session
            resp_a = await async_client.get(
                f"{server_url}/sessions",
                headers={"X-Client-ID": client_a, **auth},
            )
            assert resp_a.status_code == 200
            ids_a = [s["session_id"] for s in resp_a.json()["sessions"]]
            assert session_id in ids_a
        finally:
            await async_client.delete(
                f"{server_url}/sessions/{session_id}",
                headers={"X-Client-ID": client_a, **auth},
            )

    @pytest.mark.asyncio
    async def test_list_sessions_invalid_status_filter(
        self, server_url, client_headers, async_client
    ):
        """Invalid status filter returns 400."""
        resp = await async_client.get(
            f"{server_url}/sessions",
            params={"status": "bogus_status"},
            headers=client_headers,
        )
        assert resp.status_code == 400


@pytest.mark.integration
class TestSessionGet:
    """GET /sessions/{session_id} -- individual session retrieval."""

    @pytest.mark.asyncio
    async def test_get_session_returns_full_info(
        self, server_url, client_headers, async_client, created_session
    ):
        """Returned session matches SessionInfo model fields."""
        session_id, _ = created_session
        resp = await async_client.get(
            f"{server_url}/sessions/{session_id}",
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields from SessionInfo
        assert data["session_id"] == session_id
        assert "status" in data
        assert "cli_tool" in data
        assert "repo_name" in data
        assert "branch" in data
        assert "gpu_ids" in data
        assert isinstance(data["gpu_ids"], list)
        assert "created_at" in data
        assert "last_accessed" in data
        assert "inactivity_timeout_mins" in data

    @pytest.mark.asyncio
    async def test_get_session_not_found(
        self, server_url, client_headers, async_client
    ):
        resp = await async_client.get(
            f"{server_url}/sessions/does-not-exist-12345",
            headers=client_headers,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Pause / Resume
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSessionPauseResume:
    """POST /sessions/{id}/pause and /sessions/{id}/resume."""

    @pytest.mark.asyncio
    async def test_pause_changes_status(
        self, server_url, client_headers, async_client, created_session
    ):
        session_id, _ = created_session

        # Wait for session to become active
        for _ in range(30):
            r = await async_client.get(
                f"{server_url}/sessions/{session_id}",
                headers=client_headers,
            )
            if r.status_code == 200 and r.json()["status"] == "active":
                break
            await _async_sleep(2)
        else:
            pytest.skip("Session did not become active in time")

        # Pause
        resp = await async_client.post(
            f"{server_url}/sessions/{session_id}/pause",
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "paused"

        # Verify via GET
        r = await async_client.get(
            f"{server_url}/sessions/{session_id}",
            headers=client_headers,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "paused"

    @pytest.mark.asyncio
    async def test_resume_after_pause(
        self, server_url, client_headers, async_client, created_session
    ):
        session_id, _ = created_session

        # Wait for active
        for _ in range(30):
            r = await async_client.get(
                f"{server_url}/sessions/{session_id}",
                headers=client_headers,
            )
            if r.status_code == 200 and r.json()["status"] == "active":
                break
            await _async_sleep(2)
        else:
            pytest.skip("Session did not become active")

        # Pause then resume
        await async_client.post(
            f"{server_url}/sessions/{session_id}/pause",
            headers=client_headers,
        )
        resp = await async_client.post(
            f"{server_url}/sessions/{session_id}/resume",
            json={},
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("creating", "active")

    @pytest.mark.asyncio
    async def test_pause_nonexistent_session(
        self, server_url, client_headers, async_client
    ):
        resp = await async_client.post(
            f"{server_url}/sessions/nonexistent-id-xyz/pause",
            headers=client_headers,
        )
        assert resp.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_resume_nonexistent_session(
        self, server_url, client_headers, async_client
    ):
        resp = await async_client.post(
            f"{server_url}/sessions/nonexistent-id-xyz/resume",
            json={},
            headers=client_headers,
        )
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Tests: Termination & Batch Cleanup
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionTerminate:
    """DELETE /sessions/{id} and DELETE /sessions/terminated."""

    @pytest.mark.asyncio
    async def test_terminate_session(
        self, server_url, client_headers, async_client
    ):
        """Terminate sets status to terminated."""
        # Create
        resp = await async_client.post(
            f"{server_url}/sessions",
            json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": 0},
            headers=client_headers,
        )
        if resp.status_code != 200:
            pytest.skip("Cannot create session")
        session_id = resp.json()["session_id"]

        # Terminate
        resp = await async_client.delete(
            f"{server_url}/sessions/{session_id}",
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "terminated"

    @pytest.mark.asyncio
    async def test_terminate_nonexistent(
        self, server_url, client_headers, async_client
    ):
        resp = await async_client.delete(
            f"{server_url}/sessions/nonexistent-xyz",
            headers=client_headers,
        )
        assert resp.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_batch_delete_terminated(
        self, server_url, client_headers, async_client
    ):
        """DELETE /sessions/terminated cleans up all terminated sessions."""
        resp = await async_client.delete(
            f"{server_url}/sessions/terminated",
            headers=client_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "deleted_count" in data
        assert isinstance(data["deleted_count"], int)
        assert "message" in data


# ---------------------------------------------------------------------------
# Tests: Download Preparation
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSessionDownload:
    """POST /sessions/{id}/prepare-download and GET /sessions/{id}/download."""

    @pytest.mark.asyncio
    async def test_prepare_download_returns_info(
        self, server_url, client_headers, async_client, created_session
    ):
        session_id, _ = created_session

        resp = await async_client.post(
            f"{server_url}/sessions/{session_id}/prepare-download",
            headers=client_headers,
        )
        # May return 200 with download info or 400 if S3 not configured
        if resp.status_code == 200:
            data = resp.json()
            assert "session_id" in data
            assert "archive_ready" in data
        elif resp.status_code == 400:
            # S3 not configured is acceptable in test environments
            pass
        else:
            pytest.fail(f"Unexpected status {resp.status_code}: {resp.text}")

    @pytest.mark.asyncio
    async def test_download_without_preparation(
        self, server_url, async_client
    ):
        """GET download without prepare-download returns error or redirect."""
        resp = await async_client.get(
            f"{server_url}/sessions/nonexistent-xyz/download",
            follow_redirects=False,
        )
        # Server may return 400/404/503 (no archive) or 307 (redirect to S3)
        assert resp.status_code in (307, 400, 404, 503)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)
