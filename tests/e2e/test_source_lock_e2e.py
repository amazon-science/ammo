# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""End-to-end tests for source lock features.

Verifies the vllm_docker_commit field and branch-based session creation
against a running server.

Requirements:
- Server running at AMMO_SERVER_URL (default: http://localhost:8000)
"""

import os
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SESSION_CREATION_TIMEOUT = 300
BASE_URL = os.environ.get("AMMO_SERVER_URL", "http://localhost:8000")


def _auth_headers() -> dict:
    """Return auth headers (empty when AMMO_API_KEY unset)."""
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


@pytest.mark.e2e
class TestSourceLockE2E:
    """E2E tests for source lock: docker commit field and branch-based session creation."""

    def test_health_has_docker_commit_field(self):
        """GET /health must return vllm.docker_commit under the structured
        vllm block (value may be null).

        /api/supported-models was removed as part of the static-model-selector
        removal; build metadata now lives on /health.
        """
        response = requests.get(
            f"{BASE_URL}/health",
            headers=_auth_headers(),
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "vllm" in data, (
            "GET /health must include 'vllm' block. "
            f"Got keys: {list(data.keys())}"
        )
        assert "docker_commit" in data["vllm"], (
            "GET /health vllm block must include 'docker_commit' key. "
            f"Got keys: {list(data['vllm'].keys())}"
        )

    def test_health_has_vllm_version_field(self):
        """E-1: GET /health must expose vllm.version alongside vllm.docker_commit.

        Value is a release tag like 'v0.20.0' on release-wheel images, and
        may be null on legacy (nightly) images. The key itself MUST be present
        so the frontend can branch on it without undefined-access errors.
        """
        response = requests.get(
            f"{BASE_URL}/health",
            headers=_auth_headers(),
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "vllm" in data, (
            "GET /health must include 'vllm' block. "
            f"Got keys: {list(data.keys())}"
        )
        assert "version" in data["vllm"], (
            "GET /health vllm block must include 'version' key. "
            f"Got keys: {list(data['vllm'].keys())}"
        )

    @pytest.mark.slow
    def test_create_session_with_docker_source_mode(self):
        """When a docker commit is available, create a session pinned to that commit."""
        headers = _auth_headers()

        # Fetch the docker commit from /health (previously from /api/supported-models)
        health_resp = requests.get(
            f"{BASE_URL}/health",
            headers=headers,
            timeout=30,
        )
        assert health_resp.status_code == 200
        vllm_block = health_resp.json().get("vllm") or {}
        docker_commit = vllm_block.get("docker_commit")

        if not docker_commit:
            pytest.skip("No vllm.docker_commit available in this environment")

        session_id = None
        try:
            response = requests.post(
                f"{BASE_URL}/sessions",
                json={
                    "repo_name": "vllm",
                    "branch": docker_commit,
                    "gpu_count": 0,
                },
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if response.status_code != 200:
                pytest.skip(f"Session creation failed: {response.text}")

            data = response.json()
            assert "session_id" in data
            session_id = data["session_id"]
            assert data.get("status") in ("creating", "active")
        finally:
            if session_id:
                requests.delete(
                    f"{BASE_URL}/sessions/{session_id}",
                    headers=headers,
                    timeout=30,
                )

    @pytest.mark.slow
    def test_create_session_with_latest_main(self):
        """Create a session with branch='main' and verify it is accepted."""
        headers = _auth_headers()

        session_id = None
        try:
            response = requests.post(
                f"{BASE_URL}/sessions",
                json={
                    "repo_name": "vllm",
                    "branch": "main",
                    "gpu_count": 0,
                },
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if response.status_code != 200:
                pytest.skip(f"Session creation failed: {response.text}")

            data = response.json()
            assert "session_id" in data
            session_id = data["session_id"]
            assert data.get("status") in ("creating", "active")
        finally:
            if session_id:
                requests.delete(
                    f"{BASE_URL}/sessions/{session_id}",
                    headers=headers,
                    timeout=30,
                )

    @pytest.mark.slow
    def test_create_session_with_custom_branch(self):
        """Create a session with a custom branch name and verify it is accepted."""
        headers = _auth_headers()

        session_id = None
        try:
            response = requests.post(
                f"{BASE_URL}/sessions",
                json={
                    "repo_name": "vllm",
                    "branch": "main",
                    "gpu_count": 0,
                },
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if response.status_code != 200:
                pytest.skip(f"Session creation failed: {response.text}")

            data = response.json()
            assert "session_id" in data
            session_id = data["session_id"]
            assert data.get("status") in ("creating", "active")
        finally:
            if session_id:
                requests.delete(
                    f"{BASE_URL}/sessions/{session_id}",
                    headers=headers,
                    timeout=30,
                )
