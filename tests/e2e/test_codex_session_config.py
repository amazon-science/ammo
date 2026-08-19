# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""End-to-end test for the AMMO Codex session user config.

Requires:
- A running server at AMMO_SERVER_URL (default: http://localhost:8000)
- A Docker container named ammo-server
"""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import sys
import tomllib

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _auth_headers() -> dict:
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


DEFAULT_SERVER_URL = "http://localhost:8000"
SESSION_CREATION_TIMEOUT = 300
ACTIVE_WAIT_TIMEOUT = 120
CONTAINER_NAME = "ammo-server"


def _client_headers() -> dict:
    return {"X-Client-ID": str(uuid.uuid4()), **_auth_headers()}


def _server_is_reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _wait_for_active(server_url, session_id, headers, timeout=ACTIVE_WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") in ("active", "failed", "terminated"):
                return data
        time.sleep(2)
    return {}


def _docker_exec(cmd, timeout=10):
    if shutil.which("docker") is None:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return result.stdout

    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return None
    return result.stdout


@pytest.fixture(scope="module")
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="module")
def codex_session(server_url):
    if not _server_is_reachable(server_url):
        pytest.skip("Server not reachable")

    headers = _client_headers()
    session_id = None

    try:
        response = requests.post(
            f"{server_url}/sessions",
            json={
                "repo_name": "vllm",
                "cli_tool": "codex",
                "gpu_count": 0,
                "inactivity_timeout_mins": 60,
            },
            headers=headers,
            timeout=SESSION_CREATION_TIMEOUT,
        )
        if response.status_code != 200:
            pytest.skip(f"Session creation failed ({response.status_code}): {response.text}")

        session_id = response.json()["session_id"]
        session_data = _wait_for_active(server_url, session_id, headers)
        if session_data.get("status") != "active":
            pytest.skip(f"Session did not become active: {session_data.get('status')}")

        yield session_id, headers

    finally:
        if session_id:
            requests.delete(f"{server_url}/sessions/{session_id}", headers=headers, timeout=30)


@pytest.mark.e2e
@pytest.mark.slow
def test_codex_session_user_config_uses_default_provider(codex_session):
    session_id, _ = codex_session
    path = f"/data/sessions/{session_id}/codex-home/config.toml"
    raw = _docker_exec(["cat", path])
    assert raw is not None, f"Could not read {path}"

    config = tomllib.loads(raw)
    assert config["model"] == "gpt-5.6-sol"
    # No alternate model provider may be active — the generated config must use
    # the default provider so a session never depends on external routing setup.
    assert "model_provider" not in config
    assert "model_providers" not in config
    assert "model_provider" not in raw
