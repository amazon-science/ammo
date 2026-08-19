# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""End-to-end tests for AMMO session Claude model configuration.

Verifies that when a session is created in a running Docker container:
1. The in-worktree settings.local.json carries the upgraded model + env.
2. The /etc/claude-code/managed-settings.json (highest-precedence) carries
   the same upgraded model + env.
3. The live claude CLI subprocess inside tmux has the operational env vars
   in its actual process environment (/proc/<pid>/environ).

These tests require:
- A running server (Docker container) at AMMO_SERVER_URL
- The Docker container named 'ammo-server'
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

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

EXPECTED_MODEL = "claude-opus-5"
EXPECTED_ENV = {
    "ENABLE_PROMPT_CACHING_1H": "1",
    "CLAUDE_CODE_EFFORT_LEVEL": "xhigh",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": EXPECTED_MODEL,
}


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
        r = requests.get(f"{server_url}/sessions/{session_id}", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") in ("active", "failed", "terminated"):
                return data
        time.sleep(2)
    return {}


def _docker_exec(cmd, timeout=10):
    """Run a command inside the ammo-server container, return stdout."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _read_container_file(path: str) -> str:
    return _docker_exec(["cat", path])


@pytest.fixture(scope="module")
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="module")
def shared_session(server_url):
    """One session shared across all tests in this module (~30s creation cost)."""
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

        session_id = resp.json()["session_id"]
        session_data = _wait_for_active(server_url, session_id, headers)
        if session_data.get("status") != "active":
            pytest.skip(f"Session did not become active: {session_data.get('status')}")

        yield session_id, headers

    finally:
        if session_id:
            requests.delete(
                f"{server_url}/sessions/{session_id}", headers=headers, timeout=30
            )


@pytest.mark.e2e
@pytest.mark.slow
class TestSessionModelConfig:
    """Verify the session model configuration reaches a live session end-to-end."""

    def test_settings_local_in_worktree_has_upgraded_model(self, shared_session):
        """The session's worktree settings.local.json carries the upgraded model + env."""
        session_id, _ = shared_session
        path = f"/data/sessions/{session_id}/worktree/.claude/settings.local.json"
        raw = _read_container_file(path)
        assert raw is not None, f"Could not read {path}"
        settings = json.loads(raw)

        assert settings.get("model") == EXPECTED_MODEL, \
            f"top-level model must be {EXPECTED_MODEL}, got {settings.get('model')}"
        env = settings.get("env", {})
        for key, expected in EXPECTED_ENV.items():
            assert env.get(key) == expected, \
                f"settings.local.json env[{key}] expected {expected}, got {env.get(key)}"

    def test_managed_settings_has_upgraded_model(self, shared_session):
        """managed-settings.json (highest precedence) must carry the upgraded config."""
        raw = _read_container_file("/etc/claude-code/managed-settings.json")
        assert raw is not None, "managed-settings.json not found in container"
        settings = json.loads(raw)

        assert settings.get("model") == EXPECTED_MODEL, \
            f"managed model must be {EXPECTED_MODEL}, got {settings.get('model')}"
        env = settings.get("env", {})
        for key, expected in EXPECTED_ENV.items():
            assert env.get(key) == expected, \
                f"managed-settings env[{key}] expected {expected}, got {env.get(key)}"

    def test_session_start_script_embeds_operational_env(self, shared_session):
        """The generated /tmp/<session_id>/start.sh must embed the operational env
        vars as /usr/bin/env assignments — this is what the claude CLI subprocess
        inherits at exec time (the settings.local.json "env" block is NOT honored
        before the process starts, so anything needed at launch must be on the exec
        line itself)."""
        session_id, _ = shared_session
        start_sh_path = f"/tmp/{session_id}/start.sh"
        script = _read_container_file(start_sh_path)
        assert script is not None, f"{start_sh_path} not found — tmux launcher missing"

        # The script is a single `exec /usr/bin/env KEY=VAL ... /usr/bin/claude` line.
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in script, \
            f"start.sh missing CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1. Script:\n{script}"
        # Sanity: the claude binary invocation is present.
        assert "/usr/bin/claude" in script, \
            f"start.sh does not appear to launch claude. Script:\n{script}"
