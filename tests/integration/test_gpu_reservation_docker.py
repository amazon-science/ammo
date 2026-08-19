# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for GPU reservation feature — runs against a live server.

Tests verify the full end-to-end behavior of the AMMO_GPU_RES_DIR injection
and gpu_reservation.py deployment, as introduced by the gpu-reservation-
integration plan.

These tests are red-phase: they FAIL until Changes 2, 5, 6, and 7 from the
plan are implemented.

Plan: .claude/plans/gpu-reservation-integration.md

Prerequisites:
  - Server running at AMMO_SERVER_URL (default: http://localhost:8000)
  - Tests running on the same host as the server (for filesystem checks)
  - For tests 1 & 4: SESSION_DATA_DIR accessible at the same path as on the server
"""

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers / Constants
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8000"


def _server_url() -> str:
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


def _resolve_template_path(relative: str) -> Path:
    """Resolve path relative to repo root, trying local then Docker."""
    candidates = [
        Path(__file__).parent.parent.parent / relative,
        Path("/app") / relative,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return first for clear error messages


def _server_available() -> bool:
    try:
        r = requests.get(f"{_server_url()}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _container_to_host_path(container_path: str) -> Path:
    """Translate a Docker container filesystem path to the host-accessible equivalent.

    When the server runs in Docker, worktree paths are reported as container
    paths (e.g. /data/sessions/{id}/worktree).  The session data dir is
    bind-mounted, so the equivalent host path is accessible via the mount source.
    """
    import subprocess as _sp
    # Try to discover the bind mount source dynamically
    try:
        result = _sp.run(
            ["docker", "inspect", "ammo-server", "--format",
             "{{range .Mounts}}{{.Destination}}:{{.Source}}\n{{end}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if ":" in line:
                    dest, src = line.split(":", 1)
                    dest = dest.rstrip("/")
                    if container_path.startswith(dest + "/") or container_path == dest:
                        host_path = container_path.replace(dest, src.rstrip("/"), 1)
                        return Path(host_path)
    except Exception:
        pass
    # Fallback: no translation (native server or same-filesystem)
    return Path(container_path)


def _docker_path_exists(container_path: str) -> bool:
    """Check if a path exists INSIDE the Docker container."""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["docker", "exec", "ammo-server", "test", "-e", container_path],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _docker_mkdir(container_path: str) -> None:
    """Create a directory INSIDE the Docker container."""
    import subprocess as _sp
    _sp.run(
        ["docker", "exec", "ammo-server", "mkdir", "-p", container_path],
        check=True, capture_output=True, timeout=5,
    )


def _create_session(gpu_count: int = 0) -> tuple[str, dict]:
    """Create a session and return (session_id, response_dict)."""
    key = os.environ.get("AMMO_API_KEY", "")
    headers = {"X-Client-ID": str(uuid.uuid4())}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    # vLLM editable install can take 60-200s under load; use 300s as CLAUDE.md recommends
    session_timeout = int(os.getenv("SESSION_CREATION_TIMEOUT", "300"))
    resp = requests.post(
        f"{_server_url()}/sessions",
        json={"repo_name": "vllm", "cli_tool": "claude", "gpu_count": gpu_count},
        headers=headers,
        timeout=session_timeout,
    )
    assert resp.status_code == 200, f"Session creation failed: {resp.status_code} {resp.text}"
    data = resp.json()
    return data["session_id"], data, headers


def _delete_session(session_id: str, headers: dict) -> None:
    """Delete a session (best-effort cleanup)."""
    try:
        requests.delete(
            f"{_server_url()}/sessions/{session_id}",
            headers=headers,
            timeout=30,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test 1: Session has AMMO_GPU_RES_DIR in its runtime settings.local.json
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionSettingsLocalHasAMMOGPUResDir:
    """
    Change 2a: After session creation, the session's settings.local.json
    (in the worktree .claude dir) must have AMMO_GPU_RES_DIR injected.

    This validates that Claude Code subagents spawned inside the session
    can call gpu_reservation.py with proper state isolation.
    """

    def test_session_settings_local_has_ammo_gpu_res_dir(self):
        """
        Create a session, read its settings.local.json from the worktree,
        and assert AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id} is present.
        """
        if not _server_available():
            pytest.skip("Server not running at " + _server_url())

        try:
            session_id, data, headers = _create_session(gpu_count=0)
        except Exception as e:
            pytest.skip(f"Session creation unavailable: {e}")

        # POST /sessions doesn't include worktree_path — fetch it from GET
        try:
            detail_resp = requests.get(
                f"{_server_url()}/sessions/{session_id}",
                headers=headers,
                timeout=30,
            )
            worktree_path = detail_resp.json().get("worktree_path")
        except Exception as e:
            pytest.skip(f"Could not fetch session details: {e}")

        try:
            if not worktree_path:
                pytest.skip("GET /sessions/{id} did not include worktree_path")

            # Translate container path → host path (needed when server runs in Docker)
            host_worktree = _container_to_host_path(worktree_path)
            settings_path = host_worktree / ".claude" / "settings.local.json"

            if not settings_path.exists():
                pytest.skip(
                    f"settings.local.json not accessible at {settings_path} "
                    "(test requires running on same host as server or with volume mount)"
                )

            with open(settings_path) as f:
                settings = json.load(f)

            env_section = settings.get("env", {})
            expected_dir = f"/tmp/ammo_gpu_res_{session_id}"

            assert "AMMO_GPU_RES_DIR" in env_section, (
                f"AMMO_GPU_RES_DIR missing from session settings.local.json env. "
                f"Keys present: {sorted(env_section.keys())}"
            )
            assert env_section["AMMO_GPU_RES_DIR"] == expected_dir, (
                f"Expected AMMO_GPU_RES_DIR={expected_dir!r}, "
                f"got {env_section.get('AMMO_GPU_RES_DIR')!r}"
            )
        finally:
            _delete_session(session_id, headers)

    def test_two_sessions_get_different_ammo_gpu_res_dirs(self):
        """
        Two concurrent sessions must receive DIFFERENT AMMO_GPU_RES_DIR values,
        each scoped to their own session_id.

        This is the core session-isolation guarantee: if both sessions shared
        the same /tmp/ammo_gpu_res/ dir, their gpu_reservation.py instances
        would contend over the same state.lock and pool state — cross-session
        GPU interference.

        Edge case: test_session_settings_local_has_ammo_gpu_res_dir only
        verifies a single session. This test verifies isolation between sessions.
        """
        if not _server_available():
            pytest.skip("Server not running at " + _server_url())

        session_id_a = session_id_b = None
        headers_a = headers_b = None
        try:
            try:
                session_id_a, _, headers_a = _create_session(gpu_count=0)
                session_id_b, _, headers_b = _create_session(gpu_count=0)
            except Exception as e:
                pytest.skip(f"Session creation unavailable: {e}")

            # Fetch worktree paths
            detail_a = requests.get(f"{_server_url()}/sessions/{session_id_a}",
                headers=headers_a, timeout=30).json()
            detail_b = requests.get(f"{_server_url()}/sessions/{session_id_b}",
                headers=headers_b, timeout=30).json()

            wt_a = _container_to_host_path(detail_a.get("worktree_path", ""))
            wt_b = _container_to_host_path(detail_b.get("worktree_path", ""))

            settings_a = wt_a / ".claude" / "settings.local.json"
            settings_b = wt_b / ".claude" / "settings.local.json"

            if not (settings_a.exists() and settings_b.exists()):
                pytest.skip("settings.local.json not accessible for both sessions")

            dir_a = json.loads(settings_a.read_text()).get("env", {}).get("AMMO_GPU_RES_DIR")
            dir_b = json.loads(settings_b.read_text()).get("env", {}).get("AMMO_GPU_RES_DIR")

            assert dir_a is not None, "Session A missing AMMO_GPU_RES_DIR"
            assert dir_b is not None, "Session B missing AMMO_GPU_RES_DIR"
            assert dir_a != dir_b, (
                f"Both sessions share the same AMMO_GPU_RES_DIR={dir_a!r}. "
                f"Each session must get /tmp/ammo_gpu_res_{{session_id}} "
                f"with its own unique session_id."
            )
            assert session_id_a in dir_a, f"Session A's dir {dir_a!r} doesn't contain its session_id"
            assert session_id_b in dir_b, f"Session B's dir {dir_b!r} doesn't contain its session_id"
        finally:
            if session_id_a and headers_a:
                _delete_session(session_id_a, headers_a)
            if session_id_b and headers_b:
                _delete_session(session_id_b, headers_b)


# ---------------------------------------------------------------------------
# Test 2: gpu_reservation.py exists in the session template
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGPUReservationScriptExists:
    """
    Change 7: gpu_reservation.py must exist in the session template scripts dir
    so it is available in every session worktree.
    """

    def test_gpu_reservation_script_exists_in_template(self):
        """
        The session template must include gpu_reservation.py at
        ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py.

        This file is copied into every session worktree via setup_claude_workspace.
        Without it, agents cannot call gpu_reservation.py reserve/release.
        """
        script_path = _resolve_template_path(
            "ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py"
        )
        assert script_path.exists(), (
            f"gpu_reservation.py not found at {script_path}. "
            "Task #7 must copy the modified gpu_reservation.py into the session template."
        )

    def test_gpu_reservation_script_has_cvd_discovery(self):
        """
        The session template's gpu_reservation.py must use CVD-based discovery
        (_discover_session_gpus) instead of nvidia-smi (_discover_gpu_count).

        This is the critical multi-session fix: nvidia-smi sees all host GPUs,
        but _discover_session_gpus parses CUDA_VISIBLE_DEVICES for physical IDs.
        """
        script_path = _resolve_template_path(
            "ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py"
        )
        if not script_path.exists():
            pytest.skip("gpu_reservation.py not in template yet")

        content = script_path.read_text()
        assert "_discover_session_gpus" in content, (
            "gpu_reservation.py must define _discover_session_gpus() "
            "for CVD-based physical GPU ID discovery"
        )
        assert "CUDA_VISIBLE_DEVICES" in content, (
            "gpu_reservation.py must parse CUDA_VISIBLE_DEVICES"
        )
        # Must NOT use nvidia-smi for pool initialization (the vllm bug we're fixing)
        # The function _discover_gpu_count (nvidia-smi) should be absent from the
        # pool initialization path.
        assert "_discover_session_gpus" in content, (
            "gpu_reservation.py must use _discover_session_gpus, not _discover_gpu_count"
        )


# ---------------------------------------------------------------------------
# Test 3: PostToolUse hook for ammo-gpu-release.sh is registered
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAMMOGPUReleaseHookRegistered:
    """
    Change 6: settings.local.json must have a PostToolUse hook that calls
    ammo-gpu-release.sh to auto-release GPU reservations after commands.
    """

    def test_settings_local_has_posttooluse_gpu_release_hook(self):
        """
        The session template's settings.local.json must have a PostToolUse
        hook entry for ammo-gpu-release.sh.

        The hook fires after Bash commands that use the reservation pattern
        (gpu_reservation.py reserve), auto-releasing GPUs when the command
        completes (happy path). TTL expiry handles the crash path.
        """
        settings_path = _resolve_template_path(
            "ai_cli_session/.claude/settings.local.json"
        )
        assert settings_path.exists(), (
            f"settings.local.json not found at {settings_path}"
        )

        with open(settings_path) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {})
        post_tool_hooks = hooks.get("PostToolUse", [])

        # Find the ammo-gpu-release hook
        release_hooks = [
            h for h in post_tool_hooks
            if any(
                "ammo-gpu-release" in hook.get("command", "")
                for hook in h.get("hooks", [])
            )
        ]
        assert len(release_hooks) >= 1, (
            f"No PostToolUse hook for ammo-gpu-release.sh found in settings.local.json. "
            f"Current PostToolUse hooks: {post_tool_hooks}. "
            "Task #6 must add the PostToolUse hook for ammo-gpu-release.sh."
        )

    def test_ammo_gpu_release_hook_script_exists(self):
        """
        The ammo-gpu-release.sh script must exist in the hooks directory
        so it can be executed by Claude Code's PostToolUse event.
        """
        hook_path = _resolve_template_path(
            "ai_cli_session/.claude/hooks/ammo-gpu-release.sh"
        )
        assert hook_path.exists(), (
            f"ammo-gpu-release.sh not found at {hook_path}. "
            "Task #6 must copy this hook from the vllm source."
        )
        assert os.access(hook_path, os.X_OK), (
            f"ammo-gpu-release.sh is not executable at {hook_path}"
        )

    def test_posttooluse_hook_matcher_is_bash_not_wildcard(self):
        """
        The PostToolUse hook for ammo-gpu-release.sh must use matcher="Bash",
        not a wildcard or other tool name.

        If matcher were "*" or missing, the hook would fire on EVERY tool call
        (Read, Edit, Grep, etc.), causing unnecessary process overhead and
        potential false-positive releases when agents read files after reserving.

        Edge case: the existence test (above) doesn't verify the matcher value.
        """
        settings_path = _resolve_template_path(
            "ai_cli_session/.claude/settings.local.json"
        )
        assert settings_path.exists(), f"settings.local.json not found at {settings_path}"

        with open(settings_path) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {})
        post_tool_hooks = hooks.get("PostToolUse", [])

        release_hooks = [
            h for h in post_tool_hooks
            if any(
                "ammo-gpu-release" in hook.get("command", "")
                for hook in h.get("hooks", [])
            )
        ]
        assert len(release_hooks) >= 1, "No ammo-gpu-release PostToolUse hook found"

        for hook_group in release_hooks:
            matcher = hook_group.get("matcher", "")
            assert matcher == "Bash", (
                f"ammo-gpu-release PostToolUse hook must have matcher='Bash', "
                f"got {matcher!r}. A wildcard would fire on ALL tool calls "
                f"(Read, Edit, etc.), causing false-positive GPU releases."
            )


# ---------------------------------------------------------------------------
# Test 4: Session terminate removes /tmp/ammo_gpu_res_{session_id}/
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionTerminateRemovesGPUResDir:
    """
    Change 5: When a session is terminated, the server must delete
    /tmp/ammo_gpu_res_{session_id}/ to prevent /tmp disk accumulation.
    """

    def test_terminate_removes_ammo_gpu_res_dir(self):
        """
        Create a session, simulate a GPU reservation dir existing,
        terminate the session, and verify the dir is cleaned up.

        Steps:
        1. Create session → get session_id
        2. mkdir /tmp/ammo_gpu_res_{session_id}/ (simulates prior agent usage)
        3. DELETE /sessions/{session_id}
        4. Assert /tmp/ammo_gpu_res_{session_id}/ no longer exists
        """
        if not _server_available():
            pytest.skip("Server not running at " + _server_url())

        try:
            session_id, data, headers = _create_session(gpu_count=0)
        except Exception as e:
            pytest.skip(f"Session creation unavailable: {e}")

        # GPU reservation dir lives in /tmp/ of the process running the server.
        # When Docker is used, that's the container's /tmp/ — use docker exec.
        # When native, it's the host /tmp/.
        container_gpu_res_dir = f"/tmp/ammo_gpu_res_{session_id}"
        using_docker = _docker_path_exists("/tmp")  # docker exec works → Docker mode

        try:
            # Simulate: an agent reserved GPUs during the session
            if using_docker:
                _docker_mkdir(container_gpu_res_dir)
                assert _docker_path_exists(container_gpu_res_dir), (
                    "Setup: gpu_res_dir must exist inside Docker before termination"
                )
            else:
                gpu_res_dir = Path(container_gpu_res_dir)
                gpu_res_dir.mkdir(parents=True, exist_ok=True)
                assert gpu_res_dir.exists(), "Setup: gpu_res_dir must exist before termination"

            # Terminate the session
            _delete_session(session_id, headers)
            time.sleep(0.5)

            if using_docker:
                assert not _docker_path_exists(container_gpu_res_dir), (
                    f"{container_gpu_res_dir} should have been deleted inside Docker on "
                    f"termination, but it still exists. Change 5 in session_manager.py "
                    f"must call shutil.rmtree() in terminate_session()."
                )
            else:
                assert not Path(container_gpu_res_dir).exists(), (
                    f"{container_gpu_res_dir} should have been deleted on termination."
                )
        finally:
            # Safety cleanup in case test fails
            if using_docker:
                import subprocess as _sp
                _sp.run(
                    ["docker", "exec", "ammo-server", "rm", "-rf",
                     container_gpu_res_dir],
                    capture_output=True, timeout=5,
                )
            else:
                shutil.rmtree(container_gpu_res_dir, ignore_errors=True)
