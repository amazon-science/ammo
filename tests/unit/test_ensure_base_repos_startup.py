# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the ensure_base_repos() startup integration.

Verifies that lifespan() calls WorktreeManager.ensure_base_repos() before
S3 session discovery (so cross-host S3 restore can find the base vLLM repo on a
fresh pod), and that the result is surfaced in the /health response via a
new `base_repos_ready` boolean field. The clone failure must be non-fatal —
the server still starts so eval/profiling traffic is unaffected, but the
degradation is visible to operators.

Plan: .claude/plans/ensure-base-repos-startup.md
"""

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_app_with_fake_manager():
    """Reload app module and inject a fake gpu_manager so /health can run
    without firing the full lifespan."""
    import app as app_module
    importlib.reload(app_module)
    app_module.gpu_type = "unknown"

    gpu_mgr = MagicMock()
    gpu_mgr.get_gpu_count.return_value = 0
    gpu_mgr.get_available_gpu_count.return_value = 0
    app_module.gpu_manager = gpu_mgr
    return app_module


async def _call_health(app_module):
    """Invoke the /health route handler and return decoded JSON + response."""
    response = await app_module.health_check()
    body = json.loads(response.body.decode("utf-8"))
    return body, response


# ---------------------------------------------------------------------------
# Test 1: ensure_base_repos is called during lifespan startup
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureBaseReposCalledOnStartup:
    """The lifespan handler must call session_manager.worktree_manager.ensure_base_repos
    via run_in_executor (sync function inside an async context manager)."""

    def test_lifespan_calls_ensure_base_repos(self):
        """lifespan() source must reference ensure_base_repos."""
        from app import lifespan
        source = inspect.getsource(lifespan)
        assert "ensure_base_repos" in source, (
            "lifespan() must call session_manager.worktree_manager.ensure_base_repos "
            "during startup so the base vLLM repo exists before any cross-host S3 restore."
        )

    def test_lifespan_wraps_ensure_base_repos_in_executor(self):
        """The sync ensure_base_repos call must be off-loaded via run_in_executor.

        Otherwise the synchronous git clone would block the asyncio event loop
        for the entire startup duration (~30s on a cold clone)."""
        from app import lifespan
        source = inspect.getsource(lifespan)
        lines = source.split("\n")
        ebr_lines = [i for i, line in enumerate(lines) if "ensure_base_repos" in line]
        rie_lines = [i for i, line in enumerate(lines) if "run_in_executor" in line]
        assert ebr_lines, "ensure_base_repos must appear in lifespan source"
        assert rie_lines, "run_in_executor must appear in lifespan source"
        found_pair = any(
            abs(ebr - rie) <= 5
            for ebr in ebr_lines
            for rie in rie_lines
        )
        assert found_pair, (
            "ensure_base_repos must be wrapped in run_in_executor (within 5 lines). "
            "Pattern: await loop.run_in_executor(None, session_manager.worktree_manager.ensure_base_repos)"
        )

    def test_lifespan_uses_get_running_loop_not_get_event_loop(self):
        """Use asyncio.get_running_loop() — get_event_loop() is deprecated for async ctx."""
        from app import lifespan
        source = inspect.getsource(lifespan)
        # If lifespan acquires a loop near the ensure_base_repos block, it must use
        # get_running_loop. We accept either an existing get_running_loop or one nearby.
        lines = source.split("\n")
        ebr_lines = [i for i, line in enumerate(lines) if "ensure_base_repos" in line]
        # Look ±10 lines around ensure_base_repos for a loop acquisition
        if not ebr_lines:
            pytest.fail("ensure_base_repos call missing")
        nearby = []
        for ebr in ebr_lines:
            nearby.extend(lines[max(0, ebr - 10):ebr + 5])
        nearby_text = "\n".join(nearby)
        if "get_event_loop" in nearby_text:
            pytest.fail(
                "Use asyncio.get_running_loop() (NOT deprecated get_event_loop()) "
                "around the ensure_base_repos block."
            )


# ---------------------------------------------------------------------------
# Test 2 + 3: /health surfaces base_repos_ready
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureBaseReposHealthSurface:
    """/health must include a 'base_repos_ready' boolean reflecting startup outcome."""

    @pytest.mark.asyncio
    async def test_health_reports_base_repos_ready_true(self):
        """When the module-level _base_repos_ready flag is True, /health echoes it."""
        app_module = _fresh_app_with_fake_manager()
        assert hasattr(app_module, "_base_repos_ready"), (
            "app.py must declare a module-level `_base_repos_ready: bool = False` "
            "(initialized False; set True by lifespan when ensure_base_repos() succeeds)."
        )
        app_module._base_repos_ready = True

        body, response = await _call_health(app_module)
        assert response.status_code == 200
        assert "base_repos_ready" in body, (
            f"/health must include 'base_repos_ready'; got keys: {list(body.keys())}"
        )
        assert body["base_repos_ready"] is True

    @pytest.mark.asyncio
    async def test_health_reports_base_repos_ready_false(self):
        """When _base_repos_ready=False, /health surfaces it so operators see the
        degradation. The pod stays Ready (HTTP 200) so eval traffic flows."""
        app_module = _fresh_app_with_fake_manager()
        assert hasattr(app_module, "_base_repos_ready")
        app_module._base_repos_ready = False

        body, response = await _call_health(app_module)
        assert response.status_code == 200
        assert "base_repos_ready" in body
        assert body["base_repos_ready"] is False, (
            "When ensure_base_repos() returns {'vllm': False}, /health must show "
            "'base_repos_ready': false so the cluster dashboard surfaces the failure."
        )


# ---------------------------------------------------------------------------
# Test 4: ensure_base_repos exception is non-fatal
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureBaseReposExceptionIsNonFatal:
    """A clone failure must NOT prevent the server from starting (eval/profiling
    traffic doesn't require the base repo). The failure surfaces only via /health."""

    def test_lifespan_wraps_ensure_base_repos_in_try_except(self):
        """The ensure_base_repos call must live inside a try/except so an
        exception (e.g. transient network failure) doesn't abort startup."""
        from app import lifespan
        source = inspect.getsource(lifespan)
        lines = source.split("\n")
        ebr_idx = None
        for i, line in enumerate(lines):
            if "ensure_base_repos" in line and "def " not in line:
                ebr_idx = i
                break
        assert ebr_idx is not None, "ensure_base_repos call missing from lifespan"
        # Look backwards for `try:` (within 8 lines)
        try_found = any(
            lines[j].strip().startswith("try:")
            for j in range(max(0, ebr_idx - 8), ebr_idx)
        )
        assert try_found, (
            "The ensure_base_repos block must be guarded by `try:` so transient "
            "clone failures don't prevent server startup."
        )
        # Look forward for `except` (within 30 lines)
        except_found = any(
            lines[j].lstrip().startswith("except")
            for j in range(ebr_idx, min(len(lines), ebr_idx + 30))
        )
        assert except_found, (
            "The ensure_base_repos block must have an except clause that "
            "logs the failure and sets _base_repos_ready=False."
        )


# ---------------------------------------------------------------------------
# Test 5: ensure_base_repos runs BEFORE discover_s3_sessions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureBaseReposCallOrder:
    """Cross-pod resume calls repair_worktree_linkage() which is a no-op when the
    base repo doesn't exist. So we must clone before resume can fire — which means
    BEFORE discover_s3_sessions (it can synchronously trigger session restore)."""

    def test_ensure_base_repos_runs_before_s3_discovery(self):
        """Compare the FIRST executable references to each — ignoring comment
        lines so a forward reference in a docstring doesn't fool the check."""
        from app import lifespan
        source = inspect.getsource(lifespan)

        def _first_call_line(needle: str) -> int:
            for i, line in enumerate(source.split("\n")):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if needle in stripped:
                    return i
            return -1

        ebr_line = _first_call_line("ensure_base_repos")
        s3_line = _first_call_line("discover_s3_sessions")
        assert ebr_line != -1, "ensure_base_repos call missing in lifespan"
        assert s3_line != -1, "discover_s3_sessions call missing in lifespan (sanity)"
        assert ebr_line < s3_line, (
            f"ensure_base_repos (line {ebr_line}) must come BEFORE "
            f"discover_s3_sessions (line {s3_line}) in lifespan() — otherwise "
            f"S3-restored sessions would race against an unborn base repo."
        )


# ---------------------------------------------------------------------------
# Test 6: idempotent on existing repo (real WorktreeManager + tmp_path)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnsureBaseReposIdempotent:
    """A pre-existing cloned repo must return success WITHOUT re-cloning."""

    def test_ensure_base_repos_idempotent_on_existing_repo(self, tmp_path, monkeypatch):
        repos_dir = tmp_path / "repos"
        sessions_dir = tmp_path / "sessions"
        repos_dir.mkdir()
        sessions_dir.mkdir()

        # Pre-create a real (non-bare) git repo at repos_dir/vllm/.git so
        # WorktreeManager.is_repo_cloned("vllm") returns True.
        repo_path = repos_dir / "vllm"
        repo_path.mkdir()
        result = subprocess.run(
            ["git", "init"], cwd=repo_path,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"git init failed: {result.stderr}"
        assert (repo_path / ".git").exists(), "Test fixture setup failed: .git missing"

        from orchestration.worktree_manager import WorktreeManager
        wm = WorktreeManager(
            repos_dir=str(repos_dir),
            sessions_dir=str(sessions_dir),
        )

        # Skip the chown migration (it tries to setuid; not relevant to idempotency).
        monkeypatch.setattr(wm, "_ensure_repo_owned_by_session_user", lambda p: None)

        # Track _run_git calls — the idempotent path must NOT invoke git.
        git_calls = []
        original_run_git = wm._run_git

        def tracked_run_git(args, **kwargs):
            git_calls.append(args)
            return original_run_git(args, **kwargs)

        monkeypatch.setattr(wm, "_run_git", tracked_run_git)

        results = wm.ensure_base_repos()

        assert results == {"vllm": True}, (
            f"ensure_base_repos() must return {{'vllm': True}} for an already-cloned "
            f"repo (idempotent early-return); got {results}"
        )
        assert git_calls == [], (
            f"ensure_base_repos() should be a no-op when the repo is already cloned; "
            f"unexpected git invocations: {git_calls}"
        )
