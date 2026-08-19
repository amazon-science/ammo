# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for simplified vLLM environment initialization.

Tests 1-8: unit tests for the new initialize_vllm_environment() implementation
           (written by Track 1 implementor).

Tests 9-11 (TestDockerfileGuards): replanted from deleted
           test_vllm_precompiled_wheel.py — guard the Dockerfile's server venv
           block (lines 117-156) which MUST stay correct, or FastAPI won't start.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Ensure tests/ root is on sys.path (for fixture imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project root resolved from this file's location (tests/unit/ → root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures from session_fixtures (needed by test 5)
# ---------------------------------------------------------------------------
from fixtures.session_fixtures import (  # noqa: E402
    reset_all_singletons,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
    mock_session_manager,
)


# ---------------------------------------------------------------------------
# Helpers shared by Tests 1-8
# ---------------------------------------------------------------------------

def _make_manager(tmp_path):
    """Create a WorktreeManager with isolated tmp dirs."""
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager
    reset_worktree_manager()
    return WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )


def _success_proc():
    """Return an AsyncMock subprocess that returns returncode=0."""
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.kill = MagicMock()
    return proc


def _make_capturing_exec(captured: list):
    """Return an async side_effect for create_subprocess_exec that records calls."""
    async def _exec(*args, **kwargs):
        captured.append((args, kwargs))
        return _success_proc()
    return _exec


# ===========================================================================
# Test 1: Happy path — uv venv + precompiled editable install
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_simplified_vllm_env_init(tmp_path):
    """initialize_vllm_environment() flow:

    1. Calls 'uv venv --python 3.12 <dst_venv>'
    2. Calls 'uv pip install -e . --torch-backend auto --python <python>'
    3. env includes VLLM_USE_PRECOMPILED=1 and VLLM_PRECOMPILED_WHEEL_COMMIT
       (default branch "main" pins to the baked .docker_commit so setup.py
       fetches a wheel that is guaranteed to exist — avoids the nightly
       publish race that 404s and leaves vLLM uninstalled)
    4. --no-build-isolation is absent from the install command
    5. --extra-index-url and --index-strategy are absent (replaced by --torch-backend auto)
    6. returns {"status": "success", "timings": {venv_create, editable_install, total}}
    """
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-init-001"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)
    dst_venv = worktree_path / ".venv"

    # Simulate the image's build-verified baked commit so the default-branch
    # ("main") path pins to it (mirrors /workspace/vllm/.docker_commit in prod).
    baked_commit = "a" * 40
    base_repo = tmp_path / "base_vllm"
    base_repo.mkdir(parents=True, exist_ok=True)
    (base_repo / ".docker_commit").write_text(baked_commit + "\n")

    captured: list = []

    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
         patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}), \
         patch("shutil.copy2"):
        result = await mgr.initialize_vllm_environment(session_id=session_id, branch="main")

    # Extract only calls whose first arg is the uv binary
    uv_calls = [(args, kwargs) for args, kwargs in captured if args and args[0] == "/usr/bin/uv"]

    assert len(uv_calls) >= 2, (
        f"Expected at least 2 uv subprocess calls (uv venv + uv pip install), "
        f"got {len(uv_calls)} uv calls out of {len(captured)} total. calls={captured}"
    )

    # --- First call: uv venv --python 3.12 <dst_venv> ---
    first_args = uv_calls[0][0]
    assert "venv" in first_args, f"First uv call should be 'uv venv', got {first_args}"
    assert "--python" in first_args, f"First call missing --python, got {first_args}"
    assert "3.12" in first_args, f"First call missing '3.12', got {first_args}"
    assert str(dst_venv) in first_args, (
        f"First call should include dst_venv path '{dst_venv}', got {first_args}"
    )

    # --- Second call: uv pip install -e . with correct flags ---
    second_args, second_kwargs = uv_calls[1]
    assert "pip" in second_args, f"Second call should be 'uv pip', got {second_args}"
    assert "install" in second_args, f"Second call missing 'install', got {second_args}"
    assert "-e" in second_args, f"Second call missing -e flag, got {second_args}"
    assert "." in second_args, f"Second call missing '.' install target, got {second_args}"
    assert "--torch-backend" in second_args, (
        f"Second call missing --torch-backend, got {second_args}"
    )
    assert "auto" in second_args, (
        f"Second call missing 'auto' (for --torch-backend auto), got {second_args}"
    )
    # Old flags must NOT be present
    assert "--extra-index-url" not in second_args, (
        f"--extra-index-url must NOT be present (replaced by --torch-backend auto). "
        f"Got: {second_args}"
    )
    assert "--index-strategy" not in second_args, (
        f"--index-strategy must NOT be present (only needed with --extra-index-url). "
        f"Got: {second_args}"
    )
    assert "--python" in second_args, (
        f"Second call missing --python flag, got {second_args}"
    )

    # --no-build-isolation must NOT appear
    assert "--no-build-isolation" not in second_args, (
        f"--no-build-isolation must NOT be present in the install command. "
        f"uv's build isolation is required for a fresh venv. Got: {second_args}"
    )

    # Environment variables
    env = second_kwargs.get("env", {})
    assert env.get("VLLM_USE_PRECOMPILED") == "1", (
        f"VLLM_USE_PRECOMPILED must be '1', got {env.get('VLLM_USE_PRECOMPILED')!r}"
    )
    # branch="main" is the default branch, so VLLM_PRECOMPILED_WHEEL_COMMIT
    # must pin to the baked .docker_commit (not "") — this is what guarantees a
    # published wheel and prevents the nightly-publish-race 404 that blocks the
    # session vLLM install. setup.py self-resolution is NOT used for "main".
    assert env.get("VLLM_PRECOMPILED_WHEEL_COMMIT") == baked_commit, (
        f"VLLM_PRECOMPILED_WHEEL_COMMIT must pin to the baked commit "
        f"{baked_commit!r} for the default branch, "
        f"got {env.get('VLLM_PRECOMPILED_WHEEL_COMMIT')!r}."
    )

    # Return value
    assert isinstance(result, dict), f"Expected dict return, got {type(result)}"
    assert result.get("status") == "success", f"Expected status='success', got {result}"
    assert "timings" in result, f"Expected 'timings' key in result, got {result}"
    timings = result["timings"]
    assert "venv_create" in timings, f"timings must have 'venv_create', got {timings}"
    assert "editable_install" in timings, f"timings must have 'editable_install', got {timings}"
    assert "total" in timings, f"timings must have 'total', got {timings}"
    for key in ("venv_create", "editable_install", "total"):
        assert isinstance(timings[key], (int, float)), (
            f"timings['{key}'] must be a number, got {type(timings[key])}: {timings[key]}"
        )


# ===========================================================================
# Test 2: uv not found → WorktreeError
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_uv_not_found_raises_worktree_error(tmp_path):
    """When uv is not on PATH, raise WorktreeError with 'uv not found'.

    RED: current code does `shutil.which("uv") or "uv"`, which falls through
    to FileNotFoundError from the OS, not WorktreeError.
    """
    from orchestration.worktree_manager import WorktreeManager, WorktreeError, reset_worktree_manager

    session_id = "test-uv-missing-001"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)

    with patch("shutil.which", return_value=None):
        with pytest.raises(WorktreeError, match="uv not found"):
            await mgr.initialize_vllm_environment(session_id=session_id)


# ===========================================================================
# Test 3: Editable install timeout → WorktreeError + proc.kill()
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_install_timeout_raises_worktree_error(tmp_path):
    """When uv pip install exceeds the timeout, raise WorktreeError and kill proc.

    RED: current code has no asyncio.wait_for timeout; it never raises WorktreeError
    on a hung install. The test fails because WorktreeError is not raised.
    """
    from orchestration.worktree_manager import WorktreeManager, WorktreeError, reset_worktree_manager

    session_id = "test-timeout-001"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)

    procs: list = []

    async def mock_exec(*args, **kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.kill = MagicMock()
        procs.append(proc)
        return proc

    # asyncio.wait_for raises TimeoutError — simulates the 240s install timeout
    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
         patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        with pytest.raises(WorktreeError, match="timed out"):
            await mgr.initialize_vllm_environment(session_id=session_id)

    # The install proc must have been killed
    assert len(procs) >= 2, (
        f"Expected at least 2 subprocesses (uv venv + uv pip install), got {len(procs)}"
    )
    # The second proc (editable install) must have been killed
    procs[1].kill.assert_called_once()


# ===========================================================================
# Test 4: Return format matches session_manager contract
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_return_format_matches_session_manager(tmp_path):
    """Return dict must have keys read by session_manager.py:535 and :976.

    session_manager reads env_result.get("timings") and expects:
      timings["venv_create"], timings["editable_install"], timings["total"]

    RED: current impl returns timings with keys like "venv_hardlink", "shebang_fixes",
    "activate_fixes", "venv_cache_check" — none of which are venv_create, editable_install,
    or total as numeric floats.
    """
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-format-001"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)

    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec([])), \
         patch("shutil.copy2"):
        result = await mgr.initialize_vllm_environment(session_id=session_id)

    # Top-level shape
    assert result.get("status") == "success", f"status must be 'success', got {result}"
    assert isinstance(result.get("timings"), dict), (
        f"'timings' must be a dict, got {type(result.get('timings'))}"
    )
    timings = result["timings"]

    # Required keys (matches session_manager.py contract)
    for key in ("venv_create", "editable_install", "total"):
        assert key in timings, (
            f"timings must contain '{key}' (read by session_manager.py). "
            f"Got keys: {list(timings.keys())}"
        )
        assert isinstance(timings[key], (int, float)), (
            f"timings['{key}'] must be numeric, got {type(timings[key])}"
        )

    # Legacy keys from the old implementation must NOT be present
    old_keys = ("venv_hardlink", "shebang_fixes", "activate_fixes", "venv_cache_check",
                "venv_fresh_create")
    present_old = [k for k in old_keys if k in timings]
    assert not present_old, (
        f"Old timing keys still present: {present_old}. "
        "These were from the deleted hardlink/shebang pipeline."
    )


# ===========================================================================
# Test 5: Cross-pod resume calls initialize_vllm_environment
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_pod_resume_reinitializes(
    mock_session_manager,
    mock_session_storage,
    mock_worktree_manager,
    tmp_path,
):
    """resume_session with restored_from_s3=True and repo_name='vllm' must call
    initialize_vllm_environment.

    PASSES ALREADY — this is a locking test for existing behavior at
    session_manager.py:968-976. Ensures the rewrite does not accidentally remove
    the re-init call on cross-host S3 restore.
    """
    from shared.session_models import SessionState, SessionStatus, CLIToolType

    session_id = "test-cross-pod-001"
    session_dir = Path(mock_session_manager.sessions_dir) / session_id
    worktree_path = session_dir / "worktree"
    worktree_path.mkdir(parents=True, exist_ok=True)

    restored_state = SessionState(
        session_id=session_id,
        status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=str(worktree_path),
        session_dir=str(session_dir),
        requested_gpu_count=0,
    )

    async def mock_restore(*args, **kwargs):
        worktree_path.mkdir(parents=True, exist_ok=True)
        return restored_state

    mock_session_storage.enabled = True
    mock_session_storage.restore_session_from_s3 = AsyncMock(side_effect=mock_restore)

    # Session not in local memory → triggers S3 restore → restored_from_s3=True
    assert session_id not in mock_session_manager._sessions

    await mock_session_manager.resume_session(session_id)

    # initialize_vllm_environment must have been called
    mock_worktree_manager.initialize_vllm_environment.assert_called_once_with(
        session_id=session_id,
        branch="main",
    )


# ===========================================================================
# Test 6: No branch routing — main and feature-x use identical subprocess calls
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_branch_routing(tmp_path):
    """initialize_vllm_environment() must use identical subprocess commands for
    all branches — no 'if branch == "main"' fast path.

    RED: current code routes 'main' through cached-venv/hardlink path and non-main
    through _create_fresh_venv_from_requirements — very different subprocess calls.
    For branch='main' in a test env (no /workspace/vllm/.venv), zero subprocess calls.
    For branch='feature-x', three uv subprocess calls from the fresh-venv path.
    They differ → assert fails.
    """
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-branch-001"

    async def run_for_branch(branch: str) -> list:
        """Run initialize_vllm_environment for the given branch, return captured calls."""
        reset_worktree_manager()
        # Same sessions_dir for both runs so worktree paths are identical —
        # allows direct tuple comparison without path normalization.
        mgr = WorktreeManager(
            repos_dir=str(tmp_path / "repos"),
            sessions_dir=str(tmp_path / "sessions"),
        )
        wt = mgr.get_worktree_path(session_id)
        wt.mkdir(parents=True, exist_ok=True)

        captured: list = []
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
             patch("shutil.copy2"):
            await mgr.initialize_vllm_environment(session_id=session_id, branch=branch)
        return captured

    calls_main = await run_for_branch("main")
    calls_feature = await run_for_branch("feature-x")

    # Extract only uv command-arg tuples (normalized: just the positional args)
    def uv_args_only(calls: list) -> list:
        return [args for args, _kwargs in calls if args and args[0] == "/usr/bin/uv"]

    cmds_main = uv_args_only(calls_main)
    cmds_feature = uv_args_only(calls_feature)

    assert cmds_main == cmds_feature, (
        "Branch routing detected — subprocess commands differ between branches:\n"
        f"  branch=main:      {cmds_main}\n"
        f"  branch=feature-x: {cmds_feature}\n"
        "After the rewrite, both branches must go through the same uv venv + "
        "uv pip install path."
    )


# ===========================================================================
# Test 7: CMakeUserPresets.json is copied and patch script is invoked
# ===========================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_cmake_presets_copied(tmp_path):
    """CMakeUserPresets.json must be copied to the worktree and patch_cmake_presets.py
    must be invoked when VLLM_BASE_REPO contains the file.

    PASSES ALREADY (locking in existing behavior) — both the current and new
    implementations perform this copy+patch step.
    """
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-cmake-001"
    reset_worktree_manager()

    # Set up a fake VLLM_BASE_REPO with CMakeUserPresets.json
    base_repo = tmp_path / "vllm_base"
    base_repo.mkdir()
    presets_src = base_repo / "CMakeUserPresets.json"
    presets_src.write_text('{"version": 3, "configurePresets": []}')

    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)

    captured: list = []

    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}), \
         patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)):
        await mgr.initialize_vllm_environment(session_id=session_id)

    # CMakeUserPresets.json must have been copied to the worktree
    dst_presets = worktree_path / "CMakeUserPresets.json"
    assert dst_presets.exists(), (
        f"CMakeUserPresets.json was not copied to {dst_presets}. "
        "The new initialize_vllm_environment() must copy CMakeUserPresets.json."
    )

    # patch_cmake_presets.py must have been invoked
    patch_calls = [
        args for args, _kwargs in captured
        if any("patch_cmake_presets" in str(a) for a in args)
    ]
    assert len(patch_calls) >= 1, (
        "patch_cmake_presets.py was not invoked as a subprocess. "
        f"All subprocess calls: {[args for args, _ in captured]}"
    )


# ===========================================================================
# Test 8: Deleted helpers are not importable
# ===========================================================================

@pytest.mark.unit
def test_deleted_helpers_not_importable():
    """After the rewrite, the 11 deleted helpers must not exist on WorktreeManager
    or as module-level symbols. Also, fnmatch must not be imported by worktree_manager.

    RED: all helpers still exist in the current implementation.
    """
    import importlib
    import sys

    # Re-import fresh to avoid cached module
    if "orchestration.worktree_manager" in sys.modules:
        del sys.modules["orchestration.worktree_manager"]
    import orchestration.worktree_manager as wm

    from orchestration.worktree_manager import WorktreeManager

    # Module-level symbols that must be gone
    module_level_gone = ["DOCKER_COMMIT_FILE", "_get_precompiled_wheel_env"]
    for name in module_level_gone:
        assert not hasattr(wm, name), (
            f"Module-level '{name}' must be deleted — it was part of the "
            "old precompiled wheel commit lookup. Got: " + str(getattr(wm, name, None))
        )

    # WorktreeManager class attributes/methods that must be gone
    class_level_gone = [
        "MUTABLE_VENV_PATTERNS",
        "_is_mutable_venv_path",
        "_copy_venv_with_hardlinks",
        "_fix_venv_bin_shebangs",
        "_fix_venv_activate_scripts",
        "_get_cached_venv_path",
        "_compute_venv_version_marker",
        "_ensure_cached_venv",
        "_create_fresh_venv_from_requirements",
        "_copy_directory",
    ]
    for name in class_level_gone:
        assert not hasattr(WorktreeManager, name), (
            f"WorktreeManager.{name} must be deleted — it was part of the "
            "old venv caching/hardlinking pipeline that is being replaced."
        )

    # fnmatch must no longer be imported by worktree_manager
    # (it was used only by the deleted _is_mutable_venv_path)
    wm_source = Path(wm.__file__).read_text()
    assert "from fnmatch import fnmatch" not in wm_source, (
        "'from fnmatch import fnmatch' must be removed — fnmatch was used only by "
        "the deleted _is_mutable_venv_path()."
    )


# ===========================================================================
# Tests 8a-8e: _resolve_wheel_commit — pin policy that prevents the
#              nightly-wheel-publish-race 404 that blocks the session vLLM
#              install (and therefore vLLM server startup).
# ===========================================================================

def _wheel_commit_mgr(tmp_path, baked: str | None = None):
    """WorktreeManager with an isolated VLLM_BASE_REPO; optionally seed
    .docker_commit with `baked`. Returns (mgr, base_repo_path)."""
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager
    reset_worktree_manager()
    base_repo = tmp_path / "base_vllm"
    base_repo.mkdir(parents=True, exist_ok=True)
    if baked is not None:
        (base_repo / ".docker_commit").write_text(baked + "\n")
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    return mgr, base_repo


@pytest.mark.unit
def test_resolve_wheel_commit_default_branch_pins_baked(tmp_path):
    """Default branch 'main' → the baked .docker_commit (NOT empty). This is the
    fix for the 404 that left vLLM uninstalled: the baked commit's wheel was
    curl-verified at image build time, so it is guaranteed to exist."""
    baked = "b" * 40
    mgr, base_repo = _wheel_commit_mgr(tmp_path, baked=baked)
    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}):
        assert mgr._resolve_wheel_commit("main") == baked


@pytest.mark.unit
def test_resolve_wheel_commit_explicit_sha_passthrough(tmp_path):
    """An explicit 40-hex SHA passes through unchanged, even when a (different)
    baked commit exists — a pinned commit must never be overridden."""
    baked = "b" * 40
    pinned = "c" * 40
    mgr, base_repo = _wheel_commit_mgr(tmp_path, baked=baked)
    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}):
        assert mgr._resolve_wheel_commit(pinned) == pinned


@pytest.mark.unit
def test_resolve_wheel_commit_custom_branch_is_empty(tmp_path):
    """A user-selected non-default branch/tag → "" (setup.py self-resolves).
    The baked commit must NOT be silently substituted for a custom branch's
    binaries (would be a silent-correctness regression for Custom source mode)."""
    baked = "b" * 40
    mgr, base_repo = _wheel_commit_mgr(tmp_path, baked=baked)
    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}):
        assert mgr._resolve_wheel_commit("my-feature-branch") == ""
        assert mgr._resolve_wheel_commit("v0.21.0") == ""


@pytest.mark.unit
def test_resolve_wheel_commit_legacy_no_docker_commit(tmp_path):
    """Legacy image without .docker_commit → "" for the default branch
    (preserves the prior setup.py self-resolution behavior, no crash)."""
    mgr, base_repo = _wheel_commit_mgr(tmp_path, baked=None)
    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}):
        assert mgr._resolve_wheel_commit("main") == ""


@pytest.mark.unit
def test_resolve_wheel_commit_malformed_docker_commit(tmp_path):
    """A .docker_commit that is not a 40-hex SHA is rejected → "" (never pass a
    bad value to setup.py, which would itself 404)."""
    mgr, base_repo = _wheel_commit_mgr(tmp_path, baked="not-a-sha")
    with patch.dict(os.environ, {"VLLM_BASE_REPO": str(base_repo)}):
        assert mgr._resolve_wheel_commit("main") == ""


# ===========================================================================
# Tests 8f-8k: nvidia-cutlass-dsl[cu13] sub-wheel overlap repair.
#
#   The cu130 vLLM wheel pulls `nvidia-cutlass-dsl[cu13]`, whose extra installs
#   TWO overlapping sub-wheels — nvidia-cutlass-dsl-libs-base and
#   nvidia-cutlass-dsl-libs-cu13 — that both write the same ~99 cutlass/cute/*.py
#   paths with DIFFERENT content. uv's install order lets base clobber cu13,
#   which crashes FlashAttention-4 on Blackwell (NVFP4). The fix force-reinstalls
#   the cu13 sub-wheel LAST so it deterministically wins the overlap.
# ===========================================================================

def _seed_dist_info(venv: Path, *names: str) -> Path:
    """Create empty <name>.dist-info dirs under venv/lib/python3.12/site-packages.
    Returns the site-packages path."""
    sp = venv / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    for name in names:
        (sp / name).mkdir(parents=True, exist_ok=True)
    return sp


@pytest.mark.unit
def test_cutlass_repair_command_when_both_subwheels_present(tmp_path):
    """When BOTH libs-base and libs-cu13 dist-info dirs exist, the helper returns a
    `uv pip install --force-reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==<ver>`
    command, with the version derived from the cu13 dist-info dir name."""
    mgr = _make_manager(tmp_path)
    venv = tmp_path / "wt" / ".venv"
    _seed_dist_info(
        venv,
        "nvidia_cutlass_dsl_libs_base-4.5.2.dist-info",
        "nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info",
    )

    cmd = mgr._cutlass_cu13_repair_command(venv, "/usr/bin/uv", str(venv / "bin" / "python"))

    assert cmd is not None, "Repair command must be returned when both sub-wheels are present"
    assert cmd[0] == "/usr/bin/uv"
    assert "pip" in cmd and "install" in cmd
    assert "--force-reinstall" in cmd, f"Repair must force-reinstall, got {cmd}"
    assert "--no-deps" in cmd, f"Repair must use --no-deps (only reinstall the libs wheel), got {cmd}"
    assert "--python" in cmd, f"Repair must target the session venv via --python, got {cmd}"
    assert str(venv / "bin" / "python") in cmd, f"Repair must target the session python, got {cmd}"
    assert "nvidia-cutlass-dsl-libs-cu13==4.5.2" in cmd, (
        f"Repair must pin the cu13 libs sub-wheel to the installed version 4.5.2, got {cmd}"
    )


@pytest.mark.unit
def test_cutlass_repair_version_derived_dynamically(tmp_path):
    """The pinned version is read from the on-disk cu13 dist-info dir name, not
    hardcoded — a different version flows through unchanged."""
    mgr = _make_manager(tmp_path)
    venv = tmp_path / "wt" / ".venv"
    _seed_dist_info(
        venv,
        "nvidia_cutlass_dsl_libs_base-4.6.0.dist-info",
        "nvidia_cutlass_dsl_libs_cu13-4.6.0.dist-info",
    )

    cmd = mgr._cutlass_cu13_repair_command(venv, "/usr/bin/uv", str(venv / "bin" / "python"))

    assert cmd is not None
    assert "nvidia-cutlass-dsl-libs-cu13==4.6.0" in cmd, (
        f"Version must be derived from the dist-info dir name (4.6.0), got {cmd}"
    )


@pytest.mark.unit
def test_cutlass_repair_skipped_when_only_base(tmp_path):
    """Only the base sub-wheel present (no cu13 extra) → no repair (nothing to do)."""
    mgr = _make_manager(tmp_path)
    venv = tmp_path / "wt" / ".venv"
    _seed_dist_info(venv, "nvidia_cutlass_dsl_libs_base-4.5.2.dist-info")

    assert mgr._cutlass_cu13_repair_command(venv, "/usr/bin/uv", str(venv / "bin" / "python")) is None


@pytest.mark.unit
def test_cutlass_repair_skipped_when_only_cu13(tmp_path):
    """Only the cu13 sub-wheel present (no base to clobber it) → no repair needed."""
    mgr = _make_manager(tmp_path)
    venv = tmp_path / "wt" / ".venv"
    _seed_dist_info(venv, "nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info")

    assert mgr._cutlass_cu13_repair_command(venv, "/usr/bin/uv", str(venv / "bin" / "python")) is None


@pytest.mark.unit
def test_cutlass_repair_skipped_when_neither(tmp_path):
    """Neither sub-wheel present (e.g. a non-cu130 install) → no repair."""
    mgr = _make_manager(tmp_path)
    venv = tmp_path / "wt" / ".venv"
    _seed_dist_info(venv)  # empty site-packages

    assert mgr._cutlass_cu13_repair_command(venv, "/usr/bin/uv", str(venv / "bin" / "python")) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_install_flow_runs_cutlass_repair_after_editable_install(tmp_path):
    """initialize_vllm_environment must run the cu13 repair AS A SUBPROCESS, and
    AFTER the editable install, when both sub-wheels are present in the venv.

    The repair is the LAST uv command so it wins the cutlass/cute/*.py overlap.
    """
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-cutlass-repair-001"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)
    # Pre-seed the venv as if the editable install already landed both sub-wheels
    # (create_subprocess_exec is mocked, so it won't create these itself).
    _seed_dist_info(
        worktree_path / ".venv",
        "nvidia_cutlass_dsl_libs_base-4.5.2.dist-info",
        "nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info",
    )

    captured: list = []
    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
         patch("shutil.copy2"):
        await mgr.initialize_vllm_environment(session_id=session_id, branch="main")

    uv_cmds = [args for args, _kwargs in captured if args and args[0] == "/usr/bin/uv"]
    install_idx = next((i for i, a in enumerate(uv_cmds) if "-e" in a and "." in a), None)
    repair_idx = next((i for i, a in enumerate(uv_cmds) if "--force-reinstall" in a), None)

    assert install_idx is not None, f"editable install call missing, got {uv_cmds}"
    assert repair_idx is not None, (
        f"cu13 repair (--force-reinstall) must run when both sub-wheels are present, got {uv_cmds}"
    )
    assert repair_idx > install_idx, (
        "cu13 repair must run AFTER the editable install (so it wins the file overlap). "
        f"install_idx={install_idx} repair_idx={repair_idx} cmds={uv_cmds}"
    )
    repair = uv_cmds[repair_idx]
    assert "--no-deps" in repair and "nvidia-cutlass-dsl-libs-cu13==4.5.2" in repair, (
        f"repair command malformed: {repair}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_install_flow_skips_cutlass_repair_when_extra_absent(tmp_path):
    """When the cu13 extra is NOT installed (no sub-wheel dist-info), the flow must
    NOT emit a --force-reinstall command (no spurious work on non-cu130 installs)."""
    from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager

    session_id = "test-cutlass-repair-002"
    reset_worktree_manager()
    mgr = WorktreeManager(
        repos_dir=str(tmp_path / "repos"),
        sessions_dir=str(tmp_path / "sessions"),
    )
    worktree_path = mgr.get_worktree_path(session_id)
    worktree_path.mkdir(parents=True, exist_ok=True)

    captured: list = []
    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
         patch("shutil.copy2"):
        await mgr.initialize_vllm_environment(session_id=session_id, branch="main")

    uv_cmds = [args for args, _kwargs in captured if args and args[0] == "/usr/bin/uv"]
    assert not any("--force-reinstall" in a for a in uv_cmds), (
        f"No repair should run when the cu13 extra is absent, got {uv_cmds}"
    )


# ===========================================================================
# Tests 9-11: Dockerfile guards (Track 2 — DO NOT MODIFY)
# ===========================================================================

def _read_dockerfile() -> str:
    p = PROJECT_ROOT / "Dockerfile"
    if not p.exists():
        pytest.skip("Dockerfile not found")
    return p.read_text()


@pytest.mark.unit
class TestDockerfileGuards:
    """Guard the Dockerfile's vLLM server-venv build block.

    Replanted from TestDockerfilePrecompiledWheel in the deleted
    test_vllm_precompiled_wheel.py (DA review fix #1).
    """

    # ----------------------------------------------------------------
    # D-1 (NEW): Release-version ARGs are pinned on their own lines
    # ----------------------------------------------------------------
    def test_dockerfile_pins_vllm_release_version(self):
        """The Dockerfile must pin vLLM to a release tag on its own ARG lines:
        - declare `ARG VLLM_VERSION=0.24.0` as its own ARG line
        - NOT declare an `ARG CUDA_VERSION` (dead-code from the draft block)
        - declare `ARG VLLM_BRANCH=v0.24.0` as a static literal (cross-ARG
          substitution doesn't work reliably in Docker ARG)
        - remove the bare `uv pip install https://github.com/vllm-project/...`
          draft line that was missing its `RUN` prefix
        """
        content = _read_dockerfile()

        assert "ARG VLLM_VERSION=0.24.0" in content, (
            "Dockerfile must declare 'ARG VLLM_VERSION=0.24.0' pinning the release."
        )
        # Dead-code: the draft block had ARG CUDA_VERSION=130 which is unused
        # (the cu130 variant is selected via VLLM_PRECOMPILED_WHEEL_VARIANT,
        # not cross-ARG interpolation into a wheel URL).
        assert "ARG CUDA_VERSION" not in content, (
            "Dockerfile must NOT declare 'ARG CUDA_VERSION' — it's dead code "
            "from the draft block. The cu130 variant is selected via "
            "VLLM_PRECOMPILED_WHEEL_VARIANT, not a hardcoded wheel URL."
        )
        assert "ARG VLLM_BRANCH=v0.24.0" in content, (
            "Dockerfile must declare 'ARG VLLM_BRANCH=v0.24.0' as a static literal. "
            "Cross-ARG substitution (v${VLLM_VERSION}) doesn't work reliably in Docker ARG."
        )
        # The dead draft `uv pip install https://github.com/...` line (no RUN prefix)
        # must be gone.
        assert "vllm-${VLLM_VERSION}+cu${CUDA_VERSION}" not in content, (
            "Dockerfile must NOT contain the dead draft 'uv pip install ...' line "
            "that references ${VLLM_VERSION}+cu${CUDA_VERSION} (no RUN prefix). "
            "Replace with a proper RUN block using VLLM_PRECOMPILED_WHEEL_LOCATION."
        )
        # Also guard against the specific broken draft line format
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("uv pip install ") and "vllm-" in stripped and ".whl" in stripped:
                raise AssertionError(
                    f"Found dead draft 'uv pip install' line with no RUN prefix: {line!r}. "
                    "All uv pip install commands must be part of a RUN block."
                )

    # ----------------------------------------------------------------
    # D-2: Release wheel via the cu130 commit-probe mechanism
    # ----------------------------------------------------------------
    def test_dockerfile_uses_cu130_commit_probe_not_nightly(self):
        """The install block must use the cu130 commit-probe mechanism — it pins
        VLLM_PRECOMPILED_WHEEL_COMMIT to the cloned release commit and selects the
        cu130 variant index — and must NOT use the old nightly tag.

        This guards the *mechanism*, not the version, so it does not need editing
        on future version bumps:
        - VLLM_PRECOMPILED_WHEEL_COMMIT="nightly" must be absent
        - the wheel index host https://wheels.vllm.ai/ is probed at build time
        - VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 selects the CUDA 13 artifacts
        - --torch-backend=cu130 installs the matching torch
        - VLLM_USE_PRECOMPILED=1 + editable install remain

        Plain `in` checks only — NO regex (Python 3.12 escape-sequence warnings
        would pollute test output).
        """
        content = _read_dockerfile()
        assert 'VLLM_PRECOMPILED_WHEEL_COMMIT="nightly"' not in content, (
            "Dockerfile must not set VLLM_PRECOMPILED_WHEEL_COMMIT=\"nightly\" — "
            "it pins to the cloned release commit instead."
        )
        assert "VLLM_PRECOMPILED_WHEEL_COMMIT=" in content, (
            "Dockerfile must set VLLM_PRECOMPILED_WHEEL_COMMIT=<sha> "
            "(the cloned release commit) on the editable-install RUN block."
        )
        assert "https://wheels.vllm.ai/" in content, (
            "Dockerfile must probe the wheel index host https://wheels.vllm.ai/ "
            "to fail early if the release commit has no cu130 wheel."
        )
        assert "VLLM_PRECOMPILED_WHEEL_VARIANT=cu130" in content, (
            "Dockerfile must select the cu130 variant via "
            "VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 (CUDA 13 artifacts)."
        )
        assert "--torch-backend=cu130" in content, (
            "Dockerfile must install the matching torch via --torch-backend=cu130."
        )
        assert "VLLM_USE_PRECOMPILED=1" in content, (
            "VLLM_USE_PRECOMPILED=1 must stay set on the editable-install RUN line."
        )
        # Same RUN block still does the editable install (it's the install mechanism
        # that remains; only the source of the precompiled artifact changed).
        assert "uv pip install -e ." in content, (
            "Dockerfile must keep the editable install: 'uv pip install -e .'."
        )

    # ----------------------------------------------------------------
    # D-3 (RENAMED from test_dockerfile_uses_nightly_not_head):
    # The install RUN block no longer uses nightly or $(git rev-parse HEAD).
    # ----------------------------------------------------------------
    def test_dockerfile_install_env_block_present(self):
        """The vLLM editable-install RUN block must:
        - contain 'uv pip install -e .'
        - NOT contain `$(git rev-parse HEAD)` (dynamic HEAD lookup)
        - NOT contain `="nightly"` (the old nightly tag)
        - contain VLLM_USE_PRECOMPILED=1
        """
        content = _read_dockerfile()

        # Find the RUN line that installs vLLM as editable
        lines = content.splitlines()
        # Walk through the file and join continuation lines (ending with `\`) into logical blocks
        blocks: list[str] = []
        buf: list[str] = []
        for line in lines:
            buf.append(line)
            if not line.rstrip().endswith("\\"):
                blocks.append("\n".join(buf))
                buf = []
        if buf:
            blocks.append("\n".join(buf))

        install_blocks = [
            b for b in blocks
            if "uv pip install -e ." in b and ("VLLM_USE_PRECOMPILED" in b or "VLLM_PRECOMPILED_WHEEL_LOCATION" in b)
        ]
        assert install_blocks, (
            "Dockerfile must have a RUN block that does 'uv pip install -e .' "
            "together with VLLM_USE_PRECOMPILED / VLLM_PRECOMPILED_WHEEL_LOCATION env vars."
        )
        block = install_blocks[0]
        assert "$(git rev-parse HEAD)" not in block, (
            "The editable-install RUN block must NOT use $(git rev-parse HEAD). "
            f"Got block:\n{block}"
        )
        assert '="nightly"' not in block, (
            'The editable-install RUN block must NOT use ="nightly" (old wheel commit). '
            f"Got block:\n{block}"
        )
        assert "VLLM_USE_PRECOMPILED=1" in block, (
            "The editable-install RUN block must set VLLM_USE_PRECOMPILED=1. "
            f"Got block:\n{block}"
        )

    # ----------------------------------------------------------------
    # D-6 (comment-only update):
    # The install block still must NOT silently swallow failures with '|| true'.
    # After the switch it now guards the VLLM_PRECOMPILED_WHEEL_LOCATION block.
    # ----------------------------------------------------------------
    def test_dockerfile_no_silent_failure(self):
        """The VLLM_USE_PRECOMPILED / uv pip install block must NOT use || true.

        (D-6, comment-only update after switching to the release wheel:
        this now guards the VLLM_PRECOMPILED_WHEEL_LOCATION-based install block.)

        '|| true' would hide build failures and produce a broken image
        that only fails at session-create time, not at docker-build time.
        """
        content = _read_dockerfile()
        # Find the VLLM_USE_PRECOMPILED block
        lines = content.splitlines()
        in_block = False
        block_lines = []
        for line in lines:
            if "VLLM_USE_PRECOMPILED" in line or (in_block and "uv pip install" in line):
                in_block = True
            if in_block:
                block_lines.append(line)
                # Block ends at a blank line or the next RUN statement
                if line.strip() == "" or (line.startswith("RUN") and block_lines and len(block_lines) > 1):
                    break

        block_text = "\n".join(block_lines)
        assert "|| true" not in block_text, (
            "VLLM_USE_PRECOMPILED / uv pip install block must not use '|| true' "
            "(silent failure would produce a broken image)"
        )

    def test_dockerfile_has_so_verification(self):
        """Dockerfile must verify .abi3.so files are present after editable install.

        Without this check, a broken precompiled wheel install silently produces
        a venv with no C extensions, causing 'import vllm' to fail at runtime.
        """
        content = _read_dockerfile()
        assert ".abi3.so" in content, (
            "Dockerfile must verify .abi3.so files are present "
            "(e.g. 'find vllm -name \"*.abi3.so\" | wc -l')"
        )
        # The actual count-check RUN line
        assert "wc -l" in content, (
            "Dockerfile must include a wc -l count check for .abi3.so files"
        )

    # ----------------------------------------------------------------
    # D-7 (NEW): Both .docker_commit (SHA) and .docker_version (tag)
    # are written inside the vLLM build block.
    # ----------------------------------------------------------------
    def test_dockerfile_writes_docker_commit_and_docker_version(self):
        """worktree_manager relies on /workspace/vllm/.docker_commit being a
        40-char SHA, so the RUN line that writes it via `git rev-parse HEAD` must
        still exist. A NEW companion file /workspace/vllm/.docker_version holds
        the release tag (e.g. v0.20.0) for UI display.
        """
        content = _read_dockerfile()

        # .docker_commit — still populated via git rev-parse HEAD (40-char SHA)
        assert "/workspace/vllm/.docker_commit" in content, (
            "Dockerfile must still write /workspace/vllm/.docker_commit "
            "(worktree_manager depends on this 40-char SHA file)."
        )
        assert "git rev-parse HEAD" in content, (
            "Dockerfile must still use 'git rev-parse HEAD' to write the "
            "40-char SHA to /workspace/vllm/.docker_commit."
        )

        # .docker_version — NEW file with the release version string.
        assert "/workspace/vllm/.docker_version" in content, (
            "Dockerfile must write /workspace/vllm/.docker_version holding "
            "the release version string (e.g. v0.20.0) for UI display."
        )
        # Must be populated from ${VLLM_VERSION} (either bare or with leading 'v').
        assert "${VLLM_VERSION}" in content, (
            "The write to .docker_version must reference ${VLLM_VERSION} "
            "(either 'v${VLLM_VERSION}' or ${VLLM_VERSION} itself)."
        )

    # ----------------------------------------------------------------
    # D-8 (NEW): vLLM clone is pinned to the release tag via VLLM_BRANCH
    # ----------------------------------------------------------------
    def test_dockerfile_clones_vllm_at_release_tag(self):
        """Dockerfile must:
        - clone vLLM via `git clone -b ${VLLM_BRANCH} ${VLLM_REPO} /workspace/vllm`
        - set VLLM_BRANCH=v0.24.0 as a static literal (no cross-ARG interpolation)
        """
        content = _read_dockerfile()
        assert "git clone -b ${VLLM_BRANCH}" in content, (
            "Dockerfile must clone vLLM via 'git clone -b ${VLLM_BRANCH} ...'."
        )
        assert "${VLLM_REPO}" in content, (
            "Dockerfile must clone via ${VLLM_REPO} (parameterised repo URL)."
        )
        assert "/workspace/vllm" in content, (
            "Dockerfile must clone into /workspace/vllm."
        )
        # Static literal, NOT v${VLLM_VERSION} — Docker ARG doesn't support
        # reliable cross-ARG substitution.
        assert "ARG VLLM_BRANCH=v0.24.0" in content, (
            "ARG VLLM_BRANCH must be the static literal 'v0.24.0', "
            "not v${VLLM_VERSION}."
        )
        assert "ARG VLLM_BRANCH=v${VLLM_VERSION}" not in content, (
            "ARG VLLM_BRANCH must NOT use cross-ARG substitution v${VLLM_VERSION}."
        )
