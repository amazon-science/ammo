# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_source_build.py
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager


def _success_proc():
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _make_capturing_exec(captured):
    async def _exec(*args, **kwargs):
        captured.append((args, kwargs))
        return _success_proc()
    return _exec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_source_build_sets_precompiled_off_and_no_wheel_commit(tmp_path):
    reset_worktree_manager()
    mgr = WorktreeManager(repos_dir=str(tmp_path / "repos"),
                          sessions_dir=str(tmp_path / "sessions"))
    sid = "fork-build-1"
    wt = mgr.get_worktree_path(sid)
    wt.mkdir(parents=True, exist_ok=True)
    log_path = mgr.get_logs_dir(sid)
    log_path.mkdir(parents=True, exist_ok=True)
    build_log = log_path / "fork_build.log"

    captured = []
    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
         patch("shutil.copy2"):
        result = await mgr.initialize_vllm_environment(
            session_id=sid, branch="feature-x",
            precompiled=False, log_path=str(build_log),
        )

    assert result["status"] == "success"
    # Find the editable-install invocation and check its env.
    install_calls = [c for c in captured if "pip" in c[0] and "install" in c[0]]
    assert install_calls, "expected a pip install call"
    env = install_calls[0][1]["env"]
    assert env["VLLM_USE_PRECOMPILED"] == "0"
    assert "VLLM_PRECOMPILED_WHEEL_COMMIT" not in env or env.get("VLLM_PRECOMPILED_WHEEL_COMMIT") == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_precompiled_true_unchanged_default(tmp_path):
    """Default (precompiled=True) path still sets VLLM_USE_PRECOMPILED=1."""
    reset_worktree_manager()
    mgr = WorktreeManager(repos_dir=str(tmp_path / "repos"),
                          sessions_dir=str(tmp_path / "sessions"))
    sid = "precompiled-1"
    wt = mgr.get_worktree_path(sid)
    wt.mkdir(parents=True, exist_ok=True)
    captured = []
    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("asyncio.create_subprocess_exec", side_effect=_make_capturing_exec(captured)), \
         patch.dict(os.environ, {"VLLM_BASE_REPO": str(tmp_path / "nope")}), \
         patch("shutil.copy2"):
        result = await mgr.initialize_vllm_environment(session_id=sid, branch="main")
    assert result["status"] == "success"
    install_calls = [c for c in captured if "pip" in c[0] and "install" in c[0]]
    assert install_calls[0][1]["env"]["VLLM_USE_PRECOMPILED"] == "1"


@pytest.mark.unit
def test_build_log_cap_constant_present():
    """correctness-6: the fork build log is capped at 64MB to prevent disk exhaustion."""
    from orchestration import worktree_manager as wm
    assert wm.FORK_BUILD_LOG_MAX_BYTES == 64 * 1024 * 1024
