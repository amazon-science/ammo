# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_build_orchestration.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import reset_all_singletons  # noqa: F401
from shared.session_models import (
    SessionState, SessionStatus, CLIToolType, CreateSessionRequest,
)


def _make_sm(tmp_path):
    from orchestration.session_manager import SessionManager
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm.sessions_dir = tmp_path / "sessions"
    sm.worktree_manager = MagicMock()
    sm.cli_tool_manager = MagicMock()
    sm.terminal_manager = MagicMock()
    sm.terminal_manager.is_available.return_value = True
    sm.terminal_manager.start_terminal_with_command = AsyncMock(return_value=8042)
    sm.inactivity_monitor = None
    sm._save_session_state = MagicMock()
    sm._release_gpus = MagicMock()
    sm._chown_session_to_user = MagicMock()
    return sm


def _state(tmp_path, sid="fb1"):
    sdir = tmp_path / "sessions" / sid
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    (sdir / "worktree").mkdir(parents=True, exist_ok=True)
    return SessionState(
        session_id=sid, status=SessionStatus.BUILDING,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        session_dir=str(sdir), logs_dir=str(sdir / "logs"),
        worktree_path=str(sdir / "worktree"), gpu_ids=[0],
        vllm_fork_url="https://github.com/u/vllm.git",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_fork_build_success_path(tmp_path):
    sm = _make_sm(tmp_path)
    st = _state(tmp_path)
    sm._sessions[st.session_id] = st
    sm.worktree_manager.create_worktree = MagicMock(return_value=Path(st.worktree_path))
    sm.worktree_manager.get_worktree_path.return_value = Path(st.worktree_path)
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {"total": 1.0}}
    )
    sm.cli_tool_manager.get_cli_command.return_value = ["/usr/bin/env", "claude"]
    req = CreateSessionRequest(gpu_count=1, vllm_fork_url=st.vllm_fork_url)

    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        gf.return_value.ensure_fork_base.return_value = Path(tmp_path / "fork_base")
        await sm._run_fork_build(st, req)

    assert st.status == SessionStatus.ACTIVE
    sm.terminal_manager.start_terminal_with_command.assert_awaited()
    sm._release_gpus.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_fork_build_failure_retries_then_fails(tmp_path):
    sm = _make_sm(tmp_path)
    st = _state(tmp_path, sid="fb2")
    sm._sessions[st.session_id] = st
    sm.worktree_manager.create_worktree = MagicMock(return_value=Path(st.worktree_path))
    sm.worktree_manager.get_worktree_path.return_value = Path(st.worktree_path)
    # Always fail the build.
    from orchestration.worktree_manager import WorktreeError
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(side_effect=WorktreeError("boom"))
    req = CreateSessionRequest(gpu_count=1, vllm_fork_url=st.vllm_fork_url)

    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        gf.return_value.ensure_fork_base.return_value = Path(tmp_path / "fork_base")
        await sm._run_fork_build(st, req)

    assert st.status == SessionStatus.FAILED
    assert st.build_error
    # retried once → called twice
    assert sm.worktree_manager.initialize_vllm_environment.await_count == 2
    sm._release_gpus.assert_called_once_with(st.session_id)
