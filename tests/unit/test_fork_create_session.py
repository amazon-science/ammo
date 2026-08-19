# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_create_session.py
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import reset_all_singletons  # noqa: F401
from shared.session_models import (
    CreateSessionRequest, SessionStatus, CLIToolType,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fork_create_returns_building_and_spawns_task(tmp_path, monkeypatch):
    from orchestration.session_manager import SessionManager

    sm = SessionManager.__new__(SessionManager)  # bypass heavy __init__
    # Minimal wiring the create path touches:
    sm._sessions = {}
    sm._recovery_locks = {}
    sm.sessions_dir = tmp_path / "sessions"
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.get_session_dir.return_value = tmp_path / "sessions" / "x"
    sm.worktree_manager.get_logs_dir.return_value = tmp_path / "sessions" / "x" / "logs"
    sm.worktree_manager.create_session_dirs.return_value = {
        "session_dir": tmp_path / "sessions" / "x",
        "logs_dir": tmp_path / "sessions" / "x" / "logs",
    }
    sm.inactivity_monitor = None
    sm._save_session_state = MagicMock()
    sm._acquire_gpus = AsyncMock(return_value=[0])
    captured = {}

    async def fake_build(state, request):
        captured["ran"] = state.session_id
    sm._run_fork_build = AsyncMock(side_effect=fake_build)

    req = CreateSessionRequest(
        gpu_count=1,
        vllm_fork_url="https://github.com/u/vllm.git",
    )
    monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet.generate_key().decode())

    with patch("asyncio.create_task", side_effect=lambda coro: __import__("asyncio").ensure_future(coro)):
        resp = await sm.create_session(req, owner_id=None)

    assert resp.status == SessionStatus.BUILDING.value
    # GPUs were acquired (held through build)
    sm._acquire_gpus.assert_awaited()
