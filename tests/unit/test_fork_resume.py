# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_resume.py
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import reset_all_singletons  # noqa: F401
from shared.session_models import SessionState, SessionStatus, CLIToolType


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resume_fork_reclones_and_builds_source(tmp_path, monkeypatch):
    """On resume of a fork session, re-init must use precompiled=False and
    re-clone the fork base before rebuilding."""
    from orchestration import session_manager as smmod

    st = SessionState(
        session_id="r1", status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(tmp_path / "wt"),
        vllm_fork_url="https://github.com/u/vllm.git",
        vllm_fork_token_encrypted=None,
    )
    # Exercise the helper that resume calls for fork re-init.
    sm = smmod.SessionManager.__new__(smmod.SessionManager)
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.create_worktree = MagicMock()
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {}}
    )
    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        gf.return_value.ensure_fork_base.return_value = Path(tmp_path / "fb")
        await sm._reinit_fork_env_on_resume(st)

    # rebuilt from source
    _, kwargs = sm.worktree_manager.initialize_vllm_environment.call_args
    assert kwargs["precompiled"] is False
    gf.return_value.ensure_fork_base.assert_called_once()
