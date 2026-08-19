# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_terminate_cleanup.py
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import reset_all_singletons  # noqa: F401
from shared.session_models import SessionState, SessionStatus, CLIToolType


@pytest.mark.unit
def test_fork_base_removed_only_when_no_other_session_uses_it(tmp_path):
    from orchestration.session_manager import SessionManager
    from orchestration.fork_repo_manager import get_fork_repo_manager, reset_fork_repo_manager

    reset_fork_repo_manager()
    sm = SessionManager.__new__(SessionManager)
    url = "https://github.com/u/vllm.git"
    s1 = SessionState(session_id="a", status=SessionStatus.TERMINATED,
                      cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
                      created_at=1.0, last_accessed=1.0, vllm_fork_url=url)
    s2 = SessionState(session_id="b", status=SessionStatus.ACTIVE,
                      cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
                      created_at=1.0, last_accessed=1.0, vllm_fork_url=url)
    sm._sessions = {"a": s1, "b": s2}

    fork_mgr = MagicMock()
    # Another active session (b) shares the fork → must NOT remove.
    assert sm._fork_base_in_use_by_others("a", url) is True

    # If b is gone, it is removable.
    sm._sessions = {"a": s1}
    assert sm._fork_base_in_use_by_others("a", url) is False
