# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_session_models.py
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    CreateSessionRequest,
    SessionStatus,
    SessionState,
    CLIToolType,
    FORK_REPOS_SUBDIR,
)


@pytest.mark.unit
class TestForkRequestModel:
    def test_building_status_exists(self):
        assert SessionStatus.BUILDING.value == "building"

    def test_fork_url_optional_default_none(self):
        req = CreateSessionRequest(gpu_count=1)
        assert req.vllm_fork_url is None
        assert req.vllm_fork_token is None

    def test_fork_url_and_token_accepted(self):
        req = CreateSessionRequest(
            gpu_count=1,
            vllm_fork_url="https://github.com/u/vllm.git",
            vllm_fork_token="ghp_x",
        )
        assert req.vllm_fork_url == "https://github.com/u/vllm.git"

    def test_token_without_url_rejected(self):
        with pytest.raises(ValidationError) as exc:
            CreateSessionRequest(gpu_count=1, vllm_fork_token="ghp_x")
        assert "vllm_fork_url" in str(exc.value).lower()

    def test_fork_constant(self):
        assert FORK_REPOS_SUBDIR == "forks"


@pytest.mark.unit
class TestForkSessionState:
    def test_state_round_trip_with_fork_fields(self):
        st = SessionState(
            session_id="s1",
            status=SessionStatus.BUILDING,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="my-branch",
            created_at=1.0,
            last_accessed=1.0,
            vllm_fork_url="https://github.com/u/vllm.git",
            vllm_fork_token_encrypted="enc-blob",
            build_phase="compiling",
            build_error=None,
        )
        d = st.to_dict()
        assert d["vllm_fork_url"] == "https://github.com/u/vllm.git"
        assert d["vllm_fork_token_encrypted"] == "enc-blob"
        assert d["build_phase"] == "compiling"
        st2 = SessionState.from_dict(d)
        assert st2.vllm_fork_url == "https://github.com/u/vllm.git"
        assert st2.vllm_fork_token_encrypted == "enc-blob"
        assert st2.build_phase == "compiling"
        assert st2.status == SessionStatus.BUILDING

    def test_legacy_state_without_fork_fields(self):
        # Simulate old S3 metadata with no fork keys.
        legacy = {
            "session_id": "s2", "status": "active", "cli_tool": "claude",
            "repo_name": "vllm", "branch": "main",
            "created_at": 1.0, "last_accessed": 1.0,
        }
        st = SessionState.from_dict(legacy)
        assert st.vllm_fork_url is None
        assert st.vllm_fork_token_encrypted is None
        assert st.build_phase is None
