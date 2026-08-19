# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for AI CLI Session Service models and components.
"""

import pytest
import json
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from pydantic import ValidationError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
    CreateSessionRequest,
    SessionInfo,
    CreateSessionResponse,
    SessionListResponse,
    SessionActionResponse,
    SUPPORTED_REPOS,
)


@pytest.mark.unit
class TestSessionModels:
    """Test session data models."""

    def test_create_session_request_defaults(self):
        """Test CreateSessionRequest with defaults."""
        request = CreateSessionRequest()

        assert request.repo_name == "vllm"
        assert request.cli_tool == CLIToolType.CLAUDE
        assert request.branch == "main"
        assert request.initial_prompt is None
        assert request.gpu_count == 0
        assert request.session_id is None
        assert request.inactivity_timeout_mins == 720

    def test_create_session_request_custom(self):
        """Test CreateSessionRequest with custom values."""
        request = CreateSessionRequest(
            repo_name="vllm",
            cli_tool=CLIToolType.CODEX,
            branch="feature/test",
            initial_prompt="Hello",
            gpu_count=2,
            inactivity_timeout_mins=60,
        )

        assert request.cli_tool == CLIToolType.CODEX
        assert request.branch == "feature/test"
        assert request.initial_prompt == "Hello"
        assert request.gpu_count == 2
        assert request.inactivity_timeout_mins == 60

    def test_session_state_to_dict(self):
        """Test SessionState serialization."""
        state = SessionState(
            session_id="test-123",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            gpu_ids=[0, 1],
        )

        data = state.to_dict()

        assert data["session_id"] == "test-123"
        assert data["status"] == "active"
        assert data["cli_tool"] == "claude"
        assert data["repo_name"] == "vllm"
        assert data["gpu_ids"] == [0, 1]

    def test_session_state_from_dict(self):
        """Test SessionState deserialization."""
        data = {
            "session_id": "test-456",
            "status": "paused",
            "cli_tool": "codex",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
            "gpu_ids": [2],
            "worktree_path": "/data/sessions/test-456/worktree",
        }

        state = SessionState.from_dict(data)

        assert state.session_id == "test-456"
        assert state.status == SessionStatus.PAUSED
        assert state.cli_tool == CLIToolType.CODEX
        assert state.gpu_ids == [2]
        assert state.worktree_path == "/data/sessions/test-456/worktree"

    def test_session_state_roundtrip(self):
        """Test SessionState serialize/deserialize roundtrip."""
        original = SessionState(
            session_id="roundtrip-test",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="develop",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            gpu_ids=[0],
            worktree_path="/data/sessions/roundtrip-test/worktree",
            cli_session_id="claude-abc123",
            terminal_port=8001,
        )

        data = original.to_dict()
        restored = SessionState.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.status == original.status
        assert restored.cli_tool == original.cli_tool
        assert restored.gpu_ids == original.gpu_ids
        assert restored.worktree_path == original.worktree_path
        assert restored.cli_session_id == original.cli_session_id
        assert restored.terminal_port == original.terminal_port

    def test_session_state_to_session_info(self):
        """Test conversion to SessionInfo."""
        state = SessionState(
            session_id="info-test",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            gpu_ids=[0],
            terminal_port=8001,
        )

        info = state.to_session_info()

        assert info.session_id == "info-test"
        assert info.status == "active"
        assert info.terminal_url == "/sessions/info-test/terminal/"
        assert info.terminal_ws_url == "/sessions/info-test/terminal/ws"
        assert "pod_name" not in info.model_dump()

    def test_session_state_to_session_info_no_terminal(self):
        """Test conversion to SessionInfo without terminal."""
        state = SessionState(
            session_id="no-terminal-test",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
        )

        info = state.to_session_info()

        assert info.session_id == "no-terminal-test"
        assert info.status == "paused"
        assert info.terminal_url is None
        assert info.terminal_ws_url is None

    def test_session_list_response(self):
        """Test SessionListResponse."""
        sessions = [
            SessionInfo(
                session_id="s1",
                status="active",
                cli_tool="claude",
                repo_name="vllm",
                branch="main",
                created_at=1234567890.0,
                last_accessed=1234567900.0,
            ),
            SessionInfo(
                session_id="s2",
                status="paused",
                cli_tool="codex",
                repo_name="vllm",
                branch="develop",
                created_at=1234567800.0,
                last_accessed=1234567850.0,
            ),
        ]

        response = SessionListResponse(sessions=sessions, total=2)

        assert len(response.sessions) == 2
        assert response.total == 2
        assert response.sessions[0].session_id == "s1"
        assert response.sessions[1].session_id == "s2"


@pytest.mark.unit
class TestSupportedRepos:
    """Test supported repositories configuration."""

    def test_vllm_repo_configured(self):
        """Test vLLM repository is configured."""
        assert "vllm" in SUPPORTED_REPOS
        assert "url" in SUPPORTED_REPOS["vllm"]
        assert "default_branch" in SUPPORTED_REPOS["vllm"]

    def test_vllm_repo_url(self):
        """Test vLLM repository URL."""
        assert "github.com/vllm-project/vllm" in SUPPORTED_REPOS["vllm"]["url"]


@pytest.mark.unit
class TestSessionStatus:
    """Test session status enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert SessionStatus.CREATING.value == "creating"
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.TERMINATED.value == "terminated"
        assert SessionStatus.FAILED.value == "failed"

    def test_status_from_string(self):
        """Test creating status from string."""
        status = SessionStatus("active")
        assert status == SessionStatus.ACTIVE


@pytest.mark.unit
class TestCLIToolType:
    """Test CLI tool type enum."""

    def test_tool_values(self):
        """Test all tool values exist."""
        assert CLIToolType.CLAUDE.value == "claude"
        assert CLIToolType.CODEX.value == "codex"

    def test_tool_from_string(self):
        """Test creating tool type from string."""
        tool = CLIToolType("claude")
        assert tool == CLIToolType.CLAUDE


@pytest.mark.unit
class TestModelNameDtype:
    """Test model_name and dtype fields on session models."""

    def test_create_session_request_accepts_model_fields(self):
        """CreateSessionRequest accepts model_name and dtype."""
        request = CreateSessionRequest(
            model_name="DeepSeek-R1",
            dtype="fp8",
        )
        assert request.model_name == "DeepSeek-R1"
        assert request.dtype == "fp8"

    def test_create_session_request_model_fields_optional(self):
        """model_name and dtype are optional in CreateSessionRequest."""
        request = CreateSessionRequest()
        assert request.model_name is None
        assert request.dtype is None

    def test_session_state_to_session_info_includes_model(self):
        """to_session_info() passes model_name/dtype through to SessionInfo."""
        state = SessionState(
            session_id="model-test",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            model_name="Qwen3-Coder-480B",
            dtype="bf16",
        )
        info = state.to_session_info()
        assert info.model_name == "Qwen3-Coder-480B"
        assert info.dtype == "bf16"

    def test_session_state_serialization_roundtrip(self):
        """SessionState with model_name/dtype survives to_dict/from_dict roundtrip."""
        original = SessionState(
            session_id="roundtrip-model",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            model_name="Llama-4-Scout",
            dtype="fp16",
        )
        data = original.to_dict()
        assert data["model_name"] == "Llama-4-Scout"
        assert data["dtype"] == "fp16"

        restored = SessionState.from_dict(data)
        assert restored.model_name == "Llama-4-Scout"
        assert restored.dtype == "fp16"

    def test_session_state_from_dict_missing_model_fields(self):
        """from_dict() handles missing model_name/dtype (backward compat)."""
        data = {
            "session_id": "old-session",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
        }
        state = SessionState.from_dict(data)
        assert state.model_name is None
        assert state.dtype is None

    # ---- Tests 4-6: has_report persistence ----

    def test_has_report_survives_serialization_roundtrip(self):
        """Test 4: SessionState with has_report=True survives to_dict/from_dict roundtrip."""
        state = SessionState(
            session_id="report-roundtrip",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            has_report=True,
        )

        data = state.to_dict()
        assert data["has_report"] is True, "to_dict() must include has_report=True"

        restored = SessionState.from_dict(data)
        assert restored.has_report is True, "from_dict() must restore has_report=True"

    def test_has_report_defaults_false_on_missing_key(self):
        """Test 4 (backward compat): from_dict() with missing has_report key defaults to False."""
        data = {
            "session_id": "old-no-report",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
        }
        state = SessionState.from_dict(data)
        assert state.has_report is False, "Missing has_report key must default to False"

    def test_to_session_info_uses_persisted_has_report_true(self):
        """Test 5: to_session_info() returns has_report=True from persisted state, no filesystem needed."""
        state = SessionState(
            session_id="persisted-report",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            terminal_port=8001,
            worktree_path="/nonexistent/path/that/does/not/exist",
            has_report=True,
        )

        info = state.to_session_info()
        assert info.has_report is True, (
            "to_session_info() must return has_report=True when state.has_report is True, "
            "even if worktree path does not exist"
        )

    def test_to_session_info_never_resets_has_report_to_false(self):
        """Test 5 (never reset): Once has_report is True, filesystem check cannot reset it to False."""
        state = SessionState(
            session_id="sticky-latch",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            terminal_port=8001,
            worktree_path="/nonexistent/path",
            has_report=True,
        )

        # Call multiple times — must stay True
        for _ in range(3):
            info = state.to_session_info()
            assert info.has_report is True, "has_report must never be reset to False once True"

    def test_to_session_info_sets_sticky_latch_on_report_discovery(self):
        """Test 6: to_session_info() finds REPORT.md on disk and sets sticky latch on state."""
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create REPORT.md in the temp worktree
            report_path = Path(tmp_dir) / "REPORT.md"
            report_path.write_text("# Report\n\nTest report content.")

            state = SessionState(
                session_id="discover-report",
                status=SessionStatus.ACTIVE,
                cli_tool=CLIToolType.CLAUDE,
                repo_name="vllm",
                branch="main",
                created_at=1234567890.0,
                last_accessed=1234567900.0,
                terminal_port=8001,
                worktree_path=tmp_dir,
                has_report=False,
            )

            assert state.has_report is False, "has_report must start False"

            info = state.to_session_info()

            assert info.has_report is True, (
                "to_session_info() must return has_report=True when REPORT.md found on disk"
            )
            assert state.has_report is True, (
                "to_session_info() must set state.has_report=True (sticky latch) "
                "so subsequent calls don't re-check filesystem"
            )

            # Prove stickiness: delete REPORT.md and call again — must still return True
            report_path.unlink()
            assert not report_path.exists(), "REPORT.md should be deleted"
            info2 = state.to_session_info()
            assert info2.has_report is True, (
                "has_report must remain True after REPORT.md is deleted — "
                "sticky latch prevents re-checking filesystem"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_cross_pod_resume_has_report_preserved_via_roundtrip(self):
        """Verifier edge-case: simulates S3 cross-host S3 restore path.

        has_report=True → to_dict() → from_dict() → to_session_info()
        must return True WITHOUT filesystem access (worktree doesn't exist yet on new pod).
        """
        original = SessionState(
            session_id="cross-pod-test",
            status=SessionStatus.PAUSED,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            worktree_path="/nonexistent/pod-a/worktree",
            has_report=True,
        )
        serialized = original.to_dict()
        assert serialized["has_report"] is True

        # Simulate restoring from S3 on pod B (worktree doesn't exist yet)
        restored = SessionState.from_dict(serialized)
        assert restored.has_report is True, "from_dict() must restore has_report=True"

        info = restored.to_session_info()
        assert info.has_report is True, (
            "to_session_info() on a cross-pod-restored state must return "
            "has_report=True without filesystem access"
        )

    def test_to_session_info_returns_false_when_no_report_on_disk(self):
        """Verifier edge-case: worktree exists but contains no REPORT.md → has_report stays False."""
        tmp_dir = tempfile.mkdtemp()
        try:
            # No REPORT.md created — worktree is empty
            state = SessionState(
                session_id="no-report-disk",
                status=SessionStatus.ACTIVE,
                cli_tool=CLIToolType.CLAUDE,
                repo_name="vllm",
                branch="main",
                created_at=1234567890.0,
                last_accessed=1234567900.0,
                terminal_port=8001,
                worktree_path=tmp_dir,
                has_report=False,
            )

            info = state.to_session_info()
            assert info.has_report is False, (
                "to_session_info() must return has_report=False when "
                "REPORT.md does not exist on disk"
            )
            assert state.has_report is False, (
                "state.has_report must stay False when no REPORT.md found"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.unit
class TestTpDpParallelism:
    """Tests for tp_size/dp_size fields on CreateSessionRequest, SessionState, SessionInfo."""

    def test_create_session_request_tp_dp_defaults(self):
        """Backward compat: no tp/dp in payload → tp_size=None, dp_size=1."""
        req = CreateSessionRequest(gpu_count=0)
        assert req.tp_size is None
        assert req.dp_size == 1

    def test_create_session_request_tp_dp_accepts_values(self):
        """New clients: tp_size * dp_size == gpu_count is accepted."""
        req = CreateSessionRequest(gpu_count=8, tp_size=4, dp_size=2)
        assert req.tp_size == 4
        assert req.dp_size == 2
        assert req.gpu_count == 8

    def test_create_session_request_validator_tp_dp_gpu_mismatch(self):
        """tp_size * dp_size must be <= gpu_count when tp_size is provided.

        Post-decouple: validator allows tp*dp <= gpu_count (minimum, not equality).
        This test asserts the rejection case: tp*dp > gpu_count.
        """
        with pytest.raises(ValidationError):
            # tp*dp = 8, gpu_count = 4 → exceeds minimum → rejected
            CreateSessionRequest(gpu_count=4, tp_size=4, dp_size=2)

    def test_create_session_request_validator_tp_none_no_enforce(self):
        """Backward compat path: tp_size=None skips product-equality check."""
        # Old client: sends only gpu_count, tp_size/dp_size defaults → must not raise
        req = CreateSessionRequest(gpu_count=8)
        assert req.tp_size is None
        assert req.dp_size == 1

    def test_create_session_request_validator_dp_without_tp_rejected(self):
        """dp_size > 1 without explicit tp_size is ambiguous — must reject."""
        with pytest.raises(ValidationError) as exc_info:
            CreateSessionRequest(gpu_count=8, dp_size=4)
        # Error message should reference the missing tp_size field
        assert "tp_size" in str(exc_info.value).lower()

    def test_create_session_request_tp_dp_bounds(self):
        """tp_size and dp_size bounded [1, 8]."""
        # tp_size > 8 rejected
        with pytest.raises(ValidationError):
            CreateSessionRequest(gpu_count=8, tp_size=9, dp_size=1)
        # dp_size < 1 rejected
        with pytest.raises(ValidationError):
            CreateSessionRequest(gpu_count=0, dp_size=0)
        # tp_size < 1 rejected when provided
        with pytest.raises(ValidationError):
            CreateSessionRequest(gpu_count=1, tp_size=0, dp_size=1)

    def test_session_state_tp_dp_roundtrip(self):
        """SessionState with tp_size/dp_size survives to_dict/from_dict roundtrip."""
        original = SessionState(
            session_id="tp-dp-roundtrip",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            tp_size=4,
            dp_size=2,
            requested_gpu_count=8,
        )
        data = original.to_dict()
        assert data["tp_size"] == 4
        assert data["dp_size"] == 2

        restored = SessionState.from_dict(data)
        assert restored.tp_size == 4
        assert restored.dp_size == 2

    def test_session_state_from_dict_missing_tp_dp_legacy(self):
        """Old S3 state (no tp_size/dp_size keys) → tp_size falls back to requested_gpu_count,
        dp_size defaults to 1."""
        data = {
            "session_id": "old-session",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
            "requested_gpu_count": 8,
            # No tp_size / dp_size keys
        }
        state = SessionState.from_dict(data)
        assert state.tp_size == 8, (
            "Legacy sessions must resolve tp_size to requested_gpu_count per spec §2.2"
        )
        assert state.dp_size == 1

    def test_session_state_from_dict_legacy_zero_sentinel(self):
        """tp_size=0 in dict (sentinel) → fallback to requested_gpu_count."""
        data = {
            "session_id": "zero-sentinel",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
            "tp_size": 0,
            "dp_size": 1,
            "requested_gpu_count": 4,
        }
        state = SessionState.from_dict(data)
        assert state.tp_size == 4
        assert state.dp_size == 1

    def test_session_state_from_dict_legacy_no_requested_gpu_count(self):
        """Very old sessions (no tp_size AND no requested_gpu_count) → tp_size=0 (unknown)."""
        data = {
            "session_id": "very-old",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 1234567890.0,
            "last_accessed": 1234567900.0,
        }
        state = SessionState.from_dict(data)
        assert state.tp_size == 0
        assert state.dp_size == 1

    def test_session_info_accepts_tp_dp(self):
        """SessionInfo accepts tp_size and dp_size as Optional[int]."""
        info = SessionInfo(
            session_id="s-info",
            status="active",
            cli_tool="claude",
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            tp_size=4,
            dp_size=2,
        )
        assert info.tp_size == 4
        assert info.dp_size == 2

    def test_session_info_tp_dp_optional(self):
        """SessionInfo tp/dp fields default to None when not provided (legacy test compat)."""
        info = SessionInfo(
            session_id="s-info-legacy",
            status="active",
            cli_tool="claude",
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
        )
        assert info.tp_size is None
        assert info.dp_size is None

    def test_to_session_info_passes_tp_dp(self):
        """to_session_info() forwards tp_size/dp_size to SessionInfo."""
        state = SessionState(
            session_id="info-tp-dp",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=1234567890.0,
            last_accessed=1234567900.0,
            tp_size=4,
            dp_size=2,
        )
        info = state.to_session_info()
        assert info.tp_size == 4
        assert info.dp_size == 2


@pytest.mark.unit
class TestGpuCountDecoupling:
    """Spec: decouple gpu_count from tp*dp.

    Post-decouple semantics (see .claude/plans/gpu-decouple.md):
    - Validator enforces tp_size * dp_size <= gpu_count (a MINIMUM, not equality).
    - `gpu_count` may exceed tp*dp so extra GPUs can be used for parallel
      experiment tracks inside the session.
    - Legacy clients sending only `gpu_count` (no tp/dp) continue to work.
    """

    def test_gpu_count_equals_tp_times_dp_ok(self):
        """Baseline: gpu_count == tp*dp continues to be accepted."""
        req = CreateSessionRequest(gpu_count=4, tp_size=2, dp_size=2)
        assert req.gpu_count == 4
        assert req.tp_size == 2
        assert req.dp_size == 2

    def test_gpu_count_greater_than_tp_times_dp_ok(self):
        """Post-decouple: gpu_count > tp*dp is allowed (spare GPUs for parallel tracks)."""
        req = CreateSessionRequest(gpu_count=8, tp_size=2, dp_size=2)
        assert req.gpu_count == 8
        assert req.tp_size == 2
        assert req.dp_size == 2

    def test_gpu_count_less_than_tp_times_dp_rejected(self):
        """tp_size * dp_size cannot exceed gpu_count."""
        with pytest.raises(ValidationError) as exc_info:
            CreateSessionRequest(gpu_count=2, tp_size=2, dp_size=2)
        msg = str(exc_info.value).lower()
        assert "tp_size" in msg or "gpu_count" in msg
        # Error message should reference the relationship
        assert "gpu_count" in msg

    def test_legacy_payload_without_tp_size_still_accepted(self):
        """Legacy client sends only gpu_count → tp_size=None, dp_size=1, no raise."""
        req = CreateSessionRequest(gpu_count=8)
        assert req.gpu_count == 8
        assert req.tp_size is None
        assert req.dp_size == 1

    def test_dp_without_tp_still_rejected(self):
        """dp_size > 1 without explicit tp_size remains ambiguous — rejected."""
        with pytest.raises(ValidationError):
            CreateSessionRequest(gpu_count=8, dp_size=4)

    def test_session_state_tp_size_preserved_when_gpu_count_greater(self):
        """SessionState keeps tp_size independent from requested_gpu_count."""
        state = SessionState(
            session_id="decoupled",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=0.0,
            last_accessed=0.0,
            tp_size=2,
            dp_size=2,
            requested_gpu_count=8,
        )
        assert state.tp_size == 2  # not collapsed to 8
        assert state.dp_size == 2
        assert state.requested_gpu_count == 8

    def test_session_state_from_dict_tp_legacy_fallback_prefers_explicit_tp(self):
        """When tp_size is present in dict, from_dict must NOT fall back to requested_gpu_count."""
        data = {
            "session_id": "explicit-tp",
            "status": "active",
            "cli_tool": "claude",
            "repo_name": "vllm",
            "branch": "main",
            "created_at": 0.0,
            "last_accessed": 0.0,
            "tp_size": 2,
            "dp_size": 2,
            "requested_gpu_count": 8,
        }
        state = SessionState.from_dict(data)
        assert state.tp_size == 2, (
            "Explicit tp_size in dict must take precedence over requested_gpu_count fallback"
        )
        assert state.dp_size == 2
        assert state.requested_gpu_count == 8

    def test_session_state_roundtrip_gpu_count_decoupled(self):
        """to_dict → from_dict preserves tp_size, dp_size, requested_gpu_count independently."""
        original = SessionState(
            session_id="roundtrip-decoupled",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=0.0,
            last_accessed=0.0,
            tp_size=2,
            dp_size=2,
            requested_gpu_count=8,
        )
        data = original.to_dict()
        assert data["tp_size"] == 2
        assert data["dp_size"] == 2
        assert data["requested_gpu_count"] == 8

        restored = SessionState.from_dict(data)
        assert restored.tp_size == 2
        assert restored.dp_size == 2
        assert restored.requested_gpu_count == 8
