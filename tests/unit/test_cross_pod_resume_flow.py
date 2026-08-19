# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for cross-host S3 restore flow orchestration.

Tests the full resume_session() flow in SessionManager:
- S3 restore, GPU reacquisition, CLI configuration, --continue flag logic
- S3 format detection in restore_worktree_from_s3
- discover_s3_sessions behavior
"""

import json
import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
)
from orchestration.session_manager import SessionManager, SessionError
from orchestration.session_state import SessionS3Storage

# Import shared fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))
from fixtures.session_fixtures import (
    reset_all_singletons,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
    mock_session_manager,
    make_session_state,
)


def _make_paused_state(
    session_id: str = "sess-cross-pod-001",
    worktree_path: str = "/data/sessions/sess-cross-pod-001/worktree",
    session_dir: str = "/data/sessions/sess-cross-pod-001",
    requested_gpu_count: int = 1,
    status: SessionStatus = SessionStatus.PAUSED,
    **kwargs,
) -> SessionState:
    defaults = dict(
        session_id=session_id,
        status=status,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=worktree_path,
        session_dir=session_dir,
        requested_gpu_count=requested_gpu_count,
        gpu_ids=[0],
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


# ============================================================================
# Test Group 1: Resume from S3 — Full Flow
# ============================================================================


@pytest.mark.unit
class TestResumeFromS3:
    """Tests for full S3-based resume orchestration."""

    @pytest.mark.asyncio
    async def test_resume_from_s3_full_flow(
        self,
        mock_session_manager,
        mock_session_storage,
        tmp_path,
    ):
        """No local state -> restore from S3 -> start terminal -> session ACTIVE."""
        session_id = "sess-s3-restore-001"
        worktree_path = mock_session_manager.sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        restored_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(mock_session_manager.sessions_dir / session_id),
        )

        async def mock_restore(*args, **kwargs):
            # Simulate S3 restore creating worktree on disk
            worktree_path.mkdir(parents=True, exist_ok=True)
            return restored_state

        # Enable S3 storage and make it return a restored state
        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(
            side_effect=mock_restore
        )

        # Session is not in local memory
        assert session_id not in mock_session_manager._sessions

        response = await mock_session_manager.resume_session(session_id)

        assert response is not None
        mock_session_storage.restore_session_from_s3.assert_called_once()
        # Session should now be in local memory and ACTIVE
        assert session_id in mock_session_manager._sessions
        assert mock_session_manager._sessions[session_id].status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_resume_from_s3_missing_metadata(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """S3 returns None (no metadata) -> SessionError raised."""
        session_id = "sess-s3-no-meta-001"

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(return_value=None)

        with pytest.raises(SessionError, match="not found"):
            await mock_session_manager.resume_session(session_id)

    @pytest.mark.asyncio
    async def test_resume_from_s3_corrupted_tar(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """restore_session_from_s3 raises an exception -> SessionError, session stays PAUSED."""
        session_id = "sess-s3-corrupt-001"

        # Pre-populate with a PAUSED session that has no local worktree
        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path="/nonexistent/path/worktree",
            session_dir="/nonexistent/path",
        )
        mock_session_manager._sessions[session_id] = paused_state

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(
            side_effect=RuntimeError("tar extraction failed: corrupt archive")
        )

        with pytest.raises(SessionError):
            await mock_session_manager.resume_session(session_id)

    @pytest.mark.asyncio
    async def test_resume_forces_s3_restore_when_s3_is_newer_than_local_worktree(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """S3 LastModified > local s3_last_sync triggers S3 restore."""
        session_id = "sess-stale-resume-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Session in memory with stale local data
        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        paused_state.s3_synced = True
        paused_state.s3_last_sync = 2000.0
        paused_state.last_accessed = 1000.0
        mock_session_manager._sessions[session_id] = paused_state

        # S3 restore returns a valid state and recreates the worktree directory
        restored_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )

        async def mock_restore(*args, **kwargs):
            # Simulate S3 restore creating worktree on disk
            worktree_path.mkdir(parents=True, exist_ok=True)
            return restored_state

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(side_effect=mock_restore)
        # S3 has a newer worktree than our local s3_last_sync=2000.0
        mock_session_storage.get_s3_last_modified = AsyncMock(return_value=3000.0)

        fix_spy = MagicMock()
        mock_session_manager._fix_claude_project_dir_after_s3_restore = fix_spy

        await mock_session_manager.resume_session(session_id)

        # S3 restore should be triggered despite local worktree existing
        mock_session_storage.restore_session_from_s3.assert_called_once()
        # _fix_claude_project_dir called because restored_from_s3=True
        fix_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_skips_s3_restore_when_local_is_current(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Local worktree is current (last_accessed > s3_last_sync) -> no S3 restore."""
        session_id = "sess-current-local-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        paused_state.s3_synced = True
        paused_state.s3_last_sync = 1000.0
        paused_state.last_accessed = 2000.0
        mock_session_manager._sessions[session_id] = paused_state

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(return_value=None)

        fix_spy = MagicMock()
        mock_session_manager._fix_claude_project_dir_after_s3_restore = fix_spy

        response = await mock_session_manager.resume_session(session_id)

        # S3 restore NOT called
        mock_session_storage.restore_session_from_s3.assert_not_called()
        # _fix_claude_project_dir NOT called
        fix_spy.assert_not_called()
        # Session should be ACTIVE
        assert mock_session_manager._sessions[session_id].status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_resume_skips_s3_restore_when_not_s3_synced(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Session with s3_synced=False -> no S3 restore, resumes with local worktree."""
        session_id = "sess-no-s3-sync-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        paused_state.s3_synced = False
        paused_state.s3_last_sync = None
        mock_session_manager._sessions[session_id] = paused_state

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(return_value=None)

        response = await mock_session_manager.resume_session(session_id)

        # S3 restore NOT called
        mock_session_storage.restore_session_from_s3.assert_not_called()
        # Session should be ACTIVE
        assert mock_session_manager._sessions[session_id].status == SessionStatus.ACTIVE


# ============================================================================
# Test Group 2: GPU Reacquisition During Resume
# ============================================================================


@pytest.mark.unit
class TestResumeGPUReacquisition:
    """Tests for GPU reacquisition during session resume."""

    @pytest.mark.asyncio
    async def test_resume_reacquires_gpus_with_original_count(
        self,
        mock_session_manager,
        tmp_path,
    ):
        """Resume uses requested_gpu_count from stored state to reacquire GPUs."""
        session_id = "sess-gpu-reacquire-001"
        worktree_path = mock_session_manager.sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(mock_session_manager.sessions_dir / session_id),
            requested_gpu_count=2,
            gpu_ids=[0, 1],
        )
        mock_session_manager._sessions[session_id] = paused_state

        acquire_spy = AsyncMock(return_value=[2, 3])
        mock_session_manager.gpu_manager.acquire_gpus_for_session_async = acquire_spy

        await mock_session_manager.resume_session(session_id)

        acquire_spy.assert_called_once_with(session_id=session_id, gpu_count=2)

    @pytest.mark.asyncio
    async def test_resume_gpu_ids_updated_to_new_pod(
        self,
        mock_session_manager,
        tmp_path,
    ):
        """After GPU reacquisition, state.gpu_ids reflects new pod's GPU IDs."""
        session_id = "sess-gpu-update-001"
        worktree_path = mock_session_manager.sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Original pod had GPU IDs [0, 1]
        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(mock_session_manager.sessions_dir / session_id),
            requested_gpu_count=2,
            gpu_ids=[0, 1],
        )
        mock_session_manager._sessions[session_id] = paused_state

        # New pod allocates [2, 3]
        mock_session_manager.gpu_manager.acquire_gpus_for_session_async = AsyncMock(
            return_value=[2, 3]
        )

        await mock_session_manager.resume_session(session_id)

        active_state = mock_session_manager._sessions[session_id]
        assert active_state.gpu_ids == [2, 3]

    @pytest.mark.asyncio
    async def test_resume_gpu_failure_rolls_back(
        self,
        mock_session_manager,
        tmp_path,
    ):
        """GPU acquisition timeout -> GPUs released, SessionError raised."""
        session_id = "sess-gpu-fail-001"
        worktree_path = mock_session_manager.sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(mock_session_manager.sessions_dir / session_id),
            requested_gpu_count=1,
        )
        mock_session_manager._sessions[session_id] = paused_state

        mock_session_manager.gpu_manager.acquire_gpus_for_session_async = AsyncMock(
            side_effect=TimeoutError("No GPU available within timeout")
        )
        release_spy = MagicMock()
        mock_session_manager.gpu_manager.release_gpus_for_session = release_spy

        with pytest.raises(SessionError):
            await mock_session_manager.resume_session(session_id)

        release_spy.assert_called_once_with(session_id)


# ============================================================================
# Test Group 3: CLI Configuration During Resume
# ============================================================================


@pytest.mark.unit
class TestResumeCLIConfiguration:
    """Tests for CLI tool configuration during resume, especially --continue logic."""

    @pytest.mark.asyncio
    async def test_resume_calls_fix_claude_project_dir(
        self,
        mock_session_manager,
        mock_session_storage,
        tmp_path,
    ):
        """Cross-host S3 restore -> _fix_claude_project_dir_after_s3_restore is called."""
        session_id = "sess-fix-proj-dir-001"
        worktree_path = mock_session_manager.sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        restored_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(mock_session_manager.sessions_dir / session_id),
            requested_gpu_count=0,
        )

        async def mock_restore(*args, **kwargs):
            # Simulate S3 restore creating worktree on disk
            worktree_path.mkdir(parents=True, exist_ok=True)
            return restored_state

        mock_session_storage.enabled = True
        mock_session_storage.restore_session_from_s3 = AsyncMock(
            side_effect=mock_restore
        )

        fix_spy = MagicMock()
        mock_session_manager._fix_claude_project_dir_after_s3_restore = fix_spy

        await mock_session_manager.resume_session(session_id)

        fix_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_uses_continue_flag(
        self,
        mock_session_manager,
        mock_cli_tool_manager,
        tmp_path,
    ):
        """Project dir with .jsonl conversation data -> CLI command includes is_resume=True."""
        session_id = "sess-continue-yes-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Create projects dir with a .jsonl file to trigger --continue
        proj_dir = session_dir / "claude-config" / "projects" / "-some-project"
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "conversation.jsonl").write_text("{}")

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        mock_session_manager._sessions[session_id] = paused_state

        await mock_session_manager.resume_session(session_id)

        mock_cli_tool_manager.get_cli_command.assert_called_once()
        call_kwargs = mock_cli_tool_manager.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is True

    @pytest.mark.asyncio
    async def test_resume_skips_continue_when_no_conversation(
        self,
        mock_session_manager,
        mock_cli_tool_manager,
        tmp_path,
    ):
        """Empty claude-config/projects dir -> CLI command uses is_resume=False."""
        session_id = "sess-continue-no-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Create empty projects dir (no conversation data)
        projects_dir = session_dir / "claude-config" / "projects"
        projects_dir.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        mock_session_manager._sessions[session_id] = paused_state

        await mock_session_manager.resume_session(session_id)

        mock_cli_tool_manager.get_cli_command.assert_called_once()
        call_kwargs = mock_cli_tool_manager.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is False

    @pytest.mark.asyncio
    async def test_resume_skips_continue_when_only_corrupted_dirs(
        self,
        mock_session_manager,
        mock_cli_tool_manager,
        tmp_path,
    ):
        """Bug 3 fix: corrupted dirs (git stdout, no .jsonl) -> is_resume=False."""
        session_id = "sess-continue-corrupt-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Create corrupted project dirs (git stdout as dirname, no .jsonl inside)
        projects_dir = session_dir / "claude-config" / "projects"
        corrupted = projects_dir / "Preparing-worktree--new-branch--session-fa88376f---HEAD-is-now-at-91eea7233"
        corrupted.mkdir(parents=True, exist_ok=True)
        # Also an empty agent worktree dir (no .jsonl)
        agent_dir = projects_dir / "-data-sessions-test--claude-worktrees-op001"
        agent_dir.mkdir(parents=True, exist_ok=True)

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        mock_session_manager._sessions[session_id] = paused_state

        await mock_session_manager.resume_session(session_id)

        mock_cli_tool_manager.get_cli_command.assert_called_once()
        call_kwargs = mock_cli_tool_manager.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is False

    @pytest.mark.asyncio
    async def test_resume_uses_continue_when_valid_jsonl_among_corrupted(
        self,
        mock_session_manager,
        mock_cli_tool_manager,
        tmp_path,
    ):
        """Bug 3 fix: valid dir with .jsonl among corrupted dirs -> is_resume=True."""
        session_id = "sess-continue-mixed-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        projects_dir = session_dir / "claude-config" / "projects"
        # Corrupted dir (no .jsonl)
        (projects_dir / "Preparing-worktree--HEAD-is-now-at-abc123").mkdir(parents=True)
        # Valid dir with .jsonl
        valid_dir = projects_dir / f"-data-sessions-{session_id}-worktree"
        valid_dir.mkdir(parents=True, exist_ok=True)
        (valid_dir / "session.jsonl").write_text("{}")

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        mock_session_manager._sessions[session_id] = paused_state

        await mock_session_manager.resume_session(session_id)

        mock_cli_tool_manager.get_cli_command.assert_called_once()
        call_kwargs = mock_cli_tool_manager.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is True

    @pytest.mark.asyncio
    async def test_resume_skips_continue_when_dirs_have_non_jsonl_files_only(
        self,
        mock_session_manager,
        mock_cli_tool_manager,
        tmp_path,
    ):
        """Dirs with .json but no .jsonl -> is_resume=False."""
        session_id = "sess-continue-nojsonl-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Dir with .json but no .jsonl
        proj_dir = session_dir / "claude-config" / "projects" / "-some-project"
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "conversation.json").write_text("{}")

        paused_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            requested_gpu_count=0,
        )
        mock_session_manager._sessions[session_id] = paused_state

        await mock_session_manager.resume_session(session_id)

        mock_cli_tool_manager.get_cli_command.assert_called_once()
        call_kwargs = mock_cli_tool_manager.get_cli_command.call_args
        assert call_kwargs.kwargs.get("is_resume") is False


# ============================================================================
# Test Group 4: S3 Format Detection
# ============================================================================


@pytest.mark.unit
class TestS3FormatDetection:
    """Tests for tar.gz vs legacy per-file format detection in restore_worktree_from_s3."""

    def _make_storage(self) -> SessionS3Storage:
        storage = SessionS3Storage.__new__(SessionS3Storage)
        storage.bucket = "test-bucket"
        storage.prefix = "sessions"
        storage.ttl_days = 30
        return storage

    @pytest.mark.asyncio
    async def test_s3_format_detection_prefers_tar(self, tmp_path):
        """When tar.gz exists in S3, uses _restore_worktree_from_tar path."""
        storage = self._make_storage()

        storage._tar_gz_exists_in_s3 = AsyncMock(return_value=True)
        storage._restore_worktree_from_tar = AsyncMock(return_value=True)
        storage._restore_worktree_from_sync = AsyncMock(return_value=True)

        result = await storage.restore_worktree_from_s3(
            session_id="test-sess-001",
            target_path=tmp_path / "worktree",
        )

        assert result is True
        storage._restore_worktree_from_tar.assert_called_once()
        storage._restore_worktree_from_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_s3_format_detection_falls_back_to_sync(self, tmp_path):
        """When no tar.gz exists in S3, falls back to legacy sync path."""
        storage = self._make_storage()

        storage._tar_gz_exists_in_s3 = AsyncMock(return_value=False)
        storage._restore_worktree_from_tar = AsyncMock(return_value=True)
        storage._restore_worktree_from_sync = AsyncMock(return_value=True)

        result = await storage.restore_worktree_from_s3(
            session_id="test-sess-002",
            target_path=tmp_path / "worktree",
        )

        assert result is True
        storage._restore_worktree_from_sync.assert_called_once()
        storage._restore_worktree_from_tar.assert_not_called()


# ============================================================================
# Test Group 5: S3 Session Discovery
# ============================================================================


@pytest.mark.unit
class TestS3SessionDiscovery:
    """Tests for discover_s3_sessions() in SessionManager."""

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_skips_local(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Sessions already in _sessions are kept when S3 timestamps match."""
        session_id = "sess-already-local-001"
        local_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=1000.0,
        )
        mock_session_manager._sessions[session_id] = local_state

        s3_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=1000.0,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 0
        # load_session_metadata IS called now (for timestamp comparison) but local state kept
        mock_session_storage.load_session_metadata.assert_called_once()
        # Local state unchanged
        assert mock_session_manager._sessions[session_id].s3_last_sync == 1000.0

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_replaces_stale_local_with_newer_s3(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Local session with older s3_last_sync is replaced by newer S3 state."""
        session_id = "sess-stale-local-001"
        session_dir = mock_session_manager.sessions_dir / session_id
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        local_state = _make_paused_state(
            session_id=session_id,
            worktree_path=str(worktree_path),
            session_dir=str(session_dir),
            s3_synced=True,
            s3_last_sync=1000.0,
        )
        local_state.last_accessed = 1000.0
        mock_session_manager._sessions[session_id] = local_state

        s3_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=2000.0,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        # Patch _save_session_state to avoid file I/O
        mock_session_manager._save_session_state = MagicMock()

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 1
        mock_session_storage.load_session_metadata.assert_called_once()
        # s3_last_sync is NOT reset in discovery — the local worktree may be a
        # stale snapshot inherited from a prior pod on the same node via
        # hostPath, so we must keep the S3 metadata timestamp to let the next
        # resume_session see S3 as newer via get_s3_last_modified and trigger
        # restore_session_from_s3 (which DOES reset s3_last_sync after
        # extracting the tar). Retaining 2000.0 is correct here.
        assert mock_session_manager._sessions[session_id].s3_last_sync == 2000.0
        assert mock_session_manager._sessions[session_id].status == SessionStatus.PAUSED
        # Stale local worktree directory should be removed
        assert not worktree_path.exists()

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_keeps_local_when_s3_is_older(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Local session with newer s3_last_sync is kept over older S3 state."""
        session_id = "sess-newer-local-001"
        local_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=2000.0,
        )
        mock_session_manager._sessions[session_id] = local_state

        s3_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=1000.0,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 0
        assert mock_session_manager._sessions[session_id].s3_last_sync == 2000.0

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_keeps_local_when_no_s3_sync_timestamp(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Local session kept when S3 state has s3_last_sync=None."""
        session_id = "sess-no-s3-ts-001"
        local_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=1000.0,
        )
        mock_session_manager._sessions[session_id] = local_state

        s3_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=None,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 0
        assert mock_session_manager._sessions[session_id].s3_last_sync == 1000.0

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_keeps_local_when_both_have_no_sync_timestamp(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """Both local and S3 have s3_last_sync=None -> keep local."""
        session_id = "sess-both-no-ts-001"
        local_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=None,
        )
        mock_session_manager._sessions[session_id] = local_state

        s3_state = _make_paused_state(
            session_id=session_id,
            s3_synced=True,
            s3_last_sync=None,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 0

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_marks_active_as_paused(
        self,
        mock_session_manager,
        mock_session_storage,
        tmp_path,
    ):
        """Sessions with ACTIVE status in S3 are loaded as PAUSED (not running here)."""
        session_id = "sess-active-in-s3-001"

        # S3 metadata has ACTIVE status
        s3_state = _make_paused_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(return_value=s3_state)

        # Patch _save_session_state to avoid file I/O
        mock_session_manager._save_session_state = MagicMock()

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 1
        loaded = mock_session_manager._sessions[session_id]
        assert loaded.status == SessionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_discover_s3_sessions_skips_terminated(
        self,
        mock_session_manager,
        mock_session_storage,
    ):
        """TERMINATED sessions in S3 are skipped (not loaded into memory)."""
        session_id = "sess-terminated-s3-001"

        terminated_state = _make_paused_state(
            session_id=session_id,
            status=SessionStatus.TERMINATED,
        )

        mock_session_storage.enabled = True
        mock_session_storage.list_s3_sessions = AsyncMock(return_value=[session_id])
        mock_session_storage.load_session_metadata = AsyncMock(
            return_value=terminated_state
        )

        mock_session_manager._save_session_state = MagicMock()

        discovered = await mock_session_manager.discover_s3_sessions()

        assert discovered == 0
        assert session_id not in mock_session_manager._sessions
