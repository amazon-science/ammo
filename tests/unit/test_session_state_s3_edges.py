# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for S3 persistence edge cases in SessionS3Storage.

Tests edge cases for:
- save_session_metadata / load_session_metadata (disabled/not found)
- sync_session_to_s3 parallel gather
- restore format detection
- cleanup_stale_sessions
- create_download_archive (exclude patterns)
- get_download_url (presigned URL generation)
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
from orchestration.session_state import SessionS3Storage

# Import shared fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))
from fixtures.session_fixtures import (
    reset_all_singletons,
    mock_session_storage,
    make_session_state,
)


def _make_storage(bucket: str = "test-bucket", prefix: str = "sessions") -> SessionS3Storage:
    """Create a SessionS3Storage instance bypassing __init__ env vars."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = bucket
    storage.prefix = prefix
    storage.ttl_days = 30
    return storage


def _make_disabled_storage() -> SessionS3Storage:
    """Create a SessionS3Storage with S3 disabled (no bucket)."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = None
    storage.prefix = "sessions"
    storage.ttl_days = 30
    return storage


def _make_session_state(
    session_id: str = "test-sess-001",
    worktree_path: str = "/data/sessions/test-sess-001/worktree",
    status: SessionStatus = SessionStatus.PAUSED,
) -> SessionState:
    return SessionState(
        session_id=session_id,
        status=status,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=worktree_path,
    )


# ============================================================================
# Test Group 1: Metadata Operations
# ============================================================================


@pytest.mark.unit
class TestS3MetadataOperations:
    """Tests for save/load session metadata with S3 disabled or missing key."""

    @pytest.mark.asyncio
    async def test_save_metadata_disabled_returns_false(self):
        """When S3 not configured (no bucket), save_session_metadata returns False."""
        storage = _make_disabled_storage()
        assert not storage.enabled

        state = _make_session_state()
        result = await storage.save_session_metadata(state)

        assert result is False

    @pytest.mark.asyncio
    async def test_load_metadata_not_found(self):
        """When S3 key doesn't exist (returncode != 0, 'not found' in stderr), returns None."""
        storage = _make_storage()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"An error occurred (NoSuchKey) when calling the HeadObject operation: Not Found")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ):
            result = await storage.load_session_metadata("nonexistent-session")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_metadata_disabled_returns_none(self):
        """When S3 not configured, load_session_metadata returns None immediately."""
        storage = _make_disabled_storage()
        result = await storage.load_session_metadata("any-session-id")
        assert result is None


# ============================================================================
# Test Group 2: Sync Operations
# ============================================================================


@pytest.mark.unit
class TestS3SyncOperations:
    """Tests for worktree sync and session sync operations."""

    @pytest.mark.asyncio
    async def test_sync_worktree_includes_claude_config(self, tmp_path):
        """
        sync_worktree_to_s3 includes claude-config dir in the tar command
        when the directory exists.
        """
        storage = _make_storage()

        session_dir = tmp_path / "sess-001"
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Create claude-config dir (should be included in tar)
        claude_config = session_dir / "claude-config"
        claude_config.mkdir(parents=True, exist_ok=True)

        state = _make_session_state(
            session_id="sess-001",
            worktree_path=str(worktree_path),
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        captured_cmd = []

        async def capture_exec(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            # Also patch cleanup to avoid extra subprocess calls
            storage._cleanup_old_per_file_objects = AsyncMock()
            result = await storage.sync_worktree_to_s3(state)

        assert result is True
        # The shell command (passed to bash -c) should mention claude-config
        assert len(captured_cmd) >= 3
        shell_cmd = captured_cmd[2]  # bash -c <cmd>
        assert "claude-config" in shell_cmd

    @pytest.mark.asyncio
    async def test_sync_session_runs_parallel(self, tmp_path):
        """
        sync_session_to_s3 uses asyncio.gather to run metadata, worktree,
        and cli_state syncs in parallel.
        """
        storage = _make_storage()

        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)
        state = _make_session_state(
            session_id="sess-parallel-001",
            worktree_path=str(worktree_path),
        )

        call_order = []

        async def mock_save_metadata(s):
            call_order.append("metadata")
            return True

        async def mock_sync_worktree(s, **kwargs):
            call_order.append("worktree")
            return True

        async def mock_sync_cli(s):
            call_order.append("cli")
            return True

        storage.save_session_metadata = mock_save_metadata
        storage.sync_worktree_to_s3 = mock_sync_worktree
        storage.sync_cli_state_to_s3 = mock_sync_cli

        import asyncio
        gather_calls = []
        original_gather = asyncio.gather

        async def spy_gather(*coros, **kwargs):
            gather_calls.append(len(coros))
            return await original_gather(*coros, **kwargs)

        with patch("asyncio.gather", side_effect=spy_gather):
            result = await storage.sync_session_to_s3(state)

        assert result is True
        # asyncio.gather should have been called with 3 coroutines
        assert any(n == 3 for n in gather_calls)

    @pytest.mark.asyncio
    async def test_restore_detects_tar_format(self, tmp_path):
        """
        restore_worktree_from_s3 checks for tar.gz before choosing restore method.
        When tar.gz present: uses _restore_worktree_from_tar.
        When absent: uses _restore_worktree_from_sync.
        """
        storage = _make_storage()

        # Scenario A: tar.gz present
        storage._tar_gz_exists_in_s3 = AsyncMock(return_value=True)
        storage._restore_worktree_from_tar = AsyncMock(return_value=True)
        storage._restore_worktree_from_sync = AsyncMock(return_value=True)

        await storage.restore_worktree_from_s3("sess-tar-001", tmp_path / "wt")
        storage._restore_worktree_from_tar.assert_called_once()
        storage._restore_worktree_from_sync.assert_not_called()

        # Reset
        storage._restore_worktree_from_tar.reset_mock()
        storage._restore_worktree_from_sync.reset_mock()

        # Scenario B: no tar.gz
        storage._tar_gz_exists_in_s3 = AsyncMock(return_value=False)

        await storage.restore_worktree_from_s3("sess-sync-001", tmp_path / "wt2")
        storage._restore_worktree_from_sync.assert_called_once()
        storage._restore_worktree_from_tar.assert_not_called()


# ============================================================================
# Test Group 3: Download and Cleanup
# ============================================================================


@pytest.mark.unit
class TestS3DownloadAndCleanup:
    """Tests for cleanup_stale_sessions, download archive, and presigned URLs."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self):
        """Sessions older than TTL days are deleted; fresh ones are kept."""
        storage = _make_storage()

        stale_id = "sess-stale-001"
        fresh_id = "sess-fresh-001"

        stale_state = _make_session_state(session_id=stale_id)
        stale_state.last_accessed = time.time() - (35 * 24 * 3600)  # 35 days ago

        fresh_state = _make_session_state(session_id=fresh_id)
        fresh_state.last_accessed = time.time() - (2 * 24 * 3600)  # 2 days ago

        storage.list_s3_sessions = AsyncMock(return_value=[stale_id, fresh_id])
        storage.load_session_metadata = AsyncMock(
            side_effect=lambda sid: {
                stale_id: stale_state,
                fresh_id: fresh_state,
            }.get(sid)
        )
        delete_mock = AsyncMock(return_value=True)
        storage.delete_session_from_s3 = delete_mock

        cleaned = await storage.cleanup_stale_sessions(max_age_days=30)

        assert cleaned == 1
        delete_mock.assert_called_once_with(stale_id)

    @pytest.mark.asyncio
    async def test_download_archive_excludes_venv(self, tmp_path):
        """
        create_download_archive passes exclude flags for .venv/, venv/, node_modules/
        to aws s3 sync command.
        """
        storage = _make_storage()

        session_id = "sess-dl-001"
        temp_download_dir = tmp_path / "download_work"
        temp_download_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy file so the directory is not considered empty
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        captured_sync_cmd = []

        mock_proc_sync = MagicMock()
        mock_proc_sync.returncode = 0
        mock_proc_sync.communicate = AsyncMock(return_value=(b"", b""))

        mock_proc_upload = MagicMock()
        mock_proc_upload.returncode = 0
        mock_proc_upload.communicate = AsyncMock(return_value=(b"", b""))

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is the sync
                captured_sync_cmd.extend(args)
                return mock_proc_sync
            else:
                # Second call is the upload
                return mock_proc_upload

        import tempfile
        import shutil as real_shutil

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive") as mock_archive, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=1024), \
             patch("shutil.rmtree"), \
             patch("os.unlink"):

            mock_archive.return_value = str(temp_download_dir) + ".zip"

            result = await storage.create_download_archive(session_id)

        # Should have captured the sync command
        assert len(captured_sync_cmd) > 0
        # Check that venv exclusion patterns are present
        sync_args = " ".join(str(a) for a in captured_sync_cmd)
        assert ".venv" in sync_args or "venv" in sync_args
        assert "node_modules" in sync_args

    @pytest.mark.asyncio
    async def test_presigned_url_generation(self):
        """get_download_url calls 'aws s3 presign' with correct S3 URI and expiry."""
        storage = _make_storage()
        session_id = "sess-presign-001"
        expected_url = "https://test-bucket.s3.amazonaws.com/sessions/sess-presign-001/download/session_archive.zip?X-Amz-Signature=abc"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(expected_url.encode(), b"")
        )

        captured_args = []

        async def capture_exec(*args, **kwargs):
            captured_args.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            url = await storage.get_download_url(session_id, expires_in=7200)

        assert url == expected_url

        # Verify 'aws s3 presign' was called with the right S3 URI
        args_str = " ".join(str(a) for a in captured_args)
        assert "aws" in args_str
        assert "presign" in args_str
        assert session_id in args_str
        assert "session_archive.zip" in args_str
        assert "7200" in args_str

    @pytest.mark.asyncio
    async def test_presigned_url_returns_none_when_disabled(self):
        """When S3 not configured, get_download_url returns None without calling aws."""
        storage = _make_disabled_storage()
        result = await storage.get_download_url("any-session")
        assert result is None
