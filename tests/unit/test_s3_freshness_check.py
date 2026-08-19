# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for S3 freshness check feature.

Tests:
1. get_s3_last_modified returns epoch float for existing object
2. get_s3_last_modified returns None when object does not exist
3. get_s3_last_modified returns None when S3 storage disabled
4. resume detects stale worktree via S3 HeadObject
5. resume skips S3 restore when local is current
6. pause skips S3 overwrite when S3 is newer
7. pause proceeds with S3 sync when local is current
8. checkpoint skips upload when S3 is newer
9. resume with S3 storage disabled falls back to local-only
"""

import asyncio
import json
import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
)
from orchestration.session_state import SessionS3Storage
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


# ============================================================================
# Test Group 1: get_s3_last_modified()
# ============================================================================


@pytest.mark.unit
class TestGetS3LastModified:
    """Tests for SessionS3Storage.get_s3_last_modified()."""

    @pytest.mark.asyncio
    async def test_returns_epoch_float_for_existing_object(self):
        """get_s3_last_modified returns correct epoch float from HeadObject JSON."""
        storage = _make_storage()
        session_id = "sess-fresh-001"

        last_modified_str = "2026-04-13T22:52:30+00:00"
        head_object_json = json.dumps({
            "LastModified": last_modified_str,
            "ContentLength": 12451879839,
        }).encode()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(head_object_json, b""))

        captured_cmd = []

        async def capture_exec(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            result = await storage.get_s3_last_modified(session_id)

        # Verify correct epoch float
        from datetime import datetime
        expected_ts = datetime.fromisoformat(last_modified_str).timestamp()
        assert result == pytest.approx(expected_ts)

        # Verify correct aws s3api head-object command was issued
        args_str = " ".join(str(a) for a in captured_cmd)
        assert "aws" in args_str
        assert "s3api" in args_str
        assert "head-object" in args_str
        assert "test-bucket" in args_str
        assert session_id in args_str
        assert "worktree.tar.gz" in args_str

    @pytest.mark.asyncio
    async def test_returns_none_when_object_does_not_exist(self):
        """get_s3_last_modified returns None when HeadObject exits with 254 (not found)."""
        storage = _make_storage()

        mock_proc = MagicMock()
        mock_proc.returncode = 254
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Not Found"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await storage.get_s3_last_modified("nonexistent-sess")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_malformed_json(self):
        """get_s3_last_modified returns None when HeadObject stdout is not valid JSON."""
        storage = _make_storage()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not-json-at-all", b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await storage.get_s3_last_modified("sess-malformed")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_storage_disabled(self):
        """get_s3_last_modified returns None immediately when bucket is None."""
        storage = _make_disabled_storage()
        assert not storage.enabled

        exec_mock = AsyncMock()
        with patch("asyncio.create_subprocess_exec", exec_mock):
            result = await storage.get_s3_last_modified("any-session")

        assert result is None
        exec_mock.assert_not_called()


# ============================================================================
# Test Group 2: resume_session() stale detection
# ============================================================================


@pytest.mark.unit
class TestResumeStaleDetection:
    """Tests for stale worktree detection in resume_session()."""

    @pytest.mark.asyncio
    async def test_resume_detects_stale_worktree_via_s3_head(
        self, mock_session_manager, tmp_path
    ):
        """resume triggers S3 restore when S3 LastModified > local s3_last_sync."""
        session_id = "sess-stale-detect"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T1 = time.time() - 3600  # local sync 1 hour ago
        T2 = time.time() - 60    # S3 updated 1 minute ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T1
        mock_session_manager._sessions[session_id] = state

        # Enable S3 and mock get_s3_last_modified returning T2 (newer than local T1)
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T2)

        # restore_session_from_s3 recreates the worktree (as a real restore would) and
        # returns the state. The stale detection removes the dir before calling restore,
        # so the mock must put it back to let resume proceed past the worktree check.
        async def mock_restore(sid, target_worktree_path=None, **kwargs):
            worktree_dir.mkdir(parents=True, exist_ok=True)
            return state

        mock_session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            side_effect=mock_restore
        )

        # Terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        await mock_session_manager.resume_session(session_id)

        # restore_session_from_s3 MUST have been called (stale detected)
        mock_session_manager.session_storage.restore_session_from_s3.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_skips_restore_when_local_is_current(
        self, mock_session_manager, tmp_path
    ):
        """resume does NOT trigger S3 restore when local s3_last_sync > S3 LastModified."""
        session_id = "sess-local-current"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T1 = time.time() - 60    # S3 updated 1 minute ago
        T2 = time.time() - 30    # local sync 30 seconds ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T2  # local is newer
        mock_session_manager._sessions[session_id] = state

        # Enable S3 and mock get_s3_last_modified returning T1 (older than local T2)
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T1)
        mock_session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=None
        )

        # Terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        await mock_session_manager.resume_session(session_id)

        # restore_session_from_s3 must NOT have been called
        mock_session_manager.session_storage.restore_session_from_s3.assert_not_called()


# ============================================================================
# Test Group 3: pause_session() conflict check
# ============================================================================


@pytest.mark.unit
class TestPauseConflictCheck:
    """Tests for S3 conflict check in pause_session()."""

    @pytest.mark.asyncio
    async def test_pause_skips_sync_when_s3_is_newer(
        self, mock_session_manager, tmp_path
    ):
        """pause_session does NOT overwrite S3 when S3 LastModified > local s3_last_sync."""
        session_id = "sess-pause-conflict"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T1 = time.time() - 3600  # local sync 1 hour ago
        T2 = time.time() - 60    # S3 updated 1 minute ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T1
        mock_session_manager._sessions[session_id] = state

        # Enable S3 and mock get_s3_last_modified returning T2 (newer than local T1)
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T2)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)

        await mock_session_manager.pause_session(
            session_id, sync_to_s3=True
        )

        # sync_session_to_s3 must NOT have been called (would overwrite newer S3 data)
        mock_session_manager.session_storage.sync_session_to_s3.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_proceeds_with_sync_when_local_is_current(
        self, mock_session_manager, tmp_path
    ):
        """pause_session proceeds with S3 sync when local s3_last_sync > S3 LastModified."""
        session_id = "sess-pause-ok"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T1 = time.time() - 60    # S3 updated 1 minute ago
        T2 = time.time() - 30    # local sync 30 seconds ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T2  # local is newer
        mock_session_manager._sessions[session_id] = state

        # Enable S3 and mock get_s3_last_modified returning T1 (older than local T2)
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T1)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)

        await mock_session_manager.pause_session(
            session_id, sync_to_s3=True
        )

        # sync_session_to_s3 MUST have been called
        mock_session_manager.session_storage.sync_session_to_s3.assert_called_once_with(state)


# ============================================================================
# Test Group 4: checkpoint_session() conflict check
# ============================================================================


@pytest.mark.unit
class TestCheckpointConflictCheck:
    """Tests for S3 conflict check in CheckpointManager.checkpoint_session()."""

    @pytest.mark.asyncio
    async def test_checkpoint_skips_upload_when_s3_is_newer(
        self, mock_session_manager, tmp_path
    ):
        """checkpoint_session does NOT upload when S3 LastModified > local s3_last_sync."""
        from orchestration.checkpoint_manager import CheckpointManager

        session_id = "sess-ckpt-conflict"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T1 = time.time() - 3600  # local sync 1 hour ago
        T2 = time.time() - 60    # S3 updated 1 minute ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T1
        mock_session_manager._sessions[session_id] = state

        # Enable S3 and mock get_s3_last_modified returning T2 (newer than local)
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T2)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)
        mock_session_manager._save_session_state = MagicMock()

        checkpoint_mgr = CheckpointManager(session_manager=mock_session_manager)
        result = await checkpoint_mgr.checkpoint_session(session_id)

        # sync_session_to_s3 must NOT have been called
        mock_session_manager.session_storage.sync_session_to_s3.assert_not_called()


# ============================================================================
# Test Group 5: resume with S3 disabled
# ============================================================================


@pytest.mark.unit
class TestResumeWithS3Disabled:
    """Tests for resume behavior when S3 storage is disabled."""

    @pytest.mark.asyncio
    async def test_resume_resets_s3_last_sync_after_restore(
        self, mock_session_manager, tmp_path
    ):
        """After cross-host S3 restore, state.s3_last_sync is reset to NOW so the
        next pause/checkpoint freshness guard does not mis-fire."""
        session_id = "sess-s3ls-reset"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 86400  # 24 hours ago (stale value embedded in S3 metadata)
        T_S3 = time.time() - 3600    # S3 object 1 hour ago (newer than local => triggers restore)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = True
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T_S3)

        async def mock_restore(sid, target_worktree_path=None, **kwargs):
            worktree_dir.mkdir(parents=True, exist_ok=True)
            # Simulate what load_session_metadata does: s3_last_sync stays stale
            # because it's copied verbatim from the S3 session.json
            state.s3_last_sync = T_OLD
            return state

        mock_session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            side_effect=mock_restore
        )

        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        await mock_session_manager.resume_session(session_id)

        # After resume, s3_last_sync MUST equal the S3 object's LastModified
        # (not wall clock) so the guard correctly detects concurrent writers.
        assert state.s3_last_sync is not None
        assert state.s3_last_sync == T_S3
        assert state.s3_last_sync > T_OLD

    @pytest.mark.asyncio
    async def test_resume_with_s3_disabled_uses_local(
        self, mock_session_manager, tmp_path
    ):
        """When S3 is disabled, resume does not query S3 and resumes from local state."""
        session_id = "sess-local-only"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = None
        mock_session_manager._sessions[session_id] = state

        # S3 is disabled (default fixture behavior)
        assert not mock_session_manager.session_storage.enabled

        # Track any S3 freshness query
        get_last_modified_mock = AsyncMock(return_value=None)
        mock_session_manager.session_storage.get_s3_last_modified = get_last_modified_mock

        # Terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        response = await mock_session_manager.resume_session(session_id)

        # Session should be ACTIVE after successful resume
        assert state.status == SessionStatus.ACTIVE

        # No S3 freshness query should have been made
        get_last_modified_mock.assert_not_called()


# ============================================================================
# Test Group 6: pause shield + bookkeeping atomicity
# ============================================================================


@pytest.mark.unit
class TestPauseShieldAndMessaging:
    """Regression tests for the 2026-04-21 orphan-subprocess data-loss bug.

    Bug: an in-flight checkpoint upload got cancelled between the S3
    subprocess completing and the Python-side `state.s3_last_sync = time.time()`
    assignment. S3 got a fresh tar but local state claimed no upload had
    happened, so every subsequent pause/checkpoint was blocked by the
    freshness guard until a cross-host S3 restore destructively restored the
    orphaned tar, losing ~41 minutes of work.

    Fix: upload + bookkeeping are wrapped in asyncio.shield() inside pause
    and checkpoint paths, so a racing cancel cannot split the two.
    """

    @pytest.mark.asyncio
    async def test_pause_advances_s3_last_sync_on_success(
        self, mock_session_manager, tmp_path
    ):
        """After a successful pause upload, state.s3_last_sync must advance to
        a timestamp newer than the pre-pause value. This is the core invariant
        that prevents the staleness guard from mis-firing on the next
        checkpoint."""
        session_id = "sess-pause-advance"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600  # local sync 1 hour ago
        T_S3 = time.time() - 7200   # S3 2 hours ago (older; no conflict)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        # First call = guard check (returns old S3 timestamp, no conflict).
        # Second call = post-upload read (returns fresh timestamp after our upload).
        T_AFTER_UPLOAD = time.time()
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(
            side_effect=[T_S3, T_AFTER_UPLOAD]
        )
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)

        response = await mock_session_manager.pause_session(
            session_id, sync_to_s3=True
        )

        assert state.s3_synced is True
        assert state.s3_last_sync > T_OLD, "s3_last_sync must advance past pre-pause value"
        assert state.s3_last_sync == T_AFTER_UPLOAD, "s3_last_sync must use S3 LastModified"
        assert "State synced to S3" in response.message

    @pytest.mark.asyncio
    async def test_pause_warning_message_on_s3_conflict(
        self, mock_session_manager, tmp_path
    ):
        """When the S3 freshness safety guard fires, pause must surface the
        refusal in the response message (not silently return success)."""
        session_id = "sess-pause-conflict-msg"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_LOCAL = time.time() - 3600  # local sync 1 hour ago
        T_S3 = time.time() - 60       # S3 1 minute ago (newer)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_LOCAL
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T_S3)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)

        response = await mock_session_manager.pause_session(
            session_id, sync_to_s3=True
        )

        mock_session_manager.session_storage.sync_session_to_s3.assert_not_called()
        assert "WARNING: S3 has newer data" in response.message
        assert "NOT uploaded" in response.message

    @pytest.mark.asyncio
    async def test_pause_warning_message_on_sync_failure(
        self, mock_session_manager, tmp_path
    ):
        """When sync_session_to_s3 returns False (partial failure), pause must
        surface the failure in the response message AND not advance s3_last_sync."""
        session_id = "sess-pause-sync-fail"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600
        T_S3 = time.time() - 7200  # older than local, no conflict

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T_S3)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=False)

        response = await mock_session_manager.pause_session(
            session_id, sync_to_s3=True
        )

        assert state.s3_synced is False
        assert state.s3_last_sync == T_OLD, "s3_last_sync must NOT advance on upload failure"
        assert "WARNING: S3 sync failed" in response.message

    @pytest.mark.asyncio
    async def test_pause_persists_state_even_on_handler_cancel(
        self, mock_session_manager, tmp_path
    ):
        """If the HTTP handler is cancelled (ALB idle, client abort) during a
        pause-time upload, _save_session_state must still run so PAUSED status
        and any in-memory s3_last_sync advancement reach disk."""
        session_id = "sess-pause-cancel-persist"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600
        T_S3 = time.time() - 7200  # older, no conflict

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        T_AFTER_UPLOAD = time.time()
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(
            side_effect=[T_S3, T_AFTER_UPLOAD]
        )

        # Simulate a slow upload that the handler will cancel mid-flight.
        # Shield ensures the inner coroutine completes before CancelledError propagates.
        upload_completed = asyncio.Event()

        async def slow_upload(_state):
            await asyncio.sleep(0.05)
            upload_completed.set()
            return True

        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(side_effect=slow_upload)

        save_calls = []
        original_save = mock_session_manager._save_session_state

        def tracked_save(s):
            save_calls.append(s.s3_last_sync)
            return original_save(s)

        mock_session_manager._save_session_state = tracked_save

        # Run pause in a task and cancel it mid-flight
        pause_task = asyncio.create_task(
            mock_session_manager.pause_session(session_id, sync_to_s3=True)
        )
        await asyncio.sleep(0.01)  # let upload start
        pause_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pause_task

        # The outer task raised CancelledError immediately, but asyncio.shield
        # keeps the inner coroutine alive in the background. Wait for it to
        # actually finish before asserting bookkeeping completed.
        # Use wait_for with a timeout to fail fast instead of hanging CI if
        # the shielded inner was never reached.
        await asyncio.wait_for(upload_completed.wait(), timeout=5.0)
        await asyncio.sleep(0)  # let post-await statements run

        # Shield guarantees inner upload finished setting s3_last_sync
        assert upload_completed.is_set(), "shielded upload must complete despite cancel"
        # _save_session_state must have been invoked via the try/finally
        assert len(save_calls) >= 1, "_save_session_state must run even on cancel"
        # The persisted state carries the advanced s3_last_sync
        assert state.s3_last_sync > T_OLD, "s3_last_sync must have advanced in memory"
        assert save_calls[-1] > T_OLD, "latest _save_session_state call saw advanced s3_last_sync"


# ============================================================================
# Test Group 7: checkpoint shield + bookkeeping atomicity
# ============================================================================


@pytest.mark.unit
class TestCheckpointShieldAndBookkeeping:
    """Regression tests for the checkpoint-path variant of the orphan bug."""

    @pytest.mark.asyncio
    async def test_checkpoint_advances_s3_last_sync_on_success(
        self, mock_session_manager, tmp_path
    ):
        """A successful checkpoint must advance state.s3_last_sync."""
        from orchestration.checkpoint_manager import CheckpointManager

        session_id = "sess-ckpt-advance"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600
        T_S3 = time.time() - 7200

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        T_AFTER_UPLOAD = time.time()
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(
            side_effect=[T_S3, T_AFTER_UPLOAD]
        )
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=True)
        mock_session_manager._save_session_state = MagicMock()

        checkpoint_mgr = CheckpointManager(session_manager=mock_session_manager)
        result = await checkpoint_mgr.checkpoint_session(session_id)

        assert result is True
        assert state.s3_synced is True
        assert state.s3_last_sync == T_AFTER_UPLOAD
        mock_session_manager._save_session_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_cancel_during_upload_preserves_s3_last_sync_via_shield(
        self, mock_session_manager, tmp_path
    ):
        """Regression: if a WebSocket reconnect cancels the pending checkpoint
        task mid-upload, asyncio.shield() must ensure the inner upload +
        bookkeeping complete atomically. Without shield, s3_last_sync would
        stay at the pre-upload value while S3 got a fresh tar, triggering the
        staleness guard on every subsequent checkpoint/pause."""
        from orchestration.checkpoint_manager import CheckpointManager

        session_id = "sess-ckpt-shield"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600
        T_S3 = time.time() - 7200

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        T_AFTER_UPLOAD = time.time()
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(
            side_effect=[T_S3, T_AFTER_UPLOAD]
        )

        upload_completed = asyncio.Event()

        async def slow_upload(_state):
            # Simulate the real-world race: the aws s3 cp subprocess takes ~1s;
            # a cancel fires ~100ms in. With shield, the upload still finishes.
            await asyncio.sleep(0.05)
            upload_completed.set()
            return True

        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(side_effect=slow_upload)
        mock_session_manager._save_session_state = MagicMock()

        checkpoint_mgr = CheckpointManager(session_manager=mock_session_manager)

        ckpt_task = asyncio.create_task(checkpoint_mgr.checkpoint_session(session_id))
        await asyncio.sleep(0.01)  # let upload begin
        ckpt_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await ckpt_task

        # The outer task raised CancelledError immediately, but asyncio.shield
        # keeps the inner coroutine alive in the background. Wait for it to
        # actually finish before asserting bookkeeping completed.
        # Use wait_for with a timeout to fail fast instead of hanging CI if
        # the shielded inner was never reached.
        await asyncio.wait_for(upload_completed.wait(), timeout=5.0)
        # Give the event loop one more tick to let the inner coroutine's
        # post-await statements (state mutation, _save_session_state) run.
        await asyncio.sleep(0)

        # Shield guarantee: inner upload+bookkeeping completed before cancel propagated
        assert upload_completed.is_set()
        assert state.s3_synced is True, "shield must let bookkeeping complete"
        assert state.s3_last_sync > T_OLD, "shield must let s3_last_sync advance"
        mock_session_manager._save_session_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_checkpoint_skips_bookkeeping_when_sync_returns_false(
        self, mock_session_manager, tmp_path
    ):
        """If sync_session_to_s3 returns False (partial failure), checkpoint
        must NOT mark the session as synced or advance s3_last_sync."""
        from orchestration.checkpoint_manager import CheckpointManager

        session_id = "sess-ckpt-partial-fail"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        T_OLD = time.time() - 3600
        T_S3 = time.time() - 7200

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        state.s3_synced = False
        state.s3_last_sync = T_OLD
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.get_s3_last_modified = AsyncMock(return_value=T_S3)
        mock_session_manager.session_storage.sync_session_to_s3 = AsyncMock(return_value=False)
        mock_session_manager._save_session_state = MagicMock()

        checkpoint_mgr = CheckpointManager(session_manager=mock_session_manager)
        result = await checkpoint_mgr.checkpoint_session(session_id)

        assert result is False
        assert state.s3_synced is False
        assert state.s3_last_sync == T_OLD, "s3_last_sync must not advance on failure"
