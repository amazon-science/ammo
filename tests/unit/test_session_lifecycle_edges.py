# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for session lifecycle edge cases.

Tests creation failures, status transition guards, ownership validation,
persistence behaviour on startup, and S3 cleanup on termination.
"""

import json
import os
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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
    mock_session_manager_auth_enabled,
    mock_session_manager_auth_disabled,
    make_session_state,
    make_create_request,
)
from shared.session_models import SessionStatus, CLIToolType


# ============================================================================
# Session Creation Edge Cases
# ============================================================================

@pytest.mark.unit
class TestSessionCreationEdges:
    """Edge cases during session creation."""

    @pytest.mark.asyncio
    async def test_create_mid_failure_sets_status_failed(self, mock_session_manager):
        """If terminal startup raises, session ends up in FAILED status."""
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=Exception("ttyd not found")
        )
        # Also make terminal is_available return True so the code tries to start
        mock_session_manager.terminal_manager.is_available.return_value = True

        request = make_create_request(repo_name="vllm", gpu_count=0)

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError):
            await mock_session_manager.create_session(request)

        # Find the session that was created and check its status
        failed_sessions = [
            s for s in mock_session_manager._sessions.values()
            if s.status == SessionStatus.FAILED
        ]
        assert len(failed_sessions) == 1

    def test_corrupted_session_json_on_load(self, tmp_path):
        """Corrupted session.json is logged and skipped - manager still starts."""
        import os
        from shared.gpu_file_lock import reset_gpu_lock_manager
        from orchestration.session_manager import reset_session_manager

        reset_gpu_lock_manager()
        reset_session_manager()

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Write a corrupted session file
        bad_session_dir = sessions_dir / "bad-session-id"
        bad_session_dir.mkdir()
        (bad_session_dir / "session.json").write_text("{ this is not valid json !!!")

        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)

        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=2):
            from shared.gpu_resource_manager import GPUResourceManager
            gpu_manager = GPUResourceManager(lock_dir=lock_dir)

        mock_worktree = MagicMock()
        mock_terminal = MagicMock()
        mock_cli = MagicMock()
        mock_monitor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.enabled = False

        from orchestration.session_manager import SessionManager
        manager = SessionManager(
            sessions_dir=str(sessions_dir),
            worktree_manager=mock_worktree,
            gpu_manager=gpu_manager,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_monitor,
            session_storage=mock_storage,
        )

        # Bad session should be skipped - manager should load with 0 sessions
        assert "bad-session-id" not in manager._sessions


# ============================================================================
# Session State Transition Guards
# ============================================================================

@pytest.mark.unit
class TestSessionStateTransitions:
    """Tests for invalid state transitions raising errors."""

    @pytest.mark.asyncio
    async def test_double_terminate_is_safe(self, mock_session_manager):
        """Terminating an already-terminated session raises SessionError."""
        session_id = "sess-term"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        mock_session_manager._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        # First terminate succeeds
        await mock_session_manager.terminate_session(session_id)

        # Second terminate should raise (session no longer in _sessions)
        with pytest.raises(SessionError):
            await mock_session_manager.terminate_session(session_id)

    @pytest.mark.asyncio
    async def test_pause_already_paused_raises(self, mock_session_manager):
        """Pausing a PAUSED session raises SessionError."""
        session_id = "sess-paused"
        state = make_session_state(session_id=session_id, status=SessionStatus.PAUSED)
        mock_session_manager._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError, match="cannot pause"):
            await mock_session_manager.pause_session(session_id)

    @pytest.mark.asyncio
    async def test_resume_non_paused_raises(self, mock_session_manager):
        """Resuming a FAILED or CREATING session raises SessionError."""
        from orchestration.session_manager import SessionError

        for status in (SessionStatus.FAILED, SessionStatus.CREATING):
            session_id = f"sess-{status.value}"
            state = make_session_state(session_id=session_id, status=status)
            mock_session_manager._sessions[session_id] = state

            with pytest.raises(SessionError, match="cannot resume"):
                await mock_session_manager.resume_session(session_id)

    @pytest.mark.asyncio
    async def test_resume_active_returns_already_active(self, mock_session_manager):
        """Resuming an ACTIVE session returns 'Session already active' without error."""
        session_id = "sess-active-resume"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        # Worktree path must exist for resume to not raise FileNotFoundError
        state.worktree_path = "/tmp"
        mock_session_manager._sessions[session_id] = state

        response = await mock_session_manager.resume_session(session_id)

        assert "already active" in response.message.lower()


# ============================================================================
# Session Ownership
# ============================================================================

@pytest.mark.unit
class TestSessionOwnership:
    """Tests for owner_id-based access control."""

    @pytest.mark.asyncio
    async def test_ownership_validation_cross_client(self, mock_session_manager):
        """Client B cannot access a session owned by client A."""
        session_id = "sess-owned"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id="client-A",
        )
        mock_session_manager._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError):
            mock_session_manager._validate_ownership(session_id, owner_id="client-B")

    @pytest.mark.asyncio
    async def test_ownership_validation_correct_owner(self, mock_session_manager):
        """Client A can access its own session."""
        session_id = "sess-owned-ok"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id="client-A",
        )
        mock_session_manager._sessions[session_id] = state

        # Should not raise
        result = mock_session_manager._validate_ownership(session_id, owner_id="client-A")
        assert result is state

    def test_legacy_null_owner_accessible_by_all(self, mock_session_manager):
        """Sessions with owner_id=None (legacy) are accessible by any client."""
        session_id = "sess-legacy"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        state.owner_id = None
        mock_session_manager._sessions[session_id] = state

        # Any client ID should be able to access
        result = mock_session_manager._validate_ownership(session_id, owner_id="any-client")
        assert result is state

    def test_no_client_id_access_all_sessions(self, mock_session_manager):
        """If owner_id param is None (no header), all sessions are accessible."""
        session_id = "sess-any"
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            owner_id="client-X",
        )
        mock_session_manager._sessions[session_id] = state

        # Passing owner_id=None means no restriction
        result = mock_session_manager._validate_ownership(session_id, owner_id=None)
        assert result is state


# ============================================================================
# Session Ownership - Auth Enabled vs Disabled
# ============================================================================

@pytest.mark.unit
class TestSessionOwnershipAuthModes:
    """Tests for ownership validation parameterized across auth modes."""

    def test_cross_client_rejected_auth_enabled(self, mock_session_manager_auth_enabled):
        """Client B cannot access Client A's session when auth is enabled."""
        mgr = mock_session_manager_auth_enabled
        session_id = "sess-auth-owned"
        state = make_session_state(
            session_id=session_id, status=SessionStatus.ACTIVE, owner_id="client-A"
        )
        mgr._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError):
            mgr._validate_ownership(session_id, owner_id="client-B")

    def test_cross_client_rejected_auth_disabled(self, mock_session_manager_auth_disabled):
        """Client B cannot access Client A's session even when auth is disabled."""
        mgr = mock_session_manager_auth_disabled
        session_id = "sess-noauth-owned"
        state = make_session_state(
            session_id=session_id, status=SessionStatus.ACTIVE, owner_id="client-A"
        )
        mgr._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError):
            mgr._validate_ownership(session_id, owner_id="client-B")

    def test_null_client_id_auth_enabled_rejects_owned_session(self, mock_session_manager_auth_enabled):
        """Null client_id + auth enabled: accessing an owned session is rejected."""
        mgr = mock_session_manager_auth_enabled
        session_id = "sess-auth-null"
        state = make_session_state(
            session_id=session_id, status=SessionStatus.ACTIVE, owner_id="client-A"
        )
        mgr._sessions[session_id] = state

        from orchestration.session_manager import SessionError
        with pytest.raises(SessionError, match="not found"):
            mgr._validate_ownership(session_id, owner_id=None)

    def test_null_client_id_auth_disabled_allows_owned_session(self, mock_session_manager_auth_disabled):
        """Null client_id + auth disabled: accessing an owned session is allowed (backward compat)."""
        mgr = mock_session_manager_auth_disabled
        session_id = "sess-noauth-null"
        state = make_session_state(
            session_id=session_id, status=SessionStatus.ACTIVE, owner_id="client-A"
        )
        mgr._sessions[session_id] = state

        result = mgr._validate_ownership(session_id, owner_id=None)
        assert result is state

    def test_null_client_id_auth_enabled_allows_legacy_session(self, mock_session_manager_auth_enabled):
        """Null client_id + auth enabled: legacy sessions (owner_id=None) are accessible."""
        mgr = mock_session_manager_auth_enabled
        session_id = "sess-auth-legacy"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        state.owner_id = None
        mgr._sessions[session_id] = state

        result = mgr._validate_ownership(session_id, owner_id=None)
        assert result is state


# ============================================================================
# Session Persistence
# ============================================================================

@pytest.mark.unit
class TestSessionPersistence:
    """Tests for session loading and startup state transitions."""

    def _make_manager_from_sessions_dir(self, tmp_path, sessions_dir):
        """Helper: create a SessionManager that loads from sessions_dir."""
        import os
        from shared.gpu_file_lock import reset_gpu_lock_manager
        from orchestration.session_manager import reset_session_manager

        reset_gpu_lock_manager()
        reset_session_manager()

        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)

        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
            from shared.gpu_resource_manager import GPUResourceManager
            gpu_manager = GPUResourceManager(lock_dir=lock_dir)

        mock_worktree = MagicMock()
        mock_terminal = MagicMock()
        mock_cli = MagicMock()
        mock_monitor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.enabled = False

        from orchestration.session_manager import SessionManager
        return SessionManager(
            sessions_dir=str(sessions_dir),
            worktree_manager=mock_worktree,
            gpu_manager=gpu_manager,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_monitor,
            session_storage=mock_storage,
        )

    def _write_session_to_disk(self, sessions_dir, state):
        """Helper: write a SessionState to disk in sessions_dir."""
        session_dir = sessions_dir / state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        state_path = session_dir / "session.json"
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f)

    def test_startup_marks_active_sessions_as_paused(self, tmp_path):
        """On server restart, ACTIVE sessions loaded from disk are marked PAUSED."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        active_state = make_session_state(
            session_id="sess-was-active",
            status=SessionStatus.ACTIVE,
        )
        self._write_session_to_disk(sessions_dir, active_state)

        manager = self._make_manager_from_sessions_dir(tmp_path, sessions_dir)

        loaded = manager._sessions.get("sess-was-active")
        assert loaded is not None
        assert loaded.status == SessionStatus.PAUSED

    def test_startup_skips_terminated_sessions(self, tmp_path):
        """Terminated sessions are not loaded into memory on startup."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        terminated_state = make_session_state(
            session_id="sess-terminated",
            status=SessionStatus.TERMINATED,
        )
        self._write_session_to_disk(sessions_dir, terminated_state)

        manager = self._make_manager_from_sessions_dir(tmp_path, sessions_dir)

        assert "sess-terminated" not in manager._sessions

    def test_startup_skips_failed_sessions(self, tmp_path):
        """FAILED sessions are not loaded into memory on startup."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        failed_state = make_session_state(
            session_id="sess-failed",
            status=SessionStatus.FAILED,
        )
        self._write_session_to_disk(sessions_dir, failed_state)

        manager = self._make_manager_from_sessions_dir(tmp_path, sessions_dir)

        assert "sess-failed" not in manager._sessions

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions_removes_old_terminated(self, mock_session_manager):
        """cleanup_old_sessions removes old terminated sessions from memory and disk."""
        session_id = "sess-old-term"
        state = make_session_state(session_id=session_id, status=SessionStatus.TERMINATED)
        # Make it look very old (10 days)
        state.last_accessed = time.time() - 10 * 24 * 3600
        mock_session_manager._sessions[session_id] = state

        # Create session directory on disk
        session_dir = mock_session_manager.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(json.dumps(state.to_dict()))

        cleaned = await mock_session_manager.cleanup_old_sessions(max_age_days=7)

        assert cleaned == 1
        assert session_id not in mock_session_manager._sessions

    @pytest.mark.asyncio
    async def test_cleanup_does_not_remove_recent_terminated(self, mock_session_manager):
        """cleanup_old_sessions preserves recently terminated sessions."""
        session_id = "sess-recent-term"
        state = make_session_state(session_id=session_id, status=SessionStatus.TERMINATED)
        # Terminated only 1 hour ago
        state.last_accessed = time.time() - 3600
        mock_session_manager._sessions[session_id] = state

        cleaned = await mock_session_manager.cleanup_old_sessions(max_age_days=7)

        assert cleaned == 0
        assert session_id in mock_session_manager._sessions

    @pytest.mark.asyncio
    async def test_terminate_deletes_from_s3(self, mock_session_manager):
        """When S3 is enabled and session terminates, delete_session_from_s3 is called."""
        session_id = "sess-s3-delete"
        state = make_session_state(session_id=session_id, status=SessionStatus.ACTIVE)
        mock_session_manager._sessions[session_id] = state

        # Enable S3 storage
        mock_session_manager.session_storage.enabled = True

        await mock_session_manager.terminate_session(session_id)

        mock_session_manager.session_storage.delete_session_from_s3.assert_called_once_with(
            session_id
        )
