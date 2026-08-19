# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Regression tests for GPU leak on crash / failure conditions.

Verifies that GPU resources are properly released when:
- Session terminate encounters failures (worktree cleanup, CLI tool kill hang)

These tests prevent regression of known production issues where GPU resources
leaked on crash, causing progressive GPU starvation until server restart.
"""

import asyncio
import os
import signal
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    from shared.gpu_file_lock import reset_gpu_lock_manager
    reset_gpu_lock_manager()
    yield
    reset_gpu_lock_manager()


@pytest.fixture
def gpu_manager(tmp_path):
    """Create a GPUResourceManager with 4 mocked GPUs and a temp lock dir."""
    lock_dir = str(tmp_path / "gpu_locks")
    os.makedirs(lock_dir, exist_ok=True)

    with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
        from shared.gpu_resource_manager import GPUResourceManager
        manager = GPUResourceManager(lock_dir=lock_dir)
        yield manager


# =============================================================================
# Session terminate releases GPUs under failure conditions
# =============================================================================

@pytest.mark.unit
class TestSessionTerminateReleasesGpusOnFailure:
    """Verify session GPU release even when other cleanup steps fail."""

    @pytest.fixture
    def session_manager_with_deps(self, gpu_manager, tmp_path):
        """Create a SessionManager with mocked dependencies and real gpu_manager."""
        terminal_mgr = Mock()
        terminal_mgr.is_available.return_value = True
        terminal_mgr.stop_terminal = AsyncMock(return_value=True)

        cli_tool_mgr = Mock()
        worktree_mgr = Mock()

        inactivity_monitor = Mock()
        inactivity_monitor.unregister_session = Mock()

        session_storage = Mock()
        session_storage.enabled = False

        from orchestration.session_manager import SessionManager
        with patch.object(SessionManager, '_load_sessions'):
            sm = SessionManager(
                sessions_dir=str(tmp_path),
                worktree_manager=worktree_mgr,
                gpu_manager=gpu_manager,
                terminal_manager=terminal_mgr,
                cli_tool_manager=cli_tool_mgr,
                inactivity_monitor=inactivity_monitor,
                session_storage=session_storage,
            )
        return sm, terminal_mgr, worktree_mgr

    def _create_session_state(self, sm, session_id, gpu_ids, tmp_path):
        """Helper to inject an active session with GPU allocations."""
        from shared.session_models import SessionState, SessionStatus, CLIToolType

        session_dir = str(tmp_path / session_id)
        os.makedirs(session_dir, exist_ok=True)

        state = SessionState(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CLAUDE,
            repo_name="vllm",
            branch="main",
            created_at=time.time(),
            last_accessed=time.time(),
            terminal_port=9000,
            worktree_path=f"{session_dir}/worktree",
            session_dir=session_dir,
            gpu_ids=gpu_ids,
        )
        sm._sessions[session_id] = state
        return state

    @pytest.mark.asyncio
    async def test_gpus_released_when_worktree_cleanup_fails(
        self, session_manager_with_deps, gpu_manager, tmp_path
    ):
        """GPUs must be released even when worktree cleanup raises an exception."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_deps

        # Acquire GPUs for session
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_wt", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        state = self._create_session_state(sm, "sess_wt", gpu_ids, tmp_path)

        # Make worktree cleanup fail
        worktree_mgr.cleanup_session.side_effect = OSError("Permission denied on worktree")

        # Terminate should still release GPUs
        response = await sm.terminate_session("sess_wt")
        assert response.status == "terminated"
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_gpus_released_when_stop_terminal_fails(
        self, session_manager_with_deps, gpu_manager, tmp_path
    ):
        """GPUs must be released even when stop_terminal raises an exception."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_deps

        # Acquire GPUs for session
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_term", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        state = self._create_session_state(sm, "sess_term", gpu_ids, tmp_path)

        # Make terminal stop fail
        terminal_mgr.stop_terminal = AsyncMock(side_effect=OSError("Cannot kill ttyd"))

        # Mock _force_kill_session_processes to avoid os.getpgid on non-existent PIDs
        with patch.object(sm, '_force_kill_session_processes'):
            response = await sm.terminate_session("sess_term")

        assert response.status == "terminated"
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_gpus_released_when_terminal_stop_and_worktree_both_fail(
        self, session_manager_with_deps, gpu_manager, tmp_path
    ):
        """GPUs released even when BOTH terminal stop and worktree cleanup fail."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_deps

        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_both", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        state = self._create_session_state(sm, "sess_both", gpu_ids, tmp_path)

        terminal_mgr.stop_terminal = AsyncMock(side_effect=OSError("Cannot kill ttyd"))
        worktree_mgr.cleanup_session.side_effect = OSError("Permission denied")

        with patch.object(sm, '_force_kill_session_processes'):
            response = await sm.terminate_session("sess_both")

        assert response.status == "terminated"
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_double_release_on_terminate_is_safe(
        self, session_manager_with_deps, gpu_manager, tmp_path
    ):
        """Calling release_gpus_for_session twice (e.g., from retry logic) is safe."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_deps

        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_dbl", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        state = self._create_session_state(sm, "sess_dbl", gpu_ids, tmp_path)

        response = await sm.terminate_session("sess_dbl")
        assert response.status == "terminated"
        assert gpu_manager.get_available_gpu_count() == 4

        # Second release should be a no-op (no crash)
        gpu_manager.release_gpus_for_session("sess_dbl")
        assert gpu_manager.get_available_gpu_count() == 4
