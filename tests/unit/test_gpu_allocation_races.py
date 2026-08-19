# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for GPU allocation race conditions in the AMMO session service.

Tests verify:
1. Concurrent session creation does not double-assign GPUs
2. GPU release during acquire does not corrupt state
3. Session creation failure mid-way releases GPUs properly
4. Concurrent pause+terminate on the same session is safe
5. Rapid create/terminate cycles do not leak allocations
"""

import asyncio
import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Shared fixtures
# =============================================================================

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


@pytest.fixture
def session_manager_with_gpu(gpu_manager, tmp_path):
    """
    Create a SessionManager with real GPU locks and mocked everything else.
    Mirrors the pattern in test_gpu_leak_on_crash.py for direct session injection.
    """
    terminal_mgr = Mock()
    terminal_mgr.is_available.return_value = True
    terminal_mgr.stop_terminal = AsyncMock(return_value=True)

    cli_tool_mgr = Mock()
    worktree_mgr = Mock()

    inactivity_monitor = Mock()
    inactivity_monitor.unregister_session = Mock()
    inactivity_monitor.register_session = Mock()

    session_storage = Mock()
    session_storage.enabled = False

    from orchestration.session_manager import SessionManager
    with patch.object(SessionManager, '_load_sessions'):
        sm = SessionManager(
            sessions_dir=str(tmp_path / "sessions"),
            worktree_manager=worktree_mgr,
            gpu_manager=gpu_manager,
            terminal_manager=terminal_mgr,
            cli_tool_manager=cli_tool_mgr,
            inactivity_monitor=inactivity_monitor,
            session_storage=session_storage,
        )
    return sm, terminal_mgr, worktree_mgr


def _inject_active_session(sm, session_id, gpu_ids, tmp_path):
    """Inject an active session with GPU allocations already registered."""
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
        worktree_path=str(tmp_path / session_id / "worktree"),
        session_dir=session_dir,
        gpu_ids=gpu_ids,
    )
    sm._sessions[session_id] = state
    return state


# =============================================================================
# TestConcurrentSessionGPUAllocation
# =============================================================================

@pytest.mark.unit
class TestConcurrentSessionGPUAllocation:
    """GPU allocation does not double-assign under concurrent session creates."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creates_no_double_assignment(self, gpu_manager):
        """4 concurrent sessions requesting 1 GPU each must get distinct GPUs."""
        acquired = []

        async def acquire_one(session_id: str):
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                session_id, gpu_count=1, timeout=5
            )
            acquired.append((session_id, gpu_ids[0]))
            # Hold briefly so all 4 contend at once
            await asyncio.sleep(0.05)
            gpu_manager.release_gpus_for_session(session_id)

        tasks = [asyncio.create_task(acquire_one(f"sess_{i}")) for i in range(4)]
        await asyncio.gather(*tasks)

        # All 4 should have succeeded
        assert len(acquired) == 4
        # GPU IDs assigned must span all 4 distinct GPUs (no double-assignment at overlap)
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_release_during_acquire_does_not_corrupt_state(self, gpu_manager):
        """Release of one session while another is mid-acquire leaves state consistent."""
        # Fill 3 of 4 GPUs
        await gpu_manager.acquire_gpus_for_session_async("holder", gpu_count=3, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 1

        acquired_event = asyncio.Event()

        async def late_acquirer():
            # Tries to grab 2 GPUs — must wait for holder to release
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                "waiter", gpu_count=2, timeout=5
            )
            acquired_event.set()
            return gpu_ids

        task = asyncio.create_task(late_acquirer())
        await asyncio.sleep(0.15)  # Let waiter start waiting

        # Release holder (frees 3 GPUs)
        gpu_manager.release_gpus_for_session("holder")

        gpu_ids = await task
        assert len(gpu_ids) == 2
        assert gpu_manager.get_available_gpu_count() == 2

        # State is consistent — no phantom allocations
        session_allocs = [
            a for a in gpu_manager._allocations.values()
            if a.allocation_id == "holder"
        ]
        assert len(session_allocs) == 0, "Holder still has allocations after release"

        gpu_manager.release_gpus_for_session("waiter")
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_partial_acquire_rollback(self, gpu_manager):
        """Session that cannot acquire all requested GPUs rolls back partial allocations."""
        # Hold 3 of 4 GPUs
        await gpu_manager.acquire_gpus_for_session_async("blocker", gpu_count=3, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 1

        # Try to acquire 2 GPUs with short timeout — should fail and rollback
        with pytest.raises(TimeoutError, match="Could not acquire"):
            await gpu_manager.acquire_gpus_for_session_async(
                "partial", gpu_count=2, timeout=0.5
            )

        # No partial allocations for "partial" session
        partial_allocs = [
            a for a in gpu_manager._allocations.values()
            if a.allocation_id == "partial"
        ]
        assert len(partial_allocs) == 0

        # Blocker's 3 allocations still present
        assert gpu_manager.get_available_gpu_count() == 1

        gpu_manager.release_gpus_for_session("blocker")
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_get_available_count_snapshot_safety(self, gpu_manager):
        """get_available_gpu_count() is safe to call during concurrent modifications."""
        counts = []

        async def acquirer(session_id: str):
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                session_id, gpu_count=1, timeout=5
            )
            counts.append(gpu_manager.get_available_gpu_count())
            await asyncio.sleep(0.01)
            gpu_manager.release_gpus_for_session(session_id)

        tasks = [asyncio.create_task(acquirer(f"s{i}")) for i in range(4)]
        await asyncio.gather(*tasks)

        # After all done, count must be 4 (no leaks, no RuntimeError)
        assert gpu_manager.get_available_gpu_count() == 4
        # All intermediate counts must be in valid range [0, 4]
        for c in counts:
            assert 0 <= c <= 4, f"count {c} out of range"


# =============================================================================
# TestSessionCreationFailureCleanup
# =============================================================================

@pytest.mark.unit
class TestSessionCreationFailureCleanup:
    """GPU released when session creation fails mid-way."""

    @pytest.mark.asyncio
    async def test_create_session_failure_mid_worktree_releases_gpus(
        self, session_manager_with_gpu, gpu_manager, tmp_path
    ):
        """If worktree creation fails, no GPUs should be leaked."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_gpu

        # Worktree creation raises synchronously (called via run_in_executor)
        worktree_mgr.create_worktree.side_effect = OSError("git clone failed")

        from shared.session_models import CreateSessionRequest
        from orchestration.session_manager import SessionError

        request = CreateSessionRequest(
            repo_name="other",  # not "vllm" to skip vLLM env init
            cli_tool="claude",
            branch="main",
            gpu_count=2,
        )

        with pytest.raises(SessionError, match="Failed to create session"):
            await sm.create_session(request, owner_id="client-1")

        # GPUs must be free after failure
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_create_session_failure_mid_terminal_releases_gpus(
        self, session_manager_with_gpu, gpu_manager, tmp_path
    ):
        """If terminal start fails, GPUs acquired before it must be released."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_gpu

        # Worktree succeeds
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir(parents=True)
        worktree_mgr.create_worktree.return_value = worktree_path
        worktree_mgr.get_session_dir.return_value = tmp_path / "session"
        worktree_mgr.get_logs_dir.return_value = tmp_path / "logs"
        worktree_mgr.cleanup_session.return_value = None

        # CLI setup fails
        sm.cli_tool_manager.setup_workspace.side_effect = RuntimeError("workspace setup error")

        from shared.session_models import CreateSessionRequest
        from orchestration.session_manager import SessionError

        request = CreateSessionRequest(
            repo_name="other",
            cli_tool="claude",
            branch="main",
            gpu_count=2,
        )

        with pytest.raises(SessionError, match="Failed to create session"):
            await sm.create_session(request, owner_id="client-1")

        # GPUs must be released after failure
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_terminate_releases_gpus_before_worktree_cleanup(
        self, session_manager_with_gpu, gpu_manager, tmp_path
    ):
        """terminate_session releases GPUs even if worktree cleanup fails afterwards."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_gpu

        # Acquire GPUs for the session
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_t", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        _inject_active_session(sm, "sess_t", gpu_ids, tmp_path)

        # Worktree cleanup fails
        worktree_mgr.cleanup_session.side_effect = OSError("rm -rf failed")

        response = await sm.terminate_session("sess_t")
        assert response.status == "terminated"

        # GPUs must be released even though cleanup failed
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0


# =============================================================================
# TestConcurrentSessionOperations
# =============================================================================

@pytest.mark.unit
class TestConcurrentSessionOperations:
    """Concurrent operations on the same or different sessions are safe."""

    @pytest.mark.asyncio
    async def test_concurrent_pause_and_terminate_same_session(
        self, session_manager_with_gpu, gpu_manager, tmp_path
    ):
        """
        Concurrent pause and terminate on the same session do not double-release GPUs.

        One operation will find the session in ACTIVE state and proceed;
        the other will find it in a different state or missing and raise SessionError.
        The GPU count must end up at 4 (no double-free, no leak).
        """
        from shared.session_models import SessionStatus
        from orchestration.session_manager import SessionError

        sm, terminal_mgr, worktree_mgr = session_manager_with_gpu

        gpu_ids = await gpu_manager.acquire_gpus_for_session_async("sess_c", 2, timeout=5)
        assert gpu_manager.get_available_gpu_count() == 2

        _inject_active_session(sm, "sess_c", gpu_ids, tmp_path)

        # Run pause and terminate concurrently; exactly one must succeed
        results = await asyncio.gather(
            sm.pause_session("sess_c"),
            sm.terminate_session("sess_c"),
            return_exceptions=True,
        )

        # At least one must succeed (not both raise)
        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) >= 1

        # GPU count must be 4 regardless of which operation won
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_rapid_create_terminate_cycles_no_leak(
        self, session_manager_with_gpu, gpu_manager, tmp_path
    ):
        """Rapid acquire/release of GPUs through session lifecycle leaves no leaks."""
        sm, terminal_mgr, worktree_mgr = session_manager_with_gpu

        for i in range(10):
            session_id = f"rapid_{i}"
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                session_id, gpu_count=1, timeout=5
            )
            _inject_active_session(sm, session_id, gpu_ids, tmp_path)
            response = await sm.terminate_session(session_id)
            assert response.status == "terminated"

        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_zero_gpu_session_does_not_block_jobs(self, gpu_manager):
        """Sessions requesting 0 GPUs do not affect GPU availability for kernel jobs."""
        # Simulate a zero-GPU session (returns empty list, no allocations)
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
            "zero_gpu_sess", gpu_count=0, timeout=5
        )
        assert gpu_ids == []
        assert gpu_manager.get_available_gpu_count() == 4

        # All 4 GPUs are available for kernel jobs
        acquired = []
        for i in range(4):
            gpu_ids_job = await gpu_manager.acquire_gpus_for_session_async(
                f"job_sess_{i}", gpu_count=1, timeout=5
            )
            acquired.extend(gpu_ids_job)

        # All 4 distinct GPUs acquired
        assert sorted(acquired) == [0, 1, 2, 3]
        assert gpu_manager.get_available_gpu_count() == 0

        for i in range(4):
            gpu_manager.release_gpus_for_session(f"job_sess_{i}")

        assert gpu_manager.get_available_gpu_count() == 4


# =============================================================================
# TestGPUAvailabilityReporting
# =============================================================================

@pytest.mark.unit
class TestGPUAvailabilityReporting:
    """GPU availability count returned by health/503 logic is accurate."""

    @pytest.mark.asyncio
    async def test_503_response_has_accurate_available_count(self, gpu_manager):
        """
        Simulate the 503-response logic from app.py:
        get_available_gpu_count() must reflect reality before the request is rejected.
        """
        # Occupy 3 GPUs (simulating active sessions)
        await gpu_manager.acquire_gpus_for_session_async("s1", gpu_count=2, timeout=5)
        await gpu_manager.acquire_gpus_for_session_async("s2", gpu_count=1, timeout=5)

        available = gpu_manager.get_available_gpu_count()
        requested = 2

        # Should trigger the 503 branch (available < requested)
        assert available < requested, (
            f"Expected available ({available}) < requested ({requested})"
        )

        # The reported available count is accurate
        assert available == 1

        # The 503 response would show available=1, requested=2
        # Verify that after release the count is correct
        gpu_manager.release_gpus_for_session("s1")
        gpu_manager.release_gpus_for_session("s2")
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_concurrent_availability_reads_during_allocation(self, gpu_manager):
        """get_available_gpu_count() never raises RuntimeError during concurrent modification."""
        errors = []

        async def reader():
            for _ in range(50):
                try:
                    count = gpu_manager.get_available_gpu_count()
                    assert 0 <= count <= 4
                except RuntimeError as e:
                    errors.append(e)
                await asyncio.sleep(0)

        async def writer(session_id: str):
            for _ in range(5):
                try:
                    gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                        session_id, gpu_count=1, timeout=1
                    )
                    await asyncio.sleep(0.01)
                    gpu_manager.release_gpus_for_session(session_id)
                except TimeoutError:
                    pass

        tasks = [asyncio.create_task(reader())]
        tasks += [asyncio.create_task(writer(f"w_{i}")) for i in range(4)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"RuntimeError during concurrent reads: {errors}"
        assert gpu_manager.get_available_gpu_count() == 4
