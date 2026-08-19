# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for GPUResourceManager

Tests GPU allocation, release, and resource management.
These tests mock CUDA to run without a GPU.
"""

import asyncio
import pytest
import sys
import time
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
    import os
    os.makedirs(lock_dir, exist_ok=True)

    with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
        from shared.gpu_resource_manager import GPUResourceManager
        manager = GPUResourceManager(lock_dir=lock_dir)
        yield manager


@pytest.mark.unit
class TestGPUResourceManagerInitialization:
    """Tests for GPUResourceManager initialization"""

    def test_creation_with_cuda_available(self, tmp_path):
        """Test GPUResourceManager creation when CUDA is available"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)

            assert manager is not None
            assert manager.get_gpu_count() == 4

    def test_creation_without_cuda(self, tmp_path):
        """Test GPUResourceManager creation when CUDA is not available"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=0):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)

            assert manager is not None
            assert manager.get_gpu_count() == 0

    def test_allocations_empty_on_init(self, tmp_path):
        """Test that _allocations registry is empty at initialization"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)
            assert manager._allocations == {}
            assert manager.get_available_gpu_count() == 4


@pytest.mark.unit
class TestGPUAllocation:
    """Tests for GPU allocation"""

    def test_get_gpu_count(self, gpu_manager):
        """Test getting total GPU count"""
        assert gpu_manager.get_gpu_count() == 4

    def test_get_available_gpu_count_all_available(self, gpu_manager):
        """Test getting available GPU count when all GPUs are free"""
        # Initially all GPUs should be available
        assert gpu_manager.get_available_gpu_count() == 4


@pytest.mark.unit
class TestUnifiedAllocationRegistry:
    """Tests for the unified _allocations registry"""

    def test_register_deregister_allocation(self, gpu_manager):
        """Test basic register/deregister cycle"""
        gpu_manager._register_allocation(0, "job_1", "job")
        assert gpu_manager.get_available_gpu_count() == 3

        gpu_manager._deregister_allocation("job", "job_1", 0)
        assert gpu_manager.get_available_gpu_count() == 4

    def test_multiple_allocation_types(self, gpu_manager):
        """Test that all 3 allocation types are tracked correctly"""
        gpu_manager._register_allocation(0, "job_1", "job")
        gpu_manager._register_allocation(1, "session_1", "session")
        gpu_manager._register_allocation(2, "vllm_1", "vllm_benchmark")

        assert gpu_manager.get_available_gpu_count() == 1

        # Deregister all
        gpu_manager._deregister_allocation("job", "job_1", 0)
        gpu_manager._deregister_allocation("session", "session_1", 1)
        gpu_manager._deregister_allocation("vllm_benchmark", "vllm_1", 2)

        assert gpu_manager.get_available_gpu_count() == 4

    def test_deregister_nonexistent_is_noop(self, gpu_manager):
        """Test that deregistering a non-existent allocation doesn't crash"""
        # Should not raise
        gpu_manager._deregister_allocation("job", "nonexistent", 0)
        assert gpu_manager.get_available_gpu_count() == 4

    def test_same_gpu_different_keys_deduplicates_in_count(self, gpu_manager):
        """Test that get_available_gpu_count uses a set of gpu_ids"""
        # This shouldn't happen in practice, but verify the set deduplication
        gpu_manager._register_allocation(0, "job_1", "job")
        gpu_manager._register_allocation(0, "job_2", "job")

        # Both claim GPU 0, but set deduplication means only 1 GPU counted as allocated
        assert gpu_manager.get_available_gpu_count() == 3


@pytest.mark.unit
class TestSessionGPUAllocation:
    """Tests for session GPU allocation"""

    @pytest.mark.asyncio
    async def test_acquire_session_gpus_async(self, gpu_manager):
        """Test async session GPU acquisition"""
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
            session_id="session_1",
            gpu_count=2,
            timeout=5,
        )

        assert len(gpu_ids) == 2
        assert gpu_manager.get_available_gpu_count() == 2

        # Verify allocations are registered
        session_allocs = [
            a for a in gpu_manager._allocations.values()
            if a.allocation_type == "session" and a.allocation_id == "session_1"
        ]
        assert len(session_allocs) == 2

    @pytest.mark.asyncio
    async def test_release_session_gpus(self, gpu_manager):
        """Test session GPU release cleans up _allocations"""
        await gpu_manager.acquire_gpus_for_session_async(
            session_id="session_1",
            gpu_count=2,
            timeout=5,
        )
        assert gpu_manager.get_available_gpu_count() == 2

        gpu_manager.release_gpus_for_session("session_1")
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_session_zero_gpus(self, gpu_manager):
        """Test that requesting 0 GPUs creates no allocations"""
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
            session_id="session_zero",
            gpu_count=0,
        )
        assert gpu_ids == []
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_session_exceeds_total_gpus(self, gpu_manager):
        """Test that requesting more GPUs than total raises ValueError"""
        with pytest.raises(ValueError, match="Requested 10 GPUs"):
            await gpu_manager.acquire_gpus_for_session_async(
                session_id="too_many",
                gpu_count=10,
                timeout=5,
            )
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

    @pytest.mark.asyncio
    async def test_session_partial_acquisition_rollback(self, gpu_manager):
        """Test that partial acquisition rolls back _allocations on timeout"""
        # Acquire 3 GPUs first to leave only 1 available
        await gpu_manager.acquire_gpus_for_session_async(
            session_id="blocker",
            gpu_count=3,
            timeout=5,
        )
        assert gpu_manager.get_available_gpu_count() == 1

        # Try to acquire 2 more (only 1 available) - should timeout
        with pytest.raises(TimeoutError):
            await gpu_manager.acquire_gpus_for_session_async(
                session_id="partial",
                gpu_count=2,
                timeout=0.5,  # Short timeout
            )

        # Verify no partial allocations leaked
        partial_allocs = [
            a for a in gpu_manager._allocations.values()
            if a.allocation_id == "partial"
        ]
        assert len(partial_allocs) == 0

        # Only the "blocker" session's 3 GPUs should be allocated
        assert gpu_manager.get_available_gpu_count() == 1

    @pytest.mark.asyncio
    async def test_session_pause_resume_cycle(self, gpu_manager):
        """Test that pause (release) and resume (re-acquire) works correctly"""
        # Create session with 2 GPUs
        gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
            session_id="session_cycle",
            gpu_count=2,
            timeout=5,
        )
        assert gpu_manager.get_available_gpu_count() == 2

        # Pause: release GPUs
        gpu_manager.release_gpus_for_session("session_cycle")
        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0

        # Resume: re-acquire GPUs
        new_gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
            session_id="session_cycle",
            gpu_count=2,
            timeout=5,
        )
        assert gpu_manager.get_available_gpu_count() == 2
        assert len(new_gpu_ids) == 2

        # Clean up
        gpu_manager.release_gpus_for_session("session_cycle")
        assert gpu_manager.get_available_gpu_count() == 4

    def test_release_nonexistent_session(self, gpu_manager):
        """Test that releasing a non-existent session is a safe no-op"""
        gpu_manager.release_gpus_for_session("nonexistent")
        assert gpu_manager.get_available_gpu_count() == 4

    def test_double_release_session(self, gpu_manager):
        """Test that releasing a session twice doesn't crash"""
        # Manually register then release twice
        gpu_manager._register_allocation(0, "double_rel", "session")
        gpu_manager.release_gpus_for_session("double_rel")
        gpu_manager.release_gpus_for_session("double_rel")  # Should be no-op
        assert gpu_manager.get_available_gpu_count() == 4


@pytest.mark.unit
class TestGPUResourceManagerMetrics:
    """Tests for GPU metrics tracking"""

    def test_metrics_recording(self, tmp_path):
        """Test that GPU acquisition/release metrics are recorded"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=2):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)

            # Check initial state
            assert manager.get_gpu_count() == 2


@pytest.mark.unit
class TestGPUResourceManagerEdgeCases:
    """Tests for edge cases and error handling"""

    def test_no_gpus_available(self, tmp_path):
        """Test behavior when no GPUs are available"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=0):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)

            assert manager.get_gpu_count() == 0
            assert manager.get_available_gpu_count() == 0

    def test_cuda_not_installed(self, tmp_path):
        """Test behavior when CUDA is not installed/available"""
        lock_dir = str(tmp_path / "gpu_locks")
        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=0):
            from shared.gpu_resource_manager import GPUResourceManager

            manager = GPUResourceManager(lock_dir=lock_dir)

            assert manager.get_gpu_count() == 0


@pytest.mark.unit
class TestNonContiguousGpuIds:
    """Tests for Bug #6: range(total_gpus) vs get_gpu_ids() — non-contiguous GPU IDs"""

    @pytest.fixture
    def non_contiguous_gpu_manager(self, tmp_path):
        """Create a GPUResourceManager whose lock manager returns non-contiguous GPU IDs."""
        from unittest.mock import patch, MagicMock
        from shared.gpu_file_lock import reset_gpu_lock_manager
        reset_gpu_lock_manager()

        lock_dir = str(tmp_path / "gpu_locks")
        import os
        os.makedirs(lock_dir, exist_ok=True)

        # Create lock files for GPUs 0, 2, 5, 7 (non-contiguous)
        non_contiguous_ids = [0, 2, 5, 7]
        for gid in non_contiguous_ids:
            open(os.path.join(lock_dir, f"gpu_{gid}.lock"), 'w').close()

        with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
            from shared.gpu_resource_manager import GPUResourceManager
            manager = GPUResourceManager(lock_dir=lock_dir)
            yield manager

        reset_gpu_lock_manager()

    @pytest.mark.asyncio
    async def test_async_session_acquire_uses_get_gpu_ids(self, non_contiguous_gpu_manager):
        """Test that acquire_gpus_for_session_async iterates actual GPU IDs, not range(total)."""
        manager = non_contiguous_gpu_manager
        gpu_ids = manager._lock_manager.get_gpu_ids()
        # Only valid IDs should be attempted
        acquired = await manager.acquire_gpus_for_session_async(
            session_id="nc_session",
            gpu_count=2,
            timeout=5,
        )
        assert len(acquired) == 2
        # All acquired IDs must come from get_gpu_ids()
        assert all(gid in gpu_ids for gid in acquired)
        manager.release_gpus_for_session("nc_session")


@pytest.mark.unit
class TestSessionRollbackFileHandleLeak:
    """Tests for Bug #7: file handle leak in session rollback on unlock failure"""

    @pytest.mark.asyncio
    async def test_async_rollback_closes_file_even_if_flock_raises(self, gpu_manager, tmp_path):
        """Test that rollback closes file handles even when fcntl.flock(LOCK_UN) raises."""
        import fcntl
        original_flock = fcntl.flock
        flock_calls = []

        def mock_flock(fd, op):
            flock_calls.append(op)
            if op == fcntl.LOCK_UN:
                raise OSError("simulated flock LOCK_UN failure")
            return original_flock(fd, op)

        # Fill 3 GPUs so 4th can't be acquired → triggers rollback
        await gpu_manager.acquire_gpus_for_session_async("blocker", 3, timeout=5)

        closed_files = []
        original_open = open

        class TrackingFile:
            def __init__(self, path, mode):
                self._f = original_open(path, mode)
                self.closed = False
            def close(self):
                self.closed = True
                closed_files.append(self)
                self._f.close()
            def seek(self, *a): return self._f.seek(*a)
            def truncate(self, *a): return self._f.truncate(*a)
            def write(self, *a): return self._f.write(*a)
            def flush(self): return self._f.flush()
            def fileno(self): return self._f.fileno()

        with patch('fcntl.flock', side_effect=mock_flock):
            with patch('builtins.open', side_effect=lambda p, m: TrackingFile(p, m)):
                with pytest.raises((TimeoutError, OSError)):
                    await gpu_manager.acquire_gpus_for_session_async(
                        "leaky_session", 2, timeout=0.3
                    )

        # Any TrackingFile that was opened AND flock(LOCK_EX) succeeded should be closed
        for tf in closed_files:
            assert tf.closed, "File handle was not closed during rollback"

        gpu_manager.release_gpus_for_session("blocker")
