# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for GPU Resource Manager concurrency scenarios.

Tests concurrent session GPU acquisition including contention, timeouts,
and release-while-waiting behaviour.
"""

import asyncio
import os
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch

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


@pytest.mark.unit
class TestTimeoutBehavior:
    """Tests for GPU acquisition timeout scenarios."""

    @pytest.mark.asyncio
    async def test_session_timeout(self, gpu_manager):
        """acquire_gpus_for_session_async() times out when insufficient GPUs."""
        # Hold 3 GPUs
        await gpu_manager.acquire_gpus_for_session_async("blocker", 3, timeout=5)

        start = time.time()
        with pytest.raises(TimeoutError, match="Could not acquire"):
            await gpu_manager.acquire_gpus_for_session_async("timeout_sess", 2, timeout=1)
        elapsed = time.time() - start

        assert 0.8 <= elapsed <= 2.0

        # Verify rollback: no partial allocations
        timeout_allocs = [a for a in gpu_manager._allocations.values() if a.allocation_id == "timeout_sess"]
        assert len(timeout_allocs) == 0

        # Only blocker's 3 GPUs remain
        assert gpu_manager.get_available_gpu_count() == 1

        gpu_manager.release_gpus_for_session("blocker")

    @pytest.mark.asyncio
    async def test_timeout_no_allocation_leak(self, gpu_manager):
        """After timeout, _allocations count returns to pre-timeout state."""
        initial_count = gpu_manager.get_available_gpu_count()
        assert initial_count == 4

        # Hold all GPUs
        await gpu_manager.acquire_gpus_for_session_async("holder", 4, timeout=5)

        # Timeout on session
        with pytest.raises(TimeoutError):
            await gpu_manager.acquire_gpus_for_session_async("leak_sess", 1, timeout=0.5)

        # Only holder's allocations exist
        assert len(gpu_manager._allocations) == 4
        assert all(a.allocation_id == "holder" for a in gpu_manager._allocations.values())

        gpu_manager.release_gpus_for_session("holder")
        assert gpu_manager.get_available_gpu_count() == 4


@pytest.mark.unit
class TestConcurrentSessionAndRelease:
    """Tests for concurrent session acquire and release operations."""

    @pytest.mark.asyncio
    async def test_release_while_another_session_waiting(self, gpu_manager):
        """Releasing one session allows a waiting session to acquire."""
        # Session 1 takes all 4 GPUs
        await gpu_manager.acquire_gpus_for_session_async("s1", 4, timeout=5)

        acquired = []

        async def waiting_session():
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async("s2", 2, timeout=5)
            acquired.extend(gpu_ids)

        # Start waiting session
        task = asyncio.create_task(waiting_session())
        await asyncio.sleep(0.3)  # Let it start waiting

        # Release session 1
        gpu_manager.release_gpus_for_session("s1")

        # Waiting session should now succeed
        await task
        assert len(acquired) == 2
        assert gpu_manager.get_available_gpu_count() == 2

        gpu_manager.release_gpus_for_session("s2")
        assert gpu_manager.get_available_gpu_count() == 4

    @pytest.mark.asyncio
    async def test_rapid_acquire_release_cycles(self, gpu_manager):
        """Rapid acquire/release cycles don't leak allocations."""
        for i in range(20):
            gpu_ids = await gpu_manager.acquire_gpus_for_session_async(
                f"rapid_{i}", 2, timeout=5,
            )
            assert len(gpu_ids) == 2
            gpu_manager.release_gpus_for_session(f"rapid_{i}")

        assert gpu_manager.get_available_gpu_count() == 4
        assert len(gpu_manager._allocations) == 0
