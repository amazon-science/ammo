# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for GPUFileLockManager filesystem discovery.

Tests that GPUFileLockManager discovers valid GPU IDs from existing lock files
instead of relying on torch.cuda.device_count(), which returns incorrect results
when CUDA_VISIBLE_DEVICES restricts GPU visibility.
"""

import fcntl
import os
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.gpu_file_lock import GPUFileLockManager, reset_gpu_lock_manager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances before each test."""
    reset_gpu_lock_manager()
    yield
    reset_gpu_lock_manager()


def _create_lock_files(lock_dir: str, gpu_ids: list[int]):
    """Helper to create lock files for given GPU IDs."""
    os.makedirs(lock_dir, exist_ok=True)
    for gpu_id in gpu_ids:
        open(os.path.join(lock_dir, f"gpu_{gpu_id}.lock"), "w").close()


@pytest.mark.unit
class TestFilesystemDiscovery:
    """Tests for filesystem-based GPU discovery."""

    def test_filesystem_discovery_finds_lock_files(self, tmp_path):
        """Lock files created by main process are discovered by subprocess."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, list(range(8)))

        # Even if CUDA sees only 1 GPU, filesystem discovery finds all 8
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            manager = GPUFileLockManager(lock_dir=lock_dir)

        assert manager.get_gpu_count() == 8
        assert manager.get_gpu_ids() == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_filesystem_discovery_fallback_to_detect(self, tmp_path):
        """Empty lock dir falls back to _detect_gpu_count."""
        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)
        # No lock files exist

        with patch.object(GPUFileLockManager, "_detect_gpu_count", return_value=4):
            manager = GPUFileLockManager(lock_dir=lock_dir)

        assert manager.get_gpu_count() == 4
        assert manager.get_gpu_ids() == [0, 1, 2, 3]
        # Fallback should have created lock files
        for i in range(4):
            assert os.path.exists(os.path.join(lock_dir, f"gpu_{i}.lock"))

    def test_acquire_system_gpu_id_under_cvd_restriction(self, tmp_path):
        """Can acquire system GPU ID even when CUDA_VISIBLE_DEVICES restricts visibility."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, list(range(8)))

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
            manager = GPUFileLockManager(lock_dir=lock_dir)

        # GPU 5 should be valid (not rejected as "must be 0-0")
        with manager.acquire(gpu_id=5, job_id="test") as acquired_id:
            assert acquired_id == 5

    def test_acquire_any_iterates_discovered_gpus(self, tmp_path):
        """acquire_any iterates over discovered GPU IDs, not range(gpu_count)."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, list(range(4)))

        manager = GPUFileLockManager(lock_dir=lock_dir)

        # Hold locks on GPUs 0-2 externally
        held_files = []
        for gpu_id in range(3):
            f = open(os.path.join(lock_dir, f"gpu_{gpu_id}.lock"), "w")
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held_files.append(f)

        try:
            with manager.acquire_any(timeout=2, job_id="test") as acquired_id:
                assert acquired_id == 3
        finally:
            for f in held_files:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()

    def test_get_gpu_ids_returns_sorted_list(self, tmp_path):
        """GPU IDs are returned in sorted order regardless of filesystem order."""
        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)
        # Create in non-sequential order
        for gpu_id in [5, 2, 0]:
            open(os.path.join(lock_dir, f"gpu_{gpu_id}.lock"), "w").close()

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0, 2, 5]
        assert manager.get_gpu_count() == 3

    def test_invalid_gpu_id_rejected(self, tmp_path):
        """GPU IDs not in discovered set are rejected."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1, 2, 3])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        with pytest.raises(ValueError, match="Invalid gpu_id 7"):
            with manager.acquire(gpu_id=7, job_id="test"):
                pass

    def test_non_lock_files_ignored(self, tmp_path):
        """Non-lock files in lock dir are ignored."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1])
        # Create non-lock files
        open(os.path.join(lock_dir, "readme.txt"), "w").close()
        open(os.path.join(lock_dir, "gpu_abc.lock"), "w").close()  # non-numeric

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0, 1]


@pytest.mark.unit
class TestGpuLockWrapperFilesystem:
    """Tests for gpu_lock_wrapper using filesystem discovery."""

    def test_show_status_uses_discovered_gpus(self, tmp_path, capsys):
        """show_status reports all discovered GPUs, not just CUDA-visible ones."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, list(range(8)))

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "GPU_LOCK_DIR": lock_dir}):
            from shared.gpu_lock_wrapper import show_status
            show_status()

        output = capsys.readouterr().out
        assert "Total GPUs: 8" in output
        # Should report all 8 GPUs
        for i in range(8):
            assert f"GPU {i}:" in output


# ============================================================
# Bug #9 — Stale lock file metadata after process crash
# ============================================================

@pytest.mark.unit
class TestStaleLockMetadata:
    """Tests for Bug #9: stale lock file metadata after process crash."""

    def test_get_lock_info_returns_none_for_dead_pid(self, tmp_path):
        """get_lock_info should return None when the PID in the lock file is dead."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        # Write a dead PID into the lock file manually (simulate crash)
        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        with open(lock_path, "w") as f:
            f.write(f"999999999\ndead_job\n{time.time()}")

        # Should return None (dead PID), not a dict with stale data
        result = manager.get_lock_info(0)
        assert result is None

    def test_get_lock_info_include_stale_returns_stale_entry(self, tmp_path):
        """get_lock_info with include_stale=True should return stale entries."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        with open(lock_path, "w") as f:
            f.write(f"999999999\ndead_job\n{time.time()}")

        result = manager.get_lock_info(0, include_stale=True)
        assert result is not None
        assert result["stale"] is True
        assert result["pid"] == 999999999
        assert result["job_id"] == "dead_job"

    def test_get_lock_info_returns_live_pid_info(self, tmp_path):
        """get_lock_info should return info when the PID is alive (our own PID)."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        # Write our own PID (guaranteed alive)
        with open(lock_path, "w") as f:
            f.write(f"{os.getpid()}\nlive_job\n{time.time()}")

        result = manager.get_lock_info(0)
        assert result is not None
        assert result["pid"] == os.getpid()
        assert result["job_id"] == "live_job"

    def test_clean_stale_locks_clears_dead_pid_files(self, tmp_path):
        """clean_stale_locks() clears metadata for dead PIDs."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path_0 = os.path.join(lock_dir, "gpu_0.lock")
        lock_path_1 = os.path.join(lock_dir, "gpu_1.lock")

        # GPU 0: dead PID
        with open(lock_path_0, "w") as f:
            f.write(f"999999999\ndead_job\n{time.time()}")

        # GPU 1: live PID (our own)
        with open(lock_path_1, "w") as f:
            f.write(f"{os.getpid()}\nlive_job\n{time.time()}")

        count = manager.clean_stale_locks()
        assert count == 1  # Only 1 stale lock cleaned

        # GPU 0 lock file should be empty after cleanup
        with open(lock_path_0, "r") as f:
            content = f.read().strip()
        assert content == ""

        # GPU 1 lock file should still have content
        with open(lock_path_1, "r") as f:
            content = f.read().strip()
        assert content != ""

    def test_clean_stale_locks_returns_zero_when_all_live(self, tmp_path):
        """clean_stale_locks() returns 0 when no stale locks exist."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        with open(lock_path, "w") as f:
            f.write(f"{os.getpid()}\nlive_job\n{time.time()}")

        count = manager.clean_stale_locks()
        assert count == 0


# ============================================================
# Bug #11 — Non-atomic lock file metadata writes
# ============================================================

@pytest.mark.unit
class TestNonAtomicMetadataWrites:
    """Tests for Bug #11: defensive parsing on partial/malformed data."""

    def test_get_lock_info_returns_none_for_empty_file(self, tmp_path):
        """get_lock_info returns None when lock file is empty."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)
        # File is empty (already the case from _create_lock_files)
        result = manager.get_lock_info(0)
        assert result is None

    def test_get_lock_info_returns_none_for_partial_data(self, tmp_path):
        """get_lock_info returns None when lock file has only partial content."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        # Write only 1 of 3 lines (simulate truncated write mid-operation)
        with open(lock_path, "w") as f:
            f.write("12345")  # only PID, missing job_id and timestamp

        result = manager.get_lock_info(0)
        assert result is None

    def test_get_lock_info_returns_none_for_non_integer_pid(self, tmp_path):
        """get_lock_info returns None when PID field is not an integer."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        with open(lock_path, "w") as f:
            f.write(f"NOT_A_PID\nsome_job\n{time.time()}")

        result = manager.get_lock_info(0)
        assert result is None

    def test_get_lock_info_returns_none_for_non_float_timestamp(self, tmp_path):
        """get_lock_info returns None when timestamp field is not a float."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        lock_path = os.path.join(lock_dir, "gpu_0.lock")
        with open(lock_path, "w") as f:
            f.write(f"{os.getpid()}\nsome_job\nNOT_A_FLOAT")

        result = manager.get_lock_info(0)
        assert result is None


# ============================================================
# Bug #12 — No writability check on lock directory init
# ============================================================

@pytest.mark.unit
class TestLockDirWritabilityCheck:
    """Tests for Bug #12: writability check at init."""

    def test_init_raises_on_unwritable_directory(self, tmp_path):
        """__init__ raises RuntimeError when lock directory is not writable."""
        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)

        # Make directory unwritable
        os.chmod(lock_dir, 0o555)
        try:
            with pytest.raises(RuntimeError, match="not writable"):
                GPUFileLockManager(lock_dir=lock_dir)
        finally:
            # Restore permissions for cleanup
            os.chmod(lock_dir, 0o755)

    def test_init_succeeds_on_writable_directory(self, tmp_path):
        """__init__ succeeds when lock directory is writable."""
        lock_dir = str(tmp_path / "gpu_locks")

        with patch.object(GPUFileLockManager, "_detect_gpu_count", return_value=2):
            manager = GPUFileLockManager(lock_dir=lock_dir)

        assert manager is not None
        assert manager.get_gpu_count() == 2

    def test_init_raises_on_probe_write_failure(self, tmp_path):
        """__init__ raises RuntimeError when probe write fails."""
        lock_dir = str(tmp_path / "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)

        # Simulate PermissionError on open by patching builtins.open for probe file
        original_open = open

        def patched_open(path, mode="r", *args, **kwargs):
            if ".probe" in str(path):
                raise PermissionError("Permission denied")
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            with pytest.raises(RuntimeError, match="not writable"):
                GPUFileLockManager(lock_dir=lock_dir)


# ============================================================
# Bug #13 — Stale GPU IDs after lock file deletion
# ============================================================

@pytest.mark.unit
class TestStaleGpuIdsAfterDeletion:
    """Tests for Bug #13: stale GPU IDs after lock file deletion."""

    def test_refresh_gpu_ids_discovers_new_lock_files(self, tmp_path):
        """refresh_gpu_ids() re-runs filesystem discovery and updates _gpu_ids."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1])

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0, 1]

        # Add a new lock file after init
        open(os.path.join(lock_dir, "gpu_2.lock"), "w").close()

        # Before refresh, IDs should still be stale
        assert 2 not in manager.get_gpu_ids()

        # After refresh, should pick up new file
        manager.refresh_gpu_ids()
        assert manager.get_gpu_ids() == [0, 1, 2]

    def test_refresh_gpu_ids_handles_deleted_lock_files(self, tmp_path):
        """refresh_gpu_ids() handles case where lock files are deleted."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1, 2])

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0, 1, 2]

        # Delete a lock file
        os.remove(os.path.join(lock_dir, "gpu_2.lock"))

        manager.refresh_gpu_ids()
        assert manager.get_gpu_ids() == [0, 1]

    def test_acquire_retries_with_refresh_on_invalid_gpu_id(self, tmp_path):
        """acquire() tries refresh when gpu_id not in _gpu_ids, then raises if still invalid."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1])

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0, 1]

        # GPU 5 was never created, should raise ValueError after refresh attempt
        with pytest.raises(ValueError, match="Invalid gpu_id 5"):
            with manager.acquire(gpu_id=5):
                pass

        # Internal _gpu_ids should not have changed (still only 0, 1)
        assert manager.get_gpu_ids() == [0, 1]

    def test_acquire_succeeds_after_lock_files_restored(self, tmp_path):
        """acquire() succeeds if lock files are deleted and then re-created."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0, 1])

        manager = GPUFileLockManager(lock_dir=lock_dir)

        # Delete both lock files
        os.remove(os.path.join(lock_dir, "gpu_0.lock"))
        os.remove(os.path.join(lock_dir, "gpu_1.lock"))

        # Re-create gpu_0.lock
        open(os.path.join(lock_dir, "gpu_0.lock"), "w").close()

        # After refresh, only GPU 0 is valid
        manager.refresh_gpu_ids()
        with manager.acquire(gpu_id=0):
            pass  # should succeed

    def test_acquire_any_retries_with_refresh_on_empty_ids(self, tmp_path):
        """acquire_any() refreshes and raises when _gpu_ids is empty after all lock files deleted."""
        lock_dir = str(tmp_path / "gpu_locks")
        _create_lock_files(lock_dir, [0])

        manager = GPUFileLockManager(lock_dir=lock_dir)
        assert manager.get_gpu_ids() == [0]

        # Manually clear internal state to simulate stale IDs scenario
        manager._gpu_ids = []
        manager._gpu_count = 0

        # Delete actual lock file too so refresh confirms no GPUs
        os.remove(os.path.join(lock_dir, "gpu_0.lock"))

        # acquire_any with empty _gpu_ids should refresh then raise (no valid GPUs)
        with pytest.raises((ValueError, RuntimeError)):
            with manager.acquire_any(timeout=0.2):
                pass
