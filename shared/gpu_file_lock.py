# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
File-based GPU lock manager for container-level coordination.

Each GPU has a lock file (/tmp/gpu_locks/gpu_{id}.lock).
Uses fcntl.flock() for POSIX file locking.

This allows:
- Cross-process GPU coordination within a container
- Subprocesses to acquire/release GPU locks directly
- Partial GPU usage (acquire only when needed, release when done)
"""

import asyncio
import fcntl
import os
import time
import logging
from typing import Optional
from contextlib import contextmanager, asynccontextmanager

logger = logging.getLogger(__name__)


class GPUFileLockManager:
    """
    File-based GPU lock manager for container-level coordination.

    Each GPU has a lock file (/tmp/gpu_locks/gpu_{id}.lock).
    Uses fcntl.flock() for POSIX file locking.
    """

    def __init__(self, lock_dir: str = "/tmp/gpu_locks"):
        self.lock_dir = lock_dir
        os.makedirs(self.lock_dir, exist_ok=True)
        # Bug #12: verify lock directory is writable before proceeding
        self._check_lock_dir_writable()
        discovered = self._discover_from_filesystem()
        if discovered:
            self._gpu_ids = discovered
            self._gpu_count = len(discovered)
            logger.info(f"GPUFileLockManager: discovered {self._gpu_count} GPUs from lock files: {self._gpu_ids}")
        else:
            self._gpu_count = self._detect_gpu_count()
            self._gpu_ids = list(range(self._gpu_count))
            self._ensure_lock_dir()
            logger.info(f"GPUFileLockManager: detected {self._gpu_count} GPUs via CUDA runtime")
        self._active_locks: dict = {}  # gpu_id -> (file_handle, lock_held)

    def _check_lock_dir_writable(self):
        """Bug #12: verify lock directory is writable by creating a temp probe file."""
        probe_path = os.path.join(self.lock_dir, ".probe_writable")
        try:
            with open(probe_path, "w") as f:
                f.write("probe")
            os.remove(probe_path)
        except Exception as e:
            raise RuntimeError(
                f"Lock directory '{self.lock_dir}' is not writable: {e}"
            ) from e

    def _discover_from_filesystem(self) -> list[int]:
        """Discover valid GPU IDs from existing lock files in lock_dir.

        Returns sorted list of GPU IDs found, or empty list if no lock files exist.
        """
        gpu_ids = []
        try:
            for fname in os.listdir(self.lock_dir):
                if fname.startswith("gpu_") and fname.endswith(".lock"):
                    try:
                        gpu_id = int(fname[4:-5])  # strip "gpu_" and ".lock"
                        gpu_ids.append(gpu_id)
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        return sorted(gpu_ids)

    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                logger.debug(f"Detected {count} GPUs via PyTorch")
                return count
        except ImportError:
            pass

        # Fallback: check NVIDIA driver
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
                count = len(lines)
                logger.debug(f"Detected {count} GPUs via nvidia-smi")
                return count
        except Exception as e:
            logger.warning(f"Failed to detect GPUs via nvidia-smi: {e}")

        logger.warning("No GPUs detected")
        return 0

    def _ensure_lock_dir(self):
        """Create lock directory and lock files if they don't exist."""
        os.makedirs(self.lock_dir, exist_ok=True)
        # Create lock files for each GPU
        for gpu_id in range(self._gpu_count):
            lock_path = self._get_lock_path(gpu_id)
            if not os.path.exists(lock_path):
                open(lock_path, 'w').close()
                logger.debug(f"Created lock file: {lock_path}")

    def _get_lock_path(self, gpu_id: int) -> str:
        """Get path to lock file for given GPU."""
        return os.path.join(self.lock_dir, f"gpu_{gpu_id}.lock")

    @contextmanager
    def acquire(self, gpu_id: int, timeout: int = 300, job_id: Optional[str] = None):
        """
        Acquire lock on specific GPU.

        Args:
            gpu_id: GPU device ID
            timeout: Max seconds to wait for lock
            job_id: Optional job identifier for logging

        Yields:
            gpu_id if acquired

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
            ValueError: If gpu_id is invalid
        """
        if gpu_id not in self._gpu_ids:
            # Bug #13: try refreshing GPU IDs once before raising
            self.refresh_gpu_ids()
            if gpu_id not in self._gpu_ids:
                raise ValueError(f"Invalid gpu_id {gpu_id}, valid IDs: {self._gpu_ids}")

        lock_path = self._get_lock_path(gpu_id)
        lock_file = open(lock_path, 'w')
        start_time = time.time()
        acquired = False

        try:
            while time.time() - start_time < timeout:
                try:
                    # Try non-blocking lock
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    # Write PID and job_id for debugging
                    lock_file.seek(0)
                    lock_file.truncate()
                    lock_file.write(f"{os.getpid()}\n{job_id or 'unknown'}\n{time.time()}")
                    lock_file.flush()
                    wait_time = time.time() - start_time
                    logger.info(f"Acquired GPU {gpu_id} lock (job={job_id}) after {wait_time:.2f}s")
                    self._active_locks[gpu_id] = (lock_file, True)
                    yield gpu_id
                    return
                except BlockingIOError:
                    # Lock held by another process, wait and retry
                    time.sleep(0.1)

            # Timeout reached
            raise TimeoutError(f"Failed to acquire GPU {gpu_id} lock within {timeout}s")

        finally:
            if acquired:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    logger.info(f"Released GPU {gpu_id} lock (job={job_id})")
                except Exception as e:
                    logger.error(f"Error releasing GPU {gpu_id} lock: {e}")
                finally:
                    self._active_locks.pop(gpu_id, None)
            lock_file.close()

    @contextmanager
    def acquire_any(self, timeout: int = 300, job_id: Optional[str] = None):
        """
        Acquire lock on any available GPU.

        Tries each GPU in order, returns first available.

        Args:
            timeout: Max seconds to wait for any GPU
            job_id: Optional job identifier for logging

        Yields:
            gpu_id of acquired GPU

        Raises:
            TimeoutError: If no GPU can be acquired within timeout
        """
        if self._gpu_count == 0:
            raise RuntimeError("No GPUs available")

        start_time = time.time()
        _refreshed = False

        while time.time() - start_time < timeout:
            if not self._gpu_ids:
                # Bug #13: GPU IDs may have gone stale, refresh once
                if not _refreshed:
                    self.refresh_gpu_ids()
                    _refreshed = True
                if not self._gpu_ids:
                    raise ValueError(f"No valid GPU IDs after refresh in lock dir: {self.lock_dir}")
            for gpu_id in self._gpu_ids:
                lock_path = self._get_lock_path(gpu_id)
                lock_file = open(lock_path, 'w')
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Got lock
                    lock_file.seek(0)
                    lock_file.truncate()
                    lock_file.write(f"{os.getpid()}\n{job_id or 'unknown'}\n{time.time()}")
                    lock_file.flush()
                    wait_time = time.time() - start_time
                    logger.info(f"Acquired GPU {gpu_id} lock (job={job_id}) after {wait_time:.2f}s")
                    self._active_locks[gpu_id] = (lock_file, True)
                    try:
                        yield gpu_id
                    finally:
                        try:
                            fcntl.flock(lock_file, fcntl.LOCK_UN)
                            logger.info(f"Released GPU {gpu_id} lock (job={job_id})")
                        except Exception as e:
                            logger.error(f"Error releasing GPU {gpu_id} lock: {e}")
                        finally:
                            self._active_locks.pop(gpu_id, None)
                            lock_file.close()
                    return
                except BlockingIOError:
                    lock_file.close()
                    continue

            # All GPUs busy, wait before retry
            time.sleep(0.1)

        raise TimeoutError(f"No GPU available within {timeout} seconds")

    @asynccontextmanager
    async def acquire_any_async(self, timeout: int = 300, job_id: Optional[str] = None):
        """
        Async version of acquire_any - acquires lock on any available GPU.

        Uses non-blocking flock calls with async sleep for retry, allowing
        the asyncio event loop to remain responsive during GPU wait.

        Args:
            timeout: Max seconds to wait for any GPU
            job_id: Optional job identifier for logging

        Yields:
            gpu_id of acquired GPU

        Raises:
            TimeoutError: If no GPU can be acquired within timeout
        """
        if self._gpu_count == 0:
            raise RuntimeError("No GPUs available")

        start_time = time.time()
        acquired_gpu_id = None
        lock_file = None

        try:
            while time.time() - start_time < timeout:
                for gpu_id in self._gpu_ids:
                    lock_path = self._get_lock_path(gpu_id)
                    lock_file = open(lock_path, 'w')
                    try:
                        # Non-blocking lock attempt (returns immediately)
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Got lock
                        lock_file.seek(0)
                        lock_file.truncate()
                        lock_file.write(f"{os.getpid()}\n{job_id or 'unknown'}\n{time.time()}")
                        lock_file.flush()
                        wait_time = time.time() - start_time
                        logger.info(f"Acquired GPU {gpu_id} lock (job={job_id}) after {wait_time:.2f}s")
                        self._active_locks[gpu_id] = (lock_file, True)
                        acquired_gpu_id = gpu_id
                        yield gpu_id
                        return
                    except BlockingIOError:
                        lock_file.close()
                        lock_file = None
                        continue

                # All GPUs busy, async sleep to allow other coroutines to run
                await asyncio.sleep(0.1)

            raise TimeoutError(f"No GPU available within {timeout} seconds")

        finally:
            if acquired_gpu_id is not None and lock_file is not None:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    logger.info(f"Released GPU {acquired_gpu_id} lock (job={job_id})")
                except Exception as e:
                    logger.error(f"Error releasing GPU {acquired_gpu_id} lock: {e}")
                finally:
                    self._active_locks.pop(acquired_gpu_id, None)
                    lock_file.close()

    def get_gpu_count(self) -> int:
        """Get total number of GPUs."""
        return self._gpu_count

    def get_gpu_ids(self) -> list[int]:
        """Get list of valid GPU IDs."""
        return list(self._gpu_ids)

    def get_available_gpu_count(self) -> int:
        """Get number of currently unlocked GPUs (from in-memory tracking)."""
        return self._gpu_count - len(self._active_locks)

    def get_lock_info(self, gpu_id: int, include_stale: bool = False) -> Optional[dict]:
        """
        Get info about who holds a GPU lock (for debugging).

        Args:
            gpu_id: GPU device ID
            include_stale: If True, return stale entries (dead PIDs) with stale=True.
                           If False (default), return None for dead PIDs.

        Returns:
            Dict with pid, job_id, acquired_at if lock file has info and PID is alive,
            else None. If include_stale=True, returns dict with stale=True for dead PIDs.
        """
        if gpu_id not in self._gpu_ids:
            return None

        lock_path = self._get_lock_path(gpu_id)
        try:
            with open(lock_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return None
                lines = content.split('\n')
                # Bug #11: defensive parsing — return None on malformed data
                if len(lines) < 3:
                    return None
                try:
                    pid = int(lines[0])
                    job_id = lines[1]
                    acquired_at = float(lines[2])
                except (ValueError, IndexError):
                    return None

                # Bug #9: validate PID is alive
                pid_alive = True
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pid_alive = False
                except PermissionError:
                    # Process exists but we don't have permission to signal it
                    pid_alive = True

                if not pid_alive:
                    if include_stale:
                        return {
                            "pid": pid,
                            "job_id": job_id,
                            "acquired_at": acquired_at,
                            "gpu_id": gpu_id,
                            "stale": True,
                        }
                    return None

                return {
                    "pid": pid,
                    "job_id": job_id,
                    "acquired_at": acquired_at,
                    "gpu_id": gpu_id,
                }
        except Exception:
            pass
        return None

    def clean_stale_locks(self) -> int:
        """
        Scan all lock files and clear metadata for dead PIDs (Bug #9).

        Returns:
            Number of stale lock files cleaned.
        """
        cleaned = 0
        for gpu_id in self._gpu_ids:
            info = self.get_lock_info(gpu_id, include_stale=True)
            if info is not None and info.get("stale"):
                lock_path = self._get_lock_path(gpu_id)
                try:
                    with open(lock_path, 'w') as f:
                        f.truncate(0)
                    cleaned += 1
                    logger.info(f"Cleaned stale lock for GPU {gpu_id} (dead PID {info['pid']})")
                except Exception as e:
                    logger.warning(f"Failed to clean stale lock for GPU {gpu_id}: {e}")
        return cleaned

    def refresh_gpu_ids(self):
        """
        Re-run filesystem discovery to update _gpu_ids (Bug #13).

        Call this if lock files may have been added or removed since init.
        """
        discovered = self._discover_from_filesystem()
        self._gpu_ids = discovered
        self._gpu_count = len(discovered)
        logger.info(f"GPUFileLockManager: refreshed GPU IDs from filesystem: {self._gpu_ids}")

    def get_all_lock_info(self) -> list:
        """Get lock info for all GPUs."""
        return [self.get_lock_info(gpu_id) for gpu_id in self._gpu_ids]

    def is_gpu_locked(self, gpu_id: int) -> bool:
        """Check if a specific GPU is currently locked (from in-memory tracking)."""
        if gpu_id not in self._gpu_ids:
            return False
        return gpu_id in self._active_locks


# Singleton instance
_gpu_lock_manager: Optional[GPUFileLockManager] = None


def get_gpu_lock_manager(lock_dir: str = "/tmp/gpu_locks") -> GPUFileLockManager:
    """
    Get singleton GPU lock manager.

    Args:
        lock_dir: Directory for lock files (only used on first call)

    Returns:
        GPUFileLockManager instance
    """
    global _gpu_lock_manager
    if _gpu_lock_manager is None:
        _gpu_lock_manager = GPUFileLockManager(lock_dir=lock_dir)
    return _gpu_lock_manager


def reset_gpu_lock_manager():
    """Reset the singleton instance (for testing)."""
    global _gpu_lock_manager
    _gpu_lock_manager = None
