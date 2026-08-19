# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Centralized GPU resource management for AMMO sessions.

Uses file-based locks for container-level coordination across processes.
Single instance created in the app lifespan and injected into the session manager.

Supports long-lived session GPU allocation via explicit acquire/release.

All GPU usage is tracked in a unified _allocations registry, which is the
sole source of truth for get_available_gpu_count().
"""

import asyncio
import fcntl
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

from shared.gpu_file_lock import GPUFileLockManager, get_gpu_lock_manager
from shared.constants import GPU_TIMEOUT_SESSION

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Tracks a single GPU allocation across all use cases."""
    gpu_id: int
    allocation_id: str          # session_id
    allocation_type: str        # "session" (only type produced today)
    acquired_at: float
    lock_handle: Optional[Any] = None   # file handle (None for subprocess-held locks)
    pid: Optional[int] = None           # subprocess PID (None for main-process locks)


class GPUResourceManager:
    """
    Centralized GPU allocation manager using file locks.

    Uses file-based locking (/tmp/gpu_locks/) for cross-process coordination
    within a container. This allows:
    - Subprocesses to acquire/release GPU locks directly
    - Partial GPU usage (acquire only when needed)
    - Container-level coordination (not just process-local)

    All GPU usage is tracked in the _allocations dict, which serves as the
    sole source of truth for availability counting. The underlying file locks
    (managed by GPUFileLockManager) prevent actual concurrent GPU access;
    _allocations provides visibility to the main process.
    """

    def __init__(self, lock_dir: str = "/tmp/gpu_locks"):
        self._lock_manager = get_gpu_lock_manager(lock_dir=lock_dir)
        self._lock_dir = lock_dir
        # Unified allocation registry: key -> GPUAllocation
        # Key format: "{allocation_type}:{allocation_id}:{gpu_id}"
        self._allocations: Dict[str, GPUAllocation] = {}
        logger.info(
            f"GPUResourceManager initialized with {self._lock_manager.get_gpu_count()} GPUs (file locks)"
        )

    # =========================================================================
    # Allocation Registry (internal)
    # =========================================================================

    def _allocation_key(self, allocation_type: str, allocation_id: str, gpu_id: int) -> str:
        return f"{allocation_type}:{allocation_id}:{gpu_id}"

    def _register_allocation(
        self,
        gpu_id: int,
        allocation_id: str,
        allocation_type: str,
        lock_handle: Optional[Any] = None,
        pid: Optional[int] = None,
    ) -> None:
        key = self._allocation_key(allocation_type, allocation_id, gpu_id)
        self._allocations[key] = GPUAllocation(
            gpu_id=gpu_id,
            allocation_id=allocation_id,
            allocation_type=allocation_type,
            acquired_at=time.time(),
            lock_handle=lock_handle,
            pid=pid,
        )

    def _deregister_allocation(self, allocation_type: str, allocation_id: str, gpu_id: int) -> None:
        key = self._allocation_key(allocation_type, allocation_id, gpu_id)
        self._allocations.pop(key, None)

    # =========================================================================
    # GPU Availability
    # =========================================================================

    def get_gpu_count(self) -> int:
        """Get total number of available GPUs."""
        return self._lock_manager.get_gpu_count()

    def get_available_gpu_count(self) -> int:
        """
        Get number of currently available GPUs.

        Uses the unified _allocations registry as the sole source of truth.
        """
        # Bug #1 fix: snapshot values to avoid RuntimeError on concurrent modification
        allocated_gpu_ids = set(a.gpu_id for a in list(self._allocations.values()))
        return max(0, self._lock_manager.get_gpu_count() - len(allocated_gpu_ids))

    # =========================================================================
    # Session GPU Allocation (long-lived, explicit acquire/release)
    # =========================================================================

    async def acquire_gpus_for_session_async(
        self,
        session_id: str,
        gpu_count: int,
        timeout: int = GPU_TIMEOUT_SESSION,
    ) -> List[int]:
        """
        Acquire multiple GPUs for a session (long-lived allocation).

        Non-blocking async version that uses asyncio.sleep for the retry loop,
        keeping the FastAPI event loop responsive during GPU contention.

        Locks are held until explicitly released via release_gpus_for_session().

        Args:
            session_id: Session identifier (used for tracking and logging)
            gpu_count: Number of GPUs to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            List of acquired GPU IDs

        Raises:
            ValueError: If gpu_count exceeds total GPUs
            TimeoutError: If cannot acquire requested GPUs within timeout
        """
        total_gpus = self._lock_manager.get_gpu_count()

        if gpu_count > total_gpus:
            raise ValueError(
                f"Requested {gpu_count} GPUs but only {total_gpus} available"
            )

        if gpu_count == 0:
            return []

        acquired_locks: List[Tuple[int, Any]] = []
        acquired_gpu_ids: List[int] = []
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                # Bug #6 fix: use get_gpu_ids() to support non-contiguous GPU IDs
                for gpu_id in self._lock_manager.get_gpu_ids():
                    if len(acquired_gpu_ids) >= gpu_count:
                        break

                    if gpu_id in acquired_gpu_ids:
                        continue

                    lock_path = os.path.join(self._lock_dir, f"gpu_{gpu_id}.lock")
                    lock_file = open(lock_path, 'w')

                    try:
                        # Non-blocking lock attempt (returns immediately)
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                        lock_file.seek(0)
                        lock_file.truncate()
                        lock_file.write(f"{os.getpid()}\nsession:{session_id}\n{time.time()}")
                        lock_file.flush()

                        acquired_locks.append((gpu_id, lock_file))
                        acquired_gpu_ids.append(gpu_id)
                        self._register_allocation(
                            gpu_id, session_id, "session", lock_handle=lock_file,
                        )
                        logger.info(f"Session {session_id}: Acquired GPU {gpu_id}")

                    except BlockingIOError:
                        lock_file.close()
                        continue

                if len(acquired_gpu_ids) >= gpu_count:
                    break

                # Async sleep keeps the event loop responsive
                await asyncio.sleep(0.1)

            if len(acquired_gpu_ids) < gpu_count:
                # Rollback: deregister from _allocations and release file locks
                for gpu_id, lock_file in acquired_locks:
                    self._deregister_allocation("session", session_id, gpu_id)
                    # Bug #7 fix: use try/finally to ensure close() always runs
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
                    except Exception as e:
                        logger.error(
                            f"Session {session_id}: Failed to unlock GPU {gpu_id} during rollback: {e}"
                        )
                    finally:
                        try:
                            lock_file.close()
                        except Exception:
                            pass
                raise TimeoutError(
                    f"Session {session_id}: Could not acquire {gpu_count} GPUs, "
                    f"only {len(acquired_gpu_ids)} available within {timeout}s"
                )

            wait_time = time.time() - start_time
            logger.info(
                f"Session {session_id}: Acquired {len(acquired_gpu_ids)} GPUs "
                f"{acquired_gpu_ids} after {wait_time:.2f}s"
            )
            return acquired_gpu_ids

        except Exception:
            # Clean up on any error: deregister and release all acquired locks
            for gpu_id, lock_file in acquired_locks:
                self._deregister_allocation("session", session_id, gpu_id)
                # Bug #7 fix: use try/finally to ensure close() always runs
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except Exception as e:
                    logger.error(
                        f"Session {session_id}: Failed to unlock GPU {gpu_id} during cleanup: {e}"
                    )
                finally:
                    try:
                        lock_file.close()
                    except Exception:
                        pass
            raise

    def release_gpus_for_session(self, session_id: str) -> None:
        """
        Release all GPUs held by a session.

        Args:
            session_id: Session identifier
        """
        # Find all session allocations for this session_id
        keys_to_remove = [
            key for key, alloc in self._allocations.items()
            if alloc.allocation_type == "session" and alloc.allocation_id == session_id
        ]

        if not keys_to_remove:
            logger.debug(f"Session {session_id}: No GPUs to release")
            return

        released_gpu_ids = []
        for key in keys_to_remove:
            alloc = self._allocations.pop(key)
            try:
                if alloc.lock_handle:
                    fcntl.flock(alloc.lock_handle, fcntl.LOCK_UN)
                    alloc.lock_handle.close()
                released_gpu_ids.append(alloc.gpu_id)
                logger.info(f"Session {session_id}: Released GPU {alloc.gpu_id}")
            except Exception as e:
                logger.error(f"Session {session_id}: Failed to release GPU {alloc.gpu_id}: {e}")

        logger.info(f"Session {session_id}: Released GPUs {released_gpu_ids}")

