# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Checkpoint Manager for AI CLI Sessions.

Handles automatic checkpointing of session state to S3:
- Triggered on WebSocket disconnect (after grace period)
- Does NOT pause the session - just syncs worktree to S3
- Enables cross-node resume if node fails
"""

import asyncio
import logging
import time
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.session_manager import SessionManager

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages session checkpoints (S3 sync without pause).

    A checkpoint:
    - Syncs worktree to S3
    - Updates s3_synced flag
    - Does NOT change session status
    - Does NOT release ownership
    - Enables recovery if node fails
    """

    def __init__(self, session_manager: Optional["SessionManager"] = None):
        """
        Initialize checkpoint manager.

        Args:
            session_manager: SessionManager instance for checkpoint operations
        """
        self._session_manager = session_manager

        # Track checkpoint state
        self._last_checkpoint: Dict[str, float] = {}  # session_id -> timestamp
        self._checkpoint_in_progress: Dict[str, bool] = {}

        logger.info("CheckpointManager initialized")

    def set_session_manager(self, session_manager: "SessionManager") -> None:
        """Set the session manager (for deferred initialization)."""
        self._session_manager = session_manager

    async def checkpoint_session(self, session_id: str) -> bool:
        """
        Checkpoint a session to S3 without pausing it.

        This syncs the worktree and CLI state to S3 but keeps the session
        active. The session can continue running; this just ensures we have
        a recent snapshot for recovery.

        Args:
            session_id: Session identifier

        Returns:
            True if checkpoint succeeded
        """
        if not self._session_manager:
            logger.error(f"Session {session_id}: No session manager configured")
            return False

        # Prevent concurrent checkpoints for same session
        if self._checkpoint_in_progress.get(session_id):
            logger.debug(f"Session {session_id}: Checkpoint already in progress")
            return False

        self._checkpoint_in_progress[session_id] = True

        try:
            # Get session state
            state = self._session_manager._sessions.get(session_id)
            if not state:
                logger.warning(f"Session {session_id}: Not found, skipping checkpoint")
                return False

            # Check if session is active (only checkpoint active sessions)
            from shared.session_models import SessionStatus
            if state.status != SessionStatus.ACTIVE:
                logger.debug(
                    f"Session {session_id}: Status is {state.status.value}, skipping checkpoint"
                )
                return False

            # Check if S3 storage is available
            if not self._session_manager.session_storage:
                logger.debug(f"Session {session_id}: No S3 storage configured")
                return False

            if not self._session_manager.session_storage.enabled:
                logger.debug(f"Session {session_id}: S3 storage not enabled")
                return False

            logger.info(f"Session {session_id}: Starting checkpoint...")
            start_time = time.time()

            # Conflict check: skip upload if S3 already has a newer version
            s3_last_modified = await self._session_manager.session_storage.get_s3_last_modified(
                session_id
            )
            if s3_last_modified and state.s3_last_sync and s3_last_modified > state.s3_last_sync:
                logger.warning(
                    f"Session {session_id}: S3 has newer data "
                    f"(s3={s3_last_modified:.0f} > local={state.s3_last_sync:.0f}), "
                    "skipping checkpoint to avoid overwriting newer state"
                )
                return False

            # Sync to S3 (worktree + CLI state) — shielded from outer cancellation.
            # Without shield(), a WebSocket-reconnect cancel can fire between the S3
            # upload subprocess finishing and the bookkeeping below, leaving S3 with
            # a fresh tar but state.s3_last_sync pointing at the prior upload. The
            # next checkpoint/pause then trips the staleness guard and silently
            # drops all subsequent work until an S3 restore destructively
            # restores the orphaned tar.
            async def _upload_and_record() -> bool:
                ok = await self._session_manager.session_storage.sync_session_to_s3(state)
                if not ok:
                    return False
                state.s3_synced = True
                s3_ts = await self._session_manager.session_storage.get_s3_last_modified(session_id)
                state.s3_last_sync = s3_ts if s3_ts else time.time()
                self._session_manager._save_session_state(state)
                return True

            upload_ok = await asyncio.shield(_upload_and_record())
            if not upload_ok:
                logger.warning(f"Session {session_id}: S3 upload returned False, not marking synced")
                return False

            # Track checkpoint time
            self._last_checkpoint[session_id] = time.time()

            elapsed = time.time() - start_time
            logger.info(
                f"Session {session_id}: Checkpoint completed in {elapsed:.1f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Session {session_id}: Checkpoint failed: {e}")
            return False

        finally:
            self._checkpoint_in_progress[session_id] = False

    def get_last_checkpoint(self, session_id: str) -> Optional[float]:
        """Get timestamp of last checkpoint for a session."""
        return self._last_checkpoint.get(session_id)

    def is_checkpoint_in_progress(self, session_id: str) -> bool:
        """Check if a checkpoint is currently in progress."""
        return self._checkpoint_in_progress.get(session_id, False)

    def cleanup_session(self, session_id: str) -> None:
        """Clean up checkpoint tracking for a session."""
        self._last_checkpoint.pop(session_id, None)
        self._checkpoint_in_progress.pop(session_id, None)


# Singleton instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get singleton checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager


def reset_checkpoint_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _checkpoint_manager
    _checkpoint_manager = None
