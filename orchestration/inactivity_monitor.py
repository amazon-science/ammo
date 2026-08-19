# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Inactivity Monitor for AI CLI sessions.

Monitors session activity and handles auto-shutdown:
- Tracks last activity timestamp per session
- Background task checking for timeout
- Triggers auto-pause for inactive sessions
- Kills spawned background processes
- Syncs state to S3 before shutdown
"""

import asyncio
import logging
import os
import signal
import time
from typing import Optional, Dict, Set, Callable, Awaitable
from dataclasses import dataclass, field

from shared.session_models import (
    SessionState,
    SessionStatus,
    DEFAULT_INACTIVITY_TIMEOUT_MINS,
    ENV_SESSION_INACTIVITY_TIMEOUT_MINS,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionActivity:
    """Tracks activity for a session."""
    session_id: str
    last_activity: float  # Unix timestamp
    timeout_mins: int
    child_pids: Set[int] = field(default_factory=set)


class InactivityMonitor:
    """
    Monitors session inactivity and handles auto-shutdown.

    Tracks activity per session and triggers auto-pause when
    inactivity threshold is exceeded.
    """

    def __init__(
        self,
        default_timeout_mins: Optional[int] = None,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize inactivity monitor.

        Args:
            default_timeout_mins: Default inactivity timeout in minutes
            check_interval_seconds: How often to check for inactive sessions
        """
        self.default_timeout_mins = default_timeout_mins or int(
            os.getenv(ENV_SESSION_INACTIVITY_TIMEOUT_MINS, DEFAULT_INACTIVITY_TIMEOUT_MINS)
        )
        self.check_interval = check_interval_seconds

        # Track session activity
        self._sessions: Dict[str, SessionActivity] = {}

        # Callback for pausing sessions
        self._pause_callback: Optional[Callable[[str], Awaitable[None]]] = None

        # Monitor task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"InactivityMonitor initialized: default_timeout={self.default_timeout_mins}m, "
            f"check_interval={self.check_interval}s"
        )

    def set_pause_callback(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        Set callback function for pausing sessions.

        The callback should handle:
        - Stopping CLI and terminal processes
        - Releasing GPUs
        - Syncing state to S3
        - Updating session status

        Args:
            callback: Async function that takes session_id
        """
        self._pause_callback = callback

    def register_session(
        self,
        session_id: str,
        timeout_mins: Optional[int] = None,
    ) -> None:
        """
        Register a session for monitoring.

        Args:
            session_id: Session identifier
            timeout_mins: Custom timeout for this session
        """
        self._sessions[session_id] = SessionActivity(
            session_id=session_id,
            last_activity=time.time(),
            timeout_mins=timeout_mins or self.default_timeout_mins,
        )
        logger.debug(f"Registered session for monitoring: {session_id}")

    def unregister_session(self, session_id: str) -> None:
        """
        Unregister a session from monitoring.

        Args:
            session_id: Session identifier
        """
        activity = self._sessions.pop(session_id, None)
        if activity:
            # Kill any tracked child processes
            self._kill_child_processes(activity)
        logger.debug(f"Unregistered session from monitoring: {session_id}")

    def record_activity(self, session_id: str) -> None:
        """
        Record activity for a session.

        Should be called on any terminal I/O or user interaction.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            self._sessions[session_id].last_activity = time.time()

    def register_child_process(self, session_id: str, pid: int) -> None:
        """
        Register a child process spawned by a session.

        These processes will be killed when the session is auto-paused.

        Args:
            session_id: Session identifier
            pid: Process ID
        """
        if session_id in self._sessions:
            self._sessions[session_id].child_pids.add(pid)
            logger.debug(f"Registered child process {pid} for session {session_id}")

    def unregister_child_process(self, session_id: str, pid: int) -> None:
        """
        Unregister a child process.

        Args:
            session_id: Session identifier
            pid: Process ID
        """
        if session_id in self._sessions:
            self._sessions[session_id].child_pids.discard(pid)

    def _kill_child_processes(self, activity: SessionActivity) -> None:
        """Kill all tracked child processes for a session."""
        for pid in list(activity.child_pids):
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to child process {pid} of session {activity.session_id}")
            except ProcessLookupError:
                logger.debug(f"Child process {pid} already terminated")
            except Exception as e:
                logger.warning(f"Failed to kill child process {pid}: {e}")
        activity.child_pids.clear()

    def get_time_until_timeout(self, session_id: str) -> Optional[float]:
        """
        Get seconds until session times out.

        Args:
            session_id: Session identifier

        Returns:
            Seconds until timeout, or None if not tracked
        """
        activity = self._sessions.get(session_id)
        if not activity:
            return None

        timeout_seconds = activity.timeout_mins * 60
        elapsed = time.time() - activity.last_activity
        remaining = timeout_seconds - elapsed
        return max(0, remaining)

    def is_session_timed_out(self, session_id: str) -> bool:
        """Check if a session has timed out."""
        remaining = self.get_time_until_timeout(session_id)
        return remaining is not None and remaining <= 0

    async def _check_inactive_sessions(self) -> None:
        """Check for and handle inactive sessions."""
        now = time.time()
        sessions_to_pause = []

        for session_id, activity in self._sessions.items():
            timeout_seconds = activity.timeout_mins * 60
            elapsed = now - activity.last_activity

            if elapsed >= timeout_seconds:
                logger.info(
                    f"Session {session_id} timed out after {elapsed/60:.1f}m "
                    f"(threshold: {activity.timeout_mins}m)"
                )
                sessions_to_pause.append(session_id)

        # Pause timed-out sessions
        for session_id in sessions_to_pause:
            await self._handle_timeout(session_id)

    async def _handle_timeout(self, session_id: str) -> None:
        """Handle session timeout."""
        activity = self._sessions.get(session_id)
        if not activity:
            return

        # Kill child processes first
        self._kill_child_processes(activity)

        # Call pause callback if registered
        if self._pause_callback:
            try:
                await self._pause_callback(session_id)
                logger.info(f"Auto-paused inactive session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to auto-pause session {session_id}: {e}")
        else:
            logger.warning(f"No pause callback registered for session {session_id}")

        # Unregister from monitoring
        self.unregister_session(session_id)

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Inactivity monitor started")

        while self._running:
            try:
                await self._check_inactive_sessions()
            except Exception as e:
                logger.error(f"Error in inactivity monitor: {e}")

            await asyncio.sleep(self.check_interval)

        logger.info("Inactivity monitor stopped")

    def start(self) -> None:
        """Start the monitoring background task."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Inactivity monitor task started")

    async def stop(self) -> None:
        """Stop the monitoring background task."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Inactivity monitor task stopped")

    def get_session_activity(self, session_id: str) -> Optional[Dict]:
        """Get activity information for a session."""
        activity = self._sessions.get(session_id)
        if not activity:
            return None

        return {
            "session_id": activity.session_id,
            "last_activity": activity.last_activity,
            "timeout_mins": activity.timeout_mins,
            "time_until_timeout_seconds": self.get_time_until_timeout(session_id),
            "child_pids": list(activity.child_pids),
        }

    def get_all_activity(self) -> Dict[str, Dict]:
        """Get activity information for all sessions."""
        return {
            session_id: self.get_session_activity(session_id)
            for session_id in self._sessions
        }


# Singleton instance
_inactivity_monitor: Optional[InactivityMonitor] = None


def get_inactivity_monitor(
    default_timeout_mins: Optional[int] = None,
    check_interval_seconds: int = 60,
) -> InactivityMonitor:
    """
    Get singleton inactivity monitor instance.

    Args:
        default_timeout_mins: Default timeout (only used on first call)
        check_interval_seconds: Check interval (only used on first call)

    Returns:
        InactivityMonitor instance
    """
    global _inactivity_monitor
    if _inactivity_monitor is None:
        _inactivity_monitor = InactivityMonitor(
            default_timeout_mins=default_timeout_mins,
            check_interval_seconds=check_interval_seconds,
        )
    return _inactivity_monitor


async def reset_inactivity_monitor() -> None:
    """Reset singleton instance (for testing)."""
    global _inactivity_monitor
    if _inactivity_monitor:
        await _inactivity_monitor.stop()
    _inactivity_monitor = None
