# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
RED phase tests for async blocking fix in session_manager.py.

These tests verify that all shutil.rmtree() calls inside async methods are
wrapped in asyncio.get_running_loop().run_in_executor() to prevent blocking
the asyncio event loop (rmtree on large session directories can take 100ms-2s).

Affected methods:
  - discover_s3_sessions()  ~line 229
  - resume_session()        ~line 864
  - terminate_session()     ~line 1423
  - cleanup_old_sessions()  ~line 1593

ALL TESTS IN THIS FILE MUST FAIL BEFORE THE FIX IS APPLIED (RED phase).
"""

import inspect
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_method_source(method_name: str) -> str:
    """Return source of the given SessionManager method."""
    from orchestration.session_manager import SessionManager
    method = getattr(SessionManager, method_name)
    return inspect.getsource(method)


def _has_executor_wrapped_rmtree(source: str) -> bool:
    """
    Return True if a single line contains BOTH run_in_executor and rmtree,
    indicating that rmtree is wrapped inline via a lambda:

      await loop.run_in_executor(None, lambda: shutil.rmtree(...))

    Checking per-line prevents false positives from methods that already use
    run_in_executor for OTHER purposes (e.g. create_worktree, setup_workspace)
    while still calling rmtree directly on a separate line.
    """
    for line in source.split("\n"):
        if "run_in_executor" in line and "rmtree" in line:
            return True
    return False


@pytest.mark.unit
class TestDiscoverS3SessionsRmtreeAsync:
    """discover_s3_sessions() must use run_in_executor for shutil.rmtree."""

    def test_discover_s3_sessions_rmtree_uses_executor(self):
        """
        discover_s3_sessions() removes stale local worktrees via shutil.rmtree.
        This call must be wrapped in run_in_executor() to avoid blocking the
        event loop during S3-based session recovery on startup.

        Fails RED: Current code calls shutil.rmtree(...) synchronously.
        """
        source = _get_method_source("discover_s3_sessions")

        assert "rmtree" in source, (
            "discover_s3_sessions() must contain a shutil.rmtree call "
            "(sanity check — method may have been refactored)"
        )
        assert _has_executor_wrapped_rmtree(source), (
            "discover_s3_sessions(): shutil.rmtree must be wrapped in "
            "loop.run_in_executor() to avoid blocking the asyncio event loop. "
            "Large session worktrees can take 100ms-2s to remove. "
            "Expected pattern: "
            "await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))"
        )


@pytest.mark.unit
class TestResumeSessionRmtreeAsync:
    """resume_session() must use run_in_executor for shutil.rmtree."""

    def test_resume_session_rmtree_uses_executor(self):
        """
        resume_session() removes a stale worktree before S3 restore.
        This shutil.rmtree must be wrapped in run_in_executor() to avoid
        blocking the event loop during cross-pod session resume.

        Fails RED: Current code calls shutil.rmtree(...) synchronously.
        """
        source = _get_method_source("resume_session")

        assert "rmtree" in source, (
            "resume_session() must contain a shutil.rmtree call "
            "(sanity check — method may have been refactored)"
        )
        assert _has_executor_wrapped_rmtree(source), (
            "resume_session(): shutil.rmtree must be wrapped in "
            "loop.run_in_executor() to avoid blocking the asyncio event loop. "
            "Expected pattern: "
            "await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))"
        )


@pytest.mark.unit
class TestTerminateSessionRmtreeAsync:
    """terminate_session() must use run_in_executor for GPU res dir cleanup."""

    def test_terminate_session_rmtree_uses_executor(self):
        """
        terminate_session() removes the per-session GPU reservation state dir
        at /tmp/ammo_gpu_res_{session_id} via shutil.rmtree.
        This must be wrapped in run_in_executor() to avoid blocking the loop.

        Fails RED: Current code calls _shutil.rmtree(...) synchronously.
        """
        source = _get_method_source("terminate_session")

        assert "rmtree" in source, (
            "terminate_session() must contain a shutil.rmtree call "
            "(sanity check — method may have been refactored)"
        )
        assert _has_executor_wrapped_rmtree(source), (
            "terminate_session(): shutil.rmtree (GPU res dir cleanup) must be "
            "wrapped in loop.run_in_executor() to avoid blocking the asyncio "
            "event loop. "
            "Expected pattern: "
            "await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))"
        )


@pytest.mark.unit
class TestHelperFalsePositivePrevention:
    """Verify _has_executor_wrapped_rmtree() doesn't produce false positives."""

    def test_separate_lines_not_detected_as_executor_wrapped(self):
        """
        When run_in_executor and rmtree appear on DIFFERENT lines, the helper
        must return False. This guards against false positives from methods that
        already use run_in_executor for other purposes (e.g. create_worktree)
        while still calling rmtree directly on a separate line.

        This test should PASS immediately — it verifies the helper's key design
        invariant (per-line matching) is correctly implemented.
        """
        source_with_separate_lines = (
            "await loop.run_in_executor(None, create_worktree, args)\n"
            "shutil.rmtree(stale_path, ignore_errors=True)\n"
        )
        assert not _has_executor_wrapped_rmtree(source_with_separate_lines), (
            "_has_executor_wrapped_rmtree() must return False when run_in_executor "
            "and rmtree are on DIFFERENT lines. This prevents false positives from "
            "methods that use run_in_executor for other purposes while still calling "
            "rmtree directly on a separate line (e.g. resume_session already has "
            "run_in_executor for create_worktree/setup_workspace before this fix)."
        )

    def test_same_line_detected_as_executor_wrapped(self):
        """
        When run_in_executor and rmtree appear on the SAME line, the helper
        must return True — this is the expected pattern after the fix.

        Also PASSES immediately — validates the positive detection case.
        """
        source_with_same_line = (
            "await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))\n"
        )
        assert _has_executor_wrapped_rmtree(source_with_same_line), (
            "_has_executor_wrapped_rmtree() must return True when run_in_executor "
            "and rmtree appear on the SAME line."
        )


@pytest.mark.unit
class TestCleanupOldSessionsRmtreeAsync:
    """cleanup_old_sessions() must use run_in_executor for shutil.rmtree."""

    def test_cleanup_old_sessions_rmtree_uses_executor(self):
        """
        cleanup_old_sessions() removes terminated session directories from disk
        via shutil.rmtree. This must be wrapped in run_in_executor() since it
        runs as a periodic background task and must not block the event loop.

        Fails RED: Current code calls shutil.rmtree(session_dir) synchronously.
        """
        source = _get_method_source("cleanup_old_sessions")

        assert "rmtree" in source, (
            "cleanup_old_sessions() must contain a shutil.rmtree call "
            "(sanity check — method may have been refactored)"
        )
        assert _has_executor_wrapped_rmtree(source), (
            "cleanup_old_sessions(): shutil.rmtree must be wrapped in "
            "loop.run_in_executor() to avoid blocking the asyncio event loop. "
            "Expected pattern: "
            "await loop.run_in_executor(None, lambda: shutil.rmtree(session_dir))"
        )
