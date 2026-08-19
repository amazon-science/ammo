# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
RED phase tests for async blocking fix in session_state.py.

These tests verify that the blocking operations inside create_download_archive()
(Path.rglob + shutil.rmtree + shutil.make_archive) are wrapped in
asyncio.get_running_loop().run_in_executor() to prevent blocking the asyncio
event loop during archive creation.

The fix pattern: extract rglob+rmtree+make_archive into a sync helper
_sanitize_and_zip(), then call it via run_in_executor().

ALL TESTS IN THIS FILE MUST FAIL BEFORE THE FIX IS APPLIED (RED phase).
"""

import inspect
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_create_download_archive_source() -> str:
    """Return source of SessionS3Storage.create_download_archive."""
    from orchestration.session_state import SessionS3Storage
    return inspect.getsource(SessionS3Storage.create_download_archive)


@pytest.mark.unit
class TestCreateDownloadArchiveBlockingOps:
    """create_download_archive() must use run_in_executor for blocking I/O."""

    def test_create_download_archive_blocking_ops_use_executor(self):
        """
        create_download_archive() performs three blocking filesystem operations:
          1. Path.rglob() — recursive directory scan
          2. shutil.rmtree() — deleting matched sensitive dirs
          3. shutil.make_archive() — creating the ZIP (most expensive, can take seconds)

        All three must be wrapped in run_in_executor() so the event loop is free
        during archive creation.

        Fails RED: Current code calls make_archive() and rglob() synchronously.
        """
        source = _get_create_download_archive_source()

        # Sanity: verify we're looking at the right method
        assert "make_archive" in source, (
            "create_download_archive() must contain shutil.make_archive "
            "(sanity check — method may have been refactored)"
        )

        assert "run_in_executor" in source, (
            "create_download_archive(): blocking operations (rglob, rmtree, "
            "make_archive) must be wrapped in loop.run_in_executor() to avoid "
            "blocking the asyncio event loop during ZIP creation. "
            "Expected fix: extract into a sync helper _sanitize_and_zip() "
            "and call via: "
            "await loop.run_in_executor(None, _sanitize_and_zip, temp_dir)"
        )

    def test_create_download_archive_still_sanitizes_correctly(self):
        """
        After wrapping blocking ops in run_in_executor, the sanitization logic
        that strips .claude/, claude-config/, and CLAUDE.md must still be present
        in the sync helper that runs inside the executor.

        Fails RED: The fix requires BOTH run_in_executor AND retained sanitization
        patterns. Currently neither is present — we verify run_in_executor missing.
        """
        source = _get_create_download_archive_source()

        # Sanity: verify sanitization patterns exist in source
        assert ".claude" in source, (
            "Sanitization pattern '.claude' must remain in the archive creation code. "
            "After refactor, this should live inside the sync helper passed to executor."
        )
        assert "claude-config" in source, (
            "Sanitization pattern 'claude-config' must remain in the archive creation code."
        )
        assert "CLAUDE.md" in source, (
            "Sanitization pattern 'CLAUDE.md' must remain in the archive creation code."
        )

        # RED assertion: run_in_executor must be present (fix not yet applied)
        assert "run_in_executor" in source, (
            "create_download_archive(): blocking ops (rglob, rmtree, make_archive) must "
            "be extracted into a sync helper and called via run_in_executor(). "
            "This test confirms the refactor preserved the sanitization logic inside "
            "the executor-wrapped helper. Current code has no run_in_executor."
        )

    def test_sanitize_handles_files_with_unlink(self):
        """
        Edge case: CLAUDE.md is a FILE, not a directory. The sanitization helper
        must use unlink() (or equivalent) for file-type matches, not just
        shutil.rmtree() which silently fails on files when ignore_errors=True.

        Without unlink(), CLAUDE.md files would silently survive sanitization,
        defeating the security requirement of stripping CLAUDE.md from archives.

        This test PASSES with the correct implementation (which has both
        `if match.is_dir(): rmtree` and `else: unlink`).
        """
        source = _get_create_download_archive_source()

        assert "unlink" in source, (
            "Sanitization inside _sanitize_and_zip must handle file-type matches "
            "(e.g. CLAUDE.md at the top level of the archive) via match.unlink(). "
            "Using only shutil.rmtree(match, ignore_errors=True) silently fails on "
            "files, leaving CLAUDE.md present in the downloaded archive."
        )
