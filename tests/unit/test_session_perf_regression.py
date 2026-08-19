# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Performance regression tests for local session endpoints.

These tests encode local performance contracts to prevent regressions like:
- Recursive filesystem traversals in hot paths (e.g., glob("**/REPORT.md"))
- Unbounded response time growth with session count
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import SessionState, SessionStatus, CLIToolType


def _make_session(
    session_id: str,
    worktree_path: str = None,
    status=SessionStatus.ACTIVE,
) -> SessionState:
    """Create a minimal SessionState for testing."""
    return SessionState(
        session_id=session_id,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        status=status,
        worktree_path=worktree_path,
        gpu_ids=[],
        created_at=time.time(),
        last_accessed=time.time(),
    )


@pytest.mark.unit
class TestSessionInfoPerformance:
    """Tests that to_session_info() doesn't do expensive filesystem operations."""

    def test_to_session_info_no_recursive_glob(self, tmp_path):
        """to_session_info() must NOT recursively traverse the worktree."""
        deep_dir = tmp_path / "node_modules" / "deep" / "nested" / "path"
        deep_dir.mkdir(parents=True)
        (deep_dir / "REPORT.md").write_text("report")

        state = _make_session("test-1", worktree_path=str(tmp_path))

        info = state.to_session_info()

        assert info.has_report is False

    def test_to_session_info_finds_root_report(self, tmp_path):
        """to_session_info() should find REPORT.md at the worktree root."""
        (tmp_path / "REPORT.md").write_text("# Optimization Report")

        state = _make_session("test-2", worktree_path=str(tmp_path))

        info = state.to_session_info()
        assert info.has_report is True

    def test_to_session_info_finds_kernel_opt_artifacts_report(self, tmp_path):
        """to_session_info() should find REPORT.md under kernel_opt_artifacts/<model>/."""
        artifacts_dir = tmp_path / "kernel_opt_artifacts" / "qwen3.5-4b_l40s_bf16_tp1"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "REPORT.md").write_text("# Optimization Report")

        state = _make_session("test-2b", worktree_path=str(tmp_path))

        info = state.to_session_info()
        assert info.has_report is True

    def test_to_session_info_no_worktree(self):
        """to_session_info() handles missing worktree path gracefully."""
        state = _make_session("test-3", worktree_path=None)

        info = state.to_session_info()
        assert info.has_report is False

    def test_to_session_info_nonexistent_worktree(self):
        """to_session_info() handles nonexistent worktree path gracefully."""
        state = _make_session("test-4", worktree_path="/nonexistent/path")

        info = state.to_session_info()
        assert info.has_report is False

    def test_to_session_info_paused_session_checks_report(self, tmp_path):
        """Paused sessions should still check for REPORT.md."""
        (tmp_path / "REPORT.md").write_text("# Report")
        state = _make_session(
            "test-5",
            worktree_path=str(tmp_path),
            status=SessionStatus.PAUSED,
        )

        info = state.to_session_info()
        assert info.has_report is True

    def test_to_session_info_terminated_skips_report(self, tmp_path):
        """Terminated sessions should NOT check filesystem at all."""
        (tmp_path / "REPORT.md").write_text("# Report")
        state = _make_session(
            "test-6",
            worktree_path=str(tmp_path),
            status=SessionStatus.TERMINATED,
        )

        info = state.to_session_info()
        assert info.has_report is False

    def test_to_session_info_bounded_time(self, tmp_path):
        """to_session_info() must complete within 10ms even with a large worktree."""
        for i in range(50):
            (tmp_path / f"dir_{i}").mkdir()
            for j in range(20):
                (tmp_path / f"dir_{i}" / f"file_{j}.py").write_text("x")

        state = _make_session("test-7", worktree_path=str(tmp_path))

        start = time.monotonic()
        state.to_session_info()
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 10, (
            f"to_session_info() took {elapsed_ms:.1f}ms, budget is 10ms"
        )


@pytest.mark.unit
class TestListSessionsPerformance:
    """Tests that list_sessions() scales well with session count."""

    @pytest.mark.asyncio
    async def test_list_sessions_scales_linearly(self, tmp_path):
        """list_sessions() time should scale linearly, not quadratically."""
        from orchestration.session_manager import SessionManager

        mgr = SessionManager.__new__(SessionManager)
        mgr._sessions = {}
        mgr._session_lock = asyncio.Lock()

        for i in range(100):
            sess_dir = tmp_path / f"session_{i}" / "worktree"
            sess_dir.mkdir(parents=True)
            state = _make_session(f"sess-{i:04d}", worktree_path=str(sess_dir))
            mgr._sessions[state.session_id] = state

        start = time.monotonic()
        result = await mgr.list_sessions()
        elapsed_ms = (time.monotonic() - start) * 1000

        assert result.total == 100
        assert elapsed_ms < 100, (
            f"list_sessions(100) took {elapsed_ms:.1f}ms, budget is 100ms"
        )
