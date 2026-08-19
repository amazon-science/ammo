# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for the report viewer feature."""

import sys
import base64
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import SessionState, SessionStatus, CLIToolType, SessionInfo


# ============================================================================
# Helpers
# ============================================================================

def _make_state(status=SessionStatus.ACTIVE, worktree_path=None):
    """Create a minimal SessionState for testing has_report."""
    return SessionState(
        session_id="test-123",
        status=status,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=1000.0,
        last_accessed=2000.0,
        worktree_path=str(worktree_path) if worktree_path else None,
    )


def _mock_session_info(status="active", worktree_path="/fake/path", has_report=True):
    """Create a mock SessionInfo-like object for endpoint tests."""
    return MagicMock(
        session_id="test-123",
        status=status,
        worktree_path=worktree_path,
        has_report=has_report,
    )


@pytest.fixture
def mock_session_mgr():
    """Patch the global session_manager in app.py with an AsyncMock."""
    mgr = AsyncMock()
    with patch("app.session_manager", mgr):
        yield mgr


# ============================================================================
# has_report tests (SessionState.to_session_info)
# ============================================================================

@pytest.mark.unit
class TestHasReport:
    """Tests for SessionState.to_session_info() has_report field."""

    def test_has_report_true_when_report_exists(self, tmp_path):
        """has_report should be True when REPORT.md exists in kernel_opt_artifacts/."""
        subdir = tmp_path / "kernel_opt_artifacts" / "run1"
        subdir.mkdir(parents=True)
        (subdir / "REPORT.md").write_text("# Report")

        state = _make_state(status=SessionStatus.ACTIVE, worktree_path=tmp_path)
        info = state.to_session_info()
        assert info.has_report is True

    def test_has_report_false_when_no_report(self, tmp_path):
        """has_report should be False when no REPORT.md exists."""
        state = _make_state(status=SessionStatus.ACTIVE, worktree_path=tmp_path)
        info = state.to_session_info()
        assert info.has_report is False

    def test_has_report_false_when_terminated(self, tmp_path):
        """has_report should be False when session is terminated, even if REPORT.md exists."""
        (tmp_path / "REPORT.md").write_text("# Report")

        state = _make_state(status=SessionStatus.TERMINATED, worktree_path=tmp_path)
        info = state.to_session_info()
        assert info.has_report is False

    def test_has_report_false_when_no_worktree(self):
        """has_report should be False when worktree_path is None."""
        state = _make_state(status=SessionStatus.ACTIVE, worktree_path=None)
        info = state.to_session_info()
        assert info.has_report is False

    def test_has_report_true_when_paused(self, tmp_path):
        """has_report should be True for paused sessions with REPORT.md."""
        (tmp_path / "REPORT.md").write_text("# Report")

        state = _make_state(status=SessionStatus.PAUSED, worktree_path=tmp_path)
        info = state.to_session_info()
        assert info.has_report is True


# ============================================================================
# Report endpoint tests (GET /sessions/{session_id}/report)
# ============================================================================

@pytest.mark.unit
class TestReportEndpoint:
    """Tests for the GET /sessions/{session_id}/report endpoint."""

    @pytest.mark.asyncio
    async def test_report_endpoint_returns_markdown(self, mock_session_mgr, tmp_path):
        """Endpoint should return session_id, markdown, and report_path."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        # Create REPORT.md in a subdirectory
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "REPORT.md").write_text("# Optimization Report\n\nResults here.")

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="active",
            worktree_path=str(tmp_path),
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-123"
        assert "# Optimization Report" in data["markdown"]
        assert "report_path" in data
        assert data["report_path"] == "artifacts/REPORT.md"

    @pytest.mark.asyncio
    async def test_report_endpoint_base64_images(self, mock_session_mgr, tmp_path):
        """Image references in REPORT.md should be replaced with base64 data URIs."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        report_assets = artifacts / "report_assets"
        report_assets.mkdir()

        # Write a REPORT.md that references an image
        (artifacts / "REPORT.md").write_text(
            "# Report\n\n![Chart](report_assets/chart.png)\n"
        )

        # Write fake PNG bytes (doesn't need to be valid PNG for test)
        fake_png = b"\x89PNG\r\n\x1a\nfake_image_data"
        (report_assets / "chart.png").write_bytes(fake_png)

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="active",
            worktree_path=str(tmp_path),
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 200
        data = resp.json()
        # The image reference should be replaced with a base64 data URI
        assert "data:image/png;base64," in data["markdown"]
        # Verify the base64 content is correct
        expected_b64 = base64.b64encode(fake_png).decode("ascii")
        assert expected_b64 in data["markdown"]
        # The original path reference should NOT be present
        assert "report_assets/chart.png" not in data["markdown"]

    @pytest.mark.asyncio
    async def test_report_endpoint_404_no_report(self, mock_session_mgr, tmp_path):
        """Endpoint should return 404 when no REPORT.md exists in worktree."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="active",
            worktree_path=str(tmp_path),
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_report_endpoint_404_terminated(self, mock_session_mgr):
        """Endpoint should return 404 for terminated sessions."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="terminated",
            worktree_path="/some/path",
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_report_endpoint_path_traversal_blocked(self, mock_session_mgr, tmp_path):
        """Image references that escape the worktree should be left unchanged."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Write a REPORT.md with a path traversal attempt
        traversal_ref = "![Hack](../../etc/passwd)"
        (artifacts / "REPORT.md").write_text(f"# Report\n\n{traversal_ref}\n")

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="active",
            worktree_path=str(tmp_path),
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 200
        data = resp.json()
        # The traversal reference should be left unchanged (not replaced with base64)
        assert "../../etc/passwd" in data["markdown"]
        assert "data:" not in data["markdown"]

    @pytest.mark.asyncio
    async def test_report_endpoint_missing_image_left_as_is(self, mock_session_mgr, tmp_path):
        """References to non-existent images should be left unchanged."""
        from httpx import AsyncClient, ASGITransport
        import app as app_module

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Write REPORT.md referencing a non-existent image
        missing_ref = "![Missing](report_assets/nonexistent.png)"
        (artifacts / "REPORT.md").write_text(f"# Report\n\n{missing_ref}\n")

        mock_session_mgr.get_session.return_value = _mock_session_info(
            status="active",
            worktree_path=str(tmp_path),
        )

        transport = ASGITransport(app=app_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sessions/test-123/report",
                headers={"X-Client-ID": "test-owner"},
            )

        assert resp.status_code == 200
        data = resp.json()
        # The missing image reference should remain unchanged
        assert "report_assets/nonexistent.png" in data["markdown"]
        assert "data:" not in data["markdown"]
