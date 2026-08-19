# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
TDD tests for /api/campaigns (singular, per-client) parallelization fix.

Verifies the same asyncio.gather + Semaphore(16) pattern applied to the
per-client endpoint (same fix already proven in /api/campaigns).
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCampaignsSingularParallelLoop:
    """The /api/campaigns session loop should run concurrently via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_sessions_processed_in_parallel(self):
        """Multiple sessions should be processed concurrently, not sequentially."""
        from app import app
        from httpx import ASGITransport, AsyncClient

        active_count = 0
        max_concurrent = 0

        async def slow_find_artifact_dir(worktree_path):
            nonlocal active_count, max_concurrent
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1
            return f"/fake/artifacts/{worktree_path}"

        async def fake_read_state(artifact_dir):
            return {
                "target": {"model_id": "test/Model", "hardware": "H100"},
                "campaign": {"current_stage": "1_baseline", "current_round": 1},
            }

        fake_sessions = []
        for i in range(6):
            s = MagicMock()
            s.session_id = f"session-{i}"
            s.worktree_path = f"/worktree/{i}"
            s.created_at = "2026-01-01"
            fake_sessions.append(s)

        mock_session_mgr = AsyncMock()
        mock_session_mgr.list_sessions = AsyncMock(
            return_value=MagicMock(sessions=fake_sessions)
        )

        mock_cds = MagicMock()
        mock_cds.find_artifact_dir = AsyncMock(side_effect=slow_find_artifact_dir)
        mock_cds.read_state = AsyncMock(side_effect=fake_read_state)
        mock_cds.build_l1_projection = MagicMock(
            return_value={"session_id": "x", "model": "y"}
        )

        with patch("app.session_manager", mock_session_mgr), \
             patch("app.campaign_data_service", mock_cds):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaigns")
                assert resp.status_code == 200

        assert max_concurrent >= 4, (
            f"Expected concurrent processing (>=4), got max_concurrent={max_concurrent}. "
            f"Loop is likely still sequential."
        )
        data = resp.json()
        assert len(data["campaigns"]) == 6

    @pytest.mark.asyncio
    async def test_exception_in_one_session_does_not_break_others(self):
        """If one session's processing throws, other sessions still return."""
        from app import app
        from httpx import ASGITransport, AsyncClient

        async def find_with_error(worktree_path):
            if "0" in worktree_path:
                raise RuntimeError("disk failure")
            return f"/fake/{worktree_path}"

        async def fake_read_state(artifact_dir):
            return {
                "target": {"model_id": "test/Model", "hardware": "H100"},
                "campaign": {"current_stage": "2_optimize", "current_round": 1},
            }

        fake_sessions = []
        for i in range(3):
            s = MagicMock()
            s.session_id = f"session-{i}"
            s.worktree_path = f"/worktree/{i}"
            s.created_at = "2026-01-01"
            fake_sessions.append(s)

        mock_session_mgr = AsyncMock()
        mock_session_mgr.list_sessions = AsyncMock(
            return_value=MagicMock(sessions=fake_sessions)
        )

        mock_cds = MagicMock()
        mock_cds.find_artifact_dir = AsyncMock(side_effect=find_with_error)
        mock_cds.read_state = AsyncMock(side_effect=fake_read_state)
        mock_cds.build_l1_projection = MagicMock(
            return_value={"session_id": "x", "model": "y"}
        )

        with patch("app.session_manager", mock_session_mgr), \
             patch("app.campaign_data_service", mock_cds):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaigns")
                assert resp.status_code == 200

        data = resp.json()
        assert len(data["campaigns"]) == 2

    @pytest.mark.asyncio
    async def test_sessions_without_worktree_skipped(self):
        """Sessions with no worktree_path should be skipped without error."""
        from app import app
        from httpx import ASGITransport, AsyncClient

        fake_sessions = []
        for i in range(3):
            s = MagicMock()
            s.session_id = f"session-{i}"
            s.worktree_path = None if i == 1 else f"/worktree/{i}"
            s.created_at = "2026-01-01"
            fake_sessions.append(s)

        mock_session_mgr = AsyncMock()
        mock_session_mgr.list_sessions = AsyncMock(
            return_value=MagicMock(sessions=fake_sessions)
        )

        mock_cds = MagicMock()
        mock_cds.find_artifact_dir = AsyncMock(return_value="/fake/dir")
        mock_cds.read_state = AsyncMock(return_value={
            "target": {"model_id": "test/Model", "hardware": "A100"},
            "campaign": {"current_stage": "1_baseline", "current_round": 1},
        })
        mock_cds.build_l1_projection = MagicMock(
            return_value={"session_id": "x"}
        )

        with patch("app.session_manager", mock_session_mgr), \
             patch("app.campaign_data_service", mock_cds):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaigns")
                assert resp.status_code == 200

        assert mock_cds.find_artifact_dir.call_count == 2

    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrency_at_16(self):
        """Concurrency should be bounded by Semaphore(16), not unlimited."""
        import inspect
        from app import get_campaigns_overview

        source = inspect.getsource(get_campaigns_overview)
        assert "Semaphore(16)" in source or "Semaphore( 16)" in source, (
            "Expected asyncio.Semaphore(16) in get_campaigns_overview"
        )
        assert "asyncio.gather" in source, (
            "Expected asyncio.gather in get_campaigns_overview"
        )
