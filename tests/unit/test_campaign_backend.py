# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Campaign Backend — models, data service, and API endpoints.
TDD: written before implementation.
"""

import json
import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_STATE = {
    "target": {
        "model_id": "deepseek-ai/DeepSeek-R1",
        "hardware": "H100",
        "dtype": "fp8",
        "tp": 8,
    },
    "stage": "4_5_parallel_tracks",
    "summary": "Round 1 in progress",
    "gpu_resources": {},
    "debate": {
        "team_name": "debate_team",
        "candidates": ["flash_attn_v1", "flash_attn_v2"],
        "rounds_completed": 2,
        "selected_winners": ["flash_attn_v2"],
        "selection_rationale": "Faster on H100",
        "next_round_overlap": None,
    },
    "parallel_tracks": {
        "flash_attn_v2": {
            "status": "COMPLETED",
            "verdict": "SHIPPED",
            "classification": "attention",
            "correctness": True,
            "kernel_speedup": 1.35,
            "e2e_speedup": 1.12,
            "fail_reason": None,
        },
        "gemm_opt": {
            "status": "FAILED",
            "verdict": "FAILED",
            "fail_reason": "correctness check failed",
        },
        "rope_fused": {
            "status": "IN_PROGRESS",
            "verdict": None,
        },
    },
    "integration": {
        "status": "pending",
        "passing_candidates": [],
    },
    "stage_timestamps": {
        "1_baseline": {"started_at": "2026-01-01T00:00:00", "completed_at": "2026-01-01T01:00:00"},
    },
    "session_id": "sess-abc123",
    "campaign": {
        "status": "active",
        "current_round": 1,
        "cumulative_e2e_speedup": 1.12,
        "min_e2e_improvement_pct": 1.0,
        "e2e_deflation_factor": 2.0,
        "noise_tolerance_pct": 0.5,
        "catastrophic_regression_pct": 5.0,
        "rounds": [],
        "shipped_optimizations": ["flash_attn_v2"],
    },
}


# ---------------------------------------------------------------------------
# 2. CampaignDataService tests
# ---------------------------------------------------------------------------

class TestFindArtifactDir:
    """find_artifact_dir uses glob kernel_opt_artifacts/*/state.json."""

    @pytest.mark.asyncio
    async def test_finds_nested_state_json(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        artifact_dir = tmp_path / "kernel_opt_artifacts" / "deepseek_h100_fp8_tp8"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "state.json").write_text("{}")

        svc = CampaignDataService()
        result = await svc.find_artifact_dir(str(tmp_path))
        assert result == str(artifact_dir)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_state_json(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        result = await svc.find_artifact_dir(str(tmp_path))
        assert result is None

    @pytest.mark.asyncio
    async def test_falls_back_to_worktree_root(self, tmp_path):
        """Falls back to state.json at worktree root when no kernel_opt_artifacts."""
        from orchestration.campaign_data_service import CampaignDataService
        (tmp_path / "state.json").write_text("{}")
        svc = CampaignDataService()
        result = await svc.find_artifact_dir(str(tmp_path))
        assert result == str(tmp_path)

    @pytest.mark.asyncio
    async def test_caches_valid_result(self, tmp_path):
        """Second call with same path returns cached result without re-globbing."""
        from orchestration.campaign_data_service import CampaignDataService
        artifact_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "state.json").write_text("{}")

        svc = CampaignDataService()
        r1 = await svc.find_artifact_dir(str(tmp_path))
        r2 = await svc.find_artifact_dir(str(tmp_path))
        assert r1 == r2
        assert str(tmp_path) in svc._artifact_dir_cache
        assert svc._artifact_dir_cache[str(tmp_path)] == r1


class TestBuildL1Projection:
    """build_l1_projection() counts from ROOT-LEVEL parallel_tracks (V2)."""

    def test_counts_shipped_failed_active(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        out = svc.build_l1_projection("sess-abc123", SAMPLE_STATE)
        camp = out["campaign"]
        # flash_attn_v2 is in shipped_optimizations → shipped
        # gemm_opt is FAILED → failed
        # rope_fused is IN_PROGRESS → active
        assert camp["shipped_count"] == 1
        assert camp["failed_count"] == 1
        assert camp["active_count"] == 1

    def test_overview_fields_populated(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        out = svc.build_l1_projection("sess-abc123", SAMPLE_STATE)
        assert out["session_id"] == "sess-abc123"
        assert out["target"]["model_id"] == "deepseek-ai/DeepSeek-R1"
        assert out["target"]["hardware"] == "H100"
        assert out["target"]["dtype"] == "fp8"
        assert out["target"]["tp"] == 8
        camp = out["campaign"]
        assert camp["status"] == "active"
        assert camp["current_round"] == 1
        assert camp["cumulative_e2e_speedup"] == pytest.approx(1.12)

    def test_current_stage_populated(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        out = svc.build_l1_projection("sess-abc123", SAMPLE_STATE)
        # SAMPLE_STATE.campaign has no current_stage → falls back to state.stage
        assert out["campaign"]["current_stage"] == "4_5_parallel_tracks"

    def test_diluted_count_zero_when_no_diluted_tracks(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        out = svc.build_l1_projection("sess-abc123", SAMPLE_STATE)
        # SAMPLE_STATE has no diluted:true tracks in any round → 0
        assert out["campaign"]["diluted_count"] == 0

    def test_diluted_count_counts_across_rounds(self):
        """diluted_count sums tracks with diluted:True across ALL campaign rounds."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "target": {"model_id": "m", "hardware": "H100", "dtype": "fp8", "tp": 1},
            "campaign": {
                "status": "active",
                "current_round": 2,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "parallel_tracks": {"tracks": {
                            "op_a": {"status": "PASS", "diluted": True},
                            "op_b": {"status": "PASS"},              # no diluted key
                        }},
                    },
                    {
                        "round_id": 2,
                        "parallel_tracks": {"tracks": {
                            "op_c": {"status": "PASS", "diluted": True},
                            "op_d": {"status": "FAIL", "diluted": False},
                        }},
                    },
                ],
            },
        }
        out = svc.build_l1_projection("sess-diluted", state)
        assert out["campaign"]["diluted_count"] == 2

    def test_pipeline_progress_populated(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        out = svc.build_l1_projection("sess-abc123", SAMPLE_STATE)
        progress = out["campaign"]["pipeline_progress"]
        assert len(progress) == 6
        statuses = {p["stage"]: p["status"] for p in progress}
        # stage 4_5_parallel_tracks → index 3. baseline(0), mining(1), debate(2) < 3 → completed
        assert statuses["baseline"] == "completed"
        assert statuses["mining"] == "completed"
        assert statuses["debate"] == "completed"


class TestPipelineProgress:
    """_build_pipeline_progress() uses stage ordering heuristic."""

    def test_stage_1_baseline_in_progress(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {"stage": "1_baseline"}
        progress = svc._build_pipeline_progress(state)
        statuses = {p["stage"]: p["status"] for p in progress}
        assert statuses["baseline"] == "active"
        assert statuses["mining"] == "pending"

    def test_campaign_complete_all_done(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {"stage": "campaign_complete"}
        progress = svc._build_pipeline_progress(state)
        assert all(p["status"] == "completed" for p in progress)

    def test_unknown_stage_all_pending(self):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {"stage": "bogus_stage"}
        progress = svc._build_pipeline_progress(state)
        # current_idx = -1, so nothing is active/completed
        assert any(p["status"] == "pending" for p in progress)

    def test_7_campaign_eval_integration_completed(self):
        """stage=7_campaign_eval: integration should be completed (not active)."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {"stage": "7_campaign_eval"}
        progress = svc._build_pipeline_progress(state)
        statuses = {p["stage"]: p["status"] for p in progress}
        assert statuses["integration"] == "completed"

    def test_6_integration_integration_active(self):
        """stage=6_integration: integration should be active."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {"stage": "6_integration"}
        progress = svc._build_pipeline_progress(state)
        statuses = {p["stage"]: p["status"] for p in progress}
        assert statuses["integration"] == "active"


class TestReadArtifact:
    """read_artifact() blocks path traversal."""

    @pytest.mark.asyncio
    async def test_reads_valid_file(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        (tmp_path / "report.md").write_text("# Report")
        content, mime = await svc.read_artifact(str(tmp_path), "report.md")
        assert content == "# Report"
        assert mime == "text/markdown"

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        sensitive = tmp_path.parent / "secret.json"
        sensitive.write_text('{"secret": "value"}')
        content, mime = await svc.read_artifact(str(tmp_path), "../secret.json")
        assert content is None

    @pytest.mark.asyncio
    async def test_blocks_same_prefix_directory_bypass(self, tmp_path):
        """Sibling dir with same prefix must be blocked (classic startswith bypass)."""
        from orchestration.campaign_data_service import CampaignDataService
        # artifact_dir = tmp_path/abc123
        # evil_dir     = tmp_path/abc123_evil  ← same prefix but different dir
        base_dir = tmp_path / "abc123"
        base_dir.mkdir()
        evil_dir = tmp_path / "abc123_evil"
        evil_dir.mkdir()
        secret = evil_dir / "secret.txt"
        secret.write_text("top secret")

        svc = CampaignDataService()
        content, mime = await svc.read_artifact(str(base_dir), "../abc123_evil/secret.txt")
        assert content is None

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_file(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        content, mime = await svc.read_artifact(str(tmp_path), "nonexistent.txt")
        assert content is None

    @pytest.mark.asyncio
    async def test_json_mime_type(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        (tmp_path / "data.json").write_text('{"key": "val"}')
        content, mime = await svc.read_artifact(str(tmp_path), "data.json")
        assert mime == "application/json"

    @pytest.mark.asyncio
    async def test_python_mime_type(self, tmp_path):
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        (tmp_path / "kernel.py").write_text("import torch")
        content, mime = await svc.read_artifact(str(tmp_path), "kernel.py")
        assert mime == "text/x-python"


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

def _make_session_info(session_id, worktree_path=None):
    """Build a minimal SessionInfo-like object."""
    s = MagicMock()
    s.session_id = session_id
    s.worktree_path = worktree_path
    return s


def _make_session_list_response(sessions):
    resp = MagicMock()
    resp.sessions = sessions
    return resp


class TestCampaignsEndpoints:
    """API endpoints with mocked session_manager."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_get_campaigns_returns_campaigns_with_state(self, tmp_path):
        """GET /api/campaigns returns list of campaigns that have state.json."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        # Set up artifact dir in tmp_path
        art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
        art_dir.mkdir(parents=True)
        (art_dir / "state.json").write_text(json.dumps(SAMPLE_STATE))

        sessions = [
            _make_session_info("sess-1", str(tmp_path)),
            _make_session_info("sess-2", None),  # no worktree → skip
        ]
        mock_sm = AsyncMock()
        mock_sm.list_sessions = AsyncMock(return_value=_make_session_list_response(sessions))

        app_module.session_manager = mock_sm

        result = self._run(app_module.get_campaigns_overview(request=MagicMock(), client_id=None))
        assert "campaigns" in result
        assert len(result["campaigns"]) == 1
        assert result["campaigns"][0]["session_id"] == "sess-1"

    def test_get_campaigns_skips_sessions_without_state(self, tmp_path):
        """Sessions without state.json are not included."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        # tmp_path has no state.json
        sessions = [_make_session_info("sess-1", str(tmp_path))]
        mock_sm = AsyncMock()
        mock_sm.list_sessions = AsyncMock(return_value=_make_session_list_response(sessions))

        app_module.session_manager = mock_sm

        result = self._run(app_module.get_campaigns_overview(request=MagicMock(), client_id=None))
        assert result["campaigns"] == []

    def test_get_campaign_detail_returns_state(self, tmp_path):
        """GET /api/campaigns/{id} returns raw state dict."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        from fastapi import HTTPException as FastHTTPException

        art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
        art_dir.mkdir(parents=True)
        (art_dir / "state.json").write_text(json.dumps(SAMPLE_STATE))

        session = _make_session_info("sess-1", str(tmp_path))
        mock_sm = AsyncMock()
        mock_sm.get_session = AsyncMock(return_value=session)

        app_module.session_manager = mock_sm

        result = self._run(app_module.get_campaign_detail("sess-1", client_id=None))
        assert result["stage"] == "4_5_parallel_tracks"
        assert result["campaign"]["status"] == "active"

    def test_get_campaign_detail_404_when_no_session(self):
        """Returns 404 when session not found."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        from fastapi import HTTPException as FastHTTPException

        mock_sm = AsyncMock()
        mock_sm.get_session = AsyncMock(return_value=None)
        app_module.session_manager = mock_sm

        with pytest.raises(FastHTTPException) as exc_info:
            self._run(app_module.get_campaign_detail("sess-missing", client_id=None))
        assert exc_info.value.status_code == 404

    def test_get_campaign_detail_404_when_no_campaign_data(self, tmp_path):
        """Returns 404 when session has worktree but no state.json."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        from fastapi import HTTPException as FastHTTPException

        session = _make_session_info("sess-1", str(tmp_path))
        mock_sm = AsyncMock()
        mock_sm.get_session = AsyncMock(return_value=session)
        app_module.session_manager = mock_sm

        with pytest.raises(FastHTTPException) as exc_info:
            self._run(app_module.get_campaign_detail("sess-1", client_id=None))
        assert exc_info.value.status_code == 404

    def test_get_campaign_artifact_returns_content(self, tmp_path):
        """GET /api/campaigns/{id}/artifacts/{path} returns file content."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
        art_dir.mkdir(parents=True)
        (art_dir / "state.json").write_text(json.dumps(SAMPLE_STATE))
        (art_dir / "report.md").write_text("# My Report")

        session = _make_session_info("sess-1", str(tmp_path))
        mock_sm = AsyncMock()
        mock_sm.get_session = AsyncMock(return_value=session)
        app_module.session_manager = mock_sm

        from fastapi.responses import Response
        # Handler now accepts (request, session_id, path, client_id); the
        # method dispatcher branches on request.method so we hand it a mock
        # that reports GET.
        mock_request = MagicMock()
        mock_request.method = "GET"
        result = self._run(app_module.get_campaign_artifact(
            mock_request, "sess-1", "report.md", client_id=None
        ))
        assert result.body == b"# My Report"
        assert result.media_type == "text/markdown"

    def test_get_campaign_artifact_404_when_path_missing(self, tmp_path):
        """Returns 404 when artifact path doesn't exist."""
        import importlib
        import app as app_module
        importlib.reload(app_module)

        from fastapi import HTTPException as FastHTTPException

        art_dir = tmp_path / "kernel_opt_artifacts" / "model_h100_fp8_tp1"
        art_dir.mkdir(parents=True)
        (art_dir / "state.json").write_text(json.dumps(SAMPLE_STATE))

        session = _make_session_info("sess-1", str(tmp_path))
        mock_sm = AsyncMock()
        mock_sm.get_session = AsyncMock(return_value=session)
        app_module.session_manager = mock_sm

        mock_request = MagicMock()
        mock_request.method = "GET"
        with pytest.raises(FastHTTPException) as exc_info:
            self._run(app_module.get_campaign_artifact(
                mock_request, "sess-1", "missing.md", client_id=None
            ))
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 4. Additional edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases not covered by implementor tests."""

    def test_count_completed_but_not_shipped_counts_as_active(self):
        """A COMPLETED track NOT in shipped_optimizations must count as active, not shipped."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "parallel_tracks": {
                "op_a": {"status": "COMPLETED"},   # completed but NOT shipped
                "op_b": {"status": "FAILED"},
            },
            "campaign": {
                "shipped_optimizations": [],  # nothing shipped yet
            },
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 0, "Nothing in shipped_optimizations — shipped must be 0"
        assert failed == 1
        assert active == 1, "COMPLETED but not shipped counts as active"

    def test_count_track_statuses_accepts_terminal_schema_statuses(self):
        """Round-centric track counts must support AMMO terminal statuses."""
        from orchestration.campaign_data_service import CampaignDataService

        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "parallel_tracks": {
                            "tracks": {
                                "op_pass": {"status": "PASS", "verdict": "PASS"},
                                "op_gated": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                                "op_fail": {"status": "FAIL", "verdict": "FAIL"},
                                "op_gpu": {"status": "GPU_BLOCKED", "verdict": "GPU_BLOCKED"},
                                "op_live": {"status": "IN_PROGRESS", "verdict": None},
                            }
                        },
                    }
                ],
            }
        }

        shipped, failed, active = svc._count_track_statuses(state)

        assert shipped == 2
        assert failed == 2
        assert active == 1

    @pytest.mark.asyncio
    async def test_read_state_returns_none_on_corrupt_json(self, tmp_path):
        """read_state gracefully returns None on corrupt JSON."""
        from orchestration.campaign_data_service import CampaignDataService
        corrupt = tmp_path / "state.json"
        corrupt.write_text("{not valid json")
        svc = CampaignDataService()
        result = await svc.read_state(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# 5. Schema v3 speedup field normalization
# ---------------------------------------------------------------------------

# NOTE: TestSpeedupNormalization and TestBuildOverviewSpeedupNormalization
# were deleted as part of Artifact Layout V2 (Task 7). Their assertions
# depended on read_state() applying _normalize_speedup_field, which
# the v2 cleanup removed. The cumulative_e2e_speedup ↔
# cumulative_speedup_vs_round1 fallback now lives in
# CampaignDataService.build_l1_projection() and is covered by
# tests/unit/test_campaign_data_service_v2.py::TestBuildL1Projection.


# Debate rationale rendering used to be sidecar-driven (read_artifact_catalog
# walked .metrics.json envelopes for `labels.kind='debate_rationale'`). After
# sidecar removal, debate artifacts are discovered via the artifact tree
# endpoint plus path conventions (`rounds/N/debate/...`), and per-champion
# metadata moves to YAML frontmatter on the rationale .md files. Track 4
# (frontend rewrite) covers the new rendering path; the old sidecar-coupled
# tests in this section have been retired.


# ---------------------------------------------------------------------------
# 7. GATED_PASS rendering tests (Tests 5-6)
# ---------------------------------------------------------------------------

class TestGatedPassRendering:
    """GATED_PASS renders as validated/shipped (green) not gated (amber)."""

    def test_gated_pass_in_shipped_optimizations_counts_as_shipped(self):
        """Archived track with verdict GATED_PASS + in shipped_optimizations → shipped."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 2,
                "shipped_optimizations": [{"op_id": "op_gated_pass"}],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": ["op_gated_pass"],
                        "parallel_tracks": {
                            "tracks": {
                                "op_gated_pass": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                            }
                        },
                    },
                    {
                        "round_id": 2,
                        "shipped": [],
                        "parallel_tracks": {"tracks": {}},
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 1, "GATED_PASS in shipped_optimizations should count as shipped"
        assert failed == 0
        assert active == 0

    def test_gated_pass_not_in_shipped_counts_as_shipped_via_pass_statuses(self):
        """Archived track with verdict GATED_PASS + NOT in shipped_optimizations → still shipped (pass_statuses)."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 2,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": [],
                        "parallel_tracks": {
                            "tracks": {
                                "op_gated": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                            }
                        },
                    },
                    {
                        "round_id": 2,
                        "shipped": [],
                        "parallel_tracks": {"tracks": {}},
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        # GATED_PASS is in pass_statuses → counted as shipped
        assert shipped == 1, "GATED_PASS should be in pass_statuses and count as shipped"
        assert failed == 0
        assert active == 0

    def test_current_round_gated_pass_not_in_shipped(self):
        """Current-round track with verdict GATED_PASS + NOT in shipped_optimizations → shipped (pass_statuses)."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": [],
                        "parallel_tracks": {
                            "tracks": {
                                "op_gated": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                            }
                        },
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 1, "Current-round GATED_PASS in pass_statuses → shipped"
        assert active == 0

    def test_current_round_gated_pass_in_shipped(self):
        """Current-round track with verdict GATED_PASS + in shipped_optimizations → shipped."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [{"op_id": "op_gated"}],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": ["op_gated"],
                        "parallel_tracks": {
                            "tracks": {
                                "op_gated": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                            }
                        },
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 1, "Current-round GATED_PASS in shipped → shipped"


class TestGatedPassVariants:
    """Backend pass_statuses includes both GATED_PASS and GATED-PASS (Test 6)."""

    def test_gated_pass_underscore_counts_as_shipped(self):
        """verdict: 'GATED_PASS' in shipped list → shipped += 1."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": [],
                        "parallel_tracks": {
                            "tracks": {
                                "op1": {"status": "GATED_PASS", "verdict": "GATED_PASS"},
                            }
                        },
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 1

    def test_gated_pass_hyphen_counts_as_shipped(self):
        """verdict: 'GATED-PASS' → shipped += 1."""
        from orchestration.campaign_data_service import CampaignDataService
        svc = CampaignDataService()
        state = {
            "campaign": {
                "current_round": 1,
                "shipped_optimizations": [],
                "rounds": [
                    {
                        "round_id": 1,
                        "shipped": [],
                        "parallel_tracks": {
                            "tracks": {
                                "op1": {"status": "GATED-PASS", "verdict": "GATED-PASS"},
                            }
                        },
                    },
                ],
            }
        }
        shipped, failed, active = svc._count_track_statuses(state)
        assert shipped == 1

    def test_frontend_circuit_board_maps_gated_pass_to_validated(self):
        """circuit-board.js must NOT map GATED_PASS to 'gated' — should be validated or shipped."""
        from pathlib import Path
        cb_js = Path(__file__).parent.parent.parent / "frontend" / "js" / "circuit-board.js"
        content = cb_js.read_text()
        # After the fix, GATED_PASS should NOT result in status = 'gated'
        # Instead it should fall through to 'validated' or 'shipped'
        # The old pattern was: if (isGated) status = 'gated';
        assert "if (isGated) status = 'gated'" not in content, \
            "circuit-board.js still maps GATED_PASS to 'gated' — should be validated"

    def test_frontend_campaign_app_maps_gated_pass_to_validated(self):
        """campaign-app.js trackStatus() must NOT return 'gated' for GATED_PASS."""
        from pathlib import Path
        ca_js = Path(__file__).parent.parent.parent / "frontend" / "js" / "campaign-app.js"
        content = ca_js.read_text()
        # After the fix, the isGated → return 'gated' pattern should be removed
        assert "if (isGated) return 'gated'" not in content, \
            "campaign-app.js trackStatus() still returns 'gated' for GATED_PASS — should be validated"
