# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the v2 artifact-layout migration of CampaignDataService.

V2 layout pushes routing/inference responsibility into the artifact path
(`rounds/{N}/{stage}/...`) so the server can stop inferring stage/round/track_id
from paths. After the sidecar-removal cleanup, the L3 viewer drives off the
artifact tree endpoint plus path conventions instead of `.metrics.json` files.

Test sections:
    `build_l1_projection()` reconciled L1 shape.
    `list_artifact_tree()` + `GET /api/campaigns/{id}/tree` endpoint.
    `/api/campaign-data/{id}` state-only response and cross-pod compat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestration.campaign_data_service import CampaignDataService


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _write_state(artifact_dir: Path, **campaign_overrides) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "campaign": {
            "current_round": 1,
            "current_stage": "1_baseline",
            **campaign_overrides,
        },
    }
    (artifact_dir / "state.json").write_text(json.dumps(state))


@pytest.fixture
def svc() -> CampaignDataService:
    service = CampaignDataService()
    service._state_cache.clear()
    yield service
    service._state_cache.clear()


# =========================================================================== #
# build_l1_projection() reconciled L1 shape
# =========================================================================== #


def _full_state(**campaign_overrides) -> dict:
    """Realistic state.json with two rounds and per-track verdicts."""
    base_campaign = {
        "status": "active",
        "current_round": 2,
        "current_stage": "4_5_parallel_tracks",
        "cumulative_e2e_speedup": 1.34,
        "shipped_optimizations": [{"op_id": "op-flash-attn"}, {"op_id": "op-rope-fused"}],
        "rounds": [
            {
                "round_id": 1,
                "status": "completed",
                "shipped": ["op-flash-attn", "op-rope-fused"],
                "parallel_tracks": {
                    "tracks": {
                        "op-flash-attn": {
                            "status": "COMPLETED",
                            "verdict": "SHIPPED",
                            "kernel_speedup": 1.35,
                            "classification": "lossless",
                            "fail_reason": None,
                        },
                        "op-rope-fused": {
                            "status": "COMPLETED",
                            "verdict": "SHIPPED",
                            "kernel_speedup": 1.12,
                            "classification": "lossless",
                            "fail_reason": None,
                        },
                    },
                },
            },
            {
                "round_id": 2,
                "status": "in_progress",
                "shipped": [],
                "parallel_tracks": {
                    "tracks": {
                        "op-mlp-fused": {
                            "status": "IN_PROGRESS",
                            "verdict": None,
                            "kernel_speedup": None,
                            "classification": None,
                            "fail_reason": None,
                        },
                        "op-quant-int8": {
                            "status": "FAILED",
                            "verdict": "FAILED",
                            "kernel_speedup": None,
                            "classification": None,
                            "fail_reason": "correctness_failed",
                        },
                    },
                },
            },
        ],
    }
    base_campaign.update(campaign_overrides)
    return {
        "schema_version": "4.1",
        "target": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "L40S",
            "dtype": "fp8",
            "tp": 1,
        },
        "campaign": base_campaign,
    }


class TestBuildL1Projection:
    """`build_l1_projection()` must return a plain dict (not a Pydantic model)
    matching the reconciled FE contract:

        {
          session_id, created_at,
          target: {model_id, hardware, dtype, tp},
          campaign: {
            status, current_round, current_stage, cumulative_e2e_speedup,
            shipped_optimizations,
            shipped_count, failed_count, active_count,
            pipeline_progress: [{stage, status}],
            rounds: [{round_id, status, shipped, parallel_tracks: {tracks}}]
          }
        }
    """

    def test_returns_plain_dict(self, svc):
        out = svc.build_l1_projection("sess-1", _full_state(), created_at="2026-05-10T00:00:00Z")
        assert isinstance(out, dict)
        # Must NOT be a Pydantic model — projection has dropped that wrapper.
        assert not hasattr(out, "model_dump")

    def test_top_level_shape(self, svc):
        out = svc.build_l1_projection("sess-1", _full_state(), created_at="2026-05-10T00:00:00Z")
        assert out["session_id"] == "sess-1"
        assert out["created_at"] == "2026-05-10T00:00:00Z"
        assert out["target"] == {
            "model_id": "meta-llama/Meta-Llama-3.1-8B",
            "hardware": "L40S",
            "dtype": "fp8",
            "tp": 1,
        }

    def test_campaign_block_includes_required_fields(self, svc):
        out = svc.build_l1_projection("sess-1", _full_state())
        camp = out["campaign"]
        assert camp["status"] == "active"
        assert camp["current_round"] == 2
        assert camp["current_stage"] == "4_5_parallel_tracks"
        assert camp["cumulative_e2e_speedup"] == 1.34
        assert camp["shipped_optimizations"] == [
            {"op_id": "op-flash-attn"}, {"op_id": "op-rope-fused"},
        ]
        assert "shipped_count" in camp
        assert "failed_count" in camp
        assert "active_count" in camp
        assert "pipeline_progress" in camp
        assert "rounds" in camp

    def test_track_counts(self, svc):
        out = svc.build_l1_projection("sess-1", _full_state())
        camp = out["campaign"]
        # Two SHIPPED in round 1, one FAILED in round 2, one IN_PROGRESS in round 2.
        assert camp["shipped_count"] == 2
        assert camp["failed_count"] == 1
        assert camp["active_count"] == 1

    def test_pipeline_progress_is_list_of_stage_status_dicts(self, svc):
        out = svc.build_l1_projection("sess-1", _full_state())
        progress = out["campaign"]["pipeline_progress"]
        assert isinstance(progress, list)
        assert all(set(p.keys()) == {"stage", "status"} for p in progress)
        # current_stage = 4_5_parallel_tracks → implementation marker
        active = [p for p in progress if p["status"] == "active"]
        assert any(p["stage"] in ("implementation", "validation") for p in active)

    def test_rounds_carry_track_detail_fields(self, svc):
        """FE campaign-app.js reads round[*].parallel_tracks.tracks[*].{status,
        verdict, kernel_speedup, classification, fail_reason}."""
        out = svc.build_l1_projection("sess-1", _full_state())
        rounds = out["campaign"]["rounds"]
        assert isinstance(rounds, list) and len(rounds) == 2

        r1 = rounds[0]
        assert r1["round_id"] == 1
        assert r1["status"] == "completed"
        assert r1["shipped"] == ["op-flash-attn", "op-rope-fused"]
        tracks = r1["parallel_tracks"]["tracks"]
        flash = tracks["op-flash-attn"]
        assert flash["status"] == "COMPLETED"
        assert flash["verdict"] == "SHIPPED"
        assert flash["kernel_speedup"] == 1.35
        assert flash["classification"] == "lossless"
        assert flash["fail_reason"] is None

    def test_cumulative_e2e_speedup_falls_back_from_v3_field(self, svc):
        """When canonical `cumulative_e2e_speedup` is absent or 1.0, projection
        falls back to `cumulative_speedup_vs_round1`."""
        # Canonical absent.
        state = _full_state()
        state["campaign"].pop("cumulative_e2e_speedup", None)
        state["campaign"]["cumulative_speedup_vs_round1"] = 1.42
        out = svc.build_l1_projection("sess-1", state)
        assert out["campaign"]["cumulative_e2e_speedup"] == 1.42

        # Canonical at default (1.0); v3 field has real data — prefer v3.
        state["campaign"]["cumulative_e2e_speedup"] = 1.0
        out = svc.build_l1_projection("sess-1", state)
        assert out["campaign"]["cumulative_e2e_speedup"] == 1.42

        # Both absent.
        state["campaign"].pop("cumulative_e2e_speedup", None)
        state["campaign"].pop("cumulative_speedup_vs_round1", None)
        out = svc.build_l1_projection("sess-1", state)
        assert out["campaign"]["cumulative_e2e_speedup"] == 1.0

    def test_missing_campaign_key_returns_safe_defaults(self, svc):
        """If state has no `campaign` key, projection should still emit a
        well-formed shape with safe defaults rather than crashing."""
        out = svc.build_l1_projection(
            "sess-empty",
            {"target": {"model_id": "x", "hardware": "y", "dtype": "z", "tp": 1}},
        )
        assert out["session_id"] == "sess-empty"
        camp = out["campaign"]
        assert camp["status"] in ("unknown", "active", "")
        assert camp["current_round"] in (0, 1)
        assert camp["cumulative_e2e_speedup"] == 1.0
        assert camp["shipped_count"] == 0
        assert camp["failed_count"] == 0
        assert camp["active_count"] == 0
        assert isinstance(camp["pipeline_progress"], list)
        assert camp["rounds"] == []


# =========================================================================== #
# Tree endpoint + /api/campaign-data/{id} state-only response
# =========================================================================== #


class TestListArtifactTree:
    """`list_artifact_tree()` powers the L3 viewer after sidecar removal.

    Walks the artifact dir, returns POSIX-relative paths, excludes `_archive/`,
    compiler caches, `*.metrics.json`, `__pycache__/`, `.git/`. Empty / missing
    dirs return `{"root": <name>, "files": []}`.
    """

    @pytest.mark.asyncio
    async def test_walks_tree_and_returns_relative_posix_paths(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        _write_state(artifact_dir)
        (artifact_dir / "rounds" / "1" / "baseline").mkdir(parents=True)
        (artifact_dir / "rounds" / "1" / "baseline" / "notes.md").write_text("# notes")
        (artifact_dir / "rounds" / "1" / "mining" / "deep" / "nested").mkdir(parents=True)
        (artifact_dir / "rounds" / "1" / "mining" / "deep" / "nested" / "x.json").write_text("{}")

        result = await svc.list_artifact_tree(str(artifact_dir))

        assert result["root"] == "artifact"
        assert "state.json" in result["files"]
        assert "rounds/1/baseline/notes.md" in result["files"]
        assert "rounds/1/mining/deep/nested/x.json" in result["files"]
        # POSIX separators on every entry (no backslashes).
        assert all("\\" not in p for p in result["files"])
        # Sorted output.
        assert result["files"] == sorted(result["files"])

    @pytest.mark.asyncio
    async def test_excludes_archive_metrics_pycache_git_and_compiler_caches(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        _write_state(artifact_dir)
        # Excluded dirs.
        (artifact_dir / "_archive" / "round_1").mkdir(parents=True)
        (artifact_dir / "_archive" / "round_1" / "old.md").write_text("old")
        (artifact_dir / "__pycache__").mkdir()
        (artifact_dir / "__pycache__" / "x.pyc").write_text("")
        (artifact_dir / ".git").mkdir()
        (artifact_dir / ".git" / "HEAD").write_text("")
        for cache_name in ("cache", "triton_cache", "torch_compile_cache"):
            (artifact_dir / "rounds" / "1" / cache_name).mkdir(parents=True)
            (artifact_dir / "rounds" / "1" / cache_name / "compiled.bin").write_text("cache")
        # Excluded sidecar suffix.
        (artifact_dir / "rounds" / "1").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "rounds" / "1" / "report.md").write_text("# r")
        (artifact_dir / "rounds" / "1" / "report.md.metrics.json").write_text("{}")
        (artifact_dir / "rounds" / "1" / "cache_sensitivity.py").write_text("pass")

        result = await svc.list_artifact_tree(str(artifact_dir))

        files = result["files"]
        assert "rounds/1/report.md" in files
        assert all(not p.startswith("_archive/") for p in files), files
        assert all("__pycache__" not in p for p in files), files
        assert all(not p.startswith(".git/") for p in files), files
        assert all(not p.endswith(".metrics.json") for p in files), files
        assert all(not any(part in {"cache", "triton_cache", "torch_compile_cache"}
                           for part in p.split("/")) for p in files), files
        assert "rounds/1/cache_sensitivity.py" in files

    @pytest.mark.asyncio
    async def test_missing_dir_returns_empty_files(self, svc, tmp_path):
        result = await svc.list_artifact_tree(str(tmp_path / "does-not-exist"))
        assert result == {"root": "does-not-exist", "files": []}


class TestListArtifactChildren:
    @pytest.mark.asyncio
    async def test_lists_immediate_children_dirs_first_with_file_metadata(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        target = artifact_dir / "rounds" / "1"
        (target / "z-dir" / "nested").mkdir(parents=True)
        (target / "a-dir").mkdir()
        (target / "z-dir" / "nested" / "deep.md").write_text("not immediate")
        (target / "z.txt").write_text("hello")
        (target / "a.json").write_text("{}")

        result = await svc.list_artifact_children(str(artifact_dir), "rounds/1")

        assert result["path"] == "rounds/1"
        assert result["exists"] is True
        assert [e["name"] for e in result["entries"]] == [
            "a-dir", "z-dir", "a.json", "z.txt"
        ]
        assert result["entries"][0] == {
            "name": "a-dir", "path": "rounds/1/a-dir", "type": "directory"
        }
        json_entry = next(e for e in result["entries"] if e["name"] == "a.json")
        assert json_entry == {
            "name": "a.json",
            "path": "rounds/1/a.json",
            "type": "file",
            "size": 2,
            "mime": "application/json",
        }
        assert all(e["name"] != "deep.md" for e in result["entries"])

    @pytest.mark.asyncio
    async def test_excludes_internal_dirs_and_metrics_but_keeps_similar_filename(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        for name in ("_archive", "__pycache__", ".git", "cache", "triton_cache", "torch_compile_cache"):
            (artifact_dir / name).mkdir()
        (artifact_dir / "result.metrics.json").write_text("{}")
        (artifact_dir / "cache_sensitivity.py").write_text("pass")

        result = await svc.list_artifact_children(str(artifact_dir), "")

        assert result["path"] == ""
        assert [entry["name"] for entry in result["entries"]] == ["cache_sensitivity.py"]
        hidden = await svc.list_artifact_children(str(artifact_dir), "cache")
        assert hidden == {"path": "cache", "exists": False, "entries": []}

    @pytest.mark.asyncio
    async def test_missing_root_and_non_directory_target_return_exists_false(self, svc, tmp_path):
        missing = await svc.list_artifact_children(str(tmp_path / "missing"), "")
        assert missing == {"path": "", "exists": False, "entries": []}

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "file.txt").write_text("x")
        file_target = await svc.list_artifact_children(str(artifact_dir), "file.txt")
        assert file_target == {"path": "file.txt", "exists": False, "entries": []}

    @pytest.mark.asyncio
    async def test_rejects_traversal_and_omits_symlink_escape(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        (artifact_dir / "escape").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="escapes artifact root"):
            await svc.list_artifact_children(str(artifact_dir), "../outside")
        with pytest.raises(ValueError, match="escapes artifact root"):
            await svc.list_artifact_children(str(artifact_dir), "escape")

        root = await svc.list_artifact_children(str(artifact_dir), "")
        assert all(entry["name"] != "escape" for entry in root["entries"])


class TestCampaignTreeEndpoint:
    """`GET /api/campaigns/{id}/tree` returns the artifact tree from
    `list_artifact_tree()` with the same auth/ownership wiring as other
    campaign endpoints."""

    @pytest.mark.asyncio
    async def test_response_returns_tree_listing(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app
        from orchestration.campaign_data_service import CampaignDataService

        artifact_dir = tmp_path / "artifact"
        _write_state(artifact_dir)
        (artifact_dir / "rounds" / "1" / "baseline").mkdir(parents=True)
        (artifact_dir / "rounds" / "1" / "baseline" / "x.md").write_text("# x")

        mock_session = MagicMock(session_id="sess-1", worktree_path=str(tmp_path / "wt"))
        mock_session_mgr = AsyncMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        real_svc = CampaignDataService()

        async def _fake_find_dir(_wt):
            return str(artifact_dir)

        with patch("app.session_manager", mock_session_mgr), \
             patch.object(real_svc, "find_artifact_dir", side_effect=_fake_find_dir), \
             patch("app.campaign_data_service", real_svc):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaigns/sess-1/tree")

        assert resp.status_code == 200
        body = resp.json()
        assert body["root"] == "artifact"
        assert "rounds/1/baseline/x.md" in body["files"]
        assert "state.json" in body["files"]


class TestCampaignArtifactChildrenEndpoint:
    @pytest.mark.asyncio
    async def test_response_uses_owned_session_and_requested_directory(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app

        session = MagicMock(worktree_path=str(tmp_path / "wt"))
        session_mgr = AsyncMock()
        session_mgr.get_session = AsyncMock(return_value=session)
        listing = {
            "path": "rounds/1",
            "exists": True,
            "entries": [{"name": "tracks", "path": "rounds/1/tracks", "type": "directory"}],
        }
        owner_id = "123e4567-e89b-42d3-a456-426614174000"

        with patch("app.session_manager", session_mgr), \
             patch("app.campaign_data_service") as service:
            service.find_artifact_dir = AsyncMock(return_value="/artifacts/active")
            service.list_artifact_children = AsyncMock(return_value=listing)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/campaigns/sess-1/artifact-children",
                    params={"path": "rounds/1"},
                    headers={"X-Client-ID": owner_id},
                )

        assert response.status_code == 200
        assert response.json() == listing
        session_mgr.get_session.assert_awaited_once_with("sess-1", owner_id=owner_id)
        service.find_artifact_dir.assert_awaited_once_with(str(tmp_path / "wt"))
        service.list_artifact_children.assert_awaited_once_with(
            "/artifacts/active", "rounds/1"
        )

    @pytest.mark.asyncio
    async def test_traversal_is_reported_as_bad_request(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app

        session_mgr = AsyncMock()
        session_mgr.get_session = AsyncMock(
            return_value=MagicMock(worktree_path=str(tmp_path / "wt"))
        )
        with patch("app.session_manager", session_mgr), \
             patch("app.campaign_data_service") as service:
            service.find_artifact_dir = AsyncMock(return_value="/artifacts/active")
            service.list_artifact_children = AsyncMock(
                side_effect=ValueError("Artifact path escapes artifact root")
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/campaigns/sess-1/artifact-children",
                    params={"path": "../outside"},
                )

        assert response.status_code == 400
        assert response.json()["error"] == "Artifact path escapes artifact root"


class TestCampaignDataEndpointStateOnly:
    """The L3 `/api/campaign-data/{id}` response is now `{"state": ...}` only.

    Sidecar/`artifact_catalog` aggregation has been removed — the FE drives the
    L3 viewer off the new tree endpoint instead, and reads structured metrics
    directly from state.json.
    """

    @pytest.mark.asyncio
    async def test_response_includes_only_state_field(self, tmp_path):
        """Local-pod path: response is `{state: ...}` with no sidecars/catalog."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app
        from orchestration.campaign_data_service import CampaignDataService

        artifact_dir = tmp_path / "artifact"
        _write_state(artifact_dir, current_round=1, current_stage="2_bottleneck_mining")

        mock_session = MagicMock(session_id="sess-1", worktree_path=str(tmp_path / "wt"))
        mock_session_mgr = AsyncMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        real_svc = CampaignDataService()
        real_svc._state_cache.clear()

        async def _fake_find_all_dirs(_wt):
            return [str(artifact_dir)]

        with patch("app.session_manager", mock_session_mgr), \
             patch.object(real_svc, "find_all_artifact_dirs", side_effect=_fake_find_all_dirs), \
             patch("app.campaign_data_service", real_svc):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaign-data/sess-1")

        assert resp.status_code == 200
        body = resp.json()
        assert "state" in body, body
        assert "artifact_catalog" not in body, (
            f"sidecar removal: artifact_catalog must not appear in response; got {body!r}"
        )
        assert "sidecars" not in body, (
            f"sidecar removal: sidecars must not appear in response; got {body!r}"
        )

    @pytest.mark.asyncio
    async def test_missing_local_session_returns_404(self, tmp_path):
        """A missing local session returns 404 instead of proxying to a peer."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app

        mock_session_mgr = AsyncMock()
        mock_session_mgr.get_session = AsyncMock(return_value=None)

        with patch("app.session_manager", mock_session_mgr):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaign-data/sess-far-away")

        assert resp.status_code == 404
        assert resp.json()["error"] == "Session not found"

    @pytest.mark.asyncio
    async def test_state_in_response_has_e2e_latency_normalized(self, tmp_path):
        """State returned by L3 must carry `_normalize_e2e_latency` artifacts —
        cumulative_e2e_speedup populated, kernel_speedup flattened to scalar,
        `_s` suffix stripped from latency map keys.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app
        from orchestration.campaign_data_service import CampaignDataService

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        state = {
            "campaign": {
                "current_round": 1,
                "current_stage": "1_baseline",
                "shipped_optimizations": ["op-flash-attn"],
                "rounds": [{
                    "round_id": 1,
                    "baseline": {"e2e_latency": {"1": {"avg_s": 1.0}}},
                    "integration": {"e2e_latency_combined": {"1": {"avg_s": 0.8}}},
                    "parallel_tracks": {"tracks": {
                        "op-flash-attn": {
                            "status": "FAILED",
                            "kernel_speedup": {"1": 1.25, "target": 1.5},
                            "failure_reason": "kernel diverged",
                        },
                    }},
                }],
            }
        }
        (artifact_dir / "state.json").write_text(json.dumps(state))

        mock_session = MagicMock(session_id="sess-1", worktree_path=str(tmp_path / "wt"))
        mock_session_mgr = AsyncMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        real_svc = CampaignDataService()
        real_svc._state_cache.clear()

        async def _fake_find_all_dirs(_wt):
            return [str(artifact_dir)]

        with patch("app.session_manager", mock_session_mgr), \
             patch.object(real_svc, "find_all_artifact_dirs", side_effect=_fake_find_all_dirs), \
             patch("app.campaign_data_service", real_svc):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaign-data/sess-1")

        body = resp.json()
        st = body["state"]
        # _normalize_e2e_latency was applied — cumulative_e2e_speedup populated,
        # `_s` suffix stripped from latency entries, kernel_speedup flattened
        # to a scalar.
        assert st["campaign"].get("cumulative_e2e_speedup") == pytest.approx(1.25, abs=1e-3)
        bl = st["campaign"]["rounds"][0]["baseline"]["e2e_latency"]["1"]
        assert "avg" in bl and "avg_s" not in bl
        track = st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op-flash-attn"]
        assert isinstance(track["kernel_speedup"], (int, float))


class TestReadStateLeavesRawNormalizersAlone:
    """Task 4 — read_state() must NOT mutate the loaded JSON with
    `_normalize_fail_reasons`, `_normalize_shipped_ops`, or
    `_normalize_speedup_field`. The L2 and L3 endpoints apply
    `_normalize_e2e_latency` inline; the rest of those legacy walkers are
    now handled FE-side (or, in the case of speedup, by build_l1_projection's
    v3-field fallback)."""

    @pytest.mark.asyncio
    async def test_failure_reason_not_copied_to_fail_reason(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        state = {
            "campaign": {
                "rounds": [{
                    "round_id": 1,
                    "parallel_tracks": {"tracks": {
                        "op-a": {"failure_reason": "x", "status": "FAILED"},
                    }},
                }],
            }
        }
        (artifact_dir / "state.json").write_text(json.dumps(state))

        out = await svc.read_state(str(artifact_dir))
        track = out["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op-a"]
        assert track.get("failure_reason") == "x"
        # Server no longer mirrors failure_reason → fail_reason on read; FE
        # does cascading lookup.
        assert "fail_reason" not in track

    @pytest.mark.asyncio
    async def test_shipped_optimizations_list_of_strings_preserved(self, svc, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        state = {"campaign": {"shipped_optimizations": ["op-a", "op-b"]}}
        (artifact_dir / "state.json").write_text(json.dumps(state))

        out = await svc.read_state(str(artifact_dir))
        # Was previously coerced to list[{op_id: ...}]; FE handles either shape now.
        assert out["campaign"]["shipped_optimizations"] == ["op-a", "op-b"]

    @pytest.mark.asyncio
    async def test_speedup_field_not_copied_from_v3_name(self, svc, tmp_path):
        """`cumulative_speedup_vs_round1` must NOT auto-populate
        `cumulative_e2e_speedup` at read time — the v3-field fallback now
        lives in `build_l1_projection()` only."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        state = {
            "campaign": {
                "cumulative_speedup_vs_round1": 1.42,
                # `cumulative_e2e_speedup` intentionally absent.
            }
        }
        (artifact_dir / "state.json").write_text(json.dumps(state))

        out = await svc.read_state(str(artifact_dir))
        # No round data — _normalize_e2e_latency would set this to 1.0 if
        # called. Read_state should NOT call _normalize_speedup_field, so
        # `cumulative_e2e_speedup` should NOT be populated from the v3 field
        # (it may be populated by _normalize_e2e_latency only if read_state
        # still calls it; Task 4 removes that call too).
        camp = out["campaign"]
        # Either the normalizer left it alone (1.42 v3 still present, no
        # canonical), or it was stamped to 1.0 by _normalize_e2e_latency
        # under read_state. Task 4 removes the read-path normalizer call,
        # so canonical should NOT exist on read output.
        assert camp.get("cumulative_speedup_vs_round1") == 1.42
        assert "cumulative_e2e_speedup" not in camp, (
            "read_state must NOT inject cumulative_e2e_speedup; that's now "
            "either a build_l1_projection responsibility (L1) or an inline "
            "_normalize_e2e_latency call at the L2/L3 endpoint."
        )


class TestL3EndpointInlineNormalizer:
    """Task 4 — the L2 and L3 endpoints must apply `_normalize_e2e_latency`
    inline on the state dict, not via read_state. The L3 wrapper still
    surfaces a normalized state to circuit-board.js."""

    @pytest.mark.asyncio
    async def test_l3_inline_normalizer_invoked_on_each_request(self, tmp_path):
        from unittest.mock import AsyncMock, MagicMock, patch
        from httpx import ASGITransport, AsyncClient
        from app import app
        from orchestration.campaign_data_service import CampaignDataService

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        state = {
            "campaign": {
                "current_round": 1,
                "rounds": [{
                    "round_id": 1,
                    "baseline": {"e2e_latency": {"1": {"avg_s": 1.0}}},
                    "integration": {"e2e_latency_combined": {"1": {"avg_s": 0.5}}},
                }],
            }
        }
        (artifact_dir / "state.json").write_text(json.dumps(state))

        mock_session = MagicMock(session_id="sess-1", worktree_path=str(tmp_path / "wt"))
        mock_session_mgr = AsyncMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        real_svc = CampaignDataService()
        real_svc._state_cache.clear()

        async def _fake_find_all_dirs(_wt):
            return [str(artifact_dir)]

        with patch("app.session_manager", mock_session_mgr), \
             patch.object(real_svc, "find_all_artifact_dirs", side_effect=_fake_find_all_dirs), \
             patch("app.campaign_data_service", real_svc):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/campaign-data/sess-1")

        body = resp.json()
        st = body["state"]
        # Normalized: cumulative_e2e_speedup populated, _s suffix stripped.
        assert st["campaign"].get("cumulative_e2e_speedup") == pytest.approx(2.0, abs=1e-3)
        assert "avg" in st["campaign"]["rounds"][0]["baseline"]["e2e_latency"]["1"]
