# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the /health endpoint's new gpu + vllm blocks.

The /health response composition happens in app.py's route handler (NOT
inside JobManager.health_check()). Existing keys (status, job_stats,
gpu_manager) must remain unchanged for backward compatibility with local UI
and API consumers.
"""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fresh_app_with_fake_manager(
    gpu_type: str = "l40s",
    total_gpus: int = 4,
    available_gpus: int = 2,
    has_gpu_manager: bool = True,
):
    """Reload app.py and inject a fake gpu_manager with GPU counts."""
    import app as app_module
    importlib.reload(app_module)
    app_module.gpu_type = gpu_type

    if has_gpu_manager:
        gpu_mgr = MagicMock()
        gpu_mgr.get_gpu_count.return_value = total_gpus
        gpu_mgr.get_available_gpu_count.return_value = available_gpus
        app_module.gpu_manager = gpu_mgr
    else:
        app_module.gpu_manager = None

    return app_module


def _health_body(app_module):
    """Invoke the /health route handler and return the decoded JSON body."""
    response = _run(app_module.health_check())
    # Route returns a JSONResponse; decode the body for assertions.
    body = response.body.decode("utf-8") if hasattr(response, "body") else None
    if body:
        return json.loads(body), response
    # Fallback: dict-like
    return response, response


@pytest.mark.unit
class TestHealthGpuBlock:

    def test_health_returns_gpu_block(self):
        """/health must expose gpu.type + allowed_dtypes + counts."""
        app_module = _fresh_app_with_fake_manager(gpu_type="l40s", total_gpus=4, available_gpus=2)
        body, response = _health_body(app_module)
        assert response.status_code == 200
        assert "gpu" in body, "/health must include new gpu block"
        gpu = body["gpu"]
        assert gpu["type"] == "l40s"
        assert gpu["allowed_dtypes"] == ["fp8", "bf16", "fp16"]
        assert gpu["total_gpus"] == 4
        assert gpu["available_gpus"] == 2

    def test_health_preserves_existing_gpu_manager_field(self):
        """Existing clients rely on gpu_manager.total_gpus — keep it."""
        app_module = _fresh_app_with_fake_manager(gpu_type="h100", total_gpus=8, available_gpus=8)
        body, _ = _health_body(app_module)
        assert "gpu_manager" in body
        assert body["gpu_manager"]["total_gpus"] == 8
        assert body["gpu_manager"]["available_gpus"] == 8

    def test_health_gpu_block_handles_missing_gpu_manager(self):
        """When gpu_manager is None, counts fall back to 0 (never None).

        Local UI and API consumers expect numeric counts, so the handler must
        emit ints (0), never None.
        """
        app_module = _fresh_app_with_fake_manager(
            gpu_type="l40s", has_gpu_manager=False,
        )
        body, response = _health_body(app_module)
        assert response.status_code == 200
        # Both the gpu block and the compatibility gpu_manager block must be int.
        assert body["gpu"]["total_gpus"] == 0
        assert body["gpu"]["available_gpus"] == 0
        assert body["gpu_manager"]["total_gpus"] == 0
        assert body["gpu_manager"]["available_gpus"] == 0
        assert isinstance(body["gpu_manager"]["total_gpus"], int)
        assert isinstance(body["gpu_manager"]["available_gpus"], int)

    def test_health_gpu_counts_are_int_when_manager_read_raises(self):
        """If gpu_manager.get_*_count() raises on a degraded boot, /health must
        still emit integer counts (0), not None — preserving the API
        contract."""
        app_module = _fresh_app_with_fake_manager(gpu_type="l40s")
        app_module.gpu_manager.get_gpu_count.side_effect = RuntimeError("locks not ready")
        app_module.gpu_manager.get_available_gpu_count.side_effect = RuntimeError("locks not ready")
        body, response = _health_body(app_module)
        assert response.status_code == 200
        assert body["gpu_manager"]["total_gpus"] == 0
        assert body["gpu_manager"]["available_gpus"] == 0
        assert isinstance(body["gpu_manager"]["total_gpus"], int)
        assert isinstance(body["gpu_manager"]["available_gpus"], int)

    def test_health_gpu_type_unknown_fallback(self):
        """When gpu_type is None, fall back to 'unknown' + safe dtype list."""
        app_module = _fresh_app_with_fake_manager(gpu_type=None, total_gpus=0, available_gpus=0)
        app_module.gpu_type = None  # belt-and-braces
        body, _ = _health_body(app_module)
        assert body["gpu"]["type"] == "unknown"
        assert body["gpu"]["allowed_dtypes"] == ["bf16", "fp16"]


@pytest.mark.unit
class TestHealthVllmBlock:

    def test_health_returns_vllm_block_with_commit_and_version(self, tmp_path):
        """When /workspace/vllm/.docker_commit and .docker_version exist, /health exposes them."""
        app_module = _fresh_app_with_fake_manager()

        # Patch Path so /workspace/vllm/.docker_commit and .docker_version
        # read from tmp files.
        commit = "a" * 60  # will be truncated to 40
        version = "v0.20.0"

        def fake_read(filename, max_len=None):
            if filename == ".docker_commit":
                val = commit
            elif filename == ".docker_version":
                val = version
            else:
                return None
            if max_len is not None:
                val = val[:max_len]
            return val

        with patch.object(app_module, "_read_vllm_artifact", side_effect=fake_read):
            body, _ = _health_body(app_module)

        assert "vllm" in body
        assert body["vllm"]["docker_commit"] == "a" * 40
        assert body["vllm"]["version"] == "v0.20.0"

    def test_health_vllm_block_null_when_files_missing(self):
        """Missing artifact files -> nulls, no crash."""
        app_module = _fresh_app_with_fake_manager()

        def fake_read(filename, max_len=None):
            return None

        with patch.object(app_module, "_read_vllm_artifact", side_effect=fake_read):
            body, _ = _health_body(app_module)

        assert body["vllm"] == {"docker_commit": None, "version": None}


@pytest.mark.unit
class TestReadVllmArtifactIsModuleLevel:
    """The _read_vllm_artifact helper must be module-level in app.py."""

    def test_read_vllm_artifact_importable(self):
        """Plan Task 2 requires _read_vllm_artifact extracted to module level."""
        import app as app_module
        importlib.reload(app_module)
        assert hasattr(app_module, "_read_vllm_artifact"), (
            "_read_vllm_artifact must be a module-level helper in app.py (Task 7)"
        )
        assert callable(app_module._read_vllm_artifact)
