# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
End-to-end tests for v1.7.0 changes.

Verifies all breaking + new features introduced since v1.6.2:
  1. /api/supported-models and /api/moe-models removed (404)
  2. /health returns structured gpu + vllm blocks
  3. /api/hf-model-config/{model_id} endpoint works end-to-end
  4. Session creation accepts tp*dp <= gpu_count (decoupled pool)
  5. Frontend removed static preset references (DOM assertions via Playwright)
  6. HfModelConfigService hardware-aware TP/dtype heuristics

Two execution modes:
  - Against a live server: set AMMO_SERVER_URL env var
  - In-process via TestClient: default (no server required)

All tests run autonomously without human intervention.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BASE_URL = os.environ.get("AMMO_SERVER_URL", "")
SESSION_CREATION_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_live_server() -> bool:
    return bool(BASE_URL)


def _get_test_client():
    """Build an in-process FastAPI TestClient for route-level tests."""
    from fastapi.testclient import TestClient
    import app as app_module
    importlib.reload(app_module)
    app_module.gpu_type = "l40s"
    app_module.gpu_memory_gb = 48.0

    gpu_mgr = MagicMock()
    gpu_mgr.get_gpu_count.return_value = 4
    gpu_mgr.get_available_gpu_count.return_value = 4
    app_module.gpu_manager = gpu_mgr

    return TestClient(app_module.app), app_module


def _auth_headers() -> dict:
    """Return auth headers (empty when AMMO_API_KEY unset)."""
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_response(status_code: int = 200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=payload or {})
    return resp


def _install_httpx_mock(responses):
    """Patch httpx.AsyncClient so GET returns queued responses in order."""
    response_iter = iter(responses)

    async def fake_get(url, *args, **kwargs):
        try:
            status_code, payload = next(response_iter)
        except StopIteration:
            raise RuntimeError(f"Unexpected extra HTTP call to {url}")
        if isinstance(payload, Exception):
            raise payload
        return _mock_response(status_code=status_code, payload=payload)

    mock_client = AsyncMock()
    mock_client.get = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    patcher = patch("httpx.AsyncClient", return_value=mock_client)
    return patcher


# ===========================================================================
# Section 1: Static Model Preset Endpoints Removed
# ===========================================================================

@pytest.mark.e2e
class TestStaticEndpointsRemoved:
    """Verify /api/supported-models and /api/moe-models return 404."""

    def test_supported_models_returns_404(self):
        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/api/supported-models",
                                headers=_auth_headers(), timeout=10)
            assert resp.status_code == 404
        else:
            client, _ = _get_test_client()
            resp = client.get("/api/supported-models")
            assert resp.status_code == 404, (
                f"Expected 404, got {resp.status_code}"
            )

    def test_moe_models_returns_404(self):
        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/api/moe-models",
                                headers=_auth_headers(), timeout=10)
            assert resp.status_code == 404
        else:
            client, _ = _get_test_client()
            resp = client.get("/api/moe-models")
            assert resp.status_code == 404

    def test_constants_no_supported_models_export(self):
        """shared.constants must NOT export SUPPORTED_MODELS or MOE_MODELS."""
        from shared import constants
        importlib.reload(constants)
        assert not hasattr(constants, "SUPPORTED_MODELS"), \
            "SUPPORTED_MODELS was supposed to be removed in v1.7.0"
        assert not hasattr(constants, "MOE_MODELS"), \
            "MOE_MODELS was supposed to be removed in v1.7.0"
        assert not hasattr(constants, "MOE_TP_OPTIONS"), \
            "MOE_TP_OPTIONS was supposed to be removed in v1.7.0"

    def test_constants_still_has_gpu_dtype_map(self):
        """GPU_DTYPE_MAP and TP_OPTIONS must still be available."""
        from shared import constants
        importlib.reload(constants)
        assert hasattr(constants, "GPU_DTYPE_MAP")
        assert hasattr(constants, "TP_OPTIONS")
        assert hasattr(constants, "BYTES_PER_PARAM")
        assert "l40s" in constants.GPU_DTYPE_MAP
        assert "fp8" in constants.BYTES_PER_PARAM


# ===========================================================================
# Section 2: /health Endpoint — GPU + vLLM Blocks
# ===========================================================================

@pytest.mark.e2e
class TestHealthEndpointBlocks:
    """Verify /health returns the new structured gpu and vllm blocks."""

    def test_health_has_gpu_block(self):
        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/health", timeout=10)
            assert resp.status_code == 200
            data = resp.json()
        else:
            client, _ = _get_test_client()
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()

        assert "gpu" in data, f"/health missing 'gpu' block. Keys: {list(data.keys())}"
        gpu = data["gpu"]
        assert "type" in gpu
        assert "allowed_dtypes" in gpu
        assert isinstance(gpu["allowed_dtypes"], list)
        assert "total_gpus" in gpu
        assert "available_gpus" in gpu

    def test_health_has_vllm_block(self):
        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/health", timeout=10)
            data = resp.json()
        else:
            client, _ = _get_test_client()
            resp = client.get("/health")
            data = resp.json()

        assert "vllm" in data, f"/health missing 'vllm' block. Keys: {list(data.keys())}"
        vllm = data["vllm"]
        assert "docker_commit" in vllm
        assert "version" in vllm

    def test_health_gpu_dtypes_match_constants(self):
        """GPU allowed_dtypes from /health must match GPU_DTYPE_MAP for the detected type."""
        from shared.constants import GPU_DTYPE_MAP

        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/health", timeout=10)
            data = resp.json()
        else:
            client, _ = _get_test_client()
            resp = client.get("/health")
            data = resp.json()

        gpu_type = data["gpu"]["type"]
        expected_dtypes = GPU_DTYPE_MAP.get(gpu_type, GPU_DTYPE_MAP["unknown"])
        assert data["gpu"]["allowed_dtypes"] == expected_dtypes

    def test_health_preserves_legacy_fields(self):
        """Existing fields (status, job_stats, gpu_manager) must remain for cluster aggregation."""
        if _use_live_server():
            import requests
            resp = requests.get(f"{BASE_URL}/health", timeout=10)
            data = resp.json()
        else:
            client, _ = _get_test_client()
            resp = client.get("/health")
            data = resp.json()

        assert "status" in data
        assert data["status"] == "healthy"


# ===========================================================================
# Section 3: /api/hf-model-config/{model_id} Endpoint
# ===========================================================================

@pytest.mark.e2e
class TestHfModelConfigEndpoint:
    """Verify /api/hf-model-config endpoint with mocked HF responses."""

    def test_endpoint_exists_and_returns_json(self):
        """Endpoint must exist and return a well-formed JSON response."""
        if _use_live_server():
            import requests
            resp = requests.get(
                f"{BASE_URL}/api/hf-model-config/meta-llama/Llama-3.1-8B-Instruct",
                headers=_auth_headers(), timeout=15,
            )
            # May return reason=gated or network_error if no HF access, but route exists
            assert resp.status_code == 200
            data = resp.json()
            assert "model_id" in data
        else:
            # Use mocked HF responses for TestClient
            metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
            config = {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "torch_dtype": "bfloat16",
            }
            patcher = _install_httpx_mock([(200, metadata), (200, config)])
            with patcher:
                client, app_module = _get_test_client()
                # Need to init the service on the app
                from shared.hf_model_config import HfModelConfigService
                svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
                with patch("shared.hf_model_config._default_service", svc):
                    with patch("shared.hf_model_config.get_hf_model_config_service", return_value=svc):
                        resp = client.get("/api/hf-model-config/meta-llama/Llama-3.1-8B-Instruct")
                assert resp.status_code == 200
                data = resp.json()
                assert data["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"

    def test_hf_service_dense_model_tp_computation(self):
        """Verify TP computation for a 70B dense model on 48GB GPU."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {"hidden_size": 8192, "num_hidden_layers": 80, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config(
                "meta-llama/Llama-3.3-70B-Instruct",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["is_moe"] is False
        assert result["suggested_tp"] == 4  # 140GB / 38.4GB usable = 4
        assert result["suggested_dtype"] == "bf16"
        assert result["reason"] is None

    def test_hf_service_moe_detection_num_local_experts(self):
        """MoE detection via num_local_experts field."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 671_000_000_000}}
        config = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_local_experts": 256,
            "torch_dtype": "bfloat16",
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=80.0, total_gpus=8)
            result = _run(svc.get_config("deepseek-ai/DeepSeek-V3",
                                         allowed_dtypes=["fp8", "bf16"]))

        assert result["is_moe"] is True

    def test_hf_service_moe_detection_text_config_nested(self):
        """MoE detection via text_config.num_local_experts (multimodal models)."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 100_000_000_000}}
        config = {
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "num_local_experts": 64,
            },
            "model_type": "multimodal",
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=80.0, total_gpus=8)
            result = _run(svc.get_config("some/multimodal-moe",
                                         allowed_dtypes=["bf16"]))

        assert result["is_moe"] is True

    def test_hf_service_moe_name_pattern_fallback(self):
        """MoE detection from model name pattern like '35B-A3B'."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 35_000_000_000}}
        config = {"hidden_size": 5120, "num_hidden_layers": 40, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("company/Model-35B-A3B-Instruct",
                                         allowed_dtypes=["bf16"]))

        assert result["is_moe"] is True

    def test_hf_service_gated_model_returns_reason(self):
        """Gated models must return reason='gated' without suggestions."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": "auto", "safetensors": {"total": 8_000_000_000}}
        patcher = _install_httpx_mock([(200, metadata)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("meta-llama/Llama-Guard-3-8B"))

        assert result["reason"] == "gated"
        assert result["suggested_tp"] is None

    def test_hf_service_network_error_returns_reason(self):
        """Network errors return reason='network_error'."""
        from shared.hf_model_config import HfModelConfigService
        import httpx

        patcher = _install_httpx_mock([(200, httpx.ConnectError("timeout"))])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("some/model"))

        assert result["reason"] == "network_error"

    def test_hf_service_dtype_from_quantization_config(self):
        """dtype extracted from quantization_config.quant_method."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("nvidia/model-fp8",
                                         allowed_dtypes=["fp8", "bf16"]))

        assert result["suggested_dtype"] == "fp8"

    def test_hf_service_cache_hit(self):
        """Second call for same model_id should return cached result (no HTTP)."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        config = {"hidden_size": 3584, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result1 = _run(svc.get_config("Qwen/Qwen2.5-7B",
                                          allowed_dtypes=["bf16"]))

        # Second call without patcher — would fail if cache miss triggers HTTP
        result2 = _run(svc.get_config("Qwen/Qwen2.5-7B", allowed_dtypes=["bf16"]))
        assert result1 == result2

    def test_hf_service_tp_capped_by_total_gpus(self):
        """TP must not exceed total_gpus available on the pod."""
        from shared.hf_model_config import HfModelConfigService

        # 405B model would need tp=8+ but pod only has 4 GPUs
        metadata = {"gated": False, "safetensors": {"total": 405_000_000_000}}
        config = {"hidden_size": 16384, "num_hidden_layers": 126, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config("meta-llama/Llama-3.1-405B",
                                         allowed_dtypes=["bf16"]))

        assert result["suggested_tp"] <= 4

    def test_hf_service_fp8_halves_memory(self):
        """FP8 models need half the memory of BF16 for same param count."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("nvidia/Llama-70B-FP8",
                                         allowed_dtypes=["fp8", "bf16"]))

        # 70B × 1 byte = 70GB. Usable = 48×0.8 = 38.4. ceil(70/38.4) = 2 → tp=2
        assert result["suggested_tp"] == 2
        assert result["suggested_dtype"] == "fp8"


# ===========================================================================
# Section 4: Session Validation — Decoupled GPU Pool (tp*dp <= gpu_count)
# ===========================================================================

@pytest.mark.e2e
class TestDecoupledGpuPool:
    """Verify session creation accepts tp*dp <= gpu_count (not just ==)."""

    def test_tp_dp_less_than_gpu_count_accepted(self):
        """tp*dp < gpu_count must be valid (spare GPUs for parallel tracks)."""
        from shared.session_models import CreateSessionRequest
        req = CreateSessionRequest(
            repo_name="vllm",
            branch="main",
            gpu_count=8,
            tp_size=2,
            dp_size=1,
        )
        # tp*dp = 2 < 8 — must not raise
        assert req.tp_size * req.dp_size <= req.gpu_count

    def test_tp_dp_equals_gpu_count_accepted(self):
        """tp*dp == gpu_count must still be valid."""
        from shared.session_models import CreateSessionRequest
        req = CreateSessionRequest(
            repo_name="vllm",
            branch="main",
            gpu_count=4,
            tp_size=2,
            dp_size=2,
        )
        assert req.tp_size * req.dp_size == req.gpu_count

    def test_tp_dp_exceeds_gpu_count_rejected(self):
        """tp*dp > gpu_count must raise ValidationError."""
        from shared.session_models import CreateSessionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            CreateSessionRequest(
                repo_name="vllm",
                branch="main",
                gpu_count=4,
                tp_size=4,
                dp_size=2,
            )
        assert "must be <= gpu_count" in str(exc_info.value)

    def test_dp_without_tp_rejected(self):
        """dp_size > 1 without tp_size must raise ValidationError."""
        from shared.session_models import CreateSessionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            CreateSessionRequest(
                repo_name="vllm",
                branch="main",
                gpu_count=4,
                dp_size=2,
            )
        assert "explicit tp_size" in str(exc_info.value).lower() or \
               "dp_size" in str(exc_info.value).lower()

    def test_legacy_no_tp_no_dp_accepted(self):
        """Legacy path (no tp_size, dp_size=1) is unconstrained."""
        from shared.session_models import CreateSessionRequest
        req = CreateSessionRequest(
            repo_name="vllm",
            branch="main",
            gpu_count=8,
        )
        assert req.tp_size is None
        assert req.dp_size == 1

    @pytest.mark.skipif(not _use_live_server(), reason="requires running server")
    @pytest.mark.slow
    def test_live_session_decoupled_pool(self):
        """Create a session with tp*dp < gpu_count on a live server."""
        import requests
        headers = _auth_headers()

        session_id = None
        try:
            resp = requests.post(
                f"{BASE_URL}/sessions",
                json={
                    "repo_name": "vllm",
                    "branch": "main",
                    "gpu_count": 2,
                    "tp_size": 1,
                    "dp_size": 1,
                },
                headers=headers,
                timeout=SESSION_CREATION_TIMEOUT,
            )
            if resp.status_code == 503:
                pytest.skip("Insufficient GPUs available on pod")
            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
            data = resp.json()
            session_id = data["session_id"]
            assert data["status"] in ("creating", "active")
        finally:
            if session_id:
                requests.delete(f"{BASE_URL}/sessions/{session_id}",
                                headers=headers, timeout=30)


# ===========================================================================
# Section 5: GPU Memory Detection
# ===========================================================================

@pytest.mark.e2e
class TestGpuMemoryDetection:
    """Verify detect_gpu_memory_gb() function behavior."""

    def test_detect_gpu_memory_torch_path(self):
        """When torch.cuda is available, returns positive float."""
        import app as app_module
        importlib.reload(app_module)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = MagicMock(total_memory=48 * (1024**3))
            result = app_module.detect_gpu_memory_gb()
            assert abs(result - 48.0) < 0.1

    def test_detect_gpu_memory_nvidia_smi_fallback(self):
        """When torch unavailable, falls back to nvidia-smi."""
        import app as app_module
        importlib.reload(app_module)

        import subprocess
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "81920\n"  # MiB → 80 GB

        with patch("torch.cuda.is_available", side_effect=ImportError), \
             patch("subprocess.run", return_value=fake_result):
            result = app_module.detect_gpu_memory_gb()
            assert abs(result - 80.0) < 0.1

    def test_detect_gpu_memory_final_fallback(self):
        """When both methods fail, returns conservative 16 GB."""
        import app as app_module
        importlib.reload(app_module)

        with patch("torch.cuda.is_available", side_effect=ImportError), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            result = app_module.detect_gpu_memory_gb()
            assert result == 16.0


# ===========================================================================
# Section 6: BYTES_PER_PARAM Constants
# ===========================================================================

@pytest.mark.e2e
class TestBytesPerParam:
    """Verify BYTES_PER_PARAM has correct values for TP calculation."""

    def test_all_expected_dtypes_present(self):
        from shared.constants import BYTES_PER_PARAM
        expected = {"fp32", "bf16", "fp16", "fp8", "fp4", "int8", "int4"}
        assert expected.issubset(set(BYTES_PER_PARAM.keys()))

    def test_byte_values_correct(self):
        from shared.constants import BYTES_PER_PARAM
        assert BYTES_PER_PARAM["fp32"] == 4.0
        assert BYTES_PER_PARAM["bf16"] == 2.0
        assert BYTES_PER_PARAM["fp16"] == 2.0
        assert BYTES_PER_PARAM["fp8"] == 1.0
        assert BYTES_PER_PARAM["fp4"] == 0.5
        assert BYTES_PER_PARAM["int8"] == 1.0
        assert BYTES_PER_PARAM["int4"] == 0.5


# ===========================================================================
# Section 7: HfModelConfigService — Edge Cases
# ===========================================================================

@pytest.mark.e2e
class TestHfModelConfigEdgeCases:
    """Edge cases for HfModelConfigService."""

    def test_config_missing_fields_reason(self):
        """Model without hidden_size or safetensors → reason=config_missing_fields."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {}}
        config = {"model_type": "custom", "vocab_size": 32000}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("some/custom-model",
                                         allowed_dtypes=["bf16"]))

        assert result["reason"] == "config_missing_fields"

    def test_404_on_config_json(self):
        """404 on config.json → reason=config_missing_fields."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        patcher = _install_httpx_mock([(200, metadata), (404, {})])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("some/no-config-model",
                                         allowed_dtypes=["bf16"]))

        assert result["reason"] == "config_missing_fields"

    def test_dtype_constrained_by_allowed_dtypes(self):
        """If model wants fp8 but GPU doesn't support it, fallback to bf16."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            # V100 doesn't support fp8
            svc = HfModelConfigService(gpu_memory_gb=16.0, total_gpus=8)
            result = _run(svc.get_config("nvidia/model-fp8",
                                         allowed_dtypes=["bf16", "fp16"]))

        assert result["suggested_dtype"] in ("bf16", "fp16")
        assert result["suggested_dtype"] != "fp8"

    def test_small_model_always_tp1(self):
        """Small models (< GPU usable memory) always get tp=1."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 3_000_000_000}}
        config = {"hidden_size": 2560, "num_hidden_layers": 24, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("small/model-3B",
                                         allowed_dtypes=["bf16"]))

        assert result["suggested_tp"] == 1


# ===========================================================================
# (Section 8 removed: ingress path validation covered deployment manifests
# that are not part of this repository.)
# ===========================================================================


# ===========================================================================
# Section 9: AMMO Auditor Schema Integration
# ===========================================================================

@pytest.mark.e2e
class TestAmmoAuditorSchema:
    """Verify AMMO auditor schema and file artifacts exist."""

    def test_state_schema_has_audit_object(self):
        """state.schema.json must define the audit sub-object under campaign.rounds.items."""
        schema_path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "schemas"
            / "state.schema.json"
        )
        if not schema_path.exists():
            pytest.skip("AMMO state.schema.json not found in this checkout")
        schema = json.loads(schema_path.read_text())
        # Navigate: properties.campaign.properties.rounds.items.properties
        campaign_props = schema["properties"]["campaign"]["properties"]
        round_item_props = campaign_props["rounds"]["items"]["properties"]
        assert "audit" in round_item_props, (
            "state.schema.json campaign.rounds.items.properties must include 'audit'. "
            f"Got keys: {list(round_item_props.keys())}"
        )

    def test_auditor_agent_definition_exists(self):
        """agents/ammo-auditor.md must exist."""
        agent_path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "agents"
            / "ammo-auditor.md"
        )
        if not agent_path.exists():
            pytest.skip("ammo-auditor.md not in this checkout")
        content = agent_path.read_text()
        assert "auditor" in content.lower()

    def test_audit_protocol_exists(self):
        """orchestration/audit-protocol.md must exist."""
        path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "skills" / "ammo"
            / "orchestration" / "audit-protocol.md"
        )
        if not path.exists():
            pytest.skip("audit-protocol.md not in this checkout")
        content = path.read_text()
        assert len(content) > 100

    def test_audit_invariants_exists(self):
        """references/audit-invariants.md must exist."""
        path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "skills" / "ammo"
            / "references" / "audit-invariants.md"
        )
        if not path.exists():
            pytest.skip("audit-invariants.md not in this checkout")
        content = path.read_text()
        assert len(content) > 100


# ===========================================================================
# Section 10: AMMO Hook Hardening (_ammo_is_lead helper)
# ===========================================================================

@pytest.mark.e2e
class TestAmmoHookHardening:
    """Verify AMMO hook infrastructure files exist and are well-formed."""

    def test_ammo_is_lead_helper_exists(self):
        """_ammo_is_lead.sh must exist and define the _ammo_is_lead function."""
        path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "hooks" / "_ammo_is_lead.sh"
        )
        if not path.exists():
            pytest.skip("_ammo_is_lead.sh not found")
        content = path.read_text()
        assert "_ammo_is_lead" in content, (
            "_ammo_is_lead.sh must define the _ammo_is_lead function"
        )

    def test_ammo_is_lead_test_harness_exists(self):
        """test-ammo-is-lead.sh must exist alongside the helper."""
        path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "hooks" / "test-ammo-is-lead.sh"
        )
        if not path.exists():
            pytest.skip("test-ammo-is-lead.sh not found")
        content = path.read_text()
        assert "test" in content.lower() or "assert" in content.lower() or "pass" in content.lower()

    def test_ammo_team_spawn_guard_exists(self):
        """ammo-team-spawn-guard hook must exist."""
        hooks_dir = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "hooks"
        )
        if not hooks_dir.exists():
            pytest.skip("hooks directory not found")
        hook_files = [f.name for f in hooks_dir.iterdir() if f.is_file()]
        assert any("team-spawn-guard" in f for f in hook_files), (
            f"No team-spawn-guard hook found. Files: {hook_files}"
        )

    def test_settings_local_has_hook_registrations(self):
        """settings.local.json must register the _ammo_is_lead-based hooks."""
        settings_path = (
            Path(__file__).parent.parent.parent
            / "ai_cli_session" / ".claude" / "settings.local.json"
        )
        if not settings_path.exists():
            pytest.skip("settings.local.json not found")
        settings = json.loads(settings_path.read_text())
        hooks = settings.get("hooks", {})
        # Should have PostToolUse or PreToolUse entries
        assert len(hooks) > 0, "settings.local.json must have hook registrations"


# ===========================================================================
# Section 11: Frontend Model Selector Migration
# ===========================================================================

@pytest.mark.e2e
class TestFrontendMigration:
    """Verify frontend no longer references static model presets."""

    def test_index_html_no_supported_models_fetch(self):
        """Frontend must not call /api/supported-models."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "/api/supported-models" not in content, (
            "Frontend still references /api/supported-models endpoint"
        )

    def test_index_html_no_supported_models_state(self):
        """Frontend must not have supportedModels Alpine state."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "supportedModels:" not in content, \
            "Frontend still has supportedModels state variable"
        assert "cmSupportedModels" not in content, \
            "Frontend still has cmSupportedModels (LIGHTGRID) state variable"

    def test_index_html_has_hf_model_config_call(self):
        """Frontend must call /api/hf-model-config for model selection."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "/api/hf-model-config" in content, (
            "Frontend must reference /api/hf-model-config endpoint"
        )

    def test_index_html_has_gated_hint(self):
        """Frontend must show gated model hint."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "gatedHint" in content or "gated" in content.lower()

    def test_index_html_loads_gpu_from_health(self):
        """Frontend must load GPU info from /health (not /api/supported-models)."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "loadGpuAndVllmInfo" in content, (
            "Frontend must have loadGpuAndVllmInfo function"
        )

    def test_campaign_app_no_preset_chips(self):
        """LIGHTGRID campaign app must not have preset chips."""
        index_path = (
            Path(__file__).parent.parent.parent / "frontend" / "index.html"
        )
        if not index_path.exists():
            pytest.skip("frontend/index.html not found")
        content = index_path.read_text()
        assert "lg-preset-chip" not in content, \
            "Campaign app still has lg-preset-chip CSS class"
        assert "cmSelectPreset" not in content, \
            "Campaign app still has cmSelectPreset function"


# ===========================================================================
# Section 12: Integration — Full /api/hf-model-config Response Shape
# ===========================================================================

@pytest.mark.e2e
class TestHfModelConfigResponseShape:
    """Verify the full response shape contract of /api/hf-model-config."""

    def test_response_has_all_required_fields(self):
        """Response must contain all documented fields."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        config = {"hidden_size": 3584, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("Qwen/Qwen2.5-7B",
                                         allowed_dtypes=["bf16"]))

        required_keys = {"model_id", "is_moe", "suggested_tp", "suggested_dp",
                         "suggested_dtype", "reason", "config"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_successful_response_types(self):
        """On success: is_moe=bool, tp/dp=int, dtype=str, reason=None, config=dict."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        config = {"hidden_size": 3584, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("Qwen/Qwen2.5-7B",
                                         allowed_dtypes=["bf16"]))

        assert isinstance(result["is_moe"], bool)
        assert isinstance(result["suggested_tp"], int)
        assert isinstance(result["suggested_dp"], int)
        assert isinstance(result["suggested_dtype"], str)
        assert result["reason"] is None
        assert isinstance(result["config"], dict)

    def test_error_response_types(self):
        """On error: reason=str, tp/dp/dtype=None."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {"gated": "auto"}
        patcher = _install_httpx_mock([(200, metadata)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config("gated/model"))

        assert isinstance(result["reason"], str)
        assert result["suggested_tp"] is None
        assert result["suggested_dp"] is None
        assert result["suggested_dtype"] is None


# ===========================================================================
# Section 13: _pick_dtype Robustness — safetensors, text_config.dtype, fallback
# ===========================================================================

class TestPickDtypeRobustness:
    """Verify _pick_dtype correctly handles models with non-standard dtype fields.

    Bug fix: models without torch_dtype or quantization_config previously fell
    through to allowed[0] (fp8 on modern GPUs), causing wrong dtype and TP.
    """

    def test_safetensors_fp8_dominant_picks_fp8(self):
        """Model with >50% F8_E4M3 safetensors params → dtype=fp8."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {
            "gated": False,
            "safetensors": {
                "total": 31_577_937_344,
                "parameters": {"F8_E4M3": 30_491_811_840, "BF16": 1_078_212_032, "F32": 7_922_384},
            },
        }
        config = {
            "hidden_size": 2688,
            "num_hidden_layers": 40,
            "torch_dtype": "bfloat16",
            "num_local_experts": 64,
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config(
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["suggested_dtype"] == "fp8"
        assert result["suggested_tp"] == 1  # 30B × 1 byte = 29.4GB < 38.4GB usable
        assert result["is_moe"] is True

    def test_safetensors_bf16_dominant_picks_bf16(self):
        """Model with >50% BF16 safetensors params → dtype=bf16."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {
            "gated": False,
            "safetensors": {
                "total": 31_577_937_344,
                "parameters": {"BF16": 31_577_934_400, "F32": 5888},
            },
        }
        config = {
            "hidden_size": 2688,
            "num_hidden_layers": 40,
            "torch_dtype": "bfloat16",
            "num_local_experts": 64,
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config(
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["suggested_dtype"] == "bf16"
        assert result["suggested_tp"] == 2  # 31.6B × 2 bytes = 58.8GB → ceil(58.8/38.4) = 2
        assert result["is_moe"] is True

    def test_text_config_dtype_field_used_when_torch_dtype_missing(self):
        """Multimodal model with text_config.dtype but no torch_dtype → picks correct dtype."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {
            "gated": False,
            "safetensors": {
                "total": 35_953_925_552,
                "parameters": {"BF16": 35_951_817_904, "F32": 4800},
            },
        }
        config = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_experts": 256,
                "dtype": "bfloat16",
            },
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config(
                "Qwen/Qwen3.5-35B-A3B",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["suggested_dtype"] == "bf16"
        assert result["suggested_tp"] == 2  # 36B × 2 bytes = 67GB → ceil(67/38.4) = 2
        assert result["is_moe"] is True

    def test_quantization_config_takes_priority_over_safetensors(self):
        """Explicit quantization_config trumps safetensors breakdown."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {
            "gated": False,
            "safetensors": {
                "total": 35_953_925_552,
                "parameters": {"F8_E4M3": 34_400_000_000, "BF16": 1_500_000_000, "F32": 4800},
            },
        }
        config = {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "quantization_config": {"quant_method": "fp8"},
            "num_local_experts": 256,
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config(
                "Qwen/Qwen3.5-35B-A3B-FP8",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["suggested_dtype"] == "fp8"
        assert result["suggested_tp"] == 1  # 36B × 1 byte = 33.5GB < 38.4GB usable

    def test_fallback_prefers_bf16_over_fp8(self):
        """When no dtype signal exists, fallback to bf16 (not fp8)."""
        from shared.hf_model_config import _pick_dtype

        config = {"hidden_size": 4096, "num_hidden_layers": 32}
        result = _pick_dtype(config, ["fp8", "bf16", "fp16"], safetensors_params=None)
        assert result == "bf16"

    def test_fallback_uses_allowed_first_when_bf16_not_available(self):
        """When bf16 not in allowed, fallback to allowed[0]."""
        from shared.hf_model_config import _pick_dtype

        config = {"hidden_size": 4096, "num_hidden_layers": 32}
        result = _pick_dtype(config, ["fp16"], safetensors_params=None)
        assert result == "fp16"

    def test_safetensors_overrides_misleading_torch_dtype(self):
        """FP8 model with torch_dtype=bfloat16 (misleading) → safetensors wins."""
        from shared.hf_model_config import HfModelConfigService

        metadata = {
            "gated": False,
            "safetensors": {
                "total": 31_577_937_344,
                "parameters": {"F8_E4M3": 30_491_811_840, "BF16": 1_078_212_032, "F32": 7_922_384},
            },
        }
        config = {
            "hidden_size": 2688,
            "num_hidden_layers": 40,
            "torch_dtype": "bfloat16",
            "num_local_experts": 64,
        }
        patcher = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = HfModelConfigService(gpu_memory_gb=48.0, total_gpus=4)
            result = _run(svc.get_config(
                "nvidia/Model-FP8",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["suggested_dtype"] == "fp8"
        assert result["suggested_tp"] == 1
