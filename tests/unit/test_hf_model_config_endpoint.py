# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for /api/hf-model-config/{model_id:path}.

Covers the 2-call HuggingFace flow (metadata + config.json), MoE detection,
hardware-aware TP computation, dtype extraction from config, gated-model
reason codes, network error handling, and LRU cache behavior.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int = 200, payload=None):
    """Build a synchronous-looking mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=payload or {})
    resp.raise_for_status = MagicMock()
    return resp


def _install_httpx_mock(responses):
    """Patch httpx.AsyncClient so GET returns queued responses in order."""
    call_log = []
    response_iter = iter(responses)

    async def fake_get(url, *args, **kwargs):
        call_log.append(url)
        try:
            status_code, payload = next(response_iter)
        except StopIteration:
            raise RuntimeError(f"Unexpected extra HTTP call to {url}")
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, type) and issubclass(payload, Exception):
            raise payload()
        return _mock_response(status_code=status_code, payload=payload)

    mock_client = AsyncMock()
    mock_client.get = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    patcher = patch("httpx.AsyncClient", return_value=mock_client)
    return patcher, call_log


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fresh_service(gpu_memory_gb: float = 48.0, total_gpus: int = 8):
    """Reload the service module and return a fresh HfModelConfigService."""
    from shared import hf_model_config as mod
    importlib.reload(mod)
    return mod.HfModelConfigService(
        cache_ttl=3600,
        cache_max_size=10,
        gpu_memory_gb=gpu_memory_gb,
        total_gpus=total_gpus,
    )


# ---------------------------------------------------------------------------
# Tests — hardware-aware TP computation
# Note: safetensors.total from HF API is a PARAMETER COUNT (not bytes).
# Memory = params × bytes_per_param. Usable GPU = gpu_mem × 0.80.
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHfModelConfigService:

    def test_moe_model_tp_memory_aware(self):
        """DeepSeek-R1 (671B params, fp8) on H100 (80GB).
        Memory = 671B × 1 byte = 671GB. Usable = 80×0.8 = 64GB.
        min_gpus = ceil(671/64) = 11 -> tp=8 (max valid).
        """
        metadata = {
            "id": "deepseek-ai/DeepSeek-R1",
            "gated": False,
            "safetensors": {"total": 671_000_000_000},  # 671B params
        }
        config = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_local_experts": 256,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=80.0, total_gpus=8)
            result = _run(svc.get_config(
                "deepseek-ai/DeepSeek-R1",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))

        assert result["model_id"] == "deepseek-ai/DeepSeek-R1"
        assert result["is_moe"] is True
        assert result["suggested_dtype"] == "fp8"
        assert result["suggested_tp"] == 8
        assert result["suggested_dp"] == 1
        assert result["reason"] is None

    def test_dense_7b_bf16_on_l40s_tp1(self):
        """Qwen2.5-7B (7B params, bf16) on L40S (48GB).
        Memory = 7B × 2 = 14GB. Usable = 48×0.8 = 38.4GB.
        min_gpus = ceil(14/38.4) = 1 -> tp=1.
        """
        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        config = {"hidden_size": 3584, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config(
                "Qwen/Qwen2.5-7B-Instruct",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))
        assert result["is_moe"] is False
        assert result["suggested_dtype"] == "bf16"
        assert result["suggested_tp"] == 1

    def test_70b_bf16_on_l40s_tp4(self):
        """Llama-70B (70B params, bf16) on L40S (48GB).
        Memory = 70B × 2 = 140GB. Usable = 48×0.8 = 38.4GB.
        min_gpus = ceil(140/38.4) = 4 -> tp=4.
        """
        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {"hidden_size": 8192, "num_hidden_layers": 80, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config(
                "meta-llama/Llama-3.3-70B",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))
        assert result["suggested_tp"] == 4

    def test_70b_bf16_on_h100_tp2(self):
        """Llama-70B (bf16) on H100 (80GB).
        Memory = 140GB. Usable = 80×0.8 = 64GB.
        min_gpus = ceil(140/64) = 3 -> tp=4 (next power of 2).
        """
        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {"hidden_size": 8192, "num_hidden_layers": 80, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=80.0, total_gpus=8)
            result = _run(svc.get_config(
                "meta-llama/Llama-3.3-70B",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))
        assert result["suggested_tp"] == 4

    def test_70b_bf16_on_h200_tp2(self):
        """Llama-70B (bf16) on H200 (141GB).
        Memory = 140GB. Usable = 141×0.8 = 112.8GB.
        min_gpus = ceil(140/112.8) = 2 -> tp=2.
        """
        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {"hidden_size": 8192, "num_hidden_layers": 80, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=141.0, total_gpus=8)
            result = _run(svc.get_config(
                "meta-llama/Llama-3.3-70B",
                allowed_dtypes=["fp8", "bf16", "fp16"],
            ))
        assert result["suggested_tp"] == 2

    def test_gptq_int4_model_dtype_and_tp(self):
        """GPTQ 4-bit 70B on L40S (48GB).
        dtype=int4 (from quant config). Memory = 70B × 0.5 = 35GB.
        Usable = 48×0.8 = 38.4GB. min_gpus = ceil(35/38.4) = 1 -> tp=1.
        """
        metadata = {"gated": False, "safetensors": {"total": 70_000_000_000}}
        config = {
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "quantization_config": {"quant_method": "gptq", "bits": 4},
        }
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=48.0, total_gpus=8)
            result = _run(svc.get_config(
                "x/gptq-70b",
                allowed_dtypes=["fp8", "bf16", "fp16", "int4"],
            ))
        assert result["suggested_dtype"] == "int4"
        assert result["suggested_tp"] == 1

    def test_tp_capped_at_total_gpus(self):
        """If only 4 GPUs available, cap TP at 4 even if model needs more."""
        metadata = {"gated": False, "safetensors": {"total": 671_000_000_000}}
        config = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_local_experts": 256,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=80.0, total_gpus=4)
            result = _run(svc.get_config(
                "deepseek-ai/DeepSeek-R1",
                allowed_dtypes=["fp8", "bf16"],
            ))
        assert result["suggested_tp"] == 4

    def test_moe_unknown_params_defaults_to_max_tp(self):
        """MoE detected but no safetensors and no hidden_size -> tp=total_gpus."""
        metadata = {"gated": False}
        config = {"num_local_experts": 64}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=80.0, total_gpus=8)
            result = _run(svc.get_config("x/moe-unknown", allowed_dtypes=["bf16"]))
        assert result["is_moe"] is True
        assert result["suggested_tp"] == 8

    # -- dtype extraction tests ---

    def test_dtype_from_quantization_config_fp8(self):
        """quantization_config.quant_method=fp8 -> dtype=fp8."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization_config": {"quant_method": "fp8"},
        }
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/fp8", allowed_dtypes=["fp8", "bf16"]))
        assert result["suggested_dtype"] == "fp8"

    def test_dtype_from_torch_dtype_bfloat16(self):
        """torch_dtype=bfloat16, no quantization -> bf16."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/bf16", allowed_dtypes=["fp8", "bf16"]))
        assert result["suggested_dtype"] == "bf16"

    def test_dtype_fallback_when_not_in_allowed(self):
        """torch_dtype=bfloat16 but allowed only fp16 -> fallback to fp16."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/y", allowed_dtypes=["fp16"]))
        assert result["suggested_dtype"] == "fp16"

    def test_dtype_from_text_config_quantization(self):
        """Multimodal: quantization in text_config."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "text_config": {"quantization_config": {"quant_method": "fp8"}},
        }
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/mm", allowed_dtypes=["fp8", "bf16"]))
        assert result["suggested_dtype"] == "fp8"

    # -- gated / error / cache tests ---

    def test_gated_model_returns_null_suggestions(self):
        """gated=true -> all suggestion fields None, reason='gated'."""
        metadata = {"gated": "auto"}
        patcher, call_log = _install_httpx_mock([(200, metadata)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config(
                "meta-llama/SecretModel", allowed_dtypes=["fp8", "bf16"]
            ))
        assert result["suggested_tp"] is None
        assert result["suggested_dp"] is None
        assert result["suggested_dtype"] is None
        assert result["reason"] == "gated"
        assert len(call_log) == 1

    def test_network_error_returns_reason(self):
        """httpx.ConnectError -> reason='network_error'."""
        patcher, _ = _install_httpx_mock([
            (None, httpx.ConnectError("boom"))
        ])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
        assert result["reason"] == "network_error"
        assert result["suggested_tp"] is None

    def test_missing_config_fields_returns_reason(self):
        """Metadata OK but config.json lacks hidden_size + no safetensors."""
        metadata = {"gated": False}
        config = {"num_hidden_layers": 32}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
        assert result["reason"] == "config_missing_fields"

    def test_cache_hit_does_not_call_hf_again(self):
        """Second call with same model_id must not re-hit the HF API."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        patcher, call_log = _install_httpx_mock([
            (200, metadata), (200, config),
        ])
        with patcher:
            svc = _fresh_service()
            r1 = _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
            r2 = _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
        assert r1 == r2
        assert len(call_log) == 2

    def test_cache_expires_after_ttl(self):
        """Backdating the cache timestamp forces a second HF call."""
        metadata = {"gated": False, "safetensors": {"total": 8_000_000_000}}
        config = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        patcher, call_log = _install_httpx_mock([
            (200, metadata), (200, config),
            (200, metadata), (200, config),
        ])
        with patcher:
            from shared.hf_model_config import HfModelConfigService
            svc = HfModelConfigService(cache_ttl=0.001, cache_max_size=10)
            _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
            svc._cache_ts["x/y"] -= 10
            _run(svc.get_config("x/y", allowed_dtypes=["bf16"]))
        assert len(call_log) == 4

    def test_moe_detection_via_name_pattern(self):
        """Name pattern '35B-A3B' triggers MoE detection."""
        metadata = {"gated": False, "safetensors": {"total": 35_000_000_000}}
        config = {"hidden_size": 4096, "num_hidden_layers": 40, "torch_dtype": "bfloat16"}
        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            svc = _fresh_service(gpu_memory_gb=48.0)
            result = _run(svc.get_config(
                "Qwen/Qwen3.5-35B-A3B", allowed_dtypes=["fp8", "bf16"]
            ))
        assert result["is_moe"] is True

    def test_hf_404_returns_network_error(self):
        """nonexistent model -> HF 404 -> reason='network_error'."""
        patcher, _ = _install_httpx_mock([(404, {})])
        with patcher:
            svc = _fresh_service()
            result = _run(svc.get_config("nonexistent/model-xyz", allowed_dtypes=["bf16"]))
        assert result["reason"] == "network_error"


# ---------------------------------------------------------------------------
# Tests — endpoint wiring (FastAPI route + auth prefix)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHfModelConfigEndpoint:

    def test_endpoint_returns_service_response(self):
        """The /api/hf-model-config/{model_id:path} route delegates to the service."""
        import importlib
        import app as app_module
        importlib.reload(app_module)
        app_module.gpu_type = "l40s"
        app_module.gpu_memory_gb = 48.0

        metadata = {"gated": False, "safetensors": {"total": 7_000_000_000}}
        config = {"hidden_size": 3584, "num_hidden_layers": 28, "torch_dtype": "bfloat16"}

        patcher, _ = _install_httpx_mock([(200, metadata), (200, config)])
        with patcher:
            from shared import hf_model_config as mod
            mod._default_service = None  # force re-init

            result = _run(app_module.get_hf_model_config(model_id="Qwen/Qwen2.5-7B-Instruct"))

        assert result["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert result["suggested_tp"] == 1
        assert result["suggested_dtype"] == "bf16"

    def test_endpoint_protected_by_api_key_prefix(self):
        """`/api/hf-model-config/...` is not in OPEN_EXACT_PATHS and matches the /api/ prefix."""
        import importlib
        import app as app_module
        importlib.reload(app_module)
        assert "/api/hf-model-config/meta-llama/Llama-3.1-8B" not in app_module.OPEN_EXACT_PATHS
        assert any(
            "/api/hf-model-config/meta-llama/Llama-3.1-8B".startswith(p)
            for p in app_module.PROTECTED_PATH_PREFIXES
        )
