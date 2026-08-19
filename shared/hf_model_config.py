# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
HuggingFace Model Config Service
================================

Fetches HuggingFace model metadata + architecture config and derives
auto-suggestions for tensor-parallelism (TP), data-parallelism (DP),
dtype, and MoE detection.

Call flow:
  1. GET https://huggingface.co/api/models/{id}?expand[]=safetensors&expand[]=gated&expand[]=tags&expand[]=config
     -> extract `gated`, `safetensors.total`, `tags`, `config`.
  2. If not gated: GET https://huggingface.co/{id}/resolve/main/config.json
     -> parse architecture fields (hidden_size, num_hidden_layers,
     num_local_experts / num_experts, etc.) for MoE detection and
     parameter estimation.

TP/DP heuristic:
  - dtype: extracted from config.json (quantization_config > torch_dtype > fallback)
  - TP: ceil(model_params × bytes_per_dtype / gpu_memory_gb), rounded up to
    power-of-2 in [1, 2, 4, 8], capped at available GPUs.
  - DP is always 1 (default).
  - GPU memory detected at runtime (torch.cuda or nvidia-smi).

Response (from `HfModelConfigService.get_config`) is a plain dict:
  {
    "model_id": str,
    "is_moe": bool,
    "suggested_tp": int | None,
    "suggested_dp": int | None,
    "suggested_dtype": str | None,
    "reason": str | None,   # "gated" | "network_error" | "config_missing_fields"
    "config": dict | None,  # raw architecture fields (forwarded), None if unavailable
  }
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

HF_API_BASE = "https://huggingface.co"
HF_API_TIMEOUT = 5.0  # seconds per HTTP call
HF_CONFIG_CACHE_TTL = 24 * 60 * 60  # 24 hours
HF_CONFIG_CACHE_MAX_SIZE = 500


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_MOE_NAME_PATTERN = re.compile(r"\d+[Bb]-[Aa]\d+[Bb]")


def _is_moe_by_name(model_id: str) -> bool:
    """Fallback MoE detection from model name pattern like '35B-A3B' (total-active)."""
    return bool(_MOE_NAME_PATTERN.search(model_id))


def _is_moe(config: Dict[str, Any]) -> bool:
    """Detect MoE via expert-count fields (covers HF, DeepSeek, Mixtral, multimodal variants)."""
    moe_keys = ("num_local_experts", "num_experts", "n_routed_experts", "num_experts_per_tok")
    for key in moe_keys:
        val = config.get(key)
        if isinstance(val, int) and val > 1:
            return True
    if config.get("enable_moe_block") is True:
        return True
    text_config = config.get("text_config", {})
    if isinstance(text_config, dict):
        for key in moe_keys:
            val = text_config.get(key)
            if isinstance(val, int) and val > 1:
                return True
        if text_config.get("enable_moe_block") is True:
            return True
    return False


def _estimate_params(safetensors_total: Optional[int], config: Dict[str, Any]) -> Optional[int]:
    """Estimate parameter count.

    Preferred: safetensors.total from HF API is the sum of parameter counts
    across all dtype buckets (already a count, NOT bytes).

    Fallback: use rough formula hidden_size * num_hidden_layers * 12.
    """
    if isinstance(safetensors_total, int) and safetensors_total > 0:
        return safetensors_total

    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    if isinstance(hidden, int) and isinstance(layers, int):
        return hidden * layers * 12
    text_config = config.get("text_config", {})
    if isinstance(text_config, dict):
        hidden = text_config.get("hidden_size")
        layers = text_config.get("num_hidden_layers")
        if isinstance(hidden, int) and isinstance(layers, int):
            return hidden * layers * 12
    return None


VLLM_WEIGHT_MEMORY_FRACTION = 0.80


def _compute_tp(params: Optional[int], dtype: str, gpu_memory_gb: float, total_gpus: int, is_moe: bool = False) -> int:
    """Compute TP from model memory requirement vs per-GPU memory.

    Uses 80% of GPU memory for weights — the rest is reserved for
    KV cache, activations, CUDA context, and torch allocator overhead.
    """
    if params is None:
        return min(total_gpus, 8) if is_moe else 1

    from shared.constants import BYTES_PER_PARAM
    bytes_per_param = BYTES_PER_PARAM.get(dtype, 2.0)
    model_memory_gb = (params * bytes_per_param) / (1024**3)
    usable_gpu_memory_gb = gpu_memory_gb * VLLM_WEIGHT_MEMORY_FRACTION
    min_gpus = math.ceil(model_memory_gb / usable_gpu_memory_gb) if usable_gpu_memory_gb > 0 else 1

    VALID_TP = [1, 2, 4, 8]
    tp = next((v for v in VALID_TP if v >= min_gpus), 8)

    if total_gpus > 0:
        tp = min(tp, total_gpus)
    return tp


def _pick_dtype(
    config: Dict[str, Any],
    allowed_dtypes: Optional[List[str]],
    safetensors_params: Optional[Dict[str, int]] = None,
) -> str:
    """Pick dtype from model config, constrained by GPU capabilities.

    Priority: quantization_config → safetensors dominant dtype → torch_dtype/dtype → fallback.
    """
    allowed = allowed_dtypes or ["bf16"]

    # 1. Check quantization_config (top-level and text_config)
    for cfg in (config, config.get("text_config", {})):
        if not isinstance(cfg, dict):
            continue
        qc = cfg.get("quantization_config")
        if not isinstance(qc, dict):
            continue
        method = qc.get("quant_method", "").lower()
        mapped = None
        if method == "fp8":
            mapped = "fp8"
        elif method in ("gptq", "awq"):
            bits = qc.get("bits", 4)
            mapped = "int4" if bits <= 4 else "int8"
        elif method in ("bnb", "bitsandbytes"):
            mapped = "int8"
        if mapped and mapped in allowed:
            return mapped

    # 1.5. Check safetensors parameter breakdown for dominant weight dtype
    if isinstance(safetensors_params, dict) and safetensors_params:
        total = sum(safetensors_params.values())
        if total > 0:
            st_dtype_map = {"F8_E4M3": "fp8", "F8_E5M2": "fp8", "BF16": "bf16", "F16": "fp16", "I8": "int8"}
            dominant = max(safetensors_params, key=safetensors_params.get)
            if safetensors_params[dominant] / total > 0.5:
                mapped = st_dtype_map.get(dominant)
                if mapped and mapped in allowed:
                    return mapped

    # 2. Check torch_dtype / dtype (top-level and text_config)
    torch_dtype_map = {"bfloat16": "bf16", "float16": "fp16", "float32": "bf16"}
    for cfg in (config, config.get("text_config", {})):
        if not isinstance(cfg, dict):
            continue
        td = cfg.get("torch_dtype") or cfg.get("dtype") or ""
        if isinstance(td, str) and td in torch_dtype_map:
            mapped = torch_dtype_map[td]
            if mapped in allowed:
                return mapped

    # 3. Fallback — prefer bf16 as safe default
    if "bf16" in allowed:
        return "bf16"
    return allowed[0] if allowed else "bf16"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class HfModelConfigService:
    """Fetches and caches HuggingFace model metadata + config.json.

    Per-server in-memory LRU cache with TTL. 2-call flow to HuggingFace
    (metadata + config.json). Returns graceful reason codes on failure.
    """

    def __init__(
        self,
        cache_ttl: float = HF_CONFIG_CACHE_TTL,
        cache_max_size: int = HF_CONFIG_CACHE_MAX_SIZE,
        timeout: float = HF_API_TIMEOUT,
        gpu_memory_gb: float = 48.0,
        total_gpus: int = 8,
    ) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = cache_ttl
        self._cache_max_size = cache_max_size
        self._timeout = timeout
        self._gpu_memory_gb = gpu_memory_gb
        self._total_gpus = total_gpus

    # -- cache ---------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        ts = self._cache_ts.get(key)
        if ts is None:
            return None
        if time.time() - ts > self._cache_ttl:
            # Expired.
            self._cache.pop(key, None)
            self._cache_ts.pop(key, None)
            return None
        return self._cache.get(key)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        # Evict oldest if full.
        if len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache_ts, key=self._cache_ts.get)
            self._cache.pop(oldest_key, None)
            self._cache_ts.pop(oldest_key, None)
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_ts.clear()

    # -- HTTP ----------------------------------------------------------------

    async def _fetch_metadata(
        self, client: httpx.AsyncClient, model_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch HF model metadata (expanded). Returns (payload, error_reason)."""
        url = f"{HF_API_BASE}/api/models/{model_id}"
        params = [
            ("expand[]", "safetensors"),
            ("expand[]", "gated"),
            ("expand[]", "tags"),
            ("expand[]", "config"),
        ]
        try:
            resp = await client.get(url, params=params)
        except httpx.HTTPError as e:
            logger.warning(f"HF metadata fetch failed for {model_id}: {e}")
            return None, "network_error"

        if resp.status_code == 404:
            return None, "network_error"
        if resp.status_code >= 400:
            logger.warning(
                f"HF metadata returned {resp.status_code} for {model_id}"
            )
            return None, "network_error"

        try:
            return resp.json(), None
        except Exception as e:
            logger.warning(f"HF metadata JSON decode failed for {model_id}: {e}")
            return None, "network_error"

    async def _fetch_config_json(
        self, client: httpx.AsyncClient, model_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch HF config.json for the model. Returns (config, error_reason)."""
        url = f"{HF_API_BASE}/{model_id}/resolve/main/config.json"
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            logger.warning(f"HF config.json fetch failed for {model_id}: {e}")
            return None, "network_error"

        if resp.status_code == 404:
            return None, "config_missing_fields"
        if resp.status_code >= 400:
            logger.warning(
                f"HF config.json returned {resp.status_code} for {model_id}"
            )
            return None, "network_error"

        try:
            return resp.json(), None
        except Exception as e:
            logger.warning(f"HF config.json JSON decode failed for {model_id}: {e}")
            return None, "config_missing_fields"

    # -- public --------------------------------------------------------------

    async def get_config(
        self,
        model_id: str,
        allowed_dtypes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the auto-suggestion bundle for `model_id`.

        `allowed_dtypes` is the host GPU's `GPU_DTYPE_MAP` entry; when given
        and fp8 is in the list, the suggested dtype will be fp8, else bf16.
        """
        cached = self._cache_get(model_id)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {
            "model_id": model_id,
            "is_moe": False,
            "suggested_tp": None,
            "suggested_dp": None,
            "suggested_dtype": None,
            "reason": None,
            "config": None,
        }

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            metadata, err = await self._fetch_metadata(client, model_id)
            if err is not None or metadata is None:
                result["reason"] = err or "network_error"
                self._cache_put(model_id, result)
                return result

            # Gated check
            gated = metadata.get("gated")
            if gated and gated not in (False, "false", 0):
                result["reason"] = "gated"
                self._cache_put(model_id, result)
                return result

            # Fetch config.json for architecture details
            config, cfg_err = await self._fetch_config_json(client, model_id)

        if cfg_err is not None or config is None:
            result["reason"] = cfg_err or "network_error"
            self._cache_put(model_id, result)
            return result

        # Architecture inspection
        safetensors_total = None
        st = metadata.get("safetensors")
        if isinstance(st, dict):
            safetensors_total = st.get("total")

        is_moe = _is_moe(config) or _is_moe_by_name(model_id)

        # Required-ish fields for dense param estimation. If we have neither
        # safetensors nor hidden_size, we can't bucket sensibly.
        text_cfg = config.get("text_config", {}) if isinstance(config.get("text_config"), dict) else {}
        has_hidden = isinstance(config.get("hidden_size"), int) or isinstance(text_cfg.get("hidden_size"), int)
        has_safetensors = isinstance(safetensors_total, int) and safetensors_total > 0
        if not is_moe and not has_hidden and not has_safetensors:
            result["reason"] = "config_missing_fields"
            result["config"] = config
            self._cache_put(model_id, result)
            return result

        params = _estimate_params(safetensors_total, config)
        st_params = st.get("parameters") if isinstance(st, dict) else None
        dtype = _pick_dtype(config, allowed_dtypes, safetensors_params=st_params)
        tp = _compute_tp(params, dtype, self._gpu_memory_gb, self._total_gpus, is_moe=is_moe)

        result.update(
            is_moe=is_moe,
            suggested_tp=tp,
            suggested_dp=1,
            suggested_dtype=dtype,
            reason=None,
            config=config,
        )

        self._cache_put(model_id, result)
        return result


# ---------------------------------------------------------------------------
# Module-level singleton (app.py imports this)
# ---------------------------------------------------------------------------

_default_service: Optional[HfModelConfigService] = None


def get_hf_model_config_service(
    gpu_memory_gb: float = 48.0,
    total_gpus: int = 8,
) -> HfModelConfigService:
    """Return the process-wide HF model config service (lazy init)."""
    global _default_service
    if _default_service is None:
        _default_service = HfModelConfigService(
            gpu_memory_gb=gpu_memory_gb,
            total_gpus=total_gpus,
        )
    return _default_service
