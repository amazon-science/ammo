# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Shared constants for the AMMO session service.
Central location for default values to ensure consistency across all APIs
"""
import os

# GPU allocation timeouts (seconds)
GPU_TIMEOUT_SESSION = 120         # Session creation (multi-GPU may need longer)

# ============================================================================
# AI CLI Session Service
# ============================================================================

# Session data directories
SESSION_DATA_DIR = os.environ.get("SESSION_DATA_DIR", "/data/sessions")
SESSION_REPOS_DIR = os.environ.get("SESSION_REPOS_DIR", "/data/repos")
SESSION_TEMPLATES_DIR = os.environ.get("SESSION_TEMPLATES_DIR", "/data/templates")

# S3 Storage for session persistence (enables cross-node session resume)
SESSION_S3_BUCKET = os.environ.get("SESSION_S3_BUCKET", None)
SESSION_S3_PREFIX = os.environ.get("SESSION_S3_PREFIX", "sessions")
SESSION_S3_TTL_DAYS = int(os.environ.get("SESSION_S3_TTL_DAYS", "30"))

# ============================================================================
# AMMO (Agentic Model-on-Machine Optimizer) — Parallelism & GPU dtype map
# ============================================================================
# NOTE: The static SUPPORTED_MODELS preset list was removed in favor of
# live HuggingFace search (`/api/hf-models`) + auto-detected TP/DP/dtype
# suggestions (`/api/hf-model-config/{model_id}`). TP_OPTIONS and
# GPU_DTYPE_MAP remain as reusable constants for the new endpoints and
# the /health route handler.

# Available TP (tensor parallelism) options
TP_OPTIONS = [1, 2, 4, 8]

# Allowed dtypes per GPU architecture
# Reference: https://docs.vllm.ai/en/latest/features/quantization.html
GPU_DTYPE_MAP = {
    "b300": ["fp4", "fp8", "bf16", "fp16"],  # Blackwell Ultra
    "b200": ["fp4", "fp8", "bf16", "fp16"],  # Blackwell - native FP4
    "h100": ["fp8", "fp4", "bf16", "fp16"],
    "h200": ["fp8", "fp4", "bf16", "fp16"],
    "l40s": ["fp8", "bf16", "fp16"],
    "l40": ["fp8", "bf16", "fp16"],
    "a100": ["bf16", "fp16", "int8"],
    "a10g": ["bf16", "fp16", "int8"],
    "unknown": ["bf16", "fp16"],  # Safe fallback
}

# Bytes per parameter for each dtype (used for memory-based TP calculation)
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp4": 0.5,
    "int8": 1.0,
    "int4": 0.5,
}
