# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for TP_OPTIONS and GPU_DTYPE_MAP constants.

SUPPORTED_MODELS / MOE_MODELS / MOE_TP_OPTIONS were removed as part of the
static-model-selector removal; tests covering those constants were deleted
along with them. TP_OPTIONS and GPU_DTYPE_MAP remain as reusable constants.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.constants import TP_OPTIONS, GPU_DTYPE_MAP


@pytest.mark.unit
class TestSupportedModels:
    """Tests for the constants still exported by shared/constants.py."""

    def test_tp_options(self):
        assert TP_OPTIONS == [1, 2, 4, 8]

    def test_gpu_dtype_map_has_common_gpus(self):
        for gpu in ["b300", "b200", "h100", "h200", "l40s", "a100"]:
            assert gpu in GPU_DTYPE_MAP, f"Missing GPU type: {gpu}"
