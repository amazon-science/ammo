# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for detect_gpu_type() in app.py."""

import pytest
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app import detect_gpu_type


@pytest.mark.unit
class TestDetectGpuType:
    """Tests for detect_gpu_type() function."""

    def test_returns_env_var_when_set(self):
        """GPU_TYPE env var takes precedence over hardware detection."""
        with patch.dict('os.environ', {'GPU_TYPE': 'h200'}):
            assert detect_gpu_type() == 'h200'

    def test_returns_unknown_when_nvidia_smi_fails(self):
        """Returns 'unknown' when nvidia-smi is not available."""
        with patch.dict('os.environ', {}, clear=False):
            # Remove GPU_TYPE if present
            import os
            os.environ.pop('GPU_TYPE', None)
            with patch('app.subprocess.run', side_effect=FileNotFoundError("nvidia-smi not found")):
                assert detect_gpu_type() == 'unknown'

    def test_logs_warning_when_nvidia_smi_fails(self, caplog):
        """Must log a warning when nvidia-smi fails, not silently pass."""
        import os
        os.environ.pop('GPU_TYPE', None)
        with patch('app.subprocess.run', side_effect=FileNotFoundError("nvidia-smi not found")):
            with caplog.at_level(logging.WARNING, logger="app"):
                detect_gpu_type()

        assert any("nvidia-smi" in record.message.lower() or "gpu" in record.message.lower()
                    for record in caplog.records), \
            f"Expected warning log about nvidia-smi failure, got: {[r.message for r in caplog.records]}"

    def test_logs_warning_when_nvidia_smi_times_out(self, caplog):
        """Must log a warning when nvidia-smi times out."""
        import os
        os.environ.pop('GPU_TYPE', None)
        with patch('app.subprocess.run', side_effect=TimeoutError("timed out")):
            with caplog.at_level(logging.WARNING, logger="app"):
                detect_gpu_type()

        assert any(record.levelno >= logging.WARNING for record in caplog.records), \
            "Expected at least one WARNING log when nvidia-smi times out"

    def test_detects_gpu_from_nvidia_smi_output(self):
        """Correctly parses nvidia-smi output for known GPU types."""
        import os
        os.environ.pop('GPU_TYPE', None)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA H200 SXM\n"
        with patch('app.subprocess.run', return_value=mock_result):
            assert detect_gpu_type() == 'h200'

    def test_detects_b200_from_nvidia_smi_output(self):
        """Correctly identifies B200 GPU from nvidia-smi output."""
        import os
        os.environ.pop('GPU_TYPE', None)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA B200 SXM\n"
        with patch('app.subprocess.run', return_value=mock_result):
            assert detect_gpu_type() == 'b200'

    def test_env_var_override_b200(self):
        """GPU_TYPE=b200 env var works for B200."""
        with patch.dict('os.environ', {'GPU_TYPE': 'b200'}):
            assert detect_gpu_type() == 'b200'

    def test_detects_b300_from_nvidia_smi_output(self):
        """Correctly identifies B300 GPU from nvidia-smi output."""
        import os
        os.environ.pop('GPU_TYPE', None)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA B300 SXM\n"
        with patch('app.subprocess.run', return_value=mock_result):
            assert detect_gpu_type() == 'b300'

    def test_env_var_override_b300(self):
        """GPU_TYPE=b300 env var works for B300."""
        with patch.dict('os.environ', {'GPU_TYPE': 'b300'}):
            assert detect_gpu_type() == 'b300'
