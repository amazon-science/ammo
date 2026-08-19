# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for scripts/patch_cmake_presets.py.

Tests the detect_cuda_arch() function and patch_presets() function
for correctness, idempotency, and edge cases.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.patch_cmake_presets import detect_cuda_arch, patch_presets


# ============================================================================
# Sample CMakeUserPresets.json for testing
# ============================================================================

def _make_base_presets() -> dict:
    """Create a minimal CMakeUserPresets.json structure matching generate_cmake_presets.py output."""
    return {
        "version": 6,
        "configurePresets": [
            {
                "name": "release",
                "cacheVariables": {
                    "CMAKE_BUILD_TYPE": "Release",
                }
            }
        ],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
            }
        ],
    }


def _make_patched_presets(arch="8.9", nvcc_threads=16, jobs=4) -> dict:
    """Create a fully-patched CMakeUserPresets.json."""
    return {
        "version": 6,
        "configurePresets": [
            {
                "name": "release",
                "cacheVariables": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "TORCH_CUDA_ARCH_LIST": arch,
                    "NVCC_THREADS": str(nvcc_threads),
                    "CMAKE_JOB_POOLS": f"compile={jobs}",
                },
                "environment": {
                    "CCACHE_BASEDIR": "${sourceDir}",
                    "CCACHE_NOHASHDIR": "1",
                },
            }
        ],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
                "jobs": jobs,
            }
        ],
    }


# ============================================================================
# Tests for detect_cuda_arch()
# ============================================================================

@pytest.mark.unit
class TestDetectCudaArch:
    """Tests for GPU architecture auto-detection."""

    def test_detect_via_torch(self):
        """When PyTorch CUDA is available, detect arch from torch."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 9)

        with patch.dict('sys.modules', {'torch': mock_torch}):
            arch = detect_cuda_arch()

        assert arch == "8.9"

    def test_detect_via_nvidia_smi(self):
        """When PyTorch is unavailable, fall back to nvidia-smi."""
        # Make torch import fail
        with patch.dict('sys.modules', {'torch': None}):
            with patch('scripts.patch_cmake_presets.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="8.9\n",
                )
                arch = detect_cuda_arch()

        assert arch == "8.9"
        mock_run.assert_called_once()

    def test_detect_raises_when_no_gpu(self):
        """When neither torch nor nvidia-smi work, raise RuntimeError."""
        with patch.dict('sys.modules', {'torch': None}):
            with patch('scripts.patch_cmake_presets.subprocess.run', side_effect=FileNotFoundError):
                with pytest.raises(RuntimeError, match="Could not auto-detect"):
                    detect_cuda_arch()


# ============================================================================
# Tests for patch_presets()
# ============================================================================

@pytest.mark.unit
class TestPatchPresets:
    """Tests for CMakeUserPresets.json patching."""

    @pytest.fixture
    def presets_file(self, tmp_path):
        """Create a temporary CMakeUserPresets.json with base content."""
        p = tmp_path / "CMakeUserPresets.json"
        p.write_text(json.dumps(_make_base_presets(), indent=4))
        return p

    @pytest.fixture
    def patched_presets_file(self, tmp_path):
        """Create a temporary CMakeUserPresets.json that's already patched."""
        p = tmp_path / "CMakeUserPresets.json"
        p.write_text(json.dumps(_make_patched_presets(), indent=4))
        return p

    def test_patches_torch_cuda_arch_list(self, presets_file):
        """TORCH_CUDA_ARCH_LIST is set to the specified architecture."""
        changes = patch_presets(presets_file, cuda_arch="9.0", nvcc_threads=16, jobs=4)

        data = json.loads(presets_file.read_text())
        assert data["configurePresets"][0]["cacheVariables"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
        assert "TORCH_CUDA_ARCH_LIST" in changes

    def test_patches_nvcc_threads(self, presets_file):
        """NVCC_THREADS is set as a string."""
        changes = patch_presets(presets_file, cuda_arch="8.9", nvcc_threads=8, jobs=4)

        data = json.loads(presets_file.read_text())
        assert data["configurePresets"][0]["cacheVariables"]["NVCC_THREADS"] == "8"
        assert "NVCC_THREADS" in changes

    def test_patches_cmake_job_pools(self, presets_file):
        """CMAKE_JOB_POOLS is set to compile=N format."""
        changes = patch_presets(presets_file, cuda_arch="8.9", nvcc_threads=16, jobs=12)

        data = json.loads(presets_file.read_text())
        assert data["configurePresets"][0]["cacheVariables"]["CMAKE_JOB_POOLS"] == "compile=12"
        assert "CMAKE_JOB_POOLS" in changes

    def test_patches_build_preset_jobs(self, presets_file):
        """buildPresets[0].jobs is set to the specified job count."""
        changes = patch_presets(presets_file, cuda_arch="8.9", nvcc_threads=16, jobs=8)

        data = json.loads(presets_file.read_text())
        assert data["buildPresets"][0]["jobs"] == 8
        assert "buildPresets.jobs" in changes

    def test_patches_ccache_env_block(self, presets_file):
        """CCACHE_BASEDIR and CCACHE_NOHASHDIR are added to environment."""
        patch_presets(presets_file, cuda_arch="8.9", nvcc_threads=16, jobs=4)

        data = json.loads(presets_file.read_text())
        env = data["configurePresets"][0]["environment"]
        assert env["CCACHE_BASEDIR"] == "${sourceDir}"
        assert env["CCACHE_NOHASHDIR"] == "1"

    def test_idempotent_repatching(self, patched_presets_file):
        """Running patch_presets on an already-patched file returns no changes."""
        changes = patch_presets(patched_presets_file, cuda_arch="8.9", nvcc_threads=16, jobs=4)

        assert changes == {}  # No changes needed

        # Verify content is unchanged
        data = json.loads(patched_presets_file.read_text())
        assert data["configurePresets"][0]["cacheVariables"]["TORCH_CUDA_ARCH_LIST"] == "8.9"

    def test_repatching_with_different_arch(self, patched_presets_file):
        """Re-patching with a different arch updates TORCH_CUDA_ARCH_LIST."""
        changes = patch_presets(patched_presets_file, cuda_arch="9.0", nvcc_threads=16, jobs=4)

        assert "TORCH_CUDA_ARCH_LIST" in changes
        data = json.loads(patched_presets_file.read_text())
        assert data["configurePresets"][0]["cacheVariables"]["TORCH_CUDA_ARCH_LIST"] == "9.0"

    def test_creates_environment_block_if_missing(self, presets_file):
        """If environment block doesn't exist, it's created."""
        data = json.loads(presets_file.read_text())
        assert "environment" not in data["configurePresets"][0]

        patch_presets(presets_file, cuda_arch="8.9", nvcc_threads=16, jobs=4)

        data = json.loads(presets_file.read_text())
        assert "environment" in data["configurePresets"][0]
        assert data["configurePresets"][0]["environment"]["CCACHE_BASEDIR"] == "${sourceDir}"


# ============================================================================
# Tests for CLI main() behavior
# ============================================================================

@pytest.mark.unit
class TestPatchCMakePresetsCLI:
    """Tests for command-line interface behavior."""

    def test_patch_only_flag_skips_generate(self, tmp_path):
        """--patch-only skips running generate_cmake_presets.py."""
        presets_file = tmp_path / "CMakeUserPresets.json"
        presets_file.write_text(json.dumps(_make_base_presets(), indent=4))

        from scripts.patch_cmake_presets import main
        with patch('sys.argv', [
            'patch_cmake_presets.py',
            '--patch-only',
            '--cuda-arch', '8.9',
            '--presets-path', str(presets_file),
        ]):
            with patch('scripts.patch_cmake_presets.subprocess.run') as mock_run:
                main()
                # subprocess.run should NOT be called (no generate step)
                mock_run.assert_not_called()

    def test_without_patch_only_runs_generate(self, tmp_path):
        """Without --patch-only, generate_cmake_presets.py is run first."""
        presets_file = tmp_path / "CMakeUserPresets.json"
        presets_file.write_text(json.dumps(_make_base_presets(), indent=4))

        from scripts.patch_cmake_presets import main
        with patch('sys.argv', [
            'patch_cmake_presets.py',
            '--cuda-arch', '8.9',
            '--presets-path', str(presets_file),
        ]):
            with patch('scripts.patch_cmake_presets.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                main()
                # subprocess.run SHOULD be called for generate step
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "generate_cmake_presets.py" in call_args[1]

    def test_python_path_cli_arg_accepted(self, tmp_path):
        """--python-path argument is accepted and passed to patch_presets."""
        presets_file = tmp_path / "CMakeUserPresets.json"
        presets_file.write_text(json.dumps(_make_base_presets(), indent=4))

        from scripts.patch_cmake_presets import main
        python_path = "/data/sessions/abc123/venv/bin/python"
        with patch('sys.argv', [
            'patch_cmake_presets.py',
            '--patch-only',
            '--cuda-arch', '8.9',
            '--presets-path', str(presets_file),
            '--python-path', python_path,
        ]):
            with patch('scripts.patch_cmake_presets.patch_presets') as mock_patch:
                mock_patch.return_value = {}
                main()
                mock_patch.assert_called_once()
                call_kwargs = mock_patch.call_args
                # python_path must be passed through
                assert call_kwargs[1].get('python_path') == python_path or \
                       (len(call_kwargs[0]) > 4 and call_kwargs[0][4] == python_path), \
                    f"python_path not passed to patch_presets: {call_kwargs}"


# ============================================================================
# Tests for patch_presets() Python path patching (Bug 2)
# ============================================================================

def _make_presets_with_python_paths(old_venv: str = "/workspace/vllm/.venv") -> dict:
    """Create presets with Python-related cacheVariables using an old venv path."""
    return {
        "version": 6,
        "configurePresets": [
            {
                "name": "release",
                "cacheVariables": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "Python_ROOT_DIR": old_venv,
                    "Python3_ROOT_DIR": old_venv,
                    "Python_EXECUTABLE": f"{old_venv}/bin/python",
                    "Python3_EXECUTABLE": f"{old_venv}/bin/python",
                    "TORCH_CUDA_ARCH_LIST": "9.0",
                    "NVCC_THREADS": "16",
                    "CMAKE_JOB_POOLS": "compile=4",
                },
                "environment": {
                    "CCACHE_BASEDIR": "${sourceDir}",
                    "CCACHE_NOHASHDIR": "1",
                },
            }
        ],
        "buildPresets": [
            {
                "name": "release",
                "configurePreset": "release",
                "jobs": 4,
            }
        ],
    }


@pytest.mark.unit
class TestPatchPresetsWithPythonPath:
    """Tests for Python path patching in patch_presets() — Bug 2 fix."""

    @pytest.fixture
    def presets_with_python(self, tmp_path):
        """Presets file with old /workspace/vllm/.venv Python paths."""
        p = tmp_path / "CMakeUserPresets.json"
        p.write_text(json.dumps(_make_presets_with_python_paths(), indent=4))
        return p

    def test_patches_python3_executable(self, presets_with_python):
        """Python3_EXECUTABLE is updated to the new python path."""
        new_python = "/data/sessions/abc123/venv/bin/python"
        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=new_python)

        data = json.loads(presets_with_python.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        assert cv["Python3_EXECUTABLE"] == new_python

    def test_patches_python_executable(self, presets_with_python):
        """Python_EXECUTABLE is updated to the new python path."""
        new_python = "/data/sessions/abc123/venv/bin/python"
        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=new_python)

        data = json.loads(presets_with_python.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        assert cv["Python_EXECUTABLE"] == new_python

    def test_patches_python3_root_dir(self, presets_with_python):
        """Python3_ROOT_DIR is set to the venv root (parent of bin/)."""
        new_python = "/data/sessions/abc123/venv/bin/python"
        expected_root = "/data/sessions/abc123/venv"
        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=new_python)

        data = json.loads(presets_with_python.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        assert cv["Python3_ROOT_DIR"] == expected_root

    def test_patches_python_root_dir(self, presets_with_python):
        """Python_ROOT_DIR is set to the venv root (parent of bin/)."""
        new_python = "/data/sessions/abc123/venv/bin/python"
        expected_root = "/data/sessions/abc123/venv"
        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=new_python)

        data = json.loads(presets_with_python.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        assert cv["Python_ROOT_DIR"] == expected_root

    def test_python_patching_is_idempotent(self, presets_with_python):
        """Re-patching with the same python_path produces no Python-related changes."""
        new_python = "/data/sessions/abc123/venv/bin/python"
        # First patch
        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=new_python)
        # Second patch — should detect no changes to Python vars
        changes = patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                                python_path=new_python)

        python_changes = {k: v for k, v in changes.items() if "Python" in k or "python" in k}
        assert python_changes == {}, f"Unexpected Python changes on re-patch: {python_changes}"

    def test_none_python_path_leaves_vars_unchanged(self, presets_with_python):
        """When python_path=None, Python cacheVariables are not modified."""
        original = json.loads(presets_with_python.read_text())
        orig_exe = original["configurePresets"][0]["cacheVariables"]["Python3_EXECUTABLE"]

        patch_presets(presets_with_python, cuda_arch="9.0", nvcc_threads=16, jobs=4,
                      python_path=None)

        data = json.loads(presets_with_python.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        assert cv["Python3_EXECUTABLE"] == orig_exe, \
            "Python3_EXECUTABLE must not change when python_path=None"

    def test_python_key_with_unknown_suffix_not_modified(self, tmp_path):
        """Edge case: a cacheVariable key containing 'Python' but not ending in
        _EXECUTABLE or _ROOT_DIR (e.g., Python_FIND_VIRTUALENV) must not be touched."""
        presets = _make_presets_with_python_paths()
        # Inject a key with an unrecognised Python suffix
        presets["configurePresets"][0]["cacheVariables"]["Python_FIND_VIRTUALENV"] = "ONLY"
        p = tmp_path / "CMakeUserPresets.json"
        p.write_text(json.dumps(presets, indent=4))

        new_python = "/data/sessions/abc123/venv/bin/python"
        patch_presets(p, cuda_arch="9.0", nvcc_threads=16, jobs=4, python_path=new_python)

        data = json.loads(p.read_text())
        cv = data["configurePresets"][0]["cacheVariables"]
        # The unrecognised key must be left exactly as-is
        assert cv["Python_FIND_VIRTUALENV"] == "ONLY", (
            "patch_presets should only update *_EXECUTABLE and *_ROOT_DIR Python keys, "
            "not keys with other suffixes like Python_FIND_VIRTUALENV"
        )
