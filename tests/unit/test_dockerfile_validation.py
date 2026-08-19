# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for Dockerfile and worktree-remove-cleanup.sh correctness.

These are static analysis tests — they parse file contents to ensure
structural invariants are maintained.
"""
import pytest
from pathlib import Path

# Resolve project root relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_dockerfile_path() -> Path:
    p = PROJECT_ROOT / "Dockerfile"
    if not p.exists():
        pytest.skip("Dockerfile not found")
    return p


def _get_remove_hook_path() -> Path:
    p = PROJECT_ROOT / "ai_cli_session" / ".claude" / "hooks" / "worktree-remove-cleanup.sh"
    if not p.exists():
        pytest.skip("worktree-remove-cleanup.sh not found")
    return p


@pytest.mark.unit
class TestDockerfileTorchPin:
    """Dockerfile must not hard-pin torch version; vLLM's requirements drive the version."""

    def test_no_torch_version_pin_in_pytorch_install(self):
        """The PyTorch install block must NOT contain torch==<version>."""
        content = _get_dockerfile_path().read_text()
        # Ensure there is no torch==X.Y.Z anywhere in the file
        import re
        matches = re.findall(r'torch==\S+', content)
        assert matches == [], (
            f"Dockerfile pins a specific torch version: {matches}. "
            "Remove the pin so vLLM's own requirements/build.txt drives the torch version."
        )

    def test_no_hardcoded_cuda_index(self):
        """Dockerfile must NOT hardcode a --extra-index-url for PyTorch CUDA wheels.

        The Dockerfile uses bare 'uv pip install -e .' and lets uv resolve
        torch from PyPI defaults. Hardcoding a cu* index causes failures when
        PyTorch hasn't published wheels for that CUDA version yet.
        """
        content = _get_dockerfile_path().read_text()
        assert '--extra-index-url https://download.pytorch.org/whl/cu' not in content, (
            "Dockerfile must NOT hardcode --extra-index-url for PyTorch CUDA wheels. "
            "Use bare 'uv pip install -e .' — uv resolves torch from PyPI defaults."
        )

    def test_editable_install_present(self):
        """Dockerfile must have 'uv pip install -e' for the vLLM editable install."""
        content = _get_dockerfile_path().read_text()
        assert 'uv pip install -e' in content, (
            "Dockerfile must use 'uv pip install -e' for vLLM editable install."
        )


@pytest.mark.unit
class TestWorktreeRemoveCleanupHook:
    """worktree-remove-cleanup.sh must handle all branch naming patterns."""

    def test_session_branch_pattern_included(self):
        """Branch deletion condition must include worktree-* and ammo/* patterns."""
        content = _get_remove_hook_path().read_text()
        assert '"$BRANCH_NAME" == worktree-*' in content, (
            "Remove hook must delete worktree-* branches."
        )
        assert '"$BRANCH_NAME" == ammo/*' in content, (
            "Remove hook must delete ammo/* branches."
        )

    def test_original_patterns_still_present(self):
        """Original worktree-* and ammo/* patterns must still be present."""
        content = _get_remove_hook_path().read_text()
        assert '"$BRANCH_NAME" == worktree-*' in content, \
            "worktree-* pattern must still be present in remove hook"
        assert '"$BRANCH_NAME" == ammo/*' in content, \
            "ammo/* pattern must still be present in remove hook"

    def test_main_and_master_branches_not_deleteable(self):
        """The hook must never unconditionally delete main or master branches.

        Edge case: if BRANCH_NAME is 'main' or 'master', it must NOT match any
        of the deletion patterns (worktree-*, ammo/*).
        """
        import re
        content = _get_remove_hook_path().read_text()
        # Extract the if-condition that guards branch deletion
        # We look for the line containing the deletion guard
        deletion_guard_lines = [
            line.strip()
            for line in content.splitlines()
            if 'BRANCH_NAME' in line and ('worktree-*' in line or 'ammo/*' in line)
        ]
        assert deletion_guard_lines, "Could not find branch deletion guard in remove hook"
        guard = deletion_guard_lines[0]
        # Guard must NOT contain a bare match for 'main' or 'master'
        assert '"$BRANCH_NAME" == main' not in guard, \
            "Branch deletion guard must not match 'main' branch literally"
        assert '"$BRANCH_NAME" == master' not in guard, \
            "Branch deletion guard must not match 'master' branch literally"
        # The glob patterns worktree-*, ammo/* all require a prefix
        # that 'main' and 'master' do not have.
        assert not (
            'main'.startswith('worktree-') or
            'main'.startswith('ammo/')
        ), "Sanity: 'main' should not match any deletion pattern"


@pytest.mark.unit
class TestDockerfileNvidiaTmpPermissions:
    """Dockerfile must pre-create /tmp/nvidia with 1777 permissions.

    ncu creates /tmp/nvidia/ with drwxr-xr-x (755) owned by root.
    nsys then fails creating /tmp/nvidia/nsight_systems/ because session_user
    can't write to a 755 root-owned dir. Fix: pre-create with sticky+world-writable.
    """

    def test_tmp_nvidia_mkdir_exists(self):
        """Dockerfile must mkdir -p /tmp/nvidia."""
        content = _get_dockerfile_path().read_text()
        assert '/tmp/nvidia' in content, \
            "Dockerfile must create /tmp/nvidia for NVIDIA tool temp files"

    def test_tmp_nvidia_chmod_1777(self):
        """Dockerfile must chmod 1777 /tmp/nvidia (world-writable + sticky bit)."""
        content = _get_dockerfile_path().read_text()
        lines = content.splitlines()
        # Find lines that chmod /tmp/nvidia
        chmod_lines = [
            l for l in lines
            if 'chmod' in l and '/tmp/nvidia' in l
            and not l.strip().startswith('#')
        ]
        assert len(chmod_lines) >= 1, \
            "Dockerfile must chmod /tmp/nvidia for session_user access"
        # Must use 1777 (sticky bit) not just 777
        assert any('1777' in l and '/tmp/nvidia' in l for l in chmod_lines), \
            f"Dockerfile must use chmod 1777 (sticky bit) on /tmp/nvidia, got: {chmod_lines}"

    def test_tmp_nvidia_mkdir_in_run_block(self):
        """The /tmp/nvidia temp dir must be created via a mkdir in a RUN block."""
        content = _get_dockerfile_path().read_text()
        lines = content.splitlines()
        # The AMMO-only image no longer installs CuPy, so /tmp/nvidia is no
        # longer paired with /tmp/cupy_kernel_cache — just assert it is created.
        nvidia_idx = next(
            (i for i, l in enumerate(lines) if '/tmp/nvidia' in l and 'mkdir' in l),
            None,
        )
        assert nvidia_idx is not None, "Must have mkdir for /tmp/nvidia"


@pytest.mark.unit
class TestDockerfileJqInstalled:
    """Dockerfile must install jq for JSON processing in session hooks."""

    def test_jq_in_apt_install(self):
        """jq must be in the apt-get install list."""
        content = _get_dockerfile_path().read_text()
        # jq should appear in the system dependencies apt-get install block
        assert 'jq' in content, \
            "Dockerfile must install jq (needed for JSON processing in session hooks)"


@pytest.mark.unit
class TestALBIngressRoutes:
    """ALB ingress must route all frontend API paths."""

    def _get_ingress_path(self) -> Path:
        p = PROJECT_ROOT / "alb-sessions-ingress.yaml"
        if not p.exists():
            pytest.skip("alb-sessions-ingress.yaml not found")
        return p

    def test_hf_model_config_route_exists(self):
        """ALB ingress must route /api/hf-model-config so the frontend can
        fetch HuggingFace auto-detection for TP/DP/dtype. (Replaces the removed
        /api/supported-models and /api/moe-models rules.)"""
        content = self._get_ingress_path().read_text()
        assert '/api/hf-model-config' in content, (
            "ALB ingress must include /api/hf-model-config path "
            "(frontend calls this endpoint for HF-driven model selection)"
        )

    def test_removed_endpoints_not_routed(self):
        """ALB ingress must NOT route the deleted /api/supported-models and
        /api/moe-models paths — those endpoints were removed as part of the
        static-model-selector removal."""
        content = self._get_ingress_path().read_text()
        assert '/api/supported-models' not in content, (
            "ALB ingress must NOT route /api/supported-models — the endpoint "
            "was removed; /health exposes GPU + vllm info and "
            "/api/hf-model-config replaces preset fetching."
        )
        assert '/api/moe-models' not in content, (
            "ALB ingress must NOT route /api/moe-models — the endpoint was "
            "removed; model selection now runs entirely through "
            "/api/hf-models + /api/hf-model-config."
        )
