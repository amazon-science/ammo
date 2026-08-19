# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for filesystem permission isolation of session terminals.

Session terminal processes must run as non-root (session_user, UID 1000) and
must not be able to read server source code or AMMO skill templates. These are
static analysis tests that parse Dockerfile content and source code to verify
the security invariants are maintained.
"""
import re

import pytest
from pathlib import Path

# Resolve project root relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_dockerfile_path() -> Path:
    p = PROJECT_ROOT / "Dockerfile"
    if not p.exists():
        pytest.skip("Dockerfile not found")
    return p


def _get_dockerfile_optimized_path() -> Path:
    p = PROJECT_ROOT / "Dockerfile.optimized"
    if not p.exists():
        pytest.skip("Dockerfile.optimized not found")
    return p


# ---------------------------------------------------------------------------
# TestDockerfileSessionUser: Static analysis of Dockerfile content
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDockerfileSessionUser:
    """Dockerfile must create session_user and apply security permissions."""

    def test_useradd_session_user(self):
        """Dockerfile must contain useradd to create session_user."""
        content = _get_dockerfile_path().read_text()
        assert "useradd" in content and "session_user" in content, (
            "Dockerfile must create a session_user account via useradd"
        )

    def test_chmod_server_code_not_world_readable(self):
        """Server code must land in the image without world permissions.

        Two accepted mechanisms: the legacy post-copy `chmod -R o-rwx`, or
        `COPY --chmod=NN0 ... /app...` (commit 002a5e3), which sets the same
        other-bits-zero guarantee in the copy layer itself.
        """
        content = _get_dockerfile_path().read_text()
        legacy = "chmod -R o-rwx" in content
        copy_chmod = re.search(
            r"COPY\s+--chmod=(\d{3,4})\s+\S+\s+/app\S*", content
        )
        assert legacy or copy_chmod, (
            "Dockerfile must restrict world permissions on server code with "
            "chmod -R o-rwx or COPY --chmod"
        )
        if not legacy:
            assert copy_chmod.group(1)[-1] == "0", (
                "COPY --chmod on server code must zero the world-permission "
                f"digit, got --chmod={copy_chmod.group(1)}"
            )

    def test_gpu_lock_wrapper_copied_to_lib(self):
        """Dockerfile must copy gpu_lock_wrapper.py to /usr/local/lib/ammo/."""
        content = _get_dockerfile_path().read_text()
        assert "/usr/local/lib/ammo" in content, (
            "Dockerfile must create /usr/local/lib/ammo/ directory"
        )
        assert "gpu_lock_wrapper.py" in content, (
            "Dockerfile must copy gpu_lock_wrapper.py"
        )

    def test_gpu_lock_wrapper_chmod_755(self):
        """Dockerfile must chmod 755 on the copied gpu_lock_wrapper."""
        content = _get_dockerfile_path().read_text()
        assert "chmod 755" in content and "gpu_lock_wrapper" in content, (
            "Dockerfile must set chmod 755 on the copied gpu_lock_wrapper.py"
        )

    def test_session_dirs_chown(self):
        """Dockerfile must chown session_user:session_user on /data/sessions and /data/repos."""
        content = _get_dockerfile_path().read_text()
        assert "chown" in content and "session_user" in content, (
            "Dockerfile must chown session data directories to session_user"
        )
        assert "/data/sessions" in content, (
            "chown must include /data/sessions"
        )
        assert "/data/repos" in content, (
            "chown must include /data/repos"
        )

    def test_dockerfile_optimized_has_same_security(self):
        """Dockerfile.optimized must also have the session_user security additions."""
        content = _get_dockerfile_optimized_path().read_text()
        assert "useradd" in content and "session_user" in content, (
            "Dockerfile.optimized must create session_user"
        )
        assert "chmod -R o-rwx" in content, (
            "Dockerfile.optimized must restrict server code permissions"
        )
        assert "/usr/local/lib/ammo" in content and "gpu_lock_wrapper" in content, (
            "Dockerfile.optimized must copy gpu_lock_wrapper to /usr/local/lib/ammo/"
        )


# ---------------------------------------------------------------------------
# TestGpuLockWrapperNewPath: Verify wrapper path references
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGpuLockWrapperNewPath:
    """GPU lock wrapper must be accessible from the new isolated path."""

    def test_source_gpu_lock_wrapper_exists(self):
        """The source shared/gpu_lock_wrapper.py file must exist."""
        wrapper_path = PROJECT_ROOT / "shared" / "gpu_lock_wrapper.py"
        assert wrapper_path.exists(), (
            f"shared/gpu_lock_wrapper.py not found at {wrapper_path}"
        )

    def test_gpu_lock_wrapper_lib_path_is_world_readable(self):
        """The Dockerfile must chmod the wrapper world-readable at /usr/local/lib/ammo/."""
        content = _get_dockerfile_path().read_text()
        assert "chmod 755 /usr/local/lib/ammo/gpu_lock_wrapper.py" in content, (
            "Dockerfile must chmod 755 the wrapper so session_user can read it"
        )


# ---------------------------------------------------------------------------
# TestTerminalManagerTtydUid: ttyd must run as session_user
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestTerminalManagerTtydUid:
    """Terminal manager must launch ttyd with --uid 1000 so terminals run as session_user."""

    def test_terminal_manager_ttyd_runs_as_session_user(self):
        """ttyd must be launched with --uid 1000 so terminals run as session_user."""
        content = (PROJECT_ROOT / "orchestration" / "terminal_manager.py").read_text()
        assert '"--uid"' in content or "'--uid'" in content, \
            "ttyd command must include --uid flag to run as session_user"
