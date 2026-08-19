# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for NCU/nsys capability retention when dropping to session_user.

NCU requires CAP_SYS_ADMIN and nsys requires CAP_SYS_PTRACE to access GPU
performance counters. When the server drops to session_user (UID 1000),
Linux clears ALL capabilities. The fix uses prctl(PR_SET_KEEPCAPS) before
setuid, then raises the required capabilities in the ambient set afterward.

This file tests:
- terminal_manager.py: tmux command uses capability-aware privilege drop
- Dockerfile: capsh availability or Python wrapper approach
- Capability constant correctness

The privilege drop lives only in terminal_manager.py, which writes the
drop_privs.py wrapper that tmux runs under.
"""
import os
import re
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Task 18: NCU Capability Retention in terminal_manager.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTerminalManagerCapabilityRetention:
    """terminal_manager.py tmux command must retain capabilities through uid transition."""

    def test_terminal_build_tmux_command_no_plain_su(self):
        """tmux command must NOT use bare 'su session_user' which clears capabilities."""
        content = (PROJECT_ROOT / "orchestration" / "terminal_manager.py").read_text()
        # We're looking for the _build_tmux_command method
        # It should NOT have a plain "su session_user -s /bin/bash -c" without
        # capability retention
        #
        # Acceptable alternatives:
        # 1. capsh --keep=1 --user=session_user --addamb=cap_sys_admin ...
        # 2. A Python wrapper script that does prctl+setuid+ambient
        #
        # We check that at least one capability-aware mechanism is present
        has_capsh = "capsh" in content
        has_prctl_wrapper = "drop_privs" in content or "PR_SET_KEEPCAPS" in content
        has_cap_retain = has_capsh or has_prctl_wrapper

        assert has_cap_retain, (
            "terminal_manager.py must use a capability-aware privilege drop "
            "(capsh or Python wrapper with prctl). Plain 'su session_user' "
            "clears all capabilities including CAP_SYS_ADMIN needed for ncu."
        )

    def test_terminal_build_tmux_command_retains_capabilities(self):
        """The command must retain CAP_SYS_ADMIN through the uid transition."""
        content = (PROJECT_ROOT / "orchestration" / "terminal_manager.py").read_text()
        # Check for either approach
        has_capsh_cap = "cap_sys_admin" in content.lower()
        has_prctl_cap = "CAP_SYS_ADMIN" in content
        assert has_capsh_cap or has_prctl_cap, (
            "terminal_manager.py must retain CAP_SYS_ADMIN "
            "(via capsh --addamb=cap_sys_admin or prctl wrapper)"
        )

    def test_terminal_session_user_uid_still_1000(self):
        """The resulting process must still run as UID 1000."""
        content = (PROJECT_ROOT / "orchestration" / "terminal_manager.py").read_text()
        # Whether capsh or wrapper, session_user (uid 1000) must be the target
        has_session_user_ref = "session_user" in content
        has_uid_1000 = "1000" in content
        assert has_session_user_ref and has_uid_1000, (
            "terminal_manager.py must drop to session_user (UID 1000)"
        )


# ---------------------------------------------------------------------------
# Task 20: Dockerfile capsh or Python wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDockerfileCapabilitySupport:
    """Dockerfile must support capability retention (capsh or Python wrapper)."""

    def test_dockerfile_has_libcap_or_wrapper(self):
        """Dockerfile installs libcap2-bin (provides capsh) OR uses Python wrapper."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        content = dockerfile.read_text()

        terminal_content = (PROJECT_ROOT / "orchestration" / "terminal_manager.py").read_text()

        # Check if capsh approach is used
        uses_capsh = "capsh" in terminal_content
        # Check if Python wrapper approach is used
        uses_wrapper = "drop_privs" in terminal_content or "PR_SET_KEEPCAPS" in terminal_content

        if uses_capsh:
            assert "libcap2-bin" in content or "libcap" in content, (
                "Dockerfile must install libcap2-bin when using capsh approach"
            )
        elif uses_wrapper:
            # Python wrapper approach doesn't need Dockerfile changes
            pass
        else:
            pytest.fail(
                "Neither capsh nor Python wrapper approach found in terminal_manager.py"
            )

    def test_dockerfile_ncu_installed(self):
        """ncu (NVIDIA Nsight Compute) must be available in the container."""
        # This is a regression guard — ncu comes with CUDA toolkit
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        content = dockerfile.read_text()
        # ncu comes with cuda-devel image, verify CUDA base image is used
        assert "nvidia/cuda" in content and "devel" in content, (
            "Dockerfile must use nvidia/cuda devel image which includes ncu"
        )
