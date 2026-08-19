# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for server-side AMMO_GPU_RES_DIR injection (TDD red phase).

These tests define the expected behavior BEFORE implementation exists.
They will FAIL until Changes 2 and 5 from the gpu-reservation-integration plan
are implemented in session_manager.py and cli_tool_manager.py.

Plan: .claude/plans/gpu-reservation-integration.md

Tests:
  1. create_session injects AMMO_GPU_RES_DIR into extra_env (ttyd process env)
  2. resume_session injects AMMO_GPU_RES_DIR into extra_env (ttyd process env)
  3. setup_claude_workspace writes AMMO_GPU_RES_DIR to settings.local.json env block
  4. terminate_session cleans up /tmp/ammo_gpu_res_{session_id}/
  5. terminate_session with missing dir -> no error (idempotent cleanup)
"""

import json
import os
import shutil
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import (
    reset_all_singletons,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
    mock_session_manager,
    make_session_state,
    make_create_request,
)
from shared.session_models import CLIToolType, SessionStatus


# =============================================================================
# Test 1: create_session injects AMMO_GPU_RES_DIR into ttyd process env
# =============================================================================

@pytest.mark.unit
class TestCreateSessionInjectsAMMOGPUResDir:
    """
    Change 2b (create path): create_session must add
    AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id} to the extra_env dict
    that is passed as the `env` kwarg to start_terminal_with_command.
    """

    @pytest.mark.asyncio
    async def test_create_session_injects_ammo_gpu_res_dir_into_ttyd_env(
        self, mock_session_manager, tmp_path
    ):
        """
        After create_session, the env dict passed to start_terminal_with_command
        must contain AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id}.

        This validates Change 2b (create path) from the plan:
          extra_env["AMMO_GPU_RES_DIR"] = f"/tmp/ammo_gpu_res_{session_id}"
        injected in session_manager.py after the HF_HOME line.
        """
        # Use a tmp_path-based session dir so mkdir calls succeed in isolation
        session_dir = tmp_path / "test-session"
        mock_session_manager.worktree_manager.get_session_dir.return_value = session_dir

        request = make_create_request(gpu_count=0)
        response = await mock_session_manager.create_session(request)
        session_id = response.session_id

        # Verify the terminal was started
        terminal_mock = mock_session_manager.terminal_manager
        terminal_mock.start_terminal_with_command.assert_called_once()

        # Extract env dict passed to start_terminal_with_command
        call_args = terminal_mock.start_terminal_with_command.call_args
        env_passed = call_args.kwargs.get("env") or (call_args[1].get("env") if call_args[1] else {})

        expected_dir = f"/tmp/ammo_gpu_res_{session_id}"
        assert "AMMO_GPU_RES_DIR" in env_passed, (
            f"AMMO_GPU_RES_DIR missing from ttyd env on create_session. "
            f"Keys present: {sorted(env_passed.keys())}"
        )
        assert env_passed["AMMO_GPU_RES_DIR"] == expected_dir, (
            f"Expected AMMO_GPU_RES_DIR={expected_dir!r}, "
            f"got {env_passed.get('AMMO_GPU_RES_DIR')!r}"
        )


# =============================================================================
# Test 2: resume_session injects AMMO_GPU_RES_DIR into ttyd process env
# =============================================================================

@pytest.mark.unit
class TestResumeSessionInjectsAMMOGPUResDir:
    """
    Change 2b (resume path): resume_session must add
    AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id} to extra_env.
    """

    @pytest.mark.asyncio
    async def test_resume_session_injects_ammo_gpu_res_dir_into_ttyd_env(
        self, mock_session_manager, tmp_path
    ):
        """
        After resume_session, the env dict passed to start_terminal_with_command
        must contain AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id}.

        This validates Change 2b (resume path) from the plan:
          extra_env["AMMO_GPU_RES_DIR"] = f"/tmp/ammo_gpu_res_{session_id}"
        injected in session_manager.py resume_session() after the HF_HOME line.
        """
        session_id = "resume-ammo-gpu-res-test"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.terminal_manager.is_available.return_value = True
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )

        await mock_session_manager.resume_session(session_id)

        terminal_mock = mock_session_manager.terminal_manager
        terminal_mock.start_terminal_with_command.assert_called_once()

        call_args = terminal_mock.start_terminal_with_command.call_args
        env_passed = call_args.kwargs.get("env") or (call_args[1].get("env") if call_args[1] else {})

        expected_dir = f"/tmp/ammo_gpu_res_{session_id}"
        assert "AMMO_GPU_RES_DIR" in env_passed, (
            f"AMMO_GPU_RES_DIR missing from ttyd env on resume_session. "
            f"Keys present: {sorted(env_passed.keys())}"
        )
        assert env_passed["AMMO_GPU_RES_DIR"] == expected_dir, (
            f"Expected AMMO_GPU_RES_DIR={expected_dir!r}, "
            f"got {env_passed.get('AMMO_GPU_RES_DIR')!r}"
        )


# =============================================================================
# Test 3: setup_claude_workspace writes AMMO_GPU_RES_DIR to settings.local.json
# =============================================================================

@pytest.mark.unit
class TestSetupClaudeWorkspaceInjectsAMMOGPUResDir:
    """
    Change 2a: setup_claude_workspace must write AMMO_GPU_RES_DIR to
    settings.local.json's "env" section so Claude Code subagents (Task tool)
    inherit it.
    """

    @pytest.fixture
    def temp_worktree(self):
        """Create a temporary worktree directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_setup_claude_workspace_writes_ammo_gpu_res_dir_to_settings_local(
        self, temp_worktree
    ):
        """
        After setup_claude_workspace, settings.local.json env section must
        contain AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id}.

        This validates Change 2a from the plan:
          env["AMMO_GPU_RES_DIR"] = f"/tmp/ammo_gpu_res_{session_id}"
        injected in cli_tool_manager.py setup_claude_workspace() after
        the CUDA_VISIBLE_DEVICES injection (line ~351).
        """
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "workspace-ammo-test-session"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[4, 5],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists(), (
            "settings.local.json must exist after setup_claude_workspace"
        )

        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        expected_dir = f"/tmp/ammo_gpu_res_{session_id}"

        assert "AMMO_GPU_RES_DIR" in env_section, (
            f"AMMO_GPU_RES_DIR missing from settings.local.json env section. "
            f"Keys present: {sorted(env_section.keys())}"
        )
        assert env_section["AMMO_GPU_RES_DIR"] == expected_dir, (
            f"Expected AMMO_GPU_RES_DIR={expected_dir!r}, "
            f"got {env_section.get('AMMO_GPU_RES_DIR')!r}"
        )

    def test_setup_claude_workspace_ammo_gpu_res_dir_does_not_clobber_cuda(
        self, temp_worktree
    ):
        """
        Injecting AMMO_GPU_RES_DIR must not remove CUDA_VISIBLE_DEVICES.
        Both must coexist in the settings.local.json env section.
        """
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "workspace-coexist-test"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[2, 3],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        assert "CUDA_VISIBLE_DEVICES" in env_section, (
            "CUDA_VISIBLE_DEVICES must still be present after AMMO_GPU_RES_DIR injection"
        )
        assert env_section["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert "AMMO_GPU_RES_DIR" in env_section, (
            "AMMO_GPU_RES_DIR must be present alongside CUDA_VISIBLE_DEVICES"
        )

    def test_setup_claude_workspace_zero_gpus_still_injects_ammo_gpu_res_dir(
        self, temp_worktree
    ):
        """
        Even when gpu_ids=[] (CPU-only session, CUDA_VISIBLE_DEVICES=-1),
        AMMO_GPU_RES_DIR must still be written to settings.local.json.

        The reservation pool will be empty for this session, but the env var
        must be set so Claude Code subagents can call gpu_reservation.py
        (which will return ReservationError immediately) rather than falling
        back to the sha256 path, which would silently use a shared dir.

        Edge case: gpu_ids=[] is not covered by the two non-empty-GPU tests above.
        """
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "zero-gpu-session-ammo-test"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists(), (
            "settings.local.json must exist after setup_claude_workspace"
        )

        with open(settings_local_path) as f:
            settings = json.load(f)

        env_section = settings.get("env", {})
        expected_dir = f"/tmp/ammo_gpu_res_{session_id}"

        # CUDA_VISIBLE_DEVICES should be "" (empty string) for zero-GPU sessions
        # (setup_claude_workspace uses "" not "-1" for the settings.local.json path)
        assert "CUDA_VISIBLE_DEVICES" in env_section, (
            "CUDA_VISIBLE_DEVICES key must be present even for zero-GPU sessions"
        )
        assert env_section["CUDA_VISIBLE_DEVICES"] == "", (
            f"Expected CUDA_VISIBLE_DEVICES='' for zero-GPU session, "
            f"got {env_section.get('CUDA_VISIBLE_DEVICES')!r}"
        )

        assert "AMMO_GPU_RES_DIR" in env_section, (
            f"AMMO_GPU_RES_DIR must be present even for zero-GPU sessions. "
            f"Keys present: {sorted(env_section.keys())}"
        )
        assert env_section["AMMO_GPU_RES_DIR"] == expected_dir, (
            f"Expected AMMO_GPU_RES_DIR={expected_dir!r}, "
            f"got {env_section.get('AMMO_GPU_RES_DIR')!r}"
        )


# =============================================================================
# Test 4 & 5: terminate_session cleans up /tmp/ammo_gpu_res_{session_id}/
# =============================================================================

@pytest.mark.unit
class TestTerminateSessionCleansUpGPUResDir:
    """
    Change 5: terminate_session must delete /tmp/ammo_gpu_res_{session_id}/
    if it exists, preventing /tmp/ disk accumulation in long-lived containers.
    """

    @pytest.mark.asyncio
    async def test_terminate_session_cleans_up_existing_gpu_res_dir(
        self, mock_session_manager, tmp_path
    ):
        """
        When /tmp/ammo_gpu_res_{session_id}/ exists at termination time,
        terminate_session must delete it.

        This validates Change 5 from the plan:
          gpu_res_dir = f"/tmp/ammo_gpu_res_{session_id}"
          if os.path.exists(gpu_res_dir):
              shutil.rmtree(gpu_res_dir, ignore_errors=True)
        """
        session_id = f"terminate-cleanup-{tmp_path.name}"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        # Pre-create the GPU reservation dir with a state file
        gpu_res_dir = Path(f"/tmp/ammo_gpu_res_{session_id}")
        gpu_res_dir.mkdir(parents=True, exist_ok=True)
        (gpu_res_dir / "state.json").write_text('{"reservations": []}')
        assert gpu_res_dir.exists(), "Setup: gpu_res_dir should exist before termination"

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        try:
            await mock_session_manager.terminate_session(session_id)

            assert not gpu_res_dir.exists(), (
                f"/tmp/ammo_gpu_res_{session_id}/ should have been deleted on termination, "
                f"but it still exists"
            )
        finally:
            # Safety cleanup so a test failure doesn't leak the dir
            if gpu_res_dir.exists():
                shutil.rmtree(gpu_res_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_terminate_session_missing_gpu_res_dir_no_error(
        self, mock_session_manager, tmp_path
    ):
        """
        When /tmp/ammo_gpu_res_{session_id}/ does NOT exist, terminate_session
        must complete without raising.

        This is the common case: a session that was paused without running any
        GPU workloads, or where the dir was already cleaned up.
        """
        session_id = f"terminate-no-dir-{tmp_path.name}"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        # Ensure the GPU reservation dir does NOT exist
        gpu_res_dir = Path(f"/tmp/ammo_gpu_res_{session_id}")
        if gpu_res_dir.exists():
            shutil.rmtree(gpu_res_dir)
        assert not gpu_res_dir.exists(), "Setup: gpu_res_dir must not exist for this test"

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        # Must not raise even though the dir doesn't exist
        await mock_session_manager.terminate_session(session_id)

    @pytest.mark.asyncio
    async def test_terminate_session_cleans_up_nested_gpu_res_dir(
        self, mock_session_manager, tmp_path
    ):
        """
        When /tmp/ammo_gpu_res_{session_id}/ contains subdirectories (e.g.,
        audit logs, worktree scratch), terminate_session must delete them all
        recursively — not just the top-level dir.

        This validates that shutil.rmtree (not os.rmdir) is used, since
        os.rmdir fails on non-empty directories.

        Edge case: existing tests only create a flat state.json; this tests
        a realistic nested structure left by gpu_reservation.py during agent use.
        """
        session_id = f"terminate-nested-{tmp_path.name}"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        # Pre-create the GPU reservation dir with nested structure
        gpu_res_dir = Path(f"/tmp/ammo_gpu_res_{session_id}")
        gpu_res_dir.mkdir(parents=True, exist_ok=True)
        (gpu_res_dir / "state.json").write_text('{"gpus": {}, "audit": []}')
        (gpu_res_dir / "state.lock").write_text("")  # lock file
        audit_dir = gpu_res_dir / "audit"
        audit_dir.mkdir()
        (audit_dir / "session-log.json").write_text("[]")
        assert gpu_res_dir.exists(), "Setup: gpu_res_dir should exist before termination"
        assert (audit_dir / "session-log.json").exists(), "Setup: nested file should exist"

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            requested_gpu_count=0,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        try:
            await mock_session_manager.terminate_session(session_id)

            assert not gpu_res_dir.exists(), (
                f"/tmp/ammo_gpu_res_{session_id}/ should have been deleted "
                f"recursively (including nested audit/ subdir), but it still exists"
            )
        finally:
            if gpu_res_dir.exists():
                shutil.rmtree(gpu_res_dir, ignore_errors=True)


# =============================================================================
# T2: setup_claude_workspace writes AMMO_TP_SIZE / AMMO_DP_SIZE to settings.local.json
# =============================================================================

@pytest.mark.unit
class TestSetupClaudeWorkspaceInjectsAMMOTpDp:
    """
    Plan T2 (.claude/plans/gpu-tp-dp-awareness.md):
      setup_claude_workspace must accept optional tp_size / dp_size kwargs and
      inject AMMO_TP_SIZE / AMMO_DP_SIZE into settings.local.json's `env` block
      when they are provided. That is what lets Claude Code subagents (Task
      tool) discover parallelism programmatically without reopening
      target.json.

    The key behavioral invariants:
      - When tp_size > 0 and dp_size > 0, env keys are present as strings.
      - When tp_size / dp_size are None (legacy callers), the keys must be
        absent — an empty string would masquerade as "explicitly 1" and break
        downstream `if "${AMMO_TP_SIZE:-}" -eq 1` shell checks.
      - Injecting TP/DP must not clobber CUDA_VISIBLE_DEVICES, AMMO_GPU_RES_DIR,
        or CLAUDE_PROJECT_DIR — all four env keys coexist in one settings.local.
    """

    @pytest.fixture
    def temp_worktree(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_settings_local_contains_ammo_tp_size(self, temp_worktree):
        """tp_size=2, dp_size=2 → settings.local.json env has both keys."""
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "tp-dp-inject-test"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[0, 1, 2, 3],
            tp_size=2,
            dp_size=2,
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists()

        with open(settings_local_path) as f:
            settings = json.load(f)
        env_section = settings.get("env", {})

        assert env_section.get("AMMO_TP_SIZE") == "2", (
            f"Expected AMMO_TP_SIZE='2', got {env_section.get('AMMO_TP_SIZE')!r}"
        )
        assert env_section.get("AMMO_DP_SIZE") == "2", (
            f"Expected AMMO_DP_SIZE='2', got {env_section.get('AMMO_DP_SIZE')!r}"
        )

    def test_settings_local_defaults_when_tp_dp_unknown(self, temp_worktree):
        """Legacy callers (no tp_size/dp_size kwargs) must NOT get TP/DP env keys.

        Downstream shell guards like `if [ "${AMMO_TP_SIZE:-}" = "" ]` rely on
        absence to detect "parallelism unknown". Writing "" would masquerade
        as "explicitly 1" and flip the branch.
        """
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "legacy-no-tp-dp-test"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[0],
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)
        env_section = settings.get("env", {})

        assert "AMMO_TP_SIZE" not in env_section, (
            f"AMMO_TP_SIZE must be absent for legacy callers; "
            f"found value {env_section.get('AMMO_TP_SIZE')!r}"
        )
        assert "AMMO_DP_SIZE" not in env_section, (
            f"AMMO_DP_SIZE must be absent for legacy callers; "
            f"found value {env_section.get('AMMO_DP_SIZE')!r}"
        )

    def test_setup_does_not_clobber_cuda_or_gpu_res_dir(self, temp_worktree):
        """Regression: injecting TP/DP leaves CUDA_VISIBLE_DEVICES,
        AMMO_GPU_RES_DIR, CLAUDE_PROJECT_DIR intact."""
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "coexist-tp-dp-test"
        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=[4, 5, 6, 7],
            tp_size=2,
            dp_size=2,
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        with open(settings_local_path) as f:
            settings = json.load(f)
        env_section = settings.get("env", {})

        # All five env keys must coexist.
        assert env_section.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
        assert env_section.get("AMMO_GPU_RES_DIR") == f"/tmp/ammo_gpu_res_{session_id}"
        assert env_section.get("CLAUDE_PROJECT_DIR") == str(temp_worktree)
        assert env_section.get("AMMO_TP_SIZE") == "2"
        assert env_section.get("AMMO_DP_SIZE") == "2"


# =============================================================================
# T2b: setup_codex_workspace provisions the .codex/ template tree
# =============================================================================

@pytest.mark.unit
class TestSetupCodexWorkspaceProvisionsTemplate:
    """Codex sessions read AMMO_TP_SIZE / AMMO_DP_SIZE from the ttyd env.

    The workspace setup only provisions the .codex/ template tree; TP/DP reach
    Codex agents through the process env (see T3), not through a file.
    """

    @pytest.fixture
    def temp_worktree(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_codex_dir_and_agents_md_created(self, temp_worktree):
        from orchestration.cli_tool_manager import CLIToolManager

        manager = CLIToolManager()
        manager.setup_codex_workspace(
            worktree_path=temp_worktree,
            session_id="codex-workspace-setup-test",
            gpu_ids=[0, 1, 2, 3],
            tp_size=2,
            dp_size=2,
        )

        assert (temp_worktree / ".codex").is_dir()
        assert (temp_worktree / "AGENTS.md").exists()


# =============================================================================
# T3: session_manager injects AMMO_TP_SIZE / AMMO_DP_SIZE into ttyd extra_env
# =============================================================================

@pytest.mark.unit
class TestCreateSessionInjectsAMMOTpDp:
    """
    Plan T3 (.claude/plans/gpu-tp-dp-awareness.md):
      create_session must forward state.tp_size / state.dp_size into the
      extra_env dict that gets passed to start_terminal_with_command, so the
      ttyd process (and everything under it — tmux, bash, CLI tool, agents)
      inherits AMMO_TP_SIZE / AMMO_DP_SIZE as process env.

    This parallels the existing AMMO_GPU_RES_DIR injection — same inline
    pattern right after the HF_HOME / AMMO_GPU_RES_DIR block.
    """

    @pytest.mark.asyncio
    async def test_create_session_injects_ammo_tp_dp_into_ttyd_env(
        self, mock_session_manager, tmp_path
    ):
        session_dir = tmp_path / "test-session-tp-dp"
        mock_session_manager.worktree_manager.get_session_dir.return_value = session_dir

        request = make_create_request(gpu_count=4, tp_size=2, dp_size=2)
        response = await mock_session_manager.create_session(request)
        session_id = response.session_id

        terminal_mock = mock_session_manager.terminal_manager
        terminal_mock.start_terminal_with_command.assert_called_once()

        call_args = terminal_mock.start_terminal_with_command.call_args
        env_passed = call_args.kwargs.get("env") or (call_args[1].get("env") if call_args[1] else {})

        assert env_passed.get("AMMO_TP_SIZE") == "2", (
            f"AMMO_TP_SIZE missing or wrong in ttyd env on create_session. "
            f"Got {env_passed.get('AMMO_TP_SIZE')!r}; keys: {sorted(env_passed.keys())}"
        )
        assert env_passed.get("AMMO_DP_SIZE") == "2", (
            f"AMMO_DP_SIZE missing or wrong in ttyd env on create_session. "
            f"Got {env_passed.get('AMMO_DP_SIZE')!r}; keys: {sorted(env_passed.keys())}"
        )


@pytest.mark.unit
class TestResumeSessionInjectsAMMOTpDp:
    """Resume path must inject the same AMMO_TP_SIZE / AMMO_DP_SIZE so paused
    sessions come back with the correct parallelism env."""

    @pytest.mark.asyncio
    async def test_resume_session_injects_ammo_tp_dp_into_ttyd_env(
        self, mock_session_manager, tmp_path
    ):
        session_id = "resume-tp-dp-test"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=4,
            gpu_ids=[],
            tp_size=2,
            dp_size=2,
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.terminal_manager.is_available.return_value = True
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9002
        )

        await mock_session_manager.resume_session(session_id)

        terminal_mock = mock_session_manager.terminal_manager
        terminal_mock.start_terminal_with_command.assert_called_once()

        call_args = terminal_mock.start_terminal_with_command.call_args
        env_passed = call_args.kwargs.get("env") or (call_args[1].get("env") if call_args[1] else {})

        assert env_passed.get("AMMO_TP_SIZE") == "2", (
            f"AMMO_TP_SIZE missing or wrong in ttyd env on resume_session. "
            f"Got {env_passed.get('AMMO_TP_SIZE')!r}; keys: {sorted(env_passed.keys())}"
        )
        assert env_passed.get("AMMO_DP_SIZE") == "2", (
            f"AMMO_DP_SIZE missing or wrong in ttyd env on resume_session. "
            f"Got {env_passed.get('AMMO_DP_SIZE')!r}; keys: {sorted(env_passed.keys())}"
        )


@pytest.mark.unit
class TestLegacySessionWithoutTpSizeDoesNotInject:
    """tp_size=0 is the legacy sentinel (session_models.py:336) — resume path
    must NOT inject bogus AMMO_TP_SIZE / AMMO_DP_SIZE for those sessions."""

    @pytest.mark.asyncio
    async def test_legacy_session_without_tp_size_does_not_inject(
        self, mock_session_manager, tmp_path
    ):
        session_id = "legacy-session-no-tp"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        # Legacy: tp_size=0 means unknown; dp_size=1 means "no DP" or "unknown".
        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=1,
            gpu_ids=[],
            tp_size=0,
            dp_size=1,
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        mock_session_manager.terminal_manager.is_available.return_value = True
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9003
        )

        await mock_session_manager.resume_session(session_id)

        terminal_mock = mock_session_manager.terminal_manager
        call_args = terminal_mock.start_terminal_with_command.call_args
        env_passed = call_args.kwargs.get("env") or (call_args[1].get("env") if call_args[1] else {})

        assert "AMMO_TP_SIZE" not in env_passed, (
            f"AMMO_TP_SIZE must be absent for legacy tp_size=0 session; "
            f"got {env_passed.get('AMMO_TP_SIZE')!r}"
        )
        assert "AMMO_DP_SIZE" not in env_passed, (
            f"AMMO_DP_SIZE must be absent for legacy sessions; "
            f"got {env_passed.get('AMMO_DP_SIZE')!r}"
        )


@pytest.mark.unit
class TestDecoupledGpuCountEnvInjection:
    """Post gpu-decouple: `gpu_count` may exceed `tp_size * dp_size`.

    Plan: .claude/plans/gpu-decouple.md Task 13.

    When the session pool is larger than the model-replica footprint
    (e.g. gpu_count=6 with tp=2, dp=1, leaving 4 spare GPUs for parallel
    experiment tracks), the session workspace must still project:
      - `CUDA_VISIBLE_DEVICES` listing ALL 6 pool GPU IDs (not tp*dp=2)
      - `AMMO_TP_SIZE=2` / `AMMO_DP_SIZE=1` (the model replica, not the pool)
      - `AMMO_GPU_RES_DIR=/tmp/ammo_gpu_res_{session_id}` for the
        two-layer session-local reservation subsystem.

    The three invariants together let `gpu_reservation.py reserve
    --num-gpus {tp*dp}` allocate the E2E world size while leaving the
    remaining pool GPUs free for parallel kernel work.
    """

    @pytest.fixture
    def temp_worktree(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_decoupled_pool_settings_local_env_layout(self, temp_worktree):
        """gpu_count=6 with tp=2, dp=1 produces correct env in settings.local.json."""
        from orchestration.cli_tool_manager import CLIToolManager

        session_id = "decoupled-pool-env-test"
        # Pool = 6 physical GPUs, larger than model replica (tp*dp=2).
        pool_gpu_ids = [0, 1, 2, 3, 4, 5]

        manager = CLIToolManager()
        manager.setup_claude_workspace(
            worktree_path=temp_worktree,
            session_id=session_id,
            gpu_ids=pool_gpu_ids,
            tp_size=2,
            dp_size=1,
        )

        settings_local_path = temp_worktree / ".claude" / "settings.local.json"
        assert settings_local_path.exists(), (
            "settings.local.json must exist after setup_claude_workspace"
        )

        with open(settings_local_path) as f:
            settings = json.load(f)
        env_section = settings.get("env", {})

        # (1) CUDA_VISIBLE_DEVICES reflects full pool, NOT tp*dp.
        cvd = env_section.get("CUDA_VISIBLE_DEVICES", "")
        assert cvd == "0,1,2,3,4,5", (
            f"CUDA_VISIBLE_DEVICES must list all 6 pool GPUs, got {cvd!r}"
        )
        cvd_ids = [x for x in cvd.split(",") if x]
        assert len(cvd_ids) == 6, (
            f"CUDA_VISIBLE_DEVICES must have 6 entries (pool size), got {len(cvd_ids)}: {cvd_ids}"
        )

        # (2) AMMO_TP_SIZE / AMMO_DP_SIZE reflect model replica, not pool.
        assert env_section.get("AMMO_TP_SIZE") == "2", (
            f"AMMO_TP_SIZE must be '2' (model replica), got {env_section.get('AMMO_TP_SIZE')!r}"
        )
        assert env_section.get("AMMO_DP_SIZE") == "1", (
            f"AMMO_DP_SIZE must be '1', got {env_section.get('AMMO_DP_SIZE')!r}"
        )

        # (3) AMMO_GPU_RES_DIR correctly namespaced per session.
        expected_dir = f"/tmp/ammo_gpu_res_{session_id}"
        assert env_section.get("AMMO_GPU_RES_DIR") == expected_dir, (
            f"AMMO_GPU_RES_DIR must be {expected_dir!r}, "
            f"got {env_section.get('AMMO_GPU_RES_DIR')!r}"
        )

    # Note: the ttyd-env path (start_terminal_with_command env kwarg) is
    # already covered by TestCreateSessionInjectsAMMOTpDp +
    # TestCreateSessionInjectsAMMOGPUResDir above, so we don't re-test it
    # here. The single settings.local.json test above is sufficient for the
    # decoupled-pool invariants this class is guarding.


@pytest.mark.unit
class TestTerminalRecoveryInjectsAMMOTpDp:
    """Recovered terminals must preserve the same AMMO env as create/resume."""

    @pytest.mark.asyncio
    async def test_codex_terminal_recovery_injects_ammo_tp_dp(
        self, mock_session_manager, tmp_path
    ):
        session_id = "recovery-codex-tp-dp"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CODEX,
            requested_gpu_count=4,
            gpu_ids=[4, 5, 6, 7],
            tp_size=2,
            dp_size=2,
            worktree_path=str(worktree_dir),
        )
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        terminal_mock = mock_session_manager.terminal_manager
        terminal_mock.is_terminal_running.return_value = False
        terminal_mock.is_available.return_value = True
        terminal_mock.cleanup_dead_terminal.return_value = True
        terminal_mock._tmux_session_exists.return_value = False
        terminal_mock.start_terminal_with_command = AsyncMock(return_value=9010)

        recovered_port = await mock_session_manager.ensure_terminal_healthy(session_id)

        assert recovered_port == 9010
        terminal_mock.start_terminal_with_command.assert_called_once()
        env_passed = terminal_mock.start_terminal_with_command.call_args.kwargs["env"]
        assert env_passed["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
        assert env_passed["AMMO_GPU_RES_DIR"] == f"/tmp/ammo_gpu_res_{session_id}"
        assert env_passed["AMMO_TP_SIZE"] == "2"
        assert env_passed["AMMO_DP_SIZE"] == "2"
