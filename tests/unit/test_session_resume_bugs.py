# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for session resume bug fixes.

Bug 1: .git file excluded from S3 tar archive (should only exclude .git/ directory)
Bug 2: Session marked ACTIVE even when terminal fails to start on resume
Bug 3: Cross-pod resume doesn't repair git worktree linkage
"""

import os
import subprocess
import sys
import tempfile
import shutil
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
)
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
)


# ============================================================================
# Bug 1: .git file excluded from S3 tar archive
# ============================================================================

@pytest.mark.unit
class TestTarExcludePreservesGitFile:
    """Verify that tar exclude patterns exclude .git/ directories but preserve .git files."""

    def test_tar_exclude_does_not_exclude_dot_git(self):
        """_build_tar_exclude_args must NOT exclude .git (file or directory).

        In git worktrees, .git is a tiny file containing
        'gitdir: /path/to/.git/worktrees/...'. GNU tar's --exclude=.git
        would exclude both files AND directories named .git. Since there's
        no tar flag to exclude only directories, .git is removed from the
        exclude list entirely. The .git file is tiny and must be preserved
        for worktree linkage after S3 restore.
        """
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        args = storage._build_tar_exclude_args()

        # Extract patterns from --exclude=X args
        patterns = [arg.split("=", 1)[1] for arg in args]

        # Neither .git nor .git/ should be in the exclude patterns
        assert ".git" not in patterns, (
            "'.git' should not be in exclude patterns "
            "because it would exclude .git files needed for worktree linkage"
        )
        assert ".git/" not in patterns, (
            "'.git/' should not be in exclude patterns either"
        )

    def test_tar_preserves_git_file_in_worktree(self):
        """Real tar test: .git file (as in worktrees) survives tar+extract with .git/ exclude.

        In git worktrees, the worktree root has a .git FILE containing
        'gitdir: /path/to/base/.git/worktrees/session-id'. This file must
        be preserved for the worktree to function after S3 restore.
        """
        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "src"
            worktree = src / "worktree"
            worktree.mkdir(parents=True)

            # Create a .git file (like in a git worktree)
            git_file = worktree / ".git"
            git_file.write_text("gitdir: /repos/vllm/.git/worktrees/session-abc\n")

            # Also create a regular file for sanity
            (worktree / "code.py").write_text("x = 1")

            from orchestration.session_state import SessionS3Storage

            with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
                storage = SessionS3Storage()

            exclude_args = storage._build_tar_exclude_args()

            # Create tar and extract
            tar_create = subprocess.run(
                ["tar", "cf", "-"] + exclude_args + ["-C", str(src), "worktree"],
                capture_output=True,
                check=True,
            )

            dst = Path(tmp) / "dst"
            dst.mkdir()
            subprocess.run(
                ["tar", "xf", "-", "-C", str(dst)],
                input=tar_create.stdout,
                check=True,
            )

            # .git file should be preserved (critical for worktree linkage)
            restored_git_file = dst / "worktree" / ".git"
            assert restored_git_file.exists(), ".git file should be preserved in tar"
            assert restored_git_file.is_file(), ".git should be a file, not a directory"
            assert "gitdir:" in restored_git_file.read_text()

            # Regular file should be preserved
            assert (dst / "worktree" / "code.py").exists()

        finally:
            shutil.rmtree(tmp)

    def test_other_exclude_patterns_still_work(self):
        """Verify that other exclude patterns (__pycache__, build, etc.) still work."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        args = storage._build_tar_exclude_args()
        patterns = [arg.split("=", 1)[1] for arg in args]

        # These should still be excluded
        assert "__pycache__" in patterns
        assert "*.pyc" in patterns
        assert "build" in patterns
        assert "*.so" in patterns
        assert "venv" in patterns
        assert ".venv" in patterns


# ============================================================================
# Bug 2: Session marked ACTIVE even when terminal fails to start
# ============================================================================

@pytest.mark.unit
class TestResumeTerminalFailure:
    """Verify session stays PAUSED when terminal fails to start during resume."""

    @pytest.mark.asyncio
    async def test_resume_stays_paused_on_terminal_error(self, mock_session_manager, tmp_path):
        """When TerminalError is raised during resume, session must stay PAUSED.

        Previously, the code caught TerminalError as a warning but still set
        status=ACTIVE unconditionally, making the session unreachable (subsequent
        resume calls would return 'already active' without recovery).
        """
        from orchestration.terminal_manager import TerminalError

        session_id = "sess-terminal-fail"
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

        # Make terminal raise TerminalError
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=TerminalError("ttyd failed to bind port")
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        # Resume should still succeed (returns a response) but session stays PAUSED
        response = await mock_session_manager.resume_session(session_id)

        # Session must remain PAUSED, not ACTIVE
        assert state.status == SessionStatus.PAUSED, (
            f"Expected session to stay PAUSED on terminal failure, got {state.status.value}"
        )
        # Error should be set
        assert state.error is not None
        assert "terminal" in state.error.lower() or "ttyd" in state.error.lower()

    @pytest.mark.asyncio
    async def test_resume_sets_active_on_success(self, mock_session_manager, tmp_path):
        """When terminal starts successfully, session should transition to ACTIVE."""
        session_id = "sess-terminal-ok"
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

        # Terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        response = await mock_session_manager.resume_session(session_id)

        # Session must be ACTIVE
        assert state.status == SessionStatus.ACTIVE, (
            f"Expected session to be ACTIVE on successful terminal start, got {state.status.value}"
        )

    @pytest.mark.asyncio
    async def test_resume_after_terminal_failure_can_retry(self, mock_session_manager, tmp_path):
        """After a terminal failure, session stays PAUSED so user can retry resume."""
        from orchestration.terminal_manager import TerminalError

        session_id = "sess-retry"
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

        # First attempt: terminal fails
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=TerminalError("port busy")
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        await mock_session_manager.resume_session(session_id)
        assert state.status == SessionStatus.PAUSED

        # Second attempt: terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9002
        )

        response = await mock_session_manager.resume_session(session_id)
        assert state.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_resume_releases_gpus_on_terminal_error(self, mock_session_manager, tmp_path):
        """When TerminalError is raised during resume, acquired GPUs must be released.

        Previously, GPUs were acquired before the terminal start block but NOT
        released in the TerminalError catch handler. This caused the session to
        stay PAUSED while holding GPU allocations indefinitely (GPU leak).
        """
        from orchestration.terminal_manager import TerminalError

        session_id = "sess-gpu-leak"
        session_dir = tmp_path / "sessions" / session_id
        worktree_dir = session_dir / "worktree"
        worktree_dir.mkdir(parents=True)

        state = make_session_state(
            session_id=session_id,
            status=SessionStatus.PAUSED,
            requested_gpu_count=1,
            gpu_ids=[],
        )
        state.worktree_path = str(worktree_dir)
        state.session_dir = str(session_dir)
        mock_session_manager._sessions[session_id] = state

        # Verify GPUs are available before resume
        available_before = mock_session_manager.gpu_manager.get_available_gpu_count()

        # Make terminal raise TerminalError
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=TerminalError("ttyd failed to bind port")
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        # Resume — terminal fails
        response = await mock_session_manager.resume_session(session_id)

        # Session must remain PAUSED
        assert state.status == SessionStatus.PAUSED

        # GPUs must be released (available count should be the same as before)
        available_after = mock_session_manager.gpu_manager.get_available_gpu_count()
        assert available_after == available_before, (
            f"GPU leak: {available_before} GPUs available before resume, "
            f"only {available_after} after terminal failure (expected {available_before})"
        )

        # gpu_ids should be cleared on the session state
        assert state.gpu_ids == [], (
            f"Expected gpu_ids to be cleared after terminal failure, got {state.gpu_ids}"
        )


# ============================================================================
# Bug 3: Cross-pod resume doesn't repair git worktree linkage
# ============================================================================

@pytest.mark.unit
class TestRepairWorktreeLinkage:
    """Verify worktree git linkage is repaired after cross-host S3 restore."""

    def test_repair_creates_git_file_when_missing(self, tmp_path):
        """When .git file is missing, repair_worktree_linkage creates it with correct gitdir."""
        from orchestration.worktree_manager import WorktreeManager

        repos_dir = tmp_path / "repos"
        sessions_dir = tmp_path / "sessions"
        repos_dir.mkdir()
        sessions_dir.mkdir()

        manager = WorktreeManager(
            repos_dir=str(repos_dir),
            sessions_dir=str(sessions_dir),
        )

        # Set up a base repo (simulated)
        base_repo = repos_dir / "vllm"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        (base_repo / ".git" / "worktrees").mkdir()

        # Create session worktree directory (as if restored from S3 without .git)
        session_id = "session-abc123"
        worktree_path = sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True)
        (worktree_path / "code.py").write_text("x = 1")

        # No .git file exists (simulates restored from S3 before Bug 1 fix)
        assert not (worktree_path / ".git").exists()

        # Repair the linkage
        manager.repair_worktree_linkage(
            session_id=session_id,
            worktree_path=worktree_path,
            repo_name="vllm",
            branch="main",
        )

        # .git file should now exist with correct gitdir
        git_file = worktree_path / ".git"
        assert git_file.exists(), ".git file should be created by repair"
        assert git_file.is_file(), ".git should be a file, not a directory"

        content = git_file.read_text().strip()
        expected_gitdir = str(base_repo / ".git" / "worktrees" / session_id)
        assert content == f"gitdir: {expected_gitdir}"

        # Worktree entry in base repo should exist
        wt_entry = base_repo / ".git" / "worktrees" / session_id
        assert wt_entry.exists(), "Worktree entry should be created in base repo"
        assert (wt_entry / "gitdir").exists()
        assert (wt_entry / "HEAD").exists()
        assert (wt_entry / "commondir").exists()

        # Verify gitdir file points back to worktree's .git file
        gitdir_content = (wt_entry / "gitdir").read_text().strip()
        assert gitdir_content == str(worktree_path / ".git")

        # Verify commondir points to base repo .git
        commondir_content = (wt_entry / "commondir").read_text().strip()
        assert commondir_content == "../.."

    def test_repair_fixes_stale_git_file(self, tmp_path):
        """When .git file exists but points to wrong gitdir, repair fixes it."""
        from orchestration.worktree_manager import WorktreeManager

        repos_dir = tmp_path / "repos"
        sessions_dir = tmp_path / "sessions"
        repos_dir.mkdir()
        sessions_dir.mkdir()

        manager = WorktreeManager(
            repos_dir=str(repos_dir),
            sessions_dir=str(sessions_dir),
        )

        # Set up base repo
        base_repo = repos_dir / "vllm"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        (base_repo / ".git" / "worktrees").mkdir()

        # Create worktree with stale .git file (pointing to old pod path)
        session_id = "session-stale-git"
        worktree_path = sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True)
        (worktree_path / "code.py").write_text("x = 1")

        # Write stale .git file pointing to non-existent path
        stale_gitdir = "/old-pod/repos/vllm/.git/worktrees/session-stale-git"
        (worktree_path / ".git").write_text(f"gitdir: {stale_gitdir}\n")

        # Repair
        manager.repair_worktree_linkage(
            session_id=session_id,
            worktree_path=worktree_path,
            repo_name="vllm",
            branch="main",
        )

        # .git file should now point to correct path
        content = (worktree_path / ".git").read_text().strip()
        expected_gitdir = str(base_repo / ".git" / "worktrees" / session_id)
        assert content == f"gitdir: {expected_gitdir}"

    def test_repair_noop_when_linkage_valid(self, tmp_path):
        """When .git file and worktree entry are both valid, repair is a no-op."""
        from orchestration.worktree_manager import WorktreeManager

        repos_dir = tmp_path / "repos"
        sessions_dir = tmp_path / "sessions"
        repos_dir.mkdir()
        sessions_dir.mkdir()

        manager = WorktreeManager(
            repos_dir=str(repos_dir),
            sessions_dir=str(sessions_dir),
        )

        # Set up base repo
        base_repo = repos_dir / "vllm"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        (base_repo / ".git" / "worktrees").mkdir()

        session_id = "session-valid"
        worktree_path = sessions_dir / session_id / "worktree"
        worktree_path.mkdir(parents=True)

        # Create valid .git file
        expected_gitdir = str(base_repo / ".git" / "worktrees" / session_id)
        (worktree_path / ".git").write_text(f"gitdir: {expected_gitdir}\n")

        # Create valid worktree entry
        wt_entry = base_repo / ".git" / "worktrees" / session_id
        wt_entry.mkdir(parents=True)
        (wt_entry / "gitdir").write_text(str(worktree_path / ".git"))
        (wt_entry / "HEAD").write_text("ref: refs/heads/session/session-valid\n")
        (wt_entry / "commondir").write_text("../..")

        # Repair should be a no-op
        manager.repair_worktree_linkage(
            session_id=session_id,
            worktree_path=worktree_path,
            repo_name="vllm",
            branch="main",
        )

        # .git file should be unchanged
        content = (worktree_path / ".git").read_text().strip()
        assert content == f"gitdir: {expected_gitdir}"

    @pytest.mark.asyncio
    async def test_resume_calls_repair_after_s3_restore(self, mock_session_manager, tmp_path):
        """After restoring from S3, resume_session calls repair_worktree_linkage."""
        session_id = "sess-cross-pod"
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

        # Simulate: session not in local memory, S3 restore succeeds
        # We need the session not in _sessions initially, then S3 restore adds it
        mock_session_manager.session_storage.enabled = True
        mock_session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=state
        )

        # Track repair_worktree_linkage calls
        mock_session_manager.worktree_manager.repair_worktree_linkage = MagicMock()

        # Terminal succeeds
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        response = await mock_session_manager.resume_session(session_id)

        # repair_worktree_linkage should have been called
        mock_session_manager.worktree_manager.repair_worktree_linkage.assert_called_once_with(
            session_id=session_id,
            worktree_path=worktree_dir,
            repo_name=state.repo_name,
            branch=state.branch,
        )


# ============================================================================
# Bug 4: docker-run.sh ccache volume mount path mismatch
# ============================================================================

@pytest.mark.unit
class TestDockerRunCcacheMount:
    """Verify docker-run.sh mounts ccache volume to the correct container path."""

    def test_docker_run_sh_ccache_mount_matches_dockerfile(self):
        """The ccache volume mount in docker-run.sh must match Dockerfile CCACHE_DIR.

        The Dockerfile sets ENV CCACHE_DIR=/home/session_user/.ccache but
        docker-run.sh was mounting the volume to /root/.ccache. Sessions run
        as session_user and reference CCACHE_DIR, so the mount must target
        /home/session_user/.ccache.
        """
        docker_run_path = Path(__file__).parent.parent.parent / "docker-run.sh"
        assert docker_run_path.exists(), f"docker-run.sh not found at {docker_run_path}"

        content = docker_run_path.read_text()

        # Find the ccache volume mount line
        ccache_mount_lines = [
            line.strip() for line in content.splitlines()
            if "ccache" in line.lower() and "-v" in line.lower()
        ]
        assert len(ccache_mount_lines) > 0, "No ccache volume mount found in docker-run.sh"

        # The mount should target /home/session_user/.ccache, NOT /root/.ccache
        for line in ccache_mount_lines:
            assert "/root/.ccache" not in line, (
                f"ccache volume mount targets /root/.ccache but Dockerfile sets "
                f"CCACHE_DIR=/home/session_user/.ccache. Line: {line}"
            )
            assert "/home/session_user/.ccache" in line, (
                f"ccache volume mount should target /home/session_user/.ccache "
                f"to match Dockerfile CCACHE_DIR. Line: {line}"
            )


# ============================================================================
# Bug 5: create_session marks ACTIVE even when terminal fails
# ============================================================================

@pytest.mark.unit
class TestCreateSessionTerminalFailure:
    """Verify create_session handles TerminalError like resume_session does."""

    @pytest.mark.asyncio
    async def test_create_session_fails_on_terminal_error(self, mock_session_manager):
        """When TerminalError is raised during create, session must be FAILED (not ACTIVE).

        This is the same bug pattern fixed in resume_session: TerminalError was
        caught as a warning but status was set to ACTIVE unconditionally afterward.
        """
        from orchestration.terminal_manager import TerminalError
        from fixtures.session_fixtures import make_create_request

        # Make terminal raise TerminalError
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=TerminalError("ttyd failed to bind port")
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        request = make_create_request(gpu_count=0)
        response = await mock_session_manager.create_session(request, owner_id="test-client")

        # Find the created session
        session_id = response.session_id
        state = mock_session_manager._sessions[session_id]

        # Session must NOT be ACTIVE — terminal didn't start
        assert state.status == SessionStatus.FAILED, (
            f"Expected session to be FAILED on terminal failure, got {state.status.value}"
        )
        # Error should be set
        assert state.error is not None
        assert "terminal" in state.error.lower() or "ttyd" in state.error.lower()

    @pytest.mark.asyncio
    async def test_create_session_releases_gpus_on_terminal_error(self, mock_session_manager):
        """When TerminalError is raised during create with GPUs, GPUs must be released.

        Without this fix, GPUs acquired before terminal start would be held
        indefinitely by a FAILED session.
        """
        from orchestration.terminal_manager import TerminalError
        from fixtures.session_fixtures import make_create_request

        available_before = mock_session_manager.gpu_manager.get_available_gpu_count()

        # Make terminal raise TerminalError
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            side_effect=TerminalError("port conflict")
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        request = make_create_request(gpu_count=1)
        response = await mock_session_manager.create_session(request, owner_id="test-client")

        session_id = response.session_id
        state = mock_session_manager._sessions[session_id]

        # Session must be FAILED
        assert state.status == SessionStatus.FAILED

        # GPUs must be released
        available_after = mock_session_manager.gpu_manager.get_available_gpu_count()
        assert available_after == available_before, (
            f"GPU leak: {available_before} GPUs available before create, "
            f"only {available_after} after terminal failure"
        )

        # gpu_ids should be cleared
        assert state.gpu_ids == [], (
            f"Expected gpu_ids to be cleared after terminal failure, got {state.gpu_ids}"
        )

    @pytest.mark.asyncio
    async def test_create_session_active_on_success(self, mock_session_manager):
        """When terminal starts successfully, session should be ACTIVE (happy path)."""
        from fixtures.session_fixtures import make_create_request

        # Terminal succeeds (default mock behavior)
        mock_session_manager.terminal_manager.start_terminal_with_command = AsyncMock(
            return_value=9001
        )
        mock_session_manager.terminal_manager.is_available.return_value = True

        request = make_create_request(gpu_count=0)
        response = await mock_session_manager.create_session(request, owner_id="test-client")

        session_id = response.session_id
        state = mock_session_manager._sessions[session_id]

        # Session must be ACTIVE
        assert state.status == SessionStatus.ACTIVE, (
            f"Expected session to be ACTIVE on successful terminal start, got {state.status.value}"
        )
