# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for session permission fixes.

Validates that:
1. Dockerfile configures git safe.directory, CCACHE_DIR for session_user
2. _run_git() drops privileges to session_user via preexec_fn (no subprocess chown)
3. worktree_manager methods do NOT call subprocess chown/chmod
4. repair_worktree_linkage uses targeted os.chown (not subprocess)
5. update_base_repo does NOT re-chown after fetch (git runs as session_user)
"""
import re
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# Resolve project root relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_dockerfile_content() -> str:
    p = PROJECT_ROOT / "Dockerfile"
    if not p.exists():
        pytest.skip("Dockerfile not found")
    return p.read_text()


def _get_session_manager_content() -> str:
    p = PROJECT_ROOT / "orchestration" / "session_manager.py"
    if not p.exists():
        pytest.skip("session_manager.py not found")
    return p.read_text()


# ---------------------------------------------------------------------------
# Part A: Dockerfile validation tests (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDockerfileGitSafeDirectory:
    """Dockerfile must configure git safe.directory globally."""

    def test_dockerfile_has_safe_directory_config(self):
        """Dockerfile must contain 'git config --system --add safe.directory'."""
        content = _get_dockerfile_content()
        assert "git config --system --add safe.directory" in content, (
            "Dockerfile must run 'git config --system --add safe.directory' "
            "to prevent 'dubious ownership' errors when root reads "
            "session_user-owned repos."
        )


@pytest.mark.unit
class TestDockerfileCcacheDir:
    """Dockerfile must set CCACHE_DIR to a session_user-writable location."""

    def test_dockerfile_ccache_dir_not_root(self):
        """CCACHE_DIR must NOT point to /root/."""
        content = _get_dockerfile_content()
        matches = re.findall(r'ENV\s+CCACHE_DIR\s*=\s*(\S+)', content)
        assert matches, "Dockerfile must define ENV CCACHE_DIR"
        for val in matches:
            assert "/root/" not in val, (
                f"CCACHE_DIR={val} is under /root/ which is not writable by "
                "session_user (uid 1000). Use /home/session_user/.ccache instead."
            )

    def test_dockerfile_ccache_dir_session_user(self):
        """CCACHE_DIR must point to a session_user-writable location."""
        content = _get_dockerfile_content()
        matches = re.findall(r'ENV\s+CCACHE_DIR\s*=\s*(\S+)', content)
        assert matches, "Dockerfile must define ENV CCACHE_DIR"
        last_val = matches[-1]
        assert "/home/session_user/" in last_val or "/data/" in last_val or "/tmp/" in last_val, (
            f"CCACHE_DIR={last_val} is not under a session_user-writable path. "
            "Expected /home/session_user/.ccache or similar."
        )


@pytest.mark.unit
class TestAmmoTrackWorktreeCleanupParity:
    """Session cleanup must cover Codex track worktrees, not just Claude tracks."""

    def test_terminate_session_checks_codex_worktrees_dir(self):
        """AMMO track cleanup must inspect .codex/worktrees for Codex sessions."""
        content = _get_session_manager_content()
        cleanup_start = content.index("# Clean up AMMO track worktrees")
        cleanup_end = content.index("# Clean up worktree and session directory", cleanup_start)
        cleanup_block = content[cleanup_start:cleanup_end]

        assert '".codex"' in cleanup_block or "'.codex'" in cleanup_block, (
            "terminate_session must inspect .codex/worktrees so Codex AMMO "
            "track worktrees are removed before session cleanup."
        )
        assert '".claude"' in cleanup_block or "'.claude'" in cleanup_block, (
            "terminate_session should preserve Claude track cleanup parity."
        )

    @pytest.mark.asyncio
    async def test_terminate_session_removes_codex_track_worktree(self, tmp_path):
        """A Codex AMMO track under .codex/worktrees is passed to git worktree remove."""
        from orchestration.session_manager import SessionManager
        from shared.session_models import SessionState, SessionStatus, CLIToolType
        import time as _time

        worktree_path = tmp_path / "sessions" / "sess-codex" / "worktree"
        track_dir = worktree_path / ".codex" / "worktrees" / "op001-short-description"
        track_dir.mkdir(parents=True)

        mock_worktree = MagicMock()
        mock_worktree.cleanup_session = MagicMock()
        mock_gpu = MagicMock()
        mock_gpu.release_gpus_for_session = MagicMock()
        mock_terminal = MagicMock()
        mock_cli = MagicMock()
        mock_inactivity = MagicMock()
        mock_storage = MagicMock()
        mock_storage.enabled = False

        manager = SessionManager(
            sessions_dir=str(tmp_path / "sessions"),
            worktree_manager=mock_worktree,
            gpu_manager=mock_gpu,
            terminal_manager=mock_terminal,
            cli_tool_manager=mock_cli,
            inactivity_monitor=mock_inactivity,
            session_storage=mock_storage,
        )
        manager._sessions["sess-codex"] = SessionState(
            session_id="sess-codex",
            status=SessionStatus.ACTIVE,
            cli_tool=CLIToolType.CODEX,
            repo_name="vllm",
            branch="main",
            gpu_ids=[],
            created_at=_time.time(),
            last_accessed=_time.time(),
            terminal_port=None,
            worktree_path=str(worktree_path),
            session_dir=str(worktree_path.parent),
        )

        calls = []

        class FakeProcess:
            returncode = 0

            async def communicate(self):
                return (str(tmp_path / "base" / ".git").encode(), b"")

        async def fake_create_subprocess_exec(*args, **kwargs):
            calls.append(args)
            return FakeProcess()

        with patch("orchestration.session_manager.asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            await manager.terminate_session("sess-codex")

        assert any(
            args[:4] == ("git", "-C", str(tmp_path / "base"), "worktree")
            and "remove" in args
            and str(track_dir) in args
            for args in calls
        ), f"Expected git worktree remove for {track_dir}; got {calls}"


# ---------------------------------------------------------------------------
# Part B: _run_git drops privileges via preexec_fn
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunGitDropsPrivileges:
    """_run_git() must use preexec_fn to drop to session_user (UID 1000)."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_run_git_has_preexec_fn(self, mock_subprocess_run):
        """subprocess.run must be called with preexec_fn that does setgid+setuid."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorktreeManager(
                repos_dir=str(Path(tmpdir) / "repos"),
                sessions_dir=str(Path(tmpdir) / "sessions"),
            )
            mock_subprocess_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0
            )
            mgr._run_git(["status"], cwd=Path(tmpdir))

            preexec_fn = mock_subprocess_run.call_args.kwargs.get("preexec_fn")
            assert preexec_fn is not None, (
                "_run_git must pass preexec_fn to subprocess.run"
            )

            with patch("os.setgid") as mock_setgid, \
                 patch("os.setuid") as mock_setuid:
                preexec_fn()
                mock_setgid.assert_called_once_with(1000)
                mock_setuid.assert_called_once_with(1000)


# ---------------------------------------------------------------------------
# Part C: create_worktree does NOT call subprocess chown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateWorktreeNoSubprocessChown:
    """create_worktree() must NOT call subprocess chown/chmod — git runs as session_user."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_create_worktree_no_subprocess_chown(self, mock_subprocess_run):
        """No subprocess chown/chmod calls after git worktree add."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))

            repo_dir = repos_dir / "vllm"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            (git_dir / "worktrees").mkdir()

            session_id = "test-session-123"

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True), \
                 patch.object(mgr, 'update_base_repo'), \
                 patch.object(mgr, 'get_repo_config', return_value={'url': 'https://github.com/vllm-project/vllm.git', 'default_branch': 'main'}):

                mock_result = MagicMock(stdout="abc123def456\n", returncode=0)
                worktree_path = sessions_dir / session_id / "worktree"

                def side_effect_run_git(args, **kwargs):
                    if args[0] == "worktree" and args[1] == "add":
                        worktree_path.mkdir(parents=True, exist_ok=True)
                    return mock_result

                mock_run_git.side_effect = side_effect_run_git
                mgr.create_worktree(session_id, "vllm", branch="main")

                chown_chmod_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0
                    and c[0][0][0] in ("chown", "chmod")
                ]
                assert len(chown_chmod_calls) == 0, (
                    "create_worktree must NOT call subprocess chown/chmod — "
                    f"git runs as session_user. Found: {chown_chmod_calls}"
                )


# ---------------------------------------------------------------------------
# Part D: repair_worktree_linkage uses os.chown, not subprocess
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRepairWorktreeLinkageOsChown:
    """repair_worktree_linkage() must use os.chown (targeted), not subprocess chown."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_repair_no_subprocess_chown(self, mock_subprocess_run):
        """No subprocess chown/chmod calls during repair."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))

            repo_dir = repos_dir / "vllm"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            (git_dir / "worktrees").mkdir()

            session_id = "test-repair-session"
            worktree_path = sessions_dir / session_id / "worktree"
            worktree_path.mkdir(parents=True, exist_ok=True)

            mgr.repair_worktree_linkage(session_id, worktree_path, "vllm")

            chown_chmod_calls = [
                c for c in mock_subprocess_run.call_args_list
                if len(c[0]) > 0 and len(c[0][0]) > 0
                and c[0][0][0] in ("chown", "chmod")
            ]
            assert len(chown_chmod_calls) == 0, (
                "repair_worktree_linkage must NOT use subprocess chown/chmod — "
                f"use os.chown instead. Found: {chown_chmod_calls}"
            )

    @patch("orchestration.worktree_manager.os.chown")
    @patch("orchestration.worktree_manager.subprocess.run")
    def test_repair_uses_os_chown(self, mock_subprocess_run, mock_os_chown):
        """os.chown must be called on files written during repair."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))

            repo_dir = repos_dir / "vllm"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            (git_dir / "worktrees").mkdir()

            session_id = "test-repair-os-chown"
            worktree_path = sessions_dir / session_id / "worktree"
            worktree_path.mkdir(parents=True, exist_ok=True)

            mgr.repair_worktree_linkage(session_id, worktree_path, "vllm")

            assert mock_os_chown.call_count >= 1, (
                "repair_worktree_linkage must call os.chown() on written files"
            )
            for c in mock_os_chown.call_args_list:
                assert c[0][1] == 1000 and c[0][2] == 1000, (
                    f"os.chown must use uid=1000, gid=1000. Got: {c}"
                )


# ---------------------------------------------------------------------------
# Part E: clone_base_repo does NOT subprocess chown or set sharedRepository
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloneBaseRepoNoChown:
    """clone_base_repo() must NOT call subprocess chown or sharedRepository."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_clone_no_subprocess_chown(self, mock_subprocess_run):
        """No subprocess chown calls after clone."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))
            repo_dir = repos_dir / "vllm"

            with patch.object(mgr, '_run_git') as mock_run_git:
                def side_effect(args, **kwargs):
                    if args[0] == "clone":
                        repo_dir.mkdir(parents=True, exist_ok=True)
                        (repo_dir / ".git").mkdir()
                    return MagicMock(stdout="", returncode=0)
                mock_run_git.side_effect = side_effect

                mgr.clone_base_repo("vllm")

                chown_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] == "chown"
                ]
                assert len(chown_calls) == 0, (
                    "clone_base_repo must NOT call subprocess chown — "
                    f"git runs as session_user. Found: {chown_calls}"
                )

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_clone_no_shared_repository(self, mock_subprocess_run):
        """No core.sharedRepository config after clone."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))
            repo_dir = repos_dir / "vllm"

            with patch.object(mgr, '_run_git') as mock_run_git:
                def side_effect(args, **kwargs):
                    if args[0] == "clone":
                        repo_dir.mkdir(parents=True, exist_ok=True)
                        (repo_dir / ".git").mkdir()
                    return MagicMock(stdout="", returncode=0)
                mock_run_git.side_effect = side_effect

                mgr.clone_base_repo("vllm")

                shared_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if "core.sharedRepository" in str(c)
                ]
                assert len(shared_calls) == 0, (
                    "clone_base_repo must NOT set core.sharedRepository — "
                    f"not needed. Found: {shared_calls}"
                )

    # Note: per-repo safe.directory is NOT needed — the Dockerfile sets
    # safe.directory '*' system-wide which covers all repos. No test for it.


# ---------------------------------------------------------------------------
# Part F: update_base_repo does NOT chown/chmod after fetch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateBaseRepoNoChown:
    """update_base_repo() must NOT call subprocess chown/chmod after fetch —
    git runs as session_user so ownership is not reset."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_update_no_subprocess_chown_chmod(self, mock_subprocess_run):
        """No chown/chmod calls after git fetch."""
        from orchestration.worktree_manager import WorktreeManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            repos_dir = Path(tmpdir) / "repos"
            sessions_dir = Path(tmpdir) / "sessions"
            repos_dir.mkdir()
            sessions_dir.mkdir()

            mgr = WorktreeManager(repos_dir=str(repos_dir), sessions_dir=str(sessions_dir))

            repo_dir = repos_dir / "vllm"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            (git_dir / "worktrees").mkdir()

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True):
                mock_run_git.return_value = MagicMock(stdout="", returncode=0)
                mgr.update_base_repo("vllm")

                chown_chmod_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0
                    and c[0][0][0] in ("chown", "chmod")
                ]
                assert len(chown_chmod_calls) == 0, (
                    "update_base_repo must NOT call subprocess chown/chmod — "
                    f"git runs as session_user. Found: {chown_chmod_calls}"
                )
