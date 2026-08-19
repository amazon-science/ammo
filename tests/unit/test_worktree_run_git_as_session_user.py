# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for _run_git() running as session_user (UID 1000) and chown removal.

Validates that:
1. subprocess.run is called with preexec_fn that drops to session_user
2. setgid(1000) is called before setuid(1000) (order matters)
3. Existing parameters (cmd, cwd, check, capture_output, text, timeout) are preserved
4. chown/chmod subprocess calls are removed from clone, update, create, repair methods
5. repair_worktree_linkage uses targeted os.chown instead of subprocess chown
6. cli_tool_manager no longer chowns worktree before launch
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile


@pytest.mark.unit
class TestRunGitUsesPreexecFn:
    """_run_git() must pass a preexec_fn that drops privileges to session_user."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_run_git_uses_preexec_fn(self, mock_subprocess_run):
        """subprocess.run must be called with a preexec_fn that calls
        setgid(1000) and setuid(1000)."""
        from orchestration.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorktreeManager(repos_dir=str(Path(tmpdir) / "repos"),
                                  sessions_dir=str(Path(tmpdir) / "sessions"))

            mock_subprocess_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0
            )

            mgr._run_git(["status"], cwd=Path(tmpdir))

            # Verify preexec_fn was passed
            assert mock_subprocess_run.called
            call_kwargs = mock_subprocess_run.call_args
            assert "preexec_fn" in call_kwargs.kwargs or (
                len(call_kwargs) > 1 and "preexec_fn" in str(call_kwargs)
            ), (
                "_run_git must pass preexec_fn to subprocess.run. "
                f"Got kwargs: {call_kwargs}"
            )
            preexec_fn = call_kwargs.kwargs.get("preexec_fn")
            assert preexec_fn is not None, (
                "preexec_fn must not be None"
            )

            # Call the preexec_fn and verify it calls setgid + setuid
            with patch("os.setgid") as mock_setgid, \
                 patch("os.setuid") as mock_setuid:
                preexec_fn()
                mock_setgid.assert_called_once_with(1000)
                mock_setuid.assert_called_once_with(1000)


@pytest.mark.unit
class TestRunGitPreexecFnOrder:
    """preexec_fn must call setgid before setuid (Linux requirement)."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_run_git_preexec_fn_sets_gid_before_uid(self, mock_subprocess_run):
        """setgid(1000) must be called before setuid(1000) because once
        uid is changed, setgid may fail."""
        from orchestration.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorktreeManager(repos_dir=str(Path(tmpdir) / "repos"),
                                  sessions_dir=str(Path(tmpdir) / "sessions"))

            mock_subprocess_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0
            )

            mgr._run_git(["status"], cwd=Path(tmpdir))

            preexec_fn = mock_subprocess_run.call_args.kwargs.get("preexec_fn")
            assert preexec_fn is not None

            # Record call order
            call_order = []
            with patch("os.setgid", side_effect=lambda x: call_order.append(("setgid", x))), \
                 patch("os.setuid", side_effect=lambda x: call_order.append(("setuid", x))):
                preexec_fn()

            assert len(call_order) >= 2, f"Expected 2+ calls, got: {call_order}"
            gid_idx = next(i for i, c in enumerate(call_order) if c[0] == "setgid")
            uid_idx = next(i for i, c in enumerate(call_order) if c[0] == "setuid")
            assert gid_idx < uid_idx, (
                f"setgid must be called before setuid. Call order: {call_order}"
            )


@pytest.mark.unit
class TestRunGitPreservesExistingBehavior:
    """_run_git must still pass cmd, cwd, check, capture_output, text, timeout."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_run_git_preserves_existing_behavior(self, mock_subprocess_run):
        """All existing parameters must still be passed to subprocess.run."""
        from orchestration.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            mgr = WorktreeManager(repos_dir=str(cwd / "repos"),
                                  sessions_dir=str(cwd / "sessions"))

            mock_subprocess_run.return_value = MagicMock(
                stdout="output", stderr="", returncode=0
            )

            mgr._run_git(["status"], cwd=cwd, check=True, capture_output=True)

            args, kwargs = mock_subprocess_run.call_args
            # cmd should be ["git", "status"]
            assert args[0] == ["git", "status"], f"cmd should be ['git', 'status'], got {args[0]}"
            assert kwargs.get("cwd") == cwd
            assert kwargs.get("check") is True
            assert kwargs.get("capture_output") is True
            assert kwargs.get("text") is True
            assert kwargs.get("timeout") == 600


# ---------------------------------------------------------------------------
# Task 3: No chown in clone_base_repo()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCloneBaseRepoNoChown:
    """clone_base_repo() must NOT call subprocess chown or sharedRepository config."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_clone_base_repo_no_chown_subprocess(self, mock_subprocess_run):
        """subprocess.run must never be called with ['chown', ...] in clone_base_repo."""
        from orchestration.worktree_manager import WorktreeManager

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

                # No subprocess.run with chown should be called
                chown_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] == "chown"
                ]
                assert len(chown_calls) == 0, (
                    "clone_base_repo must NOT call subprocess chown — git now runs as "
                    f"session_user via preexec_fn. Found calls: {chown_calls}"
                )

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_clone_base_repo_no_shared_repository_config(self, mock_subprocess_run):
        """subprocess.run must NOT be called with core.sharedRepository."""
        from orchestration.worktree_manager import WorktreeManager

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

                shared_repo_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if "core.sharedRepository" in str(c)
                ]
                assert len(shared_repo_calls) == 0, (
                    "clone_base_repo must NOT set core.sharedRepository — no longer needed "
                    f"when git runs as session_user. Found calls: {shared_repo_calls}"
                )


# ---------------------------------------------------------------------------
# Task 5: No chown in update_base_repo()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateBaseRepoNoChown:
    """update_base_repo() must NOT call subprocess chown/chmod after fetch."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_update_base_repo_no_chown_subprocess(self, mock_subprocess_run):
        """No chown/chmod subprocess calls after git fetch."""
        from orchestration.worktree_manager import WorktreeManager

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
            claude_worktrees = repo_dir / ".claude" / "worktrees"
            claude_worktrees.mkdir(parents=True)

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True):
                mock_run_git.return_value = MagicMock(stdout="", returncode=0)
                mgr.update_base_repo("vllm")

                chown_chmod_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] in ("chown", "chmod")
                ]
                assert len(chown_chmod_calls) == 0, (
                    "update_base_repo must NOT call subprocess chown/chmod after fetch — "
                    f"git runs as session_user. Found calls: {chown_chmod_calls}"
                )


# ---------------------------------------------------------------------------
# Task 7: No chown in create_worktree()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateWorktreeNoChown:
    """create_worktree() must NOT call subprocess chown/chmod."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_create_worktree_no_chown_chmod_subprocess(self, mock_subprocess_run):
        """No chown/chmod subprocess calls in create_worktree."""
        from orchestration.worktree_manager import WorktreeManager

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

            session_id = "test-no-chown-create"

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
                    if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] in ("chown", "chmod")
                ]
                assert len(chown_chmod_calls) == 0, (
                    "create_worktree must NOT call subprocess chown/chmod — git runs as "
                    f"session_user. Found calls: {chown_chmod_calls}"
                )

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_create_worktree_no_sticky_bit(self, mock_subprocess_run):
        """No chmod 1777 in create_worktree."""
        from orchestration.worktree_manager import WorktreeManager

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

            session_id = "test-no-sticky-create"

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

                chmod_1777_calls = [
                    c for c in mock_subprocess_run.call_args_list
                    if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] == "chmod"
                    and "1777" in str(c[0][0])
                ]
                assert len(chmod_1777_calls) == 0, (
                    "create_worktree must NOT call chmod 1777 — not needed when git runs "
                    f"as session_user. Found calls: {chmod_1777_calls}"
                )


# ---------------------------------------------------------------------------
# Task 9: No subprocess chown in repair_worktree_linkage(), use os.chown
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRepairWorktreeLinkageNoSubprocessChown:
    """repair_worktree_linkage() must NOT use subprocess chown, but must use os.chown."""

    @patch("orchestration.worktree_manager.subprocess.run")
    def test_repair_worktree_linkage_no_chown_chmod_subprocess(self, mock_subprocess_run):
        """No subprocess chown/chmod calls in repair_worktree_linkage."""
        from orchestration.worktree_manager import WorktreeManager

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

            session_id = "test-repair-no-chown"
            worktree_path = sessions_dir / session_id / "worktree"
            worktree_path.mkdir(parents=True, exist_ok=True)

            mgr.repair_worktree_linkage(session_id, worktree_path, "vllm")

            chown_chmod_calls = [
                c for c in mock_subprocess_run.call_args_list
                if len(c[0]) > 0 and len(c[0][0]) > 0 and c[0][0][0] in ("chown", "chmod")
            ]
            assert len(chown_chmod_calls) == 0, (
                "repair_worktree_linkage must NOT use subprocess chown/chmod — "
                f"use targeted os.chown instead. Found calls: {chown_chmod_calls}"
            )

    @patch("orchestration.worktree_manager.os.chown")
    @patch("orchestration.worktree_manager.subprocess.run")
    def test_repair_worktree_linkage_uses_targeted_os_chown(self, mock_subprocess_run, mock_os_chown):
        """Python os.chown() must be called on files written by Path.write_text()."""
        from orchestration.worktree_manager import WorktreeManager

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

            # os.chown should be called for files written: .git, gitdir, HEAD, commondir, and the dir
            assert mock_os_chown.call_count >= 1, (
                "repair_worktree_linkage must call os.chown() on files it writes "
                f"via Path.write_text(). Got {mock_os_chown.call_count} calls."
            )
            # Verify uid/gid 1000 in at least one call
            for c in mock_os_chown.call_args_list:
                assert c[0][1] == 1000 and c[0][2] == 1000, (
                    f"os.chown must use uid=1000, gid=1000. Got: {c}"
                )


# ---------------------------------------------------------------------------
# Task 11: No chown in cli_tool_manager.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCliToolManagerNoChown:
    """cli_tool_manager must NOT call subprocess chown on the worktree."""

    def test_cli_tool_manager_no_worktree_chown(self):
        """Workspace setup must not call subprocess with chown on worktree."""
        # Static analysis: read the source and check for subprocess chown patterns
        cli_tool_manager_path = (
            Path(__file__).parent.parent.parent / "orchestration" / "cli_tool_manager.py"
        )
        if not cli_tool_manager_path.exists():
            pytest.skip("cli_tool_manager.py not found")

        content = cli_tool_manager_path.read_text()

        # The specific pattern we're checking: subprocess.run(["chown", ...]) near worktree
        import re
        # Find all subprocess.run calls with chown
        chown_subprocess_pattern = re.compile(
            r'subprocess\.run\(\s*\[.*"chown".*\]',
            re.DOTALL
        )
        matches = chown_subprocess_pattern.findall(content)
        assert len(matches) == 0, (
            "cli_tool_manager.py must NOT call subprocess.run with chown — "
            f"git creates files as session_user via preexec_fn. Found: {matches}"
        )


# ---------------------------------------------------------------------------
# Commit SHA branch resolution (Default source mode)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateWorktreeCommitShaResolution:
    """create_worktree() must handle 40-char commit SHAs directly (no ls-remote)."""

    def test_commit_sha_uses_cat_file_not_ls_remote(self):
        """When branch is a 40-char hex SHA, use cat-file to verify it exists
        and skip ls-remote --heads (which only finds branch names)."""
        from orchestration.worktree_manager import WorktreeManager

        sha = "88d34c6409e9fb3c7b8ca0c04756f061d2099eb1"

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

            session_id = "test-sha-branch"

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True), \
                 patch.object(mgr, 'update_base_repo'), \
                 patch.object(mgr, 'get_repo_config', return_value={
                     'url': 'https://github.com/vllm-project/vllm.git',
                     'default_branch': 'main',
                 }):

                worktree_path = sessions_dir / session_id / "worktree"

                def side_effect_run_git(args, **kwargs):
                    if args == ["cat-file", "-t", "--end-of-options", sha]:
                        return MagicMock(stdout="commit\n", returncode=0)
                    if args[0] == "worktree" and args[1] == "add":
                        worktree_path.mkdir(parents=True, exist_ok=True)
                    return MagicMock(stdout="", returncode=0)

                mock_run_git.side_effect = side_effect_run_git
                mgr.create_worktree(session_id, "vllm", branch=sha)

                # Should have called cat-file with check=False to verify the SHA.
                # --end-of-options terminates option parsing so a dash-leading
                # value can never be treated as a git flag (cmdinj-1 hardening).
                cat_file_calls = [
                    c for c in mock_run_git.call_args_list
                    if c[0][0] == ["cat-file", "-t", "--end-of-options", sha]
                ]
                assert len(cat_file_calls) == 1, (
                    f"Expected cat-file -t call for SHA, got: {mock_run_git.call_args_list}"
                )
                assert cat_file_calls[0].kwargs.get("check") is False, (
                    "cat-file must use check=False so invalid SHAs fall back gracefully"
                )

                # Should NOT have called ls-remote --heads
                ls_remote_calls = [
                    c for c in mock_run_git.call_args_list
                    if len(c[0][0]) >= 2 and c[0][0][0] == "ls-remote"
                ]
                assert len(ls_remote_calls) == 0, (
                    f"Should not call ls-remote for commit SHA, got: {ls_remote_calls}"
                )

                # worktree add should use the SHA directly as commit
                worktree_add_calls = [
                    c for c in mock_run_git.call_args_list
                    if len(c[0][0]) >= 3 and c[0][0][0] == "worktree" and c[0][0][1] == "add"
                ]
                assert len(worktree_add_calls) == 1
                assert worktree_add_calls[0][0][0][-1] == sha, (
                    f"worktree add should use SHA directly, got: {worktree_add_calls[0][0][0]}"
                )

    def test_commit_sha_not_found_falls_back_to_default(self):
        """When SHA is not found via cat-file, fall back to default branch."""
        from orchestration.worktree_manager import WorktreeManager

        sha = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

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

            session_id = "test-sha-fallback"

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True), \
                 patch.object(mgr, 'update_base_repo'), \
                 patch.object(mgr, 'get_repo_config', return_value={
                     'url': 'https://github.com/vllm-project/vllm.git',
                     'default_branch': 'main',
                 }):

                worktree_path = sessions_dir / session_id / "worktree"

                def side_effect_run_git(args, **kwargs):
                    if args == ["cat-file", "-t", "--end-of-options", sha]:
                        return MagicMock(stdout="", returncode=128)
                    if args == ["ls-remote", "--heads", "origin", "main"]:
                        return MagicMock(stdout="abc123 refs/heads/main\n", returncode=0)
                    if args == ["rev-parse", "origin/main"]:
                        return MagicMock(stdout="abc123def456\n", returncode=0)
                    if args[0] == "worktree" and args[1] == "add":
                        worktree_path.mkdir(parents=True, exist_ok=True)
                    return MagicMock(stdout="", returncode=0)

                mock_run_git.side_effect = side_effect_run_git
                mgr.create_worktree(session_id, "vllm", branch=sha)

                # Should have fallen back to rev-parse origin/main
                rev_parse_calls = [
                    c for c in mock_run_git.call_args_list
                    if c[0][0] == ["rev-parse", "origin/main"]
                ]
                assert len(rev_parse_calls) == 1, (
                    "Should fall back to default branch when SHA not found"
                )

    def test_non_commit_object_type_falls_back_to_default(self):
        """When SHA resolves to a non-commit object (tag/tree/blob), fall back."""
        from orchestration.worktree_manager import WorktreeManager

        sha = "aabbccddaabbccddaabbccddaabbccddaabbccdd"

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

            session_id = "test-sha-tag-object"

            with patch.object(mgr, '_run_git') as mock_run_git, \
                 patch.object(mgr, 'is_repo_cloned', return_value=True), \
                 patch.object(mgr, 'update_base_repo'), \
                 patch.object(mgr, 'get_repo_config', return_value={
                     'url': 'https://github.com/vllm-project/vllm.git',
                     'default_branch': 'main',
                 }):

                worktree_path = sessions_dir / session_id / "worktree"

                def side_effect_run_git(args, **kwargs):
                    if args == ["cat-file", "-t", "--end-of-options", sha]:
                        # Object exists but is a tag object, not a commit
                        return MagicMock(stdout="tag\n", returncode=0)
                    if args == ["ls-remote", "--heads", "origin", "main"]:
                        return MagicMock(stdout="abc123 refs/heads/main\n", returncode=0)
                    if args == ["rev-parse", "origin/main"]:
                        return MagicMock(stdout="abc123def456\n", returncode=0)
                    if args[0] == "worktree" and args[1] == "add":
                        worktree_path.mkdir(parents=True, exist_ok=True)
                    return MagicMock(stdout="", returncode=0)

                mock_run_git.side_effect = side_effect_run_git
                mgr.create_worktree(session_id, "vllm", branch=sha)

                # Should have fallen back to rev-parse origin/main
                rev_parse_calls = [
                    c for c in mock_run_git.call_args_list
                    if c[0][0] == ["rev-parse", "origin/main"]
                ]
                assert len(rev_parse_calls) == 1, (
                    "Should fall back to default branch when SHA is not a commit object"
                )
