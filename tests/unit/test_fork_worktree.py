# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_worktree.py
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.worktree_manager import WorktreeManager, reset_worktree_manager


def _git(args, cwd):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
         "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, env=e)


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_create_worktree_from_fork_base(tmp_path, monkeypatch):
    # Build a fake fork base repo (already cloned, with a session-able commit).
    monkeypatch.setattr(WorktreeManager, "_drop_privileges", staticmethod(lambda: None))
    fork_base = tmp_path / "repos" / "forks" / "abc123" / "vllm"
    fork_base.mkdir(parents=True)
    _git(["init", "-b", "main", "."], fork_base)
    (fork_base / "setup.py").write_text("# vllm\n")
    _git(["add", "."], fork_base)
    _git(["commit", "-m", "init"], fork_base)
    # Simulate an 'origin' so origin/main resolves (worktree resolves origin/<branch>)
    _git(["remote", "add", "origin", str(fork_base)], fork_base)
    _git(["fetch", "origin"], fork_base)

    reset_worktree_manager()
    mgr = WorktreeManager(repos_dir=str(tmp_path / "repos"),
                          sessions_dir=str(tmp_path / "sessions"))
    wt = mgr.create_worktree(
        session_id="fork-sess-1", repo_name="vllm", branch="main",
        fork_base_path=fork_base,
    )
    assert Path(wt).exists()
    assert (Path(wt) / "setup.py").exists()
    # session branch checked out
    out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                         cwd=str(wt), capture_output=True, text=True)
    assert out.stdout.strip() == "session/fork-sess-1"
