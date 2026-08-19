# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_repo_manager.py
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.fork_repo_manager import ForkRepoManager


def _git(args, cwd, env=None):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
         "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if env:
        e.update(env)
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, env=e)


@pytest.fixture
def fake_fork(tmp_path):
    """A local git repo acting as the 'fork remote' with a feature branch."""
    origin = tmp_path / "fake_fork_origin"
    origin.mkdir()
    _git(["init", "-b", "main", "."], origin)
    (origin / "setup.py").write_text("# vllm\n")
    _git(["add", "."], origin)
    _git(["commit", "-m", "init"], origin)
    _git(["checkout", "-b", "feature-x"], origin)
    (origin / "feat.txt").write_text("x\n")
    _git(["add", "."], origin)
    _git(["commit", "-m", "feat"], origin)
    _git(["checkout", "main"], origin)
    return origin


@pytest.mark.unit
class TestForkRepoManager:
    def test_dir_for_url_is_stable_and_hashed(self, tmp_path):
        mgr = ForkRepoManager(repos_dir=str(tmp_path / "repos"))
        url = "https://github.com/u/vllm.git"
        p1 = mgr.fork_base_path(url)
        p2 = mgr.fork_base_path(url)
        assert p1 == p2
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert p1 == tmp_path / "repos" / "forks" / digest / "vllm"

    @pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
    def test_ensure_fork_base_clones_and_resolves_branch(self, tmp_path, fake_fork, monkeypatch):
        # No privilege drop in tests (we are not root): make _drop_privileges a no-op.
        monkeypatch.setattr(ForkRepoManager, "_preexec", staticmethod(lambda: None))
        mgr = ForkRepoManager(repos_dir=str(tmp_path / "repos"))
        base = mgr.ensure_fork_base(f"file://{fake_fork}", "feature-x", token=None)
        assert (base / ".git").exists()
        # feature-x must be resolvable as origin/feature-x
        out = subprocess.run(
            ["git", "rev-parse", "origin/feature-x"],
            cwd=str(base), capture_output=True, text=True,
        )
        assert out.returncode == 0 and out.stdout.strip()

    @pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
    def test_ensure_fork_base_idempotent_refetch(self, tmp_path, fake_fork, monkeypatch):
        monkeypatch.setattr(ForkRepoManager, "_preexec", staticmethod(lambda: None))
        mgr = ForkRepoManager(repos_dir=str(tmp_path / "repos"))
        base1 = mgr.ensure_fork_base(f"file://{fake_fork}", "main", token=None)
        base2 = mgr.ensure_fork_base(f"file://{fake_fork}", "main", token=None)
        assert base1 == base2  # reused, not re-cloned to a new path

    def test_remove_fork_base_takes_lock_then_removes(self, tmp_path, monkeypatch):
        """remove_fork_base takes the per-fork clone lock (LOCK_EX) before
        removing the vllm/ subdir, then drops the parent (incl. the lock file)
        — and tolerates a pre-existing .clone-lock without raising."""
        import fcntl as _fcntl

        import orchestration.fork_repo_manager as frm

        mgr = ForkRepoManager(repos_dir=str(tmp_path / "repos"))
        url = "https://github.com/u/vllm.git"
        base = mgr.fork_base_path(url)
        base_parent = base.parent

        # Fake a populated base repo + a pre-existing clone lock file.
        base.mkdir(parents=True, exist_ok=True)
        (base / ".git").mkdir()
        (base / "setup.py").write_text("# vllm\n")
        lock_path = mgr._lock_path(url)
        lock_path.write_text("")  # pre-existing lock file must not break removal

        # Spy on flock to prove an exclusive lock is taken, and capture whether
        # the base dir still exists at the moment the lock is acquired (it must,
        # so the removal happens UNDER the lock — not before it).
        flock_calls = []
        base_present_at_lock = {}
        real_flock = _fcntl.flock

        def spy_flock(fd, op):
            flock_calls.append(op)
            if op & _fcntl.LOCK_EX:
                base_present_at_lock["v"] = base.exists()
            return real_flock(fd, op)

        monkeypatch.setattr(frm.fcntl, "flock", spy_flock)

        # Should not raise even though a lock file already exists.
        mgr.remove_fork_base(url)

        # Exclusive lock was acquired, and the base still existed at that point.
        assert any(op & _fcntl.LOCK_EX for op in flock_calls), \
            "remove_fork_base must take an exclusive .clone-lock"
        assert base_present_at_lock.get("v") is True, \
            "base repo must still exist when the lock is taken (removed under lock)"

        # Both the repo dir and its parent (incl. the lock file) are gone.
        assert not base.exists()
        assert not base_parent.exists()
        assert not lock_path.exists()

    def test_remove_fork_base_missing_parent_is_noop(self, tmp_path):
        """Removing a fork that was never cloned is a quiet no-op."""
        mgr = ForkRepoManager(repos_dir=str(tmp_path / "repos"))
        # Never created — parent does not exist.
        mgr.remove_fork_base("https://github.com/u/never-cloned.git")  # no raise
