# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# orchestration/fork_repo_manager.py
"""Manage per-fork base repositories for custom vLLM fork sessions.

Each distinct fork URL is cloned once into a hash-keyed shared base repo at
   {repos_dir}/forks/{sha256(url)[:16]}/vllm
so that WorktreeManager can `git worktree add` session worktrees from it,
exactly as it does for the canonical upstream base. Private forks authenticate
via a GIT_ASKPASS helper fed the token through the GIT_FORK_TOKEN env var — the
token never appears in argv or on disk.
"""

import os
import fcntl
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from shared.session_models import (
    DEFAULT_SESSION_REPOS_DIR,
    ENV_SESSION_REPOS_DIR,
)

try:
    # FORK_REPOS_SUBDIR is added by the session-models task in this feature.
    from shared.session_models import FORK_REPOS_SUBDIR
except ImportError:  # pragma: no cover - parallel-task ordering fallback
    # Matches the value defined alongside SUPPORTED_REPOS in session_models.
    FORK_REPOS_SUBDIR = "forks"

logger = logging.getLogger(__name__)

# Static askpass helper shipped alongside the server source.
_ASKPASS_DOCKER = Path(
    "/app/scripts/git_askpass_helper.sh"
)
_ASKPASS_LOCAL = Path(__file__).parent.parent / "scripts" / "git_askpass_helper.sh"


class ForkRepoError(Exception):
    """Raised for fork base repo operations."""


class ForkRepoManager:
    """Clones/fetches user vLLM forks into hash-keyed shared base repos."""

    GIT_TIMEOUT = 600  # 10 min for the initial fork clone

    def __init__(self, repos_dir: Optional[str] = None):
        self.repos_dir = Path(
            repos_dir or os.getenv(ENV_SESSION_REPOS_DIR, DEFAULT_SESSION_REPOS_DIR)
        )
        self.forks_dir = self.repos_dir / FORK_REPOS_SUBDIR
        self.forks_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chown(str(self.forks_dir), 1000, 1000)
        except OSError:
            pass  # non-root dev/test

    @staticmethod
    def _preexec():
        """Drop to session_user (UID/GID 1000). Overridden to no-op in tests."""
        os.setgid(1000)
        os.setuid(1000)

    @staticmethod
    def _askpass_path() -> Optional[str]:
        if _ASKPASS_DOCKER.exists():
            return str(_ASKPASS_DOCKER)
        if _ASKPASS_LOCAL.exists():
            return str(_ASKPASS_LOCAL)
        return None

    def fork_base_path(self, fork_url: str) -> Path:
        """Stable hash-keyed base repo path for a fork URL."""
        digest = hashlib.sha256(fork_url.encode()).hexdigest()[:16]
        return self.forks_dir / digest / "vllm"

    def _lock_path(self, fork_url: str) -> Path:
        return self.fork_base_path(fork_url).parent / ".clone-lock"

    def _git_env(self, token: Optional[str]) -> dict:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        if token:
            askpass = self._askpass_path()
            if not askpass:
                raise ForkRepoError("git_askpass_helper.sh not found")
            env["GIT_ASKPASS"] = askpass
            env["GIT_FORK_TOKEN"] = token
        return env

    def _run_git(self, args: list, cwd: Optional[Path], token: Optional[str]):
        cmd = ["git", *args]
        try:
            result = subprocess.run(
                cmd, cwd=str(cwd) if cwd else None,
                check=True, capture_output=True, text=True,
                timeout=self.GIT_TIMEOUT,
                preexec_fn=self._preexec,
                env=self._git_env(token),
            )
            return result
        except subprocess.CalledProcessError as e:
            # Never echo stderr verbatim to the user-facing layer if a token was
            # used (defense in depth); the caller logs a sanitized message.
            raise ForkRepoError(f"git {args[0]} failed (exit {e.returncode})")
        except subprocess.TimeoutExpired:
            raise ForkRepoError(f"git {args[0]} timed out")

    def ensure_fork_base(
        self, fork_url: str, branch: str, token: Optional[str] = None
    ) -> Path:
        """Clone (or fetch) the fork into its hash-keyed base repo.

        Returns the base repo path. Serialized per-fork via fcntl so concurrent
        sessions on the same fork don't race the clone.
        """
        base = self.fork_base_path(fork_url)
        base.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chown(str(base.parent), 1000, 1000)
        except OSError:
            pass

        lock_path = self._lock_path(fork_url)
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if (base / ".git").exists():
                logger.info(f"Fork base exists, fetching: {base}")
                self._run_git(["fetch", "--all", "--prune"], cwd=base, token=token)
            else:
                logger.info(f"Cloning fork into {base}")
                self._run_git(
                    ["clone", fork_url, str(base)],
                    cwd=base.parent, token=token,
                )
                self._run_git(["fetch", "--all"], cwd=base, token=token)
            # Verify the requested branch/ref resolves; fail early with a clear msg.
            ref = branch
            probe = subprocess.run(
                ["git", "rev-parse", "--verify", "--end-of-options", f"origin/{branch}"],
                cwd=str(base), capture_output=True, text=True,
                preexec_fn=self._preexec, env=self._git_env(token),
            )
            if probe.returncode != 0:
                # maybe it's a tag or a raw SHA — verify generically
                probe2 = subprocess.run(
                    ["git", "cat-file", "-e", "--end-of-options", f"{branch}^{{commit}}"],
                    cwd=str(base), capture_output=True, text=True,
                    preexec_fn=self._preexec, env=self._git_env(token),
                )
                if probe2.returncode != 0:
                    raise ForkRepoError(
                        f"Branch/ref '{branch}' not found in fork"
                    )
        return base

    def remove_fork_base(self, fork_url: str) -> None:
        """Remove a fork's base repo (lock-aware, best-effort).

        Takes the per-fork ``.clone-lock`` (``LOCK_EX``) and removes the
        ``vllm/`` repo dir *under* the lock so a concurrent ``ensure_fork_base``
        (which holds this same lock while cloning) is never yanked out from
        under it. The parent dir (including the lock file itself) is dropped
        only after the lock is released.
        """
        base = self.fork_base_path(fork_url)
        base_parent = base.parent
        if not base_parent.exists():
            return
        lock_path = self._lock_path(fork_url)
        try:
            with open(lock_path, "w") as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                # Remove the repo dir while holding the lock so a concurrent
                # ensure_fork_base cannot be mid-clone into it.
                shutil.rmtree(base, ignore_errors=True)
            # Drop the parent (incl. the lock file) only after releasing the lock.
            shutil.rmtree(base_parent, ignore_errors=True)
        except OSError:
            shutil.rmtree(base_parent, ignore_errors=True)


# Singleton
_fork_repo_manager: Optional[ForkRepoManager] = None


def get_fork_repo_manager(repos_dir: Optional[str] = None) -> ForkRepoManager:
    global _fork_repo_manager
    if _fork_repo_manager is None:
        _fork_repo_manager = ForkRepoManager(repos_dir=repos_dir)
    return _fork_repo_manager


def reset_fork_repo_manager() -> None:
    global _fork_repo_manager
    _fork_repo_manager = None
