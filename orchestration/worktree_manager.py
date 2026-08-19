# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Git worktree manager for AI CLI sessions.

Manages base repository checkouts and creates isolated worktrees for each session.
Each session gets its own worktree directory for isolated development.
"""

import os
import re
import subprocess
import shutil
import logging
import asyncio
import time
import fcntl
from typing import Optional, Dict, Any
from pathlib import Path

from shared.session_models import (
    SUPPORTED_REPOS,
    DEFAULT_SESSION_REPOS_DIR,
    DEFAULT_SESSION_DATA_DIR,
    ENV_SESSION_REPOS_DIR,
    ENV_SESSION_DATA_DIR,
)

logger = logging.getLogger(__name__)

# Editable-install timeouts. Precompiled wheel install is fast; a full source
# build (custom fork) can take 15-20 min cold, ~1-3 min with warm ccache.
PRECOMPILED_INSTALL_TIMEOUT = 240
SOURCE_BUILD_TIMEOUT = 1500  # 25 min ceiling for a cold fork source build

# Cap the in-flight fork build log so a runaway nvcc/cmake build cannot fill
# the data volume (correctness-6). Once exceeded mid-build we kill the process
# and fail the build rather than letting the log grow unbounded.
FORK_BUILD_LOG_MAX_BYTES = 64 * 1024 * 1024  # 64 MB


class WorktreeError(Exception):
    """Exception raised for worktree operations."""
    pass


class WorktreeManager:
    """
    Manages git worktrees for AI CLI sessions.

    Each session gets an isolated worktree from a base repository.
    Changes in the worktree don't affect the base repository.
    """

    def __init__(
        self,
        repos_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
    ):
        """
        Initialize worktree manager.

        Args:
            repos_dir: Directory for base repository clones
            sessions_dir: Directory for session data including worktrees
        """
        self.repos_dir = Path(repos_dir or os.getenv(ENV_SESSION_REPOS_DIR, DEFAULT_SESSION_REPOS_DIR))
        self.sessions_dir = Path(sessions_dir or os.getenv(ENV_SESSION_DATA_DIR, DEFAULT_SESSION_DATA_DIR))

        # Ensure directories exist
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WorktreeManager initialized: repos={self.repos_dir}, sessions={self.sessions_dir}")

    @staticmethod
    def _drop_privileges():
        """Drop to session_user (UID/GID 1000) before exec.

        Called as preexec_fn in subprocess.run so git commands run as
        session_user rather than root.  setgid must precede setuid because
        once uid is changed the process may lack permission to change gid.
        """
        os.setgid(1000)
        os.setuid(1000)

    def _run_git(
        self,
        args: list,
        cwd: Optional[Path] = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a git command as session_user (UID 1000).

        Args:
            args: Git arguments (without 'git' prefix)
            cwd: Working directory
            check: Raise exception on non-zero exit
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        cmd = ["git"] + args
        logger.debug(f"Running: {' '.join(cmd)} in {cwd or '.'}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=check,
                capture_output=capture_output,
                text=True,
                timeout=600,  # 10 minute timeout for git operations
                preexec_fn=self._drop_privileges,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            raise WorktreeError(f"Git command failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Git command timed out: {' '.join(cmd)}")

    def get_repo_config(self, repo_name: str) -> Dict[str, Any]:
        """
        Get configuration for a repository.

        Args:
            repo_name: Repository identifier (e.g., 'vllm')

        Returns:
            Repository configuration dict

        Raises:
            WorktreeError: If repository not configured
        """
        if repo_name not in SUPPORTED_REPOS:
            raise WorktreeError(
                f"Unsupported repository: {repo_name}. "
                f"Supported: {list(SUPPORTED_REPOS.keys())}"
            )
        return SUPPORTED_REPOS[repo_name]

    def _ensure_repo_owned_by_session_user(self, repo_path: Path) -> None:
        """Ensure base repo is owned by session_user (UID 1000).

        Migration path: repos cloned before the git-as-session_user refactor
        are root-owned.  This one-time chown makes them writable by session_user
        so that _run_git() (which drops to UID 1000) can operate on them.
        """
        git_dir = repo_path / ".git"
        try:
            st = git_dir.stat()
            if st.st_uid != 1000:
                logger.info(
                    f"Migrating repo ownership to session_user: {repo_path} "
                    f"(current uid={st.st_uid})"
                )
                subprocess.run(
                    ["chown", "-R", "1000:1000", str(repo_path)],
                    capture_output=True, timeout=120,
                )
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to migrate repo ownership: {e}")

    def get_base_repo_path(self, repo_name: str) -> Path:
        """Get path to base repository clone."""
        return self.repos_dir / repo_name

    def is_repo_cloned(self, repo_name: str) -> bool:
        """Check if base repository is already cloned."""
        repo_path = self.get_base_repo_path(repo_name)
        return (repo_path / ".git").exists()

    def clone_base_repo(self, repo_name: str, force: bool = False) -> Path:
        """
        Clone the base repository if not already cloned.

        Args:
            repo_name: Repository identifier
            force: Force re-clone even if exists

        Returns:
            Path to cloned repository
        """
        config = self.get_repo_config(repo_name)
        repo_path = self.get_base_repo_path(repo_name)

        if repo_path.exists() and not force:
            if self.is_repo_cloned(repo_name):
                logger.info(f"Base repo {repo_name} already cloned at {repo_path}")
                # Migration: ensure repo is owned by session_user.
                # Repos cloned before the git-as-session_user refactor are root-owned.
                self._ensure_repo_owned_by_session_user(repo_path)
                return repo_path

        # Remove existing if force
        if repo_path.exists() and force:
            logger.info(f"Removing existing repo at {repo_path}")
            shutil.rmtree(repo_path)

        # Clone repository
        logger.info(f"Cloning {repo_name} from {config['url']}")
        self._run_git(
            ["clone", config["url"], str(repo_path)],
            cwd=self.repos_dir,
        )

        # Fetch all branches
        self._run_git(["fetch", "--all"], cwd=repo_path)

        # No chown needed — _run_git() drops to session_user via preexec_fn,
        # so all git-created files are already owned by 1000:1000.

        logger.info(f"Successfully cloned {repo_name} to {repo_path}")
        return repo_path

    def update_base_repo(self, repo_name: str) -> None:
        """
        Update the base repository with latest changes.

        Args:
            repo_name: Repository identifier
        """
        repo_path = self.get_base_repo_path(repo_name)

        if not self.is_repo_cloned(repo_name):
            raise WorktreeError(f"Base repo {repo_name} not cloned")

        logger.info(f"Updating base repo {repo_name}")
        self._run_git(["fetch", "--all", "--prune"], cwd=repo_path)

        # No chown/chmod needed after fetch — _run_git() drops to session_user
        # via preexec_fn, so fetch doesn't reset ownership.

    def get_session_dir(self, session_id: str) -> Path:
        """Get base directory for a session."""
        return self.sessions_dir / session_id

    def get_worktree_path(self, session_id: str) -> Path:
        """Get worktree path for a session."""
        return self.get_session_dir(session_id) / "worktree"

    def get_logs_dir(self, session_id: str) -> Path:
        """Get logs directory for a session."""
        return self.get_session_dir(session_id) / "logs"

    def create_session_dirs(self, session_id: str) -> Dict[str, Path]:
        """
        Create session directory structure.

        Args:
            session_id: Session identifier

        Returns:
            Dict with paths to session directories
        """
        session_dir = self.get_session_dir(session_id)
        logs_dir = self.get_logs_dir(session_id)

        session_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Chown to session_user so git (running as session_user via preexec_fn)
        # can create files inside the session directory (e.g., worktree/.git).
        try:
            os.chown(str(session_dir), 1000, 1000)
            os.chown(str(logs_dir), 1000, 1000)
        except OSError:
            pass  # Non-root environments (dev/test)

        return {
            "session_dir": session_dir,
            "logs_dir": logs_dir,
        }

    def create_worktree(
        self,
        session_id: str,
        repo_name: str,
        branch: str = "main",
        fork_base_path: Optional[Path] = None,
    ) -> Path:
        """
        Create a git worktree for a session.

        Args:
            session_id: Session identifier
            repo_name: Repository to create worktree from
            branch: Branch to checkout in worktree
            fork_base_path: Explicit base repo for custom-fork sessions (already
                cloned + fetched by ForkRepoManager); when set, skips the
                upstream SUPPORTED_REPOS clone/update path.

        Returns:
            Path to created worktree
        """
        is_fork = fork_base_path is not None

        if is_fork:
            # Fork base repo was already cloned + fetched by ForkRepoManager.
            base_repo_path = Path(fork_base_path)
            self._ensure_repo_owned_by_session_user(base_repo_path)
            # Forks have no SUPPORTED_REPOS entry — default ref fallback is 'main'.
            config = {"default_branch": "main"}
        else:
            # Ensure base repo exists
            if not self.is_repo_cloned(repo_name):
                self.clone_base_repo(repo_name)
            else:
                # Migration: ensure existing repo is owned by session_user
                self._ensure_repo_owned_by_session_user(self.get_base_repo_path(repo_name))
            base_repo_path = self.get_base_repo_path(repo_name)
            config = self.get_repo_config(repo_name)

        worktree_path = self.get_worktree_path(session_id)

        # Create session directories
        self.create_session_dirs(session_id)

        # Check if worktree already exists
        if worktree_path.exists():
            logger.warning(f"Worktree already exists at {worktree_path}, removing")
            self.remove_worktree(session_id, repo_name, base_repo_path=base_repo_path)

        # Update base repo before creating worktree (upstream only — fork already fetched)
        if not is_fork:
            try:
                self.update_base_repo(repo_name)
            except WorktreeError as e:
                logger.warning(f"Failed to update base repo: {e}")

        # Resolve the target commit — branch name or raw SHA
        is_commit_sha = bool(re.match(r'^[0-9a-f]{40}$', branch, re.IGNORECASE))

        if is_commit_sha:
            # Direct commit SHA (e.g. from "Default" source mode with pinned release).
            # Verify it exists locally after fetch.
            result = self._run_git(
                ["cat-file", "-t", "--end-of-options", branch],
                cwd=base_repo_path,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip() == "commit":
                commit = branch
            else:
                logger.warning(
                    f"Commit {branch[:12]} not found in local repo, "
                    f"using {config['default_branch']}"
                )
                branch = config['default_branch']
                is_commit_sha = False
                commit = None
        else:
            commit = None

        if not is_commit_sha:
            # Branch name — check remote, then resolve to commit
            try:
                result = self._run_git(
                    ["ls-remote", "--heads", "origin", "--end-of-options", branch],
                    cwd=base_repo_path,
                    check=False,
                )
                if not result.stdout.strip():
                    logger.warning(f"Branch {branch} not found, using {config['default_branch']}")
                    branch = config['default_branch']
            except Exception:
                branch = config['default_branch']

            # NOTE: plain `git rev-parse` (no --verify) ECHOES unrecognized
            # non-revision tokens (incl. --end-of-options) into stdout, which
            # corrupts the captured commit SHA. So we cannot add --end-of-options
            # here. The Pydantic branch validator already blocks dash-leading
            # refs at the API boundary, so option-injection is covered.
            result = self._run_git(
                ["rev-parse", f"origin/{branch}"],
                cwd=base_repo_path,
            )
            commit = result.stdout.strip()

        # Create worktree with detached HEAD from the resolved commit
        logger.info(f"Creating worktree for session {session_id} from {repo_name}:{branch[:12] if is_commit_sha else branch}")

        # Create worktree at specific commit (detached HEAD)
        # Use file lock for concurrency safety (git worktree add is not thread-safe)
        lock_path = base_repo_path / ".claude" / "worktrees" / ".create-lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._run_git(
                ["worktree", "add", "--detach", str(worktree_path), commit],
                cwd=base_repo_path,
            )

        # No chown/chmod needed — _run_git() drops to session_user via preexec_fn,
        # so git worktree add creates all files as session_user (1000:1000).
        # The flock at .claude/worktrees/.create-lock is held by root (server
        # process) — this is fine, session_user doesn't need to write the lock file.

        # Create a local branch for the session so changes can be committed
        session_branch = f"session/{session_id}"
        self._run_git(
            ["checkout", "-b", session_branch],
            cwd=worktree_path,
        )

        logger.info(f"Created worktree at {worktree_path} on branch {session_branch}")
        return worktree_path

    def remove_worktree(
        self,
        session_id: str,
        repo_name: str,
        force: bool = True,
        base_repo_path: Optional[Path] = None,
    ) -> None:
        """Remove a session's worktree.

        Args:
            session_id: Session identifier
            repo_name: Repository the worktree belongs to
            force: Force removal even if untracked files
            base_repo_path: Explicit base repo (fork sessions); defaults to the
                upstream base repo path for repo_name.
        """
        base_repo_path = Path(base_repo_path) if base_repo_path else self.get_base_repo_path(repo_name)
        worktree_path = self.get_worktree_path(session_id)

        if not worktree_path.exists():
            logger.info(f"Worktree {worktree_path} doesn't exist, nothing to remove")
            return

        logger.info(f"Removing worktree at {worktree_path}")

        # Remove worktree via git
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(str(worktree_path))

            self._run_git(args, cwd=base_repo_path, check=False)
        except Exception as e:
            logger.warning(f"Git worktree remove failed: {e}, falling back to rm -rf")

        # Fallback: remove directory directly
        if worktree_path.exists():
            shutil.rmtree(worktree_path)

        # Prune worktree references
        try:
            self._run_git(["worktree", "prune"], cwd=base_repo_path, check=False)
        except Exception:
            pass

        # Delete the session branch ref to prevent accumulation in the base repo
        session_branch = f"session/{session_id}"
        self._run_git(["branch", "-D", session_branch], cwd=base_repo_path, check=False)

        logger.info(f"Removed worktree for session {session_id}")

    def repair_worktree_linkage(
        self,
        session_id: str,
        worktree_path: Path,
        repo_name: str,
        branch: str = "main",
    ) -> None:
        """
        Repair git worktree linkage after cross-host S3 restore.

        After restoring a worktree from S3 on a different host/path, the .git
        file (if it exists) may point to the original host's base repo path,
        which doesn't exist locally. This method repairs the linkage by:
        1. Creating or updating the .git file with the correct gitdir path
        2. Creating the corresponding worktree entry in the base repo's
           .git/worktrees/{session_id}/ directory

        This is a no-op if the linkage is already valid.

        Args:
            session_id: Session identifier
            worktree_path: Path to the restored worktree directory
            repo_name: Repository name (e.g., 'vllm')
            branch: Branch the session was on
        """
        worktree_path = Path(worktree_path)
        base_repo_path = self.get_base_repo_path(repo_name)
        base_git_dir = base_repo_path / ".git"

        if not base_git_dir.exists():
            logger.warning(
                f"Session {session_id}: Cannot repair worktree linkage - "
                f"base repo .git not found at {base_git_dir}"
            )
            return

        expected_gitdir = str(base_git_dir / "worktrees" / session_id)
        git_file = worktree_path / ".git"
        needs_repair = False

        # Check if .git file exists and points to the correct location
        if git_file.exists() and git_file.is_file():
            content = git_file.read_text().strip()
            if content == f"gitdir: {expected_gitdir}":
                # Also verify the worktree entry exists in the base repo
                # AND that its gitdir content points to the correct path
                wt_entry = base_git_dir / "worktrees" / session_id
                if wt_entry.exists() and (wt_entry / "gitdir").exists():
                    wt_gitdir_content = (wt_entry / "gitdir").read_text().strip()
                    if wt_gitdir_content == str(worktree_path / ".git"):
                        logger.debug(
                            f"Session {session_id}: Worktree linkage is valid, no repair needed"
                        )
                        return
            needs_repair = True
        elif git_file.exists() and git_file.is_dir():
            # .git is a full directory (not a worktree), no repair needed
            logger.debug(
                f"Session {session_id}: .git is a directory (not a worktree file), skipping repair"
            )
            return
        else:
            # .git file missing entirely
            needs_repair = True

        if not needs_repair:
            return

        logger.info(
            f"Session {session_id}: Repairing worktree linkage "
            f"(gitdir -> {expected_gitdir})"
        )

        # 1. Write the .git file with the correct gitdir path
        git_file.write_text(f"gitdir: {expected_gitdir}\n")
        os.chown(str(git_file), 1000, 1000)

        # 2. Create the worktree entry in the base repo
        wt_entry_dir = base_git_dir / "worktrees" / session_id
        wt_entry_dir.mkdir(parents=True, exist_ok=True)
        os.chown(str(wt_entry_dir), 1000, 1000)

        # gitdir: points back to the worktree's .git file location
        gitdir_file = wt_entry_dir / "gitdir"
        gitdir_file.write_text(str(worktree_path / ".git"))
        os.chown(str(gitdir_file), 1000, 1000)

        # HEAD: use the session branch
        session_branch = f"session/{session_id}"
        head_file = wt_entry_dir / "HEAD"
        head_file.write_text(f"ref: refs/heads/{session_branch}\n")
        os.chown(str(head_file), 1000, 1000)

        # commondir: relative path from worktrees/{session_id}/ to the base .git/
        commondir_file = wt_entry_dir / "commondir"
        commondir_file.write_text("../..")
        os.chown(str(commondir_file), 1000, 1000)

        # Also fix parent .git/worktrees/ directory ownership (targeted, no subprocess)
        worktrees_parent = base_git_dir / "worktrees"
        if worktrees_parent.exists():
            os.chown(str(worktrees_parent), 1000, 1000)

        logger.info(
            f"Session {session_id}: Worktree linkage repaired successfully"
        )

    def cleanup_session(self, session_id: str, repo_name: str) -> None:
        """
        Clean up all session data including worktree.

        Args:
            session_id: Session identifier
            repo_name: Repository the session uses
        """
        # Remove worktree first
        self.remove_worktree(session_id, repo_name)

        # Remove session directory
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            logger.info(f"Removing session directory {session_dir}")
            shutil.rmtree(session_dir)

    def ensure_base_repos(self) -> Dict[str, bool]:
        """
        Ensure all supported base repositories are cloned.

        Returns:
            Dict mapping repo name to success status
        """
        results = {}
        for repo_name in SUPPORTED_REPOS:
            try:
                self.clone_base_repo(repo_name)
                results[repo_name] = True
            except Exception as e:
                logger.error(f"Failed to clone {repo_name}: {e}")
                results[repo_name] = False
        return results

    # --- vLLM Development Environment Methods ---

    def _resolve_wheel_commit(self, branch: str) -> str:
        """Choose VLLM_PRECOMPILED_WHEEL_COMMIT for the editable install.

        setup.py downloads precompiled .so wheels from wheels.vllm.ai keyed by
        commit. A wheel only exists once the nightly build for that commit has
        published (~build lag after merge), so resolving the *live* main HEAD
        races publication and 404s — which leaves vLLM uninstalled and the
        vLLM server unable to launch. Policy:

          - explicit 40-hex SHA  → pass through. A pinned commit ("Default"
            source mode or a user-typed hash) is old enough that its wheel
            exists.
          - default branch ("main") → pin to the image's build-time
            wheel-verified commit (<VLLM_BASE_REPO>/.docker_commit), which the
            Dockerfile already curl-verified at build time. This guarantees a
            published wheel and matches the baked image. Falls back to "" for
            legacy images without .docker_commit (preserving setup.py
            self-resolution).
          - any other branch/tag → "" (unchanged: setup.py resolves). A
            user-selected branch is never silently swapped to the baked
            binaries.
        """
        if re.match(r'^[0-9a-f]{40}$', branch, re.IGNORECASE):
            return branch

        default_branch = self.get_repo_config("vllm").get("default_branch", "main")
        if branch != default_branch:
            return ""

        # Default branch → pin to the baked, build-verified commit.
        try:
            base_repo = Path(os.getenv("VLLM_BASE_REPO", "/workspace/vllm"))
            commit_file = base_repo / ".docker_commit"
            if commit_file.exists():
                baked = commit_file.read_text().strip()
                if re.match(r'^[0-9a-f]{40}$', baked, re.IGNORECASE):
                    logger.info(
                        f"Pinning vLLM precompiled wheel to baked commit "
                        f"{baked[:12]} for default-branch session"
                    )
                    return baked
                logger.warning(
                    f".docker_commit is not a 40-char SHA ({baked!r}); "
                    f"falling back to setup.py wheel resolution"
                )
        except OSError as e:
            logger.warning(f"Could not read .docker_commit: {e}")
        return ""

    @staticmethod
    def _cutlass_cu13_repair_command(
        venv: Path, uv_path: str, python: str
    ) -> Optional[list]:
        """Return a uv command to repair the nvidia-cutlass-dsl[cu13] overlap, or None.

        This works around an UPSTREAM PACKAGING DEFECT in ``nvidia-cutlass-dsl``
        — not anything vLLM or this repo configures. vLLM >=0.22.1 declares
        ``nvidia-cutlass-dsl[cu13]==4.5.2`` in ``requirements/cuda.txt`` ([cu13]
        is upstream's intended default on CUDA 13, stripped only on CUDA 12 in
        vLLM's setup.py). The ``[cu13]`` extra installs TWO overlapping sub-wheels
        — ``nvidia-cutlass-dsl-libs-base`` and ``nvidia-cutlass-dsl-libs-cu13`` —
        that both claim the same ~99 ``cutlass/cute/*.py`` paths with DIFFERENT
        content. Two wheels owning the same files is malformed packaging: pip/uv
        have no defined winner, so the result is install-order-dependent. When
        base lands last it clobbers the cu13 variants, crashing FlashAttention-4
        on Blackwell (NVFP4). (The Docker image venv happens to be clean only
        because cu13 lands last there by luck; a fresh session venv can resolve
        the other way and break.)

        When both sub-wheels are present we force-reinstall the cu13 libs wheel
        LAST (``--no-deps`` so only that wheel is touched) so the cu13 variants
        deterministically win the overlap. The pinned version is read from the
        on-disk cu13 ``*.dist-info`` dir name rather than hardcoded, so it tracks
        whatever version the editable install resolved. This self-disables once
        upstream fixes the overlap (the sub-wheel names/overlap go away → None).

        Returns None (no-op) when the overlap cannot occur — i.e. the base
        sub-wheel is absent, or the cu13 sub-wheel is absent, or neither is
        installed (a non-cu130 install).
        """
        import glob as _glob

        sp_glob = str(venv / "lib" / "python*" / "site-packages")
        base_hits = _glob.glob(
            os.path.join(sp_glob, "nvidia_cutlass_dsl_libs_base-*.dist-info")
        )
        cu13_hits = _glob.glob(
            os.path.join(sp_glob, "nvidia_cutlass_dsl_libs_cu13-*.dist-info")
        )
        if not base_hits or not cu13_hits:
            return None

        # Derive the version from the cu13 dist-info dir name:
        #   nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info -> 4.5.2
        dirname = os.path.basename(cu13_hits[0])
        stem = dirname[: -len(".dist-info")]
        version = stem.split("-", 1)[1] if "-" in stem else ""
        if not version:
            return None

        return [
            uv_path, "pip", "install",
            "--force-reinstall", "--no-deps",
            "--python", python,
            f"nvidia-cutlass-dsl-libs-cu13=={version}",
        ]

    async def initialize_vllm_environment(
        self,
        session_id: str,
        branch: str = "main",
        precompiled: bool = True,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install vLLM with precompiled binaries. No cmake, no hardlinks.

        Creates a fresh venv with uv, then runs editable install with
        VLLM_USE_PRECOMPILED=1 so setup.py downloads pre-built .so files
        instead of triggering cmake.

        VLLM_PRECOMPILED_WHEEL_COMMIT pins which prebuilt wheel setup.py
        downloads: an explicit 40-char SHA passes through; the default branch
        pins to the image's build-verified baked commit (.docker_commit) to
        avoid the nightly-wheel publish race; other branches let setup.py
        resolve.  See _resolve_wheel_commit().

        Returns:
            {"status": "success", "timings": {"venv_create": float,
                                              "editable_install": float,
                                              "total": float}}
        """
        worktree_path = self.get_worktree_path(session_id)
        dst_venv = worktree_path / ".venv"
        python = str(dst_venv / "bin" / "python")
        uv_path = shutil.which("uv")
        if not uv_path:
            raise WorktreeError("uv not found on PATH")
        timings: Dict[str, float] = {}
        start = time.time()

        # 1. Create fresh venv
        t0 = time.time()
        # start_new_session=True on every session-scoped spawn: the pause/
        # terminate /proc sweep (session_manager._scan_session_processes) skips
        # pids in the server's own pgid, so a build left in our pgid would
        # survive pause and race worktree cleanup.
        proc = await asyncio.create_subprocess_exec(
            uv_path, "venv", "--python", "3.12", str(dst_venv),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise WorktreeError(f"uv venv failed: {stderr.decode()}")
        timings["venv_create"] = round(time.time() - t0, 2)

        # 2. Editable install.
        #    precompiled=True  → download prebuilt .so wheels (fast, upstream).
        #    precompiled=False → full source build for the fork's own C++/CUDA
        #                        (VLLM_USE_PRECOMPILED=0; setup.py runs cmake).
        t0 = time.time()
        if precompiled:
            wheel_commit = self._resolve_wheel_commit(branch)
            env = {
                **os.environ,
                "VLLM_USE_PRECOMPILED": "1",
                "VLLM_PRECOMPILED_WHEEL_COMMIT": wheel_commit,
            }
            timeout = PRECOMPILED_INSTALL_TIMEOUT
        else:
            env = {
                **os.environ,
                "VLLM_USE_PRECOMPILED": "0",
            }
            # Drop any inherited precompiled-commit pin so setup.py builds source.
            env.pop("VLLM_PRECOMPILED_WHEEL_COMMIT", None)
            timeout = SOURCE_BUILD_TIMEOUT

        # Stream build output to a log file (tailed by the build console
        # terminal) when log_path is given; otherwise capture to PIPE.
        log_fh = open(log_path, "ab", buffering=0) if log_path else None
        try:
            proc = await asyncio.create_subprocess_exec(
                uv_path, "pip", "install", "-e", ".",
                "--torch-backend", "auto",
                "--python", python,
                cwd=str(worktree_path),
                stdout=(log_fh if log_fh else asyncio.subprocess.PIPE),
                stderr=(log_fh if log_fh else asyncio.subprocess.PIPE),
                env=env,
                start_new_session=True,
            )
            try:
                if log_fh:
                    # Poll for completion while capping log growth. The outer
                    # `timeout` (SOURCE_BUILD_TIMEOUT / PRECOMPILED_INSTALL_TIMEOUT)
                    # is preserved as a hard deadline across the poll loop.
                    start = time.time()
                    while True:
                        if time.time() - start > timeout:
                            proc.kill()
                            raise WorktreeError(
                                f"vLLM install timed out after {timeout}s"
                            )
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=10)
                            break
                        except asyncio.TimeoutError:
                            try:
                                if log_path and os.path.getsize(log_path) > FORK_BUILD_LOG_MAX_BYTES:
                                    with open(log_path, "ab") as _cap:
                                        _cap.write(b"\n[log truncated: exceeded 64MB cap]\n")
                                    # Hard-stop a runaway-output build.
                                    proc.kill()
                                    raise WorktreeError(
                                        "vLLM build log exceeded 64MB cap (runaway output)"
                                    )
                            except OSError:
                                pass
                    stderr = b""
                else:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise WorktreeError(f"vLLM install timed out after {timeout}s")
        finally:
            if log_fh:
                log_fh.close()
        if proc.returncode != 0:
            detail = stderr.decode()[:200] if stderr else f"see {log_path}"
            raise WorktreeError(f"vLLM install failed: {detail}")
        timings["editable_install"] = round(time.time() - t0, 2)

        # 2b. Repair the nvidia-cutlass-dsl[cu13] sub-wheel overlap.
        #     The cu130 wheel's [cu13] extra ships two overlapping sub-wheels
        #     (libs-base + libs-cu13) that write the same ~99 cutlass/cute/*.py
        #     paths; when base lands last it clobbers the cu13 variants and
        #     crashes FlashAttention-4 on Blackwell. Force-reinstall the cu13
        #     libs wheel LAST so it deterministically wins the overlap. This is
        #     best-effort: a repair failure must not abort an otherwise-good
        #     install (the env is still usable on non-Blackwell / non-FA4 paths),
        #     so we log loudly and continue.
        t0 = time.time()
        repair_cmd = self._cutlass_cu13_repair_command(dst_venv, uv_path, python)
        if repair_cmd is not None:
            logger.info(
                f"Repairing nvidia-cutlass-dsl[cu13] overlap for {session_id}: "
                f"{' '.join(repair_cmd)}"
            )
            try:
                repair_proc = await asyncio.create_subprocess_exec(
                    *repair_cmd,
                    cwd=str(worktree_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    start_new_session=True,
                )
                _, repair_stderr = await asyncio.wait_for(
                    repair_proc.communicate(), timeout=PRECOMPILED_INSTALL_TIMEOUT
                )
                if repair_proc.returncode != 0:
                    logger.warning(
                        f"cutlass-dsl cu13 repair failed (non-fatal) for "
                        f"{session_id}: {repair_stderr.decode()[:300]}"
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"cutlass-dsl cu13 repair timed out (non-fatal) for {session_id}"
                )
        timings["cutlass_repair"] = round(time.time() - t0, 2)

        # 3. Copy CMakeUserPresets.json for on-demand C++ builds
        base_repo = Path(os.getenv("VLLM_BASE_REPO", "/workspace/vllm"))
        src_presets = base_repo / "CMakeUserPresets.json"
        dst_presets = worktree_path / "CMakeUserPresets.json"
        if src_presets.exists():
            shutil.copy2(src_presets, dst_presets)
            patch_script = Path("/app/scripts/patch_cmake_presets.py")
            if not patch_script.exists():
                patch_script = Path(__file__).parent.parent / "scripts" / "patch_cmake_presets.py"
            if patch_script.exists() and dst_presets.exists():
                python_path = str(dst_venv / "bin" / "python") if (dst_venv / "bin" / "python").exists() else "python3"
                cmd = [python_path, str(patch_script), "--patch-only", "--auto-detect",
                       "--presets-path", str(dst_presets),
                       "--python-path", str(dst_venv / "bin" / "python")]
                patch_proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                _, patch_stderr = await patch_proc.communicate()
                if patch_proc.returncode == 0:
                    logger.info("Patched CMakeUserPresets.json with GPU-specific settings")
                else:
                    logger.warning(f"Failed to patch CMakeUserPresets.json: {patch_stderr.decode()}")

        timings["total"] = round(time.time() - start, 2)
        logger.info(f"vLLM env initialized for {session_id} in {timings['total']}s")
        return {"status": "success", "timings": timings}


# Singleton instance
_worktree_manager: Optional[WorktreeManager] = None


def get_worktree_manager(
    repos_dir: Optional[str] = None,
    sessions_dir: Optional[str] = None,
) -> WorktreeManager:
    """
    Get singleton worktree manager instance.

    Args:
        repos_dir: Directory for base repos (only used on first call)
        sessions_dir: Directory for sessions (only used on first call)

    Returns:
        WorktreeManager instance
    """
    global _worktree_manager
    if _worktree_manager is None:
        _worktree_manager = WorktreeManager(
            repos_dir=repos_dir,
            sessions_dir=sessions_dir,
        )
    return _worktree_manager


def reset_worktree_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _worktree_manager
    _worktree_manager = None
