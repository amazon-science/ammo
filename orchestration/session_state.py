# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Session State Persistence for AI CLI sessions.

Handles S3 storage for cross-node session resume using AWS CLI:
- Session metadata sync to S3 (via aws s3 cp)
- Worktree directory sync (via aws s3 sync)
- CLI tool state sync (via aws s3 sync)
"""

import asyncio
import json
import logging
import os
import shlex
import shutil
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta

from shared.session_models import (
    SessionState,
    SessionStatus,
    ENV_SESSION_S3_BUCKET,
    ENV_SESSION_S3_PREFIX,
    ENV_SESSION_S3_TTL_DAYS,
    DEFAULT_SESSION_S3_PREFIX,
    DEFAULT_SESSION_S3_TTL_DAYS,
)

logger = logging.getLogger(__name__)


DEFAULT_TAR_EXCLUDE_PATTERNS = [
    "__pycache__", "*.pyc",
    ".pytest_cache", "*.egg-info",
    ".mypy_cache", "node_modules",
    "venv", ".venv",
    "cmake-build-*", "build",
    "*.so", "*.a", "*.o",
    "CMakeCache.txt", "CMakeFiles",
    "download", ".deps",
]

REQUIRED_TAR_EXCLUDE_PATTERNS = [
    "codex-home/auth.json",
]


class SessionS3Storage:
    """
    S3 storage for session state.

    Uses `aws s3 sync` (AWS CLI v2 with CRT library) for fast, incremental uploads.
    This provides significant performance improvements:
    - Parallel uploads using CRT library
    - Incremental sync (only changed files uploaded)
    - No CPU-bound compression step

    S3 Structure:
    - session.json: Session metadata (uploaded via aws s3 cp)
    - worktree/: Worktree directory (synced via aws s3 sync)
    - claude_state/: AI CLI state (synced via aws s3 sync)
    """

    def __init__(self):
        self.bucket = os.getenv(ENV_SESSION_S3_BUCKET)
        self.prefix = os.getenv(ENV_SESSION_S3_PREFIX, DEFAULT_SESSION_S3_PREFIX)
        self.ttl_days = int(os.getenv(ENV_SESSION_S3_TTL_DAYS, DEFAULT_SESSION_S3_TTL_DAYS))

        if self.bucket:
            logger.info(f"Session S3 storage configured: bucket={self.bucket}, prefix={self.prefix}")
        else:
            logger.info("Session S3 storage not configured - using local storage only")

    @property
    def enabled(self) -> bool:
        """Check if S3 storage is configured."""
        return self.bucket is not None

    def _key(self, session_id: str, filename: str) -> str:
        """Generate S3 object key for a session file."""
        return f"{self.prefix}/{session_id}/{filename}"

    # ========================================================================
    # Tar+pigz helper methods
    # ========================================================================

    def _get_nproc(self) -> int:
        """Get number of CPU cores for parallel compression."""
        return os.cpu_count() or 4

    def _get_compressor(self) -> tuple:
        """Get the compression tool and args. Prefers pigz, falls back to gzip."""
        nproc = self._get_nproc()
        if shutil.which("pigz"):
            return "pigz", ["-1", "-p", str(nproc)]
        else:
            logger.warning("pigz not found, falling back to gzip (slower)")
            return "gzip", ["-1"]

    def _get_decompressor(self) -> tuple:
        """Get the decompression tool and args."""
        nproc = self._get_nproc()
        if shutil.which("pigz"):
            return "pigz", ["-d", "-p", str(nproc)]
        else:
            return "gzip", ["-d"]

    def _build_tar_exclude_args(self, exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """Build tar --exclude arguments."""
        if exclude_patterns is None:
            patterns = list(DEFAULT_TAR_EXCLUDE_PATTERNS)
        else:
            patterns = list(exclude_patterns)

        # Required excludes protect credentials even when callers provide a
        # custom performance-oriented exclude list.
        for pattern in REQUIRED_TAR_EXCLUDE_PATTERNS:
            if pattern not in patterns:
                patterns.append(pattern)

        return [f"--exclude={pattern}" for pattern in patterns]

    async def _tar_gz_exists_in_s3(self, session_id: str) -> bool:
        """Check if worktree.tar.gz exists in S3."""
        try:
            cmd = [
                "aws", "s3api", "head-object",
                "--bucket", self.bucket,
                "--key", f"{self.prefix}/{session_id}/worktree.tar.gz"
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    async def get_s3_last_modified(self, session_id: str) -> Optional[float]:
        """Return the LastModified timestamp of worktree.tar.gz in S3 as an epoch float.

        Checks the worktree archive (not session.json) because worktree.tar.gz is the
        authoritative signal that a real worktree sync happened elsewhere.

        Returns None if S3 is disabled, the object does not exist, JSON parsing
        fails, or any other error occurs.
        """
        if not self.enabled:
            return None
        try:
            cmd = [
                "aws", "s3api", "head-object",
                "--output", "json",
                "--bucket", self.bucket,
                "--key", f"{self.prefix}/{session_id}/worktree.tar.gz",
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                return None
            data = json.loads(stdout)
            return datetime.fromisoformat(data["LastModified"]).timestamp()
        except Exception:
            return None

    async def _cleanup_old_per_file_objects(self, session_id: str) -> None:
        """Remove old per-file worktree/ and claude_state/ objects after tar upload."""
        for prefix_suffix in ["worktree/", "claude_state/"]:
            s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/{prefix_suffix}"
            cmd = ["aws", "s3", "rm", s3_uri, "--recursive", "--quiet"]
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            except Exception:
                pass  # Best-effort cleanup

    async def save_session_metadata(self, state: SessionState) -> bool:
        """
        Save session metadata to S3 using AWS CLI.

        Args:
            state: Session state to save

        Returns:
            True if successful
        """
        import tempfile

        if not self.enabled:
            return False

        try:
            data = state.to_dict()
            data["synced_at"] = datetime.utcnow().isoformat()

            # Write JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name

            # Upload using aws s3 cp
            s3_uri = f"s3://{self.bucket}/{self._key(state.session_id, 'session.json')}"
            cmd = [
                "aws", "s3", "cp",
                tmp_path,
                s3_uri,
                "--content-type", "application/json",
                "--quiet"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            if process.returncode != 0:
                logger.warning(f"Failed to save session metadata to S3: {stderr.decode()}")
                return False

            logger.debug(f"Saved session metadata to S3: {state.session_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save session metadata to S3: {e}")
            return False

    async def load_session_metadata(self, session_id: str) -> Optional[SessionState]:
        """
        Load session metadata from S3 using AWS CLI.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            s3_uri = f"s3://{self.bucket}/{self._key(session_id, 'session.json')}"
            cmd = ["aws", "s3", "cp", s3_uri, "-"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                # Check if it's a "not found" error
                error_msg = stderr.decode().lower()
                if "not found" in error_msg or "does not exist" in error_msg or "nosuchkey" in error_msg:
                    return None
                logger.warning(f"Failed to load session metadata from S3: {stderr.decode()}")
                return None

            data = json.loads(stdout.decode('utf-8'))
            return SessionState.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to load session metadata from S3: {e}")
            return None

    async def sync_worktree_to_s3(
        self,
        state: SessionState,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Upload worktree to S3 as a tar.gz archive via piped tar+pigz+aws s3 cp.

        This replaces the old aws s3 sync approach for 256x faster uploads.
        The .claude/ directory is included in the archive, so sync_cli_state_to_s3
        becomes a no-op.

        Args:
            state: Session state containing worktree path
            exclude_patterns: Patterns to exclude (default: .git, __pycache__, etc.)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        if not state.worktree_path:
            logger.warning(f"Session {state.session_id}: No worktree path configured")
            return False

        worktree_path = Path(state.worktree_path)
        if not worktree_path.exists():
            logger.warning(f"Session {state.session_id}: Worktree not found: {worktree_path}")
            return False

        session_dir = str(worktree_path.parent)
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{state.session_id}/worktree.tar.gz"

        # Build tar command with excludes
        exclude_args = self._build_tar_exclude_args(exclude_patterns)
        exclude_str = " ".join(exclude_args)

        # Get compressor
        compressor, comp_args = self._get_compressor()
        comp_cmd = " ".join([compressor] + comp_args)

        # Include CLI state directories when they exist. They are siblings of
        # worktree/ so S3 restore extracts them back into the session dir.
        tar_dirs = "worktree"
        claude_config = Path(session_dir) / "claude-config"
        if claude_config.exists():
            tar_dirs += " claude-config"
        codex_home = Path(session_dir) / "codex-home"
        if codex_home.exists():
            tar_dirs += " codex-home"

        # Pipe: tar | pigz | aws s3 cp
        shell_cmd = (
            f"tar cf - {exclude_str} -C {session_dir} {tar_dirs} "
            f"| {comp_cmd} "
            f"| aws s3 cp - {s3_uri}"
        )

        logger.info(f"Session {state.session_id}: Uploading worktree tar to S3...")

        try:
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(
                    f"Session {state.session_id}: Tar upload failed: {stderr.decode()}"
                )
                return False

            # Best-effort cleanup of old per-file objects
            await self._cleanup_old_per_file_objects(state.session_id)

            logger.info(f"Session {state.session_id}: Worktree tar uploaded to S3")
            return True

        except Exception as e:
            logger.error(f"Session {state.session_id}: Tar upload error: {e}")
            return False

    async def sync_ccache_to_s3(self, session_id: str) -> bool:
        """Upload this server's ccache to S3 for warm fork rebuilds on resume.

        Keyed per-session so a resumed fork session restores the cache it warmed.
        ccache is host-shared, so this is a snapshot at pause time, not exclusive.
        """
        if not self.enabled:
            return False
        ccache_dir = os.getenv("CCACHE_DIR", "/home/session_user/.ccache")
        if not Path(ccache_dir).exists():
            return False
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/ccache.tar.gz"
        comp_cmd = "pigz" if shutil.which("pigz") else "gzip"
        shell_cmd = (
            f"tar cf - -C {shlex.quote(str(Path(ccache_dir).parent))} "
            f"{shlex.quote(Path(ccache_dir).name)} | {comp_cmd} | aws s3 cp - {s3_uri}"
        )
        # Run under bash with pipefail so a tar/compress failure (e.g. ccache
        # being mutated mid-snapshot) fails the pipeline instead of being masked
        # by a 0-exit from the final `aws s3 cp`.
        proc = await asyncio.create_subprocess_shell(
            "set -o pipefail; " + shell_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"ccache upload failed for {session_id}: {stderr.decode()[:200]}")
            return False
        return True

    async def restore_ccache_from_s3(self, session_id: str) -> bool:
        """Restore a session's ccache snapshot before a fork rebuild."""
        if not self.enabled:
            return False
        ccache_dir = os.getenv("CCACHE_DIR", "/home/session_user/.ccache")
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/ccache.tar.gz"
        # Skip silently if the object doesn't exist.
        head = await asyncio.create_subprocess_shell(
            f"aws s3 ls {s3_uri}", stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await head.communicate()
        if head.returncode != 0:
            return False
        decomp = "pigz -d" if shutil.which("pigz") else "gzip -d"
        ccache_parent = Path(ccache_dir).parent
        ccache_parent.mkdir(parents=True, exist_ok=True)

        # Download to a staging temp file and VERIFY the archive end-to-end
        # before extracting into the live shared ccache. A truncated/corrupt
        # upload (the old streaming `aws s3 cp - | gzip -d | tar xf -` pipeline)
        # would otherwise scatter partial files into a host-shared cache.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            tmp_tar = tf.name
        try:
            dl = await asyncio.create_subprocess_shell(
                f"aws s3 cp {s3_uri} {shlex.quote(tmp_tar)}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                executable="/bin/bash",
            )
            _, e1 = await dl.communicate()
            if dl.returncode != 0:
                logger.warning(
                    f"ccache download failed for {session_id}: {e1.decode()[:200]}"
                )
                return False

            # Verify the archive is intact BEFORE touching the live cache.
            verify = await asyncio.create_subprocess_shell(
                f"set -o pipefail; {decomp} < {shlex.quote(tmp_tar)} | tar tf - >/dev/null",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                executable="/bin/bash",
            )
            _, e2 = await verify.communicate()
            if verify.returncode != 0:
                logger.warning(
                    f"ccache archive corrupt for {session_id}, skipping restore: "
                    f"{e2.decode()[:200]}"
                )
                return False

            ext = await asyncio.create_subprocess_shell(
                f"set -o pipefail; {decomp} < {shlex.quote(tmp_tar)} | "
                f"tar xf - -C {shlex.quote(str(ccache_parent))}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                executable="/bin/bash",
            )
            _, e3 = await ext.communicate()
            if ext.returncode != 0:
                logger.warning(
                    f"ccache extract failed for {session_id}: {e3.decode()[:200]}"
                )
                return False
            return True
        finally:
            Path(tmp_tar).unlink(missing_ok=True)

    async def sync_cli_state_to_s3(self, state: SessionState) -> bool:
        """
        Sync CLI state (.claude/) to S3.

        With tar format, .claude/ is included in the worktree tar archive,
        so this is a no-op. Returns True immediately.

        Args:
            state: Session state containing worktree path

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        if not state.worktree_path:
            return True  # Not an error if no worktree

        # With tar format, .claude/ is inside the worktree tar -- no separate sync needed
        logger.debug(f"Session {state.session_id}: CLI state included in worktree tar (no-op)")
        return True

    async def restore_worktree_from_s3(
        self,
        session_id: str,
        target_path: Path,
    ) -> bool:
        """
        Restore worktree from S3 with format detection.

        Checks if worktree.tar.gz exists (new tar format). If yes, uses piped
        aws s3 cp | pigz -d | tar xf for 47x faster downloads. If not, falls
        back to legacy aws s3 sync for backward compatibility.

        Args:
            session_id: Session identifier
            target_path: Path to restore worktree to

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        # Detect format: tar.gz or per-file
        use_tar = await self._tar_gz_exists_in_s3(session_id)

        if use_tar:
            return await self._restore_worktree_from_tar(session_id, target_path)
        else:
            return await self._restore_worktree_from_sync(session_id, target_path)

    async def _restore_worktree_from_tar(
        self, session_id: str, target_path: Path
    ) -> bool:
        """Restore worktree from tar.gz archive via piped download."""
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/worktree.tar.gz"
        session_dir = str(target_path.parent)

        # Ensure session directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        decompressor, decomp_args = self._get_decompressor()
        decomp_cmd = " ".join([decompressor] + decomp_args)

        shell_cmd = (
            f"aws s3 cp {s3_uri} - "
            f"| {decomp_cmd} "
            f"| tar xf - -C {session_dir}"
        )

        logger.info(f"Session {session_id}: Restoring worktree from tar...")

        try:
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(
                    f"Session {session_id}: Tar restore failed: {stderr.decode()}"
                )
                # Clean up any partially-extracted files to avoid a corrupt worktree
                # on subsequent resume attempts
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)
                return False

            logger.info(f"Session {session_id}: Worktree restored from tar")
            return True

        except Exception as e:
            logger.error(f"Session {session_id}: Tar restore error: {e}")
            return False

    async def _restore_worktree_from_sync(
        self, session_id: str, target_path: Path
    ) -> bool:
        """Legacy: Restore worktree from per-file S3 objects via aws s3 sync."""
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/worktree/"

        # Ensure target directory exists
        target_path.mkdir(parents=True, exist_ok=True)

        cmd = ["aws", "s3", "sync", s3_uri, str(target_path)]

        logger.info(f"Session {session_id}: Restoring worktree from S3 (legacy sync)...")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                stderr_str = stderr.decode()
                if "fatal error" in stderr_str.lower() or "404" in stderr_str:
                    logger.warning(f"Session {session_id}: No worktree found in S3")
                    return False
                logger.error(
                    f"Session {session_id}: S3 worktree restore failed: {stderr_str}"
                )
                return False

            logger.info(f"Session {session_id}: Worktree restored from S3 (legacy)")
            return True

        except Exception as e:
            logger.error(f"Session {session_id}: S3 worktree restore error: {e}")
            return False

    async def restore_cli_state_from_s3(
        self,
        session_id: str,
        worktree_path: Path,
    ) -> bool:
        """
        Restore CLI state (.claude/) from S3.

        If .claude/ already exists in the worktree (extracted from tar), this is
        a no-op. Otherwise, falls back to legacy aws s3 sync for old-format sessions.

        Args:
            session_id: Session identifier
            worktree_path: Path to worktree (will restore .claude/ inside)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        claude_dir = worktree_path / ".claude"

        # If .claude/ already exists, it was extracted from the worktree tar -- no-op
        if claude_dir.exists():
            logger.debug(
                f"Session {session_id}: CLI state already present from tar extraction (no-op)"
            )
            return True

        # Legacy fallback: sync from separate claude_state/ prefix
        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/claude_state/"
        claude_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["aws", "s3", "sync", s3_uri, str(claude_dir)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                stderr_str = stderr.decode()
                if "fatal error" in stderr_str.lower():
                    logger.warning(f"Session {session_id}: CLI state restore failed: {stderr_str}")
                    return False
                logger.debug(f"Session {session_id}: No CLI state found in S3 (this is OK)")
                return True

            logger.debug(f"Session {session_id}: CLI state restored from S3 (legacy)")
            return True

        except Exception as e:
            logger.error(f"Session {session_id}: CLI state restore error: {e}")
            return False

    async def sync_session_to_s3(self, state: SessionState, include_ccache: bool = False) -> bool:
        """
        Full sync of session state to S3.

        Uses aws s3 sync for fast, incremental uploads of:
        - Session metadata (small JSON file via aws s3 cp)
        - Worktree directory (via aws s3 sync)
        - CLI state directory (via aws s3 sync)

        Args:
            state: Session state to sync
            include_ccache: When True, also upload the fork ccache (only set on
                the pause path; ccache is large and uploading it on every
                disconnect checkpoint is wasteful). Defaults to False so
                checkpoint syncs never upload ccache.

        Returns:
            True if all syncs successful
        """
        if not self.enabled:
            return False

        if not state.worktree_path:
            logger.warning(f"Session {state.session_id}: No worktree path configured")
            return False

        # Run syncs in parallel using new aws s3 sync methods
        gather_targets = [
            self.save_session_metadata(state),
            self.sync_worktree_to_s3(state),
            self.sync_cli_state_to_s3(state),
        ]
        # Fork sessions persist ccache for warm rebuilds, but ONLY on pause
        # (it is large; uploading on every disconnect checkpoint is wasteful).
        if include_ccache and getattr(state, "vllm_fork_url", None):
            gather_targets.append(self.sync_ccache_to_s3(state.session_id))
        results = await asyncio.gather(
            *gather_targets,
            return_exceptions=True
        )

        # Check for failures
        success = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Session {state.session_id}: Sync task {i} failed: {result}")
                success = False
            elif result is False:
                success = False

        if success:
            logger.info(f"Session {state.session_id}: Synced to S3 successfully")
        else:
            logger.warning(f"Session {state.session_id}: Partial sync failure")

        return success

    async def restore_session_from_s3(
        self,
        session_id: str,
        target_worktree_path: Path,
    ) -> Optional[SessionState]:
        """
        Full restore of session state from S3.

        Uses aws s3 sync for fast directory-based restore.

        Args:
            session_id: Session identifier
            target_worktree_path: Where to restore worktree

        Returns:
            SessionState if successful, None otherwise
        """
        if not self.enabled:
            return None

        # Load metadata first
        state = await self.load_session_metadata(session_id)
        if not state:
            logger.warning(f"Session {session_id}: No metadata found in S3")
            return None

        # Restore worktree using aws s3 sync
        worktree_success = await self.restore_worktree_from_s3(
            session_id, target_worktree_path
        )

        if not worktree_success:
            logger.warning(f"Session {session_id}: Failed to restore worktree")
            return None

        # Update state with new worktree path
        state.worktree_path = str(target_worktree_path)

        # Restore CLI state using aws s3 sync
        await self.restore_cli_state_from_s3(session_id, target_worktree_path)

        logger.info(f"Session {session_id}: Restored from S3 successfully")
        return state

    async def session_exists_in_s3(self, session_id: str) -> bool:
        """Check if a session exists in S3 using AWS CLI."""
        if not self.enabled:
            return False

        try:
            cmd = [
                "aws", "s3api", "head-object",
                "--bucket", self.bucket,
                "--key", self._key(session_id, "session.json")
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    async def delete_session_from_s3(self, session_id: str) -> bool:
        """
        Delete all session data from S3.

        Uses aws s3 rm --recursive for efficient deletion of large directories.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        s3_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/"
        cmd = ["aws", "s3", "rm", s3_uri, "--recursive"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"S3 delete failed: {stderr.decode()}")
                return False

            logger.info(f"Deleted session from S3: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session from S3: {e}")
            return False

    async def list_s3_sessions(self) -> List[str]:
        """List all session IDs stored in S3 using AWS CLI."""
        if not self.enabled:
            return []

        try:
            # Use aws s3 ls to list "directories" under the prefix
            s3_uri = f"s3://{self.bucket}/{self.prefix}/"
            cmd = ["aws", "s3", "ls", s3_uri]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to list S3 sessions: {stderr.decode()}")
                return []

            # Parse output - format is "PRE session_id/"
            session_ids = []
            for line in stdout.decode().strip().split('\n'):
                if line.strip():
                    # Lines look like: "PRE abc123-def456.../"
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == "PRE":
                        session_id = parts[1].rstrip('/')
                        session_ids.append(session_id)

            return session_ids

        except Exception as e:
            logger.error(f"Failed to list S3 sessions: {e}")
            return []

    async def cleanup_stale_sessions(
        self,
        max_age_days: Optional[int] = None,
    ) -> int:
        """
        Clean up stale sessions from S3.

        Args:
            max_age_days: Maximum age in days (default: configured TTL)

        Returns:
            Number of sessions cleaned up
        """
        if not self.enabled:
            return 0

        max_age = max_age_days or self.ttl_days
        cutoff = datetime.utcnow() - timedelta(days=max_age)
        cleaned = 0

        try:
            session_ids = await self.list_s3_sessions()

            for session_id in session_ids:
                state = await self.load_session_metadata(session_id)
                if not state:
                    continue

                # Check if session is stale
                last_accessed = datetime.fromtimestamp(state.last_accessed)
                if last_accessed < cutoff:
                    logger.info(f"Cleaning up stale S3 session: {session_id}")
                    await self.delete_session_from_s3(session_id)
                    cleaned += 1

            if cleaned:
                logger.info(f"Cleaned up {cleaned} stale S3 sessions")

            return cleaned

        except Exception as e:
            logger.error(f"Failed to cleanup stale sessions: {e}")
            return cleaned

    # ========================================================================
    # Download Archive Methods (using AWS CLI for consistency)
    # ========================================================================

    async def create_download_archive(self, session_id: str) -> Optional[str]:
        """
        Create ZIP archive of session data from S3 using AWS CLI.

        This method works regardless of which server handles the request:
        1. Downloads session data from S3 to temp directory via `aws s3 sync`
        2. Creates ZIP archive locally
        3. Uploads ZIP to S3 via `aws s3 cp`

        Args:
            session_id: Session identifier

        Returns:
            S3 key of uploaded archive, or None if failed
        """
        import tempfile
        import tarfile
        import shutil

        if not self.enabled:
            logger.warning("S3 storage not enabled, cannot create download archive")
            return None

        s3_session_uri = f"s3://{self.bucket}/{self.prefix}/{session_id}/"
        archive_key = f"{self.prefix}/{session_id}/download/session_archive.zip"
        s3_archive_uri = f"s3://{self.bucket}/{archive_key}"

        temp_dir = None
        zip_path = None

        try:
            # Create temp directory for download
            temp_dir = tempfile.mkdtemp(prefix=f"session_{session_id[:8]}_")
            zip_path = f"{temp_dir}.zip"

            logger.info(f"Session {session_id}: Downloading from S3 to create archive")

            # Step 1: Download session data from S3 using aws s3 sync
            # Exclude large/unnecessary directories to speed up download
            # Note: S3 structure is sessions/{id}/worktree/... so patterns match from worktree/
            sync_cmd = [
                "aws", "s3", "sync",
                s3_session_uri,
                temp_dir,
                # Previous download archives
                "--exclude", "download/*",
                # Git (at root and in subdirs)
                "--exclude", "worktree/.git/*",
                "--exclude", "worktree/*/.git/*",
                # Python caches
                "--exclude", "worktree/__pycache__/*",
                "--exclude", "worktree/*/__pycache__/*",
                "--exclude", "worktree/**/__pycache__/*",
                "--exclude", "worktree/.mypy_cache/*",
                "--exclude", "worktree/*/.mypy_cache/*",
                "--exclude", "worktree/.pytest_cache/*",
                "--exclude", "worktree/*/.pytest_cache/*",
                "--exclude", "worktree/*.egg-info/*",
                "--exclude", "worktree/*/*.egg-info/*",
                "--exclude", "worktree/.tox/*",
                "--exclude", "worktree/*/.tox/*",
                # Virtual environments (CRITICAL - these are huge)
                "--exclude", "worktree/.venv/*",
                "--exclude", "worktree/*/.venv/*",
                "--exclude", "worktree/venv/*",
                "--exclude", "worktree/*/venv/*",
                # Build artifacts
                "--exclude", "worktree/build/*",
                "--exclude", "worktree/*/build/*",
                "--exclude", "worktree/cmake-build-*/*",
                "--exclude", "worktree/*/cmake-build-*/*",
                "--exclude", "worktree/*.so",
                "--exclude", "worktree/*/*.so",
                "--exclude", "worktree/**/*.so",
                "--exclude", "worktree/*.a",
                "--exclude", "worktree/*/*.a",
                "--exclude", "worktree/*.o",
                "--exclude", "worktree/*/*.o",
                # Node
                "--exclude", "worktree/node_modules/*",
                "--exclude", "worktree/*/node_modules/*",
                # Sensitive files: AMMO skill, agents, hooks, conversation history
                "--exclude", "worktree/.claude/*",
                "--exclude", "worktree/.claude/**/*",
                "--exclude", "claude-config/*",
                "--exclude", "claude-config/**/*",
                "--exclude", "codex-home/auth.json",
                "--exclude", "codex-home/auth.json.*",
                "--exclude", "worktree/CLAUDE.md",
                "--exclude", "worktree/.codex/*",
                "--exclude", "worktree/.codex/**/*",
                "--exclude", "worktree/AGENTS.md",
                "--quiet"
            ]

            process = await asyncio.create_subprocess_exec(
                *sync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Session {session_id}: S3 sync failed: {stderr.decode()}")
                return None

            # Check if any files were downloaded
            if not os.listdir(temp_dir):
                logger.warning(f"Session {session_id}: No files found in S3")
                return None

            # Step 2: Sanitize and zip in thread pool (rglob + rmtree + make_archive
            # are all blocking filesystem ops that must not block the event loop)
            logger.info(f"Session {session_id}: Creating ZIP archive")

            def _sanitize_and_zip(base_dir: str) -> None:
                """Sync helper: strip sensitive files then create ZIP archive."""
                base_path = Path(base_dir)

                # Tar-format checkpoints are stored as a single worktree.tar.gz
                # object in S3. Expand it first so the same sanitization rules are
                # applied to its contents before creating the user download.
                checkpoint_tar = base_path / "worktree.tar.gz"
                if checkpoint_tar.exists():
                    with tarfile.open(checkpoint_tar, "r:gz") as tar:
                        tar.extractall(base_path, filter="data")
                    checkpoint_tar.unlink(missing_ok=True)

                # Defense in depth: remove sensitive dirs/files after sync, before zip
                # (S3 --exclude may not catch everything, e.g. nested .claude dirs)
                # .codex / AGENTS.md are the Codex analogues of .claude / CLAUDE.md.
                sensitive_patterns = [
                    ".claude", "claude-config", "codex-home", "CLAUDE.md",
                    ".codex", "AGENTS.md",
                ]
                for pattern in sensitive_patterns:
                    for match in base_path.rglob(pattern):
                        if match.is_dir():
                            shutil.rmtree(match, ignore_errors=True)
                        else:
                            match.unlink(missing_ok=True)

                # Fork token scrub: session.json must survive (the UI reads it),
                # but it carries the encrypted fork access token. Strip that field
                # from every session.json before zipping so the secret never
                # leaves the server in a user download.
                for meta in base_path.rglob("session.json"):
                    try:
                        data = json.loads(meta.read_text())
                    except (OSError, ValueError):
                        continue
                    if isinstance(data, dict) and "vllm_fork_token_encrypted" in data:
                        data["vllm_fork_token_encrypted"] = None
                        try:
                            meta.write_text(json.dumps(data, indent=2))
                        except OSError:
                            pass
                shutil.make_archive(
                    base_dir,  # Base name (without .zip)
                    'zip',
                    base_dir   # Root directory to archive
                )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _sanitize_and_zip, temp_dir)

            if not os.path.exists(zip_path):
                logger.error(f"Session {session_id}: ZIP creation failed")
                return None

            zip_size = os.path.getsize(zip_path)
            logger.info(f"Session {session_id}: ZIP created, size={zip_size} bytes")

            # Step 3: Upload ZIP to S3 using aws s3 cp
            upload_cmd = [
                "aws", "s3", "cp",
                zip_path,
                s3_archive_uri,
                "--content-type", "application/zip",
                "--quiet"
            ]

            process = await asyncio.create_subprocess_exec(
                *upload_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Session {session_id}: S3 upload failed: {stderr.decode()}")
                return None

            logger.info(f"Session {session_id}: Archive uploaded to S3")
            return archive_key

        except Exception as e:
            logger.error(f"Session {session_id}: Failed to create download archive: {e}")
            return None

        finally:
            # Cleanup temp files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if zip_path and os.path.exists(zip_path):
                try:
                    os.unlink(zip_path)
                except Exception:
                    pass

    async def get_download_url(
        self,
        session_id: str,
        expires_in: int = 3600
    ) -> Optional[str]:
        """
        Generate presigned URL for session archive download using AWS CLI.

        Args:
            session_id: Session identifier
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL or None if archive doesn't exist
        """
        if not self.enabled:
            return None

        archive_key = f"{self.prefix}/{session_id}/download/session_archive.zip"
        s3_uri = f"s3://{self.bucket}/{archive_key}"

        try:
            # Generate presigned URL using aws s3 presign
            presign_cmd = [
                "aws", "s3", "presign",
                s3_uri,
                "--expires-in", str(expires_in)
            ]

            process = await asyncio.create_subprocess_exec(
                *presign_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                # Check if it's a "not found" error
                if "does not exist" in error_msg.lower() or "not found" in error_msg.lower():
                    logger.debug(f"Session {session_id}: Archive not found in S3")
                else:
                    logger.error(f"Session {session_id}: Failed to generate presigned URL: {error_msg}")
                return None

            url = stdout.decode().strip()
            return url if url else None

        except Exception as e:
            logger.error(f"Session {session_id}: Failed to generate download URL: {e}")
            return None

    async def get_download_size(self, session_id: str) -> Optional[int]:
        """
        Get size of download archive in bytes using AWS CLI.

        Args:
            session_id: Session identifier

        Returns:
            Size in bytes or None if archive doesn't exist
        """
        if not self.enabled:
            return None

        archive_key = f"{self.prefix}/{session_id}/download/session_archive.zip"

        try:
            # Use aws s3api head-object to get size
            head_cmd = [
                "aws", "s3api", "head-object",
                "--bucket", self.bucket,
                "--key", archive_key,
                "--query", "ContentLength",
                "--output", "text"
            ]

            process = await asyncio.create_subprocess_exec(
                *head_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return None

            size_str = stdout.decode().strip()
            return int(size_str) if size_str and size_str.isdigit() else None

        except Exception:
            return None

    async def ensure_session_synced(
        self,
        session_id: str,
        state: Optional[SessionState] = None
    ) -> bool:
        """
        Ensure session data is synced to S3 before download.

        For active/paused sessions with local data, sync to S3 first.
        For sessions only in S3 (cross-node), this is a no-op.

        Args:
            session_id: Session identifier
            state: Optional session state (if available locally)

        Returns:
            True if session data exists in S3, False otherwise
        """
        if not self.enabled:
            return False

        # If we have local state with a worktree, sync it
        if state and state.worktree_path and Path(state.worktree_path).exists():
            logger.info(f"Session {session_id}: Local data exists, syncing to S3")
            return await self.sync_session_to_s3(state)

        # No local data - check if session exists in S3
        return await self.session_exists_in_s3(session_id)


# Singleton instance
_session_storage: Optional[SessionS3Storage] = None


def get_session_storage() -> SessionS3Storage:
    """Get singleton session storage instance."""
    global _session_storage
    if _session_storage is None:
        _session_storage = SessionS3Storage()
    return _session_storage
