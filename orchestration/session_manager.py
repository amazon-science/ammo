# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Session manager for AI CLI sessions.

Orchestrates the full session lifecycle:
- Create/destroy/pause/resume sessions
- Track session metadata and state
- Coordinate with worktree, CLI tool, and terminal managers
- Handle GPU allocation
"""

import asyncio
import re
import uuid
import time
import json
import logging
import os
import signal
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, List, Any, NamedTuple
from pathlib import Path
from contextlib import asynccontextmanager

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionInfo,
    SessionListResponse,
    SessionActionResponse,
    SessionDownloadInfo,
    DEFAULT_SESSION_DATA_DIR,
    DEFAULT_INACTIVITY_TIMEOUT_MINS,
    ENV_SESSION_DATA_DIR,
    SESSION_STATUS_DESCRIPTIONS,
)
from shared.gpu_resource_manager import GPUResourceManager
from orchestration.worktree_manager import WorktreeManager, WorktreeError, get_worktree_manager
from orchestration.terminal_manager import TerminalManager, TerminalError, get_terminal_manager
from orchestration.cli_tool_manager import CLIToolManager, CLIToolError, get_cli_tool_manager
from orchestration.inactivity_monitor import InactivityMonitor, get_inactivity_monitor
from orchestration.session_state import SessionS3Storage, get_session_storage
from orchestration.fork_repo_manager import get_fork_repo_manager
from shared.fork_token_crypto import encrypt_fork_token

logger = logging.getLogger(__name__)

MAX_SESSIONS_PER_CLIENT = int(os.getenv("MAX_SESSIONS_PER_CLIENT", "8"))


def _read_ammo_version() -> Optional[str]:
    """Read AMMO version from VERSION file. Returns None on any failure.

    Tries the in-container template path first (populated at Docker build
    time via cp -r) and falls back to the in-repo source-of-truth so dev /
    native runs and stale-image cases still report a version.
    """
    try:
        version_file = Path(
            os.getenv("SESSION_TEMPLATES_DIR", "/data/templates")
        ) / "claude/.claude/VERSION"
        if not version_file.exists():
            # Repo path: <repo>/ai_cli_session/.claude/VERSION
            version_file = (
                Path(__file__).resolve().parent.parent
                / "ai_cli_session/.claude/VERSION"
            )
        first_line = version_file.read_text().splitlines()[0]
        m = re.match(r"^version:\s+(\S+)", first_line)
        return m.group(1) if m else None
    except Exception:
        return None


def _toml_string(value: str) -> str:
    """Return a TOML-compatible quoted string for generated config."""
    return json.dumps(value)


class SessionError(Exception):
    """Exception raised for session operations."""
    pass


class SessionLimitError(SessionError):
    """Raised when a client exceeds the maximum number of active sessions."""
    pass


class ProcIdentity(NamedTuple):
    """
    One process's identity, kept in separate fields so a marker match can respect
    token boundaries.

    A single concatenated blob makes every marker a plaintext substring, which any
    in-session agent can spoof — a marker inside a shell comment on argv, or inside
    an unrelated environment variable's value, matched and (for an exemption
    marker) made the process immune to the pause sweep. The fields below keep each
    token whole:

    ``environ`` holds exact ``KEY=VALUE`` lines (NUL-split, never re-joined).
    ``argv`` holds exact argument vector elements (NUL-split).
    ``cwd`` is the resolved working directory, on its own.
    """
    environ: tuple
    argv: tuple
    cwd: Optional[str]

    def has_env_line(self, line: str) -> bool:
        """True when ``line`` is an environ entry in full (``KEY=VALUE`` equality)."""
        return line in self.environ

    def has_argv_basename(self, basename: str) -> bool:
        """True when an argv element is ``basename`` or a path ending in it."""
        suffix = f"/{basename}"
        return any(
            arg == basename or arg.endswith(suffix) for arg in self.argv
        )

    def has_path(self, path: str) -> bool:
        """
        True when an argv element, the cwd, or an environ line's VALUE is ``path``
        or a path under it.

        A path marker matches only at a ``/`` boundary, so session ``abc``'s marker
        never matches session ``abc-2``'s path, and never matches mid-word.
        """
        prefix = f"{path}/"

        def _matches(value: str) -> bool:
            return value == path or value.startswith(prefix)

        if self.cwd and _matches(self.cwd):
            return True
        if any(_matches(arg) for arg in self.argv):
            return True
        for line in self.environ:
            _, sep, value = line.partition("=")
            if sep and _matches(value):
                return True
        return False


class SessionMarkers(NamedTuple):
    """
    One session's ownership markers, split by how each one is matched.

    ``env_line`` is the complete ``AMMO_SESSION_ID={session_id}`` environ entry.
    ``paths`` holds the session paths that embed the session id.
    """
    env_line: Optional[str]
    paths: tuple

    def valid(self) -> bool:
        """False when the session has no id, so the sweep must match nothing."""
        return self.env_line is not None

    def matches(self, identity: ProcIdentity) -> bool:
        """True when this process belongs to the session, boundary-matched."""
        if self.env_line and identity.has_env_line(self.env_line):
            return True
        return any(identity.has_path(path) for path in self.paths)


class SessionManager:
    """
    Central manager for AI CLI sessions.

    Coordinates:
    - Worktree creation/cleanup
    - CLI tool process management (Phase 2)
    - Terminal process management (Phase 3)
    - GPU allocation
    - Session state persistence
    """

    def __init__(
        self,
        sessions_dir: Optional[str] = None,
        worktree_manager: Optional[WorktreeManager] = None,
        gpu_manager: Optional[GPUResourceManager] = None,
        terminal_manager: Optional[TerminalManager] = None,
        cli_tool_manager: Optional[CLIToolManager] = None,
        inactivity_monitor: Optional[InactivityMonitor] = None,
        session_storage: Optional[SessionS3Storage] = None,
    ):
        """
        Initialize session manager.

        Args:
            sessions_dir: Directory for session data
            worktree_manager: Worktree manager instance (created if not provided)
            gpu_manager: GPU resource manager instance (created if not provided)
            terminal_manager: Terminal manager instance (created if not provided)
            cli_tool_manager: CLI tool manager instance (created if not provided)
            inactivity_monitor: Inactivity monitor instance (created if not provided)
            session_storage: S3 storage instance (created if not provided)
        """
        self.sessions_dir = Path(
            sessions_dir or os.getenv(ENV_SESSION_DATA_DIR, DEFAULT_SESSION_DATA_DIR)
        )
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Component managers
        self.worktree_manager = worktree_manager or get_worktree_manager()
        self.gpu_manager = gpu_manager or GPUResourceManager()
        self.terminal_manager = terminal_manager or get_terminal_manager()
        self.cli_tool_manager = cli_tool_manager or get_cli_tool_manager()
        self.inactivity_monitor = inactivity_monitor or get_inactivity_monitor()
        self.session_storage = session_storage or get_session_storage()

        # In-memory session state
        self._sessions: Dict[str, SessionState] = {}

        # Per-session recovery locks to prevent concurrent double-spawns
        self._recovery_locks: Dict[str, asyncio.Lock] = {}

        # Load existing sessions from disk
        self._load_sessions()

        logger.info(
            f"SessionManager initialized: sessions_dir={self.sessions_dir}, "
            f"loaded {len(self._sessions)} existing sessions, "
            f"S3 storage available: {self.session_storage.enabled}"
        )

    def _get_session_state_path(self, session_id: str) -> Path:
        """Get path to session state file."""
        return self.sessions_dir / session_id / "session.json"

    def _load_sessions(self) -> None:
        """Load existing sessions from disk."""
        if not self.sessions_dir.exists():
            return

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            state_path = session_dir / "session.json"
            if not state_path.exists():
                continue

            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                state = SessionState.from_dict(data)

                # Skip terminated/failed sessions - they shouldn't be reloaded
                if state.status in (SessionStatus.TERMINATED, SessionStatus.FAILED):
                    logger.debug(
                        f"Skipping {state.status.value} session {state.session_id}"
                    )
                    continue

                # A crashed/restarted server can never resume an in-flight
                # source build (the background task died). Land it in a
                # terminal, GPU-released, user-visible FAILED state instead
                # of an immortal BUILDING zombie.
                if state.status == SessionStatus.BUILDING:
                    logger.warning(
                        f"Session {state.session_id} was BUILDING at restart; "
                        f"marking FAILED (build task did not survive)"
                    )
                    state.status = SessionStatus.FAILED
                    state.build_phase = None
                    state.build_error = (
                        (state.build_error or "")
                        + "\n[server restarted during build — session failed]"
                    ).strip()
                    state.gpu_ids = []
                    state.cli_process_pid = None
                    state.ttyd_process_pid = None
                    state.terminal_port = None
                    self._save_session_state(state)
                    # Skip: FAILED sessions are not added to the live registry.
                    continue

                # Mark previously active sessions as paused (server restarted)
                if state.status in (SessionStatus.ACTIVE, SessionStatus.CREATING):
                    logger.info(
                        f"Session {state.session_id} was {state.status.value}, marking as paused"
                    )
                    state.status = SessionStatus.PAUSED
                    state.cli_process_pid = None
                    state.ttyd_process_pid = None
                    state.terminal_port = None
                    self._save_session_state(state)

                self._sessions[state.session_id] = state
                logger.debug(f"Loaded session {state.session_id}: {state.status.value}")

            except Exception as e:
                logger.error(f"Failed to load session from {session_dir}: {e}")

    async def cleanup_orphaned_worktrees(self) -> None:
        """
        Clean up orphaned worktrees from crashed sessions.

        Called on startup to prune stale worktree references that may
        have been left behind by sessions that crashed or were killed
        without proper cleanup.
        """
        try:
            repos_dir = self.worktree_manager.repos_dir
            for repo_dir in repos_dir.iterdir():
                if repo_dir.is_dir() and (repo_dir / ".git").exists():
                    result = await asyncio.create_subprocess_exec(
                        "git", "-C", str(repo_dir), "worktree", "prune",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    await result.communicate()
                    if result.returncode == 0:
                        logger.debug(f"Pruned stale worktrees for {repo_dir.name}")
        except Exception as e:
            logger.warning(f"Failed to prune orphaned worktrees: {e}")

    async def discover_s3_sessions(self) -> int:
        """
        Discover sessions from S3 not already loaded locally.

        Called on startup to recover sessions after local host recovery
        when local disk storage is ephemeral.

        Returns:
            Number of sessions discovered from S3
        """
        if not self.session_storage or not self.session_storage.enabled:
            logger.info("S3 storage not configured, skipping S3 session discovery")
            return 0

        discovered = 0
        try:
            s3_session_ids = await self.session_storage.list_s3_sessions()
            logger.info(f"Found {len(s3_session_ids)} sessions in S3")

            for session_id in s3_session_ids:
                # Check if session already exists locally
                if session_id in self._sessions:
                    # Compare timestamps to detect stale local state
                    local_state = self._sessions[session_id]
                    try:
                        s3_state = await self.session_storage.load_session_metadata(session_id)
                        if not s3_state:
                            logger.debug(f"Session {session_id}: Already local, no S3 metadata")
                            continue
                        local_ts = local_state.s3_last_sync or 0
                        s3_ts = s3_state.s3_last_sync or 0
                        if s3_ts > local_ts:
                            logger.info(
                                f"Session {session_id}: S3 is newer "
                                f"(s3_ts={s3_ts}, local_ts={local_ts}), replacing local state"
                            )
                            # Remove stale local worktree
                            if local_state.worktree_path and Path(local_state.worktree_path).exists():
                                import shutil
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, lambda: shutil.rmtree(local_state.worktree_path, ignore_errors=True))
                                logger.info(f"Session {session_id}: Removed stale worktree at {local_state.worktree_path}")

                            # Replace with S3 state (fall through to processing below)
                            pass
                        else:
                            logger.debug(f"Session {session_id}: Local is current, keeping local state")
                            continue
                    except Exception as e:
                        logger.error(f"Session {session_id}: Failed timestamp comparison: {e}")
                        continue

                    state = s3_state
                else:
                    # New session not in local memory — load from S3
                    try:
                        state = await self.session_storage.load_session_metadata(session_id)
                    except Exception as e:
                        logger.error(f"Session {session_id}: Failed to load from S3: {e}")
                        continue

                try:
                    if not state:
                        logger.warning(f"Session {session_id}: No metadata in S3")
                        continue

                    # Skip terminated/failed sessions
                    if state.status in (SessionStatus.TERMINATED, SessionStatus.FAILED):
                        logger.debug(f"Session {session_id}: Skipping {state.status.value} from S3")
                        continue

                    # A BUILDING session in S3 had its build task die with the
                    # prior host; there is no resume path for an in-flight source
                    # build, so mark it terminally FAILED (GPU-released) instead
                    # of importing an immortal BUILDING zombie.
                    if state.status == SessionStatus.BUILDING:
                        logger.warning(f"Session {session_id}: BUILDING in S3, marking FAILED (build did not survive)")
                        state.status = SessionStatus.FAILED
                        state.build_phase = None
                        state.build_error = ((state.build_error or "") + "\n[server restarted during build — session failed]").strip()
                        state.gpu_ids = []
                        state.cli_process_pid = None
                        state.ttyd_process_pid = None
                        state.terminal_port = None
                        self._save_session_state(state)
                        continue

                    # Mark as PAUSED (not running on this host)
                    if state.status in (SessionStatus.ACTIVE, SessionStatus.CREATING):
                        logger.info(f"Session {session_id}: Was {state.status.value}, marking as paused")
                        state.status = SessionStatus.PAUSED

                    # Clear process state (not running locally)
                    state.cli_process_pid = None
                    state.ttyd_process_pid = None
                    state.terminal_port = None

                    # Update paths to local directory
                    state.worktree_path = str(self.sessions_dir / session_id / "worktree")
                    state.session_dir = str(self.sessions_dir / session_id)
                    state.logs_dir = str(self.sessions_dir / session_id / "logs")

                    # Do NOT reset s3_last_sync here. Discovery only imports the
                    # S3 *metadata* — the actual worktree is whatever happens to
                    # be on this host's NVMe (local storage persists across server
                    # rescheduling on the same node; may be a stale snapshot
                    # from a prior host). The resume_session path handles the
                    # reset after restore_session_from_s3 extracts the tar.
                    # Keep the imported value so future cross-host S3 restores still
                    # see S3 as newer via get_s3_last_modified comparison.

                    # Add to memory and save locally
                    self._sessions[session_id] = state
                    self._save_session_state(state)
                    discovered += 1

                    logger.info(f"Session {session_id}: Discovered from S3 (owner_id={state.owner_id})")

                except Exception as e:
                    logger.error(f"Session {session_id}: Failed to load from S3: {e}")

            if discovered:
                logger.info(f"Discovered {discovered} sessions from S3")
            return discovered

        except Exception as e:
            logger.error(f"Failed to discover sessions from S3: {e}")
            return 0

    def _save_session_state(self, state: SessionState) -> None:
        """Save session state to disk."""
        state_path = self._get_session_state_path(state.session_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=str(state_path.parent))
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            os.replace(tmp_path, state_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return str(uuid.uuid4())

    def _validate_ownership(
        self,
        session_id: str,
        owner_id: Optional[str],
    ) -> SessionState:
        """
        Validate that a session exists and belongs to the specified owner.

        Ownership rules:
        - Legacy sessions (owner_id=None) are accessible by all clients
        - New sessions require matching owner_id
        - If owner_id param is None (no client ID provided), all sessions are accessible

        Args:
            session_id: Session identifier
            owner_id: Client ID to validate against

        Returns:
            SessionState if validation passes

        Raises:
            SessionError: If session not found or ownership doesn't match
        """
        state = self._sessions.get(session_id)
        if not state:
            raise SessionError(f"Session {session_id} not found")

        # If no client ID provided
        if owner_id is None:
            ammo_api_key = os.getenv("AMMO_API_KEY", "")
            if ammo_api_key:
                # Auth enabled: null client_id can only see null-owner (legacy) sessions
                if state.owner_id is not None:
                    raise SessionError(f"Session {session_id} not found")
                return state
            # Auth disabled: allow access to all sessions (backward compatibility)
            return state

        # Legacy sessions (owner_id=None) are accessible by all
        if state.owner_id is None:
            return state

        # New sessions require matching owner_id
        if state.owner_id != owner_id:
            # Don't reveal existence of session to unauthorized clients
            raise SessionError(f"Session {session_id} not found")

        return state

    async def _acquire_gpus(
        self,
        session_id: str,
        gpu_count: int,
        timeout: Optional[int] = None,
    ) -> List[int]:
        """
        Acquire multiple GPUs for a session.

        Delegates to GPUResourceManager for unified GPU tracking.
        Uses async acquisition to avoid blocking the event loop.

        Args:
            session_id: Session identifier
            gpu_count: Number of GPUs to acquire
            timeout: Timeout in seconds (uses GPU_TIMEOUT_SESSION default)

        Returns:
            List of acquired GPU IDs
        """
        if gpu_count <= 0:
            return []

        try:
            kwargs = dict(session_id=session_id, gpu_count=gpu_count)
            if timeout is not None:
                kwargs["timeout"] = timeout
            return await self.gpu_manager.acquire_gpus_for_session_async(**kwargs)
        except (ValueError, TimeoutError) as e:
            raise SessionError(str(e))

    def _release_gpus(self, session_id: str) -> None:
        """Release GPUs held by a session."""
        self.gpu_manager.release_gpus_for_session(session_id)

    def _prepare_codex_home(self, state: SessionState) -> Path:
        """Create isolated Codex state/config and trust this session worktree."""
        if not state.session_dir or not state.worktree_path:
            raise SessionError("Cannot prepare Codex home before session paths are set")

        codex_home = Path(state.session_dir) / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        worktree_path = str(Path(state.worktree_path))
        (codex_home / "config.toml").write_text(
            "\n".join([
                "# Generated by AMMO session manager.",
                'model = "gpt-5.6-sol"',
                "check_for_update_on_startup = false",
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
                f"log_dir = {_toml_string(str(codex_home / 'log'))}",
                f"sqlite_home = {_toml_string(str(codex_home))}",
                "",
                "[features]",
                "multi_agent = false",
                "hooks = true",
                "shell_tool = true",
                "unified_exec = false",
                "",
                "[features.multi_agent_v2]",
                "enabled = true",
                "hide_spawn_agent_metadata = false",
                'tool_namespace = "agents"',
                "max_concurrent_threads_per_session = 17",
                "",
                "[agents]",
                "job_max_runtime_seconds = 7200",
                "",
                f"[projects.{_toml_string(worktree_path)}]",
                'trust_level = "trusted"',
                "",
            ])
        )
        self._seed_codex_auth(codex_home)
        return codex_home

    def _seed_codex_auth(self, codex_home: Path) -> None:
        """Best-effort auth setup for isolated per-session Codex homes."""
        auth_path = codex_home / "auth.json"
        if auth_path.exists():
            return

        source_auth = Path(
            os.getenv("CODEX_AUTH_JSON_PATH")
            or (Path.home() / ".codex" / "auth.json")
        )
        if source_auth.exists():
            try:
                shutil.copyfile(source_auth, auth_path)
                os.chmod(auth_path, 0o600)
                logger.debug("Seeded Codex auth from auth.json")
                return
            except OSError as e:
                logger.warning(f"Failed to seed Codex auth from auth.json: {type(e).__name__}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return

        try:
            env = os.environ.copy()
            env["CODEX_HOME"] = str(codex_home)
            env["HOME"] = "/home/session_user"
            env["OPENAI_API_KEY"] = ""
            subprocess.run(
                ["/usr/bin/codex", "login", "--with-api-key"],
                input=f"{api_key}\n",
                text=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=True,
            )
            logger.debug("Seeded Codex auth from OPENAI_API_KEY")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to seed Codex auth from OPENAI_API_KEY: {type(e).__name__}")

    def _has_codex_history(self, codex_home: Path) -> bool:
        """Return True when Codex has resumable conversation history."""
        for history_file in ("session_index.jsonl", "history.jsonl"):
            path = codex_home / history_file
            if path.exists() and path.stat().st_size > 0:
                return True

        sessions_dir = codex_home / "sessions"
        if sessions_dir.exists():
            for path in sessions_dir.rglob("*.jsonl"):
                if path.is_file() and path.stat().st_size > 0:
                    return True

        return False

    def _prepare_claude_config(self, state: SessionState) -> Path:
        """Create Claude Code state/config and trust this session worktree."""
        if not state.session_dir or not state.worktree_path:
            raise SessionError("Cannot prepare Claude config before session paths are set")

        claude_config_dir = Path(state.session_dir) / "claude-config"
        claude_config_dir.mkdir(parents=True, exist_ok=True)
        claude_prefs_file = claude_config_dir / ".claude.json"
        if not claude_prefs_file.exists():
            claude_prefs_file.write_text(json.dumps({
                "theme": "dark",
                "hasCompletedOnboarding": True,
                "projects": {
                    str(state.worktree_path): {
                        "hasTrustDialogAccepted": True,
                        "allowedTools": [],
                    }
                }
            }, indent=2))
        return claude_config_dir

    def _force_kill_session_processes(self, state) -> None:
        """
        Force-kill CLI tool and terminal processes for a session.

        Bug #8 fix: Called when stop_terminal() fails to ensure processes are
        killed before releasing GPUs. Attempts SIGKILL on the process group.
        """
        pids_to_kill = []
        if state.cli_process_pid:
            pids_to_kill.append(state.cli_process_pid)
        if state.ttyd_process_pid:
            pids_to_kill.append(state.ttyd_process_pid)

        for pid in pids_to_kill:
            try:
                pgid = os.getpgid(pid)
                os.kill(-pgid, signal.SIGKILL)
                logger.info(f"Session {state.session_id}: Force-killed process group {pgid} (pid={pid})")
            except OSError as e:
                logger.debug(f"Session {state.session_id}: Could not kill pid {pid}: {e}")

    # --- Detached CLI daemon teardown -------------------------------------
    #
    # Since Claude Code 2.1.x the CLI re-hosts itself under a background daemon
    # that is detached from the tmux pane tree. Killing ttyd plus the session
    # tmux server therefore leaves the lead and every teammate running headless
    # under that daemon, still executing GPU work on a session whose GPU locks
    # the server already released; the daemon also respawns a child killed by
    # pid alone. Pause and terminate must enumerate the session's processes from
    # /proc and kill the whole set, daemon included.
    #
    # A process is session-owned when its /proc/<pid>/environ, cmdline, or cwd
    # carries a marker that embeds the session id. Every marker MUST embed the
    # session id, and every match MUST respect token boundaries (see
    # ``ProcIdentity``) — a plaintext substring search over the concatenated
    # blob is forbidden, because it lets any in-session agent spoof a marker
    # inside a shell comment or inside another variable's value:
    #   1. the session directory (/data/sessions/{session_id}), matched against a
    #      whole argv element, the cwd, or an environ line's value, and only as
    #      that path itself or a path under it (a '/' boundary) — so session
    #      "abc" never matches session "abc-2"
    #   2. AMMO_SESSION_ID={session_id} (exported into every CLI process env),
    #      matched only as a complete environ line
    #   3. the worktree path, when it contains the session id, matched as in (1)
    #
    # EXEMPTION (by design): a detached run started through the skill's
    # reserved_detached_run.sh must SURVIVE pause — it owns its own GPU
    # reservation and is expected to outlive the interactive session. A process
    # carrying one of the exemption markers below is never signalled, and
    # neither is its process group. Each exemption marker is boundary-matched
    # too: the env flag only as a complete environ line, the script only as a
    # whole argv element (bare or a path ending in it).
    _DETACHED_RUN_EXEMPT_SCRIPT = "reserved_detached_run.sh"
    _DETACHED_RUN_EXEMPT_ENV_LINE = "AMMO_DETACHED_RUN=1"
    _DETACHED_RUN_EXEMPT_MARKERS = (
        _DETACHED_RUN_EXEMPT_SCRIPT,
        _DETACHED_RUN_EXEMPT_ENV_LINE,
    )

    # Seconds between the SIGTERM sweep and the SIGKILL escalation.
    _DAEMON_KILL_GRACE_SEC = 2.0

    _PROC_ROOT = "/proc"

    def _session_process_markers(self, state) -> "SessionMarkers":
        """
        Markers that identify this session's processes. Each embeds the session
        id, and each is boundary-matched against a ``ProcIdentity``:
        ``env_line`` only as a complete environ line, every path in ``paths``
        only as a whole argv element, the cwd, or an environ line's value, and
        only as that path or a path under it. Session ``abc`` therefore never
        matches session ``abc-2``, and a marker buried in a longer string
        (a shell comment, another variable's value) never matches at all.
        """
        session_id = getattr(state, "session_id", None)
        if not session_id:
            return SessionMarkers(env_line=None, paths=())
        sessions_dir = getattr(self, "sessions_dir", None)
        candidates = [
            state.session_dir,
            state.worktree_path,
            str(Path(sessions_dir) / session_id) if sessions_dir else None,
        ]
        paths = []
        # A path marker is used only when it contains the session id.
        for path in candidates:
            if path and session_id in path and path not in paths:
                paths.append(path)
        return SessionMarkers(
            env_line=f"AMMO_SESSION_ID={session_id}", paths=tuple(paths)
        )

    def _protected_pids(self, proc_root: str) -> set:
        """PIDs the sweep must never signal: init, this server, and its ancestors."""
        protected = {1, os.getpid()}
        pid = os.getpid()
        for _ in range(64):  # bounded walk; guards against a cyclic/faked /proc
            try:
                with open(f"{proc_root}/{pid}/stat", "rb") as f:
                    fields = f.read().rsplit(b")", 1)[-1].split()
                ppid = int(fields[1])
            except (OSError, IndexError, ValueError):
                break
            if ppid <= 0 or ppid in protected:
                break
            protected.add(ppid)
            pid = ppid
        return protected

    @staticmethod
    def _read_proc_tokens(path: str) -> Optional[tuple]:
        """Read a NUL-separated /proc file into its exact tokens."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            return None
        return tuple(
            chunk.decode("utf-8", "replace")
            for chunk in raw.split(b"\0")
            if chunk
        )

    def _read_proc_identity(self, pid: int, proc_root: str) -> Optional[ProcIdentity]:
        """
        Read a pid's environ, cmdline, and cwd into a ``ProcIdentity``.

        environ and cmdline are NUL-split into their exact tokens and kept apart,
        because a concatenated blob turns every marker into a spoofable plaintext
        substring. See ``ProcIdentity``.
        """
        environ = self._read_proc_tokens(f"{proc_root}/{pid}/environ")
        argv = self._read_proc_tokens(f"{proc_root}/{pid}/cmdline")
        try:
            cwd = os.readlink(f"{proc_root}/{pid}/cwd")
        except OSError:
            cwd = None
        if environ is None and argv is None and cwd is None:
            return None
        return ProcIdentity(
            environ=environ or (), argv=argv or (), cwd=cwd
        )

    def _is_exempt_detached_run(self, identity: ProcIdentity) -> bool:
        """
        True for a detached run that must survive pause (see the EXEMPTION note).

        Both markers are boundary-matched: the env flag only as a complete
        environ line, the script only as a whole argv element (bare or a path
        ending in it). A marker inside a shell comment, inside another
        variable's value, or as part of a longer word does NOT exempt.
        """
        if identity.has_env_line(self._DETACHED_RUN_EXEMPT_ENV_LINE):
            return True
        return identity.has_argv_basename(self._DETACHED_RUN_EXEMPT_SCRIPT)

    def _scan_session_processes(self, state, proc_root: Optional[str] = None) -> tuple:
        """
        Scan /proc once and return ``(owned_pids, exempt_pgids)``.

        ``owned_pids`` is every live process this session owns — daemon and
        children alike — in ascending pid order (oldest ancestor first).
        ``exempt_pgids`` holds the process groups of the exempt detached runs, so
        the sweep never signals a group that shelters one.

        A pid in the server's OWN process group is never owned: no session
        process ever inherits the server pgid, because ttyd and the tmux server
        are spawned with ``start_new_session=True`` (terminal_manager.py), while
        the server's own helpers DO inherit it — the pause-time S3 upload runs
        ``bash -c "tar ... -C {session_dir} ... | aws s3 cp -"``, which carries
        the session dir on its argv. Signalling that pipeline truncates the
        upload the pause was taken to produce, so it is excluded here rather
        than downgraded to a bare-pid kill (which still kills tar).
        """
        proc_root = proc_root or self._PROC_ROOT
        markers = self._session_process_markers(state)
        if not markers.valid():
            return [], set()
        protected = self._protected_pids(proc_root)
        try:
            own_pgid = os.getpgid(0)
        except OSError:
            own_pgid = None
        try:
            entries = os.listdir(proc_root)
        except OSError as e:
            logger.warning(f"Session {state.session_id}: Cannot scan {proc_root}: {e}")
            return [], set()

        owned = []
        exempt_pgids = set()
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in protected:
                continue
            identity = self._read_proc_identity(pid, proc_root)
            if identity is None:  # pid exited between listdir and the read
                continue
            if self._is_exempt_detached_run(identity):
                try:
                    exempt_pgids.add(os.getpgid(pid))
                except OSError:
                    pass
                continue
            if not markers.matches(identity):
                continue
            if own_pgid is not None and self._pgid_of(pid, proc_root) == own_pgid:
                logger.debug(
                    f"Session {state.session_id}: Skipping pid {pid} in the "
                    f"server's own process group ({own_pgid}); it is a "
                    "server-spawned helper, not a session process"
                )
                continue
            owned.append(pid)
        return sorted(owned), exempt_pgids

    @staticmethod
    def _pgid_of(pid: int, proc_root: str) -> Optional[int]:
        """
        A pid's process group id, read from ``proc_root`` so the scan stays
        consistent with the /proc tree it walked (field 5 of stat, counted after
        the comm field, which may itself hold spaces or parentheses).
        """
        try:
            with open(f"{proc_root}/{pid}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[-1].split()
            return int(fields[2])
        except (OSError, IndexError, ValueError):
            return None

    def _find_session_owned_pids(self, state, proc_root: Optional[str] = None) -> List[int]:
        """Session-owned pids only; see ``_scan_session_processes``."""
        return self._scan_session_processes(state, proc_root=proc_root)[0]

    def _signal_session_pid(self, pid: int, sig, session_id: str, exempt_pgids=()) -> None:
        """
        Signal a session-owned pid's process group, or the bare pid when that
        group must not be signalled wholesale: a group that shelters an exempt
        detached run, or — as a last-resort guard, since
        ``_scan_session_processes`` already excludes these — the server's own
        group.
        """
        try:
            pgid = os.getpgid(pid)
        except OSError as e:
            logger.debug(f"Session {session_id}: Could not read pgid of {pid}: {e}")
            pgid = None
        try:
            if pgid and pgid > 1 and pgid != os.getpgid(0) and pgid not in exempt_pgids:
                os.kill(-pgid, sig)
            else:
                os.kill(pid, sig)
        except OSError as e:
            logger.debug(f"Session {session_id}: Could not signal pid {pid}: {e}")

    async def _kill_session_owned_processes(self, state, proc_root: Optional[str] = None) -> List[int]:
        """
        Kill the session's detached CLI daemon and every process it re-hosts.

        SIGTERMs the whole set in one sweep (daemon first, so it cannot respawn a
        child), waits ``_DAEMON_KILL_GRACE_SEC``, then escalates: survivors get
        SIGKILL and ``_force_kill_session_processes`` runs against the recorded
        cli/ttyd process groups. Returns the pids still alive at the end.
        """
        session_id = state.session_id
        # The /proc walk reads one small file per pid; keep it off the event loop
        # so a busy container cannot stall /health during a pause.
        async def _scan() -> tuple:
            return await asyncio.to_thread(
                self._scan_session_processes, state, proc_root
            )

        pids, exempt_pgids = await _scan()
        if not pids:
            logger.debug(f"Session {session_id}: No session-owned processes found")
            return []

        logger.info(
            f"Session {session_id}: Terminating {len(pids)} session-owned "
            f"process(es) (pids={pids})"
        )
        for pid in pids:
            self._signal_session_pid(pid, signal.SIGTERM, session_id, exempt_pgids)

        await asyncio.sleep(self._DAEMON_KILL_GRACE_SEC)

        survivors, exempt_pgids = await _scan()
        if survivors:
            logger.error(
                f"Session {session_id}: {len(survivors)} session-owned process(es) "
                f"survived SIGTERM (pids={survivors}); escalating to SIGKILL"
            )
            self._force_kill_session_processes(state)
            for pid in survivors:
                self._signal_session_pid(pid, signal.SIGKILL, session_id, exempt_pgids)
            await asyncio.sleep(0.5)
            survivors, _ = await _scan()
            if survivors:
                logger.error(
                    f"Session {session_id}: Session-owned process(es) survived "
                    f"SIGKILL (pids={survivors}); GPU collision risk on next session"
                )
        return survivors

    async def _stop_session_processes(self, state) -> None:
        """
        Stop everything this session runs, in order: ttyd plus the tmux server,
        then the detached CLI daemon tree that outlives them.

        Must complete before GPUs are released, otherwise a surviving agent runs
        GPU work against locks the server has already handed to another session.
        """
        session_id = state.session_id
        if self.terminal_manager and state.terminal_port:
            try:
                await self.terminal_manager.stop_terminal(session_id)
                logger.info(f"Session {session_id}: Stopped terminal")
            except Exception as e:
                logger.error(f"Session {session_id}: Failed to stop terminal: {e}")
                # Escalate: force-kill child processes before releasing GPUs
                self._force_kill_session_processes(state)
        await self._kill_session_owned_processes(state)

    def _fork_base_in_use_by_others(self, session_id: str, fork_url: str) -> bool:
        """True if any OTHER non-terminated session shares this fork base."""
        for sid, st in self._sessions.items():
            if sid == session_id:
                continue
            if st.vllm_fork_url == fork_url and st.status not in (
                SessionStatus.TERMINATED, SessionStatus.FAILED
            ):
                return True
        return False

    # Build-console wrapper script (shipped with the server). In the container the
    # server package lives at /app; the relative fallback below covers a source
    # checkout.
    _FORK_CONSOLE_DOCKER = "/app/scripts/fork_build_console.sh"

    def _fork_console_path(self) -> str:
        from pathlib import Path as _P
        docker = self._FORK_CONSOLE_DOCKER
        if _P(docker).exists():
            return docker
        return str(_P(__file__).parent.parent / "scripts" / "fork_build_console.sh")

    # Teammate exec wrapper (shipped with the server). Set as the orchestrator's
    # CLAUDE_CODE_TEAMMATE_COMMAND so every tmux teammate execs through it; the
    # wrapper arms the high-effort configuration for the champion agent types.
    # See scripts/teammate-cmd-wrapper.sh.
    #
    # CRITICAL — path must be TRAVERSABLE by session_user (uid 1000): the
    # orchestrator runs as session_user and execs this path raw. The in-tree
    # copy lives under /app, whose dirs are 0750 root:root (from
    # `COPY --chmod=750 . /app`), so uid 1000 cannot traverse into them and
    # exec fails with 126 — which would break ALL teammate spawns. The
    # Dockerfile therefore copies the wrapper to
    # /usr/local/lib/ammo/ (world-traversable, same pattern as
    # gpu_lock_wrapper.py) and we point at that copy. The in-tree path is only a
    # source-checkout (non-Docker) fallback for local dev where the server runs
    # as the developer's own uid and the tree is fully traversable.
    _TEAMMATE_CMD_DOCKER = "/usr/local/lib/ammo/teammate-cmd-wrapper.sh"

    def _teammate_cmd_path(self) -> str:
        from pathlib import Path as _P
        docker = self._TEAMMATE_CMD_DOCKER
        if _P(docker).exists():
            return docker
        return str(_P(__file__).parent.parent / "scripts" / "teammate-cmd-wrapper.sh")

    async def _run_fork_build(self, state: "SessionState", request: "CreateSessionRequest") -> None:
        """Background task: clone fork → worktree → source build → terminal+CLI.

        Streams the build to <logs>/fork_build.log (tailed live by the build
        console terminal). Retries the source build once, then FAILED.
        """
        from orchestration.fork_repo_manager import get_fork_repo_manager
        from orchestration.worktree_manager import WorktreeError
        from shared.fork_token_crypto import decrypt_fork_token
        from shared.fork_url_validator import validate_fork_url, ForkUrlError

        session_id = state.session_id
        logs_dir = Path(state.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        build_log = logs_dir / "fork_build.log"
        status_file = logs_dir / "fork_build.status"
        # Ensure the console (running as session_user) can read+tail these.
        build_log.write_text("")
        status_file.write_text("")
        try:
            os.chown(str(build_log), 1000, 1000)
            os.chown(str(status_file), 1000, 1000)
        except OSError:
            pass

        loop = asyncio.get_event_loop()
        try:
            # Re-validate the persisted fork URL before any clone, so a tampered
            # or legacy on-disk state can't drive a clone of a non-github host.
            try:
                state.vllm_fork_url = validate_fork_url(state.vllm_fork_url)
            except ForkUrlError as e:
                raise RuntimeError(f"persisted fork URL failed re-validation: {e}")
            token = decrypt_fork_token(state.vllm_fork_token_encrypted)
            # Fix G (symmetry): refuse to anonymously clone a private fork when
            # its token is present but undecryptable on this host.
            if state.vllm_fork_token_encrypted and token is None:
                raise RuntimeError(
                    "fork access token could not be decrypted on this host "
                    "(AMMO_FORK_TOKEN_KEY missing or rotated); cannot clone private fork"
                )

            # 1. Clone/fetch the fork base repo.
            state.build_phase = "fetching"
            self._save_session_state(state)
            fork_mgr = get_fork_repo_manager()
            fork_base = await loop.run_in_executor(
                None,
                lambda: fork_mgr.ensure_fork_base(state.vllm_fork_url, state.branch, token),
            )

            # 2. Create the worktree from the fork base.
            worktree_path = await loop.run_in_executor(
                None,
                lambda: self.worktree_manager.create_worktree(
                    session_id=session_id, repo_name=state.repo_name,
                    branch=state.branch, fork_base_path=fork_base,
                ),
            )
            state.worktree_path = str(worktree_path)

            # 3. Set up CLI workspace + env (same as the normal path).
            _tp = state.tp_size if (state.tp_size and state.tp_size > 0) else None
            _dp = state.dp_size if (state.dp_size and state.dp_size > 0) else None
            await loop.run_in_executor(
                None,
                lambda: self.cli_tool_manager.setup_workspace(
                    tool_type=state.cli_tool, worktree_path=Path(state.worktree_path),
                    session_id=session_id, gpu_ids=state.gpu_ids,
                    repo_name=state.repo_name, branch=state.branch,
                    tp_size=_tp, dp_size=_dp,
                ),
            )
            extra_env = self._build_extra_env(state)

            # 4. Open the build-console terminal NOW (before the build) so the
            #    user watches it live. It tails build_log and execs the CLI on
            #    the "ok" sentinel.
            cli_command = self.cli_tool_manager.get_cli_command(
                state.cli_tool, extra_env=extra_env, initial_prompt=state.initial_prompt,
            )
            console_command = [
                self._fork_console_path(), str(status_file), str(build_log), "--",
                *cli_command,
            ]
            await loop.run_in_executor(None, self._chown_session_to_user, state)
            if self.terminal_manager and self.terminal_manager.is_available():
                tmux_session_name = f"ammo-{session_id[:12]}"
                try:
                    port = await self.terminal_manager.start_terminal_with_command(
                        session_id=session_id, command=console_command,
                        working_dir=Path(state.worktree_path), env=extra_env,
                        title=None, tmux_session_name=tmux_session_name,
                    )
                    state.terminal_port = port
                except TerminalError as e:
                    logger.warning(f"Session {session_id}: build console terminal failed: {e}")

            # 5. Run the source build (retry once), streaming to build_log.
            state.build_phase = "compiling"
            self._save_session_state(state)
            last_err = None
            for attempt in (1, 2):
                try:
                    env_result = await self.worktree_manager.initialize_vllm_environment(
                        session_id=session_id, branch=state.branch,
                        precompiled=False, log_path=str(build_log),
                    )
                    state.build_initialized = True
                    state.build_timings = env_result.get("timings")
                    last_err = None
                    break
                except asyncio.CancelledError:
                    raise
                except (WorktreeError, Exception) as e:  # retry transient failures
                    last_err = e
                    logger.warning(f"Session {session_id}: fork build attempt {attempt} failed: {e}")
                    with open(build_log, "a") as fh:
                        fh.write(f"\n[attempt {attempt} failed: {e}]\n")
            if last_err is not None:
                raise last_err

            # Abort if the session was terminated/removed while we built.
            current = self._sessions.get(session_id)
            if current is None or current.status == SessionStatus.TERMINATED:
                logger.info(f"Session {session_id}: terminated during build; not starting CLI")
                return

            # 6. Success — chown the freshly built .venv, flip sentinel → CLI starts.
            state.build_phase = "installing"
            await loop.run_in_executor(None, self._chown_session_to_user, state)
            status_file.write_text("ok")
            try:
                os.chown(str(status_file), 1000, 1000)
            except OSError:
                pass
            state.status = SessionStatus.ACTIVE
            state.build_phase = None
            state.last_accessed = time.time()
            self._save_session_state(state)
            # Re-arm inactivity monitoring now that the session is ACTIVE
            # (a timeout firing during BUILDING would have unregistered it).
            if self.inactivity_monitor:
                self.inactivity_monitor.register_session(
                    session_id, timeout_mins=state.inactivity_timeout_mins
                )
            logger.info(f"Session {session_id}: fork build succeeded, agent starting")

        except asyncio.CancelledError:
            # Cancelled (terminate/shutdown). Mark FAILED + release GPUs so the
            # on-disk state is terminal; then re-raise so the task ends cancelled.
            try:
                self._fail_build(state, RuntimeError("build cancelled"), build_log, status_file)
            finally:
                raise
        except Exception as e:
            self._fail_build(state, e, build_log, status_file)

    def _fail_build(self, state: "SessionState", error: Exception,
                    build_log: Path, status_file: Path) -> None:
        """Mark a fork session FAILED: surface log, release GPUs, signal console."""
        session_id = state.session_id
        logger.error(f"Session {session_id}: fork build failed: {error}")
        # Last-N build log lines for the UI.
        tail_lines = ""
        try:
            lines = Path(build_log).read_text(errors="replace").splitlines()
            tail_lines = "\n".join(lines[-40:])
        except OSError:
            pass
        state.build_error = (tail_lines + f"\n[build failed: {error}]").strip()
        state.status = SessionStatus.FAILED
        state.build_phase = None
        # Tell the console terminal to stop tailing and show the failure banner.
        try:
            status_file.write_text("failed")
            os.chown(str(status_file), 1000, 1000)
        except OSError:
            pass
        # Release held GPUs.
        if state.gpu_ids:
            self._release_gpus(session_id)
            state.gpu_ids = []
        self._save_session_state(state)

    async def _reinit_fork_env_on_resume(self, state: "SessionState") -> None:
        """Re-clone the fork base (with token) and rebuild from source on resume."""
        from orchestration.fork_repo_manager import get_fork_repo_manager
        from shared.fork_token_crypto import decrypt_fork_token
        from shared.fork_url_validator import validate_fork_url, ForkUrlError

        loop = asyncio.get_event_loop()
        # Re-validate the persisted fork URL before any clone, so a tampered or
        # legacy on-disk state can't drive a clone of a non-github host on resume.
        try:
            state.vllm_fork_url = validate_fork_url(state.vllm_fork_url)
        except ForkUrlError as e:
            raise RuntimeError(f"persisted fork URL failed re-validation: {e}")
        token = decrypt_fork_token(state.vllm_fork_token_encrypted)
        # Fix G: a private fork resumed on a host whose AMMO_FORK_TOKEN_KEY is
        # missing/rotated cannot decrypt the stored token. Fail loudly instead
        # of silently cloning the private fork anonymously (which would either
        # 404 or clone the wrong/public tree).
        if state.vllm_fork_token_encrypted and token is None:
            raise RuntimeError(
                "fork access token could not be decrypted on this host "
                "(AMMO_FORK_TOKEN_KEY missing or rotated); cannot clone private fork"
            )
        fork_mgr = get_fork_repo_manager()
        fork_base = await loop.run_in_executor(
            None, lambda: fork_mgr.ensure_fork_base(state.vllm_fork_url, state.branch, token)
        )
        # Repair worktree linkage against the fork base, then rebuild.
        try:
            self.worktree_manager.repair_worktree_linkage(
                session_id=state.session_id, worktree_path=Path(state.worktree_path),
                repo_name=state.repo_name, branch=state.branch,
            )
        except Exception as e:
            logger.warning(f"Session {state.session_id}: fork linkage repair failed: {e}")
        logs_dir = Path(state.logs_dir) if state.logs_dir else Path(state.worktree_path).parent / "logs"
        build_log = logs_dir / "fork_build.log"
        # Warm the build with the session's prior ccache (best-effort).
        session_storage = getattr(self, "session_storage", None)
        if session_storage and session_storage.enabled:
            try:
                await session_storage.restore_ccache_from_s3(state.session_id)
            except Exception as e:
                logger.warning(f"Session {state.session_id}: ccache restore failed: {e}")
        env_result = await self.worktree_manager.initialize_vllm_environment(
            session_id=state.session_id, branch=state.branch,
            precompiled=False, log_path=str(build_log),
        )
        state.build_initialized = True
        state.build_timings = env_result.get("timings")

    async def create_session(
        self,
        request: CreateSessionRequest,
        owner_id: Optional[str] = None,
    ) -> CreateSessionResponse:
        """
        Create a new AI CLI session.

        Args:
            request: Session creation request
            owner_id: Optional client ID for session ownership isolation

        Returns:
            CreateSessionResponse with session info
        """
        # Check if resuming existing session
        if request.session_id:
            return await self.resume_session(request.session_id, request.initial_prompt, owner_id=owner_id)

        # Session limit enforcement
        ammo_api_key = os.getenv("AMMO_API_KEY", "")
        if ammo_api_key and not owner_id:
            raise SessionError("X-Client-ID header required for session creation")

        active_count = sum(
            1 for s in self._sessions.values()
            if s.owner_id == owner_id and s.status not in (
                SessionStatus.TERMINATED, SessionStatus.FAILED, SessionStatus.PAUSED
            )
        )
        if active_count >= MAX_SESSIONS_PER_CLIENT:
            raise SessionLimitError(
                f"Maximum {MAX_SESSIONS_PER_CLIENT} active sessions per client. "
                f"Terminate existing sessions first."
            )

        # Generate new session ID
        session_id = self._generate_session_id()

        logger.info(
            f"Creating session {session_id}: repo={request.repo_name}, "
            f"cli={request.cli_tool}, branch={request.branch}, "
            f"gpus={request.gpu_count}"
        )

        # Create initial state
        now = time.time()
        state = SessionState(
            session_id=session_id,
            status=SessionStatus.CREATING,
            cli_tool=CLIToolType(request.cli_tool),
            repo_name=request.repo_name,
            branch=request.branch,
            created_at=now,
            last_accessed=now,
            inactivity_timeout_mins=request.inactivity_timeout_mins,
            requested_gpu_count=request.gpu_count,
            tp_size=(
                request.tp_size
                if request.tp_size is not None
                else request.gpu_count
            ),
            dp_size=request.dp_size,
            owner_id=owner_id,
            model_name=request.model_name,
            dtype=request.dtype,
            ammo_version=_read_ammo_version(),
        )

        # Bug #3 fix: Register session in CREATING state BEFORE the first await
        # so concurrent create_session calls see it in the limit check.
        self._sessions[session_id] = state

        # ── Custom fork path ──────────────────────────────────────────────
        # Forks build vLLM from source (15-20 min). Run that asynchronously:
        # allocate GPUs now (held through the build), set BUILDING, return
        # immediately, and let _run_fork_build() drive clone → worktree →
        # source build → terminal+CLI in the background.
        if request.vllm_fork_url:
            # Fix C: do ALL fork setup (dirs + GPU acquire + task spawn) inside
            # one try/except so any failure removes the session and releases
            # GPUs — never leaving a phantom BUILDING session holding GPUs.
            try:
                state.vllm_fork_url = request.vllm_fork_url
                state.vllm_fork_token_encrypted = encrypt_fork_token(request.vllm_fork_token)
                state.initial_prompt = request.initial_prompt
                state.status = SessionStatus.BUILDING
                state.build_phase = "fetching"
                # Session dirs (worktree created later inside the build task).
                dirs = self.worktree_manager.create_session_dirs(session_id)
                state.session_dir = str(dirs["session_dir"])
                state.logs_dir = str(dirs["logs_dir"])
                # Acquire GPUs up front (held through the build).
                if request.gpu_count > 0:
                    logger.info(f"Session {session_id}: Acquiring {request.gpu_count} GPUs (fork build)")
                    state.gpu_ids = await self._acquire_gpus(session_id, request.gpu_count)
                self._save_session_state(state)
                # Spawn the background build. Track it in a per-session dict so
                # terminate/shutdown can find and cancel the right task.
                task = asyncio.create_task(self._run_fork_build(state, request))
                if not hasattr(self, "_fork_build_tasks") or not isinstance(self._fork_build_tasks, dict):
                    self._fork_build_tasks = {}
                self._fork_build_tasks[session_id] = task
                task.add_done_callback(
                    lambda t, sid=session_id: self._fork_build_tasks.pop(sid, None)
                )
                if self.inactivity_monitor:
                    self.inactivity_monitor.register_session(
                        session_id, timeout_mins=state.inactivity_timeout_mins
                    )
                return state.to_create_response(
                    message="Session created. Building vLLM from fork (this may take 15-20 min)."
                )
            except Exception as e:
                logger.error(f"Session {session_id}: fork session setup failed before build: {e}")
                if state.gpu_ids:
                    self._release_gpus(session_id)
                    state.gpu_ids = []
                self._sessions.pop(session_id, None)
                raise SessionError(f"Failed to start fork session: {e}")
        # ── End custom fork path ──────────────────────────────────────────

        try:
            # Create worktree (run in executor to avoid blocking the event loop;
            # git clone/fetch operations take 6-8 seconds)
            logger.info(f"Session {session_id}: Creating worktree")
            loop = asyncio.get_event_loop()
            worktree_path = await loop.run_in_executor(
                None,
                lambda: self.worktree_manager.create_worktree(
                    session_id=session_id,
                    repo_name=request.repo_name,
                    branch=request.branch,
                ),
            )
            state.worktree_path = str(worktree_path)
            state.session_dir = str(self.worktree_manager.get_session_dir(session_id))
            state.logs_dir = str(self.worktree_manager.get_logs_dir(session_id))

            # Acquire GPUs if requested
            if request.gpu_count > 0:
                logger.info(f"Session {session_id}: Acquiring {request.gpu_count} GPUs")
                gpu_ids = await self._acquire_gpus(session_id, request.gpu_count)
                state.gpu_ids = gpu_ids

            # Save state to disk
            self._save_session_state(state)

            # Initialize vLLM development environment (vLLM repos only)
            # Copies venv and CMakeUserPresets.json. C++ builds are on-demand.
            if request.repo_name == "vllm":
                logger.info(f"Session {session_id}: Initializing vLLM environment")
                try:
                    env_result = await self.worktree_manager.initialize_vllm_environment(
                        session_id=session_id,
                        branch=request.branch,
                    )
                    state.build_initialized = True
                    state.build_timings = env_result.get("timings")
                    logger.info(
                        f"Session {session_id}: vLLM environment initialized "
                        f"(timings={state.build_timings})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Session {session_id}: vLLM env init failed (non-fatal): {e}"
                    )
                    # Non-fatal: session can still work, user can set up manually
                    state.error = f"Env init failed: {e}"
                self._save_session_state(state)

            # Set up CLI tool workspace (run in executor to avoid blocking the event loop;
            # file copies for hooks/skills/settings take 0.5-2 seconds)
            logger.info(f"Session {session_id}: Setting up {state.cli_tool.value} workspace")
            # Forward tp/dp into workspace setup. Claude writes these to
            # settings.local.json for Task subagents; Codex agents read them
            # from the ttyd env. tp_size=0 is the legacy sentinel — leave as
            # None so setup skips injection.
            _tp = state.tp_size if (state.tp_size and state.tp_size > 0) else None
            _dp = state.dp_size if (state.dp_size and state.dp_size > 0) else None
            await loop.run_in_executor(
                None,
                lambda: self.cli_tool_manager.setup_workspace(
                    tool_type=state.cli_tool,
                    worktree_path=Path(state.worktree_path),
                    session_id=session_id,
                    gpu_ids=state.gpu_ids,
                    repo_name=state.repo_name,
                    branch=state.branch,
                    tp_size=_tp,
                    dp_size=_dp,
                ),
            )

            # Build extra environment variables for CLI tool. These are
            # embedded in the command via /usr/bin/env and passed to ttyd.
            extra_env = self._build_extra_env(state)
            logger.info(f"Session {session_id}: HF cache at {extra_env['HF_HOME']}")

            # Chown session directory to session_user BEFORE launching CLI.
            # Everything above (worktree, workspace setup, claude-config, hf-cache)
            # was created as root. The CLI runs as session_user via `su` and
            # needs write access to the entire tree.
            await loop.run_in_executor(None, self._chown_session_to_user, state)

            # Start terminal with CLI tool (ttyd spawns the CLI command)
            terminal_started = False
            if self.terminal_manager.is_available():
                logger.info(f"Session {session_id}: Starting terminal with CLI tool")
                # Store initial prompt in state so auto-recovery can re-send it
                # if the terminal dies before Claude Code processes it.
                state.initial_prompt = request.initial_prompt
                cli_command = self.cli_tool_manager.get_cli_command(
                    state.cli_tool,
                    extra_env=extra_env,
                    initial_prompt=request.initial_prompt,
                )

                try:
                    # Pass env vars both via /usr/bin/env in command AND to ttyd process
                    # CUDA_VISIBLE_DEVICES must be in ttyd's env to propagate to child
                    tmux_session_name = f"ammo-{session_id[:12]}"
                    terminal_port = await self.terminal_manager.start_terminal_with_command(
                        session_id=session_id,
                        command=cli_command,
                        working_dir=Path(state.worktree_path),
                        env=extra_env,  # Pass env to ttyd process for proper propagation
                        title=None,  # Disabled: --title flag causes execvp issues with ttyd
                        tmux_session_name=tmux_session_name,
                    )
                    state.terminal_port = terminal_port
                    terminal_started = True
                    logger.info(f"Session {session_id}: Terminal started on port {terminal_port}")
                except TerminalError as e:
                    terminal_started = False
                    logger.warning(f"Session {session_id}: Failed to start terminal - {e}")
                    state.error = f"Terminal failed to start: {e}"
                    # Release GPUs acquired earlier — session is FAILED
                    # and must not hold GPU allocations indefinitely
                    if state.gpu_ids:
                        logger.info(
                            f"Session {session_id}: Releasing GPUs {state.gpu_ids} "
                            f"due to terminal failure"
                        )
                        self._release_gpus(session_id)
                        state.gpu_ids = []
            else:
                logger.warning(f"Session {session_id}: ttyd not available, skipping terminal")

            # Only mark ACTIVE if terminal started successfully.
            # If terminal failed, mark as FAILED so it doesn't appear functional.
            if terminal_started:
                state.status = SessionStatus.ACTIVE
            else:
                state.status = SessionStatus.FAILED
                logger.warning(
                    f"Session {session_id}: Marked FAILED due to terminal failure"
                )
            state.last_accessed = time.time()
            self._save_session_state(state)

            # Register with inactivity monitor
            if self.inactivity_monitor:
                self.inactivity_monitor.register_session(
                    session_id,
                    timeout_mins=state.inactivity_timeout_mins,
                )
                logger.debug(f"Session {session_id}: Registered with inactivity monitor")

            logger.info(f"Session {session_id}: Created successfully")

            return state.to_create_response(
                message=f"Session created. Worktree at {state.worktree_path}"
            )

        except Exception as e:
            logger.error(f"Session {session_id}: Creation failed - {e}")
            state.status = SessionStatus.FAILED
            state.error = str(e)
            self._save_session_state(state)

            # Cleanup on failure
            self._release_gpus(session_id)
            try:
                self.worktree_manager.cleanup_session(session_id, request.repo_name)
            except Exception:
                pass

            # Bug #3 fix: Remove from _sessions so it doesn't block the limit check.
            # Only remove if still CREATING — terminal failure sets FAILED and should
            # remain visible for diagnostics.
            if session_id in self._sessions and self._sessions[session_id].status == SessionStatus.CREATING:
                self._sessions.pop(session_id, None)

            raise SessionError(f"Failed to create session: {e}")

    async def get_session(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
    ) -> Optional[SessionInfo]:
        """
        Get session information.

        Args:
            session_id: Session identifier
            owner_id: Optional client ID for ownership validation

        Returns:
            SessionInfo or None if not found/not owned
        """
        try:
            state = self._validate_ownership(session_id, owner_id)
            return state.to_session_info()
        except SessionError:
            return None

    async def list_sessions(
        self,
        status_filter: Optional[SessionStatus] = None,
        owner_id: Optional[str] = None,
    ) -> SessionListResponse:
        """
        List all sessions.

        Args:
            status_filter: Optional filter by status
            owner_id: Optional client ID to filter by ownership

        Returns:
            SessionListResponse with session list
        """
        sessions = []
        auth_enabled = bool(os.getenv("AMMO_API_KEY", ""))
        for state in self._sessions.values():
            if status_filter and state.status != status_filter:
                continue

            # Filter by ownership
            # - If no owner_id provided + auth disabled (backward compat), show all sessions
            # - If no owner_id provided + auth enabled, show only legacy (null-owner) sessions
            # - Legacy sessions (state.owner_id=None) are visible to all authenticated clients
            # - Otherwise, only show sessions owned by this client
            if owner_id is None:
                if auth_enabled and state.owner_id is not None:
                    continue
            elif state.owner_id is not None and state.owner_id != owner_id:
                continue

            sessions.append(state.to_session_info())

        # Sort by last_accessed descending
        sessions.sort(key=lambda s: s.last_accessed, reverse=True)

        return SessionListResponse(
            sessions=sessions,
            total=len(sessions),
        )

    async def pause_session(
        self,
        session_id: str,
        sync_to_s3: bool = False,
        owner_id: Optional[str] = None,
    ) -> SessionActionResponse:
        """
        Pause an active session.

        Kills ttyd, the session tmux server, and the detached CLI daemon tree
        (see ``_stop_session_processes``); a run started through
        reserved_detached_run.sh survives. The worktree is preserved.

        Args:
            session_id: Session identifier
            sync_to_s3: If True, sync session state to S3 for cross-node resume
            owner_id: Optional client ID for ownership validation

        Returns:
            SessionActionResponse with result
        """
        state = self._validate_ownership(session_id, owner_id)

        if state.status != SessionStatus.ACTIVE:
            raise SessionError(
                f"Session {session_id} is {state.status.value}, cannot pause"
            )

        logger.info(f"Pausing session {session_id} (sync_to_s3={sync_to_s3})")

        # Stop ttyd, tmux, and the detached CLI daemon tree BEFORE releasing GPUs,
        # so no surviving agent runs GPU work against reallocated locks.
        await self._stop_session_processes(state)

        # Unregister from inactivity monitor
        if self.inactivity_monitor:
            self.inactivity_monitor.unregister_session(session_id)

        # Release GPUs (only after terminal is confirmed stopped or force-killed)
        self._release_gpus(session_id)

        # Clean up recovery lock for this session
        self._recovery_locks.pop(session_id, None)

        # Update state
        state.status = SessionStatus.PAUSED
        state.cli_process_pid = None
        state.ttyd_process_pid = None
        state.terminal_port = None
        state.last_accessed = time.time()
        # Note: Preserve state.gpu_ids for display purposes (GPUs are released but IDs retained)

        # Sync to S3 if requested (for cross-node resume).
        # User-initiated pause: upload is shielded so a racing cancel can't
        # orphan the bookkeeping, and the conflict guard no longer silently
        # drops the request. If S3 appears newer than local, we refuse to
        # clobber (S3 freshness safety) but surface the refusal to the caller.
        # The try/finally ensures _save_session_state runs even if an HTTP
        # handler cancellation (reverse-proxy timeout or client abort) fires during the upload —
        # otherwise the in-memory PAUSED status and advanced s3_last_sync
        # would be lost on server restart.
        sync_conflict = False
        sync_failed = False
        try:
            if sync_to_s3 and self.session_storage and self.session_storage.enabled:
                try:
                    s3_last_modified = await self.session_storage.get_s3_last_modified(session_id)
                    if s3_last_modified and state.s3_last_sync and s3_last_modified > state.s3_last_sync:
                        logger.warning(
                            f"Session {session_id}: S3 has newer data "
                            f"(s3={s3_last_modified:.0f} > local={state.s3_last_sync:.0f}), "
                            "refusing pause-sync to avoid overwriting newer state"
                        )
                        sync_conflict = True
                    else:
                        logger.info(f"Session {session_id}: Syncing to S3...")

                        async def _pause_upload_and_record() -> bool:
                            ok = await self.session_storage.sync_session_to_s3(state, include_ccache=True)
                            if not ok:
                                return False
                            state.s3_synced = True
                            s3_ts = await self.session_storage.get_s3_last_modified(session_id)
                            state.s3_last_sync = s3_ts if s3_ts else time.time()
                            # Persist advanced s3_last_sync inside the shield so the
                            # on-disk value cannot regress to the pre-upload value
                            # if the outer handler is cancelled before the finally
                            # block reads state.s3_last_sync.
                            self._save_session_state(state)
                            return True

                        ok = await asyncio.shield(_pause_upload_and_record())
                        if ok:
                            logger.info(f"Session {session_id}: Synced to S3 successfully")
                        else:
                            sync_failed = True
                            logger.warning(f"Session {session_id}: S3 sync returned False")
                except Exception as e:
                    sync_failed = True
                    logger.warning(f"Session {session_id}: S3 sync failed: {e}")
        finally:
            # Always persist local state (PAUSED status + any advanced s3_last_sync)
            # so a subsequent restart doesn't regress to stale ACTIVE/old-sync state.
            self._save_session_state(state)

        message = "Session paused. Use resume to continue."
        if sync_to_s3:
            if sync_conflict:
                message += " WARNING: S3 has newer data from another host; local work NOT uploaded."
            elif sync_failed:
                message += " WARNING: S3 sync failed; local work NOT uploaded."
            elif state.s3_synced:
                message += " State synced to S3."

        return SessionActionResponse(
            session_id=session_id,
            status=state.status.value,
            message=message,
        )

    async def resume_session(
        self,
        session_id: str,
        initial_prompt: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> CreateSessionResponse:
        """
        Resume a paused session.

        If the session is not in local memory or the worktree is missing,
        attempts to restore from S3 (for cross-node resume).

        Args:
            session_id: Session identifier
            initial_prompt: Optional prompt to send after resuming
            owner_id: Optional client ID for ownership validation

        Returns:
            CreateSessionResponse with session info
        """
        state = self._sessions.get(session_id)
        restored_from_s3 = False

        # Determine if S3 restore is needed:
        # 1. Session not in local memory
        # 2. Worktree path missing on disk
        # 3. S3 has newer data than local (stale worktree) — detected via HeadObject
        has_stale_worktree = False
        if (state is not None
                and self.session_storage
                and self.session_storage.enabled):
            s3_last_modified = await self.session_storage.get_s3_last_modified(session_id)
            if s3_last_modified and state.s3_last_sync:
                has_stale_worktree = s3_last_modified > state.s3_last_sync
        needs_s3_restore = (
            state is None
            or (state.worktree_path and not Path(state.worktree_path).exists())
            or has_stale_worktree
        )

        if needs_s3_restore:
            if self.session_storage and self.session_storage.enabled:
                logger.info(f"Session {session_id}: Attempting to restore from S3...")
                try:
                    target_worktree_path = self.sessions_dir / session_id / "worktree"
                    # Remove stale worktree before S3 restore (only when S3 is newer)
                    if has_stale_worktree and target_worktree_path.exists():
                        import shutil
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, lambda: shutil.rmtree(target_worktree_path, ignore_errors=True))
                        logger.info(f"Session {session_id}: Removed stale worktree before S3 restore")
                    restored_state = await self.session_storage.restore_session_from_s3(
                        session_id,
                        target_worktree_path=target_worktree_path,
                    )
                    if restored_state:
                        state = restored_state
                        # Force status to PAUSED - session can't be genuinely active on
                        # this host since it was just restored from S3. Without this,
                        # the early-return at `if status == ACTIVE: return` below would
                        # skip terminal startup, causing [exited] in the terminal.
                        state.status = SessionStatus.PAUSED
                        state.cli_process_pid = None
                        state.ttyd_process_pid = None
                        state.terminal_port = None
                        # Reset s3_last_sync to the S3 object's actual LastModified:
                        # this host's on-disk tree matches that exact S3 version.
                        # Using the S3 timestamp (not time.time()) prevents the race
                        # where a concurrent writer lands between our restore and
                        # our next pause — the guard correctly detects their newer
                        # upload because s3_last_modified > our recorded value.
                        s3_ts = await self.session_storage.get_s3_last_modified(session_id)
                        state.s3_last_sync = s3_ts if s3_ts else time.time()
                        # Fix session_dir and logs_dir to point to the actual local
                        # paths, not the stale paths from the old host's S3 metadata.
                        state.session_dir = str(self.sessions_dir / session_id)
                        state.logs_dir = str(self.sessions_dir / session_id / "logs")
                        self._sessions[session_id] = state
                        restored_from_s3 = True
                        logger.info(f"Session {session_id}: Restored from S3 (forced status=PAUSED)")
                except Exception as e:
                    logger.warning(f"Session {session_id}: S3 restore failed: {e}")

        if not state:
            raise SessionError(f"Session {session_id} not found")

        # Validate ownership (after potential S3 restore)
        # Note: For S3-restored sessions, we validate against the stored owner_id
        if owner_id is not None and state.owner_id is not None:
            if state.owner_id != owner_id:
                raise SessionError(f"Session {session_id} not found")

        if state.status not in (SessionStatus.PAUSED, SessionStatus.ACTIVE):
            raise SessionError(
                f"Session {session_id} is {state.status.value}, cannot resume"
            )

        if state.status == SessionStatus.ACTIVE:
            return state.to_create_response(message="Session already active")

        logger.info(f"Resuming session {session_id} (restored_from_s3={restored_from_s3})")

        # Verify worktree exists
        if not state.worktree_path or not Path(state.worktree_path).exists():
            raise SessionError(f"Session worktree not found at {state.worktree_path}")

        try:
            # Repair git worktree linkage after cross-host S3 restore.
            # The .git file may be missing or point to the original host's
            # base repo path. This creates/updates the .git file and the
            # corresponding entry in the base repo's .git/worktrees/.
            if restored_from_s3:
                try:
                    self.worktree_manager.repair_worktree_linkage(
                        session_id=session_id,
                        worktree_path=Path(state.worktree_path),
                        repo_name=state.repo_name,
                        branch=state.branch,
                    )
                except Exception as e:
                    logger.warning(
                        f"Session {session_id}: Worktree linkage repair failed (non-fatal): {e}"
                    )

            # Re-initialize vLLM environment if restored from S3
            # (venv is not synced to S3, need to re-copy).
            if restored_from_s3 and state.repo_name == "vllm":
                if state.vllm_fork_url:
                    logger.info(f"Session {session_id}: Rebuilding fork from source (resume)")
                    # A fork session with no built vLLM is useless; do NOT
                    # swallow. Let the rebuild failure propagate so the outer
                    # resume except releases GPUs and resets to PAUSED
                    # (recoverable) instead of a broken-but-ACTIVE session.
                    await self._reinit_fork_env_on_resume(state)
                else:
                    logger.info(f"Session {session_id}: Re-initializing vLLM environment (restored from S3)")
                    try:
                        env_result = await self.worktree_manager.initialize_vllm_environment(
                            session_id=session_id,
                            branch=state.branch,
                        )
                        state.build_initialized = True
                        state.build_timings = env_result.get("timings")
                        logger.info(
                            f"Session {session_id}: vLLM environment re-initialized "
                            f"(timings={state.build_timings})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Session {session_id}: vLLM env re-init failed (non-fatal): {e}"
                        )
                        state.error = f"Env re-init failed: {e}"

            # Fix Claude Code project directory name for cross-host S3 restore.
            # After S3 restore, worktree path may differ from original host,
            # causing Claude to not find its conversation history.
            if restored_from_s3 and state.cli_tool == CLIToolType.CLAUDE:
                self._fix_claude_project_dir_after_s3_restore(state)

            # Re-acquire GPUs if session originally requested them
            if state.requested_gpu_count > 0:
                logger.info(f"Session {session_id}: Re-acquiring {state.requested_gpu_count} GPUs")
                gpu_ids = await self._acquire_gpus(session_id, state.requested_gpu_count)
                state.gpu_ids = gpu_ids

            # Build extra environment variables for CLI tool. These are
            # embedded in the command via /usr/bin/env and passed to ttyd.
            extra_env = self._build_extra_env(state)
            logger.info(f"Session {session_id}: HF cache at {extra_env['HF_HOME']}")

            loop = asyncio.get_event_loop()

            # Refresh the subagent env in settings.local.json. It was written at
            # create time and is restored verbatim from S3, so the GPU set and
            # the worktree path in it are stale after a resume onto other GPUs
            # or another host. Without this, the parent and its Task subagents
            # disagree about which GPUs the session owns.
            if state.cli_tool == CLIToolType.CLAUDE and state.worktree_path:
                _tp = state.tp_size if (state.tp_size and state.tp_size > 0) else None
                _dp = state.dp_size if (state.dp_size and state.dp_size > 0) else None
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: self.cli_tool_manager.refresh_session_env(
                            Path(state.worktree_path), session_id, state.gpu_ids,
                            tp_size=_tp, dp_size=_dp,
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"Session {session_id}: settings.local.json env refresh "
                        f"failed (non-fatal): {e}"
                    )

            # Chown session directory to session_user BEFORE launching CLI.
            # On resume, S3-restored files and freshly re-created dirs are owned
            # by root.  The CLI runs as session_user and needs write access.
            await loop.run_in_executor(None, self._chown_session_to_user, state)

            # Start terminal with CLI tool (ttyd spawns the CLI command)
            terminal_started = False
            if self.terminal_manager and self.terminal_manager.is_available():
                logger.info(f"Session {session_id}: Starting terminal with CLI tool")

                # Tool-specific resume. Both CLIs need a history guard:
                # Claude exits on empty `--continue`, and Codex would open
                # the auth/resume picker instead of replaying the stored prompt.
                use_resume = False
                if state.cli_tool == CLIToolType.CLAUDE:
                    claude_config_dir = Path(extra_env["CLAUDE_CONFIG_DIR"])
                    if claude_config_dir.exists():
                        projects_dir = claude_config_dir / "projects"
                        if projects_dir.exists():
                            for project_entry in projects_dir.iterdir():
                                if project_entry.is_dir() and any(project_entry.glob("*.jsonl")):
                                    use_resume = True
                                    break
                        if not use_resume:
                            logger.info(
                                f"Session {session_id}: No conversation history in "
                                f"claude-config/projects, skipping --continue"
                            )
                elif state.cli_tool == CLIToolType.CODEX:
                    codex_home = Path(extra_env["CODEX_HOME"])
                    use_resume = self._has_codex_history(codex_home)
                    if not use_resume:
                        logger.info(
                            f"Session {session_id}: No Codex conversation history, "
                            f"starting without resume"
                        )

                launch_prompt = initial_prompt
                if not use_resume and not launch_prompt:
                    launch_prompt = state.initial_prompt

                cli_command = self.cli_tool_manager.get_cli_command(
                    state.cli_tool,
                    extra_env=extra_env,
                    initial_prompt=launch_prompt,
                    is_resume=use_resume,
                )

                try:
                    # Pass env vars both via /usr/bin/env in command AND to ttyd process
                    # CUDA_VISIBLE_DEVICES must be in ttyd's env to propagate to child
                    tmux_session_name = f"ammo-{session_id[:12]}"
                    terminal_port = await self.terminal_manager.start_terminal_with_command(
                        session_id=session_id,
                        command=cli_command,
                        working_dir=Path(state.worktree_path),
                        env=extra_env,  # Pass env to ttyd process for proper propagation
                        title=None,  # Disabled: --title flag causes execvp issues with ttyd
                        tmux_session_name=tmux_session_name,
                    )
                    state.terminal_port = terminal_port
                    terminal_started = True
                    logger.info(f"Session {session_id}: Terminal started on port {terminal_port}")
                except TerminalError as e:
                    terminal_started = False
                    logger.warning(f"Session {session_id}: Failed to start terminal - {e}")
                    state.error = f"Terminal failed to start: {e}"
                    # Release GPUs acquired earlier — session stays PAUSED
                    # and must not hold GPU allocations indefinitely
                    if state.gpu_ids:
                        logger.info(
                            f"Session {session_id}: Releasing GPUs {state.gpu_ids} "
                            f"due to terminal failure"
                        )
                        self._release_gpus(session_id)
                        state.gpu_ids = []

            # Only mark ACTIVE if terminal started successfully.
            # If terminal failed, keep PAUSED so the user can retry resume.
            if terminal_started:
                state.status = SessionStatus.ACTIVE
            else:
                # Keep state as PAUSED (don't change status)
                logger.warning(
                    f"Session {session_id}: Staying PAUSED due to terminal failure"
                )
            state.last_accessed = time.time()

            self._save_session_state(state)

            # Re-register with inactivity monitor
            if self.inactivity_monitor:
                self.inactivity_monitor.register_session(
                    session_id,
                    timeout_mins=state.inactivity_timeout_mins,
                )

            message = "Session resumed."
            if restored_from_s3:
                message += " (Restored from S3)"

            return state.to_create_response(message=message)

        except Exception as e:
            logger.error(f"Session {session_id}: Resume failed - {e}")
            # Stop terminal if it was started — prevents orphan ttyd/tmux
            # blocking future resume attempts (start_terminal_with_command
            # early-returns when session is still in _terminals)
            try:
                await self.terminal_manager.stop_terminal(session_id)
            except Exception:
                pass  # Best-effort; don't mask the original error
            # Release any acquired GPUs
            self._release_gpus(session_id)
            # Bug #22 fix: Clear stale gpu_ids, reset to PAUSED, and persist so
            # GET /sessions/{id} doesn't show GPUs the session no longer holds
            # or an ACTIVE status with no resources.
            state.gpu_ids = []
            state.status = SessionStatus.PAUSED
            self._save_session_state(state)
            raise SessionError(f"Failed to resume session: {e}")

    def _fix_claude_project_dir_after_s3_restore(self, state: SessionState) -> None:
        """
        Fix Claude Code project directory name after cross-host S3 restore.

        Claude Code stores conversation data in:
            claude-config/projects/{worktree-path-encoded}/
        where path encoding is: replace ALL '/' with '-' (including leading '/').

        After S3 restore, the worktree path may differ from the original
        host (e.g., /data/sessions/ vs /local/sessions/), causing the encoded
        directory name to not match. Claude starts fresh instead of resuming.

        This method detects the mismatch and renames the project directory
        (and updates .claude.json) so Claude Code finds its conversation data.
        """
        if not state.session_dir or not state.worktree_path:
            return

        claude_config_dir = Path(state.session_dir) / "claude-config"
        projects_dir = claude_config_dir / "projects"

        if not projects_dir.exists():
            logger.debug(f"Session {state.session_id}: No claude-config/projects dir, skipping project dir fix")
            return

        # List subdirectories in projects/
        project_dirs = [d for d in projects_dir.iterdir() if d.is_dir()]

        if len(project_dirs) == 0:
            logger.debug(f"Session {state.session_id}: No project directories found, skipping")
            return

        # Claude Code encodes paths by replacing ALL '/' with '-', including the
        # leading '/', producing a leading '-'. Do NOT lstrip("/") here.
        expected_encoded = state.worktree_path.replace("/", "-")

        # If a directory with the expected name already exists, no rename needed
        if any(d.name == expected_encoded for d in project_dirs):
            logger.debug(f"Session {state.session_id}: Project dir already matches worktree path")
            return

        # Find the main worktree project dir among potentially many dirs.
        # After S3 restore, the projects/ dir may contain:
        #   - The main worktree dir (old host path encoding, needs rename)
        #   - Agent worktree dirs (e.g. ...-worktree--claude-worktrees-op001-...)
        #   - Corrupted dirs (git stdout as dirname: Preparing-worktree-..., HEAD-is-now-at-...)
        #   - Base repo dirs (e.g. -data-repos-vllm)
        # We only rename the main worktree dir: contains session_id, ends with "-worktree".
        candidates = [
            d for d in project_dirs
            if state.session_id in d.name
            and d.name.endswith("-worktree")
            and "Preparing-worktree" not in d.name
            and "HEAD-is-now-at" not in d.name
        ]

        if len(candidates) != 1:
            logger.warning(
                f"Session {state.session_id}: Cannot identify main project dir for rename "
                f"(found {len(candidates)} candidates among {len(project_dirs)} dirs), skipping"
            )
            return

        # Rename the identified main worktree project dir
        existing_dir = candidates[0]
        new_dir = projects_dir / expected_encoded
        logger.info(
            f"Session {state.session_id}: S3 restore project dir rename: "
            f"{existing_dir.name} -> {expected_encoded}"
        )
        existing_dir.rename(new_dir)

        # Update .claude.json if it has a projects key with the old path
        claude_json_path = claude_config_dir / ".claude.json"
        if not claude_json_path.exists():
            logger.debug(f"Session {state.session_id}: No .claude.json, skipping JSON update")
            return

        try:
            claude_json = json.loads(claude_json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Session {state.session_id}: Failed to read .claude.json: {e}")
            return

        if "projects" not in claude_json or not isinstance(claude_json["projects"], dict):
            logger.debug(f"Session {state.session_id}: No projects key in .claude.json, skipping")
            return

        # Find the old path key (the one that doesn't match current worktree_path)
        old_keys = [k for k in claude_json["projects"] if k != state.worktree_path]
        if len(old_keys) == 1:
            old_key = old_keys[0]
            project_data = claude_json["projects"].pop(old_key)
            claude_json["projects"][state.worktree_path] = project_data
            claude_json_path.write_text(json.dumps(claude_json, indent=2))
            logger.info(
                f"Session {state.session_id}: Updated .claude.json projects key: "
                f"{old_key} -> {state.worktree_path}"
            )

    def _get_recovery_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session recovery lock."""
        if session_id not in self._recovery_locks:
            self._recovery_locks[session_id] = asyncio.Lock()
        return self._recovery_locks[session_id]

    def _build_extra_env(self, state: SessionState) -> Dict[str, str]:
        """Build extra environment variables for CLI tool launch."""
        extra_env = {"HOME": "/home/session_user"}
        if state.gpu_ids:
            extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in state.gpu_ids)
        else:
            extra_env["CUDA_VISIBLE_DEVICES"] = "-1"

        if state.cli_tool == CLIToolType.CLAUDE:
            extra_env["CLAUDE_CONFIG_DIR"] = str(self._prepare_claude_config(state))
            # Per-agent-type teammate wrapper. The orchestrator (team-lead)
            # reads CLAUDE_CODE_TEAMMATE_COMMAND from its OWN process env and
            # uses it as the exec path for every tmux teammate it spawns, so the
            # wrapper is the one seam where per-agent-type settings can be
            # applied at spawn time (see scripts/teammate-cmd-wrapper.sh). This
            # var must live in the orchestrator's own process env (here / the
            # /usr/bin/env argv prefix), NOT in settings.local.json.
            extra_env["CLAUDE_CODE_TEAMMATE_COMMAND"] = self._teammate_cmd_path()
        elif state.cli_tool == CLIToolType.CODEX:
            extra_env["CODEX_HOME"] = str(self._prepare_codex_home(state))
            # Auth is materialized into CODEX_HOME/auth.json above. Blank the
            # raw key so ttyd/tmux/Codex and inherited shell tool environments
            # do not expose the server's API key.
            extra_env["OPENAI_API_KEY"] = ""

        hf_cache_dir = Path(state.session_dir) / "hf-cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        extra_env["HF_HOME"] = str(hf_cache_dir)

        # Stable server-owned campaign identity. Codex's hook payload session_id
        # is a different UUID (the root Codex thread) and is bound separately
        # into state.codex_thread_id by the PostToolUse validator.
        extra_env["AMMO_SESSION_ID"] = str(state.session_id)
        extra_env["AMMO_GPU_RES_DIR"] = f"/tmp/ammo_gpu_res_{state.session_id}"
        if state.tp_size and state.tp_size > 0:
            extra_env["AMMO_TP_SIZE"] = str(state.tp_size)
            if state.dp_size and state.dp_size > 0:
                extra_env["AMMO_DP_SIZE"] = str(state.dp_size)

        # Fork sessions run a source build; keep on-demand rebuilds source-based.
        if state.vllm_fork_url:
            extra_env["VLLM_USE_PRECOMPILED"] = "0"

        return extra_env

    def _chown_session_to_user(self, state: SessionState) -> None:
        """
        Chown the session directory tree to session_user (UID 1000).

        The server creates session files (worktree, claude-config, hf-cache,
        .claude/ settings) as root. Before launching the CLI tool as
        session_user via ``su``, the entire session directory must be owned
        by session_user so that Claude Code can write logs, conversation
        history, and tool artefacts.

        This covers the full session directory (which also contains
        claude-config and hf-cache).
        """
        session_dir = state.session_dir
        if not session_dir:
            return
        try:
            subprocess.run(
                ["chown", "-R", "1000:1000", session_dir],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            logger.debug(
                f"Session {state.session_id}: Chowned {session_dir} to session_user"
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(
                f"Session {state.session_id}: Failed to chown session dir: {e}"
            )

    async def ensure_terminal_healthy(self, session_id: str) -> Optional[int]:
        """
        Check terminal health and auto-recover if dead.

        Only recovers for ACTIVE sessions. Uses tool-specific resume
        semantics when the backing CLI has resumable conversation state.

        Args:
            session_id: Session identifier

        Returns:
            Terminal port if healthy/recovered, None if cannot recover
        """
        state = self._sessions.get(session_id)
        if not state:
            return None

        # Only auto-recover ACTIVE sessions
        if state.status != SessionStatus.ACTIVE:
            return None

        # Check if terminal is alive
        if self.terminal_manager.is_terminal_running(session_id):
            return state.terminal_port

        # Terminal is dead - need recovery
        if not self.terminal_manager.is_available():
            logger.warning(f"Session {session_id}: ttyd not available for recovery")
            return None

        # Use per-session lock to prevent concurrent recovery
        async with self._get_recovery_lock(session_id):
            # Double-check after acquiring lock (another task may have recovered)
            if self.terminal_manager.is_terminal_running(session_id):
                return state.terminal_port

            logger.info(f"Session {session_id}: Terminal dead, attempting auto-recovery")

            # Clean up stale entry
            self.terminal_manager.cleanup_dead_terminal(session_id)

            try:
                extra_env = self._build_extra_env(state)
                tmux_name = f"ammo-{session_id[:12]}"
                # Derive dedicated socket path for this session
                tmux_socket_path = f"/tmp/{session_id}/tmux.sock"

                # Check if tmux session is still alive (Claude Code still running)
                tmux_alive = (
                    hasattr(self.terminal_manager, '_tmux_session_exists')
                    and self.terminal_manager._tmux_session_exists(
                        tmux_name, socket_path=tmux_socket_path
                    )
                )

                if tmux_alive:
                    # tmux session survived -- just restart ttyd with attach
                    logger.info(
                        f"Session {session_id}: tmux session {tmux_name} alive, "
                        f"reattaching ttyd"
                    )
                    terminal_port = await self.terminal_manager.restart_ttyd_with_tmux_attach(
                        session_id=session_id,
                        tmux_session_name=tmux_name,
                        working_dir=Path(state.worktree_path),
                        env=extra_env,
                    )
                else:
                    # Both dead -- full restart with tmux new-session.
                    # Both CLIs only resume when there is real conversation
                    # history. This preserves the initial prompt on first
                    # terminal attach, before ttyd has created tmux.
                    use_resume = False
                    if state.cli_tool == CLIToolType.CLAUDE:
                        claude_config_dir = Path(extra_env["CLAUDE_CONFIG_DIR"])
                        if claude_config_dir.exists():
                            projects_dir = claude_config_dir / "projects"
                            if projects_dir.exists():
                                for proj_dir in projects_dir.iterdir():
                                    if proj_dir.is_dir() and any(proj_dir.glob("*.jsonl")):
                                        use_resume = True
                                        break
                    elif state.cli_tool == CLIToolType.CODEX:
                        use_resume = self._has_codex_history(Path(extra_env["CODEX_HOME"]))

                    # Re-send initial prompt if stored and not resuming.
                    # The prompt is lost when terminal dies before the CLI
                    # processes it.
                    recovery_prompt = None if use_resume else state.initial_prompt

                    cli_command = self.cli_tool_manager.get_cli_command(
                        state.cli_tool,
                        extra_env=extra_env,
                        is_resume=use_resume,
                        initial_prompt=recovery_prompt,
                    )

                    terminal_port = await self.terminal_manager.start_terminal_with_command(
                        session_id=session_id,
                        command=cli_command,
                        working_dir=Path(state.worktree_path),
                        env=extra_env,
                        title=None,
                        tmux_session_name=tmux_name,
                    )

                state.terminal_port = terminal_port
                state.last_accessed = time.time()
                self._save_session_state(state)

                logger.info(f"Session {session_id}: Auto-recovered terminal on port {terminal_port}")
                return terminal_port

            except TerminalError as e:
                logger.error(f"Session {session_id}: Auto-recovery failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Session {session_id}: Unexpected error during recovery: {e}")
                return None

    async def terminate_session(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
    ) -> SessionActionResponse:
        """
        Terminate a session and clean up all resources.

        Kills ttyd, the session tmux server, and the detached CLI daemon tree
        (see ``_stop_session_processes``); a run started through
        reserved_detached_run.sh survives.

        Args:
            session_id: Session identifier
            owner_id: Optional client ID for ownership validation

        Returns:
            SessionActionResponse with result
        """
        state = self._validate_ownership(session_id, owner_id)

        logger.info(f"Terminating session {session_id}")

        # Unregister from inactivity monitor first
        if self.inactivity_monitor:
            self.inactivity_monitor.unregister_session(session_id)

        # Stop ttyd, tmux, and the detached CLI daemon tree BEFORE releasing GPUs,
        # so no surviving agent runs GPU work against reallocated locks.
        await self._stop_session_processes(state)

        # Cancel an in-flight fork build so it can't resurrect this session
        # (set ACTIVE / start a terminal) after we release its GPUs.
        build_task = getattr(self, "_fork_build_tasks", {}).pop(session_id, None) \
            if isinstance(getattr(self, "_fork_build_tasks", None), dict) else None
        if build_task is not None and not build_task.done():
            build_task.cancel()
            try:
                await build_task
            except (asyncio.CancelledError, Exception):
                pass
            logger.info(f"Session {session_id}: cancelled in-flight fork build task")

        # Release GPUs (only after terminal is confirmed stopped or force-killed)
        self._release_gpus(session_id)

        # Clean up AMMO track worktrees (if any)
        worktree_path = state.worktree_path
        if worktree_path:
            try:
                # Check if this worktree has any child worktrees (AMMO tracks).
                # Claude hooks use .claude/worktrees; Codex parity helpers use .codex/worktrees.
                ammo_track_dirs = [
                    Path(worktree_path) / ".claude" / "worktrees",
                    Path(worktree_path) / ".codex" / "worktrees",
                ]
                if any(track_dir.exists() for track_dir in ammo_track_dirs):
                    # Get the git common dir to run worktree prune
                    result = await asyncio.create_subprocess_exec(
                        "git", "-C", str(worktree_path), "rev-parse", "--git-common-dir",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await result.communicate()
                    if result.returncode == 0:
                        git_common_dir = stdout.decode().strip()
                        base_repo = str(Path(git_common_dir).parent) if git_common_dir.endswith("/.git") else git_common_dir
                        # List and remove track worktrees
                        for ammo_tracks_dir in ammo_track_dirs:
                            if not ammo_tracks_dir.exists():
                                continue
                            for track_dir in ammo_tracks_dir.iterdir():
                                if track_dir.is_dir() and (
                                    ammo_tracks_dir.name == "worktrees"
                                    and ammo_tracks_dir.parent.name == ".codex"
                                    or "ammo-track" in track_dir.name
                                ):
                                    remove_proc = await asyncio.create_subprocess_exec(
                                        "git", "-C", base_repo, "worktree", "remove", "--force", str(track_dir),
                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                                    )
                                    _, remove_stderr = await remove_proc.communicate()
                                    if remove_proc.returncode != 0:
                                        logger.warning(
                                            "Failed to remove AMMO track worktree %s for session %s: %s",
                                            track_dir,
                                            session_id,
                                            remove_stderr.decode(errors="replace").strip(),
                                        )
                        # Prune stale worktree entries
                        prune_proc = await asyncio.create_subprocess_exec(
                            "git", "-C", base_repo, "worktree", "prune",
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        _, prune_stderr = await prune_proc.communicate()
                        if prune_proc.returncode != 0:
                            logger.warning(
                                "Failed to prune AMMO track worktrees for session %s: %s",
                                session_id,
                                prune_stderr.decode(errors="replace").strip(),
                            )
                        logger.info(f"Cleaned up AMMO track worktrees for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up AMMO tracks for session {session_id}: {e}")

        # Clean up worktree and session directory
        try:
            self.worktree_manager.cleanup_session(session_id, state.repo_name)
        except Exception as e:
            logger.error(f"Failed to cleanup worktree: {e}")

        # Clean up per-session GPU reservation state dir to prevent /tmp/ accumulation
        import shutil as _shutil
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: _shutil.rmtree(f"/tmp/ammo_gpu_res_{session_id}", ignore_errors=True))

        # Bug #23 fix: Clean up tmux config/socket dir (/tmp/{session_id}/)
        await loop.run_in_executor(None, lambda: _shutil.rmtree(f"/tmp/{session_id}", ignore_errors=True))

        # Clean up the per-fork base repo if no other session still uses it.
        if state.vllm_fork_url and not self._fork_base_in_use_by_others(
            session_id, state.vllm_fork_url
        ):
            try:
                from orchestration.fork_repo_manager import get_fork_repo_manager
                fork_mgr = get_fork_repo_manager()
                await loop.run_in_executor(
                    None, lambda: fork_mgr.remove_fork_base(state.vllm_fork_url)
                )
                logger.info(f"Session {session_id}: removed fork base for {state.vllm_fork_url}")
            except Exception as e:
                logger.warning(f"Session {session_id}: fork base cleanup failed: {e}")

        # Delete from S3 if applicable
        if self.session_storage and self.session_storage.enabled:
            try:
                await self.session_storage.delete_session_from_s3(session_id)
                logger.info(f"Session {session_id}: Deleted from S3")
            except Exception as e:
                logger.warning(f"Session {session_id}: Failed to delete from S3: {e}")

        # Clean up recovery lock for this session
        self._recovery_locks.pop(session_id, None)

        # Update state
        state.status = SessionStatus.TERMINATED
        state.last_accessed = time.time()

        self._save_session_state(state)

        # Bug #9 fix: Use pop() instead of del to avoid KeyError when two
        # concurrent terminate requests race past _validate_ownership.
        self._sessions.pop(session_id, None)

        return SessionActionResponse(
            session_id=session_id,
            status=SessionStatus.TERMINATED.value,
            message="Session terminated and resources cleaned up.",
        )

    async def prepare_download(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
    ) -> SessionDownloadInfo:
        """
        Prepare session for download.

        Works for both local and cross-node scenarios:
        - If session exists locally with unsaved changes → sync to S3 first
        - If session only exists in S3 (cross-node) → create archive from S3 directly

        Steps:
        1. Check if session exists locally with newer data
        2. If yes, sync local changes to S3
        3. Create ZIP archive from S3 objects
        4. Generate presigned URL
        5. Return download info

        Args:
            session_id: Session identifier
            owner_id: Optional client ID for ownership validation

        Returns:
            SessionDownloadInfo with download URL, size, and expiration
        """
        # Check if S3 storage is configured
        if not self.session_storage or not self.session_storage.enabled:
            return SessionDownloadInfo(
                session_id=session_id,
                archive_ready=False,
                error="S3 storage not configured - downloads require S3"
            )

        # Get local state if available (may be None for cross-node)
        state = self._sessions.get(session_id)

        # Validate ownership
        if state is not None:
            # Session exists locally — use standard ownership validation
            self._validate_ownership(session_id, owner_id)
        elif owner_id is not None:
            # Cross-host restore: session not local, load metadata from S3 to check ownership
            s3_state = await self.session_storage.load_session_metadata(session_id)
            if s3_state is not None and s3_state.owner_id is not None:
                if s3_state.owner_id != owner_id:
                    raise SessionError(f"Session {session_id} not found")

        # Ensure latest data is in S3
        synced = await self.session_storage.ensure_session_synced(session_id, state)
        if not synced:
            return SessionDownloadInfo(
                session_id=session_id,
                archive_ready=False,
                error="Session not found in S3"
            )

        # Create archive from S3 objects
        archive_key = await self.session_storage.create_download_archive(session_id)
        if not archive_key:
            return SessionDownloadInfo(
                session_id=session_id,
                archive_ready=False,
                error="Failed to create download archive"
            )

        # Get presigned URL and size
        expires_in = 3600  # 1 hour
        download_url = await self.session_storage.get_download_url(session_id, expires_in)
        download_size = await self.session_storage.get_download_size(session_id)

        return SessionDownloadInfo(
            session_id=session_id,
            download_url=download_url,
            download_size_bytes=download_size,
            archive_ready=True,
            expires_at=time.time() + expires_in,
            error=None
        )

    async def cleanup_old_sessions(
        self,
        max_age_days: int = 7,
        status_filter: Optional[List[SessionStatus]] = None,
    ) -> int:
        """
        Clean up old sessions.

        Args:
            max_age_days: Maximum age in days for sessions to keep
            status_filter: Only clean sessions with these statuses
                          (default: TERMINATED, FAILED)

        Returns:
            Number of sessions cleaned up
        """
        if status_filter is None:
            status_filter = [SessionStatus.TERMINATED, SessionStatus.FAILED]

        max_age_seconds = max_age_days * 24 * 3600
        now = time.time()
        cleaned = 0

        for session_id in list(self._sessions.keys()):
            state = self._sessions[session_id]

            if state.status not in status_filter:
                continue

            age = now - state.last_accessed
            if age < max_age_seconds:
                continue

            logger.info(
                f"Cleaning up old session {session_id} "
                f"(age={age/3600:.1f}h, status={state.status.value})"
            )

            try:
                # Remove from disk
                session_dir = self.sessions_dir / session_id
                if session_dir.exists():
                    import shutil
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: shutil.rmtree(session_dir))

                # Remove from memory
                del self._sessions[session_id]
                cleaned += 1

            except Exception as e:
                logger.error(f"Failed to cleanup session {session_id}: {e}")

        if cleaned:
            logger.info(f"Cleaned up {cleaned} old sessions")

        return cleaned


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(
    sessions_dir: Optional[str] = None,
    gpu_manager: Optional["GPUResourceManager"] = None,
) -> SessionManager:
    """
    Get singleton session manager instance.

    Args:
        sessions_dir: Sessions directory (only used on first call)
        gpu_manager: GPU resource manager instance (only used on first call).
                     IMPORTANT: Pass the same instance used by JobManager to ensure
                     session GPU locks are visible to /health endpoint.

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(sessions_dir=sessions_dir, gpu_manager=gpu_manager)
    return _session_manager


def reset_session_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _session_manager
    _session_manager = None
