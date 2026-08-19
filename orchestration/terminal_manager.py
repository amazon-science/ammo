# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Terminal Manager for AI CLI sessions.

Manages web terminal access via ttyd:
- Spawns ttyd process per session attached to CLI tool's PTY
- Handles port allocation
- Tracks terminal processes
- Supports WebSocket proxying through FastAPI
"""

import asyncio
import os
import shlex
import signal
import subprocess
import shutil
import logging
import socket
from typing import Optional, Dict, Any, Set
from pathlib import Path
from dataclasses import dataclass, field

from shared.session_models import (
    DEFAULT_TERMINAL_BASE_PORT,
    MAX_TERMINAL_PORTS,
)

logger = logging.getLogger(__name__)


class TerminalError(Exception):
    """Exception raised for terminal operations."""
    pass


@dataclass
class TerminalProcess:
    """Information about a terminal process."""
    session_id: str
    port: int
    pid: int
    master_fd: int
    tmux_session_name: Optional[str] = None
    tmux_socket_path: Optional[str] = None


class TerminalManager:
    """
    Manages web terminals for AI CLI sessions.

    Uses ttyd to provide web-based terminal access to CLI tool processes.
    Each session gets its own ttyd instance on a unique port.
    """

    def __init__(
        self,
        base_port: int = DEFAULT_TERMINAL_BASE_PORT,
        max_ports: int = MAX_TERMINAL_PORTS,
    ):
        """
        Initialize terminal manager.

        Args:
            base_port: Starting port for terminal allocation
            max_ports: Maximum number of concurrent terminals
        """
        self.base_port = base_port
        self.max_ports = max_ports

        # Track active terminals
        self._terminals: Dict[str, TerminalProcess] = {}
        self._used_ports: Set[int] = set()

        # Check if ttyd is available
        self._ttyd_available = self._check_ttyd()

        # Check if tmux is available
        self._tmux_available = self._check_tmux()

        logger.info(
            f"TerminalManager initialized: base_port={base_port}, "
            f"max_ports={max_ports}, ttyd_available={self._ttyd_available}, "
            f"tmux_available={self._tmux_available}"
        )

    def _check_ttyd(self) -> bool:
        """Check if ttyd is available."""
        ttyd_path = shutil.which("ttyd")
        if ttyd_path:
            logger.debug(f"ttyd found at: {ttyd_path}")
            return True
        logger.warning("ttyd not found - terminal features will be unavailable")
        return False

    def _check_tmux(self) -> bool:
        """Check if tmux is available."""
        tmux_path = shutil.which("tmux")
        if tmux_path:
            logger.debug(f"tmux found at: {tmux_path}")
            return True
        logger.warning("tmux not found - session persistence will be unavailable")
        return False

    def _tmux_session_exists(self, tmux_session_name: str, socket_path: Optional[str] = None) -> bool:
        """Check if a tmux session exists.

        Args:
            tmux_session_name: Name of the tmux session
            socket_path: Optional dedicated socket path (-S flag)
        """
        try:
            cmd = ["tmux"]
            if socket_path:
                cmd.extend(["-S", socket_path])
            cmd.extend(["has-session", "-t", tmux_session_name])
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _kill_tmux_session(self, tmux_session_name: str, socket_path: Optional[str] = None) -> bool:
        """Kill a tmux session. Returns True if killed, False otherwise.

        Args:
            tmux_session_name: Name of the tmux session
            socket_path: Optional dedicated socket path (-S flag)
        """
        try:
            cmd = ["tmux"]
            if socket_path:
                cmd.extend(["-S", socket_path])
            cmd.extend(["kill-session", "-t", tmux_session_name])
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                logger.info(f"Killed tmux session: {tmux_session_name}")
                return True
            logger.debug(f"tmux session {tmux_session_name} not found or already dead")
            return False
        except Exception as e:
            logger.warning(f"Failed to kill tmux session {tmux_session_name}: {e}")
            return False

    def _write_hardened_tmux_config(self, session_id: str) -> Path:
        """
        Write a hardened tmux config for a dedicated per-session tmux server.

        Creates a config at /tmp/{session_id}/tmux.conf with:
        - Prefix changed to C-Space (frees C-b for application use)
        - All prefix bindings removed (unbind-key -a -T prefix)
        - default-terminal set to xterm-256color (matches outer xterm.js)
        - No status bar
        - Mouse ON by default
        - Auto-destroy when last client disconnects
        - Panes close when process exits

        By changing the prefix to C-Space instead of the default C-b, the
        Ctrl+B keypress passes straight through to the application (e.g.
        Claude Code uses Ctrl+B to background tasks). The old approach of
        unbinding C-b from the root table did not work reliably because
        tmux still intercepted the default prefix key internally.

        This config is loaded at tmux startup via -f flag, so hardening
        is applied from the very start -- no post-hoc source-file needed.

        Args:
            session_id: Session identifier

        Returns:
            Path to the written config file
        """
        config_content = """\
# Dedicated tmux server hardening (allowlist model)
# Change prefix away from C-b so Ctrl+B passes through to the application.
# Claude Code uses Ctrl+B to background tasks. The old approach of
# "unbind-key -T root C-b" did not fully free C-b because tmux still
# intercepts the default prefix key internally.
set -g prefix C-Space
# Remove ALL prefix table bindings so no tmux command is reachable
# (even via the new C-Space prefix).
unbind-key -a -T prefix
# IMPORTANT: Do NOT use "unbind-key -a -T root" — that wipes the default
# mouse event bindings (WheelUpPane, WheelDownPane, MouseDrag1Pane, etc.)
# which breaks mouse scrolling in the web terminal.
# ---- Terminal rendering fix ----
# Override default-terminal from "tmux-256color" to "xterm-256color".
# tmux-256color is missing key capabilities (bce, ech, rmam/smam) that
# xterm.js supports. Without bce (background color erase), erase operations
# don't fill with the current background color, leaving ghost text remnants.
# Without ech (erase characters), partial line clears fall back to slower
# methods that cause rendering artifacts. Since the outer terminal is always
# xterm.js (via ttyd), using xterm-256color inside tmux eliminates the
# terminfo translation layer and prevents garbled/overlapping text.
set -g default-terminal "xterm-256color"
# No status bar
set -g status off
# Pane closes when process exits
set -g remain-on-exit off
# Exit tmux server when last session ends (no orphaned tmux processes)
set -g exit-empty on
# Keep tmux session alive when browser disconnects (destroy-unattached off).
# With 'on', browser close -> ttyd detaches -> tmux destroys session -> all
# team panes are lost. With 'off', the session survives for reconnect.
# Explicit teardown happens via _kill_tmux_session on pause/terminate.
set -g destroy-unattached off
# Mouse ON by default (user toggles via API)
set -g mouse on
# ---- Auto-resize panes on split/exit/resize ----
# Automatically re-tile panes when the layout changes.
# 2 panes → side-by-side (even-horizontal); 3+ → tiled.
set-hook -g after-split-window 'if-shell -F "#{==:#{window_panes},2}" "select-layout even-horizontal" "select-layout tiled"'
set-hook -g pane-exited 'if-shell -F "#{==:#{window_panes},2}" "select-layout even-horizontal" "select-layout tiled"'
set-hook -g client-resized 'if-shell -F "#{==:#{window_panes},2}" "select-layout even-horizontal" "select-layout tiled"'
"""
        config_path = Path(f"/tmp/{session_id}/tmux.conf")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # chown the directory to session_user (uid=1000, gid=1000) so that
        # `su session_user` can create the tmux socket file inside it.
        # Without this, the directory is owned by root (server process) and
        # session_user gets "permission denied" when tmux tries to create
        # /tmp/{session_id}/tmux.sock.
        os.chown(str(config_path.parent), 1000, 1000)
        config_path.write_text(config_content)
        logger.debug(f"Wrote hardened tmux config to {config_path}")
        return config_path

    def _build_ttyd_theme_args(self) -> list[str]:
        """Return --client-option flags to apply the LIGHTGRID ANSI color theme to ttyd."""
        import json
        theme = {
            "background": "#000000", "foreground": "#c8c8d0", "cursor": "#00f3ff",
            "cursorAccent": "#000000", "selectionBackground": "rgba(0,243,255,0.15)",
            "black": "#05050a", "red": "#ff3355", "green": "#00ffb2", "yellow": "#ffaa00",
            "blue": "#00f3ff", "magenta": "#ff00aa", "cyan": "#c78aff", "white": "#e8e8f0",
            "brightBlack": "#2a2a3a", "brightRed": "#ff6680", "brightGreen": "#66ffd0",
            "brightYellow": "#ffcc55", "brightBlue": "#66f7ff", "brightMagenta": "#ff66cc",
            "brightCyan": "#dbb3ff", "brightWhite": "#ffffff",
        }
        return ["--client-option", "rendererType=canvas",
                "--client-option", "fontSize=10",
                "--client-option", f"theme={json.dumps(theme)}"]

    def _apply_tmux_hardening(self, session_id: str) -> None:
        """
        No-op. Hardening is now built into tmux startup via -f config flag.

        Retained for backward compatibility -- callers that still invoke this
        method will not break.

        Args:
            session_id: The tmux session name (unused)
        """
        pass

    def _build_tmux_command(
        self,
        tmux_session_name: str,
        inner_command: list,
        attach_only: bool = False,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Build the tmux wrapper command with dedicated server per session.

        Uses a dedicated tmux server per session via -S (socket) and loads
        a hardened config via -f (config file). This isolates bindings from
        other sessions and applies security hardening from startup.

        Uses 'tmux new-session -A' which creates-or-attaches idempotently,
        avoiding race conditions when parallel requests hit the same session.

        The inner command is written to a launcher script file at
        /tmp/{session_id}/start.sh so that each argument is properly
        shlex.quote()-d in the script. This avoids multi-layer shell
        escaping bugs when special characters (e.g. brackets in model names
        like "model-v1[1m]") pass through ttyd -> bash -> su -> tmux -> sh.

        The entire command is wrapped with `su session_user -s /bin/bash -c`
        to ensure the tmux SERVER process runs under session_user's UID.
        (ttyd's --uid flag does NOT actually change the spawned process's
        UID in ttyd 1.7.7, so su is required.)

        Args:
            tmux_session_name: Name for the tmux session
            inner_command: Command to run inside tmux
            attach_only: If True, attach to existing session
            session_id: Session ID for socket/config paths

        Returns:
            Tuple of (cmd_list, socket_path, config_path)
        """
        # Derive socket and config paths from session_id
        socket_path = f"/tmp/{session_id}/tmux.sock"
        config_path = f"/tmp/{session_id}/tmux.conf"

        if attach_only:
            tmux_cmd = f"tmux -S {socket_path} -f {config_path} attach -t {tmux_session_name}"
        else:
            # Write inner_command to a launcher script to avoid shell quoting
            # issues with special characters (e.g. brackets in model names
            # like "model-v1[1m]") and multi-layer shell escaping through
            # ttyd -> su -> bash -> tmux -> /bin/sh.
            script_path = Path(f"/tmp/{session_id}/start.sh")
            # Build the shell command with proper quoting for each argument
            script_lines = ["#!/bin/bash"]
            script_lines.append("exec " + " ".join(shlex.quote(part) for part in inner_command))
            script_content = "\n".join(script_lines) + "\n"
            script_path.write_text(script_content)
            os.chmod(str(script_path), 0o755)
            # chown to session_user so the script is executable by session_user
            os.chown(str(script_path), 1000, 1000)
            logger.debug(f"Wrote tmux launcher script to {script_path}")

            tmux_cmd = (
                f"tmux -S {socket_path} -f {config_path} "
                f"new-session -A -s {tmux_session_name} {script_path}"
            )

        # Drop the interactive agent shell to UID/GID 1000 with no retained
        # ambient capabilities. Profilers receive narrow file capabilities on
        # their immutable executables at image build time; the shell must not
        # inherit CAP_SYS_ADMIN/CAP_SYS_PTRACE because those defeat managed-hook
        # immutability.
        drop_privs_path = Path(f"/tmp/{session_id}/drop_privs.py")
        drop_privs_path.parent.mkdir(parents=True, exist_ok=True)
        drop_privs_content = '''\
#!/usr/bin/env python3
"""Drop to session_user (UID 1000) without ambient Linux capabilities."""
import os, sys

if os.geteuid() == 0:
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)

os.execvp("/bin/bash", ["/bin/bash", "-c", " ".join(sys.argv[1:])])
'''
        drop_privs_path.write_text(drop_privs_content)
        os.chmod(str(drop_privs_path), 0o755)
        logger.debug(f"Wrote privilege-drop wrapper to {drop_privs_path}")

        # CRITICAL: Return as a SINGLE shell string wrapped in ["/bin/bash", "-c", ...]
        # because ttyd splits list elements into separate argv — if we return
        # ["python3", ..., tmux_cmd], ttyd passes them as separate args.
        wrapper_cmd = f"python3 {shlex.quote(str(drop_privs_path))} {shlex.quote(tmux_cmd)}"
        cmd = ["/bin/bash", "-c", wrapper_cmd]
        return (cmd, socket_path, config_path)

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available (can be bound to)."""
        if port in self._used_ports:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if something is listening on a port (connection can be made)."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("127.0.0.1", port))
                return result == 0  # 0 means connection succeeded
        except OSError:
            return False

    def _allocate_port(self) -> int:
        """
        Allocate an available port.

        Returns:
            Available port number

        Raises:
            TerminalError: If no ports available
        """
        for offset in range(self.max_ports):
            port = self.base_port + offset
            if self._is_port_available(port):
                self._used_ports.add(port)
                return port

        raise TerminalError(f"No available ports (base={self.base_port}, max={self.max_ports})")

    def _release_port(self, port: int) -> None:
        """Release a port back to the pool."""
        self._used_ports.discard(port)

    async def start_terminal_with_command(
        self,
        session_id: str,
        command: list,
        working_dir: Path,
        env: Dict[str, str],
        title: Optional[str] = None,
        tmux_session_name: Optional[str] = None,
    ) -> int:
        """
        Start a ttyd terminal with a specific command.

        This is the preferred approach - ttyd manages the command directly.
        When tmux_session_name is provided and tmux is available, the command
        is wrapped in a tmux session for persistence across reconnections.

        Args:
            session_id: Session identifier
            command: Command and arguments to run
            working_dir: Working directory for command
            env: Environment variables
            title: Optional browser tab title
            tmux_session_name: Optional tmux session name for persistence

        Returns:
            Port number the terminal is running on

        Raises:
            TerminalError: If terminal cannot be started
        """
        if not self._ttyd_available:
            raise TerminalError("ttyd is not available")

        if session_id in self._terminals:
            # Verify the existing terminal is actually healthy before returning.
            # Without this, a stale entry (e.g., tmux dead + ttyd alive that
            # cleanup_dead_terminal missed) would silently return the old port.
            if self.is_terminal_running(session_id):
                return self._terminals[session_id].port
            # Stale entry — clean up before proceeding with new terminal
            self.cleanup_dead_terminal(session_id)

        # Allocate port
        port = self._allocate_port()

        try:
            # Determine the actual command to run (possibly tmux-wrapped)
            use_tmux = self._tmux_available and tmux_session_name is not None
            socket_path = None
            if use_tmux:
                # Write hardened config BEFORE building tmux command
                self._write_hardened_tmux_config(session_id)

                # Check if tmux session already exists on this socket
                socket_path_candidate = f"/tmp/{session_id}/tmux.sock"
                if self._tmux_session_exists(tmux_session_name, socket_path=socket_path_candidate):
                    for key, value in env.items():
                        try:
                            subprocess.run(
                                ["tmux", "-S", socket_path_candidate,
                                 "set-environment", "-t", tmux_session_name, key, value],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            pass
                # Use new-session -A (creates-or-attaches idempotently, no race condition)
                actual_command, socket_path, _config_path = self._build_tmux_command(
                    tmux_session_name, command, attach_only=False,
                    session_id=session_id,
                )
            else:
                actual_command = command

            # Build ttyd command
            # Note: --reconnect is NOT a valid ttyd option in 1.7.7
            # Reconnection is handled client-side by ttyd's web interface
            ttyd_cmd_parts = [
                "ttyd",
                "--port", str(port),
                "--writable",  # Allow input to terminal
                "--cwd", str(working_dir),  # Working directory
                "--uid", "1000",  # Run as session_user
            ]

            if title:
                ttyd_cmd_parts.extend(["--title", title])

            # Inject LIGHTGRID theme colors
            ttyd_cmd_parts.extend(self._build_ttyd_theme_args())

            # Add the command to run
            ttyd_cmd_parts.extend(["--"])
            ttyd_cmd_parts.extend(actual_command)

            cmd = ttyd_cmd_parts

            logger.info(f"Starting ttyd for session {session_id} on port {port}")
            logger.debug(f"ttyd command: {' '.join(cmd)}")

            # Prepare environment
            process_env = os.environ.copy()
            process_env.update(env)

            # Start ttyd process
            # stderr=PIPE to capture crash diagnostics (Bug E fix: was DEVNULL)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                start_new_session=True,
            )

            # Give ttyd a moment to start and bind to port
            await asyncio.sleep(1.0)

            # Check if ttyd is listening on the port (more reliable than checking process)
            if not self._is_port_in_use(port):
                # Process might have failed - capture stderr for diagnostics
                stderr_output = ""
                if process.returncode is not None and process.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                        stderr_output = stderr_data.decode(errors="replace")
                    except Exception:
                        pass
                if process.returncode is not None:
                    raise TerminalError(
                        f"ttyd exited with code {process.returncode}"
                        + (f": {stderr_output}" if stderr_output else "")
                    )
                raise TerminalError(f"ttyd not listening on port {port} after 1 second")

            # Store terminal info (including socket path for mouse mode API)
            self._terminals[session_id] = TerminalProcess(
                session_id=session_id,
                port=port,
                pid=process.pid,
                master_fd=-1,  # Not using PTY directly in this approach
                tmux_session_name=tmux_session_name if use_tmux else None,
                tmux_socket_path=socket_path,
            )

            logger.info(f"Started terminal for session {session_id}: port={port}, pid={process.pid}")

            # _apply_tmux_hardening is a no-op -- hardening is built into -f config
            if use_tmux:
                self._apply_tmux_hardening(tmux_session_name)

            return port

        except Exception as e:
            self._release_port(port)
            raise TerminalError(f"Failed to start terminal: {e}")

    async def restart_ttyd_with_tmux_attach(
        self,
        session_id: str,
        tmux_session_name: str,
        working_dir: Path,
        env: Dict[str, str],
    ) -> int:
        """
        Restart ttyd pointing at an existing tmux session.

        Used when ttyd dies but tmux session is still alive.
        The command is just `tmux attach -t {name}`.

        Args:
            session_id: Session identifier
            tmux_session_name: Name of the existing tmux session
            working_dir: Working directory
            env: Environment variables

        Returns:
            The new port number

        Raises:
            TerminalError: If tmux session doesn't exist or ttyd fails to start
        """
        # Derive socket path for dedicated tmux server
        socket_path = f"/tmp/{session_id}/tmux.sock"

        if not self._tmux_session_exists(tmux_session_name, socket_path=socket_path):
            raise TerminalError(
                f"tmux session '{tmux_session_name}' does not exist, cannot reattach"
            )
        # NOTE: Known TOCTOU — tmux could die between this check and the ttyd exec
        # below. If that happens, ttyd fails to attach, _is_port_in_use() returns
        # False, and a TerminalError is raised (handled gracefully by caller).

        # Update tmux environment before attaching (using dedicated socket)
        for key, value in env.items():
            try:
                subprocess.run(
                    ["tmux", "-S", socket_path,
                     "set-environment", "-t", tmux_session_name, key, value],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        # Clean up any stale terminal entry
        self.cleanup_dead_terminal(session_id)

        # Allocate port
        port = self._allocate_port()

        try:
            attach_command, socket_path, _config_path = self._build_tmux_command(
                tmux_session_name, [], attach_only=True,
                session_id=session_id,
            )

            ttyd_cmd_parts = [
                "ttyd",
                "--port", str(port),
                "--writable",
                "--cwd", str(working_dir),
            ]
            ttyd_cmd_parts.extend(self._build_ttyd_theme_args())
            ttyd_cmd_parts.extend(["--"])
            ttyd_cmd_parts.extend(attach_command)

            logger.info(
                f"Restarting ttyd for session {session_id} on port {port} "
                f"(tmux attach to {tmux_session_name})"
            )

            process_env = os.environ.copy()
            process_env.update(env)

            process = await asyncio.create_subprocess_exec(
                *ttyd_cmd_parts,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=process_env,
                start_new_session=True,
            )

            await asyncio.sleep(1.0)

            if not self._is_port_in_use(port):
                if process.returncode is not None:
                    raise TerminalError(f"ttyd exited with code {process.returncode}")
                raise TerminalError(f"ttyd not listening on port {port} after 1 second")

            self._terminals[session_id] = TerminalProcess(
                session_id=session_id,
                port=port,
                pid=process.pid,
                master_fd=-1,
                tmux_session_name=tmux_session_name,
                tmux_socket_path=socket_path,
            )

            logger.info(
                f"Restarted ttyd for session {session_id}: port={port}, pid={process.pid}"
            )
            return port

        except Exception as e:
            self._release_port(port)
            raise TerminalError(f"Failed to restart ttyd with tmux attach: {e}")

    def get_terminal_url(self, session_id: str, host: str = "localhost") -> Optional[str]:
        """
        Get the URL for a session's terminal.

        Args:
            session_id: Session identifier
            host: Host for URL (default: localhost)

        Returns:
            Terminal URL or None if not running
        """
        terminal = self._terminals.get(session_id)
        if not terminal:
            return None
        return f"http://{host}:{terminal.port}/"

    def get_terminal_port(self, session_id: str) -> Optional[int]:
        """Get the port for a session's terminal."""
        terminal = self._terminals.get(session_id)
        return terminal.port if terminal else None

    def is_terminal_running(self, session_id: str) -> bool:
        """Check if terminal is running for a session."""
        terminal = self._terminals.get(session_id)
        if not terminal:
            return False

        try:
            # Check if ttyd process is still running
            os.kill(terminal.pid, 0)
        except (ProcessLookupError, PermissionError):
            return False

        # Also verify tmux session is alive — ttyd stays up showing an error
        # when its inner tmux command exits, which is indistinguishable from
        # a healthy terminal by PID check alone.
        if terminal.tmux_session_name and not self._tmux_session_exists(
            terminal.tmux_session_name, socket_path=terminal.tmux_socket_path
        ):
            return False

        return True

    def cleanup_dead_terminal(self, session_id: str) -> bool:
        """
        Check if terminal is dead and clean up stale state if so.

        A terminal is considered dead if:
        - ttyd process has exited (PID not found), OR
        - ttyd is alive but its tmux session has died (stale ttyd showing
          "[Process exited]" — happens when Claude Code exits and tmux
          destroys the pane via remain-on-exit off + exit-empty on)

        Args:
            session_id: Session identifier

        Returns:
            True if terminal was dead and cleaned up, False if alive or unknown
        """
        terminal = self._terminals.get(session_id)
        if not terminal:
            return False

        try:
            os.kill(terminal.pid, 0)
            # ttyd PID is alive — check if tmux session is also alive.
            # When Claude Code exits, tmux destroys the pane (remain-on-exit
            # off) and the server exits (exit-empty on), but ttyd stays alive
            # showing "[Process exited]". This stale ttyd blocks recovery
            # because start_terminal_with_command() early-returns when the
            # session is still in _terminals.
            if terminal.tmux_session_name and not self._tmux_session_exists(
                terminal.tmux_session_name, socket_path=terminal.tmux_socket_path
            ):
                logger.info(
                    f"Session {session_id}: tmux session dead but ttyd alive "
                    f"(pid={terminal.pid}), killing stale ttyd"
                )
                try:
                    os.kill(terminal.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Raced with natural exit
                try:
                    os.waitpid(terminal.pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass  # Already reaped or not our child
                self._terminals.pop(session_id, None)
                self._release_port(terminal.port)
                return True
            return False  # Both ttyd and tmux alive
        except PermissionError:
            return False  # Treat as alive (can't verify)
        except ProcessLookupError:
            # Dead - clean up
            self._terminals.pop(session_id, None)
            self._release_port(terminal.port)
            logger.info(
                f"Cleaned up dead terminal for session {session_id} "
                f"(pid={terminal.pid}, port={terminal.port})"
            )
            # Note: we do NOT kill tmux here. restart_ttyd_with_tmux_attach calls
            # cleanup_dead_terminal first and depends on tmux surviving so the new
            # ttyd can reattach to the user's still-running Claude session.
            # Tmux cleanup happens in stop_terminal() during pause/terminate.
            return True

    async def stop_terminal(self, session_id: str, force: bool = False) -> bool:
        """
        Stop a session's terminal.

        Args:
            session_id: Session identifier
            force: Use SIGKILL instead of SIGTERM

        Returns:
            True if stopped successfully
        """
        terminal = self._terminals.pop(session_id, None)
        if not terminal:
            logger.debug(f"No terminal to stop for session {session_id}")
            return True

        try:
            # Send signal
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(terminal.pid, sig)
            logger.info(f"Sent {sig.name} to ttyd process {terminal.pid}")

            # Wait a bit for process to terminate
            await asyncio.sleep(0.5)

            # Check if still running and force kill
            try:
                os.kill(terminal.pid, 0)
                if not force:
                    os.kill(terminal.pid, signal.SIGKILL)
                    await asyncio.sleep(0.5)
            except ProcessLookupError:
                pass

            # Reap the child process to prevent zombie accumulation
            try:
                os.waitpid(terminal.pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass  # Already reaped or not our child

            # Kill tmux session if present (best effort, using dedicated socket)
            if terminal.tmux_session_name:
                self._kill_tmux_session(terminal.tmux_session_name, socket_path=terminal.tmux_socket_path)

            # Release port
            self._release_port(terminal.port)

            logger.info(f"Stopped terminal for session {session_id}")
            return True

        except ProcessLookupError:
            logger.debug(f"ttyd process {terminal.pid} already terminated")
            # Still try to reap in case it's a zombie
            try:
                os.waitpid(terminal.pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
            # Still clean up tmux session if present
            if terminal.tmux_session_name:
                self._kill_tmux_session(terminal.tmux_session_name, socket_path=terminal.tmux_socket_path)
            self._release_port(terminal.port)
            return True
        except Exception as e:
            logger.error(f"Failed to stop terminal: {e}")
            # Best-effort zombie reap
            try:
                os.waitpid(terminal.pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
            # Still try tmux cleanup
            if terminal.tmux_session_name:
                self._kill_tmux_session(terminal.tmux_session_name, socket_path=terminal.tmux_socket_path)
            self._release_port(terminal.port)
            return False

    def get_tmux_session_name(self, session_id: str) -> Optional[str]:
        """Get the tmux session name for a session, or None if not available."""
        terminal = self._terminals.get(session_id)
        return terminal.tmux_session_name if terminal else None

    def get_tmux_socket_path(self, session_id: str) -> Optional[str]:
        """Get the tmux socket path for a session, or None if not available."""
        terminal = self._terminals.get(session_id)
        return terminal.tmux_socket_path if terminal else None

    async def get_tmux_mouse_mode(self, session_id: str) -> str:
        """Query tmux mouse mode for a specific session. Returns 'on' or 'off'.

        Uses the dedicated socket (-S) if available for the session.
        Falls back to 'on' if no socket path (pre-migration sessions).
        """
        tmux_name = self.get_tmux_session_name(session_id)
        if not tmux_name:
            return "on"
        socket_path = self.get_tmux_socket_path(session_id)
        if not socket_path:
            return "on"
        try:
            cmd = ["tmux", "-S", socket_path, "show-options", "-t", tmux_name, "-v", "mouse"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return stdout.decode().strip() if proc.returncode == 0 else "on"
        except Exception:
            return "on"

    async def set_tmux_mouse_mode(self, session_id: str, mode: str) -> str:
        """Set tmux mouse mode for a specific session. Returns the mode that was set.

        Uses the dedicated socket (-S) if available for the session.
        """
        tmux_name = self.get_tmux_session_name(session_id)
        if not tmux_name:
            raise ValueError("No tmux session found")

        socket_path = self.get_tmux_socket_path(session_id)

        # Check if tmux session is actually running (it only starts when browser connects)
        if not self._tmux_session_exists(tmux_name, socket_path=socket_path):
            raise ValueError("Terminal not connected yet — open the terminal first")

        if mode == "toggle":
            current = await self.get_tmux_mouse_mode(session_id)
            mode = "off" if current == "on" else "on"

        if mode not in ("on", "off"):
            raise ValueError(f"Invalid mouse mode: {mode}")

        cmd = ["tmux"]
        if socket_path:
            cmd.extend(["-S", socket_path])
        cmd.extend(["set-option", "-t", tmux_name, "mouse", mode])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode != 0:
            raise RuntimeError(f"tmux error: {stderr.decode().strip()}")

        logger.info(f"Set tmux mouse mode to '{mode}' for session {session_id} (tmux:{tmux_name})")
        return mode

    async def cleanup(self) -> None:
        """Stop all running terminals."""
        for session_id in list(self._terminals.keys()):
            await self.stop_terminal(session_id, force=True)

    def is_available(self) -> bool:
        """Check if terminal manager is functional (ttyd available)."""
        return self._ttyd_available


# Singleton instance
_terminal_manager: Optional[TerminalManager] = None


def get_terminal_manager(
    base_port: int = DEFAULT_TERMINAL_BASE_PORT,
    max_ports: int = MAX_TERMINAL_PORTS,
) -> TerminalManager:
    """
    Get singleton terminal manager instance.

    Args:
        base_port: Base port for allocation (only used on first call)
        max_ports: Maximum ports (only used on first call)

    Returns:
        TerminalManager instance
    """
    global _terminal_manager
    if _terminal_manager is None:
        _terminal_manager = TerminalManager(
            base_port=base_port,
            max_ports=max_ports,
        )
    return _terminal_manager


async def reset_terminal_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _terminal_manager
    if _terminal_manager:
        await _terminal_manager.cleanup()
    _terminal_manager = None
