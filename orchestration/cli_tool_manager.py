# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
CLI Tool Manager for AI CLI sessions.

Manages AI CLI tools (Claude Code, Codex CLI) for interactive sessions:
- Copies pre-configured template files to worktree
- Injects server-provided credentials via environment variables
- Builds the launch argv that terminal_manager runs under ttyd/tmux

The CLI process itself is started by terminal_manager, not here.
"""

import os
import shutil
import logging
import json
import tempfile
from typing import Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass, field

from shared.session_models import CLIToolType
from shared.constants import SESSION_DATA_DIR

logger = logging.getLogger(__name__)


class CLIToolError(Exception):
    """Exception raised for CLI tool operations."""
    pass


@dataclass
class CLIToolConfig:
    """Configuration for a CLI tool."""
    tool_type: CLIToolType
    command: str
    env_vars: Dict[str, str] = field(default_factory=dict)
    startup_flags: List[str] = field(default_factory=list)


# Default configurations for supported CLI tools
# CC 2.1.114+ ships as a native ELF binary (bin/claude.exe), not a Node.js script.
# The /usr/bin/claude symlink points directly at the ELF, so no shebang workaround needed.
CLI_TOOL_CONFIGS = {
    CLIToolType.CLAUDE: CLIToolConfig(
        tool_type=CLIToolType.CLAUDE,
        command="/usr/bin/claude",
        env_vars={
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_COST_WARNINGS": "1",
        },
        # Permissions managed via:
        # 1. /etc/claude-code/managed-settings.json (highest precedence, auto-trusted)
        # 2. ~/.claude.json projects[path].hasTrustDialogAccepted (skip trust prompt)
        # 3. .claude/settings.json in worktree (allow/deny rules)
        startup_flags=[],
    ),
    CLIToolType.CODEX: CLIToolConfig(
        tool_type=CLIToolType.CODEX,
        command="/usr/bin/codex",
        env_vars={},  # OPENAI_API_KEY set separately
        # AMMO command hooks are provisioned by the image as root-owned managed
        # hooks under /opt and enforced by /etc/codex/requirements.toml.
        # Project hooks remain agent-readable but are not an authority surface.
        startup_flags=[],
    ),
}


class CLIToolManager:
    """
    Manages AI CLI tools for sessions.

    Handles:
    - Template file copying to worktree
    - Environment variable injection
    - Launch argv construction for ttyd/tmux
    """

    def get_cli_command(
        self,
        tool_type: CLIToolType,
        extra_env: Optional[Dict[str, str]] = None,
        initial_prompt: Optional[str] = None,
        is_resume: bool = False,
    ) -> List[str]:
        """
        Get the command to launch a CLI tool with environment variables embedded.

        Uses /usr/bin/env to set environment variables directly in the command,
        ensuring they propagate to the child process when spawned by ttyd.
        This bypasses shell indirection issues that prevent env var propagation.

        Args:
            tool_type: Type of CLI tool
            extra_env: Additional environment variables to set (e.g., CUDA_VISIBLE_DEVICES)
            initial_prompt: Optional initial prompt to start the CLI with (Claude Code supports
                           passing the prompt as a positional argument)
            is_resume: If True, add the tool-specific resume flag/subcommand

        Returns:
            Command as list of strings starting with /usr/bin/env
        """
        config = CLI_TOOL_CONFIGS.get(tool_type)
        if not config:
            raise CLIToolError(f"Unsupported CLI tool: {tool_type}")

        # Check if CLI tool is available
        which_result = shutil.which(config.command)
        if not which_result:
            raise CLIToolError(f"CLI tool not found: {config.command}")

        # Build command with /usr/bin/env to embed environment variables
        # This ensures env vars reach the child process spawned by ttyd
        cmd_parts = ["/usr/bin/env"]

        # Add config env vars (DISABLE_AUTOUPDATER, DISABLE_COST_WARNINGS, etc.)
        for key, value in config.env_vars.items():
            cmd_parts.append(f"{key}={value}")

        # Add extra env vars (CUDA_VISIBLE_DEVICES, HOME, etc.)
        if extra_env:
            for key, value in extra_env.items():
                cmd_parts.append(f"{key}={value}")

        # Add the actual command and flags
        cmd_parts.append(config.command)
        cmd_parts.extend(config.startup_flags)

        # Add tool-specific automatic resume flags.
        # Claude resumes from CLAUDE_CONFIG_DIR with --continue; Codex resumes
        # the latest interactive session in CODEX_HOME with `resume --last`.
        if is_resume and tool_type == CLIToolType.CLAUDE:
            cmd_parts.append("--continue")
        elif is_resume and tool_type == CLIToolType.CODEX:
            cmd_parts.extend(["resume", "--last"])

        # Append initial prompt as positional argument
        # Claude Code supports: `claude "your prompt here"` to start interactive mode
        # with the prompt pre-entered
        if initial_prompt:
            cmd_parts.append(initial_prompt)

        return cmd_parts

    def _create_sandboxed_settings(
        self,
        worktree_path: Path,
        session_id: str,
        settings_dst: Path,
    ) -> None:
        """
        Create sandboxed settings.json with session-specific allowed directories.

        Args:
            worktree_path: Path to session worktree
            session_id: Session identifier
            settings_dst: Destination path for settings file
        """
        # Server template paths
        server_dir = Path(__file__).parent.parent
        docker_base = "/app"
        ai_cli_session_dir = "ai_cli_session"

        # Load template settings (settings.local.json has permissions, hooks, model, env).
        # We only need the permissions block here — hooks/model/env stay in
        # settings.local.json (copied by shutil.copytree). Without stripping them,
        # Claude Code merges both files additively and every hook fires twice.
        settings_src_docker = f"{docker_base}/{ai_cli_session_dir}/.claude/settings.local.json"
        settings_src_local = server_dir / ai_cli_session_dir / ".claude" / "settings.local.json"
        settings_src = Path(settings_src_docker) if os.path.exists(settings_src_docker) else settings_src_local

        if settings_src.exists():
            with open(settings_src) as f:
                settings = json.load(f)
            for key in ("hooks", "model", "effortLevel", "env"):
                settings.pop(key, None)
        else:
            # Fallback to minimal settings. No Bash(tmux:*) deny rule: the
            # template dropped it in 30a6b3e so agents can close stuck panes.
            # Session isolation is owned by terminal_manager's per-session tmux
            # socket plus its hardened config.
            settings = {
                "permissions": {
                    "allow": [],
                    "deny": [],
                    "defaultMode": "acceptEdits",
                    "additionalDirectories": []
                }
            }

        # Inject session-specific allowed directories
        # Allow full access to worktree and /tmp
        worktree_str = str(worktree_path)
        session_dirs = [
            worktree_str,
            f"{worktree_str}/**",  # All subdirectories
            "/tmp",
            f"/tmp/{session_id}/**",  # Session-specific temp dir
        ]

        # Update additionalDirectories
        existing_dirs = settings.get("permissions", {}).get("additionalDirectories", [])
        settings.setdefault("permissions", {})["additionalDirectories"] = list(set(existing_dirs + session_dirs))

        # Add deny rules for other sessions (security)
        deny_patterns = settings.get("permissions", {}).get("deny", [])
        # Deny access to other sessions' data (but allow own session)
        deny_patterns.extend([
            f"Read({SESSION_DATA_DIR}/**/session.json)",  # Deny reading other session configs
            f"Write({SESSION_DATA_DIR}/**/)",  # Deny writing to other sessions
        ])
        settings["permissions"]["deny"] = list(set(deny_patterns))

        # Write customized settings (atomic write-then-rename to avoid partial files on crash)
        settings_dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=str(settings_dst.parent))
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(settings, f, indent=2)
            os.replace(tmp_path, settings_dst)
        except Exception:
            os.unlink(tmp_path)
            raise
        logger.debug(f"Created sandboxed settings: {settings_dst}")

    def refresh_session_env(
        self,
        worktree_path: Path,
        session_id: str,
        gpu_ids: List[int],
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
    ) -> None:
        """
        Write the per-session dynamic env into .claude/settings.local.json.

        Claude Code subagents (Task tool) read env vars from the
        settings.local.json "env" section, NOT from the process environment.
        These five keys change per lifecycle transition, so create AND resume
        must both call this. Resume can land a different GPU set or a different
        worktree path after S3 restore, which makes the create-time values wrong.

        The write merges into the existing file so hooks, model, and
        permissions survive. It is idempotent: same inputs, same result.
        No-op when the file is absent.
        """
        settings_local_path = worktree_path / ".claude" / "settings.local.json"
        if not settings_local_path.exists():
            return

        with open(settings_local_path) as f:
            settings_local = json.load(f)
        env = settings_local.setdefault("env", {})
        env["CUDA_VISIBLE_DEVICES"] = (
            ",".join(str(g) for g in gpu_ids) if gpu_ids else ""
        )
        env["AMMO_GPU_RES_DIR"] = f"/tmp/ammo_gpu_res_{session_id}"
        env["CLAUDE_PROJECT_DIR"] = str(worktree_path)
        # Expose parallelism sizes so agents can compute --num-gpus {tp*dp}
        # from env alone. Only inject when known (>0) — an empty string or
        # a silent "1" would mislead downstream shell guards that special-case
        # the unknown case. Drop a stale key when the size is now unknown so a
        # resume cannot leave a wrong value behind.
        for key, value in (("AMMO_TP_SIZE", tp_size), ("AMMO_DP_SIZE", dp_size)):
            if value is not None and value > 0:
                env[key] = str(value)
            else:
                env.pop(key, None)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=str(settings_local_path.parent))
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(settings_local, f, indent=2)
            os.replace(tmp_path, settings_local_path)
        except Exception:
            os.unlink(tmp_path)
            raise
        logger.debug(
            f"Injected CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}, "
            f"AMMO_GPU_RES_DIR={env['AMMO_GPU_RES_DIR']}, "
            f"CLAUDE_PROJECT_DIR={env['CLAUDE_PROJECT_DIR']} into settings.local.json"
        )

    def setup_claude_workspace(
        self,
        worktree_path: Path,
        session_id: str,
        gpu_ids: List[int],
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
    ) -> None:
        """
        Set up Claude Code workspace with required files.

        Args:
            worktree_path: Path to session worktree
            session_id: Session identifier
            gpu_ids: Allocated GPU IDs
            repo_name: Repository name (for context)
            branch: Branch name (for context)
            tp_size: Tensor-parallel size (injected into settings.local.json as
                AMMO_TP_SIZE when > 0). None / 0 means "unknown"; the key is
                omitted so downstream code can distinguish from "explicit 1".
            dp_size: Data-parallel size (injected as AMMO_DP_SIZE under the
                same rules as tp_size). Agents read these to compute
                --num-gpus {tp*dp} without reopening target.json.
        """
        claude_dir = worktree_path / ".claude"

        # Server template paths
        server_dir = Path(__file__).parent.parent
        docker_base = "/app"
        ai_cli_session_dir = "ai_cli_session"

        # 1. Bulk-copy entire .claude/ template directory
        template_src_docker = Path(f"{docker_base}/{ai_cli_session_dir}/.claude")
        template_src_local = server_dir / ai_cli_session_dir / ".claude"
        template_src = template_src_docker if template_src_docker.exists() else template_src_local

        if template_src.exists() and template_src.is_dir():
            if claude_dir.exists():
                shutil.rmtree(claude_dir)
            shutil.copytree(template_src, claude_dir)
            logger.debug(f"Copied Claude template to {claude_dir}")
        else:
            claude_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No Claude template found at {template_src_docker} or {template_src_local}")

        # 2. Make hook scripts executable
        hooks_dir = claude_dir / "hooks"
        if hooks_dir.exists():
            for hook_file in hooks_dir.iterdir():
                if hook_file.is_file() and hook_file.suffix == ".sh":
                    os.chmod(hook_file, 0o755)

        # 3. Generate sandboxed settings.json (overwrites the template copy)
        settings_dst = claude_dir / "settings.json"
        self._create_sandboxed_settings(worktree_path, session_id, settings_dst)

        # 3b. Inject the per-session dynamic env into settings.local.json.
        self.refresh_session_env(
            worktree_path,
            session_id,
            gpu_ids,
            tp_size=tp_size,
            dp_size=dp_size,
        )

        # 4. Write CLAUDE.md to worktree root with template variable substitution
        # (only if repo doesn't already have one)
        claude_md_src_docker = f"{docker_base}/{ai_cli_session_dir}/.claude/CLAUDE.md"
        claude_md_src_local = server_dir / ai_cli_session_dir / ".claude" / "CLAUDE.md"
        claude_md_dst = worktree_path / "CLAUDE.md"
        if not claude_md_dst.exists():
            claude_md_src = Path(claude_md_src_docker) if os.path.exists(claude_md_src_docker) else claude_md_src_local
            if claude_md_src.exists():
                with open(claude_md_src) as f:
                    content = f.read()
                gpu_ids_str = ",".join(str(g) for g in gpu_ids) if gpu_ids else "none"
                content = content.replace("{session_id}", session_id)
                content = content.replace("{repo_name}", repo_name or "unknown")
                content = content.replace("{branch}", branch or "main")
                content = content.replace("{gpu_ids}", gpu_ids_str)
                with open(claude_md_dst, 'w') as f:
                    f.write(content)
                logger.debug(f"Created CLAUDE.md with substitutions: {claude_md_dst}")
            else:
                logger.warning(f"CLAUDE.md template not found at {claude_md_src_docker} or {claude_md_src_local}")

        logger.info(f"Claude workspace setup complete: {claude_dir}")

    def setup_codex_workspace(
        self,
        worktree_path: Path,
        session_id: str,
        gpu_ids: List[int],
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
    ) -> None:
        """
        Set up Codex CLI workspace with required files.

        Args:
            worktree_path: Path to session worktree
            session_id: Session identifier
            gpu_ids: Allocated GPU IDs
            repo_name: Repository name (for context)
            branch: Branch name (for context)
            tp_size: Accepted for signature parity with setup_claude_workspace.
                Codex agents read AMMO_TP_SIZE from the process env, which
                session_manager injects into the ttyd env.
            dp_size: Accepted for parity; AMMO_DP_SIZE comes from the env too.
        """
        codex_dir = worktree_path / ".codex"

        # Server template paths
        server_dir = Path(__file__).parent.parent
        docker_base = "/app"
        ai_cli_session_dir = "ai_cli_session"

        # 1. Bulk-copy the full .codex template directory. Codex project-scoped
        # config, hooks, skills, and custom agent config layers all live here.
        template_src_docker = Path(f"{docker_base}/{ai_cli_session_dir}/.codex")
        template_src_local = server_dir / ai_cli_session_dir / ".codex"
        template_src = template_src_docker if template_src_docker.exists() else template_src_local

        if template_src.exists() and template_src.is_dir():
            if codex_dir.exists():
                shutil.rmtree(codex_dir)
            shutil.copytree(template_src, codex_dir)
            logger.debug(f"Copied Codex template to {codex_dir}")
        else:
            codex_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No Codex template found at {template_src_docker} or {template_src_local}")

        # 2. Substitute dynamic session values in Codex project config.
        gpu_ids_display = ",".join(str(g) for g in gpu_ids) if gpu_ids else "none"
        gpu_ids_env = ",".join(str(g) for g in gpu_ids) if gpu_ids else ""
        substitutions = {
            "{session_id}": session_id,
            "{repo_name}": repo_name or "unknown",
            "{branch}": branch or "main",
            "{gpu_ids}": gpu_ids_display,
            "{gpu_ids_env}": gpu_ids_env,
            "{worktree_path}": str(worktree_path),
        }
        for rel_path in ("config.toml", "AGENTS.md"):
            path = codex_dir / rel_path
            if path.exists():
                content = path.read_text()
                for needle, replacement in substitutions.items():
                    content = content.replace(needle, replacement)
                path.write_text(content)

        # 3. Write Codex project instructions to root AGENTS.md. Codex loads
        # AGENTS.md for project guidance; CODEX.md/settings.json were legacy.
        agents_md_src = codex_dir / "AGENTS.md"
        agents_md_dst = worktree_path / "AGENTS.md"
        if agents_md_src.exists() and not agents_md_dst.exists():
            content = agents_md_src.read_text()
            for needle, replacement in substitutions.items():
                content = content.replace(needle, replacement)
            agents_md_dst.write_text(content)
            logger.debug(f"Created AGENTS.md with substitutions: {agents_md_dst}")

        logger.info(f"Codex workspace setup complete: {codex_dir}")

    def setup_workspace(
        self,
        tool_type: CLIToolType,
        worktree_path: Path,
        session_id: str,
        gpu_ids: List[int],
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
    ) -> None:
        """
        Set up workspace for CLI tool.

        Args:
            tool_type: Type of CLI tool
            worktree_path: Path to session worktree
            session_id: Session identifier
            gpu_ids: Allocated GPU IDs
            repo_name: Repository name (for context)
            branch: Branch name (for context)
            tp_size: Tensor-parallel size (forwarded to Claude env injection
                when > 0; Codex reads it from the ttyd env).
            dp_size: Data-parallel size (forwarded likewise as AMMO_DP_SIZE).
        """
        if tool_type == CLIToolType.CLAUDE:
            self.setup_claude_workspace(
                worktree_path,
                session_id,
                gpu_ids,
                repo_name,
                branch,
                tp_size=tp_size,
                dp_size=dp_size,
            )
        elif tool_type == CLIToolType.CODEX:
            self.setup_codex_workspace(
                worktree_path,
                session_id,
                gpu_ids,
                repo_name,
                branch,
                tp_size=tp_size,
                dp_size=dp_size,
            )
        else:
            raise CLIToolError(f"Unsupported CLI tool: {tool_type}")


# Singleton instance
_cli_tool_manager: Optional[CLIToolManager] = None


def get_cli_tool_manager() -> CLIToolManager:
    """
    Get singleton CLI tool manager instance.

    Returns:
        CLIToolManager instance
    """
    global _cli_tool_manager
    if _cli_tool_manager is None:
        _cli_tool_manager = CLIToolManager()
    return _cli_tool_manager


def reset_cli_tool_manager() -> None:
    """Reset singleton instance (for testing)."""
    global _cli_tool_manager
    _cli_tool_manager = None
