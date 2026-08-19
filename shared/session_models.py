# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Models for AI CLI Session Service.

This module defines request/response models for the AI CLI session service
that allows users to create isolated git worktree sessions with AI CLI tools.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import time


# ============================================================================
# Enums
# ============================================================================

class SessionStatus(str, Enum):
    """Session lifecycle states."""
    CREATING = "creating"
    BUILDING = "building"  # async source build in progress (custom forks)
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    FAILED = "failed"


class CLIToolType(str, Enum):
    """Supported AI CLI tools."""
    CLAUDE = "claude"
    CODEX = "codex"


# ============================================================================
# Request Models
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new AI CLI session."""

    repo_name: str = Field(
        "vllm",
        description="Pre-configured repository identifier (e.g., 'vllm')"
    )
    cli_tool: CLIToolType = Field(
        CLIToolType.CLAUDE,
        description="AI CLI tool to use: 'claude' or 'codex'"
    )
    branch: str = Field(
        "main",
        description="Git branch to base worktree on"
    )
    initial_prompt: Optional[str] = Field(
        None,
        description="Optional initial prompt to send to AI CLI on startup"
    )
    gpu_count: int = Field(
        0,
        ge=0,
        description="Number of GPUs to allocate to session (default: 0)"
    )
    tp_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=8,
        description="Tensor parallel size. When omitted, falls back to gpu_count."
    )
    dp_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Data parallel size (MoE models only)."
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional: resume existing session by ID"
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional MoE model name (e.g., 'DeepSeek-R1')"
    )
    dtype: Optional[str] = Field(
        None,
        description="Optional dtype for the model (e.g., 'fp8', 'bf16')"
    )
    vllm_fork_url: Optional[str] = Field(
        None,
        description="Optional custom vLLM fork URL (https://github.com/<owner>/<repo>). "
                    "When set, the session builds vLLM from source for this fork.",
    )
    vllm_fork_token: Optional[str] = Field(
        None,
        description="Optional access token for a private fork (used for clone/fetch, "
                    "stored encrypted). Requires vllm_fork_url and AMMO_FORK_TOKEN_KEY.",
    )
    inactivity_timeout_mins: int = Field(
        1440,
        ge=5,
        le=2880,
        description="Minutes of inactivity before auto-pause (5 min to 48 hours)"
    )

    class Config:
        extra = "forbid"
        use_enum_values = True

    @field_validator("branch")
    @classmethod
    def _validate_branch(cls, v: str) -> str:
        """Reject branch values that could be option-injected into git ref cmds.

        A git ref name must not start with '-' (option injection) and must not
        contain whitespace or shell/ref metacharacters. We allow the practical
        ref charset: alnum, dot, slash, underscore, hyphen (hyphen only non-leading).
        """
        if v is None or v == "":
            return "main"
        if v.startswith("-"):
            raise ValueError("branch must not start with '-'")
        if not re.match(r"^[A-Za-z0-9._/-]+$", v):
            raise ValueError(
                "branch may only contain letters, digits, '.', '_', '/', '-'"
            )
        # Reject git-special sequences that survive the charset but are unsafe refs.
        if ".." in v or v.endswith("/") or v.endswith(".lock") or "//" in v:
            raise ValueError("branch is not a valid git ref name")
        return v

    @model_validator(mode="after")
    def _validate_parallelism(self):
        """Cross-field parallelism checks.

        - dp_size > 1 requires explicit tp_size (otherwise gpu_count semantics
          are ambiguous vs tp*dp).
        - When tp_size is provided, tp_size * dp_size must be <= gpu_count.
          The session's GPU pool may exceed the per-model-replica footprint
          (TP*DP) so spare GPUs can power parallel experiment tracks inside
          the session. See .claude/plans/gpu-decouple.md.
        - Legacy path (tp_size=None, dp_size=1) is unconstrained for
          backward compatibility with clients that only send gpu_count.
        """
        if self.dp_size > 1 and self.tp_size is None:
            raise ValueError(
                "tp_size is required when dp_size > 1 "
                "(otherwise gpu_count vs tp_size*dp_size is ambiguous)"
            )
        if self.tp_size is not None:
            if self.tp_size * self.dp_size > self.gpu_count:
                raise ValueError(
                    f"tp_size ({self.tp_size}) * dp_size ({self.dp_size}) "
                    f"must be <= gpu_count ({self.gpu_count})"
                )
        if self.vllm_fork_token and not self.vllm_fork_url:
            raise ValueError(
                "vllm_fork_token requires vllm_fork_url to be set"
            )
        return self


# ============================================================================
# Response Models
# ============================================================================

class SessionInfo(BaseModel):
    """Information about a session."""

    session_id: str = Field(..., description="Unique session identifier")
    cli_session_id: Optional[str] = Field(
        None,
        description="Native CLI tool session ID (for resume)"
    )
    status: str = Field(..., description="Current session status")
    cli_tool: str = Field(..., description="AI CLI tool type")
    repo_name: str = Field(..., description="Repository name")
    branch: str = Field(..., description="Git branch")
    gpu_ids: List[int] = Field(
        default_factory=list,
        description="Currently allocated GPU IDs (empty when paused)"
    )
    requested_gpu_count: int = Field(
        0,
        description="Originally requested GPU count (preserved across pause/resume)"
    )
    tp_size: Optional[int] = Field(
        None,
        description="Tensor parallel size (None for legacy sessions without this metadata)"
    )
    dp_size: Optional[int] = Field(
        None,
        description="Data parallel size (None for legacy sessions without this metadata)"
    )
    terminal_url: Optional[str] = Field(
        None,
        description="Web terminal URL (relative path)"
    )
    terminal_ws_url: Optional[str] = Field(
        None,
        description="WebSocket URL for terminal connection"
    )
    created_at: float = Field(..., description="Unix timestamp when session was created")
    last_accessed: float = Field(..., description="Unix timestamp of last activity")
    inactivity_timeout_mins: int = Field(
        1440,
        description="Minutes of inactivity before auto-pause"
    )
    worktree_path: Optional[str] = Field(
        None,
        description="Path to session worktree (internal use)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if session failed"
    )
    owner_id: Optional[str] = Field(
        None,
        description="Client identifier that owns this session (for multi-user isolation)"
    )
    model_name: Optional[str] = Field(
        None,
        description="MoE model name associated with this session"
    )
    dtype: Optional[str] = Field(
        None,
        description="dtype used for the model in this session"
    )
    has_report: bool = Field(
        False,
        description="Whether a REPORT.md exists in the session worktree"
    )
    ammo_version: Optional[str] = Field(
        None,
        description="AMMO config version at session creation time"
    )
    build_timings: Optional[Dict[str, Any]] = Field(
        None,
        description="vLLM environment initialization timings (venv_create, editable_install, total)"
    )
    vllm_fork_url: Optional[str] = Field(
        None,
        description="Custom vLLM fork URL if this session uses one"
    )
    build_phase: Optional[str] = Field(
        None,
        description="Current build phase when status is BUILDING"
    )
    build_error: Optional[str] = Field(
        None,
        description="Build error (last lines) when status is FAILED"
    )


class CreateSessionResponse(BaseModel):
    """Response after creating or resuming a session."""

    session_id: str = Field(..., description="Unique session identifier")
    cli_session_id: Optional[str] = Field(
        None,
        description="Native CLI tool session ID"
    )
    status: str = Field(..., description="Current session status")
    cli_tool: str = Field(..., description="AI CLI tool type")
    repo_name: str = Field(..., description="Repository name")
    gpu_ids: List[int] = Field(
        default_factory=list,
        description="Allocated GPU IDs"
    )
    terminal_url: Optional[str] = Field(
        None,
        description="Web terminal URL (relative)"
    )
    terminal_ws_url: Optional[str] = Field(
        None,
        description="WebSocket URL for terminal"
    )
    created_at: float = Field(..., description="Creation timestamp")
    last_accessed: float = Field(..., description="Last activity timestamp")
    inactivity_timeout_mins: int = Field(..., description="Inactivity timeout")
    message: str = Field(..., description="Human-readable status message")
    model_name: Optional[str] = Field(
        None,
        description="MoE model name associated with this session"
    )
    dtype: Optional[str] = Field(
        None,
        description="dtype used for the model in this session"
    )


class SessionListResponse(BaseModel):
    """Response for listing all sessions."""

    sessions: List[SessionInfo] = Field(
        default_factory=list,
        description="List of sessions"
    )
    total: int = Field(0, description="Total number of sessions")


class SessionActionResponse(BaseModel):
    """Response for session actions (pause, terminate)."""

    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="New session status")
    message: str = Field(..., description="Action result message")


class SessionDownloadInfo(BaseModel):
    """Response for session download preparation."""

    session_id: str = Field(..., description="Session identifier")
    download_url: Optional[str] = Field(
        None,
        description="Presigned S3 URL for downloading the session archive"
    )
    download_size_bytes: Optional[int] = Field(
        None,
        description="Size of the download archive in bytes"
    )
    archive_ready: bool = Field(
        False,
        description="Whether the archive is ready for download"
    )
    expires_at: Optional[float] = Field(
        None,
        description="Unix timestamp when the download URL expires"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if archive creation failed"
    )


# ============================================================================
# Internal State Models
# ============================================================================

@dataclass
class SessionState:
    """Internal state for tracking a session."""

    session_id: str
    status: SessionStatus
    cli_tool: CLIToolType
    repo_name: str
    branch: str
    created_at: float
    last_accessed: float
    inactivity_timeout_mins: int = 1440

    # Session directories
    session_dir: Optional[str] = None
    worktree_path: Optional[str] = None
    logs_dir: Optional[str] = None

    # CLI state
    cli_session_id: Optional[str] = None
    cli_process_pid: Optional[int] = None

    # Terminal state
    terminal_port: Optional[int] = None
    ttyd_process_pid: Optional[int] = None

    # GPU allocation
    gpu_ids: List[int] = field(default_factory=list)
    requested_gpu_count: int = 0  # Track original request for resume

    # Parallelism layout (spec: 2026-04-27 MoE DP/EP controls).
    # tp_size=0 is the "unknown" sentinel for in-memory construction; from_dict
    # resolves to requested_gpu_count for legacy S3 state without this key.
    tp_size: int = 0
    dp_size: int = 1

    # Error tracking
    error: Optional[str] = None

    # S3 state
    s3_synced: bool = False
    s3_last_sync: Optional[float] = None

    # vLLM Build state
    build_initialized: bool = False
    build_timings: Optional[Dict[str, float]] = None

    # Owner identification for multi-user isolation
    # Sessions with owner_id=None are visible to all clients (legacy behavior)
    owner_id: Optional[str] = None

    # Model metadata
    model_name: Optional[str] = None
    dtype: Optional[str] = None

    # Report availability — sticky latch: once True, never reset to False.
    # Persisted via to_dict()/from_dict() so S3 restore retains the report button.
    # On crash-recovery (no S3 state), reverts to False but re-discovered from filesystem.
    has_report: bool = False

    # AMMO config version at session creation time (read from VERSION file).
    ammo_version: Optional[str] = None

    # Initial prompt preserved for auto-recovery (lost when terminal dies
    # before Claude Code processes it; auto-recovery re-sends it).
    initial_prompt: Optional[str] = None

    # Custom vLLM fork support (spec: 2026-06-01 custom-vllm-fork-support).
    vllm_fork_url: Optional[str] = None
    vllm_fork_token_encrypted: Optional[str] = None
    build_phase: Optional[str] = None      # fetching | compiling | installing
    build_error: Optional[str] = None      # last-N build log lines on failure

    def to_session_info(self) -> SessionInfo:
        """Convert internal state to API response."""
        terminal_url = None
        terminal_ws_url = None
        if self.terminal_port and self.status == SessionStatus.ACTIVE:
            terminal_url = f"/sessions/{self.session_id}/terminal/"
            terminal_ws_url = f"/sessions/{self.session_id}/terminal/ws"

        # Check for REPORT.md — sticky latch: once True, skip filesystem re-check.
        # On first discovery, set self.has_report = True so subsequent calls are O(1).
        # Mutation here requires a subsequent save_session_metadata() to persist (happens on pause).
        if not self.has_report and (
            self.worktree_path
            and self.status in (SessionStatus.ACTIVE, SessionStatus.PAUSED)
        ):
            wt = Path(self.worktree_path)
            if wt.joinpath("REPORT.md").is_file() or any(
                wt.joinpath("kernel_opt_artifacts").glob("*/REPORT.md")
            ):
                self.has_report = True
        has_report = self.has_report

        return SessionInfo(
            session_id=self.session_id,
            cli_session_id=self.cli_session_id,
            status=self.status.value,
            cli_tool=self.cli_tool.value,
            repo_name=self.repo_name,
            branch=self.branch,
            gpu_ids=self.gpu_ids,
            requested_gpu_count=self.requested_gpu_count,
            tp_size=(self.tp_size if self.tp_size > 0 else None),
            dp_size=self.dp_size,
            terminal_url=terminal_url,
            terminal_ws_url=terminal_ws_url,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            inactivity_timeout_mins=self.inactivity_timeout_mins,
            worktree_path=self.worktree_path,
            error=self.error,
            owner_id=self.owner_id,
            model_name=self.model_name,
            dtype=self.dtype,
            has_report=has_report,
            ammo_version=self.ammo_version,
            build_timings=self.build_timings,
            vllm_fork_url=self.vllm_fork_url,
            build_phase=self.build_phase,
            build_error=self.build_error,
        )

    def to_create_response(self, message: str) -> CreateSessionResponse:
        """Convert to create/resume response."""
        terminal_url = None
        terminal_ws_url = None
        if self.terminal_port and self.status == SessionStatus.ACTIVE:
            terminal_url = f"/sessions/{self.session_id}/terminal/"
            terminal_ws_url = f"/sessions/{self.session_id}/terminal/ws"

        return CreateSessionResponse(
            session_id=self.session_id,
            cli_session_id=self.cli_session_id,
            status=self.status.value,
            cli_tool=self.cli_tool.value,
            repo_name=self.repo_name,
            gpu_ids=self.gpu_ids,
            terminal_url=terminal_url,
            terminal_ws_url=terminal_ws_url,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            inactivity_timeout_mins=self.inactivity_timeout_mins,
            message=message,
            model_name=self.model_name,
            dtype=self.dtype,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "cli_tool": self.cli_tool.value,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "inactivity_timeout_mins": self.inactivity_timeout_mins,
            "session_dir": self.session_dir,
            "worktree_path": self.worktree_path,
            "logs_dir": self.logs_dir,
            "cli_session_id": self.cli_session_id,
            "cli_process_pid": self.cli_process_pid,
            "terminal_port": self.terminal_port,
            "ttyd_process_pid": self.ttyd_process_pid,
            "gpu_ids": self.gpu_ids,
            "requested_gpu_count": self.requested_gpu_count,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "error": self.error,
            "s3_synced": self.s3_synced,
            "s3_last_sync": self.s3_last_sync,
            "build_initialized": self.build_initialized,
            "build_timings": self.build_timings,
            "owner_id": self.owner_id,
            "model_name": self.model_name,
            "dtype": self.dtype,
            "has_report": self.has_report,
            "ammo_version": self.ammo_version,
            "initial_prompt": self.initial_prompt,
            "vllm_fork_url": self.vllm_fork_url,
            "vllm_fork_token_encrypted": self.vllm_fork_token_encrypted,
            "build_phase": self.build_phase,
            "build_error": self.build_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Deserialize from dictionary."""
        return cls(
            session_id=data["session_id"],
            status=SessionStatus(data["status"]),
            cli_tool=CLIToolType(data["cli_tool"]),
            repo_name=data["repo_name"],
            branch=data["branch"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            inactivity_timeout_mins=data.get("inactivity_timeout_mins", DEFAULT_INACTIVITY_TIMEOUT_MINS),
            session_dir=data.get("session_dir"),
            worktree_path=data.get("worktree_path"),
            logs_dir=data.get("logs_dir"),
            cli_session_id=data.get("cli_session_id"),
            cli_process_pid=data.get("cli_process_pid"),
            terminal_port=data.get("terminal_port"),
            ttyd_process_pid=data.get("ttyd_process_pid"),
            gpu_ids=data.get("gpu_ids", []),
            requested_gpu_count=data.get("requested_gpu_count", 0),
            # Legacy S3 state won't have tp_size/dp_size: spec §2.2 says old
            # sessions default to tp_size = gpu_count, dp_size = 1. We fall
            # back to requested_gpu_count (persisted) since it survives
            # pause/resume. Also handles the 0 sentinel (dataclass default).
            tp_size=data.get("tp_size") or data.get("requested_gpu_count", 0),
            dp_size=data.get("dp_size", 1),
            error=data.get("error"),
            s3_synced=data.get("s3_synced", False),
            s3_last_sync=data.get("s3_last_sync"),
            build_initialized=data.get("build_initialized", False),
            build_timings=data.get("build_timings"),
            owner_id=data.get("owner_id"),
            model_name=data.get("model_name"),
            dtype=data.get("dtype"),
            has_report=data.get("has_report", False),
            ammo_version=data.get("ammo_version"),
            initial_prompt=data.get("initial_prompt"),
            vllm_fork_url=data.get("vllm_fork_url"),
            vllm_fork_token_encrypted=data.get("vllm_fork_token_encrypted"),
            build_phase=data.get("build_phase"),
            build_error=data.get("build_error"),
        )


# ============================================================================
# Constants
# ============================================================================

# Environment variable names
ENV_SESSION_S3_BUCKET = "SESSION_S3_BUCKET"
ENV_SESSION_S3_PREFIX = "SESSION_S3_PREFIX"
ENV_SESSION_S3_TTL_DAYS = "SESSION_S3_TTL_DAYS"
ENV_SESSION_INACTIVITY_TIMEOUT_MINS = "SESSION_INACTIVITY_TIMEOUT_MINS"
ENV_SESSION_DATA_DIR = "SESSION_DATA_DIR"
ENV_SESSION_REPOS_DIR = "SESSION_REPOS_DIR"
ENV_SESSION_TEMPLATES_DIR = "SESSION_TEMPLATES_DIR"

# Default values
DEFAULT_SESSION_DATA_DIR = "/data/sessions"
DEFAULT_SESSION_REPOS_DIR = "/data/repos"
DEFAULT_SESSION_TEMPLATES_DIR = "/data/templates"
DEFAULT_SESSION_S3_PREFIX = "sessions"
DEFAULT_SESSION_S3_TTL_DAYS = 30
DEFAULT_INACTIVITY_TIMEOUT_MINS = 1440

# Terminal configuration
DEFAULT_TERMINAL_BASE_PORT = 8001
MAX_TERMINAL_PORTS = 100

# Pre-configured repositories
SUPPORTED_REPOS = {
    "vllm": {
        "url": "https://github.com/vllm-project/vllm.git",
        "default_branch": "main",
    },
}

# Subdirectory under repos_dir for per-fork base clones (custom fork support).
FORK_REPOS_SUBDIR = "forks"

# Status descriptions for human-readable output
SESSION_STATUS_DESCRIPTIONS = {
    SessionStatus.CREATING: "Session is being created",
    SessionStatus.BUILDING: "Building vLLM from source (custom fork)",
    SessionStatus.ACTIVE: "Session is active and ready",
    SessionStatus.PAUSED: "Session is paused (can be resumed)",
    SessionStatus.TERMINATED: "Session has been terminated",
    SessionStatus.FAILED: "Session creation failed",
}
