# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Shared fixtures for AMMO session tests.

Provides reusable fixtures for SessionManager, GPUResourceManager,
and other session components with all external dependencies mocked.

Usage:
    from tests.fixtures.session_fixtures import (
        mock_session_manager,
        gpu_manager_for_sessions,
        mock_session_state,
    )
"""

import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionInfo,
)


# ============================================================================
# SINGLETON RESETS
# ============================================================================

@pytest.fixture(autouse=True)
def reset_all_singletons():
    """Reset all singleton instances before and after each test."""
    from shared.gpu_file_lock import reset_gpu_lock_manager
    from orchestration.session_manager import reset_session_manager
    from orchestration.worktree_manager import reset_worktree_manager
    from orchestration.cli_tool_manager import reset_cli_tool_manager

    reset_gpu_lock_manager()
    reset_session_manager()
    reset_worktree_manager()
    reset_cli_tool_manager()
    yield
    reset_gpu_lock_manager()
    reset_session_manager()
    reset_worktree_manager()
    reset_cli_tool_manager()


# ============================================================================
# GPU MANAGER FIXTURES
# ============================================================================

@pytest.fixture
def gpu_manager_4(tmp_path):
    """Create a GPUResourceManager with 4 mocked GPUs and a temp lock dir."""
    lock_dir = str(tmp_path / "gpu_locks")
    os.makedirs(lock_dir, exist_ok=True)

    with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=4):
        from shared.gpu_resource_manager import GPUResourceManager
        manager = GPUResourceManager(lock_dir=lock_dir)
        yield manager


@pytest.fixture
def gpu_manager_8(tmp_path):
    """Create a GPUResourceManager with 8 mocked GPUs and a temp lock dir."""
    lock_dir = str(tmp_path / "gpu_locks")
    os.makedirs(lock_dir, exist_ok=True)

    with patch('shared.gpu_file_lock.GPUFileLockManager._detect_gpu_count', return_value=8):
        from shared.gpu_resource_manager import GPUResourceManager
        manager = GPUResourceManager(lock_dir=lock_dir)
        yield manager


# ============================================================================
# MOCK MANAGER FACTORIES
# ============================================================================

@pytest.fixture
def mock_worktree_manager():
    """Create a mocked WorktreeManager."""
    manager = MagicMock()
    manager.get_repo_config.return_value = {
        "url": "https://github.com/vllm-project/vllm.git",
        "default_branch": "main",
    }
    manager.create_worktree = AsyncMock(return_value=Path("/tmp/test-worktree"))
    manager.remove_worktree = AsyncMock()
    manager.cleanup_session = AsyncMock()
    manager.get_worktree_path.return_value = Path("/tmp/test-worktree")
    manager.get_session_dir.return_value = Path("/tmp/test-session-dir")
    manager.create_session_dirs.return_value = {
        "session_dir": Path("/tmp/test-session-dir"),
        "worktree_dir": Path("/tmp/test-worktree"),
        "logs_dir": Path("/tmp/test-logs"),
    }
    manager.initialize_vllm_environment = AsyncMock(return_value={"status": "ok"})
    return manager


@pytest.fixture
def mock_terminal_manager():
    """Create a mocked TerminalManager."""
    manager = MagicMock()
    manager.is_available.return_value = True
    manager.start_terminal_with_command = AsyncMock(return_value=9001)
    manager.stop_terminal = AsyncMock(return_value=True)
    manager.is_terminal_running.return_value = True
    manager.get_terminal_port.return_value = 9001
    manager.get_terminal_url.return_value = "http://localhost:9001/"
    manager.cleanup_dead_terminal.return_value = False
    manager.get_tmux_session_name.return_value = "sess_test"
    manager.restart_ttyd_with_tmux_attach = AsyncMock(return_value=9001)
    manager.cleanup = AsyncMock()
    return manager


@pytest.fixture
def mock_cli_tool_manager():
    """Create a mocked CLIToolManager."""
    manager = MagicMock()
    manager.setup_workspace = MagicMock()
    manager.get_cli_command.return_value = ["/usr/bin/node", "cli.js"]
    return manager


@pytest.fixture
def mock_inactivity_monitor():
    """Create a mocked InactivityMonitor."""
    monitor = MagicMock()
    monitor.register_session = MagicMock()
    monitor.unregister_session = MagicMock()
    monitor.record_activity = MagicMock()
    monitor.start = MagicMock()
    monitor.stop = AsyncMock()
    return monitor


@pytest.fixture
def mock_session_storage():
    """Create a mocked SessionS3Storage."""
    storage = MagicMock()
    storage.enabled = False
    storage.save_session_metadata = AsyncMock(return_value=True)
    storage.load_session_metadata = AsyncMock(return_value=None)
    storage.sync_session_to_s3 = AsyncMock(return_value=True)
    storage.sync_worktree_to_s3 = AsyncMock(return_value=True)
    storage.restore_session_from_s3 = AsyncMock(return_value=None)
    storage.restore_worktree_from_s3 = AsyncMock(return_value=True)
    storage.restore_cli_state_from_s3 = AsyncMock(return_value=True)
    storage.session_exists_in_s3 = AsyncMock(return_value=False)
    storage.delete_session_from_s3 = AsyncMock(return_value=True)
    storage.list_s3_sessions = AsyncMock(return_value=[])
    storage.create_download_archive = AsyncMock(return_value="archives/test.zip")
    storage.get_download_url = AsyncMock(return_value="https://s3.example.com/presigned")
    storage.get_download_size = AsyncMock(return_value=1024)
    storage.ensure_session_synced = AsyncMock(return_value=True)
    storage.get_s3_last_modified = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def mock_session_manager(
    tmp_path,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
):
    """
    Create a SessionManager with all dependencies mocked except GPU manager.

    GPU manager uses real file locks in tmp_path for race condition testing.
    All other managers are mocked.
    """
    from orchestration.session_manager import SessionManager

    sessions_dir = str(tmp_path / "sessions")
    os.makedirs(sessions_dir, exist_ok=True)

    manager = SessionManager(
        sessions_dir=sessions_dir,
        worktree_manager=mock_worktree_manager,
        gpu_manager=gpu_manager_4,
        terminal_manager=mock_terminal_manager,
        cli_tool_manager=mock_cli_tool_manager,
        inactivity_monitor=mock_inactivity_monitor,
        session_storage=mock_session_storage,
    )
    return manager


# ============================================================================
# SESSION STATE FACTORIES
# ============================================================================

def make_session_state(
    session_id: str = "test-session-001",
    status: SessionStatus = SessionStatus.ACTIVE,
    cli_tool: CLIToolType = CLIToolType.CLAUDE,
    repo_name: str = "vllm",
    branch: str = "main",
    gpu_ids: list = None,
    requested_gpu_count: int = 1,
    owner_id: str = "test-client-001",
    model_name: str = "DeepSeek-R1",
    dtype: str = "fp8",
    worktree_path: str = "/tmp/test-worktree",
    inactivity_timeout_mins: int = 720,
    tp_size: int = 0,
    dp_size: int = 1,
) -> SessionState:
    """Factory for creating SessionState objects with sensible defaults.

    tp_size defaults to 0 (the legacy "unknown" sentinel per session_models.py).
    Pass tp_size/dp_size explicitly to exercise TP/DP-aware code paths.
    """
    return SessionState(
        session_id=session_id,
        status=status,
        cli_tool=cli_tool,
        repo_name=repo_name,
        branch=branch,
        worktree_path=worktree_path,
        gpu_ids=gpu_ids or [0],
        requested_gpu_count=requested_gpu_count,
        created_at=time.time(),
        last_accessed=time.time(),
        inactivity_timeout_mins=inactivity_timeout_mins,
        owner_id=owner_id,
        model_name=model_name,
        dtype=dtype,
        tp_size=tp_size,
        dp_size=dp_size,
    )


@pytest.fixture
def mock_session_manager_auth_enabled(mock_session_manager, monkeypatch):
    """SessionManager with AMMO_API_KEY set."""
    monkeypatch.setenv("AMMO_API_KEY", "test-api-key-32bytes-minimum-length-00")
    return mock_session_manager


@pytest.fixture
def mock_session_manager_auth_disabled(mock_session_manager, monkeypatch):
    """SessionManager with AMMO_API_KEY unset."""
    monkeypatch.delenv("AMMO_API_KEY", raising=False)
    return mock_session_manager


def make_create_request(
    repo_name: str = "vllm",
    cli_tool: str = "claude",
    branch: str = "main",
    gpu_count: int = 1,
    model_name: str = "DeepSeek-R1",
    dtype: str = "fp8",
    inactivity_timeout_mins: int = 720,
    initial_prompt: str = "",
    tp_size: int = None,
    dp_size: int = 1,
) -> CreateSessionRequest:
    """Factory for creating CreateSessionRequest objects."""
    kwargs = dict(
        repo_name=repo_name,
        cli_tool=cli_tool,
        branch=branch,
        gpu_count=gpu_count,
        model_name=model_name,
        dtype=dtype,
        inactivity_timeout_mins=inactivity_timeout_mins,
        initial_prompt=initial_prompt,
        dp_size=dp_size,
    )
    if tp_size is not None:
        kwargs["tp_size"] = tp_size
    return CreateSessionRequest(**kwargs)
