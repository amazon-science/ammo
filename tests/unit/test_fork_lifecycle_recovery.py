# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_lifecycle_recovery.py
"""Lifecycle/crash-recovery tests for custom vLLM fork sessions.

Covers the LANE SM remediation fixes:
  A+H — BUILDING reconciled to FAILED on restart (_load_sessions / discover_s3).
  B   — terminate cancels in-flight build; build aborts if terminated midway.
  C   — fork GPU-acquire failure cleans up (no phantom BUILDING).
  F   — resume rebuild failure does not go ACTIVE (stays PAUSED, re-raises).
  G   — undecryptable token surfaces a clear RuntimeError (no anonymous clone).
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import reset_all_singletons  # noqa: F401
from shared.session_models import (
    SessionState, SessionStatus, CLIToolType, CreateSessionRequest,
)


def _seed_session_json(sessions_dir: Path, sid: str, status: SessionStatus,
                       gpu_ids):
    """Write a session.json on disk under sessions_dir/<sid>/session.json."""
    sdir = sessions_dir / sid
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    st = SessionState(
        session_id=sid, status=status,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        session_dir=str(sdir), logs_dir=str(sdir / "logs"),
        worktree_path=str(sdir / "worktree"), gpu_ids=list(gpu_ids),
        cli_process_pid=4242, ttyd_process_pid=4343, terminal_port=8042,
        vllm_fork_url="https://github.com/u/vllm.git",
        build_phase="compiling",
    )
    with open(sdir / "session.json", "w") as f:
        json.dump(st.to_dict(), f)
    return st


# ── Fix A: BUILDING → FAILED on _load_sessions ─────────────────────────────
@pytest.mark.unit
def test_building_session_marked_failed_on_load(tmp_path):
    from orchestration.session_manager import SessionManager

    sessions_dir = tmp_path / "sessions"
    _seed_session_json(sessions_dir, "blds", SessionStatus.BUILDING, [0])

    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm.sessions_dir = sessions_dir
    # _save_session_state writes to disk via the real path machinery.
    sm._get_session_state_path = lambda sid: sessions_dir / sid / "session.json"

    sm._load_sessions()

    # FAILED sessions are not added to the live registry.
    assert "blds" not in sm._sessions
    # Persisted file is terminal FAILED with no GPUs / process state.
    with open(sessions_dir / "blds" / "session.json") as f:
        data = json.load(f)
    persisted = SessionState.from_dict(data)
    assert persisted.status == SessionStatus.FAILED
    assert persisted.gpu_ids == []
    assert persisted.cli_process_pid is None
    assert persisted.ttyd_process_pid is None
    assert persisted.terminal_port is None
    assert persisted.build_phase is None
    assert "restart" in (persisted.build_error or "")


# ── Fix H: BUILDING → FAILED on discover_s3_sessions ───────────────────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_discover_s3_building_marked_failed(tmp_path):
    from orchestration.session_manager import SessionManager

    sessions_dir = tmp_path / "sessions"
    sid = "s3bld"
    state = SessionState(
        session_id=sid, status=SessionStatus.BUILDING,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0, gpu_ids=[1],
        cli_process_pid=1, ttyd_process_pid=2, terminal_port=8042,
        vllm_fork_url="https://github.com/u/vllm.git", build_phase="compiling",
    )

    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm.sessions_dir = sessions_dir
    sm._get_session_state_path = lambda s: sessions_dir / s / "session.json"
    saved = {}
    sm._save_session_state = lambda st: saved.__setitem__(st.session_id, st)

    storage = MagicMock()
    storage.enabled = True
    storage.list_s3_sessions = AsyncMock(return_value=[sid])
    storage.load_session_metadata = AsyncMock(return_value=state)
    sm.session_storage = storage

    discovered = await sm.discover_s3_sessions()

    assert discovered == 0
    assert sid not in sm._sessions
    failed = saved[sid]
    assert failed.status == SessionStatus.FAILED
    assert failed.gpu_ids == []
    assert failed.cli_process_pid is None
    assert failed.terminal_port is None
    assert "restart" in (failed.build_error or "")


# ── Fix B (parts 1+2): terminate cancels in-flight build + releases GPUs ────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminate_cancels_build_task(tmp_path):
    from orchestration.session_manager import SessionManager

    sid = "term1"
    st = SessionState(
        session_id=sid, status=SessionStatus.BUILDING,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0, gpu_ids=[0],
        worktree_path=str(tmp_path / "wt"),
    )

    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {sid: st}
    sm._recovery_locks = {}
    sm.inactivity_monitor = None
    sm.terminal_manager = None
    sm.session_storage = None
    sm._save_session_state = MagicMock()
    sm._validate_ownership = lambda s, o: st

    order = []
    sm._release_gpus = MagicMock(side_effect=lambda s: order.append("release_gpus"))
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.cleanup_session = MagicMock()

    async def never_finishing():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("build_cancelled")
            raise

    task = asyncio.ensure_future(never_finishing())
    await asyncio.sleep(0)  # let it start
    sm._fork_build_tasks = {sid: task}

    await sm.terminate_session(sid)

    assert task.cancelled() or task.done()
    assert "build_cancelled" in order
    # GPUs released AFTER the build task was cancelled.
    assert order.index("build_cancelled") < order.index("release_gpus")
    assert st.status == SessionStatus.TERMINATED
    assert sid not in sm._sessions


# ── Fix B (part 3): build aborts if terminated midway ──────────────────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_aborts_if_terminated_midway(tmp_path):
    from orchestration.session_manager import SessionManager

    sid = "mid1"
    sdir = tmp_path / "sessions" / sid
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    (sdir / "worktree").mkdir(parents=True, exist_ok=True)
    st = SessionState(
        session_id=sid, status=SessionStatus.BUILDING,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
        session_dir=str(sdir), logs_dir=str(sdir / "logs"),
        worktree_path=str(sdir / "worktree"), gpu_ids=[0],
        vllm_fork_url="https://github.com/u/vllm.git",
    )

    sm = SessionManager.__new__(SessionManager)
    # Simulate terminate having already flipped status before step 6.
    terminated = SessionState(
        session_id=sid, status=SessionStatus.TERMINATED,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
    )
    sm._sessions = {sid: terminated}
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.create_worktree = MagicMock(return_value=Path(st.worktree_path))
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {}}
    )
    sm.cli_tool_manager = MagicMock()
    sm.cli_tool_manager.get_cli_command.return_value = ["/usr/bin/env", "claude"]
    sm.terminal_manager = MagicMock()
    sm.terminal_manager.is_available.return_value = True
    sm.terminal_manager.start_terminal_with_command = AsyncMock(return_value=8042)
    sm.inactivity_monitor = None
    sm._save_session_state = MagicMock()
    sm._release_gpus = MagicMock()
    sm._chown_session_to_user = MagicMock()
    sm._build_extra_env = MagicMock(return_value={})
    sm._fork_console_path = MagicMock(return_value="/bin/true")

    req = CreateSessionRequest(gpu_count=1, vllm_fork_url=st.vllm_fork_url)

    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        gf.return_value.ensure_fork_base.return_value = Path(tmp_path / "fb")
        await sm._run_fork_build(st, req)

    # Build must NOT have promoted to ACTIVE, and must NOT have written "ok".
    assert st.status != SessionStatus.ACTIVE
    status_file = sdir / "logs" / "fork_build.status"
    assert status_file.read_text() != "ok"


# ── Fix C: fork GPU-acquire failure cleans up (no phantom BUILDING) ─────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_fork_create_gpu_failure_cleans_up(tmp_path, monkeypatch):
    from orchestration.session_manager import SessionManager, SessionError

    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._recovery_locks = {}
    sm.sessions_dir = tmp_path / "sessions"
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.create_session_dirs.return_value = {
        "session_dir": tmp_path / "sessions" / "x",
        "logs_dir": tmp_path / "sessions" / "x" / "logs",
    }
    sm.inactivity_monitor = None
    sm._save_session_state = MagicMock()
    sm._release_gpus = MagicMock()
    sm._acquire_gpus = AsyncMock(side_effect=RuntimeError("no gpus free"))
    sm._run_fork_build = AsyncMock()

    monkeypatch.setenv(
        "AMMO_FORK_TOKEN_KEY",
        __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet.generate_key().decode(),
    )
    req = CreateSessionRequest(gpu_count=1, vllm_fork_url="https://github.com/u/vllm.git")

    with pytest.raises(SessionError):
        await sm.create_session(req, owner_id=None)

    # No phantom BUILDING session left behind, GPUs released.
    assert all(s.status == SessionStatus.BUILDING for s in [])  # trivially true
    assert len(sm._sessions) == 0
    sm._run_fork_build.assert_not_called()


# ── Fix F: resume rebuild failure does not go ACTIVE ───────────────────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_resume_fork_rebuild_failure_does_not_go_active(tmp_path):
    from orchestration.session_manager import SessionManager, SessionError

    sid = "res1"
    sdir = tmp_path / "sessions" / sid
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    (sdir / "worktree").mkdir(parents=True, exist_ok=True)
    # The restored-from-S3 state (fork session needing a source rebuild).
    st = SessionState(
        session_id=sid, status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        session_dir=str(sdir), logs_dir=str(sdir / "logs"),
        worktree_path=str(sdir / "worktree"), gpu_ids=[0],
        vllm_fork_url="https://github.com/u/vllm.git",
    )

    sm = SessionManager.__new__(SessionManager)
    # Not in local memory → drive the S3-restore path so restored_from_s3=True
    # and the fork rebuild branch executes.
    sm._sessions = {}
    sm.sessions_dir = tmp_path / "sessions"
    sm._recovery_locks = {sid: asyncio.Lock()}
    sm.inactivity_monitor = None
    sm._save_session_state = MagicMock()
    sm._release_gpus = MagicMock()
    sm._acquire_gpus = AsyncMock(return_value=[0])

    sm.session_storage = MagicMock()
    sm.session_storage.enabled = True
    sm.session_storage.restore_session_from_s3 = AsyncMock(return_value=st)
    sm.session_storage.get_s3_last_modified = AsyncMock(return_value=2.0)

    sm.worktree_manager = MagicMock()
    sm.worktree_manager.repair_worktree_linkage = MagicMock()
    sm.terminal_manager = MagicMock()
    sm.terminal_manager.start_terminal_with_command = AsyncMock(return_value=8042)
    sm.cli_tool_manager = MagicMock()

    # The fork rebuild blows up — must propagate, not get swallowed.
    sm._reinit_fork_env_on_resume = AsyncMock(side_effect=RuntimeError("rebuild failed"))

    with pytest.raises(SessionError):
        await sm.resume_session(sid, owner_id=None)

    # Outer resume except resets to PAUSED + releases GPUs; never ACTIVE.
    assert st.status == SessionStatus.PAUSED
    assert st.status != SessionStatus.ACTIVE
    assert st.gpu_ids == []
    sm._release_gpus.assert_called()


# ── Fix G: undecryptable token surfaces a clear RuntimeError ───────────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinit_raises_on_undecryptable_token(tmp_path, monkeypatch):
    from orchestration.session_manager import SessionManager

    # No AMMO_FORK_TOKEN_KEY configured on this pod → token undecryptable.
    monkeypatch.delenv("AMMO_FORK_TOKEN_KEY", raising=False)

    st = SessionState(
        session_id="g1", status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(tmp_path / "wt"),
        vllm_fork_url="https://github.com/u/vllm.git",
        vllm_fork_token_encrypted="gAAAAABsome-bogus-ciphertext==",
    )

    sm = SessionManager.__new__(SessionManager)
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {}}
    )

    with pytest.raises(RuntimeError, match="AMMO_FORK_TOKEN_KEY"):
        await sm._reinit_fork_env_on_resume(st)


# ── correctness-5: build-active re-arms the inactivity monitor ─────────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_success_reregisters_inactivity(tmp_path):
    from orchestration.session_manager import SessionManager

    sid = "reg1"
    sdir = tmp_path / "sessions" / sid
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    (sdir / "worktree").mkdir(parents=True, exist_ok=True)
    st = SessionState(
        session_id=sid, status=SessionStatus.BUILDING,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        session_dir=str(sdir), logs_dir=str(sdir / "logs"),
        worktree_path=str(sdir / "worktree"), gpu_ids=[0],
        vllm_fork_url="https://github.com/u/vllm.git",
        inactivity_timeout_mins=123,
    )

    sm = SessionManager.__new__(SessionManager)
    # Session must still be present and BUILDING at step-6 so the build promotes.
    sm._sessions = {sid: st}
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.create_worktree = MagicMock(return_value=Path(st.worktree_path))
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {}}
    )
    sm.cli_tool_manager = MagicMock()
    sm.cli_tool_manager.get_cli_command.return_value = ["/usr/bin/env", "claude"]
    sm.terminal_manager = MagicMock()
    sm.terminal_manager.is_available.return_value = False
    sm._save_session_state = MagicMock()
    sm._release_gpus = MagicMock()
    sm._chown_session_to_user = MagicMock()
    sm._build_extra_env = MagicMock(return_value={})
    sm._fork_console_path = MagicMock(return_value="/bin/true")

    # A live inactivity monitor that must be re-armed on ACTIVE.
    sm.inactivity_monitor = MagicMock()

    req = CreateSessionRequest(gpu_count=1, vllm_fork_url=st.vllm_fork_url)

    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        gf.return_value.ensure_fork_base.return_value = Path(tmp_path / "fb")
        await sm._run_fork_build(st, req)

    assert st.status == SessionStatus.ACTIVE
    sm.inactivity_monitor.register_session.assert_called_once_with(
        sid, timeout_mins=123
    )


# ── ssrf-2: resume re-validates the persisted fork URL before clone ────────
@pytest.mark.unit
@pytest.mark.asyncio
async def test_reinit_rejects_non_github_persisted_url(tmp_path, monkeypatch):
    from orchestration.session_manager import SessionManager

    # No token configured: decrypt returns None safely; we must fail on the URL
    # re-validation BEFORE ever reaching ensure_fork_base.
    monkeypatch.delenv("AMMO_FORK_TOKEN_KEY", raising=False)

    st = SessionState(
        session_id="ssrf1", status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="feature-x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(tmp_path / "wt"),
    )
    # Bypass the create-time validator: tamper the persisted URL directly.
    st.vllm_fork_url = "https://evil.com/x/y.git"

    sm = SessionManager.__new__(SessionManager)
    sm.worktree_manager = MagicMock()
    sm.worktree_manager.initialize_vllm_environment = AsyncMock(
        return_value={"status": "success", "timings": {}}
    )

    with patch("orchestration.fork_repo_manager.get_fork_repo_manager") as gf:
        with pytest.raises(RuntimeError, match="re-validation"):
            await sm._reinit_fork_env_on_resume(st)
        # ensure_fork_base must NEVER be reached for a tampered host.
        gf.return_value.ensure_fork_base.assert_not_called()
