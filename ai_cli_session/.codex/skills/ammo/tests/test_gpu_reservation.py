#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for gpu_reservation.py — the foundation GPU reservation module.

Run with:
    python -m pytest .codex/skills/ammo/tests/test_gpu_reservation.py -v
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gpu_reservation
from gpu_reservation import (
    LockTimeoutError,
    ReservationError,
    check_and_reclaim_expired,
    force_clear,
    read_state,
    release_by_session,
    reserve,
    write_reservation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_dir(tmp_path: Path) -> Path:
    """Return a subdirectory inside tmp_path that does NOT yet exist (so
    _init_state can create it)."""
    d = tmp_path / "ammo_gpu_res"
    return d


# ---------------------------------------------------------------------------
# TestInitState
# ---------------------------------------------------------------------------

class TestInitState:
    def test_creates_state_with_gpu_count(self, tmp_path):
        """_init_state with 4 discovered GPUs creates state with 4 null entries."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            state = read_state()

        assert state["gpu_count"] == 4
        assert len(state["gpus"]) == 4
        for v in state["gpus"].values():
            assert v is None

    def test_reads_existing_state(self, tmp_path):
        """read_state returns an already-written state without re-initialising."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)

        existing = {
            "gpus": {"0": None, "1": None},
            "gpu_count": 2,
            "audit": [],
        }
        state_file = state_dir / "state.json"
        state_file.write_text(json.dumps(existing))

        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(99))):
            # _discover_session_gpus should NOT be called because state already exists
            state = read_state()

        assert state["gpu_count"] == 2
        assert len(state["gpus"]) == 2


# ---------------------------------------------------------------------------
# TestAcquireFlock
# ---------------------------------------------------------------------------

class TestAcquireFlock:
    def test_acquires_lock_on_first_try(self, tmp_path):
        """Basic lock/unlock cycle succeeds without errors."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_file = state_dir / "state.lock"

        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir):
            fh = gpu_reservation._acquire_flock()
            assert fh is not None
            gpu_reservation._release_flock(fh)

    def test_raises_after_5_retries(self, tmp_path):
        """LockTimeoutError raised when lock cannot be acquired after 5 attempts."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_file = state_dir / "state.lock"

        # Hold the lock in this process
        holder = open(lock_file, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
                 mock.patch.object(gpu_reservation, "BACKOFF_DELAYS", [0.001] * 4):
                with pytest.raises(LockTimeoutError):
                    gpu_reservation._acquire_flock()
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()


# ---------------------------------------------------------------------------
# TestWriteReservation
# ---------------------------------------------------------------------------

class TestWriteReservation:
    def test_reserves_free_gpus(self, tmp_path):
        """Reserving free GPUs writes entries into state."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            write_reservation(
                gpu_ids=[0, 1],
                session_id="sess001",
                command_snippet="python bench.py",
                lease_hours=2.0,
            )
            state = read_state()

        assert state["gpus"]["0"] is not None
        assert state["gpus"]["0"]["session_id"] == "sess001"
        assert state["gpus"]["1"] is not None
        assert state["gpus"]["2"] is None
        assert state["gpus"]["3"] is None

    def test_rejects_unknown_gpu_ids(self, tmp_path):
        """Reserving a GPU ID that doesn't exist in state raises ReservationError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            with pytest.raises(ReservationError, match="not found in state"):
                write_reservation(
                    gpu_ids=[7],
                    session_id="sess_x",
                    command_snippet="python phantom.py",
                    lease_hours=2.0,
                )

    def test_truncates_command_snippet(self, tmp_path):
        """command_snippet is truncated to 80 characters in the stored entry."""
        state_dir = _make_temp_dir(tmp_path)
        long_snippet = "x" * 200
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(1))):
            write_reservation(
                gpu_ids=[0],
                session_id="sess_t",
                command_snippet=long_snippet,
                lease_hours=2.0,
            )
            state = read_state()

        assert len(state["gpus"]["0"]["command_snippet"]) == 80

    def test_blocks_on_held_gpu(self, tmp_path):
        """Trying to reserve an already-held GPU raises ReservationError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(2))):
            # First reservation succeeds
            write_reservation(
                gpu_ids=[0],
                session_id="sess_a",
                command_snippet="python a.py",
                lease_hours=2.0,
            )
            # Second attempt on same GPU must fail
            with pytest.raises(ReservationError):
                write_reservation(
                    gpu_ids=[0],
                    session_id="sess_b",
                    command_snippet="python b.py",
                    lease_hours=2.0,
                )


# ---------------------------------------------------------------------------
# TestLeaseExpiry
# ---------------------------------------------------------------------------

class TestLeaseExpiry:
    def test_reclaims_expired_leases(self, tmp_path):
        """check_and_reclaim_expired removes entries with lease_hours=0 (already expired)."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(2))):
            # lease_hours=0 means it expires immediately (at the time of writing)
            write_reservation(
                gpu_ids=[0],
                session_id="sess_d",
                command_snippet="python d.py",
                lease_hours=0.0,
            )
            # Small sleep to ensure expiry has passed
            time.sleep(0.01)
            reclaimed = check_and_reclaim_expired()
            state = read_state()

        assert 0 in reclaimed
        assert state["gpus"]["0"] is None


# ---------------------------------------------------------------------------
# TestForceClear
# ---------------------------------------------------------------------------

class TestForceClear:
    def test_clears_by_session_id(self, tmp_path):
        """force_clear(session_id='s1') removes only entries owned by s1."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(2))):
            write_reservation(
                gpu_ids=[0],
                session_id="s1",
                command_snippet="python e.py",
                lease_hours=2.0,
            )
            write_reservation(
                gpu_ids=[1],
                session_id="s2",
                command_snippet="python f.py",
                lease_hours=2.0,
            )
            force_clear(session_id="s1")
            state = read_state()

        assert state["gpus"]["0"] is None, "GPU 0 (owned by s1) should be cleared"
        assert state["gpus"]["1"] is not None, "GPU 1 (owned by s2) should remain"

    def test_requires_session_or_flag(self, tmp_path):
        """force_clear() with no args raises ValueError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(1))):
            with pytest.raises(ValueError):
                force_clear()


# ---------------------------------------------------------------------------
# TestReserve
# ---------------------------------------------------------------------------

class TestReserve:
    def test_basic_allocation(self, tmp_path):
        """reserve(1) on a 4-GPU system returns [0] and marks GPU 0 reserved."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            result = reserve(num_gpus=1, session_id="s1")
            state = read_state()

        assert result == [0]
        assert state["gpus"]["0"] is not None
        assert state["gpus"]["0"]["session_id"] == "s1"
        assert state["gpus"]["1"] is None

    def test_contiguous_allocation(self, tmp_path):
        """reserve(2) on a 4-GPU system returns [0, 1] (contiguous block)."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            result = reserve(num_gpus=2, session_id="s1")

        assert result == [0, 1]

    def test_contiguous_skips_held(self, tmp_path):
        """GPU 0 held, reserve(2) returns [1, 2] (next contiguous block)."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # Hold GPU 0
            write_reservation(gpu_ids=[0], session_id="other")
            # Reserve 2 contiguous GPUs
            result = reserve(num_gpus=2, session_id="s1")

        assert result == [1, 2]

    def test_contiguous_fails_fragmented(self, tmp_path):
        """GPUs 0,2 held on 4-GPU system, reserve(2) fails (only [1] and [3] free)."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # Hold GPUs 0 and 2 with different sessions
            write_reservation(gpu_ids=[0], session_id="other_a")
            write_reservation(gpu_ids=[2], session_id="other_b")
            # Try to reserve 2 contiguous — should fail
            with pytest.raises(ReservationError, match="No contiguous block"):
                reserve(num_gpus=2, session_id="s1")

    def test_auto_release_on_retry(self, tmp_path):
        """Session 's1' holds GPUs 0,1. Calling reserve(2, 's1') releases old first."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # First reservation
            result1 = reserve(num_gpus=2, session_id="s1")
            assert result1 == [0, 1]
            # Retry — should auto-release [0,1] then re-allocate
            result2 = reserve(num_gpus=2, session_id="s1")
            assert result2 == [0, 1]
            state = read_state()
            assert state["gpus"]["0"]["session_id"] == "s1"
            assert state["gpus"]["1"]["session_id"] == "s1"

    def test_pool_exhausted(self, tmp_path):
        """All 4 GPUs held by different sessions, reserve(1) fails."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # Fill all GPUs with different sessions
            for i in range(4):
                write_reservation(gpu_ids=[i], session_id=f"other_{i}")
            # Try to reserve 1 — should fail
            with pytest.raises(ReservationError, match="Not enough free GPUs"):
                reserve(num_gpus=1, session_id="s_new")

    def test_contiguous_skips_middle_gap(self, tmp_path):
        """GPU 1 held by different session, GPUs 0,2,3 free, request 2 -> [2,3]."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # Hold GPU 1
            write_reservation(gpu_ids=[1], session_id="other")
            # Reserve 2 contiguous — GPU 0 alone is not contiguous with
            # anything useful (GPU 1 is held), so [2,3] is the first pair.
            result = reserve(num_gpus=2, session_id="s1")

        assert result == [2, 3]

    def test_non_contiguous_session_pool_uses_physical_ids(self, tmp_path):
        """Session pools may expose non-contiguous physical GPU IDs via CVD."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=[4, 6, 7]):
            result = reserve(num_gpus=2, session_id="s1")
            state = read_state()

        assert result == [6, 7]
        assert sorted(state["gpus"].keys()) == ["4", "6", "7"]
        assert state["gpus"]["4"] is None
        assert state["gpus"]["6"]["session_id"] == "s1"
        assert state["gpus"]["7"]["session_id"] == "s1"

    def test_failed_reserve_preserves_existing(self, tmp_path):
        """Failed reserve must NOT destroy the session's existing reservations."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            # s1 holds GPU 0
            reserve(num_gpus=1, session_id="s1")
            # Fill GPUs 1,2,3 with other sessions
            write_reservation(gpu_ids=[1], session_id="other_1")
            write_reservation(gpu_ids=[2], session_id="other_2")
            write_reservation(gpu_ids=[3], session_id="other_3")

            # Try to reserve 2 GPUs for s1 — must fail (after auto-releasing
            # GPU 0, only 1 GPU free, need 2)
            with pytest.raises(ReservationError):
                reserve(num_gpus=2, session_id="s1")

            # GPU 0 must STILL be held by s1 (failed reserve should not
            # have persisted the auto-release)
            state = read_state()
            assert state["gpus"]["0"] is not None
            assert state["gpus"]["0"]["session_id"] == "s1"

    def test_num_gpus_zero_raises(self, tmp_path):
        """reserve(num_gpus=0) raises ValueError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            with pytest.raises(ValueError, match="num_gpus must be positive"):
                reserve(num_gpus=0, session_id="s1")

    def test_num_gpus_negative_raises(self, tmp_path):
        """reserve(num_gpus=-1) raises ValueError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            with pytest.raises(ValueError, match="num_gpus must be positive"):
                reserve(num_gpus=-1, session_id="s1")

    def test_reclaims_expired_before_allocating(self, tmp_path):
        """GPU 0 has expired lease, reserve(1) reclaims it and returns [0]."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(1))):
            # Create an immediately-expiring reservation
            write_reservation(
                gpu_ids=[0],
                session_id="old_session",
                lease_hours=0.0,
            )
            time.sleep(0.01)
            # Reserve should reclaim the expired lease and allocate GPU 0
            result = reserve(num_gpus=1, session_id="new_session")

        assert result == [0]


# ---------------------------------------------------------------------------
# TestReleaseBySession
# ---------------------------------------------------------------------------

class TestReleaseBySession:
    def test_releases_all_session_gpus(self, tmp_path):
        """release_by_session('s1') clears all GPUs held by s1."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            reserve(num_gpus=2, session_id="s1")
            released = release_by_session("s1")
            state = read_state()

        assert sorted(released) == [0, 1]
        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"] is None

    def test_leaves_other_sessions(self, tmp_path):
        """release_by_session('s1') only clears s1, leaves s2 intact."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            write_reservation(gpu_ids=[0], session_id="s1")
            write_reservation(gpu_ids=[1], session_id="s2")
            released = release_by_session("s1")
            state = read_state()

        assert released == [0]
        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"] is not None
        assert state["gpus"]["1"]["session_id"] == "s2"

    def test_unknown_session_returns_empty(self, tmp_path):
        """release_by_session('nonexistent') returns []."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(2))):
            result = release_by_session("nonexistent")

        assert result == []

    def test_include_children_releases_prefixed_subagent_sessions(self, tmp_path):
        """release_by_session(..., include_children=True) clears session child reservations."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_session_gpus", return_value=list(range(4))):
            write_reservation(gpu_ids=[0], session_id="sess")
            write_reservation(gpu_ids=[1], session_id="sess:agent-a")
            write_reservation(gpu_ids=[2], session_id="sess-other")
            write_reservation(gpu_ids=[3], session_id="other:sess")

            released = release_by_session("sess", include_children=True)
            state = read_state()

        assert sorted(released) == [0, 1]
        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"] is None
        assert state["gpus"]["2"]["session_id"] == "sess-other"
        assert state["gpus"]["3"]["session_id"] == "other:sess"


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_reserve_stdout_format(self, tmp_path):
        """CLI 'reserve --num-gpus 2 --session-id test' prints '0,1' to stdout."""
        state_dir = _make_temp_dir(tmp_path)
        script = str(_SCRIPTS_DIR / "gpu_reservation.py")
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(state_dir)

        # Pre-create state so nvidia-smi isn't needed
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "state.json"
        state_file.write_text(json.dumps({
            "gpus": {"0": None, "1": None, "2": None, "3": None},
            "gpu_count": 4,
            "audit": [],
        }))

        result = subprocess.run(
            [sys.executable, script, "reserve", "--num-gpus", "2",
             "--session-id", "test"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "0,1"

    def test_reserve_failure_exits_nonzero(self, tmp_path):
        """CLI 'reserve --num-gpus 1' when all GPUs held exits with code 1."""
        state_dir = _make_temp_dir(tmp_path)
        script = str(_SCRIPTS_DIR / "gpu_reservation.py")
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(state_dir)

        # Pre-create state with all GPUs held
        state_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        far_future = "2099-01-01T00:00:00Z"
        state_file = state_dir / "state.json"
        state_file.write_text(json.dumps({
            "gpus": {
                str(i): {
                    "session_id": f"blocker_{i}",
                    "reserved_at": now_str,
                    "lease_expires": far_future,
                    "command_snippet": "hold",
                } for i in range(4)
            },
            "gpu_count": 4,
            "audit": [],
        }))

        result = subprocess.run(
            [sys.executable, script, "reserve", "--num-gpus", "1",
             "--session-id", "new"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert result.returncode == 1
        assert "Not enough free GPUs" in result.stderr

    def test_release_session_cli_include_children(self, tmp_path):
        """CLI release-session --include-children releases exact and child session ids."""
        state_dir = _make_temp_dir(tmp_path)
        script = str(_SCRIPTS_DIR / "gpu_reservation.py")
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(state_dir)

        state_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        far_future = "2099-01-01T00:00:00Z"
        (state_dir / "state.json").write_text(json.dumps({
            "gpus": {
                "0": {"session_id": "sess", "reserved_at": now_str, "lease_expires": far_future, "command_snippet": "hold"},
                "1": {"session_id": "sess:agent", "reserved_at": now_str, "lease_expires": far_future, "command_snippet": "hold"},
                "2": {"session_id": "other", "reserved_at": now_str, "lease_expires": far_future, "command_snippet": "hold"},
            },
            "gpu_count": 3,
            "audit": [],
        }))

        result = subprocess.run(
            [sys.executable, script, "release-session", "--session-id", "sess", "--include-children"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        state = json.loads((state_dir / "state.json").read_text())

        assert result.returncode == 0
        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"] is None
        assert state["gpus"]["2"]["session_id"] == "other"
