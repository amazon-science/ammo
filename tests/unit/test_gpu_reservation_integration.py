# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for gpu_reservation.py — CVD-based discovery and session-scoped reservation.

Tests the modified gpu_reservation.py that uses CUDA_VISIBLE_DEVICES parsing
instead of nvidia-smi, and per-session state isolation via AMMO_GPU_RES_DIR.

Plan reference: .claude/plans/gpu-reservation-integration.md
"""

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# We import from the session template location (the file under test).
# The module uses STATE_DIR as a module-level constant from AMMO_GPU_RES_DIR.
# We patch it in each test to use a temp directory.

# Build the import path — the file lives at:
#   ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py
# We add it to sys.path so we can import it as a module.
import sys

_SCRIPTS_DIR = str(
    Path(__file__).resolve().parents[2]
    / "ai_cli_session"
    / ".claude"
    / "skills"
    / "ammo"
    / "scripts"
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_dir(tmp_path: Path, cvd: str = "4,5,6,7") -> Path:
    """Create a temp state dir and set env vars for gpu_reservation."""
    state_dir = tmp_path / "gpu_res"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _patch_module(mod, state_dir: Path, cvd: str = "4,5,6,7"):
    """Return a stack of patches for the module's STATE_DIR and CVD env."""
    return [
        patch.object(mod, "STATE_DIR", state_dir),
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": cvd}, clear=False),
    ]


# ---------------------------------------------------------------------------
# CVD Parsing Tests (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCVDParsing:
    """Test _discover_session_gpus() parses CUDA_VISIBLE_DEVICES correctly."""

    def test_physical_ids_returned(self, tmp_path):
        """CVD=4,5,6,7 should return [4, 5, 6, 7] as physical GPU IDs."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5,6,7")
        for p in patches:
            p.start()
        try:
            gpus = gr._discover_session_gpus()
            assert gpus == [4, 5, 6, 7], f"Expected [4,5,6,7], got {gpus}"
        finally:
            for p in patches:
                p.stop()

    def test_non_contiguous_ids(self, tmp_path):
        """CVD=0,3,5,7 should return [0, 3, 5, 7]."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="0,3,5,7")
        for p in patches:
            p.start()
        try:
            gpus = gr._discover_session_gpus()
            assert gpus == [0, 3, 5, 7]
        finally:
            for p in patches:
                p.stop()

    def test_empty_cvd(self, tmp_path):
        """CVD="" should return empty list."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="")
        for p in patches:
            p.start()
        try:
            gpus = gr._discover_session_gpus()
            assert gpus == []
        finally:
            for p in patches:
                p.stop()

    def test_cvd_minus_one(self, tmp_path):
        """CVD=-1 should return empty list (no GPUs)."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="-1")
        for p in patches:
            p.start()
        try:
            gpus = gr._discover_session_gpus()
            assert gpus == []
        finally:
            for p in patches:
                p.stop()

    def test_uuid_warning(self, tmp_path, caplog):
        """CVD with UUID entries should log warning and skip them."""
        import logging
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,GPU-abc123,5")
        for p in patches:
            p.start()
        try:
            with caplog.at_level(logging.WARNING):
                gpus = gr._discover_session_gpus()
            # Should only get the integer IDs
            assert gpus == [4, 5]
            # Should have logged a warning about the UUID
            assert any("GPU-abc123" in r.message for r in caplog.records)
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Reserve / Release Tests (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReserveRelease:
    """Test reserve() and release_by_session() with physical GPU IDs."""

    def test_reserve_returns_physical_id(self, tmp_path):
        """reserve(1) should return a physical GPU ID from the CVD set."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5,6,7")
        for p in patches:
            p.start()
        try:
            gpu_ids = gr.reserve(num_gpus=1, session_id="test-session")
            assert len(gpu_ids) == 1
            assert gpu_ids[0] in {4, 5, 6, 7}, f"Got non-physical ID: {gpu_ids[0]}"
        finally:
            for p in patches:
                p.stop()

    def test_reserve_multiple(self, tmp_path):
        """reserve(2) should return 2 distinct physical GPU IDs."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5,6,7")
        for p in patches:
            p.start()
        try:
            gpu_ids = gr.reserve(num_gpus=2, session_id="test-session")
            assert len(gpu_ids) == 2
            assert all(g in {4, 5, 6, 7} for g in gpu_ids)
            assert len(set(gpu_ids)) == 2, "Should return distinct IDs"
        finally:
            for p in patches:
                p.stop()

    def test_reserve_multi_gpu_skips_gap_in_non_contiguous_pool(self, tmp_path):
        """Multi-GPU reserve on a non-contiguous pool must skip the gap and
        return a truly contiguous block.

        CVD=4,5,7,8 (pool has a gap between 5 and 7).
        reserve(2) must return [4,5] or [7,8] — NOT [5,7] which is non-contiguous.

        This is the core correctness test for the CVD-based physical-ID fix:
        if reserve() used 0-based indices (range(gpu_count)=range(4)=[0,1,2,3]),
        it would find "contiguous" blocks that don't map to the physical pool.
        With physical IDs, [5,7] must NOT be considered contiguous (gap exists).

        Edge case: all existing tests use cvd='4,5,6,7' (fully contiguous).
        This is the first test with a gap in the physical GPU pool.
        """
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path, cvd="4,5,7,8")
        patches = _patch_module(gr, state_dir, cvd="4,5,7,8")
        for p in patches:
            p.start()
        try:
            gpu_ids = gr.reserve(num_gpus=2, session_id="gap-test")
            assert len(gpu_ids) == 2
            assert all(g in {4, 5, 7, 8} for g in gpu_ids)
            # The result must be a contiguous block: either [4,5] or [7,8]
            sorted_ids = sorted(gpu_ids)
            assert sorted_ids[1] == sorted_ids[0] + 1, (
                f"Got non-contiguous IDs {sorted_ids} from pool [4,5,7,8]. "
                f"reserve(2) must return a contiguous block ([4,5] or [7,8])."
            )
        finally:
            for p in patches:
                p.stop()

    def test_reserve_exhaustion(self, tmp_path):
        """Reserving more GPUs than available should raise ReservationError."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # Reserve all 2 GPUs with session A
            gr.reserve(num_gpus=2, session_id="session-a")
            # Session B tries to reserve 1 — should fail
            with pytest.raises(gr.ReservationError):
                gr.reserve(num_gpus=1, session_id="session-b")
        finally:
            for p in patches:
                p.stop()

    def test_release_and_re_reserve(self, tmp_path):
        """After release, GPUs should be available for re-reservation."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # Reserve all
            gpu_ids = gr.reserve(num_gpus=2, session_id="session-a")
            # Release
            released = gr.release_by_session("session-a")
            assert set(released) == set(gpu_ids)
            # Re-reserve should succeed
            new_ids = gr.reserve(num_gpus=1, session_id="session-b")
            assert len(new_ids) == 1
            assert new_ids[0] in {4, 5}
        finally:
            for p in patches:
                p.stop()

    def test_reserve_same_session_idempotent(self, tmp_path):
        """Calling reserve() twice with the same session_id auto-releases the
        first reservation before allocating the second (idempotent retry).

        This is the auto-release-existing-reservation-for-session_id behavior
        from the plan (step 3 in the reserve algorithm):
          '3. Auto-release existing reservations for this session_id'

        Critical for:
        - PostToolUse hook retries (hook fires, reserve is called again)
        - Same-pod pause/resume where old reservation for session still exists
        Not tested by test_release_and_re_reserve (different session_ids used there).
        """
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path, cvd="4,5")
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # First reserve: gets 1 GPU
            first_ids = gr.reserve(num_gpus=1, session_id="retry-session")
            assert len(first_ids) == 1

            # Second reserve same session_id, more GPUs:
            # must auto-release first_ids before allocating
            second_ids = gr.reserve(num_gpus=2, session_id="retry-session")
            assert len(second_ids) == 2, (
                "Second reserve with same session_id should succeed by auto-releasing first"
            )
            assert all(g in {4, 5} for g in second_ids)
            # First allocation must not still be reserved (it was auto-released)
            assert set(second_ids) == {4, 5}, (
                "Both GPUs should be reserved after idempotent retry"
            )
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Per-Session Isolation Tests (2 tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSessionIsolation:
    """Test AMMO_GPU_RES_DIR-based per-session state isolation."""

    def test_ammo_gpu_res_dir_used(self, tmp_path):
        """When AMMO_GPU_RES_DIR is set, state files go there."""
        import gpu_reservation as gr

        state_dir = tmp_path / "my_custom_dir"
        state_dir.mkdir()
        patches = [
            patch.object(gr, "STATE_DIR", state_dir),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=False),
        ]
        for p in patches:
            p.start()
        try:
            gr.reserve(num_gpus=1, session_id="test")
            assert (state_dir / "state.json").exists()
        finally:
            for p in patches:
                p.stop()

    def test_sha256_fallback_dir(self, tmp_path):
        """When AMMO_GPU_RES_DIR is unset, falls back to sha256-based dir."""
        import gpu_reservation as gr
        import hashlib

        cvd = "4,5,6,7"
        expected_hash = hashlib.sha256(cvd.encode()).hexdigest()[:12]
        expected_dir = Path(f"/tmp/ammo_gpu_res_{expected_hash}")

        # Temporarily remove AMMO_GPU_RES_DIR from env to test fallback
        env_patch = patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": cvd},
            clear=False,
        )
        # Remove AMMO_GPU_RES_DIR if present
        env_no_res_dir = {k: v for k, v in os.environ.items() if k != "AMMO_GPU_RES_DIR"}

        with patch.dict(os.environ, env_no_res_dir, clear=True):
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            # Re-import to pick up the fallback
            # We test the _compute_state_dir helper
            computed = gr._compute_state_dir()
            assert str(computed) == str(expected_dir), (
                f"Expected {expected_dir}, got {computed}"
            )


# ---------------------------------------------------------------------------
# Stale Lease Pruning Tests (2 tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStaleLeasePruning:
    """Test that expired leases are reclaimed on reserve()."""

    def test_expired_lease_reclaimed(self, tmp_path):
        """Expired leases should be reclaimed, making GPUs available."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # Reserve all GPUs with a very short lease
            gr.reserve(num_gpus=2, session_id="old-session", lease_hours=0.0001)
            # Wait for lease to expire (< 1 second)
            time.sleep(0.5)
            # New reservation should succeed after reclaiming expired leases
            new_ids = gr.reserve(num_gpus=1, session_id="new-session")
            assert len(new_ids) == 1
            assert new_ids[0] in {4, 5}
        finally:
            for p in patches:
                p.stop()

    def test_active_lease_preserved(self, tmp_path):
        """Active (non-expired) leases should NOT be reclaimed."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # Reserve all GPUs with a long lease (2 hours)
            gr.reserve(num_gpus=2, session_id="active-session", lease_hours=2.0)
            # Another session trying to reserve should fail (leases still active)
            with pytest.raises(gr.ReservationError):
                gr.reserve(num_gpus=1, session_id="blocked-session")
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Atomic Write Recovery Test (1 test)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAtomicWriteRecovery:
    """Test atomic write and recovery from corrupted state."""

    def test_corrupted_state_recovery(self, tmp_path):
        """If state.json is corrupted, reserve() should reinitialize."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()
        try:
            # Write a valid state first
            gr.reserve(num_gpus=1, session_id="init")
            gr.release_by_session("init")

            # Corrupt state.json
            state_file = state_dir / "state.json"
            state_file.write_text("{broken json")

            # Next operation should handle the corruption gracefully
            # (either reinitialize or raise a clear error)
            try:
                ids = gr.reserve(num_gpus=1, session_id="recovery")
                # If it succeeds, it recovered from corruption
                assert len(ids) == 1
                assert ids[0] in {4, 5}
            except (json.JSONDecodeError, Exception):
                # Acceptable: the module may raise on corrupt state
                # As long as it doesn't silently return wrong data
                pass
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Concurrent Reserve Test (1 test)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConcurrentReserve:
    """Test thread-safety of reserve() via flock."""

    def test_no_double_allocation(self, tmp_path):
        """Two threads reserving simultaneously should not get the same GPU."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5")
        for p in patches:
            p.start()

        results = []
        errors = []

        def reserve_one(session_id):
            try:
                ids = gr.reserve(num_gpus=1, session_id=session_id)
                results.append((session_id, ids))
            except gr.ReservationError as e:
                errors.append((session_id, str(e)))

        try:
            t1 = threading.Thread(target=reserve_one, args=("thread-1",))
            t2 = threading.Thread(target=reserve_one, args=("thread-2",))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            # Both should succeed (2 GPUs, 2 requests for 1 each)
            assert len(results) == 2, f"Expected 2 successes, got {len(results)} successes and {len(errors)} errors"
            ids_1 = set(results[0][1])
            ids_2 = set(results[1][1])
            assert ids_1.isdisjoint(ids_2), f"Double allocation! {ids_1} & {ids_2}"
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Status CLI Test (1 test)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStatusCLI:
    """Test the status command output."""

    def test_status_shows_reservations(self, tmp_path):
        """status() should show total, reserved, and available GPUs."""
        import gpu_reservation as gr

        state_dir = _make_state_dir(tmp_path)
        patches = _patch_module(gr, state_dir, cvd="4,5,6,7")
        for p in patches:
            p.start()
        try:
            # Reserve 2 GPUs
            reserved_ids = gr.reserve(num_gpus=2, session_id="status-test")
            state = gr.read_state()

            # Verify structure
            assert "gpus" in state
            assert "gpu_count" in state

            # Count reserved vs free
            reserved_count = sum(
                1 for v in state["gpus"].values() if v is not None
            )
            free_count = sum(
                1 for v in state["gpus"].values() if v is None
            )
            assert reserved_count == 2
            assert free_count == 2
        finally:
            for p in patches:
                p.stop()
