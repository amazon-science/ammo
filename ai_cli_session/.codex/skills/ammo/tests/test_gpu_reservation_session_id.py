# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for GPU reservation session_id and auto-release fixes.

Bug 1: shared default session_id ("cli") causes auto-eviction between parallel agents
Bug 2: hooks and CLI reservations must agree on the session-id default chain.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# The module under test. Live at skills/ammo/scripts/; make it importable
# regardless of the pytest invocation cwd (matches tests/test_sweep_dp.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
SCRIPT_PATH = _SCRIPTS_DIR / "gpu_reservation.py"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Each test gets its own state directory and a 2-GPU pool."""
    state_dir = tmp_path / "gpu_res"
    state_dir.mkdir()
    monkeypatch.setattr("gpu_reservation.STATE_DIR", state_dir)
    # Suppress real nvidia-smi calls: seed with 2 GPUs
    state = {"gpus": {"0": None, "1": None}, "gpu_count": 2, "audit": []}
    (state_dir / "state.json").write_text(json.dumps(state))
    # Fast lock retries for tests
    monkeypatch.setattr("gpu_reservation.BACKOFF_DELAYS", [0.01])
    return state_dir


# Import after fixture definition so monkeypatch can apply
import gpu_reservation  # noqa: E402
from gpu_reservation import ReservationError, reserve, release_by_session  # noqa: E402


# ---------------------------------------------------------------------------
# Bug 1: Two agents with same default session_id auto-evict each other
# ---------------------------------------------------------------------------

class TestAutoEvictionBug:
    """Reserve auto-releases previous reservations for the same session_id.
    When two agents both use "cli", agent B evicts agent A."""

    def test_same_session_id_causes_eviction(self):
        """Demonstrates the bug: agent A's reservation is silently released."""
        # Agent A reserves GPU 0
        result_a = reserve(num_gpus=1, session_id="cli")
        assert result_a == [0]

        # Agent B reserves GPU — auto-release frees agent A's GPU first
        result_b = reserve(num_gpus=1, session_id="cli")
        # Bug: B gets GPU 0 (A's was freed) instead of GPU 1
        assert result_b == [0], "Same session_id causes eviction — B took A's GPU"

    def test_different_session_ids_no_eviction(self):
        """With unique session_ids, agents don't evict each other."""
        result_a = reserve(num_gpus=1, session_id="session-aaa")
        assert result_a == [0]

        result_b = reserve(num_gpus=1, session_id="session-bbb")
        assert result_b == [1], "Different session_ids should not evict"

    def test_completion_release_for_one_unique_agent_does_not_free_sibling(self):
        """PostToolUse-style release is safe only with per-agent session ids."""
        reserve(num_gpus=1, session_id="campaign-op007-agent-a", auto_release=False)
        reserve(num_gpus=1, session_id="campaign-op007-agent-b", auto_release=False)

        released = release_by_session("campaign-op007-agent-a")

        assert released == [0]
        state = gpu_reservation.read_state()
        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"]["session_id"] == "campaign-op007-agent-b"

    def test_no_auto_release_prevents_eviction(self):
        """--no-auto-release flag should skip the auto-release step."""
        # Agent A reserves GPU 0
        reserve(num_gpus=1, session_id="shared-session")

        # Agent B reserves with no_auto_release=True — should get GPU 1, not 0
        result_b = reserve(
            num_gpus=1, session_id="shared-session", auto_release=False
        )
        assert result_b == [1], "no_auto_release should skip eviction"

    def test_no_auto_release_fails_when_pool_exhausted(self):
        """With no_auto_release, if pool is full, should raise ReservationError."""
        reserve(num_gpus=1, session_id="shared-session")
        reserve(num_gpus=1, session_id="shared-session", auto_release=False)

        # Pool is now full (2 GPUs, both reserved)
        with pytest.raises(ReservationError, match="Not enough free GPUs"):
            reserve(num_gpus=1, session_id="shared-session", auto_release=False)

    def test_auto_release_default_still_works_for_retries(self):
        """Default behavior (auto_release=True) still works for retry scenarios."""
        # Agent reserves GPU 0
        reserve(num_gpus=1, session_id="retry-session")

        # Same agent retries — auto-release frees old reservation first
        result = reserve(num_gpus=1, session_id="retry-session")
        assert result == [0], "Auto-release should still work by default for retries"


# ---------------------------------------------------------------------------
# Bug 2: CLI default session_id should follow the session-id env chain
# ---------------------------------------------------------------------------

class TestCLISessionIdDefault:
    """CLI --session-id default follows $AMMO_SESSION_ID, then $CLAUDE_SESSION_ID."""

    def _run_cli(self, args: list[str], env_override: dict | None = None):
        """Run gpu_reservation.py as a subprocess with controlled env."""
        env = os.environ.copy()
        # Point at the test's isolated state dir
        env["AMMO_GPU_RES_DIR"] = str(gpu_reservation.STATE_DIR)
        if env_override:
            env.update(env_override)
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH)] + args,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

    def test_cli_uses_ammo_session_id_env(self):
        """When AMMO_SESSION_ID is set, reserve should use it as session_id."""
        result = self._run_cli(
            ["reserve", "--num-gpus", "1"],
            env_override={"AMMO_SESSION_ID": "agent-xyz-123"},
        )
        assert result.returncode == 0
        gpu_id = result.stdout.strip()
        assert gpu_id == "0"

        # Verify the reservation was stored with the env var session_id
        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"]["session_id"] == "agent-xyz-123"

    def test_cli_uses_claude_session_id_when_ammo_unset(self):
        """CLAUDE_SESSION_ID is the runtime fallback when AMMO_SESSION_ID is absent."""
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(gpu_reservation.STATE_DIR)
        env.pop("AMMO_SESSION_ID", None)
        env["CLAUDE_SESSION_ID"] = "claude-fallback-id"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "reserve", "--num-gpus", "1"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0

        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"]["session_id"] == "claude-fallback-id"

    def test_ammo_session_id_wins_over_claude_session_id(self):
        """AMMO_SESSION_ID is the portable value the server injects; it wins."""
        result = self._run_cli(
            ["reserve", "--num-gpus", "1"],
            env_override={
                "AMMO_SESSION_ID": "ammo-id",
                "CLAUDE_SESSION_ID": "claude-id",
            },
        )
        assert result.returncode == 0

        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"]["session_id"] == "ammo-id"

    def test_cli_falls_back_to_cli_without_env(self):
        """Without either env var, default should still be 'cli'."""
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(gpu_reservation.STATE_DIR)
        env.pop("AMMO_SESSION_ID", None)
        env.pop("CLAUDE_SESSION_ID", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "reserve", "--num-gpus", "1"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0

        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"]["session_id"] == "cli"

    def test_cli_explicit_session_id_overrides_env(self):
        """Explicit --session-id should override the env var default."""
        result = self._run_cli(
            ["reserve", "--num-gpus", "1", "--session-id", "explicit-id"],
            env_override={"AMMO_SESSION_ID": "agent-xyz-123"},
        )
        assert result.returncode == 0

        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"]["session_id"] == "explicit-id"

    def test_cli_no_auto_release_flag(self):
        """CLI --no-auto-release flag should be accepted and skip auto-release."""
        # First reservation
        self._run_cli(
            ["reserve", "--num-gpus", "1"],
            env_override={"AMMO_SESSION_ID": "shared-agent"},
        )

        # Second reservation with --no-auto-release
        result = self._run_cli(
            ["reserve", "--num-gpus", "1", "--no-auto-release"],
            env_override={"AMMO_SESSION_ID": "shared-agent"},
        )
        assert result.returncode == 0
        gpu_id = result.stdout.strip()
        assert gpu_id == "1", "Should get GPU 1, not re-take GPU 0"

        # Both GPUs should be reserved
        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"] is not None
        assert state["gpus"]["1"] is not None


# ---------------------------------------------------------------------------
# Hook compatibility: release-session matches the env session id
# ---------------------------------------------------------------------------

class TestHookCompatibility:
    """The PostToolUse hook releases by the session-id env chain.
    After the fix, reserve uses the same value, so release-session matches."""

    def test_hook_release_matches_env_session_id(self):
        """Reserve via CLI with the env session id, then release-session matches."""
        env = os.environ.copy()
        env["AMMO_GPU_RES_DIR"] = str(gpu_reservation.STATE_DIR)
        env["AMMO_SESSION_ID"] = "hook-test-session"

        # Reserve (uses the env session id as default)
        r1 = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "reserve", "--num-gpus", "1"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert r1.returncode == 0

        # Release by session (what the hook does)
        r2 = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "release-session",
             "--session-id", "hook-test-session"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert r2.returncode == 0

        # GPU should be free now
        state = json.loads(
            (gpu_reservation.STATE_DIR / "state.json").read_text()
        )
        assert state["gpus"]["0"] is None, "Hook release should free the GPU"
