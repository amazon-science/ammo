# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for ammo-state-validate.sh audit-gate blocking (Task 2 / RED).

These tests should FAIL until Task 4 lands the audit-blocking logic in the
hook. Each test crafts a schema-valid state.json and invokes the bash hook
via subprocess, asserting on the decision/reason JSON output.

Legacy gate: audit gate fires ONLY when `audit` key is present in the round.
If absent entirely (legacy campaign), gate is skipped — preserves backward
compat for in-flight campaigns. Only NEW campaigns (post-new_target.py
update) will carry the `audit` key, which activates the gate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).parent.parent.parent
HOOK = ROOT / "ai_cli_session" / ".claude" / "hooks" / "ammo-state-validate.sh"
SCHEMA = ROOT / "ai_cli_session" / ".claude" / "schemas" / "state.schema.json"
NEW_TARGET_SCRIPT = ROOT / "ai_cli_session" / ".claude" / "skills" / "ammo" / "scripts" / "new_target.py"


def _full_round(round_id: int = 1, *, with_audit: bool = False,
                audit_fields: Optional[Dict[str, Any]] = None,
                tracks_terminal: bool = False) -> Dict[str, Any]:
    rnd: Dict[str, Any] = {
        "round_id": round_id,
        "status": "IN_PROGRESS",
        "team_name": None,
        "profiling_baseline_path": None,
        "baseline": {"started_at": "2026-05-05T10:00:00Z", "completed_at": "2026-05-05T10:30:00Z"},
        "bottleneck_mining": {"started_at": None, "completed_at": None, "top_bottleneck_share_pct": None},
        "debate": {
            "started_at": None, "completed_at": None,
            "candidates": [], "rounds_completed": 0, "max_rounds": 4, "selected_winners": [],
        },
        "parallel_tracks": {"started_at": None, "completed_at": None, "tracks": {}},
        "integration": {
            "started_at": None, "completed_at": None, "status": "pending",
            "passing_candidates": [], "failed_candidates": [], "selected_candidates": [],
            "conflict_analysis": None, "combined_patch_branch": None, "combined_e2e_result": None,
            "final_decision": None, "resolver_invoked": None, "resolver_outcome": None,
            "conflicting_tracks": None,
        },
        "campaign_eval": {"started_at": None, "completed_at": None},
        "shipped": [], "dropped": [],
        "cumulative_speedup_after": None, "combined_e2e_speedup_x": None,
        "combined_e2e_delta_pp": None, "note": None, "round_summary": None,
    }
    if tracks_terminal:
        rnd["parallel_tracks"]["tracks"] = {
            "op-a": {"status": "PASS"},
            "op-b": {"status": "GATED_PASS"},
        }
        rnd["parallel_tracks"]["started_at"] = "2026-05-05T11:00:00Z"
        rnd["parallel_tracks"]["completed_at"] = "2026-05-05T11:30:00Z"
    if with_audit:
        rnd["audit"] = audit_fields if audit_fields is not None else {}
    return rnd


def _state(current_stage: str, current_round: int = 1,
           rounds: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "target": {"model_id": "m", "hardware": "H100", "dtype": "bf16", "tp": 1, "ep": 1, "component": "fused_moe"},
        "session_id": None,
        "gpu_resources": {"gpu_count": 0, "gpu_model": "H100", "memory_total_gib": 0, "cuda_visible_devices": "0"},
        "campaign": {
            "schema_version": "3.0",
            "status": "active",
            "current_round": current_round,
            "current_stage": current_stage,
            "config": {
                "min_e2e_improvement_pct": 3,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
            },
            "cumulative_speedup_vs_round1": 1.0,
            "round_1_baseline_latency_s": None,
            "shipped_optimizations": [],
            "agent_costs": [],
            "rounds": rounds if rounds is not None else [_full_round()],
        },
    }


def _stage_project(tmp_path: Path, state: Dict[str, Any]) -> Path:
    """Create artifact dir with state.json AND a schema-resolution point.

    The hook walks up from state.json looking for .claude/schemas/state.schema.json.
    Symlink or copy the schema so the walk finds it inside tmp_path.
    """
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "auto_target"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_file = artifact_dir / "state.json"
    state_file.write_text(json.dumps(state, indent=2))

    # Provide the schema alongside the tmp project so the hook can find it.
    schema_dir = tmp_path / ".claude" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "state.schema.json").write_text(SCHEMA.read_text())
    return state_file


def _run_hook(state_file: Path) -> Dict[str, Any]:
    payload = {
        "session_id": f"test-{uuid.uuid4().hex[:8]}",
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(state_file)},
    }
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(state_file.parent.parent.parent)
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    parsed: Dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = {}
    return {
        "decision": parsed.get("decision"),
        "reason": parsed.get("reason", ""),
        "ctx": parsed.get("hookSpecificOutput", {}).get("additionalContext", ""),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "rc": proc.returncode,
    }


# ───────────── Tests ─────────────


class TestAuditGateBlocks:
    def test_blocks_transition_1_to_2_without_audit(self, tmp_path):
        """audit:{} present but no stage_1.passed_at, current_stage=2_bottleneck_mining → block."""
        rnd = _full_round(with_audit=True, audit_fields={})
        state = _state("2_bottleneck_mining", rounds=[rnd])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] == "block", (
            f"Expected block on stage 1→2 w/o audit. Got: {result!r}"
        )
        assert "stage_1" in result["reason"].lower() or "audit" in result["reason"].lower()

    def test_allows_transition_1_to_2_with_audit(self, tmp_path):
        """audit.stage_1.passed_at set → no block on 2_bottleneck_mining."""
        rnd = _full_round(
            with_audit=True,
            audit_fields={"stage_1": {"passed_at": "2026-05-05T11:00:00Z",
                                      "verdict_file": "audits/s1.md"}},
        )
        state = _state("2_bottleneck_mining", rounds=[rnd])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] != "block", (
            f"Unexpected block when audit.stage_1.passed_at is set. Got: {result!r}"
        )

    def test_blocks_transition_45_to_6_without_audit(self, tmp_path):
        """current_stage=6_integration, audit:{} (no stage_45.passed_at), all tracks terminal → block."""
        rnd = _full_round(with_audit=True, audit_fields={}, tracks_terminal=True)
        state = _state("6_integration", rounds=[rnd])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] == "block", (
            f"Expected block on 4-5→6 w/o audit. Got: {result!r}"
        )
        assert "stage_45" in result["reason"].lower() or "audit" in result["reason"].lower()

    def test_blocks_transition_6_to_7_without_audit(self, tmp_path):
        """current_stage=7_campaign_eval, audit:{} (no stage_6.passed_at) → block."""
        rnd = _full_round(with_audit=True, audit_fields={})
        rnd["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        rnd["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        rnd["integration"]["status"] = "completed"
        rnd["shipped"] = ["op-winner"]
        state = _state("7_campaign_eval", rounds=[rnd])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] == "block", (
            f"Expected block on 6→7 w/o audit. Got: {result!r}"
        )
        assert "stage_6" in result["reason"].lower() or "audit" in result["reason"].lower()


class TestLegacyPassthrough:
    def test_legacy_campaign_no_audit_key_transitions_freely(self, tmp_path):
        """No audit key (legacy) at current_stage=2_bottleneck_mining → no block."""
        rnd = _full_round(with_audit=False)  # no audit key at all
        state = _state("2_bottleneck_mining", rounds=[rnd])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] != "block", (
            f"Legacy campaign without audit key should not be blocked. Got: {result!r}"
        )


class TestNewRoundAuditGate:
    def test_blocks_new_round_start_without_stage_7_audit(self, tmp_path):
        """round 2 at 1_baseline, rounds[0].audit:{} (no stage_7.passed_at) → block."""
        r1 = _full_round(round_id=1, with_audit=True, audit_fields={})
        r1["status"] = "completed"
        r1["integration"]["status"] = "completed"
        r1["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        r1["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        r1["shipped"] = ["op-w"]
        r2 = _full_round(round_id=2, with_audit=True, audit_fields={})
        r2["baseline"]["started_at"] = "2026-05-06T09:00:00Z"
        r2["baseline"]["completed_at"] = None
        state = _state("1_baseline", current_round=2, rounds=[r1, r2])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] == "block", (
            f"Expected block on round 2 start without round 1 stage_7 audit. Got: {result!r}"
        )
        assert "stage_7" in result["reason"].lower() or "audit" in result["reason"].lower()

    def test_allows_new_round_start_with_stage_7_audit(self, tmp_path):
        """round 2 at 1_baseline, rounds[0].audit.stage_7.passed_at set → no block."""
        r1 = _full_round(
            round_id=1, with_audit=True,
            audit_fields={"stage_7": {"passed_at": "2026-05-05T14:00:00Z",
                                      "verdict_file": "audits/s7.md"}},
        )
        r1["status"] = "completed"
        r1["shipped"] = ["op-w"]
        r1["integration"]["status"] = "completed"
        r1["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        r1["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        r2 = _full_round(round_id=2, with_audit=True, audit_fields={})
        r2["baseline"]["started_at"] = "2026-05-06T09:00:00Z"
        r2["baseline"]["completed_at"] = None
        state = _state("1_baseline", current_round=2, rounds=[r1, r2])
        state_file = _stage_project(tmp_path, state)

        result = _run_hook(state_file)
        assert result["decision"] != "block", (
            f"Unexpected block when round 1 stage_7 audit is set. Got: {result!r}"
        )


class TestNewCampaignBootstrap:
    def test_new_campaign_state_triggers_audit_gates(self, tmp_path):
        """state.json from new_target.py has audit:{} → gate active, S1→S2 blocks."""
        # Run new_target.py to scaffold a campaign state.json
        artifact_dir = tmp_path / "kernel_opt_artifacts" / "auto_newtgt"
        subprocess.run(
            [sys.executable, str(NEW_TARGET_SCRIPT),
             "--artifact-dir", str(artifact_dir),
             "--model-id", "TestModel", "--hardware", "H100",
             "--dtype", "bf16", "--tp", "1", "--ep", "1"],
            check=True,
            capture_output=True,
        )
        # Verify audit:{} seeded
        state_file = artifact_dir / "state.json"
        state = json.loads(state_file.read_text())
        assert "audit" in state["campaign"]["rounds"][0], (
            "new_target.py must seed audit:{} in rounds[0]"
        )
        # Set baseline.completed_at and advance to 2_bottleneck_mining
        state["campaign"]["current_stage"] = "2_bottleneck_mining"
        state["campaign"]["rounds"][0]["baseline"]["started_at"] = "2026-05-05T10:00:00Z"
        state["campaign"]["rounds"][0]["baseline"]["completed_at"] = "2026-05-05T10:30:00Z"
        state_file.write_text(json.dumps(state, indent=2))

        # Stage schema alongside the project for the walk-up resolution
        schema_dir = tmp_path / ".claude" / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "state.schema.json").write_text(SCHEMA.read_text())

        result = _run_hook(state_file)
        assert result["decision"] == "block", (
            f"new_target.py-bootstrapped campaign at 2_bottleneck_mining without "
            f"stage_1 audit should be blocked. Got: {result!r}"
        )
