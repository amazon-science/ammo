# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for ammo-next-step-reminder.sh audit-gate reminders (Task 1 / RED).

These tests should FAIL until Task 3 lands the audit-reminder logic in the
hook. Each test constructs a tmp project dir with a state.json, invokes the
bash hook via subprocess, and asserts on the emitted additionalContext.

Legacy-gate semantics: the reminder fires ONLY when the `audit` key exists
(even as `{}`) in the current round. Absent `audit` key ⇒ legacy campaign ⇒
normal reminder (no "AUDIT REQUIRED").
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).parent.parent.parent
HOOK = ROOT / "ai_cli_session" / ".claude" / "hooks" / "ammo-next-step-reminder.sh"


def _base_round(round_id: int = 1) -> Dict[str, Any]:
    return {
        "round_id": round_id,
        "status": "IN_PROGRESS",
        "team_name": None,
        "profiling_baseline_path": None,
        "baseline": {"started_at": None, "completed_at": None},
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


def _base_state(current_stage: str, rounds: list[Dict[str, Any]] | None = None,
                current_round: int = 1) -> Dict[str, Any]:
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
            "rounds": rounds if rounds is not None else [_base_round()],
        },
    }


def _write_project(tmp_path: Path, state: Dict[str, Any]) -> Path:
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "auto_target"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_file = artifact_dir / "state.json"
    state_file.write_text(json.dumps(state, indent=2))
    return state_file


def _run_hook(tmp_path: Path, payload_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the reminder hook and parse emitted JSON.

    Returns the additionalContext string (may be empty) in key 'ctx', plus
    stdout/stderr/returncode for debugging.
    """
    sid = f"test-{uuid.uuid4().hex[:8]}"
    cfg_dir = tmp_path / "claude_cfg"
    cfg_dir.mkdir(exist_ok=True)
    payload = {
        "session_id": sid,
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(tmp_path / "kernel_opt_artifacts" / "auto_target" / "state.json")},
        "cwd": str(tmp_path),
    }
    if payload_extra:
        payload.update(payload_extra)

    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(tmp_path)
    env["CLAUDE_SESSION_ID"] = sid
    env["CLAUDE_CONFIG_DIR"] = str(cfg_dir)
    env["CLAUDE_SUBAGENT"] = "0"

    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    ctx = ""
    if proc.stdout.strip():
        try:
            out = json.loads(proc.stdout)
            ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
        except json.JSONDecodeError:
            ctx = proc.stdout
    return {"ctx": ctx, "stdout": proc.stdout, "stderr": proc.stderr, "rc": proc.returncode}


# ───────────── Tests ─────────────


class TestAuditReminderStage1:
    def test_audit_reminder_emitted_when_stage_1_done_and_audit_missing(self, tmp_path):
        """baseline done, audit:{} present (no stage_1.passed_at) → AUDIT REQUIRED."""
        rnd = _base_round()
        rnd["baseline"]["started_at"] = "2026-05-05T10:00:00Z"
        rnd["baseline"]["completed_at"] = "2026-05-05T10:30:00Z"
        rnd["audit"] = {}
        state = _base_state("1_baseline", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" in result["ctx"], (
            f"Expected AUDIT REQUIRED (stage_1) in context. Got: {result!r}"
        )

    def test_normal_reminder_when_stage_1_audit_passed(self, tmp_path):
        """baseline done + audit.stage_1.passed_at set → normal reminder."""
        rnd = _base_round()
        rnd["baseline"]["started_at"] = "2026-05-05T10:00:00Z"
        rnd["baseline"]["completed_at"] = "2026-05-05T10:30:00Z"
        rnd["audit"] = {"stage_1": {"passed_at": "2026-05-05T10:45:00Z",
                                    "verdict_file": "audits/s1.md"}}
        state = _base_state("1_baseline", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" not in result["ctx"], (
            f"Unexpected AUDIT REQUIRED. ctx={result['ctx']!r}"
        )
        # Normal post-baseline reminder should fire.
        assert "2_bottleneck_mining" in result["ctx"] or "mining" in result["ctx"].lower()


class TestAuditReminderStage45:
    def test_audit_reminder_at_stage_45_transition(self, tmp_path):
        """All tracks terminal, audit:{} (no stage_45.passed_at) → AUDIT REQUIRED."""
        rnd = _base_round()
        rnd["parallel_tracks"]["started_at"] = "2026-05-05T11:00:00Z"
        rnd["parallel_tracks"]["tracks"] = {
            "op-a": {"status": "PASS"},
            "op-b": {"status": "GATED_PASS"},
            "op-c": {"status": "FAIL"},
        }
        rnd["audit"] = {}
        state = _base_state("4_5_parallel_tracks", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" in result["ctx"], (
            f"Expected AUDIT REQUIRED (stage_45). Got: {result!r}"
        )

    def test_normal_reminder_when_stage_45_audit_passed(self, tmp_path):
        """All tracks terminal + audit.stage_45.passed_at set → normal reminder."""
        rnd = _base_round()
        rnd["parallel_tracks"]["started_at"] = "2026-05-05T11:00:00Z"
        rnd["parallel_tracks"]["tracks"] = {
            "op-a": {"status": "PASS"},
            "op-b": {"status": "FAIL"},
        }
        rnd["audit"] = {"stage_45": {"passed_at": "2026-05-05T12:00:00Z",
                                     "verdict_file": "audits/s45.md"}}
        state = _base_state("4_5_parallel_tracks", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" not in result["ctx"], (
            f"Unexpected AUDIT REQUIRED. ctx={result['ctx']!r}"
        )


class TestAuditReminderStage6:
    def test_audit_reminder_at_stage_6_ship(self, tmp_path):
        """SHIPPED_COUNT>0, audit:{} (no stage_6.passed_at) → AUDIT REQUIRED."""
        rnd = _base_round()
        rnd["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        rnd["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        rnd["integration"]["status"] = "completed"
        rnd["shipped"] = ["op-winner"]
        rnd["audit"] = {}
        state = _base_state("6_integration", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" in result["ctx"], (
            f"Expected AUDIT REQUIRED (stage_6). Got: {result!r}"
        )

    def test_normal_reminder_when_stage_6_audit_passed(self, tmp_path):
        """SHIPPED + audit.stage_6.passed_at set → normal reminder."""
        rnd = _base_round()
        rnd["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        rnd["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        rnd["integration"]["status"] = "completed"
        rnd["shipped"] = ["op-winner"]
        rnd["audit"] = {"stage_6": {"passed_at": "2026-05-05T12:45:00Z",
                                    "verdict_file": "audits/s6.md"}}
        state = _base_state("6_integration", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" not in result["ctx"], (
            f"Unexpected AUDIT REQUIRED. ctx={result['ctx']!r}"
        )

    def test_audit_reminder_at_stage_6_exhausted(self, tmp_path):
        """shipped=[], integration.status=exhausted, audit:{} → AUDIT REQUIRED.

        Stage 6 audit fires on any terminal integration status, not just SHIP.
        """
        rnd = _base_round()
        rnd["integration"]["started_at"] = "2026-05-05T12:00:00Z"
        rnd["integration"]["completed_at"] = "2026-05-05T12:30:00Z"
        rnd["integration"]["status"] = "exhausted"
        rnd["shipped"] = []
        rnd["audit"] = {}
        state = _base_state("6_integration", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" in result["ctx"], (
            f"Expected AUDIT REQUIRED (stage_6, exhausted). Got: {result!r}"
        )


class TestAuditReminderStage7:
    def test_audit_reminder_at_stage_7(self, tmp_path):
        """current_stage 7_campaign_eval, audit:{} (no stage_7.passed_at) → AUDIT REQUIRED."""
        rnd = _base_round()
        rnd["campaign_eval"]["started_at"] = "2026-05-05T13:00:00Z"
        rnd["audit"] = {}
        state = _base_state("7_campaign_eval", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" in result["ctx"], (
            f"Expected AUDIT REQUIRED (stage_7). Got: {result!r}"
        )


class TestLegacyCampaignNoAuditKey:
    def test_legacy_campaign_no_audit_key_gets_normal_reminder(self, tmp_path):
        """No audit key (legacy campaign): emit normal next-step reminder.

        State at 1_baseline with baseline.completed_at set — would trigger
        audit check if audit key were present — but omit the key entirely.
        Expect normal "mining" reminder, NOT AUDIT REQUIRED.
        """
        rnd = _base_round()
        rnd["baseline"]["started_at"] = "2026-05-05T10:00:00Z"
        rnd["baseline"]["completed_at"] = "2026-05-05T10:30:00Z"
        # NO audit key — legacy campaign
        state = _base_state("1_baseline", rounds=[rnd])
        _write_project(tmp_path, state)

        result = _run_hook(tmp_path)
        assert "AUDIT REQUIRED" not in result["ctx"], (
            f"Legacy campaign should not emit AUDIT REQUIRED. ctx={result['ctx']!r}"
        )
        # Normal post-baseline reminder should fire.
        assert result["ctx"], f"Expected a normal reminder. stdout={result['stdout']!r}"
