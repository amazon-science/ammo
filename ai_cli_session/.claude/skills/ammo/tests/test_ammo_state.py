#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the AMMO state engine (scripts/ammo_state.py).

Covers: atomic set round-trip + schema-invalid rejection; validate-verb parity
with the bash harness on representative cases; next-step ladder per stage
(substring parity with the 27-test reminder harness); first-fire = no socratic;
edge fires with prev; advance
SHIP / EXHAUSTED(+mining_invalidated); enrich --mining table parse; backfill.

tmp_path fixtures shared across the state-engine suites.

Run: python3 -m pytest tests/test_ammo_state.py -q
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_ENGINE = _SCRIPTS_DIR / "ammo_state.py"
_SCHEMA = Path(__file__).resolve().parents[3] / "schemas" / "state.schema.json"

sys.path.insert(0, str(_SCRIPTS_DIR))
import ammo_state  # noqa: E402


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _base_state():
    """Canonical valid v2/v4.1 state — mirrors the bash harness base_v2 + new_target shape."""
    return {
        "target": {
            "model_id": "test-model", "hardware": "H100", "dtype": "bf16",
            "tp": 1, "dp": 1, "ep": 1, "component": "auto",
        },
        "session_id": None,
        "gpu_resources": {
            "gpu_count": 1, "gpu_model": "NVIDIA H100",
            "memory_total_gib": 80.0, "cuda_visible_devices": "0",
        },
        "campaign": {
            "schema_version": "4.1",
            "status": "active",
            "current_round": 1,
            "current_stage": "1_baseline",
            "config": {
                "min_e2e_improvement_pct": 1.0,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
            },
            "cumulative_speedup_vs_round1": 1.0,
            "round_1_baseline_latency_s": None,
            "shipped_optimizations": [],
            "agent_costs": [],
            "rounds": [_round_skel(1)],
        },
    }


def _round_skel(round_id, max_rounds=4):
    return {
        "round_id": round_id, "status": "IN_PROGRESS", "team_name": None,
        "profiling_baseline_path": None,
        "baseline": {"started_at": None, "completed_at": None, "e2e_latency": None, "per_bs_verdict": None},
        "bottleneck_mining": {"started_at": None, "completed_at": None, "top_bottleneck_share_pct": None},
        "debate": {"started_at": None, "completed_at": None, "candidates": [],
                   "rounds_completed": 0, "max_rounds": max_rounds, "selected_winners": []},
        "parallel_tracks": {"started_at": None, "completed_at": None, "tracks": {}},
        "integration": {"started_at": None, "completed_at": None, "status": "pending",
                        "passing_candidates": [], "failed_candidates": [], "selected_candidates": [],
                        "conflict_analysis": None, "combined_patch_branch": None, "combined_e2e_result": None,
                        "e2e_latency_combined": None, "per_bs_verdict": None, "commit_sha": None,
                        "final_decision": None, "resolver_invoked": None, "resolver_outcome": None,
                        "conflicting_tracks": None},
        "campaign_eval": {"started_at": None, "completed_at": None},
        "audit": {},
        "shipped": [], "dropped": [],
        "cumulative_speedup_after": None, "combined_e2e_speedup_x": None,
        "combined_e2e_delta_pp": None, "note": None, "round_summary": None,
    }


def _write_state(tmp_path, state):
    """Write a state.json inside a .claude/schemas-walkable artifact dir."""
    root = tmp_path
    (root / ".git").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "schemas").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "schemas" / "state.schema.json").write_text(
        _SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    art = root / "kernel_opt_artifacts" / "t"
    art.mkdir(parents=True, exist_ok=True)
    sp = art / "state.json"
    sp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return sp


def _run(*argv):
    return subprocess.run([sys.executable, str(_ENGINE), *argv],
                          capture_output=True, text=True)


# ─────────────────────────────────────────────
# get / set
# ─────────────────────────────────────────────

class TestGetSet:
    def test_get_whole_doc(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("get", str(sp))
        assert r.returncode == 0
        assert json.loads(r.stdout)["campaign"]["current_stage"] == "1_baseline"

    def test_get_field_and_artifact_dir(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("get", str(sp), "--field", "campaign.current_stage")
        assert r.returncode == 0 and r.stdout.strip() == "1_baseline"
        # accept artifact dir form
        r2 = _run("get", str(sp.parent), "--field", "campaign.current_round")
        assert r2.returncode == 0 and r2.stdout.strip() == "1"

    def test_get_missing_state_exit1(self, tmp_path):
        r = _run("get", str(tmp_path / "nope"), "--field", "campaign.status")
        assert r.returncode == 1 and "FAIL" in r.stderr

    def test_set_round_trip_atomic(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("set", "--state", str(sp),
                 "--field", "campaign.current_stage", "--value", '"2_bottleneck_mining"',
                 "--field", "campaign.rounds.0.baseline.completed_at", "--value", '"2026-01-01T00:00:00Z"',
                 "--field", "campaign.rounds.0.audit.stage_1.passed_at", "--value", '"2026-01-01T00:01:00Z"')
        assert r.returncode == 0, r.stderr
        doc = json.loads(sp.read_text())
        assert doc["campaign"]["current_stage"] == "2_bottleneck_mining"
        assert doc["campaign"]["rounds"][0]["baseline"]["completed_at"] == "2026-01-01T00:00:00Z"

    def test_set_rejects_schema_invalid_leaves_untouched(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        before = sp.read_text()
        r = _run("set", "--state", str(sp),
                 "--field", "campaign.current_stage", "--value", '"campaign_complete"')
        assert r.returncode == 1
        assert "is not one of" in r.stdout
        assert sp.read_text() == before  # untouched

    def test_set_rejects_cross_field_violation_leaves_untouched(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op-001": {"status": "FAIL", "diluted": False}
        }
        sp = _write_state(tmp_path, st)
        before = sp.read_text()

        r = _run(
            "set",
            "--state",
            str(sp),
            "--field",
            "campaign.rounds.0.parallel_tracks.tracks.op-001.diluted",
            "--value",
            "true",
        )

        assert r.returncode == 1
        assert "diluted=true with status != PASS" in r.stdout or "'PASS' was expected" in r.stdout
        assert sp.read_text() == before


# ─────────────────────────────────────────────
# validate — parity with bash harness representative cases
# ─────────────────────────────────────────────

class TestValidate:
    def test_valid_passes(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0 and "PASS" in r.stdout

    def test_bad_enum_blocks(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "nonsense_stage"
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1 and "is not one of" in r.stdout

    def test_stage6_non_terminal_track_blocks(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "6_integration"
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op-001": {"status": "PASS"}, "op-002": {"status": "IN_PROGRESS"}}
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1 and "Stage 6 transition blocked" in r.stdout

    def test_audit_gate_blocks(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "2_bottleneck_mining"
        st["campaign"]["rounds"][0]["audit"] = {}
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1 and "stage_1" in r.stdout

    def test_post_ship_remine_exemption_allows(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_round"] = 2
        st["campaign"]["current_stage"] = "2_bottleneck_mining"
        st["campaign"]["rounds"][0]["status"] = "completed"
        st["campaign"]["rounds"][0]["audit"] = {"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}
        r2 = _round_skel(2, max_rounds=3)
        r2["audit"] = {"stage_2": {"passed_at": "2026-06-09T15:35:00Z"}}
        st["campaign"]["rounds"].append(r2)
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_post_ship_remine_fail_closed(self, tmp_path):
        # AG9: round 1 omits audit key entirely; round 2 audit={} -> stage_1 still required -> BLOCK
        st = _base_state()
        st["campaign"]["current_round"] = 2
        st["campaign"]["current_stage"] = "2_bottleneck_mining"
        st["campaign"]["rounds"][0]["status"] = "completed"
        st["campaign"]["rounds"][0].pop("audit")
        r2 = _round_skel(2, max_rounds=3)
        r2["audit"] = {}
        st["campaign"]["rounds"].append(r2)
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1 and "stage_1" in r.stdout

    def test_new_round_start_gate_blocks(self, tmp_path):
        # AG6: prev round has audit key but no stage_67 -> new-round-start gate blocks
        st = _base_state()
        st["campaign"]["current_round"] = 2
        st["campaign"]["current_stage"] = "2_bottleneck_mining"
        st["campaign"]["rounds"][0]["status"] = "completed"
        st["campaign"]["rounds"][0]["audit"] = {}
        r2 = _round_skel(2, max_rounds=3)
        r2["audit"] = {"stage_2": {"passed_at": "2026-06-09T15:35:00Z"}}
        st["campaign"]["rounds"].append(r2)
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1 and "new round start blocked" in r.stdout

    def test_emit_hook_block_json(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "campaign_complete"
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp), "--emit", "hook")
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["decision"] == "block"
        assert "additionalContext" in out["hookSpecificOutput"]

    def test_emit_hook_pass_is_silent(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("validate", "--state", str(sp), "--emit", "hook")
        assert r.returncode == 0 and r.stdout.strip() == ""

    def test_stage2_gate_only_v41(self, tmp_path):
        # v4.0: stage_2 audit not enforced for round 2 debate
        st = _base_state()
        st["campaign"]["schema_version"] = "4.0"
        st["campaign"]["current_round"] = 2
        st["campaign"]["current_stage"] = "3_debate"
        st["campaign"]["rounds"][0]["status"] = "completed"
        st["campaign"]["rounds"][0]["audit"] = {"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}
        r2 = _round_skel(2, max_rounds=3)
        r2["audit"] = {}
        st["campaign"]["rounds"].append(r2)
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_diluted_true_requires_pass_status(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op-001": {"status": "FAIL", "diluted": True}
        }
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1
        assert "diluted=true with status != PASS" in r.stdout or "'PASS' was expected" in r.stdout

    def test_diluted_true_with_pass_status_is_valid(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op-001": {"status": "PASS", "diluted": True}
        }
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_historical_round_diluted_invariant_is_not_ignored(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op-old": {"status": "FAIL", "diluted": True}
        }
        st["campaign"]["rounds"].append(_round_skel(2))
        st["campaign"]["current_round"] = 2
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1
        assert "diluted=true with status != PASS" in r.stdout or "'PASS' was expected" in r.stdout


# ─────────────────────────────────────────────
# mining-enrichment round-advance gate (replaces the schema allOf hard-block)
# ─────────────────────────────────────────────

_MINING_ENRICHED = {
    "started_at": "2026-01-01T00:10:00Z", "completed_at": "2026-01-01T00:40:00Z",
    "top_bottleneck_share_pct": 30.0, "top_component": "fused_moe",
    "top_f_decode_pct": 45.0, "top_f_e2e_pct": 30.0,
    "top_addressable_e2e_pct": 12.0,
    "amdahl_ceiling": 30.0, "decode_frac": 0.78,
    "component_breakdown": [{"name": "fused_moe", "pct": 30.0}],
}
_MINING_UNENRICHED = {
    "started_at": "2026-01-01T00:10:00Z", "completed_at": "2026-01-01T00:40:00Z",
    "top_bottleneck_share_pct": 30.0,
}


class TestMiningEnrichmentRoundAdvanceGate:
    def _state(self, stage, mining):
        st = _base_state()
        st["campaign"]["current_stage"] = stage
        st["campaign"]["rounds"][0]["bottleneck_mining"] = dict(mining)
        if stage == "2_bottleneck_mining":
            st["campaign"]["rounds"][0]["audit"] = {
                "stage_1": {"passed_at": "2026-01-01T00:00:00Z"}
            }
        elif stage == "3_debate":
            st["campaign"]["rounds"][0]["audit"] = {
                "stage_2": {"passed_at": "2026-01-01T00:00:00Z"}
            }
        return st

    def test_unenriched_at_mining_stage_allows(self, tmp_path):
        # THE deadlock fix: this exact write was rejected by the old schema allOf.
        sp = _write_state(tmp_path, self._state("2_bottleneck_mining", _MINING_UNENRICHED))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_unenriched_round_advance_blocks(self, tmp_path):
        sp = _write_state(tmp_path, self._state("3_debate", _MINING_UNENRICHED))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1
        assert "Round-advance blocked: bottleneck_mining enrichment incomplete" in r.stdout
        assert "backfill" in r.stdout

    def test_enriched_round_advance_passes(self, tmp_path):
        sp = _write_state(tmp_path, self._state("3_debate", _MINING_ENRICHED))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_null_field_named_in_reason(self, tmp_path):
        mining = dict(_MINING_ENRICHED)
        mining["decode_frac"] = None
        sp = _write_state(tmp_path, self._state("3_debate", mining))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 1
        assert "decode_frac" in r.stdout

    def test_missing_addressable_impact_named_in_reason(self, tmp_path):
        mining = dict(_MINING_ENRICHED)
        mining["top_addressable_e2e_pct"] = None
        sp = _write_state(tmp_path, self._state("3_debate", mining))

        result = _run("validate", "--state", str(sp))

        assert result.returncode == 1
        assert "top_addressable_e2e_pct" in result.stdout

    def test_incomplete_mining_past_stage_allows(self, tmp_path):
        # EXHAUSTED round re-entering 3_debate without a re-mine: no completed_at, no gate.
        sp = _write_state(tmp_path, self._state("3_debate", {
            "started_at": None, "completed_at": None, "top_bottleneck_share_pct": None}))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout

    def test_later_write_not_deadlocked_by_unenriched_historical_round(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["status"] = "completed"
        st["campaign"]["rounds"][0]["bottleneck_mining"] = dict(_MINING_UNENRICHED)
        st["campaign"]["rounds"].append(_round_skel(2))
        st["campaign"]["rounds"][0]["audit"] = {
            "stage_67": {"passed_at": "2026-01-01T00:00:00Z"}
        }
        st["campaign"]["current_round"] = 2
        st["campaign"]["current_stage"] = "1_baseline"
        sp = _write_state(tmp_path, st)

        r = _run(
            "set",
            "--state",
            str(sp),
            "--field",
            "campaign.rounds.1.note",
            "--value",
            '"current round write"',
        )

        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(sp.read_text())["campaign"]["rounds"][1]["note"] == "current round write"

    def test_legacy_misnamed_f_e2e_field_remains_accepted_on_resume(self, tmp_path):
        mining = dict(_MINING_ENRICHED)
        mining.pop("top_f_e2e_pct")
        mining["top_f_decode_pct"] = 30.0
        sp = _write_state(tmp_path, self._state("3_debate", mining))

        result = _run("validate", "--state", str(sp))

        assert result.returncode == 0, result.stdout + result.stderr


class TestStage2AuditTransitionGate:
    def test_stage3_requires_stage2_audit(self, tmp_path):
        state = _base_state()
        state["campaign"]["current_stage"] = "3_debate"
        state_path = _write_state(tmp_path, state)
        result = _run("validate", "--state", str(state_path))
        assert result.returncode == 1
        assert "stage_2" in result.stdout

    def test_stage2_audit_allows_stage3(self, tmp_path):
        state = _base_state()
        state["campaign"]["current_stage"] = "3_debate"
        state["campaign"]["rounds"][0]["audit"] = {
            "stage_2": {"passed_at": "2026-01-01T00:00:00Z"}
        }
        state_path = _write_state(tmp_path, state)
        result = _run("validate", "--state", str(state_path))
        assert result.returncode == 0, result.stdout + result.stderr


# ─────────────────────────────────────────────
# next-step ladder — substring parity with reminder harness
# ─────────────────────────────────────────────

def _ns(state, prev=None):
    """Call compute_next_step in-process, returning (msg, terminal)."""
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    transitions = ammo_state.load_transitions()
    return ammo_state.compute_next_step(copy.deepcopy(state), prev, transitions, schema)


def _stage(state, stage, **round_over):
    st = copy.deepcopy(state)
    st["campaign"]["current_stage"] = stage
    rnd = st["campaign"]["rounds"][st["campaign"]["current_round"] - 1]
    for k, v in round_over.items():
        # dotted into the round
        ammo_state.set_field(rnd, k, v)
    return st


class TestNextStepLadder:
    def test_baseline_pending(self):
        st = _stage(_base_state(), "1_baseline")
        msg, _ = _ns(st)
        assert "task_type: baseline" in msg

    def test_baseline_done_mining(self):
        st = _stage(_base_state(), "1_baseline", **{
            "baseline.completed_at": "2026-01-01T00:00:00Z",
            "audit.stage_1.passed_at": "2026-01-01T00:01:00Z",
        })
        msg, _ = _ns(st)
        assert "mining" in msg

    def test_mining_pending(self):
        st = _stage(_base_state(), "2_bottleneck_mining",
                    **{"baseline.completed_at": "2026-01-01T00:00:00Z"})
        msg, _ = _ns(st)
        assert "mining" in msg

    def test_mining_done_advance_to_debate(self):
        st = _stage(_base_state(), "2_bottleneck_mining",
                    **{"baseline.completed_at": "2026-01-01T00:00:00Z",
                       "bottleneck_mining.completed_at": "2026-01-01T00:00:00Z",
                       "audit.stage_2.passed_at": "2026-01-01T00:01:00Z"})
        msg, _ = _ns(st)
        assert "Mining done" in msg

    def test_debate_no_team(self):
        st = _stage(_base_state(), "3_debate")
        msg, _ = _ns(st)
        assert "implicit team (no TeamCreate)" in msg

    def test_debate_in_progress(self):
        st = _stage(_base_state(), "3_debate", **{"team_name": "round1-team"})
        msg, _ = _ns(st)
        assert "Min 1 round" in msg

    def test_debate_winners_spawn_impl(self):
        st = _stage(_base_state(), "3_debate",
                    **{"team_name": "round1-team",
                       "debate.selected_candidates": [{"op_id": "op001"}, {"op_id": "op002"}]})
        msg, _ = _ns(st)
        assert "ammo-impl-champion" in msg

    def test_tracks_non_terminal_wait(self):
        st = _stage(_base_state(), "4_5_parallel_tracks",
                    **{"team_name": "t",
                       "parallel_tracks.tracks": {"op001": {"status": "IN_PROGRESS"},
                                                  "op002": {"status": "PASS"}}})
        msg, _ = _ns(st)
        assert "Do NOT advance to Stage 6" in msg

    def test_tracks_all_terminal_shutdown_drain(self):
        st = _stage(_base_state(), "4_5_parallel_tracks",
                    **{"team_name": "t",
                       "audit.stage_45.passed_at": "2026-01-01T00:01:00Z",
                       "parallel_tracks.tracks": {"op001": {"status": "PASS"},
                                                  "op002": {"status": "PASS"}}})
        msg, _ = _ns(st)
        assert "SendMessage shutdown_request" in msg
        assert "shutdown_approved" in msg

    def test_integration_fresh_cache(self):
        st = _stage(_base_state(), "6_integration",
                    **{"parallel_tracks.started_at": "x", "integration.started_at": "x",
                       "parallel_tracks.tracks": {"op001": {"status": "PASS"},
                                                  "op002": {"status": "PASS"}}})
        msg, _ = _ns(st)
        assert "fresh-cache" in msg

    def test_integration_gated_pass_is_ready_for_pre_ship_not_rerun(self):
        st = _stage(
            _base_state(),
            "6_integration",
            **{
                "parallel_tracks.started_at": "x",
                "integration.started_at": "x",
                "integration.status": "gated_pass",
                "parallel_tracks.tracks": {"op001": {"status": "GATED_PASS"}},
            },
        )
        msg, _ = _ns(st)
        assert "Pre-SHIP" in msg
        assert "do not rerun" in msg
        assert "Single passer — run short-circuit" not in msg

    def test_integration_shipped_advances(self):
        # no audit key => legacy bypass => T_AUDIT_S67 passed => advance to 7_campaign_eval
        st = _stage(_base_state(), "6_integration",
                    **{"integration.started_at": "x", "integration.status": "combined",
                       "audit.stage_67.passed_at": "2026-01-01T00:01:00Z",
                       "shipped": ["op001"],
                       "parallel_tracks.tracks": {"op001": {"status": "PASS"}}})
        msg, _ = _ns(st)
        assert "7_campaign_eval" in msg

    def test_campaign_eval_mechanical(self):
        st = _stage(_base_state(), "7_campaign_eval", **{
            "audit.stage_67.passed_at": "2026-01-01T00:01:00Z"
        })
        msg, _ = _ns(st)
        assert "Mechanical check" in msg
        assert "all-diluted SHIP → 3_debate" in msg

    def test_terminal_report_present(self):
        st = _base_state()
        st["campaign"]["current_stage"] = "7_campaign_eval"
        st["campaign"]["status"] = "campaign_complete"
        st["_report_present"] = True
        msg, _ = _ns(st)
        assert "Campaign complete. Session may stop." in msg

    def test_terminal_no_report_spawns_writer(self):
        st = _base_state()
        st["campaign"]["current_stage"] = "7_campaign_eval"
        st["campaign"]["status"] = "campaign_exhausted"
        st["_report_present"] = False
        msg, _ = _ns(st)
        assert "ammo-report-writer" in msg

    def test_audit_required_s1(self):
        st = _stage(_base_state(), "1_baseline",
                    **{"baseline.completed_at": "2026-01-01T00:00:00Z", "audit": {}})
        msg, _ = _ns(st)
        assert "AUDIT REQUIRED" in msg and "stage_1" in msg


# ─────────────────────────────────────────────
# first-fire (no prev) vs edge (with prev)
# ─────────────────────────────────────────────

class TestSocraticEdges:
    def test_malformed_legacy_state_keeps_deliberate_quarter_percent_fallback(self):
        st = _base_state()
        st["campaign"].pop("config")
        msg, _ = _ns(st, prev=None)
        assert "fallback 0.25%" in msg

    def test_first_fire_no_socratic(self):
        # baseline just completed but prev=None => NO socratic, only stage-ladder reminder
        st = _stage(_base_state(), "2_bottleneck_mining",
                    **{"baseline.completed_at": "2026-01-01T00:00:00Z"})
        msg, terminal = _ns(st, prev=None)
        assert "REASON THROUGH THIS" not in msg
        assert terminal is False
        assert msg.startswith("AMMO NEXT STEP:")

    def test_edge_fires_with_prev(self):
        prev = _stage(_base_state(), "1_baseline")  # baseline NOT done
        cur = _stage(_base_state(), "1_baseline",
                     **{"baseline.completed_at": "2026-01-01T00:00:00Z"})
        msg, _ = _ns(cur, prev=prev)
        assert "REASON THROUGH THIS" in msg
        assert "mine bottlenecks from this baseline" in msg

    def test_mining_edge_uses_canonical_f_e2e_not_decode_diagnostic(self):
        prev = _stage(_base_state(), "2_bottleneck_mining")
        cur = _stage(
            _base_state(),
            "2_bottleneck_mining",
            **{
                "bottleneck_mining.completed_at": "2026-01-01T00:00:00Z",
                "bottleneck_mining.top_component": "fused_moe",
                "bottleneck_mining.top_f_e2e_pct": 30.0,
                "bottleneck_mining.top_f_decode_pct": 45.0,
                "bottleneck_mining.amdahl_ceiling": 30.0,
            },
        )

        msg, _ = _ns(cur, prev=prev)

        assert "f_e2e = 30" in msg
        assert "f_e2e = 45" not in msg

    def test_terminal_transition_requires_prev(self):
        prev = _base_state()  # active
        cur = _base_state()
        cur["campaign"]["status"] = "campaign_complete"
        cur["campaign"]["current_stage"] = "7_campaign_eval"
        cur["campaign"]["config"]["min_e2e_improvement_pct"] = 0.5
        mining = cur["campaign"]["rounds"][0]["bottleneck_mining"]
        mining["top_f_e2e_pct"] = 10.0
        mining["top_addressable_e2e_pct"] = 0.49
        msg, terminal = _ns(cur, prev=prev)
        assert terminal is True
        assert "irreversible" in msg
        assert "top_addressable_e2e_pct is 0.49%" in msg
        assert "#2 component" not in msg
        assert "f_e2e = 10" not in msg

    def test_no_terminal_transition_without_prev(self):
        cur = _base_state()
        cur["campaign"]["status"] = "campaign_complete"
        cur["campaign"]["current_stage"] = "7_campaign_eval"
        msg, terminal = _ns(cur, prev=None)
        assert terminal is False


# ─────────────────────────────────────────────
# advance
# ─────────────────────────────────────────────

def _audited_round1_state():
    """Round 1 at 7_campaign_eval with a full audit chain (incl stage_67) so the
    new-round-start gate is satisfied after advance creates round 2."""
    st = _base_state()
    st["campaign"]["current_stage"] = "7_campaign_eval"
    st["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-06-08T14:56:31Z"},
        "stage_2": {"passed_at": "2026-06-08T15:48:49Z"},
        "stage_45": {"passed_at": "2026-06-08T18:26:35Z"},
        "stage_67": {"passed_at": "2026-06-08T18:42:23Z"},
    }
    return st


class TestAdvance:
    def test_ship_new_round_at_mining(self, tmp_path):
        st = _audited_round1_state()
        sp = _write_state(tmp_path, st)
        r = _run("advance", "--state", str(sp), "--outcome", "SHIP")
        assert r.returncode == 0, r.stdout
        doc = json.loads(sp.read_text())
        assert doc["campaign"]["current_round"] == 2
        assert doc["campaign"]["current_stage"] == "2_bottleneck_mining"
        assert doc["campaign"]["rounds"][0]["status"] == "SHIPPED"
        assert doc["campaign"]["rounds"][1]["round_id"] == 2
        # New rounds seed the audit map so all four current audit gates bind.
        assert doc["campaign"]["rounds"][1]["audit"] == {}

    def test_all_diluted_ship_skips_remine_and_goes_directly_to_debate(self, tmp_path):
        st = _audited_round1_state()
        st["campaign"]["rounds"][0]["shipped"] = ["op001", "op002"]
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op001": {"status": "PASS", "diluted": True},
            "op002": {"status": "PASS", "diluted": True},
        }
        sp = _write_state(tmp_path, st)

        r = _run("advance", "--state", str(sp), "--outcome", "SHIP")

        assert r.returncode == 0, r.stdout + r.stderr
        doc = json.loads(sp.read_text())
        assert doc["campaign"]["current_stage"] == "3_debate"
        assert doc["campaign"]["rounds"][0]["status"] == "SHIPPED"

    def test_mixed_diluted_ship_still_remines(self, tmp_path):
        st = _audited_round1_state()
        st["campaign"]["rounds"][0]["shipped"] = ["op001", "op002"]
        st["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "op001": {"status": "PASS", "diluted": True},
            "op002": {"status": "PASS", "diluted": False},
        }
        sp = _write_state(tmp_path, st)

        r = _run("advance", "--state", str(sp), "--outcome", "SHIP")

        assert r.returncode == 0, r.stdout + r.stderr
        assert json.loads(sp.read_text())["campaign"]["current_stage"] == "2_bottleneck_mining"

    def test_exhausted_new_round_at_debate(self, tmp_path):
        st = _audited_round1_state()
        sp = _write_state(tmp_path, st)
        r = _run("advance", "--state", str(sp), "--outcome", "EXHAUSTED")
        assert r.returncode == 0, r.stdout
        doc = json.loads(sp.read_text())
        assert doc["campaign"]["current_stage"] == "3_debate"
        assert doc["campaign"]["rounds"][0]["status"] == "EXHAUSTED"

    def test_exhausted_mining_invalidated_at_mining(self):
        # `mining_invalidated` is read by the engine but is not a schema property
        # (rounds are additionalProperties:false), so exercise the branch in-process.
        st = _base_state()
        st["campaign"]["current_stage"] = "7_campaign_eval"
        st["campaign"]["rounds"][0]["mining_invalidated"] = True
        transitions = ammo_state.load_transitions()
        new_state, new_stage = ammo_state.advance(copy.deepcopy(st), "EXHAUSTED", transitions)
        assert new_stage == "2_bottleneck_mining"
        assert new_state["campaign"]["current_stage"] == "2_bottleneck_mining"
        assert new_state["campaign"]["rounds"][0]["status"] == "EXHAUSTED"

    def test_exhausted_without_invalidation_goes_to_debate(self):
        st = _base_state()
        st["campaign"]["current_stage"] = "7_campaign_eval"
        transitions = ammo_state.load_transitions()
        _, new_stage = ammo_state.advance(copy.deepcopy(st), "EXHAUSTED", transitions)
        assert new_stage == "3_debate"


# ─────────────────────────────────────────────
# enrich --mining
# ─────────────────────────────────────────────

# Verbatim bottleneck_analysis.md from a real campaign (Qwen3.5-4B / L40S).
# It carries the typographic characters the researcher contract emits
# (`f_e2e × (1-1/ceiling)`), so it is the only fixture that binds the parser to
# real prose rather than to a hand-invented shape.
_REAL_CAMPAIGN_MD = (
    Path(__file__).resolve().parent
    / "fixtures" / "mining" / "real_campaign_bottleneck_analysis.md"
)

_MINING_MD = """# Bottleneck Analysis

## Technology Landscape

Some content.

## Workload Dilution

| Component | decode_busy | decode_share_of_e2e | f_e2e | f_e2e x (1-1/ceiling) |
|---|---|---|---|---|
| fused_moe | 0.57 | 0.82 | 0.30 | 0.12 |
| attention_decode | 0.40 | 0.82 | 0.12 | 0.08 |
| rmsnorm | 0.10 | 0.82 | 0.03 | 0.02 |

## Profile Summary

decode_frac = 0.78
"""


class TestEnrichMining:
    def test_mining_table_parse(self, tmp_path):
        st = _base_state()
        # completed_at set so schema requires the enriched fields; enrich fills them
        st["campaign"]["rounds"][0]["bottleneck_mining"]["completed_at"] = "2026-01-01T00:00:00Z"
        st["campaign"]["rounds"][0]["bottleneck_mining"]["top_bottleneck_share_pct"] = 30.0
        sp = _write_state(tmp_path, st)
        md = sp.parent / "mining.md"
        md.write_text(_MINING_MD, encoding="utf-8")
        r = _run("enrich", "--state", str(sp), "--mining", "--from", str(md))
        assert r.returncode == 0, r.stdout + r.stderr
        doc = json.loads(sp.read_text())
        m = doc["campaign"]["rounds"][0]["bottleneck_mining"]
        assert m["top_component"] == "fused_moe"
        assert m["top_f_e2e_pct"] == 30.0  # 0.30 fraction -> 30 percent
        assert m["top_addressable_e2e_pct"] == 12.0
        assert m["top_f_decode_pct"] is None
        assert m["amdahl_ceiling"] == 30.0
        assert m["decode_frac"] == 0.78
        assert isinstance(m["component_breakdown"], list)
        assert m["component_breakdown"][0]["name"] == "fused_moe"
        assert m["component_breakdown"][0]["pct"] == 30.0

    def test_mining_malformed_nulls_rest(self):
        parsed = ammo_state.parse_mining_md("garbage with no table")
        assert parsed["top_component"] is None
        assert parsed["top_addressable_e2e_pct"] is None
        assert parsed["component_breakdown"] is None
        assert parsed["decode_frac"] is None


# The EXACT two-table format ammo-researcher.md mandates: typographic x/minus in
# the addressable column header, decode_share_of_e2e as a per-BS column, and NO
# `decode_frac` scalar anywhere. The parser must satisfy the round-advance gate
# from this text alone.
_MINING_MD_CONTRACT = """# Bottleneck Analysis (Stage 2, Round 1)

## Workload Dilution (per BS)

| BS | total_e2e_s | prefill_s | decode_wall_s | decode_kernel_s | decode_busy | decode_share_of_e2e | inter_kernel_share | prefill_share |
|---|---|---|---|---|---|---|---|---|
| 32 | 10.0931 | 0.1093 | 9.9189 | 9.8826 | 0.9963 | 0.9891 | 0.00362 | 0.0109 |

## Top Components (by f_e2e)

| Component | BS | decode-graph % | f_e2e | physical_ceiling | f_e2e × (1−1/ceiling) | prefill-active? |
|---|---|---|---|---|---|---|
| dense_proj_GEMM | 32 | 69.19 | 0.6818 | 1.619x | 0.2607 | Yes |
| GDN_recurrent | 32 | 13.30 | 0.1310 | (disclosed) | (disclosed) | No |
"""


class TestMiningContractFormat:
    """The mandated researcher format must satisfy the gate with no hand edits."""

    def test_contract_table_parses_every_gate_field(self):
        parsed = ammo_state.parse_mining_md(_MINING_MD_CONTRACT)
        assert parsed["top_component"] == "dense_proj_GEMM"
        assert parsed["top_f_e2e_pct"] == pytest.approx(68.18)
        assert parsed["top_bottleneck_share_pct"] == pytest.approx(68.18)
        assert parsed["amdahl_ceiling"] == pytest.approx(68.18)
        # Unicode x/minus in the header must still resolve the addressable column.
        assert parsed["top_addressable_e2e_pct"] == pytest.approx(26.07)
        # No decode_frac scalar in the source: read the per-BS column instead.
        assert parsed["decode_frac"] == pytest.approx(0.99)
        assert parsed["component_breakdown"][0]["name"] == "dense_proj_GEMM"

    def test_contract_table_clears_round_advance_gate(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "2_bottleneck_mining"
        st["campaign"]["rounds"][0]["baseline"]["completed_at"] = "2026-01-01T00:00:00Z"
        st["campaign"]["rounds"][0]["bottleneck_mining"]["completed_at"] = "2026-01-01T00:00:00Z"
        st["campaign"]["rounds"][0]["audit"] = {
            "stage_1": {"passed_at": "2026-01-01T00:00:00Z"},
            "stage_2": {"passed_at": "2026-01-01T00:01:00Z"},
        }
        sp = _write_state(tmp_path, st)
        md = sp.parent / "bottleneck_analysis.md"
        md.write_text(_MINING_MD_CONTRACT, encoding="utf-8")
        assert _run("enrich", "--state", str(sp), "--mining", "--from", str(md)).returncode == 0
        assert _run("set", "--state", str(sp), "--field", "campaign.current_stage",
                    "--value", '"3_debate"').returncode == 0

    def test_real_campaign_doc_parses(self):
        """Golden fixture: the verbatim doc a real campaign produced. Committed
        in-repo so the parser stays bound to real prose, not to a hand shape."""
        parsed = ammo_state.parse_mining_md(
            _REAL_CAMPAIGN_MD.read_text(encoding="utf-8")
        )
        for field in ("top_component", "top_f_e2e_pct", "top_bottleneck_share_pct",
                      "top_addressable_e2e_pct", "amdahl_ceiling", "decode_frac",
                      "component_breakdown"):
            assert parsed[field] is not None, field
        assert parsed["top_component"].startswith("dense_proj_GEMM")
        assert parsed["top_f_e2e_pct"] == pytest.approx(68.18)
        assert parsed["top_addressable_e2e_pct"] == pytest.approx(26.07)
        assert parsed["decode_frac"] == pytest.approx(0.9891)

    @pytest.mark.parametrize("dash", ["-", "−", "–", "—", " - "])
    def test_real_campaign_doc_parses_every_dash_style(self, dash):
        """The addressable column header carries a typographic dash whose style
        no contract pins. Every style must resolve the same column."""
        text = _REAL_CAMPAIGN_MD.read_text(encoding="utf-8").replace(
            "(1-1/ceiling)", "(1%s1/ceiling)" % dash
        )
        parsed = ammo_state.parse_mining_md(text)
        assert parsed["top_addressable_e2e_pct"] == pytest.approx(26.07)

    @pytest.mark.parametrize("times", ["x", "×", "✕", "✖"])
    def test_real_campaign_doc_parses_every_times_style(self, times):
        """Same for the multiplication sign in `f_e2e x (1-1/ceiling)`."""
        text = _REAL_CAMPAIGN_MD.read_text(encoding="utf-8").replace(
            "f_e2e × (1-1/ceiling)", "f_e2e %s (1-1/ceiling)" % times
        )
        parsed = ammo_state.parse_mining_md(text)
        assert parsed["top_addressable_e2e_pct"] == pytest.approx(26.07)


_MINED_JSON = {
    "schema": "mine_trace/1",
    "provenance": {"nsys_product_version": "2025.1", "traces": ["baseline_bs32.sqlite"]},
    "per_bs": [
        {
            "bs": 32,
            "decode_avg_s": 9.9189,
            "avg_s": 10.0931,
            "decode_share_of_e2e": 0.9891,
            "decode_busy": 0.9963,
            "families": [
                {"label": "dense_proj_GEMM", "f_decode": 0.6919, "f_e2e": 0.6818,
                 "busy_ns": 5898000, "physical_ceiling": 1.619,
                 "addressable_e2e": 0.2607},
                {"label": "GDN_recurrent", "f_decode": 0.1330, "f_e2e": 0.1310,
                 "busy_ns": 1130000, "physical_ceiling": None,
                 "addressable_e2e": None},
            ],
            "residual_pct": 0.4,
            "partition_coverage": 0.996,
            "future_field_tools_may_add": "tolerated",
        }
    ],
    "warnings": [],
}


class TestEnrichMiningFromMinedJson:
    def _write_mined(self, sp, doc, round_id=1):
        mining_dir = sp.parent / "rounds" / str(round_id) / "mining"
        mining_dir.mkdir(parents=True, exist_ok=True)
        (mining_dir / "mined.json").write_text(json.dumps(doc), encoding="utf-8")
        return mining_dir

    def test_mined_json_is_preferred_over_markdown(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["bottleneck_mining"]["completed_at"] = "2026-01-01T00:00:00Z"
        sp = _write_state(tmp_path, st)
        mining_dir = self._write_mined(sp, _MINED_JSON)
        md = mining_dir / "bottleneck_analysis.md"
        md.write_text(_MINING_MD, encoding="utf-8")  # names fused_moe, decode_frac 0.78

        r = _run("enrich", "--state", str(sp), "--mining", "--from", str(md))

        assert r.returncode == 0, r.stdout + r.stderr
        m = json.loads(sp.read_text())["campaign"]["rounds"][0]["bottleneck_mining"]
        assert m["top_component"] == "dense_proj_GEMM"     # from mined.json
        assert m["decode_frac"] == pytest.approx(0.99)     # from mined.json
        assert m["top_f_e2e_pct"] == pytest.approx(68.18)
        assert m["top_bottleneck_share_pct"] == pytest.approx(68.18)
        # From families[].addressable_e2e, not the markdown's 12.0.
        assert m["top_addressable_e2e_pct"] == pytest.approx(26.07)

    def test_mined_json_alone_clears_every_gate_field(self):
        parsed = ammo_state.parse_mined_json(_MINED_JSON)
        for field in ("top_component", "top_f_e2e_pct", "top_bottleneck_share_pct",
                      "top_addressable_e2e_pct", "amdahl_ceiling", "decode_frac",
                      "component_breakdown"):
            assert parsed[field] is not None, field

    def test_markdown_supplies_addressable_when_no_ceiling_disclosed(self):
        """A family with no physical_ceiling has addressable_e2e null; the
        markdown table's final column then remains the only source."""
        doc = json.loads(json.dumps(_MINED_JSON))
        for fam in doc["per_bs"][0]["families"]:
            fam["addressable_e2e"] = None
            fam["physical_ceiling"] = None
        merged = ammo_state.resolve_mining_fields(doc, _MINING_MD)
        assert merged["top_component"] == "dense_proj_GEMM"    # mined.json
        assert merged["top_addressable_e2e_pct"] == pytest.approx(12.0)  # markdown

    def test_malformed_mined_json_falls_back_to_markdown(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"][0]["bottleneck_mining"]["completed_at"] = "2026-01-01T00:00:00Z"
        sp = _write_state(tmp_path, st)
        mining_dir = sp.parent / "rounds" / "1" / "mining"
        mining_dir.mkdir(parents=True, exist_ok=True)
        (mining_dir / "mined.json").write_text("{not json", encoding="utf-8")
        md = mining_dir / "bottleneck_analysis.md"
        md.write_text(_MINING_MD, encoding="utf-8")

        r = _run("enrich", "--state", str(sp), "--mining", "--from", str(md))

        assert r.returncode == 0, r.stdout + r.stderr
        m = json.loads(sp.read_text())["campaign"]["rounds"][0]["bottleneck_mining"]
        assert m["top_component"] == "fused_moe"
        assert m["decode_frac"] == 0.78

    def test_mined_json_missing_per_bs_yields_nothing(self):
        parsed = ammo_state.parse_mined_json({"schema": "mine_trace/1"})
        assert all(v is None for v in parsed.values())

    def test_enrich_does_not_null_out_recorded_values(self, tmp_path):
        """No parser produces top_f_decode_pct, so a re-enrich must preserve a
        recorded value: the round-advance gate blocks on null, so a null write
        would wedge the campaign."""
        st = _base_state()
        mining = st["campaign"]["rounds"][0]["bottleneck_mining"]
        mining["completed_at"] = "2026-01-01T00:00:00Z"
        mining["top_f_decode_pct"] = 69.19
        sp = _write_state(tmp_path, st)
        mining_dir = self._write_mined(sp, _MINED_JSON)
        md = mining_dir / "bottleneck_analysis.md"
        md.write_text("no tables here", encoding="utf-8")

        assert _run("enrich", "--state", str(sp), "--mining",
                    "--from", str(md)).returncode == 0
        m = json.loads(sp.read_text())["campaign"]["rounds"][0]["bottleneck_mining"]
        assert m["top_f_decode_pct"] == 69.19
        assert m["top_component"] == "dense_proj_GEMM"  # mined.json still applied


# ─────────────────────────────────────────────
# enrich --gate: contract-name lookup at any depth
# ─────────────────────────────────────────────

class TestEnrichGateContract:
    def _state(self, tmp_path):
        st = _base_state()
        st["campaign"]["current_stage"] = "4_5_parallel_tracks"
        return _write_state(tmp_path, st)

    def test_gate_5_2_reads_contract_names_nested(self, tmp_path):
        """ammo-impl-champion.md names kernel_speedup{,_warm,_cold}; the champion
        authors the file per track, so the names are rarely at the top level."""
        sp = self._state(tmp_path)
        src = sp.parent / "gate_5_2_results.json"
        src.write_text(json.dumps({
            "gate": "5.2",
            "shapes": {
                "qkv": {"kernel_speedup_warm": 1.61, "kernel_speedup_cold": 1.20},
                "down": {"kernel_speedup_warm": 1.05, "kernel_speedup_cold": 1.01},
            },
        }), encoding="utf-8")

        r = _run("enrich", "--state", str(sp), "--gate", "5_2",
                 "--op-id", "op001", "--from", str(src))

        assert r.returncode == 0, r.stdout + r.stderr
        track = json.loads(sp.read_text())["campaign"]["rounds"][0][
            "parallel_tracks"]["tracks"]["op001"]
        assert track["gate_5_2_metrics"]["weighted_speedup_warm"] == 1.61
        assert track["gate_5_2_metrics"]["weighted_speedup_cold"] == 1.20
        assert track["gate_5_2_metrics"]["shapes_tested"] == 2

    def test_gate_5_2_bare_speedup_is_the_warm_figure(self, tmp_path):
        sp = self._state(tmp_path)
        src = sp.parent / "g.json"
        src.write_text(json.dumps(
            {"per_batch": {"8": {"speedup": 1.27}}}), encoding="utf-8")

        assert _run("enrich", "--state", str(sp), "--gate", "5_2",
                    "--op-id", "op001", "--from", str(src)).returncode == 0
        metrics = json.loads(sp.read_text())["campaign"]["rounds"][0][
            "parallel_tracks"]["tracks"]["op001"]["gate_5_2_metrics"]
        assert metrics["weighted_speedup_warm"] == 1.27
        assert "weighted_speedup_cold" not in metrics  # omitted, never null

    def test_gate_5_2_no_contract_field_fails_loudly(self, tmp_path):
        sp = self._state(tmp_path)
        src = sp.parent / "g.json"
        src.write_text(json.dumps({"gate": "5.2", "note": "prose only"}),
                       encoding="utf-8")

        r = _run("enrich", "--state", str(sp), "--gate", "5_2",
                 "--op-id", "op001", "--from", str(src))

        assert r.returncode == 1
        assert "found no contract field" in r.stderr
        assert "weighted_speedup_cold" in r.stderr
        # File untouched: a silent all-null write is the failure this prevents.
        assert json.loads(sp.read_text())["campaign"]["rounds"][0][
            "parallel_tracks"]["tracks"] == {}

    def test_gate_5_1a_reads_nested_max_abs_err_and_verdict(self, tmp_path):
        sp = self._state(tmp_path)
        src = sp.parent / "gate_5_1a_results.json"
        src.write_text(json.dumps({
            "gate": "5.1a",
            "tests": [
                {"shape": "a", "detail": {"max_abs_err": 1e-4}},
                {"shape": "b", "detail": {"max_abs_err": 3e-4}},
            ],
            "summary": {"overall": "PASS"},
        }), encoding="utf-8")

        assert _run("enrich", "--state", str(sp), "--gate", "5_1a",
                    "--op-id", "op001", "--from", str(src)).returncode == 0
        metrics = json.loads(sp.read_text())["campaign"]["rounds"][0][
            "parallel_tracks"]["tracks"]["op001"]["gate_5_1a_metrics"]
        assert metrics["overall"] == "PASS"
        assert metrics["max_abs_err"] == 3e-4   # worst across shapes
        assert metrics["shapes_tested"] == 2


# ─────────────────────────────────────────────
# ingest-baseline
# ─────────────────────────────────────────────

def _sweep_json(*rows, label="baseline"):
    return {
        "execution_mode": "inproc_sweep",
        "bench": {"baseline_label": label, "opt_label": "opt"},
        "results": list(rows),
    }


def _sweep_row(bs, avg, label="baseline", **percentiles):
    metrics = {"avg_s": avg}
    metrics.update({f"{k}_s": v for k, v in percentiles.items()})
    return {"batch_size": bs, "input_len": 64, "output_len": 512,
            label: {"avg_s": avg, "ok": True, "metrics": metrics}}


class TestIngestBaseline:
    def test_ingest_writes_per_bs_map_and_source_path(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        src = sp.parent / "e2e_latency_results.json"
        src.write_text(json.dumps(_sweep_json(
            _sweep_row(1, 1.5, p50=1.49, p90=1.6),
            _sweep_row(32, 10.0931, p10=10.06, p50=10.0729, p99=10.283),
        )), encoding="utf-8")

        r = _run("ingest-baseline", "--state", str(sp), "--from", str(src))

        assert r.returncode == 0, r.stdout + r.stderr
        rnd = json.loads(sp.read_text())["campaign"]["rounds"][0]
        e2e = rnd["baseline"]["e2e_latency"]
        assert set(e2e) == {"1", "32"}
        assert e2e["32"]["avg"] == 10.0931
        assert e2e["32"]["p50"] == 10.0729
        assert e2e["32"]["p99"] == 10.283
        # An unmeasured percentile stays absent; never synthesized from the mean.
        assert "p75" not in e2e["32"]
        assert rnd["profiling_baseline_path"] == str(src)
        # per_bs_verdict belongs to track/integration evaluation.
        assert rnd["baseline"]["per_bs_verdict"] is None

    def test_ingest_anchors_cumulative_speedup_at_smallest_bs(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        src = sp.parent / "s.json"
        src.write_text(json.dumps(_sweep_json(
            _sweep_row(32, 10.0), _sweep_row(1, 1.25))), encoding="utf-8")

        assert _run("ingest-baseline", "--state", str(sp),
                    "--from", str(src)).returncode == 0
        assert ammo_state._r1_baseline(json.loads(sp.read_text())) == 1.25

    def test_ingest_resolves_aggregate_mean_latency_ladder(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        src = sp.parent / "s.json"
        src.write_text(json.dumps({
            "bench": {"baseline_label": "baseline"},
            "results": [{"batch_size": 4,
                         "baseline": {"aggregate": {"mean_latency": 2.5},
                                      "avg_s": 99.0}}],
        }), encoding="utf-8")

        assert _run("ingest-baseline", "--state", str(sp),
                    "--from", str(src)).returncode == 0
        e2e = json.loads(sp.read_text())["campaign"]["rounds"][0][
            "baseline"]["e2e_latency"]
        assert e2e["4"]["avg"] == 2.5

    def test_ingest_named_round(self, tmp_path):
        st = _base_state()
        st["campaign"]["rounds"].append(_round_skel(2))
        st["campaign"]["rounds"][0]["audit"] = {
            "stage_67": {"passed_at": "2026-01-01T00:00:00Z"}}
        st["campaign"]["current_round"] = 2
        sp = _write_state(tmp_path, st)
        src = sp.parent / "s.json"
        src.write_text(json.dumps(_sweep_json(_sweep_row(8, 3.0))), encoding="utf-8")

        assert _run("ingest-baseline", "--state", str(sp), "--from", str(src),
                    "--round", "1").returncode == 0
        rounds = json.loads(sp.read_text())["campaign"]["rounds"]
        assert rounds[0]["baseline"]["e2e_latency"]["8"]["avg"] == 3.0
        assert rounds[1]["baseline"]["e2e_latency"] is None

    def test_ingest_malformed_json_leaves_state_untouched(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        before = sp.read_text()
        src = sp.parent / "s.json"
        src.write_text("{ this is not json", encoding="utf-8")

        r = _run("ingest-baseline", "--state", str(sp), "--from", str(src))

        assert r.returncode == 1
        assert "not parseable" in r.stderr
        assert sp.read_text() == before

    def test_ingest_no_results_rows_fails(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        before = sp.read_text()
        src = sp.parent / "s.json"
        src.write_text(json.dumps({"bench": {}, "results": []}), encoding="utf-8")

        r = _run("ingest-baseline", "--state", str(sp), "--from", str(src))

        assert r.returncode == 1
        assert "no results[] rows" in r.stderr
        assert sp.read_text() == before

    def test_ingest_wrong_label_fails_without_partial_write(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        before = sp.read_text()
        src = sp.parent / "s.json"
        src.write_text(json.dumps(_sweep_json(_sweep_row(8, 3.0))), encoding="utf-8")

        r = _run("ingest-baseline", "--state", str(sp), "--from", str(src),
                 "--label", "nosucharm")

        assert r.returncode == 1
        assert sp.read_text() == before

    def test_ingest_missing_source_file_fails(self, tmp_path):
        sp = _write_state(tmp_path, _base_state())
        r = _run("ingest-baseline", "--state", str(sp),
                 "--from", str(tmp_path / "nope.json"))
        assert r.returncode == 1 and "not found" in r.stderr

    def test_ingest_real_sweep_matches_source_numbers(self):
        """Golden: no rounding or renaming drift versus the real sweep JSON."""
        src = Path("/tmp/ammo-traces/e2e_latency_results.json")
        if not src.is_file():
            pytest.skip("real sweep fixture not staged")
        doc = json.loads(src.read_text(encoding="utf-8"))
        latency, errors = ammo_state.parse_baseline_sweep(doc)
        assert errors == []
        row = doc["results"][0]
        metrics = row["baseline"]["metrics"]
        entry = latency[str(row["batch_size"])]
        assert entry["avg"] == metrics["avg_s"]
        for p in ("p10", "p25", "p50", "p75", "p90", "p99"):
            assert entry[p] == metrics[p + "_s"]

    def test_stage_1_nudge_names_the_subcommand(self):
        st = _stage(_base_state(), "1_baseline", **{
            "baseline.completed_at": "2026-01-01T00:00:00Z",
            "audit.stage_1.passed_at": "2026-01-01T00:01:00Z",
        })
        msg, _ = _ns(st)
        assert "ingest-baseline" in msg


# ─────────────────────────────────────────────
# score_breakdown tier cap + EV identity
# ─────────────────────────────────────────────

class TestScoreBreakdownGates:
    def _t(self):
        return ammo_state.load_transitions()

    def test_typed_feasibility_over_tier_cap_blocks(self):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_1", "feasibility": 10,
            "expected_e2e_pct": 3.0, "weighted_total": 3.0,
        })
        prev = _debate_state(selected=[])
        reason = ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev)
        assert reason and "feasibility=10 > tier cap 3" in reason
        assert "tier_1" in reason

    def test_typed_feasibility_at_tier_cap_passes(self):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_1", "feasibility": 3,
            "expected_e2e_pct": 3.0, "weighted_total": 0.9,
        })
        prev = _debate_state(selected=[])
        assert ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev) is None

    def test_tier_3_typed_feasibility_uncapped(self):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_3", "feasibility": 10,
            "expected_e2e_pct": 2.0, "weighted_total": 2.0,
        })
        prev = _debate_state(selected=[])
        assert ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev) is None

    def test_ev_identity_violation_blocks_with_numbers(self):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_2", "feasibility": 7,
            "expected_e2e_pct": 1.2, "weighted_total": 6.5,
        })
        prev = _debate_state(selected=[])
        reason = ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev)
        assert reason and "weighted_total=6.5" in reason
        assert "7/10 x 1.2" in reason

    def test_ev_identity_within_one_percent_passes(self):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_2", "feasibility": 7,
            "expected_e2e_pct": 1.2, "weighted_total": 0.84,
        })
        prev = _debate_state(selected=[])
        assert ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev) is None

    # No contract pins the precision of weighted_total, and champions publish
    # it to 1 dp. Honest rounding must not read as a forged EV.
    @pytest.mark.parametrize("tier,feasibility,expected,rounded", [
        ("tier_3", 9, 2.6, 2.3),    # EV 2.34
        ("tier_2", 7, 1.17, 0.8),   # EV 0.819
        ("tier_1", 3, 0.5, 0.1),    # EV 0.15
        ("tier_3", 8, 3.2, 2.6),    # EV 2.56
        ("tier_2", 6, 0.9, 0.5),    # EV 0.54
    ])
    def test_ev_identity_tolerates_one_decimal_rounding(
        self, tier, feasibility, expected, rounded
    ):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": tier, "feasibility": feasibility,
            "expected_e2e_pct": expected, "weighted_total": rounded,
        })
        prev = _debate_state(selected=[])
        assert ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev) is None

    def test_ev_identity_still_catches_a_composite_score(self):
        """The legacy 0-10 composite (weighted_total=8.3 for EV 2.34) is the
        exact confusion the check exists to catch; rounding slack must not hide
        it."""
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_3", "feasibility": 9,
            "expected_e2e_pct": 2.6, "weighted_total": 8.3,
        })
        prev = _debate_state(selected=[])
        reason = ammo_state.gate_violation(
            _debate_state(selected=[cand]), self._t(), prev)
        assert reason and "weighted_total=8.3" in reason
        assert "9/10 x 2.6" in reason

    def test_legacy_breakdown_grandfathered_without_prev(self, tmp_path):
        cand = _selected_candidate("OP-001", evidence_scope="bound")
        cand["score_breakdown"].update({
            "evidence_tier": "tier_1", "feasibility": 10,
            "expected_e2e_pct": 3.0, "weighted_total": 99.0,
        })
        sp = _write_state(tmp_path, _debate_state(selected=[cand]))
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout


# ─────────────────────────────────────────────
# backfill
# ─────────────────────────────────────────────

class TestBackfill:
    def test_backfill_fills_missing(self, tmp_path):
        st = _base_state()
        rnd = st["campaign"]["rounds"][0]
        rnd["bottleneck_mining"]["completed_at"] = "2026-01-01T00:00:00Z"
        rnd["bottleneck_mining"]["top_bottleneck_share_pct"] = 30.0
        # decode_frac/component_breakdown absent (None) -> backfill should fill
        sp = _write_state(tmp_path, st)
        art = sp.parent
        md_dir = art / "rounds" / "1" / "mining"
        md_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / "bottleneck_analysis.md").write_text(_MINING_MD, encoding="utf-8")
        r = _run("backfill", "--state", str(sp), "--artifact-dir", str(art))
        assert r.returncode == 0, r.stdout + r.stderr
        doc = json.loads(sp.read_text())
        m = doc["campaign"]["rounds"][0]["bottleneck_mining"]
        assert m["decode_frac"] == 0.78
        assert m["component_breakdown"] is not None
        assert "backfilled round 1" in r.stdout

    def test_backfill_skips_incomplete_rounds(self, tmp_path):
        st = _base_state()  # mining not completed
        sp = _write_state(tmp_path, st)
        r = _run("backfill", "--state", str(sp), "--artifact-dir", str(sp.parent))
        assert r.returncode == 0
        assert "nothing to do" in r.stdout


# ─────────────────────────────────────────────
# scope gates — V1 tier P-score cap + V2 proxy-scope selection
# ─────────────────────────────────────────────

def _sb_entry(tier, p_score):
    """Free-form scoreboard entry shaped like the live campaign scoreboard."""
    return {
        "disposition": "selected", "evidence_tier": tier, "ev_pct": 0.5,
        "p_score": p_score, "projected_e2e_pct": 0.5, "rationale": "x",
    }


def _selected_candidate(
    op_id,
    *,
    evidence_scope=None,
    category="kernel_replacement",
    selection_mode="ordinary",
):
    # Contract-legal breakdown: feasibility (the P-score) is at the tier_2 cap
    # of 7, and weighted_total is EV = feasibility/10 x expected_e2e_pct.
    sb = {
        "evidence_tier": "tier_2", "expected_e2e_pct": 0.8,
        "feasibility": 7, "weighted_total": 0.56,
    }
    if evidence_scope is not None:
        sb["evidence_scope"] = evidence_scope
    return {
        "op_id": op_id,
        "selection_mode": selection_mode,
        "proposal_file": f"rounds/1/debate/proposals/{op_id}_proposal.md",
        "track_assignment": "lossless", "score_breakdown": sb,
        "stage_4_validation_obligations": [], "cited_evidence": ["file.py:1"],
        "category": category,
        "projected_e2e_improvement_pct": 0.8,
    }


def _debate_state(stage="1_baseline", *, scoreboard=None, selected=None):
    st = _base_state()
    st["campaign"]["current_stage"] = stage
    debate = st["campaign"]["rounds"][0]["debate"]
    if scoreboard is not None:
        debate["scoreboard"] = scoreboard
    if selected is not None:
        debate["selected_candidates"] = selected
    return st


class TestScopeGates:
    def _t(self):
        return ammo_state.load_transitions()

    def test_scoreboard_tier_cap_grandfathered_no_prev(self, tmp_path):
        # Locks the live back-compat guarantee: tier_2/p_score=8 with no prev PASSes.
        st = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 8)})
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout
        assert "PASS" in r.stdout

    def test_scoreboard_tier_cap_new_entry_blocks(self):
        prev = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 7)})
        new = _debate_state(scoreboard={
            "OP-001": _sb_entry("tier_2", 7),
            "OP-002": _sb_entry("tier_2", 9),
        })
        reason = ammo_state.gate_violation(new, self._t(), prev)
        assert reason and "p_score=9 > tier cap 7" in reason

    def test_scoreboard_tier_cap_modified_entry_blocks(self):
        prev = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 7)})
        new = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 8)})
        reason = ammo_state.gate_violation(new, self._t(), prev)
        assert reason and "p_score=8 > tier cap 7" in reason

    def test_scoreboard_tier_cap_unchanged_entry_passes(self):
        prev = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 8)})
        new = _debate_state(scoreboard={"OP-001": _sb_entry("tier_2", 8)})
        assert ammo_state.gate_violation(new, self._t(), prev) is None

    def test_scoreboard_tier3_no_cap(self):
        prev = _debate_state(scoreboard={})
        new = _debate_state(scoreboard={"OP-001": _sb_entry("tier_3", 10)})
        assert ammo_state.gate_violation(new, self._t(), prev) is None

    def test_selected_proxy_grandfathered_no_prev(self, tmp_path):
        st = _debate_state(selected=[_selected_candidate("OP-001", evidence_scope="proxy")])
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout
        assert "PASS" in r.stdout

    def test_selected_proxy_new_blocks(self):
        prev = _debate_state(selected=[
            _selected_candidate("OP-001", evidence_scope="production_boundary")])
        new = _debate_state(selected=[
            _selected_candidate("OP-001", evidence_scope="production_boundary"),
            _selected_candidate("OP-002", evidence_scope="proxy"),
        ])
        reason = ammo_state.gate_violation(new, self._t(), prev)
        assert reason and "evidence_scope='proxy'" in reason

    def test_selected_scope_nonproxy_passes(self):
        prev = _debate_state(selected=[])
        new = _debate_state(selected=[
            _selected_candidate("OP-002", evidence_scope="production_boundary")])
        assert ammo_state.gate_violation(new, self._t(), prev) is None

    def test_contingent_host_spike_typed_contract_passes(self):
        contingent = _selected_candidate(
            "OP-HOST", evidence_scope="bound", selection_mode="contingent_host_spike"
        )
        contingent["score_breakdown"]["expected_e2e_pct"] = 0
        contingent["score_breakdown"]["weighted_total"] = 0
        contingent["projected_e2e_improvement_pct"] = 0
        contingent["host_slice_ceiling_pct"] = 1.1
        contingent["stage_4_validation_obligations"] = [
            "production_boundary_spike"
        ]
        prev = _debate_state(selected=[])
        new = _debate_state(selected=[contingent])

        assert ammo_state.gate_violation(new, self._t(), prev) is None

    def test_contingent_host_spike_cannot_fabricate_ev(self):
        contingent = _selected_candidate(
            "OP-HOST", evidence_scope="bound", selection_mode="contingent_host_spike"
        )
        contingent["host_slice_ceiling_pct"] = 1.1
        contingent["stage_4_validation_obligations"] = [
            "production_boundary_spike"
        ]
        prev = _debate_state(selected=[])
        new = _debate_state(selected=[contingent])

        reason = ammo_state.gate_violation(new, self._t(), prev)

        assert reason and "zero projected magnitude and EV" in reason

    def test_only_one_contingent_host_spike_may_be_selected(self):
        candidates = []
        for op_id in ("OP-H1", "OP-H2"):
            candidate = _selected_candidate(
                op_id,
                evidence_scope="bound",
                selection_mode="contingent_host_spike",
            )
            candidate["score_breakdown"]["expected_e2e_pct"] = 0
            candidate["score_breakdown"]["weighted_total"] = 0
            candidate["projected_e2e_improvement_pct"] = 0
            candidate["host_slice_ceiling_pct"] = 1.1
            candidate["stage_4_validation_obligations"] = [
                "production_boundary_spike"
            ]
            candidates.append(candidate)
        prev = _debate_state(selected=[])
        new = _debate_state(selected=candidates)

        reason = ammo_state.gate_violation(new, self._t(), prev)

        assert reason and "at most one contingent_host_spike" in reason

    def test_contingent_contract_mutation_is_revalidated(self):
        contingent = _selected_candidate(
            "OP-HOST", evidence_scope="bound", selection_mode="contingent_host_spike"
        )
        contingent["score_breakdown"]["expected_e2e_pct"] = 0
        contingent["score_breakdown"]["weighted_total"] = 0
        contingent["projected_e2e_improvement_pct"] = 0
        contingent["host_slice_ceiling_pct"] = 1.1
        contingent["stage_4_validation_obligations"] = [
            "production_boundary_spike"
        ]
        prev = _debate_state(selected=[contingent])
        weakened = json.loads(json.dumps(contingent))
        weakened["host_slice_ceiling_pct"] = 0.1
        new = _debate_state(selected=[weakened])

        reason = ammo_state.gate_violation(new, self._t(), prev)

        assert reason and "meets or exceeds the campaign floor" in reason

    def test_selected_missing_scope_blocks(self):
        prev = _debate_state(selected=[])
        new = _debate_state(selected=[_selected_candidate("OP-002")])
        reason = ammo_state.gate_violation(new, self._t(), prev)
        assert reason and "has no score_breakdown.evidence_scope" in reason

    def test_selected_missing_category_blocks(self):
        prev = _debate_state(selected=[])
        new = _debate_state(selected=[
            _selected_candidate(
                "OP-002", evidence_scope="production_boundary", category=None
            )
        ])
        reason = ammo_state.gate_violation(new, self._t(), prev)
        assert reason and "has no category token" in reason

    def test_schema_accepts_evidence_scope_enum(self, tmp_path):
        st = _debate_state(selected=[_selected_candidate("OP-001")])
        sp = _write_state(tmp_path, st)
        field = ("campaign.rounds.0.debate.selected_candidates.0."
                 "score_breakdown.evidence_scope")
        ok = _run("set", "--state", str(sp), "--field", field, "--value", '"clean_e2e"')
        assert ok.returncode == 0, ok.stdout + ok.stderr
        bad = _run("set", "--state", str(sp), "--field", field, "--value", '"nonsense"')
        assert bad.returncode == 1
        assert "is not one of" in bad.stdout

    def test_live_state_shape_passes(self, tmp_path):
        # Integration guard for the back-compat requirement: the exact live shape
        # (tier_2/p_score=8 on 3 ops + candidates with no evidence_scope) PASSes.
        st = _base_state()
        st["campaign"]["current_stage"] = "6_integration"
        rnd = st["campaign"]["rounds"][0]
        rnd["audit"] = {
            "stage_45": {"passed_at": "2026-01-01T00:00:00Z"}
        }
        rnd["parallel_tracks"]["tracks"] = {
            "OP-006": {"status": "PASS"}, "OP-002": {"status": "PASS"}}
        rnd["debate"]["scoreboard"] = {
            "OP-001": _sb_entry("tier_2", 8),
            "OP-002": _sb_entry("tier_2", 8),
            "OP-006": _sb_entry("tier_2", 8),
        }
        rnd["debate"]["selected_candidates"] = [
            _selected_candidate("OP-006"), _selected_candidate("OP-002")]
        sp = _write_state(tmp_path, st)
        r = _run("validate", "--state", str(sp))
        assert r.returncode == 0, r.stdout
        assert "PASS" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
