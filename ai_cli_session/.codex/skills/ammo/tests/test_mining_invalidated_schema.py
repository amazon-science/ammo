#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Z2-T1 / P4#7: schema legality of `mining_invalidated` (+ reason) round fields.

Pins the schema gap fix: the round object is `additionalProperties:false`, so
before P4#7 the engine reads `mining_invalidated` (ammo_state.advance, :1219)
that no schema-legal write could ever persist. This file adds the two optional
round properties and proves:

  1. a round carrying BOTH fields (boolean `true` + reason string) validates and
     round-trips through `advance --outcome EXHAUSTED` to 2_bottleneck_mining;
  2. the load-bearing boolean requirement holds at BOTH layers — the schema
     REJECTS the string form `"true"` (the `"type":"boolean"` declaration makes
     the misuse impossible to persist), and the in-process `advance` on a dict
     carrying the string routes to 3_debate (because `is True` is False for a
     string).

ADDITIVE: this is a NEW file. It does NOT import, edit, mutate, or weaken the
frozen tests/test_ammo_state.py (TestAdvance, test_exhausted_mining_invalidated_at_mining,
test_exhausted_without_invalidation_goes_to_debate). Helpers are mirrored locally
on purpose so this file is self-contained.

Run: python3 -m pytest tests/test_mining_invalidated_schema.py -q
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_ENGINE = _SCRIPTS_DIR / "ammo_state.py"
_SCHEMA = Path(__file__).resolve().parents[3] / "schemas" / "state.schema.json"

sys.path.insert(0, str(_SCRIPTS_DIR))
import ammo_state  # noqa: E402


# ─────────────────────────────────────────────
# Local fixtures (mirrored — NO import from the frozen test_ammo_state.py)
# ─────────────────────────────────────────────

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


def _base_state():
    """Canonical valid v4.1 state — mirrors the bash harness base_v2 shape."""
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


def _audited_round1_state():
    """Round 1 at 7_campaign_eval with a full audit chain (incl stage_67) so the
    new-round-start gate is satisfied after advance creates round 2. Mirrored
    locally; NOT imported from the frozen test."""
    st = _base_state()
    st["campaign"]["current_stage"] = "7_campaign_eval"
    st["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-06-08T14:56:31Z"},
        "stage_2": {"passed_at": "2026-06-08T15:48:49Z"},
        "stage_45": {"passed_at": "2026-06-08T18:26:35Z"},
        "stage_67": {"passed_at": "2026-06-08T18:42:23Z"},
    }
    return st


def _write_state(tmp_path, state):
    """Write a state.json inside a .codex/schemas-walkable artifact dir."""
    root = tmp_path
    (root / ".git").mkdir(exist_ok=True)
    (root / ".codex" / "schemas").mkdir(parents=True, exist_ok=True)
    (root / ".codex" / "schemas" / "state.schema.json").write_text(
        _SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    art = root / "kernel_opt_artifacts" / "t"
    art.mkdir(parents=True, exist_ok=True)
    sp = art / "state.json"
    sp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return sp


def _run(*argv):
    return subprocess.run([sys.executable, str(_ENGINE), *argv],
                          capture_output=True, text=True)


def _schema_validates(state):
    """In-process schema check using the same Draft202012Validator the engine uses."""
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    errs = ammo_state.schema_errors(state, schema)
    # schema_errors returns None only when jsonschema is unavailable (not here)
    assert errs is not None, "jsonschema must be present to run this test"
    return errs == []


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_round_with_both_fields_validates(tmp_path):
    """A round carrying mining_invalidated=true + reason string is schema-legal
    and passes the engine `validate` verb (exit 0 / PASS)."""
    st = _base_state()
    st["campaign"]["rounds"][0]["mining_invalidated"] = True
    st["campaign"]["rounds"][0]["mining_invalidated_reason"] = "wrong component attribution"
    sp = _write_state(tmp_path, st)
    r = _run("validate", "--state", str(sp))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout, r.stdout


def test_advance_exhausted_mining_invalidated_file_roundtrip(tmp_path):
    """advance --outcome EXHAUSTED on an audited round-1 state carrying
    mining_invalidated=true persists (validates before write) and routes the new
    round to 2_bottleneck_mining (re-mine). The written doc still validates."""
    st = _audited_round1_state()
    st["campaign"]["rounds"][0]["mining_invalidated"] = True
    st["campaign"]["rounds"][0]["mining_invalidated_reason"] = "diagnosis was wrong"
    sp = _write_state(tmp_path, st)
    r = _run("advance", "--state", str(sp), "--outcome", "EXHAUSTED")
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(sp.read_text())
    assert doc["campaign"]["current_stage"] == "2_bottleneck_mining", doc["campaign"]["current_stage"]
    assert doc["campaign"]["rounds"][0]["status"] == "EXHAUSTED"
    # the written doc (now carrying the optional fields) re-validates clean
    assert _schema_validates(doc), "advanced doc with mining_invalidated must validate"


def test_string_true_is_rejected_and_does_not_trigger_remine(tmp_path):
    """Boolean-sensitivity guard (team-lead ruling 3), two assertions:

    (i)  SCHEMA rejects the string form: mining_invalidated="true" (string) fails
         validate — the "type":"boolean" declaration makes the misuse impossible
         to persist (the strongest guarantee).
    (ii) ENGINE `is True` branch: advance(<dict-with-string-true>, EXHAUSTED)
         routes to 3_debate, NOT 2_bottleneck_mining, because
         `cur.get("mining_invalidated") is True` (ammo_state.py :1219) is False
         for a string.
    """
    # (i) schema rejects the string form
    st = _base_state()
    st["campaign"]["rounds"][0]["mining_invalidated"] = "true"  # string, not bool
    sp = _write_state(tmp_path, st)
    r = _run("validate", "--state", str(sp))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "PASS" not in r.stdout, r.stdout
    assert not _schema_validates(st), "string 'true' must FAIL boolean type validation"

    # (ii) in-process advance with the string routes to 3_debate (is True is False)
    st2 = _base_state()
    st2["campaign"]["current_stage"] = "7_campaign_eval"
    st2["campaign"]["rounds"][0]["mining_invalidated"] = "true"  # string
    transitions = ammo_state.load_transitions()
    _, new_stage = ammo_state.advance(copy.deepcopy(st2), "EXHAUSTED", transitions)
    assert new_stage == "3_debate", new_stage
