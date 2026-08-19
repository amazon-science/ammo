#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Z2-T2 / P4#6: fail-CLOSED validation (default-on; emit decision:block; escape hatch).

Pins the engine half of the fail-closed flip. "fail-closed" = EMIT a
`decision:block` JSON payload on stdout (PostToolUse enforcement), never rely on
a nonzero exit code. Default-on; escape hatch `AMMO_VALIDATE_FAIL_OPEN=1`.

Unavailable validation blocks for missing jsonschema, missing/corrupt schema,
validator exceptions, and state-file JSONDecodeError. A valid state with the
library and schema present passes silently. AMMO_VALIDATE_FAIL_OPEN=1 is the
only degraded bypass.

These tests cover the Codex Python state engine's fail-closed paths.
PostToolUse integration coverage lives in test_hook_semantics.py.

Run: python3 -m pytest tests/test_fail_closed.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _advanceable_state():
    state = _base_state()
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_67": {"passed_at": "2026-07-16T00:00:00Z"}
    }
    return state


def _write_state(tmp_path, state, with_schema=True):
    """Write state.json in a .git-bounded artifact dir. with_schema=True drops a
    schema copy where find_schema() will locate it; False leaves none."""
    root = tmp_path
    (root / ".git").mkdir(exist_ok=True)
    if with_schema:
        (root / ".codex" / "schemas").mkdir(parents=True, exist_ok=True)
        (root / ".codex" / "schemas" / "state.schema.json").write_text(
            _SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    art = root / "kernel_opt_artifacts" / "t"
    art.mkdir(parents=True, exist_ok=True)
    sp = art / "state.json"
    sp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return sp


def _args(sp, emit=None, schema=None):
    argv = ["validate", "--state", str(sp)]
    if schema is not None:
        argv += ["--schema", str(schema)]
    if emit:
        argv += ["--emit", emit]
    return ammo_state.build_parser().parse_args(argv)


@pytest.fixture(autouse=True)
def _clear_failopen_env(monkeypatch):
    """Default-on: ensure no AMMO_VALIDATE_FAIL_OPEN leaks in from the runner env."""
    monkeypatch.delenv("AMMO_VALIDATE_FAIL_OPEN", raising=False)


# ─────────────────────────────────────────────
# FLIP case 1: missing jsonschema LIB
# ─────────────────────────────────────────────

def test_missing_jsonschema_lib_blocks_default(tmp_path, capsys, monkeypatch):
    """Lib unavailable + default-on (emit hook) → decision:block with remediation."""
    monkeypatch.setattr(ammo_state, "_HAVE_JSONSCHEMA", False)
    monkeypatch.setattr(ammo_state, "Draft202012Validator", None)
    sp = _write_state(tmp_path, _base_state())
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["decision"] == "block"
    assert "jsonschema" in payload["reason"]
    assert "AMMO_VALIDATE_FAIL_OPEN=1" in payload["reason"]


def test_missing_jsonschema_lib_failopen_with_env(tmp_path, capsys, monkeypatch):
    """Lib unavailable + AMMO_VALIDATE_FAIL_OPEN=1 → silent (no block)."""
    monkeypatch.setattr(ammo_state, "_HAVE_JSONSCHEMA", False)
    monkeypatch.setattr(ammo_state, "Draft202012Validator", None)
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _base_state())
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_missing_jsonschema_lib_cli_mode_returns_1(tmp_path, capsys, monkeypatch):
    """CLI mode (no --emit hook): lib-missing default-on prints reason + exit 1
    (no decision:block JSON in CLI mode — that's the hook-only enforcement form)."""
    monkeypatch.setattr(ammo_state, "_HAVE_JSONSCHEMA", False)
    monkeypatch.setattr(ammo_state, "Draft202012Validator", None)
    sp = _write_state(tmp_path, _base_state())
    rc = ammo_state.cmd_validate(_args(sp))
    out = capsys.readouterr().out
    assert rc == 1
    assert "jsonschema" in out
    assert '"decision"' not in out


# ─────────────────────────────────────────────
# FLIP case 2: state-file JSONDecodeError
# ─────────────────────────────────────────────

def test_state_jsondecodeerror_blocks_default(tmp_path, capsys):
    """Corrupt JSON state + default-on (emit hook) → decision:block."""
    sp = _write_state(tmp_path, _base_state())
    sp.write_text("{this is not valid json", encoding="utf-8")
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "AMMO_VALIDATE_FAIL_OPEN=1" in payload["reason"]


def test_state_jsondecodeerror_failopen_with_env(tmp_path, capsys, monkeypatch):
    """Corrupt JSON + AMMO_VALIDATE_FAIL_OPEN=1 → silent (legacy bypass restored)."""
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _base_state())
    sp.write_text("{this is not valid json", encoding="utf-8")
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


# ─────────────────────────────────────────────
# Missing schema FILE is also unverifiable and therefore blocks by default.
# ─────────────────────────────────────────────

def test_missing_schema_file_blocks_default(tmp_path, capsys):
    """Valid state with no discoverable schema is unverifiable and blocks."""
    sp = _write_state(tmp_path, _base_state(), with_schema=False)
    missing = tmp_path / "missing-state.schema.json"
    rc = ammo_state.cmd_validate(_args(sp, emit="hook", schema=missing))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "schema" in payload["reason"]
    assert "AMMO_VALIDATE_FAIL_OPEN=1" in payload["reason"]


def test_missing_schema_file_failopen_with_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _base_state(), with_schema=False)
    missing = tmp_path / "missing-state.schema.json"
    rc = ammo_state.cmd_validate(_args(sp, emit="hook", schema=missing))
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_validator_exception_blocks_default(tmp_path, capsys, monkeypatch):
    sp = _write_state(tmp_path, _base_state())

    def explode(*_args, **_kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(ammo_state, "schema_errors", explode)
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "validation could not run" in payload["reason"]


def test_validator_exception_failopen_with_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _base_state())

    def explode(*_args, **_kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(ammo_state, "schema_errors", explode)
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_set_blocks_when_jsonschema_library_unavailable(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ammo_state, "_HAVE_JSONSCHEMA", False)
    monkeypatch.setattr(ammo_state, "Draft202012Validator", None)
    sp = _write_state(tmp_path, _base_state())
    before = sp.read_text(encoding="utf-8")
    args = ammo_state.build_parser().parse_args(
        [
            "set",
            "--state",
            str(sp),
            "--field",
            "session_id",
            "--value",
            '"degraded-test"',
        ]
    )

    rc = ammo_state.cmd_set(args)

    assert rc == 1
    assert "validation could not run" in capsys.readouterr().out
    assert sp.read_text(encoding="utf-8") == before


def test_set_failopen_escape_hatch_allows_degraded_write(tmp_path, monkeypatch):
    monkeypatch.setattr(ammo_state, "_HAVE_JSONSCHEMA", False)
    monkeypatch.setattr(ammo_state, "Draft202012Validator", None)
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _base_state())
    args = ammo_state.build_parser().parse_args(
        [
            "set",
            "--state",
            str(sp),
            "--field",
            "session_id",
            "--value",
            '"degraded-test"',
        ]
    )

    rc = ammo_state.cmd_set(args)

    assert rc == 0
    assert json.loads(sp.read_text(encoding="utf-8"))["session_id"] == "degraded-test"


def test_advance_blocks_on_schema_load_exception_and_leaves_state_untouched(tmp_path, capsys):
    sp = _write_state(tmp_path, _advanceable_state())
    schema_path = tmp_path / ".codex" / "schemas" / "state.schema.json"
    schema_path.write_text("{broken schema", encoding="utf-8")
    before = sp.read_text(encoding="utf-8")
    args = ammo_state.build_parser().parse_args(
        [
            "advance", "--state", str(sp), "--schema", str(schema_path),
            "--outcome", "SHIP",
        ]
    )

    rc = ammo_state.cmd_advance(args)

    assert rc == 1
    assert "validation could not run" in capsys.readouterr().out
    assert sp.read_text(encoding="utf-8") == before


def test_advance_schema_load_exception_escape_hatch_allows_degraded_write(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AMMO_VALIDATE_FAIL_OPEN", "1")
    sp = _write_state(tmp_path, _advanceable_state())
    schema_path = tmp_path / ".codex" / "schemas" / "state.schema.json"
    schema_path.write_text("{broken schema", encoding="utf-8")
    args = ammo_state.build_parser().parse_args(
        [
            "advance", "--state", str(sp), "--schema", str(schema_path),
            "--outcome", "SHIP",
        ]
    )

    rc = ammo_state.cmd_advance(args)

    assert rc == 0
    assert json.loads(sp.read_text(encoding="utf-8"))["campaign"]["current_round"] == 2


# ─────────────────────────────────────────────
# Over-block guard: valid state + jsonschema present passes silently
# ─────────────────────────────────────────────

def test_valid_state_with_jsonschema_passes_silent(tmp_path, capsys):
    """Positive control — lib present, valid state, default-on (emit hook) → no block."""
    sp = _write_state(tmp_path, _base_state())
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_schema_invalid_state_still_blocks_with_block_json(tmp_path, capsys):
    """Regression: a genuinely schema-invalid state still emits decision:block
    (the fail-closed flip must not shadow the normal schema-error path)."""
    st = _base_state()
    st["campaign"]["current_stage"] = "campaign_complete"  # not a real stage
    sp = _write_state(tmp_path, st)
    rc = ammo_state.cmd_validate(_args(sp, emit="hook"))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "is not one of" in payload["reason"]
