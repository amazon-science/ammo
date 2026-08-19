#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the mechanical audit start stamp (ammo_state.py audit-started).

started_at and cycle record that an ammo-auditor was actually dispatched for a
gate. The auditor-spawn hook writes them; the lead writes only passed_at after
verdict review. Two behaviors are covered here: the stamping subcommand, and
the schema-4.2 backstop that rejects a passed gate with no start stamp.

Run: python3 -m pytest tests/test_audit_started.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_ENGINE = _SCRIPTS_DIR / "ammo_state.py"

_spec = importlib.util.spec_from_file_location("ammo_state_audit_started", str(_ENGINE))
engine = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(engine)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ENGINE), *args],
        text=True,
        capture_output=True,
    )


def _write_state(artifact_dir: Path, doc: dict) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "state.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _campaign_state(schema_version: str = "4.2", audit=None, rounds=1) -> dict:
    """Minimal campaign that reaches the backstop with no earlier violation."""
    round_list = []
    for round_id in range(1, rounds + 1):
        rnd = {
            "round_id": round_id,
            "status": "IN_PROGRESS",
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {}},
            "integration": {"status": "pending"},
            "debate": {},
        }
        if audit is not None:
            rnd["audit"] = copy.deepcopy(audit)
        round_list.append(rnd)
    return {
        "campaign": {
            "schema_version": schema_version,
            "status": "active",
            "current_round": 1,
            "current_stage": "1_baseline",
            "config": {"min_e2e_improvement_pct": 0.5},
            "rounds": round_list,
        }
    }


def _gate(state: dict, round_pos: int = 1, stage: str = "stage_1") -> dict:
    return state["campaign"]["rounds"][round_pos - 1]["audit"][stage]


# ─────────────────────────────────────────────
# audit-started: stamping
# ─────────────────────────────────────────────

def test_stamps_started_at_and_cycle_on_an_empty_gate(tmp_path: Path):
    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={}))

    result = _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_45", "--round", "1", "--cycle", "1",
    )

    assert result.returncode == 0, result.stderr
    gate = _gate(json.loads(path.read_text(encoding="utf-8")), stage="stage_45")
    assert gate["cycle"] == 1
    assert gate["started_at"].endswith("Z")
    # Seconds precision, timezone-aware — the shape passed_at uses elsewhere.
    assert len(gate["started_at"]) == len("2026-08-04T12:00:00Z")
    assert "passed_at" not in gate


def test_stamping_preserves_sibling_gates_and_existing_gate_fields(tmp_path: Path):
    artifact = tmp_path / "artifact"
    audit = {
        "stage_1": {"passed_at": "2026-08-04T10:00:00Z", "started_at": "2026-08-04T09:30:00Z"},
        "stage_2": {"verdict_file": "rounds/1/audits/stage_2.md"},
    }
    path = _write_state(artifact, _campaign_state(audit=audit))

    assert _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_2", "--round", "1", "--cycle", "3",
    ).returncode == 0

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert _gate(doc, stage="stage_1") == audit["stage_1"]
    stage_2 = _gate(doc, stage="stage_2")
    assert stage_2["verdict_file"] == "rounds/1/audits/stage_2.md"
    assert stage_2["cycle"] == 3


def test_re_audit_overwrites_the_previous_stamp(tmp_path: Path):
    artifact = tmp_path / "artifact"
    audit = {"stage_67": {"started_at": "2020-01-01T00:00:00Z", "cycle": 1}}
    path = _write_state(artifact, _campaign_state(audit=audit))

    assert _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_67", "--round", "1", "--cycle", "2",
    ).returncode == 0

    gate = _gate(json.loads(path.read_text(encoding="utf-8")), stage="stage_67")
    assert gate["cycle"] == 2
    assert gate["started_at"] != "2020-01-01T00:00:00Z"


def test_null_gate_becomes_an_object(tmp_path: Path):
    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={"stage_1": None}))

    assert _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_1", "--round", "1", "--cycle", "1",
    ).returncode == 0

    assert _gate(json.loads(path.read_text(encoding="utf-8")))["cycle"] == 1


def test_lock_file_is_created_next_to_state_json(tmp_path: Path):
    artifact = tmp_path / "artifact"
    _write_state(artifact, _campaign_state(audit={}))

    assert _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_1", "--round", "1", "--cycle", "1",
    ).returncode == 0

    assert (artifact / "state.json.lock").is_file()


def test_a_held_lock_blocks_every_other_writer_verb(tmp_path: Path):
    # A lock only one verb takes is no lock. The auditor spawns in the
    # background, so a lead `set` overlaps the hook's stamp and the unlocked
    # read-modify-write erases it — which then trips the 4.2 backstop on an
    # audit that really ran.
    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={"stage_2": {}}))
    verbs = {
        "set": ["set", "--state", str(path),
                "--field", "campaign.current_stage", "--value", '"1_baseline"'],
        "advance": ["advance", "--state", str(path), "--outcome", "SHIP"],
        "enrich": ["enrich", "--state", str(path), "--mining"],
        "backfill": ["backfill", "--state", str(path), "--artifact-dir", str(artifact)],
        "ingest-baseline": ["ingest-baseline", "--state", str(path),
                            "--from", str(artifact / "sweep.json")],
    }
    (artifact / "sweep.json").write_text("{}", encoding="utf-8")

    for name, args in verbs.items():
        with engine.state_lock(path):
            proc = subprocess.Popen(
                [sys.executable, str(_ENGINE), *args],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                proc.wait(timeout=3)
                raise AssertionError(
                    "%s ran while the state lock was held; stdout=%r stderr=%r"
                    % ((name,) + proc.communicate())
                )
            except subprocess.TimeoutExpired:
                pass
        proc.kill()
        proc.wait(timeout=10)


def test_reconcile_track_state_takes_the_same_lock(tmp_path: Path):
    # reconcile_track_state.py writes state.json too, with its own
    # tempfile+replace, so it has to serialize on the same inode.
    import importlib.util

    script = _SCRIPTS_DIR / "reconcile_track_state.py"
    spec = importlib.util.spec_from_file_location("reconcile_for_lock_test", str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={}))
    track_dir = artifact / "rounds" / "1" / "tracks" / "op001"
    track_dir.mkdir(parents=True)
    (track_dir / "evidence.json").write_text("{}", encoding="utf-8")
    state = json.loads(path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "IN_PROGRESS"}
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    with engine.state_lock(path):
        proc = subprocess.Popen(
            [sys.executable, str(script), "--artifact-dir", str(artifact),
             "--track-id", "op001", "--write"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            proc.wait(timeout=3)
            raise AssertionError(
                "reconcile ran while the state lock was held; stdout=%r stderr=%r"
                % proc.communicate()
            )
        except subprocess.TimeoutExpired:
            pass
    assert proc.wait(timeout=30) is not None


def test_a_held_lock_blocks_a_second_writer(tmp_path: Path):
    # os.replace() makes each write atomic but does not stop a lost update: two
    # writers can read the same state.json and then overwrite each other. The
    # exclusive lock has to cover the whole load-mutate-write, so a second
    # writer must wait rather than read stale content.
    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={"stage_2": {}}))

    with engine.state_lock(path):
        proc = subprocess.Popen(
            [
                sys.executable, str(_ENGINE), "audit-started",
                "--artifact-dir", str(artifact), "--stage", "stage_2",
                "--round", "1", "--cycle", "4",
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            proc.wait(timeout=2)
            raise AssertionError(
                "the second writer ran while the lock was held; "
                "stdout=%r stderr=%r" % proc.communicate()
            )
        except subprocess.TimeoutExpired:
            pass
        # Nothing was written while the lock was held.
        assert _gate(json.loads(path.read_text(encoding="utf-8")), stage="stage_2") == {}

    assert proc.wait(timeout=30) == 0, proc.stderr.read()
    assert _gate(json.loads(path.read_text(encoding="utf-8")), stage="stage_2")["cycle"] == 4


# ─────────────────────────────────────────────
# audit-started: fail-open no-ops (hooks are the caller)
# ─────────────────────────────────────────────

def test_round_without_an_audit_key_is_a_no_op(tmp_path: Path):
    # Adding the audit key would switch gate enforcement on for a legacy
    # campaign that never had it. Warn and leave the file alone.
    artifact = tmp_path / "artifact"
    doc = _campaign_state(schema_version="4.0")
    path = _write_state(artifact, doc)
    before = path.read_text(encoding="utf-8")

    result = _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_1", "--round", "1", "--cycle", "1",
    )

    assert result.returncode == 0
    assert "no audit key" in result.stdout
    assert path.read_text(encoding="utf-8") == before


def test_absent_round_is_a_no_op(tmp_path: Path):
    artifact = tmp_path / "artifact"
    path = _write_state(artifact, _campaign_state(audit={}))
    before = path.read_text(encoding="utf-8")

    result = _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_1", "--round", "7", "--cycle", "1",
    )

    assert result.returncode == 0
    assert "round 7 absent" in result.stdout
    assert path.read_text(encoding="utf-8") == before


# ─────────────────────────────────────────────
# audit-started: real errors exit non-zero
# ─────────────────────────────────────────────

def test_missing_state_json_exits_non_zero(tmp_path: Path):
    result = _run(
        "audit-started", "--artifact-dir", str(tmp_path / "nope"),
        "--stage", "stage_1", "--round", "1", "--cycle", "1",
    )
    assert result.returncode == 1
    assert "state.json not found" in result.stderr


def test_corrupt_state_json_exits_non_zero(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "state.json").write_text("{not json", encoding="utf-8")

    result = _run(
        "audit-started", "--artifact-dir", str(artifact),
        "--stage", "stage_1", "--round", "1", "--cycle", "1",
    )

    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_out_of_range_round_and_cycle_exit_non_zero(tmp_path: Path):
    artifact = tmp_path / "artifact"
    _write_state(artifact, _campaign_state(audit={}))

    for flag, value in (("--round", "0"), ("--cycle", "0")):
        args = [
            "audit-started", "--artifact-dir", str(artifact),
            "--stage", "stage_1", "--round", "1", "--cycle", "1",
        ]
        args[args.index(flag) + 1] = value
        assert _run(*args).returncode == 1


def test_stage_6_and_stage_7_are_not_dispatch_targets(tmp_path: Path):
    artifact = tmp_path / "artifact"
    _write_state(artifact, _campaign_state(audit={}))

    for stage in ("stage_6", "stage_7"):
        result = _run(
            "audit-started", "--artifact-dir", str(artifact),
            "--stage", stage, "--round", "1", "--cycle", "1",
        )
        assert result.returncode != 0


# ─────────────────────────────────────────────
# Backstop: passed_at without started_at
# ─────────────────────────────────────────────

def test_backstop_blocks_passed_gate_without_start_stamp_at_4_2():
    transitions = engine.load_transitions()
    state = _campaign_state(audit={"stage_1": {"passed_at": "2026-08-04T10:00:00Z"}})

    reason = engine.gate_violation(state, transitions)

    assert reason is not None
    assert "Audit provenance violation" in reason
    assert "round 1 audit.stage_1" in reason
    assert "audit-started --artifact-dir <artifact_dir> --stage stage_1 --round 1 --cycle 1" in reason
    assert ".codex/skills/ammo/scripts/ammo_state.py" in reason


def test_backstop_passes_once_started_at_is_present():
    transitions = engine.load_transitions()
    state = _campaign_state(audit={
        "stage_1": {
            "started_at": "2026-08-04T09:30:00Z",
            "cycle": 1,
            "passed_at": "2026-08-04T10:00:00Z",
        }
    })

    assert engine.gate_violation(state, transitions) is None


def test_backstop_ignores_a_gate_that_has_not_passed_yet():
    transitions = engine.load_transitions()
    state = _campaign_state(audit={"stage_1": {"passed_at": None}})

    assert engine.gate_violation(state, transitions) is None


def test_backstop_does_not_apply_below_schema_4_2():
    transitions = engine.load_transitions()
    for version in ("4.1", "4.0", "3.0"):
        state = _campaign_state(
            schema_version=version,
            audit={"stage_1": {"passed_at": "2026-08-04T10:00:00Z"}},
        )
        assert engine.gate_violation(state, transitions) is None, version


def test_backstop_names_the_round_it_found():
    transitions = engine.load_transitions()
    state = _campaign_state(rounds=2, audit={})
    state["campaign"]["rounds"][1]["audit"] = {
        "stage_67": {"passed_at": "2026-08-04T10:00:00Z"}
    }

    reason = engine.gate_violation(state, transitions)

    assert "round 2 audit.stage_67" in reason
    assert "--round 2" in reason


def test_backstop_floor_comes_from_transitions_json():
    transitions = engine.load_transitions()
    assert transitions["audit_started_min_schema"] == {"major": 4, "minor": 2}


# ─────────────────────────────────────────────
# Backstop: pre-consolidation aliases the gates still honor
# ─────────────────────────────────────────────

def test_backstop_scans_the_legacy_s67_aliases():
    # gate_violation accepts stage_6/stage_7 in place of stage_67, so a lead
    # that types the legacy key would otherwise satisfy the S67 gate with no
    # mechanical evidence that an auditor ran.
    transitions = engine.load_transitions()
    for stage_key in ("stage_6", "stage_7"):
        state = _campaign_state(audit={stage_key: {"passed_at": "2026-08-04T10:00:00Z"}})

        reason = engine.gate_violation(state, transitions)

        assert reason is not None, stage_key
        assert "Audit provenance violation" in reason
        assert "round 1 audit.%s" % stage_key in reason


def test_legacy_alias_message_steers_to_stage_67_not_a_backfill():
    # audit-started cannot address stage_6/stage_7, so telling the lead to
    # backfill that key would be an impossible instruction.
    transitions = engine.load_transitions()
    state = _campaign_state(audit={"stage_6": {"passed_at": "2026-08-04T10:00:00Z"}})

    reason = engine.gate_violation(state, transitions)

    assert "Move the verdict to audit.stage_67" in reason
    assert "--stage stage_67 --round 1 --cycle 1" in reason
    assert "--stage stage_6 " not in reason


def test_legacy_alias_with_a_start_stamp_passes():
    transitions = engine.load_transitions()
    state = _campaign_state(audit={
        "stage_6": {
            "started_at": "2026-08-04T09:30:00Z",
            "cycle": 1,
            "passed_at": "2026-08-04T10:00:00Z",
        }
    })

    assert engine.gate_violation(state, transitions) is None


def test_legacy_aliases_are_still_not_dispatch_targets():
    # Only the backstop's READ side widened. The stamping side must stay at the
    # four dispatchable gates.
    assert engine.AUDIT_GATE_STAGES == ("stage_1", "stage_2", "stage_45", "stage_67")
    assert engine.LEGACY_AUDIT_GATE_STAGES == ("stage_6", "stage_7")
    assert engine.PROVENANCED_GATE_STAGES == (
        engine.AUDIT_GATE_STAGES + engine.LEGACY_AUDIT_GATE_STAGES
    )
