# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Task-2 regressions for the merged Codex AMMO state schema."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "state.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _campaign_properties() -> dict:
    return _schema()["properties"]["campaign"]["properties"]


def _round_properties() -> dict:
    return _campaign_properties()["rounds"]["items"]["properties"]


def _track_properties() -> dict:
    return _round_properties()["parallel_tracks"]["properties"]["tracks"]["additionalProperties"]["properties"]


def _audit_gate_properties(stage: str) -> dict:
    return _round_properties()["audit"]["properties"][stage]


def test_new_campaign_floor_is_half_percent_without_deflation_contract():
    assert "4.2" in _campaign_properties()["schema_version"]["enum"]
    config = _campaign_properties()["config"]
    assert config["properties"]["min_e2e_improvement_pct"]["default"] == 0.5
    assert "lossless_e2e_deflation_factor" not in config["required"]
    assert "lossy_quant_e2e_deflation_factor" not in config["required"]
    assert "lossless_e2e_deflation_factor" not in config["properties"]
    assert "lossy_quant_e2e_deflation_factor" not in config["properties"]
    legacy_floor = config["properties"]["debate_elimination_score_floor"]["description"]
    assert "LEGACY" in legacy_floor
    assert "EV_pct < min_e2e_improvement_pct" in legacy_floor
    assert "backward compatibility" in legacy_floor


def test_round_schema_adds_mining_invalidation_and_diluted_rollup():
    props = _round_properties()
    assert props["mining_invalidated"]["type"] == "boolean"
    assert props["mining_invalidated_reason"]["type"] == ["string", "null"]
    diluted_tracks = props["diluted_tracks"]
    assert diluted_tracks["type"] == ["array", "null"]
    assert diluted_tracks["items"]["properties"]["op_id"]["type"] == "string"


def test_track_schema_preserves_codex_fields_and_adds_diluted_marker():
    props = _track_properties()
    assert props["diluted"]["type"] == ["boolean", "null"]
    assert "SKIPPED" in props["gate_5_1a"]["enum"]
    assert "SKIPPED" in props["gate_5_2"]["enum"]
    for field in (
        "gate_5_1a_skip_reason",
        "gate_5_2_skip_reason",
        "branch",
        "worktree_path",
        "evidence_path",
        "kill_criteria_results",
        "completed_at",
        "implementer_agent",
        "implementer_rollout_id",
        "monitor_agent",
        "monitor_evidence_path",
        "monitor_offsets_path",
        "monitor_summary_path",
    ):
        assert field in props
    tracks = _round_properties()["parallel_tracks"]["properties"]["tracks"]
    assert tracks["propertyNames"]["pattern"] == "^[A-Za-z0-9_-]+$"
    boundary = props["gate_5_2_boundary"]
    assert "e2e_equivalent_improvement_pct" in boundary["required"]


def test_selected_candidate_preserves_winning_proposal_locator():
    selected = _round_properties()["debate"]["properties"]["selected_candidates"]["items"]

    assert "proposal_file" in selected["properties"]
    assert "rounds/{N}/debate/proposals/" in selected["properties"]["proposal_file"][
        "description"
    ]
    assert "proposal_file" in selected["required"]


def test_selected_candidate_carries_optional_display_name():
    """The lead mints a short display name with each op_id; the op_id stays the
    key. Optional for back-compat: pre-4.2 selected_candidates omit it."""
    selected = _round_properties()["debate"]["properties"]["selected_candidates"]["items"]

    name = selected["properties"]["name"]
    assert name["type"] == "string"
    assert name["minLength"] == 1
    assert "name" not in selected["required"]


def test_selected_candidate_supports_typed_contingent_host_spike():
    selected = _round_properties()["debate"]["properties"]["selected_candidates"][
        "items"
    ]

    assert selected["properties"]["selection_mode"]["enum"] == [
        "ordinary",
        "contingent_host_spike",
    ]
    obligations = selected["properties"]["stage_4_validation_obligations"]
    assert "production_boundary_spike" in obligations["items"]["enum"]


def test_mining_schema_distinguishes_f_e2e_from_f_decode():
    mining = _round_properties()["bottleneck_mining"]["properties"]

    assert "decode wall time" in mining["top_f_decode_pct"]["description"]
    assert "total E2E wall time" in mining["top_f_e2e_pct"]["description"]
    assert "Stage 7 stop check" in mining["top_addressable_e2e_pct"]["description"]


def test_shipped_optimization_can_record_diluted():
    item = _campaign_properties()["shipped_optimizations"]["items"]
    assert item["properties"]["diluted"]["type"] == ["boolean", "null"]


def test_completed_mining_is_not_globally_deadlocking_schema():
    mining = _round_properties()["bottleneck_mining"]
    assert "allOf" not in mining


def test_auditor_escalation_accepts_track_array_scope():
    spec = _campaign_properties()["auditor_escalation"]
    payload = {
        "stage": "4_5_parallel_tracks",
        "round": 2,
        "reason": "three audit cycles exhausted",
        "verdict_files": ["rounds/2/audits/stage_45.md"],
        "scope": ["op001", "op007"],
        "resumable_task": "fix and re-audit quarantined tracks",
    }
    assert list(Draft202012Validator(spec).iter_errors(payload)) == []


def test_auditor_escalation_accepts_campaign_scope_and_rejects_other_string():
    spec = _campaign_properties()["auditor_escalation"]
    payload = {
        "stage": "6_integration",
        "round": 3,
        "reason": "campaign-wide invariant failed",
        "verdict_files": ["rounds/3/audits/stage_67.md"],
        "scope": "campaign",
        "resumable_task": "resolve the campaign blocker and re-audit",
    }
    validator = Draft202012Validator(spec)
    assert list(validator.iter_errors(payload)) == []
    payload["scope"] = "op001"
    assert list(validator.iter_errors(payload))


def test_schema_version_enum_is_unchanged_by_the_audit_start_fields():
    # started_at/cycle are additive on the existing 4.2 gate objects. A new
    # version entry would force every consumer to learn a version it does not
    # need.
    assert _campaign_properties()["schema_version"]["enum"] == [
        "3.0", "4.0", "4.1", "4.2"
    ]


def test_audit_gates_carry_optional_started_at_and_cycle():
    for stage in ("stage_2", "stage_67"):
        gate = _audit_gate_properties(stage)
        props = gate["properties"]

        started = props["started_at"]
        assert started["type"] == ["string", "null"]
        assert started["format"] == "date-time"
        assert "ammo-audit-start-stamp.sh" in started["description"]

        cycle = props["cycle"]
        assert cycle["type"] == ["integer", "null"]
        assert cycle["minimum"] == 1

        # Optional: a legacy gate that carries only passed_at still validates
        # against the schema. The 4.2 backstop lives in ammo_state.py, not here.
        assert "started_at" not in gate.get("required", [])
        assert "cycle" not in gate.get("required", [])


def test_audit_gate_accepts_the_stamped_shape_and_rejects_cycle_zero():
    spec = _audit_gate_properties("stage_45")
    validator = Draft202012Validator(spec)
    stamped = {
        "started_at": "2026-08-04T12:00:00Z",
        "cycle": 2,
        "passed_at": "2026-08-04T12:31:00Z",
        "verdict_file": "rounds/1/audits/stage_45_cycle_2.md",
    }
    assert list(validator.iter_errors(stamped)) == []
    assert list(validator.iter_errors({"cycle": 0}))
