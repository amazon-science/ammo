# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Zone A — Schema Validation Tests for E2E Latency v4.0 restructure.

Tests 1-18 from the TDD plan: validate $defs.e2e_latency_entry,
map-based e2e_latency fields, per_bs_verdict siblings, kernel_speedup split,
e2e_speedup simplification, additionalProperties locking, and schema_version enum.
"""

import json
import pathlib

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "ai_cli_session"
    / ".claude"
    / "schemas"
    / "state.schema.json"
)


@pytest.fixture(scope="module")
def schema():
    """Load the full state schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def entry_schema(schema):
    """Extract just the $defs/e2e_latency_entry sub-schema for isolated testing."""
    return schema["$defs"]["e2e_latency_entry"]


def _minimal_state(schema_version="4.0", **round_overrides):
    """Build a minimal valid state.json for full-schema validation tests."""
    baseline = {
        "started_at": None,
        "completed_at": None,
        "e2e_latency": None,
        "per_bs_verdict": None,
    }
    bottleneck_mining = {
        "started_at": None,
        "completed_at": None,
        "top_bottleneck_share_pct": None,
    }
    debate = {
        "started_at": None,
        "completed_at": None,
        "candidates": [],
        "rounds_completed": 0,
        "max_rounds": 3,
        "selected_winners": [],
        "selected_candidates": None,
        "selection_rationale": None,
    }
    parallel_tracks = {
        "started_at": None,
        "completed_at": None,
        "tracks": {},
    }
    integration = {
        "started_at": None,
        "completed_at": None,
        "status": "pending",
        "passing_candidates": [],
        "failed_candidates": [],
        "selected_candidates": [],
        "conflict_analysis": None,
        "combined_patch_branch": None,
        "combined_e2e_result": None,
        "final_decision": None,
        "resolver_invoked": None,
        "resolver_outcome": None,
        "conflicting_tracks": None,
        "e2e_latency_combined": None,
        "per_bs_verdict": None,
        "commit_sha": None,
    }
    campaign_eval = {"started_at": None, "completed_at": None}
    audit = None

    round_data = {
        "round_id": 1,
        "status": "IN_PROGRESS",
        "team_name": None,
        "profiling_baseline_path": None,
        "baseline": baseline,
        "bottleneck_mining": bottleneck_mining,
        "debate": debate,
        "parallel_tracks": parallel_tracks,
        "integration": integration,
        "campaign_eval": campaign_eval,
        "audit": audit,
        "shipped": [],
        "dropped": [],
        "cumulative_speedup_after": None,
        "combined_e2e_speedup_x": None,
        "combined_e2e_delta_pp": None,
        "note": None,
        "round_summary": None,
        "exhausted_technologies": [],
    }
    round_data.update(round_overrides)

    return {
        "target": {
            "model_id": "test-model",
            "hardware": "H100",
            "dtype": "fp8",
            "tp": 1,
            "dp": 1,
            "ep": 1,
            "component": "fused_moe",
        },
        "session_id": "test-session-001",
        "gpu_resources": {
            "gpu_count": 8,
            "gpu_model": "H100",
            "memory_total_gib": 640.0,
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        },
        "campaign": {
            "schema_version": schema_version,
            "status": "active",
            "current_round": 1,
            "current_stage": "1_baseline",
            "config": {
                "min_e2e_improvement_pct": 2.0,
                "noise_tolerance_pct": 0.5,
                "catastrophic_regression_pct": 5.0,
                "debate_elimination_score_floor": 5.0,
            },
            "shipped_optimizations": [],
            "agent_costs": [],
            "rounds": [round_data],
        },
    }


# ---------------------------------------------------------------------------
# Test 1: e2e_latency_entry validates full percentile shape
# ---------------------------------------------------------------------------
class TestE2ELatencyEntry:
    def test_e2e_latency_entry_validates_full_percentile_shape(self, entry_schema):
        """Test 1: Full percentile entry validates."""
        entry = {
            "avg": 7.66,
            "p50": 7.55,
            "p10": 7.2,
            "p25": 7.4,
            "p75": 7.8,
            "p90": 8.0,
            "p99": 8.5,
        }
        jsonschema.validate(entry, entry_schema)

    def test_e2e_latency_entry_requires_only_avg(self, entry_schema):
        """Test 2: avg-only entries validate; missing avg fails."""
        jsonschema.validate({"avg": 7.66}, entry_schema)
        jsonschema.validate({"avg": 7.66, "p50": None, "p90": None}, entry_schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"p50": 7.55}, entry_schema)

    def test_e2e_latency_entry_rejects_additional_properties(self, entry_schema):
        """Test 3: Extra field rejected by additionalProperties: false."""
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"avg": 7.66, "baseline_avg_s": 7.66}, entry_schema
            )


# ---------------------------------------------------------------------------
# Test 4-6: Map-based e2e_latency on baseline
# ---------------------------------------------------------------------------
class TestBaselineE2ELatencyMap:
    def test_e2e_latency_map_validates_dynamic_bs_keys(self, schema):
        """Test 4: Baseline with e2e_latency map of dynamic BS keys validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["baseline"]["e2e_latency"] = {
            "128": {"avg": 7.66, "p50": 7.55},
            "256": {"avg": 8.2, "p50": 8.1},
        }
        jsonschema.validate(state, schema)

    def test_e2e_latency_map_null_validates(self, schema):
        """Test 5: baseline.e2e_latency = null validates (nullable)."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["baseline"]["e2e_latency"] = None
        jsonschema.validate(state, schema)

    def test_per_bs_verdict_is_sibling_of_e2e_latency(self, schema):
        """Test 6: per_bs_verdict at same level as e2e_latency validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["baseline"]["e2e_latency"] = {
            "128": {"avg": 7.66, "p50": 7.55},
        }
        state["campaign"]["rounds"][0]["baseline"]["per_bs_verdict"] = {
            "128": "PASS",
            "256": "NOISE",
        }
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 7: Track e2e_latency_opt
# ---------------------------------------------------------------------------
class TestTrackE2ELatencyOpt:
    def test_track_e2e_latency_opt_validates(self, schema):
        """Test 7: Track with e2e_latency_opt map validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": {"cold_bs8": 1.42, "warm_bs1": 1.28},
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": {"128": "PASS"},
                "e2e_latency_opt": {
                    "128": {"avg": 6.8, "p50": 6.7},
                },
                "e2e_amdahl_prediction_pp": 2.5,
                "gate_5_1a": "PASS",
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        jsonschema.validate(state, schema)

    def test_track_e2e_latency_opt_null_validates(self, schema):
        """Test 7b: Track with e2e_latency_opt = null validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 8: Integration e2e_latency_combined
# ---------------------------------------------------------------------------
class TestIntegrationE2ELatencyCombined:
    def test_integration_e2e_latency_combined_validates(self, schema):
        """Test 8: Integration with e2e_latency_combined map validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["e2e_latency_combined"] = {
            "128": {"avg": 6.5, "p50": 6.4},
        }
        jsonschema.validate(state, schema)

    def test_integration_e2e_latency_combined_null_validates(self, schema):
        """Test 8b: e2e_latency_combined = null validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["e2e_latency_combined"] = None
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 9: Kernel speedup split
# ---------------------------------------------------------------------------
class TestKernelSpeedupSplit:
    def test_kernel_speedup_split_scalar_and_variants(self, schema):
        """Test 9: kernel_speedup as number + kernel_speedup_variants validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": {"cold_bs8": 1.42, "warm_bs1": 1.28},
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": {"128": "PASS"},
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        jsonschema.validate(state, schema)

    def test_kernel_speedup_object_form_fails(self, schema):
        """Test 9b: kernel_speedup as object (old form) fails validation."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": {"cold_bs8": 1.42},
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 10: e2e_speedup simplified to number|null
# ---------------------------------------------------------------------------
class TestE2ESpeedupSimplified:
    def test_e2e_speedup_simplified_to_number_null(self, schema):
        """Test 10: e2e_speedup as number validates; as object fails."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        jsonschema.validate(state, schema)

    def test_e2e_speedup_object_form_fails(self, schema):
        """Test 10b: e2e_speedup as object (old form) fails."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": {"speedup_x": 1.18},
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 11: passing_candidates e2e_speedup simplified
# ---------------------------------------------------------------------------
class TestPassingCandidatesE2ESpeedup:
    def test_passing_candidates_e2e_speedup_simplified(self, schema):
        """Test 11: passing_candidates[*].e2e_speedup as number validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["passing_candidates"] = [
            {
                "op_id": "triton_fused_moe",
                "verdict": "PASS",
                "e2e_speedup": 1.2,
                "files_changed": None,
                "gating": None,
            }
        ]
        jsonschema.validate(state, schema)

    def test_passing_candidates_e2e_speedup_object_fails(self, schema):
        """Test 11b: passing_candidates[*].e2e_speedup as object fails."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["passing_candidates"] = [
            {
                "op_id": "triton_fused_moe",
                "verdict": "PASS",
                "e2e_speedup": {"speedup_x": 1.2},
                "files_changed": None,
                "gating": None,
            }
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 12: additionalProperties: false on locked objects
# ---------------------------------------------------------------------------
class TestAdditionalPropertiesFalse:
    def test_additional_properties_false_on_locked_objects(self, schema):
        """Test 12: Injecting garbage key into locked objects fails validation."""
        locked_paths = [
            ("target",),
            ("gpu_resources",),
        ]
        for path in locked_paths:
            state = _minimal_state()
            obj = state
            for key in path:
                obj = obj[key]
            obj["_garbage_key"] = True
            with pytest.raises(jsonschema.ValidationError, match="_garbage_key|additional"):
                jsonschema.validate(state, schema)

    def test_additional_properties_false_on_baseline(self, schema):
        """Test 12b: Baseline rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["baseline"]["_garbage_key"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_bottleneck_mining(self, schema):
        """Test 12c: bottleneck_mining rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["bottleneck_mining"]["_garbage_key"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_parallel_tracks(self, schema):
        """Test 12d: parallel_tracks (outer) rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["_garbage_key"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_campaign_eval(self, schema):
        """Test 12e: campaign_eval rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["campaign_eval"]["_garbage_key"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_integration(self, schema):
        """Test 12f: integration rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["_garbage_key"] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_audit_stages(self, schema):
        """Test 12g: audit.stage_* sub-objects reject unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["audit"] = {
            "stage_1": {"passed_at": None, "verdict_file": None, "_garbage_key": True},
            "stage_45": None,
            "stage_6": None,
            "stage_7": None,
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)

    def test_additional_properties_false_on_track(self, schema):
        """Test 12h: Track inner object rejects unknown properties."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
                "_garbage_key": True,
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 13: target has dp field
# ---------------------------------------------------------------------------
class TestTargetDp:
    def test_target_has_dp_field(self, schema):
        """Test 13: target with dp field validates."""
        state = _minimal_state()
        state["target"]["dp"] = 2
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 14: debate has selection_rationale
# ---------------------------------------------------------------------------
class TestDebateSelectionRationale:
    def test_debate_has_selection_rationale(self, schema):
        """Test 14: debate with selection_rationale validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["debate"]["selection_rationale"] = (
            "Best trade-off between feasibility and expected speedup."
        )
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 15: schema_version accepts both 3.0 and 4.0
# ---------------------------------------------------------------------------
class TestSchemaVersion:
    def test_schema_version_accepts_both_3_and_4(self, schema):
        """Test 15: schema_version 3.0 and 4.0 both validate; 5.0 fails."""
        state_v3 = _minimal_state(schema_version="3.0")
        jsonschema.validate(state_v3, schema)

        state_v4 = _minimal_state(schema_version="4.0")
        jsonschema.validate(state_v4, schema)

        state_v5 = _minimal_state(schema_version="5.0")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state_v5, schema)


# ---------------------------------------------------------------------------
# Test 16: integration has commit_sha
# ---------------------------------------------------------------------------
class TestIntegrationCommitSha:
    def test_integration_has_commit_sha(self, schema):
        """Test 16: integration with commit_sha validates. opt_s is NOT declared."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["commit_sha"] = "abc123"
        jsonschema.validate(state, schema)

    def test_integration_rejects_opt_s(self, schema):
        """Test 16b: opt_s is not a declared field - rejected by additionalProperties: false."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["integration"]["opt_s"] = 6.5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 17: Track has e2e_amdahl_prediction_pp
# ---------------------------------------------------------------------------
class TestTrackAmdahlPrediction:
    def test_track_has_e2e_amdahl_prediction_pp(self, schema):
        """Test 17: Track with e2e_amdahl_prediction_pp validates."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "PASS",
                "verdict": "PASS",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": None,
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": 2.5,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": None,
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        jsonschema.validate(state, schema)


# ---------------------------------------------------------------------------
# Test 18: gating sub-objects allow additional properties
# ---------------------------------------------------------------------------
class TestGatingAllowsAdditional:
    def test_gating_sub_objects_allow_additional_properties(self, schema):
        """Test 18: gating, pre_gating_results, post_gating_results, crossover_probing allow extras."""
        state = _minimal_state()
        state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
            "triton_fused_moe": {
                "status": "GATING_REQUIRED",
                "verdict": "GATING_REQUIRED",
                "classification": "lossless",
                "correctness": True,
                "kernel_speedup": 1.35,
                "kernel_speedup_variants": None,
                "e2e_speedup": 1.18,
                "kernel_speedup_warm": None,
                "kernel_speedup_cold": None,
                "per_bs_verdict": {"128": "PASS", "256": "REGRESSED"},
                "e2e_latency_opt": None,
                "e2e_amdahl_prediction_pp": None,
                "gate_5_1a": None,
                "gate_5_2": None,
                "gating": {
                    "mechanism": "bs_dispatch",
                    "env_var": "VLLM_FUSED_MOE_DISPATCH",
                    "dispatch_condition": "bs >= crossover",
                    "crossover_threshold_bs": 256,
                    "crossover_probing": {
                        "probed_points": [64, 128, 256],
                        "predicted_bs": 192,
                        "confirmed_bs": 256,
                        "time_minutes": 3.5,
                        "converged": True,
                        "custom_key": "allowed",
                    },
                    "pre_gating_results": {
                        "custom_key": "value",
                        "custom_metric": 42,
                    },
                    "post_gating_results": {
                        "custom_key": "value",
                    },
                    "regressing_bs": [256, 512],
                },
                "worktree_branch": None,
                "description": None,
                "commit_sha": None,
                "fail_reason": None,
                "validation_results_path": None,
                "remediation_items_status": None,
            }
        }
        # Should validate without error — gating sub-objects are NOT locked
        jsonschema.validate(state, schema)
