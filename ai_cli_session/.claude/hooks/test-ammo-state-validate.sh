#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-state-validate.sh (PostToolUse Write|Edit hook)
# Run: bash .claude/hooks/test-ammo-state-validate.sh
#
# Validates v2 state.schema.json enforcement via the PostToolUse hook:
#   - Fast bail on non-state paths / missing files / missing schema
#   - Top-level additionalProperties:false (v1 fields rejected)
#   - campaign.required: [status, current_round, current_stage, config, rounds]
#   - campaign.current_stage enum (7 real stages; no terminal pseudo-stages)
#   - campaign.config required thresholds + numeric types
#   - campaign.rounds minItems:1
#   - Nested: debate, parallel_tracks.tracks.*, integration.status enums
#   - Pass-through on valid v2 state
#
# Canonical valid v2 state is produced by jq_base and mutated per-test via `jq`.

set -euo pipefail

# AMMO_TEST_HOOK lets a reviewer aim this harness at an ARCHIVED copy of the
# hook (red->green evidence for the reconcile_track_state.py --write cases).
HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-state-validate.sh}"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
ARTIFACT_DIR="$TMPDIR/kernel_opt_artifacts/test_target"
mkdir -p "$ARTIFACT_DIR"
mkdir -p "$TMPDIR/.claude/schemas"

SCHEMA_SRC="$(cd "$(dirname "$0")" && pwd)/../schemas/state.schema.json"
cp "$SCHEMA_SRC" "$TMPDIR/.claude/schemas/state.schema.json"

mkdir -p "$TMPDIR/.git"

STATE_FILE="$ARTIFACT_DIR/state.json"

# The hook refreshes the /tmp/ammo-state-prev-<SID>.json snapshot after each
# validate (it owns that ordering contract; see the hook header). Point the
# snapshot root at TMPDIR and clear it per case, so one case's post-write state
# never becomes the next case's --prev baseline — every case here asserts a
# single write in isolation, not a sequence.
export AMMO_REMINDER_STATE_DIR="$TMPDIR/prev"
mkdir -p "$AMMO_REMINDER_STATE_DIR"
_clear_prev_snapshot() { rm -f "$AMMO_REMINDER_STATE_DIR"/ammo-state-prev-*.json 2>/dev/null || true; }

cleanup() {
    rm -rf "$TMPDIR"
    rm -f "$TMPDIR/hook-stdout" "$TMPDIR/hook-stderr"
}
trap cleanup EXIT

# Canonical valid v2 state — every test mutates this via jq to produce its payload.
# Matches new_target.py's _state_json output shape.
base_v2() {
    cat <<'JSON'
{
  "target": {
    "model_id": "test-model",
    "hardware": "H100",
    "dtype": "bf16",
    "tp": 1,
    "dp": 1,
    "ep": 1,
    "component": "auto"
  },
  "session_id": null,
  "gpu_resources": {
    "gpu_count": 1,
    "gpu_model": "NVIDIA H100",
    "memory_total_gib": 80.0,
    "cuda_visible_devices": "0"
  },
  "campaign": {
    "schema_version": "4.0",
    "status": "active",
    "current_round": 1,
    "current_stage": "1_baseline",
    "config": {
      "min_e2e_improvement_pct": 1.0,
      "noise_tolerance_pct": 0.5,
      "catastrophic_regression_pct": 5.0
    },
    "cumulative_speedup_vs_round1": 1.0,
    "round_1_baseline_latency_s": null,
    "shipped_optimizations": [],
    "agent_costs": [],
    "rounds": [
      {
        "round_id": 1,
        "status": "IN_PROGRESS",
        "team_name": null,
        "profiling_baseline_path": null,
        "baseline": {"started_at": null, "completed_at": null, "e2e_latency": null, "per_bs_verdict": null},
        "bottleneck_mining": {"started_at": null, "completed_at": null, "top_bottleneck_share_pct": null},
        "debate": {
          "started_at": null, "completed_at": null,
          "candidates": [], "rounds_completed": 0, "max_rounds": 4,
          "selected_winners": []
        },
        "parallel_tracks": {"started_at": null, "completed_at": null, "tracks": {}},
        "integration": {
          "started_at": null, "completed_at": null,
          "status": "pending",
          "passing_candidates": [], "failed_candidates": [], "selected_candidates": [],
          "conflict_analysis": null, "combined_patch_branch": null,
          "combined_e2e_result": null, "e2e_latency_combined": null, "per_bs_verdict": null, "commit_sha": null,
          "final_decision": null,
          "resolver_invoked": null, "resolver_outcome": null, "conflicting_tracks": null
        },
        "campaign_eval": {"started_at": null, "completed_at": null},
        "shipped": [], "dropped": [],
        "cumulative_speedup_after": null,
        "combined_e2e_speedup_x": null,
        "combined_e2e_delta_pp": null,
        "note": null, "round_summary": null
      }
    ]
  }
}
JSON
}

# Write a v2 state.json, optionally applying a jq mutation.
# Usage: write_state [jq_expr]
write_state() {
    local expr="${1:-.}"
    base_v2 | jq "$expr" > "$STATE_FILE"
}

run_test() {
    local test_name="$1"
    local expected_exit="$2"
    local expected_block="$3"
    local expected_pattern="${4:-}"
    local file_path="${5:-$STATE_FILE}"
    local actual_exit=0

    TOTAL=$((TOTAL + 1))
    _clear_prev_snapshot
    echo '{"tool_input": {"file_path": "'"$file_path"'"}}' | \
        bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?

    if [ "$actual_exit" -ne "$expected_exit" ]; then
        echo "  FAIL [$TOTAL]: $test_name — exit code (expected=$expected_exit, got=$actual_exit)"
        echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
        echo "        stderr: $(cat "$TMPDIR/hook-stderr")"
        FAIL=$((FAIL + 1))
        return
    fi

    local has_block="false"
    if grep -q '"decision"' "$TMPDIR/hook-stdout" 2>/dev/null; then
        has_block="true"
    fi

    if [ "$has_block" != "$expected_block" ]; then
        echo "  FAIL [$TOTAL]: $test_name — block state (expected=$expected_block, got=$has_block)"
        echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ -n "$expected_pattern" ]; then
        if ! grep -qE "$expected_pattern" "$TMPDIR/hook-stdout" 2>/dev/null; then
            echo "  FAIL [$TOTAL]: $test_name — missing pattern '$expected_pattern'"
            echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    echo "  PASS [$TOTAL]: $test_name"
    PASS=$((PASS + 1))
}

# ══════════════════════════════════════════════════
echo "== Fast bail: path filtering and missing preconditions =="
# ══════════════════════════════════════════════════

# Test 1: Non-state path → hook ignores
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "/tmp/random.md"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: Non-state path is ignored (no block)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Non-state path is ignored (no block)"
    FAIL=$((FAIL + 1))
fi

# Test 2: Empty file_path → fast bail
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {}}' | bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: Missing file_path is ignored (fast bail)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Missing file_path is ignored (fast bail)"
    FAIL=$((FAIL + 1))
fi

# Test 3: state.json path but file doesn't exist → fast bail
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$TMPDIR"'/kernel_opt_artifacts/nonexistent/state.json"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: Nonexistent state.json is ignored (fast bail)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Nonexistent state.json is ignored (fast bail)"
    FAIL=$((FAIL + 1))
fi

# ══════════════════════════════════════════════════
echo ""
echo "== Valid v2 states (should NOT block) =="
# ══════════════════════════════════════════════════

# Test 4: Canonical bootstrap v2 state
write_state
run_test "Bootstrap v2 state (round 1, stage 1_baseline) passes" 0 "false"

# Test 5: Stage 2 transition
write_state '.campaign.current_stage = "2_bottleneck_mining" | .campaign.rounds[0].baseline.started_at = "2026-04-23T10:00:00Z" | .campaign.rounds[0].baseline.completed_at = "2026-04-23T10:05:00Z"'
run_test "current_stage '2_bottleneck_mining' with baseline timestamps passes" 0 "false"

# Test 6: Stage 7b_report
write_state '.campaign.current_stage = "7b_report"'
run_test "current_stage '7b_report' passes" 0 "false"

# Test 7: Valid tracks status/verdict
write_state '.campaign.current_stage = "4_5_parallel_tracks" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "OP-001": {"status": "PASS", "verdict": "PASS", "correctness": true},
    "OP-002": {"status": "IN_PROGRESS", "verdict": null}
  }'
run_test "parallel_tracks.tracks with status/verdict passes" 0 "false"

# Test 8: Valid integration.status='completed'
write_state '.campaign.current_stage = "6_integration" | .campaign.rounds[0].integration.status = "completed"'
run_test "rounds[0].integration.status='completed' passes" 0 "false"

# Test 9: Multi-round campaign (round 2 IN_PROGRESS, round 1 completed)
write_state '.campaign.current_round = 2 |
  .campaign.rounds[0].status = "completed" |
  .campaign.rounds[0].debate.completed_at = "2026-04-23T11:00:00Z" |
  .campaign.rounds += [{
    "round_id": 2, "status": "IN_PROGRESS",
    "team_name": "ammo-round-2", "profiling_baseline_path": null,
    "baseline": {"started_at": null, "completed_at": null, "e2e_latency": null, "per_bs_verdict": null},
    "bottleneck_mining": {"started_at": null, "completed_at": null, "top_bottleneck_share_pct": null},
    "debate": {"started_at": null, "completed_at": null, "candidates": [], "rounds_completed": 0, "max_rounds": 3, "selected_winners": []},
    "parallel_tracks": {"started_at": null, "completed_at": null, "tracks": {}},
    "integration": {"started_at": null, "completed_at": null, "status": "pending", "passing_candidates": [], "failed_candidates": [], "selected_candidates": [], "conflict_analysis": null, "combined_patch_branch": null, "combined_e2e_result": null, "e2e_latency_combined": null, "per_bs_verdict": null, "commit_sha": null, "final_decision": null, "resolver_invoked": null, "resolver_outcome": null, "conflicting_tracks": null},
    "campaign_eval": {"started_at": null, "completed_at": null},
    "shipped": [], "dropped": [],
    "cumulative_speedup_after": null, "combined_e2e_speedup_x": null, "combined_e2e_delta_pp": null,
    "note": null, "round_summary": null
  }]'
run_test "Multi-round (round 2 IN_PROGRESS) passes" 0 "false"

# Test 11: Terminal campaign status with current_stage at last active stage (no terminal pseudo-stage).
# Terminal statuses are gated: they need the stage_67 audit stamp, a terminal
# integration status, and the numeric Stage 7 stop-check inputs.
write_state '.campaign.status = "campaign_complete" | .campaign.current_stage = "7_campaign_eval"
  | .campaign.rounds[0].audit = {"stage_67": {"passed_at": "2026-04-23T12:00:00Z"}}
  | .campaign.rounds[0].integration.status = "combined"
  | .campaign.rounds[0].bottleneck_mining.top_f_e2e_pct = 5.0
  | .campaign.rounds[0].bottleneck_mining.top_addressable_e2e_pct = 0.4'
run_test "Terminal campaign.status + current_stage='7_campaign_eval' passes" 0 "false"

# Test 11b: Terminal status WITHOUT the stage_67 audit stamp is fail-closed.
write_state '.campaign.status = "campaign_complete" | .campaign.current_stage = "7_campaign_eval"'
run_test "Terminal campaign.status without stage_67 audit BLOCKS" 0 "true" "stage_67"

# Test 12: shipped_optimizations canonical form
write_state '.campaign.shipped_optimizations = [{"op_id": "OP-fused-rms-qkv", "round": 1, "classification": "lossless"}]'
run_test "shipped_optimizations canonical {op_id,round,classification} passes" 0 "false"

# Test 12b: debate.candidates with string op-IDs (early debate phase)
write_state '.campaign.rounds[0].debate.candidates = ["op-001", "op-002"]'
run_test "debate.candidates with string op-IDs passes" 0 "false"

# Test 12c: debate.candidates with full scored objects (post-selection)
write_state '.campaign.rounds[0].debate.candidates = [{"op_id": "op-001", "champion_id": "c-1", "target": "attention_decode"}, {"op_id": "op-002", "champion_id": "c-2", "target": "rmsnorm"}]'
run_test "debate.candidates with full scored objects passes" 0 "false"

# Test 12d: debate.candidates rejects empty-string item
write_state '.campaign.rounds[0].debate.candidates = [""]'
run_test "debate.candidates rejects empty string" 0 "true" "is not valid"

# Test 12e: debate.candidates rejects object missing op_id
write_state '.campaign.rounds[0].debate.candidates = [{"champion_id": "c-1", "target": "attention_decode"}]'
run_test "debate.candidates rejects object without op_id" 0 "true" "is not valid"

# ══════════════════════════════════════════════════
echo ""
echo "== optimization_category enum (NN#8 widening — monotonic superset) =="
# ══════════════════════════════════════════════════

# A full, schema-valid selected_candidates entry carrying a category, parameterized by category value.
# Used to prove: every legacy value STILL validates AND every new value validates (add-only widening).
sel_cand_with_category() {
    local cat="$1"
    echo ".campaign.rounds[0].debate.selected_candidates = [{
        \"op_id\": \"OP-cat-test\",
        \"track_assignment\": \"structural\",
        \"score_breakdown\": {\"feasibility\": 7, \"evidence_tier\": \"tier_2\", \"expected_e2e_pct\": 1.2, \"weighted_total\": 6.5},
        \"stage_4_validation_obligations\": [\"cuda_graph_check\"],
        \"cited_evidence\": [\"rounds/1/mining/bottleneck_analysis.md:42\"],
        \"proposal_file\": \"rounds/1/debate/proposals/op-cat-test_proposal.md\",
        \"category\": \"$cat\"
    }]"
}

# Test 12f-h: the THREE LEGACY values must still validate (C1/C4 — retained, never deleted).
for legacy in kernel_replacement kernel_fusion dispatch_optimization; do
    write_state "$(sel_cand_with_category "$legacy")"
    run_test "selected_candidates.category='$legacy' (legacy) still validates" 0 "false"
done

# Test 12i-n: the SIX NEW values must validate (widening is live at the single enforcement point).
for newcat in custom_kernel weight_layout_transform compute_graph_pass execution_pipeline_restructuring communication_strategy attention_kv_layout; do
    write_state "$(sel_cand_with_category "$newcat")"
    run_test "selected_candidates.category='$newcat' (new) validates" 0 "false"
done

# Test 12o: null category still allowed (anyOf:[null,$ref]; category not in required).
write_state "$(sel_cand_with_category "custom_kernel" | sed 's/"custom_kernel"/null/')"
run_test "selected_candidates.category=null still validates" 0 "false"

# Test 12p: an off-catalog garbage value is still rejected (enum is closed, just wider).
# category is anyOf:[null,$ref], so jsonschema reports "not valid under any of the given schemas".
write_state "$(sel_cand_with_category "flag_flip_lol")"
run_test "selected_candidates.category='flag_flip_lol' rejected (not in enum)" 0 "true" "is not valid under any of the given schemas"

# ══════════════════════════════════════════════════
echo ""
echo "== audit gate: same-round + new-round-start + post-SHIP stage_1 exemption =="
# ══════════════════════════════════════════════════
# The audit gate (ammo-state-validate.sh) blocks stage transitions that skip an
# audit. It activates only when the round's `audit` key is PRESENT (even `{}`).
# These tests pin both legs of the gate AND the post-round-1 mining exemption:
# a re-mine in round N>1 (post-SHIP) has no fresh Stage 1, so the same-round
# `2_bottleneck_mining → stage_1` requirement is exempted for N>1 — but the
# independent new-round-start gate (prev round `stage_67`) still guards entry,
# so round 1's full audit chain can never be bypassed.

# Helper: write a 2-round state with controllable audit objects + stage + version.
#   $1 = round-1 audit JSON ('{}' / '{"stage_67":{...}}' / 'null' to omit the key)
#   $2 = round-2 audit JSON ('null' to omit the key)
#   $3 = current_stage
#   $4 = schema_version (default 4.1)
write_two_round() {
    local r1audit="$1" r2audit="$2" stage="$3" sver="${4:-4.1}"
    base_v2 | jq \
        --argjson r1audit "$r1audit" \
        --argjson r2audit "$r2audit" \
        --arg stage "$stage" \
        --arg sver "$sver" '
        .campaign.schema_version = $sver
        | .campaign.current_round = 2
        | .campaign.current_stage = $stage
        | .campaign.rounds[0].status = "completed"
        | (if $r1audit == null then . else .campaign.rounds[0].audit = $r1audit end)
        | .campaign.rounds += [ (.campaign.rounds[0]
            | .round_id = 2 | .status = "IN_PROGRESS"
            | .bottleneck_mining = {"started_at": null, "completed_at": null, "top_bottleneck_share_pct": null}
            | del(.audit)) ]
        | (if $r2audit == null then . else .campaign.rounds[1].audit = $r2audit end)
    ' > "$STATE_FILE"
}

# --- Round 1: same-round stage_1 gate is FULLY enforced (no exemption) ---

# Test AG1: round 1 mining with audit present but stage_1 missing → BLOCK.
# Round 1 has a real Stage 1 baseline; the exemption must NOT leak to round 1.
write_state '.campaign.current_stage = "2_bottleneck_mining" | .campaign.rounds[0].audit = {}'
run_test "AG1: round 1 mining + audit={} (no stage_1) BLOCKS" 0 "true" "stage_1"

# Test AG2: round 1 mining with stage_1 stamped → ALLOW.
write_state '.campaign.current_stage = "2_bottleneck_mining" | .campaign.rounds[0].audit = {"stage_1": {"passed_at": "2026-06-08T14:56:31Z"}}'
run_test "AG2: round 1 mining + stage_1 set ALLOWS" 0 "false"

# Test AG3: round 1 mining with NO audit key → gate skipped (legacy bypass) → ALLOW.
write_state '.campaign.current_stage = "2_bottleneck_mining" | del(.campaign.rounds[0].audit)'
run_test "AG3: round 1 mining + no audit key (legacy bypass) ALLOWS" 0 "false"

# --- Round N>1 (post-SHIP re-mine): same-round stage_1 requirement is EXEMPTED ---

# Test AG4 (THE FIX): round 2 re-mine, audit carries stage_2 but NOT stage_1,
# previous round stage_67 present → ALLOW. Post-SHIP rounds skip Stage 1
# (T16 eliminated), so requiring same-round stage_1 here would deadlock.
write_two_round '{"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' '{"stage_2": {"passed_at": "2026-06-09T15:35:00Z"}}' "2_bottleneck_mining"
run_test "AG4: round 2 re-mine, stage_2 set / stage_1 absent ALLOWS (post-SHIP exemption)" 0 "false"

# Test AG5: schema 4.1+ rounds must carry audit={} (the engine scaffolds it on
# advance) — a round 2 append that OMITS the audit key entirely fails closed,
# while the canonical write carrying audit={} passes.
write_two_round '{"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' 'null' "2_bottleneck_mining"
run_test "AG5: round 2 re-mine without audit key BLOCKS (must carry audit={})" 0 "true" "Audit state missing"
write_two_round '{"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' '{}' "2_bottleneck_mining"
run_test "AG5b: round 2 re-mine with audit={} ALLOWS" 0 "false"

# Test AG6 (SAFETY — new-round-start gate intact): round 2 re-mine where the
# same-round stage_1 check is exempted, but round 1 never stamped stage_67 →
# BLOCK via the independent new-round-start gate. Proves the exemption cannot
# be used to skip round 1's audit chain.
write_two_round '{}' '{"stage_2": {"passed_at": "2026-06-09T15:35:00Z"}}' "2_bottleneck_mining"
run_test "AG6: round 2 re-mine blocked when round 1 stage_67 missing (new-round gate)" 0 "true" "new round start blocked"

# Test AG7 (scope): the exemption is mining-only. Round 2 integration without
# stage_45 still BLOCKS regardless of round number (stage_45 is never exempted).
write_two_round '{"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' '{}' "6_integration"
run_test "AG7: round 2 integration without stage_45 still BLOCKS (exemption is mining-only)" 0 "true" "stage_45"

# Test AG8 (scope): stage_2 gate still enforced for round 2 debate (v4.1) —
# the exemption does not touch stage_2. The gate keys off a COMPLETED round-2
# re-mine (an un-mined round N>1 defers the stage_2 audit until mining lands),
# so round 2 carries fully-enriched completed mining here.
write_two_round '{"stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' '{}' "3_debate" "4.1"
jq '.campaign.rounds[1].bottleneck_mining = {
      "started_at": "2026-06-09T15:00:00Z", "completed_at": "2026-06-09T15:30:00Z",
      "top_bottleneck_share_pct": 30.0, "top_component": "fused_moe",
      "top_f_e2e_pct": 30.0, "top_addressable_e2e_pct": 10.0,
      "amdahl_ceiling": 1.4, "decode_frac": 0.78,
      "component_breakdown": [{"name": "fused_moe", "pct": 30.0}]
    }' "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
run_test "AG8: round 2 debate without stage_2 (v4.1) BLOCKS" 0 "true" "stage_2"

# Test AG9 (REGRESSION — fail-closed when predecessor has NO audit chain): round 2
# re-mine where the PREVIOUS round omits its `audit` key entirely (schema-valid —
# audit is optional) and round 2 carries audit={}. The same-round stage_1 exemption
# must NOT fire here, because the new-round-start gate is silent (PREV_AUDIT_EXISTS
# false) and would let a round begin mining with ZERO audit chain anywhere. The
# exemption is conditioned on the predecessor carrying an audit key, so this BLOCKS.
write_two_round 'null' '{}' "2_bottleneck_mining"
run_test "AG9: round 2 re-mine blocked when round 1 omits audit key entirely (fail-closed)" 0 "true" "stage_1"

# Test AG10 (boundary — predecessor audit:null still guards): round 2 re-mine where
# round 1 has audit:null (key PRESENT, no stage_67). The new-round-start gate fires
# (has("audit")=true) and BLOCKS on the missing stage_67. Distinguishes null (guarded)
# from absent (AG9). Either block reason is acceptable; assert it blocks.
write_two_round 'null' '{}' "2_bottleneck_mining"  # placeholder, overwritten below
base_v2 | jq '
    .campaign.current_round = 2
    | .campaign.current_stage = "2_bottleneck_mining"
    | .campaign.rounds[0].status = "completed"
    | .campaign.rounds[0].audit = null
    | .campaign.rounds += [ (.campaign.rounds[0]
        | .round_id = 2 | .status = "IN_PROGRESS"
        | .bottleneck_mining = {"started_at": null, "completed_at": null, "top_bottleneck_share_pct": null}
        | .audit = {}) ]
' > "$STATE_FILE"
run_test "AG10: round 2 re-mine with round 1 audit:null (key present) still BLOCKS" 0 "true"

# Test AG11 (post-fix, the intended path still works): round 2 re-mine, round 1 carries
# a full audit chain incl stage_67 (key present), round 2 audit={stage_2} / no stage_1.
# Exemption fires (predecessor has audit key + stage_67 satisfies new-round gate) → ALLOW.
write_two_round '{"stage_1": {"passed_at": "2026-06-08T14:56:31Z"}, "stage_2": {"passed_at": "2026-06-08T15:48:49Z"}, "stage_45": {"passed_at": "2026-06-08T18:26:35Z"}, "stage_67": {"passed_at": "2026-06-08T18:42:23Z"}}' '{"stage_2": {"passed_at": "2026-06-09T15:35:00Z"}}' "2_bottleneck_mining"
run_test "AG11: round 2 re-mine ALLOWS when predecessor carries full chain incl stage_67" 0 "false"

# Test AG12: diluted=true with status=FAIL is a structurally invalid combination -> BLOCK.
# The v4.2 schema's explicit diluted:true → status:PASS conditional catches this
# at schema level ("'PASS' was expected"); the engine cross-field rule ("diluted")
# is the backstop. Either block is correct.
write_state '.campaign.rounds[0].parallel_tracks.tracks = {"op-001": {"status": "FAIL", "diluted": true}}'
run_test "AG12: diluted=true with status=FAIL BLOCKS" 0 "true" "diluted|'PASS' was expected"

# Test AG13: diluted=true with status=PASS is the only valid combination -> ALLOW.
write_state '.campaign.rounds[0].parallel_tracks.tracks = {"op-001": {"status": "PASS", "diluted": true}}'
run_test "AG13: diluted=true with status=PASS ALLOWS" 0 "false"

# Test AG14: diluted=false (explicit) with status=FAIL must NOT trigger the rule -> ALLOW.
# Guards the truthiness contract: A18 fires ONLY on `diluted is True`, so an explicit
# `diluted:false` on a legitimately-FAILed track is a normal, valid combination. A future
# refactor loosening the check to `if not diluted` / `if diluted != True` would silently
# regress this without a test to catch it.
write_state '.campaign.rounds[0].parallel_tracks.tracks = {"op-001": {"status": "FAIL", "diluted": false}}'
run_test "AG14: diluted=false with status=FAIL ALLOWS" 0 "false"

# ══════════════════════════════════════════════════
echo ""
echo "== mining-enrichment round-advance gate (replaces schema allOf hard-block) =="
# ══════════════════════════════════════════════════
# The former schema allOf made the six enrichment fields non-null whenever
# mining.completed_at was set — rejecting EVERY subsequent write once one round
# lacked them (legacy-round resume deadlock). The engine now blocks ONLY the
# write that advances current_stage past 2_bottleneck_mining; writes at or
# before mining, and unrelated writes on degraded legacy rounds, pass through.

# All six enrichment fields populated — used by the ALLOW leg.
mining_enriched='{"started_at": "2026-04-23T10:10:00Z", "completed_at": "2026-04-23T10:40:00Z", "top_bottleneck_share_pct": 30.0, "top_component": "fused_moe", "top_f_e2e_pct": 30.0, "top_addressable_e2e_pct": 10.0, "amdahl_ceiling": 30.0, "decode_frac": 0.78, "component_breakdown": [{"name": "fused_moe", "pct": 30.0}]}'
# completed_at set but enrichment fields absent (legacy round shape).
mining_unenriched='{"started_at": "2026-04-23T10:10:00Z", "completed_at": "2026-04-23T10:40:00Z", "top_bottleneck_share_pct": 30.0}'

# Test ME1 (deadlock fix): mining complete, enrichment MISSING, stage still at
# 2_bottleneck_mining → ALLOW (this exact write used to be schema-rejected).
write_state ".campaign.current_stage = \"2_bottleneck_mining\" | .campaign.rounds[0].bottleneck_mining = $mining_unenriched"
run_test "ME1: unenriched completed mining at stage 2 ALLOWS (no more all-writes block)" 0 "false"

# Test ME2 (gate fires at round-advance): same unenriched round, stage advanced
# to 3_debate → BLOCK with the backfill remediation.
write_state ".campaign.current_stage = \"3_debate\" | .campaign.rounds[0].bottleneck_mining = $mining_unenriched"
run_test "ME2: advancing to 3_debate with unenriched mining BLOCKS" 0 "true" "Round-advance blocked: bottleneck_mining enrichment incomplete"

# Test ME3 (enriched round advances freely): all six fields set, stage 3_debate → ALLOW.
write_state ".campaign.current_stage = \"3_debate\" | .campaign.rounds[0].bottleneck_mining = $mining_enriched"
run_test "ME3: advancing to 3_debate with fully enriched mining ALLOWS" 0 "false"

# Test ME4 (null counts as missing): fields present but decode_frac null → BLOCK, names the field.
write_state ".campaign.current_stage = \"3_debate\" | .campaign.rounds[0].bottleneck_mining = ($mining_enriched | .decode_frac = null)"
run_test "ME4: null decode_frac at 3_debate BLOCKS naming the field" 0 "true" "decode_frac"

# Test ME5 (mining not complete = no gate): stage past mining but completed_at
# null (e.g. EXHAUSTED round re-entering 3_debate without a re-mine) → ALLOW.
write_state '.campaign.current_stage = "3_debate"'
run_test "ME5: 3_debate with mining not completed (no re-mine round) ALLOWS" 0 "false"

# Test ME6 (gate applies deeper in the ladder too): unenriched round at 6_integration → BLOCK.
write_state ".campaign.current_stage = \"6_integration\" | .campaign.rounds[0].bottleneck_mining = $mining_unenriched"
run_test "ME6: 6_integration with unenriched mining BLOCKS" 0 "true" "Round-advance blocked"

# ══════════════════════════════════════════════════
echo ""
echo "== current_stage enum violations =="
# ══════════════════════════════════════════════════

# Test 13: Terminal pseudo-stage eliminated in v2
write_state '.campaign.current_stage = "campaign_complete"'
run_test "current_stage='campaign_complete' blocked (terminal pseudo-stage removed)" 0 "true" "is not one of"

# Test 14: Round-suffixed stage rejected
write_state '.campaign.current_stage = "2_bottleneck_mining_r2"'
run_test "Round-suffixed stage blocked" 0 "true" "is not one of"

# Test 15: Unknown stage
write_state '.campaign.current_stage = "nonsense_stage"'
run_test "Unknown stage 'nonsense_stage' blocked" 0 "true" "is not one of"

# Test 16: Missing current_stage blocked
write_state 'del(.campaign.current_stage)'
run_test "Missing campaign.current_stage blocked" 0 "true" "required property"

# ══════════════════════════════════════════════════
echo ""
echo "== campaign.current_round checks =="
# ══════════════════════════════════════════════════

# Test 17: Missing current_round
write_state 'del(.campaign.current_round)'
run_test "Missing campaign.current_round blocked" 0 "true" "required property"

# Test 18: current_round = 0 (below minimum)
write_state '.campaign.current_round = 0'
run_test "campaign.current_round=0 blocked (minimum=1)" 0 "true" "minimum"

# Test 19: current_round as string
write_state '.campaign.current_round = "two"'
run_test "campaign.current_round='two' blocked" 0 "true" "is not of type"

# ══════════════════════════════════════════════════
echo ""
echo "== campaign.config checks (NEW in v2) =="
# ══════════════════════════════════════════════════

# Test 20: Missing campaign.config
write_state 'del(.campaign.config)'
run_test "Missing campaign.config blocked" 0 "true" "required property"

# Test 21: Missing config.min_e2e_improvement_pct
write_state 'del(.campaign.config.min_e2e_improvement_pct)'
run_test "Missing config.min_e2e_improvement_pct blocked" 0 "true" "required property"

# Test 22: Non-numeric config.min_e2e_improvement_pct
write_state '.campaign.config.min_e2e_improvement_pct = "fast"'
run_test "Non-numeric config.min_e2e_improvement_pct blocked" 0 "true" "is not of type"

# ══════════════════════════════════════════════════
echo ""
echo "== rounds array checks (NEW in v2) =="
# ══════════════════════════════════════════════════

# Test 24: empty rounds[]
write_state '.campaign.rounds = []'
run_test "Empty campaign.rounds blocked (minItems=1)" 0 "true" "non-empty|too short|minItems"

# Test 25: missing rounds array entirely
write_state 'del(.campaign.rounds)'
run_test "Missing campaign.rounds blocked" 0 "true" "required property"

# ══════════════════════════════════════════════════
echo ""
echo "== campaign.status checks =="
# ══════════════════════════════════════════════════

# Test 26: Invalid campaign.status
write_state '.campaign.status = "in_progress"'
run_test "campaign.status='in_progress' blocked" 0 "true" "is not one of"

# Test 27-30: All four valid campaign.status values. Terminal statuses also
# need their cross-field terminal preconditions (stage_67 audit, terminal
# integration status, numeric Stage 7 stop inputs) — the enum itself is what
# these cases pin.
for status in "active" "paused"; do
    write_state '.campaign.status = "'"$status"'"'
    run_test "Valid campaign.status='$status' passes" 0 "false"
done
for status in "campaign_complete" "campaign_exhausted"; do
    integ="combined"
    if [ "$status" = "campaign_exhausted" ]; then integ="exhausted"; fi
    write_state '.campaign.status = "'"$status"'" | .campaign.current_stage = "7_campaign_eval"
      | .campaign.rounds[0].audit = {"stage_67": {"passed_at": "2026-04-23T12:00:00Z"}}
      | .campaign.rounds[0].integration.status = "'"$integ"'"
      | .campaign.rounds[0].bottleneck_mining.top_f_e2e_pct = 5.0
      | .campaign.rounds[0].bottleneck_mining.top_addressable_e2e_pct = 0.4'
    run_test "Valid campaign.status='$status' passes" 0 "false"
done

# ══════════════════════════════════════════════════
echo ""
echo "== Nested rounds[].* validation =="
# ══════════════════════════════════════════════════

# Test 31: invalid track status
write_state '.campaign.rounds[0].parallel_tracks.tracks = {"op-001": {"status": "track_complete"}}'
run_test "rounds[0].parallel_tracks.tracks[op].status='track_complete' blocked" 0 "true" "is not one of"

# Test 32: invalid track verdict
write_state '.campaign.rounds[0].parallel_tracks.tracks = {"op-001": {"status": "IN_PROGRESS", "verdict": "GPU_BLOCKED_TRIAL_RUN"}}'
run_test "tracks[op].verdict='GPU_BLOCKED_TRIAL_RUN' blocked" 0 "true" "is not one of"

# Test 33: invalid integration.status (nested)
write_state '.campaign.rounds[0].integration.status = "complete"'
run_test "rounds[0].integration.status='complete' blocked" 0 "true" "is not one of"

# Test 34: Missing required sub-object (parallel_tracks)
write_state 'del(.campaign.rounds[0].parallel_tracks)'
run_test "Missing rounds[0].parallel_tracks blocked" 0 "true" "required property"

# ══════════════════════════════════════════════════
echo ""
echo "== v1-shape rejection (additionalProperties:false) =="
# ══════════════════════════════════════════════════

# Test 36: v1 top-level parallel_tracks
write_state '. + {"parallel_tracks": {"op-001": {"status": "PASS"}}}'
run_test "v1 top-level parallel_tracks blocked (additionalProperties:false)" 0 "true" "Additional properties"

# Test 37: v1 top-level stage field
write_state '. + {"stage": "1_baseline"}'
run_test "v1 top-level 'stage' blocked" 0 "true" "Additional properties"

# Test 38: v1 top-level stage_timestamps
write_state '. + {"stage_timestamps": {}}'
run_test "v1 top-level 'stage_timestamps' blocked" 0 "true" "Additional properties"

# Test 39: v1 top-level debate
write_state '. + {"debate": {}}'
run_test "v1 top-level 'debate' blocked" 0 "true" "Additional properties"

# Test 40: v1 top-level schema_version
write_state '. + {"schema_version": "1.0"}'
run_test "v1 top-level 'schema_version' blocked" 0 "true" "Additional properties"

# ══════════════════════════════════════════════════
echo ""
echo "== Track A17: Stage 6 all-tracks-terminal guard =="
# ══════════════════════════════════════════════════

# Test 40a: Stage 6 with non-terminal track (IN_PROGRESS) blocked
write_state '.campaign.current_stage = "6_integration" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "op-001": {"status": "PASS"},
    "op-002": {"status": "IN_PROGRESS"}
  }'
run_test "Stage 6 blocked when a track is IN_PROGRESS" 0 "true" "Stage 6 transition blocked"

# Test 40b: Stage 6 with GATING_REQUIRED track blocked
write_state '.campaign.current_stage = "6_integration" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "op-001": {"status": "PASS"},
    "op-002": {"status": "GATING_REQUIRED"}
  }'
run_test "Stage 6 blocked when a track is GATING_REQUIRED" 0 "true" "Stage 6 transition blocked"

# Test 40c: Stage 6 with GPU_BLOCKED track blocked
write_state '.campaign.current_stage = "6_integration" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "op-001": {"status": "PASS"},
    "op-002": {"status": "GPU_BLOCKED"}
  }'
run_test "Stage 6 blocked when a track is GPU_BLOCKED" 0 "true" "Stage 6 transition blocked"

# Test 40d: Stage 6 with all terminal (PASS/GATED_PASS/FAIL) passes
write_state '.campaign.current_stage = "6_integration" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "op-001": {"status": "PASS"},
    "op-002": {"status": "GATED_PASS"},
    "op-003": {"status": "FAIL"}
  }'
run_test "Stage 6 passes when all tracks terminal (PASS/GATED_PASS/FAIL)" 0 "false"

# Test 40e: Stage 6 with empty tracks (no impl yet) passes (vacuously terminal)
write_state '.campaign.current_stage = "6_integration" |
  .campaign.rounds[0].parallel_tracks.tracks = {}'
run_test "Stage 6 passes when tracks object is empty" 0 "false"

# Test 40f: Non-Stage-6 state with IN_PROGRESS track is NOT affected by guard
write_state '.campaign.current_stage = "4_5_parallel_tracks" |
  .campaign.rounds[0].parallel_tracks.tracks = {
    "op-001": {"status": "IN_PROGRESS"}
  }'
run_test "Stage 4-5 with IN_PROGRESS track NOT blocked by A17 guard" 0 "false"

# ══════════════════════════════════════════════════
echo ""
echo "== Edge cases =="
# ══════════════════════════════════════════════════

# Test 41: No schema found → fast bail
NOSCHEMA_DIR=$(mktemp -d)
mkdir -p "$NOSCHEMA_DIR/.git"
NOSCHEMA_STATE="$NOSCHEMA_DIR/kernel_opt_artifacts/noschema/state.json"
mkdir -p "$(dirname "$NOSCHEMA_STATE")"
cat > "$NOSCHEMA_STATE" << 'EOF'
{"target": {"model_id": "x"}, "campaign": {"status": "active", "current_round": 1}}
EOF
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$NOSCHEMA_STATE"'"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: Missing schema → fast bail (no block)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Missing schema → fast bail (no block)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$NOSCHEMA_DIR"

# Test 42: Malformed JSON → fail-open
cat > "$STATE_FILE" << 'EOF'
{this is not valid json
EOF
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ]; then
    echo "  PASS [$TOTAL]: Malformed state.json → fail-open (exit 0)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Malformed state.json → fail-open (expected exit 0, got $actual_exit)"
    FAIL=$((FAIL + 1))
fi

# Test 43: Empty object blocks on missing required fields
cat > "$STATE_FILE" << 'EOF'
{}
EOF
run_test "Empty state.json blocks on missing required fields" 0 "true" "required property"

# ══════════════════════════════════════════════════
echo ""
echo "== Bash tool detection (PostToolUse/Bash path) =="
# ══════════════════════════════════════════════════

# Helper: run hook with a Bash-style tool_input (command field instead of file_path)
run_bash_test() {
    local test_name="$1"
    local expected_exit="$2"
    local expected_block="$3"
    local command_str="$4"
    local expected_pattern="${5:-}"
    local actual_exit=0

    TOTAL=$((TOTAL + 1))
    _clear_prev_snapshot
    export CLAUDE_PROJECT_DIR="$TMPDIR"
    jq -c -n --arg cmd "$command_str" '{"tool_input": {"command": $cmd}}' | \
        bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
    unset CLAUDE_PROJECT_DIR

    if [ "$actual_exit" -ne "$expected_exit" ]; then
        echo "  FAIL [$TOTAL]: $test_name — exit code (expected=$expected_exit, got=$actual_exit)"
        echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
        echo "        stderr: $(cat "$TMPDIR/hook-stderr")"
        FAIL=$((FAIL + 1))
        return
    fi

    local has_block="false"
    if grep -q '"decision"' "$TMPDIR/hook-stdout" 2>/dev/null; then
        has_block="true"
    fi

    if [ "$has_block" != "$expected_block" ]; then
        echo "  FAIL [$TOTAL]: $test_name — block state (expected=$expected_block, got=$has_block)"
        echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ -n "$expected_pattern" ]; then
        if ! grep -qE "$expected_pattern" "$TMPDIR/hook-stdout" 2>/dev/null; then
            echo "  FAIL [$TOTAL]: $test_name — missing pattern '$expected_pattern'"
            echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    echo "  PASS [$TOTAL]: $test_name"
    PASS=$((PASS + 1))
}

# Test 49: Bash command with jq redirect to state.json — valid state → passes
write_state '.'
run_bash_test "Bash jq redirect to state.json (valid) passes" 0 "false" \
    "jq '.campaign.current_stage = \"3_debate\"' kernel_opt_artifacts/test_target/state.json > /tmp/s.json && mv /tmp/s.json kernel_opt_artifacts/test_target/state.json"

# Test 50: Bash command with jq redirect to state.json — invalid state → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Bash jq redirect to state.json (invalid current_stage) blocks" 0 "true" \
    "jq '.campaign.current_stage = \"campaign_complete\"' kernel_opt_artifacts/test_target/state.json > /tmp/s.json && mv /tmp/s.json kernel_opt_artifacts/test_target/state.json" \
    "is not one of"

# Test 51: Bash command with Python open('w') to state.json — invalid → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Bash Python open('w') to state.json (invalid) blocks" 0 "true" \
    "python3 -c \"import json; state['campaign']['current_stage']='campaign_complete'; open('kernel_opt_artifacts/test_target/state.json','w').write(json.dumps(state))\"" \
    "is not one of"

# Test 52: Bash command with json.dump to state.json — invalid → blocks
write_state '.campaign.current_stage = "not_real_stage"'
run_bash_test "Bash json.dump to state.json (invalid) blocks" 0 "true" \
    "python3 -c \"import json; json.dump(data, open('kernel_opt_artifacts/test_target/state.json','w'))\"" \
    "is not one of"

# Test 53: Bash command that only reads state.json (grep) → fast bail (no block)
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Bash read-only (grep state.json) is ignored" 0 "false" \
    "grep current_stage kernel_opt_artifacts/test_target/state.json"

# Test 54: Bash command that only reads state.json (cat) → fast bail
run_bash_test "Bash read-only (cat state.json) is ignored" 0 "false" \
    "cat kernel_opt_artifacts/test_target/state.json | jq .campaign.status"

# Test 55: Bash command that mentions state.json in jq without redirect → fast bail
run_bash_test "Bash read-only (jq without redirect) is ignored" 0 "false" \
    "jq '.campaign.current_round' kernel_opt_artifacts/test_target/state.json"

# Test 56: Bash command with mv into state.json — valid state → passes
write_state '.'
run_bash_test "Bash mv into state.json (valid) passes" 0 "false" \
    "mv /tmp/state_update.json kernel_opt_artifacts/test_target/state.json"

# Test 57: Bash command unrelated to state.json → fast bail
run_bash_test "Bash command not mentioning state.json is ignored" 0 "false" \
    "python3 benchmark.py --model test --batch-size 32"

# Test 58: Bash command with tee to state.json — invalid → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Bash tee to state.json (invalid) blocks" 0 "true" \
    "echo '{}' | tee kernel_opt_artifacts/test_target/state.json" \
    "is not one of"

# ══════════════════════════════════════════════════
echo ""
echo "== Bash: multi-target and false-positive prevention =="
# ══════════════════════════════════════════════════

# Test 59: Write to target_A when target_B has invalid state → should NOT block
mkdir -p "$TMPDIR/kernel_opt_artifacts/target_B"
write_state '.'  # target_A is valid (test_target)
base_v2 | jq '.campaign.current_stage = "campaign_complete"' > "$TMPDIR/kernel_opt_artifacts/target_B/state.json"
run_bash_test "Multi-target: write to valid target_A doesn't block on invalid target_B" 0 "false" \
    "jq '.campaign.current_round = 3' kernel_opt_artifacts/test_target/state.json > /tmp/a.json && mv /tmp/a.json kernel_opt_artifacts/test_target/state.json"
rm -rf "$TMPDIR/kernel_opt_artifacts/target_B"

# Test 60: Write to unrelated /tmp/state.json → should NOT trigger validation
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Unrelated /tmp/state.json write is ignored (no kernel_opt_artifacts path)" 0 "false" \
    "echo '{}' > /tmp/unrelated_state.json"

# Test 61: Write to /workspace/other/state.json (no kernel_opt_artifacts) → ignored
run_bash_test "Non-artifact state.json write is ignored" 0 "false" \
    "jq '.foo = 1' /workspace/other/state.json > /tmp/x.json && mv /tmp/x.json /workspace/other/state.json"

# Test 62: Failed Bash command (tool_response.status=error) → skip validation
write_state '.campaign.current_stage = "campaign_complete"'
TOTAL=$((TOTAL + 1))
export CLAUDE_PROJECT_DIR="$TMPDIR"
jq -c -n --arg cmd 'jq ".campaign.current_stage = \"bad\"" kernel_opt_artifacts/test_target/state.json > /tmp/s.json && mv /tmp/s.json kernel_opt_artifacts/test_target/state.json' \
    '{"tool_input": {"command": $cmd}, "tool_response": {"status": "error"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || true
unset CLAUDE_PROJECT_DIR
if grep -q '"decision"' "$TMPDIR/hook-stdout" 2>/dev/null; then
    echo "  FAIL [$TOTAL]: Failed Bash (status=error) still blocked"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
else
    echo "  PASS [$TOTAL]: Failed Bash (status=error) skips validation"
    PASS=$((PASS + 1))
fi

# Test 63: Multi-line python with json.dump writing to kernel_opt_artifacts state.json — blocks on invalid
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Multi-line python json.dump detects kernel_opt_artifacts path" 0 "true" \
    "python3 -c \"import json; state={'bad':1}; json.dump(state, open('kernel_opt_artifacts/test_target/state.json','w'), indent=2)\"" \
    "is not one of"

# ══════════════════════════════════════════════════
echo ""
echo "== Bypass: reconcile_track_state.py --write is a first-party writer =="
# ══════════════════════════════════════════════════
# parallel-tracks.md § Track State Reconciliation names
# `scripts/reconcile_track_state.py --write` as the canonical per-track lead
# write path, and the script itself calls no validator. It was absent from the
# LOOKS_LIKE_WRITE case list AND from the *state.json* fast filter, so the write
# reached state.json completely unvalidated. Every DENY case here is "no block"
# on the pre-fix hook; the Codex twin
# (post_tool_use_guard.py:_first_party_state_paths) already covered it.

write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "reconcile_track_state.py --write (invalid) blocks" 0 "true" \
    ".venv/bin/python .claude/skills/ammo/scripts/reconcile_track_state.py --artifact-dir kernel_opt_artifacts/test_target --track-id op001 --write" \
    "is not one of"

write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "reconcile_track_state.py --artifact-dir=X --write (invalid) blocks" 0 "true" \
    "python3 .claude/skills/ammo/scripts/reconcile_track_state.py --artifact-dir=kernel_opt_artifacts/test_target --write --track-id op001" \
    "is not one of"

write_state '.'
run_bash_test "reconcile_track_state.py --write (valid) passes" 0 "false" \
    ".venv/bin/python .claude/skills/ammo/scripts/reconcile_track_state.py --artifact-dir kernel_opt_artifacts/test_target --track-id op001 --write"

# Dry-run has no --write, so it is not a write and must not be validated.
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "reconcile_track_state.py WITHOUT --write is ignored (read-only)" 0 "false" \
    ".venv/bin/python .claude/skills/ammo/scripts/reconcile_track_state.py --artifact-dir kernel_opt_artifacts/test_target --track-id op001"

# A --write outside kernel_opt_artifacts must not be validated.
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "reconcile_track_state.py --write on a non-artifact dir is ignored" 0 "false" \
    ".venv/bin/python scripts/reconcile_track_state.py --artifact-dir /tmp/scratch --write"

# ══════════════════════════════════════════════════
echo ""
echo "== PREV_STATE snapshot: this hook owns the read-then-write order =="
# ══════════════════════════════════════════════════
# The snapshot write used to live in ammo-next-step-reminder.sh (ADVISORY), so a
# blocking gate's grandfathering baseline depended on an advisory hook's side
# effect and on PostToolUse array order. The blocking validator now owns both
# halves: read as the pre-write baseline, then refresh. On the pre-fix hook the
# snapshot is never created, so case 1 below finds no file.

TOTAL=$((TOTAL + 1))
_clear_prev_snapshot
write_state '.'
export CLAUDE_PROJECT_DIR="$TMPDIR"
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}, "session_id": "snapsid"}' \
    | bash "$HOOK" >"$TMPDIR/hook-stdout" 2>/dev/null || true
unset CLAUDE_PROJECT_DIR
_SNAP="$AMMO_REMINDER_STATE_DIR/ammo-state-prev-snapsid.json"
if [ -f "$_SNAP" ] && jq -e '.campaign' "$_SNAP" >/dev/null 2>&1; then
    echo "  PASS [$TOTAL]: validator writes the PREV_STATE snapshot after validate"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: no PREV_STATE snapshot at $_SNAP after a validated write"
    FAIL=$((FAIL + 1))
fi

# The snapshot must hold the POST-write state, so the NEXT call sees this write
# as the baseline rather than re-judging it.
TOTAL=$((TOTAL + 1))
if [ -f "$_SNAP" ] && \
   [ "$(jq -r '.campaign.current_stage' "$_SNAP" 2>/dev/null)" = "$(jq -r '.campaign.current_stage' "$STATE_FILE" 2>/dev/null)" ]; then
    echo "  PASS [$TOTAL]: snapshot content equals the state just validated"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: snapshot content does not match the validated state"
    FAIL=$((FAIL + 1))
fi

# No stray .tmp.<pid> files may survive — the refresh is atomic.
TOTAL=$((TOTAL + 1))
if [ -z "$(find "$AMMO_REMINDER_STATE_DIR" -name '*.tmp.*' 2>/dev/null)" ]; then
    echo "  PASS [$TOTAL]: snapshot refresh leaves no temp files"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: temp files left behind by the snapshot refresh"
    FAIL=$((FAIL + 1))
fi
_clear_prev_snapshot

# ══════════════════════════════════════════════════
echo ""
echo "== Bash: path-in-variable + atomic-replace write idioms (live-campaign gap) =="
# ══════════════════════════════════════════════════
# A live campaign wrote invalid state via:
#   path = '...state.json'; s = json.load(open(path)); ...; json.dump(s, tmp);
#   os.replace(tmp.name, path)
# None of the ordered *'json.dump'*state.json* patterns matched (state.json
# precedes json.dump in the command text; the dump target is a variable).
# These pin the widened unordered idioms.

# Test 63a: heredoc with path variable + tempfile + os.replace — invalid → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Heredoc path-var + NamedTemporaryFile + os.replace (invalid) blocks" 0 "true" \
    "python3 - <<'PY'
import json, os, tempfile
path = 'kernel_opt_artifacts/test_target/state.json'
s = json.load(open(path))
s['campaign']['current_stage'] = 'campaign_complete'
tmp = tempfile.NamedTemporaryFile('w', dir=os.path.dirname(path), delete=False, suffix='.tmp')
json.dump(s, tmp, indent=2); tmp.close()
os.replace(tmp.name, path)
PY" \
    "is not one of"

# Test 63b: json.dump AFTER state.json mention (unordered) — invalid → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "json.dump after state.json mention (unordered) blocks" 0 "true" \
    "python3 -c \"import json; p='kernel_opt_artifacts/test_target/state.json'; d=json.load(open(p)); json.dump(d, open(p,'w'))\"" \
    "is not one of"

# Test 63c: pathlib write_text with path variable — invalid → blocks
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "pathlib write_text via path variable (invalid) blocks" 0 "true" \
    "python3 -c \"import json, pathlib; p=pathlib.Path('kernel_opt_artifacts/test_target/state.json'); d=json.loads(p.read_text()); p.write_text(json.dumps(d))\"" \
    "is not one of"

# Test 63d: ammo_state.py set CLI invocation — invalid persisted state → blocks
# (the engine validates before writing, but the hook must STILL fire as the
# outer safety net — e.g. engine run with schema discovery failing)
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "ammo_state.py set CLI counts as a write (invalid state blocks)" 0 "true" \
    "python3 .claude/skills/ammo/scripts/ammo_state.py set --state kernel_opt_artifacts/test_target/state.json --field campaign.current_round --value 2" \
    "is not one of"

# Test 63e: read-only commands stay ignored despite widened idioms
write_state '.campaign.current_stage = "campaign_complete"'
run_bash_test "Read-only json.load + print (no write idiom) still ignored" 0 "false" \
    "python3 -c \"import json; print(json.load(open('kernel_opt_artifacts/test_target/state.json'))['campaign']['status'])\""

# ══════════════════════════════════════════════════
echo ""
echo "== P4#6: fail-CLOSED validation (default-on; decision:block; escape hatch) =="
# ══════════════════════════════════════════════════
# These pin the Phase-4 flip: when validation CANNOT run (missing python3/engine,
# missing jsonschema LIB, or unparseable state JSON), the hook/engine emits a
# decision:block on stdout (NOT an exit code) by DEFAULT, with escape hatch
# AMMO_VALIDATE_FAIL_OPEN=1. The missing-schema-FILE fast-bail (Test 41) stays
# fail-OPEN — re-asserted below. Append-only; all prior 90 cases are untouched.

# Build a PATH dir holding jq+coreutils but deliberately NO python3, to exercise
# the hook-side missing-python3 flip (PATH scrub is the verified mechanism).
make_nopython_bin() {
    local d
    d=$(mktemp -d)
    local t src
    for t in jq cat dirname grep head mktemp sed tr cut env bash sh ls rm mkdir cp; do
        src=$(command -v "$t" 2>/dev/null) && ln -s "$src" "$d/$t"
    done
    echo "$d"
}

# Make a throwaway dir whose sitecustomize.py forces `import jsonschema` to fail
# at interpreter startup (sets sys.modules["jsonschema"]=None BEFORE ammo_state's
# module-level import) → engine sees _HAVE_JSONSCHEMA=False. This is the VERIFIED
# shim; PYTHONPATH=/dev/null does NOT work (it prepends, site-packages still wins).
make_jsonschema_shim() {
    local d
    d=$(mktemp -d)
    printf 'import sys\nsys.modules["jsonschema"]=None\n' > "$d/sitecustomize.py"
    echo "$d"
}

# Test FC1: missing python3 (PATH scrub) over a valid state write → BLOCK (FLIP).
write_state '.'
NOPY=$(make_nopython_bin)
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    PATH="$NOPY" bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && grep -q '"decision":"block"' "$TMPDIR/hook-stdout" && grep -q 'AMMO_VALIDATE_FAIL_OPEN=1' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC1 missing python3 → decision:block + remediation (FLIP)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC1 missing python3 → decision:block + remediation (FLIP)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$NOPY"

# Test FC2: missing python3 + AMMO_VALIDATE_FAIL_OPEN=1 → NO block (escape hatch).
write_state '.'
NOPY=$(make_nopython_bin)
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    PATH="$NOPY" AMMO_VALIDATE_FAIL_OPEN=1 bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC2 missing python3 + FAIL_OPEN=1 → no block (escape hatch)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC2 missing python3 + FAIL_OPEN=1 → no block (escape hatch)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$NOPY"

# Test FC3: jsonschema LIB missing (sitecustomize shim) over a valid state → BLOCK.
write_state '.'
SHIM=$(make_jsonschema_shim)
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    PYTHONPATH="$SHIM" bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && grep -q '"decision":"block"' "$TMPDIR/hook-stdout" && grep -q 'jsonschema' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC3 jsonschema lib missing → decision:block (sitecustomize shim, FLIP)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC3 jsonschema lib missing → decision:block (sitecustomize shim, FLIP)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$SHIM"

# Test FC4: jsonschema LIB missing + AMMO_VALIDATE_FAIL_OPEN=1 → NO block.
write_state '.'
SHIM=$(make_jsonschema_shim)
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    PYTHONPATH="$SHIM" AMMO_VALIDATE_FAIL_OPEN=1 bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC4 jsonschema lib missing + FAIL_OPEN=1 → no block (escape hatch)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC4 jsonschema lib missing + FAIL_OPEN=1 → no block (escape hatch)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$SHIM"

# Test FC5: unparseable state JSON (schema present) → BLOCK by default (JSONDecodeError FLIP).
cat > "$STATE_FILE" << 'EOF'
{this is not valid json
EOF
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && grep -q '"decision":"block"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC5 corrupt state JSON → decision:block (JSONDecodeError FLIP)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC5 corrupt state JSON → decision:block (JSONDecodeError FLIP)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi

# Test FC6: unparseable state JSON + AMMO_VALIDATE_FAIL_OPEN=1 → NO block (legacy fail-open).
cat > "$STATE_FILE" << 'EOF'
{this is not valid json
EOF
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    AMMO_VALIDATE_FAIL_OPEN=1 bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC6 corrupt state JSON + FAIL_OPEN=1 → no block (escape hatch)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC6 corrupt state JSON + FAIL_OPEN=1 → no block (escape hatch)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi

# Test FC7: missing schema FILE → STILL NO block (fast-bail preserved; mirrors Test 41).
# The :98 `[ -z "$SCHEMA" ] && exit 0` schema-file fast-bail is NOT a flip case.
FC7_DIR=$(mktemp -d)
mkdir -p "$FC7_DIR/.git"
FC7_STATE="$FC7_DIR/kernel_opt_artifacts/noschema/state.json"
mkdir -p "$(dirname "$FC7_STATE")"
base_v2 > "$FC7_STATE"
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$FC7_STATE"'"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC7 missing schema FILE → no block (fast-bail intact, NOT a flip)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC7 missing schema FILE → no block (fast-bail intact, NOT a flip)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi
rm -rf "$FC7_DIR"

# Test FC8: valid state + jsonschema present (normal env) → silent pass (over-block guard).
write_state '.'
TOTAL=$((TOTAL + 1))
actual_exit=0
echo '{"tool_input": {"file_path": "'"$STATE_FILE"'"}}' | \
    bash "$HOOK" >"$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -q '"decision"' "$TMPDIR/hook-stdout"; then
    echo "  PASS [$TOTAL]: FC8 valid state + lib present → silent pass (over-block guard)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: FC8 valid state + lib present → silent pass (over-block guard)"
    echo "        stdout: $(cat "$TMPDIR/hook-stdout")"
    FAIL=$((FAIL + 1))
fi

# ══════════════════════════════════════════════════
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
