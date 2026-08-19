#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-next-step-reminder.sh (PostToolUse hook).
#
# Covers the 14-state lookup table (Stage 1-7 + terminal campaign states)
# plus subagent filtering, throttling, and graceful degradation.
#
# Run: bash .claude/hooks/test-ammo-next-step-reminder.sh
set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-next-step-reminder.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() {
    rm -rf "$TMPDIR" "$TMPDIR/hook-stdout" "$TMPDIR/hook-stderr" 2>/dev/null || true
    # Throttle + worktree dedup markers created during this run
    rm -f /tmp/ammo-reminder-last-test-* 2>/dev/null || true
    rm -f /tmp/ammo-worktree-warned-test-* 2>/dev/null || true
    # PREV_STATE snapshots seeded per-case (see run_test_env)
    rm -f /tmp/ammo-state-prev-test-* 2>/dev/null || true
}
trap cleanup EXIT

# ─────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────

# make_orchestrator_transcript: no agentName in first 5 lines
make_orchestrator_transcript() {
    local file="$1"
    echo '{"type":"user","message":{"role":"user","content":"hi"}}' > "$file"
}

# make_subagent_transcript: agentName=<name> on line 2
make_subagent_transcript() {
    local file="$1" agent_name="${2:-champion-1}"
    echo '{"type":"permission-mode"}' > "$file"
    echo "{\"agentName\":\"$agent_name\",\"type\":\"user\"}" >> "$file"
}

truthy() { case "${1:-}" in 1|true|yes|y|Y) return 0;; *) return 1;; esac; }

# Positional args (all optional after stage/status):
#   $1 stage, $2 status,
#   $3 baseline_done, $4 mining_done,
#   $5 team_name, $6 selected_count,
#   $7 tracks_started, $8 integ_started,
#   $9 shipped_count, $10 track_count, $11 all_tracks_terminal,
#  $12 next_baseline_started
make_state_round1() {
    local stage="$1" status="$2"
    local baseline_done="${3:-false}" mining_done="${4:-false}"
    local team_name="${5:-}" selected="${6:-0}"
    local tracks_started="${7:-false}" integ_started="${8:-false}"
    local shipped="${9:-0}" track_count="${10:-0}" all_terminal="${11:-false}"
    local next_baseline="${12:-false}"

    local BD MD TS IS NB
    if truthy "$baseline_done"; then BD='"2026-01-01T00:00:00Z"'; else BD="null"; fi
    if truthy "$mining_done";   then MD='"2026-01-01T00:00:00Z"'; else MD="null"; fi
    if truthy "$tracks_started";then TS='"2026-01-01T00:00:00Z"'; else TS="null"; fi
    if truthy "$integ_started"; then IS='"2026-01-01T00:00:00Z"'; else IS="null"; fi
    if truthy "$next_baseline"; then NB='"2026-01-01T00:00:00Z"'; else NB="null"; fi

    local tracks_json="{}"
    if [ "$track_count" -gt 0 ]; then
        local items=""
        local i
        for ((i=1; i<=track_count; i++)); do
            local st="PASS"
            if ! truthy "$all_terminal" && [ "$i" = "1" ]; then st="IMPLEMENTING"; fi
            [ -n "$items" ] && items="$items,"
            items="$items\"op00$i\":{\"status\":\"$st\"}"
        done
        tracks_json="{$items}"
    fi

    local sel_json="[]"
    if [ "$selected" -gt 0 ]; then
        local items="" i
        for ((i=1; i<=selected; i++)); do
            [ -n "$items" ] && items="$items,"
            items="$items\"op00$i\""
        done
        sel_json="[$items]"
    fi

    local ship_json="[]"
    if [ "$shipped" -gt 0 ]; then
        local items="" i
        for ((i=1; i<=shipped; i++)); do
            [ -n "$items" ] && items="$items,"
            items="$items\"op00$i\""
        done
        ship_json="[$items]"
    fi

    local team_json="null"
    [ -n "$team_name" ] && team_json="\"$team_name\""

    cat <<EOF
{"campaign":{"current_stage":"$stage","status":"$status","current_round":1,"config":{"min_e2e_improvement_pct":0.25,"noise_tolerance_pct":0.5,"catastrophic_regression_pct":5.0},"rounds":[{"baseline":{"completed_at":$BD},"bottleneck_mining":{"completed_at":$MD},"team_name":$team_json,"debate":{"selected_candidates":$sel_json},"parallel_tracks":{"started_at":$TS,"tracks":$tracks_json},"integration":{"started_at":$IS},"shipped":$ship_json},{"baseline":{"started_at":$NB}}]}}
EOF
}

make_proj() {
    local state_json="$1"
    local proj
    proj=$(mktemp -d -p "$TMPDIR" proj.XXXXXX)
    mkdir -p "$proj/kernel_opt_artifacts/run1"
    echo "$state_json" > "$proj/kernel_opt_artifacts/run1/state.json"
    echo "$proj"
}

# ─────────────────────────────────────────────
# Test runner (env-aware)
# ─────────────────────────────────────────────
# Counter lives in a file so increments survive $() subshells.
SID_COUNTER_FILE="$TMPDIR/.sid_counter"
echo 0 > "$SID_COUNTER_FILE"
next_sid() {
    local n
    n=$(< "$SID_COUNTER_FILE")
    n=$((n + 1))
    echo "$n" > "$SID_COUNTER_FILE"
    echo "test-$n-$$"
}

run_test_env() {
    local name="$1" expected_exit="$2" proj="$3" input="$4"
    local need="${5:-}" forbid="${6:-}"
    local got=0
    TOTAL=$((TOTAL + 1))

    # Seed PREV_STATE = current state so the edge-triggered SOCRATIC chains stay
    # quiescent and the steady-state REMINDER dispatch (the thing these cases
    # assert) is what fires. In production the orchestrator's state.json has
    # already evolved across many tool calls, so /tmp/ammo-state-prev-<sid>.json
    # essentially always exists; an unseeded harness instead trips the Stage 1→2
    # mining nudge on every fixture (baseline.completed_at reads as a fresh
    # transition) and that nudge shadows the REMINDER. Cases that assert a
    # SOCRATIC transition explicitly pass their own forbid/need and clear this.
    local _sid _state
    _sid=$(echo "$input" | jq -r '.session_id // empty' 2>/dev/null) || _sid=""
    _state="$proj/kernel_opt_artifacts/run1/state.json"
    if [ -n "$_sid" ] && [ -f "$_state" ]; then
        cp "$_state" "/tmp/ammo-state-prev-${_sid}.json" 2>/dev/null || true
    fi

    echo "$input" | env HOME="$TMPDIR" CLAUDE_PROJECT_DIR="$proj" bash "$HOOK" > "$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || got=$?

    local pass=true
    [ "$got" -ne "$expected_exit" ] && pass=false
    if [ -n "$need" ]; then
        grep -qF "$need" "$TMPDIR/hook-stdout" 2>/dev/null || pass=false
    fi
    if [ -n "$forbid" ]; then
        if grep -qF "$forbid" "$TMPDIR/hook-stdout" 2>/dev/null; then pass=false; fi
    fi
    if [ -n "$need" ] && [ -s "$TMPDIR/hook-stdout" ]; then
        jq . "$TMPDIR/hook-stdout" >/dev/null 2>&1 || { pass=false; echo "  WARN: invalid JSON"; }
    fi

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected=$expected_exit, got=$got)"
        [ -n "$need" ]   && echo "        required: $need"
        [ -n "$forbid" ] && echo "        forbidden: $forbid"
        echo "        stdout: $(head -3 "$TMPDIR/hook-stdout" 2>/dev/null)"
        echo "        stderr: $(head -3 "$TMPDIR/hook-stderr" 2>/dev/null)"
        FAIL=$((FAIL + 1))
    fi
}

ORCH_T="$TMPDIR/orch_transcript.jsonl"
make_orchestrator_transcript "$ORCH_T"

# ═════════════════════════════════════════════
echo "== 14-state lookup table =="
# ═════════════════════════════════════════════

# 1) 1_baseline pending → ammo-researcher task_type: baseline
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
run_test_env "[01] 1_baseline pending → task_type: baseline" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "task_type: baseline"

# 2) 1_baseline done → dispatch mining
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active true)")
run_test_env "[02] 1_baseline done → dispatch mining" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "mining"

# 3) 2_bottleneck_mining pending → dispatch mining
SID=$(next_sid); P=$(make_proj "$(make_state_round1 2_bottleneck_mining active true false)")
run_test_env "[03] 2_mining pending → ammo-researcher mining" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "mining"

# 4) 2_bottleneck_mining done → advance to debate
SID=$(next_sid); P=$(make_proj "$(make_state_round1 2_bottleneck_mining active true true)")
run_test_env "[04] 2_mining done → advance to debate" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Mining done"

# 5) 3_debate, no team_name → spawn named ammo-champion into implicit team (no TeamCreate)
SID=$(next_sid); P=$(make_proj "$(make_state_round1 3_debate active true true "" 0)")
run_test_env "[05] 3_debate no team → spawn named champions into implicit team" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "spawn 2-4 ammo-champion (named) into the implicit team (no TeamCreate)"

# 6) 3_debate, team exists, no selected candidates → debate in progress
SID=$(next_sid); P=$(make_proj "$(make_state_round1 3_debate active true true round1-team 0)")
run_test_env "[06] 3_debate in progress → min 1 round" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Min 1 round"

# 7) 3_debate, winners selected, no tracks_started → spawn ammo-impl-champion
SID=$(next_sid); P=$(make_proj "$(make_state_round1 3_debate active true true round1-team 2 false)")
run_test_env "[07] 3_debate winners → spawn ammo-impl-champion" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "ammo-impl-champion"

# 8) 4_5_parallel_tracks, some tracks non-terminal → wait
SID=$(next_sid); P=$(make_proj "$(make_state_round1 4_5_parallel_tracks active true true round1-team 2 true false 0 2 false)")
run_test_env "[08] 4_5 tracks non-terminal → wait for all" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Do NOT advance to Stage 6"

# 9) 4_5_parallel_tracks, all tracks terminal, integration not started → shut down round-team agents + Stage 6
SID=$(next_sid); P=$(make_proj "$(make_state_round1 4_5_parallel_tracks active true true round1-team 2 true false 0 2 true)")
run_test_env "[09] 4_5 all terminal → shut down round-team agents + integration" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Shut down round-team agents via SendMessage shutdown_request, confirm shutdown_approved, then proceed"

# 10) 6_integration, shipped==0, passing tracks exist → integration sweep with --fresh-cache
SID=$(next_sid); P=$(make_proj "$(make_state_round1 6_integration active true true round1-team 2 true true 0 2 true)")
run_test_env "[10] 6_integration shipped=0, tracks passing → integration sweep" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "fresh-cache"

# 11) 6_integration, shipped>0, no audit key → T_AUDIT_S67 passed (legacy bypass), advance to 7_campaign_eval
SID=$(next_sid); P=$(make_proj "$(make_state_round1 6_integration active true true round1-team 2 true true 1 2 true false)")
run_test_env "[11] 6_integration shipped>0 → advance to 7_campaign_eval" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "7_campaign_eval"

# 12) 7_campaign_eval → mechanical check
SID=$(next_sid); P=$(make_proj "$(make_state_round1 7_campaign_eval active true true round1-team 2 true true 1 2 true true)")
run_test_env "[12] 7_campaign_eval → mechanical check" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Mechanical check"

# 13) status=campaign_complete, REPORT.md present → session may stop
SID=$(next_sid); P=$(make_proj "$(make_state_round1 7_campaign_eval campaign_complete true true round1-team 2 true true 1 2 true true)")
touch "$P/kernel_opt_artifacts/run1/REPORT.md"
run_test_env "[13] campaign_complete + REPORT.md → session may stop" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "Session may stop"

# 14) status=campaign_exhausted, no REPORT.md → spawn ammo-report-writer
SID=$(next_sid); P=$(make_proj "$(make_state_round1 7_campaign_eval campaign_exhausted true true round1-team 2 true true 1 2 true true)")
run_test_env "[14] campaign_exhausted no report → spawn report-writer" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "ammo-report-writer"

# ═════════════════════════════════════════════
echo ""; echo "== Subagent filtering & throttle =="
# ═════════════════════════════════════════════

# Subagent via agent_type field → suppressed
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
run_test_env "[15] agent_type set → suppressed" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"agent_type\":\"ammo-champion\"}" \
    "" "additionalContext"

# Subagent via transcript agentName → suppressed
SUB_T="$TMPDIR/sub_transcript.jsonl"
make_subagent_transcript "$SUB_T" "champion-1"
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
run_test_env "[16] transcript agentName=champion-1 → suppressed" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$SUB_T\"}" \
    "" "additionalContext"

# agentName=team-lead should NOT suppress
TL_T="$TMPDIR/teamlead_transcript.jsonl"
make_subagent_transcript "$TL_T" "team-lead"
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
run_test_env "[17] transcript agentName=team-lead → fires" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$TL_T\"}" \
    "task_type: baseline"

# Throttle: second call with same session id within 30s → suppressed
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
echo "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" | \
    env HOME="$TMPDIR" CLAUDE_PROJECT_DIR="$P" bash "$HOOK" >/dev/null 2>&1 || true
run_test_env "[18] throttle (same session <30s) → suppressed" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

# No state.json at all → silent exit 0
P_EMPTY=$(mktemp -d -p "$TMPDIR" empty.XXXXXX)
SID=$(next_sid)
run_test_env "[19] no state.json → exit 0 silent" 0 "$P_EMPTY" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\"}" \
    "" "additionalContext"

# ═════════════════════════════════════════════
echo ""; echo "== Worktree cwd drift warning (W1-W7) =="
# ═════════════════════════════════════════════
# Warning keyed on session_id + worktree basename. First encounter warns
# instead of emitting the normal reminder. Subsequent encounters with the
# same (session, basename) fall through to normal reminder.

# Keep tests deterministic: each W-test gets a fresh SID (isolated throttle
# marker) and an isolated /tmp/ammo-worktree-warned-* namespace.
# Clean any stale markers from prior runs of this test file.
rm -f /tmp/ammo-worktree-warned-test-* 2>/dev/null || true

# W1: orchestrator in worktree op001 → warn
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
WT_CWD="$P/.claude/worktrees/op001/sub"
run_test_env "[W1] orchestrator in worktree op001 → CWD DRIFT warn" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\"}" \
    "CWD DRIFT"

# W2: orchestrator in main repo → normal reminder, no warning
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
run_test_env "[W2] orchestrator cwd=main repo → baseline capture (no drift warn)" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$P\"}" \
    "task_type: baseline" "CWD DRIFT"

# W3: subagent (agent_type set) in a worktree → no warning (subagent filter wins)
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
WT_CWD="$P/.claude/worktrees/op003/sub"
run_test_env "[W3] subagent agent_type=ammo-champion in worktree → suppressed" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\",\"agent_type\":\"ammo-champion\"}" \
    "" "CWD DRIFT"

# W4a/W4b: dedup — first call warns, second call with same (sid, basename) falls
# through to normal reminder. DA requires unsetting the 30s throttle marker
# between the two calls.
SID=$(next_sid); P=$(make_proj "$(make_state_round1 1_baseline active false)")
WT_CWD="$P/.claude/worktrees/op001/sub"
run_test_env "[W4a] 1st call in worktree op001 → CWD DRIFT warn" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\"}" \
    "CWD DRIFT"
rm -f "/tmp/ammo-reminder-last-$SID"  # W4b prep: clear throttle per DA
run_test_env "[W4b] 2nd call in worktree op001 (marker exists) → normal reminder" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\"}" \
    "task_type: baseline" "CWD DRIFT"

# W5: same session but a different worktree basename → warn (new dedup key).
# W4 already warmed up op001; now switch to op002 which has its own marker.
rm -f "/tmp/ammo-reminder-last-$SID"
WT_CWD2="$P/.claude/worktrees/op002/sub"
run_test_env "[W5] same session, new worktree op002 → CWD DRIFT warn" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD2\"}" \
    "CWD DRIFT"

# W6: campaign_complete + worktree → warn fires regardless of stage/status
SID=$(next_sid); P=$(make_proj "$(make_state_round1 7_campaign_eval campaign_complete true true round1-team 2 true true 1 2 true true)")
WT_CWD="$P/.claude/worktrees/op001/sub"
run_test_env "[W6] campaign_complete in worktree → CWD DRIFT warn" 0 "$P" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\"}" \
    "CWD DRIFT"

# W7: no state.json + worktree → still warn (runs before state.json lookup)
P_EMPTY2=$(mktemp -d -p "$TMPDIR" empty-wt.XXXXXX)
SID=$(next_sid)
WT_CWD="$P_EMPTY2/.claude/worktrees/op001/sub"
run_test_env "[W7] no state.json in worktree → CWD DRIFT warn" 0 "$P_EMPTY2" \
    "{\"session_id\":\"$SID\",\"transcript_path\":\"$ORCH_T\",\"cwd\":\"$WT_CWD\"}" \
    "CWD DRIFT"

# Cleanup worktree markers used by this suite
rm -f /tmp/ammo-worktree-warned-test-* 2>/dev/null || true

# ═════════════════════════════════════════════
echo ""; echo "== Summary =="
echo "Total: $TOTAL | Passed: $PASS | Failed: $FAIL"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
