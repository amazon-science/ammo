#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for the compaction pair: ammo-precompact.sh (PreCompact) and
# ammo-postcompact.sh (SessionStart).
# Run: bash .claude/hooks/test-ammo-compaction.sh
#
# The pair carries the campaign across a compaction, so both documents must be
# valid JSON for every state value an agent can write. Cases:
#   1. Plain state → valid checkpoint + valid briefing
#   2. round_summary with newline + double-quote + backslash (the reported bug)
#   3. model / team_name with a newline and a quote
#   4. Missing parallel_tracks → track_count stays numeric
#   5. Corrupt state.json → both hooks still emit valid JSON
#   6. No AMMO state → empty briefing document
#   7. Atomic write leaves no temp file behind
#   8. Symlink checkpoint is refused
#   9. Broken jq → both hooks still emit valid JSON

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PRE="$HERE/ammo-precompact.sh"
POST="$HERE/ammo-postcompact.sh"
PASS=0
FAIL=0
TOTAL=0

TMPROOT=$(mktemp -d)
cleanup() { rm -rf "$TMPROOT"; }
trap cleanup EXIT

# ─────────────────────────────────────────────
# new_sandbox: fresh CLAUDE_PROJECT_DIR with one artifact dir
# echoes the sandbox root
# ─────────────────────────────────────────────
new_sandbox() {
    local sb
    sb=$(mktemp -d "$TMPROOT/sb.XXXXXX")
    mkdir -p "$sb/kernel_opt_artifacts/tgt"
    echo "$sb"
}

STATE_REL="kernel_opt_artifacts/tgt/state.json"
CHECKPOINT_REL="kernel_opt_artifacts/tgt/compaction_checkpoint.json"

# ─────────────────────────────────────────────
# write_state: build a state.json with the given model / team / summary.
#   $1 = sandbox  $2 = model  $3 = team_name  $4 = round_summary
#   $5 = (optional) "no_tracks" to omit parallel_tracks entirely
# ─────────────────────────────────────────────
write_state() {
    local sb="$1" model="$2" team="$3" summary="$4" mode="${5:-}"
    MODEL="$model" TEAM="$team" SUMMARY="$summary" MODE="$mode" \
    python3 - "$sb/$STATE_REL" <<'PY'
import json, os, sys
rnd = {"team_name": os.environ["TEAM"], "round_summary": os.environ["SUMMARY"],
       "status": "in_progress"}
if os.environ.get("MODE") != "no_tracks":
    rnd["parallel_tracks"] = {"tracks": [{"id": "op-001"}, {"id": "op-002"}]}
doc = {"target": {"model_id": os.environ["MODEL"]},
       "campaign": {"current_round": 1, "current_stage": "implement",
                    "status": "in_progress", "cumulative_speedup_vs_round1": 1.23,
                    "rounds": [rnd]}}
with open(sys.argv[1], "w") as fh:
    json.dump(doc, fh, indent=2)
PY
}

report() {
    local ok="$1" name="$2" detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name ${detail:+— $detail}"
    fi
}

# ─────────────────────────────────────────────
# run_pair: run precompact then postcompact in $1; assert both JSON docs parse.
#   $2 = case name
# postcompact deletes the checkpoint, so a copy is kept in
# $sb/checkpoint.snapshot.json. The briefing lands in $sb/out.json.
# ─────────────────────────────────────────────
run_pair() {
    local sb="$1" name="$2" rc=0
    CLAUDE_PROJECT_DIR="$sb" bash "$PRE" </dev/null >/dev/null 2>&1 || rc=$?
    report "$([ "$rc" = "0" ] && echo 1 || echo 0)" "$name: precompact exit 0" "exit $rc"

    cp "$sb/$CHECKPOINT_REL" "$sb/checkpoint.snapshot.json" 2>/dev/null || true
    if jq empty "$sb/$CHECKPOINT_REL" 2>/dev/null; then
        report 1 "$name: checkpoint is valid JSON"
    else
        report 0 "$name: checkpoint is valid JSON" "$(head -5 "$sb/$CHECKPOINT_REL" 2>/dev/null)"
    fi

    rc=0
    CLAUDE_PROJECT_DIR="$sb" bash "$POST" </dev/null >"$sb/out.json" 2>/dev/null || rc=$?
    report "$([ "$rc" = "0" ] && echo 1 || echo 0)" "$name: postcompact exit 0" "exit $rc"

    if jq empty "$sb/out.json" 2>/dev/null; then
        report 1 "$name: briefing is valid JSON"
    else
        report 0 "$name: briefing is valid JSON" "$(head -5 "$sb/out.json")"
    fi
}

echo "=== Case 1: plain state ==="
SB=$(new_sandbox)
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "Round one shipped one optimization."
run_pair "$SB" "plain"
SUMMARY_OUT=$(jq -r '.hookSpecificOutput.additionalContext' "$SB/out.json")
case "$SUMMARY_OUT" in
    *"Round one shipped one optimization."*) report 1 "plain: summary reaches briefing" ;;
    *) report 0 "plain: summary reaches briefing" "$SUMMARY_OUT" ;;
esac

echo "=== Case 2: round_summary with newline + double-quote + backslash ==="
SB=$(new_sandbox)
ADVERSARIAL='Line one of the summary.
Line two says "quoted" and holds a backslash \ plus a tab	here.'
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "$ADVERSARIAL"
run_pair "$SB" "adversarial summary"
BRIEF=$(jq -r '.hookSpecificOutput.additionalContext' "$SB/out.json")
case "$BRIEF" in
    *'Line two says "quoted" and holds a backslash \ '*)
        report 1 "adversarial summary: text survives round-trip" ;;
    *) report 0 "adversarial summary: text survives round-trip" "$BRIEF" ;;
esac

echo "=== Case 3: model + team_name with newline and quote ==="
SB=$(new_sandbox)
write_state "$SB" 'Qwen "Q3"
32B \ fork' 'team
-r1' "Plain summary."
run_pair "$SB" "adversarial model/team"
MODEL_OUT=$(jq -r '.model' "$SB/checkpoint.snapshot.json")
case "$MODEL_OUT" in
    *'Qwen "Q3"'*) report 1 "adversarial model/team: model survives checkpoint" ;;
    *) report 0 "adversarial model/team: model survives checkpoint" "$MODEL_OUT" ;;
esac

echo "=== Case 4: no parallel_tracks → numeric track_count ==="
SB=$(new_sandbox)
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "No tracks yet." "no_tracks"
run_pair "$SB" "no tracks"
TC_TYPE=$(jq -r '.track_count | type' "$SB/checkpoint.snapshot.json")
report "$([ "$TC_TYPE" = "number" ] && echo 1 || echo 0)" "no tracks: track_count is a number" "type=$TC_TYPE"

echo "=== Case 5: corrupt state.json ==="
SB=$(new_sandbox)
printf '{ this is not json' > "$SB/$STATE_REL"
run_pair "$SB" "corrupt state"

echo "=== Case 6: no AMMO state → empty briefing ==="
SB=$(mktemp -d "$TMPROOT/sb.XXXXXX")
RC=0
CLAUDE_PROJECT_DIR="$SB" bash "$PRE" </dev/null >/dev/null 2>&1 || RC=$?
report "$([ "$RC" = "0" ] && echo 1 || echo 0)" "no state: precompact exit 0" "exit $RC"
RC=0
CLAUDE_PROJECT_DIR="$SB" bash "$POST" </dev/null >"$SB/out.json" 2>/dev/null || RC=$?
report "$([ "$RC" = "0" ] && echo 1 || echo 0)" "no state: postcompact exit 0" "exit $RC"
if jq -e '.hookSpecificOutput.additionalContext == ""' "$SB/out.json" >/dev/null 2>&1; then
    report 1 "no state: briefing is the empty document"
else
    report 0 "no state: briefing is the empty document" "$(head -5 "$SB/out.json")"
fi

echo "=== Case 7: atomic write leaves no temp file ==="
SB=$(new_sandbox)
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "Atomic write check."
CLAUDE_PROJECT_DIR="$SB" bash "$PRE" </dev/null >/dev/null 2>&1 || true
LEFTOVER=$(find "$SB/kernel_opt_artifacts/tgt" -name '.compaction_checkpoint.json.*' | wc -l)
report "$([ "$LEFTOVER" = "0" ] && echo 1 || echo 0)" "atomic write: no temp file left" "found $LEFTOVER"

echo "=== Case 8: symlink checkpoint is refused ==="
SB=$(new_sandbox)
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "Symlink refusal check."
VICTIM="$SB/victim.json"
printf '{"keep":"me"}' > "$VICTIM"
ln -s "$VICTIM" "$SB/$CHECKPOINT_REL"
CLAUDE_PROJECT_DIR="$SB" bash "$PRE" </dev/null >/dev/null 2>&1 || true
if [ "$(cat "$VICTIM")" = '{"keep":"me"}' ]; then
    report 1 "symlink: target left untouched"
else
    report 0 "symlink: target left untouched" "$(cat "$VICTIM")"
fi

echo "=== Case 9: broken jq → both hooks still emit valid JSON ==="
SB=$(new_sandbox)
write_state "$SB" "Qwen/Qwen3-32B" "team-r1" "Broken jq check."
FAKEBIN="$SB/fakebin"
mkdir -p "$FAKEBIN"
printf '#!/bin/sh\nexit 127\n' > "$FAKEBIN/jq"
chmod +x "$FAKEBIN/jq"
RC=0
CLAUDE_PROJECT_DIR="$SB" PATH="$FAKEBIN:$PATH" bash "$PRE" </dev/null >/dev/null 2>&1 || RC=$?
report "$([ "$RC" = "0" ] && echo 1 || echo 0)" "broken jq: precompact exit 0" "exit $RC"
RC=0
CLAUDE_PROJECT_DIR="$SB" PATH="$FAKEBIN:$PATH" bash "$POST" </dev/null >"$SB/out.json" 2>/dev/null || RC=$?
report "$([ "$RC" = "0" ] && echo 1 || echo 0)" "broken jq: postcompact exit 0" "exit $RC"
if jq empty "$SB/out.json" 2>/dev/null; then
    report 1 "broken jq: briefing is valid JSON"
else
    report 0 "broken jq: briefing is valid JSON" "$(head -3 "$SB/out.json")"
fi

echo ""
echo "=== Summary ==="
echo "$PASS / $TOTAL passed"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED: $FAIL"
    exit 1
fi
exit 0
