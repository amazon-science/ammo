#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-artifact-layout-warn.sh (PostToolUse, matcher=Write|Edit|Bash)
# Run: bash .claude/hooks/test-ammo-artifact-layout-warn.sh
#
# The hook is ADVISORY: it always exits 0 and signals only through
# hookSpecificOutput.additionalContext ("LAYOUT WARN: ..."). So every case here
# asserts on STDOUT presence/absence, not on the exit code.
#
# It derives its allowlist from skills/ammo/scripts/artifact_layout.json. Three
# properties matter and each has cases below:
#   1. every canonical slot in the manifest is SILENT (the four regressions that
#      motivated the manifest — evidence.json, validation_summary.json,
#      diff.patch, post_ship_profiling/ — are pinned explicitly),
#   2. the doc's Prohibited Patterns still WARN,
#   3. a missing or corrupt manifest disables checking LOUDLY on stderr and
#      never degrades to an empty allowlist that warns on everything.
set -uo pipefail

HOOK="${AMMO_TEST_HOOK:-$(cd "$(dirname "$0")" && pwd)/ammo-artifact-layout-warn.sh}"
MANIFEST="$(cd "$(dirname "$0")" && pwd)/../skills/ammo/scripts/artifact_layout.json"
ART="/w/kernel_opt_artifacts/tgt"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# run_case: $1 label  $2 silent|warn  $3 payload JSON
run_case() {
    local label="$1" want="$2" payload="$3"
    local rc=0
    TOTAL=$((TOTAL + 1))
    printf '%s' "$payload" | bash "$HOOK" \
        >"$TMPDIR/out" 2>"$TMPDIR/err" || rc=$?

    local pass=true
    [ "$rc" -ne 0 ] && pass=false          # advisory: always exit 0
    if grep -q "LAYOUT WARN:" "$TMPDIR/out" 2>/dev/null; then
        [ "$want" = "warn" ] || pass=false
    else
        [ "$want" = "silent" ] || pass=false
    fi

    if $pass; then
        echo "  PASS [$TOTAL]: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $label (want $want, exit $rc)"
        echo "        stdout: $(head -c 200 "$TMPDIR/out")"
        FAIL=$((FAIL + 1))
    fi
}

write_case() { run_case "$1" "$2" "$(jq -n -c --arg p "$ART/$3" \
    '{tool_name:"Write",tool_input:{file_path:$p}}')"; }

bash_case() { run_case "$1" "$2" "$(jq -n -c --arg c "$3" \
    '{tool_name:"Bash",tool_input:{command:$c}}')"; }

# ══════════════════════════════════════════════
echo "== The four slots the hand-written allowlist warned on =="
# ══════════════════════════════════════════════
write_case "evidence.json is canonical" silent "rounds/1/tracks/op-001/evidence.json"
write_case "per-track validation_summary.json is canonical" silent "rounds/1/tracks/op-001/validation_summary.json"
write_case "diff.patch is canonical" silent "rounds/1/tracks/OP-001/diff.patch"
write_case "post_ship_profiling slot is canonical" silent "rounds/1/sweeps/post_ship_profiling/json/x.json"

# ══════════════════════════════════════════════
echo ""; echo "== Rest of the canonical surface =="
# ══════════════════════════════════════════════
write_case "state.json at root" silent "state.json"
write_case "target.json at root" silent "target.json"
write_case "REPORT.md at root" silent "REPORT.md"
write_case "report_assets subtree" silent "report_assets/fig1.png"
write_case "blockers are cross-round" silent "blockers/stage_5_2026-05-05.md"
write_case "round constraints" silent "rounds/3/constraints.md"
write_case "cohort gate report" silent "rounds/3/validation_gate_report.json"
write_case "nsys traces" silent "rounds/1/profiling/nsys/baseline_bs8.nsys-rep"
write_case "opt sweep with op_id" silent "rounds/1/sweeps/opt/op007/e2e_latency_results.json"
write_case "opt_correctness sweep" silent "rounds/1/sweeps/opt_correctness/op007/json/correctness_verdict.json"
write_case "mined.json" silent "rounds/1/mining/mined.json"
write_case "mine_config.json" silent "rounds/1/mining/mine_config.json"
write_case "debate sub-round argument" silent "rounds/1/debate/round_2/op007_argument.md"
write_case "runtime_pkg.patch" silent "rounds/1/tracks/op-001/runtime_pkg.patch"
write_case "monitor audits under the track" silent "rounds/1/tracks/op-001/monitor_audits/m1_observations.md"
write_case "track scratch" silent "rounds/1/tracks/op-001/_scratch/draft.md"
write_case "audit verdict" silent "rounds/1/audits/stage_45.md"
write_case "audit re-audit cycle" silent "rounds/12/audits/stage_45_cycle_2.md"
write_case "early per-track S45 evidence" silent "rounds/1/audits/stage_45_partial_op007.md"
write_case "archive keeps its timestamp" silent "rounds/1/_archive/baseline_2026-05-05T181212Z/x.json"
write_case "two-digit round number" silent "rounds/12/sweeps/baseline/e2e_latency_results.json"

# ══════════════════════════════════════════════
echo ""; echo "== Prohibited Patterns still warn =="
# ══════════════════════════════════════════════
write_case "monitor log at campaign root" warn "monitor_log_champion_1.md"
write_case "DRAFT verdict at track root" warn "rounds/1/tracks/op-001/validation_results_DRAFT.md"
write_case "ad-hoc e2e_latency_opt dir" warn "e2e_latency_opt3/x.json"
write_case "removed investigation/ dir" warn "investigation/notes.md"
write_case "timestamped ACTIVE slot" warn "rounds/1/sweeps/baseline_2026-05-05T181212Z/x.json"
write_case "bottleneck_analysis at root" warn "bottleneck_analysis.md"
# KNOWN GAP, pre-dates the manifest: a sweep slot is a directory PREFIX, so
# anything under it conforms and prohibited pattern #9 (nsys traces inside a
# sweep output dir) is not detected here. The hand-written allowlist had the
# same prefix semantics. The sweep script is the real owner — it writes traces
# to the profiling/ sibling itself (_v2_profiling_dir), so an agent cannot
# reach this path through the sanctioned tool.
write_case "nsys inside a sweep dir is NOT detected (documented gap)" silent \
    "rounds/1/sweeps/baseline/nsys/t.nsys-rep"
write_case "debate nesting by campaign round" warn "rounds/1/debate/campaign_round_2/x.md"
write_case "audits keyed by round at root" warn "audits/stage_45_round_2.md"

# ══════════════════════════════════════════════
echo ""; echo "== Directory tokens are references, not drift =="
# ══════════════════════════════════════════════
# The Bash extractor yields bare directory tokens from ordinary ls/grep, and a
# proper ancestor of a canonical slot must stay silent.
bash_case "ls a round dir" silent "ls $ART/rounds/12"
bash_case "ls the sweeps dir" silent "ls $ART/rounds/1/sweeps"
bash_case "ls the opt slot parent" silent "ls $ART/rounds/1/sweeps/opt"
bash_case "ls the tracks dir" silent "ls $ART/rounds/1/tracks"
bash_case "repo-relative token (no leading slash)" silent "cat kernel_opt_artifacts/tgt/rounds/1/mining/mined.json"
bash_case "bash drift is still caught" warn "mkdir -p $ART/monitoring/logs"

# ══════════════════════════════════════════════
echo ""; echo "== Out of scope =="
# ══════════════════════════════════════════════
run_case "path outside kernel_opt_artifacts" silent \
    '{"tool_name":"Write","tool_input":{"file_path":"/etc/hosts"}}'
run_case "artifact root itself has no {target} segment" silent \
    '{"tool_name":"Write","tool_input":{"file_path":"/w/kernel_opt_artifacts/state.json"}}'
run_case "unhandled tool name" silent '{"tool_name":"Read","tool_input":{"file_path":"/x"}}'
run_case "missing file_path" silent '{"tool_name":"Write","tool_input":{}}'

# ══════════════════════════════════════════════
echo ""; echo "== Manifest failure is LOUD and fails open =="
# ══════════════════════════════════════════════
# A silent empty allowlist would warn on every correct write, which is worse
# than not checking: it trains the reader to ignore the channel.
STAGE="$TMPDIR/stage"
mkdir -p "$STAGE/hooks" "$STAGE/skills/ammo/scripts"
cp "$HOOK" "$STAGE/hooks/"
STAGED_HOOK="$STAGE/hooks/$(basename "$HOOK")"
GOOD_PAYLOAD=$(jq -n -c --arg p "$ART/state.json" '{tool_name:"Write",tool_input:{file_path:$p}}')
DRIFT_PAYLOAD=$(jq -n -c --arg p "$ART/monitor_log_x.md" '{tool_name:"Write",tool_input:{file_path:$p}}')

check_disabled() {
    local label="$1" payload="$2"
    local rc=0
    TOTAL=$((TOTAL + 1))
    printf '%s' "$payload" | bash "$STAGED_HOOK" \
        >"$TMPDIR/out" 2>"$TMPDIR/err" || rc=$?
    local pass=true
    [ "$rc" -ne 0 ] && pass=false
    grep -q "LAYOUT WARN:" "$TMPDIR/out" 2>/dev/null && pass=false
    grep -q "LAYOUT WARN HOOK DISABLED" "$TMPDIR/err" 2>/dev/null || pass=false
    if $pass; then
        echo "  PASS [$TOTAL]: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $label (exit $rc)"
        echo "        stdout: $(head -c 160 "$TMPDIR/out")"
        echo "        stderr: $(head -c 160 "$TMPDIR/err")"
        FAIL=$((FAIL + 1))
    fi
}

check_disabled "missing manifest: loud stderr, no warn on a canonical path" "$GOOD_PAYLOAD"
check_disabled "missing manifest: loud stderr, no warn on real drift either" "$DRIFT_PAYLOAD"

printf '{ not json' > "$STAGE/skills/ammo/scripts/artifact_layout.json"
check_disabled "corrupt manifest: loud stderr, checking disabled" "$GOOD_PAYLOAD"

printf '{"placeholders":{},"slots":{}}' > "$STAGE/skills/ammo/scripts/artifact_layout.json"
check_disabled "empty slots: loud stderr, never an empty allowlist" "$GOOD_PAYLOAD"

# Restoring a good manifest must re-enable checking, so a disabled hook is a
# transient state and not a latch.
cp "$MANIFEST" "$STAGE/skills/ammo/scripts/artifact_layout.json"
TOTAL=$((TOTAL + 1))
restore_rc=0
printf '%s' "$DRIFT_PAYLOAD" | bash "$STAGED_HOOK" \
    >"$TMPDIR/out" 2>"$TMPDIR/err" || restore_rc=$?
if [ "$restore_rc" -eq 0 ] && grep -q "LAYOUT WARN:" "$TMPDIR/out" 2>/dev/null; then
    echo "  PASS [$TOTAL]: restored manifest re-enables checking"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: restored manifest did not re-enable checking (exit $restore_rc)"
    FAIL=$((FAIL + 1))
fi

# ══════════════════════════════════════════════
echo ""; echo "== jq unavailable =="
# ══════════════════════════════════════════════
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env grep printf python3 dirname pwd test rm; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
TOTAL=$((TOTAL + 1))
nojq_rc=0
printf '%s' "$DRIFT_PAYLOAD" | env PATH="$MIN_PATH" bash "$HOOK" \
    >"$TMPDIR/out" 2>/dev/null || nojq_rc=$?
if [ "$nojq_rc" -eq 0 ] && ! grep -q "LAYOUT WARN:" "$TMPDIR/out" 2>/dev/null; then
    echo "  PASS [$TOTAL]: no jq → silent fail-open"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: no jq → expected silent exit 0, got $nojq_rc"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
