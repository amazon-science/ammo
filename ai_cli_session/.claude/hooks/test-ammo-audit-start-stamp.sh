#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Test harness for ammo-audit-start-stamp.sh (PostToolUse hook, matcher=Agent)
# Run: bash .claude/hooks/test-ammo-audit-start-stamp.sh
#
# The hook stamps campaign.rounds[N-1].audit.{stage}.started_at + cycle when the
# ORCHESTRATOR dispatches an ammo-auditor. It calls the REAL rendered
# ammo_state.py audit-started, so these cases exercise the whole chain:
#   piped hook JSON → prompt parse → state engine → state.json on disk.
#
# Key fields consumed from hook stdin:
#   .agent_type                 — empty on the orchestrator, set inside a subagent
#   .tool_input.subagent_type   — must be "ammo-auditor"
#   .tool_input.prompt          — the canonical dispatch block:
#                                   task: audit_gate
#                                   artifact_dir: <dir>
#                                   stage: stage_2     # inline comment tolerated
#                                   round: 1
#                                   cycle: 2
#
# The hook is fail-open and SILENT: every case expects exit 0 and no stdout.
# What distinguishes a stamp from a no-stamp is the state.json content, so each
# case asserts on the file, not on the hook's output.

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK="$HOOK_DIR/ammo-audit-start-stamp.sh"
# The hook resolves the engine through CLAUDE_PROJECT_DIR, so the fake project
# needs the real script at the real relative path.
ENGINE_REL=".claude/skills/ammo/scripts/ammo_state.py"
REAL_ENGINE="$(cd "$HOOK_DIR/../skills/ammo/scripts" && pwd)/ammo_state.py"

PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" 2>/dev/null || true; }
trap cleanup EXIT

# One fake project dir shared by every case; the engine is symlinked in so we
# test the rendered script, not a copy.
PROJ="$TMPDIR/proj"
mkdir -p "$PROJ/$(dirname "$ENGINE_REL")"
ln -s "$REAL_ENGINE" "$PROJ/$ENGINE_REL"

ART="$PROJ/kernel_opt_artifacts/tgt"
mkdir -p "$ART"
STATE="$ART/state.json"

# write_state: $1 = "audit" to give round 1 an audit:{} key, "noaudit" to omit it.
write_state() {
    local mode="$1"
    if [ "$mode" = "audit" ]; then
        jq -n '{
            schema_version: "4.2",
            campaign: { rounds: [ { round_id: 1, status: "IN_PROGRESS", audit: {} } ] }
        }' > "$STATE"
    else
        jq -n '{
            schema_version: "4.2",
            campaign: { rounds: [ { round_id: 1, status: "IN_PROGRESS" } ] }
        }' > "$STATE"
    fi
}

# canonical_prompt: $1 stage  $2 round  $3 cycle
# Reproduces the audit-protocol.md block verbatim, indentation and inline
# stage comment included — the parser must tolerate both.
canonical_prompt() {
    printf '    task: audit_gate\n    artifact_dir: %s\n    stage: %s       # stage_1 | stage_2 | stage_45 | stage_67\n    round: %s\n    cycle: %s\n' \
        "$ART" "$1" "$2" "$3"
}

# run_hook: pipes $1 (json) into the hook under CLAUDE_PROJECT_DIR=$PROJ.
# Records exit code in RC and stdout in $TMPDIR/out.
RC=0
run_hook() {
    RC=0
    printf '%s' "$1" | env CLAUDE_PROJECT_DIR="$PROJ" bash "$HOOK" \
        >"$TMPDIR/out" 2>"$TMPDIR/err" || RC=$?
}

# gate_field: $1 stage  $2 field → value, or "null" when unset
gate_field() {
    jq -r --arg s "$1" --arg f "$2" \
        '(.campaign.rounds[0].audit[$s][$f] // null) | tostring' "$STATE"
}

assert() {
    local name="$1" cond="$2" detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [ "$cond" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name"
        [ -n "$detail" ] && echo "        $detail"
        FAIL=$((FAIL + 1))
    fi
}

# Every case must be silent with exit 0; assert that once per case.
assert_silent_exit0() {
    local name="$1"
    local ok=true
    [ "$RC" -eq 0 ] || ok=false
    [ -s "$TMPDIR/out" ] && ok=false
    assert "$name" "$ok" "exit=$RC stdout=$(head -c 120 "$TMPDIR/out" 2>/dev/null)"
}

# ══════════════════════════════════════════════
echo "== (a) orchestrator dispatch of ammo-auditor → stamped =="
# ══════════════════════════════════════════════

write_state audit
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "a1 hook is silent and exits 0"

STARTED=$(gate_field stage_2 started_at)
CYCLE=$(gate_field stage_2 cycle)
OK=true
printf '%s' "$STARTED" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' || OK=false
assert "a2 started_at stamped as UTC ISO-8601 seconds" "$OK" "started_at=$STARTED"
assert "a3 cycle stamped as 1" "$([ "$CYCLE" = "1" ] && echo true || echo false)" "cycle=$CYCLE"
# passed_at stays the lead's field — the hook must not invent one.
PA=$(gate_field stage_2 passed_at)
assert "a4 passed_at untouched (still absent)" \
    "$([ "$PA" = "null" ] && echo true || echo false)" "passed_at=$PA"

# ══════════════════════════════════════════════
echo ""; echo "== (b) same payload from inside a subagent → NOT stamped =="
# ══════════════════════════════════════════════
# A champion that quotes the dispatch block in a prompt of its own would
# otherwise forge audit provenance.

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",agent_type:"ammo-impl-champion",
      tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "b1 hook is silent and exits 0"
assert "b2 state.json byte-unchanged (agent_type set)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)" \
    "started_at=$(gate_field stage_2 started_at)"

# ══════════════════════════════════════════════
echo ""; echo "== (c) non-auditor subagent_type → NOT stamped =="
# ══════════════════════════════════════════════

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-impl-champion",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "c1 hook is silent and exits 0"
assert "c2 state.json byte-unchanged (subagent_type is not ammo-auditor)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

# ══════════════════════════════════════════════
echo ""; echo "== (d) malformed prompt (no stage line) → NOT stamped =="
# ══════════════════════════════════════════════

write_state audit
BEFORE=$(cat "$STATE")
BAD=$(printf '    task: audit_gate\n    artifact_dir: %s\n    round: 1\n    cycle: 1\n' "$ART")
PAYLOAD=$(jq -nc --arg p "$BAD" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "d1 hook is silent and exits 0"
assert "d2 state.json byte-unchanged (stage missing)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

# Sibling malformed shapes that must also no-op.
write_state audit
BEFORE=$(cat "$STATE")
BAD=$(printf '    task: review_verdict\n    artifact_dir: %s\n    stage: stage_2\n    round: 1\n    cycle: 1\n' "$ART")
PAYLOAD=$(jq -nc --arg p "$BAD" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "d3 wrong task: value — silent, exit 0"
assert "d4 state.json byte-unchanged (task is not audit_gate)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_6 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "d5 non-dispatch stage_6 — silent, exit 0"
assert "d6 state.json byte-unchanged (stage_6 is not dispatchable)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 0 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "d7 round 0 — silent, exit 0"
assert "d8 state.json byte-unchanged (round must be >= 1)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
# artifact_dir that does not exist: rewrite the block against a bogus path.
GONE=$(printf '    task: audit_gate\n    artifact_dir: %s/nope\n    stage: stage_2\n    round: 1\n    cycle: 1\n' "$ART")
PAYLOAD=$(jq -nc --arg p "$GONE" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "d9 artifact_dir does not exist — silent, exit 0"
assert "d10 state.json byte-unchanged (artifact_dir must be a directory)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

# ══════════════════════════════════════════════
echo ""; echo "== (e) round with no audit key → NOT stamped, key NOT created =="
# ══════════════════════════════════════════════
# A legacy campaign has no audit key. Creating one would switch gate
# enforcement on mid-campaign, so the engine no-ops and the hook stays quiet.

write_state noaudit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "e1 hook is silent and exits 0"
assert "e2 state.json byte-unchanged" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"
HAS_AUDIT=$(jq -r 'has("audit") | tostring' <<<"$(jq -c '.campaign.rounds[0]' "$STATE")")
assert "e3 audit key NOT created" \
    "$([ "$HAS_AUDIT" = "false" ] && echo true || echo false)" "has(audit)=$HAS_AUDIT"

# ══════════════════════════════════════════════
echo ""; echo "== (f) re-dispatch with cycle 2 → started_at overwritten, cycle==2 =="
# ══════════════════════════════════════════════

write_state audit
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_45 2 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
# Round 2 does not exist yet → fail-open no-op; add it, then stamp cycle 1.
jq '.campaign.rounds += [{round_id:2,status:"IN_PROGRESS",audit:{}}]' "$STATE" > "$STATE.tmp" \
    && mv "$STATE.tmp" "$STATE"
run_hook "$PAYLOAD"
assert_silent_exit0 "f1 first dispatch is silent and exits 0"
FIRST=$(jq -r '.campaign.rounds[1].audit.stage_45.started_at // "null"' "$STATE")
FIRST_CYCLE=$(jq -r '.campaign.rounds[1].audit.stage_45.cycle // "null"' "$STATE")
assert "f2 round 2 stage_45 stamped on cycle 1" \
    "$([ "$FIRST" != "null" ] && [ "$FIRST_CYCLE" = "1" ] && echo true || echo false)" \
    "started_at=$FIRST cycle=$FIRST_CYCLE"

# Back-date the stamp so the overwrite is observable regardless of clock
# resolution — a same-second re-dispatch would otherwise be indistinguishable.
jq '.campaign.rounds[1].audit.stage_45.started_at = "2020-01-01T00:00:00Z"' "$STATE" > "$STATE.tmp" \
    && mv "$STATE.tmp" "$STATE"
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_45 2 2)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "f3 re-dispatch is silent and exits 0"
SECOND=$(jq -r '.campaign.rounds[1].audit.stage_45.started_at // "null"' "$STATE")
SECOND_CYCLE=$(jq -r '.campaign.rounds[1].audit.stage_45.cycle // "null"' "$STATE")
assert "f4 started_at overwritten by the re-dispatch" \
    "$([ "$SECOND" != "2020-01-01T00:00:00Z" ] && [ "$SECOND" != "null" ] && echo true || echo false)" \
    "started_at=$SECOND"
assert "f5 cycle advanced to 2" \
    "$([ "$SECOND_CYCLE" = "2" ] && echo true || echo false)" "cycle=$SECOND_CYCLE"
# Round 1's gate must not move when round 2 is stamped.
R1=$(jq -r '.campaign.rounds[0].audit.stage_45 // "null" | tostring' "$STATE")
assert "f6 round 1 gate untouched" \
    "$([ "$R1" = "null" ] && echo true || echo false)" "round1.stage_45=$R1"

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open: no jq =="
# ══════════════════════════════════════════════

write_state audit
BEFORE=$(cat "$STATE")
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env printf grep sed dirname basename test rm ls head python3; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
printf '%s' "$PAYLOAD" > "$TMPDIR/payload.json"
RC=0
env -i PATH="$MIN_PATH" CLAUDE_PROJECT_DIR="$PROJ" bash "$HOOK" \
    <"$TMPDIR/payload.json" >"$TMPDIR/out" 2>/dev/null || RC=$?
assert_silent_exit0 "g1 hook fails open without jq"
assert "g2 state.json byte-unchanged without jq" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

# ══════════════════════════════════════════════
echo ""; echo "== Edge payloads =="
# ══════════════════════════════════════════════

write_state audit
BEFORE=$(cat "$STATE")
run_hook '{}'
assert_silent_exit0 "h1 empty JSON — silent, exit 0"
assert "h2 state.json byte-unchanged on empty JSON" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

run_hook '{"tool_name":"Agent","tool_input":null}'
assert_silent_exit0 "h3 null tool_input — silent, exit 0"

PAYLOAD=$(jq -nc '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:""}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "h4 empty prompt — silent, exit 0"
assert "h5 state.json byte-unchanged on empty prompt" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)"

# run_in_background is NOT a gate: a foreground dispatch stamps too.
write_state audit
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_1 1 3)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p,run_in_background:false}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "h6 run_in_background:false — silent, exit 0"
S1C=$(gate_field stage_1 cycle)
assert "h7 foreground dispatch still stamps (cycle 3)" \
    "$([ "$S1C" = "3" ] && echo true || echo false)" "cycle=$S1C"

# ══════════════════════════════════════════════
echo ""; echo "== (i) failed spawn (tool_response error) → NOT stamped =="
# ══════════════════════════════════════════════
# PostToolUse fires for errored tool calls too. A spawn that never launched must
# not attest that an auditor started — the same reason PreToolUse was rejected.

for RESP in \
    '{"status":"error","error":"spawn failed"}' \
    '{"error":"permission denied"}' \
    '{"is_error":true}' \
    '{"isError":true}'
do
    write_state audit
    BEFORE=$(cat "$STATE")
    PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 1)" --argjson r "$RESP" \
        '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p},tool_response:$r}')
    run_hook "$PAYLOAD"
    assert_silent_exit0 "i1 errored spawn is silent and exits 0 ($RESP)"
    assert "i2 state.json byte-unchanged on errored spawn ($RESP)" \
        "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)" \
        "started_at=$(gate_field stage_2 started_at)"
done

# A successful tool_response must still stamp — the error check is not a blanket
# requirement that tool_response be absent.
write_state audit
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 5)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p},
      tool_response:{status:"success"}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "i3 successful spawn is silent and exits 0"
I3C=$(gate_field stage_2 cycle)
assert "i4 successful tool_response still stamps (cycle 5)" \
    "$([ "$I3C" = "5" ] && echo true || echo false)" "cycle=$I3C"

# ══════════════════════════════════════════════
echo ""; echo "== (j) teammate payload without agent_type → NOT stamped =="
# ══════════════════════════════════════════════
# A tmux teammate carries no .agent_type and no CLAUDE_SUBAGENT (see
# test-ammo-is-lead.sh). Only the shared _ammo_is_lead predicate catches it.

write_state audit
BEFORE=$(cat "$STATE")
TRANSCRIPT="$TMPDIR/teammate.jsonl"
printf '%s\n' '{"agentName":"champion-1","teamName":"session-abc"}' > "$TRANSCRIPT"
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_1 1 2)" --arg t "$TRANSCRIPT" \
    '{tool_name:"Agent",transcript_path:$t,
      tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
run_hook "$PAYLOAD"
assert_silent_exit0 "j1 teammate payload is silent and exits 0"
assert "j2 state.json byte-unchanged (transcript agentName is not team-lead)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)" \
    "started_at=$(gate_field stage_1 started_at)"

write_state audit
BEFORE=$(cat "$STATE")
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_1 1 2)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
RC=0
printf '%s' "$PAYLOAD" | env CLAUDE_PROJECT_DIR="$PROJ" CLAUDE_SUBAGENT=1 bash "$HOOK" \
    >"$TMPDIR/out" 2>"$TMPDIR/err" || RC=$?
assert_silent_exit0 "j3 CLAUDE_SUBAGENT=1 is silent and exits 0"
assert "j4 state.json byte-unchanged (CLAUDE_SUBAGENT=1)" \
    "$([ "$BEFORE" = "$(cat "$STATE")" ] && echo true || echo false)" \
    "started_at=$(gate_field stage_1 started_at)"

# ══════════════════════════════════════════════
echo ""; echo "== (k) planted engine under artifact_dir is NEVER executed =="
# ══════════════════════════════════════════════
# artifact_dir arrives in the spawn prompt and the artifact tree is writable by
# the campaign agent, so a walk-up from it would let prompt content choose the
# code this hook runs. The engine resolves from the hook's own dir only.

PLANT="$ART/.claude/skills/ammo/scripts"
mkdir -p "$PLANT"
cat > "$PLANT/ammo_state.py" <<'PLANTED'
import pathlib, sys
pathlib.Path(__file__).parent.joinpath("PWNED").write_text(" ".join(sys.argv))
PLANTED
write_state audit
PAYLOAD=$(jq -nc --arg p "$(canonical_prompt stage_2 1 4)" \
    '{tool_name:"Agent",tool_input:{subagent_type:"ammo-auditor",prompt:$p}}')
# No CLAUDE_PROJECT_DIR at all: the old walk-up path was reachable this way.
RC=0
printf '%s' "$PAYLOAD" | env -u CLAUDE_PROJECT_DIR bash "$HOOK" \
    >"$TMPDIR/out" 2>"$TMPDIR/err" || RC=$?
assert_silent_exit0 "k1 hook is silent and exits 0 without CLAUDE_PROJECT_DIR"
assert "k2 planted engine under artifact_dir was NOT executed" \
    "$([ ! -f "$PLANT/PWNED" ] && echo true || echo false)" \
    "PWNED=$(cat "$PLANT/PWNED" 2>/dev/null)"
# The trusted engine still ran, so the real stamp landed.
K2C=$(gate_field stage_2 cycle)
assert "k3 trusted engine still stamped (cycle 4)" \
    "$([ "$K2C" = "4" ] && echo true || echo false)" "cycle=$K2C"
rm -rf "$ART/.claude"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
