#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""AMMO state engine — the python source of truth for state.json mutation,
validation, the next-step reminder, round/stage transitions, and FE-metric
enrichment.

This consolidates logic that previously lived as inline shell+jq+python in the
two PostToolUse hooks (ammo-state-validate.sh, ammo-next-step-reminder.sh) so
they can be tested and reused. The hooks will be rewired to call this engine.

Style: python3 stdlib only. `jsonschema` is an OPTIONAL import — when absent,
schema validation degrades gracefully (skipped with a stderr warning) while the
cross-field / audit gates still run.

Verbs:
  get      ARTIFACT_DIR_OR_STATE [--field DOTTED.PATH]
  set      --state FILE --field PATH --value JSON [...repeatable]
  validate --state FILE [--schema FILE] [--emit hook]
  next-step --state FILE [--prev FILE] [--emit hook] [--print-terminal]
  advance  --state FILE --outcome SHIP|EXHAUSTED|CONTINUE
  enrich   --state FILE (--gate 5_1a|--gate 5_2 --op-id ID --from F | --mining --from F | --debate --from-dir D)
  ingest-baseline --state FILE --from e2e_latency_results.json [--round N]
  backfill --state FILE --artifact-dir DIR
  audit-started --artifact-dir DIR --stage STAGE --round N --cycle C
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
    _HAVE_JSONSCHEMA = True
except ImportError:  # pragma: no cover - exercised in environments without jsonschema
    Draft202012Validator = None  # type: ignore
    _HAVE_JSONSCHEMA = False


_THIS_DIR = Path(__file__).resolve().parent
_TRANSITIONS_PATH = _THIS_DIR / "transitions.json"

_AUDIT_STARTED_SCRIPT = ".codex/skills/ammo/scripts/ammo_state.py"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_transitions():
    return json.loads(_TRANSITIONS_PATH.read_text(encoding="utf-8"))


def find_schema(start_path):
    """Walk up from `start_path` to locate .codex/schemas/state.schema.json,
    stopping at a .git boundary — same algorithm as ammo-state-validate.sh.
    Returns a Path or None.
    """
    # The immutable container copy is rooted at /opt/codex-managed-hooks rather
    # than in a directory literally named .codex. Keep that Codex packaging
    # adapter ahead of the Claude-compatible repository walk below.
    bundled = _THIS_DIR.parents[2] / "schemas" / "state.schema.json"
    if bundled.is_file():
        return bundled
    d = Path(start_path).resolve()
    if d.is_file():
        d = d.parent
    for _ in range(10):
        candidate = d / ".codex" / "schemas" / "state.schema.json"
        if candidate.is_file():
            return candidate
        parent = d.parent
        if parent == d:
            break
        if (d / ".git").is_dir():
            break
        d = parent
    return None


def load_schema(start_path, explicit=None):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), p
        return None, None
    sp = find_schema(start_path)
    if sp is None:
        return None, None
    return json.loads(sp.read_text(encoding="utf-8")), sp


def _config_defaults_from_schema(schema):
    """Pull the campaign.config.* numeric defaults out of the schema so writers
    never drift from the single source of truth. Returns a dict path->default.
    """
    out = {}
    try:
        props = schema["properties"]["campaign"]["properties"]["config"]["properties"]
        for key, spec in props.items():
            if isinstance(spec, dict) and "default" in spec:
                out[key] = spec["default"]
    except (KeyError, TypeError):
        pass
    return out


# ---------------------------------------------------------------------------
# Dotted-path helpers
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


def _split_path(dotted):
    # `a.b[0].c` and `a.b.0.c` are equivalent. Without expanding brackets,
    # set_field() creates a literal "b[0]" key instead of indexing the list.
    segs = []
    for part in dotted.split("."):
        if part == "":
            continue
        head = _BRACKET_RE.sub("", part)
        if head != "":
            segs.append(head)
        segs.extend(m.group(1) for m in _BRACKET_RE.finditer(part))
    return segs


def get_field(doc, dotted):
    """Resolve a dotted path. Integer-looking segments index lists.
    Returns (found, value)."""
    cur = doc
    for seg in _split_path(dotted):
        if isinstance(cur, dict):
            if seg not in cur:
                return False, None
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return False, None
            if idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


def set_field(doc, dotted, value):
    """Set a dotted path in-place, creating intermediate dicts as needed."""
    segs = _split_path(dotted)
    cur = doc
    for i, seg in enumerate(segs[:-1]):
        if isinstance(cur, list):
            cur = cur[_list_index(cur, seg, dotted)]
        else:
            if seg not in cur or not isinstance(cur[seg], (dict, list)):
                cur[seg] = {}
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        cur[_list_index(cur, last, dotted)] = value
    else:
        cur[last] = value


def _list_index(seq, seg, dotted):
    """Resolve a list index segment, refusing to silently grow or wrap."""
    try:
        idx = int(seg)
    except ValueError:
        raise KeyError(
            "path %r indexes a list with non-integer segment %r" % (dotted, seg)
        )
    if idx < 0 or idx >= len(seq):
        raise IndexError(
            "path %r index %d is out of range (list has %d items)"
            % (dotted, idx, len(seq))
        )
    return idx


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def resolve_state_path(arg):
    """Accept either a direct state.json path or an artifact dir containing one."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "state.json"
        if cand.is_file():
            return cand
    # path that ends in state.json but doesn't exist yet
    if p.name == "state.json":
        return p
    return p / "state.json"


def load_state(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, doc):
    """Atomic write: tempfile in same dir + os.replace (gpu_reservation.py pattern)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# The four gates an ammo-auditor is ever dispatched for. stage_6 and stage_7
# exist in the schema only for pre-consolidation campaigns; they are not
# dispatch targets, so `audit-started` cannot address them.
AUDIT_GATE_STAGES = ("stage_1", "stage_2", "stage_45", "stage_67")

# Pre-consolidation aliases the pass-checks below still accept in place of
# stage_67. They are read-only: the provenance backstop must inspect them, or a
# lead satisfies the S67 gate by typing the legacy key and skips the check.
LEGACY_AUDIT_GATE_STAGES = ("stage_6", "stage_7")

# Every gate key that can carry a passed_at the gates honor.
PROVENANCED_GATE_STAGES = AUDIT_GATE_STAGES + LEGACY_AUDIT_GATE_STAGES


def _utc_stamp():
    """Timezone-aware UTC ISO-8601 at seconds precision — the same shape the
    lead writes into audit.{stage}.passed_at."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def state_lock(state_path):
    """Hold an exclusive advisory lock across a whole load-mutate-write.

    os.replace() makes each individual write atomic but does not stop a lost
    update: the auditor-spawn hook and the orchestrator can read the same
    state.json and then write over each other. The lock file sits next to
    state.json so every writer serializes on one inode.

    EVERY writer of state.json takes this lock, not only `audit-started`. A lock
    one writer skips is no lock: the auditor spawns in the background, so the
    lead keeps writing while the hook stamps, and an unlocked `set` erases the
    stamp. The gate then has passed_at with no started_at, which the schema-4.2
    backstop rejects, so a lost stamp wedges a campaign whose audit really ran.
    reconcile_track_state.py takes the same lock file.
    """
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    handle = open(str(lock_path), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# Schema validation (mirrors ammo-state-validate.sh inline python, lines 100-132)
# ---------------------------------------------------------------------------

def schema_errors(state, schema):
    """Return formatted error lines ("  - path: msg") for a Draft 2020-12
    validation, max 10 then "... and N more". Empty list = valid.
    Returns None if jsonschema is unavailable (caller should skip)."""
    if not _HAVE_JSONSCHEMA:
        return None
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(state), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        msg = err.message
        if len(msg) > 200:
            msg = msg[:200] + "..."
        errors.append(f"  - {path}: {msg}")
    if not errors:
        return []
    n = len(errors)
    out = errors[:10]
    if n > 10:
        out.append(f"  ... and {n - 10} more")
    return out


# ---------------------------------------------------------------------------
# Cross-field / audit gates (port of ammo-state-validate.sh lines 136-261)
# ---------------------------------------------------------------------------

def _round_idx(state):
    cr = state.get("campaign", {}).get("current_round", 1) or 1
    return cr - 1


def _round(state, idx):
    rounds = state.get("campaign", {}).get("rounds", [])
    if 0 <= idx < len(rounds):
        return rounds[idx]
    return {}


def _effective_top_addressable_e2e_pct(state, idx):
    """Resolve the current mining result, including an explicitly reused one."""
    for round_idx in range(idx, -1, -1):
        rnd = _round(state, round_idx)
        if rnd.get("mining_invalidated") is True:
            return None
        value = (rnd.get("bottleneck_mining") or {}).get(
            "top_addressable_e2e_pct"
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _selected_op_ids(round_state):
    """Return the two Stage-3 selection views and their normalized op-id sets."""
    debate = round_state.get("debate") or {}
    winners = debate.get("selected_winners") or []
    candidates = debate.get("selected_candidates") or []
    winner_ids = {
        value for value in winners if isinstance(value, str) and value
    }
    candidate_ids = {
        value.get("op_id") for value in candidates
        if isinstance(value, dict)
        and isinstance(value.get("op_id"), str)
        and value.get("op_id")
    }
    return debate, winners, candidates, winner_ids, candidate_ids


def _material_ship(round_state):
    """Whether a SHIPPED round changed the production baseline materially."""
    if round_state.get("status") != "SHIPPED":
        return False
    shipped = round_state.get("shipped") or []
    tracks = (round_state.get("parallel_tracks") or {}).get("tracks") or {}
    return bool(shipped) and not all(
        isinstance(tracks.get(op_id), dict)
        and tracks[op_id].get("diluted") is True
        for op_id in shipped
    )


def _post_ship_mining_round(state, idx):
    """Whether idx mines the promoted Stage-6 result of a material SHIP."""
    if idx <= 0:
        return False
    return _material_ship(_round(state, idx - 1))


def _effective_integration_status(state, idx):
    """Use the prior SHIP outcome while its promoted baseline is being mined."""
    current = (_round(state, idx).get("integration") or {}).get("status")
    if _post_ship_mining_round(state, idx) and current in {None, "pending"}:
        return ((_round(state, idx - 1).get("integration") or {}).get("status"))
    return current


def _schema_ver_tuple(state):
    sv = str(state.get("campaign", {}).get("schema_version", "4.0"))
    parts = sv.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        major = 4
    try:
        minor = int(parts[1])
    except (ValueError, IndexError):
        minor = 0
    return major, minor


# Tier P-score caps (references/debate-rules.md § Evidence Tier × Scope).
_TIER_PSCORE_CAP = {"tier_1": 3, "tier_2": 7, "tier_3": 10}


def _score_breakdown_violation(op_id, sbrk):
    """Tier cap + EV identity on the typed selected_candidates score_breakdown.

    `feasibility` is the P-score, so the same tier cap applies.
    `weighted_total` is EV_pct = feasibility/10 x expected_e2e_pct. Both are
    schema-required, so a new or modified entry always has them to check.
    Returns a block-reason string or None.
    """
    if not isinstance(sbrk, dict):
        return None
    tier = sbrk.get("evidence_tier")
    feasibility = sbrk.get("feasibility")
    if tier in _TIER_PSCORE_CAP and isinstance(feasibility, (int, float)) \
            and not isinstance(feasibility, bool):
        cap = _TIER_PSCORE_CAP[tier]
        if feasibility > cap:
            return (
                "Selected-candidate violation: %s has score_breakdown."
                "feasibility=%s > tier cap %d for evidence_tier=%s "
                "(references/debate-rules.md § Execution Confidence)."
                % (op_id, feasibility, cap, tier)
            )
    expected = sbrk.get("expected_e2e_pct")
    weighted = sbrk.get("weighted_total")
    numbers = (feasibility, expected, weighted)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in numbers):
        return None
    computed = feasibility / 10.0 * expected
    # Tolerance = one-decimal rounding (0.05) or 1% of the EV, whichever is
    # larger. No contract pins the precision of weighted_total, and champions
    # publish it to 1 dp: EV(9/10 x 2.6) = 2.34 is legitimately written 2.3.
    # A tolerance below 0.05 turns honest rounding into a gate failure while
    # still catching the arbitrary EV the check exists to stop.
    if abs(weighted - computed) > max(0.05, abs(computed) * 0.01):
        return (
            "Selected-candidate violation: %s has score_breakdown."
            "weighted_total=%s but feasibility/10 x expected_e2e_pct = "
            "%s/10 x %s = %s (references/debate-scoring-rubric.md § EV)."
            % (op_id, weighted, feasibility, expected, computed)
        )
    return None


def _scope_gate_checks(state, prev_state):
    """Tier-cap and evidence-contract checks for debate selections.

    Enforced ONLY on entries that are NEW or MODIFIED vs prev_state, and only
    when the constrained fields are present. When prev_state is None (no
    snapshot — e.g. first fire, or direct `validate` without --prev), every
    pre-existing entry is grandfathered (returns None) so a paused campaign's
    state (tier_2/p_score=8, no evidence_scope) still validates on resume.
    Returns a block-reason string or None.
    """
    rounds = state.get("campaign", {}).get("rounds", []) or []
    prounds = (prev_state or {}).get("campaign", {}).get("rounds", []) or []
    for idx, rnd in enumerate(rounds):
        debate = (rnd.get("debate") or {})
        pdebate = (prounds[idx].get("debate") if idx < len(prounds) else {}) or {}

        # ── V1: scoreboard p_score <= cap(evidence_tier) ──
        sb = debate.get("scoreboard") or {}
        psb = pdebate.get("scoreboard") or {}
        if isinstance(sb, dict):
            for op_id, entry in sb.items():
                if not isinstance(entry, dict):
                    continue
                tier = entry.get("evidence_tier")
                pscore = entry.get("p_score")
                if tier not in _TIER_PSCORE_CAP or not isinstance(pscore, (int, float)):
                    continue  # fields absent → nothing to enforce
                prev_entry = psb.get(op_id) if isinstance(psb, dict) else None
                # Grandfather: unchanged (tier,p_score) pair that already existed.
                if isinstance(prev_entry, dict) \
                        and prev_entry.get("evidence_tier") == tier \
                        and prev_entry.get("p_score") == pscore:
                    continue
                # prev_state is None → we cannot prove it is new → grandfather.
                if prev_state is None:
                    continue
                cap = _TIER_PSCORE_CAP[tier]
                if pscore > cap:
                    return (
                        "Debate scoreboard violation: %s has p_score=%s > tier cap %d "
                        "for evidence_tier=%s (references/debate-rules.md § Evidence "
                        "Tier × Scope)." % (op_id, pscore, cap, tier)
                    )

        # ── V2: new/modified selections need typed category + scope ──
        sel = debate.get("selected_candidates") or []
        psel = pdebate.get("selected_candidates") or []
        prev_by_op = {
            c.get("op_id"): c for c in psel
            if isinstance(c, dict) and c.get("op_id") is not None
        }
        if isinstance(sel, list):
            if prev_state is not None and sel != psel:
                contingent_count = sum(
                    1
                    for cand in sel
                    if isinstance(cand, dict)
                    and cand.get("selection_mode") == "contingent_host_spike"
                )
                if contingent_count > 1:
                    return (
                        "Selected-candidate violation: at most one "
                        "contingent_host_spike may be selected per round."
                    )
            for cand in sel:
                if not isinstance(cand, dict):
                    continue
                op_id = cand.get("op_id")
                prev_cand = prev_by_op.get(op_id)
                # A missing previous snapshot cannot distinguish legacy state
                # from a new write. Unrelated resume repairs (for example,
                # proposal_file backfill) do not promote a legacy claim; only
                # new entries or changes to its typed claim contract bind.
                if prev_state is None:
                    continue
                contract_fields = (
                    "score_breakdown",
                    "category",
                    "selection_mode",
                    "host_slice_ceiling_pct",
                    "projected_e2e_improvement_pct",
                    "stage_4_validation_obligations",
                )
                claim_changed = prev_cand is None or any(
                    cand.get(field) != prev_cand.get(field)
                    for field in contract_fields
                )
                if not claim_changed:
                    continue
                sbrk = cand.get("score_breakdown") or {}
                scope = sbrk.get("evidence_scope") if isinstance(sbrk, dict) else None
                if not cand.get("category"):
                    return (
                        "Selected-candidate violation: %s is new or modified but "
                        "has no category token. Declare the closest schema category."
                        % op_id
                    )
                if not scope:
                    return (
                        "Selected-candidate violation: %s is new or modified but "
                        "has no score_breakdown.evidence_scope token."
                        % op_id
                    )
                if scope == "proxy":
                    return (
                        "Selected-candidate violation: %s has "
                        "score_breakdown.evidence_scope='proxy' (proxy magnitude "
                        "is not EV-admissible; references/debate-rules.md § "
                        "Evidence-Scope Ladder)." % op_id
                    )
                # The typed contract carries the SAME two numbers the free-form
                # scoreboard is capped on: feasibility IS the P-score and
                # weighted_total IS the EV (references/debate-scoring-rubric.md
                # § State Contract). Enforce both here, else a tier_1 candidate
                # with feasibility 10 and an arbitrary EV passes.
                reason = _score_breakdown_violation(op_id, sbrk)
                if reason:
                    return reason
                mode = cand.get("selection_mode")
                if mode not in {"ordinary", "contingent_host_spike"}:
                    return (
                        "Selected-candidate violation: %s is new or modified but "
                        "has no valid selection_mode." % op_id
                    )
                if mode == "ordinary":
                    continue
                ceiling = cand.get("host_slice_ceiling_pct")
                threshold = (
                    state.get("campaign", {}).get("config", {}).get(
                        "min_e2e_improvement_pct"
                    )
                )
                obligations = cand.get("stage_4_validation_obligations") or []
                projected = cand.get("projected_e2e_improvement_pct")
                expected = sbrk.get("expected_e2e_pct")
                weighted = sbrk.get("weighted_total")
                if scope != "bound":
                    return (
                        "Selected-candidate violation: contingent_host_spike %s "
                        "must declare bound evidence_scope." % op_id
                    )
                if projected != 0 or expected != 0 or weighted != 0:
                    return (
                        "Selected-candidate violation: contingent_host_spike %s "
                        "must record zero projected magnitude and EV." % op_id
                    )
                if (
                    not isinstance(ceiling, (int, float))
                    or isinstance(ceiling, bool)
                    or not isinstance(threshold, (int, float))
                    or isinstance(threshold, bool)
                    or ceiling < threshold
                ):
                    return (
                        "Selected-candidate violation: contingent_host_spike %s "
                        "requires a numeric host_slice_ceiling_pct that meets "
                        "or exceeds the campaign floor." % op_id
                    )
                if not obligations or obligations[0] != "production_boundary_spike":
                    return (
                        "Selected-candidate violation: contingent_host_spike %s "
                        "requires production_boundary_spike as its first "
                        "implementation obligation." % op_id
                    )
    return None


def gate_violation(state, transitions, prev_state=None):
    """Run the cross-field Stage-6 guard, the audit gates, and the new-round
    start gate. Returns a block-reason string on the FIRST violation, else None.

    Block-reason strings are kept byte-identical to ammo-state-validate.sh (the
    90-test harness greps substrings of them).
    """
    campaign = state.get("campaign", {})
    stage = campaign.get("current_stage", "") or ""
    cr_idx = _round_idx(state)

    # Stage motion is a mechanical invariant. Same-round writes may stay put
    # or advance one rung; round creation is allowed only from campaign eval
    # to the outcome-determined Stage 2/3 start.
    if prev_state is not None:
        prev_campaign = prev_state.get("campaign", {}) or {}
        prev_stage = prev_campaign.get("current_stage", "") or ""
        prev_round_id = prev_campaign.get("current_round", 1) or 1
        round_id = campaign.get("current_round", 1) or 1
        ladder = transitions.get("stage_ladder", [])
        if round_id == prev_round_id:
            if stage != prev_stage:
                legal_adjacent = (
                    prev_stage in ladder and stage in ladder
                    and ladder.index(stage) == ladder.index(prev_stage) + 1
                )
                if not legal_adjacent:
                    # Post-SHIP re-mine short-circuit: a round created by a
                    # material SHIP exists to re-mine the promoted baseline;
                    # once that fresh mining is complete, the mechanical
                    # threshold check runs at campaign eval directly — debate
                    # and tracks are skipped when the check terminates the
                    # campaign. Audit/enrichment gates still apply separately.
                    prior = _round(state, round_id - 2) if round_id >= 2 else {}
                    mining_done = bool(
                        (_round(state, round_id - 1).get("bottleneck_mining") or {})
                        .get("completed_at")
                    )
                    remine_shortcut = (
                        prev_stage == "2_bottleneck_mining"
                        and stage == "7_campaign_eval"
                        and _material_ship(prior)
                        and mining_done
                    )
                    if not remine_shortcut:
                        return (
                            "Stage-transition violation: same-round transition %s -> %s "
                            "is not adjacent in the canonical stage ladder."
                            % (prev_stage, stage)
                        )
        elif round_id == prev_round_id + 1:
            prior = _round(state, round_id - 2)
            prior_status = prior.get("status")
            if prior_status == "SHIPPED":
                shipped = prior.get("shipped") or []
                prior_tracks = (
                    (prior.get("parallel_tracks") or {}).get("tracks") or {}
                )
                all_diluted = bool(shipped) and all(
                    isinstance(prior_tracks.get(op_id), dict)
                    and prior_tracks[op_id].get("diluted") is True
                    for op_id in shipped
                )
                expected = "3_debate" if all_diluted else "2_bottleneck_mining"
            elif prior_status == "EXHAUSTED":
                expected = (
                    "2_bottleneck_mining"
                    if prior.get("mining_invalidated") is True
                    else "3_debate"
                )
            else:
                expected = None
            if prev_stage != "7_campaign_eval" or stage != expected:
                return (
                    "Stage-transition violation: new round must follow campaign eval "
                    "and enter the stage implied by SHIPPED/EXHAUSTED outcome; got "
                    "%s round %s -> %s round %s."
                    % (prev_stage, prev_round_id, stage, round_id)
                )
        else:
            return (
                "Stage-transition violation: current_round changed from %s to %s; "
                "only one new round may be created at a time."
                % (prev_round_id, round_id)
            )

    _scope_reason = _scope_gate_checks(state, prev_state)
    if _scope_reason:
        return _scope_reason

    # Stage 4/6 cohort identity is structural, not an agent judgment. Bind the
    # two debate views to exactly one track plus one implementer/monitor pair.
    cur_round = _round(state, cr_idx)
    debate, winner_values, candidate_values, winner_ids, candidate_ids = (
        _selected_op_ids(cur_round)
    )
    tracks = (cur_round.get("parallel_tracks") or {}).get("tracks") or {}
    track_ids = set(tracks)
    schema_major, schema_minor = _schema_ver_tuple(state)
    cohort_contract_active = (
        schema_major > 4
        or (schema_major == 4 and schema_minor >= 2)
    )
    entered_tracks = (
        stage == "4_5_parallel_tracks"
        and prev_state is not None
        and (prev_state.get("campaign") or {}).get("current_stage")
        == "3_debate"
    )
    if (cohort_contract_active
            and (entered_tracks or stage == "6_integration")):
        raw_candidate_ids = [
            value.get("op_id") for value in candidate_values
            if isinstance(value, dict)
        ]
        if (
            not 2 <= len(winner_values) <= 3
            or not 2 <= len(candidate_values) <= 3
            or len(winner_ids) != len(winner_values)
            or len(candidate_ids) != len(candidate_values)
            or any(not isinstance(value, str) or not value for value in winner_values)
            or any(not isinstance(value, str) or not value for value in raw_candidate_ids)
        ):
            return (
                "Cohort violation: Stage 4/6 requires 2-3 unique winner ids "
                "and 2-3 unique typed candidate op_ids before dispatch "
                "(winners=%s candidates=%s)."
                % (winner_values, raw_candidate_ids)
            )
        if winner_ids != candidate_ids or candidate_ids != track_ids:
            return (
                "Cohort violation: debate.selected_winners, "
                "debate.selected_candidates op_ids, and parallel track keys "
                "must match exactly (winners=%s candidates=%s tracks=%s)."
                % (sorted(winner_ids), sorted(candidate_ids), sorted(track_ids))
            )
        missing_pairs = [
            op_id for op_id in sorted(track_ids)
            if not all(
                isinstance(tracks[op_id].get(field), str)
                and bool(tracks[op_id].get(field).strip())
                for field in (
                    "implementer_agent", "implementer_rollout_id",
                    "monitor_agent", "monitor_evidence_path",
                    "monitor_offsets_path", "monitor_summary_path",
                )
            )
        ]
        if missing_pairs:
            return (
                "Cohort violation: selected track(s) lack durable implementer/"
                "monitor pairing evidence: %s." % ", ".join(missing_pairs)
            )
        selected_by_id = {
            value.get("op_id"): value
            for value in candidate_values
            if isinstance(value, dict)
        }
        threshold = (campaign.get("config") or {}).get(
            "min_e2e_improvement_pct"
        )
        for op_id in sorted(track_ids):
            if selected_by_id[op_id].get("selection_mode") \
                    != "contingent_host_spike":
                continue
            boundary = tracks[op_id].get("gate_5_2_boundary")
            required = (
                "baseline_duration_us", "optimized_duration_us",
                "occurrence_count", "baseline_e2e_us",
                "e2e_equivalent_improvement_pct", "campaign_floor_pct",
                "meets_floor",
            )
            if not isinstance(boundary, dict) or any(
                field not in boundary for field in required
            ):
                return (
                    "Contingent Gate 5.2 violation: %s lacks complete "
                    "production-boundary A/B arithmetic." % op_id
                )
            base = boundary.get("baseline_duration_us")
            opt = boundary.get("optimized_duration_us")
            count = boundary.get("occurrence_count")
            e2e = boundary.get("baseline_e2e_us")
            recorded = boundary.get("e2e_equivalent_improvement_pct")
            floor = boundary.get("campaign_floor_pct")
            numeric = (base, opt, count, e2e, recorded, floor, threshold)
            if (
                any(not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in numeric)
                or base <= 0 or opt < 0 or count < 1 or e2e <= 0
            ):
                return (
                    "Contingent Gate 5.2 violation: %s has invalid boundary "
                    "durations, occurrence count, E2E baseline, or floor." % op_id
                )
            computed = 100.0 * (base - opt) * count / e2e
            tolerance = max(1e-6, abs(computed) * 0.01)
            if (
                abs(recorded - computed) > tolerance
                or abs(floor - threshold) > 1e-9
                or boundary.get("meets_floor") is not True
                or computed < threshold
                or tracks[op_id].get("gate_5_2") != "PASS"
            ):
                return (
                    "Contingent Gate 5.2 violation: %s boundary arithmetic "
                    "does not reproduce a PASS at the campaign floor." % op_id
                )

    # ── Track A17: Stage 6 requires all tracks terminal (PASS/GATED_PASS/FAIL) ──
    if stage == "6_integration":
        terminal = set(transitions["track_terminal_statuses"])
        non_terminal = [
            (op_id, (t.get("status")))
            for op_id, t in tracks.items()
            if t.get("status") not in terminal
        ]
        if non_terminal:
            non_terminal_list = ", ".join(f"{k}={v}" for k, v in non_terminal)
            return (
                "Stage 6 transition blocked: %d track(s) still non-terminal (%s). "
                "All tracks must reach PASS/GATED_PASS/FAIL before current_stage=6_integration."
                % (len(non_terminal), non_terminal_list)
            )

    # ── Track A18 (new): diluted:true requires status=="PASS" ──
    # A malformed diluted/status pair is invalid at ANY stage, so this check
    # runs unconditionally (not stage-gated). It does NOT re-verify the
    # underlying sweep evidence — it only rejects a structurally nonsensical
    # combination (diluted:true + non-PASS status).
    bad_diluted = []
    for round_pos, round_state in enumerate(campaign.get("rounds", []) or [], 1):
        tracks_diluted = (
            (round_state.get("parallel_tracks") or {}).get("tracks") or {}
        )
        bad_diluted.extend(
            ("round%d/%s" % (round_pos, op_id), t.get("status"))
            for op_id, t in tracks_diluted.items()
            if t.get("diluted") is True and t.get("status") != "PASS"
        )
    if bad_diluted:
        bad_list = ", ".join(f"{k}={v}" for k, v in bad_diluted)
        return (
            "Cross-field violation: %d track(s) have diluted=true with status != PASS "
            "(%s). diluted:true is only valid when status=='PASS'."
            % (len(bad_diluted), bad_list)
        )

    # ── Audit gate ──
    audit_exists = "audit" in cur_round
    if not audit_exists:
        major, minor = _schema_ver_tuple(state)
        if major > 4 or (major == 4 and minor >= 1):
            return (
                "Audit state missing from current round. Schema 4.1+ rounds must "
                "carry audit={} so mandatory audit gates cannot fail open."
            )
    if audit_exists:
        audit_key = transitions["audit_stage_key_map"].get(stage, "")
        # 7_campaign_eval* prefix match (only 7_campaign_eval is canonical, but be safe)
        if not audit_key and stage.startswith("7_campaign_eval"):
            audit_key = "stage_67"

        # Post-SHIP re-mine shortcut: a round created by a material SHIP that
        # ran ONLY fresh mining (no tracks, no integration) reaches campaign
        # eval for the mechanical threshold check with no same-round Stage 6-7
        # to audit — its gate is the stage_2 audit; the PREVIOUS round's
        # stage_67 (enforced by the new-round start gate) attests the SHIP.
        if audit_key == "stage_67" and cr_idx > 0:
            prev_round = _round(state, cr_idx - 1)
            cur_tracks = (
                (cur_round.get("parallel_tracks") or {}).get("tracks") or {}
            )
            cur_integration_status = (
                (cur_round.get("integration") or {}).get("status") or "pending"
            )
            mining_done = bool(
                (cur_round.get("bottleneck_mining") or {}).get("completed_at")
            )
            prev_s67 = _audit_passed_at(prev_round.get("audit") or {}, "stage_67")
            if (
                _material_ship(prev_round)
                and prev_s67
                and mining_done
                and not cur_tracks
                and cur_integration_status == "pending"
            ):
                audit_key = "stage_2"

        # Post-SHIP Stage-2 exemption: drop same-round stage_1 requirement for
        # round N>1 ONLY when the PREVIOUS round carries an audit key (fail-closed).
        if audit_key == "stage_1" and cr_idx > 0:
            prev_round = _round(state, cr_idx - 1)
            prev_has_audit = "audit" in prev_round
            if prev_has_audit:
                audit_key = ""

        # EXHAUSTED and all-diluted SHIP start the next round directly at
        # debate with unchanged mining. There is no same-round Stage-2 run to
        # audit; the previous round's stage_67 attests the reused evidence and
        # the new-round gate below requires that attestation. A newly mined round
        # has completed_at set and still requires its own stage_2 audit.
        if audit_key == "stage_2" and cr_idx > 0:
            mining = cur_round.get("bottleneck_mining") or {}
            if not mining.get("completed_at"):
                audit_key = ""

        # stage_2 gate only applies to schema v4.1+
        if audit_key == "stage_2":
            major, minor = _schema_ver_tuple(state)
            min_major = transitions["stage_2_audit_min_schema"]["major"]
            min_minor = transitions["stage_2_audit_min_schema"]["minor"]
            if major < min_major or (major == min_major and minor < min_minor):
                audit_key = ""

        if audit_key:
            audit = cur_round.get("audit") or {}
            passed = _audit_passed_at(audit, audit_key)
            # Backward compat: accept legacy stage_6 if stage_67 not set
            if not passed and audit_key == "stage_67":
                passed = _audit_passed_at(audit, "stage_6")
            if not passed:
                return (
                    "Audit gate (4-phase audit): transition to %s blocked — "
                    "audit.%s.passed_at not set in current round. Spawn ammo-auditor "
                    "for %s first (see .codex/skills/ammo/orchestration/audit-protocol.md)."
                    % (stage, audit_key, audit_key)
                )

    # ── Audit start-stamp backstop (schema 4.2+) ──
    _started_reason = _audit_started_backstop(
        state, transitions, _AUDIT_STARTED_SCRIPT
    )
    if _started_reason:
        return _started_reason

    # ── Mining-enrichment round-advance gate ──
    # Replaces the former schema allOf that made the six enrichment fields
    # non-null whenever completed_at was set: that unconditional rule rejected
    # EVERY subsequent state write once one round lacked the fields (legacy
    # rounds pre-dating decode_frac/component_breakdown wedged campaigns on
    # resume). Enforcement now fires only on the write that ADVANCES the
    # current round past mining — mid-mining writes and unrelated updates on
    # degraded legacy rounds pass through.
    mining_enrich_fields = (
        "top_bottleneck_share_pct", "top_component",
        "top_addressable_e2e_pct", "amdahl_ceiling", "decode_frac",
        "component_breakdown",
    )
    ladder = transitions.get("stage_ladder", [])
    if stage in ladder and "2_bottleneck_mining" in ladder \
            and ladder.index(stage) > ladder.index("2_bottleneck_mining"):
        mining = cur_round.get("bottleneck_mining") or {}
        if mining.get("completed_at"):
            missing = [f for f in mining_enrich_fields if mining.get(f) is None]
            # New campaigns store the total-wall-time share under its correct
            # name. Accept the old misnamed field only as a resume guard.
            if (mining.get("top_f_e2e_pct") is None
                    and mining.get("top_f_decode_pct") is None):
                missing.append("top_f_e2e_pct")
            if missing:
                return (
                    "Round-advance blocked: bottleneck_mining enrichment incomplete — "
                    "%s null/missing on the current round while current_stage=%s. "
                    "Parse rounds/{N}/mining/bottleneck_analysis.md and write the fields "
                    "(SKILL.md § FE Metric Enrichment), or run "
                    "`.venv/bin/python .codex/skills/ammo/scripts/ammo_state.py backfill "
                    "--state <state.json> --artifact-dir <artifact_dir>`. "
                    "Writes that do not advance the round past mining are not blocked."
                    % (", ".join(missing), stage)
                )

    # ── New-round start gate: round N>1 in stage 1/2/3 requires prev round
    #    audit.stage_67 (or legacy stage_7) passed_at when prev carries audit ──
    cr = campaign.get("current_round", 1) or 1
    if cr > 1:
        prev_idx = cr - 2
        prev_round = _round(state, prev_idx)
        prev_audit_exists = "audit" in prev_round
        if prev_audit_exists and stage in transitions["audit_new_round_start_stages"]:
            audit = prev_round.get("audit") or {}
            prev_passed = _audit_passed_at(audit, "stage_67")
            if not prev_passed:
                prev_passed = _audit_passed_at(audit, "stage_7")
            if not prev_passed:
                return (
                    "Audit gate (4-phase audit): new round start blocked — "
                    "audit.stage_67.passed_at not set on previous round (round %d). "
                    "Spawn ammo-auditor for stage_67 first." % (prev_idx + 1)
                )

    # Terminal token must describe the actual integration outcome and can only
    # be written after the consolidated Stage-6/7 audit has passed. A material
    # SHIP is evaluated after mining its promoted Stage-6 result in the next
    # round; that terminal edge remains bound to the prior round's ship audit,
    # not to a fresh baseline/profile/provenance or a same-round Stage-2 audit.
    campaign_status = campaign.get("status")
    integration_status = _effective_integration_status(state, cr_idx)
    if campaign_status in set(transitions["terminal_statuses"]):
        post_ship_eval = _post_ship_mining_round(state, cr_idx)
        if post_ship_eval:
            audit = _round(state, cr_idx - 1).get("audit") or {}
            terminal_audit = _audit_passed_at(audit, "stage_67")
            if not terminal_audit:
                terminal_audit = _audit_passed_at(audit, "stage_7")
        else:
            audit = cur_round.get("audit") or {}
            terminal_audit = _audit_passed_at(audit, "stage_67")
            if not terminal_audit:
                terminal_audit = _audit_passed_at(audit, "stage_6")
        if not terminal_audit:
            return (
                "Terminal-status violation: campaign status cannot become terminal "
                "before audit.stage_67.passed_at is recorded."
            )
        if _material_ship(cur_round):
            return (
                "Terminal-status violation: a material SHIP requires fresh "
                "post-promotion Stage-2 mining (next round, from the promoted "
                "Stage-6 result) before campaign termination."
            )
        if post_ship_eval:
            top_addressable = (cur_round.get("bottleneck_mining") or {}).get(
                "top_addressable_e2e_pct"
            )
        else:
            top_addressable = _effective_top_addressable_e2e_pct(state, cr_idx)
        threshold = (campaign.get("config") or {}).get(
            "min_e2e_improvement_pct"
        )
        if (not isinstance(top_addressable, (int, float))
                or isinstance(top_addressable, bool)
                or not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)):
            return (
                "Terminal-status violation: termination requires numeric "
                "bottleneck_mining.top_addressable_e2e_pct and "
                "campaign.config.min_e2e_improvement_pct."
            )
        if top_addressable >= threshold:
            return (
                "Terminal-status violation: top addressable E2E impact %.6g%% is "
                "not below min_e2e_improvement_pct %.6g%%; continue the "
                "campaign." % (top_addressable, threshold)
            )
    if campaign_status == "campaign_complete" \
            and integration_status not in set(transitions["integration_ship_statuses"]):
        return (
            "Terminal-status violation: campaign_complete requires a SHIP integration "
            "status (single_pass, combined, or gated_pass); use campaign_exhausted "
            "for an exhausted/failed integration."
        )
    if campaign_status == "campaign_exhausted" \
            and integration_status not in {"exhausted", "failed"}:
        return (
            "Terminal-status violation: campaign_exhausted requires integration "
            "status exhausted or failed; a SHIP status requires campaign_complete."
        )

    return None


def _audit_passed_at(audit, key):
    if not isinstance(audit, dict):
        return ""
    sub = audit.get(key)
    if not isinstance(sub, dict):
        return ""
    return sub.get("passed_at") or ""


def _audit_started_backstop(state, transitions, script_path):
    """Schema 4.2+: a passed audit gate must also carry the mechanical start stamp.

    started_at and cycle are written by the auditor-spawn hook at dispatch. A
    gate that reached passed_at without started_at means the hook never saw a
    dispatch, so nothing mechanical attests that an auditor ran. Fail closed and
    hand the lead the exact backfill command. Schema below the floor keeps the
    old behavior.
    """
    major, minor = _schema_ver_tuple(state)
    floor = transitions.get("audit_started_min_schema") or {}
    min_major = floor.get("major", 4)
    min_minor = floor.get("minor", 2)
    if major < min_major or (major == min_major and minor < min_minor):
        return None
    rounds = state.get("campaign", {}).get("rounds", []) or []
    for round_pos, round_state in enumerate(rounds, 1):
        if not isinstance(round_state, dict):
            continue
        audit = round_state.get("audit")
        if not isinstance(audit, dict):
            continue
        for stage_key in PROVENANCED_GATE_STAGES:
            gate = audit.get(stage_key)
            if not isinstance(gate, dict):
                continue
            if not gate.get("passed_at") or gate.get("started_at"):
                continue
            head = (
                "Audit provenance violation: round %d audit.%s has passed_at but "
                "no started_at. started_at is stamped mechanically when the "
                "ammo-auditor is dispatched, so a gate without it carries no "
                "record that an auditor ran. " % (round_pos, stage_key)
            )
            if stage_key in LEGACY_AUDIT_GATE_STAGES:
                # audit-started cannot address a legacy alias, so the fix is a
                # migration: move the verdict to stage_67 and stamp that gate.
                return head + (
                    "%s is a pre-consolidation alias for stage_67 and cannot be "
                    "stamped. Move the verdict to audit.stage_67, then run "
                    "`python3 %s audit-started --artifact-dir <artifact_dir> "
                    "--stage stage_67 --round %d --cycle 1`."
                    % (stage_key, script_path, round_pos)
                )
            return head + (
                "Backfill it with `python3 %s audit-started --artifact-dir "
                "<artifact_dir> --stage %s --round %d --cycle 1`."
                % (script_path, stage_key, round_pos)
            )
    return None


def _block_json(reason):
    """Compact one-line block JSON, mirroring the hooks' jq output."""
    return json.dumps(
        {
            "decision": "block",
            "reason": reason,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": reason,
            },
        },
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Next-step reminder (port of ammo-next-step-reminder.sh lines 67-650)
# ---------------------------------------------------------------------------

def _jget(doc, dotted, default=None):
    found, val = get_field(doc, dotted)
    if not found or val is None:
        return default
    return val


def _round_field(state, idx, dotted, default=None):
    return _jget(state, "campaign.rounds.%d.%s" % (idx, dotted), default)


def _track_keys_by(tracks, predicate):
    return sorted([k for k, v in tracks.items() if predicate(v)])


def _set_diff(cur, prev):
    """Keys in `cur` not in `prev` (both are sorted lists)."""
    prevset = set(prev)
    return [k for k in cur if k not in prevset]


def compute_next_step(state, prev_state, transitions, schema=None):
    """Return (message_or_empty, is_terminal_transition).

    `prev_state` is None when no PREV_STATE snapshot exists (first fire) — in
    that case NO Socratic edge nudges fire (an edge is undefined without a
    baseline) and IS_TERMINAL_TRANSITION is False.
    """
    campaign = state.get("campaign", {})
    stage = campaign.get("current_stage", "unknown") or "unknown"
    status = campaign.get("status", "active") or "active"
    cr = campaign.get("current_round", 1) or 1
    schema_ver = str(campaign.get("schema_version", "4.0"))
    idx = cr - 1

    config_defaults = _config_defaults_from_schema(schema) if schema else {}

    # ── Per-round sub-state ──
    baseline_done = _round_field(state, idx, "baseline.completed_at", "") or ""
    mining_done = _round_field(state, idx, "bottleneck_mining.completed_at", "") or ""
    team_name = _round_field(state, idx, "team_name", "") or ""
    selected_count = len(_round_field(state, idx, "debate.selected_candidates", []) or [])
    selected_winners_count = len(_round_field(state, idx, "debate.selected_winners", []) or [])
    debate_rounds_completed = _round_field(state, idx, "debate.rounds_completed", 0) or 0
    debate_max_rounds = _round_field(state, idx, "debate.max_rounds", 0) or 0
    tracks_started = _round_field(state, idx, "parallel_tracks.started_at", "") or ""
    integ_started = _round_field(state, idx, "integration.started_at", "") or ""
    integ_status = _round_field(state, idx, "integration.status", "") or ""
    shipped_count = len(_round_field(state, idx, "shipped", []) or [])

    cur_round = _round(state, idx)
    mining_invalidated_present = "mining_invalidated" in cur_round
    mining_invalidated = cur_round.get("mining_invalidated") if mining_invalidated_present else None

    audit_exists = "audit" in cur_round
    audit = cur_round.get("audit") or {}
    audit_s1 = _audit_passed_at(audit, "stage_1") if audit_exists else ""
    audit_s2 = _audit_passed_at(audit, "stage_2") if audit_exists else ""
    audit_s45 = _audit_passed_at(audit, "stage_45") if audit_exists else ""
    audit_s67 = _audit_passed_at(audit, "stage_67") if audit_exists else ""
    if audit_exists and not audit_s67:
        audit_s67 = _audit_passed_at(audit, "stage_6")

    tracks = _round_field(state, idx, "parallel_tracks.tracks", {}) or {}
    passing = set(transitions["track_passing_statuses"])
    terminal = set(transitions["track_terminal_statuses"])
    tracks_passing = sum(1 for t in tracks.values() if t.get("status") in passing)
    tracks_non_terminal = sum(1 for t in tracks.values() if t.get("status") not in terminal)
    track_count = len(tracks)
    tracks_missing_lat_opt = sum(
        1 for t in tracks.values()
        if t.get("per_bs_verdict") is not None and t.get("e2e_latency_opt") is None
    )

    def _verdict_keys(trks, *verdicts):
        return _track_keys_by(trks, lambda v: v.get("verdict") in verdicts)

    def _status_keys(trks, *statuses):
        return _track_keys_by(trks, lambda v: v.get("status") in statuses)

    tracks_pass_keys = _verdict_keys(tracks, *passing)
    tracks_fail_keys = _verdict_keys(tracks, "FAIL")
    tracks_gating_keys = _status_keys(tracks, "GATING_REQUIRED")

    # ── Mining metrics ──
    f_e2e = _round_field(state, idx, "bottleneck_mining.top_f_e2e_pct", None)
    if f_e2e is None:
        # Resume guard for pre-fix campaigns that stored f_e2e under the
        # top_f_decode_pct name.
        f_e2e = _round_field(state, idx, "bottleneck_mining.top_f_decode_pct", "?")
    if f_e2e is None:
        f_e2e = "?"
    top_addressable = _effective_top_addressable_e2e_pct(state, idx)
    if top_addressable is None:
        top_addressable = "?"
    top_component = _round_field(state, idx, "bottleneck_mining.top_component", "?") or "?"
    amdahl_ceiling = _round_field(state, idx, "bottleneck_mining.amdahl_ceiling", "?")
    if amdahl_ceiling is None:
        amdahl_ceiling = "?"
    decode_share = _round_field(state, idx, "bottleneck_mining.decode_frac", None)
    # At the Stage 1→2 edge mining has not populated decode_frac, so the
    # baseline sweep is the normal source of decode_share_of_e2e.
    if decode_share is None:
        fb = _decode_share_from_baseline_file(state.get("_artifact_dir"), cr)
        if fb is not None:
            decode_share = fb
    if decode_share is None:
        decode_share = "?"
    top_component_prev_round = "?"
    f_e2e_prev = "?"
    if cr > 1:
        prev_round_idx = cr - 2
        top_component_prev_round = _round_field(state, prev_round_idx, "bottleneck_mining.top_component", "?") or "?"
        f_e2e_prev = _round_field(
            state, prev_round_idx, "bottleneck_mining.top_f_e2e_pct", None
        )
        if f_e2e_prev is None:
            f_e2e_prev = _round_field(
                state, prev_round_idx, "bottleneck_mining.top_f_decode_pct", "?"
            )
        if f_e2e_prev is None:
            f_e2e_prev = "?"

    # Threshold from config (schema-default fallback)
    threshold = _jget(state, "campaign.config.min_e2e_improvement_pct", None)
    threshold_warn = ""
    if threshold is None:
        threshold = "0.25"
        threshold_warn = ("⚠️ min_e2e_improvement_pct missing from state.json config — "
                          "using fallback 0.25%. Run new_target.py to initialize properly.")

    # Workload (ISL/OSL/BS)
    isl = _jget(state, "target.input_len", None)
    if isl is None:
        isl = _jget(state, "campaign.workload.input_len", "?")
    osl = _jget(state, "target.output_len", None)
    if osl is None:
        osl = _jget(state, "campaign.workload.output_len", "?")
    bs_list_raw = _jget(state, "target.batch_sizes", None)
    if bs_list_raw is None:
        bs_list_raw = _jget(state, "campaign.workload.batch_sizes", [])
    bs_list = ",".join(str(b) for b in (bs_list_raw or []))

    # Winners list/detail
    winner_list, winner_detail = _winner_strings(cur_round)

    # Round 1 baseline anchor
    r1_baseline_s = _r1_baseline(state)

    # ── PREV snapshot reads ──
    prev_exists = prev_state is not None
    prev_status = ""
    prev_stage = ""
    prev_baseline_done = ""
    prev_mining_done = ""
    prev_integ_status = ""
    prev_selected_count = 0
    prev_selected_winners_count = 0
    prev_debate_rounds_completed = 0
    prev_tracks_pass_keys = []
    prev_tracks_fail_keys = []
    prev_tracks_gating_keys = []
    prev_tracks_non_terminal = 0
    prev_mining_invalidated_present = False
    prev_mining_invalidated = None
    if prev_exists:
        pcamp = prev_state.get("campaign", {})
        prev_status = pcamp.get("status", "active") or "active"
        prev_stage = pcamp.get("current_stage", "unknown") or "unknown"
        prev_cr = pcamp.get("current_round", 1) or 1
        prev_idx = prev_cr - 1
        if prev_idx < 0:
            prev_idx = 0
        prev_baseline_done = _round_field(prev_state, prev_idx, "baseline.completed_at", "") or ""
        prev_mining_done = _round_field(prev_state, prev_idx, "bottleneck_mining.completed_at", "") or ""
        prev_integ_status = _round_field(prev_state, prev_idx, "integration.status", "") or ""
        prev_selected_count = len(_round_field(prev_state, prev_idx, "debate.selected_candidates", []) or [])
        prev_selected_winners_count = len(_round_field(prev_state, prev_idx, "debate.selected_winners", []) or [])
        prev_debate_rounds_completed = _round_field(prev_state, prev_idx, "debate.rounds_completed", 0) or 0
        ptracks = _round_field(prev_state, prev_idx, "parallel_tracks.tracks", {}) or {}
        prev_tracks_pass_keys = _verdict_keys(ptracks, *passing)
        prev_tracks_fail_keys = _verdict_keys(ptracks, "FAIL")
        prev_tracks_gating_keys = _status_keys(ptracks, "GATING_REQUIRED")
        prev_tracks_non_terminal = sum(1 for t in ptracks.values() if t.get("status") not in terminal)
        pround = _round(prev_state, prev_idx)
        prev_mining_invalidated_present = "mining_invalidated" in pround
        prev_mining_invalidated = pround.get("mining_invalidated") if prev_mining_invalidated_present else None

    new_pass_keys = _set_diff(tracks_pass_keys, prev_tracks_pass_keys) if prev_exists else []
    new_fail_keys = _set_diff(tracks_fail_keys, prev_tracks_fail_keys) if prev_exists else []
    new_gating_keys = _set_diff(tracks_gating_keys, prev_tracks_gating_keys) if prev_exists else []

    # ── Terminal transition (requires prev to exist AND prev status to differ) ──
    is_terminal_transition = False
    if (prev_exists and status in set(transitions["terminal_statuses"])
            and status != prev_status):
        is_terminal_transition = True

    # ── SOCRATIC edge chain (only when prev exists) ──
    socratic = ""
    if prev_exists:
        socratic = _build_socratic(
            stage=stage, status=status, cr=cr, idx=idx,
            baseline_done=baseline_done, prev_baseline_done=prev_baseline_done,
            mining_done=mining_done, prev_mining_done=prev_mining_done,
            isl=isl, osl=osl, bs_list=bs_list, decode_share=decode_share,
            top_component=top_component, top_component_prev_round=top_component_prev_round,
            f_e2e=f_e2e, f_e2e_prev=f_e2e_prev,
            top_addressable=top_addressable, amdahl_ceiling=amdahl_ceiling,
            selected_count=selected_count, prev_selected_count=prev_selected_count,
            selected_winners_count=selected_winners_count,
            prev_selected_winners_count=prev_selected_winners_count,
            winner_list=winner_list, winner_detail=winner_detail,
            threshold=threshold,
            debate_max_rounds=debate_max_rounds, debate_rounds_completed=debate_rounds_completed,
            prev_debate_rounds_completed=prev_debate_rounds_completed,
            new_pass_keys=new_pass_keys, new_fail_keys=new_fail_keys, new_gating_keys=new_gating_keys,
            tracks=tracks, prev_stage=prev_stage,
            track_count=track_count, tracks_non_terminal=tracks_non_terminal,
            prev_tracks_non_terminal=prev_tracks_non_terminal,
            tracks_missing_lat_opt=tracks_missing_lat_opt, team_name=team_name,
            integ_status=integ_status, prev_integ_status=prev_integ_status,
            r1_baseline_s=r1_baseline_s, state=state,
            is_terminal_transition=is_terminal_transition,
            mining_invalidated_present=mining_invalidated_present,
            mining_invalidated=mining_invalidated,
            prev_mining_invalidated_present=prev_mining_invalidated_present,
            prev_mining_invalidated=prev_mining_invalidated,
            transitions=transitions,
        )

    # ── REMINDER (stage-ladder dispatch) ──
    reminder = _build_reminder(
        transitions=transitions,
        state=state, stage=stage, status=status, cr=cr, idx=idx,
        schema_ver=schema_ver, baseline_done=baseline_done, mining_done=mining_done,
        team_name=team_name, selected_count=selected_count, tracks_started=tracks_started,
        integ_started=integ_started, integ_status=integ_status, shipped_count=shipped_count,
        track_count=track_count, tracks_non_terminal=tracks_non_terminal,
        tracks_passing=tracks_passing, tracks_missing_lat_opt=tracks_missing_lat_opt,
        audit_exists=audit_exists, audit_s1=audit_s1, audit_s2=audit_s2,
        audit_s45=audit_s45, audit_s67=audit_s67,
    )

    full_msg = ""
    if socratic:
        full_msg = socratic
    elif reminder:
        full_msg = "AMMO NEXT STEP: " + reminder
    if threshold_warn and full_msg:
        full_msg = full_msg + "\n\n" + threshold_warn

    return full_msg, is_terminal_transition


def _num_str(v):
    """Render a config number the way jq/bash did (e.g. 2.0 not 2)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # jq prints 2 for 2.0; bash fallbacks used "2.0"/"4.0". The schema
        # defaults are 1.5/2.0. Render trailing .0 to match the bash fallback
        # style only when whole; otherwise normal float repr.
        if v.is_integer():
            return str(int(v)) if False else ("%g" % v)
        return ("%g" % v)
    return str(v)


def _winner_strings(cur_round):
    debate = cur_round.get("debate", {}) or {}
    winners = (debate.get("selected_winners") or []) + (debate.get("selected_candidates") or [])
    ids = []
    for w in winners:
        if isinstance(w, dict):
            ident = w.get("op_id") or w.get("id") or w.get("name")
            if ident is None:
                ident = json.dumps(w, separators=(",", ":"))
            ids.append(str(ident))
        else:
            ids.append(str(w))
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    winner_list = ", ".join(seen)

    details = []
    for w in winners:
        if isinstance(w, dict):
            opid = w.get("op_id") or w.get("id") or "?"
            track = w.get("track_assignment", "lossless")
            sb = w.get("score_breakdown") or {}
            exp = sb.get("expected_e2e_pct")
            if exp is None:
                exp = w.get("expected_e2e_pct", 0)
            details.append("%s (%s, expected_e2e=%s%%)" % (opid, track, _jq_num(exp)))
    winner_detail = "; ".join(details)
    return winner_list, winner_detail


def _jq_num(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _decode_share_from_baseline_file(artifact_dir, cr):
    """Port of the original hook's decode_share fallback: when mining hasn't
    written top_f_decode_pct yet, average decode_share_of_e2e across the baseline
    sweep results and render it the way the old jq did (mean fraction, 2-dp).

    jq was: [.results[]? | (.baseline.aggregate.decode_share_of_e2e //
            .baseline.decode_share_of_e2e // empty)]
            | if length>0 then (add/length*100|round/100|tostring) else "?" end
    Returns a string (matching jq's `tostring`) or None when unavailable.
    """
    if not artifact_dir:
        return None
    f = Path(artifact_dir) / "rounds" / str(cr) / "sweeps" / "baseline" / "e2e_latency_results.json"
    if not f.is_file():
        return None
    data = _load_json_quiet(f)
    if not isinstance(data, dict):
        return None
    vals = []
    for r in (data.get("results") or []):
        if not isinstance(r, dict):
            continue
        base = r.get("baseline")
        if not isinstance(base, dict):
            continue
        agg = base.get("aggregate")
        # Derive from decode_avg_s/avg_s where possible: sweeps predating the
        # denominator fix store decode_avg/(prefill_avg+decode_avg) under
        # decode_share_of_e2e, which overstates the share. Prefer the aggregate
        # arm only because it is the authoritative wall on a multi-launch run.
        ds = None
        for src, wall_key in ((agg, "mean_latency"), (base, "avg_s")):
            if not isinstance(src, dict):
                continue
            dec = src.get("decode_avg_s")
            wall = src.get(wall_key)
            if isinstance(dec, (int, float)) and isinstance(wall, (int, float)) and wall > 0:
                ds = dec / wall
                break
            if src.get("decode_share_of_e2e") is not None:
                ds = src.get("decode_share_of_e2e")
                break
        if ds is None:
            continue
        try:
            vals.append(float(ds))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    # mean*100, round to nearest int, /100  → mean fraction rounded to 2 dp.
    rounded = round(sum(vals) / len(vals) * 100) / 100
    return _jq_num(rounded)


def _r1_baseline(state):
    v = _jget(state, "campaign.round_1_baseline_latency_s", None)
    if v is not None:
        return v
    e2e = _round_field(state, 0, "baseline.e2e_latency", {}) or {}
    if isinstance(e2e, dict) and e2e:
        try:
            keys = sorted(e2e.keys(), key=lambda k: float(k))
            avg = e2e[keys[0]].get("avg")
            if avg is not None:
                return avg
        except (ValueError, AttributeError, TypeError):
            pass
    return "?"


def _build_socratic(**kw):
    """Edge-triggered Socratic nudges — text VERBATIM from the hook.
    Returns the first matching message, or empty string."""
    stage = kw["stage"]
    cr = kw["cr"]
    idx = kw["idx"]
    isl, osl, bs_list = kw["isl"], kw["osl"], kw["bs_list"]
    decode_share = kw["decode_share"]
    top_component = kw["top_component"]
    f_e2e = kw["f_e2e"]
    f_e2e_prev = kw["f_e2e_prev"]
    amdahl_ceiling = kw["amdahl_ceiling"]
    threshold = kw["threshold"]

    socratic = ""

    # Stage 1 → 2
    if kw["baseline_done"] and not kw["prev_baseline_done"]:
        socratic = (
            "REASON THROUGH THIS: You're about to mine bottlenecks from this baseline. "
            "The user's target shape is ISL=%s/OSL=%s/BS=[%s]. Look at the sweep config in "
            "rounds/%s/sweeps/baseline/. Does it cover those exact values, or did it run on "
            "defaults (64/512)? If defaults: walk forward — the bottleneck you find next stage "
            "will reflect a workload the user doesn't care about. What would you do with that finding?\n\n"
            "decode_share_of_e2e = %s. For ISL/OSL = %s/%s, can you estimate from arithmetic alone "
            "what decode_share should roughly be? If your estimate and the measurement diverge "
            "significantly, which one is wrong — and how would you tell?\n\n"
            "If you cannot cite a specific file:line from the sweep config confirming the user's "
            "ISL/OSL/BS were covered, that's not uncertainty — it's unverified. Spawn ammo-investigator "
            "to compare swept workload against target.json before mining begins."
            % (isl, osl, bs_list, cr, decode_share, isl, osl)
        )

    # Re-mining same component (checked FIRST on the mining edge)
    if (not socratic and kw["mining_done"] and not kw["prev_mining_done"]
            and cr > 1 and top_component != "?"
            and top_component == kw["top_component_prev_round"]):
        socratic = (
            "REASON THROUGH THIS: Mining on the new baseline found the same top bottleneck: "
            "%s at f_e2e = %s%% (was %s%% last round).\n\n"
            "Walk through why it's still #1: did the recent SHIP attack a DIFFERENT component "
            "(so %s rose in relative ranking), or did it attack %s itself and only partially fix it?\n\n"
            "If the latter: are the remaining technology classes for %s distinct from what just "
            "shipped? If last round shipped a Triton kernel replacement and this round will also "
            "attempt Triton kernel replacement on the same component, you'll converge on no-op winners.\n\n"
            "Before spawning champions: have you appended the previous round's technology to "
            "exhausted_technologies[] so candidate generation filters it out? If not, why would this "
            "round produce a different outcome?\n\n"
            "If you cannot cite which specific technologies in exhausted_technologies[] are distinct "
            "from what you're about to attempt, spawn ammo-investigator to compare the pre-SHIP "
            "attribution with the promoted Stage-6 and activation evidence before champions repeat "
            "a dead path."
            % (top_component, f_e2e, f_e2e_prev, top_component, top_component, top_component)
        )

    # Stage 2 → 3 general mining fallback
    if not socratic and kw["mining_done"] and not kw["prev_mining_done"]:
        socratic = (
            "REASON THROUGH THIS: Mining is complete. Top component: %s at f_e2e = %s%%. "
            "You're about to spawn champions.\n\n"
            "Settle this now: f_e2e is the component's share of TOTAL wall time. f_decode is its share "
            "of decode-only time. If you computed f_e2e = f_decode × decode_busy × "
            "decode_share_of_e2e, plug those three factors in right now. Do you get back %s%%? If you "
            "cannot reproduce the number from the sidecar metrics, the debate will be built on an "
            "unverified premise.\n\n"
            "Also: what's the Amdahl ceiling for this component? (Sidecar reports amdahl_ceiling = %s.) "
            "Is a full round of champion effort justified for that ceiling, or are you chasing "
            "single-digit gains?\n\n"
            "If you cannot cite the three dilution factors and show they multiply to %s%%, spawn "
            "ammo-investigator to recompute f_e2e from raw traces before champions commit."
            % (top_component, f_e2e, f_e2e, amdahl_ceiling, f_e2e)
        )

    # Stage 3 winners selected
    if not socratic and (
        (kw["selected_count"] > 0 and kw["prev_selected_count"] == 0)
        or (kw["selected_winners_count"] > 0 and kw["prev_selected_winners_count"] == 0)
    ):
        winner_list = kw["winner_list"] or "(see debate.selected_candidates)"
        detail_line = ""
        if kw["winner_detail"]:
            detail_line = "Per-winner data: %s." % kw["winner_detail"]
        socratic = (
            "REASON THROUGH THIS: You selected [%s] as debate winners. These are the ONLY "
            "implementations this round will attempt. %s\n\n"
            "For each winner: take its expected_e2e_pct (from score_breakdown) and compare it to "
            "min_e2e_improvement_pct = %s%%. Which winners survive that math? If a winner's projection "
            "is below threshold, why is it on the list?\n\n"
            "If all winners target the same component: what's your contingency when that approach hits a "
            "wall? \"Try harder\" isn't a contingency. Is the entire round's capacity riding on one bet? Why?\n\n"
            "If you cannot cite the projection number for each winner and show it exceeds %s%%, "
            "the selection is unverified. Spawn ammo-investigator to assess whether better candidates were "
            "overlooked."
            % (winner_list, detail_line, threshold, threshold)
        )

    # Debate exceeding max rounds
    if (not socratic and kw["debate_max_rounds"] > 0
            and kw["debate_rounds_completed"] >= kw["debate_max_rounds"]
            and kw["selected_count"] == 0 and kw["selected_winners_count"] == 0
            and kw["debate_rounds_completed"] > kw["prev_debate_rounds_completed"]):
        socratic = (
            "REASON THROUGH THIS: Debate has run %s of %s rounds with no convergence. Before spawning "
            "another round, reason through WHY:\n\n"
            "Are champions disagreeing on facts (e.g., \"this kernel is 30%% of E2E\" vs \"no it's 8%%\") — "
            "which is resolvable by checking artifacts? Or on projections (resolvable by micro-experiments)? "
            "Or on philosophy (which another round won't resolve)?\n\n"
            "If it's facts: the missing data is the bottleneck, not more debate. If it's philosophy: you "
            "decide. You're the orchestrator.\n\n"
            "Is the real issue that no candidate is strong enough? If so, that's a signal the routing or "
            "mining is off — not that debate needs more time. What would the next round change that this "
            "round didn't?\n\n"
            "If you cannot cite what specific NEW information the next round would produce that this round "
            "didn't, end debate now and pick the strongest candidates. If the disagreement is factual and "
            "you cannot resolve it from existing artifacts, spawn ammo-investigator to surface the ground truth."
            % (kw["debate_rounds_completed"], kw["debate_max_rounds"])
        )

    # Stage 4-5 PASS / GATED_PASS
    if not socratic and kw["new_pass_keys"]:
        op_id = kw["new_pass_keys"][0]
        tracks = kw["tracks"]
        tk = tracks.get(op_id, {})
        ks = tk.get("kernel_speedup", "?")
        if ks is None:
            ks = "?"
        es = tk.get("e2e_speedup", "?")
        if es is None:
            es = "?"
        track_f_e2e = f_e2e
        track_component = top_component
        breakdown = _round_field(kw["state"], idx, "bottleneck_mining.component_breakdown", []) or []
        alt_comp = ""
        for comp in breakdown:
            name = comp.get("name") if isinstance(comp, dict) else None
            if name and re.search(name, op_id, re.IGNORECASE):
                alt_comp = name
                break
        if alt_comp and alt_comp != top_component:
            track_component = alt_comp
            for comp in breakdown:
                if isinstance(comp, dict) and comp.get("name") == alt_comp:
                    track_f_e2e = _jq_num(comp.get("pct", 0))
                    break
        socratic = (
            "REASON THROUGH THIS: Track %s reports kernel_speedup = %sx, e2e_speedup = %sx. The "
            "component's f_e2e was %s%% (top component: %s). If this track targets a DIFFERENT component, "
            "look up its f_e2e from bottleneck_analysis.md and use that instead.\n\n"
            "Walk Amdahl: ceiling = 1 / (1 - f_e2e/100 + (f_e2e/100) / %s), where f_e2e = %s%% for this "
            "track's component. Compute that number right now. Is %sx above or below ceiling?\n"
            "  - If above: the measurement is too good. What's the contamination story? (Cache bleed from "
            "prior track? Env-var leak? Different baseline than you think?)\n"
            "  - If kernel is fast but E2E ≈ 1.0: the kernel ran faster in isolation but isn't dispatching "
            "in E2E. Why not? Is your optimized kernel actually executing during the sweep, or is something "
            "else serving that op?\n\n"
            "Trace the causal chain in one sentence: \"kernel got %sx faster on %s%% of runtime → E2E "
            "improved by %sx → that's consistent with Amdahl because ___.\" If you cannot complete that "
            "sentence with a concrete number from e2e_latency_results.json, the verdict is unverified. Spawn "
            "ammo-investigator to check for contamination before accepting it."
            % (op_id, ks, es, track_f_e2e, top_component, ks, track_f_e2e, es, ks, track_f_e2e, es)
        )

    # Stage 4-5 FAIL
    if not socratic and kw["new_fail_keys"]:
        op_id = kw["new_fail_keys"][0]
        socratic = (
            "REASON THROUGH THIS: You're about to write FAIL for %s. FAIL means \"fundamentally "
            "unviable — no further attempt should be made.\"\n\n"
            "Walk the ladder in references/validation-defaults.md § Per-BS Verdicts and "
            "Track-Level Fallback Ladder rung-by-rung:\n"
            "  - Are there untried rungs (gating, contingency, crossover probing)? If yes, this isn't FAIL "
            "— it's \"gave up early.\" Why are you skipping those options?\n"
            "  - Are there ANY batch sizes that PASSED or showed NOISE? If yes, GATED_PASS is available. Why "
            "are you ruling it out?\n"
            "  - State the failure mode in one sentence. Is it \"algorithm fundamentally incompatible with "
            "this workload\" (genuinely FAIL) or \"implementation has a bug I didn't fix\" (not FAIL, just "
            "unfinished)? How do you know the difference?\n\n"
            "If you cannot cite the specific output of each ladder rung you tried (file:line from "
            "validation_results.md or remediation log), then FAIL is undocumented. Spawn ammo-investigator "
            "to assess whether untried remediation paths exist before writing a terminal verdict."
            % op_id
        )

    # Stage 4-5 GATING_REQUIRED
    if not socratic and kw["new_gating_keys"]:
        op_id = kw["new_gating_keys"][0]
        socratic = (
            "REASON THROUGH THIS: Track %s is now GATING_REQUIRED — PASS at some batch sizes, regression "
            "at others, env-var dispatch needed.\n\n"
            "Look at the per-BS verdicts. Where's the crossover point? Has the champion run crossover-probing "
            "sweeps (in-between BS values) to find the actual threshold, or are you guessing from the original "
            "sweep buckets? If guessing, the env-var threshold will be wrong and the gated optimization will "
            "fire at BS values where it regresses.\n\n"
            "Also: which batch sizes regressed? Are those BS values ones the user explicitly cares about "
            "(check target.json)? If the user's primary BS is in the regressing set, GATED_PASS is harder to "
            "justify — they're paying for that workload shape.\n\n"
            "If you cannot cite the exact BS threshold from a crossover-probing sweep (not inferred from the "
            "original buckets), the gating boundary is a guess. Spawn ammo-investigator to verify the gating "
            "story before shipping with an unvalidated env-var threshold."
            % op_id
        )

    # Pre-drain (all tracks terminal)
    if (not socratic and stage == "4_5_parallel_tracks"
            and kw["track_count"] > 0 and kw["tracks_non_terminal"] == 0
            and kw["prev_tracks_non_terminal"] > 0 and kw["team_name"]):
        socratic = (
            "REASON THROUGH THIS: You're about to interrupt any still-running round agents and "
            "monitors via interrupt_agent; completed or idle agents need no close action. "
            "After active turns stop, track context exists only in artifact files.\n\n"
            "Have all per-track validation_results.md been written and committed? Has every track's verdict "
            "been mirrored into state.json? If a track's e2e_latency_opt is still null but verdict is PASS, "
            "the persistent record disagrees with what the team knew — fix it before deletion or the data is "
            "lost.\n\n"
            "(%s track(s) currently have per_bs_verdict but null e2e_latency_opt.)\n\n"
            "If you cannot confirm that every track with a PASS verdict also has a non-null e2e_latency_opt "
            "in state.json, the record is incomplete. Spawn ammo-investigator to inventory rounds/%s/tracks/ "
            "against state.json before draining the round-team agents."
            % (kw["tracks_missing_lat_opt"], cr)
        )

    # Stage 5 → 6
    if not socratic and stage == "6_integration" and kw["prev_stage"] != "6_integration":
        tracks = kw["tracks"]
        n_pass = sum(1 for t in tracks.values() if t.get("verdict") == "PASS")
        n_gated = sum(1 for t in tracks.values() if t.get("verdict") == "GATED_PASS")
        socratic = (
            "REASON THROUGH THIS: You're entering integration with %s PASS and %s GATED_PASS tracks.\n\n"
            "Before the integration sweep runs, predict the combined E2E using Amdahl with all passing "
            "tracks' f_e2e values. Write down that number. If the actual sweep comes back worse than your "
            "prediction, that's an interaction effect — how would you diagnose it?\n\n"
            "  - If two tracks target the SAME component: only one ships. Which one and why? (Best E2E wins, "
            "but is the runner-up close enough to keep its branch around as a fallback?)\n"
            "  - For GATED_PASS tracks: check the env-var default. Should it be 0 (opt-in) or 1 (opt-out)? If "
            "it defaults ON and there's a regressing BS, that regression ships to production. Is that the "
            "case here?\n\n"
            "If you cannot write down the predicted combined E2E number right now (derived from each track's "
            "f_e2e via Amdahl), that prediction is unverified. Spawn ammo-investigator to inspect each track's "
            "gating block and compute the expected combined result before the integration sweep."
            % (n_pass, n_gated)
        )

    # Stage 6 SHIP
    if (not socratic
            and kw["integ_status"] in set(kw["transitions"]["integration_ship_statuses"])
            and kw["integ_status"] != kw["prev_integ_status"]):
        cum_s = _jget(kw["state"], "campaign.cumulative_speedup_vs_round1", "?")
        socratic = (
            "REASON THROUGH THIS: You're about to SHIP. Once shipped, this becomes the new baseline for all "
            "future rounds.\n\n"
            "State out loud: cumulative_speedup = round_1_baseline_latency_s / current_integrated_latency_s. "
            "Plug in the two numbers from state.json (round_1_baseline_latency_s = %s, current "
            "cumulative_speedup_vs_round1 = %s). What value do you get? If you've been multiplying "
            "round-over-round improvements, that's wrong — drift compounds. Which method did you use?\n\n"
            "The pre-SHIP mechanical checks (merge-conflict residue, regression in integration sweep, opt "
            "returncode, env promotion) — for EACH one, what specific output line did you read that confirmed "
            "it passed? \"I ran them\" is not the same as \"I read the output.\"\n\n"
            "baseline_env promotion: cat target.json right now. Are the new keys present? What's the "
            "difference between \"I updated it\" and \"I verified the file contains the update\"?\n\n"
            "If you cannot cite the specific output line for each mechanical check (not \"I ran them\" but the "
            "actual result text), that's unverified. Spawn ammo-investigator to confirm each check passed "
            "before shipping becomes the permanent new baseline."
            % (kw["r1_baseline_s"], cum_s)
        )

    # Stage 7 terminal
    if not socratic and kw["is_terminal_transition"]:
        socratic = (
            "REASON THROUGH THIS: You just set campaign.status = \"%s\". This is irreversible — no more "
            "optimization rounds will run.\n\n"
            "The canonical top_addressable_e2e_pct is %s%% and the threshold is "
            "%s%%. Confirm %s < %s from state and its Stage 2 source. Raw component "
            "or decode shares are diagnostic and cannot overturn this stop check. "
            "If the canonical value or its Stage-2 evidence is missing, mine the "
            "promoted Stage-6 result or backfill state before committing to terminal status."
            % (
                kw["status"],
                kw["top_addressable"],
                threshold,
                kw["top_addressable"],
                threshold,
            )
        )

    # mining_invalidated written
    if (not socratic and kw["mining_invalidated_present"]
            and (not kw["prev_mining_invalidated_present"]
                 or kw["mining_invalidated"] != kw["prev_mining_invalidated"])):
        mi = kw["mining_invalidated"]
        mi_str = "true" if mi is True else ("false" if mi is False else str(mi))
        socratic = (
            "REASON THROUGH THIS: You're setting mining_invalidated = %s after an EXHAUSTED round.\n\n"
            "Two scenarios exist:\n"
            "  (a) Diagnosis was right, every approach failed → keep mining valid, pivot technology "
            "(mining_invalidated = false)\n"
            "  (b) Tracks revealed the diagnosis was WRONG (e.g., the bottleneck was actually elsewhere, or "
            "the component's f_e2e was miscomputed) → re-mine (mining_invalidated = true)\n\n"
            "Which scenario are you in? Cite the specific track artifact (validation_results.md or track "
            "messaging) that supports your choice. If you can't cite evidence that the DIAGNOSIS (not the "
            "fix) was wrong, leave the flag false — re-mining without cause wastes a full round.\n\n"
            "If you cannot cite the specific track artifact (file:line) that proves the DIAGNOSIS was wrong "
            "(not just the fix), then mining_invalidated=true is unjustified. Spawn ammo-investigator to read "
            "all track results and determine whether re-mining is warranted."
            % mi_str
        )

    return socratic


def _build_reminder(**kw):
    """Stage-ladder REMINDER dispatch — text VERBATIM from the hook."""
    state = kw["state"]
    stage = kw["stage"]
    status = kw["status"]
    cr = kw["cr"]
    idx = kw["idx"]
    schema_ver = kw["schema_ver"]
    audit_exists = kw["audit_exists"]

    # Terminal campaign
    if status in set(kw["transitions"]["terminal_statuses"]):
        artifact_dir = state.get("_artifact_dir")
        report_present = bool(state.get("_report_present"))
        if report_present:
            return "Campaign complete. Session may stop."
        return "Campaign terminal. Spawn ammo-report-writer (background)."

    if stage == "1_baseline":
        if not kw["baseline_done"]:
            return "Next: dispatch ammo-researcher (task_type: baseline). T2."
        if audit_exists and not kw["audit_s1"]:
            return ("AUDIT REQUIRED (T_AUDIT_S1): Stage 1 baseline complete. Spawn ammo-auditor "
                    "(4-phase: inventory → reconstruction → checklist → reconciliation). "
                    "Stage: stage_1.")
        return ("Baseline done. Ingest the sweep with `ammo_state.py ingest-baseline "
                "--state <state.json> --from rounds/%s/sweeps/baseline/e2e_latency_results.json` "
                "(never retype the per-BS numbers), then set stage → 2_bottleneck_mining "
                "and dispatch researcher (task_type: mining). T4." % kw["cr"])

    if stage == "2_bottleneck_mining":
        if not kw["mining_done"]:
            return "Next: dispatch ammo-researcher (task_type: mining). Analyzes existing traces. T4."
        s2_applicable = False
        parts = schema_ver.split(".")
        try:
            s_major = int(parts[0])
        except (ValueError, IndexError):
            s_major = 4
        try:
            s_minor = int(parts[1])
        except (ValueError, IndexError):
            s_minor = 0
        if s_major > 4 or (s_major == 4 and s_minor >= 1):
            s2_applicable = True
        if s2_applicable and audit_exists and not kw["audit_s2"]:
            return ("AUDIT REQUIRED (T_AUDIT_S2): Stage 2 mining complete. Spawn ammo-auditor "
                    "(4-phase: inventory → reconstruction → checklist → reconciliation). "
                    "Stage: stage_2.")
        return "Mining done. Set stage → 3_debate."

    if stage == "3_debate":
        if not kw["team_name"]:
            return "Next: use spawn_agent to start 2-4 ammo-champion agents. No monitors for debate."
        if kw["selected_count"] == 0:
            return ("Debate in progress. Min 1 round (full A/B/C). Round 2 if champions declare open "
                    "items. Custom kernel mandate (Triton/CuTeDSL/CUTLASS/CUDA).")
        if not kw["tracks_started"]:
            return ("Winners selected. Use interrupt_agent on any debate champion still running; "
                    "completed or idle champions need no close action. Then spawn one "
                    "ammo-implementer + paired monitor per winner.")
        return ""

    if stage == "4_5_parallel_tracks":
        if kw["tracks_missing_lat_opt"] > 0 and kw["tracks_non_terminal"] == 0:
            return ("%s track(s) have per_bs_verdict but null e2e_latency_opt. Extract per-BS latency "
                    "map from each track's rounds/{CR}/sweeps/opt/{op_id}/e2e_latency_results.json (opt "
                    "label: avg_s→avg, p50_s→p50, p10_s→p10, p25_s→p25, p75_s→p75, "
                    "p90_s→p90, p99_s→p99) and write to "
                    ".campaign.rounds[$IDX].parallel_tracks.tracks[op_id].e2e_latency_opt. Same shape as "
                    "baseline.e2e_latency." % kw["tracks_missing_lat_opt"])
        if kw["track_count"] > 0 and kw["tracks_non_terminal"] > 0:
            return ("Tracks running. Wait for ALL to reach terminal (PASS/GATED_PASS/FAIL). Do NOT advance "
                    "to Stage 6 early.")
        if kw["track_count"] > 0 and kw["tracks_non_terminal"] == 0 and not kw["integ_started"]:
            if audit_exists and not kw["audit_s45"]:
                return ("AUDIT REQUIRED (T_AUDIT_S45): All parallel tracks terminal. Spawn ammo-auditor "
                        "(4-phase: inventory → reconstruction → checklist → reconciliation). "
                        "Stage: stage_45.")
            if kw["tracks_passing"] == 1:
                return ("All tracks terminal (1 passer). Interrupt any still-running round agents and "
                        "monitors via interrupt_agent; completed or idle agents need no close action. "
                        "Then Stage 6 single-track short-circuit: "
                        "copy Stage 5 results to integration slot, set status=single_pass, run Pre-SHIP "
                        "checks, SHIP. See orchestration/integration-logic.md § Single-Track "
                        "Short-Circuit.")
            return ("All tracks terminal. Interrupt any still-running round agents and monitors via "
                    "interrupt_agent; completed or idle agents need no close action. Then proceed "
                    "to Stage 6 integration (file-set conflict analysis).")
        return ""

    if stage == "6_integration":
        integ_status = kw["integ_status"]
        shipped_count = kw["shipped_count"]
        integ_terminal = integ_status in set(kw["transitions"]["integration_terminal_statuses"])
        if shipped_count > 0 or integ_terminal:
            if audit_exists and not kw["audit_s67"]:
                return ("AUDIT REQUIRED (T_AUDIT_S67): Spawn ammo-auditor (4-phase: inventory → "
                        "reconstruction → checklist → reconciliation). Stage: stage_67. No "
                        "auto-pass — auditor always runs.")
            if shipped_count > 0:
                return ("T_AUDIT_S67 passed. Set stage → 7_campaign_eval. A material "
                        "SHIP must create the next round at 2_bottleneck_mining and mine "
                        "the promoted Stage-6 result before the threshold check; do not "
                        "reprofile. The all-diluted exception reuses valid mining and "
                        "enters 3_debate.")
            return ""
        if integ_status in set(kw["transitions"]["integration_ship_statuses"]):
            return (
                "Integration validation already passed. Run Pre-SHIP checks and SHIP; "
                "do not rerun the single-track or combined integration sweep."
            )
        if shipped_count == 0:
            if kw["tracks_passing"] == 1:
                return ("Single passer — run short-circuit: copy Stage 5 results into "
                        "rounds/{CR}/sweeps/integration/, set integration.status=single_pass, write "
                        "e2e_latency_combined. Then Pre-SHIP checks + SHIP. See "
                        "orchestration/integration-logic.md § Single-Track Short-Circuit.")
            if kw["tracks_passing"] > 1:
                return ("Multiple passers — run combined integration sweep with .venv/bin/python + "
                        "--fresh-cache. Pre-SHIP checks: grep merge-conflict residue, jq dual-verdict "
                        "override, jq opt returncode. If all pass → SHIP + merge + env promotion + "
                        "golden-refs capture (~15s).")
            return "Run integration decision matrix. No passing tracks → round EXHAUSTED."
        return ""

    if stage.startswith("7_campaign_eval"):
        if audit_exists and not kw["audit_s67"]:
            return ("AUDIT REQUIRED (T_AUDIT_S67): audit.stage_67.passed_at not set. Spawn ammo-auditor "
                    "(4-phase: inventory → reconstruction → checklist → reconciliation). "
                    "Stage: stage_67.")
        return ("Interrupt any still-running round agents and monitors with interrupt_agent; completed "
                "or idle agents need no close action. Mechanical check: compare "
                "bottleneck_mining.top_addressable_e2e_pct with campaign.config."
                "min_e2e_improvement_pct — no qualitative judgment. Material SHIP → create the next "
                "round at 2_bottleneck_mining, mine promoted Stage-6 evidence without "
                "reprofiling, then re-run the same mechanical check; "
                "all-diluted SHIP → 3_debate (reuse valid mining). "
                "EXHAUSTED → compare the still-valid mining now: below the floor is "
                "campaign_exhausted, otherwise 3_debate with a technology pivot. "
                "Override: set mining_invalidated=true on the previous round if mining was "
                "wrong, then re-enter Stage 2 on the same measured baseline. No user prompting.")

    return ""


# ---------------------------------------------------------------------------
# advance: round/stage transition
# ---------------------------------------------------------------------------

def _new_round_skeleton(round_id, max_rounds=4):
    """Round skeleton consistent with new_target.py's shape + the schema."""
    return {
        "round_id": round_id,
        "status": "IN_PROGRESS",
        "team_name": None,
        "profiling_baseline_path": None,
        "baseline": {"started_at": None, "completed_at": None, "e2e_latency": None, "per_bs_verdict": None},
        "bottleneck_mining": {"started_at": None, "completed_at": None, "top_bottleneck_share_pct": None},
        "debate": {
            "started_at": None, "completed_at": None, "candidates": [],
            "rounds_completed": 0, "max_rounds": max_rounds, "selected_winners": [],
        },
        "parallel_tracks": {"started_at": None, "completed_at": None, "tracks": {}},
        "integration": {
            "started_at": None, "completed_at": None, "status": "pending",
            "passing_candidates": [], "failed_candidates": [], "selected_candidates": [],
            "conflict_analysis": None, "combined_patch_branch": None, "combined_e2e_result": None,
            "e2e_latency_combined": None, "per_bs_verdict": None, "commit_sha": None,
            "final_decision": None, "resolver_invoked": None, "resolver_outcome": None,
            "conflicting_tracks": None,
        },
        "campaign_eval": {"started_at": None, "completed_at": None},
        "audit": {},
        "shipped": [], "dropped": [],
        "cumulative_speedup_after": None, "combined_e2e_speedup_x": None,
        "combined_e2e_delta_pp": None, "note": None, "round_summary": None,
    }


def advance(state, outcome, transitions):
    """Apply a round/stage transition. Returns (new_state, new_stage)."""
    campaign = state["campaign"]
    cur_round = campaign.get("current_round", 1) or 1
    idx = cur_round - 1
    rounds = campaign.get("rounds", [])
    cur = rounds[idx] if 0 <= idx < len(rounds) else {}

    if outcome == "CONTINUE":
        return state, campaign.get("current_stage")

    if outcome == "SHIP":
        new_stage = transitions["outcome_next_stage"]["SHIP"]
        shipped = cur.get("shipped") or []
        tracks = (cur.get("parallel_tracks") or {}).get("tracks") or {}
        if shipped and all(
            isinstance(tracks.get(op_id), dict)
            and tracks[op_id].get("diluted") is True
            for op_id in shipped
        ):
            new_stage = transitions["outcome_next_stage"]["EXHAUSTED"]
        cur["status"] = "SHIPPED"
    elif outcome == "EXHAUSTED":
        if cur.get("mining_invalidated") is True:
            new_stage = transitions["outcome_next_stage"]["EXHAUSTED_MINING_INVALIDATED"]
        else:
            new_stage = transitions["outcome_next_stage"]["EXHAUSTED"]
        cur["status"] = "EXHAUSTED"
    else:
        raise ValueError("unknown outcome: %s" % outcome)

    new_round_id = cur_round + 1
    skel = _new_round_skeleton(new_round_id)
    campaign.setdefault("rounds", []).append(skel)
    campaign["current_round"] = new_round_id
    campaign["current_stage"] = new_stage
    return state, new_stage


# ---------------------------------------------------------------------------
# enrich: FE Metric Enrichment
# ---------------------------------------------------------------------------

def _load_json_quiet(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _walk_values(doc, names):
    """Collect every value stored under any of `names`, at any nesting depth.
    Champions author the gate scripts per track, so the number is rarely at the
    top level. Returns {name: [values in document order]}."""
    found = {name: [] for name in names}
    stack = [doc]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if key in found:
                    found[key].append(value)
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _first_number(found, *names):
    for name in names:
        for value in found.get(name, ()):
            num = _as_number(value)
            if num is not None:
                return num
    return None


def _max_number(found, *names):
    vals = [
        _as_number(v) for name in names for v in found.get(name, ())
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _first_status(found, *names):
    for name in names:
        for value in found.get(name, ()):
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, bool):
                return "PASS" if value else "FAIL"
    return None


def _shapes_tested(doc, found):
    """The contract does not name a count, so derive it: an explicit count wins,
    otherwise the length of the per-shape collection the champion emitted."""
    explicit = None
    for value in found.get("shapes_tested", ()):
        explicit = _as_int(value)
        if explicit is not None:
            return explicit
    for key in ("shapes", "per_shape", "tests", "cases", "results",
                "per_batch", "per_batch_size", "batch_sizes_tested"):
        for value in _walk_values(doc, (key,)).get(key, ()):
            if isinstance(value, (list, dict)) and value:
                return len(value)
    return None


def enrich_gate_5_1a(state, op_id, results):
    """Populate gate_5_1a_metrics from gate_5_1a_results.json.

    The champion authors that file fresh per track, so read the CONTRACT names
    (ammo-impl-champion.md § inline gates) at any depth instead of assuming a
    flat shape. `overall` accepts the recorded synonyms; `max_abs_err` takes the
    worst value seen across shapes.
    """
    idx = _round_idx(state)
    track = _ensure_track(state, idx, op_id)
    src = results or {}
    found = _walk_values(src, (
        "overall", "gate_5_1a", "correctness", "overall_verdict",
        "overall_pass", "verdict", "status",
        "max_abs_err", "max_abs_diff", "max_abs_error",
        "shapes_tested",
    ))
    metrics = {
        "overall": _first_status(
            found, "overall", "gate_5_1a", "correctness", "overall_verdict",
            "overall_pass", "verdict", "status",
        ),
        "max_abs_err": _max_number(found, "max_abs_err", "max_abs_diff", "max_abs_error"),
        "shapes_tested": _shapes_tested(src, found),
    }
    # The schema types every property as a bare number/string, so an unresolved
    # field must be OMITTED, not written as null. A null write fails schema
    # validation and discards the fields that did resolve.
    track["gate_5_1a_metrics"] = {k: v for k, v in metrics.items() if v is not None}
    return metrics


def enrich_gate_5_2(state, op_id, results):
    """Populate gate_5_2_metrics from gate_5_2_results.json.

    Reads the CONTRACT names (kernel_speedup / kernel_speedup_warm /
    kernel_speedup_cold, ammo-impl-champion.md § inline gates) at any depth. The
    state field names keep their `weighted_` prefix for schema compatibility.
    A bare `speedup` is the warm figure only when no warm/cold split exists.
    """
    idx = _round_idx(state)
    track = _ensure_track(state, idx, op_id)
    src = results or {}
    found = _walk_values(src, (
        "kernel_speedup", "kernel_speedup_cold", "kernel_speedup_warm",
        "weighted_speedup_cold", "weighted_speedup_warm", "weighted_speedup",
        "speedup_cold", "speedup_warm", "cold_speedup", "warm_speedup",
        "speedup", "shapes_tested",
    ))
    cold = _first_number(found, "kernel_speedup_cold", "weighted_speedup_cold",
                         "speedup_cold", "cold_speedup")
    warm = _first_number(found, "kernel_speedup_warm", "weighted_speedup_warm",
                         "speedup_warm", "warm_speedup")
    if warm is None:
        warm = _first_number(found, "kernel_speedup", "weighted_speedup", "speedup")
    metrics = {
        "weighted_speedup_cold": cold,
        "weighted_speedup_warm": warm,
        "shapes_tested": _shapes_tested(src, found),
    }
    # See enrich_gate_5_1a: omit unresolved fields rather than writing null.
    track["gate_5_2_metrics"] = {k: v for k, v in metrics.items() if v is not None}
    return metrics


def _gate_enrich_missing(metrics):
    return sorted(k for k, v in metrics.items() if v is None)


def _ensure_track(state, idx, op_id):
    rounds = state["campaign"]["rounds"]
    rnd = rounds[idx]
    pt = rnd.setdefault("parallel_tracks", {})
    tracks = pt.setdefault("tracks", {})
    return tracks.setdefault(op_id, {})


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _empty_mining_fields():
    return {
        "top_component": None,
        "top_f_e2e_pct": None,
        "top_addressable_e2e_pct": None,
        "top_f_decode_pct": None,
        "amdahl_ceiling": None,
        "decode_frac": None,
        "component_breakdown": None,
        "top_bottleneck_share_pct": None,
    }


def parse_mined_json(doc):
    """Parse rounds/{N}/mining/mined.json (schema `mine_trace/1`) into the
    mining enrichment fields. This is the MECHANIZED producer; the markdown
    parse below stays as the fallback for hand-written analyses.

    Reads only the stable per_bs subset (bs, decode_share_of_e2e, families[]
    with label/f_e2e); extra fields are tolerated. Missing or malformed source
    => fill what you can, null the rest. Never invent values.
    """
    out = _empty_mining_fields()
    if not isinstance(doc, dict):
        return out
    per_bs = doc.get("per_bs")
    if isinstance(per_bs, dict):
        per_bs = list(per_bs.values())
    if not isinstance(per_bs, list):
        return out

    shares = []
    totals = {}
    addressable = []
    for entry in per_bs:
        if not isinstance(entry, dict):
            continue
        share = _as_number(entry.get("decode_share_of_e2e"))
        if share is not None and 0.0 <= share <= 1.0:
            shares.append(share)
        families = entry.get("families")
        if not isinstance(families, list):
            continue
        for fam in families:
            if not isinstance(fam, dict):
                continue
            label = fam.get("label")
            f_e2e = _as_number(fam.get("f_e2e"))
            if not isinstance(label, str) or not label or f_e2e is None:
                continue
            pct = f_e2e * 100.0 if 0.0 <= f_e2e <= 1.0 else f_e2e
            totals.setdefault(label, []).append(pct)
            # `addressable_e2e` is f_e2e x (1 - 1/physical_ceiling): the same
            # quantity the markdown table's final column publishes. It is null
            # for a family with no disclosed ceiling.
            removable = _as_number(fam.get("addressable_e2e"))
            if removable is not None:
                addressable.append(
                    removable * 100.0 if 0.0 <= removable <= 1.0 else removable
                )

    if shares:
        out["decode_frac"] = _mean_fraction(shares)
    if addressable:
        out["top_addressable_e2e_pct"] = max(addressable)
    if totals:
        breakdown = [
            {"name": label, "pct": sum(vals) / len(vals)}
            for label, vals in totals.items()
        ]
        breakdown.sort(key=lambda b: b["pct"], reverse=True)
        out["component_breakdown"] = breakdown
        top = breakdown[0]
        out["top_component"] = top["name"]
        out["top_f_e2e_pct"] = top["pct"]
        out["amdahl_ceiling"] = top["pct"]
        # The gate's v3-compat alias of top_f_e2e_pct: the top family's share of
        # total wall time. No other script produces it, so derive it here.
        out["top_bottleneck_share_pct"] = top["pct"]
    return out


def parse_mining_md(text):
    """Parse the structured dilution/components markdown table + profile summary
    from bottleneck_analysis.md into the mining enrichment fields. Missing or
    malformed source => fill what you can, null the rest. Never invent values.

    Returns the top raw component share, the largest physically addressable
    impact, and the remaining mining enrichment fields.
    """
    out = _empty_mining_fields()
    if not text:
        return out

    rows = _parse_first_component_table(text)
    if rows:
        breakdown = []
        addressable = []
        for name, f_e2e, removable_e2e in rows:
            pct = f_e2e
            # f_e2e is published as a decimal fraction in the dilution table
            # (e.g. 0.30). component_breakdown.pct is a PERCENT in [0,100].
            if pct is not None and 0.0 <= pct <= 1.0:
                pct = pct * 100.0
            breakdown.append({"name": name, "pct": pct})
            if removable_e2e is not None:
                removable_pct = removable_e2e
                if 0.0 <= removable_pct <= 1.0:
                    removable_pct *= 100.0
                addressable.append(removable_pct)
        breakdown = [b for b in breakdown if b["pct"] is not None]
        if breakdown:
            breakdown.sort(key=lambda b: b["pct"], reverse=True)
            out["component_breakdown"] = breakdown
            top = breakdown[0]
            out["top_component"] = top["name"]
            out["top_f_e2e_pct"] = top["pct"]
            out["amdahl_ceiling"] = top["pct"]
            out["top_bottleneck_share_pct"] = top["pct"]
        if addressable:
            out["top_addressable_e2e_pct"] = max(addressable)

    decode_frac = _parse_decode_frac(text)
    if decode_frac is not None:
        out["decode_frac"] = decode_frac
    return out


def _fold_unicode(text):
    """Fold the typographic characters the researcher contract emits into ASCII
    so column matching is stable. `f_e2e × (1−1/ceiling)` and
    `f_e2e x (1-1/ceiling)` must match the same column.
    """
    folded = unicodedata.normalize("NFKD", text)
    for src, dst in (
        ("×", "x"), ("✕", "x"), ("✖", "x"), ("⋅", "x"),
        ("−", "-"), ("–", "-"), ("—", "-"), ("―", "-"),
        ("⁄", "/"),
    ):
        folded = folded.replace(src, dst)
    return folded


def _parse_first_component_table(text):
    """Return name, raw f_e2e, and removable impact from the mining table."""
    lines = _fold_unicode(text).splitlines()
    f_idx = None
    addressable_idx = None
    name_idx = 0
    rows = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table and rows:
                break
            f_idx = None
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        lowered = [c.lower() for c in cells]
        if f_idx is None:
            if "f_e2e" in lowered:
                f_idx = lowered.index("f_e2e")
                addressable_idx = next(
                    (
                        i
                        for i, cell in enumerate(lowered)
                        if "f_e2e" in cell
                        and "1-1/ceiling" in cell.replace(" ", "")
                    ),
                    None,
                )
                # name col: first column unless an explicit 'component' header
                if "component" in lowered:
                    name_idx = lowered.index("component")
                else:
                    name_idx = 0
                in_table = True
            continue
        if re.match(r"^[\s|:-]+$", line):
            continue
        if f_idx < len(cells) and name_idx < len(cells):
            name = cells[name_idx]
            raw = cells[f_idx].rstrip("%")
            try:
                val = float(raw)
            except ValueError:
                continue
            if name:
                removable = None
                if addressable_idx is not None and addressable_idx < len(cells):
                    try:
                        removable = float(cells[addressable_idx].rstrip("%"))
                    except ValueError:
                        pass
                rows.append((name, val, removable))
    return rows


def _parse_decode_frac(text):
    """Return decode_frac in [0,1].

    Prefer an explicit profile-summary `decode_frac: <n>` scalar. When absent,
    average the `decode_share_of_e2e` column of the mandated
    `## Workload Dilution (per BS)` table — that table is what the researcher
    contract emits, and the scalar is optional.
    """
    folded = _fold_unicode(text)
    m = re.search(r"decode[_ ]frac(?:tion)?\s*[:=|]\s*([0-9]*\.?[0-9]+)", folded, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    shares = _parse_decode_share_column(folded)
    if shares:
        return _mean_fraction(shares)
    return None


def _mean_fraction(values):
    """Mean of per-BS fractions, rounded to 2 dp — same semantics as
    _decode_share_from_baseline_file's jq port."""
    return round(sum(values) / len(values) * 100) / 100


def _parse_decode_share_column(folded):
    """Collect the per-BS `decode_share_of_e2e` cells of the first table that
    carries that column. Returns a list of fractions in [0,1]."""
    share_idx = None
    shares = []
    for line in folded.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if share_idx is not None and shares:
                break
            share_idx = None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        lowered = [c.lower() for c in cells]
        if share_idx is None:
            if "decode_share_of_e2e" in lowered:
                share_idx = lowered.index("decode_share_of_e2e")
            continue
        if re.match(r"^[\s|:-]+$", line):
            continue
        if share_idx >= len(cells):
            continue
        try:
            v = float(cells[share_idx].rstrip("%"))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            shares.append(v)
    return shares


def resolve_mining_fields(mined_doc, text):
    """Merge the two mining producers. mined.json (mechanized) wins for every
    field it carries; the bottleneck_analysis.md parse fills the rest — it is
    the fallback when mined.json is absent, unreadable, or has no ceiling for
    the top family (so no addressable impact)."""
    fields = parse_mined_json(mined_doc)
    md = parse_mining_md(text)
    for k, v in md.items():
        if fields.get(k) is None:
            fields[k] = v
    return fields


def _mined_json_path(artifact_dir, round_id, from_path=None):
    """Locate rounds/{N}/mining/mined.json. Prefer the sibling of an explicit
    --from artifact so a non-canonical layout still resolves."""
    if from_path:
        sibling = Path(from_path).parent / "mined.json"
        if sibling.is_file():
            return sibling
    if artifact_dir:
        canonical = (Path(artifact_dir) / "rounds" / str(round_id)
                     / "mining" / "mined.json")
        if canonical.is_file():
            return canonical
    return None


def enrich_mining(state, text, mined_doc=None):
    idx = _round_idx(state)
    rnd = state["campaign"]["rounds"][idx]
    mining = rnd.setdefault("bottleneck_mining", {})
    fields = resolve_mining_fields(mined_doc, text)
    for k, v in fields.items():
        # A parse that cannot see a field must not delete a recorded value:
        # the round-advance gate blocks on null, so a null write is a wedge.
        if v is None and mining.get(k) is not None:
            continue
        mining[k] = v
    return fields


def enrich_debate(state, debate_dir):
    idx = _round_idx(state)
    rnd = state["campaign"]["rounds"][idx]
    debate = rnd.setdefault("debate", {})
    d = Path(debate_dir)

    proposal_count = None
    rationale_count = None
    champions_count = None
    proposals_dir = d / "proposals"
    if proposals_dir.is_dir():
        proposal_files = [p for p in proposals_dir.iterdir() if p.is_file() and p.suffix == ".md"]
        proposal_count = len(proposal_files)
        champ_ids = set()
        for p in proposal_files:
            m = re.match(r"(.+?)_proposal\.md$", p.name)
            if m:
                champ_ids.add(m.group(1))
        champions_count = len(champ_ids) if champ_ids else None

    if d.is_dir():
        rationale = 0
        for p in d.rglob("*.md"):
            n = p.name
            if n.endswith("_argument.md") or n.endswith("_rebuttal.md") or "_critique_" in n:
                rationale += 1
        rationale_count = rationale

    winners = debate.get("selected_winners") or []
    winners = list(winners) if winners else None
    result = None
    if winners:
        result = "%d winner(s) selected: %s" % (len(winners), ", ".join(winners))

    enriched = {
        "proposal_count": proposal_count,
        "rationale_count": rationale_count,
        "champions_count": champions_count,
        "winners": winners,
        "result": result,
    }
    for k, v in enriched.items():
        debate[k] = v
    return enriched


# ---------------------------------------------------------------------------
# ingest-baseline: Stage 1 handoff — sweep JSON -> baseline.e2e_latency
# ---------------------------------------------------------------------------

_PERCENTILE_KEYS = ("p10", "p25", "p50", "p75", "p90", "p99")


def _latency_seconds(entry):
    """Resolve the average latency of one sweep arm, in seconds.

    Mirrors the ladder in references/validation-defaults.md and
    generate_validation_report.py: aggregate mean first (legacy artifacts),
    then avg_latency, then avg_s.
    """
    if not isinstance(entry, dict):
        return None
    agg = entry.get("aggregate")
    if isinstance(agg, dict):
        v = _as_number(agg.get("mean_latency"))
        if v is not None:
            return v
    for key in ("avg_latency", "avg_s"):
        v = _as_number(entry.get(key))
        if v is not None:
            return v
    return None


def parse_baseline_sweep(doc, label=None):
    """Build a baseline.e2e_latency map from an e2e_latency_results.json.

    Returns (latency_map, errors). Applies the `_s`-suffix strip the server-side
    normalizer already performs (avg_s -> avg, p50_s -> p50). A percentile that
    is not measured stays absent; it is never synthesized from the mean.
    """
    errors = []
    if not isinstance(doc, dict):
        return None, ["sweep JSON is not an object"]
    rows = doc.get("results")
    if not isinstance(rows, list) or not rows:
        return None, ["sweep JSON has no results[] rows"]
    if label is None:
        bench = doc.get("bench")
        label = bench.get("baseline_label") if isinstance(bench, dict) else None
        label = label or "baseline"

    out = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("sweep row is not an object")
            continue
        bs = row.get("batch_size")
        if isinstance(bs, bool) or not isinstance(bs, int):
            errors.append("sweep row has no integer batch_size")
            continue
        arm = row.get(label)
        if not isinstance(arm, dict):
            errors.append("sweep row bs=%s has no '%s' arm" % (bs, label))
            continue
        metrics = arm.get("metrics") if isinstance(arm.get("metrics"), dict) else {}
        avg = _latency_seconds(arm)
        if avg is None:
            avg = _latency_seconds(metrics)
        if avg is None:
            errors.append("sweep row bs=%s has no resolvable average latency" % bs)
            continue
        entry = {"avg": avg}
        for pkey in _PERCENTILE_KEYS:
            v = _as_number(metrics.get(pkey + "_s"))
            if v is None:
                v = _as_number(arm.get(pkey + "_s"))
            if v is not None:
                entry[pkey] = v
        out[str(bs)] = entry
    if not out:
        errors.append("no usable rows in sweep JSON")
        return None, errors
    return out, errors


def ingest_baseline(state, doc, source_path, round_id=None, label=None):
    """Write the per-BS baseline latency map into the round's baseline block.

    Replaces the hand-retyped `set` calls of the Stage-1 handoff. Leaves
    per_bs_verdict alone (its enum belongs to track/integration evaluation) and
    records profiling_baseline_path so the sweep that produced the numbers stays
    citable. Returns (latency_map, errors).
    """
    idx = _round_idx(state) if round_id is None else int(round_id) - 1
    rounds = state.get("campaign", {}).get("rounds", []) or []
    if idx < 0 or idx >= len(rounds):
        return None, ["round %s does not exist in state.json" % (round_id or idx + 1)]
    latency, errors = parse_baseline_sweep(doc, label)
    if latency is None:
        return None, errors
    rnd = rounds[idx]
    baseline = rnd.setdefault("baseline", {})
    baseline["e2e_latency"] = latency
    if source_path:
        rnd["profiling_baseline_path"] = str(source_path)
    return latency, errors


# ---------------------------------------------------------------------------
# backfill: Resume step 9 — fill missing decode_frac/component_breakdown
# ---------------------------------------------------------------------------

def backfill(state, artifact_dir, transitions):
    """For every round with bottleneck_mining.completed_at set but
    decode_frac/component_breakdown null, re-parse the round's mined.json (then
    bottleneck_analysis.md) and fill the fields. Returns a list of summaries."""
    backfilled = []
    rounds = state.get("campaign", {}).get("rounds", [])
    target_fields = transitions["backfill_mining_fields"]
    for i, rnd in enumerate(rounds):
        mining = rnd.get("bottleneck_mining", {}) or {}
        if not mining.get("completed_at"):
            continue
        missing = [f for f in target_fields if mining.get(f) is None]
        if not missing:
            continue
        round_id = rnd.get("round_id", i + 1)
        md_path = Path(artifact_dir) / "rounds" / str(round_id) / "mining" / "bottleneck_analysis.md"
        text = None
        if md_path.is_file():
            text = md_path.read_text(encoding="utf-8")
        mined_path = _mined_json_path(artifact_dir, round_id)
        mined_doc = _load_json_quiet(mined_path) if mined_path else None
        parsed = resolve_mining_fields(mined_doc, text or "")
        filled = []
        for f in target_fields:
            if mining.get(f) is None and parsed.get(f) is not None:
                mining[f] = parsed[f]
                filled.append(f)
        # Also opportunistically fill the other required-once-completed fields
        for f in (
            "top_component",
            "top_f_e2e_pct",
            "top_addressable_e2e_pct",
            "amdahl_ceiling",
            "top_bottleneck_share_pct",
        ):
            if mining.get(f) is None and parsed.get(f) is not None:
                mining[f] = parsed[f]
                filled.append(f)
        rnd["bottleneck_mining"] = mining
        if filled:
            backfilled.append({"round": round_id, "filled": filled})

    # Schema 4.1+ makes the winning proposal locator part of the typed
    # cross-agent contract. Resume repair is deterministic when the selected
    # entry already cites its proposal; never guess among multiple files.
    for i, rnd in enumerate(rounds):
        round_id = rnd.get("round_id", i + 1)
        selected = (rnd.get("debate") or {}).get("selected_candidates") or []
        filled = []
        for cand in selected:
            if not isinstance(cand, dict) or cand.get("proposal_file"):
                continue
            cited = cand.get("cited_evidence") or []
            matches = [
                str(path)
                for path in cited
                if isinstance(path, str)
                and "/debate/proposals/" in path
                and path.endswith(".md")
            ]
            if len(set(matches)) == 1:
                cand["proposal_file"] = matches[0]
                filled.append(
                    "selected_candidates.%s.proposal_file"
                    % (cand.get("op_id") or "unknown")
                )
        if filled:
            backfilled.append({"round": round_id, "filled": filled})
    return backfilled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _schema_unavailable(message):
    if os.environ.get("AMMO_VALIDATE_FAIL_OPEN", "") == "1":
        return []
    return [
        "state.json validation could not run (%s) — restore the managed "
        "schema/jsonschema runtime or set AMMO_VALIDATE_FAIL_OPEN=1 to bypass "
        "(degraded)." % message
    ]


def _load_schema_checked(state_path, explicit=None):
    try:
        schema, schema_path = load_schema(state_path, explicit)
    except Exception as exc:
        return None, None, _schema_unavailable(
            "schema load failed: %s: %s" % (type(exc).__name__, exc)
        )
    if schema is None:
        return None, schema_path, _schema_unavailable("schema file unavailable")
    if not _HAVE_JSONSCHEMA or Draft202012Validator is None:
        return None, schema_path, _schema_unavailable("jsonschema unavailable")
    return schema, schema_path, []


def _schema_errors_checked(state, schema):
    if schema is None:
        return []
    try:
        errs = schema_errors(state, schema)
    except Exception as exc:
        return _schema_unavailable(
            "validator failed: %s: %s" % (type(exc).__name__, exc)
        )
    if errs is None:
        return _schema_unavailable("jsonschema validator unavailable")
    return errs


def _validate_before_write(state, state_path, schema, transitions, prev_state=None):
    """Run schema + gate validation. Returns (ok, error_lines_or_reason)."""
    errs = _schema_errors_checked(state, schema)
    if errs:
        return False, errs
    reason = gate_violation(state, transitions, prev_state)
    if reason:
        return False, [reason]
    return True, []


def cmd_get(args):
    state_path = resolve_state_path(args.target)
    if not Path(state_path).is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    doc = load_state(state_path)
    if args.field:
        found, val = get_field(doc, args.field)
        if not found:
            print("FAIL: field not found: %s" % args.field, file=sys.stderr)
            return 1
        if isinstance(val, (dict, list)):
            print(json.dumps(val))
        elif val is None:
            print("null")
        else:
            print(val)
    else:
        print(json.dumps(doc, indent=2))
    return 0


def cmd_set(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    with state_lock(state_path):
        doc = load_state(state_path)
        prev_doc = copy.deepcopy(doc)
        if len(args.field) != len(args.value):
            print("FAIL: --field and --value must be paired (equal counts)", file=sys.stderr)
            return 1
        for field, raw in zip(args.field, args.value):
            try:
                val = json.loads(raw)
            except json.JSONDecodeError:
                print("FAIL: --value is not valid JSON: %s" % raw, file=sys.stderr)
                return 1
            try:
                set_field(doc, field, val)
            except (KeyError, IndexError) as exc:
                print("FAIL: bad --field path (file left untouched): %s" % exc,
                      file=sys.stderr)
                return 1

        transitions = load_transitions()
        schema, _, errs = _load_schema_checked(state_path, args.schema)
        if not errs:
            errs = _schema_errors_checked(doc, schema)
        if errs:
            print("FAIL: schema validation errors (file left untouched):")
            for e in errs:
                print(e)
            return 1
        reason = gate_violation(doc, transitions, prev_doc)
        if reason:
            print("FAIL: validation failed after set (file left untouched):")
            print(reason)
            return 1
        atomic_write(state_path, doc)
    return 0


# P4#6: fail-closed escape hatch + remediation text. Default posture is
# fail-CLOSED (emit decision:block) when validation CANNOT RUN; setting
# AMMO_VALIDATE_FAIL_OPEN=1 reverts to the legacy fail-open for degraded envs.
def _fail_open_enabled():
    return os.environ.get("AMMO_VALIDATE_FAIL_OPEN", "") == "1"


_FAIL_CLOSED_LIB_REMEDIATION = (
    "state.json validation could not run (jsonschema unavailable) — install "
    "jsonschema into the session .venv or set AMMO_VALIDATE_FAIL_OPEN=1 to "
    "bypass (degraded)."
)
_FAIL_CLOSED_PARSE_REMEDIATION = (
    "state.json validation could not run (state file is not parseable JSON) — "
    "fix the file or set AMMO_VALIDATE_FAIL_OPEN=1 to bypass (degraded)."
)


def cmd_validate(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        if args.emit == "hook":
            return 0
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    try:
        doc = load_state(state_path)
    except json.JSONDecodeError:
        # P4#6 FLIP: a state.json that won't parse is an anomaly (validation
        # cannot run), not the legacy ImportError fast-bail. Fail CLOSED by
        # default; AMMO_VALIDATE_FAIL_OPEN=1 restores the old fail-open.
        if _fail_open_enabled():
            return 0
        if args.emit == "hook":
            print(_block_json(_FAIL_CLOSED_PARSE_REMEDIATION))
            return 0
        print(_FAIL_CLOSED_PARSE_REMEDIATION)
        return 1
    transitions = load_transitions()
    schema, _, errs = _load_schema_checked(state_path, args.schema)
    if not errs:
        errs = _schema_errors_checked(doc, schema)
    if errs:
        reason = ("state.json violates schema (%s):\n%s\nFix the values and retry the write."
                  % (_schema_path_str(state_path, args.schema), "\n".join(errs)))
        if args.emit == "hook":
            print(_block_json(reason))
            return 0
        print(reason)
        return 1

    prev = None
    prev_arg = getattr(args, "prev", None)
    if prev_arg and Path(prev_arg).is_file():
        try:
            prev = load_state(prev_arg)
        except json.JSONDecodeError:
            prev = None

    reason = gate_violation(doc, transitions, prev)
    if reason:
        if args.emit == "hook":
            print(_block_json(reason))
            return 0
        print(reason)
        return 1

    if args.emit == "hook":
        return 0
    print("PASS")
    return 0


def _schema_path_str(state_path, explicit):
    if explicit:
        return explicit
    sp = find_schema(state_path)
    return str(sp) if sp else "(schema)"


def cmd_next_step(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        return 0
    try:
        doc = load_state(state_path)
    except json.JSONDecodeError:
        return 0
    prev = None
    if args.prev and Path(args.prev).is_file():
        try:
            prev = load_state(args.prev)
        except json.JSONDecodeError:
            prev = None

    # surface REPORT.md presence for terminal reminder (artifact_dir = state's parent)
    artifact_dir = state_path.parent
    doc["_artifact_dir"] = str(artifact_dir)
    doc["_report_present"] = (artifact_dir / "REPORT.md").is_file()

    transitions = load_transitions()
    schema, _ = load_schema(state_path, getattr(args, "schema", None))

    # strip the private keys before passing into computation paths that read campaign only
    msg, is_terminal = compute_next_step(doc, prev, transitions, schema)

    if args.print_terminal:
        print("1" if is_terminal else "0")
        return 0

    if args.emit == "hook":
        if msg:
            print(json.dumps(
                {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}},
                separators=(",", ":"),
            ))
        return 0

    if msg:
        print(msg)
    return 0


def cmd_advance(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    with state_lock(state_path):
        doc = load_state(state_path)
        prev_doc = copy.deepcopy(doc)
        transitions = load_transitions()
        doc, new_stage = advance(doc, args.outcome, transitions)
        schema, _, errs = _load_schema_checked(state_path, args.schema)
        ok = not errs
        if ok:
            ok, errs = _validate_before_write(
                doc, state_path, schema, transitions, prev_doc
            )
        if not ok:
            print("FAIL: validation failed after advance (file left untouched):")
            for e in errs:
                print(e)
            return 1
        atomic_write(state_path, doc)
    print("advanced: outcome=%s new_stage=%s round=%s" % (args.outcome, new_stage, doc["campaign"]["current_round"]))
    return 0


def cmd_enrich(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    with state_lock(state_path):
        doc = load_state(state_path)
        prev_doc = copy.deepcopy(doc)
        transitions = load_transitions()

        if args.gate:
            if not args.op_id or not getattr(args, "from_", None):
                print("FAIL: --gate requires --op-id and --from", file=sys.stderr)
                return 1
            results = _load_json_quiet(args.from_) or {}
            if args.gate == "5_1a":
                metrics = enrich_gate_5_1a(doc, args.op_id, results)
            elif args.gate == "5_2":
                metrics = enrich_gate_5_2(doc, args.op_id, results)
            else:
                print("FAIL: --gate must be 5_1a or 5_2", file=sys.stderr)
                return 1
            missing = _gate_enrich_missing(metrics)
            if len(missing) == len(metrics):
                # Fail loudly. A silent all-null write looks mechanized and is not:
                # the numbers then reach state.json only by hand.
                print(
                    "FAIL: gate %s enrichment found no contract field in %s "
                    "(looked for %s at any nesting depth). Emit the names in "
                    "ammo-impl-champion.md § inline gates, then rerun."
                    % (args.gate, args.from_, ", ".join(missing)),
                    file=sys.stderr,
                )
                return 1
            if missing:
                print("WARN: gate %s enrichment could not resolve %s from %s"
                      % (args.gate, ", ".join(missing), args.from_), file=sys.stderr)
        elif args.mining:
            text = ""
            if getattr(args, "from_", None) and Path(args.from_).is_file():
                text = Path(args.from_).read_text(encoding="utf-8")
            mined_path = _mined_json_path(
                state_path.parent, doc.get("campaign", {}).get("current_round", 1) or 1,
                getattr(args, "from_", None),
            )
            mined_doc = _load_json_quiet(mined_path) if mined_path else None
            enrich_mining(doc, text, mined_doc)
        elif args.debate:
            if not args.from_dir:
                print("FAIL: --debate requires --from-dir", file=sys.stderr)
                return 1
            enrich_debate(doc, args.from_dir)
        else:
            print("FAIL: one of --gate / --mining / --debate required", file=sys.stderr)
            return 1

        schema, _, errs = _load_schema_checked(state_path, args.schema)
        ok = not errs
        if ok:
            ok, errs = _validate_before_write(
                doc, state_path, schema, transitions, prev_doc
            )
        if not ok:
            print("FAIL: validation failed after enrich (file left untouched):")
            for e in errs:
                print(e)
            return 1
        atomic_write(state_path, doc)
    return 0


def cmd_ingest_baseline(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    src_path = Path(args.from_)
    if not src_path.is_file():
        print("FAIL: sweep JSON not found at %s" % src_path, file=sys.stderr)
        return 1
    try:
        sweep = json.loads(src_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print("FAIL: sweep JSON is not parseable (%s): %s" % (src_path, exc),
              file=sys.stderr)
        return 1
    with state_lock(state_path):
        doc = load_state(state_path)
        prev_doc = copy.deepcopy(doc)
        transitions = load_transitions()
        latency, errors = ingest_baseline(
            doc, sweep, src_path, getattr(args, "round", None), args.label
        )
        if latency is None:
            print("FAIL: cannot ingest baseline from %s (file left untouched):" % src_path,
                  file=sys.stderr)
            for e in errors:
                print("  - %s" % e, file=sys.stderr)
            return 1
        for e in errors:
            print("WARN: %s" % e, file=sys.stderr)

        schema, _, errs = _load_schema_checked(state_path, args.schema)
        ok = not errs
        if ok:
            ok, errs = _validate_before_write(
                doc, state_path, schema, transitions, prev_doc
            )
        if not ok:
            print("FAIL: validation failed after ingest-baseline (file left untouched):")
            for e in errs:
                print(e)
            return 1
        atomic_write(state_path, doc)
    print("ingested baseline for %d batch size(s): %s"
          % (len(latency), ", ".join(sorted(latency, key=lambda k: float(k)))))
    return 0


def cmd_backfill(args):
    state_path = Path(args.state)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    with state_lock(state_path):
        doc = load_state(state_path)
        prev_doc = copy.deepcopy(doc)
        transitions = load_transitions()
        artifact_dir = args.artifact_dir or str(state_path.parent)
        results = backfill(doc, artifact_dir, transitions)
        schema, _, errs = _load_schema_checked(state_path, args.schema)
        ok = not errs
        if ok:
            ok, errs = _validate_before_write(
                doc, state_path, schema, transitions, prev_doc
            )
        if not ok:
            print("FAIL: validation failed after backfill (file left untouched):")
            for e in errs:
                print(e)
            return 1
        atomic_write(state_path, doc)
    if results:
        for r in results:
            print("backfilled round %s: %s" % (r["round"], ", ".join(r["filled"])))
    else:
        print("backfill: nothing to do")
    return 0


def cmd_audit_started(args):
    """Stamp audit.{stage}.started_at + cycle for a dispatched ammo-auditor.

    The auditor-spawn hook calls this; the lead never writes these two fields by
    hand. Fail-open by design: a round the campaign has not created yet, and a
    legacy round with no audit key at all, are one-line warnings with exit 0 —
    adding the audit key would switch gate enforcement on for that campaign.
    Re-running overwrites both fields, which is what a re-audit needs.
    """
    if args.round < 1:
        print("FAIL: --round must be >= 1", file=sys.stderr)
        return 1
    if args.cycle < 1:
        print("FAIL: --cycle must be >= 1", file=sys.stderr)
        return 1
    state_path = resolve_state_path(args.artifact_dir)
    if not state_path.is_file():
        print("FAIL: state.json not found at %s" % state_path, file=sys.stderr)
        return 1
    with state_lock(state_path):
        try:
            doc = load_state(state_path)
        except json.JSONDecodeError as exc:
            print(
                "FAIL: state.json at %s is not valid JSON: %s" % (state_path, exc),
                file=sys.stderr,
            )
            return 1
        rounds = (doc.get("campaign") or {}).get("rounds")
        if not isinstance(rounds, list) or args.round > len(rounds):
            print(
                "audit-started: round %d absent from campaign.rounds; nothing stamped"
                % args.round
            )
            return 0
        round_state = rounds[args.round - 1]
        if not isinstance(round_state, dict) or "audit" not in round_state:
            print(
                "audit-started: round %d carries no audit key (legacy campaign); "
                "nothing stamped" % args.round
            )
            return 0
        audit = round_state.get("audit")
        if not isinstance(audit, dict):
            audit = {}
            round_state["audit"] = audit
        gate = audit.get(args.stage)
        if not isinstance(gate, dict):
            gate = {}
            audit[args.stage] = gate
        stamp = _utc_stamp()
        gate["started_at"] = stamp
        gate["cycle"] = args.cycle
        atomic_write(state_path, doc)
    print(
        "audit-started: round %d audit.%s started_at=%s cycle=%d"
        % (args.round, args.stage, stamp, args.cycle)
    )
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="AMMO state engine")
    sub = p.add_subparsers(dest="verb", required=True)

    g = sub.add_parser("get")
    g.add_argument("target")
    g.add_argument("--field", default=None)
    g.set_defaults(func=cmd_get)

    s = sub.add_parser("set")
    s.add_argument("--state", required=True)
    s.add_argument("--field", action="append", default=[])
    s.add_argument("--value", action="append", default=[])
    s.add_argument("--schema", default=None)
    s.set_defaults(func=cmd_set)

    v = sub.add_parser("validate")
    v.add_argument("--state", required=True)
    v.add_argument("--schema", default=None)
    v.add_argument("--prev", default=None)
    v.add_argument("--emit", choices=["hook"], default=None)
    # P4#6: fail-closed is DEFAULT-ON. --fail-closed is accepted (and passed by
    # the hook) for explicitness; the live escape hatch is the env var
    # AMMO_VALIDATE_FAIL_OPEN=1, read in cmd_validate. The flag does not, by
    # itself, change behavior (default already on) — it documents intent.
    v.add_argument("--fail-closed", dest="fail_closed", action="store_true")
    v.set_defaults(func=cmd_validate)

    n = sub.add_parser("next-step")
    n.add_argument("--state", required=True)
    n.add_argument("--prev", default=None)
    n.add_argument("--schema", default=None)
    n.add_argument("--emit", choices=["hook"], default=None)
    n.add_argument("--print-terminal", action="store_true")
    n.set_defaults(func=cmd_next_step)

    a = sub.add_parser("advance")
    a.add_argument("--state", required=True)
    a.add_argument("--outcome", required=True, choices=["SHIP", "EXHAUSTED", "CONTINUE"])
    a.add_argument("--schema", default=None)
    a.set_defaults(func=cmd_advance)

    e = sub.add_parser("enrich")
    e.add_argument("--state", required=True)
    e.add_argument("--gate", choices=["5_1a", "5_2"], default=None)
    e.add_argument("--op-id", dest="op_id", default=None)
    e.add_argument("--mining", action="store_true")
    e.add_argument("--debate", action="store_true")
    e.add_argument("--from", dest="from_", default=None)
    e.add_argument("--from-dir", dest="from_dir", default=None)
    e.add_argument("--schema", default=None)
    e.set_defaults(func=cmd_enrich)

    i = sub.add_parser("ingest-baseline")
    i.add_argument("--state", required=True)
    i.add_argument("--from", dest="from_", required=True,
                   help="baseline sweep e2e_latency_results.json")
    i.add_argument("--round", type=int, default=None,
                   help="1-based round id (default: current round)")
    i.add_argument("--label", default=None,
                   help="sweep arm label (default: bench.baseline_label)")
    i.add_argument("--schema", default=None)
    i.set_defaults(func=cmd_ingest_baseline)

    b = sub.add_parser("backfill")
    b.add_argument("--state", required=True)
    b.add_argument("--artifact-dir", dest="artifact_dir", default=None)
    b.add_argument("--schema", default=None)
    b.set_defaults(func=cmd_backfill)

    au = sub.add_parser("audit-started")
    au.add_argument("--artifact-dir", dest="artifact_dir", required=True)
    au.add_argument("--stage", required=True, choices=list(AUDIT_GATE_STAGES))
    au.add_argument("--round", dest="round", type=int, required=True)
    au.add_argument("--cycle", type=int, required=True)
    au.set_defaults(func=cmd_audit_started)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
