#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for parse_session_logs.py round inference under CC 2.1.179.

CC 2.1.179 REMOVED the TeamCreate / TeamDelete tools. The session now auto-forms a
single implicit team and teammate membership derives purely from spawning with a
``name``. As a result, a real 2.1.179 transcript contains NO TeamCreate/TeamDelete
tool_use entries, so the old ``num_debate_rounds = len(team_creates)`` logic would
collapse every multi-round campaign down to a single round.

These tests build synthetic transcripts that mimic the post-2.1.179 emission (no
TeamCreate/TeamDelete) and assert that round boundaries are still recovered from the
still-emitted researcher / champion / implementer spawn sequence. The legacy path
(TeamCreate present) is also covered to prove pre-2.1.179 logs parse identically.
"""

import json
import sys
from pathlib import Path

import pytest

# Add the eval/scripts dir to path so we can import parse_session_logs by name.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from parse_session_logs import (  # noqa: E402
    SessionParser,
    _derive_debate_round_starts,
    _infer_stage_timestamps,
    parse_session,
)


# ---------------------------------------------------------------------------
# Synthetic transcript builders (CC 2.1.179: no TeamCreate / TeamDelete)
# ---------------------------------------------------------------------------

def _agent_spawn(ts, tool_use_id, name, subagent_type, *, team_name=None, session_id="sess-2179"):
    """An assistant record spawning a named teammate via the Agent tool.

    Under 2.1.179 there is no `team_name` on the Agent input; membership is by `name`.
    `team_name` is only included when explicitly testing the legacy fallback.
    """
    inp = {"name": name, "subagent_type": subagent_type, "description": f"{name} work", "model": "sonnet"}
    if team_name is not None:
        inp["team_name"] = team_name
    return {
        "type": "assistant",
        "timestamp": ts,
        "sessionId": session_id,
        "message": {"content": [
            {"type": "tool_use", "id": tool_use_id, "name": "Agent", "input": inp},
        ]},
    }


def _task_complete(ts, tool_use_id, session_id="sess-2179"):
    """A user record carrying a <task-notification> for a completed teammate."""
    notif = (
        "<task-notification>"
        f"<task-id>task-{tool_use_id}</task-id>"
        f"<tool-use-id>{tool_use_id}</tool-use-id>"
        "<status>completed</status><summary>done</summary>"
        "<usage><total_tokens>1000</total_tokens><tool_uses>3</tool_uses>"
        "<duration_ms>60000</duration_ms></usage>"
        "</task-notification>"
    )
    return {"type": "user", "timestamp": ts, "sessionId": session_id,
            "message": {"content": notif}}


def _two_round_records():
    """A 2-round campaign with NO TeamCreate/TeamDelete (true 2.1.179 emission).

    Round 1: researcher -> 2 champions (debate) -> implementer.
    Round 2: re-profile researcher -> 2 champions (debate) -> implementer.
    """
    return [
        # --- Round 1 ---
        _agent_spawn("2026-06-01T10:00:00.000Z", "tu-res-1", "researcher-1", "ammo-researcher"),
        _task_complete("2026-06-01T10:10:00.000Z", "tu-res-1"),
        _agent_spawn("2026-06-01T10:15:00.000Z", "tu-champ-1a", "champion-1", "ammo-champion"),
        _agent_spawn("2026-06-01T10:16:00.000Z", "tu-champ-1b", "champion-2", "ammo-champion"),
        _task_complete("2026-06-01T10:30:00.000Z", "tu-champ-1a"),
        _task_complete("2026-06-01T10:31:00.000Z", "tu-champ-1b"),
        _agent_spawn("2026-06-01T10:40:00.000Z", "tu-impl-1", "impl-champion-op001", "ammo-impl-champion"),
        _task_complete("2026-06-01T11:00:00.000Z", "tu-impl-1"),
        # --- Round 2 (re-profile + new debate + new impl) ---
        _agent_spawn("2026-06-01T11:05:00.000Z", "tu-res-2", "researcher-2", "ammo-researcher"),
        _task_complete("2026-06-01T11:15:00.000Z", "tu-res-2"),
        _agent_spawn("2026-06-01T11:20:00.000Z", "tu-champ-2a", "champion-3", "ammo-champion"),
        _agent_spawn("2026-06-01T11:21:00.000Z", "tu-champ-2b", "champion-4", "ammo-champion"),
        _task_complete("2026-06-01T11:35:00.000Z", "tu-champ-2a"),
        _task_complete("2026-06-01T11:36:00.000Z", "tu-champ-2b"),
        _agent_spawn("2026-06-01T11:45:00.000Z", "tu-impl-2", "impl-champion-op002", "ammo-impl-champion"),
        _task_complete("2026-06-01T12:05:00.000Z", "tu-impl-2"),
    ]


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _parse_records(records):
    """Run the records through SessionParser without touching disk shape assumptions."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        for r in records:
            tf.write(json.dumps(r) + "\n")
        tf_path = Path(tf.name)
    parser = SessionParser(tf_path)
    parser.parse()
    tf_path.unlink()
    return parser


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_team_events_under_2179():
    """A 2.1.179 transcript emits NO TeamCreate/TeamDelete -> team_events is empty."""
    parser = _parse_records(_two_round_records())
    assert parser.team_events == [], (
        "Synthetic 2.1.179 transcript must contain no TeamCreate/TeamDelete events"
    )


def test_two_rounds_inferred_without_teamcreate():
    """num_debate_rounds is 2 even though there are zero TeamCreate events."""
    parser = _parse_records(_two_round_records())
    starts = _derive_debate_round_starts(parser.events, parser.team_events)
    assert len(starts) == 2, f"expected 2 debate rounds, got {len(starts)}: {starts}"
    # The two debate rounds start at the first champion spawn of each round.
    assert starts[0] == "2026-06-01T10:15:00.000Z"
    assert starts[1] == "2026-06-01T11:20:00.000Z"


def test_stage_timestamps_has_two_rounds():
    """stage_timestamps contains round_1 AND round_2 with sane per-stage markers."""
    parser = _parse_records(_two_round_records())
    st = _infer_stage_timestamps(
        parser.events, parser.agent_spawns, parser.task_notifications, parser.team_events,
    )
    assert "round_1" in st and "round_2" in st, f"both rounds expected, got keys: {list(st)}"

    r1 = st["round_1"]
    assert r1["stage_1_baseline_start"] == "2026-06-01T10:00:00.000Z"
    assert r1["stage_3_debate_start"] == "2026-06-01T10:15:00.000Z"
    # debate end derived from champion completions (no TeamDelete marker)
    assert r1["stage_3_debate_end"] == "2026-06-01T10:31:00.000Z"
    assert r1["stage_4_5_impl_start"] == "2026-06-01T10:40:00.000Z"
    assert r1["stage_4_5_impl_end"] == "2026-06-01T11:00:00.000Z"

    r2 = st["round_2"]
    assert r2["stage_3_debate_start"] == "2026-06-01T11:20:00.000Z"
    assert r2["stage_3_debate_end"] == "2026-06-01T11:36:00.000Z"
    assert r2["stage_4_5_impl_start"] == "2026-06-01T11:45:00.000Z"
    assert r2["stage_4_5_impl_end"] == "2026-06-01T12:05:00.000Z"


def test_round2_implementers_attributed_to_round2():
    """Round-2 implementers must land in round_2, not bleed into round_1."""
    parser = _parse_records(_two_round_records())
    st = _infer_stage_timestamps(
        parser.events, parser.agent_spawns, parser.task_notifications, parser.team_events,
    )
    # op002 (round 2 impl) starts at 11:45, must be in round_2's window, not round_1's.
    assert st["round_1"]["stage_4_5_impl_end"] == "2026-06-01T11:00:00.000Z"
    assert st["round_2"]["stage_4_5_impl_start"] == "2026-06-01T11:45:00.000Z"


def test_single_round_when_one_debate():
    """A single-round 2.1.179 campaign yields exactly one round."""
    records = _two_round_records()[:8]  # round 1 only
    parser = _parse_records(records)
    starts = _derive_debate_round_starts(parser.events, parser.team_events)
    assert len(starts) == 1
    st = _infer_stage_timestamps(
        parser.events, parser.agent_spawns, parser.task_notifications, parser.team_events,
    )
    assert "round_1" in st
    assert "round_2" not in st


def test_legacy_teamcreate_path_still_counts_rounds():
    """Pre-2.1.179 logs with TeamCreate markers still drive round inference (back-compat)."""
    # Two TeamCreate markers -> two rounds, regardless of champion clustering.
    records = [
        {"type": "assistant", "timestamp": "2026-01-01T10:00:00.000Z", "sessionId": "legacy",
         "message": {"content": [
             {"type": "tool_use", "id": "tu-tc-1", "name": "TeamCreate",
              "input": {"team_name": "ammo-round-1", "description": "r1"}}]}},
        {"type": "assistant", "timestamp": "2026-01-01T10:30:00.000Z", "sessionId": "legacy",
         "message": {"content": [
             {"type": "tool_use", "id": "tu-td-1", "name": "TeamDelete", "input": {}}]}},
        {"type": "assistant", "timestamp": "2026-01-01T11:00:00.000Z", "sessionId": "legacy",
         "message": {"content": [
             {"type": "tool_use", "id": "tu-tc-2", "name": "TeamCreate",
              "input": {"team_name": "ammo-round-2", "description": "r2"}}]}},
        {"type": "assistant", "timestamp": "2026-01-01T11:30:00.000Z", "sessionId": "legacy",
         "message": {"content": [
             {"type": "tool_use", "id": "tu-td-2", "name": "TeamDelete", "input": {}}]}},
    ]
    parser = _parse_records(records)
    starts = _derive_debate_round_starts(parser.events, parser.team_events)
    assert starts == ["2026-01-01T10:00:00.000Z", "2026-01-01T11:00:00.000Z"]


def test_record_level_team_name_attribution(tmp_path):
    """team_name on agent_costs is sourced from record-level teamName when Agent input lacks it."""
    sid = "abcd1234-0000-0000-0000-000000000000"
    records = _two_round_records()
    # Stamp the session-derived implicit-team name on every record (as CC 2.1.179 does).
    for r in records:
        r["teamName"] = f"session-{sid[:8]}"
        r["sessionId"] = sid
    session_dir = tmp_path
    _write_jsonl(session_dir / f"{sid}.jsonl", records)
    data = parse_session(sid, session_dir)
    # Every champion/impl/researcher spawn should carry the implicit team name.
    team_names = {c["team_name"] for c in data["agent_costs"]}
    assert team_names == {"session-abcd1234"}, team_names


def test_none_team_name_does_not_crash(tmp_path):
    """No teamName anywhere and no Agent team_name -> team_name None, no crash."""
    sid = "ffff0000-0000-0000-0000-000000000000"
    records = _two_round_records()
    session_dir = tmp_path
    _write_jsonl(session_dir / f"{sid}.jsonl", records)
    data = parse_session(sid, session_dir)
    assert all(c["team_name"] is None for c in data["agent_costs"])
    # Multi-round inference still works end-to-end through parse_session.
    assert "round_2" in data["stage_timestamps"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
