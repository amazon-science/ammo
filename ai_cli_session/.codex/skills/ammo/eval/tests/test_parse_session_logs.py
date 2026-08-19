# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for Codex session log parsing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "parse_session_logs.py"
)


def _load_parser_module():
    spec = importlib.util.spec_from_file_location("parse_session_logs", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assistant(ts: str, tool_id: str, tool_name: str, tool_input: dict) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            ]
        },
    }


def _completion(ts: str, tool_id: str, task_id: str) -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "message": {
            "content": (
                "<task-notification>"
                f"<task-id>{task_id}</task-id>"
                f"<tool-use-id>{tool_id}</tool-use-id>"
                "<status>completed</status>"
                "<usage><total_tokens>10</total_tokens>"
                "<tool_uses>1</tool_uses>"
                "<duration_ms>1000</duration_ms></usage>"
                "</task-notification>"
            )
        },
    }


def test_codex_parser_preserves_round_team_name_contract(tmp_path):
    parser = _load_parser_module()
    session_id = "session-001"
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    session_path = session_dir / f"{session_id}.jsonl"

    rows = [
        _assistant(
            "2026-03-19T10:00:00.000Z",
            "tu-researcher",
            "functions.spawn_agent",
            {
                "target": "researcher-1",
                "agent_type": "ammo-researcher",
                "message": "Baseline capture and bottleneck mining",
            },
        ),
        _completion("2026-03-19T10:05:00.000Z", "tu-researcher", "task-researcher"),
        _assistant(
            "2026-03-19T10:10:00.000Z",
            "tu-cohort",
            "Codex subagent cohort spawn",
            {"cohort_name": "ammo-round-1-test", "description": "Round 1 debate"},
        ),
        _assistant(
            "2026-03-19T10:11:00.000Z",
            "tu-champion",
            "spawn_agent",
            {
                "name": "champion-1",
                "agent_type": "ammo-champion",
                "description": "Champion proposal",
                "team_name": "ammo-round-1-test",
            },
        ),
        _completion("2026-03-19T10:15:00.000Z", "tu-champion", "task-champion"),
        _assistant(
            "2026-03-19T10:20:00.000Z",
            "tu-close",
            "functions.close_agent",
            {},
        ),
    ]
    session_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    data = parser.parse_session(session_id, session_dir)

    assert data["stage_timestamps"]["round_1"]["stage_3_debate_start"] == (
        "2026-03-19T10:10:00.000Z"
    )
    assert data["stage_timestamps"]["round_1"]["stage_3_debate_end"] == (
        "2026-03-19T10:20:00.000Z"
    )
    assert data["team_lifecycle"] == [
        {
            "name": "ammo-round-1-test",
            "create_timestamp": "2026-03-19T10:10:00.000Z",
            "description": "Round 1 debate",
            "delete_timestamp": "2026-03-19T10:20:00.000Z",
            "duration_seconds": 600.0,
        }
    ]

    champion_cost = next(row for row in data["agent_costs"] if row["name"] == "champion-1")
    assert champion_cost["team_name"] == "ammo-round-1-test"
    assert champion_cost["cohort_name"] == "ammo-round-1-test"
