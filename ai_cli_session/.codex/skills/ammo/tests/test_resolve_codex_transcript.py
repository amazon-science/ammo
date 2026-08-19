# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from resolve_codex_transcript import resolve_transcript


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "state.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            agent_path TEXT,
            created_at_ms INTEGER
        );
        CREATE TABLE thread_spawn_edges (
            parent_thread_id TEXT NOT NULL,
            child_thread_id TEXT PRIMARY KEY,
            status TEXT NOT NULL
        );
        """
    )
    connection.close()
    return path


def _insert(db: Path, *, parent: str, child: str, agent_path: str, transcript: Path, created: int):
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?)",
        (child, str(transcript), agent_path, created),
    )
    connection.execute(
        "INSERT INTO thread_spawn_edges VALUES (?, ?, 'open')",
        (parent, child),
    )
    connection.commit()
    connection.close()


def test_resolves_newest_exact_child_and_bounded_replacement_paths(tmp_path: Path):
    db = _db(tmp_path)
    parent = "campaign-uuid"
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    other = tmp_path / "other.jsonl"
    _insert(db, parent=parent, child="old-id", agent_path="/root/impl_op007", transcript=old, created=1)
    _insert(db, parent=parent, child="new-id", agent_path="/root/impl_op007", transcript=new, created=2)
    _insert(db, parent=parent, child="other-id", agent_path="/root/impl_op008", transcript=other, created=3)

    result = resolve_transcript(db, parent, "/root/impl_op007")

    assert result["target_rollout_id"] == "new-id"
    assert result["target_transcript_path"] == str(new.resolve())
    assert result["collision_candidate_paths"] == [str(new.resolve()), str(old.resolve())]
    assert str(other.resolve()) not in result["collision_candidate_paths"]


def test_fails_closed_when_exact_child_has_no_readable_transcript(tmp_path: Path):
    db = _db(tmp_path)
    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?)",
        ("child", str(tmp_path / "missing.jsonl"), "/root/impl_op007", 1),
    )
    connection.execute(
        "INSERT INTO thread_spawn_edges VALUES (?, ?, 'open')",
        ("campaign", "child"),
    )
    connection.commit()
    connection.close()

    with pytest.raises(LookupError, match="No readable child transcript"):
        resolve_transcript(db, "campaign", "/root/impl_op007")


def test_rejects_noncanonical_agent_path(tmp_path: Path):
    with pytest.raises(ValueError, match="canonical absolute"):
        resolve_transcript(_db(tmp_path), "campaign", "impl_op007")
