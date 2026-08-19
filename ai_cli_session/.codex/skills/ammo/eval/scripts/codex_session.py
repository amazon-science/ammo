#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Read-only discovery helpers for current Codex CLI session rollouts.

Codex 0.144.x stores the authoritative thread graph in ``state_5.sqlite`` and
one rollout JSONL per thread under ``$CODEX_HOME/sessions``.  Subagent rollouts
may begin with a replay of their parent's history.  ``iter_owned_records``
skips that fork prelude so callers do not double-count parent events.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


CURRENT_SOURCE_FORMAT = "codex_sqlite_rollouts_v1"
MAX_THREAD_DEPTH = 128


class CodexThreadNotFound(FileNotFoundError):
    """The exact requested thread ID is absent from the current state DB."""


class CodexRolloutMissing(FileNotFoundError):
    """A resolved current thread points to a missing rollout file."""


def default_codex_home() -> Path:
    """Return ``$CODEX_HOME`` or the standard ``~/.codex`` location."""
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open SQLite without creating or mutating the database."""
    db_path = db_path.expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Codex state database not found: {db_path}")
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _optional_select(columns: set[str], name: str) -> str:
    if name in columns:
        return f't."{name}" AS "{name}"'
    return f'NULL AS "{name}"'


def resolve_thread_tree(
    session_id: str,
    *,
    codex_home: Optional[Path] = None,
    state_db: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve an exact Codex thread ID and all descendants, read-only.

    ``session_id`` is matched only against ``threads.id``.  No title, cwd, or
    rollout-path guessing is used.  The spawn graph is traversed recursively
    through ``thread_spawn_edges`` and cycle-guarded to a bounded depth.
    """
    home = (codex_home or default_codex_home()).expanduser().resolve()
    db_path = (state_db or (home / "state_5.sqlite")).expanduser().resolve()

    with closing(_open_readonly(db_path)) as conn:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required_tables = {"threads", "thread_spawn_edges"}
        missing = sorted(required_tables - tables)
        if missing:
            raise RuntimeError(
                f"Codex state database is missing required tables: {', '.join(missing)}"
            )

        columns = _table_columns(conn, "threads")
        required_columns = {"id", "rollout_path"}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise RuntimeError(
                "Codex threads table is missing required columns: "
                + ", ".join(missing_columns)
            )

        optional = [
            "tokens_used",
            "agent_path",
            "agent_role",
            "agent_nickname",
            "created_at",
            "updated_at",
            "created_at_ms",
            "updated_at_ms",
            "cli_version",
            "model",
            "thread_source",
        ]
        select_fields = ",\n                   ".join(
            _optional_select(columns, name) for name in optional
        )
        if "created_at_ms" in columns and "created_at" in columns:
            order_time = "COALESCE(t.created_at_ms, t.created_at * 1000, 0)"
        elif "created_at_ms" in columns:
            order_time = "COALESCE(t.created_at_ms, 0)"
        elif "created_at" in columns:
            order_time = "COALESCE(t.created_at * 1000, 0)"
        else:
            order_time = "0"
        sql = f"""
            WITH RECURSIVE tree(id, parent_thread_id, depth, visited) AS (
                SELECT t.id, NULL, 0, ',' || t.id || ','
                  FROM threads AS t
                 WHERE t.id = ?
                UNION ALL
                SELECT e.child_thread_id,
                       e.parent_thread_id,
                       tree.depth + 1,
                       tree.visited || e.child_thread_id || ','
                  FROM tree
                  JOIN thread_spawn_edges AS e
                    ON e.parent_thread_id = tree.id
                 WHERE tree.depth < ?
                   AND instr(tree.visited, ',' || e.child_thread_id || ',') = 0
            )
            SELECT tree.parent_thread_id,
                   tree.depth,
                   edge.status AS edge_status,
                   t.id,
                   t.rollout_path,
                   {select_fields}
              FROM tree
              JOIN threads AS t ON t.id = tree.id
             LEFT JOIN thread_spawn_edges AS edge
                ON edge.child_thread_id = tree.id
             ORDER BY tree.depth,
                      {order_time},
                      t.id
        """
        rows = [dict(row) for row in conn.execute(sql, (session_id, MAX_THREAD_DEPTH))]

    if not rows:
        raise CodexThreadNotFound(
            f"Exact Codex thread ID not found in {db_path}: {session_id}"
        )

    for row in rows:
        raw_path = Path(str(row["rollout_path"])).expanduser()
        if not raw_path.is_absolute():
            raw_path = home / raw_path
        rollout_path = raw_path.resolve()
        row["rollout_path"] = str(rollout_path)
        row["rollout_exists"] = rollout_path.is_file()
        created_at_ms = row.get("created_at_ms")
        if created_at_ms is None and row.get("created_at") is not None:
            created_at_ms = int(row["created_at"]) * 1000
        updated_at_ms = row.get("updated_at_ms")
        if updated_at_ms is None and row.get("updated_at") is not None:
            updated_at_ms = int(row["updated_at"]) * 1000
        row["created_at_ms"] = created_at_ms
        row["updated_at_ms"] = updated_at_ms

    return {
        "source_format": CURRENT_SOURCE_FORMAT,
        "session_id": session_id,
        "codex_home": str(home),
        "state_db": str(db_path),
        "threads": rows,
    }


def _iter_jsonl(path: Path, *, start_line: int = 1) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj.setdefault("_line_number", line_number)
                yield obj


def _owned_start_line(thread: Dict[str, Any]) -> int:
    """Find the first real turn in a subagent rollout.

    A forked rollout can contain a replay whose wrapper timestamps are rewritten
    to the child creation time.  The embedded ``task_started.started_at`` keeps
    the original epoch, so the first start at or after this thread's creation is
    the stable ownership boundary.
    """
    # A caller may intentionally evaluate a subagent thread as the root of a
    # smaller tree, so traversal depth alone does not identify a user thread.
    if int(thread.get("depth") or 0) == 0 and not thread.get("agent_path"):
        return 1
    path = Path(str(thread["rollout_path"]))
    if not path.is_file():
        return 1
    created_at_ms = thread.get("created_at_ms")
    last_task_start_line: Optional[int] = None
    fallback_candidate: Optional[int] = None
    for obj in _iter_jsonl(path):
        # Current subagent delivery writes this marker immediately after the
        # child's own task_started/turn_context. Fork-history replay does not
        # copy these delivery records, making it the strongest boundary signal.
        if obj.get("type") == "inter_agent_communication_metadata":
            if last_task_start_line is not None:
                return last_task_start_line
            continue
        if obj.get("type") != "event_msg":
            continue
        payload = obj.get("payload") or {}
        if payload.get("type") != "task_started":
            continue
        last_task_start_line = int(obj.get("_line_number") or 1)
        if created_at_ms is None:
            continue
        started_at = payload.get("started_at")
        if not isinstance(started_at, (int, float)):
            continue
        started_at_ms = int(started_at)
        if started_at_ms < 10_000_000_000:
            started_at_ms *= 1000
        # Fallback for rollouts without an inter-agent marker. Integer epoch
        # seconds can precede created_at_ms by up to 999 ms after truncation.
        if started_at_ms >= int(created_at_ms) - 1000:
            candidate = int(obj.get("_line_number") or 1)
            # Keep scanning for the delivery marker before accepting fallback.
            last_task_start_line = candidate
            if fallback_candidate is None:
                fallback_candidate = candidate
    if fallback_candidate is not None:
        return fallback_candidate
    return 1


def iter_owned_records(thread: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield records authored by ``thread``, excluding any fork-history replay."""
    path = Path(str(thread["rollout_path"]))
    if not path.is_file():
        return
    yield from _iter_jsonl(path, start_line=_owned_start_line(thread))


def epoch_ms_to_iso(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def parse_arguments(value: Any) -> Dict[str, Any]:
    """Decode a current function-call argument object without executing it."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def agent_name(thread: Dict[str, Any]) -> Optional[str]:
    path = thread.get("agent_path")
    if isinstance(path, str) and path:
        return path.rstrip("/").rsplit("/", 1)[-1]
    nickname = thread.get("agent_nickname")
    return str(nickname) if nickname else None
