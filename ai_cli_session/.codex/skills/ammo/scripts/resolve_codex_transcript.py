#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Resolve an exact Codex child rollout and transcript from the local index.

This intentionally queries only one parent thread plus one canonical agent path.
It never scans session JSONL contents or guesses from timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def resolve_transcript(
    state_db: Path,
    campaign_session_id: str,
    agent_path: str,
    *,
    max_candidates: int = 8,
) -> dict[str, Any]:
    if not campaign_session_id.strip():
        raise ValueError("campaign_session_id must be non-empty")
    if not agent_path.startswith("/"):
        raise ValueError("agent_path must be the canonical absolute task path")
    if max_candidates < 1 or max_candidates > 32:
        raise ValueError("max_candidates must be between 1 and 32")
    db = state_db.expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(f"Codex state database not found: {db}")

    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT t.id, t.rollout_path, e.status, t.created_at_ms
            FROM thread_spawn_edges AS e
            JOIN threads AS t ON t.id = e.child_thread_id
            WHERE e.parent_thread_id = ? AND t.agent_path = ?
            ORDER BY t.created_at_ms DESC, t.id DESC
            LIMIT ?
            """,
            (campaign_session_id, agent_path, max_candidates),
        ).fetchall()
    finally:
        connection.close()

    candidates: list[dict[str, Any]] = []
    for rollout_id, rollout_path, status, created_at_ms in rows:
        path = Path(str(rollout_path)).expanduser().resolve()
        if not path.is_file() or path.suffix != ".jsonl":
            continue
        candidates.append(
            {
                "rollout_id": str(rollout_id),
                "transcript_path": str(path),
                "status": str(status),
                "created_at_ms": int(created_at_ms or 0),
            }
        )

    if not candidates:
        raise LookupError(
            f"No readable child transcript for parent={campaign_session_id!r}, "
            f"agent_path={agent_path!r}"
        )
    target = candidates[0]
    return {
        "campaign_session_id": campaign_session_id,
        "agent_path": agent_path,
        "target_rollout_id": target["rollout_id"],
        "target_transcript_path": target["transcript_path"],
        "collision_candidate_paths": [item["transcript_path"] for item in candidates],
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    parser.add_argument("--state-db", type=Path, default=codex_home / "state_5.sqlite")
    parser.add_argument("--campaign-session-id", required=True)
    parser.add_argument("--agent-path", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    try:
        result = resolve_transcript(
            args.state_db,
            args.campaign_session_id,
            args.agent_path,
            max_candidates=args.max_candidates,
        )
    except (FileNotFoundError, LookupError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
