#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Acknowledge AMMO transcript-monitor queue records for one session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[3] / "hooks"
sys.path.insert(0, str(HOOKS_DIR))
from common import blocking_monitor_records, find_repo_root, monitor_queue_paths, read_monitor_jsonl, record_session_id, session_id_from_payload, stable_monitor_record_id  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _queue_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    repo = find_repo_root(os.getcwd())
    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if args.queue:
        # An explicit queue is an exact security binding, not an additional
        # discovery hint. Do not fan out through environment/artifact queues.
        paths.extend(resolve_path(p) for p in args.queue)
    else:
        env_queue = os.environ.get("AMMO_MONITOR_QUEUE")
        if env_queue:
            paths.append(resolve_path(env_queue))
        artifact_dir = args.artifact_dir or os.environ.get("AMMO_ARTIFACT_DIR")
        if artifact_dir:
            root = resolve_path(artifact_dir)
            paths.append(root / "monitor_interventions.jsonl")
            paths.extend(sorted(root.glob("**/monitor_interventions.jsonl")))
        if not paths:
            paths.extend(monitor_queue_paths(repo, {}))
    seen: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _is_open_for_session(record: dict, session_id: str) -> bool:
    if str(record.get("target_session_id") or record.get("session_id") or record.get("targetSessionId") or "") != session_id:
        return False
    status = str(record.get("status") or "").lower()
    if status in {"acknowledged", "resolved", "dismissed", "closed"}:
        return False
    return not (record.get("acknowledged_at") or record.get("resolved_at"))


def _require_current_binding(args: argparse.Namespace, repo: Path, paths: list[Path]) -> None:
    if len(args.queue or []) != 1 or len(paths) != 1:
        raise ValueError("pass exactly one --queue for an exact current binding")
    if not Path(args.queue[0]).expanduser().is_absolute():
        raise ValueError("pass an absolute --queue for an exact current binding")
    selected = {path.resolve() for path in paths}
    matching = [
        record
        for record in blocking_monitor_records(repo, {})
        if record_session_id(record) == args.session_id
        and Path(str(record.get("_queue_path") or "")).resolve() in selected
    ]
    if not matching:
        raise ValueError("no current blocking record matches the session and queue")
    if not args.all:
        pending_ids = {str(record.get("_record_id") or "") for record in matching}
        if not args.record_id or any(record_id not in pending_ids for record_id in args.record_id):
            raise ValueError("one or more record ids do not match a current blocking record")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", help="Codex root target_session_id to acknowledge; defaults from CODEX_SESSION_ID")
    parser.add_argument("--queue", action="append", help="queue path; may be repeated")
    parser.add_argument("--artifact-dir", help="artifact directory containing monitor queue files")
    parser.add_argument("--record-id", action="append", help="specific monitor record id to acknowledge; may be repeated")
    parser.add_argument("--all", action="store_true", help="acknowledge all open records for the session")
    parser.add_argument("--status", choices=["acknowledged", "resolved"], default="acknowledged")
    parser.add_argument("--note", required=True, help="what changed, or evidence-based rebuttal")
    args = parser.parse_args()
    if not args.session_id:
        args.session_id = session_id_from_payload({}, None)
    if not args.session_id:
        parser.error("pass --session-id (normally state.json.codex_thread_id) or set CODEX_SESSION_ID")
    if not args.all and not args.record_id:
        parser.error("pass --record-id for each intervention to acknowledge, or --all")

    repo = find_repo_root(os.getcwd())
    paths = _queue_paths(args)
    try:
        _require_current_binding(args, repo, paths)
    except ValueError as exc:
        parser.error(str(exc))

    changed = 0
    scanned = 0
    for path in paths:
        entries = read_monitor_jsonl(path)
        if not entries:
            continue
        scanned += 1
        touched = False
        wanted_ids = set(args.record_id or [])
        for entry in entries:
            record = entry.get("record")
            if not isinstance(record, dict):
                continue
            idx = entry.get("line_no") or 0
            rid = stable_monitor_record_id(path, idx, record)
            if _is_open_for_session(record, args.session_id) and (args.all or rid in wanted_ids):
                record["status"] = args.status
                record[f"{args.status}_at"] = _now()
                record["ack_note"] = args.note
                record.setdefault("record_id", rid)
                changed += 1
                touched = True
        if touched:
            lines = []
            for entry in entries:
                record = entry.get("record")
                if isinstance(record, dict):
                    lines.append(json.dumps(record, sort_keys=True))
                else:
                    lines.append(str(entry.get("raw", "")))
            path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

    print(json.dumps({"changed": changed, "queues_scanned": scanned, "session_id": args.session_id}, sort_keys=True))
    return 0 if changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
