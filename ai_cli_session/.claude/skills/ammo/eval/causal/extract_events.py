#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Extract fine-grained events from a Claude Code session JSONL file.

Reads stage boundaries from --session-data (produced by parse_session_logs.py)
and emits a structured event log with file_write, file_read, agent_spawn,
send_message, agent_complete, and data_output events.

Usage:
    python extract_events.py \\
        --session-jsonl /path/to/session.jsonl \\
        --session-data /path/to/session_data.json \\
        --output /path/to/events.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime, returning None on failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        ts_clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_clean)
    except (ValueError, TypeError):
        return None


def _infer_role(
    name: Optional[str],
    subagent_type: Optional[str],
    description: Optional[str],
) -> str:
    """Infer agent role from name, subagent_type, and description.

    Mirrors parse_session_logs.py._infer_role exactly.
    """
    # Prefer explicit subagent_type
    if subagent_type:
        st = subagent_type.lower()
        if "researcher" in st:
            return "researcher"
        # ``ammo-impl-champion`` contains both "impl" and "champion"; the implementer
        # check MUST precede the champion check so the implementation agent is not
        # mis-labelled as a debate champion. Mirrors parse_session_logs.py._infer_role.
        if "impl" in st or "implementer" in st:
            return "implementer"
        if "champion" in st:
            return "champion"
        if "delegate" in st:
            return "delegate"

    # Fall back to name pattern
    if name:
        n = name.lower()
        if re.match(r"champion-?\d+", n):
            return "champion"
        if re.match(r"delegate-?\d+\w*", n):
            return "delegate"
        if "implementer" in n:
            return "implementer"
        if "researcher" in n:
            return "researcher"

    # Fall back to description keywords
    if description:
        d = description.lower()
        if "researcher" in d or "baseline" in d or "bottleneck" in d:
            return "researcher"
        if "implementer" in d or "implementation" in d:
            return "implementer"
        if "champion" in d:
            return "champion"
        if "delegate" in d:
            return "delegate"

    return "unknown"


def _extract_task_notification(content_str: str) -> Optional[Dict[str, Any]]:
    """Extract task-notification data from a content string."""
    m = re.search(r"<task-notification>(.*?)</task-notification>", content_str, re.DOTALL)
    if not m:
        return None
    xml = m.group(0)

    def _field(tag: str) -> Optional[str]:
        fm = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
        return fm.group(1).strip() if fm else None

    task_id = _field("task-id")
    tool_use_id = _field("tool-use-id")
    status = _field("status")
    summary = _field("summary")

    total_tokens = None
    tool_uses_count = None
    duration_ms = None

    usage_m = re.search(r"<usage>(.*?)</usage>", xml, re.DOTALL)
    if usage_m:
        usage_text = usage_m.group(0)
        tt = re.search(r"<total_tokens>(\d+)</total_tokens>", usage_text)
        tu = re.search(r"<tool_uses>(\d+)</tool_uses>", usage_text)
        dm = re.search(r"<duration_ms>(\d+)</duration_ms>", usage_text)
        if tt:
            total_tokens = int(tt.group(1))
        if tu:
            tool_uses_count = int(tu.group(1))
        if dm:
            duration_ms = int(dm.group(1))

    return {
        "task_id": task_id,
        "tool_use_id": tool_use_id,
        "status": status,
        "summary": summary,
        "total_tokens": total_tokens,
        "tool_uses": tool_uses_count,
        "duration_ms": duration_ms,
    }


def _get_content_text(content: Any) -> str:
    """Extract plain text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Data-output pattern scanning
# ---------------------------------------------------------------------------

# Each entry: (datum_type, compiled_regex, group_index_for_value)
_DATA_PATTERNS: List[tuple] = [
    (
        "f_value",
        re.compile(r"(\d{1,3}\.\d{1,2})\s*%\s*(?:of\s+(?:decode|total|latency))", re.IGNORECASE),
        1,
    ),
    (
        "kernel_timing",
        re.compile(r"(\d+\.?\d*)\s*(?:µs|us|microsec)", re.IGNORECASE),
        1,
    ),
    (
        "speedup",
        re.compile(r"(\d+\.\d+)\s*x\b|\+(\d+\.?\d*)\s*%", re.IGNORECASE),
        None,  # special: two groups
    ),
    (
        "memory_size",
        re.compile(r"(\d+\.?\d*)\s*(?:MB|GB|KiB|MiB|GiB)\b", re.IGNORECASE),
        1,
    ),
    (
        "bandwidth",
        re.compile(r"(\d+\.?\d*)\s*(?:GB/s|TB/s)\b", re.IGNORECASE),
        1,
    ),
]


def _scan_text_for_data(text: str) -> List[Dict[str, Any]]:
    """Find all numeric data patterns in text and return data_output detail dicts."""
    results = []
    for datum_type, pattern, group_idx in _DATA_PATTERNS:
        for m in pattern.finditer(text):
            if datum_type == "speedup":
                # Two capture groups: group 1 = Nx, group 2 = +X%
                raw = m.group(0)
                if m.group(1) is not None:
                    value = float(m.group(1))
                else:
                    value = float(m.group(2))
            else:
                raw = m.group(0)
                value = float(m.group(group_idx))

            results.append({
                "datum_type": datum_type,
                "value": value,
                "raw_text": raw,
            })
    return results


# ---------------------------------------------------------------------------
# Stage-range builder
# ---------------------------------------------------------------------------

def _build_stage_ranges(stage_timestamps: Dict[str, Any]) -> List[tuple]:
    """Build a sorted list of (start_dt, end_dt, stage_label) from session_data.

    Returns list sorted by start_dt so we can binary-search.
    """
    ranges = []

    # Map stage key fragments → human stage name
    _STAGE_KEY_MAP = [
        ("stage_1_baseline", "baseline"),
        ("stage_2_bottleneck", "bottleneck"),
        ("stage_3_debate", "debate"),
        ("stage_4_5_impl", "implementation"),
        ("stage_6_7_eval", "evaluation"),
    ]

    for round_key, round_data in stage_timestamps.items():
        if not isinstance(round_data, dict):
            continue
        round_label = round_key  # e.g. "round_1"

        # Collect all timestamps for this round, sort, then assign stages
        # Strategy: find pairs of start/end keys per stage
        # Collect all keys for this round
        keys = list(round_data.keys())

        # Build stage start→end pairs
        # Each stage_X_foo_start has a corresponding stage_X_foo_end (or next stage start)
        start_keys = sorted(k for k in keys if not k.endswith("_end"))
        end_keys = sorted(k for k in keys if k.endswith("_end"))

        # Match starts to ends by prefix
        for stage_frag, stage_name in _STAGE_KEY_MAP:
            # Find matching start key
            start_key = next(
                (k for k in keys if stage_frag in k and not k.endswith("_end")),
                None,
            )
            end_key = next(
                (k for k in keys if stage_frag in k and k.endswith("_end")),
                None,
            )

            if start_key is None:
                # Some stages have only a single timestamp (start becomes marker)
                continue

            start_ts = round_data.get(start_key)
            end_ts = round_data.get(end_key) if end_key else None

            start_dt = _parse_iso(start_ts)
            if start_dt is None:
                continue

            # If no end, use a far-future sentinel
            end_dt = _parse_iso(end_ts) if end_ts else None

            label = f"{round_label}:{stage_name}"
            ranges.append((start_dt, end_dt, label))

    # Sort by start_dt
    ranges.sort(key=lambda r: r[0])
    return ranges


def _assign_stage(ts: Optional[str], stage_ranges: List[tuple]) -> Optional[str]:
    """Assign a stage label to a timestamp using stage_ranges."""
    if not ts or not stage_ranges:
        return None

    event_dt = _parse_iso(ts)
    if event_dt is None:
        return None

    # Linear scan — ranges are small (< 20)
    best: Optional[str] = None
    for start_dt, end_dt, label in stage_ranges:
        if event_dt >= start_dt:
            if end_dt is None or event_dt <= end_dt:
                best = label
    return best


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_events(
    session_jsonl_path: Path,
    session_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Parse a Claude Code session JSONL file and produce a structured event log.

    Args:
        session_jsonl_path: Path to the session .jsonl file.
        session_data: Parsed output from parse_session_logs.py (dict with
            stage_timestamps, agent_costs, etc.).

    Returns:
        List of event dicts, each with keys:
            timestamp, event_type, agent_id, agent_name, agent_role, stage, details

    Raises:
        FileNotFoundError: If session_jsonl_path does not exist.
    """
    session_jsonl_path = Path(session_jsonl_path)
    if not session_jsonl_path.exists():
        raise FileNotFoundError(f"Session JSONL not found: {session_jsonl_path}")

    # Build stage ranges from session_data
    stage_timestamps = session_data.get("stage_timestamps", {})
    stage_ranges = _build_stage_ranges(stage_timestamps)

    events: List[Dict[str, Any]] = []

    def _make_event(
        event_type: str,
        timestamp: Optional[str],
        agent_id: Optional[str],
        agent_name: Optional[str],
        agent_role: Optional[str],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "stage": _assign_stage(timestamp, stage_ranges),
            "details": details,
        }

    with open(session_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip truncated / malformed lines

            msg_type = obj.get("type")
            ts = obj.get("timestamp")
            # agentId is set when the message comes FROM a subagent
            raw_agent_id = obj.get("agentId")
            # CC 2.1.179: the Agent tool's `team_name` param is deprecated & ignored
            # (single implicit team; membership by `name`), so use the still-emitted
            # record-level session-derived `teamName` for attribution, falling back to
            # the legacy Agent-input team_name when present in pre-2.1.179 logs.
            record_team_name = obj.get("teamName")

            if msg_type == "assistant":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type")

                    if block_type == "tool_use":
                        tool_name = block.get("name", "")
                        block_id = block.get("id")  # the tool_use UUID
                        inp = block.get("input") or {}

                        if tool_name == "Write":
                            file_path = inp.get("file_path") or inp.get("path", "")
                            events.append(_make_event(
                                event_type="file_write",
                                timestamp=ts,
                                agent_id=raw_agent_id,
                                agent_name=None,
                                agent_role=None,
                                details={"path": file_path},
                            ))

                        elif tool_name == "Read":
                            file_path = inp.get("file_path") or inp.get("path", "")
                            events.append(_make_event(
                                event_type="file_read",
                                timestamp=ts,
                                agent_id=raw_agent_id,
                                agent_name=None,
                                agent_role=None,
                                details={"path": file_path},
                            ))

                        elif tool_name == "Agent":
                            agent_name = inp.get("name")
                            subagent_type = inp.get("subagent_type")
                            description = inp.get("description")
                            role = _infer_role(agent_name, subagent_type, description)

                            events.append(_make_event(
                                event_type="agent_spawn",
                                timestamp=ts,
                                agent_id=block_id,  # the tool_use id IS the agent's UUID reference
                                agent_name=agent_name,
                                agent_role=role,
                                details={
                                    "description": description,
                                    "subagent_type": subagent_type,
                                    "model": inp.get("model"),
                                    "team_name": inp.get("team_name") or record_team_name,
                                },
                            ))

                        elif tool_name == "SendMessage":
                            from_agent = raw_agent_id
                            to_agent = inp.get("to")
                            events.append(_make_event(
                                event_type="send_message",
                                timestamp=ts,
                                agent_id=raw_agent_id,
                                agent_name=None,
                                agent_role=None,
                                details={
                                    "from_agent": from_agent,
                                    "to_agent": to_agent,
                                    "message": inp.get("message"),
                                    "summary": inp.get("summary"),
                                },
                            ))

                    elif block_type == "text":
                        text = block.get("text", "")
                        if not text:
                            continue
                        # Scan for data patterns
                        data_hits = _scan_text_for_data(text)
                        for hit in data_hits:
                            events.append(_make_event(
                                event_type="data_output",
                                timestamp=ts,
                                agent_id=raw_agent_id,
                                agent_name=None,
                                agent_role=None,
                                details=hit,
                            ))

            elif msg_type == "user":
                msg = obj.get("message", {})
                content = msg.get("content", "")
                content_text = _get_content_text(content)

                if "<task-notification>" in content_text:
                    notif = _extract_task_notification(content_text)
                    if notif:
                        events.append(_make_event(
                            event_type="agent_complete",
                            timestamp=ts,
                            agent_id=notif.get("tool_use_id"),
                            agent_name=None,
                            agent_role=None,
                            details={
                                "task_id": notif.get("task_id"),
                                "tool_use_id": notif.get("tool_use_id"),
                                "status": notif.get("status"),
                                "summary": notif.get("summary"),
                                "total_tokens": notif.get("total_tokens"),
                                "tool_uses": notif.get("tool_uses"),
                                "duration_ms": notif.get("duration_ms"),
                            },
                        ))

    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session-jsonl", required=True, help="Path to session JSONL file")
    p.add_argument(
        "--session-data",
        required=True,
        help="Path to session_data.json (from parse_session_logs.py)",
    )
    p.add_argument("--output", required=True, help="Output path for events JSON")

    args = p.parse_args()

    session_jsonl_path = Path(args.session_jsonl)
    session_data_path = Path(args.session_data)

    if not session_data_path.exists():
        print(f"ERROR: session-data file not found: {session_data_path}", file=sys.stderr)
        return 1

    try:
        session_data = json.loads(session_data_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading session-data: {e}", file=sys.stderr)
        return 1

    try:
        events = extract_events(session_jsonl_path, session_data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.write_text(json.dumps(events, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {len(events)} events to: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
