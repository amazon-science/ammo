#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Extract fine-grained events from current or legacy CLI session rollouts.

Reads stage boundaries from --session-data (produced by parse_session_logs.py)
and emits a structured event log with file_write, file_read, agent_spawn,
send_message, agent_complete, and data_output events.

Usage:
    python extract_events.py \\
        --session-data /path/to/session_data.json \\
        --output /path/to/events.json

Current Codex rollout paths come from ``session_data.rollout_files``.  Use
``--session-jsonl`` only for the legacy single-JSONL format.
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


_EVAL_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from codex_session import (  # noqa: E402
    CURRENT_SOURCE_FORMAT,
    iter_owned_records,
    parse_arguments,
)


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
        # ``ammo-impl-champion`` contains both "impl" and "champion"; the
        # implementer check must precede champion so implementation work is not
        # mis-labelled as debate.
        if "impl" in st or "implementer" in st:
            return "implementer"
        if "champion" in st:
            return "champion"
        if "delegate" in st:
            return "delegate"

    # Fall back to name pattern
    if name:
        n = name.lower()
        normalized = n.replace("_", "-")
        if "implementer" in n or ("impl" in n and "champion" in n):
            return "implementer"
        if n.startswith("impl-") or n.startswith("impl_"):
            return "implementer"
        if "researcher" in n:
            return "researcher"
        if re.match(r"champion-?\d+", normalized) or "champion" in n:
            return "champion"
        if re.match(r"delegate-?\d+\w*", normalized) or "delegate" in n:
            return "delegate"

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
    """Parse a Codex CLI session JSONL file and produce a structured event log.

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

                        elif tool_name in {"Agent", "spawn_agent", "functions.spawn_agent"}:
                            agent_name = inp.get("task_name") or inp.get("name") or inp.get("target")
                            subagent_type = inp.get("subagent_type") or inp.get("agent_type")
                            description = inp.get("description") or inp.get("message")
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
                                },
                            ))

                        elif tool_name in {
                            "subagent_prompt", "send_input", "functions.send_input", "CodexSubagentPrompt",
                            "send_message", "functions.send_message", "followup_task", "functions.followup_task",
                        }:
                            from_agent = raw_agent_id
                            to_agent = inp.get("to") or inp.get("target")
                            events.append(_make_event(
                                event_type="send_message",
                                timestamp=ts,
                                agent_id=raw_agent_id,
                                agent_name=None,
                                agent_role=None,
                                details={
                                    "from_agent": from_agent,
                                    "to_agent": to_agent,
                                    "message": inp.get("message") or inp.get("prompt"),
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
# Current Codex response_item/event_msg extraction
# ---------------------------------------------------------------------------

def _bounded(value: Any, limit: int = 2000) -> Any:
    """Keep causal output useful without copying multi-megabyte tool payloads."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"...<{len(value) - limit} chars omitted>"
    if isinstance(value, list):
        return [_bounded(item, limit) for item in value[:50]]
    if isinstance(value, dict):
        return {str(key): _bounded(item, limit) for key, item in list(value.items())[:50]}
    return value


def _current_visible_text(payload: Dict[str, Any]) -> str:
    """Extract only plaintext message blocks; encrypted blocks stay opaque."""
    direct = payload.get("message")
    if isinstance(direct, str):
        return direct
    content = payload.get("content")
    if isinstance(content, str):
        return content
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _tool_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_tool_output_text(item) for item in value)))
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            text = value.get("text")
            return text if isinstance(text, str) else ""
        # Do not expose encrypted payloads as pseudo-text.
        if block_type == "encrypted_content":
            return ""
    return ""


def _nested_exec_tools(source: str) -> List[str]:
    """Name nested tools invoked by the current outer ``exec`` code tool."""
    return sorted(set(re.findall(r"\btools\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)))


def extract_current_events(session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract events from every rollout discovered by parse_session_logs.py."""
    rollout_files = session_data.get("rollout_files")
    if not isinstance(rollout_files, list) or not rollout_files:
        raise FileNotFoundError("session-data contains no discovered rollout_files")

    entries: List[Dict[str, Any]] = []
    for raw in rollout_files:
        if not isinstance(raw, dict):
            continue
        path_value = raw.get("path") or raw.get("rollout_path")
        if not isinstance(path_value, str):
            continue
        entry = dict(raw)
        entry["rollout_path"] = path_value
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Session rollout not found: {path}")
        entries.append(entry)
    if not entries:
        raise FileNotFoundError("session-data contains no readable rollout paths")

    stage_ranges = _build_stage_ranges(session_data.get("stage_timestamps", {}))
    agent_rows = {
        str(row.get("agent_id")): row
        for row in session_data.get("agent_costs", [])
        if isinstance(row, dict) and row.get("agent_id")
    }
    thread_rows = {str(row.get("thread_id")): row for row in entries}
    events: List[Dict[str, Any]] = []
    seen: set[tuple] = set()

    def make_event(
        event_type: str,
        timestamp: Optional[str],
        thread_id: Optional[str],
        details: Dict[str, Any],
        *,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = agent_rows.get(str(thread_id), {}) if thread_id else {}
        thread = thread_rows.get(str(thread_id), {}) if thread_id else {}
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "agent_id": thread_id,
            "agent_name": agent_name or metadata.get("name") or thread.get("agent_path"),
            "agent_role": agent_role or metadata.get("role") or thread.get("agent_role"),
            "stage": _assign_stage(timestamp, stage_ranges),
            "details": details,
        }

    for entry in entries:
        owner_id = str(entry.get("thread_id") or "") or None
        function_calls: Dict[str, Dict[str, Any]] = {}
        for obj in iter_owned_records(entry):
            record_type = obj.get("type")
            timestamp = obj.get("timestamp")
            payload = obj.get("payload") or {}
            payload_type = payload.get("type")

            if record_type == "response_item" and payload_type in {
                "function_call",
                "custom_tool_call",
            }:
                tool_name = str(payload.get("name") or "unknown")
                call_id = payload.get("call_id")
                if payload_type == "function_call":
                    arguments = parse_arguments(payload.get("arguments"))
                    if isinstance(call_id, str):
                        function_calls[call_id] = {
                            "tool_name": tool_name,
                            "arguments": arguments,
                        }
                    nested_tools: List[str] = []
                    tool_input: Any = arguments
                else:
                    source = payload.get("input")
                    source = source if isinstance(source, str) else ""
                    nested_tools = _nested_exec_tools(source) if tool_name == "exec" else []
                    tool_input = source

                events.append(
                    make_event(
                        "tool_call",
                        timestamp,
                        owner_id,
                        {
                            "tool_name": tool_name,
                            "tool_record_type": payload_type,
                            "namespace": payload.get("namespace"),
                            "call_id": call_id,
                            "status": payload.get("status"),
                            "nested_tools": nested_tools,
                            "input": _bounded(tool_input),
                            "rollout_path": entry["rollout_path"],
                            "line_number": obj.get("_line_number"),
                        },
                    )
                )

                if payload_type == "function_call" and tool_name in {"Write", "apply_patch"}:
                    path = arguments.get("file_path") or arguments.get("path")
                    if isinstance(path, str):
                        events.append(
                            make_event(
                                "file_write",
                                timestamp,
                                owner_id,
                                {"path": path, "via": tool_name, "call_id": call_id},
                            )
                        )
                elif payload_type == "function_call" and tool_name == "Read":
                    path = arguments.get("file_path") or arguments.get("path")
                    if isinstance(path, str):
                        events.append(
                            make_event(
                                "file_read",
                                timestamp,
                                owner_id,
                                {"path": path, "via": tool_name, "call_id": call_id},
                            )
                        )

                if payload_type == "function_call" and tool_name in {
                    "send_message",
                    "functions.send_message",
                    "followup_task",
                    "functions.followup_task",
                    "send_input",
                    "functions.send_input",
                }:
                    events.append(
                        make_event(
                            "send_message",
                            timestamp,
                            owner_id,
                            {
                                "from_agent": owner_id,
                                "to_agent": arguments.get("target") or arguments.get("to"),
                                "message": _bounded(
                                    arguments.get("message") or arguments.get("prompt")
                                ),
                                "call_id": call_id,
                                "tool_name": tool_name,
                            },
                        )
                    )

            elif record_type == "response_item" and payload_type in {
                "function_call_output",
                "custom_tool_call_output",
            }:
                output = payload.get("output")
                events.append(
                    make_event(
                        "tool_result",
                        timestamp,
                        owner_id,
                        {
                            "call_id": payload.get("call_id"),
                            "tool_record_type": payload_type,
                            "output": _bounded(output),
                            "rollout_path": entry["rollout_path"],
                            "line_number": obj.get("_line_number"),
                        },
                    )
                )
                for hit in _scan_text_for_data(_tool_output_text(output)):
                    details = dict(hit)
                    details.update(
                        {
                            "source": "tool_result",
                            "call_id": payload.get("call_id"),
                        }
                    )
                    events.append(
                        make_event("data_output", timestamp, owner_id, details)
                    )

            elif record_type == "event_msg" and payload_type == "patch_apply_end":
                changes = payload.get("changes")
                if not isinstance(changes, dict):
                    changes = {}
                for path, change in changes.items():
                    change = change if isinstance(change, dict) else {}
                    events.append(
                        make_event(
                            "file_write",
                            timestamp,
                            owner_id,
                            {
                                "path": path,
                                "via": "apply_patch",
                                "call_id": payload.get("call_id"),
                                "success": payload.get("success"),
                                "change_type": change.get("type"),
                                "move_path": change.get("move_path"),
                                "unified_diff": _bounded(change.get("unified_diff")),
                            },
                        )
                    )
                events.append(
                    make_event(
                        "tool_result",
                        timestamp,
                        owner_id,
                        {
                            "tool_name": "apply_patch",
                            "call_id": payload.get("call_id"),
                            "success": payload.get("success"),
                            "status": payload.get("status"),
                            "stdout": _bounded(payload.get("stdout")),
                            "stderr": _bounded(payload.get("stderr")),
                            "changed_files": len(changes),
                            "rollout_path": entry["rollout_path"],
                            "line_number": obj.get("_line_number"),
                        },
                    )
                )

            elif record_type == "event_msg" and payload_type == "sub_agent_activity":
                child_id = payload.get("agent_thread_id")
                kind = payload.get("kind")
                event_id = payload.get("event_id")
                event_timestamp = timestamp
                occurred_at_ms = payload.get("occurred_at_ms")
                if isinstance(occurred_at_ms, (int, float)):
                    event_timestamp = datetime.fromtimestamp(
                        float(occurred_at_ms) / 1000,
                        tz=timezone.utc,
                    ).isoformat()
                call = function_calls.get(str(event_id), {})
                arguments = call.get("arguments") or {}
                child_row = agent_rows.get(str(child_id), {})
                dedupe_key = (owner_id, child_id, kind, event_id, occurred_at_ms)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                if kind == "started":
                    event_name = "agent_spawn"
                elif kind == "interrupted":
                    event_name = "agent_complete"
                else:
                    event_name = "agent_interaction"
                events.append(
                    make_event(
                        event_name,
                        event_timestamp,
                        str(child_id) if child_id else None,
                        {
                            "kind": kind,
                            "event_id": event_id,
                            "parent_thread_id": owner_id,
                            "agent_path": payload.get("agent_path"),
                            "task_name": arguments.get("task_name"),
                            "status": "interrupted" if kind == "interrupted" else kind,
                        },
                        agent_name=(
                            arguments.get("task_name")
                            or child_row.get("name")
                            or payload.get("agent_path")
                        ),
                        agent_role=child_row.get("role"),
                    )
                )

            elif record_type == "event_msg" and payload_type == "task_started":
                events.append(
                    make_event(
                        "task_started",
                        timestamp,
                        owner_id,
                        {"turn_id": payload.get("turn_id")},
                    )
                )

            elif record_type == "event_msg" and payload_type == "task_complete":
                events.append(
                    make_event(
                        "agent_complete",
                        timestamp,
                        owner_id,
                        {
                            "turn_id": payload.get("turn_id"),
                            "status": "completed",
                            "duration_ms": payload.get("duration_ms"),
                            "time_to_first_token_ms": payload.get("time_to_first_token_ms"),
                            "total_tokens": agent_rows.get(str(owner_id), {}).get("total_tokens"),
                            "last_agent_message": _bounded(payload.get("last_agent_message")),
                        },
                    )
                )

            elif record_type == "event_msg" and payload_type == "turn_aborted":
                events.append(
                    make_event(
                        "agent_complete",
                        timestamp,
                        owner_id,
                        {
                            "turn_id": payload.get("turn_id"),
                            "status": "aborted",
                            "reason": payload.get("reason"),
                            "duration_ms": payload.get("duration_ms"),
                            "total_tokens": agent_rows.get(str(owner_id), {}).get("total_tokens"),
                        },
                    )
                )

            if (
                (record_type == "response_item" and payload_type in {"message", "agent_message"})
                or (record_type == "event_msg" and payload_type == "agent_message")
            ):
                text = _current_visible_text(payload)
                for hit in _scan_text_for_data(text):
                    events.append(
                        make_event("data_output", timestamp, owner_id, hit)
                    )

    events.sort(
        key=lambda event: (
            _parse_iso(event.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            event.get("event_type") or "",
        )
    )
    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--session-jsonl",
        required=False,
        help="Legacy-only path to a single session JSONL file",
    )
    p.add_argument(
        "--session-data",
        required=True,
        help="Path to session_data.json (from parse_session_logs.py)",
    )
    p.add_argument("--output", required=True, help="Output path for events JSON")

    args = p.parse_args()

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
        if session_data.get("source_format") == CURRENT_SOURCE_FORMAT:
            events = extract_current_events(session_data)
        else:
            if not args.session_jsonl:
                print(
                    "ERROR: --session-jsonl is required for legacy session-data",
                    file=sys.stderr,
                )
                return 1
            events = extract_events(Path(args.session_jsonl), session_data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.write_text(json.dumps(events, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {len(events)} events to: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
