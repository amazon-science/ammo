#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Filter session JSONL transcripts for AMMO drift-audit monitoring.

This utility is retained for exported JSONL sessions and historical AMMO evals.
The Codex CLI port uses hooks and deterministic validation gates as the active
control plane; this filter is optional audit tooling, not a blocking hook.

Reads a .jsonl transcript, drops noise records (progress, file-history-snapshot,
queue-operation), and extracts human-readable content from assistant/user/system
records. Outputs a conversation-log-style digest with timestamps.

With --include-subagents, also discovers and processes transcripts of subagents
spawned by the champion (depth-1 only).

Usage:
    python3 transcript_filter.py <path> [--start-line N] [--max-content-len N]
                                        [--state-file PATH]
                                        [--include-subagents] [--projects-dir PATH]

Outputs digest to stdout.
Writes LAST_LINE_PROCESSED as the final stdout line for convenience.
Writes subagent state to --state-file (JSON format).
"""

import glob
import json
import os
import re
import sqlite3
import sys
from datetime import datetime

NOISE_TYPES = {"progress", "file-history-snapshot", "queue-operation"}
MAX_CONTENT_LEN = 0  # 0 = no truncation. The filter strips ~97% noise; remaining content is what the monitor needs.
WRITE_EDIT_PREVIEW_LEN = 0  # 0 = no truncation for Write/Edit previews.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Filter AMMO session .jsonl transcripts")
    p.add_argument("path", help="Path to .jsonl transcript file")
    p.add_argument("--start-line", type=int, default=0,
                   help="Skip lines before this (0-indexed). Use LAST_LINE_PROCESSED from previous run.")
    p.add_argument("--max-content-len", type=int, default=MAX_CONTENT_LEN,
                   help="Max chars per content block before truncation")
    p.add_argument("--state-file", type=str, default=None,
                   help="Path to write subagent state (JSON)")
    p.add_argument("--include-subagents", action="store_true",
                   help="Discover and include subagent transcripts")
    p.add_argument("--projects-dir", type=str, default=None,
                   help="Directory containing .jsonl transcript files (for subagent discovery)")
    p.add_argument("--state-db", type=str, default=None,
                   help="Current Codex state_5.sqlite for exact descendant discovery")
    p.add_argument("--target-thread-id", type=str, default=None,
                   help="Exact Codex implementer thread ID whose descendants are included")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_truncate(text, max_len):
    """Truncate text with [TRUNCATED] marker. max_len=0 means no truncation."""
    if not text or max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len] + f"\n[TRUNCATED at {max_len} chars, total {len(text)}]"


def looks_like_opaque_mailbox_body(value):
    """Detect encrypted/opaque V2 mailbox bodies without leaking ciphertext noise."""
    if not isinstance(value, str):
        return False
    compact = value.strip()
    return len(compact) >= 80 and (
        compact.startswith("gAAAA")
        or bool(re.fullmatch(r"[A-Za-z0-9_+/=-]+", compact))
    )


def format_timestamp(ts):
    """Format ISO timestamp to readable form."""
    if not ts:
        return "??:??:??"
    try:
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return ts[:19] if len(ts) >= 19 else ts


def safe_get_message(record):
    """Safely extract message dict from record, handling string messages."""
    msg = record.get("message")
    if isinstance(msg, dict):
        return msg
    return {}


METADATA_ALIASES = {
    "agentName": ("agentName", "agent_name"),
    "teamName": ("teamName", "team_name"),
    "sessionId": ("sessionId", "session_id", "rollout_id"),
}
WRAPPER_TYPES = {
    "",
    "event",
    "payload",
    "response",
    "response_item",
    "response_event",
    "session_event",
    "turn",
    "turn_event",
}


def first_present(record, keys):
    """Return the first non-empty value present for any key."""
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None


def metadata_value(record, canonical_key):
    """Read metadata using Claude and Codex alias spellings."""
    return first_present(record, METADATA_ALIASES[canonical_key])


def inherit_parent_metadata(record, parent):
    """Copy wrapper timestamp/session metadata onto a child record."""
    if not parent:
        return record

    inherited = dict(record)
    if "timestamp" not in inherited and parent.get("timestamp"):
        inherited["timestamp"] = parent["timestamp"]

    for canonical, aliases in METADATA_ALIASES.items():
        if first_present(inherited, aliases):
            continue
        value = first_present(parent, aliases)
        if value:
            inherited[canonical] = value

    return inherited


def normalize_text_content(content):
    """Convert Codex input/output text blocks to Claude-compatible text blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return []

    normalized = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            continue

        btype = block.get("type")
        if btype in ("output_text", "input_text"):
            new_block = dict(block)
            new_block["type"] = "text"
            normalized.append(new_block)
        elif btype == "reasoning":
            thinking = block.get("text") or block.get("thinking") or block.get("summary") or ""
            if isinstance(thinking, list):
                thinking = "\n".join(str(item) for item in thinking)
            normalized.append({"type": "thinking", "thinking": str(thinking)})
        else:
            normalized.append(block)

    return normalized


def parse_function_arguments(arguments):
    """Parse Codex function_call arguments into a tool input dict."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}
    if arguments is None:
        return {}
    return {"arguments": arguments}


def normalize_codex_message(record):
    """Normalize a Codex message item to an assistant/user/system record."""
    role = record.get("role")
    if role not in ("assistant", "user", "system"):
        return None

    normalized = dict(record)
    normalized["type"] = role
    normalized["message"] = {
        "role": role,
        "content": normalize_text_content(record.get("content", [])),
    }
    normalized.pop("content", None)
    normalized.pop("role", None)
    return normalized


def normalize_codex_function_call(record):
    """Normalize a Codex function_call item to an assistant tool_use block."""
    name = record.get("name", "unknown")
    block = {
        "type": "tool_use",
        "name": name,
        "input": parse_function_arguments(record.get("arguments")),
    }
    for key in ("call_id", "id"):
        if record.get(key):
            block[key] = record[key]

    normalized = dict(record)
    normalized["type"] = "assistant"
    normalized["message"] = {
        "role": "assistant",
        "content": [block],
    }
    return normalized


def normalize_codex_function_output(record):
    """Normalize Codex function_call_output to a stdout-style user result."""
    output = record.get("output", "")
    normalized = dict(record)
    normalized["type"] = "user"
    normalized["toolUseResult"] = {"stdout": output if isinstance(output, str) else str(output)}
    return normalized


def _custom_output_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_custom_output_text(item) for item in value)))
    if isinstance(value, dict) and value.get("type") in {"input_text", "output_text", "text"}:
        return str(value.get("text") or "")
    return ""


def normalize_codex_custom_call(record):
    source = str(record.get("input") or "")
    nested = sorted(set(re.findall(r"\btools\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)))
    command = ""
    match = re.search(r'["\']cmd["\']\s*:\s*("(?:\\.|[^"\\])*")', source)
    if match:
        try:
            command = json.loads(match.group(1))
        except json.JSONDecodeError:
            command = ""
    block = {
        "type": "tool_use",
        "id": record.get("call_id") or record.get("id"),
        "name": "CodexExec" if record.get("name") == "exec" else str(record.get("name") or "custom_tool"),
        "input": {"nested_tools": nested, "command": command, "source": source},
    }
    normalized = dict(record)
    normalized["type"] = "assistant"
    normalized["message"] = {"role": "assistant", "content": [block]}
    return normalized


def normalize_codex_custom_output(record):
    normalized = dict(record)
    normalized["type"] = "user"
    normalized["toolUseResult"] = {"stdout": _custom_output_text(record.get("output"))}
    return normalized


def normalize_record(record, parent=None):
    """Return Claude-compatible records for a raw transcript record."""
    if not isinstance(record, dict):
        return []

    record = inherit_parent_metadata(record, parent)
    rtype = record.get("type", "")

    # Existing Claude-style records already drive the formatter; keep them intact.
    if rtype in ("assistant", "user", "system"):
        return [record]

    if rtype in WRAPPER_TYPES:
        if isinstance(record.get("item"), dict):
            return normalize_record(record["item"], parent=record)

        items = record.get("items")
        if isinstance(items, list):
            normalized = []
            for item in items:
                normalized.extend(normalize_record(item, parent=record))
            return normalized

        for wrapper_key in ("event", "payload"):
            wrapped = record.get(wrapper_key)
            if isinstance(wrapped, dict):
                return normalize_record(wrapped, parent=record)
            if isinstance(wrapped, list):
                normalized = []
                for item in wrapped:
                    normalized.extend(normalize_record(item, parent=record))
                return normalized

    if rtype == "message":
        normalized = normalize_codex_message(record)
        return [normalized] if normalized else []

    if rtype == "function_call":
        return [normalize_codex_function_call(record)]

    if rtype == "function_call_output":
        return [normalize_codex_function_output(record)]
    if rtype == "custom_tool_call":
        return [normalize_codex_custom_call(record)]
    if rtype == "custom_tool_call_output":
        return [normalize_codex_custom_output(record)]

    return [record]


# ---------------------------------------------------------------------------
# Record formatting
# ---------------------------------------------------------------------------

def extract_tool_use(block, max_len):
    """Extract readable summary from a tool_use block."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})
    if not isinstance(inp, dict):
        inp = {}

    if name == "Bash":
        desc = inp.get("description", "")
        cmd = inp.get("command", "")
        bg = " [BACKGROUND]" if inp.get("run_in_background") else ""
        timeout = f" [timeout={inp['timeout']}ms]" if inp.get("timeout") else ""
        lines = [f"  BASH{bg}{timeout}: {desc}"]
        lines.append(f"    $ {safe_truncate(cmd, max_len)}")
        return "\n".join(lines)

    elif name == "Read":
        path = inp.get("file_path", "?")
        offset = f" (offset={inp['offset']})" if inp.get("offset") else ""
        limit = f" (limit={inp['limit']})" if inp.get("limit") else ""
        return f"  READ: {path}{offset}{limit}"

    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "")
        preview = safe_truncate(content, WRITE_EDIT_PREVIEW_LEN)
        return f"  WRITE: {path} ({len(content)} chars)\n    Preview: {preview}"

    elif name == "Edit":
        path = inp.get("file_path", "?")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        return (f"  EDIT: {path}\n"
                f"    old ({len(old)} chars): {safe_truncate(old, WRITE_EDIT_PREVIEW_LEN)}\n"
                f"    new ({len(new)} chars): {safe_truncate(new, WRITE_EDIT_PREVIEW_LEN)}")

    elif name == "Agent":
        desc = inp.get("description", "")
        agent_type = inp.get("subagent_type", "general")
        prompt = inp.get("prompt", "")
        agent_name = inp.get("name", "unnamed")
        model = inp.get("model", "default")
        bg = " [BACKGROUND]" if inp.get("run_in_background") else ""
        iso = f" [isolation={inp['isolation']}]" if inp.get("isolation") else ""
        team = f" [team={inp['team_name']}]" if inp.get("team_name") else ""
        lines = [f"  AGENT{bg}{iso}{team}: {desc} (type={agent_type}, name={agent_name}, model={model})"]
        lines.append(f"    Prompt: {safe_truncate(prompt, max_len)}")
        return "\n".join(lines)

    elif name == "CodexExec":
        nested = ", ".join(inp.get("nested_tools") or []) or "unknown"
        command = inp.get("command") or ""
        source = inp.get("source") or ""
        lines = [f"  CODEX_EXEC nested=[{nested}]"]
        if command:
            lines.append(f"    $ {safe_truncate(str(command), max_len)}")
        lines.append(f"    Source: {safe_truncate(str(source), max_len)}")
        return "\n".join(lines)

    elif name in ("send_input", "send_message", "followup_task"):
        to = inp.get("target", inp.get("to", inp.get("recipient", "?")))
        summary = inp.get("summary", "")
        msg = inp.get("message", inp.get("content", ""))
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=2)
        elif looks_like_opaque_mailbox_body(msg):
            msg = "[OPAQUE V2 MAILBOX BODY — use the monitor observation log/queue for plaintext audit evidence]"
        lines = [f"  SUBAGENT_MESSAGE -> {to}: {summary}"]
        lines.append(f"    {safe_truncate(str(msg), max_len)}")
        return "\n".join(lines)

    elif name in ("Grep", "Glob"):
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        return f"  {name.upper()}: pattern='{pattern}' path={path}"

    elif name == "TaskCreate":
        subject = inp.get("subject", "")
        return f"  TASK_CREATE: {subject}"

    elif name == "TaskUpdate":
        tid = inp.get("taskId", "?")
        status = inp.get("status", "")
        return f"  TASK_UPDATE: #{tid} -> {status}"

    elif name == "Skill":
        skill = inp.get("skillName", inp.get("name", "?"))
        return f"  SKILL: {skill}"

    elif name in ("spawn_agent", "functions.spawn_agent"):
        agent_name = inp.get("task_name", inp.get("agent_type", inp.get("name", "?")))
        return f"  SPAWN_AGENT: {agent_name}"

    elif name in (
        "send_input", "functions.send_input", "send_message", "functions.send_message",
        "followup_task", "functions.followup_task",
    ):
        target = inp.get("target", "?")
        return f"  SEND_MESSAGE: {target}"

    elif name in ("close_agent", "functions.close_agent"):
        target = inp.get("target", inp.get("name", "?"))
        return f"  CLOSE_AGENT: {target}"

    elif name in ("interrupt_agent", "functions.interrupt_agent"):
        target = inp.get("target", "?")
        return f"  INTERRUPT_AGENT: {target}"

    elif name in ("Codex subagent cohort spawn", "CodexSubagentClose"):
        target = inp.get("team_name", inp.get("name", "?"))
        return f"  LEGACY_SUBAGENT_EVENT: {target}"

    elif name in ("EnterWorktree", "ExitWorktree"):
        wt = inp.get("name", "?")
        return f"  {name.upper()}: {wt}"

    else:
        inp_str = json.dumps(inp)
        return f"  {name.upper()}: {safe_truncate(inp_str, min(500, max_len))}"


def extract_tool_result(record, max_len):
    """Extract readable summary from a tool result user record."""
    tur = record.get("toolUseResult")
    if tur is None:
        return None

    if isinstance(tur, str):
        return f"  RESULT [ERROR]: {safe_truncate(tur, max_len)}"

    if not isinstance(tur, dict):
        return f"  RESULT: {safe_truncate(str(tur), 200)}"

    # Agent spawn result
    if tur.get("status") in ("teammate_spawned", "async_launched", "completed"):
        status = tur["status"]
        agent_id = tur.get("agentId", tur.get("agent_id", "?"))
        name = tur.get("name", "")
        team = tur.get("team_name", "")

        if status == "completed":
            content = tur.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text = c.get("text", "")
                        break
            elif isinstance(content, str):
                text = content
            dur_ms = tur.get("totalDurationMs", 0)
            tokens = tur.get("totalTokens", 0)
            tools = tur.get("totalToolUseCount", 0)
            return (f"  AGENT_RESULT [completed]: id={agent_id} "
                    f"dur={dur_ms/1000:.0f}s tokens={tokens} tools={tools}\n"
                    f"    {safe_truncate(text, max_len)}")

        return f"  AGENT_RESULT [{status}]: name={name} id={agent_id} team={team}"

    # send_input result
    if "recipients" in tur or tur.get("routing"):
        success = tur.get("success", False)
        msg = tur.get("message", "")
        return f"  MSG_RESULT: {'OK' if success else 'FAIL'} - {msg}"

    # Shutdown request/response
    if "request_id" in tur and "target" in tur:
        target = tur.get("target", "?")
        return f"  SHUTDOWN_SENT: target={target} id={tur.get('request_id', '?')}"

    # Historical team create/delete result
    if "team_name" in tur and "team_file_path" in tur:
        return f"  TEAM_RESULT: {tur.get('team_name')} -> {tur.get('message', '')}"

    # File read result - show path only, skip content
    if "file" in tur and isinstance(tur["file"], dict):
        fp = tur["file"].get("filePath", "?")
        content_len = len(tur["file"].get("content", ""))
        return f"  READ_RESULT: {fp} ({content_len} chars)"

    # Write/Edit result
    if "filePath" in tur and ("newString" in tur or "content" in tur):
        fp = tur["filePath"]
        if "newString" in tur:
            return f"  EDIT_RESULT: {fp} (modified={not tur.get('userModified', False)})"
        tp = tur.get("type", "write")
        return f"  WRITE_RESULT [{tp}]: {fp}"

    # Bash result (stdout/stderr)
    if "stdout" in tur or "stderr" in tur:
        stdout = tur.get("stdout", "")
        stderr = tur.get("stderr", "")
        interrupted = tur.get("interrupted", False)
        bg_id = tur.get("backgroundTaskId", "")

        parts = []
        if interrupted:
            parts.append("[INTERRUPTED]")
        if bg_id:
            parts.append(f"[bg={bg_id}]")

        prefix = " ".join(parts) + " " if parts else ""

        if stderr and not stdout:
            return f"  BASH_RESULT {prefix}[STDERR]:\n    {safe_truncate(stderr, max_len)}"
        elif stdout:
            return f"  BASH_RESULT {prefix}:\n    {safe_truncate(stdout, max_len)}"
        else:
            return f"  BASH_RESULT {prefix}: (no output)"

    # Grep/Glob result
    if "filenames" in tur:
        mode = tur.get("mode", "?")
        num = tur.get("numFiles", len(tur.get("filenames", [])))
        content = tur.get("content", "")
        if mode == "content" and content:
            return f"  SEARCH_RESULT [{mode}]: {num} files\n    {safe_truncate(content, max_len)}"
        else:
            files = tur.get("filenames", [])
            file_list = ", ".join(files[:10])
            if len(files) > 10:
                file_list += f"... (+{len(files)-10} more)"
            return f"  SEARCH_RESULT [{mode}]: {num} files: {file_list}"

    # Task result
    if "task" in tur or "tasks" in tur:
        if "tasks" in tur:
            tasks = tur["tasks"]
            summaries = [f"#{t.get('id','?')}: {t.get('subject','?')} [{t.get('status','?')}]" for t in tasks[:5]]
            return f"  TASK_LIST: {'; '.join(summaries)}"
        else:
            t = tur["task"]
            return f"  TASK: #{t.get('id','?')}: {t.get('subject','?')}"

    if "taskId" in tur:
        change = tur.get("statusChange", {})
        if change:
            return f"  TASK_RESULT: #{tur['taskId']} {change.get('from','')} -> {change.get('to','')}"
        return f"  TASK_RESULT: #{tur['taskId']} success={tur.get('success')}"

    # Plan result
    if "plan" in tur:
        return f"  PLAN: {safe_truncate(str(tur.get('plan', '')), 300)}"

    # Skill result
    if "commandName" in tur:
        return f"  SKILL_RESULT: {tur['commandName']} success={tur.get('success')}"

    # Fallback
    keys = sorted(tur.keys())
    return f"  RESULT: keys={keys}, preview={safe_truncate(json.dumps(tur), 500)}"


def process_assistant_record(record, max_len):
    """Extract content from an assistant record."""
    lines = []
    content = safe_get_message(record).get("content", [])

    if not isinstance(content, list):
        return lines

    for block in content:
        if not isinstance(block, dict):
            continue

        btype = block.get("type")

        if btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                lines.append(f"  THINKING ({len(thinking)} chars):")
                lines.append(f"    {safe_truncate(thinking, max_len)}")

        elif btype == "text":
            text = block.get("text", "")
            if text:
                lines.append(f"  TEXT ({len(text)} chars):")
                lines.append(f"    {safe_truncate(text, max_len)}")

        elif btype == "tool_use":
            lines.append(extract_tool_use(block, max_len))

        elif btype == "tool_result":
            result_content = block.get("content", "")
            lines.append(f"  TOOL_RESULT: {safe_truncate(str(result_content), max_len)}")

    return lines


def process_user_record(record, max_len):
    """Extract content from a user record."""
    lines = []

    tool_result = extract_tool_result(record, max_len)
    if tool_result:
        lines.append(tool_result)
        return lines

    content = safe_get_message(record).get("content", [])
    if isinstance(content, str):
        lines.append(f"  MESSAGE: {safe_truncate(content, max_len)}")
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    lines.append(f"  MESSAGE: {safe_truncate(text, max_len)}")
                elif btype == "tool_result":
                    rc = block.get("content", "")
                    is_error = block.get("is_error", False)
                    tag = "ERROR" if is_error else "RESULT"
                    lines.append(f"  {tag}: {safe_truncate(str(rc), max_len)}")
            elif isinstance(block, str):
                lines.append(f"  MESSAGE: {safe_truncate(block, max_len)}")

    return lines


def process_system_record(record, max_len):
    """Extract content from a system record."""
    content = safe_get_message(record).get("content", [])
    lines = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                lines.append(f"  SYSTEM ({len(text)} chars): {safe_truncate(text, max_len)}")
    elif isinstance(content, str):
        lines.append(f"  SYSTEM ({len(content)} chars): {safe_truncate(content, 300)}")
    return lines


# ---------------------------------------------------------------------------
# Core processing — extracted from main() for reuse with subagents
# ---------------------------------------------------------------------------

def process_transcript(path, start_line, max_content_len, label=""):
    """Process a single .jsonl transcript file.

    Args:
        path: Path to .jsonl file
        start_line: Skip lines before this (0-indexed)
        max_content_len: Truncation limit per content block
        label: Prefix for output lines (e.g., "sub:validator-1")

    Returns:
        (formatted_lines, raw_records, last_line_processed, metadata)
        metadata: dict with agent_name, team_name, session_id, stats
    """
    formatted_lines = []
    raw_records = []

    line_num = 0
    last_successful = start_line
    processed = 0
    skipped_noise = 0
    skipped_parse = 0

    agent_name = None
    team_name = None
    session_id = None

    prefix = f"[{label}] " if label else ""

    try:
        with open(path, "r") as f:
            for raw_line in f:
                current_line = line_num
                line_num += 1

                if current_line < start_line:
                    continue

                raw_line = raw_line.strip()
                if not raw_line:
                    last_successful = line_num
                    continue

                if len(raw_line) > 5_000_000:
                    skipped_parse += 1
                    last_successful = line_num
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    # C1 fix: STOP at first parse error — likely the active write frontier.
                    skipped_parse += 1
                    break

                records = normalize_record(record)
                raw_records.extend(records)

                for record in records:
                    rtype = record.get("type", "")

                    if not agent_name:
                        agent_name = metadata_value(record, "agentName")
                    if not team_name:
                        team_name = metadata_value(record, "teamName")
                    if not session_id:
                        session_id = metadata_value(record, "sessionId")

                    if rtype in NOISE_TYPES:
                        skipped_noise += 1
                        continue

                    ts = format_timestamp(record.get("timestamp"))

                    try:
                        if rtype == "assistant":
                            content_lines = process_assistant_record(record, max_content_len)
                            if content_lines:
                                formatted_lines.append(f"{prefix}[{ts}] ASSISTANT (line {current_line}):")
                                for cl in content_lines:
                                    formatted_lines.append(f"{prefix}{cl}" if prefix else cl)
                                processed += 1

                        elif rtype == "user":
                            user_type = record.get("userType", "")
                            content_lines = process_user_record(record, max_content_len)
                            if content_lines:
                                rec_label = "TEAMMATE" if user_type == "external" else "USER"
                                formatted_lines.append(f"{prefix}[{ts}] {rec_label} (line {current_line}):")
                                for cl in content_lines:
                                    formatted_lines.append(f"{prefix}{cl}" if prefix else cl)
                                processed += 1

                        elif rtype == "system":
                            content_lines = process_system_record(record, max_content_len)
                            if content_lines:
                                formatted_lines.append(f"{prefix}[{ts}] SYSTEM (line {current_line}):")
                                for cl in content_lines:
                                    formatted_lines.append(f"{prefix}{cl}" if prefix else cl)
                                processed += 1

                    except Exception as e:
                        formatted_lines.append(
                            f"{prefix}[{ts}] PROCESS_ERROR (line {current_line}): {type(e).__name__}: {e}")
                        skipped_parse += 1

                last_successful = line_num

    except Exception as e:
        print(f"FATAL_ERROR: {type(e).__name__}: {e}", file=sys.stderr)

    metadata = {
        "agent_name": agent_name,
        "team_name": team_name,
        "session_id": session_id,
        "lines_read": line_num - start_line,
        "records_output": processed,
        "noise_skipped": skipped_noise,
        "parse_errors": skipped_parse,
    }

    return formatted_lines, raw_records, last_successful, metadata


# ---------------------------------------------------------------------------
# Subagent discovery
# ---------------------------------------------------------------------------

def discover_subagents(champion_path, projects_dir, raw_records, pending_names=None):
    """Find subagent transcripts by matching spawned agent names.

    Args:
        champion_path: Path to champion's .jsonl (to skip in glob)
        projects_dir: Directory containing .jsonl transcript files
        raw_records: List of raw JSON record dicts from champion processing
        pending_names: Set of agent names from state file that haven't been
                       found yet (R3-C1 fix: prevents spawn window gap)

    Returns:
        (discovered, still_pending):
            discovered: dict {agent_name: transcript_path}
            still_pending: set of names not yet found
    """
    spawned_names = set(pending_names or [])
    spawn_tool_names = {"Agent", "spawn_agent", "functions.spawn_agent"}
    for record in raw_records:
        if record.get('type') != 'assistant':
            continue
        msg = record.get('message', {})
        if not isinstance(msg, dict):
            continue
        for block in msg.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') in spawn_tool_names:
                inp = block.get('input', {})
                if not isinstance(inp, dict):
                    inp = {}
                agent_name = inp.get('name') or inp.get('target') or inp.get('agent_type')
                if agent_name:
                    spawned_names.add(agent_name)

    if not spawned_names:
        return {}, set()

    discovered = {}
    champion_real = os.path.realpath(champion_path)
    files = sorted(
        glob.glob(os.path.join(projects_dir, '*.jsonl')),
        key=os.path.getmtime, reverse=True)[:50]

    for f in files:
        if os.path.realpath(f) == champion_real:
            continue
        if len(discovered) == len(spawned_names):
            break
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i > 20:
                    break
                try:
                    d = json.loads(line.strip())
                    for record in normalize_record(d):
                        name = metadata_value(record, "agentName")
                        if name in spawned_names and name not in discovered:
                            discovered[name] = f
                            break
                except Exception:
                    continue

    still_pending = spawned_names - set(discovered.keys())
    return discovered, still_pending


def discover_codex_descendants(state_db, target_thread_id):
    """Resolve every descendant through the exact current Codex spawn graph."""
    db = os.path.realpath(os.path.expanduser(state_db))
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            WITH RECURSIVE descendants(id, depth) AS (
              SELECT child_thread_id, 1 FROM thread_spawn_edges WHERE parent_thread_id = ?
              UNION ALL
              SELECT edge.child_thread_id, descendants.depth + 1
                FROM descendants
                JOIN thread_spawn_edges AS edge ON edge.parent_thread_id = descendants.id
               WHERE descendants.depth < 64
            )
            SELECT threads.id, threads.rollout_path, threads.agent_path
              FROM descendants JOIN threads ON threads.id = descendants.id
             ORDER BY descendants.depth, threads.created_at_ms, threads.id
            """,
            (target_thread_id,),
        ).fetchall()
    finally:
        connection.close()
    discovered = {}
    for thread_id, rollout_path, agent_path in rows:
        path = os.path.realpath(os.path.expanduser(str(rollout_path)))
        if os.path.isfile(path):
            discovered[str(thread_id)] = {
                "path": path,
                "label": str(agent_path or thread_id).rstrip("/").rsplit("/", 1)[-1],
            }
    return discovered


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------

def read_state(state_file):
    """Read state file, handling old format and corruption gracefully."""
    if not state_file or not os.path.exists(state_file):
        return {}
    try:
        with open(state_file) as f:
            content = f.read().strip()
        if not content:
            return {}
        # Old format: bare integer (champion line only)
        try:
            int(content)
            return {}  # Old format has no subagent state; ignore it
        except ValueError:
            pass
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


def write_state_atomic(state_file, state):
    """Write state file atomically via temp-file + rename."""
    if not state_file:
        return
    tmp = state_file + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.rename(tmp, state_file)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    path = args.path

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    state = read_state(args.state_file)
    start_line = args.start_line

    # Process champion transcript
    champion_lines, raw_records, last_line, meta = process_transcript(
        path, start_line, args.max_content_len)

    for line in champion_lines:
        print(line)

    # Subagent processing
    subagent_state = state.get('subagents', {})
    pending_subagents = set(state.get('pending_subagents', []))
    still_pending = set()

    if args.include_subagents and args.state_db and args.target_thread_id:
        discovered_threads = discover_codex_descendants(args.state_db, args.target_thread_id)
        new_subagent_state = {}
        for thread_id, info in discovered_threads.items():
            previous = subagent_state.get(thread_id, {})
            start = previous.get("last_line", 0) if previous.get("path") == info["path"] else 0
            sub_lines, _, sub_last_line, _ = process_transcript(
                info["path"], start, args.max_content_len, label=f"sub:{info['label']}"
            )
            for line in sub_lines:
                print(line)
            new_subagent_state[thread_id] = {
                "path": info["path"], "label": info["label"], "last_line": sub_last_line
            }
        state["subagents"] = new_subagent_state
        state["pending_subagents"] = []
    elif args.include_subagents and args.projects_dir:
        discovered, still_pending = discover_subagents(
            path, args.projects_dir, raw_records,
            pending_names=pending_subagents)

        # Include previously discovered subagents from state
        for name, info in subagent_state.items():
            if name not in discovered and os.path.exists(info['path']):
                discovered[name] = info['path']

        new_subagent_state = {}
        for name, sub_path in sorted(discovered.items()):
            # R3-I1: path mismatch = new transcript, reset offset
            prev_info = subagent_state.get(name, {})
            if prev_info.get('path') != sub_path:
                sub_start = 0
            else:
                sub_start = prev_info.get('last_line', 0)

            sub_lines, _, sub_last_line, _ = process_transcript(
                sub_path, sub_start, args.max_content_len, label=f"sub:{name}")

            for line in sub_lines:
                print(line)

            new_subagent_state[name] = {'path': sub_path, 'last_line': sub_last_line}

        subagent_state = new_subagent_state

    # Print summary
    print(f"\n--- FILTER SUMMARY ---")
    if meta["agent_name"]:
        print(f"AGENT: {meta['agent_name']}")
    if meta["team_name"]:
        print(f"TEAM: {meta['team_name']}")
    if meta["session_id"]:
        print(f"SESSION: {meta['session_id']}")
    print(f"LINES_READ: {meta['lines_read']}")
    print(f"RECORDS_OUTPUT: {meta['records_output']}")
    print(f"NOISE_SKIPPED: {meta['noise_skipped']}")
    print(f"PARSE_ERRORS: {meta['parse_errors']}")
    print(f"LAST_LINE_PROCESSED: {last_line}")

    if args.include_subagents:
        discovered_names = sorted(subagent_state.keys())
        pending_names_sorted = sorted(still_pending)
        if discovered_names:
            print(f"SUBAGENTS_DISCOVERED: {len(discovered_names)} ({', '.join(discovered_names)})")
        if pending_names_sorted:
            print(f"SUBAGENTS_PENDING: {len(pending_names_sorted)} ({', '.join(pending_names_sorted)})")
        if discovered_names:
            parts = [f"{n}={subagent_state[n]['last_line']}" for n in discovered_names]
            print(f"SUBAGENT_LINES: {', '.join(parts)}")

    # Write state file (atomic, JSON)
    write_state_atomic(args.state_file, {
        'pending_subagents': sorted(still_pending),
        'subagents': subagent_state
    })


if __name__ == "__main__":
    main()
