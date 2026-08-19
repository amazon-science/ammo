#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Parse Codex CLI session rollouts for ground-truth timing and token costs.

Replaces manual tracking in state.json with data derived from the actual session logs.

Current Codex (0.144.x) stores the exact thread-to-rollout mapping in
``$CODEX_HOME/state_5.sqlite``. The parser resolves ``threads.id`` exactly,
walks ``thread_spawn_edges`` recursively, and reads every discovered rollout.

Legacy Claude-layout JSONL is still supported. Its format is:
  type: "user", "assistant", "progress", "queue-operation", "system", "file-history-snapshot"
  timestamp: ISO 8601 (e.g., "2026-03-11T18:07:00.123Z")
  sessionId: parent session UUID
  agentId: present on subagent messages
  message.content: array of text/tool_use blocks

Usage:
  python parse_session_logs.py \\
    --session-id <THREAD_UUID> \\
    [--codex-home ~/.codex] \\
    [--state-db ~/.codex/state_5.sqlite] \\
    [--artifact-dir <path>] \\
    --output /tmp/ammo_eval_session_data.json

For a legacy projects-layout transcript, add
``--session-dir ~/.codex/projects`` (searched one level down) or the
specific project directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from codex_session import (  # noqa: E402
    CURRENT_SOURCE_FORMAT,
    CodexRolloutMissing,
    agent_name as _codex_agent_name,
    default_codex_home,
    epoch_ms_to_iso,
    iter_owned_records,
    parse_arguments,
    resolve_thread_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime, returning None on failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Handle Z suffix
        ts_clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_clean)
    except (ValueError, TypeError):
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime as ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


def _infer_role(name: Optional[str], subagent_type: Optional[str], description: Optional[str]) -> str:
    """Infer agent role from name, subagent_type, and description."""
    # Prefer explicit subagent_type
    if subagent_type:
        st = subagent_type.lower()
        if "researcher" in st:
            return "researcher"
        # Legacy Claude implementation agents use ``ammo-impl-champion``. That
        # contains "champion", so implementation must be detected first.
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
    """Extract text from message content (string or list of blocks)."""
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
# Session JSONL Streaming Parser
# ---------------------------------------------------------------------------

class SessionParser:
    """Streams a session JSONL file and extracts relevant events."""

    def __init__(self, session_path: Path):
        self.session_path = session_path

        # Registry: tool_use_id -> agent spawn info
        self.agent_spawns: Dict[str, Dict[str, Any]] = {}
        # Registry: task_id -> task-notification data
        self.task_notifications: Dict[str, Dict[str, Any]] = {}

        # Team lifecycle events: list of {action, name, timestamp}
        self.team_events: List[Dict[str, Any]] = []

        # All events in order for stage inference
        self.events: List[Dict[str, Any]] = []

        # Session boundaries
        self.session_start: Optional[str] = None
        self.session_end: Optional[str] = None

    def parse(self) -> None:
        """Stream and parse the JSONL file."""
        with open(self.session_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._process_line(obj)

    def _process_line(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        if ts:
            if self.session_start is None:
                self.session_start = ts
            self.session_end = ts

        msg_type = obj.get("type")

        if msg_type == "assistant":
            self._process_assistant(obj)
        elif msg_type == "user":
            self._process_user(obj)

    def _process_assistant(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return
        record_team_name = obj.get("teamName")

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = block.get("name", "")
            tool_use_id = block.get("id")
            inp = block.get("input", {}) or {}

            if tool_name in {"Agent", "spawn_agent", "functions.spawn_agent"}:
                agent_name = inp.get("task_name") or inp.get("name") or inp.get("target")
                subagent_type = inp.get("subagent_type") or inp.get("agent_type")
                description = inp.get("description") or inp.get("message")
                model = inp.get("model")
                team_name = inp.get("team_name") or inp.get("cohort_name") or record_team_name
                role = _infer_role(agent_name, subagent_type, description)

                spawn_info = {
                    "tool_use_id": tool_use_id,
                    "spawn_timestamp": ts,
                    "name": agent_name or description,
                    "display_name": agent_name,
                    "subagent_type": subagent_type,
                    "description": description,
                    "model": model,
                    "team_name": team_name,
                    "cohort_name": team_name,
                    "role": role,
                }
                if tool_use_id:
                    self.agent_spawns[tool_use_id] = spawn_info

                self.events.append({
                    "event": "agent_spawn",
                    "timestamp": ts,
                    "tool_use_id": tool_use_id,
                    "role": role,
                    "name": agent_name or description,
                    "subagent_type": subagent_type,
                    "team_name": team_name,
                    "cohort_name": team_name,
                })

            elif tool_name in {"TeamCreate", "Codex subagent cohort spawn"}:
                cohort_name = inp.get("cohort_name") or inp.get("team_name")
                self.team_events.append({
                    "action": "create",
                    "name": cohort_name,
                    "team_name": cohort_name,
                    "cohort_name": cohort_name,
                    "timestamp": ts,
                    "description": inp.get("description"),
                })
                self.events.append({
                    "event": "team_create",
                    "timestamp": ts,
                    "cohort_name": cohort_name,
                })

            elif tool_name in {
                "TeamDelete", "close_agent", "functions.close_agent", "CodexSubagentClose",
                "interrupt_agent", "functions.interrupt_agent",
            }:
                # Close input is often empty; infer team from the most recent
                # create marker without a matching delete.
                cohort_name = inp.get("cohort_name") or inp.get("team_name")
                if not cohort_name:
                    # Find the most recently created team not yet deleted
                    deleted_names = {e.get("name") for e in self.team_events if e["action"] == "delete"}
                    active = [e for e in self.team_events
                              if e["action"] == "create" and e.get("name") not in deleted_names]
                    if active:
                        cohort_name = active[-1]["name"]
                if cohort_name is not None:
                    self.team_events.append({
                        "action": "delete",
                        "name": cohort_name,
                        "team_name": cohort_name,
                        "cohort_name": cohort_name,
                        "timestamp": ts,
                    })
                    self.events.append({
                        "event": "team_delete",
                        "timestamp": ts,
                        "cohort_name": cohort_name,
                    })

    def _process_user(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        msg = obj.get("message", {})
        content = msg.get("content", "")
        content_text = _get_content_text(content)

        if "<task-notification>" not in content_text:
            return

        notif = _extract_task_notification(content_text)
        if not notif:
            return

        notif["completion_timestamp"] = ts
        task_id = notif.get("task_id")
        tool_use_id = notif.get("tool_use_id")

        if task_id:
            self.task_notifications[task_id] = notif
        # Also index by tool_use_id for quick lookup
        if tool_use_id:
            self.task_notifications[f"by_tool_use_id:{tool_use_id}"] = notif

        self.events.append({
            "event": "task_complete",
            "timestamp": ts,
            "task_id": task_id,
            "tool_use_id": tool_use_id,
            "status": notif.get("status"),
            "total_tokens": notif.get("total_tokens"),
            "duration_ms": notif.get("duration_ms"),
        })


# ---------------------------------------------------------------------------
# Stage Timestamp Inference
# ---------------------------------------------------------------------------

def _derive_debate_round_starts(
    events: List[Dict[str, Any]],
    team_events: List[Dict[str, Any]],
) -> List[str]:
    """Derive debate-round starts without relying on explicit team tools.

    Claude Code 2.1.179 removed TeamCreate/TeamDelete from normal AMMO logs; Codex
    can likewise run without a synthetic cohort marker. When explicit lifecycle
    markers exist, they remain authoritative. Otherwise, each new champion spawn
    cluster after a researcher or implementer phase starts a new debate round.
    """
    team_creates = [
        e["timestamp"]
        for e in team_events
        if e.get("action") == "create" and e.get("timestamp")
    ]
    if team_creates:
        return sorted(team_creates)

    round_starts: List[str] = []
    seen_phase_break_since_last_champion = True
    for event in events:
        if event.get("event") != "agent_spawn":
            continue
        role = event.get("role")
        ts = event.get("timestamp")
        if role == "champion":
            if seen_phase_break_since_last_champion and ts is not None:
                round_starts.append(ts)
            seen_phase_break_since_last_champion = False
        elif role in ("implementer", "researcher"):
            seen_phase_break_since_last_champion = True
    return round_starts


def _infer_stage_timestamps(
    events: List[Dict[str, Any]],
    agent_spawns: Dict[str, Dict[str, Any]],
    task_notifications: Dict[str, Dict[str, Any]],
    team_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Infer per-round stage timestamps from observable events."""

    debate_round_starts = _derive_debate_round_starts(events, team_events)
    create_names = {
        e["timestamp"]: e.get("name") or e.get("team_name") or e.get("cohort_name")
        for e in team_events
        if e.get("action") == "create" and e.get("timestamp")
    }
    team_creates = [(ts, create_names.get(ts)) for ts in debate_round_starts]
    team_deletes = [
        (e["timestamp"], e.get("name") or e.get("team_name") or e.get("cohort_name"))
        for e in team_events
        if e.get("action") == "delete"
    ]

    # Get researcher spawns and completions
    researcher_spawns = [e for e in events if e.get("event") == "agent_spawn" and e.get("role") == "researcher"]
    researcher_completions = [e for e in events if e.get("event") == "task_complete" and
                               e.get("tool_use_id") in {s.get("tool_use_id") for s in researcher_spawns}]

    # Get implementer spawns
    implementer_spawns = [e for e in events if e.get("event") == "agent_spawn" and e.get("role") == "implementer"]
    implementer_completions = [e for e in events if e.get("event") == "task_complete" and
                                e.get("tool_use_id") in {s.get("tool_use_id") for s in implementer_spawns}]

    champion_spawns = [e for e in events if e.get("event") == "agent_spawn" and e.get("role") == "champion"]
    champion_completions = [e for e in events if e.get("event") == "task_complete" and
                            e.get("tool_use_id") in {s.get("tool_use_id") for s in champion_spawns}]

    def _derive_debate_end(
        debate_start: Optional[str],
        next_round_start: Optional[str],
    ) -> Optional[str]:
        """Estimate debate end when no explicit close marker exists."""
        if debate_start is None:
            return None
        completions = [
            c["timestamp"]
            for c in champion_completions
            if c.get("timestamp")
            and c["timestamp"] >= debate_start
            and (next_round_start is None or c["timestamp"] < next_round_start)
        ]
        if completions:
            return max(completions)
        spawns = [
            s["timestamp"]
            for s in champion_spawns
            if s.get("timestamp")
            and s["timestamp"] >= debate_start
            and (next_round_start is None or s["timestamp"] < next_round_start)
        ]
        if spawns:
            return max(spawns)
        return None

    result = {}

    # Number of debate rounds = number of derived debate-round starts.
    num_debate_rounds = len(team_creates)

    # Round 1 always exists
    # Stage 1: researcher spawn
    # Stage 2: researcher completion
    # Stage 3: Codex subagent cohort spawn for debate
    # Stage 3 end: first legacy close or current interrupt marker
    # Stage 4-5: implementer spawns
    # Stage 4-5 end: last implementer completion
    # Stages 6-7: inferred from remaining

    # For multi-round: between agent release and next Codex subagent cohort spawn = campaign eval (stages 6-7)
    # Then next Codex subagent cohort spawn starts next debate round

    session_start_ts = events[0]["timestamp"] if events else None

    # --- Round 1 ---
    r1: Dict[str, Any] = {}

    # Stage 1 start: first researcher spawn
    r1_researcher_spawns = [s for s in researcher_spawns][:2]  # could be 2 researchers in round 1 (retry)
    if r1_researcher_spawns:
        r1["stage_1_baseline_start"] = r1_researcher_spawns[0]["timestamp"]
        # Stage 2 completion: last researcher completion before Codex subagent cohort spawn
        # Find the completion of the last researcher spawned before Codex subagent cohort spawn
        tc1_ts = team_creates[0][0] if team_creates else None
        pre_tc1_completions = [c for c in researcher_completions
                                if tc1_ts is None or c["timestamp"] <= tc1_ts]
        if pre_tc1_completions:
            r1["stage_2_bottleneck_end"] = pre_tc1_completions[-1]["timestamp"]

    # Stage 3: Codex subagent cohort spawn for debate
    if team_creates:
        r1["stage_3_debate_start"] = team_creates[0][0]
        tc2_start_for_end = team_creates[1][0] if len(team_creates) > 1 else None
        # Stage 3 end: prefer an explicit close marker; otherwise derive from
        # champion completions in this round.
        post_tc1_deletes = [d for d in team_deletes if d[0] >= team_creates[0][0]]
        if post_tc1_deletes:
            r1["stage_3_debate_end"] = post_tc1_deletes[0][0]
        else:
            derived_end = _derive_debate_end(team_creates[0][0], tc2_start_for_end)
            if derived_end is not None:
                r1["stage_3_debate_end"] = derived_end

    # Stage 4-5: implementer spawns
    # Find implementers spawned after debate-agent release and before the next cohort (if any)
    r1_debate_end = r1.get("stage_3_debate_end")
    tc2_ts = team_creates[1][0] if len(team_creates) > 1 else None

    r1_implementers = [s for s in implementer_spawns
                       if (r1_debate_end is None or s["timestamp"] >= r1_debate_end) and
                          (tc2_ts is None or s["timestamp"] < tc2_ts)]
    if r1_implementers:
        r1["stage_4_5_impl_start"] = r1_implementers[0]["timestamp"]
        # Stage 4-5 end: last completion of those implementers
        r1_impl_ids = {s.get("tool_use_id") for s in r1_implementers}
        r1_impl_completions = [c for c in implementer_completions if c.get("tool_use_id") in r1_impl_ids]
        if r1_impl_completions:
            r1["stage_4_5_impl_end"] = max(c["timestamp"] for c in r1_impl_completions)

    # Stages 6-7: between impl end and next round's researcher spawn (or session end)
    # Note: async pipeline means Round 2 debate may START during Round 1 implementation,
    # so tc2_ts can be BEFORE r1_impl_end. Use the later of tc2_ts and r1_impl_end.
    r1_impl_end = r1.get("stage_4_5_impl_end")
    if r1_impl_end:
        r1["stage_6_7_eval_start"] = r1_impl_end
        # Find the next researcher spawn after impl end (signals next round's re-profiling)
        next_researchers = [e for e in events
                           if e.get("event") == "agent_spawn" and e.get("role") == "researcher"
                           and e["timestamp"] > r1_impl_end]
        if next_researchers:
            r1["stage_6_7_eval_end"] = next_researchers[0]["timestamp"]
        elif events:
            r1["stage_6_7_eval_end"] = events[-1]["timestamp"]

    result["round_1"] = r1

    # --- Round 2+ (if multi-round) ---
    for ri in range(1, len(team_creates)):
        round_num = ri + 1
        rN: Dict[str, Any] = {}
        tc_ts = team_creates[ri][0]
        tc_next_ts = team_creates[ri + 1][0] if ri + 1 < len(team_creates) else None

        # Stage 3: debate for this round
        rN["stage_3_debate_start"] = tc_ts
        # Find corresponding close marker after this debate start; otherwise
        # derive from this round's champion completions.
        post_tc_deletes = [d for d in team_deletes if d[0] >= tc_ts and
                           (tc_next_ts is None or d[0] < tc_next_ts)]
        if post_tc_deletes:
            rN["stage_3_debate_end"] = post_tc_deletes[0][0]
        else:
            derived_end = _derive_debate_end(tc_ts, tc_next_ts)
            if derived_end is not None:
                rN["stage_3_debate_end"] = derived_end

        # Stage 4-5: implementers after debate ends
        rN_debate_end = rN.get("stage_3_debate_end")
        rN_implementers = [s for s in implementer_spawns
                           if (rN_debate_end is not None and s["timestamp"] >= rN_debate_end) and
                              (tc_next_ts is None or s["timestamp"] < tc_next_ts)]
        if rN_implementers:
            rN["stage_4_5_impl_start"] = rN_implementers[0]["timestamp"]
            rN_impl_ids = {s.get("tool_use_id") for s in rN_implementers}
            rN_impl_completions = [c for c in implementer_completions if c.get("tool_use_id") in rN_impl_ids]
            if rN_impl_completions:
                rN["stage_4_5_impl_end"] = max(c["timestamp"] for c in rN_impl_completions)

        result[f"round_{round_num}"] = rN

    return result


# ---------------------------------------------------------------------------
# Subagent Discovery
# ---------------------------------------------------------------------------

def _discover_subagents(session_dir: Path, session_id: str) -> List[Dict[str, Any]]:
    """Discover and summarize subagent JSONL files."""
    subagents_dir = session_dir / session_id / "subagents"
    if not subagents_dir.exists():
        return []

    result = []
    for jsonl_path in sorted(subagents_dir.glob("agent-*.jsonl")):
        agent_id = jsonl_path.stem.replace("agent-", "")
        meta_path = subagents_dir / f"agent-{agent_id}.meta.json"
        agent_type = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                agent_type = meta.get("agentType")
            except (json.JSONDecodeError, OSError):
                pass

        # Read first and last lines
        first_ts = None
        last_ts = None
        msg_count = 0
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = obj.get("timestamp")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                    msg_count += 1
        except OSError:
            pass

        result.append({
            "agent_id": agent_id,
            "agent_type": agent_type,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "message_count": msg_count,
        })

    return result


# ---------------------------------------------------------------------------
# Cost Summary
# ---------------------------------------------------------------------------

def _build_agent_costs(
    agent_spawns: Dict[str, Dict[str, Any]],
    task_notifications: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build agent_costs array and cost_summary."""
    agent_costs = []

    for tool_use_id, spawn in agent_spawns.items():
        notif_key = f"by_tool_use_id:{tool_use_id}"
        notif = task_notifications.get(notif_key)

        entry: Dict[str, Any] = {
            "name": spawn.get("name"),
            "display_name": spawn.get("display_name"),
            "role": spawn.get("role", "unknown"),
            "subagent_type": spawn.get("subagent_type"),
            "description": spawn.get("description"),
            "team_name": spawn.get("team_name") or spawn.get("cohort_name"),
            "cohort_name": spawn.get("cohort_name") or spawn.get("team_name"),
            "spawn_timestamp": spawn.get("spawn_timestamp"),
            "tool_use_id": tool_use_id,
        }

        if notif:
            entry["total_tokens"] = notif.get("total_tokens")
            entry["tool_uses"] = notif.get("tool_uses")
            entry["duration_ms"] = notif.get("duration_ms")
            entry["status"] = notif.get("status")
            entry["completion_timestamp"] = notif.get("completion_timestamp")
        else:
            entry["total_tokens"] = None
            entry["tool_uses"] = None
            entry["duration_ms"] = None
            entry["status"] = "killed_or_no_notification"
            entry["completion_timestamp"] = None

        agent_costs.append(entry)

    # Build cost_summary
    by_role: Dict[str, Any] = {}
    total_tokens = 0
    total_duration_ms = 0
    total_invocations = 0

    for entry in agent_costs:
        role = entry.get("role", "unknown")
        tokens = entry.get("total_tokens") or 0
        duration = entry.get("duration_ms") or 0

        total_tokens += tokens
        total_duration_ms += duration
        total_invocations += 1

        if role not in by_role:
            by_role[role] = {"count": 0, "total_tokens": 0, "total_duration_ms": 0}
        by_role[role]["count"] += 1
        by_role[role]["total_tokens"] += tokens
        by_role[role]["total_duration_ms"] += duration

    cost_summary = {
        "total_agent_invocations": total_invocations,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms,
        "by_role": by_role,
    }

    return agent_costs, cost_summary


# ---------------------------------------------------------------------------
# Team Lifecycle Summary
# ---------------------------------------------------------------------------

def _build_team_lifecycle(team_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a structured team lifecycle from raw events."""
    teams: Dict[str, Dict[str, Any]] = {}
    result = []

    for event in team_events:
        action = event["action"]
        name = event.get("name")
        ts = event["timestamp"]

        if action == "create":
            teams[name or "_unnamed"] = {
                "name": name,
                "create_timestamp": ts,
                "description": event.get("description"),
                "delete_timestamp": None,
            }
        elif action == "delete":
            key = name or "_unnamed"
            if key in teams:
                teams[key]["delete_timestamp"] = ts
                entry = dict(teams[key])
                if entry.get("create_timestamp") and entry.get("delete_timestamp"):
                    c = _parse_iso(entry["create_timestamp"])
                    d = _parse_iso(entry["delete_timestamp"])
                    if c and d:
                        entry["duration_seconds"] = round((d - c).total_seconds(), 1)
                result.append(entry)
            else:
                result.append({
                    "name": name,
                    "create_timestamp": None,
                    "delete_timestamp": ts,
                    "duration_seconds": None,
                })

    # Add any teams without delete
    for key, team in teams.items():
        if team.get("delete_timestamp") is None:
            result.append(team)

    return result


# ---------------------------------------------------------------------------
# Current Codex SQLite + rollout parser
# ---------------------------------------------------------------------------

def _timestamp_sort_key(value: Optional[str]) -> float:
    parsed = _parse_iso(value)
    return parsed.timestamp() if parsed is not None else float("-inf")


def _payload_time(payload: Dict[str, Any], field: str, fallback: Optional[str]) -> Optional[str]:
    value = payload.get(field)
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric < 10_000_000_000:
            numeric *= 1000
        return epoch_ms_to_iso(numeric)
    return fallback


def _nested_tool_names(source: Any) -> List[str]:
    if not isinstance(source, str):
        return []
    return re.findall(r"\btools\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)


def _parse_current_session(
    session_id: str,
    *,
    codex_home: Optional[Path] = None,
    state_db: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse a current Codex thread tree resolved from ``state_5.sqlite``."""
    del artifact_dir  # Reserved for future state-backed role enrichment.
    tree = resolve_thread_tree(
        session_id,
        codex_home=codex_home,
        state_db=state_db,
    )
    threads = tree["threads"]
    missing_rollouts = [
        str(thread.get("rollout_path"))
        for thread in threads
        if not thread.get("rollout_exists")
    ]
    if missing_rollouts:
        raise CodexRolloutMissing(
            "Codex thread graph references missing rollout file(s): "
            + ", ".join(missing_rollouts)
        )
    by_id = {str(thread["id"]): thread for thread in threads}

    scans: Dict[str, Dict[str, Any]] = {
        thread_id: {
            "task_starts": [],
            "task_completions": [],
            "task_aborts": [],
            "tool_calls": 0,
            "tool_names": Counter(),
            "nested_tool_names": Counter(),
            "message_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
        for thread_id in by_id
    }
    spawn_calls: Dict[Tuple[str, str], Dict[str, Any]] = {}
    activity_by_child: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    lifecycle_events: List[Dict[str, Any]] = []
    all_timestamps: List[str] = []

    for thread in threads:
        owner_id = str(thread["id"])
        scan = scans[owner_id]
        for obj in iter_owned_records(thread):
            timestamp = obj.get("timestamp")
            if isinstance(timestamp, str):
                all_timestamps.append(timestamp)
                if scan["first_timestamp"] is None:
                    scan["first_timestamp"] = timestamp
                scan["last_timestamp"] = timestamp

            record_type = obj.get("type")
            payload = obj.get("payload") or {}

            if record_type == "response_item" and payload.get("type") in {
                "message",
                "agent_message",
            }:
                scan["message_count"] += 1

            if record_type == "response_item" and payload.get("type") in {
                "function_call",
                "custom_tool_call",
            }:
                tool_name = str(payload.get("name") or "unknown")
                scan["tool_calls"] += 1
                scan["tool_names"][tool_name] += 1
                if payload.get("type") == "custom_tool_call" and tool_name == "exec":
                    scan["nested_tool_names"].update(
                        _nested_tool_names(payload.get("input"))
                    )
                if payload.get("type") == "function_call" and tool_name in {
                    "Agent",
                    "spawn_agent",
                    "functions.spawn_agent",
                }:
                    call_id = payload.get("call_id")
                    if isinstance(call_id, str):
                        spawn_calls[(owner_id, call_id)] = parse_arguments(
                            payload.get("arguments")
                        )

            if record_type != "event_msg":
                continue
            event_type = payload.get("type")
            if event_type == "task_started":
                started_at = _payload_time(payload, "started_at", timestamp)
                event = {
                    "event": "task_started",
                    "thread_id": owner_id,
                    "turn_id": payload.get("turn_id"),
                    "timestamp": started_at,
                }
                scan["task_starts"].append(event)
                lifecycle_events.append(event)
            elif event_type == "task_complete":
                completed_at = _payload_time(payload, "completed_at", timestamp)
                event = {
                    "event": "task_complete",
                    "thread_id": owner_id,
                    "turn_id": payload.get("turn_id"),
                    "timestamp": completed_at,
                    "duration_ms": payload.get("duration_ms"),
                    "time_to_first_token_ms": payload.get("time_to_first_token_ms"),
                }
                scan["task_completions"].append(event)
                lifecycle_events.append(event)
            elif event_type == "turn_aborted":
                completed_at = _payload_time(payload, "completed_at", timestamp)
                event = {
                    "event": "task_aborted",
                    "thread_id": owner_id,
                    "turn_id": payload.get("turn_id"),
                    "timestamp": completed_at,
                    "reason": payload.get("reason"),
                    "duration_ms": payload.get("duration_ms"),
                }
                scan["task_aborts"].append(event)
                lifecycle_events.append(event)
            elif event_type == "sub_agent_activity":
                child_id = payload.get("agent_thread_id")
                occurred_at = _payload_time(payload, "occurred_at_ms", timestamp)
                event = {
                    "event": "sub_agent_activity",
                    "parent_thread_id": owner_id,
                    "agent_thread_id": child_id,
                    "agent_path": payload.get("agent_path"),
                    "kind": payload.get("kind"),
                    "event_id": payload.get("event_id"),
                    "timestamp": occurred_at,
                }
                lifecycle_events.append(event)
                child = by_id.get(str(child_id)) if child_id is not None else None
                if child is not None and child.get("parent_thread_id") == owner_id:
                    activity_by_child[str(child_id)].append(event)

    agent_costs: List[Dict[str, Any]] = []
    subagents: List[Dict[str, Any]] = []
    stage_events: List[Dict[str, Any]] = []
    role_summary: Dict[str, Dict[str, int]] = {}

    for thread in threads:
        if int(thread.get("depth") or 0) == 0:
            continue
        thread_id = str(thread["id"])
        parent_id = str(thread.get("parent_thread_id") or "")
        scan = scans[thread_id]
        activities = sorted(
            activity_by_child.get(thread_id, []),
            key=lambda event: _timestamp_sort_key(event.get("timestamp")),
        )
        started = next(
            (event for event in activities if event.get("kind") == "started"),
            None,
        )
        interrupted = [
            event for event in activities if event.get("kind") == "interrupted"
        ]
        spawn_event_id = started.get("event_id") if started else None
        spawn_args = (
            spawn_calls.get((parent_id, str(spawn_event_id)), {})
            if spawn_event_id
            else {}
        )

        name = (
            spawn_args.get("task_name")
            or spawn_args.get("name")
            or _codex_agent_name(thread)
        )
        explicit_role = thread.get("agent_role") or spawn_args.get("agent_type")
        role = _infer_role(name, explicit_role, None)
        spawn_timestamp = (
            started.get("timestamp")
            if started
            else epoch_ms_to_iso(thread.get("created_at_ms"))
        )
        completions = sorted(
            scan["task_completions"],
            key=lambda event: _timestamp_sort_key(event.get("timestamp")),
        )
        starts = sorted(
            scan["task_starts"],
            key=lambda event: _timestamp_sort_key(event.get("timestamp")),
        )
        last_completion = completions[-1] if completions else None
        aborts = sorted(
            scan["task_aborts"],
            key=lambda event: _timestamp_sort_key(event.get("timestamp")),
        )
        last_abort = aborts[-1] if aborts else None
        last_start = starts[-1] if starts else None
        last_interrupt = interrupted[-1] if interrupted else None

        if thread.get("edge_status") == "closed":
            status = "closed"
        elif (
            last_completion
            and (
                last_start is None
                or _timestamp_sort_key(last_completion.get("timestamp"))
                >= _timestamp_sort_key(last_start.get("timestamp"))
            )
            and (
                last_abort is None
                or _timestamp_sort_key(last_completion.get("timestamp"))
                >= _timestamp_sort_key(last_abort.get("timestamp"))
            )
        ):
            status = "completed"
        elif (
            (last_abort or last_interrupt)
            and (
                last_start is None
                or max(
                    _timestamp_sort_key(last_abort.get("timestamp")) if last_abort else float("-inf"),
                    _timestamp_sort_key(last_interrupt.get("timestamp")) if last_interrupt else float("-inf"),
                )
                >= _timestamp_sort_key(last_start.get("timestamp"))
            )
        ):
            status = "interrupted"
        elif starts:
            status = "running"
        else:
            status = str(thread.get("edge_status") or "unknown")

        duration_ms = sum(
            int(event["duration_ms"])
            for event in completions + aborts
            if isinstance(event.get("duration_ms"), (int, float))
        )
        terminal_events = sorted(
            completions + aborts,
            key=lambda event: _timestamp_sort_key(event.get("timestamp")),
        )
        last_terminal = terminal_events[-1] if terminal_events else None
        total_tokens = int(thread.get("tokens_used") or 0)
        entry: Dict[str, Any] = {
            "agent_id": thread_id,
            "parent_thread_id": thread.get("parent_thread_id"),
            "agent_path": thread.get("agent_path"),
            "name": name,
            "display_name": thread.get("agent_nickname") or name,
            "role": role,
            "subagent_type": explicit_role,
            "description": None,
            "team_name": None,
            "cohort_name": None,
            "spawn_timestamp": spawn_timestamp,
            "tool_use_id": spawn_event_id,
            "total_tokens": total_tokens,
            "tool_uses": int(scan["tool_calls"]),
            "duration_ms": duration_ms,
            "status": status,
            "edge_status": thread.get("edge_status"),
            "completion_timestamp": (
                last_terminal.get("timestamp") if last_terminal else None
            ),
            "rollout_path": thread.get("rollout_path"),
            "depth": thread.get("depth"),
            "model": thread.get("model"),
        }
        agent_costs.append(entry)
        subagents.append(
            {
                "agent_id": thread_id,
                "parent_thread_id": thread.get("parent_thread_id"),
                "agent_path": thread.get("agent_path"),
                "agent_type": explicit_role,
                "role": role,
                "first_timestamp": scan["first_timestamp"] or spawn_timestamp,
                "last_timestamp": scan["last_timestamp"],
                "message_count": int(scan["message_count"]),
                "tool_call_count": int(scan["tool_calls"]),
                "tokens_used": total_tokens,
                "status": status,
                "depth": thread.get("depth"),
                "rollout_path": thread.get("rollout_path"),
            }
        )
        stage_events.append(
            {
                "event": "agent_spawn",
                "timestamp": spawn_timestamp,
                "tool_use_id": spawn_event_id or thread_id,
                "role": role,
                "name": name,
                "subagent_type": explicit_role,
                "team_name": None,
                "cohort_name": None,
            }
        )
        if completions:
            stage_events.append(
                {
                    "event": "task_complete",
                    "timestamp": completions[0].get("timestamp"),
                    "task_id": completions[0].get("turn_id"),
                    "tool_use_id": spawn_event_id or thread_id,
                    "status": "completed",
                    "total_tokens": total_tokens,
                    "duration_ms": completions[0].get("duration_ms"),
                }
            )

        by_role = role_summary.setdefault(
            role,
            {"count": 0, "total_tokens": 0, "total_duration_ms": 0},
        )
        by_role["count"] += 1
        by_role["total_tokens"] += total_tokens
        by_role["total_duration_ms"] += duration_ms

    stage_events.sort(key=lambda event: _timestamp_sort_key(event.get("timestamp")))
    stage_timestamps = _infer_stage_timestamps(stage_events, {}, {}, [])
    lifecycle_events.sort(key=lambda event: _timestamp_sort_key(event.get("timestamp")))

    root = threads[0]
    root_id = str(root["id"])
    agent_tokens = sum(int(entry.get("total_tokens") or 0) for entry in agent_costs)
    agent_duration = sum(int(entry.get("duration_ms") or 0) for entry in agent_costs)
    root_tokens = int(root.get("tokens_used") or 0)
    per_thread_tools = {
        thread_id: {
            "total_calls": int(scan["tool_calls"]),
            "by_tool": dict(sorted(scan["tool_names"].items())),
            "by_nested_tool": dict(sorted(scan["nested_tool_names"].items())),
        }
        for thread_id, scan in scans.items()
    }
    aggregate_tools = Counter()
    aggregate_nested_tools = Counter()
    for scan in scans.values():
        aggregate_tools.update(scan["tool_names"])
        aggregate_nested_tools.update(scan["nested_tool_names"])

    session_start = epoch_ms_to_iso(root.get("created_at_ms"))
    session_end = max(all_timestamps, key=_timestamp_sort_key) if all_timestamps else (
        epoch_ms_to_iso(root.get("updated_at_ms"))
    )
    rollout_files = [
        {
            "thread_id": thread.get("id"),
            "parent_thread_id": thread.get("parent_thread_id"),
            "depth": thread.get("depth"),
            "agent_path": thread.get("agent_path"),
            "agent_role": thread.get("agent_role"),
            "edge_status": thread.get("edge_status"),
            "created_at_ms": thread.get("created_at_ms"),
            "updated_at_ms": thread.get("updated_at_ms"),
            "path": thread.get("rollout_path"),
            "exists": thread.get("rollout_exists"),
        }
        for thread in threads
    ]

    return {
        "session_id": session_id,
        "source_format": CURRENT_SOURCE_FORMAT,
        "state_db": tree["state_db"],
        "codex_home": tree["codex_home"],
        "root_thread": {
            "thread_id": root_id,
            "rollout_path": root.get("rollout_path"),
            "tokens_used": root_tokens,
            "cli_version": root.get("cli_version"),
            "model": root.get("model"),
        },
        "rollout_files": rollout_files,
        "session_start": session_start,
        "session_end": session_end,
        "stage_timestamps": stage_timestamps,
        "agent_costs": agent_costs,
        "team_lifecycle": [],
        "subagents": subagents,
        "lifecycle_events": lifecycle_events,
        "tool_summary": {
            "total_calls": sum(scan["tool_calls"] for scan in scans.values()),
            "by_tool": dict(sorted(aggregate_tools.items())),
            "by_nested_tool": dict(sorted(aggregate_nested_tools.items())),
            "per_thread": per_thread_tools,
        },
        "cost_summary": {
            "total_agent_invocations": len(agent_costs),
            "total_tokens": agent_tokens,
            "total_duration_ms": agent_duration,
            "by_role": role_summary,
            "root_tokens": root_tokens,
            "campaign_total_tokens": root_tokens + agent_tokens,
        },
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

def parse_session(
    session_id: str,
    session_dir: Path,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse a session and return structured data."""

    # Find the JSONL file
    # session_dir may be the project dir itself or a projects root, in which
    # case we search one level down.
    session_path = session_dir / f"{session_id}.jsonl"
    if not session_path.exists():
        matches = sorted(session_dir.glob(f"*/{session_id}.jsonl"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Session JSONL not found: {session_path}"
                + (f" ({len(matches)} candidates under {session_dir})" if matches else "")
            )
        session_path = matches[0]
        session_dir = session_path.parent

    parser = SessionParser(session_path)
    parser.parse()

    # Build agent costs
    agent_costs, cost_summary = _build_agent_costs(parser.agent_spawns, parser.task_notifications)

    # Build stage timestamps
    stage_timestamps = _infer_stage_timestamps(
        parser.events,
        parser.agent_spawns,
        parser.task_notifications,
        parser.team_events,
    )

    # Build team lifecycle
    team_lifecycle = _build_team_lifecycle(parser.team_events)

    # Discover subagents
    subagents = _discover_subagents(session_dir, session_id)

    return {
        "session_id": session_id,
        "source_format": "legacy_projects_jsonl_v1",
        "session_jsonl": str(session_path.resolve()),
        "rollout_files": [
            {
                "thread_id": session_id,
                "parent_thread_id": None,
                "depth": 0,
                "agent_path": None,
                "agent_role": None,
                "edge_status": None,
                "path": str(session_path.resolve()),
                "exists": True,
            }
        ],
        "session_start": parser.session_start,
        "session_end": parser.session_end,
        "stage_timestamps": stage_timestamps,
        "agent_costs": agent_costs,
        "team_lifecycle": team_lifecycle,
        "subagents": subagents,
        "cost_summary": cost_summary,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_codex_session(
    session_id: str,
    *,
    codex_home: Optional[Path] = None,
    state_db: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Public current-Codex parser; legacy ``parse_session`` stays compatible."""
    return _parse_current_session(
        session_id,
        codex_home=codex_home,
        state_db=state_db,
        artifact_dir=artifact_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session-id", required=True, help="Session UUID")
    p.add_argument(
        "--session-dir",
        default=None,
        help=(
            "Legacy-only directory containing <session-id>.jsonl. When set, "
            "skip current Codex SQLite discovery."
        ),
    )
    p.add_argument(
        "--codex-home",
        default=None,
        help="Current Codex home (default: $CODEX_HOME or ~/.codex)",
    )
    p.add_argument(
        "--state-db",
        default=None,
        help="Current Codex state_5.sqlite path (opened read-only)",
    )
    p.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional: artifact directory to cross-reference state.json for role mapping",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )

    args = p.parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else None
    if args.session_dir:
        session_dir = Path(args.session_dir).expanduser().resolve()
        if not session_dir.exists():
            print(f"ERROR: Session directory does not exist: {session_dir}", file=sys.stderr)
            return 1
        try:
            data = parse_session(args.session_id, session_dir, artifact_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        codex_home = (
            Path(args.codex_home).expanduser().resolve()
            if args.codex_home
            else default_codex_home()
        )
        state_db = Path(args.state_db).expanduser().resolve() if args.state_db else None
        try:
            data = parse_codex_session(
                args.session_id,
                codex_home=codex_home,
                state_db=state_db,
                artifact_dir=artifact_dir,
            )
        except CodexRolloutMissing as current_error:
            print(f"ERROR: {current_error}", file=sys.stderr)
            return 1
        except (FileNotFoundError, RuntimeError, OSError) as current_error:
            # Preserve the historical no-extra-flags CLI for installations that
            # have no current state database but still retain project JSONLs.
            legacy_dir = Path("~/.codex/projects").expanduser().resolve()
            try:
                data = parse_session(args.session_id, legacy_dir, artifact_dir)
            except FileNotFoundError:
                print(f"ERROR: {current_error}", file=sys.stderr)
                print(
                    f"Legacy fallback also not found: {legacy_dir / (args.session_id + '.jsonl')}",
                    file=sys.stderr,
                )
                return 1

    out_path = Path(args.output)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
