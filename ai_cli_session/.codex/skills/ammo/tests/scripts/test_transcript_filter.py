#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for transcript_filter.py — the AMMO transcript monitor's filter script.

Tests cover:
- Basic filtering (noise removal, content extraction)
- Incremental reading (--start-line)
- Truncated line handling (break at write frontier)
- Subagent discovery (--include-subagents)
- Pending subagents state persistence (spawn window gap fix)
- Atomic state writes (write_state_atomic)
- Reused agent name offset reset (path mismatch detection)
- Combined output format ([sub:name] prefix)
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

# We'll import from the module once it exists
SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts"
)
FILTER_SCRIPT = os.path.join(SCRIPT_DIR, "transcript_filter.py")


# ---------------------------------------------------------------------------
# Test fixtures — helper to create .jsonl transcript files
# ---------------------------------------------------------------------------

def make_record(rtype, **kwargs):
    """Create a minimal .jsonl transcript record."""
    record = {"type": rtype, "timestamp": "2026-03-26T14:32:01.000Z"}
    record.update(kwargs)
    return record


def make_assistant_text(text, agent_name=None):
    """Assistant record with a text block."""
    r = make_record(
        "assistant",
        message={"role": "assistant", "content": [
            {"type": "text", "text": text}
        ]},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_assistant_thinking(thinking, agent_name=None):
    """Assistant record with a thinking block."""
    r = make_record(
        "assistant",
        message={"role": "assistant", "content": [
            {"type": "thinking", "thinking": thinking}
        ]},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_assistant_bash(command, description="", agent_name=None):
    """Assistant record with a Bash tool_use block."""
    r = make_record(
        "assistant",
        message={"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": command, "description": description}}
        ]},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_assistant_agent_spawn(spawned_name, description="spawn", agent_name=None, background=False):
    """Assistant record with an Codex subagent_use block (spawning a subagent)."""
    r = make_record(
        "assistant",
        message={"role": "assistant", "content": [
            {"type": "tool_use", "name": "Agent",
             "input": {"name": spawned_name, "description": description,
                       "prompt": f"Do work as {spawned_name}",
                       "subagent_type": "general",
                       "run_in_background": background}}
        ]},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_user_bash_result(stdout="", stderr="", agent_name=None):
    """User record with a Bash tool result."""
    r = make_record(
        "user",
        toolUseResult={"stdout": stdout, "stderr": stderr},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_noise(noise_type="progress", agent_name=None):
    """A noise record that should be filtered out."""
    r = make_record(noise_type)
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_system_record(text="system prompt", agent_name=None):
    """System record."""
    r = make_record(
        "system",
        message={"role": "system", "content": [
            {"type": "text", "text": text}
        ]},
    )
    if agent_name:
        r["agentName"] = agent_name
    return r


def make_codex_response_item(item, **kwargs):
    """Codex rollout-style wrapper around one response item."""
    record = {"type": "response_item", "timestamp": "2026-03-26T14:32:01.000Z", "item": item}
    record.update(kwargs)
    return record


def write_jsonl(path, records):
    """Write a list of dicts as .jsonl."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def run_filter(transcript_path, start_line=0, state_file=None,
               include_subagents=False, projects_dir=None,
               max_content_len=None):
    """Run transcript_filter.py as a subprocess and return (stdout, stderr, returncode)."""
    cmd = [sys.executable, FILTER_SCRIPT, transcript_path,
           "--start-line", str(start_line)]
    if state_file:
        cmd += ["--state-file", state_file]
    if include_subagents:
        cmd.append("--include-subagents")
    if projects_dir:
        cmd += ["--projects-dir", projects_dir]
    if max_content_len is not None:
        cmd += ["--max-content-len", str(max_content_len)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout, result.stderr, result.returncode


def parse_summary(stdout):
    """Extract key-value pairs from the --- FILTER SUMMARY --- section."""
    summary = {}
    in_summary = False
    for line in stdout.splitlines():
        if "--- FILTER SUMMARY ---" in line:
            in_summary = True
            continue
        if in_summary and ": " in line:
            key, _, val = line.partition(": ")
            summary[key.strip()] = val.strip()
    return summary


# ---------------------------------------------------------------------------
# Module-level imports for unit tests (imported after script exists)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _add_script_to_path():
    """Add script directory to sys.path so we can import transcript_filter."""
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    yield
    # Don't remove — other tests may need it


# ===========================================================================
# 1. Basic filtering — noise removal and content extraction
# ===========================================================================

class TestBasicFiltering:
    """Test that noise is removed and content records are extracted."""

    def test_noise_records_filtered(self, tmp_path):
        """progress, file-history-snapshot, queue-operation are dropped."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_noise("progress"),
            make_noise("file-history-snapshot"),
            make_noise("queue-operation"),
            make_assistant_text("Hello world", agent_name="champ"),
        ])

        stdout, stderr, rc = run_filter(str(transcript))
        assert rc == 0
        summary = parse_summary(stdout)
        assert summary["NOISE_SKIPPED"] == "3"
        assert summary["RECORDS_OUTPUT"] == "1"
        assert "Hello world" in stdout

    def test_assistant_text_extracted(self, tmp_path):
        """Assistant text blocks appear in output."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_text("This is analysis", agent_name="champ"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "ASSISTANT" in stdout
        assert "This is analysis" in stdout

    def test_assistant_thinking_extracted(self, tmp_path):
        """Assistant thinking blocks appear in output."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_thinking("Let me think about this", agent_name="champ"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "THINKING" in stdout
        assert "Let me think about this" in stdout

    def test_assistant_bash_extracted(self, tmp_path):
        """Bash tool_use blocks show the command."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_bash("ncu --set full", description="Profile kernel"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "BASH" in stdout
        assert "ncu --set full" in stdout

    def test_user_bash_result_extracted(self, tmp_path):
        """User records with bash stdout are extracted."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_user_bash_result(stdout="Memory BW: 423 GB/s"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "BASH_RESULT" in stdout
        assert "Memory BW: 423 GB/s" in stdout

    def test_system_record_extracted(self, tmp_path):
        """System records appear with truncated content."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_system_record("System prompt content here"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "SYSTEM" in stdout
        assert "System prompt" in stdout


# ===========================================================================
# 1b. Codex rollout JSONL adaptation
# ===========================================================================

class TestCodexRolloutRecords:
    """Codex rollout records normalize to the same digest as Claude records."""

    def test_response_item_assistant_output_text_extracted(self, tmp_path):
        """Codex response_item/message/output_text is emitted as assistant text."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            make_codex_response_item(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Codex rollout analysis"}],
                },
                session_id="codex-session-123",
            )
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "ASSISTANT" in stdout
        assert "Codex rollout analysis" in stdout
        assert parse_summary(stdout)["SESSION"] == "codex-session-123"

    def test_top_level_codex_message_extracted(self, tmp_path):
        """Top-level Codex message records do not need a Claude message wrapper."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            {
                "type": "message",
                "role": "assistant",
                "timestamp": "2026-03-26T14:32:01.000Z",
                "content": [{"type": "output_text", "text": "Top level Codex text"}],
            }
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "ASSISTANT" in stdout
        assert "Top level Codex text" in stdout

    def test_codex_function_call_spawn_discovers_subagent(self, tmp_path):
        """Codex function_call spawn_agent records feed subagent discovery."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        champion = projects_dir / "rollout.jsonl"
        write_jsonl(champion, [
            make_codex_response_item(
                {
                    "type": "function_call",
                    "call_id": "call-spawn-1",
                    "name": "functions.spawn_agent",
                    "arguments": json.dumps({
                        "name": "monitor-1",
                        "agent_type": "ammo-transcript-monitor",
                        "message": "Watch the champion transcript",
                    }),
                },
                agent_name="champion-1",
            )
        ])

        subagent = projects_dir / "monitor-session.jsonl"
        write_jsonl(subagent, [
            make_assistant_text("monitor observed drift", agent_name="monitor-1"),
        ])

        stdout, _, rc = run_filter(
            str(champion), include_subagents=True, projects_dir=str(projects_dir))

        assert rc == 0
        assert "SPAWN_AGENT" in stdout
        assert "[sub:monitor-1]" in stdout
        assert "monitor observed drift" in stdout

    def test_codex_spawn_discovers_codex_shaped_subagent_file(self, tmp_path):
        """Subagent discovery scans Codex metadata aliases in candidate files."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        champion = projects_dir / "champion-rollout.jsonl"
        write_jsonl(champion, [
            make_codex_response_item({
                "type": "function_call",
                "name": "spawn_agent",
                "arguments": {"name": "monitor-2", "agent_type": "ammo-transcript-monitor"},
            })
        ])

        subagent = projects_dir / "monitor-rollout.jsonl"
        write_jsonl(subagent, [
            make_codex_response_item(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "codex-shaped subagent"}],
                },
                agent_name="monitor-2",
            )
        ])

        stdout, _, rc = run_filter(
            str(champion), include_subagents=True, projects_dir=str(projects_dir))

        assert rc == 0
        assert "[sub:monitor-2]" in stdout
        assert "codex-shaped subagent" in stdout

    def test_codex_function_output_is_plain_result_not_error(self, tmp_path):
        """Plain Codex function_call_output strings are stdout-style results."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            make_codex_response_item(
                {
                    "type": "function_call_output",
                    "call_id": "call-shell-1",
                    "output": "pytest passed",
                }
            )
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "BASH_RESULT" in stdout
        assert "pytest passed" in stdout
        assert "RESULT [ERROR]" not in stdout

    def test_codex_items_list_normalized_in_order(self, tmp_path):
        """A rollout line with multiple items preserves item order."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            {
                "type": "response",
                "timestamp": "2026-03-26T14:32:01.000Z",
                "items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "first item"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "second item"}],
                    },
                ],
            }
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert stdout.find("first item") < stdout.find("second item")

    def test_unknown_payload_wrapper_ignored(self, tmp_path):
        """Non-conversation payload wrappers do not become digest output."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            {
                "type": "metrics",
                "timestamp": "2026-03-26T14:32:01.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "do not print me"}],
                },
            }
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "do not print me" not in stdout
        assert parse_summary(stdout)["RECORDS_OUTPUT"] == "0"

    def test_malformed_function_arguments_do_not_crash(self, tmp_path):
        """Malformed function_call arguments are treated as an empty input dict."""
        transcript = tmp_path / "rollout.jsonl"
        write_jsonl(transcript, [
            make_codex_response_item(
                {
                    "type": "function_call",
                    "name": "functions.spawn_agent",
                    "arguments": "{not json",
                }
            )
        ])

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "SPAWN_AGENT" in stdout

    def test_agent_name_in_summary(self, tmp_path):
        """AGENT field in summary comes from agentName in records."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_text("hi", agent_name="champion-1"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        summary = parse_summary(stdout)
        assert summary["AGENT"] == "champion-1"

    def test_empty_transcript(self, tmp_path):
        """Empty file produces summary with 0 records."""
        transcript = tmp_path / "champion.jsonl"
        transcript.write_text("")

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        summary = parse_summary(stdout)
        assert summary["RECORDS_OUTPUT"] == "0"

    def test_content_truncation_when_explicitly_set(self, tmp_path):
        """Content exceeding explicit --max-content-len is truncated."""
        big_text = "x" * 200
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_thinking(big_text),
        ])

        stdout, _, _ = run_filter(str(transcript), max_content_len=50)
        assert "[TRUNCATED" in stdout

    def test_no_truncation_by_default(self, tmp_path):
        """Default (no --max-content-len) does not truncate."""
        big_text = "x" * 100_000
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_thinking(big_text),
        ])

        stdout, _, _ = run_filter(str(transcript))
        assert "[TRUNCATED" not in stdout
        assert big_text in stdout


# ===========================================================================
# 2. Incremental reading (--start-line)
# ===========================================================================

class TestIncrementalReading:
    """Test --start-line skips already-processed lines."""

    def test_start_line_skips_earlier_records(self, tmp_path):
        """Records before --start-line are not in output."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_assistant_text("line zero"),      # line 0
            make_assistant_text("line one"),       # line 1
            make_assistant_text("line two"),       # line 2
        ])

        stdout, _, _ = run_filter(str(transcript), start_line=2)
        assert "line zero" not in stdout
        assert "line one" not in stdout
        assert "line two" in stdout

    def test_last_line_processed_advances(self, tmp_path):
        """LAST_LINE_PROCESSED in summary reflects lines read."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [
            make_noise("progress"),
            make_assistant_text("content"),
            make_noise("progress"),
        ])

        stdout, _, _ = run_filter(str(transcript))
        summary = parse_summary(stdout)
        assert int(summary["LAST_LINE_PROCESSED"]) == 3

    def test_start_line_beyond_file(self, tmp_path):
        """start_line past end of file produces 0 records."""
        transcript = tmp_path / "champion.jsonl"
        write_jsonl(transcript, [make_assistant_text("only record")])

        stdout, _, _ = run_filter(str(transcript), start_line=999)
        summary = parse_summary(stdout)
        assert summary["RECORDS_OUTPUT"] == "0"


# ===========================================================================
# 3. Truncated line handling
# ===========================================================================

class TestTruncatedLines:
    """Test behavior at the write frontier (incomplete final line)."""

    def test_truncated_json_stops_processing(self, tmp_path):
        """Incomplete JSON at end of file stops, doesn't crash."""
        transcript = tmp_path / "champion.jsonl"
        records = [
            make_assistant_text("complete record"),
        ]
        with open(transcript, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            # Write a truncated line (no closing brace)
            f.write('{"type": "assistant", "message": {"role": "assis')

        stdout, _, rc = run_filter(str(transcript))
        assert rc == 0
        assert "complete record" in stdout
        summary = parse_summary(stdout)
        # LAST_LINE_PROCESSED should NOT advance past the truncated line
        assert int(summary["LAST_LINE_PROCESSED"]) == 1

    def test_retry_after_truncation(self, tmp_path):
        """Next poll with same start_line picks up the completed line."""
        transcript = tmp_path / "champion.jsonl"

        # First write: complete record + truncated line
        with open(transcript, "w") as f:
            f.write(json.dumps(make_assistant_text("first")) + "\n")
            f.write('{"type": "assistant", "truncated')

        stdout1, _, _ = run_filter(str(transcript))
        summary1 = parse_summary(stdout1)
        last_line = int(summary1["LAST_LINE_PROCESSED"])

        # Second write: complete the truncated line + add new record
        with open(transcript, "w") as f:
            f.write(json.dumps(make_assistant_text("first")) + "\n")
            f.write(json.dumps(make_assistant_text("second")) + "\n")
            f.write(json.dumps(make_assistant_text("third")) + "\n")

        stdout2, _, _ = run_filter(str(transcript), start_line=last_line)
        assert "second" in stdout2
        assert "third" in stdout2

    def test_large_line_skipped(self, tmp_path):
        """Lines >5MB are skipped."""
        transcript = tmp_path / "champion.jsonl"
        with open(transcript, "w") as f:
            f.write(json.dumps(make_assistant_text("before")) + "\n")
            # Write a >5MB line
            f.write("x" * 5_100_000 + "\n")
            f.write(json.dumps(make_assistant_text("after")) + "\n")

        stdout, _, _ = run_filter(str(transcript))
        assert "before" in stdout
        assert "after" in stdout


# ===========================================================================
# 4. Subagent discovery (--include-subagents)
# ===========================================================================

class TestSubagentDiscovery:
    """Test --include-subagents finds child transcripts."""

    def test_discovers_subagent_transcript(self, tmp_path):
        """When champion spawns an agent, its transcript is discovered and included."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        # Champion transcript with Agent spawn
        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
            make_assistant_text("champion continues", agent_name="champion-1"),
        ])

        # Subagent transcript
        subagent = projects_dir / "validator-session.jsonl"
        write_jsonl(subagent, [
            make_assistant_bash("python3 test.py", agent_name="validator-1"),
            make_user_bash_result(stdout="All tests passed", agent_name="validator-1"),
        ])

        stdout, _, rc = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir))

        assert rc == 0
        assert "[sub:validator-1]" in stdout
        assert "All tests passed" in stdout
        summary = parse_summary(stdout)
        assert "validator-1" in summary.get("SUBAGENTS_DISCOVERED", "")

    def test_no_subagents_without_flag(self, tmp_path):
        """Without --include-subagents, subagent transcripts are not processed."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
        ])

        subagent = projects_dir / "validator-session.jsonl"
        write_jsonl(subagent, [
            make_assistant_text("subagent work", agent_name="validator-1"),
        ])

        stdout, _, _ = run_filter(str(champion))
        assert "[sub:validator-1]" not in stdout

    def test_missing_subagent_transcript(self, tmp_path):
        """If subagent transcript doesn't exist yet, it becomes pending."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
        ])
        # No subagent transcript exists

        state_file = str(tmp_path / "state.txt")
        stdout, _, rc = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        assert rc == 0
        summary = parse_summary(stdout)
        assert "validator-1" in summary.get("SUBAGENTS_PENDING", "")

        # State file should have pending_subagents
        with open(state_file) as f:
            state = json.load(f)
        assert "validator-1" in state["pending_subagents"]

    def test_multiple_subagents(self, tmp_path):
        """Multiple spawned agents are all discovered."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
            make_assistant_agent_spawn("researcher-1", agent_name="champion-1"),
        ])

        for name in ["validator-1", "researcher-1"]:
            sub = projects_dir / f"{name}-session.jsonl"
            write_jsonl(sub, [
                make_assistant_text(f"work from {name}", agent_name=name),
            ])

        stdout, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir))

        assert "[sub:validator-1]" in stdout
        assert "[sub:researcher-1]" in stdout
        assert "work from validator-1" in stdout
        assert "work from researcher-1" in stdout

    def test_champion_transcript_not_included_as_subagent(self, tmp_path):
        """The champion's own transcript is not processed as a subagent."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        # Champion that spawns an agent with its own name (edge case)
        champion = projects_dir / "champ.jsonl"
        write_jsonl(champion, [
            make_assistant_text("champion work", agent_name="champion-1"),
            make_assistant_agent_spawn("champion-1", agent_name="champion-1"),
        ])

        stdout, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir))

        # Should not have [sub:champion-1] since the only matching file is the champion itself
        assert "[sub:champion-1]" not in stdout


# ===========================================================================
# 5. Pending subagents state persistence (spawn window gap fix)
# ===========================================================================

class TestPendingSubagents:
    """Test that pending_subagents survives across polls."""

    def test_pending_subagent_discovered_on_next_poll(self, tmp_path):
        """A pending subagent is discovered when its transcript appears."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
            make_assistant_text("more work", agent_name="champion-1"),
        ])

        state_file = str(tmp_path / "state.txt")

        # Poll 1: spawn detected, transcript doesn't exist yet
        stdout1, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        summary1 = parse_summary(stdout1)
        assert "validator-1" in summary1.get("SUBAGENTS_PENDING", "")

        # Now create the subagent transcript
        subagent = projects_dir / "validator-session.jsonl"
        write_jsonl(subagent, [
            make_assistant_text("validator work", agent_name="validator-1"),
        ])

        # Poll 2: start_line past the Agent spawn, but pending_subagents in state
        last_line = int(summary1["LAST_LINE_PROCESSED"])
        stdout2, _, _ = run_filter(
            str(champion), start_line=last_line,
            include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        assert "[sub:validator-1]" in stdout2
        assert "validator work" in stdout2

        # State file should now have the subagent, not pending
        with open(state_file) as f:
            state = json.load(f)
        assert "validator-1" in state["subagents"]
        assert "validator-1" not in state.get("pending_subagents", [])

    def test_pending_persists_across_multiple_polls(self, tmp_path):
        """Pending stays pending if transcript still doesn't exist."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("slow-agent", agent_name="champion-1"),
            make_assistant_text("work", agent_name="champion-1"),
        ])

        state_file = str(tmp_path / "state.txt")

        # Poll 1
        stdout1, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)
        last_line = int(parse_summary(stdout1)["LAST_LINE_PROCESSED"])

        # Poll 2 — still no subagent transcript
        stdout2, _, _ = run_filter(
            str(champion), start_line=last_line,
            include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        summary2 = parse_summary(stdout2)
        assert "slow-agent" in summary2.get("SUBAGENTS_PENDING", "")

        with open(state_file) as f:
            state = json.load(f)
        assert "slow-agent" in state["pending_subagents"]


# ===========================================================================
# 6. Atomic state writes
# ===========================================================================

class TestStateFileIO:
    """Test read_state and write_state_atomic."""

    def test_atomic_write_creates_file(self, tmp_path):
        """write_state_atomic creates a valid JSON state file."""
        import transcript_filter as tf

        state_file = str(tmp_path / "state.txt")
        tf.write_state_atomic(state_file, {
            "pending_subagents": ["a"],
            "subagents": {"b": {"path": "/x.jsonl", "last_line": 10}}
        })

        with open(state_file) as f:
            state = json.load(f)
        assert state["pending_subagents"] == ["a"]
        assert state["subagents"]["b"]["last_line"] == 10

    def test_atomic_write_no_temp_file_left(self, tmp_path):
        """After successful write, no .tmp file remains."""
        import transcript_filter as tf

        state_file = str(tmp_path / "state.txt")
        tf.write_state_atomic(state_file, {"pending_subagents": []})
        assert not os.path.exists(state_file + ".tmp")

    def test_read_state_empty_file(self, tmp_path):
        """read_state returns {} for empty file."""
        import transcript_filter as tf

        state_file = tmp_path / "state.txt"
        state_file.write_text("")
        assert tf.read_state(str(state_file)) == {}

    def test_read_state_old_format_integer(self, tmp_path):
        """read_state returns {} for old bare-integer format."""
        import transcript_filter as tf

        state_file = tmp_path / "state.txt"
        state_file.write_text("1234")
        assert tf.read_state(str(state_file)) == {}

    def test_read_state_corrupt_json(self, tmp_path):
        """read_state returns {} for corrupt JSON."""
        import transcript_filter as tf

        state_file = tmp_path / "state.txt"
        state_file.write_text('{"partial": ')
        assert tf.read_state(str(state_file)) == {}

    def test_read_state_nonexistent(self, tmp_path):
        """read_state returns {} for missing file."""
        import transcript_filter as tf

        assert tf.read_state(str(tmp_path / "nope.txt")) == {}

    def test_read_state_valid_json(self, tmp_path):
        """read_state parses valid JSON state file."""
        import transcript_filter as tf

        state_file = tmp_path / "state.txt"
        state = {"pending_subagents": ["x"], "subagents": {}}
        state_file.write_text(json.dumps(state))
        assert tf.read_state(str(state_file)) == state

    def test_write_state_none_path(self):
        """write_state_atomic with None path is a no-op."""
        import transcript_filter as tf
        tf.write_state_atomic(None, {})  # Should not raise


# ===========================================================================
# 7. Reused agent name offset reset
# ===========================================================================

class TestReusedAgentName:
    """Test that a reused agent name with a new transcript resets last_line."""

    def test_path_mismatch_resets_offset(self, tmp_path):
        """When discovered path differs from stored path, last_line resets to 0."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion-session.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
        ])

        # Old subagent transcript (3 records)
        old_sub = projects_dir / "old-validator.jsonl"
        write_jsonl(old_sub, [
            make_assistant_text("old work 1", agent_name="validator-1"),
            make_assistant_text("old work 2", agent_name="validator-1"),
            make_assistant_text("old work 3", agent_name="validator-1"),
        ])

        # New subagent transcript (1 record, newer mtime)
        new_sub = projects_dir / "new-validator.jsonl"
        write_jsonl(new_sub, [
            make_assistant_text("fresh start", agent_name="validator-1"),
        ])
        # Ensure new file has newer mtime
        import time
        time.sleep(0.05)
        os.utime(new_sub, None)

        # State file with old path and high last_line
        state_file = str(tmp_path / "state.txt")
        old_state = {
            "pending_subagents": [],
            "subagents": {
                "validator-1": {"path": str(old_sub), "last_line": 3}
            }
        }
        with open(state_file, "w") as f:
            json.dump(old_state, f)

        stdout, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        # Should see the new transcript's content (reset to line 0)
        assert "fresh start" in stdout

        # State should now point to the new path
        with open(state_file) as f:
            state = json.load(f)
        assert state["subagents"]["validator-1"]["path"] == str(new_sub)


# ===========================================================================
# 8. Combined output format
# ===========================================================================

class TestCombinedOutput:
    """Test [sub:name] prefix and ordering."""

    def test_subagent_lines_prefixed(self, tmp_path):
        """Subagent output lines have [sub:name] prefix."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("val-1", agent_name="champ"),
        ])

        sub = projects_dir / "val-1-session.jsonl"
        write_jsonl(sub, [
            make_assistant_text("sub output", agent_name="val-1"),
        ])

        stdout, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir))

        # Find lines with [sub:val-1] prefix
        sub_lines = [l for l in stdout.splitlines() if "[sub:val-1]" in l]
        assert len(sub_lines) > 0

    def test_champion_records_before_subagent(self, tmp_path):
        """Champion records appear before subagent records."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion.jsonl"
        write_jsonl(champion, [
            make_assistant_text("champion first", agent_name="champ"),
            make_assistant_agent_spawn("val-1", agent_name="champ"),
        ])

        sub = projects_dir / "val-1-session.jsonl"
        write_jsonl(sub, [
            make_assistant_text("sub second", agent_name="val-1"),
        ])

        stdout, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir))

        champ_pos = stdout.find("champion first")
        sub_pos = stdout.find("sub second")
        assert champ_pos < sub_pos, "Champion records should come before subagent records"

    def test_subagent_incremental_reading(self, tmp_path):
        """Subagent transcripts are read incrementally via state file."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        champion = projects_dir / "champion.jsonl"
        write_jsonl(champion, [
            make_assistant_agent_spawn("val-1", agent_name="champ"),
            make_assistant_text("more champ", agent_name="champ"),
        ])

        sub = projects_dir / "val-1-session.jsonl"
        write_jsonl(sub, [
            make_assistant_text("sub line 0", agent_name="val-1"),
            make_assistant_text("sub line 1", agent_name="val-1"),
        ])

        state_file = str(tmp_path / "state.txt")

        # Poll 1: read everything
        stdout1, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)
        assert "sub line 0" in stdout1
        assert "sub line 1" in stdout1

        # Add more subagent content
        with open(sub, "a") as f:
            f.write(json.dumps(make_assistant_text("sub line 2", agent_name="val-1")) + "\n")

        # Poll 2: only new subagent content
        last_line = int(parse_summary(stdout1)["LAST_LINE_PROCESSED"])
        stdout2, _, _ = run_filter(
            str(champion), start_line=last_line,
            include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        assert "sub line 0" not in stdout2  # Already processed
        assert "sub line 1" not in stdout2
        assert "sub line 2" in stdout2


# ===========================================================================
# 9. Integration: end-to-end multi-poll scenario
# ===========================================================================

class TestEndToEnd:
    """Full scenario: champion + subagent over multiple polls."""

    def test_multi_poll_lifecycle(self, tmp_path):
        """Simulate 3 polls: spawn, discover, incremental."""
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        state_file = str(tmp_path / "state.txt")

        # --- Poll 1: Champion starts, spawns validator ---
        champion = projects_dir / "champion.jsonl"
        write_jsonl(champion, [
            make_assistant_text("Starting optimization", agent_name="champion-1"),
            make_assistant_agent_spawn("validator-1", agent_name="champion-1"),
        ])

        stdout1, _, _ = run_filter(
            str(champion), include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)
        last_line_1 = int(parse_summary(stdout1)["LAST_LINE_PROCESSED"])

        assert "Starting optimization" in stdout1
        assert "validator-1" in parse_summary(stdout1).get("SUBAGENTS_PENDING", "")

        # --- Poll 2: Validator transcript appears ---
        # Add more champion content
        with open(champion, "a") as f:
            f.write(json.dumps(make_assistant_bash("ncu --set full", agent_name="champion-1")) + "\n")

        # Create validator transcript
        validator = projects_dir / "validator-session.jsonl"
        write_jsonl(validator, [
            make_assistant_bash("python3 test.py", agent_name="validator-1"),
            make_user_bash_result(stdout="PASS: 5/5", agent_name="validator-1"),
        ])

        stdout2, _, _ = run_filter(
            str(champion), start_line=last_line_1,
            include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)
        last_line_2 = int(parse_summary(stdout2)["LAST_LINE_PROCESSED"])

        assert "ncu --set full" in stdout2
        assert "[sub:validator-1]" in stdout2
        assert "PASS: 5/5" in stdout2

        # --- Poll 3: Incremental for both ---
        with open(champion, "a") as f:
            f.write(json.dumps(make_assistant_text("Results look good", agent_name="champion-1")) + "\n")
        with open(validator, "a") as f:
            f.write(json.dumps(make_assistant_text("Validation complete", agent_name="validator-1")) + "\n")

        stdout3, _, _ = run_filter(
            str(champion), start_line=last_line_2,
            include_subagents=True,
            projects_dir=str(projects_dir), state_file=state_file)

        assert "Results look good" in stdout3
        assert "Validation complete" in stdout3
        # Previous content should not appear
        assert "Starting optimization" not in stdout3
        assert "PASS: 5/5" not in stdout3
