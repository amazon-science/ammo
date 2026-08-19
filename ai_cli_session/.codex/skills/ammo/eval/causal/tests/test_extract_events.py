# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for extract_events.py — JSONL → structured event log."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_session_data():
    return json.loads((FIXTURES / "sample_session_data.json").read_text())


def _extract():
    from extract_events import extract_events
    return extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=_load_session_data(),
    )


def test_extract_produces_events_list():
    """Basic: extraction produces a non-empty events list."""
    events = _extract()
    assert isinstance(events, list)
    assert len(events) > 0


def test_file_write_events_extracted():
    """File write tool calls become file_write events."""
    events = _extract()
    file_writes = [e for e in events if e["event_type"] == "file_write"]
    paths = [e["details"]["path"] for e in file_writes]
    assert any("bottleneck_analysis.md" in p for p in paths)


def test_file_read_events_extracted():
    """File read tool calls become file_read events."""
    events = _extract()
    file_reads = [e for e in events if e["event_type"] == "file_read"]
    paths = [e["details"]["path"] for e in file_reads]
    assert any("bottleneck_analysis.md" in p for p in paths)


def test_data_output_events_extracted():
    """Numeric patterns in text outputs become data_output events."""
    events = _extract()
    data_outputs = [e for e in events if e["event_type"] == "data_output"]
    assert len(data_outputs) > 0
    datum_types = {e["details"]["datum_type"] for e in data_outputs}
    assert "f_value" in datum_types


def test_agent_id_and_name_separated():
    """Events have separate agent_id (UUID) and agent_name fields."""
    events = _extract()
    spawns = [e for e in events if e["event_type"] == "agent_spawn"]
    assert len(spawns) > 0
    for s in spawns:
        assert "agent_id" in s
        assert "agent_name" in s


def test_stage_assignment_from_session_data():
    """Stage field comes from session_data timestamps, not re-derived."""
    events = _extract()
    stages = {e.get("stage") for e in events if e.get("stage")}
    assert any("baseline" in s for s in stages)


def test_graceful_fail_missing_jsonl():
    """Missing JSONL file raises FileNotFoundError."""
    from extract_events import extract_events
    with pytest.raises(FileNotFoundError):
        extract_events(
            session_jsonl_path=Path("/nonexistent/session.jsonl"),
            session_data=_load_session_data(),
        )


def test_truncated_jsonl_handles_gracefully(tmp_path):
    """Truncated JSONL processes valid lines and ignores the broken last line."""
    from extract_events import extract_events
    content = (FIXTURES / "sample_session.jsonl").read_text()
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text(content + '{"type": "assistant", "truncated_mid_')
    events = extract_events(
        session_jsonl_path=truncated,
        session_data=_load_session_data(),
    )
    assert isinstance(events, list)
    assert len(events) > 0


def test_cli_interface(tmp_path):
    """CLI produces output JSON file."""
    import subprocess
    out = tmp_path / "events.json"
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "extract_events.py"),
         "--session-jsonl", str(FIXTURES / "sample_session.jsonl"),
         "--session-data", str(FIXTURES / "sample_session_data.json"),
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert isinstance(data, list)
