#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Z2-T3 / P4#2: status/verdict set sync-oracle (assertion-only).

This file IS the deliverable — it edits NO production file. It pins the
load-bearing status/verdict sets that are duplicated across four sources and
fails loudly the moment any one of them drifts:

  - schema track `status` enum + `verdict` enum (state.schema.json)
  - transitions.json `track_terminal_statuses` / `track_passing_statuses` /
    `integration_terminal_statuses` / `integration_ship_statuses` /
    `terminal_statuses`
  - ammo_state.py, which must READ those sets from transitions.json and hold no
    inline copy of them
  - generate_validation_report.py's passing-COMPAT tuple
    `("PASS", "GATING_REQUIRED", "GATED_PASS")`

All are located by CONTENT (enum-signature / regex over source), never by line
number, so they survive code motion. The test only READS frozen files; a guard
asserts it never mutates them.

Run: python3 -m pytest tests/test_status_set_sync.py -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent.parent / "scripts"
_CLAUDE_DIR = _HERE.parents[3]  # .../ai_cli_session/.claude
_SCHEMA_PATH = _CLAUDE_DIR / "schemas" / "state.schema.json"
_TRANSITIONS_PATH = _SCRIPTS_DIR / "transitions.json"
_ENGINE_PATH = _SCRIPTS_DIR / "ammo_state.py"
_REPORT_PATH = _SCRIPTS_DIR / "generate_validation_report.py"


# ─────────────────────────────────────────────
# Content-keyed locators (no line pins)
# ─────────────────────────────────────────────

def _all_enums(node, acc):
    """Recursively collect every `enum` array (as a list) in a JSON-schema dict."""
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list):
            acc.append(node["enum"])
        for v in node.values():
            _all_enums(v, acc)
    elif isinstance(node, list):
        for v in node:
            _all_enums(v, acc)
    return acc


def _schema_status_enum():
    """The track-status enum = the enum array containing BOTH 'IN_PROGRESS' and
    'GPU_BLOCKED' (uniquely identifies parallel_tracks.tracks.*.status)."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    matches = [set(e) for e in _all_enums(schema, []) if "IN_PROGRESS" in e and "GPU_BLOCKED" in e]
    assert len(matches) == 1, "expected exactly one status enum, found %d" % len(matches)
    return matches[0]


def _schema_verdict_enum():
    """The verdict enum = the enum array containing BOTH 'GATING_REQUIRED' and
    null (None). 'GATING_REQUIRED'+null uniquely separates it from the
    classification enum (lossless/lossy/null, no GATING_REQUIRED) and from the
    status enum (no null)."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    matches = [set(e) for e in _all_enums(schema, []) if "GATING_REQUIRED" in e and None in e]
    assert len(matches) == 1, "expected exactly one verdict enum, found %d" % len(matches)
    return matches[0]


def _set_from_quoted(text):
    """Pull every double-quoted token out of a matched literal into a set."""
    return set(re.findall(r'"([^"]+)"', text))


# Every status set the engine must read from transitions.json, paired with a
# whitespace-tolerant regex that matches an inline copy of the same members.
_ENGINE_SET_KEYS = {
    "track_terminal_statuses": r'"PASS"\s*,\s*"GATED_PASS"\s*,\s*"FAIL"',
    "track_passing_statuses": r'"PASS"\s*,\s*"GATED_PASS"',
    "terminal_statuses": r'"campaign_complete"\s*,\s*"campaign_exhausted"',
    "integration_terminal_statuses": r'"completed"\s*,\s*"exhausted"\s*,\s*"failed"',
    "integration_ship_statuses": r'"combined"\s*,\s*"single_pass"\s*,\s*"gated_pass"',
}


def _transitions():
    return json.loads(_TRANSITIONS_PATH.read_text(encoding="utf-8"))


def _engine_inline_status_literals():
    """Content-grep ammo_state.py for any inline copy of a transitions status set.
    Returns the offending transitions keys (empty when the engine reads them all)."""
    src = _ENGINE_PATH.read_text(encoding="utf-8")
    return sorted(key for key, pattern in _ENGINE_SET_KEYS.items() if re.search(pattern, src))


def _report_passing_compat_tuple():
    """Content-grep generate_validation_report.py for the passing-COMPAT tuple
    ("PASS", "GATING_REQUIRED", "GATED_PASS") (whitespace-tolerant)."""
    src = _REPORT_PATH.read_text(encoding="utf-8")
    m = re.search(r'\(\s*"PASS"\s*,\s*"GATING_REQUIRED"\s*,\s*"GATED_PASS"\s*\)', src)
    assert m is not None, "report passing-compat tuple not found"
    return _set_from_quoted(m.group(0))


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_schema_status_enum_exact():
    assert _schema_status_enum() == {
        "IN_PROGRESS", "PASS", "GATING_REQUIRED", "GATED_PASS", "FAIL", "GPU_BLOCKED"
    }


def test_schema_verdict_enum_exact():
    assert _schema_verdict_enum() == {
        "PASS", "GATING_REQUIRED", "GATED_PASS", "FAIL", None
    }


def test_transitions_terminal_set_and_subset_of_schema():
    terminal = set(_transitions()["track_terminal_statuses"])
    assert terminal == {"PASS", "GATED_PASS", "FAIL"}
    # terminal ⊆ schema status enum
    assert terminal <= _schema_status_enum()


def test_engine_reads_every_status_set_from_transitions():
    """transitions.json is the SOLE owner: the engine looks every status set up by
    key and keeps no inline literal that can drift out of sync (the gated_pass
    defect, where one semantic change needed three hand edits)."""
    src = _ENGINE_PATH.read_text(encoding="utf-8")
    transitions = _transitions()
    for key in _ENGINE_SET_KEYS:
        assert key in transitions, "transitions.json lost status set %s" % key
        assert '["%s"]' % key in src, "ammo_state.py does not read transitions[%r]" % key
    assert _engine_inline_status_literals() == [], (
        "ammo_state.py hardcodes status sets that transitions.json owns: %s"
        % _engine_inline_status_literals()
    )


def test_report_passing_compat_subset_of_verdict_and_distinct_from_terminal():
    compat = _report_passing_compat_tuple()
    assert compat == {"PASS", "GATING_REQUIRED", "GATED_PASS"}
    # passing-compat ⊆ verdict enum (compare on the string members; verdict enum
    # also carries None, which the tuple legitimately omits)
    verdict_strings = {v for v in _schema_verdict_enum() if isinstance(v, str)}
    assert compat <= verdict_strings
    # explicitly a DIFFERENT set from the terminal set: includes GATING_REQUIRED,
    # excludes FAIL.
    terminal = set(_transitions()["track_terminal_statuses"])
    assert compat != terminal
    assert "GATING_REQUIRED" in compat and "GATING_REQUIRED" not in terminal
    assert "FAIL" in terminal and "FAIL" not in compat


def test_passing_set_subset_of_terminal():
    """transitions track_passing_statuses ⊆ track_terminal_statuses (a passing
    status is always terminal)."""
    transitions = _transitions()
    passing = set(transitions["track_passing_statuses"])
    terminal = set(transitions["track_terminal_statuses"])
    assert passing == {"PASS", "GATED_PASS"}
    assert passing <= terminal


def test_integration_ship_statuses_preserve_lowercase_gated_pass():
    transitions = json.loads(_TRANSITIONS_PATH.read_text(encoding="utf-8"))
    assert set(transitions["integration_ship_statuses"]) == {
        "combined",
        "single_pass",
        "gated_pass",
    }


def test_jsonschema_importable():
    """CI-insurance: the 90+/98 state-validate baseline asserts jsonschema-specific
    error strings, and P4#6 fail-closed is DEFAULT-ON — both silently degrade if
    jsonschema vanishes from the env. This cheap assertion surfaces that loudly."""
    import jsonschema  # noqa: F401


def test_oracle_reads_only_no_writes():
    """Guard: this oracle must never mutate the frozen files it inspects. Snapshot
    (size, mtime_ns) before running every locator, then re-check after."""
    frozen = [_SCHEMA_PATH, _TRANSITIONS_PATH, _ENGINE_PATH, _REPORT_PATH]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in frozen}
    # exercise every read path
    _schema_status_enum()
    _schema_verdict_enum()
    _engine_inline_status_literals()
    _report_passing_compat_tuple()
    json.loads(_TRANSITIONS_PATH.read_text(encoding="utf-8"))
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in frozen}
    assert before == after, "sync-oracle must not modify frozen files"
