# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Structural cross-reference tests for ammo-auditor documentation.

Verifies that the auditor protocol, invariants reference, SKILL.md triggers,
schema extension, and agent definition are consistent and internally well-formed.
These are pure string / JSON / YAML checks — no subprocess, no runtime.

Test cases (per plan Task 8):
  1. audit-protocol.md exists with required sections
  2. audit-invariants.md exists with per-stage sections
  3. SKILL.md references audit (T_AUDIT_S1, audit-protocol.md, audit-invariants.md)
  4. state.schema.json allows the audit sub-object
  5. audit-invariants.md item counts per section match plan
  6. audit-invariants.md field refs resolve in schema
  7. audit-invariants.md severity values valid (BLOCKING / HIGH / LOW)
  8. ammo-auditor.md agent definition exists with required frontmatter + sections
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
AMMO = ROOT / "ai_cli_session" / ".claude" / "skills" / "ammo"
SKILL = AMMO / "SKILL.md"
PROTOCOL = AMMO / "orchestration" / "audit-protocol.md"
INVARIANTS = AMMO / "references" / "audit-invariants.md"
SCHEMA = ROOT / "ai_cli_session" / ".claude" / "schemas" / "state.schema.json"
AUDITOR_AGENT = ROOT / "ai_cli_session" / ".claude" / "agents" / "ammo-auditor.md"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: audit-protocol.md exists with required sections
# ---------------------------------------------------------------------------

def test_audit_protocol_exists_with_sections() -> None:
    body = _read(PROTOCOL)

    required_headings = [
        "## Dispatch",
        "## Mandatory Triggers",
        "## Optional Early S45 Evidence",
        "## Verdict Handling",
        "## Repair and Re-audit",
        "## Artifact Paths",
    ]
    for heading in required_headings:
        assert heading in body, f"audit-protocol.md missing section: {heading}"


# ---------------------------------------------------------------------------
# Test 2: audit-invariants.md exists with per-stage sections
# ---------------------------------------------------------------------------

def test_audit_invariants_exists_with_stages() -> None:
    body = _read(INVARIANTS)

    stage_headings = [
        "## Pre-Check",
        "## After Stage 1",
        "## After Stage 2",
        "## After Stages 4-5",
        "## After Stage 6-7",
        "## Cross-Artifact Checks",
    ]
    for heading in stage_headings:
        assert heading in body, f"audit-invariants.md missing section: {heading}"


# ---------------------------------------------------------------------------
# Test 3: SKILL.md references audit gate machinery
# ---------------------------------------------------------------------------

def test_skill_md_references_audit() -> None:
    body = _read(SKILL)

    for token in [
        "T_AUDIT_S1",
        "T_AUDIT_S45",
        "T_AUDIT_S67",
        "audit-protocol.md",
        "audit-invariants.md",
        "## Audit Gates",
    ]:
        assert token in body, f"SKILL.md missing audit reference: {token}"

    # Must reference the round-bootstrap rule
    assert (
        '"audit": {}' in body or "audit: {}" in body
    ), "SKILL.md should document seeding audit:{} in new round entries"


# ---------------------------------------------------------------------------
# Test 4: state.schema.json allows audit sub-object
# ---------------------------------------------------------------------------

def test_schema_allows_audit_field() -> None:
    schema = json.loads(_read(SCHEMA))

    rounds_items = (
        schema["properties"]["campaign"]["properties"]["rounds"]["items"]
    )
    assert "audit" in rounds_items["properties"], (
        "rounds[*] schema must declare an 'audit' property"
    )

    audit_prop = rounds_items["properties"]["audit"]
    # audit must be nullable/optional so legacy campaigns without the key still validate
    assert "audit" not in rounds_items.get("required", []), (
        "'audit' must NOT be required on rounds (preserves backward compat)"
    )

    # Each stage sub-field should be declared
    assert "properties" in audit_prop, "audit must be a typed object with properties"
    for stage_key in ("stage_1", "stage_45", "stage_6", "stage_7", "stage_67"):
        assert stage_key in audit_prop["properties"], (
            f"audit schema missing {stage_key} sub-property"
        )
        stage_schema = audit_prop["properties"][stage_key]
        # Each stage sub-object must declare passed_at + verdict_file (both nullable)
        assert "passed_at" in stage_schema["properties"], (
            f"audit.{stage_key} schema missing 'passed_at'"
        )
        assert "verdict_file" in stage_schema["properties"], (
            f"audit.{stage_key} schema missing 'verdict_file'"
        )

    # stage_67 must additionally have auto_pass
    stage_67_schema = audit_prop["properties"]["stage_67"]
    assert "auto_pass" in stage_67_schema["properties"], (
        "audit.stage_67 schema missing 'auto_pass' property"
    )


# ---------------------------------------------------------------------------
# Helpers for parsing invariants.md bullet rows
# ---------------------------------------------------------------------------

# The minified invariants use bold-bullet items ("- **Name (SEVERITY).** ...")
# instead of numbered tables. Section name → minimum item count (coverage
# floor; adding items is always allowed).
EXPECTED_MIN_COUNTS: Dict[str, int] = {
    "Pre-Check": 4,
    "After Stage 1": 5,
    "After Stage 2": 5,
    "After Stages 4-5": 8,
    "After Stage 6-7": 7,
    "Cross-Artifact Checks": 5,
}

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
BULLET_RE = re.compile(r"^- \*\*(?P<name>[^*]+?)(?:\s*\((?P<sev>[A-Z]+)\))?\.?\*\*", re.MULTILINE)


def _section_slices(body: str) -> Dict[str, str]:
    """Split the invariants doc into {section_name: section_body}."""
    matches = list(SECTION_RE.finditer(body))
    slices: Dict[str, str] = {}
    for i, m in enumerate(matches):
        raw = m.group(1).strip()
        name = re.split(r"\s*[\(—]", raw, maxsplit=1)[0].strip().rstrip("`").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        slices[name] = body[start:end]
    return slices


def _count_invariant_rows(section_body: str) -> int:
    return len(BULLET_RE.findall(section_body))


# ---------------------------------------------------------------------------
# Test 5: Invariant item coverage floor per section
# ---------------------------------------------------------------------------

def test_audit_invariants_item_counts() -> None:
    body = _read(INVARIANTS)
    sections = _section_slices(body)

    for section_name, minimum in EXPECTED_MIN_COUNTS.items():
        assert section_name in sections, (
            f"audit-invariants.md missing section '{section_name}'. "
            f"Found sections: {sorted(sections)}"
        )
        actual = _count_invariant_rows(sections[section_name])
        assert actual >= minimum, (
            f"Section '{section_name}' has {actual} invariant items, expected >= {minimum}"
        )


# ---------------------------------------------------------------------------
# Test 6: Invariant field references resolve in schema
# ---------------------------------------------------------------------------

# Field-like names the invariants cite in backticks that must resolve to real
# schema property names (documentation referencing dead fields rots silently).
FIELD_REF_RE = re.compile(r"`([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)`")

# Backticked tokens that are legitimately not schema fields (files, concepts).
NON_FIELD_TOKENS: Set[str] = {
    "state.json",
    "evidence.json",
    "target.json",
    "diff.patch",
    "runtime_pkg.patch",
    "diluted_pass",
    "f_e2e",
    "f_decode",
    "inter_kernel_share",
    "decode_busy",
    "decode_share_of_e2e",
    # sweep-artifact fields (e2e_latency_results.json), not state.json fields
    "avg_s",
    "decode_avg_s",
    "prefill_avg_s",
    "prefill_share",
    "prefill_avg",
    "decode_avg",
    "decode_kernel_s",
    "decode_wall_s",
    "decode_share_of_phase_sum",
    # mined.json fields (mine_trace/1 schema), not state.json fields
    "config_echo",
    "e2e_results",
    "first_kernel_after",
    "partition_coverage",
    "provenance",
    "residual_pct",
    "separation_ratio",
    "step_count_source",
}

# Backticked filenames. The "/" guard below misses bare names like `tables.md`.
FILENAME_REF_RE = re.compile(r"\.(md|json|patch)$")


def _collect_schema_property_names(node: dict) -> Set[str]:
    """Walk the schema and collect every `properties` key (field name)."""
    names: Set[str] = set()
    if not isinstance(node, dict):
        return names
    if "properties" in node and isinstance(node["properties"], dict):
        for k, v in node["properties"].items():
            names.add(k)
            names.update(_collect_schema_property_names(v))
    if "items" in node:
        names.update(_collect_schema_property_names(node["items"]))
    if "additionalProperties" in node and isinstance(
        node["additionalProperties"], dict
    ):
        names.update(_collect_schema_property_names(node["additionalProperties"]))
    return names


def test_audit_invariants_field_refs_resolve() -> None:
    schema = json.loads(_read(SCHEMA))
    schema_names = _collect_schema_property_names(schema)

    body = _read(INVARIANTS)
    for ref in FIELD_REF_RE.findall(body):
        if ref in NON_FIELD_TOKENS or "/" in ref or FILENAME_REF_RE.search(ref):
            continue
        leaf = ref.split(".")[-1]
        assert leaf in schema_names, (
            f"audit-invariants.md cites field {ref!r} whose leaf {leaf!r} is not "
            f"declared in state.schema.json"
        )


# ---------------------------------------------------------------------------
# Test 7: Severity values must be from the closed enum
# ---------------------------------------------------------------------------

VALID_SEVERITIES: Set[str] = {"BLOCKING", "HIGH", "LOW"}


def _collect_severities(body: str) -> List[Tuple[str, str]]:
    """Return (item_name, severity) for every bullet carrying a severity tag."""
    results: List[Tuple[str, str]] = []
    for m in BULLET_RE.finditer(body):
        if m.group("sev"):
            results.append((m.group("name").strip(), m.group("sev")))
    return results


def test_audit_invariants_severity_values_valid() -> None:
    body = _read(INVARIANTS)
    rows = _collect_severities(body)
    assert rows, "No severity-tagged invariant items parsed from audit-invariants.md"

    invalid = [(rid, sev) for rid, sev in rows if sev not in VALID_SEVERITIES]
    assert not invalid, (
        f"Invalid severity values in audit-invariants.md: {invalid}. "
        f"Allowed: {sorted(VALID_SEVERITIES)}"
    )


# ---------------------------------------------------------------------------
# Test 8: ammo-auditor.md agent definition exists with frontmatter + sections
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(body: str) -> Dict[str, str]:
    """Minimal YAML-ish parser for agent frontmatter. Sufficient for flat key: value."""
    m = FRONTMATTER_RE.match(body)
    assert m, "Agent definition missing --- frontmatter block at file start"
    raw = m.group(1)
    fields: Dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("-"):
            continue
        key, _, val = line.partition(":")
        fields[key.strip()] = val.strip()
    return fields


def test_ammo_auditor_agent_definition_exists() -> None:
    body = _read(AUDITOR_AGENT)
    fields = _parse_frontmatter(body)

    assert fields.get("name") == "ammo-auditor", (
        f"ammo-auditor.md frontmatter `name` should be 'ammo-auditor', got {fields.get('name')!r}"
    )
    assert fields.get("model") == "opus", (
        f"ammo-auditor.md frontmatter `model` should be 'opus', got {fields.get('model')!r}"
    )
    assert "description" in fields and fields["description"], (
        "ammo-auditor.md frontmatter must include a non-empty description"
    )

    # Required content sections (h2). These mirror the investigator + devil-advocate
    # fusion mandated by the plan.
    # `## Operating Procedure` was split into the stronger Phase 1 / Phase 2 pair,
    # and `## Dispatch Interface` moved to orchestration/audit-protocol.md § Dispatch.
    required_sections = [
        "## Phase 1 - Independent Reconstruction",  # investigator methodology
        "## Phase 2 - Checklist Verification",      # hook-delivered checklist
        "## Evidence Mandate",       # hard rule: path:line:quote
        "## Verdict Format",         # BLOCKING / HIGH / LOW output template
        "## Severity Classification",
    ]
    for heading in required_sections:
        assert heading in body, f"ammo-auditor.md missing section: {heading}"
