# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for AMMO Codex agent runtime-contract drift."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
AGENTS = ROOT / ".codex" / "agents"
AMMO_AGENT_DOCS = ROOT / ".codex" / "skills" / "ammo" / "agents"


def _toml(name: str) -> dict:
    return tomllib.loads((AGENTS / f"{name}.toml").read_text(encoding="utf-8"))


def test_agent_tomls_are_bootstrap_not_policy_layers():
    for path in sorted(AGENTS.glob("ammo-*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        instructions = data.get("developer_instructions", "")
        assert "runtime bootstrap only" in instructions, path.name
        assert "Do not add requirements from this TOML" in instructions, path.name
        assert "orchestration/audit-protocol.md" not in instructions, path.name
        assert "before acting" not in instructions.lower().replace("first, then", ""), path.name


def test_auditor_toml_does_not_load_phase2_checklist_at_spawn():
    instructions = _toml("ammo-auditor")["developer_instructions"]
    assert "audit-invariants.md before acting" not in instructions
    assert "Do not read the Phase 2 checklist at spawn" in instructions
    assert "the hook delivers it after the Phase 1 verdict write" in instructions


def test_auditor_prompt_keeps_phase1_before_hook_delivered_checklist():
    text = (AMMO_AGENT_DOCS / "ammo-auditor.md").read_text(encoding="utf-8")
    phase1 = text.index("## Phase 1 - Independent Reconstruction")
    phase2 = text.index("## Phase 2 - Checklist Verification")
    assert phase1 < phase2
    assert "Do **not** read `references/audit-invariants.md` at spawn" in text
    assert "PostToolUse hook injects instructions" in text
