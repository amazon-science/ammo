# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Codex hook-registration parity between hooks.json and requirements.toml.

Two files declare the same Codex hook wiring:

* ``.codex/requirements.toml`` is the ENFORCED one. The Dockerfile installs it
  at /etc/codex/requirements.toml and it sets allow_managed_hooks_only = true,
  so the managed config wins inside the container.
* ``.codex/hooks.json`` is the readable project/template copy. Agents and the
  codex hook suite read it.

They drifted once already: hooks.json matched ``spawn_agent`` on PostToolUse and
requirements.toml did not, so a registration a codex test asserted was absent in
production. These tests pin the event/matcher/timeout tuples together, so drift
in either direction fails here.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
CODEX = ROOT / "ai_cli_session" / ".codex"
HOOKS_JSON = CODEX / "hooks.json"
REQUIREMENTS_TOML = CODEX / "requirements.toml"

EXPECTED_EVENTS = {
    "SessionStart",
    "PreCompact",
    "PreToolUse",
    "PostToolUse",
    "SubagentStart",
    "SubagentStop",
    "Stop",
}

# hooks.json runs the project-relative script via a /bin/sh resolver; the
# managed file runs an absolute path under /opt. Compare the script basename.
SCRIPTS = (
    "session_start.py",
    "pre_compact.py",
    "pre_tool_use_guard.py",
    "post_tool_use_guard.py",
    "stop_gate_guard.py",
)


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


def _script_of(command: str) -> str:
    found = [name for name in SCRIPTS if name in command]
    assert len(found) == 1, f"command names {found} hook scripts, expected 1: {command}"
    return found[0]


def _registrations(events: dict) -> dict:
    """Normalize one file's hook block into {event: [(matcher, script, timeout, msg)]}."""
    out: dict[str, list[tuple]] = {}
    for event, entries in events.items():
        if not isinstance(entries, list):
            continue
        normalized = []
        for entry in entries:
            for hook in entry.get("hooks", []):
                normalized.append((
                    entry.get("matcher"),
                    hook.get("type"),
                    _script_of(hook.get("command", "")),
                    hook.get("timeout"),
                    hook.get("statusMessage"),
                ))
        out[event] = normalized
    return out


@pytest.fixture(scope="module")
def project_registrations() -> dict:
    return _registrations(json.loads(_read(HOOKS_JSON))["hooks"])


@pytest.fixture(scope="module")
def managed_registrations() -> dict:
    return _registrations(tomllib.loads(_read(REQUIREMENTS_TOML))["hooks"])


@pytest.mark.unit
class TestCodexHookRegistrationParity:
    def test_managed_config_is_the_enforced_one(self):
        managed = tomllib.loads(_read(REQUIREMENTS_TOML))
        assert managed.get("allow_managed_hooks_only") is True, (
            "requirements.toml must keep allow_managed_hooks_only = true; "
            "the parity contract assumes the managed file wins in the container"
        )
        assert managed["hooks"]["managed_dir"] == "/opt/codex-managed-hooks", (
            "managed_dir moved; the Dockerfile installs the closure at "
            "/opt/codex-managed-hooks"
        )

    def test_both_files_declare_the_same_event_set(
        self, project_registrations, managed_registrations
    ):
        assert set(project_registrations) == EXPECTED_EVENTS
        assert set(managed_registrations) == EXPECTED_EVENTS

    def test_every_event_declares_identical_registrations(
        self, project_registrations, managed_registrations
    ):
        for event in sorted(EXPECTED_EVENTS):
            assert project_registrations[event] == managed_registrations[event], (
                f"{event} registration diverged.\n"
                f"  hooks.json:        {project_registrations[event]}\n"
                f"  requirements.toml: {managed_registrations[event]}"
            )

    def test_spawn_agent_is_registered_on_the_enforced_post_tool_matcher(
        self, managed_registrations
    ):
        # The codex hook suite asserts this registration against hooks.json.
        # Only the managed file actually registers it in the container.
        matchers = [entry[0] or "" for entry in managed_registrations["PostToolUse"]]
        assert any("spawn_agent" in matcher for matcher in matchers), (
            "requirements.toml PostToolUse must match spawn_agent, otherwise the "
            "monitor-ledger path the codex suite asserts is unregistered in "
            f"production; matchers were {matchers}"
        )

    def test_post_tool_and_subagent_start_run_the_same_guard(
        self, managed_registrations
    ):
        post_scripts = {entry[2] for entry in managed_registrations["PostToolUse"]}
        start_scripts = {entry[2] for entry in managed_registrations["SubagentStart"]}
        assert post_scripts == start_scripts == {"post_tool_use_guard.py"}, (
            "the spawn-time ledger write is safe only because both events run "
            "the same idempotent guard"
        )
