# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Model-roster agreement tests.

Model IDs are pinned in four uncoordinated places: settings.local.json,
every agent frontmatter, the generated Codex config in session_manager, and
the project Codex config.
ai_cli_session/.claude/models.json records the current pins. These tests
assert every site still agrees with the roster, so a model bump that misses
a site fails here instead of drifting silently.

The roster does NOT select models at runtime. It only makes divergence loud.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
AI_CLI = ROOT / "ai_cli_session"
ROSTER_PATH = AI_CLI / ".claude" / "models.json"
SETTINGS_LOCAL = AI_CLI / ".claude" / "settings.local.json"
AGENTS_DIR = AI_CLI / ".claude" / "agents"
CODEX_CONFIG = AI_CLI / ".codex" / "config.toml"
SESSION_MANAGER = ROOT / "orchestration" / "session_manager.py"

MODEL_LINE = re.compile(r"^\s*model:\s*(\S+)\s*$")


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def roster() -> dict:
    return json.loads(_read(ROSTER_PATH))


def _frontmatter_model(text: str) -> str | None:
    """Return the top-level frontmatter `model:` value, ignoring nested ones."""
    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", "agent file must open with frontmatter"
    for line in lines[1:]:
        if line.strip() == "---":
            break
        # Top-level keys carry no leading whitespace; nested hook models do.
        if line.startswith("model:"):
            return line.split(":", 1)[1].strip()
    return None


@pytest.mark.unit
class TestRosterShape:
    def test_roster_parses_and_declares_every_section(self, roster):
        for key in ("aliases", "claude", "codex"):
            assert key in roster, f"roster missing '{key}' section"
        assert roster["claude"]["agents"], "roster declares no agents"

    def test_every_agent_value_is_a_known_alias_or_explicit_id(self, roster):
        aliases = {k for k in roster["aliases"] if not k.startswith("_")}
        for name, value in roster["claude"]["agents"].items():
            assert value in aliases or "." in value, (
                f"roster agent {name}={value!r} is neither a known alias "
                f"{sorted(aliases)} nor an explicit model id"
            )


@pytest.mark.unit
class TestSettingsLocalAgreesWithRoster:
    def test_session_model_matches(self, roster):
        settings = json.loads(_read(SETTINGS_LOCAL))
        assert settings.get("model") == roster["claude"]["session_model"], (
            "settings.local.json 'model' diverged from models.json "
            "claude.session_model"
        )

    def test_alias_default_env_vars_match(self, roster):
        settings = json.loads(_read(SETTINGS_LOCAL))
        env = settings.get("env", {})
        for alias, var in (
            ("opus", "ANTHROPIC_DEFAULT_OPUS_MODEL"),
            ("sonnet", "ANTHROPIC_DEFAULT_SONNET_MODEL"),
            ("haiku", "ANTHROPIC_DEFAULT_HAIKU_MODEL"),
        ):
            assert env.get(var) == roster["aliases"][alias], (
                f"settings.local.json env.{var} diverged from models.json "
                f"aliases.{alias}"
            )


@pytest.mark.unit
class TestAgentFrontmatterAgreesWithRoster:
    def test_roster_covers_exactly_the_agent_files(self, roster):
        on_disk = {p.stem for p in sorted(AGENTS_DIR.glob("*.md"))}
        in_roster = set(roster["claude"]["agents"])
        assert on_disk == in_roster, (
            f"agent files and roster disagree: only on disk {sorted(on_disk - in_roster)}, "
            f"only in roster {sorted(in_roster - on_disk)}"
        )

    def test_each_agent_frontmatter_model_matches(self, roster):
        for name, expected in roster["claude"]["agents"].items():
            actual = _frontmatter_model(_read(AGENTS_DIR / f"{name}.md"))
            assert actual == expected, (
                f"{name}.md frontmatter model={actual!r} but roster says "
                f"{expected!r}"
            )

    def test_inline_hook_model_pins_match(self, roster):
        pins = roster["claude"]["inline_hook_models"]
        for key, expected in pins.items():
            if key.startswith("_"):
                continue
            agent = key.split(":", 1)[0]
            text = _read(AGENTS_DIR / f"{agent}.md")
            nested = [
                MODEL_LINE.match(line).group(1)
                for line in text.splitlines()
                if MODEL_LINE.match(line) and line.startswith((" ", "\t"))
            ]
            assert expected in nested, (
                f"{agent}.md has no nested hook model {expected!r}; found {nested}"
            )

    def test_no_unrecorded_inline_hook_model_pins(self, roster):
        recorded = {
            v for k, v in roster["claude"]["inline_hook_models"].items()
            if not k.startswith("_")
        }
        for path in sorted(AGENTS_DIR.glob("*.md")):
            for line in _read(path).splitlines():
                m = MODEL_LINE.match(line)
                if m and line.startswith((" ", "\t")):
                    assert m.group(1) in recorded, (
                        f"{path.name} pins an unrecorded nested model "
                        f"{m.group(1)!r}; add it to models.json "
                        f"claude.inline_hook_models"
                    )


@pytest.mark.unit
class TestCodexConfigAgreesWithRoster:
    def test_project_config_model_matches(self, roster):
        expected = roster["codex"]["project_config"]
        assert f'model = "{expected}"' in _read(CODEX_CONFIG), (
            ".codex/config.toml model diverged from models.json "
            "codex.project_config"
        )

    def test_generated_codex_home_config_model_matches(self, roster):
        expected = roster["codex"]["generated_config"]
        assert f"'model = \"{expected}\"'" in _read(SESSION_MANAGER), (
            "session_manager._prepare_codex_home model diverged from "
            "models.json codex.generated_config"
        )

    def test_project_and_generated_codex_models_agree(self, roster):
        assert roster["codex"]["project_config"] == roster["codex"]["generated_config"], (
            "the generated CODEX_HOME config wins at runtime; the project copy "
            "must state the same model or it misleads readers"
        )
