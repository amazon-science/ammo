# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Regression tests: teammateMode must not appear in any generated session config.

Prevents re-introduction of `teammateMode: "in-process"` in session configs.
Three locations are tested:
  1. ai_cli_session/managed-settings.json  (static file)
  2. _create_session_internal's .claude.json dict (create path in session_manager.py)
  3. _resume_session's .claude.json dict         (resume path in session_manager.py)
  4. Full codebase grep across all .py/.json files
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.mark.unit
class TestTeammateModeRemoval:
    """Regression: teammateMode=in-process must not appear in any session config."""

    # ------------------------------------------------------------------
    # Test 1: static managed-settings.json
    # ------------------------------------------------------------------

    def test_managed_settings_json_no_teammate_mode(self):
        """managed-settings.json must not contain the key 'teammateMode'."""
        managed_settings = ROOT / "ai_cli_session" / "managed-settings.json"
        assert managed_settings.exists(), (
            f"managed-settings.json not found at {managed_settings}"
        )
        data = json.loads(managed_settings.read_text())
        assert "teammateMode" not in data, (
            "managed-settings.json contains 'teammateMode' — it must be removed. "
            f"Current keys: {list(data.keys())}"
        )

    # ------------------------------------------------------------------
    # Test 2: _create_session_internal .claude.json write
    # ------------------------------------------------------------------

    def test_create_session_claude_json_no_teammate_mode(self):
        """create_session must not write 'teammateMode' to .claude.json.

        Uses inspect.getsource() to capture the exact source of the method and
        assert that 'teammateMode' does not appear in the .claude.json write block.
        """
        from orchestration.session_manager import SessionManager

        src = inspect.getsource(SessionManager.create_session)
        assert "teammateMode" not in src, (
            "create_session contains 'teammateMode' — "
            "it must not be written to .claude.json at session create time."
        )

    # ------------------------------------------------------------------
    # Test 3: resume_session .claude.json write
    # ------------------------------------------------------------------

    def test_resume_session_claude_json_no_teammate_mode(self):
        """resume_session must not write 'teammateMode' to .claude.json.

        Uses inspect.getsource() to capture the exact source of the method and
        assert that 'teammateMode' does not appear in the .claude.json write block.
        """
        from orchestration.session_manager import SessionManager

        src = inspect.getsource(SessionManager.resume_session)
        assert "teammateMode" not in src, (
            "resume_session contains 'teammateMode' — "
            "it must not be written to .claude.json at session resume time."
        )

    # ------------------------------------------------------------------
    # Test 4: full codebase grep
    # ------------------------------------------------------------------

    def test_no_teammate_mode_anywhere_in_codebase(self):
        """No .py or .json file may set 'teammateMode' to 'in-process'.

        The session template (ai_cli_session/.claude/settings.local.json) may
        pin 'teammateMode': 'tmux' explicitly — that IS the desired mode this
        regression guards. Only the 'in-process' value is forbidden.

        Excludes:
          - .playwright-mcp/
          - node_modules/
          - __pycache__/
          - .claude/plans/
          - This test file itself (the string appears only in assertion messages)
        """
        excluded_subdirs = {
            ".playwright-mcp",
            "node_modules",
            "__pycache__",
        }
        excluded_path_parts = {".claude/plans", ".claude\\plans"}

        matches: list[str] = []

        for ext in ("*.py", "*.json"):
            for fpath in ROOT.rglob(ext):
                # Skip excluded directories
                parts = fpath.parts
                rel = fpath.relative_to(ROOT)
                rel_str = str(rel)

                # Skip if any path component is an excluded subdir name
                if any(p in excluded_subdirs for p in parts):
                    continue
                # Skip .claude/plans/ paths
                if any(ep in rel_str for ep in excluded_path_parts):
                    continue
                # Skip this test file itself (it contains the string in docstrings/comments)
                if fpath.name == "test_teammate_mode_removal.py":
                    continue

                try:
                    content = fpath.read_text(errors="replace")
                except OSError:
                    continue

                # Only teammateMode lines that pin the forbidden 'in-process'
                # value count; an explicit 'tmux' pin is allowed (and desired
                # in the session template).
                if "teammateMode" in content:
                    # Collect the specific lines for a helpful error message
                    for lineno, line in enumerate(content.splitlines(), 1):
                        if "teammateMode" in line and "tmux" not in line:
                            matches.append(f"{rel}:{lineno}: {line.strip()}")

        assert not matches, (
            f"Found 'teammateMode' (non-tmux) in {len(matches)} location(s) — "
            "only an explicit 'tmux' pin is allowed:\n"
            + "\n".join(matches)
        )

    # ------------------------------------------------------------------
    # Edge-case probe (verifier): full session_manager.py source body
    # ------------------------------------------------------------------

    def test_session_manager_module_source_no_teammate_mode(self):
        """Whole session_manager.py source must be free of 'teammateMode'.

        Tests 2 & 3 only scan specific methods; this test catches any helper
        method or class-level code that might inject 'teammateMode' into the
        .claude.json write path.
        """
        session_manager_path = ROOT / "orchestration" / "session_manager.py"
        assert session_manager_path.exists(), (
            f"session_manager.py not found at {session_manager_path}"
        )
        src = session_manager_path.read_text()
        assert "teammateMode" not in src, (
            "orchestration/session_manager.py still contains 'teammateMode' — "
            "all occurrences must be removed."
        )
