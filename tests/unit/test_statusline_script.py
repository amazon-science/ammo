# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for ammo-statusline.sh — stdin→stdout contract (Test 3).

The script reads Claude Code statusline JSON from stdin and outputs
a single pipe-delimited line with 7 fields:
  session_id|model|cwd|input_tokens/output_tokens|current_usage|total_duration_ms|version

Platform guard: requires bash + jq (Linux only).
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script under test
SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "ai_cli_session"
    / ".claude"
    / "hooks"
    / "ammo-statusline.sh"
)

LINUX_ONLY = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="needs bash + jq (Linux only)",
)

# ---- Sample JSON matching the Claude Code statusline protocol ----
SAMPLE_JSON = {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "model": {"display_name": "claude-opus-4-6"},
    "workspace": {
        "current_dir": "/home/session_user/vllm",
    },
    "context_window": {
        "total_input_tokens": 50000,
        "total_output_tokens": 10000,
        "current_usage": {"input_tokens": 3000, "output_tokens": 2000},
    },
    "cost": {
        "total_duration_ms": 12345,
    },
    "version": "1.2.3",
}


def _run_script(input_data: dict) -> subprocess.CompletedProcess:
    """Run the statusline script with the given JSON as stdin."""
    return subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.unit
@LINUX_ONLY
class TestStatuslineScript:
    """Test 3: ammo-statusline.sh parses JSON stdin and outputs pipe-delimited line."""

    def test_script_exits_zero(self):
        """Script must exit with code 0 on valid input."""
        result = _run_script(SAMPLE_JSON)
        assert result.returncode == 0, (
            f"Script exited with {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    def test_output_is_single_line(self):
        """Output must be a single non-empty line."""
        result = _run_script(SAMPLE_JSON)
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1, (
            f"Expected single-line output, got {len(lines)} lines: {result.stdout!r}"
        )

    def test_output_has_seven_pipe_delimited_fields(self):
        """Output must have exactly 7 pipe-delimited fields."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert len(fields) == 7, (
            f"Expected 7 pipe-delimited fields, got {len(fields)}: {fields!r}"
        )

    def test_output_contains_session_id(self):
        """First field must be the full UUID session_id."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert fields[0] == SAMPLE_JSON["session_id"], (
            f"Field 0 (session_id) expected {SAMPLE_JSON['session_id']!r}, got {fields[0]!r}"
        )

    def test_output_contains_model_name(self):
        """Second field must be the model display_name."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert fields[1] == SAMPLE_JSON["model"]["display_name"], (
            f"Field 1 (model) expected {SAMPLE_JSON['model']['display_name']!r}, got {fields[1]!r}"
        )

    def test_output_contains_current_dir(self):
        """Third field must be workspace.current_dir."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert fields[2] == SAMPLE_JSON["workspace"]["current_dir"], (
            f"Field 2 (cwd) expected {SAMPLE_JSON['workspace']['current_dir']!r}, "
            f"got {fields[2]!r}"
        )

    def test_output_contains_input_output_tokens(self):
        """Fourth field must be labeled tokens with input/output."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        expected = (
            f"tokens: {SAMPLE_JSON['context_window']['total_input_tokens']}in / "
            f"{SAMPLE_JSON['context_window']['total_output_tokens']}out"
        )
        assert fields[3] == expected, (
            f"Field 3 (tokens) expected {expected!r}, got {fields[3]!r}"
        )

    def test_output_contains_current_usage_with_percentage(self):
        """Fifth field must be labeled ctx with input/output tokens and used percentage."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        usage = SAMPLE_JSON["context_window"]["current_usage"]
        # (3000+2000) / (50000+10000) * 100 = 8.3%
        expected = f"ctx: {usage['input_tokens']}in / {usage['output_tokens']}out (used 8.3%)"
        assert fields[4] == expected, (
            f"Field 4 (current_usage) expected {expected!r}, got {fields[4]!r}"
        )

    def test_output_contains_total_duration_ms(self):
        """Sixth field must be labeled dur with ms suffix."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        expected = f"dur: {SAMPLE_JSON['cost']['total_duration_ms']}ms"
        assert fields[5] == expected, (
            f"Field 5 (total_duration_ms) expected {expected!r}, got {fields[5]!r}"
        )

    def test_output_contains_version(self):
        """Seventh field must be the version prefixed with 'v'."""
        result = _run_script(SAMPLE_JSON)
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert fields[6] == f"v{SAMPLE_JSON['version']}", (
            f"Field 6 (version) expected 'v{SAMPLE_JSON['version']}', got {fields[6]!r}"
        )

    def test_graceful_handling_of_missing_fields(self):
        """Script must not crash when fields are absent — outputs '?' or empty string."""
        minimal_json = {}  # All fields missing
        result = _run_script(minimal_json)
        assert result.returncode == 0, (
            f"Script must exit 0 even with empty JSON.\nstderr: {result.stderr}"
        )
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1, "Must still produce single-line output for empty JSON"
        # Each missing field should produce '?' or empty — no raw 'null' or crash
        output = result.stdout.strip()
        assert "null" not in output, (
            "Output must not contain literal 'null' for missing fields; use '?' or empty"
        )

    def test_partial_json_missing_nested_subkeys(self):
        """Verifier edge-case: partially populated JSON — top-level keys present but nested subkeys absent.

        Simulates early-session Claude messages where context_window exists but
        total_output_tokens/current_usage are not yet populated.
        """
        partial = {
            "session_id": "partial-test-session",
            "model": "test-model",
            "context_window": {
                "total_input_tokens": 1000,
                # total_output_tokens and current_usage are absent
            },
            # workspace, cost, version all absent
        }
        result = _run_script(partial)
        assert result.returncode == 0, (
            f"Script must exit 0 on partial JSON.\nstderr: {result.stderr}"
        )
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1, f"Must produce single-line output, got {len(lines)} lines"
        fields = [f.strip() for f in result.stdout.strip().split("|")]
        assert len(fields) == 7, f"Must produce 7 fields, got {len(fields)}: {fields!r}"
        assert fields[0] == "partial-test-session", f"session_id field wrong: {fields[0]!r}"
        assert fields[1] == "test-model", f"model field wrong: {fields[1]!r}"
        assert "null" not in result.stdout, (
            "Output must not contain literal 'null' for missing nested subkeys"
        )
