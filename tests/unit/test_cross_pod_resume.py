# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for cross-pod session resume logic.

Tests the `_fix_claude_project_dir_after_s3_restore()` method in
SessionManager which handles renaming the Claude Code project directory
after S3 restore to a different pod with a different worktree path.

Also tests SessionS3Storage.restore_session_from_s3() for metadata loading
and worktree restore orchestration.

KEY BEHAVIOR: Claude Code encodes project paths by replacing ALL '/' with '-',
INCLUDING the leading '/'. So:
    /data/sessions/abc/worktree  ->  -data-sessions-abc-worktree
                                     ^ leading dash is critical
"""

import json
import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
)
from orchestration.session_manager import SessionManager
from orchestration.session_state import SessionS3Storage


def _make_session_state(
    session_id: str = "test-session-abc123",
    worktree_path: str = "/data/sessions/test-session-abc123/worktree",
    session_dir: str = "/data/sessions/test-session-abc123",
    **kwargs,
) -> SessionState:
    """Helper to create a SessionState for testing."""
    defaults = dict(
        session_id=session_id,
        status=SessionStatus.PAUSED,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=worktree_path,
        session_dir=session_dir,
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


def _make_session_manager(tmp_path):
    """Helper to create a SessionManager with mocked dependencies."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    with patch("orchestration.session_manager.get_worktree_manager"), \
         patch("orchestration.session_manager.GPUResourceManager"), \
         patch("orchestration.session_manager.get_terminal_manager"), \
         patch("orchestration.session_manager.get_cli_tool_manager"), \
         patch("orchestration.session_manager.get_inactivity_monitor"), \
         patch("orchestration.session_manager.get_session_storage") as mock_ss:

        mock_ss.return_value = MagicMock(enabled=False)
        mgr = SessionManager(sessions_dir=str(sessions_dir))

    return mgr


# ============================================================================
# Test Group 1: Path Encoding — The Leading-Dash Bug Fix
# ============================================================================


@pytest.mark.unit
class TestPathEncodingLeadingDash:
    """
    Verify that worktree path encoding produces a LEADING DASH.

    Claude Code encodes paths by replacing ALL '/' with '-', including the
    leading '/'. The old buggy code used lstrip("/") first, which dropped
    the leading dash. These tests verify the fix.
    """

    @pytest.fixture
    def session_manager(self, tmp_path):
        return _make_session_manager(tmp_path)

    @pytest.mark.parametrize("session_id,worktree_path,expected_encoded", [
        # Cross-pod renames — realistic session worktree paths with leading dash
        ("abc", "/data/sessions/abc/worktree", "-data-sessions-abc-worktree"),
        ("abc", "/local/sessions/abc/worktree", "-local-sessions-abc-worktree"),
        ("test-session-abc123", "/data/sessions/test-session-abc123/worktree",
         "-data-sessions-test-session-abc123-worktree"),
        # UUID-style session IDs
        ("887d4000-80e0-41ab-ad9d-ec153a172859",
         "/data/sessions/887d4000-80e0-41ab-ad9d-ec153a172859/worktree",
         "-data-sessions-887d4000-80e0-41ab-ad9d-ec153a172859-worktree"),
    ])
    def test_encodes_worktree_paths_with_leading_dash(
        self, session_manager, tmp_path, session_id, worktree_path, expected_encoded
    ):
        """
        Verify the path encoding: replace ALL '/' with '-', producing a
        leading dash. This is what Claude Code expects.
        """
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=worktree_path,
        )

        # Create old-pod dir (different prefix, same session_id, ends with -worktree)
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-old-pod-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Verify the new dir uses correct encoding WITH leading dash
        assert (projects_dir / expected_encoded).exists(), (
            f"Expected directory '{expected_encoded}' not found. "
            f"Contents: {[d.name for d in projects_dir.iterdir()]}"
        )
        assert not old_dir.exists()

    def test_encoding_formula_produces_leading_dash(self):
        """
        Verify the raw encoding formula: path.replace("/", "-") produces
        a leading dash for ALL absolute paths.
        """
        test_cases = [
            ("/data/sessions/abc/worktree", "-data-sessions-abc-worktree"),
            ("/local/test/path", "-local-test-path"),
            ("/a/b/c/d/e/f", "-a-b-c-d-e-f"),
            ("/single", "-single"),
            ("/root", "-root"),
        ]
        for path, expected in test_cases:
            assert path.replace("/", "-") == expected
            assert expected.startswith("-"), f"Encoding of {path} must start with dash"

    def test_encoding_matches_claude_code_behavior(self, session_manager, tmp_path):
        """
        End-to-end verification: Claude Code does exactly path.replace("/", "-")
        with NO lstrip. Verify our code produces the same result.
        """
        session_id = "test-session-abc123"
        worktree_path = f"/data/sessions/{session_id}/worktree"

        # This is what Claude Code does internally:
        claude_code_encoding = worktree_path.replace("/", "-")
        assert claude_code_encoding == "-data-sessions-test-session-abc123-worktree"
        assert claude_code_encoding.startswith("-"), "Claude Code encoding must start with dash"

        # Now verify our fix matches
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=worktree_path,
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        # Old pod used /local/ prefix
        old_dir = projects_dir / f"-local-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        assert (projects_dir / claude_code_encoding).exists()

    def test_old_buggy_encoding_would_be_wrong(self, tmp_path):
        """
        Explicitly demonstrate the old buggy encoding (lstrip) differs from
        the correct encoding. This documents the bug for posterity.
        """
        worktree_path = "/data/sessions/abc/worktree"

        # Old buggy encoding:
        buggy_encoded = worktree_path.lstrip("/").replace("/", "-")
        assert buggy_encoded == "data-sessions-abc-worktree"

        # Correct encoding:
        correct_encoded = worktree_path.replace("/", "-")
        assert correct_encoded == "-data-sessions-abc-worktree"

        # They differ:
        assert buggy_encoded != correct_encoded
        assert not correct_encoded.startswith("d")  # doesn't start with 'd'
        assert correct_encoded.startswith("-")       # starts with '-'


# ============================================================================
# Test Group 2: _fix_claude_project_dir_after_s3_restore — Rename Logic
# ============================================================================


@pytest.mark.unit
class TestFixClaudeProjectDirRename:
    """Tests for the core rename logic of _fix_claude_project_dir_after_s3_restore."""

    @pytest.fixture
    def session_manager(self, tmp_path):
        return _make_session_manager(tmp_path)

    def test_renames_project_dir_on_path_mismatch(self, session_manager, tmp_path):
        """When worktree path differs from original pod, renames the project dir."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path="/data/sessions/test-session-abc123/worktree",
        )

        # Simulate old pod had worktree at /local/sessions/test-session-abc123/worktree
        # Claude Code on old pod would have encoded it as:
        # /local/sessions/test-session-abc123/worktree -> -local-sessions-test-session-abc123-worktree
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_encoded = "-local-sessions-test-session-abc123-worktree"
        old_dir = projects_dir / old_encoded
        old_dir.mkdir(parents=True)
        (old_dir / "conversation.json").write_text('{"test": true}')

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Old dir should be gone, new dir should exist with leading dash
        expected_encoded = "-data-sessions-test-session-abc123-worktree"
        new_dir = projects_dir / expected_encoded
        assert not old_dir.exists()
        assert new_dir.exists()
        assert (new_dir / "conversation.json").read_text() == '{"test": true}'

    def test_no_op_when_path_already_matches(self, session_manager, tmp_path):
        """When project dir already matches current worktree path, do nothing."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path="/data/sessions/test-session-abc123/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        # Correct encoding with leading dash
        expected_encoded = "-data-sessions-test-session-abc123-worktree"
        matching_dir = projects_dir / expected_encoded
        matching_dir.mkdir(parents=True)
        (matching_dir / "data.json").write_text("ok")

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should still exist unchanged
        assert matching_dir.exists()
        assert (matching_dir / "data.json").read_text() == "ok"

    def test_renames_cross_pod_data_to_local(self, session_manager, tmp_path):
        """Simulate /data -> /local cross-pod migration."""
        session_id = "test-session-42"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/local/sessions/{session_id}/worktree",
        )

        # Old pod used /data
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-data-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)
        (old_dir / "conv.json").write_text("data")

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        expected = projects_dir / f"-local-sessions-{session_id}-worktree"
        assert expected.exists()
        assert not old_dir.exists()
        assert (expected / "conv.json").read_text() == "data"

    def test_preserves_all_files_in_renamed_dir(self, session_manager, tmp_path):
        """Renaming preserves all files and subdirectories."""
        session_id = "test-session-abc123"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/new/sessions/{session_id}/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-old-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        # Create nested structure
        (old_dir / "conversation.json").write_text("conv")
        (old_dir / "settings.json").write_text("settings")
        subdir = old_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested")

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        new_dir = projects_dir / f"-new-sessions-{session_id}-worktree"
        assert new_dir.exists()
        assert (new_dir / "conversation.json").read_text() == "conv"
        assert (new_dir / "settings.json").read_text() == "settings"
        assert (new_dir / "subdir" / "nested_file.txt").read_text() == "nested"

    def test_skips_when_no_session_dir(self, session_manager):
        """When session_dir is None, returns early without error."""
        state = _make_session_state(session_dir=None, worktree_path="/some/path")
        # Should not raise
        session_manager._fix_claude_project_dir_after_s3_restore(state)

    def test_skips_when_no_worktree_path(self, session_manager, tmp_path):
        """When worktree_path is None, returns early without error."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=None,
        )
        # Should not raise
        session_manager._fix_claude_project_dir_after_s3_restore(state)

    def test_skips_when_projects_dir_missing(self, session_manager, tmp_path):
        """When claude-config/projects doesn't exist, returns early."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path="/data/sessions/test/worktree",
        )
        # Create session_dir but NOT projects subdir
        Path(state.session_dir).mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)
        # No error

    def test_skips_when_no_project_directories(self, session_manager, tmp_path):
        """When projects/ is empty, returns early."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path="/data/sessions/test/worktree",
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        projects_dir.mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)
        # No error, still empty
        assert list(projects_dir.iterdir()) == []

    def test_skips_when_multiple_ambiguous_directories(self, session_manager, tmp_path):
        """When multiple dirs exist with no identifiable main worktree, skip safely."""
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path="/data/sessions/test/worktree",
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        # Two unrelated dirs — neither contains session_id, so no candidate found
        (projects_dir / "dir-one").mkdir(parents=True)
        (projects_dir / "dir-two").mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Both dirs should still exist (not renamed)
        assert (projects_dir / "dir-one").exists()
        assert (projects_dir / "dir-two").exists()

    def test_renames_main_worktree_among_many_dirs(self, session_manager, tmp_path):
        """Bug 2 fix: with 11 dirs (main + agents + corrupted), renames only the main one."""
        session_id = "887d4000-80e0-41ab-ad9d-ec153a172859"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"

        # Old-pod main worktree dir (needs rename)
        old_main = f"-local-sessions-{session_id}-worktree"
        (projects_dir / old_main).mkdir(parents=True)
        (projects_dir / old_main / "conversation.jsonl").write_text("{}")

        # Base repo dir
        (projects_dir / "-data-repos-vllm").mkdir(parents=True)

        # Agent worktree dirs (should NOT be renamed)
        agent_dirs = [
            f"-data-sessions-{session_id}-worktree--claude-worktrees-op001-triton-gemm",
            f"-data-sessions-{session_id}-worktree--claude-worktrees-op002-fused-rmsnorm",
            f"-data-sessions-{session_id}-worktree--claude-worktrees-op003-cuda-thin-gemm",
        ]
        for d in agent_dirs:
            (projects_dir / d).mkdir(parents=True)

        # Corrupted dirs (git stdout as dirname)
        corrupted_dirs = [
            "Preparing-worktree--new-branch--session-fa88376f-ammo-agent-a85d60d6---HEAD-is-now-at-91eea7233",
            "Preparing-worktree--new-branch--session-fa88376f-ammo-agent-ac1891d4---HEAD-is-now-at-91eea7233",
        ]
        for d in corrupted_dirs:
            (projects_dir / d).mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Main dir renamed to new encoding
        expected = f"-data-sessions-{session_id}-worktree"
        assert (projects_dir / expected).exists()
        assert (projects_dir / expected / "conversation.jsonl").read_text() == "{}"
        # Old main dir gone
        assert not (projects_dir / old_main).exists()
        # Agent dirs untouched
        for d in agent_dirs:
            assert (projects_dir / d).exists()
        # Corrupted dirs untouched
        for d in corrupted_dirs:
            assert (projects_dir / d).exists()
        # Base repo dir untouched
        assert (projects_dir / "-data-repos-vllm").exists()

    def test_skips_when_expected_dir_already_exists_among_many(self, session_manager, tmp_path):
        """When expected dir already exists alongside other dirs, no-op."""
        session_id = "test-session-abc123"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"

        # Expected dir already exists
        expected = f"-data-sessions-{session_id}-worktree"
        (projects_dir / expected).mkdir(parents=True)
        # Plus some agent dirs
        (projects_dir / f"-data-sessions-{session_id}-worktree--claude-worktrees-op001").mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Both still exist, no changes
        assert (projects_dir / expected).exists()
        assert (projects_dir / f"-data-sessions-{session_id}-worktree--claude-worktrees-op001").exists()

    def test_skips_files_in_projects_dir_only_counts_dirs(self, session_manager, tmp_path):
        """Files in projects/ should not be counted as project directories."""
        session_id = "test-session-abc123"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        projects_dir.mkdir(parents=True)

        # Create a file (not a directory) — should be ignored
        (projects_dir / "some-file.json").write_text("{}")
        # Create one actual project dir that needs renaming
        old_dir = projects_dir / f"-old-sessions-{session_id}-worktree"
        old_dir.mkdir()
        (old_dir / "data.json").write_text("ok")

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Should rename the one directory
        expected = projects_dir / f"-data-sessions-{session_id}-worktree"
        assert expected.exists()
        assert not old_dir.exists()


# ============================================================================
# Test Group 3: .claude.json Update During Rename
# ============================================================================


@pytest.mark.unit
class TestFixClaudeProjectDirJsonUpdate:
    """Tests for .claude.json project key updates during cross-host S3 restore."""

    @pytest.fixture
    def session_manager(self, tmp_path):
        return _make_session_manager(tmp_path)

    def test_updates_claude_json_project_key(self, session_manager, tmp_path):
        """Renames the project key in .claude.json to match new worktree path."""
        session_id = "test-session-42"
        new_worktree = f"/data/sessions/{session_id}/worktree"
        old_worktree = f"/local/sessions/{session_id}/worktree"

        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=new_worktree,
        )

        # Set up project dir with old encoded path (using correct leading-dash encoding)
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_encoded = old_worktree.replace("/", "-")
        old_dir = projects_dir / old_encoded
        old_dir.mkdir(parents=True)

        # Set up .claude.json with old project key
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json = {
            "theme": "dark",
            "hasCompletedOnboarding": True,
            "projects": {
                old_worktree: {
                    "hasTrustDialogAccepted": True,
                    "allowedTools": [],
                }
            }
        }
        claude_json_path.write_text(json.dumps(claude_json))

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Verify .claude.json was updated
        updated_json = json.loads(claude_json_path.read_text())
        assert new_worktree in updated_json["projects"]
        assert old_worktree not in updated_json["projects"]
        assert updated_json["projects"][new_worktree]["hasTrustDialogAccepted"] is True

    def test_preserves_non_project_keys_in_claude_json(self, session_manager, tmp_path):
        """Other keys in .claude.json (theme, onboarding, etc.) are preserved."""
        session_id = "test-sess-77"
        new_worktree = f"/data/sessions/{session_id}/worktree"
        old_worktree = f"/local/sessions/{session_id}/worktree"

        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=new_worktree,
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_encoded = old_worktree.replace("/", "-")
        old_dir = projects_dir / old_encoded
        old_dir.mkdir(parents=True)

        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json = {
            "theme": "dark",
            "hasCompletedOnboarding": True,
            "customSetting": "preserved",
            "projects": {
                old_worktree: {"hasTrustDialogAccepted": True}
            }
        }
        claude_json_path.write_text(json.dumps(claude_json))

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        updated_json = json.loads(claude_json_path.read_text())
        assert updated_json["theme"] == "dark"
        assert updated_json["hasCompletedOnboarding"] is True
        assert updated_json["customSetting"] == "preserved"

    def test_skips_json_update_when_no_claude_json(self, session_manager, tmp_path):
        """When .claude.json doesn't exist, still renames dir but skips JSON update."""
        session_id = "test-sess-88"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-local-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should be renamed with leading dash
        expected_encoded = f"-data-sessions-{session_id}-worktree"
        assert (projects_dir / expected_encoded).exists()
        # No .claude.json should exist
        assert not (Path(state.session_dir) / "claude-config" / ".claude.json").exists()

    def test_handles_corrupt_claude_json(self, session_manager, tmp_path):
        """When .claude.json contains invalid JSON, rename dir but don't crash."""
        session_id = "test-sess-99"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-local-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        # Write corrupt JSON
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json_path.write_text("{invalid json!!!")

        # Should not raise
        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should still be renamed
        expected_encoded = f"-data-sessions-{session_id}-worktree"
        assert (projects_dir / expected_encoded).exists()

    def test_skips_json_update_when_no_projects_key(self, session_manager, tmp_path):
        """When .claude.json has no 'projects' key, skip JSON update."""
        session_id = "test-sess-100"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-local-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        # Write JSON without projects key
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json_path.write_text(json.dumps({"theme": "dark"}))

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir renamed, JSON unchanged (no projects key)
        expected_encoded = f"-data-sessions-{session_id}-worktree"
        assert (projects_dir / expected_encoded).exists()
        updated = json.loads(claude_json_path.read_text())
        assert "projects" not in updated

    def test_preserves_existing_matching_project_key(self, session_manager, tmp_path):
        """When .claude.json already has the correct project key, leave it."""
        worktree_path = "/data/sessions/test/worktree"
        state = _make_session_state(
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=worktree_path,
        )

        # Dir already matches (with leading dash)
        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        expected_encoded = worktree_path.replace("/", "-")  # -data-sessions-test-worktree
        matching_dir = projects_dir / expected_encoded
        matching_dir.mkdir(parents=True)

        # JSON has the correct key
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json = {
            "projects": {
                worktree_path: {"hasTrustDialogAccepted": True}
            }
        }
        claude_json_path.write_text(json.dumps(claude_json))

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # JSON should be unchanged
        updated = json.loads(claude_json_path.read_text())
        assert worktree_path in updated["projects"]

    def test_handles_json_with_projects_as_non_dict(self, session_manager, tmp_path):
        """When .claude.json has 'projects' as a non-dict value, skip gracefully."""
        session_id = "test-sess-111"
        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=f"/data/sessions/{session_id}/worktree",
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_dir = projects_dir / f"-old-sessions-{session_id}-worktree"
        old_dir.mkdir(parents=True)

        # "projects" is a list, not a dict
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json_path.write_text(json.dumps({"projects": ["not", "a", "dict"]}))

        # Should not raise
        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should be renamed, JSON not modified
        expected = f"-data-sessions-{session_id}-worktree"
        assert (projects_dir / expected).exists()

    def test_json_update_with_multiple_project_keys(self, session_manager, tmp_path):
        """When .claude.json has multiple project keys, only rename the old one."""
        session_id = "test-sess-222"
        new_worktree = f"/data/sessions/{session_id}/worktree"
        old_worktree = f"/local/sessions/{session_id}/worktree"

        state = _make_session_state(
            session_id=session_id,
            session_dir=str(tmp_path / "session_dir"),
            worktree_path=new_worktree,
        )

        projects_dir = Path(state.session_dir) / "claude-config" / "projects"
        old_encoded = old_worktree.replace("/", "-")
        old_dir = projects_dir / old_encoded
        old_dir.mkdir(parents=True)

        # .claude.json has old key AND new key already (edge case)
        claude_json_path = Path(state.session_dir) / "claude-config" / ".claude.json"
        claude_json = {
            "projects": {
                old_worktree: {"hasTrustDialogAccepted": True},
                new_worktree: {"hasTrustDialogAccepted": True},
            }
        }
        claude_json_path.write_text(json.dumps(claude_json))

        session_manager._fix_claude_project_dir_after_s3_restore(state)

        # Dir should be renamed; JSON has no old_keys to rename
        # (since new_worktree already exists, old_keys would be empty list)
        expected = new_worktree.replace("/", "-")
        assert (projects_dir / expected).exists()


# ============================================================================
# Test Group 4: SessionS3Storage.restore_session_from_s3
# ============================================================================


@pytest.mark.unit
class TestRestoreSessionFromS3:
    """Tests for SessionS3Storage.restore_session_from_s3 orchestration."""

    @pytest.fixture
    def storage(self):
        """Create SessionS3Storage with S3 enabled."""
        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            s3 = SessionS3Storage()
        return s3

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """When S3 is not configured, returns None."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure SESSION_S3_BUCKET is not set
            os.environ.pop("SESSION_S3_BUCKET", None)
            storage = SessionS3Storage()

        result = await storage.restore_session_from_s3(
            "test-session", Path("/tmp/test/worktree")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_metadata(self, storage):
        """When session metadata is not in S3, returns None."""
        storage.load_session_metadata = AsyncMock(return_value=None)

        result = await storage.restore_session_from_s3(
            "missing-session", Path("/tmp/test/worktree")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_worktree_restore_fails(self, storage):
        """When worktree restore from S3 fails, returns None."""
        mock_state = _make_session_state()
        storage.load_session_metadata = AsyncMock(return_value=mock_state)
        storage.restore_worktree_from_s3 = AsyncMock(return_value=False)

        result = await storage.restore_session_from_s3(
            "test-session", Path("/tmp/test/worktree")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_state_with_updated_worktree_path(self, storage):
        """On successful restore, returns state with updated worktree_path."""
        mock_state = _make_session_state(
            worktree_path="/old/pod/worktree"
        )
        storage.load_session_metadata = AsyncMock(return_value=mock_state)
        storage.restore_worktree_from_s3 = AsyncMock(return_value=True)
        storage.restore_cli_state_from_s3 = AsyncMock(return_value=True)

        target_path = Path("/new/pod/sessions/test-session/worktree")
        result = await storage.restore_session_from_s3(
            "test-session-abc123", target_path
        )

        assert result is not None
        assert result.worktree_path == str(target_path)

    @pytest.mark.asyncio
    async def test_calls_restore_cli_state_after_worktree(self, storage):
        """Verify that CLI state restore is called after worktree restore."""
        mock_state = _make_session_state()
        storage.load_session_metadata = AsyncMock(return_value=mock_state)
        storage.restore_worktree_from_s3 = AsyncMock(return_value=True)
        storage.restore_cli_state_from_s3 = AsyncMock(return_value=True)

        target_path = Path("/data/sessions/test/worktree")
        await storage.restore_session_from_s3("test-session-abc123", target_path)

        storage.restore_cli_state_from_s3.assert_called_once_with(
            "test-session-abc123", target_path
        )


# ============================================================================
# Test Group 5: Resume Session Integration with S3 Restore
# ============================================================================


@pytest.mark.unit
class TestResumeSessionWithS3Restore:
    """Tests for resume_session flow when session needs S3 restoration."""

    @pytest.fixture
    def session_manager(self, tmp_path):
        """Create SessionManager with mocked dependencies for resume testing."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        mock_storage = MagicMock()
        mock_storage.enabled = True
        mock_storage.get_s3_last_modified = AsyncMock(return_value=time.time())

        mock_worktree = MagicMock()
        mock_gpu = MagicMock()
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(return_value=[0])
        mock_gpu.release_gpus_for_session = MagicMock()

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = True
        mock_terminal.start_terminal_with_command = AsyncMock(return_value=8001)

        mock_cli = MagicMock()
        mock_cli.get_cli_command.return_value = "claude --continue"

        mock_inactivity = MagicMock()

        with patch("orchestration.session_manager.get_worktree_manager", return_value=mock_worktree), \
             patch("orchestration.session_manager.GPUResourceManager", return_value=mock_gpu), \
             patch("orchestration.session_manager.get_terminal_manager", return_value=mock_terminal), \
             patch("orchestration.session_manager.get_cli_tool_manager", return_value=mock_cli), \
             patch("orchestration.session_manager.get_inactivity_monitor", return_value=mock_inactivity), \
             patch("orchestration.session_manager.get_session_storage", return_value=mock_storage):

            mgr = SessionManager(
                sessions_dir=str(sessions_dir),
                session_storage=mock_storage,
                gpu_manager=mock_gpu,
                terminal_manager=mock_terminal,
                cli_tool_manager=mock_cli,
                inactivity_monitor=mock_inactivity,
                worktree_manager=mock_worktree,
            )

        return mgr

    @pytest.mark.asyncio
    async def test_resume_restores_from_s3_when_session_not_in_memory(
        self, session_manager, tmp_path
    ):
        """When session is not in memory, attempt S3 restore."""
        session_id = "s3-session-123"

        # Create a worktree on disk so the "worktree exists" check passes
        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(exist_ok=True)

        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        # Call resume
        response = await session_manager.resume_session(session_id)

        # Should have attempted S3 restore
        session_manager.session_storage.restore_session_from_s3.assert_called_once()
        assert response.session_id == session_id
        assert response.status == "active"

    @pytest.mark.asyncio
    async def test_resume_raises_when_session_not_found_anywhere(
        self, session_manager
    ):
        """When session not in memory and S3 restore returns None, raise SessionError."""
        from orchestration.session_manager import SessionError

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=None
        )

        with pytest.raises(SessionError, match="not found"):
            await session_manager.resume_session("nonexistent-session")

    @pytest.mark.asyncio
    async def test_resume_calls_fix_project_dir_on_s3_restore(
        self, session_manager, tmp_path
    ):
        """After S3 restore, _fix_claude_project_dir_after_s3_restore should be called."""
        session_id = "cross-pod-session"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(exist_ok=True)

        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        with patch.object(
            session_manager, "_fix_claude_project_dir_after_s3_restore"
        ) as mock_fix:
            await session_manager.resume_session(session_id)
            mock_fix.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_does_not_call_fix_for_local_session(
        self, session_manager, tmp_path
    ):
        """For sessions already in memory (not S3 restored), don't call fix."""
        session_id = "local-session"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id

        state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
        )
        session_manager._sessions[session_id] = state

        with patch.object(
            session_manager, "_fix_claude_project_dir_after_s3_restore"
        ) as mock_fix:
            await session_manager.resume_session(session_id)
            mock_fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_forces_paused_status_on_s3_restore(
        self, session_manager, tmp_path
    ):
        """S3-restored session should be forced to PAUSED status before resume logic."""
        session_id = "active-on-old-pod"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id

        # State from S3 says ACTIVE (stale from old pod)
        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            status=SessionStatus.ACTIVE,  # Stale status
            requested_gpu_count=0,
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        response = await session_manager.resume_session(session_id)

        # Should have gone through resume flow (not early return for "already active")
        # The response should indicate an active session
        assert response.status == "active"
        # The session should be in memory
        assert session_id in session_manager._sessions

    @pytest.mark.asyncio
    async def test_resume_end_to_end_cross_pod_with_project_dir_fix(
        self, session_manager, tmp_path
    ):
        """
        Full integration: S3 restore -> project dir fix -> terminal start.

        Simulates the complete cross-host S3 restore flow with actual filesystem
        operations (not mocked _fix_claude_project_dir_after_s3_restore).
        """
        session_id = "e2e-cross-pod"

        # Set up worktree dir
        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id

        # Simulate S3 restore that set worktree_path to local path
        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            status=SessionStatus.PAUSED,
            requested_gpu_count=0,
        )

        # Create the claude-config with old pod's project dir encoding
        # Old pod used /old/pod/sessions/e2e-cross-pod/worktree
        projects_dir = session_dir / "claude-config" / "projects"
        old_encoded = "-old-pod-sessions-e2e-cross-pod-worktree"
        old_dir = projects_dir / old_encoded
        old_dir.mkdir(parents=True)
        (old_dir / "conversation.json").write_text('{"messages": []}')

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )
        session_manager.session_storage.get_s3_last_modified = AsyncMock(
            return_value=time.time()
        )

        # Do NOT mock _fix_claude_project_dir_after_s3_restore — let it run
        response = await session_manager.resume_session(session_id)

        # Verify the project dir was renamed correctly
        expected_encoded = str(worktree_dir).replace("/", "-")
        new_dir = projects_dir / expected_encoded
        assert new_dir.exists(), (
            f"Expected project dir '{expected_encoded}' not found after resume. "
            f"Contents: {[d.name for d in projects_dir.iterdir() if d.is_dir()]}"
        )
        assert not old_dir.exists()
        assert (new_dir / "conversation.json").read_text() == '{"messages": []}'
        assert response.status == "active"


# ============================================================================
# Test Group 6: Ownership Validation During Cross-Pod Resume
# ============================================================================


@pytest.mark.unit
class TestCrossPodResumeOwnership:
    """Tests for ownership validation during S3-restored resume."""

    @pytest.fixture
    def session_manager(self, tmp_path):
        """Create SessionManager with mocked dependencies."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        mock_storage = MagicMock()
        mock_storage.enabled = True

        mock_terminal = MagicMock()
        mock_terminal.is_available.return_value = True
        mock_terminal.start_terminal_with_command = AsyncMock(return_value=8001)

        mock_cli = MagicMock()
        mock_cli.get_cli_command.return_value = "claude --continue"

        mock_gpu = MagicMock()
        mock_gpu.acquire_gpus_for_session_async = AsyncMock(return_value=[])
        mock_gpu.release_gpus_for_session = MagicMock()

        with patch("orchestration.session_manager.get_worktree_manager") as mock_wm, \
             patch("orchestration.session_manager.GPUResourceManager", return_value=mock_gpu), \
             patch("orchestration.session_manager.get_terminal_manager", return_value=mock_terminal), \
             patch("orchestration.session_manager.get_cli_tool_manager", return_value=mock_cli), \
             patch("orchestration.session_manager.get_inactivity_monitor") as mock_im, \
             patch("orchestration.session_manager.get_session_storage", return_value=mock_storage):

            mgr = SessionManager(
                sessions_dir=str(sessions_dir),
                session_storage=mock_storage,
                gpu_manager=mock_gpu,
                terminal_manager=mock_terminal,
                cli_tool_manager=mock_cli,
                worktree_manager=mock_wm.return_value,
            )

        return mgr

    @pytest.mark.asyncio
    async def test_resume_rejects_wrong_owner_after_s3_restore(
        self, session_manager, tmp_path
    ):
        """After S3 restore, reject resume if owner_id doesn't match stored owner."""
        from orchestration.session_manager import SessionError

        session_id = "owned-session"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)

        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(tmp_path / "sessions" / session_id),
            owner_id="owner-A",
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        with pytest.raises(SessionError, match="not found"):
            await session_manager.resume_session(
                session_id, owner_id="owner-B"
            )

    @pytest.mark.asyncio
    async def test_resume_allows_matching_owner_after_s3_restore(
        self, session_manager, tmp_path
    ):
        """After S3 restore, allow resume if owner_id matches stored owner."""
        session_id = "owned-session"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id

        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            owner_id="owner-A",
            requested_gpu_count=0,
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        response = await session_manager.resume_session(
            session_id, owner_id="owner-A"
        )

        assert response.session_id == session_id

    @pytest.mark.asyncio
    async def test_resume_allows_no_owner_id_for_legacy_sessions(
        self, session_manager, tmp_path
    ):
        """Legacy sessions (owner_id=None) should be resumable by anyone."""
        session_id = "legacy-session"

        worktree_dir = tmp_path / "sessions" / session_id / "worktree"
        worktree_dir.mkdir(parents=True)
        session_dir = tmp_path / "sessions" / session_id

        restored_state = _make_session_state(
            session_id=session_id,
            worktree_path=str(worktree_dir),
            session_dir=str(session_dir),
            owner_id=None,
            requested_gpu_count=0,
        )

        session_manager.session_storage.restore_session_from_s3 = AsyncMock(
            return_value=restored_state
        )

        # Any owner_id should be allowed
        response = await session_manager.resume_session(
            session_id, owner_id="some-client"
        )

        assert response.session_id == session_id
