# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for ammo_version field on SessionInfo and SessionState.

Tests that the ammo_version field is:
- Present on SessionInfo (Pydantic model), defaulting to None
- Present on SessionState (dataclass), defaulting to None
- Propagated from SessionState → SessionInfo via to_session_info()
- Serialized in SessionState.to_dict()
- Deserialized from SessionState.from_dict() (with and without the key)

Follows the has_report field precedent.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionInfo,
    SessionState,
    SessionStatus,
    CLIToolType,
)
from orchestration.session_manager import _read_ammo_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_info(**kwargs) -> SessionInfo:
    """Create a SessionInfo with required fields filled."""
    defaults = dict(
        session_id="test-session-001",
        status="active",
        cli_tool="claude",
        repo_name="vllm",
        branch="main",
        created_at=1700000000.0,
        last_accessed=1700000001.0,
    )
    defaults.update(kwargs)
    return SessionInfo(**defaults)


def _make_session_state(**kwargs) -> SessionState:
    """Create a SessionState with required fields filled."""
    defaults = dict(
        session_id="test-session-001",
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=1700000000.0,
        last_accessed=1700000001.0,
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


def _make_state_dict(**kwargs) -> dict:
    """Return a minimal dict suitable for SessionState.from_dict()."""
    defaults = dict(
        session_id="test-session-001",
        status="active",
        cli_tool="claude",
        repo_name="vllm",
        branch="main",
        created_at=1700000000.0,
        last_accessed=1700000001.0,
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSessionAmmoVersion:
    """Tests for the ammo_version field on session models."""

    def test_session_info_includes_ammo_version(self):
        """SessionInfo.dict() includes ammo_version when set."""
        info = _make_session_info(ammo_version="1.2.0")
        data = info.model_dump()
        assert "ammo_version" in data
        assert data["ammo_version"] == "1.2.0"

    def test_session_info_ammo_version_default_none(self):
        """SessionInfo without ammo_version arg defaults to None."""
        info = _make_session_info()
        assert info.ammo_version is None

    def test_session_state_to_session_info_propagates_version(self):
        """SessionState.to_session_info() propagates ammo_version."""
        state = _make_session_state(ammo_version="1.0.0")
        info = state.to_session_info()
        assert info.ammo_version == "1.0.0"

    def test_session_state_to_dict_includes_version(self):
        """SessionState.to_dict() includes ammo_version key."""
        state = _make_session_state(ammo_version="1.1.0")
        data = state.to_dict()
        assert "ammo_version" in data
        assert data["ammo_version"] == "1.1.0"

    def test_session_state_from_dict_reads_version(self):
        """SessionState.from_dict() reads ammo_version from dict."""
        d = _make_state_dict(ammo_version="1.1.0")
        state = SessionState.from_dict(d)
        assert state.ammo_version == "1.1.0"

    def test_session_state_from_dict_missing_version_defaults_none(self):
        """SessionState.from_dict() with no ammo_version key defaults to None."""
        d = _make_state_dict()
        assert "ammo_version" not in d  # confirm key is absent
        state = SessionState.from_dict(d)
        assert state.ammo_version is None

    def test_session_state_none_ammo_version_round_trips(self):
        """ammo_version=None serializes to dict with key present, round-trips via from_dict."""
        state = _make_session_state()  # no ammo_version → defaults to None
        data = state.to_dict()
        # Key must be present (not missing) so consumers can distinguish old sessions
        assert "ammo_version" in data
        assert data["ammo_version"] is None
        # Round-trip through from_dict
        restored = SessionState.from_dict(data)
        assert restored.ammo_version is None


# ---------------------------------------------------------------------------
# Edge-case probe: _read_ammo_version helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReadAmmoVersion:
    """Direct tests for the _read_ammo_version() helper in session_manager."""

    def test_reads_version_from_valid_file(self, monkeypatch, tmp_path):
        """Well-formed VERSION file returns the version string."""
        version_dir = tmp_path / "claude" / ".claude"
        version_dir.mkdir(parents=True)
        (version_dir / "VERSION").write_text("version: 1.2.0\n\n### 1.2.0 (2026-04-03)\n- change\n")
        monkeypatch.setenv("SESSION_TEMPLATES_DIR", str(tmp_path))
        assert _read_ammo_version() == "1.2.0"

    def test_empty_file_returns_none(self, monkeypatch, tmp_path):
        """Empty VERSION file (IndexError on splitlines()[0]) returns None gracefully."""
        version_dir = tmp_path / "claude" / ".claude"
        version_dir.mkdir(parents=True)
        (version_dir / "VERSION").write_text("")
        monkeypatch.setenv("SESSION_TEMPLATES_DIR", str(tmp_path))
        # splitlines() returns [] → [0] raises IndexError → caught → None
        assert _read_ammo_version() is None

    def test_missing_file_returns_none(self, monkeypatch, tmp_path):
        """Non-existent VERSION file returns None gracefully."""
        monkeypatch.setenv("SESSION_TEMPLATES_DIR", str(tmp_path))
        assert _read_ammo_version() is None
