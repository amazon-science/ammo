# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the GET /api/changelog endpoint.

Tests that the endpoint parses ai_cli_session/.claude/VERSION and returns
structured version+changelog data, and handles edge cases gracefully.
"""

import sys
import json
import importlib
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_VERSION_CONTENT = """\
version: 1.2.0

## Changelog

### 1.2.0 (2026-04-03)
- Accuracy failure persistence
- Transcript monitor dual role

### 1.1.0 (2026-04-02)
- Redesign Gate 5.1b

### 1.0.0 (2026-03-31)
- Initial versioned release
"""

VERSION_ONLY_CONTENT = "version: 2.0.0\n"

MALFORMED_VERSION_CONTENT = """\
garbage text
- some change
"""

EMPTY_CONTENT = ""


def _make_client_with_version_file(monkeypatch, tmp_path, content):
    """
    Write a fake VERSION file and point SESSION_TEMPLATES_DIR to its parent.
    Returns a TestClient for the reloaded app.
    """
    # Replicate the path structure the endpoint expects:
    # SESSION_TEMPLATES_DIR / "claude/.claude/VERSION"
    version_dir = tmp_path / "claude" / ".claude"
    version_dir.mkdir(parents=True)
    version_file = version_dir / "VERSION"
    version_file.write_text(content)

    monkeypatch.setenv("SESSION_TEMPLATES_DIR", str(tmp_path))
    monkeypatch.delenv("AMMO_API_KEY", raising=False)

    import app as app_module
    importlib.reload(app_module)

    from starlette.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


def _make_client_no_version_file(monkeypatch, tmp_path):
    """Point SESSION_TEMPLATES_DIR to a directory with no VERSION file."""
    monkeypatch.setenv("SESSION_TEMPLATES_DIR", str(tmp_path))
    monkeypatch.delenv("AMMO_API_KEY", raising=False)

    import app as app_module
    importlib.reload(app_module)

    from starlette.testclient import TestClient
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChangelogEndpoint:
    """Tests for GET /api/changelog."""

    def test_changelog_happy_path(self, monkeypatch, tmp_path):
        """Well-formed VERSION with 3 sections returns HTTP 200 with correct structure."""
        client = _make_client_with_version_file(monkeypatch, tmp_path, VALID_VERSION_CONTENT)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] == "1.2.0"
        assert len(data["entries"]) == 3

        e0 = data["entries"][0]
        assert e0["version"] == "1.2.0"
        assert e0["date"] == "2026-04-03"
        assert "Accuracy failure persistence" in e0["changes"]
        assert "Transcript monitor dual role" in e0["changes"]

        e1 = data["entries"][1]
        assert e1["version"] == "1.1.0"
        assert e1["date"] == "2026-04-02"
        assert "Redesign Gate 5.1b" in e1["changes"]

        e2 = data["entries"][2]
        assert e2["version"] == "1.0.0"
        assert e2["date"] == "2026-03-31"
        assert "Initial versioned release" in e2["changes"]

    def test_changelog_missing_file(self, monkeypatch, tmp_path):
        """Non-existent VERSION file returns HTTP 200 with null version and empty entries."""
        client = _make_client_no_version_file(monkeypatch, tmp_path)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] is None
        assert data["entries"] == []

    def test_changelog_malformed_version_line(self, monkeypatch, tmp_path):
        """First line is garbage text → version is null, entries empty."""
        client = _make_client_with_version_file(monkeypatch, tmp_path, MALFORMED_VERSION_CONTENT)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] is None
        assert data["entries"] == []

    def test_changelog_empty_file(self, monkeypatch, tmp_path):
        """Empty VERSION file returns HTTP 200 with null version and empty entries."""
        client = _make_client_with_version_file(monkeypatch, tmp_path, EMPTY_CONTENT)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] is None
        assert data["entries"] == []

    def test_changelog_version_only_no_sections(self, monkeypatch, tmp_path):
        """VERSION file with version header only, no ### sections → entries is empty."""
        client = _make_client_with_version_file(monkeypatch, tmp_path, VERSION_ONLY_CONTENT)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] == "2.0.0"
        assert data["entries"] == []

    def test_changelog_section_with_no_bullets(self, monkeypatch, tmp_path):
        """A ### section with no bullet items → entry present with changes: []."""
        content = """\
version: 3.0.0

### 3.0.0 (2026-05-01)

### 2.0.0 (2026-04-01)
- First change
"""
        client = _make_client_with_version_file(monkeypatch, tmp_path, content)

        resp = client.get("/api/changelog")
        assert resp.status_code == 200

        data = resp.json()
        assert data["version"] == "3.0.0"
        assert len(data["entries"]) == 2

        # Section with no bullets still appears, with empty changes list
        e0 = data["entries"][0]
        assert e0["version"] == "3.0.0"
        assert e0["date"] == "2026-05-01"
        assert e0["changes"] == []

        e1 = data["entries"][1]
        assert e1["version"] == "2.0.0"
        assert e1["changes"] == ["First change"]
