# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_download_scrub.py
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.session_state import SessionS3Storage


def _make_storage(bucket: str = "test-bucket", prefix: str = "sessions") -> SessionS3Storage:
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = bucket
    storage.prefix = prefix
    storage.ttl_days = 30
    return storage


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fork_token_scrubbed_from_download_session_json(tmp_path):
    """The encrypted fork token must be stripped from session.json before the
    download archive is created, while session.json itself survives."""
    storage = _make_storage()
    session_id = "fork-scrub-001"

    temp_download_dir = tmp_path / "download_work"
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    (temp_download_dir / "session.json").write_text(json.dumps({
        "session_id": session_id,
        "vllm_fork_url": "https://github.com/u/vllm.git",
        "vllm_fork_token_encrypted": "gAAAAA-super-secret-encrypted-token",
    }))

    captured = {}

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    async def mock_exec(*args, **kwargs):
        return mock_proc

    def mock_make_archive(base_name, fmt, root_dir):
        meta = Path(root_dir) / "session.json"
        captured["session_json_exists"] = meta.exists()
        captured["data"] = json.loads(meta.read_text())
        zip_path = base_name + ".zip"
        Path(zip_path).write_text("fake zip")
        return zip_path

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
         patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
         patch("shutil.make_archive", side_effect=mock_make_archive), \
         patch("os.path.getsize", return_value=1024):
        await storage.create_download_archive(session_id)

    # session.json survives (UI reads it) ...
    assert captured["session_json_exists"] is True
    # ... but the encrypted token field is scrubbed to None.
    assert captured["data"]["vllm_fork_token_encrypted"] is None
    # Non-secret fork metadata is preserved.
    assert captured["data"]["vllm_fork_url"] == "https://github.com/u/vllm.git"
    # The raw secret string never appears in the file content.
    assert "super-secret-encrypted-token" not in json.dumps(captured["data"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plain_session_json_unmodified(tmp_path):
    """A non-fork session.json (no token field) is left intact."""
    storage = _make_storage()
    session_id = "plain-scrub-002"

    temp_download_dir = tmp_path / "download_work"
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    (temp_download_dir / "session.json").write_text(json.dumps({
        "session_id": session_id,
    }))

    captured = {}
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    async def mock_exec(*args, **kwargs):
        return mock_proc

    def mock_make_archive(base_name, fmt, root_dir):
        captured["data"] = json.loads((Path(root_dir) / "session.json").read_text())
        zip_path = base_name + ".zip"
        Path(zip_path).write_text("fake zip")
        return zip_path

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
         patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
         patch("shutil.make_archive", side_effect=mock_make_archive), \
         patch("os.path.getsize", return_value=1024):
        await storage.create_download_archive(session_id)

    assert captured["data"] == {"session_id": session_id}


@pytest.mark.unit
def test_sanitizer_names_the_token_field():
    """Source-level guard: the sanitizer references the token-bearing field so
    the scrub can never silently regress to a no-op rename."""
    import orchestration.session_state as ss
    src = Path(ss.__file__).read_text()
    assert "vllm_fork_token_encrypted" in src
    assert "session.json" in src
