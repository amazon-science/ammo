# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_ccache_s3.py
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.session_state import SessionS3Storage


def _fake_proc(returncode=0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b"", b""))
    return proc


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ccache_upload_uses_session_keyed_object(tmp_path, monkeypatch):
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"  # enabled is a derived property (bucket is not None)
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))
    (tmp_path / "ccache").mkdir()

    calls = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        return _fake_proc(0)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.sync_ccache_to_s3("sess-1")

    assert ok
    assert any("sess-1/ccache.tar" in c for c in calls)
    assert any("CCACHE" in c or "ccache" in c for c in calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ccache_upload_disabled_returns_false(tmp_path, monkeypatch):
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = None  # enabled property is False when bucket is None
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))
    (tmp_path / "ccache").mkdir()

    ok = await storage.sync_ccache_to_s3("sess-1")
    assert ok is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ccache_upload_missing_dir_returns_false(tmp_path, monkeypatch):
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "does-not-exist"))

    ok = await storage.sync_ccache_to_s3("sess-1")
    assert ok is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ccache_restore_skips_when_object_missing(tmp_path, monkeypatch):
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))

    calls = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        # Simulate the `aws s3 ls` head probe returning non-zero (missing).
        return _fake_proc(1)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.restore_ccache_from_s3("sess-1")

    assert ok is False
    # Only the head probe should have run (no download/extract).
    assert any("aws s3 ls" in c for c in calls)
    assert not any("tar xf" in c for c in calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ccache_restore_downloads_when_object_exists(tmp_path, monkeypatch):
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))

    calls = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        return _fake_proc(0)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.restore_ccache_from_s3("sess-1")

    assert ok is True
    assert any("aws s3 ls" in c for c in calls)
    # Download stages the per-session object into a temp file first.
    assert any("aws s3 cp" in c and "sess-1/ccache.tar" in c for c in calls)
    # An extract still runs into the live ccache parent.
    assert any("tar xf" in c for c in calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_restore_uses_pipefail_and_staging(tmp_path, monkeypatch):
    """Restore runs its pipelines under /bin/bash with `set -o pipefail`, and
    downloads to a staging temp file (not a `-` stdin stream) before extracting."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))

    calls = []
    kwargs_seen = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        kwargs_seen.append(kwargs)
        return _fake_proc(0)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.restore_ccache_from_s3("sess-1")

    assert ok is True

    # The head probe is `aws s3 ls`; everything after it is a pipeline.
    head_idx = next(i for i, c in enumerate(calls) if "aws s3 ls" in c)
    pipeline_calls = calls[head_idx + 1:]
    assert pipeline_calls, "expected download/verify/extract calls after the head probe"

    # The download targets a real temp-file path, NOT a `-` stdin stream.
    dl_calls = [c for c in pipeline_calls if "aws s3 cp" in c]
    assert dl_calls, "expected an `aws s3 cp` download"
    for c in dl_calls:
        assert "sess-1/ccache.tar" in c
        # Not streaming directly into a pipe: the cp target is a file, not `-`.
        assert not c.rstrip().endswith("-")
        assert "| tar" not in c and "| gzip" not in c and "| pigz" not in c

    # Both the verify and extract pipelines must run under pipefail + bash.
    piped = [c for c in pipeline_calls if "| tar" in c or "tar tf" in c or "tar xf" in c]
    assert piped, "expected verify/extract pipelines"
    for c in piped:
        assert "set -o pipefail" in c
    # All pipeline subprocesses are launched with executable=/bin/bash.
    for kw in kwargs_seen[head_idx + 1:]:
        assert kw.get("executable") == "/bin/bash"

    # An archive verification (tar tf / list) runs before the extract.
    verify_idx = next(i for i, c in enumerate(calls) if "tar tf" in c)
    extract_idx = next(i for i, c in enumerate(calls) if "tar xf" in c)
    assert verify_idx < extract_idx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_restore_skips_on_corrupt_archive(tmp_path, monkeypatch):
    """If the archive verification step fails, no extract runs and restore False."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))

    calls = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        # head probe (ls) and download (cp) succeed; the verify (tar tf) fails.
        if "tar tf" in cmd:
            return _fake_proc(1)
        return _fake_proc(0)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.restore_ccache_from_s3("sess-1")

    assert ok is False
    # Verification was attempted...
    assert any("tar tf" in c for c in calls)
    # ...but no extraction into the live cache happened.
    assert not any("tar xf" in c for c in calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_ccache_uses_pipefail_and_bash(tmp_path, monkeypatch):
    """The upload pipeline also runs under /bin/bash with `set -o pipefail`."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"
    monkeypatch.setenv("CCACHE_DIR", str(tmp_path / "ccache"))
    (tmp_path / "ccache").mkdir()

    calls = []
    kwargs_seen = []

    async def fake_shell(cmd, *args, **kwargs):
        calls.append(cmd)
        kwargs_seen.append(kwargs)
        return _fake_proc(0)

    with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
        ok = await storage.sync_ccache_to_s3("sess-1")

    assert ok
    assert any("set -o pipefail" in c for c in calls)
    assert all(kw.get("executable") == "/bin/bash" for kw in kwargs_seen)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_session_to_s3_includes_ccache_only_for_fork(tmp_path, monkeypatch):
    """sync_session_to_s3 appends sync_ccache_to_s3 only when state has a fork URL
    AND include_ccache=True is passed (the pause path). The default checkpoint call
    never uploads ccache, even for a fork session."""
    from shared.session_models import SessionState, SessionStatus, CLIToolType

    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"

    storage.save_session_metadata = AsyncMock(return_value=True)
    storage.sync_worktree_to_s3 = AsyncMock(return_value=True)
    storage.sync_cli_state_to_s3 = AsyncMock(return_value=True)
    storage.sync_ccache_to_s3 = AsyncMock(return_value=True)

    wt = tmp_path / "wt"
    wt.mkdir()

    fork_state = SessionState(
        session_id="fork-s", status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(wt),
        vllm_fork_url="https://github.com/u/vllm.git",
    )
    # Pause path (include_ccache=True) on a fork: ccache IS uploaded.
    ok = await storage.sync_session_to_s3(fork_state, include_ccache=True)
    assert ok
    storage.sync_ccache_to_s3.assert_awaited_once_with("fork-s")

    storage.sync_ccache_to_s3.reset_mock()
    non_fork_state = SessionState(
        session_id="plain-s", status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(wt),
    )
    # A non-fork session never uploads ccache, even with include_ccache=True.
    ok = await storage.sync_session_to_s3(non_fork_state, include_ccache=True)
    assert ok
    storage.sync_ccache_to_s3.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_checkpoint_sync_skips_ccache(tmp_path, monkeypatch):
    """A default checkpoint sync (include_ccache defaulting to False) does NOT
    upload ccache, even for a fork session — uploading on every disconnect
    checkpoint is wasteful (ccache-1)."""
    from shared.session_models import SessionState, SessionStatus, CLIToolType

    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"  # enabled is derived from bucket being non-None
    storage.prefix = "sessions"

    storage.save_session_metadata = AsyncMock(return_value=True)
    storage.sync_worktree_to_s3 = AsyncMock(return_value=True)
    storage.sync_cli_state_to_s3 = AsyncMock(return_value=True)
    storage.sync_ccache_to_s3 = AsyncMock(return_value=True)

    wt = tmp_path / "wt"
    wt.mkdir()

    fork_state = SessionState(
        session_id="fork-s", status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(wt),
        vllm_fork_url="https://github.com/u/vllm.git",
    )
    # Default call (no include_ccache) → checkpoint semantics.
    ok = await storage.sync_session_to_s3(fork_state)
    assert ok
    storage.sync_ccache_to_s3.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pause_sync_includes_ccache(tmp_path, monkeypatch):
    """The pause path passes include_ccache=True, so ccache IS uploaded for a
    fork session (ccache-1)."""
    from shared.session_models import SessionState, SessionStatus, CLIToolType

    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = "b"
    storage.prefix = "sessions"

    storage.save_session_metadata = AsyncMock(return_value=True)
    storage.sync_worktree_to_s3 = AsyncMock(return_value=True)
    storage.sync_cli_state_to_s3 = AsyncMock(return_value=True)
    storage.sync_ccache_to_s3 = AsyncMock(return_value=True)

    wt = tmp_path / "wt"
    wt.mkdir()

    fork_state = SessionState(
        session_id="fork-s", status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE, repo_name="vllm", branch="x",
        created_at=1.0, last_accessed=1.0,
        worktree_path=str(wt),
        vllm_fork_url="https://github.com/u/vllm.git",
    )
    ok = await storage.sync_session_to_s3(fork_state, include_ccache=True)
    assert ok
    storage.sync_ccache_to_s3.assert_awaited_once_with("fork-s")
