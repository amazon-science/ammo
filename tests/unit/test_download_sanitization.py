# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for download archive sanitization.

Ensures that sensitive files (.claude/, .codex/, claude-config/, codex-home/,
CLAUDE.md, AGENTS.md) are stripped from download archives but preserved in S3
backups needed for cross-host S3 restore.

.codex/ and AGENTS.md are the Codex analogues of .claude/ and CLAUDE.md:
setup_codex_workspace copies the full AMMO skill tree to worktree/.codex and
writes worktree/AGENTS.md. Both variants get mirrored coverage here.
"""

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
from orchestration.session_state import SessionS3Storage


def _make_storage(bucket: str = "test-bucket", prefix: str = "sessions") -> SessionS3Storage:
    """Create a SessionS3Storage instance bypassing __init__ env vars."""
    storage = SessionS3Storage.__new__(SessionS3Storage)
    storage.bucket = bucket
    storage.prefix = prefix
    storage.ttl_days = 30
    return storage


def _make_session_state(
    session_id: str = "test-sess-001",
    worktree_path: str = "/data/sessions/test-sess-001/worktree",
    status: SessionStatus = SessionStatus.PAUSED,
) -> SessionState:
    return SessionState(
        session_id=session_id,
        status=status,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=worktree_path,
    )


# ============================================================================
# Test Group 1: Download Archive Exclusions (aws s3 sync --exclude flags)
# ============================================================================


@pytest.mark.unit
class TestDownloadArchiveExclusions:
    """Verify create_download_archive() S3 sync excludes sensitive paths."""

    async def _capture_sync_cmd(self, storage, session_id, tmp_path):
        """Helper: run create_download_archive and capture the aws s3 sync command."""
        temp_download_dir = tmp_path / "download_work"
        temp_download_dir.mkdir(parents=True, exist_ok=True)
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        captured_sync_cmd = []

        mock_proc_sync = MagicMock()
        mock_proc_sync.returncode = 0
        mock_proc_sync.communicate = AsyncMock(return_value=(b"", b""))

        mock_proc_upload = MagicMock()
        mock_proc_upload.returncode = 0
        mock_proc_upload.communicate = AsyncMock(return_value=(b"", b""))

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                captured_sync_cmd.extend(args)
                return mock_proc_sync
            else:
                return mock_proc_upload

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive") as mock_archive, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=1024), \
             patch("shutil.rmtree"), \
             patch("os.unlink"):
            mock_archive.return_value = str(temp_download_dir) + ".zip"
            await storage.create_download_archive(session_id)

        return captured_sync_cmd

    @pytest.mark.asyncio
    async def test_excludes_worktree_claude_dir(self, tmp_path):
        """aws s3 sync must exclude worktree/.claude/*."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-001", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/.claude/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_worktree_claude_dir_recursive(self, tmp_path):
        """aws s3 sync must exclude worktree/.claude/**/*."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-002", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/.claude/**/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_claude_config_dir(self, tmp_path):
        """aws s3 sync must exclude claude-config/*."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-003", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "claude-config/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_claude_config_dir_recursive(self, tmp_path):
        """aws s3 sync must exclude claude-config/**/*."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-004", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "claude-config/**/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_codex_auth_json(self, tmp_path):
        """aws s3 sync must exclude codex-home/auth.json."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-codex", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "codex-home/auth.json" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_worktree_claude_md(self, tmp_path):
        """aws s3 sync must exclude worktree/CLAUDE.md."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-005", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/CLAUDE.md" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_worktree_codex_dir(self, tmp_path):
        """aws s3 sync must exclude worktree/.codex/* (mirrors .claude)."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-007", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/.codex/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_worktree_codex_dir_recursive(self, tmp_path):
        """aws s3 sync must exclude worktree/.codex/**/* (mirrors .claude)."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-008", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/.codex/**/*" in cmd_str

    @pytest.mark.asyncio
    async def test_excludes_worktree_agents_md(self, tmp_path):
        """aws s3 sync must exclude worktree/AGENTS.md (mirrors CLAUDE.md)."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-009", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "worktree/AGENTS.md" in cmd_str

    @pytest.mark.asyncio
    async def test_regular_files_not_excluded(self, tmp_path):
        """Regular repo files like worktree/src/main.py must NOT be excluded."""
        storage = _make_storage()
        cmd = await self._capture_sync_cmd(storage, "sess-excl-006", tmp_path)
        cmd_str = " ".join(str(a) for a in cmd)
        assert "src/main.py" not in cmd_str


# ============================================================================
# Test Group 2: Post-Sync Cleanup (defense in depth before zip)
# ============================================================================


@pytest.mark.unit
class TestPostSyncCleanup:
    """Verify post-sync cleanup removes sensitive files before zipping."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree_claude_dir(self, tmp_path):
        """Post-sync cleanup deletes worktree/.claude/ before zip."""
        storage = _make_storage()
        session_id = "sess-cleanup-001"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        claude_dir = worktree_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / "settings.json").write_text("{}")
        (worktree_dir / "src").mkdir(parents=True, exist_ok=True)
        (worktree_dir / "src" / "main.py").write_text("print('hello')")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            # At the time make_archive is called, .claude should be gone
            archive_created.append({
                "claude_exists": (Path(root_dir) / "worktree" / ".claude").exists(),
                "src_exists": (Path(root_dir) / "worktree" / "src" / "main.py").exists(),
            })
            # Create the zip file so the rest of the flow works
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["claude_exists"] is False, ".claude/ should be removed before zip"
        assert archive_created[0]["src_exists"] is True, "Regular files should survive"

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree_codex_dir(self, tmp_path):
        """Post-sync cleanup deletes worktree/.codex/ (the AMMO skill corpus)."""
        storage = _make_storage()
        session_id = "sess-cleanup-codex-dir"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        codex_dir = worktree_dir / ".codex"
        (codex_dir / "skills" / "ammo").mkdir(parents=True, exist_ok=True)
        (codex_dir / "skills" / "ammo" / "SKILL.md").write_text("# AMMO")
        (codex_dir / "config.toml").write_text("model = 'x'\n")
        (worktree_dir / "src").mkdir(parents=True, exist_ok=True)
        (worktree_dir / "src" / "main.py").write_text("print('hello')")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "codex_exists": (Path(root_dir) / "worktree" / ".codex").exists(),
                "src_exists": (Path(root_dir) / "worktree" / "src" / "main.py").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["codex_exists"] is False, ".codex/ should be removed before zip"
        assert archive_created[0]["src_exists"] is True, "Regular files should survive"

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree_agents_md(self, tmp_path):
        """Post-sync cleanup deletes worktree/AGENTS.md (Codex CLAUDE.md analogue)."""
        storage = _make_storage()
        session_id = "sess-cleanup-agents-md"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        (worktree_dir / "AGENTS.md").write_text("# Sensitive Codex instructions")
        (worktree_dir / "README.md").write_text("# Readme")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "agents_md_exists": (Path(root_dir) / "worktree" / "AGENTS.md").exists(),
                "readme_exists": (Path(root_dir) / "worktree" / "README.md").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["agents_md_exists"] is False, "AGENTS.md should be removed"
        assert archive_created[0]["readme_exists"] is True, "README.md should survive"

    @pytest.mark.asyncio
    async def test_cleanup_removes_nested_codex_dirs(self, tmp_path):
        """Nested .codex dirs (e.g. worktree/subdir/.codex/) are also removed."""
        storage = _make_storage()
        session_id = "sess-cleanup-codex-nested"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        nested_codex = worktree_dir / "subdir" / ".codex"
        nested_codex.mkdir(parents=True, exist_ok=True)
        (nested_codex / "hooks.json").write_text("{}")
        (worktree_dir / "subdir" / "code.py").write_text("pass")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "nested_codex_exists": (Path(root_dir) / "worktree" / "subdir" / ".codex").exists(),
                "code_exists": (Path(root_dir) / "worktree" / "subdir" / "code.py").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["nested_codex_exists"] is False
        assert archive_created[0]["code_exists"] is True

    @pytest.mark.asyncio
    async def test_cleanup_removes_claude_config_dir(self, tmp_path):
        """Post-sync cleanup deletes claude-config/ before zip."""
        storage = _make_storage()
        session_id = "sess-cleanup-002"

        temp_download_dir = tmp_path / "download_work"
        claude_config_dir = temp_download_dir / "claude-config"
        claude_config_dir.mkdir(parents=True, exist_ok=True)
        (claude_config_dir / "conversation.json").write_text("{}")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "claude_config_exists": (Path(root_dir) / "claude-config").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["claude_config_exists"] is False

    @pytest.mark.asyncio
    async def test_cleanup_removes_codex_home_dir(self, tmp_path):
        """Post-sync cleanup deletes codex-home/ before zip."""
        storage = _make_storage()
        session_id = "sess-cleanup-codex"

        temp_download_dir = tmp_path / "download_work"
        codex_home_dir = temp_download_dir / "codex-home"
        codex_home_dir.mkdir(parents=True, exist_ok=True)
        (codex_home_dir / "auth.json").write_text('{"secret": true}')
        (codex_home_dir / "history.jsonl").write_text('{"event": "test"}\n')
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "codex_home_exists": (Path(root_dir) / "codex-home").exists(),
                "session_json_exists": (Path(root_dir) / "session.json").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["codex_home_exists"] is False
        assert archive_created[0]["session_json_exists"] is True

    @pytest.mark.asyncio
    async def test_tar_checkpoint_is_extracted_and_sanitized_before_zip(self, tmp_path):
        """Download archive must not contain raw checkpoint tar or Codex history."""
        import tarfile
        import zipfile

        storage = _make_storage()
        session_id = "sess-cleanup-tar"

        temp_download_dir = tmp_path / "download_work"
        temp_download_dir.mkdir(parents=True, exist_ok=True)
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        tar_source = tmp_path / "tar_source"
        (tar_source / "worktree" / "src").mkdir(parents=True)
        (tar_source / "worktree" / ".claude").mkdir(parents=True)
        (tar_source / "worktree" / ".codex" / "skills").mkdir(parents=True)
        (tar_source / "claude-config").mkdir(parents=True)
        (tar_source / "codex-home").mkdir(parents=True)
        (tar_source / "worktree" / "src" / "main.py").write_text("print('ok')\n")
        (tar_source / "worktree" / ".claude" / "settings.json").write_text("{}")
        (tar_source / "worktree" / ".codex" / "hooks.json").write_text("{}")
        (tar_source / "worktree" / ".codex" / "skills" / "SKILL.md").write_text("# AMMO\n")
        (tar_source / "worktree" / "CLAUDE.md").write_text("# claude\n")
        (tar_source / "worktree" / "AGENTS.md").write_text("# agents\n")
        (tar_source / "claude-config" / "history.jsonl").write_text("{}\n")
        (tar_source / "codex-home" / "history.jsonl").write_text("{}\n")
        (tar_source / "codex-home" / "auth.json").write_text('{"secret": true}\n')

        with tarfile.open(temp_download_dir / "worktree.tar.gz", "w:gz") as tar:
            tar.add(tar_source / "worktree", arcname="worktree")
            tar.add(tar_source / "claude-config", arcname="claude-config")
            tar.add(tar_source / "codex-home", arcname="codex-home")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        real_unlink = os.unlink

        def keep_final_zip(path, *args, **kwargs):
            if str(path).endswith(".zip"):
                return None
            return real_unlink(path, *args, **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("os.unlink", side_effect=keep_final_zip):
            archive_key = await storage.create_download_archive(session_id)

        assert archive_key == f"sessions/{session_id}/download/session_archive.zip"

        zip_path = Path(str(temp_download_dir) + ".zip")
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())

        assert "session.json" in names
        assert "worktree/src/main.py" in names
        assert "worktree.tar.gz" not in names
        assert not any(name.startswith("codex-home/") for name in names)
        assert not any(name.startswith("claude-config/") for name in names)
        assert not any("/.claude/" in name or name.startswith(".claude/") for name in names)
        assert not any("/.codex/" in name or name.startswith(".codex/") for name in names)
        assert not any(Path(name).name == "CLAUDE.md" for name in names)
        assert not any(Path(name).name == "AGENTS.md" for name in names)

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree_claude_md(self, tmp_path):
        """Post-sync cleanup deletes worktree/CLAUDE.md before zip."""
        storage = _make_storage()
        session_id = "sess-cleanup-003"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        (worktree_dir / "CLAUDE.md").write_text("# Sensitive config")
        (worktree_dir / "README.md").write_text("# Readme")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "claude_md_exists": (Path(root_dir) / "worktree" / "CLAUDE.md").exists(),
                "readme_exists": (Path(root_dir) / "worktree" / "README.md").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["claude_md_exists"] is False, "CLAUDE.md should be removed"
        assert archive_created[0]["readme_exists"] is True, "README.md should survive"

    @pytest.mark.asyncio
    async def test_session_json_survives_cleanup(self, tmp_path):
        """session.json at the root must not be removed by cleanup."""
        storage = _make_storage()
        session_id = "sess-cleanup-004"

        temp_download_dir = tmp_path / "download_work"
        temp_download_dir.mkdir(parents=True, exist_ok=True)
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "session_json_exists": (Path(root_dir) / "session.json").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["session_json_exists"] is True

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_dirs_gracefully(self, tmp_path):
        """No exception when .claude/ or claude-config/ don't exist in the download."""
        storage = _make_storage()
        session_id = "sess-cleanup-005"

        temp_download_dir = tmp_path / "download_work"
        temp_download_dir.mkdir(parents=True, exist_ok=True)
        # Only session.json, no .claude or claude-config
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            # Should not raise any exception
            result = await storage.create_download_archive(session_id)

        assert result is not None

    @pytest.mark.asyncio
    async def test_cleanup_handles_nested_claude_dirs(self, tmp_path):
        """Nested .claude dirs (e.g. worktree/subdir/.claude/) are also removed."""
        storage = _make_storage()
        session_id = "sess-cleanup-006"

        temp_download_dir = tmp_path / "download_work"
        worktree_dir = temp_download_dir / "worktree"
        nested_claude = worktree_dir / "subdir" / ".claude"
        nested_claude.mkdir(parents=True, exist_ok=True)
        (nested_claude / "config.json").write_text("{}")
        (worktree_dir / "subdir" / "code.py").write_text("pass")
        (temp_download_dir / "session.json").write_text('{"session_id": "test"}')

        archive_created = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_exec(*args, **kwargs):
            return mock_proc

        def mock_make_archive(base_name, fmt, root_dir):
            archive_created.append({
                "nested_claude_exists": (Path(root_dir) / "worktree" / "subdir" / ".claude").exists(),
                "code_exists": (Path(root_dir) / "worktree" / "subdir" / "code.py").exists(),
            })
            zip_path = base_name + ".zip"
            Path(zip_path).write_text("fake zip")
            return zip_path

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec), \
             patch("tempfile.mkdtemp", return_value=str(temp_download_dir)), \
             patch("shutil.make_archive", side_effect=mock_make_archive), \
             patch("os.path.getsize", return_value=1024):
            await storage.create_download_archive(session_id)

        assert len(archive_created) == 1
        assert archive_created[0]["nested_claude_exists"] is False
        assert archive_created[0]["code_exists"] is True


# ============================================================================
# Test Group 3: S3 Backup Must Retain Sensitive Files (for resume)
# ============================================================================


@pytest.mark.unit
class TestS3BackupRetainsSensitiveFiles:
    """sync_worktree_to_s3 / _build_tar_exclude_args must NOT strip .claude or .codex.

    Download sanitization and S3 retention pull in opposite directions on
    purpose: the user download must not ship the skill corpus, and the S3
    backup must keep it so cross-host S3 restore restores a working session.
    """

    @pytest.mark.asyncio
    async def test_sync_worktree_tar_does_not_exclude_claude(self, tmp_path):
        """sync_worktree_to_s3 tar command must NOT exclude .claude/ (needed for resume)."""
        storage = _make_storage()

        session_dir = tmp_path / "sess-backup-001"
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        state = _make_session_state(
            session_id="sess-backup-001",
            worktree_path=str(worktree_path),
        )

        captured_cmd = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def capture_exec(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            storage._cleanup_old_per_file_objects = AsyncMock()
            await storage.sync_worktree_to_s3(state)

        assert len(captured_cmd) >= 3
        shell_cmd = captured_cmd[2]  # bash -c <cmd>
        # The tar exclude list should NOT contain .claude
        assert "--exclude=.claude" not in shell_cmd

    @pytest.mark.asyncio
    async def test_sync_worktree_tar_does_not_exclude_codex(self, tmp_path):
        """sync_worktree_to_s3 tar command must NOT exclude .codex/ (needed for resume)."""
        storage = _make_storage()

        session_dir = tmp_path / "sess-backup-002"
        worktree_path = session_dir / "worktree"
        worktree_path.mkdir(parents=True, exist_ok=True)

        state = _make_session_state(
            session_id="sess-backup-002",
            worktree_path=str(worktree_path),
        )

        captured_cmd = []

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        async def capture_exec(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
            storage._cleanup_old_per_file_objects = AsyncMock()
            await storage.sync_worktree_to_s3(state)

        assert len(captured_cmd) >= 3
        shell_cmd = captured_cmd[2]  # bash -c <cmd>
        assert "--exclude=.codex" not in shell_cmd

    def test_build_tar_exclude_args_does_not_contain_claude(self):
        """_build_tar_exclude_args() default list must NOT include .claude."""
        storage = _make_storage()
        exclude_args = storage._build_tar_exclude_args()
        for arg in exclude_args:
            assert ".claude" not in arg, f"Unexpected .claude exclusion in tar args: {arg}"
            assert "claude-config" not in arg, f"Unexpected claude-config exclusion in tar args: {arg}"
            assert "CLAUDE.md" not in arg, f"Unexpected CLAUDE.md exclusion in tar args: {arg}"

    def test_build_tar_exclude_args_does_not_contain_codex_skill_tree(self):
        """_build_tar_exclude_args() must NOT strip .codex or AGENTS.md.

        codex-home/auth.json stays excluded: it is a credential, not resume state.
        """
        storage = _make_storage()
        exclude_args = storage._build_tar_exclude_args()
        for arg in exclude_args:
            assert ".codex" not in arg, f"Unexpected .codex exclusion in tar args: {arg}"
            assert "AGENTS.md" not in arg, f"Unexpected AGENTS.md exclusion in tar args: {arg}"
        assert "--exclude=codex-home/auth.json" in exclude_args, (
            "the Codex credential must stay excluded from S3 backups"
        )
