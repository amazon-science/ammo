# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for tar+pigz checkpoint implementation in SessionS3Storage.

Tests the replacement of aws s3 sync with tar+pigz+aws s3 cp for
session checkpointing (256x faster uploads, 47x faster downloads).

Test Groups 1-4: Compression tool detection, tar exclude patterns,
upload (sync_worktree_to_s3), download (restore_worktree_from_s3).
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import shutil
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.session_models import (
    SessionState,
    SessionStatus,
    CLIToolType,
)


def _make_session_state(
    session_id: str = "test-session-123",
    worktree_path: str = "/data/sessions/test-session-123/worktree",
    **kwargs,
) -> SessionState:
    """Helper to create a SessionState for testing."""
    defaults = dict(
        session_id=session_id,
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        worktree_path=worktree_path,
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


# ============================================================================
# Test Group 1: Compression Tool Detection
# ============================================================================


@pytest.mark.unit
class TestCompressionToolDetection:
    """Test pigz/gzip detection and nproc helpers."""

    def test_pigz_detected_when_available(self):
        """When pigz is on PATH, _get_compressor returns pigz with parallel args."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=48):
            compressor, args = storage._get_compressor()

        assert compressor == "pigz"
        assert "-1" in args
        assert "-p" in args
        assert "48" in args

    def test_gzip_fallback_when_pigz_missing(self):
        """When pigz is not on PATH, fall back to gzip."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("shutil.which", return_value=None):
            compressor, args = storage._get_compressor()

        assert compressor == "gzip"
        assert "-1" in args
        # gzip does not support -p
        assert "-p" not in args

    def test_nproc_detection(self):
        """_get_nproc returns os.cpu_count() value."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("os.cpu_count", return_value=48):
            assert storage._get_nproc() == 48

    def test_nproc_fallback_on_none(self):
        """When os.cpu_count() returns None, fall back to 4."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("os.cpu_count", return_value=None):
            assert storage._get_nproc() == 4

    def test_decompressor_uses_pigz_when_available(self):
        """_get_decompressor returns pigz -d with parallel args when available."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=16):
            decompressor, args = storage._get_decompressor()

        assert decompressor == "pigz"
        assert "-d" in args
        assert "-p" in args
        assert "16" in args

    def test_decompressor_falls_back_to_gzip(self):
        """_get_decompressor falls back to gzip -d when pigz missing."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        with patch("shutil.which", return_value=None):
            decompressor, args = storage._get_decompressor()

        assert decompressor == "gzip"
        assert "-d" in args


# ============================================================================
# Test Group 2: Tar Exclude Pattern Construction
# ============================================================================


@pytest.mark.unit
class TestTarExcludePatterns:
    """Test tar --exclude argument construction."""

    def test_default_exclude_patterns_cover_all_s3_sync_patterns(self):
        """Default tar excludes must cover all patterns from old s3 sync."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        args = storage._build_tar_exclude_args()

        # Convert to set of pattern values for checking
        patterns = set()
        for arg in args:
            assert arg.startswith("--exclude=")
            patterns.add(arg.split("=", 1)[1])

        # All patterns that the old s3 sync excluded must be covered.
        # Note: .git is intentionally NOT excluded — in worktrees, .git is a
        # tiny file (not a directory) needed for worktree linkage after restore.
        required_patterns = {
            "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info",
            ".mypy_cache", "node_modules", "venv", ".venv",
            "cmake-build-*", "build", "*.so", "*.a", "*.o",
            "CMakeCache.txt", "CMakeFiles", "download",
        }
        missing = required_patterns - patterns
        assert not missing, f"Missing tar exclude patterns: {missing}"

    def test_exclude_patterns_as_tar_args(self):
        """_build_tar_exclude_args returns flat list of --exclude=X strings."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        args = storage._build_tar_exclude_args()
        assert isinstance(args, list)
        assert len(args) > 0
        for arg in args:
            assert isinstance(arg, str)
            assert arg.startswith("--exclude=")

    def test_custom_exclude_patterns_override(self):
        """Custom exclude_patterns replace the defaults."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        custom = ["*.log", "temp"]
        args = storage._build_tar_exclude_args(exclude_patterns=custom)

        patterns = [arg.split("=", 1)[1] for arg in args]
        assert patterns == ["*.log", "temp"]

    def test_excluded_paths_match_expected(self):
        """Verify tar with exclude args actually excludes the right files."""
        # Create a real temp directory structure
        tmp = tempfile.mkdtemp()
        try:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()

            # Files that should be included
            (worktree / "foo.py").write_text("print('hello')")
            (worktree / "subdir").mkdir()
            (worktree / "subdir" / "bar.txt").write_text("bar")
            (worktree / ".claude").mkdir()
            (worktree / ".claude" / "session.json").write_text("{}")

            # Files/dirs that should be excluded
            # Note: .git is no longer excluded (needed for worktree linkage)
            (worktree / "__pycache__").mkdir()
            (worktree / "__pycache__" / "foo.cpython-312.pyc").write_bytes(b"\x00")
            (worktree / "build").mkdir()
            (worktree / "build" / "lib.so").write_bytes(b"\x00")

            from orchestration.session_state import SessionS3Storage

            with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
                storage = SessionS3Storage()

            exclude_args = storage._build_tar_exclude_args()

            # Run tar --list to see what would be included
            cmd = ["tar", "cf", "-"] + exclude_args + ["-C", tmp, "worktree"]
            # Pipe to tar -t to list contents
            tar_create = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            tar_list = subprocess.Popen(
                ["tar", "tf", "-"],
                stdin=tar_create.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tar_create.stdout.close()
            stdout, _ = tar_list.communicate()
            listed = stdout.decode().strip().split("\n")

            # Normalize: remove trailing slashes for directory entries
            listed = [p.rstrip("/") for p in listed if p.strip()]

            # Should include
            assert any("foo.py" in p for p in listed), f"foo.py not in tar: {listed}"
            assert any("bar.txt" in p for p in listed), f"bar.txt not in tar: {listed}"
            assert any(".claude" in p for p in listed), f".claude not in tar: {listed}"

            # Should exclude
            assert not any("__pycache__" in p for p in listed), f"__pycache__ in tar: {listed}"
            assert not any("build/lib.so" in p for p in listed), f"build in tar: {listed}"
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Test Group 3: Upload (sync_worktree_to_s3) -- Tar Format
# ============================================================================


@pytest.mark.unit
class TestSyncWorktreeToS3Tar:
    """Test the tar+pigz upload path."""

    @pytest.fixture
    def storage(self):
        """Create a SessionS3Storage with test bucket configured."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {
            "SESSION_S3_BUCKET": "test-bucket",
            "SESSION_S3_PREFIX": "sessions",
        }):
            s = SessionS3Storage()
        return s

    @pytest.fixture
    def temp_worktree(self):
        """Create a temporary worktree directory."""
        tmp = tempfile.mkdtemp()
        worktree = Path(tmp) / "worktree"
        worktree.mkdir()
        (worktree / "test.py").write_text("print('hello')")
        yield str(worktree)
        shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_upload_builds_correct_tar_pigz_pipe_command(self, storage, temp_worktree):
        """The subprocess command should be a bash -c tar|pigz|aws s3 cp pipe."""
        state = _make_session_state(worktree_path=temp_worktree)
        session_dir = str(Path(temp_worktree).parent)

        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=48):
            result = await storage.sync_worktree_to_s3(state)

        assert result is True
        # First command is the tar|pigz|aws s3 cp pipe
        args = captured_cmds[0]
        assert args[0] == "bash"
        assert args[1] == "-c"
        shell_cmd = args[2]

        # Verify the pipe structure
        assert "tar cf -" in shell_cmd
        assert "pigz -1 -p 48" in shell_cmd
        assert f"aws s3 cp - s3://test-bucket/sessions/{state.session_id}/worktree.tar.gz" in shell_cmd
        assert f"-C {session_dir} worktree" in shell_cmd

    @pytest.mark.asyncio
    async def test_upload_success_returns_true(self, storage, temp_worktree):
        """When subprocess returns 0, method returns True."""
        state = _make_session_state(worktree_path=temp_worktree)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4):
            result = await storage.sync_worktree_to_s3(state)

        assert result is True

    @pytest.mark.asyncio
    async def test_upload_failure_returns_false(self, storage, temp_worktree):
        """When subprocess returns non-zero, method returns False."""
        state = _make_session_state(worktree_path=temp_worktree)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"upload failed"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4):
            result = await storage.sync_worktree_to_s3(state)

        assert result is False

    @pytest.mark.asyncio
    async def test_upload_exception_returns_false(self, storage, temp_worktree):
        """When subprocess raises an exception, method returns False."""
        state = _make_session_state(worktree_path=temp_worktree)

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("command not found")), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4):
            result = await storage.sync_worktree_to_s3(state)

        assert result is False

    @pytest.mark.asyncio
    async def test_upload_disabled_when_no_bucket(self):
        """When S3 is not configured (bucket=None), returns False immediately."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {}, clear=True):
            # Remove SESSION_S3_BUCKET from env
            os.environ.pop("SESSION_S3_BUCKET", None)
            storage = SessionS3Storage()

        state = _make_session_state()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await storage.sync_worktree_to_s3(state)

        assert result is False
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_no_worktree_path(self, storage):
        """When state.worktree_path is None, returns False."""
        state = _make_session_state(worktree_path=None)

        result = await storage.sync_worktree_to_s3(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_upload_worktree_not_exists(self, storage):
        """When worktree path doesn't exist on disk, returns False."""
        state = _make_session_state(worktree_path="/nonexistent/path/worktree")

        result = await storage.sync_worktree_to_s3(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_upload_uses_session_dir_parent(self, storage, temp_worktree):
        """tar command uses -C {session_dir} worktree to preserve directory structure."""
        state = _make_session_state(worktree_path=temp_worktree)
        expected_session_dir = str(Path(temp_worktree).parent)

        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4):
            await storage.sync_worktree_to_s3(state)

        # First command is the tar pipe
        shell_cmd = captured_cmds[0][2]
        assert f"-C {expected_session_dir} worktree" in shell_cmd

    @pytest.mark.asyncio
    async def test_upload_uses_gzip_fallback(self, storage, temp_worktree):
        """When pigz is not available, the pipe uses gzip -1."""
        state = _make_session_state(worktree_path=temp_worktree)

        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value=None):
            await storage.sync_worktree_to_s3(state)

        # First command is the tar pipe
        shell_cmd = captured_cmds[0][2]
        assert "gzip -1" in shell_cmd
        assert "pigz" not in shell_cmd


# ============================================================================
# Test Group 4: Download (restore_worktree_from_s3) -- Tar Format
# ============================================================================


@pytest.mark.unit
class TestRestoreWorktreeFromS3Tar:
    """Test the tar+pigz download path."""

    @pytest.fixture
    def storage(self):
        """Create a SessionS3Storage with test bucket configured."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {
            "SESSION_S3_BUCKET": "test-bucket",
            "SESSION_S3_PREFIX": "sessions",
        }):
            s = SessionS3Storage()
        return s

    @pytest.mark.asyncio
    async def test_download_builds_correct_pipe_command(self, storage):
        """Command should be aws s3 cp | pigz -d | tar xf pipe."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")
        session_dir = str(target_path.parent)

        call_count = {"n": 0}
        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=48), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is True

        # First call: head-object to check if tar.gz exists
        head_cmd = captured_cmds[0]
        assert "head-object" in head_cmd or "s3api" in str(head_cmd)

        # Second call: the actual download pipe
        download_cmd = captured_cmds[1]
        assert download_cmd[0] == "bash"
        assert download_cmd[1] == "-c"
        shell_cmd = download_cmd[2]

        assert f"aws s3 cp s3://test-bucket/sessions/{session_id}/worktree.tar.gz -" in shell_cmd
        assert "pigz -d -p 48" in shell_cmd
        assert f"tar xf - -C {session_dir}" in shell_cmd

    @pytest.mark.asyncio
    async def test_download_success_returns_true(self, storage):
        """returncode=0 -> True, with format detection (head-object) + download pipe."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        call_count = {"n": 0}
        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is True
        # Must have at least 2 calls: head-object check + download pipe
        assert call_count["n"] >= 2, f"Expected >=2 subprocess calls, got {call_count['n']}"
        # First call should be head-object for format detection
        assert "head-object" in str(captured_cmds[0])

    @pytest.mark.asyncio
    async def test_download_failure_returns_false(self, storage):
        """returncode=1 on the download pipe -> False."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        call_count = {"n": 0}

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            mock_proc = AsyncMock()
            if call_count["n"] == 1:
                # head-object succeeds (tar.gz exists)
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
            else:
                # download pipe fails
                mock_proc.communicate = AsyncMock(return_value=(b"", b"download failed"))
                mock_proc.returncode = 1
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_download_creates_target_directory(self, storage):
        """session_dir (target_path.parent) should be created before extraction."""
        session_id = "test-session-123"
        tmp = tempfile.mkdtemp()
        try:
            target_path = Path(tmp) / "newsession" / "worktree"

            call_count = {"n": 0}
            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                call_count["n"] += 1
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                await storage.restore_worktree_from_s3(session_id, target_path)

            # The parent directory (session_dir) should have been created
            assert target_path.parent.exists()
            # Must have format detection (head-object) + download pipe
            assert call_count["n"] >= 2, f"Expected >=2 subprocess calls, got {call_count['n']}"
            assert "head-object" in str(captured_cmds[0])
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_download_not_found_returns_false(self, storage):
        """When tar.gz does not exist in S3, falls back to legacy sync which also fails."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        # head-object returns non-zero for tar.gz (not found)
        # Then falls back to legacy sync, which also fails
        call_count = {"n": 0}
        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            if call_count["n"] == 1:
                # head-object: tar.gz not found
                mock_proc.communicate = AsyncMock(return_value=(b"", b"Not Found"))
                mock_proc.returncode = 254
            else:
                # Legacy s3 sync fallback also fails (benign stderr -- no "fatal error")
                mock_proc.communicate = AsyncMock(
                    return_value=(b"", b"no objects found")
                )
                mock_proc.returncode = 1
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is False
        # Must have done format detection first (head-object call)
        assert call_count["n"] >= 2, f"Expected >=2 subprocess calls for format detection + fallback, got {call_count['n']}"
        assert "head-object" in str(captured_cmds[0])

    @pytest.mark.asyncio
    async def test_download_exception_returns_false(self, storage):
        """On exception during tar download pipe, returns False."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        call_count = {"n": 0}
        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            captured_cmds.append(args)
            if call_count["n"] == 1:
                # head-object succeeds (tar.gz found)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc
            else:
                # Download pipe raises exception
                raise OSError("command not found")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is False
        # Must have done format detection (head-object) before the pipe failed
        assert call_count["n"] == 2, f"Expected 2 subprocess calls (head-object + pipe), got {call_count['n']}"
        assert "head-object" in str(captured_cmds[0])


# ============================================================================
# Test Group 5: Backward Compatibility (Format Detection)
# ============================================================================


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test format detection and backward compat with old per-file S3 format."""

    @pytest.fixture
    def storage(self):
        """Create a SessionS3Storage with test bucket configured."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {
            "SESSION_S3_BUCKET": "test-bucket",
            "SESSION_S3_PREFIX": "sessions",
        }):
            s = SessionS3Storage()
        return s

    @pytest.fixture
    def temp_worktree(self):
        """Create a temporary worktree directory."""
        tmp = tempfile.mkdtemp()
        worktree = Path(tmp) / "worktree"
        worktree.mkdir()
        (worktree / "test.py").write_text("print('hello')")
        (worktree / ".claude").mkdir()
        (worktree / ".claude" / "state.json").write_text("{}")
        yield str(worktree)
        shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_restore_detects_tar_format(self, storage):
        """When worktree.tar.gz exists in S3, uses tar-based restore (bash -c pipe)."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch("shutil.which", return_value="/usr/bin/pigz"), \
             patch("os.cpu_count", return_value=4), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is True
        # First call: head-object (format detection)
        assert "head-object" in str(captured_cmds[0])
        # Second call: tar-based download pipe (bash -c)
        assert len(captured_cmds) >= 2
        assert captured_cmds[1][0] == "bash"
        assert captured_cmds[1][1] == "-c"
        assert "tar xf" in captured_cmds[1][2]

    @pytest.mark.asyncio
    async def test_restore_falls_back_to_sync_format(self, storage):
        """When worktree.tar.gz does not exist, falls back to aws s3 sync."""
        session_id = "test-session-123"
        target_path = Path("/data/sessions/test-session-123/worktree")

        call_count = {"n": 0}
        captured_cmds = []

        async def fake_subprocess(*args, **kwargs):
            call_count["n"] += 1
            captured_cmds.append(args)
            mock_proc = AsyncMock()
            if call_count["n"] == 1:
                # head-object: tar.gz not found
                mock_proc.communicate = AsyncMock(return_value=(b"", b"Not Found"))
                mock_proc.returncode = 254
            else:
                # Legacy s3 sync succeeds
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
             patch.object(Path, "mkdir"):
            result = await storage.restore_worktree_from_s3(session_id, target_path)

        assert result is True
        # First call: head-object (format detection)
        assert "head-object" in str(captured_cmds[0])
        # Second call: legacy aws s3 sync (NOT bash -c pipe)
        assert len(captured_cmds) >= 2
        sync_cmd = captured_cmds[1]
        assert "s3" in str(sync_cmd)
        assert "sync" in str(sync_cmd)
        # Must NOT be a bash pipe command
        assert sync_cmd[0] != "bash"

    @pytest.mark.asyncio
    async def test_cli_state_noop_for_tar_format(self, storage, temp_worktree):
        """sync_cli_state_to_s3 is a no-op when tar format is used (.claude/ in tar)."""
        state = _make_session_state(worktree_path=temp_worktree)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await storage.sync_cli_state_to_s3(state)

        # Should return True without invoking any subprocess
        assert result is True
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_state_restore_noop_when_claude_dir_exists(self, storage, temp_worktree):
        """restore_cli_state_from_s3 is a no-op when .claude/ already exists (from tar)."""
        worktree_path = Path(temp_worktree)
        # .claude/ already exists from temp_worktree fixture

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await storage.restore_cli_state_from_s3(
                "test-session-123", worktree_path
            )

        # Should return True without invoking subprocess (already extracted from tar)
        assert result is True
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_state_restore_falls_back_to_sync_when_no_claude_dir(self, storage):
        """restore_cli_state_from_s3 falls back to s3 sync when .claude/ missing."""
        tmp = tempfile.mkdtemp()
        try:
            worktree_path = Path(tmp) / "worktree"
            worktree_path.mkdir()
            # No .claude/ directory -- should trigger legacy sync

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
                result = await storage.restore_cli_state_from_s3(
                    "test-session-123", worktree_path
                )

            # Should have invoked subprocess for legacy s3 sync
            assert mock_exec.called
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Test Group 6: CLI State Handling
# ============================================================================


@pytest.mark.unit
class TestCLIStateTarIntegration:
    """Test that .claude/ is included in worktree tar and handled correctly."""

    def test_cli_state_included_in_worktree_tar(self):
        """When tarring worktree, .claude/ must NOT be in the exclude list."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        args = storage._build_tar_exclude_args()
        patterns = [arg.split("=", 1)[1] for arg in args]

        # .claude must NOT be excluded (it's part of the worktree tar)
        assert ".claude" not in patterns
        assert ".claude/" not in patterns
        assert ".claude/*" not in patterns

    def test_restore_extracts_cli_state_with_worktree(self):
        """After tar extraction, .claude/ exists inside the worktree (real tar)."""
        tmp = tempfile.mkdtemp()
        try:
            # Create source worktree with .claude/
            src = Path(tmp) / "src"
            src.mkdir()
            worktree = src / "worktree"
            worktree.mkdir()
            (worktree / "code.py").write_text("x = 1")
            (worktree / ".claude").mkdir()
            (worktree / ".claude" / "session.json").write_text('{"id": "test"}')

            from orchestration.session_state import SessionS3Storage

            with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
                storage = SessionS3Storage()

            exclude_args = storage._build_tar_exclude_args()

            # Create tar archive
            tar_path = Path(tmp) / "worktree.tar.gz"
            cmd_create = (
                ["tar", "cf", "-"] + exclude_args +
                ["-C", str(src), "worktree"]
            )
            # Compress with gzip (always available)
            compress = subprocess.run(
                cmd_create, capture_output=True, check=True
            )
            with open(tar_path, "wb") as f:
                subprocess.run(
                    ["gzip", "-1"], input=compress.stdout,
                    stdout=f, check=True
                )

            # Extract to destination
            dst = Path(tmp) / "dst"
            dst.mkdir()
            with open(tar_path, "rb") as f:
                decompress = subprocess.Popen(
                    ["gzip", "-d"], stdin=f, stdout=subprocess.PIPE
                )
                subprocess.run(
                    ["tar", "xf", "-", "-C", str(dst)],
                    stdin=decompress.stdout, check=True
                )
                decompress.wait()

            # Verify .claude/ was included in the extraction
            assert (dst / "worktree" / ".claude").is_dir()
            assert (dst / "worktree" / ".claude" / "session.json").exists()
            content = (dst / "worktree" / ".claude" / "session.json").read_text()
            assert content == '{"id": "test"}'

            # Verify code file too
            assert (dst / "worktree" / "code.py").read_text() == "x = 1"
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Test Group 7: Integration (Round-Trip)
# ============================================================================


@pytest.mark.unit
class TestTarRoundTrip:
    """Round-trip integration tests using real tar/gzip commands."""

    def _create_test_worktree(self, base: Path) -> Path:
        """Create a worktree with known files for round-trip testing."""
        worktree = base / "worktree"
        worktree.mkdir()

        # Files that should survive the round-trip
        (worktree / "foo.py").write_text("print('hello world')")
        (worktree / "bar").mkdir()
        (worktree / "bar" / "baz.txt").write_text("nested file content")
        (worktree / ".claude").mkdir()
        (worktree / ".claude" / "session.json").write_text('{"key": "value"}')

        # Files that should be excluded
        (worktree / ".git").mkdir()
        (worktree / ".git" / "config").write_text("[core]\n\tbare = false")
        (worktree / "__pycache__").mkdir()
        (worktree / "__pycache__" / "foo.cpython-312.pyc").write_bytes(b"\x00\x01\x02")
        (worktree / "build").mkdir()
        (worktree / "build" / "lib.so").write_bytes(b"\x7fELF")

        return worktree

    def _tar_compress(self, src_base: Path, exclude_args: list) -> bytes:
        """Create tar.gz archive of worktree/, return compressed bytes."""
        cmd = ["tar", "cf", "-"] + exclude_args + ["-C", str(src_base), "worktree"]
        tar_proc = subprocess.run(cmd, capture_output=True, check=True)
        gz_proc = subprocess.run(
            ["gzip", "-1"], input=tar_proc.stdout, capture_output=True, check=True
        )
        return gz_proc.stdout

    def _tar_decompress(self, archive_bytes: bytes, dst: Path):
        """Extract tar.gz archive bytes to destination."""
        decompress = subprocess.Popen(
            ["gzip", "-d"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        extract = subprocess.Popen(
            ["tar", "xf", "-", "-C", str(dst)],
            stdin=decompress.stdout
        )
        decompress.stdout.close()
        decompress.stdin.write(archive_bytes)
        decompress.stdin.close()
        extract.wait()
        decompress.wait()
        assert extract.returncode == 0

    def test_tar_compress_decompress_preserves_files(self):
        """Round-trip: tar+gzip compress then decompress preserves included files."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "src"
            src.mkdir()
            self._create_test_worktree(src)

            exclude_args = storage._build_tar_exclude_args()
            archive = self._tar_compress(src, exclude_args)

            dst = Path(tmp) / "dst"
            dst.mkdir()
            self._tar_decompress(archive, dst)

            # Included files should exist with correct content
            assert (dst / "worktree" / "foo.py").read_text() == "print('hello world')"
            assert (dst / "worktree" / "bar" / "baz.txt").read_text() == "nested file content"
            assert (dst / "worktree" / ".claude" / "session.json").read_text() == '{"key": "value"}'

            # Excluded files should NOT exist
            # Note: .git is no longer excluded (preserved for worktree linkage)
            assert not (dst / "worktree" / "__pycache__").exists()
            assert not (dst / "worktree" / "build").exists()
        finally:
            shutil.rmtree(tmp)

    def test_tar_preserves_file_permissions(self):
        """Executable bit is preserved through tar round-trip."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "src"
            src.mkdir()
            worktree = src / "worktree"
            worktree.mkdir()

            script = worktree / "run.sh"
            script.write_text("#!/bin/bash\necho hello")
            os.chmod(str(script), 0o755)

            exclude_args = storage._build_tar_exclude_args()
            archive = self._tar_compress(src, exclude_args)

            dst = Path(tmp) / "dst"
            dst.mkdir()
            self._tar_decompress(archive, dst)

            restored = dst / "worktree" / "run.sh"
            assert restored.exists()
            assert os.access(str(restored), os.X_OK), "Executable bit was not preserved"
        finally:
            shutil.rmtree(tmp)

    def test_tar_preserves_symlinks(self):
        """Symlinks inside the worktree are preserved through tar round-trip."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "src"
            src.mkdir()
            worktree = src / "worktree"
            worktree.mkdir()

            target_file = worktree / "real_file.txt"
            target_file.write_text("symlink target")
            link = worktree / "link.txt"
            link.symlink_to("real_file.txt")

            exclude_args = storage._build_tar_exclude_args()
            archive = self._tar_compress(src, exclude_args)

            dst = Path(tmp) / "dst"
            dst.mkdir()
            self._tar_decompress(archive, dst)

            restored_link = dst / "worktree" / "link.txt"
            assert restored_link.is_symlink(), "Symlink was not preserved"
            assert os.readlink(str(restored_link)) == "real_file.txt"
            assert restored_link.read_text() == "symlink target"
        finally:
            shutil.rmtree(tmp)

    def test_tar_handles_empty_directory(self):
        """Empty worktree doesn't cause tar errors."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {"SESSION_S3_BUCKET": "test-bucket"}):
            storage = SessionS3Storage()

        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "src"
            src.mkdir()
            worktree = src / "worktree"
            worktree.mkdir()
            # worktree is intentionally empty

            exclude_args = storage._build_tar_exclude_args()
            archive = self._tar_compress(src, exclude_args)
            assert len(archive) > 0

            dst = Path(tmp) / "dst"
            dst.mkdir()
            self._tar_decompress(archive, dst)

            assert (dst / "worktree").is_dir()
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Test Group 8: Error Handling and Edge Cases
# ============================================================================


@pytest.mark.unit
class TestTarEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.fixture
    def storage(self):
        """Create a SessionS3Storage with test bucket configured."""
        from orchestration.session_state import SessionS3Storage

        with patch.dict(os.environ, {
            "SESSION_S3_BUCKET": "test-bucket",
            "SESSION_S3_PREFIX": "sessions",
        }):
            s = SessionS3Storage()
        return s

    @pytest.mark.asyncio
    async def test_concurrent_uploads_dont_conflict(self, storage):
        """Two simultaneous uploads for different sessions use distinct S3 URIs."""
        tmp = tempfile.mkdtemp()
        try:
            # Create two worktrees
            for sid in ["session-aaa", "session-bbb"]:
                wt = Path(tmp) / sid / "worktree"
                wt.mkdir(parents=True)
                (wt / "file.py").write_text(f"# {sid}")

            state_a = _make_session_state(
                session_id="session-aaa",
                worktree_path=str(Path(tmp) / "session-aaa" / "worktree"),
            )
            state_b = _make_session_state(
                session_id="session-bbb",
                worktree_path=str(Path(tmp) / "session-bbb" / "worktree"),
            )

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                results = await asyncio.gather(
                    storage.sync_worktree_to_s3(state_a),
                    storage.sync_worktree_to_s3(state_b),
                )

            assert results == [True, True]

            # Extract S3 URIs from captured commands
            s3_uris = []
            for cmd in captured_cmds:
                if cmd[0] == "bash" and len(cmd) >= 3:
                    shell_cmd = cmd[2]
                    if "worktree.tar.gz" in shell_cmd:
                        s3_uris.append(shell_cmd)

            assert len(s3_uris) == 2
            assert "session-aaa" in s3_uris[0] or "session-aaa" in s3_uris[1]
            assert "session-bbb" in s3_uris[0] or "session-bbb" in s3_uris[1]
            # They must be different URIs
            assert s3_uris[0] != s3_uris[1]
        finally:
            shutil.rmtree(tmp)

    def test_very_large_exclude_list_doesnt_exceed_arg_limits(self, storage):
        """The tar command with all excludes stays well under ARG_MAX."""
        exclude_args = storage._build_tar_exclude_args()

        # Build a representative full command string
        cmd_parts = (
            ["tar", "cf", "-"] + exclude_args +
            ["-C", "/data/sessions/some-long-session-id-that-is-typical/", "worktree"]
        )
        total_len = sum(len(p) for p in cmd_parts) + len(cmd_parts)  # +spaces

        # ARG_MAX is typically 2097152 on Linux; we should be well under
        assert total_len < 200000, f"Command too long: {total_len} bytes"

    @pytest.mark.asyncio
    async def test_s3_uri_construction_with_special_characters(self, storage):
        """Session IDs with hyphens and underscores produce valid S3 URIs."""
        session_ids = [
            "abc-def-123",
            "session_with_underscores",
            "MiXeD-CaSe_123-abc",
            "a" * 64,  # long session ID
        ]

        for sid in session_ids:
            tmp = tempfile.mkdtemp()
            try:
                worktree = Path(tmp) / "worktree"
                worktree.mkdir()
                (worktree / "f.py").write_text("x=1")
                state = _make_session_state(session_id=sid, worktree_path=str(worktree))

                captured_cmds = []

                async def fake_subprocess(*args, **kwargs):
                    captured_cmds.append(args)
                    mock_proc = AsyncMock()
                    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                    mock_proc.returncode = 0
                    return mock_proc

                with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                     patch("shutil.which", return_value="/usr/bin/pigz"), \
                     patch("os.cpu_count", return_value=4):
                    result = await storage.sync_worktree_to_s3(state)

                assert result is True
                # First command is the tar pipe
                shell_cmd = captured_cmds[0][2]
                expected_uri = f"s3://test-bucket/sessions/{sid}/worktree.tar.gz"
                assert expected_uri in shell_cmd, f"Bad URI for session_id={sid}"
            finally:
                shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_upload_cleanup_on_old_format(self, storage):
        """After successful tar upload, old per-file worktree/ objects are cleaned up."""
        tmp = tempfile.mkdtemp()
        try:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()
            (worktree / "f.py").write_text("x=1")
            state = _make_session_state(worktree_path=str(worktree))

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                result = await storage.sync_worktree_to_s3(state)

            assert result is True

            # After the tar upload, there should be cleanup calls for old objects
            # Look for "aws s3 rm --recursive" commands targeting worktree/ and claude_state/
            cleanup_cmds = [
                cmd for cmd in captured_cmds
                if "s3" in str(cmd) and "rm" in str(cmd) and "--recursive" in str(cmd)
            ]
            assert len(cleanup_cmds) >= 1, (
                f"Expected cleanup of old per-file objects, but found no 'aws s3 rm --recursive' calls. "
                f"All commands: {captured_cmds}"
            )

            # Verify cleanup targets the right prefixes
            cleanup_strs = [str(cmd) for cmd in cleanup_cmds]
            assert any("worktree/" in s for s in cleanup_strs), (
                f"Expected cleanup of worktree/ prefix, got: {cleanup_strs}"
            )
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Test Group 7: Bug A - claude-config/ included in S3 tar
# ============================================================================


@pytest.mark.unit
class TestClaudeConfigIncludedInTar:
    """
    Bug A: claude-config/ must be included in the tar archive uploaded to S3.

    Without claude-config/, cross-host S3 restores get an empty CLAUDE_CONFIG_DIR,
    causing `claude --continue` to exit immediately ("[exited]" in terminal).
    """

    @pytest.mark.asyncio
    async def test_sync_to_s3_includes_claude_config_when_it_exists(self):
        """S3 tar command must include both 'worktree' AND 'claude-config' directories."""
        from orchestration.session_state import SessionS3Storage

        tmp = tempfile.mkdtemp()
        try:
            session_dir = Path(tmp) / "test-session-123"
            worktree_dir = session_dir / "worktree"
            claude_config_dir = session_dir / "claude-config"
            worktree_dir.mkdir(parents=True)
            claude_config_dir.mkdir(parents=True)
            # Add a file inside claude-config to make it non-empty
            (claude_config_dir / "history.jsonl").write_text('{"test": true}')

            state = _make_session_state(
                worktree_path=str(worktree_dir),
            )

            storage = SessionS3Storage()
            storage.bucket = "test-bucket"
            storage.prefix = "sessions"

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                result = await storage.sync_worktree_to_s3(state)

            assert result is True

            # Find the tar command (it's run via "bash", "-c", shell_cmd)
            tar_cmds = [
                cmd for cmd in captured_cmds
                if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c" and "tar cf" in str(cmd[2])
            ]
            assert len(tar_cmds) == 1, f"Expected exactly one tar command, got {len(tar_cmds)}: {captured_cmds}"

            tar_shell_cmd = tar_cmds[0][2]
            # The tar command must include both "worktree" and "claude-config"
            assert "worktree" in tar_shell_cmd, (
                f"tar command must include 'worktree', got: {tar_shell_cmd}"
            )
            assert "claude-config" in tar_shell_cmd, (
                f"tar command must include 'claude-config' when it exists, got: {tar_shell_cmd}"
            )
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_sync_to_s3_omits_claude_config_when_not_present(self):
        """When claude-config/ doesn't exist, tar should only include worktree."""
        from orchestration.session_state import SessionS3Storage

        tmp = tempfile.mkdtemp()
        try:
            session_dir = Path(tmp) / "test-session-456"
            worktree_dir = session_dir / "worktree"
            worktree_dir.mkdir(parents=True)
            # No claude-config directory

            state = _make_session_state(
                session_id="test-session-456",
                worktree_path=str(worktree_dir),
            )

            storage = SessionS3Storage()
            storage.bucket = "test-bucket"
            storage.prefix = "sessions"

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                result = await storage.sync_worktree_to_s3(state)

            assert result is True

            tar_cmds = [
                cmd for cmd in captured_cmds
                if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c" and "tar cf" in str(cmd[2])
            ]
            assert len(tar_cmds) == 1

            tar_shell_cmd = tar_cmds[0][2]
            assert "worktree" in tar_shell_cmd
            # claude-config should NOT be in the command when the dir doesn't exist
            assert "claude-config" not in tar_shell_cmd, (
                f"tar command should NOT include 'claude-config' when it doesn't exist, got: {tar_shell_cmd}"
            )
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.asyncio
    async def test_sync_to_s3_includes_codex_home_without_auth_when_it_exists(self):
        """S3 tar command must include Codex history but exclude Codex credentials."""
        from orchestration.session_state import SessionS3Storage

        tmp = tempfile.mkdtemp()
        try:
            session_dir = Path(tmp) / "test-session-codex"
            worktree_dir = session_dir / "worktree"
            codex_home_dir = session_dir / "codex-home"
            worktree_dir.mkdir(parents=True)
            codex_home_dir.mkdir(parents=True)
            (codex_home_dir / "history.jsonl").write_text('{"event":"test"}\n')
            (codex_home_dir / "auth.json").write_text('{"secret": true}\n')

            state = _make_session_state(
                session_id="test-session-codex",
                worktree_path=str(worktree_dir),
                cli_tool=CLIToolType.CODEX,
            )

            storage = SessionS3Storage()
            storage.bucket = "test-bucket"
            storage.prefix = "sessions"

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                result = await storage.sync_worktree_to_s3(state)

            assert result is True

            tar_cmds = [
                cmd for cmd in captured_cmds
                if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c" and "tar cf" in str(cmd[2])
            ]
            assert len(tar_cmds) == 1, f"Expected exactly one tar command, got: {captured_cmds}"

            tar_shell_cmd = tar_cmds[0][2]
            assert "worktree" in tar_shell_cmd
            assert "codex-home" in tar_shell_cmd, (
                f"tar command must include 'codex-home' when it exists, got: {tar_shell_cmd}"
            )
            assert "--exclude=codex-home/auth.json" in tar_shell_cmd, (
                f"tar command must exclude Codex auth, got: {tar_shell_cmd}"
            )
        finally:
            shutil.rmtree(tmp)

    def test_required_tar_excludes_apply_to_custom_exclude_patterns(self):
        """Custom tar excludes must not disable Codex credential protection."""
        from orchestration.session_state import SessionS3Storage

        storage = SessionS3Storage()
        exclude_args = storage._build_tar_exclude_args(["node_modules"])

        assert "--exclude=node_modules" in exclude_args
        assert "--exclude=codex-home/auth.json" in exclude_args

    @pytest.mark.asyncio
    async def test_restore_from_s3_extracts_claude_config(self):
        """S3 restore must extract claude-config/ alongside worktree/ from tar."""
        from orchestration.session_state import SessionS3Storage

        tmp = tempfile.mkdtemp()
        try:
            target_worktree = Path(tmp) / "restored-session" / "worktree"

            storage = SessionS3Storage()
            storage.bucket = "test-bucket"
            storage.prefix = "sessions"

            captured_cmds = []

            async def fake_subprocess(*args, **kwargs):
                captured_cmds.append(args)
                mock_proc = AsyncMock()
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                return mock_proc

            # Make _tar_gz_exists_in_s3 return True to use tar path
            with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess), \
                 patch("shutil.which", return_value="/usr/bin/pigz"), \
                 patch("os.cpu_count", return_value=4):
                result = await storage._restore_worktree_from_tar(
                    "test-session-123", target_worktree
                )

            assert result is True

            # The tar extraction command extracts to the session_dir (parent of worktree)
            # so anything in the tar (worktree/, claude-config/) gets extracted there
            tar_cmds = [
                cmd for cmd in captured_cmds
                if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c" and "tar xf" in str(cmd[2])
            ]
            assert len(tar_cmds) == 1, f"Expected tar extract command, got: {captured_cmds}"

            # Verify extraction target is the session dir (parent of worktree)
            tar_shell_cmd = tar_cmds[0][2]
            session_dir = str(target_worktree.parent)
            assert session_dir in tar_shell_cmd, (
                f"tar extract must target session dir {session_dir}, got: {tar_shell_cmd}"
            )
        finally:
            shutil.rmtree(tmp)
