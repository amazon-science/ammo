# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for worktree-create-with-build.sh hook fixes."""
import json
import os
import pytest
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_hook_path():
    """Find the hook script."""
    paths = [
        Path(__file__).parent.parent.parent / "ai_cli_session" / ".claude" / "hooks" / "worktree-create-with-build.sh",
        Path("/app/ai_cli_session/.claude/hooks/worktree-create-with-build.sh"),
    ]
    for p in paths:
        if p.exists():
            return p
    pytest.skip("worktree-create-with-build.sh not found")


def _get_codex_create_script_path():
    """Find the Codex worktree creation helper."""
    path = (
        Path(__file__).parent.parent.parent
        / "ai_cli_session"
        / ".codex"
        / "skills"
        / "ammo"
        / "scripts"
        / "create_worktree_with_build.sh"
    )
    if path.exists():
        return path
    pytest.skip("create_worktree_with_build.sh not found")


def _init_codex_worktree_test_repo(tmp_path):
    """Create a minimal repo with the untracked Codex template and main venv."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
    (repo_dir / "README.md").write_text("test\n")
    (repo_dir / "vllm").mkdir()
    (repo_dir / "vllm" / "__init__.py").write_text("__version__ = 'test'\n")
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )

    python_bin = repo_dir / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    os.symlink(sys.executable, python_bin)

    codex_dir = repo_dir / ".codex"
    (codex_dir / "hooks").mkdir(parents=True)
    (codex_dir / "agents").mkdir()
    (codex_dir / "schemas").mkdir()
    (codex_dir / "skills" / "ammo" / "scripts").mkdir(parents=True)
    (codex_dir / "worktrees" / "do-not-copy").mkdir(parents=True)
    (codex_dir / "AGENTS.md").write_text("Use AMMO\n")
    (codex_dir / "config.toml").write_text("model = \"test\"\n")
    (codex_dir / "hooks.json").write_text('{"hooks": {}}\n')
    (codex_dir / "hooks" / "session_start.py").write_text("print('hook')\n")
    (codex_dir / "agents" / "ammo-implementer.toml").write_text("name = \"ammo-implementer\"\n")
    (codex_dir / "schemas" / "state.schema.json").write_text("{}\n")
    (codex_dir / "skills" / "ammo" / "scripts" / "gpu_reservation.py").write_text("# gpu\n")
    (codex_dir / "skills" / "ammo" / "scripts" / "reconcile_track_state.py").write_text("# reconcile\n")
    return repo_dir


def _py_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _main_site_packages(repo_dir):
    """Path to the fake MAIN venv site-packages the hooks point `main-venv.pth` at."""
    return repo_dir / ".venv" / "lib" / f"python{_py_version()}" / "site-packages"


def _populate_fake_main_venv_site_packages(main_sp):
    """Populate a fake MAIN venv site-packages to exercise the GENERAL .pth-replay loop.

    Mirrors the real exposure mechanisms the worktree venv must reach, with MULTIPLE
    path-style redirects (proving the fix is not a single hard-coded entry):

      * Redirect A (NO trailing newline)  -> mimics nvidia_cutlass_dsl.pth, the
        blocking-bug regression guard for the `|| [ -n "$line" ]` newline-safe loop.
      * Redirect B (WITH trailing newline, different nesting) -> proves the loop
        generalizes beyond one entry.
      * Redirect C (dangling target dir does NOT exist) -> must be skipped by `[ -d ]`.
      * Editable directive (`import ...`)  -> must NOT be replayed (would load a finder).
      * Shim directive (`;`-line, no leading `import `) -> must be skipped by `*;*` arm.
      * Direct dir (no .pth)               -> regression guard for mechanism (a),
        reachable via the line-1 main-site-packages entry.
      * Tempting main-venv `vllm/`         -> precedence guard: WorktreeFinder must win.
    """
    main_sp.mkdir(parents=True, exist_ok=True)

    # Redirect A — no trailing newline (nvidia_cutlass_dsl.pth shape).
    (main_sp / "fakecutlass_dsl.pth").write_bytes(b"fakecutlass_dsl/python_packages")
    pkg_a = main_sp / "fakecutlass_dsl" / "python_packages" / "fakecutlass"
    pkg_a.mkdir(parents=True)
    (pkg_a / "__init__.py").write_text('MARKER = "redirected-A"\n')

    # Redirect B — trailing newline, different nesting depth.
    (main_sp / "fakeflash_dsl.pth").write_text("fakeflash_dsl/pkgs\n")
    pkg_b = main_sp / "fakeflash_dsl" / "pkgs" / "fakeflash"
    pkg_b.mkdir(parents=True)
    (pkg_b / "__init__.py").write_text('MARKER = "redirected-B"\n')

    # Redirect C — dangling target (no dir created) → must be skipped.
    (main_sp / "fakedangling.pth").write_text("fakedangling/missing\n")

    # Editable `import `-directive → must NOT be replayed/executed at startup.
    (main_sp / "evil_editable.pth").write_text(
        "import _evil_finder; _evil_finder.install()\n"
    )
    (main_sp / "_evil_finder.py").write_text(
        "import sys\n"
        "sys.modules['_evil_finder_loaded'] = True\n"
        "def install():\n"
        "    pass\n"
    )

    # Shim `;`-line that does NOT start with `import ` → must be skipped by *;* arm.
    (main_sp / "fakeshim.pth").write_text("var = 1; __import__('os')\n")

    # Direct dir (mechanism (a)) — reachable via the line-1 main-sp entry, no .pth.
    direct = main_sp / "fakedirect"
    direct.mkdir()
    (direct / "__init__.py").write_text('MARKER = "direct"\n')

    # AMMO-editable runtime package roots should be copied into each track venv
    # rather than edited through the shared backing main venv.
    flashinfer = main_sp / "flashinfer"
    flashinfer.mkdir()
    (flashinfer / "__init__.py").write_text('MARKER = "main-flashinfer"\n')
    (main_sp / "flashinfer_python-1.2.3.dist-info").mkdir()

    # nvidia_cutlass_dsl exposes the importable `cutlass` package through a
    # path-style redirect, so materialization must also rewrite that redirect to
    # the track-local copy.
    (main_sp / "nvidia_cutlass_dsl.pth").write_text(
        "nvidia_cutlass_dsl/python_packages\n"
    )
    cutlass = main_sp / "nvidia_cutlass_dsl" / "python_packages" / "cutlass"
    cutlass.mkdir(parents=True)
    (cutlass / "__init__.py").write_text('MARKER = "main-cutlass"\n')
    (main_sp / "nvidia_cutlass_dsl-1.2.3.dist-info").mkdir()

    # Tempting main-venv vllm — WorktreeFinder must take precedence over this.
    vllm_main = main_sp / "vllm"
    vllm_main.mkdir()
    (vllm_main / "__init__.py").write_text('SOURCE = "main-venv"\n')


def _commit_file(repo_dir, rel_path, content):
    """Overwrite + commit a tracked file so it lands in HEAD (and thus the worktree)."""
    target = repo_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", rel_path],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", f"set {rel_path}"],
        check=True, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def _run_codex_create(script_path, repo_dir, name):
    """Invoke the Codex worktree creation helper; return (result, worktree_dir)."""
    result = subprocess.run(
        ["bash", str(script_path), name, f"ammo/{name}", str(repo_dir)],
        capture_output=True, text=True, timeout=60,
    )
    return result, repo_dir / ".codex" / "worktrees" / name


def _run_claude_hook(hook_path, repo_dir, name):
    """Invoke the Claude worktree-create hook; return (result, worktree_dir)."""
    hook_input = json.dumps({
        "session_id": "test-session",
        "cwd": str(repo_dir),
        "hook_event_name": "PreToolUse",
        "tool_name": "EnterWorktree",
        "tool_input": {"name": name},
    })
    result = subprocess.run(
        ["bash", str(hook_path)],
        input=hook_input, capture_output=True, text=True, timeout=60,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(repo_dir)},
    )
    return result, repo_dir / ".claude" / "worktrees" / name


def _wt_site_packages(worktree_dir):
    return worktree_dir / ".venv" / "lib" / f"python{_py_version()}" / "site-packages"


def _build_repo_with_fake_main_venv(tmp_path, vllm_source=None):
    """Repo (git + .codex template + fake main venv site-packages) for redirect tests.

    `_init_codex_worktree_test_repo` symlinks `.venv/bin/python` -> sys.executable,
    which suffices for the static content-grep tests but produces a worktree venv that
    CANNOT bootstrap (its `home=` points at the symlink dir, not a real install). These
    redirect tests EXECUTE the worktree python, so we replace the symlink with a REAL
    main venv before populating its site-packages.
    """
    repo_dir = _init_codex_worktree_test_repo(tmp_path)
    shutil.rmtree(repo_dir / ".venv")
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(repo_dir / ".venv")],
        check=True, capture_output=True,
    )
    _populate_fake_main_venv_site_packages(_main_site_packages(repo_dir))
    if vllm_source is not None:
        _commit_file(repo_dir, "vllm/__init__.py", f'SOURCE = "{vllm_source}"\n')
    return repo_dir


def _create_worktree(kind, repo_dir, name):
    """Drive the requested worktree-creation script; return (result, worktree_dir)."""
    if kind == "codex":
        return _run_codex_create(_get_codex_create_script_path(), repo_dir, name)
    if kind == "claude":
        if shutil.which("jq") is None:
            pytest.skip("jq not available (required by the Claude hook)")
        return _run_claude_hook(_get_hook_path(), repo_dir, name)
    raise ValueError(kind)


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
@pytest.mark.parametrize("kind", ["codex", "claude"])
class TestWorktreePthRedirect:
    """Both worktree-creation scripts must replay PATH-STYLE .pth redirects from the
    main venv into the worktree venv (generically, content-driven), while NEVER
    replaying `import `-directive / shim lines (which would load the editable-vllm
    finder and break WorktreeFinder precedence).

    `kind == "codex"` exercises T1–T3; `kind == "claude"` mirrors them (plan T4).
    """

    def test_propagates_multiple_path_style_redirects(self, kind, tmp_path):
        """T1: BOTH path-style redirects (no-newline + newline, different nestings)
        AND the direct-dir package import; paths resolve under their redirect dirs."""
        repo_dir = _build_repo_with_fake_main_venv(tmp_path)
        result, wt = _create_worktree(kind, repo_dir, "redir1")
        assert result.returncode == 0, result.stderr

        out = subprocess.run(
            [str(wt / ".venv" / "bin" / "python"), "-c",
             "import fakecutlass, fakeflash, fakedirect; "
             "print(fakecutlass.MARKER, fakeflash.MARKER, fakedirect.MARKER, "
             "fakecutlass.__file__, fakeflash.__file__)"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        assert "redirected-A" in out.stdout
        assert "redirected-B" in out.stdout
        assert "direct" in out.stdout
        # Resolves under each redirect's nested dir.
        assert "fakecutlass_dsl/python_packages/fakecutlass" in out.stdout
        assert "fakeflash_dsl/pkgs/fakeflash" in out.stdout

    def test_skips_import_directives_and_dangling_paths(self, kind, tmp_path):
        """T2: no `import `-directive, `;`-shim, or dangling line reaches
        main-venv.pth; the editable directive does NOT execute at startup."""
        repo_dir = _build_repo_with_fake_main_venv(tmp_path)
        result, wt = _create_worktree(kind, repo_dir, "redir2")
        assert result.returncode == 0, result.stderr

        lines = (_wt_site_packages(wt) / "main-venv.pth").read_text().splitlines()
        for ln in lines:
            assert not ln.startswith("import "), f"import-directive line emitted: {ln!r}"
            assert ";" not in ln, f"shim line emitted: {ln!r}"
            assert "fakedangling/missing" not in ln, f"dangling target emitted: {ln!r}"

        out = subprocess.run(
            [str(wt / ".venv" / "bin" / "python"), "-c",
             "import sys; print('_evil_finder_loaded' in sys.modules)"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False", \
            "import-directive .pth must not be replayed/executed at startup"

    def test_import_vllm_resolves_to_worktree_source(self, kind, tmp_path):
        """T3: WorktreeFinder precedence — `import vllm` resolves to worktree source
        even with a tempting main-venv vllm/ on the replayed path; no editable finder."""
        repo_dir = _build_repo_with_fake_main_venv(tmp_path, vllm_source="worktree-source")
        result, wt = _create_worktree(kind, repo_dir, "redir3")
        assert result.returncode == 0, result.stderr

        out = subprocess.run(
            [str(wt / ".venv" / "bin" / "python"), "-c",
             "import vllm, sys; print(vllm.SOURCE); "
             "names=[getattr(f,'__name__','') for f in sys.meta_path]; print(names); "
             "print(any('editable' in n.lower() for n in names))"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        lines = out.stdout.splitlines()
        assert lines[0] == "worktree-source", \
            f"WorktreeFinder must win: {out.stdout} {out.stderr}"
        assert "WorktreeFinder" in lines[1] and \
            lines[1].index("WorktreeFinder") < lines[1].index("PathFinder"), \
            f"WorktreeFinder must precede PathFinder: {lines[1]}"
        assert lines[2] == "False", "editable-vllm finder must NOT load"

    def test_materializes_track_local_runtime_packages(self, kind, tmp_path):
        """T4: AMMO-editable third-party runtime packages are physically copied
        into the track venv, including redirected package trees such as CUTLASS."""
        repo_dir = _build_repo_with_fake_main_venv(tmp_path)
        result, wt = _create_worktree(kind, repo_dir, "runtimepkg")
        assert result.returncode == 0, result.stderr

        site_packages = _wt_site_packages(wt)
        assert (site_packages / "flashinfer" / "__init__.py").exists()
        assert (site_packages / "flashinfer_python-1.2.3.dist-info").exists()
        assert (site_packages / "nvidia_cutlass_dsl" / "python_packages" / "cutlass" / "__init__.py").exists()
        assert (site_packages / "nvidia_cutlass_dsl-1.2.3.dist-info").exists()

        out = subprocess.run(
            [str(wt / ".venv" / "bin" / "python"), "-c",
             "import flashinfer, cutlass; "
             "print(flashinfer.__file__); print(cutlass.__file__); "
             "print(flashinfer.MARKER, cutlass.MARKER)"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        lines = out.stdout.splitlines()
        assert str(site_packages / "flashinfer" / "__init__.py") == lines[0]
        assert str(site_packages / "nvidia_cutlass_dsl" / "python_packages" / "cutlass" / "__init__.py") == lines[1]
        assert lines[2] == "main-flashinfer main-cutlass"

        pth_lines = (site_packages / "main-venv.pth").read_text().splitlines()
        assert str(site_packages / "nvidia_cutlass_dsl" / "python_packages") in pth_lines
        assert str(_main_site_packages(repo_dir) / "nvidia_cutlass_dsl" / "python_packages") not in pth_lines


@pytest.mark.unit
class TestWorktreeHookMainRepo:
    """Test MAIN_REPO resolution in worktree-create-with-build.sh."""

    def test_hook_trusts_claude_project_dir(self):
        """When CLAUDE_PROJECT_DIR is set, MAIN_REPO should not be overridden by git-common-dir."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        # The fix: only override MAIN_REPO with git-common-dir when CLAUDE_PROJECT_DIR is NOT set
        assert '[ -z "${CLAUDE_PROJECT_DIR:-}" ]' in content, \
            "Hook should check if CLAUDE_PROJECT_DIR is unset before git-common-dir override"

    def test_hook_has_session_namespaced_branches(self):
        """Hook should create session-namespaced branch names when session_id is available."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        assert 'session_id' in content.lower() or 'SESSION_ID' in content, \
            "Hook should reference session_id for branch namespacing"
        assert 'session/${SESSION_ID}' in content or "session/$SESSION_ID" in content, \
            "Hook should create session-namespaced branches"


@pytest.mark.unit
class TestWorktreeHookBaseRepo:
    """Test BASE_REPO variable for shared lock and ccache path mapping."""

    def test_hook_defines_base_repo_variable(self):
        """Verify the script defines a BASE_REPO variable resolved from git-common-dir."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        # BASE_REPO must be assigned using git rev-parse --git-common-dir
        assert 'BASE_REPO=' in content, \
            "Hook must define a BASE_REPO variable"
        assert 'rev-parse' in content and '--git-common-dir' in content, \
            "BASE_REPO must use git rev-parse --git-common-dir"
        # The sed stripping of /.git$ must be present on the BASE_REPO line
        # Find the line(s) that assign BASE_REPO and check for sed
        base_repo_lines = [l for l in content.splitlines() if 'BASE_REPO=' in l and 'rev-parse' in l]
        assert len(base_repo_lines) >= 1, \
            "Must have a BASE_REPO assignment using rev-parse"
        assert any("sed" in l and ".git" in l for l in base_repo_lines), \
            "BASE_REPO resolution must strip trailing /.git via sed"

    def test_lockfile_uses_main_repo_not_base_repo(self):
        """LOCKFILE must use $MAIN_REPO (session worktree, writable), not $BASE_REPO (shared base, may be root-owned).

        Root cause of 100% worktree creation failure on B200 pods:
        session_user cannot write to /local/repos/vllm/.claude/worktrees/.create-lock
        because the base repo is owned by root. The lock only needs to serialize
        concurrent worktree creation within the same session, so $MAIN_REPO is correct.
        """
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        lockfile_lines = [l for l in content.splitlines() if l.strip().startswith('LOCKFILE=')]
        assert len(lockfile_lines) >= 1, "Must have a LOCKFILE assignment"
        lockfile_line = lockfile_lines[0]
        assert '$MAIN_REPO' in lockfile_line or '${MAIN_REPO' in lockfile_line, \
            f"LOCKFILE must use $MAIN_REPO (writable session worktree), got: {lockfile_line}"
        assert '$BASE_REPO' not in lockfile_line, \
            f"LOCKFILE must NOT use $BASE_REPO (may be root-owned), got: {lockfile_line}"

    def test_ccache_path_map_uses_base_repo_not_main_repo(self):
        """Verify the CCACHE_PATH_MAP jq injection maps to $BASE_REPO, not $MAIN_REPO."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        # Find the jq --arg pm line for CCACHE_PATH_MAP
        jq_lines = [l for l in content.splitlines() if 'jq --arg pm' in l]
        assert len(jq_lines) >= 1, "Must have a jq --arg pm line for CCACHE_PATH_MAP"
        jq_line = jq_lines[0]
        assert '$BASE_REPO' in jq_line or '${BASE_REPO' in jq_line, \
            f"CCACHE_PATH_MAP must map to $BASE_REPO, got: {jq_line}"
        assert '$MAIN_REPO' not in jq_line, \
            f"CCACHE_PATH_MAP must NOT map to $MAIN_REPO, got: {jq_line}"

    def test_no_mkdir_on_base_repo_worktrees_dir(self):
        """Hook must NOT mkdir on $BASE_REPO/.claude/worktrees — it may be root-owned and unwritable.

        The mkdir for the lock directory only needs $MAIN_REPO/.claude/worktrees
        (already created on line 49). Creating dirs in BASE_REPO causes Permission denied.
        """
        hook_path = _get_hook_path()
        content = hook_path.read_text()
        lines = content.splitlines()

        # There should be NO mkdir that references BASE_REPO and worktrees
        mkdir_base_lines = [
            l for l in lines
            if 'mkdir' in l and 'BASE_REPO' in l and 'worktrees' in l
            and not l.strip().startswith('#')
        ]
        assert len(mkdir_base_lines) == 0, \
            f"Must NOT mkdir on $BASE_REPO/.claude/worktrees (unwritable by session_user): {mkdir_base_lines}"

    def test_main_repo_lockdir_mkdir_before_lockfile(self):
        """Verify mkdir -p $MAIN_REPO/.claude/worktrees appears before LOCKFILE assignment."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()
        lines = content.splitlines()

        mkdir_main_indices = [
            i for i, l in enumerate(lines)
            if 'mkdir' in l and 'MAIN_REPO' in l and 'worktrees' in l
            and not l.strip().startswith('#')
        ]
        assert len(mkdir_main_indices) >= 1, \
            "Must have mkdir -p for $MAIN_REPO/.claude/worktrees"

        lockfile_indices = [
            i for i, l in enumerate(lines)
            if l.strip().startswith('LOCKFILE=')
        ]
        assert len(lockfile_indices) >= 1, "Must have LOCKFILE assignment"
        assert mkdir_main_indices[0] < lockfile_indices[0], \
            "mkdir -p $MAIN_REPO/.claude/worktrees must appear before LOCKFILE"

    def test_base_repo_falls_back_to_main_repo(self):
        """Verify backward compatibility -- if git resolution fails, BASE_REPO falls back to MAIN_REPO."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        # Check for fallback pattern: BASE_REPO="${BASE_REPO:-$MAIN_REPO}" or || pattern
        has_default_fallback = '${BASE_REPO:-$MAIN_REPO}' in content or '${BASE_REPO:-${MAIN_REPO}}' in content
        has_or_fallback = any(
            'BASE_REPO=' in l and '||' in l and 'MAIN_REPO' in l
            for l in content.splitlines()
        )
        assert has_default_fallback or has_or_fallback, \
            "BASE_REPO must fall back to MAIN_REPO if git resolution fails"

        # The fallback must appear after the git resolution attempt
        lines = content.splitlines()
        resolution_indices = [
            i for i, l in enumerate(lines)
            if 'BASE_REPO=' in l and 'rev-parse' in l
        ]
        fallback_indices = [
            i for i, l in enumerate(lines)
            if 'BASE_REPO=' in l and 'MAIN_REPO' in l and 'rev-parse' not in l
        ]
        if resolution_indices and fallback_indices:
            assert fallback_indices[0] > resolution_indices[0], \
                "Fallback must appear after git resolution attempt"

    def test_git_worktree_add_uses_main_repo_not_base_repo(self):
        """Verify that git worktree add still uses $MAIN_REPO, not $BASE_REPO."""
        hook_path = _get_hook_path()
        content = hook_path.read_text()

        # git worktree add commands must use MAIN_REPO (exclude comments)
        worktree_add_lines = [
            l for l in content.splitlines()
            if 'git' in l and 'worktree add' in l and not l.strip().startswith('#')
        ]
        assert len(worktree_add_lines) >= 1, "Must have git worktree add commands"
        for line in worktree_add_lines:
            assert '$MAIN_REPO' in line or '${MAIN_REPO' in line, \
                f"git worktree add must use $MAIN_REPO, got: {line}"
            assert '$BASE_REPO' not in line and '${BASE_REPO' not in line, \
                f"git worktree add must NOT use $BASE_REPO, got: {line}"

        # WORKTREE_DIR must still use MAIN_REPO
        worktree_dir_lines = [
            l for l in content.splitlines()
            if l.strip().startswith('WORKTREE_DIR=')
        ]
        assert len(worktree_dir_lines) >= 1, "Must have WORKTREE_DIR assignment"
        for line in worktree_dir_lines:
            assert '$MAIN_REPO' in line or '${MAIN_REPO' in line, \
                f"WORKTREE_DIR must use $MAIN_REPO, got: {line}"


@pytest.mark.unit
class TestWorktreeHookBaseRepoResolution:
    """Integration-style tests that verify git rev-parse --git-common-dir works correctly."""

    @pytest.fixture
    def git_repo(self, tmp_path):
        """Create a real git repo for testing."""
        repo_dir = tmp_path / "base_repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        # Create an initial commit so worktrees can be created
        dummy_file = repo_dir / "README.md"
        dummy_file.write_text("test")
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "README.md"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "init"],
            check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )
        return repo_dir

    @pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git not available",
    )
    def test_base_repo_resolution_in_worktree_context(self, git_repo, tmp_path):
        """In a worktree, git-common-dir should resolve to the base repo."""
        worktree_dir = tmp_path / "worktree_child"
        subprocess.run(
            ["git", "-C", str(git_repo), "worktree", "add", str(worktree_dir), "HEAD"],
            check=True, capture_output=True,
        )

        # Run the exact command the hook uses
        result = subprocess.run(
            ["git", "-C", str(worktree_dir), "rev-parse",
             "--path-format=absolute", "--git-common-dir"],
            check=True, capture_output=True, text=True,
        )
        import re
        resolved = re.sub(r'/\.git$', '', result.stdout.strip())
        assert resolved == str(git_repo), \
            f"Expected {git_repo}, got {resolved}"

        # Cleanup
        subprocess.run(
            ["git", "-C", str(git_repo), "worktree", "remove", str(worktree_dir)],
            capture_output=True,
        )

    @pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git not available",
    )
    def test_base_repo_resolution_plain_repo_is_self(self, git_repo):
        """In a plain (non-worktree) repo, git-common-dir should resolve to itself."""
        result = subprocess.run(
            ["git", "-C", str(git_repo), "rev-parse",
             "--path-format=absolute", "--git-common-dir"],
            check=True, capture_output=True, text=True,
        )
        import re
        resolved = re.sub(r'/\.git$', '', result.stdout.strip())
        assert resolved == str(git_repo), \
            f"Expected {git_repo}, got {resolved}"


@pytest.mark.unit
class TestWorktreeHookLockPermission:
    """Reproduce the lock file permission failure from B200 pod audit.

    The base repo (/local/repos/vllm) is owned by root. When the hook tries to
    create a lock file at $BASE_REPO/.claude/worktrees/.create-lock, it fails
    with 'Permission denied'. The lock must use $MAIN_REPO (session worktree).
    """

    @pytest.fixture
    def git_worktree_env(self, tmp_path):
        """Create a base repo (read-only .claude/) and a session worktree (writable)."""
        # Create base repo
        base = tmp_path / "base_repo"
        base.mkdir()
        subprocess.run(["git", "init", str(base)], check=True, capture_output=True)
        (base / "README.md").write_text("test")
        subprocess.run(
            ["git", "-C", str(base), "add", "README.md"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(base), "commit", "-m", "init"],
            check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )

        # Create session worktree via git worktree add
        session_wt = tmp_path / "session_worktree"
        subprocess.run(
            ["git", "-C", str(base), "worktree", "add", "-b", "session-test", str(session_wt), "HEAD"],
            check=True, capture_output=True,
        )

        # Make base repo's .claude dir read-only to simulate root ownership
        base_claude = base / ".claude" / "worktrees"
        base_claude.mkdir(parents=True)
        (base / ".claude").chmod(0o555)

        yield {"base": base, "session_wt": session_wt}

        # Cleanup: restore permissions so tmp_path can be deleted
        (base / ".claude").chmod(0o755)
        subprocess.run(
            ["git", "-C", str(base), "worktree", "remove", "--force", str(session_wt)],
            capture_output=True,
        )

    @pytest.mark.skipif(
        shutil.which("git") is None or shutil.which("jq") is None,
        reason="git or jq not available",
    )
    def test_hook_succeeds_with_readonly_base_repo(self, git_worktree_env, tmp_path):
        """Hook must succeed even when BASE_REPO/.claude is read-only (root-owned).

        This reproduces the exact failure from the B200 pod:
          line 61: /local/repos/vllm/.claude/worktrees/.create-lock: Permission denied
        """
        hook_path = _get_hook_path()
        session_wt = git_worktree_env["session_wt"]

        # Prepare hook input JSON
        import json
        hook_input = json.dumps({
            "session_id": "test-session-123",
            "cwd": str(session_wt),
            "hook_event_name": "PreToolUse",
            "tool_name": "EnterWorktree",
            "tool_input": {"name": "test-track"},
        })

        result = subprocess.run(
            ["bash", str(hook_path)],
            input=hook_input,
            capture_output=True, text=True,
            timeout=30,
            env={
                **os.environ,
                "CLAUDE_PROJECT_DIR": str(session_wt),
            },
        )

        # The core assertion: no Permission denied from the lock file or mkdir
        assert "Permission denied" not in result.stderr, \
            f"Hook failed with Permission denied (lock file in read-only BASE_REPO): {result.stderr}"

        # The git worktree add step should succeed (visible in stdout)
        assert "Preparing worktree" in result.stdout or "Creating worktree" in result.stderr, \
            f"Git worktree add did not start: stdout={result.stdout}, stderr={result.stderr}"

        # Verify the worktree dir was actually created under session worktree
        expected_dir = session_wt / ".claude" / "worktrees" / "test-track"
        assert expected_dir.exists(), \
            f"Worktree directory was not created at {expected_dir}"

        # Later hook steps (copy .so files, create .venv) may fail in test
        # environment since the repo doesn't have vllm/ or .venv/ — that's OK.
        # The point is the lock + worktree creation worked without Permission denied.


@pytest.mark.unit
class TestCodexWorktreeCreateRepair:
    """Codex worktree helper must preserve Claude hook parity for repair/setup."""

    @pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git not available",
    )
    def test_existing_codex_worktree_repairs_missing_venv_and_codex_template(self, tmp_path):
        """Existing track dirs must not skip repair of .venv or .codex config/hooks."""
        script_path = _get_codex_create_script_path()
        repo_dir = _init_codex_worktree_test_repo(tmp_path)
        worktree_dir = repo_dir / ".codex" / "worktrees" / "op001"
        worktree_dir.mkdir(parents=True)

        result = subprocess.run(
            ["bash", str(script_path), "op001", "ammo/op001", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(worktree_dir)
        assert (worktree_dir / ".venv" / "bin" / "python").exists()
        assert (worktree_dir / ".codex" / "config.toml").read_text() == 'model = "test"\n'
        assert (worktree_dir / ".codex" / "hooks.json").read_text() == '{"hooks": {}}\n'
        assert (worktree_dir / ".codex" / "hooks" / "session_start.py").exists()
        assert (worktree_dir / ".codex" / "AGENTS.md").exists()
        assert (worktree_dir / ".codex" / "agents" / "ammo-implementer.toml").exists()
        assert (worktree_dir / ".codex" / "schemas" / "state.schema.json").exists()
        assert (worktree_dir / ".codex" / "skills" / "ammo" / "scripts" / "gpu_reservation.py").exists()
        assert (worktree_dir / ".codex" / "skills" / "ammo" / "scripts" / "reconcile_track_state.py").exists()
        assert not (worktree_dir / ".codex" / "worktrees").exists()

    @pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git not available",
    )
    def test_new_codex_worktree_copies_track_local_codex_hooks_and_config(self, tmp_path):
        """New child worktrees need the untracked project-local Codex config installed."""
        script_path = _get_codex_create_script_path()
        repo_dir = _init_codex_worktree_test_repo(tmp_path)

        result = subprocess.run(
            ["bash", str(script_path), "op002", "ammo/op002", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        worktree_dir = repo_dir / ".codex" / "worktrees" / "op002"
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(worktree_dir)
        assert (worktree_dir / ".git").exists()
        assert (worktree_dir / ".venv" / "bin" / "python").exists()
        assert (worktree_dir / ".codex" / "config.toml").read_text() == 'model = "test"\n'
        assert (worktree_dir / ".codex" / "hooks.json").read_text() == '{"hooks": {}}\n'
        assert (worktree_dir / ".codex" / "hooks" / "session_start.py").exists()
        assert (worktree_dir / ".codex" / "AGENTS.md").exists()
        assert (worktree_dir / ".codex" / "agents" / "ammo-implementer.toml").exists()
        assert (worktree_dir / ".codex" / "schemas" / "state.schema.json").exists()
        assert (worktree_dir / ".codex" / "skills" / "ammo" / "scripts" / "gpu_reservation.py").exists()
        assert (worktree_dir / ".codex" / "skills" / "ammo" / "scripts" / "reconcile_track_state.py").exists()
        assert not (worktree_dir / ".codex" / "worktrees").exists()

    @pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git not available",
    )
    def test_existing_codex_worktree_repairs_partial_venv_layout(self, tmp_path):
        """Existing .venv/bin/python is not enough; pth files and wrappers are repaired."""
        script_path = _get_codex_create_script_path()
        repo_dir = _init_codex_worktree_test_repo(tmp_path)
        worktree_dir = repo_dir / ".codex" / "worktrees" / "op003"
        (worktree_dir / ".venv" / "bin").mkdir(parents=True)
        os.symlink(sys.executable, worktree_dir / ".venv" / "bin" / "python")

        result = subprocess.run(
            ["bash", str(script_path), "op003", "ammo/op003", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = worktree_dir / ".venv" / "lib" / f"python{py_version}" / "site-packages"
        # main-venv.pth: the main site-packages must be the FIRST line. The .pth-replay
        # loop may append path-style redirect lines after it, so assert presence-as-line-1
        # rather than exact single-line equality (this fixture's fake main venv has no
        # site-packages dir, so the loop finds no .pth and the file stays single-line —
        # but the relaxed assertion holds either way).
        main_sp_line = str(repo_dir / ".venv" / "lib" / f"python{py_version}" / "site-packages")
        contents = (site_packages / "main-venv.pth").read_text().splitlines()
        assert contents[0] == main_sp_line, \
            f"first line must be main site-packages, got {contents!r}"
        # worktree.pth holds the import-directive that installs WorktreeFinder; the
        # worktree dir path itself lives in worktree-path.pth (the assertion below
        # previously targeted worktree.pth by mistake — a pre-existing test bug).
        assert (site_packages / "worktree.pth").read_text() == \
            "import _worktree_finder; _worktree_finder.install()\n"
        assert (site_packages / "worktree-path.pth").read_text() == str(worktree_dir) + "\n"
        assert (worktree_dir / ".venv" / "bin" / "pytest").exists()
        assert (worktree_dir / ".venv" / "bin" / "vllm").exists()
