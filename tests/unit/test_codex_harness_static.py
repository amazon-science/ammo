# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Static checks for Codex as an AMMO Sessions harness option."""

from pathlib import Path
import time
import tomllib
from unittest.mock import patch

from shared.session_models import CLIToolType, SessionState, SessionStatus
from orchestration.session_manager import SessionManager


ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = ROOT / "frontend" / "index.html"
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"
SESSION_MANAGER = ROOT / "orchestration" / "session_manager.py"


def test_classic_create_modal_posts_selected_cli_tool():
    src = INDEX_HTML.read_text()
    assert "cliTool: 'claude'" in src
    assert "cli_tool: this.createForm.cliTool" in src
    assert "Claude Code" in src
    assert "Codex CLI" in src


def test_lightgrid_create_modal_posts_selected_cli_tool():
    html = INDEX_HTML.read_text()
    js = CAMPAIGN_APP_JS.read_text()
    assert "Harness" in html
    assert "cliTool: 'claude'" in js
    assert "cli_tool: this.cmForm.cliTool" in js


def test_session_manager_isolates_codex_home_per_session():
    src = SESSION_MANAGER.read_text()
    assert "CODEX_HOME" in src
    assert "codex-home" in src
    assert 'trust_level = "trusted"' in src
    assert 'approval_policy = "never"' in src
    assert 'sandbox_mode = "danger-full-access"' in src


def test_build_extra_env_writes_trusted_codex_home(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(tmp_path / "missing-auth.json"))
    session_dir = tmp_path / "session-a"
    worktree = session_dir / "worktree"
    worktree.mkdir(parents=True)
    state = SessionState(
        session_id="session-a",
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CODEX,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        session_dir=str(session_dir),
        worktree_path=str(worktree),
        gpu_ids=[2, 3],
        tp_size=2,
        dp_size=1,
    )

    manager = SessionManager.__new__(SessionManager)
    env = manager._build_extra_env(state)

    codex_home = session_dir / "codex-home"
    assert env["CODEX_HOME"] == str(codex_home)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["AMMO_GPU_RES_DIR"] == "/tmp/ammo_gpu_res_session-a"
    raw_config = (codex_home / "config.toml").read_text()
    config = tomllib.loads(raw_config)
    assert config["model"] == "gpt-5.6-sol"
    # No alternate model provider may be active — the generated config must use
    # the default provider so a session never depends on external routing setup.
    assert "model_provider" not in config
    assert "model_providers" not in config
    assert "model_provider" not in raw_config
    assert config["approval_policy"] == "never"
    assert config["sandbox_mode"] == "danger-full-access"
    assert config["projects"][str(worktree)]["trust_level"] == "trusted"


def test_build_extra_env_seeds_codex_auth_from_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(tmp_path / "missing-auth.json"))
    session_dir = tmp_path / "session-a"
    worktree = session_dir / "worktree"
    worktree.mkdir(parents=True)
    state = SessionState(
        session_id="session-a",
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CODEX,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        session_dir=str(session_dir),
        worktree_path=str(worktree),
    )

    manager = SessionManager.__new__(SessionManager)
    with patch("orchestration.session_manager.subprocess.run") as run:
        env = manager._build_extra_env(state)

    run.assert_called_once()
    kwargs = run.call_args.kwargs
    assert run.call_args.args[0] == ["/usr/bin/codex", "login", "--with-api-key"]
    assert kwargs["input"] == "sk-test\n"
    assert kwargs["env"]["CODEX_HOME"] == str(session_dir / "codex-home")
    assert kwargs["env"]["OPENAI_API_KEY"] == ""
    assert env["OPENAI_API_KEY"] == ""


def test_build_extra_env_prefers_codex_auth_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    source_auth = tmp_path / "auth.json"
    source_auth.write_text('{"mode":"test"}')
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(source_auth))

    session_dir = tmp_path / "session-a"
    worktree = session_dir / "worktree"
    worktree.mkdir(parents=True)
    state = SessionState(
        session_id="session-a",
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CODEX,
        repo_name="vllm",
        branch="main",
        created_at=time.time(),
        last_accessed=time.time(),
        session_dir=str(session_dir),
        worktree_path=str(worktree),
    )

    manager = SessionManager.__new__(SessionManager)
    with patch("orchestration.session_manager.subprocess.run") as run:
        env = manager._build_extra_env(state)

    run.assert_not_called()
    assert (session_dir / "codex-home" / "auth.json").read_text() == '{"mode":"test"}'
    assert env["OPENAI_API_KEY"] == ""
