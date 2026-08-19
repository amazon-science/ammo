# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK_DIR = ROOT / "ai_cli_session" / ".codex" / "hooks"
TRUSTED_PYTHON = Path("/usr/bin/python3")


def _clean_env(**overrides):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AMMO_") and key != "CODEX_SESSION_ID"
    }
    env.update(overrides)
    return env


def _run_hook(script_name, payload, env=None):
    return subprocess.run(
        [sys.executable, str(HOOK_DIR / script_name)],
        input=json.dumps(payload),
        env=env if env is not None else _clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def _ammo_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "cuda_vllm_kernel_target"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "state.json").write_text(
        json.dumps({"campaign": {"status": "active"}}),
        encoding="utf-8",
    )
    return artifact_dir


def _write_campaign_round_state(artifact_dir, current_round=1):
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": current_round,
                    "rounds": [{"round_id": idx + 1} for idx in range(current_round)],
                }
            }
        ),
        encoding="utf-8",
    )


def _bind_artifact_to_sessions(artifact_dir, *, server_session_id, codex_session_id):
    state_path = artifact_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["session_id"] = server_session_id
    state["codex_thread_id"] = codex_session_id
    state_path.write_text(json.dumps(state), encoding="utf-8")


def _pretool_payload(command, cwd=None, session_id=None):
    payload = {
        "cwd": str(cwd or ROOT),
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def _json_stdout(result):
    assert result.stdout.strip()
    return json.loads(result.stdout)


def test_codex_pretool_allows_static_rg_mentions_of_gpu_tools_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("rg 'vllm|nsys|ncu|vllm bench latency' docs ai_cli_session"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_static_sed_path_mentions_of_gpu_tools_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("sed -n '1,120p' docs/vllm/nsys_ncu_notes.md"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_chained_static_rg_mentions_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("cd docs && rg vllm"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_apply_patch_text_mentions_of_gpu_tools_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    patch_text = """*** Begin Patch
*** Update File: docs/gpu_notes.md
@@
+Mention nsys, ncu, and vllm bench latency as examples only.
*** End Patch
"""
    payload = {
        "cwd": str(ROOT),
        "tool_name": "apply_patch",
        "tool_input": {"patch": patch_text},
    }
    result = _run_hook(
        "pre_tool_use_guard.py",
        payload,
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_blocks_vllm_op_defaults_enabled_in_envs_py():
    payload = {
        "cwd": str(ROOT),
        "tool_name": "apply_patch",
        "tool_input": {
            "file_path": "vllm/envs.py",
            "content": "VLLM_OP123: bool = True\n",
        },
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env=_clean_env())

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "VLLM_OP feature flags in envs.py must default off" in payload["reason"]


def test_codex_pretool_blocks_actual_nsys_ncu_and_vllm_bench_commands(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    env = _clean_env(
        AMMO_ARTIFACT_DIR=str(artifact_dir),
        AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
        AMMO_SESSION_ID="session-a",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("nsys profile -o trace python bench.py"),
        env=env,
    )
    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "without reservation" in payload["reason"]


def test_codex_pretool_missing_gpu_reservation_is_one_shot_per_session(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    env = _clean_env(
        AMMO_ARTIFACT_DIR=str(artifact_dir),
        AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
        AMMO_SESSION_ID="session-a",
    )

    first = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env=env,
    )
    second = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env=env,
    )
    other_session = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env={**env, "AMMO_SESSION_ID": "session-b"},
    )

    assert _json_stdout(first)["decision"] == "block"
    assert second.returncode == 0
    assert second.stdout == ""
    assert _json_stdout(other_session)["decision"] == "block"


def test_codex_pretool_gpu_reservation_one_shot_uses_campaign_session_for_subagents(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    (artifact_dir / "state.json").write_text(
        json.dumps({"session_id": "campaign-id", "campaign": {"status": "active"}}),
        encoding="utf-8",
    )
    base_env = _clean_env(
        AMMO_ARTIFACT_DIR=str(artifact_dir),
        AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
    )

    first = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env={**base_env, "CODEX_SESSION_ID": "subagent-a"},
    )
    second = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env={**base_env, "CODEX_SESSION_ID": "subagent-b"},
    )

    assert _json_stdout(first)["decision"] == "block"
    assert second.returncode == 0
    assert second.stdout == ""


def test_codex_pretool_allows_unreserved_gpu_override(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("ncu --set full python bench.py"),
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
            AMMO_SESSION_ID="session-a",
            AMMO_ALLOW_UNRESERVED_GPU="1",
        ),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_warns_for_direct_vllm_bench_latency_without_hard_block(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    env = _clean_env(
        AMMO_ARTIFACT_DIR=str(artifact_dir),
        AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
        AMMO_SESSION_ID="session-a",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload('CUDA_VISIBLE_DEVICES="" vllm bench latency --model Qwen/Qwen3-0.6B'),
        env=env,
    )

    payload = _json_stdout(result)
    assert "decision" not in payload
    assert payload["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "raw vllm bench latency detected" in payload["hookSpecificOutput"]["additionalContext"]


def test_codex_pretool_warns_for_production_parity_flags_without_hard_block(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload('CUDA_VISIBLE_DEVICES="" TORCH_COMPILE_DISABLE=1 python bench.py'),
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
            AMMO_SESSION_ID="session-a",
        ),
    )

    payload = _json_stdout(result)
    assert "decision" not in payload
    assert "disable production parity" in payload["hookSpecificOutput"]["additionalContext"]


def test_codex_pretool_detects_nvidia_smi_query_compute_as_gpu_heavy(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("nvidia-smi --query-compute-apps=pid,name --format=csv"),
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"),
            AMMO_SESSION_ID="session-a",
        ),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "without reservation" in payload["reason"]


def test_codex_pretool_allows_python_json_tool_state_inspection(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(f".venv/bin/python -m json.tool {artifact_dir / 'state.json'}"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_python_heredoc_state_inspection(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    command = f""".venv/bin/python - <<'PY'
import json
state = json.load(open("{artifact_dir / 'state.json'}"))
print(state["campaign"]["status"])
PY"""

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_python_c_state_inspection(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    command = (
        ".venv/bin/python -c "
        f"\"import json; from pathlib import Path; data=json.loads(Path('{artifact_dir / 'state.json'}').read_text()); print(data['campaign']['status'])\""
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_monitor_ack_command_for_pending_hard_gate(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "CRITICAL",
                "ack_required": True,
                "summary": "review required",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    command = (
        f"{TRUSTED_PYTHON} {ack_script} --session-id session-a --queue {queue} "
        '--all --note "read and acknowledged"'
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command, session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_allows_monitor_ack_note_with_quoted_semicolon(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    record = {
        "target_session_id": "session-a",
        "severity": "CRITICAL",
        "ack_required": True,
        "summary": "review required",
        "record_id": "record-a",
    }
    queue.write_text(json.dumps(record) + "\n", encoding="utf-8")
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    command = (
        f"{TRUSTED_PYTHON} {ack_script} --session-id session-a --queue {queue} "
        '--record-id record-a --note "Retained state-authorized monitor; '
        'nested duplicate was interrupted."'
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command, session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


@pytest.mark.parametrize(
    "suffix",
    [
        "; touch /tmp/monitor-ack-chain",
        " && touch /tmp/monitor-ack-chain",
        " | tee /tmp/monitor-ack-pipe",
        " > /tmp/monitor-ack-redirect",
        "\ntrue",
        ' --status "$(touch /tmp/monitor-ack-substitution)"',
        ' --status "`touch /tmp/monitor-ack-backtick`"',
        r"\; touch /tmp/monitor-ack-escaped-chain",
        ' --note "unterminated',
    ],
)
def test_codex_pretool_rejects_unsafe_or_malformed_monitor_ack_commands(tmp_path, suffix):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "CRITICAL",
                "ack_required": True,
                "summary": "review required",
                "record_id": "record-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    command = (
        f"{TRUSTED_PYTHON} {ack_script} --session-id session-a --queue {queue} "
        f'--record-id record-a --note "read and acknowledged"{suffix}'
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command, session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


@pytest.mark.parametrize(
    "mismatch",
    ["session", "queue", "relative-queue", "record", "duplicate-session"],
)
def test_codex_pretool_requires_exact_pending_monitor_ack_binding(tmp_path, mismatch):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    record = {
        "target_session_id": "session-a",
        "severity": "HARD_GATE",
        "ack_required": True,
        "summary": "review required",
        "record_id": "record-a",
    }
    original = json.dumps(record) + "\n"
    queue.write_text(original, encoding="utf-8")
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    session_args = "--session-id session-a"
    queue_arg = str(queue)
    record_arg = "record-a"
    if mismatch == "session":
        session_args = "--session-id session-b"
    elif mismatch == "queue":
        queue_arg = str(artifact_dir / "other-monitor-interventions.jsonl")
    elif mismatch == "relative-queue":
        queue_arg = queue.name
    elif mismatch == "record":
        record_arg = "record-b"
    elif mismatch == "duplicate-session":
        session_args += " --session-id session-a"
    command = (
        f"{TRUSTED_PYTHON} {ack_script} {session_args} --queue {queue_arg} "
        f'--record-id {record_arg} --note "read and acknowledged"'
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(
            command,
            cwd=artifact_dir if mismatch == "relative-queue" else None,
            session_id="session-a",
        ),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]
    assert queue.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "invocation_style",
    [
        "direct",
        "env",
        "chain",
        "nested-shell",
        "substitution",
        "sed-exec",
        "quoted-script-path",
        "fake-reader",
        "globbed-script-path",
        "fake-python",
        "script-symlink-alias",
        "globbed-script-symlink-alias",
    ],
)
def test_codex_pretool_rejects_monitor_ack_without_a_current_block(
    tmp_path, invocation_style
):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    undiscovered_queue = tmp_path / "other-session-monitor-interventions.jsonl"
    undiscovered_queue.write_text(
        json.dumps(
            {
                "target_session_id": "other-session",
                "severity": "CRITICAL",
                "ack_required": True,
                "record_id": "other-record",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    base_command = (
        f"{TRUSTED_PYTHON} {ack_script} --session-id other-session "
        f"--queue {undiscovered_queue} --record-id other-record "
        '--note "must not acknowledge an unbound queue"'
    )
    fake_reader = tmp_path / "head"
    fake_reader.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_reader.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    script_alias = tmp_path / "queue-helper-alias"
    script_alias.symlink_to(ack_script)
    command = {
        "direct": base_command,
        "env": f"env {base_command}",
        "chain": f"true && {base_command}",
        "nested-shell": f"sh -c '{base_command}'",
        "substitution": f'echo "$({base_command})"',
        "sed-exec": f"sed -n 'e {base_command}' /etc/hostname",
        "quoted-script-path": base_command.replace(
            "monitor_queue_ack.py", "monitor_queue_'ack.py'"
        ),
        "fake-reader": f"{fake_reader} {ack_script}",
        "globbed-script-path": base_command.replace(
            "monitor_queue_ack.py", "monitor_queue_ack.p?"
        ),
        "fake-python": base_command.replace(str(TRUSTED_PYTHON), str(fake_python)),
        "script-symlink-alias": base_command.replace(str(ack_script), str(script_alias)),
        "globbed-script-symlink-alias": base_command.replace(
            str(ack_script), str(script_alias)[:-1] + "?"
        ),
    }[invocation_style]

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command, session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "no matching current blocking record" in payload["reason"]


def test_codex_pretool_allows_static_monitor_ack_inspection_without_a_current_block(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(f"/usr/bin/head -n 80 {ack_script}", session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_rejects_monitor_ack_option_missing_its_value(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "CRITICAL",
                "ack_required": True,
                "record_id": "record-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"
    command = (
        f"{TRUSTED_PYTHON} {ack_script} --session-id session-a --queue {queue} "
        "--record-id record-a --note --all"
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload(command, session_id="session-a"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_monitor_queue_ack_explicit_queue_does_not_fan_out_to_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "session_id": "server-session",
                "codex_thread_id": "session-a",
                "campaign": {"status": "active"},
            }
        ),
        encoding="utf-8",
    )
    explicit_queue = tmp_path / "selected-monitor-interventions.jsonl"
    hidden_queue = artifact_dir / "monitor_interventions.jsonl"
    selected = {
        "target_session_id": "session-a",
        "severity": "CRITICAL",
        "ack_required": True,
        "record_id": "selected-record",
    }
    hidden = {
        "target_session_id": "session-a",
        "severity": "CRITICAL",
        "ack_required": True,
        "record_id": "hidden-record",
    }
    explicit_queue.write_text(json.dumps(selected) + "\n", encoding="utf-8")
    hidden_original = json.dumps(hidden) + "\n"
    hidden_queue.write_text(hidden_original, encoding="utf-8")
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"

    result = subprocess.run(
        [
            sys.executable,
            str(ack_script),
            "--session-id",
            "session-a",
            "--queue",
            str(explicit_queue),
            "--all",
            "--note",
            "selected queue only; do not fan out",
        ],
        cwd=ROOT,
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            AMMO_MONITOR_QUEUE=str(explicit_queue),
            AMMO_SESSION_ID="server-session",
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["changed"] == 1
    assert json.loads(explicit_queue.read_text(encoding="utf-8"))["status"] == "acknowledged"
    assert hidden_queue.read_text(encoding="utf-8") == hidden_original


def test_monitor_queue_ack_rejects_queue_without_current_binding(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    queue = tmp_path / "unbound-monitor-interventions.jsonl"
    original = json.dumps(
        {
            "target_session_id": "other-session",
            "severity": "CRITICAL",
            "ack_required": True,
            "record_id": "other-record",
        }
    ) + "\n"
    queue.write_text(original, encoding="utf-8")
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"

    result = subprocess.run(
        [
            sys.executable,
            str(ack_script),
            "--session-id",
            "other-session",
            "--queue",
            str(queue),
            "--record-id",
            "other-record",
            "--note",
            "must remain unbound",
        ],
        cwd=ROOT,
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            AMMO_SESSION_ID="missing-server-session",
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "no current blocking record matches" in result.stderr
    assert queue.read_text(encoding="utf-8") == original


def test_monitor_queue_ack_rejects_multiple_explicit_queues(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _bind_artifact_to_sessions(
        artifact_dir,
        server_session_id="server-session",
        codex_session_id="session-a",
    )
    current_queue = artifact_dir / "monitor_interventions.jsonl"
    current_original = json.dumps(
        {
            "target_session_id": "session-a",
            "severity": "CRITICAL",
            "ack_required": True,
            "record_id": "current-record",
        }
    ) + "\n"
    current_queue.write_text(current_original, encoding="utf-8")
    unbound_queue = tmp_path / "unbound-monitor-interventions.jsonl"
    unbound_original = json.dumps(
        {
            "target_session_id": "session-a",
            "severity": "CRITICAL",
            "ack_required": True,
            "record_id": "unbound-record",
        }
    ) + "\n"
    unbound_queue.write_text(unbound_original, encoding="utf-8")
    ack_script = ROOT / "ai_cli_session/.codex/skills/ammo/scripts/monitor_queue_ack.py"

    result = subprocess.run(
        [
            sys.executable,
            str(ack_script),
            "--session-id",
            "session-a",
            "--queue",
            str(current_queue),
            "--queue",
            str(unbound_queue),
            "--all",
            "--note",
            "must not union queue bindings",
        ],
        cwd=ROOT,
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="server-session"
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pass exactly one --queue" in result.stderr
    assert current_queue.read_text(encoding="utf-8") == current_original
    assert unbound_queue.read_text(encoding="utf-8") == unbound_original


def test_codex_pretool_blocks_non_ack_command_for_pending_hard_gate(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "CRITICAL",
                "ack_required": True,
                "summary": "review required",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("true"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_codex_pretool_blocks_ack_required_hard_gate(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "summary": "review required",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("true"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_codex_pretool_stage3_reconcile_gate_allows_champion_proposal_write(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal_path = (
        artifact_dir
        / "debate"
        / "campaign_round_1"
        / "proposals"
        / "attention_tail_proposal.md"
    )
    payload = {
        "cwd": str(ROOT),
        "tool_name": "Write",
        "tool_input": {
            "file_path": str(proposal_path),
            "content": "proposal text",
        },
    }

    result = _run_hook(
        "pre_tool_use_guard.py",
        payload,
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_stage3_reconcile_gate_allows_static_inspection(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("rg dispatch_unquantized_gemm vllm"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_stage3_reconcile_gate_still_blocks_state_transition_writes(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "cwd": str(ROOT),
        "tool_name": "Write",
        "tool_input": {
            "file_path": str(artifact_dir / "state.json"),
            "content": json.dumps({"campaign": {"current_stage": "4_5_parallel_tracks"}}),
        },
    }

    result = _run_hook(
        "pre_tool_use_guard.py",
        payload,
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_codex_pretool_stage3_reconcile_exception_does_not_bypass_generic_critical(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "CRITICAL",
                "ack_required": True,
                "category": "unrelated_correctness_issue",
                "summary": "Generic critical intervention.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal_path = (
        artifact_dir
        / "debate"
        / "campaign_round_1"
        / "proposals"
        / "attention_tail_proposal.md"
    )
    payload = {
        "cwd": str(ROOT),
        "tool_name": "Write",
        "tool_input": {"file_path": str(proposal_path), "content": "proposal text"},
    }

    result = _run_hook(
        "pre_tool_use_guard.py",
        payload,
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Generic critical intervention" in payload["reason"]


def test_codex_pretool_stage3_reconcile_gate_only_allows_current_round_proposals(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=2)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    wrong_round_path = (
        artifact_dir
        / "debate"
        / "campaign_round_1"
        / "proposals"
        / "stale_proposal.md"
    )
    current_round_path = (
        artifact_dir
        / "debate"
        / "campaign_round_2"
        / "proposals"
        / "fresh_proposal.md"
    )

    wrong_round = _run_hook(
        "pre_tool_use_guard.py",
        {
            "cwd": str(ROOT),
            "tool_name": "Write",
            "tool_input": {"file_path": str(wrong_round_path), "content": "stale"},
        },
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )
    current_round = _run_hook(
        "pre_tool_use_guard.py",
        {
            "cwd": str(ROOT),
            "tool_name": "Write",
            "tool_input": {"file_path": str(current_round_path), "content": "fresh"},
        },
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    assert _json_stdout(wrong_round)["decision"] == "block"
    assert current_round.returncode == 0
    assert current_round.stdout == ""


def test_codex_pretool_stage3_reconcile_gate_blocks_summary_and_winner_writes(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    for path in (
        artifact_dir / "debate" / "campaign_round_1" / "summary.md",
        artifact_dir / "debate" / "campaign_round_1" / "selected_winners.json",
    ):
        result = _run_hook(
            "pre_tool_use_guard.py",
            {
                "cwd": str(ROOT),
                "tool_name": "Write",
                "tool_input": {"file_path": str(path), "content": "premature"},
            },
            env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
        )

        payload = _json_stdout(result)
        assert payload["decision"] == "block"
        assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_codex_pretool_stage3_reconcile_gate_blocks_proposal_path_traversal_to_summary(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    _write_campaign_round_state(artifact_dir, current_round=1)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "category": "stage3_debate_reconciliation",
                "summary": "Do not leave AMMO Stage 3 until debate artifacts are reconciled.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    traversal_path = (
        artifact_dir
        / "debate"
        / "campaign_round_1"
        / "proposals"
        / ".."
        / "summary.md"
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        {
            "cwd": str(ROOT),
            "tool_name": "Write",
            "tool_input": {"file_path": str(traversal_path), "content": "premature"},
        },
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "Pending AMMO monitor intervention requires acknowledgement" in payload["reason"]


def test_codex_pretool_allows_acknowledged_or_resolved_hard_gate(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "acknowledged_at": "2026-04-30T00:00:00Z",
                "summary": "already acknowledged",
            }
        )
        + "\n"
        + json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "HARD_GATE",
                "ack_required": True,
                "status": "resolved",
                "summary": "already resolved",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("true"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_pretool_monitor_queue_prefers_campaign_session_over_codex_rollout(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    (artifact_dir / "state.json").write_text(
        json.dumps({"session_id": "campaign-id", "campaign": {"status": "active"}}),
        encoding="utf-8",
    )
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "campaign-id",
                "severity": "CRITICAL",
                "ack_required": True,
                "summary": "campaign-level critical",
            }
        )
        + "\n"
        + json.dumps(
            {
                "target_session_id": "campaign-id",
                "severity": "HARD_GATE",
                "ack_required": True,
                "summary": "campaign-level hard gate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("true"),
        env=_clean_env(
            AMMO_ARTIFACT_DIR=str(artifact_dir),
            CODEX_SESSION_ID="subagent-rollout-id",
        ),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "campaign-level critical" in payload["reason"]
    assert "--session-id campaign-id" in payload["reason"]


def test_codex_pretool_campaign_session_ack_command_for_subagent_critical_and_hard_gate(tmp_path):
    for severity in ("CRITICAL", "HARD_GATE"):
        artifact_dir = _ammo_artifact_dir(tmp_path / severity.lower())
        (artifact_dir / "state.json").write_text(
            json.dumps({"session_id": "campaign-id", "campaign": {"status": "active"}}),
            encoding="utf-8",
        )
        queue = artifact_dir / "monitor_interventions.jsonl"
        queue.write_text(
            json.dumps(
                {
                    "target_session_id": "campaign-id",
                    "severity": severity,
                    "ack_required": True,
                    "summary": f"{severity} for campaign",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = _run_hook(
            "pre_tool_use_guard.py",
            _pretool_payload("true"),
            env=_clean_env(
                AMMO_ARTIFACT_DIR=str(artifact_dir),
                CODEX_SESSION_ID="subagent-id",
            ),
        )

        payload = _json_stdout(result)
        assert payload["decision"] == "block"
        assert f"{severity} for campaign" in payload["reason"]
        assert "--session-id campaign-id" in payload["reason"]
        assert "--session-id subagent-id" not in payload["reason"]


def test_codex_pretool_always_blocks_wrong_venv_python_inside_codex_worktree(tmp_path):
    worktree = tmp_path / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True)
    env = _clean_env(AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"), AMMO_SESSION_ID="session-a")

    first = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("python bench.py", cwd=worktree),
        env=env,
    )
    second = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("python bench.py", cwd=worktree),
        env=env,
    )

    first_payload = _json_stdout(first)
    second_payload = _json_stdout(second)
    assert first_payload["decision"] == "block"
    assert second_payload["decision"] == "block"
    assert "must use the worktree virtualenv" in second_payload["reason"]


def test_codex_pretool_blocks_wrong_venv_collect_only_inside_codex_worktree(tmp_path):
    worktree = tmp_path / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True)

    result = _run_hook(
        "pre_tool_use_guard.py",
        _pretool_payload("pytest --collect-only", cwd=worktree),
        env=_clean_env(AMMO_GPU_RES_DIR=str(tmp_path / "gpu_res"), AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "must use the worktree virtualenv" in payload["reason"]


def test_codex_session_start_is_quiet_without_ammo_context(tmp_path):
    result = _run_hook(
        "session_start.py",
        {"cwd": str(tmp_path)},
        env=_clean_env(),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_session_start_is_quiet_at_repo_root_without_active_ammo_context():
    result = _run_hook(
        "session_start.py",
        {"cwd": str(ROOT)},
        env=_clean_env(),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_session_start_emits_context_when_ammo_configured(tmp_path):
    result = _run_hook(
        "session_start.py",
        {"cwd": str(tmp_path)},
        env=_clean_env(AMMO_SESSION_ID="session-a"),
    )

    payload = _json_stdout(result)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "AMMO Codex port available" in payload["hookSpecificOutput"]["additionalContext"]
    assert "WARN blocks stage transitions" not in payload["hookSpecificOutput"]["additionalContext"]
    assert "structured JSON is authoritative" not in payload["hookSpecificOutput"]["additionalContext"]
    assert "Claude AMMO is the behavioral source of truth" in payload["hookSpecificOutput"]["additionalContext"]


def test_codex_stop_hook_allows_preflight_warn_without_stage0_block(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "target"
    runs = artifact_dir / "runs"
    runs.mkdir(parents=True)
    (runs / "preflight_report.json").write_text(
        json.dumps({"overall_status": "WARN"}),
        encoding="utf-8",
    )
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "0_preflight",
                    "rounds": [{"round_id": 1, "debate": {}}],
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_hook(
        "stop_gate_guard.py",
        {"cwd": str(ROOT)},
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "still active at 0_preflight" in payload["reason"]
    assert "WARN" not in payload["reason"]


def test_codex_stop_hook_requires_two_to_three_debate_winners(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "target"
    debate_dir = artifact_dir / "debate" / "campaign_round_1"
    debate_dir.mkdir(parents=True)
    (debate_dir / "summary.md").write_text("summary", encoding="utf-8")
    (debate_dir / "selected_winners.json").write_text(
        json.dumps(["op001"]),
        encoding="utf-8",
    )
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "3_debate",
                    "rounds": [
                        {
                            "round_id": 1,
                            "debate": {
                                "completed_at": "2026-04-30T00:00:00Z",
                                "candidates": ["op001", "op002"],
                                "selected_winners": ["op001"],
                                "selection_rationale": "op001 looked best",
                                "rounds_completed": 2,
                                "max_rounds": 3,
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_hook(
        "stop_gate_guard.py",
        {"cwd": str(ROOT)},
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "selected_winners must list 2-3 selected winners" in payload["reason"]


def test_codex_posttool_ignores_static_sweep_mentions_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "post_tool_use_guard.py",
        _pretool_payload("rg run_vllm_bench_latency_sweep.py docs"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_posttool_suppresses_pending_monitor_notice_for_unrelated_command(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    queue = artifact_dir / "monitor_interventions.jsonl"
    queue.write_text(
        json.dumps(
            {
                "target_session_id": "session-a",
                "severity": "WARNING",
                "summary": "review artifact metrics",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        _pretool_payload("true"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir), AMMO_SESSION_ID="session-a"),
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_codex_posttool_advises_after_actual_sweep_command_in_ammo_context(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "post_tool_use_guard.py",
        _pretool_payload("python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --artifact-dir out"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    payload = _json_stdout(result)
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "Review run_purpose" in payload["hookSpecificOutput"]["additionalContext"]


def test_codex_posttool_explains_evidence_complete_no_pass_is_exhausted_only(tmp_path):
    artifact_dir = _ammo_artifact_dir(tmp_path)
    result = _run_hook(
        "post_tool_use_guard.py",
        _pretool_payload("python .codex/skills/ammo/scripts/verify_validation_gates.py kernel_opt_artifacts/target"),
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    payload = _json_stdout(result)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "EVIDENCE_COMPLETE_NO_PASS" in context
    assert "mark the round EXHAUSTED" in context
    assert "not candidate success" in context


def test_codex_stop_hook_accepts_evidence_complete_no_pass_for_stage6_exhaustion(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "target"
    round_dir = artifact_dir / "rounds" / "1"
    round_dir.mkdir(parents=True)
    (round_dir / "validation_gate_report.json").write_text(
        json.dumps({"overall_status": "EVIDENCE_COMPLETE_NO_PASS", "advance_to_stage6": True}),
        encoding="utf-8",
    )
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "4_5_parallel_tracks",
                    "rounds": [{"round_id": 1, "debate": {}}],
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_hook(
        "stop_gate_guard.py",
        {"cwd": str(ROOT)},
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    # EVIDENCE_COMPLETE_NO_PASS is an accepted gate outcome: the block message
    # must direct the round forward, not demand a re-run of the gate.
    payload = _json_stdout(result)
    assert payload["decision"] == "block"
    assert "validation gate is missing" not in payload["reason"]
    assert "T_AUDIT_S45" in payload["reason"]


def test_codex_stop_hook_requires_report_but_not_maintenance_or_bundle_artifacts(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "target"
    artifact_dir.mkdir(parents=True)
    report_body = b"# Report\n"
    (artifact_dir / "REPORT.md").write_bytes(report_body)
    fact_check = artifact_dir / "report_assets" / "report_fact_check.json"
    fact_check.parent.mkdir(parents=True)
    fact_check.write_text(
        json.dumps(
            {
                "ok": True,
                "report_sha256": hashlib.sha256(report_body).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "campaign_complete",
                    "current_round": 1,
                    "current_stage": "7_report",
                    "rounds": [{"round_id": 1, "debate": {}}],
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_hook(
        "stop_gate_guard.py",
        {"cwd": str(ROOT)},
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact_dir)),
    )

    assert result.returncode == 0
    assert result.stdout == ""
