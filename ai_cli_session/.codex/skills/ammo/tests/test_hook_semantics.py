# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Regression tests for Codex AMMO hook drift fixes."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HOOKS = ROOT / ".codex" / "hooks"


def _run_hook(script: str, payload: dict, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        timeout=10,
    )


def _hook_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".codex" / "hooks").mkdir(parents=True, exist_ok=True)
    _write_stage_enum_schema(repo)
    return repo


def _env(repo: Path, artifact_dir: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AMMO_ARTIFACT_DIR": str(artifact_dir),
            "AMMO_SESSION_ID": "sess-test",
            "AMMO_GPU_RES_DIR": str(repo / "gpu_res"),
            "AMMO_AUDIT_PHASE2_DIR": str(repo / "audit_phase2"),
        }
    )
    env.update(extra)
    return env


def _active_stage1_artifact(repo: Path) -> Path:
    artifact = repo / "kernel_opt_artifacts" / "campaign"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".codex" / "skills" / "ammo" / "scripts" / "new_target.py"),
            "--artifact-dir", str(artifact),
            "--model-id", "test-model",
            "--hardware", "H100",
            "--dtype", "bf16",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["session_id"] = "sess-test"
    # Most hook fixtures model resumable pre-v4.2 campaigns. Dedicated v4.2
    # tests cover the selected-cohort/pairing contract.
    state["campaign"]["schema_version"] = "4.1"
    # Keep fixtures independent from the developer's live Codex thread.
    state["codex_thread_id"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return artifact


def _active_stage2_artifact(repo: Path) -> Path:
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    state["campaign"]["rounds"][0]["bottleneck_mining"] = {
        "started_at": "2026-04-30T00:00:00Z",
        "completed_at": "2026-04-30T00:10:00Z",
        "top_component": "moe",
        "top_f_decode_pct": 17.6,
        "top_f_e2e_pct": 12.3,
        "top_addressable_e2e_pct": 1.12,
        "amdahl_ceiling": 1.12,
        "decode_frac": 0.7,
        "component_breakdown": [{"name": "moe", "pct": 12.3}],
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    analysis = artifact / "rounds" / "1" / "mining" / "bottleneck_analysis.md"
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(
        "# Bottleneck Analysis\n\n## Technology Landscape\n\n### moe\n- Authoring class: unknown\n",
        encoding="utf-8",
    )
    return artifact


def _set_two_rounds(state: dict, previous_audit: dict | None) -> None:
    """Expand the current schema-valid round fixture without partial objects."""
    template = state["campaign"]["rounds"][0]
    previous = json.loads(json.dumps(template))
    current = json.loads(json.dumps(template))
    previous["round_id"] = 1
    previous["status"] = "SHIPPED"
    previous["integration"]["status"] = "validated"
    if previous_audit is None:
        previous.pop("audit", None)
    else:
        previous["audit"] = previous_audit
    current["round_id"] = 2
    current["status"] = "IN_PROGRESS"
    current["audit"] = {}
    state["campaign"]["rounds"] = [previous, current]


def _write_validation_report(artifact: Path, round_id: int = 1, status: str = "PASS") -> None:
    path = artifact / "rounds" / str(round_id) / "validation_gate_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"overall_status": status}), encoding="utf-8")


def _write_audit_invariants(repo: Path) -> None:
    path = repo / ".codex" / "skills" / "ammo" / "references" / "audit-invariants.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Audit Invariants\n\n## Pre-Check\n", encoding="utf-8")


def test_stop_guard_active_campaign_cannot_bypass_by_retry(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    first = _run_hook("stop_gate_guard.py", payload, env)
    second = _run_hook("stop_gate_guard.py", payload, env)
    third = _run_hook("stop_gate_guard.py", payload, env)

    assert first.returncode == 0
    assert json.loads(first.stdout)["decision"] == "block"
    assert second.returncode == 0
    assert json.loads(second.stdout)["decision"] == "block"
    assert third.returncode == 0
    assert json.loads(third.stdout)["decision"] == "block"


def test_stop_guard_allow_env_bypasses_first_block(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact, AMMO_ALLOW_STOP="1")

    result = _run_hook("stop_gate_guard.py", {"cwd": str(repo), "session_id": "sess-test"}, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_stop_guard_reevaluates_gate_on_every_stop(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    first = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(first.stdout)["decision"] == "block"

    constraints = artifact / "rounds" / "1" / "constraints.md"
    constraints.parent.mkdir(parents=True, exist_ok=True)
    constraints.write_text("# Constraints\n", encoding="utf-8")
    second = _run_hook("stop_gate_guard.py", payload, env)
    assert second.stdout.strip() == ""

    constraints.unlink()
    third = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(third.stdout)["decision"] == "block"


def test_stop_guard_requires_stage2_audit_for_stage2_completion(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage2_artifact(repo)
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    first = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(first.stdout)["decision"] == "block"
    assert "audit.stage_2.passed_at" in first.stdout

    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_2": {"passed_at": "2026-04-30T00:11:00Z"}
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    second = _run_hook("stop_gate_guard.py", payload, env)
    assert second.stdout.strip() == ""


def test_stop_guard_stage7_names_addressable_impact_not_raw_share(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)

    result = _run_hook(
        "stop_gate_guard.py",
        {"cwd": str(repo), "session_id": "stage7-addressable"},
        env,
    )

    body = json.loads(result.stdout)
    assert "top_addressable_e2e_pct" in body["reason"]
    assert "top bottleneck share" not in body["reason"]


def test_stop_guard_v2_constraints_are_round_scoped_not_root_duplicated(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (artifact / "constraints.md").write_text("# stale root constraints\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "constraints-v2"}

    missing = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(missing.stdout)["decision"] == "block"
    assert "rounds/1/constraints.md" in missing.stdout

    canonical = artifact / "rounds" / "1" / "constraints.md"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("# canonical constraints\n", encoding="utf-8")
    fixed = _run_hook("stop_gate_guard.py", payload, env)
    assert fixed.stdout.strip() == ""


def test_stop_guard_requires_report_fact_check_for_final_campaign(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["status"] = "campaign_complete"
    state["campaign"]["current_stage"] = "7b_report"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (artifact / "REPORT.md").write_text("# Report\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    first = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(first.stdout)["decision"] == "block"
    assert "Report fact-check review artifact missing" in first.stdout

    repeated = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(repeated.stdout)["decision"] == "block"
    assert "Report fact-check review artifact missing" in repeated.stdout

    fact_check = artifact / "report_assets" / "report_fact_check.json"
    fact_check.parent.mkdir(parents=True, exist_ok=True)
    report_sha = hashlib.sha256((artifact / "REPORT.md").read_bytes()).hexdigest()
    fact_check.write_text(
        json.dumps({"ok": True, "report_sha256": report_sha}), encoding="utf-8"
    )
    second = _run_hook("stop_gate_guard.py", payload, env)
    assert second.stdout.strip() == ""

    (artifact / "REPORT.md").write_text("# Changed report\n", encoding="utf-8")
    stale = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(stale.stdout)["decision"] == "block"
    assert "stale" in stale.stdout


def test_stop_guard_reads_round_scoped_validation_gate_report(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "4_5_parallel_tracks"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    first = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(first.stdout)["decision"] == "block"
    assert "AMMO validation gate is missing" in first.stdout

    _write_validation_report(artifact, status="PASS")
    second = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(second.stdout)["decision"] == "block"
    assert "Run T_AUDIT_S45" in second.stdout
    third = _run_hook("stop_gate_guard.py", payload, env)
    assert json.loads(third.stdout)["decision"] == "block"
    assert "active campaign is not complete" in third.stdout


def test_stop_guard_ignores_stale_runs_validation_report_for_v2_campaign(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (artifact / "rounds" / "1").mkdir(parents=True, exist_ok=True)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "4_5_parallel_tracks"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runs = artifact / "runs"
    runs.mkdir()
    (runs / "validation_gate_report.json").write_text(
        json.dumps({"overall_status": "PASS"}),
        encoding="utf-8",
    )
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "session_id": "sess-test"}

    result = _run_hook("stop_gate_guard.py", payload, env)

    assert json.loads(result.stdout)["decision"] == "block"
    assert "AMMO validation gate is missing" in result.stdout


def _write_monitor_record(artifact: Path) -> None:
    (artifact / "monitor_interventions.jsonl").write_text(
        json.dumps(
            {
                "target_session_id": "sess-test",
                "severity": "CRITICAL",
                "summary": "Stop unsafe mutation",
                "evidence": "evidence",
                "recommended_action": "ack or fix",
                "ack_required": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_monitor_ack_allows_read_only_inspection(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_monitor_record(artifact)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "tool_input": {"command": "rg current_stage kernel_opt_artifacts/campaign/state.json"},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_monitor_ack_allows_claude_source_read_only_inspection(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_monitor_record(artifact)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "tool_input": {"command": "cat ai_cli_session/.codex/skills/ammo/SKILL.md"},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_claude_source_execution_still_blocked(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "tool_input": {
            "command": "python ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py status"
        },
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "Use .codex/skills/ammo" in body["reason"]


def test_monitor_ack_still_blocks_mutating_command(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_monitor_record(artifact)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "tool_input": {"command": "python ai_cli_session/.codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py"},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_spawn_agent_post_hook_reminds_to_spawn_monitor_for_implementer(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "implementer-op001",
            "agent_type": "ammo-implementer",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    context = body["hookSpecificOutput"]["additionalContext"]
    assert "AMMO MONITOR REMINDER" in context
    assert "ammo-transcript-monitor" in context
    assert "monitor_op001" in context


def test_spawn_agent_post_hook_reminds_for_legacy_impl_champion_team_lead_payload(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "agentName": "team-lead",
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "impl-champion-op001",
            "agent_type": "ammo-impl-champion",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "AMMO MONITOR REMINDER" in context
    assert "monitor_champion_op001" in context


def test_spawn_agent_post_hook_reminds_for_implementer_team_lead_transcript(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    transcript = repo / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "permission-mode"}) + "\n"
        + json.dumps({"agentName": "team-lead", "type": "user"}) + "\n",
        encoding="utf-8",
    )
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "transcript_path": str(transcript),
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "implementer-op001",
            "agent_type": "ammo-implementer",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "AMMO MONITOR REMINDER" in context
    assert "monitor_op001" in context


def test_spawn_agent_post_hook_suppresses_debate_champion_monitor(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "champion-1",
            "agent_type": "ammo-champion",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_spawn_agent_post_hook_suppresses_for_child_agent_payload(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "agentName": "champion-1",
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "champion-2",
            "agent_type": "ammo-champion",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_v2_subagent_start_maps_round_task_slug_to_raw_op_id(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "OP-001": {"status": "IN_PROGRESS"}
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    codex_home = repo / "codex_home"
    codex_home.mkdir()
    connection = sqlite3.connect(codex_home / "state_5.sqlite")
    try:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, agent_path TEXT)")
        connection.execute(
            "INSERT INTO threads (id, agent_path) VALUES (?, ?)",
            ("implementer-thread", "/root/implementer_r1_op_001"),
        )
        connection.execute(
            "INSERT INTO threads (id, agent_path) VALUES (?, ?)",
            ("monitor-thread", "/root/monitor_r1_op_001"),
        )
        connection.commit()
    finally:
        connection.close()

    env = _env(
        repo,
        artifact,
        CODEX_HOME=str(codex_home),
        CODEX_SESSION_ID="root-thread",
    )
    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "hook_event_name": "SubagentStart",
            "agent_id": "implementer-thread",
        },
        env,
    )

    assert result.returncode == 0
    ledgers = list((repo / "gpu_res").glob("codex_monitor_pairs_*.json"))
    assert len(ledgers) == 1
    record = json.loads(ledgers[0].read_text(encoding="utf-8"))["pending"][0]
    assert record == {
        "expected_monitor_name": "monitor_r1_op_001",
        "implementer_agent_id": "implementer-thread",
        "implementer_name": "implementer_r1_op_001",
        "op_id": "OP-001",
        "status": "pending",
    }

    monitor_start = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "hook_event_name": "SubagentStart",
            "agent_id": "monitor-thread",
        },
        env,
    )
    assert monitor_start.returncode == 0

    audit_dir = artifact / "rounds" / "1" / "tracks" / "OP-001" / "monitor_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    observations = audit_dir / "monitor_observations.md"
    observations.write_text("# Monitor observations\n\nNo unresolved issue.\n", encoding="utf-8")
    (artifact / "monitor_interventions.jsonl").write_text(
        json.dumps(
            {
                "emitter": "ammo-transcript-monitor",
                "severity": "INFO",
                "target_rollout_id": "implementer-thread",
                "summary": "Final poll completed without an unresolved issue",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monitor_stop = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "hook_event_name": "SubagentStop",
            "agent_id": "monitor-thread",
        },
        env,
    )
    assert monitor_stop.returncode == 0
    assert '"decision":"block"' not in monitor_stop.stdout.replace(" ", "")
    completed = json.loads(ledgers[0].read_text(encoding="utf-8"))["pending"][0]
    assert completed["status"] == "satisfied"
    assert completed["summary_path"] == str(observations)


def test_spawn_agent_post_hook_registered_for_codex_spawn_tool():
    hooks_json = json.loads((ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    matchers = [
        entry.get("matcher", "")
        for entry in hooks_json["hooks"].get("PostToolUse", [])
    ]

    assert any("spawn_agent" in matcher for matcher in matchers)


def test_session_start_hook_matches_claude_active_state_resume_context(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)

    result = _run_hook("session_start.py", {"cwd": str(repo)}, env)

    assert result.returncode == 0
    assert "AMMO Optimization Detected" in result.stdout
    assert "Existing optimization state at" in result.stdout
    assert ".codex/skills/ammo/SKILL.md" in result.stdout
    assert "$ammo" not in result.stdout


def test_session_start_holds_paused_campaign_without_checkpoint(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["status"] = "paused"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = _run_hook("session_start.py", {"cwd": str(repo)}, _env(repo, artifact))

    assert result.returncode == 0
    assert "AMMO Campaign Paused" in result.stdout
    assert "until the user explicitly resumes" in result.stdout
    assert "Resume current stage" not in result.stdout


def test_session_start_hook_matches_claude_compaction_resume_context(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (artifact / "compaction_checkpoint.json").write_text(
        json.dumps(
            {
                "model": "test-model",
                "stage": "4_5_parallel_tracks",
                "track_count": 2,
                "campaign_round": 1,
                "campaign_status": "active",
                "cumulative_speedup": 1.05,
            }
        ),
        encoding="utf-8",
    )
    env = _env(repo, artifact)

    result = _run_hook("session_start.py", {"cwd": str(repo)}, env)

    assert result.returncode == 0
    assert "Session Resumed After Compaction" in result.stdout
    assert "Parallel tracks (2 active)" in result.stdout
    assert "test-model" in result.stdout
    assert not (artifact / "compaction_checkpoint.json").exists()


def test_session_start_holds_paused_campaign_after_compaction(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["status"] = "paused"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    checkpoint = artifact / "compaction_checkpoint.json"
    checkpoint.write_text(
        json.dumps({"campaign_status": "paused", "campaign_stage": "1_baseline"}),
        encoding="utf-8",
    )

    result = _run_hook("session_start.py", {"cwd": str(repo)}, _env(repo, artifact))

    assert result.returncode == 0
    assert "AMMO Campaign Paused" in result.stdout
    assert "Resume current stage" not in result.stdout
    assert not checkpoint.exists()


def test_lifecycle_hooks_do_not_register_codex_only_prompt_reminder():
    hooks_json = json.loads((ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    session_start_hooks = hooks_json["hooks"].get("SessionStart", [])
    commands = [
        hook
        for entry in session_start_hooks
        for hook in entry.get("hooks", [])
        if "session_start.py" in hook.get("command", "")
    ]

    assert commands
    assert all("statusMessage" not in hook for hook in commands)
    assert "UserPromptSubmit" not in hooks_json["hooks"]


def test_env_default_guard_uses_file_path_not_patch_text(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    envs = repo / "vllm" / "envs.py"
    envs.parent.mkdir(parents=True, exist_ok=True)
    envs.write_text("", encoding="utf-8")
    payload = {
        "cwd": str(repo),
        "tool_input": {
            "file_path": str(envs),
            "new_string": 'VLLM_OP123: bool = True\n',
        },
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "default off" in body["reason"]


def test_post_hook_blocks_invalid_state_json_schema(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["target"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}
    env = _env(repo, artifact)

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "state.json validation hard issue" in body["reason"]


def test_post_hook_blocks_stage2_without_stage1_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    state["campaign"]["rounds"][0]["audit"] = {}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "Audit gate (4-phase audit)" in body["reason"]
    assert "stage_1 first" in body["reason"]


def test_post_hook_detects_bare_state_json_command(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    state["campaign"]["rounds"][0]["audit"] = {}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {
        "cwd": str(artifact),
        "tool_input": {"command": "jq '.campaign.current_stage=\"2_bottleneck_mining\"' state.json > state.json.tmp && mv state.json.tmp state.json"},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Audit gate (4-phase audit)" in result.stdout
    assert "stage_1 first" in result.stdout


def test_post_hook_allows_stage2_after_stage1_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    state["campaign"]["rounds"][0]["audit"] = {"stage_1": {"passed_at": "2026-05-06T00:00:00Z"}}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_blocks_stage3_without_stage2_audit_for_v41(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["schema_version"] = "4.1"
    state["campaign"]["current_stage"] = "3_debate"
    state["campaign"]["rounds"][0]["audit"] = {"stage_1": {"passed_at": "2026-05-06T00:00:00Z"}}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "Audit gate (4-phase audit)" in body["reason"]
    assert "stage_2 first" in body["reason"]


def test_post_hook_allows_stage3_without_stage2_audit_for_legacy_schema(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["schema_version"] = "4.0"
    state["campaign"]["current_stage"] = "3_debate"
    state["campaign"]["rounds"][0]["audit"] = {"stage_1": {"passed_at": "2026-05-06T00:00:00Z"}}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_blocks_stage6_without_stage45_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {"stage_1": {"passed_at": "2026-05-06T00:00:00Z"}}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Audit gate (4-phase audit)" in result.stdout
    assert "stage_45 first" in result.stdout


def test_post_hook_blocks_stage6_with_in_progress_track_even_after_stage45_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "IN_PROGRESS"}
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "Stage 6 transition blocked" in body["reason"]
    assert "op001=IN_PROGRESS" in body["reason"]


def test_post_hook_blocks_stage6_with_gating_required_track_even_after_stage45_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "GATING_REQUIRED"}
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "op001=GATING_REQUIRED" in body["reason"]


def test_post_hook_blocks_stage6_with_gpu_blocked_track_even_after_stage45_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "GPU_BLOCKED"}
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "op001=GPU_BLOCKED" in body["reason"]


def test_post_hook_allows_stage6_with_terminal_tracks_after_stage45_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "PASS"},
        "op002": {"status": "GATED_PASS"},
        "op003": {"status": "FAIL"},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Stage 6 transition blocked" not in result.stdout
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_blocks_stage7_without_stage67_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Audit gate (4-phase audit)" in result.stdout
    assert "stage_67 first" in result.stdout


def test_post_hook_blocks_stage7_with_null_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state["campaign"]["rounds"][0]["audit"] = None
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Audit gate (4-phase audit)" in result.stdout
    assert "stage_67 first" in result.stdout


def test_post_hook_accepts_stage7_with_stage67_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_67": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_accepts_stage7_with_legacy_stage6_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "7_campaign_eval"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_6": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_allows_post_ship_remine_without_same_round_stage1_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = 2
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    _set_two_rounds(
        state,
        {"stage_67": {"passed_at": "2026-05-06T00:00:00Z"}},
    )
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "stage_1 first" not in result.stdout
    assert "new round start blocked" not in result.stdout


def test_post_hook_post_ship_remine_exemption_fails_closed_without_previous_audit_key(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = 2
    state["campaign"]["current_stage"] = "2_bottleneck_mining"
    _set_two_rounds(state, None)
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "stage_1 first" in result.stdout


def test_post_hook_blocks_new_round_without_previous_stage7_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = 2
    state["campaign"]["current_stage"] = "3_debate"
    _set_two_rounds(
        state,
        {"stage_6": {"passed_at": "2026-05-06T00:00:00Z"}},
    )
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "new round start blocked" in result.stdout
    assert "stage_67 first" in result.stdout


def test_post_hook_accepts_new_round_with_previous_stage67_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = 2
    state["campaign"]["current_stage"] = "3_debate"
    _set_two_rounds(
        state,
        {"stage_67": {"passed_at": "2026-05-06T00:00:00Z"}},
    )
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_accepts_new_round_with_previous_legacy_stage7_audit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_round"] = 2
    state["campaign"]["current_stage"] = "3_debate"
    _set_two_rounds(
        state,
        {"stage_7": {"passed_at": "2026-05-06T00:00:00Z"}},
    )
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED" not in result.stdout


def test_post_hook_injects_auditor_phase2_after_phase1_verdict_write(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_audit_invariants(repo)
    verdict = artifact / "rounds" / "1" / "audits" / "stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("# Audit Verdict\n\n## Phase 1 - Independent Reconstruction\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "agent_type": "ammo-auditor",
        "agent_id": "auditor-1",
        "tool_input": {"file_path": str(verdict)},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    context = body["hookSpecificOutput"]["additionalContext"]
    assert "Phase 1 written successfully. Now complete Phase 2" in context
    assert "audit-invariants.md" in context
    assert "Phase 2 cannot downgrade Phase 1 BLOCKING" in context


def test_post_hook_does_not_inject_auditor_phase2_after_phase2_exists(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_audit_invariants(repo)
    verdict = artifact / "rounds" / "1" / "audits" / "stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("# Audit Verdict\n\n## Phase 2 - Checklist Verification\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "agent_type": "ammo-auditor",
        "agent_id": "auditor-1",
        "tool_input": {"file_path": str(verdict)},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Phase 2" not in result.stdout


def test_post_hook_phase2_injection_is_auditor_local(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_audit_invariants(repo)
    verdict = artifact / "rounds" / "1" / "audits" / "stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("# Audit Verdict\n\n## Phase 1 - Independent Reconstruction\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "agent_type": "team-lead",
        "tool_input": {"file_path": str(verdict)},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Phase 1 written successfully" not in result.stdout


def test_subagent_stop_blocks_auditor_until_hook_injected_phase2_is_written(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_audit_invariants(repo)
    constraints = artifact / "rounds" / "1" / "constraints.md"
    constraints.parent.mkdir(parents=True, exist_ok=True)
    constraints.write_text("# Constraints\n", encoding="utf-8")
    verdict = artifact / "rounds" / "1" / "audits" / "stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("# Audit Verdict\n\n## Phase 1 - Independent Reconstruction\n", encoding="utf-8")
    env = _env(repo, artifact)
    post_payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "agent_type": "ammo-auditor",
        "agent_id": "auditor-1",
        "tool_input": {"file_path": str(verdict)},
    }
    post = _run_hook("post_tool_use_guard.py", post_payload, env)
    assert post.returncode == 0
    assert "Phase 1 written successfully" in post.stdout

    stop_payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "hook_event_name": "SubagentStop",
        "agent_type": "ammo-auditor",
        "agent_id": "auditor-1",
    }
    blocked = _run_hook("post_tool_use_guard.py", stop_payload, env)
    assert blocked.returncode == 0
    body = json.loads(blocked.stdout)
    assert body["decision"] == "block"
    assert "append the required Phase 2 checklist/reconciliation" in body["reason"]

    verdict.write_text(
        "# Audit Verdict\n\n## Phase 1 - Independent Reconstruction\n\n## Phase 2 - Checklist Verification\n",
        encoding="utf-8",
    )
    allowed = _run_hook("post_tool_use_guard.py", stop_payload, env)
    assert allowed.returncode == 0
    assert allowed.stdout.strip() == ""


def test_post_hook_ignores_stage45_partial_evidence_on_auditor_stop(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    partial = artifact / "rounds" / "1" / "audits" / "stage_45_partial_OP-001.md"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("# Read-only partial evidence\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "session_id": "sess-test",
        "hook_event_name": "SubagentStop",
        "agent_type": "ammo-auditor",
        "agent_id": "auditor-1",
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_stage45_terminal_tracks_reminds_missing_e2e_latency_opt(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "4_5_parallel_tracks"
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {
            "status": "PASS",
            "per_bs_verdict": {"1": "PASS", "8": "NOISE"},
            "e2e_latency_opt": None,
        },
        "op002": {
            "status": "FAIL",
            "per_bs_verdict": {"1": "REGRESSED"},
            "e2e_latency_opt": {"1": {"avg": 1.23, "p50": 1.2}},
        },
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "1 track(s) have per_bs_verdict but null e2e_latency_opt" in result.stdout
    assert "avg_s" in result.stdout and "avg" in result.stdout
    assert "Same shape as baseline.e2e_latency" in result.stdout


def test_post_hook_stage45_running_tracks_do_not_remind_missing_e2e_latency_opt(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "4_5_parallel_tracks"
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {
            "status": "IN_PROGRESS",
            "per_bs_verdict": {"1": "PASS"},
            "e2e_latency_opt": None,
        }
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "null e2e_latency_opt" not in result.stdout


def test_post_hook_stage6_single_passing_track_reminds_short_circuit(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "PASS"}
    }
    state["campaign"]["rounds"][0]["integration"]["status"] = "pending"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Single passer" in result.stdout
    assert "run short-circuit" in result.stdout
    assert "Run the integration sweep with --fresh-cache" not in result.stdout


def test_post_hook_stage6_multiple_passing_tracks_reminds_fresh_cache(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {
        "op001": {"status": "PASS"},
        "op002": {"status": "GATED_PASS"},
    }
    state["campaign"]["rounds"][0]["integration"]["status"] = "pending"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "Multiple passers" in result.stdout
    assert "--fresh-cache" in result.stdout


def test_post_hook_stage6_exhausted_requires_stage67_auditor(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "6_integration"
    state["campaign"]["rounds"][0]["audit"] = {
        "stage_1": {"passed_at": "2026-05-06T00:00:00Z"},
        "stage_45": {"passed_at": "2026-05-06T00:00:00Z"},
    }
    state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"] = {}
    state["campaign"]["rounds"][0]["integration"]["status"] = "exhausted"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AUDIT REQUIRED (T_AUDIT_S67)" in result.stdout
    assert "No auto-pass" in result.stdout
    assert "auditor always runs" in result.stdout


def test_post_hook_warns_on_artifact_layout_drift(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    misplaced = artifact / "monitor_log_champion_1.md"
    misplaced.write_text("# misplaced\n", encoding="utf-8")
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(misplaced)}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "LAYOUT WARN" in result.stdout
    assert "monitor_log_champion_1.md is outside the canonical AMMO V2 layout" in result.stdout


def test_subagent_start_cwd_warns_from_non_root(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    worktree = repo / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True, exist_ok=True)
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(worktree),
        "hook_event_name": "SubagentStart",
        "tool_input": {"message": "x"},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    context = body["hookSpecificOutput"]["additionalContext"]
    assert "outside the AMMO project root" in context
    assert str(repo) in context


def test_pretool_spawn_cwd_silent_at_root(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {"cwd": str(repo), "tool_name": "functions.spawn_agent", "tool_input": {"message": "x"}}

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_subagent_start_cwd_blocks_typed_spawn_from_non_root(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    worktree = repo / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True, exist_ok=True)
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(worktree),
        "hook_event_name": "SubagentStart",
        "tool_input": {
            "agent_type": "ammo-implementer",
            "task_name": "impl_op001",
            "message": "OP_ID: OP-001",
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    context = body["hookSpecificOutput"]["additionalContext"]
    assert "outside the AMMO project root" in context
    assert str(repo) in context


def test_pretool_spawn_cwd_silent_inside_current_child_agent(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    worktree = repo / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True, exist_ok=True)
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(worktree),
        "agent_type": "ammo-implementer",
        "tool_name": "functions.spawn_agent",
        "tool_input": {"agent_type": "ammo-delegate", "message": "x"},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_pretool_worktree_venv_guard_cannot_be_bypassed_by_retry(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    worktree = repo / ".codex" / "worktrees" / "op001"
    worktree.mkdir(parents=True, exist_ok=True)
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(worktree),
        "session_id": "sess-test",
        "tool_input": {"command": "python --version"},
    }

    first = _run_hook("pre_tool_use_guard.py", payload, env)
    second = _run_hook("pre_tool_use_guard.py", payload, env)

    assert first.returncode == 0
    body = json.loads(first.stdout)
    assert body["decision"] == "block"
    assert "worktree virtualenv" in body["reason"]
    assert str(worktree / ".venv" / "bin") in body["reason"]
    assert second.returncode == 0
    assert json.loads(second.stdout)["decision"] == "block"


def test_pretool_blocks_child_edits_to_orchestrator_owned_ammo_files(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    protected = repo / ".codex" / "skills" / "ammo" / "references" / "optimization-categories.md"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("# protected\n", encoding="utf-8")
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(repo),
        "agent_type": "ammo-implementer",
        "tool_input": {"file_path": str(protected), "new_string": "# changed\n"},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "Subagents may not edit" in body["reason"]
    assert ".codex/skills/ammo/references/" in body["reason"]


def test_pretool_allows_lead_edits_to_orchestrator_owned_ammo_files(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    protected = repo / ".codex" / "agents" / "ammo-champion.toml"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text('name = "ammo-champion"\n', encoding="utf-8")
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(repo),
        "tool_input": {"file_path": str(protected), "new_string": 'name = "ammo-champion"\n'},
    }

    result = _run_hook("pre_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_pretool_always_blocks_package_install_and_uninstall(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    commands = [
        "pip install foo",
        "pip3 uninstall foo",
        "uv pip install -e .",
        "python -m pip uninstall foo",
        "cd /tmp && FEATURE=1 uv pip install foo",
        "/usr/bin/pip3 install foo",
        ".venv/bin/python -m pip install foo",
        "env pip install foo",
        "command pip install foo",
        "pip --quiet install foo",
        "bash -c 'pip install foo'",
        "sh -lc 'uv pip uninstall foo'",
    ]

    for command in commands:
        result = _run_hook(
            "pre_tool_use_guard.py",
            {"cwd": str(repo), "tool_input": {"command": command}},
            env,
        )
        assert result.returncode == 0, command
        body = json.loads(result.stdout)
        assert body["decision"] == "block", command
        assert "package install/uninstall" in body["reason"], command


def test_pretool_pip_guard_is_active_without_ammo_campaign_context(tmp_path):
    repo = _hook_repo(tmp_path)
    env = os.environ.copy()
    env.update({"AMMO_GPU_RES_DIR": str(repo / "gpu_res")})

    result = _run_hook(
        "pre_tool_use_guard.py",
        {"cwd": str(repo), "tool_input": {"command": "pip install foo"}},
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"


def test_pretool_pip_guard_allows_quoted_search_mentions_and_read_only_forms(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    commands = [
        'grep "pip install" file.py',
        "echo 'do not uv pip install things'",
        'rg "uv pip uninstall" docs/',
        'git log --grep="pip install"',
        "pip list",
        "uv pip show vllm",
    ]

    for command in commands:
        result = _run_hook(
            "pre_tool_use_guard.py",
            {"cwd": str(repo), "tool_input": {"command": command}},
            env,
        )
        assert result.returncode == 0, command
        assert result.stdout.strip() == "", command


def test_pretool_pip_guard_honors_explicit_provisioning_escape_hatch(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact, AMMO_ALLOW_PIP="1")

    result = _run_hook(
        "pre_tool_use_guard.py",
        {"cwd": str(repo), "tool_input": {"command": "uv pip install -e ."}},
        env,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def _write_stage_enum_schema(repo: Path) -> None:
    path = repo / ".codex" / "schemas" / "state.schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["campaign"],
                "properties": {
                    "campaign": {
                        "type": "object",
                        "required": ["current_stage"],
                        "properties": {
                            "current_stage": {
                                "enum": [
                                    "1_baseline",
                                    "2_bottleneck_mining",
                                    "3_debate",
                                    "4_5_parallel_tracks",
                                    "6_integration",
                                    "7_campaign_eval",
                                    "7b_report",
                                ]
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_post_hook_detects_variable_based_atomic_state_mutation(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_stage_enum_schema(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "invalid_stage"
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    command = (
        'STATE="$AMMO_ARTIFACT_DIR/state.json"; '
        'TMP=$(mktemp); python -c "import json; json.dump({}, open(\"$TMP\", \"w\"))"; '
        'mv "$TMP" "$STATE"'
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_input": {"command": command}, "tool_response": {"status": "success"}},
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "invalid_stage" in body["reason"] or "is not one of" in body["reason"]


def test_post_hook_uses_managed_schema_not_mutable_repo_copy(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (repo / ".codex" / "schemas" / "state.schema.json").unlink()
    env = _env(repo, artifact)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_state_schema_unavailable_escape_hatch(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (repo / ".codex" / "schemas" / "state.schema.json").unlink()
    env = _env(repo, artifact, AMMO_VALIDATE_FAIL_OPEN="1")
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_blocks_when_jsonschema_library_is_unavailable(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_stage_enum_schema(repo)
    shim = repo / "python_shim"
    shim.mkdir()
    (shim / "jsonschema.py").write_text("raise ImportError('simulated unavailable jsonschema')\n", encoding="utf-8")
    env = _env(repo, artifact, PYTHONPATH=str(shim))
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "jsonschema" in body["reason"]
    assert "AMMO_VALIDATE_FAIL_OPEN=1" in body["reason"]


def test_post_hook_jsonschema_unavailable_escape_hatch(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_stage_enum_schema(repo)
    shim = repo / "python_shim"
    shim.mkdir()
    (shim / "jsonschema.py").write_text("raise ImportError('simulated unavailable jsonschema')\n", encoding="utf-8")
    env = _env(
        repo,
        artifact,
        PYTHONPATH=str(shim),
        AMMO_VALIDATE_FAIL_OPEN="1",
    )
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_next_step_first_fire_then_socratic_edge(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_stage_enum_schema(repo)
    env = _env(repo, artifact, AMMO_REMINDER_STATE_DIR=str(repo / "reminders"))
    payload = {
        "cwd": str(repo),
        "session_id": "next-step-session",
        "tool_input": {"file_path": str(artifact / "state.json")},
    }

    first = _run_hook("post_tool_use_guard.py", payload, env)
    assert first.returncode == 0
    first_body = json.loads(first.stdout)
    first_context = first_body["hookSpecificOutput"]["additionalContext"]
    assert "AMMO NEXT STEP:" in first_context
    assert "REASON THROUGH THIS" not in first_context

    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0]["baseline"] = {
        "started_at": "2026-07-10T00:00:00Z",
        "completed_at": "2026-07-10T00:01:00Z",
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    second = _run_hook("post_tool_use_guard.py", payload, env)
    assert second.returncode == 0
    second_body = json.loads(second.stdout)
    second_context = second_body["hookSpecificOutput"]["additionalContext"]
    assert "REASON THROUGH THIS" in second_context
    assert "mine bottlenecks from this baseline" in second_context


def test_post_hook_next_step_is_silent_for_child_agent(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _write_stage_enum_schema(repo)
    env = _env(repo, artifact, AMMO_REMINDER_STATE_DIR=str(repo / "reminders"), CODEX_SUBAGENT="1")
    payload = {
        "cwd": str(repo),
        "session_id": "child-next-step-session",
        "tool_input": {"file_path": str(artifact / "state.json")},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    if result.stdout.strip():
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "AMMO NEXT STEP:" not in context


def test_post_hook_does_not_validate_read_only_state_inspection(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    (repo / ".codex" / "schemas" / "state.schema.json").unlink()
    state_path = artifact / "state.json"
    state_path.write_text("{invalid json", encoding="utf-8")
    env = _env(repo, artifact)

    payloads = [
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": f"cat {state_path}"}},
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": f"jq . {state_path}"}},
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": f"rg current_stage {state_path}"}},
        {"cwd": str(repo), "tool_name": "Read", "tool_input": {"file_path": str(state_path)}},
    ]

    for hook_payload in payloads:
        result = _run_hook("post_tool_use_guard.py", hook_payload, env)
        assert result.returncode == 0
        assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_validates_reconcile_write_explicit_non_active_artifact(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    other_doc = json.loads((active / "state.json").read_text(encoding="utf-8"))
    other_doc["campaign"]["current_stage"] = "invalid_stage"
    other = repo / "kernel_opt_artifacts" / "other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "state.json").write_text(json.dumps(other_doc), encoding="utf-8")
    env = _env(repo, active)
    command = (
        "python .codex/skills/ammo/scripts/reconcile_track_state.py "
        f"--artifact-dir {other} --track-id op001 --write"
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "Bash",
            "tool_input": {"command": command},
            "tool_response": {"status": "success"},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "invalid_stage" in body["reason"] or "is not one of" in body["reason"]


def test_post_hook_validates_apply_patch_state_mutation(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "invalid_stage"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)
    patch_text = (
        f"*** Begin Patch\n*** Update File: {state_path}\n"
        "@@\n- old\n+ new\n*** End Patch"
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "apply_patch",
            "tool_input": {"patch": patch_text},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "invalid_stage" in body["reason"] or "is not one of" in body["reason"]


def test_post_hook_validates_absolute_non_active_state_write(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    other = repo / "kernel_opt_artifacts" / "other"
    other.mkdir(parents=True, exist_ok=True)
    other_state = other / "state.json"
    other_state.write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "invalid_stage",
                    "rounds": [{"round_id": 1}],
                }
            }
        ),
        encoding="utf-8",
    )
    env = _env(repo, active)
    command = f"python -c \"from pathlib import Path; Path('{other_state}').write_text('{{}}')\""

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "invalid_stage" in body["reason"] or "is not one of" in body["reason"]


def test_post_hook_concrete_valid_target_ignores_unrelated_invalid_active_state(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    active_state = active / "state.json"
    active_doc = json.loads(active_state.read_text(encoding="utf-8"))
    valid_other_doc = json.loads(json.dumps(active_doc))
    active_doc["campaign"]["current_stage"] = "invalid_stage"
    active_state.write_text(json.dumps(active_doc), encoding="utf-8")
    other = repo / "kernel_opt_artifacts" / "other"
    other.mkdir(parents=True, exist_ok=True)
    other_state = other / "state.json"
    other_state.write_text(json.dumps(valid_other_doc), encoding="utf-8")
    env = _env(repo, active, AMMO_REMINDER_STATE_DIR=str(repo / "reminders"))
    command = f"python -c \"from pathlib import Path; Path('{other_state}').write_text('{{}}')\""

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_expands_variable_target_for_non_active_state_write(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    other = repo / "kernel_opt_artifacts" / "other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "invalid_stage",
                    "rounds": [{"round_id": 1}],
                }
            }
        ),
        encoding="utf-8",
    )
    env = _env(repo, active, OTHER_ARTIFACT=str(other))
    command = 'STATE="$OTHER_ARTIFACT/state.json"; python -c "import os; os.replace(\"tmp\", os.environ[\"STATE\"])"'

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "invalid_stage" in body["reason"] or "is not one of" in body["reason"]


def test_post_hook_fail_closed_on_unexpected_engine_output(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    for index, output in enumerate(("{}", "[]", "not-json")):
        shim = repo / f"engine_{index}.py"
        shim.write_text(f"print({output!r})\n", encoding="utf-8")
        env = _env(repo, artifact, AMMO_STATE_ENGINE=str(shim))
        result = _run_hook("post_tool_use_guard.py", payload, env)
        assert result.returncode == 0
        body = json.loads(result.stdout)
        assert body["decision"] == "block"
        assert "validation could not run" in body["reason"]


def test_post_hook_unexpected_engine_output_honors_failopen(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    shim = repo / "engine.py"
    shim.write_text("print('{}')\n", encoding="utf-8")
    env = _env(
        repo,
        artifact,
        AMMO_STATE_ENGINE=str(shim),
        AMMO_VALIDATE_FAIL_OPEN="1",
    )
    payload = {"cwd": str(repo), "tool_input": {"file_path": str(artifact / "state.json")}}

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_mcp_read_file_does_not_validate_state(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.write_text("{invalid json", encoding="utf-8")
    (repo / ".codex" / "schemas" / "state.schema.json").unlink()
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "mcp__filesystem__read_file",
            "tool_input": {"file_path": str(state_path)},
        },
        env,
    )

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_mcp_write_file_validates_state(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "invalid_stage"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "mcp__filesystem__write_file",
            "tool_input": {"file_path": str(state_path), "content": "{}"},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"


def test_post_hook_mcp_move_destination_validates_state(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["campaign"]["current_stage"] = "invalid_stage"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "mcp__filesystem__move_file",
            "tool_input": {
                "source": str(repo / "tmp-state.json"),
                "destination_path": str(state_path),
            },
        },
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_mcp_move_away_blocks_missing_state(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    backup = artifact / "state.backup.json"
    state_path.replace(backup)
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "mcp__filesystem__move_file",
            "tool_input": {"source": str(state_path), "destination": str(backup)},
        },
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_validates_new_target_first_party_writer(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    generated = repo / "kernel_opt_artifacts" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "state.json").write_text("{}", encoding="utf-8")
    env = _env(repo, active)
    command = (
        'python ".codex/skills/ammo/scripts/new_target.py" '
        f"--artifact-dir {generated} --min-e2e-improvement -1"
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_validates_quoted_ammo_state_mutator(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    env = _env(repo, artifact)
    command = (
        'python ".codex/skills/ammo/scripts/ammo_state.py"   set '
        f'--state "{state_path}" --field campaign.status --value ' + "'\"active\"'"
    )

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_validates_python_open_write(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.write_text("{", encoding="utf-8")
    env = _env(repo, artifact)
    command = f'''python -c "open('{state_path}', 'w').write('{{')"'''

    result = _run_hook(
        "post_tool_use_guard.py",
        {"cwd": str(repo), "tool_name": "Bash", "tool_input": {"command": command}},
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_validates_dot_relative_non_active_state_target(tmp_path):
    repo = _hook_repo(tmp_path)
    active = _active_stage1_artifact(repo)
    other = repo / "kernel_opt_artifacts" / "other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "state.json").write_text("{", encoding="utf-8")
    env = _env(repo, active)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(other),
            "tool_name": "Bash",
            "tool_input": {"command": "printf '{' > ./state.json"},
        },
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_validates_partial_write_even_when_command_exits_error(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.write_text("{", encoding="utf-8")
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(artifact),
            "tool_name": "Bash",
            "tool_input": {"command": "printf '{' > state.json; false"},
            "tool_response": {"status": "error"},
        },
        env,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"


def test_post_hook_does_not_treat_write_content_state_path_as_target(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.write_text("{", encoding="utf-8")
    notes_path = repo / "notes.md"
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "Write",
            "tool_input": {
                "file_path": str(notes_path),
                "content": f"Mention only: {state_path}",
            },
        },
        env,
    )

    assert result.returncode == 0
    assert '"decision":"block"' not in result.stdout.replace(" ", "")


def test_post_hook_schema_lookup_stops_at_nested_git_file_boundary(tmp_path):
    repo = _hook_repo(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: /tmp/nonexistent", encoding="utf-8")
    artifact = nested / "kernel_opt_artifacts" / "campaign"
    artifact.mkdir(parents=True, exist_ok=True)
    state_path = artifact / "state.json"
    state_path.write_text(json.dumps({"campaign": {"current_stage": "1_baseline"}}), encoding="utf-8")
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(nested),
            "tool_name": "Write",
            "tool_input": {"file_path": str(state_path), "content": "{}"},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "schema" in body["reason"].lower()


def test_post_hook_blocks_bash_state_delete(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.unlink()
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(artifact),
            "tool_name": "Bash",
            "tool_input": {"command": "rm state.json"},
            "tool_response": {"status": "success"},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "not a regular file" in body["reason"]


def test_post_hook_blocks_bash_state_move_away(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    backup = artifact / "state.backup.json"
    state_path.replace(backup)
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(artifact),
            "tool_name": "Bash",
            "tool_input": {"command": "mv state.json state.backup.json"},
            "tool_response": {"status": "success"},
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "not a regular file" in body["reason"]


def test_post_hook_blocks_apply_patch_state_delete(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state_path = artifact / "state.json"
    state_path.unlink()
    env = _env(repo, artifact)

    result = _run_hook(
        "post_tool_use_guard.py",
        {
            "cwd": str(repo),
            "tool_name": "apply_patch",
            "tool_input": {
                "patch": f"*** Begin Patch\n*** Delete File: {state_path}\n*** End Patch"
            },
        },
        env,
    )

    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["decision"] == "block"
    assert "not a regular file" in body["reason"]


def _audit_dispatch_message(artifact: Path, stage: str = "stage_2", round_id: int = 1, cycle: int = 2) -> str:
    """The canonical dispatch block from orchestration/audit-protocol.md."""
    return (
        "    task: audit_gate\n"
        f"    artifact_dir: {artifact}\n"
        f"    stage: {stage}       # stage_1 | stage_2 | stage_45 | stage_67\n"
        f"    round: {round_id}\n"
        f"    cycle: {cycle}\n"
    )


def _gate(artifact: Path, stage: str, round_id: int = 1) -> dict:
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    audit = state["campaign"]["rounds"][round_id - 1].get("audit") or {}
    return audit.get(stage) or {}


def test_spawn_agent_post_hook_stamps_audit_started_for_auditor(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c2",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" in result.stdout
    gate = _gate(artifact, "stage_2")
    assert gate["cycle"] == 2
    assert gate["started_at"].endswith("Z")
    assert "passed_at" not in gate
    assert _gate(artifact, "stage_1") == {}


def test_spawn_agent_post_hook_stamps_audit_started_for_stage45_recycle(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "functions.spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_45_1_c3",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, stage="stage_45", cycle=3),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert _gate(artifact, "stage_45")["cycle"] == 3


def test_spawn_agent_post_hook_does_not_stamp_for_non_auditor_role(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "name": "champion-1",
            "agent_type": "ammo-champion",
            "message": _audit_dispatch_message(artifact),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_does_not_stamp_malformed_dispatch(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c1",
            "agent_type": "ammo-auditor",
            "message": (
                "    task: audit_gate\n"
                f"    artifact_dir: {artifact}\n"
                "    stage: stage_2\n"
                "    cycle: notanumber\n"
            ),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_does_not_stamp_unknown_stage(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_6_1_c1",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, stage="stage_6"),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_survives_failing_audit_started_helper(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    broken = repo / "broken_engine.py"
    broken.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    env = _env(repo, artifact, AMMO_STATE_ENGINE=str(broken))
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c1",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, cycle=1),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_skips_audit_stamp_inside_child_agent(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "agent_type": "ammo-auditor",
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c1",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, cycle=1),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_subagent_start_does_not_stamp_audit_started(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact, CODEX_PROJECT_DIR=str(repo))
    payload = {
        "cwd": str(repo),
        "hook_event_name": "SubagentStart",
        "tool_input": {
            "agent_type": "ammo-auditor",
            "task_name": "audit_stage_2_1_c1",
            "message": _audit_dispatch_message(artifact, cycle=1),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_does_not_stamp_failed_spawn_call(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c1",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, cycle=1),
        },
        "tool_response": {"error": "permission denied"},
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_does_not_claim_a_stamp_on_a_legacy_round(tmp_path):
    # `audit-started` exits 0 on its fail-open no-ops, so the exit code alone
    # cannot prove a write. Claiming "stamped" here sends the lead to record
    # passed_at, which the 4.2 provenance backstop then rejects.
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
    state["campaign"]["rounds"][0].pop("audit", None)
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_1_c2",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_does_not_claim_a_stamp_for_an_absent_round(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    env = _env(repo, artifact)
    payload = {
        "cwd": str(repo),
        "tool_name": "spawn_agent",
        "tool_input": {
            "task_name": "audit_stage_2_2_c1",
            "agent_type": "ammo-auditor",
            "message": _audit_dispatch_message(artifact, round_id=2, cycle=1),
        },
    }

    result = _run_hook("post_tool_use_guard.py", payload, env)

    assert result.returncode == 0
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_spawn_agent_post_hook_audit_stamp_is_idempotent_per_cycle(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact)

    def spawn(cycle: int) -> None:
        result = _run_hook(
            "post_tool_use_guard.py",
            {
                "cwd": str(repo),
                "tool_name": "spawn_agent",
                "tool_input": {
                    "task_name": f"audit_stage_2_1_c{cycle}",
                    "agent_type": "ammo-auditor",
                    "message": _audit_dispatch_message(artifact, cycle=cycle),
                },
            },
            env,
        )
        assert result.returncode == 0
        assert "AMMO AUDIT STARTED" in result.stdout

    spawn(1)
    assert _gate(artifact, "stage_2")["cycle"] == 1
    spawn(2)
    gate = _gate(artifact, "stage_2")
    assert gate["cycle"] == 2
    assert gate["started_at"].endswith("Z")
