# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Campaign identity and continuation regressions for both compaction paths."""

import json
import shutil

import pytest

from test_hook_semantics import (
    ROOT, _active_stage1_artifact, _audit_dispatch_message, _env, _gate,
    _hook_repo, _run_hook, _write_audit_invariants,
)


def _state(artifact, **campaign_changes):
    path = artifact / "state.json"
    document = json.loads(path.read_text())
    document["campaign"].update(campaign_changes)
    path.write_text(json.dumps(document))
    return document


def _checkpoint(artifact, **fields):
    path = artifact / "compaction_checkpoint.json"
    path.write_text(json.dumps(fields))
    return path


@pytest.mark.parametrize("source", ["compact", "resume"])
def test_explicit_campaign_without_checkpoint_never_uses_other_campaign(tmp_path, source):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    other = artifact.with_name("other")
    shutil.copytree(artifact, other)
    checkpoint = _checkpoint(other, campaign_stage="4_5_parallel_tracks")
    before = checkpoint.read_bytes()
    result = _run_hook("session_start.py", {"cwd": str(repo), "source": source}, _env(repo, artifact))
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert str(artifact / "state.json") in context
    assert str(other) not in context
    assert "Stage: 1_baseline" in context
    assert checkpoint.read_bytes() == before
    assert (ROOT / ".codex/skills/ammo/SKILL.md").read_text() in context


@pytest.mark.parametrize("live_status,old_status", [
    ("active", "paused"), ("paused", "active"), ("campaign_complete", "active"),
])
def test_live_status_owns_resume_and_checkpoint_is_retained(tmp_path, live_status, old_status):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    _state(artifact, status=live_status)
    checkpoint = _checkpoint(artifact, campaign_status=old_status, campaign_stage="old_stage", campaign_round=9)
    before = checkpoint.read_bytes()
    for _ in range(2):
        result = _run_hook("session_start.py", {"cwd": str(repo), "source": "compact"}, _env(repo, artifact))
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "Stage: 1_baseline" in context
        assert f"Campaign: {live_status}" in context
        assert "old_stage" not in context
        assert checkpoint.read_bytes() == before
        if live_status == "active":
            assert "AMMO Root Context Restored" in context
        elif live_status == "paused":
            assert "until the user explicitly resumes" in context
            assert "Current AMMO entry skill" not in context
        else:
            assert "Do not restart completed stages" in context


@pytest.mark.parametrize("trigger", ["manual", "auto"])
def test_only_root_publishes_campaign_checkpoint(tmp_path, trigger):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact, CODEX_SESSION_ID="root-thread")
    payload = {"cwd": str(repo), "session_id": "root-thread", "hook_event_name": "PreCompact", "trigger": trigger}
    result = _run_hook("pre_compact.py", payload, env)
    assert json.loads(result.stdout)["continue"] is True
    checkpoint = artifact / "compaction_checkpoint.json"
    before = checkpoint.read_bytes()
    result = _run_hook("pre_compact.py", {**payload, "agent_id": "child-thread", "agent_type": "ammo-implementer"}, env)
    assert json.loads(result.stdout)["continue"] is True
    assert checkpoint.read_bytes() == before
    child_start = _run_hook("session_start.py", {**payload, "source": "compact", "agent_id": "child-thread"}, env)
    assert child_start.stdout.strip() == ""


def test_unique_trusted_thread_recovers_without_explicit_binding(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    document = _state(artifact)
    document["codex_thread_id"] = "root-thread"
    (artifact / "state.json").write_text(json.dumps(document))
    other = artifact.with_name("other")
    shutil.copytree(artifact, other)
    document["codex_thread_id"] = "different-root"
    (other / "state.json").write_text(json.dumps(document))
    env = _env(repo, artifact, CODEX_SESSION_ID="root-thread")
    env.pop("AMMO_ARTIFACT_DIR")
    result = _run_hook("pre_compact.py", {"cwd": str(repo), "hook_event_name": "PreCompact"}, env)
    assert json.loads(result.stdout)["continue"] is True
    assert (artifact / "compaction_checkpoint.json").is_file()
    assert not (other / "compaction_checkpoint.json").exists()


def test_ambiguous_trusted_campaigns_block_compaction_without_writing(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    other = artifact.with_name("other")
    shutil.copytree(artifact, other)
    env = _env(repo, artifact)
    env.pop("AMMO_ARTIFACT_DIR")
    result = _run_hook("pre_compact.py", {"cwd": str(repo), "hook_event_name": "PreCompact"}, env)
    assert json.loads(result.stdout)["continue"] is False
    assert "multiple campaigns" in result.stdout
    assert not list(repo.glob("**/compaction_checkpoint.json"))


def test_missing_explicit_state_does_not_fall_back(tmp_path):
    repo = _hook_repo(tmp_path)
    other = _active_stage1_artifact(repo)
    missing = other.with_name("missing")
    env = _env(repo, missing)
    result = _run_hook("pre_compact.py", {"cwd": str(repo), "hook_event_name": "PreCompact"}, env)
    assert json.loads(result.stdout)["continue"] is False
    assert not (other / "compaction_checkpoint.json").exists()


def test_unbound_start_does_not_resurrect_an_unrelated_campaign(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    env = _env(repo, artifact, AMMO_SESSION_ID="another-server", CODEX_SESSION_ID="another-root")
    env.pop("AMMO_ARTIFACT_DIR")
    result = _run_hook("session_start.py", {"cwd": str(repo), "source": "startup"}, env)
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("script,event", [
    ("pre_tool_use_guard.py", "PreToolUse"),
    ("post_tool_use_guard.py", "PostToolUse"),
    ("stop_gate_guard.py", "Stop"),
])
def test_conflicting_explicit_identity_fails_closed_in_guards(tmp_path, script, event):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    result = _run_hook(script, {
        "cwd": str(repo), "hook_event_name": event, "tool_name": "Bash",
        "tool_input": {"command": "touch candidate.txt"},
    }, _env(repo, artifact, AMMO_SESSION_ID="different-server"))
    assert json.loads(result.stdout)["decision"] == "block"


@pytest.mark.parametrize("tool_name", ["agentsspawn_agent", "agents.spawn_agent"])
def test_native_v2_root_audit_provenance_is_idempotent(tmp_path, tool_name):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    payload = {
        "cwd": str(repo), "hook_event_name": "PostToolUse", "tool_name": tool_name,
        "tool_input": {"task_name": "audit_stage_2_r1_c2", "agent_type": "ammo-auditor", "message": _audit_dispatch_message(artifact)},
        "tool_response": {"agent_id": "auditor-thread"},
    }
    result = _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert "AMMO AUDIT STARTED" in result.stdout
    first = _gate(artifact, "stage_2")
    _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert _gate(artifact, "stage_2") == first
    assert "passed_at" not in first


@pytest.mark.parametrize("extra,response", [
    ({}, {"error": "spawn rejected"}),
    ({"agent_id": "child", "agent_type": "ammo-investigator"}, {"agent_id": "auditor"}),
])
def test_native_v2_failed_or_child_spawn_does_not_stamp_audit(tmp_path, extra, response):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    before = (artifact / "state.json").read_bytes()
    result = _run_hook("post_tool_use_guard.py", {
        "cwd": str(repo), "hook_event_name": "PostToolUse", "tool_name": "agentsspawn_agent",
        "tool_input": {"task_name": "audit_stage_2_r1_c2", "agent_type": "ammo-auditor", "message": _audit_dispatch_message(artifact)},
        "tool_response": response, **extra,
    }, _env(repo, artifact))
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


def test_assigned_recovery_note_has_no_layout_warning_or_gate_effect(tmp_path):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    note = artifact / "recovery/root-thread/root/auditor/delegate/context.md"
    note.parent.mkdir(parents=True)
    note.write_text("Assignment and evidence pointers only.\n")
    before = (artifact / "state.json").read_bytes()
    result = _run_hook("post_tool_use_guard.py", {
        "cwd": str(repo), "tool_name": "Write", "agent_id": "delegate-thread",
        "tool_input": {"file_path": str(note)},
    }, _env(repo, artifact))
    assert "LAYOUT WARN" not in result.stdout
    assert "LAYOUT WARNING" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


@pytest.mark.parametrize("stage", ["stage_1", "stage_2", "stage_45", "stage_67"])
def test_opaque_native_audit_message_uses_exact_public_route(tmp_path, stage):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    payload = {
        "cwd": str(repo), "hook_event_name": "PostToolUse", "tool_name": "agentsspawn_agent",
        "tool_input": {"task_name": f"audit_{stage}_r1_c2_attempt1", "agent_type": "ammo-auditor",
                       "message": "opaque-native-message"},
        "tool_response": {"task_name": f"/root/audit_{stage}_r1_c2_attempt1"},
    }
    result = _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert "AMMO AUDIT STARTED" in result.stdout
    first = _gate(artifact, stage)
    assert first["cycle"] == 2 and first["started_at"]
    assert "passed_at" not in first
    _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert _gate(artifact, stage) == first


@pytest.mark.parametrize("name,message,extra,response", [
    ("audit_stage_1_r1_c2", None, {"agent_id": "child", "agent_type": "ammo-investigator"}, {}),
    ("audit_stage_1_r1_c2", None, {}, {"error": "spawn failed"}),
    ("audit_stage_1_r0_c2", None, {}, {}),
    ("audit_stage_1_r1_c0", None, {}, {}),
    ("audit_stage_99_r1_c2", None, {}, {}),
    ("audit_stage_1_1_c2", None, {}, {}),
    ("audit_stage_1_r1_c2", "task: audit_gate\nstage: stage_1\ncycle: malformed", {}, {}),
    ("audit_stage_1_r1_c2", "conflicting_valid_dispatch", {}, {}),
])
def test_opaque_audit_route_cannot_override_scope_or_failed_spawn(tmp_path, name, message, extra, response):
    repo = _hook_repo(tmp_path)
    artifact = _active_stage1_artifact(repo)
    if message == "conflicting_valid_dispatch":
        message = _audit_dispatch_message(artifact, stage="stage_2")
    before = (artifact / "state.json").read_bytes()
    result = _run_hook("post_tool_use_guard.py", {
        "cwd": str(repo), "hook_event_name": "PostToolUse", "tool_name": "agentsspawn_agent",
        "tool_input": {"task_name": name, "agent_type": "ammo-auditor", "message": message or "opaque-native-message"},
        "tool_response": response, **extra,
    }, _env(repo, artifact))
    assert "AMMO AUDIT STARTED" not in result.stdout
    assert (artifact / "state.json").read_bytes() == before


@pytest.mark.parametrize("content", [
    "# Audit Verdict\n\n## Definition of Done/Expectations\nExpected baseline evidence.\n",
    "# Audit Verdict\n\n## Phase 1 - Independent Reconstruction\n",
    "**Phase 1 preliminary**: BLOCKED\n\n## Phase 1 - Independent Reconstruction\n### Findings\n\n## Summary\nHALTED\n",
])
def test_partial_phase1_never_activates_phase2(tmp_path, content):
    repo = _hook_repo(tmp_path)
    _write_audit_invariants(repo)
    artifact = _active_stage1_artifact(repo)
    verdict = artifact / "rounds/1/audits/stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text(content)
    result = _run_hook("post_tool_use_guard.py", {
        "cwd": str(repo), "tool_name": "apply_patch", "agent_type": "ammo-auditor", "agent_id": "fresh-auditor",
        "tool_input": {"patch": f"*** Begin Patch\n*** Add File: {verdict}\n+{content}\n*** End Patch"},
    }, _env(repo, artifact))
    assert "Phase 1 written successfully" not in result.stdout
    assert not list((repo / "audit_phase2").rglob("*.json"))


@pytest.mark.parametrize("tool", ["apply_patch", "Bash"])
def test_note_reference_to_earlier_verdict_does_not_activate_fresh_auditor(tmp_path, tool):
    repo = _hook_repo(tmp_path)
    _write_audit_invariants(repo)
    artifact = _active_stage1_artifact(repo)
    verdict = artifact / "rounds/1/audits/stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("**Phase 1 preliminary**: BLOCKED\n\n## Phase 1 - Independent Reconstruction\nMissing baseline.\n\n## Summary\nHALTED\n")
    note = artifact / "recovery/root-thread/root/audit_stage_1_r1_c2/context.md"
    note.parent.mkdir(parents=True)
    note.write_text(f"Prior verdict pointer: {verdict.relative_to(repo)}\n")
    inputs = {
        "apply_patch": {"patch": f"*** Begin Patch\n*** Add File: {note}\n+Prior verdict pointer: {verdict.relative_to(repo)}\n*** End Patch"},
        "Bash": {"command": f"printf '%s\\n' 'Later: cat > {verdict}' > '{note}'"},
    }
    result = _run_hook("post_tool_use_guard.py", {
        "cwd": str(repo), "tool_name": tool, "agent_type": "ammo-auditor", "agent_id": "fresh-auditor",
        "tool_input": inputs[tool],
    }, _env(repo, artifact))
    assert "Phase 1 written successfully" not in result.stdout
    assert not list((repo / "audit_phase2").rglob("*.json"))


@pytest.mark.parametrize("tool", ["apply_patch", "Write"])
def test_completed_phase1_write_activates_exact_auditor_once(tmp_path, tool):
    repo = _hook_repo(tmp_path)
    _write_audit_invariants(repo)
    artifact = _active_stage1_artifact(repo)
    verdict = artifact / "rounds/1/audits/stage_1.md"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text("**Phase 1 preliminary**: BLOCKED\n\n## Phase 1 - Independent Reconstruction\nMissing baseline.\n\n## Summary\nHALTED\n")
    inputs = {
        "apply_patch": {"patch": f"*** Begin Patch\n*** Update File: {verdict}\n@@\n+Phase 1 complete.\n*** End Patch"},
        "Write": {"file_path": str(verdict)},
    }
    payload = {"cwd": str(repo), "tool_name": tool, "agent_type": "ammo-auditor",
               "agent_id": "fresh-auditor", "tool_input": inputs[tool]}
    result = _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert "Phase 1 written successfully" in result.stdout
    sentinels = list((repo / "audit_phase2").rglob("*.json"))
    assert len(sentinels) == 1
    record = json.loads(sentinels[0].read_text())
    assert record["agent_id"] == "fresh-auditor" and record["verdict_file"] == str(verdict)
    second = _run_hook("post_tool_use_guard.py", payload, _env(repo, artifact))
    assert "Phase 1 written successfully" not in second.stdout
