#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    active_artifact_dir,
    block_stop,
    find_repo_root,
    lifecycle_task_name,
    load_json,
    read_stdin_json,
)

PAUSED_STATUSES = {"paused"}
FINAL_CAMPAIGN_STATUSES = {
    "campaign_complete",
    "campaign_exhausted",
    "complete",
    "finished",
}


def _is_subagent_payload(payload: dict[str, Any]) -> bool:
    if os.environ.get("CODEX_SUBAGENT") == "1":
        return True
    tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    for key in ("agent_type", "agentType", "subagent_type", "subagentType"):
        if payload.get(key) or tool_input.get(key):
            return True
    name = str(
        payload.get("agent_name")
        or payload.get("agentName")
        or tool_input.get("agent_name")
        or tool_input.get("agentName")
        or ""
    ).strip()
    if name and name not in {"team-lead", "root"}:
        return True
    return bool(lifecycle_task_name(payload))


def _campaign(state: Any) -> dict[str, Any]:
    campaign = state.get("campaign") if isinstance(state, dict) else None
    return campaign if isinstance(campaign, dict) else {}


def _server_session_id(state: Any) -> str:
    value = os.environ.get("AMMO_SESSION_ID")
    if value:
        return value
    if isinstance(state, dict):
        value = state.get("session_id") or state.get("sessionId")
        if value:
            return str(value)
    return "unknown"


def _cleanup_session_gpus(session_id: str) -> None:
    if not session_id or session_id == "unknown":
        return
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "ammo"
        / "scripts"
        / "gpu_reservation.py"
    )
    if not script.is_file():
        return
    try:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "release-session",
                "--session-id",
                session_id,
                "--include-children",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        pass


def _stop_nudge(artifact_dir: Path, state: Any) -> str | None:
    campaign = _campaign(state)
    status = str(campaign.get("status") or "active")
    if status in PAUSED_STATUSES:
        return None
    if status in FINAL_CAMPAIGN_STATUSES:
        report_path = artifact_dir / "REPORT.md"
        if report_path.is_file():
            fact_path = artifact_dir / "report_assets" / "report_fact_check.json"
            fact = load_json(fact_path)
            report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if (
                isinstance(fact, dict)
                and fact.get("ok") is True
                and fact.get("report_sha256") == report_sha256
            ):
                return None
            return (
                "Report fact-check review artifact missing, stale, or not accepted. "
                "Run a fresh adversarial fact-check and write "
                f"{fact_path} with ok=true and report_sha256={report_sha256} "
                "before stopping."
            )
        return (
            f"Campaign is {status} but REPORT.md has not been generated. "
            "Use spawn_agent with a fresh report agent, have it read "
            ".codex/skills/ammo/report/SKILL.md, and generate "
            f"{artifact_dir / 'REPORT.md'}."
        )

    stage = str(campaign.get("current_stage") or "unknown")
    round_id = campaign.get("current_round")
    rounds = campaign.get("rounds")
    current_round: dict[str, Any] = {}
    if isinstance(round_id, int) and round_id >= 1 and isinstance(rounds, list) \
            and len(rounds) >= round_id and isinstance(rounds[round_id - 1], dict):
        current_round = rounds[round_id - 1]
    round_root = artifact_dir / "rounds" / str(round_id or 1)

    if stage == "1_baseline":
        constraints = round_root / "constraints.md"
        if not constraints.is_file():
            return (
                "AMMO Stage 1 evidence is incomplete: missing "
                f"{constraints}. Capture the target constraints before stopping."
            )
        return None

    if stage == "2_bottleneck_mining":
        mining = current_round.get("bottleneck_mining") or {}
        analysis = round_root / "mining" / "bottleneck_analysis.md"
        required = (
            "top_component", "top_f_e2e_pct", "top_addressable_e2e_pct",
            "amdahl_ceiling",
            "decode_frac", "component_breakdown",
        )
        missing = [name for name in required if mining.get(name) in (None, "", [])]
        if not analysis.is_file():
            missing.insert(0, str(analysis.relative_to(artifact_dir)))
        audit = current_round.get("audit") or {}
        stage2 = audit.get("stage_2") if isinstance(audit, dict) else None
        if not isinstance(stage2, dict) or not stage2.get("passed_at"):
            missing.append("audit.stage_2.passed_at")
        if missing:
            return (
                "AMMO Stage 2 evidence/audit is incomplete: "
                + ", ".join(missing[:8])
            )
        return None

    if stage in {"4_5_parallel_tracks", "6_integration"}:
        report = load_json(round_root / "validation_gate_report.json")
        report_status = (
            str(report.get("overall_status") or "").upper()
            if isinstance(report, dict) else ""
        )
        if report_status not in {"PASS", "EVIDENCE_COMPLETE_NO_PASS"}:
            return (
                "AMMO validation gate is missing or incomplete. Run the current-round "
                f"validation gate and write {round_root / 'validation_gate_report.json'}."
            )
        if stage == "4_5_parallel_tracks":
            return (
                "AMMO Stage 4-5 evidence is reconciled, but the active campaign is "
                "not complete. Run T_AUDIT_S45 and advance to Stage 6 only after "
                "the cohort and audit gates pass."
            )
        return (
            "AMMO Stage 6 is still active. Complete the integration decision, "
            "promotion or exhaustion bookkeeping, T_AUDIT_S67, and Stage 7 "
            "campaign evaluation before stopping."
        )

    if stage.startswith("7_campaign_eval"):
        return (
            "You are at Stage 7 (Campaign Evaluation). Do not stop or ask the user. "
            "Execute the mechanical threshold check: compare "
            "bottleneck_mining.top_addressable_e2e_pct with "
            "campaign.config.min_e2e_improvement_pct. If the addressable impact meets "
            "the threshold, advance and continue. Otherwise set campaign_complete after "
            "a SHIP integration status, or campaign_exhausted after an EXHAUSTED/failed "
            "integration, then immediately proceed to Stage 7b report generation."
        )
    if stage.startswith("7b_report") or "report_gen" in stage:
        return (
            "You are at Stage 7b (Report Generation). Use spawn_agent with a fresh "
            "report agent, have it read .codex/skills/ammo/report/SKILL.md, and "
            f"generate {artifact_dir / 'REPORT.md'} before stopping."
        )
    return f"AMMO campaign is still active at {stage}; complete or advance the stage before stopping."


def main() -> None:
    # No one-shot breaker: an active gate blocks every Stop; the escape hatch is AMMO_ALLOW_STOP=1.
    payload = read_stdin_json()
    if _is_subagent_payload(payload):
        raise SystemExit(0)

    repo = find_repo_root(payload.get("cwd"))
    artifact_dir = active_artifact_dir(repo, payload)
    if artifact_dir is None:
        raise SystemExit(0)
    state = load_json(artifact_dir / "state.json")

    server_session_id = _server_session_id(state)
    if isinstance(state, dict):
        state_server = str(state.get("session_id") or state.get("sessionId") or "")
        trusted_server = str(os.environ.get("AMMO_SESSION_ID") or "")
        if state_server and trusted_server and state_server != trusted_server:
            block_stop("AMMO state.session_id does not match trusted AMMO_SESSION_ID.")
            raise SystemExit(0)
        state_thread = str(state.get("codex_thread_id") or "")
        payload_thread = str(payload.get("session_id") or payload.get("sessionId") or "")
        if state_thread and payload_thread and state_thread != payload_thread:
            block_stop("AMMO state.codex_thread_id does not match the trusted Stop hook thread.")
            raise SystemExit(0)

    campaign_status = str(_campaign(state).get("status") or "")

    # Terminal report integrity is never one-shot: a repeated Stop cannot bypass
    # a missing, rejected, or stale fact-check artifact.
    if campaign_status in FINAL_CAMPAIGN_STATUSES:
        reason = _stop_nudge(artifact_dir, state)
        if reason:
            block_stop(reason)
            raise SystemExit(0)

    if os.environ.get("AMMO_ALLOW_STOP") == "1":
        _cleanup_session_gpus(server_session_id)
        raise SystemExit(0)

    reason = _stop_nudge(artifact_dir, state)
    if reason:
        block_stop(reason)
        raise SystemExit(0)

    _cleanup_session_gpus(server_session_id)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
