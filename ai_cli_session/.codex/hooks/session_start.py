#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Restore root campaign context from the exact current state."""

from pathlib import Path

from common import (
    CampaignResolutionError, active_artifact_dir, additional_context,
    find_repo_root, is_child_agent_payload, load_json, read_stdin_json,
    record_trusted_session_identity,
)

TERMINAL_STATUSES = {"campaign_complete", "campaign_exhausted", "complete", "finished"}


def main() -> None:
    payload = read_stdin_json()
    # Codex does not dispatch child SessionStart on compaction. If a caller
    # supplies one, never inject the lead contract into that child.
    if is_child_agent_payload(payload):
        return
    repo = find_repo_root(payload.get("cwd"))
    try:
        record_trusted_session_identity(payload)
        artifact = active_artifact_dir(repo, payload)
    except (CampaignResolutionError, RuntimeError) as exc:
        additional_context(
            "SessionStart",
            f"AMMO recovery needs an exact campaign binding: {exc}. "
            "Inspect the state and resolve the binding before campaign work. "
            "Do not select a campaign by modification time.",
        )
        return
    if artifact is None:
        return

    state_file = artifact / "state.json"
    state = load_json(state_file) or {}
    campaign = state.get("campaign") or {}
    status = campaign.get("status", "unknown")
    source = payload.get("source", "startup")
    position = (
        f"State: `{state_file}`. Round: {campaign.get('current_round', 'unknown')} | "
        f"Stage: {campaign.get('current_stage', 'unknown')} | Campaign: {status}."
    )
    if status == "paused":
        additional_context(
            "SessionStart",
            "# AMMO Campaign Paused\n\n" + position + "\n"
            "Read for context only. Do not spawn agents, reserve GPUs, mutate campaign "
            "artifacts/state, or advance a stage until the user explicitly resumes this campaign.",
        )
        return
    if status in TERMINAL_STATUSES:
        additional_context(
            "SessionStart",
            "# AMMO Campaign Terminal\n\n" + position + "\n"
            "Do not restart completed stages. Check only outstanding terminal report and "
            "fact-check requirements in `.codex/skills/ammo/SKILL.md`; the Stop gate "
            "owns the final report acceptance check.",
        )
        return
    if source not in {"compact", "resume"}:
        additional_context(
            "SessionStart",
            "# AMMO Optimization Detected\n\n"
            f"Existing optimization state at: `{state_file}`.\n"
            "If the user continues this campaign, read `.codex/skills/ammo/SKILL.md` "
            "and its current state. This startup notice does not resume it.",
        )
        return

    # Load the current installed router, including from the root-owned managed
    # closure. State paths and reference pointers remain campaign-local.
    skill_path = Path(__file__).resolve().parents[1] / "skills" / "ammo" / "SKILL.md"
    try:
        skill = skill_path.read_text(encoding="utf-8")
    except OSError:
        skill = "Read `.codex/skills/ammo/SKILL.md` before continuing."
    current_round = campaign.get("current_round")
    rounds = campaign.get("rounds")
    round_doc = {}
    if isinstance(current_round, int) and isinstance(rounds, list) and 0 < current_round <= len(rounds):
        round_doc = rounds[current_round - 1]
    if not isinstance(round_doc, dict):
        round_doc = {}
    parallel = round_doc.get("parallel_tracks") or {}
    tracks = parallel.get("tracks") if isinstance(parallel, dict) else {}
    if not isinstance(tracks, dict):
        tracks = {}
    active = [
        f"{op_id} [{track.get('status')}]"
        for op_id, track in tracks.items()
        if isinstance(track, dict) and track.get("status") in {"IN_PROGRESS", "GATING_REQUIRED", "GPU_BLOCKED"}
    ]
    text = (
        "# AMMO Root Context Restored\n\n" + position + "\n"
        "Live state owns current status, stage, round, and tracks. A retained compaction "
        "checkpoint is advisory and must not override these facts. Read the exact state "
        "before resuming; reconcile existing agents and evidence before any redispatch.\n"
        + ("Active tracks: " + ", ".join(active) + ".\n" if active else "")
        + "\n## Current AMMO entry skill\n\n" + skill
    )
    additional_context("SessionStart", text)


if __name__ == "__main__":
    main()
