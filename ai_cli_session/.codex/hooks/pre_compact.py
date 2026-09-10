#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Persist the active AMMO control-plane position before Codex compaction."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common import (
    CampaignResolutionError, active_artifact_dir, emit, find_repo_root,
    is_child_agent_payload, load_json, read_stdin_json,
)


TERMINAL_CAMPAIGN_STATUSES = {"campaign_complete", "campaign_exhausted"}
KNOWN_CAMPAIGN_STATUSES = {"active", "paused"} | TERMINAL_CAMPAIGN_STATUSES
ACTIVE_TRACK_STATUSES = {"IN_PROGRESS", "GATING_REQUIRED", "GPU_BLOCKED"}


class CheckpointError(RuntimeError):
    """An active AMMO campaign could not be snapshotted safely."""


def _discover_active_state(
    repo: Path, payload: dict[str, Any]
) -> tuple[Optional[Path], Optional[dict[str, Any]], list[str]]:
    try:
        artifact = active_artifact_dir(repo, payload)
    except CampaignResolutionError as exc:
        raise CheckpointError(str(exc)) from exc
    if artifact is None:
        if os.environ.get("AMMO_ARTIFACT_DIR"):
            raise CheckpointError("AMMO_ARTIFACT_DIR has no safe campaign state")
        return None, None, []
    state_path = artifact / "state.json"
    state = load_json(state_path)
    campaign = state.get("campaign") if isinstance(state, dict) else None
    if not isinstance(campaign, dict) or campaign.get("status") not in KNOWN_CAMPAIGN_STATUSES:
        raise CheckpointError(f"active AMMO state has invalid campaign.status: {state_path}")
    if campaign["status"] in TERMINAL_CAMPAIGN_STATUSES:
        return None, None, []
    return state_path, state, []


def _active_tracks(round_doc: dict[str, Any]) -> list[dict[str, Any]]:
    parallel = round_doc.get("parallel_tracks")
    tracks = parallel.get("tracks") if isinstance(parallel, dict) else None
    if not isinstance(tracks, dict):
        return []

    active: list[dict[str, Any]] = []
    for op_id, track in sorted(tracks.items(), key=lambda item: str(item[0])):
        if not isinstance(track, dict):
            continue
        status = str(track.get("status") or "")
        if status not in ACTIVE_TRACK_STATUSES:
            continue
        identity: dict[str, Any] = {"op_id": str(op_id), "status": status}
        for key in ("worktree_path", "worktree_branch", "evidence_path"):
            value = track.get(key)
            if isinstance(value, str) and value:
                identity[key] = value
        active.append(identity)
    return active


def _checkpoint_document(
    state_path: Path, state: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    campaign = state["campaign"]
    current_round = campaign.get("current_round")
    rounds = campaign.get("rounds")
    if not isinstance(current_round, int) or current_round < 1:
        raise CheckpointError("active AMMO state has an invalid campaign.current_round")
    if not isinstance(rounds, list) or current_round > len(rounds):
        raise CheckpointError("active AMMO state does not contain its current round")
    round_doc = rounds[current_round - 1]
    if not isinstance(round_doc, dict):
        raise CheckpointError("active AMMO current-round record is malformed")

    stage = campaign.get("current_stage")
    campaign_status = campaign.get("status")
    round_status = round_doc.get("status")
    if not isinstance(stage, str) or not stage:
        raise CheckpointError("active AMMO state has no canonical campaign.current_stage")
    if not isinstance(campaign_status, str) or not campaign_status:
        raise CheckpointError("active AMMO state has no canonical campaign.status")
    if not isinstance(round_status, str) or not round_status:
        raise CheckpointError("active AMMO state has no canonical current-round status")

    target = state.get("target") if isinstance(state.get("target"), dict) else {}
    model = payload.get("model") or target.get("model_id") or "unknown"
    return {
        "checkpoint_type": "pre_compaction",
        "checkpoint_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger": payload.get("trigger") or "unknown",
        "model": str(model),
        "source_session_id": str(payload.get("session_id") or ""),
        "source_turn_id": str(payload.get("turn_id") or ""),
        "state_file": str(state_path),
        "campaign_round": current_round,
        "campaign_stage": stage,
        "campaign_status": campaign_status,
        "round_status": round_status,
        "active_tracks": _active_tracks(round_doc),
    }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink():
        raise CheckpointError(f"refusing to replace symlink checkpoint: {path}")
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(temp_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _allow(system_message: Optional[str] = None) -> None:
    output: dict[str, Any] = {
        "continue": True,
        "suppressOutput": system_message is None,
    }
    if system_message:
        output["systemMessage"] = system_message
    emit(output)


def _stop(reason: str) -> None:
    # Codex PreCompact accepts universal continue:false; the
    # decision:block shape used by tool hooks is invalid for this event.
    emit(
        {
            "continue": False,
            "stopReason": reason,
            "suppressOutput": False,
            "systemMessage": reason,
        }
    )


def main() -> None:
    payload = read_stdin_json()
    if payload.get("hook_event_name") != "PreCompact" or is_child_agent_payload(payload):
        _allow()
        return

    repo = find_repo_root(payload.get("cwd"))
    try:
        state_path, state, warnings = _discover_active_state(repo, payload)
        if state_path is None or state is None:
            _allow("; ".join(warnings) if warnings else None)
            return
        checkpoint = _checkpoint_document(state_path, state, payload)
        _atomic_write_json(state_path.parent / "compaction_checkpoint.json", checkpoint)
    except (CheckpointError, OSError, TypeError, ValueError) as exc:
        _stop(f"AMMO PreCompact checkpoint failed: {exc}")
        return
    _allow("; ".join(warnings) if warnings else None)


if __name__ == "__main__":
    main()
