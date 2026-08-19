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

from common import emit, find_repo_root, load_json, read_stdin_json


TERMINAL_CAMPAIGN_STATUSES = {"campaign_complete", "campaign_exhausted"}
KNOWN_CAMPAIGN_STATUSES = {"active", "paused"} | TERMINAL_CAMPAIGN_STATUSES
ACTIVE_TRACK_STATUSES = {"IN_PROGRESS", "GATING_REQUIRED", "GPU_BLOCKED"}


class CheckpointError(RuntimeError):
    """An active AMMO campaign could not be snapshotted safely."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_state_path(candidate: Path, artifact_root: Path) -> Optional[Path]:
    """Resolve a real, non-symlink state file contained by artifact_root."""
    try:
        lexical_root = Path(os.path.abspath(artifact_root))
        lexical_candidate = Path(os.path.abspath(candidate))
        lexical_relative = lexical_candidate.relative_to(lexical_root)
        cursor = lexical_root
        for part in lexical_relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        root = artifact_root.resolve()
        if not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _candidate_state_paths(repo: Path) -> list[tuple[Path, bool, Path]]:
    """Return explicit-env first, then newest campaign-local state candidates."""
    candidates: list[tuple[Path, bool, Path]] = []
    env_path = os.environ.get("AMMO_ARTIFACT_DIR")
    if env_path:
        raw = Path(env_path).expanduser()
        artifact_dir = raw if raw.is_absolute() else repo / raw
        candidates.append((artifact_dir / "state.json", True, artifact_dir))

    artifact_root = repo / "kernel_opt_artifacts"
    try:
        discovered = sorted(
            artifact_root.glob("**/state.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        discovered = []
    candidates.extend((path, False, artifact_root) for path in discovered)

    result: list[tuple[Path, bool, Path]] = []
    seen: set[Path] = set()
    for raw_path, explicit, safety_root in candidates:
        safe_path = _safe_state_path(raw_path, safety_root)
        identity = safe_path or raw_path.absolute()
        if identity in seen:
            continue
        seen.add(identity)
        result.append((safe_path or raw_path, explicit, safety_root))
    return result


def _state_session_matches(state: dict[str, Any], payload: dict[str, Any]) -> bool:
    state_session = str(state.get("session_id") or state.get("sessionId") or "")
    if not state_session:
        return False
    server_session = str(os.environ.get("AMMO_SESSION_ID") or "")
    if server_session:
        return state_session == server_session
    # Legacy states predate the split server-session/root-thread identity. Only
    # use the hook payload as a fallback when no native codex_thread_id exists.
    if state.get("codex_thread_id"):
        return False
    legacy_thread = str(
        os.environ.get("CODEX_SESSION_ID") or payload.get("session_id") or ""
    )
    return bool(legacy_thread and state_session == legacy_thread)


def _discover_active_state(
    repo: Path, payload: dict[str, Any]
) -> tuple[Optional[Path], Optional[dict[str, Any]], list[str]]:
    valid: list[tuple[int, float, Path, dict[str, Any]]] = []
    warnings: list[str] = []
    for candidate, explicit, safety_root in _candidate_state_paths(repo):
        safe_path = _safe_state_path(candidate, safety_root)
        if safe_path is None:
            if explicit:
                raise CheckpointError(
                    f"AMMO_ARTIFACT_DIR does not resolve to a safe campaign state: {candidate}"
                )
            warnings.append(f"ignored unsafe AMMO state path: {candidate}")
            continue
        state = load_json(safe_path)
        if not isinstance(state, dict) or not isinstance(state.get("campaign"), dict):
            if explicit:
                raise CheckpointError(f"active AMMO state is unreadable or malformed: {safe_path}")
            warnings.append(f"ignored malformed AMMO state: {safe_path}")
            continue
        campaign = state["campaign"]
        status = campaign.get("status")
        if status not in KNOWN_CAMPAIGN_STATUSES:
            if explicit:
                raise CheckpointError(
                    f"active AMMO state has invalid campaign.status={status!r}: {safe_path}"
                )
            warnings.append(f"ignored AMMO state with invalid campaign.status: {safe_path}")
            continue
        if status in TERMINAL_CAMPAIGN_STATUSES:
            if explicit:
                return None, None, warnings
            continue
        score = 2 if explicit else (1 if _state_session_matches(state, payload) else 0)
        try:
            mtime = safe_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        valid.append((score, mtime, safe_path, state))

    if not valid:
        return None, None, warnings
    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, state_path, state = valid[0]
    return state_path, state, warnings


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
    # Codex 0.144.1 PreCompact accepts universal continue:false; the
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
    if payload.get("hook_event_name") != "PreCompact":
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
