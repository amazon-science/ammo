#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import os
from pathlib import Path
from typing import Optional

from common import additional_context,find_repo_root,load_json,read_stdin_json,record_trusted_session_identity

p=read_stdin_json(); record_trusted_session_identity(p); repo=find_repo_root(p.get('cwd'))


def _first_file(root: Path, name: str) -> Optional[Path]:
    try:
        candidates = [
            path for path in root.glob(f"**/{name}")
            if path.is_file() and not path.is_symlink()
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    except (StopIteration, IndexError):
        return None
    except Exception:
        return None


def _state_value(state: dict, dotted: str, default: str = "unknown") -> str:
    current = state
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    return str(current) if current not in (None, "") else default


def _explicit_artifact_dir() -> Optional[Path]:
    value = os.environ.get("AMMO_ARTIFACT_DIR")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def _checkpoint_state_file(checkpoint: Path, checkpoint_doc: dict) -> Path:
    """Resolve the checkpoint's canonical state path with relocation fallback."""
    fallback = checkpoint.parent / "state.json"
    declared = checkpoint_doc.get("state_file")
    candidate = Path(declared).expanduser() if isinstance(declared, str) and declared else fallback
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        root = checkpoint.parent.resolve()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if resolved.parent == checkpoint.parent.resolve() and not candidate.is_symlink():
            return resolved
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        pass
    return fallback


def _checkpoint_active_tracks(checkpoint_doc: dict) -> list[dict]:
    tracks = checkpoint_doc.get("active_tracks")
    if not isinstance(tracks, list):
        return []
    return [track for track in tracks if isinstance(track, dict) and track.get("op_id")]


artifact_root = repo / "kernel_opt_artifacts"
explicit_artifact = _explicit_artifact_dir()
explicit_checkpoint = explicit_artifact / "compaction_checkpoint.json" if explicit_artifact else None
checkpoint = (
    explicit_checkpoint
    if explicit_checkpoint and explicit_checkpoint.is_file() and not explicit_checkpoint.is_symlink()
    else _first_file(artifact_root, "compaction_checkpoint.json")
)

if checkpoint:
    checkpoint_doc = load_json(checkpoint) or {}
    state_file = _checkpoint_state_file(checkpoint, checkpoint_doc)
    state = load_json(state_file) or {}
    campaign = state.get("campaign") if isinstance(state, dict) else {}
    current_round = campaign.get("current_round") if isinstance(campaign, dict) else None
    round_summary = ""
    if isinstance(current_round, int) and isinstance(campaign.get("rounds"), list):
        rounds = campaign["rounds"]
        if 0 < current_round <= len(rounds) and isinstance(rounds[current_round - 1], dict):
            round_summary = str(rounds[current_round - 1].get("round_summary") or "")

    checkpoint_round = checkpoint_doc.get("campaign_round") or current_round or "unknown"
    checkpoint_stage = (
        checkpoint_doc.get("campaign_stage")
        or checkpoint_doc.get("stage")  # legacy checkpoint compatibility
        or _state_value(state, "campaign.current_stage")
    )
    checkpoint_status = (
        checkpoint_doc.get("campaign_status")
        or checkpoint_doc.get("status")  # legacy checkpoint compatibility
        or _state_value(state, "campaign.status")
    )
    round_status = checkpoint_doc.get("round_status") or "unknown"
    active_tracks = _checkpoint_active_tracks(checkpoint_doc)
    extra = (
        f"\n4. **Checkpoint position**: Round {checkpoint_round} | "
        f"Stage: {checkpoint_stage} | Campaign: {checkpoint_status} | Round: {round_status}."
    )
    if active_tracks:
        track_summary = ", ".join(
            f"{track.get('op_id')} [{track.get('status') or 'unknown'}]"
            for track in active_tracks
        )
        extra += (
            f"\n5. **Active tracks ({len(active_tracks)})**: {track_summary}. "
            "Reload their canonical state entries before deciding whether to respawn or follow up."
        )
    else:
        legacy_track_count = checkpoint_doc.get("track_count") or 0
        if isinstance(legacy_track_count, int) and legacy_track_count > 0:
            extra += (
                f"\n5. **Parallel tracks ({legacy_track_count} active)**: Reload "
                "campaign.rounds[current_round-1].parallel_tracks.tracks from state.json."
            )

    try:
        checkpoint.unlink()
    except Exception:
        pass

    if checkpoint_status == "paused" or _state_value(
        state, "campaign.status"
    ) == "paused":
        additional_context(
            "SessionStart",
            "# AMMO Campaign Paused\n\n"
            f"Load the preserved state at `{state_file}` for context only. "
            "Do not spawn agents, reserve GPUs, mutate campaign artifacts/state, "
            "or advance a stage until the user explicitly resumes this campaign.",
        )
        raise SystemExit(0)

    additional_context(
        "SessionStart",
        "# Session Resumed After Compaction\n\n"
        "## You Are The AMMO Lead Orchestrator\n\n"
        "This session was compacted while orchestrating an AMMO optimization.\n\n"
        "### Immediate Actions\n\n"
        "1. **Read the skill**: `.codex/skills/ammo/SKILL.md`\n"
        f"2. **Load state**: `cat {state_file}`\n"
        f"3. **Resume current stage** - spawn subagents as needed{extra}\n\n"
        f"### Model: {checkpoint_doc.get('model') or _state_value(state, 'target.model_id')} | Stage: {checkpoint_stage} | Status: {checkpoint_status}\n"
        f"### Round summary: {round_summary}\n\n"
        "You are the LEAD - scaffold, delegate, gate. Do not implement directly.",
    )
    raise SystemExit(0)

explicit_state = explicit_artifact / "state.json" if explicit_artifact else None
state_file = (
    explicit_state
    if explicit_state and explicit_state.is_file() and not explicit_state.is_symlink()
    else _first_file(artifact_root, "state.json")
)
if state_file:
    state = load_json(state_file) or {}
    if _state_value(state, "campaign.status") == "paused":
        additional_context(
            "SessionStart",
            "# AMMO Campaign Paused\n\n"
            f"Existing paused state at: `{state_file}`. Read-only inspection is "
            "allowed, but do not spawn agents, reserve GPUs, mutate campaign "
            "artifacts/state, or advance a stage until the user explicitly resumes.",
        )
        raise SystemExit(0)
    additional_context(
        "SessionStart",
        "# AMMO Optimization Detected\n\n"
        f"Existing optimization state at: `{state_file}`\n"
        f"Model: {_state_value(state, 'target.model_id')} | Stage: {_state_value(state, 'campaign.current_stage')}\n\n"
        "If continuing this optimization:\n"
        "- Read the skill: `.codex/skills/ammo/SKILL.md`\n"
        "- You are the lead orchestrator - scaffold, delegate, gate. Do not implement directly.",
    )
