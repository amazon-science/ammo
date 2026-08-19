#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Reconcile one AMMO track's structured evidence back into state.json."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


TERMINAL_VERDICTS = {"PASS", "GATED_PASS", "FAIL"}
BLOCKER_VERDICTS = {"GPU_BLOCKED"}
TRACK_VERDICTS = TERMINAL_VERDICTS | BLOCKER_VERDICTS
PASS_FAIL = {"PASS", "FAIL"}
SCHEMA_VERDICTS = {"PASS", "GATING_REQUIRED", "GATED_PASS", "FAIL"}
TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ReconciliationResult(NamedTuple):
    changed: bool
    changes: list[str]
    verdict: str | None
    state_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


@contextmanager
def _state_lock(path: Path):
    """Exclusive advisory lock on `<state.json>.lock`, same inode ammo_state.py
    state_lock() uses.

    temp.replace() makes one write atomic but does not stop a lost update. The
    auditor-spawn hook stamps audit.{stage}.started_at while the lead reconciles
    tracks, and a lost stamp wedges the campaign on the schema-4.2 provenance
    backstop. The lock covers the whole read-mutate-write, so every writer of
    state.json serializes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path.parent / (path.name + ".lock")), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def _status(value: Any) -> str:
    return str(value or "").upper()


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _validate_track_id(track_id: str) -> None:
    if not isinstance(track_id, str) or not TRACK_ID_PATTERN.fullmatch(track_id):
        raise ValueError(f"Invalid track id: {track_id!r}")


def _relative_track_path(
    track_id: str,
    filename: str,
    round_id: int | None,
) -> str:
    if round_id is None:
        return f"tracks/{track_id}/{filename}"
    return f"rounds/{round_id}/tracks/{track_id}/{filename}"


def _kill_criteria_results(evidence: dict[str, Any]) -> dict[str, str]:
    criteria = evidence.get("kill_criteria")
    if not isinstance(criteria, dict):
        return {}

    results: dict[str, str] = {}
    for name, result in sorted(criteria.items()):
        if not isinstance(result, dict):
            continue
        status = _status(result.get("status"))
        if status in PASS_FAIL:
            results[str(name)] = status
    return results


def _has_nonempty_kill_results(evidence: dict[str, Any]) -> bool:
    criteria = evidence.get("kill_criteria")
    return isinstance(criteria, dict) and bool(_kill_criteria_results(evidence))


def _derive_track_verdict(evidence: dict[str, Any]) -> str | None:
    for key in ("verdict", "overall_verdict", "track_verdict"):
        explicit = _status(evidence.get(key))
        if explicit in TRACK_VERDICTS:
            return explicit

    component_statuses: list[str] = []
    for section_name in ("correctness", "kernel_bench", "e2e"):
        section = evidence.get(section_name)
        if isinstance(section, dict):
            status = _status(section.get("status"))
            if status:
                component_statuses.append(status)

    e2e = evidence.get("e2e")
    if isinstance(e2e, dict):
        for nested_name in ("admissibility", "fastpath_proof"):
            nested = e2e.get(nested_name)
            if isinstance(nested, dict):
                status = _status(nested.get("status"))
                if status:
                    component_statuses.append(status)

    kill_results = _kill_criteria_results(evidence)
    if "FAIL" in component_statuses or "FAIL" in kill_results.values():
        return "FAIL"

    required_sections = {"correctness", "kernel_bench", "e2e"}
    if required_sections.issubset(evidence) and component_statuses:
        if not _has_nonempty_kill_results(evidence):
            return None
        if all(value == "PASS" for value in component_statuses) and all(
            value == "PASS" for value in kill_results.values()
        ):
            return "PASS"
    return None


def _schema_verdict(status: str) -> str | None:
    return status if status in SCHEMA_VERDICTS else None


def _round_container(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
    """Return the active track container without creating missing V2 fields."""
    if "campaign" not in state:
        tracks = state.get("parallel_tracks")
        if not isinstance(tracks, dict):
            raise ValueError("state.json parallel_tracks is not an object")
        return tracks, None, None

    campaign = state.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("Could not locate the current-round track container")
    current_round = campaign.get("current_round")
    rounds = campaign.get("rounds")
    if (
        isinstance(current_round, bool)
        or not isinstance(current_round, int)
        or current_round < 1
        or not isinstance(rounds, list)
    ):
        raise ValueError("Could not locate the current-round track container")

    round_state = next(
        (
            item
            for item in rounds
            if isinstance(item, dict) and item.get("round_id") == current_round
        ),
        None,
    )
    if not isinstance(round_state, dict):
        raise ValueError("Could not locate the current-round track container")
    parallel = round_state.get("parallel_tracks")
    tracks = parallel.get("tracks") if isinstance(parallel, dict) else None
    if not isinstance(tracks, dict):
        raise ValueError("Could not locate the current-round track container")
    return tracks, round_state, current_round


def _set_if_changed(
    target: dict[str, Any],
    key: str,
    value: Any,
    changes: list[str],
) -> None:
    current = target.get(key)
    if key not in target or current != value:
        changes.append(f"{key}: {_fmt(current)} -> {_fmt(value)}")
        target[key] = value


def _require_status(
    value: Any,
    label: str,
    *,
    expected: str = "PASS",
) -> None:
    if _status(value) != expected:
        raise ValueError(
            "Generated diluted PASS cannot waive binding validation gates: "
            f"{label} is not {expected}"
        )


def _validate_diluted_gates(
    evidence: dict[str, Any],
    track: dict[str, Any],
) -> None:
    correctness = evidence.get("correctness")
    _require_status(
        correctness.get("status") if isinstance(correctness, dict) else None,
        "correctness",
    )
    _require_status(track.get("gate_5_1a"), "Gate 5.1a")
    if track.get("correctness") is not True:
        raise ValueError(
            "Generated diluted PASS cannot waive binding validation gates: "
            "Gate 5.1b correctness is not true"
        )

    kernel = evidence.get("kernel_bench")
    _require_status(
        kernel.get("status") if isinstance(kernel, dict) else None,
        "kernel performance",
    )
    _require_status(track.get("gate_5_2"), "Gate 5.2")

    e2e = evidence.get("e2e")
    admissibility = e2e.get("admissibility") if isinstance(e2e, dict) else None
    _require_status(
        admissibility.get("status") if isinstance(admissibility, dict) else None,
        "admissibility",
    )
    fastpath = e2e.get("fastpath_proof") if isinstance(e2e, dict) else None
    _require_status(
        fastpath.get("status") if isinstance(fastpath, dict) else None,
        "fast-path activation proof",
    )
    hits = fastpath.get("hits") if isinstance(fastpath, dict) else None
    if isinstance(hits, bool) or not isinstance(hits, (int, float)) or hits <= 0:
        raise ValueError(
            "Generated diluted PASS cannot waive binding validation gates: "
            "fast-path activation proof has no hits"
        )

    kill_results = _kill_criteria_results(evidence)
    if not kill_results or any(value != "PASS" for value in kill_results.values()):
        raise ValueError(
            "Generated diluted PASS cannot waive binding validation gates: "
            "kill criteria are missing or failed"
        )


def _generated_diluted_decision(
    summary_path: Path,
    evidence: dict[str, Any],
    track: dict[str, Any],
) -> bool | None:
    """Return the generated report's diluted decision, if a report exists."""
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    gate = summary.get("e2e_gate")
    if not isinstance(gate, dict):
        raise ValueError("validation_summary.json e2e_gate must be an object")
    if "diluted_pass" not in gate:
        return False

    diluted = gate.get("diluted_pass")
    if (
        not isinstance(diluted, dict)
        or diluted.get("eligible") is not True
        or _status(gate.get("track_verdict")) != "PASS"
    ):
        raise ValueError(
            "validation_summary.json diluted_pass requires eligible=true "
            "and track_verdict=PASS"
        )
    _validate_diluted_gates(evidence, track)
    return True


def _update_diluted_rollup(
    round_state: dict[str, Any] | None,
    track_id: str,
    diluted: bool,
    changes: list[str],
) -> None:
    if round_state is None:
        return
    rollup = round_state.get("diluted_tracks")
    if rollup is None:
        rollup = []
    if not isinstance(rollup, list):
        raise ValueError("round diluted_tracks must be an array")

    matches = [
        item for item in rollup if isinstance(item, dict) and item.get("op_id") == track_id
    ]
    if diluted:
        if matches:
            updated = rollup
        else:
            updated = [
                *rollup,
                {
                    "op_id": track_id,
                    "tpot_improvement_pct": None,
                    "decode_share_of_e2e": None,
                },
            ]
    else:
        updated = [
            item
            for item in rollup
            if not (isinstance(item, dict) and item.get("op_id") == track_id)
        ]
    if round_state.get("diluted_tracks") != updated:
        changes.append(f"diluted_tracks[{track_id}]: reconciled")
        round_state["diluted_tracks"] = updated


def reconcile_track_state(
    artifact_dir: Path | str,
    track_id: str,
    write: bool = False,
) -> ReconciliationResult:
    _validate_track_id(track_id)
    artifact_path = Path(artifact_dir).expanduser().resolve()
    state_path = artifact_path / "state.json"
    with _state_lock(state_path):
        state = _read_json(state_path)
        tracks, round_state, round_id = _round_container(state)

        evidence_path = artifact_path / _relative_track_path(
            track_id,
            "evidence.json",
            round_id,
        )
        validation_path = artifact_path / _relative_track_path(
            track_id,
            "validation_results.md",
            round_id,
        )
        summary_path = artifact_path / _relative_track_path(
            track_id,
            "validation_summary.json",
            round_id,
        )
        evidence = _read_json(evidence_path)

        existing = tracks.get(track_id)
        if existing is None:
            existing = {}
            tracks[track_id] = existing
        if not isinstance(existing, dict):
            raise TypeError(f"Track {track_id!r} in state.json is not an object")

        diluted_field = evidence.get("diluted")
        if "diluted" in evidence and not isinstance(diluted_field, bool):
            raise ValueError("evidence diluted field must be boolean")

        generated_diluted = _generated_diluted_decision(
            summary_path,
            evidence,
            existing,
        )
        diluted = generated_diluted if generated_diluted is not None else diluted_field
        verdict = _derive_track_verdict(evidence)
        if generated_diluted is True:
            verdict = "PASS"
        if verdict is None:
            raise ValueError(f"Could not derive a track verdict from {evidence_path}")
        if diluted is True and verdict != "PASS":
            raise ValueError("diluted=true requires verdict PASS")

        changes: list[str] = []
        _set_if_changed(existing, "status", verdict, changes)
        _set_if_changed(existing, "verdict", _schema_verdict(verdict), changes)
        _set_if_changed(
            existing,
            "evidence_path",
            _relative_track_path(track_id, "evidence.json", round_id),
            changes,
        )
        if validation_path.exists():
            _set_if_changed(
                existing,
                "validation_results_path",
                _relative_track_path(track_id, "validation_results.md", round_id),
                changes,
            )
        _set_if_changed(
            existing,
            "kill_criteria_results",
            _kill_criteria_results(evidence),
            changes,
        )
        kernel_bench = evidence.get("kernel_bench")
        if isinstance(kernel_bench, dict) and kernel_bench.get("boundary_ab") is not None:
            boundary = kernel_bench.get("boundary_ab")
            if not isinstance(boundary, dict):
                raise ValueError("kernel_bench.boundary_ab must be an object")
            _set_if_changed(existing, "gate_5_2_boundary", boundary, changes)
        if diluted is not None:
            _set_if_changed(existing, "diluted", diluted, changes)
            _update_diluted_rollup(round_state, track_id, diluted, changes)

        if verdict in TERMINAL_VERDICTS and not existing.get("completed_at"):
            completed_at = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            _set_if_changed(existing, "completed_at", completed_at, changes)
        elif verdict in BLOCKER_VERDICTS and "completed_at" in existing:
            changes.append(f"completed_at: {_fmt(existing.get('completed_at'))} -> <removed>")
            existing.pop("completed_at")

        if changes and write:
            _write_json(state_path, state)
        return ReconciliationResult(bool(changes), changes, verdict, state_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--track-id", "--track", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    try:
        result = reconcile_track_state(
            args.artifact_dir,
            args.track_id,
            write=args.write,
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result.changed:
        if args.write:
            print(f"Reconciled {result.state_path} for {args.track_id}: {result.verdict}")
            for change in result.changes:
                print(f"- {change}")
            return 0
        print("state.json is not reconciled with track evidence:", file=sys.stderr)
        for change in result.changes:
            print(f"- {change}", file=sys.stderr)
        return 1 if args.check else 0

    print(f"state.json already reconciled for {args.track_id}: {result.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
