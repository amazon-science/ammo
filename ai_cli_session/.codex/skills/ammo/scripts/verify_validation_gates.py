#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Verify AMMO Stage 5 validation gates from structured evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PLACEHOLDER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [r"\bTODO\b", r"<FILL_ME>", r"\bTBD\b", r"placeholder"]
]
TERMINAL_TRACK_STATUSES = {"PASS", "GATED_PASS", "FAIL"}
TRACK_VERDICT_STATUSES = TERMINAL_TRACK_STATUSES | {"GPU_BLOCKED"}
PASS_FAIL = {"PASS", "FAIL"}


@dataclass
class GateResult:
    name: str
    status: str
    message: str
    evidence: List[str] = field(default_factory=list)
    track_id: Optional[str] = None


@dataclass
class TrackContext:
    track_id: str
    metadata: Dict[str, Any]
    validation_path: Path
    validation_text: str
    evidence_path: Path
    evidence: Optional[Dict[str, Any]]


@dataclass
class VerificationReport:
    artifact_dir: str
    phase: str = "5_validation"
    overall_status: str = "UNKNOWN"
    advance_to_stage6: bool = False
    track_outcomes: Dict[str, str] = field(default_factory=dict)
    gates: List[GateResult] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "phase": self.phase,
            "overall_status": self.overall_status,
            "advance_to_stage6": self.advance_to_stage6,
            "track_outcomes": self.track_outcomes,
            "gates": [asdict(gate) for gate in self.gates],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(artifact_dir: Path) -> Dict[str, Any]:
    state_path = artifact_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json: {state_path}")
    return _read_json(state_path)


def _normalize_path(path_value: str, artifact_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (artifact_dir / candidate).resolve()


def _validate_track_id(track_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", track_id):
        raise ValueError(f"Invalid track id: {track_id!r}")


def _validate_current_track_path(
    path_value: str,
    artifact_dir: Path,
    current_round: int,
    track_id: str,
) -> Path:
    """Resolve a V2 evidence path without allowing round/track escape."""
    raw = Path(path_value)
    parts = raw.parts
    for index, part in enumerate(parts[:-1]):
        if part == "rounds" and index + 1 < len(parts):
            if parts[index + 1] != str(current_round):
                raise ValueError(
                    f"Track {track_id!r} evidence points to a prior round: "
                    f"{path_value}"
                )

    resolved = _normalize_path(path_value, artifact_dir)
    expected = (
        artifact_dir
        / "rounds"
        / str(current_round)
        / "tracks"
        / track_id
    ).resolve()
    if not resolved.is_relative_to(expected):
        raise ValueError(
            f"Track {track_id!r} evidence must stay in its current-round "
            f"track directory: {expected}"
        )
    return resolved


def _current_round(state: Dict[str, Any]) -> Optional[int]:
    campaign = state.get("campaign")
    if not isinstance(campaign, dict):
        return None
    current_round = campaign.get("current_round")
    if isinstance(current_round, int) and not isinstance(current_round, bool) and current_round > 0:
        return current_round
    return None


def _default_track_path(state: Dict[str, Any], track_id: str, filename: str) -> str:
    current_round = _current_round(state)
    if current_round is not None:
        return f"rounds/{current_round}/tracks/{track_id}/{filename}"
    return f"tracks/{track_id}/{filename}"


def _stage1_baseline_files(artifact_dir: Path) -> tuple[List[Path], str]:
    try:
        state = _load_state(artifact_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    current_round = _current_round(state)
    if current_round is not None:
        pattern = f"rounds/{current_round}/sweeps/baseline/json/baseline_bs*.json"
        return sorted((artifact_dir / "rounds" / str(current_round) / "sweeps" / "baseline" / "json").glob("baseline_bs*.json")), pattern
    pattern = "runs/baseline_bs*.json"
    return sorted((artifact_dir / "runs").glob("baseline_bs*.json")), pattern


def _branch_matches_worktree(metadata: Dict[str, Any]) -> Optional[str]:
    worktree_path = metadata.get("worktree_path")
    expected_branch = metadata.get("branch")
    if not worktree_path or not expected_branch:
        return None
    path = Path(worktree_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        return None
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output if output == expected_branch else f"expected {expected_branch}, found {output}"


def _state_tracks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return track metadata from round-centric state, with flat v1 fallback."""
    campaign = state.get("campaign")
    if isinstance(campaign, dict):
        current_round = campaign.get("current_round")
        rounds = campaign.get("rounds")
        if (
            isinstance(current_round, int)
            and not isinstance(current_round, bool)
            and current_round > 0
            and isinstance(rounds, list)
            and len(rounds) >= current_round
            and isinstance(rounds[current_round - 1], dict)
        ):
            tracks = rounds[current_round - 1].get("parallel_tracks", {}).get("tracks", {})
            return {k: v for k, v in tracks.items() if isinstance(v, dict)} if isinstance(tracks, dict) else {}
        return {}

    tracks = state.get("parallel_tracks") or {}
    return tracks if isinstance(tracks, dict) else {}


def _selected_track_ids(state: Dict[str, Any]) -> Optional[set[str]]:
    campaign = state.get("campaign")
    current_round = _current_round(state)
    if not isinstance(campaign, dict) or current_round is None:
        return None
    try:
        major, minor = (int(part) for part in str(
            campaign.get("schema_version", "4.0")
        ).split(".", 1))
    except (TypeError, ValueError):
        major, minor = 4, 0
    if major < 4 or (major == 4 and minor < 2):
        return None
    rounds = campaign.get("rounds") or []
    if len(rounds) < current_round or not isinstance(rounds[current_round - 1], dict):
        return set()
    debate = rounds[current_round - 1].get("debate") or {}
    winners = {
        value for value in (debate.get("selected_winners") or [])
        if isinstance(value, str) and value
    }
    candidates = {
        value.get("op_id") for value in (debate.get("selected_candidates") or [])
        if isinstance(value, dict)
        and isinstance(value.get("op_id"), str)
        and value.get("op_id")
    }
    if winners != candidates:
        raise ValueError(
            "debate.selected_winners and selected_candidates op_ids differ"
        )
    if candidates and not 2 <= len(candidates) <= 3:
        raise ValueError("current-round cohort must contain 2-3 selected candidates")
    return candidates


def _verify_monitor_evidence(
    artifact_dir: Path,
    current_round: int,
    op_id: str,
    metadata: Dict[str, Any],
) -> None:
    expected_agent = metadata.get("implementer_agent")
    expected_rollout = metadata.get("implementer_rollout_id")
    expected_monitor = metadata.get("monitor_agent")
    paths = {}
    for field in (
        "monitor_evidence_path", "monitor_offsets_path", "monitor_summary_path"
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{op_id} missing {field}")
        path = _validate_current_track_path(value, artifact_dir, current_round, op_id)
        if not path.is_file():
            raise ValueError(f"{op_id} monitor evidence missing: {path}")
        paths[field] = path

    if not paths["monitor_evidence_path"].read_text(encoding="utf-8").strip():
        raise ValueError(f"{op_id} monitor observation log is empty")

    offsets = _read_json(paths["monitor_offsets_path"])
    summary = _read_json(paths["monitor_summary_path"])
    if not isinstance(offsets, dict) or not isinstance(summary, dict):
        raise ValueError(f"{op_id} monitor offsets/summary must be JSON objects")
    for document, label in ((offsets, "offsets"), (summary, "summary")):
        if document.get("target_agent") != expected_agent:
            raise ValueError(f"{op_id} monitor {label} target_agent mismatch")
        if document.get("target_rollout_id") != expected_rollout:
            raise ValueError(f"{op_id} monitor {label} rollout mismatch")
    if summary.get("monitor_agent") != expected_monitor:
        raise ValueError(f"{op_id} monitor summary identity mismatch")
    if summary.get("coverage_status") not in {
        "TRACK_COMPLETE", "INFEASIBLE", "LEAD_SHUTDOWN", "HANDOFF_COMPLETE"
    }:
        raise ValueError(f"{op_id} monitor summary is not terminal")
    if not isinstance(offsets.get("last_offset"), int) or offsets["last_offset"] < 0:
        raise ValueError(f"{op_id} monitor offsets lack a valid last_offset")


def _resolve_tracks(artifact_dir: Path, state: Dict[str, Any], track_id: Optional[str]) -> List[TrackContext]:
    tracks = _state_tracks(state)
    resolved: List[TrackContext] = []

    selected = _selected_track_ids(state)
    if selected is not None and selected and not track_id and set(tracks) != selected:
        raise ValueError(
            "selected candidate/track cohort mismatch: selected=%s tracks=%s"
            % (sorted(selected), sorted(tracks))
        )
    if selected is not None and selected and not track_id:
        missing_pairs = [
            op_id for op_id in sorted(selected)
            if not all(
                isinstance(tracks[op_id].get(field), str)
                and bool(tracks[op_id].get(field).strip())
                for field in (
                    "implementer_agent", "implementer_rollout_id",
                    "monitor_agent", "monitor_evidence_path",
                    "monitor_offsets_path", "monitor_summary_path",
                )
            )
        ]
        if missing_pairs:
            raise ValueError(
                "selected track(s) lack durable implementer/monitor pairing "
                "evidence: %s" % ", ".join(missing_pairs)
            )
        current_round = _current_round(state)
        assert current_round is not None
        for op_id in sorted(selected):
            _verify_monitor_evidence(
                artifact_dir, current_round, op_id, tracks[op_id]
            )

    if tracks:
        selected_ids: Iterable[str]
        if track_id:
            if track_id not in tracks:
                raise KeyError(f"Track '{track_id}' not found in state.json")
            selected_ids = [track_id]
        else:
            selected_ids = tracks.keys()

        for selected_id in selected_ids:
            _validate_track_id(selected_id)
            metadata = tracks[selected_id]
            validation_value = metadata.get("validation_results_path") or _default_track_path(
                state, selected_id, "validation_results.md"
            )
            current_round = _current_round(state)
            if isinstance(state.get("campaign"), dict) and current_round is not None:
                validation_path = _validate_current_track_path(
                    str(validation_value), artifact_dir, current_round, selected_id
                )
            else:
                validation_path = _normalize_path(str(validation_value), artifact_dir)
            evidence_value = metadata.get("evidence_path") or _default_track_path(
                state, selected_id, "evidence.json"
            )
            if isinstance(state.get("campaign"), dict) and current_round is not None:
                evidence_path = _validate_current_track_path(
                    str(evidence_value), artifact_dir, current_round, selected_id
                )
            else:
                evidence_path = _normalize_path(str(evidence_value), artifact_dir)
            resolved.append(
                TrackContext(
                    track_id=selected_id,
                    metadata=metadata,
                    validation_path=validation_path,
                    validation_text=_read_text(validation_path) if validation_path.exists() else "",
                    evidence_path=evidence_path,
                    evidence=_read_json(evidence_path) if evidence_path.exists() else None,
                )
            )
        return resolved

    if isinstance(state.get("campaign"), dict):
        if _current_round(state) is None:
            raise ValueError(
                "Round-centric campaign state is incomplete; current-round "
                "evidence cannot be resolved"
            )
        raise ValueError("No tracks exist in the current round")
    if not state or "parallel_tracks" not in state:
        raise ValueError(
            "Root evidence fallback requires an explicit legacy state context"
        )

    validation_path = artifact_dir / "validation_results.md"
    evidence_path = artifact_dir / "evidence.json"
    return [
        TrackContext(
            track_id=track_id or "root",
            metadata={},
            validation_path=validation_path,
            validation_text=_read_text(validation_path) if validation_path.exists() else "",
            evidence_path=evidence_path,
            evidence=_read_json(evidence_path) if evidence_path.exists() else None,
        )
    ]


def _expect_dict(obj: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = obj.get(key)
    return value if isinstance(value, dict) else None


def _expect_number(obj: Dict[str, Any], key: str) -> Optional[float]:
    value = obj.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _evidence_terminal_verdict(evidence: Dict[str, Any]) -> str:
    for key in ("verdict", "overall_verdict", "track_verdict"):
        verdict = str(evidence.get(key, "")).upper()
        if verdict in TRACK_VERDICT_STATUSES:
            return verdict
    return ""


def _is_gpu_blocked(track: TrackContext) -> bool:
    evidence = track.evidence or {}
    return (
        str(track.metadata.get("status", "")).upper() == "GPU_BLOCKED"
        or _evidence_terminal_verdict(evidence) == "GPU_BLOCKED"
    )


def check_inline_kernel_gates(track: TrackContext) -> GateResult:
    """Verify the state-level Gate 5.1a/5.2 disposition.

    A terminal FAIL may retain failing gate outcomes as complete evidence. A
    passing pure inter-kernel track may skip both local-kernel gates only when
    each skip is explicit and explains the binding production boundary.
    """
    status = str(track.metadata.get("status", "")).upper()
    gate_5_1a = str(track.metadata.get("gate_5_1a", "")).upper()
    gate_5_2 = str(track.metadata.get("gate_5_2", "")).upper()
    if status == "FAIL":
        if gate_5_1a in {"PASS", "FAIL", "SKIPPED"} and gate_5_2 in {
            "PASS",
            "FAIL",
            "SKIPPED",
        }:
            return GateResult(
                "inline_kernel_gates",
                "PASS",
                "Terminal FAIL records complete inline gate outcomes",
                [f"gate_5_1a={gate_5_1a}", f"gate_5_2={gate_5_2}"],
                track.track_id,
            )

    if gate_5_1a == "PASS" and gate_5_2 == "PASS":
        return GateResult(
            "inline_kernel_gates",
            "PASS",
            "Gate 5.1a and Gate 5.2 PASS",
            [],
            track.track_id,
        )

    skip_5_1a = str(track.metadata.get("gate_5_1a_skip_reason", "")).lower()
    skip_5_2 = str(track.metadata.get("gate_5_2_skip_reason", "")).lower()
    pure_inter_kernel = (
        gate_5_1a == "SKIPPED"
        and gate_5_2 == "SKIPPED"
        and "inter-kernel" in skip_5_1a
        and "no kernel" in skip_5_1a
        and "component wall" in skip_5_2
    )
    if status in {"PASS", "GATED_PASS"} and pure_inter_kernel:
        return GateResult(
            "inline_kernel_gates",
            "PASS",
            "Pure inter-kernel track documents the inapplicable local gates",
            [skip_5_1a, skip_5_2],
            track.track_id,
        )

    return GateResult(
        "inline_kernel_gates",
        "FAIL",
        "Passing kernel-changing tracks require Gate 5.1a and Gate 5.2 PASS",
        [f"gate_5_1a={gate_5_1a}", f"gate_5_2={gate_5_2}"],
        track.track_id,
    )


def _approx_equal(a: float, b: float, tol: float = 0.25) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.01)


def _selected_candidate(state: Dict[str, Any], track_id: str) -> Dict[str, Any]:
    current_round = _current_round(state)
    rounds = (state.get("campaign") or {}).get("rounds") or []
    if current_round is None or len(rounds) < current_round:
        return {}
    debate = (rounds[current_round - 1].get("debate") or {})
    for candidate in debate.get("selected_candidates") or []:
        if isinstance(candidate, dict) and candidate.get("op_id") == track_id:
            return candidate
    return {}


def check_contingent_boundary(
    track: TrackContext, state: Dict[str, Any]
) -> GateResult:
    candidate = _selected_candidate(state, track.track_id)
    if candidate.get("selection_mode") != "contingent_host_spike":
        return GateResult(
            "contingent_boundary", "PASS", "Not a contingent host spike", [],
            track.track_id,
        )
    boundary = track.metadata.get("gate_5_2_boundary")
    if not isinstance(boundary, dict):
        return GateResult(
            "contingent_boundary", "FAIL",
            "Contingent host spike lacks reconciled boundary A/B arithmetic", [],
            track.track_id,
        )
    try:
        base = float(boundary["baseline_duration_us"])
        opt = float(boundary["optimized_duration_us"])
        count = int(boundary["occurrence_count"])
        e2e = float(boundary["baseline_e2e_us"])
        recorded = float(boundary["e2e_equivalent_improvement_pct"])
        floor = float(boundary["campaign_floor_pct"])
        threshold = float(
            (state.get("campaign") or {})["config"]["min_e2e_improvement_pct"]
        )
    except (KeyError, TypeError, ValueError):
        return GateResult(
            "contingent_boundary", "FAIL",
            "Contingent boundary A/B fields are incomplete or nonnumeric", [],
            track.track_id,
        )
    computed = 100.0 * (base - opt) * count / e2e if e2e > 0 else float("nan")
    valid = (
        base > 0 and opt >= 0 and count >= 1 and e2e > 0
        and math.isclose(recorded, computed, abs_tol=1e-6, rel_tol=0.01)
        and math.isclose(floor, threshold, abs_tol=1e-9, rel_tol=0.0)
        and boundary.get("meets_floor") is True
        and computed >= threshold
        and track.metadata.get("gate_5_2") == "PASS"
    )
    return GateResult(
        "contingent_boundary",
        "PASS" if valid else "FAIL",
        "Contingent boundary arithmetic reproduces Gate 5.2 PASS"
        if valid else "Contingent boundary arithmetic does not reproduce Gate 5.2 PASS",
        [f"computed_e2e_equivalent_pct={computed}", f"campaign_floor_pct={threshold}"],
        track.track_id,
    )


def check_track_state(track: TrackContext) -> GateResult:
    metadata = track.metadata
    evidence: List[str] = []
    status = str(metadata.get("status", "")).upper()

    if track.track_id != "root":
        if status == "GPU_BLOCKED":
            evidence.append("GPU_BLOCKED requires lead triage before terminal Stage 5 accounting")
            evidence.append("recorded blocker status: GPU_BLOCKED")
        elif status not in TERMINAL_TRACK_STATUSES:
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Track status is missing or not terminal in state.json",
                evidence=[f"track={track.track_id}", f"status={metadata.get('status')!r}"],
                track_id=track.track_id,
            )
        else:
            evidence.append(f"terminal status: {status}")

        for field_name in ["branch", "worktree_path", "validation_results_path"]:
            if not metadata.get(field_name):
                return GateResult(
                    name="track_state_recorded",
                    status="FAIL",
                    message=f"Track metadata missing required field '{field_name}'",
                    evidence=[f"track={track.track_id}"],
                    track_id=track.track_id,
                )
            evidence.append(f"{field_name}: {metadata.get(field_name)}")

        if "correctness" not in metadata:
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Track metadata missing 'correctness' field",
                evidence=[f"track={track.track_id}"],
                track_id=track.track_id,
            )
        evidence.append(f"correctness: {metadata.get('correctness')}")
        if status in {"PASS", "GATED_PASS"} and metadata.get("correctness") is not True:
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Passing track requires binding full-model correctness",
                evidence=[f"correctness={metadata.get('correctness')!r}"],
                track_id=track.track_id,
            )

        branch_check = _branch_matches_worktree(metadata)
        if branch_check and branch_check.startswith("expected "):
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Worktree branch does not match recorded branch",
                evidence=[branch_check],
                track_id=track.track_id,
            )
        if branch_check:
            evidence.append(f"worktree branch matches: {branch_check}")

    return GateResult(
        name="track_state_recorded",
        status="PASS",
        message=(
            "Track metadata is present with recorded GPU blocker"
            if status == "GPU_BLOCKED"
            else "Track metadata is present and terminal"
        ),
        evidence=evidence,
        track_id=track.track_id,
    )


def check_structured_evidence(track: TrackContext) -> GateResult:
    if track.evidence is None:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence JSON is missing",
            evidence=[str(track.evidence_path)],
            track_id=track.track_id,
        )

    version = track.evidence.get("schema_version")
    if version not in {2, 3}:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence must declare current v3 or legacy v2",
            evidence=[f"schema_version={version!r}"],
            track_id=track.track_id,
        )

    evidence_track = track.evidence.get("track_id")
    if evidence_track and evidence_track != track.track_id:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence track_id does not match the track being validated",
            evidence=[f"evidence.track_id={evidence_track!r}", f"track_id={track.track_id!r}"],
            track_id=track.track_id,
        )

    return GateResult(
        name="structured_evidence",
        status="PASS",
        message="Structured evidence JSON is present",
        evidence=[str(track.evidence_path)],
        track_id=track.track_id,
    )


def check_validation_summary(track: TrackContext) -> GateResult:
    if not track.validation_path.exists():
        return GateResult(
            name="validation_summary",
            status="FAIL",
            message="validation_results.md summary is missing",
            evidence=[str(track.validation_path)],
            track_id=track.track_id,
        )

    placeholder_hits = [pattern.pattern for pattern in PLACEHOLDER_PATTERNS if pattern.search(track.validation_text)]
    if placeholder_hits:
        return GateResult(
            name="validation_summary",
            status="FAIL",
            message="validation_results.md still contains placeholders or TODO markers",
            evidence=placeholder_hits,
            track_id=track.track_id,
        )

    return GateResult(
        name="validation_summary",
        status="PASS",
        message="validation_results.md summary is present",
        evidence=[str(track.validation_path)],
        track_id=track.track_id,
    )


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _path_from_reference(base: Path, value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _read_canonical_json(path: Path, label: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        errors.append(f"missing {label}: {path}")
        return None
    if path.is_symlink():
        errors.append(f"{label} must not be a symlink: {path}")
        return None
    try:
        value = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"invalid {label}: top level must be an object")
        return None
    return value


def _same_path(left: Any, right: Path, base: Path) -> bool:
    candidate = _path_from_reference(base, left)
    return candidate is not None and candidate.resolve() == right.resolve()


def _classify_e2e_row(
    speedup: float,
    min_pct: float,
    noise_pct: float,
    catastrophic_pct: float,
    significant: Optional[bool],
) -> str:
    if speedup >= 1.0 + max(min_pct, noise_pct) / 100.0 and significant is not False:
        return "PASS"
    if speedup >= 1.0 - noise_pct / 100.0:
        return "NOISE"
    if speedup >= 1.0 - catastrophic_pct / 100.0:
        return "REGRESSED"
    return "CATASTROPHIC"


def _rederive_diluted_pass(
    rows: List[Dict[str, Any]], noise_pct: float
) -> bool:
    if not rows:
        return False
    for row in rows:
        speedup = _finite_number(row.get("speedup"))
        phase_sig = row.get("phase_significance")
        if speedup is None or speedup < 1.0 or not isinstance(phase_sig, dict):
            return False
        decode_sig = phase_sig.get("decode")
        if not isinstance(decode_sig, dict):
            return False
        if decode_sig.get("significant") is not True:
            return False
        if not all(
            isinstance(decode_sig.get(key), int)
            and not isinstance(decode_sig.get(key), bool)
            and decode_sig[key] >= 30
            for key in ("n_baseline", "n_opt")
        ):
            return False

        bench = row.get("phase")
        if not isinstance(bench, dict):
            return False
        decode_delta = _finite_number(bench.get("decode_improvement_pct"))
        decode_share = _finite_number(bench.get("decode_share_of_e2e"))
        if (
            decode_delta is None
            or decode_share is None
            or decode_delta * decode_share < 2.0 * noise_pct
            or bench.get("amdahl_consistent") is not True
        ):
            return False
        prefill_delta = _finite_number(bench.get("prefill_improvement_pct"))
        prefill_sig = phase_sig.get("prefill")
        prefill_significant = (
            prefill_sig.get("significant")
            if isinstance(prefill_sig, dict)
            else None
        )
        if (
            prefill_delta is not None
            and prefill_delta < -noise_pct
            and prefill_significant is not False
        ):
            return False
    return True


def check_canonical_validation_artifacts(
    track: TrackContext,
    artifact_dir: Path,
    state: Dict[str, Any],
) -> GateResult:
    """Recompute Gate 5.1b and official E2E results from canonical artifacts."""
    errors: List[str] = []
    current_round = _current_round(state)
    if current_round is None:
        if not isinstance(state.get("campaign"), dict):
            return GateResult(
                "canonical_validation_artifacts",
                "PASS",
                "Canonical round-scoped artifacts are not applicable to "
                "explicit legacy state",
                [],
                track.track_id,
            )
        return GateResult(
            "canonical_validation_artifacts",
            "FAIL",
            "Current round is required for canonical validation",
            [],
            track.track_id,
        )

    round_dir = artifact_dir / "rounds" / str(current_round)
    correctness_path = (
        round_dir
        / "sweeps"
        / "opt_correctness"
        / track.track_id
        / "json"
        / "correctness_verdict.json"
    )
    sweep_path = (
        round_dir
        / "sweeps"
        / "opt"
        / track.track_id
        / "e2e_latency_results.json"
    )
    summary_path = (
        round_dir / "tracks" / track.track_id / "validation_summary.json"
    )
    target_path = artifact_dir / "target.json"

    correctness = _read_canonical_json(
        correctness_path, "Gate 5.1b correctness artifact", errors
    )
    sweep = _read_canonical_json(sweep_path, "official E2E sweep", errors)
    summary = _read_canonical_json(summary_path, "validation summary", errors)
    target = _read_canonical_json(target_path, "target contract", errors)

    if correctness is not None:
        if str(correctness.get("gate", "")) != "5.1b":
            errors.append("correctness artifact does not identify Gate 5.1b")
        questions = correctness.get("num_questions")
        baseline_count = correctness.get("baseline_correct_count")
        optimized_count = correctness.get("optimized_correct_count")
        valid_counts = (
            isinstance(questions, int)
            and not isinstance(questions, bool)
            and questions > 0
            and isinstance(baseline_count, int)
            and not isinstance(baseline_count, bool)
            and isinstance(optimized_count, int)
            and not isinstance(optimized_count, bool)
            and 0 <= baseline_count <= questions
            and 0 <= optimized_count <= questions
        )
        if not valid_counts:
            errors.append("correctness counts are invalid")
        else:
            baseline_accuracy = baseline_count / questions
            optimized_accuracy = optimized_count / questions
            declared_baseline = _finite_number(correctness.get("baseline_accuracy"))
            declared_optimized = _finite_number(correctness.get("optimized_accuracy"))
            declared_delta = _finite_number(correctness.get("accuracy_delta"))
            tolerance = _finite_number(correctness.get("tolerance_pct"))
            threshold = _finite_number(correctness.get("threshold"))
            expected_threshold = (
                baseline_accuracy - tolerance / 100.0
                if tolerance is not None
                else None
            )
            comparisons = [
                ("baseline_accuracy", declared_baseline, baseline_accuracy),
                ("optimized_accuracy", declared_optimized, optimized_accuracy),
                ("accuracy_delta", declared_delta, optimized_accuracy - baseline_accuracy),
                ("threshold", threshold, expected_threshold),
            ]
            for name, declared, expected in comparisons:
                if (
                    declared is None
                    or expected is None
                    or not math.isclose(declared, expected, abs_tol=1e-9)
                ):
                    errors.append(
                        f"correctness {name} does not match recomputed value"
                    )
            expected_verdict = (
                "PASS"
                if expected_threshold is not None
                and optimized_accuracy >= expected_threshold
                else "FAIL"
            )
            if str(correctness.get("verdict", "")).upper() != expected_verdict:
                errors.append("correctness verdict does not match recomputed counts")
            if (
                str(track.metadata.get("status", "")).upper()
                in {"PASS", "GATED_PASS"}
                and expected_verdict != "PASS"
            ):
                errors.append("passing track failed canonical correctness")

    campaign = state.get("campaign")
    config = campaign.get("config") if isinstance(campaign, dict) else {}
    target_gating = target.get("gating") if isinstance(target, dict) else {}
    target_workload = target.get("workload") if isinstance(target, dict) else {}
    min_pct = _finite_number(config.get("min_e2e_improvement_pct"))
    noise_pct = _finite_number(config.get("noise_tolerance_pct"))
    catastrophic_pct = _finite_number(config.get("catastrophic_regression_pct"))
    min_pct = 0.5 if min_pct is None else min_pct
    noise_pct = 0.5 if noise_pct is None else noise_pct
    catastrophic_pct = 5.0 if catastrophic_pct is None else catastrophic_pct
    for key, expected in (
        ("min_e2e_improvement_pct", min_pct),
        ("noise_tolerance_pct", noise_pct),
        ("catastrophic_regression_pct", catastrophic_pct),
    ):
        if isinstance(target_gating, dict):
            declared = _finite_number(target_gating.get(key))
            if declared is None or not math.isclose(declared, expected, abs_tol=1e-9):
                errors.append(f"target gating {key} disagrees with campaign state")

    canonical_rows: List[Dict[str, Any]] = []
    if sweep is not None:
        sweep_dir = sweep_path.parent
        if sweep.get("execution_mode") != "inproc_sweep":
            errors.append("official E2E sweep has invalid execution_mode")
        if not _same_path(sweep.get("out_dir"), sweep_dir, artifact_dir):
            errors.append("official E2E sweep out_dir is not canonical")
        if not _same_path(sweep.get("target_json"), target_path, artifact_dir):
            errors.append("official E2E sweep target_json is not canonical")
        expected_baseline = round_dir / "sweeps" / "baseline"
        if not _same_path(
            sweep.get("baseline_source"), expected_baseline, artifact_dir
        ):
            errors.append("official E2E sweep baseline source is not current-round")

        bench = sweep.get("bench")
        baseline_label = (
            bench.get("baseline_label") if isinstance(bench, dict) else None
        )
        opt_label = bench.get("opt_label") if isinstance(bench, dict) else None
        rows = sweep.get("results")
        if not baseline_label or not opt_label or not isinstance(rows, list) or not rows:
            errors.append("official E2E sweep labels/results are invalid")
            rows = []
        expected_batches = (
            target_workload.get("batch_sizes")
            if isinstance(target_workload, dict)
            else None
        )
        seen_batches: List[Any] = []
        for row in rows:
            if not isinstance(row, dict):
                errors.append("official E2E sweep row is invalid")
                continue
            seen_batches.append(row.get("batch_size"))
            baseline = row.get(baseline_label)
            optimized = row.get(opt_label)
            if not isinstance(baseline, dict) or not isinstance(optimized, dict):
                errors.append("official E2E sweep row is missing labeled arms")
                continue
            for arm_name, arm in (("baseline", baseline), ("optimized", optimized)):
                if arm.get("ok") is not True or arm.get("returncode") != 0:
                    errors.append(f"{arm_name} official runner did not succeed")
                output_path = _path_from_reference(sweep_dir, arm.get("output_json"))
                runner_path = _path_from_reference(sweep_dir, arm.get("runner_json"))
                for ref_name, ref_path in (
                    (f"{arm_name} output", output_path),
                    (f"{arm_name} runner", runner_path),
                ):
                    if (
                        ref_path is None
                        or not ref_path.resolve().is_relative_to(sweep_dir.resolve())
                        or not ref_path.exists()
                        or ref_path.is_symlink()
                    ):
                        errors.append(f"invalid canonical {ref_name} artifact")
                if runner_path is not None and runner_path.exists() and not runner_path.is_symlink():
                    runner = _read_canonical_json(
                        runner_path, f"{arm_name} runner artifact", errors
                    )
                    if runner is not None and runner.get("ok") is not True:
                        errors.append(f"{arm_name} official runner did not succeed")
                if output_path is not None and output_path.exists() and not output_path.is_symlink():
                    output = _read_canonical_json(
                        output_path, f"{arm_name} latency artifact", errors
                    )
                    output_avg = (
                        _finite_number(output.get("avg_latency"))
                        if output is not None
                        else None
                    )
                    row_avg = _finite_number(arm.get("avg_s"))
                    if (
                        output_avg is None
                        or row_avg is None
                        or not math.isclose(output_avg, row_avg, rel_tol=1e-9)
                    ):
                        errors.append(
                            f"{arm_name} average is invalid or disagrees with raw output"
                        )

            baseline_avg = _finite_number(baseline.get("avg_s"))
            optimized_avg = _finite_number(optimized.get("avg_s"))
            speedup = _finite_number(row.get("speedup"))
            improvement = _finite_number(row.get("improvement_pct"))
            if (
                baseline_avg is None
                or optimized_avg is None
                or baseline_avg <= 0
                or optimized_avg <= 0
                or speedup is None
                or improvement is None
            ):
                errors.append("official E2E row contains invalid non-finite metrics")
                continue
            expected_speedup = baseline_avg / optimized_avg
            expected_improvement = (1.0 - optimized_avg / baseline_avg) * 100.0
            if not math.isclose(speedup, expected_speedup, rel_tol=1e-9):
                errors.append("speedup does not match official arm latencies")
            if not math.isclose(improvement, expected_improvement, rel_tol=1e-9):
                errors.append("improvement_pct does not match official arm latencies")
            significance = row.get("significance")
            significant = (
                significance.get("significant")
                if isinstance(significance, dict)
                else None
            )
            phase: Dict[str, Any] = {}
            baseline_prefill = _finite_number(baseline.get("prefill_avg_s"))
            optimized_prefill = _finite_number(optimized.get("prefill_avg_s"))
            baseline_decode = _finite_number(baseline.get("decode_avg_s"))
            optimized_decode = _finite_number(optimized.get("decode_avg_s"))
            # Derive the share here rather than trusting the stored key: baselines
            # captured before the denominator fix carry decode_avg/(prefill_avg+
            # decode_avg) under this same name, which is larger and would relax the
            # DILUTED_PASS floor below. references/e2e-delta-math.md § Denominator
            # rule. Fall back to the stored value only when decode_avg_s is absent.
            if baseline_decode is not None:
                decode_share = baseline_decode / baseline_avg
            else:
                decode_share = _finite_number(baseline.get("decode_share_of_e2e"))
            if decode_share is not None:
                phase["decode_share_of_e2e"] = decode_share
            if (
                baseline_prefill is not None
                and baseline_prefill > 0
                and optimized_prefill is not None
            ):
                phase["prefill_improvement_pct"] = (
                    (baseline_prefill - optimized_prefill)
                    / baseline_prefill
                    * 100.0
                )
            if (
                baseline_decode is not None
                and baseline_decode > 0
                and optimized_decode is not None
            ):
                phase["decode_improvement_pct"] = (
                    (baseline_decode - optimized_decode)
                    / baseline_decode
                    * 100.0
                )
            if (
                decode_share is not None
                and "decode_improvement_pct" in phase
            ):
                expected_from_phases = (
                    decode_share * phase["decode_improvement_pct"]
                    + (1.0 - decode_share)
                    * phase.get("prefill_improvement_pct", 0.0)
                )
                phase["amdahl_consistent"] = abs(
                    expected_improvement - expected_from_phases
                ) <= max(noise_pct, 0.5 * abs(expected_from_phases))

            canonical_rows.append(
                {
                    **row,
                    "speedup": expected_speedup,
                    "improvement_pct": expected_improvement,
                    "verdict": _classify_e2e_row(
                        expected_speedup,
                        min_pct,
                        noise_pct,
                        catastrophic_pct,
                        significant,
                    ),
                    "phase": phase,
                    "_baseline_avg_s": baseline_avg,
                    "_optimized_avg_s": optimized_avg,
                }
            )
        if isinstance(expected_batches, list) and sorted(seen_batches) != sorted(expected_batches):
            errors.append("official E2E sweep does not cover target batch sizes")

    if summary is not None:
        inputs = summary.get("inputs")
        if not isinstance(inputs, dict) or not _same_path(
            inputs.get("e2e_json"), sweep_path, artifact_dir
        ):
            errors.append(
                "validation summary does not cite the canonical current-round "
                "official sweep"
            )
        gate = summary.get("e2e_gate")
        if not isinstance(gate, dict):
            errors.append("validation summary e2e_gate is invalid")
        else:
            thresholds = gate.get("thresholds")
            expected_thresholds = {
                "noise_tolerance_pct": noise_pct,
                "catastrophic_regression_pct": catastrophic_pct,
                "min_e2e_improvement_pct": min_pct,
                "pass_floor_speedup": 1.0 + max(min_pct, noise_pct) / 100.0,
            }
            if not isinstance(thresholds, dict):
                errors.append("validation summary thresholds are missing")
            else:
                for key, expected in expected_thresholds.items():
                    declared = _finite_number(thresholds.get(key))
                    if declared is None or not math.isclose(
                        declared, expected, abs_tol=1e-9
                    ):
                        errors.append(
                            f"validation summary threshold {key} is not canonical"
                        )
            expected_verdicts = [row["verdict"] for row in canonical_rows]
            if "CATASTROPHIC" in expected_verdicts:
                expected_track_verdict = "FAIL"
            elif "REGRESSED" in expected_verdicts and "PASS" in expected_verdicts:
                expected_track_verdict = "GATING_REQUIRED"
            elif "PASS" in expected_verdicts:
                expected_track_verdict = "PASS"
            else:
                expected_track_verdict = "FAIL"
            diluted_block = gate.get("diluted_pass")
            if diluted_block is not None:
                if (
                    not isinstance(diluted_block, dict)
                    or diluted_block.get("eligible") is not True
                    or not _rederive_diluted_pass(canonical_rows, noise_pct)
                ):
                    errors.append(
                        "diluted PASS is not supported by canonical phase proof"
                    )
                else:
                    expected_track_verdict = "PASS"
            if bool(track.metadata.get("diluted")) != (diluted_block is not None):
                errors.append("diluted PASS marker disagrees between state and summary")
            if str(gate.get("track_verdict", "")).upper() != expected_track_verdict:
                errors.append("validation summary track verdict is not reproducible")
            declared_rows = gate.get("per_bs_verdicts")
            if not isinstance(declared_rows, list) or len(declared_rows) != len(canonical_rows):
                errors.append("validation summary per-BS rows are incomplete")
            else:
                by_batch = {row.get("batch_size"): row for row in canonical_rows}
                for declared in declared_rows:
                    if not isinstance(declared, dict):
                        errors.append("validation summary per-BS row is invalid")
                        continue
                    expected = by_batch.get(declared.get("batch_size"))
                    if (
                        expected is None
                        or str(declared.get("verdict", "")).upper()
                        != expected["verdict"]
                        or _finite_number(declared.get("speedup")) is None
                        or not math.isclose(
                            float(declared["speedup"]),
                            expected["speedup"],
                            rel_tol=1e-9,
                        )
                    ):
                        errors.append("validation summary per-BS verdict is not reproducible")
            declared_state_verdicts = track.metadata.get("per_bs_verdict")
            if isinstance(declared_state_verdicts, dict):
                expected_state_verdicts = {
                    str(row.get("batch_size")): row["verdict"]
                    for row in canonical_rows
                }
                normalized_state_verdicts = {
                    str(key): str(value).upper()
                    for key, value in declared_state_verdicts.items()
                }
                if normalized_state_verdicts != expected_state_verdicts:
                    errors.append(
                        "state per_bs_verdict disagrees with canonical E2E sweep"
                    )

    structured_e2e = _expect_dict(track.evidence or {}, "e2e")
    if structured_e2e is not None and canonical_rows:
        # The evidence e2e block is scalar-shaped, so it summarizes the SMALLEST
        # batch size (the cumulative-speedup anchor). Reconcile against that row
        # on every sweep, single- or multi-BS: the previous
        # `len(canonical_rows) == 1` guard disabled the only primary-data
        # reconciliation on exactly the multi-BS campaigns that carry the most
        # hand-transcribed numbers.
        summary_row = min(
            canonical_rows,
            key=lambda r: (
                _finite_number(r.get("batch_size"))
                if _finite_number(r.get("batch_size")) is not None
                else float("inf")
            ),
        )
        comparisons = (
            ("baseline_avg_s", summary_row["_baseline_avg_s"]),
            ("optimized_avg_s", summary_row["_optimized_avg_s"]),
            ("speedup", summary_row["speedup"]),
            ("improvement_pct", summary_row["improvement_pct"]),
        )
        for key, expected in comparisons:
            declared = _finite_number(structured_e2e.get(key))
            if declared is None or not math.isclose(
                declared, float(expected), rel_tol=1e-9, abs_tol=1e-12
            ):
                errors.append(
                    f"structured E2E evidence {key} disagrees with official "
                    f"sweep row bs={summary_row.get('batch_size')}"
                )
        errors.extend(_reconcile_e2e_per_bs(structured_e2e, canonical_rows))

    errors.extend(
        _reconcile_kernel_bench_buckets(
            _expect_dict(track.evidence or {}, "kernel_bench"),
            target_workload.get("batch_sizes")
            if isinstance(target_workload, dict)
            else None,
        )
    )

    return GateResult(
        "canonical_validation_artifacts",
        "FAIL" if errors else "PASS",
        (
            "Canonical validation artifacts are inconsistent"
            if errors
            else "Canonical validation artifacts recompute cleanly"
        ),
        errors or [str(correctness_path), str(sweep_path), str(summary_path)],
        track.track_id,
    )


def _reconcile_kernel_bench_buckets(
    kernel: Optional[Dict[str, Any]], expected_batches: Any
) -> List[str]:
    """Require Gate-5.2 bucket coverage of every target batch size.

    ammo-impl-champion.md § inline gates mandates the gates run "against every
    target batch size" with "the same bucket set as Stage 1". Buckets index
    microbenchmark shapes, so only the batch identity is comparable to the sweep
    — the timings inside a bucket belong to a different measurement boundary and
    are NOT reconciled against E2E latencies. Buckets that declare no batch
    identity at all are left alone (a shape-only microbenchmark is legitimate).
    """
    if not isinstance(kernel, dict):
        return []
    if str(kernel.get("status", "")).upper() == "SKIPPED":
        return []
    buckets = kernel.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        return []
    if not isinstance(expected_batches, list) or len(expected_batches) < 2:
        return []

    seen = set()
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        for key in ("batch_size", "bs", "m", "M"):
            value = bucket.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            seen.add(value)
            break
    if not seen:
        return []
    missing = sorted(
        bs for bs in expected_batches
        if isinstance(bs, int) and not isinstance(bs, bool) and bs not in seen
    )
    if missing:
        return [
            "kernel_bench buckets do not cover target batch size(s) "
            + ", ".join(str(bs) for bs in missing)
        ]
    return []


def _reconcile_e2e_per_bs(
    structured_e2e: Dict[str, Any], canonical_rows: List[Dict[str, Any]]
) -> List[str]:
    """Reconcile the OPTIONAL `e2e.per_bs[]` array row-for-row against the sweep.

    The scalar e2e fields summarize the smallest BS only, so a multi-BS campaign
    can index every row here. The array stays optional for back-compat; when it
    is present it must cover every sweep row exactly.
    """
    declared_rows = structured_e2e.get("per_bs")
    if declared_rows is None:
        return []
    if not isinstance(declared_rows, list) or not declared_rows:
        return ["structured E2E evidence per_bs is not a non-empty list"]

    errors: List[str] = []
    by_batch = {row.get("batch_size"): row for row in canonical_rows}
    seen = []
    for declared in declared_rows:
        if not isinstance(declared, dict):
            errors.append("structured E2E evidence per_bs row is invalid")
            continue
        batch = declared.get("batch_size")
        seen.append(batch)
        expected = by_batch.get(batch)
        if expected is None:
            errors.append(
                f"structured E2E evidence per_bs row bs={batch} is not in the "
                "official sweep"
            )
            continue
        for key, value in (
            ("baseline_avg_s", expected["_baseline_avg_s"]),
            ("optimized_avg_s", expected["_optimized_avg_s"]),
            ("speedup", expected["speedup"]),
            ("improvement_pct", expected["improvement_pct"]),
        ):
            number = _finite_number(declared.get(key))
            if number is None or not math.isclose(
                number, float(value), rel_tol=1e-9, abs_tol=1e-12
            ):
                errors.append(
                    f"structured E2E evidence per_bs {key} disagrees with "
                    f"official sweep row bs={batch}"
                )
        verdict = declared.get("verdict")
        if verdict is not None and str(verdict).upper() != expected["verdict"]:
            errors.append(
                f"structured E2E evidence per_bs verdict disagrees with "
                f"official sweep row bs={batch}"
            )
    if sorted(map(str, seen)) != sorted(str(row.get("batch_size")) for row in canonical_rows):
        errors.append(
            "structured E2E evidence per_bs does not cover the official sweep "
            "batch sizes"
        )
    return errors


def check_stage1_baseline(track: TrackContext, artifact_dir: Path) -> GateResult:
    baseline_files, baseline_pattern = _stage1_baseline_files(artifact_dir)
    evidence = track.evidence or {}
    baseline = _expect_dict(evidence, "baseline_source")
    if baseline is None:
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Structured evidence is missing baseline_source",
            evidence=[str(track.evidence_path)],
            track_id=track.track_id,
        )

    kind = str(baseline.get("kind", "")).lower()
    citation = str(baseline.get("citation", ""))
    evidence_version = int(evidence.get("schema_version", 0) or 0)
    if evidence_version >= 3:
        # v3 evidence must bind the opt sweep result AND cite Stage 1 baseline
        # reuse — a worktree-run baseline arm can execute the optimized path
        # and contaminate both arms, so paired worktree A/B is not accepted.
        required = ("sweep_result",)
        missing = [key for key in required if not str(baseline.get(key) or "").strip()]
        sweep_ref = str(baseline.get("sweep_result") or "")
        sweep_path = Path(sweep_ref)
        if not sweep_path.is_absolute():
            sweep_path = artifact_dir / sweep_path
        if kind != "stage1" or missing or not sweep_path.is_file():
            return GateResult(
                name="stage1_baseline_reuse",
                status="FAIL",
                message=(
                    "Structured evidence requires Stage 1 baseline reuse "
                    "(kind='stage1') with the opt sweep result bound"
                ),
                evidence=[
                    f"kind={kind!r}",
                    f"missing={missing}",
                    f"sweep_result={sweep_ref!r}",
                ],
                track_id=track.track_id,
            )
        if "not re-run" not in citation.lower():
            return GateResult(
                name="stage1_baseline_reuse",
                status="FAIL",
                message="Structured evidence must cite Stage 1 reuse explicitly ('not re-run')",
                evidence=[f"kind={kind!r}", f"citation={citation!r}"],
                track_id=track.track_id,
            )
    elif kind != "stage1" or "not re-run" not in citation.lower():
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Legacy structured evidence must cite Stage 1 reuse explicitly",
            evidence=[f"kind={kind!r}", f"citation={citation!r}"],
            track_id=track.track_id,
        )

    if not baseline_files:
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Stage 1 baseline JSON files are missing from the current round baseline directory",
            evidence=[f"{baseline_pattern} not found"],
            track_id=track.track_id,
        )

    return GateResult(
        name="stage1_baseline_reuse",
        status="PASS",
        message=(
            "Structured evidence cites Stage 1 baseline reuse with the opt sweep bound"
            if evidence_version >= 3
            else "Legacy structured evidence cites Stage 1 baseline reuse"
        ),
        evidence=(
            [str(baseline.get("sweep_result"))]
            if evidence_version >= 3
            else [str(path.relative_to(artifact_dir)) for path in baseline_files]
        ),
        track_id=track.track_id,
    )


def check_correctness(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    correctness = _expect_dict(evidence, "correctness")
    if correctness is None:
        return GateResult(
            name="correctness_evidence",
            status="FAIL",
            message="Structured evidence is missing correctness",
            evidence=[],
            track_id=track.track_id,
        )

    status = str(correctness.get("status", "")).upper()
    track_status = str(track.metadata.get("status", "")).upper()
    skip_reason = str(correctness.get("reason", "")).lower()
    if status == "SKIPPED":
        pure_inter_kernel = (
            track.metadata.get("correctness") is True
            and "inter-kernel" in skip_reason
            and "no kernel" in skip_reason
        )
        if pure_inter_kernel:
            return GateResult(
                "correctness_evidence",
                "PASS",
                "Local kernel correctness is inapplicable; binding full-model "
                "correctness is recorded in track state",
                [str(correctness.get("reason", ""))],
                track.track_id,
            )
        return GateResult(
            "correctness_evidence",
            "FAIL",
            "Correctness may be skipped only for a documented pure "
            "inter-kernel change with binding full-model correctness",
            [str(correctness.get("reason", ""))],
            track.track_id,
        )
    method = str(correctness.get("method", ""))
    atol = _expect_number(correctness, "atol")
    rtol = _expect_number(correctness, "rtol")
    max_abs_diff = _expect_number(correctness, "max_abs_diff")
    nan_inf_check = correctness.get("nan_inf_check")
    graph_replay_check = correctness.get("graph_replay_check")

    if status not in PASS_FAIL:
        return GateResult("correctness_evidence", "FAIL", "correctness.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if "allclose" not in method and "assert_close" not in method:
        return GateResult("correctness_evidence", "FAIL", "correctness.method must name torch.allclose or assert_close", [method], track.track_id)
    if atol is None or rtol is None or max_abs_diff is None:
        return GateResult("correctness_evidence", "FAIL", "correctness must include numeric tolerances and max_abs_diff", [], track.track_id)
    if status == "PASS" and (
        nan_inf_check is not True or graph_replay_check is not True
    ):
        return GateResult("correctness_evidence", "FAIL", "correctness must explicitly pass NaN/INF and graph replay checks", [f"nan_inf_check={nan_inf_check!r}", f"graph_replay_check={graph_replay_check!r}"], track.track_id)
    if status == "FAIL" and track_status != "FAIL":
        return GateResult(
            "correctness_evidence",
            "FAIL",
            "Passing track cannot carry failed correctness evidence",
            [f"track_status={track_status}", "correctness.status=FAIL"],
            track.track_id,
        )

    return GateResult(
        name="correctness_evidence",
        status="PASS",
        message="Structured correctness evidence is complete",
        evidence=[f"method={method}", f"atol={atol}", f"rtol={rtol}", f"max_abs_diff={max_abs_diff}"],
        track_id=track.track_id,
    )


def check_kernel_bench(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    kernel = _expect_dict(evidence, "kernel_bench")
    if kernel is None:
        return GateResult("kernel_bench", "FAIL", "Structured evidence is missing kernel_bench", [], track.track_id)

    status = str(kernel.get("status", "")).upper()
    track_status = str(track.metadata.get("status", "")).upper()
    skip_reason = str(kernel.get("reason", "")).lower()
    if status == "SKIPPED":
        gate_status = str(track.metadata.get("gate_5_2", "")).upper()
        documented = (
            gate_status == "SKIPPED"
            and "component wall" in skip_reason
            and bool(track.metadata.get("gate_5_2_skip_reason"))
        )
        if documented:
            return GateResult(
                "kernel_bench",
                "PASS",
                "Kernel microbenchmark is inapplicable; component wall time "
                "is the binding production boundary",
                [str(kernel.get("reason", ""))],
                track.track_id,
            )
        return GateResult(
            "kernel_bench",
            "FAIL",
            "Kernel benchmark skip lacks a documented binding boundary",
            [str(kernel.get("reason", ""))],
            track.track_id,
        )
    weighted_speedup = _expect_number(kernel, "weighted_speedup")
    measured_under_cuda_graphs = kernel.get("measured_under_cuda_graphs")
    buckets = kernel.get("buckets")

    if status not in PASS_FAIL:
        return GateResult("kernel_bench", "FAIL", "kernel_bench.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if weighted_speedup is None:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must include weighted_speedup", [], track.track_id)
    if status == "PASS" and measured_under_cuda_graphs is not True:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must prove CUDA graph measurement", [f"measured_under_cuda_graphs={measured_under_cuda_graphs!r}"], track.track_id)
    if not isinstance(buckets, list) or not buckets:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must include at least one validated bucket", [], track.track_id)
    if status == "FAIL" and track_status != "FAIL":
        return GateResult(
            "kernel_bench",
            "FAIL",
            "Passing track cannot carry failed kernel benchmark evidence",
            [f"track_status={track_status}", "kernel_bench.status=FAIL"],
            track.track_id,
        )

    return GateResult(
        name="kernel_bench",
        status="PASS",
        message="Structured kernel benchmark evidence is complete",
        evidence=[f"weighted_speedup={weighted_speedup:.4f}x", f"bucket_count={len(buckets)}"],
        track_id=track.track_id,
    )


def check_e2e_validation(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    e2e = _expect_dict(evidence, "e2e")
    if e2e is None:
        return GateResult("e2e_validation", "FAIL", "Structured evidence is missing e2e", [], track.track_id)

    status = str(e2e.get("status", "")).upper()
    run_purpose = str(e2e.get("run_purpose", "")).lower()
    baseline_avg_s = _expect_number(e2e, "baseline_avg_s")
    optimized_avg_s = _expect_number(e2e, "optimized_avg_s")
    speedup = _expect_number(e2e, "speedup")
    improvement_pct = _expect_number(e2e, "improvement_pct")
    admissibility = _expect_dict(e2e, "admissibility") or {}
    fastpath_proof = _expect_dict(e2e, "fastpath_proof") or {}

    if status not in PASS_FAIL:
        return GateResult("e2e_validation", "FAIL", "e2e.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if run_purpose != "official":
        return GateResult("e2e_validation", "FAIL", "Only official runs may determine Stage 5 E2E verdicts", [f"run_purpose={run_purpose!r}"], track.track_id)
    if None in (baseline_avg_s, optimized_avg_s, speedup, improvement_pct):
        return GateResult("e2e_validation", "FAIL", "e2e must include numeric baseline, optimized, speedup, and improvement metrics", [], track.track_id)
    if str(admissibility.get("status", "")).upper() != "PASS":
        return GateResult("e2e_validation", "FAIL", "Official E2E run is not admissible", [json.dumps(admissibility, sort_keys=True)], track.track_id)

    proof_status = str(fastpath_proof.get("status", "")).upper()
    hits = fastpath_proof.get("hits")
    if proof_status != "PASS":
        return GateResult("e2e_validation", "FAIL", "Official optimized run is missing explicit fast-path proof", [json.dumps(fastpath_proof, sort_keys=True)], track.track_id)
    if not isinstance(hits, int) or hits < 1:
        return GateResult("e2e_validation", "FAIL", "Explicit fast-path proof must include hits >= 1", [json.dumps(fastpath_proof, sort_keys=True)], track.track_id)

    source_json = fastpath_proof.get("source_json")
    if isinstance(source_json, str) and source_json:
        source_path = Path(source_json)
        if not source_path.is_absolute():
            source_path = (track.evidence_path.parent / source_path).resolve()
        if not source_path.exists():
            return GateResult("e2e_validation", "FAIL", "fastpath_proof.source_json does not exist", [str(source_path)], track.track_id)

    return GateResult(
        name="e2e_validation",
        status="PASS",
        message="Structured E2E evidence is complete and admissible",
        evidence=[f"speedup={speedup:.4f}x", f"improvement_pct={improvement_pct:.3f}", f"fastpath_hits={hits}"],
        track_id=track.track_id,
    )


def check_kill_criteria(track: TrackContext, state: Dict[str, Any]) -> GateResult:
    evidence = track.evidence or {}
    kill_criteria = evidence.get("kill_criteria")
    if not isinstance(kill_criteria, dict) or not kill_criteria:
        return GateResult("kill_criteria", "FAIL", "Structured evidence is missing kill_criteria", [], track.track_id)

    structured_statuses: Dict[str, str] = {}
    evidence_lines: List[str] = []
    for name, result in kill_criteria.items():
        if not isinstance(result, dict):
            return GateResult("kill_criteria", "FAIL", "Each kill criterion must be an object", [f"{name}={result!r}"], track.track_id)
        status = str(result.get("status", "")).upper()
        source_run_purpose = str(result.get("source_run_purpose", "")).lower()
        promoted = bool(result.get("promoted", False))
        if status not in PASS_FAIL:
            return GateResult("kill_criteria", "FAIL", "Each kill criterion must have PASS or FAIL status", [f"{name}.status={status!r}"], track.track_id)
        if source_run_purpose != "official" and not promoted:
            return GateResult("kill_criteria", "FAIL", "Kill criteria must come from an official run unless explicitly promoted", [f"{name}.source_run_purpose={source_run_purpose!r}", f"{name}.promoted={promoted!r}"], track.track_id)
        structured_statuses[name] = status
        evidence_lines.append(f"{name}: {status}")

    state_results = track.metadata.get("kill_criteria_results") or state.get("route_decision", {}).get("kill_criteria_results") or {}
    if isinstance(state_results, dict) and state_results:
        mismatches = []
        for name, status in structured_statuses.items():
            state_status = str(state_results.get(name, "")).upper()
            if state_status and state_status != status:
                mismatches.append(f"{name}: evidence={status}, state={state_status}")
        if mismatches:
            return GateResult("kill_criteria", "FAIL", "Structured evidence and state.json disagree on kill criteria", mismatches, track.track_id)

    return GateResult("kill_criteria", "PASS", "Kill criteria are structured and consistent", evidence_lines, track.track_id)


def check_amdahl_sanity(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    amdahl = _expect_dict(evidence, "amdahl")
    e2e = _expect_dict(evidence, "e2e") or {}
    if amdahl is None:
        return GateResult("amdahl_sanity", "FAIL", "Structured evidence is missing amdahl", [], track.track_id)

    share = _expect_number(amdahl, "component_share_f")
    kernel_speedup = _expect_number(amdahl, "kernel_speedup")
    expected_pct = _expect_number(amdahl, "expected_e2e_pct")
    actual_pct = _expect_number(amdahl, "actual_e2e_pct")
    e2e_improvement = _expect_number(e2e, "improvement_pct")
    if None in (share, kernel_speedup, expected_pct, actual_pct, e2e_improvement):
        return GateResult("amdahl_sanity", "FAIL", "amdahl must include numeric component share, kernel speedup, expected, and actual values", [], track.track_id)

    expected_calc = share * (1.0 - (1.0 / kernel_speedup)) * 100.0
    if not _approx_equal(expected_pct, expected_calc):
        return GateResult("amdahl_sanity", "FAIL", "amdahl.expected_e2e_pct does not match the structured inputs", [f"expected_e2e_pct={expected_pct:.3f}", f"recomputed={expected_calc:.3f}"], track.track_id)
    if not _approx_equal(actual_pct, e2e_improvement):
        return GateResult("amdahl_sanity", "FAIL", "amdahl.actual_e2e_pct must match e2e.improvement_pct", [f"amdahl.actual_e2e_pct={actual_pct:.3f}", f"e2e.improvement_pct={e2e_improvement:.3f}"], track.track_id)

    return GateResult(
        name="amdahl_sanity",
        status="PASS",
        message="Amdahl sanity inputs are structured and internally consistent",
        evidence=[f"component_share_f={share:.4f}", f"kernel_speedup={kernel_speedup:.4f}x", f"expected_e2e_pct={expected_pct:.3f}", f"actual_e2e_pct={actual_pct:.3f}"],
        track_id=track.track_id,
    )


def check_cross_track_contamination(track: TrackContext, all_tracks: List[TrackContext]) -> GateResult:
    evidence = track.evidence or {}
    contamination = _expect_dict(evidence, "cross_track_contamination")
    if contamination is None:
        return GateResult("cross_track_contamination", "FAIL", "Structured evidence is missing cross_track_contamination", [], track.track_id)

    status = str(contamination.get("status", "")).upper()
    note = str(contamination.get("note", "")).strip()
    if status not in {"PASS", "FAIL", "N/A"}:
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination.status must be PASS, FAIL, or N/A", [f"status={status!r}"], track.track_id)
    if len(all_tracks) > 1 and status == "N/A":
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination cannot be N/A in multi-track validation", [], track.track_id)
    if not note:
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination.note is required", [], track.track_id)
    return GateResult("cross_track_contamination", "PASS", "Cross-track contamination audit is structured", [f"status={status}", note], track.track_id)


def check_early_kill_fail(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    track_status = str(track.metadata.get("status", "")).upper()
    failure_class = str(evidence.get("failure_class", "")).lower()
    official_e2e_run = evidence.get("official_e2e_run")
    attempts = evidence.get("implementation_attempts") or evidence.get("fix_attempts") or evidence.get("attempt_history")
    changed_files = evidence.get("changed_files")
    kill_criteria = evidence.get("kill_criteria")
    fallback = _expect_dict(evidence, "fallback_ladder") or {}

    blockers: List[str] = []
    if track_status != "FAIL" or _evidence_terminal_verdict(evidence) != "FAIL":
        blockers.append(f"track_status={track_status or 'UNKNOWN'} evidence_verdict={_evidence_terminal_verdict(evidence) or 'UNKNOWN'}")
    if failure_class not in {"pre_validation_early_kill", "kernel_timing_gate_fail"}:
        blockers.append(f"failure_class={failure_class or 'MISSING'}")
    if official_e2e_run is not False:
        blockers.append(f"official_e2e_run={official_e2e_run!r}")
    if not (isinstance(attempts, list) and attempts) and not (isinstance(changed_files, list) and changed_files):
        blockers.append("missing implementation_attempts/fix_attempts/attempt_history or changed_files")
    if not isinstance(kill_criteria, dict) or not kill_criteria:
        blockers.append("missing kill_criteria")
    else:
        has_fail = False
        for name, result in kill_criteria.items():
            if not isinstance(result, dict):
                blockers.append(f"{name}: kill criterion is not an object")
                continue
            status = str(result.get("status", "")).upper()
            promoted = bool(result.get("promoted", False))
            refs = result.get("evidence_refs")
            if status == "FAIL":
                has_fail = True
            if not promoted:
                blockers.append(f"{name}: promoted is not true")
            if not isinstance(refs, list) or not refs:
                blockers.append(f"{name}: missing evidence_refs")
        if not has_fail:
            blockers.append("kill_criteria has no FAIL entry")
    if str(fallback.get("disposition", "")).upper() != "FAIL":
        blockers.append("fallback_ladder.disposition is not FAIL")
    rungs = fallback.get("rungs_considered")
    if not isinstance(rungs, list) or not rungs:
        blockers.append("fallback_ladder.rungs_considered is missing")

    if blockers:
        return GateResult(
            "early_kill_fail",
            "FAIL",
            "Early-kill FAIL evidence is incomplete",
            blockers,
            track.track_id,
        )

    return GateResult(
        "early_kill_fail",
        "PASS",
        "Pre-validation terminal FAIL is explicitly documented",
        [
            f"failure_class={failure_class}",
            f"implementation_attempts={len(attempts) if isinstance(attempts, list) else 0}",
            f"changed_files={len(changed_files) if isinstance(changed_files, list) else 0}",
        ],
        track.track_id,
    )


def _is_early_kill_fail(track: TrackContext) -> bool:
    evidence = track.evidence or {}
    return (
        str(track.metadata.get("status", "")).upper() == "FAIL"
        and _evidence_terminal_verdict(evidence) == "FAIL"
        and str(evidence.get("failure_class", "")).lower()
        in {"pre_validation_early_kill", "kernel_timing_gate_fail"}
    )


def check_gpu_blocked_evidence(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    track_status = str(track.metadata.get("status", "")).upper()
    evidence_verdict = _evidence_terminal_verdict(evidence)
    gpu_blocked = _expect_dict(evidence, "gpu_blocked") or {}

    if track_status != "GPU_BLOCKED":
        return GateResult(
            "gpu_blocked_evidence",
            "FAIL",
            "GPU_BLOCKED evidence must be reconciled into state.json status",
            [f"state.status={track_status or 'UNKNOWN'}"],
            track.track_id,
        )
    if evidence_verdict != "GPU_BLOCKED":
        return GateResult(
            "gpu_blocked_evidence",
            "FAIL",
            "GPU_BLOCKED tracks must carry an explicit GPU_BLOCKED evidence verdict",
            [f"evidence verdict={evidence_verdict or 'UNKNOWN'}"],
            track.track_id,
        )

    reason = str(gpu_blocked.get("reason", "")).strip()
    evidence_items = gpu_blocked.get("evidence")
    if not reason or not isinstance(evidence_items, list) or not evidence_items:
        return GateResult(
            "gpu_blocked_evidence",
            "FAIL",
            "GPU_BLOCKED evidence must include gpu_blocked.reason and non-empty gpu_blocked.evidence",
            [
                f"reason_present={bool(reason)!r}",
                f"evidence_count={len(evidence_items) if isinstance(evidence_items, list) else 'invalid'}",
            ],
            track.track_id,
        )

    return GateResult(
        "gpu_blocked_evidence",
        "PASS",
        "GPU_BLOCKED evidence is reconciled and structured",
        [reason, f"evidence_count={len(evidence_items)}"],
        track.track_id,
    )


def _campaign_min_e2e_improvement_pct(state: Dict[str, Any]) -> float:
    campaign = state.get("campaign")
    config = campaign.get("config") if isinstance(campaign, dict) else None
    value = config.get("min_e2e_improvement_pct") if isinstance(config, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    # Pre-V2 evidence used a quarter-percent gated-pass floor. Malformed or
    # partial state cannot silently adopt the newer campaign default.
    return 0.25


def _gated_pass_results(post_gating_results: Any) -> List[tuple[str, str, Optional[float]]]:
    results: List[tuple[str, str, Optional[float]]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            raw_improvement = value.get("improvement_pct")
            improvement = (
                float(raw_improvement)
                if isinstance(raw_improvement, (int, float)) and not isinstance(raw_improvement, bool)
                else None
            )
            status = str(value.get("status") or value.get("verdict") or value.get("outcome") or "").upper()
            if status or improvement is not None:
                results.append((path, status, improvement))
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]")

    visit(post_gating_results, "post_gating_results")
    return results


def check_candidate_outcome(track: TrackContext, state: Optional[Dict[str, Any]] = None) -> GateResult:
    state = state or {}
    evidence = track.evidence or {}
    e2e = _expect_dict(evidence, "e2e") or {}
    kill_criteria = evidence.get("kill_criteria") or {}
    track_status = str(track.metadata.get("status", "")).upper()
    e2e_status = str(e2e.get("status", "")).upper()

    if track_status in {"PASS", "GATED_PASS"} and e2e_status != "PASS":
        return GateResult("candidate_outcome", "FAIL", "Track is marked passed in state.json but structured E2E verdict is not PASS", [f"track_status={track_status}", f"e2e.status={e2e_status}"], track.track_id)

    if track_status == "GATED_PASS":
        metadata_verdict = str(track.metadata.get("verdict", "")).upper()
        if metadata_verdict != "GATED_PASS":
            return GateResult(
                "candidate_outcome",
                "FAIL",
                "GATED_PASS tracks must carry metadata.verdict == GATED_PASS",
                [f"metadata.verdict={metadata_verdict or 'UNKNOWN'}"],
                track.track_id,
            )
        gating = track.metadata.get("gating")
        required_gating_fields = [
            "mechanism",
            "env_var",
            "dispatch_condition",
            "crossover_threshold_bs",
            "crossover_probing",
            "pre_gating_results",
            "post_gating_results",
        ]
        missing = []
        if not isinstance(gating, dict):
            missing = required_gating_fields
        else:
            for field_name in required_gating_fields:
                value = gating.get(field_name)
                if value is None or value == "" or value == [] or value == {}:
                    missing.append(field_name)
        if missing:
            return GateResult(
                "candidate_outcome",
                "FAIL",
                "GATED_PASS tracks require structured metadata.gating evidence",
                missing,
                track.track_id,
            )
        threshold = _campaign_min_e2e_improvement_pct(state)
        post_results = _gated_pass_results(gating.get("post_gating_results"))
        failures = [
            (path, status, improvement)
            for path, status, improvement in post_results
            if status and status not in {"PASS", "NOISE"}
        ]
        if failures:
            return GateResult(
                "candidate_outcome",
                "FAIL",
                "GATED_PASS post-gating results must pass across all reported batch sizes",
                [
                    f"{path}: status={status} improvement_pct={improvement:.3f}"
                    if improvement is not None
                    else f"{path}: status={status} improvement_pct=MISSING"
                    for path, status, improvement in failures
                ],
                track.track_id,
            )
        qualifying = [
            (path, status, improvement)
            for path, status, improvement in post_results
            if status in {"PASS", "NOISE"} and improvement is not None and improvement >= threshold
        ]
        if not qualifying:
            observed = [
                f"{path}: status={status or 'UNKNOWN'} improvement_pct={improvement:.3f}"
                if improvement is not None
                else f"{path}: status={status or 'UNKNOWN'} improvement_pct=MISSING"
                for path, status, improvement in post_results
            ]
            return GateResult(
                "candidate_outcome",
                "FAIL",
                "GATED_PASS requires at least one post-gating batch with E2E improvement >= campaign threshold",
                [f"threshold={threshold:.3f}", *observed],
                track.track_id,
            )

    failed_criteria = [
        name
        for name, result in kill_criteria.items()
        if isinstance(result, dict) and str(result.get("status", "")).upper() == "FAIL"
    ]
    if track_status in {"PASS", "GATED_PASS"} and failed_criteria:
        return GateResult("candidate_outcome", "FAIL", "Track is marked passed but one or more kill criteria failed", failed_criteria, track.track_id)

    if track_status == "FAIL":
        changed_files = evidence.get("changed_files")
        attempts = evidence.get("implementation_attempts") or evidence.get("fix_attempts") or evidence.get("attempt_history")
        has_changed_files = isinstance(changed_files, list) and bool(changed_files)
        has_attempts = isinstance(attempts, list) and bool(attempts)

        if not has_changed_files and not has_attempts:
            return GateResult(
                "candidate_outcome",
                "FAIL",
                "Selected track failed without changed files or documented implementation attempts",
                [
                    "Selection is an implementation mandate; no-patch FAIL is not a terminal implementation result.",
                    "Reopen/revise the dispatch packet or require a concrete implementation attempt before terminal FAIL.",
                ],
                track.track_id,
            )

    return GateResult("candidate_outcome", "PASS", "Track outcome is consistent with structured evidence", [f"track_status={track_status or 'UNKNOWN'}"], track.track_id)


def finalize_report_status(report: VerificationReport) -> None:
    if any(outcome == "GPU_BLOCKED" for outcome in report.track_outcomes.values()):
        report.overall_status = "GPU_BLOCKED"
        report.advance_to_stage6 = False
        report.recommendation = (
            "At least one track is GPU_BLOCKED. Lead triage is required before retrying, closing, "
            "or marking the round exhausted."
        )
    elif report.blockers:
        report.overall_status = "BLOCKED"
        report.advance_to_stage6 = False
        report.recommendation = "Validation evidence is incomplete or inconsistent. Fix every failing gate before Stage 6."
    elif any(outcome in {"PASS", "GATED_PASS"} for outcome in report.track_outcomes.values()):
        report.overall_status = "PASS"
        report.advance_to_stage6 = True
        if report.warnings:
            report.recommendation = "Validation evidence is complete. Stage 6 may proceed with the passing track set; warnings are recorded for follow-up."
        else:
            report.recommendation = "Validation evidence is complete. Stage 6 may proceed with the passing track set."
    else:
        report.overall_status = "EVIDENCE_COMPLETE_NO_PASS"
        report.advance_to_stage6 = True
        report.recommendation = (
            "Validation evidence is complete, but no track passed Stage 5. "
            "Proceed to Stage 6 only to mark the round EXHAUSTED; do not report candidate success or integrate a track."
        )


def verify_validation(artifact_dir: Path, track_id: Optional[str]) -> VerificationReport:
    state = _load_state(artifact_dir)
    tracks = _resolve_tracks(artifact_dir, state, track_id)
    report = VerificationReport(artifact_dir=str(artifact_dir))

    gates: List[GateResult] = []
    for track in tracks:
        report.track_outcomes[track.track_id] = str(track.metadata.get("status", "UNKNOWN")).upper() or "UNKNOWN"
        track_gates = [
            check_track_state(track),
            check_structured_evidence(track),
            check_validation_summary(track),
        ]
        if _is_gpu_blocked(track):
            track_gates.extend(
                [
                    check_gpu_blocked_evidence(track),
                    check_kill_criteria(track, state),
                    check_candidate_outcome(track, state),
                    check_contingent_boundary(track, state),
                ]
            )
        else:
            if _is_early_kill_fail(track):
                track_gates.extend(
                    [
                        check_stage1_baseline(track, artifact_dir),
                        check_early_kill_fail(track),
                        check_kill_criteria(track, state),
                        check_cross_track_contamination(track, tracks),
                        check_candidate_outcome(track, state),
                        check_contingent_boundary(track, state),
                    ]
                )
            else:
                track_gates.extend(
                    [
                        check_stage1_baseline(track, artifact_dir),
                        check_inline_kernel_gates(track),
                        check_correctness(track),
                        check_kernel_bench(track),
                        check_e2e_validation(track),
                        check_canonical_validation_artifacts(
                            track, artifact_dir, state
                        ),
                        check_kill_criteria(track, state),
                        check_amdahl_sanity(track),
                        check_cross_track_contamination(track, tracks),
                        check_candidate_outcome(track, state),
                        check_contingent_boundary(track, state),
                    ]
                )
        gates.extend(track_gates)

    report.gates = gates
    for gate in gates:
        label = f"{gate.track_id}:{gate.name}" if gate.track_id else gate.name
        if gate.status == "FAIL":
            report.blockers.append(f"{label}: {gate.message}")
        elif gate.status == "WARN":
            report.warnings.append(f"{label}: {gate.message}")

    finalize_report_status(report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", help="Artifact directory path")
    parser.add_argument("--json-output", default=None, help="Optional JSON report output path")
    parser.add_argument("--quiet", action="store_true", help="Only emit JSON")
    parser.add_argument("--track", default=None, help="Validate only one track id")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    if not artifact_dir.exists():
        print(f"ERROR: artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 2

    try:
        report = verify_validation(artifact_dir, args.track)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(report.to_dict(), indent=2)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")

    if args.quiet:
        print(output)
    else:
        print("=" * 60)
        print("AMMO Validation Gate Report")
        print("=" * 60)
        print(f"Artifact dir: {artifact_dir}")
        print(f"Overall status: {report.overall_status}")
        print(f"Advance to Stage 6: {report.advance_to_stage6}")
        print("Track outcomes:")
        for track_name, outcome in sorted(report.track_outcomes.items()):
            print(f"  - {track_name}: {outcome}")
        print()
        for gate in report.gates:
            prefix = f"[{gate.status}]"
            track_label = f" track={gate.track_id}" if gate.track_id else ""
            print(f"{prefix} {gate.name}{track_label}")
            print(f"  {gate.message}")
            for item in gate.evidence:
                print(f"  - {item}")
            print()
        print(output)

    return 0 if report.overall_status in {"PASS", "EVIDENCE_COMPLETE_NO_PASS"} else 1


if __name__ == "__main__":
    sys.exit(main())
