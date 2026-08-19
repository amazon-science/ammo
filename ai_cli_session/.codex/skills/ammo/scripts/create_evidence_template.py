#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Create the current AMMO evidence.json skeleton for one track."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
LEGACY_STATE_MARKERS = {
    "baseline",
    "parallel_tracks",
    "route_decision",
    "selected_candidates",
    "stage",
}


def template(track_id: str) -> dict[str, Any]:
    """Return a fail-closed structured-evidence skeleton."""
    _validate_track_id(track_id)
    return {
        "schema_version": 3,
        "track_id": track_id,
        "diluted": False,
        "baseline_source": {
            "kind": "stage1",
            "citation": "Baseline source: Stage 1 (not re-run).",
            "sweep_result": None,
        },
        "correctness": {
            "status": "FAIL",
            "method": "torch.allclose",
            "atol": None,
            "rtol": None,
            "max_abs_diff": None,
            "nan_inf_check": False,
            "graph_replay_check": False,
        },
        "kernel_bench": {
            "status": "FAIL",
            "weighted_speedup": None,
            "measured_under_cuda_graphs": False,
            "buckets": [],
            "boundary_ab": None,
        },
        "e2e": {
            "status": "FAIL",
            "run_purpose": "official",
            "baseline_avg_s": None,
            "optimized_avg_s": None,
            "speedup": None,
            "improvement_pct": None,
            "admissibility": {"status": "FAIL", "issues": []},
            "fastpath_proof": {"status": "FAIL", "hits": 0},
        },
        "kill_criteria": {},
        "amdahl": {
            "component_share_f": None,
            "kernel_speedup": None,
            "expected_e2e_pct": None,
            "actual_e2e_pct": None,
        },
        "cross_track_contamination": {
            "status": "N/A",
            "note": "single-track validation or fill multi-track audit",
        },
    }


def _validate_track_id(track_id: str) -> None:
    if not isinstance(track_id, str) or not TRACK_ID_PATTERN.fullmatch(track_id):
        raise ValueError(f"Invalid track id: {track_id!r}")


def _load_state(artifact_dir: Path) -> dict[str, Any] | None:
    state_path = artifact_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON {state_path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"{state_path} must contain a JSON object")
    return state


def _v2_current_round(state: dict[str, Any]) -> int | None:
    if "campaign" not in state:
        return None
    campaign = state.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("Round-centric state.json campaign must be an object")
    value = campaign.get("current_round")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "Round-centric state.json requires a positive integer current_round"
        )
    return value


def _is_explicit_legacy_state(state: dict[str, Any] | None) -> bool:
    return bool(
        state
        and "campaign" not in state
        and LEGACY_STATE_MARKERS.intersection(state)
    )


def _current_round(artifact_dir: Path) -> int:
    """Return current V2 round, preserving round 1 for unscaffolded output."""
    state = _load_state(artifact_dir)
    if state is None or "campaign" not in state:
        return 1
    current_round = _v2_current_round(state)
    assert current_round is not None
    return current_round


def _discover_artifact_dir(output: Path) -> Path | None:
    """Find the nearest state.json ancestor for output-only invocations."""
    for parent in output.absolute().parents:
        if (parent / "state.json").is_file():
            return parent.resolve()
    return None


def _same_lexical_path(left: Path, right: Path) -> bool:
    """Compare normalized paths without allowing a symlink alias."""
    return Path(os.path.abspath(left)) == Path(os.path.abspath(right))


def _resolve_output(
    *,
    artifact_dir: Path | None,
    output: Path | None,
    track_id: str,
    round_arg: int | None,
    legacy: bool,
) -> Path:
    discovered = _discover_artifact_dir(output) if output is not None else None
    if artifact_dir is None:
        artifact_dir = discovered
    elif discovered is not None and discovered != artifact_dir:
        raise ValueError(
            "--output belongs to a different artifact directory than --artifact-dir"
        )

    state = _load_state(artifact_dir) if artifact_dir is not None else None
    current_round = _v2_current_round(state) if state is not None else None

    if legacy and round_arg is not None:
        raise ValueError("--legacy and --round cannot be used together")

    if current_round is not None:
        if legacy:
            raise ValueError(
                "V2 round-centric campaigns cannot write legacy root evidence"
            )
        if round_arg is not None and round_arg != current_round:
            raise ValueError(
                f"Cannot write round {round_arg}; state.json current_round is "
                f"{current_round}"
            )
        assert artifact_dir is not None
        expected = (
            artifact_dir
            / "rounds"
            / str(current_round)
            / "tracks"
            / track_id
            / "evidence.json"
        )
        if output is not None and (
            not _same_lexical_path(output, expected)
            or output.resolve() != expected.resolve()
        ):
            raise ValueError(
                "V2 evidence output must be in the current-round track directory: "
                f"{expected}"
            )
        return expected

    if state is not None and "campaign" in state:
        # `_v2_current_round` fails before this branch. Keep this explicit for
        # type checkers and to document that V2 never falls back to flat paths.
        raise ValueError("Round-centric state.json has no usable current round")

    if legacy:
        if not _is_explicit_legacy_state(state):
            raise ValueError(
                "Legacy output requires explicit legacy state.json metadata"
            )
        assert artifact_dir is not None
        expected = artifact_dir / "tracks" / track_id / "evidence.json"
        if output is not None and (
            not _same_lexical_path(output, expected)
            or output.resolve() != expected.resolve()
        ):
            raise ValueError(f"Legacy evidence output must be {expected}")
        return expected

    if output is not None:
        return output
    assert artifact_dir is not None
    round_id = round_arg if round_arg is not None else 1
    if isinstance(round_id, bool) or round_id < 1:
        raise ValueError("--round must be a positive integer")
    return (
        artifact_dir
        / "rounds"
        / str(round_id)
        / "tracks"
        / track_id
        / "evidence.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir_pos", nargs="?", help="Artifact directory")
    parser.add_argument(
        "--artifact-dir", dest="artifact_dir_opt", help="Artifact directory alias"
    )
    parser.add_argument(
        "--track-id",
        "--track",
        dest="track_id",
        required=True,
        help="Track id, e.g. op001",
    )
    parser.add_argument(
        "--round",
        dest="round_id",
        type=int,
        default=None,
        help="Campaign round for round-centric layout",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Write artifact_dir/tracks/{track_id}/evidence.json for legacy state",
    )
    parser.add_argument(
        "--output",
        "--output-json",
        dest="output",
        default=None,
        help="Optional explicit output path",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        _validate_track_id(args.track_id)
        artifact_value = args.artifact_dir_opt or args.artifact_dir_pos
        if args.artifact_dir_opt and args.artifact_dir_pos:
            positional = Path(args.artifact_dir_pos).expanduser().resolve()
            option = Path(args.artifact_dir_opt).expanduser().resolve()
            if positional != option:
                raise ValueError(
                    "positional artifact directory and --artifact-dir disagree"
                )
        if not artifact_value and not args.output:
            parser.error(
                "artifact_dir positional/--artifact-dir or --output is required"
            )

        artifact_dir = (
            Path(artifact_value).expanduser().resolve() if artifact_value else None
        )
        output = Path(args.output).expanduser().absolute() if args.output else None
        out = _resolve_output(
            artifact_dir=artifact_dir,
            output=output,
            track_id=args.track_id,
            round_arg=args.round_id,
            legacy=args.legacy,
        )
        if out.exists() and not args.force:
            raise ValueError(f"Refusing to overwrite {out}; use --force")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(template(args.track_id), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(out)
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
