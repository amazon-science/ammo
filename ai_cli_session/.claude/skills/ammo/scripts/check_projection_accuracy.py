#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Stage-5 projected-vs-realized E2E accuracy check (Phase 1D / spec §3.8).

Compares the champion's declared `projected_e2e_improvement_pct` (from
`state.json` → `campaign.rounds[current_round-1].debate.selected_candidates[i]`)
against the realized per-batch-size `improvement_pct` from the Stage 5 sweep
(`{artifact_dir}/{sweep_dir}/e2e_latency_results.json`).

Flagging rules (per BS):
- realized_pct < 0           → flag "PROJECTION MISMATCH — regression"
- 0 ≤ realized_pct ≤ 0.1     → flag if projected_pct > 1.0
- else                        → flag if (projected/realized) > 2.0
                                OR |projected - realized| > max(0.5, 2 × |realized|)

Behaviour:
- Always appends a `## Projection Accuracy` section to
  `{artifact_dir}/validation_results.md` (creates the file if absent).
- Always exits 0 (informational backstop — never blocks).

Usage:
    python check_projection_accuracy.py \
        --artifact-dir <path> \
        --op-id <op_id_to_inspect> \
        [--sweep-dir <dirname-relative-to-artifact-dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SECTION_HEADER = "## Projection Accuracy"


# ─── State / sweep readers ──────────────────────────────────────────────────


def _read_state(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = artifact_dir / "state.json"
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _find_projected_pct(state: Dict[str, Any], op_id: str) -> Optional[float]:
    """Return the `projected_e2e_improvement_pct` for the matching op_id, or None."""
    try:
        campaign = state.get("campaign") or {}
        rounds = campaign.get("rounds") or []
        cr = campaign.get("current_round") or len(rounds)
        if not rounds:
            return None
        idx = max(0, min(int(cr) - 1, len(rounds) - 1))
        debate = (rounds[idx].get("debate") or {})
        for cand in debate.get("selected_candidates") or []:
            if cand.get("op_id") == op_id:
                v = cand.get("projected_e2e_improvement_pct")
                if isinstance(v, (int, float)):
                    return float(v)
                return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def _read_realized(artifact_dir: Path, sweep_dir: str) -> List[Dict[str, Any]]:
    """Read per-BS rows from {artifact_dir}/{sweep_dir}/e2e_latency_results.json.

    Returns list of {batch_size, improvement_pct} dicts. Empty list if absent
    or unparseable.
    """
    path = artifact_dir / sweep_dir / "e2e_latency_results.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    rows: List[Dict[str, Any]] = []
    # Accept several common shapes
    candidates: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for key in ("results", "per_batch", "rows"):
            v = data.get(key)
            if isinstance(v, list):
                candidates = v
                break
        if not candidates and "improvement_pct" in data:
            # Single-row at top level
            candidates = [data]
    elif isinstance(data, list):
        candidates = data

    for row in candidates:
        if not isinstance(row, dict):
            continue
        bs = row.get("batch_size")
        ip = row.get("improvement_pct")
        if bs is None and ip is None:
            continue
        if not isinstance(ip, (int, float)):
            continue
        rows.append({"batch_size": bs, "improvement_pct": float(ip)})
    return rows


# ─── Flagging rules (spec §3.8) ────────────────────────────────────────────


def _evaluate_row(
    projected: Optional[float], realized: float
) -> Dict[str, Any]:
    """Return {flag: bool, reason: str, ratio: Optional[float], abs_diff: float}."""
    if projected is None:
        return {
            "flag": False,
            "reason": "no projection recorded",
            "ratio": None,
            "abs_diff": 0.0,
        }

    if realized < 0:
        return {
            "flag": True,
            "reason": "PROJECTION MISMATCH — regression (realized < 0 cleared other gates)",
            "ratio": None,
            "abs_diff": abs(projected - realized),
        }

    abs_diff = abs(projected - realized)
    if realized <= 0.1:
        # Noise-floor edge case
        if projected > 1.0:
            return {
                "flag": True,
                "reason": (
                    f"PROJECTION MISMATCH — realized≈0 ({realized:+.2f}%) but projected="
                    f"{projected:+.2f}% > 1.0%"
                ),
                "ratio": None,
                "abs_diff": abs_diff,
            }
        return {
            "flag": False,
            "reason": "realized at noise floor; projection within 1.0% — OK",
            "ratio": None,
            "abs_diff": abs_diff,
        }

    ratio = projected / realized if realized != 0 else None
    diff_threshold = max(0.5, 2.0 * abs(realized))

    flagged = False
    parts: List[str] = []
    if ratio is not None and ratio > 2.0:
        flagged = True
        parts.append(f"ratio={ratio:.2f}× > 2.0")
    if abs_diff > diff_threshold:
        flagged = True
        parts.append(f"|Δ|={abs_diff:.2f}pp > max(0.5, 2×|realized|)={diff_threshold:.2f}")

    if flagged:
        return {
            "flag": True,
            "reason": "PROJECTION MISMATCH — " + "; ".join(parts),
            "ratio": ratio,
            "abs_diff": abs_diff,
        }
    return {
        "flag": False,
        "reason": "within tolerance",
        "ratio": ratio,
        "abs_diff": abs_diff,
    }


# ─── Section rendering ─────────────────────────────────────────────────────


def _render_section(
    op_id: str,
    projected: Optional[float],
    rows_with_eval: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(SECTION_HEADER)
    lines.append("")
    proj_label = (
        f"{projected:+.2f}%" if isinstance(projected, (int, float)) else "n/a"
    )
    lines.append(f"- Op ID: `{op_id}`")
    lines.append(f"- Projected E2E improvement: {proj_label}")
    if not rows_with_eval:
        lines.append("- No realized E2E results found — skipping comparison.")
        lines.append("")
        return "\n".join(lines)

    lines.append("")
    lines.append("| BS | Projected | Realized | |Δ| (pp) | Ratio | Flag |")
    lines.append("|----|-----------|----------|---------|-------|------|")
    flagged_rows: List[str] = []
    for entry in rows_with_eval:
        bs = entry["batch_size"]
        realized = entry["realized"]
        ev = entry["eval"]
        flag_cell = "FLAG" if ev["flag"] else "ok"
        ratio_cell = f"{ev['ratio']:.2f}×" if ev["ratio"] is not None else "—"
        proj_cell = (
            f"{projected:+.2f}%" if isinstance(projected, (int, float)) else "n/a"
        )
        lines.append(
            f"| {bs} | {proj_cell} | {realized:+.2f}% | {ev['abs_diff']:.2f} | "
            f"{ratio_cell} | {flag_cell} |"
        )
        if ev["flag"]:
            flagged_rows.append(f"  - BS={bs}: {ev['reason']}")

    lines.append("")
    if flagged_rows:
        lines.append("**Flagged batch sizes** (informational — does not block):")
        lines.extend(flagged_rows)
    else:
        lines.append("All batch sizes within projection tolerance.")
    lines.append("")
    return "\n".join(lines)


def _is_v2_layout(artifact_dir: Path) -> bool:
    return (artifact_dir / "rounds").is_dir()


def _resolve_validation_results_path(
    artifact_dir: Path, round_arg: Optional[int], track_id: Optional[str]
) -> Path:
    """v2: rounds/{N}/tracks/{op_id}/validation_results.md (per-track)
    legacy: {artifact_dir}/validation_results.md
    """
    if _is_v2_layout(artifact_dir) and round_arg is not None and track_id:
        return (
            artifact_dir / "rounds" / str(round_arg) / "tracks" / track_id
            / "validation_results.md"
        )
    return artifact_dir / "validation_results.md"


def _append_section(out: Path, section_text: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        existing = out.read_text()
        if not existing.endswith("\n"):
            existing += "\n"
        out.write_text(existing + section_text)
    else:
        out.write_text("# Validation Results\n" + section_text)


# ─── CLI entry ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument(
        "--op-id",
        required=True,
        help="op_id of the candidate whose projected_e2e_improvement_pct to check.",
    )
    p.add_argument(
        "--sweep-dir",
        default=None,
        help=("Sub-directory under artifact-dir containing e2e_latency_results.json. "
              "If omitted on a v2 layout with --round/--track-id, defaults to "
              "rounds/{N}/sweeps/opt/{track_id}/. Legacy default: 'e2e_latency'."),
    )
    p.add_argument(
        "--round", type=int, default=None,
        help=("Campaign round (1-indexed). Combined with --track-id, redirects "
              "validation_results.md output to rounds/{N}/tracks/{track_id}/."),
    )
    p.add_argument(
        "--track-id", type=str, default=None,
        help=("Track op_id. Combined with --round, redirects output to the "
              "per-track validation_results.md and auto-derives sweep_dir."),
    )
    args = p.parse_args(argv)

    try:
        return _run(args.artifact_dir, args.op_id, args.sweep_dir,
                    round_arg=args.round, track_id=args.track_id)
    except Exception as e:  # pragma: no cover - fail-open guard
        # Always exit 0 (informational only).
        print(f"check_projection_accuracy: internal error: {e}", file=sys.stderr)
        return 0


def _run(artifact_dir: Path, op_id: str, sweep_dir: Optional[str],
         round_arg: Optional[int] = None, track_id: Optional[str] = None) -> int:
    state = _read_state(artifact_dir)
    projected = _find_projected_pct(state, op_id) if state else None

    # v2: derive sweep_dir from --round/--track-id when not given explicitly.
    if sweep_dir is None:
        if _is_v2_layout(artifact_dir) and round_arg is not None and track_id:
            sweep_dir = f"rounds/{round_arg}/sweeps/opt/{track_id}"
        else:
            sweep_dir = "e2e_latency"

    realized_rows = _read_realized(artifact_dir, sweep_dir)

    rows_with_eval = []
    for row in realized_rows:
        ev = _evaluate_row(projected, row["improvement_pct"])
        rows_with_eval.append(
            {"batch_size": row["batch_size"], "realized": row["improvement_pct"], "eval": ev}
        )

    section = _render_section(op_id, projected, rows_with_eval)
    out = _resolve_validation_results_path(artifact_dir, round_arg, track_id)
    _append_section(out, section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
