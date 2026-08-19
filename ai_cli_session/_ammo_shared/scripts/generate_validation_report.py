#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate `{artifact_dir}/validation_results.md` from recorded evidence.

This is deliberately conservative: it only reports what it can *prove* from
existing files. It will not invent kernel timings or correctness tolerances.

Inputs (all optional, but the report is more useful with them):
- {artifact_dir}/e2e_latency/e2e_latency_results.json (from scripts/run_vllm_bench_latency_sweep.py)

Outputs:
- {artifact_dir}/validation_results.md (overwritten)
- {artifact_dir}/validation_summary.json (machine-readable)

Guardrails:
- If evidence is missing, the report includes TODO blocks rather than guessing.

Example:
  python scripts/generate_validation_report.py --artifact-dir artifacts/qwen3_l40s_fp8_tp1
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _fmt_cmd(cmd: List[str], env_overrides: Dict[str, str]) -> str:
    env_prefix = " ".join([f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()])
    cmd_str = " ".join([shlex.quote(x) for x in cmd])
    return (env_prefix + " " + cmd_str).strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _classify_verdict(speedup: float, noise_tol: float, catastrophic_tol: float,
                      min_floor: float = 0.0,
                      significant: Optional[bool] = None) -> str:
    """Classify a per-BS E2E speedup into a tiered verdict.

    Args:
        speedup: E2E speedup ratio (e.g. 1.02 = 2% improvement, 0.97 = 3% regression).
        noise_tol: Noise tolerance as a fraction (e.g. 0.005 for 0.5%).
        catastrophic_tol: Catastrophic regression threshold as a fraction (e.g. 0.05 for 5%).
        min_floor: Minimum E2E improvement as a fraction (campaign
            min_e2e_improvement_pct / 100). PASS requires clearing
            `1 + max(min_floor, noise_tol)` — a delta smaller than the declared
            noise band or the campaign floor is not a win, it is NOISE.
        significant: Within-launch Welch verdict from the sweep row's
            `significance.significant` field. False blocks PASS (the delta is
            statistically indistinguishable from noise regardless of size);
            None (legacy rows without the field) leaves the floor as the sole
            promotion gate.

    Returns:
        One of "PASS", "NOISE", "REGRESSED", "CATASTROPHIC".
    """
    pass_floor = 1.0 + max(min_floor, noise_tol)
    if speedup >= pass_floor and significant is not False:
        return "PASS"
    elif speedup >= (1.0 - noise_tol):
        return "NOISE"
    elif speedup >= (1.0 - catastrophic_tol):
        return "REGRESSED"
    else:
        return "CATASTROPHIC"


def _row_phase_metrics(row: Dict[str, Any], baseline_label: str,
                       opt_label: str) -> Optional[Dict[str, float]]:
    """Extract the per-BS phase decomposition already harvested by the sweep.

    Sweep label entries carry ``prefill_avg_s`` / ``decode_avg_s`` /
    ``decode_share_of_e2e`` (from ``RequestOutput.metrics``) and derived
    ``tpot_s`` / ``otps``; rows carry ``tpot_improvement_pct`` /
    ``otps_gain_pct`` siblings. These are batch-mode phase decompositions of
    the SAME fixed-batch measurement — not serving TTFT/ITL under load.

    Returns None when the row has no phase data at all (legacy sweep JSON),
    so legacy artifacts produce the original report byte-for-byte.
    """
    b = row.get(baseline_label) if isinstance(row.get(baseline_label), dict) else {}
    o = row.get(opt_label) if isinstance(row.get(opt_label), dict) else {}

    def num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    out: Dict[str, float] = {}
    # Derive rather than trust the stored key: baselines written before the
    # denominator fix carry decode_avg/(prefill_avg+decode_avg) under this name,
    # which is larger and would let _classify_diluted_pass declare a BS eligible
    # on a win that is below floor. references/e2e-delta-math.md § Denominator rule.
    b_avg = num(b, "avg_s")
    b_dec = num(b, "decode_avg_s")
    if b_dec is not None and b_avg is not None and b_avg > 0:
        out["decode_share_of_e2e"] = b_dec / b_avg
    else:
        share = num(b, "decode_share_of_e2e")
        if share is not None:
            out["decode_share_of_e2e"] = share
    b_pf, o_pf = num(b, "prefill_avg_s"), num(o, "prefill_avg_s")
    if b_pf and o_pf and b_pf > 0:
        out["prefill_improvement_pct"] = (b_pf - o_pf) / b_pf * 100.0
    b_dc, o_dc = num(b, "decode_avg_s"), num(o, "decode_avg_s")
    if b_dc and o_dc and b_dc > 0:
        out["decode_improvement_pct"] = (b_dc - o_dc) / b_dc * 100.0
    for row_key in ("tpot_improvement_pct", "otps_gain_pct"):
        v = row.get(row_key)
        if isinstance(v, (int, float)):
            out[row_key] = float(v)
    o_otps = num(o, "otps")
    if o_otps is not None:
        out["opt_otps"] = o_otps
    return out or None


def _phase_flags(phase: Dict[str, float], improvement_pct: Optional[float],
                 verdict: str, min_floor_pct: float,
                 noise_tol_pct: float) -> Dict[str, Any]:
    """Derive the two informational phase signals for one per-BS row.

    - ``diluted_decode_win``: the decode-slice improvement clears the campaign
      floor while the e2e verdict is below PASS, AND the observed e2e delta
      matches the phase-decomposition expectation
      ``(1 - share) * prefill_delta + share * decode_delta``. The mechanism
      worked; the small e2e number is Amdahl dilution — the instrument
      working correctly, not instrument blindness. Never promotes a verdict.
    - ``phase_warning``: e2e PASS while a phase regressed beyond the noise
      tolerance — a net win masking a phase-level regression that a different
      ISL/OSL mix could flip. Must be addressed in Decision before SHIP.

    Phase metrics are per-request means with no per-iteration distributions,
    so these are informational triage, not statistical verdicts.
    """
    out: Dict[str, Any] = {}
    share = phase.get("decode_share_of_e2e")
    dc_imp = phase.get("decode_improvement_pct")
    pf_imp = phase.get("prefill_improvement_pct")

    if share is not None and dc_imp is not None:
        expected = share * dc_imp + (1.0 - share) * (pf_imp if pf_imp is not None else 0.0)
        out["expected_e2e_from_phases_pct"] = expected
        if improvement_pct is not None:
            consistent = abs(improvement_pct - expected) <= max(
                noise_tol_pct, 0.5 * abs(expected))
            out["amdahl_consistent"] = consistent
            out["diluted_decode_win"] = bool(
                verdict != "PASS"
                and dc_imp >= max(min_floor_pct, noise_tol_pct)
                and consistent
            )

    if verdict == "PASS":
        warnings = []
        if pf_imp is not None and pf_imp < -noise_tol_pct:
            warnings.append({"phase": "prefill", "improvement_pct": pf_imp})
        if dc_imp is not None and dc_imp < -noise_tol_pct:
            warnings.append({"phase": "decode", "improvement_pct": dc_imp})
        if warnings:
            out["phase_warnings"] = warnings
    return out


def _classify_diluted_pass(per_bs_verdicts: List[Dict[str, Any]],
                           phase_rows: List[Dict[str, Any]],
                           noise_tol_pct: float,
                           min_num_iters: int = 30) -> Optional[Dict[str, Any]]:
    """Constraint-6b ship gate, checked ONLY when the existing ladder already
    produced track_verdict == 'FAIL'. Returns None the moment any condition
    fails or evidence is incomplete (fail-closed — default FAIL stands
    unchanged). Returns {"eligible": True, "candidate_bs": [...],
    "decode_floor_pct": ..., "per_bs": [...]} only when every row in the
    sweep satisfies every mechanical condition. Requires
    entry["phase_significance"] to already be populated on each
    per_bs_verdicts item (see the wiring step in main()) — this function does
    not itself read the raw sweep JSON.
    """
    decode_floor_pct = 2.0 * noise_tol_pct   # e2e-EQUIVALENT floor (cond 2)
    by_bs_phase = {pr["batch_size"]: pr for pr in phase_rows}
    by_bs_sig = {v["batch_size"]: (v.get("phase_significance") or {}) for v in per_bs_verdicts}
    if not phase_rows or len(by_bs_phase) != len(per_bs_verdicts):
        return None                                   # incomplete phase coverage -> fail closed
    candidate_bs, per_bs_evidence = [], []
    for v in per_bs_verdicts:
        bs = v["batch_size"]
        speedup = v.get("speedup")
        if not isinstance(speedup, (int, float)) or speedup < 1.0:
            return None                               # cond 4: net regression anywhere -> disqualify
        pr = by_bs_phase.get(bs)
        if pr is None:
            return None                               # no phase evidence at this BS -> disqualify
        phase, flags = pr["phase"], pr["flags"]
        sig_at_bs = by_bs_sig.get(bs, {})
        dsig = sig_at_bs.get("decode")
        if not isinstance(dsig, dict):
            return None                               # cond 0/1: no decode Welch evidence -> disqualify
        n_b, n_o = dsig.get("n_baseline"), dsig.get("n_opt")
        if not (isinstance(n_b, int) and isinstance(n_o, int)
                and n_b >= min_num_iters and n_o >= min_num_iters):
            return None                               # cond 0: sample-floor precondition
        if dsig.get("significant") is not True:
            return None                               # cond 1: every row must carry significant decode evidence
        pf_imp = phase.get("prefill_improvement_pct")
        pf_sig = sig_at_bs.get("prefill")
        pf_significant = pf_sig.get("significant") if isinstance(pf_sig, dict) else None
        if pf_imp is not None and pf_imp < -noise_tol_pct and pf_significant is not False:
            return None                               # cond 3: prefill regressed, not ruled out as noise -> veto
        dc_imp = phase.get("decode_improvement_pct")
        share = phase.get("decode_share_of_e2e")
        e2e_equiv_decode_pct = dc_imp * share if (dc_imp is not None and share is not None) else None
        if e2e_equiv_decode_pct is None or e2e_equiv_decode_pct < decode_floor_pct:
            return None                               # cond 2: e2e-equivalent decode floor
        if flags.get("amdahl_consistent") is not True:
            return None                               # cond 5: None or False both fail closed
        candidate_bs.append(bs)
        per_bs_evidence.append({
            "batch_size": bs, "decode_improvement_pct": dc_imp,
            "decode_significant": True, "e2e_equiv_decode_pct": e2e_equiv_decode_pct,
            "prefill_improvement_pct": pf_imp, "speedup": speedup,
            "amdahl_consistent": True,
        })
    if not candidate_bs:
        return None
    return {"eligible": True, "candidate_bs": candidate_bs,
            "decode_floor_pct": decode_floor_pct, "per_bs": per_bs_evidence}


def _render_phase_section(phase_rows: List[Dict[str, Any]],
                          noise_tol_pct: float) -> str:
    """Render the per-BS phase-decomposition table + flag notes."""
    lines: List[str] = []
    lines.append("### Phase decomposition (prefill/decode, same instrument)")
    lines.append("")
    lines.append("> Batch-mode phase timing from `RequestOutput.metrics` — the same "
                 "fixed-batch measurement decomposed, NOT serving TTFT/ITL under load "
                 "(no queueing/arrival dynamics). Informational triage only: flags "
                 "never change a verdict. See `references/validation-defaults.md` "
                 "§ Phase Decomposition.")
    lines.append("")
    lines.append("| Batch Size | Decode share | Prefill Δ | Decode (TPOT) Δ | Opt OTPS (tok/s) | Expected E2E Δ (phases) | Actual E2E Δ | Flag |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")

    def fmt(x: Any, suffix: str = "") -> str:
        if not isinstance(x, (int, float)):
            return ""
        return f"{x:.3g}{suffix}"

    notes: List[str] = []
    for pr in phase_rows:
        phase = pr["phase"]
        flags = pr["flags"]
        bs = pr["batch_size"]
        flag_txt = ""
        if flags.get("diluted_decode_win"):
            flag_txt = "DILUTED-WIN"
            notes.append(
                f"- **BS {bs} DILUTED-WIN**: decode Δ "
                f"{fmt(phase.get('decode_improvement_pct'), '%')} × share "
                f"{fmt(phase.get('decode_share_of_e2e'))} ≈ E2E Δ "
                f"{fmt(pr.get('improvement_pct'), '%')} — Amdahl-consistent. The "
                f"mechanism worked; the small E2E number is dilution by prefill "
                f"share, not a failed optimization. If this track fails on the "
                f"E2E floor, record `fail_reason` as \"diluted (Amdahl-consistent)\", "
                f"not \"ineffective\".")
        elif flags.get("phase_warnings"):
            flag_txt = "PHASE-REGRESSION"
            for w in flags["phase_warnings"]:
                notes.append(
                    f"- **BS {bs} PHASE-REGRESSION**: E2E PASS but {w['phase']} "
                    f"regressed {fmt(w['improvement_pct'], '%')} (beyond noise "
                    f"tolerance {noise_tol_pct}%). The net win masks a phase-level "
                    f"regression; a workload with a different ISL/OSL mix could "
                    f"flip the sign. Address in § Decision (explain acceptability "
                    f"for the target workload, or gate) before SHIP.")
        elif flags.get("amdahl_consistent") is False:
            flag_txt = "INCONSISTENT"
            notes.append(
                f"- **BS {bs} INCONSISTENT**: actual E2E Δ "
                f"{fmt(pr.get('improvement_pct'), '%')} does not match the "
                f"phase-decomposition expectation "
                f"{fmt(flags.get('expected_e2e_from_phases_pct'), '%')} — the "
                f"wall-clock latency and `RequestOutput.metrics` timestamps "
                f"disagree. Investigate before trusting either number.")

        lines.append(
            "| {bs} | {share} | {pf} | {dc} | {otps} | {exp} | {act} | {flag} |".format(
                bs=bs,
                share=fmt(phase.get("decode_share_of_e2e")),
                pf=fmt(phase.get("prefill_improvement_pct"), "%"),
                dc=fmt(phase.get("decode_improvement_pct"), "%"),
                otps=fmt(phase.get("opt_otps")),
                exp=fmt(flags.get("expected_e2e_from_phases_pct"), "%"),
                act=fmt(pr.get("improvement_pct"), "%"),
                flag=flag_txt,
            ))

    lines.append("")
    if notes:
        lines.extend(notes)
        lines.append("")
    return "\n".join(lines)


def _render_e2e_section(e2e: Dict[str, Any]) -> str:
    baseline_label = e2e.get("bench", {}).get("baseline_label", "baseline")
    opt_label = e2e.get("bench", {}).get("opt_label", "opt")

    lines: List[str] = []
    lines.append("## E2E latency (vllm bench latency)")
    lines.append("")

    wl = e2e.get("workload", {})
    lines.append("Workload:")
    lines.append(f"- model_id: {e2e.get('model_id')}")
    lines.append(f"- input_len: {wl.get('input_len')}, output_len: {wl.get('output_len')}")
    lines.append(f"- tp: {e2e.get('tp')}, max_model_len: {e2e.get('max_model_len')}")
    lines.append(f"- num_iters: {wl.get('num_iters')}")
    lines.append("")

    # Table — detect heterogeneous rows (different IL/OL per row).
    results = e2e.get("results", [])
    if not isinstance(results, list):
        results = []

    il_ol_set: set = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        il = row.get("input_len")
        ol = row.get("output_len")
        if il is not None and ol is not None:
            il_ol_set.add((il, ol))
    heterogeneous = len(il_ol_set) > 1

    if heterogeneous:
        header = f"| Input Len | Output Len | Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement | Fast-path evidence |"
        sep = "|---:|---:|---:|---:|---:|---:|---:|---|"
    else:
        header = f"| Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement | Fast-path evidence |"
        sep = "|---:|---:|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)

    for row in results:
        if not isinstance(row, dict):
            continue
        bs = row.get("batch_size")
        b = row.get(baseline_label, {}) if isinstance(row.get(baseline_label), dict) else {}
        o = row.get(opt_label, {}) if isinstance(row.get(opt_label), dict) else {}

        b_avg = b.get("avg_s")
        o_avg = o.get("avg_s")
        speedup = row.get("speedup")
        improve = row.get("improvement_pct")
        evidence = o.get("fastpath_evidence", {}).get("status", "unknown")

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                return f"{x:.6g}"
            return str(x)

        if heterogeneous:
            il = row.get("input_len", "")
            ol = row.get("output_len", "")
            lines.append(f"| {il} | {ol} | {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% | {evidence} |")
        else:
            lines.append(f"| {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% | {evidence} |")

    lines.append("")

    # Repro commands (first bucket only, for brevity)
    if results:
        first = results[0]
        if isinstance(first, dict):
            b0 = first.get(baseline_label, {}) if isinstance(first.get(baseline_label), dict) else {}
            o0 = first.get(opt_label, {}) if isinstance(first.get(opt_label), dict) else {}

            b_cmd = b0.get("cmd")
            o_cmd = o0.get("cmd")
            b_env = b0.get("env_overrides", {}) if isinstance(b0.get("env_overrides"), dict) else {}
            o_env = o0.get("env_overrides", {}) if isinstance(o0.get("env_overrides"), dict) else {}

            if isinstance(b_cmd, list) and all(isinstance(x, str) for x in b_cmd):
                lines.append("Repro (baseline example):")
                lines.append("```bash")
                lines.append(_fmt_cmd(b_cmd, b_env))
                lines.append("```")
                lines.append("")

            if isinstance(o_cmd, list) and all(isinstance(x, str) for x in o_cmd):
                lines.append("Repro (optimized example):")
                lines.append("```bash")
                lines.append(_fmt_cmd(o_cmd, o_env))
                lines.append("```")
                lines.append("")

    lines.append("> Note: ensure CUDA graphs / torch.compile settings match production parity per `references/e2e-latency-guide.md`.")
    lines.append("")

    return "\n".join(lines)


def _is_v2_layout(artifact_dir: Path) -> bool:
    return (artifact_dir / "rounds").is_dir()


def _current_round(artifact_dir: Path) -> int:
    """Return the current campaign round, defaulting to round 1."""
    state = _load_json(artifact_dir / "state.json") or {}
    value = state.get("campaign", {}).get("current_round")
    return int(value) if isinstance(value, int) and value > 0 else 1


def _track_context(evidence_path: Path,
                   evidence: Dict[str, Any]) -> tuple[Path, Optional[int], str]:
    """Infer campaign root, round, and track from a canonical evidence path."""
    track_dir = evidence_path.parent
    track_id = evidence.get("track_id")
    if not isinstance(track_id, str) or not track_id:
        track_id = track_dir.name

    # v2: artifact/rounds/{N}/tracks/{op_id}/evidence.json
    if (track_dir.parent.name == "tracks"
            and track_dir.parent.parent.name.isdigit()
            and track_dir.parent.parent.parent.name == "rounds"):
        return track_dir.parents[3], int(track_dir.parent.parent.name), track_id

    # Legacy: artifact/tracks/{op_id}/evidence.json
    if track_dir.parent.name == "tracks":
        return track_dir.parents[1], None, track_id

    # Standalone evidence files remain usable; their directory is the artifact
    # root and the report/summary stay beside the input.
    return track_dir, None, track_id


def _resolve_e2e_path(artifact_dir: Path, round_arg: Optional[int],
                      slot: Optional[str]) -> Path:
    """v2: rounds/{N}/sweeps/{slot}/e2e_latency_results.json
    legacy: {artifact_dir}/e2e_latency/e2e_latency_results.json
    """
    if _is_v2_layout(artifact_dir) and round_arg is not None and slot:
        return (artifact_dir / "rounds" / str(round_arg) / "sweeps" / slot
                / "e2e_latency_results.json")
    return artifact_dir / "e2e_latency" / "e2e_latency_results.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence_pos", nargs="?",
                   help="Compatibility: path to a track evidence.json")
    p.add_argument("--artifact-dir", type=str, default=None)
    p.add_argument("--track", type=str, default=None)
    p.add_argument("--legacy-track-layout", action="store_true",
                   help=("With --artifact-dir --track, use "
                         "artifact_dir/tracks/{track} instead of the current "
                         "round-scoped track directory."))
    p.add_argument("--evidence-json", type=str, default=None)
    p.add_argument("--e2e-json", type=str, default=None)
    p.add_argument("--round", type=int, default=None,
                   help=("Campaign round (1-indexed). With --slot, reads "
                         "rounds/{N}/sweeps/{SLOT}/e2e_latency_results.json."))
    p.add_argument("--slot", type=str, default=None,
                   help="Sweep slot under rounds/{N}/sweeps/ (e.g. baseline, opt/op007, integration).")
    p.add_argument("--output-md", type=str, default=None)
    p.add_argument("--output", dest="output_alias", type=str, default=None,
                   help="Compatibility alias for --output-md")

    args = p.parse_args()
    evidence_arg = args.evidence_json or args.evidence_pos
    output_arg = args.output_md or args.output_alias
    track_evidence: Optional[Dict[str, Any]] = None
    track_dir: Optional[Path] = None
    track_id: Optional[str] = None

    if evidence_arg:
        evidence_path = Path(evidence_arg).expanduser().resolve()
        track_evidence = _load_json(evidence_path)
        if track_evidence is None:
            raise SystemExit(f"Evidence JSON not found: {evidence_path}")
        track_dir = evidence_path.parent
        artifact_dir, inferred_round, track_id = _track_context(
            evidence_path, track_evidence
        )
        if args.round is None:
            args.round = inferred_round
        if args.slot is None and args.round is not None:
            args.slot = f"opt/{track_id}"
    else:
        if not args.artifact_dir:
            raise SystemExit(
                "Either a positional evidence.json, --evidence-json, or "
                "--artifact-dir is required."
            )
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        if args.track:
            track_id = args.track
            if args.legacy_track_layout:
                track_dir = artifact_dir / "tracks" / track_id
            else:
                args.round = args.round or _current_round(artifact_dir)
                track_dir = (artifact_dir / "rounds" / str(args.round)
                             / "tracks" / track_id)
            evidence_path = track_dir / "evidence.json"
            track_evidence = _load_json(evidence_path)
            if track_evidence is None:
                raise SystemExit(f"Evidence JSON not found: {evidence_path}")
            if args.slot is None and not args.legacy_track_layout:
                args.slot = f"opt/{track_id}"

    if args.e2e_json:
        e2e_path = Path(args.e2e_json).expanduser().resolve()
    else:
        e2e_path = _resolve_e2e_path(artifact_dir, args.round, args.slot)
    target_path = artifact_dir / "target.json"

    e2e = _load_json(e2e_path)

    # Load gating thresholds from target.json (backward compat: use defaults if missing)
    target_config = _load_json(target_path) or {}
    gating_config = target_config.get("gating", {})
    noise_tol = gating_config.get("noise_tolerance_pct", 0.5) / 100.0
    catastrophic_tol = gating_config.get("catastrophic_regression_pct", 5.0) / 100.0

    # min_e2e_improvement_pct: prefer target.json gating block, fall back to
    # state.json campaign.config (where new_target.py scaffolds it), then the
    # documented default. PASS must clear max(min_floor, noise_tol) — see
    # references/validation-defaults.md § Per-BS Tiered Verdict.
    min_floor_pct = gating_config.get("min_e2e_improvement_pct")
    if not isinstance(min_floor_pct, (int, float)):
        state_config = (_load_json(artifact_dir / "state.json") or {})
        min_floor_pct = (
            state_config.get("campaign", {}).get("config", {})
            .get("min_e2e_improvement_pct", 0.5)
        )
    min_floor = float(min_floor_pct) / 100.0

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "inputs": {
            "e2e_json": str(e2e_path) if e2e is not None else None,
        },
        "status": {
            "has_e2e": e2e is not None,
        },
    }
    if track_evidence is not None:
        summary["track_id"] = track_id
        summary["inputs"]["evidence_json"] = str(evidence_path)

    lines: List[str] = []
    lines.append("# Validation Results (Phase 4)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']} (UTC)")
    lines.append("")

    lines.append("> Default gates + required reporting checklist: `references/validation-defaults.md`.")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append("")
    lines.append("- TODO: record GPU + driver, torch + CUDA, vLLM commit, model id + quant, and TP/EP topology (`references/validation-defaults.md` § Required reporting checklist).")
    lines.append("")

    # Correctness placeholder
    lines.append("## Correctness")
    lines.append("")
    correctness = (
        track_evidence.get("correctness", {})
        if isinstance(track_evidence, dict) else {}
    )
    if isinstance(correctness, dict) and correctness:
        lines.append(f"- Status: `{correctness.get('status', 'UNKNOWN')}`")
        lines.append(f"- Method: `{correctness.get('method', 'unknown')}`")
        lines.append(
            "- Tolerance (atol/rtol): "
            f"`{correctness.get('atol')}` / `{correctness.get('rtol')}`"
        )
        lines.append(
            f"- Max abs diff: `{correctness.get('max_abs_diff')}`"
        )
        summary["correctness_gate"] = {
            "status": correctness.get("status", "UNKNOWN"),
            "pass": correctness.get("status") == "PASS",
        }
    else:
        lines.append("- TODO: run model-appropriate correctness tests (prefer existing vLLM tests for the model).")
    lines.append("")

    # Kernel perf placeholder
    lines.append("## Kernel perf (CUDA graphs)")
    lines.append("")
    kernel = (
        track_evidence.get("kernel_bench", {})
        if isinstance(track_evidence, dict) else {}
    )
    if isinstance(kernel, dict) and kernel:
        lines.append(f"- Status: `{kernel.get('status', 'UNKNOWN')}`")
        lines.append(
            "- Measured under CUDA graphs: "
            f"`{kernel.get('measured_under_cuda_graphs')}`"
        )
        lines.append(f"- Weighted speedup: `{kernel.get('weighted_speedup')}x`")
    else:
        lines.append("- TODO: add GPU kernel-time table (baseline vs optimized) under CUDA graphs for the validated bucket set.")
    lines.append("")

    # E2E section
    if e2e is None:
        lines.append("## E2E latency (vllm bench latency)")
        lines.append("")
        lines.append("- TODO: run `python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir} --run`.")
        lines.append("")
    else:
        lines.append(_render_e2e_section(e2e))

        # Compute per-BS tiered verdicts using thresholds from target.json
        buckets = e2e.get("results", [])
        bench = e2e.get("bench", {}) if isinstance(e2e.get("bench"), dict) else {}
        baseline_label = bench.get("baseline_label", "baseline")
        opt_label = bench.get("opt_label", "opt")

        per_bs_verdicts: List[Dict[str, Any]] = []
        phase_rows: List[Dict[str, Any]] = []
        if isinstance(buckets, list):
            for row in buckets:
                if not isinstance(row, dict):
                    continue
                bs = row.get("batch_size") or row.get("num_prompts")
                speedup = row.get("speedup")
                if bs is None or not isinstance(speedup, (int, float)):
                    continue
                sig = row.get("significance")
                significant = sig.get("significant") if isinstance(sig, dict) else None
                verdict = _classify_verdict(float(speedup), noise_tol, catastrophic_tol,
                                            min_floor=min_floor, significant=significant)
                entry = {"batch_size": bs, "speedup": speedup, "verdict": verdict}
                if significant is not None:
                    entry["significant"] = bool(significant)

                # CRITICAL WIRING STEP: copy the sweep's row-level phase Welch
                # onto the per-BS entry so _classify_diluted_pass can read it.
                # Without this copy, by_bs_sig is always {} and DILUTED_PASS can
                # never fire (dead on arrival). Omit-not-null: only when present.
                phase_sig_field = row.get("phase_significance")
                if isinstance(phase_sig_field, dict):
                    entry["phase_significance"] = phase_sig_field

                # Phase decomposition (informational, never changes the verdict).
                phase = _row_phase_metrics(row, baseline_label, opt_label)
                if phase is not None:
                    improvement_pct = row.get("improvement_pct")
                    imp = float(improvement_pct) if isinstance(improvement_pct, (int, float)) else None
                    flags = _phase_flags(
                        phase, imp, verdict,
                        min_floor_pct=min_floor * 100.0,
                        noise_tol_pct=noise_tol * 100.0,
                    )
                    entry["phase"] = {**phase, **flags}
                    phase_rows.append({
                        "batch_size": bs,
                        "improvement_pct": imp,
                        "phase": phase,
                        "flags": flags,
                    })

                per_bs_verdicts.append(entry)

        # Compute track-level verdict
        verdicts = [v["verdict"] for v in per_bs_verdicts]
        has_pass = "PASS" in verdicts
        has_catastrophic = "CATASTROPHIC" in verdicts
        has_regressed = "REGRESSED" in verdicts

        if has_catastrophic:
            track_verdict = "FAIL"
        elif has_regressed and has_pass:
            track_verdict = "GATING_REQUIRED"
        elif has_pass:
            track_verdict = "PASS"
        else:
            track_verdict = "FAIL"  # All NOISE/REGRESSED, no PASS

        # DILUTED_PASS ship path: pure post-hoc reclassification of a FAIL track
        # whose decode-phase win is Welch-significant, >= 2x noise floor,
        # Amdahl-consistent, with no BS regressed on E2E or prefill. Checked ONLY
        # when the ladder already produced FAIL — never overrides PASS/GATING.
        diluted = None
        if track_verdict == "FAIL" and phase_rows:
            diluted = _classify_diluted_pass(per_bs_verdicts, phase_rows, noise_tol * 100.0)
        if diluted is not None:
            track_verdict = "PASS"

        # A performance result can never rescue failed or incomplete model
        # correctness. Campaign-only reports have no track evidence and retain
        # their historical E2E-only classification behavior.
        if track_evidence is not None:
            correctness_status = correctness.get("status")
            if correctness_status != "PASS":
                track_verdict = "FAIL"
                diluted = None

        if phase_rows:
            lines.append(_render_phase_section(phase_rows, noise_tol * 100.0))
            if diluted is not None:
                lines.append(
                    "\n**Ships as PASS (diluted)**: E2E gate did not clear its floor at any "
                    "BS, but decode-phase win is Welch-significant, >= 2x noise floor, "
                    "Amdahl-consistent, and no BS regressed on E2E or prefill. Cumulative "
                    "accounting uses the MEASURED ~1.00x E2E, not the TPOT gain shown above — "
                    "see `references/validation-defaults.md` § DILUTED_PASS Ship Path."
                )

        summary["e2e_gate"] = {
            "per_bs_verdicts": per_bs_verdicts,
            "track_verdict": track_verdict,
            "thresholds": {
                "noise_tolerance_pct": gating_config.get("noise_tolerance_pct", 0.5),
                "catastrophic_regression_pct": gating_config.get("catastrophic_regression_pct", 5.0),
                "min_e2e_improvement_pct": float(min_floor_pct),
                "pass_floor_speedup": 1.0 + max(min_floor, noise_tol),
            },
            "pass": track_verdict in ("PASS", "GATING_REQUIRED", "GATED_PASS"),  # backward compat
            "failing": [v for v in per_bs_verdicts if v["verdict"] in ("REGRESSED", "CATASTROPHIC")],  # backward compat
        }
        # DILUTED_PASS evidence block (omit-not-null — only when the diluted
        # ship path fired). summary["e2e_gate"] must already exist here.
        if diluted is not None:
            summary["e2e_gate"]["diluted_pass"] = diluted
        # Track-level phase rollup (informational — never alters track_verdict).
        if phase_rows:
            summary["e2e_gate"]["phase_triage"] = {
                "diluted_decode_win_bs": [
                    pr["batch_size"] for pr in phase_rows
                    if pr["flags"].get("diluted_decode_win")
                ],
                "phase_regression_bs": [
                    pr["batch_size"] for pr in phase_rows
                    if pr["flags"].get("phase_warnings")
                ],
                "amdahl_inconsistent_bs": [
                    pr["batch_size"] for pr in phase_rows
                    if pr["flags"].get("amdahl_consistent") is False
                ],
            }

    # Decision placeholder
    lines.append("## Decision")
    lines.append("")
    lines.append("- TODO: Ship / restrict envelope / pivot route / stop (justify using kernel + E2E evidence).")
    lines.append("")

    if track_dir is not None:
        out_md = (
            Path(output_arg).expanduser().resolve()
            if output_arg else track_dir / "validation_results.md"
        )
        out_summary = track_dir / "validation_summary.json"
    else:
        out_md = (
            Path(output_arg).expanduser().resolve()
            if output_arg else artifact_dir / "validation_results.md"
        )
        out_summary = artifact_dir / "validation_summary.json"

    _write_text(out_md, "\n".join(lines))
    _write_json(out_summary, summary)

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
