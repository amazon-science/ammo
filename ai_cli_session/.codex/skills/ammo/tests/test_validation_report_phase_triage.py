#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the phase-decomposition triage layer in generate_validation_report.py.

The layer consumes ONLY fields the sweep already harvests (prefill_avg_s /
decode_avg_s / decode_share_of_e2e / tpot_s / otps + row-level
tpot_improvement_pct / otps_gain_pct) — zero extra benchmark runs. It is
informational: flags must NEVER change a per-BS or track verdict.

Pinned behaviors:
  1. DILUTED-WIN: decode win >= floor, e2e below PASS, Amdahl-consistent
     -> diluted_decode_win=true, verdict unchanged (still NOISE).
  2. PHASE-REGRESSION: e2e PASS but a phase regressed beyond noise tolerance
     -> phase_warnings emitted, verdict unchanged (still PASS).
  3. INCONSISTENT: actual e2e delta disagrees with the phase-decomposition
     expectation -> amdahl_consistent=false, no diluted_decode_win.
  4. Legacy rows without phase fields -> no phase section, no phase keys
     (byte-compatible report path).
  5. Clean PASS with consistent phases -> no flags, table still rendered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_GEN = _SCRIPTS_DIR / "generate_validation_report.py"


def _payload(rows: list[dict]) -> str:
    return json.dumps({
        "model_id": "test/model",
        "tp": 1,
        "max_model_len": 4096,
        "workload": {"input_len": 1024, "output_len": 128, "num_iters": 10},
        "results": rows,
        "bench": {"baseline_label": "baseline", "opt_label": "opt"},
    })


def _row(bs: int, speedup: float, *, b_prefill: float | None = None,
         b_decode: float | None = None, o_prefill: float | None = None,
         o_decode: float | None = None, significant: bool | None = None,
         phase_significance: dict | None = None) -> dict:
    b_avg = 1.0
    o_avg = b_avg / speedup
    b: dict = {"avg_s": b_avg}
    o: dict = {"avg_s": o_avg}
    if b_prefill is not None and b_decode is not None:
        b["prefill_avg_s"] = b_prefill
        b["decode_avg_s"] = b_decode
        b["decode_share_of_e2e"] = b_decode / (b_prefill + b_decode)
    if o_prefill is not None and o_decode is not None:
        o["prefill_avg_s"] = o_prefill
        o["decode_avg_s"] = o_decode
        o["decode_share_of_e2e"] = o_decode / (o_prefill + o_decode)
    row = {
        "batch_size": bs,
        "speedup": speedup,
        "improvement_pct": (1.0 - o_avg / b_avg) * 100.0,
        "baseline": b,
        "opt": o,
    }
    if significant is not None:
        row["significance"] = {"significant": significant}
    if phase_significance is not None:
        row["phase_significance"] = phase_significance
    return row


def _run(tmp_path: Path, rows: list[dict]) -> tuple[str, dict]:
    artifact = tmp_path / "artifact"
    e2e_dir = artifact / "e2e_latency"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "e2e_latency_results.json").write_text(_payload(rows))
    result = subprocess.run(
        [sys.executable, str(_GEN), "--artifact-dir", str(artifact)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"report generation failed: {result.stderr}"
    md = (artifact / "validation_results.md").read_text()
    summary = json.loads((artifact / "validation_summary.json").read_text())
    return md, summary


class TestDilutedWin:
    """A real decode win diluted below the e2e floor by prefill share."""

    def test_flags_diluted_decode_win_without_promoting_verdict(self, tmp_path):
        # decode improves 4% but decode share is only 0.10:
        # expected e2e ~ 0.10*4% = 0.4% < 0.5% floor -> NOISE.
        # speedup consistent with the phase expectation.
        b_pf, b_dc = 0.90, 0.10
        o_dc = b_dc * (1 - 0.04)          # 4% decode win
        o_pf = b_pf                        # prefill unchanged
        o_avg_ratio = (o_pf + o_dc) / (b_pf + b_dc)
        speedup = 1.0 / o_avg_ratio        # ~1.004
        md, summary = _run(tmp_path, [_row(
            1, speedup, b_prefill=b_pf, b_decode=b_dc,
            o_prefill=o_pf, o_decode=o_dc)])

        verdicts = summary["e2e_gate"]["per_bs_verdicts"]
        assert verdicts[0]["verdict"] == "NOISE", \
            "phase triage must not promote the verdict"
        assert verdicts[0]["phase"]["diluted_decode_win"] is True
        assert summary["e2e_gate"]["phase_triage"]["diluted_decode_win_bs"] == [1]
        assert "DILUTED-WIN" in md
        assert "Amdahl-consistent" in md

    def test_track_verdict_unchanged_by_dilution_flag(self, tmp_path):
        b_pf, b_dc = 0.90, 0.10
        o_dc = b_dc * (1 - 0.04)
        o_avg_ratio = (b_pf + o_dc) / (b_pf + b_dc)
        md, summary = _run(tmp_path, [_row(
            1, 1.0 / o_avg_ratio, b_prefill=b_pf, b_decode=b_dc,
            o_prefill=b_pf, o_decode=o_dc)])
        # All-NOISE rows -> FAIL candidate; the flag must not rescue it.
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"


class TestPhaseRegression:
    """A net e2e PASS masking a phase-level regression."""

    def test_pass_with_prefill_regression_warns(self, tmp_path):
        # decode improves 6%, prefill regresses 2%; decode share 0.8
        # -> e2e ~ +4.4% PASS, but prefill regressed beyond 0.5% noise.
        b_pf, b_dc = 0.20, 0.80
        o_pf = b_pf * 1.02
        o_dc = b_dc * (1 - 0.06)
        o_avg_ratio = (o_pf + o_dc) / (b_pf + b_dc)
        md, summary = _run(tmp_path, [_row(
            8, 1.0 / o_avg_ratio, b_prefill=b_pf, b_decode=b_dc,
            o_prefill=o_pf, o_decode=o_dc, significant=True)])

        verdicts = summary["e2e_gate"]["per_bs_verdicts"]
        assert verdicts[0]["verdict"] == "PASS", \
            "phase warning must not demote a PASS"
        warns = verdicts[0]["phase"]["phase_warnings"]
        assert [w["phase"] for w in warns] == ["prefill"]
        assert summary["e2e_gate"]["phase_triage"]["phase_regression_bs"] == [8]
        assert "PHASE-REGRESSION" in md

    def test_clean_pass_no_flags(self, tmp_path):
        # Uniform 3% win in both phases: PASS, no flags, table rendered.
        b_pf, b_dc = 0.30, 0.70
        md, summary = _run(tmp_path, [_row(
            1, 1.0 / 0.97, b_prefill=b_pf, b_decode=b_dc,
            o_prefill=b_pf * 0.97, o_decode=b_dc * 0.97, significant=True)])
        phase = summary["e2e_gate"]["per_bs_verdicts"][0]["phase"]
        assert "phase_warnings" not in phase
        assert phase.get("diluted_decode_win") is not True
        triage = summary["e2e_gate"]["phase_triage"]
        assert triage["diluted_decode_win_bs"] == []
        assert triage["phase_regression_bs"] == []
        assert "Phase decomposition" in md
        assert "DILUTED-WIN" not in md
        assert "PHASE-REGRESSION" not in md


class TestAmdahlConsistency:
    def test_inconsistent_phase_math_flagged_not_diluted(self, tmp_path):
        # Phases claim a 4% decode win at share 0.9 (expected e2e ~3.6%)
        # but wall-clock speedup says 0.1% -> INCONSISTENT, not DILUTED-WIN.
        b_pf, b_dc = 0.10, 0.90
        o_dc = b_dc * (1 - 0.04)
        md, summary = _run(tmp_path, [_row(
            1, 1.001, b_prefill=b_pf, b_decode=b_dc,
            o_prefill=b_pf, o_decode=o_dc)])
        phase = summary["e2e_gate"]["per_bs_verdicts"][0]["phase"]
        assert phase["amdahl_consistent"] is False
        assert phase.get("diluted_decode_win") is not True
        assert summary["e2e_gate"]["phase_triage"]["amdahl_inconsistent_bs"] == [1]
        assert "INCONSISTENT" in md


class TestLegacyPassthrough:
    def test_rows_without_phase_fields_render_original_report(self, tmp_path):
        md, summary = _run(tmp_path, [_row(1, 1.05)])
        assert "Phase decomposition" not in md
        assert "phase" not in summary["e2e_gate"]["per_bs_verdicts"][0]
        assert "phase_triage" not in summary["e2e_gate"]


class TestDilutedPassIntegration:
    """Two cases wired into the existing triage suite (design §5 line item).

    Uses the file's default gating (no target.json, so min_e2e floor and
    noise are both 0.5%). The DILUTED_PASS window opens by having a
    non-significant prefill regression pull e2e into the NOISE band while a
    Welch-significant decode win clears the e2e-equivalent decode floor
    (decode_improvement_pct * decode_share >= 2*noise = 1.0%)."""

    def _diluted_row(self, bs: int) -> dict:
        # b: prefill 0.2 / decode 0.8. opt: prefill regresses 10% (0.22),
        # decode wins 3% (0.776). e2e improvement ~0.4% (NOISE), speedup
        # 1.004 (>=1.0). decode e2e-equiv = 3% * 0.8 = 2.4% (>=1.0 floor).
        return _row(
            bs, 1.0 / 0.996, b_prefill=0.2, b_decode=0.8,
            o_prefill=0.22, o_decode=0.776,
            phase_significance={
                "decode": {"significant": True, "n_baseline": 30, "n_opt": 30},
                "prefill": {"significant": False, "n_baseline": 30, "n_opt": 30},
            },
        )

    def test_legacy_all_noise_row_stays_fail_no_diluted_pass(self, tmp_path):
        md, summary = _run(tmp_path, [_row(1, 1.004), _row(8, 1.003)])
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in md

    def test_fully_qualifying_row_ships_pass_diluted(self, tmp_path):
        rows = [self._diluted_row(1), self._diluted_row(8)]
        md, summary = _run(tmp_path, rows)
        assert summary["e2e_gate"]["track_verdict"] == "PASS"
        assert summary["e2e_gate"]["diluted_pass"]["eligible"] is True
        assert "Ships as PASS (diluted)" in md


class TestSignificanceFloor:
    def test_1_0001x_non_significant_result_is_not_pass(self, tmp_path):
        _, summary = _run(tmp_path, [_row(1, 1.0001, significant=False)])
        verdict = summary["e2e_gate"]["per_bs_verdicts"][0]
        assert verdict["verdict"] == "NOISE"
        assert verdict["significant"] is False
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

