#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the DILUTED_PASS ship path.

Exercises sweep-side phase Welch attachment, report-side fail-closed
classification, end-to-end rendering, and legacy golden fixtures. Schema/state
round-trip coverage is an explicit Task-2 dependency at the end of this file.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_vllm_bench_latency_sweep as sweep_mod

_welch_significance = sweep_mod._welch_significance
_row_phase_significance = sweep_mod._row_phase_significance
_GEN = _SCRIPTS_DIR / "generate_validation_report.py"  # subprocess end-to-end + byte-diff classes

import generate_validation_report as gen_mod  # noqa: E402

_classify_diluted_pass = gen_mod._classify_diluted_pass

# Byte-compat golden fixtures (Step 7a).
_FIXDIR = Path(__file__).resolve().parent / "fixtures" / "diluted_pass"


# --- End-to-end fixture helpers (subprocess into generate_validation_report.py) ---

def _qual_row(bs: int, *, b_pf: float = 0.2, b_dc: float = 0.8,
              decode_win_pct: float = 2.5, prefill_win_pct: float = 0.0,
              decode_significant: bool = True, n: int = 30,
              prefill_significant: bool | None = False) -> dict:
    """Build a row whose phase decomposition is a diluted decode win.

    decode improves `decode_win_pct`% at baseline decode-share
    b_dc/(b_pf+b_dc); prefill improves `prefill_win_pct`% (0 = unchanged).
    speedup/improvement_pct are derived from the avg_s the phases imply, so
    amdahl-consistency holds by construction. phase_significance is attached
    exactly as the sweep would emit it (Step 2)."""
    o_pf = b_pf * (1.0 - prefill_win_pct / 100.0)
    o_dc = b_dc * (1.0 - decode_win_pct / 100.0)
    b_avg = b_pf + b_dc
    o_avg = o_pf + o_dc
    speedup = b_avg / o_avg
    row = {
        "batch_size": bs,
        "speedup": speedup,
        "improvement_pct": (1.0 - o_avg / b_avg) * 100.0,
        "baseline": {"avg_s": b_avg, "prefill_avg_s": b_pf, "decode_avg_s": b_dc,
                     "decode_share_of_e2e": b_dc / (b_pf + b_dc)},
        "opt": {"avg_s": o_avg, "prefill_avg_s": o_pf, "decode_avg_s": o_dc,
                "decode_share_of_e2e": o_dc / (o_pf + o_dc)},
    }
    phase_sig: dict = {}
    if decode_significant is not None:
        phase_sig["decode"] = {"significant": bool(decode_significant),
                               "n_baseline": n, "n_opt": n}
    if prefill_significant is not None:
        phase_sig["prefill"] = {"significant": bool(prefill_significant),
                                "n_baseline": n, "n_opt": n}
    if phase_sig:
        row["phase_significance"] = phase_sig
    return row


def _run_report(tmp_path: Path, rows: list, *, gating: dict | None = None) -> tuple:
    """Run generate_validation_report.py against a fabricated sweep JSON.

    When `gating` is provided, a target.json is written so min_e2e_improvement_pct
    can exceed 2*noise (the window in which DILUTED_PASS is even reachable)."""
    artifact = tmp_path / "artifact"
    e2e_dir = artifact / "e2e_latency"
    e2e_dir.mkdir(parents=True)
    payload = {
        "model_id": "test/model",
        "tp": 1,
        "max_model_len": 4096,
        "workload": {"input_len": 1024, "output_len": 128, "num_iters": 30},
        "results": rows,
        "bench": {"baseline_label": "baseline", "opt_label": "opt"},
    }
    (e2e_dir / "e2e_latency_results.json").write_text(json.dumps(payload))
    if gating is not None:
        (artifact / "target.json").write_text(json.dumps({"gating": gating}))
    result = subprocess.run(
        [sys.executable, str(_GEN), "--artifact-dir", str(artifact)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"report generation failed: {result.stderr}"
    md = (artifact / "validation_results.md").read_text()
    summary = json.loads((artifact / "validation_summary.json").read_text())
    return md, summary


# Campaign gating that opens the DILUTED_PASS window: min_e2e floor (3%) is
# above the decode e2e-equivalent floor (2*noise = 1%), so a diluted decode
# win lands as NOISE on e2e while clearing the decode floor.
_DILUTED_GATING = {"noise_tolerance_pct": 0.5, "catastrophic_regression_pct": 5.0,
                   "min_e2e_improvement_pct": 3.0}


class TestWelchReuseForPhase:
    def test_welch_returns_none_for_missing_arrays(self):
        assert _welch_significance(None, None) is None
        assert _welch_significance([1.0], [2.0]) is None  # <2 samples each side

    def test_welch_shape_matches_e2e_significance_shape(self):
        b = [1.0, 1.1, 0.9, 1.05, 0.95]
        o = [0.5, 0.55, 0.45, 0.52, 0.48]
        sig = _welch_significance(b, o)
        assert sig is not None
        assert set(sig) >= {"method", "n_baseline", "n_opt", "t_stat", "significant"}
        assert sig["n_baseline"] == 5 and sig["n_opt"] == 5


class TestRowPhaseSignificanceAttachment:
    """Exercises the SWEEP-side helper _row_phase_significance that main()'s
    row-assembly loop calls — the field generate_validation_report.py later
    consumes."""

    def _arrays(self, base, opt, n=30):
        # n length-matched draws around base/opt means (deterministic, no RNG).
        b = [base + (i % 3 - 1) * 0.001 for i in range(n)]
        o = [opt + (i % 3 - 1) * 0.001 for i in range(n)]
        return b, o

    def test_both_phases_present_when_both_significant(self):
        pf_b, pf_o = self._arrays(0.10, 0.05)  # clear prefill delta
        dc_b, dc_o = self._arrays(0.90, 0.45)  # clear decode delta
        baseline_raw = {"prefill_iter_means_s": pf_b, "decode_iter_means_s": dc_b}
        opt_raw = {"prefill_iter_means_s": pf_o, "decode_iter_means_s": dc_o}
        phase_sig = _row_phase_significance(baseline_raw, opt_raw)
        assert set(phase_sig) == {"prefill", "decode"}
        assert phase_sig["decode"]["n_baseline"] == 30 and phase_sig["decode"]["n_opt"] == 30
        assert phase_sig["decode"]["significant"] is True

    def test_decode_only_when_prefill_arrays_absent(self):
        dc_b, dc_o = self._arrays(0.90, 0.45)
        baseline_raw = {"decode_iter_means_s": dc_b}  # no prefill array
        opt_raw = {"decode_iter_means_s": dc_o}
        phase_sig = _row_phase_significance(baseline_raw, opt_raw)
        assert set(phase_sig) == {"decode"}  # prefill omitted, not None

    def test_empty_dict_on_legacy_raw_json(self):
        # Legacy/Tier-B/C: no *_iter_means_s keys at all -> {} -> caller omits the key.
        assert _row_phase_significance({"prefill_avg_s": 0.1}, {"prefill_avg_s": 0.1}) == {}
        assert _row_phase_significance(None, None) == {}

    def test_new_row_gets_key_only_when_nonempty(self):
        # Mirrors the `if phase_sig: new_row["phase_significance"] = phase_sig` guard.
        new_row = {}
        phase_sig = _row_phase_significance({"prefill_avg_s": 0.1}, {"prefill_avg_s": 0.1})
        if phase_sig:
            new_row["phase_significance"] = phase_sig
        assert "phase_significance" not in new_row


class TestClassifyDilutedPassUnit:
    """Direct unit tests on _classify_diluted_pass — one violated condition each.

    noise_tol_pct=0.5 throughout, so decode_floor_pct == 1.0. A qualifying row:
    decode_improvement_pct=2.5 * decode_share_of_e2e=0.8 == 2.0 e2e-equiv (>=1.0),
    prefill unchanged, amdahl_consistent True, speedup 1.02, decode significant,
    n_baseline/n_opt == 30.
    """

    def _pbv(self, bs, speedup=1.02, *, decode_sig=True, n=30,
             prefill_sig=False, drop_decode=False, drop_prefill=True):
        entry = {"batch_size": bs, "speedup": speedup, "verdict": "NOISE"}
        phase_sig = {}
        if not drop_decode:
            phase_sig["decode"] = {"significant": decode_sig,
                                   "n_baseline": n, "n_opt": n}
        if not drop_prefill:
            phase_sig["prefill"] = {"significant": prefill_sig,
                                    "n_baseline": n, "n_opt": n}
        if phase_sig:
            entry["phase_significance"] = phase_sig
        return entry

    def _pr(self, bs, *, dc_imp=2.5, share=0.8, pf_imp=0.0, amdahl=True):
        return {
            "batch_size": bs,
            "improvement_pct": share * dc_imp + (1.0 - share) * pf_imp,
            "phase": {"decode_improvement_pct": dc_imp,
                      "decode_share_of_e2e": share,
                      "prefill_improvement_pct": pf_imp},
            "flags": {"amdahl_consistent": amdahl},
        }

    def _good_verdicts_and_phase_rows(self, bss=(1, 8, 16)):
        pbv = [self._pbv(bs) for bs in bss]
        pr = [self._pr(bs) for bs in bss]
        return pbv, pr

    def test_all_conditions_met_returns_eligible(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        result = _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5)
        assert result is not None and result["eligible"] is True
        assert result["candidate_bs"] == [1, 8, 16]
        assert result["decode_floor_pct"] == 1.0
        assert len(result["per_bs"]) == 3

    def test_condition0_sample_floor_violated_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pbv[1]["phase_significance"]["decode"]["n_baseline"] = 10  # below 30 floor
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition1_decode_not_significant_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pbv[0]["phase_significance"]["decode"]["significant"] = False
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition1_missing_decode_significance_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pbv[2] = self._pbv(16, drop_decode=True)  # no decode Welch at this BS
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition2_decode_floor_not_cleared_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        # e2e-equiv = dc_imp*share; make it just under 1.0 (2*0.5): 1.2*0.8=0.96
        pr[1] = self._pr(8, dc_imp=1.2, share=0.8)
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition3_prefill_regressed_and_significant_vetoes(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr[0] = self._pr(1, pf_imp=-2.0)                     # prefill regressed 2%
        pbv[0] = self._pbv(1, prefill_sig=True, drop_prefill=False)  # significant
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition3_prefill_regressed_but_not_significant_does_not_veto(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr[0] = self._pr(1, pf_imp=-2.0)                     # prefill regressed 2%
        pbv[0] = self._pbv(1, prefill_sig=False, drop_prefill=False)  # NOT significant
        result = _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5)
        assert result is not None and result["eligible"] is True

    def test_condition3_prefill_missing_significance_still_vetoes(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr[0] = self._pr(1, pf_imp=-2.0)     # prefill regressed, no prefill Welch key
        # default _pbv drops prefill significance -> pf_significant is None -> vetoes
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition4_net_regression_at_any_bs_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pbv[2]["speedup"] = 0.998            # < 1.0 net regression
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition5_amdahl_inconsistent_disqualifies(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr[1]["flags"]["amdahl_consistent"] = False
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_condition5_amdahl_none_fails_closed(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr[1]["flags"]["amdahl_consistent"] = None
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None

    def test_incomplete_phase_coverage_disqualifies_whole_track(self):
        pbv, pr = self._good_verdicts_and_phase_rows()
        pr.pop()  # len(phase_rows) != len(per_bs_verdicts)
        assert _classify_diluted_pass(pbv, pr, noise_tol_pct=0.5) is None


class TestDilutedPassEndToEnd:
    """subprocess into generate_validation_report.py with a full raw+row fixture."""

    def test_qualifying_track_ships_as_pass_diluted(self, tmp_path):
        rows = [_qual_row(1), _qual_row(8), _qual_row(16)]
        md, summary = _run_report(tmp_path, rows, gating=_DILUTED_GATING)
        assert summary["e2e_gate"]["track_verdict"] == "PASS"
        assert summary["e2e_gate"]["diluted_pass"]["eligible"] is True
        assert summary["e2e_gate"]["diluted_pass"]["candidate_bs"] == [1, 8, 16]
        assert "Ships as PASS (diluted)" in md

    def test_non_qualifying_all_noise_track_stays_fail(self, tmp_path):
        # decode NOT significant at one BS -> disqualify whole track.
        rows = [_qual_row(1), _qual_row(8, decode_significant=False), _qual_row(16)]
        md, summary = _run_report(tmp_path, rows, gating=_DILUTED_GATING)
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in md

    def test_track_that_already_passes_never_enters_diluted_path(self, tmp_path):
        # Real e2e win (>=3% floor) -> PASS via the ladder; diluted path must
        # never be invoked even though decode significance is present.
        rows = [_qual_row(1, decode_win_pct=6.0), _qual_row(8, decode_win_pct=6.0)]
        md, summary = _run_report(tmp_path, rows, gating=_DILUTED_GATING)
        assert summary["e2e_gate"]["track_verdict"] == "PASS"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in md

    def test_legacy_row_without_phase_significance_key_stays_fail_byte_compat(self, tmp_path):
        # Scalar phase fields present but no phase_significance -> condition 1
        # fails on first row -> track stays FAIL, no diluted marker.
        row = _qual_row(1, decode_significant=None, prefill_significant=None)
        assert "phase_significance" not in row
        md, summary = _run_report(tmp_path, [row], gating=_DILUTED_GATING)
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in md

    @staticmethod
    def _track_evidence(*, correctness_status: str = "PASS") -> dict:
        return {
            "schema_version": 2,
            "track_id": "op007",
            "baseline_source": {"kind": "stage1", "citation": "round-2 baseline"},
            "correctness": {
                "status": correctness_status,
                "method": "torch.allclose",
                "atol": 1e-5,
                "rtol": 1e-5,
                "max_abs_diff": 0.0,
                "nan_inf_check": True,
                "graph_replay_check": True,
            },
            "kernel_bench": {
                "status": "PASS",
                "weighted_speedup": 1.03,
                "measured_under_cuda_graphs": True,
                "buckets": [{"batch_size": 1}],
            },
            "e2e": {
                "status": "PASS",
                "run_purpose": "official",
                "baseline_avg_s": 1.0,
                "optimized_avg_s": 0.99,
                "speedup": 1.01,
                "improvement_pct": 1.0,
                "admissibility": {"status": "PASS", "issues": []},
                "fastpath_proof": {"status": "PASS", "hits": 1},
            },
            "kill_criteria": {"mechanism_active": {"status": "PASS"}},
            "amdahl": {},
        }

    @staticmethod
    def _write_track_fixture(tmp_path: Path, *, correctness_status: str = "PASS") -> tuple[Path, Path]:
        artifact = tmp_path / "artifact"
        track_dir = artifact / "rounds" / "2" / "tracks" / "op007"
        sweep_dir = artifact / "rounds" / "2" / "sweeps" / "opt" / "op007"
        track_dir.mkdir(parents=True)
        sweep_dir.mkdir(parents=True)
        (artifact / "state.json").write_text(
            json.dumps({"campaign": {"current_round": 2}}), encoding="utf-8"
        )
        (artifact / "target.json").write_text(
            json.dumps({"gating": _DILUTED_GATING}), encoding="utf-8"
        )
        evidence = TestDilutedPassEndToEnd._track_evidence(
            correctness_status=correctness_status
        )
        (track_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        payload = {
            "model_id": "test/model",
            "tp": 1,
            "max_model_len": 4096,
            "workload": {"input_len": 1024, "output_len": 128, "num_iters": 30},
            "results": [_qual_row(1), _qual_row(8)],
            "bench": {"baseline_label": "baseline", "opt_label": "opt"},
        }
        (sweep_dir / "e2e_latency_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return artifact, track_dir

    def test_canonical_track_command_writes_track_local_diluted_summary(self, tmp_path):
        _artifact, track_dir = self._write_track_fixture(tmp_path)
        evidence_path = track_dir / "evidence.json"
        output_path = track_dir / "validation_results.md"

        result = subprocess.run(
            [sys.executable, str(_GEN), str(evidence_path), "--output", str(output_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"report generation failed: {result.stderr}"
        summary = json.loads((track_dir / "validation_summary.json").read_text())
        assert summary["e2e_gate"]["track_verdict"] == "PASS"
        assert summary["e2e_gate"]["diluted_pass"]["eligible"] is True
        assert "Ships as PASS (diluted)" in output_path.read_text()
        assert not (_artifact / "validation_summary.json").exists()

    def test_track_local_diluted_candidate_does_not_override_failed_correctness(self, tmp_path):
        _artifact, track_dir = self._write_track_fixture(
            tmp_path, correctness_status="FAIL"
        )

        result = subprocess.run(
            [sys.executable, str(_GEN), str(track_dir / "evidence.json")],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"report generation failed: {result.stderr}"
        summary = json.loads((track_dir / "validation_summary.json").read_text())
        assert summary["e2e_gate"]["track_verdict"] == "FAIL"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in (track_dir / "validation_results.md").read_text()


def _run_report_on_fixture(tmp_path: Path, fixture_name: str) -> tuple:
    """Run generate_validation_report.py against a checked-in golden INPUT
    fixture (tests/fixtures/diluted_pass/<fixture_name>.json) and return the
    (validation_results.md, validation_summary.json) text. No target.json is
    written — these fixtures use the documented default gating (0.5% noise /
    0.5% min-floor), exactly as they were captured pre-DILUTED_PASS."""
    artifact = tmp_path / "artifact"
    e2e_dir = artifact / "e2e_latency"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "e2e_latency_results.json").write_text(
        (_FIXDIR / f"{fixture_name}.json").read_text()
    )
    result = subprocess.run(
        [sys.executable, str(_GEN), "--artifact-dir", str(artifact)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"report generation failed: {result.stderr}"
    md = (artifact / "validation_results.md").read_text()
    summary = (artifact / "validation_summary.json").read_text()
    return md, summary


class TestLegacyByteIdentical:
    """Pre-change golden fixtures must produce byte-identical
    validation_results.md / validation_summary.json (modulo the volatile
    generated_at timestamp and the per-run absolute artifact_dir path).

    The goldens under fixtures/diluted_pass/ were captured from the
    pre-DILUTED_PASS revision of generate_validation_report.py (the
    post-phase-triage script with only the DILUTED_PASS additions removed) and
    verified byte-identical to the current script's output on these legacy
    inputs — a true regression guard, not a tautology. See the module/step
    notes: git HEAD predates BOTH the phase-triage layer and DILUTED_PASS, so
    the authoritative back-compat baseline is post-phase-triage, not HEAD."""

    @staticmethod
    def _norm(s: str) -> str:
        # Normalize the only run-varying fields: the generated_at timestamp
        # (present in the summary JSON and the md "Generated: ... (UTC)" line)
        # and the absolute artifact_dir / e2e_json paths (vary per tmp_path).
        s = re.sub(r'"generated_at":\s*"[^"]*"', '"generated_at": "GENERATED_AT"', s)
        s = re.sub(r'Generated: [^\n]*\(UTC\)', 'Generated: GENERATED_AT (UTC)', s)
        s = re.sub(r'"artifact_dir":\s*"[^"]*"', '"artifact_dir": "ARTIFACT_DIR"', s)
        s = re.sub(
            r'"e2e_json":\s*"[^"]*"',
            '"e2e_json": "ARTIFACT_DIR/e2e_latency/e2e_latency_results.json"', s,
        )
        return s

    def _assert_byte_identical(self, tmp_path: Path, fixture_name: str) -> tuple:
        md, summary = _run_report_on_fixture(tmp_path, fixture_name)
        golden_md = (_FIXDIR / f"{fixture_name}.golden.md").read_text()
        golden_summary = (_FIXDIR / f"{fixture_name}.golden.summary.json").read_text()
        assert self._norm(md) == golden_md, (
            f"{fixture_name}: validation_results.md diverged from golden — "
            "DILUTED_PASS additions changed the legacy render path"
        )
        assert self._norm(summary) == golden_summary, (
            f"{fixture_name}: validation_summary.json diverged from golden"
        )
        return md, json.loads(summary)

    def test_legacy_no_phase_byte_identical(self, tmp_path):
        # No prefill/decode fields at all -> no phase section, no phase keys.
        md, summary = self._assert_byte_identical(tmp_path, "legacy_no_phase")
        assert "Phase decomposition" not in md
        assert "phase" not in summary["e2e_gate"]["per_bs_verdicts"][0]
        assert "phase_triage" not in summary["e2e_gate"]

    def test_legacy_scalar_phase_byte_identical(self, tmp_path):
        # Scalar phase fields present (phase table renders) but no
        # *_iter_means_s / phase_significance -> no phase Welch, no DILUTED_PASS.
        md, summary = self._assert_byte_identical(tmp_path, "legacy_scalar_phase")
        assert "Phase decomposition" in md          # phase path IS exercised
        assert "Ships as PASS (diluted)" not in md
        assert "diluted_pass" not in summary["e2e_gate"]
        for v in summary["e2e_gate"]["per_bs_verdicts"]:
            assert "phase_significance" not in v

    def test_track_already_pass_never_touches_diluted_path_byte_compat(self, tmp_path):
        # A normal significant-win PASS track WITH phase_significance present
        # (would qualify for DILUTED_PASS if it were reached). The gating-order
        # precondition (track_verdict == "FAIL") must keep the diluted path
        # unreachable -> no "diluted_pass" key at all.
        row = _qual_row(8, decode_win_pct=6.0, prefill_win_pct=0.0)
        row["significance"] = {"significant": True}
        md, summary = _run_report(tmp_path, [row], gating=_DILUTED_GATING)
        assert summary["e2e_gate"]["track_verdict"] == "PASS"
        assert "diluted_pass" not in summary["e2e_gate"]
        assert "Ships as PASS (diluted)" not in md

def test_schema_diluted_properties_dependency_is_explicit():
    schema = json.loads(
        (Path(__file__).resolve().parents[3] / "schemas" / "state.schema.json").read_text(
            encoding="utf-8"
        )
    )
    round_properties = schema["properties"]["campaign"]["properties"]["rounds"]["items"][
        "properties"
    ]
    track_schema = round_properties["parallel_tracks"]["properties"]["tracks"][
        "additionalProperties"
    ]

    assert track_schema["properties"]["diluted"]["type"] == ["boolean", "null"]
    assert {
        "if": {"properties": {"diluted": {"const": True}}, "required": ["diluted"]},
        "then": {"properties": {"status": {"const": "PASS"}}, "required": ["status"]},
    } in track_schema["allOf"]
    assert "diluted_tracks" in round_properties
