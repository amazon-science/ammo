#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the --baseline-from-arm sweep flag.

`--baseline-from` imports the comparator by LITERAL filename
`{prefix}_{tag}.json`, so the comparator identity is the filename prefix, not
the directory. Before this flag, a round >= 2 integration sweep pointed at a
pre-SHIP directory silently imported that directory's bf16 baseline arm and the
cumulative delta was read as incremental.

Covers:
- default is 'baseline' (byte-identical to the pre-flag behavior)
- --baseline-from-arm opt imports the 'opt' filename prefix instead
- an unknown arm is rejected by argparse
- a non-default arm that the run cannot honor is disclosed as IGNORED
- a honored non-default arm names the imported comparator in the .md report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

BASELINE_AVG = 0.050
IMPORTED_OPT_AVG = 0.030
CHILD_OPT_AVG = 0.020


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str]):
    """Build the sweep argparser and parse argv (excluding program name)."""
    from run_vllm_bench_latency_sweep import main
    import argparse as _ap

    captured: Dict[str, Any] = {}

    def intercept(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(_ap.ArgumentParser, "parse_args", intercept):
        try:
            main()
        except SystemExit:
            pass

    parser = captured.get("parser")
    assert parser is not None, "Could not capture parser"
    return parser.parse_args(argv)


class TestBaselineFromArmArg:
    def test_default_is_baseline(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x"])
        assert ns.baseline_from_arm == "baseline"

    def test_opt_is_accepted(self):
        ns = _parse_args(
            ["--artifact-dir", "/tmp/x", "--baseline-from-arm", "opt"]
        )
        assert ns.baseline_from_arm == "opt"

    def test_unknown_arm_is_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(
                ["--artifact-dir", "/tmp/x", "--baseline-from-arm", "mainline"]
            )


# ---------------------------------------------------------------------------
# Import behavior: which filename prefix is read
# ---------------------------------------------------------------------------


def _bench_json(avg_latency: float) -> Dict[str, Any]:
    return {
        "avg_latency": avg_latency,
        "latencies": [avg_latency, avg_latency],
        "percentiles": {
            "10": avg_latency,
            "50": avg_latency,
            "90": avg_latency,
            "99": avg_latency,
        },
    }


def _runner_json() -> Dict[str, Any]:
    return {
        "ok": True,
        "label": "test",
        "batch_size": 1,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:01:00Z",
        "duration_s": 60.0,
    }


def _write_target_json(artifact_dir: Path) -> None:
    target = {
        "artifact_dir": str(artifact_dir),
        "target": {
            "model_id": "test-model/test",
            "dtype": "fp16",
            "tp": 1,
            "ep": 1,
            "max_model_len": 4096,
        },
        "workload": {
            "input_len": 64,
            "output_len": 128,
            "batch_sizes": [1],
            "num_iters": 2,
        },
        "bench": {
            "runner": "vllm_bench_latency",
            "vllm_cmd": "vllm",
            "extra_args": [],
            "baseline_extra_args": [],
            "opt_extra_args": ["--some-opt-flag"],
            "baseline_env": {},
            "opt_env": {"ENABLE_OPT": "1"},
            "baseline_label": "baseline",
            "opt_label": "opt",
            "fastpath_evidence": {
                "baseline": {"require_patterns": [], "forbid_patterns": []},
                "opt": {"require_patterns": [], "forbid_patterns": []},
            },
        },
    }
    (artifact_dir / "target.json").write_text(json.dumps(target, indent=2))


def _write_stage1_dir(artifact_dir: Path) -> Path:
    """A pre-SHIP sweep dir: it holds BOTH arms, exactly as the live bug did."""
    stage1 = artifact_dir / "e2e_baseline"
    json_dir = stage1 / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for prefix, avg in (
        ("baseline", BASELINE_AVG),
        ("opt", IMPORTED_OPT_AVG),
    ):
        (json_dir / f"{prefix}_bs1.json").write_text(json.dumps(_bench_json(avg)))
        (json_dir / f"{prefix}_bs1.runner.json").write_text(
            json.dumps(_runner_json())
        )
    return stage1


def _child_flag(cmd: List[str], flag: str) -> str | None:
    for i, arg in enumerate(cmd):
        if arg == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _child_side_effect(cmd, *, env, cwd, timeout_s, log_path, heartbeat_s,
                       status_path=None):
    """Write only the opt child's artifacts; the comparator must come from the
    import, so a mistaken re-run of a baseline arm shows up as a missing row."""
    label = _child_flag(cmd, "--_child-label")
    out_root = _child_flag(cmd, "--_out-root")
    if label and out_root:
        json_dir = Path(out_root) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"{label}_bs1.json").write_text(
            json.dumps(_bench_json(CHILD_OPT_AVG))
        )
        (json_dir / f"{label}_bs1.runner.json").write_text(json.dumps(_runner_json()))
    return {"ok": True, "returncode": 0}


def _run_sweep(
    tmp_path: Path, extra_args: List[str], labels: str = "opt"
) -> Tuple[Dict[str, Any], str]:
    """Run one sweep and return (results JSON, results .md text)."""
    import run_vllm_bench_latency_sweep as sweep

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_target_json(artifact_dir)
    stage1 = _write_stage1_dir(artifact_dir)

    argv = [
        "run_vllm_bench_latency_sweep.py",
        "--artifact-dir", str(artifact_dir),
        "--labels", labels,
        "--overwrite",
        "--baseline-from", str(stage1),
        # No GSM8K corpus is bundled next to the canonical script copy, and the
        # import path under test never reads prompts.
        "--dummy-prompt-source", "random",
    ] + extra_args

    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(sweep, "_run_cmd_streaming", side_effect=_child_side_effect),
        mock.patch("shutil.which", return_value="/usr/bin/vllm"),
    ):
        try:
            sweep.main()
        except SystemExit as exc:
            if exc.code:
                raise

    results = sorted(artifact_dir.rglob("e2e_latency_results.json"))
    assert results, "sweep wrote no e2e_latency_results.json"
    md_path = results[0].parent / "e2e_latency_results.md"
    assert md_path.exists(), "sweep wrote no e2e_latency_results.md"
    return json.loads(results[0].read_text()), md_path.read_text()


def _run_gate_sweep(tmp_path: Path, extra_args: List[str]) -> Dict[str, Any]:
    return _run_sweep(tmp_path, extra_args)[0]


class TestBaselineFromArmImport:
    def test_default_imports_the_baseline_prefix(self, tmp_path):
        """Absent the flag, the comparator is the {baseline_label}_ prefix."""
        results = _run_gate_sweep(tmp_path, [])
        row = results["results"][0]
        assert row["baseline"]["avg_s"] == pytest.approx(BASELINE_AVG)
        # Backward compat: the default run's header gains no new key.
        assert "baseline_source_arm" not in results

    def test_arm_opt_imports_the_opt_prefix(self, tmp_path):
        """--baseline-from-arm opt reads the promoted arm's files instead."""
        results = _run_gate_sweep(tmp_path, ["--baseline-from-arm", "opt"])
        row = results["results"][0]
        assert row["baseline"]["avg_s"] == pytest.approx(IMPORTED_OPT_AVG)
        assert row["baseline"]["avg_s"] != pytest.approx(BASELINE_AVG)
        # A non-default comparator arm is disclosed in the artifact.
        assert results["baseline_source_arm"] == "opt"

    def test_imported_rows_still_land_under_baseline_label(self, tmp_path):
        """The import writes under baseline_label whichever arm it read, so the
        results collection and every downstream gate keep reading one key."""
        results = _run_gate_sweep(tmp_path, ["--baseline-from-arm", "opt"])
        row = results["results"][0]
        assert row["opt"]["avg_s"] == pytest.approx(CHILD_OPT_AVG)
        assert results["bench"]["baseline_label"] == "baseline"


# ---------------------------------------------------------------------------
# Disclosure: the flag is never silently dead, and never silently honored
# ---------------------------------------------------------------------------


class TestBaselineFromArmDisclosure:
    """The .md report is what an auditor reads. Both the ignored case and the
    honored case must be visible there, not only in the JSON header."""

    def test_honored_arm_is_named_in_the_md_report(self, tmp_path):
        results, md = _run_sweep(tmp_path, ["--baseline-from-arm", "opt"])
        warnings = results.get("warnings") or []
        provenance = [w for w in warnings if "comparator imported from arm" in w]
        assert provenance, f"no comparator-provenance warning in {warnings}"
        assert "'opt'" in provenance[0]
        assert "--baseline-from-arm" in provenance[0]
        # Visible to a reader of the .md alone.
        assert "## Warnings" in md
        assert "comparator imported from arm 'opt'" in md

    def test_ignored_arm_is_disclosed_when_the_import_is_gated_off(self, tmp_path):
        """--labels baseline,opt gates the import block off, so the flag cannot
        act. The run must say so instead of implying a promoted comparator."""
        results, md = _run_sweep(
            tmp_path, ["--baseline-from-arm", "opt"], labels="baseline,opt"
        )
        warnings = results.get("warnings") or []
        ignored = [w for w in warnings if "was IGNORED" in w]
        assert ignored, f"no ignored-flag warning in {warnings}"
        assert "--baseline-from-arm opt" in ignored[0]
        assert "## Warnings" in md
        assert "--baseline-from-arm opt was IGNORED" in md
        # The gated-off run imports nothing, so it must not also claim provenance.
        assert not [w for w in warnings if "comparator imported from arm" in w]

    def test_default_run_gains_no_warning_and_no_warnings_section(self, tmp_path):
        """Backward compat: the default arm adds neither warning nor .md section."""
        results, md = _run_sweep(tmp_path, [])
        warnings = results.get("warnings") or []
        assert not [w for w in warnings if "--baseline-from-arm" in w]
        assert not [w for w in warnings if "comparator imported from arm" in w]
        assert "## Warnings" not in md
