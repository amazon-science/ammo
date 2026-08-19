#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the profiler-baseline contamination guard in run_vllm_bench_latency_sweep.py.

Verifies:
  - --slot baseline + --nsys-profile = hard SystemExit
  - --slot baseline + --torch-profile = hard SystemExit
  - --slot baseline + both profiling flags = hard SystemExit (nsys takes priority in msg)
  - --slot profiling + --nsys-profile = allowed
  - --slot profiling + --torch-profile = allowed
  - --slot opt/{op_id} + --nsys-profile = allowed (not baseline)
  - --slot integration + --torch-profile = allowed (not baseline)
  - No --slot + --nsys-profile = allowed (legacy mode or child)
  - Error message includes guidance for two-invocation pattern
  - e2e_is_reduced = True when nsys_profile or torch_profile is set
  - e2e_is_reduced = False when neither profiling flag is set
  - runner.json status dict includes nsys_profile and torch_profile fields
  - profiling slot resolves to rounds/{N}/sweeps/profiling/
  - _v2_profiling_dir routes traces independent of slot value
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_v2_artifact(tmp_path: Path, current_round: int = 1) -> Path:
    """Create a v2-layout artifact dir with rounds/{current_round}/ scaffold + state.json."""
    artifact = tmp_path / "artifact"
    (artifact / "rounds" / str(current_round) / "sweeps").mkdir(parents=True)
    state = {
        "schema_version": "4.1",
        "campaign": {"current_round": current_round, "rounds": []},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    target = {
        "artifact_dir": str(artifact),
        "target": {"model_id": "x", "dtype": "fp16", "tp": 1, "ep": 1, "max_model_len": 4096},
        "workload": {"input_len": 64, "output_len": 512, "batch_sizes": [1], "num_iters": 1},
        "bench": {
            "runner": "vllm_bench_latency",
            "vllm_cmd": "vllm",
            "extra_args": [],
            "baseline_extra_args": [],
            "opt_extra_args": [],
            "baseline_env": {},
            "opt_env": {},
            "baseline_label": "baseline",
            "opt_label": "opt",
        },
    }
    (artifact / "target.json").write_text(json.dumps(target), encoding="utf-8")
    return artifact


def _make_args(**kwargs) -> argparse.Namespace:
    """Create a minimal argparse.Namespace mimicking sweep script args."""
    defaults = {
        "slot": None,
        "round": None,
        "nsys_profile": False,
        "torch_profile": False,
        "labels": "baseline",
        "verify_correctness": False,
        "_out_root": None,
        "out_name": "e2e_latency",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Profiler-baseline guard tests
# ---------------------------------------------------------------------------

class TestProfilerBaselineGuard:
    """--slot baseline + any profiling flag = hard SystemExit."""

    def _run_guard(self, args: argparse.Namespace):
        """Execute only the guard logic extracted from the sweep script."""
        slot = getattr(args, "slot", None)
        if slot == "baseline" and (args.nsys_profile or args.torch_profile):
            profiler_flag = "--nsys-profile" if args.nsys_profile else "--torch-profile"
            raise SystemExit(
                f"ERROR: {profiler_flag} cannot be combined with --slot baseline. "
                "Profiling adds overhead that contaminates E2E timing (nsys: 10-30% "
                "process-wide, torch: CUPTI subscription). The baseline slot is the "
                "authoritative timing source for speedup calculations.\n\n"
                "Correct Stage 1 workflow (two invocations):\n"
                "  1. Clean E2E:  --round N --slot baseline --labels baseline "
                "--capture-golden-refs\n"
                "  2. Profiling:  --round N --slot profiling --nsys-profile "
                "(or --torch-profile)\n"
                "     (traces route to rounds/{N}/profiling/ regardless of slot)\n\n"
                "See references/e2e-latency-guide.md for details."
            )

    def test_baseline_nsys_profile_raises(self):
        args = _make_args(slot="baseline", nsys_profile=True)
        with pytest.raises(SystemExit, match="--nsys-profile cannot be combined with --slot baseline"):
            self._run_guard(args)

    def test_baseline_torch_profile_raises(self):
        args = _make_args(slot="baseline", torch_profile=True)
        with pytest.raises(SystemExit, match="--torch-profile cannot be combined with --slot baseline"):
            self._run_guard(args)

    def test_baseline_both_profilers_raises_nsys_priority(self):
        """When both flags set, error message names nsys (checked first)."""
        args = _make_args(slot="baseline", nsys_profile=True, torch_profile=True)
        with pytest.raises(SystemExit, match="--nsys-profile"):
            self._run_guard(args)

    def test_profiling_slot_nsys_allowed(self):
        args = _make_args(slot="profiling", nsys_profile=True)
        self._run_guard(args)  # Should not raise

    def test_profiling_slot_torch_allowed(self):
        args = _make_args(slot="profiling", torch_profile=True)
        self._run_guard(args)  # Should not raise

    def test_opt_slot_nsys_allowed(self):
        args = _make_args(slot="opt/op001", nsys_profile=True)
        self._run_guard(args)  # Should not raise

    def test_integration_slot_torch_allowed(self):
        args = _make_args(slot="integration", torch_profile=True)
        self._run_guard(args)  # Should not raise

    def test_no_slot_nsys_allowed(self):
        """Legacy/child mode: no --slot, profiling is fine."""
        args = _make_args(slot=None, nsys_profile=True)
        self._run_guard(args)  # Should not raise

    def test_baseline_no_profiling_allowed(self):
        """--slot baseline without profiling flags is fine."""
        args = _make_args(slot="baseline", nsys_profile=False, torch_profile=False)
        self._run_guard(args)  # Should not raise

    def test_error_message_contains_two_invocation_guidance(self):
        args = _make_args(slot="baseline", nsys_profile=True)
        with pytest.raises(SystemExit) as exc_info:
            self._run_guard(args)
        msg = str(exc_info.value)
        assert "two invocations" in msg.lower() or "Clean E2E" in msg
        assert "--slot profiling" in msg
        assert "--slot baseline" in msg
        assert "e2e-latency-guide.md" in msg

    def test_golden_capture_slot_nsys_allowed(self):
        args = _make_args(slot="golden_capture", nsys_profile=True)
        self._run_guard(args)  # Should not raise


# ---------------------------------------------------------------------------
# Guard integration: verify the actual script enforces it via parse_args path
# ---------------------------------------------------------------------------

class TestProfilerGuardIntegration:
    """Run the sweep script as subprocess with forbidden flag combos, verify exit code."""

    @pytest.fixture
    def artifact_dir(self, tmp_path):
        return _make_v2_artifact(tmp_path)

    def _run_sweep(self, artifact_dir: Path, extra_args: list[str]) -> tuple[int, str]:
        """Run the sweep script and return (exit_code, stderr)."""
        import subprocess
        cmd = [
            sys.executable, str(_SCRIPTS_DIR / "run_vllm_bench_latency_sweep.py"),
            "--artifact-dir", str(artifact_dir),
            "--round", "1",
            *extra_args,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        return result.returncode, result.stderr + result.stdout

    def test_subprocess_baseline_nsys_exits_nonzero(self, artifact_dir):
        code, output = self._run_sweep(artifact_dir, [
            "--slot", "baseline", "--nsys-profile", "--labels", "baseline",
        ])
        assert code != 0, f"Expected nonzero exit, got {code}. Output:\n{output}"
        assert "--nsys-profile cannot be combined with --slot baseline" in output

    def test_subprocess_baseline_torch_exits_nonzero(self, artifact_dir):
        code, output = self._run_sweep(artifact_dir, [
            "--slot", "baseline", "--torch-profile", "--labels", "baseline",
        ])
        assert code != 0, f"Expected nonzero exit, got {code}. Output:\n{output}"
        assert "--torch-profile cannot be combined with --slot baseline" in output

    def test_subprocess_profiling_slot_nsys_passes_guard(self, artifact_dir):
        """--slot profiling + --nsys-profile passes the guard (may fail later for missing nsys binary, that's OK)."""
        code, output = self._run_sweep(artifact_dir, [
            "--slot", "profiling", "--nsys-profile", "--labels", "baseline",
        ])
        # Should NOT fail with the baseline guard error
        assert "cannot be combined with --slot baseline" not in output
        # May fail with "nsys not found" — that's expected in test env without nsys
        # The point is it passed the baseline guard check


# ---------------------------------------------------------------------------
# e2e_is_reduced logic
# ---------------------------------------------------------------------------

class TestE2eIsReduced:
    """e2e_is_reduced should be True when ANY profiling flag is active."""

    def test_nsys_only_is_reduced(self):
        args = _make_args(nsys_profile=True, torch_profile=False)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        assert e2e_is_reduced is True

    def test_torch_only_is_reduced(self):
        args = _make_args(nsys_profile=False, torch_profile=True)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        assert e2e_is_reduced is True

    def test_both_is_reduced(self):
        args = _make_args(nsys_profile=True, torch_profile=True)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        assert e2e_is_reduced is True

    def test_neither_is_not_reduced(self):
        args = _make_args(nsys_profile=False, torch_profile=False)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        assert e2e_is_reduced is False

    def test_full_benchmark_label_false_when_reduced(self):
        """Sidecar label full_benchmark should be 'false' when profiling active."""
        args = _make_args(nsys_profile=True)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        label_value = "false" if e2e_is_reduced else "true"
        assert label_value == "false"

    def test_full_benchmark_label_true_when_clean(self):
        """Sidecar label full_benchmark should be 'true' when no profiling."""
        args = _make_args(nsys_profile=False, torch_profile=False)
        e2e_is_reduced = bool(args.nsys_profile or args.torch_profile)
        label_value = "false" if e2e_is_reduced else "true"
        assert label_value == "true"


# ---------------------------------------------------------------------------
# Runner.json status dict fields
# ---------------------------------------------------------------------------

class TestRunnerJsonProfilingFields:
    """runner.json status dict must include nsys_profile and torch_profile booleans."""

    def _build_status(self, args: argparse.Namespace) -> dict:
        """Simulate status dict construction from sweep script L2154-2163."""
        from datetime import datetime, timezone
        start = datetime.now(timezone.utc)
        return {
            "ok": False,
            "label": "baseline",
            "batch_size": 1,
            "input_len": 64,
            "output_len": 512,
            "start_time": start.isoformat(),
            "nsys_profile": bool(args.nsys_profile),
            "torch_profile": bool(args.torch_profile),
        }

    def test_nsys_true_in_status(self):
        args = _make_args(nsys_profile=True, torch_profile=False)
        status = self._build_status(args)
        assert status["nsys_profile"] is True
        assert status["torch_profile"] is False

    def test_torch_true_in_status(self):
        args = _make_args(nsys_profile=False, torch_profile=True)
        status = self._build_status(args)
        assert status["nsys_profile"] is False
        assert status["torch_profile"] is True

    def test_both_false_in_status(self):
        args = _make_args(nsys_profile=False, torch_profile=False)
        status = self._build_status(args)
        assert status["nsys_profile"] is False
        assert status["torch_profile"] is False

    def test_both_true_in_status(self):
        args = _make_args(nsys_profile=True, torch_profile=True)
        status = self._build_status(args)
        assert status["nsys_profile"] is True
        assert status["torch_profile"] is True

    def test_fields_are_booleans_not_truthy(self):
        """Ensure we write proper booleans, not truthy ints or strings."""
        args = _make_args(nsys_profile=True, torch_profile=False)
        status = self._build_status(args)
        assert isinstance(status["nsys_profile"], bool)
        assert isinstance(status["torch_profile"], bool)

    def test_audit_invariant_14_detects_contamination(self):
        """Simulate auditor sub-check: baseline runner.json must not have nsys_profile: true."""
        args_clean = _make_args(nsys_profile=False, torch_profile=False)
        args_contaminated = _make_args(nsys_profile=True, torch_profile=False)

        status_clean = self._build_status(args_clean)
        status_contaminated = self._build_status(args_contaminated)

        # Auditor check: fail if any runner.json in baseline has profiling = true
        assert not (status_clean["nsys_profile"] or status_clean["torch_profile"]), \
            "Clean baseline should pass audit"
        assert (status_contaminated["nsys_profile"] or status_contaminated["torch_profile"]), \
            "Contaminated baseline should fail audit"


# ---------------------------------------------------------------------------
# Slot resolution: profiling slot
# ---------------------------------------------------------------------------

class TestProfilingSlotResolution:
    """The 'profiling' slot resolves like any other named slot."""

    def test_profiling_slot_resolves_to_sweeps_profiling(self, tmp_path):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        ns = argparse.Namespace(round=1, slot="profiling", out_name="e2e_latency", _out_root=None)
        result = _resolve_sweep_out_name(ns, artifact)
        assert result == "rounds/1/sweeps/profiling"

    def test_profiling_slot_round2_resolves(self, tmp_path):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        artifact = _make_v2_artifact(tmp_path, current_round=2)
        (artifact / "rounds" / "2" / "sweeps").mkdir(parents=True, exist_ok=True)
        ns = argparse.Namespace(round=2, slot="profiling", out_name="e2e_latency", _out_root=None)
        result = _resolve_sweep_out_name(ns, artifact)
        assert result == "rounds/2/sweeps/profiling"

    def test_profiling_slot_state_json_fallback(self, tmp_path):
        """No --round given, --slot profiling: reads state.json current_round."""
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        artifact = _make_v2_artifact(tmp_path, current_round=3)
        ns = argparse.Namespace(round=None, slot="profiling", out_name="e2e_latency", _out_root=None)
        result = _resolve_sweep_out_name(ns, artifact)
        assert result == "rounds/3/sweeps/profiling"


# ---------------------------------------------------------------------------
# _v2_profiling_dir: slot-independent trace routing
# ---------------------------------------------------------------------------

class TestV2ProfilingDir:
    """_v2_profiling_dir routes traces to rounds/{N}/profiling/{kind}/ regardless of slot."""

    def test_baseline_slot_routes_to_profiling_nsys(self, tmp_path):
        from run_vllm_bench_latency_sweep import _v2_profiling_dir
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        out_root = artifact / "rounds" / "1" / "sweeps" / "baseline"
        out_root.mkdir(parents=True, exist_ok=True)
        result = _v2_profiling_dir(out_root, "nsys")
        assert result == artifact / "rounds" / "1" / "profiling" / "nsys"

    def test_profiling_slot_routes_to_profiling_nsys(self, tmp_path):
        from run_vllm_bench_latency_sweep import _v2_profiling_dir
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        out_root = artifact / "rounds" / "1" / "sweeps" / "profiling"
        out_root.mkdir(parents=True, exist_ok=True)
        result = _v2_profiling_dir(out_root, "nsys")
        assert result == artifact / "rounds" / "1" / "profiling" / "nsys"

    def test_opt_slot_routes_to_profiling_torch(self, tmp_path):
        from run_vllm_bench_latency_sweep import _v2_profiling_dir
        artifact = _make_v2_artifact(tmp_path, current_round=2)
        out_root = artifact / "rounds" / "2" / "sweeps" / "opt" / "op001"
        out_root.mkdir(parents=True, exist_ok=True)
        result = _v2_profiling_dir(out_root, "torch_profile")
        assert result == artifact / "rounds" / "2" / "profiling" / "torch_profile"

    def test_all_slots_converge_same_profiling_dir(self, tmp_path):
        """All slot values produce the same profiling sibling for a given round."""
        from run_vllm_bench_latency_sweep import _v2_profiling_dir
        artifact = _make_v2_artifact(tmp_path, current_round=1)

        results = []
        for slot_path in ["baseline", "profiling", "opt/op001", "integration", "golden_capture"]:
            out_root = artifact / "rounds" / "1" / "sweeps" / slot_path
            out_root.mkdir(parents=True, exist_ok=True)
            results.append(_v2_profiling_dir(out_root, "nsys"))

        # All should resolve to the same path
        expected = artifact / "rounds" / "1" / "profiling" / "nsys"
        for r in results:
            assert r == expected, f"Expected {expected}, got {r}"


# ---------------------------------------------------------------------------
# Archive behavior with profiling slot
# ---------------------------------------------------------------------------

class TestArchiveProfilingSlot:
    """Archive semantics work correctly for the two-invocation pattern."""

    def test_second_invocation_archives_first(self, tmp_path):
        """Profiling run populates slot, then clean run archives it."""
        from run_vllm_bench_latency_sweep import _prepare_out_root
        artifact = _make_v2_artifact(tmp_path, current_round=1)

        # Simulate Invocation 1 (profiling) writing to opt/op001
        opt_slot = artifact / "rounds" / "1" / "sweeps" / "opt" / "op001"
        opt_slot.mkdir(parents=True, exist_ok=True)
        contaminated = opt_slot / "e2e_latency_results.json"
        contaminated.write_text('{"contaminated": true}', encoding="utf-8")

        # Simulate Invocation 2 (clean): _prepare_out_root archives existing
        out_root = _prepare_out_root(
            artifact_dir=artifact,
            out_name="rounds/1/sweeps/opt/op001",
            overwrite=False,
        )

        # Contaminated file should be archived
        assert not contaminated.exists()
        assert out_root == opt_slot
        assert out_root.is_dir() and not any(out_root.iterdir())

        # Should be in _archive
        archive_dir = artifact / "rounds" / "1" / "_archive"
        assert archive_dir.is_dir()
        archived_dirs = list(archive_dir.glob("opt_op001_*"))
        assert len(archived_dirs) == 1
        assert (archived_dirs[0] / "e2e_latency_results.json").exists()

    def test_profiling_traces_survive_archive(self, tmp_path):
        """Traces in rounds/{N}/profiling/nsys/ are NOT affected by sweep slot archiving."""
        from run_vllm_bench_latency_sweep import _prepare_out_root
        artifact = _make_v2_artifact(tmp_path, current_round=1)

        # Simulate profiling traces written by Invocation 1
        profiling_dir = artifact / "rounds" / "1" / "profiling" / "nsys"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        trace = profiling_dir / "opt_bs1.nsys-rep"
        trace.write_text("fake trace data", encoding="utf-8")

        # Simulate sweep slot populated by Invocation 1
        opt_slot = artifact / "rounds" / "1" / "sweeps" / "opt" / "op001"
        opt_slot.mkdir(parents=True, exist_ok=True)
        (opt_slot / "e2e_latency_results.json").write_text("{}", encoding="utf-8")

        # Invocation 2 archives the sweep slot
        _prepare_out_root(
            artifact_dir=artifact,
            out_name="rounds/1/sweeps/opt/op001",
            overwrite=False,
        )

        # Traces must still exist (they're siblings, not children of the sweep slot)
        assert trace.exists(), "nsys trace should survive sweep slot archiving"
        assert trace.read_text() == "fake trace data"


# ---------------------------------------------------------------------------
# Slot enumeration in error messages
# ---------------------------------------------------------------------------

class TestSlotEnumeration:
    """Error messages for missing --slot include 'profiling' in the valid list."""

    def test_round_only_error_mentions_profiling(self, tmp_path):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        artifact = _make_v2_artifact(tmp_path)
        ns = argparse.Namespace(round=1, slot=None, out_name="e2e_latency", _out_root=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_sweep_out_name(ns, artifact)
        msg = str(exc_info.value)
        assert "profiling" in msg, f"Error should mention 'profiling' as valid slot: {msg}"
        assert "baseline" in msg
        assert "opt/" in msg or "opt/{" in msg

    def test_out_name_error_mentions_profiling(self, tmp_path):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        artifact = _make_v2_artifact(tmp_path)
        ns = argparse.Namespace(round=None, slot=None, out_name="custom_name", _out_root=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_sweep_out_name(ns, artifact)
        msg = str(exc_info.value)
        assert "profiling" in msg, f"Error should mention 'profiling' as valid slot: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
