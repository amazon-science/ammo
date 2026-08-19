#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the sweep script refactor: GPU lock, argparse, config validation.

Covers the three removed options and their replacements:
  1. --no-gpu-lock / --gpu-lock  → lock always on, children skip via --_child-label
  2. --allow-identical-config     → unconditional SystemExit on identical configs
  3. --execution-mode cli_per_bs  → only inproc_sweep exists

Also covers previously-untested pure functions: _is_placeholder, _require*,
_sanitize_filename, _parse_latency_metrics, _check_patterns, _format_cmd_for_md,
_bench_exe_tokens, _prepare_out_root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Argparse: removed flags should not be recognized
# ---------------------------------------------------------------------------

class TestArgparseRemovedFlags:
    """Verify removed CLI flags are rejected by the argument parser."""

    def _parser(self):
        """Get the script's argument parser by importing main and intercepting."""
        from run_vllm_bench_latency_sweep import main
        import argparse as _ap

        # Capture the parser before it calls parse_args.
        original_parse = _ap.ArgumentParser.parse_args

        captured = {}

        def intercept(self, args=None, namespace=None):
            captured["parser"] = self
            raise SystemExit(0)  # Don't actually run main()

        with mock.patch.object(_ap.ArgumentParser, "parse_args", intercept):
            try:
                main()
            except SystemExit:
                pass

        return captured.get("parser")

    def test_no_gpu_lock_rejected(self):
        """--no-gpu-lock is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--no-gpu-lock"])

    def test_gpu_lock_rejected(self):
        """--gpu-lock is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--gpu-lock"])

    def test_allow_identical_config_rejected(self):
        """--allow-identical-config is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--allow-identical-config"])

    def test_execution_mode_rejected(self):
        """--execution-mode is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--execution-mode", "inproc_sweep"])

    def test_child_label_still_accepted(self):
        """Internal --_child-label is still accepted."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        ns = p.parse_args(["--artifact-dir", "/tmp/x", "--_child-label", "baseline"])
        assert ns._child_label == "baseline"

    def test_nsys_capture_output_steps_requires_nsys_profile(self, tmp_path):
        from run_vllm_bench_latency_sweep import main

        argv = [
            "run_vllm_bench_latency_sweep.py",
            "--artifact-dir",
            str(tmp_path),
            "--labels",
            "baseline",
            "--nsys-capture-output-steps",
            "2,50%,100%",
        ]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit, match="require --nsys-profile"):
                main()


# ---------------------------------------------------------------------------
# Config identity check (unconditional)
# ---------------------------------------------------------------------------

class TestIdenticalConfigCheck:
    """The identical-config check is now unconditional — no bypass flag."""

    def _make_target_json(self, tmp_path, *, opt_env=None, opt_extra_args=None,
                          baseline_extra_args=None):
        """Write a minimal target.json and return its path."""
        target = {
            "artifact_dir": str(tmp_path),
            "target": {
                "model_id": "test/model",
                "dtype": "fp16",
                "tp": 1,
                "ep": 1,
                "max_model_len": 4096,
            },
            "workload": {
                "input_len": 64,
                "output_len": 512,
                "batch_sizes": [1],
                "num_iters": 1,
            },
            "bench": {
                "runner": "vllm_bench_latency",
                "vllm_cmd": "vllm",
                "extra_args": [],
                "baseline_extra_args": baseline_extra_args or [],
                "opt_extra_args": opt_extra_args or [],
                "baseline_env": {},
                "opt_env": opt_env or {},
                "baseline_label": "baseline",
                "opt_label": "opt",
            },
        }
        p = tmp_path / "target.json"
        p.write_text(json.dumps(target), encoding="utf-8")
        return p

    def test_identical_config_both_labels_raises(self, tmp_path):
        """When both labels selected and configs identical, script fails fast."""
        self._make_target_json(tmp_path)
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            with pytest.raises(SystemExit, match="identical"):
                main()

    def test_baseline_only_skips_check(self, tmp_path):
        """With --labels baseline, the identity check is skipped (only one label)."""
        self._make_target_json(tmp_path)
        from run_vllm_bench_latency_sweep import main
        # This should NOT raise about identical config — it'll fail later
        # at the actual benchmark step, but not at the config check.
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline",
        ]):
            # We expect it to get past config validation. It'll fail at
            # GPU lock or benchmark execution, but not at config identity.
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)

    def test_different_opt_env_passes_check(self, tmp_path):
        """When opt_env is set, configs differ, so the check passes."""
        self._make_target_json(tmp_path, opt_env={"MY_OPT_FLAG": "1"})
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)

    def test_different_opt_extra_args_passes_check(self, tmp_path):
        """When opt_extra_args differ from baseline, check passes."""
        self._make_target_json(tmp_path, opt_extra_args=["--some-flag"])
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)


# ---------------------------------------------------------------------------
# nsys capture output steps
# ---------------------------------------------------------------------------

class TestNsysCaptureOutputSteps:
    """Selected-step nsys profiling shifts prompt length and keeps traces short."""

    def test_parse_steps_resolves_percentages(self):
        from run_vllm_bench_latency_sweep import _parse_nsys_capture_output_steps

        assert _parse_nsys_capture_output_steps("2,50%,100%", 512) == [2, 256, 512]

    def test_parse_steps_deduplicates_preserving_order(self):
        from run_vllm_bench_latency_sweep import _parse_nsys_capture_output_steps

        assert _parse_nsys_capture_output_steps("2,50%,2,100%", 4) == [2, 4]

    def test_parse_steps_rejects_beyond_output_len(self):
        from run_vllm_bench_latency_sweep import _parse_nsys_capture_output_steps

        with pytest.raises(SystemExit, match="exceeds"):
            _parse_nsys_capture_output_steps("513", 512)

    def test_expand_floors_window_above_prefill_and_shifts_horizon(self):
        """Window is floored child-wide ABOVE chunked prefill; input_len shifts to keep
        the captured decode depth (src_il + step) invariant.

        For il=8192,bs=8 the deepest step (512) gives il_eff_max=8702 →
        ceil(8702*8/16384)=5 prefill steps, +MARGIN(6) → window floored to 11 (the
        requested 2 is just a lower bound). input_len_eff = src_il + step - 11; the
        sum src_il+step (8194/8448/8704) is preserved, so depth is unchanged.
        """
        from run_vllm_bench_latency_sweep import _expand_nsys_profile_buckets

        buckets = [{"input_len": 8192, "output_len": 512, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_window_output_len=2,
            nsys_capture_output_steps="2,50%,100%",
        )

        # window floored to 11 (n_prefill=5 + MARGIN=6); il shifted by -11; depth preserved.
        assert [(b["input_len"], b["output_len"]) for b in expanded] == [
            (8183, 11),
            (8437, 11),
            (8693, 11),
        ]
        # captured decode depth = src_il + step stays invariant under the floored window.
        assert [b["input_len"] + b["output_len"] for b in expanded] == [
            8192 + 2,
            8192 + 256,
            8192 + 512,
        ]
        assert [b["nsys_source_input_len"] for b in expanded] == [8192, 8192, 8192]
        assert [b["nsys_source_output_len"] for b in expanded] == [512, 512, 512]
        assert [b["nsys_capture_output_step"] for b in expanded] == [2, 256, 512]
        assert [b["nsys_capture_window_output_len"] for b in expanded] == [11, 11, 11]
        assert [b["nsys_capture_target_output_len"] for b in expanded] == [512, 512, 512]

    def test_expand_default_window_is_lower_bound_then_floored(self):
        """With no --nsys-capture-window-output-len the requested window defaults to 2,
        but it is only a LOWER BOUND: it gets floored above chunked prefill.

        il=4096,bs=1, deepest step 512 → il_eff_max=4606 → ceil(4606/16384)=1 prefill
        step, +MARGIN(6) → window floored to 7. il shifts by -7; depth preserved.
        """
        from run_vllm_bench_latency_sweep import _expand_nsys_profile_buckets

        buckets = [{"input_len": 4096, "output_len": 512, "batch_size": 1}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_output_steps="50%,100%",
        )

        assert [(b["input_len"], b["output_len"]) for b in expanded] == [
            (4345, 7),
            (4601, 7),
        ]
        # depth = src_il + step invariant (4096+256, 4096+512).
        assert [b["input_len"] + b["output_len"] for b in expanded] == [4352, 4608]
        assert [b["nsys_capture_window_output_len"] for b in expanded] == [7, 7]

    def test_expand_accepts_step_before_window_no_longer_rejected(self):
        """REGRESSION: the OLD `step < window` SystemExit guard was REMOVED by the fix.

        A shallow step (2) is now capturable because the input_len shift lands the
        capture at the correct decode depth regardless of the (floored) window. Here
        il=64,bs=8 floors the window to 7 (n_prefill=1 + MARGIN=6); step 2 yields
        il_eff = 64 + 2 - 7 = 59 (>= 1), so it is emitted, NOT rejected.
        """
        from run_vllm_bench_latency_sweep import _expand_nsys_profile_buckets

        buckets = [{"input_len": 64, "output_len": 512, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_window_output_len=4,
            nsys_capture_output_steps="2,50%,100%",
        )

        # No SystemExit. Window floored to 7; step 2 survives (il_eff=59 >= 1).
        assert [(b["input_len"], b["output_len"]) for b in expanded] == [
            (59, 7),
            (313, 7),
            (569, 7),
        ]
        assert [b["nsys_capture_output_step"] for b in expanded] == [2, 256, 512]
        assert all(b["nsys_capture_window_output_len"] == 7 for b in expanded)

    def test_nsys_tag_buckets_expand_and_filter_dp_skips(self):
        from run_vllm_bench_latency_sweep import _nsys_tag_buckets_for_dp

        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]
        measured = _nsys_tag_buckets_for_dp(
            buckets,
            4,
            nsys_capture_output_steps="2,50%,100%",
        )

        # bs=1 dropped by DP-skip; bs=8 expanded. Window floored to 7 (n_prefill=1 +
        # MARGIN=6 for il=64); input_len shifted by -7 (64+step-7 = 59/313/569).
        assert [(b["batch_size"], b["input_len"], b["output_len"]) for b in measured] == [
            (8, 59, 7),
            (8, 313, 7),
            (8, 569, 7),
        ]

    def test_nsys_tag_buckets_dedupe_duplicate_expanded_tags(self):
        from run_vllm_bench_latency_sweep import _bucket_file_tag, _nsys_tag_buckets_for_dp

        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 8},
            {"input_len": 318, "output_len": 512, "batch_size": 8},
        ]
        measured = _nsys_tag_buckets_for_dp(
            buckets,
            1,
            nsys_output_len=None,
            nsys_capture_window_output_len=2,
            nsys_capture_output_steps="2,50%",
        )
        tags = [_bucket_file_tag(bucket, measured) for bucket in measured]

        # Window floored to 7. il=64 → {59, 313}; il=318 → {313, 567}; the duplicate
        # il_eff=313 is deduped, leaving three unique shapes.
        assert [(b["input_len"], b["output_len"], b["batch_size"]) for b in measured] == [
            (59, 7, 8),
            (313, 7, 8),
            (567, 7, 8),
        ]
        assert len(tags) == len(set(tags))

    def test_output_len_remains_horizon_override_window_is_separate(self):
        from run_vllm_bench_latency_sweep import _expand_nsys_profile_buckets

        buckets = [{"input_len": 64, "output_len": 1024, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_output_len=512,
            nsys_capture_window_output_len=2,
            nsys_capture_output_steps="50%,100%",
        )

        # Percentages resolve against nsys_output_len=512 (NOT the workload's 1024), so
        # steps are [256, 512]. Window floored to 7 (il=64 → il_eff_max=569 →
        # ceil(569*8/16384)=1 + MARGIN(6)); il shifted by -7.
        assert [(b["input_len"], b["output_len"]) for b in expanded] == [
            (313, 7),
            (569, 7),
        ]
        assert [b["nsys_capture_target_output_len"] for b in expanded] == [512, 512]

    def test_nsys_capture_output_steps_requires_single_iter(self, tmp_path):
        from run_vllm_bench_latency_sweep import main

        argv = [
            "run_vllm_bench_latency_sweep.py",
            "--artifact-dir",
            str(tmp_path),
            "--labels",
            "baseline",
            "--nsys-profile",
            "--nsys-capture-output-steps",
            "2,50%,100%",
            "--nsys-num-iters",
            "2",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch("shutil.which", return_value="/usr/bin/nsys"):
            with pytest.raises(SystemExit, match="requires --nsys-num-iters 1"):
                main()

    def test_configure_selected_step_profiler_sets_cuda_delay(self):
        """delay_iterations == the floored capture window, and max_num_batched_tokens is
        pinned to the chunk size used in the floor arithmetic.

        Bucket is a realistic post-expand selected-step bucket (il_eff=8693, bs=8,
        window=11). n_prefill(8693,8)=5 < window=11, so the defensive guard passes and
        the profiler arms at delay=11.
        """
        from run_vllm_bench_latency_sweep import (
            _PROFILING_CHUNK_TOKENS,
            _apply_selected_step_profiler_config,
        )

        ea_dict = {"profiler_config": {"profiler": "none"}}
        _apply_selected_step_profiler_config(
            ea_dict,
            {
                "input_len": 8693,
                "batch_size": 8,
                "nsys_capture_output_step": 512,
                "nsys_capture_window_output_len": 11,
            },
        )

        assert ea_dict["profiler_config"] == {
            "profiler": "cuda",
            "delay_iterations": 11,
            "max_iterations": 1,
        }
        assert ea_dict["max_num_batched_tokens"] == _PROFILING_CHUNK_TOKENS

# ---------------------------------------------------------------------------
# Sanitize filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for GPU lock key sanitization."""

    def _call(self, s):
        from run_vllm_bench_latency_sweep import _sanitize_filename
        return _sanitize_filename(s)

    def test_simple_device_list(self):
        assert self._call("0,1,2") == "0_1_2"

    def test_alpha_numeric(self):
        assert self._call("gpu-0") == "gpu-0"

    def test_special_chars(self):
        assert self._call("a/b:c") == "a_b_c"

    def test_empty_string(self):
        assert self._call("") == "default"

    def test_all_special(self):
        assert self._call("///") == "default"

    def test_preserves_dots_equals(self):
        """Dots and equals are kept (conservative but safe)."""
        assert self._call("key=val.1") == "key=val.1"


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

class TestIsPlaceholder:
    """Tests for detecting placeholder values in target.json."""

    def _call(self, v):
        from run_vllm_bench_latency_sweep import _is_placeholder
        return _is_placeholder(v)

    def test_fill_me_marker(self):
        assert self._call("<FILL_ME>") is True

    def test_angle_bracket_custom(self):
        assert self._call("<ENABLE_FLAG>") is True

    def test_angle_with_whitespace(self):
        assert self._call("  <FLAG>  ") is True

    def test_real_value(self):
        assert self._call("VLLM_ENABLE_OPT") is False

    def test_non_string(self):
        assert self._call(42) is False
        assert self._call(None) is False
        assert self._call(["<FOO>"]) is False


# ---------------------------------------------------------------------------
# Require field helpers
# ---------------------------------------------------------------------------

class TestRequireFields:
    """Tests for _require, _require_int, _require_list_int."""

    def test_require_present(self):
        from run_vllm_bench_latency_sweep import _require
        assert _require({"key": "val"}, "key", "ctx") == "val"

    def test_require_missing_raises(self):
        from run_vllm_bench_latency_sweep import _require
        with pytest.raises(SystemExit, match="Missing required field"):
            _require({}, "key", "ctx")

    def test_require_placeholder_raises(self):
        from run_vllm_bench_latency_sweep import _require
        with pytest.raises(SystemExit, match="still placeholder"):
            _require({"key": "<FILL_ME>"}, "key", "ctx")

    def test_require_int_valid(self):
        from run_vllm_bench_latency_sweep import _require_int
        assert _require_int({"n": 42}, "n", "ctx") == 42

    def test_require_int_wrong_type(self):
        from run_vllm_bench_latency_sweep import _require_int
        with pytest.raises(SystemExit, match="Expected int"):
            _require_int({"n": "42"}, "n", "ctx")

    def test_require_list_int_valid(self):
        from run_vllm_bench_latency_sweep import _require_list_int
        assert _require_list_int({"bs": [1, 4, 8]}, "bs", "ctx") == [1, 4, 8]

    def test_require_list_int_mixed_types(self):
        from run_vllm_bench_latency_sweep import _require_list_int
        with pytest.raises(SystemExit, match="Expected list"):
            _require_list_int({"bs": [1, "four"]}, "bs", "ctx")


# ---------------------------------------------------------------------------
# Parse latency metrics
# ---------------------------------------------------------------------------

class TestParseLatencyMetrics:
    """Tests for stdout parsing of vllm bench latency output."""

    def _call(self, stdout):
        from run_vllm_bench_latency_sweep import _parse_latency_metrics
        return _parse_latency_metrics(stdout)

    def test_typical_output(self):
        stdout = (
            "Avg latency: 0.1234 seconds\n"
            "50% percentile latency: 0.1200 seconds\n"
            "99% percentile latency: 0.1300 seconds\n"
        )
        m = self._call(stdout)
        assert m["avg_s"] == pytest.approx(0.1234)
        assert m["p50_s"] == pytest.approx(0.1200)
        assert m["p99_s"] == pytest.approx(0.1300)

    def test_empty_output(self):
        assert self._call("") == {}

    def test_garbage_output(self):
        assert self._call("ERROR: model not found\nSegfault") == {}

    def test_scientific_notation(self):
        stdout = "Avg latency: 1.5e-02 seconds\n"
        m = self._call(stdout)
        assert m["avg_s"] == pytest.approx(0.015)

    def test_partial_output(self):
        """Only avg, no percentiles."""
        stdout = "Avg latency: 2.5 seconds\n"
        m = self._call(stdout)
        assert m == {"avg_s": 2.5}


# ---------------------------------------------------------------------------
# Pattern matching (fastpath evidence)
# ---------------------------------------------------------------------------

class TestCheckPatterns:
    """Tests for _check_patterns used in fast-path evidence checking."""

    def _call(self, text, require, forbid):
        from run_vllm_bench_latency_sweep import _check_patterns
        return _check_patterns(text, require, forbid)

    def test_all_required_present(self):
        result = self._call("kernel launched", ["kernel"], [])
        assert result["ok"] is True
        assert result["require_hits"] == ["kernel"]
        assert result["require_miss"] == []

    def test_required_missing(self):
        result = self._call("idle", ["kernel"], [])
        assert result["ok"] is False
        assert result["require_miss"] == ["kernel"]

    def test_forbidden_found(self):
        result = self._call("eager fallback", [], ["eager"])
        assert result["ok"] is False
        assert result["forbid_hits"] == ["eager"]

    def test_empty_patterns(self):
        result = self._call("anything", [], [])
        assert result["ok"] is True

    def test_regex_patterns(self):
        result = self._call("compiled graph v2.1", [r"graph v\d+\.\d+"], [])
        assert result["ok"] is True

    def test_combined_require_and_forbid(self):
        text = "CUDA graph captured, no eager fallback"
        result = self._call(text, ["CUDA graph"], ["eager"])
        assert result["ok"] is False  # forbid hit overrides require hit


# ---------------------------------------------------------------------------
# Format command for markdown
# ---------------------------------------------------------------------------

class TestFormatCmdForMd:
    """Tests for shell command formatting."""

    def _call(self, cmd, env):
        from run_vllm_bench_latency_sweep import _format_cmd_for_md
        return _format_cmd_for_md(cmd, env)

    def test_simple_command(self):
        result = self._call(["vllm", "bench", "latency"], {})
        assert result == "vllm bench latency"

    def test_with_env(self):
        result = self._call(["vllm", "bench"], {"FOO": "1"})
        assert result.startswith("FOO=")
        assert "vllm bench" in result

    def test_quotes_special_chars(self):
        result = self._call(["cmd", "arg with spaces"], {})
        assert "arg with spaces" in result or "'arg with spaces'" in result


# ---------------------------------------------------------------------------
# Bench exe tokens
# ---------------------------------------------------------------------------

class TestBenchExeTokens:
    """Tests for vllm command parsing."""

    def _call(self, vllm_cmd):
        from run_vllm_bench_latency_sweep import _bench_exe_tokens
        return _bench_exe_tokens(vllm_cmd)

    def test_string_command(self):
        assert self._call("vllm") == ["vllm"]

    def test_string_with_spaces(self):
        assert self._call("python -m vllm") == ["python", "-m", "vllm"]

    def test_list_command(self):
        assert self._call(["python", "-m", "vllm"]) == ["python", "-m", "vllm"]

    def test_invalid_type_raises(self):
        with pytest.raises(SystemExit, match="must be str or list"):
            self._call(42)

    def test_list_with_non_string_raises(self):
        with pytest.raises(SystemExit, match="must be list"):
            self._call(["vllm", 42])


# ---------------------------------------------------------------------------
# Prepare output root
# ---------------------------------------------------------------------------

class TestPrepareOutRoot:
    """Tests for output directory setup logic."""

    def _call(self, *, artifact_dir, out_name, overwrite):
        from run_vllm_bench_latency_sweep import _prepare_out_root
        return _prepare_out_root(
            artifact_dir=artifact_dir, out_name=out_name, overwrite=overwrite,
        )

    def test_creates_fresh_dir(self, tmp_path):
        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=False)
        assert out.exists()
        assert out == tmp_path / "e2e"

    def test_archives_existing_nonempty(self, tmp_path):
        """Existing non-empty dir is archived (renamed), not deleted."""
        existing = tmp_path / "e2e"
        existing.mkdir()
        (existing / "old_data.json").write_text("{}")

        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=False)
        assert out.exists()
        # Original should be gone (renamed).
        assert not (existing / "old_data.json").exists()
        # An archive directory should exist.
        archives = [p for p in tmp_path.iterdir() if p.name.startswith("e2e_")]
        assert len(archives) == 1

    def test_overwrite_deletes_existing(self, tmp_path):
        """With overwrite=True, existing dir is removed."""
        existing = tmp_path / "e2e"
        existing.mkdir()
        (existing / "old_data.json").write_text("{}")

        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=True)
        assert out.exists()
        # No archive created.
        archives = [p for p in tmp_path.iterdir() if p.name.startswith("e2e_")]
        assert len(archives) == 0
        # Old file should be gone.
        assert not (out / "old_data.json").exists()


# ---------------------------------------------------------------------------
# Maybe list str
# ---------------------------------------------------------------------------

class TestMaybeListStr:
    """Tests for _maybe_list_str helper."""

    def _call(self, obj, key):
        from run_vllm_bench_latency_sweep import _maybe_list_str
        return _maybe_list_str(obj, key)

    def test_present_list(self):
        assert self._call({"a": ["x", "y"]}, "a") == ["x", "y"]

    def test_missing_key(self):
        assert self._call({}, "a") == []

    def test_none_value(self):
        assert self._call({"a": None}, "a") == []

    def test_invalid_type_raises(self):
        with pytest.raises(SystemExit, match="Expected list"):
            self._call({"a": "not_a_list"}, "a")


# ---------------------------------------------------------------------------
# _ensure_worktree_pythonpath: prepend CWD so worktree vllm is found first
# ---------------------------------------------------------------------------

class TestEnsureWorktreePythonpath:
    """Verify _ensure_worktree_pythonpath correctly sets PYTHONPATH."""

    def _call(self, env, cwd=None):
        from run_vllm_bench_latency_sweep import _ensure_worktree_pythonpath
        if cwd is not None:
            with mock.patch("run_vllm_bench_latency_sweep.os.getcwd", return_value=cwd):
                return _ensure_worktree_pythonpath(env)
        return _ensure_worktree_pythonpath(env)

    def test_no_existing_pythonpath(self):
        """CWD is set as PYTHONPATH when none exists."""
        env = {"FOO": "bar"}
        result = self._call(env, cwd="/tmp/worktree")
        assert result["PYTHONPATH"] == "/tmp/worktree"
        assert result["FOO"] == "bar"

    def test_existing_pythonpath_is_preserved(self):
        """CWD is prepended to existing PYTHONPATH."""
        env = {"PYTHONPATH": "/some/other/path"}
        result = self._call(env, cwd="/tmp/worktree")
        parts = result["PYTHONPATH"].split(":")
        assert parts[0] == "/tmp/worktree"
        assert "/some/other/path" in parts

    def test_cwd_already_first_in_pythonpath(self):
        """No duplication when CWD is already first entry."""
        env = {"PYTHONPATH": "/tmp/worktree:/other"}
        result = self._call(env, cwd="/tmp/worktree")
        # Should not duplicate
        parts = result["PYTHONPATH"].split(":")
        assert parts.count("/tmp/worktree") == 1
        assert parts[0] == "/tmp/worktree"

    def test_does_not_mutate_input(self):
        """Input dict is not modified in place."""
        env = {"PYTHONPATH": "/old"}
        result = self._call(env, cwd="/tmp/worktree")
        assert env["PYTHONPATH"] == "/old"
        assert result["PYTHONPATH"].startswith("/tmp/worktree")

    def test_empty_pythonpath(self):
        """Empty string PYTHONPATH is treated as absent."""
        env = {"PYTHONPATH": ""}
        result = self._call(env, cwd="/tmp/worktree")
        assert result["PYTHONPATH"] == "/tmp/worktree"




# ---------------------------------------------------------------------------
# _sanitize_vllm_op_env — Track A6 preserve_keys contract
# ---------------------------------------------------------------------------


class TestSanitizeVllmOpEnv:
    """Verify the broadened sanitizer strips all VLLM_* except preserve_keys."""

    def _fn(self):
        import run_vllm_bench_latency_sweep as sweep
        return sweep._sanitize_vllm_op_env

    def test_strips_vllm_op_numeric(self):
        """Numeric op flags (VLLM_OP001, VLLM_OP042) are stripped by default."""
        sanitize = self._fn()
        env = {"VLLM_OP001": "1", "VLLM_OP042": "1", "PATH": "/usr/bin"}
        out = sanitize(env)
        assert "VLLM_OP001" not in out
        assert "VLLM_OP042" not in out
        assert out["PATH"] == "/usr/bin"

    def test_strips_broad_vllm_prefix(self):
        """After A6 broadening, VLLM_* named flags (not just VLLM_OP\\d+) are stripped."""
        sanitize = self._fn()
        env = {
            "VLLM_MOE_TRITON_ROUTER": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_USE_V1": "1",
            "PATH": "/usr/bin",
        }
        out = sanitize(env)
        assert "VLLM_MOE_TRITON_ROUTER" not in out
        assert "VLLM_ATTENTION_BACKEND" not in out
        assert "VLLM_USE_V1" not in out
        assert out["PATH"] == "/usr/bin"

    def test_preserve_keys_survive(self):
        """Keys in preserve_keys pass through even though they match ^VLLM_."""
        sanitize = self._fn()
        env = {
            "VLLM_MOE_TRITON_ROUTER": "1",
            "VLLM_OP001": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        }
        out = sanitize(env, preserve_keys={"VLLM_MOE_TRITON_ROUTER", "VLLM_ATTENTION_BACKEND"})
        assert out["VLLM_MOE_TRITON_ROUTER"] == "1"
        assert out["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"
        assert "VLLM_OP001" not in out  # not in preserve set

    def test_preserve_keys_accepts_frozenset(self):
        """preserve_keys parameter accepts frozenset as well as set."""
        sanitize = self._fn()
        env = {"VLLM_OP001": "1", "VLLM_OP002": "1"}
        out = sanitize(env, preserve_keys=frozenset({"VLLM_OP001"}))
        assert out == {"VLLM_OP001": "1"}

    def test_default_preserve_keys_is_empty(self):
        """With no preserve_keys arg, all VLLM_* are stripped."""
        sanitize = self._fn()
        env = {"VLLM_OP001": "1", "VLLM_FOO": "bar"}
        out = sanitize(env)
        assert "VLLM_OP001" not in out
        assert "VLLM_FOO" not in out

    def test_non_vllm_prefix_always_passes_through(self):
        """Non-VLLM_* keys are never touched, regardless of preserve_keys."""
        sanitize = self._fn()
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/x",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": "/opt/vllm",
        }
        out = sanitize(env, preserve_keys=set())
        assert out == env


# ---------------------------------------------------------------------------
# vLLM-coupling instrumentation: throughput_method guard, version stamp,
# hardware-derived nsys trace backend.
# ---------------------------------------------------------------------------


class TestThroughputMethodMismatch:
    """Cross-tier OTPS comparison must be refused, not silently emitted."""

    def _fn(self):
        import run_vllm_bench_latency_sweep as sweep
        return sweep._throughput_method_mismatch

    def test_same_method_returns_none(self):
        f = self._fn()
        assert f({"throughput_method": "decode_metrics"},
                 {"throughput_method": "decode_metrics"}) is None

    def test_mismatch_returns_both_methods(self):
        f = self._fn()
        assert f({"throughput_method": "decode_metrics"},
                 {"throughput_method": "gross_e2e"}) == ("decode_metrics", "gross_e2e")

    def test_missing_method_returns_none(self):
        """Legacy entries carry no method — nothing to compare, no false alarm."""
        f = self._fn()
        assert f({}, {"throughput_method": "gross_e2e"}) is None
        assert f({"throughput_method": "decode_metrics"}, {}) is None
        assert f({}, {}) is None

    def test_non_dict_inputs_return_none(self):
        f = self._fn()
        assert f(None, None) is None
        assert f({"throughput_method": "decode_metrics"}, None) is None

    def test_guard_is_wired_into_the_gain_computation(self):
        """otps_gain_pct / tpot_improvement_pct sit behind the guard."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "run_vllm_bench_latency_sweep.py").read_text(encoding="utf-8")
        guard_at = src.index("tp_mismatch = _throughput_method_mismatch(")
        assert guard_at < src.index('new_row["otps_gain_pct"]')
        assert guard_at < src.index('new_row["tpot_improvement_pct"]')

    def test_tier_switch_would_have_inverted_the_verdict(self):
        """A tier switch turns a real +5.3% E2E win into a reported -44.7% OTPS loss."""
        import run_vllm_bench_latency_sweep as sweep

        b = sweep._compute_token_throughput({"avg_s": 10.0, "decode_avg_s": 5.24}, 512, 32)
        o = sweep._compute_token_throughput({"avg_s": 9.5}, 512, 32)
        assert b["throughput_method"] == "decode_metrics"
        assert o["throughput_method"] == "gross_e2e"
        cross_tier_gain = (o["otps"] - b["otps"]) / b["otps"] * 100.0
        assert cross_tier_gain < -40.0  # the wrong number the guard suppresses
        assert sweep._throughput_method_mismatch(b, o) is not None


class TestVllmVersionStamp:
    """The results header must record the vLLM version for cross-version A/B."""

    def test_returns_str_or_none(self):
        import run_vllm_bench_latency_sweep as sweep
        v = sweep._vllm_version()
        assert v is None or isinstance(v, str)

    def test_never_raises_without_vllm(self):
        import run_vllm_bench_latency_sweep as sweep
        with mock.patch("importlib.metadata.version", side_effect=Exception("boom")):
            assert sweep._vllm_version() is None

    def test_header_and_baseline_import_are_wired(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "run_vllm_bench_latency_sweep.py").read_text(encoding="utf-8")
        assert '"vllm_version": _vllm_version(),' in src
        assert "CROSS-VERSION A/B" in src


class TestDefaultNsysTrace:
    """--nsys-trace default comes from target.hardware, not agent memory."""

    def _fn(self):
        import run_vllm_bench_latency_sweep as sweep
        return sweep._default_nsys_trace

    @pytest.mark.parametrize("hardware", [
        "B200", "b200", "8xB300", "GB200", "GB300", "NVIDIA B200 SXM", "Blackwell",
    ])
    def test_blackwell_selects_software_tracing(self, hardware):
        assert self._fn()(hardware) == "cuda-sw"

    @pytest.mark.parametrize("hardware", [
        "H100", "H200", "A100", "L40S", "A10G",
        # "GB" as a memory unit must NOT trip the Blackwell rule.
        "A100-80GB", "NVIDIA H100 80GB HBM3", "H200-141GB",
    ])
    def test_non_blackwell_selects_hardware_tracing(self, hardware):
        assert self._fn()(hardware) == "cuda"

    def test_missing_hardware_falls_back_to_cuda(self):
        f = self._fn()
        assert f(None) == "cuda"
        assert f("") == "cuda"
        assert f(123) == "cuda"

    def test_flag_default_is_none_so_override_is_detectable(self):
        """argparse default must be None; main() resolves it from target.json."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts"
               / "run_vllm_bench_latency_sweep.py").read_text(encoding="utf-8")
        block = src[src.index('"--nsys-trace"'):]
        block = block[:block.index("p.add_argument", 10)]
        assert "default=None," in block
        assert "if args.nsys_trace is None:" in src
        assert "_default_nsys_trace(target.get(\"hardware\"))" in src
