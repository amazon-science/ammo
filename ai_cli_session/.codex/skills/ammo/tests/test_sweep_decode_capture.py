#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the decode-isolated selected-step profiler capture fix.

Covers the BLOCKING prefill-contamination fix in run_vllm_bench_latency_sweep.py:
  - _n_prefill_worker_steps: chunked-prefill worker-step count
  - _selected_step_effective_window: child-wide window floored ABOVE all prefill steps
  - _expand_nsys_profile_buckets: applies the floor + input_len shift; drops only il_eff<1
  - _apply_selected_step_profiler_config: pins max_num_batched_tokens + arms delay==window
  - evaluate_decode_shape: pure decode-vs-prefill classifier
  - assert_decode_shaped_capture: hard guard wrapper

Root cause being verified (INVESTIGATION_prefill_contamination.md): the old
`delay_iterations = window` (=2) armed vLLM's WorkerProfiler on prefill chunk #2 on
long-context workloads, capturing a chunked-prefill forward instead of decode.

The fix (validated 5/5 against real Nemotron sqlite traces + vLLM source):
the script already SHIFTS input_len (il_eff = src_il + step - window) so the captured
decode depth = src_il + step, INVARIANT under window. So we simply FLOOR the child-wide
window above the chunked-prefill worker-step count; the shift absorbs the larger window
and the capture lands on a clean full-batch decode step. delay == window (the proven
Nemotron mechanism); no decouple, no per-bucket split, no step dropping (except the
il_eff<1 corner).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_vllm_bench_latency_sweep import (  # noqa: E402
    _PROFILING_CHUNK_TOKENS,
    _SELECTED_STEP_DECODE_MARGIN,
    _apply_selected_step_profiler_config,
    _compute_nsys_cudagraph_capture_sizes,
    _dedupe_nsys_profile_buckets,
    _expand_nsys_profile_buckets,
    _n_prefill_worker_steps,
    _nsys_runner_mapping,
    _parse_positions_per_step,
    _selected_step_effective_window,
    assert_decode_shaped_capture,
    evaluate_decode_shape,
)

# The real (gemma-4-31B-it-NVFP4) spec-decode extra_args carry an inline JSON
# --speculative-config with num_speculative_tokens=4 -> positions_per_step=5.
_REAL_SPEC_CONFIG = (
    '{"method":"mtp","model":"google/gemma-4-31B-it-assistant",'
    '"num_speculative_tokens":4}'
)

CHUNK = _PROFILING_CHUNK_TOKENS
MARGIN = _SELECTED_STEP_DECODE_MARGIN


# ---------------------------------------------------------------------------
# _n_prefill_worker_steps: chunked-prefill drain count
# ---------------------------------------------------------------------------

class TestNPrefillWorkerSteps:

    def test_long_context(self):
        # 10998*8 = 87984 tokens; ceil(87984/16384) = 6
        assert _n_prefill_worker_steps(10998, 8, CHUNK) == 6

    def test_short_context_is_one(self):
        assert _n_prefill_worker_steps(64, 1, CHUNK) == 1

    def test_exact_multiple(self):
        # 16384*3 / bs=1 = 49152 → exactly 3
        assert _n_prefill_worker_steps(49152, 1, CHUNK) == 3

    def test_scales_with_batch(self):
        assert _n_prefill_worker_steps(10000, 8, CHUNK) > _n_prefill_worker_steps(
            10000, 1, CHUNK
        )

    def test_invalid_chunk_raises(self):
        with pytest.raises(SystemExit):
            _n_prefill_worker_steps(100, 1, 0)


# ---------------------------------------------------------------------------
# _selected_step_effective_window: floor ABOVE prefill for every capture point
# ---------------------------------------------------------------------------

class TestSelectedStepEffectiveWindow:

    def test_floors_above_prefill_long_context(self):
        """il=10000, bs=8, steps 2,500,1000: must clear the deepest prefill.

        For the deepest il_eff (smallest window=requested=2): il_eff=10998 at step 1000,
        n_prefill(10998,8)=6, so floor = 6 + margin(6) = 12 (> requested 2).
        """
        buckets = [{"input_len": 10000, "output_len": 1000, "batch_size": 8}]
        w = _selected_step_effective_window(
            buckets,
            requested_window=2,
            nsys_output_len=None,
            nsys_capture_output_steps="2,50%,100%",
        )
        assert w == 6 + MARGIN, w
        assert w > 2

    def test_short_context_keeps_requested(self):
        """Tiny context: n_prefill==1, floor=1+margin, but requested may already exceed."""
        buckets = [{"input_len": 64, "output_len": 8, "batch_size": 1}]
        w = _selected_step_effective_window(
            buckets,
            requested_window=2,
            nsys_output_len=None,
            nsys_capture_output_steps="100%",  # step 8
        )
        # n_prefill(64+8-2, 1) = 1 → floor = 1 + margin = 7; requested 2 < 7.
        assert w == 1 + MARGIN

    def test_requested_window_is_lower_bound(self):
        """A large explicit window is honored when it exceeds the computed floor."""
        buckets = [{"input_len": 64, "output_len": 8, "batch_size": 1}]
        w = _selected_step_effective_window(
            buckets,
            requested_window=100,
            nsys_output_len=None,
            nsys_capture_output_steps="100%",
        )
        assert w == 100

    def test_floor_clears_prefill_for_every_bucket(self):
        """Mixed bs: a single child-wide window must beat the worst bucket's prefill.

        bs=32 at il_eff≈27000: n_prefill≈53, floor=53+6=59. Verify w > n_prefill for
        every (bucket, step) at the shifted il_eff.
        """
        buckets = [
            {"input_len": 27000, "output_len": 1000, "batch_size": 8},
            {"input_len": 27000, "output_len": 1000, "batch_size": 32},
        ]
        w = _selected_step_effective_window(
            buckets,
            requested_window=2,
            nsys_output_len=None,
            nsys_capture_output_steps="2,50%,100%",
        )
        for b in buckets:
            for step in (2, 500, 1000):
                il_eff = b["input_len"] + step - w
                assert w > _n_prefill_worker_steps(il_eff, b["batch_size"], CHUNK), (
                    f"window {w} fails to clear prefill for bs={b['batch_size']} step={step} "
                    f"(il_eff={il_eff})"
                )

    def test_kimi_regression_window_exceeds_prefill(self):
        """The original failing case: il=10000, bs=8, window=2 was CONTAMINATED.

        The floor must raise the window above n_prefill(il_eff,8)=5 for this workload.
        """
        buckets = [{"input_len": 10000, "output_len": 2, "batch_size": 8}]
        w = _selected_step_effective_window(
            buckets,
            requested_window=2,
            nsys_output_len=None,
            nsys_capture_output_steps="100%",  # step 2
        )
        il_eff = 10000 + 2 - w
        assert w > _n_prefill_worker_steps(il_eff, 8, CHUNK)
        assert w > 2  # not the old broken value


# ---------------------------------------------------------------------------
# _expand_nsys_profile_buckets: floor + shift; drop only il_eff<1
# ---------------------------------------------------------------------------

class TestExpandSelectedStepBuckets:

    def test_shift_and_floor_applied(self):
        """Every expanded bucket gets the SAME child-wide floored window, and the
        input_len is shifted so captured depth = src_il + step."""
        buckets = [{"input_len": 10000, "output_len": 1000, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_output_steps="2,50%,100%",
        )
        windows = {b["nsys_capture_window_output_len"] for b in expanded}
        assert len(windows) == 1, "single child-wide window expected"
        w = windows.pop()
        assert w == 6 + MARGIN
        for b in expanded:
            step = b["nsys_capture_output_step"]
            # captured depth invariant
            assert b["input_len"] + w == b["nsys_source_input_len"] + step
            assert b["output_len"] == w

    def test_step2_is_capturable_not_dropped(self):
        """Regression for the DA's Q2b: 'step 2' is capturable via the shift, NOT dropped.

        The window floor makes il_eff drop below src_il, but the capture is still a clean
        full-batch decode (window > n_prefill(il_eff)).
        """
        buckets = [{"input_len": 27000, "output_len": 1000, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_output_steps="2,50%,100%",
        )
        steps = sorted(b["nsys_capture_output_step"] for b in expanded)
        assert 2 in steps, "step 2 must be captured, not dropped"
        for b in expanded:
            w = b["nsys_capture_window_output_len"]
            assert w > _n_prefill_worker_steps(b["input_len"], b["batch_size"], CHUNK)

    def test_drops_only_when_il_eff_below_one(self, capsys):
        """The single sub-floor corner: context too short to host the window → drop+WARN."""
        # Tiny context, large requested window forces il_eff < 1 for the shallow step.
        buckets = [{"input_len": 4, "output_len": 8, "batch_size": 1}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_window_output_len=50,
            nsys_capture_output_steps="100%",  # step 8 → il_eff = 4 + 8 - 50 < 1
        )
        # That capture point is dropped (no expanded bucket for it).
        assert all(b.get("nsys_capture_output_step") != 8 for b in expanded)
        err = capsys.readouterr().err
        assert "dropping selected-step capture" in err

    def test_non_selected_step_unchanged(self):
        """Without --nsys-capture-output-steps, expansion is the plain passthrough."""
        buckets = [{"input_len": 100, "output_len": 8, "batch_size": 1}]
        expanded = _expand_nsys_profile_buckets(buckets)
        assert len(expanded) == 1
        assert "nsys_capture_output_step" not in expanded[0]


# ---------------------------------------------------------------------------
# _apply_selected_step_profiler_config: pins chunk + arms delay==window
# ---------------------------------------------------------------------------

class TestApplySelectedStepProfilerConfig:

    def _bucket(self, **kw):
        # input_len here is the ALREADY-SHIFTED il_eff (as produced by _expand_…).
        b = {
            "input_len": 10998,
            "batch_size": 8,
            "nsys_capture_window_output_len": 6 + MARGIN,  # floored window
            "nsys_capture_output_step": 1000,
        }
        b.update(kw)
        return b

    def test_pins_chunk_tokens(self):
        ea = {}
        _apply_selected_step_profiler_config(ea, self._bucket())
        assert ea["max_num_batched_tokens"] == _PROFILING_CHUNK_TOKENS

    def test_pins_chunk_even_when_set_differently(self):
        """A100-safety: force the pin regardless of any prior value (warns)."""
        ea = {"max_num_batched_tokens": 8192}
        _apply_selected_step_profiler_config(ea, self._bucket())
        assert ea["max_num_batched_tokens"] == _PROFILING_CHUNK_TOKENS

    def test_delay_equals_window(self):
        ea = {}
        b = self._bucket()
        _apply_selected_step_profiler_config(ea, b)
        assert ea["profiler_config"]["delay_iterations"] == b[
            "nsys_capture_window_output_len"
        ]
        assert ea["profiler_config"]["max_iterations"] == 1
        assert ea["profiler_config"]["profiler"] == "cuda"

    def test_delay_clears_prefill(self):
        """delay (=window) must be strictly past the bucket's prefill steps."""
        ea = {}
        b = self._bucket()
        _apply_selected_step_profiler_config(ea, b)
        n_prefill = _n_prefill_worker_steps(b["input_len"], b["batch_size"], CHUNK)
        assert ea["profiler_config"]["delay_iterations"] > n_prefill

    def test_fails_loud_if_window_below_prefill(self):
        """Defensive guard: an unfloored window (e.g. old window=2) fails loud."""
        ea = {}
        bad = self._bucket(nsys_capture_window_output_len=2)
        with pytest.raises(SystemExit, match="does not clear"):
            _apply_selected_step_profiler_config(ea, bad)

    def test_invalid_window_raises(self):
        ea = {}
        with pytest.raises(SystemExit):
            _apply_selected_step_profiler_config(
                ea, self._bucket(nsys_capture_window_output_len=0)
            )


# ---------------------------------------------------------------------------
# evaluate_decode_shape: pure classifier (unchanged from recovered fix)
# ---------------------------------------------------------------------------

class TestEvaluateDecodeShape:

    def test_decode_shaped_passes(self):
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_8(8)",
            rmsnorm_grid_x=8,
            paged_kv_attn_total_us=950.0,
            prefill_flash_attn_total_us=30.0,
            batch_size=8,
        )
        assert ok is True, reason

    def test_prefill_marker_fails(self):
        """The exact contaminating marker must be rejected."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_2(16383)_generation_1(1)",
            rmsnorm_grid_x=16384,
            paged_kv_attn_total_us=957.5,
            prefill_flash_attn_total_us=29008.7,
            batch_size=8,
        )
        assert ok is False
        assert "PREFILL" in reason

    def test_giant_rmsnorm_grid_fails_even_without_marker(self):
        ok, _ = evaluate_decode_shape(
            nvtx_marker=None,
            rmsnorm_grid_x=16384,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok is False

    def test_prefill_flash_dominates_fails(self):
        ok, _ = evaluate_decode_shape(
            nvtx_marker=None,
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=15.7,
            prefill_flash_attn_total_us=475.6,
            batch_size=8,
        )
        assert ok is False

    def test_no_signals_is_not_confirmed(self):
        """Absence of evidence is not decode-confirmation."""
        ok, _ = evaluate_decode_shape(
            nvtx_marker=None,
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok is False

    def test_compiled_hybrid_rmsnorm_grid_is_demoted_by_decode_marker(self):
        """A fused layout kernel name must not override direct scheduler evidence."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_8(8)",
            rmsnorm_grid_x=48,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok is True, reason
        assert "not scored" in reason

    def test_prefill_marker_still_overrides_small_rmsnorm_grid(self):
        """Demoting the grid heuristic must not weaken direct prefill rejection."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_2(16383)_generation_1(1)",
            rmsnorm_grid_x=8,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok is False
        assert "PREFILL" in reason

    def test_grid_tolerance_allows_small_multiple(self):
        """gridX a small multiple of bs (e.g. padded) still counts as decode."""
        ok, _ = evaluate_decode_shape(
            nvtx_marker=None,
            rmsnorm_grid_x=16,  # 2x bs=8
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok is True

    def test_marker_decode_with_only_nvtx_passes(self):
        """NVTX marker is dispositive: absent kernel symbols are SKIPPED, not failed."""
        ok, _ = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_32(32)",
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=32,
        )
        assert ok is True


# ---------------------------------------------------------------------------
# assert_decode_shaped_capture: sqlite-backed hard guard
# ---------------------------------------------------------------------------

class TestAssertDecodeShapedCaptureGuard:

    def _make_sqlite(self, tmp_path, *, marker, rmsnorm_grid, paged_us, flash_us):
        import sqlite3
        p = tmp_path / "trace.sqlite"
        con = sqlite3.connect(str(p))
        cur = con.cursor()
        cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        cur.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(demangledName INTEGER, gridX INTEGER, start INTEGER, end INTEGER)"
        )
        cur.execute("CREATE TABLE NVTX_EVENTS (text TEXT, start INTEGER, end INTEGER)")
        if marker is not None:
            cur.execute(
                "INSERT INTO NVTX_EVENTS (text, start, end) VALUES (?, 0, 1000)",
                (marker,),
            )
        sid = 1

        def add_kernel(name, grid, dur_ns, n=1):
            nonlocal sid
            cur.execute("INSERT INTO StringIds (id, value) VALUES (?, ?)", (sid, name))
            for _ in range(n):
                cur.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                    "(demangledName, gridX, start, end) VALUES (?, ?, 0, ?)",
                    (sid, grid, dur_ns),
                )
            sid += 1

        add_kernel("triton_red_fused_fused_add_rms_norm_0", rmsnorm_grid, 1000)
        if paged_us is not None:
            add_kernel("fmhaSm100xyzPagedKvDenseHV512", 22, int(paged_us * 1000))
        if flash_us is not None:
            add_kernel("flash_fwd_sm100xyz", 1040, int(flash_us * 1000))
        con.commit()
        con.close()
        return str(p)

    def test_guard_passes_on_decode_trace(self, tmp_path):
        sq = self._make_sqlite(
            tmp_path,
            marker="execute_context_0(0)_generation_8(8)",
            rmsnorm_grid=8,
            paged_us=950.0,
            flash_us=30.0,
        )
        assert_decode_shaped_capture(sq, batch_size=8)  # must not raise

    def test_guard_raises_on_prefill_trace(self, tmp_path):
        sq = self._make_sqlite(
            tmp_path,
            marker="execute_context_2(16383)_generation_1(1)",
            rmsnorm_grid=16384,
            paged_us=15.7,
            flash_us=475.6,
        )
        with pytest.raises(SystemExit, match="NOT a steady-state decode step"):
            assert_decode_shaped_capture(sq, batch_size=8)

    def test_guard_passes_qwen_hybrid_compiled_grid(self, tmp_path):
        sq = self._make_sqlite(
            tmp_path,
            marker="execute_context_0(0)_generation_32(32)",
            rmsnorm_grid=192,
            paged_us=None,
            flash_us=None,
        )
        assert_decode_shaped_capture(sq, batch_size=32)


class TestNsysRunnerMapping:

    def test_selected_step_mapping_uses_source_and_capture_shapes(self):
        assert _nsys_runner_mapping({
            "input_len": 568,
            "output_len": 8,
            "batch_size": 32,
            "nsys_source_input_len": 64,
            "nsys_source_output_len": 512,
            "nsys_capture_output_step": 512,
            "nsys_capture_window_output_len": 8,
            "nsys_capture_target_output_len": 512,
        }) == {
            "source_input_len": 64,
            "source_output_len": 512,
            "capture_output_step": 512,
            "capture_window_output_len": 8,
            "capture_target_output_len": 512,
        }

    def test_non_selected_step_mapping_is_empty(self):
        assert _nsys_runner_mapping({
            "input_len": 64,
            "output_len": 2,
            "batch_size": 32,
        }) == {}


# ---------------------------------------------------------------------------
# Spec-decode awareness: _parse_positions_per_step
# ---------------------------------------------------------------------------

class TestParsePositionsPerStep:

    def test_real_spec_config(self):
        args = ["--speculative-config", _REAL_SPEC_CONFIG, "--max-num-seqs", "32"]
        assert _parse_positions_per_step(args) == 5  # 1 + 4

    def test_no_spec_is_one(self):
        assert _parse_positions_per_step(["--max-num-seqs", "32"]) == 1
        assert _parse_positions_per_step([]) == 1

    def test_sc_alias(self):
        args = ["-sc", '{"num_speculative_tokens":7}']
        assert _parse_positions_per_step(args) == 8

    def test_count_missing_is_one(self):
        # method derives the count from the draft -> not statically recoverable.
        args = ["--speculative-config", '{"method":"eagle","model":"d"}']
        assert _parse_positions_per_step(args) == 1

    def test_at_file_is_one(self):
        # -sc @file.json is not inline JSON -> conservative fallback.
        args = ["--speculative-config", "@/path/to/spec.json"]
        assert _parse_positions_per_step(args) == 1

    def test_count_zero_is_vanilla(self):
        args = ["-sc", '{"num_speculative_tokens":0}']
        assert _parse_positions_per_step(args) == 1

    def test_non_int_count_is_one(self):
        args = ["-sc", '{"num_speculative_tokens":"abc"}']
        assert _parse_positions_per_step(args) == 1

    def test_non_dict_json_is_one(self):
        args = ["-sc", "[1,2,3]"]
        assert _parse_positions_per_step(args) == 1


# ---------------------------------------------------------------------------
# _compute_nsys_cudagraph_capture_sizes: spec-decode effective token counts
# ---------------------------------------------------------------------------

class TestComputeNsysCudagraphCaptureSizes:

    def test_selected_step_spec_decode_unions_nominal_and_effective_sizes(self):
        buckets = [
            {"batch_size": bs, "nsys_capture_positions_per_step": 2}
            for bs in (1, 8, 32)
        ]
        assert _compute_nsys_cudagraph_capture_sizes(buckets) == [1, 2, 8, 16, 32, 64]

    def test_vanilla_positions_per_step_keeps_nominal_sizes(self):
        buckets = [
            {"batch_size": bs, "nsys_capture_positions_per_step": 1}
            for bs in (1, 8, 32)
        ]
        assert _compute_nsys_cudagraph_capture_sizes(buckets) == [1, 8, 32]

    def test_without_positions_per_step_metadata_stays_nominal_only(self):
        buckets = [{"batch_size": bs} for bs in (1, 8, 32)]
        assert _compute_nsys_cudagraph_capture_sizes(buckets) == [1, 8, 32]

    def test_selected_step_wiring_uses_expanded_bucket_ppt(self):
        buckets = [
            {"input_len": 27000, "output_len": 150, "batch_size": bs}
            for bs in (1, 8, 32)
        ]
        expanded = _expand_nsys_profile_buckets(
            buckets, nsys_capture_output_steps="100%", positions_per_step=2
        )
        expected, deduped = _dedupe_nsys_profile_buckets(expanded)
        assert not deduped
        assert _compute_nsys_cudagraph_capture_sizes(expected) == [
            1, 2, 8, 16, 32, 64,
        ]

    def test_selected_step_wiring_matches_recent_session_shape(self):
        buckets = [
            {"input_len": 27000, "output_len": 150, "batch_size": bs}
            for bs in (2, 4, 8)
        ]
        expanded = _expand_nsys_profile_buckets(
            buckets, nsys_capture_output_steps="100%", positions_per_step=5
        )
        expected, deduped = _dedupe_nsys_profile_buckets(expanded)
        assert not deduped
        assert _compute_nsys_cudagraph_capture_sizes(expected) == [
            2, 4, 8, 10, 20, 40,
        ]

    def test_nsys_output_len_wiring_keeps_spec_decode_effective_sizes(self):
        buckets = [
            {"input_len": 4096, "output_len": 150, "batch_size": bs}
            for bs in (2, 4, 8)
        ]
        expanded = _expand_nsys_profile_buckets(
            buckets, nsys_output_len=45, positions_per_step=5
        )
        expected, deduped = _dedupe_nsys_profile_buckets(expanded)
        assert not deduped
        assert _compute_nsys_cudagraph_capture_sizes(expected) == [
            2, 4, 8, 10, 20, 40,
        ]

    def test_plain_nsys_wiring_keeps_spec_decode_effective_sizes(self):
        buckets = [
            {"input_len": 4096, "output_len": 150, "batch_size": bs}
            for bs in (2, 4, 8)
        ]
        expanded = _expand_nsys_profile_buckets(buckets, positions_per_step=5)
        expected, deduped = _dedupe_nsys_profile_buckets(expanded)
        assert not deduped
        assert _compute_nsys_cudagraph_capture_sizes(expected) == [
            2, 4, 8, 10, 20, 40,
        ]


# ---------------------------------------------------------------------------
# _expand_nsys_profile_buckets: output_len = window * positions_per_step
# ---------------------------------------------------------------------------

class TestExpandSpecDecode:

    def test_output_len_scales_with_ppt(self):
        """Under spec-decode the generation output_len = window * ppt (so the profiler
        arms after `window` decode worker-steps), while the capture window (delay) stays."""
        buckets = [{"input_len": 27000, "output_len": 150, "batch_size": 8}]
        expanded = _expand_nsys_profile_buckets(
            buckets,
            nsys_capture_output_steps="100%",
            positions_per_step=5,
        )
        assert expanded, "expected at least one expanded bucket"
        for b in expanded:
            w = b["nsys_capture_window_output_len"]
            assert b["nsys_capture_positions_per_step"] == 5
            assert b["output_len"] == w * 5
            # delay (window) stays a worker-step count, unchanged by ppt.
            assert b["output_len"] != w or w == 0

    def test_ppt_one_is_byte_for_byte(self):
        """ppt=1 (vanilla) -> output_len == window exactly (original behavior)."""
        buckets = [{"input_len": 10000, "output_len": 1000, "batch_size": 8}]
        baseline = _expand_nsys_profile_buckets(
            buckets, nsys_capture_output_steps="2,50%,100%"
        )
        explicit = _expand_nsys_profile_buckets(
            buckets, nsys_capture_output_steps="2,50%,100%", positions_per_step=1
        )
        for b in baseline:
            assert b["output_len"] == b["nsys_capture_window_output_len"]
        # Default and explicit ppt=1 produce identical output_len.
        assert [b["output_len"] for b in baseline] == [
            b["output_len"] for b in explicit
        ]

    def test_total_worker_steps_reach_arm_threshold(self):
        """The whole point: n_prefill + ceil(output_len/ppt) >= window for every bucket
        and every batch size, so the CUDA profiler arms (counter hits delay==window)."""
        import math
        buckets = [
            {"input_len": 27000, "output_len": 150, "batch_size": bs}
            for bs in (1, 8, 32)
        ]
        expanded = _expand_nsys_profile_buckets(
            buckets, nsys_capture_output_steps="100%", positions_per_step=5
        )
        for b in expanded:
            w = b["nsys_capture_window_output_len"]
            n_prefill = _n_prefill_worker_steps(
                b["input_len"], b["batch_size"], CHUNK
            )
            available = n_prefill + math.ceil(b["output_len"] / 5)
            assert available >= w, (
                f"bs={b['batch_size']} available={available} < window={w}"
            )


# ---------------------------------------------------------------------------
# evaluate_decode_shape: spec-decode-aware acceptance + gridX demotion
# ---------------------------------------------------------------------------

class TestEvaluateDecodeShapeSpec:

    def test_accepts_real_spec_marker(self):
        """The exact failed-artifact marker: context_0(0)_generation_10(50), bs=1, ppt=5.
        gen=50 == gen_reqs(10) * ppt(5); active-request count 10 != nominal bs 1 is OK."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_10(50)",
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=1,
            positions_per_step=5,
        )
        assert ok is True, reason

    def test_gridx_demoted_under_spec_when_marker_confirms(self):
        """Fused RMSNorm gridX=800 must NOT false-reject a marker-confirmed spec decode."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_10(50)",
            rmsnorm_grid_x=800,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=1,
            positions_per_step=5,
        )
        assert ok is True, reason
        assert "not scored" in reason

    def test_spec_true_prefill_still_rejected(self):
        """ctx>0 (real prefill chunk) is rejected even under spec-decode."""
        ok, reason = evaluate_decode_shape(
            nvtx_marker="execute_context_2(16383)_generation_10(50)",
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=1,
            positions_per_step=5,
        )
        assert ok is False
        assert "PREFILL" in reason

    def test_spec_non_multiple_of_ppt_rejected(self):
        """gen not a multiple of ppt is not a clean spec decode step."""
        ok, _ = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_10(47)",
            rmsnorm_grid_x=None,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=1,
            positions_per_step=5,
        )
        assert ok is False

    def test_vanilla_predicate_unchanged_by_default(self):
        """ppt default (1): the EXACT old gen==batch_size predicate and gridX vote."""
        ok, _ = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_8(8)",
            rmsnorm_grid_x=8,
            paged_kv_attn_total_us=950.0,
            prefill_flash_attn_total_us=30.0,
            batch_size=8,
        )
        assert ok is True
        # The exact vanilla decode marker also demotes the RMSNorm-name grid heuristic.
        ok2, _ = evaluate_decode_shape(
            nvtx_marker="execute_context_0(0)_generation_8(8)",
            rmsnorm_grid_x=16384,
            paged_kv_attn_total_us=None,
            prefill_flash_attn_total_us=None,
            batch_size=8,
        )
        assert ok2 is True


# ---------------------------------------------------------------------------
# _apply_selected_step_profiler_config: arm-reachability under spec-decode
# ---------------------------------------------------------------------------

class TestArmReachability:

    def test_arms_when_output_len_sized_for_spec(self):
        # window floored child-wide (=60 for the il=27000 workload) clears prefill;
        # output_len sized = window * ppt so the profiler arms.
        window = 60
        ea = {}
        bucket = {
            "input_len": 27000,
            "batch_size": 8,
            "nsys_capture_window_output_len": window,
            "nsys_capture_output_step": 150,
            "nsys_capture_positions_per_step": 5,
            "output_len": window * 5,
        }
        _apply_selected_step_profiler_config(ea, bucket)
        assert ea["profiler_config"]["delay_iterations"] == window

    def test_fails_loud_when_generation_too_short(self):
        """An under-sized output_len (window tokens, not window*ppt) under spec-decode
        runs fewer worker-steps than `window` -> the profiler would never arm -> SystemExit."""
        window = 60
        ea = {}
        bucket = {
            "input_len": 27000,
            "batch_size": 8,
            "nsys_capture_window_output_len": window,
            "nsys_capture_output_step": 150,
            "nsys_capture_positions_per_step": 5,
            # BUG shape: output_len == window (not window*ppt) -> only n_prefill+ceil(60/5)
            # = n_prefill+12 worker-steps, below window=60.
            "output_len": window,
        }
        with pytest.raises(SystemExit, match="never arm"):
            _apply_selected_step_profiler_config(ea, bucket)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
