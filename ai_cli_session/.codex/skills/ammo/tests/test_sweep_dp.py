#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for DP/EP sweep-script integration (Section 4 of the spec).

Covers:
- _parse_parallelism_from_args: standalone argparse of tp/pp/dp/prefill_cp/backend/beam_search
- _resolve_parallelism_and_backend: per-label validation + auto-inject backend
- _should_skip_bucket / _partition_prompts / _stitch_gathered: SPMD helpers
- _build_child_cmd: torchrun dual-path command construction
- rank-0 gating (_IS_RANK0, _rank0_*): reimport-based rank twiddle
- nsys rank-0 gating helpers

These tests do NOT require vLLM to be importable — helpers use standalone argparse.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Task 1: _parse_parallelism_from_args
# ---------------------------------------------------------------------------


class TestParseParallelism:
    """Standalone argparse of tp/pp/dp/prefill_cp/backend/beam_search."""

    def _call(self, args_list):
        from run_vllm_bench_latency_sweep import _parse_parallelism_from_args
        return _parse_parallelism_from_args(args_list)

    def test_no_flags_defaults(self):
        r = self._call([])
        assert r == {
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "prefill_cp": 1,
            "distributed_backend": None,
            "use_beam_search": False,
        }

    def test_tp_dp_pp_extracted(self):
        r = self._call([
            "--tensor-parallel-size", "4",
            "--pipeline-parallel-size", "2",
            "--data-parallel-size", "2",
        ])
        assert r["tp"] == 4
        assert r["pp"] == 2
        assert r["dp"] == 2

    def test_prefill_cp_captured(self):
        r = self._call(["--prefill-context-parallel-size", "2"])
        assert r["prefill_cp"] == 2

    def test_distributed_backend_captured(self):
        r = self._call(["--distributed-executor-backend", "mp"])
        assert r["distributed_backend"] == "mp"

    def test_external_launcher_captured(self):
        r = self._call(["--distributed-executor-backend", "external_launcher"])
        assert r["distributed_backend"] == "external_launcher"

    def test_beam_search_captured(self):
        r = self._call(["--use-beam-search"])
        assert r["use_beam_search"] is True

    def test_unknown_flags_ignored(self):
        """Parser must not trip on unrelated vLLM flags (json-style etc.)."""
        r = self._call([
            "-cc.pass_config.enable_sp=false",
            "--enable-expert-parallel",
            "--data-parallel-size", "2",
            "--some-unknown-flag", "value",
        ])
        assert r["dp"] == 2
        # no raise


# ---------------------------------------------------------------------------
# Task 1: _resolve_parallelism_and_backend
# ---------------------------------------------------------------------------


class TestResolveParallelism:

    def _call(self, extra=None, baseline=None, opt=None):
        from run_vllm_bench_latency_sweep import _resolve_parallelism_and_backend
        return _resolve_parallelism_and_backend(
            extra_args=extra or [],
            baseline_extra_args=baseline or [],
            opt_extra_args=opt or [],
        )

    def test_equal_agreement_dp1(self):
        extra_out, p = self._call(extra=["--tensor-parallel-size", "2"])
        assert p["dp"] == 1
        assert p["tp"] == 2
        assert p["nproc"] == 2
        # No auto-inject when dp=1
        assert "--distributed-executor-backend" not in extra_out

    def test_equal_agreement_dp2_auto_injects_backend(self):
        extra_out, p = self._call(extra=[
            "--tensor-parallel-size", "2",
            "--data-parallel-size", "2",
        ])
        assert p["dp"] == 2
        assert p["tp"] == 2
        assert p["nproc"] == 4
        assert "--distributed-executor-backend" in extra_out
        idx = extra_out.index("--distributed-executor-backend")
        assert extra_out[idx + 1] == "external_launcher"

    def test_explicit_external_launcher_no_reinjection(self):
        extra_out, p = self._call(extra=[
            "--data-parallel-size", "2",
            "--distributed-executor-backend", "external_launcher",
        ])
        # backend already set → no duplication
        assert extra_out.count("--distributed-executor-backend") == 1

    def test_dp_gt_1_bad_backend_in_opt_raises(self):
        with pytest.raises(SystemExit, match="external_launcher"):
            self._call(
                extra=["--data-parallel-size", "2"],
                opt=["--distributed-executor-backend", "mp"],
            )

    def test_dp_gt_1_bad_backend_in_baseline_raises(self):
        with pytest.raises(SystemExit, match="external_launcher"):
            self._call(
                extra=["--data-parallel-size", "2"],
                baseline=["--distributed-executor-backend", "mp"],
            )

    def test_dp1_bad_backend_allowed(self):
        """DP=1 permits any backend — no coupling."""
        extra_out, p = self._call(
            extra=["--distributed-executor-backend", "mp"],
        )
        assert p["dp"] == 1
        # Unchanged
        assert extra_out == ["--distributed-executor-backend", "mp"]

    def test_per_label_dp_divergence_raises(self):
        """Baseline has DP=2 in its label args; opt doesn't → raise."""
        with pytest.raises(SystemExit, match="per-label parallelism"):
            self._call(
                baseline=["--data-parallel-size", "2"],
                opt=[],
            )

    def test_per_label_tp_divergence_raises(self):
        with pytest.raises(SystemExit, match="per-label parallelism"):
            self._call(
                baseline=["--tensor-parallel-size", "2"],
                opt=["--tensor-parallel-size", "4"],
            )

    def test_prefill_cp_nonzero_raises(self):
        with pytest.raises(SystemExit, match="prefill_context_parallel_size"):
            self._call(extra=["--prefill-context-parallel-size", "2"])

    def test_beam_search_plus_dp_raises(self):
        with pytest.raises(SystemExit, match="beam"):
            self._call(
                extra=["--data-parallel-size", "2"],
                baseline=["--use-beam-search"],
            )

    def test_beam_search_dp1_ok(self):
        """Legacy DP=1 beam-search path is preserved."""
        extra_out, p = self._call(extra=["--use-beam-search"])
        assert p["dp"] == 1
        # no raise

    def test_nproc_computed_tp_pp_dp(self):
        _, p = self._call(extra=[
            "--tensor-parallel-size", "2",
            "--pipeline-parallel-size", "2",
            "--data-parallel-size", "2",
        ])
        assert p["nproc"] == 8  # 2*2*2

    def test_per_label_extras_also_inherit_extra_args(self):
        """Agreement is on the EFFECTIVE per-label list (extra + label_extra)."""
        # Both labels effectively have dp=2 because extra_args has it.
        extra_out, p = self._call(
            extra=["--data-parallel-size", "2"],
            baseline=["--enable-expert-parallel"],
            opt=[],  # dp still from extra_args
        )
        assert p["dp"] == 2


# ---------------------------------------------------------------------------
# Task 2: Rank-0 gating (_IS_RANK0, _rank0_*, _NullFile)
# ---------------------------------------------------------------------------


def _reload_with_rank(rank: str):
    """Reload the sweep module with RANK env set to *rank*.

    Tests must access rank-gated globals via `mod._IS_RANK0` attribute access,
    not via `from run_vllm_bench_latency_sweep import _IS_RANK0` — the `from`
    import binds the old constant, and reloading does not update bound names.
    """
    os.environ["RANK"] = rank
    if "run_vllm_bench_latency_sweep" in sys.modules:
        mod = importlib.reload(sys.modules["run_vllm_bench_latency_sweep"])
    else:
        import run_vllm_bench_latency_sweep as mod
    return mod


@pytest.fixture(autouse=False)
def reset_rank_after(monkeypatch):
    """Test-local fixture: restore RANK=0 and reload after each rank-twiddle test."""
    yield
    os.environ["RANK"] = "0"
    if "run_vllm_bench_latency_sweep" in sys.modules:
        importlib.reload(sys.modules["run_vllm_bench_latency_sweep"])


class TestRank0Gating:
    """Rank-0 gating uses a module-level constant captured at import.

    Tests must go through `_reload_with_rank` to twiddle RANK — bound imports
    (`from ... import _IS_RANK0`) will not update on reload.
    """

    def test_rank0_default_true_when_unset(self, monkeypatch, reset_rank_after):
        monkeypatch.delenv("RANK", raising=False)
        # Force reload without setting RANK to probe the "unset" path.
        if "run_vllm_bench_latency_sweep" in sys.modules:
            mod = importlib.reload(sys.modules["run_vllm_bench_latency_sweep"])
        else:
            import run_vllm_bench_latency_sweep as mod
        assert mod._IS_RANK0 is True
        assert mod._GLOBAL_RANK == 0

    def test_rank0_true_on_zero(self, reset_rank_after):
        mod = _reload_with_rank("0")
        assert mod._IS_RANK0 is True
        assert mod._GLOBAL_RANK == 0

    def test_rank0_false_on_nonzero(self, reset_rank_after):
        mod = _reload_with_rank("3")
        assert mod._IS_RANK0 is False
        assert mod._GLOBAL_RANK == 3

    def test_rank0_write_json_noop_when_not_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("2")
        target = tmp_path / "ignored.json"
        mod._rank0_write_json(target, {"a": 1})
        assert not target.exists()

    def test_rank0_write_json_writes_when_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("0")
        target = tmp_path / "written.json"
        mod._rank0_write_json(target, {"a": 1})
        assert target.exists()
        assert json.loads(target.read_text())["a"] == 1

    def test_rank0_write_json_atomic_noop_when_not_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("1")
        target = tmp_path / "atomic.json"
        mod._rank0_write_json_atomic(target, {"b": 2})
        assert not target.exists()

    def test_rank0_write_text_noop_when_not_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("1")
        target = tmp_path / "txt.log"
        mod._rank0_write_text(target, "hello")
        assert not target.exists()

    def test_rank0_write_text_writes_when_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("0")
        target = tmp_path / "txt.log"
        mod._rank0_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_rank0_log_silent_when_not_rank0(self, capsys, reset_rank_after):
        mod = _reload_with_rank("2")
        mod._rank0_log("should be silent")
        out = capsys.readouterr().out
        assert "should be silent" not in out

    def test_rank0_log_prints_when_rank0(self, capsys, reset_rank_after):
        mod = _reload_with_rank("0")
        mod._rank0_log("should appear")
        out = capsys.readouterr().out
        assert "should appear" in out

    def test_null_file_swallows_writes(self, reset_rank_after):
        mod = _reload_with_rank("1")
        nf = mod._NullFile()
        assert nf.write("anything") == 0
        nf.flush()
        nf.close()
        # Context manager works too.
        with mod._NullFile() as f:
            assert f.write("ok") == 0

    def test_rank0_open_log_returns_null_file_when_not_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("1")
        target = tmp_path / "child.log"
        f = mod._rank0_open_log(target, "w")
        try:
            # _NullFile instance — writes are no-ops.
            f.write("data")
            f.flush()
        finally:
            f.close()
        assert not target.exists()

    def test_rank0_open_log_returns_real_file_when_rank0(self, tmp_path, reset_rank_after):
        mod = _reload_with_rank("0")
        target = tmp_path / "child.log"
        with mod._rank0_open_log(target, "w") as f:
            f.write("hello")
        assert target.read_text() == "hello"


# ---------------------------------------------------------------------------
# Task 4: _build_child_cmd (torchrun dual-path) + child_timeout formula
# ---------------------------------------------------------------------------


class TestBuildChildCmd:
    """Parent-side child command construction."""

    def _call(self, **kw):
        from run_vllm_bench_latency_sweep import _build_child_cmd
        defaults = dict(
            python_exe="/venv/bin/python",
            script_path=Path("/sweep.py"),
            run_label="baseline",
            artifact_dir=Path("/art"),
            target_path=Path("/art/target.json"),
            timeout_s=1800,
            out_name="e2e",
            out_root=Path("/art/e2e"),
            dp=1,
            nproc=1,
            extra_child_flags=[],
        )
        defaults.update(kw)
        return _build_child_cmd(**defaults)

    def test_dp1_plain_python_no_torchrun(self):
        cmd = self._call(dp=1, nproc=1)
        assert "torch.distributed.run" not in cmd
        assert cmd[0] == "/venv/bin/python"
        assert cmd[1] == "/sweep.py"

    def test_dp2_prepends_torchrun(self):
        cmd = self._call(dp=2, nproc=4)  # e.g. tp=2*dp=2
        assert cmd[0] == "/venv/bin/python"
        assert cmd[1] == "-m"
        assert cmd[2] == "torch.distributed.run"
        assert cmd[3] == "--standalone"
        assert cmd[4] == "--nnodes=1"
        assert "--nproc-per-node" in cmd
        nproc_idx = cmd.index("--nproc-per-node")
        assert cmd[nproc_idx + 1] == "4"
        # Script path appears AFTER torchrun args
        assert "/sweep.py" in cmd
        script_idx = cmd.index("/sweep.py")
        assert script_idx > nproc_idx

    def test_child_label_flag_preserved(self):
        for dp in (1, 2):
            cmd = self._call(dp=dp, nproc=dp, run_label="opt")
            idx = cmd.index("--_child-label")
            assert cmd[idx + 1] == "opt"

    def test_extra_child_flags_appended(self):
        flags = ["--_nsys-profile", "--_correctness-num-questions", "50"]
        cmd = self._call(dp=1, nproc=1, extra_child_flags=flags)
        # All forwarded to the child args tail
        for f in flags:
            assert f in cmd
        # Preserve order
        assert cmd.index("--_nsys-profile") < cmd.index("--_correctness-num-questions")

    def test_standard_args_present(self):
        cmd = self._call(dp=1)
        assert "--artifact-dir" in cmd
        assert "/art" in cmd
        assert "--target-json" in cmd
        assert "/art/target.json" in cmd
        assert "--timeout-s" in cmd
        assert "1800" in cmd
        # Current sweep behavior carries the fully resolved output path through
        # --_out-root; the legacy --out-name flag is not forwarded to children.
        assert "--out-name" not in cmd
        assert "--_out-root" in cmd
        assert "/art/e2e" in cmd

    def test_dp2_large_nproc(self):
        # tp=4 * pp=2 * dp=2 = 16
        cmd = self._call(dp=2, nproc=16)
        idx = cmd.index("--nproc-per-node")
        assert cmd[idx + 1] == "16"


class TestChildTimeoutFormula:
    """Verify the dp-aware timeout formula (spec §4.2)."""

    def _formula(self, *, timeout_s, buckets, dp, nsys=False, nsys_timeout_s=600):
        # Mirrors the parent's computation at main() near L2116.
        child_timeout = int(timeout_s) * max(1, buckets) + 1800
        if nsys:
            child_timeout = nsys_timeout_s * max(1, buckets)
        if dp > 1:
            child_timeout += 60 * dp
        return child_timeout

    def test_dp1_plain_unchanged(self):
        assert self._formula(timeout_s=1800, buckets=3, dp=1) == 1800 * 3 + 1800

    def test_dp2_adds_120(self):
        base = 1800 * 3 + 1800
        assert self._formula(timeout_s=1800, buckets=3, dp=2) == base + 120

    def test_dp4_adds_240(self):
        base = 1800 * 3 + 1800
        assert self._formula(timeout_s=1800, buckets=3, dp=4) == base + 240

    def test_nsys_dp_stacks(self):
        # With nsys: 600 * 3 = 1800; + 60*dp for dp>1.
        assert self._formula(
            timeout_s=1800, buckets=3, dp=2, nsys=True, nsys_timeout_s=600
        ) == 1800 + 120


# ---------------------------------------------------------------------------
# Task 6: _should_skip_bucket / _partition_prompts (SPMD partitioning)
# ---------------------------------------------------------------------------


class TestShouldSkipBucket:
    """Skip buckets where batch_size < dp_size (cannot partition across ranks)."""

    def _call(self, bs, dp):
        from run_vllm_bench_latency_sweep import _should_skip_bucket
        return _should_skip_bucket(bs, dp)

    def test_dp1_never_skips(self):
        for bs in (1, 2, 4, 8, 128):
            assert self._call(bs, 1) is False

    def test_bs_lt_dp_skips(self):
        assert self._call(2, 4) is True
        assert self._call(1, 2) is True
        assert self._call(3, 4) is True

    def test_bs_eq_dp_does_not_skip(self):
        assert self._call(4, 4) is False
        assert self._call(2, 2) is False

    def test_bs_gt_dp_does_not_skip(self):
        assert self._call(8, 4) is False
        assert self._call(16, 2) is False


class TestPartitionPrompts:
    """Contract: returns this rank's share of prompts (always non-empty list)."""

    def _call(self, all_prompts, *, dp_size, dp_rank, input_len=64):
        from run_vllm_bench_latency_sweep import _partition_prompts
        return _partition_prompts(
            all_prompts, dp_size=dp_size, dp_rank=dp_rank, input_len=input_len
        )

    def test_dp1_identity(self):
        prompts = [{"prompt_token_ids": [i] * 8} for i in range(5)]
        out = self._call(prompts, dp_size=1, dp_rank=0)
        assert out == prompts

    def test_dp2_rank0_evens(self):
        prompts = [{"prompt_token_ids": [i]} for i in range(8)]
        out = self._call(prompts, dp_size=2, dp_rank=0)
        # indices 0, 2, 4, 6
        assert [p["prompt_token_ids"][0] for p in out] == [0, 2, 4, 6]

    def test_dp2_rank1_odds(self):
        prompts = [{"prompt_token_ids": [i]} for i in range(8)]
        out = self._call(prompts, dp_size=2, dp_rank=1)
        assert [p["prompt_token_ids"][0] for p in out] == [1, 3, 5, 7]

    def test_dp4_bs5_rank0_gets_two(self):
        # 5 prompts, dp=4 → rank 0 gets indices [0, 4]; ranks 1..3 get one each.
        prompts = [{"prompt_token_ids": [i]} for i in range(5)]
        out0 = self._call(prompts, dp_size=4, dp_rank=0)
        assert [p["prompt_token_ids"][0] for p in out0] == [0, 4]

    def test_dp4_bs5_other_ranks_get_one(self):
        prompts = [{"prompt_token_ids": [i]} for i in range(5)]
        for r in (1, 2, 3):
            out = self._call(prompts, dp_size=4, dp_rank=r)
            assert [p["prompt_token_ids"][0] for p in out] == [r]

    def test_empty_rank_gets_placeholder(self):
        # With 3 prompts and dp=4, rank 3 has no real prompt → placeholder.
        prompts = [{"prompt_token_ids": [i]} for i in range(3)]
        out = self._call(prompts, dp_size=4, dp_rank=3, input_len=16)
        assert len(out) == 1
        # Placeholder uses [1] * input_len
        assert out[0] == {"prompt_token_ids": [1] * 16}

    def test_dp2_does_not_mutate_input(self):
        prompts = [{"prompt_token_ids": [i]} for i in range(4)]
        snapshot = [dict(p) for p in prompts]
        _ = self._call(prompts, dp_size=2, dp_rank=0)
        assert prompts == snapshot


class TestStitchGathered:
    """Reassemble per-rank outputs back into canonical prompt order."""

    def _call(self, gathered, n_total, dp_size):
        from run_vllm_bench_latency_sweep import _stitch_gathered
        return _stitch_gathered(gathered, n_total, dp_size)

    def test_dp1_identity(self):
        outputs = ["o0", "o1", "o2"]
        assert self._call([outputs], n_total=3, dp_size=1) == outputs

    def test_dp2_interleaves_even(self):
        # Partition of 6 across dp=2: rank0=[o0,o2,o4], rank1=[o1,o3,o5].
        gathered = [["o0", "o2", "o4"], ["o1", "o3", "o5"]]
        assert self._call(gathered, n_total=6, dp_size=2) == [
            "o0", "o1", "o2", "o3", "o4", "o5",
        ]

    def test_dp4_uneven(self):
        # 10 prompts across dp=4:
        #   rank 0: indices 0,4,8
        #   rank 1: indices 1,5,9
        #   rank 2: indices 2,6
        #   rank 3: indices 3,7
        gathered = [
            ["o0", "o4", "o8"],
            ["o1", "o5", "o9"],
            ["o2", "o6"],
            ["o3", "o7"],
        ]
        out = self._call(gathered, n_total=10, dp_size=4)
        assert out == [f"o{i}" for i in range(10)]

    def test_dp2_n3_matches_partition(self):
        # 3 prompts, dp=2: rank 0 = [0, 2], rank 1 = [1].
        gathered = [["o0", "o2"], ["o1"]]
        assert self._call(gathered, n_total=3, dp_size=2) == ["o0", "o1", "o2"]


# ---------------------------------------------------------------------------
# Task 7: GSM8K DP-correctness precondition + verdict code mapping
# ---------------------------------------------------------------------------


class TestCorrectnessPrecondition:
    """_check_correctness_dp_precondition rejects too-few questions for DP>1."""

    def _call(self, *, num_q, dp):
        from run_vllm_bench_latency_sweep import _check_correctness_dp_precondition
        return _check_correctness_dp_precondition(num_q, dp)

    def test_dp1_any_num_q_ok(self):
        # DP=1: never raises (even nq=0, which wouldn't hit this path anyway).
        self._call(num_q=1, dp=1)
        self._call(num_q=200, dp=1)

    def test_dp2_nq_lt_dp_raises(self):
        with pytest.raises(ValueError, match="dp_size"):
            self._call(num_q=1, dp=2)

    def test_dp4_nq_eq_dp_ok(self):
        # Exactly equal means every rank gets ≥1 real prompt — allowed.
        self._call(num_q=4, dp=4)

    def test_dp4_nq_gt_dp_ok(self):
        self._call(num_q=200, dp=4)


class TestVerdictCodeMapping:
    """_verdict_to_code maps correctness verdict dicts → int for broadcast."""

    def _call(self, verdict):
        from run_vllm_bench_latency_sweep import _verdict_to_code
        return _verdict_to_code(verdict)

    def test_pass_is_zero(self):
        assert self._call({"verdict": "PASS"}) == 0

    def test_fail_is_three(self):
        assert self._call({"verdict": "FAIL"}) == 3

    def test_infrastructure_error_is_four(self):
        assert self._call({
            "verdict": "FAIL",
            "infrastructure_error": True,
        }) == 4

    def test_infrastructure_error_even_on_pass(self):
        # Infra error wins over the verdict string (defensive).
        assert self._call({
            "verdict": "PASS",
            "infrastructure_error": True,
        }) == 4

    def test_missing_verdict_is_four(self):
        # Defensive: a malformed verdict dict counts as infra error.
        assert self._call({}) == 4


# ---------------------------------------------------------------------------
# Task 8: nsys rank-0 gating helpers
# ---------------------------------------------------------------------------


class _FakeCudart:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    def cudaProfilerStart(self):
        self.start_calls += 1

    def cudaProfilerStop(self):
        self.stop_calls += 1


class _FakeCuda:
    def __init__(self):
        self._cudart = _FakeCudart()
        self.sync_calls = 0

    def cudart(self):
        return self._cudart

    def synchronize(self):
        self.sync_calls += 1


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class TestNsysRankGating:
    """cudaProfilerStart/Stop on rank 0 only; synchronize on all ranks."""

    def _load(self, rank: str):
        os.environ["RANK"] = rank
        if "run_vllm_bench_latency_sweep" in sys.modules:
            mod = importlib.reload(sys.modules["run_vllm_bench_latency_sweep"])
        else:
            import run_vllm_bench_latency_sweep as mod
        return mod

    def test_start_rank0_calls_profiler_and_sync(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        mod = self._load("0")
        fake = _FakeTorch()
        mod._nsys_start_if_rank0(fake)
        assert fake.cuda._cudart.start_calls == 1
        assert fake.cuda.sync_calls == 1

    def test_start_nonrank0_skips_profiler_but_syncs(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        mod = self._load("2")
        fake = _FakeTorch()
        mod._nsys_start_if_rank0(fake)
        # Rank-gated: profiler NOT invoked on non-rank-0.
        assert fake.cuda._cudart.start_calls == 0
        # But synchronize must fire on ALL ranks (keeps nsys capture-range
        # aligned across the world group).
        assert fake.cuda.sync_calls == 1

    def test_stop_rank0_calls_profiler_and_sync(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        mod = self._load("0")
        fake = _FakeTorch()
        mod._nsys_stop_if_rank0(fake)
        assert fake.cuda._cudart.stop_calls == 1
        assert fake.cuda.sync_calls == 1

    def test_stop_nonrank0_skips_profiler_but_syncs(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        mod = self._load("3")
        fake = _FakeTorch()
        mod._nsys_stop_if_rank0(fake)
        assert fake.cuda._cudart.stop_calls == 0
        assert fake.cuda.sync_calls == 1

    def test_cleanup_resets_rank(self, monkeypatch):
        # Last test in class: restore rank 0 so subsequent test files
        # (or reruns) see the default module state.
        monkeypatch.setenv("RANK", "0")
        mod = self._load("0")
        assert mod._IS_RANK0 is True


# ---------------------------------------------------------------------------
# Task 9: EP (expert-parallel) pass-through regression
# ---------------------------------------------------------------------------


class TestExpertParallelPassthrough:
    """--enable-expert-parallel is a plain vLLM flag — it must flow through
    extra_args unchanged and must not break the DP resolver."""

    def test_ep_alone_no_dp_resolver_unchanged(self):
        from run_vllm_bench_latency_sweep import _resolve_parallelism_and_backend
        extra = ["--enable-expert-parallel"]
        out_extra, par = _resolve_parallelism_and_backend(extra, [], [])
        # dp=1, no auto-inject, EP flag preserved verbatim.
        assert par["dp"] == 1
        assert par["nproc"] == 1
        assert "--enable-expert-parallel" in out_extra
        # No accidental --distributed-executor-backend injection for DP=1.
        assert "--distributed-executor-backend" not in out_extra

    def test_ep_plus_dp_auto_injects_backend(self):
        from run_vllm_bench_latency_sweep import _resolve_parallelism_and_backend
        extra = [
            "--enable-expert-parallel",
            "--tensor-parallel-size", "2",
            "--data-parallel-size", "4",
        ]
        out_extra, par = _resolve_parallelism_and_backend(extra, [], [])
        assert par["tp"] == 2
        assert par["dp"] == 4
        assert par["nproc"] == 8  # tp * pp * dp
        # EP still present + external_launcher auto-injected.
        assert "--enable-expert-parallel" in out_extra
        assert "--distributed-executor-backend" in out_extra
        idx = out_extra.index("--distributed-executor-backend")
        assert out_extra[idx + 1] == "external_launcher"

    def test_ep_with_explicit_external_launcher_ok(self):
        from run_vllm_bench_latency_sweep import _resolve_parallelism_and_backend
        extra = [
            "--enable-expert-parallel",
            "--tensor-parallel-size", "2",
            "--data-parallel-size", "2",
            "--distributed-executor-backend", "external_launcher",
        ]
        out_extra, par = _resolve_parallelism_and_backend(extra, [], [])
        # Already external_launcher → no duplicate injection.
        assert out_extra.count("--distributed-executor-backend") == 1
        assert par["dp"] == 2
        assert par["nproc"] == 4


# ---------------------------------------------------------------------------
# T1 — Tolerance boundary tests (Gate 5.1b _compare_correctness)
# ---------------------------------------------------------------------------


def _make_q(token_ids, logprobs=None):
    """Synthetic question dict for _compare_correctness."""
    if logprobs is None:
        logprobs = []
    return {
        "token_ids": token_ids,
        "logprobs": logprobs,
        "text": "",
        "num_tokens": len(token_ids),
        "prompt_index": 0,
    }


def _build_correctness_inputs(n, baseline_correct_count, opt_correct_count):
    """Build golden/opt/labels/preds with a specific number correct for each."""
    assert 0 <= baseline_correct_count <= n
    assert 0 <= opt_correct_count <= n
    golden = [_make_q([i + 1]) for i in range(n)]
    opt = [_make_q([i + 1]) for i in range(n)]
    labels = list(range(n))
    baseline_preds = [
        i if i < baseline_correct_count else -1 for i in range(n)
    ]
    opt_preds = [
        i if i < opt_correct_count else -1 for i in range(n)
    ]
    return golden, opt, labels, baseline_preds, opt_preds


class TestToleranceGate:
    """Gate 5.1b tolerance boundary semantics."""

    def _call(self, *, n, baseline_correct, opt_correct, tolerance_pct=None):
        from run_vllm_bench_latency_sweep import _compare_correctness
        g, o, lbl, bp, op = _build_correctness_inputs(
            n, baseline_correct, opt_correct
        )
        kwargs = dict(
            golden_refs=g, opt_outputs=o,
            labels=lbl, baseline_preds=bp, opt_preds=op,
        )
        if tolerance_pct is not None:
            kwargs["tolerance_pct"] = tolerance_pct
        return _compare_correctness(**kwargs)

    def test_tolerance_pass_at_exact_boundary(self):
        r = self._call(n=200, baseline_correct=200, opt_correct=198,
                       tolerance_pct=1.0)
        assert r["verdict"] == "PASS"
        assert r["threshold"] == round(1.0 - 0.01, 4)
        assert r["accuracy_delta"] == round(0.99 - 1.0, 4)

    def test_tolerance_fail_just_below_boundary(self):
        r = self._call(n=200, baseline_correct=200, opt_correct=196,
                       tolerance_pct=1.0)
        assert r["verdict"] == "FAIL"

    def test_tolerance_zero_is_strict(self):
        r = self._call(n=200, baseline_correct=200, opt_correct=199,
                       tolerance_pct=0.0)
        assert r["verdict"] == "FAIL"

    def test_tolerance_zero_pass_when_equal(self):
        r = self._call(n=10, baseline_correct=5, opt_correct=5,
                       tolerance_pct=0.0)
        assert r["verdict"] == "PASS"

    def test_tolerance_default_is_one_pp(self):
        r = self._call(n=100, baseline_correct=89, opt_correct=88)
        assert r["verdict"] == "PASS"
        assert r["accuracy_delta"] == round(0.88 - 0.89, 4)
        assert r["threshold"] == round(0.89 - 0.01, 4)

    def test_tolerance_opt_above_baseline_always_pass(self):
        for tol in (0.0, 1.0, 5.0):
            r = self._call(n=10, baseline_correct=8, opt_correct=9,
                           tolerance_pct=tol)
            assert r["verdict"] == "PASS"

    def test_message_field_pass_format(self):
        r = self._call(n=200, baseline_correct=200, opt_correct=198,
                       tolerance_pct=1.0)
        msg = r["message"]
        assert "PASS:" in msg
        assert "threshold" in msg
        assert "baseline" in msg
        assert "tolerance" in msg
        assert "99.0%" in msg

    def test_message_field_fail_format(self):
        r = self._call(n=200, baseline_correct=200, opt_correct=196,
                       tolerance_pct=1.0)
        msg = r["message"]
        assert "FAIL:" in msg
        assert "(196/200)" in msg
        assert "threshold" in msg
        assert "1.0pp tolerance" in msg

    def test_tolerance_pct_field_in_return(self):
        r = self._call(n=10, baseline_correct=5, opt_correct=5,
                       tolerance_pct=2.5)
        assert r["tolerance_pct"] == 2.5

    def test_threshold_field_in_return(self):
        r = self._call(n=100, baseline_correct=89, opt_correct=88,
                       tolerance_pct=1.0)
        assert r["threshold"] == round(0.89 - 0.01, 4)

    def test_infrastructure_error_overrides_tolerance(self):
        r = self._call(n=10, baseline_correct=0, opt_correct=5,
                       tolerance_pct=100.0)
        assert r.get("infrastructure_error") is True
        assert r["verdict"] == "FAIL"

    def test_all_empty_outputs_fail_overrides_tolerance(self):
        from run_vllm_bench_latency_sweep import _compare_correctness
        n = 4
        golden = [_make_q([]) for _ in range(n)]
        opt = [_make_q([]) for _ in range(n)]
        labels = list(range(n))
        baseline_preds = list(range(n))
        opt_preds = list(range(n))
        r = _compare_correctness(
            golden_refs=golden, opt_outputs=opt,
            labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
            tolerance_pct=100.0,
        )
        assert r["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# T2 — GSM8k loader tests (_load_gsm8k_data two-tier + fail-loud)
# ---------------------------------------------------------------------------


class TestGsm8kLoader:
    """Fail-loud two-tier lookup: subset for nq<=200, full for nq>200."""

    def _make_full(self, path, test_count=1319, train_count=5):
        data = {
            "metadata": {
                "source": "test",
                "train_count": train_count,
                "test_count": test_count,
            },
            "train": [
                {"question": f"train-q{i}", "answer": f"reasoning #### {i + 1}"}
                for i in range(train_count)
            ],
            "test": [
                {"question": f"test-q{i}", "answer": f"reasoning #### {i + 1}"}
                for i in range(test_count)
            ],
        }
        path.write_text(json.dumps(data))

    def _make_subset(self, path, test_count=200, train_count=5):
        self._make_full(path, test_count=test_count, train_count=train_count)

    def test_loader_uses_subset_when_nq_le_200_and_subset_exists(
        self, tmp_path, monkeypatch
    ):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        self._make_subset(subset)
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        train, test = mod._load_gsm8k_data(50)
        assert len(train) == 5
        assert len(test) == 50

    def test_loader_uses_full_when_nq_gt_200_and_full_exists(
        self, tmp_path, monkeypatch
    ):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        self._make_full(full, test_count=1319)
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        train, test = mod._load_gsm8k_data(1319)
        assert len(train) == 5
        assert len(test) == 1319

    def test_loader_fails_loud_when_no_bundled_data(self, tmp_path, monkeypatch):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        with pytest.raises(FileNotFoundError) as exc:
            mod._load_gsm8k_data(1319)
        assert str(full) in str(exc.value) or "gsm8k_full" in str(exc.value)

    def test_loader_fails_loud_when_nq_gt_200_no_full(
        self, tmp_path, monkeypatch
    ):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        self._make_subset(subset)
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        with pytest.raises(FileNotFoundError):
            mod._load_gsm8k_data(500)

    def test_loader_no_network_fallback(self, tmp_path, monkeypatch):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        with pytest.raises(FileNotFoundError):
            mod._load_gsm8k_data(1319)
        assert not hasattr(mod, "_download_and_cache_file")
        assert not hasattr(mod, "_read_jsonl")

    def test_loader_fails_when_nq_gt_1319(self, tmp_path, monkeypatch):
        import run_vllm_bench_latency_sweep as mod
        subset = tmp_path / "gsm8k_subset.json"
        full = tmp_path / "gsm8k_full.json"
        self._make_full(full, test_count=1319)
        monkeypatch.setattr(mod, "_GSM8K_SUBSET_PATH", subset)
        monkeypatch.setattr(mod, "_GSM8K_FULL_PATH", full)
        with pytest.raises(FileNotFoundError):
            mod._load_gsm8k_data(2000)


# ---------------------------------------------------------------------------
# T3 — CLI argparse defaults + _run_inproc_latency_sweep_child signature
# ---------------------------------------------------------------------------


class TestCorrectnessCliDefaults:
    """Verify default=1319 and tolerance arg default=1.0."""

    def _source(self):
        import run_vllm_bench_latency_sweep as mod
        return Path(mod.__file__).read_text(encoding="utf-8")

    def test_cli_public_correctness_num_questions_default_is_1319(self):
        assert (
            'p.add_argument("--correctness-num-questions", type=int, default=1319'
            in self._source()
        )

    def test_cli_hidden_correctness_num_questions_default_1319(self):
        assert (
            'p.add_argument("--_correctness-num-questions", type=int, default=1319'
            in self._source()
        )

    def test_cli_public_correctness_tolerance_pct_default_is_1_0(self):
        assert (
            'p.add_argument("--correctness-tolerance-pct", type=float, default=1.0'
            in self._source()
        )

    def test_cli_hidden_correctness_tolerance_pct_default_1_0(self):
        assert (
            'p.add_argument("--_correctness-tolerance-pct", type=float, default=1.0'
            in self._source()
        )

    def test_cli_fallback_getattr_sentinel_is_1319(self):
        assert 'getattr(args, "_correctness_num_questions", 1319)' in self._source()

    def test_cli_fallback_getattr_tolerance_is_1_0(self):
        assert 'getattr(args, "_correctness_tolerance_pct", 1.0)' in self._source()

    def test_run_inproc_signature_nq_default_1319(self):
        from run_vllm_bench_latency_sweep import _run_inproc_latency_sweep_child
        sig = inspect.signature(_run_inproc_latency_sweep_child)
        assert sig.parameters["correctness_num_questions"].default == 1319

    def test_run_inproc_signature_has_tolerance_kwarg(self):
        from run_vllm_bench_latency_sweep import _run_inproc_latency_sweep_child
        sig = inspect.signature(_run_inproc_latency_sweep_child)
        assert "correctness_tolerance_pct" in sig.parameters
        assert sig.parameters["correctness_tolerance_pct"].default == 1.0


# ---------------------------------------------------------------------------
# T4 — Parent→child forwarding helper (_build_correctness_child_flags)
# ---------------------------------------------------------------------------


class TestBuildCorrectnessChildFlags:
    def _call(self, *, capture, verify, num_questions, tolerance_pct):
        from run_vllm_bench_latency_sweep import _build_correctness_child_flags
        return _build_correctness_child_flags(
            capture=capture,
            verify=verify,
            num_questions=num_questions,
            tolerance_pct=tolerance_pct,
        )

    def test_extra_child_flags_contain_tolerance_when_capture(self):
        flags = self._call(
            capture=True, verify=False,
            num_questions=1319, tolerance_pct=1.5,
        )
        assert "--_correctness-num-questions" in flags
        idx_nq = flags.index("--_correctness-num-questions")
        assert flags[idx_nq + 1] == "1319"
        assert "--_correctness-tolerance-pct" in flags
        idx_tol = flags.index("--_correctness-tolerance-pct")
        assert flags[idx_tol + 1] == "1.5"

    def test_extra_child_flags_contain_tolerance_when_verify(self):
        flags = self._call(
            capture=False, verify=True,
            num_questions=1319, tolerance_pct=2.0,
        )
        assert "--_correctness-num-questions" in flags
        assert "--_correctness-tolerance-pct" in flags

    def test_empty_list_when_neither_capture_nor_verify(self):
        flags = self._call(
            capture=False, verify=False,
            num_questions=1319, tolerance_pct=1.0,
        )
        assert flags == []


# ---------------------------------------------------------------------------
# T5 — parse_artifacts.py Pattern 0 + near-miss threshold + approximate flag
# ---------------------------------------------------------------------------


_EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "scripts"
if str(_EVAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS_DIR))


class TestParseArtifactsPattern0:
    def _extract(self, fail_reason):
        from parse_artifacts import _extract_accuracy_numbers
        return _extract_accuracy_numbers(fail_reason)

    def test_pattern0_matches_tolerance_bearing_format(self):
        fr = (
            "Gate 5.1b v2 FAIL: opt_accuracy 86.0% (1134/1319) < threshold "
            "88.0% (baseline 89.0% - 1.0pp tolerance)"
        )
        r = self._extract(fr)
        assert r is not None
        assert r["opt_accuracy_pct"] == 86.0
        assert r["opt_correct"] == 1134
        assert r["opt_total"] == 1319
        assert r["baseline_accuracy_pct"] == 89.0
        assert r["accuracy_gap_pct"] == 3.0
        assert r.get("baseline_correct_approximate") is True

    def test_pattern0_gap_uses_baseline_minus_opt(self):
        fr = (
            "Gate 5.1b v2 FAIL: opt_accuracy 88.0% (1160/1319) < threshold "
            "89.0% (baseline 90.0% - 1.0pp tolerance)"
        )
        r = self._extract(fr)
        assert r is not None
        assert r["accuracy_gap_pct"] == 2.0

    def test_pattern1_still_works_for_legacy_format(self):
        fr = (
            "opt_accuracy 89.0% (178/200) < baseline_accuracy 89.5% (179/200)"
        )
        r = self._extract(fr)
        assert r is not None
        assert r["opt_accuracy_pct"] == 89.0
        assert r["baseline_accuracy_pct"] == 89.5
        assert r["opt_correct"] == 178
        assert r["opt_total"] == 200
        assert r["baseline_correct"] == 179
        assert r["baseline_total"] == 200
        assert r.get("baseline_correct_approximate") is False

    def test_pattern0_sets_approximate_flag_true(self):
        fr = (
            "opt_accuracy 86.0% (1134/1319) < threshold 88.0% "
            "(baseline 89.0% - 1.0pp tolerance)"
        )
        r = self._extract(fr)
        assert r.get("baseline_correct_approximate") is True
        assert r.get("baseline_total") == 1319
        assert r.get("baseline_correct") == round(1319 * 89.0 / 100)

    def test_pattern1_sets_approximate_flag_false(self):
        fr = (
            "opt_accuracy 89.0% (178/200) < baseline_accuracy 89.5% (179/200)"
        )
        r = self._extract(fr)
        assert r.get("baseline_correct_approximate") is False


class TestNearMissThreshold:
    def _near_miss(self, gap):
        return (gap if gap is not None else 999) <= 1.5

    def test_near_miss_at_1_2_is_true(self):
        assert self._near_miss(1.2) is True

    def test_near_miss_at_1_6_is_false(self):
        assert self._near_miss(1.6) is False

    def test_near_miss_at_1_5_is_true(self):
        assert self._near_miss(1.5) is True

    def test_parse_artifacts_near_miss_source_uses_1_5(self):
        src_path = (
            Path(__file__).resolve().parent.parent
            / "eval" / "scripts" / "parse_artifacts.py"
        )
        src = src_path.read_text(encoding="utf-8")
        assert '<= 1.5' in src
        assert 'accuracy_gap_pct") or 999) <= 1.0' not in src


# ---------------------------------------------------------------------------
# Codex-specific NSYS bucket filtering coverage
# ---------------------------------------------------------------------------


class TestNsysBucketFiltering:
    """NSYS tagging excludes unmeasurable DP buckets and preserves metadata."""

    def test_nsys_tag_buckets_exclude_skipped_prefix(self):
        from run_vllm_bench_latency_sweep import _nsys_tag_buckets_for_dp

        buckets = [
            {"input_len": 64, "output_len": 128, "batch_size": 1},
            {"input_len": 64, "output_len": 128, "batch_size": 8},
            {"input_len": 64, "output_len": 128, "batch_size": 32},
        ]
        measured = _nsys_tag_buckets_for_dp(buckets, 2)

        assert [b["batch_size"] for b in measured] == [8, 32]

    def test_nsys_tag_buckets_apply_output_override_after_filtering(self):
        from run_vllm_bench_latency_sweep import _nsys_tag_buckets_for_dp

        buckets = [
            {"input_len": 32, "output_len": 128, "batch_size": 1},
            {"input_len": 64, "output_len": 128, "batch_size": 8},
        ]
        measured = _nsys_tag_buckets_for_dp(buckets, 4, nsys_output_len=16)

        assert len(measured) == 1
        assert measured[0]["input_len"] == 64
        assert measured[0]["output_len"] == 16
        assert measured[0]["batch_size"] == 8
