#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""TDD tests for multi-dimensional workload matrix sweeps.

Plan: .codex/plans/workload-matrix-sweep.md

Tests cover:
  Phase 1: _expand_workload_to_buckets
  Phase 2: _bucket_file_tag
  Phase 3: _render_md_table (heterogeneous columns)
  Phase 4: _validate_buckets_model_len
  Phase 7: generate_validation_report._render_e2e_section (heterogeneous rows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Phase 1: _expand_workload_to_buckets
# ---------------------------------------------------------------------------

class TestExpandWorkloadToBuckets:
    """Tests for converting workload spec → list of bucket dicts."""

    def _call(self, workload):
        from run_vllm_bench_latency_sweep import _expand_workload_to_buckets
        return _expand_workload_to_buckets(workload)

    def test_legacy_flat_format(self):
        """Legacy flat {input_len, output_len, batch_sizes} → correct buckets."""
        wl = {"input_len": 64, "output_len": 512, "batch_sizes": [1, 4, 8], "num_iters": 5}
        buckets = self._call(wl)
        assert buckets == [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 4},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]

    def test_matrix_format(self):
        """New workload_matrix format → correct buckets in order."""
        wl = {
            "input_len": 64,
            "output_len": 512,
            "batch_sizes": [1],
            "num_iters": 5,
            "workload_matrix": [
                {"input_len": 128, "output_len": 256, "batch_size": 2},
                {"input_len": 64, "output_len": 512, "batch_size": 8},
            ],
        }
        buckets = self._call(wl)
        assert buckets == [
            {"input_len": 128, "output_len": 256, "batch_size": 2},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]

    def test_matrix_overrides_flat(self):
        """When workload_matrix is present, flat batch_sizes are ignored."""
        wl = {
            "input_len": 64,
            "output_len": 512,
            "batch_sizes": [1, 4, 8],
            "num_iters": 5,
            "workload_matrix": [
                {"input_len": 128, "output_len": 256, "batch_size": 2},
            ],
        }
        buckets = self._call(wl)
        assert len(buckets) == 1
        assert buckets[0]["batch_size"] == 2

    def test_matrix_missing_keys_raises(self):
        """Each matrix entry must have input_len, output_len, batch_size."""
        wl = {
            "input_len": 64,
            "output_len": 512,
            "batch_sizes": [1],
            "num_iters": 5,
            "workload_matrix": [
                {"input_len": 128, "batch_size": 2},  # missing output_len
            ],
        }
        with pytest.raises(SystemExit):
            self._call(wl)

    def test_empty_matrix_raises(self):
        """An empty workload_matrix should raise."""
        wl = {
            "input_len": 64,
            "output_len": 512,
            "batch_sizes": [1],
            "num_iters": 5,
            "workload_matrix": [],
        }
        with pytest.raises(SystemExit):
            self._call(wl)

    def test_duplicate_tuples_raises(self):
        """Duplicate (input_len, output_len, batch_size) tuples should raise."""
        wl = {
            "input_len": 64,
            "output_len": 512,
            "batch_sizes": [1],
            "num_iters": 5,
            "workload_matrix": [
                {"input_len": 128, "output_len": 256, "batch_size": 2},
                {"input_len": 128, "output_len": 256, "batch_size": 2},
            ],
        }
        with pytest.raises(SystemExit):
            self._call(wl)

    def test_legacy_flat_no_matrix_key(self):
        """Legacy format works even when workload_matrix key is absent."""
        wl = {"input_len": 32, "output_len": 128, "batch_sizes": [16], "num_iters": 3}
        buckets = self._call(wl)
        assert len(buckets) == 1
        assert buckets[0] == {"input_len": 32, "output_len": 128, "batch_size": 16}


# ---------------------------------------------------------------------------
# Phase 2: _bucket_file_tag
# ---------------------------------------------------------------------------

class TestBucketFileTag:
    """Tests for file naming helper."""

    def _call(self, bucket, all_buckets):
        from run_vllm_bench_latency_sweep import _bucket_file_tag
        return _bucket_file_tag(bucket, all_buckets)

    def test_homogeneous_short_form(self):
        """All buckets share same IL/OL → short form bs{BS}."""
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 4},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]
        assert self._call(buckets[0], buckets) == "bs1"
        assert self._call(buckets[1], buckets) == "bs4"
        assert self._call(buckets[2], buckets) == "bs8"

    def test_heterogeneous_long_form(self):
        """Mixed IL/OL across buckets → long form il{IL}_ol{OL}_bs{BS}."""
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 128, "output_len": 256, "batch_size": 4},
        ]
        assert self._call(buckets[0], buckets) == "il64_ol512_bs1"
        assert self._call(buckets[1], buckets) == "il128_ol256_bs4"

    def test_single_bucket_short_form(self):
        """A single bucket is always homogeneous → short form."""
        buckets = [{"input_len": 64, "output_len": 512, "batch_size": 1}]
        assert self._call(buckets[0], buckets) == "bs1"


# ---------------------------------------------------------------------------
# Phase 3: _render_md_table (heterogeneous IL/OL columns)
# ---------------------------------------------------------------------------

class TestRenderMdTable:
    """Tests for MD table rendering with optional IL/OL columns."""

    def _call(self, rows, baseline_label="baseline", opt_label="opt"):
        from run_vllm_bench_latency_sweep import _render_md_table
        return _render_md_table(rows, baseline_label, opt_label)

    def test_homogeneous_omits_il_ol(self):
        """When all rows share the same IL/OL (or have no IL/OL), no IL/OL columns."""
        rows = [
            {
                "batch_size": 1,
                "input_len": 64,
                "output_len": 512,
                "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                "speedup": 1.25,
                "improvement_pct": 20.0,
            },
            {
                "batch_size": 4,
                "input_len": 64,
                "output_len": 512,
                "baseline": {"avg_s": 2.0, "fastpath_evidence": {"status": "unknown"}},
                "opt": {"avg_s": 1.5, "fastpath_evidence": {"status": "pass"}},
                "speedup": 1.333,
                "improvement_pct": 25.0,
            },
        ]
        md = self._call(rows)
        assert "Input Len" not in md
        assert "Output Len" not in md
        # Batch size column should still be present.
        assert "Batch Size" in md

    def test_heterogeneous_includes_il_ol(self):
        """When rows have different IL/OL, add Input Len and Output Len columns."""
        rows = [
            {
                "batch_size": 1,
                "input_len": 64,
                "output_len": 512,
                "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                "speedup": 1.25,
                "improvement_pct": 20.0,
            },
            {
                "batch_size": 4,
                "input_len": 128,
                "output_len": 256,
                "baseline": {"avg_s": 2.0, "fastpath_evidence": {"status": "unknown"}},
                "opt": {"avg_s": 1.5, "fastpath_evidence": {"status": "pass"}},
                "speedup": 1.333,
                "improvement_pct": 25.0,
            },
        ]
        md = self._call(rows)
        assert "Input Len" in md
        assert "Output Len" in md

    def test_legacy_rows_without_il_ol_no_crash(self):
        """Legacy rows that don't have input_len/output_len keys should not crash."""
        rows = [
            {
                "batch_size": 1,
                "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                "speedup": 1.25,
                "improvement_pct": 20.0,
            },
        ]
        md = self._call(rows)
        assert "Batch Size" in md
        # Should not crash — just omit IL/OL columns.
        assert "Input Len" not in md


# ---------------------------------------------------------------------------
# Phase 4: _validate_buckets_model_len
# ---------------------------------------------------------------------------

class TestValidateBucketsModelLen:
    """Tests for max_model_len validation against buckets."""

    def _call(self, buckets, max_model_len):
        from run_vllm_bench_latency_sweep import _validate_buckets_model_len
        return _validate_buckets_model_len(buckets, max_model_len)

    def test_invalid_bucket_raises(self):
        """Bucket with input_len + output_len > max_model_len should raise."""
        buckets = [{"input_len": 2048, "output_len": 2048, "batch_size": 1}]
        with pytest.raises(SystemExit):
            self._call(buckets, max_model_len=2048)

    def test_all_valid_passes(self):
        """Bucket with input_len + output_len <= max_model_len should pass."""
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 128, "output_len": 256, "batch_size": 4},
        ]
        # Should not raise.
        self._call(buckets, max_model_len=4096)

    def test_exact_boundary_passes(self):
        """Bucket with input_len + output_len == max_model_len should pass."""
        buckets = [{"input_len": 2048, "output_len": 2048, "batch_size": 1}]
        self._call(buckets, max_model_len=4096)


# ---------------------------------------------------------------------------
# Phase 7: generate_validation_report._render_e2e_section
# ---------------------------------------------------------------------------

class TestRenderE2ESection:
    """Tests for the validation report E2E section with heterogeneous rows."""

    def _call(self, e2e):
        from generate_validation_report import _render_e2e_section
        return _render_e2e_section(e2e)

    def test_heterogeneous_rows_render(self):
        """Heterogeneous rows (different IL/OL per row) should render IL/OL columns."""
        e2e = {
            "model_id": "test-model",
            "tp": 1,
            "max_model_len": 4096,
            "bench": {"baseline_label": "baseline", "opt_label": "opt"},
            "workload": {"num_iters": 5},
            "results": [
                {
                    "batch_size": 1,
                    "input_len": 64,
                    "output_len": 512,
                    "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                    "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                    "speedup": 1.25,
                    "improvement_pct": 20.0,
                },
                {
                    "batch_size": 4,
                    "input_len": 128,
                    "output_len": 256,
                    "baseline": {"avg_s": 2.0, "fastpath_evidence": {"status": "unknown"}},
                    "opt": {"avg_s": 1.5, "fastpath_evidence": {"status": "pass"}},
                    "speedup": 1.333,
                    "improvement_pct": 25.0,
                },
            ],
        }
        md = self._call(e2e)
        assert "Input Len" in md
        assert "Output Len" in md

    def test_legacy_rows_still_work(self):
        """Legacy rows (same IL/OL or missing IL/OL) should still render fine."""
        e2e = {
            "model_id": "test-model",
            "tp": 1,
            "max_model_len": 4096,
            "bench": {"baseline_label": "baseline", "opt_label": "opt"},
            "workload": {"input_len": 64, "output_len": 512, "num_iters": 5},
            "results": [
                {
                    "batch_size": 1,
                    "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                    "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                    "speedup": 1.25,
                    "improvement_pct": 20.0,
                },
            ],
        }
        md = self._call(e2e)
        # Should not crash. Should not include IL/OL columns for legacy data.
        assert "Batch Size" in md
        assert "Input Len" not in md

    def test_gate_logic_unchanged(self):
        """Gate logic in main() still operates on batch_size — verify section renders."""
        e2e = {
            "model_id": "test-model",
            "tp": 1,
            "max_model_len": 4096,
            "bench": {"baseline_label": "baseline", "opt_label": "opt"},
            "workload": {"input_len": 64, "output_len": 512, "num_iters": 5},
            "results": [
                {
                    "batch_size": 1,
                    "baseline": {"avg_s": 1.0, "fastpath_evidence": {"status": "unknown"}},
                    "opt": {"avg_s": 0.8, "fastpath_evidence": {"status": "pass"}},
                    "speedup": 1.25,
                    "improvement_pct": 20.0,
                },
            ],
        }
        md = self._call(e2e)
        # The section should include a table row for BS=1.
        assert "| 1 |" in md


class TestNsysProfileIntegration:
    """Tests for nsys profiling integration (prefix construction, file rename, error handling)."""

    def _build_nsys_prefix(self, buckets, label="baseline", nsys_dir="/tmp/nsys", nsys_trace="cuda"):
        """Replicate the nsys prefix construction logic from main()."""
        return [
            "nsys", "profile",
            f"--trace={nsys_trace},nvtx",
            "--sample=none",
            "--capture-range=cudaProfilerApi",
            f"--capture-range-end=repeat:{len(buckets)}",
            "--cuda-graph-trace=node",
            "--trace-fork-before-exec=true",
            "--force-overwrite=true",
            "-o", f"{nsys_dir}/{label}_profile",
        ]

    def test_nsys_prefix_repeat_count_matches_buckets(self):
        """Verify repeat:N in constructed prefix matches bucket count."""
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
            {"input_len": 64, "output_len": 512, "batch_size": 32},
        ]
        prefix = self._build_nsys_prefix(buckets)
        assert "--capture-range-end=repeat:3" in prefix
        assert "--capture-range=cudaProfilerApi" in prefix
        assert "--cuda-graph-trace=node" in prefix

    def test_nsys_prefix_single_bucket(self):
        """Single bucket should produce repeat:1."""
        buckets = [{"input_len": 64, "output_len": 512, "batch_size": 1}]
        prefix = self._build_nsys_prefix(buckets)
        assert "--capture-range-end=repeat:1" in prefix

    def test_nsys_prefix_supports_cuda_sw_trace_backend(self):
        """Blackwell (B200/B300) runs use software CUDA tracing."""
        buckets = [{"input_len": 64, "output_len": 2, "batch_size": 8}]
        prefix = self._build_nsys_prefix(buckets, nsys_trace="cuda-sw")
        assert "--trace=cuda-sw,nvtx" in prefix
        assert "--cuda-graph-trace=node" in prefix

    def test_nsys_file_rename_sequence(self, tmp_path):
        """Verify actual file rename from nsys sequential numbering to bucket tags."""
        from run_vllm_bench_latency_sweep import _bucket_file_tag

        nsys_dir = tmp_path / "nsys"
        nsys_dir.mkdir()
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]
        # Create fake nsys output files as nsys repeat mode would
        for i in range(1, len(buckets) + 1):
            (nsys_dir / f"baseline_profile.{i}.nsys-rep").write_text(f"fake-trace-{i}")

        # Replicate the parent's rename logic
        for i, bucket in enumerate(buckets, 1):
            src = nsys_dir / f"baseline_profile.{i}.nsys-rep"
            tag = _bucket_file_tag(bucket, buckets)
            dst = nsys_dir / f"baseline_{tag}.nsys-rep"
            if src.exists():
                src.rename(dst)

        # Verify renamed files exist with correct names and original content
        assert (nsys_dir / "baseline_bs1.nsys-rep").read_text() == "fake-trace-1"
        assert (nsys_dir / "baseline_bs8.nsys-rep").read_text() == "fake-trace-2"
        assert not (nsys_dir / "baseline_profile.1.nsys-rep").exists()
        assert not (nsys_dir / "baseline_profile.2.nsys-rep").exists()

    def test_nsys_file_rename_heterogeneous(self, tmp_path):
        """Verify rename uses long-form tags for heterogeneous buckets."""
        from run_vllm_bench_latency_sweep import _bucket_file_tag

        nsys_dir = tmp_path / "nsys"
        nsys_dir.mkdir()
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 128, "output_len": 256, "batch_size": 4},
        ]
        for i in range(1, len(buckets) + 1):
            (nsys_dir / f"opt_profile.{i}.nsys-rep").write_text(f"data-{i}")

        for i, bucket in enumerate(buckets, 1):
            src = nsys_dir / f"opt_profile.{i}.nsys-rep"
            tag = _bucket_file_tag(bucket, buckets)
            dst = nsys_dir / f"opt_{tag}.nsys-rep"
            if src.exists():
                src.rename(dst)

        assert (nsys_dir / "opt_il64_ol512_bs1.nsys-rep").exists()
        assert (nsys_dir / "opt_il128_ol256_bs4.nsys-rep").exists()

    def test_nsys_rename_missing_files_no_crash(self, tmp_path):
        """Rename logic should silently skip missing nsys output files."""
        from run_vllm_bench_latency_sweep import _bucket_file_tag

        nsys_dir = tmp_path / "nsys"
        nsys_dir.mkdir()
        buckets = [
            {"input_len": 64, "output_len": 512, "batch_size": 1},
            {"input_len": 64, "output_len": 512, "batch_size": 8},
        ]
        # Only create file for first bucket (simulate nsys failure on second)
        (nsys_dir / "baseline_profile.1.nsys-rep").write_text("ok")

        for i, bucket in enumerate(buckets, 1):
            src = nsys_dir / f"baseline_profile.{i}.nsys-rep"
            tag = _bucket_file_tag(bucket, buckets)
            dst = nsys_dir / f"baseline_{tag}.nsys-rep"
            if src.exists():
                src.rename(dst)

        assert (nsys_dir / "baseline_bs1.nsys-rep").exists()
        assert not (nsys_dir / "baseline_bs8.nsys-rep").exists()  # gracefully missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
