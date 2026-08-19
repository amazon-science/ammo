# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for prefill/decode metric harvest in run_vllm_bench_latency_sweep.py.


The sweep script must:
1. Pass `disable_log_stats=False` to the `LLM(...)` constructor.
2. Capture the return value of `llm.generate(...)` in the non-beam-search branch.
3. Harvest `prefill_s = first_token_ts - scheduled_ts` and
   `decode_s = last_token_ts - first_token_ts` per RequestOutput, with a
   defensive `getattr(out, "metrics", None)` check.
4. Aggregate (mean/p50) into the per-bucket raw JSON.
5. Propagate new fields through `_metrics_from_vllm_latency_json()` and
   `_build_label_result_entry()`.
6. Add `torch.cuda.synchronize()` before `t1 = _time.perf_counter()` in
   the DP=1 `run_to_completion()` for wall-time accuracy.
7. Beam-search branch must NOT capture outputs — it falls back to Tier-C.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = (
    REPO_ROOT
    / "ai_cli_session"
    / ".claude"
    / "skills"
    / "ammo"
    / "scripts"
    / "run_vllm_bench_latency_sweep.py"
)


# ---------------------------------------------------------------------------
# Module loader (the script has heavy imports; load helpers via importlib).
# ---------------------------------------------------------------------------


def _load_sweep_module():
    """Load the sweep script as a module so its pure-Python helpers are testable.

    The module's top-level imports (numpy, vllm) are skipped if unavailable —
    we only need the helpers, but the module body imports a few packages.
    Cache it to avoid repeated import cost.
    """
    cached = sys.modules.get("_run_vllm_bench_latency_sweep_test")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "_run_vllm_bench_latency_sweep_test", SWEEP_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        pytest.skip(f"Sweep script depends on missing module: {e.name}")
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load_sweep_module()


@pytest.fixture(scope="module")
def sweep_source():
    return SWEEP_SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Mock RequestOutput that mimics vLLM v1's RequestOutput.metrics interface.
# ---------------------------------------------------------------------------


class _MockMetrics:
    def __init__(self, scheduled_ts, first_token_ts, last_token_ts):
        self.scheduled_ts = scheduled_ts
        self.first_token_ts = first_token_ts
        self.last_token_ts = last_token_ts


class _MockOutput:
    def __init__(self, metrics):
        self.metrics = metrics


# ---------------------------------------------------------------------------
# 1. _request_phase_deltas / _harvest_request_metrics
#
# _request_phase_deltas is the SINGLE owner of the vLLM metrics field names;
# the live child bench loop calls it, so these are the real coverage for the
# production harvest. _harvest_request_metrics aggregates the same deltas.
# ---------------------------------------------------------------------------


class TestRequestPhaseDeltas:
    """The extractor the live child loop calls must own the field names."""

    def test_helper_exists(self, sweep):
        assert hasattr(sweep, "_request_phase_deltas"), (
            "Sweep script must expose `_request_phase_deltas(outputs)` helper"
        )

    def test_returns_per_request_deltas(self, sweep):
        outputs = [
            _MockOutput(_MockMetrics(scheduled_ts=10.0, first_token_ts=11.0, last_token_ts=15.0)),
            _MockOutput(_MockMetrics(scheduled_ts=20.0, first_token_ts=22.0, last_token_ts=30.0)),
        ]
        prefills, decodes = sweep._request_phase_deltas(outputs)
        assert prefills == pytest.approx([1.0, 2.0])
        assert decodes == pytest.approx([4.0, 8.0])

    def test_returns_empty_lists_when_no_outputs(self, sweep):
        assert sweep._request_phase_deltas(None) == ([], [])
        assert sweep._request_phase_deltas([]) == ([], [])

    def test_skips_outputs_without_usable_metrics(self, sweep):
        outputs = [
            object(),                                    # no `.metrics`
            _MockOutput(metrics=None),                   # None metrics
            _MockOutput(_MockMetrics(None, 11.0, 15.0)),  # non-numeric ts
            _MockOutput(_MockMetrics(scheduled_ts=30.0, first_token_ts=33.0, last_token_ts=45.0)),
        ]
        prefills, decodes = sweep._request_phase_deltas(outputs)
        assert prefills == pytest.approx([3.0])
        assert decodes == pytest.approx([12.0])

    def test_field_names_live_in_exactly_one_place(self, sweep_source):
        """A vLLM rename must be a one-line patch: the metrics field names may
        appear only inside _request_phase_deltas (plus prose/docstrings)."""
        for field in ("scheduled_ts", "first_token_ts", "last_token_ts"):
            getattr_sites = re.findall(
                r"getattr\(\s*\w+\s*,\s*[\"']" + field + r"[\"']", sweep_source
            )
            assert len(getattr_sites) == 1, (
                f"`{field}` is read via getattr in {len(getattr_sites)} places; "
                "the vLLM metrics coupling must live only in _request_phase_deltas"
            )


class TestHarvestRequestMetrics:
    """The sweep must expose a helper that converts a list of RequestOutput-like
    objects into per-request prefill/decode timing dict.
    """

    def test_helper_exists(self, sweep):
        assert hasattr(sweep, "_harvest_request_metrics"), (
            "Sweep script must expose `_harvest_request_metrics(outputs)` helper"
        )

    def test_returns_means_and_p50_for_well_formed_outputs(self, sweep):
        outputs = [
            _MockOutput(_MockMetrics(scheduled_ts=10.0, first_token_ts=11.0, last_token_ts=15.0)),
            _MockOutput(_MockMetrics(scheduled_ts=20.0, first_token_ts=22.0, last_token_ts=30.0)),
            _MockOutput(_MockMetrics(scheduled_ts=30.0, first_token_ts=33.0, last_token_ts=45.0)),
        ]
        # prefill_s = [1.0, 2.0, 3.0], mean=2.0, p50=2.0
        # decode_s  = [4.0, 8.0, 12.0], mean=8.0, p50=8.0
        result = sweep._harvest_request_metrics(outputs)
        assert result["prefill_avg_s"] == pytest.approx(2.0)
        assert result["decode_avg_s"] == pytest.approx(8.0)
        assert result["prefill_p50_s"] == pytest.approx(2.0)
        assert result["decode_p50_s"] == pytest.approx(8.0)

    def test_returns_empty_dict_when_outputs_is_none(self, sweep):
        assert sweep._harvest_request_metrics(None) == {}

    def test_returns_empty_dict_when_outputs_is_empty(self, sweep):
        assert sweep._harvest_request_metrics([]) == {}

    def test_handles_missing_metrics_attr_gracefully(self, sweep):
        """Defensive `getattr(out, "metrics", None)` — must not crash on objects
        that lack a `metrics` attribute (older vLLM build, beam-search outputs).
        """
        plain_obj = object()  # no `.metrics`
        assert sweep._harvest_request_metrics([plain_obj]) == {}

    def test_handles_none_metrics(self, sweep):
        outputs = [_MockOutput(metrics=None)]
        assert sweep._harvest_request_metrics(outputs) == {}

    def test_skips_outputs_with_none_metrics_partial(self, sweep):
        """If some outputs have metrics and some don't, the helper aggregates
        across only the ones with metrics."""
        outputs = [
            _MockOutput(_MockMetrics(scheduled_ts=10.0, first_token_ts=11.0, last_token_ts=15.0)),
            _MockOutput(metrics=None),
            _MockOutput(_MockMetrics(scheduled_ts=30.0, first_token_ts=33.0, last_token_ts=45.0)),
        ]
        result = sweep._harvest_request_metrics(outputs)
        # prefill: [1.0, 3.0] -> mean=2.0
        # decode:  [4.0, 12.0] -> mean=8.0
        assert result["prefill_avg_s"] == pytest.approx(2.0)
        assert result["decode_avg_s"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 2. _metrics_from_vllm_latency_json: must propagate new fields
# ---------------------------------------------------------------------------


class TestMetricsFromVllmLatencyJson:
    def test_parses_prefill_and_decode_fields(self, sweep):
        obj = {
            "avg_latency": 1.5,
            "latencies": [1.4, 1.5, 1.6],
            "percentiles": {"50": 1.5, "90": 1.59},
            "prefill_avg_s": 0.3,
            "decode_avg_s": 1.2,
            "prefill_p50_s": 0.29,
            "decode_p50_s": 1.21,
            "decode_share_of_e2e": 0.8,
        }
        m = sweep._metrics_from_vllm_latency_json(obj)
        assert m["avg_s"] == pytest.approx(1.5)
        assert m["prefill_avg_s"] == pytest.approx(0.3)
        assert m["decode_avg_s"] == pytest.approx(1.2)
        assert m["prefill_p50_s"] == pytest.approx(0.29)
        assert m["decode_p50_s"] == pytest.approx(1.21)
        assert m["decode_share_of_e2e"] == pytest.approx(0.8)

    def test_omits_new_fields_when_absent(self, sweep):
        """Backwards compatible: legacy raw JSON without prefill/decode fields
        still produces a valid metrics dict."""
        obj = {"avg_latency": 1.5, "percentiles": {"50": 1.5}}
        m = sweep._metrics_from_vllm_latency_json(obj)
        assert m["avg_s"] == pytest.approx(1.5)
        assert "prefill_avg_s" not in m
        assert "decode_avg_s" not in m
        assert "decode_share_of_e2e" not in m

    def test_ignores_non_numeric_prefill_decode(self, sweep):
        """Tier-C fallback may emit None for prefill/decode — must not crash
        and must not propagate a None into the metrics dict."""
        obj = {
            "avg_latency": 1.5,
            "prefill_avg_s": None,
            "decode_avg_s": None,
        }
        m = sweep._metrics_from_vllm_latency_json(obj)
        assert m["avg_s"] == pytest.approx(1.5)
        assert "prefill_avg_s" not in m
        assert "decode_avg_s" not in m


# ---------------------------------------------------------------------------
# 3. _build_label_result_entry: must surface prefill/decode at top of row
# ---------------------------------------------------------------------------


class TestBuildLabelResultEntryPropagation:
    def test_prefill_decode_fields_surface_at_top_level(self, sweep):
        """Per spec §3.1 step 6: prefill/decode timing is workload-property,
        emitted as top-of-row siblings (peer to `batch_size`, `improvement_pct`)."""
        metrics = {
            "avg_s": 1.5,
            "prefill_avg_s": 0.3,
            "decode_avg_s": 1.2,
            "decode_share_of_e2e": 0.8,
        }
        entry = sweep._build_label_result_entry(
            cmd=["x"],
            env_overrides={},
            metrics=metrics,
            log_rel="logs/x.log",
            output_json_rel="json/x.json",
            runner_json_rel="status/x.json",
            ok=True,
            returncode=0,
            evidence_status="n/a",
            evidence={},
            timing={},
        )
        # The metrics dict is preserved
        assert entry["metrics"]["prefill_avg_s"] == pytest.approx(0.3)
        assert entry["metrics"]["decode_avg_s"] == pytest.approx(1.2)
        # And the entry exposes them at the top level for easy access by
        # the new_row builder at line ~3365.
        assert entry.get("prefill_avg_s") == pytest.approx(0.3)
        assert entry.get("decode_avg_s") == pytest.approx(1.2)
        assert entry.get("decode_share_of_e2e") == pytest.approx(0.8)

    def test_omits_new_fields_when_absent_from_metrics(self, sweep):
        """No new top-level keys when the metrics dict doesn't carry them
        (backwards-compatible — legacy raw JSON path)."""
        metrics = {"avg_s": 1.5}
        entry = sweep._build_label_result_entry(
            cmd=["x"],
            env_overrides={},
            metrics=metrics,
            log_rel="logs/x.log",
            output_json_rel="json/x.json",
            runner_json_rel="status/x.json",
            ok=True,
            returncode=0,
            evidence_status="n/a",
            evidence={},
            timing={},
        )
        assert "prefill_avg_s" not in entry
        assert "decode_avg_s" not in entry
        assert "decode_share_of_e2e" not in entry


# ---------------------------------------------------------------------------
# 4. Live child-loop harvest: the bench loop must call _request_phase_deltas
#    (the retired _aggregate_launches cross-launch aggregation used to sit
#    here; the ship gate now decides from a single launch).
# ---------------------------------------------------------------------------


class TestChildLoopCallsSharedExtractor:
    def test_bench_loop_calls_request_phase_deltas(self, sweep_source):
        """The measured-iteration loop must delegate the harvest, not inline it."""
        assert "_iter_pf, _iter_dc = _request_phase_deltas(iter_outputs)" in sweep_source, (
            "The child bench loop must call _request_phase_deltas so the vLLM "
            "metrics field names exist in exactly one place"
        )

    def test_pooled_and_per_iteration_series_both_fed(self, sweep_source):
        """Bucket-level means pool across iterations; the per-iteration series
        keeps one sample per iteration for the phase Welch."""
        assert "prefills_local.extend(_iter_pf)" in sweep_source
        assert "decodes_local.extend(_iter_dc)" in sweep_source

    def test_aggregate_launches_is_retired(self, sweep):
        assert not hasattr(sweep, "_aggregate_launches"), (
            "The multi-launch aggregation is retired (VERSION 1.15.x)"
        )
        assert not hasattr(sweep, "_compute_noise_flag"), (
            "The cross-launch noise flag is retired; NOISE is owned by "
            "generate_validation_report.py::_classify_verdict"
        )


# ---------------------------------------------------------------------------
# 5. Source-level invariants (textual checks for surgical patches)
# ---------------------------------------------------------------------------


class TestSourcePatches:
    def test_disable_log_stats_set_in_ea_dict(self, sweep_source):
        """The LLM(...) constructor receives `disable_log_stats=False` — must be
        wired through `ea_dict` so that vLLM publishes RequestOutput.metrics."""
        # Pattern: ea_dict["disable_log_stats"] = False  (or similar)
        assert re.search(
            r"ea_dict\[\s*['\"]disable_log_stats['\"]\s*\]\s*=\s*False",
            sweep_source,
        ), (
            "ea_dict must set disable_log_stats=False so vLLM populates "
            "RequestOutput.metrics (spec §3.1 step 1)."
        )

    def test_non_beam_search_branch_captures_outputs(self, sweep_source):
        """The non-beam-search branch of llm_generate() must capture the return
        value of `llm.generate(...)`. Per spec, the beam-search branch must NOT.
        """
        # Find the llm_generate function body
        m = re.search(
            r"def llm_generate\(\) -> [^\n:]*:\n(.*?)\n(?=\s{16}\S|\s{12}\S|^\S)",
            sweep_source,
            re.DOTALL,
        )
        assert m, "Could not locate llm_generate() inner function"
        body = m.group(1)
        # The llm.generate(...) call must be assigned (capture) — i.e., appear
        # to the right of `=` on its own line.
        assert re.search(r"=\s*llm\.generate\(", body), (
            "Non-beam-search branch must capture `outputs = llm.generate(...)` "
            "to harvest RequestOutput.metrics (spec §3.1 step 2)."
        )
        # Beam-search branch must remain non-capturing — `llm.beam_search(`
        # should NOT have an `=` immediately to its left.
        assert not re.search(r"=\s*llm\.beam_search\(", body), (
            "Beam-search branch must NOT capture (Tier-C fallback per spec)."
        )

    def test_dp1_path_has_cuda_synchronize_before_t1(self, sweep_source):
        """Per spec §3.1 step 8, the DP=1 run_to_completion() needs
        torch.cuda.synchronize() before `t1 = _time.perf_counter()` for
        wall-time accuracy."""
        # The DP=1 branch is the `else:` arm of `if dp_size > 1:` and defines
        # its own `run_to_completion`. Locate the else block by scanning from
        # the second `def run_to_completion(` onward (the first is the DP>1
        # branch). The body extends until the next dedent (a non-indented
        # `warmup =` line or similar).
        # Find all `def run_to_completion(` occurrences.
        run_iter = list(
            re.finditer(r"def run_to_completion\(", sweep_source)
        )
        assert len(run_iter) >= 2, (
            "Expected at least two run_to_completion definitions (DP>1 + DP=1). "
            f"Found {len(run_iter)}."
        )
        dp1_start = run_iter[1].start()
        # Scan the next ~600 characters of the DP=1 body looking for
        # cuda.synchronize and t1 = perf_counter. The body terminates at the
        # next 16-space-indented `warmup =` (the next sibling statement).
        body = sweep_source[dp1_start : dp1_start + 800]
        # Trim at the first occurrence of the warmup loop start.
        end_marker = body.find("warmup = int(getattr")
        if end_marker > 0:
            body = body[:end_marker]
        sync_idx = body.find("cuda.synchronize()")
        t1_idx = body.find("t1 = _time.perf_counter()")
        assert sync_idx != -1, (
            "DP=1 branch must call torch.cuda.synchronize() before stopping the "
            "wall-time stopwatch (spec §3.1 step 8). Body inspected:\n" + body
        )
        assert t1_idx != -1, "DP=1 branch must record t1 with perf_counter"
        assert sync_idx < t1_idx, (
            "torch.cuda.synchronize() must come BEFORE t1 = perf_counter()"
        )

    def test_raw_json_emits_prefill_decode_fields(self, sweep_source):
        """The per-bucket raw JSON emit (line ~2060) must include
        prefill_avg_s, decode_avg_s, and decode_share_of_e2e — either as
        literal keys in the `raw = {...}` dict OR via subsequent
        `raw[key] = ...` assignments before `_rank0_write_json(raw_json,
        raw)`."""
        # Find the block from `raw = {` up to (and including) the call to
        # write the raw JSON.
        m = re.search(
            r"raw\s*=\s*\{[^}]*?\"avg_latency\".*?_rank0_write_json\(raw_json, raw\)",
            sweep_source,
            re.DOTALL,
        )
        assert m, "Could not locate `raw = { ... }` -> _rank0_write_json block"
        block = m.group(0)
        for key in ("prefill_avg_s", "decode_avg_s", "decode_share_of_e2e"):
            assert key in block, (
                f"`raw` JSON emit block must reference `{key}` "
                f"(spec §3.1 step 4)"
            )
