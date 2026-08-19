#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for preflight_vllm_api.py — the vLLM API coupling self-check.

The preflight must (1) run without a GPU and without building an engine,
(2) name every symbol the sweep script actually reads, and (3) FAIL when a
symbol is renamed. Point 3 is the whole value: a silent rename demotes the OTPS
denominator from decode-only to gross end-to-end.

These tests run without vLLM installed — the checks are driven through fake
modules injected into sys.modules, so the failure paths are exercised on any
machine.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_PREFLIGHT = _SCRIPTS_DIR / "preflight_vllm_api.py"
_SWEEP = _SCRIPTS_DIR / "run_vllm_bench_latency_sweep.py"


def _load_preflight():
    """Load a FRESH module instance (tests mutate its coupling tables)."""
    spec = importlib.util.spec_from_file_location("_preflight_under_test", _PREFLIGHT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pf():
    return _load_preflight()


# ---------------------------------------------------------------------------
# Fake vLLM: every symbol the real coupling table names, at the right place.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeRequestStateStats:
    num_generation_tokens: int = 0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0


@dataclasses.dataclass
class _FakeEngineArgs:
    model: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int = 0
    disable_log_stats: bool = False
    profiler_config: dict = dataclasses.field(default_factory=dict)
    max_num_batched_tokens: int = 0
    compilation_config: dict = dataclasses.field(default_factory=dict)
    attention_config: dict = dataclasses.field(default_factory=dict)
    structured_outputs_config: dict = dataclasses.field(default_factory=dict)
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    @classmethod
    def from_cli_args(cls, args):
        return cls()


@dataclasses.dataclass
class _FakeSamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 16
    detokenize: bool = True
    stop: list = dataclasses.field(default_factory=list)
    seed: int = 0
    logprobs: int = 0


@dataclasses.dataclass
class _FakeBeamSearchParams:
    beam_width: int = 1
    max_tokens: int = 16
    ignore_eos: bool = False


@dataclasses.dataclass
class _FakeProfilerConfig:
    profiler: str = ""
    delay_iterations: int = 0
    max_iterations: int = 0
    torch_profiler_dir: str = ""


@dataclasses.dataclass
class _FakeVllmConfig:
    parallel_config: object = None
    profiler_config: object = None
    model_config: object = None


@dataclasses.dataclass
class _FakeParallelConfig:
    data_parallel_rank: int = 0


@dataclasses.dataclass
class _FakeModelConfig:
    max_model_len: int = 0


class _FakeLLM:
    def __init__(self, **kwargs):
        self.llm_engine = None

    def generate(self, *a, **k):
        return []

    def beam_search(self, *a, **k):
        return []

    def get_tokenizer(self):
        return None

    def get_metrics(self):
        return []

    def start_profile(self):
        pass

    def stop_profile(self):
        pass


class _FakeFlexibleArgumentParser:
    def __init__(self, *args, **kwargs):
        kwargs.pop("add_json_tip", None)
        self.kwargs = kwargs


_FAKE_LATENCY_SOURCE = '''
def add_cli_args(parser):
    parser.add_argument("--input-len")
    parser.add_argument("--output-len")
    parser.add_argument("--batch-size")
    parser.add_argument("--num-iters")
    parser.add_argument("--output-json")
    parser.add_argument("--num-iters-warmup")
    parser.add_argument("--use-beam-search")
    parser.add_argument("--disable-detokenize")
    parser.add_argument("--profile")


def save_to_pytorch_benchmark_format(args, results):
    pass
'''

_FAKE_GPU_WORKER_SOURCE = '''
def annotate():
    return "execute_context_0(0)_generation_8(8)"
'''

_FAKE_SPEC_METRICS_SOURCE = '''
NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
'''


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _source_module(name, source, tmp_path):
    """Build a real importable module so inspect.getsource() works on it."""
    path = tmp_path / (name.replace(".", "_") + ".py")
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_vllm(monkeypatch, tmp_path):
    """Install a complete fake vLLM in sys.modules."""
    mods = {
        "vllm": _module(
            "vllm", LLM=_FakeLLM, SamplingParams=_FakeSamplingParams,
            __version__="9.9.9-fake",
        ),
        "vllm.benchmarks": _module("vllm.benchmarks"),
        "vllm.benchmarks.latency": _source_module(
            "vllm.benchmarks.latency", _FAKE_LATENCY_SOURCE, tmp_path
        ),
        "vllm.engine": _module("vllm.engine"),
        "vllm.engine.arg_utils": _module("vllm.engine.arg_utils", EngineArgs=_FakeEngineArgs),
        "vllm.inputs": _module("vllm.inputs", PromptType=dict),
        "vllm.sampling_params": _module(
            "vllm.sampling_params", BeamSearchParams=_FakeBeamSearchParams,
            SamplingParams=_FakeSamplingParams,
        ),
        "vllm.utils": _module("vllm.utils"),
        "vllm.utils.argparse_utils": _module(
            "vllm.utils.argparse_utils",
            FlexibleArgumentParser=_FakeFlexibleArgumentParser,
        ),
        "vllm.outputs": _module("vllm.outputs", RequestOutput=object),
        "vllm.v1": _module("vllm.v1"),
        "vllm.v1.metrics": _module("vllm.v1.metrics"),
        "vllm.v1.metrics.stats": _module(
            "vllm.v1.metrics.stats", RequestStateStats=_FakeRequestStateStats
        ),
        "vllm.distributed": _module("vllm.distributed"),
        "vllm.distributed.parallel_state": _module(
            "vllm.distributed.parallel_state", get_world_group=lambda: None
        ),
        "vllm.config": _module(
            "vllm.config", ProfilerConfig=_FakeProfilerConfig,
            VllmConfig=_FakeVllmConfig, ParallelConfig=_FakeParallelConfig,
            ModelConfig=_FakeModelConfig,
        ),
        "vllm.envs": _module(
            "vllm.envs",
            environment_variables={
                "VLLM_CACHE_ROOT": None,
                "VLLM_DISABLE_COMPILE_CACHE": None,
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": None,
                "VLLM_WORKER_MULTIPROC_METHOD": None,
            },
        ),
        "vllm.v1.worker": _module("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": _source_module(
            "vllm.v1.worker.gpu_worker", _FAKE_GPU_WORKER_SOURCE, tmp_path
        ),
        "vllm.v1.spec_decode": _module("vllm.v1.spec_decode"),
        "vllm.v1.spec_decode.metrics": _source_module(
            "vllm.v1.spec_decode.metrics", _FAKE_SPEC_METRICS_SOURCE, tmp_path
        ),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return mods


# ---------------------------------------------------------------------------
# Green path
# ---------------------------------------------------------------------------


class TestPreflightPasses:
    def test_complete_api_surface_passes(self, pf, fake_vllm):
        assert pf.run_checks() == []

    def test_main_exits_zero_and_prints_version(self, pf, fake_vllm, capsys):
        assert pf.main([]) == 0
        out = capsys.readouterr().out
        assert out.startswith("OK:")
        assert "9.9.9-fake" in out

    def test_verbose_lists_each_check(self, pf, fake_vllm, capsys):
        pf.run_checks(verbose=True)
        out = capsys.readouterr().out
        assert "ok   RequestOutput.metrics.scheduled_ts" in out
        assert "ok   LLM.generate" in out


# ---------------------------------------------------------------------------
# Failure paths — one per check class
# ---------------------------------------------------------------------------


class TestPreflightDetectsRenames:
    def test_renamed_metrics_timestamp_fails(self, pf, fake_vllm):
        pf.REQUEST_METRICS_FIELDS = ["scheduled_ts", "first_token_time"]
        fails = pf.run_checks()
        assert any("first_token_time" in f for f in fails)

    def test_missing_top_level_symbol_fails(self, pf, fake_vllm):
        pf.MODULE_ATTRS = dict(pf.MODULE_ATTRS)
        pf.MODULE_ATTRS["vllm"] = ["LLM", "GoneSymbol"]
        assert any("vllm.GoneSymbol" in f for f in pf.run_checks())

    def test_missing_module_fails(self, pf, fake_vllm):
        pf.MODULE_ATTRS = dict(pf.MODULE_ATTRS)
        pf.MODULE_ATTRS["vllm.nope.gone"] = ["Thing"]
        assert any("vllm.nope.gone" in f for f in pf.run_checks())

    def test_renamed_engine_args_field_fails(self, pf, fake_vllm):
        pf.ENGINE_ARGS_FIELDS = list(pf.ENGINE_ARGS_FIELDS) + ["renamed_field"]
        assert any("EngineArgs.renamed_field" in f for f in pf.run_checks())

    def test_removed_latency_flag_fails(self, pf, fake_vllm):
        pf.LATENCY_CLI_FLAGS = ["--output-len", "--retired-flag"]
        assert any("--retired-flag" in f for f in pf.run_checks())

    def test_removed_env_var_fails(self, pf, fake_vllm):
        pf.ENV_VARS = ["VLLM_CACHE_ROOT", "VLLM_RETIRED"]
        assert any("VLLM_RETIRED" in f for f in pf.run_checks())

    def test_changed_nvtx_marker_fails(self, pf, fake_vllm):
        pf.NVTX_MARKER_TOKENS = ["_generation_", "_new_marker_"]
        assert any("_new_marker_" in f for f in pf.run_checks())

    def test_renamed_spec_counter_fails(self, pf, fake_vllm):
        pf.SPEC_DECODE_COUNTERS = ["vllm:spec_decode_num_drafts", "vllm:renamed"]
        assert any("vllm:renamed" in f for f in pf.run_checks())

    def test_rejected_parser_kwarg_fails(self, pf, fake_vllm, monkeypatch):
        class _StrictParser:
            def __init__(self, *a, **k):
                raise TypeError("unexpected keyword argument 'add_json_tip'")

        monkeypatch.setitem(
            sys.modules, "vllm.utils.argparse_utils",
            _module("vllm.utils.argparse_utils", FlexibleArgumentParser=_StrictParser),
        )
        assert any("FlexibleArgumentParser" in f for f in pf.run_checks())

    def test_main_exits_one_and_lists_failures(self, pf, fake_vllm, capsys):
        pf.REQUEST_METRICS_FIELDS = ["gone_ts"]
        assert pf.main([]) == 1
        out = capsys.readouterr().out
        assert out.startswith("FAIL:")
        assert "gone_ts" in out

    def test_main_exits_two_without_vllm(self, pf, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "vllm", None)
        assert pf.main([]) == 2
        assert "cannot import vllm" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The table must stay in sync with the sweep script, and stay GPU-free.
# ---------------------------------------------------------------------------


class TestCouplingTableMatchesSweep:
    @pytest.fixture(scope="class")
    def sweep_source(self):
        return _SWEEP.read_text(encoding="utf-8")

    def test_metrics_timestamps_match_the_harvest(self, pf, sweep_source):
        """_request_phase_deltas is the single owner; preflight must name its fields."""
        harvest = sweep_source[sweep_source.index("def _request_phase_deltas"):]
        harvest = harvest[:harvest.index("def _p50")]
        for field in pf.REQUEST_METRICS_FIELDS:
            assert field in harvest, f"{field} is not read by _request_phase_deltas"

    def test_every_sweep_vllm_import_is_covered(self, pf, sweep_source):
        import re

        for mod, names in re.findall(
            r"^\s+from (vllm[\w.]*) import ([\w, ]+)", sweep_source, re.M
        ):
            covered = pf.MODULE_ATTRS.get(mod, [])
            for name in [n.strip() for n in names.split(",")]:
                base = name.split(" as ")[0].strip()
                # Two import forms: `from vllm import LLM` (attribute of mod)
                # and `from vllm.benchmarks import latency` (submodule).
                assert base in covered or f"{mod}.{base}" in pf.MODULE_ATTRS, (
                    f"sweep imports {mod}.{base}; preflight does not check it"
                )

    def test_env_vars_match_the_sweep_injectors(self, pf, sweep_source):
        for name in pf.ENV_VARS:
            assert name in sweep_source, f"{name} is no longer injected by the sweep"

    def test_spec_counters_match_the_snapshot(self, pf, sweep_source):
        for name in pf.SPEC_DECODE_COUNTERS:
            assert name in sweep_source

    def test_preflight_never_builds_an_engine(self, pf):
        """No-GPU contract: the script must not instantiate LLM or call generate."""
        src = _PREFLIGHT.read_text(encoding="utf-8")
        assert "LLM(" not in src
        assert ".generate(" not in src
        assert "import torch" not in src
        assert "add_cli_args(" not in src  # builds pydantic configs; needs a device
