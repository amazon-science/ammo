#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""AMMO preflight — assert every vLLM symbol the sweep couples to still exists.

``run_vllm_bench_latency_sweep.py`` reads vLLM internals (benchmark parser,
EngineArgs fields, ``RequestOutput.metrics`` timestamps, NVTX marker text,
Prometheus counter names). A vLLM bump that renames any of them degrades the
sweep SILENTLY: the metrics harvest yields ``{}`` through ``getattr``-and-skip,
OTPS switches from the decode-only denominator to the gross end-to-end one, and
a track cannot qualify for DILUTED_PASS. Run this once at campaign start and
after any vLLM change.

Import + introspection only. It never builds an engine, so it needs NO GPU.

Exit codes:
  0  every symbol present (one OK line, with the vLLM version)
  1  one or more symbols missing/renamed (one line per failure)
  2  vLLM itself cannot be imported

Usage:
  .venv/bin/python .claude/skills/ammo/scripts/preflight_vllm_api.py [--verbose]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import sys


# ---------------------------------------------------------------------------
# The coupling table. Each entry names ONE thing the sweep reads.
# ---------------------------------------------------------------------------

# module -> attribute names imported by the sweep child runner.
MODULE_ATTRS = {
    "vllm": ["LLM", "SamplingParams", "__version__"],
    "vllm.benchmarks.latency": ["add_cli_args", "save_to_pytorch_benchmark_format"],
    "vllm.engine.arg_utils": ["EngineArgs"],
    "vllm.inputs": ["PromptType"],
    "vllm.sampling_params": ["BeamSearchParams"],
    "vllm.utils.argparse_utils": ["FlexibleArgumentParser"],
    "vllm.outputs": ["RequestOutput"],
    "vllm.v1.metrics.stats": ["RequestStateStats"],
    "vllm.distributed.parallel_state": ["get_world_group"],
    "vllm.config": ["ProfilerConfig", "VllmConfig", "ParallelConfig", "ModelConfig"],
}

# LLM methods/attributes the child calls.
LLM_ATTRS = [
    "generate",
    "beam_search",
    "get_tokenizer",
    "get_metrics",
    "start_profile",
    "stop_profile",
]

# RequestOutput.metrics timestamps. _request_phase_deltas computes
# prefill_s = first_token_ts - scheduled_ts and decode_s = last_token_ts -
# first_token_ts. A rename here is the #1 silent-degradation surface.
REQUEST_METRICS_FIELDS = ["scheduled_ts", "first_token_ts", "last_token_ts"]

# EngineArgs fields the child sets or strips in the asdict() round-trip. The
# first three also back the --model / --tensor-parallel-size / --max-model-len
# argv the sweep builds: EngineArgs.add_cli_args generates those flags from
# these fields, so the field IS the flag contract.
ENGINE_ARGS_FIELDS = [
    "model",
    "tensor_parallel_size",
    "max_model_len",
    "disable_log_stats",
    "profiler_config",
    "max_num_batched_tokens",
    "compilation_config",
    "attention_config",
    "structured_outputs_config",
    "data_parallel_size",
    "pipeline_parallel_size",
]

# SamplingParams fields the timing + correctness phases pass.
SAMPLING_PARAMS_FIELDS = [
    "n",
    "temperature",
    "top_p",
    "ignore_eos",
    "max_tokens",
    "detokenize",
    "stop",
    "seed",
    "logprobs",
]

BEAM_SEARCH_PARAMS_FIELDS = ["beam_width", "max_tokens", "ignore_eos"]

# ProfilerConfig keys the selected-step nsys capture writes.
PROFILER_CONFIG_FIELDS = [
    "profiler",
    "delay_iterations",
    "max_iterations",
    "torch_profiler_dir",
]

# Config attributes read through llm.llm_engine / llm.vllm_config.
CONFIG_FIELDS = {
    "vllm.config:VllmConfig": ["parallel_config", "profiler_config", "model_config"],
    "vllm.config:ParallelConfig": ["data_parallel_rank"],
    "vllm.config:ModelConfig": ["max_model_len"],
}

# Flags `vllm bench latency` itself defines (the rest of the sweep's argv comes
# from EngineArgs.add_cli_args — see ENGINE_ARGS_FIELDS).
LATENCY_CLI_FLAGS = [
    "--input-len",
    "--output-len",
    "--batch-size",
    "--num-iters",
    "--output-json",
    "--num-iters-warmup",
    "--use-beam-search",
    "--disable-detokenize",
    "--profile",
]

# Env vars the sweep injects into the child.
ENV_VARS = [
    "VLLM_CACHE_ROOT",
    "VLLM_DISABLE_COMPILE_CACHE",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
    "VLLM_WORKER_MULTIPROC_METHOD",
]

# NVTX forward-pass marker substrings _NVTX_DECODE_MARKER_RE depends on.
NVTX_MARKER_TOKENS = ["_context_", "_generation_"]

# Prometheus counter names _snapshot_spec_decode_counters sums.
SPEC_DECODE_COUNTERS = [
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _field_names(obj):
    """Field/parameter names of a dataclass, pydantic dataclass, or callable."""
    if dataclasses.is_dataclass(obj):
        return {f.name for f in dataclasses.fields(obj)}
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return set(dir(obj))


def _check_module_attrs(failures, verbose):
    for mod_name, attrs in MODULE_ATTRS.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            failures.append("MODULE MISSING %s (%s: %s)" % (mod_name, type(exc).__name__, exc))
            continue
        for attr in attrs:
            if hasattr(mod, attr):
                if verbose:
                    print("ok   %s.%s" % (mod_name, attr))
            else:
                failures.append("SYMBOL MISSING %s.%s" % (mod_name, attr))


def _check_members(failures, verbose, label, owner, names, getter):
    """Assert each name in *names* is a member of *owner* per *getter*."""
    if owner is None:
        failures.append("SKIPPED %s (owner unavailable)" % label)
        return
    present = getter(owner)
    for name in names:
        if name in present:
            if verbose:
                print("ok   %s.%s" % (label, name))
        else:
            failures.append("FIELD MISSING %s.%s" % (label, name))


def _import_or_none(failures, mod_name, attr):
    try:
        return getattr(importlib.import_module(mod_name), attr)
    except Exception:
        return None


def _check_latency_cli(failures, verbose):
    """`vllm bench latency` must still define the flags the sweep builds.

    ``add_cli_args`` calls ``EngineArgs.add_cli_args``, which instantiates
    pydantic config defaults and needs a device on some builds. So read the
    module SOURCE rather than building the parser — this keeps the check
    GPU-free.
    """
    try:
        mod = importlib.import_module("vllm.benchmarks.latency")
        src = inspect.getsource(mod)
    except Exception as exc:
        failures.append("SOURCE UNAVAILABLE vllm.benchmarks.latency (%s)" % exc)
        return
    for flag in LATENCY_CLI_FLAGS:
        if flag in src:
            if verbose:
                print("ok   vllm bench latency %s" % flag)
        else:
            failures.append("CLI FLAG MISSING vllm bench latency %s" % flag)


def _check_env_vars(failures, verbose):
    try:
        envs = importlib.import_module("vllm.envs")
    except Exception as exc:
        failures.append("MODULE MISSING vllm.envs (%s)" % exc)
        return
    known = set(getattr(envs, "environment_variables", {}) or {})
    for name in ENV_VARS:
        if name in known or hasattr(envs, name):
            if verbose:
                print("ok   env %s" % name)
        else:
            failures.append("ENV VAR MISSING vllm.envs.%s" % name)


def _check_source_tokens(failures, verbose, mod_name, tokens, label):
    try:
        src = inspect.getsource(importlib.import_module(mod_name))
    except Exception as exc:
        failures.append("SOURCE UNAVAILABLE %s (%s)" % (mod_name, exc))
        return
    for token in tokens:
        if token in src:
            if verbose:
                print("ok   %s %r" % (label, token))
        else:
            failures.append("%s MISSING %r in %s" % (label, token, mod_name))


def run_checks(verbose=False):
    """Return a list of failure strings (empty when the API surface matches)."""
    failures = []

    _check_module_attrs(failures, verbose)

    llm = _import_or_none(failures, "vllm", "LLM")
    _check_members(failures, verbose, "LLM", llm, LLM_ATTRS, lambda o: set(dir(o)))

    stats = _import_or_none(failures, "vllm.v1.metrics.stats", "RequestStateStats")
    _check_members(
        failures, verbose, "RequestOutput.metrics", stats,
        REQUEST_METRICS_FIELDS, _field_names,
    )

    engine_args = _import_or_none(failures, "vllm.engine.arg_utils", "EngineArgs")
    _check_members(
        failures, verbose, "EngineArgs", engine_args, ENGINE_ARGS_FIELDS, _field_names
    )
    if engine_args is not None and not hasattr(engine_args, "from_cli_args"):
        failures.append("FIELD MISSING EngineArgs.from_cli_args")

    sampling = _import_or_none(failures, "vllm", "SamplingParams")
    _check_members(
        failures, verbose, "SamplingParams", sampling,
        SAMPLING_PARAMS_FIELDS, _field_names,
    )

    beam = _import_or_none(failures, "vllm.sampling_params", "BeamSearchParams")
    _check_members(
        failures, verbose, "BeamSearchParams", beam,
        BEAM_SEARCH_PARAMS_FIELDS, _field_names,
    )

    prof_cfg = _import_or_none(failures, "vllm.config", "ProfilerConfig")
    _check_members(
        failures, verbose, "ProfilerConfig", prof_cfg,
        PROFILER_CONFIG_FIELDS, _field_names,
    )

    for label, names in CONFIG_FIELDS.items():
        mod_name, attr = label.split(":")
        owner = _import_or_none(failures, mod_name, attr)
        _check_members(failures, verbose, attr, owner, names, _field_names)

    # FlexibleArgumentParser must still accept the two kwargs the child passes.
    fap = _import_or_none(failures, "vllm.utils.argparse_utils", "FlexibleArgumentParser")
    if fap is None:
        failures.append("SYMBOL MISSING vllm.utils.argparse_utils.FlexibleArgumentParser")
    else:
        try:
            fap(add_help=False, add_json_tip=False)
            if verbose:
                print("ok   FlexibleArgumentParser(add_help, add_json_tip)")
        except Exception as exc:
            failures.append(
                "CONSTRUCTOR REJECTED FlexibleArgumentParser(add_help=False, "
                "add_json_tip=False) (%s: %s)" % (type(exc).__name__, exc)
            )

    _check_latency_cli(failures, verbose)
    _check_env_vars(failures, verbose)

    # llm.llm_engine is set in LLM.__init__, so hasattr on the CLASS is False.
    if llm is not None:
        try:
            if "self.llm_engine" not in inspect.getsource(llm.__init__):
                failures.append("FIELD MISSING LLM.llm_engine (set in LLM.__init__)")
            elif verbose:
                print("ok   LLM.llm_engine")
        except Exception as exc:
            failures.append("SOURCE UNAVAILABLE LLM.__init__ (%s)" % exc)

    _check_source_tokens(
        failures, verbose, "vllm.v1.worker.gpu_worker",
        NVTX_MARKER_TOKENS, "NVTX MARKER",
    )
    _check_source_tokens(
        failures, verbose, "vllm.v1.spec_decode.metrics",
        SPEC_DECODE_COUNTERS, "SPEC COUNTER",
    )

    return failures


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Assert the vLLM API surface the AMMO sweep couples to."
    )
    p.add_argument("--verbose", action="store_true", help="print every passing check")
    args = p.parse_args(argv)

    try:
        version = importlib.import_module("vllm").__version__
    except Exception as exc:
        print("FAIL: cannot import vllm (%s: %s)" % (type(exc).__name__, exc))
        print("Activate the session venv, then re-run.")
        return 2

    failures = run_checks(verbose=args.verbose)
    if failures:
        print("FAIL: %d vLLM API mismatch(es) on vllm %s" % (len(failures), version))
        for f in failures:
            print("  " + f)
        print(
            "Patch run_vllm_bench_latency_sweep.py at each site above before "
            "measuring. Cross-version numbers from an unpatched sweep are wrong, "
            "not noisy."
        )
        return 1

    print("OK: vLLM API surface matches the sweep (vllm %s)" % version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
