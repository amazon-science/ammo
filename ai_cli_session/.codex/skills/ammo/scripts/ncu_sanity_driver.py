#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Optional claim-driven NCU driver — minimal steady-state replay target.

Use this only for a specific unresolved hardware-counter claim. The wrapper
runs it under `ncu --replay-mode application`. The driver loads
vLLM once (paying the ~3-5 min cold start / torch.compile / CUDA-graph capture
just once), warms up for a handful of decode steps, then runs the exact number
of steady-state decode steps the caller asks for. Under
`--replay-mode application`, ncu re-runs this entire process once per metric
pass, so the driver does NOT loop internally on a per-pass basis — it runs
ONE benchmark from start to finish, ncu handles pass orchestration externally.

Why a dedicated driver instead of `vllm bench latency`:

`vllm bench latency` takes `--num-iters N` and runs a full warmup + N iters.
Under ncu application-replay, that launches ncu's workload once per pass and
ncu captures EVERY kernel launch in the target. With L decoder layers,
default warmup=10, and a broad kernel filter, a single ncu run easily replays
~400 kernel launches per pass × 18+ passes for default `--set` metrics =
tens of minutes just on the cold-start-adjacent launches. Adding
`--launch-count 30` to ncu helps, but only bounds what ncu *records* — the
underlying vLLM process still runs the full warmup every pass.

This driver fixes that by making warmup tiny (default 3) and steady-state
decode steps explicit (default 10). A caller who wants a 30-launch ncu
window picks `--warmup 3 --iters 10` and sets ncu's `--launch-count 30`;
the two numbers multiply to a predictable kernel-launch budget.

Target invocation (called by ncu_sanity.sh):
  ncu ... --launch-count 30 -- \\
    python ncu_sanity_driver.py \\
      --target-json <path> --batch-size 8 --warmup 3 --iters 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# vLLM V1 launches the engine in a subprocess by default. ncu's
# --profile-from-start off gates capture via cudaProfilerStart(), but the call
# only affects the calling process's CUDA context — the engine subprocess keeps
# capture disabled, so ncu records "No kernels were profiled." Force V1 into
# single-process mode so the driver's cudaProfilerStart() gates the same
# context that launches kernels. Must be set before any vllm import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _read_target(target_json: Path) -> dict:
    with target_json.open() as f:
        return json.load(f)


def _pick_bucket(target: dict, batch_size: int) -> dict:
    workload = target.get("workload", {})
    matrix = workload.get("workload_matrix")
    if matrix:
        for b in matrix:
            if int(b["batch_size"]) == batch_size:
                return {
                    "input_len": int(b["input_len"]),
                    "output_len": int(b["output_len"]),
                    "batch_size": batch_size,
                }
        raise SystemExit(
            f"batch_size={batch_size} not in workload_matrix; "
            f"available: {[b['batch_size'] for b in matrix]}"
        )
    return {
        "input_len": int(workload.get("input_len", 64)),
        "output_len": int(workload.get("output_len", 512)),
        "batch_size": batch_size,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-json", type=Path, required=True,
                   help="target.json — used for model_id, tp, max_model_len, workload shape.")
    p.add_argument("--batch-size", type=int, required=True,
                   help="Which decode bucket to replay. Must be present in workload.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Steady-state warmup iterations before the measurement window. "
                        "Small by design — ncu application-replay pays this per pass.")
    p.add_argument("--iters", type=int, default=10,
                   help="Steady-state iterations to time. ncu will capture the first "
                        "`--launch-count` kernel launches within these iters; 10 iters × "
                        "L decoder layers is the upper bound on captured launches per pass.")
    p.add_argument("--output-len", type=int, default=None,
                   help="Override workload.output_len. Defaults to the bucket's output_len. "
                        "Short OL (e.g. 8) is fine for ncu — we're measuring kernel counters, "
                        "not latency — and it keeps each iter cheap.")
    args = p.parse_args()

    target = _read_target(args.target_json)
    bucket = _pick_bucket(target, args.batch_size)
    if args.output_len is not None:
        bucket["output_len"] = args.output_len

    model_id = target["target"]["model_id"]
    tp = int(target["target"].get("tp", 1))
    max_model_len = int(target["target"].get("max_model_len", 4096))

    # Import inside main so `--help` doesn't require torch/vllm.
    import torch
    from vllm import LLM, SamplingParams

    print(f"[driver] model={model_id} tp={tp} bs={bucket['batch_size']} "
          f"il={bucket['input_len']} ol={bucket['output_len']} "
          f"warmup={args.warmup} iters={args.iters}", flush=True)

    llm_kwargs = dict(
        model=model_id,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        enforce_eager=False,
        enable_prefix_caching=False,
        max_num_seqs=bucket["batch_size"],
        compilation_config={
            "cudagraph_capture_sizes": [bucket["batch_size"]],
        },
        # Pinned explicitly because the two construction paths below default it
        # differently: LLM(**kwargs) injects True, EngineArgs defaults to False.
        # Stat loggers inside an ncu-profiled process add launches we do not
        # want, so both paths must get True.
        disable_log_stats=True,
    )
    # vLLM builds that expose LLM.from_engine_args want an EngineArgs object;
    # older builds take the same fields as constructor kwargs. Every field is
    # passed explicitly, so both paths build the same engine.
    if hasattr(LLM, "from_engine_args"):
        from vllm.engine.arg_utils import EngineArgs

        llm = LLM.from_engine_args(EngineArgs(**llm_kwargs))
    else:
        llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        max_tokens=bucket["output_len"],
        temperature=0.0,
        ignore_eos=True,
    )

    # Synthetic prompts of approximately input_len tokens — content doesn't
    # matter for kernel-counter capture, and BPE merges mean the actual token
    # count lands within ~20% of input_len. Exact length doesn't affect
    # steady-state decode counters; we just need a cheap-to-tokenize prompt.
    prompt = "hi " * bucket["input_len"]
    prompts = [prompt] * bucket["batch_size"]

    print("[driver] warmup...", flush=True)
    for _ in range(args.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)

    # Application-replay: ncu runs this whole process multiple times, capturing
    # a bounded kernel window per pass. Gate the measured region with
    # torch.cuda.cudart().cudaProfilerStart/Stop so that when ncu is launched
    # with `--profile-from-start off` the cold start + warmup launches stay
    # out of the captured launch-count window.
    print("[driver] measure...", flush=True)
    profiler_started = False
    try:
        torch.cuda.cudart().cudaProfilerStart()
        profiler_started = True
    except Exception as exc:  # noqa: BLE001 — we need to surface any failure mode
        # Without Start(), `--profile-from-start off` means ncu captures nothing.
        # Warn loudly — silent degradation here produces an empty CSV that
        # looks superficially valid and wastes a full rerun to diagnose.
        print(f"[driver] WARNING: cudaProfilerStart failed ({exc!r}); "
              f"ncu capture window may be empty if --profile-from-start off is set.",
              file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        llm.generate(prompts, sampling, use_tqdm=False)
    dt = time.perf_counter() - t0

    if profiler_started:
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as exc:  # noqa: BLE001
            print(f"[driver] WARNING: cudaProfilerStop failed ({exc!r})",
                  file=sys.stderr, flush=True)

    print(f"[driver] done: {args.iters} iters in {dt:.3f}s "
          f"({dt / args.iters:.3f}s/iter avg)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
