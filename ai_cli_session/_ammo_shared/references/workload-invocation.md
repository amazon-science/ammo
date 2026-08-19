# Workload Invocation — `new_target.py` Details

This file is the supplementary detail for §Invocation in SKILL.md. Read it when
you scaffold a campaign with `new_target.py` and you need the flag a user
parameter maps to, the default when the user gives none, or the reason a flag
behaves in a surprising way. The verbose examples and the edge-case explanations
live here, not in SKILL.md, so the orchestrator does not read them cold.

## Parameter Table

| Parameter | Prompt format (extract from user) | `new_target.py` flag | Default (if user omits) |
|-----------|----------------------------------|---------------------|------------------------|
| Batch sizes | `--batch-sizes 1 8 32` | `--batch-sizes 1 8 32` | `[1, 8, 32]` |
| Max model length | `--max-model-len=auto` or `--max-model-len=65536` | same value; `auto` resolves from the cached model config (fails closed if uncached) | `4096` (omit flag for default) |
| Input/output lengths | `--input-len=<N> --output-len=<N>` | `--input-len <N> --output-len <N>` | `64` / `512` (only if user doesn't specify) |
| Max num sequences | `--max-num-seqs=32` | `--max-num-seqs 32` | omit (vLLM default) |
| Multi ISL/OSL pairs | `--isl-osl=64:512,2048:256` | same value | N/A |
| Data parallel size | `--data-parallel-size 2` | `--data-parallel-size 2` | `1` (single-process) |
| Expert parallelism | `--enable-expert-parallel` | `--enable-expert-parallel` | off |

Batch sizes define the decode buckets for all profiling and validation throughout the campaign.

## Canonical Invocation

```bash
.venv/bin/python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP> \
  [--batch-sizes <BATCH_SIZES>] \
  [--input-len <INPUT_LEN> --output-len <OUTPUT_LEN>] \
  [--max-model-len <MAX_MODEL_LEN>] \
  [--max-num-seqs <MAX_NUM_SEQS>] \
  [--isl-osl <ISL:OSL,...>] \
  [--data-parallel-size <DP_SIZE>] \
  [--enable-expert-parallel]
```

Take `<INPUT_LEN>` and `<OUTPUT_LEN>` from the user's prompt. If the user
specifies neither, omit both flags; the scaffold then freezes the 64/512
decode-heavy workload.

## Sequence Lengths and the Workload Matrix

One ISL/OSL pair is the common case: pass `--input-len` and `--output-len`
straight to `new_target.py`.

For more than one pair (`--isl-osl=`), pass the comma-separated pairs straight
to the flag. `new_target.py` crosses those pairs with every batch size and
freezes the result as `workload_matrix`. Do not mutate `target.json` after
scaffolding.

```json
"workload": {
    "workload_matrix": [
        {"input_len": 64, "output_len": 512, "batch_size": 1},
        {"input_len": 64, "output_len": 512, "batch_size": 8},
        {"input_len": 2048, "output_len": 256, "batch_size": 1},
        {"input_len": 2048, "output_len": 256, "batch_size": 8}
    ],
    "num_iters": 10
}
```

## Max Model Length (`--max-model-len`)

`--max-model-len=auto` resolves `max_position_embeddings` from the locally
cached HF model config. If no cached config exists, the scaffold fails closed
before it writes anything — supply the concrete model limit as an integer
instead. With `auto`, `target.json` also records the request in
`notes.max_model_len_request`, so later stages do not mistake the resolved value
for a hand-picked limit. Omit the flag and the scaffold freezes its default
`4096`.

## Serving Concurrency (`--max-num-seqs`)

Pass an explicit `--max-num-seqs` value to `new_target.py` only when the user
requests one. The scaffold records it in `bench.extra_args`:

```json
"bench": {
    "extra_args": ["--max-num-seqs", "32"],
    ...
}
```

Omit the flag and `extra_args` stays empty, so vLLM's own default serving
concurrency applies. The scaffold does not invent a value.

## Expert Parallelism (`--ep` vs `--enable-expert-parallel`)

`--enable-expert-parallel` is the flag that turns on vLLM expert parallelism.
`--ep N` (integer) does not: it is a legacy sizing field persisted to
`target.ep` in `state.json`, and it is not passed through to the benchmark
command. Most MoE workloads should leave `--ep 1` (default) and rely on
`--enable-expert-parallel` plus the TP size.

## Data Parallelism (`--data-parallel-size`)

`new_target.py` unconditionally injects
`--distributed-executor-backend external_launcher` into `bench.extra_args`
when `--data-parallel-size > 1`, because vLLM requires this backend for
torchrun DP. The sweep script (`run_vllm_bench_latency_sweep.py`) validates that
no conflicting `--distributed-executor-backend` value was appended post-hoc
(e.g., via frontend `additionalFlags`); conflicts fail fast.

## vLLM API Preflight

Run `.venv/bin/python .claude/skills/ammo/scripts/preflight_vllm_api.py` once at
campaign start and again after any vLLM change. It asserts every vLLM symbol the
sweep reads (needs no GPU); exit 1 lists each renamed symbol to patch in
`run_vllm_bench_latency_sweep.py` before you measure.
