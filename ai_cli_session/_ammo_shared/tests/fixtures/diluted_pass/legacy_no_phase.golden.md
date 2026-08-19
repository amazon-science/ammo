# Validation Results (Phase 4)

Generated: GENERATED_AT (UTC)

> Default gates + required reporting checklist: `references/validation-defaults.md`.

## Environment

- TODO: record GPU + driver, torch + CUDA, vLLM commit, model id + quant, and TP/EP topology (`references/validation-defaults.md` § Required reporting checklist).

## Correctness

- TODO: run model-appropriate correctness tests (prefer existing vLLM tests for the model).

## Kernel perf (CUDA graphs)

- TODO: add GPU kernel-time table (baseline vs optimized) under CUDA graphs for the validated bucket set.

## E2E latency (vllm bench latency)

Workload:
- model_id: test/model
- input_len: 1024, output_len: 128
- tp: 1, max_model_len: 4096
- num_iters: 10

| Batch Size | baseline avg (s) | opt avg (s) | Speedup | Improvement | Fast-path evidence |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.952381 | 1.05x | 4.7619% | unknown |

> Note: ensure CUDA graphs / torch.compile settings match production parity per `references/e2e-latency-guide.md`.

## Decision

- TODO: Ship / restrict envelope / pivot route / stop (justify using kernel + E2E evidence).
