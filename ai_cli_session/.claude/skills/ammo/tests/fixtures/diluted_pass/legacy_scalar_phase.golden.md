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
| 1 | 1 | 0.996 | 1.00402x | 0.4% | unknown |

> Note: ensure CUDA graphs / torch.compile settings match production parity per `references/e2e-latency-guide.md`.

### Phase decomposition (prefill/decode, same instrument)

> Batch-mode phase timing from `RequestOutput.metrics` — the same fixed-batch measurement decomposed, NOT serving TTFT/ITL under load (no queueing/arrival dynamics). Informational triage only: flags never change a verdict. See `references/validation-defaults.md` § Phase Decomposition.

| Batch Size | Decode share | Prefill Δ | Decode (TPOT) Δ | Opt OTPS (tok/s) | Expected E2E Δ (phases) | Actual E2E Δ | Flag |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1 | 0% | 4% |  | 0.4% | 0.4% | DILUTED-WIN |

- **BS 1 DILUTED-WIN**: decode Δ 4% × share 0.1 ≈ E2E Δ 0.4% — Amdahl-consistent. The mechanism worked; the small E2E number is dilution by prefill share, not a failed optimization. If this track fails on the E2E floor, record `fail_reason` as "diluted (Amdahl-consistent)", not "ineffective".

## Decision

- TODO: Ship / restrict envelope / pivot route / stop (justify using kernel + E2E evidence).
