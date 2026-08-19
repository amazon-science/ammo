# Bundled dataset attribution

## GSM8K (`gsm8k_full.json`, `gsm8k_subset.json`)

Grade School Math 8K (GSM8K) — 8.5K linguistically diverse grade-school math
word problems.

- **Source**: https://github.com/openai/grade-school-math
- **Paper**: Cobbe et al., *Training Verifiers to Solve Math Word Problems*,
  arXiv:2110.14168 (2021)
- **License**: MIT (per the upstream repository's `LICENSE`)
- **Copyright**: © OpenAI

`gsm8k_full.json` contains the full 1319-item test split plus 5 train items.
`gsm8k_subset.json` is a smaller sample of the same data.

### Why it is bundled

AMMO sessions run inside a sandbox with no outbound network access to package
indexes or dataset hosts, and `ammo-pip-guard.sh` blocks package installation.
The data is read offline by `scripts/run_vllm_bench_latency_sweep.py` as the
correctness reference for end-to-end validation, so it must be present in the
session template rather than fetched at run time.

No modifications were made to the problem or solution text.

## Adapted code: GSM8K evaluation helpers

The GSM8K prompt-building and answer-extraction helpers in
`scripts/run_vllm_bench_latency_sweep.py` are adapted from vLLM's
`tests/evals/gsm8k/gsm8k_eval.py`:

- **Source**: https://github.com/vllm-project/vllm
- **License**: Apache License 2.0
  (https://www.apache.org/licenses/LICENSE-2.0)
- **Provenance**: the vLLM file is itself adapted from SGLang
  (https://github.com/sgl-project/sglang, Apache License 2.0)
