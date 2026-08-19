#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Portions adapted from vLLM (Apache-2.0); see ai_cli_session/.claude/skills/ammo/data/ATTRIBUTION.md
"""Run production-parity E2E latency sweeps via `vllm bench latency`.

Inputs
------
Reads a JSON target spec (default: {artifact_dir}/target.json) created by scripts/new_target.py.

Minimal expected schema:

{
  "artifact_dir": "...",
  "target": {
    "model_id": "Qwen/...",
    "dtype": "fp8",
    "tp": 1,
    "ep": 1,
    "max_model_len": 4096
  },
  "workload": {
    "input_len": 64,
    "output_len": 512,
    "batch_sizes": [1,4,8],
    "num_iters": 10
  },
  "bench": {
    "runner": "vllm_bench_latency",
    "vllm_cmd": "vllm",
    "extra_args": [],
    "baseline_extra_args": [],
    "opt_extra_args": [],
    "baseline_env": {},
    "opt_env": {"<ENABLE_FLAG>": "1"},
    "baseline_label": "baseline",
    "opt_label": "opt",
    "fastpath_evidence": {
      "opt": {"require_patterns": ["..."], "forbid_patterns": ["..."]},
      "baseline": {"require_patterns": [], "forbid_patterns": []}
    }
  }
}

Guardrails / anti-thrash
-----------------------
- Fails fast if required fields are missing or still placeholders.
- Records full commands + env vars + stdout/stderr logs per bucket.
- Optional fast-path evidence checks: require/forbid regex patterns per run.
- Avoids model reload thrash: uses an in-process sweep runner that loads the
  model once per label (baseline/opt) and benchmarks all batch sizes in that
  process. This reduces end-to-end sweep time without changing per-iteration
  latency measurement.
- Supports vLLM's dotted "json-style" CLI flags in `inproc_sweep` (e.g.
  `-cc.pass_config.enable_sp=false`), because it uses vLLM's
  `FlexibleArgumentParser` for argument parsing.
- Archives an existing output directory automatically (instead of refusing to
  run again) unless `--overwrite` is passed.
- Emits live progress to logs and `status/*.json` so agents can distinguish
  "slow" from "hung".

Timing prompts
--------------
- The in-process timing loop feeds REAL prompts by default: deterministic
  windows of the bundled GSM8K corpus (tokenized once per tokenizer), sized to
  exactly `input_len` tokens. This is required for faithful speculative-decode
  acceptance (uniform-random tokens make draft/target models disagree, so spec
  acceptance collapses to ~0 and ngram yields zero proposals). Prompts are
  identical for the baseline and optimized arms (crc32-seeded by workload), so
  the kernel is the only variable in the A/B. `--dummy-prompt-source random`
  restores the legacy uniform-random behavior (tests/debug only; no data needed).
  NOTE: absolute latency/OTPS from real-prompt rounds are NOT comparable to
  historical random-token rounds.
- When real draft activity is observed, each label entry additionally carries a
  `spec_decode` block (`num_drafts`, `num_draft_tokens`, `num_accepted_tokens`,
  and the guarded `draft_acceptance_rate` / `mean_acceptance_length`). The block
  is purely additive and omitted entirely for vanilla-decode targets.

Outputs
-------
Writes into:
  {artifact_dir}/{out_name}/
    - e2e_latency_results.json / e2e_latency_results.md
    - logs/{label}_bs{BS}.log
    - json/{label}_bs{BS}.json  (raw vllm bench --output-json)
    - status/{label}.json  (heartbeat + current phase)

Usage
-----
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir>
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir> --out-name e2e_latency_combined

If your vLLM CLI differs, edit target.json: bench.vllm_cmd, bench.extra_args,
bench.baseline_extra_args, bench.opt_extra_args.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import selectors
import signal
import socket
import subprocess
import sys
import time
import traceback
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast


PLACEHOLDER = "<FILL_ME>"


# ---------------------------------------------------------------------------
# Rank-0 gating for SPMD torchrun ranks (§4.3 of the DP/EP parallelism spec)
#
# Under torchrun, each DP rank re-enters this script with a distinct RANK env.
# All artifact I/O (JSON, logs, status) must go through the _rank0_* helpers
# below so only rank 0 writes. When RANK is unset (existing DP=1 path),
# _GLOBAL_RANK defaults to 0 → _IS_RANK0 is True → helpers delegate to the
# original _write_* / open() functions, preserving byte-equivalent behavior.
# ---------------------------------------------------------------------------

_GLOBAL_RANK: int = int(os.environ.get("RANK", "0"))
_IS_RANK0: bool = (_GLOBAL_RANK == 0)


class _NullFile:
    """File-shaped sink for non-rank-0 log opens.

    Swallows writes so shared bucket-loop code paths don't need per-site
    ``if _IS_RANK0`` guards around every ``log.write()`` call.
    """

    def write(self, *_a, **_kw) -> int:
        return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullFile":
        return self

    def __exit__(self, *_a) -> bool:
        return False


@dataclass
class RunSpec:
    label: str
    env: Dict[str, str]
    require_patterns: List[str]
    forbid_patterns: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Target spec not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _vllm_version() -> Optional[str]:
    """Return the installed vLLM version string, or None when unavailable.

    Stamped into the results header so a cross-version A/B (a Stage-1 baseline
    imported via ``--baseline-from`` weeks later) is visible in the artifact.
    Reads package METADATA only — it must not import vLLM, because the parent
    process deliberately keeps the engine out of its address space. Defensive:
    never raises.
    """
    try:
        import importlib.metadata as _md

        return _md.version("vllm")
    except Exception:
        return None


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and (v.strip() == PLACEHOLDER or v.strip().startswith("<") and v.strip().endswith(">"))


def _ensure_worktree_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    """Prepend CWD to PYTHONPATH so worktree's ``vllm`` package is imported first.

    When the sweep script runs inside a git worktree, the editable install
    resolves to the main repo.  By prepending CWD (the worktree root) to
    PYTHONPATH, child subprocesses will ``import vllm`` from the worktree's
    modified code rather than the main repo's.
    """
    env = dict(env)  # shallow copy — don't mutate caller's dict
    cwd = os.getcwd()
    existing = env.get("PYTHONPATH", "")
    if existing:
        # Avoid duplicating if CWD is already the first entry.
        parts = existing.split(":")
        if parts[0] != cwd:
            env["PYTHONPATH"] = f"{cwd}:{existing}"
    else:
        env["PYTHONPATH"] = cwd
    return env


def _inject_fresh_cache_env(env: Dict[str, str], sweep_cache_root: Path) -> Dict[str, str]:
    """Inject per-sweep ``VLLM_CACHE_ROOT`` + ``TRITON_CACHE_DIR`` into *env*.

    When ``--fresh-cache`` is set, each sweep uses an isolated compile-cache
    directory so measurements are not skewed by partially-warm caches from a
    previous sweep. ``VLLM_CACHE_ROOT`` is the sweep-scoped cache root; the
    Triton compile cache lives inside it so both are torn down together.

    Do NOT set ``VLLM_DISABLE_COMPILE_CACHE=1`` — save must still work within
    the run so each batch size amortizes compile across launches 2..N.

    Does not mutate ``env`` in place.
    """
    env = dict(env)
    env["VLLM_CACHE_ROOT"] = str(sweep_cache_root)
    env["TRITON_CACHE_DIR"] = str(sweep_cache_root / "triton_cache")
    return env


def _inject_nsys_execute_timeout_env(
    env: Dict[str, str], nsys_timeout_s: int
) -> Dict[str, str]:
    """Keep vLLM worker RPC timeout at least as high as the nsys bucket timeout.

    nsys node-mode replay can make a single worker RPC, especially
    ``sample_tokens``, run longer than vLLM's default 300s internal timeout.
    The outer sweep timeout cannot help if the engine kills itself first, so
    profiling runs must raise ``VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`` too.
    """
    env = dict(env)
    current_raw = env.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")
    try:
        current = int(current_raw) if current_raw is not None else 0
    except ValueError:
        current = 0
    env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(max(current, int(nsys_timeout_s)))
    return env


# Blackwell markers in target.hardware. Hardware CUDA tracing stalls under CUDA
# graph replay on these parts, so nsys must use software tracing (cuda-sw).
# GB must be followed by a digit (GB200/GB300) so a memory size such as
# "A100-80GB" does not match.
_CUDA_SW_HARDWARE_RE = re.compile(r"B200|B300|GB\d|BLACKWELL", re.IGNORECASE)


def _default_nsys_trace(hardware: Any) -> str:
    """Derive the nsys CUDA trace backend from ``target.hardware``.

    Returns ``"cuda-sw"`` for Blackwell (B200/B300/GB200/GB300, SM100/SM120),
    else ``"cuda"``. This replaces a prose rule the agent had to remember on
    every capture: forgetting it produces a replay hang that looks like an
    ordinary profiling failure. ``--nsys-trace`` still overrides.
    """
    hw = hardware if isinstance(hardware, str) else ""
    return "cuda-sw" if _CUDA_SW_HARDWARE_RE.search(hw) else "cuda"


def _request_phase_deltas(outputs: Any) -> Tuple[List[float], List[float]]:
    """Return ``(prefills_s, decodes_s)`` from vLLM ``RequestOutput.metrics``.

    THE single place that names the vLLM v1 metrics fields. Per spec §3.1
    those monotonic timestamps give:
      - ``prefill_s = first_token_ts - scheduled_ts``
      - ``decode_s  = last_token_ts - first_token_ts``

    An output is skipped when ``.metrics`` is absent/None or any timestamp is
    non-numeric (older vLLM, beam-search Tier-C fallback). Both lists are
    empty when no output qualifies. Defensive — never raises on missing
    attributes. On a vLLM bump that renames these fields, patch here only.
    """
    prefills: List[float] = []
    decodes: List[float] = []
    for out in outputs or []:
        m = getattr(out, "metrics", None)
        if m is None:
            continue
        sched = getattr(m, "scheduled_ts", None)
        first = getattr(m, "first_token_ts", None)
        last = getattr(m, "last_token_ts", None)
        if not isinstance(sched, (int, float)):
            continue
        if not isinstance(first, (int, float)):
            continue
        if not isinstance(last, (int, float)):
            continue
        prefills.append(float(first) - float(sched))
        decodes.append(float(last) - float(first))
    return prefills, decodes


def _p50(vals: List[float]) -> float:
    """Median with linear interpolation between closest ranks (numpy default)."""
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return s[0]
    k = (n - 1) * 0.5
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _harvest_request_metrics(outputs: Any) -> Dict[str, float]:
    """Aggregate ``_request_phase_deltas`` into mean + p50 of each phase.

    Returns an empty dict when no output carries usable metrics. Defensive —
    never raises on missing attributes.
    """
    prefills, decodes = _request_phase_deltas(outputs)
    if not prefills:
        return {}

    prefill_avg = sum(prefills) / len(prefills)
    decode_avg = sum(decodes) / len(decodes)
    total = prefill_avg + decode_avg
    out: Dict[str, float] = {
        "prefill_avg_s": prefill_avg,
        "decode_avg_s": decode_avg,
        "prefill_p50_s": _p50(prefills),
        "decode_p50_s": _p50(decodes),
    }
    # Per-request metrics carry no authoritative wall, so a wall-denominated
    # decode_share_of_e2e cannot be computed here — emit only the honestly
    # labeled phase-sum ratio.
    if total > 0:
        out["decode_share_of_phase_sum"] = decode_avg / total
    return out


def _compute_token_throughput(
    entry: Dict[str, Any], output_len: int, batch_size: int
) -> Dict[str, float]:
    """Derive output-token throughput (OTPS) and time-per-output-token (TPOT).

    Prefers the REAL per-request decode timing harvested by
    ``_harvest_request_metrics`` (``decode_avg_s = last_token_ts -
    first_token_ts``). The prefill produces the first token and the decode
    phase produces the remaining ``output_len - 1`` tokens, so

      ``tpot_s = decode_avg_s / (output_len - 1)``   (decode-only, exact)
      ``otps   = batch_size * (output_len - 1) / decode_avg_s`` = batch_size / tpot_s

    No prefill-subtraction is needed — the decode time is measured directly.

    Falls back to GROSS end-to-end throughput from ``avg_s`` (prefill + decode)
    when decode timing is unavailable (older vLLM without ``RequestOutput.metrics``,
    beam-search Tier-C fallback). In that case ``tpot_s`` is omitted (prefill and
    decode cannot be separated from a single E2E number) and ``otps`` is the gross
    ``batch_size * output_len / avg_s``. ``throughput_method`` records which path
    was taken. Defensive — returns ``{}`` rather than raising on bad inputs.

    Only numeric keys are returned, so a legacy/failed row produces no keys.
    """
    if not isinstance(entry, dict):
        return {}
    out: Dict[str, float] = {}
    decode_avg_s = entry.get("decode_avg_s")
    if (
        isinstance(decode_avg_s, (int, float))
        and decode_avg_s > 0
        and isinstance(output_len, int)
        and output_len > 1
    ):
        decode_tokens = output_len - 1
        out["tpot_s"] = decode_avg_s / decode_tokens
        out["otps"] = batch_size * decode_tokens / decode_avg_s
        out["throughput_method"] = "decode_metrics"
        return out
    # Fallback: gross E2E throughput (cannot isolate decode → no TPOT).
    avg_s = entry.get("avg_s")
    if (
        isinstance(avg_s, (int, float))
        and avg_s > 0
        and isinstance(output_len, int)
        and output_len > 0
    ):
        out["otps"] = batch_size * output_len / avg_s
        out["throughput_method"] = "gross_e2e"
    return out


def _throughput_method_mismatch(
    baseline_tp: Dict[str, Any], opt_tp: Dict[str, Any]
) -> Optional[Tuple[str, str]]:
    """Return ``(baseline_method, opt_method)`` when the two tiers disagree.

    ``decode_metrics`` and ``gross_e2e`` denominate OTPS differently (decode-only
    vs prefill+decode, ~1.8x apart on a decode-heavy workload), so a cross-tier
    otps_gain_pct is a wrong number, not a noisy one. The usual cause is a vLLM
    bump that renamed ``RequestOutput.metrics`` fields, which silently demotes
    only the arm measured after the bump. Returns None when either side has no
    method (legacy entry, no numbers to compare) or the tiers agree.
    """
    b = baseline_tp.get("throughput_method") if isinstance(baseline_tp, dict) else None
    o = opt_tp.get("throughput_method") if isinstance(opt_tp, dict) else None
    if not isinstance(b, str) or not isinstance(o, str) or b == o:
        return None
    return (b, o)


def _welch_significance(
    baseline_lats: List[Any],
    opt_lats: List[Any],
) -> Optional[Dict[str, Any]]:
    """Within-launch Welch check on per-iteration latencies (single launch).

    Both labels' raw bucket JSONs carry the full per-iteration ``latencies``
    list (num_iters samples). A Welch-style two-sample test on those lists
    costs zero extra launches and zero model reloads — it is the default
    significance evidence for gate verdicts. ``significant`` is True when
    |t| >= 2.0 (~95% two-sided at the default num_iters).

    Caveat: iterations within one launch share a warm server, so this
    UNDERESTIMATES cross-launch variance. It is paired with the PASS floor
    (max(min_e2e_improvement_pct, noise_tolerance_pct)) in
    generate_validation_report.py, which covers the residual gap.

    Returns None when either side has < 2 numeric samples.
    """
    import math

    b = [float(x) for x in (baseline_lats or []) if isinstance(x, (int, float))]
    o = [float(x) for x in (opt_lats or []) if isinstance(x, (int, float))]
    nb, no = len(b), len(o)
    if nb < 2 or no < 2:
        return None
    mb = sum(b) / nb
    mo = sum(o) / no
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    vo = sum((x - mo) ** 2 for x in o) / (no - 1)
    se = math.sqrt(vb / nb + vo / no)
    delta = mb - mo
    if se == 0.0:
        t_stat: Optional[float] = None
        significant = delta != 0.0
    else:
        t_stat = delta / se
        significant = abs(t_stat) >= 2.0
    return {
        "method": "welch_within_launch",
        "n_baseline": nb,
        "n_opt": no,
        "mean_baseline_s": mb,
        "mean_opt_s": mo,
        "stderr_s": se,
        "t_stat": t_stat,
        "significant": significant,
    }


def _row_phase_significance(baseline_raw: Any, opt_raw: Any) -> Dict[str, Any]:
    """Phase-level Welch for one A/B row. Reuses _welch_significance verbatim on the
    per-iteration phase-mean arrays the child runner emits (prefill_iter_means_s /
    decode_iter_means_s). Returns {} when neither phase produced a Welch result
    (legacy/Tier-B/C raw JSON, or <2 samples) — the caller omits the key when empty,
    preserving byte-compat with pre-existing raw JSON."""
    phase_sig: Dict[str, Any] = {}
    pf_sig = _welch_significance(
        baseline_raw.get("prefill_iter_means_s") if isinstance(baseline_raw, dict) else None,
        opt_raw.get("prefill_iter_means_s") if isinstance(opt_raw, dict) else None,
    )
    if pf_sig is not None:
        phase_sig["prefill"] = pf_sig
    dc_sig = _welch_significance(
        baseline_raw.get("decode_iter_means_s") if isinstance(baseline_raw, dict) else None,
        opt_raw.get("decode_iter_means_s") if isinstance(opt_raw, dict) else None,
    )
    if dc_sig is not None:
        phase_sig["decode"] = dc_sig
    return phase_sig


def _build_label_result_entry(
    *,
    cmd: List[str],
    env_overrides: Dict[str, str],
    metrics: Dict[str, float],
    log_rel: str,
    output_json_rel: str,
    runner_json_rel: str,
    ok: Any,
    returncode: int,
    evidence_status: str,
    evidence: Dict[str, Any],
    timing: Dict[str, Any],
    spec_decode: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a label result dict for the sweep JSON output."""
    entry: Dict[str, Any] = {
        "ok": ok,
        "returncode": returncode,
        "cmd": cmd,
        "env_overrides": env_overrides,
        "metrics": metrics,
        "avg_s": metrics.get("avg_s") if isinstance(metrics, dict) else None,
        "fastpath_evidence": {
            **evidence,
            "status": evidence_status,
        },
        "log": log_rel,
        "output_json": output_json_rel,
        "runner_json": runner_json_rel,
        "timing": timing,
    }
    # Per spec §3.1 step 6, prefill/decode timing is workload-property —
    # surface as top-of-row siblings so the consolidated row reads cleanly
    # without diving into ``metrics``. Only emit keys that are actually
    # numeric (back-compat: legacy raw JSON without these fields produces
    # the original schema).
    if isinstance(metrics, dict):
        for key in (
            "prefill_avg_s",
            "decode_avg_s",
            "prefill_p50_s",
            "decode_p50_s",
            "decode_share_of_e2e",
            "decode_share_of_phase_sum",
        ):
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                entry[key] = float(v)
    # Additive spec-decode acceptance block, lifted directly from the child's raw
    # JSON (NOT routed through the float-only _metrics_from_vllm_latency_json,
    # which would drop the nested dict). Key absent when there was no draft
    # activity (omit-not-null).
    if spec_decode is not None:
        entry["spec_decode"] = spec_decode
    return entry


def _sanitize_vllm_op_env(
    env: Dict[str, str],
    preserve_keys: "frozenset[str] | set[str]" = frozenset(),
) -> Dict[str, str]:
    """Strip stale ``VLLM_*`` env vars from *env* to prevent cross-track contamination.

    After the Track A6 baseline-promotion flow, a shipped round's ``opt_env`` keys
    get merged into ``target.json:bench.baseline_env`` so subsequent rounds treat
    them as the new baseline. The sanitizer therefore needs to remove ALL stale
    ``VLLM_*`` vars (not just ``VLLM_OP\\d+``), because the set of live "optimization
    flags" now includes campaign-defined names like ``VLLM_MOE_TRITON_ROUTER`` that
    don't match the numeric op-ID pattern.

    *preserve_keys* lists the names that MUST survive sanitization — typically
    the union of ``baseline_env.keys()`` (promoted flags the current run needs)
    and ``opt_env.keys()`` (the current round's experimental flag). Anything
    else matching ``^VLLM_`` is dropped because it was inherited from the shell
    or a prior process and would silently contaminate the measurement.

    Non-``VLLM_`` vars pass through untouched.
    """
    import re as _re
    _vllm_prefix_re = _re.compile(r"^VLLM_")
    preserve = set(preserve_keys)
    return {
        k: v
        for k, v in env.items()
        if not _vllm_prefix_re.match(k) or k in preserve
    }


# ---------------------------------------------------------------------------
# DP/EP parallelism parsing (§4.2 of the MoE DP/EP parallelism-controls spec)
# ---------------------------------------------------------------------------


def _parse_parallelism_from_args(args_list: List[str]) -> Dict[str, Any]:
    """Parse tp/pp/dp/prefill_cp/distributed-executor-backend/beam_search from argv.

    Uses a standalone argparse.ArgumentParser so the helper runs in environments
    without vLLM installed (unit tests). Flags mirror vLLM's canonical names.
    Unknown flags are ignored (parse_known_args).

    Returns a dict: {tp, pp, dp, prefill_cp, distributed_backend, use_beam_search}.
    No validation — caller validates.
    """
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    p.add_argument("--data-parallel-size", "-dp", type=int, default=1)
    p.add_argument("--prefill-context-parallel-size", type=int, default=1)
    p.add_argument("--distributed-executor-backend", type=str, default=None)
    p.add_argument("--use-beam-search", action="store_true", default=False)
    ns, _unknown = p.parse_known_args(args_list)
    return {
        "tp": int(ns.tensor_parallel_size),
        "pp": int(ns.pipeline_parallel_size),
        "dp": int(ns.data_parallel_size),
        "prefill_cp": int(ns.prefill_context_parallel_size),
        "distributed_backend": ns.distributed_executor_backend,
        "use_beam_search": bool(ns.use_beam_search),
    }


def _parse_positions_per_step(args_list: List[str]) -> int:
    """positions advanced per decode worker-step = 1 + num_speculative_tokens.

    Under vanilla decode every worker-step emits exactly one output token per request,
    so positions_per_step == 1. Under vLLM speculative decoding each target-verification
    worker-step instead advances the bonus token plus ``num_speculative_tokens`` draft
    positions, i.e. positions_per_step == 1 + num_speculative_tokens. The marker's
    ``num_generation_tokens`` sums num_scheduled_tokens over generation reqs (not accepted
    tokens), so this count is method-agnostic (eagle/mtp/ngram/draft) and acceptance-
    independent.

    Uses a standalone argparse.ArgumentParser (like ``_parse_parallelism_from_args``) so the
    helper runs without vLLM imported. Returns 1 when no ``--speculative-config``/``-sc`` is
    present (vanilla decode) OR when the count cannot be recovered statically (conservative;
    never raises — a wrong-low fallback is caught loudly downstream by the arm-reachability
    assertion and the rename-undercount escalation rather than silently mislabeling reps).
    """
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--speculative-config", "-sc", type=str, default=None)
    ns, _unknown = p.parse_known_args(args_list)
    if ns.speculative_config is None:
        return 1
    try:
        cfg = json.loads(ns.speculative_config)
    except (ValueError, TypeError):
        # Non-inline JSON (e.g. -sc @file.json or a bare path). We cannot recover the count
        # statically; fall back to vanilla (loud downstream if it under-arms/over-rejects).
        print(
            "WARNING: --speculative-config value is not inline JSON; positions_per_step=1 "
            "(spec-decode nsys capture may under-arm/over-reject).",
            file=sys.stderr,
        )
        return 1
    if not isinstance(cfg, dict):
        return 1
    n = cfg.get("num_speculative_tokens")
    if n is None:
        # Spec config present but count missing (e.g. method derives it from the draft).
        print(
            "WARNING: --speculative-config present but num_speculative_tokens absent; "
            "positions_per_step=1 (spec-decode nsys capture may under-arm).",
            file=sys.stderr,
        )
        return 1
    try:
        return 1 + int(n)
    except (ValueError, TypeError):
        print(
            "WARNING: num_speculative_tokens not int-coercible; positions_per_step=1.",
            file=sys.stderr,
        )
        return 1


def _resolve_parallelism_and_backend(
    extra_args: List[str],
    baseline_extra_args: List[str],
    opt_extra_args: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Resolve effective parallelism across both labels, validate, auto-inject.

    Flow (spec §4.2):
      1. Parse `extra_args + baseline_extra_args` → baseline_parse.
      2. Parse `extra_args + opt_extra_args` → opt_parse.
      3. Assert tp/pp/dp agree between the two labels (raise on divergence).
      4. Assert prefill_cp == 1 (raise otherwise — unsupported in this flow).
      5. If dp > 1 and either label sets `--distributed-executor-backend`
         to anything other than `external_launcher` → raise.
      6. If dp > 1 and neither label sets the backend → auto-inject
         `--distributed-executor-backend external_launcher` into `extra_args`.
      7. If dp > 1 and either label requests `--use-beam-search` → raise
         (beam-search + DP semantics undefined under external_launcher).

    Returns:
      (augmented_extra_args, {"tp","pp","dp","prefill_cp","nproc"})
    """
    baseline_parse = _parse_parallelism_from_args(extra_args + baseline_extra_args)
    opt_parse = _parse_parallelism_from_args(extra_args + opt_extra_args)

    for key in ("tp", "pp", "dp"):
        if baseline_parse[key] != opt_parse[key]:
            raise SystemExit(
                f"per-label parallelism mismatch: baseline has {key}={baseline_parse[key]}, "
                f"opt has {key}={opt_parse[key]}. Parallelism flags must live in "
                f"bench.extra_args (shared) or both label-local lists must agree."
            )

    tp = baseline_parse["tp"]
    pp = baseline_parse["pp"]
    dp = baseline_parse["dp"]

    if baseline_parse["prefill_cp"] != 1 or opt_parse["prefill_cp"] != 1:
        raise SystemExit(
            f"prefill_context_parallel_size must be 1 (unsupported by this sweep flow); "
            f"got baseline={baseline_parse['prefill_cp']}, opt={opt_parse['prefill_cp']}."
        )

    # Backend coupling check (only when DP > 1).
    if dp > 1:
        for label, parsed in (("baseline", baseline_parse), ("opt", opt_parse)):
            backend = parsed["distributed_backend"]
            if backend is not None and backend != "external_launcher":
                raise SystemExit(
                    f"--data-parallel-size > 1 requires "
                    f"--distributed-executor-backend external_launcher; "
                    f"label {label!r} has backend={backend!r}. "
                    f"Remove it or let the sweep auto-inject external_launcher."
                )
        # Beam-search incompatibility.
        if baseline_parse["use_beam_search"] or opt_parse["use_beam_search"]:
            raise SystemExit(
                "beam-search + data-parallel-size > 1 is unsupported "
                "(undefined semantics under external_launcher). "
                "Drop --use-beam-search or set --data-parallel-size 1."
            )

    # Auto-inject backend when dp > 1 and neither label set it.
    augmented = list(extra_args)
    if (
        dp > 1
        and baseline_parse["distributed_backend"] is None
        and opt_parse["distributed_backend"] is None
    ):
        augmented = augmented + ["--distributed-executor-backend", "external_launcher"]

    nproc = max(1, tp) * max(1, pp) * max(1, dp)
    return (
        augmented,
        {"tp": tp, "pp": pp, "dp": dp, "prefill_cp": 1, "nproc": nproc},
    )


def _build_child_cmd(
    *,
    python_exe: str,
    script_path: Path,
    run_label: str,
    artifact_dir: Path,
    target_path: Path,
    timeout_s: int,
    out_name: str,
    out_root: Path,
    dp: int,
    nproc: int,
    extra_child_flags: List[str],
) -> List[str]:
    """Build the child command with torchrun dual-path selection (spec §4.2).

    When ``dp == 1``: produces ``[python, script.py, ...args]`` — the existing
    in-process child path.

    When ``dp > 1``: wraps with ``torch.distributed.run``:
        ``[python, -m, torch.distributed.run, --standalone, --nnodes=1,
           --nproc-per-node, N, script.py, ...args]``
    where ``nproc`` = tp * pp * dp.

    ``extra_child_flags`` are appended verbatim (nsys overrides, correctness
    flags, torch-profile flag, etc.). Ordering: forwarded-to-child flags come
    after the core script args and before torchrun's consumption of stderr.
    """
    # Note: --out-name is NOT passed to children. The resolved absolute path
    # is forwarded via --_out-root only. (`out_name` parameter is retained
    # in this function signature for back-compat with existing callers but is
    # not threaded into the child command.)
    _ = out_name  # explicit: out_name is no longer forwarded to children
    base_script_args: List[str] = [
        "--artifact-dir", str(artifact_dir),
        "--target-json", str(target_path),
        "--timeout-s", str(timeout_s),
        "--_out-root", str(out_root),
        "--_child-label", run_label,
    ] + list(extra_child_flags)

    if dp > 1:
        return [
            python_exe, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1",
            "--nproc-per-node", str(nproc),
            str(script_path),
        ] + base_script_args
    return [python_exe, str(script_path)] + base_script_args


def _should_skip_bucket(batch_size: int, dp_size: int) -> bool:
    """Skip a bucket when batch_size cannot be partitioned across all DP ranks.

    vLLM requires each DP rank to participate in the collective generate()
    call; if ``batch_size < dp_size`` at least one rank has no prompt of its
    own, and feeding it a placeholder biases latency upward — the spec chooses
    to skip such buckets rather than contaminate the measurement.
    """
    return dp_size > 1 and batch_size < dp_size


def _partition_prompts(
    all_prompts: List[Any],
    *,
    dp_size: int,
    dp_rank: int,
    input_len: int,
) -> List[Any]:
    """Return this rank's share of ``all_prompts`` using ``i % dp_size`` striping.

    Contract:
      * ``dp_size == 1``: returns ``all_prompts`` unchanged (identity).
      * ``dp_size > 1``: returns prompts at indices ``i`` with
        ``i % dp_size == dp_rank`` (canonical stride-based partition).
      * If a rank's share is empty (e.g. ``len(all_prompts) < dp_size``),
        returns ``[{"prompt_token_ids": [1] * input_len}]`` as a placeholder
        so vLLM's collective ``generate()`` does not hang on an empty list.

    Callers guard the ``batch_size < dp_size`` case via ``_should_skip_bucket``;
    the placeholder branch is a safety net for GSM8K / uneven correctness
    partitions where all ranks must participate.
    """
    if dp_size == 1:
        return all_prompts
    my_prompts = [p for i, p in enumerate(all_prompts) if i % dp_size == dp_rank]
    if not my_prompts:
        return [{"prompt_token_ids": [1] * input_len}]
    return my_prompts


def _nsys_start_if_rank0(torch_mod: Any) -> None:
    """Start nsys cudaProfiler capture (rank-0 only); sync on all ranks.

    torch.cuda.synchronize() must run on ALL ranks so the nsys capture-range
    boundary is aligned across DP siblings — nsys follows the rank-0
    cudaProfilerStart() globally via ``--capture-range=cudaProfilerApi``
    combined with ``--trace-fork-before-exec=true``, but per-rank streams are
    only in the same epoch if every rank has drained its device queue at the
    same wall-clock point.
    """
    torch_mod.cuda.synchronize()
    if _IS_RANK0:
        torch_mod.cuda.cudart().cudaProfilerStart()


def _nsys_stop_if_rank0(torch_mod: Any) -> None:
    """Stop nsys cudaProfiler capture (rank-0 only); sync on all ranks.

    Mirror of ``_nsys_start_if_rank0``. Must be called even on exception —
    leaving the capture range open corrupts the repeat:N counter and nsys
    hangs waiting for the matching stop.
    """
    torch_mod.cuda.synchronize()
    if _IS_RANK0:
        torch_mod.cuda.cudart().cudaProfilerStop()


def _check_correctness_dp_precondition(num_questions: int, dp_size: int) -> None:
    """Reject GSM8K runs where num_questions < dp_size (spec §4.4).

    Under DP>1 each rank must receive at least one real prompt — otherwise
    the all_gather_object at the end yields a rank with an empty list, which
    then gets mapped via _stitch_gathered to ``None`` entries, breaking
    ``_score_gsm8k_predictions``. Fail loud at the start of the correctness
    phase instead of deep inside the gather.
    """
    if dp_size > 1 and num_questions < dp_size:
        raise ValueError(
            f"correctness_num_questions={num_questions} < dp_size={dp_size}; "
            "every DP rank must receive at least one prompt. Increase "
            "--correctness-num-questions or run with --data-parallel-size 1."
        )


def _verdict_to_code(verdict: Dict[str, Any]) -> int:
    """Map a correctness verdict dict to an exit-code integer for DP broadcast.

    Codes: 0=PASS, 3=FAIL, 4=infrastructure error (or malformed verdict).
    ``infrastructure_error=True`` always maps to 4 regardless of the verdict
    string — the collector should treat infra issues as highest-severity.
    """
    if not isinstance(verdict, dict) or "verdict" not in verdict:
        return 4
    if verdict.get("infrastructure_error"):
        return 4
    return 0 if verdict.get("verdict") == "PASS" else 3


def _stitch_gathered(
    gathered: List[List[Any]],
    n_total: int,
    dp_size: int,
) -> List[Any]:
    """Reassemble per-rank outputs into canonical prompt order.

    Inverse of ``_partition_prompts``: for index ``i`` in ``[0, n_total)``,
    the owning rank is ``i % dp_size`` and the within-rank offset is
    ``i // dp_size``. Returns a list of length ``n_total`` in canonical order.
    """
    if dp_size == 1:
        # gathered is [[out0, out1, ...]] — flatten.
        return list(gathered[0])[:n_total]
    stitched: List[Any] = [None] * n_total
    for i in range(n_total):
        r = i % dp_size
        off = i // dp_size
        stitched[i] = gathered[r][off]
    return stitched


def _require(obj: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise SystemExit(f"Missing required field: {ctx}.{key}")
    val = obj[key]
    if _is_placeholder(val):
        raise SystemExit(f"Field still placeholder: {ctx}.{key}={val!r}")
    return val


def _require_int(obj: Dict[str, Any], key: str, ctx: str) -> int:
    val = _require(obj, key, ctx)
    if not isinstance(val, int):
        raise SystemExit(f"Expected int for {ctx}.{key}, got {type(val).__name__}")
    return val


def _require_list_int(obj: Dict[str, Any], key: str, ctx: str) -> List[int]:
    val = _require(obj, key, ctx)
    if not isinstance(val, list) or not all(isinstance(x, int) for x in val):
        raise SystemExit(f"Expected list[int] for {ctx}.{key}, got {val!r}")
    return val


def _maybe_list_str(obj: Dict[str, Any], key: str) -> List[str]:
    val = obj.get(key, [])
    if val is None:
        return []
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise SystemExit(f"Expected list[str] for bench.{key}, got {val!r}")
    return val


def _expand_workload_to_buckets(workload: Dict[str, Any]) -> List[Dict[str, int]]:
    """Normalize workload spec into a list of {input_len, output_len, batch_size} dicts.

    Supports two formats:
    1. New ``workload_matrix`` (list of dicts) — used when present, flat fields ignored.
    2. Legacy flat ``{input_len, output_len, batch_sizes}`` — expanded into matrix internally.
    """
    matrix = workload.get("workload_matrix")
    if matrix is not None:
        if not isinstance(matrix, list) or len(matrix) == 0:
            raise SystemExit("workload.workload_matrix must be a non-empty list")
        required_keys = {"input_len", "output_len", "batch_size"}
        buckets: List[Dict[str, int]] = []
        seen: set = set()
        for i, entry in enumerate(matrix):
            if not isinstance(entry, dict):
                raise SystemExit(f"workload.workload_matrix[{i}] must be a dict, got {type(entry).__name__}")
            missing = required_keys - set(entry.keys())
            if missing:
                raise SystemExit(f"workload.workload_matrix[{i}] missing keys: {missing}")
            bucket = {
                "input_len": int(entry["input_len"]),
                "output_len": int(entry["output_len"]),
                "batch_size": int(entry["batch_size"]),
            }
            key = (bucket["input_len"], bucket["output_len"], bucket["batch_size"])
            if key in seen:
                raise SystemExit(
                    f"Duplicate bucket in workload.workload_matrix: "
                    f"input_len={bucket['input_len']}, output_len={bucket['output_len']}, batch_size={bucket['batch_size']}"
                )
            seen.add(key)
            buckets.append(bucket)
        return buckets
    else:
        # Legacy flat format.
        input_len = workload.get("input_len")
        output_len = workload.get("output_len")
        batch_sizes = workload.get("batch_sizes")
        if input_len is None or output_len is None or batch_sizes is None:
            raise SystemExit("workload must have input_len, output_len, and batch_sizes (or workload_matrix)")
        return [
            {"input_len": int(input_len), "output_len": int(output_len), "batch_size": int(bs)}
            for bs in batch_sizes
        ]


def _bucket_file_tag(bucket: Dict[str, int], all_buckets: List[Dict[str, int]]) -> str:
    """Return a file-name tag for *bucket*.

    Returns ``bs{BS}`` when all buckets share the same (input_len, output_len)
    (homogeneous), otherwise ``il{IL}_ol{OL}_bs{BS}`` (heterogeneous).
    """
    il_ol_set = {(b["input_len"], b["output_len"]) for b in all_buckets}
    if len(il_ol_set) <= 1:
        return f"bs{bucket['batch_size']}"
    return f"il{bucket['input_len']}_ol{bucket['output_len']}_bs{bucket['batch_size']}"


def _validate_buckets_model_len(buckets: List[Dict[str, int]], max_model_len: int) -> None:
    """Raise SystemExit if any bucket exceeds max_model_len."""
    for b in buckets:
        total = b["input_len"] + b["output_len"]
        if total > max_model_len:
            raise SystemExit(
                f"Bucket (input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}) "
                f"requires {total} tokens but max_model_len={max_model_len}"
            )


def _parse_nsys_capture_output_steps(raw: Optional[str], output_len: int) -> List[int]:
    """Parse comma-separated nsys OL capture steps against ``output_len``.

    Example: ``"2,50%,100%"`` with ``output_len=512`` resolves to
    ``[2, 256, 512]``. Percentages use the profiling horizon, which is either
    the workload output length or ``--nsys-output-len`` when supplied.
    """
    if raw is None or not raw.strip():
        return []
    if output_len <= 0:
        raise SystemExit(f"output_len must be positive, got {output_len}")

    out: List[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise SystemExit("--nsys-capture-output-steps contains an empty item")
        if token.endswith("%"):
            pct_raw = token[:-1].strip()
            try:
                pct = float(pct_raw)
            except ValueError:
                raise SystemExit(f"Invalid percentage in --nsys-capture-output-steps: {token!r}")
            if pct <= 0.0 or pct > 100.0:
                raise SystemExit(
                    f"--nsys-capture-output-steps percentage must be in (0,100], got {token!r}"
                )
            step = max(1, int(round(output_len * pct / 100.0)))
        else:
            if not re.fullmatch(r"\d+", token):
                raise SystemExit(f"Invalid integer in --nsys-capture-output-steps: {token!r}")
            step = int(token)
        if step <= 0:
            raise SystemExit(f"--nsys-capture-output-steps values must be positive, got {step}")
        if step > output_len:
            raise SystemExit(
                f"--nsys-capture-output-steps value {step} exceeds profiling output_len {output_len}"
            )
        if step not in seen:
            out.append(step)
            seen.add(step)
    return out


# vLLM pins the chunked-prefill token budget per device: 16384 for >=70 GiB non-A100
# LLM_CLASS engines (B200/B300/H100/H200), 8192 for A100 and smaller
# (vllm/engine/arg_utils.py:2053-2068). For a selected-step DECODE capture we PIN this
# value into the profiling child (see _apply_selected_step_profiler_config) so the number
# of chunked-prefill worker-steps is deterministic and matches the arithmetic below. A
# decode step's shape (m == batch_size, KV depth) is independent of the prefill chunk size
# (confirmed: vLLM cudagraph dispatch / _is_uniform_decode key only on the current batch),
# so pinning is parity-safe for the capture AND A100-safe — we force the value rather than
# trusting the device default.
_PROFILING_CHUNK_TOKENS = 16384

# Extra decode worker-steps the capture arms PAST the last prefill chunk, so the captured
# step is steady-state (clear of cold-cache transients) — matching the headroom the
# Nemotron campaign used. 1 step covers the (provably <1-step) prefill-drain undercount;
# the remaining ~5 clear transients.
_SELECTED_STEP_DECODE_MARGIN = 6


def _n_prefill_worker_steps(input_len: int, batch_size: int, chunk_tokens: int) -> int:
    """Chunked-prefill worker-steps needed to drain ``input_len * batch_size`` tokens.

    vLLM's v1 scheduler shares ONE ``chunk_tokens`` token budget across the whole batch per
    worker step, so a long prompt is consumed over ``ceil(total_tokens / chunk_tokens)``
    prefill worker-steps before any pure decode step runs.
    """
    import math
    if chunk_tokens <= 0:
        raise SystemExit(f"Invalid profiling chunk_tokens: {chunk_tokens}")
    total_prefill_tokens = int(input_len) * int(batch_size)
    return max(1, math.ceil(total_prefill_tokens / int(chunk_tokens)))


def _selected_step_effective_window(
    buckets: List[Dict[str, int]],
    *,
    requested_window: int,
    nsys_output_len: Optional[int],
    nsys_capture_output_steps: str,
    chunk_tokens: int = _PROFILING_CHUNK_TOKENS,
    margin: int = _SELECTED_STEP_DECODE_MARGIN,
) -> int:
    """Child-wide capture window that clears chunked prefill for EVERY capture point.

    The selected-step mechanism shifts ``input_len`` so the captured (final) decode token
    of a short ``output_len == window`` generation sits at the requested decode depth: for
    a target step ``k`` it runs ``input_len_eff = src_il + k - window``. The capture arms on
    worker-step ``window`` (``delay_iterations == window``); that step is a clean full-batch
    decode iff ``window`` exceeds the chunked-prefill worker-steps
    ``n_prefill(input_len_eff, bs)``. The OLD code armed at ``window == 2`` and captured a
    prefill chunk on long-context workloads.

    ``n_prefill`` is non-increasing in ``window`` (a larger window means a smaller
    ``input_len_eff``), so evaluating it at the smallest plausible window
    (``requested_window``) over-estimates the prefill steps. The window returned therefore
    provably satisfies ``window > n_prefill(input_len_eff, bs)`` for every capture point in
    ONE pass — no fixed-point iteration. A single child-wide window keeps the existing
    one-window-per-child architecture, and the captured decode depth (``src_il + k``) is
    invariant under the window, so flooring it never distorts the requested depth.
    """
    floor = int(requested_window)
    for bucket in buckets:
        horizon = int(nsys_output_len) if nsys_output_len is not None else int(bucket["output_len"])
        for step in _parse_nsys_capture_output_steps(nsys_capture_output_steps, horizon):
            # Largest possible input_len_eff (smallest window) => largest n_prefill.
            il_eff_max = int(bucket["input_len"]) + int(step) - int(requested_window)
            n_prefill = _n_prefill_worker_steps(
                il_eff_max, int(bucket["batch_size"]), chunk_tokens
            )
            floor = max(floor, n_prefill + int(margin))
    return floor


def _expand_nsys_profile_buckets(
    buckets: List[Dict[str, int]],
    *,
    nsys_output_len: Optional[int] = None,
    nsys_capture_window_output_len: Optional[int] = None,
    nsys_capture_output_steps: Optional[str] = None,
    positions_per_step: int = 1,
) -> List[Dict[str, int]]:
    """Return the bucket list the nsys child should actually profile.

    ``--nsys-capture-output-steps`` creates shape-equivalent selected-step captures. For
    target output step ``k`` and capture window ``w``, the child runs a short generation and
    uses vLLM's CUDA profiler delay (``delay_iterations == w``, a WORKER-STEP count) to
    capture only the final worker step in that short window — landing the capture at decode
    depth ``input_len + k`` (invariant under ``w``).

    ``w`` is floored child-wide (``_selected_step_effective_window``) so the capture arms
    PAST all chunked-prefill worker-steps and lands on a genuine full-batch decode step
    rather than a prefill chunk. The user-supplied ``--nsys-capture-window-output-len`` is a
    lower bound on this floor, never an override below it.

    ``positions_per_step`` (= 1 + num_speculative_tokens; 1 for vanilla decode) decouples
    the WORKER-STEP arm threshold ``w`` from the OUTPUT-TOKEN count of the short generation.
    The profiler counts worker-steps, but each spec-decode worker-step advances
    ``positions_per_step`` output tokens, so a generation of ``output_len`` tokens completes
    in only ``n_prefill + ceil(output_len / positions_per_step)`` worker-steps. To guarantee
    the arm threshold ``w`` is reached for EVERY batch size, the generation output_len is
    sized ``w * positions_per_step`` (uniform), giving exactly ``w`` decode worker-steps
    (total ``n_prefill + w >= w``). At ``positions_per_step == 1`` this is ``output_len == w``,
    byte-for-byte the original behavior. ``delay_iterations`` / the capture-window floor
    stay UNCHANGED (still worker-step counts that clear prefill).
    """
    ppt = max(1, int(positions_per_step))
    expanded: List[Dict[str, int]] = []
    effective_window: Optional[int] = None
    if nsys_capture_output_steps:
        requested_window = (
            int(nsys_capture_window_output_len)
            if nsys_capture_window_output_len is not None
            else 2
        )
        if requested_window <= 0:
            raise SystemExit(
                f"--nsys-capture-window-output-len must be positive, got {requested_window}"
            )
        effective_window = _selected_step_effective_window(
            buckets,
            requested_window=requested_window,
            nsys_output_len=nsys_output_len,
            nsys_capture_output_steps=nsys_capture_output_steps,
        )
    for bucket in buckets:
        horizon = int(nsys_output_len) if nsys_output_len is not None else int(bucket["output_len"])
        if nsys_capture_output_steps:
            window = int(effective_window)
            for step in _parse_nsys_capture_output_steps(nsys_capture_output_steps, horizon):
                input_len_eff = int(bucket["input_len"]) + int(step) - window
                if input_len_eff < 1:
                    # The only sub-floor corner: context too short to host a steady-state
                    # full-batch decode at this window. Drop with a loud warning rather than
                    # emit a degenerate/prefill capture. Such a tiny context has no chunked
                    # prefill to clear anyway, so it is not the contamination class we fix.
                    print(
                        "WARNING: dropping selected-step capture "
                        f"(input_len={bucket['input_len']}, batch_size={bucket['batch_size']}, "
                        f"step={step}): effective capture window {window} >= input_len+step "
                        f"({int(bucket['input_len']) + int(step)}), leaving no room for a "
                        "decode-shaped capture. Use a larger input_len or a shallower step.",
                        file=sys.stderr,
                    )
                    continue
                expanded.append({
                    **bucket,
                    "input_len": input_len_eff,
                    # output_len sized window*ppt so the short generation runs exactly
                    # `window` decode worker-steps (ceil(window*ppt/ppt)==window) and the
                    # CUDA profiler arm threshold (delay_iterations==window) is reached for
                    # EVERY batch size under spec-decode. ppt==1 -> output_len==window
                    # (unchanged). delay_iterations stays == window (a worker-step count).
                    "output_len": window * ppt,
                    "nsys_source_input_len": int(bucket["input_len"]),
                    "nsys_source_output_len": int(bucket["output_len"]),
                    "nsys_capture_output_step": int(step),
                    "nsys_capture_window_output_len": window,
                    "nsys_capture_positions_per_step": ppt,
                    "nsys_capture_target_output_len": int(horizon),
                })
        elif nsys_output_len is not None:
            expanded.append({
                **bucket,
                "output_len": int(nsys_output_len),
                "nsys_source_output_len": int(bucket["output_len"]),
                "nsys_capture_target_output_len": int(nsys_output_len),
                "nsys_capture_positions_per_step": ppt,
            })
        else:
            expanded.append({
                **bucket,
                "nsys_capture_positions_per_step": ppt,
            })
    return expanded


def _dedupe_nsys_profile_buckets(
    buckets: List[Dict[str, int]],
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """Drop duplicate effective nsys shapes while preserving first occurrence."""
    deduped: List[Dict[str, int]] = []
    skipped: List[Dict[str, int]] = []
    seen: set[Tuple[int, int, int]] = set()
    for bucket in buckets:
        key = (
            int(bucket["input_len"]),
            int(bucket["output_len"]),
            int(bucket["batch_size"]),
        )
        if key in seen:
            skipped.append(dict(bucket))
            continue
        seen.add(key)
        deduped.append(bucket)
    return deduped, skipped


def _format_nsys_deduped_buckets(skipped: List[Dict[str, int]], limit: int = 8) -> str:
    """Human-readable summary for duplicate effective nsys shapes."""
    shown = []
    for bucket in skipped[:limit]:
        shown.append(
            "il={il},ol={ol},bs={bs},source_ol={source_ol}".format(
                il=bucket["input_len"],
                ol=bucket["output_len"],
                bs=bucket["batch_size"],
                source_ol=bucket.get("nsys_source_output_len", bucket["output_len"]),
            )
        )
    more = "" if len(skipped) <= limit else f" (+{len(skipped) - limit} more)"
    return ", ".join(shown) + more


def _nsys_tag_buckets_for_dp(
    buckets: List[Dict[str, int]],
    dp_size: int,
    *,
    nsys_output_len: Optional[int] = None,
    nsys_capture_window_output_len: Optional[int] = None,
    nsys_capture_output_steps: Optional[str] = None,
    positions_per_step: int = 1,
) -> List[Dict[str, int]]:
    """Profile buckets expected to emit nsys reports after DP skip filtering."""
    expanded = _expand_nsys_profile_buckets(
        buckets,
        nsys_output_len=nsys_output_len,
        nsys_capture_window_output_len=nsys_capture_window_output_len,
        nsys_capture_output_steps=nsys_capture_output_steps,
        positions_per_step=positions_per_step,
    )
    filtered = [b for b in expanded if not _should_skip_bucket(int(b["batch_size"]), dp_size)]
    deduped, _ = _dedupe_nsys_profile_buckets(filtered)
    return deduped


def _is_nsys_selected_step_bucket(bucket: Dict[str, int]) -> bool:
    return "nsys_capture_output_step" in bucket


def _nsys_runner_mapping(bucket: Dict[str, int]) -> Dict[str, int]:
    """Return the source/capture mapping recorded in a profiling runner sidecar."""
    if not _is_nsys_selected_step_bucket(bucket):
        return {}
    return {
        "source_input_len": int(bucket["nsys_source_input_len"]),
        "source_output_len": int(bucket["nsys_source_output_len"]),
        "capture_output_step": int(bucket["nsys_capture_output_step"]),
        "capture_window_output_len": int(
            bucket["nsys_capture_window_output_len"]
        ),
        "capture_target_output_len": int(
            bucket["nsys_capture_target_output_len"]
        ),
    }


def _compute_nsys_cudagraph_capture_sizes(buckets: List[Dict[str, int]]) -> List[int]:
    """Bound nsys CUDA graph capture sizes to profiled nominal/effective shapes."""
    sizes: set[int] = set()
    for bucket in buckets:
        batch_size = int(bucket["batch_size"])
        sizes.add(batch_size)
        if "nsys_capture_positions_per_step" in bucket:
            positions_per_step = max(1, int(bucket["nsys_capture_positions_per_step"]))
            sizes.add(batch_size * positions_per_step)
    return sorted(sizes)


def _apply_selected_step_profiler_config(
    ea_dict: Dict[str, Any],
    bucket: Dict[str, int],
) -> None:
    """Configure vLLM's built-in CUDA profiler for selected-step nsys capture.

    ``WorkerProfiler.step()`` (vllm/profiler/wrapper.py) increments a counter on EVERY
    worker step and arms the profiler when the counter == ``delay_iterations``;
    ``gpu_worker.annotate_profile()`` calls it once per scheduler iteration. The capture
    window stored on the bucket was floored child-wide (``_selected_step_effective_window``)
    to exceed the chunked-prefill worker-step count, so arming at ``delay == window`` lands
    on a steady-state full-batch decode step rather than a prefill chunk.

    ``delay_iterations`` is a WORKER-STEP count and stays == ``window``. Under spec-decode the
    short generation produces only ``n_prefill + ceil(output_len / positions_per_step)``
    worker-steps; ``_expand_nsys_profile_buckets`` sizes ``output_len = window *
    positions_per_step`` so that total is ``n_prefill + window >= window`` for every bucket and
    the profiler always arms. We re-derive that count here and fail LOUD (SystemExit) if the
    generation is too short to reach the threshold — converting a silent never-arm (the
    BS8/BS32-never-emitted bug) into an explicit error. At ``positions_per_step == 1`` the
    available worker-steps are ``n_prefill + window``, so this assertion never fires for
    vanilla decode.

    We also PIN ``max_num_batched_tokens`` to the chunk size used in that floor arithmetic
    so the engine's prefill-step count matches ours deterministically (A100-safe: forced,
    not read from the device default). The pin affects only this profiling child; a decode
    step's shape is independent of the prefill chunk size.
    """
    import math

    window = int(bucket["nsys_capture_window_output_len"])
    if window <= 0:
        raise SystemExit(f"Invalid selected-step capture window: {window}")
    prior = ea_dict.get("max_num_batched_tokens")
    if prior is not None and int(prior) != _PROFILING_CHUNK_TOKENS:
        print(
            "WARNING: pinning max_num_batched_tokens="
            f"{_PROFILING_CHUNK_TOKENS} for selected-step decode capture "
            f"(was {prior}); this profiling child's prefill-step count must match the "
            "capture-window arithmetic. Perf-measurement children are unaffected.",
            file=sys.stderr,
        )
    ea_dict["max_num_batched_tokens"] = _PROFILING_CHUNK_TOKENS
    n_prefill = _n_prefill_worker_steps(
        int(bucket["input_len"]), int(bucket["batch_size"]), _PROFILING_CHUNK_TOKENS
    )
    if window <= n_prefill:
        # Defensive: the child-wide floor guarantees window > n_prefill for every bucket.
        # If this fires, the window was set without going through
        # _selected_step_effective_window — fail loud rather than capture prefill.
        raise SystemExit(
            f"Selected-step capture window {window} does not clear the "
            f"{n_prefill} chunked-prefill worker-step(s) for bucket "
            f"(input_len={bucket['input_len']}, batch_size={bucket['batch_size']}); "
            "the capture would land on a prefill chunk. This indicates the window was "
            "not floored via _selected_step_effective_window."
        )
    # Arm-reachability assertion (spec-decode-aware): the profiler arms only when the
    # worker-step counter reaches delay_iterations==window. The short generation runs
    # n_prefill + ceil(output_len/ppt) worker-steps; if that is below window the profiler
    # NEVER arms (the silent BS8/BS32-no-rep failure under spec-decode). After the
    # _expand_nsys_profile_buckets sizing (output_len=window*ppt) this is always
    # n_prefill+window >= window; this guard makes any residual mis-sizing fail loud.
    ppt = max(1, int(bucket.get("nsys_capture_positions_per_step", 1)))
    out_len = int(bucket.get("output_len", window))
    available_ws = n_prefill + math.ceil(out_len / ppt)
    if available_ws < window:
        raise SystemExit(
            f"Selected-step generation (output_len={out_len}, positions_per_step={ppt}) runs "
            f"only {available_ws} worker-step(s) for bucket "
            f"(input_len={bucket['input_len']}, batch_size={bucket['batch_size']}), below "
            f"delay_iterations={window}; the CUDA profiler would never arm. output_len must "
            "be sized window*positions_per_step via _expand_nsys_profile_buckets."
        )
    ea_dict["profiler_config"] = {
        "profiler": "cuda",
        "delay_iterations": window,
        "max_iterations": 1,
    }


# Marker format (vllm/v1/worker/gpu_worker.py):
#   context_{num_ctx_requests}({num_ctx_tokens})_generation_{num_gen_requests}({num_gen_tokens})
# ctx captures num_ctx_TOKENS (>0 only during prefill); gen captures num_generation_TOKENS;
# the new gen_reqs group captures num_generation_REQUESTS (the active-request count, which
# under continuous batching can differ from the nominal --batch-size). Under spec-decode
# num_generation_tokens == num_generation_requests * positions_per_step.
_NVTX_DECODE_MARKER_RE = re.compile(
    r"context_\d+\((?P<ctx>\d+)\)_generation_(?P<gen_reqs>\d+)\((?P<gen>\d+)\)"
)


def evaluate_decode_shape(
    *,
    nvtx_marker: Optional[str],
    rmsnorm_grid_x: Optional[int],
    paged_kv_attn_total_us: Optional[float],
    prefill_flash_attn_total_us: Optional[float],
    batch_size: int,
    positions_per_step: int = 1,
) -> Tuple[bool, str]:
    """Pure decode-vs-prefill classifier for a captured nsys forward pass.

    Returns ``(is_decode, reason)``. A genuine steady-state decode step shows: the forward
    NVTX marker is ``context_R(0)_generation_R(N)`` with no prefill context tokens
    (num_ctx_tokens == 0); token-count-driven kernels run at a grid sized to the per-step
    token count (RMSNorm gridX small, not thousands); and decode paged-KV attention dominates
    prefill-flash attention. A chunked-prefill forward shows the inverse. This is the guard
    that prevents a prefill-shaped capture from being mined as decode (the contamination class
    documented in INVESTIGATION_prefill_contamination.md).

    ``positions_per_step`` (= 1 + num_speculative_tokens; 1 for vanilla decode) makes the
    classifier spec-decode-aware:

      * Vanilla (ppt <= 1): the EXACT original predicate — decode iff num_ctx_tokens == 0 AND
        num_generation_tokens == batch_size — and the gridX vote with threshold batch_size*4.
      * Spec-decode (ppt > 1): each decode worker-step schedules ``ppt`` tokens per active
        request, so num_generation_tokens == num_generation_REQUESTS * ppt. We accept
        num_ctx_tokens == 0 AND gen > 0 AND gen % ppt == 0 AND gen_reqs >= 1 AND
        gen == gen_reqs * ppt. The active-request count R = gen // ppt is NOT compared to the
        nominal batch_size: continuous batching legitimately runs more/fewer concurrent
        requests than the bench --batch-size (the real artifact shows gen_reqs=10 at nominal
        bs in [1,32]).

    For both vanilla and speculative decode, an exact decode-shaped marker demotes the
    RMSNorm-name gridX vote to non-scoring. Compiled graphs can fuse RMSNorm with independently
    tiled layout work, so the maximum grid among names containing ``rms_norm`` is less specific
    than the scheduler's direct context/generation counts. A prefill-shaped marker still fails
    unconditionally, and an oversized RMSNorm grid still fails when no exact marker is present.

    Evidence is combined conservatively: the NVTX marker is dispositive when present;
    otherwise any one prefill-shaped signal fails the check. Signals whose kernel symbols are
    absent (e.g. Blackwell-only attention names on a different GPU) are SKIPPED, not failed,
    so the classifier degrades gracefully across architectures.
    """
    ppt = max(1, int(positions_per_step))
    reasons: List[str] = []
    decode_signals = 0
    prefill_signals = 0
    marker_confirmed_decode = False

    if nvtx_marker:
        m = _NVTX_DECODE_MARKER_RE.search(nvtx_marker)
        if m:
            ctx = int(m.group("ctx"))
            gen = int(m.group("gen"))
            gr = m.groupdict().get("gen_reqs")
            gen_reqs = int(gr) if gr is not None else None
            if ppt <= 1 or gen_reqs is None:
                # Vanilla decode (or pre-gen_reqs marker): EXACT original predicate.
                decode_ok = ctx == 0 and gen == batch_size
                expected = f"context=0 generation={batch_size}"
            else:
                # Spec-decode: num_generation_tokens == active_requests * positions_per_step.
                # active_requests (=gen//ppt) is NOT capped at the nominal batch_size.
                decode_ok = (
                    ctx == 0
                    and gen > 0
                    and gen % ppt == 0
                    and gen_reqs >= 1
                    and gen == gen_reqs * ppt
                )
                expected = f"context=0 generation==gen_reqs*{ppt}"
            if decode_ok:
                decode_signals += 1
                # The scheduler marker directly reports zero context tokens and the expected
                # generation shape. It is therefore more specific than the RMSNorm-name grid
                # heuristic, which can match unrelated compiled fusion/layout kernels.
                marker_confirmed_decode = True
                reasons.append(
                    "NVTX marker decode-shaped "
                    f"(context_*(0)_generation_{gen_reqs}({gen}), ppt={ppt})"
                )
            else:
                prefill_signals += 1
                reasons.append(
                    f"NVTX marker PREFILL-shaped (context={ctx}, gen_reqs={gen_reqs}, "
                    f"generation={gen}, expected {expected})"
                )
        else:
            reasons.append(f"NVTX marker unrecognized ({nvtx_marker!r})")

    if rmsnorm_grid_x is not None:
        if marker_confirmed_decode:
            # Marker-confirmed decode: RMSNorm-name gridX is not a reliable per-token-count
            # proxy for compiled graphs. A name can include rms_norm while also fusing
            # cat/clone/slice/layout work with independent tiling. Do not let that heuristic
            # override direct scheduler evidence. Other prefill signals remain scored.
            reasons.append(
                f"RMSNorm gridX={rmsnorm_grid_x} (not scored; NVTX marker dispositive "
                "for decode shape)"
            )
        elif rmsnorm_grid_x <= max(1, batch_size) * ppt * 4:
            decode_signals += 1
            reasons.append(f"RMSNorm gridX={rmsnorm_grid_x} ~= batch_size*ppt")
        else:
            prefill_signals += 1
            reasons.append(
                f"RMSNorm gridX={rmsnorm_grid_x} >> batch_size*ppt="
                f"{max(1, batch_size) * ppt} (prefill token count)"
            )

    if (
        paged_kv_attn_total_us is not None
        and prefill_flash_attn_total_us is not None
    ):
        if paged_kv_attn_total_us >= prefill_flash_attn_total_us:
            decode_signals += 1
            reasons.append(
                f"paged-KV attn ({paged_kv_attn_total_us:.1f}us) >= "
                f"prefill-flash ({prefill_flash_attn_total_us:.1f}us)"
            )
        else:
            prefill_signals += 1
            reasons.append(
                f"prefill-flash attn ({prefill_flash_attn_total_us:.1f}us) > "
                f"paged-KV ({paged_kv_attn_total_us:.1f}us) (prefill-dominated)"
            )

    summary = "; ".join(reasons) if reasons else "no decode-shape signals available"
    if prefill_signals > 0:
        return False, f"PREFILL-contaminated capture: {summary}"
    if decode_signals == 0:
        return False, f"could not confirm decode shape: {summary}"
    return True, f"decode-shaped capture confirmed: {summary}"


def _extract_decode_shape_signals(sqlite_path: str, batch_size: int) -> Dict[str, Any]:
    """Extract the decode-shape signals from a captured nsys sqlite export."""
    import sqlite3
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.cursor()
        tables = {
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        nvtx_marker = None
        if "NVTX_EVENTS" in tables:
            row = cur.execute(
                "SELECT text FROM NVTX_EVENTS WHERE text LIKE '%context_%generation_%' "
                "ORDER BY (end-start) DESC LIMIT 1"
            ).fetchone()
            if row:
                nvtx_marker = row[0]

        rmsnorm_grid_x = None
        paged = None
        flash = None
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables and "StringIds" in tables:
            r = cur.execute(
                "SELECT k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id "
                "WHERE s.value LIKE '%rms_norm%' ORDER BY k.gridX DESC LIMIT 1"
            ).fetchone()
            if r:
                rmsnorm_grid_x = int(r[0])

            def _attn_total(like: str):
                rr = cur.execute(
                    "SELECT SUM(k.end-k.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE ?",
                    (like,),
                ).fetchone()
                return float(rr[0]) if rr and rr[0] is not None else None
            paged = _attn_total("%fmhaSm100%PagedKv%")
            flash = _attn_total("%flash_fwd_sm100%")

        return {
            "nvtx_marker": nvtx_marker,
            "rmsnorm_grid_x": rmsnorm_grid_x,
            "paged_kv_attn_total_us": paged,
            "prefill_flash_attn_total_us": flash,
        }
    finally:
        con.close()


def assert_decode_shaped_capture(
    sqlite_path: str, batch_size: int, positions_per_step: int = 1
) -> None:
    """Hard guard: SystemExit if a selected-step nsys capture is prefill-shaped.

    Run on each selected-step sqlite before mining derives any 'decode-graph' component
    shares. Prevents the prefill-as-decode contamination class
    (INVESTIGATION_prefill_contamination.md) from silently recurring.

    ``positions_per_step`` (= 1 + num_speculative_tokens) makes the underlying classifier
    spec-decode-aware so a genuine spec-decode bulk-decode step (num_generation_tokens ==
    active_requests * positions_per_step) is accepted rather than false-rejected.
    """
    signals = _extract_decode_shape_signals(sqlite_path, batch_size)
    is_decode, reason = evaluate_decode_shape(
        batch_size=batch_size, positions_per_step=positions_per_step, **signals
    )
    if not is_decode:
        raise SystemExit(
            f"ERROR: nsys capture {sqlite_path} is NOT a steady-state decode step.\n"
            f"  {reason}\n\n"
            "The selected-step profiler armed on a non-decode forward (typically a "
            "chunked-prefill chunk on a long-context workload). Mining this trace as "
            "'decode-graph %' produces a prefill-weighted bottleneck ranking that is wrong "
            "for a decode-dominated workload.\n"
            "Fix: ensure the capture window clears all prefill worker-steps "
            "(_selected_step_effective_window) and that the profiling child pins "
            "max_num_batched_tokens. Re-profile before mining."
        )


def _export_nsys_sqlite(nsys_rep: Path) -> Optional[Path]:
    """Export an ``.nsys-rep`` to a sibling ``.sqlite`` for the decode-shape guard.

    Returns the sqlite path on success, or ``None`` if ``nsys`` is unavailable or the export
    fails (the caller treats that as "guard could not run" rather than a hard error, since a
    missing exporter is an environment issue, not a contaminated capture).
    """
    sqlite_path = nsys_rep.with_suffix(".sqlite")
    if sqlite_path.exists():
        return sqlite_path
    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        return None
    try:
        proc = subprocess.run(
            [
                nsys_bin, "export",
                "--type", "sqlite",
                "--force-overwrite=true",
                "--output", str(sqlite_path),
                str(nsys_rep),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not sqlite_path.exists():
        return None
    return sqlite_path


def _bench_exe_tokens(vllm_cmd: Any) -> List[str]:
    if isinstance(vllm_cmd, list):
        if not all(isinstance(x, str) for x in vllm_cmd):
            raise SystemExit(f"bench.vllm_cmd list must be list[str], got {vllm_cmd!r}")
        return vllm_cmd
    if isinstance(vllm_cmd, str):
        return shlex.split(vllm_cmd)
    raise SystemExit(f"bench.vllm_cmd must be str or list[str], got {type(vllm_cmd).__name__}")


_LAT_RE = re.compile(r"^\s*(?P<key>Avg latency|\d+% percentile latency):\s*(?P<val>[0-9.eE+-]+)\s*seconds\s*$", re.MULTILINE)


def _parse_latency_metrics(stdout: str) -> Dict[str, float]:
    """Parse vllm bench latency stdout.

    Expected lines resemble:
      Avg latency: 10.9455 seconds
      50% percentile latency: 10.9064 seconds

    If this format changes, do NOT guess; just return empty dict.
    """
    out: Dict[str, float] = {}
    for m in _LAT_RE.finditer(stdout):
        key = m.group("key").strip()
        val_s = m.group("val")
        try:
            val = float(val_s)
        except ValueError:
            continue
        if key == "Avg latency":
            out["avg_s"] = val
        else:
            # "50% percentile latency" -> p50_s
            pct = key.split("%", 1)[0]
            if pct.isdigit():
                out[f"p{pct}_s"] = val
    return out


def _check_patterns(text: str, require: List[str], forbid: List[str]) -> Dict[str, Any]:
    """Return pattern check results without throwing.

    Patterns are treated as regex. We record which matched.
    """
    req_hits = []
    req_miss = []
    for pat in require:
        if re.search(pat, text):
            req_hits.append(pat)
        else:
            req_miss.append(pat)
    forb_hits = []
    for pat in forbid:
        if re.search(pat, text):
            forb_hits.append(pat)

    ok = (len(req_miss) == 0) and (len(forb_hits) == 0)
    return {
        "ok": ok,
        "require_hits": req_hits,
        "require_miss": req_miss,
        "forbid_hits": forb_hits,
    }


def _format_cmd_for_md(cmd: List[str], env_overrides: Dict[str, str]) -> str:
    # Keep it copy/pasteable.
    env_prefix = " ".join([f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()])
    cmd_str = " ".join([shlex.quote(x) for x in cmd])
    return (env_prefix + " " + cmd_str).strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# Rank-0-gated write/log helpers (see §4.3 header block near _NullFile).
# Under DP=1 (RANK unset → _IS_RANK0=True) these are straight delegates.

def _rank0_write_json(path: Path, obj: Dict[str, Any]) -> None:
    if _IS_RANK0:
        _write_json(path, obj)


def _rank0_write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    if _IS_RANK0:
        _write_json_atomic(path, obj)


def _rank0_write_text(path: Path, text: str) -> None:
    if _IS_RANK0:
        _write_text(path, text)


def _rank0_log(msg: str) -> None:
    """Rank-0-only stdout log. Silent on non-rank-0 to keep parent stdout clean."""
    if _IS_RANK0:
        print(msg, flush=True)


def _rank0_open_log(path: Path, mode: str) -> Any:
    """Open a log file on rank 0; return a _NullFile sink on other ranks.

    Keeps shared bucket-loop code paths free of per-site ``if _IS_RANK0`` guards.
    """
    if _IS_RANK0:
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, mode, encoding="utf-8", buffering=1)
    return _NullFile()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Expected JSON output not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _metrics_from_vllm_latency_json(obj: Dict[str, Any]) -> Dict[str, float]:
    """Convert vllm bench latency JSON into this script's metrics dict.

    vLLM JSON schema (as of vllm/benchmarks/latency.py):
      {"avg_latency": float, "latencies": [...], "percentiles": {"10": float, ...}}

    Per spec §3.1, the AMMO sweep additionally emits per-bucket prefill/decode
    timing harvested from ``RequestOutput.metrics``:
      ``prefill_avg_s``, ``decode_avg_s``, ``prefill_p50_s``, ``decode_p50_s``,
      ``decode_share_of_e2e``.
    These propagate into the metrics dict so downstream consumers
    (``_build_label_result_entry`` and ``e2e_latency_results.json``) can
    surface them as top-of-row siblings to ``batch_size`` / ``improvement_pct``.
    """
    out: Dict[str, float] = {}
    avg = obj.get("avg_latency")
    if isinstance(avg, (int, float)):
        out["avg_s"] = float(avg)
    percentiles = obj.get("percentiles", {})
    if isinstance(percentiles, dict):
        for k, v in percentiles.items():
            if isinstance(k, str) and k.isdigit() and isinstance(v, (int, float)):
                out[f"p{k}_s"] = float(v)
    # New: pass-through prefill/decode timing fields when present + numeric.
    for key in (
        "prefill_avg_s",
        "decode_avg_s",
        "prefill_p50_s",
        "decode_p50_s",
        "decode_share_of_e2e",
        "decode_share_of_phase_sum",
    ):
        v = obj.get(key)
        if isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _sanitize_filename(s: str) -> str:
    # Keep it conservative for shared filesystems.
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s).strip("_") or "default"


def _pick_archive_path(path: Path, run_id: str) -> Path:
    """Pick a unique archive name for an existing output directory."""
    base = path.with_name(f"{path.name}_{run_id}")
    if not base.exists():
        return base
    for i in range(1, 1000):
        cand = path.with_name(f"{path.name}_{run_id}:{i}")
        if not cand.exists():
            return cand
    raise SystemExit(f"Failed to pick unique archive name for {path}")


# ---------------------------------------------------------------------------
# Layout v2: round-scoped path resolution
# ---------------------------------------------------------------------------
#
# Reference: ai_cli_session/.codex/skills/ammo/references/artifact-layout.md
#
# v2 layout is detected by filesystem presence of `rounds/` under
# {artifact_dir}. When v2: sweep output goes to rounds/{N}/sweeps/{slot}/,
# and archive moves to rounds/{N}/_archive/{slot}_{ts}/.
# When NOT v2 (old campaigns): legacy --out-name fallback applies.

def _is_v2_layout(artifact_dir: Path) -> bool:
    """Return True iff `{artifact_dir}/rounds/` exists (filesystem-only check).

    No coupling to state.json["schema_version"]. Layout is purely structural.
    """
    return (artifact_dir / "rounds").is_dir()


def _read_current_round_from_state(artifact_dir: Path) -> int:
    """Read `campaign.current_round` from state.json. Defaults to 1 if missing."""
    state_path = artifact_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state["campaign"]["current_round"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return 1


def _resolve_sweep_out_name(args, artifact_dir: Path) -> str:
    """Resolve the sweep `out_name` (relative to artifact_dir) per spec §Path Resolution.

    Resolution logic (in order):
      1. `--round N --slot SLOT`            -> "rounds/{N}/sweeps/{SLOT}"
      2. `--round N` only                   -> SystemExit (slot required)
      3. `--slot SLOT` only (no --round)    -> read current_round from state.json
      4. `--_out-root` set (child run)      -> bypass entirely (caller uses _out_root)
      5. `--out-name` explicitly set on a v2 layout (no --round/--slot) ->
         hard error pointing to --round/--slot
      6. v1 (legacy) layout                 -> return args.out_name unchanged

    Returns a path string relative to artifact_dir.
    """
    # Child run: caller passes --_out-root directly; no resolution needed.
    if getattr(args, "_out_root", None):
        return getattr(args, "out_name", "e2e_latency")

    round_ = getattr(args, "round", None)
    slot = getattr(args, "slot", None)

    if round_ is not None and slot is not None:
        return f"rounds/{round_}/sweeps/{slot}"
    if round_ is not None and slot is None:
        raise SystemExit(
            "ERROR: --round requires --slot. Pass --slot baseline | profiling | "
            "opt/{op_id} | opt_correctness/{op_id} | opt_profiling/{op_id} | "
            "integration | integration_profiling | golden_capture "
            "(see references/artifact-layout.md)."
        )
    if slot is not None:
        # No --round given: derive from state.json.
        if not _is_v2_layout(artifact_dir):
            raise SystemExit(
                f"ERROR: --slot was passed but {artifact_dir}/rounds/ does not exist. "
                "Either run new_target.py to scaffold v2, or omit --slot for legacy mode."
            )
        n = _read_current_round_from_state(artifact_dir)
        return f"rounds/{n}/sweeps/{slot}"

    # Neither --round nor --slot given.
    out_name = getattr(args, "out_name", "e2e_latency")
    if _is_v2_layout(artifact_dir) and out_name != "e2e_latency":
        # v2 layout + non-default --out-name => caller is using the removed flag.
        raise SystemExit(
            f"ERROR: --out-name is removed for v2 layouts. "
            f"Use --round N --slot <baseline|profiling|opt/{{op_id}}|opt_correctness/{{op_id}}|opt_profiling/{{op_id}}|integration|integration_profiling|golden_capture> instead. "
            f"See references/artifact-layout.md for path resolution rules."
        )
    # Legacy (v1) artifact dir: default out_name fallback applies.
    return out_name


def _is_gate_slot(slot: Optional[str]) -> bool:
    """Return whether a V2 slot contributes binding Stage 5/6 evidence."""
    return bool(slot) and (
        slot == "integration"
        or slot == "integration_profiling"
        or slot.startswith("opt/")
        or slot.startswith("opt_correctness/")
        or slot.startswith("opt_profiling/")
    )


def _prepare_out_root(
    *,
    artifact_dir: Path,
    out_name: str,
    overwrite: bool,
) -> Path:
    """Create/clean the sweep output root.

    For v2 paths (`out_name` matches "rounds/{N}/sweeps/{slot}..."), existing
    contents are archived to `rounds/{N}/_archive/{slot}_{ts}/`. For legacy
    out_names, archive is colocated with the original (legacy behavior).
    """
    out_root = artifact_dir / out_name
    if out_root.exists() and any(out_root.iterdir()):
        if overwrite:
            # Be explicit: overwrite means discard previous evidence.
            shutil.rmtree(out_root, ignore_errors=True)
        else:
            run_id = _utc_run_id()
            archived = _pick_v2_archive_path(artifact_dir, out_name, run_id)
            if archived is None:
                # Legacy fallback: colocated archive with timestamp suffix.
                archived = _pick_archive_path(out_root, run_id)
            print(f"Archiving existing output dir: {out_root} -> {archived}")
            archived.parent.mkdir(parents=True, exist_ok=True)
            out_root.replace(archived)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _v2_profiling_dir(out_root: Path, kind: str) -> Path:
    """Return the v2 sibling profiling path for `kind`.

    For v2 sweep paths (`{artifact}/rounds/{N}/sweeps/{slot}/...`) returns
    `{artifact}/rounds/{N}/profiling/{kind}/`. For legacy out_roots (no
    `rounds/{N}/sweeps/` ancestors) falls back to `out_root / kind`. New AMMO
    campaigns use `nsys` and targeted `ncu`; other kinds are compatibility or
    manual fallback paths.
    """
    parts = out_root.parts
    # Find a "rounds/{N}/sweeps" segment in the path so we can hop to the
    # round-scoped profiling sibling.
    for i in range(len(parts) - 2):
        if parts[i] == "rounds" and parts[i + 2] == "sweeps":
            round_num = parts[i + 1]
            # Reconstruct the artifact root = parts[:i].
            artifact_root = Path(*parts[:i]) if i > 0 else Path(parts[0])
            base = artifact_root / "rounds" / round_num / "profiling" / kind
            slot_parts = parts[i + 3 :]
            if len(slot_parts) >= 2 and slot_parts[0] == "opt_profiling":
                return base / "opt" / slot_parts[1]
            if slot_parts and slot_parts[0] == "integration_profiling":
                return base / "integration"
            if slot_parts and slot_parts[0] == "post_ship_profiling":
                return base / "post_ship"
            return base
    # Legacy fallback: colocate inside the sweep output.
    return out_root / kind


def _pick_v2_archive_path(artifact_dir: Path, out_name: str, run_id: str):
    """If `out_name` is a v2 sweep path, return rounds/{N}/_archive/{slot}_{run_id}/.

    Returns None for legacy paths so the caller falls back to colocated archive.
    `out_name` shape: "rounds/{N}/sweeps/{slot...}".
    """
    parts = Path(out_name).parts
    if len(parts) >= 4 and parts[0] == "rounds" and parts[2] == "sweeps":
        round_num = parts[1]
        # slot may be multi-segment (e.g. opt/op007); flatten to a single name.
        slot_segments = parts[3:]
        slot_name = "_".join(slot_segments) if slot_segments else "unknown"
        archive_dir = artifact_dir / "rounds" / round_num / "_archive"
        base = archive_dir / f"{slot_name}_{run_id}"
        if not base.exists():
            return base
        for i in range(1, 1000):
            cand = archive_dir / f"{slot_name}_{run_id}:{i}"
            if not cand.exists():
                return cand
        raise SystemExit(f"Failed to pick unique v2 archive name under {archive_dir}")
    return None


def _hang_stale_limit_s(
    *, input_len: int, output_len: int, batch_size: int, base_s: int
) -> int:
    """Phase-gated staleness threshold (seconds) for the hang watchdog.

    The child rewrites its status file every ~5s DURING warmup/benchmark, but the
    genuine in-engine hang wedges inside a single ``generate()`` that never
    returns — so the status freezes and stays frozen. The threshold must exceed
    the longest LEGITIMATE single-iteration time for the bucket (one warmup or
    timed generate) with margin, so a slow-but-healthy heavy bucket is not
    false-killed. Heavy prefill / large batch make one iteration take many
    seconds, so widen for those; the real hang sits frozen for minutes, far past
    any of these. Mirrors the values validated live on the bs64/long-prefill
    shapes during the PR #45717 sweep.
    """
    s = int(base_s)
    if input_len >= 100000:
        s = max(s, 360)
    if batch_size >= 64:
        s = max(s, 360)
    if output_len >= 512:
        s += 60
    return s


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """Best-effort: SIGTERM then SIGKILL the child's whole process group.

    The child is launched with ``start_new_session=True`` so it leads its own
    process group (PGID == child PID); vLLM's EngineCore subprocesses inherit
    that group, so killing the group reaps them too — no kill-by-name (which
    would be unsafe in a shared host) and no per-PID hunting. Falls back to a
    plain proc kill if the group signal is unavailable.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
            return
        except Exception:
            continue


def _run_cmd_streaming_watchdog(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: Optional[Path],
    timeout_s: int,
    log_path: Path,
    heartbeat_s: int,
    status_path: Path,
    bucket_meta: Dict[str, Dict[str, int]],
    base_stale_s: int,
) -> Dict[str, Any]:
    """Like ``_run_cmd_streaming`` but kills the child's process GROUP when its
    status file goes stale past the phase-gated threshold while it is wedged
    inside ``generate()`` (the per-bucket deadline cannot catch that — it is only
    checked BETWEEN generate() calls).

    Returns the same dict as ``_run_cmd_streaming`` plus ``hung`` (bool) and
    ``wedged_tag`` (the bucket tag that was frozen when we killed, or ""). Only
    used when ``--hang-watchdog`` is set; the model-load phase (status frozen on
    a NON warmup/benchmark phase, or no status yet) uses a generous grace so a
    legitimately long load is never mistaken for a hang.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    hung = False
    wedged_tag = ""
    # Grace for the model-load / import phase, where the status legitimately does
    # not tick (one big blocking LLM() construction). Generous on purpose.
    load_grace_s = max(900, int(timeout_s) if timeout_s and timeout_s > 0 else 0)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"=== cmd ===\n{_format_cmd_for_md(cmd, {})}\n")
        log_f.write(f"=== start (hang-watchdog) ===\n{start.isoformat()}\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # own process group: killpg reaps EngineCore too
        )

        sel = selectors.DefaultSelector()
        if proc.stdout is not None:
            sel.register(proc.stdout, selectors.EVENT_READ)

        last_output_t = time.time()
        last_status_mono = time.time()      # wall-clock when status last advanced
        last_status_update = None           # the status file's own last_update string
        while True:
            if proc.poll() is not None:
                break

            now = time.time()
            if timeout_s > 0 and (now - start.timestamp()) > timeout_s:
                _kill_process_group(proc)
                break

            # --- staleness watchdog (the reason this variant exists) ---
            phase = None
            cur_tag = ""
            st = None
            stale_for = now - last_status_mono
            try:
                if status_path.exists():
                    st = _read_json(status_path)
                    upd = st.get("last_update")
                    if upd != last_status_update:
                        last_status_update = upd
                        last_status_mono = now
                        stale_for = 0.0
                    phase = st.get("phase")
                    il = int(st.get("input_len") or 0)
                    ol = int(st.get("output_len") or 0)
                    bs = int(st.get("batch_size") or 0)
                    cur_tag = st.get("tag") or _meta_tag(bucket_meta, il, ol, bs)
            except Exception:
                phase = None  # unreadable status: fall through to load grace
            # Only phases where the child SHOULD be ticking get the tight limit;
            # everything else (model load, import, correctness) gets load grace.
            if phase in ("warmup", "benchmark", "bucket_start"):
                limit = _hang_stale_limit_s(
                    input_len=int((st or {}).get("input_len") or 0),
                    output_len=int((st or {}).get("output_len") or 0),
                    batch_size=int((st or {}).get("batch_size") or 0),
                    base_s=base_stale_s,
                )
            else:
                limit = load_grace_s
            if stale_for >= limit and limit > 0:
                hung = True
                wedged_tag = cur_tag
                msg = (
                    f"[hang-watchdog] status frozen {stale_for:.0f}s >= {limit}s "
                    f"(phase={phase}, bucket={wedged_tag or '?'}) — killing child process group\n"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
                log_f.write(msg)
                log_f.flush()
                _kill_process_group(proc)
                break

            events = sel.select(timeout=0.25)
            if events:
                for key, _mask in events:
                    stream = key.fileobj
                    line = stream.readline()
                    if not line:
                        continue
                    last_output_t = time.time()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()

            now = time.time()
            if heartbeat_s > 0 and (now - last_output_t) >= heartbeat_s:
                hb = (
                    f"[heartbeat] still running; elapsed={(now - start.timestamp()):.0f}s; "
                    f"status_stale={stale_for:.0f}s phase={phase}\n"
                )
                sys.stdout.write(hb)
                sys.stdout.flush()
                log_f.write(hb)
                log_f.flush()
                last_output_t = now

        try:
            sel.close()
        except Exception:
            pass

        try:
            rc = proc.wait(timeout=10)
        except Exception:
            rc = proc.returncode if proc.returncode is not None else -9
        end = datetime.now(timezone.utc)
        log_f.write(
            f"\n=== end (hang-watchdog) ===\n{end.isoformat()}\n"
            f"returncode={rc} hung={hung} wedged_tag={wedged_tag}\n"
        )
        log_f.flush()

    return {
        "ok": (rc == 0) and not hung,
        "returncode": rc,
        "hung": hung,
        "wedged_tag": wedged_tag,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_s": (end - start).total_seconds(),
    }


def _meta_tag(bucket_meta: Dict[str, Dict[str, int]], il: int, ol: int, bs: int) -> str:
    """Reverse-lookup a bucket tag from (il, ol, bs) using the parent's planned
    map, so the watchdog can name the wedged bucket even if the child status
    omits an explicit tag. Returns "" if no match."""
    for tag, m in (bucket_meta or {}).items():
        if int(m.get("input_len", -1)) == il and int(m.get("output_len", -1)) == ol \
                and int(m.get("batch_size", -1)) == bs:
            return tag
    return ""


def _run_cmd_streaming(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: Optional[Path],
    timeout_s: int,
    log_path: Path,
    heartbeat_s: int,
    status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a command and stream its output to a log (and stdout).

    This is intentionally "agent friendly": it keeps logs hot so a supervisor can
    tell whether we're progressing.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    # Create the log immediately so tailing works even if exec/import is slow.
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"=== cmd ===\n{_format_cmd_for_md(cmd, {})}\n")
        log_f.write(f"=== start ===\n{start.isoformat()}\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        sel = selectors.DefaultSelector()
        if proc.stdout is not None:
            sel.register(proc.stdout, selectors.EVENT_READ)

        last_output_t = time.time()
        while True:
            if proc.poll() is not None:
                break

            now = time.time()
            if timeout_s > 0 and (now - start.timestamp()) > timeout_s:
                proc.kill()
                break

            events = sel.select(timeout=0.25)
            if events:
                for key, _mask in events:
                    stream = key.fileobj
                    line = stream.readline()
                    if not line:
                        continue
                    last_output_t = time.time()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()

            now = time.time()
            if heartbeat_s > 0 and (now - last_output_t) >= heartbeat_s:
                hb = f"[heartbeat] still running; elapsed={(now - start.timestamp()):.0f}s\n"
                if status_path and status_path.exists():
                    try:
                        hb_status = _read_json(status_path)
                        hb = (
                            f"[heartbeat] still running; elapsed={(now - start.timestamp()):.0f}s; "
                            f"status={hb_status.get('phase')} bs={hb_status.get('batch_size')} "
                            f"last_update={hb_status.get('last_update')}\n"
                        )
                    except Exception:
                        pass
                sys.stdout.write(hb)
                sys.stdout.flush()
                log_f.write(hb)
                log_f.flush()
                last_output_t = now

        try:
            sel.close()
        except Exception:
            pass

        rc = proc.wait(timeout=10)
        end = datetime.now(timezone.utc)
        log_f.write(f"\n=== end ===\n{end.isoformat()}\nreturncode={rc}\n")
        log_f.flush()

    return {
        "ok": rc == 0,
        "returncode": rc,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_s": (end - start).total_seconds(),
    }


def _build_vllm_bench_cmd(
    *,
    vllm_exe: List[str],
    model_id: str,
    tp: int,
    max_model_len: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_iters: int,
    output_json: Path,
    extra_args: List[str],
) -> List[str]:
    # Command template based on validation/E2E_LATENCY_GUIDE.md
    cmd = (
        vllm_exe
        + [
            "bench",
            "latency",
            "--model",
            model_id,
            "--tensor-parallel-size",
            str(tp),
            "--max-model-len",
            str(max_model_len),
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--batch-size",
            str(batch_size),
            "--num-iters",
            str(num_iters),
            "--output-json",
            str(output_json),
        ]
    )
    if extra_args:
        cmd += extra_args
    return cmd


def _build_cli_equivalent_args_for_inproc(
    *,
    model_id: str,
    tp: int,
    max_model_len: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_iters: int,
    output_json: Path,
    extra_args: List[str],
) -> List[str]:
    # Mirror _build_vllm_bench_cmd, but without the leading `vllm bench latency`.
    base = [
        "--model",
        model_id,
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(max_model_len),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(batch_size),
        "--num-iters",
        str(num_iters),
        "--output-json",
        str(output_json),
    ]
    return base + (extra_args or [])


# ---- GSM8K helpers ----
# Adapted from vLLM's tests/evals/gsm8k/gsm8k_eval.py
# (https://github.com/vllm-project/vllm, Apache License 2.0), itself adapted
# from SGLang (Apache License 2.0). See data/ATTRIBUTION.md.

_GSM8K_SUBSET_PATH = Path(__file__).parent.parent / "data" / "gsm8k_subset.json"
_GSM8K_FULL_PATH = Path(__file__).parent.parent / "data" / "gsm8k_full.json"
_INVALID_ANSWER = -9999999


def _load_gsm8k_data(num_questions: int) -> Tuple[List[dict], List[dict]]:
    """Load bundled GSM8K data — fail-loud if the requested split is missing.

    Two-tier lookup:
      * ``num_questions <= 200``: prefer ``gsm8k_subset.json`` (200 test items).
      * Otherwise: require ``gsm8k_full.json`` (1319 test items). No GitHub
        download fallback — AMMO sessions run in a sandbox with no outbound
        network access, so a silent fallback is structurally broken.

    Raises:
        FileNotFoundError: when the bundled file needed for ``num_questions``
            does not exist. The message names the expected path so the fix is
            obvious (re-bundle the file).
    """
    if num_questions <= 200 and _GSM8K_SUBSET_PATH.exists():
        with open(_GSM8K_SUBSET_PATH) as f:
            data = json.load(f)
        return data["train"], data["test"][:num_questions]
    if _GSM8K_FULL_PATH.exists():
        with open(_GSM8K_FULL_PATH) as f:
            data = json.load(f)
        test = data["test"]
        if num_questions > len(test):
            raise FileNotFoundError(
                f"Requested {num_questions} GSM8K questions but bundled "
                f"{_GSM8K_FULL_PATH} only contains {len(test)} test items. "
                "Re-bundle the dataset or reduce --correctness-num-questions."
            )
        return data["train"], test[:num_questions]
    raise FileNotFoundError(
        f"No bundled GSM8K data covers {num_questions} questions. "
        f"Expected {_GSM8K_FULL_PATH} (for num_questions>200) or "
        f"{_GSM8K_SUBSET_PATH} (for num_questions<=200). "
        "AMMO sessions cannot download from GitHub — re-bundle the file."
    )


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return _INVALID_ANSWER
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return _INVALID_ANSWER


def _build_gsm8k_prompts(
    num_questions: int = 200, num_shots: int = 5
) -> Tuple[List[str], List[int]]:
    """Build few-shot GSM8K prompts and ground-truth labels."""
    if num_questions == 0:
        return [], []
    train_data, test_data = _load_gsm8k_data(num_questions)
    num_questions = min(num_questions, len(test_data))
    few_shot_examples = ""
    for i in range(min(num_shots, len(train_data))):
        few_shot_examples += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    prompts, labels = [], []
    for i in range(num_questions):
        prompts.append(few_shot_examples + f"Question: {test_data[i]['question']}\nAnswer:")
        labels.append(_get_answer_value(test_data[i]["answer"]))
    assert all(label != _INVALID_ANSWER for label in labels), "Some labels are invalid"
    return prompts, labels


def _serialize_correctness_outputs(outputs) -> List[Dict[str, Any]]:
    """Serialize vLLM RequestOutput objects to JSON-safe dicts."""
    serialized = []
    for i, req_output in enumerate(outputs):
        comp = req_output.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs_list = []
        if comp.logprobs is not None:
            for pos_logprobs in comp.logprobs:
                top_lps = {str(tid): lp.logprob for tid, lp in pos_logprobs.items()}
                logprobs_list.append({"top_logprobs": top_lps})
        serialized.append({
            "prompt_index": i,
            "token_ids": token_ids,
            "text": comp.text,
            "logprobs": logprobs_list,
            "num_tokens": len(token_ids),
        })
    return serialized


def _score_gsm8k_predictions(outputs, labels: List[int]) -> Tuple[List[int], float]:
    """Score GSM8K outputs. Returns (predictions_list, accuracy)."""
    preds = []
    for req_output in outputs:
        text = req_output.outputs[0].text
        preds.append(_get_answer_value(text))
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0
    return preds, accuracy


# ---- Real-prompt timing corpus (bundled GSM8K windowed into timing prompts) ----
#
# The timing loop historically fed uniform-random token IDs (np.random.randint),
# which is actively misleading for speculative decoding: draft/target models never
# agree on random tokens, so realized acceptance collapses to ~0 and ngram spec
# yields zero proposals. We instead window real GSM8K text (already bundled with
# the skill) into exactly-input_len token windows, deterministically seeded so the
# baseline and optimized arms see byte-identical prompts. See
# docs/.../2026-06-22-ammo-real-prompt-timing-design.md.

# Module-level cache, keyed by tokenizer NAME (not id(), which can be reused after
# the original tokenizer is GC'd). Repeated buckets and both labels reuse the cost.
_TIMING_CORPUS_CACHE: Dict[str, List[int]] = {}


def _load_timing_corpus_text() -> str:
    """Concatenate ALL bundled GSM8K question+answer text into one corpus string.

    Forces the FULL file (1319 test + 5 train items, ~177K tokens) — never the
    200-item subset, whose ~27K tokens tiled into large windows would inflate
    spec-decode (especially ngram) acceptance, the exact metric this change exists
    to measure faithfully. Fail-loud if the full file is missing.

    Items are joined by a blank line so window edges that straddle a Q&A boundary
    read as a document break rather than a garbled splice.
    """
    if not _GSM8K_FULL_PATH.exists():
        raise FileNotFoundError(
            f"Bundled GSM8K full dataset required for real-prompt timing not found "
            f"at {_GSM8K_FULL_PATH}. ALL timing runs now require the bundled GSM8K "
            "data co-located with the script — re-bundle the file, or pass "
            "--dummy-prompt-source random to use legacy uniform-random tokens. "
            "AMMO sessions cannot download from GitHub."
        )
    with open(_GSM8K_FULL_PATH) as f:
        data = json.load(f)
    items: List[dict] = list(data.get("train", [])) + list(data.get("test", []))
    parts = [
        f"Question: {item['question']}\nAnswer: {item['answer']}"
        for item in items
    ]
    return "\n\n".join(parts)


def _load_timing_corpus_tokens(tokenizer) -> List[int]:
    """Tokenize the bundled GSM8K corpus once per tokenizer NAME (cached).

    Returns a flat list of real, in-vocab token IDs (encoded without special
    tokens). Asserts the corpus tokenizes to a non-empty list.

    Raises a clear, actionable error when the tokenizer is unavailable (e.g.
    ``--skip-tokenizer-init`` makes ``llm.get_tokenizer()`` return None / a
    tokenizer without ``encode``): real prompts require a tokenizer; the caller
    should fall back to ``--dummy-prompt-source random`` in that case.
    """
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise RuntimeError(
            "Real-prompt timing (--dummy-prompt-source gsm8k) requires a usable "
            "tokenizer, but the engine did not provide one (e.g. built with "
            "--skip-tokenizer-init). Pass --dummy-prompt-source random to use "
            "legacy uniform-random tokens instead."
        )
    key = getattr(tokenizer, "name_or_path", None)
    if not isinstance(key, str) or not key:
        # Stable fallback key when name_or_path is unavailable. type() name keeps
        # distinct tokenizer classes from colliding; it is intentionally NOT id().
        key = f"<{type(tokenizer).__module__}.{type(tokenizer).__name__}>"
    cached = _TIMING_CORPUS_CACHE.get(key)
    if cached is not None:
        return cached
    text = _load_timing_corpus_text()
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        # Some tokenizers (and our test stubs) don't accept add_special_tokens.
        ids = tokenizer.encode(text)
    ids = list(ids)
    assert len(ids) > 0, "GSM8K timing corpus tokenized to an empty list"
    _TIMING_CORPUS_CACHE[key] = ids
    return ids


def _timing_prompt_seed(input_len: int, output_len: int, batch_size: int) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed for the timing windows.

    Uses ``zlib.crc32`` over a canonical string. Python's builtin ``hash()`` is
    salted per process, so it is NOT stable across the separate torchrun rank
    processes; crc32 is, which makes every DP rank derive the same windows from
    the same corpus before ``_partition_prompts`` slices its stride.
    """
    import zlib
    return zlib.crc32(f"{input_len}:{output_len}:{batch_size}".encode("utf-8"))


def _build_timing_prompts(
    corpus_tokens,
    *,
    batch_size: int,
    input_len: int,
    output_len: int,
    source: str = "gsm8k",
    np_module=None,
) -> List[Dict[str, Any]]:
    """Build a list of ``{"prompt_token_ids": [...]}`` of length ``batch_size``.

    Each window is exactly ``input_len`` token IDs.

    ``source="gsm8k"`` (default): deterministic crc32-seeded windows sliced from
    ``corpus_tokens`` with EXACT modulo wrap, so a window is correct for every
    input_len/corpus-length ratio (including input_len > corpus length). Identical
    for the baseline and optimized arms and across launches/rounds for the same
    workload — removing prompt variance from the A/B entirely.

    ``source="random"`` (legacy): byte-for-byte the old
    ``np.random.randint(10000, size=(batch_size, input_len))`` behavior, retained
    only for tests and manual debugging. Unseeded (DP-only-safe).

    Pure function: numpy is injected via ``np_module`` (the child passes its ``np``;
    tests inject their own) and the corpus is passed in, so unit tests run without
    importing vLLM or touching a GPU.
    """
    np = np_module
    if np is None:
        import numpy as np  # type: ignore[no-redef]
    if source == "random":
        token_ids = np.random.randint(10000, size=(batch_size, input_len))
        return [{"prompt_token_ids": batch} for batch in token_ids.tolist()]
    if source != "gsm8k":
        raise ValueError(
            f"Unknown dummy_prompt_source={source!r}; expected 'gsm8k' or 'random'."
        )
    n = len(corpus_tokens)
    assert n > 0, "GSM8K timing corpus is empty; cannot build windows"
    rng = np.random.default_rng(
        _timing_prompt_seed(input_len, output_len, batch_size)
    )
    prompts: List[Dict[str, Any]] = []
    for _ in range(batch_size):
        start = int(rng.integers(0, n))
        window = [int(corpus_tokens[(start + k) % n]) for k in range(input_len)]
        prompts.append({"prompt_token_ids": window})
    return prompts


def _snapshot_spec_decode_counters(llm) -> Optional[Dict[str, int]]:
    """Sum the spec-decode Prometheus counters from ``llm.get_metrics()``.

    Returns ``{"num_drafts", "num_draft_tokens", "num_accepted_tokens"}`` summed
    across all label sets (e.g. multiple engines), or ``None`` when
    ``get_metrics()`` is unavailable/raises OR no spec-decode counters are present
    (vanilla decode, the draft path never armed, or an older vLLM). Fully
    defensive: any failure returns ``None`` rather than raising, so the timing run
    is never broken by an acceptance-counter read.
    """
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None
    names = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    }
    totals: Dict[str, int] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    found = False
    try:
        for metric in metrics or []:
            name = getattr(metric, "name", None)
            key = names.get(name)
            if key is None:
                continue
            value = getattr(metric, "value", None)
            if not isinstance(value, (int, float)):
                continue
            totals[key] += int(value)
            found = True
    except Exception:
        return None
    if not found:
        return None
    return totals


def _spec_decode_block_from_diff(
    before: Optional[dict], after: Optional[dict]
) -> Optional[dict]:
    """Diff two spec-decode counter snapshots into the additive ``spec_decode`` block.

    The Prometheus counters are cumulative + monotonic and accumulate across
    buckets in the one child process, so a per-bucket ``after - before`` diff (not a
    final absolute read) is required for correct per-bucket attribution.

    Returns ``None`` (emit nothing) when both snapshots are ``None`` OR there was no
    real draft activity (``num_drafts == 0`` AND ``num_draft_tokens == 0``). The two
    ratios are guarded INDEPENDENTLY:
      * ``draft_acceptance_rate = accepted / draft_tokens``  only if draft_tokens > 0
      * ``mean_acceptance_length = 1 + accepted / drafts``    only if drafts > 0
    The three raw counts are always included when the block is emitted.
    """
    if before is None or after is None:
        return None
    drafts = int(after.get("num_drafts", 0)) - int(before.get("num_drafts", 0))
    draft_tokens = int(after.get("num_draft_tokens", 0)) - int(
        before.get("num_draft_tokens", 0)
    )
    accepted = int(after.get("num_accepted_tokens", 0)) - int(
        before.get("num_accepted_tokens", 0)
    )
    if drafts == 0 and draft_tokens == 0:
        return None
    block: Dict[str, Any] = {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
    }
    if draft_tokens > 0:
        block["draft_acceptance_rate"] = accepted / draft_tokens
    if drafts > 0:
        block["mean_acceptance_length"] = 1.0 + accepted / drafts
    return block


def _build_correctness_child_flags(
    *,
    capture: bool,
    verify: bool,
    num_questions: int,
    tolerance_pct: float,
) -> List[str]:
    """Build the hidden child-only correctness flags for parent→child dispatch.

    Returns an empty list when neither capture nor verify is requested. When
    at least one is set, includes both ``--_correctness-num-questions`` and
    ``--_correctness-tolerance-pct`` flag/value pairs with their stringified
    arguments adjacent (preserves the order _build_child_cmd expects). This
    is a pure function so tests can exercise it without touching main().
    """
    if not (capture or verify):
        return []
    return [
        "--_correctness-num-questions", str(num_questions),
        "--_correctness-tolerance-pct", str(tolerance_pct),
    ]


def _format_correctness_message(
    verdict: str,
    *,
    opt_acc: float,
    opt_correct_count: int,
    n: int,
    threshold: float,
    baseline_acc: float,
    tolerance_pct: float,
) -> str:
    """Build the human-readable PASS/FAIL verdict message for Gate 5.1b.

    Matches the format parsed by eval/scripts/parse_artifacts.py Pattern 0:
      "{PASS|FAIL}: opt_accuracy X% (A/B) {>=|<} threshold Y% "
      "(baseline Z% - Tpp tolerance)"
    """
    op = ">=" if verdict == "PASS" else "<"
    return (
        f"{verdict}: opt_accuracy {opt_acc * 100:.1f}% "
        f"({opt_correct_count}/{n}) {op} threshold {threshold * 100:.1f}% "
        f"(baseline {baseline_acc * 100:.1f}% - "
        f"{tolerance_pct:.1f}pp tolerance)"
    )


def _compare_correctness(
    *,
    golden_refs: List[Dict[str, Any]],
    opt_outputs: List[Dict[str, Any]],
    labels: Optional[List[int]],
    baseline_preds: Optional[List[int]],
    opt_preds: Optional[List[int]],
    tolerance_pct: float = 1.0,
) -> Dict[str, Any]:
    """Compare optimized outputs against golden references (Gate 5.1b v2).

    Verdict: ``opt_accuracy >= baseline_accuracy - tolerance_pct/100``.
    ``tolerance_pct=0.0`` recovers the legacy strict behavior; the default
    ``1.0`` pp allows ~1% drift (e.g. ~13 questions at N=1319) before FAIL.
    Token-level data is computed as diagnostics only (never affects verdict).
    """
    tolerance_pct = float(tolerance_pct)
    num_questions = min(len(golden_refs), len(opt_outputs))

    # ---- Edge case: empty question set ----
    if num_questions == 0:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": 0,
            "baseline_accuracy": 0.0, "optimized_accuracy": 0.0,
            "accuracy_delta": 0.0,
            "baseline_correct_count": 0, "optimized_correct_count": 0,
            "questions_lost": [], "questions_gained": [],
            "tolerance_pct": tolerance_pct,
            "threshold": 0.0,
            "message": "FAIL: no questions to compare (num_questions=0)",
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
        }

    # ---- Edge case: accuracy gate requires labels + preds ----
    if labels is None or baseline_preds is None or opt_preds is None:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": num_questions,
            "baseline_accuracy": 0.0, "optimized_accuracy": 0.0,
            "accuracy_delta": 0.0,
            "baseline_correct_count": 0, "optimized_correct_count": 0,
            "questions_lost": [], "questions_gained": [],
            "tolerance_pct": tolerance_pct,
            "threshold": 0.0,
            "message": "FAIL: accuracy gate requires labels, baseline_preds, and opt_preds",
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
            "_error": "Accuracy gate requires labels, baseline_preds, and opt_preds.",
        }

    # ---- Accuracy computation ----
    n = min(len(labels), len(baseline_preds), len(opt_preds), num_questions)
    labels_t = labels[:n]
    baseline_preds_t = baseline_preds[:n]
    opt_preds_t = opt_preds[:n]

    baseline_correct = {i for i, (p, l) in enumerate(zip(baseline_preds_t, labels_t)) if p == l}
    opt_correct = {i for i, (p, l) in enumerate(zip(opt_preds_t, labels_t)) if p == l}
    questions_lost = sorted(baseline_correct - opt_correct)
    questions_gained = sorted(opt_correct - baseline_correct)

    baseline_acc = len(baseline_correct) / n
    opt_acc = len(opt_correct) / n

    # ---- Baseline accuracy floor ----
    if len(baseline_correct) == 0:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": n,
            "baseline_accuracy": 0.0, "optimized_accuracy": round(opt_acc, 4),
            "accuracy_delta": round(opt_acc, 4),
            "baseline_correct_count": 0,
            "optimized_correct_count": len(opt_correct),
            "questions_lost": [], "questions_gained": questions_gained,
            "tolerance_pct": tolerance_pct,
            "threshold": 0.0,
            "message": "FAIL: baseline accuracy is 0% (infrastructure error)",
            "infrastructure_error": True,
            "infrastructure_message": "Baseline accuracy is 0% — model cannot solve any GSM8K questions; environment suspect.",
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
        }

    # ---- Verdict: opt_accuracy >= baseline_accuracy - tolerance_pct/100 ----
    threshold = baseline_acc - (tolerance_pct / 100.0)
    verdict = "PASS" if opt_acc >= threshold else "FAIL"

    # ---- Token-level diagnostics (informational only) ----
    first_divergence_positions: List[int] = []
    divergent_questions = 0
    all_empty = True

    for q in range(n):
        b_ids = golden_refs[q]["token_ids"]
        o_ids = opt_outputs[q]["token_ids"]
        if b_ids or o_ids:
            all_empty = False
        min_len = min(len(b_ids), len(o_ids))
        first_div = next((p for p in range(min_len) if b_ids[p] != o_ids[p]), -1)
        if first_div == -1 and len(b_ids) != len(o_ids):
            first_div = min_len
        if first_div >= 0:
            divergent_questions += 1
            first_divergence_positions.append(first_div)

    # All outputs empty → override to FAIL regardless of accuracy
    if all_empty and n > 0:
        verdict = "FAIL"

    # p50/p95 of first_divergence_positions (across divergent questions only)
    if first_divergence_positions:
        sorted_pos = sorted(first_divergence_positions)
        p50_idx = max(0, int(len(sorted_pos) * 0.50) - 1)
        p95_idx = max(0, int(len(sorted_pos) * 0.95) - 1)
        fdp_p50 = sorted_pos[p50_idx]
        fdp_p95 = sorted_pos[p95_idx]
    else:
        fdp_p50 = -1
        fdp_p95 = -1

    churn_rate = round((len(questions_lost) + len(questions_gained)) / n, 4) if n else 0.0

    message = _format_correctness_message(
        verdict,
        opt_acc=opt_acc,
        opt_correct_count=len(opt_correct),
        n=n,
        threshold=threshold,
        baseline_acc=baseline_acc,
        tolerance_pct=tolerance_pct,
    )

    return {
        "gate": "5.1b",
        "verdict": verdict,
        "num_questions": n,
        "baseline_accuracy": round(baseline_acc, 4),
        "optimized_accuracy": round(opt_acc, 4),
        "accuracy_delta": round(opt_acc - baseline_acc, 4),
        "baseline_correct_count": len(baseline_correct),
        "optimized_correct_count": len(opt_correct),
        "questions_lost": questions_lost,
        "questions_gained": questions_gained,
        "tolerance_pct": tolerance_pct,
        "threshold": round(threshold, 4),
        "message": message,
        "diagnostics": {
            "divergent_questions": divergent_questions,
            "first_divergence_positions_p50": fdp_p50,
            "first_divergence_positions_p95": fdp_p95,
            "churn_rate": churn_rate,
            "note": "Token-level data is informational only and does not affect the verdict.",
        },
        "_diagnostic_notes": "p50/p95 computed across questions where first_divergence_pos >= 0; set to -1 if no questions diverge. churn_rate = (questions_lost + questions_gained) / num_questions.",
    }


def _run_inproc_latency_sweep_child(
    *,
    label: str,
    model_id: str,
    tp: int,
    max_model_len: int,
    buckets: List[Dict[str, int]],
    num_iters: int,
    extra_args: List[str],
    out_root: Path,
    timeout_s_per_bucket: int,
    nsys_profile: bool = False,
    torch_profile: bool = False,
    capture_golden_refs: bool = False,
    verify_correctness: bool = False,
    correctness_num_questions: int = 1319,
    correctness_tolerance_pct: float = 1.0,
    skip_tags: "frozenset[str] | set[str] | None" = None,
    dummy_prompt_source: str = "gsm8k",
) -> int:
    """Child-mode runner: load model once, benchmark all buckets.

    Each bucket is a dict with ``input_len``, ``output_len``, ``batch_size``.
    SamplingParams are reconstructed per-bucket (output_len may vary).

    Writes per-bucket artifacts using ``_bucket_file_tag`` for naming:
      logs/{label}_{tag}.log
      json/{label}_{tag}.json          (raw vllm bench latency format)
      json/{label}_{tag}.runner.json   (runner status + timing + errors)
    """
    logs_dir = out_root / "logs"
    json_dir = out_root / "json"
    status_dir = out_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    status_path = status_dir / f"{label}.json"
    child_log_path = logs_dir / f"{label}_child.log"

    def _update_status(phase: str, *, batch_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        # Rank-0-gated: non-rank-0 SPMD ranks must not race on status JSON writes.
        if not _IS_RANK0:
            return
        payload: Dict[str, Any] = {
            "label": label,
            "phase": phase,
            "batch_size": batch_size,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        _write_json_atomic(status_path, payload)

    # Emit status/logs *before* heavy imports so supervisors can tell we're alive.
    _update_status("starting_import", extra={"model_id": model_id})
    with _rank0_open_log(child_log_path, "a") as child_log:
        child_log.write(f"=== child start ({label}) ===\n{datetime.now(timezone.utc).isoformat()}\n")
        child_log.write(f"model_id={model_id} tp={tp} max_model_len={max_model_len}\n")
        child_log.flush()

    # Lazy imports: avoid importing vLLM unless the child is actually executing.
    import dataclasses as _dataclasses
    import time as _time

    try:
        import numpy as np  # type: ignore

        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.benchmarks import latency as vllm_latency  # type: ignore
        from vllm.engine.arg_utils import EngineArgs  # type: ignore
        from vllm.inputs import PromptType  # type: ignore
        from vllm.sampling_params import BeamSearchParams  # type: ignore
        from vllm.utils.argparse_utils import FlexibleArgumentParser  # type: ignore
    except Exception as e:
        # Import errors are common when not running under the vLLM venv; make it explicit.
        err = f"Failed to import vLLM benchmark deps in child runner: {e}"
        _update_status("import_failed", extra={"error": err})
        _rank0_write_text(out_root / f"child_{label}_import_error.log", err + "\n" + traceback.format_exc())
        with _rank0_open_log(child_log_path, "a") as child_log:
            child_log.write(err + "\n")
            child_log.write(traceback.format_exc() + "\n")
        return 2

    # Parse CLI-equivalent args once (using the first bucket), then override per-bucket.
    # This ensures we honor bench.extra_args using vLLM's own argparse schema.
    # Use vLLM's FlexibleArgumentParser so we support dotted "json-style" flags
    # like `-cc.pass_config.enable_sp=false` (CompilationConfig), etc.
    seed = buckets[0] if buckets else {"input_len": 64, "output_len": 512, "batch_size": 1}
    parser = FlexibleArgumentParser(add_help=False, add_json_tip=False)
    vllm_latency.add_cli_args(parser)
    seed_tag = _bucket_file_tag(seed, buckets)
    seed_out = json_dir / f"{label}_{seed_tag}.json"
    seed_argv = _build_cli_equivalent_args_for_inproc(
        model_id=model_id,
        tp=tp,
        max_model_len=max_model_len,
        input_len=seed["input_len"],
        output_len=seed["output_len"],
        batch_size=seed["batch_size"],
        num_iters=num_iters,
        output_json=seed_out,
        extra_args=extra_args,
    )
    args = parser.parse_args(seed_argv)

    if getattr(args, "profile", False) and not nsys_profile and not torch_profile:
        # vllm bench latency --profile is a single-run action; sweeping doesn't make sense.
        # (When nsys_profile or torch_profile is active, we handle profiler start/stop per-bucket ourselves.)
        _rank0_write_text(out_root / f"child_{label}_error.log", "Refusing to sweep with --profile enabled.\n")
        _update_status("error", extra={"error": "Refusing to sweep with --profile enabled."})
        return 2

    engine_args = EngineArgs.from_cli_args(args)
    _update_status("loading_model")
    # Work around pydantic validation: _dataclasses.asdict may produce None
    # values inside nested config dicts (e.g. compilation_config.cudagraph_capture_sizes
    # or compilation_config.pass_config.fuse_minimax_qk_norm)                                                                                                                                                                                                                                                                                                           
    # which CompilationConfig/PassConfig reject. Filter them out recursively.           
    def _strip_none_recursive(d):                                                                                                                                                                                                                                                                                                                                       
        """Remove None values from a dict, recursing into nested dicts."""              
        cleaned = {}                                                                    
        for k, v in d.items():                                                          
            if v is None:                                                               
                continue                                                                
            if isinstance(v, dict):                                                     
                v = _strip_none_recursive(v)                                            
                if not v:  # skip empty dicts too                                       
                    continue                                                            
            cleaned[k] = v                                                              
        return cleaned                                                                  
                                                                                        
    ea_dict = _dataclasses.asdict(engine_args)                                          
    for _cfg_key in ("compilation_config", "profiler_config", "attention_config",       
                    "structured_outputs_config"):                                     
        if isinstance(ea_dict.get(_cfg_key), dict):                                     
            ea_dict[_cfg_key] = _strip_none_recursive(ea_dict[_cfg_key])
    # Pristine snapshot taken after None-filtering and before the overrides
    # below. The from_engine_args path replays exactly the changed keys onto
    # engine_args, so both construction paths build the same engine.
    _ea_pristine = dict(ea_dict)
    # Configure torch profiler on the engine when --torch-profile is active.
    if torch_profile:
        # v2 layout: traces go to rounds/{N}/profiling/torch_profile/, sibling
        # to the sweep output. Legacy: out_root / torch_profile/.
        torch_profile_base = str(_v2_profiling_dir(out_root, "torch_profile"))
        Path(torch_profile_base).mkdir(parents=True, exist_ok=True)
        ea_dict["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": torch_profile_base,
        }
    selected_step_buckets = [b for b in buckets if _is_nsys_selected_step_bucket(b)]
    if selected_step_buckets:
        # _expand_nsys_profile_buckets floors ONE child-wide capture window across all
        # selected-step buckets, so this is a cheap invariant rather than a constraint:
        # a single delay_iterations serves every capture point (the floor clears prefill
        # for the deepest-prefill bucket, and a larger-than-needed window only helps).
        windows = {int(b["nsys_capture_window_output_len"]) for b in selected_step_buckets}
        if len(windows) != 1:
            raise SystemExit(
                "Selected-step nsys capture requires one capture window per child; "
                f"got windows={sorted(windows)} (expected a single child-wide window from "
                "_selected_step_effective_window)"
            )
        _apply_selected_step_profiler_config(ea_dict, selected_step_buckets[0])
    # Per spec §3.1 step 1, force `disable_log_stats=False` so vLLM v1 wires
    # `RequestOutput.metrics` (prefill/decode timestamps) into outputs. The
    # default in `LLM(...)` is True, which would silently hide the per-request
    # timing we need for f_e2e computation.
    ea_dict["disable_log_stats"] = False
    if hasattr(LLM, "from_engine_args"):
        # Preferred path: replay only the keys this function overrode onto
        # engine_args, then let vLLM build the engine from EngineArgs. Nested
        # config fields stay as their own config objects, so the None-filtering
        # that `LLM(**ea_dict)` needs never applies here.
        for _k, _v in ea_dict.items():
            if _k not in _ea_pristine or _ea_pristine[_k] != _v:
                setattr(engine_args, _k, _v)
        llm = LLM.from_engine_args(engine_args)
    else:
        # Fallback for vLLM builds without the classmethod. `LLM(**engine_args)`
        # kwargs expansion is deprecated; keep it only while from_engine_args
        # is absent.
        llm = LLM(**ea_dict)
    _update_status("model_loaded")

    # Real-prompt timing corpus: load + tokenize the bundled GSM8K once (cached
    # by tokenizer name), so every bucket windows real tokens deterministically.
    # source="random" keeps _timing_corpus=None (legacy uniform-random path).
    _timing_corpus: Optional[List[int]] = None
    if dummy_prompt_source == "gsm8k":
        try:
            _tok = llm.get_tokenizer()
        except Exception as _tok_exc:
            raise RuntimeError(
                "Real-prompt timing (--dummy-prompt-source gsm8k, the default) "
                "requires a tokenizer, but llm.get_tokenizer() failed "
                f"({_tok_exc!r}). This usually means the engine was built with "
                "--skip-tokenizer-init. Pass --dummy-prompt-source random to use "
                "legacy uniform-random tokens instead."
            ) from _tok_exc
        _timing_corpus = _load_timing_corpus_tokens(_tok)

    # SPMD parallelism context (spec §4.3). Under DP=1 this is a no-op path:
    # dp_size=1, dp_rank=0, error_flag is None and no dist.barrier() calls.
    dp_size = int(getattr(args, "data_parallel_size", 1))
    tp_size = int(getattr(args, "tensor_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_parallel_size", 1))
    dp_rank = _GLOBAL_RANK // max(1, tp_size * pp_size)
    dist = None  # type: ignore[assignment]
    error_flag = None
    if dp_size > 1:
        import torch as _torch
        import torch.distributed as dist  # type: ignore[no-redef]
        error_flag = _torch.tensor([0], dtype=_torch.int32, device="cuda")
        # Rank-layout drift check: if vLLM reorders ranks (tp x pp x dp vs
        # dp x pp x tp) our partitioning assumptions break. Fail loud now.
        try:
            vllm_dp_rank = int(llm.vllm_config.parallel_config.data_parallel_rank)
        except Exception:
            vllm_dp_rank = dp_rank  # vLLM version without this attribute
        assert vllm_dp_rank == dp_rank, (
            f"rank-layout convention drift: computed dp_rank={dp_rank} "
            f"!= vllm.parallel_config.data_parallel_rank={vllm_dp_rank}"
        )

    # ---- Phase 1: Correctness (GSM8K greedy decode) ----
    if capture_golden_refs or verify_correctness:
        import torch
        from vllm import SamplingParams as _CorrectnessSP
        _update_status("correctness_phase_start")
        json_dir = out_root / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        # DP precondition: every rank needs at least one real prompt.
        _check_correctness_dp_precondition(correctness_num_questions, dp_size)

        # verdict_code is broadcast at the end of this block so all ranks
        # return together: 0=PASS, 3=FAIL, 4=infra error.
        verdict_code_tensor: Optional[Any] = None
        if dp_size > 1:
            verdict_code_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")

        try:
            prompts, gsm8k_labels = _build_gsm8k_prompts(num_questions=correctness_num_questions)
            correctness_sp = _CorrectnessSP(
                temperature=0.0, max_tokens=1024,
                stop=["Question", "Assistant:", "<|separator|>"],
                seed=42, logprobs=5,
            )

            # Partition prompts across DP ranks — each rank generates only
            # its own share. The final gather + stitch reassembles canonical
            # order on all ranks (but only rank 0 scores/writes).
            if dp_size > 1:
                my_prompts = _partition_prompts(
                    prompts, dp_size=dp_size, dp_rank=dp_rank, input_len=1,
                )
                _rank0_log(
                    f"[correctness] Running GSM8K greedy decode: "
                    f"{len(prompts)} questions partitioned across dp={dp_size}"
                )
            else:
                my_prompts = prompts
                print(f"[correctness] Running GSM8K greedy decode: {len(prompts)} questions")

            t0 = time.time()
            my_outputs = llm.generate(my_prompts, sampling_params=correctness_sp, use_tqdm=False)
            duration = time.time() - t0
            _rank0_log(f"[correctness] Generation done in {duration:.1f}s")

            # Gather per-rank outputs → canonical order on every rank.
            if dp_size > 1:
                import torch.distributed as _dist_cp
                from vllm.distributed.parallel_state import get_world_group  # type: ignore
                cpu_group = get_world_group().cpu_group
                gathered: List[Any] = [None] * dp_size
                _dist_cp.all_gather_object(gathered, my_outputs, group=cpu_group)
                outputs = _stitch_gathered(gathered, len(prompts), dp_size)
            else:
                outputs = my_outputs

            serialized = _serialize_correctness_outputs(outputs)
            preds, accuracy = _score_gsm8k_predictions(outputs, gsm8k_labels)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

            # Rank 0 writes golden/opt artifacts + runs the comparator.
            if _IS_RANK0 and capture_golden_refs:
                # Self-consistency check is only meaningful under DP=1.
                # Under DP>1, running llm.generate(prompts) on rank 0 alone
                # deadlocks — other ranks have already moved on. Skip it
                # and record deterministic=True (we already partitioned and
                # gathered deterministically on the first pass).
                if dp_size == 1:
                    print("[correctness] Running self-consistency check...")
                    outputs2 = llm.generate(prompts, sampling_params=correctness_sp, use_tqdm=False)
                    serialized2 = _serialize_correctness_outputs(outputs2)
                    deterministic = all(
                        s1["token_ids"] == s2["token_ids"]
                        for s1, s2 in zip(serialized, serialized2)
                    )
                    if not deterministic:
                        print("[correctness] WARNING: Self-consistency check FAILED — environment is non-deterministic.")
                    else:
                        print("[correctness] Self-consistency check PASSED — greedy decode is deterministic.")
                else:
                    print("[correctness] Self-consistency check skipped under DP>1 (requires all-ranks re-run).")
                    deterministic = True

                golden_data = {
                    "metadata": {
                        "num_questions": len(prompts), "num_shots": 5, "max_tokens": 1024,
                        "seed": 42, "logprobs_k": 5, "gsm8k_accuracy": round(accuracy, 4),
                        "capture_duration_s": round(duration, 2), "gpu_name": gpu_name,
                        "deterministic": deterministic,
                        "baseline_preds": preds, "labels": gsm8k_labels,
                        "dp_size": dp_size,
                        # Informational only: tolerance is NOT enforced in the
                        # metadata-mismatch check on resume — agents may tune
                        # it per-campaign without re-capturing golden refs.
                        "tolerance_pct": float(correctness_tolerance_pct),
                    },
                    "outputs": serialized,
                }
                golden_path = json_dir / "golden_refs.json"
                _rank0_write_json(golden_path, golden_data)
                print(f"[correctness] Golden refs saved to {golden_path}")
                _update_status("correctness_done", extra={"accuracy": accuracy, "deterministic": deterministic})

            if verify_correctness:
                # Verdict code starts at 0 (PASS). Non-rank-0 keeps 0 and
                # receives the authoritative code from the MAX-reduce.
                local_code = 0
                if _IS_RANK0:
                    golden_path = json_dir / "golden_refs.json"
                    if not golden_path.exists():
                        print(f"[correctness] ERROR: golden_refs.json not found at {golden_path}")
                        _update_status("correctness_error", extra={"error": "golden_refs.json not found"})
                        local_code = 4
                    else:
                        golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
                        golden_refs = golden_data["outputs"]
                        golden_meta = golden_data.get("metadata", {})
                        baseline_preds = golden_meta.get("baseline_preds")
                        golden_labels = golden_meta.get("labels")

                        # Metadata mismatch detection (exit code 4)
                        golden_nq = golden_meta.get("num_questions")
                        if golden_nq is not None and golden_nq != correctness_num_questions:
                            msg = (f"[correctness] ERROR: golden refs num_questions={golden_nq} "
                                   f"!= current num_questions={correctness_num_questions}. "
                                   "Re-capture golden refs with matching --correctness-num-questions.")
                            print(msg)
                            _update_status("correctness_error", extra={"error": msg})
                            local_code = 4
                        else:
                            golden_mt = golden_meta.get("max_tokens")
                            if golden_mt is not None and golden_mt != 1024:
                                msg = (f"[correctness] ERROR: golden refs max_tokens={golden_mt} "
                                       f"!= current max_tokens=1024. "
                                       "Re-capture golden refs with v2 settings.")
                                print(msg)
                                _update_status("correctness_error", extra={"error": msg})
                                local_code = 4
                            else:
                                # GPU name mismatch warning
                                golden_gpu = golden_meta.get("gpu_name", "")
                                if golden_gpu and golden_gpu != gpu_name:
                                    print(f"[correctness] WARNING: GPU mismatch — golden refs captured on '{golden_gpu}', current GPU is '{gpu_name}'")

                                # Save opt outputs
                                opt_path = json_dir / "opt_outputs.json"
                                _rank0_write_json(opt_path, {"outputs": serialized})

                                # Run comparator
                                verdict = _compare_correctness(
                                    golden_refs=golden_refs, opt_outputs=serialized,
                                    labels=golden_labels, baseline_preds=baseline_preds, opt_preds=preds,
                                    tolerance_pct=correctness_tolerance_pct,
                                )
                                verdict["duration_s"] = round(duration, 2)

                                verdict_path = json_dir / "correctness_verdict.json"
                                _rank0_write_json(verdict_path, verdict)
                                print(f"[correctness] Verdict: {verdict['verdict']}")
                                print(f"[correctness] Written to {verdict_path}")

                                local_code = _verdict_to_code(verdict)
                                if verdict.get("infrastructure_error"):
                                    print(f"[correctness] {verdict.get('infrastructure_message', 'Infrastructure error')}")
                                    _update_status("correctness_error", extra=verdict)
                                elif local_code != 0:
                                    _update_status("correctness_failed", extra=verdict)
                                else:
                                    _update_status("correctness_done", extra=verdict)

                # Broadcast the verdict code: all ranks return the same exit
                # code so torchrun reports a unified result.
                if dp_size > 1:
                    assert verdict_code_tensor is not None
                    verdict_code_tensor[0] = local_code
                    import torch.distributed as _dist_vc
                    _dist_vc.all_reduce(verdict_code_tensor, op=_dist_vc.ReduceOp.MAX)
                    final_code = int(verdict_code_tensor.item())
                else:
                    final_code = local_code

                if final_code != 0:
                    return final_code

        except Exception as e:
            _rank0_log(f"[correctness] ERROR: {type(e).__name__}: {e}")
            if _IS_RANK0:
                traceback.print_exc()
            _update_status("correctness_error", extra={"error": str(e)})
            # Broadcast exception outcome so peer ranks also abort.
            if dp_size > 1:
                assert verdict_code_tensor is not None
                verdict_code_tensor[0] = 4
                import torch.distributed as _dist_exc
                _dist_exc.all_reduce(verdict_code_tensor, op=_dist_exc.ReduceOp.MAX)
            return 4  # Infrastructure error
    # ---- End Phase 1 ----

    _skip_tags = set(skip_tags or ())
    for bucket in buckets:
        b_input_len = bucket["input_len"]
        b_output_len = bucket["output_len"]
        bs = bucket["batch_size"]
        tag = _bucket_file_tag(bucket, buckets)

        # Hang-watchdog resume: skip buckets the parent has already collected
        # from a prior launch, plus any bucket that wedged the engine on a prior
        # launch. The parent passes these via --_skip-tags so a relaunch does not
        # redo completed work or re-trigger the same hang. A bucket that already
        # has a valid raw JSON on disk needs no marker; for a wedged bucket with
        # no JSON we drop a `skipped` runner.json so the parent's collection step
        # stays deterministic (the row is reported as not-run, never fabricated).
        if tag in _skip_tags:
            if _IS_RANK0:
                print(f"SKIP bucket {tag}: in --_skip-tags (done or wedged on a prior launch)", flush=True)
                runner_json = json_dir / f"{label}_{tag}.runner.json"
                raw_json = json_dir / f"{label}_{tag}.json"
                if not raw_json.exists():
                    _rank0_write_json(runner_json, {
                        "ok": False,
                        "skipped": True,
                        "reason": "hang-watchdog: bucket skipped on relaunch (completed earlier or wedged the engine)",
                        "label": label,
                        "batch_size": bs,
                        "input_len": b_input_len,
                        "output_len": b_output_len,
                    })
            if dist is not None:
                dist.barrier()
            continue

        # Skip buckets whose batch_size cannot be partitioned across all DP
        # ranks (spec §4.3). Rank 0 writes a skip marker; all ranks call
        # dist.barrier() to stay in lockstep before continuing.
        if _should_skip_bucket(bs, dp_size):
            skip_raw_json = json_dir / f"{label}_{tag}.json"
            skip_runner_json = json_dir / f"{label}_{tag}.runner.json"
            if _IS_RANK0:
                print(
                    f"SKIP bucket bs={bs}: batch_size < dp_size={dp_size} "
                    f"(il={b_input_len}, ol={b_output_len})",
                    flush=True,
                )
                _rank0_write_json(skip_raw_json, {
                    "skipped": True,
                    "reason": "batch_size < data_parallel_size",
                    "batch_size": bs,
                    "dp_size": dp_size,
                    "input_len": b_input_len,
                    "output_len": b_output_len,
                })
                _rank0_write_json(skip_runner_json, {
                    "ok": True,
                    "skipped": True,
                    "label": label,
                    "batch_size": bs,
                    "input_len": b_input_len,
                    "output_len": b_output_len,
                    "reason": "batch_size < data_parallel_size",
                })
            if dist is not None:
                dist.barrier()
            continue

        # Validate per-bucket model len constraint.
        if llm.llm_engine.model_config.max_model_len < (b_input_len + b_output_len):
            _rank0_write_text(
                out_root / f"child_{label}_error.log",
                f"max_model_len is smaller than input_len + output_len for bucket {tag}; adjust target.json.\n",
            )
            _update_status("error", extra={"error": f"max_model_len < input_len + output_len for bucket {tag}"})
            return 2

        # Reconstruct SamplingParams per-bucket (output_len may vary).
        sampling_params = SamplingParams(
            n=int(getattr(args, "n", 1)),
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=b_output_len,
            detokenize=not bool(getattr(args, "disable_detokenize", False)),
        )

        raw_json = json_dir / f"{label}_{tag}.json"
        runner_json = json_dir / f"{label}_{tag}.runner.json"
        log_path = logs_dir / f"{label}_{tag}.log"

        # Override per-bucket args that should affect only prompt construction / output paths.
        setattr(args, "batch_size", int(bs))
        setattr(args, "input_len", int(b_input_len))
        setattr(args, "output_len", int(b_output_len))
        setattr(args, "num_iters", int(num_iters))
        setattr(args, "output_json", str(raw_json))

        start = datetime.now(timezone.utc)
        status: Dict[str, Any] = {
            "ok": False,
            "returncode": 1,
            "label": label,
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
            "start_time": start.isoformat(),
            "nsys_profile": bool(nsys_profile),
            "torch_profile": bool(torch_profile),
        }
        status.update(_nsys_runner_mapping(bucket))

        with _rank0_open_log(log_path, "w") as bucket_log, _rank0_open_log(
            child_log_path, "a"
        ) as child_log:

            def _log(msg: str) -> None:
                bucket_log.write(msg + "\n")
                bucket_log.flush()
                child_log.write(msg + "\n")
                child_log.flush()
                if _IS_RANK0:
                    print(msg, flush=True)

            try:
                _update_status("bucket_start", batch_size=bs)
                _log(f"=== inproc vllm bench latency sweep ({label}) {tag} ===")
                _log(
                    _format_cmd_for_md(
                        ["vllm", "bench", "latency"]
                        + _build_cli_equivalent_args_for_inproc(
                            model_id=model_id,
                            tp=tp,
                            max_model_len=max_model_len,
                            input_len=b_input_len,
                            output_len=b_output_len,
                            batch_size=bs,
                            num_iters=num_iters,
                            output_json=raw_json,
                            extra_args=extra_args,
                        ),
                        {},
                    )
                )

                # Build the per-bucket timing prompts. For source="gsm8k"
                # (default) these are deterministic crc32-seeded windows of real
                # bundled-GSM8K tokens — content-derived seeding makes the old
                # "each rank generates the same tensor then partitions" comment
                # TRUE by construction: every DP rank computes identical windows
                # from the identical corpus, then _partition_prompts slices its
                # stride so the union covers the original batch exactly once.
                # For source="random" the legacy unseeded np.random.randint path
                # is used (DP-only-safe; baseline and opt may differ).
                all_dummy_prompts: "list[PromptType]" = _build_timing_prompts(
                    _timing_corpus,
                    batch_size=bs,
                    input_len=b_input_len,
                    output_len=b_output_len,
                    source=dummy_prompt_source,
                    np_module=np,
                )
                if dp_size > 1:
                    dummy_prompts = _partition_prompts(
                        all_dummy_prompts,
                        dp_size=dp_size,
                        dp_rank=dp_rank,
                        input_len=b_input_len,
                    )
                else:
                    dummy_prompts = all_dummy_prompts

                # `llm_generate` returns the captured outputs from the
                # non-beam-search branch so the timing loop can harvest
                # `RequestOutput.metrics` (spec §3.1 step 2). The beam-search
                # branch deliberately discards — beam_search() does not emit
                # the v1 metrics payload — and the bucket falls back to
                # Tier-C (`OL/(IL+OL)`) downstream.
                def llm_generate() -> Any:
                    if not bool(getattr(args, "use_beam_search", False)):
                        outputs = llm.generate(
                            dummy_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                        return outputs
                    else:
                        llm.beam_search(
                            dummy_prompts,
                            BeamSearchParams(
                                beam_width=int(getattr(args, "n", 1)),
                                max_tokens=b_output_len,
                                ignore_eos=True,
                            ),
                        )
                        return None

                # Always import torch for the per-step cuda.synchronize() —
                # spec §3.1 step 8: DP=1 path needs cuda.sync before t1 for
                # wall-time accuracy (llm.generate returns when host queues
                # drain; device work may still be in flight).
                import torch as _torch_sync

                if dp_size > 1:
                    # DP timing: rendezvous before start, CUDA-sync + rendezvous
                    # after — the bucket latency is the wall time of the slowest
                    # rank's generate(), which is what an external client would
                    # observe. torch.cuda.synchronize() is essential because
                    # llm.generate() returns as soon as host queues are drained;
                    # device work may still be in flight.

                    def run_to_completion() -> Tuple[float, Any]:
                        dist.barrier()
                        t0 = _time.perf_counter()
                        outputs = llm_generate()
                        _torch_sync.cuda.synchronize()
                        dist.barrier()
                        t1 = _time.perf_counter()
                        return (t1 - t0), outputs
                else:
                    def run_to_completion() -> Tuple[float, Any]:
                        t0 = _time.perf_counter()
                        outputs = llm_generate()
                        # Spec §3.1 step 8: cuda.sync BEFORE t1 for wall-time
                        # accuracy on DP=1 path (fixes prior bug where t1 was
                        # captured before async device work completed).
                        _torch_sync.cuda.synchronize()
                        t1 = _time.perf_counter()
                        return (t1 - t0), outputs

                warmup = int(getattr(args, "num_iters_warmup", 10))
                deadline = _time.monotonic() + float(timeout_s_per_bucket)
                _update_status("warmup", batch_size=bs, extra={"num_iters_warmup": warmup})
                _log("Warming up...")
                last_status_t = _time.monotonic()
                for i in range(warmup):
                    if _time.monotonic() > deadline:
                        raise TimeoutError(f"Bucket timeout exceeded ({timeout_s_per_bucket}s)")
                    llm_generate()
                    if (_time.monotonic() - last_status_t) > 5.0:
                        _update_status("warmup", batch_size=bs, extra={"warmup_iter": i + 1})
                        last_status_t = _time.monotonic()

                # Spec-decode acceptance: snapshot the cumulative counters AFTER
                # warmup (so warmup drafts are excluded) and BEFORE the measured
                # loop. Diffed after the loop for correct per-bucket attribution
                # (the Prometheus counters accumulate across buckets in this one
                # child process). Fully defensive — None when unavailable.
                _spec_before = _snapshot_spec_decode_counters(llm)

                # Benchmark.
                _update_status("benchmark", batch_size=bs, extra={"num_iters": int(getattr(args, "num_iters", 30))})
                latencies: List[float] = []
                last_status_t = _time.monotonic()
                num_iters_eff = int(getattr(args, "num_iters", 30))

                # Start nsys capture for this bucket (if enabled).
                # Selected-step nsys uses vLLM's CUDA profiler on every rank so
                # each external-launcher DP rank arms its local worker profiler.
                # Plain nsys buckets use direct cudaProfilerStart/Stop helpers,
                # which are rank-0-gated while synchronizing CUDA on every rank.
                _torch_prof = None
                selected_step_nsys = bool(nsys_profile and _is_nsys_selected_step_bucket(bucket))
                if selected_step_nsys:
                    llm.start_profile()
                    _log(
                        "[nsys] vLLM CUDA profiler armed for selected step "
                        f"{bucket['nsys_capture_output_step']} "
                        f"(window={bucket['nsys_capture_window_output_len']})"
                    )
                elif nsys_profile:
                    import torch as _torch_prof
                    _nsys_start_if_rank0(_torch_prof)
                    _log(f"[nsys] cudaProfilerStart for {tag}")

                # Start torch profiler capture for this bucket (if enabled).
                # Under DP>1, torch_profile is rank-0-only: a DP-aware
                # merge of multiple profiler traces is out of scope for this
                # script, so rank 0's trace is the representative artifact.
                if torch_profile and _IS_RANK0:
                    bucket_profile_dir = str(_v2_profiling_dir(out_root, "torch_profile") / f"{label}_{tag}")
                    Path(bucket_profile_dir).mkdir(parents=True, exist_ok=True)
                    if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'vllm_config'):
                        llm.llm_engine.vllm_config.profiler_config.torch_profiler_dir = bucket_profile_dir
                    llm.start_profile()
                    _log(f"[torch_profile] started for {tag} -> {bucket_profile_dir}")

                # Per-request prefill/decode timing accumulators (spec §3.1).
                # Pool per-request deltas across ALL measured iterations so the
                # bucket-level mean/p50 is robust to per-iteration jitter.
                # Beam-search branch returns None and is skipped (Tier-C
                # fallback is applied downstream when these stay empty).
                prefills_local: List[float] = []
                decodes_local: List[float] = []
                # Per-iteration batch-mean phase series (spec: DILUTED_PASS
                # phase-level Welch). Each entry is the mean prefill/decode
                # delta across THAT iteration's batch of concurrent requests,
                # so len == len(latencies) (one sample per iteration) — the
                # same sample unit as the e2e Welch on `latencies`.
                prefill_iter_means: List[float] = []
                decode_iter_means: List[float] = []

                try:
                    for i in range(num_iters_eff):
                        if _time.monotonic() > deadline:
                            raise TimeoutError(f"Bucket timeout exceeded ({timeout_s_per_bucket}s)")
                        wall_s, iter_outputs = run_to_completion()
                        latencies.append(wall_s)
                        # Per-iteration throwaway phase accumulators (reset each
                        # pass) — feed the per-iteration batch-mean series below.
                        _iter_pf: List[float] = []
                        _iter_dc: List[float] = []
                        # Defensive harvest — never break the timing loop on a
                        # metrics extraction failure. The vLLM metrics field
                        # names live ONLY in _request_phase_deltas.
                        if iter_outputs:
                            try:
                                _iter_pf, _iter_dc = _request_phase_deltas(iter_outputs)
                                prefills_local.extend(_iter_pf)
                                decodes_local.extend(_iter_dc)
                            except Exception:
                                # Don't let metric harvest derail the bench.
                                pass
                        # Batch mean for THIS iteration (one sample per iter,
                        # aligned with latencies[i]). Empty on beam-search /
                        # metrics-gap iterations -> series stays shorter than
                        # latencies -> emission guard below drops the arrays.
                        if _iter_pf:
                            prefill_iter_means.append(sum(_iter_pf) / len(_iter_pf))
                        if _iter_dc:
                            decode_iter_means.append(sum(_iter_dc) / len(_iter_dc))
                        if (_time.monotonic() - last_status_t) > 5.0:
                            _update_status("benchmark", batch_size=bs, extra={"iter": i + 1})
                            last_status_t = _time.monotonic()
                finally:
                    # Stop nsys capture for this bucket. Must happen even on
                    # exception to avoid corrupting the repeat:N capture count
                    # and leaving nsys waiting indefinitely. Under DP the
                    # rank-0-only cudaProfilerStop still ends the global
                    # capture range (nsys-level mechanism).
                    if selected_step_nsys:
                        llm.stop_profile()
                        _log(
                            "[nsys] vLLM CUDA profiler stopped for selected step "
                            f"{bucket['nsys_capture_output_step']}"
                        )
                    elif nsys_profile and _torch_prof is not None:
                        _nsys_stop_if_rank0(_torch_prof)
                        _log(f"[nsys] cudaProfilerStop for {tag}")
                    if torch_profile and _IS_RANK0:
                        llm.stop_profile()
                        _log(f"[torch_profile] stopped for {tag}")

                # Spec-decode acceptance: snapshot again after the measured loop
                # and diff against the pre-loop snapshot for this bucket. Emits a
                # block only when there was real draft activity; never raises.
                _spec_after = _snapshot_spec_decode_counters(llm)
                _spec_block = _spec_decode_block_from_diff(_spec_before, _spec_after)

                # Match vLLM bench JSON schema.
                arr = np.array(latencies, dtype=np.float64)
                percentages = [10, 25, 50, 75, 90, 99]
                percentiles = np.percentile(arr, percentages)
                avg_latency = float(np.mean(arr))

                _log(f"Avg latency: {avg_latency} seconds")
                for percentage, percentile in zip(percentages, percentiles):
                    _log(f"{percentage}% percentile latency: {float(percentile)} seconds")

                raw = {
                    "avg_latency": avg_latency,
                    "latencies": [float(x) for x in latencies],
                    "percentiles": {str(k): float(v) for k, v in zip(percentages, percentiles)},
                }
                # Spec §3.1 step 4: emit per-bucket prefill/decode aggregates
                # harvested from RequestOutput.metrics. Beam-search and
                # legacy vLLM builds leave these unset (consumer falls back
                # to Tier-C: OL/(IL+OL)).
                if prefills_local and decodes_local:
                    _pf_avg = sum(prefills_local) / len(prefills_local)
                    _dc_avg = sum(decodes_local) / len(decodes_local)
                    raw["prefill_avg_s"] = float(_pf_avg)
                    raw["decode_avg_s"] = float(_dc_avg)
                    raw["prefill_p50_s"] = float(np.median(np.array(prefills_local, dtype=np.float64)))
                    raw["decode_p50_s"] = float(np.median(np.array(decodes_local, dtype=np.float64)))
                    # Denominate the AUTHORITATIVE wall (avg_latency), not
                    # prefill_avg + decode_avg. The latter is a per-request-mean
                    # sub-interval that discards admission/ramp wall, so on a
                    # prefill-heavy workload it overstates the share badly (0.879
                    # vs a true 0.524 at ISL 8192 / BS 32). Consumers multiply this
                    # by a decode delta to get an E2E-equivalent figure — see
                    # references/validation-defaults.md § DILUTED_PASS and
                    # references/e2e-delta-math.md § Denominator rule — so a share
                    # normalized to anything but avg_latency makes that product
                    # too large and lets a gate pass on too small a real win.
                    if avg_latency > 0:
                        raw["decode_share_of_e2e"] = float(_dc_avg / avg_latency)
                    # Retained for provenance/diagnostics: the phase-only split.
                    _total = _pf_avg + _dc_avg
                    if _total > 0:
                        raw["decode_share_of_phase_sum"] = float(_dc_avg / _total)
                    raw["num_request_metric_samples"] = int(len(prefills_local))
                # Per-iteration batch-mean phase series (DILUTED_PASS phase
                # Welch). Emitted only when every iteration harvested a phase
                # sample (length-matched to `latencies`) so the phase Welch has
                # the same n as the e2e Welch — a partial/misaligned series is
                # dropped rather than silently shortened.
                if (prefill_iter_means and decode_iter_means
                        and len(prefill_iter_means) == len(latencies)):
                    raw["prefill_iter_means_s"] = [float(x) for x in prefill_iter_means]
                    raw["decode_iter_means_s"] = [float(x) for x in decode_iter_means]
                # Additive spec-decode acceptance block (rank-0 raw write only).
                # Under DP>1 these are rank-0-shard counts, not batch totals.
                if _spec_block:
                    raw["spec_decode"] = _spec_block
                _rank0_write_json(raw_json, raw)

                # Optional: keep parity with vLLM bench output sidecar format.
                if _IS_RANK0:
                    try:
                        vllm_latency.save_to_pytorch_benchmark_format(args, raw)
                    except Exception:
                        _log("WARNING: failed to write pytorch benchmark sidecar JSON.")
                        _log(traceback.format_exc())

                end = datetime.now(timezone.utc)
                status.update({
                    "ok": True,
                    "returncode": 0,
                    "end_time": end.isoformat(),
                    "duration_s": (end - start).total_seconds(),
                })
                _update_status("bucket_done", batch_size=bs, extra={"ok": True})
            except Exception as e:
                end = datetime.now(timezone.utc)
                status.update({
                    "ok": False,
                    "returncode": 1,
                    "error": f"{type(e).__name__}: {e}",
                    "end_time": end.isoformat(),
                    "duration_s": (end - start).total_seconds(),
                    "traceback": traceback.format_exc(),
                })
                _update_status("bucket_failed", batch_size=bs, extra={"ok": False, "error": status.get("error")})
                # Ensure the raw JSON exists even on failure (helps parent remain deterministic).
                if _IS_RANK0 and not raw_json.exists():
                    _rank0_write_json(raw_json, {"error": status["error"]})

            _rank0_write_json(runner_json, status)

        # DP>1 collective error barrier (outside the try/except and the log
        # context mgr): any rank that failed locally triggers all ranks to
        # abort the sweep — otherwise a rank whose generate() raised leaves
        # surviving ranks waiting forever on the next dist.barrier().
        if dp_size > 1:
            assert dist is not None and error_flag is not None
            error_flag.zero_()
            if not status.get("ok"):
                error_flag[0] = 1
            dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
            if int(error_flag.item()) > 0:
                if _IS_RANK0:
                    detail = f" (local error: {status.get('error', 'ok on this rank')})"
                    print(
                        f"ERROR in bucket {tag}: rank failure detected — all ranks aborting{detail}",
                        flush=True,
                    )
                break

    return 0


def _render_md_table(rows: List[Dict[str, Any]], baseline_label: str, opt_label: str) -> str:
    # Determine if rows are heterogeneous (mixed IL/OL values).
    il_ol_set: set = set()
    for r in rows:
        il = r.get("input_len")
        ol = r.get("output_len")
        if il is not None and ol is not None:
            il_ol_set.add((il, ol))
    heterogeneous = len(il_ol_set) > 1

    # OTPS/TPOT columns are shown only when at least one row carries the
    # derived throughput data, so legacy runs render the original table.
    has_throughput = any(
        isinstance(r.get(opt_label, {}).get("otps"), (int, float))
        or isinstance(r.get("otps_gain_pct"), (int, float))
        for r in rows
    )
    tput_hdr = f" {opt_label} OTPS (tok/s) | OTPS Δ | {opt_label} TPOT (ms) | TPOT Δ |" if has_throughput else ""
    tput_sep = "---:|---:|---:|---:|" if has_throughput else ""

    if heterogeneous:
        header = f"| Input Len | Output Len | Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement |{tput_hdr} Fast-path evidence |"
        sep = f"|---:|---:|---:|---:|---:|---:|---:|{tput_sep}---|"
    else:
        header = f"| Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement |{tput_hdr} Fast-path evidence |"
        sep = f"|---:|---:|---:|---:|---:|{tput_sep}---|"

    lines = [header, sep]
    for r in rows:
        bs = r["batch_size"]
        b = r.get(baseline_label, {})
        o = r.get(opt_label, {})
        b_avg = b.get("avg_s")
        o_avg = o.get("avg_s")
        speedup = r.get("speedup")
        improve = r.get("improvement_pct")
        evidence = o.get("fastpath_evidence", {}).get("status", "unknown")

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                return f"{x:.6g}"
            return str(x)

        # Throughput cells: opt OTPS (tok/s), OTPS gain %, opt TPOT (ms), TPOT improvement %.
        tput_cells = ""
        if has_throughput:
            o_otps = o.get("otps")
            o_tpot_s = o.get("tpot_s")
            o_tpot_ms = o_tpot_s * 1000.0 if isinstance(o_tpot_s, (int, float)) else None
            otps_gain = r.get("otps_gain_pct")
            tpot_improve = r.get("tpot_improvement_pct")
            tput_cells = f" {fmt(o_otps)} | {fmt(otps_gain)}% | {fmt(o_tpot_ms)} | {fmt(tpot_improve)}% |"

        if heterogeneous:
            il = r.get("input_len", "")
            ol = r.get("output_len", "")
            lines.append(
                f"| {il} | {ol} | {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% |{tput_cells} {evidence} |"
            )
        else:
            lines.append(
                f"| {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% |{tput_cells} {evidence} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument("--target-json", type=str, default=None, help="Path to target.json (default: {artifact_dir}/target.json)")
    p.add_argument("--out-name", type=str, default="e2e_latency",
                   help=("(LEGACY) Output directory name under {artifact_dir}. "
                         "On v2 layouts (rounds/ exists), this is replaced by --round/--slot "
                         "and emitting a custom value here is a hard error."))
    # v2 layout: --round + --slot resolve to rounds/{N}/sweeps/{SLOT}/.
    # See references/artifact-layout.md.
    p.add_argument("--round", type=int, default=None,
                   help=("Campaign round (1-indexed). With --slot, resolves to "
                         "rounds/{N}/sweeps/{SLOT}/. Reads state.json.campaign.current_round "
                         "if omitted while --slot is given."))
    p.add_argument("--slot", type=str, default=None,
                   help=("Sweep slot under rounds/{N}/sweeps/. One of: "
                         "baseline | opt/{op_id} | integration | golden_capture."))
    p.add_argument("--timeout-s", type=int, default=1800, help="Timeout per bucket (seconds)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output dir instead of archiving it")
    p.add_argument("--require-fastpath", action="store_true", help="Fail if opt fast-path evidence patterns do not pass")
    p.add_argument("--heartbeat-s", type=int, default=30, help="Heartbeat interval while a subprocess is quiet")
    p.add_argument(
        "--labels",
        type=str,
        default="baseline",
        help=(
            "Comma-separated labels to run: 'baseline' (default), 'opt', or 'baseline,opt'. "
            "Use '--labels baseline' for Stage 1 baseline-only sweeps (no opt). "
            "Use '--labels opt' with --baseline-from for Stage 5 validation "
            "(opt only, compared against the Stage 1 baseline — never re-run a "
            "baseline from the worktree)."
        ),
    )
    p.add_argument(
        "--nsys-profile",
        action="store_true",
        default=False,
        help=(
            "Capture per-bucket nsys profiles during inproc_sweep. "
            "Produces one .nsys-rep per bucket in {out}/nsys/. "
            "Uses --capture-range=cudaProfilerApi with repeat mode."
        ),
    )
    p.add_argument(
        "--nsys-mode",
        choices=["node", "graph"],
        default="node",
        help=(
            "nsys cuda-graph-trace mode. 'node' is the Stage 2 ranking "
            "default; 'graph' is optional diagnostic enrichment."
        ),
    )
    p.add_argument(
        "--nsys-trace",
        choices=["cuda", "cuda-sw"],
        default=None,
        help=(
            "CUDA tracing backend for nsys. Default is derived from "
            "target.hardware: 'cuda-sw' on Blackwell (B200/B300/GB*, where "
            "hardware tracing stalls during graph replay), else 'cuda'. "
            "Pass the flag to override."
        ),
    )
    p.add_argument(
        "--nsys-extra-flags",
        type=str,
        default="",
        help="Extra flags to pass to nsys profile (e.g. '--stats=true')",
    )
    p.add_argument(
        "--nsys-timeout-s",
        type=int,
        default=600,
        help=(
            "Per-bucket timeout in seconds when --nsys-profile is active (default: 600). "
            "Total timeout = nsys_timeout_s * num_buckets."
        ),
    )
    p.add_argument(
        "--nsys-output-len",
        type=int,
        default=None,
        help=(
            "Override output_len for nsys profiling only (default: use workload output_len). "
            "Decouples profiling sequence length from benchmark sequence length to avoid "
            "superlinear nsys overhead on models with many kernels/step. "
            "Requires --nsys-profile."
        ),
    )
    p.add_argument(
        "--nsys-num-iters",
        type=int,
        default=None,
        help=(
            "Override num_iters for nsys profiling only. Defaults to 1 when "
            "--nsys-output-len is set. Requires --nsys-profile."
        ),
    )
    p.add_argument(
        "--nsys-capture-output-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated output-depth steps or percentages to profile as "
            "selected-step, shape-equivalent short captures, e.g. '2,50%%,100%%'. "
            "Percentages resolve against workload output_len or --nsys-output-len. "
            "Requires --nsys-profile and --nsys-num-iters 1."
        ),
    )
    p.add_argument(
        "--nsys-capture-window-output-len",
        type=int,
        default=None,
        help=(
            "Lower bound on the short generation window used with "
            "--nsys-capture-output-steps (default: 2). The script shifts input_len so the "
            "final token in this window has the requested decode-step shape, and "
            "automatically RAISES the window above this value when needed so the capture "
            "clears all chunked-prefill worker-steps (avoids capturing a prefill chunk on "
            "long-context workloads). The captured decode depth is unaffected by the raise."
        ),
    )
    p.add_argument(
        "--torch-profile",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--baseline-from",
        type=str,
        default=None,
        help=(
            "Path to the accepted Stage 1 baseline output containing golden refs "
            "and baseline timing. Use with '--labels opt' for Stage 5 validation "
            "and with opt-only correctness/profiling slots."
        ),
    )
    p.add_argument(
        "--baseline-from-arm",
        type=str,
        choices=["baseline", "opt"],
        default="baseline",
        help=(
            "Which arm of --baseline-from is imported as the comparator. The import "
            "is by literal filename {prefix}_{tag}.json, so 'baseline' (default) reads "
            "that directory's baseline arm and 'opt' reads its optimized arm. Round >= 2 "
            "integration sweeps whose --baseline-from is a pre-SHIP directory must pass "
            "'opt' to compare against the promoted mainline instead of the stale "
            "pre-SHIP baseline."
        ),
    )
    p.add_argument(
        "--fresh-cache",
        action="store_true",
        default=False,
        help=(
            "Isolate vLLM/Triton compile caches per sweep. First launch pays full "
            "compile (~5 min for large models). Total wall-clock = compile_time + "
            "(N-1) * bench_time per label. Cache dir is removed after successful "
            "sweep. Does NOT disable the compile cache within a single launch."
        ),
    )
    p.add_argument("--capture-golden-refs", action="store_true", default=False,
                   help="Stage 1: run GSM8K greedy decode and save golden refs to json/golden_refs.json")
    p.add_argument("--verify-correctness", action="store_true", default=False,
                   help="Stage 5: compare GSM8K outputs against golden refs; exit nonzero on mismatch")
    p.add_argument("--correctness-num-questions", type=int, default=1319,
                   help="Number of GSM8K questions for correctness phase (default: 1319 = full test split)")
    p.add_argument("--correctness-tolerance-pct", type=float, default=1.0,
                   help=("Allowed accuracy drop in percentage points (default: 1.0 = 1pp). "
                         "PASS iff opt_accuracy >= baseline_accuracy - tolerance/100. "
                         "Set to 0.0 for strict opt >= baseline comparison."))
    p.add_argument("--dummy-prompt-source", choices=["gsm8k", "random"], default="gsm8k",
                   help=("Prompt source for the TIMING loop: 'gsm8k' (default; real "
                         "windowed bundled-GSM8K tokens, deterministic per workload) or "
                         "'random' (legacy uniform random token IDs). Real prompts are "
                         "required for faithful speculative-decode acceptance; 'random' "
                         "is retained for tests/debugging and does not need the bundled data."))
    p.add_argument("--_child-label", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_out-root", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-profile", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_cudagraph-capture-sizes", nargs="+", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-output-len", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-num-iters", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-capture-output-steps", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-capture-window-output-len", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_torch-profile", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_capture-golden-refs", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_verify-correctness", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_correctness-num-questions", type=int, default=1319, help=argparse.SUPPRESS)
    p.add_argument("--_correctness-tolerance-pct", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--_dummy-prompt-source", type=str, default="gsm8k", help=argparse.SUPPRESS)
    # ---- Hang watchdog (default OFF; opt-in for callers that hit the
    # intermittent in-engine GPU-kernel hang under multi-seq chunked prefill).
    # When ON, the parent launches each child in its OWN process group and, if a
    # bucket's status file goes stale past the phase-gated threshold while the
    # child is wedged inside a single generate() call (which the per-bucket
    # deadline cannot catch, since it is only checked BETWEEN generate() calls),
    # it killpg()s the whole group (taking vLLM EngineCore subprocesses with it),
    # then RELAUNCHES skipping the buckets already completed on disk PLUS the
    # wedged one. Completed buckets are already persisted to json/, so partial
    # results survive and the sweep reports the buckets that ran. Default-OFF
    # keeps every existing caller byte-for-byte unchanged. ----
    p.add_argument("--hang-watchdog", action="store_true", default=False,
                   help=("Detect an in-engine kernel hang (status file frozen while "
                         "the child is wedged inside generate()), kill the child's "
                         "process group, and relaunch skipping done+wedged buckets. "
                         "Partial results are preserved. Default OFF."))
    p.add_argument("--hang-stale-warmup-s", type=int, default=130,
                   help=("Hang watchdog: max seconds a bucket's status may stay frozen "
                         "in the warmup/benchmark phase before it is treated as hung "
                         "(default 130). Auto-widened for heavy buckets (long prefill / "
                         "large batch). Ignored unless --hang-watchdog."))
    p.add_argument("--hang-max-relaunch", type=int, default=8,
                   help=("Hang watchdog: cap on total child relaunches across the whole "
                         "sweep (backstop against a relaunch loop; default 8). Ignored "
                         "unless --hang-watchdog."))
    p.add_argument("--_skip-tags", type=str, default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

    # [MF4] Real-prompt timing fail-fast precheck (runs in BOTH parent and child
    # since the child re-enters main()). ALL timing runs now require the bundled
    # GSM8K data co-located with the script — fail at startup, BEFORE the
    # expensive model load, rather than mid-bucket. Skipped for the legacy
    # --dummy-prompt-source random path, the only one that does not need the data.
    #
    # The PUBLIC --dummy-prompt-source is authoritative in the parent; the parent
    # forwards it to the child only as the HIDDEN --_dummy-prompt-source (so the
    # child's public flag stays at its gsm8k default). Resolve the effective
    # source from the hidden flag when this is a child invocation so the precheck
    # matches what the child will actually run.
    _effective_prompt_source = (
        getattr(args, "_dummy_prompt_source", "gsm8k")
        if getattr(args, "_child_label", None)
        else getattr(args, "dummy_prompt_source", "gsm8k")
    )
    if _effective_prompt_source == "gsm8k" and not _GSM8K_FULL_PATH.exists():
        raise SystemExit(
            f"Real-prompt timing requires the bundled GSM8K full dataset at "
            f"{_GSM8K_FULL_PATH}, but it was not found. ALL timing runs now use "
            "real windowed GSM8K prompts by default and need this file co-located "
            "with the script. Re-bundle the dataset, or pass "
            "--dummy-prompt-source random to use legacy uniform-random tokens."
        )

    # Transition safety: warn if old locking system files exist
    old_lock_dir = Path('/tmp/ammo_gpu_locks')
    if old_lock_dir.exists() and any(old_lock_dir.glob('*.lock')):
        print(
            'WARNING: Old GPU lock files found at /tmp/ammo_gpu_locks/. '
            'GPU reservation is now managed by hooks. Old locks can be safely deleted.',
            file=sys.stderr,
        )

    # Validate --nsys-profile constraints early.
    if args.nsys_profile:
        if not shutil.which("nsys"):
            raise SystemExit(
                "nsys not found on PATH. Install Nsight Systems CLI or remove --nsys-profile."
            )
    if (
        args.nsys_output_len is not None
        or args.nsys_num_iters is not None
        or args.nsys_capture_output_steps is not None
        or args.nsys_capture_window_output_len is not None
    ) and not args.nsys_profile:
        raise SystemExit(
            "--nsys-output-len, --nsys-num-iters, --nsys-capture-output-steps, "
            "and --nsys-capture-window-output-len all require --nsys-profile."
        )
    if args.nsys_output_len is not None and args.nsys_output_len <= 0:
        raise SystemExit(
            f"--nsys-output-len must be positive, got {args.nsys_output_len}"
        )
    if args.nsys_num_iters is not None and args.nsys_num_iters <= 0:
        raise SystemExit(
            f"--nsys-num-iters must be positive, got {args.nsys_num_iters}"
        )
    if args.nsys_capture_window_output_len is not None and args.nsys_capture_window_output_len <= 0:
        raise SystemExit(
            "--nsys-capture-window-output-len must be positive, got "
            f"{args.nsys_capture_window_output_len}"
        )
    if args.nsys_capture_window_output_len is not None and args.nsys_capture_output_steps is None:
        raise SystemExit(
            "--nsys-capture-window-output-len requires --nsys-capture-output-steps."
        )
    if args.nsys_capture_output_steps is not None and args.nsys_num_iters not in (None, 1):
        raise SystemExit(
            "--nsys-capture-output-steps requires --nsys-num-iters 1 because "
            "each selected-step profile must capture exactly one short generate()."
        )

    # Validate correctness flag constraints.
    if args.capture_golden_refs and args.verify_correctness:
        raise SystemExit("--capture-golden-refs and --verify-correctness are mutually exclusive")
    if args.verify_correctness and not args.baseline_from:
        raise SystemExit("--verify-correctness requires --baseline-from (to import golden_refs.json)")
    # Gate 5.1b (correctness) and Gate 5.3a (nsys) MUST be separate invocations.
    # nsys wraps the entire child process, so its trace buffers grow across the
    # 1319-question GSM8K correctness pass until the cgroup OOM-kills EngineCore
    # (return code 137 / SIGKILL, no traceback). --nsys-output-len only bounds
    # the bench loop, NOT the correctness phase, so retrying with smaller
    # --nsys-output-len does not help. See references/nsys-profiling-guide.md
    # §3.14 for the full mechanism and the correct two-invocation workflow.
    if args.nsys_profile and args.verify_correctness:
        raise SystemExit(
            "--nsys-profile and --verify-correctness cannot be combined: nsys "
            "trace buffers grow across the full GSM8K correctness pass and "
            "cgroup-OOM EngineCore. Run Gate 5.1b (--verify-correctness) and "
            "Gate 5.3a (--nsys-profile) as separate sweep invocations. See "
            "references/nsys-profiling-guide.md §3.14."
        )

    # Profiling flags are incompatible with --slot baseline: nsys wraps the
    # entire process (10-30% overhead on ALL iterations regardless of capture
    # range), and torch.profiler adds CUPTI subscription overhead. The baseline
    # slot is the authoritative E2E timing source for speedup calculations —
    # contaminated numbers propagate into every subsequent stage's comparisons.
    # Run a clean sweep first (--slot baseline), then a separate profiling
    # invocation (traces route to rounds/{N}/profiling/ automatically).
    slot = getattr(args, "slot", None)
    if slot == "baseline" and (args.nsys_profile or args.torch_profile):
        profiler_flag = "--nsys-profile" if args.nsys_profile else "--torch-profile"
        raise SystemExit(
            f"ERROR: {profiler_flag} cannot be combined with --slot baseline. "
            "Profiling adds overhead that contaminates E2E timing (nsys: 10-30% "
            "process-wide, torch: CUPTI subscription). The baseline slot is the "
            "authoritative timing source for speedup calculations.\n\n"
            "Correct Stage 1 workflow (two invocations):\n"
            "  1. Clean E2E:  --round N --slot baseline --labels baseline "
            "--capture-golden-refs\n"
            "  2. Profiling:  --round N --slot profiling --nsys-profile "
            "(or --torch-profile)\n"
            "     (traces route to rounds/{N}/profiling/ regardless of slot)\n\n"
            "See references/e2e-latency-guide.md for details."
        )

    # Parse and validate --labels.
    selected_labels = {s.strip() for s in args.labels.split(",")}
    invalid_labels = selected_labels - {"baseline", "opt"}
    if invalid_labels:
        raise SystemExit(
            f"Invalid --labels value(s): {invalid_labels}. "
            "Accepted: 'baseline', 'opt', or 'baseline,opt'."
        )
    if not selected_labels:
        raise SystemExit("--labels must specify at least one label.")

    # Validate --baseline-from early.
    baseline_from: Optional[Path] = None
    if args.baseline_from:
        baseline_from = Path(args.baseline_from).expanduser().resolve()
        if not baseline_from.is_dir():
            raise SystemExit(
                f"--baseline-from path does not exist or is not a directory: {baseline_from}"
            )

    # Gate-slot misinvocation guard: Stage 5 sibling slots and integration feed
    # ship verdicts, so the script enforces gate-quality invocation instead of
    # trusting the caller's flags.
    is_gate_slot = _is_gate_slot(slot)
    if is_gate_slot and not getattr(args, "_child_label", None):
        # 1. Clean timing gates compare the optimized arm against the Stage 1
        #    baseline (--labels opt with --baseline-from). Re-running a
        #    baseline arm from the worktree risks executing the optimized code
        #    path via the editable install, contaminating both arms —
        #    baseline,opt remains allowed only when the caller can prove the
        #    baseline arm runs the original path (reversible control).
        is_profiling_run = bool(args.nsys_profile or args.torch_profile)
        is_timing_gate = bool(
            (slot == "integration" or (slot and slot.startswith("opt/")))
            and not is_profiling_run
            and not args.verify_correctness
        )
        if is_timing_gate and selected_labels == {"opt"} and not args.baseline_from:
            print(
                f"WARNING: clean timing --slot {slot} with --labels opt has no "
                "--baseline-from; Gate 5.3b comparison requires the Stage 1 "
                "baseline reference.",
                file=sys.stderr,
            )

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    target_path = Path(args.target_json).expanduser().resolve() if args.target_json else (artifact_dir / "target.json")

    spec = _load_json(target_path)

    artifact_dir_spec = Path(_require(spec, "artifact_dir", "root")).expanduser().resolve()
    if artifact_dir_spec != artifact_dir:
        # This mismatch often indicates copy/paste mistakes; fail fast.
        raise SystemExit(
            f"artifact_dir mismatch: CLI={artifact_dir} vs target.json={artifact_dir_spec}. "
            "Fix target.json or pass --artifact-dir to match."
        )

    target = _require(spec, "target", "root")
    if not isinstance(target, dict):
        raise SystemExit("root.target must be an object")

    model_id = _require(target, "model_id", "target")
    tp = _require_int(target, "tp", "target")
    ep = _require_int(target, "ep", "target")
    max_model_len = _require_int(target, "max_model_len", "target")

    # nsys trace backend: hardware-derived unless the flag was passed.
    if args.nsys_trace is None:
        args.nsys_trace = _default_nsys_trace(target.get("hardware"))
        nsys_trace_source = f"auto from target.hardware={target.get('hardware')!r}"
    else:
        nsys_trace_source = "explicit --nsys-trace"

    workload = _require(spec, "workload", "root")
    if not isinstance(workload, dict):
        raise SystemExit("root.workload must be an object")

    # Expand workload into buckets (supports both legacy flat and new matrix format).
    # Legacy flat fields are still required when workload_matrix is absent.
    has_matrix = "workload_matrix" in workload
    if not has_matrix:
        # Validate legacy required fields.
        _require_int(workload, "input_len", "workload")
        _require_int(workload, "output_len", "workload")
        _require_list_int(workload, "batch_sizes", "workload")

    buckets = _expand_workload_to_buckets(workload)
    num_iters = _require_int(workload, "num_iters", "workload")

    # Legacy convenience vars (for backward-compat in console output and workload JSON).
    input_len = workload.get("input_len")
    output_len = workload.get("output_len")
    batch_sizes = workload.get("batch_sizes")

    bench = _require(spec, "bench", "root")
    if not isinstance(bench, dict):
        raise SystemExit("root.bench must be an object")

    runner = _require(bench, "runner", "bench")
    if runner != "vllm_bench_latency":
        raise SystemExit(f"Unsupported bench.runner: {runner!r} (expected 'vllm_bench_latency')")

    vllm_cmd = _require(bench, "vllm_cmd", "bench")
    vllm_exe = _bench_exe_tokens(vllm_cmd)
    # Agent-friendly fallback: if `vllm` isn't on PATH, run via `python -m ...` in
    # the current interpreter env so CLI mode still works under managed shells.
    if (
        vllm_exe
        and len(vllm_exe) >= 1
        and vllm_exe[0] == "vllm"
        and shutil.which("vllm") is None
    ):
        vllm_exe = [sys.executable, "-m", "vllm.entrypoints.cli.main"]

    extra_args = _maybe_list_str(bench, "extra_args")

    baseline_env = bench.get("baseline_env", {})
    opt_env = bench.get("opt_env", {})
    if not isinstance(baseline_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in baseline_env.items()):
        raise SystemExit("bench.baseline_env must be a dict[str,str]")
    if not isinstance(opt_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in opt_env.items()):
        raise SystemExit("bench.opt_env must be a dict[str,str]")

    # Validate env dict keys and values are not placeholders.
    for env_name, env_dict in [("baseline_env", baseline_env), ("opt_env", opt_env)]:
        for k, v in env_dict.items():
            if _is_placeholder(k):
                raise SystemExit(
                    f"bench.{env_name} key is still a placeholder: {k!r}. "
                    "Replace it with the actual environment variable name (e.g., VLLM_MY_OPT=1)."
                )
            if _is_placeholder(v):
                raise SystemExit(f"bench.{env_name}[{k!r}] value is still a placeholder: {v!r}")

    baseline_label = bench.get("baseline_label", "baseline")
    opt_label = bench.get("opt_label", "opt")

    # --baseline-from imports by literal filename, so the comparator identity is
    # the filename prefix, not the directory. --baseline-from-arm selects which
    # arm of that directory is read; 'baseline' (default) keeps baseline_label.
    baseline_from_arm = getattr(args, "baseline_from_arm", "baseline") or "baseline"
    import_label = opt_label if baseline_from_arm == "opt" else baseline_label

    fpe = bench.get("fastpath_evidence", {})
    if not isinstance(fpe, dict):
        fpe = {}

    def _read_evidence(label: str) -> Tuple[List[str], List[str]]:
        cfg = fpe.get(label, {})
        if not isinstance(cfg, dict):
            return ([], [])
        req = cfg.get("require_patterns", [])
        forb = cfg.get("forbid_patterns", [])
        if not isinstance(req, list) or not all(isinstance(x, str) for x in req):
            raise SystemExit(f"bench.fastpath_evidence.{label}.require_patterns must be list[str]")
        if not isinstance(forb, list) or not all(isinstance(x, str) for x in forb):
            raise SystemExit(f"bench.fastpath_evidence.{label}.forbid_patterns must be list[str]")
        return (req, forb)

    baseline_req, baseline_forb = _read_evidence("baseline")
    opt_req, opt_forb = _read_evidence("opt")

    baseline_extra_args = _maybe_list_str(bench, "baseline_extra_args")
    opt_extra_args = _maybe_list_str(bench, "opt_extra_args")

    # DP/EP resolver (spec §4.2): parse effective per-label parallelism flags,
    # enforce agreement, reject incompatible distributed-executor-backend, gate
    # beam-search under DP>1, and auto-inject --distributed-executor-backend
    # external_launcher into the shared extra_args when DP>1 and neither label
    # set a backend explicitly.
    extra_args, parallelism = _resolve_parallelism_and_backend(
        extra_args, baseline_extra_args, opt_extra_args
    )
    dp = parallelism["dp"]
    nproc = parallelism["nproc"]

    baseline_args = extra_args + baseline_extra_args
    opt_args = extra_args + opt_extra_args

    # Output directory:
    # - Parent run: resolve out_name via --round/--slot (v2) or --out-name (legacy),
    #   then choose/create and archive existing outputs unless --overwrite.
    # - Child run: parent passes the resolved out_root explicitly via --_out-root.
    if args._out_root:
        out_root = Path(args._out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        resolved_out_name = args.out_name  # not used by child after v2; retained for metadata
    else:
        resolved_out_name = _resolve_sweep_out_name(args, artifact_dir)
        out_root = _prepare_out_root(
            artifact_dir=artifact_dir,
            out_name=resolved_out_name,
            overwrite=args.overwrite,
        )

    logs_dir = out_root / "logs"
    json_dir = out_root / "json"
    status_dir = out_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = out_root / "e2e_latency_results.json"
    out_md_path = out_root / "e2e_latency_results.md"

    # Prepare envs: start from current env, then strip stale VLLM_* vars.
    # Cross-track contamination fix: prior-round gating flags (VLLM_OP001,
    # VLLM_MOE_TRITON_ROUTER, etc.) may be set in the inherited environment.
    # Stripping them ensures each sweep measures only the target track's
    # optimization. The target track's env vars are re-added explicitly via
    # baseline_env / opt_env from target.json. After Track A6 baseline
    # promotion, shipped opt_env keys live in baseline_env — preserve_keys
    # ensures those promoted keys survive sanitization.
    preserve_for_baseline = set(baseline_env.keys())
    preserve_for_opt = preserve_for_baseline | set(opt_env.keys())
    base_env_baseline = _sanitize_vllm_op_env(dict(os.environ), preserve_keys=preserve_for_baseline)
    base_env_opt = _sanitize_vllm_op_env(dict(os.environ), preserve_keys=preserve_for_opt)

    baseline_run = RunSpec(
        label=baseline_label,
        env={**base_env_baseline, **baseline_env},
        require_patterns=baseline_req,
        forbid_patterns=baseline_forb,
    )
    opt_run = RunSpec(
        label=opt_label,
        env={**base_env_opt, **baseline_env, **opt_env},
        require_patterns=opt_req,
        forbid_patterns=opt_forb,
    )

    # Config identity check: warn or fail when baseline and opt are identical.
    if "baseline" in selected_labels and "opt" in selected_labels:
        if not opt_env and opt_extra_args == baseline_extra_args:
            raise SystemExit(
                "baseline and opt configs are identical — opt would produce the "
                "same results as baseline.\n"
                "Update opt_env in target.json, or use --labels baseline to skip opt."
            )

    # Validate all buckets fit within max_model_len at plan time.
    _validate_buckets_model_len(buckets, max_model_len)
    # Spec-decode-aware nsys capture: positions_per_step = 1 + num_speculative_tokens. The
    # parent's expected-bucket set (used for repeat:N and the positional rename) must carry the
    # SAME output_len=window*ppt the child profiles, so we parse ppt from both labels' effective
    # args and require agreement — the single child-wide window + one repeat:N architecture
    # cannot represent two different ppt values. For the target workload the spec-config lives in
    # the shared extra_args and the label-local lists are empty, so both equal 5 (no-op guard).
    _ppt_baseline = _parse_positions_per_step(baseline_args)
    _ppt_opt = _parse_positions_per_step(opt_args)
    if _ppt_baseline != _ppt_opt:
        raise SystemExit(
            "baseline/opt spec-decode positions_per_step differ "
            f"({_ppt_baseline} vs {_ppt_opt}); selected-step nsys capture needs identical "
            "spec config across labels (single child-wide window + one repeat:N)."
        )
    _parent_ppt = _ppt_baseline
    nsys_profile_buckets = _expand_nsys_profile_buckets(
        buckets,
        nsys_output_len=args.nsys_output_len,
        nsys_capture_window_output_len=args.nsys_capture_window_output_len,
        nsys_capture_output_steps=args.nsys_capture_output_steps,
        positions_per_step=_parent_ppt,
    )
    if args.nsys_profile:
        _validate_buckets_model_len(nsys_profile_buckets, max_model_len)
        nsys_filtered_buckets = [
            b for b in nsys_profile_buckets
            if not _should_skip_bucket(int(b["batch_size"]), dp)
        ]
        nsys_expected_buckets, nsys_deduped_buckets = _dedupe_nsys_profile_buckets(
            nsys_filtered_buckets
        )
        if not nsys_expected_buckets:
            raise SystemExit(
                "No nsys profile buckets remain after DP skip filtering. "
                "Increase batch_sizes or reduce --data-parallel-size."
            )
        if nsys_deduped_buckets:
            print(
                "WARNING: deduped "
                f"{len(nsys_deduped_buckets)} duplicate nsys profiling buckets "
                "after expansion/DP filtering; identical effective shapes are "
                "profiled once: "
                f"{_format_nsys_deduped_buckets(nsys_deduped_buckets)}",
                file=sys.stderr,
            )
    else:
        nsys_expected_buckets = []

    print("=== Target ===")
    print(f"artifact_dir: {artifact_dir}")
    print(f"out_dir: {out_root}")
    print(f"model_id: {model_id}")
    print(f"tp: {tp}, ep: {ep}, max_model_len: {max_model_len}")
    if has_matrix:
        print(f"workload: {len(buckets)} buckets (workload_matrix), num_iters={num_iters}")
        for i, b in enumerate(buckets):
            print(f"  [{i}] input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}")
    else:
        print(f"workload: input_len={input_len}, output_len={output_len}, num_iters={num_iters}")
        print(f"batch_sizes: {batch_sizes}")
    print(f"baseline_label: {baseline_label}, opt_label: {opt_label}")
    if args.nsys_profile:
        nsys_ol_eff = args.nsys_output_len if args.nsys_output_len is not None else "(workload)"
        nsys_window_eff = (
            args.nsys_capture_window_output_len
            if args.nsys_capture_window_output_len is not None
            else ("2 (auto)" if args.nsys_capture_output_steps else "(none)")
        )
        nsys_ni_eff = args.nsys_num_iters
        if nsys_ni_eff is None and (args.nsys_output_len is not None or args.nsys_capture_output_steps is not None):
            nsys_ni_eff = "1 (auto)"
        elif nsys_ni_eff is None:
            nsys_ni_eff = "(workload)"
        print(
            f"nsys_profile: enabled, nsys_output_len={nsys_ol_eff}, "
            f"nsys_num_iters={nsys_ni_eff}, "
            f"nsys_capture_window_output_len={nsys_window_eff}, "
            f"nsys_capture_output_steps={args.nsys_capture_output_steps or '(none)'}"
        )
        print(f"nsys_trace: {args.nsys_trace} ({nsys_trace_source})")
        if args.nsys_capture_output_steps:
            resolved_by_horizon = {
                int(b["nsys_capture_target_output_len"]): _parse_nsys_capture_output_steps(
                    args.nsys_capture_output_steps,
                    int(b["nsys_capture_target_output_len"]),
                )
                for b in nsys_profile_buckets
                if "nsys_capture_target_output_len" in b
            }
            print(f"nsys_capture_output_steps_resolved: {resolved_by_horizon}")
        print(
            "WARNING: nsys profiling adds overhead — latency results from this run "
            "should not be used for performance comparison.",
            file=sys.stderr,
        )
        if dp > 1:
            print(
                f"WARNING: nsys + DP>1 (dp={dp}) — nsys wraps torchrun and will emit "
                f"per-rank profile files. Report files are rank-0 only. Expect larger "
                f"profile size and higher runtime overhead.",
                file=sys.stderr,
            )
    if args.torch_profile:
        print("torch_profile: enabled (manual per-bucket traces in torch_profile/)")
        if not args.nsys_profile:
            print(
                "WARNING: torch profiler adds overhead — latency results from this run "
                "should not be used for performance comparison.",
                file=sys.stderr,
            )

    if ep != 1:
        print(
            "WARNING: target.ep != 1, but this script does not pass an explicit EP flag to "
            "`vllm bench latency`. Ensure your EP configuration is applied via bench.extra_args "
            "and/or environment, and that baseline/opt runs are truly production-parity.",
            file=sys.stderr,
        )

    all_runs: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_json": str(target_path),
        "out_dir": str(out_root),
        "out_name": args.out_name,
        "execution_mode": "inproc_sweep",
        "vllm_version": _vllm_version(),
        "model_id": model_id,
        "tp": tp,
        "ep": ep,
        "max_model_len": max_model_len,
        "workload": {
            "input_len": input_len,
            "output_len": output_len,
            "num_iters": num_iters,
            "batch_sizes": batch_sizes,
            "buckets": buckets,
        },
        "bench": {
            "vllm_exe": vllm_exe,
            "extra_args": extra_args,
            "baseline_extra_args": baseline_extra_args,
            "opt_extra_args": opt_extra_args,
            "baseline_env": baseline_env,
            "opt_env": opt_env,
            "baseline_label": baseline_label,
            "opt_label": opt_label,
            "fastpath_evidence": fpe,
        },
        "results": [],
    }

    # Track baseline source and warnings for gate runs (--labels opt).
    run_warnings: List[str] = []
    if "opt" in selected_labels and "baseline" not in selected_labels:
        if baseline_from:
            all_runs["baseline_source"] = str(baseline_from)
            if baseline_from_arm != "baseline":
                # Additive and default-absent: a non-default comparator arm must
                # be visible in the artifact, not only in the run log.
                all_runs["baseline_source_arm"] = baseline_from_arm
        else:
            all_runs["baseline_source"] = "none"
            run_warnings.append(
                "No baseline reference provided (--baseline-from not set); "
                "speedup calculations unavailable."
            )
            print(
                "WARNING: --labels opt without --baseline-from; baseline data "
                "will be null in results. Pass --baseline-from <stage1_dir> for "
                "gate results with speedup calculations.",
                file=sys.stderr,
            )

    # A non-default comparator arm acts only inside the import path below, which
    # needs --baseline-from together with --labels opt and no baseline label.
    # Every other invocation drops the flag, so disclose the drop instead of
    # letting the caller believe the promoted arm was the comparator.
    if baseline_from_arm != "baseline" and not (
        baseline_from and "opt" in selected_labels and "baseline" not in selected_labels
    ):
        msg = (
            f"--baseline-from-arm {baseline_from_arm} was IGNORED: the flag only "
            "selects which arm of --baseline-from is imported, and this run "
            "imports no comparator (import requires --baseline-from with "
            "--labels opt and no baseline label). The comparator arm is "
            "unchanged."
        )
        run_warnings.append(msg)
        print("WARNING: " + msg, file=sys.stderr)

    # GPU reservation is now managed by PreToolUse/PostToolUse hooks.
    # The hooks auto-reserve when CUDA_VISIBLE_DEVICES=X is in the command
    # and auto-release when the command completes. No in-script locking needed.
    #
    # DP note (spec §4.7): the PostToolUse/Bash hook fires once per tool
    # invocation — for a DP>1 sweep that's the single `torch.distributed.run`
    # launch, not the per-rank child processes. Rank forks happen inside
    # torchrun and do NOT re-enter the hook. The hook's lease timer therefore
    # must cover the full sweep duration (child_timeout, including the
    # 60*dp padding for NCCL init / torchrun rendezvous added in Task 5).
    # AMMO_GPU_RES_DIR is session-scoped; per-rank workers inherit it via
    # environment but do not mutate reservation state.

    is_child = bool(args._child_label)

    # Record the per-bucket plan deterministically (for stable paths + repro commands).
    planned_buckets = nsys_expected_buckets if args.nsys_profile else buckets
    planned: List[Dict[str, Any]] = []
    for bucket in planned_buckets:
        b_input_len = bucket["input_len"]
        b_output_len = bucket["output_len"]
        bs = bucket["batch_size"]
        tag = _bucket_file_tag(bucket, planned_buckets)

        baseline_json_p = json_dir / f"{baseline_label}_{tag}.json"
        opt_json_p = json_dir / f"{opt_label}_{tag}.json"
        baseline_log_p = logs_dir / f"{baseline_label}_{tag}.log"
        opt_log_p = logs_dir / f"{opt_label}_{tag}.log"

        baseline_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=b_input_len,
            output_len=b_output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=baseline_json_p,
            extra_args=baseline_args,
        )
        opt_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=b_input_len,
            output_len=b_output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=opt_json_p,
            extra_args=opt_args,
        )

        print(f"\n=== {tag} (il={b_input_len}, ol={b_output_len}, bs={bs}) ===")
        print(f"Baseline cmd: {_format_cmd_for_md(baseline_cmd, baseline_env)}")
        print(f"Opt cmd: {_format_cmd_for_md(opt_cmd, {**baseline_env, **opt_env})}")

        planned.append({
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
            "tag": tag,
            "baseline_cmd": baseline_cmd,
            "opt_cmd": opt_cmd,
            "baseline_log": baseline_log_p,
            "opt_log": opt_log_p,
            "baseline_json": baseline_json_p,
            "opt_json": opt_json_p,
        })

        # Populate a minimal row deterministically (stable paths + repro commands).
        all_runs["results"].append({
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
            baseline_label: {
                "cmd": baseline_cmd,
                "env_overrides": baseline_env,
                "log": str(baseline_log_p.relative_to(out_root)),
                "output_json": str(baseline_json_p.relative_to(out_root)),
                "fastpath_evidence": {"status": "unknown"},
            },
            opt_label: {
                "cmd": opt_cmd,
                "env_overrides": {**baseline_env, **opt_env},
                "log": str(opt_log_p.relative_to(out_root)),
                "output_json": str(opt_json_p.relative_to(out_root)),
                "fastpath_evidence": {"status": "unknown"},
            },
        })

    # Ensure per-bucket log files exist early (so tailing works even if model load is slow).
    for p in planned:
        for log_path in (Path(p["baseline_log"]), Path(p["opt_log"])):
            if not log_path.exists():
                _write_text(
                    log_path,
                    "PENDING: benchmark has not started for this bucket yet.\n"
                    "If this stays unchanged for a long time, check logs/*_child.log and status/*.json.\n",
                )

    # Seed status files so supervisors can poll immediately.
    # Rank-0 gated: under torchrun (DP>1), all ranks execute this path but
    # only rank 0 owns the status tree. Without the gate, ranks race on the
    # .tmp→final rename in _write_json_atomic and one gets FileNotFoundError
    # from os.replace(). Matches the rank-0 guard pattern used elsewhere in
    # this file (e.g., lines 406, 418, 1238, 1845).
    for label in [l for l in (baseline_label, opt_label)
                  if ("baseline" if l == baseline_label else "opt") in selected_labels]:
        st = status_dir / f"{label}.json"
        if _IS_RANK0 and not st.exists():
            _write_json_atomic(
                st,
                {
                    "label": label,
                    "phase": "queued",
                    "batch_size": None,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "last_update": datetime.now(timezone.utc).isoformat(),
                },
            )

    if args._child_label:
        # Child mode: just run a single label sweep into the provided out_root.
        label = args._child_label
        if label not in (baseline_label, opt_label):
            raise SystemExit(f"Invalid --_child-label: {label!r} (expected {baseline_label!r} or {opt_label!r})")
        label_args = baseline_args if label == baseline_label else opt_args
        # Inject --cudagraph-capture-sizes if the parent passed it (nsys mode).
        if args._cudagraph_capture_sizes:
            label_args = (
                label_args
                + ["--cudagraph-capture-sizes"]
                + [str(x) for x in args._cudagraph_capture_sizes]
            )
        # Apply nsys overrides when present (decouples profiling OL from benchmark OL).
        child_buckets = buckets
        child_num_iters = num_iters
        nsys_ol = getattr(args, "_nsys_output_len", None)
        nsys_ni = getattr(args, "_nsys_num_iters", None)
        nsys_steps = getattr(args, "_nsys_capture_output_steps", None)
        nsys_window = getattr(args, "_nsys_capture_window_output_len", None)
        # Size the actually-profiled output_len for THIS label's spec-decode config so the
        # short generation runs enough worker-steps to arm the CUDA profiler (output_len =
        # window * positions_per_step). label_args == baseline_args/opt_args selected above.
        child_ppt = _parse_positions_per_step(label_args)
        child_buckets = _expand_nsys_profile_buckets(
            buckets,
            nsys_output_len=nsys_ol,
            nsys_capture_window_output_len=nsys_window,
            nsys_capture_output_steps=nsys_steps,
            positions_per_step=child_ppt,
        )
        if getattr(args, "_nsys_profile", False):
            child_buckets, child_deduped_buckets = _dedupe_nsys_profile_buckets(child_buckets)
            if child_deduped_buckets and _IS_RANK0:
                print(
                    "WARNING: child deduped "
                    f"{len(child_deduped_buckets)} duplicate nsys profiling buckets; "
                    "identical effective shapes are profiled once.",
                    file=sys.stderr,
                )
        if nsys_ni is not None:
            child_num_iters = nsys_ni
        returncode = _run_inproc_latency_sweep_child(
            label=label,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            buckets=child_buckets,
            num_iters=child_num_iters,
            extra_args=label_args,
            out_root=out_root,
            timeout_s_per_bucket=args.timeout_s,
            nsys_profile=getattr(args, "_nsys_profile", False),
            torch_profile=getattr(args, "_torch_profile", False),
            capture_golden_refs=getattr(args, "_capture_golden_refs", False),
            verify_correctness=getattr(args, "_verify_correctness", False),
            correctness_num_questions=getattr(args, "_correctness_num_questions", 1319),
            correctness_tolerance_pct=getattr(args, "_correctness_tolerance_pct", 1.0),
            skip_tags=frozenset(
                t for t in (getattr(args, "_skip_tags", None) or "").split(",") if t
            ),
            dummy_prompt_source=getattr(args, "_dummy_prompt_source", "gsm8k"),
        )
        raise SystemExit(returncode)

    # Summaries
    def _status_from_evidence(ev: Dict[str, Any], patterns_configured: bool) -> str:
        if not patterns_configured:
            return "unknown"
        return "pass" if ev.get("ok") else "fail"

    baseline_patterns_configured = bool(baseline_run.require_patterns or baseline_run.forbid_patterns)
    opt_patterns_configured = bool(opt_run.require_patterns or opt_run.forbid_patterns)

    script_path = Path(__file__).resolve()
    child_timeout = int(args.timeout_s) * max(1, len(buckets)) + 1800
    if args.nsys_profile:
        child_timeout = args.nsys_timeout_s * max(1, len(nsys_expected_buckets))
    # DP>1 adds torchrun rendezvous + per-rank model-load overhead; pad the
    # supervisor timeout linearly with DP so warm-ups do not trip spurious kills.
    if dp > 1:
        child_timeout += 60 * dp

    # Set up nsys output directory if profiling.
    # v2: traces are written to rounds/{N}/profiling/nsys/ (sibling to sweep
    # output) so multiple sweep slots in the same round share a single trace
    # repository rather than nesting nsys/ inside each sweep.
    nsys_dir: Optional[Path] = None
    nsys_capture_sizes: Optional[List[int]] = None
    if args.nsys_profile:
        nsys_dir = _v2_profiling_dir(out_root, "nsys")
        nsys_dir.mkdir(parents=True, exist_ok=True)
        # Restrict CUDA graph capture to only the profiled nominal/effective sizes.
        # Default vLLM captures ~50 sizes (~2,142 CUDAGraph objects); restricting
        # to workload shapes reduces this to a small bounded list, mitigating the
        # memory pressure that causes nsys --cuda-graph-trace=node replay hangs.
        nsys_capture_sizes = _compute_nsys_cudagraph_capture_sizes(nsys_expected_buckets)
        print(
            f"nsys mode: restricting cudagraph_capture_sizes to {nsys_capture_sizes} "
            f"(from workload batch_sizes/effective decode token counts)"
        )

    runs_to_execute = []
    if "baseline" in selected_labels:
        runs_to_execute.append(baseline_run)
    if "opt" in selected_labels:
        runs_to_execute.append(opt_run)

    # Copy golden refs BEFORE spawning children — children need it during Phase 1.
    if baseline_from and args.verify_correctness:
        golden_src = baseline_from / "json" / "golden_refs.json"
        golden_dst = json_dir / "golden_refs.json"
        if golden_src.exists():
            shutil.copy2(str(golden_src), str(golden_dst))
            print(f"Imported golden references from {golden_src}")
        else:
            raise SystemExit(
                f"--verify-correctness requires golden_refs.json but "
                f"--baseline-from has none at {golden_src}"
            )

    # --fresh-cache: allocate a per-sweep cache root under out_root/cache/{sweep_id}
    # that will be removed at end of a successful sweep. We pick a stable
    # sweep_id derived from the out_root basename + a timestamp so concurrent
    # sweeps don't collide (tests call the injector directly with a path).
    sweep_cache_root: Optional[Path] = None
    if args.fresh_cache:
        sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sweep_cache_root = out_root / "cache" / sweep_id
        sweep_cache_root.mkdir(parents=True, exist_ok=True)
        (sweep_cache_root / "triton_cache").mkdir(parents=True, exist_ok=True)
        print(
            f"[fresh-cache] isolating compile caches at {sweep_cache_root} "
            "(first launch pays full compile; cache is removed after success)"
        )

    # Hang watchdog plumbing (default OFF). bucket_meta lets the watchdog name the
    # wedged bucket from the child's (il,ol,bs) status; relaunch_budget caps total
    # kill+relaunch cycles across the whole sweep; hung_tags accumulates buckets we
    # gave up on so we never re-trigger the same hang on a later label.
    hang_watchdog = bool(getattr(args, "hang_watchdog", False))
    bucket_meta: Dict[str, Dict[str, int]] = {
        p["tag"]: {
            "input_len": int(p["input_len"]),
            "output_len": int(p["output_len"]),
            "batch_size": int(p["batch_size"]),
        }
        for p in planned
    }
    relaunch_budget = [max(0, int(getattr(args, "hang_max_relaunch", 8)))]
    hung_tags: set = set()

    for run in runs_to_execute:
        extra_child_flags: List[str] = []
        if args.nsys_profile:
            extra_child_flags.append("--_nsys-profile")
            if nsys_capture_sizes:
                extra_child_flags.extend(
                    ["--_cudagraph-capture-sizes"] + [str(bs) for bs in nsys_capture_sizes]
                )
            # Pass nsys overrides to child (decouples profiling OL from benchmark OL).
            nsys_ol = args.nsys_output_len
            nsys_ni = args.nsys_num_iters
            if nsys_ni is None and (nsys_ol is not None or args.nsys_capture_output_steps is not None):
                nsys_ni = 1  # Default to 1 iter when profiling OL is decoupled.
            if nsys_ol is not None:
                extra_child_flags.extend(["--_nsys-output-len", str(nsys_ol)])
            if nsys_ni is not None:
                extra_child_flags.extend(["--_nsys-num-iters", str(nsys_ni)])
            if args.nsys_capture_output_steps is not None:
                extra_child_flags.extend([
                    "--_nsys-capture-output-steps",
                    args.nsys_capture_output_steps,
                ])
            if args.nsys_capture_window_output_len is not None:
                extra_child_flags.extend([
                    "--_nsys-capture-window-output-len",
                    str(args.nsys_capture_window_output_len),
                ])
        if args.torch_profile:
            extra_child_flags.append("--_torch-profile")

        # Forward correctness flags to child.
        if args.capture_golden_refs:
            extra_child_flags.append("--_capture-golden-refs")
        if args.verify_correctness:
            extra_child_flags.append("--_verify-correctness")
        extra_child_flags.extend(_build_correctness_child_flags(
            capture=args.capture_golden_refs,
            verify=args.verify_correctness,
            num_questions=args.correctness_num_questions,
            tolerance_pct=args.correctness_tolerance_pct,
        ))

        # Forward the timing prompt source to the child (public -> hidden).
        extra_child_flags.extend(["--_dummy-prompt-source", args.dummy_prompt_source])

        child_cmd = _build_child_cmd(
            python_exe=sys.executable,
            script_path=script_path,
            run_label=run.label,
            artifact_dir=artifact_dir,
            target_path=target_path,
            timeout_s=args.timeout_s,
            out_name=args.out_name,
            out_root=out_root,
            dp=dp,
            nproc=nproc,
            extra_child_flags=extra_child_flags,
        )

        # Prepend nsys wrapper when profiling.
        #
        # Under DP>1, nsys wraps torchrun at the outermost layer and follows
        # forked rank processes via --trace-fork-before-exec=true (set
        # below). The child uses rank-0-only cudaProfilerStart/Stop; because
        # --capture-range=cudaProfilerApi is an nsys-level mechanism, a
        # single rank's cudaProfilerStart() triggers capture across every
        # traced process in the world group. Precondition: validate
        # capture-range propagation across torchrun siblings with a DP=2
        # sanity run before first production DP profiling (spec §4.5).
        child_env = dict(run.env) if run.env else dict(os.environ)
        child_env = _ensure_worktree_pythonpath(child_env)
        # --fresh-cache: inject per-sweep VLLM_CACHE_ROOT/TRITON_CACHE_DIR
        # AFTER _ensure_worktree_pythonpath so both the run.env and
        # os.environ branches receive identical cache overrides.
        if sweep_cache_root is not None:
            child_env = _inject_fresh_cache_env(child_env, sweep_cache_root)
        if args.nsys_profile:
            child_env = _inject_nsys_execute_timeout_env(
                child_env, args.nsys_timeout_s
            )
            assert nsys_dir is not None
            # fork multiproc semantics conflict with torch.distributed.run's
            # own bootstrapping; only force 'spawn' in the single-process path.
            if dp == 1:
                child_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            nsys_prefix = [
                "nsys", "profile",
                f"--trace={args.nsys_trace},nvtx",
                "--sample=none",
                "--capture-range=cudaProfilerApi",
                f"--capture-range-end=repeat:{len(nsys_expected_buckets)}",
                f"--cuda-graph-trace={args.nsys_mode}",
                "--trace-fork-before-exec=true",
                "--force-overwrite=true",
                "-o", str(nsys_dir / f"{run.label}_profile"),
            ]
            if args.nsys_extra_flags:
                import shlex as _shlex
                nsys_prefix.extend(_shlex.split(args.nsys_extra_flags))
            # nsys wraps the outermost process (torchrun under DP>1, python
            # under DP=1). --trace-fork-before-exec ensures child ranks are
            # captured; per-rank .nsys-rep files are emitted.
            child_cmd = nsys_prefix + child_cmd

        print(f"\n=== Running inproc sweep for {run.label} ===")
        print(f"Child cmd: {_format_cmd_for_md(child_cmd, {})}")
        child_log = logs_dir / f"{run.label}_supervisor.log"
        child_status = status_dir / f"{run.label}.json"

        if not hang_watchdog:
            # ---- Legacy path: single streamed run; fail-fast on non-zero. ----
            child_res = _run_cmd_streaming(
                child_cmd,
                env=child_env,
                cwd=None,
                timeout_s=child_timeout,
                log_path=child_log,
                heartbeat_s=int(args.heartbeat_s),
                status_path=child_status,
            )
            if not child_res.get("ok"):
                hint = ""
                if args.nsys_profile and child_res.get("returncode") in (-9, 137, None):
                    hint = (
                        " This is likely an nsys --cuda-graph-trace=node replay hang. "
                        "Try --nsys-capture-output-steps 2,50%,100% "
                        "--nsys-num-iters 1 for a bounded selected-step "
                        f"node-mode capture, use --nsys-trace cuda-sw (this run "
                        f"used {args.nsys_trace}), switch to --nsys-mode graph, "
                        "reduce workload batch_sizes, or increase --nsys-timeout-s "
                        f"(current: {args.nsys_timeout_s}s per bucket)."
                    )
                raise SystemExit(
                    f"Inproc sweep child failed for {run.label}: "
                    f"returncode={child_res.get('returncode')}. "
                    f"See {child_log} and {logs_dir / f'{run.label}_child.log'}"
                    f"{hint}"
                )
        else:
            # ---- Hang-watchdog path: detect an in-engine freeze, kill the
            # child's process group, and relaunch skipping the buckets already
            # done on disk PLUS the wedged one. Completed buckets are already
            # persisted (json/{label}_{tag}.json), so partial results survive
            # a hang and the sweep reports what ran. Buckets carried in
            # hung_tags (gave up earlier) are skipped from the first attempt. ----
            this_label_json = (lambda tag: json_dir / f"{run.label}_{tag}.json")
            attempt = 0
            while True:
                attempt += 1
                # Done = buckets with a valid (non-error, non-skipped) raw JSON.
                done_tags: set = set()
                for tag in bucket_meta:
                    jp = this_label_json(tag)
                    if jp.exists():
                        try:
                            raw = _read_json(jp)
                        except Exception:
                            raw = {}
                        if isinstance(raw, dict) and "avg_latency" in raw:
                            done_tags.add(tag)
                skip = done_tags | hung_tags
                attempt_cmd = list(child_cmd)
                if skip:
                    attempt_cmd += ["--_skip-tags", ",".join(sorted(skip))]
                if attempt > 1 or skip:
                    print(
                        f"[hang-watchdog] {run.label}: attempt {attempt}; "
                        f"skipping {len(skip)} bucket(s) already done/wedged "
                        f"({sorted(skip) if skip else 'none'})",
                        flush=True,
                    )
                child_res = _run_cmd_streaming_watchdog(
                    attempt_cmd,
                    env=child_env,
                    cwd=None,
                    timeout_s=child_timeout,
                    log_path=child_log,
                    heartbeat_s=int(args.heartbeat_s),
                    status_path=child_status,
                    bucket_meta=bucket_meta,
                    base_stale_s=int(getattr(args, "hang_stale_warmup_s", 130)),
                )
                if child_res.get("ok"):
                    break  # child finished all remaining buckets cleanly
                if child_res.get("hung"):
                    wedged = child_res.get("wedged_tag") or ""
                    if wedged:
                        hung_tags.add(wedged)
                    if relaunch_budget[0] <= 0:
                        print(
                            f"[hang-watchdog] {run.label}: relaunch budget exhausted "
                            f"after wedged bucket '{wedged}'. Reporting partial results "
                            "for the buckets that completed.",
                            flush=True,
                        )
                        break
                    relaunch_budget[0] -= 1
                    print(
                        f"[hang-watchdog] {run.label}: bucket '{wedged or '?'}' wedged the "
                        f"engine; relaunching (budget left={relaunch_budget[0]}).",
                        flush=True,
                    )
                    continue
                # Genuine non-hung failure (crash/OOM/non-zero exit): preserve
                # the legacy fail-fast contract.
                raise SystemExit(
                    f"Inproc sweep child failed for {run.label}: "
                    f"returncode={child_res.get('returncode')}. "
                    f"See {child_log} and {logs_dir / f'{run.label}_child.log'}"
                )

        # Rename nsys output files to match bucket tags.
        if nsys_dir is not None:
            # This label's spec-decode positions_per_step (used as the fallback when a
            # bucket predates the per-bucket stamp). label_args mirrors the child dispatch.
            label_args = baseline_args if run.label == baseline_label else opt_args
            run_ppt = _parse_positions_per_step(label_args)
            nsys_tag_buckets = nsys_expected_buckets
            renamed = 0
            renamed_selected_step: List[Tuple[Path, Dict[str, int]]] = []
            for i, bucket in enumerate(nsys_tag_buckets, 1):
                src = nsys_dir / f"{run.label}_profile.{i}.nsys-rep"
                tag = _bucket_file_tag(bucket, nsys_tag_buckets)
                dst = nsys_dir / f"{run.label}_{tag}.nsys-rep"
                if src.exists():
                    src.rename(dst)
                    print(f"  nsys: {src.name} -> {dst.name}")
                    renamed += 1
                    if _is_nsys_selected_step_bucket(bucket):
                        renamed_selected_step.append((dst, bucket))
            if renamed < len(nsys_tag_buckets):
                # Selected-step reps are renamed POSITIONALLY (profile.{i} -> bucket[i]).
                # If the CUDA profiler failed to arm for some buckets (the spec-decode
                # never-arm bug: a generation too short to reach delay_iterations), the
                # surviving reps would be silently mislabeled (e.g. an armed bs32 step
                # renamed as bs1 — the exact root cause of the failed artifacts). After the
                # spec-aware sizing every selected-step bucket arms, so a shortfall now
                # means a real capture failure: fail loud rather than mine mislabeled reps.
                if any(_is_nsys_selected_step_bucket(b) for b in nsys_tag_buckets):
                    raise SystemExit(
                        f"ERROR: nsys produced {renamed} of {len(nsys_tag_buckets)} "
                        f"selected-step reps for {run.label}; a capture failed to arm. "
                        "Selected-step reps are renamed POSITIONALLY so a partial emission "
                        "would silently mislabel captures (e.g. an armed bs32 step renamed "
                        "as bs1). Re-profile; do not mine these traces."
                    )
                print(
                    f"  WARNING: nsys produced {renamed} of {len(nsys_tag_buckets)} "
                    f"expected profile files for {run.label}"
                )
            # Guard each selected-step capture: export to sqlite and assert it is a
            # genuine steady-state decode step (NOT a chunked-prefill chunk). This is
            # the wired backstop for the prefill-contamination class — mining must never
            # consume a prefill-shaped capture as 'decode-graph %'. Fails loud.
            for dst, bucket in renamed_selected_step:
                sqlite_path = _export_nsys_sqlite(dst)
                if sqlite_path is None:
                    print(
                        f"  WARNING: could not export {dst.name} to sqlite for the "
                        "decode-shape guard; skipping the automatic check. Run "
                        "assert_decode_shaped_capture manually before mining.",
                        file=sys.stderr,
                    )
                    continue
                # Prefer the per-bucket stamped ppt (matches what was profiled); fall back
                # to the label-derived value for buckets that predate the stamp.
                bucket_ppt = int(
                    bucket.get("nsys_capture_positions_per_step", run_ppt)
                )
                assert_decode_shaped_capture(
                    str(sqlite_path),
                    int(bucket["batch_size"]),
                    positions_per_step=bucket_ppt,
                )
                print(
                    f"  decode-shape guard PASSED for {dst.name} "
                    f"(bs={bucket['batch_size']}, step={bucket['nsys_capture_output_step']}, "
                    f"positions_per_step={bucket_ppt})"
                )

    # Import Stage 1 baseline artifacts when --baseline-from is set.
    # This copies baseline JSON + runner JSON into the gate run's json/ dir
    # so the results collection below picks them up transparently.
    if baseline_from and "opt" in selected_labels and "baseline" not in selected_labels:
        baseline_from_json = baseline_from / "json"
        if not baseline_from_json.is_dir():
            raise SystemExit(
                f"--baseline-from has no json/ subdirectory: {baseline_from}. "
                "Expected a Stage 1 output directory with json/{baseline_label}_*.json files."
            )
        imported = 0
        missing_tags: List[str] = []
        for p in planned:
            tag = p["tag"]
            for suffix in (".json", ".runner.json"):
                # Read the selected arm's filename prefix; always write under
                # baseline_label, which is what the results collection reads.
                src = baseline_from_json / f"{import_label}_{tag}{suffix}"
                dst = json_dir / f"{baseline_label}_{tag}{suffix}"
                if src.exists():
                    import shutil as _shutil_copy
                    _shutil_copy.copy2(str(src), str(dst))
                    if suffix == ".json":
                        imported += 1
                else:
                    if suffix == ".json":
                        missing_tags.append(tag)
        if missing_tags:
            raise SystemExit(
                f"--baseline-from is missing baseline data for {len(missing_tags)} "
                f"bucket(s): {missing_tags}. "
                f"Looked in {baseline_from_json} for {import_label}_<tag>.json files."
            )
        print(
            f"Imported {imported} baseline artifact(s) from {baseline_from} "
            f"(--baseline-from-arm {baseline_from_arm}, filename prefix "
            f"{import_label!r})"
        )

        # Non-default comparator arm disclosure. The imported rows land under
        # baseline_label, so the .md table shows an ordinary "baseline" column
        # and hides the arm identity. An auditor who reads only the .md must
        # still see which arm became the comparator.
        if baseline_from_arm != "baseline":
            msg = (
                f"comparator imported from arm {baseline_from_arm!r} of "
                f"{baseline_from} (--baseline-from-arm); the "
                f"{baseline_label!r} column holds that arm's numbers, read by "
                f"filename prefix {import_label!r}."
            )
            run_warnings.append(msg)
            print("WARNING: " + msg, file=sys.stderr)

        # Cross-version A/B disclosure. The imported baseline was measured on
        # whatever vLLM was installed then; a bump between Stage 1 and this gate
        # run changes kernels AND the metrics surface. Warn, never block — a
        # deliberate cross-version comparison is legitimate but must be visible.
        imported_header = baseline_from / "e2e_latency_results.json"
        if imported_header.exists():
            try:
                _hdr = _read_json(imported_header)
            except SystemExit:
                _hdr = {}  # unreadable header: nothing to compare, not fatal
            b_ver = _hdr.get("vllm_version") if isinstance(_hdr, dict) else None
            cur_ver = all_runs.get("vllm_version")
            if isinstance(b_ver, str) and isinstance(cur_ver, str) and b_ver != cur_ver:
                msg = (
                    "CROSS-VERSION A/B: imported baseline ran on vLLM "
                    f"{b_ver}, this run uses {cur_ver}. Kernel and metrics "
                    "differences between versions are inside the measured delta."
                )
                run_warnings.append(msg)
                print("WARNING: " + msg, file=sys.stderr)

    # Populate per-bucket entries from artifacts written by children.
    new_rows: List[Dict[str, Any]] = []
    for p in planned:
        bs = int(p["batch_size"])
        b_il = int(p["input_len"])
        b_ol = int(p["output_len"])
        tag = p["tag"]
        baseline_log = Path(p["baseline_log"])
        opt_log = Path(p["opt_log"])
        baseline_json = Path(p["baseline_json"])
        opt_json = Path(p["opt_json"])

        baseline_runner_json = json_dir / f"{baseline_label}_{tag}.runner.json"
        opt_runner_json = json_dir / f"{opt_label}_{tag}.runner.json"

        baseline_text = ""
        opt_text = ""
        if baseline_log.exists():
            baseline_text = baseline_log.read_text(encoding="utf-8", errors="replace")
        if opt_log.exists():
            opt_text = opt_log.read_text(encoding="utf-8", errors="replace")

        baseline_raw = _read_json(baseline_json) if baseline_json.exists() else {}
        opt_raw = _read_json(opt_json) if opt_json.exists() else {}

        # Lift the additive spec-decode acceptance block DIRECTLY out of the
        # child raw JSON (NOT via _metrics_from_vllm_latency_json, which is
        # float-only and would silently drop this nested dict).
        baseline_spec = baseline_raw.get("spec_decode") if isinstance(baseline_raw, dict) else None
        opt_spec = opt_raw.get("spec_decode") if isinstance(opt_raw, dict) else None

        baseline_metrics = _metrics_from_vllm_latency_json(baseline_raw) or _parse_latency_metrics(baseline_text)
        opt_metrics = _metrics_from_vllm_latency_json(opt_raw) or _parse_latency_metrics(opt_text)

        baseline_evidence = _check_patterns(baseline_text, baseline_run.require_patterns, baseline_run.forbid_patterns)
        opt_evidence = _check_patterns(opt_text, opt_run.require_patterns, opt_run.forbid_patterns)

        baseline_status = _read_json(baseline_runner_json) if baseline_runner_json.exists() else {"ok": None}
        opt_status = _read_json(opt_runner_json) if opt_runner_json.exists() else {"ok": None}

        baseline_entry = _build_label_result_entry(
            cmd=p["baseline_cmd"],
            env_overrides=baseline_env,
            metrics=baseline_metrics,
            log_rel=str(baseline_log.relative_to(out_root)),
            output_json_rel=str(baseline_json.relative_to(out_root)),
            runner_json_rel=str(baseline_runner_json.relative_to(out_root)),
            ok=baseline_status.get("ok"),
            returncode=0 if baseline_status.get("ok") else 1,
            evidence_status=_status_from_evidence(baseline_evidence, baseline_patterns_configured),
            evidence=baseline_evidence,
            timing={
                "start_time": baseline_status.get("start_time"),
                "end_time": baseline_status.get("end_time"),
                "duration_s": baseline_status.get("duration_s"),
            },
            spec_decode=baseline_spec,
        )
        opt_entry = _build_label_result_entry(
            cmd=p["opt_cmd"],
            env_overrides={**baseline_env, **opt_env},
            metrics=opt_metrics,
            log_rel=str(opt_log.relative_to(out_root)),
            output_json_rel=str(opt_json.relative_to(out_root)),
            runner_json_rel=str(opt_runner_json.relative_to(out_root)),
            ok=opt_status.get("ok"),
            returncode=0 if opt_status.get("ok") else 1,
            evidence_status=_status_from_evidence(opt_evidence, opt_patterns_configured),
            evidence=opt_evidence,
            timing={
                "start_time": opt_status.get("start_time"),
                "end_time": opt_status.get("end_time"),
                "duration_s": opt_status.get("duration_s"),
            },
            spec_decode=opt_spec,
        )

        new_row: Dict[str, Any] = {
            "batch_size": bs,
            "input_len": b_il,
            "output_len": b_ol,
            baseline_label: baseline_entry,
            opt_label: opt_entry,
        }

        b_avg = baseline_entry.get("avg_s")
        o_avg = opt_entry.get("avg_s")
        if isinstance(b_avg, (int, float)) and isinstance(o_avg, (int, float)) and o_avg > 0:
            speedup = b_avg / o_avg
            improvement_pct = (b_avg - o_avg) / b_avg * 100.0 if b_avg != 0 else None
            new_row["speedup"] = speedup
            new_row["improvement_pct"] = improvement_pct

        # OTPS (output tokens/sec) + TPOT (time per output token), derived from
        # the real decode timing already on each entry (decode_avg_s). Stored on
        # each label entry; A/B deltas surfaced as row siblings. Higher OTPS and
        # lower TPOT are better, so otps_gain_pct uses opt-over-baseline and
        # tpot_improvement_pct uses baseline-over-opt (mirrors improvement_pct).
        # A tier mismatch is fail-closed (mirrors evaluate_decode_shape): emit
        # NO gain number and disclose both methods instead.
        b_tp = _compute_token_throughput(baseline_entry, b_ol, bs)
        o_tp = _compute_token_throughput(opt_entry, b_ol, bs)
        baseline_entry.update(b_tp)
        opt_entry.update(o_tp)
        tp_mismatch = _throughput_method_mismatch(b_tp, o_tp)
        if tp_mismatch is not None:
            b_method, o_method = tp_mismatch
            msg = (
                f"{tag}: throughput_method mismatch (baseline={b_method}, "
                f"opt={o_method}); otps_gain_pct and tpot_improvement_pct "
                "omitted — the two tiers use different denominators. Usual "
                "cause: a vLLM bump renamed RequestOutput.metrics fields. Run "
                "scripts/preflight_vllm_api.py."
            )
            run_warnings.append(msg)
            print("WARNING: " + msg, file=sys.stderr)
        else:
            b_otps, o_otps = b_tp.get("otps"), o_tp.get("otps")
            if isinstance(b_otps, (int, float)) and isinstance(o_otps, (int, float)) and b_otps > 0:
                new_row["otps_gain_pct"] = (o_otps - b_otps) / b_otps * 100.0
            b_tpot, o_tpot = b_tp.get("tpot_s"), o_tp.get("tpot_s")
            if isinstance(b_tpot, (int, float)) and isinstance(o_tpot, (int, float)) and b_tpot > 0:
                new_row["tpot_improvement_pct"] = (b_tpot - o_tpot) / b_tpot * 100.0

        # Within-launch Welch significance from the per-iteration latencies
        # already saved in each label's raw bucket JSON. Runs on the default
        # single-launch config — no extra launches, no model reloads.
        # generate_validation_report.py consumes row["significance"] for the
        # per-BS verdict; a non-significant delta cannot earn plain PASS.
        sig = _welch_significance(
            baseline_raw.get("latencies") if isinstance(baseline_raw, dict) else None,
            opt_raw.get("latencies") if isinstance(opt_raw, dict) else None,
        )
        if sig is not None:
            new_row["significance"] = sig

        # Phase-level Welch (prefill/decode) from the per-iteration phase-mean
        # arrays the child runner emits alongside "latencies". Reuses the exact
        # same _welch_significance math as the e2e significance above; attaches
        # only when >= 1 phase produced a result. Omitted (byte-identical to
        # legacy) on Tier-B/C or pre-array raw JSON. generate_validation_report.py
        # consumes row["phase_significance"] for the DILUTED_PASS ship gate.
        phase_sig = _row_phase_significance(baseline_raw, opt_raw)
        if phase_sig:
            new_row["phase_significance"] = phase_sig

        new_rows.append(new_row)

        if args.require_fastpath and opt_patterns_configured and not opt_evidence.get("ok"):
            raise SystemExit(
                f"Fast-path evidence FAILED for opt at {tag}. "
                f"Missing={opt_evidence.get('require_miss')}, forbidden_hits={opt_evidence.get('forbid_hits')}. "
                f"See {opt_log}"
            )

    all_runs["results"] = new_rows

    # Append warnings accumulated during the run.
    if run_warnings:
        all_runs["warnings"] = run_warnings

    # Write outputs.
    _write_json(out_json_path, all_runs)

    md_lines: List[str] = []
    md_lines.append("# E2E Latency Sweep (vllm bench latency)")
    md_lines.append("")
    md_lines.append(f"Generated: {all_runs['generated_at']} (UTC)")
    md_lines.append("")
    md_lines.append("## Workload")
    md_lines.append("")
    md_lines.append(f"- model_id: {model_id}")
    if has_matrix:
        md_lines.append(f"- buckets: {len(buckets)} (workload_matrix)")
        for i, b in enumerate(buckets):
            md_lines.append(f"  - [{i}] input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}")
    else:
        md_lines.append(f"- input_len: {input_len}, output_len: {output_len}")
    md_lines.append(f"- tp: {tp}, max_model_len: {max_model_len}")
    md_lines.append(f"- num_iters: {num_iters}")
    md_lines.append("")
    md_lines.append("## Results")
    md_lines.append("")
    md_lines.append(_render_md_table(all_runs["results"], baseline_label, opt_label))

    # Warnings reach the .md too. The JSON header alone is not read by a human
    # auditor, so a comparator-provenance or contamination warning that lands
    # only there is invisible. Section is omitted when there are no warnings,
    # keeping clean runs byte-identical to the legacy report.
    if run_warnings:
        md_lines.append("")
        md_lines.append("## Warnings")
        md_lines.append("")
        for w in run_warnings:
            md_lines.append(f"- {w}")
        md_lines.append("")

    _write_text(out_md_path, "\n".join(md_lines))

    print(f"\nWrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")

    # --fresh-cache: remove the per-sweep cache dir after a successful sweep.
    # We remove the entire out_root/cache/ tree to catch any sibling temp dirs
    # left behind by prior runs that did not finish cleanly.
    if args.fresh_cache:
        cache_parent = out_root / "cache"
        if cache_parent.exists():
            try:
                shutil.rmtree(cache_parent, ignore_errors=True)
                print(f"[fresh-cache] removed ephemeral cache dir: {cache_parent}")
            except Exception as e:  # pragma: no cover — best-effort cleanup
                print(
                    f"[fresh-cache] WARNING: failed to remove cache dir "
                    f"{cache_parent}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
