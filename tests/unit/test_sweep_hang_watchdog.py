# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the opt-in hang watchdog in run_vllm_bench_latency_sweep.py.

The watchdog exists because the intermittent in-engine GPU-kernel hang under
multi-seq chunked prefill wedges INSIDE a single ``llm.generate()`` call that
never returns. The per-bucket deadline cannot catch it (it is only checked
BETWEEN generate() calls), and the parent's legacy behavior was to `raise
SystemExit` the moment the child exited non-zero — throwing away every bucket
that DID complete (their raw JSON is already on disk).

The watchdog (default OFF, opt-in via --hang-watchdog):
1. `_hang_stale_limit_s` — phase-gated staleness threshold; widened for heavy
   buckets (long prefill / large batch) so a slow-but-healthy bucket is never
   false-killed.
2. `_kill_process_group` — SIGTERM→SIGKILL the child's whole process group
   (the child is a session leader; vLLM EngineCore subprocesses inherit the
   group, so no kill-by-name is needed).
3. `_run_cmd_streaming_watchdog` — streams output but kills the group when the
   child's status file goes stale past the threshold while in a warmup/benchmark
   phase, returning hung=True + the wedged bucket tag.
4. `_meta_tag` — reverse-lookup a bucket tag from (il, ol, bs).

These tests exercise the pure helpers + the supervisor against a FAKE child
process (a small python script) — no GPU and no vLLM import required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import time
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


def _load_sweep_module():
    cached = sys.modules.get("_run_vllm_bench_latency_sweep_hwtest")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "_run_vllm_bench_latency_sweep_hwtest", SWEEP_SCRIPT
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
# 1. _hang_stale_limit_s — phase-gated threshold, widened for heavy buckets
# ---------------------------------------------------------------------------


class TestHangStaleLimit:
    def test_default_for_light_bucket(self, sweep):
        # Small ctx, small batch, short OSL -> base threshold unchanged.
        assert sweep._hang_stale_limit_s(
            input_len=1024, output_len=128, batch_size=1, base_s=130
        ) == 130

    def test_long_prefill_widens_to_360(self, sweep):
        assert sweep._hang_stale_limit_s(
            input_len=120000, output_len=150, batch_size=1, base_s=130
        ) == 360

    def test_large_batch_widens_to_360(self, sweep):
        assert sweep._hang_stale_limit_s(
            input_len=4096, output_len=256, batch_size=64, base_s=130
        ) == 360

    def test_long_osl_adds_60(self, sweep):
        # OSL>=512 adds 60 on top of the (possibly widened) base.
        assert sweep._hang_stale_limit_s(
            input_len=1024, output_len=512, batch_size=1, base_s=130
        ) == 190

    def test_long_prefill_and_long_osl_stack(self, sweep):
        # max(130,360) then +60 for OSL.
        assert sweep._hang_stale_limit_s(
            input_len=120000, output_len=512, batch_size=1, base_s=130
        ) == 420

    def test_base_override_respected_when_larger(self, sweep):
        # A caller-supplied larger base is never shrunk by the light path.
        assert sweep._hang_stale_limit_s(
            input_len=1024, output_len=128, batch_size=1, base_s=500
        ) == 500


# ---------------------------------------------------------------------------
# 2. _meta_tag — reverse lookup of a bucket tag from (il, ol, bs)
# ---------------------------------------------------------------------------


class TestMetaTag:
    def test_exact_match(self, sweep):
        meta = {
            "il27000_ol150_bs1": {"input_len": 27000, "output_len": 150, "batch_size": 1},
            "il4096_ol256_bs64": {"input_len": 4096, "output_len": 256, "batch_size": 64},
        }
        assert sweep._meta_tag(meta, 4096, 256, 64) == "il4096_ol256_bs64"

    def test_no_match_returns_empty(self, sweep):
        meta = {"il27000_ol150_bs1": {"input_len": 27000, "output_len": 150, "batch_size": 1}}
        assert sweep._meta_tag(meta, 1, 1, 1) == ""

    def test_empty_meta(self, sweep):
        assert sweep._meta_tag({}, 1, 2, 3) == ""


# ---------------------------------------------------------------------------
# 3. _run_cmd_streaming_watchdog — kills a frozen child, preserves a healthy one
# ---------------------------------------------------------------------------


def _write_fake_child(path: Path, *, behavior: str) -> None:
    """Write a small python script that emulates the sweep child by writing a
    status JSON and then either finishing or hanging.

    Uses a LIGHT bucket (il1024/ol128/bs1) so `_hang_stale_limit_s` does NOT
    widen the threshold — the test's small `base_stale_s` then governs and the
    staleness kill fires fast. (The widening for heavy buckets is covered by the
    `TestHangStaleLimit` unit tests, no GPU/process needed there.)

    behavior:
      - "complete": write a couple of warmup status updates, then exit 0.
      - "hang": write ONE benchmark status update, then spin forever (status
        stays frozen) — emulating a wedged generate().
    """
    src = textwrap.dedent(
        f"""
        import json, os, sys, time
        from datetime import datetime, timezone

        status_path = sys.argv[1]
        behavior = {behavior!r}

        def write_status(phase, **extra):
            payload = {{
                "label": "baseline",
                "phase": phase,
                "input_len": 1024,
                "output_len": 128,
                "batch_size": 1,
                "pid": os.getpid(),
                "last_update": datetime.now(timezone.utc).isoformat(),
            }}
            payload.update(extra)
            tmp = status_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, status_path)

        if behavior == "complete":
            for i in range(3):
                write_status("warmup", warmup_iter=i)
                print("warmup", i, flush=True)
                time.sleep(0.2)
            write_status("bucket_done", ok=True)
            sys.exit(0)
        else:  # hang
            write_status("benchmark", iter=1)
            print("entering wedged generate()", flush=True)
            # Spin forever WITHOUT updating status -> watchdog must kill us.
            while True:
                time.sleep(0.5)
        """
    )
    path.write_text(src, encoding="utf-8")


class TestWatchdogSupervisor:
    def test_healthy_child_completes(self, sweep, tmp_path):
        child = tmp_path / "fake_complete.py"
        _write_fake_child(child, behavior="complete")
        status_path = tmp_path / "baseline.json"
        log_path = tmp_path / "sup.log"
        meta = {"il1024_ol128_bs1": {"input_len": 1024, "output_len": 128, "batch_size": 1}}

        res = sweep._run_cmd_streaming_watchdog(
            [sys.executable, str(child), str(status_path)],
            env=dict(**__import__("os").environ),
            cwd=None,
            timeout_s=60,
            log_path=log_path,
            heartbeat_s=0,
            status_path=status_path,
            bucket_meta=meta,
            base_stale_s=3,
        )
        assert res["ok"] is True
        assert res["hung"] is False
        assert res["returncode"] == 0

    def test_frozen_child_is_killed_and_named(self, sweep, tmp_path):
        child = tmp_path / "fake_hang.py"
        _write_fake_child(child, behavior="hang")
        status_path = tmp_path / "baseline.json"
        log_path = tmp_path / "sup.log"
        # Light bucket (bs1, il1024, ol128): _hang_stale_limit_s does NOT widen,
        # so the effective threshold == base_stale_s (2s) and the kill fires fast.
        meta = {"il1024_ol128_bs1": {"input_len": 1024, "output_len": 128, "batch_size": 1}}

        t0 = time.time()
        res = sweep._run_cmd_streaming_watchdog(
            [sys.executable, str(child), str(status_path)],
            env=dict(**__import__("os").environ),
            cwd=None,
            timeout_s=60,            # outer timeout is generous; staleness must fire first
            log_path=log_path,
            heartbeat_s=0,
            status_path=status_path,
            bucket_meta=meta,
            base_stale_s=2,
        )
        elapsed = time.time() - t0
        assert res["hung"] is True, res
        assert res["ok"] is False
        assert res["wedged_tag"] == "il1024_ol128_bs1"
        # Staleness (2s) must fire well before the 60s outer timeout.
        assert elapsed < 30

    def test_outer_timeout_kills_when_phase_not_benchmarking(self, sweep, tmp_path):
        # A child that never writes a warmup/benchmark status (e.g. stuck in
        # model-load) must still be killed by the OUTER timeout, not left forever.
        child = tmp_path / "fake_loadstuck.py"
        child.write_text(
            textwrap.dedent(
                """
                import json, os, sys, time
                from datetime import datetime, timezone
                status_path = sys.argv[1]
                payload = {"label":"baseline","phase":"loading_model",
                           "last_update": datetime.now(timezone.utc).isoformat()}
                with open(status_path, "w") as f:
                    json.dump(payload, f)
                while True:
                    time.sleep(0.5)
                """
            ),
            encoding="utf-8",
        )
        status_path = tmp_path / "baseline.json"
        log_path = tmp_path / "sup.log"
        res = sweep._run_cmd_streaming_watchdog(
            [sys.executable, str(child), str(status_path)],
            env=dict(**__import__("os").environ),
            cwd=None,
            timeout_s=3,             # tiny outer timeout
            log_path=log_path,
            heartbeat_s=0,
            status_path=status_path,
            bucket_meta={},
            base_stale_s=2,
        )
        # loading_model is NOT a tick phase -> staleness uses the generous load
        # grace, so the OUTER timeout (3s) is what kills it. Not flagged as hung.
        assert res["ok"] is False
        assert res["hung"] is False
        assert res["returncode"] != 0


# ---------------------------------------------------------------------------
# 4. Default-OFF contract: the new args exist and default to off/legacy values
# ---------------------------------------------------------------------------


class TestDefaultOffContract:
    def test_args_registered_with_safe_defaults(self, sweep_source):
        assert '"--hang-watchdog"' in sweep_source
        assert 'action="store_true"' in sweep_source
        # Watchdog flag must default False (legacy callers unchanged).
        assert '"--hang-watchdog", action="store_true", default=False' in sweep_source

    def test_skip_tags_is_internal(self, sweep_source):
        # The resume skip-list is an internal (suppressed) flag, parent->child only.
        assert '"--_skip-tags"' in sweep_source
        assert "help=argparse.SUPPRESS" in sweep_source

    def test_legacy_path_still_uses_plain_streaming(self, sweep_source):
        # When the watchdog is OFF, the parent must call the unchanged
        # _run_cmd_streaming (legacy fail-fast contract preserved).
        assert "if not hang_watchdog:" in sweep_source
        assert "child_res = _run_cmd_streaming(" in sweep_source
