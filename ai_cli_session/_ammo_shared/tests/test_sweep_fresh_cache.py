#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for the --fresh-cache sweep flag.

Covers:
- --fresh-cache CLI argument (default OFF; injects VLLM_CACHE_ROOT + TRITON_CACHE_DIR)
- Help text mentions the compile cost

The retired --num-launches multi-launch subsystem was tested here too; the ship
gate now decides from a single launch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv):
    """Build the sweep argparser and parse argv (excluding program name)."""
    from run_vllm_bench_latency_sweep import main
    import argparse as _ap

    captured = {}

    def intercept(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(_ap.ArgumentParser, "parse_args", intercept):
        try:
            main()
        except SystemExit:
            pass

    parser = captured.get("parser")
    assert parser is not None, "Could not capture parser"
    return parser.parse_args(argv)


class TestFreshCacheArg:
    """--fresh-cache (default OFF)."""

    def test_default_false(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x"])
        assert ns.fresh_cache is False

    def test_flag_sets_true(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x", "--fresh-cache"])
        assert ns.fresh_cache is True


# ---------------------------------------------------------------------------
# Fresh cache env var injection
# ---------------------------------------------------------------------------


class TestFreshCacheEnvInjection:
    """When --fresh-cache is set, VLLM_CACHE_ROOT + TRITON_CACHE_DIR injected
    into child_env for BOTH run.env and os.environ branches."""

    def _inject(self, base_env, sweep_cache_root):
        from run_vllm_bench_latency_sweep import _inject_fresh_cache_env
        return _inject_fresh_cache_env(base_env, sweep_cache_root)

    def test_injects_vllm_cache_root(self, tmp_path):
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "VLLM_CACHE_ROOT" in out
        assert str(tmp_path / "cache" / "abc") == out["VLLM_CACHE_ROOT"]

    def test_injects_triton_cache_dir(self, tmp_path):
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "TRITON_CACHE_DIR" in out
        # TRITON must be inside sweep cache root
        assert out["TRITON_CACHE_DIR"].startswith(str(tmp_path / "cache" / "abc"))

    def test_does_not_set_vllm_disable_compile_cache(self, tmp_path):
        """Task requirement: do NOT set VLLM_DISABLE_COMPILE_CACHE=1."""
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "VLLM_DISABLE_COMPILE_CACHE" not in out

    def test_preserves_existing_vars(self, tmp_path):
        env = {"PYTHONPATH": "/foo", "UNRELATED": "bar"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert out["PYTHONPATH"] == "/foo"
        assert out["UNRELATED"] == "bar"

    def test_does_not_mutate_input(self, tmp_path):
        env = {"PYTHONPATH": "/foo"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        # input env should not have been modified in place
        assert "VLLM_CACHE_ROOT" not in env

    def test_overrides_existing_cache_root(self, tmp_path):
        env = {"VLLM_CACHE_ROOT": "/already/set"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert out["VLLM_CACHE_ROOT"] == str(tmp_path / "cache" / "abc")


# ---------------------------------------------------------------------------
# Row schema: the flat avg_latency entry carries no launches/aggregate keys
# ---------------------------------------------------------------------------


class TestFlatRowSchema:
    def test_entry_has_no_launches_or_aggregate_key(self):
        from run_vllm_bench_latency_sweep import _build_label_result_entry

        entry = _build_label_result_entry(
            cmd=["python", "x"],
            env_overrides={},
            metrics={"avg_s": 5.0},
            log_rel="logs/x.log",
            output_json_rel="json/x.json",
            runner_json_rel="json/x.runner.json",
            ok=True,
            returncode=0,
            evidence_status="unknown",
            evidence={},
            timing={},
        )
        assert "launches" not in entry
        assert "aggregate" not in entry
        assert entry["avg_s"] == 5.0


# ---------------------------------------------------------------------------
# --fresh-cache help text references compile cost
# ---------------------------------------------------------------------------


class TestHelpText:
    """Help text should mention compile-cache characteristics."""

    def test_fresh_cache_help_mentions_compile(self, capsys):
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", ["sweep", "--help"]):
            with pytest.raises(SystemExit):
                main()
        out = capsys.readouterr().out
        assert "--fresh-cache" in out
        # Help text should explain:
        # "First launch pays full compile (~5 min for large models)."
        assert "compile" in out.lower()
