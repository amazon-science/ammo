#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""TDD tests for DP / EP pass-through in new_target.py.

Covers:
  - --data-parallel-size flag on new_target.py
  - --enable-expert-parallel flag on new_target.py
  - Injection of --distributed-executor-backend external_launcher when DP > 1
  - Backward compatibility: defaults produce empty bench.extra_args
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable (matches test_workload_matrix.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _make_fields(**overrides):
    """Build a TargetFields via _default_target_fields with test-friendly defaults.

    Mirrors the argparse.Namespace shape new_target.main() produces so we can
    exercise the JSON-composition logic directly without invoking argparse.
    """
    from new_target import _default_target_fields

    ns = argparse.Namespace(
        model_id="Qwen/Qwen3-Coder-480B",
        hardware="H100",
        dtype="fp8",
        tp=1,
        ep=1,
        max_model_len=4096,
        input_len=64,
        output_len=512,
        batch_sizes=[1, 8, 32],
        num_iters=5,
        noise_tolerance_pct=0.5,
        catastrophic_regression_pct=5.0,
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return _default_target_fields(ns)


def _target_extra_args(**overrides):
    """Build TargetFields + run _target_json, return bench.extra_args list."""
    from new_target import _target_json

    fields = _make_fields(**overrides)
    out = _target_json(fields, Path("/tmp/fake_artifact_dir"))
    return out["bench"]["extra_args"]


class TestDefaults:
    """Backward compatibility: old callers with no DP/EP flags."""

    def test_defaults_no_dp_ep(self):
        """No DP/EP overrides → extra_args stays empty (backward compat)."""
        assert _target_extra_args() == []

    def test_dp_1_extra_args_empty(self):
        """Explicit --data-parallel-size 1 → no injection (DP=1 is single-process)."""
        assert _target_extra_args(data_parallel_size=1) == []


class TestDpInjection:
    """DP > 1 triggers auto-injection of external_launcher backend."""

    def test_dp_2_injects_backend(self):
        """--data-parallel-size 2 → DP flag + external_launcher backend, in order."""
        assert _target_extra_args(data_parallel_size=2) == [
            "--data-parallel-size", "2",
            "--distributed-executor-backend", "external_launcher",
        ]

    def test_dp_4_injects_backend(self):
        """Higher DP values stringify correctly."""
        assert _target_extra_args(data_parallel_size=4) == [
            "--data-parallel-size", "4",
            "--distributed-executor-backend", "external_launcher",
        ]


class TestEpFlag:
    """--enable-expert-parallel is independent of DP."""

    def test_enable_expert_parallel_only(self):
        """EP=True, DP=1 → only the EP flag in extra_args."""
        assert _target_extra_args(enable_expert_parallel=True) == [
            "--enable-expert-parallel",
        ]


class TestCombined:
    """DP and EP compose together. Order is deterministic: DP group first."""

    def test_dp_and_ep_compose(self):
        assert _target_extra_args(data_parallel_size=4, enable_expert_parallel=True) == [
            "--data-parallel-size", "4",
            "--distributed-executor-backend", "external_launcher",
            "--enable-expert-parallel",
        ]


class TestExistingFieldsUnaffected:
    """Regression: existing workload / target fields are not disturbed by DP/EP."""

    def test_dp_preserves_existing_tp_and_batch_sizes(self):
        from new_target import _target_json

        fields = _make_fields(tp=4, batch_sizes=[1, 8, 32], data_parallel_size=2)
        out = _target_json(fields, Path("/tmp/fake_artifact_dir"))

        assert out["target"]["tp"] == 4
        assert out["workload"]["batch_sizes"] == [1, 8, 32]
        assert out["bench"]["extra_args"] == [
            "--data-parallel-size", "2",
            "--distributed-executor-backend", "external_launcher",
        ]


class TestArgparseIntegration:
    """End-to-end: run new_target.py as a subprocess to catch argparse wiring bugs.

    Direct imports of _target_json bypass argparse, so dest=/action= typos
    in add_argument() would silently slip through. One subprocess invocation
    guards against that.
    """

    def test_cli_flags_produce_expected_extra_args(self, tmp_path):
        script = _SCRIPTS_DIR / "new_target.py"
        artifact_dir = tmp_path / "artifact"
        cmd = [
            sys.executable,
            str(script),
            "--artifact-dir", str(artifact_dir),
            "--model-id", "test-model",
            "--hardware", "H100",
            "--dtype", "fp8",
            "--tp", "2",
            "--data-parallel-size", "2",
            "--enable-expert-parallel",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        target = json.loads((artifact_dir / "target.json").read_text())
        assert target["bench"]["extra_args"] == [
            "--data-parallel-size", "2",
            "--distributed-executor-backend", "external_launcher",
            "--enable-expert-parallel",
        ]


class TestStateJsonDpPersistence:
    """state.json must record dp parallelism alongside tp / ep."""

    def test_state_json_includes_dp_default_1(self):
        from new_target import _state_json

        fields = _make_fields()
        state = _state_json(fields, Path("/tmp/fake_artifact_dir"))
        assert state["target"]["dp"] == 1
        assert state["campaign"]["schema_version"] == "4.2"

    def test_state_json_includes_dp_when_set(self):
        from new_target import _state_json

        fields = _make_fields(tp=2, ep=1, data_parallel_size=4)
        state = _state_json(fields, Path("/tmp/fake_artifact_dir"))
        assert state["target"]["dp"] == 4
        assert state["target"]["tp"] == 2
        assert state["target"]["ep"] == 1

    def test_state_json_dp_via_argparse_subprocess(self, tmp_path):
        script = _SCRIPTS_DIR / "new_target.py"
        artifact_dir = tmp_path / "artifact"
        cmd = [
            sys.executable,
            str(script),
            "--artifact-dir", str(artifact_dir),
            "--model-id", "test-model",
            "--hardware", "H100",
            "--dtype", "fp8",
            "--tp", "2",
            "--data-parallel-size", "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        state = json.loads((artifact_dir / "state.json").read_text())
        assert state["target"]["dp"] == 2
        assert state["target"]["tp"] == 2

    def test_state_schema_allows_legacy_missing_dp_but_defines_optional_dp(self):
        schema_path = _SCRIPTS_DIR.parents[2] / "schemas" / "state.schema.json"
        schema = json.loads(schema_path.read_text())
        target_schema = schema["properties"]["target"]
        assert "dp" not in target_schema["required"]
        assert target_schema["properties"]["dp"]["minimum"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
