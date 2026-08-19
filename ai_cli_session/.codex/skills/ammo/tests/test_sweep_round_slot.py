#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Round-scoped sweep path tests for run_vllm_bench_latency_sweep.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _make_v2_artifact(tmp_path: Path, current_round: int = 1) -> Path:
    artifact = tmp_path / "artifact"
    (artifact / "rounds" / str(current_round) / "sweeps").mkdir(parents=True)
    (artifact / "state.json").write_text(
        json.dumps({"campaign": {"current_round": current_round, "rounds": []}}),
        encoding="utf-8",
    )
    return artifact


def _make_legacy_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "legacy"
    artifact.mkdir(parents=True)
    return artifact


class TestRoundSlotPathResolution:
    def _resolve(self, *, artifact: Path, round_=None, slot=None, out_name="e2e_latency", _out_root=None):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name

        ns = mock.Mock(round=round_, slot=slot, out_name=out_name, _out_root=_out_root)
        return _resolve_sweep_out_name(ns, artifact)

    def test_round_and_slot_resolve_to_round_scoped_sweep(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path)

        assert self._resolve(artifact=artifact, round_=1, slot="profiling") == "rounds/1/sweeps/profiling"
        assert self._resolve(artifact=artifact, round_=1, slot="opt/op007") == "rounds/1/sweeps/opt/op007"

    def test_slot_only_reads_current_round_from_state(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=3)

        assert self._resolve(artifact=artifact, slot="baseline") == "rounds/3/sweeps/baseline"

    def test_round_without_slot_fails(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path)

        with pytest.raises(SystemExit, match="--round requires --slot"):
            self._resolve(artifact=artifact, round_=2)

    def test_custom_out_name_fails_on_v2_layout(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path)

        with pytest.raises(SystemExit, match="--out-name is removed"):
            self._resolve(artifact=artifact, out_name="profiling")

    def test_legacy_layout_keeps_out_name(self, tmp_path):
        artifact = _make_legacy_artifact(tmp_path)

        assert self._resolve(artifact=artifact, out_name="profiling") == "profiling"


def test_prepare_out_root_archives_v2_slot_under_round_archive(tmp_path):
    from run_vllm_bench_latency_sweep import _prepare_out_root

    artifact = _make_v2_artifact(tmp_path)
    out = artifact / "rounds" / "1" / "sweeps" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "old.json").write_text("{}", encoding="utf-8")

    new_out = _prepare_out_root(
        artifact_dir=artifact,
        out_name="rounds/1/sweeps/baseline",
        overwrite=False,
    )

    assert new_out == out
    assert not (out / "old.json").exists()
    archived = list((artifact / "rounds" / "1" / "_archive").glob("baseline_*"))
    assert archived
    assert (archived[0] / "old.json").exists()


def test_all_stage5_sibling_slots_are_gate_slots():
    from run_vllm_bench_latency_sweep import _is_gate_slot

    for slot in (
        "opt/op007",
        "opt_correctness/op007",
        "opt_profiling/op007",
        "integration",
        "integration_profiling",
    ):
        assert _is_gate_slot(slot), slot
    for slot in (None, "baseline", "profiling", "golden_capture"):
        assert not _is_gate_slot(slot), slot


def test_stage5_profiling_dirs_are_track_scoped_while_stage1_stays_round_scoped(tmp_path):
    from run_vllm_bench_latency_sweep import _v2_profiling_dir

    round_root = tmp_path / "artifact" / "rounds" / "2"
    op1 = round_root / "sweeps" / "opt_profiling" / "op001"
    op2 = round_root / "sweeps" / "opt_profiling" / "op002"
    stage1 = round_root / "sweeps" / "profiling"
    post_ship = round_root / "sweeps" / "post_ship_profiling"
    integration = round_root / "sweeps" / "integration_profiling"

    assert _v2_profiling_dir(op1, "nsys") == round_root / "profiling" / "nsys" / "opt" / "op001"
    assert _v2_profiling_dir(op2, "nsys") == round_root / "profiling" / "nsys" / "opt" / "op002"
    assert _v2_profiling_dir(stage1, "nsys") == round_root / "profiling" / "nsys"
    assert _v2_profiling_dir(post_ship, "nsys") == round_root / "profiling" / "nsys" / "post_ship"
    assert _v2_profiling_dir(integration, "nsys") == round_root / "profiling" / "nsys" / "integration"
    assert _v2_profiling_dir(op1, "nsys") != _v2_profiling_dir(op2, "nsys")


def test_v2_profiling_dir_routes_outside_sweep_slot(tmp_path):
    from run_vllm_bench_latency_sweep import _v2_profiling_dir

    out_root = tmp_path / "artifact" / "rounds" / "2" / "sweeps" / "profiling"

    assert _v2_profiling_dir(out_root, "nsys") == tmp_path / "artifact" / "rounds" / "2" / "profiling" / "nsys"


def test_build_child_cmd_omits_out_name():
    from run_vllm_bench_latency_sweep import _build_child_cmd

    cmd = _build_child_cmd(
        python_exe="/venv/bin/python",
        script_path=Path("/sweep.py"),
        run_label="baseline",
        artifact_dir=Path("/art"),
        target_path=Path("/art/target.json"),
        timeout_s=1800,
        out_name="rounds/1/sweeps/baseline",
        out_root=Path("/art/rounds/1/sweeps/baseline"),
        dp=1,
        nproc=1,
        extra_child_flags=[],
    )

    assert "--out-name" not in cmd
    assert cmd[cmd.index("--_out-root") + 1] == "/art/rounds/1/sweeps/baseline"


def test_baseline_slot_rejects_profiling_flags_before_target_load(tmp_path):
    from run_vllm_bench_latency_sweep import main

    artifact = _make_v2_artifact(tmp_path)
    argv = [
        "sweep",
        "--artifact-dir",
        str(artifact),
        "--round",
        "1",
        "--slot",
        "baseline",
        "--labels",
        "baseline",
        "--nsys-profile",
    ]

    with mock.patch("sys.argv", argv), mock.patch("shutil.which", return_value="/usr/bin/nsys"):
        with pytest.raises(SystemExit, match="cannot be combined with --slot baseline"):
            main()


def test_clean_gate_slot_accepts_opt_only_with_stage1_baseline(tmp_path):
    """Stage-1 baseline reuse: opt-only + --baseline-from is the canonical
    Gate 5.3b invocation — the gate-slot guard must NOT reject it (a worktree
    baseline arm can execute the optimized path and contaminate both arms)."""
    from run_vllm_bench_latency_sweep import main

    artifact = _make_v2_artifact(tmp_path)
    imported = artifact / "rounds" / "1" / "sweeps" / "baseline"
    imported.mkdir(parents=True, exist_ok=True)
    argv = [
        "sweep",
        "--artifact-dir",
        str(artifact),
        "--round",
        "1",
        "--slot",
        "opt/OP-001",
        "--labels",
        "opt",
        "--baseline-from",
        str(imported),
    ]

    with mock.patch("sys.argv", argv):
        try:
            main()
        except SystemExit as exc:
            # The invocation proceeds past the label guard; the fixture has no
            # target.json so a later failure is expected — but it must not be
            # the old paired-A/B rejection.
            assert "requires --labels baseline,opt" not in str(exc)
        except Exception:
            pass  # later pipeline failure on the minimal fixture is fine
