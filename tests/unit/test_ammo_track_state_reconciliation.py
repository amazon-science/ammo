# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "scripts" / "reconcile_track_state.py"
IMPLEMENTER_DOC = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "agents" / "ammo-implementer.md"
IMPL_RULES_DOC = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "references" / "impl-track-rules.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("reconcile_track_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_stale_fail_track(artifact_dir: Path) -> None:
    track_dir = artifact_dir / "rounds" / "1" / "tracks" / "op001"
    track_dir.mkdir(parents=True)
    (artifact_dir / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": "4_5_parallel_tracks",
                    "rounds": [
                        {
                            "round_id": 1,
                            "status": "IN_PROGRESS",
                            "parallel_tracks": {
                                "started_at": "2026-04-29T00:00:00Z",
                                "completed_at": None,
                                "tracks": {
                                    "op001": {
                                        "status": "IN_PROGRESS",
                                        "verdict": None,
                                        "branch": "ammo/op001",
                                        "worktree_path": "/tmp/worktree-op001",
                                        "validation_results_path": "rounds/1/tracks/op001/validation_results.md",
                                        "evidence_path": "rounds/1/tracks/op001/evidence.json",
                                        "kill_criteria_results": {},
                                    }
                                },
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    (track_dir / "evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "track_id": "op001",
                "correctness": {"status": "FAIL"},
                "kernel_bench": {"status": "FAIL"},
                "e2e": {
                    "status": "FAIL",
                    "run_purpose": "official",
                    "admissibility": {"status": "FAIL", "issues": ["accuracy mismatch"]},
                    "fastpath_proof": {"status": "FAIL", "hits": 0},
                },
                "kill_criteria": {
                    "accuracy": {
                        "status": "FAIL",
                        "source_run_purpose": "official",
                        "note": "Gate 5.1b failed.",
                    },
                    "fastpath": {
                        "status": "FAIL",
                        "source_run_purpose": "official",
                        "note": "Expected kernel did not appear.",
                    },
                },
                "amdahl": {},
                "cross_track_contamination": {"status": "N/A", "note": "single track"},
            }
        ),
        encoding="utf-8",
    )
    (track_dir / "validation_results.md").write_text(
        "# op001 validation results\n\nOverall verdict: FAIL\n",
        encoding="utf-8",
    )


def test_fail_evidence_reconciles_round_track_status_verdict_and_kill_criteria(tmp_path):
    _write_stale_fail_track(tmp_path)
    module = _load_module()

    result = module.reconcile_track_state(tmp_path, "op001", write=True)

    assert result.changed is True
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]
    assert track["status"] == "FAIL"
    assert track["verdict"] == "FAIL"
    assert track["kill_criteria_results"] == {"accuracy": "FAIL", "fastpath": "FAIL"}
    assert track["evidence_path"] == "rounds/1/tracks/op001/evidence.json"
    assert track["validation_results_path"] == "rounds/1/tracks/op001/validation_results.md"
    assert track["completed_at"]


def test_check_mode_blocks_when_fail_evidence_and_state_are_not_reconciled(tmp_path):
    _write_stale_fail_track(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(tmp_path),
            "--track-id",
            "op001",
            "--check",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert "state.json is not reconciled" in result.stderr
    assert "status: IN_PROGRESS -> FAIL" in result.stderr
    assert "verdict: None -> FAIL" in result.stderr
    assert "kill_criteria_results: {} -> {'accuracy': 'FAIL', 'fastpath': 'FAIL'}" in result.stderr


def test_gpu_blocked_status_keeps_schema_valid_null_verdict(tmp_path):
    _write_stale_fail_track(tmp_path)
    evidence_path = tmp_path / "rounds" / "1" / "tracks" / "op001" / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["verdict"] = "GPU_BLOCKED"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    module = _load_module()

    result = module.reconcile_track_state(tmp_path, "op001", write=True)

    assert result.changed is True
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    track = state["campaign"]["rounds"][0]["parallel_tracks"]["tracks"]["op001"]
    assert track["status"] == "GPU_BLOCKED"
    assert track["verdict"] is None


def test_implementer_docs_require_state_reconciliation_for_failure_outputs():
    implementer = IMPLEMENTER_DOC.read_text(encoding="utf-8")
    rules = IMPL_RULES_DOC.read_text(encoding="utf-8")

    required_phrases = [
        "python .codex/skills/ammo/scripts/reconcile_track_state.py",
        "--write",
        "--check",
        "Do NOT report `TRACK_COMPLETE`",
        "state.json has the same terminal status",
        "kill_criteria_results",
    ]
    for phrase in required_phrases:
        assert phrase in implementer

    assert "failure is not complete until `state.json` is reconciled" in rules
