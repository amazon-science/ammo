# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Codex Stop-hook gate: terminal-status and fail-closed contract.

Scope note: the hook exposes `_stop_nudge(artifact_dir, state)`. Stage-2,
Stage-3 debate, and round-index assertions that used to live here pinned a
pre-minification API that never existed in this file; the live Stage-2 and
report fact-check contracts are owned in-tree by
ai_cli_session/.codex/skills/ammo/tests/test_hook_semantics.py.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent
HOOK_DIR = ROOT / "ai_cli_session" / ".codex" / "hooks"
HOOK = HOOK_DIR / "stop_gate_guard.py"

sys.path.insert(0, str(HOOK_DIR))
spec = importlib.util.spec_from_file_location("codex_stop_gate_guard", HOOK)
stop_gate_guard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stop_gate_guard)


def _state(current_round=1, current_stage="7_campaign_eval", status="active"):
    return {
        "campaign": {
            "status": status,
            "current_round": current_round,
            "current_stage": current_stage,
            "rounds": [{"round_id": 1}],
        }
    }


def _write_accepted_report(artifact_dir):
    report = artifact_dir / "REPORT.md"
    report.write_text("report", encoding="utf-8")
    assets = artifact_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "report_fact_check.json").write_text(
        json.dumps(
            {
                "ok": True,
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_paused_status_allows_stop_without_final_artifacts(tmp_path):
    reason = stop_gate_guard._stop_nudge(tmp_path, _state(status="paused"))

    assert reason is None


def test_campaign_complete_requires_report(tmp_path):
    reason = stop_gate_guard._stop_nudge(tmp_path, _state(status="campaign_complete"))

    assert reason is not None
    assert "REPORT.md has not been generated" in reason


def test_campaign_exhausted_requires_report(tmp_path):
    reason = stop_gate_guard._stop_nudge(tmp_path, _state(status="campaign_exhausted"))

    assert reason is not None
    assert "REPORT.md has not been generated" in reason


def test_final_status_blocks_report_without_accepted_fact_check(tmp_path):
    (tmp_path / "REPORT.md").write_text("report", encoding="utf-8")

    reason = stop_gate_guard._stop_nudge(tmp_path, _state(status="campaign_complete"))

    assert reason is not None
    assert "fact-check" in reason


def test_final_campaign_statuses_pass_when_report_is_fact_checked(tmp_path):
    _write_accepted_report(tmp_path)

    for status in ("campaign_complete", "campaign_exhausted"):
        reason = stop_gate_guard._stop_nudge(tmp_path, _state(status=status))

        assert reason is None, f"{status}: {reason}"


def test_active_campaign_blocks_stop(tmp_path):
    reason = stop_gate_guard._stop_nudge(tmp_path, _state(current_stage="3_debate"))

    assert reason is not None


def test_stop_hook_fails_closed_on_invalid_active_state_json(tmp_path):
    artifact_dir = tmp_path / "kernel_opt_artifacts" / "target"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "state.json").write_text("{not json", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"cwd": str(ROOT)}),
        env={**os.environ, "AMMO_ARTIFACT_DIR": str(artifact_dir)},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["decision"] == "block"
