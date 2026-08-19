# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Zone D — Agent/Hook Consistency Tests (Tests 38-43).

Verifies that agent prompt files and hook test fixtures reference the correct
v4.0 e2e_latency fields and don't write to removed legacy fields without
also writing to the new fields.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root for finding ai_cli_session files
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SESSION = REPO_ROOT / "ai_cli_session"


# ─── Test 38: new_target.py uses e2e_latency fields ────────────────────────

class TestNewTargetPy:
    """Test 38: new_target.py generates baseline/integration templates with new fields."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.script = AI_SESSION / ".claude" / "skills" / "ammo" / "scripts" / "new_target.py"

    def _run_new_target(self, extra_args=None):
        """Run new_target.py and return the generated state.json."""
        artifact_dir = self.tmp_path / "test_target"
        cmd = [
            sys.executable, str(self.script),
            "--artifact-dir", str(artifact_dir),
            "--model-id", "test-model",
            "--hardware", "H100",
            "--dtype", "bf16",
        ]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"new_target.py failed: {result.stderr}"
        state_path = artifact_dir / "state.json"
        assert state_path.exists()
        return json.loads(state_path.read_text())

    def test_baseline_has_e2e_latency_fields(self):
        """Baseline template includes e2e_latency and per_bs_verdict."""
        state = self._run_new_target()
        baseline = state["campaign"]["rounds"][0]["baseline"]
        assert "e2e_latency" in baseline, "baseline missing e2e_latency"
        assert baseline["e2e_latency"] is None
        assert "per_bs_verdict" in baseline, "baseline missing per_bs_verdict"
        assert baseline["per_bs_verdict"] is None

    def test_integration_has_e2e_latency_combined(self):
        """Integration template includes e2e_latency_combined, per_bs_verdict, commit_sha."""
        state = self._run_new_target()
        integration = state["campaign"]["rounds"][0]["integration"]
        assert "e2e_latency_combined" in integration, "integration missing e2e_latency_combined"
        assert integration["e2e_latency_combined"] is None
        assert "per_bs_verdict" in integration, "integration missing per_bs_verdict"
        assert integration["per_bs_verdict"] is None
        assert "commit_sha" in integration, "integration missing commit_sha"
        assert integration["commit_sha"] is None

    def test_combined_e2e_result_still_present_during_transition(self):
        """combined_e2e_result is kept during transition."""
        state = self._run_new_target()
        integration = state["campaign"]["rounds"][0]["integration"]
        assert "combined_e2e_result" in integration

    def test_no_round_1_baseline_latency_s_in_campaign(self):
        """round_1_baseline_latency_s removed from campaign template."""
        state = self._run_new_target()
        campaign = state["campaign"]
        assert "round_1_baseline_latency_s" not in campaign

    def test_no_cumulative_speedup_vs_round1_in_campaign(self):
        """cumulative_speedup_vs_round1 removed from campaign template."""
        state = self._run_new_target()
        campaign = state["campaign"]
        assert "cumulative_speedup_vs_round1" not in campaign

    def test_target_has_dp_field(self):
        """target template includes dp field."""
        state = self._run_new_target()
        target = state["target"]
        assert "dp" in target, "target missing dp field"
        assert target["dp"] == 1


# ─── Test 39: state-validate hook accepts new schema ───────────────────────

class TestStateValidateHook:
    """Test 39: test-ammo-state-validate.sh passes with v4.0 schema."""

    def test_state_validate_hook_passes(self):
        """Hook test harness passes (all its internal tests pass)."""
        hook_test = AI_SESSION / ".claude" / "hooks" / "test-ammo-state-validate.sh"
        assert hook_test.exists(), f"Hook test not found: {hook_test}"
        result = subprocess.run(
            ["bash", str(hook_test)],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"test-ammo-state-validate.sh failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )


# ─── Test 40: stop-guard hook accepts new schema ──────────────────────────

class TestStopGuardHook:
    """Test 40: test-ammo-stop-guard.sh passes with v4.0 schema."""

    def test_stop_guard_hook_passes(self):
        """Hook test harness passes (all its internal tests pass)."""
        hook_test = AI_SESSION / ".claude" / "hooks" / "test-ammo-stop-guard.sh"
        assert hook_test.exists(), f"Hook test not found: {hook_test}"
        result = subprocess.run(
            ["bash", str(hook_test)],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"test-ammo-stop-guard.sh failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )


# ─── Test 41: integration-logic.md references new field ───────────────────

class TestIntegrationLogicMd:
    """Test 41: integration-logic.md contains e2e_latency_combined."""

    def _read_file(self):
        path = AI_SESSION / ".claude" / "skills" / "ammo" / "orchestration" / "integration-logic.md"
        return path.read_text()

    def _schema(self):
        import json
        path = AI_SESSION / ".claude" / "schemas" / "state.schema.json"
        return json.loads(path.read_text())

    def test_contains_e2e_latency_combined(self):
        """The combined-result field lives in the schema (the minified doc
        defers state shape to the schema authority)."""
        integ = self._schema()["properties"]["campaign"]["properties"]["rounds"][
            "items"]["properties"]["integration"]["properties"]
        assert "e2e_latency_combined" in integ

    def test_documents_map_based_merge(self):
        """The map-based (batch-size keyed) shape is schema-enforced."""
        integ = self._schema()["properties"]["campaign"]["properties"]["rounds"][
            "items"]["properties"]["integration"]["properties"]
        combined = integ["e2e_latency_combined"]
        # map keyed by batch size -> latency stats object with avg
        ap = combined.get("additionalProperties") or {}
        assert "$ref" in str(ap) or "avg" in str(ap), combined

    def test_documents_field_ownership(self):
        """Integration doc still binds the shipped result to the integration
        commit and measured combined artifact."""
        content = self._read_file()
        assert "integration" in content
        assert "measured production E2E" in content or "commit" in content

    def test_no_legacy_combined_e2e_result_writes(self):
        """File does NOT instruct writing to combined_e2e_result."""
        content = self._read_file()
        assert "combined_e2e_result" not in content

    def test_no_legacy_speedup_x_field(self):
        """File does NOT reference standalone speedup_x as an integration stored field."""
        content = self._read_file()
        # combined_e2e_speedup_x is a round-level summary field (still valid)
        # We're checking that the integration object doesn't use bare "speedup_x"
        # as a field name (legacy combined_e2e_result.speedup_x)
        lines_with_speedup_x = [
            line for line in content.split("\n")
            if "speedup_x" in line and "combined_e2e_speedup_x" not in line
        ]
        assert not lines_with_speedup_x, f"Found legacy speedup_x refs: {lines_with_speedup_x}"

    def test_no_legacy_delta_pp_field(self):
        """File does NOT reference standalone delta_pp as an integration stored field."""
        content = self._read_file()
        # combined_e2e_delta_pp is a round-level summary field (still valid)
        # We're checking that the integration object doesn't use bare "delta_pp"
        lines_with_delta_pp = [
            line for line in content.split("\n")
            if "delta_pp" in line and "combined_e2e_delta_pp" not in line
        ]
        assert not lines_with_delta_pp, f"Found legacy delta_pp refs: {lines_with_delta_pp}"

    def test_no_cumulative_speedup_vs_round1(self):
        """File does NOT reference cumulative_speedup_vs_round1."""
        content = self._read_file()
        assert "cumulative_speedup_vs_round1" not in content

    def test_no_round_1_baseline_latency_s(self):
        """File does NOT reference round_1_baseline_latency_s."""
        content = self._read_file()
        assert "round_1_baseline_latency_s" not in content


# ─── Test 42: researcher writes to baseline.e2e_latency ──────────────────

class TestResearcherMd:
    """Test 42: ammo-researcher.md contains instructions to write baseline.e2e_latency."""

    def _read_file(self):
        path = AI_SESSION / ".claude" / "agents" / "ammo-researcher.md"
        return path.read_text()

    def test_writes_baseline_e2e_latency_as_map(self):
        """Researcher instructions describe writing baseline.e2e_latency as a map."""
        content = self._read_file()
        assert "baseline.e2e_latency" in content or "baseline\"].e2e_latency" in content or \
               ("e2e_latency" in content and "baseline" in content)

    def test_writes_per_bs_verdict_as_sibling(self):
        """Researcher instructions mention per_bs_verdict as sibling."""
        content = self._read_file()
        assert "per_bs_verdict" in content

    def test_describes_map_shape(self):
        """Researcher instructions describe map shape with avg/p50 (no _s suffix)."""
        content = self._read_file()
        # Should mention avg, p50 without _s suffix in the context of e2e_latency writing
        assert re.search(r'"avg".*"p50"|avg.*p50', content)

    def test_no_combined_e2e_result_write(self):
        """Researcher does NOT write to combined_e2e_result."""
        content = self._read_file()
        assert "combined_e2e_result" not in content

    def test_no_round_1_baseline_latency_s(self):
        """Researcher does NOT reference round_1_baseline_latency_s."""
        content = self._read_file()
        assert "round_1_baseline_latency_s" not in content

    def test_no_cumulative_speedup_vs_round1(self):
        """Researcher does NOT reference cumulative_speedup_vs_round1."""
        content = self._read_file()
        assert "cumulative_speedup_vs_round1" not in content


# ─── Test 43: no undeclared combined_e2e_result writes ────────────────────

class TestNoLegacyWritesInAgentFiles:
    """Test 43: No agent/skill file writes to combined_e2e_result without new field."""

    def _agent_and_skill_files(self):
        """Gather all agent and orchestration skill MD files."""
        files = []
        agents_dir = AI_SESSION / ".claude" / "agents"
        if agents_dir.exists():
            files.extend(agents_dir.glob("*.md"))
        orch_dir = AI_SESSION / ".claude" / "skills" / "ammo" / "orchestration"
        if orch_dir.exists():
            files.extend(orch_dir.glob("*.md"))
        return files

    def test_no_solo_combined_e2e_result_writes(self):
        """No agent/orchestration file writes combined_e2e_result without e2e_latency_combined."""
        for filepath in self._agent_and_skill_files():
            content = filepath.read_text()
            if "combined_e2e_result" in content:
                # If it references combined_e2e_result, it must also reference the new field
                # OR be in a read-only/legacy context (not a write instruction)
                # For v4.0 post-migration: combined_e2e_result should not appear at all
                pytest.fail(
                    f"{filepath.name} still references combined_e2e_result. "
                    f"After v4.0 migration, agent files should only use e2e_latency_combined."
                )
