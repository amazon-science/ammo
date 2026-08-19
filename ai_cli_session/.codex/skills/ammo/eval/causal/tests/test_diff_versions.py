# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# test_diff_versions.py
"""Tests for diff_versions.py — cross-version causal DAG comparison."""
import json
import copy
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_dag():
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes
    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(session_jsonl_path=FIXTURES / "sample_session.jsonl", session_data=session_data)
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, _ = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag


def _make_v2(dag):
    v2 = copy.deepcopy(dag)
    for node in v2["nodes"]:
        if node["id"] == "debate":
            node["metrics"]["data_grounding"] = 0.3
            node["anomaly"] = {"type": "DATA_CORRUPTION", "triggered_by": {"data_grounding": 0.3}}
        if node["id"] == "bottleneck_mining":
            node["metrics"]["retry_rate"] = 0.05
            node["anomaly"] = None
    return v2


def test_metric_deltas_computed():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "metric_deltas" in diff
    assert "debate" in diff["metric_deltas"]


def test_new_anomalies_detected():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    new = diff.get("new_anomalies", [])
    assert any(a["node"] == "debate" for a in new)


def test_resolved_anomalies_detected():
    from diff_versions import diff_versions
    v1 = _build_dag()
    for n in v1["nodes"]:
        if n["id"] == "bottleneck_mining":
            n["anomaly"] = {"type": "THRASHING", "triggered_by": {"retry_rate": 0.4}}
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    resolved = diff.get("resolved_anomalies", [])
    assert any(a["node"] == "bottleneck_mining" for a in resolved)


def test_edge_count_changes():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = copy.deepcopy(v1)
    v2["edges"] = [e for e in v2["edges"] if e["type"] != "data_citation"]
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "edge_changes" in diff
    if "data_citation" in diff["edge_changes"]:
        assert diff["edge_changes"]["data_citation"]["delta"] < 0


def test_diff_json_schema():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "version" in diff
    assert "compared" in diff
    assert "metric_deltas" in diff
    assert "new_anomalies" in diff
    assert "resolved_anomalies" in diff
    assert "edge_changes" in diff
    assert "anomaly_distribution_shift" in diff


def test_cli_with_two_dags(tmp_path):
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    diff = json.loads(out.read_text())
    assert "metric_deltas" in diff


def test_regression_report_with_deep_flag(tmp_path):
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    report = tmp_path / "regression_report.md"
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out), "--deep",
         "--regression-report", str(report)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert report.exists()
    assert len(report.read_text()) > 0


# ---------------------------------------------------------------------------
# New tests: skill-diff correlation
# ---------------------------------------------------------------------------

def test_skill_diff_parsing():
    """Parse a unified diff into structured hunks."""
    from diff_versions import _parse_skill_diff
    patch = """diff --git a/.codex/skills/ammo/SKILL.md b/.codex/skills/ammo/SKILL.md
--- a/.codex/skills/ammo/SKILL.md
+++ b/.codex/skills/ammo/SKILL.md
@@ -50,6 +50,8 @@
 ## Stage 3: Candidate Proposal + Adversarial Debate

 - Spawn 2-4 ammo-champion agents
+- Champions MUST cite exact line numbers from bottleneck_analysis.md
+- Reject proposals that don't ground f-values in profiling data
 - Phase 0: Each champion independently proposes
"""
    changes = _parse_skill_diff(patch)
    assert len(changes) >= 1
    assert changes[0]["file"] == ".codex/skills/ammo/SKILL.md"
    assert len(changes[0]["added_lines"]) == 2


def test_skill_diff_parsing_researcher_file():
    """Unified diff for a researcher-guide file infers researcher_guidance section."""
    from diff_versions import _parse_skill_diff
    patch = """diff --git a/references/researcher-guide.md b/references/researcher-guide.md
--- a/references/researcher-guide.md
+++ b/references/researcher-guide.md
@@ -10,4 +10,5 @@
 Some context line
+- New rule for researchers
 - Old rule
"""
    changes = _parse_skill_diff(patch)
    assert len(changes) >= 1
    assert changes[0]["section"] == "researcher_guidance"


def test_skill_diff_parsing_impl_file():
    """Unified diff for an impl file infers implementation_guidance section."""
    from diff_versions import _parse_skill_diff
    patch = """diff --git a/impl_runner.py b/impl_runner.py
--- a/impl_runner.py
+++ b/impl_runner.py
@@ -1,3 +1,4 @@
 def run():
+    pass
     return
"""
    changes = _parse_skill_diff(patch)
    assert len(changes) >= 1
    assert changes[0]["section"] == "implementation_guidance"


def test_skill_change_to_node_mapping():
    """Skill changes map to correct DAG nodes."""
    from diff_versions import _map_skill_changes_to_nodes
    changes = [
        {"file": "SKILL.md", "section": "debate_guidance", "added_lines": ["test"], "removed_lines": []},
        {"file": "references/researcher-guide.md", "section": "researcher_guidance", "added_lines": ["test"], "removed_lines": []},
    ]
    mapping = _map_skill_changes_to_nodes(changes)
    assert "debate" in mapping
    assert "bottleneck_mining" in mapping


def test_skill_change_impl_guidance_with_node_list():
    """implementation_guidance changes map to impl_* nodes when node list is provided."""
    from diff_versions import _map_skill_changes_to_nodes
    changes = [
        {"file": "impl_runner.py", "section": "implementation_guidance", "added_lines": ["x"], "removed_lines": []},
    ]
    mapping = _map_skill_changes_to_nodes(changes, all_node_ids=["impl_op001", "impl_op002", "debate"])
    assert "impl_op001" in mapping
    assert "impl_op002" in mapping
    assert "debate" not in mapping


def test_skill_change_general_maps_to_all_nodes():
    """general section changes map to all provided node IDs."""
    from diff_versions import _map_skill_changes_to_nodes
    changes = [
        {"file": "README.md", "section": "general", "added_lines": ["x"], "removed_lines": []},
    ]
    mapping = _map_skill_changes_to_nodes(changes, all_node_ids=["debate", "baseline_capture", "bottleneck_mining"])
    for nid in ["debate", "baseline_capture", "bottleneck_mining"]:
        assert nid in mapping


def test_correlation_produces_hypotheses():
    """Correlated skill + behavioral changes produce hypotheses."""
    from diff_versions import _correlate_changes
    skill_map = {
        "debate": [{"file": "SKILL.md", "section": "debate_guidance", "added_lines": ["new rule"], "removed_lines": []}]
    }
    metric_deltas = {"debate": {"data_grounding": -0.25}}
    hypotheses = _correlate_changes(
        skill_map,
        metric_deltas,
        new_anomalies=[{"node": "debate", "type": "DATA_CORRUPTION"}],
        resolved_anomalies=[],
    )
    assert len(hypotheses) >= 1
    assert hypotheses[0]["node_id"] == "debate"
    assert "hypothesis" in hypotheses[0]


def test_correlation_high_confidence_single_change():
    """Single skill change + single behavioral regression → high confidence."""
    from diff_versions import _correlate_changes
    skill_map = {
        "debate": [{"file": "SKILL.md", "section": "debate_guidance", "added_lines": ["change"], "removed_lines": []}]
    }
    metric_deltas = {"debate": {"data_grounding": -0.5}}
    hypotheses = _correlate_changes(skill_map, metric_deltas, new_anomalies=[], resolved_anomalies=[])
    assert len(hypotheses) == 1
    assert hypotheses[0]["confidence"] == "high"


def test_correlation_no_match_when_below_threshold():
    """Metric delta below 0.1 does not produce a hypothesis."""
    from diff_versions import _correlate_changes
    skill_map = {
        "debate": [{"file": "SKILL.md", "section": "debate_guidance", "added_lines": ["x"], "removed_lines": []}]
    }
    metric_deltas = {"debate": {"data_grounding": -0.05}}
    hypotheses = _correlate_changes(skill_map, metric_deltas, new_anomalies=[], resolved_anomalies=[])
    assert len(hypotheses) == 0


def test_deep_with_skill_diff_produces_rich_report(tmp_path):
    """--deep with --skill-diff produces report with hypotheses."""
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    report = tmp_path / "regression_report.md"
    skill_diff = tmp_path / "skill.patch"
    skill_diff.write_text('''diff --git a/.codex/skills/ammo/SKILL.md b/.codex/skills/ammo/SKILL.md
--- a/.codex/skills/ammo/SKILL.md
+++ b/.codex/skills/ammo/SKILL.md
@@ -50,6 +50,7 @@
 ## Stage 3: Debate

 - Spawn champions
+- Reduced grounding requirements
''')
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out), "--deep",
         "--skill-diff", str(skill_diff),
         "--regression-report", str(report)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    diff_data = json.loads(out.read_text())
    assert diff_data.get("deep_investigation") is not None
    assert diff_data["deep_investigation"].get("hypotheses") is not None
    report_text = report.read_text()
    assert "Causal Hypotheses" in report_text


def test_deep_without_skill_diff_still_works(tmp_path):
    """--deep WITHOUT --skill-diff still produces structural report (backward compat)."""
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    report = tmp_path / "regression_report.md"
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out), "--deep",
         "--regression-report", str(report)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    diff_data = json.loads(out.read_text())
    assert diff_data["deep_investigation"]["mode"] == "structural_diff"
    # Should NOT have hypotheses key (or it should be absent / None)
    assert "hypotheses" not in diff_data["deep_investigation"]
    report_text = report.read_text()
    assert len(report_text) > 0
    # Rich sections should NOT appear
    assert "Causal Hypotheses" not in report_text


def test_deep_investigation_json_schema_with_skill_diff(tmp_path):
    """version_diff.json deep_investigation block has the required schema keys."""
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    skill_diff = tmp_path / "skill.patch"
    skill_diff.write_text('''diff --git a/.codex/skills/ammo/SKILL.md b/.codex/skills/ammo/SKILL.md
--- a/.codex/skills/ammo/SKILL.md
+++ b/.codex/skills/ammo/SKILL.md
@@ -50,6 +50,7 @@
 ## Stage 3: Debate

 - Spawn champions
+- Changed rule
''')
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out), "--deep",
         "--skill-diff", str(skill_diff)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    diff_data = json.loads(out.read_text())
    di = diff_data["deep_investigation"]
    assert "skill_changes_parsed" in di
    assert "correlations_found" in di
    assert "hypotheses" in di
    assert "uncorrelated_skill_changes" in di
    assert "uncorrelated_behavioral_changes" in di
