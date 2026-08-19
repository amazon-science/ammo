# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for L2 Circuit Board Visualization (Task #9).

Tests SVG coordinate algorithm, node-building logic, and HTML structure.
"""

import subprocess
import re
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CIRCUIT_BOARD_JS = ROOT / "frontend" / "js" / "circuit-board.js"
CAMPAIGN_APP_JS  = ROOT / "frontend" / "js" / "campaign-app.js"
INDEX_HTML       = ROOT / "frontend" / "index.html"


def run_js(script: str) -> str:
    # circuit-board.js is large (~100KB) — passing via `-e` on argv hits
    # ARG_MAX on most systems. Pipe the script via stdin instead.
    result = subprocess.run(
        ["node", "--input-type=commonjs", "-e", "eval(require('fs').readFileSync(0,'utf8'))"],
        input=script, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error: {result.stderr}")
    return result.stdout.strip()


def extract_cb(call_expr: str) -> str:
    """Extract CircuitBoard functions from circuit-board.js and call them."""
    js_src = CIRCUIT_BOARD_JS.read_text()
    script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const result = CircuitBoard.{call_expr};
if (result === null || result === undefined) {{
    console.log('');
}} else if (typeof result === 'object') {{
    console.log(JSON.stringify(result));
}} else {{
    console.log(String(result));
}}
"""
    return run_js(script)


# Sample state matching SKILL.md format
SAMPLE_STATE = {
    "target": {"model_id": "DeepSeek-R1", "hardware": "H100", "dtype": "fp8", "tp": 8},
    "stage": "4_5_parallel_tracks",
    "debate": {
        "candidates": ["flash_attn", "gemm_opt", "rope_fused"],
        "selected_winners": ["flash_attn", "rope_fused"],
        "rounds_completed": 2,
    },
    "parallel_tracks": {
        "flash_attn": {"status": "COMPLETED", "kernel_speedup": 1.35, "e2e_speedup": 1.12},
        "gemm_opt":   {"status": "FAILED", "fail_reason": "correctness fail"},
        "rope_fused": {"status": "IN_PROGRESS"},
    },
    "integration": {"status": "pending", "passing_candidates": []},
    "campaign": {
        "status": "active",
        "current_round": 1,
        "cumulative_e2e_speedup": 1.12,
        "shipped_optimizations": ["flash_attn"],
        "rounds": [],
    },
}


# ---------------------------------------------------------------------------
# 1. SVG Coordinate Algorithm
# ---------------------------------------------------------------------------

class TestSvgCoordinates:
    """Coordinate functions exported via CircuitBoard global."""

    def test_tierY_increases_with_row(self):
        """tierY(row) must increase as row increases."""
        y0 = float(extract_cb("tierY(0)"))
        y1 = float(extract_cb("tierY(1)"))
        y3 = float(extract_cb("tierY(3)"))
        assert y1 > y0
        assert y3 > y1

    def test_tierY_row0_is_positive(self):
        y0 = float(extract_cb("tierY(0)"))
        assert y0 > 0

    def test_tierY_consistent_spacing(self):
        """Consecutive rows have equal spacing."""
        y0 = float(extract_cb("tierY(0)"))
        y1 = float(extract_cb("tierY(1)"))
        y2 = float(extract_cb("tierY(2)"))
        assert abs((y1 - y0) - (y2 - y1)) < 0.01

    def test_trackX_single_centered(self):
        # Single track at center 400: center - TRACK_W/2 = 400 - 100 = 300
        result = float(extract_cb("trackX(0, 1, 400)"))
        assert result == pytest.approx(300.0)

    def test_trackX_two_tracks_second_greater_than_first(self):
        x0 = float(extract_cb("trackX(0, 2, 400)"))
        x1 = float(extract_cb("trackX(1, 2, 400)"))
        assert x1 > x0

    def test_trackX_symmetric(self):
        """For 2 tracks, first track and second track are symmetric around centerX."""
        x0 = float(extract_cb("trackX(0, 2, 400)"))
        x1 = float(extract_cb("trackX(1, 2, 400)"))
        # Center of first track = x0 + TRACK_W/2, center of second = x1 + TRACK_W/2
        # Their midpoint should be near centerX (400)
        mid = (x0 + x1) / 2 + 100  # + TRACK_W/2 center correction... actually just check symmetry
        # x0 = 400 - total/2; x1 = x0 + (TRACK_W + H_GAP)
        # centers: c0 = x0 + 100, c1 = x1 + 100; midpoint = (c0+c1)/2 = x0+x1+200)/2
        assert abs((x0 + x1 + 200) / 2 - 400) < 0.1

    def test_rightAngleTrace_format(self):
        result = extract_cb("rightAngleTrace(100, 100, 300, 300)")
        assert result.startswith("M 100 100"), f"unexpected trace: {result}"
        assert "L 300 300" in result

    def test_rightAngleTrace_midpoint_bend(self):
        """Trace must bend at the vertical midpoint between ay and by."""
        result = extract_cb("rightAngleTrace(0, 0, 200, 200)")
        # mid = 100
        assert "L 0 100" in result
        assert "L 200 100" in result

    def test_rightAngleTrace_straight_horizontal(self):
        """Same y: path should be a straight line (no bend)."""
        result = extract_cb("rightAngleTrace(50, 100, 150, 100)")
        # mid = 100; M 50 100 L 50 100 L 150 100 L 150 100
        assert "L 50 100" in result
        assert "L 150 100" in result


# ---------------------------------------------------------------------------
# 2. Node Building Logic
# ---------------------------------------------------------------------------

class TestBuildL2Nodes:
    """buildL2Nodes(state) extracts nodes for current-round rendering."""

    def _run_build(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const nodes = CircuitBoard.buildL2Nodes(state);
console.log(JSON.stringify(nodes));
"""
        return json.loads(run_js(script))

    def test_baseline_node_present(self):
        nodes = self._run_build(SAMPLE_STATE)
        tiers = {n["tier"] for n in nodes}
        assert 0 in tiers, "baseline tier (0) missing"

    def test_debate_candidates_become_nodes(self):
        nodes = self._run_build(SAMPLE_STATE)
        debate_nodes = [n for n in nodes if n["tier"] == 2]
        assert len(debate_nodes) == 3, "expected 3 debate candidate nodes"

    def test_parallel_tracks_become_nodes(self):
        nodes = self._run_build(SAMPLE_STATE)
        track_nodes = [n for n in nodes if n["tier"] == 3]
        assert len(track_nodes) == 3, "expected 3 parallel track nodes"

    def test_shipped_node_has_shipped_status(self):
        nodes = self._run_build(SAMPLE_STATE)
        track_nodes = [n for n in nodes if n["tier"] == 3]
        flash_node = next((n for n in track_nodes if n["id"] == "flash_attn"), None)
        assert flash_node is not None
        assert flash_node["status"] == "shipped"

    def test_failed_node_has_failed_status(self):
        nodes = self._run_build(SAMPLE_STATE)
        track_nodes = [n for n in nodes if n["tier"] == 3]
        gemm_node = next((n for n in track_nodes if n["id"] == "gemm_opt"), None)
        assert gemm_node is not None
        assert gemm_node["status"] == "failed"

    def test_active_node_has_active_status(self):
        nodes = self._run_build(SAMPLE_STATE)
        track_nodes = [n for n in nodes if n["tier"] == 3]
        rope_node = next((n for n in track_nodes if n["id"] == "rope_fused"), None)
        assert rope_node is not None
        assert rope_node["status"] == "active"

    def test_integration_node_present(self):
        nodes = self._run_build(SAMPLE_STATE)
        tiers = {n["tier"] for n in nodes}
        assert 5 in tiers, "integration tier (5) missing"

    def test_empty_state_returns_baseline(self):
        nodes = self._run_build({})
        tiers = [n["tier"] for n in nodes]
        assert 0 in tiers

    def test_display_name_labels_track_node_id_stays_op_id(self):
        """selected_candidates[].name relabels the node; the id stays the op_id
        (the shipped-set and navigation key). Ops without a name keep the raw
        op_id label."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "debate": {
                "candidates": ["flash_attn", "gemm_opt", "rope_fused"],
                "selected_winners": ["flash_attn", "rope_fused"],
                "selected_candidates": [
                    {"op_id": "flash_attn", "name": "fused flash attention"},
                    {"op_id": "rope_fused", "name": "  "},  # blank → fallback
                ],
            },
            "parallel_tracks": {
                "tracks": {
                    "flash_attn": {"status": "COMPLETED"},
                    "gemm_opt": {"status": "FAILED"},
                    "rope_fused": {"status": "IN_PROGRESS"},
                },
            },
        }]
        nodes = self._run_build(state)
        track_nodes = {n["id"]: n for n in nodes if n["tier"] == 3}
        assert track_nodes["flash_attn"]["label"] == "fused flash attention"
        assert track_nodes["flash_attn"]["status"] == "shipped"
        assert track_nodes["rope_fused"]["label"] == "rope_fused"
        assert track_nodes["gemm_opt"]["label"] == "gemm_opt"
        debate_labels = {n["label"] for n in nodes if n["tier"] == 2}
        assert "fused flash attention" in debate_labels


# ---------------------------------------------------------------------------
# 2b. auditGateStates(state) — audit-gate "fuse hex" state derivation
# ---------------------------------------------------------------------------

class TestAuditGateStates:
    """auditGateStates(state) derives bypass/pending/running/passed/escalated
    per round per gate (stage_1/stage_2/stage_45/stage_67)."""

    def _run_audit(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const gates = CircuitBoard.auditGateStates(state);
console.log(JSON.stringify(gates));
"""
        return json.loads(run_js(script))

    def test_legacy_round_without_audit_key_is_bypass(self):
        """Round has no `audit` key at all — zero change from today's plain via."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {"completed_at": "2026-01-01T00:00:00Z"},
            "parallel_tracks": {
                "tracks": {
                    "flash_attn": {"status": "COMPLETED"},
                    "rope_fused": {"status": "COMPLETED"},
                },
            },
            "integration": {"status": "completed"},
        }]
        gates = self._run_audit(state)
        round_gates = gates["1"]
        for key in ("stage_1", "stage_2", "stage_45", "stage_67"):
            assert round_gates[key]["state"] == "bypass"

    def test_started_at_without_passed_at_is_running_and_cycle_passes_through(self):
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {}},
            "integration": {"status": "pending"},
            "audit": {
                "stage_2": {"started_at": "2026-01-02T00:00:00Z", "cycle": 2},
            },
        }]
        gates = self._run_audit(state)
        gate = gates["1"]["stage_2"]
        assert gate["state"] == "running"
        assert gate["cycle"] == 2

    def test_heuristic_running_when_stage_complete_but_started_at_missing(self):
        """No started_at, audit key present, audited stage terminal, round is
        current -> heuristic fallback treats the auditor as already running."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {},
            "bottleneck_mining": {},
            "parallel_tracks": {
                "tracks": {
                    "flash_attn": {"status": "COMPLETED"},
                    "gemm_opt": {"status": "FAILED"},
                },
            },
            "integration": {"status": "pending"},
            "audit": {},
        }]
        gates = self._run_audit(state)
        gate = gates["1"]["stage_45"]
        assert gate["state"] == "running"
        assert gate["started_at"] is None

    def test_passed_at_set_is_passed(self):
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {}},
            "integration": {"status": "pending"},
            "audit": {
                "stage_1": {"started_at": "2026-01-01T01:00:00Z", "passed_at": "2026-01-01T02:00:00Z"},
            },
        }]
        gates = self._run_audit(state)
        gate = gates["1"]["stage_1"]
        assert gate["state"] == "passed"

    def test_matching_escalation_marks_stage_45_escalated(self):
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["auditor_escalation"] = {"stage": "4_5_parallel_tracks", "round": 1}
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {},
            "bottleneck_mining": {},
            "parallel_tracks": {
                "tracks": {
                    "flash_attn": {"status": "COMPLETED"},
                },
            },
            "integration": {"status": "pending"},
            "audit": {},
        }]
        gates = self._run_audit(state)
        gate = gates["1"]["stage_45"]
        assert gate["state"] == "escalated"
        # sibling gates on the same round are unaffected by the stage_45 escalation
        assert gates["1"]["stage_1"]["state"] != "escalated"

    def test_future_gate_when_upstream_stage_not_run_is_pending(self):
        """stage_2's audited stage (bottleneck_mining) hasn't completed yet —
        pending, not running, even though the audit key is present."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {}},
            "integration": {"status": "pending"},
            "audit": {},
        }]
        gates = self._run_audit(state)
        gate = gates["1"]["stage_2"]
        assert gate["state"] == "pending"

    def test_passed_at_outranks_a_matching_escalation(self):
        """A track-scoped escalation quarantines tracks, the surviving scope is
        re-audited and may stamp PASS. Nothing ever clears auditor_escalation,
        so escalation-first would keep a passed gate red for the whole round."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["auditor_escalation"] = {
            "stage": "4_5_parallel_tracks", "round": 1, "scope": ["gemm_opt"],
        }
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {"flash_attn": {"status": "PASS"}}},
            "integration": {"status": "pending"},
            "audit": {
                "stage_45": {
                    "started_at": "2026-01-01T04:00:00Z",
                    "cycle": 4,
                    "passed_at": "2026-01-01T05:00:00Z",
                },
            },
        }]
        gates = self._run_audit(state)
        assert gates["1"]["stage_45"]["state"] == "passed"

    def test_escalation_without_passed_at_is_still_escalated(self):
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["auditor_escalation"] = {"stage": "4_5_parallel_tracks", "round": 1}
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {"flash_attn": {"status": "PASS"}}},
            "integration": {"status": "pending"},
            "audit": {"stage_45": {"started_at": "2026-01-01T04:00:00Z", "cycle": 4}},
        }]
        gates = self._run_audit(state)
        assert gates["1"]["stage_45"]["state"] == "escalated"

    def test_legacy_stage_6_passed_at_resolves_the_stage_67_gate(self):
        """ammo_state.py accepts stage_6/stage_7 as the S67 PASS, so a fully
        audited pre-consolidation round must not render as running/pending."""
        for legacy_key in ("stage_6", "stage_7"):
            state = json.loads(json.dumps(SAMPLE_STATE))
            state["campaign"]["current_round"] = 1
            state["campaign"]["rounds"] = [{
                "round_id": 1,
                "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
                "bottleneck_mining": {"completed_at": "2026-01-01T00:00:00Z"},
                "parallel_tracks": {"tracks": {"flash_attn": {"status": "PASS"}}},
                "integration": {"status": "completed"},
                "audit": {
                    legacy_key: {"passed_at": "2026-01-01T06:00:00Z"},
                },
            }]
            gates = self._run_audit(state)
            gate = gates["1"]["stage_67"]
            assert gate["state"] == "passed", legacy_key
            assert gate["passed_at"] == "2026-01-01T06:00:00Z"

    def test_post_ship_stage_1_exemption_is_not_guessed_as_running(self):
        """ammo_state.py drops the same-round stage_1 requirement on round N>1
        whose predecessor carries an audit key, so no auditor is dispatched and
        the heuristic would animate a live audit forever."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 2
        state["campaign"]["rounds"] = [
            {
                "round_id": 1,
                "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
                "bottleneck_mining": {"completed_at": "2026-01-01T01:00:00Z"},
                "parallel_tracks": {"tracks": {"flash_attn": {"status": "PASS"}}},
                "integration": {"status": "completed"},
                "audit": {
                    "stage_67": {
                        "started_at": "2026-01-01T05:00:00Z",
                        "passed_at": "2026-01-01T06:00:00Z",
                    },
                },
            },
            {
                "round_id": 2,
                "baseline": {"completed_at": "2026-01-02T00:00:00Z"},
                "bottleneck_mining": {},
                "parallel_tracks": {"tracks": {}},
                "integration": {"status": "pending"},
                "audit": {},
            },
        ]
        gates = self._run_audit(state)
        assert gates["2"]["stage_1"]["state"] == "pending"

    def test_stage_1_heuristic_still_fires_on_round_1(self):
        """Round 1 has no predecessor, so nothing exempts its stage_1 gate."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {},
            "parallel_tracks": {"tracks": {}},
            "integration": {"status": "pending"},
            "audit": {},
        }]
        assert self._run_audit(state)["1"]["stage_1"]["state"] == "running"

    def test_duplicate_round_entries_resolve_to_the_richest_entry(self):
        """parseRounds dedups duplicate round_id entries; auditGateStates must
        pick the SAME entry or the row and its gates come from different data."""
        rich = {
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {"completed_at": "2026-01-01T01:00:00Z"},
            "parallel_tracks": {"tracks": {"flash_attn": {"status": "PASS"}}},
            "integration": {"status": "completed"},
            "shipped": ["op-001"],
            "audit": {
                k: {"started_at": "2026-01-01T02:00:00Z", "passed_at": "2026-01-01T03:00:00Z"}
                for k in ("stage_1", "stage_2", "stage_45", "stage_67")
            },
        }
        thin = {"round_id": 1, "audit": {}}
        for rounds in ([rich, thin], [thin, rich]):
            state = json.loads(json.dumps(SAMPLE_STATE))
            state["campaign"]["current_round"] = 2
            state["campaign"]["rounds"] = json.loads(json.dumps(rounds))
            gates = self._run_audit(state)
            for key in ("stage_1", "stage_2", "stage_45", "stage_67"):
                assert gates["1"][key]["state"] == "passed", (key, rounds.index(thin))


# ---------------------------------------------------------------------------
# 3. nodeStatusClass helper (3-arg signature via nodeStatusClassFull)
# ---------------------------------------------------------------------------

class TestNodeStatusClass:
    """nodeStatusClassFull(opId, track, shippedOps[]) maps to CSS class."""

    def test_shipped_op_returns_shipped(self):
        result = extract_cb("nodeStatusClassFull('op1', {status: 'COMPLETED'}, ['op1'])")
        assert result == "shipped"

    def test_failed_track_returns_failed(self):
        result = extract_cb("nodeStatusClassFull('op1', {status: 'FAILED'}, [])")
        assert result == "failed"

    def test_in_progress_returns_active(self):
        result = extract_cb("nodeStatusClassFull('op1', {status: 'IN_PROGRESS'}, [])")
        assert result == "active"

    def test_completed_not_in_shipped_returns_active(self):
        result = extract_cb("nodeStatusClassFull('op1', {status: 'COMPLETED'}, [])")
        assert result == "active"


# ---------------------------------------------------------------------------
# 4. HTML structure
# ---------------------------------------------------------------------------

class TestHtmlStructure:
    """L2 template in index.html has required elements."""

    @pytest.fixture(autouse=True)
    def html(self):
        self._html = INDEX_HTML.read_text()

    def test_l2_view_references_circuit_board(self):
        """L2 view must call renderCircuitBoard or reference CircuitBoard."""
        assert re.search(r'renderCircuitBoard|circuit.?board', self._html, re.IGNORECASE), \
            "L2 template missing renderCircuitBoard call"

    def test_l2_sidebar_container_present(self):
        """L2 view must have the cb-mount container where buildSidebar() appends the sidebar."""
        # The sidebar is built imperatively by circuit-board.js buildSidebar()
        # and appended into #cb-mount by renderCircuitBoard().
        assert 'cb-mount' in self._html, \
            "L2 view missing #cb-mount container (renderCircuitBoard target)"

    def test_circuit_board_js_loaded(self):
        assert "circuit-board.js" in self._html, "index.html missing circuit-board.js script"

    def test_l2_placeholder_replaced(self):
        """The 'implemented in Task 9' placeholder must be gone."""
        assert "implemented in Task 9" not in self._html, \
            "L2 placeholder comment still present — replace with actual circuit board"


# ---------------------------------------------------------------------------
# 5. Edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesL2:
    """Edge cases not covered by implementor tests."""

    def _run_build(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const nodes = CircuitBoard.buildL2Nodes(state);
console.log(JSON.stringify(nodes));
"""
        return json.loads(run_js(script))

    def test_debate_candidates_inferred_from_tracks_when_empty(self):
        """When debate.candidates is empty, debate nodes inferred from parallel_tracks keys."""
        state = {
            "debate": {"candidates": [], "selected_winners": []},
            "parallel_tracks": {
                "op_x": {"status": "IN_PROGRESS"},
                "op_y": {"status": "FAILED"},
            },
            "integration": {"status": "pending"},
            "campaign": {"shipped_optimizations": []},
        }
        nodes = self._run_build(state)
        debate_nodes = [n for n in nodes if n["tier"] == 2]
        debate_labels = {n["label"] for n in debate_nodes}
        assert "op_x" in debate_labels or "op_y" in debate_labels, \
            "debate candidates should be inferred from parallel_tracks keys when debate.candidates is empty"

    def test_completed_in_shipped_returns_shipped(self):
        """COMPLETED status + op in shippedOps → 'shipped' (not 'active')."""
        result = extract_cb("nodeStatusClassFull('op1', {status: 'COMPLETED'}, ['op1'])")
        assert result == "shipped", \
            "COMPLETED track that IS shipped should return 'shipped', not 'active'"

    def test_rightAngleTrace_negative_coords(self):
        """rightAngleTrace should handle negative coordinates without crashing."""
        result = extract_cb("rightAngleTrace(-50, -100, 50, 100)")
        assert result.startswith("M -50 -100"), f"unexpected: {result}"
        assert "L 50 100" in result


# ---------------------------------------------------------------------------
# 6. Round label semantics: campaign.current_round is authoritative
# ---------------------------------------------------------------------------

class TestRoundLabelSemantics:
    """parseRounds must label the current row from campaign.current_round ONLY.

    Regression guard for the bug where FE conflated debate.rounds_completed
    (a Stage-3 sub-round counter 0..debate.max_rounds) with the campaign
    round counter, mis-labeling an R1 session as "ROUND 2".
    """

    def _run_parse(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const rows = CircuitBoard.parseRounds(state);
console.log(JSON.stringify(rows));
"""
        return json.loads(run_js(script))

    def test_parseRounds_uses_campaign_current_round_not_debate_subrounds(self):
        """Active R1 campaign with debate sub-round 2/4 must render as round 1, not round 2."""
        state = {
            "stage": "4_5_parallel_tracks",
            "debate": {"rounds_completed": 2, "max_rounds": 4},
            "parallel_tracks": {
                "C1": {"status": "IN_PROGRESS"},
                "C3": {"status": "IN_PROGRESS"},
            },
            "campaign": {"current_round": 1, "rounds": []},
        }
        rows = self._run_parse(state)
        assert len(rows) == 1, f"expected 1 synthetic current row, got {len(rows)}"
        assert rows[0]["roundId"] == 1, (
            f"roundId must come from campaign.current_round (1), not debate.rounds_completed (2); "
            f"got {rows[0]['roundId']}"
        )
        assert rows[0]["isCurrent"] is True

    def test_parseRounds_current_row_matches_archived_round_id(self):
        """When campaign.rounds has entries, the current row is selected by round_id == current_round."""
        state = {
            "stage": "4_5_parallel_tracks",
            "debate": {"rounds_completed": 3, "max_rounds": 4},
            "parallel_tracks": {"C1": {"status": "IN_PROGRESS"}},
            "campaign": {
                "current_round": 2,
                "rounds": [
                    {"round_id": 1, "selected_candidates": ["op1"]},
                    {"round_id": 2, "selected_candidates": []},
                ],
            },
        }
        rows = self._run_parse(state)
        current = [r for r in rows if r["isCurrent"]]
        assert len(current) == 1
        assert current[0]["roundId"] == 2, (
            f"current row must match campaign.current_round=2, got {current[0]['roundId']}"
        )

    def test_no_mathmax_anti_pattern_in_circuit_board(self):
        """Regression guard: the Math.max(current_round, rounds_completed) anti-pattern must not return."""
        src = CIRCUIT_BOARD_JS.read_text()
        assert "Math.max(campaign.current_round" not in src, (
            "circuit-board.js re-introduced Math.max(campaign.current_round, debate.rounds_completed) — "
            "this conflates Stage-3 sub-rounds with campaign round"
        )

    def test_no_mathmax_anti_pattern_in_campaign_app(self):
        """Regression guard for _stageReachable and other callsites in campaign-app.js."""
        src = CAMPAIGN_APP_JS.read_text()
        # Pattern matches both single-line and multi-line forms of the anti-pattern.
        assert not re.search(
            r"Math\.max\(\s*campaign\.current_round[^,]*,\s*\(?state\.debate",
            src,
        ), (
            "campaign-app.js re-introduced Math.max(campaign.current_round, state.debate.rounds_completed)"
        )

    def test_no_mathmax_anti_pattern_in_index_html(self):
        """Regression guard for the L2 header x-text expression."""
        src = INDEX_HTML.read_text()
        assert "campaignState?.debate?.rounds_completed" not in src or \
               "Math.max(campaignState?.campaign?.current_round" not in src, (
            "index.html still uses Math.max(campaignState?.campaign?.current_round, "
            "campaignState?.debate?.rounds_completed) for the round label"
        )


# ---------------------------------------------------------------------------
# 8. Current-row inter-stage trace color must reflect per-stage status
# ---------------------------------------------------------------------------

class TestCurrentRowTraceColors:
    """Traces between stage chips on the current (active) row must color per stage.

    Regression guard for the bug where EVERY inter-stage trace on the active row
    rendered cyan (#00f3ff, url(#cb2-gC)), even for stages that were already
    completed. Correct semantic: the trace into stage si is cyan only when
    stage si IS the active stage; otherwise (si already past) it is green
    (#00ffb2, url(#cb2-gM)).
    """

    def test_no_round_active_trace_color_anti_pattern(self):
        """Source must not paint the inter-stage trace on round.active alone.

        The exact anti-pattern that produced the bug was:
            const traceColor = round.active ? '#00f3ff' : '#00ffb2';
            const tf         = round.active ? 'url(#cb2-gC)' : 'url(#cb2-gM)';
        These condition the color on the ROUND being active, not the incoming
        STAGE being active, so past stages on the live row glow cyan.

        Scoped to the `traceColor` / `tf` assignments inside the inter-stage
        loop; the branch-via junction dot (`drawVia(... branchX, stackCY ...)`)
        legitimately tracks round.active since it marks the fan-out node at
        the entry to the currently-running parallel-tracks stage.
        """
        src = CIRCUIT_BOARD_JS.read_text()
        assert not re.search(
            r"const\s+traceColor\s*=\s*round\.active\s*\?\s*'#00f3ff'\s*:\s*'#00ffb2'",
            src,
        ), (
            "circuit-board.js still paints inter-stage traces with "
            "`const traceColor = round.active ? '#00f3ff' : '#00ffb2'` — "
            "this lights up ALL traces on the active row cyan, even for "
            "completed stages. Condition on the incoming stage's active "
            "status instead."
        )
        assert not re.search(
            r"const\s+tf\s*=\s*round\.active\s*\?\s*'url\(#cb2-gC\)'\s*:\s*'url\(#cb2-gM\)'",
            src,
        ), (
            "circuit-board.js still selects inter-stage trace glow filter "
            "with `const tf = round.active ? 'url(#cb2-gC)' : 'url(#cb2-gM)'` "
            "— same bug as the color: the glow filter must follow per-stage "
            "status."
        )

    def _run_parse(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const rows = CircuitBoard.parseRounds(state);
console.log(JSON.stringify(rows));
"""
        return json.loads(run_js(script))

    def test_parseRounds_marks_past_stages_non_active_on_current_row(self):
        """On the current row, stages before currentStageIdx must report a
        non-active status (shipped/completed), so the render path can color
        inter-stage traces correctly.

        Repro: stage=4_5_parallel_tracks (idx 3). Stages 0,1,2 are past; stage
        3 is active. Anything past must NOT carry status=='active', else the
        trace-color conditional would paint their incoming traces cyan.
        """
        state = {
            "stage": "4_5_parallel_tracks",
            "debate": {"rounds_completed": 2, "max_rounds": 4},
            "parallel_tracks": {
                "C1": {"status": "IN_PROGRESS"},
                "C3": {"status": "IN_PROGRESS"},
            },
            "campaign": {"current_round": 1, "rounds": []},
        }
        rows = self._run_parse(state)
        assert len(rows) == 1 and rows[0]["isCurrent"] is True
        nodes = rows[0]["stageNodes"]
        # Stage 3 (Implement) is the active stage in this repro.
        assert nodes[3]["status"] == "active", (
            f"Implement stage must be active; got {nodes[3]['status']}"
        )
        for i in (0, 1, 2):
            assert nodes[i]["status"] != "active", (
                f"Stage {i} ({nodes[i]['label']}) must NOT be active on a row "
                f"whose stage is 4_5_parallel_tracks (idx 3); got "
                f"status={nodes[i]['status']}. If this stage is flagged "
                f"active, the inter-stage trace leading into it would be "
                f"painted cyan even though the stage is already done."
            )


# ---------------------------------------------------------------------------
# 9. Trace extent must not exceed current stage (no forward-stage traces)
# ---------------------------------------------------------------------------

class TestTraceExtentGating:
    """Current-row traces must not extend past the currently-reached stage.

    Repro: campaign is at stage `4_5_parallel_tracks` with two IN_PROGRESS
    tracks (C1, C3) and NO integration + NO eval. The row already renders the
    completed BASELINE→MINING→DEBATE chain and the fan-out traces into the
    two track chips — those end at the track chips' left edges (x≈820).
    The bug: `buildBoard` then draws one more long horizontal trace from the
    track block's right edge (x≈1094) all the way to the EVAL column
    (x≈1354), even though the campaign has not reached the EVAL stage yet.

    The fix: only draw the track → EVAL forward trace when there is an actual
    destination (an eval verdict or an integration chip). Otherwise terminate
    the row at the track block's right edge.
    """

    def _render_and_capture_paths(self, state_dict):
        """Run `renderCircuitBoard` with a minimal DOM mock and return every
        `d` attribute set on a path element, plus every point passed through
        `drawTrace`. Returns a dict: { 'paths': [...], 'traces': [...] }.
        """
        js_src = CIRCUIT_BOARD_JS.read_text()
        # A DOM mock that tracks nodeName, setAttribute calls, and allows
        # classList / appendChild / addEventListener / style / innerHTML /
        # textContent. Enough to get through buildBoard() without crashing.
        script = r"""
const _capturedPathDs = [];
function _mkEl(nodeName, ns) {
    const attrs = {};
    const el = {
        nodeName, namespaceURI: ns || null, tagName: nodeName,
        style: new Proxy({}, { set: (t, k, v) => { t[k] = v; return true; } }),
        className: '', innerHTML: '', textContent: '',
        classList: { add(){}, remove(){}, contains(){ return false; }, toggle(){} },
        children: [], childNodes: [], attributes: attrs,
        parentNode: null,
        dataset: {},
        appendChild(c) { this.children.push(c); this.childNodes.push(c); if (c) c.parentNode = this; return c; },
        removeChild(c) { return c; },
        insertBefore(n) { this.children.push(n); return n; },
        addEventListener() {}, removeEventListener() {},
        setAttribute(k, v) {
            attrs[k] = v;
            if (nodeName === 'path' && k === 'd') _capturedPathDs.push(String(v));
        },
        getAttribute(k) { return attrs[k]; },
        removeAttribute(k) { delete attrs[k]; },
        querySelector() { return _mkEl('div'); },
        querySelectorAll() { return []; },
        focus() {},
        contains() { return false; },
        getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
    };
    return el;
}
const document = {
    createElement(tag) { return _mkEl(tag); },
    createElementNS(ns, tag) { return _mkEl(tag, ns); },
    addEventListener() {}, removeEventListener() {},
    body: _mkEl('body'),
};
const window = { innerWidth: 1920, innerHeight: 1080, CircuitBoard: null };
// Ensure setTimeout is available and non-blocking; node provides it natively.
"""
        script += js_src + "\n"
        script += (
            "const state = " + json.dumps(state_dict) + ";\n"
            "const container = _mkEl('div');\n"
            "try { renderCircuitBoard(container, state, null, null, null, null); } "
            "catch (e) { console.error('RENDER ERROR:', e.message, e.stack); process.exit(2); }\n"
            "console.log(JSON.stringify({ paths: _capturedPathDs }));\n"
        )
        out = run_js(script)
        return json.loads(out)

    # Live-repro state: campaign is mid-IMPL with two IN_PROGRESS tracks,
    # no integration, no eval. Only the tracks→eval forward trace should be
    # suppressed; everything else (inter-stage, fan-out) must still render.
    LIVE_REPRO_STATE = {
        "target": {"model_id": "Qwen3", "hardware": "H100", "dtype": "fp8", "tp": 8},
        "stage": "4_5_parallel_tracks",
        "debate": {"rounds_completed": 3, "max_rounds": 3,
                   "candidates": ["C1", "C2", "C3"],
                   "selected_winners": ["C1", "C3"]},
        "parallel_tracks": {
            "C1": {"status": "IN_PROGRESS"},
            "C3": {"status": "IN_PROGRESS"},
        },
        "integration": {"status": "pending", "passing_candidates": []},
        "campaign": {
            "status": "active",
            "current_round": 1,
            "cumulative_e2e_speedup": 1.0,
            "shipped_optimizations": [],
            "rounds": [],
        },
    }

    @staticmethod
    def _path_bbox(d: str):
        """Extract min/max X coords from an SVG path `d` attribute built by
        buildPathD(): `M x,y L x,y L x,y ...`."""
        # Coords are comma-separated "x,y" tokens; extract the x's.
        xs = []
        for tok in re.findall(r'[-+]?\d+(?:\.\d+)?,[-+]?\d+(?:\.\d+)?', d):
            xs.append(float(tok.split(',')[0]))
        return (min(xs), max(xs)) if xs else (0, 0)

    def test_no_trace_segment_past_track_block_when_mid_impl(self):
        """Forward trace into EVAL column must NOT render when the current
        stage is 4_5_parallel_tracks with IN_PROGRESS tracks, no integration,
        no eval. Concretely: no path may have xMax >= 1094 (the x where the
        bogus `M1094,138 L1354,138` trace starts)."""
        result = self._render_and_capture_paths(self.LIVE_REPRO_STATE)
        paths = result["paths"]
        assert paths, "expected some trace paths to be rendered"
        # Gather every (xMin, xMax) from each path. The legitimate traces on
        # R1 in this state all end by x≈1090 (track-chip right edge at
        # trackStartX + TRACK_W = 830 + 260 = 1090; fan-out traces end ~10px
        # before each track's left edge at ~820).
        # Any trace whose xMax exceeds 1090 is the bogus forward connector.
        FORWARD_X_THRESHOLD = 1090
        offenders = []
        for d in paths:
            xmin, xmax = self._path_bbox(d)
            if xmax > FORWARD_X_THRESHOLD:
                offenders.append((d, xmin, xmax))
        assert not offenders, (
            "No trace segment may extend past x={} (track-block right edge) "
            "when the campaign is mid-IMPL with no integration and no eval. "
            "Offenders:\n{}".format(
                FORWARD_X_THRESHOLD,
                "\n".join(f"  d={d!r} xMin={xi} xMax={xa}" for d, xi, xa in offenders)
            )
        )

    def test_forward_trace_renders_when_integration_present(self):
        """Control: when integration exists, the integration→eval forward trace
        IS expected to render (legitimate). This guards against overshooting
        the fix — we must not suppress the forward trace universally."""
        state = dict(self.LIVE_REPRO_STATE)
        state = json.loads(json.dumps(state))  # deep copy
        state["integration"] = {
            "status": "in_progress", "passing_candidates": [], "failed_candidates": [],
        }
        state["stage"] = "6_integration"
        result = self._render_and_capture_paths(state)
        paths = result["paths"]
        # At stage 6_integration with integration in-progress, we expect the
        # integration chip → eval column trace to render (at ~x=1183→1354).
        has_forward = any(self._path_bbox(d)[1] > 1200 for d in paths)
        assert has_forward, (
            "expected a forward trace from integration chip into EVAL column "
            "when stage=6_integration with integration in progress"
        )

    def test_forward_trace_renders_when_eval_verdict_present(self):
        """Control: when tracks are done + eval verdict is set, the forward
        trace from the track block to the eval diamond IS expected to render.
        Suppression must gate on *destination absence*, not on track status."""
        state = json.loads(json.dumps(self.LIVE_REPRO_STATE))
        state["parallel_tracks"] = {
            "C1": {"status": "COMPLETED", "verdict": "PASS", "kernel_speedup": 1.5},
        }
        state["stage"] = "7_campaign_eval"
        state["campaign"]["shipped_optimizations"] = ["C1"]
        state["campaign"]["status"] = "completed"
        # parseRounds at stage 7 sets mockRound.eval based on campaign status.
        result = self._render_and_capture_paths(state)
        paths = result["paths"]
        has_forward = any(self._path_bbox(d)[1] > 1200 for d in paths)
        assert has_forward, (
            "expected a forward trace into EVAL column when eval verdict is "
            "set (stage=7_campaign_eval)"
        )


# ---------------------------------------------------------------------------
# 9b. Audit-hex render gating — drawAuditHex call sites
# ---------------------------------------------------------------------------

class TestAuditHexRenderGating:
    """The derivation layer can be right while the render layer never draws it.

    `auditGateStates` is pure, so these cases go through `renderCircuitBoard`
    and log every `drawAuditHex` call plus every `drawVia` coordinate.
    """

    def _render_and_capture_hexes(self, state_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = r"""
const _hexCalls = [];
const _viaCalls = [];
function _mkEl(nodeName, ns) {
    const attrs = {};
    const el = {
        nodeName, namespaceURI: ns || null, tagName: nodeName,
        style: new Proxy({}, { set: (t, k, v) => { t[k] = v; return true; } }),
        className: '', innerHTML: '', textContent: '',
        classList: { add(){}, remove(){}, contains(){ return false; }, toggle(){} },
        children: [], childNodes: [], attributes: attrs, parentNode: null,
        dataset: {},
        appendChild(c) { this.children.push(c); this.childNodes.push(c); if (c) c.parentNode = this; return c; },
        removeChild(c) { return c; },
        insertBefore(n) { this.children.push(n); return n; },
        addEventListener() {}, removeEventListener() {},
        setAttribute(k, v) { attrs[k] = v; },
        getAttribute(k) { return attrs[k]; },
        removeAttribute(k) { delete attrs[k]; },
        querySelector() { return _mkEl('div'); },
        querySelectorAll() { return []; },
        focus() {}, contains() { return false; },
        getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; },
    };
    return el;
}
const document = {
    createElement(tag) { return _mkEl(tag); },
    createElementNS(ns, tag) { return _mkEl(tag, ns); },
    addEventListener() {}, removeEventListener() {},
    body: _mkEl('body'),
};
const window = { innerWidth: 1920, innerHeight: 1080, CircuitBoard: null };
"""
        script += js_src + "\n"
        script += (
            "const _realHex = drawAuditHex;\n"
            "drawAuditHex = function (canvas, svg, x, y, gate, gateKey, roundId, delay, cb) {\n"
            "    _hexCalls.push({ gateKey, state: gate && gate.state, x: Math.round(x), roundId });\n"
            "    return _realHex.apply(null, arguments);\n"
            "};\n"
            "const _realVia = drawVia;\n"
            "drawVia = function (svg, x, y, color, delay) {\n"
            "    _viaCalls.push({ x: Math.round(x), y: Math.round(y) });\n"
            "    return _realVia.apply(null, arguments);\n"
            "};\n"
            "const state = " + json.dumps(state_dict) + ";\n"
            "const container = _mkEl('div');\n"
            "try { renderCircuitBoard(container, state, null, null, null, null, null); } "
            "catch (e) { console.error('RENDER ERROR:', e.message, e.stack); process.exit(2); }\n"
            "console.log(JSON.stringify({ hexes: _hexCalls, vias: _viaCalls }));\n"
        )
        return json.loads(run_js(script))

    def _s45_running_state(self, integration_status="pending"):
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 1
        state["campaign"]["current_stage"] = "4_5_parallel_tracks"
        state["campaign"]["rounds"] = [{
            "round_id": 1,
            "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
            "bottleneck_mining": {"completed_at": "2026-01-01T01:00:00Z"},
            "parallel_tracks": {"tracks": {"C1": {"status": "PASS"}}},
            "integration": {"status": integration_status},
            "audit": {
                "stage_1": {"started_at": "2026-01-01T00:30:00Z", "passed_at": "2026-01-01T00:40:00Z"},
                "stage_2": {"started_at": "2026-01-01T01:10:00Z", "passed_at": "2026-01-01T01:20:00Z"},
                "stage_45": {"started_at": "2026-01-01T02:00:00Z", "cycle": 1},
            },
        }]
        return state

    def test_stage_45_hex_renders_before_integration_starts(self):
        """T_AUDIT_S45 fires once every track is terminal and BEFORE integration
        starts, so gating the hex on `round.integration` hides it for the whole
        window the audit is actually running."""
        result = self._render_and_capture_hexes(self._s45_running_state())
        drawn = {h["gateKey"]: h["state"] for h in result["hexes"]}
        assert drawn.get("stage_45") == "running", result["hexes"]

    def test_stage_45_hex_still_renders_once_integration_starts(self):
        result = self._render_and_capture_hexes(self._s45_running_state("in_progress"))
        drawn = {h["gateKey"]: h["state"] for h in result["hexes"]}
        assert drawn.get("stage_45") == "running", result["hexes"]

    def test_pending_gate_keeps_its_plain_via(self):
        """A gate with no dispatch yet must keep the opaque via the board has
        always drawn, not a near-invisible hollow hex with no tooltip."""
        state = json.loads(json.dumps(SAMPLE_STATE))
        state["campaign"]["current_round"] = 2
        state["campaign"]["current_stage"] = "2_bottleneck_mining"
        state["campaign"]["rounds"] = [
            {
                "round_id": 1,
                "baseline": {"completed_at": "2026-01-01T00:00:00Z"},
                "bottleneck_mining": {"completed_at": "2026-01-01T01:00:00Z"},
                "parallel_tracks": {"tracks": {}},
                "integration": {"status": "pending"},
                "audit": {},
            },
            {
                "round_id": 2,
                "baseline": {},
                "bottleneck_mining": {},
                "parallel_tracks": {"tracks": {}},
                "integration": {"status": "pending"},
                "audit": {},
            },
        ]
        result = self._render_and_capture_hexes(state)
        pending = [h for h in result["hexes"] if h["state"] == "pending"]
        assert not pending, f"pending gates must not reach drawAuditHex: {pending}"
        assert result["vias"], "the plain vias must still be drawn"


# ---------------------------------------------------------------------------
# 10. baselineByRound derivation — prefer state.campaign.rounds[N].integration
#     over sidecars; reject nsys probes / non-full-benchmark sidecars.
# ---------------------------------------------------------------------------

class TestBaselineByRoundDerivation:
    """Regression guard for FIX B.1 (nsys-probe leak) updated for schema v4.0.

    Under v4.0 the authoritative per-round baseline is
    `state.campaign.rounds[N-1].baseline.e2e_latency` — a map keyed by BS,
    each value {avg, p50, p10, ...}. The backend normalizer populates this
    field for legacy v3 states too (from combined_e2e_result). The L3 BASELINE
    hero must NOT be fooled by the Stage-1 nsys Tier-0 short-run sidecar at
    `e2e_latency/e2e_latency_results.json.metrics.json`.
    """

    def _enrich(self, state_dict, catalog_dict):
        js_src = CIRCUIT_BOARD_JS.read_text()
        script = f"""
const window = {{}};
const document = {{ addEventListener: () => {{}}, createElement: () => {{
    return {{ style: {{}}, className: '', innerHTML: '', appendChild: () => {{}}, children: [] }};
}} }};
{js_src}
const state = {json.dumps(state_dict)};
const catalog = {json.dumps(catalog_dict)};
CircuitBoard.enrichFromCatalog(state, catalog);
console.log(JSON.stringify(state._catalog.baselineByRound));
"""
        return json.loads(run_js(script))

    def test_primary_source_state_integration_latency_baseline_s(self):
        """v4.0 primary source: baseline.e2e_latency map.
        primaryBsLatencyMs must equal avg*1000, source must be
        `baseline_e2e_latency`."""
        state = {
            "campaign": {
                "rounds": [
                    {
                        "round_id": 1,
                        "baseline": {
                            "e2e_latency": {
                                "8": {"avg": 1.66684, "p50": 1.66},
                            },
                            "per_bs_verdict": None,
                        },
                        "integration": {},
                    },
                ],
            },
        }
        # Include a contradicting nsys-probe sidecar to prove it is ignored.
        catalog = {
            "entries": {
                "e2e_latency/e2e_latency_results.json.metrics.json": {
                    "round": 1,
                    "stage": "baseline",
                    "metrics": {"baseline_avg_s": 0.14161},
                    "labels": {"full_benchmark": False, "nsys_probe": True},
                    "emitted_at": "2026-04-20T00:00:00Z",
                },
            },
        }
        bbr = self._enrich(state, catalog)
        assert "1" in bbr, f"expected round 1 in baselineByRound; got {list(bbr)}"
        b1 = bbr["1"]
        assert b1["source"] == "baseline_e2e_latency"
        assert abs(b1["primaryBsLatencyMs"] - 1666.84) < 0.01, (
            f"primaryBsLatencyMs must be 1666.84 (state value), got {b1['primaryBsLatencyMs']}"
        )
        assert b1["primaryBs"] == 8

    def test_v3_legacy_combined_e2e_result_uses_compat_fallback(self):
        """Pre-v4 data with only combined_e2e_result falls back to the
        compat path (normalizer would backfill in production; this test
        covers the frontend's defensive path)."""
        state = {
            "campaign": {
                "rounds": [
                    {
                        "round_id": 1,
                        "baseline": {},
                        "integration": {
                            "combined_e2e_result": {
                                "latency_baseline_s": 1.66684,
                                "per_bs_verdict": {"8": {}},
                            },
                        },
                    },
                ],
            },
        }
        bbr = self._enrich(state, {"entries": {}})
        assert bbr["1"]["source"] == "compat_fallback"
        assert abs(bbr["1"]["primaryBsLatencyMs"] - 1666.84) < 0.01
        assert bbr["1"]["primaryBs"] == 8

    def test_fallback_to_sidecar_when_no_integration_data(self):
        """When state.campaign.rounds[*] has no integration data but an e2e
        sidecar is present with `stage:"baseline"` and is a full benchmark,
        the sidecar is the fallback source."""
        state = {"campaign": {"rounds": []}}
        catalog = {
            "entries": {
                "e2e_latency/e2e_latency_results.json.metrics.json": {
                    "round": 1,
                    "stage": "baseline",
                    "metrics": {
                        "batch_sizes": {"8": {"baseline_avg_s": 1.66684}},
                    },
                    # No labels set OR labels.full_benchmark=true → must pick up.
                    "labels": {"full_benchmark": True},
                    "emitted_at": "2026-04-20T00:00:00Z",
                },
            },
        }
        bbr = self._enrich(state, catalog)
        assert "1" in bbr
        b1 = bbr["1"]
        assert b1["source"] == "e2e_latency_sweep"
        assert abs(b1["primaryBsLatencyMs"] - 1666.84) < 0.01

    def test_fallback_rejects_nsys_probe_sidecar(self):
        """Sidecar with labels.nsys_probe=true must be skipped — otherwise
        a Stage-1 Tier-0 short-run's 141.6 ms reading would leak into the hero."""
        state = {"campaign": {"rounds": []}}
        catalog = {
            "entries": {
                "e2e_latency/e2e_latency_results.json.metrics.json": {
                    "round": 1,
                    "stage": "baseline",
                    "metrics": {"baseline_avg_s": 0.14161},
                    "labels": {"nsys_probe": True},
                },
            },
        }
        bbr = self._enrich(state, catalog)
        assert "1" not in bbr, (
            f"nsys_probe sidecar must be skipped; got baselineByRound={bbr}"
        )

    def test_fallback_rejects_non_full_benchmark_sidecar(self):
        """Sidecar with labels.full_benchmark=false must also be skipped."""
        state = {"campaign": {"rounds": []}}
        catalog = {
            "entries": {
                "e2e_latency/e2e_latency_results.json.metrics.json": {
                    "round": 1,
                    "stage": "baseline",
                    "metrics": {"baseline_avg_s": 0.14161},
                    "labels": {"full_benchmark": False},
                },
            },
        }
        bbr = self._enrich(state, catalog)
        assert "1" not in bbr, (
            f"full_benchmark=false sidecar must be skipped; got {bbr}"
        )

    def test_state_integration_beats_contradicting_sidecar(self):
        """Even if a (well-formed) sidecar is present with a different value,
        the state.campaign baseline.e2e_latency map value wins."""
        state = {
            "campaign": {
                "rounds": [
                    {
                        "round_id": 1,
                        "baseline": {
                            "e2e_latency": {"8": {"avg": 1.66684, "p50": 1.66}},
                        },
                        "integration": {},
                    },
                ],
            },
        }
        catalog = {
            "entries": {
                "e2e_latency/e2e_latency_results.json.metrics.json": {
                    "round": 1,
                    "stage": "baseline",
                    "metrics": {"batch_sizes": {"8": {"baseline_avg_s": 0.9}}},
                    "labels": {"full_benchmark": True},
                    "emitted_at": "2026-04-20T00:00:00Z",
                },
            },
        }
        bbr = self._enrich(state, catalog)
        assert bbr["1"]["source"] == "baseline_e2e_latency"
        assert abs(bbr["1"]["primaryBsLatencyMs"] - 1666.84) < 0.01
