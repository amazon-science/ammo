# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Semantic-preservation checks for the concise AMMO guidance layout.

These tests intentionally assert topology, authority, and machine-facing
contracts rather than duplicated prose sentences.

Claude-variant layout: role documents live in `.claude/agents/*.md` (frontmatter
IS the runtime registration — there are no bootstrap TOMLs), the schema in
`.claude/schemas/`, and the lifecycle guards are shell hooks in `.claude/hooks/`.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = ROOT.parents[1]
AGENTS = CLAUDE_DIR / "agents"
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")

# Claude keeps the legacy role names for the implementation and report roles.
ROLE_TO_AGENT_FILE = {
    "ammo-auditor": "ammo-auditor.md",
    "ammo-champion": "ammo-champion.md",
    "ammo-delegate": "ammo-delegate.md",
    "ammo-impl-champion": "ammo-impl-champion.md",
    "ammo-investigator": "ammo-investigator.md",
    "ammo-researcher": "ammo-researcher.md",
    "ammo-resolver": "ammo-resolver.md",
    "ammo-transcript-monitor": "ammo-transcript-monitor.md",
    "ammo-report-writer": "ammo-report-writer.md",
}


def _agent_text(role: str) -> str:
    return (AGENTS / ROLE_TO_AGENT_FILE[role]).read_text(encoding="utf-8")


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _words(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def _state_engine():
    path = ROOT / "scripts" / "ammo_state.py"
    spec = importlib.util.spec_from_file_location("ammo_state_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_runtime_state(stage="1_baseline", status="active") -> dict:
    return {
        "campaign": {
            "schema_version": "4.1",
            "status": status,
            "current_round": 1,
            "current_stage": stage,
            "config": {},
            "rounds": [
                {
                    "status": "IN_PROGRESS",
                    "audit": {},
                    "parallel_tracks": {"tracks": {}},
                    "integration": {"status": "pending"},
                    "debate": {"selected_candidates": []},
                    "bottleneck_mining": {},
                }
            ],
        }
    }


def test_stage_graph_and_all_roles_are_preserved() -> None:
    stages = [
        "1_baseline",
        "2_bottleneck_mining",
        "3_debate",
        "4_5_parallel_tracks",
        "6_integration",
        "7_campaign_eval",
        "7b_report",
    ]
    positions = [SKILL.index(stage) for stage in stages]
    assert positions == sorted(positions)

    for role, filename in ROLE_TO_AGENT_FILE.items():
        assert role in SKILL
        agent_path = AGENTS / filename
        assert agent_path.is_file()
        # Frontmatter is the Claude runtime registration: name must match.
        head = agent_path.read_text(encoding="utf-8")
        assert head.startswith("---\n")
        frontmatter = head.split("---", 2)[1]
        assert f"name: {role}" in frontmatter
        assert "description:" in frontmatter


def test_agent_judgment_is_not_replaced_by_deterministic_code() -> None:
    assert "Preserve agent judgment" in SKILL
    assert "mechanical invariants" in SKILL
    assert "never substitutes for an agent's technical review" in SKILL
    assert "You are the lead orchestrator" in SKILL
    assert "you never implement" in SKILL


def test_stage1_clean_timing_and_profiling_remain_separate() -> None:
    researcher = _agent_text("ammo-researcher")
    assert "exactly two separate sweep invocations" in researcher
    assert "clean authoritative E2E timing" in researcher
    assert "bounded attribution profiling" in researcher
    assert "Never use profiler-run latency as official timing" in researcher
    assert "## Technology Landscape" in researcher
    assert "top three addressable kernel opportunities by `f_e2e × removable_fraction`" in researcher
    assert "`f_decode` as a diagnostic column" in researcher
    assert "## Workload Dilution (per BS)" in researcher
    assert "## Top Components (by f_e2e)" in researcher


def test_stage3_multi_champion_debate_and_scope_challenge_remain() -> None:
    champion = _agent_text("ammo-champion")
    debate = _text("orchestration/debate-protocol.md")
    assert "2-4 `ammo-champion`" in SKILL
    assert "2-3" in champion
    assert "Analyze every component" in champion
    assert "do not recapture or broaden it in Stage 3" in champion
    assert "Bounded, claim-driven single-kernel Nsys/NCU evidence remains allowed" in champion
    for marker in (
        "Technology Selection",
        "Category block",
        "Precision Classification",
        "## Open Items Declaration",
        "critique",
        "rebuttal",
    ):
        assert marker in champion
    assert "cannot self-close a boundary-scope objection" in champion
    assert "only by the lead or critic" in SKILL
    assert "## Convergence Criteria" in debate
    assert "## Winner Selection" in debate


def test_evidence_scope_and_projection_authorities_are_preserved() -> None:
    rules = _text("references/debate-rules.md")
    math = _text("references/e2e-delta-math.md")
    scoring = _text("references/debate-scoring-rubric.md")
    assert "## Evidence-Scope Ladder" in rules
    assert "`proxy`" in rules
    assert "`production_boundary`" in rules
    assert "clean_e2e" in rules
    assert "proxy-scope" in scoring or "proxy" in scoring
    assert "f_e2e = f_decode × decode_busy × decode_share_of_e2e" in math
    assert "T_component_old / T_component_new" in math
    assert "T_component_new / T_component" not in math
    assert "**Non-overlap rule.**" in math


def test_implementation_gates_slots_and_fallback_remain() -> None:
    implementer = _agent_text("ammo-impl-champion")
    validation = _text("references/validation-defaults.md")
    protocol = _text("orchestration/debate-protocol.md")
    for gate in ("Gate 5.1a", "Gate 5.1b", "Gate 5.2", "Gate 5.3a", "Gate 5.3b"):
        assert gate in implementer
    for slot in ("opt_correctness/{op_id}", "opt_profiling/{op_id}", "opt/{op_id}"):
        assert slot in implementer
        assert slot in validation
    assert "claim-appropriate profiler evidence" in implementer
    assert "Nsys is normally sufficient" in implementer
    assert "complete blocking validation contract" in implementer
    assert "complete blocking validation contract" in protocol
    assert "Proposal prose may recommend" in implementer
    assert "clean timing vs the Stage 1 baseline" in implementer
    assert "never re-run a baseline arm from the worktree" in implementer
    assert "pure inter-kernel host/dispatch" in implementer
    assert "Gate 5.1b remains mandatory" in implementer
    assert "PASS -> GATED_PASS -> GATING_REQUIRED -> RETRY_WITH_CONTINGENCY -> FAIL" in implementer
    assert "DILUTED_PASS" in implementer
    assert "ammo-investigator" in implementer


def test_monitor_is_continuous_read_only_and_machine_compatible() -> None:
    monitor = _agent_text("ammo-transcript-monitor")
    assert "never edit source, tests, gate results, `state.json`" in monitor
    assert "two-cadence" in monitor
    assert "full-lifetime coverage" in monitor
    assert "transcript_offsets.json" in monitor
    assert "_summary.json" in monitor
    assert "coverage_status" in monitor
    assert "poll count" in monitor
    assert "does not author" in monitor or "authors the verdict" in monitor
    assert "Accuracy-Failure Persistence" in monitor
    assert "DA-MONITOR" in monitor
    dispatch = _text("orchestration/parallel-tracks.md")
    assert "`op_id`" in dispatch and "`worktree_branch`" in dispatch
    assert "absolute worktree path" in dispatch
    assert "monitor_audits" in dispatch


def test_four_independent_audits_and_two_phase_order_remain() -> None:
    auditor = _agent_text("ammo-auditor")
    protocol = _text("orchestration/audit-protocol.md")
    for trigger in ("T_AUDIT_S1", "T_AUDIT_S2", "T_AUDIT_S45", "T_AUDIT_S67"):
        assert trigger in SKILL
        assert trigger in protocol
    assert auditor.index("## Phase 1 - Independent Reconstruction") < auditor.index(
        "## Phase 2 - Checklist Verification"
    )
    assert "Do **not** read `references/audit-invariants.md` at spawn" in auditor
    assert "{stage}_cycle_{N}.md" in auditor
    assert "stage_67.md" in auditor
    assert "ammo-delegate" in auditor
    assert "not blocking by itself" in auditor


def test_investigator_resolver_and_report_review_remain() -> None:
    investigator = _agent_text("ammo-investigator")
    resolver = _agent_text("ammo-resolver")
    report = _agent_text("ammo-report-writer")
    assert "read-only" in investigator
    assert "decision_support" in investigator
    assert "ROOT CAUSE IDENTIFIED" in investigator
    assert "fresh, independent DA reviewer" in resolver
    assert "Post-Merge Testing" in resolver
    # Claude runs the adversarial fact-checker as the report-writer's Stop hook.
    assert "adversarial fact-checker" in report
    assert '"ok": true' in report or "ok: true" in report
    report_authority = _text("report/SKILL.md")
    for path in (
        "rounds/{N}/constraints.md",
        "rounds/{N}/mining/bottleneck_analysis.md",
        "rounds/{N}/sweeps/opt_profiling/{op_id}/",
        "rounds/{N}/sweeps/opt_correctness/{op_id}/",
        "rounds/{N}/sweeps/opt/{op_id}/",
    ):
        assert path in report_authority
    assert "highest version" not in report_authority


def test_integration_and_campaign_loop_semantics_remain() -> None:
    integration = _text("orchestration/integration-logic.md")
    assert "## Single-Track Short-Circuit" in integration
    assert "## Combined Validation Workflow" in integration
    assert "ammo-resolver" in integration
    assert "direct" in SKILL and "never multiply round deltas" in SKILL
    assert "stop iff" in SKILL
    assert "top_addressable_e2e_pct" in SKILL
    assert "After SHIP" in SKILL and "After EXHAUSTED" in SKILL
    assert "mining_invalidated:true" in SKILL
    assert "exhausted_technologies" in SKILL
    assert "integration_profiling" in integration
    assert "use `campaign_complete` after a SHIP" in SKILL
    assert "`campaign_exhausted` after an EXHAUSTED" in SKILL


def test_schema_and_transition_enums_stay_authoritative() -> None:
    schema = json.loads(
        (CLAUDE_DIR / "schemas" / "state.schema.json").read_text(encoding="utf-8")
    )
    transitions = json.loads((ROOT / "scripts" / "transitions.json").read_text())
    stages = [
        "1_baseline",
        "2_bottleneck_mining",
        "3_debate",
        "4_5_parallel_tracks",
        "6_integration",
        "7_campaign_eval",
        "7b_report",
    ]
    assert schema["properties"]["campaign"]["properties"]["current_stage"][
        "enum"
    ] == stages
    assert transitions["stage_ladder"] == stages
    track = schema["properties"]["campaign"]["properties"]["rounds"]["items"][
        "properties"
    ]["parallel_tracks"]["properties"]["tracks"]["additionalProperties"]
    terminal = {"PASS", "GATED_PASS", "FAIL"}
    assert set(transitions["track_terminal_statuses"]) == terminal
    assert terminal <= set(track["properties"]["status"]["enum"])
    selected = schema["properties"]["campaign"]["properties"]["rounds"][
        "items"
    ]["properties"]["debate"]["properties"]["selected_candidates"]["items"]
    assert "proposal_file" in selected["required"]
    assert "SKIPPED" in track["properties"]["gate_5_1a"]["enum"]
    assert "gate_5_1a_skip_reason" in track["properties"]
    schema_text = json.dumps(schema, sort_keys=True)
    for status in ("IN_PROGRESS", "PASS", "GATING_REQUIRED", "GATED_PASS", "FAIL", "GPU_BLOCKED"):
        assert status in schema_text


def test_retired_hook_names_and_bare_python3_are_absent() -> None:
    active = [ROOT / "SKILL.md"]
    active += sorted(AGENTS.glob("ammo-*.md"))
    active += sorted((ROOT / "orchestration").glob("*.md"))
    active += sorted((ROOT / "references").glob("*.md"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in active)
    # The Claude hooks referenced by guidance must exist; dead references rot.
    for hook in re.findall(r"\b([\w-]+\.sh)\b(?![\w.])", text):
        candidates = [CLAUDE_DIR / "hooks" / hook, ROOT / "scripts" / hook]
        assert any(c.is_file() for c in candidates), f"referenced hook missing: {hook}"
    # Bare python3 at line start invites system-python vllm imports; guidance
    # must show the venv-activated form (hook YAML command lines excepted —
    # they run outside the venv contract).
    for path in [ROOT / "SKILL.md"] + sorted((ROOT / "orchestration").glob("*.md")) + sorted(
        (ROOT / "references").glob("*.md")
    ):
        body = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?m)^\s*python3\b", body):
            raise AssertionError(f"bare python3 in {path}: {match.group(0)!r}")
    for dead in ("kernel-benchmark-template.py", "da-audit-checklist.md",
                 "claude-codex-equivalents.md"):
        assert dead not in text


def test_progressive_disclosure_reduces_hot_path_without_deleting_roles() -> None:
    assert _words(ROOT / "SKILL.md") < 2500
    role_words = sum(
        _words(AGENTS / filename) for filename in ROLE_TO_AGENT_FILE.values()
    )
    assert role_words < 9500
    assert not (ROOT / "README.md").exists()


def test_changed_markdown_fences_are_balanced() -> None:
    for path in ROOT.rglob("*.md"):
        assert path.read_text(encoding="utf-8").count("```") % 2 == 0, path


def test_active_relative_markdown_links_resolve() -> None:
    docs = [ROOT / "SKILL.md", ROOT / "report" / "SKILL.md"]
    docs += sorted(AGENTS.glob("ammo-*.md"))
    docs += sorted((ROOT / "orchestration").glob("*.md"))
    docs += sorted((ROOT / "references").glob("*.md"))
    pattern = re.compile(r"`((?:orchestration|references)/[^`*]+\.md)`")
    agent_pattern = re.compile(r"`\.claude/agents/([^`*]+\.md)`")
    for doc in docs:
        body = doc.read_text(encoding="utf-8")
        for relative in pattern.findall(body):
            assert (ROOT / relative).is_file(), f"{doc}: {relative}"
        for agent_file in agent_pattern.findall(body):
            assert (AGENTS / agent_file).is_file(), f"{doc}: .claude/agents/{agent_file}"


def test_stop_guard_blocks_active_campaign_and_requires_report() -> None:
    guard = (CLAUDE_DIR / "hooks" / "ammo-stop-guard.sh").read_text(encoding="utf-8")
    # The Claude stop guard blocks session end while the campaign is active and
    # requires REPORT.md before a terminal campaign may stop.
    assert "campaign.status" in guard or '.campaign.status' in guard
    assert "REPORT.md" in guard
    assert "campaign_complete" in guard and "campaign_exhausted" in guard


def test_runtime_guidance_dependencies_are_registered() -> None:
    layout = _text("references/artifact-layout.md")
    runner = _text("scripts/run_vllm_bench_latency_sweep.py")
    for slot in (
        "opt/{op_id}",
        "opt_correctness/{op_id}",
        "opt_profiling/{op_id}",
    ):
        assert slot in layout
    assert "def _is_gate_slot" in runner
    assert 'slot.startswith("opt/")' in runner
    assert 'slot.startswith("opt_correctness/")' in runner
    assert 'slot.startswith("opt_profiling/")' in runner
    state_engine = _text("scripts/ammo_state.py")
    assert '"audit": {}' in state_engine
    # Bare-python sweep invocations are guarded by the Claude venv hook.
    venv_guard = (CLAUDE_DIR / "hooks" / "ammo-venv-python-guard.sh").read_text(
        encoding="utf-8"
    )
    assert "run_vllm_bench_latency_sweep" in venv_guard
    assert ".venv/bin/python" in venv_guard


def test_runtime_fail_closed_policy_invariants_bind() -> None:
    engine = _state_engine()
    transitions = engine.load_transitions()

    previous = _minimal_runtime_state()
    jumped = copy.deepcopy(previous)
    jumped["campaign"]["current_stage"] = "7b_report"
    assert "not adjacent" in engine.gate_violation(jumped, transitions, previous)

    selection = copy.deepcopy(previous)
    selection["campaign"]["rounds"][0]["debate"]["selected_candidates"] = [
        {"op_id": "OP-X", "score_breakdown": {}}
    ]
    assert "no category token" in engine.gate_violation(
        selection, transitions, previous
    )
    selection["campaign"]["rounds"][0]["debate"]["selected_candidates"][0][
        "category"
    ] = "kernel_replacement"
    assert "no score_breakdown.evidence_scope" in engine.gate_violation(
        selection, transitions, previous
    )

    terminal = _minimal_runtime_state("7_campaign_eval", "campaign_exhausted")
    terminal_round = terminal["campaign"]["rounds"][0]
    terminal_round["audit"] = {"stage_67": {"passed_at": "now"}}
    terminal["campaign"]["config"]["min_e2e_improvement_pct"] = 0.5
    terminal_round["bottleneck_mining"]["top_f_e2e_pct"] = 10.0
    terminal_round["bottleneck_mining"]["top_addressable_e2e_pct"] = 0.49
    assert "requires integration status exhausted or failed" in engine.gate_violation(
        terminal, transitions
    )
    terminal_round["integration"]["status"] = "exhausted"
    assert engine.gate_violation(terminal, transitions) is None

    for value in (None, 0.5, 0.6):
        blocked = copy.deepcopy(terminal)
        blocked["campaign"]["rounds"][0]["bottleneck_mining"][
            "top_addressable_e2e_pct"
        ] = value
        assert "Terminal-status violation" in engine.gate_violation(
            blocked, transitions
        )

    shipped = copy.deepcopy(terminal)
    shipped["campaign"]["status"] = "campaign_complete"
    shipped["campaign"]["rounds"][0]["integration"]["status"] = "combined"
    assert engine.gate_violation(shipped, transitions) is None

    parsed = engine.parse_mining_md(
        "| Component | f_e2e | f_e2e x (1-1/ceiling) |\n"
        "|---|---|---|\n| moe | 0.30 | 0.0049 |"
    )
    assert parsed["top_f_e2e_pct"] == 30.0
    assert parsed["top_addressable_e2e_pct"] == 0.49
    assert parsed["top_f_decode_pct"] is None

    reused = copy.deepcopy(terminal)
    reused["campaign"]["rounds"].append(
        {
            "bottleneck_mining": {},
            "audit": {"stage_67": {"passed_at": "now"}},
            "integration": {"status": "exhausted"},
        }
    )
    reused["campaign"]["current_round"] = 2
    assert engine.gate_violation(reused, transitions) is None


def test_selected_cohort_and_monitor_pairing_fail_closed() -> None:
    engine = _state_engine()
    transitions = engine.load_transitions()
    empty = _minimal_runtime_state("6_integration")
    empty["campaign"]["schema_version"] = "4.2"
    empty["campaign"]["rounds"][0]["audit"] = {
        "stage_45": {"passed_at": "now", "started_at": "then", "cycle": 1}
    }
    assert "2-3 unique winner ids" in engine.gate_violation(empty, transitions)

    state = _minimal_runtime_state("6_integration")
    state["campaign"]["schema_version"] = "4.2"
    rnd = state["campaign"]["rounds"][0]
    rnd["debate"] = {
        "completed_at": "now",
        "selected_winners": ["OP-X", "OP-Y"],
        "selected_candidates": [
            {"op_id": "OP-X", "selection_mode": "contingent_host_spike"},
            {"op_id": "OP-Y", "selection_mode": "ordinary"},
        ],
    }
    rnd["audit"] = {
        "stage_45": {"passed_at": "now", "started_at": "then", "cycle": 1}
    }

    assert "Cohort violation" in engine.gate_violation(state, transitions)

    rnd["parallel_tracks"]["tracks"] = {
        "OP-X": {"status": "PASS"},
        "OP-Y": {"status": "FAIL"},
    }
    reason = engine.gate_violation(state, transitions)
    assert "pairing evidence" in reason

    for op_id in ("OP-X", "OP-Y"):
        rnd["parallel_tracks"]["tracks"][op_id].update(
            {
                "implementer_agent": f"/root/impl_{op_id}",
                "implementer_rollout_id": f"rollout_{op_id}",
                "monitor_agent": f"/root/monitor_{op_id}",
                "monitor_evidence_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/obs.md"
                ),
                "monitor_offsets_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/offsets.json"
                ),
                "monitor_summary_path": (
                    f"rounds/1/tracks/{op_id}/monitor_audits/summary.json"
                ),
            }
        )
    rnd["parallel_tracks"]["tracks"]["OP-X"].update(
        {
            "gate_5_2": "PASS",
            "gate_5_2_boundary": {
                "baseline_duration_us": 100.0,
                "optimized_duration_us": 90.0,
                "occurrence_count": 1000,
                "baseline_e2e_us": 2_000_000.0,
                "e2e_equivalent_improvement_pct": 0.5,
                "campaign_floor_pct": 0.5,
                "meets_floor": True,
            },
        }
    )
    state["campaign"]["config"]["min_e2e_improvement_pct"] = 0.5
    assert engine.gate_violation(state, transitions) is None

    duplicate = copy.deepcopy(state)
    duplicate["campaign"]["current_stage"] = "6_integration"
    duplicate["campaign"]["rounds"][0]["debate"]["selected_winners"] = [
        "OP-X", "OP-X"
    ]
    assert "unique winner ids" in engine.gate_violation(duplicate, transitions)

    legacy_prev = _minimal_runtime_state("3_debate")
    legacy_prev_rnd = legacy_prev["campaign"]["rounds"][0]
    legacy_prev_rnd["debate"]["selected_winners"] = ["OP-LEGACY"]
    legacy_prev_rnd["debate"]["selected_candidates"] = [
        {"op_id": "OP-LEGACY"}
    ]
    legacy = copy.deepcopy(legacy_prev)
    legacy["campaign"]["current_stage"] = "4_5_parallel_tracks"
    legacy_rnd = legacy["campaign"]["rounds"][0]
    legacy_rnd["parallel_tracks"]["tracks"] = {
        "OP-LEGACY": {"status": "IN_PROGRESS"}
    }
    assert engine.gate_violation(legacy, transitions, legacy_prev) is None


def test_material_ship_requires_fresh_remine_before_terminal() -> None:
    engine = _state_engine()
    transitions = engine.load_transitions()
    shipped = _minimal_runtime_state("7_campaign_eval", "campaign_complete")
    campaign = shipped["campaign"]
    campaign["config"]["min_e2e_improvement_pct"] = 0.5
    rnd = campaign["rounds"][0]
    rnd["status"] = "SHIPPED"
    rnd["shipped"] = ["OP-X"]
    rnd["parallel_tracks"]["tracks"] = {"OP-X": {"status": "PASS"}}
    rnd["integration"]["status"] = "combined"
    rnd["audit"] = {"stage_67": {"passed_at": "now"}}
    rnd["bottleneck_mining"]["top_addressable_e2e_pct"] = 0.1
    assert "fresh post-promotion" in engine.gate_violation(shipped, transitions)

    post_ship = copy.deepcopy(shipped)
    post_ship["campaign"]["status"] = "active"
    post_ship, new_stage = engine.advance(post_ship, "SHIP", transitions)
    assert new_stage == "2_bottleneck_mining"
    current = post_ship["campaign"]["rounds"][1]
    current["bottleneck_mining"] = {
        "completed_at": "now",
        "top_component": "none",
        "top_bottleneck_share_pct": 0.1,
        "top_f_e2e_pct": 0.1,
        "top_addressable_e2e_pct": 0.1,
        "amdahl_ceiling": 1.001,
        "decode_frac": 0.5,
        "component_breakdown": [],
    }
    current["audit"] = {"stage_2": {"passed_at": "now"}}
    before_eval = copy.deepcopy(post_ship)
    post_ship["campaign"]["current_stage"] = "7_campaign_eval"
    assert engine.gate_violation(post_ship, transitions, before_eval) is None
    post_ship["campaign"]["status"] = "campaign_complete"
    assert engine.gate_violation(post_ship, transitions) is None


def test_all_diluted_ship_and_legacy_backfill_contracts_remain() -> None:
    engine = _state_engine()
    transitions = engine.load_transitions()

    state = _minimal_runtime_state("7_campaign_eval")
    current = state["campaign"]["rounds"][0]
    current["shipped"] = ["OP-X"]
    current["parallel_tracks"]["tracks"] = {
        "OP-X": {"status": "PASS", "diluted": True}
    }
    _, next_stage = engine.advance(state, "SHIP", transitions)
    assert next_stage == "3_debate"

    legacy = _minimal_runtime_state()
    candidate = {
        "op_id": "OP-X",
        "cited_evidence": ["rounds/1/debate/proposals/x_proposal.md"],
        "score_breakdown": {},
    }
    legacy["campaign"]["rounds"][0]["debate"]["selected_candidates"] = [candidate]
    before = copy.deepcopy(legacy)
    engine.backfill(legacy, CLAUDE_DIR / "unused", transitions)
    assert candidate["proposal_file"] == "rounds/1/debate/proposals/x_proposal.md"
    assert engine.gate_violation(legacy, transitions, before) is None


def test_removed_authority_headings_are_not_referenced() -> None:
    docs = [ROOT / "SKILL.md"]
    docs += sorted(AGENTS.glob("ammo-*.md"))
    docs += sorted((ROOT / "orchestration").glob("*.md"))
    docs += sorted((ROOT / "references").glob("*.md"))
    # Agent-read prose also lives in the engine prompts and the role test packs.
    docs += sorted((ROOT / "scripts").glob("*.py"))
    docs += sorted((ROOT / "tests" / "agents").glob("*.md"))
    for stale in (
        "SKILL.md § Non-Negotiables",
        "SKILL.md § Resume Protocol",
        "SKILL.md § Campaign Loop",
        "Non-Negotiable #",
    ):
        offenders = [
            str(path.relative_to(ROOT))
            for path in docs
            if stale in path.read_text(encoding="utf-8")
        ]
        assert not offenders, "stale authority pointer %r in %s" % (stale, offenders)


def test_progressive_disclosure_pointers_resolve_to_current_headings() -> None:
    pointers = (
        ("orchestration/debate-protocol.md", "## Phase 0: Independent Proposals"),
        ("orchestration/debate-protocol.md", "### Eligibility, revision, and elimination"),
        ("orchestration/debate-protocol.md", "## Debate Sub-Rounds"),
        ("orchestration/parallel-tracks.md", "## Reconciliation and Cohort Barrier"),
        ("orchestration/audit-protocol.md", "## Optional Early S45 Evidence"),
        ("references/champion-common-patterns.md", "## Worktree Environment"),
    )
    for relative, heading in pointers:
        assert heading in _text(relative), f"missing pointer target: {relative} § {heading}"


def test_debate_critique_assignments_remain_independent_under_churn() -> None:
    debate = _text("orchestration/debate-protocol.md")
    assert "never critiqued by" in debate and "champion who proposed it" in debate
    assert "fewer than two distinct champion owners" in debate
    assert "repair every affected cross-owner assignment" in debate
