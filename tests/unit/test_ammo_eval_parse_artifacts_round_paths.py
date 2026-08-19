# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
CODEX_PARSER = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "eval" / "scripts" / "parse_artifacts.py"
CLAUDE_PARSER = ROOT / "ai_cli_session" / ".claude" / "skills" / "ammo" / "eval" / "scripts" / "parse_artifacts.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_parser():
    return _load_module(CODEX_PARSER, "ammo_eval_parse_artifacts_codex")


@pytest.fixture(params=[
    ("codex", CODEX_PARSER),
    ("claude", CLAUDE_PARSER),
], ids=["codex", "claude"])
def parser_both(request):
    name, path = request.param
    return _load_module(path, f"ammo_eval_parse_artifacts_{name}")


def _write(path: Path, text: str = "micro-experiment 12 us bottleneck_analysis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_round_one_eval_parser_prefers_campaign_round_namespace(tmp_path):
    parser = _load_parser()
    artifact_dir = tmp_path / "artifacts"
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [{"debate": {"candidates": ["op001"], "selected_winners": ["op001"]}}],
        }
    }

    _write(artifact_dir / "debate" / "summary.md", "| op | Final Score |\n| op-old | 1.0/10 |")
    _write(artifact_dir / "debate" / "proposals" / "legacy.md")
    _write(artifact_dir / "debate" / "round_1" / "legacy_argument.md")

    _write(artifact_dir / "debate" / "campaign_round_1" / "summary.md", "| op | Final Score |\n| op-new | 9.0/10 |")
    _write(artifact_dir / "debate" / "campaign_round_1" / "proposals" / "canonical.md")
    _write(artifact_dir / "debate" / "campaign_round_1" / "round_1" / "canonical_argument.md")

    debate = parser._parse_debate(artifact_dir, state)

    assert debate["total_proposals"] == 1
    assert debate["all_candidate_scores"] == [9.0]
    assert debate["per_campaign_round"][0]["proposals_count"] == 1
    assert debate["per_campaign_round"][0]["debate_rounds"] == 1


def test_round_one_eval_parser_keeps_legacy_fallback(tmp_path):
    parser = _load_parser()
    artifact_dir = tmp_path / "artifacts"
    state = {"campaign": {"current_round": 1, "rounds": [{"debate": {}}]}}

    _write(artifact_dir / "debate" / "summary.md", "| op | Final Score |\n| op-old | 7.0/10 |")
    _write(artifact_dir / "debate" / "proposals" / "legacy.md")
    _write(artifact_dir / "debate" / "round_1" / "legacy_argument.md")

    debate = parser._parse_debate(artifact_dir, state)

    assert debate["total_proposals"] == 1
    assert debate["all_candidate_scores"] == [7.0]
    assert debate["per_campaign_round"][0]["proposals_count"] == 1
    assert debate["per_campaign_round"][0]["debate_rounds"] == 1


# --------------------------------------------------------------------------- #
# Artifact Layout V2 dual-glob support
# --------------------------------------------------------------------------- #
#
# Under V2 the artifact layout is self-describing:
#   rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json
#   rounds/{N}/constraints.md
#   rounds/{N}/debate/proposals/*.md
#   rounds/{N}/debate/round_*/*.md
#   rounds/{N}/debate/summary.md
#
# Both the .claude and .codex copies of parse_artifacts.py must search both
# the legacy flat paths AND the v2 round-namespaced paths so a campaign that
# emitted artifacts under either layout produces a valid snapshot.


def _v2_e2e_payload() -> dict:
    return {
        "bench": {"baseline_label": "baseline", "opt_label": "opt"},
        "results": [
            {
                "batch_size": 1,
                "input_len": 512,
                "output_len": 128,
                "baseline": {"avg_s": 1.00},
                "opt": {"avg_s": 0.80},
                "speedup": 1.25,
                "improvement_pct": 25.0,
            }
        ],
    }


def test_v2_layout_e2e_results_parsed_from_rounds_namespace(parser_both, tmp_path):
    """rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json must be discovered."""
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    e2e_path = artifact_dir / "rounds" / "1" / "sweeps" / "opt" / "op-001" / "e2e_latency_results.json"
    e2e_path.parent.mkdir(parents=True, exist_ok=True)
    e2e_path.write_text(json.dumps(_v2_e2e_payload()), encoding="utf-8")

    parsed = parser._parse_e2e_results(artifact_dir)
    assert parsed is not None, "v2-layout e2e results must be discovered under rounds/{N}/sweeps/opt/{op_id}/"
    assert parsed["max_speedup"] == pytest.approx(1.25)
    assert parsed["batch_sizes"] == [1]


def test_legacy_layout_e2e_results_still_works(parser_both, tmp_path):
    """Existing flat tracks/op*/e2e_latency layout must keep working."""
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    e2e_path = artifact_dir / "tracks" / "op-001" / "e2e_latency" / "e2e_latency_results.json"
    e2e_path.parent.mkdir(parents=True, exist_ok=True)
    e2e_path.write_text(json.dumps(_v2_e2e_payload()), encoding="utf-8")

    parsed = parser._parse_e2e_results(artifact_dir)
    assert parsed is not None
    assert parsed["max_speedup"] == pytest.approx(1.25)


def test_v2_layout_constraints_md_recognized_for_phase1_gate(parser_both, tmp_path):
    """rounds/{N}/constraints.md must satisfy the Phase 1 baseline gate check."""
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    _write(artifact_dir / "rounds" / "1" / "constraints.md", "# Constraints\n")

    state = {"campaign": {"current_round": 1, "rounds": [{}]}}
    gates = parser._parse_gates(artifact_dir, state)
    assert gates["phase1_baseline"]["status"] == "PASS", (
        "v2 rounds/{N}/constraints.md must count as Phase 1 baseline pass"
    )


def test_legacy_constraints_md_still_recognized(parser_both, tmp_path):
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    _write(artifact_dir / "constraints.md", "# Constraints\n")

    state = {"campaign": {"current_round": 1, "rounds": [{}]}}
    gates = parser._parse_gates(artifact_dir, state)
    assert gates["phase1_baseline"]["status"] == "PASS"


def test_v2_layout_debate_parsed_from_rounds_namespace(parser_both, tmp_path):
    """rounds/{N}/debate/{summary,proposals,round_*} must be discovered alongside legacy paths."""
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [{"debate": {"candidates": ["op-v2"], "selected_winners": ["op-v2"]}}],
        }
    }

    _write(
        artifact_dir / "rounds" / "1" / "debate" / "summary.md",
        "| op | Final Score |\n| op-v2 | 8.5/10 |",
    )
    _write(artifact_dir / "rounds" / "1" / "debate" / "proposals" / "v2.md")
    _write(artifact_dir / "rounds" / "1" / "debate" / "round_1" / "argument.md")

    debate = parser._parse_debate(artifact_dir, state)
    assert debate["total_proposals"] == 1, (
        "v2 rounds/1/debate/proposals/*.md must be counted"
    )
    assert debate["all_candidate_scores"] == [8.5], (
        "v2 rounds/1/debate/summary.md must be parsed for scores"
    )
    assert debate["per_campaign_round"][0]["debate_rounds"] == 1, (
        "v2 rounds/1/debate/round_*/ must count toward debate_rounds"
    )


def test_v2_and_legacy_debate_paths_coexist(parser_both, tmp_path):
    """Mixed campaigns (legacy flat + v2 rounds/) must union proposals from both."""
    parser = parser_both
    artifact_dir = tmp_path / "artifacts"
    state = {"campaign": {"current_round": 1, "rounds": [{"debate": {}}]}}

    _write(artifact_dir / "debate" / "summary.md", "| op | Final Score |\n| op-old | 7.0/10 |")
    _write(artifact_dir / "debate" / "proposals" / "legacy.md")

    _write(artifact_dir / "rounds" / "1" / "debate" / "proposals" / "v2.md")

    debate = parser._parse_debate(artifact_dir, state)
    assert debate["total_proposals"] == 2, "legacy + v2 proposals must be unioned"
