# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "ai_cli_session" / ".codex" / "skills" / "ammo" / "eval" / "scripts"
PARSER = SCRIPTS / "parse_artifacts.py"
SCORER = SCRIPTS / "score_campaign.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_campaign_keeps_explicit_empty_selected_candidates_empty():
    parser = _load_module("ammo_eval_parse_artifacts_semantic", PARSER)
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "parallel_tracks": {
                        "tracks": {
                            "op001": {"status": "PASS"},
                            "op002": {"status": "PASS"},
                        }
                    },
                    "integration": {
                        "status": "pending",
                        "selected_candidates": [],
                    },
                }
            ],
        }
    }

    campaign = parser._parse_campaign(state)

    assert campaign["rounds"][0]["candidates_selected"] == 0


def test_parse_campaign_preserves_legacy_missing_selected_candidates_fallback():
    parser = _load_module("ammo_eval_parse_artifacts_legacy", PARSER)
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "parallel_tracks": {
                        "tracks": {
                            "op001": {"status": "PASS"},
                            "op002": {"status": "FAIL"},
                        }
                    },
                    "integration": {"status": "pending"},
                }
            ],
        }
    }

    campaign = parser._parse_campaign(state)

    assert campaign["rounds"][0]["candidates_selected"] == 2


def test_parse_campaign_uses_passing_candidates_when_selection_is_legacy_explicit():
    parser = _load_module("ammo_eval_parse_artifacts_passing", PARSER)
    state = {
        "campaign": {
            "current_round": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "parallel_tracks": {
                        "tracks": {
                            "op001": {"status": "PASS"},
                            "op002": {"status": "PASS"},
                        }
                    },
                    "integration": {
                        "status": "pending",
                        "passing_candidates": ["op002"],
                    },
                }
            ],
        }
    }

    campaign = parser._parse_campaign(state)

    assert campaign["rounds"][0]["candidates_selected"] == 1


def test_score_gates_counts_claude_gated_pass_integration_as_passed():
    scorer = _load_module("ammo_eval_score_campaign_semantic", SCORER)
    snapshot = {
        "gates": {
            "phase1_baseline": {"status": "PASS"},
            "validation_gates": [],
            "integration": {"status": "gated_pass"},
        }
    }

    result = scorer.score_gates(snapshot)

    assert result["sub_scores"]["total_gates_checked"] == 2
    assert result["sub_scores"]["total_passed"] == 2
    assert result["sub_scores"]["pass_rate"] == 1.0
