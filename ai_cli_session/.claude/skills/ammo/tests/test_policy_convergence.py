# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Policy-convergence bridge.

Single-authority policy is tested semantically in
``test_guidance_semantic_contract.py``. This bridge prevents reintroducing the
old pattern of requiring the same sentence or numbering in many files.
"""

from pathlib import Path


def test_policy_uses_a_semantic_contract_instead_of_prose_snapshots() -> None:
    path = Path(__file__).with_name("test_guidance_semantic_contract.py")
    text = path.read_text(encoding="utf-8")
    assert "test_evidence_scope_and_projection_authorities_are_preserved" in text
    assert "test_schema_and_transition_enums_stay_authoritative" in text
    assert "test_retired_hook_names_and_bare_python3_are_absent" in text
