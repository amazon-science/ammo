# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Coverage tripwire for the AMMO guidance semantic-contract suite.

Guidance compatibility is owned by ``test_guidance_semantic_contract.py``.
Nothing else asserts that suite exists, so an agent who deletes or guts it
still gets a green ``run_all.sh``. These tests are the tripwire.

They anchor on the suite's BEHAVIOR-BEARING content — the stage slugs, role
names, gate ids, artifact slots, and audit-phase order it binds — read from its
string literals via ``ast``. Renaming a test function is therefore free;
removing a contract is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONTRACT = Path(__file__).with_name("test_guidance_semantic_contract.py")

# Floors, not exact counts. The suite currently has 24 test functions and 160
# asserts. These allow normal editing and catch wholesale gutting.
MIN_TEST_FUNCTIONS = 18
MIN_ASSERTS = 120


def _tree() -> ast.Module:
    assert CONTRACT.is_file(), f"guidance semantic-contract suite missing: {CONTRACT}"
    return ast.parse(CONTRACT.read_text(encoding="utf-8"), filename=str(CONTRACT))


def _string_literals(tree: ast.Module) -> set[str]:
    """Every string constant in the suite, excluding docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def _assert_anchors(anchors, literals, label) -> None:
    missing = [a for a in anchors if not any(a in text for text in literals)]
    assert not missing, (
        f"{label} no longer bound by test_guidance_semantic_contract.py: {missing}"
    )


def test_semantic_contract_suite_still_asserts_at_scale() -> None:
    tree = _tree()
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert len(functions) >= MIN_TEST_FUNCTIONS, (
        f"guidance contract suite shrank to {len(functions)} test functions "
        f"(floor {MIN_TEST_FUNCTIONS}); it is the only guard on AMMO guidance "
        "semantics"
    )
    assert len(asserts) >= MIN_ASSERTS, (
        f"guidance contract suite shrank to {len(asserts)} asserts "
        f"(floor {MIN_ASSERTS})"
    )


def test_semantic_contract_suite_still_binds_stage_graph_and_roles() -> None:
    literals = _string_literals(_tree())
    _assert_anchors(
        [
            "1_baseline",
            "2_bottleneck_mining",
            "3_debate",
            "4_5_parallel_tracks",
            "6_integration",
            "7_campaign_eval",
            "7b_report",
        ],
        literals,
        "stage graph",
    )
    _assert_anchors(
        [
            "ammo-auditor",
            "ammo-champion",
            "ammo-delegate",
            "ammo-implementer",
            "ammo-investigator",
            "ammo-researcher",
            "ammo-resolver",
            "ammo-transcript-monitor",
            "ammo-report",
        ],
        literals,
        "role set",
    )


def test_semantic_contract_suite_still_binds_gates_slots_and_fallback() -> None:
    literals = _string_literals(_tree())
    _assert_anchors(
        ["Gate 5.1a", "Gate 5.1b", "Gate 5.2", "Gate 5.3a", "Gate 5.3b"],
        literals,
        "implementation gates",
    )
    _assert_anchors(
        ["opt_correctness/{op_id}", "opt_profiling/{op_id}", "opt/{op_id}"],
        literals,
        "artifact slots",
    )
    _assert_anchors(
        ["GATED_PASS", "RETRY_WITH_CONTINGENCY", "DILUTED_PASS"],
        literals,
        "outcome fallback chain",
    )


def test_semantic_contract_suite_still_binds_audits_and_phase_order() -> None:
    tree = _tree()
    literals = _string_literals(tree)
    _assert_anchors(
        ["T_AUDIT_S1", "T_AUDIT_S2", "T_AUDIT_S45", "T_AUDIT_S67"],
        literals,
        "audit triggers",
    )
    _assert_anchors(
        ["## Phase 1 - Independent Reconstruction", "## Phase 2 - Checklist Verification"],
        literals,
        "auditor phase headings",
    )
    # The order itself is a contract: Phase 1 must precede Phase 2. The suite
    # states it as an index comparison, so require an ORDERING assert (Lt/Gt),
    # not a mere membership check.
    has_order_assert = any(
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
            for op in node.test.ops
        )
        and "Phase 1 - Independent Reconstruction" in ast.dump(node)
        for node in ast.walk(tree)
    )
    assert has_order_assert, (
        "test_guidance_semantic_contract.py no longer asserts Phase 1 precedes "
        "Phase 2 in the auditor contract"
    )
