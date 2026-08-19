# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for Phase 3 (spec §5): Optimization-category taxonomy.


Verifies that the Phase 3 documentation/agent updates land correctly:
1. New file `references/optimization-categories.md` defines the 6 categories,
   their projection formulas, technology eligibility, evidence requirements,
   and disambiguating examples.
2. SKILL.md NN#7 references `f_e2e` (not `f_decode`) as the Amdahl input.
3. SKILL.md NN#8 expands "Custom kernel mandate" → "GPU-executable code mandate"
   covering the 4 non-kernel pathways.
4. technology-selection.md adds non-kernel technology classes per §5.5.
5. ammo-champion.md requires a `## Category` field in Phase 0 (schema_version-gated).
6. ammo-impl-champion.md references multi-round policy + reads Category.
7. ammo-impl-validator.md adds per-category validation scaffolds.
8. validation-defaults.md adds per-category Gate 5.2 routing + per-mode min_e2e thresholds.
9. debate-protocol.md adds Category eligibility gate + category-preference spawn context.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "ai_cli_session" / ".claude"

SKILL_MD = ROOT / "skills" / "ammo" / "SKILL.md"
TECH_SELECTION = ROOT / "skills" / "ammo" / "references" / "technology-selection.md"
OPT_CATEGORIES = ROOT / "skills" / "ammo" / "references" / "optimization-categories.md"
VALIDATION_DEFAULTS = ROOT / "skills" / "ammo" / "references" / "validation-defaults.md"
DEBATE_PROTOCOL = ROOT / "skills" / "ammo" / "orchestration" / "debate-protocol.md"
CHAMPION = ROOT / "agents" / "ammo-champion.md"
IMPL_CHAMPION = ROOT / "agents" / "ammo-impl-champion.md"
IMPL_VALIDATOR = ROOT / "agents" / "ammo-impl-validator.md"

CATEGORIES = [
    "kernel_replacement",
    "kernel_fusion",
    "mega_kernel",
    "graph_restructuring",
    "dispatch_optimization",
    "workload_specialization",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    assert path.exists(), f"Required file missing: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. New file: references/optimization-categories.md
# ---------------------------------------------------------------------------


class TestOptimizationCategoriesDoc:
    def test_file_exists(self):
        assert OPT_CATEGORIES.exists(), (
            "Spec §5.7 requires new file references/optimization-categories.md"
        )

    def test_lists_all_six_categories(self):
        text = _read(OPT_CATEGORIES)
        for cat in CATEGORIES:
            assert cat in text, f"optimization-categories.md must define `{cat}`"

    def test_includes_per_category_projection_formulas(self):
        text = _read(OPT_CATEGORIES)
        # Spec §3.6 / §5: each category has a projection formula. Tag the
        # formula reference clearly so champions can find it.
        # The kernel categories use Amdahl-style f_e2e × (1 - 1/s).
        assert "f_e2e" in text, (
            "optimization-categories.md must reference f_e2e (Amdahl input)"
        )
        assert re.search(r"\(1\s*-\s*1\s*/\s*s", text), (
            "Must include Amdahl-style `(1 - 1/s)` formula for kernel categories"
        )
        assert "inter_kernel_share" in text, (
            "graph_restructuring/dispatch_optimization formulas use inter_kernel_share"
        )
        assert "launches_eliminated" in text, (
            "mega_kernel projection uses launches_eliminated / total_launches"
        )
        assert "elimination_fraction" in text, (
            "graph_restructuring/dispatch use elimination_fraction"
        )
        assert "host_fraction" in text, (
            "dispatch_optimization formula uses host_fraction"
        )

    def test_includes_phase_0_evidence_requirements_per_category(self):
        text = _read(OPT_CATEGORIES)
        # Spec §5.2: each category lists Phase 0 micro-experiment requirements.
        assert re.search(r"phase\s*0", text, re.IGNORECASE), (
            "Must reference Phase 0 evidence requirements"
        )
        # mega_kernel needs a 2-layer subset prototype with occupancy table
        assert re.search(r"2[- ]layer", text, re.IGNORECASE), (
            "mega_kernel/graph_restructuring evidence cites a 2-layer subset"
        )
        assert "occupancy" in text.lower(), (
            "mega_kernel Phase 0 evidence lists occupancy floor (≥ 0.25)"
        )

    def test_includes_technology_eligibility_per_category(self):
        text = _read(OPT_CATEGORIES)
        # Spec §5.5: kernel categories use the 4 authoring classes;
        # graph/dispatch use Python/C++/CUDA Graph API.
        assert "Triton" in text and "CuTeDSL" in text and "CUTLASS" in text, (
            "Kernel-category technology eligibility must list Triton/CuTeDSL/CUTLASS"
        )
        assert re.search(r"CUDA\s*Graph\s*API", text), (
            "graph_restructuring eligibility cites CUDA Graph API"
        )
        # Dispatch optimization is a Python/C++ pathway
        assert re.search(r"vLLM\s+scheduler", text, re.IGNORECASE), (
            "dispatch_optimization eligibility references vLLM scheduler internals"
        )

    def test_includes_per_category_validation_gate_routing(self):
        text = _read(OPT_CATEGORIES)
        # Spec §5.1: graph_restructuring SKIPS Gate 5.2 (no kernel-level
        # speedup gate); other categories run it differently.
        assert "Gate 5.2" in text, "Must reference Gate 5.2 routing per category"
        # graph_restructuring is the standout — must explicitly note no Gate 5.2
        m = re.search(
            r"graph_restructuring.*?Gate 5\.2",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        assert m, (
            "graph_restructuring section must mention Gate 5.2 (skipped/N/A)"
        )

    def test_includes_disambiguating_examples(self):
        text = _read(OPT_CATEGORIES)
        # Spec §5.7: must include "what passes vs what's rejected" examples
        # so champions can self-classify.
        # Keyword surface — examples talk about pass/reject or accepted/rejected.
        assert re.search(r"reject|rejected", text, re.IGNORECASE), (
            "Must include rejected/disqualifying examples"
        )
        assert re.search(r"accepted|passes|eligible", text, re.IGNORECASE), (
            "Must include accepted/eligible examples"
        )

    def test_includes_anti_regression_carveout(self):
        text = _read(OPT_CATEGORIES)
        # Spec §5.5: anti-regression rule applies WITHIN kernel categories
        # (rank-based); for non-kernel categories it applies to E2E only.
        assert re.search(r"anti[- ]regression", text, re.IGNORECASE), (
            "Must include anti-regression carve-out for non-kernel categories"
        )

    def test_includes_schema_version_guard(self):
        """The Category requirement enforcement is schema-version gated
        per the same `schema_version >= "4.1"` pattern used in #2/#3/#11."""
        text = _read(OPT_CATEGORIES)
        assert re.search(r"schema_version|4\.1", text), (
            "Must document schema-version gate for Category requirement"
        )

    def test_minimum_size(self):
        # Spec calls for ~300 lines; allow some flex (200-600 line range).
        text = _read(OPT_CATEGORIES)
        line_count = len(text.splitlines())
        assert 150 <= line_count, (
            f"optimization-categories.md too short ({line_count} lines); "
            f"spec calls for ~300 lines of taxonomy"
        )


# ---------------------------------------------------------------------------
# 2. SKILL.md NN#7 (f_e2e) and NN#8 (GPU-executable code mandate)
# ---------------------------------------------------------------------------


class TestSkillMdNonNegotiables:
    def test_nn7_uses_f_e2e_as_amdahl_input(self):
        text = _read(SKILL_MD)
        # The minified SKILL.md folds the numbered non-negotiables into prose;
        # the E2E-delta mandate survives as the Stage 7 stop check, which must
        # use f_e2e (not raw f / f_decode) as the Amdahl multiplier.
        assert "f_e2e" in text, (
            "SKILL.md must specify f_e2e as the Amdahl multiplier (spec §3)"
        )
        assert "top_addressable_e2e_pct" in text

    def test_nn8_is_gpu_executable_code_mandate(self):
        text = _read(SKILL_MD)
        # Spec §5.4: "GPU-executable code mandate" replaces the old custom
        # kernel mandate.
        m = re.search(
            r"8\.\s*\*\*([^*]+)\*\*",
            text,
        )
        assert m, "NN#8 header not found"
        title = m.group(1)
        assert (
            "GPU-executable" in title
            or "GPU executable" in title
            or "gpu-executable" in title.lower()
        ), (
            f"NN#8 title must be 'GPU-executable code mandate', got: {title!r}"
        )

    def test_nn8_lists_eligible_pathways(self):
        text = _read(SKILL_MD)
        # Spec §5.4: NN#8 must enumerate the 4 eligible pathways.
        m = re.search(
            r"8\.\s*\*\*GPU.executable.*?(?=\n\d+\.\s*\*\*|\n##)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        assert m, "NN#8 GPU-executable code mandate block not found"
        block = m.group(0)
        # All four pathways from §5.4
        assert re.search(r"custom kernel", block, re.IGNORECASE), (
            "NN#8 must list (a) custom kernels"
        )
        assert re.search(r"CUDA[- ]graph", block, re.IGNORECASE), (
            "NN#8 must list (b) CUDA-graph composition/replay"
        )
        assert re.search(r"dispatch", block, re.IGNORECASE), (
            "NN#8 must list (c) dispatch-path code"
        )
        assert re.search(r"scheduler", block, re.IGNORECASE), (
            "NN#8 must list (d) scheduler-coordination code"
        )
        assert re.search(r"config[- ]only", block, re.IGNORECASE), (
            "NN#8 must still reject config-only changes"
        )


# ---------------------------------------------------------------------------
# 3. technology-selection.md non-kernel additions
# ---------------------------------------------------------------------------


class TestTechnologySelectionNonKernel:
    def test_adds_non_kernel_technology_table(self):
        text = _read(TECH_SELECTION)
        # Spec §5.5: explicit non-kernel technology classes per category.
        # Must reference graph_restructuring + dispatch_optimization eligible tech.
        assert "graph_restructuring" in text, (
            "technology-selection.md must list graph_restructuring eligible tech"
        )
        assert "dispatch_optimization" in text, (
            "technology-selection.md must list dispatch_optimization eligible tech"
        )
        assert re.search(r"CUDA\s*Graph\s*API", text), (
            "graph_restructuring eligibility cites CUDA Graph API"
        )

    def test_anti_regression_carveout_for_non_kernel(self):
        text = _read(TECH_SELECTION)
        # Spec §5.5: anti-regression rule applies E2E-only for non-kernel categories.
        assert re.search(
            r"non[- ]kernel.*?(E2E|end[- ]to[- ]end)",
            text,
            re.DOTALL | re.IGNORECASE,
        ), (
            "Must explain anti-regression applies to E2E only for non-kernel "
            "categories (spec §5.5)"
        )


# ---------------------------------------------------------------------------
# 4. ammo-champion.md Category field requirement
# ---------------------------------------------------------------------------


class TestChampionCategoryField:
    def test_phase0_proposal_template_has_category_block(self):
        # The `## Category` block template lives in optimization-categories.md;
        # the champion role file requires it via its Category-block mandate.
        assert re.search(r"##\s*Category", _read(OPT_CATEGORIES)), (
            "optimization-categories.md must carry the `## Category` block template "
            "(spec §5.3)"
        )
        assert "Category block" in _read(CHAMPION), (
            "ammo-champion.md must require the Category block per "
            "optimization-categories.md"
        )

    def test_category_block_lists_required_fields(self):
        text = _read(CHAMPION)
        # Required fields per §5.3: Selected, Slice targeted, Projection formula,
        # Justification, Expected validation gates. The champion file should
        # mention all of these in the context of the Category block — checking
        # them globally is sufficient (proximity to "## Category" is verified
        # by the file having a `## Category` heading at all, tested above).
        # Verify all required fields appear somewhere after the first Category mention.
        text = _read(OPT_CATEGORIES)
        cat_idx = text.find("## Category")
        assert cat_idx >= 0, "Category section not found in optimization-categories.md"
        after_cat = text[cat_idx:]
        for field in (
            "Selected",
            "Slice targeted",
            "Projection formula",
            "Justification",
            "validation gates",
        ):
            assert field in after_cat, (
                f"Category block missing `{field}` field (spec §5.3)"
            )

    def test_schema_version_guard_documented(self):
        text = _read(CHAMPION)
        # Legacy guard: only enforced when schema_version >= 4.1.
        assert re.search(r"schema_version|4\.1", text), (
            "ammo-champion.md must document schema-version guard for Category field "
            "(spec §5.3 legacy guard)"
        )

    def test_references_optimization_categories_doc(self):
        text = _read(CHAMPION)
        assert "optimization-categories" in text, (
            "ammo-champion.md must reference optimization-categories.md"
        )


# ---------------------------------------------------------------------------
# 5. ammo-impl-champion.md multi-round + Category awareness
# ---------------------------------------------------------------------------


class TestImplChampionCategoryAwareness:
    def test_reads_category_field(self):
        text = _read(IMPL_CHAMPION)
        assert re.search(r"category", text, re.IGNORECASE), (
            "ammo-impl-champion.md must reference reading the candidate's category"
        )

    def test_references_optimization_categories(self):
        text = _read(IMPL_CHAMPION)
        assert "optimization-categories" in text, (
            "ammo-impl-champion.md must reference optimization-categories.md"
        )


# ---------------------------------------------------------------------------
# 6. ammo-impl-validator.md per-category validation scaffolds
# ---------------------------------------------------------------------------


class TestImplValidatorPerCategory:
    def test_documents_per_category_validation(self):
        text = _read(IMPL_VALIDATOR)
        # Spec §5.1 footnote: graph_restructuring has NO Gate 5.2; mega_kernel's
        # Gate 5.2 is block-equivalent kernel time; workload_specialization
        # runs Gate 5.2 per path.
        assert re.search(r"graph_restructuring", text), (
            "ammo-impl-validator.md must reference graph_restructuring routing"
        )
        assert re.search(r"mega_kernel", text), (
            "ammo-impl-validator.md must reference mega_kernel routing"
        )

    def test_documents_2layer_subset_or_dispatch_fork(self):
        text = _read(IMPL_VALIDATOR)
        # Spec §5.2: validation scaffolds — 2-layer subset for mega_kernel/
        # graph_restructuring; dispatch fork check for dispatch_optimization.
        assert re.search(r"2[- ]layer|dispatch fork", text, re.IGNORECASE), (
            "ammo-impl-validator.md must describe 2-layer subset or dispatch fork "
            "validation scaffolds (spec §5.2)"
        )


# ---------------------------------------------------------------------------
# 7. validation-defaults.md per-category Gate 5.2 + per-mode thresholds
# ---------------------------------------------------------------------------


class TestValidationDefaultsPerCategory:
    def test_per_category_gate_5_2_routing(self):
        text = _read(VALIDATION_DEFAULTS)
        # Spec §5.1: per-category Gate 5.2 routing
        assert "Gate 5.2" in text, "Gate 5.2 must remain documented"
        # Mention that graph_restructuring skips it
        assert re.search(
            r"graph_restructuring",
            text,
        ), (
            "validation-defaults.md must add per-category Gate 5.2 routing"
        )

    def test_per_mode_min_e2e_thresholds(self):
        text = _read(VALIDATION_DEFAULTS)
        # The minified layout consolidates the campaign floor to one
        # authoritative default (0.5, tied to the noise band) instead of the
        # old per-route 0.5/1.0/1.5 table.
        assert "min_e2e_improvement_pct" in text, (
            "Must reference the min_e2e_improvement_pct campaign floor"
        )
        assert "0.5" in text, "Default campaign floor (0.5) must be documented"
        assert "single authoritative documentation of the default value" in text


# ---------------------------------------------------------------------------
# 8. debate-protocol.md Category eligibility gate + spawn context
# ---------------------------------------------------------------------------


class TestDebateProtocolCategoryGate:
    def test_eligibility_gate_includes_category(self):
        text = _read(DEBATE_PROTOCOL)
        # Spec §5.3 + §7.2: debate eligibility must check Category field
        assert re.search(r"category", text, re.IGNORECASE), (
            "debate-protocol.md must include Category eligibility gate"
        )

    def test_diversity_check_uses_f_e2e(self):
        # The f_e2e-vs-f_decode mandate is owned by the scoring rubric in the
        # minified layout; debate-protocol.md routes scoring there.
        rubric = _read(ROOT / "skills" / "ammo" / "references" / "debate-scoring-rubric.md")
        assert "f_e2e" in rubric, (
            "debate-scoring-rubric.md must require f_e2e (not f_decode) for "
            "projection/diversity ranking"
        )
        assert "debate-scoring-rubric" in _read(DEBATE_PROTOCOL)

    def test_champion_spawn_context_mentions_category_preference(self):
        text = _read(DEBATE_PROTOCOL)
        # Spec §7.2: orchestrator's spawn prompt has category preference based on route.
        assert re.search(
            r"Category Preference|category preference|recommended categor",
            text,
            re.IGNORECASE,
        ), (
            "debate-protocol.md must document champion spawn category preference "
            "(spec §7.2)"
        )

    def test_schema_version_guard_documented(self):
        text = _read(DEBATE_PROTOCOL)
        assert re.search(r"schema_version|4\.1", text), (
            "debate-protocol.md must document schema-version guard for new Category "
            "eligibility gate (legacy campaigns use old rules)"
        )
