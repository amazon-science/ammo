# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Variant-parity tests for the rendered AMMO trees.

`ai_cli_session/_ammo_shared/` is the single source for every AMMO file both
CLI runtimes share. `scripts/render_ammo_variants.py` writes it into
`ai_cli_session/.claude/...` and `ai_cli_session/.codex/...` as real files.

These tests re-render into a tmpdir and sha256-compare every generated file
against the committed trees, so editing a rendered copy instead of the
canonical one fails loudly here. They also assert the override table stays
honest: each per-variant twin still exists in both trees, and none of them is
also canonical.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
RENDERER = ROOT / "scripts" / "render_ammo_variants.py"
CANON = ROOT / "ai_cli_session" / "_ammo_shared"
CLAUDE = ROOT / "ai_cli_session" / ".claude"
CODEX = ROOT / "ai_cli_session" / ".codex"

REMEDIATION = "python scripts/render_ammo_variants.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_ammo_variants", RENDERER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_ammo_variants"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def renderer():
    return _load_renderer()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_canonical_tree_is_populated(renderer):
    files = renderer.iter_canonical()
    assert files, f"{CANON} has no canonical files"


def test_rendered_trees_match_committed_bytes(renderer, tmp_path):
    """Re-render into a tmpdir; every generated file must match what is committed."""
    rc = renderer.main(["--out-dir", str(tmp_path)])
    assert rc == 0

    mismatched: list[str] = []
    for _src, claude_dest, codex_dest in renderer.iter_canonical():
        for dest in (claude_dest, codex_dest):
            rel = dest.relative_to(ROOT)
            fresh = tmp_path / rel
            assert fresh.is_file(), f"renderer did not emit {rel}"
            if not dest.is_file():
                mismatched.append(f"{rel} (missing from the committed tree)")
                continue
            if _sha(dest.read_bytes()) != _sha(fresh.read_bytes()):
                mismatched.append(str(rel))

    assert not mismatched, (
        "Rendered AMMO files differ from ai_cli_session/_ammo_shared/. "
        "Edit the canonical file, not the rendered copy, then run: "
        f"{REMEDIATION}\n  " + "\n  ".join(mismatched)
    )


def test_check_mode_is_clean(renderer):
    """--check is the drift alarm; it must pass on a committed tree."""
    assert renderer.main(["--check"]) == 0, (
        f"render_ammo_variants.py --check reports drift. Fix with: {REMEDIATION}"
    )


def test_override_table_files_exist_in_both_variants(renderer):
    """Every declared per-variant twin is present in both trees."""
    missing: list[str] = []
    for rel in sorted(renderer.PER_VARIANT):
        for base, root in ((CLAUDE, ".claude"), (CODEX, ".codex")):
            target = base / rel
            if rel.startswith("skills/ammo/tests/agents/") and base is CODEX:
                # Agent conformance prompts follow the Codex role names.
                target = base / renderer.codex_name(rel)
            if not target.is_file():
                missing.append(f"{root}/{rel}")
    assert not missing, (
        "PER_VARIANT names files that do not exist. Drop the stale entries "
        "from scripts/render_ammo_variants.py:\n  " + "\n  ".join(missing)
    )


def test_override_table_files_are_not_also_canonical(renderer):
    """A per-variant file must not live in _ammo_shared/ as well."""
    canonical = {
        str(claude_dest.relative_to(CLAUDE))
        for _src, claude_dest, _codex in renderer.iter_canonical()
    }
    both = sorted(set(renderer.PER_VARIANT) & canonical)
    assert not both, (
        "These files are declared per-variant AND rendered from _ammo_shared. "
        "Pick one:\n  " + "\n  ".join(both)
    )


def test_override_table_reasons_are_documented(renderer):
    """Each override carries a NOT GENERATED reason header."""
    bad = [
        rel
        for rel, reason in sorted(renderer.PER_VARIANT.items())
        if not reason.startswith("NOT GENERATED") or len(reason) < 40
    ]
    assert not bad, (
        "PER_VARIANT entries need a 'NOT GENERATED - <why>' reason:\n  "
        + "\n  ".join(bad)
    )


def test_every_shared_twin_is_canonical_or_declared(renderer):
    """No silent third category: a twin is generated or explicitly per-variant."""
    canonical = {
        str(claude_dest.relative_to(CLAUDE))
        for _src, claude_dest, _codex in renderer.iter_canonical()
    }
    scopes = (
        "skills/ammo/scripts",
        "skills/ammo/references",
        "skills/ammo/orchestration",
        "skills/ammo/tests",
        "schemas",
    )
    undeclared: list[str] = []
    for scope in scopes:
        base = CLAUDE / scope
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(CLAUDE))
            codex_twin = CODEX / renderer.codex_name(rel)
            if not codex_twin.is_file():
                continue  # Claude-only file; nothing to converge.
            if rel in canonical or rel in renderer.PER_VARIANT:
                continue
            undeclared.append(rel)
    assert not undeclared, (
        "These files exist in both variant trees but are neither rendered from "
        "_ammo_shared nor declared in PER_VARIANT. Move them into "
        "_ammo_shared/ or add an override entry with a reason:\n  "
        + "\n  ".join(undeclared)
    )


def test_no_symlinks_in_rendered_trees():
    """Symlinks break copytree, make_archive, and the Dockerfile cp -a re-root."""
    links: list[str] = []
    for base in (CLAUDE, CODEX):
        for path in base.rglob("*"):
            if path.is_symlink():
                links.append(str(path.relative_to(ROOT)))
    assert not links, (
        "Symlinks found in a variant tree. The renderer must write real files "
        "(see the 'Why real files' note in scripts/render_ammo_variants.py):\n  "
        + "\n  ".join(links)
    )
