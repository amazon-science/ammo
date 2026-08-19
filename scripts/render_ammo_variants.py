#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Render the canonical AMMO tree into the .claude and .codex variant trees.

`ai_cli_session/_ammo_shared/` is the single source for every AMMO file that
both CLI runtimes share. This script copies each canonical file into
`ai_cli_session/.claude/...` verbatim and into `ai_cli_session/.codex/...`
after a closed set of path rewrites. Canonical files are written in the Claude
dialect, so the Claude render is a plain copy.

Usage
-----
    python scripts/render_ammo_variants.py            # write both trees
    python scripts/render_ammo_variants.py --check     # drift alarm, no writes

Why real files and not symlinks (or a shared package dir)
---------------------------------------------------------
The renderer writes REAL FILES. Symlinks were tested and disproved. They
survive `cp -r`, `cp -a`, and a tar roundtrip, but three consumers break:

1. `orchestration/cli_tool_manager.py` — `shutil.copytree(template_src, ...)`
   for the Claude and Codex workspace templates DEREFERENCES symlinks, and a
   DANGLING symlink makes copytree raise `shutil.Error`, which would hard-fail
   `POST /sessions`.
2. `orchestration/session_state.py` — `shutil.make_archive(..., "zip")` in the
   download path DEREFERENCES symlinks.
3. `Dockerfile` — `cp -a .../ai_cli_session/.codex/. /opt/codex-managed-hooks/`
   RE-ROOTS the tree, so any relative symlink escaping `.codex/` dangles. Once
   dangling, `find /opt/codex-managed-hooks -type f -exec chmod 0444` silently
   matches nothing and the next build step, the `ammo_state.py --help` smoke
   test, fails the image build.

A shared package directory fails for the same re-rooting reason and would also
need `sys.path` surgery in every script. Real files keep all five consumers
(copytree x2, `cp -a`, tar/pigz S3 checkpoint, make_archive zip) working with
zero changes, because the on-disk layout is bit-for-bit what shipped before.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANON_ROOT = REPO_ROOT / "ai_cli_session" / "_ammo_shared"

# Canonical subdir -> (claude destination, codex destination), repo-relative.
DEST_MAP: dict[str, tuple[str, str]] = {
    "scripts": (
        "ai_cli_session/.claude/skills/ammo/scripts",
        "ai_cli_session/.codex/skills/ammo/scripts",
    ),
    "references": (
        "ai_cli_session/.claude/skills/ammo/references",
        "ai_cli_session/.codex/skills/ammo/references",
    ),
    "orchestration": (
        "ai_cli_session/.claude/skills/ammo/orchestration",
        "ai_cli_session/.codex/skills/ammo/orchestration",
    ),
    "tests": (
        "ai_cli_session/.claude/skills/ammo/tests",
        "ai_cli_session/.codex/skills/ammo/tests",
    ),
    "schemas": (
        "ai_cli_session/.claude/schemas",
        "ai_cli_session/.codex/schemas",
    ),
}

# The closed path-rewrite rule set for the Codex render. Order matters: the
# two agent renames are scoped to the .claude/agents/ path form so that bare
# prose mentions of a role name are left alone.
CODEX_RULES: tuple[tuple[str, str], ...] = (
    (".claude/agents/ammo-impl-champion", "agents/ammo-implementer"),
    (".claude/agents/ammo-report-writer", "agents/ammo-report"),
    (".claude/schemas/", ".codex/schemas/"),
    (".claude/agents/", "agents/"),
    (".claude/skills/ammo/", ".codex/skills/ammo/"),
    (".claude/worktrees", ".codex/worktrees"),
    (".claude/hooks/", ".codex/hooks/"),
    ("locate .claude", "locate .codex"),
)

# Filename renames applied to the Codex render.
CODEX_FILENAME_RENAMES: tuple[tuple[str, str], ...] = (
    ("ammo-impl-champion", "ammo-implementer"),
    ("ammo-report-writer", "ammo-report"),
)

# Twins that exist in BOTH variant trees and are deliberately NOT generated.
# Each entry is the documented reason. A file listed here must be absent from
# _ammo_shared/; tests/unit/test_ammo_variant_parity.py enforces that.
PER_VARIANT: dict[str, str] = {
    # --- runtime shape: transcript format + subagent discovery ---
    "skills/ammo/scripts/transcript_filter.py": (
        "NOT GENERATED - runtime-specific. Codex adds a message normalization "
        "layer and discovers descendants from state_5.sqlite; Claude discovers "
        "them from agent-*.meta.json sidecars."
    ),
    "skills/ammo/tests/scripts/test_transcript_filter.py": (
        "NOT GENERATED - runtime-specific. Follows transcript_filter.py; the "
        "two sidecar-discovery tests have no Codex equivalent."
    ),
    # --- runtime shape: teardown / spawn primitives ---
    "skills/ammo/references/shutdown-protocol.md": (
        "NOT GENERATED - runtime-specific. Claude uses the SendMessage "
        "shutdown_request / shutdown_approved handshake; Codex has only "
        "interrupt_agent and followup_task."
    ),
    "skills/ammo/references/champion-common-patterns.md": (
        "NOT GENERATED - runtime-specific. Delegate spawn primitives differ "
        "(Agent tool with run_in_background vs Codex tasks with fork_turns)."
    ),
    "skills/ammo/orchestration/audit-protocol.md": (
        "NOT GENERATED - runtime-specific. Contains the spawn-primitive "
        "snippets for the auditor and its delegates."
    ),
    "skills/ammo/orchestration/debate-protocol.md": (
        "NOT GENERATED - runtime-specific. Champion teardown wording follows "
        "shutdown-protocol.md."
    ),
    # --- runtime shape: hook names and packaging ---
    "skills/ammo/scripts/ammo_state.py": (
        "NOT GENERATED - runtime-specific. The Codex copy carries the "
        "/opt/codex-managed-hooks packaging adapter in find_schema()."
    ),
    "skills/ammo/tests/test_ammo_state.py": (
        "NOT GENERATED - runtime-specific. Follows ammo_state.py."
    ),
    "skills/ammo/tests/test_fail_closed.py": (
        "NOT GENERATED - runtime-specific. Builds the variant's own "
        "schemas fixture tree from Path components, not a rewritable literal."
    ),
    "skills/ammo/tests/test_mining_invalidated_schema.py": (
        "NOT GENERATED - runtime-specific. Same Path-component schema fixture "
        "as test_fail_closed.py."
    ),
    "skills/ammo/tests/test_status_set_sync.py": (
        "NOT GENERATED - runtime-specific. Resolves the variant schema dir "
        "from Path parents and asserts Codex-only engine invariants."
    ),
    "skills/ammo/scripts/new_target.py": (
        "NOT GENERATED - runtime-specific. The Codex copy seeds "
        "codex_thread_id from the trusted hook identity file."
    ),
    "skills/ammo/references/gpu-pool.md": (
        "NOT GENERATED - runtime-specific. Documents each runtime's own "
        "SubagentStart/SubagentStop reservation-owner contract."
    ),
    "skills/ammo/references/artifact-layout.md": (
        "NOT GENERATED - runtime-specific. Names the variant's own layout "
        "warn hook."
    ),
    "skills/ammo/references/validation-defaults.md": (
        "NOT GENERATED - runtime-specific. Names the variant's own stop-gate "
        "hook."
    ),
    "skills/ammo/orchestration/parallel-tracks.md": (
        "NOT GENERATED - policy divergence pending adjudication. The Codex "
        "copy dropped the in-flight-tracks section."
    ),
    "skills/ammo/tests/run_all.sh": (
        "NOT GENERATED - runtime-specific. Each variant runs its own suite "
        "list."
    ),
    "skills/ammo/tests/test_guidance_semantic_contract.py": (
        "NOT GENERATED - runtime-specific. Resolves role documents through the "
        "Claude agent frontmatter registry vs the Codex bootstrap TOMLs."
    ),
    # --- prose drift, converge in a later pass ---
    "skills/ammo/tests/test_projection_accuracy_v2.py": (
        "NOT GENERATED - prose drift only. The Codex copy carries a "
        "reformatted docstring and helper signature; behavior is equal."
    ),
    "skills/ammo/tests/test_sweep_dp.py": (
        "NOT GENERATED - prose drift only. Comment rewording plus a Codex-only "
        "eval/scripts/parse_artifacts.py import path."
    ),
    "skills/ammo/tests/test_sweep_refactor.py": (
        "NOT GENERATED - coverage drift. Each variant carries cases the other "
        "lacks (flag-combination guard vs identical-config check)."
    ),
    "skills/ammo/tests/test_validation_report_phase_triage.py": (
        "NOT GENERATED - coverage drift. The Codex copy adds the significance "
        "floor case."
    ),
    "skills/ammo/tests/test_workload_matrix.py": (
        "NOT GENERATED - stale plan-path reference in the docstring header."
    ),
}

_AGENT_CONFORMANCE_REASON = (
    "NOT GENERATED - runtime-specific. Agent conformance prompts embed the "
    "runtime's spawn and teardown primitives."
)
for _name in (
    "test-champion.md",
    "test-impl-champion.md",
    "test-implementer.md",
    "test-orchestrator.md",
    "test-researcher.md",
    "test-transcript-monitor.md",
):
    PER_VARIANT[f"skills/ammo/tests/agents/{_name}"] = _AGENT_CONFORMANCE_REASON


def render_codex(data: bytes) -> bytes:
    """Apply the Codex path rewrites to canonical file *data*."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for old, new in CODEX_RULES:
        text = text.replace(old, new)
    return text.encode("utf-8")


def codex_name(name: str) -> str:
    for old, new in CODEX_FILENAME_RENAMES:
        name = name.replace(old, new)
    return name


def iter_canonical() -> list[tuple[Path, Path, Path]]:
    """Yield (canonical_path, claude_dest, codex_dest) for every shared file."""
    out: list[tuple[Path, Path, Path]] = []
    for subdir, (claude_rel, codex_rel) in sorted(DEST_MAP.items()):
        src_root = CANON_ROOT / subdir
        if not src_root.is_dir():
            continue
        for src in sorted(src_root.rglob("*")):
            if not src.is_file() or "__pycache__" in src.parts:
                continue
            rel = src.relative_to(src_root)
            claude_dest = REPO_ROOT / claude_rel / rel
            codex_rel_parts = [codex_name(p) for p in rel.parts]
            codex_dest = REPO_ROOT / codex_rel / Path(*codex_rel_parts)
            out.append((src, claude_dest, codex_dest))
    return out


def write(dest: Path, data: bytes, src: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    shutil.copymode(src, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit non-zero instead of writing files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Render into this directory instead of the repo (test harness use)",
    )
    args = ap.parse_args(argv)

    root = args.out_dir.resolve() if args.out_dir else REPO_ROOT
    files = iter_canonical()
    if not files:
        print(f"No canonical files under {CANON_ROOT}", file=sys.stderr)
        return 1

    drift: list[str] = []
    written = 0
    for src, claude_dest, codex_dest in files:
        data = src.read_bytes()
        for dest, payload in ((claude_dest, data), (codex_dest, render_codex(data))):
            target = root / dest.relative_to(REPO_ROOT)
            if args.check:
                have = target.read_bytes() if target.is_file() else b""
                if have != payload:
                    drift.append(str(dest.relative_to(REPO_ROOT)))
                continue
            if target.is_file() and target.read_bytes() == payload:
                continue
            write(target, payload, src)
            written += 1

    if args.check:
        for path in drift:
            print(f"DRIFT {path}", file=sys.stderr)
        if drift:
            print(
                f"{len(drift)} rendered file(s) differ from "
                f"ai_cli_session/_ammo_shared/. Fix with: "
                f"python scripts/render_ammo_variants.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK {len(files)} canonical file(s), both variants in sync")
        return 0

    print(f"Rendered {len(files)} canonical file(s); {written} variant file(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
