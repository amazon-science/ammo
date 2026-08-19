# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Static guardrails for AMMO skill docs and new_target.py — DP-aware GPU guidance.

These tests enforce that agent-facing docs under `ai_cli_session/.claude/` and
`ai_cli_session/.codex/` tell agents to reserve `tp*dp` GPUs for E2E sweeps
(not `tp`). They also pin the `_state_json()` schema in both mirrors to include a
`"dp"` key alongside `"tp"` / `"ep"`.

Plan: .claude/plans/gpu-tp-dp-awareness.md (T4).

Scope note: `.claude` is the source mirror and `.codex` must stay semantically
aligned for AMMO guidance that is not runtime-specific.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLAUDE_AMMO_ROOT = PROJECT_ROOT / "ai_cli_session" / ".claude"
CODEX_AMMO_ROOT = PROJECT_ROOT / "ai_cli_session" / ".codex"
AMMO_ROOTS = (CLAUDE_AMMO_ROOT, CODEX_AMMO_ROOT)


def _require_path(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"Required path not found: {p}")
    return p


@pytest.mark.unit
class TestGpuPoolDocsDpAware:
    """gpu-pool.md must teach the TP×DP rule for E2E sweeps.

    The agent reads this reference when composing reservation commands. If the
    "E2E sweep" row still shows `{tp}`, an MoE session with DP>1 will only
    reserve one TP group, starving the other DP replicas and deadlocking the
    sweep (torchrun expects TP×DP processes).
    """

    def test_gpu_pool_md_says_tp_times_dp_for_e2e(self):
        for root in AMMO_ROOTS:
            path = _require_path(
                root / "skills" / "ammo" / "references" / "gpu-pool.md"
            )
            text = path.read_text(encoding="utf-8")

            # The E2E sweep row must advertise tp*dp, not just tp.
            assert "{tp*dp}" in text, (
                f"{path} E2E sweep row must reference `{{tp*dp}}` so agents "
                "reserve the full TP×DP world. Current text:\n"
                f"{text}"
            )
            # And the raw `{tp}` hint in the table should be gone.
            # Allow other occurrences of `{tp}` in sentences, but the E2E sweep
            # row must not still present bare `{tp}` as the count.
            assert "| `{tp}` |" not in text, (
                f"{path} table still shows `| `{{tp}}` |` as the count for some "
                "row. Replace with `{tp*dp}` for DP-aware sizing."
            )

            # The free-prose sentence about contiguous allocation must mention
            # TP×DP explicitly so agents understand the block shape.
            assert "TP×DP" in text or "TP*DP" in text or "tp*dp" in text, (
                f"{path} free-prose section should explain that contiguous "
                "blocks are sized to TP×DP, not just TP."
            )


@pytest.mark.unit
class TestNoStaleNumGpusTpTemplate:
    """No agent-facing AMMO doc should tell agents `--num-gpus {tp}`.

    The correct template for E2E sweeps is `{tp*dp}`. Kernel benchmarks
    (`--num-gpus 1`) are untouched — this test only flags the `{tp}` variant.
    """

    def test_no_stale_num_gpus_tp_template(self):
        md_files = []
        for root in AMMO_ROOTS:
            md_files.extend(root.rglob("*.md"))
        assert md_files, (
            "Expected to find some .md files under ai_cli_session/.claude/ or .codex/ — "
            "glob returned nothing, which suggests the path is wrong."
        )

        stale = []
        pattern = re.compile(r"--num-gpus\s+\{tp\}")
        for md in md_files:
            text = md.read_text(encoding="utf-8")
            if pattern.search(text):
                # Record the file + line for a readable failure message.
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if pattern.search(line):
                        rel = md.relative_to(PROJECT_ROOT)
                        stale.append(f"{rel}:{lineno}: {line.strip()}")

        assert not stale, (
            "Found stale `--num-gpus {tp}` references. Replace with "
            "`--num-gpus {tp*dp}` so DP>1 sessions reserve the full world:\n"
            + "\n".join(stale)
        )


@pytest.mark.unit
class TestStateJsonSchemaMentionsDp:
    """The `.claude` mirror of new_target.py must persist `dp` in state.json.

    Pins `_state_json()` against regressions: the `"target"` block needs a
    `"dp"` key so agents can recover the parallelism config from state alone
    without reopening target.json.
    """

    def test_state_json_schema_mentions_dp(self):
        for root in AMMO_ROOTS:
            path = _require_path(
                root / "skills" / "ammo" / "scripts" / "new_target.py"
            )
            source = path.read_text(encoding="utf-8")

            # Scope the scan to `_state_json()` so we don't match `"dp":` strings
            # elsewhere in the file (e.g., constraints.md text). We locate the
            # function header and then scan until the next top-level `def `.
            start = source.find("def _state_json(")
            assert start != -1, (
                f"Could not locate `def _state_json(` in {path}. "
                "Did the function name or signature change?"
            )
            next_def = source.find("\ndef ", start + 1)
            body = source[start:] if next_def == -1 else source[start:next_def]

            # The target sub-block opens with `"target": {` inside the return
            # dict. Grab everything from that line until the matching `},`
            # (the block is flat, so the first top-level `},` closes it).
            target_match = re.search(
                r'"target"\s*:\s*\{(?P<target>[^{}]*?)\},',
                body,
                re.DOTALL,
            )
            assert target_match, (
                f'Could not locate the `"target": {{ ... }}` sub-block inside '
                f"_state_json's return dict in {path}."
            )
            target_block = target_match.group("target")

            assert '"dp"' in target_block, (
                f'{path} state.json `target` block in _state_json() is missing `"dp":`. '
                'Add `"dp": fields.data_parallel_size,` alongside tp/ep so the '
                "parallelism snapshot is complete. Current target block:\n"
                f"{target_block}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
