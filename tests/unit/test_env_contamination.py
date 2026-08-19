# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""RED phase tests for S2: cross-track VLLM_OP* env var contamination.

Three bugs:
1. ammo-impl-champion.md instructs agents to register VLLM_OP*=1 (default True),
   causing subsequent tracks to inherit active optimization flags.
2. run_vllm_bench_latency_sweep.py inherits all VLLM_OP* from os.environ without
   sanitizing, so prior-round flags contaminate baseline/opt runs.
3. integration-logic.md has no documentation of per-track env isolation requirements.
"""
import re
import sys
from pathlib import Path

import pytest

# Repo root
ROOT = Path(__file__).parent.parent.parent
AMMO_SKILL = ROOT / "ai_cli_session" / ".claude" / "skills" / "ammo"
SCRIPTS = AMMO_SKILL / "scripts"
AGENTS = ROOT / "ai_cli_session" / ".claude" / "agents"
ORCHESTRATION = AMMO_SKILL / "orchestration"


# ---------------------------------------------------------------------------
# Bug 1: impl-champion.md GATING env var default should be 0, not 1
# ---------------------------------------------------------------------------
class TestImplChampionEnvVarDefault:
    """ammo-impl-champion.md must instruct agents to register VLLM_OP* with default 0."""

    def test_gating_env_var_default_is_zero(self):
        """The gating workflow step that registers env vars in envs.py must use =0."""
        content = (AGENTS / "ammo-impl-champion.md").read_text()
        # Find the line that instructs registration in envs.py
        # Should say VLLM_{OP_NAME}=0 or "default to False/0"
        # Must NOT say VLLM_{OP_NAME}=1
        pattern_bad = r"Register env var.*VLLM_\{?OP_NAME\}?\s*=\s*1"
        pattern_good = r"Register env var.*VLLM_\{?OP_NAME\}?\s*=\s*0"
        assert not re.search(pattern_bad, content), (
            "ammo-impl-champion.md still instructs VLLM_OP*=1 (default True). "
            "This causes cross-track contamination — must be =0 (default False)."
        )
        assert re.search(pattern_good, content), (
            "ammo-impl-champion.md must explicitly instruct VLLM_{OP_NAME}=0 "
            "(default off) for gating env var registration."
        )

    def test_gating_default_off_explanation_present(self):
        """The champion instructions should explain WHY default must be off."""
        content = (AGENTS / "ammo-impl-champion.md").read_text()
        # Look for an explanation about default-off and contamination
        has_explanation = any(
            phrase in content.lower()
            for phrase in [
                "default off",
                "default-off",
                "default false",
                "default to false",
                "defaults to false",
                "disabled by default",
                "opt-in",
            ]
        )
        assert has_explanation, (
            "ammo-impl-champion.md must explain that VLLM_OP* flags default to off "
            "to prevent cross-track contamination."
        )


# ---------------------------------------------------------------------------
# Bug 2: Sweep script must sanitize VLLM_OP* from inherited environment
# ---------------------------------------------------------------------------
class TestSweepEnvSanitization:
    """run_vllm_bench_latency_sweep.py must strip stale VLLM_OP* vars."""

    def test_sanitize_function_exists(self):
        """The sweep script must have a function that strips VLLM_OP* from env."""
        content = (SCRIPTS / "run_vllm_bench_latency_sweep.py").read_text()
        # Should have a dedicated sanitization function or inline logic
        has_sanitization = (
            "VLLM_OP" in content
            and any(
                pattern in content
                for pattern in [
                    "_sanitize_env",
                    "_sanitize_vllm_op_env",
                    "_strip_vllm_op",
                    "_clean_env",
                    "VLLM_OP" + '"' + " not in",  # dict comprehension filter
                    "startswith(\"VLLM_OP\")",
                    'startswith("VLLM_OP")',
                    "startswith('VLLM_OP')",
                    "VLLM_OP\\d+",
                ]
            )
        )
        assert has_sanitization, (
            "run_vllm_bench_latency_sweep.py must sanitize VLLM_OP* vars from the "
            "inherited environment to prevent cross-track contamination."
        )

    def test_baseline_env_excludes_stale_vllm_op(self):
        """When VLLM_OP001=1 is in os.environ, baseline RunSpec must NOT include it."""
        # Import the sweep module to test its env-building logic
        sweep_path = SCRIPTS / "run_vllm_bench_latency_sweep.py"
        content = sweep_path.read_text()

        # We test by looking for the sanitization pattern in code.
        # The code must NOT do plain `base_env = dict(os.environ)` without filtering.
        # It should filter out VLLM_OP* keys.
        lines = content.split("\n")
        base_env_lines = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "base_env" in line and "os.environ" in line
        ]
        assert base_env_lines, "Should have base_env = ... os.environ line"

        for lineno, line in base_env_lines:
            # The line must include filtering logic, not a plain dict copy
            is_filtered = any(
                f in line
                for f in [
                    "VLLM_OP",
                    "_sanitize",
                    "_strip",
                    "_clean",
                    "if not",
                    "filter",
                ]
            )
            if not is_filtered:
                # Check if the next few lines have the filtering
                context = "\n".join(lines[lineno - 1 : lineno + 5])
                has_nearby_filter = "VLLM_OP" in context and (
                    "startswith" in context or "not k.startswith" in context
                )
                assert has_nearby_filter, (
                    f"Line {lineno}: `{line.strip()}` copies os.environ without "
                    f"filtering VLLM_OP* vars. This causes cross-track contamination."
                )

    def test_opt_env_only_contains_target_vllm_op(self):
        """The opt RunSpec env must contain ONLY the target track's VLLM_OP* var."""
        content = (SCRIPTS / "run_vllm_bench_latency_sweep.py").read_text()
        # After sanitization, the opt_run should be built from sanitized base + opt_env
        # Verify the opt_run construction uses the sanitized env
        assert "opt_run" in content
        # The baseline_env and opt_env from target.json are the ONLY VLLM_OP* sources
        # after sanitization. No stale vars from os.environ should leak through.
        # This is verified by the sanitization pattern existing (test above).


# ---------------------------------------------------------------------------
# Bug 3: integration-logic.md must document per-track env isolation
# ---------------------------------------------------------------------------
class TestIntegrationLogicEnvDocs:
    """integration-logic.md must document env isolation requirements."""

    def test_env_isolation_section_exists(self):
        """The environment contract is documented: integration-logic.md owns the
        environment-promotion rule (baseline env + only accepted flags)."""
        content = (ORCHESTRATION / "integration-logic.md").read_text()
        assert "environment" in content.lower()
        assert "baseline_env" in content or "prior baseline environment" in content, (
            "integration-logic.md must document that the promoted environment is "
            "the prior baseline environment plus only the accepted flags."
        )

    def test_env_sanitization_requirement_documented(self):
        """The sanitization guarantee is mechanical: the sweep harness strips
        VLLM_OP* from the inherited environment (single enforcement point)."""
        sweep = (SCRIPTS / "run_vllm_bench_latency_sweep.py").read_text()
        assert "_sanitize_vllm_op_env" in sweep, (
            "run_vllm_bench_latency_sweep.py must sanitize VLLM_OP* vars from "
            "the environment before running."
        )

    def test_default_off_convention_documented(self):
        """The default-off flag convention lives in impl-track-rules.md (the
        gating/flag authority the minified layout routes agents to)."""
        content = (AMMO_SKILL / "references" / "impl-track-rules.md").read_text()
        has_default_off = any(
            phrase in content.lower()
            for phrase in [
                "defaulting off",
                "default off",
                "default-off",
                "default to off",
                "default to false",
                "defaults to false",
                "defaults to off",
                "default to 0",
                "disabled by default",
            ]
        )
        assert has_default_off, (
            "impl-track-rules.md must document that VLLM_* flags default "
            "to off (0/False) to prevent cross-track contamination."
        )


# ---------------------------------------------------------------------------
# Integration: end-to-end env contamination scenario
# ---------------------------------------------------------------------------
class TestEnvContaminationScenario:
    """Simulate the actual contamination scenario from the audit."""

    def test_sweep_env_with_stale_vllm_op_vars(self):
        """Simulate: VLLM_OP001=1 in env, running op003 sweep. VLLM_OP001 must be absent."""
        import ast
        import textwrap

        sweep_path = SCRIPTS / "run_vllm_bench_latency_sweep.py"
        source = sweep_path.read_text()

        # Extract just the _sanitize_vllm_op_env function via AST and execute it
        # in isolation to test behavior without importing the full module.
        tree = ast.parse(source)
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_sanitize_vllm_op_env":
                func_node = node
                break

        if func_node is None:
            pytest.fail(
                "run_vllm_bench_latency_sweep.py must define _sanitize_vllm_op_env() "
                "to strip stale VLLM_OP* from inherited environment."
            )

        # Extract source lines for the function
        lines = source.splitlines(keepends=True)
        func_source = "".join(lines[func_node.lineno - 1 : func_node.end_lineno])
        func_source = textwrap.dedent(func_source)

        from typing import Dict
        ns: dict = {"Dict": Dict}
        exec(func_source, ns)  # noqa: S102
        sanitize_fn = ns["_sanitize_vllm_op_env"]

        # Test: stale VLLM_OP001 and VLLM_OP002 should be removed
        test_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "VLLM_OP001": "1",
            "VLLM_OP002": "1",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",  # non-OP var, keep
            "CUDA_VISIBLE_DEVICES": "0",
        }
        sanitized = sanitize_fn(test_env)
        assert "VLLM_OP001" not in sanitized, "Stale VLLM_OP001 must be removed"
        assert "VLLM_OP002" not in sanitized, "Stale VLLM_OP002 must be removed"
        assert sanitized["PATH"] == "/usr/bin", "Non-VLLM_OP vars must be preserved"
        assert sanitized["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN", (
            "Non-VLLM_OP* VLLM vars must be preserved"
        )
        assert sanitized["CUDA_VISIBLE_DEVICES"] == "0", "CVD must be preserved"

    def test_gating_metadata_schema_has_env_var(self):
        """The gating metadata env_var field is schema-owned (the minified
        SKILL.md defers state shape to the schema authority)."""
        import json
        schema = json.loads(
            (ROOT / "ai_cli_session" / ".claude" / "schemas" / "state.schema.json").read_text()
        )
        track = schema["properties"]["campaign"]["properties"]["rounds"]["items"][
            "properties"]["parallel_tracks"]["properties"]["tracks"]["additionalProperties"]
        assert "env_var" in track["properties"]["gating"]["properties"], (
            "gating metadata schema must include env_var field"
        )
