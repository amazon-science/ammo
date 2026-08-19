# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for the pydantic validation fix in run_vllm_bench_latency_sweep.py.

Bug: `_dataclasses.asdict(engine_args)` produces None values for unset
optional fields in nested config dataclasses (CompilationConfig, etc.).
When passed to `LLM(**ea_dict)`, pydantic rejects None for fields
expecting `list[int]`.

Fix: Filter out None values from known nested config dict keys before
passing to LLM().
"""

import ast
import re
import textwrap
from pathlib import Path

import pytest

SWEEP_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ai_cli_session"
    / ".claude"
    / "skills"
    / "ammo"
    / "scripts"
    / "run_vllm_bench_latency_sweep.py"
)


@pytest.fixture
def sweep_source():
    return SWEEP_SCRIPT.read_text()


@pytest.fixture
def strip_none_recursive(sweep_source):
    """The real `_strip_none_recursive`, lifted out of its enclosing function.

    The helper is nested inside the sweep entrypoint, so it cannot be imported.
    Extracting the live AST node keeps this a behavioral test: a refactor that
    breaks the filter fails here, and a refactor that only moves it does not.
    """
    for node in ast.walk(ast.parse(sweep_source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_none_recursive":
            namespace: dict = {}
            exec(compile(ast.Module([node], []), str(SWEEP_SCRIPT), "exec"), namespace)
            return namespace["_strip_none_recursive"]
    pytest.fail("Sweep script must define _strip_none_recursive to filter None values")


class TestPydanticNoneFiltering:
    """The sweep script must filter None values from nested config dicts
    before passing to LLM()."""

    def test_does_not_pass_raw_asdict_to_llm(self, sweep_source):
        """LLM() must NOT be called with raw _dataclasses.asdict() output.

        The pattern `LLM(**_dataclasses.asdict(engine_args))` crashes when
        nested config dataclasses have unset optional fields (produces None
        values that pydantic rejects).
        """
        # This exact pattern is the bug — it should NOT exist in the file
        assert "LLM(**_dataclasses.asdict(engine_args))" not in sweep_source, (
            "Sweep script must not pass raw _dataclasses.asdict() to LLM(). "
            "None values in nested config dicts cause pydantic ValidationError."
        )

    def test_filters_none_from_config_dicts(self, sweep_source):
        """The fix must filter None values from known nested config dict keys."""
        # Must have the intermediate ea_dict variable
        assert "ea_dict = _dataclasses.asdict(engine_args)" in sweep_source, (
            "Sweep script must store asdict result in ea_dict before filtering"
        )

        # Must filter at least compilation_config (the one that crashed)
        assert re.search(
            r"compilation_config", sweep_source
        ), "Must reference compilation_config in the filtering logic"

        # Must pass filtered dict to LLM
        assert re.search(
            r"LLM\(\*\*ea_dict\)", sweep_source
        ), "Must pass filtered ea_dict to LLM()"

    def test_filters_all_known_config_keys(self, sweep_source):
        """All known nested config keys must be filtered, not just compilation_config."""
        required_keys = [
            "compilation_config",
            "profiler_config",
            "attention_config",
            "structured_outputs_config",
        ]
        for key in required_keys:
            assert re.search(
                rf'["\']?{key}["\']?', sweep_source
            ), f"Must filter None values from '{key}' config dict"

    def test_filtering_preserves_non_none_values(self, sweep_source):
        """The filter must use isinstance check to only process dict-type configs."""
        assert re.search(
            r"isinstance\(.*dict\)", sweep_source
        ), "Must check isinstance(ea_dict.get(key), dict) before filtering"

    def test_filter_drops_none_and_keeps_values(self, strip_none_recursive):
        """Behavioral: None keys go, real values stay, False/0 survive."""
        assert strip_none_recursive(
            {"a": None, "b": 1, "c": False, "d": 0, "e": []}
        ) == {"b": 1, "c": False, "d": 0, "e": []}

    def test_filter_recurses_and_drops_emptied_dicts(self, strip_none_recursive):
        """Behavioral: nested None goes, and a dict emptied by filtering is dropped.

        This is the crash shape — compilation_config.pass_config with only unset
        optional fields must not reach LLM() as an empty dict.
        """
        assert strip_none_recursive(
            {
                "cudagraph_capture_sizes": None,
                "pass_config": {"fuse_minimax_qk_norm": None},
                "level": 3,
                "nested": {"keep": 1, "drop": None},
            }
        ) == {"level": 3, "nested": {"keep": 1}}
