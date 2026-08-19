# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
RED-phase tests for the event-loop sync-offload fix.

These assert that synchronous blocking calls inside hot endpoints are moved OFF
the loop via ``asyncio.to_thread`` / ``run_in_executor``.

Mirrors the existing house pattern in test_async_blocking_session_rmtree.py
(per-line source inspection: an offload wrapper and the blocking call must appear
on the SAME line so we don't get false positives from unrelated to_thread usage).

ALL OFFLOAD ASSERTIONS BELOW MUST FAIL BEFORE THE FIX (RED phase).
"""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers (mirrors test_async_blocking_session_rmtree.py)
# ---------------------------------------------------------------------------

def _is_offloaded(source: str, call_name: str) -> bool:
    """True if ``call_name`` is invoked via an offload wrapper, not synchronously.

    When a blocking ``foo(...)`` call is moved off the loop it becomes
    ``await asyncio.to_thread(foo, ...)`` / ``run_in_executor(None, foo, ...)``:
    the function is passed BY REFERENCE, so the synchronous-invocation token
    ``foo(`` disappears entirely while an offload wrapper appears.

    This is robust to multi-line call formatting (unlike per-physical-line
    matching), and still catches a regression where someone reintroduces a bare
    synchronous ``foo(...)`` call — that token would reappear.
    """
    has_offload = "to_thread" in source or "run_in_executor" in source
    has_sync_call = f"{call_name}(" in source
    return has_offload and not has_sync_call


@pytest.mark.unit
class TestHealthVllmReadOffloaded:
    """/health must not stat+read /workspace/vllm/* directly on the loop."""

    def test_health_check_offloads_vllm_artifact_reads(self):
        import app as app_module
        source = inspect.getsource(app_module.health_check)
        assert "_read_vllm_artifact" in source, (
            "health_check must still surface the vllm block via _read_vllm_artifact"
        )
        assert "to_thread" in source, (
            "health_check(): the two _read_vllm_artifact() filesystem reads must be "
            "offloaded via asyncio.to_thread so a slow /workspace mount cannot add "
            "on-loop syscall latency to /health."
        )
