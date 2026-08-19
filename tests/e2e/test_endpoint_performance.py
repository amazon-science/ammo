# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
E2E performance regression tests for the AMMO sessions server.

These tests run against the actual Docker container and verify that endpoint
latency stays within acceptable bounds. They catch performance issues that
unit tests miss — like recursive globs on large worktrees, self-referential
HTTP deadlocks, and event loop starvation under concurrent load.

Requirements:
- Server must be running at AMMO_SERVER_URL (default: http://localhost:8000)

Run:
    pytest tests/e2e/test_endpoint_performance.py -v

Origin: These tests were written after a production incident where:
- A recursive glob("**/REPORT.md") on 52 sessions with 11GB worktrees caused
  list_sessions() to take 20+ seconds
- /sessions made self-referential HTTP calls that deadlocked the event loop
- Combined effect: server at 100% CPU, 504 timeouts on terminal proxy
"""

import os
import sys
import time
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://localhost:8000"

# Latency budgets (seconds). These are generous — any well-behaved server
# should beat these easily. If a test fails, something is seriously wrong.
HEALTH_BUDGET_S = 0.5
SESSIONS_BUDGET_S = 1.0
SESSIONS_POLL_BUDGET_S = 2.0

# Number of requests per latency measurement
SAMPLE_SIZE = 10

# Concurrent polling simulation
CONCURRENT_CLIENTS = 5
POLL_ROUNDS = 3
POLL_INTERVAL_S = 2  # shortened from real 10s for test speed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server_url():
    return os.getenv("AMMO_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture
def auth_headers():
    key = __import__("os").environ.get("AMMO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


@pytest.fixture(autouse=True)
def _skip_if_server_down(server_url):
    """Skip all tests if the server is not reachable."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code != 200:
            pytest.skip("Server not healthy")
    except requests.ConnectionError:
        pytest.skip("Server not reachable")


def _measure_latency(url: str, n: int = SAMPLE_SIZE, headers: dict = None) -> dict:
    """Measure endpoint latency over n requests. Returns stats dict."""
    latencies = []
    for _ in range(n):
        start = time.monotonic()
        resp = requests.get(url, headers=headers or {}, timeout=30)
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"

    latencies.sort()
    return {
        "min": latencies[0],
        "max": latencies[-1],
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": latencies[int(0.95 * len(latencies))],
        "samples": len(latencies),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEndpointLatencyBudgets:
    """Every polled endpoint must respond within its latency budget.

    These budgets are deliberately generous. If a test fails, it means the
    endpoint is orders of magnitude slower than expected — likely a bug,
    not a flaky test.
    """

    def test_health_latency(self, server_url):
        """/health must respond within 500ms p95."""
        stats = _measure_latency(f"{server_url}/health")
        assert stats["p95"] < HEALTH_BUDGET_S, (
            f"GET /health p95={stats['p95']:.3f}s exceeds budget of {HEALTH_BUDGET_S}s. "
            f"Stats: {stats}"
        )

    def test_sessions_latency(self, server_url, auth_headers):
        """/sessions must respond within 1s p95 regardless of session count.

        This test runs against whatever sessions exist in the container.
        The budget must hold whether there are 0 or 200 sessions.
        """
        stats = _measure_latency(f"{server_url}/sessions", headers=auth_headers)

        # Also capture session count for diagnostics
        resp = requests.get(f"{server_url}/sessions", headers=auth_headers, timeout=10)
        session_count = resp.json().get("total", "unknown")

        assert stats["p95"] < SESSIONS_BUDGET_S, (
            f"GET /sessions p95={stats['p95']:.3f}s with {session_count} sessions, "
            f"budget is {SESSIONS_BUDGET_S}s. Stats: {stats}"
        )

    def test_sessions_poll_latency(self, server_url, auth_headers):
        """/sessions must respond within 2s p95."""
        stats = _measure_latency(f"{server_url}/sessions", headers=auth_headers)
        assert stats["p95"] < SESSIONS_POLL_BUDGET_S, (
            f"GET /sessions p95={stats['p95']:.3f}s exceeds budget of "
            f"{SESSIONS_POLL_BUDGET_S}s. Stats: {stats}"
        )


@pytest.mark.e2e
class TestConcurrentPollingStability:
    """Simulate multiple browser tabs polling /sessions every few seconds.

    The server must maintain acceptable latency under concurrent polling load.
    This catches event loop starvation and self-referential HTTP deadlocks.
    """

    def test_concurrent_polling_latency(self, server_url, auth_headers):
        """5 concurrent clients polling /sessions must all stay under 2s p95."""
        url = f"{server_url}/sessions"

        def poll_client():
            """Simulate one browser tab polling."""
            latencies = []
            for _ in range(POLL_ROUNDS):
                start = time.monotonic()
                resp = requests.get(url, headers=auth_headers, timeout=30)
                latencies.append(time.monotonic() - start)
                assert resp.status_code == 200
                time.sleep(POLL_INTERVAL_S)
            return latencies

        all_latencies = []
        with ThreadPoolExecutor(max_workers=CONCURRENT_CLIENTS) as pool:
            futures = [pool.submit(poll_client) for _ in range(CONCURRENT_CLIENTS)]
            for f in as_completed(futures):
                all_latencies.extend(f.result())

        all_latencies.sort()
        p95 = all_latencies[int(0.95 * len(all_latencies))]
        mean = statistics.mean(all_latencies)

        assert p95 < SESSIONS_POLL_BUDGET_S, (
            f"Concurrent polling ({CONCURRENT_CLIENTS} clients × {POLL_ROUNDS} rounds): "
            f"p95={p95:.3f}s, mean={mean:.3f}s, budget={SESSIONS_POLL_BUDGET_S}s"
        )

    def test_no_latency_degradation_over_time(self, server_url, auth_headers):
        """Latency should not degrade over successive polls.

        If the first batch is fast and later batches are slow, it indicates
        resource leaks, cache thrashing, or event loop accumulation.
        """
        url = f"{server_url}/sessions"
        batches = []

        for batch_num in range(3):
            batch_latencies = []
            for _ in range(5):
                start = time.monotonic()
                resp = requests.get(url, headers=auth_headers, timeout=30)
                batch_latencies.append(time.monotonic() - start)
                assert resp.status_code == 200
            batches.append(statistics.mean(batch_latencies))
            time.sleep(POLL_INTERVAL_S)

        # Last batch should not be more than 3x the first batch
        degradation_ratio = batches[-1] / max(batches[0], 0.001)
        assert degradation_ratio < 3.0, (
            f"Latency degraded {degradation_ratio:.1f}x over {len(batches)} batches: "
            f"{[f'{b:.3f}s' for b in batches]}"
        )


@pytest.mark.e2e
class TestSessionsNoSelfRequests:
    """/sessions must not cause self-referential HTTP calls.

    When running in Docker, /sessions should return immediately
    from the local session list — not make HTTP calls back to itself.
    """

    def test_sessions_faster_than_historical_peer_timeout(self, server_url, auth_headers):
        """/sessions must respond well under the historical peer timeout.

        If it takes close to 5s, it's likely doing self-referential HTTP calls
        that are timing out.
        """
        stats = _measure_latency(
            f"{server_url}/sessions", n=5, headers=auth_headers
        )
        # Must be well under the historical peer-request timeout budget.
        assert stats["max"] < 1.0, (
            f"GET /sessions max={stats['max']:.3f}s — suspiciously close to "
            f"the historical peer timeout (5s). Possible self-referential HTTP calls. "
            f"Stats: {stats}"
        )

    def test_sessions_response_is_stable(self, server_url, auth_headers):
        """Successive local /sessions reads should return the same session IDs."""
        first = requests.get(f"{server_url}/sessions", headers=auth_headers, timeout=10)
        second = requests.get(f"{server_url}/sessions", headers=auth_headers, timeout=10)

        assert first.status_code == 200
        assert second.status_code == 200

        first_ids = {s["session_id"] for s in first.json()["sessions"]}
        second_ids = {s["session_id"] for s in second.json()["sessions"]}
        assert first_ids == second_ids
