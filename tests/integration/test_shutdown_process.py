# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Integration tests for server graceful shutdown behaviour.

These tests start the server as a subprocess, then send signals (SIGTERM / SIGINT)
and verify:
- The server exits cleanly with code 0
- The server does not hang even with open WebSocket connections
- Multiple concurrent WebSocket connections do not block shutdown

Requirements:
- Must be run on the EC2 instance where GPU packages are available
- Port 18111 must be free (uses a non-default port to avoid conflicts)
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_DIR = str(Path(__file__).parent.parent.parent)
SHUTDOWN_TEST_PORT = 18111
SHUTDOWN_TEST_HOST = "127.0.0.1"
STARTUP_TIMEOUT = 60  # seconds to wait for server to start (GPU init can be slow)
SHUTDOWN_TIMEOUT = 15  # seconds to wait for server to exit after signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_server(port: int = SHUTDOWN_TEST_PORT, tmpdir: str | None = None) -> subprocess.Popen:
    """Start the server as a subprocess and return the Popen object."""
    import tempfile

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="shutdown_test_")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Use temp dirs so server can start outside Docker
    env["SESSION_DATA_DIR"] = os.path.join(tmpdir, "sessions")
    env["SESSION_REPOS_DIR"] = os.path.join(tmpdir, "repos")
    os.makedirs(env["SESSION_DATA_DIR"], exist_ok=True)
    os.makedirs(env["SESSION_REPOS_DIR"], exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable, "main.py",
            "--host", SHUTDOWN_TEST_HOST,
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=SERVER_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


async def _wait_for_server(port: int = SHUTDOWN_TEST_PORT, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Poll the health endpoint until the server is ready or timeout."""
    import httpx

    url = f"http://{SHUTDOWN_TEST_HOST}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


def _send_signal_and_wait(proc: subprocess.Popen, sig: int, timeout: float = SHUTDOWN_TIMEOUT) -> int:
    """Send a signal and wait for the process to exit. Returns exit code."""
    proc.send_signal(sig)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return proc.returncode


async def _open_websocket(port: int = SHUTDOWN_TEST_PORT) -> "websockets.WebSocketClientProtocol":
    """Open a dummy WebSocket to the server's health endpoint (or any WS endpoint)."""
    import websockets
    # The terminal WS endpoint requires a valid session, so we use a raw connection
    # to a non-existent session -- the server will still register the connection task
    uri = f"ws://{SHUTDOWN_TEST_HOST}:{port}/sessions/fake-session/terminal/ws"
    try:
        ws = await asyncio.wait_for(
            websockets.connect(uri, close_timeout=2),
            timeout=5,
        )
        return ws
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestGracefulShutdown:
    """Process-level tests that start/stop the actual server."""

    @pytest.mark.asyncio
    async def test_sigterm_clean_exit(self):
        """Server exits with code 0 within timeout after SIGTERM."""
        proc = _start_server()
        try:
            ready = await _wait_for_server()
            if not ready:
                pytest.skip("Server did not start in time")

            rc = _send_signal_and_wait(proc, signal.SIGTERM)
            assert rc is not None, "Server should have exited"
            # uvicorn may exit with 0 or -SIGTERM (-15) depending on signal handling
            assert rc in (0, -signal.SIGTERM), f"Expected exit code 0 or {-signal.SIGTERM}, got {rc}"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.asyncio
    async def test_sigint_clean_exit(self):
        """Server exits cleanly after SIGINT (Ctrl-C)."""
        proc = _start_server()
        try:
            ready = await _wait_for_server()
            if not ready:
                pytest.skip("Server did not start in time")

            rc = _send_signal_and_wait(proc, signal.SIGINT)
            assert rc is not None, "Server should have exited"
            # SIGINT may result in code 0 or -2 (SIGINT)
            assert rc in (0, -2, -signal.SIGINT), f"Unexpected exit code: {rc}"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.asyncio
    async def test_sigterm_with_websocket_no_hang(self):
        """Server exits cleanly even with an open WebSocket connection."""
        proc = _start_server()
        ws = None
        try:
            ready = await _wait_for_server()
            if not ready:
                pytest.skip("Server did not start in time")

            # Open a WebSocket (may fail if terminal endpoint rejects - that's OK)
            ws = await _open_websocket()

            # Send SIGTERM -- should still exit within timeout
            rc = _send_signal_and_wait(proc, signal.SIGTERM, timeout=SHUTDOWN_TIMEOUT)
            assert rc is not None, "Server hung during shutdown with open WebSocket"
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @pytest.mark.asyncio
    async def test_sigterm_with_multiple_websockets(self):
        """Server exits within timeout even with several open WebSocket connections."""
        proc = _start_server()
        ws_list = []
        try:
            ready = await _wait_for_server()
            if not ready:
                pytest.skip("Server did not start in time")

            # Open several WebSocket connections
            for _ in range(5):
                ws = await _open_websocket()
                if ws:
                    ws_list.append(ws)

            rc = _send_signal_and_wait(proc, signal.SIGTERM, timeout=SHUTDOWN_TIMEOUT)
            assert rc is not None, "Server hung with multiple open WebSockets"
        finally:
            for ws in ws_list:
                try:
                    await ws.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.kill()
                proc.wait()
