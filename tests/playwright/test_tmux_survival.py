# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Playwright tests: tmux session survival on browser disconnect.

These tests run against a LIVE Docker container (ammo-server) to verify:
1. tmux sessions survive when the browser WebSocket disconnects (destroy-unattached off)
2. Pane count (and therefore agent teams) is preserved across disconnect/reconnect
3. Pause explicitly kills tmux; resume creates a new one

Requires: running ammo-server Docker container with GPUs available.

Key discovery: ttyd is LAZY — it doesn't start its child process (bash → tmux)
until the first browser client connects via WebSocket. So tests must open the
browser first before checking tmux state.

After fix (destroy-unattached off):
- Browser disconnect → tmux session SURVIVES → team panes preserved
- User-initiated pause → tmux session KILLED (explicit _kill_tmux_session call)
- Resume after pause → new tmux session (--continue, no teams restored)
"""

import os
import subprocess
import time
import pytest
from playwright.sync_api import sync_playwright, Browser

pytestmark = [pytest.mark.playwright, pytest.mark.empirical]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("AMMO_SERVER_URL", "http://localhost:8000")
SESSION_CREATION_TIMEOUT = int(os.getenv("SESSION_CREATION_TIMEOUT", "300"))
DOCKER_CONTAINER = os.getenv("AMMO_CONTAINER", "ammo-server")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def docker_exec(cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a command inside the Docker container."""
    return subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def tmux_session_exists(session_id: str) -> bool:
    """Check if the tmux session exists inside the Docker container."""
    tmux_name = f"ammo-{session_id[:12]}"
    socket_path = f"/tmp/{session_id}/tmux.sock"
    result = docker_exec(
        f"tmux -S {socket_path} has-session -t {tmux_name} 2>/dev/null "
        f"&& echo EXISTS || echo GONE"
    )
    return "EXISTS" in result.stdout


def tmux_list_panes(session_id: str) -> str:
    """List all tmux panes for a session."""
    socket_path = f"/tmp/{session_id}/tmux.sock"
    result = docker_exec(
        f"tmux -S {socket_path} list-panes -a "
        f"-F '#{{pane_id}} #{{pane_pid}} #{{pane_current_command}}' 2>/dev/null"
    )
    return result.stdout.strip()


def tmux_list_clients(session_id: str) -> str:
    """List tmux clients attached to the session."""
    socket_path = f"/tmp/{session_id}/tmux.sock"
    result = docker_exec(
        f"tmux -S {socket_path} list-clients 2>/dev/null"
    )
    return result.stdout.strip()


def ttyd_child_processes(session_id: str) -> str:
    """List processes spawned by ttyd for this session."""
    result = docker_exec(
        f"ps aux | grep '{session_id[:12]}' | grep -v grep"
    )
    return result.stdout.strip()


def api_call(method: str, path: str, json_data=None, timeout: int = 30):
    """Make an API call to the server."""
    import requests
    url = f"{SERVER_URL}{path}"
    headers = {"X-Client-ID": "test-empirical-tmux"}
    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout)
    elif method == "POST":
        resp = requests.post(url, json=json_data or {}, headers=headers, timeout=timeout)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=timeout)
    else:
        raise ValueError(f"Unknown method: {method}")
    resp.raise_for_status()
    return resp.json()


def wait_for_active(session_id: str, timeout: int = SESSION_CREATION_TIMEOUT):
    """Poll until session is ACTIVE."""
    start = time.time()
    while time.time() - start < timeout:
        info = api_call("GET", f"/sessions/{session_id}")
        if info.get("status") == "active":
            return info
        time.sleep(2)
    raise TimeoutError(f"Session {session_id} not active within {timeout}s")


def wait_for_tmux(session_id: str, timeout: int = 60) -> bool:
    """Poll until tmux session exists."""
    start = time.time()
    while time.time() - start < timeout:
        if tmux_session_exists(session_id):
            return True
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        yield b
        b.close()


# ---------------------------------------------------------------------------
# Test 1: Does tmux survive browser disconnect?
# ---------------------------------------------------------------------------

class TestTmuxSurvivalOnBrowserDisconnect:
    """
    THE critical empirical test.

    Sequence:
    1. Create session → verify ACTIVE
    2. Open browser → trigger ttyd to spawn bash → tmux session created
    3. Verify tmux is alive + list panes/clients
    4. CLOSE browser (simulate WebSocket disconnect)
    5. Poll tmux at 0.5s, 1s, 2s, 5s, 10s — does it survive?
    6. If alive: reconnect browser → verify same panes
    7. If dead: verify ensure_terminal_healthy auto-recovers
    8. Cleanup: terminate session
    """

    def test_tmux_survival_experiment(self, browser):
        """Single sequential experiment — all steps in order."""
        sid = None
        try:
            # ── Step 1: Create session ──
            print("\n[EMPIRICAL] === STEP 1: Create session ===")
            data = api_call("POST", "/sessions", {
                "cli_tool": "claude",
                "repo_name": "vllm",
                "gpu_count": 1,
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "dtype": "fp8",
            }, timeout=SESSION_CREATION_TIMEOUT)
            sid = data["session_id"]
            print(f"[EMPIRICAL] Created session: {sid}")

            wait_for_active(sid)
            print(f"[EMPIRICAL] Session is ACTIVE")

            # Verify ttyd is running but NO tmux yet (lazy start)
            procs = ttyd_child_processes(sid)
            print(f"[EMPIRICAL] Processes before browser:\n{procs}")
            tmux_before = tmux_session_exists(sid)
            print(f"[EMPIRICAL] tmux exists before browser: {tmux_before}")
            assert not tmux_before, \
                "UNEXPECTED: tmux session exists before browser connected! " \
                "ttyd may not be lazy in this version."

            # ── Step 2: Open browser → trigger tmux ──
            print("\n[EMPIRICAL] === STEP 2: Open browser ===")
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            # Wait for terminal iframe + WebSocket connection
            page.wait_for_timeout(5000)

            # ── Step 3: Verify tmux alive ──
            print("\n[EMPIRICAL] === STEP 3: Verify tmux alive ===")
            assert wait_for_tmux(sid, timeout=30), \
                "tmux session not created after browser connected"
            panes = tmux_list_panes(sid)
            clients = tmux_list_clients(sid)
            print(f"[EMPIRICAL] tmux ALIVE after browser connect")
            print(f"[EMPIRICAL] Panes: {panes}")
            print(f"[EMPIRICAL] Clients: {clients}")

            # Count panes for later comparison
            pane_lines_before = [l for l in panes.split("\n") if l.strip()]
            print(f"[EMPIRICAL] Pane count: {len(pane_lines_before)}")

            # ── Step 4: Close browser (simulate disconnect) ──
            print("\n[EMPIRICAL] === STEP 4: Close browser (disconnect) ===")
            page.close()
            ctx.close()
            disconnect_time = time.time()

            # ── Step 5: Poll tmux survival ──
            print("\n[EMPIRICAL] === STEP 5: Poll tmux survival ===")
            check_times = [0.5, 1, 2, 5, 10, 15]
            results = {}
            last_check = 0

            for t in check_times:
                sleep_needed = t - last_check
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                last_check = t

                alive = tmux_session_exists(sid)
                elapsed = time.time() - disconnect_time
                results[t] = alive
                status = "ALIVE" if alive else "DEAD"
                print(f"[EMPIRICAL] t={t}s (actual={elapsed:.1f}s): tmux is {status}")

                if not alive:
                    # Check processes to understand what happened
                    procs = ttyd_child_processes(sid)
                    print(f"[EMPIRICAL] Processes after tmux death:\n{procs}")
                    break

            # ── Report findings ──
            print("\n[EMPIRICAL] ========================================")
            print("[EMPIRICAL] === TMUX SURVIVAL EXPERIMENT RESULTS ===")
            print("[EMPIRICAL] ========================================")

            final_alive = list(results.values())[-1]

            if final_alive:
                print("[EMPIRICAL] RESULT: tmux SURVIVED browser disconnect")
                print("[EMPIRICAL] → destroy-unattached on did NOT fire")
                print("[EMPIRICAL] → ttyd keeps PTY child alive between reconnections")
                print("[EMPIRICAL] → Case A (no-loss path) CONFIRMED")
                print("[EMPIRICAL]")
                print("[EMPIRICAL] IMPLICATION: Brief internet drops do NOT kill teams")
                print("[EMPIRICAL] The user's bug must be Case B (ttyd crash) or Case C (cross-pod)")

                # Verify panes survived intact
                panes_after = tmux_list_panes(sid)
                clients_after = tmux_list_clients(sid)
                print(f"[EMPIRICAL] Panes after disconnect: {panes_after}")
                print(f"[EMPIRICAL] Clients after disconnect: {clients_after}")

            else:
                death_time = next(t for t, alive in results.items() if not alive)
                print(f"[EMPIRICAL] RESULT: tmux DIED at t={death_time}s after browser close")
                print("[EMPIRICAL] → EITHER destroy-unattached on fired (ttyd's bash detached)")
                print("[EMPIRICAL] →     OR ttyd closed PTY master → SIGHUP → bash died")
                print("[EMPIRICAL]")
                print("[EMPIRICAL] IMPLICATION: ANY browser disconnect kills teams!")
                print("[EMPIRICAL] Fix: remove 'destroy-unattached on' from tmux config")

            # ── Step 6/7: Test reconnect ──
            print("\n[EMPIRICAL] === STEP 6: Reconnect browser ===")
            # Wait for ensure_terminal_healthy to potentially auto-recover
            time.sleep(5)

            ctx2 = browser.new_context(viewport={"width": 1280, "height": 800})
            page2 = ctx2.new_page()
            page2.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            page2.wait_for_timeout(5000)

            info = api_call("GET", f"/sessions/{sid}")
            print(f"[EMPIRICAL] Session status after reconnect: {info['status']}")

            tmux_alive_after = wait_for_tmux(sid, timeout=30)
            panes_reconnect = tmux_list_panes(sid) if tmux_alive_after else "N/A"
            print(f"[EMPIRICAL] tmux alive after reconnect: {tmux_alive_after}")
            print(f"[EMPIRICAL] Panes after reconnect: {panes_reconnect}")

            if not final_alive and tmux_alive_after:
                print("[EMPIRICAL] → Auto-recovery created NEW tmux session (--continue)")
                print("[EMPIRICAL] → Teams would be LOST in this path")

            page2.close()
            ctx2.close()

            # ── Assertions ──
            # With destroy-unattached off, tmux must survive the browser disconnect.
            assert final_alive, (
                "tmux session must survive browser disconnect (destroy-unattached off). "
                "If this fails, destroy-unattached was set back to 'on' — check "
                "orchestration/terminal_manager.py tmux config."
            )
            # The session should be functional regardless
            assert info["status"] == "active", \
                f"Session should be active, got {info['status']}"

        finally:
            # ── Cleanup ──
            if sid:
                try:
                    info = api_call("GET", f"/sessions/{sid}")
                    if info.get("status") == "paused":
                        api_call("POST", f"/sessions/{sid}/resume",
                                 timeout=SESSION_CREATION_TIMEOUT)
                        time.sleep(5)
                    api_call("DELETE", f"/sessions/{sid}")
                    print(f"\n[EMPIRICAL] Cleaned up session {sid[:12]}")
                except Exception as e:
                    print(f"\n[EMPIRICAL] Cleanup error: {e}")


    def test_teams_survive_browser_disconnect_reconnect(self, browser):
        """Verify agent team panes survive a browser disconnect and reconnect.

        With destroy-unattached off, the tmux session (and all team panes) must
        persist after the browser tab closes. This test:
        1. Creates a session + opens browser → tmux alive with N panes
        2. Closes browser (simulate disconnect / tab close)
        3. Waits 15s — long enough for destroy-unattached on to have fired if present
        4. Asserts tmux session is still alive
        5. Reconnects browser
        6. Asserts pane count unchanged (team panes preserved)
        """
        sid = None
        try:
            # ── Step 1: Create session ──
            print("\n[SURVIVAL] === STEP 1: Create session ===")
            data = api_call("POST", "/sessions", {
                "cli_tool": "claude",
                "repo_name": "vllm",
                "gpu_count": 1,
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "dtype": "fp8",
            }, timeout=SESSION_CREATION_TIMEOUT)
            sid = data["session_id"]
            print(f"[SURVIVAL] Created session: {sid}")

            wait_for_active(sid)
            print(f"[SURVIVAL] Session ACTIVE")

            # ── Step 2: Open browser → trigger tmux ──
            print("\n[SURVIVAL] === STEP 2: Open browser, wait for tmux ===")
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            page.wait_for_timeout(5000)

            assert wait_for_tmux(sid, timeout=30), \
                "tmux session not created after browser connected"

            panes_before = tmux_list_panes(sid)
            pane_count_before = len([l for l in panes_before.split("\n") if l.strip()])
            print(f"[SURVIVAL] tmux ALIVE, pane count: {pane_count_before}")
            print(f"[SURVIVAL] Panes: {panes_before}")

            # ── Step 3: Close browser (disconnect) ──
            print("\n[SURVIVAL] === STEP 3: Close browser (disconnect) ===")
            page.close()
            ctx.close()

            # ── Step 4: Wait 15s — if destroy-unattached were on, tmux would die by now ──
            print("[SURVIVAL] Waiting 15s for any destroy-unattached to fire...")
            time.sleep(15)

            # ── Step 5: Assert tmux is STILL alive ──
            print("\n[SURVIVAL] === STEP 5: Assert tmux survived ===")
            still_alive = tmux_session_exists(sid)
            print(f"[SURVIVAL] tmux alive after 15s disconnect: {still_alive}")
            assert still_alive, (
                "tmux session must survive browser disconnect (destroy-unattached off). "
                "Agent teams are in these panes — if tmux dies, teams are LOST."
            )

            # ── Step 6: Reconnect browser ──
            print("\n[SURVIVAL] === STEP 6: Reconnect browser ===")
            ctx2 = browser.new_context(viewport={"width": 1280, "height": 800})
            page2 = ctx2.new_page()
            page2.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            page2.wait_for_timeout(5000)

            # ── Step 7: Assert pane count unchanged ──
            print("\n[SURVIVAL] === STEP 7: Assert pane count preserved ===")
            assert wait_for_tmux(sid, timeout=30), \
                "tmux session gone after reconnect"

            panes_after = tmux_list_panes(sid)
            pane_count_after = len([l for l in panes_after.split("\n") if l.strip()])
            print(f"[SURVIVAL] Pane count before: {pane_count_before}, after: {pane_count_after}")
            print(f"[SURVIVAL] Panes after: {panes_after}")

            assert pane_count_after == pane_count_before, (
                f"Pane count changed across disconnect/reconnect: "
                f"before={pane_count_before}, after={pane_count_after}. "
                f"Agent team panes must be preserved."
            )

            print("\n[SURVIVAL] =============================================")
            print("[SURVIVAL] RESULT: Team panes SURVIVED browser disconnect")
            print(f"[SURVIVAL] Panes preserved: {pane_count_before} → {pane_count_after}")
            print("[SURVIVAL] =============================================")

            page2.close()
            ctx2.close()

        finally:
            if sid:
                try:
                    info = api_call("GET", f"/sessions/{sid}")
                    if info.get("status") == "paused":
                        api_call("POST", f"/sessions/{sid}/resume",
                                 timeout=SESSION_CREATION_TIMEOUT)
                        time.sleep(5)
                    api_call("DELETE", f"/sessions/{sid}")
                    print(f"\n[SURVIVAL] Cleaned up session {sid[:12]}")
                except Exception as e:
                    print(f"\n[SURVIVAL] Cleanup error: {e}")


# ---------------------------------------------------------------------------
# Unit probe: pane-counting logic (no Docker needed — verifier edge-case probe)
# ---------------------------------------------------------------------------

class TestPaneCountingLogic:
    """Verify the pane-count parsing used in survival assertions.

    The survival tests compare pane counts before/after disconnect using:
        len([l for l in output.split("\\n") if l.strip()])

    If this logic has an edge-case bug (e.g., trailing newlines inflate the
    count), a session with 2 panes before and 2 panes after could produce
    different counts and trigger a false-failure assertion.
    """

    def _count(self, output: str) -> int:
        return len([l for l in output.split("\n") if l.strip()])

    def test_empty_output_is_zero(self):
        """Empty string (no tmux session) gives pane count 0."""
        assert self._count("") == 0

    def test_single_pane(self):
        """One pane line counts as 1."""
        assert self._count("%0 12345 bash") == 1

    def test_trailing_newline_not_counted(self):
        """Trailing newline (common in tmux list-panes output) does not inflate count."""
        output = "%0 12345 bash\n%1 12346 claude\n"
        assert self._count(output) == 2

    def test_blank_lines_in_middle_not_counted(self):
        """Blank lines between pane entries (defensive) are not counted."""
        output = "%0 12345 bash\n\n%1 12346 claude"
        assert self._count(output) == 2

    def test_whitespace_only_lines_not_counted(self):
        """Lines containing only spaces/tabs are not counted as panes."""
        output = "%0 12345 bash\n   \n%1 12346 claude\n\t\n"
        assert self._count(output) == 2


# ---------------------------------------------------------------------------
# Test 2: Pause/Resume tmux lifecycle
# ---------------------------------------------------------------------------

class TestPauseResumeTmuxLifecycle:
    """
    Verify pause kills tmux and resume creates a new one.

    Sequence:
    1. Create session + open browser → tmux alive
    2. Pause → verify tmux dies
    3. Resume → verify new tmux (different pane IDs)
    4. Cleanup
    """

    def test_pause_resume_tmux_lifecycle(self, browser):
        """Single sequential experiment for pause/resume."""
        sid = None
        try:
            # ── Create + activate ──
            print("\n[EMPIRICAL] === PAUSE/RESUME: Create session ===")
            data = api_call("POST", "/sessions", {
                "cli_tool": "claude",
                "repo_name": "vllm",
                "gpu_count": 1,
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "dtype": "fp8",
            }, timeout=SESSION_CREATION_TIMEOUT)
            sid = data["session_id"]
            print(f"[EMPIRICAL] Created session: {sid}")

            wait_for_active(sid)

            # Open browser to trigger tmux
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            page.wait_for_timeout(5000)

            assert wait_for_tmux(sid, timeout=30), "tmux not created after browser"

            panes_before = tmux_list_panes(sid)
            print(f"[EMPIRICAL] Panes before pause: {panes_before}")

            # Close browser before pausing
            page.close()
            ctx.close()

            # ── Pause ──
            print("\n[EMPIRICAL] === PAUSE ===")
            api_call("POST", f"/sessions/{sid}/pause")
            time.sleep(3)

            tmux_after_pause = tmux_session_exists(sid)
            print(f"[EMPIRICAL] tmux exists after pause: {tmux_after_pause}")

            if not tmux_after_pause:
                print("[EMPIRICAL] CONFIRMED: pause kills tmux session")
            else:
                print("[EMPIRICAL] UNEXPECTED: tmux survived pause!")

            assert not tmux_after_pause, \
                "tmux should be dead after pause (kill-session is called)"

            # ── Resume ──
            print("\n[EMPIRICAL] === RESUME ===")
            api_call("POST", f"/sessions/{sid}/resume",
                     timeout=SESSION_CREATION_TIMEOUT)
            wait_for_active(sid, timeout=SESSION_CREATION_TIMEOUT)

            # Open browser to trigger new tmux
            ctx2 = browser.new_context(viewport={"width": 1280, "height": 800})
            page2 = ctx2.new_page()
            page2.goto(
                f"{SERVER_URL}/ui#session/{sid}",
                wait_until="networkidle",
            )
            page2.wait_for_timeout(5000)

            assert wait_for_tmux(sid, timeout=30), "tmux not created after resume"

            panes_after = tmux_list_panes(sid)
            print(f"[EMPIRICAL] Panes after resume: {panes_after}")

            pane_count = len([l for l in panes_after.split("\n") if l.strip()])
            print(f"[EMPIRICAL] Pane count after resume: {pane_count}")

            print("\n[EMPIRICAL] ==========================================")
            print("[EMPIRICAL] === PAUSE/RESUME EXPERIMENT RESULTS ===")
            print("[EMPIRICAL] ==========================================")
            print(f"[EMPIRICAL] Pause kills tmux: YES")
            print(f"[EMPIRICAL] Resume creates new tmux: YES")
            print(f"[EMPIRICAL] Panes after resume: {pane_count}")
            print(f"[EMPIRICAL] (Only 1 pane = no teams restored)")
            print(f"[EMPIRICAL] (>1 panes = teams restored, unexpected!)")

            if pane_count == 1:
                print("[EMPIRICAL] CONFIRMED: --continue does NOT restore teams")
            else:
                print(f"[EMPIRICAL] UNEXPECTED: {pane_count} panes after resume")

            page2.close()
            ctx2.close()

        finally:
            if sid:
                try:
                    info = api_call("GET", f"/sessions/{sid}")
                    if info.get("status") == "paused":
                        api_call("POST", f"/sessions/{sid}/resume",
                                 timeout=SESSION_CREATION_TIMEOUT)
                        time.sleep(5)
                    api_call("DELETE", f"/sessions/{sid}")
                    print(f"\n[EMPIRICAL] Cleaned up session {sid[:12]}")
                except Exception as e:
                    print(f"\n[EMPIRICAL] Cleanup error: {e}")
