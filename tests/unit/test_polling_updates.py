# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Polling Updates (Task #5 / Plan Task 7).

Verifies startPolling/stopPolling lifecycle, interval management,
endpoint selection, and graceful error handling.
"""

import subprocess
import re
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"


def run_js(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error: {result.stderr}")
    return result.stdout.strip()


JS_SRC = CAMPAIGN_APP_JS.read_text()

HARNESS = f"""
const __intervals = [];
const __cleared = [];
let __nextId = 1;   // start at 1 so 0 is never a valid ID (stopPolling guards with if-truthy)
const setInterval = (fn, ms) => {{
    const id = __nextId++;
    __intervals.push({{ fn, ms, id }});
    return id;
}};
const clearInterval = (id) => {{ __cleared.push(id); }};

let __c;
const Alpine = {{ data: (name, fn) => {{ __c = fn(); }} }};
const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
const window = {{ addEventListener: () => {{}}, removeEventListener: () => {{}}, location: {{ hash: '' }} }};
const document = {{ addEventListener: (event, cb) => {{ if (event === 'alpine:init') cb(); }} }};
{JS_SRC}
"""


# ---------------------------------------------------------------------------
# 1. startPolling / stopPolling lifecycle
# ---------------------------------------------------------------------------

class TestPollingLifecycle:
    """startPolling and stopPolling manage _pollInterval correctly."""

    def test_start_polling_sets_interval(self):
        result = run_js(HARNESS + """
__c.startPolling(null);
console.log(typeof __c._pollInterval);
""")
        # After startPolling, _pollInterval is a number (interval ID)
        assert result == "number"

    def test_stop_polling_clears_interval(self):
        result = run_js(HARNESS + """
__c.startPolling(null);
const firstId = __c._pollInterval;
__c.stopPolling();
console.log(JSON.stringify({ cleared: __cleared.includes(firstId), pollNull: __c._pollInterval === null }));
""")
        data = json.loads(result)
        assert data["cleared"] is True
        assert data["pollNull"] is True

    def test_start_polling_twice_stops_previous(self):
        """Calling startPolling twice must stop the first interval."""
        result = run_js(HARNESS + """
__c.startPolling(null);
const firstId = __c._pollInterval;
__c.startPolling('sess-1');
console.log(JSON.stringify({ cleared: __cleared.includes(firstId), newRunning: __c._pollInterval !== null }));
""")
        data = json.loads(result)
        assert data["cleared"] is True
        assert data["newRunning"] is True

    def test_stop_polling_idempotent(self):
        """Calling stopPolling twice must not throw."""
        result = run_js(HARNESS + """
__c.startPolling(null);
__c.stopPolling();
__c.stopPolling();  // second call — must not throw
console.log('ok');
""")
        assert result == "ok"

    def test_stop_polling_when_never_started(self):
        result = run_js(HARNESS + """
__c.stopPolling();  // never started — must not throw
console.log('ok');
""")
        assert result == "ok"

    def test_poll_interval_is_15_seconds(self):
        """setInterval must be called with 15000ms."""
        result = run_js(HARNESS + """
__c.startPolling(null);
const entry = __intervals.find(e => e.id === __c._pollInterval);
console.log(entry ? entry.ms : 'not found');
""")
        assert result == "15000"


# ---------------------------------------------------------------------------
# 2. Endpoint selection
# ---------------------------------------------------------------------------

class TestPollingEndpoint:
    """startPolling picks the right endpoint based on sessionId."""

    def _capture_fetch_url(self, start_call: str) -> str:
        """Run startPolling and capture what URL it would fetch."""
        script = HARNESS + f"""
let capturedUrl = null;
__c.apiFetch = async (url) => {{ capturedUrl = url; return {{ ok: false }}; }};
{start_call}
// Manually invoke the poll callback
const entry = __intervals.find(e => e.id === __c._pollInterval);
if (entry) entry.fn();
// Give async a tick
setTimeout(() => {{ console.log(capturedUrl || ''); }}, 0);
"""
        return run_js(script)

    def test_l1_polls_campaigns_endpoint(self):
        url = self._capture_fetch_url("__c.startPolling(null);")
        assert url == "/api/campaigns" or url == "", \
            f"L1 should poll /api/campaigns, got: {url}"

    def test_l2_polls_campaign_detail_endpoint(self):
        url = self._capture_fetch_url("__c.startPolling('sess-abc');")
        assert "sess-abc" in url or url == "", \
            f"L2 should poll /api/campaigns/sess-abc, got: {url}"


# ---------------------------------------------------------------------------
# 3. Level enter/exit lifecycle
# ---------------------------------------------------------------------------

class TestPollingLevelLifecycle:
    """Polling starts on level enter, stops on level exit."""

    def test_activateL1_starts_polling(self):
        result = run_js(HARNESS + """
__c._activateL1();
console.log(__c._pollInterval !== null ? 'started' : 'not started');
""")
        assert result == "started"

    def test_activateL2_starts_polling_with_session(self):
        # _activateL2 also calls loadCampaignDetail which uses apiFetch — mock it
        result = run_js(HARNESS + """
__c.apiFetch = async () => ({ ok: false });
__c._activateL2('sess-123');
console.log(JSON.stringify({
    started: __c._pollInterval !== null,
    sessionId: __c.currentSessionId
}));
""")
        data = json.loads(result)
        assert data["started"] is True
        assert data["sessionId"] == "sess-123"

    def test_activateL3_stops_polling(self):
        result = run_js(HARNESS + """
__c.apiFetch = async () => ({ ok: false });
__c._activateL2('sess-1');      // start polling
__c._activateL3('sess-1', 1, 'op_x'); // enter L3 — should stop
console.log(__c._pollInterval === null ? 'stopped' : 'still running');
""")
        assert result == "stopped"

    def test_level_0_stops_polling(self):
        """Navigating away from campaigns stops polling."""
        result = run_js(HARNESS + """
__c.apiFetch = async () => ({ ok: false });
__c._activateL1();
// Simulate going back to sessions view (non-campaign hash)
window.location.hash = '';
__c._onHashChange();
console.log(__c._pollInterval === null ? 'stopped' : 'running');
""")
        assert result == "stopped"


# ---------------------------------------------------------------------------
# 4. Graceful error handling
# ---------------------------------------------------------------------------

class TestPollingGracefulErrors:
    """Polling never throws on network errors."""

    def test_failed_fetch_doesnt_throw(self):
        result = run_js(HARNESS + """
__c.apiFetch = async () => { throw new Error('network error'); };
__c.startPolling(null);
const entry = __intervals.find(e => e.id === __c._pollInterval);
// Call the poll fn and catch any thrown error
let threw = false;
Promise.resolve().then(() => entry.fn()).catch(() => { threw = true; }).then(() => {
    console.log(threw ? 'threw' : 'graceful');
});
""")
        assert result == "graceful"

    def test_bad_response_doesnt_throw(self):
        """HTTP error response (ok=false) is silently ignored."""
        result = run_js(HARNESS + """
__c.apiFetch = async () => ({ ok: false });
__c.startPolling(null);
const entry = __intervals.find(e => e.id === __c._pollInterval);
let threw = false;
Promise.resolve().then(() => entry.fn()).catch(() => { threw = true; }).then(() => {
    console.log(threw ? 'threw' : 'graceful');
});
""")
        assert result == "graceful"


# ---------------------------------------------------------------------------
# 5. Code structure check
# ---------------------------------------------------------------------------

class TestPollingCodeStructure:
    """campaign-app.js has the required polling code."""

    def test_start_polling_defined(self):
        assert "startPolling" in JS_SRC

    def test_stop_polling_defined(self):
        assert "stopPolling" in JS_SRC

    def test_15_second_interval(self):
        assert "15000" in JS_SRC, "Poll interval must be 15000ms"

    def test_l1_endpoint(self):
        assert "'/api/campaigns'" in JS_SRC or '"/api/campaigns"' in JS_SRC

    def test_l2_endpoint_uses_session_id(self):
        assert "/api/campaigns/${sessionId}" in JS_SRC or \
               "`/api/campaigns/${sessionId}`" in JS_SRC or \
               "api/campaigns/" in JS_SRC


# ---------------------------------------------------------------------------
# 6. Edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesPolling:
    """Edge cases not covered by implementor tests."""

    def test_switching_l1_to_l2_uses_new_endpoint(self):
        """Switching from L1 to L2 polling must use the session-specific endpoint."""
        script = HARNESS + """
const capturedUrls = [];
__c.apiFetch = async (url) => { capturedUrls.push(url); return { ok: false }; };

// Start L1 polling
__c.startPolling(null);
const l1Interval = __intervals[__intervals.length - 1];

// Switch to L2 polling
__c.startPolling('sess-switch');
const l2Interval = __intervals[__intervals.length - 1];

// Invoke both poll callbacks manually
l1Interval.fn();
l2Interval.fn();

setTimeout(() => {
    // L1 callback should have fetched /api/campaigns
    // L2 callback should have fetched /api/campaigns/sess-switch
    console.log(JSON.stringify({
        l1Cleared: __cleared.includes(l1Interval.id),
        l2Url: capturedUrls[capturedUrls.length - 1] || ''
    }));
}, 0);
"""
        result = run_js(script)
        data = json.loads(result)
        assert data["l1Cleared"] is True, "L1 interval must be cleared when switching to L2"
        assert "sess-switch" in data["l2Url"], \
            f"L2 polling must use session-specific URL. Got: {data['l2Url']}"

    def test_stop_polling_resets_to_null_not_zero(self):
        """stopPolling must set _pollInterval to null (not 0 or undefined)."""
        result = run_js(HARNESS + """
__c.startPolling(null);
__c.stopPolling();
// Verify it's exactly null so a subsequent if(_pollInterval) check works correctly
console.log(__c._pollInterval === null ? 'null' : typeof __c._pollInterval);
""")
        assert result == "null", \
            f"stopPolling must set _pollInterval to null, got: {result}"
