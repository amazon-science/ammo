# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for L1 Campaign Sessions Grid (Task #3).

Tests the JavaScript helper logic in campaign-app.js via Node.js subprocess,
and validates HTML template structure in index.html.
"""

import subprocess
import sys
import re
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"
INDEX_HTML = ROOT / "frontend" / "index.html"


def run_js(script: str) -> str:
    """Run a JavaScript snippet via Node.js and return stdout."""
    result = subprocess.run(
        ["node", "--input-type=commonjs", "-e",
         "eval(require('fs').readFileSync(0,'utf8'))"],
        input=script,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error: {result.stderr}")
    return result.stdout.strip()


def extract_and_run(func_name: str, call_expr: str) -> str:
    """
    Extract a named function from campaign-app.js and call it in isolation.

    The JS wraps Alpine.data inside document.addEventListener('alpine:init', cb),
    so we must invoke that callback to capture the component definition.
    """
    js_src = CAMPAIGN_APP_JS.read_text()
    # Invoke the alpine:init callback immediately so Alpine.data gets called
    script = f"""
const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{ data: (name, fn) => {{ _data = fn(); }}, directive: () => {{}}, store: () => {{}} }};
    const localStorage = {{
        getItem: () => null,
        setItem: () => {{}},
    }};
    const window = {{
        addEventListener: () => {{}},
        removeEventListener: () => {{}},
        location: {{ hash: '' }},
    }};
    const document = {{
        addEventListener: (event, cb) => {{
            if (event === 'alpine:init') _initCb = cb;
        }},
    }};
    {js_src}
    if (_initCb) _initCb();
    return _data;
}})();
const result = __componentDef.{call_expr};
if (result === null || result === undefined) {{
    console.log('');
}} else if (typeof result === 'object') {{
    console.log(JSON.stringify(result));
}} else {{
    console.log(String(result));
}}
"""
    return run_js(script)


# ---------------------------------------------------------------------------
# 1. cardClass helper
# ---------------------------------------------------------------------------

class TestCardClass:
    """cardClass(session) returns the right CSS class for status-based styling."""

    def test_active_session_gets_active_class(self):
        result = extract_and_run("cardClass", "cardClass({status: 'active'})")
        assert result == "active"

    def test_paused_session_gets_paused_class(self):
        result = extract_and_run("cardClass", "cardClass({status: 'paused'})")
        assert result == "paused"

    def test_creating_session_gets_creating_class(self):
        result = extract_and_run("cardClass", "cardClass({status: 'creating'})")
        assert result == "creating"

    def test_unknown_status_returns_empty_string(self):
        result = extract_and_run("cardClass", "cardClass({status: 'terminated'})")
        assert result == ""

    def test_missing_status_doesnt_throw(self):
        result = extract_and_run("cardClass", "cardClass({})")
        assert result == ""


# ---------------------------------------------------------------------------
# 2. mergeSessionsAndCampaigns helper
# ---------------------------------------------------------------------------

class TestMergeSessionsAndCampaigns:
    """mergeSessionsAndCampaigns merges /sessions + /api/campaigns data."""

    def test_campaign_session_gets_campaign_data_attached(self):
        result = extract_and_run("mergeSessionsAndCampaigns", """mergeSessionsAndCampaigns(
            [{session_id: 'a', status: 'active', model_name: 'DeepSeek'}],
            [{session_id: 'a', model_id: 'DeepSeek-R1', shipped_count: 2}]
        )""")
        import json
        cards = json.loads(result)
        assert len(cards) == 1
        assert cards[0]["hasCampaign"] is True
        assert cards[0]["campaign"]["shipped_count"] == 2

    def test_session_without_campaign_gets_hasCampaign_false(self):
        result = extract_and_run("mergeSessionsAndCampaigns", """mergeSessionsAndCampaigns(
            [{session_id: 'b', status: 'paused', model_name: 'Llama'}],
            []
        )""")
        import json
        cards = json.loads(result)
        assert len(cards) == 1
        assert cards[0]["hasCampaign"] is False

    def test_non_terminated_sessions_included(self):
        result = extract_and_run("mergeSessionsAndCampaigns", """mergeSessionsAndCampaigns(
            [
                {session_id: 'a', status: 'active'},
                {session_id: 'b', status: 'paused'},
                {session_id: 'c', status: 'creating'},
                {session_id: 'd', status: 'terminated'}
            ],
            []
        )""")
        import json
        cards = json.loads(result)
        # terminated sessions excluded
        ids = [c["session_id"] for c in cards]
        assert 'a' in ids
        assert 'b' in ids
        assert 'c' in ids
        assert 'd' not in ids

    def test_order_active_before_paused(self):
        """Active sessions should appear before paused ones."""
        result = extract_and_run("mergeSessionsAndCampaigns", """mergeSessionsAndCampaigns(
            [
                {session_id: 'p', status: 'paused'},
                {session_id: 'a', status: 'active'}
            ],
            []
        )""")
        import json
        cards = json.loads(result)
        statuses = [c["status"] for c in cards]
        assert statuses.index("active") < statuses.index("paused")


# ---------------------------------------------------------------------------
# 3. HTML template structure checks
# ---------------------------------------------------------------------------

class TestHtmlTemplate:
    """Validate the L1 template in index.html has the required elements."""

    @pytest.fixture(autouse=True)
    def html(self):
        self._html = INDEX_HTML.read_text()

    def test_new_session_button_present(self):
        """L1 view must have a '+ New Session' or 'New Session' button."""
        assert re.search(r'[Nn]ew\s+[Ss]ession', self._html), \
            "L1 view missing '+ New Session' button"

    def test_card_class_binding_present(self):
        """Campaign cards must bind status class via :class or x-bind:class."""
        assert re.search(r"cardStateClass\(|cardClass\(", self._html), \
            "Campaign cards missing cardStateClass() or cardClass() binding"

    def test_simplified_card_for_no_campaign(self):
        """Non-campaign sessions need a simplified card layout."""
        # Should have a template/section that shows terminal button for non-campaign sessions
        assert re.search(r"hasCampaign|has_campaign|openTerminal|terminal.*btn|btn.*terminal",
                         self._html, re.IGNORECASE), \
            "Missing simplified non-campaign card layout or terminal button"

    def test_paused_opacity_in_css(self):
        """CSS must define opacity/style for paused cards."""
        css_text = (ROOT / "frontend" / "css" / "lightgrid.css").read_text()
        assert re.search(r'\.lg-campaign-card\.paused|card.*paused.*opacity|paused.*0\.82',
                         css_text, re.IGNORECASE), \
            "lightgrid.css missing paused card opacity rule"

    def test_creating_sweep_in_css(self):
        """CSS must have creating-line sweep animation."""
        css_text = (ROOT / "frontend" / "css" / "lightgrid.css").read_text()
        assert "creating" in css_text, "lightgrid.css missing .creating card styles"

    def test_merge_call_in_campaign_app(self):
        """campaign-app.js must call mergeSessionsAndCampaigns."""
        js_src = CAMPAIGN_APP_JS.read_text()
        assert "mergeSessionsAndCampaigns" in js_src, \
            "campaign-app.js missing mergeSessionsAndCampaigns call"

    def test_sessions_endpoint_fetched_in_campaign_app(self):
        """campaign-app.js must fetch the local /sessions endpoint."""
        js_src = CAMPAIGN_APP_JS.read_text()
        assert "/sessions" in js_src, \
            "campaign-app.js does not fetch /sessions endpoint"


# ---------------------------------------------------------------------------
# 4. Edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesL1:
    """Edge cases not covered by implementor tests."""

    def test_failed_sessions_excluded(self):
        """'failed' sessions must be excluded just like 'terminated'."""
        result = extract_and_run("mergeSessionsAndCampaigns", """mergeSessionsAndCampaigns(
            [
                {session_id: 'a', status: 'active'},
                {session_id: 'b', status: 'failed'},
                {session_id: 'c', status: 'terminated'},
            ],
            []
        )""")
        import json
        cards = json.loads(result)
        ids = [c["session_id"] for c in cards]
        assert 'b' not in ids, "'failed' session should be excluded from L1 grid"
        assert 'c' not in ids, "'terminated' session should be excluded"
        assert 'a' in ids, "'active' session should be included"

    def test_formatSpeedup_exactly_1_shows_baseline(self):
        """formatSpeedup(1.0) should show '1.00x' not a raw number."""
        result = extract_and_run("formatSpeedup", "formatSpeedup(1.0)")
        assert result == "1.00x"

    def test_formatSpeedup_null_shows_baseline(self):
        """formatSpeedup(null) should not throw, should return '1.00x'."""
        result = extract_and_run("formatSpeedup", "formatSpeedup(null)")
        assert result == "1.00x"
