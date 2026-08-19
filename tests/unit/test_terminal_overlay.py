# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for Terminal Overlay + Shared Components (Task #7).

Tests terminal overlay state machine, keyboard shortcut dispatch,
and HTML/CSS structure.
"""

import subprocess
import re
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"
INDEX_HTML      = ROOT / "frontend" / "index.html"
LIGHTGRID_CSS   = ROOT / "frontend" / "css" / "lightgrid.css"


def run_js(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error: {result.stderr}")
    return result.stdout.strip()


def extract_app(call_expr: str) -> str:
    """Run a method call on the Alpine campaignApp component."""
    js_src = CAMPAIGN_APP_JS.read_text()
    script = f"""
const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{ data: (name, fn) => {{ _data = fn(); }} }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
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
# 1. Terminal overlay state machine
# ---------------------------------------------------------------------------

JS_SRC = CAMPAIGN_APP_JS.read_text()

def _make_component() -> str:
    """Boilerplate to extract the campaignApp component in Node."""
    return f"""
const __c = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{ data: (name, fn) => {{ _data = fn(); }} }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
    const window = {{ addEventListener: () => {{}}, removeEventListener: () => {{}}, location: {{ hash: '' }} }};
    const document = {{ addEventListener: (event, cb) => {{ if (event === 'alpine:init') _initCb = cb; }} }};
    {JS_SRC}
    if (_initCb) _initCb();
    return _data;
}})();
"""


class TestTerminalOverlayState:
    """Terminal overlay state machine (uses flat termMode / termSessionId)."""

    def test_term_mode_initially_closed(self):
        result = extract_app("termMode")
        assert result == "closed"

    def test_term_session_id_initially_null(self):
        result = extract_app("termSessionId")
        assert result == "" or result == "null"

    def test_open_overlay_sets_mode_half(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-123');
console.log(JSON.stringify({ mode: __c.termMode, sessionId: __c.termSessionId }));
""")
        data = json.loads(result)
        assert data["mode"] == "half"
        assert data["sessionId"] == "sess-123"

    def test_close_overlay_sets_mode_closed(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-abc');
__c.closeTerminalOverlay();
console.log(__c.termMode);
""")
        assert result == "closed"

    def test_toggle_opens_when_closed(self):
        result = run_js(_make_component() + """
__c.toggleTerminalOverlay();
console.log(__c.termMode);
""")
        assert result in ("half", "full"), f"toggle from closed should open: got {result}"

    def test_toggle_closes_when_open(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.toggleTerminalOverlay();
console.log(__c.termMode);
""")
        assert result == "closed"

    def test_set_terminal_size_full(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.setTerminalSize('full');
console.log(__c.termMode);
""")
        assert result == "full"

    def test_set_terminal_size_half(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.setTerminalSize('half');
console.log(__c.termMode);
""")
        assert result == "half"

    def test_esc_key_closes_overlay(self):
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.handleKeydown({ key: 'Escape', preventDefault: () => {} });
console.log(__c.termMode);
""")
        assert result == "closed"

    def test_ctrl_backtick_toggles_overlay(self):
        result = run_js(_make_component() + """
__c.handleKeydown({ key: '`', ctrlKey: true, preventDefault: () => {} });
console.log(__c.termMode);
""")
        assert result in ("half", "full"), \
            f"Ctrl+` from closed should open overlay: got {result}"


# ---------------------------------------------------------------------------
# 2. terminalIframeSrc helper
# ---------------------------------------------------------------------------

class TestTerminalIframeSrc:
    """terminalIframeSrc(sessionId) returns the correct ttyd URL."""

    def test_returns_terminal_url(self):
        result = extract_app("terminalIframeSrc('sess-xyz')")
        assert "sess-xyz" in result, f"Session ID missing from URL: {result}"

    def test_url_is_string(self):
        result = extract_app("terminalIframeSrc('sess-abc')")
        assert isinstance(result, str) and len(result) > 0

    def test_null_session_returns_empty_or_blank(self):
        result = extract_app("terminalIframeSrc(null)")
        assert result == "" or result == "about:blank" or result == "null"


# ---------------------------------------------------------------------------
# 3. HTML structure
# ---------------------------------------------------------------------------

class TestHtmlTerminalOverlay:
    """Terminal overlay elements present in index.html."""

    @pytest.fixture(autouse=True)
    def html(self):
        self._html = INDEX_HTML.read_text()

    def test_fab_button_present(self):
        """FAB (floating action button) for terminal must be present."""
        assert re.search(r'terminal.*fab|fab.*terminal|lg-terminal-fab|openTerminalOverlay',
                         self._html, re.IGNORECASE), \
            "Terminal FAB button missing from index.html"

    def test_overlay_element_present(self):
        """Terminal overlay container must be present (lg-term-panel or termMode)."""
        assert re.search(r'lg-term-panel|termMode|term-overlay|terminal-overlay',
                         self._html, re.IGNORECASE), \
            "Terminal overlay element missing from index.html"

    def test_iframe_binding_present(self):
        """Overlay must have an iframe for ttyd with src binding (termUrl or terminalIframeSrc)."""
        assert re.search(r'<iframe.*termUrl|termUrl.*iframe|<iframe.*terminalIframeSrc|terminalIframeSrc.*iframe',
                         self._html, re.IGNORECASE), \
            "Terminal iframe with src binding missing"

    def test_esc_key_binding_present(self):
        """Overlay must respond to Escape key (@keydown.escape or handleKeydown or termClose)."""
        assert re.search(r'Escape|keydown.*escape|escape.*keydown|handleKeydown|termClose',
                         self._html, re.IGNORECASE), \
            "Escape key binding missing from terminal overlay"

    def test_size_controls_present(self):
        """Overlay must have half/full size toggle controls (setTerminalSize or termSetMode)."""
        assert re.search(r"setTerminalSize|termSetMode|terminal.*size|size.*terminal",
                         self._html, re.IGNORECASE), \
            "Terminal size controls missing"


# ---------------------------------------------------------------------------
# 4. CSS
# ---------------------------------------------------------------------------

class TestCssTerminalOverlay:
    """lightgrid.css has the terminal overlay styles."""

    @pytest.fixture(autouse=True)
    def css(self):
        self._css = LIGHTGRID_CSS.read_text()

    def test_overlay_css_present(self):
        """Terminal overlay CSS present (lg-term-panel or lg-terminal-overlay)."""
        assert re.search(r'lg-term-panel|lg-terminal-overlay|terminal-overlay', self._css, re.IGNORECASE), \
            "Terminal overlay CSS missing from lightgrid.css"

    def test_fab_css_present(self):
        """Terminal FAB CSS present (lg-term-fab or terminal-fab or lg-fab)."""
        assert re.search(r'lg-term-fab|terminal-fab|lg-fab', self._css, re.IGNORECASE), \
            "Terminal FAB CSS missing from lightgrid.css"

    def test_uses_void_not_bg_void(self):
        """Must use --void (not deprecated --bg-void) per plan NOTE."""
        assert "--bg-void" not in self._css, \
            "lightgrid.css must use --void, not deprecated --bg-void"

    def test_uses_panel_not_bg_panel(self):
        """Must use --panel (not deprecated --bg-panel) per plan NOTE."""
        assert "--bg-panel" not in self._css, \
            "lightgrid.css must use --panel, not deprecated --bg-panel"


# ---------------------------------------------------------------------------
# 5. Edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesTerminal:
    """Edge cases not covered by implementor tests."""

    def test_esc_from_full_goes_to_half_first(self):
        """Esc from full should step down to half, not jump straight to closed."""
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.setTerminalSize('full');
__c.handleKeydown({ key: 'Escape', preventDefault: () => {} });
console.log(__c.termMode);
""")
        assert result == "half", \
            f"Esc from full should step down to half (not closed). Got: {result}"

    def test_esc_twice_from_full_closes(self):
        """Two Esc presses from full: full → half → closed."""
        result = run_js(_make_component() + """
__c.openTerminalOverlay('sess-1');
__c.setTerminalSize('full');
__c.handleKeydown({ key: 'Escape', preventDefault: () => {} });
__c.handleKeydown({ key: 'Escape', preventDefault: () => {} });
console.log(__c.termMode);
""")
        assert result == "closed", \
            f"Two Esc presses from full should reach closed. Got: {result}"

    def test_term_url_encodes_special_chars_in_api_key(self):
        """API key with special chars must be URL-encoded in termUrl."""
        result = run_js(_make_component() + r"""
__c.termMode = 'half';
// Simulate having an api key with special chars
Object.defineProperty(__c, 'apiKey', { get: () => 'key with spaces & stuff' });
const url = __c.termUrl('sess-xyz');
console.log(url);
""")
        assert " " not in result, "Spaces must be URL-encoded in termUrl"
        assert "key+with" in result or "key%20with" in result, \
            "Space in API key must be percent-encoded in URL"
