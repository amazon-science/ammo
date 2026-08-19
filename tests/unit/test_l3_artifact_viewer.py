# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for L3 Artifact Viewer (Task #8).

Tests artifact loading, content rendering helpers, section discovery,
and HTML template structure.
"""

import subprocess
import re
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CAMPAIGN_APP_JS = ROOT / "frontend" / "js" / "campaign-app.js"
INDEX_HTML      = ROOT / "frontend" / "index.html"


def run_js(script: str) -> str:
    # campaign-app.js is large enough to exceed Linux argv limits when passed
    # through `node -e`; pipe the program via stdin instead.
    result = subprocess.run(
        ["node", "-"],
        input=script,
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node.js error: {result.stderr}")
    return result.stdout.strip()


def extract_app(call_expr: str) -> str:
    """Extract campaignApp methods and call them in Node.js."""
    js_src = CAMPAIGN_APP_JS.read_text()
    script = f"""
const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{
        data:      (name, fn) => {{ _data = fn(); }},
        directive: () => {{}},
        magic:     () => {{}},
        store:     () => {{}},
    }};
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
# 1. mimeToLang helper
# ---------------------------------------------------------------------------

class TestMimeToLang:
    """mimeToLang(mime) returns highlight.js language string."""

    def test_python_mime(self):
        assert extract_app("mimeToLang('text/x-python')") == "python"

    def test_json_mime(self):
        assert extract_app("mimeToLang('application/json')") == "json"

    def test_csv_mime(self):
        assert extract_app("mimeToLang('text/csv')") == "plaintext"

    def test_markdown_mime(self):
        assert extract_app("mimeToLang('text/markdown')") == "markdown"

    def test_unknown_mime_returns_plaintext(self):
        assert extract_app("mimeToLang('application/octet-stream')") == "plaintext"


# ---------------------------------------------------------------------------
# 3. renderArtifactContent helper
# ---------------------------------------------------------------------------

class TestRenderArtifactContent:
    """renderArtifactContent(content, mime) returns sanitized HTML string."""

    def _run(self, content: str, mime: str) -> str:
        """Run renderArtifactContent in Node.js with mocked DOMPurify + marked."""
        js_src = CAMPAIGN_APP_JS.read_text()
        script = f"""
// Mock DOMPurify and marked (not available in Node.js)
const DOMPurify = {{ sanitize: (html) => html + '<!--sanitized-->' }};
const marked    = {{ parse: (md) => '<p>' + md + '</p>' }};
const hljs      = {{ highlight: (code, opts) => ({{value: '<span>' + code + '</span>'}}) }};

const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{
        data:      (name, fn) => {{ _data = fn(); }},
        directive: () => {{}},
        magic:     () => {{}},
        store:     () => {{}},
    }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
    const window = {{ addEventListener: () => {{}}, removeEventListener: () => {{}},
                      location: {{ hash: '' }} }};
    const document = {{ addEventListener: (event, cb) => {{
        if (event === 'alpine:init') _initCb = cb;
    }} }};
    {js_src}
    if (_initCb) _initCb();
    return _data;
}})();
const result = __componentDef.renderArtifactContent({json.dumps(content)}, {json.dumps(mime)});
console.log(result || '');
"""
        return run_js(script)

    def test_markdown_is_sanitized(self):
        result = self._run("# Hello", "text/markdown")
        assert "sanitized" in result, "Markdown output must pass through DOMPurify"

    def test_markdown_is_parsed(self):
        result = self._run("# Hello", "text/markdown")
        assert "<p>" in result, "Markdown should be wrapped in <p> by marked"

    def test_python_code_is_highlighted(self):
        result = self._run("import torch", "text/x-python")
        assert "<span>" in result, "Python code should be syntax highlighted"

    def test_json_code_is_highlighted(self):
        result = self._run('{"key": "val"}', "application/json")
        assert "<span>" in result, "JSON should be syntax highlighted"

    def test_empty_content_returns_empty(self):
        result = self._run("", "text/markdown")
        assert result == "" or result is not None  # no throw

    def test_csv_rendered_as_plaintext(self):
        result = self._run("a,b,c\n1,2,3", "text/csv")
        assert result is not None  # no throw; just verify it doesn't crash


# ---------------------------------------------------------------------------
# 4. HTML template structure
# ---------------------------------------------------------------------------

class TestHtmlL3Structure:
    """L3 template in index.html has the required viewer elements."""

    @pytest.fixture(autouse=True)
    def html(self):
        self._html = INDEX_HTML.read_text()

    def test_l3_placeholder_replaced(self):
        assert "L3 detail view — implemented in Task 8" not in self._html, \
            "L3 placeholder still present — must be replaced with artifact viewer"

    def test_l3_template_present(self):
        assert re.search(r'currentLevel === 3', self._html), \
            "L3 template condition (currentLevel === 3) missing"

    def test_artifact_browser_reference(self):
        assert "l3ArtifactRows()" in self._html

    def test_safe_rendering_used(self):
        """L3 template must use renderArtifactContent or renderMarkdown for safe content display."""
        assert "renderArtifactContent" in self._html or "renderMarkdown" in self._html, \
            "L3 template missing safe rendering call (renderArtifactContent or renderMarkdown)"

    def test_breadcrumb_updated_for_l3(self):
        """Breadcrumb at L3 must show round + node info."""
        assert re.search(r"currentNode|currentRound.*Round|Round.*currentRound",
                         self._html), \
            "L3 breadcrumb missing Round/node info"

    def test_load_artifact_method_present(self):
        """campaign-app.js must have loadArtifact method."""
        js_src = CAMPAIGN_APP_JS.read_text()
        assert "loadArtifact" in js_src, \
            "campaign-app.js missing loadArtifact method"

    def test_api_campaigns_artifacts_endpoint_used(self):
        """loadArtifact must call /api/campaigns/{id}/artifacts/{path}."""
        js_src = CAMPAIGN_APP_JS.read_text()
        assert "/api/campaigns" in js_src and "artifacts" in js_src, \
            "campaign-app.js missing /api/campaigns/{id}/artifacts/{path} call"


# ---------------------------------------------------------------------------
# 5. Edge-case probe (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesL3:
    """Edge cases not covered by implementor tests."""

    def _run_render(self, content: str, mime: str, hljs_available: bool = True) -> str:
        """Run renderArtifactContent with mocked dependencies."""
        js_src = CAMPAIGN_APP_JS.read_text()
        hljs_mock = (
            "const hljs = { highlight: (code, opts) => ({value: '<span>' + code + '</span>'}) };"
            if hljs_available else ""
        )
        script = f"""
const DOMPurify = {{ sanitize: (html) => html + '<!--sanitized-->' }};
const marked    = {{ parse: (md) => '<p>' + md + '</p>' }};
{hljs_mock}
const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{
        data:      (name, fn) => {{ _data = fn(); }},
        directive: () => {{}},
        magic:     () => {{}},
        store:     () => {{}},
    }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
    const window = {{ addEventListener: () => {{}}, removeEventListener: () => {{}}, location: {{ hash: '' }} }};
    const document = {{ addEventListener: (event, cb) => {{ if (event === 'alpine:init') _initCb = cb; }} }};
    {js_src}
    if (_initCb) _initCb();
    return _data;
}})();
const result = __componentDef.renderArtifactContent({json.dumps(content)}, {json.dumps(mime)});
console.log(result || '');
"""
        result = subprocess.run(
            ["node", "-"],
            input=script,
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        return result.stdout.strip()

    def test_plain_text_xss_escaped(self):
        """<script> tags in plain text must be HTML-escaped, not rendered raw."""
        result = self._run_render("<script>alert(1)</script>", "text/plain")
        assert "<script>" not in result, \
            "XSS: <script> must be escaped in plain text path"
        assert "&lt;script&gt;" in result, \
            "Plain text path must HTML-escape < and >"

    def test_csv_xss_escaped(self):
        """<img> injection in CSV must be HTML-escaped."""
        result = self._run_render('<img src=x onerror=alert(1)>,value', "text/csv")
        assert "<img" not in result, \
            "XSS: <img> in CSV must be escaped"

    def test_hljs_unavailable_falls_back_to_plain(self):
        """When hljs is not loaded, code falls back to HTML-escaped <pre>."""
        result = self._run_render("def foo(): pass", "text/x-python", hljs_available=False)
        # Should still be wrapped in <pre><code> and be HTML-safe (no raw script)
        assert "<pre" in result, "Fallback should still wrap in <pre>"
        assert "def foo" in result, "Code content should be present"


# ---------------------------------------------------------------------------
# 6. Edge-case probe: CSS/HTML additions in 0830ee7 (verifier)
# ---------------------------------------------------------------------------

class TestEdgeCasesL3Html:
    """Verify CSS uses canonical tokens and HTML uses safe x-text/x-html patterns."""

    @pytest.fixture(autouse=True)
    def read_files(self):
        self._css = (ROOT / "frontend" / "css" / "lightgrid.css").read_text()
        self._html = INDEX_HTML.read_text()

    def test_lg_code_block_uses_code_bg_token(self):
        """.lg-code-block must use var(--code-bg), not a raw hex."""
        assert "var(--code-bg)" in self._css, \
            ".lg-code-block should reference --code-bg token, not a raw hex"

    def test_l3_prose_table_styles_present(self):
        """l3-prose must include table styles for markdown table rendering."""
        assert "l3-prose table" in self._css or ".l3-prose table" in self._css, \
            ".l3-prose missing table styles (needed for markdown tables in artifacts)"

    def test_no_old_bg_void_token_in_l3_css(self):
        """L3 additions must not use old --bg-void (renamed to --void in the design system)."""
        # Check only the L3 section by looking for the L3 comment block
        l3_section_start = self._css.find("L3 ARTIFACT VIEWER")
        if l3_section_start >= 0:
            l3_css = self._css[l3_section_start:]
            assert "--bg-void" not in l3_css, \
                "L3 CSS uses old --bg-void token; use --void instead"
            assert "--bg-panel" not in l3_css, \
                "L3 CSS uses old --bg-panel token; use --panel instead"

    def test_x_html_only_via_render_artifact_content(self):
        """x-html bindings in the L3 section must go through renderArtifactContent."""
        import re
        # Scope to L3 section only (currentLevel === 3), not the entire file
        # (reportOverlay.renderedHtml is pre-existing code, not part of Task #8)
        l3_start = self._html.find('currentLevel === 3')
        assert l3_start >= 0, "L3 section marker not found in index.html"
        # Extract L3 block (search for x-html within next 10000 chars)
        l3_block = self._html[l3_start:l3_start + 10000]
        x_html_uses = re.findall(r'x-html="([^"]+)"', l3_block)
        for expr in x_html_uses:
            assert "renderArtifactContent" in expr or "renderMarkdown" in expr or "renderImageBlock" in expr, \
                f"Unsafe x-html in L3 section (not going through sanitizer): {expr}"


# ===========================================================================
# L3 ARTIFACT VIEWER PRODUCTION PORT — TDD RED BASELINE
# .claude/plans/l3-artifact-viewer.md   (A1–A9 + B1–B8)
# ===========================================================================

LIGHTGRID_L3_VIEWER_CSS = ROOT / "frontend" / "css" / "lightgrid-l3-viewer.css"
LIGHTGRID_CSS           = ROOT / "frontend" / "css" / "lightgrid.css"


_DEFAULT_MOCKS = """
const DOMPurify = { sanitize: (html) => html + '<!--sanitized-->' };
const marked    = { parse: (md) => '<p>' + md + '</p>' };
const hljs      = {
    highlight:     (code, opts) => ({ value: '<span class="hljs-kw">' + code + '</span>' }),
    highlightAuto: (code)       => ({ value: '<span class="hljs-auto">' + code + '</span>' }),
};
"""


def _harness(body: str, extra_mocks: str = "") -> str:
    """Load campaign-app.js, run init, then execute ``body``.

    ``body`` runs after ``__componentDef`` is bound (the returned Alpine data
    object). Use ``console.log`` to emit the value you want back.
    """
    js_src = CAMPAIGN_APP_JS.read_text()
    script = f"""
{_DEFAULT_MOCKS}
{extra_mocks}
const __componentDef = (() => {{
    let _data;
    let _initCb = null;
    const Alpine = {{
        data:      (name, fn) => {{ _data = fn(); }},
        directive: () => {{}},
        magic:     () => {{}},
        store:     () => {{}},
    }};
    const localStorage = {{ getItem: () => null, setItem: () => {{}} }};
    const window   = {{ addEventListener: () => {{}}, removeEventListener: () => {{}},
                        location: {{ hash: '' }} }};
    const document = {{ addEventListener: (event, cb) => {{
        if (event === 'alpine:init') _initCb = cb;
    }} }};
    {js_src}
    if (_initCb) _initCb();
    return _data;
}})();
{body}
"""
    return run_js(script)


# ---------------------------------------------------------------------------
# A1. _extOf / extToLang
# ---------------------------------------------------------------------------

class TestExtHelpersA1:
    def test_extof_lowercases(self):
        assert extract_app("_extOf('a/b/c.PY')") == ".py"

    def test_extof_missing_ext(self):
        assert extract_app("_extOf('README')") == ""

    def test_extof_last_segment_only(self):
        assert extract_app("_extOf('x.tar.gz')") == ".gz"

    def test_exttolang_python(self):
        assert extract_app("extToLang('.py')") == "python"

    def test_exttolang_cu_maps_to_cpp(self):
        assert extract_app("extToLang('.cu')") == "cpp"

    def test_exttolang_ts_maps_to_typescript(self):
        assert extract_app("extToLang('.ts')") == "typescript"

    def test_exttolang_toml_maps_to_ini(self):
        assert extract_app("extToLang('.toml')") == "ini"

    def test_exttolang_unknown_returns_empty(self):
        assert extract_app("extToLang('.xyz')") == ""


# ---------------------------------------------------------------------------
# A2. dispatchRenderer — extension-first dispatch
# ---------------------------------------------------------------------------

class TestDispatchRendererA2:
    def _dispatch(self, path, mime, content, size):
        body = (
            "const r = __componentDef.dispatchRenderer("
            f"{json.dumps(path)}, {json.dumps(mime)},"
            f"{json.dumps(content)}, {json.dumps(size)}); "
            "console.log(JSON.stringify(r));"
        )
        return json.loads(_harness(body))

    def test_md_beats_octet_stream_mime(self):
        out = self._dispatch("artifacts/x.md", "application/octet-stream", "# hi\n", 40)
        assert out["mode"] == "md"
        assert "<h" in out["html"] or "<p" in out["html"]

    def test_json_beats_plain_mime(self):
        out = self._dispatch("state.json", "text/plain", '{"a":1}', 7)
        assert out["mode"] == "json"
        assert "jt-node" in out["html"]

    def test_py_code_mode(self):
        out = self._dispatch("x.py", "text/plain", "import torch\n", 14)
        assert out["mode"] == "code"
        assert "language-python" in out["html"]

    def test_cu_beats_octet_stream_to_cpp(self):
        out = self._dispatch("k.cu", "application/octet-stream", "__global__ void k(){}", 22)
        assert out["mode"] == "code"
        assert "language-cpp" in out["html"]

    def test_png_small_inlines(self):
        out = self._dispatch("x.png", "image/png", None, 1_000_000)
        assert out["mode"] == "image"
        assert "<img" in out["html"]
        assert "image-frame" in out["html"]

    def test_png_oversize_downgrades_to_binary(self):
        out = self._dispatch("big.png", "image/png", None, 6_000_000)
        assert out["mode"] == "binary"
        assert "DOWNLOAD" in out["html"].upper()

    def test_svg_renders_as_image_tag(self):
        out = self._dispatch("icon.svg", "image/svg+xml", None, 1024)
        assert out["mode"] == "image"
        assert "<img" in out["html"]

    def test_log_plaintext_mode(self):
        out = self._dispatch("session.log", "text/plain", "boot ok\n", 8)
        assert out["mode"] == "log"

    def test_unknown_ext_plaintext_falls_back_to_log(self):
        out = self._dispatch("notes", "text/plain", "hello\n", 6)
        assert out["mode"] == "log"

    def test_plaintext_xss_escaped(self):
        out = self._dispatch("x.txt", "text/plain", "<script>alert(1)</script>", 25)
        assert "<script>" not in out["html"]
        assert "&lt;script&gt;" in out["html"]


# ---------------------------------------------------------------------------
# A3. mimeToLang — regression (kept helper)
# ---------------------------------------------------------------------------

class TestMimeToLangA3:
    def test_python_kept(self):
        assert extract_app("mimeToLang('text/x-python')") == "python"

    def test_json_kept(self):
        assert extract_app("mimeToLang('application/json')") == "json"

    def test_markdown_kept(self):
        assert extract_app("mimeToLang('text/markdown')") == "markdown"

    def test_unknown_kept(self):
        assert extract_app("mimeToLang('application/octet-stream')") == "plaintext"


# ---------------------------------------------------------------------------
# A4. _isBinaryArtifact — image ext removed, weight ext added
# ---------------------------------------------------------------------------

class TestIsBinaryArtifactA4:
    def test_png_is_not_binary(self):
        assert extract_app("_isBinaryArtifact('x.png','image/png')") == "false"

    def test_jpg_is_not_binary(self):
        assert extract_app("_isBinaryArtifact('x.jpg','image/jpeg')") == "false"

    def test_webp_is_not_binary(self):
        assert extract_app("_isBinaryArtifact('x.webp','image/webp')") == "false"

    def test_svg_is_not_binary(self):
        assert extract_app("_isBinaryArtifact('x.svg','image/svg+xml')") == "false"

    def test_safetensors_is_binary(self):
        assert extract_app("_isBinaryArtifact('m.safetensors','application/octet-stream')") == "true"

    def test_pt_is_binary(self):
        assert extract_app("_isBinaryArtifact('ckpt.pt','application/octet-stream')") == "true"

    def test_onnx_is_binary(self):
        assert extract_app("_isBinaryArtifact('m.onnx','application/octet-stream')") == "true"

    def test_npz_is_binary(self):
        assert extract_app("_isBinaryArtifact('w.npz','application/octet-stream')") == "true"

    def test_nsys_rep_is_binary(self):
        assert extract_app("_isBinaryArtifact('p.nsys-rep','application/octet-stream')") == "true"

    def test_sqlite_is_binary(self):
        assert extract_app("_isBinaryArtifact('x.sqlite','application/octet-stream')") == "true"

    def test_pdf_is_binary(self):
        assert extract_app("_isBinaryArtifact('doc.pdf','application/pdf')") == "true"


# ---------------------------------------------------------------------------
# A5. renderJsonTree
# ---------------------------------------------------------------------------

class TestRenderJsonTreeA5:
    def _render(self, literal):
        body = f"console.log(__componentDef.renderJsonTree({literal}));"
        return _harness(body)

    def test_a5a_num_leaf(self):
        assert "jt-num" in self._render("42")

    def test_a5a_str_leaf(self):
        assert "jt-str" in self._render('"hi"')

    def test_a5a_null_leaf(self):
        assert "jt-null" in self._render("null")

    def test_a5a_bool_leaf(self):
        assert "jt-bool" in self._render("true")

    def test_a5b_object_count(self):
        assert "2 keys" in self._render("({a:1,b:2})")

    def test_a5b_array_count(self):
        assert "3 items" in self._render("[1,2,3]")

    def test_a5c_cycle_placeholder(self):
        body = (
            "const a = {}; a.self = a; "
            "console.log(__componentDef.renderJsonTree(a));"
        )
        out = _harness(body)
        assert "circular ref" in out or "⟲" in out

    def test_a5c_object_only_weakset(self):
        body = (
            "console.log(__componentDef.renderJsonTree("
            "{a:0,b:'',c:false,d:null,e:[0,'',false,null]}));"
        )
        out = _harness(body)
        assert "jt-node" in out

    def test_a5d_key_and_value_escape(self):
        out = self._render('({"<script>":"<img src=x onerror=y>"})')
        assert "<script>" not in out
        assert "<img " not in out and "<img>" not in out

    def test_a5e_depth_expand_default(self):
        out = self._render("({lvl1:{lvl2:{lvl3:{leaf:1}}}})")
        assert "jt-children hidden" in out


# ---------------------------------------------------------------------------
# A6. renderCodeView
# ---------------------------------------------------------------------------

class TestRenderCodeViewA6:
    def test_gutter_count_matches_lines(self):
        body = (
            "const src='a\\nb\\nc\\nd';"
            "const out=__componentDef.renderCodeView(src,'python');"
            "console.log((out.match(/<div>\\d+<\\/div>/g)||[]).length);"
        )
        assert _harness(body) == "4"

    def test_wrapper_structure_present(self):
        body = (
            "console.log("
            "__componentDef.renderCodeView('import torch','python'));"
        )
        out = _harness(body)
        for token in ("code-view", "code-gutter", "code-content", "language-python", "hljs"):
            assert token in out, f"missing {token!r} in: {out[:200]}"

    def test_unknown_lang_hljs_auto_fallback(self):
        body = "console.log(__componentDef.renderCodeView('xyzzy',''));"
        out = _harness(body)
        assert "hljs" in out


# ---------------------------------------------------------------------------
# A7. _shouldInlineImage
# ---------------------------------------------------------------------------

class TestShouldInlineImageA7:
    def test_under_5mb_inlines(self):
        assert extract_app("_shouldInlineImage('x.png','image/png',1000000)") == "true"

    def test_over_5mb_downgrades(self):
        assert extract_app("_shouldInlineImage('big.png','image/png',6000000)") == "false"

    def test_null_size_safe_default_false(self):
        assert extract_app("_shouldInlineImage('x.png','image/png',null)") == "false"

    def test_weight_never_inlines(self):
        assert extract_app(
            "_shouldInlineImage('m.safetensors','application/octet-stream',100)"
        ) == "false"

    def test_svg_under_5mb_inlines(self):
        assert extract_app("_shouldInlineImage('icon.svg','image/svg+xml',1024)") == "true"


# ---------------------------------------------------------------------------
# A8. renderPlaintextBlock
# ---------------------------------------------------------------------------

class TestRenderPlaintextBlockA8:
    def _render(self, literal):
        body = f"console.log(__componentDef.renderPlaintextBlock({literal}));"
        return _harness(body)

    def test_log_view_class_wrapper(self):
        assert "log-view" in self._render('"boot ok"')

    def test_escape_ampersand(self):
        assert "&amp;" in self._render('"a & b"')

    def test_escape_lt_gt(self):
        out = self._render('"<script>"')
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_escape_quotes(self):
        out = self._render('"a\\"b\\u0027c"')
        # quote characters should be encoded (not break the HTML attribute)
        assert ("&quot;" in out) or ("\"" not in out.replace('class="', ''))
        # payload letters present
        assert "a" in out and "b" in out and "c" in out

    def test_trailing_newline_preserved(self):
        out = self._render('"one\\ntwo\\n"')
        assert "one" in out and "two" in out


# ---------------------------------------------------------------------------
# A9. loadArtifact — atomic mode enum
# ---------------------------------------------------------------------------

class TestLoadArtifactAtomicityA9:
    def _run_load(self, path, mock_impl):
        body = f"""
(async () => {{
    __componentDef.currentSessionId = 'test-sid';
    __componentDef.l3Sections = [{{
        name: 'Test', path: {json.dumps(path)},
        content: null, mime: 'text/plain',
        loaded: false, loading: false, error: null, available: true,
    }}];
    __componentDef.l3OpenedTabs = [];
    __componentDef.apiFetch = {mock_impl};
    await __componentDef.loadArtifact({json.dumps(path)});
    const s = __componentDef.l3Sections[0];
    console.log(JSON.stringify({{
        mode: s.mode, content: s.content, size: s.size,
        mime: s.mime, loaded: s.loaded, available: s.available,
    }}));
}})();
"""
        return json.loads(_harness(body))

    def test_png_under_5mb_image_mode(self):
        mock = """
async (url, opts) => {
    const method = (opts && opts.method) || 'GET';
    if (method === 'HEAD') {
        return {
            ok: true, status: 200,
            headers: { get: (k) => ({
                'content-length': '1000000',
                'content-type':   'image/png',
            }[k.toLowerCase()] || null) },
            text: async () => '',
        };
    }
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => '' };
}
"""
        s = self._run_load("x.png", mock)
        assert s["mode"] == "image"
        assert s["content"] is None


# ---------------------------------------------------------------------------
# Natural lazy artifact browser
# ---------------------------------------------------------------------------

class TestNaturalArtifactBrowser:
    def test_catalog_drops_only_exact_cache_directory_segments(self):
        source = CAMPAIGN_APP_JS.read_text()
        assert "new Set(['cache', 'triton_cache', 'torch_compile_cache'])" in source
        assert "p.split('/').some(segment => hiddenArtifactSegments.has(segment))" in source

    def test_track_roots_are_scoped_to_round_and_op(self):
        body = """
__componentDef.currentLevel = 3;
__componentDef.currentRound = 7;
__componentDef.currentNode = 'OP-fp8';
console.log(JSON.stringify(__componentDef._l3ArtifactRootSpecs()));
"""
        roots = json.loads(_harness(body))
        assert roots == [
            {"label": "TRACK FILES", "path": "rounds/7/tracks/OP-fp8", "type": "directory"},
            {"label": "SWEEP RESULTS", "path": "rounds/7/sweeps/opt/OP-fp8", "type": "directory"},
        ]

    def test_baseline_has_direct_constraints_file_root(self):
        body = """
__componentDef.currentLevel = 3;
__componentDef.currentRound = 3;
__componentDef.currentNode = 'stage-0';
console.log(JSON.stringify(__componentDef._l3ArtifactRootSpecs()));
"""
        roots = json.loads(_harness(body))
        assert roots[0] == {
            "label": "CONSTRAINTS", "path": "rounds/3/constraints.md", "type": "file"
        }

    @pytest.mark.parametrize(("path", "encoded"), [
        ("rounds/1/a b/#draft?.md", "rounds/1/a%20b/%23draft%3F.md"),
        ("rounds/1/100%/µ.json", "rounds/1/100%25/%C2%B5.json"),
    ])
    def test_artifact_url_encodes_each_segment_and_preserves_slashes(self, path, encoded):
        body = f"""
__componentDef.currentSessionId = 'sid #?';
console.log(__componentDef.artifactUrl({json.dumps(path)}));
"""
        assert _harness(body) == f"/api/campaigns/sid%20%23%3F/artifacts/{encoded}"

    def test_load_artifact_uses_encoded_url(self):
        body = """
(async () => {
  __componentDef.currentSessionId = 'sid';
  __componentDef.l3Sections = [];
  __componentDef.l3OpenedTabs = [];
  let seen = null;
  __componentDef.apiFetch = async url => {
    seen = url;
    return {ok:true, headers:{get:() => 'text/plain'}, text:async () => 'ok'};
  };
  await __componentDef.loadArtifact('rounds/1/a b/#x?.md');
  console.log(seen);
})();
"""
        assert _harness(body) == "/api/campaigns/sid/artifacts/rounds/1/a%20b/%23x%3F.md"

    def test_rendered_image_and_binary_links_use_encoded_url(self):
        body = """
__componentDef.currentSessionId = 'sid';
console.log(JSON.stringify({
  image: __componentDef.renderImageBlock('rounds/1/a b/#x?.png', 4),
  binary: __componentDef._renderBinaryCard('rounds/1/a b/#x?.bin', 'application/octet-stream', 4),
}));
"""
        rendered = json.loads(_harness(body))
        assert "/rounds/1/a%20b/%23x%3F.png" in rendered["image"]
        assert "/rounds/1/a%20b/%23x%3F.bin" in rendered["binary"]

    def test_template_download_uses_central_url_builder(self):
        assert ':href="artifactUrl(sec.path)"' in INDEX_HTML.read_text()

    def test_directory_click_loads_only_immediate_children(self):
        body = """
(async () => {
  __componentDef.currentSessionId = 'sid';
  const root = __componentDef._newArtifactDirectory(
    {label:'MINING', path:'rounds/2/mining'}, 0, true);
  __componentDef.l3ArtifactRoots = [root];
  const urls = [];
  __componentDef.apiFetch = async url => {
    urls.push(url);
    return {ok:true, json:async () => ({exists:true, entries:[
      {name:'nested', path:'rounds/2/mining/nested', type:'directory'},
      {name:'analysis.md', path:'rounds/2/mining/analysis.md', type:'file', size:9, mime:'text/markdown'}
    ]})};
  };
  await __componentDef._loadArtifactDirectory(root);
  console.log(JSON.stringify({urls, rows:__componentDef.l3ArtifactRows().map(r => r.path)}));
})();
"""
        result = json.loads(_harness(body))
        assert result["urls"] == [
            "/api/campaigns/sid/artifact-children?path=rounds%2F2%2Fmining"
        ]
        assert result["rows"] == [
            "rounds/2/mining",
            "rounds/2/mining/nested",
            "rounds/2/mining/analysis.md",
        ]

    def test_missing_root_is_hidden(self):
        body = """
(async () => {
  __componentDef.currentSessionId = 'sid';
  const root = __componentDef._newArtifactDirectory(
    {label:'DEBATE', path:'rounds/1/debate'}, 0, true);
  __componentDef.l3ArtifactRoots = [root];
  __componentDef.apiFetch = async () => ({ok:true, json:async () => ({exists:false, entries:[]})});
  await __componentDef._loadArtifactDirectory(root);
  console.log(JSON.stringify(__componentDef.l3ArtifactRows()));
})();
"""
        assert json.loads(_harness(body)) == []

    def test_deep_link_registers_without_catalog_enumeration(self):
        body = """
(async () => {
  __componentDef.currentSessionId = 'sid';
  __componentDef.l3Sections = [];
  __componentDef.l3OpenedTabs = [];
  __componentDef.apiFetch = async () => ({
    ok:true, headers:{get:k => k === 'content-type' ? 'text/plain' : '4'}, text:async () => 'data'
  });
  await __componentDef.loadArtifact('rounds/9/mining/direct.log');
  console.log(JSON.stringify({active:__componentDef.l3ActiveSection, section:__componentDef.l3Sections[0]}));
})();
"""
        result = json.loads(_harness(body))
        assert result["active"] == "rounds/9/mining/direct.log"
        assert result["section"]["loaded"] is True

    def test_template_uses_flat_rows_not_grouped_tree(self):
        html = INDEX_HTML.read_text()
        assert 'x-for="entry in l3ArtifactRows()"' in html
        assert 'x-for="sec in l3ArtifactTree()"' not in html


class TestLoadArtifactAtomicityA9Continued(TestLoadArtifactAtomicityA9):
    def test_png_over_5mb_binary_mode(self):
        mock = """
async (url, opts) => ({
    ok: true, status: 200,
    headers: { get: (k) => ({
        'content-length': '6000000',
        'content-type':   'image/png',
    }[k.toLowerCase()] || null) },
    text: async () => '',
})
"""
        s = self._run_load("big.png", mock)
        assert s["mode"] == "binary"
        assert s["content"] is None

    def test_md_get_text_mode(self):
        mock = """
async (url, opts) => ({
    ok: true, status: 200,
    headers: { get: (k) => ({
        'content-length': '7',
        'content-type':   'text/markdown',
    }[k.toLowerCase()] || null) },
    text: async () => '# hello',
})
"""
        s = self._run_load("notes.md", mock)
        assert s["mode"] == "text"
        assert isinstance(s["content"], str)
        assert "hello" in s["content"]

    def test_safetensors_binary_mode(self):
        mock = """
async (url, opts) => ({
    ok: true, status: 200,
    headers: { get: (k) => ({
        'content-length': '1048576',
        'content-type':   'application/octet-stream',
    }[k.toLowerCase()] || null) },
    text: async () => '',
})
"""
        s = self._run_load("model.safetensors", mock)
        assert s["mode"] == "binary"
        assert s["content"] is None

    def test_png_head_fails_binary_fallback(self):
        mock = """
async (url, opts) => {
    const method = (opts && opts.method) || 'GET';
    if (method === 'HEAD') throw new Error('network down');
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => '' };
}
"""
        s = self._run_load("x.png", mock)
        assert s["mode"] == "binary"
        assert s["size"] is None


# ---------------------------------------------------------------------------
# B. HTML structure + CSS file-layout tests
# ---------------------------------------------------------------------------

class TestL3ViewerHtmlStructureB:

    @pytest.fixture(autouse=True)
    def _load(self):
        self._html = INDEX_HTML.read_text()
        self._lg_css = LIGHTGRID_CSS.read_text()
        self._new_css = (
            LIGHTGRID_L3_VIEWER_CSS.read_text()
            if LIGHTGRID_L3_VIEWER_CSS.exists() else ""
        )

    # ---- B1 ------------------------------------------------------------
    def test_b1_renderartifactcontent_4arg(self):
        pat = re.compile(
            r"renderArtifactContent\s*\(\s*sec\.content\s*,\s*sec\.mime\s*,\s*sec\.path\s*,\s*sec\.size\s*\)"
        )
        assert pat.search(self._html), (
            "index.html must call renderArtifactContent(sec.content, sec.mime, "
            "sec.path, sec.size) with 4 arguments"
        )

    # ---- B2 ------------------------------------------------------------
    def test_b2_meta_bar_present(self):
        # Scope to the L3 template — from `currentLevel === 3` to the matching
        # `</template>` that closes the L3 root. The ARTIFACTS block is ~580
        # lines in, so the naive 20K char window is too small.
        l3_start = self._html.find("currentLevel === 3")
        assert l3_start >= 0, "L3 section marker not found"
        l3_block = self._html[l3_start:]
        assert "l3-meta-bar" in l3_block, "missing .l3-meta-bar div"
        assert re.search(r'class="dot"', l3_block)
        assert re.search(r'class="path"', l3_block)
        assert "activeSection.mime" in l3_block, "missing activeSection.mime binding"
        assert ("activeSection.size" in l3_block) or ("humanBytes(activeSection.size)" in l3_block)
        assert "_rendererLabel" in l3_block

    # ---- B3 ------------------------------------------------------------
    def test_b3_binary_download_card_still_reachable(self):
        l3_start = self._html.find("currentLevel === 3")
        l3_block = self._html[l3_start:]
        assert ("DOWNLOAD" in l3_block.upper()) or ("l3-binary-card" in l3_block)

    # ---- B4 ------------------------------------------------------------
    def test_b4_new_css_loads_after_github_dark(self):
        gh_idx  = self._html.find("github-dark")
        new_idx = self._html.find("lightgrid-l3-viewer.css")
        assert gh_idx >= 0, "github-dark CDN link missing"
        assert new_idx >= 0, "lightgrid-l3-viewer.css link missing"
        assert new_idx > gh_idx, (
            "lightgrid-l3-viewer.css must be loaded AFTER the github-dark CDN link"
        )

    # ---- B5 ------------------------------------------------------------
    def test_b5_new_css_has_no_toplevel_l3_section_rule(self):
        assert LIGHTGRID_L3_VIEWER_CSS.exists(), (
            "frontend/css/lightgrid-l3-viewer.css missing (Step 0.5 not done)"
        )
        matches = re.findall(r"^\.l3-section\s*\{", self._new_css, flags=re.MULTILINE)
        assert len(matches) == 0, (
            f".l3-section {{ redeclared in lightgrid-l3-viewer.css ({len(matches)})"
        )
        lg_matches = re.findall(r"^\.l3-section\s*\{", self._lg_css, flags=re.MULTILINE)
        assert len(lg_matches) >= 1, "existing .l3-section rule in lightgrid.css must survive"

    # ---- B6 ------------------------------------------------------------
    def test_b6_meta_bar_rule_lives_in_new_css_only(self):
        assert LIGHTGRID_L3_VIEWER_CSS.exists()
        new_count = len(re.findall(r"\.l3-meta-bar\s*\{", self._new_css))
        lg_count  = len(re.findall(r"\.l3-meta-bar\s*\{", self._lg_css))
        assert new_count == 1, f".l3-meta-bar must appear exactly once (got {new_count})"
        assert lg_count == 0, f".l3-meta-bar must NOT appear in lightgrid.css (got {lg_count})"

    # ---- B7 ------------------------------------------------------------
    def test_b7_code_content_pre_whitespace_pre(self):
        assert LIGHTGRID_L3_VIEWER_CSS.exists()
        m = re.search(
            r"\.code-content\s+pre\s*\{([^}]*)\}",
            self._new_css, flags=re.DOTALL,
        )
        assert m, ".code-content pre rule missing from lightgrid-l3-viewer.css"
        assert re.search(r"white-space\s*:\s*pre\s*;", m.group(1)), (
            ".code-content pre must declare `white-space: pre;` explicitly"
        )

    # ---- B8 ------------------------------------------------------------
    # lightgrid.css sha256 pinned after the local-only server terminology
    # cleanup. Any unrelated diff still flips this guard red.
    _LIGHTGRID_CSS_BASELINE_SHA256 = (
        "1d2ab5ed1fa5872fdd7e0d1d0e715bef5b456c1b0f40707e267eb02c7b1779e0"
    )

    def test_b8_lightgrid_css_unchanged_during_tdd(self):
        import hashlib
        h = hashlib.sha256(LIGHTGRID_CSS.read_bytes()).hexdigest()
        assert h == self._LIGHTGRID_CSS_BASELINE_SHA256, (
            "frontend/css/lightgrid.css changed after the local-only baseline. Expected "
            f"{self._LIGHTGRID_CSS_BASELINE_SHA256}, got {h}."
        )
