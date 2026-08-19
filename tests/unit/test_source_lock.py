# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for source lock features: vllm.docker_commit + vllm.version blocks on
/health, nightly wheel commit guard, and cardSourceLabel JS function logic.

TestHealthDockerCommit (3 tests): guard the vllm.docker_commit field on /health.
TestHealthVllmVersion (3 tests, B-1/B-2/B-3): new vllm.version field on /health.
TestWorktreeManagerNightlyGuard (1 test): guard VLLM_PRECOMPILED_WHEEL_COMMIT source.
TestCardSourceLabelLogic (8 tests): Python mirror of JS cardSourceLabel() function.
TestCardSourceLabelGated (F-4): gated signature — version only when branch matches
                                 the docker commit (7 parameterised cases + 2 JS guards).

Note: /api/supported-models was removed as part of the static-model-selector
removal; vllm_version and vllm_docker_commit now live on /health under a
structured `vllm: {docker_commit, version}` block.
"""
import asyncio
import importlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _fresh_app_with_fake_manager(
    gpu_type: str = "l40s",
    total_gpus: int = 4,
    available_gpus: int = 2,
):
    """Reload app.py and inject a fake gpu_manager so /health can run without lifespan."""
    import app as app_module
    importlib.reload(app_module)
    app_module.gpu_type = gpu_type

    gpu_mgr = MagicMock()
    gpu_mgr.get_gpu_count.return_value = total_gpus
    gpu_mgr.get_available_gpu_count.return_value = available_gpus
    app_module.gpu_manager = gpu_mgr

    return app_module


async def _call_health(app_module):
    """Invoke the /health route handler and return the decoded JSON body + response."""
    response = await app_module.health_check()
    body = json.loads(response.body.decode("utf-8"))
    return body, response


# ---------------------------------------------------------------------------
# Python equivalent of JS cardSourceLabel(card) in campaign-app.js
# ---------------------------------------------------------------------------

def _card_source_label(branch: str) -> str:
    """Python mirror of frontend/js/campaign-app.js cardSourceLabel().

    cardSourceLabel(card) {
        const b = card.branch || '';
        if (/^[0-9a-f]{40}$/i.test(b)) return 'vllm@' + b.slice(0, 7);
        if (b === 'main') return 'vllm@main';
        return 'vllm@' + b;
    }
    """
    b = branch or ''
    if re.match(r'^[0-9a-f]{40}$', b, re.IGNORECASE):
        return 'vllm@' + b[:7]
    if b == 'main':
        return 'vllm@main'
    return 'vllm@' + b


def _card_source_label_gated(branch: str, cm_docker_commit: str | None,
                             cm_vllm_version: str | None) -> str:
    """Python mirror of the NEW gated cardSourceLabel(card) in campaign-app.js.

    Gated signature: shows the release version label ONLY when the card's
    branch equals cmDockerCommit (i.e. the session is pinned to the Docker
    image's commit). Any other branch falls back to the original behaviour.

    cardSourceLabel(card) {
        const b = card.branch || '';
        if (this.cmVllmVersion && b && b === this.cmDockerCommit) {
            return 'vllm@' + this.cmVllmVersion;
        }
        if (/^[0-9a-f]{40}$/i.test(b)) return 'vllm@' + b.slice(0, 7);
        if (b === 'main') return 'vllm@main';
        return 'vllm@' + b;
    }
    """
    b = branch or ''
    if cm_vllm_version and b and b == cm_docker_commit:
        return 'vllm@' + cm_vllm_version
    if re.match(r'^[0-9a-f]{40}$', b, re.IGNORECASE):
        return 'vllm@' + b[:7]
    if b == 'main':
        return 'vllm@main'
    return 'vllm@' + b


# ===========================================================================
# TestHealthDockerCommit (3 tests)
# ===========================================================================

@pytest.mark.unit
class TestHealthDockerCommit:
    """Tests for vllm.docker_commit field on GET /health.

    (Formerly TestSupportedModelsDockerCommit on the removed
    /api/supported-models endpoint. vllm info now lives under
    /health's `vllm: {docker_commit, version}` block.)
    """

    @pytest.mark.asyncio
    async def test_health_returns_docker_commit(self):
        """is_file()=True with a 40-char hex → vllm.docker_commit returned in response."""
        commit_hash = "a" * 40
        app_module = _fresh_app_with_fake_manager()

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=commit_hash, docker_version=None,
        )):
            data, response = await _call_health(app_module)

        assert response.status_code == 200
        assert "vllm" in data, (
            f"/health must contain 'vllm' block. Got keys: {list(data.keys())}"
        )
        assert "docker_commit" in data["vllm"], (
            f"/health vllm block must contain 'docker_commit'. "
            f"Got keys: {list(data['vllm'].keys())}"
        )
        assert data["vllm"]["docker_commit"] == commit_hash, (
            f"vllm.docker_commit must be {commit_hash!r}, "
            f"got {data['vllm']['docker_commit']!r}"
        )

    @pytest.mark.asyncio
    async def test_health_no_docker_commit_file(self):
        """is_file()=False → vllm.docker_commit is None."""
        app_module = _fresh_app_with_fake_manager()

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=None, docker_version=None,
        )):
            data, response = await _call_health(app_module)

        assert response.status_code == 200
        assert "vllm" in data, (
            f"/health must contain 'vllm' block. Got keys: {list(data.keys())}"
        )
        assert "docker_commit" in data["vllm"], (
            f"/health vllm block must contain 'docker_commit'. "
            f"Got keys: {list(data['vllm'].keys())}"
        )
        assert data["vllm"]["docker_commit"] is None, (
            f"vllm.docker_commit must be None when file does not exist, "
            f"got {data['vllm']['docker_commit']!r}"
        )

    @pytest.mark.asyncio
    async def test_health_truncates_long_commit(self):
        """50-char string from read_text() is truncated to 40 chars in the response."""
        long_commit = "b" * 50
        app_module = _fresh_app_with_fake_manager()

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=long_commit, docker_version=None,
        )):
            data, response = await _call_health(app_module)

        assert response.status_code == 200
        assert data["vllm"]["docker_commit"] == "b" * 40, (
            f"vllm.docker_commit must be truncated to 40 chars. "
            f"Got {data['vllm']['docker_commit']!r} "
            f"(len={len(data['vllm']['docker_commit'] or '')})"
        )


# ===========================================================================
# TestSupportedModelsVllmVersion (3 tests — B-1, B-2, B-3)
# ===========================================================================


def _make_path_side_effect(docker_commit: str | None, docker_version: str | None):
    """Return a side_effect for `patch('app.Path')` that routes two paths.

    Discriminates on the string argument to Path(...):
      - '/workspace/vllm/.docker_commit'   → mock with commit or absent
      - '/workspace/vllm/.docker_version'  → mock with version or absent
    Anything else returns a MagicMock with is_file()=False (harmless default).
    """
    def _factory(*args, **kwargs):
        arg = str(args[0]) if args else ""
        m = MagicMock()
        if arg.endswith(".docker_commit"):
            if docker_commit is None:
                m.is_file.return_value = False
            else:
                m.is_file.return_value = True
                m.read_text.return_value = docker_commit
            return m
        if arg.endswith(".docker_version"):
            if docker_version is None:
                m.is_file.return_value = False
            else:
                m.is_file.return_value = True
                m.read_text.return_value = docker_version
            return m
        # Default: unknown path → behaves as non-existent file
        m.is_file.return_value = False
        return m
    return _factory


@pytest.mark.unit
class TestHealthVllmVersion:
    """B-1/B-2/B-3: /health exposes vllm.version alongside vllm.docker_commit.
    Both are read independently from their respective files (.docker_version
    and .docker_commit) in /workspace/vllm/.

    (Formerly TestSupportedModelsVllmVersion on the removed
    /api/supported-models endpoint.)
    """

    @pytest.mark.asyncio
    async def test_health_returns_vllm_version(self):
        """B-1: .docker_version present → response has vllm.version='v0.20.0'."""
        import httpx
        from app import app

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=None, docker_version="v0.20.0\n",
        )):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "vllm" in data, (
            f"/health must contain 'vllm' block. Got keys: {list(data.keys())}"
        )
        assert "version" in data["vllm"], (
            f"/health vllm block must contain 'version'. Got keys: {list(data['vllm'].keys())}"
        )
        assert data["vllm"]["version"] == "v0.20.0", (
            f"vllm.version must be 'v0.20.0', got {data['vllm']['version']!r}"
        )

    @pytest.mark.asyncio
    async def test_health_no_docker_version_file(self):
        """B-2: .docker_version absent → vllm.version is None but key present."""
        import httpx
        from app import app

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=None, docker_version=None,
        )):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "vllm" in data, (
            f"/health must contain 'vllm' block. Got keys: {list(data.keys())}"
        )
        assert "version" in data["vllm"], (
            f"/health vllm block must contain 'version' key even when file is absent. "
            f"Got keys: {list(data['vllm'].keys())}"
        )
        assert data["vllm"]["version"] is None, (
            f"vllm.version must be None when .docker_version does not exist, "
            f"got {data['vllm']['version']!r}"
        )

    @pytest.mark.asyncio
    async def test_health_returns_both_independently(self):
        """B-3: Both files present → both fields populated independently."""
        import httpx
        from app import app

        commit_hash = "a" * 40

        with patch("app.Path", side_effect=_make_path_side_effect(
            docker_commit=commit_hash, docker_version="v0.20.0\n",
        )):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["vllm"]["version"] == "v0.20.0", (
            f"vllm.version must be 'v0.20.0', got {data['vllm']['version']!r}"
        )
        assert data["vllm"]["docker_commit"] == commit_hash, (
            f"vllm.docker_commit must be {commit_hash!r}, "
            f"got {data['vllm']['docker_commit']!r}"
        )


# ===========================================================================
# TestWorktreeManagerNightlyGuard (1 test)
# ===========================================================================

@pytest.mark.unit
class TestWorktreeManagerCommitPassthrough:
    """Source-code guard: worktree_manager.py must pass 40-char branch hashes
    as VLLM_PRECOMPILED_WHEEL_COMMIT so setup.py fetches wheels for the exact
    pinned commit instead of resolving HEAD (which may not have wheels yet)."""

    def test_worktree_manager_passes_commit_hash_to_wheel_commit(self):
        """worktree_manager.py must derive VLLM_PRECOMPILED_WHEEL_COMMIT from the
        branch parameter: 40-char hex → pass through; the default branch
        ("main") → the baked .docker_commit (so setup.py fetches a wheel that is
        guaranteed to exist, instead of racing nightly publication on the live
        main HEAD); any other branch/tag → empty string (setup.py self-resolves).
        Must NOT hardcode 'nightly' (not a valid value for setup.py).
        """
        source_path = PROJECT_ROOT / "orchestration" / "worktree_manager.py"
        assert source_path.exists(), f"worktree_manager.py not found at {source_path}"
        source = source_path.read_text()

        # Must use dynamic commit derivation, not a hardcoded value
        assert "VLLM_PRECOMPILED_WHEEL_COMMIT" in source, (
            "worktree_manager.py must set VLLM_PRECOMPILED_WHEEL_COMMIT."
        )
        assert '"VLLM_PRECOMPILED_WHEEL_COMMIT": "nightly"' not in source, (
            "VLLM_PRECOMPILED_WHEEL_COMMIT must NOT be 'nightly' — "
            "vLLM setup.py requires a 40-char hash or empty string."
        )
        # Must contain the regex check for 40-char hex
        assert "[0-9a-f]{40}" in source, (
            "worktree_manager.py must check if branch is a 40-char commit hash "
            "to decide VLLM_PRECOMPILED_WHEEL_COMMIT value."
        )


# ===========================================================================
# TestCardSourceLabelLogic (8 tests)
# ===========================================================================

@pytest.mark.unit
class TestCardSourceLabelLogic:
    """Python mirror of JS cardSourceLabel() — tests the Python implementation
    and guards the JS source for the function and its regex pattern."""

    def test_40_char_commit_hash(self):
        """40-char lowercase hex → 'vllm@' + first 7 chars."""
        branch = "a" * 40
        result = _card_source_label(branch)
        assert result == "vllm@" + "a" * 7, (
            f"40-char hex should produce 'vllm@' + first 7 chars, got {result!r}"
        )

    def test_main_branch(self):
        """'main' → 'vllm@main'."""
        assert _card_source_label("main") == "vllm@main"

    def test_custom_branch(self):
        """'feature-x' → 'vllm@feature-x'."""
        assert _card_source_label("feature-x") == "vllm@feature-x"

    def test_empty_branch(self):
        """'' → 'vllm@'."""
        assert _card_source_label("") == "vllm@"

    def test_uppercase_hex_still_matches(self):
        """Uppercase 40-char hex is recognised as a commit hash (case-insensitive match)."""
        branch = "A" * 40
        result = _card_source_label(branch)
        assert result == "vllm@" + "A" * 7, (
            f"Uppercase 40-char hex should be treated as a commit hash, got {result!r}"
        )

    def test_39_char_hex_is_not_commit(self):
        """39-char hex is NOT a commit hash → treated as a branch name."""
        branch = "a" * 39
        result = _card_source_label(branch)
        assert result == "vllm@" + "a" * 39, (
            f"39-char hex must not match the commit pattern; "
            f"expected 'vllm@{'a'*39}', got {result!r}"
        )

    def test_js_source_has_card_source_label(self):
        """Guard: frontend/js/campaign-app.js must define a cardSourceLabel function."""
        js_path = PROJECT_ROOT / "frontend" / "js" / "campaign-app.js"
        assert js_path.exists(), f"campaign-app.js not found at {js_path}"
        js_source = js_path.read_text()
        assert "cardSourceLabel" in js_source, (
            "frontend/js/campaign-app.js does not contain 'cardSourceLabel'. "
            "The function must exist for source labels to render on session cards."
        )

    def test_js_regex_pattern_matches_python(self):
        """Guard: JS source must contain [0-9a-f]{40} (same regex as the Python impl)."""
        js_path = PROJECT_ROOT / "frontend" / "js" / "campaign-app.js"
        assert js_path.exists(), f"campaign-app.js not found at {js_path}"
        js_source = js_path.read_text()
        assert "[0-9a-f]{40}" in js_source, (
            "frontend/js/campaign-app.js must contain the pattern '[0-9a-f]{40}' "
            "to detect 40-char commit hashes. "
            "Python equivalent uses the same regex (re.match(r'^[0-9a-f]{40}$', ...))."
        )


# ===========================================================================
# F-1 / F-2 / F-3: Frontend version wiring (grep-based guards)
# ===========================================================================

@pytest.mark.unit
class TestFrontendVllmVersionWiring:
    """Frontend wiring for the release-version display.

    F-1: campaign-app.js declares cmVllmVersion state.
    F-2: campaign-app.js reads data.vllm.version from /health (the new
         structured vllm block; formerly `data.vllm_version` from the removed
         /api/supported-models endpoint).
    F-3: index.html modal's source-info block renders cmVllmVersion with
         a fallback to the truncated commit hash.
    """

    def _campaign_js(self) -> str:
        p = PROJECT_ROOT / "frontend" / "js" / "campaign-app.js"
        assert p.exists(), f"campaign-app.js not found at {p}"
        return p.read_text()

    def _index_html(self) -> str:
        p = PROJECT_ROOT / "frontend" / "index.html"
        assert p.exists(), f"index.html not found at {p}"
        return p.read_text()

    def test_campaign_app_has_cmvllmversion_state(self):
        """F-1: campaign-app.js must declare `cmVllmVersion:` as Alpine state."""
        src = self._campaign_js()
        assert "cmVllmVersion:" in src, (
            "campaign-app.js must declare cmVllmVersion state (default null)."
        )

    def test_campaign_app_reads_vllm_version_from_api(self):
        """F-2: campaign-app.js must read the vllm version from /health's
        structured `vllm: {version, docker_commit}` block and assign it to
        this.cmVllmVersion."""
        src = self._campaign_js()
        # New contract: read data.vllm.version (or data.vllm?.version) from /health.
        assert ("data.vllm.version" in src) or ("data.vllm?.version" in src), (
            "campaign-app.js must read the vllm version from the /health response "
            "(expected 'data.vllm.version' or 'data.vllm?.version'). The old "
            "/api/supported-models endpoint + flat 'vllm_version' field was removed."
        )
        assert "this.cmVllmVersion =" in src, (
            "campaign-app.js must assign 'this.cmVllmVersion = ...' after reading "
            "the vllm version from /health."
        )

    def test_index_html_modal_shows_version_string(self):
        """F-3: index.html's source-info block must render cmVllmVersion with a
        fallback to the truncated commit hash — so users see 'v0.20.0' on the
        release image and the legacy 12-char hash on older images."""
        src = self._index_html()
        # cmVllmVersion must appear in an x-text= attribute on the source-info block.
        assert "cmVllmVersion" in src, (
            "index.html must reference cmVllmVersion (source-info block's x-text)."
        )
        # Fallback expression — release version wins, truncated commit is the fallback.
        assert "cmVllmVersion || (cmDockerCommit ? cmDockerCommit.slice(0, 12)" in src, (
            "index.html source-info block must use the fallback expression "
            "'cmVllmVersion || (cmDockerCommit ? cmDockerCommit.slice(0, 12) ...)'."
        )


# ===========================================================================
# F-4: Gated cardSourceLabel — version label ONLY when branch matches commit
# ===========================================================================

@pytest.mark.unit
class TestCardSourceLabelGated:
    """F-4: cardSourceLabel(card) with the gated signature.

    Shows `vllm@<version>` ONLY when the card's branch equals cmDockerCommit
    and cmVllmVersion is available. Every other case falls back to the
    original (un-gated) behaviour.

    7 parameterised Python cases + 2 JS source guards.
    """

    COMMIT = "a" * 40
    VERSION = "v0.20.0"

    @pytest.mark.parametrize(
        "branch, commit, version, expected",
        [
            # 1. Default mode, image has release version → show version.
            ("a" * 40, "a" * 40, "v0.20.0", "vllm@v0.20.0"),
            # 2. Default mode, image is legacy (no version) → fall back to short commit.
            ("a" * 40, "a" * 40, None, "vllm@aaaaaaa"),
            # 3. Non-default commit hash (not the image's commit) → short commit.
            ("b" * 40, "a" * 40, "v0.20.0", "vllm@bbbbbbb"),
            # 4. 'main' branch with release version available → still 'vllm@main'.
            ("main", "a" * 40, "v0.20.0", "vllm@main"),
            # 5. Custom feature branch with release version available → 'vllm@<branch>'.
            ("feature-x", "a" * 40, "v0.20.0", "vllm@feature-x"),
            # 6. Empty branch with release version available → 'vllm@'.
            ("", "a" * 40, "v0.20.0", "vllm@"),
            # 7. Commit matches but cmDockerCommit is null (no docker image) → short commit.
            ("a" * 40, None, "v0.20.0", "vllm@aaaaaaa"),
        ],
    )
    def test_gated_cases(self, branch, commit, version, expected):
        assert _card_source_label_gated(branch, commit, version) == expected, (
            f"gated cardSourceLabel({branch!r}, commit={commit!r}, version={version!r}) "
            f"should return {expected!r}"
        )

    @staticmethod
    def _card_source_label_body() -> str:
        """Extract the cardSourceLabel(card) function body from campaign-app.js.

        Locates the definition, then scans forward for the matching closing
        brace (balanced brace counter). Returns everything from the opening
        paren through the closing brace, so window size is implementation-size
        independent.
        """
        js_path = PROJECT_ROOT / "frontend" / "js" / "campaign-app.js"
        assert js_path.exists(), f"campaign-app.js not found at {js_path}"
        src = js_path.read_text()
        idx = src.find("cardSourceLabel(card)")
        assert idx >= 0, "cardSourceLabel(card) definition not found"
        # Find the opening { of the function body.
        brace_open = src.find("{", idx)
        assert brace_open > idx, "no opening brace for cardSourceLabel"
        depth = 0
        for i in range(brace_open, len(src)):
            c = src[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return src[idx: i + 1]
        raise AssertionError("unbalanced braces in cardSourceLabel")

    def test_js_source_references_cmvllmversion_in_cardsourcelabel(self):
        """Guard: campaign-app.js cardSourceLabel must reference cmVllmVersion
        AND compare against cmDockerCommit to implement the gate."""
        body = self._card_source_label_body()
        assert "cmVllmVersion" in body, (
            "cardSourceLabel must reference this.cmVllmVersion to show the "
            "release version label."
        )
        assert "cmDockerCommit" in body, (
            "cardSourceLabel must compare against this.cmDockerCommit to gate "
            "the version label on the image-pinned commit."
        )

    def test_js_source_preserves_40char_fallback(self):
        """Guard: even after adding the gate, cardSourceLabel must still fall
        back to the 40-char hex → short-hash branch for non-matching cases."""
        body = self._card_source_label_body()
        assert "[0-9a-f]{40}" in body, (
            "cardSourceLabel must keep the [0-9a-f]{40} regex as fallback "
            "for non-image-pinned commit hashes."
        )
        assert "slice(0, 7)" in body, (
            "cardSourceLabel must keep the slice(0, 7) short-hash fallback."
        )
