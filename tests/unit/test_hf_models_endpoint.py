# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for the /api/hf-models endpoint.

Tests that the endpoint proxies HuggingFace API search results,
caches responses, and handles errors gracefully.
"""

import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_response():
    """Return a sample HuggingFace API response list."""
    return [
        {
            "id": "meta-llama/Meta-Llama-3.1-8B",
            "downloads": 5_000_000,
            "likes": 1200,
            "pipeline_tag": "text-generation",
            "tags": ["text-generation", "transformers", "llama"],
        },
        {
            "id": "mistralai/Mistral-7B-v0.1",
            "downloads": 3_000_000,
            "likes": 800,
            "pipeline_tag": "text-generation",
            "tags": ["text-generation", "transformers", "mistral"],
        },
    ]


def _clear_hf_cache():
    """Clear the module-level HF cache between tests."""
    import app as app_module
    app_module._hf_cache.clear()
    app_module._hf_cache_ts.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHfModelsEndpoint:
    """Tests for /api/hf-models endpoint."""

    def test_empty_query_returns_empty_list(self):
        """Empty query string should return empty models list without calling HF."""
        import asyncio
        import importlib
        import app as app_module
        importlib.reload(app_module)
        _clear_hf_cache()

        response = asyncio.get_event_loop().run_until_complete(
            app_module.search_hf_models(q="", limit=20)
        )

        assert response["models"] == []
        assert response["source"] == "huggingface"

    def test_whitespace_query_returns_empty_list(self):
        """Whitespace-only query should return empty models list."""
        import asyncio
        import importlib
        import app as app_module
        importlib.reload(app_module)
        _clear_hf_cache()

        response = asyncio.get_event_loop().run_until_complete(
            app_module.search_hf_models(q="   ", limit=20)
        )

        assert response["models"] == []
        assert response["source"] == "huggingface"

    def test_successful_hf_search(self):
        """Successful HF API call should return formatted models."""
        import asyncio
        import importlib

        mock_response = MagicMock()
        mock_response.json.return_value = _make_hf_response()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            response = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="llama", limit=20)
            )

        assert response["source"] == "huggingface"
        assert len(response["models"]) == 2
        assert response["models"][0]["id"] == "meta-llama/Meta-Llama-3.1-8B"
        assert response["models"][0]["downloads"] == 5_000_000
        assert response["models"][0]["likes"] == 1200
        assert response["models"][0]["pipeline_tag"] == "text-generation"
        assert response["models"][0]["tags"] == ["text-generation", "transformers", "llama"]

    def test_hf_api_failure_returns_error_source(self):
        """When HF API call fails, return empty list with source='error'."""
        import asyncio
        import importlib

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            response = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="llama", limit=20)
            )

        assert response["source"] == "error"
        assert response["models"] == []

    def test_cache_returns_same_result(self):
        """Second call with same query should use cache, not call HF API again."""
        import asyncio
        import importlib

        call_count = 0

        mock_response = MagicMock()
        mock_response.json.return_value = _make_hf_response()
        mock_response.raise_for_status = MagicMock()

        async def counting_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = counting_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            # First call - should hit HF API
            r1 = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="llama", limit=20)
            )
            # Second call - should use cache
            r2 = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="llama", limit=20)
            )

        assert call_count == 1, "HF API should only be called once due to caching"
        assert r1 == r2

    def test_cache_expires_after_ttl(self):
        """Cache should expire after TTL and call HF API again."""
        import asyncio
        import importlib

        call_count = 0

        mock_response = MagicMock()
        mock_response.json.return_value = _make_hf_response()
        mock_response.raise_for_status = MagicMock()

        async def counting_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = counting_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            # First call
            asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="mistral", limit=20)
            )
            assert call_count == 1

            # Expire the cache by backdating the timestamp
            app_module._hf_cache_ts["mistral:20"] = time.time() - 120

            # Second call after cache expiry
            asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="mistral", limit=20)
            )

        assert call_count == 2, "HF API should be called again after cache expiry"

    def test_different_queries_cached_separately(self):
        """Different query strings should have separate cache entries."""
        import asyncio
        import importlib

        call_count = 0

        mock_response = MagicMock()
        mock_response.json.return_value = _make_hf_response()
        mock_response.raise_for_status = MagicMock()

        async def counting_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = counting_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="llama", limit=20)
            )
            asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="mistral", limit=20)
            )

        assert call_count == 2, "Different queries should not share cache"

    def test_response_filters_models_without_id(self):
        """Models without an 'id' field should be filtered out."""
        import asyncio
        import importlib

        bad_data = [
            {"id": "valid/model", "downloads": 100, "likes": 10, "pipeline_tag": "text-generation"},
            {"downloads": 50, "likes": 5, "pipeline_tag": "text-generation"},  # no id
            {"id": "", "downloads": 50, "likes": 5, "pipeline_tag": "text-generation"},  # empty id
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = bad_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            response = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="test", limit=20)
            )

        assert len(response["models"]) == 1
        assert response["models"][0]["id"] == "valid/model"

    def test_tags_passthrough_from_hf_response(self):
        """HF API `tags` field must flow through unchanged (needed by FE MoE detection)."""
        import asyncio
        import importlib

        raw = [
            {
                "id": "mistralai/Mixtral-8x7B-Instruct",
                "downloads": 100,
                "likes": 10,
                "pipeline_tag": "text-generation",
                "tags": ["text-generation", "mixtral", "transformers"],
            }
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            response = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="mixtral", limit=20)
            )

        assert response["models"][0]["tags"] == [
            "text-generation",
            "mixtral",
            "transformers",
        ]

    def test_tags_defaults_to_empty_list_when_missing(self):
        """HF response without `tags` key → response exposes empty list (not missing)."""
        import asyncio
        import importlib

        raw = [
            {
                "id": "no-tags/model",
                "downloads": 1,
                "likes": 0,
                "pipeline_tag": "text-generation",
                # No "tags" field
            }
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = raw
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            import app as app_module
            importlib.reload(app_module)
            _clear_hf_cache()

            response = asyncio.get_event_loop().run_until_complete(
                app_module.search_hf_models(q="no-tags", limit=20)
            )

        assert "tags" in response["models"][0]
        assert response["models"][0]["tags"] == []


