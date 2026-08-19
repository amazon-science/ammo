# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_create_endpoint.py
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    # Stub the session manager so create returns a canned response.
    sm = MagicMock()
    sm.create_session = AsyncMock(return_value=MagicMock(
        model_dump=MagicMock(return_value={"session_id": "x", "status": "building"})
    ))
    monkeypatch.setattr(app_module, "session_manager", sm)
    gpu_mgr = MagicMock()
    gpu_mgr.get_available_gpu_count.return_value = 8
    gpu_mgr.get_gpu_count.return_value = 8
    monkeypatch.setattr(app_module, "gpu_manager", gpu_mgr)
    monkeypatch.delenv("AMMO_API_KEY", raising=False)
    return TestClient(app_module.app)


def _error_detail(response):
    """Extract the human-readable error string regardless of body shape.

    FastAPI's default HTTPException body is {"detail": ...}, but this server
    registers a custom HTTPException handler that emits {"error": ...,
    "status": "error"}. Accept either so the test pins behavior, not shape.
    """
    body = response.json()
    return body.get("detail") or body.get("error") or ""


@pytest.mark.unit
def test_invalid_fork_url_rejected_400(client):
    r = client.post("/sessions", json={
        "gpu_count": 1, "vllm_fork_url": "https://gitlab.com/u/r",
    })
    assert r.status_code == 400
    assert "github.com" in _error_detail(r).lower()


@pytest.mark.unit
def test_token_without_key_rejected_400(client, monkeypatch):
    monkeypatch.delenv("AMMO_FORK_TOKEN_KEY", raising=False)
    r = client.post("/sessions", json={
        "gpu_count": 1,
        "vllm_fork_url": "https://github.com/u/vllm",
        "vllm_fork_token": "ghp_x",
    })
    assert r.status_code == 400
    assert "AMMO_FORK_TOKEN_KEY" in _error_detail(r)


@pytest.mark.unit
def test_valid_public_fork_url_normalized_and_passed(client):
    r = client.post("/sessions", json={
        "gpu_count": 1, "vllm_fork_url": "https://github.com/u/vllm",
    })
    assert r.status_code == 200
    # session_manager.create_session called with normalized .git URL
    import app as app_module
    req = app_module.session_manager.create_session.call_args.args[0]
    assert req.vllm_fork_url == "https://github.com/u/vllm.git"
