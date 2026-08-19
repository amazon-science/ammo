# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Verifies that the static model preset surface is gone:
  - /api/supported-models  -> 404 (endpoint deleted)
  - /api/moe-models        -> 404 (endpoint deleted)
  - shared.constants.SUPPORTED_MODELS, MOE_MODELS, MOE_TP_OPTIONS  -> ImportError
  - shared.constants.TP_OPTIONS, GPU_DTYPE_MAP                     -> still present
"""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _fresh_client():
    """Reload app module and build a TestClient.

    Tests only exercise route registration, not lifespan startup, so we use
    the FastAPI TestClient directly against the app instance.
    """
    import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), app_module


# ---------------------------------------------------------------------------
# 404 on deleted endpoints
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOldEndpointsDeleted:

    def test_supported_models_endpoint_returns_404(self):
        """GET /api/supported-models must return 404 (route deleted)."""
        client, _ = _fresh_client()
        resp = client.get("/api/supported-models")
        assert resp.status_code == 404, (
            f"/api/supported-models must return 404 after Task #1 deletion; got {resp.status_code}"
        )

    def test_moe_models_endpoint_returns_404(self):
        """GET /api/moe-models must return 404 (route deleted)."""
        client, _ = _fresh_client()
        resp = client.get("/api/moe-models")
        assert resp.status_code == 404, (
            f"/api/moe-models must return 404 after Task #1 deletion; got {resp.status_code}"
        )

    def test_app_has_no_get_supported_models_attr(self):
        """The Python function get_supported_models must no longer exist on app.py."""
        import app as app_module
        importlib.reload(app_module)
        assert not hasattr(app_module, "get_supported_models"), (
            "app.get_supported_models function must be deleted (Task #1)"
        )
        assert not hasattr(app_module, "get_moe_models"), (
            "app.get_moe_models function must be deleted (Task #1)"
        )

    def test_cluster_and_all_routes_return_404(self):
        """Cluster aggregation routes are not part of the local-only server."""
        client, _ = _fresh_client()
        removed_paths = [
            "/" + "cluster/status",
            "/" + "sessions/all",
            "/" + "api/campaigns/all",
        ]
        for path in removed_paths:
            resp = client.get(path)
            assert resp.status_code == 404, f"{path} must be removed"

    def test_cluster_runtime_symbols_removed(self):
        """app.py must not expose cluster aggregation/proxy internals."""
        import app as app_module
        importlib.reload(app_module)
        for name in (
            "cluster_status_service",
            "get_cluster_status",
            "list_all_sessions",
            "get_all_campaigns_overview",
            "_proxy_delete_to_cluster",
            "_proxy_resume_to_cluster",
            "_proxy_campaign_to_cluster",
            "_find_session_owners",
            "is_proxied_request",
        ):
            assert not hasattr(app_module, name), f"app.{name} must be removed"

    def test_inter_pod_helpers_removed_from_shared_utils(self):
        """No inter-pod auth or hop-guard helpers remain in local-only mode."""
        import shared.utils as utils
        importlib.reload(utils)
        assert not hasattr(utils, "get_inter_pod_auth_headers")
        assert not hasattr(utils, "INTER_POD_PROXY_HEADER")


# ---------------------------------------------------------------------------
# shared.constants — removed vs kept symbols
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstantsCleanup:

    def test_supported_models_deleted(self):
        """SUPPORTED_MODELS must be removed from shared.constants."""
        import shared.constants as constants
        importlib.reload(constants)
        assert not hasattr(constants, "SUPPORTED_MODELS"), (
            "SUPPORTED_MODELS must be deleted from shared/constants.py (Task #1)"
        )

    def test_moe_aliases_deleted(self):
        """MOE_MODELS and MOE_TP_OPTIONS aliases must be removed."""
        import shared.constants as constants
        importlib.reload(constants)
        assert not hasattr(constants, "MOE_MODELS"), (
            "MOE_MODELS alias must be deleted (Task #1)"
        )
        assert not hasattr(constants, "MOE_TP_OPTIONS"), (
            "MOE_TP_OPTIONS alias must be deleted (Task #1)"
        )

    def test_tp_options_preserved(self):
        """TP_OPTIONS must remain (reusable constant, kept per plan)."""
        import shared.constants as constants
        importlib.reload(constants)
        assert hasattr(constants, "TP_OPTIONS")
        assert constants.TP_OPTIONS == [1, 2, 4, 8]

    def test_gpu_dtype_map_preserved(self):
        """GPU_DTYPE_MAP must remain (used by /health and /api/hf-model-config)."""
        import shared.constants as constants
        importlib.reload(constants)
        assert hasattr(constants, "GPU_DTYPE_MAP")
        for gpu in ["b300", "b200", "h100", "h200", "l40s", "a100", "unknown"]:
            assert gpu in constants.GPU_DTYPE_MAP
