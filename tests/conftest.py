# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Pytest configuration and shared fixtures for AMMO Sessions Server tests
"""

import os
import sys
import uuid
import pytest
import logging
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Pytest markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests that don't require server")
    config.addinivalue_line("markers", "integration: Integration tests requiring server")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests that take > 10 seconds")
    config.addinivalue_line("markers", "empirical: Tests that measure real runtime behaviour")

# Common fixtures
@pytest.fixture
def server_url():
    """Get server URL from environment or use default"""
    return os.getenv("AMMO_SERVER_URL", "http://localhost:8000")


# ============================================================================
# AUTH FIXTURES
# ============================================================================
# Global fixtures for API key authentication. Tests work in both modes:
# - AMMO_API_KEY set: auth headers are included
# - AMMO_API_KEY unset: headers are empty (backward compatible)

@pytest.fixture
def ammo_api_key():
    """API key for auth-enabled tests. None if auth disabled."""
    return os.getenv("AMMO_API_KEY")


@pytest.fixture
def auth_headers(ammo_api_key):
    """HTTP headers with API key when AMMO_API_KEY is set."""
    if ammo_api_key:
        return {"Authorization": f"Bearer {ammo_api_key}"}
    return {}


@pytest.fixture
def client_id():
    """Unique client ID for session isolation."""
    return str(uuid.uuid4())


@pytest.fixture
def client_headers_with_auth(client_id, auth_headers):
    """Combined X-Client-ID + auth headers."""
    return {"X-Client-ID": client_id, **auth_headers}
