# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Shared utilities for the AMMO session service.
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_error_response(error_message: str) -> Dict[str, Any]:
    """Create a standardized error response body."""
    return {
        "status": "error",
        "error": error_message,
    }

def create_no_cache_headers() -> Dict[str, str]:
    """Create headers to prevent caching - maintains compatibility with existing API"""
    return {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
