# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pytest configuration for causal tests.

Adds the causal package directory to sys.path so tests can import
extract_events and other causal modules directly by name.
"""
import sys
from pathlib import Path

# Add the causal directory to path so `from extract_events import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent))
