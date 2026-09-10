# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Required E2E runs fail on missing prerequisites instead of hiding skips."""

import os
import shutil
import subprocess

import pytest
import requests


@pytest.fixture(scope="session", autouse=True)
def required_e2e_environment():
    if os.getenv("AMMO_E2E_REQUIRED") != "1":
        return
    assert shutil.which("docker"), "AMMO_E2E_REQUIRED needs Docker on the test host"
    container = os.getenv("AMMO_TEST_CONTAINER", "ammo-server")
    for executable in ("codex", "claude"):
        result = subprocess.run(
            ["docker", "exec", container, executable, "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Required {executable} CLI unavailable in {container}"
    url = os.getenv("AMMO_SERVER_URL", "http://localhost:8000")
    response = requests.get(f"{url}/health", timeout=10)
    assert response.status_code == 200, f"Required server unhealthy at {url}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.getenv("AMMO_E2E_REQUIRED") == "1" and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Required E2E test was skipped: {report.longrepr}"
