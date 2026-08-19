# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.session_models import CreateSessionRequest

@pytest.mark.unit
@pytest.mark.parametrize("bad", [
    "--upload-pack=touch /tmp/x", "-c", "--output=/etc/passwd",
    "feat;rm -rf", "a b", "foo..bar", "tip/", "x.lock", "a//b", "$(id)",
])
def test_branch_rejects_injection(bad):
    with pytest.raises(Exception):
        CreateSessionRequest(branch=bad, vllm_fork_url="https://github.com/u/vllm.git")

@pytest.mark.unit
@pytest.mark.parametrize("ok", ["main", "release/v0.2.0", "feature_x", "v1.2.3", "a-b-c"])
def test_branch_accepts_valid(ok):
    r = CreateSessionRequest(branch=ok)
    assert r.branch == ok

@pytest.mark.unit
def test_empty_branch_defaults_main():
    assert CreateSessionRequest(branch="").branch == "main"
