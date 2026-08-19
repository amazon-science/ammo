# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_url_validator.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.fork_url_validator import validate_fork_url, ForkUrlError


@pytest.mark.unit
class TestForkUrlValidator:
    def test_accepts_plain_https_github(self):
        assert validate_fork_url("https://github.com/octocat/vllm") == \
            "https://github.com/octocat/vllm.git"

    def test_accepts_with_dot_git_suffix(self):
        assert validate_fork_url("https://github.com/my-user/vllm.git") == \
            "https://github.com/my-user/vllm.git"

    def test_normalizes_trailing_slash(self):
        assert validate_fork_url("https://github.com/u/r/") == \
            "https://github.com/u/r.git"

    @pytest.mark.parametrize("bad", [
        "http://github.com/u/r",                # not https
        "git://github.com/u/r",                 # scheme
        "ssh://git@github.com/u/r",             # scheme
        "file:///etc/passwd",                   # local file
        "https://github.com.evil.com/u/r",      # host suffix attack
        "https://raw.githubusercontent.com/u/r",# subdomain
        "https://gitlab.com/u/r",               # wrong host
        "https://user:pass@github.com/u/r",     # userinfo
        "https://github.com:22/u/r",            # explicit port
        "https://192.168.0.1/u/r",              # IP literal
        "https://github.com/u",                 # missing repo
        "https://github.com/u/r/extra",         # extra path segment
        "https://github.com//r",                # empty owner
        "https://github.com/u/r/../../x",       # traversal
        "not a url",
        "",
    ])
    def test_rejects(self, bad):
        with pytest.raises(ForkUrlError):
            validate_fork_url(bad)
