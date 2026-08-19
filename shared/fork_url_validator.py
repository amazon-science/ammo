# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# shared/fork_url_validator.py
"""Validate a user-supplied vLLM fork git URL.

Multi-tenant SSRF / abuse gate: only anonymous github.com HTTPS repo URLs of
the exact shape https://github.com/<owner>/<repo>[.git] are accepted. Every
other scheme, host, subdomain, userinfo, port, or path shape is rejected.
"""

import re
from urllib.parse import urlsplit

# owner / repo: GitHub allows alnum, hyphen, underscore, dot (no slash).
_SEGMENT = r"[A-Za-z0-9._-]+"
_PATH_RE = re.compile(rf"^/({_SEGMENT})/({_SEGMENT})$")


class ForkUrlError(ValueError):
    """Raised when a fork URL fails the allowlist."""


def validate_fork_url(url: str) -> str:
    """Return the normalized `https://github.com/<owner>/<repo>.git` form.

    Raises ForkUrlError for anything outside the allowlist.
    """
    if not url or not isinstance(url, str):
        raise ForkUrlError("Fork URL is required")
    url = url.strip()

    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ForkUrlError("Fork URL must use https://")
    # urlsplit puts userinfo + host (+ port) in netloc; hostname/port isolate them.
    if parts.username is not None or parts.password is not None:
        raise ForkUrlError("Fork URL must not contain credentials")
    if parts.port is not None:
        raise ForkUrlError("Fork URL must not specify a port")
    if (parts.hostname or "").lower() != "github.com":
        raise ForkUrlError("Only github.com fork URLs are allowed")

    path = parts.path
    if path.endswith(".git"):
        path = path[: -len(".git")]
    path = path.rstrip("/")

    m = _PATH_RE.match(path)
    if not m:
        raise ForkUrlError(
            "Fork URL must be https://github.com/<owner>/<repo>"
        )
    owner, repo = m.group(1), m.group(2)
    if owner in (".", "..") or repo in (".", ".."):
        raise ForkUrlError("Invalid owner/repo in fork URL")

    return f"https://github.com/{owner}/{repo}.git"
