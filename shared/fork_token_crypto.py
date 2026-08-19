# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# shared/fork_token_crypto.py
"""Encrypt/decrypt the per-session fork access token at rest.

The token is needed to clone/fetch a private fork (initial create and again on
S3 restore, since the fork base repo is not synced to S3). It is stored
encrypted in SessionState and persisted to S3 metadata. Encryption key comes
from the AMMO_FORK_TOKEN_KEY env var (a urlsafe-base64 Fernet key). When the
key is unset, private-fork tokens are refused at the API boundary (see app.py).
"""

import os
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

ENV_FORK_TOKEN_KEY = "AMMO_FORK_TOKEN_KEY"


class ForkTokenKeyError(RuntimeError):
    """Raised when encryption is attempted without a configured key."""


def fork_token_key_configured() -> bool:
    """True if AMMO_FORK_TOKEN_KEY is set to a usable Fernet key."""
    raw = os.getenv(ENV_FORK_TOKEN_KEY, "")
    if not raw:
        return False
    try:
        Fernet(raw.encode())
        return True
    except (ValueError, TypeError):
        return False


def _fernet() -> Fernet:
    raw = os.getenv(ENV_FORK_TOKEN_KEY, "")
    if not raw:
        raise ForkTokenKeyError(
            "AMMO_FORK_TOKEN_KEY is not configured; cannot handle fork tokens"
        )
    try:
        return Fernet(raw.encode())
    except (ValueError, TypeError) as e:
        raise ForkTokenKeyError(f"AMMO_FORK_TOKEN_KEY is invalid: {e}")


def encrypt_fork_token(token: Optional[str]) -> Optional[str]:
    """Encrypt a token. None/empty → None. Raises ForkTokenKeyError if no key."""
    if not token:
        return None
    return _fernet().encrypt(token.encode()).decode()


def decrypt_fork_token(encrypted: Optional[str]) -> Optional[str]:
    """Decrypt a token. None → None. Wrong key / tampered → None (logged)."""
    if not encrypted:
        return None
    try:
        return _fernet().decrypt(encrypted.encode()).decode()
    except (InvalidToken, ForkTokenKeyError) as e:
        logger.warning(f"Failed to decrypt fork token: {type(e).__name__}")
        return None
