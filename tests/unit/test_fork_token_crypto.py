# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# tests/unit/test_fork_token_crypto.py
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.fork_token_crypto import (
    encrypt_fork_token,
    decrypt_fork_token,
    fork_token_key_configured,
    ForkTokenKeyError,
)
from cryptography.fernet import Fernet


@pytest.mark.unit
class TestForkTokenCrypto:
    def test_round_trip(self, monkeypatch):
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", key)
        enc = encrypt_fork_token("ghp_secrettoken123")
        assert enc != "ghp_secrettoken123"  # actually encrypted
        assert decrypt_fork_token(enc) == "ghp_secrettoken123"

    def test_none_and_empty_passthrough(self, monkeypatch):
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", key)
        assert encrypt_fork_token(None) is None
        assert encrypt_fork_token("") is None
        assert decrypt_fork_token(None) is None

    def test_key_configured_flag(self, monkeypatch):
        monkeypatch.delenv("AMMO_FORK_TOKEN_KEY", raising=False)
        assert fork_token_key_configured() is False
        monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", Fernet.generate_key().decode())
        assert fork_token_key_configured() is True

    def test_encrypt_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("AMMO_FORK_TOKEN_KEY", raising=False)
        with pytest.raises(ForkTokenKeyError):
            encrypt_fork_token("ghp_x")

    def test_decrypt_with_wrong_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", Fernet.generate_key().decode())
        enc = encrypt_fork_token("ghp_x")
        monkeypatch.setenv("AMMO_FORK_TOKEN_KEY", Fernet.generate_key().decode())
        assert decrypt_fork_token(enc) is None  # tamper/wrong-key → None, no crash
