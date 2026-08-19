# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Unit tests for per-client session limits and ownership validation in auth mode.

Tests:
- Session limit enforcement (under/at/over limit, status counting rules)
- Client ID requirement when auth is enabled (AMMO_API_KEY set)
- Client isolation (limits are per-client)
- Ownership validation tightening when auth is enabled
"""

import os
import pytest
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.session_fixtures import (
    reset_all_singletons,
    gpu_manager_4,
    mock_worktree_manager,
    mock_terminal_manager,
    mock_cli_tool_manager,
    mock_inactivity_monitor,
    mock_session_storage,
    mock_session_manager,
    make_session_state,
    make_create_request,
)
from orchestration.session_manager import SessionError, SessionLimitError
from shared.session_models import SessionStatus


# ============================================================================
# Helpers
# ============================================================================

import orchestration.session_manager as _sm


@pytest.fixture
def session_limit_3():
    """Override MAX_SESSIONS_PER_CLIENT to 3 for the duration of the test."""
    original = _sm.MAX_SESSIONS_PER_CLIENT
    _sm.MAX_SESSIONS_PER_CLIENT = 3
    yield 3
    _sm.MAX_SESSIONS_PER_CLIENT = original


def _populate_sessions(manager, owner_id, statuses):
    """Add sessions with the given statuses for an owner."""
    for i, status in enumerate(statuses):
        state = make_session_state(
            session_id=f"sess-{owner_id}-{status.value}-{i}",
            status=status,
            owner_id=owner_id,
        )
        manager._sessions[state.session_id] = state


# ============================================================================
# Session Limit Enforcement
# ============================================================================

@pytest.mark.unit
class TestSessionLimitEnforcement:
    """Tests that per-client session limits are enforced correctly."""

    @pytest.mark.asyncio
    async def test_under_limit_succeeds(self, mock_session_manager, session_limit_3):
        """Under limit (2 active, limit=3) -> session created successfully."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 2)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        response = await mock_session_manager.create_session(request, owner_id="client-A")
        assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_at_limit_returns_error(self, mock_session_manager, session_limit_3):
        """At limit (3 active, limit=3) -> raises SessionLimitError."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 3)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        with pytest.raises(SessionLimitError, match="Maximum 3 active sessions"):
            await mock_session_manager.create_session(request, owner_id="client-A")

    @pytest.mark.asyncio
    async def test_over_limit_returns_error(self, mock_session_manager, session_limit_3):
        """Over limit (4 active, limit=3) -> raises SessionLimitError."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 4)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        with pytest.raises(SessionLimitError, match="Maximum 3 active sessions"):
            await mock_session_manager.create_session(request, owner_id="client-A")

    @pytest.mark.asyncio
    async def test_default_limit_is_3(self, mock_session_manager):
        """Default limit is 3 when MAX_SESSIONS_PER_CLIENT not set."""
        original = _sm.MAX_SESSIONS_PER_CLIENT
        os.environ.pop("MAX_SESSIONS_PER_CLIENT", None)
        _sm.MAX_SESSIONS_PER_CLIENT = int(os.getenv("MAX_SESSIONS_PER_CLIENT", "3"))
        try:
            assert _sm.MAX_SESSIONS_PER_CLIENT == 3

            _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 3)
            request = make_create_request(repo_name="vllm", gpu_count=0)

            with pytest.raises(SessionLimitError):
                await mock_session_manager.create_session(request, owner_id="client-A")
        finally:
            _sm.MAX_SESSIONS_PER_CLIENT = original

    @pytest.mark.asyncio
    async def test_terminated_sessions_not_counted(self, mock_session_manager, session_limit_3):
        """Terminated sessions NOT counted (3 terminated + 2 active, limit=3 -> succeeds)."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.TERMINATED] * 3)
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 2)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        response = await mock_session_manager.create_session(request, owner_id="client-A")
        assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_paused_sessions_do_not_count_toward_limit(self, mock_session_manager, session_limit_3):
        """Paused sessions don't count (2 active + 1 paused = 2 active, limit=3 -> succeeds)."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 2 + [SessionStatus.PAUSED])
        request = make_create_request(repo_name="vllm", gpu_count=0)

        response = await mock_session_manager.create_session(request, owner_id="client-A")
        assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_creating_sessions_count_toward_limit(self, mock_session_manager, session_limit_3):
        """Creating-status sessions count (2 active + 1 creating = 3 -> 429)."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 2 + [SessionStatus.CREATING])
        request = make_create_request(repo_name="vllm", gpu_count=0)

        with pytest.raises(SessionLimitError, match="Maximum 3 active sessions"):
            await mock_session_manager.create_session(request, owner_id="client-A")


# ============================================================================
# Client ID Requirement
# ============================================================================

@pytest.mark.unit
class TestSessionLimitClientIdRequirement:
    """Tests that X-Client-ID is required when auth is enabled."""

    @pytest.mark.asyncio
    async def test_no_client_id_with_auth_enabled_returns_error(self, mock_session_manager):
        """No client_id + AMMO_API_KEY set -> raises SessionError with 'X-Client-ID header required'."""
        request = make_create_request(repo_name="vllm", gpu_count=0)

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key-123"}, clear=False):
            with pytest.raises(SessionError, match="X-Client-ID header required"):
                await mock_session_manager.create_session(request, owner_id=None)

    @pytest.mark.asyncio
    async def test_no_client_id_without_auth_allowed(self, mock_session_manager):
        """No client_id + AMMO_API_KEY unset -> allowed (backward compat)."""
        request = make_create_request(repo_name="vllm", gpu_count=0)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMMO_API_KEY", None)
            response = await mock_session_manager.create_session(request, owner_id=None)
            assert response.session_id is not None


# ============================================================================
# Client Isolation
# ============================================================================

@pytest.mark.unit
class TestSessionLimitClientIsolation:
    """Tests that session limits are per-client, not global."""

    @pytest.mark.asyncio
    async def test_client_a_at_limit_client_b_can_create(self, mock_session_manager, session_limit_3):
        """Client A at limit, Client B at 0 -> B can create."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 3)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        response = await mock_session_manager.create_session(request, owner_id="client-B")
        assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_limit_counts_only_matching_owner(self, mock_session_manager, session_limit_3):
        """Limit counts only sessions matching owner_id."""
        _populate_sessions(mock_session_manager, "client-A", [SessionStatus.ACTIVE] * 2)
        _populate_sessions(mock_session_manager, "client-B", [SessionStatus.ACTIVE] * 2)
        request = make_create_request(repo_name="vllm", gpu_count=0)

        # Client A has 2/3, should succeed
        response = await mock_session_manager.create_session(request, owner_id="client-A")
        assert response.session_id is not None


# ============================================================================
# Ownership Validation in Auth Mode
# ============================================================================

@pytest.mark.unit
class TestOwnershipValidationAuthMode:
    """Tests for tightened ownership validation when AMMO_API_KEY is set."""

    @pytest.mark.asyncio
    async def test_null_client_id_auth_enabled_list_sessions_only_legacy(self, mock_session_manager):
        """Null client_id + auth enabled: list_sessions -> only sees null-owner (legacy) sessions."""
        legacy = make_session_state(session_id="sess-legacy", status=SessionStatus.ACTIVE)
        legacy.owner_id = None
        mock_session_manager._sessions["sess-legacy"] = legacy

        owned = make_session_state(session_id="sess-owned", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-owned"] = owned

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key"}, clear=False):
            response = await mock_session_manager.list_sessions(owner_id=None)
            session_ids = [s.session_id for s in response.sessions]
            assert "sess-legacy" in session_ids
            assert "sess-owned" not in session_ids

    @pytest.mark.asyncio
    async def test_null_client_id_auth_disabled_list_sessions_sees_all(self, mock_session_manager):
        """Null client_id + auth disabled: list_sessions -> sees all sessions (backward compat)."""
        legacy = make_session_state(session_id="sess-legacy", status=SessionStatus.ACTIVE)
        legacy.owner_id = None
        mock_session_manager._sessions["sess-legacy"] = legacy

        owned = make_session_state(session_id="sess-owned", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-owned"] = owned

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMMO_API_KEY", None)
            response = await mock_session_manager.list_sessions(owner_id=None)
            session_ids = [s.session_id for s in response.sessions]
            assert "sess-legacy" in session_ids
            assert "sess-owned" in session_ids

    def test_null_client_id_auth_enabled_get_session_on_owned_raises(self, mock_session_manager):
        """Null client_id + auth enabled: _validate_ownership on owned session -> raises SessionError."""
        owned = make_session_state(session_id="sess-owned", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-owned"] = owned

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key"}, clear=False):
            with pytest.raises(SessionError, match="not found"):
                mock_session_manager._validate_ownership("sess-owned", owner_id=None)

    @pytest.mark.asyncio
    async def test_null_client_id_auth_enabled_pause_session_on_owned_raises(self, mock_session_manager):
        """Null client_id + auth enabled: pause_session on owned session -> raises SessionError."""
        owned = make_session_state(session_id="sess-owned", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-owned"] = owned

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key"}, clear=False):
            with pytest.raises(SessionError, match="not found"):
                await mock_session_manager.pause_session("sess-owned", owner_id=None)

    @pytest.mark.asyncio
    async def test_null_client_id_auth_enabled_terminate_session_on_owned_raises(self, mock_session_manager):
        """Null client_id + auth enabled: terminate_session on owned session -> raises SessionError."""
        owned = make_session_state(session_id="sess-owned", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-owned"] = owned

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key"}, clear=False):
            with pytest.raises(SessionError, match="not found"):
                await mock_session_manager.terminate_session("sess-owned", owner_id=None)

    @pytest.mark.asyncio
    async def test_client_a_auth_enabled_sees_own_and_legacy_not_client_b(self, mock_session_manager):
        """Client A + auth enabled: sees own sessions + legacy sessions, not Client B sessions."""
        legacy = make_session_state(session_id="sess-legacy", status=SessionStatus.ACTIVE)
        legacy.owner_id = None
        mock_session_manager._sessions["sess-legacy"] = legacy

        own = make_session_state(session_id="sess-a", status=SessionStatus.ACTIVE, owner_id="client-A")
        mock_session_manager._sessions["sess-a"] = own

        other = make_session_state(session_id="sess-b", status=SessionStatus.ACTIVE, owner_id="client-B")
        mock_session_manager._sessions["sess-b"] = other

        with patch.dict(os.environ, {"AMMO_API_KEY": "test-key"}, clear=False):
            response = await mock_session_manager.list_sessions(owner_id="client-A")
            session_ids = [s.session_id for s in response.sessions]
            assert "sess-legacy" in session_ids
            assert "sess-a" in session_ids
            assert "sess-b" not in session_ids
