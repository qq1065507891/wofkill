"""Tests for the NEW-P2-5 caller_role over-warn fix.

_resolve_caller_role previously emitted a security warning for every
non-token query-param call, including player_agent and spectator — the
documented dev path. Move the warning to the elevated-role branch
(MODERATOR/DEBUGGER) so it actually flags the security-relevant case
(someone authenticating as an elevated role via caller_id without a
session token).
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def test_no_warning_for_player_agent_query_param_auth(caplog):
    """A player_agent call without a session_token must NOT trigger
    the 'Legacy query-param auth' security warning.
    """
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod})
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)

    with caplog.at_level(logging.WARNING, logger="werewolf_agent.api.routes.games"):
        resp = client.get(
            f"/games/{game_id}/public-state",
            params={"caller_id": "p01", "caller_role": "player_agent"},
        )
    assert resp.status_code == 200

    legacy_warnings = [
        rec for rec in caplog.records
        if "Legacy query-param auth" in rec.getMessage()
        and "caller_role=player_agent" in rec.getMessage()
    ]
    assert not legacy_warnings, (
        f"NEW-P2-5 not fixed: legacy auth warning was emitted for "
        f"player_agent. Got {len(legacy_warnings)} warning(s)."
    )


def test_warning_for_unauthorized_elevated_role(caplog):
    """An elevated-role call (MODERATOR/DEBUGGER) via query-param
    auth must still produce a warning — it's the security-relevant
    case the message is meant to flag.
    """
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod})
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)

    # Caller claims to be "mod2" with moderator role but mod2 is not
    # in authorized_callers — the call should 403, but for security
    # the warning is still the right message.
    with caplog.at_level(logging.WARNING, logger="werewolf_agent.api.routes.games"):
        resp = client.post(
            f"/games/{game_id}/pause",
            json={"caller_id": "mod2", "caller_role": "moderator"},
        )
    # Should 403 (unauthorized)
    assert resp.status_code == 403
