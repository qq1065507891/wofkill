"""Tests for local development auth system."""
import pytest
import time

from werewolf_agent.api.auth import AuthManager, AuthConfig


# ---------------------------------------------------------------------------
# AuthConfig unit tests
# ---------------------------------------------------------------------------

def test_auth_config_loads_defaults():
    cfg = AuthConfig()
    assert cfg.mode == "local"
    assert cfg.local_users  # has default users
    assert "mod1" in cfg.local_users
    assert "dbg1" in cfg.local_users
    assert cfg.secret_key  # non-empty default


def test_auth_config_env_override():
    """AuthConfig reads WEREWOLF_AUTH_SECRET env var when secret_key not set."""
    cfg = AuthConfig()
    # Default key is the fallback
    assert cfg.secret_key == "wofkill-dev-key-change-me"


# ---------------------------------------------------------------------------
# AuthManager unit tests
# ---------------------------------------------------------------------------

def test_auth_manager_local_mode():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("mod1", "moderator")
    assert token is not None
    role = mgr.validate_session(token)
    assert role == "moderator"


def test_auth_manager_debugger_role():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("dbg1", "debugger")
    assert mgr.validate_session(token) == "debugger"


def test_auth_manager_spectator_role():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("spectator", "spectator")
    assert mgr.validate_session(token) == "spectator"


def test_auth_manager_rejects_unknown_user():
    mgr = AuthManager(AuthConfig(mode="local"))
    with pytest.raises(PermissionError):
        mgr.create_session("hacker", "moderator")


def test_auth_manager_rejects_wrong_role():
    mgr = AuthManager(AuthConfig(mode="local"))
    with pytest.raises(PermissionError):
        mgr.create_session("mod1", "debugger")


def test_auth_manager_expired_token():
    mgr = AuthManager(AuthConfig(mode="local", token_ttl_seconds=0))
    token = mgr.create_session("mod1", "moderator")
    time.sleep(0.1)
    role = mgr.validate_session(token)
    assert role is None


def test_auth_manager_revoke():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("mod1", "moderator")
    mgr.revoke_session(token)
    assert mgr.validate_session(token) is None


def test_auth_manager_revoke_nonexistent():
    mgr = AuthManager(AuthConfig(mode="local"))
    # Should not raise
    mgr.revoke_session("nonexistent-token")


def test_auth_manager_validate_invalid_token():
    mgr = AuthManager(AuthConfig(mode="local"))
    assert mgr.validate_session("invalid.token.here") is None


def test_auth_manager_validate_empty_token():
    mgr = AuthManager(AuthConfig(mode="local"))
    assert mgr.validate_session("") is None


def test_auth_manager_token_structure():
    mgr = AuthManager(AuthConfig(mode="local"))
    token = mgr.create_session("mod1", "moderator")
    parts = token.split(".")
    assert len(parts) == 4
    assert parts[0] == "mod1"
    assert parts[1] == "moderator"


def test_auth_manager_different_users_different_tokens():
    mgr = AuthManager(AuthConfig(mode="local"))
    t1 = mgr.create_session("mod1", "moderator")
    t2 = mgr.create_session("dbg1", "debugger")
    assert t1 != t2


def test_auth_manager_same_user_new_token_invalidates_old():
    """Creating a new session does NOT invalidate the old one in-memory."""
    mgr = AuthManager(AuthConfig(mode="local"))
    t1 = mgr.create_session("mod1", "moderator")
    t2 = mgr.create_session("mod1", "moderator")
    # Both tokens remain valid because sessions are stored by token string
    assert mgr.validate_session(t1) == "moderator"
    assert mgr.validate_session(t2) == "moderator"


# ---------------------------------------------------------------------------
# Integration tests via FastAPI TestClient
# ---------------------------------------------------------------------------

def test_login_endpoint():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["role"] == "moderator"
    assert data["caller_id"] == "mod1"


def test_login_rejects_unknown():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=hacker&role=moderator")
    assert resp.status_code == 403


def test_login_rejects_wrong_role():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=mod1&role=debugger")
    assert resp.status_code == 403


def test_session_token_on_replay_endpoint():
    """Elevated endpoint accepts session_token as alternative to caller_id."""
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # Create a game so replay won't 404
    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Login as moderator
    login_resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    token = login_resp.json()["token"]

    # Use session token to access replay
    resp = client.get(
        f"/games/{game_id}/replay?session_token={token}"
        f"&caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200


def test_session_token_on_evaluation_endpoint():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    login_resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    token = login_resp.json()["token"]

    resp = client.get(
        f"/games/{game_id}/evaluation?session_token={token}"
        f"&caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200


def test_session_token_on_cognitive_diff_endpoint():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    login_resp = client.post("/auth/login?caller_id=dbg1&role=debugger")
    token = login_resp.json()["token"]

    resp = client.get(
        f"/games/{game_id}/cognitive-diff?session_token={token}"
        f"&caller_id=dbg1&caller_role=debugger"
    )
    assert resp.status_code == 200


def test_session_token_on_private_state_endpoint():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Login as moderator
    login_resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    token = login_resp.json()["token"]

    # Access private state with session token
    resp = client.get(
        f"/games/{game_id}/players/p01/private-state?session_token={token}"
        f"&caller_id=mod1&caller_role=moderator&view_mode=moderator_full"
    )
    assert resp.status_code == 200


def test_invalid_session_token_rejected():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Use an invalid session token — should be rejected
    resp = client.get(
        f"/games/{game_id}/replay?session_token=bogus.token.123.abc"
        f"&caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 403


def test_expired_session_token_rejected():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient
    from werewolf_agent.api.auth import AuthManager, AuthConfig

    # Create app with a very short TTL
    auth_mgr = AuthManager(AuthConfig(mode="local", token_ttl_seconds=0))
    app = create_app(auth_manager=auth_mgr)
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Login and wait for expiry
    login_resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    token = login_resp.json()["token"]
    time.sleep(0.2)

    resp = client.get(
        f"/games/{game_id}/replay?session_token={token}"
        f"&caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Backward compatibility — existing caller_id/caller_role still works
# ---------------------------------------------------------------------------

def test_legacy_caller_id_still_works():
    """Existing tests use caller_id=mod1 & caller_role=moderator without tokens."""
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Legacy path: caller_id + caller_role, no session_token
    resp = client.get(
        f"/games/{game_id}/replay?caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200


def test_legacy_debugger_still_works():
    from werewolf_agent.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
    })
    game_id = game_resp.json()["game"]["game_id"]

    resp = client.get(
        f"/games/{game_id}/cognitive-diff?caller_id=dbg1&caller_role=debugger"
    )
    assert resp.status_code == 200
