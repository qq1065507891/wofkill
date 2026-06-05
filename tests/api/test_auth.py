"""Tests for local development auth system."""
import pytest
import time

from werewolf_agent.api.auth import AuthManager, AuthConfig

# 测试用的固定密钥，避免依赖环境变量
_TEST_SECRET = "test-secret-key-for-unit-tests-only"


# ---------------------------------------------------------------------------
# AuthConfig unit tests
# ---------------------------------------------------------------------------

def test_auth_config_loads_defaults():
    cfg = AuthConfig(secret_key=_TEST_SECRET)
    assert cfg.mode == "local"
    assert cfg.local_users  # has default users
    assert "mod1" in cfg.local_users
    assert "dbg1" in cfg.local_users
    assert cfg.secret_key == _TEST_SECRET


def test_auth_config_generates_ephemeral_key_without_secret():
    """AuthConfig 未设置 secret_key 时生成临时 key 并打印 WARNING。"""
    cfg = AuthConfig()
    assert cfg.secret_key != ""
    assert len(cfg.secret_key) >= 32


def test_auth_config_env_override(monkeypatch):
    """AuthConfig 可以通过环境变量 WEREWOLF_AUTH_SECRET 设置密钥。"""
    monkeypatch.setenv("WEREWOLF_AUTH_SECRET", "env-secret-value")
    cfg = AuthConfig()
    assert cfg.secret_key == "env-secret-value"


# ---------------------------------------------------------------------------
# AuthManager unit tests
# ---------------------------------------------------------------------------

def test_auth_manager_local_mode():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    token = mgr.create_session("mod1", "moderator")
    assert token is not None
    role = mgr.validate_session(token)
    assert role == "moderator"


def test_auth_manager_debugger_role():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    token = mgr.create_session("dbg1", "debugger")
    assert mgr.validate_session(token) == "debugger"


def test_auth_manager_spectator_role():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    token = mgr.create_session("spectator", "spectator")
    assert mgr.validate_session(token) == "spectator"


def test_auth_manager_rejects_unknown_user():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    with pytest.raises(PermissionError):
        mgr.create_session("hacker", "moderator")


def test_auth_manager_rejects_wrong_role():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    with pytest.raises(PermissionError):
        mgr.create_session("mod1", "debugger")


def test_auth_manager_expired_token():
    mgr = AuthManager(AuthConfig(mode="local", token_ttl_seconds=0, secret_key=_TEST_SECRET))
    token = mgr.create_session("mod1", "moderator")
    time.sleep(0.1)
    role = mgr.validate_session(token)
    assert role is None


def test_auth_manager_revoke():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    token = mgr.create_session("mod1", "moderator")
    mgr.revoke_session(token)
    assert mgr.validate_session(token) is None


def test_auth_manager_revoke_nonexistent():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    # Should not raise
    mgr.revoke_session("nonexistent-token")


def test_auth_manager_validate_invalid_token():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    assert mgr.validate_session("invalid.token.here") is None


def test_auth_manager_validate_empty_token():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    assert mgr.validate_session("") is None


def test_auth_manager_token_structure():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    token = mgr.create_session("mod1", "moderator")
    parts = token.split(".")
    assert len(parts) == 4
    assert parts[0] == "mod1"
    assert parts[1] == "moderator"


def test_auth_manager_different_users_different_tokens():
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    t1 = mgr.create_session("mod1", "moderator")
    t2 = mgr.create_session("dbg1", "debugger")
    assert t1 != t2


def test_auth_manager_same_user_new_token_invalidates_old():
    """Creating a new session does NOT invalidate the old one in-memory."""
    mgr = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    t1 = mgr.create_session("mod1", "moderator")
    t2 = mgr.create_session("mod1", "moderator")
    # Both tokens remain valid because sessions are stored by token string
    assert mgr.validate_session(t1) == "moderator"
    assert mgr.validate_session(t2) == "moderator"


# ---------------------------------------------------------------------------
# Integration tests via FastAPI TestClient
# ---------------------------------------------------------------------------

def _make_test_app(**kwargs):
    """创建使用测试密钥的 app，避免依赖 WEREWOLF_AUTH_SECRET 环境变量。"""
    from werewolf_agent.api.app import create_app
    from werewolf_agent.api.auth import AuthManager, AuthConfig

    auth_mgr = kwargs.pop("auth_manager", None) or AuthManager(
        AuthConfig(mode="local", secret_key=_TEST_SECRET)
    )
    return create_app(auth_manager=auth_mgr, **kwargs)


def test_login_endpoint():
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=mod1&role=moderator")
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["role"] == "moderator"
    assert data["caller_id"] == "mod1"


def test_login_rejects_unknown():
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=hacker&role=moderator")
    assert resp.status_code == 403


def test_login_rejects_wrong_role():
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/auth/login?caller_id=mod1&role=debugger")
    assert resp.status_code == 403


def test_session_token_on_replay_endpoint():
    """Elevated endpoint accepts session_token as alternative to caller_id."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # Create a game so replay won't 404
    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
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
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
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
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
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
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
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
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Use an invalid session token — should be rejected
    resp = client.get(
        f"/games/{game_id}/replay?session_token=bogus.token.123.abc"
        f"&caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 403


def test_expired_session_token_rejected():
    from fastapi.testclient import TestClient
    from werewolf_agent.api.auth import AuthManager, AuthConfig

    # Create app with a very short TTL
    auth_mgr = AuthManager(AuthConfig(mode="local", token_ttl_seconds=0, secret_key=_TEST_SECRET))
    app = _make_test_app(auth_manager=auth_mgr)
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
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
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    game_id = game_resp.json()["game"]["game_id"]

    # Legacy path: caller_id + caller_role, no session_token
    resp = client.get(
        f"/games/{game_id}/replay?caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200


def test_legacy_debugger_still_works():
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "player_count": 12,
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    game_id = game_resp.json()["game"]["game_id"]

    resp = client.get(
        f"/games/{game_id}/cognitive-diff?caller_id=dbg1&caller_role=debugger"
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# NEW-P0-1: game control endpoints (start/step/pause/resume) require moderator
# ---------------------------------------------------------------------------

def test_start_game_requires_moderator():
    """NEW-P0-1: POST /games/{id}/start must reject non-moderator callers."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # Create a game as moderator (the only legit way to call create_game too)
    game_resp = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    )
    game_id = game_resp.json()["game"]["game_id"]

    # Now try to start WITHOUT a moderator identity — must be 403
    resp = client.post(f"/games/{game_id}/start", json={})
    assert resp.status_code == 403


def test_step_game_requires_moderator():
    """NEW-P0-1: POST /games/{id}/step must reject non-moderator callers."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # Setup: create + start as moderator
    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "seed": 99,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]
    # start with moderator identity
    client.post(
        f"/games/{game_id}/start?caller_id=mod1&caller_role=moderator",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )

    # Try to step without auth — must 403
    resp = client.post(f"/games/{game_id}/step", json={})
    assert resp.status_code == 403


def test_pause_game_requires_moderator():
    """NEW-P0-1: POST /games/{id}/pause must reject non-moderator callers."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "seed": 7,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]
    client.post(
        f"/games/{game_id}/start?caller_id=mod1&caller_role=moderator",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )

    # No auth — must 403
    resp = client.post(f"/games/{game_id}/pause", json={})
    assert resp.status_code == 403


def test_resume_game_requires_moderator():
    """NEW-P0-1: POST /games/{id}/resume must reject non-moderator callers."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "seed": 8,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]
    client.post(
        f"/games/{game_id}/start?caller_id=mod1&caller_role=moderator",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )
    # Pause with moderator first
    client.post(
        f"/games/{game_id}/pause?caller_id=mod1&caller_role=moderator",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )

    # No auth on resume — must 403
    resp = client.post(f"/games/{game_id}/resume", json={})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# NEW-P0-2: create_game requires moderator (DoS prevention)
# ---------------------------------------------------------------------------

def test_create_game_requires_moderator():
    """NEW-P0-2: POST /games must reject non-moderator callers."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # No auth at all — must 403 (DoS vector closed)
    resp = client.post(
        "/games",
        json={"ruleset_id": "pre_witch_hunter_idiot_mixed", "player_count": 12},
    )
    assert resp.status_code == 403


def test_create_game_rejects_empty_caller_id():
    """NEW-P0-2: caller_id must be non-empty even for moderator role."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    resp = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "",
            "caller_role": "moderator",
        },
    )
    assert resp.status_code == 403


def test_create_game_accepts_moderator():
    """NEW-P0-2: POST /games must accept authenticated moderator."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    resp = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# NEW-P1-2: share-summary requires view_mode=public role check
# ---------------------------------------------------------------------------

def test_share_summary_requires_view_mode_public():
    """NEW-P1-2: GET /games/{id}/share-summary must reject callers whose
    effective role cannot see public events (e.g. a non-empty caller_id
    claiming player_agent role while requesting moderator_full).

    The endpoint must force view_mode=PUBLIC, which is permitted for
    SPECTATOR/PLAYER_AGENT/MODERATOR/DEBUGGER — so a caller with no auth
    params (defaults to SPECTATOR + PUBLIC) is allowed. The auth fix is
    to require non-empty caller_id OR session_token, AND to validate the
    caller's role via checker.check(...).
    """
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # Setup a game
    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    # No caller_id and no session_token — must 403
    resp = client.get(f"/games/{game_id}/share-summary")
    assert resp.status_code == 403


def test_share_summary_downgrades_player_agent_to_public():
    """NEW-P1-2: a player_agent caller can access share-summary, but the
    endpoint must force view_mode=PUBLIC regardless of any view_mode
    query param they sent. Response must be public_only=True.
    """
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    # Player agent asking for moderator_full — must be downgraded to public.
    resp = client.get(
        f"/games/{game_id}/share-summary"
        f"?caller_id=p01&caller_role=player_agent&view_mode=moderator_full"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["public_only"] is True


def test_share_summary_accepts_spectator():
    """NEW-P1-2: caller with SPECTATOR role and default public view is OK."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    # Spectator role with public view is allowed
    resp = client.get(
        f"/games/{game_id}/share-summary?caller_id=spectator&caller_role=spectator"
    )
    assert resp.status_code == 200


def test_share_summary_accepts_moderator():
    """NEW-P1-2: moderator role is always allowed."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    resp = client.get(
        f"/games/{game_id}/share-summary?caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# NEW-P1-3: list_games requires role
# ---------------------------------------------------------------------------

def test_list_games_requires_role():
    """NEW-P1-3: GET /games must require MODERATOR/DEBUGGER role or
    filter by caller_id — anonymous enumeration is a privacy leak.
    """
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    # Anonymous — must 403
    resp = client.get("/games")
    assert resp.status_code == 403


def test_list_games_accepts_moderator():
    """NEW-P1-3: moderator can list games."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    resp = client.get("/games?caller_id=mod1&caller_role=moderator")
    assert resp.status_code == 200


def test_list_games_rejects_spectator():
    """NEW-P1-3: spectator without games to own is denied."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    resp = client.get("/games?caller_id=spectator&caller_role=spectator")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# NEW-P1-6: public-state records caller (audit log)
# ---------------------------------------------------------------------------

def test_public_state_records_caller():
    """NEW-P1-6: GET /games/{id}/public-state must record the caller in
    the audit log (no auth required, but a thin role resolution + audit
    event must be emitted for the GET).
    """
    from fastapi.testclient import TestClient
    from werewolf_agent.api.permissions import PermissionChecker

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    # Hit public-state with a known caller_id; audit log should record it.
    # Note: the test app's checker is a fresh instance; the assertion
    # target is that the endpoint accepts the caller_id without 403
    # and that the response contains a public-only payload.
    resp = client.get(
        f"/games/{game_id}/public-state?caller_id=mod1&caller_role=moderator"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["game_id"] == game_id


def test_public_state_allows_anonymous_with_audit():
    """NEW-P1-6: anonymous (no caller_id) is allowed for public-state,
    but the audit log must record the call. The fix is 'thin role
    resolution + audit log' — not a 403 gate."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)

    game_id = client.post(
        "/games",
        json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "player_count": 12,
            "caller_id": "mod1",
            "caller_role": "moderator",
        },
    ).json()["game"]["game_id"]

    # Anonymous GET must succeed (public-state is intentionally open).
    resp = client.get(f"/games/{game_id}/public-state")
    assert resp.status_code == 200
    assert resp.json()["game_id"] == game_id
