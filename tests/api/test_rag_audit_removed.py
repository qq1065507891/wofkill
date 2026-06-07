"""Tests for the U1 (post-review-v2) restoration of the /rag-audit endpoint.

History: NEW-P2-6 had removed this endpoint because it always returned an
empty list (no code path emits ``rag_injection_audit`` events). U1 reverses
that decision because ``dashboard.js`` calls the endpoint unconditionally
and the resulting 404 in the browser console is a visible regression.

The endpoint is now useful for any future code path that emits
``rag_injection_audit`` events (e.g. RAG injection logging in
``rag/knowledge_service.py``) — see the rag_audit_event_type.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def test_rag_audit_endpoint_exists() -> None:
    """U1: /games/{id}/rag-audit 路由已加回；moderator 可调用并收到
    {"rag_audits": [...]} 响应。"""
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post(
        "/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod}
    )
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)

    resp = client.get(
        f"/games/{game_id}/rag-audit",
        params={"caller_id": "mod1", "caller_role": "moderator"},
    )
    assert resp.status_code == 200, (
        f"U1: /rag-audit endpoint should return 200 (route re-added), "
        f"got {resp.status_code} body={resp.text!r}"
    )
    data = resp.json()
    assert "rag_audits" in data, f"U1: response missing 'rag_audits' key: {data!r}"
    assert isinstance(data["rag_audits"], list)
    # 当前 game 没有 RAG 注入事件，列表应为空
    assert data["rag_audits"] == []


def test_rag_audit_endpoint_denies_spectator() -> None:
    """U1: spectator 角色不能看 rag-audit（403）。"""
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post(
        "/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod}
    )
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)

    # Spectator without auth
    resp = client.get(
        f"/games/{game_id}/rag-audit",
        params={"caller_id": "spec1", "caller_role": "spectator"},
    )
    assert resp.status_code in (403, 401), (
        f"U1: spectator should be denied, got {resp.status_code}"
    )
