"""Tests for the NEW-P2-6 removal of dead /rag-audit endpoint.

The /games/{id}/rag-audit endpoint queried state.events for
'rag_injection_audit' entries, but no code path emits that event type —
the endpoint always returned an empty list. It is removed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def test_rag_audit_endpoint_removed():
    """NEW-P2-6: /games/{id}/rag-audit must return 404 (not 200) because
    the endpoint was removed (it was always returning an empty list)."""
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod})
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)

    # The endpoint should not exist (404). With the dead endpoint
    # present, it would have returned 200 with an empty rag_audits list.
    resp = client.get(
        f"/games/{game_id}/rag-audit",
        params={"caller_id": "mod1", "caller_role": "moderator"},
    )
    assert resp.status_code == 404, (
        f"NEW-P2-6 not fixed: /rag-audit endpoint should be removed "
        f"(404), got {resp.status_code} body={resp.text!r}"
    )
