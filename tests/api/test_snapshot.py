"""Tests for the NEW-P2-3 /snapshot alias for the single-snapshot replay.

The /replay endpoint has always returned exactly one ReplaySnapshot
built from the current GameState, not a historical sequence. The name
is misleading. /snapshot exposes the same behavior under a clearer
name. /replay is kept for backward compatibility.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def _make_client() -> tuple[TestClient, str]:
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod})
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)
    return client, game_id


def test_snapshot_endpoint_exists_and_works():
    """NEW-P2-3: /games/{id}/snapshot must exist and return the
    single-snapshot ReplayResponse shape."""
    client, game_id = _make_client()

    resp = client.get(
        f"/games/{game_id}/snapshot",
        params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
    )
    assert resp.status_code == 200, (
        f"NEW-P2-3 not fixed: /snapshot endpoint returned {resp.status_code} "
        f"body={resp.text!r}"
    )
    data = resp.json()
    # The replay/snapshot response is a single-element snapshot list.
    assert "snapshots" in data
    assert len(data["snapshots"]) == 1, (
        f"expected single snapshot, got {len(data['snapshots'])}"
    )


def test_replay_endpoint_still_works():
    """Sanity: /replay must still work for backward compatibility."""
    client, game_id = _make_client()

    resp = client.get(
        f"/games/{game_id}/replay",
        params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
    )
    assert resp.status_code == 200
    assert "snapshots" in resp.json()
