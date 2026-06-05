"""Tests for the NEW-P1-7 evaluation cross-game audit leak fix and
NEW-P2-2 info_leak_count from denial count.

The evaluation endpoint was returning the entire ``checker.audit_log()``
which is process-wide — so audit events from game A leaked into the
evaluation response of game B. The fix filters by the request's
``game_id``.

Additionally, ``info_leak_count`` was hardcoded to 0 in the view layer.
It should be computed as the number of denied audit events.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.api.schemas import ViewMode
from werewolf_agent.api.views import build_evaluation
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def _make_client() -> tuple[TestClient, str, str]:
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    g1 = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod}).json()["game"]["game_id"]
    client.post(f"/games/{g1}/start", json=mod)
    g2 = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod}).json()["game"]["game_id"]
    client.post(f"/games/{g2}/start", json=mod)
    return client, g1, g2


def test_evaluation_filters_audit_by_game_id():
    """Evaluation response for game A must not include audit events
    generated for game B (or any other game)."""
    client, g1, g2 = _make_client()

    # Generate audit events for both games by hitting moderator_full
    # endpoints with caller_id=mod1 (authorized), then attempting the
    # same as p01 (denied) to produce audit denials for each game.
    for gid in (g1, g2):
        # successful moderator access — should appear in the audit log
        client.get(
            f"/games/{gid}/replay",
            params={"caller_id": "mod1", "caller_role": "moderator"},
        )
        # denied player access — produces a denial audit event
        client.get(
            f"/games/{gid}/replay",
            params={"caller_id": "p01", "caller_role": "player_agent", "view_mode": "moderator_full"},
        )

    # Now query evaluation for g1 only.
    resp = client.get(
        f"/games/{g1}/evaluation",
        params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
    )
    assert resp.status_code == 200
    audit_events = resp.json()["metrics"]["audit_events"]
    assert audit_events, "expected at least one audit event to be returned"

    # Each returned event must belong to g1.
    for event in audit_events:
        assert event.get("game_id") == g1, (
            f"NEW-P1-7 not fixed: evaluation for game {g1} leaked audit "
            f"event from game {event.get('game_id')!r}"
        )

    # Sanity: the audit log on the checker should still contain BOTH
    # games' events (the bug was about the *response* filter, not the
    # log itself).
    checker = client.app.state.checker
    seen_game_ids = {e.game_id for e in checker.audit_log()}
    assert g1 in seen_game_ids and g2 in seen_game_ids, (
        "test setup error: expected audit log to contain both games"
    )


def test_info_leak_count_uses_denial_count():
    """info_leak_count must equal the number of denied audit events.

    Tests the view function directly so we control the audit_events
    payload. Two denials + one grant should yield info_leak_count == 2.
    """
    state = GameState(game_id="g_info_leak", players={
        "p1": PlayerState(id="p1", role="villager"),
        "p2": PlayerState(id="p2", role="seer"),
    })
    audit_events = [
        {"game_id": "g_info_leak", "granted": False, "caller_id": "p1",
         "caller_role": "player_agent", "requested_view": "moderator_full",
         "endpoint": "replay"},
        {"game_id": "g_info_leak", "granted": False, "caller_id": "p1",
         "caller_role": "player_agent", "requested_view": "moderator_full",
         "endpoint": "evaluation"},
        {"game_id": "g_info_leak", "granted": True, "caller_id": "mod1",
         "caller_role": "moderator", "requested_view": "moderator_full",
         "endpoint": "replay"},
    ]
    response = build_evaluation(state, ViewMode.MODERATOR_FULL, audit_events=audit_events)
    assert response.metrics is not None
    assert response.metrics.info_leak_count == 2, (
        f"NEW-P2-2 not fixed: expected info_leak_count=2 (denied events), "
        f"got {response.metrics.info_leak_count}"
    )
