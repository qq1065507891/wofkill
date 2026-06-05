"""Tests for the NEW-P2-9 private-state 404 fix, the NEW-P2-10
known-sensitive-field list, and the NEW-P2-11 MVP prefer-villager fix.

NEW-P2-9: build_private_state previously returned ``role="unknown"`` when
the player wasn't found in the game, silently masking typos and stale
ids. The fix returns 404 from the route handler.

NEW-P2-10: the timeline view's "strip private info" pass had a hard-coded
list of payload keys. New private keys would silently leak through. The
fix centralizes the list in a module-level constant so the test can
assert it covers every event type.

NEW-P2-11: _pick_public_mvp_candidate sorted by player id and returned
the lowest, so whoever happened to be p01 always won. The fix
prefers an alive good-faction player.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.api.views import (
    KNOWN_SENSITIVE_FIELDS,
    _build_event,
    _build_public_event,
)
from werewolf_agent.api.routes.games import _pick_public_mvp_candidate
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.api.schemas import ViewMode
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


def test_private_state_404_when_player_not_found():
    """NEW-P2-9: asking for a non-existent player's private state must
    return 404, not a 200 with role='unknown'."""
    client, game_id = _make_client()

    resp = client.get(
        f"/games/{game_id}/players/p99/private-state",
        params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
    )
    assert resp.status_code == 404, (
        f"NEW-P2-9 not fixed: expected 404 for missing player, "
        f"got {resp.status_code} body={resp.text!r}"
    )


def test_private_state_200_for_real_player():
    """Sanity: a real player still gets 200 from private-state."""
    client, game_id = _make_client()

    resp = client.get(
        f"/games/{game_id}/players/p01/private-state",
        params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
    )
    assert resp.status_code == 200
    assert resp.json()["player_info"]["player_id"] == "p01"


# ---------------------------------------------------------------------------
# NEW-P2-10
# ---------------------------------------------------------------------------


def test_known_sensitive_fields_is_nonempty_constant():
    """NEW-P2-10: KNOWN_SENSITIVE_FIELDS must be a module-level
    non-empty frozenset (or set) so the timeline view consults it."""
    assert isinstance(KNOWN_SENSITIVE_FIELDS, (set, frozenset))
    # It must include the obvious private payloads from the existing
    # view code so the refactor doesn't accidentally drop coverage.
    for field in ("actual_role", "wolf_teammates", "private_intent"):
        assert field in KNOWN_SENSITIVE_FIELDS, (
            f"NEW-P2-10: {field!r} missing from KNOWN_SENSITIVE_FIELDS"
        )


def test_all_event_types_filter_known_sensitive_fields():
    """NEW-P2-10: for any event type, the public + player_view builders
    must strip every key in KNOWN_SENSITIVE_FIELDS, not just the three
    the old hard-coded list mentioned. The test uses a synthetic event
    carrying a *new* sensitive key (e.g. ``seer_target_id``) that was
    not in the legacy list — if the view consults the constant it will
    be stripped; if it consults the legacy literal it will leak.
    """
    # Add a new sensitive field to the constant to simulate a future
    # contributor extending the set. This proves the view *reads* the
    # constant dynamically.
    new_field = "seer_target_id"
    # We deliberately mutate the module-level set so the view picks it
    # up. Restore in finally.
    original = set(KNOWN_SENSITIVE_FIELDS)
    KNOWN_SENSITIVE_FIELDS.add(new_field)
    try:
        # Use an event type that's neither public-only nor player-only
        # (e.g. wolf_discussion) so we exercise the non-moderator path
        # where the strip happens.
        event = GameEvent(
            type="wolf_kill_selected",
            payload={
                "night_number": 1, "phase": "night",
                "actual_role": "werewolf",  # legacy sensitive
                "seer_target_id": "p05",     # new sensitive
            },
        )

        # public view must strip every key in the constant
        public = _build_public_event(event)
        assert new_field not in public.data, (
            f"NEW-P2-10: public view leaked {new_field!r} — view is not "
            "consulting KNOWN_SENSITIVE_FIELDS"
        )
        assert "actual_role" not in public.data

        # player_view (non-moderator) must also strip them
        from werewolf_agent.api.views import _build_event
        player = _build_event(event, ViewMode.PLAYER_VIEW)
        assert new_field not in player.data
        assert "actual_role" not in player.data

        # moderator_full keeps them (this is the whole point of the
        # view mode system — moderator sees everything)
        mod = _build_event(event, ViewMode.MODERATOR_FULL)
        assert "actual_role" in mod.data
        assert new_field in mod.data, (
            "moderator_full must keep all fields; found leak in the "
            "opposite direction"
        )
    finally:
        # Restore the constant to its declared state.
        KNOWN_SENSITIVE_FIELDS.clear()
        KNOWN_SENSITIVE_FIELDS.update(original)


# ---------------------------------------------------------------------------
# NEW-P2-11
# ---------------------------------------------------------------------------


def test_mvp_picks_first_alive_villager():
    """NEW-P2-11: MVP candidate must prefer an alive good-faction
    player (villager/seer/etc.) over a wolf, even when the wolf has a
    lower id. The legacy code just sorted by id, so ``p01`` always won
    regardless of role.
    """
    # p01 is a wolf, p02 is a villager → MVP must be p02.
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="seer", alive=True),
        "p04": PlayerState(id="p04", role="idiot", alive=False),  # dead
    }
    state = GameState(game_id="g_mvp", players=players)
    assert _pick_public_mvp_candidate(state) == "p02", (
        "NEW-P2-11 not fixed: MVP picked by id-sort, ignoring role"
    )

    # Sanity: when only wolves are alive, MVP must still return *some*
    # alive id (the fallback path).
    only_wolves = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p05": PlayerState(id="p05", role="werewolf", alive=True),
    }
    state2 = GameState(game_id="g_mvp2", players=only_wolves)
    assert _pick_public_mvp_candidate(state2) == "p01"

    # Edge: no alive players → return lowest id from the full roster.
    all_dead = {
        "p01": PlayerState(id="p01", role="villager", alive=False),
        "p02": PlayerState(id="p02", role="werewolf", alive=False),
    }
    state3 = GameState(game_id="g_mvp3", players=all_dead)
    assert _pick_public_mvp_candidate(state3) == "p01"
