"""Tests for the NEW-P1-5 cognitive_diff always-empty fix.

The cognitive_diff view previously always received ``cognition_data=None``,
so every entry was hardcoded to ``guessed_role="unknown"``,
``guessed_confidence=0.0``, ``faction_read="unknown"``. The fix is to
compute real belief data from the game state via the belief updater and
pass it into ``build_cognitive_diff``.
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


def test_cognitive_diff_returns_real_belief_state():
    """The cognitive-diff endpoint must return non-empty belief data
    derived from the actual game state, not always-empty defaults.

    Before the fix, every entry was ``{"guessed_role": "unknown",
    "guessed_confidence": 0.0, "faction_read": "unknown"}``. With real
    belief data, at least one entry should have a non-default value.
    """
    client, game_id = _make_client()

    resp = client.get(
        f"/games/{game_id}/cognitive-diff",
        params={"caller_id": "dbg1", "caller_role": "debugger", "player_id": "p01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    entries = data["entries"]
    assert len(entries) >= 1, "expected at least one entry"
    # In a freshly-started game the players have roles assigned but no
    # observations yet, so the belief updater will produce uniform
    # probabilities — which means guessed_role will be a real role name
    # (not "unknown") once we wire it up. If we see "unknown" on every
    # entry, the bug is still there.
    unknowns = sum(1 for e in entries if e["guessed_role"] == "unknown")
    real = len(entries) - unknowns
    assert real > 0, (
        f"NEW-P1-5 not fixed: every entry has guessed_role='unknown' "
        f"({unknowns}/{len(entries)}); cognitive_diff is not consulting "
        f"the belief updater."
    )
