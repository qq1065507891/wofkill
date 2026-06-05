"""Tests for the NEW-P2-4 start_game 409 fix and related route guards.

start_game previously only checked ``state.phase != \"setup\"`` to detect
\"already started\" — but the runner is the resource that's actually
running. If a runner is already in ``runners`` (e.g. from a previous
start that didn't clean up, or a race), the second start would
overwrite it. The fix: 409 if the runner already exists.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def test_start_game_returns_409_if_already_running():
    """Calling /games/{id}/start twice must return 409 on the second
    call, because the runner is already in the runners dict."""
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    mod = {"caller_id": "mod1", "caller_role": "moderator"}

    game_id = client.post(
        "/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod}
    ).json()["game"]["game_id"]

    # First start: succeeds.
    first = client.post(f"/games/{game_id}/start", json=mod)
    assert first.status_code == 200, first.text

    # Sanity: runner is registered.
    assert game_id in app.state.runners

    # Simulate the bug condition: the runner is in `runners` but the
    # state has been reset to "setup" (e.g. a test fixture or a
    # hypothetical operator reset). The current code only checks
    # state.phase, so it would 400 "Game already started" instead of
    # the more informative 409 "runner already running".
    #
    # To test the exact NEW-P2-4 contract, restart the game from a
    # fresh state but with the runner still present. The simplest way
    # is to call start again on a different game whose runner slot
    # is already populated, but the contract per the spec is:
    # 409 if game_id in runners.
    new_state_game_id = "g_synthetic_id"
    # Pre-populate runners to mimic "runner already exists".
    class _FakeRunner:
        pass
    app.state.runners[new_state_game_id] = _FakeRunner()
    # Create a real game with that id by directly seeding games dict.
    from werewolf_agent.core.models import GameState
    app.state.games[new_state_game_id] = GameState(
        game_id=new_state_game_id, phase="setup",
    )

    resp = client.post(f"/games/{new_state_game_id}/start", json=mod)
    assert resp.status_code == 409, (
        f"NEW-P2-4 not fixed: expected 409 when runner already exists, "
        f"got {resp.status_code} {resp.text}"
    )
