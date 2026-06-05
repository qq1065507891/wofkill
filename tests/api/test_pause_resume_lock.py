"""Tests for the NEW-P1-4 pause/resume TOCTOU fix.

pause_game and resume_game must acquire the executor's per-game lock to
avoid a TOCTOU race with an in-flight step. Without the lock, a step
running in a background thread can swap state in/out from underneath
pause/resume, causing lost updates or out-of-order events.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.api.schemas import CallerRole
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def _make_client() -> TestClient:
    auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)
    client = TestClient(app)
    # Create + start a game so a runner is registered.
    mod = {"caller_id": "mod1", "caller_role": "moderator"}
    resp = client.post("/games", json={"ruleset_id": "pre_witch_hunter_idiot_mixed", **mod})
    game_id = resp.json()["game"]["game_id"]
    client.post(f"/games/{game_id}/start", json=mod)
    return client, game_id


def test_pause_resume_serialized_with_step():
    """pause/resume must hold the executor's per-game lock so they
    cannot interleave with an in-flight step.

    The fix: pause_game and resume_game should call
    ``executor.lock_for(game_id).acquire()`` (with timeout / try-acquire
    so the request doesn't block forever) before mutating
    ``state.paused`` and the runner's ``_state``.
    """
    client, game_id = _make_client()
    executor = client.app.state.executor
    mod = {"caller_id": "mod1", "caller_role": "moderator"}

    # Acquire the lock the way a background step would — non-blocking.
    lock = executor.lock_for(game_id)
    assert lock.acquire(blocking=False) is True, "lock should be free initially"

    # While the lock is held, pause_game must NOT mutate state.
    # If pause/resume use the lock, they will fail to acquire it and
    # raise 409. (We accept either 409 or 400 — both demonstrate that
    # the request did not silently race.)
    resp = client.post(f"/games/{game_id}/pause", json=mod)
    assert resp.status_code in (409, 423, 400, 500), (
        f"pause_game must be serialized with step; got status={resp.status_code} "
        f"detail={resp.text!r}"
    )
    # Most importantly: the request must NOT have completed with 200
    # while the lock was held (that would be the bug).
    assert resp.status_code != 200, (
        "pause_game returned 200 while executor lock was held — TOCTOU bug"
    )

    # Release and verify the next pause works.
    lock.release()
    resp = client.post(f"/games/{game_id}/pause", json=mod)
    assert resp.status_code == 200
    # resume while another step holds the lock should also fail.
    assert lock.acquire(blocking=False) is True
    resp = client.post(f"/games/{game_id}/resume", json=mod)
    assert resp.status_code != 200, (
        "resume_game returned 200 while executor lock was held — TOCTOU bug"
    )
    lock.release()
    resp = client.post(f"/games/{game_id}/resume", json=mod)
    assert resp.status_code == 200
