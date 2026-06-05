"""API permission tests: view modes, boundary enforcement, audit logging."""

import pytest
from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthManager, AuthConfig
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.schemas import (
    AuditEvent,
    CallerRole,
    ViewMode,
)
from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.storage.memory_store import InMemoryGameRepository

# 测试用的固定密钥，避免依赖 WEREWOLF_AUTH_SECRET 环境变量
_TEST_SECRET = "test-secret-key-for-unit-tests-only"
_test_auth = AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))


def test_create_app_initializes_rag_service_from_env(monkeypatch):
    monkeypatch.setenv("WEREWOLF_VECTOR_BACKEND", "local")

    app = create_app(repository=InMemoryGameRepository(), auth_manager=_test_auth)

    assert hasattr(app.state, "rag_service")
    assert app.state.rag_service is not None
    assert any(
        entry["entry_id"] == "seed_jingcheng_wolf_god_hunt_260227"
        for entry in app.state.repository.load_rag_entries()
    )


def _make_client() -> TestClient:
    app = create_app(auth_manager=_test_auth)
    client = TestClient(app)
    # Create and start a game — game control endpoints require moderator auth
    resp = client.post("/games", json={
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    game_id = resp.json()["game"]["game_id"]
    client.post(
        f"/games/{game_id}/start",
        json={"caller_id": "mod1", "caller_role": "moderator"},
    )
    return client, game_id


def _find_player_by_role(client: TestClient, game_id: str, role: str) -> str | None:
    """Find a player ID with the given role via internal game state."""
    # Access the app's internal game state for role lookup.
    # This is acceptable in tests where we need to know role assignments
    # that are not available through public API endpoints.
    app = client.app  # type: ignore[attr-defined]
    gs = app.state.games.get(game_id)
    if gs is None:
        return None
    for pid, p in gs.players.items():
        if p.role == role:
            return pid
    return None


# ---------------------------------------------------------------------------
# PermissionChecker unit tests
# ---------------------------------------------------------------------------

class TestPermissionChecker:

    def test_moderator_can_access_moderator_full(self):
        pc = PermissionChecker()
        result = pc.check("mod1", CallerRole.MODERATOR, ViewMode.MODERATOR_FULL, game_active=True)
        assert result == ViewMode.MODERATOR_FULL

    def test_debugger_can_access_moderator_full(self):
        pc = PermissionChecker()
        result = pc.check("dbg1", CallerRole.DEBUGGER, ViewMode.MODERATOR_FULL, game_active=True)
        assert result == ViewMode.MODERATOR_FULL

    def test_spectator_can_access_public(self):
        pc = PermissionChecker()
        result = pc.check("spec1", CallerRole.SPECTATOR, ViewMode.PUBLIC)
        assert result == ViewMode.PUBLIC

    def test_spectator_player_view_downgraded_to_public(self):
        pc = PermissionChecker()
        result = pc.check("spec1", CallerRole.SPECTATOR, ViewMode.PLAYER_VIEW)
        assert result == ViewMode.PUBLIC

    def test_spectator_cannot_access_moderator_full(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied) as exc_info:
            pc.check("spec1", CallerRole.SPECTATOR, ViewMode.MODERATOR_FULL)
        assert "moderator_full" in str(exc_info.value).lower() or "Spectator" in str(exc_info.value)

    def test_player_agent_can_access_public(self):
        pc = PermissionChecker()
        result = pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.PUBLIC)
        assert result == ViewMode.PUBLIC

    def test_player_agent_can_access_own_player_view(self):
        pc = PermissionChecker()
        result = pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.PLAYER_VIEW)
        assert result == ViewMode.PLAYER_VIEW

    def test_player_agent_cannot_access_moderator_full_live(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied) as exc_info:
            pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL, game_active=True)
        assert "live play" in str(exc_info.value).lower() or "Player" in str(exc_info.value)

    def test_player_agent_cannot_access_moderator_full_post_game(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL, game_active=False)

    def test_private_state_own_player(self):
        pc = PermissionChecker()
        result = pc.check_private_state("p1", CallerRole.PLAYER_AGENT, "p1")
        assert result == ViewMode.PLAYER_VIEW

    def test_private_state_other_player_denied(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied) as exc_info:
            pc.check_private_state("p1", CallerRole.PLAYER_AGENT, "p2")
        assert "p2" in str(exc_info.value)

    def test_private_state_spectator_denied(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check_private_state("spec1", CallerRole.SPECTATOR, "p1")

    def test_private_state_moderator_allowed(self):
        pc = PermissionChecker()
        result = pc.check_private_state("mod1", CallerRole.MODERATOR, "p1")
        assert result == ViewMode.MODERATOR_FULL

    def test_cognitive_diff_moderator_allowed(self):
        pc = PermissionChecker()
        result = pc.check_cognitive_diff("mod1", CallerRole.MODERATOR)
        assert result == ViewMode.MODERATOR_FULL

    def test_cognitive_diff_player_denied(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check_cognitive_diff("p1", CallerRole.PLAYER_AGENT)

    def test_cognitive_diff_spectator_denied(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check_cognitive_diff("spec1", CallerRole.SPECTATOR)

    def test_denial_logged_in_audit(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL, game_active=True)
        assert len(pc.denials()) == 1
        assert pc.denials()[0].granted is False

    def test_audit_log_tracks_all(self):
        pc = PermissionChecker()
        # Successful check for moderator
        pc.check("mod1", CallerRole.MODERATOR, ViewMode.MODERATOR_FULL)
        # Spectator player_view gets logged (downgraded)
        pc.check("spec1", CallerRole.SPECTATOR, ViewMode.PLAYER_VIEW)
        # Denied check for player
        with pytest.raises(PermissionDenied):
            pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL, game_active=True)
        assert len(pc.audit_log()) == 2  # Only logged events (spectator downgrade + denial)
        assert len(pc.denials()) == 1

    def test_clear_audit_log(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied):
            pc.check("p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL, game_active=True)
        pc.clear_audit_log()
        assert len(pc.audit_log()) == 0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:

    def test_create_game(self):
        client, game_id = _make_client()
        assert game_id.startswith("g_")

    def test_list_games(self):
        client, _ = _make_client()
        resp = client.get("/games")
        assert resp.status_code == 200
        assert len(resp.json()["game_ids"]) >= 1

    def test_start_game(self):
        client, game_id = _make_client()
        # Game already started in _make_client
        resp = client.post(
            f"/games/{game_id}/start",
            json={"caller_id": "mod1", "caller_role": "moderator"},
        )
        assert resp.status_code == 400  # Already started

    def test_pause_resume(self):
        client, game_id = _make_client()
        mod = {"caller_id": "mod1", "caller_role": "moderator"}
        resp = client.post(f"/games/{game_id}/pause", json=mod)
        assert resp.status_code == 200
        resp = client.post(f"/games/{game_id}/resume", json=mod)
        assert resp.status_code == 200

    def test_pause_already_paused(self):
        client, game_id = _make_client()
        mod = {"caller_id": "mod1", "caller_role": "moderator"}
        client.post(f"/games/{game_id}/pause", json=mod)
        resp = client.post(f"/games/{game_id}/pause", json=mod)
        assert resp.status_code == 400

    def test_resume_not_paused(self):
        client, game_id = _make_client()
        resp = client.post(
            f"/games/{game_id}/resume",
            json={"caller_id": "mod1", "caller_role": "moderator"},
        )
        assert resp.status_code == 400

    def test_public_state(self):
        client, game_id = _make_client()
        resp = client.get(f"/games/{game_id}/public-state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_id"] == game_id
        assert len(data["players"]) == 12
        assert data["alive_count"] == 12

    def test_game_not_found(self):
        client, _ = _make_client()
        resp = client.get("/games/nonexistent/public-state")
        assert resp.status_code == 404


class TestPrivateStatePermissions:

    def test_own_private_state_allowed(self):
        client, game_id = _make_client()
        wolf_id = _find_player_by_role(client, game_id, "werewolf")
        assert wolf_id is not None, "No werewolf found in game"
        resp = client.get(
            f"/games/{game_id}/players/{wolf_id}/private-state",
            params={"caller_id": wolf_id, "caller_role": "player_agent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["viewer_id"] == wolf_id
        assert data["player_info"]["role"] == "werewolf"
        assert data["view_mode"] == "player_view"

    def test_wolf_teammates_visible_to_self(self):
        client, game_id = _make_client()
        wolf_id = _find_player_by_role(client, game_id, "werewolf")
        assert wolf_id is not None, "No werewolf found in game"
        resp = client.get(
            f"/games/{game_id}/players/{wolf_id}/private-state",
            params={"caller_id": wolf_id, "caller_role": "player_agent"},
        )
        data = resp.json()
        wolf_teammates = data["player_info"].get("wolf_teammates")
        assert wolf_teammates is not None
        assert len(wolf_teammates) == 3  # 4 wolves minus self

    def test_witch_potion_availability(self):
        client, game_id = _make_client()
        witch_id = _find_player_by_role(client, game_id, "witch")
        assert witch_id is not None, "No witch found in game"
        resp = client.get(
            f"/games/{game_id}/players/{witch_id}/private-state",
            params={"caller_id": witch_id, "caller_role": "player_agent"},
        )
        data = resp.json()
        info = data["player_info"]
        assert info["antidote_available"] is True
        assert info["poison_available"] is True

    def test_other_player_private_state_denied(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/players/p02/private-state",
            params={"caller_id": "p01", "caller_role": "player_agent"},
        )
        assert resp.status_code == 403

    def test_spectator_private_state_denied(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/players/p01/private-state",
            params={"caller_id": "spec1", "caller_role": "spectator"},
        )
        assert resp.status_code == 403

    def test_moderator_private_state_allowed(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/players/p01/private-state",
            params={"caller_id": "mod1", "caller_role": "moderator"},
        )
        assert resp.status_code == 200
        assert resp.json()["view_mode"] == "moderator_full"

    def test_query_role_spoof_cannot_read_moderator_private_state(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/players/p01/private-state",
            params={"caller_id": "p02", "caller_role": "moderator"},
        )
        assert resp.status_code == 403


class TestTimelinePermissions:

    def test_public_timeline(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/timeline",
            params={"caller_role": "spectator", "view_mode": "public"},
        )
        assert resp.status_code == 200
        assert resp.json()["view_mode"] == "public"

    def test_spectator_player_view_downgraded(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/timeline",
            params={"caller_id": "spec1", "caller_role": "spectator", "view_mode": "player_view"},
        )
        assert resp.status_code == 200
        assert resp.json()["view_mode"] == "public"  # Downgraded

    def test_spectator_moderator_full_denied(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/timeline",
            params={"caller_id": "spec1", "caller_role": "spectator", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 403

    def test_player_agent_moderator_full_denied_live(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/timeline",
            params={"caller_id": "p01", "caller_role": "player_agent", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 403


class TestReplayPermissions:

    def test_moderator_replay(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/replay",
            params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 200
        data = resp.json()
        snap = data["snapshots"][0]
        assert "moderator_full" in snap
        # moderator_full should contain all roles
        assert "all_roles" in snap["moderator_full"]

    def test_player_replay_denied_moderator_full(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/replay",
            params={"caller_id": "p01", "caller_role": "player_agent", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 403

    def test_player_replay_only_contains_own_private_view(self):
        client, game_id = _make_client()
        wolf_id = _find_player_by_role(client, game_id, "werewolf")
        assert wolf_id is not None, "No werewolf found in game"
        resp = client.get(
            f"/games/{game_id}/replay",
            params={"caller_id": wolf_id, "caller_role": "player_agent", "view_mode": "player_view"},
        )
        assert resp.status_code == 200
        snapshot = resp.json()["snapshots"][0]
        assert set(snapshot["player_views"]) == {wolf_id}
        assert snapshot["player_views"][wolf_id]["player_info"]["role"] == "werewolf"


class TestEvaluationPermissions:

    def test_moderator_evaluation(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/evaluation",
            params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # moderator_full shows roles
        stats = data["metrics"]["player_stats"]
        for pid, s in stats.items():
            assert s["role"] != "[hidden]"

    def test_public_evaluation_hides_roles(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/evaluation",
            params={"caller_role": "spectator", "view_mode": "public"},
        )
        assert resp.status_code == 200
        data = resp.json()
        stats = data["metrics"]["player_stats"]
        for pid, s in stats.items():
            assert s["role"] == "[hidden]"


class TestCognitiveDiffPermissions:

    def test_moderator_cognitive_diff(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/cognitive-diff",
            params={"caller_id": "mod1", "caller_role": "moderator", "player_id": "p01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # moderator_full should show actual roles
        for entry in data["entries"]:
            assert entry["actual_role"] is not None

    def test_debugger_cognitive_diff(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/cognitive-diff",
            params={"caller_id": "dbg1", "caller_role": "debugger", "player_id": "p01"},
        )
        assert resp.status_code == 200

    def test_player_cognitive_diff_denied(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/cognitive-diff",
            params={"caller_id": "p01", "caller_role": "player_agent", "player_id": "p01"},
        )
        assert resp.status_code == 403

    def test_spectator_cognitive_diff_denied(self):
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/cognitive-diff",
            params={"caller_id": "spec1", "caller_role": "spectator"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Information leak prevention
# ---------------------------------------------------------------------------

class TestInformationLeakPrevention:

    def test_public_state_no_roles(self):
        """Public state must not reveal player roles."""
        client, game_id = _make_client()
        resp = client.get(f"/games/{game_id}/public-state")
        data = resp.json()
        for p in data["players"]:
            assert "role" not in p or p.get("revealed_role") is None

    def test_public_state_shows_revealed_living_idiot(self):
        client, game_id = _make_client()
        state = client.app.state.games[game_id]
        state.players["p11"] = PlayerState(
            id="p11",
            role="idiot",
            alive=True,
            revealed_idiot=True,
            vote_enabled=False,
            badge_eligible=False,
            exile_immune=True,
        )

        resp = client.get(f"/games/{game_id}/public-state")
        data = resp.json()
        idiot = next(p for p in data["players"] if p["player_id"] == "p11")
        assert idiot["revealed_role"] == "idiot"
        assert idiot["alive"] is True

    def test_private_state_no_other_private_intent(self):
        """Private state response must not contain other players' private_intent."""
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/players/p01/private-state",
            params={"caller_id": "p01", "caller_role": "player_agent"},
        )
        data = resp.json()
        data_str = str(data)
        assert "private_intent" not in data_str

    def test_replay_moderator_full_no_private_intent(self):
        """Even moderator_full replay must not expose raw private_intent."""
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/replay",
            params={"caller_id": "mod1", "caller_role": "moderator", "view_mode": "moderator_full"},
        )
        data = resp.json()
        mf = data["snapshots"][0].get("moderator_full", {})
        assert mf.get("private_intents") == "[redacted in api - audit only]"

    def test_player_cannot_see_other_roles_via_timeline(self):
        """Player view timeline must not contain role information."""
        client, game_id = _make_client()
        resp = client.get(
            f"/games/{game_id}/timeline",
            params={"caller_id": "p01", "caller_role": "player_agent", "view_mode": "player_view"},
        )
        assert resp.status_code == 200

    def test_player_view_timeline_hides_private_night_events(self):
        from werewolf_agent.api.views import build_timeline

        state = GameState(
            game_id="timeline_leak",
            players={
                "w1": PlayerState(id="w1", role="werewolf"),
                "v1": PlayerState(id="v1", role="villager"),
                "witch": PlayerState(id="witch", role="witch"),
            },
            events=[
                GameEvent(type="wolf_kill_selected", payload={"night_number": 1, "target_id": "v1"}),
                GameEvent(type="wolf_no_kill_declared", payload={"night_number": 2, "reason": "pressure"}),
                GameEvent(type="seer_check", payload={"seer_id": "seer", "target_id": "w1", "result": "werewolf"}),
                GameEvent(type="player_died", payload={"player_id": "v1", "reason": "wolf_kill", "timing": "night"}),
            ],
        )

        villager = build_timeline(state, ViewMode.PLAYER_VIEW, viewer_id="v1")
        wolf = build_timeline(state, ViewMode.PLAYER_VIEW, viewer_id="w1")
        witch = build_timeline(state, ViewMode.PLAYER_VIEW, viewer_id="witch")

        assert [event.event_type for event in villager.events] == ["player_died"]
        assert villager.events[0].data == {"player_id": "v1"}
        assert "wolf_kill_selected" in {event.event_type for event in wolf.events}
        assert "wolf_kill_selected" not in {event.event_type for event in witch.events}

    def test_witch_private_state_includes_current_kill_target_only_when_selected(self):
        from werewolf_agent.api.views import build_private_state

        state = GameState(
            game_id="witch_target",
            night_number=1,
            players={"witch": PlayerState(id="witch", role="witch")},
            events=[GameEvent(type="wolf_kill_selected", payload={"night_number": 1, "target_id": "v1"})],
        )

        response = build_private_state(state, "witch", ViewMode.PLAYER_VIEW)

        assert response.player_info.current_wolf_kill_target_id == "v1"

    def test_cognitive_diff_hides_actual_roles_for_non_moderator(self):
        """Non-moderator must not see actual_role in cognitive diff."""
        # This tests the view function directly since API denies access
        from werewolf_agent.api.views import build_cognitive_diff
        from werewolf_agent.core.models import GameState, PlayerState

        state = GameState(
            game_id="test",
            players={
                "p1": PlayerState(id="p1", role="werewolf", alive=True),
                "p2": PlayerState(id="p2", role="seer", alive=True),
            },
        )
        # player_view should not show actual roles
        result = build_cognitive_diff(state, "p1", ViewMode.PLAYER_VIEW)
        for entry in result.entries:
            assert entry.actual_role is None
            assert entry.actual_faction is None

        # moderator_full should show actual roles
        result = build_cognitive_diff(state, "p1", ViewMode.MODERATOR_FULL)
        for entry in result.entries:
            assert entry.actual_role is not None


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:

    def test_unauthorized_access_generates_audit(self):
        client, game_id = _make_client()
        # Attempt unauthorized access
        resp = client.get(
            f"/games/{game_id}/replay",
            params={"caller_id": "p01", "caller_role": "player_agent", "view_mode": "moderator_full"},
        )
        assert resp.status_code == 403

        checker = client.app.state.checker
        denials = checker.denials()
        assert len(denials) >= 1
        denial = denials[-1]
        assert denial.caller_id == "p01"
        assert denial.granted is False

    def test_audit_event_structure(self):
        pc = PermissionChecker()
        with pytest.raises(PermissionDenied) as exc_info:
            pc.check(
                "p1", CallerRole.PLAYER_AGENT, ViewMode.MODERATOR_FULL,
                game_id="g1", endpoint="replay", game_active=True,
            )
        audit = exc_info.value.audit
        assert audit is not None
        assert audit.caller_id == "p1"
        assert audit.caller_role == CallerRole.PLAYER_AGENT
        assert audit.requested_view == ViewMode.MODERATOR_FULL
        assert audit.granted is False
        assert audit.game_id == "g1"
