"""UI tests: observer dashboard static files and API integration.

Covers:
1. Dashboard HTML is served at root /
2. Dashboard contains required sections (game list, public state, timeline, etc.)
3. Dashboard enforces permission modes in UI (no moderator content in public mode)
4. API endpoints return data that the dashboard can consume
5. Moderator/debug view shows private audit data
6. Cognitive diff endpoint accessible
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _create_and_start_game(client: TestClient) -> str:
    resp = client.post("/games", json={
        "player_count": 12,
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "seed": 99,
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    assert resp.status_code == 200
    game_id = resp.json()["game"]["game_id"]
    resp = client.post(f"/games/{game_id}/start", json={
        "caller_id": "mod1",
        "caller_role": "moderator",
    })
    assert resp.status_code == 200
    return game_id


# ---------------------------------------------------------------------------
# 1. Dashboard HTML served
# ---------------------------------------------------------------------------


class TestDashboardServed:
    def test_root_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_dashboard_has_title(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text
        assert "狼人杀" in html or "Werewolf Agent" in html or "werewolf" in html.lower()


# ---------------------------------------------------------------------------
# 2. Required sections present
# ---------------------------------------------------------------------------


class TestDashboardSections:
    def test_has_game_list_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "game" in html

    def test_has_timeline_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "timeline" in html

    def test_has_player_status_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "player" in html

    def test_has_pause_resume_controls(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "pause" in html or "resume" in html

    def test_has_death_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "death" in html

    def test_has_vote_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "vote" in html

    def test_has_cognitive_diff_section(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "cognitive" in html

    def test_has_moderator_view_toggle(self, client: TestClient) -> None:
        resp = client.get("/")
        html = resp.text.lower()
        assert "moderator" in html or "debug" in html


# ---------------------------------------------------------------------------
# 3. Permission enforcement in UI data
# ---------------------------------------------------------------------------


class TestDashboardPermissions:
    def test_public_timeline_no_private_events(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/timeline", params={
            "view_mode": "public",
        })
        data = resp.json()
        event_types = [e.get("event_type", "") for e in data.get("events", [])]
        forbidden = {"seer_check", "wolf_discussion", "witch_antidote_used", "hybrid_master_chosen"}
        for ft in forbidden:
            assert ft not in event_types, f"Public timeline must not contain {ft}"

    def test_public_state_no_roles(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/public-state")
        data = resp.json()
        for p in data.get("players", []):
            role = p.get("revealed_role")
            assert role is None or role == "idiot", "Public state must not reveal roles"

    def test_player_cannot_see_others_private(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(
            f"/games/{game_id}/players/p02/private-state",
            params={"caller_id": "p01", "caller_role": "player_agent"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 4. API endpoints return dashboard-consumable data
# ---------------------------------------------------------------------------


class TestDashboardAPI:
    def test_list_games_returns_dict(self, client: TestClient) -> None:
        resp = client.get(
            "/games", params={"caller_id": "mod1", "caller_role": "moderator"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict) and "game_ids" in resp.json()

    def test_game_list_after_create(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(
            "/games", params={"caller_id": "mod1", "caller_role": "moderator"}
        )
        games = resp.json()
        assert len(games["game_ids"]) >= 1
        assert game_id in games["game_ids"]

    def test_public_state_structure(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/public-state")
        data = resp.json()
        assert "phase" in data
        assert "players" in data
        assert "game_id" in data

    def test_timeline_structure(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/timeline", params={
            "view_mode": "public",
        })
        data = resp.json()
        assert "events" in data

    def test_moderator_replay_shows_roles(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/replay", params={
            "caller_id": "mod1",
            "caller_role": "moderator",
            "view_mode": "moderator_full",
        })
        assert resp.status_code == 200
        data = resp.json()
        snaps = data.get("snapshots", [])
        if snaps:
            mod = snaps[0].get("moderator_full")
            if mod:
                assert "all_roles" in mod


# ---------------------------------------------------------------------------
# 5. Moderator/debug views show private data
# ---------------------------------------------------------------------------


class TestModeratorViews:
    def test_moderator_can_access_private_state(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(
            f"/games/{game_id}/players/p01/private-state",
            params={
                "caller_id": "mod1",
                "caller_role": "moderator",
                "view_mode": "moderator_full",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("player_info", {}).get("role") is not None

    def test_moderator_evaluation_shows_data(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/evaluation", params={
            "caller_id": "mod1",
            "caller_role": "moderator",
            "view_mode": "moderator_full",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. Cognitive diff endpoint
# ---------------------------------------------------------------------------


class TestCognitiveDiffEndpoint:
    def test_cognitive_diff_accessible(self, client: TestClient) -> None:
        game_id = _create_and_start_game(client)
        resp = client.get(f"/games/{game_id}/cognitive-diff", params={
            "caller_id": "dbg1",
            "caller_role": "debugger",
            "player_id": "p01",
            "view_mode": "moderator_full",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data

    def test_cognitive_diff_public_hides_actual_roles(self, client: TestClient) -> None:
        """Non-moderator view must not show actual roles in cognitive diff."""
        game_id = _create_and_start_game(client)
        # public view should be denied for cognitive-diff (only debugger/moderator)
        resp = client.get(f"/games/{game_id}/cognitive-diff", params={
            "caller_id": "",
            "caller_role": "spectator",
            "player_id": "p01",
            "view_mode": "public",
        })
        # Spectator should be denied (403) or get filtered data
        assert resp.status_code in (200, 403)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.get("entries", []):
                assert entry.get("actual_role") is None
