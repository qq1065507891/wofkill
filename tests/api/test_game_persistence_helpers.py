# -*- coding: utf-8 -*-
"""
游戏路由持久化、认知视图和公开分享 helper 的拆分兼容测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/api/test_game_persistence_helpers.py
"""

from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.api.routes import game_cognition_views
from werewolf_agent.api.routes import game_persistence
from werewolf_agent.api.routes import game_public_share
from werewolf_agent.api.routes import games
from werewolf_agent.api.schemas import CreateGameRequest
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.storage.memory_store import InMemoryGameRepository


class _Repo:
    def __init__(self) -> None:
        self.saved: list[GameState] = []

    def save_game(self, state: GameState) -> None:
        self.saved.append(state)


def test_helper_modules_are_reexported_from_games_facade() -> None:
    assert games._get_game is game_persistence._get_game
    assert games._persist is game_persistence._persist
    assert games._build_cognition_data_for_viewer is game_cognition_views._build_cognition_data_for_viewer
    assert games._build_locked_config_snapshot is game_cognition_views._build_locked_config_snapshot
    assert games._event_is_public_for_share is game_public_share._event_is_public_for_share
    assert games._pick_public_mvp_candidate is game_public_share._pick_public_mvp_candidate


def test_game_router_keeps_route_registration_surface() -> None:
    auth = AuthManager(AuthConfig(mode="local", secret_key="test-secret"))
    app = create_app(repository=InMemoryGameRepository(), auth_manager=auth)

    routes = {
        (",".join(sorted(route.methods or [])), route.path)
        for route in app.routes
        if route.path == "/" or route.path.startswith(("/auth", "/games"))
    }

    assert routes == {
        ("GET", "/"),
        ("POST", "/auth/login"),
        ("POST", "/games"),
        ("GET", "/games"),
        ("POST", "/games/{game_id}/start"),
        ("POST", "/games/{game_id}/step"),
        ("POST", "/games/{game_id}/pause"),
        ("POST", "/games/{game_id}/resume"),
        ("GET", "/games/{game_id}/public-state"),
        ("GET", "/games/{game_id}/players/{player_id}/private-state"),
        ("GET", "/games/{game_id}/timeline"),
        ("GET", "/games/{game_id}/replay"),
        ("GET", "/games/{game_id}/snapshot"),
        ("GET", "/games/{game_id}/evaluation"),
        ("GET", "/games/{game_id}/cognitive-diff"),
        ("GET", "/games/{game_id}/share-summary"),
        ("GET", "/games/{game_id}/rag-audit"),
        ("GET", "/games/{game_id}/world-model-audit"),
    }


def test_get_game_raises_404_for_missing_game() -> None:
    with pytest.raises(HTTPException) as exc_info:
        game_persistence._get_game({}, "g_missing")

    assert exc_info.value.status_code == 404


def test_persist_updates_memory_and_repository() -> None:
    repo = _Repo()
    games_dict: dict[str, GameState] = {}
    state = GameState(game_id="g_persist", phase="night")

    game_persistence._persist(state, games_dict, threading.Lock(), repo)

    assert games_dict["g_persist"] is state
    assert repo.saved == [state]


def test_locked_config_snapshot_rejects_path_traversal(tmp_path) -> None:
    req = CreateGameRequest(
        ruleset_id="../secret",
        profile_pack_id="default_12_ai_players",
        caller_id="mod1",
        caller_role="moderator",
    )

    with pytest.raises(HTTPException) as exc_info:
        game_cognition_views._build_locked_config_snapshot(req, tmp_path)

    assert exc_info.value.status_code == 400


def test_public_share_filters_private_events_and_picks_good_mvp() -> None:
    private_event = GameEvent(type="seer_check", payload={"visibility": "seer_private"})
    public_event = GameEvent(type="day_vote", payload={"visibility": "public"})
    state = GameState(
        game_id="g_share",
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
    )

    assert not game_public_share._event_is_public_for_share(private_event)
    assert game_public_share._event_is_public_for_share(public_event)
    assert game_public_share._pick_public_mvp_candidate(state) == "p02"
