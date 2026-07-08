# -*- coding: utf-8 -*-
"""
游戏快照公开与私有状态视图路由注册。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.api.routes.game_snapshot_state import register_game_snapshot_state_routes
    >>> register_game_snapshot_state_routes(router=router, games=games, ...)
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, HTTPException, Query

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.routes.game_persistence import _get_game
from werewolf_agent.api.schemas import (
    CallerRole,
    PrivateStateResponse,
    PublicStateResponse,
    ViewMode,
)
from werewolf_agent.api.views import build_private_state, build_public_state
from werewolf_agent.core.models import GameState


def register_game_snapshot_state_routes(
    *,
    router: APIRouter,
    games: dict[str, GameState],
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """注册公开状态和玩家私有状态端点。"""

    @router.get("/games/{game_id}/public-state", response_model=PublicStateResponse)
    def get_public_state(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
    ) -> PublicStateResponse:
        state = _get_game(games, game_id)
        # public-state 允许匿名读取，但仍进入权限检查审计。
        resolved_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
        )
        try:
            checker.check(
                caller_id=caller_id,
                caller_role=resolved_role,
                requested_view=ViewMode.PUBLIC,
                game_id=game_id,
                endpoint="public-state",
                game_active=state.winning_faction is None,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_public_state(state)

    @router.get(
        "/games/{game_id}/players/{player_id}/private-state",
        response_model=PrivateStateResponse,
    )
    def get_private_state(
        game_id: str,
        player_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.PLAYER_AGENT),
        view_mode: ViewMode = Query(ViewMode.PLAYER_VIEW),
        session_token: str = Query(""),
    ) -> PrivateStateResponse:
        state = _get_game(games, game_id)
        if player_id not in state.players:
            raise HTTPException(404, f"Player {player_id} not found in game {game_id}")
        caller_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            allowed_view = checker.check_private_state(
                caller_id=caller_id,
                caller_role=caller_role,
                target_player_id=player_id,
                game_id=game_id,
                endpoint="private-state",
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_private_state(state, player_id, allowed_view)


__all__ = ["register_game_snapshot_state_routes"]
