# -*- coding: utf-8 -*-
"""
游戏创建与列表路由注册。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from fastapi import APIRouter
    >>> router = APIRouter()
    >>> register_game_lifecycle_routes(router=router, ...)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker
from werewolf_agent.api.routes.game_cognition_views import _build_locked_config_snapshot
from werewolf_agent.api.schemas import (
    CallerRole,
    CreateGameRequest,
    GameCreateResponse,
    GameInfo,
)
from werewolf_agent.core.models import GameEvent, GameState


def register_game_lifecycle_routes(
    *,
    router: APIRouter,
    games: dict[str, GameState],
    games_lock: Any,
    repo: Any,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    project_root: Path,
    enforce_create_game_auth: Callable[
        [CreateGameRequest, AuthManager, PermissionChecker, dict[str, CallerRole]],
        None,
    ],
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """注册创建游戏与游戏列表端点。"""

    @router.post("/games", response_model=GameCreateResponse)
    def create_game(req: CreateGameRequest) -> GameCreateResponse:
        enforce_create_game_auth(req, auth, checker, authorized_callers)
        if req.experience_mode == "human_seat" and (
            req.human_seat is None or req.human_seat < 1 or req.human_seat > 12
        ):
            raise HTTPException(
                400,
                "human_seat must be between 1 and 12 when experience_mode is human_seat",
            )
        game_id = f"game_{req.seed}" if req.seed is not None else str(uuid.uuid4())[:8]
        game_id = f"g_{game_id}" if not game_id.startswith("g_") else game_id
        config_snapshot = _build_locked_config_snapshot(req, project_root)
        state = GameState(
            game_id=game_id,
            ruleset_id=req.ruleset_id,
            phase="setup",
            events=[
                GameEvent(
                    type="config_snapshot_locked",
                    payload={"config_snapshot": config_snapshot},
                ),
            ],
        )
        with games_lock:
            games[game_id] = state
        if repo is not None:
            repo.save_game(state)
            repo.save_config_snapshot(game_id, config_snapshot)
        return GameCreateResponse(
            game=GameInfo(
                game_id=game_id,
                ruleset_id=req.ruleset_id,
                status="created",
                player_count=req.player_count,
                experience_mode=req.experience_mode,
                human_seat=req.human_seat,
                profile_pack_id=req.profile_pack_id,
                share_code=req.share_code,
            ),
            message="Game created",
        )

    @router.get("/games")
    async def list_games(
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        session_token: str = Query(""),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict:
        # 列表接口只允许控制台角色访问，避免匿名枚举游戏 id。
        if not caller_id and not session_token:
            raise HTTPException(
                403,
                "list_games requires caller_id or session_token",
            )
        resolved_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            raise HTTPException(
                403,
                "list_games requires moderator or debugger role",
            )
        with games_lock:
            all_game_ids = list(games.keys())
        page = all_game_ids[offset:offset + limit]
        return {"game_ids": page, "total": len(all_game_ids)}


__all__ = ["register_game_lifecycle_routes"]
