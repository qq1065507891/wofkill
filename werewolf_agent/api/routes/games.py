# -*- coding: utf-8 -*-
"""
游戏 API 路由工厂 facade，负责鉴权入口、仪表盘和路由组装配。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-07

使用示例:
    >>> router = create_game_router(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker
from werewolf_agent.api.routes.game_auth import (
    _enforce_create_game_auth_with_resolver,
    _enforce_moderator_only_with_resolver,
    _resolve_caller_role,
)
from werewolf_agent.api.routes.game_cognition_views import (
    _build_cognition_data_for_viewer,
    _build_locked_config_snapshot,
)
from werewolf_agent.api.routes.game_commands import register_game_command_routes
from werewolf_agent.api.routes.game_lifecycle import register_game_lifecycle_routes
from werewolf_agent.api.routes.game_persistence import _get_game, _persist
from werewolf_agent.api.routes.game_public_share import (
    _event_is_public_for_share,
    _pick_public_mvp_candidate,
)
from werewolf_agent.api.routes.game_snapshots import register_game_snapshot_routes
from werewolf_agent.api.schemas import (
    CallerRole,
    CreateGameRequest,
    GameActionRequest,
)
from werewolf_agent.core.models import GameState


def _enforce_moderator_only(
    req: GameActionRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
    game_id: str,
    endpoint: str,
) -> None:
    """兼容旧入口：允许调用方 patch games._resolve_caller_role。"""
    _enforce_moderator_only_with_resolver(
        req,
        auth_manager,
        checker,
        authorized_callers,
        game_id,
        endpoint,
        resolve_caller_role=_resolve_caller_role,
    )


def _enforce_create_game_auth(
    req: CreateGameRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
) -> None:
    """兼容旧入口：允许调用方 patch games._resolve_caller_role。"""
    _enforce_create_game_auth_with_resolver(
        req,
        auth_manager,
        checker,
        authorized_callers,
        resolve_caller_role=_resolve_caller_role,
    )


def _resolve_current_caller_role(*args: Any, **kwargs: Any) -> CallerRole:
    """路由闭包通过 facade 当前全局变量解析身份，保留 monkeypatch 兼容。"""
    return _resolve_caller_role(*args, **kwargs)


def create_game_router(
    *,
    games: dict[str, GameState],
    games_lock: Any,
    runners: dict[str, Any],
    runners_lock: Any,
    repo: Any,
    rag_service: Any,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    executor: Any,
    project_root: Path,
) -> APIRouter:
    router = APIRouter()
    dashboard_path = Path(__file__).parent.parent.parent / "ui" / "static" / "dashboard.html"

    @router.post("/auth/login")
    def auth_login(
        caller_id: str = Query(...),
        role: str = Query(...),
    ) -> dict:
        try:
            token = auth.create_session(caller_id, role)
        except PermissionError as e:
            raise HTTPException(403, detail=str(e))
        return {"token": token, "caller_id": caller_id, "role": role}

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))

    register_game_lifecycle_routes(
        router=router,
        games=games,
        games_lock=games_lock,
        repo=repo,
        authorized_callers=authorized_callers,
        auth=auth,
        checker=checker,
        project_root=project_root,
        enforce_create_game_auth=_enforce_create_game_auth,
        resolve_caller_role=_resolve_current_caller_role,
    )
    register_game_command_routes(
        router=router,
        games=games,
        games_lock=games_lock,
        runners=runners,
        runners_lock=runners_lock,
        repo=repo,
        rag_service=rag_service,
        authorized_callers=authorized_callers,
        auth=auth,
        checker=checker,
        executor=executor,
        enforce_moderator_only=_enforce_moderator_only,
    )
    register_game_snapshot_routes(
        router=router,
        games=games,
        repo=repo,
        authorized_callers=authorized_callers,
        auth=auth,
        checker=checker,
        resolve_caller_role=_resolve_current_caller_role,
    )
    return router


__all__ = [
    "_build_cognition_data_for_viewer",
    "_build_locked_config_snapshot",
    "_enforce_create_game_auth",
    "_enforce_moderator_only",
    "_event_is_public_for_share",
    "_get_game",
    "_persist",
    "_pick_public_mvp_candidate",
    "_resolve_caller_role",
    "create_game_router",
]
