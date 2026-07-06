# -*- coding: utf-8 -*-
"""
游戏 API 路由的调用者身份解析与控制端点鉴权 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.api.routes.game_auth import _resolve_caller_role
    >>> _resolve_caller_role({}, "", CallerRole.SPECTATOR)
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from collections.abc import Callable

from werewolf_agent.api.schemas import CallerRole, CreateGameRequest, GameActionRequest, ViewMode


logger = logging.getLogger("werewolf_agent.api.routes.games")


def _resolve_caller_role(
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    requested_role: CallerRole,
    session_token: str = "",
    auth_manager: AuthManager | None = None,
) -> CallerRole:
    if session_token and auth_manager is not None:
        validated_role = auth_manager.validate_session(session_token)
        if validated_role is not None:
            try:
                return CallerRole(validated_role)
            except ValueError:
                pass
        raise HTTPException(403, "Invalid or expired session token")
    if requested_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        if caller_id and authorized_callers.get(caller_id) == requested_role:
            # 仅对提权的旧查询参数鉴权路径记录安全警告。
            logger.warning(
                "Legacy query-param auth for elevated role without "
                "session_token: caller_id=%s, caller_role=%s — no "
                "cryptographic verification performed",
                caller_id,
                requested_role.value,
            )
            return requested_role
        raise HTTPException(403, "Elevated caller role is not authorized")
    if (
        requested_role == CallerRole.PLAYER_AGENT
        and auth_manager is not None
        and auth_manager.config.mode != "local"
    ):
        raise HTTPException(
            403,
            "Player agent role requires session_token outside local auth mode",
        )
    return requested_role


def _enforce_moderator_only(
    req: GameActionRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
    game_id: str,
    endpoint: str,
) -> None:
    """限制游戏控制端点只能由 MODERATOR/DEBUGGER 调用。"""
    _enforce_moderator_only_with_resolver(
        req,
        auth_manager,
        checker,
        authorized_callers,
        game_id,
        endpoint,
        resolve_caller_role=_resolve_caller_role,
    )


def _enforce_moderator_only_with_resolver(
    req: GameActionRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
    game_id: str,
    endpoint: str,
    *,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """使用显式 resolver 执行游戏控制端点鉴权。"""
    caller_role = resolve_caller_role(
        authorized_callers,
        req.caller_id,
        req.caller_role,
        session_token=req.session_token,
        auth_manager=auth_manager,
    )
    if caller_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        try:
            checker.check(
                caller_id=req.caller_id,
                caller_role=caller_role,
                requested_view=ViewMode.MODERATOR_FULL,
                game_id=game_id,
                endpoint=endpoint,
                game_active=True,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        raise HTTPException(403, "Game control endpoints require moderator or debugger role")
    if endpoint == "start" and not req.caller_id:
        raise HTTPException(403, "start_game requires a non-empty caller_id")


def _enforce_create_game_auth(
    req: CreateGameRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
) -> None:
    """创建游戏需要非空 caller_id 和 MODERATOR 角色。"""
    _enforce_create_game_auth_with_resolver(
        req,
        auth_manager,
        checker,
        authorized_callers,
        resolve_caller_role=_resolve_caller_role,
    )


def _enforce_create_game_auth_with_resolver(
    req: CreateGameRequest,
    auth_manager: AuthManager,
    checker: PermissionChecker,
    authorized_callers: dict[str, CallerRole],
    *,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """使用显式 resolver 执行创建游戏鉴权。"""
    if not req.caller_id:
        raise HTTPException(403, "create_game requires a non-empty caller_id")
    caller_role = resolve_caller_role(
        authorized_callers,
        req.caller_id,
        req.caller_role,
        session_token=req.session_token,
        auth_manager=auth_manager,
    )
    if caller_role != CallerRole.MODERATOR:
        try:
            checker.check(
                caller_id=req.caller_id,
                caller_role=caller_role,
                requested_view=ViewMode.MODERATOR_FULL,
                endpoint="create-game",
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        raise HTTPException(403, "create_game requires moderator role")


__all__ = [
    "_enforce_create_game_auth",
    "_enforce_create_game_auth_with_resolver",
    "_enforce_moderator_only",
    "_enforce_moderator_only_with_resolver",
    "_resolve_caller_role",
]
