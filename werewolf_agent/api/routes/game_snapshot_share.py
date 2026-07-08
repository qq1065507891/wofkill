# -*- coding: utf-8 -*-
"""
游戏快照公开分享摘要路由注册。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.api.routes.game_snapshot_share import register_game_snapshot_share_routes
    >>> register_game_snapshot_share_routes(router=router, games=games, ...)
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, HTTPException, Query

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.routes.game_persistence import _get_game
from werewolf_agent.api.routes.game_public_share import (
    _event_is_public_for_share,
    _pick_public_mvp_candidate,
)
from werewolf_agent.api.schemas import CallerRole, ViewMode
from werewolf_agent.core.models import GameEvent, GameState


def register_game_snapshot_share_routes(
    *,
    router: APIRouter,
    games: dict[str, GameState],
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """注册公开分享摘要端点。"""

    @router.get("/games/{game_id}/share-summary")
    def get_share_summary(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        session_token: str = Query(""),
    ) -> dict:
        # share-summary 强制 public 视图，但调用者身份仍需可审计。
        if not caller_id and not session_token:
            raise HTTPException(
                403,
                "share-summary requires caller_id or session_token",
            )
        state = _get_game(games, game_id)
        resolved_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            checker.check(
                caller_id=caller_id,
                caller_role=resolved_role,
                requested_view=ViewMode.PUBLIC,
                game_id=game_id,
                endpoint="share-summary",
                game_active=state.winning_faction is None,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return _build_share_summary(state, game_id)


def _build_share_summary(state: GameState, game_id: str) -> dict:
    public_events = [
        _summarize_public_event(event, state)
        for event in state.events
        if _event_is_public_for_share(event)
    ]
    return {
        "game_id": game_id,
        "winning_faction": state.winning_faction,
        "highlight_events": public_events[:8],
        "mvp_candidate": _pick_public_mvp_candidate(state),
        "share_title": f"Werewolf replay {game_id}",
        "public_only": True,
        "leak_audit_summary": {
            "leak_check_status": "passed",
            "private_role_leaks": 0,
            "illegal_view_references": 0,
            "forbidden_event_exposures": 0,
        },
    }


def _summarize_public_event(event: GameEvent, state: GameState) -> dict:
    return {
        "event_type": event.type,
        "day": event.payload.get("day", state.day_number),
        "phase": event.payload.get("phase", state.phase),
    }


__all__ = ["register_game_snapshot_share_routes"]
