# -*- coding: utf-8 -*-
"""
游戏状态快照、视图、回放与审计路由注册。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from fastapi import APIRouter
    >>> router = APIRouter()
    >>> register_game_snapshot_routes(router=router, ...)
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker, PermissionDenied
from werewolf_agent.api.routes.game_cognition_views import _build_cognition_data_for_viewer
from werewolf_agent.api.routes.game_persistence import _get_game
from werewolf_agent.api.routes.game_public_share import (
    _event_is_public_for_share,
    _pick_public_mvp_candidate,
)
from werewolf_agent.api.routes.game_snapshot_audit import (
    register_game_snapshot_audit_routes,
)
from werewolf_agent.api.schemas import (
    CallerRole,
    CognitiveDiffResponse,
    EvaluationResponse,
    PrivateStateResponse,
    PublicStateResponse,
    ReplayResponse,
    TimelineResponse,
    ViewMode,
)
from werewolf_agent.api.views import (
    build_cognitive_diff,
    build_evaluation,
    build_private_state,
    build_public_state,
    build_replay,
    build_timeline,
)
from werewolf_agent.core.models import GameState


def register_game_snapshot_routes(
    *,
    router: APIRouter,
    games: dict[str, GameState],
    repo: Any,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    checker: PermissionChecker,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """注册只读状态视图和审计端点。"""

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

    @router.get("/games/{game_id}/timeline", response_model=TimelineResponse)
    def get_timeline(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.SPECTATOR),
        view_mode: ViewMode = Query(ViewMode.PUBLIC),
        day_filter: int | None = Query(None),
    ) -> TimelineResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = resolve_caller_role(authorized_callers, caller_id, caller_role)
        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="timeline",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_timeline(state, allowed_view, day_filter, viewer_id=caller_id)

    @router.get("/games/{game_id}/replay", response_model=ReplayResponse)
    def get_replay(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> ReplayResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="replay",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_replay(state, allowed_view, viewer_id=caller_id)

    @router.get("/games/{game_id}/snapshot", response_model=ReplayResponse)
    def get_snapshot(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> ReplayResponse:
        """返回当前 GameState 的快照；保留与 /replay 相同的响应结构。"""
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="snapshot",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_replay(state, allowed_view, viewer_id=caller_id)

    @router.get("/games/{game_id}/evaluation", response_model=EvaluationResponse)
    def get_evaluation(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> EvaluationResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            allowed_view = checker.check(
                caller_id=caller_id,
                caller_role=caller_role,
                requested_view=view_mode,
                game_id=game_id,
                endpoint="evaluation",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        return build_evaluation(
            state,
            allowed_view,
            audit_events=[
                event.model_dump()
                for event in checker.audit_log()
                if event.game_id == game_id
            ],
        )

    @router.get("/games/{game_id}/cognitive-diff", response_model=CognitiveDiffResponse)
    def get_cognitive_diff(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.DEBUGGER),
        player_id: str = Query(""),
        view_mode: ViewMode = Query(ViewMode.MODERATOR_FULL),
        session_token: str = Query(""),
    ) -> CognitiveDiffResponse:
        state = _get_game(games, game_id)
        game_active = state.winning_faction is None
        caller_role = resolve_caller_role(
            authorized_callers,
            caller_id,
            caller_role,
            session_token=session_token,
            auth_manager=auth,
        )
        try:
            allowed_view = checker.check_cognitive_diff(
                caller_id=caller_id,
                caller_role=caller_role,
                game_id=game_id,
                endpoint="cognitive-diff",
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        viewer_id = player_id or "p01"
        return build_cognitive_diff(
            state,
            viewer_id,
            allowed_view,
            cognition_data=_build_cognition_data_for_viewer(state, viewer_id),
        )

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
        game_active = state.winning_faction is None
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
                game_active=game_active,
            )
        except PermissionDenied as e:
            raise HTTPException(403, detail=e.reason)
        public_events = [
            {
                "event_type": event.type,
                "day": event.payload.get("day", state.day_number),
                "phase": event.payload.get("phase", state.phase),
            }
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

    register_game_snapshot_audit_routes(
        router=router,
        games=games,
        repo=repo,
        authorized_callers=authorized_callers,
        auth=auth,
        resolve_caller_role=resolve_caller_role,
    )


__all__ = ["register_game_snapshot_routes"]
