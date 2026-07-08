# -*- coding: utf-8 -*-
"""
游戏快照审计端点注册，包括 RAG 注入审计和世界模型审计。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.api.routes.game_snapshot_audit import register_game_snapshot_audit_routes
    >>> register_game_snapshot_audit_routes(router=router, games=games, repo=repo, ...)
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.routes.game_persistence import _get_game
from werewolf_agent.api.schemas import CallerRole, ViewMode
from werewolf_agent.api.views import build_world_model_audit
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.world_model_audit import extract_world_model_audits_from_events


def register_game_snapshot_audit_routes(
    *,
    router: APIRouter,
    games: dict[str, GameState],
    repo: Any,
    authorized_callers: dict[str, CallerRole],
    auth: AuthManager,
    resolve_caller_role: Callable[..., CallerRole],
) -> None:
    """注册只读审计端点。"""

    @router.get("/games/{game_id}/rag-audit")
    def get_rag_audit(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        session_token: str = Query(""),
    ) -> dict:
        """返回该局所有 RAG 注入审计事件。"""
        state = _get_game(games, game_id)
        _require_audit_role(
            authorized_callers=authorized_callers,
            caller_id=caller_id,
            caller_role=caller_role,
            session_token=session_token,
            auth=auth,
            resolve_caller_role=resolve_caller_role,
            error_message="rag-audit requires moderator or debugger role",
        )
        audits = _load_rag_audit_payloads(game_id, state=state, repo=repo)
        return {
            "game_id": game_id,
            "rag_audits": audits,
            "audits": audits,
        }

    @router.get("/games/{game_id}/world-model-audit")
    def get_world_model_audit(
        game_id: str,
        caller_id: str = Query(""),
        caller_role: CallerRole = Query(CallerRole.MODERATOR),
        session_token: str = Query(""),
    ) -> dict:
        state = _get_game(games, game_id)
        _require_audit_role(
            authorized_callers=authorized_callers,
            caller_id=caller_id,
            caller_role=caller_role,
            session_token=session_token,
            auth=auth,
            resolve_caller_role=resolve_caller_role,
            error_message="world-model-audit requires moderator or debugger role",
        )
        events = _load_game_events(game_id, state=state, repo=repo)
        audits = extract_world_model_audits_from_events(list(events))
        return build_world_model_audit(
            state,
            ViewMode.MODERATOR_FULL,
            audit_events=audits,
        )


def _require_audit_role(
    *,
    authorized_callers: dict[str, CallerRole],
    caller_id: str,
    caller_role: CallerRole,
    session_token: str,
    auth: AuthManager,
    resolve_caller_role: Callable[..., CallerRole],
    error_message: str,
) -> None:
    resolved_role = resolve_caller_role(
        authorized_callers,
        caller_id,
        caller_role,
        session_token=session_token,
        auth_manager=auth,
    )
    if resolved_role not in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
        raise HTTPException(403, error_message)


def _load_rag_audit_payloads(
    game_id: str,
    *,
    state: GameState,
    repo: Any,
) -> list[dict]:
    audits: list[dict] = []
    for event in _load_game_events(game_id, state=state, repo=repo):
        event_type = _event_type(event)
        if event_type != "rag_injection_audit":
            continue
        payload = _event_payload(event)
        audits.append(payload if isinstance(payload, dict) else {})
    return audits


def _load_game_events(
    game_id: str,
    *,
    state: GameState,
    repo: Any,
) -> list[GameEvent | dict[str, Any]]:
    if repo is not None and hasattr(repo, "load_events"):
        try:
            return list(repo.load_events(game_id))
        except Exception:
            return list(state.events)
    return list(state.events)


def _event_type(event: GameEvent | dict[str, Any]) -> str | None:
    return getattr(event, "type", None) or (
        event.get("type") if isinstance(event, dict) else None
    )


def _event_payload(event: GameEvent | dict[str, Any]) -> Any:
    return getattr(event, "payload", None) or (
        event.get("payload") if isinstance(event, dict) else {}
    )


__all__ = ["register_game_snapshot_audit_routes"]
