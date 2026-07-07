# -*- coding: utf-8 -*-
"""
游戏启动、推进、暂停与恢复路由注册。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from fastapi import APIRouter
    >>> router = APIRouter()
    >>> register_game_command_routes(router=router, ...)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from werewolf_agent.api.auth import AuthManager
from werewolf_agent.api.permissions import PermissionChecker
from werewolf_agent.api.routes.game_persistence import _get_game, _persist
from werewolf_agent.api.schemas import (
    CallerRole,
    GameActionRequest,
    GameActionResponse,
)
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


def register_game_command_routes(
    *,
    router: APIRouter,
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
    enforce_moderator_only: Callable[
        [GameActionRequest, AuthManager, PermissionChecker, dict[str, CallerRole], str, str],
        None,
    ],
) -> None:
    """注册所有会改变游戏运行状态的命令端点。"""

    @router.post("/games/{game_id}/start", response_model=GameActionResponse)
    def start_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        # runner 是长生命周期资源，已存在时不能静默覆盖。
        with runners_lock:
            if game_id in runners:
                raise HTTPException(409, f"Game {game_id} is already running")
        enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "start")
        state = _get_game(games, game_id)
        if state.phase != "setup":
            raise HTTPException(400, "Game already started")
        seed = int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:4], "big") & 0x7FFFFFFF
        runner = GameRunner(
            GameRunnerConfig(
                ruleset_id=state.ruleset_id,
                seed=seed,
                use_agent_registry=os.environ.get("WEREWOLF_USE_LLM_AGENTS") == "1",
                model_config_path=os.environ.get("WEREWOLF_MODEL_CONFIG", "config/models.yaml"),
                repository=repo,
                rag_service=rag_service,
            ),
        )
        runner.reset_game_id(game_id)
        players = runner.engine.assign_roles(
            [f"p{i:02d}" for i in range(1, 13)],
            seed=seed,
        )
        players_data: dict[str, dict] = {}
        for pid, p in players.items():
            players_data[pid] = {"id": pid, "role": p.role}
        event = GameEvent(
            type="game_started",
            payload={
                "game_id": game_id,
                "players": players_data,
            },
        )
        state = replace(
            state,
            players=players,
            phase="night",
            events=state.events + [event],
        )
        runner._state = state
        with runners_lock:
            runners[game_id] = runner
        _persist(state, games, games_lock, repo)
        return GameActionResponse(
            game_id=game_id,
            action="start",
            success=True,
            message="Game started",
        )

    @router.post("/games/{game_id}/step", response_model=GameActionResponse)
    def step_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "step")
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Game is paused")
        with runners_lock:
            runner = runners.get(game_id)
        if runner is None:
            raise HTTPException(404, f"No runner for game {game_id}. Start the game first.")
        if runner.finished:
            raise HTTPException(400, "Game already finished")
        result = executor.try_step(game_id, runner)
        if result.status == "busy":
            raise HTTPException(409, result.message)
        if not result.success:
            raise HTTPException(500, result.message or "Step execution failed")
        with games_lock:
            games[game_id] = runner.state
        _persist(runner.state, games, games_lock, repo)
        return GameActionResponse(
            game_id=game_id,
            action="step",
            success=True,
            message=f"Step {runner.step_count}: phase={runner.state.phase}",
        )

    @router.post("/games/{game_id}/pause", response_model=GameActionResponse)
    def pause_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "pause")
        state = _get_game(games, game_id)
        if state.paused:
            raise HTTPException(400, "Already paused")
        lock = executor.lock_for(game_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Game is currently executing a step; retry")
        try:
            event = GameEvent(
                type="game_paused",
                payload={"game_id": game_id, "phase": state.phase},
            )
            state = replace(state, paused=True, events=state.events + [event])
            with runners_lock:
                if game_id in runners:
                    runners[game_id]._state = state
            _persist(state, games, games_lock, repo)
        finally:
            lock.release()
        return GameActionResponse(
            game_id=game_id,
            action="pause",
            success=True,
            message="Game paused",
        )

    @router.post("/games/{game_id}/resume", response_model=GameActionResponse)
    def resume_game(game_id: str, req: GameActionRequest) -> GameActionResponse:
        enforce_moderator_only(req, auth, checker, authorized_callers, game_id, "resume")
        state = _get_game(games, game_id)
        if not state.paused:
            raise HTTPException(400, "Not paused")
        lock = executor.lock_for(game_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Game is currently executing a step; retry")
        try:
            event = GameEvent(
                type="game_resumed",
                payload={"game_id": game_id, "phase": state.phase},
            )
            state = replace(state, paused=False, events=state.events + [event])
            with runners_lock:
                if game_id in runners:
                    runners[game_id]._state = state
            _persist(state, games, games_lock, repo)
        finally:
            lock.release()
        return GameActionResponse(
            game_id=game_id,
            action="resume",
            success=True,
            message="Game resumed",
        )


__all__ = ["register_game_command_routes"]
