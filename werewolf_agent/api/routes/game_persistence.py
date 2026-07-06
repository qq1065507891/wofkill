# -*- coding: utf-8 -*-
"""
游戏 API 路由的内存状态读取与持久化 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.api.routes.game_persistence import _get_game
    >>> _get_game({"g1": state}, "g1")
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from werewolf_agent.core.models import GameState


def _get_game(games: dict[str, GameState], game_id: str) -> GameState:
    state = games.get(game_id)
    if state is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return state


def _persist(state: GameState, games: dict, lock: Any, repo: Any) -> None:
    with lock:
        games[state.game_id] = state
    if repo is not None:
        repo.save_game(state)


__all__ = [
    "_get_game",
    "_persist",
]
