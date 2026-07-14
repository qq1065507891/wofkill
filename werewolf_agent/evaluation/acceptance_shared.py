# -*- coding: utf-8 -*-
"""
提供验收指标投影器共用的窄类型与玩家角色读取辅助函数。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

from typing import Any


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _game_player_roles(game: dict[str, Any]) -> dict[str, str] | None:
    players = game.get("players")
    if not isinstance(players, dict) or not players:
        return None
    roles: dict[str, str] = {}
    for player_id, player in players.items():
        role = player.get("role") if isinstance(player, dict) else None
        if not isinstance(player_id, str) or not isinstance(role, str) or not role:
            return None
        roles[player_id] = role
    return roles
