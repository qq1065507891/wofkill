# -*- coding: utf-8 -*-
"""
提供验收指标投影器共用的窄类型、语义审计结构与玩家角色读取辅助函数。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-24
"""

from __future__ import annotations

from typing import Any, Mapping


_REPAIR_FAILURE_CODES = frozenset({
    "speech_quality",
    "semantic_claim_retention",
})


def _has_valid_repair_failure_history(
    row: Mapping[str, Any],
    *,
    container_type: type,
) -> bool:
    """兼容字段缺失，并严格校验容器、元素类型和稳定码集合。"""
    if "repair_failure_history" not in row:
        return True
    history = row["repair_failure_history"]
    return type(history) is container_type and all(
        type(code) is str and code in _REPAIR_FAILURE_CODES
        for code in history
    )


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _game_player_roles(game: dict[str, Any]) -> dict[str, str] | None:
    players = game.get("players")
    if not isinstance(players, Mapping) or not players:
        return None
    roles: dict[str, str] = {}
    for player_id, player in players.items():
        role = player.get("role") if isinstance(player, Mapping) else None
        if not isinstance(player_id, str) or not isinstance(role, str) or not role:
            return None
        roles[player_id] = role
    return roles
