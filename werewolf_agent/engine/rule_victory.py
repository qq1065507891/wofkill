# -*- coding: utf-8 -*-
"""
RuleEngine 的胜利条件判定 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_victory import check_victory
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState, VictoryResult


def check_victory(raw: dict[str, Any], state: GameState) -> VictoryResult:
    players = state.players
    wolves_alive = any(p.alive and p.role == "werewolf" for p in players.values())
    if not wolves_alive:
        return VictoryResult(winner="good", reason="all_werewolves_out")

    villagers_alive = [pid for pid, p in players.items() if p.alive and p.role == "villager"]
    god_roles = {
        role
        for role, cfg in raw.get("roles", {}).items()
        if cfg.get("category") == "god"
    } or {"seer", "witch", "hunter", "idiot"}
    gods_alive = [
        pid for pid, player in players.items()
        if player.alive and player.role in god_roles
    ]

    if not gods_alive:
        return VictoryResult(winner="werewolf", reason="slaughter_gods")

    if not villagers_alive:
        hybrid = next((p for p in players.values() if p.role == "hybrid"), None)
        master_faction = state.hybrid_master_faction
        if master_faction == "good":
            if hybrid and not hybrid.alive:
                return VictoryResult(winner="werewolf", reason="slaughter_villagers")
        elif master_faction is not None:
            return VictoryResult(winner="werewolf", reason="slaughter_villagers")

    return VictoryResult(winner=None, reason=None)


__all__ = ["check_victory"]
