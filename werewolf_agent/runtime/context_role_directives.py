# -*- coding: utf-8 -*-
"""
为 AgentContext 注入角色相关的策略指令和临时可见字段。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> apply_role_strategy_context(
    ...     visible={}, strategy_directive={}, gs=None, player_id="p01"
    ... )
    ({}, {})
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.strategy import build_witch_pressure_targets


def apply_role_strategy_context(
    *,
    visible: dict[str, Any],
    strategy_directive: dict[str, Any],
    gs: GameState | None,
    player_id: str,
    wolf_kill_target_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """根据玩家角色补充 prompt 侧策略指令和必要私有可见字段。"""
    if gs is None:
        return visible, strategy_directive

    player = gs.players.get(player_id)
    if player is None:
        return visible, strategy_directive

    if player.role == "witch" and not gs.poison_used and gs.phase == "day":
        alive = sum(1 for state in gs.players.values() if state.alive)
        if alive <= 8:
            strategy_directive["witch_poison_deterrent"] = (
                "你的毒药还未使用。如果场上有人持续踩你、试图把你放逐出局，"
                "你可以在发言中暗示自己有底牌——'我手里还有东西没用，不要太冲动'。"
                "狼人听到这种暗示可能会退缩。但不要明报身份。"
            )

    if player.role == "witch" and wolf_kill_target_id:
        visible["wolf_kill_target"] = wolf_kill_target_id
    if player.role == "witch" and not gs.poison_used:
        pressure_targets = build_witch_pressure_targets(gs)
        if pressure_targets:
            visible["poison_pressure_targets"] = pressure_targets

    if player.role == "hybrid" and gs.hybrid_master_id:
        master = gs.players.get(gs.hybrid_master_id)
        if master and not master.alive:
            strategy_directive["hybrid_master_dead"] = (
                f"你的主人{gs.hybrid_master_id}已死亡。"
                "你的胜利绑定仍按主人的原始阵营结算，但你仍不知道主人的阵营。"
                "继续根据主人的公开行为和场上信息独立判断。"
            )

    return visible, strategy_directive


__all__ = [
    "apply_role_strategy_context",
    "build_witch_pressure_targets",
]
