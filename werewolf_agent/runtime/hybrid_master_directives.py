# -*- coding: utf-8 -*-
"""
构建混血儿首夜选择主人阶段的候选、指令和结果。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.hybrid_master_directives import build_hybrid_master_choice_directive
    >>> build_hybrid_master_choice_directive({})
"""

from __future__ import annotations

from typing import Any, Sequence

from werewolf_agent.agents.schemas import ActionType
from werewolf_agent.core.models import GameState


def build_hybrid_master_candidates(gs: GameState, hybrid_id: str) -> list[str]:
    """构建混血儿可选择的主人候选列表。"""
    return [
        pid for pid, player in gs.players.items()
        if player.alive and pid != hybrid_id
    ]


def build_hybrid_master_choice_directive(master_assessment: dict[str, Any]) -> dict[str, Any]:
    """构建混血儿首夜选择主人指令。"""
    return {
        "hybrid_master_choice": (
            "你是混血儿，N1 / 首夜需要选择一名玩家作为你的主人。"
            "你不知道主人的身份和阵营，但你将跟随主人的原始阵营获胜。"
            "如果主人是好人阵营，你跟好人赢；如果主人是狼人阵营，你跟狼人赢。"
            "选择后不能更改。speech字段留空。"
        ),
        "master_assessment": master_assessment,
    }


def choose_hybrid_master_target(
    *,
    action_type: ActionType,
    target_id: str | None,
    candidates: Sequence[str],
) -> str | None:
    """根据 agent 行动和候选列表确定最终主人目标。"""
    master_target_id = target_id if action_type == ActionType.CHOOSE_MASTER else None
    if master_target_id is None and candidates:
        master_target_id = candidates[0]
    return master_target_id
