# -*- coding: utf-8 -*-
"""
构建警徽去向决策阶段的策略指令和结果。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.badge_decision_directives import build_badge_decision_directive
    >>> build_badge_decision_directive("seer", ["p02"])
"""

from __future__ import annotations

from typing import Any, Sequence

from werewolf_agent.agents.schemas import ActionType


def build_badge_role_hint(player_role: str) -> str:
    """根据离场警长身份构建警徽去向提示。"""
    if player_role == "werewolf":
        return (
            "你是狼人警长，移交必须为狼队利益服务：\n"
            "   - 移交给狼队友：让狼队继续控制警徽和归票权。\n"
            "   - 移交给被狼队深度迷惑的好人：利用他替狼队带节奏。\n"
            "   - 撕毁：如果移交任何人都对狼队不利，撕掉不让好人拿到归票权。"
        )
    if player_role == "seer":
        return (
            "你是预言家警长：移交给你验过的金水（被你验出好人的玩家）。"
            "让金水拿到警徽，继续传递你的验人信息。"
        )
    return (
        "你是好人警长：移交给你最信任的明好人，"
        "确保警徽不落入狼人手中。如果场上没有明确的明好人，可以撕毁。"
    )


def build_badge_decision_directive(
    player_role: str,
    alive_players: Sequence[str],
) -> dict[str, Any]:
    """构建警徽移交或撕毁决策指令。"""
    role_hint = build_badge_role_hint(player_role)
    return {
        "badge_decision": (
            "你是即将离场的警长，必须决定警徽去向：\n"
            "1) 移交（BADGE_TRANSFER）：选择一名存活玩家作为新警长。\n"
            f"   {role_hint}\n"
            "2) 撕毁（BADGE_TEAR）：撕毁警徽，本局不再有警长。\n"
            "请根据你的身份和阵营做出最有利的决定。"
        ),
        "alive_players": list(alive_players),
    }


def build_badge_decision_result(
    *,
    action_type: ActionType,
    target_id: str | None,
    action_trace: dict[str, Any],
) -> dict[str, Any]:
    """把警徽行动转换为运行时返回结构。"""
    if action_type == ActionType.BADGE_TRANSFER and target_id:
        return {
            "badge_decision": "transfer",
            "badge_target_id": target_id,
            "action_trace": action_trace,
        }
    if action_type == ActionType.BADGE_TEAR:
        return {
            "badge_decision": "tear",
            "badge_target_id": None,
            "action_trace": action_trace,
        }
    raise ValueError("badge decision requires BADGE_TRANSFER or BADGE_TEAR")
