# -*- coding: utf-8 -*-
"""
构建警长行动阶段的目标列表、指令和结果。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.sheriff_action_directives import living_non_sheriff_ids
    >>> living_non_sheriff_ids(...)
"""

from __future__ import annotations

from typing import Any, Sequence

from werewolf_agent.core.models import GameState


def living_non_sheriff_ids(gs: GameState, sheriff_id: str) -> list[str]:
    """列出存活且不是当前警长的玩家。"""
    return [
        pid for pid, player in gs.players.items()
        if player.alive and pid != sheriff_id
    ]


def build_sheriff_speech_order_directive(alive_players: Sequence[str]) -> dict[str, Any]:
    """构建警长选择白天发言顺序的策略指令。"""
    return {
        "choose_speech_order": (
            "你是警长，需要选择发言顺序。请选择第一个发言的玩家（你将最后一个发言进行归票）。"
            "在speech字段中说明你的选择理由。"
        ),
        "alive_players": list(alive_players),
    }


def build_sheriff_speech_order(
    *,
    first_speaker: str | None,
    alive_players: Sequence[str],
    sheriff_id: str,
) -> list[str] | None:
    """根据警长选择的首发言人组装完整发言顺序。"""
    if not first_speaker or first_speaker not in alive_players:
        return None
    remaining = [pid for pid in alive_players if pid != first_speaker]
    return [first_speaker] + remaining + [sheriff_id]


def build_sheriff_endorse_directive(alive_others: Sequence[str]) -> dict[str, Any]:
    """构建警长私人归票决策指令。"""
    return {
        "sheriff_endorse": (
            "你是警长。现在所有玩家已经发言完毕，即将开始放逐投票。"
            "作为警长，你需要归票——选择你认为应该被投票放逐的玩家。"
            "这是你的私人决策，你的内心理由不会让其他玩家看到。"
            "但你的归票目标会被法官公开宣布。"
        ),
        "legal_endorse_targets": list(alive_others),
    }


def build_sheriff_endorse_result(
    *,
    target: str | None,
    alive_others: Sequence[str],
    private_reason: str,
    action_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    """把警长归票行动转换为运行时返回结构。"""
    if target and target in alive_others:
        return {
            "endorse_target": target,
            "private_reason": private_reason,
            "action_trace": action_trace,
        }
    return {
        "endorse_target": "",
        "private_reason": "",
        "action_trace": None,
    }
