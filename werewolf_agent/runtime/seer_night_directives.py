# -*- coding: utf-8 -*-
"""
构建预言家夜晚验人阶段的目标过滤和策略指令。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.seer_night_directives import build_seer_night_guidance
    >>> build_seer_night_guidance(1)
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.timeline import phase_label


def collect_seer_checked_target_ids(gs: GameState) -> set[str]:
    """收集预言家已经查验过的目标。"""
    checked_ids: set[str] = set()
    for event in gs.events:
        if event.type == "seer_check":
            checked_ids.add(event.payload["target_id"])
    return checked_ids


def build_seer_legal_targets(
    gs: GameState,
    *,
    seer_id: str,
    counterclaiming_seers: set[str],
) -> list[str]:
    """构建预言家当前可查验目标列表。"""
    legal_targets = [
        pid for pid, player in gs.players.items()
        if player.alive and pid != seer_id
    ]
    if counterclaiming_seers:
        legal_targets = [
            pid for pid in legal_targets
            if pid not in counterclaiming_seers
        ]

    checked_ids = collect_seer_checked_target_ids(gs)
    if checked_ids:
        legal_targets = [
            pid for pid in legal_targets
            if pid not in checked_ids
        ]
    return legal_targets


def build_badge_flow_next_targets(
    gs: GameState,
    *,
    seer_id: str,
    legal_targets: Sequence[str],
) -> list[str] | None:
    """从预言家警上发言中提取当前仍合法的警徽流查验目标。"""
    legal_target_set = set(legal_targets)
    for event in gs.events:
        if event.type == "sheriff_speech" and event.payload.get("speaker") == seer_id:
            text = event.payload.get("text", "")
            mentioned = re.findall(r"p\d+", text)
            if mentioned:
                return [pid for pid in mentioned if pid in legal_target_set]
            break
    return None


def build_seer_night_guidance(night_number: int) -> str:
    """根据夜晚轮次构建预言家验人策略提示。"""
    night_label = phase_label("night", night_number)
    if night_number == 1:
        return (
            f"{night_label} 验人策略：选择你最怀疑的人，或者按照你上警时承诺的警徽流首夜验人对象。"
            "如果上警时没有明确指定，优先验发言最少、最不透明的人。"
            "不要查验对跳预言家的玩家；对跳位应通过白天发言、票型和放逐解决，夜晚验人用于开新视角。"
        )
    return (
        f"{night_label} 验人策略：根据白天讨论中你最怀疑的人选择查验目标。"
        "优先验：1) 发言前后矛盾的人；2) 站边不明确的人；3) 被多人怀疑但你不确定的人。"
        "不要查验对跳预言家的玩家；对跳位应通过白天发言、票型和放逐解决，夜晚验人用于开新视角。"
    )


def build_seer_night_strategy_directive(
    *,
    night_number: int,
    check_value: Mapping[str, Any] | Sequence[Any] | None,
    badge_flow_next: Sequence[str] | None,
    counterclaiming_seers: set[str],
) -> dict[str, Any]:
    """组合预言家夜晚验人的完整策略指令。"""
    seer_guidance = build_seer_night_guidance(night_number)
    strategy_directive: dict[str, Any] = {
        "seer_night_check": (
            "你是预言家，现在是夜间验人阶段。你必须选择一名玩家查验其身份。"
            "验人结果（好人/狼人）将在明天白天得知。"
            f"\n\n{seer_guidance}"
            "\n\n注意：本局没有守卫，预言家无法被守护，必须谨慎选择。"
            "\n\n【重要】本局存在混血儿角色，你的验人技能对混血儿显示'好人'，"
            "但混血儿可能在狼人阵营（取决于其主人阵营）。验出'好人'不代表100%安全。"
            "speech字段留空（夜间行动不需要发言）。"
        ),
    }
    if check_value:
        strategy_directive["check_value_assessment"] = check_value

    filtered_badge_flow_next = list(badge_flow_next or [])
    if filtered_badge_flow_next and counterclaiming_seers:
        filtered_badge_flow_next = [
            pid for pid in filtered_badge_flow_next
            if pid not in counterclaiming_seers
        ]
    if filtered_badge_flow_next:
        strategy_directive["badge_flow_plan"] = (
            f"你在警上承诺的警徽流计划中提到的验人对象: {filtered_badge_flow_next}，"
            "请优先按此计划验人以保持信息传递的一致性。"
        )
    if counterclaiming_seers:
        strategy_directive["excluded_counterclaiming_seers"] = sorted(counterclaiming_seers)
    return strategy_directive
