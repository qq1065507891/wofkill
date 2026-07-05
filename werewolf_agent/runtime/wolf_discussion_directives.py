# -*- coding: utf-8 -*-
"""
构建狼队夜聊阶段的策略指令、队友摘要和兜底文案。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.wolf_discussion_directives import build_wolf_discussion_instruction
    >>> build_wolf_discussion_instruction("w1", night_number=1, has_teammate_input=False, has_previous_speeches=False)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def living_wolf_ids(gs: GameState) -> list[str]:
    """返回当前存活狼人 ID。"""
    return [
        player_id
        for player_id, player in gs.players.items()
        if player.alive and player.role == "werewolf"
    ]


def living_wolf_teammates(gs: GameState, wolf_id: str) -> list[str]:
    """返回当前狼人之外的存活狼队队友。"""
    return [player_id for player_id in living_wolf_ids(gs) if player_id != wolf_id]


def collect_wolf_discussion_speeches(
    gs: GameState,
    wolf_ids: list[str],
) -> list[dict[str, str]]:
    """收集狼队成员已有的夜聊发言。"""
    wolf_id_set = set(wolf_ids)
    speeches: list[dict[str, str]] = []
    for event in gs.events:
        if event.type == "wolf_discussion" and event.payload.get("wolf_id") in wolf_id_set:
            speeches.append({
                "wolf_id": str(event.payload.get("wolf_id", "")),
                "round": str(event.payload.get("round", "")),
                "text": str(event.payload.get("text", "")),
            })
    return speeches


def teammate_discussion_speeches(
    previous_speeches: list[dict[str, str]],
    wolf_id: str,
) -> list[dict[str, str]]:
    """从夜聊历史中筛出队友发言。"""
    return [
        speech
        for speech in previous_speeches
        if speech["wolf_id"] != wolf_id
    ]


def build_wolf_discussion_instruction(
    wolf_id: str,
    *,
    night_number: int,
    has_teammate_input: bool,
    has_previous_speeches: bool,
) -> str:
    """构建当前狼人夜聊发言的核心提示。"""
    instruction = (
        "这是狼队密谈，只有狼人队友能看到。你必须以狼人视角发言，讨论狼队策略。"
        "禁止假装好人视角发言，禁止质疑或试探队友身份——你清楚知道谁是队友。"
        "必须发言，不能沉默。必须提出具体的击杀目标或战术建议。\n"
        "注意用词：被放逐或已死的队友是'队友'或'悍跳狼'，不要叫TA'预言家'。"
        "即使TA白天冒充了预言家，在狼队内部你们应该用真实身份称呼。\n"
        f"【身份约束】你的玩家ID是{wolf_id}。在发言中只能以{wolf_id}自称，"
        "绝对不能自称其他玩家的ID或使用别人的ID发言。"
    )
    if has_teammate_input:
        instruction += (
            "\n\n重要：你必须回应队友的发言！看看队友提出了什么建议，"
            "表示同意、反对或补充意见，形成真正的团队讨论，而不是自顾自发言。"
        )
    if night_number == 1 and not has_previous_speeches:
        instruction += (
            "\n\n【首夜角色分工建议】狼队可以分工配合 (字段名 ↔ 中文):\n"
            "1) fake_seer (悍跳位)——白天假装预言家争夺警徽（建议由能言善辩的队友担任）\n"
            "2) pusher (冲锋位)——为悍跳队友强力站边，质疑真预言家\n"
            "3) hooker (倒钩位)——表面上站边真预言家，暗中破坏好人节奏\n"
            "4) deep_cover (深水位)——保持低调，活到最后为团队收尾\n"
            "讨论谁适合什么角色，但不一定每局都需要悍跳。"
            "如果真预言家查验理由薄弱，悍跳是很好的选择。"
            "建议在发言里明确写出角色名 (例如 '我做悍跳' / 'p04 做倒钩'), "
            "队长会基于这些表态做最终结构化分工。"
        )
    return instruction


def build_wolf_discussion_strategy_directive(
    *,
    discussion_instruction: str,
    round_focus: str,
    wolf_teammates: list[str],
    previous_speeches: list[dict[str, str]],
) -> dict[str, Any]:
    """构建狼队夜聊的策略指令字典。"""
    return {
        "wolf_team_discussion": discussion_instruction,
        "round_focus": round_focus,
        "wolf_teammates": wolf_teammates,
        "previous_discussion": previous_speeches[-8:],
    }


def build_teammate_transcript(
    teammate_speeches: list[dict[str, str]],
) -> list[dict[str, str]]:
    """把队友夜聊发言转换为 AgentContext 的 transcript 条目。"""
    return [
        {"speaker": speech["wolf_id"], "text": speech["text"]}
        for speech in teammate_speeches[-6:]
    ]


def build_empty_wolf_discussion_fallback(
    wolf_id: str,
    fallback_target: str,
    required_text: str,
) -> str:
    """构建空狼队夜聊发言的兜底文本。"""
    return (
        f"我是{wolf_id}，本轮讨论我认为应该刀{fallback_target}。"
        f"{required_text}请大家发表意见。"
    )
