# -*- coding: utf-8 -*-
"""
测试狼队夜聊阶段的策略指令辅助函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.wolf_discussion_directives import build_wolf_discussion_instruction
    >>> build_wolf_discussion_instruction("w1", night_number=1, has_teammate_input=False, has_previous_speeches=False)
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_empty_wolf_discussion_fallback,
    build_teammate_transcript,
    build_wolf_discussion_instruction,
    build_wolf_discussion_strategy_directive,
    collect_wolf_discussion_speeches,
    living_wolf_ids,
    living_wolf_teammates,
    teammate_discussion_speeches,
)


def _make_game_state() -> GameState:
    players = {
        "w1": PlayerState(id="w1", role="werewolf", alive=True),
        "w2": PlayerState(id="w2", role="werewolf", alive=True),
        "w3": PlayerState(id="w3", role="werewolf", alive=False),
        "p1": PlayerState(id="p1", role="villager", alive=True),
    }
    return GameState(
        game_id="wolf_discussion_directives",
        phase="night",
        night_number=1,
        players=players,
        events=[
            GameEvent(type="wolf_discussion", payload={"wolf_id": "w1", "round": 1, "text": "我想刀p1"}),
            GameEvent(type="wolf_discussion", payload={"wolf_id": "w2", "round": 1, "text": "我同意"}),
            GameEvent(type="wolf_discussion", payload={"wolf_id": "w3", "round": 1, "text": "死狼不应出现"}),
        ],
    )


def test_living_wolf_ids_and_teammates_use_alive_wolves_only() -> None:
    """只把存活狼人视为夜聊成员和队友。"""
    gs = _make_game_state()

    assert living_wolf_ids(gs) == ["w1", "w2"]
    assert living_wolf_teammates(gs, "w1") == ["w2"]


def test_collect_wolf_discussion_speeches_filters_non_members() -> None:
    """夜聊历史只收集存活狼队成员的发言。"""
    speeches = collect_wolf_discussion_speeches(_make_game_state(), ["w1", "w2"])

    assert speeches == [
        {"wolf_id": "w1", "round": "1", "text": "我想刀p1"},
        {"wolf_id": "w2", "round": "1", "text": "我同意"},
    ]


def test_teammate_discussion_speeches_excludes_self() -> None:
    """队友发言不包含当前狼人自己的历史发言。"""
    speeches = collect_wolf_discussion_speeches(_make_game_state(), ["w1", "w2"])

    assert teammate_discussion_speeches(speeches, "w1") == [
        {"wolf_id": "w2", "round": "1", "text": "我同意"},
    ]


def test_build_wolf_discussion_instruction_adds_response_and_first_night_role_split() -> None:
    """有队友输入时要求回应，首夜首发时追加角色分工建议。"""
    first = build_wolf_discussion_instruction(
        "w1",
        night_number=1,
        has_teammate_input=False,
        has_previous_speeches=False,
    )
    response = build_wolf_discussion_instruction(
        "w1",
        night_number=2,
        has_teammate_input=True,
        has_previous_speeches=True,
    )

    assert "【身份约束】你的玩家ID是w1" in first
    assert "【首夜角色分工建议】" in first
    assert "你必须回应队友的发言" in response
    assert "【首夜角色分工建议】" not in response


def test_build_wolf_discussion_strategy_directive_keeps_last_eight_speeches() -> None:
    """策略指令只保留最近 8 条夜聊历史。"""
    speeches = [
        {"wolf_id": f"w{i}", "round": "1", "text": str(i)}
        for i in range(10)
    ]

    directive = build_wolf_discussion_strategy_directive(
        discussion_instruction="instruction",
        round_focus="focus",
        wolf_teammates=["w2"],
        previous_speeches=speeches,
    )

    assert directive["wolf_team_discussion"] == "instruction"
    assert directive["round_focus"] == "focus"
    assert directive["previous_discussion"] == speeches[-8:]


def test_build_teammate_transcript_uses_recent_six_teammate_speeches() -> None:
    """注入上下文的队友 transcript 只保留最近 6 条。"""
    speeches = [
        {"wolf_id": f"w{i}", "round": "1", "text": str(i)}
        for i in range(8)
    ]

    assert build_teammate_transcript(speeches) == [
        {"speaker": f"w{i}", "text": str(i)}
        for i in range(2, 8)
    ]


def test_build_empty_wolf_discussion_fallback_keeps_existing_wording() -> None:
    """空狼队夜聊发言兜底文案保持稳定。"""
    assert build_empty_wolf_discussion_fallback("w1", "p1", "本轮需要统一刀口。") == (
        "我是w1，本轮讨论我认为应该刀p1。"
        "本轮需要统一刀口。请大家发表意见。"
    )
