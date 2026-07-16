# -*- coding: utf-8 -*-
"""
测试狼队夜聊阶段的策略指令辅助函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.wolf_discussion_directives import build_wolf_discussion_instruction
    >>> build_wolf_discussion_instruction("w1", night_number=1, has_teammate_input=False, has_previous_speeches=False)
"""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.action_contract import ActionContract
from werewolf_agent.agents.output_parser import action_from_data
from werewolf_agent.agents.schemas import (
    ActionType,
    OutputMode,
    TaskType,
    WolfTargetStance,
)
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.event_metadata import new_game_event
from werewolf_agent.runtime.wolf_discussion_directives import (
    build_validated_wolf_target_stance,
    build_empty_wolf_discussion_fallback,
    build_teammate_transcript,
    build_wolf_discussion_instruction,
    build_wolf_discussion_strategy_directive,
    collect_current_wolf_target_stances,
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
        events=[],
    )


def _trusted_discussion_event(
    gs: GameState,
    *,
    wolf_id: str,
    round_number: int,
    text: str,
) -> GameEvent:
    return new_game_event(
        gs,
        "wolf_discussion",
        {
            "wolf_id": wolf_id,
            "round": round_number,
            "night_number": gs.night_number,
            "text": text,
        },
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        now=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


def test_living_wolf_ids_and_teammates_use_alive_wolves_only() -> None:
    """只把存活狼人视为夜聊成员和队友。"""
    gs = _make_game_state()

    assert living_wolf_ids(gs) == ["w1", "w2"]
    assert living_wolf_teammates(gs, "w1") == ["w2"]


def test_collect_wolf_discussion_speeches_filters_non_members() -> None:
    """夜聊历史只收集存活狼队成员的发言。"""
    gs = _make_game_state()
    first = _trusted_discussion_event(
        gs, wolf_id="w1", round_number=1, text="我想刀p1"
    )
    gs = replace(gs, events=[first])
    second = _trusted_discussion_event(
        gs, wolf_id="w2", round_number=1, text="我同意"
    )
    dead = _trusted_discussion_event(
        replace(gs, events=[first, second]),
        wolf_id="w3",
        round_number=1,
        text="死狼不应出现",
    )
    gs = replace(gs, events=[first, second, dead])
    speeches = collect_wolf_discussion_speeches(gs, ["w1", "w2"])

    assert speeches == [
        {"wolf_id": "w1", "round": "1", "text": "我想刀p1"},
        {"wolf_id": "w2", "round": "1", "text": "我同意"},
    ]


def test_teammate_discussion_speeches_excludes_self() -> None:
    """队友发言不包含当前狼人自己的历史发言。"""
    gs = _make_game_state()
    first = _trusted_discussion_event(
        gs, wolf_id="w1", round_number=1, text="我想刀p1"
    )
    gs = replace(gs, events=[first])
    second = _trusted_discussion_event(
        gs, wolf_id="w2", round_number=1, text="我同意"
    )
    speeches = collect_wolf_discussion_speeches(
        replace(gs, events=[first, second]),
        ["w1", "w2"],
    )

    assert teammate_discussion_speeches(speeches, "w1") == [
        {"wolf_id": "w2", "round": "1", "text": "我同意"},
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, schema_version=None),
        lambda event: replace(event, game_id="other_game"),
        lambda event: replace(event, visibility=EventVisibility.PUBLIC),
        lambda event: replace(event, visibility=EventVisibility.MODERATOR_ONLY),
        lambda event: replace(
            event,
            payload={**event.payload, "night_number": 2},
        ),
        lambda event: replace(
            event,
            payload={**event.payload, "round": 0},
        ),
        lambda event: replace(
            event,
            payload={**event.payload, "wolf_id": "p1"},
        ),
    ],
)
def test_collect_wolf_discussion_speeches_rejects_untrusted_events(mutate) -> None:
    """队友 prompt 不得接收旧版、跨局、跨夜或错误可见性的文本。"""
    gs = _make_game_state()
    event = _trusted_discussion_event(
        gs, wolf_id="w1", round_number=1, text="trusted text"
    )
    gs = replace(gs, events=[mutate(event)])

    assert collect_wolf_discussion_speeches(gs, ["w1", "w2"]) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, game_id="other_game"),
        lambda event: replace(event, visibility=EventVisibility.PUBLIC),
        lambda event: replace(
            event,
            payload={**event.payload, "wolf_id": "w2"},
        ),
        lambda event: replace(
            event,
            payload={**event.payload, "round": 2},
        ),
    ],
    ids=["cross_game", "public", "actor_mismatch", "round_mismatch"],
)
def test_build_validated_stance_rejects_mismatched_source_event(mutate) -> None:
    """构造边界直接拒绝跨局、公开、actor 或轮次不一致的 source。"""
    gs = _make_game_state()
    event = _trusted_discussion_event(
        gs, wolf_id="w1", round_number=1, text=""
    )

    with pytest.raises(ValueError):
        build_validated_wolf_target_stance(
            gs,
            mutate(event),
            wolf_id="w1",
            round_number=1,
            raw_stance={
                "target_id": "p1",
                "stance": "propose",
                "priority": "primary",
            },
        )


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


def test_first_night_role_split_restrains_mechanical_fake_seer_push() -> None:
    """首夜分工提示不能无条件鼓励机械悍跳。"""
    first = build_wolf_discussion_instruction(
        "w1",
        night_number=1,
        has_teammate_input=False,
        has_previous_speeches=False,
    )

    assert "fake_seer (悍跳位)" in first
    assert "悍跳是很好的选择" not in first
    assert "不要在缺少白天公开发言证据时机械悍跳" in first


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
    """空夜聊兜底不得猜测击杀目标，结构化立场由调用方记为 abstain。"""
    speech = build_empty_wolf_discussion_fallback("w1", "本轮需要统一刀口。")

    assert speech == (
        "我是w1，本轮暂不提出击杀目标。"
        "本轮需要统一刀口。请大家发表意见。"
    )
    assert "p1" not in speech


@pytest.mark.parametrize(
    ("stance", "target_id"),
    [
        ("propose", "p1"),
        ("support", "p1"),
        ("oppose", "p1"),
        ("abstain", None),
    ],
)
def test_wolf_target_stance_schema_accepts_all_legal_stances(
    stance: str,
    target_id: str | None,
) -> None:
    """完整 stance schema 覆盖四种合法立场。"""
    parsed = WolfTargetStance.model_validate(
        {
            "wolf_id": "w1",
            "target_id": target_id,
            "stance": stance,
            "priority": "primary",
            "source_event_id": "wolf_discussion_directives:e000003",
            "round_number": 1,
        }
    )

    assert parsed.stance == stance
    assert parsed.target_id == target_id


def test_wolf_target_stance_schema_rejects_abstain_with_target() -> None:
    """abstain 必须显式不带目标。"""
    with pytest.raises(ValidationError, match="abstain"):
        WolfTargetStance.model_validate(
            {
                "wolf_id": "w1",
                "target_id": "p1",
                "stance": "abstain",
                "priority": "primary",
                "source_event_id": "wolf_discussion_directives:e000003",
                "round_number": 1,
            }
        )


@pytest.mark.parametrize(
    ("wolf_id", "target_id", "match"),
    [
        ("p1", "w1", "alive werewolf"),
        ("w1", "dead", "alive non-werewolf"),
        ("w1", "w2", "alive non-werewolf"),
    ],
)
def test_runtime_stance_validation_rejects_illegal_actor_or_target(
    wolf_id: str,
    target_id: str,
    match: str,
) -> None:
    """运行时结合存活状态拒绝非法 actor、死亡目标和狼队友目标。"""
    gs = _make_game_state()
    gs.players["dead"] = PlayerState(id="dead", role="villager", alive=False)
    event = new_game_event(
        gs,
        "wolf_discussion",
        {"wolf_id": wolf_id, "round": 1, "night_number": 1, "text": ""},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )

    with pytest.raises(ValueError, match=match):
        build_validated_wolf_target_stance(
            gs,
            event,
            wolf_id=wolf_id,
            round_number=1,
            raw_stance={
                "target_id": target_id,
                "stance": "propose",
                "priority": "primary",
            },
        )


def test_runtime_stance_references_same_night_v2_discussion_event() -> None:
    """写入的 stance 必须引用同夜 V2 wolf_discussion 事件 ID。"""
    gs = _make_game_state()
    event = new_game_event(
        gs,
        "wolf_discussion",
        {"wolf_id": "w1", "round": 1, "night_number": 1, "text": "建议刀p1"},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )

    stance = build_validated_wolf_target_stance(
        gs,
        event,
        wolf_id="w1",
        round_number=1,
        raw_stance={
            "target_id": "p1",
            "stance": "support",
            "priority": "backup",
        },
    )

    assert event.schema_version == "2"
    assert stance.source_event_id == event.event_id
    assert stance.round_number == 1


def test_structured_stance_collector_rejects_forged_non_v2_or_dead_target() -> None:
    """下游只能读取仍满足实时约束的 V2 stance。"""
    gs = _make_game_state()
    gs.players["dead"] = PlayerState(id="dead", role="villager", alive=False)
    forged = GameEvent(
        type="wolf_discussion",
        payload={
            "wolf_id": "w1",
            "round": 1,
            "night_number": 1,
            "text": "",
            "target_stance": {
                "wolf_id": "w1",
                "target_id": "dead",
                "stance": "propose",
                "priority": "primary",
                "source_event_id": "wolf_discussion_directives:e000099",
                "round_number": 1,
            },
        },
        event_id="wolf_discussion_directives:e000099",
    )
    gs = GameState(
        game_id=gs.game_id,
        phase=gs.phase,
        night_number=gs.night_number,
        players=gs.players,
        events=[forged],
    )

    assert collect_current_wolf_target_stances(gs) == []


def test_collector_retains_valid_stance_when_target_dies_after_event() -> None:
    """事件写入时合法的目标稍后死亡，权威立场仍用于判断主刀失效与备刀。"""
    gs = _make_game_state()
    event = new_game_event(
        gs,
        "wolf_discussion",
        {"wolf_id": "w1", "round": 1, "night_number": 1, "text": ""},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    )
    stance = build_validated_wolf_target_stance(
        gs,
        event,
        wolf_id="w1",
        round_number=1,
        raw_stance={
            "target_id": "p1",
            "stance": "support",
            "priority": "primary",
        },
    )
    event = replace(
        event,
        payload={**event.payload, "target_stance": stance.model_dump()},
    )
    players = {
        **gs.players,
        "p1": replace(gs.players["p1"], alive=False),
    }
    gs = replace(gs, players=players, events=[event])

    assert collect_current_wolf_target_stances(gs) == [stance.model_dump()]


def _with_naive_occurred_at(event: GameEvent) -> GameEvent:
    """构造绕过 dataclass 入口的损坏内存事件，验证读取边界 fail closed。"""
    forged = replace(event)
    object.__setattr__(forged, "occurred_at", datetime(2026, 7, 16))
    return forged


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, sequence_number=None),
        lambda event: replace(event, sequence_number=-1),
        lambda event: replace(event, occurred_at=None),
        _with_naive_occurred_at,
        lambda event: replace(event, event_id="forged:e999999"),
        lambda event: replace(event, game_id="other_game"),
        lambda event: replace(event, visibility=EventVisibility.PUBLIC),
        lambda event: replace(event, visibility=EventVisibility.MODERATOR_ONLY),
    ],
    ids=[
        "missing_sequence",
        "invalid_sequence",
        "missing_occurred_at",
        "invalid_occurred_at",
        "noncanonical_event_id",
        "wrong_game_id",
        "public_visibility",
        "moderator_visibility",
    ],
)
def test_structured_stance_collector_requires_authoritative_private_v2_event(
    mutate,
) -> None:
    """伪造或错误可见性的 V2 事件必须 fail closed。"""
    gs = _make_game_state()
    event = new_game_event(
        gs,
        "wolf_discussion",
        {"wolf_id": "w1", "round": 1, "night_number": 1, "text": ""},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        now=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )
    stance = build_validated_wolf_target_stance(
        gs,
        event,
        wolf_id="w1",
        round_number=1,
        raw_stance={
            "target_id": "p1",
            "stance": "propose",
            "priority": "primary",
        },
    )
    event = replace(
        event,
        payload={**event.payload, "target_stance": stance.model_dump()},
    )
    forged = mutate(event)
    gs = replace(gs, events=[forged])

    assert collect_current_wolf_target_stances(gs) == []


def test_only_wolf_discussion_action_contract_exposes_target_stance() -> None:
    """普通白天 speech schema 不得暴露狼队私有 stance。"""
    wolf_schema = ActionContract.build(
        output_mode=OutputMode.FULL_ACTION,
        task_type=TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p1"],
    ).json_schema
    day_schema = ActionContract.build(
        output_mode=OutputMode.FULL_ACTION,
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p1"],
    ).json_schema

    assert "target_stance" in wolf_schema["properties"]
    assert "target_stance" not in day_schema["properties"]

    action, error = action_from_data(
        {
            "action_type": "speech",
            "target_id": None,
            "speech": "今晚先观察。",
            "reason": "狼队夜聊",
            "confidence": 0.5,
            "target_stance": {
                "target_id": None,
                "stance": "abstain",
                "priority": "primary",
            },
        },
        task_type=TaskType.WOLF_DISCUSSION,
    )
    assert error is None
    assert action is not None
    assert action.target_stance.stance == "abstain"


def test_generic_action_parser_rejects_target_stance_without_wolf_context() -> None:
    """普通 action 解析不得因模型多返回字段而接纳狼队私有 stance。"""
    action, error = action_from_data(
        {
            "action_type": "speech",
            "target_id": None,
            "speech": "白天正常发言。",
            "reason": "公开讨论",
            "confidence": 0.5,
            "target_stance": {
                "target_id": None,
                "stance": "abstain",
                "priority": "primary",
            },
        }
    )

    assert action is None
    assert error is not None
    assert "target_stance" in error
