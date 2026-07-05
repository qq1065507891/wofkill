# -*- coding: utf-8 -*-
"""
测试预言家夜晚验人阶段的指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.seer_night_directives import build_seer_night_guidance
    >>> build_seer_night_guidance(1)
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.seer_night_directives import (
    build_badge_flow_next_targets,
    build_seer_legal_targets,
    build_seer_night_guidance,
    build_seer_night_strategy_directive,
    collect_seer_checked_target_ids,
)


def _make_game_state() -> GameState:
    players = {
        "seer": PlayerState(id="seer", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=False),
        "p04": PlayerState(id="p04", role="werewolf", alive=True),
        "p05": PlayerState(id="p05", role="villager", alive=True),
    }
    return GameState(
        game_id="seer_night_directives",
        players=players,
        phase="night",
        night_number=2,
        events=[
            GameEvent(
                type="seer_check",
                payload={"target_id": "p05", "alignment": "good", "night_number": 1},
            ),
            GameEvent(
                type="sheriff_speech",
                payload={"speaker": "seer", "text": "我的警徽流先验p05，再验p02。"},
            ),
        ],
    )


def test_collect_seer_checked_target_ids_reads_existing_checks() -> None:
    """已验目标集合应来自 seer_check 事件。"""
    checked = collect_seer_checked_target_ids(_make_game_state())

    assert checked == {"p05"}


def test_build_seer_legal_targets_excludes_self_dead_counterclaims_and_checked() -> None:
    """合法验人目标应排除自己、死亡者、对跳预言家和已验目标。"""
    gs = _make_game_state()

    targets = build_seer_legal_targets(
        gs,
        seer_id="seer",
        counterclaiming_seers={"p04"},
    )

    assert targets == ["p02"]


def test_build_badge_flow_next_targets_keeps_only_current_legal_targets() -> None:
    """警徽流下一验只保留当前仍可合法查验的对象。"""
    gs = _make_game_state()

    next_targets = build_badge_flow_next_targets(
        gs,
        seer_id="seer",
        legal_targets=["p02"],
    )

    assert next_targets == ["p02"]


def test_build_seer_night_guidance_uses_night_specific_strategy() -> None:
    """首夜和后续夜晚应使用不同验人策略文案。"""
    first_night = build_seer_night_guidance(1)
    later_night = build_seer_night_guidance(2)

    assert "首夜验人对象" in first_night
    assert "白天讨论中你最怀疑的人" in later_night
    assert "不要查验对跳预言家的玩家" in first_night
    assert "不要查验对跳预言家的玩家" in later_night


def test_build_seer_night_strategy_directive_adds_optional_assessments() -> None:
    """完整策略指令应合并查验价值、警徽流和对跳排除说明。"""
    check_value = {"p02": {"score": 8, "reason": "发言矛盾"}}

    directive = build_seer_night_strategy_directive(
        night_number=1,
        check_value=check_value,
        badge_flow_next=["p02", "p04"],
        counterclaiming_seers={"p04"},
    )

    assert "你是预言家，现在是夜间验人阶段" in directive["seer_night_check"]
    assert directive["check_value_assessment"] == check_value
    assert directive["badge_flow_plan"] == (
        "你在警上承诺的警徽流计划中提到的验人对象: ['p02']，"
        "请优先按此计划验人以保持信息传递的一致性。"
    )
    assert directive["excluded_counterclaiming_seers"] == ["p04"]
