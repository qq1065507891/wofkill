# -*- coding: utf-8 -*-
"""
测试混血儿首夜选择主人阶段的指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.hybrid_master_directives import build_hybrid_master_choice_directive
    >>> build_hybrid_master_choice_directive({})
"""

from werewolf_agent.agents.schemas import ActionType
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.hybrid_master_directives import (
    build_hybrid_master_candidates,
    build_hybrid_master_choice_directive,
    choose_hybrid_master_target,
)


def _make_game_state() -> GameState:
    players = {
        "hybrid": PlayerState(id="hybrid", role="hybrid", alive=True),
        "p02": PlayerState(id="p02", role="seer", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=False),
    }
    return GameState(game_id="hybrid_master_directives", players=players)


def test_build_hybrid_master_candidates_excludes_self_and_dead_players() -> None:
    """混血儿主人候选应只包含存活的非本人玩家。"""
    assert build_hybrid_master_candidates(_make_game_state(), "hybrid") == ["p02"]


def test_build_hybrid_master_choice_directive_includes_assessment() -> None:
    """选择主人指令应包含胜利绑定说明和候选评估。"""
    assessment = {"ranked_targets": [{"target": "p02", "value": 5}]}

    directive = build_hybrid_master_choice_directive(assessment)

    assert "你是混血儿" in directive["hybrid_master_choice"]
    assert "选择后不能更改" in directive["hybrid_master_choice"]
    assert directive["master_assessment"] == assessment


def test_choose_hybrid_master_target_uses_action_then_first_candidate() -> None:
    """混血儿目标选择应优先使用合法行动目标，否则回退第一个候选。"""
    assert choose_hybrid_master_target(
        action_type=ActionType.CHOOSE_MASTER,
        target_id="p02",
        candidates=["p02"],
    ) == "p02"
    assert choose_hybrid_master_target(
        action_type=ActionType.NO_ACTION,
        target_id=None,
        candidates=["p02"],
    ) == "p02"
    assert choose_hybrid_master_target(
        action_type=ActionType.NO_ACTION,
        target_id=None,
        candidates=[],
    ) is None
