# -*- coding: utf-8 -*-
"""
测试白天投票阶段的策略指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.runtime.day_vote_directives import build_day_vote_base_directive
    >>> build_day_vote_base_directive("villager", allow_abstain=False, consecutive_no_exile=0)
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.day_vote_directives import (
    build_day_vote_base_directive,
    build_fallback_seer_vote_strategy,
    build_hunter_vote_strategy,
    build_hybrid_vote_strategy,
    build_seer_vote_strategy,
    build_vote_anti_herd_directive,
    build_villager_vote_strategy,
    build_witch_vote_strategy,
)


def _make_game_state_with_checks() -> GameState:
    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    return GameState(
        game_id="day_vote_directives",
        phase="day",
        day_number=2,
        players=players,
        events=[
            GameEvent(
                type="seer_check",
                payload={"target_id": "p02", "alignment": "wolf", "night_number": 1},
            ),
            GameEvent(
                type="seer_check",
                payload={"target_id": "p03", "alignment": "good", "night_number": 2},
            ),
        ],
    )


def test_build_day_vote_base_directive_adds_contract_pressure_and_good_guard() -> None:
    """基础投票指令包含结构化契约、强制投票压力和好人纪律。"""
    directive = build_day_vote_base_directive(
        "villager",
        allow_abstain=False,
        consecutive_no_exile=2,
    )

    assert directive["require_vote_quality"] is True
    assert "seer_stance" in directive["vote_structured_contract"]
    assert directive["vote_strategy"] == (
        "投票原则：\n"
        "1) 有查杀走查杀：如果被信任的预言家查杀了某人，优先投查杀对象。"
        "2) 跟预言家走：听完发言后，根据你信任的预言家的归票方向投票。"
        "3) 警上单边预言家可信度高，警下跳预言家的可信度很低。"
        "4) 如果没有明确查杀，投发言最可疑、逻辑最不通的人。"
    )
    assert directive["vote_pressure"] == "必须投票选出一名玩家放逐，不能弃票。 已经连续2天无人出局，必须做出决定。"
    assert "good_vote_decision_guard" in directive


def test_build_day_vote_base_directive_omits_pressure_when_abstain_allowed() -> None:
    """允许弃票时不注入强制投票压力，也不把狼人当好人角色处理。"""
    directive = build_day_vote_base_directive(
        "werewolf",
        allow_abstain=True,
        consecutive_no_exile=1,
    )

    assert "vote_pressure" not in directive
    assert "good_vote_decision_guard" not in directive


def test_build_seer_vote_strategy_lists_check_results() -> None:
    """预言家投票策略应列出已查到的狼人和好人。"""
    strategy = build_seer_vote_strategy(_make_game_state_with_checks())

    assert "你已查验出狼人: p02" in strategy
    assert "你已查验出好人: p03" in strategy


def test_build_villager_vote_strategy_uses_single_claimant_branch() -> None:
    """只有一个预言家声称者时使用单边预言家策略分支。"""
    strategy = build_villager_vote_strategy({"p06"})

    assert "场上只有p06单边跳预言家" in strategy
    assert "无对跳预言家" in strategy


def test_build_villager_vote_strategy_lists_counterclaims() -> None:
    """多个预言家声称者时列出对跳名单。"""
    strategy = build_villager_vote_strategy({"p08", "p06"})

    assert "对跳预言家 ['p06', 'p08']" in strategy


def test_static_role_vote_strategies_keep_role_specific_constraints() -> None:
    """女巫、猎人、混血儿策略保留原有角色约束。"""
    assert "不要在公开投票理由中提及药水使用细节" in build_witch_vote_strategy()
    assert "一旦你被放逐，你会开枪" in build_hunter_vote_strategy()
    hybrid_strategy = build_hybrid_vote_strategy("p03")
    assert "私下参考" in hybrid_strategy
    assert "投票理由中" in hybrid_strategy
    assert "不要暴露" in hybrid_strategy
    assert "你是混血儿" not in hybrid_strategy
    assert "主人是 p03" not in hybrid_strategy


def test_vote_anti_herd_and_fallback_seer_strategy_are_explicit_helpers() -> None:
    """通用防跟票和预言家 fallback 投票策略应由投票指令模块提供。"""
    assert "near-unanimous push" in build_vote_anti_herd_directive()
    assert "查验结果为核心依据" in build_fallback_seer_vote_strategy()
