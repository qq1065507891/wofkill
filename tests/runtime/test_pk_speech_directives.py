# -*- coding: utf-8 -*-
"""
测试 PK 发言阶段的策略指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy
    >>> build_pk_speech_strategy(...)
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy


def _make_game_state(role: str, *, master_id: str = "") -> GameState:
    players = {
        "speaker": PlayerState(id="speaker", role=role, alive=True),
        "target": PlayerState(id="target", role="werewolf", alive=True),
    }
    return GameState(
        game_id="pk_speech_directives",
        phase="pk",
        players=players,
        hybrid_master_id=master_id,
        events=[
            GameEvent(
                type="seer_check",
                payload={"target_id": "target", "alignment": "wolf", "night_number": 1},
            ),
        ],
    )


def test_build_pk_speech_strategy_always_adds_urgent_instruction() -> None:
    """PK 发言策略应始终包含紧迫性提示。"""
    strategy = build_pk_speech_strategy(_make_game_state("villager"), "speaker")

    assert "pk_urgent" in strategy
    assert "不要再'等下一轮'" in strategy["pk_urgent"]


def test_build_pk_speech_strategy_adds_wolf_push() -> None:
    """狼人 PK 发言应收到伪装好人并攻击对手的提示。"""
    strategy = build_pk_speech_strategy(_make_game_state("werewolf"), "speaker")

    assert "wolf_pk_push" in strategy
    assert "表现得像一个有分析能力的好人" in strategy["wolf_pk_push"]


def test_build_pk_speech_strategy_adds_seer_check_evidence() -> None:
    """预言家 PK 发言应锚定已有查验结果。"""
    strategy = build_pk_speech_strategy(_make_game_state("seer"), "speaker")

    assert "seer_pk_check_evidence" in strategy
    assert "你已获得 1 个查验结果" in strategy["seer_pk_check_evidence"]


def test_build_pk_speech_strategy_adds_hybrid_master_alignment() -> None:
    """混血儿有主人时应收到主人方向提示。"""
    strategy = build_pk_speech_strategy(
        _make_game_state("hybrid", master_id="target"),
        "speaker",
    )

    assert "hybrid_pk_master_align" in strategy
    assert "主人是target" in strategy["hybrid_pk_master_align"]
