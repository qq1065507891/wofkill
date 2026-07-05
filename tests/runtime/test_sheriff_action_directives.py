# -*- coding: utf-8 -*-
"""
测试警长行动阶段的目标和指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.sheriff_action_directives import living_non_sheriff_ids
    >>> living_non_sheriff_ids(...)
"""

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.sheriff_action_directives import (
    build_sheriff_endorse_directive,
    build_sheriff_endorse_result,
    build_sheriff_speech_order,
    build_sheriff_speech_order_directive,
    living_non_sheriff_ids,
)


def _make_game_state() -> GameState:
    players = {
        "sheriff": PlayerState(id="sheriff", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="seer", alive=True),
        "p03": PlayerState(id="p03", role="witch", alive=True),
        "p04": PlayerState(id="p04", role="werewolf", alive=False),
    }
    return GameState(
        game_id="sheriff_action_directives",
        phase="day",
        players=players,
        sheriff_id="sheriff",
    )


def test_living_non_sheriff_ids_excludes_sheriff_and_dead_players() -> None:
    """警长行动目标应只包含存活的非警长玩家。"""
    assert living_non_sheriff_ids(_make_game_state(), "sheriff") == ["p02", "p03"]


def test_build_sheriff_speech_order_directive_lists_alive_players() -> None:
    """发言顺序指令应包含可选玩家列表。"""
    directive = build_sheriff_speech_order_directive(["p02", "p03"])

    assert "choose_speech_order" in directive
    assert directive["alive_players"] == ["p02", "p03"]
    assert "请选择第一个发言的玩家" in directive["choose_speech_order"]


def test_build_sheriff_speech_order_moves_first_speaker_and_keeps_sheriff_last() -> None:
    """警长选择首发言人后，应保持该玩家第一、警长最后。"""
    assert build_sheriff_speech_order(
        first_speaker="p03",
        alive_players=["p02", "p03"],
        sheriff_id="sheriff",
    ) == ["p03", "p02", "sheriff"]
    assert build_sheriff_speech_order(
        first_speaker="p05",
        alive_players=["p02", "p03"],
        sheriff_id="sheriff",
    ) is None


def test_build_sheriff_endorse_directive_and_result_validate_target() -> None:
    """警长归票指令和结果应显式列出合法目标，并过滤非法目标。"""
    directive = build_sheriff_endorse_directive(["p02"])
    valid = build_sheriff_endorse_result(
        target="p02",
        alive_others=["p02"],
        private_reason="归票理由",
        action_trace={"action": "vote"},
    )
    invalid = build_sheriff_endorse_result(
        target="p03",
        alive_others=["p02"],
        private_reason="归票理由",
        action_trace={"action": "vote"},
    )

    assert directive["legal_endorse_targets"] == ["p02"]
    assert "作为警长，你需要归票" in directive["sheriff_endorse"]
    assert valid == {
        "endorse_target": "p02",
        "private_reason": "归票理由",
        "action_trace": {"action": "vote"},
    }
    assert invalid == {
        "endorse_target": "",
        "private_reason": "",
        "action_trace": None,
    }
