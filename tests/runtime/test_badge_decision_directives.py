# -*- coding: utf-8 -*-
"""
测试警徽去向决策阶段的指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.badge_decision_directives import build_badge_decision_directive
    >>> build_badge_decision_directive("seer", ["p02"])
"""

import pytest

from werewolf_agent.agents.player_fallback_speech import build_task_terminal_fallback
from werewolf_agent.agents.schemas import (
    ActionTrace,
    ActionType,
    FallbackAction,
    RetryInfo,
)
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.badge_decision_directives import (
    build_badge_decision_directive,
    build_badge_decision_result,
    build_badge_role_hint,
)


def test_build_badge_role_hint_is_role_specific() -> None:
    """警徽去向角色提示应区分狼人、预言家和普通好人警长。"""
    assert "狼队利益" in build_badge_role_hint("werewolf")
    assert "金水" in build_badge_role_hint("seer")
    assert "明好人" in build_badge_role_hint("villager")


def test_build_badge_decision_directive_lists_options_and_alive_players() -> None:
    """警徽决策指令应列出移交/撕毁选项和存活候选。"""
    directive = build_badge_decision_directive("seer", ["p02", "p03"])

    assert "badge_decision" in directive
    assert "移交（BADGE_TRANSFER）" in directive["badge_decision"]
    assert "撕毁（BADGE_TEAR）" in directive["badge_decision"]
    assert directive["alive_players"] == ["p02", "p03"]


def test_build_badge_decision_result_maps_transfer_and_tear() -> None:
    """警徽行动结果应把移交映射为 transfer，其余映射为 tear。"""
    transfer = build_badge_decision_result(
        action_type=ActionType.BADGE_TRANSFER,
        target_id="p02",
        action_trace={"action": "badge_transfer"},
    )
    tear = build_badge_decision_result(
        action_type=ActionType.BADGE_TEAR,
        target_id=None,
        action_trace={"action": "badge_tear"},
    )

    assert transfer == {
        "badge_decision": "transfer",
        "badge_target_id": "p02",
        "action_trace": {"action": "badge_transfer"},
    }
    assert tear == {
        "badge_decision": "tear",
        "badge_target_id": None,
        "action_trace": {"action": "badge_tear"},
    }


def test_build_badge_decision_result_rejects_non_badge_fallback_action() -> None:
    with pytest.raises(ValueError, match="badge decision requires"):
        build_badge_decision_result(
            action_type=ActionType.NO_ACTION,
            target_id=None,
            action_trace={"fallback_kind": "last_words_not_generated"},
        )


def test_agent_badge_decision_keeps_badge_terminal_fallback_semantics() -> None:
    from werewolf_agent.runtime.agent_special_actions import agent_badge_decision

    class TerminalAgent:
        def act(self, context):
            action, fallback_kind = build_task_terminal_fallback(
                context,
                FallbackAction(action_type=ActionType.NO_ACTION),
            )
            trace = ActionTrace(
                generated_by="terminal_fallback",
                terminal_failure_code="schema_validation",
                original_failure_code="schema_validation",
                failure_stage="protocol",
                fallback_kind=fallback_kind,
            )
            return action.model_copy(update={"trace": trace}), RetryInfo()

    class Registry:
        def get_agent(self, _player_id):
            return TerminalAgent()

    gs = GameState(
        game_id="badge-terminal",
        players={
            "p01": PlayerState(id="p01", role="seer", alive=False),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
        sheriff_id="p01",
    )

    result = agent_badge_decision(
        {"game_state": gs},
        RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml"),
        Registry(),
        "p01",
    )

    assert result["badge_decision"] == "transfer"
    assert result["badge_target_id"] == "p02"
    assert result["action_trace"]["fallback_kind"] == "badge_transfer"
