# -*- coding: utf-8 -*-
"""
验证玩家动作合法性校验器的基础行为。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.action_validation import DefaultActionValidator
    >>> DefaultActionValidator()
"""

from werewolf_agent.agents.action_validation import DefaultActionValidator
from werewolf_agent.agents.schemas import ActionType


def test_default_action_validator_rejects_illegal_action_type():
    validator = DefaultActionValidator()

    valid, error = validator.validate(
        ActionType.VOTE,
        "p01",
        legal_actions=[ActionType.NO_ACTION],
        legal_targets=["p01"],
    )

    assert valid is False
    assert error == "action_type=vote not in legal_actions"


def test_default_action_validator_rejects_target_outside_legal_targets():
    validator = DefaultActionValidator()

    valid, error = validator.validate(
        ActionType.VOTE,
        "p02",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p01"],
    )

    assert valid is False
    assert error == "target_id=p02 not in legal_targets"
