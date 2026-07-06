# -*- coding: utf-8 -*-
"""
验证 action 数据归一化 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_action_normalization.py -q
"""

from __future__ import annotations


def test_action_normalization_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import action_normalization
    from werewolf_agent.agents import output_parser

    assert output_parser.normalize_action_data is action_normalization.normalize_action_data
    assert output_parser._normalize_typos is action_normalization._normalize_typos
    assert output_parser.clean_enum_value is action_normalization.clean_enum_value
    assert output_parser.clean_reason is action_normalization.clean_reason
    assert output_parser.sanitize_optional_private_fields is action_normalization.sanitize_optional_private_fields
    assert output_parser._safe_default_claimed_view is action_normalization._safe_default_claimed_view
    assert output_parser._TYPO_ALIASES is action_normalization._TYPO_ALIASES
    assert output_parser._REASON_PLACEHOLDERS is action_normalization._REASON_PLACEHOLDERS
    assert output_parser._VALID_CLAIMED_VIEW_VALUES is action_normalization._VALID_CLAIMED_VIEW_VALUES


def test_output_parser_facade_preserves_legacy_schema_aliases() -> None:
    import logging

    from werewolf_agent.agents import output_parser
    from werewolf_agent.agents.schemas import ActionType, FactionGoal, PlayerAction, RiskFlag, SeerStance, VoteBasis

    assert output_parser.ActionType is ActionType
    assert output_parser.FactionGoal is FactionGoal
    assert output_parser.PlayerAction is PlayerAction
    assert output_parser.RiskFlag is RiskFlag
    assert output_parser.SeerStance is SeerStance
    assert output_parser.VoteBasis is VoteBasis
    assert isinstance(output_parser.logger, logging.Logger)
