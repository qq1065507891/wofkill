# -*- coding: utf-8 -*-
"""
验证 action 修复 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_action_repair.py -q
"""

from __future__ import annotations


def test_action_repair_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import action_repair
    from werewolf_agent.agents import output_parser

    assert output_parser.vote_choice_map is action_repair.vote_choice_map
    assert output_parser.target_from_vote_decision is action_repair.target_from_vote_decision
    assert output_parser.repair_vote_decision is action_repair.repair_vote_decision
    assert output_parser.repair_target_decision is action_repair.repair_target_decision
    assert output_parser.repair_speech_intent_decision is action_repair.repair_speech_intent_decision
