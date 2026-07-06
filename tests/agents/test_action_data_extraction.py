# -*- coding: utf-8 -*-
"""
验证 action 原始数据提取 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_action_data_extraction.py -q
"""

from __future__ import annotations


def test_action_data_extraction_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import action_data_extraction
    from werewolf_agent.agents import output_parser

    assert output_parser.extract_parameter_tag_action is action_data_extraction.extract_parameter_tag_action
    assert output_parser.extract_partial_decision_data is action_data_extraction.extract_partial_decision_data
    assert output_parser.extract_decision_data is action_data_extraction.extract_decision_data
