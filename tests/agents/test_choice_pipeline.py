# -*- coding: utf-8 -*-
"""
验证 choice / speech-intent pipeline 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_choice_pipeline.py -q
"""

from __future__ import annotations


def test_choice_pipeline_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import choice_pipeline
    from werewolf_agent.agents import output_parser

    assert output_parser.parse_choice_action is choice_pipeline.parse_choice_action
    assert output_parser.parse_speech_intent_action is choice_pipeline.parse_speech_intent_action
    assert output_parser.uses_choice_pipeline is choice_pipeline.uses_choice_pipeline
    assert output_parser.uses_speech_intent_pipeline is choice_pipeline.uses_speech_intent_pipeline
