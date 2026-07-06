# -*- coding: utf-8 -*-
"""
验证发言意图解析 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_speech_intent_parser.py -q
"""

from __future__ import annotations


def test_speech_intent_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.agents import output_parser
    from werewolf_agent.agents import speech_intent_parser

    assert output_parser.infer_speech_intent is speech_intent_parser.infer_speech_intent
    assert output_parser.ensure_speech_quality_components is speech_intent_parser.ensure_speech_quality_components
    assert output_parser.infer_vote_basis is speech_intent_parser.infer_vote_basis
