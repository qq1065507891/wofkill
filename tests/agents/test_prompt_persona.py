# -*- coding: utf-8 -*-
"""
验证 persona prompt 渲染拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_persona.py -q
"""

from __future__ import annotations


def test_prompt_persona_methods_remain_compatibly_available() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_persona import PromptPersonaMixin

    assert PlayerPromptBuilder._build_persona is PromptPersonaMixin._build_persona
    assert PlayerPromptBuilder._slim_numeric_params is PromptPersonaMixin._slim_numeric_params
