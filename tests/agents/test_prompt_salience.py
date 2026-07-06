# -*- coding: utf-8 -*-
"""
验证 salience prompt 渲染拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_salience.py -q
"""

from __future__ import annotations


def test_prompt_salience_methods_remain_compatibly_available() -> None:
    from werewolf_agent.agents import prompt_builder
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_salience import PromptSalienceMixin
    from werewolf_agent.agents import prompt_salience

    assert PlayerPromptBuilder._build_salience_events is PromptSalienceMixin._build_salience_events
    assert prompt_builder._slim_salience_item is prompt_salience._slim_salience_item
    assert prompt_builder._SALIENCE_PUBLIC_FIELDS == prompt_salience._SALIENCE_PUBLIC_FIELDS
    assert prompt_builder._SALIENCE_PRIVATE_KEYS == prompt_salience._SALIENCE_PRIVATE_KEYS
