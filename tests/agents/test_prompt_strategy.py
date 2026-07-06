# -*- coding: utf-8 -*-
"""
验证 strategy prompt 渲染拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_strategy.py -q
"""

from __future__ import annotations


def test_prompt_strategy_methods_remain_compatibly_available() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_strategy import PromptStrategyMixin

    assert PlayerPromptBuilder._build_strategy_directive is PromptStrategyMixin._build_strategy_directive
    assert PlayerPromptBuilder._render_strategy_section is PromptStrategyMixin._render_strategy_section
    assert PlayerPromptBuilder._render_skill_tactical_advice is PromptStrategyMixin._render_skill_tactical_advice
