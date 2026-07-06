# -*- coding: utf-8 -*-
"""
验证 prompt section 元数据拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_sections.py -q
"""

from __future__ import annotations


def test_prompt_section_metadata_remains_compatibly_importable() -> None:
    from werewolf_agent.agents import prompt_sections
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    assert PlayerPromptBuilder._USER_SECTION_SPECS is prompt_sections.USER_SECTION_SPECS
    assert PlayerPromptBuilder._SECTION_SPEC_BY_NAME == prompt_sections.SECTION_SPEC_BY_NAME
    assert PlayerPromptBuilder._NEVER_DROP == prompt_sections.NEVER_DROP_SECTIONS
    assert PlayerPromptBuilder._LOW_VALUE_SECTIONS == prompt_sections.LOW_VALUE_SECTIONS
    assert PlayerPromptBuilder._SECTION_PRIORITIES == prompt_sections.SECTION_PRIORITIES


def test_prompt_builder_reexports_split_compatibility_constants() -> None:
    from werewolf_agent.agents import prompt_builder
    from werewolf_agent.agents import prompt_memory
    from werewolf_agent.agents import prompt_salience
    from werewolf_agent.agents import prompt_sections
    from werewolf_agent.agents import prompt_strategy

    assert prompt_builder.REFLECTION_CARD_BUDGET == prompt_memory.REFLECTION_CARD_BUDGET
    assert prompt_builder._MAX_LEARNING_TEXT_CHARS == prompt_memory._MAX_LEARNING_TEXT_CHARS
    assert prompt_builder._MAX_RAG_TEXT_CHARS == prompt_memory._MAX_RAG_TEXT_CHARS
    assert prompt_builder._MAX_LEARNING_CONTEXT_CHARS == prompt_memory._MAX_LEARNING_CONTEXT_CHARS
    assert prompt_builder._MAX_SALIENCE_ITEMS == prompt_salience._MAX_SALIENCE_ITEMS
    assert prompt_builder._MAX_SKILL_TACTICAL_ADVICE_ITEMS == prompt_strategy._MAX_SKILL_TACTICAL_ADVICE_ITEMS
    assert prompt_builder._MAX_SKILL_TACTICAL_ADVICE_CHARS == prompt_strategy._MAX_SKILL_TACTICAL_ADVICE_CHARS
    assert prompt_builder._STRATEGY_GROUP_ORDER == prompt_strategy._STRATEGY_GROUP_ORDER
    assert prompt_builder._NEVER_DROP_TIER is prompt_sections._NEVER_DROP_TIER
