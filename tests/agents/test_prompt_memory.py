# -*- coding: utf-8 -*-
"""
验证 memory prompt 渲染拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/agents/test_prompt_memory.py -q
"""

from __future__ import annotations


def test_prompt_memory_methods_remain_compatibly_available() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.prompt_memory import PromptMemoryMixin
    from werewolf_agent.agents import prompt_memory, prompt_rag_memory

    assert PlayerPromptBuilder._build_private_memory_hints is PromptMemoryMixin._build_private_memory_hints
    assert PlayerPromptBuilder._build_learning_context is PromptMemoryMixin._build_learning_context
    assert PlayerPromptBuilder._build_rag_hints is PromptMemoryMixin._build_rag_hints
    assert PlayerPromptBuilder._slim_rag_hint_items is PromptMemoryMixin._slim_rag_hint_items
    assert prompt_memory.build_rag_hints is prompt_rag_memory.build_rag_hints
    assert (
        prompt_memory.slim_rag_hint_items
        is prompt_rag_memory.slim_rag_hint_items
    )
    assert (
        prompt_memory.render_rag_hint_cards
        is prompt_rag_memory.render_rag_hint_cards
    )
    assert isinstance(PromptMemoryMixin.__dict__["_slim_reflection_hints"], classmethod)
    assert (
        PlayerPromptBuilder._slim_reflection_hints.__func__
        is PromptMemoryMixin._slim_reflection_hints.__func__
    )


def test_prompt_memory_classmethod_keeps_builder_cleaner_override() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    class CustomPromptBuilder(PlayerPromptBuilder):
        @staticmethod
        def _clean_prompt_text(value, *, max_chars: int = 180) -> str:
            return f"custom:{str(value)[:max_chars]}"

    hints = [{"summary": "历史玩家 p01 的判断"}]

    assert CustomPromptBuilder._slim_reflection_hints(hints) == [
        {"summary": "custom:历史玩家 p01 的判断"}
    ]
