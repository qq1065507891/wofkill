# -*- coding: utf-8 -*-
"""
验证 context memory hints 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_memory_hints.py -q
"""

from __future__ import annotations


def test_memory_hint_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_memory_hints

    assert context._profile_memory_hint is context_memory_hints._profile_memory_hint
    assert context._reflection_memory_hints is context_memory_hints._reflection_memory_hints
    assert context._evidence_id_ref is context_memory_hints._evidence_id_ref
    assert context._cognition_matrix_hint is context_memory_hints._cognition_matrix_hint
    assert context.HINT_BUDGET == context_memory_hints.HINT_BUDGET
    assert context.REFLECTION_CARD_BUDGET == context_memory_hints.REFLECTION_CARD_BUDGET
