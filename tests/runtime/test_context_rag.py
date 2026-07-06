# -*- coding: utf-8 -*-
"""
验证 context RAG 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_rag.py -q
"""

from __future__ import annotations


def test_rag_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_rag

    assert context._rag_phase_for_task is context_rag._rag_phase_for_task
    assert context._normalize_legal_actions_to_tags is context_rag._normalize_legal_actions_to_tags
    assert context._inject_seed_rag_hints is context_rag._inject_seed_rag_hints
    assert context._extract_suspects is context_rag._extract_suspects
    assert context._extract_trusts is context_rag._extract_trusts
    assert context._extract_role_claim is context_rag._extract_role_claim
    assert context._extract_vote_intent is context_rag._extract_vote_intent
    assert context._first_sentence is context_rag._first_sentence
