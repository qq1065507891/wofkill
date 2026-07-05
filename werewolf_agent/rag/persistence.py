# -*- coding: utf-8 -*-
"""
功能描述：RAG 条目序列化/反序列化工具，输出 JSON 兼容字典。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.rag.ingestion import validate_rag_entry_prompt_safe
from werewolf_agent.rag.schemas import RAGEntry


def save_rag_entries(source: list[RAGEntry] | Any) -> list[dict[str, Any]]:
    """Serialize RAG entries from a list, retriever, or ingester."""
    if hasattr(source, "all_entries"):
        entries = source.all_entries()
    elif hasattr(source, "_entries"):
        entries = list(source._entries.values())
    elif isinstance(source, list):
        entries = source
    else:
        entries = list(source)
    return [e.model_dump() for e in entries]


def load_rag_entries(data: list[dict[str, Any]]) -> list[RAGEntry]:
    """Deserialize RAG entries from dicts."""
    entries: list[RAGEntry] = []
    for raw in data:
        item = dict(raw)
        item.setdefault("schema_version", 1)
        entry = RAGEntry(**item)
        validate_rag_entry_prompt_safe(entry)
        entries.append(entry)
    return entries
