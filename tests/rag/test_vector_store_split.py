# -*- coding: utf-8 -*-
"""
验证 RAG 向量存储实现拆分后的兼容导入和基础行为。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> python -m pytest tests/rag/test_vector_store_split.py
"""

from __future__ import annotations


def test_local_vector_store_reexport_preserves_basic_query_behavior() -> None:
    from werewolf_agent.rag.local_vector_store import LocalVectorStore as SplitLocalVectorStore
    from werewolf_agent.rag.vector_store import LocalVectorStore as LegacyLocalVectorStore

    assert issubclass(LegacyLocalVectorStore, SplitLocalVectorStore)

    store = SplitLocalVectorStore()
    store.add("doc1", "预言家 查验 策略", {"role": "seer"})
    store.add("doc2", "狼人 隐藏 身份", {"role": "werewolf"})

    results = store.query("预言家 查验", top_k=2)

    assert results
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["metadata"] == {"role": "seer"}


def test_embedding_vector_store_reexport_preserves_hash_embedding_behavior() -> None:
    from werewolf_agent.rag.embedding_vector_store import (
        EmbeddingVectorStore as SplitEmbeddingVectorStore,
        _text_to_embedding as split_text_to_embedding,
    )
    from werewolf_agent.rag.vector_store import (
        EmbeddingVectorStore as LegacyEmbeddingVectorStore,
        _text_to_embedding as legacy_text_to_embedding,
    )

    assert issubclass(LegacyEmbeddingVectorStore, SplitEmbeddingVectorStore)
    assert legacy_text_to_embedding is split_text_to_embedding

    emb1 = split_text_to_embedding("预言家查验狼人")
    emb2 = split_text_to_embedding("预言家查验狼人")

    assert len(emb1) == 128
    assert emb1 == emb2
    assert any(value != 0.0 for value in emb1)

    store = SplitEmbeddingVectorStore()
    store.add("seer", "seer should check suspicious players", {"role": "seer"})
    store.add("wolf", "werewolf should hide identity", {"role": "werewolf"})

    results = store.query("seer checking", top_k=2)

    assert results
    assert results[0]["doc_id"] == "seer"


def test_legacy_vector_store_tokenize_monkeypatch_reaches_local_store(
    monkeypatch,
) -> None:
    import werewolf_agent.rag.vector_store as vector_store

    monkeypatch.setattr(vector_store, "_tokenize", lambda text: ["patched"])

    store = vector_store.LocalVectorStore()
    store.add("doc1", "alpha", {"source": "patched"})
    results = store.query("beta", top_k=1)

    assert results
    assert results[0]["doc_id"] == "doc1"


def test_legacy_vector_store_embedding_monkeypatch_reaches_embedding_store(
    monkeypatch,
) -> None:
    import werewolf_agent.rag.vector_store as vector_store

    calls: list[tuple[str, int]] = []

    def fake_embedding(text: str, dim: int = 128) -> list[float]:
        calls.append((text, dim))
        return [1.0] + [0.0] * (dim - 1)

    monkeypatch.setattr(vector_store, "_text_to_embedding", fake_embedding)

    store = vector_store.EmbeddingVectorStore(dim=4)
    store.add("doc1", "alpha", {})
    results = store.query("beta", top_k=1)

    assert results
    assert results[0]["doc_id"] == "doc1"
    assert calls == [("alpha", 4), ("beta", 4)]
