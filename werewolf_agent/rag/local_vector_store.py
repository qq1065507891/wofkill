# -*- coding: utf-8 -*-
"""
RAG 本地 TF-IDF 风格向量存储实现。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.rag.local_vector_store import LocalVectorStore
    >>> store = LocalVectorStore()
    >>> store.add("doc1", "预言家 查验", {})
    >>> store.query("预言家")
"""

from __future__ import annotations

import math
from typing import Any

from werewolf_agent.rag.query_processing import _tokenize


class LocalVectorStore:
    """Local in-memory vector store using TF-IDF-like scoring.

    Not a real embedding model — uses token overlap and IDF weighting
    for approximate similarity. Suitable for development and testing.
    For production, replace with Qdrant/pgvector adapter implementing VectorStore.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._doc_freq: dict[str, int] = {}
        self._total_docs = 0

    def _tokenize(self, text: str) -> list[str]:
        """返回当前实现使用的 tokenizer，供兼容 shim 覆盖。"""
        return _tokenize(text)

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        if doc_id in self._docs:
            self.delete(doc_id)
        tokens = self._tokenize(text)
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) if tokens else 1
        for t in tf:
            tf[t] /= total

        self._docs[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata,
            "tf": tf,
            "tokens": set(tokens),
        }
        for t in set(tokens):
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        self._total_docs += 1

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        query_tokens = set(self._tokenize(query_text))
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for doc_id, doc in self._docs.items():
            score = self._score_doc(query_tokens, doc)
            scored.append((score, doc_id))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, doc_id in scored[:top_k]:
            if score <= 0:
                continue
            results.append({
                "doc_id": doc_id,
                "text": self._docs[doc_id]["text"],
                "metadata": self._docs[doc_id]["metadata"],
                "score": score,
            })
        return results

    def delete(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        doc = self._docs[doc_id]
        for t in doc["tokens"]:
            self._doc_freq[t] = max(0, self._doc_freq.get(t, 0) - 1)
        del self._docs[doc_id]
        self._total_docs -= 1

    def count(self) -> int:
        return self._total_docs

    def _score_doc(self, query_tokens: set[str], doc: dict[str, Any]) -> float:
        """TF-IDF-like scoring: sum of (query_tf * doc_tf * idf) for shared tokens."""
        score = 0.0
        doc_tf = doc["tf"]
        for t in query_tokens:
            if t not in doc_tf:
                continue
            df = self._doc_freq.get(t, 0)
            if df == 0:
                continue
            idf = math.log((self._total_docs + 1) / (df + 1)) + 1
            score += doc_tf[t] * idf
        return score
