# -*- coding: utf-8 -*-
"""
功能描述：轻量级词袋向量索引——为反思记忆提供无依赖的余弦相似度检索，可插拔替换为真实嵌入后端。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


# Chinese + Latin word boundary. ASCII letters and digits are kept as
# tokens; CJK runs are split per character (a pragmatic choice for a
# bag-of-words index without jieba/word-segmenter deps).
#
# MEM-NEW-4: the regex must cover THREE CJK blocks, not just the
# Unified Ideographs block. The pre-fix ``[一-鿿]`` (U+4E00-9FFF)
# silently dropped CJK Extension A (U+3400-4DBF) and CJK
# Compatibility Ideographs (U+F900-FAFF) — both are real characters
# in Chinese text (older / less-frequent hanzi, and round-trip
# compatibility mappings). A reflection containing one of those
# characters was tokenized into zero CJK tokens and the cosine
# similarity collapsed to 0.
_TOKEN_RE = re.compile(r"[A-Za-z]+|[㐀-䶿一-鿿豈-﫿]")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BagOfWordsVectorIndex:
    """Tiny bag-of-words TF cosine similarity index.

    Usage:
        idx = BagOfWordsVectorIndex()
        idx.add_text("r1", "上次站边预言家被冲爆,票投错了")
        idx.add_text("r2", "金水轻信核对查验")
        idx.finalize()
        scores = idx.similarity("站边 预言家 票型")
        # scores = {"r1": 0.42, "r2": 0.08}
    """

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._norms: dict[str, float] = {}
        self._finalized = False
        # MEM-18: per-query cache. The hot path (reflection
        # retrieval) often re-queries the same string many times in a
        # single game, and the q_norm / q_weights computation is
        # pure — it depends only on the query string and the
        # (finalized) IDF table. Cache by q_str; invalidate on
        # finalize() because the IDF table may change.
        self._query_cache: dict[str, dict[str, Any]] = {}

    def add_text(self, doc_id: str, text: str) -> None:
        """Register one document. May be called multiple times for the same id.

        MEM-14: if the index has been finalized, the cached IDF and
        norms are stale (they don't reflect this new doc). The legacy
        behavior was to raise; the post-fix behavior is to invalidate
        the cache so the next similarity() call re-finalizes with
        the new docs included.
        """
        if self._finalized:
            # Invalidate the cached stats so the next similarity()
            # call re-finalizes from the current ``_docs``. This is
            # safer than raising — the caller can still add more
            # texts and call finalize/similarity when ready, and
            # their results will be correct.
            self._finalized = False
            self._idf = {}
            self._norms = {}
        self._docs[doc_id] = _tokenize(text)

    def finalize(self) -> None:
        """Lock the index and precompute IDF + L2 norms."""
        if self._finalized:
            return
        n = max(1, len(self._docs))
        for tokens in self._docs.values():
            for term in set(tokens):
                self._df[term] += 1
        # Standard smoothed IDF.
        for term, df in self._df.items():
            self._idf[term] = math.log((1 + n) / (1 + df)) + 1.0
        for doc_id, tokens in self._docs.items():
            self._norms[doc_id] = self._vector_norm(tokens)
        self._finalized = True
        # MEM-18: the IDF table may have changed; cached q_weights
        # reference the old IDF, so the query cache is no longer
        # valid.
        self._query_cache.clear()

    def _vector_norm(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        sq = 0.0
        for term, count in tf.items():
            w = self._idf.get(term, 0.0) * count
            sq += w * w
        return math.sqrt(sq) if sq > 0 else 0.0

    def similarity(self, query_text: str) -> dict[str, float]:
        """Return cosine similarity scores for every document. Empty query → all zeros."""
        if not self._finalized:
            self.finalize()
        if not self._docs:
            return {}
        # MEM-18: cache the per-query work (q_tokens, q_norm, q_weights)
        # by query string. The hot path (cross-game reflection query)
        # re-queries the same string many times in one game; the cache
        # makes the second-and-onwards calls O(|docs|) instead of
        # O(|docs| + |query|). The cache is invalidated by finalize()
        # because the IDF table may have changed.
        cached = self._query_cache.get(query_text)
        if cached is not None:
            # MEM-NEW-5: short-circuit on the empty-result sentinel
            # so repeated empty-query calls are O(1) lookups (no
            # rebuild, no allocs). Pre-fix the empty_result was
            # stored in the cache but never read — every call
            # fell through to the dot-product loop's empty branch
            # and rebuilt a fresh dict.
            empty = cached.get("empty_result")
            if empty is not None:
                return empty
            q_norm = cached["q_norm"]
            q_weights = cached["q_weights"]
        else:
            q_tokens = _tokenize(query_text)
            if not q_tokens:
                empty = {doc_id: 0.0 for doc_id in self._docs}
                # Still cache the empty result so repeated empty-query
                # calls are also O(1) lookups.
                self._query_cache[query_text] = {
                    "q_norm": 0.0,
                    "q_weights": {},
                    "empty_result": empty,
                }
                return empty
            q_tf = Counter(q_tokens)
            q_norm_sq = 0.0
            q_weights = {}
            for term, count in q_tf.items():
                if term not in self._idf:
                    continue
                w = self._idf[term] * count
                q_weights[term] = w
                q_norm_sq += w * w
            q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 0.0
            self._query_cache[query_text] = {
                "q_norm": q_norm,
                "q_weights": q_weights,
            }
        # If the cached entry is the "empty result" sentinel, return it
        # without recomputing. Otherwise fall through to the dot-product
        # loop, which depends on per-doc state and is not cached (it
        # would be a deeper change with diminishing returns).
        if not q_weights and q_norm == 0.0:
            return {doc_id: 0.0 for doc_id in self._docs}
        scores: dict[str, float] = {}
        for doc_id, doc_tokens in self._docs.items():
            doc_norm = self._norms.get(doc_id, 0.0)
            if doc_norm == 0.0 or q_norm == 0.0:
                scores[doc_id] = 0.0
                continue
            doc_tf = Counter(doc_tokens)
            dot = 0.0
            for term, w_q in q_weights.items():
                if term in doc_tf:
                    dot += w_q * (self._idf[term] * doc_tf[term])
            scores[doc_id] = dot / (q_norm * doc_norm)
        return scores

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs


__all__ = ["BagOfWordsVectorIndex"]
