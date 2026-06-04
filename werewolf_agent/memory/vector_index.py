"""Bag-of-words vector index for reflection memory.

Design doc §10.2: 经验段落进入长期**向量**记忆. This module provides a
lightweight, dependency-free bag-of-words cosine similarity index. We
deliberately avoid heavy embedding libraries (numpy, sentence-transformers)
to keep the test environment minimal; the index is pluggable — callers may
swap in a real embedding-backed implementation later.

The index is **opt-in**: ``ReflectionMemory.query`` only consults it when
the caller passes one in. Without an index, the exact-match fallback
behaves identically to the pre-P0-M6 implementation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable


# Chinese + Latin word boundary. ASCII letters and digits are kept as
# tokens; CJK runs are split per character (a pragmatic choice for a
# bag-of-words index without jieba/word-segmenter deps).
_TOKEN_RE = re.compile(r"[A-Za-z]+|[一-鿿]")


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

    def add_text(self, doc_id: str, text: str) -> None:
        """Register one document. May be called multiple times for the same id."""
        if self._finalized:
            raise RuntimeError(
                "BagOfWordsVectorIndex: cannot add_text after finalize()"
            )
        self._docs[doc_id] = _tokenize(text)

    def add_texts(self, items: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add_text(doc_id, text)

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
        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return {doc_id: 0.0 for doc_id in self._docs}
        q_tf = Counter(q_tokens)
        q_norm_sq = 0.0
        q_weights: dict[str, float] = {}
        for term, count in q_tf.items():
            if term not in self._idf:
                continue
            w = self._idf[term] * count
            q_weights[term] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 0.0
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

    def document_ids(self) -> list[str]:
        return list(self._docs.keys())

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs


__all__ = ["BagOfWordsVectorIndex"]
