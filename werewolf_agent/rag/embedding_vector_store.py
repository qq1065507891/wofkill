# -*- coding: utf-8 -*-
"""
RAG hash embedding 向量存储实现。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.rag.embedding_vector_store import EmbeddingVectorStore
    >>> store = EmbeddingVectorStore()
    >>> store.add("doc1", "seer strategy", {})
    >>> store.query("seer")
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

from werewolf_agent.rag.query_processing import _tokenize


_EMBEDDING_DIM = 128
_NGRAM_SIZES = (2, 3, 4)


class VectorStoreConfigError(RuntimeError):
    """Raised when an optional vector backend is requested but not configured."""


def _hash_ngram(ngram: str, dim: int) -> tuple[int, float]:
    """Hash an n-gram to a bucket index with a sign.

    Uses FNV-1a-inspired hash for deterministic mapping.
    """
    h = 2166136261
    for ch in ngram:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    bucket = h % dim
    sign = 1.0 if (h >> 16) & 1 else -1.0
    return bucket, sign


def _text_to_embedding(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """Convert text to a fixed-dimension embedding via n-gram hashing.

    CJK characters produce unigram and bigram features.
    Latin words produce character n-grams (2,3,4).
    No external model required — pure hash-based projection.
    """
    vec = [0.0] * dim
    chars = list(text.lower())

    # 从完整文本提取字符 n-gram 特征，保持原有 hash embedding 行为。
    for n in _NGRAM_SIZES:
        for i in range(len(chars) - n + 1):
            ngram = "".join(chars[i:i + n])
            bucket, sign = _hash_ngram(ngram, dim)
            vec[bucket] += sign

    # 叠加词级特征，使精确 token 命中保持更高权重。
    for token in _tokenize(text):
        bucket, sign = _hash_ngram(f"w:{token}", dim)
        vec[bucket] += sign * 2

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class EmbeddingVectorStore:
    """Vector store using hash-based embeddings with cosine similarity.

    Uses n-gram hashing for deterministic embedding vectors.
    No pre-trained model required. Falls back gracefully when numpy
    is not available (uses pure-Python dot product).

    For production, replace with sentence-transformers + Qdrant/pgvector.
    """

    def __init__(self, dim: int = _EMBEDDING_DIM) -> None:
        self._dim = dim
        self._docs: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._total_docs = 0
        self._use_numpy = False
        try:
            import numpy  # noqa: F401
            self._use_numpy = True
        except ImportError:
            pass

    def _text_to_embedding(self, text: str) -> list[float]:
        """返回当前实现使用的 embedding，供兼容 shim 覆盖。"""
        return _text_to_embedding(text, self._dim)

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        if doc_id in self._docs:
            self.delete(doc_id)
        embedding = self._text_to_embedding(text)
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata,
        }
        self._embeddings[doc_id] = embedding
        self._total_docs += 1

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        query_emb = self._text_to_embedding(query_text)
        scored: list[tuple[float, str]] = []
        for doc_id, doc_emb in self._embeddings.items():
            score = self._cosine_similarity(query_emb, doc_emb)
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
        del self._docs[doc_id]
        del self._embeddings[doc_id]
        self._total_docs -= 1

    def count(self) -> int:
        return self._total_docs

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if self._use_numpy:
            import numpy as np
            na = np.array(a, dtype=np.float32)
            nb = np.array(b, dtype=np.float32)
            dot = float(np.dot(na, nb))
            norm_a = float(np.linalg.norm(na))
            norm_b = float(np.linalg.norm(nb))
        else:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class PgVectorStore:
    """PostgreSQL pgvector-backed VectorStore."""

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = _EMBEDDING_DIM,
        table_name: str = "rag_vectors",
        initialize: bool = True,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._dsn = dsn
        if embed_fn is not None:
            probe = embed_fn("")
            self._dim = len(probe)
        else:
            self._dim = dim
        self._table_name = table_name
        self._conn: Any | None = None
        if initialize:
            self._connect()
            self._ensure_schema()

    @property
    def backend(self) -> str:
        return "pgvector"

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_text(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        return _text_to_embedding(text, self._dim)

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        embedding = self._embed_text(text)
        vector = _to_pgvector_literal(embedding)
        conn = self._connect()
        conn.execute(
            f"""
            INSERT INTO {self._table_name} (doc_id, text, metadata, embedding)
            VALUES (%s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (doc_id) DO UPDATE SET
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            (doc_id, text, json.dumps(metadata, ensure_ascii=False), vector),
        )
        conn.commit()

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = _to_pgvector_literal(self._embed_text(query_text))
        conn = self._connect()
        rows = conn.execute(
            f"""
            SELECT doc_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM {self._table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector, vector, top_k),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for doc_id, text, metadata, score in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "score": float(score),
            })
        return results

    def delete(self, doc_id: str) -> None:
        conn = self._connect()
        conn.execute(f"DELETE FROM {self._table_name} WHERE doc_id = %s", (doc_id,))
        conn.commit()

    def count(self) -> int:
        conn = self._connect()
        row = conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connect(self) -> Any:
        if self._conn is not None:
            return self._conn
        try:
            import psycopg
        except ImportError as exc:
            raise VectorStoreConfigError("psycopg is required for pgvector backend") from exc
        self._conn = psycopg.connect(self._dsn)
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({self._dim}) NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
            ON {self._table_name}
            USING ivfflat (embedding vector_cosine_ops)
            """
        )
        conn.commit()


def _to_pgvector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


class SiliconFlowVectorStore:
    """Vector store using SiliconFlow embedding API for semantic search."""

    def __init__(self, embedding_client: Any) -> None:
        self._client = embedding_client
        self._docs: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._total_docs = 0
        self._use_numpy = False
        try:
            import numpy  # noqa: F401
            self._use_numpy = True
        except ImportError:
            pass

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        if doc_id in self._docs:
            self.delete(doc_id)
        embedding = self._client.embed_single(text)
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata,
        }
        self._embeddings[doc_id] = embedding
        self._total_docs += 1

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        query_emb = self._client.embed_single(query_text)
        scored: list[tuple[float, str]] = []
        for doc_id, doc_emb in self._embeddings.items():
            score = self._cosine_similarity(query_emb, doc_emb)
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
        del self._docs[doc_id]
        del self._embeddings[doc_id]
        self._total_docs -= 1

    def count(self) -> int:
        return self._total_docs

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if self._use_numpy:
            import numpy as np
            na = np.array(a, dtype=np.float32)
            nb = np.array(b, dtype=np.float32)
            dot = float(np.dot(na, nb))
            norm_a = float(np.linalg.norm(na))
            norm_b = float(np.linalg.norm(nb))
        else:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def dimension(self) -> int:
        return self._client.dimension
