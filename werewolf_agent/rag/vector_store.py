# -*- coding: utf-8 -*-
"""
功能描述：向量存储抽象层，保留后端工厂并兼容导出本地与 embedding 实现。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-07
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Protocol

from werewolf_agent.rag.embedding_vector_store import (
    _EMBEDDING_DIM,
    _hash_ngram,
    _text_to_embedding,
    EmbeddingVectorStore as _SplitEmbeddingVectorStore,
)
from werewolf_agent.rag.local_vector_store import LocalVectorStore as _SplitLocalVectorStore
from werewolf_agent.rag.query_processing import _tokenize


class VectorStoreConfigError(RuntimeError):
    """Raised when an optional vector backend is requested but not configured."""


class VectorStore(Protocol):
    """Interface for vector similarity search over text documents.

    Implementations must support add, query, delete, and count.
    V1 uses LocalVectorStore with TF-IDF-like heuristic scoring.
    Future: Qdrant, pgvector, ChromaDB adapters.
    """

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None: ...
    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]: ...
    def delete(self, doc_id: str) -> None: ...
    def count(self) -> int: ...


class LocalVectorStore(_SplitLocalVectorStore):
    """旧 ``vector_store`` 导入路径的兼容子类。"""

    def _tokenize(self, text: str) -> list[str]:
        # 通过旧模块全局名查找，保留 monkeypatch.setattr(vector_store, "_tokenize", ...)
        # 对 LocalVectorStore.add/query 的历史影响。
        return _tokenize(text)


class EmbeddingVectorStore(_SplitEmbeddingVectorStore):
    """旧 ``vector_store`` 导入路径的兼容子类。"""

    def _text_to_embedding(self, text: str) -> list[float]:
        # 通过旧模块全局名查找，保留 monkeypatch.setattr(vector_store, "_text_to_embedding", ...)
        # 对 EmbeddingVectorStore.add/query 的历史影响。
        return _text_to_embedding(text, self._dim)


# ---------------------------------------------------------------------------
# Auto-selecting vector store
# ---------------------------------------------------------------------------


class AutoVectorStore:
    """Automatically selects the best available vector store implementation.

    Priority: SiliconFlow (real API embeddings) > EmbeddingVectorStore
    (hash-based, numpy) > LocalVectorStore (TF-IDF, pure Python).
    """

    def __init__(self, dim: int = _EMBEDDING_DIM) -> None:
        self._store: LocalVectorStore | EmbeddingVectorStore | SiliconFlowVectorStore
        import logging
        logger = logging.getLogger("werewolf_agent.rag.vector_store")

        # Try SiliconFlow first (real embeddings via API)
        try:
            from werewolf_agent.rag.embedding_client import (
                EmbeddingClientError,
                SiliconFlowEmbeddingClient,
            )
            client = SiliconFlowEmbeddingClient()
            self._store = SiliconFlowVectorStore(client)
            self._backend = "siliconflow"
            # R10: log the selected backend at INFO so operators can
            # see which path is live without having to query the
            # ``backend`` property. Previously the constructor only
            # logged WARNING on the fallback paths, which made the
            # final chosen backend invisible from logs alone.
            logger.info("vector backend: %s", self._backend)
            return
        except (ImportError, EmbeddingClientError) as exc:
            logger.warning(
                "SiliconFlow vector store initialization failed (%s); falling back to local/embedding store",
                exc,
            )

        # Fall back to hash-based embeddings (numpy-accelerated)
        try:
            import numpy  # noqa: F401
            self._store = EmbeddingVectorStore(dim=dim)
            self._backend = "embedding"
            logger.info("vector backend: %s", self._backend)
            return
        except ImportError as exc:
            logger.warning(
                "Numpy not available; falling back to TF-IDF/local store (%s)",
                exc,
            )

        # Final fallback: TF-IDF heuristic
        self._store = LocalVectorStore()
        self._backend = "tfidf"
        logger.info("vector backend: %s", self._backend)

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._store.add(doc_id, text, metadata)

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._store.query(query_text, top_k)

    def delete(self, doc_id: str) -> None:
        self._store.delete(doc_id)

    def count(self) -> int:
        return self._store.count()


class PgVectorStore:
    """PostgreSQL pgvector-backed VectorStore.

    Uses the existing deterministic hash embedding to avoid introducing an
    external embedding dependency for the database path. The table can later
    be reindexed or repopulated with provider embeddings behind the same
    VectorStore interface.
    """

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = _EMBEDDING_DIM,
        table_name: str = "rag_vectors",
        initialize: bool = True,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        # R11: callers can inject a custom ``embed_fn`` (e.g. a 1024-dim
        # SiliconFlow client) to override the default 128-dim hash
        # embedding. The injected function wins; when omitted, the
        # legacy 128-dim hash is preserved so existing deployments
        # keep working without a code change.
        self._embed_fn = embed_fn
        self._dsn = dsn
        # When an embed_fn is injected, prefer its output dimension
        # over the parameter so the schema matches the embeddings
        # actually written. We probe a single empty string to learn
        # the dim; the call cost is negligible at construction time.
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
        """R11: return an embedding vector for ``text``.

        Uses the injected ``embed_fn`` when one was provided at
        construction time (e.g. a 1024-dim SiliconFlow client);
        otherwise falls back to the legacy 128-dim hash embedding.
        This is the single embed path used by ``add`` and
        ``query`` so the schema dimension and the written vector
        always agree.
        """
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


def create_vector_store(backend: str = "auto", *, initialize: bool = True) -> VectorStore:
    """Create a vector store backend by name.

    Production backends are explicit opt-in. When requested without required
    configuration, raise a clear error instead of silently falling back.
    """
    normalized = backend.lower().strip()
    if normalized in ("auto", ""):
        return AutoVectorStore()
    if normalized in ("local", "tfidf"):
        return LocalVectorStore()
    if normalized == "embedding":
        return EmbeddingVectorStore()
    if normalized == "siliconflow":
        from werewolf_agent.rag.embedding_client import (
            EmbeddingClientError,
            SiliconFlowEmbeddingClient,
        )
        try:
            return SiliconFlowVectorStore(SiliconFlowEmbeddingClient())
        except EmbeddingClientError as exc:
            raise VectorStoreConfigError(str(exc)) from exc
    if normalized == "qdrant":
        raise VectorStoreConfigError("Qdrant is not supported; use pgvector")
    if normalized == "pgvector":
        dsn = os.getenv("PGVECTOR_DSN", "")
        if not dsn:
            raise VectorStoreConfigError("PGVECTOR_DSN is required for pgvector vector store")
        return PgVectorStore(dsn, initialize=initialize)
    raise VectorStoreConfigError(f"Unknown vector store backend: {backend}")


# ---------------------------------------------------------------------------
# SiliconFlow API-backed vector store
# ---------------------------------------------------------------------------


class SiliconFlowVectorStore:
    """Vector store using SiliconFlow embedding API for semantic search.

    Uses BAAI/bge-large-zh-v1.5 (1024-dim) via SiliconFlow API.
    Embeddings are cached in memory after first API call.
    Falls back gracefully when API is unavailable.
    """

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
