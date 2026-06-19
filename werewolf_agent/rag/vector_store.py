"""Vector store abstraction for RAG semantic retrieval.

Design doc §13: V1 can use local/heuristic retrieval first, then upgrade
to Qdrant/pgvector. This module provides the interface boundary.

Two implementations:
- LocalVectorStore: TF-IDF-like heuristic scoring (no numpy required)
- EmbeddingVectorStore: Hash-based embedding with cosine similarity (numpy)
- AutoVectorStore: selects best available implementation at runtime
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable, Protocol


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


# ---------------------------------------------------------------------------
# TF-IDF heuristic (original, no numpy)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + CJK character tokenizer."""
    tokens: list[str] = []
    for part in re.findall(r'[一-鿿]|[a-zA-Z0-9]+', text.lower()):
        tokens.append(part)
    return tokens


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

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        if doc_id in self._docs:
            self.delete(doc_id)
        tokens = _tokenize(text)
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
        query_tokens = set(_tokenize(query_text))
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


# ---------------------------------------------------------------------------
# Hash-based embedding with cosine similarity (requires numpy)
# ---------------------------------------------------------------------------


_EMBEDDING_DIM = 128
_NGRAM_SIZES = (2, 3, 4)


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

    # Character n-grams from the full text
    for n in _NGRAM_SIZES:
        for i in range(len(chars) - n + 1):
            ngram = "".join(chars[i:i + n])
            bucket, sign = _hash_ngram(ngram, dim)
            vec[bucket] += sign

    # Also tokenize and hash word-level features
    for token in _tokenize(text):
        bucket, sign = _hash_ngram(f"w:{token}", dim)
        vec[bucket] += sign * 2  # Boost word-level matches

    # Normalize
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

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        if doc_id in self._docs:
            self.delete(doc_id)
        embedding = _text_to_embedding(text, self._dim)
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
        query_emb = _text_to_embedding(query_text, self._dim)
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
