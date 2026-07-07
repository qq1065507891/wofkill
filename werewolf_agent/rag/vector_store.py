# -*- coding: utf-8 -*-
"""
功能描述：向量存储稳定门面，提供工厂、协议和旧路径兼容导出。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-07
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol

from werewolf_agent.rag.embedding_vector_store import (
    _EMBEDDING_DIM,
    _NGRAM_SIZES,
    _hash_ngram,
    _text_to_embedding,
    _to_pgvector_literal,
    EmbeddingVectorStore as _SplitEmbeddingVectorStore,
    PgVectorStore as _SplitPgVectorStore,
    SiliconFlowVectorStore as _SplitSiliconFlowVectorStore,
    VectorStoreConfigError,
)
from werewolf_agent.rag.local_vector_store import LocalVectorStore as _SplitLocalVectorStore
from werewolf_agent.rag.query_processing import _tokenize


__all__ = [
    "AutoVectorStore",
    "EmbeddingVectorStore",
    "LocalVectorStore",
    "PgVectorStore",
    "SiliconFlowVectorStore",
    "VectorStore",
    "VectorStoreConfigError",
    "create_vector_store",
    "_EMBEDDING_DIM",
    "_NGRAM_SIZES",
    "_hash_ngram",
    "_text_to_embedding",
    "_to_pgvector_literal",
    "_tokenize",
]


class VectorStore(Protocol):
    """Interface for vector similarity search over text documents."""

    def add(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None: ...
    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]: ...
    def delete(self, doc_id: str) -> None: ...
    def count(self) -> int: ...


class LocalVectorStore(_SplitLocalVectorStore):
    """旧 ``vector_store`` 导入路径的兼容子类。"""

    def _tokenize(self, text: str) -> list[str]:
        # 通过旧模块全局名查找，保留 monkeypatch.setattr(vector_store, "_tokenize", ...).
        return _tokenize(text)


class EmbeddingVectorStore(_SplitEmbeddingVectorStore):
    """旧 ``vector_store`` 导入路径的兼容子类。"""

    def _text_to_embedding(self, text: str) -> list[float]:
        # 通过旧模块全局名查找，保留 monkeypatch.setattr(vector_store, "_text_to_embedding", ...).
        return _text_to_embedding(text, self._dim)


class PgVectorStore(_SplitPgVectorStore):
    """旧 ``vector_store`` 导入路径的 pgvector 兼容子类。"""

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = _EMBEDDING_DIM,
        table_name: str = "rag_vectors",
        initialize: bool = True,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._legacy_dim = dim
        super().__init__(
            dsn,
            dim=dim,
            table_name=table_name,
            initialize=initialize,
            embed_fn=embed_fn,
        )

    def _embed_text(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        # 旧路径默认 embedding 必须走 vector_store._text_to_embedding，保留 monkeypatch 语义。
        return _text_to_embedding(text, self._legacy_dim)


class SiliconFlowVectorStore(_SplitSiliconFlowVectorStore):
    """旧 ``vector_store`` 导入路径的 SiliconFlow 兼容子类。"""


class AutoVectorStore:
    """Automatically selects the best available vector store implementation."""

    def __init__(self, dim: int = _EMBEDDING_DIM) -> None:
        self._store: LocalVectorStore | EmbeddingVectorStore | SiliconFlowVectorStore
        import logging
        logger = logging.getLogger("werewolf_agent.rag.vector_store")

        try:
            from werewolf_agent.rag.embedding_client import (
                EmbeddingClientError,
                SiliconFlowEmbeddingClient,
            )
            client = SiliconFlowEmbeddingClient()
            self._store = SiliconFlowVectorStore(client)
            self._backend = "siliconflow"
            logger.info("vector backend: %s", self._backend)
            return
        except (ImportError, EmbeddingClientError) as exc:
            logger.warning(
                "SiliconFlow vector store initialization failed (%s); falling back to local/embedding store",
                exc,
            )

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


def create_vector_store(backend: str = "auto", *, initialize: bool = True) -> VectorStore:
    """Create a vector store backend by name."""
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
