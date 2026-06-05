"""Tests for the FastAPI app factory: env-driven wiring of optional
components (reranker, vector store, repository)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from werewolf_agent.api.app import create_app
from werewolf_agent.storage.memory_store import InMemoryGameRepository

# 复用 test_api.py 里的固定密钥与 AuthManager 设置
_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def _build_auth():
    from werewolf_agent.api.auth import AuthManager, AuthConfig
    return AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))


# ---------------------------------------------------------------------------
# N1: api/app.py must auto-init reranker when SILICONFLOW_API_KEY is set
#
# Before this fix, RAGKnowledgeService was constructed with
# `repository=repo, vector_store=vector_store` and no `reranker` kwarg.
# In production that meant the reranker path was dead — the SiliconFlow
# reranker (when configured) was never invoked even though the env var
# was set. The fix mirrors the auto-detect in
# `retriever.create_retriever()`: when SILICONFLOW_API_KEY is present,
# construct a SiliconFlowRerankerClient and pass it in.
# ---------------------------------------------------------------------------


def test_app_passes_reranker_when_api_key_set(monkeypatch):
    """N1: when ``SILICONFLOW_API_KEY`` is set, ``create_app`` must
    construct a :class:`SiliconFlowRerankerClient` and pass it to the
    ``RAGKnowledgeService`` so the reranker path is actually live in
    production."""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "stub-key-for-test")

    sentinel_reranker = MagicMock(name="SentinelReranker")
    # Patch the class object on the source module — the import inside
    # create_app will resolve to our sentinel.
    monkeypatch.setattr(
        "werewolf_agent.rag.reranker_client.SiliconFlowRerankerClient",
        lambda *a, **kw: sentinel_reranker,
    )

    app = create_app(
        repository=InMemoryGameRepository(),
        auth_manager=_build_auth(),
    )

    rag_service = app.state.rag_service
    assert rag_service is not None, "N1: rag_service must initialize"
    assert rag_service._reranker is sentinel_reranker, (
        "N1: RAGKnowledgeService._reranker must be the auto-constructed "
        "SiliconFlowRerankerClient when SILICONFLOW_API_KEY is set; "
        "production path was bypassing the reranker entirely."
    )


def test_app_passes_none_reranker_when_api_key_unset(monkeypatch):
    """N1 backward compat: when ``SILICONFLOW_API_KEY`` is NOT set, the
    service must get ``reranker=None`` so the rule-based-only path
    remains available (no crash, no half-initialized client)."""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    app = create_app(
        repository=InMemoryGameRepository(),
        auth_manager=_build_auth(),
    )

    rag_service = app.state.rag_service
    assert rag_service is not None
    assert rag_service._reranker is None, (
        "N1: reranker must default to None when no API key is set; "
        "got non-None reranker"
    )
