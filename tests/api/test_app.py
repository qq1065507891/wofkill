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


# ---------------------------------------------------------------------------
# P-A3: _persist_custom_config must surface a WARNING when the game repo
# lacks save_custom_config — otherwise the customization save succeeds
# from the caller's perspective but the record is silently dropped on
# restart. The closure inside create_app() reads the outer `repo`, so
# the test exercises the module-level impl directly.
# ---------------------------------------------------------------------------


class TestPersistCustomConfigNoOp:
    """审查 A3: 缺 save_custom_config 时必须 warn，不再静默丢数据。"""

    def test_persist_writes_when_repo_supports_it(self):
        from werewolf_agent.api.app import _persist_custom_config_impl
        from werewolf_agent.customization.repository import CustomConfigRecord

        class StubRepo:
            def __init__(self):
                self.saved = []

            def save_custom_config(self, record):
                self.saved.append(record)

        record = CustomConfigRecord(
            config_id="test_id",
            config_type="ruleset",
            raw_yaml="ruleset_id: t",
            normalized={"ruleset_id": "t"},
            validation_result={"valid": True},
            content_hash="abc",
            status="playable",
            version="1",
            maturity="validated",
            compatibility_matrix={"status": "playable"},
            diff_against_default=[],
            creator_id="mod1",
            created_at="2026-06-07T00:00:00+00:00",
            updated_at="2026-06-07T00:00:00+00:00",
        )

        repo = StubRepo()
        _persist_custom_config_impl(repo, record)

        assert len(repo.saved) == 1
        assert repo.saved[0]["config_id"] == "test_id"
        assert repo.saved[0]["config_type"] == "ruleset"

    def test_persist_warns_when_repo_lacks_method(self, caplog):
        from werewolf_agent.api.app import _persist_custom_config_impl
        from werewolf_agent.customization.repository import CustomConfigRecord

        record = CustomConfigRecord(
            config_id="test_id",
            config_type="ruleset",
            raw_yaml="",
            normalized={},
            validation_result={},
            content_hash="",
            status="draft",
            version="1",
            maturity="validated",
            compatibility_matrix={},
            diff_against_default=[],
            creator_id="mod1",
            created_at="2026-06-07T00:00:00+00:00",
            updated_at="2026-06-07T00:00:00+00:00",
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="werewolf_agent.api.app"):
            _persist_custom_config_impl(type("NoOpRepo", (), {})(), record)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, f"no WARNING logged: {[r.message for r in caplog.records]}"
        assert any("save_custom_config" in r.message for r in warnings), (
            f"WARNING should mention save_custom_config: "
            f"{[r.message for r in warnings]}"
        )
