"""RAG hardening tests: persistence, hit audit, and vector store abstraction.

Covers Task 8 Steps 1-3:
1. RAG entries persist through storage interface (save/load round-trip)
2. Vector store abstraction interface with local implementation
3. RAG hit audit: every injection is traceable by game_id, player_id, phase, source, quality, visibility
"""

from __future__ import annotations

import os
import tempfile

import pytest

from werewolf_agent.rag.schemas import (
    CaseMetadata,
    CaseType,
    QualityGrade,
    RAGEntry,
    RAGHit,
    RAGQuery,
    ReviewStatus,
    SourceMetadata,
    SourceType,
    VisibilityBoundary,
)
from werewolf_agent.rag.ingestion import CaseIngester
from werewolf_agent.rag.retriever import StrategyRetriever
from werewolf_agent.rag.injector import RAGInjector, InjectionContext


def _make_entry(entry_id: str = "e1", role: str = "seer") -> RAGEntry:
    return RAGEntry(
        schema_version=1,
        entry_id=entry_id,
        title="Test entry",
        summary="A test strategy entry",
        key_decisions=["check wolf"],
        short_quotes=["I am the seer"],
        metadata=CaseMetadata(
            case_type=CaseType.PROJECT_HISTORY,
            quality_grade=QualityGrade.RULE_DERIVED_SEED,
            review_status=ReviewStatus.APPROVED,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            role_perspective=role,
            phase="night",
            visibility_boundary=VisibilityBoundary.PUBLIC_ONLY,
            source=SourceMetadata(source_type=SourceType.SELF_PLAY),
            tags=["seer", "strategy"],
        ),
    )


# ---------------------------------------------------------------------------
# Step 1: RAG persistence through storage interface
# ---------------------------------------------------------------------------


class TestRAGPersistence:
    def test_retriever_save_and_load_entries(self) -> None:
        from werewolf_agent.rag.persistence import save_rag_entries, load_rag_entries

        entries = [_make_entry("e1"), _make_entry("e2")]
        retriever = StrategyRetriever(entries)

        data = save_rag_entries(retriever)
        assert len(data) == 2
        assert data[0]["entry_id"] == "e1"

        loaded = load_rag_entries(data)
        assert len(loaded) == 2
        assert loaded[0].entry_id == "e1"
        assert loaded[1].entry_id == "e2"

    def test_ingester_save_and_restore(self) -> None:
        from werewolf_agent.rag.persistence import save_rag_entries, load_rag_entries

        ingester = CaseIngester()
        ingester.ingest(_make_entry("i1"))
        ingester.ingest(_make_entry("i2"))

        data = save_rag_entries(ingester)
        loaded = load_rag_entries(data)

        new_ingester = CaseIngester()
        for entry in loaded:
            new_ingester.ingest(entry)
        assert new_ingester.count() == 2

    def test_rag_entries_survive_json_roundtrip(self) -> None:
        import json
        from werewolf_agent.rag.persistence import save_rag_entries, load_rag_entries

        entries = [_make_entry("json1")]
        data = save_rag_entries(entries)
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        loaded = load_rag_entries(parsed)
        assert loaded[0].entry_id == "json1"
        assert loaded[0].metadata.role_perspective == "seer"

    def test_load_legacy_dict_without_schema_version_as_v1(self) -> None:
        from werewolf_agent.rag.persistence import load_rag_entries

        legacy = _make_entry("legacy_dict").model_dump()
        legacy.pop("schema_version", None)
        legacy.pop("tactical_frame", None)

        loaded = load_rag_entries([legacy])

        assert len(loaded) == 1
        assert loaded[0].schema_version == 1
        assert loaded[0].tactical_frame is None


# ---------------------------------------------------------------------------
# Step 2: Vector store abstraction
# ---------------------------------------------------------------------------


class TestVectorStoreAbstraction:
    def test_vector_store_interface_exists(self) -> None:
        from werewolf_agent.rag.vector_store import VectorStore

    def test_local_vector_store_add_and_query(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore

        store = LocalVectorStore()
        store.add("doc1", "预言家应该优先查验可疑玩家", {"role": "seer", "phase": "night"})
        store.add("doc2", "狼人应该隐藏身份不暴露", {"role": "werewolf", "phase": "day"})

        results = store.query("预言家查验策略", top_k=2)
        assert len(results) >= 1
        assert results[0]["doc_id"] == "doc1"

    def test_local_vector_store_returns_with_scores(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore

        store = LocalVectorStore()
        store.add("d1", "seer strategy check alignment", {"role": "seer"})
        store.add("d2", "wolf hide identity", {"role": "werewolf"})

        results = store.query("seer check", top_k=2)
        assert all("score" in r for r in results)

    def test_local_vector_store_delete(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore

        store = LocalVectorStore()
        store.add("d1", "test document", {})
        store.delete("d1")
        results = store.query("test", top_k=5)
        assert len(results) == 0

    def test_local_vector_store_count(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore

        store = LocalVectorStore()
        assert store.count() == 0
        store.add("d1", "doc one", {})
        store.add("d2", "doc two", {})
        assert store.count() == 2


# ---------------------------------------------------------------------------
# Step 3: RAG hit audit
# ---------------------------------------------------------------------------


class TestRAGHitAudit:
    def test_injector_produces_audit_record(self) -> None:
        retriever = StrategyRetriever([_make_entry("a1")])
        injector = RAGInjector(retriever)

        query = RAGQuery(role="seer", phase="night")
        hits = injector.inject(query, injection_context=InjectionContext.LIVE_PLAYER)

        audit = injector.last_audit()
        assert audit is not None
        assert len(audit.hits) >= 1
        assert audit.game_id is not None or audit.query_role == "seer"

    def test_audit_record_has_traceable_fields(self) -> None:
        retriever = StrategyRetriever([_make_entry("t1")])
        injector = RAGInjector(retriever)

        query = RAGQuery(role="seer", phase="night", situation="suspicious p2")
        injector.inject(query, injection_context=InjectionContext.LIVE_PLAYER)

        audit = injector.last_audit()
        assert audit is not None
        for hit_record in audit.hits:
            assert "entry_id" in hit_record
            assert "relevance_score" in hit_record
            assert "quality_grade" in hit_record
            assert "source_type" in hit_record
            assert "visibility_boundary" in hit_record

    def test_audit_with_game_context(self) -> None:
        retriever = StrategyRetriever([_make_entry("g1")])
        injector = RAGInjector(retriever)

        query = RAGQuery(role="seer", phase="night")
        injector.inject(
            query,
            injection_context=InjectionContext.LIVE_PLAYER,
            game_id="game_42",
            player_id="p01",
        )

        audit = injector.last_audit()
        assert audit is not None
        assert audit.game_id == "game_42"
        assert audit.player_id == "p01"
        assert audit.phase == "night"

    def test_audit_log_accumulates(self) -> None:
        retriever = StrategyRetriever([_make_entry("log1"), _make_entry("log2")])
        injector = RAGInjector(retriever)

        injector.inject(RAGQuery(role="seer", phase="night"))
        injector.inject(RAGQuery(role="werewolf", phase="night"))

        log = injector.audit_log()
        assert len(log) == 2

    def test_audit_respects_visibility_boundary(self) -> None:
        """Live player audit should only show allowed hits."""
        from werewolf_agent.rag.schemas import RAGEntry, CaseMetadata, SourceMetadata, SourceType

        god_view_entry = RAGEntry(
            schema_version=1,
            entry_id="god1",
            title="God view case",
            summary="Only visible in review mode",
            key_decisions=[],
            short_quotes=[],
            metadata=CaseMetadata(
                case_type=CaseType.PROJECT_HISTORY,
                quality_grade=QualityGrade.RULE_DERIVED_SEED,
                review_status=ReviewStatus.APPROVED,
                visibility_boundary=VisibilityBoundary.MODERATOR_ONLY,
                source=SourceMetadata(source_type=SourceType.SELF_PLAY),
            ),
        )
        public_entry = _make_entry("pub1")

        retriever = StrategyRetriever([god_view_entry, public_entry])
        injector = RAGInjector(retriever)

        hits = injector.inject(
            RAGQuery(role="seer", phase="night"),
            injection_context=InjectionContext.LIVE_PLAYER,
        )

        # Live player must not receive moderator_only hits
        for h in hits:
            assert h.visibility_boundary != VisibilityBoundary.MODERATOR_ONLY


# ---------------------------------------------------------------------------
# Embedding-based vector store
# ---------------------------------------------------------------------------


class TestEmbeddingVectorStore:
    """EmbeddingVectorStore uses hash-based embeddings with cosine similarity."""

    def test_basic_add_and_query(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("doc1", "预言家应该优先查验可疑玩家", {"role": "seer"})
        store.add("doc2", "狼人应该隐藏身份不暴露", {"role": "werewolf"})
        results = store.query("预言家查验策略", top_k=2)
        assert len(results) >= 1
        assert results[0]["doc_id"] == "doc1"

    def test_semantic_similarity_english(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("d1", "seer should check suspicious players at night", {"role": "seer"})
        store.add("d2", "werewolf should hide identity during day", {"role": "werewolf"})
        store.add("d3", "witch has antidote and poison", {"role": "witch"})
        results = store.query("seer checking strategy", top_k=3)
        assert len(results) >= 1
        # seer document should rank higher than witch
        seer_rank = next((i for i, r in enumerate(results) if r["doc_id"] == "d1"), 999)
        witch_rank = next((i for i, r in enumerate(results) if r["doc_id"] == "d3"), 999)
        assert seer_rank < witch_rank

    def test_cosine_scores_between_minus_one_and_one(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("d1", "hello world test document", {})
        results = store.query("hello", top_k=1)
        if results:
            assert -1.0 <= results[0]["score"] <= 1.0

    def test_delete_and_query(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("d1", "test document one", {})
        store.add("d2", "test document two", {})
        store.delete("d1")
        results = store.query("test", top_k=5)
        assert all(r["doc_id"] != "d1" for r in results)
        assert store.count() == 1

    def test_count(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        assert store.count() == 0
        store.add("d1", "doc one", {})
        store.add("d2", "doc two", {})
        assert store.count() == 2

    def test_empty_query_returns_empty(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        results = store.query("anything", top_k=5)
        assert results == []

    def test_overwrite_on_duplicate_id(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("dup", "version one", {})
        store.add("dup", "version two", {})
        assert store.count() == 1
        results = store.query("version two", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "version two"

    def test_cjk_ngram_hashing(self) -> None:
        """CJK text produces valid embeddings via n-gram hashing."""
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore, _text_to_embedding
        emb = _text_to_embedding("预言家查验狼人")
        assert len(emb) == 128  # default dimension
        assert any(v != 0.0 for v in emb)  # non-zero embedding

    def test_embedding_deterministic(self) -> None:
        """Same text always produces same embedding."""
        from werewolf_agent.rag.vector_store import _text_to_embedding
        e1 = _text_to_embedding("test text for determinism")
        e2 = _text_to_embedding("test text for determinism")
        assert e1 == e2


class TestAutoVectorStore:
    """AutoVectorStore selects best available backend."""

    def test_auto_selects_embedding_when_numpy_available(self) -> None:
        from werewolf_agent.rag.vector_store import AutoVectorStore
        store = AutoVectorStore()
        # numpy is available in test env
        assert store.backend == "embedding"

    def test_auto_store_full_workflow(self) -> None:
        from werewolf_agent.rag.vector_store import AutoVectorStore
        store = AutoVectorStore()
        store.add("d1", "预言家查验策略", {"role": "seer"})
        store.add("d2", "狼人隐藏身份", {"role": "werewolf"})
        results = store.query("预言家", top_k=2)
        assert len(results) >= 1
        assert store.count() == 2

    def test_auto_delete_and_count(self) -> None:
        from werewolf_agent.rag.vector_store import AutoVectorStore
        store = AutoVectorStore()
        store.add("d1", "test", {})
        store.add("d2", "test2", {})
        store.delete("d1")
        assert store.count() == 1


class TestProductionVectorStoreBoundary:
    def test_create_vector_store_local_backend(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore, create_vector_store

        store = create_vector_store("local")

        assert isinstance(store, LocalVectorStore)

    def test_create_vector_store_auto_backend(self) -> None:
        from werewolf_agent.rag.vector_store import AutoVectorStore, create_vector_store

        store = create_vector_store("auto")

        assert isinstance(store, AutoVectorStore)

    def test_qdrant_backend_requires_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.rag.vector_store import VectorStoreConfigError, create_vector_store

        monkeypatch.delenv("QDRANT_URL", raising=False)

        with pytest.raises(VectorStoreConfigError, match="Qdrant is not supported"):
            create_vector_store("qdrant")

    def test_pgvector_backend_requires_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.rag.vector_store import VectorStoreConfigError, create_vector_store

        monkeypatch.delenv("PGVECTOR_DSN", raising=False)

        with pytest.raises(VectorStoreConfigError, match="PGVECTOR_DSN"):
            create_vector_store("pgvector")

    def test_siliconflow_backend_requires_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.rag.vector_store import VectorStoreConfigError, create_vector_store

        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

        with pytest.raises(VectorStoreConfigError, match="SILICONFLOW_API_KEY"):
            create_vector_store("siliconflow")

    def test_qdrant_backend_is_not_supported(self) -> None:
        from werewolf_agent.rag.vector_store import VectorStoreConfigError, create_vector_store

        with pytest.raises(VectorStoreConfigError, match="Qdrant is not supported"):
            create_vector_store("qdrant")

    def test_pgvector_backend_uses_configured_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.rag.vector_store import PgVectorStore, create_vector_store

        monkeypatch.setenv("PGVECTOR_DSN", "postgresql://wofkill:wofkill-dev@localhost:5432/wofkill")

        store = create_vector_store("pgvector", initialize=False)

        assert isinstance(store, PgVectorStore)
        assert store.backend == "pgvector"


class TestEmbeddingVsTFIDFRanking:
    """Verify embedding store produces meaningful ranking differences."""

    def test_embedding_ranks_relevant_higher(self) -> None:
        from werewolf_agent.rag.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add("seer1", "预言家第一天起跳给查杀", {"role": "seer", "action": "claim"})
        store.add("wolf1", "狼人悍跳预言家发假查杀", {"role": "werewolf", "action": "deep_hook"})
        store.add("witch1", "女巫首夜救人与毒人策略", {"role": "witch", "action": "potion"})
        store.add("hybrid1", "混血儿绑定主人存活策略", {"role": "hybrid", "action": "survive"})

        # Query about seer strategy — seer doc should rank first or second
        results = store.query("预言家起跳查杀策略", top_k=4)
        if len(results) >= 2:
            top_ids = [r["doc_id"] for r in results[:2]]
            assert "seer1" in top_ids, f"seer1 not in top 2: {top_ids}"

    def test_tfidf_ranks_relevant_higher(self) -> None:
        from werewolf_agent.rag.vector_store import LocalVectorStore
        store = LocalVectorStore()
        store.add("seer1", "预言家第一天起跳给查杀", {"role": "seer"})
        store.add("wolf1", "狼人悍跳预言家发假查杀", {"role": "werewolf"})
        store.add("witch1", "女巫首夜救人与毒人策略", {"role": "witch"})

        results = store.query("预言家查杀", top_k=3)
        if len(results) >= 2:
            top_ids = [r["doc_id"] for r in results[:2]]
            # Either seer or wolf should be top (both mention 预言家/查杀)
            assert "seer1" in top_ids or "wolf1" in top_ids


# ---------------------------------------------------------------------------
# PgVectorStore mocked unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_psycopg(monkeypatch: pytest.MonkeyPatch):
    """Inject a mock psycopg module via sys.modules so import psycopg succeeds."""
    import sys
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []

    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    monkeypatch.setitem(sys.modules, "psycopg", mock_psycopg)

    return mock_psycopg, mock_conn, mock_cursor


class TestPgVectorStore:
    """Unit tests for PgVectorStore with mocked psycopg connection."""

    def _make_store(self, _mock_psycopg, **kwargs):
        from werewolf_agent.rag.vector_store import PgVectorStore
        return PgVectorStore("postgresql://test:test@localhost:5432/testdb", **kwargs)

    # -- add --

    def test_add_inserts_embedding(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store.add("doc1", "test text", {"role": "seer"})
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    def test_add_upsert_on_conflict(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store.add("doc1", "test text", {"role": "seer"})
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql_arg

    # -- query --

    def test_query_returns_results(self, _mock_psycopg):
        _, mock_conn, mock_cursor = _mock_psycopg
        mock_cursor.fetchall.return_value = [
            ("doc1", "some text", '{"role": "seer"}', 0.95),
        ]
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        results = store.query("seer strategy")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"
        assert results[0]["text"] == "some text"
        assert results[0]["score"] == 0.95

    def test_query_parses_string_metadata(self, _mock_psycopg):
        _, mock_conn, mock_cursor = _mock_psycopg
        mock_cursor.fetchall.return_value = [
            ("d1", "text", '{"role": "seer", "phase": "night"}', 0.8),
        ]
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        results = store.query("test")
        assert results[0]["metadata"] == {"role": "seer", "phase": "night"}

    def test_query_empty_returns_empty(self, _mock_psycopg):
        _, mock_conn, mock_cursor = _mock_psycopg
        mock_cursor.fetchall.return_value = []
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        results = store.query("nothing matches")
        assert results == []

    # -- delete --

    def test_delete_removes_entry(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store.delete("doc1")
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    # -- count --

    def test_count_returns_number(self, _mock_psycopg):
        _, mock_conn, mock_cursor = _mock_psycopg
        mock_cursor.fetchone.return_value = (42,)
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        assert store.count() == 42

    def test_count_returns_zero(self, _mock_psycopg):
        _, mock_conn, mock_cursor = _mock_psycopg
        mock_cursor.fetchone.return_value = (0,)
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        assert store.count() == 0

    # -- close --

    def test_close_closes_connection(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store.close()
        mock_conn.close.assert_called_once()

    def test_close_idempotent(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store.close()
        store.close()  # second call must not raise
        assert mock_conn.close.call_count == 1

    # -- schema --

    def test_schema_creates_table_and_index(self, _mock_psycopg):
        _, mock_conn, _ = _mock_psycopg
        store = self._make_store(_mock_psycopg, initialize=False)
        store._connect()
        store._ensure_schema()
        all_sql = " ".join(call.args[0] for call in mock_conn.execute.call_args_list)
        assert "rag_vectors" in all_sql
        assert "ivfflat" in all_sql

    # -- error handling --

    def test_psycopg_missing_raises_error(self, monkeypatch: pytest.MonkeyPatch):
        from werewolf_agent.rag.vector_store import PgVectorStore, VectorStoreConfigError

        # Remove psycopg from the vector_store module level if present
        import werewolf_agent.rag.vector_store as _vs
        monkeypatch.delattr(_vs, "psycopg", raising=False)

        # Make import of psycopg fail
        import builtins
        orig_import = builtins.__import__

        def _block_psycopg(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("no psycopg")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_psycopg)

        store = PgVectorStore("postgresql://x:x@localhost/db", initialize=False)
        with pytest.raises(VectorStoreConfigError, match="psycopg"):
            store._connect()

    # -- properties --

    def test_backend_property(self, _mock_psycopg):
        store = self._make_store(_mock_psycopg, initialize=False)
        assert store.backend == "pgvector"

    def test_dimension_property(self, _mock_psycopg):
        store = self._make_store(_mock_psycopg, dim=256, initialize=False)
        assert store.dimension == 256
