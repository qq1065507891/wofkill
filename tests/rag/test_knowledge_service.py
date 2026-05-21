"""RAG knowledge service tests.

Covers seed fallback, repository seeding, vector indexing, and live-safe
retrieval used by runtime player contexts.
"""

from __future__ import annotations

import pytest

from werewolf_agent.rag.schemas import RAGQuery


@pytest.fixture
def repo():
    from werewolf_agent.storage.memory_store import InMemoryGameRepository

    return InMemoryGameRepository()


def test_service_falls_back_to_seed_entries_without_repository() -> None:
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    service = RAGKnowledgeService()
    hits = service.retrieve_live_hints(RAGQuery(
        role="werewolf",
        phase="night_discussion",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        max_results=3,
    ))

    assert hits
    assert any("京城大师赛" in hit.title for hit in hits)
    assert all(hit.allowed_in_live_context for hit in hits)


def test_service_ensure_seeded_upserts_seed_entries(repo) -> None:
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    service = RAGKnowledgeService(repository=repo)

    result = service.ensure_seeded()
    loaded = repo.load_rag_entries()

    assert result["seed_count"] >= 11
    assert any(e["entry_id"] == "seed_jingcheng_wolf_god_hunt_260227" for e in loaded)


def test_ensure_seeded_indexes_seed_entries_in_vector_store(repo) -> None:
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.vector_store import LocalVectorStore

    vector_store = LocalVectorStore()
    service = RAGKnowledgeService(repository=repo, vector_store=vector_store)

    service.ensure_seeded()

    assert vector_store.count() >= 11
    results = vector_store.query("狼人夜聊 抗推预言家 神牌信息", top_k=5)
    assert any(r["doc_id"] == "seed_jingcheng_wolf_god_hunt_260227" for r in results)


def test_vector_backed_retrieval_returns_full_rag_hits(repo) -> None:
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.vector_store import LocalVectorStore

    service = RAGKnowledgeService(repository=repo, vector_store=LocalVectorStore())
    service.ensure_seeded()

    hits = service.retrieve_live_hints(RAGQuery(
        role="werewolf",
        phase="night_discussion",
        situation="抗推预言家后讨论神牌信息",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        max_results=3,
    ))

    assert any(h.entry_id == "seed_jingcheng_wolf_god_hunt_260227" for h in hits)
    assert all(h.allowed_in_live_context for h in hits)


def test_service_excludes_god_view_from_live_hints(repo) -> None:
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    service = RAGKnowledgeService(repository=repo)
    service.ensure_seeded()

    hits = service.retrieve_live_hints(RAGQuery(
        role="god_view",
        phase="review",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        max_results=10,
    ))

    assert all(hit.visibility_boundary.value != "god_view" for hit in hits)
