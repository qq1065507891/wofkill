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
        # R12: case_type is now a first-class sort key above
        # quality, so an EXTERNAL_TACTICS/EXTERNAL_HIGH_END_CASE
        # entry can outrank the target EXTERNAL_TACTICS entry
        # we're looking for. Use a wider window so the test still
        # exercises the vector-backed retrieval contract (presence
        # + live-safe visibility) without depending on case_type
        # ordering.
        max_results=10,
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


# ---------------------------------------------------------------------------
# P1-G7: situation string is semantic, not raw concat
# ---------------------------------------------------------------------------


class _FakeRAGService:
    """Records the RAGQuery it receives so tests can assert on
    situation/role/phase format."""

    def __init__(self) -> None:
        self.calls: list[RAGQuery] = []

    def retrieve_live_hints(self, query, *, game_id: str = "", player_id: str = ""):
        self.calls.append(query)
        return []

    def hits_to_prompt_lines(self, hits, max_items: int = 3):
        return []


def test_situation_includes_role_phase_task() -> None:
    """P1-G7: the situation string passed to the RAG retriever is a
    semantic key=value blob, not just a space-joined concat of task
    type / phase / actions. The retriever can then tokenize on `=`
    and key the tag-overlap score on the value tokens.
    """
    from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
    from werewolf_agent.runtime.context import _inject_seed_rag_hints

    fake = _FakeRAGService()
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="seer",
        legal_actions=[ActionType.VOTE, ActionType.SPEECH],
    )
    _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    assert len(fake.calls) == 1
    situation = fake.calls[0].situation
    # The new format uses key=value tokens so the retriever can
    # separate keys from values when computing tag overlap.
    assert "role=seer" in situation
    assert "phase=day" in situation
    assert "task=speech" in situation
    assert "actions=" in situation


def test_situation_format_is_stable_under_punctuation() -> None:
    """P1-G7: the situation is a small semantic dict, not a noisy
    concat. The test asserts the format does not include the old
    "space-then-action.value" raw join (which would produce
    'speech day vote speech' with no semantic structure)."""
    from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
    from werewolf_agent.runtime.context import _inject_seed_rag_hints

    fake = _FakeRAGService()
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="seer",
        legal_actions=[ActionType.VOTE],
    )
    _inject_seed_rag_hints(
        ctx,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        rag_service=fake,
        game_id="g_test",
    )
    situation = fake.calls[0].situation
    # Must NOT be the legacy space-joined format. The new format
    # always has '=' between key and value.
    assert "=" in situation
    # And the role/phase/task tokens are well-defined substrings.
    for key in ("role", "phase", "task", "actions"):
        assert f"{key}=" in situation


# ---------------------------------------------------------------------------
# R1: RAGKnowledgeService must wire the reranker into StrategyRetriever.
#
# Before this fix, `retrieve_live_hints` built a StrategyRetriever without
# passing the reranker, so the rerank path was effectively dead code in
# production. After: the service accepts a reranker in __init__ and forwards
# it on every retrieve_live_hints call.
# ---------------------------------------------------------------------------


class _RecordingReranker:
    """Records every rerank_hits call so tests can assert the reranker
    was invoked by StrategyRetriever inside RAGKnowledgeService."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank_hits(self, *, query, documents, text_key="text", top_n=None):
        self.calls.append(
            {
                "query": query,
                "doc_count": len(documents),
                "text_key": text_key,
                "top_n": top_n,
            }
        )
        # Return the input docs unchanged but stamp a rerank_score so
        # the merge math in StrategyRetriever runs.
        results = []
        for i, d in enumerate(documents[: top_n or len(documents)]):
            out = dict(d)
            out["rerank_score"] = 0.5
            results.append(out)
        return results


def test_retrieve_live_hints_uses_reranker() -> None:
    """R1: when a reranker is injected into RAGKnowledgeService, the
    reranker.rerank_hits is invoked at least once per retrieval."""
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    reranker = _RecordingReranker()
    service = RAGKnowledgeService(reranker=reranker)
    hits = service.retrieve_live_hints(
        RAGQuery(
            role="werewolf",
            phase="night_discussion",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            max_results=3,
        )
    )

    assert hits, "service should still return hits"
    assert len(reranker.calls) >= 1, (
        f"R1: reranker.rerank_hits must be called at least once per "
        f"retrieval; got {len(reranker.calls)} calls"
    )
    assert reranker.calls[0]["doc_count"] > 0


def test_retrieve_live_hints_works_without_reranker() -> None:
    """R1 backward compat: RAGKnowledgeService instances without a reranker
    must continue to work exactly as before — no crash, hits returned via
    the rule-based path."""
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    service = RAGKnowledgeService()  # no reranker injected
    hits = service.retrieve_live_hints(
        RAGQuery(
            role="werewolf",
            phase="night_discussion",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            max_results=3,
        )
    )
    assert hits


# ---------------------------------------------------------------------------
# R9: role/phase fallback in _vector_candidates must be AND, not OR
# ---------------------------------------------------------------------------


def _make_rag_entry(
    *,
    entry_id: str,
    role_perspective: str,
    phase: str,
    ruleset_id: str = "pre_witch_hunter_idiot_mixed",
) -> "RAGEntry":
    """Build a minimal RAGEntry with the given role/phase for R9 tests."""
    from werewolf_agent.rag.schemas import (
        CaseMetadata,
        CaseType,
        QualityGrade,
        RAGEntry,
        ReviewStatus,
        SourceMetadata,
        SourceType,
        VisibilityBoundary,
    )

    return RAGEntry(
        entry_id=entry_id,
        title=entry_id,
        summary="r9 test summary",
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.COMMUNITY_CASE,
            review_status=ReviewStatus.APPROVED,
            reviewer="test",
            ruleset_id=ruleset_id,
            player_count=12,
            phase=phase,
            role_perspective=role_perspective,
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            source=SourceMetadata(source_type=SourceType.MANUAL_ENTRY),
            tags=[role_perspective, phase],
        ),
    )


def test_role_phase_fallback_is_and_not_or() -> None:
    """R9: ``_vector_candidates`` previously used parallel ``if``
    statements (OR semantics): a candidate was selected if either the
    role matched OR the phase matched. That pulled cross-role cases
    into the candidate pool — e.g. a werewolf case slipped in for a
    villager query just because the phase happened to be ``speech``.

    After the fix the fallback must use AND: BOTH role (or
    ``general`` wildcard) AND phase (or ``general`` wildcard) must
    match for the entry to be admitted via the metadata fallback.
    """
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.vector_store import LocalVectorStore

    # Three entries:
    # - wolf_speech: role=werewolf, phase=speech — should be REJECTED
    #   for a villager/speech query (role mismatch).
    # - villager_day: role=villager, phase=day — should be REJECTED
    #   for a villager/speech query (phase mismatch).
    # - villager_speech: role=villager, phase=speech — should be
    #   ACCEPTED (both match).
    wolf_speech = _make_rag_entry(
        entry_id="wolf_speech",
        role_perspective="werewolf",
        phase="speech",
    )
    villager_day = _make_rag_entry(
        entry_id="villager_day",
        role_perspective="villager",
        phase="day",
    )
    villager_speech = _make_rag_entry(
        entry_id="villager_speech",
        role_perspective="villager",
        phase="speech",
    )

    # Use a LocalVectorStore with no docs so the vector-recall path
    # returns nothing and the metadata-fallback path runs.
    service = RAGKnowledgeService(
        seed_provider=lambda: [wolf_speech, villager_day, villager_speech],
        vector_store=LocalVectorStore(),
    )

    out = service._vector_candidates(
        RAGQuery(role="villager", phase="speech", max_results=5),
        [wolf_speech, villager_day, villager_speech],
    )

    selected_ids = {entry.entry_id for _, entry in out}

    # AND semantics: wolf_speech fails role match → excluded.
    assert "wolf_speech" not in selected_ids, (
        f"R9: werewolf/speech entry leaked into villager/speech query "
        f"via OR semantics; selected={selected_ids!r}"
    )
    # AND semantics: villager_day fails phase match → excluded.
    assert "villager_day" not in selected_ids, (
        f"R9: villager/day entry leaked into villager/speech query "
        f"via OR semantics; selected={selected_ids!r}"
    )
    # The only entry that matches BOTH must be admitted.
    assert "villager_speech" in selected_ids, (
        f"R9: the role-and-phase-matching entry was rejected; "
        f"selected={selected_ids!r}"
    )


def test_role_phase_fallback_admits_general_wildcard() -> None:
    """R9: a ``role_perspective='general'`` entry must still be
    admitted by the role/phase AND check (it's the universal entry).
    Same for ``phase='general'``. The AND check must accept either
    wildcard side so the universal entries keep reaching the
    retriever."""
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.vector_store import LocalVectorStore

    general_role_speech = _make_rag_entry(
        entry_id="general_role_speech",
        role_perspective="general",
        phase="speech",
    )
    villager_general_phase = _make_rag_entry(
        entry_id="villager_general_phase",
        role_perspective="villager",
        phase="general",
    )

    service = RAGKnowledgeService(
        seed_provider=lambda: [general_role_speech, villager_general_phase],
        vector_store=LocalVectorStore(),
    )

    out = service._vector_candidates(
        RAGQuery(role="villager", phase="speech", max_results=5),
        [general_role_speech, villager_general_phase],
    )
    selected_ids = {entry.entry_id for _, entry in out}
    assert "general_role_speech" in selected_ids
    assert "villager_general_phase" in selected_ids


# ---------------------------------------------------------------------------
# G-R4-02 (P0): ``role_perspective='any'`` seeds must reach the retriever
# ---------------------------------------------------------------------------


def test_role_perspective_any_included_in_metadata_fallback() -> None:
    """G-R4-02: the metadata fallback path of ``_vector_candidates``
    previously only accepted ``role_perspective in (query.role,
    "general", "")`` — which silently dropped every seed whose
    ``role_perspective`` was ``"any"``. About 11 foundation seeds
    (金水 / 银水 / 对跳判断 / 警徽票权重, etc.) carry ``"any"`` and
    were therefore unreachable from the metadata fallback path.

    The bug only manifests when at least one entry IS in the
    ``selected`` pool (either a vector hit or another entry that
    passes the metadata filter) — once ``selected`` is non-empty,
    the function short-circuits the final "return all entries"
    fallback and the role_perspective='any' entries are dropped
    on the floor. The test therefore sets up a vector store with
    a villager role hit so the "any" entry can only reach the
    candidate pool through the metadata filter.
    """
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService
    from werewolf_agent.rag.vector_store import LocalVectorStore

    any_role_speech = _make_rag_entry(
        entry_id="any_role_speech",
        role_perspective="any",
        phase="speech",
    )
    villager_speech = _make_rag_entry(
        entry_id="villager_speech",
        role_perspective="villager",
        phase="speech",
    )
    vector_store = LocalVectorStore()
    # Seed the vector store with the villager entry so it
    # produces a real hit. ``any_role_speech`` has no vector
    # representation, so it must be admitted via the metadata
    # filter — which is the path that the bug breaks.
    vector_store.add(
        villager_speech.entry_id,
        f"{villager_speech.title}\n{villager_speech.summary}",
        {"role_perspective": "villager", "phase": "speech"},
    )

    service = RAGKnowledgeService(
        seed_provider=lambda: [any_role_speech, villager_speech],
        vector_store=vector_store,
    )

    out = service._vector_candidates(
        RAGQuery(role="villager", phase="speech", max_results=5),
        [any_role_speech, villager_speech],
    )
    selected_ids = {entry.entry_id for _, entry in out}
    # Sanity: the villager entry comes through the vector path.
    assert "villager_speech" in selected_ids, (
        f"G-R4-02 sanity: villager entry should be in candidates; "
        f"selected={selected_ids!r}"
    )
    # The "any" entry must ALSO be in the candidate pool, reached
    # via the metadata fallback. Pre-fix the filter rejected
    # role_perspective='any' so the final "return all" fallback
    # was not triggered and the entry was silently dropped.
    assert "any_role_speech" in selected_ids, (
        f"G-R4-02: role_perspective='any' seed was silently dropped "
        f"from the metadata fallback; selected={selected_ids!r}"
    )


def test_role_perspective_any_scores_same_as_general_in_rule_based() -> None:
    """G-R4-02: the rule-based ``_score`` previously rewarded
    ``role_perspective='general'`` with a +0.05 bonus over a
    non-matching role, but did not reward ``'any'`` the same way.
    After the fix, ``'any'`` must receive the same +0.05 bonus so
    universal-knowledge seeds rank at parity with the existing
    ``'general'`` universal seeds.
    """
    from werewolf_agent.rag.retriever import StrategyRetriever

    any_entry = _make_rag_entry(
        entry_id="any_entry",
        role_perspective="any",
        phase="speech",
    )
    general_entry = _make_rag_entry(
        entry_id="general_entry",
        role_perspective="general",
        phase="speech",
    )

    retriever = StrategyRetriever([any_entry, general_entry])
    query = RAGQuery(role="villager", phase="speech", max_results=5)

    score_any = retriever._score(any_entry, query)
    score_general = retriever._score(general_entry, query)
    # ``any`` and ``general`` should both contribute the wildcard
    # bonus (+0.05) when the query role does not match. Pre-fix,
    # ``any`` got 0.0 here, which put universal-knowledge seeds
    # below role-specific entries regardless of relevance.
    assert score_any == score_general, (
        f"G-R4-02: role_perspective='any' should score the same as "
        f"'general' for a non-matching query role; got "
        f"any={score_any!r} general={score_general!r}"
    )
