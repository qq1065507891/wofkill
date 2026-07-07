# -*- coding: utf-8 -*-
"""
功能描述：按查询检索和排序 RAG 条目，协调过滤、排序 helper 与命中结果构造。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-07
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.rag import retrieval_ranking as _ranking
from werewolf_agent.rag.retrieval_ranking import (
    _CASE_TYPE_PRIORITY,
    _DISPLAY_CASE_TYPE_LABELS,
    _DISPLAY_QUALITY_LABELS,
    _DISPLAY_SOURCE_LABELS,
    _QUALITY_ORDER,
    _sigmoid,
    role_phase_matches,
)
from werewolf_agent.rag.schemas import (
    CaseType,
    QualityGrade,
    RAGEntry,
    RAGHit,
    RAGQuery,
)


logger = logging.getLogger(__name__)

__all__ = [
    "StrategyRetriever",
    "create_retriever",
    "role_phase_matches",
    "_CASE_TYPE_PRIORITY",
    "_DISPLAY_CASE_TYPE_LABELS",
    "_DISPLAY_QUALITY_LABELS",
    "_DISPLAY_SOURCE_LABELS",
    "_QUALITY_ORDER",
    "_case_type_priority",
    "_quality_priority",
    "_sigmoid",
    "_tokenize_situation",
]


def _quality_priority(grade: QualityGrade, *, entry_id: str = "") -> int:
    """旧 retriever 路径的 priority 兼容入口。"""
    return _ranking.quality_priority_from_order(
        grade,
        _QUALITY_ORDER,
        entry_id=entry_id,
    )


def _case_type_priority(case_type: CaseType, *, entry_id: str = "") -> int:
    """旧 retriever 路径的 priority 兼容入口。"""
    return _ranking.case_type_priority_from_order(
        case_type,
        _CASE_TYPE_PRIORITY,
        entry_id=entry_id,
    )


def _tokenize_situation(situation: str) -> set[str]:
    """旧 retriever 路径的 situation tokenizer 兼容入口。"""
    return _ranking._tokenize_situation(situation)


# ---------------------------------------------------------------------------
# Strategy Retriever
# ---------------------------------------------------------------------------

class StrategyRetriever:
    """Retrieves and ranks RAG entries by query criteria.

    When a reranker is provided, rule-based scoring narrows the candidate
    pool and the reranker semantically re-ranks the top candidates.

    R2: when ``vector_scores`` is provided (typically from a BGE-m3
    vector recall pass), the per-entry vector score is folded into the
    final rule-based score with weight ``merge_vector_score`` (default
    0.5). This prevents the vector signal from being discarded.
    """

    def __init__(
        self,
        entries: list[RAGEntry] | None = None,
        *,
        reranker: Any = None,
        vector_scores: dict[str, float] | None = None,
        merge_vector_score: float = 0.5,
    ) -> None:
        self._entries: dict[str, RAGEntry] = {}
        self._reranker = reranker
        # R2: vector scores keyed by entry_id; missing keys default to
        # 0.0 in the merge step so entries without a vector signal
        # simply keep their rule-based score (scaled by 1 - weight).
        self._vector_scores: dict[str, float] = dict(vector_scores or {})
        # Clamp the merge weight to [0,1]; outside that range produces
        # nonsense merged scores. 0.0 = ignore vector entirely (legacy
        # behavior); 1.0 = use vector score only.
        self._merge_vector_score: float = max(0.0, min(1.0, float(merge_vector_score)))
        if entries:
            for e in entries:
                self._entries[e.entry_id] = e

    def add_entry(self, entry: RAGEntry) -> None:
        self._entries[entry.entry_id] = entry

    def add_entries(self, entries: list[RAGEntry]) -> None:
        for e in entries:
            self.add_entry(e)

    def retrieve(self, query: RAGQuery) -> list[RAGHit]:
        """Retrieve ranked RAG hits matching the query.

        When a reranker is configured, rule-based scoring selects a wider
        candidate pool (max_results * 3), then the reranker semantically
        re-ranks the candidates.

        R2: when ``vector_scores`` was passed to ``__init__`` (typically
        by ``RAGKnowledgeService._vector_candidates``), the per-entry
        vector score is folded into the rule-based score with weight
        ``merge_vector_score``. Entries without a vector score keep
        their rule-based score scaled down by ``(1 - weight)``.
        """
        return _ranking.retrieve_ranked_hits(
            entries=list(self._entries.values()),
            query=query,
            reranker=self._reranker,
            filter_candidates_fn=self._filter_candidates,
            merged_score_fn=self._merged_score,
            entry_to_hit_fn=self._entry_to_hit,
            case_type_priority_fn=_case_type_priority,
            quality_priority_fn=_quality_priority,
        )

    def _build_rerank_query(self, query: RAGQuery) -> str:
        """Build a semantic query string for the reranker."""
        return _ranking.build_rerank_query(query)

    def _filter_candidates(self, query: RAGQuery) -> list[RAGEntry]:
        """Filter entries by hard criteria."""
        return _ranking.filter_candidates(
            list(self._entries.values()),
            query,
            quality_priority_fn=_quality_priority,
            role_phase_matches_fn=role_phase_matches,
        )

    def _score(self, entry: RAGEntry, query: RAGQuery) -> float:
        """Compute relevance score [0..1] for an entry."""
        return _ranking.score_entry(
            entry,
            query,
            case_type_priority_fn=_case_type_priority,
            quality_priority_fn=_quality_priority,
            tokenize_situation_fn=_tokenize_situation,
        )

    def _merged_score(self, entry: RAGEntry, query: RAGQuery) -> float:
        """Combine the rule-based score with the optional vector score.

        R2: when no vector score is registered for the entry, returns
        the pure rule-based score (legacy behavior).

        G-R4-05: when a vector score IS registered, return
        ``max(rule, vec)`` rather than the asymmetric
        ``(1 - w) * rule + w * vec``. The asymmetric denominator
        had two failure modes: (1) a strong vector signal was
        diluted by a moderate rule score (vec=1.0, rule=0.5
        → merged 0.75, LOWER than a rule-only with rule=0.8);
        (2) a weak vector signal unfairly suppressed a strong
        rule-only match (vec=0.2, rule=0.8 → merged 0.5).
        ``max(rule, vec)`` is the strongest signal the system
        can attest to, which is the contract we want.
        """
        return _ranking.merged_score(
            entry,
            query,
            vector_scores=self._vector_scores,
            score_fn=self._score,
        )

    def _entry_to_hit(self, entry: RAGEntry, score: float, query: RAGQuery) -> RAGHit:
        """Convert an entry to a retrieval hit with display annotation."""
        return _ranking.entry_to_hit(entry, score, query)


def create_retriever(
    entries: list[RAGEntry] | None = None,
    *,
    enable_reranker: bool = True,
) -> StrategyRetriever:
    """Create a StrategyRetriever with optional reranker auto-detected.

    When SILICONFLOW_API_KEY is set, the reranker is enabled automatically.
    Pass enable_reranker=False to force rule-based-only mode.
    """
    reranker = None
    if enable_reranker:
        try:
            from werewolf_agent.rag.reranker_client import SiliconFlowRerankerClient
            reranker = SiliconFlowRerankerClient()
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to initialize SiliconFlow reranker, continuing without it",
                exc_info=True,
            )
    return StrategyRetriever(entries, reranker=reranker)
