"""Strategy Retriever: search and rank RAG entries by query.

Priority order (design doc §9.2):
1. Same ruleset + phase + role, external high-end cases
2. External tactics (transferable patterns)
3. Project history + review (error correction)
4. Generic speech templates (avoid homogenization)

Results are ranked by relevance score and filtered by quality/visibility.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.rag.schemas import (
    CaseType,
    QualityGrade,
    RAGEntry,
    RAGHit,
    RAGQuery,
    SourceType,
    VisibilityBoundary,
)


# ---------------------------------------------------------------------------
# Quality grade ordering (higher = better)
# ---------------------------------------------------------------------------

_QUALITY_ORDER: dict[QualityGrade, int] = {
    QualityGrade.PRO_MATCH: 6,
    QualityGrade.EXPERT_REVIEW: 5,
    QualityGrade.HIGH_RANK_GAME: 4,
    QualityGrade.RULE_DERIVED_SEED: 3,
    QualityGrade.COMMUNITY_CASE: 2,
    QualityGrade.SELF_PLAY_CANDIDATE: 1,
    QualityGrade.UNREVIEWED: 0,
}

# Retrieval priority by case type (design doc §9.2)
_CASE_TYPE_PRIORITY: dict[CaseType, int] = {
    CaseType.EXTERNAL_HIGH_END_CASE: 4,
    CaseType.EXTERNAL_TACTICS: 3,
    CaseType.PROJECT_HISTORY: 2,
    CaseType.PROJECT_REVIEW: 2,
    CaseType.ROLE_STRATEGY: 1,
    CaseType.SPEECH_TEMPLATE: 0,
}


# ---------------------------------------------------------------------------
# Strategy Retriever
# ---------------------------------------------------------------------------

class StrategyRetriever:
    """Retrieves and ranks RAG entries by query criteria."""

    def __init__(self, entries: list[RAGEntry] | None = None) -> None:
        self._entries: dict[str, RAGEntry] = {}
        if entries:
            for e in entries:
                self._entries[e.entry_id] = e

    def add_entry(self, entry: RAGEntry) -> None:
        self._entries[entry.entry_id] = entry

    def add_entries(self, entries: list[RAGEntry]) -> None:
        for e in entries:
            self.add_entry(e)

    def retrieve(self, query: RAGQuery) -> list[RAGHit]:
        """Retrieve ranked RAG hits matching the query."""
        candidates = self._filter_candidates(query)
        scored = [(self._score(entry, query), entry) for entry in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RAGHit] = []
        for score, entry in scored[:query.max_results]:
            hit = self._entry_to_hit(entry, score, query)
            results.append(hit)
        return results

    def _filter_candidates(self, query: RAGQuery) -> list[RAGEntry]:
        """Filter entries by hard criteria."""
        results: list[RAGEntry] = []
        for entry in self._entries.values():
            meta = entry.metadata

            # Visibility: god-view only if explicitly requested
            if meta.visibility_boundary == VisibilityBoundary.GOD_VIEW:
                if not query.include_god_view:
                    continue

            # Ruleset filter
            if query.ruleset_id and meta.ruleset_id:
                if meta.ruleset_id != query.ruleset_id:
                    continue

            # Quality minimum
            if query.quality_min:
                if _QUALITY_ORDER.get(meta.quality_grade, 0) < _QUALITY_ORDER.get(query.quality_min, 0):
                    continue

            # Source type filter
            if query.source_types:
                if meta.source.source_type not in query.source_types:
                    continue

            # Case type filter
            if query.case_types:
                if meta.case_type not in query.case_types:
                    continue

            results.append(entry)
        return results

    def _score(self, entry: RAGEntry, query: RAGQuery) -> float:
        """Compute relevance score [0..1] for an entry."""
        score = 0.0
        meta = entry.metadata

        # Case type priority (0.0–0.3)
        score += _CASE_TYPE_PRIORITY.get(meta.case_type, 0) * 0.075

        # Quality grade bonus (0.0–0.3)
        score += _QUALITY_ORDER.get(meta.quality_grade, 0) / 20.0

        # Role match (0.15)
        if query.role and meta.role_perspective:
            if query.role == meta.role_perspective:
                score += 0.15
            elif meta.role_perspective == "general":
                score += 0.05

        # Phase match (0.1)
        if query.phase and meta.phase:
            if query.phase == meta.phase:
                score += 0.1
            elif meta.phase == "general":
                score += 0.03

        # Tag overlap (0.1)
        if query.situation:
            situation_words = set(query.situation.lower().split())
            tag_words = set(" ".join(meta.tags).lower().split())
            overlap = len(situation_words & tag_words)
            if overlap > 0:
                score += min(0.1, overlap * 0.03)

        return min(score, 1.0)

    def _entry_to_hit(self, entry: RAGEntry, score: float, query: RAGQuery) -> RAGHit:
        """Convert an entry to a retrieval hit with display annotation."""
        meta = entry.metadata
        allowed_in_live = meta.visibility_boundary in (
            VisibilityBoundary.PUBLIC_ONLY,
            VisibilityBoundary.PLAYER_PERSPECTIVE,
        )

        # Build display annotation
        source_label = meta.source.source_type.value
        quality_label = meta.quality_grade.value
        annotation = f"[{source_label}|{quality_label}]"

        return RAGHit(
            entry_id=entry.entry_id,
            title=entry.title,
            summary=entry.summary[:300],
            relevance_score=round(score, 3),
            quality_grade=meta.quality_grade,
            source_type=meta.source.source_type,
            visibility_boundary=meta.visibility_boundary,
            case_type=meta.case_type,
            role_perspective=meta.role_perspective,
            phase=meta.phase,
            key_decisions=entry.key_decisions,
            short_quotes=entry.short_quotes,
            tags=meta.tags,
            allowed_in_live_context=allowed_in_live,
            display_annotation=annotation,
        )
