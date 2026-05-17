"""RAG Injector: filter hits through visibility and inject into context.

Enforces:
- Visibility Policy hard boundary on RAG content
- God-view reviews only in review/moderator contexts, never live player
- Each hit annotated with source, quality, visibility for spectating
- RAG hits never contain base rules or adjudication truth
- Audit logging: every injection is traceable by game_id, player_id, phase, source, quality, visibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.agents.schemas import AgentContext
from werewolf_agent.rag.schemas import (
    RAGHit,
    RAGQuery,
    VisibilityBoundary,
)
from werewolf_agent.rag.retriever import StrategyRetriever


# ---------------------------------------------------------------------------
# RAG injection context modes
# ---------------------------------------------------------------------------

class InjectionContext(str):
    LIVE_PLAYER = "live_player"
    REVIEW = "review"
    MODERATOR = "moderator"
    SPECTATOR = "spectator"


# ---------------------------------------------------------------------------
# RAG hit audit record
# ---------------------------------------------------------------------------

@dataclass
class InjectionAuditRecord:
    """Audit record for a single RAG injection call."""
    game_id: str | None = None
    player_id: str | None = None
    phase: str | None = None
    query_role: str | None = None
    injection_context: str | None = None
    hits: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG Injector
# ---------------------------------------------------------------------------

class RAGInjector:
    """Injects RAG hits into agent context with visibility enforcement."""

    def __init__(self, retriever: StrategyRetriever) -> None:
        self._retriever = retriever
        self._audit_log: list[InjectionAuditRecord] = []
        self._last_audit: InjectionAuditRecord | None = None

    def inject(
        self,
        query: RAGQuery,
        injection_context: str = InjectionContext.LIVE_PLAYER,
        *,
        game_id: str | None = None,
        player_id: str | None = None,
    ) -> list[RAGHit]:
        """Retrieve and filter RAG hits for the given context."""
        # God-view only allowed in review/moderator
        if injection_context in (InjectionContext.REVIEW, InjectionContext.MODERATOR):
            query.include_god_view = True
        else:
            query.include_god_view = False

        hits = self._retriever.retrieve(query)

        # Filter by injection context
        if injection_context == InjectionContext.LIVE_PLAYER:
            hits = [h for h in hits if h.allowed_in_live_context]
        elif injection_context == InjectionContext.SPECTATOR:
            hits = [
                h for h in hits
                if h.visibility_boundary in (
                    VisibilityBoundary.PUBLIC_ONLY,
                    VisibilityBoundary.PLAYER_PERSPECTIVE,
                )
            ]
        # review and moderator see everything

        # Build audit record
        audit = InjectionAuditRecord(
            game_id=game_id,
            player_id=player_id,
            phase=query.phase,
            query_role=query.role,
            injection_context=injection_context,
            hits=[
                {
                    "entry_id": h.entry_id,
                    "relevance_score": h.relevance_score,
                    "quality_grade": h.quality_grade.value,
                    "source_type": h.source_type.value,
                    "visibility_boundary": h.visibility_boundary.value,
                    "case_type": h.case_type.value if h.case_type else None,
                }
                for h in hits
            ],
        )
        self._last_audit = audit
        self._audit_log.append(audit)

        return hits

    def last_audit(self) -> InjectionAuditRecord | None:
        return self._last_audit

    def audit_log(self) -> list[InjectionAuditRecord]:
        return list(self._audit_log)

    def hits_to_context_items(
        self,
        hits: list[RAGHit],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Convert RAG hits to context items for AgentContext injection.

        Each item includes source annotation for spectating/audit.
        """
        items: list[dict[str, Any]] = []
        for hit in hits[:max_items]:
            items.append({
                "type": "rag_hit",
                "entry_id": hit.entry_id,
                "title": hit.title,
                "summary": hit.summary,
                "key_decisions": hit.key_decisions,
                "relevance": hit.relevance_score,
                "quality": hit.quality_grade.value,
                "source_type": hit.source_type.value,
                "visibility": hit.visibility_boundary.value,
                "annotation": hit.display_annotation,
                "allowed_in_live": hit.allowed_in_live_context,
            })
        return items

    def build_rag_query(
        self,
        role: str,
        phase: str,
        situation: str = "",
        ruleset_id: str = "pre_witch_hunter_idiot_mixed",
        persona_style: str = "",
        max_results: int = 5,
    ) -> RAGQuery:
        """Build a RAGQuery from game context."""
        return RAGQuery(
            role=role,
            phase=phase,
            situation=situation,
            ruleset_id=ruleset_id,
            persona_style=persona_style,
            max_results=max_results,
            viewer_role=role,
        )
