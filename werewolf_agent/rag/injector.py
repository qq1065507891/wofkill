"""RAG Injector: filter hits through visibility and inject into context.

Enforces:
- Visibility Policy hard boundary on RAG content
- God-view reviews only in review/moderator contexts, never live player
- Each hit annotated with source, quality, visibility for spectating
- RAG hits never contain base rules or adjudication truth
- Audit logging: every injection is traceable by game_id, player_id, phase, source, quality, visibility
"""

from __future__ import annotations

from collections import deque
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

    # R8: cap the audit log so a long-running service doesn't leak
    # memory. 1000 is plenty for live debugging (one entry per
    # inject call) and well under any reasonable heap budget.
    _AUDIT_LOG_MAXLEN = 1000

    def __init__(self, retriever: StrategyRetriever) -> None:
        self._retriever = retriever
        self._audit_log: deque[InjectionAuditRecord] = deque(
            maxlen=self._AUDIT_LOG_MAXLEN,
        )
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
        # R6: build a derived query instead of mutating the caller's
        # RAGQuery. The old code did
        # ``query.include_god_view = True/False`` which silently
        # changed the caller's state and leaked the injector's
        # internal visibility decision into unrelated call sites.
        want_god_view = injection_context in (
            InjectionContext.REVIEW,
            InjectionContext.MODERATOR,
        )
        effective_query = (
            query if query.include_god_view == want_god_view
            else query.model_copy(update={"include_god_view": want_god_view})
        )

        hits = self._retriever.retrieve(effective_query)

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

    @staticmethod
    def hits_to_context_items(
        hits: list[RAGHit],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Convert RAG hits to context items for AgentContext injection.

        Each item includes source annotation for spectating/audit. The
        returned items carry the full audit payload (relevance, quality,
        source_type, visibility, display annotation, etc.) — use this
        path when populating an audit log or a moderator/review view.

        For the live-player prompt, prefer :func:`hits_to_prompt_lines`
        (or :meth:`RAGInjector.hits_to_prompt_lines`) which strips the
        audit-only fields and keeps only title/summary/key_decisions.
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

    @staticmethod
    def hits_to_prompt_lines(
        hits: list[RAGHit],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Convert RAG hits to slim prompt lines for the live player.

        P0-G1: only title, summary, and a truncated ``key_decisions``
        list are returned. All audit-only fields (relevance, quality,
        source, visibility, display annotation) stay on the
        :class:`RAGHit` and in :attr:`audit_log` — they are dropped
        here to keep the live prompt focused on actionable takeaways.

        Use this when populating ``AgentContext.rag_hints`` for a live
        player. Use :meth:`hits_to_context_items` for audit /
        moderator / review views.
        """
        from werewolf_agent.rag.prompt_renderer import hits_to_prompt_lines
        return hits_to_prompt_lines(hits, max_items=max_items)

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
        )
