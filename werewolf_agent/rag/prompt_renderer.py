"""RAG prompt renderer — slim rendering of hits for live-player prompts.

P0-G1: The full ``hits_to_context_items()`` payload (relevance, source,
quality, visibility, display annotation, etc.) belongs in the audit log
where the moderator needs to know *why* a hit surfaced, NOT in the live
LLM prompt where it only burns context window.

P0-G2: Slim rendering strips the audit-only metadata from the live prompt
so the LLM sees only what a player can reason about: the case's title,
summary, and 2-3 key decisions. Relevance score, quality grade, source
type, visibility boundary, and display annotation stay in the audit log
on ``RAGInjector.audit_log()`` and ``RAGHit`` itself.

The slim renderer is deliberately a tiny pure function so it can be unit
tested without spinning up the retriever, injector, or any IO.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.rag.schemas import RAGHit


# Fields that must NEVER appear in a live-player prompt. Anything in this
# set is treated as audit/observability metadata only.
_FORBIDDEN_LIVE_FIELDS: frozenset[str] = frozenset({
    "relevance_score",
    "relevance",
    "quality_grade",
    "quality",
    "source_type",
    "visibility",
    "visibility_boundary",
    "display_annotation",
    "annotation",
    "allowed_in_live",
    "allowed_in_live_context",
    "case_type",
    "role_perspective",
    "tags",
    "short_quotes",
})


# Maximum number of key_decisions surfaced in the live prompt. The slim
# renderer's whole point is to give the LLM actionable takeaways without
# dumping the full entry; cap is intentionally small.
_MAX_KEY_DECISIONS_IN_PROMPT = 3


def render_hit_for_prompt(hit: RAGHit) -> dict[str, Any]:
    """Render a single RAGHit as a slim dict for the live-player prompt.

    The returned dict has at most three keys: ``title``, ``summary``,
    and ``key_decisions``. All audit metadata (relevance, quality,
    source, visibility, display_annotation) is dropped here — see
    :data:`_FORBIDDEN_LIVE_FIELDS` for the full exclusion list.

    Parameters
    ----------
    hit:
        The hit returned by ``StrategyRetriever.retrieve`` /
        ``RAGInjector.inject``.

    Returns
    -------
    dict
        A new dict with keys ``title``, ``summary``, and
        ``key_decisions`` (truncated to
        :data:`_MAX_KEY_DECISIONS_IN_PROMPT`).
    """
    return {
        "title": hit.title,
        "summary": hit.summary,
        "key_decisions": list(hit.key_decisions)[:_MAX_KEY_DECISIONS_IN_PROMPT],
    }


def hits_to_prompt_lines(
    hits: list[RAGHit],
    max_items: int = 3,
) -> list[dict[str, Any]]:
    """Render a list of RAGHits as slim prompt lines for the live player.

    Equivalent in shape to ``RAGInjector.hits_to_context_items()`` but
    drops every audit-only field. Use this when populating
    ``AgentContext.rag_hints`` for a live player; use
    ``RAGInjector.hits_to_context_items()`` when populating an audit
    log or moderator/review view.

    Parameters
    ----------
    hits:
        Hits as returned by ``RAGInjector.inject``. Caller is
        responsible for visibility filtering.
    max_items:
        Maximum number of hits to render. Default 3 matches the
        existing live-injection cap.

    Returns
    -------
    list[dict]
        A list with at most ``max_items`` slim dicts. Each dict
        contains only ``title``, ``summary``, and ``key_decisions``.
    """
    return [render_hit_for_prompt(h) for h in hits[:max_items]]


def hits_to_prompt_lines_json(
    hits: list[RAGHit],
    max_items: int = 3,
) -> str:
    """Convenience wrapper: render hits and serialize to compact JSON.

    Mirrors the ``_compact_json`` usage in
    ``PlayerPromptBuilder._build_rag_hints`` — the slim line dicts
    serialize cleanly without any nested Pydantic objects.
    """
    import json

    return json.dumps(
        hits_to_prompt_lines(hits, max_items=max_items),
        ensure_ascii=False,
        separators=(",", ":"),
    )
