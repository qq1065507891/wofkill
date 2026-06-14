"""RAG prompt renderer — slim rendering of hits for live-player prompts.

P0-G1: The full ``hits_to_context_items()`` payload (relevance, source,
quality, visibility, display annotation, etc.) belongs in the audit log
where the moderator needs to know *why* a hit surfaced, NOT in the live
LLM prompt where it only burns context window.

P0-G2: Slim rendering strips the audit-only metadata from the live prompt
so the LLM sees only what a player can reason about: the case's title
and prompt-safe V2 tactical frame. Legacy summary/key_decisions,
relevance score, quality grade, source type, visibility boundary, and
display annotation stay in the audit log on ``RAGInjector.audit_log()``
and ``RAGHit`` itself.

P1-G5: 3 RAG hits may be near-duplicates (same tactic, different
framing). The slim path now runs ``dedup_hits_by_similarity`` before
rendering so the LLM never sees two hits covering the same idea.

The slim renderer is deliberately a tiny pure function so it can be unit
tested without spinning up the retriever, injector, or any IO.
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.rag.schemas import RAGHit
from werewolf_agent.rag.tactical_text import (
    build_rag_retrieval_text,
    prompt_safe_tactical_frame_dict,
)


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
    "summary",
    "key_decisions",
})


# Legacy compatibility constant. RAG V2 live prompts no longer render
# ``key_decisions`` directly; the cap remains only so older imports do
# not break while legacy fields continue to exist on RAGHit/audit data.
_MAX_KEY_DECISIONS_IN_PROMPT = 3


# P2-6: single source of truth for the live-prompt RAG cap.  Three
# places previously hard-coded ``3``:
#   - context.py:287 (retriever max_results)
#   - context.py:298 (slim-renderer max_items)
#   - prompt_builder.py:737 and prompt_builder.py:794 (slice [:3])
# Drift between any of these would let a stray case slip through
# past the LLM-visible cap.  Import this constant in all 3 sites.
# The cap is intentionally small — the LLM only needs 2-3 distinct
# reference cases per turn; more inflates the prompt budget without
# changing decisions.
RAG_LIVE_PROMPT_CAP = 3


# P1-G5: Jaccard threshold for "near-duplicate" RAG hits. 0.6 is a
# reasonable middle ground for tokenized Chinese: two cases covering
# the same tactic typically share >60% of their retrieval-text tokens.
_DEDUP_DEFAULT_SIMILARITY_THRESHOLD = 0.6

# P1-G5: cap on the live-prompt hit list. Lower than the retriever's
# max_results=3 so that even when the retriever surfaces 3, we keep
# prompt density high (2 distinct tactics beat 2 variants of 1 tactic).
_DEDUP_DEFAULT_MAX_ITEMS = 2


def _tokenize(text: str) -> set[str]:
    """Tokenize text for Jaccard similarity.

    Treats every CJK character as its own token (no Chinese word
    segmentation dependency) and falls back to whitespace-split for
    Latin / number tokens. This is a deliberately crude heuristic — the
    plan calls for similarity to drop obvious near-duplicates, not
    semantically cluster cases.
    """
    if not text:
        return set()
    # Split each CJK char into its own token, keep Latin words intact.
    tokens: set[str] = set()
    for piece in re.findall(r"[A-Za-z0-9_]+|[一-鿿]", text):
        if piece:
            tokens.add(piece.lower())
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets. Returns 0.0 when both
    are empty (no signal)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedup_hits_by_similarity(
    hits: list[RAGHit],
    *,
    max_items: int = _DEDUP_DEFAULT_MAX_ITEMS,
    similarity_threshold: float = _DEDUP_DEFAULT_SIMILARITY_THRESHOLD,
) -> list[RAGHit]:
    """Drop near-duplicate RAG hits, then cap the list at ``max_items``.

    P1-G5: when 3 hits cover the same tactic (different framing), 2 of
    them are wasted context window. Two hits are "near duplicates" when
    their retrieval-text token Jaccard similarity exceeds
    ``similarity_threshold`` (default 0.6). Legacy no-frame hits also
    need title+summary overlap so shared fallback/key-decision text does
    not collapse unrelated old entries. When a duplicate pair is found,
    the higher-relevance hit wins.

    G-R4-09: the cap is now caller-controlled. The previous behavior
    silently capped the returned list at the module default
    (``_DEDUP_DEFAULT_MAX_ITEMS=2``) regardless of the caller's
    ``max_items`` — so ``hits_to_prompt_lines(max_items=3)`` was a
    no-op (always returned 2). The cap is now the caller's
    ``max_items``; the module default only applies when the caller
    does not pass it. Callers that want the old density target
    should pass ``max_items=2`` explicitly.

    Parameters
    ----------
    hits:
        Hits to dedup. Caller is responsible for any prior ordering
        (typically already ranked by the retriever).
    max_items:
        Upper bound on the returned list. Defaults to
        :data:`_DEDUP_DEFAULT_MAX_ITEMS` (2) for backward
        compatibility. Callers can request more (e.g. 3) and get
        that many distinct hits.
    similarity_threshold:
        Jaccard threshold in [0.0, 1.0]. Default 0.6.

    Returns
    -------
    list[RAGHit]
        A new list with at most ``max_items`` hits. Order is
        preserved from the input (which is already relevance-ordered
        by the retriever); only the lower-relevance member of a
        near-duplicate pair is dropped.
    """
    if not hits:
        return []
    # G-R4-09: caller-controlled cap. The module default only applies
    # via the function's default value, not as an implicit ceiling.
    effective_cap = max(int(max_items), 0)
    token_cache: list[set[str]] = [
        _tokenize(build_rag_retrieval_text(h, max_chars=1500)) for h in hits
    ]
    legacy_token_cache: list[set[str] | None] = [
        _tokenize(f"{h.title} {h.summary}") if h.tactical_frame is None else None
        for h in hits
    ]
    kept: list[RAGHit] = []
    kept_tokens: list[set[str]] = []
    kept_legacy_tokens: list[set[str] | None] = []
    for hit, tokens, legacy_tokens in zip(hits, token_cache, legacy_token_cache):
        # Walk the kept list, drop the first near-duplicate we find.
        merged = False
        for i, k_tokens in enumerate(kept_tokens):
            is_duplicate = _jaccard(tokens, k_tokens) > similarity_threshold
            if (
                is_duplicate
                and legacy_tokens is not None
                and kept_legacy_tokens[i] is not None
            ):
                is_duplicate = (
                    _jaccard(legacy_tokens, kept_legacy_tokens[i])
                    > similarity_threshold
                )
            if is_duplicate:
                # Near-duplicate. Keep the higher-relevance one.
                if hit.relevance_score > kept[i].relevance_score:
                    kept[i] = hit
                    kept_tokens[i] = tokens
                    kept_legacy_tokens[i] = legacy_tokens
                merged = True
                break
        if not merged:
            kept.append(hit)
            kept_tokens.append(tokens)
            kept_legacy_tokens.append(legacy_tokens)
    return kept[:effective_cap]


def render_hit_for_prompt(hit: RAGHit) -> dict[str, Any]:
    """Render a single RAGHit as a V2 prompt-safe live-player dict.

    The returned dict contains ``type``, ``title``, and prompt-safe V2
    tactical-frame fields. Legacy ``summary`` / ``key_decisions`` and
    audit metadata (relevance, quality, source, visibility,
    display_annotation, quotes) are dropped here — see
    :data:`_FORBIDDEN_LIVE_FIELDS` for the full exclusion list.

    Parameters
    ----------
    hit:
        The hit returned by ``StrategyRetriever.retrieve`` /
        ``RAGInjector.inject``.

    Returns
    -------
    dict
        A new dict with ``type``, ``title``, and prompt-safe tactical
        frame fields.
    """
    return {
        # R3: the ``type`` discriminator is what
        # ``runtime.context._inject_seed_rag_hints`` uses to clear
        # previous rag_hit slim items between turns. Without it, the
        # filter ``[item for item in ctx.rag_hints if item.get("type")
        # != "rag_hit"]`` is a no-op and old slim items accumulate.
        "type": "rag_hit",
        "title": hit.title,
        **prompt_safe_tactical_frame_dict(hit),
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

    P1-G5: Before rendering, dedup near-duplicate hits via
    ``dedup_hits_by_similarity`` (Jaccard on retrieval-text tokens,
    default threshold 0.6, cap 2 hits). The caller-supplied
    ``max_items`` is preserved as an upper bound — if the caller asks
    for 3 and the dedup cap is 2, dedup wins; if the caller asks for
    1, that wins. ``max_items=3`` (default) is the live-prompt
    default but the live runtime already passes it through unchanged.

    rag-hardening-1: defense-in-depth filter for ``allowed_in_live_context``.
    The retriever in ``rag/retriever.py:_filter_candidates`` already
    drops ``GOD_VIEW`` / ``MODERATOR_ONLY`` entries; this renderer
    re-checks the per-hit ``allowed_in_live_context`` flag and drops
    any hit that fails it. If the retriever ever regresses or is
    bypassed (e.g. an internal caller feeding hits directly), the
    slim renderer refuses to surface disallowed content to a live
    player. Drops are logged at WARNING so operators can spot the
    mismatch instead of silently leaking.

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
        contains only ``type``, ``title``, and prompt-safe V2 tactical
        frame fields.
    """
    # rag-hardening-1: defense-in-depth — drop hits that the
    # visibility layer already rejected, so a retriever regression
    # never reaches the live prompt.
    safe_hits: list[RAGHit] = []
    for hit in hits:
        if not getattr(hit, "allowed_in_live_context", True):
            import logging
            logging.getLogger(__name__).warning(
                "rag-hardening-1: dropping hit %s from live prompt — "
                "allowed_in_live_context=False (retriever may have regressed)",
                getattr(hit, "entry_id", "?"),
            )
            continue
        safe_hits.append(hit)
    deduped = dedup_hits_by_similarity(safe_hits, max_items=max_items)
    return [render_hit_for_prompt(h) for h in deduped]


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
