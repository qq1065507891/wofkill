"""Strategy Retriever: search and rank RAG entries by query.

Priority order (design doc §9.2):
1. Same ruleset + phase + role, external high-end cases
2. External tactics (transferable patterns)
3. Project history + review (error correction)
4. Generic speech templates (avoid homogenization)

Results are ranked by relevance score and filtered by quality/visibility.
"""

from __future__ import annotations

import logging
import math
import re
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


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reranker score normalization
# ---------------------------------------------------------------------------
#
# R5: BGE-reranker-v2-m3 emits raw logits, which can be negative when the
# document is judged irrelevant. The merge formula
# ``(rerank + rule) / 2`` then pumped a negative number into
# ``RAGHit.relevance_score`` (pydantic ``Field(ge=0.0)``), crashing the
# live path. Sigmoid maps any real number into (0, 1) cleanly:
#   - very negative → near 0   (effectively ignored)
#   - 0             → 0.5      (neutral)
#   - very positive → near 1   (strong endorsement)
# After merging, we also clamp to [0,1] as a belt-and-suspenders guard
# in case some other component sneaks an out-of-range value through.


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid that survives both extreme tails."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


def _quality_priority(grade: QualityGrade, *, entry_id: str = "") -> int:
    """Return the priority for a QualityGrade, warning on missing.

    R20: previously ``_QUALITY_ORDER.get(missing_grade, 0)`` silently
    returned 0 when a new ``QualityGrade`` enum value was added
    without wiring its priority. That made every such entry get
    filtered out by ``quality_min`` with no log line — operators
    had no way to notice the gap.

    Now: if the grade is missing from the mapping, emit a WARNING
    identifying the entry and the grade, then fall through to the
    lowest priority (0). The behavior of treating the missing
    grade as 0 is preserved for backward compatibility; the
    warning is the only new operator-visible signal.
    """
    priority = _QUALITY_ORDER.get(grade)
    if priority is None:
        logger.warning(
            "RAG entry '%s' has quality_grade='%s' which is "
            "unregistered in _QUALITY_ORDER; treating as lowest "
            "priority (0). Add it to _QUALITY_ORDER in retriever.py.",
            entry_id, grade.value if hasattr(grade, "value") else grade,
        )
        return 0
    return priority

# Retrieval priority by case type (design doc §9.2)
_CASE_TYPE_PRIORITY: dict[CaseType, int] = {
    CaseType.EXTERNAL_HIGH_END_CASE: 4,
    CaseType.EXTERNAL_TACTICS: 3,
    CaseType.PROJECT_HISTORY: 2,
    CaseType.PROJECT_REVIEW: 2,
    CaseType.ROLE_STRATEGY: 1,
    CaseType.SPEECH_TEMPLATE: 0,
}


def _case_type_priority(case_type: CaseType, *, entry_id: str = "") -> int:
    """Return the priority for a CaseType, warning on missing.

    N7: ``_CASE_TYPE_PRIORITY.get(missing_case_type, 0)`` silently
    returned 0 when a new ``CaseType`` enum value was added without
    wiring its priority. That made the sort key and the rule-based
    score treat the new case_type as the lowest priority, and the
    drop was invisible to operators.

    Now: if the case_type is missing from the mapping, emit a
    WARNING identifying the entry and the case_type, then fall
    through to 0 (treat as lowest). The behavior is preserved for
    backward compatibility; the warning is the new operator signal.
    """
    priority = _CASE_TYPE_PRIORITY.get(case_type)
    if priority is None:
        logger.warning(
            "RAG entry '%s' has case_type='%s' which is "
            "unregistered in _CASE_TYPE_PRIORITY; treating as "
            "lowest priority (0). Add it to _CASE_TYPE_PRIORITY "
            "in retriever.py.",
            entry_id,
            case_type.value if hasattr(case_type, "value") else case_type,
        )
        return 0
    return priority


# P1-G8: human-readable display labels for the RAG hit's
# ``display_annotation`` field. The raw enum values stay on
# RAGHit.source_type / RAGHit.quality_grade for the audit log; the
# annotation is the moderator-facing one-liner and must read like
# a sentence, not a snake_case dump. Chinese-first per the project
# locale; English for the term that has a widely-recognized English
# rendering (e.g. 实战 / 公开赛 / 高段位赛).
_DISPLAY_SOURCE_LABELS: dict[SourceType, str] = {
    SourceType.PUBLIC_TOURNAMENT: "公开赛",
    SourceType.PUBLIC_REVIEW: "公开复盘",
    SourceType.EXPERT_COMMENTARY: "专家解说",
    SourceType.TRAINING_SESSION: "训练赛",
    SourceType.SELF_PLAY: "实战",
    SourceType.RULE_DERIVED: "规则推导",
    SourceType.MANUAL_ENTRY: "人工录入",
}

_DISPLAY_QUALITY_LABELS: dict[QualityGrade, str] = {
    QualityGrade.PRO_MATCH: "职业级",
    QualityGrade.EXPERT_REVIEW: "专家审核",
    QualityGrade.HIGH_RANK_GAME: "高段位赛",
    QualityGrade.RULE_DERIVED_SEED: "规则种子",
    QualityGrade.COMMUNITY_CASE: "社区案例",
    QualityGrade.SELF_PLAY_CANDIDATE: "实战候选",
    QualityGrade.UNREVIEWED: "未审核",
}

# R17: human-readable case_type label set for the hit annotation.
# ``source|quality|case_type`` lets a moderator see at a glance
# whether the hit is an external high-end case, a tactic, project
# history, a speech template, or a role strategy — not just the
# source + quality pair. Chinese-first per the project locale.
_DISPLAY_CASE_TYPE_LABELS: dict[CaseType, str] = {
    CaseType.EXTERNAL_HIGH_END_CASE: "高端案例",
    CaseType.EXTERNAL_TACTICS: "战术",
    CaseType.PROJECT_HISTORY: "历史",
    CaseType.PROJECT_REVIEW: "复盘",
    CaseType.ROLE_STRATEGY: "角色策略",
    CaseType.SPEECH_TEMPLATE: "模板",
}


def _tokenize_situation(situation: str) -> set[str]:
    """P1-G7: turn a key=value situation blob into a token set.

    The new format (``"role=seer phase=day task=speech actions=['vote']"``)
    is noisy: the raw whitespace split would emit tokens like
    ``"actions=['vote']"`` that never match any tag. Splitting on
    ``=`` first, then stripping list/quote noise from the value side,
    yields a clean token set that the tag-overlap scorer can use.

    Backward compatible: legacy space-joined situations (e.g. the
    string ``"抗推预言家"``) still tokenize correctly because there
    is no ``=`` in them and the value side just becomes the whole
    word.

    R7: the no-``=`` branch (chunks like ``'speech']`` produced when
    the situation is ``actions=['vote', 'speech']``) used to add the
    chunk verbatim, leaving the trailing ``']`` glued to ``speech``.
    We now run the same strip-list-syntax step on every chunk so
    both ``vote`` and ``speech`` are recovered cleanly.
    """
    if not situation:
        return set()
    tokens: set[str] = set()
    for chunk in situation.split():
        if "=" not in chunk:
            # R7: even when there is no ``=`` the chunk may still
            # carry list/quote noise (e.g. ``'speech']`` from a
            # Python-style actions list). Run it through the same
            # strip pass so ``speech`` is recoverable.
            value = chunk
        else:
            key, _, value = chunk.partition("=")
            # Drop the key (e.g. "role", "actions") — only the value
            # tokens are useful for tag overlap. We still keep the key
            # when the value is empty (e.g. "actions=") so the
            # token set is non-empty.
            if not value.strip():
                tokens.add(key.lower())
                continue
        # Strip list / set / dict syntax around the value. We split
        # on common delimiters and quote chars; the leftover pieces
        # are individual tokens (e.g. "vote", "speech", "12").
        for piece in re.split(r"[\[\]\(\)\{\},'\"\s]+", value):
            piece = piece.strip().lower()
            if piece:
                tokens.add(piece)
    return tokens


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
        candidates = self._filter_candidates(query)
        scored = [
            (self._merged_score(entry, query), entry)
            for entry in candidates
        ]
        # R12: case_type is a first-class sort key — strictly above
        # quality — so an EXTERNAL_HIGH_END_CASE outranks a
        # SPEECH_TEMPLATE regardless of the quality gap. The previous
        # additive scoring (case_type * 0.075 + quality / 20) let
        # a high-quality template tie or beat a low-quality external
        # case, and Python's stable sort then preserved insertion
        # order — meaning the case_type priority effectively didn't
        # dominate the final ranking. The new sort is
        # (case_type_priority desc, quality desc, rule_score desc),
        # so case_type wins first, then quality, then the rest of
        # the rule-based signal.
        scored.sort(
            key=lambda x: (
                _case_type_priority(
                    x[1].metadata.case_type, entry_id=x[1].entry_id,
                ),
                # N3: route the quality sort key through
                # ``_quality_priority`` so a missing grade emits a
                # WARNING (matching the asymmetry fix in N2 for the
                # query filter and in R20 for the entry score). The
                # old ``_QUALITY_ORDER.get(grade, 0)`` was a silent
                # no-op; behavior was identical (both default to 0)
                # but operators had no signal of the gap.
                _quality_priority(
                    x[1].metadata.quality_grade, entry_id=x[1].entry_id,
                ),
                x[0],
            ),
            reverse=True,
        )

        if self._reranker and scored:
            # Take a wider pool for reranking
            rerank_pool_size = min(len(scored), query.max_results * 3)
            rerank_pool = scored[:rerank_pool_size]
            query_text = self._build_rerank_query(query)

            # Rerank by semantic relevance
            # N5: truncate the summary to the same 800-char cap
            # ``_entry_to_hit`` enforces on the audit side, BEFORE
            # building the reranker input dict. The old code passed
            # the full ``e.summary`` to the reranker and only
            # truncated later when building the hit — so the model
            # scored on text the operator never saw in the audit
            # JSON. Truncating up front means the two paths agree on
            # what the model sees.
            #
            # G-R4-04: include the title and key_decisions in the
            # text the reranker scores on. The reranker's input
            # contract is ``text_key`` — it scores on whatever
            # field is named by that key. Pre-fix we built a
            # ``"summary"``-only field and dropped the title and
            # key_decisions (often the most informative parts of
            # a RAG case) on the floor. The new field is
            # ``"text"`` and is built as
            # ``f"{title}\n{summary}\n{key_decisions}"[:1500]``:
            # the 1500-char cap keeps the reranker's input budget
            # bounded and the union is exposed under a single
            # key so the reranker doesn't need to know about the
            # three sub-fields.
            reranked = self._reranker.rerank_hits(
                query=query_text,
                documents=[
                    {
                        "score": s,
                        "entry": e,
                        "text": (
                            f"{e.title}\n{e.summary[:800]}\n"
                            f"{' '.join(e.key_decisions)}"
                        )[:1500],
                    }
                    for s, e in rerank_pool
                ],
                text_key="text",
                top_n=query.max_results,
            )
            results: list[RAGHit] = []
            for doc in reranked:
                entry = doc["entry"]
                # R5: sigmoid-normalize the raw reranker logit before
                # merging so a negative score (which BGE emits for
                # judged-irrelevant docs) doesn't push the combined
                # score below 0 and crash RAGHit pydantic validation.
                raw_rerank = float(doc.get("rerank_score", 0.0))
                normalized_rerank = _sigmoid(raw_rerank)
                rule_score = float(doc.get("score", 0.0))
                combined_score = (normalized_rerank + rule_score) / 2.0
                # Belt and suspenders: clamp to [0,1] even though
                # sigmoid already lives in (0,1) and rule_score lives
                # in [0,1] — a future tweak to either side could
                # break the invariant without this guard.
                combined_score = max(0.0, min(1.0, combined_score))
                hit = self._entry_to_hit(entry, round(combined_score, 3), query)
                results.append(hit)
            return results

        results = []
        for score, entry in scored[:query.max_results]:
            hit = self._entry_to_hit(entry, score, query)
            results.append(hit)
        return results

    def _build_rerank_query(self, query: RAGQuery) -> str:
        """Build a semantic query string for the reranker."""
        parts = []
        if query.role:
            parts.append(f"角色:{query.role}")
        if query.phase:
            parts.append(f"阶段:{query.phase}")
        if query.situation:
            parts.append(query.situation)
        if query.persona_style:
            parts.append(f"风格:{query.persona_style}")
        return " ".join(parts) if parts else "通用策略检索"

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
                # R20: route the ENTRY's grade through _quality_priority
                # so a missing entry's grade emits a WARNING instead of
                # silently falling through to 0.
                entry_priority = _quality_priority(
                    meta.quality_grade, entry_id=entry.entry_id,
                )
                # N2: route the QUERY's quality_min through the same
                # helper. The old ``_QUALITY_ORDER.get(query.quality_min,
                # 0)`` silently returned 0 for a missing grade, which
                # made the entire filter a no-op and dropped every
                # entry (or admitted every entry, depending on the
                # comparison direction) with no operator-visible
                # signal. The treat-as-lowest behavior is preserved;
                # the warning is the new operator signal.
                min_priority = _quality_priority(
                    query.quality_min, entry_id=f"query:{query.quality_min.value}",
                )
                if entry_priority < min_priority:
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
        # N7: route through ``_case_type_priority`` so a missing
        # case_type logs a WARNING rather than silently defaulting
        # to 0.
        score += _case_type_priority(
            meta.case_type, entry_id=entry.entry_id,
        ) * 0.075

        # Quality grade bonus (0.0–0.3)
        # R20: route through _quality_priority so a missing grade
        # logs a WARNING rather than silently defaulting to 0.
        score += _quality_priority(
            meta.quality_grade, entry_id=entry.entry_id,
        ) / 20.0

        # Role match (0.15)
        # G-R4-02: ``role_perspective='any'`` is the universal-
        # perspective marker used by the ``基础常识`` seed family
        # (金水 / 银水 / 对跳判断 / 警徽票权重, etc.). It must
        # receive the same wildcard bonus as ``'general'`` so
        # universal-knowledge seeds rank at parity with the
        # existing universal seeds rather than being demoted
        # below role-specific entries regardless of relevance.
        if query.role and meta.role_perspective:
            if query.role == meta.role_perspective:
                score += 0.15
            elif meta.role_perspective in ("general", "any"):
                score += 0.05

        # Phase match (0.1)
        if query.phase and meta.phase:
            if query.phase == meta.phase:
                score += 0.1
            elif meta.phase == "general":
                score += 0.03

        # Tag overlap (0.1)
        if query.situation:
            # P1-G7: situation is a key=value blob (e.g.
            # "role=seer phase=day task=speech actions=['vote']").
            # Tokenize on '=' to recover the value tokens that the
            # rule-based scorer can match against the entry's tag
            # list. We keep both keys and values, but strip the
            # list-bracket/list-comma noise around action values.
            situation_words = _tokenize_situation(query.situation)
            tag_words = set(" ".join(meta.tags).lower().split())
            overlap = len(situation_words & tag_words)
            if overlap > 0:
                score += min(0.1, overlap * 0.03)

        return min(score, 1.0)

    def _merged_score(self, entry: RAGEntry, query: RAGQuery) -> float:
        """Combine the rule-based score with the optional vector score.

        R2: when no vector score is registered for the entry, returns
        the pure rule-based score (legacy behavior). When a vector
        score is registered, returns a convex combination weighted by
        ``merge_vector_score``. The merge happens on already-clamped
        [0,1] values; result is clamped to [0,1] for the RAGHit
        relevance_score field.
        """
        rule = self._score(entry, query)
        if not self._vector_scores or entry.entry_id not in self._vector_scores:
            return rule
        w = self._merge_vector_score
        vec = max(0.0, min(1.0, float(self._vector_scores[entry.entry_id])))
        merged = (1.0 - w) * rule + w * vec
        return max(0.0, min(1.0, merged))

    def _entry_to_hit(self, entry: RAGEntry, score: float, query: RAGQuery) -> RAGHit:
        """Convert an entry to a retrieval hit with display annotation."""
        meta = entry.metadata
        allowed_in_live = meta.visibility_boundary in (
            VisibilityBoundary.PUBLIC_ONLY,
            VisibilityBoundary.PLAYER_PERSPECTIVE,
        )

        # Build display annotation
        # P1-G8: human-readable labels instead of raw enum values
        # like "[public_tournament|self_play_candidate]". The raw
        # values are still on RAGHit.source_type / RAGHit.quality_grade
        # for the audit log; this annotation is the moderator-facing
        # one-liner and must read like a phrase.
        # R17: case_type is appended as a third pipe-delimited slot
        # so the moderator can tell external-high-end cases apart
        # from tactics / history / templates at a glance.
        source_label = _DISPLAY_SOURCE_LABELS.get(
            meta.source.source_type, meta.source.source_type.value,
        )
        quality_label = _DISPLAY_QUALITY_LABELS.get(
            meta.quality_grade, meta.quality_grade.value,
        )
        case_type_label = _DISPLAY_CASE_TYPE_LABELS.get(
            meta.case_type, meta.case_type.value,
        )
        annotation = f"[{source_label}|{quality_label}|{case_type_label}]"

        return RAGHit(
            entry_id=entry.entry_id,
            title=entry.title,
            summary=entry.summary[:800],
            relevance_score=round(score, 3),
            quality_grade=meta.quality_grade,
            source_type=meta.source.source_type,
            visibility_boundary=meta.visibility_boundary,
            case_type=meta.case_type,
            role_perspective=meta.role_perspective,
            phase=meta.phase,
            key_decisions=entry.key_decisions[:5],  # R4: audit 5 / prompt 3 is intentional — prompt_renderer caps at 3 for the live LLM, but the audit JSON keeps the full 5 for review.
            short_quotes=entry.short_quotes,
            tags=meta.tags,
            allowed_in_live_context=allowed_in_live,
            display_annotation=annotation,
        )


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
