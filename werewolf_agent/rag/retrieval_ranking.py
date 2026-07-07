# -*- coding: utf-8 -*-
"""
RAG 检索排序与过滤辅助函数。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.rag.retrieval_ranking import role_phase_matches
    >>> role_phase_matches(query, metadata)
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
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
from werewolf_agent.rag.tactical_text import build_rag_retrieval_text


# 保持历史 logger 名称，避免拆分后运维日志筛选和旧测试漂移。
logger = logging.getLogger("werewolf_agent.rag.retriever")


def role_phase_matches(query: RAGQuery, meta: Any) -> bool:
    """Hard role/phase gate shared by all RAG retrieval paths.

    Single source of truth for the wildcard convention:
    - role matches when ``meta.role_perspective`` equals ``query.role``,
      or is a universal marker (``"general"`` / ``"any"`` / empty), or
      when the query carries no role.
    - phase matches when ``meta.phase`` equals ``query.phase``, or is
      ``"general"`` / empty, or when the query carries no phase.

    Both must hold (AND semantics) so a cross-role case cannot leak in
    just because the phase happens to match. ``meta`` is a RAGMetadata
    (duck-typed: needs ``role_perspective`` and ``phase`` attrs).
    """
    role_ok = (
        not query.role
        or meta.role_perspective in (query.role, "general", "any", "")
    )
    phase_ok = (
        not query.phase
        or meta.phase in (query.phase, "general", "")
    )
    return role_ok and phase_ok


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid that survives both extreme tails."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
    """Return the priority for a QualityGrade, warning on missing."""
    return quality_priority_from_order(grade, _QUALITY_ORDER, entry_id=entry_id)


def quality_priority_from_order(
    grade: QualityGrade,
    order: dict[QualityGrade, int],
    *,
    entry_id: str = "",
) -> int:
    """按给定映射计算 QualityGrade 优先级，供旧模块兼容入口复用。"""
    priority = order.get(grade)
    if priority is None:
        logger.warning(
            "RAG entry '%s' has quality_grade='%s' which is "
            "unregistered in _QUALITY_ORDER; treating as lowest "
            "priority (0). Add it to _QUALITY_ORDER in retriever.py.",
            entry_id, grade.value if hasattr(grade, "value") else grade,
        )
        return 0
    return priority


_CASE_TYPE_PRIORITY: dict[CaseType, int] = {
    CaseType.EXTERNAL_HIGH_END_CASE: 4,
    CaseType.EXTERNAL_TACTICS: 3,
    CaseType.PROJECT_HISTORY: 2,
    CaseType.PROJECT_REVIEW: 2,
    CaseType.ROLE_STRATEGY: 1,
    CaseType.SPEECH_TEMPLATE: 0,
}


def _case_type_priority(case_type: CaseType, *, entry_id: str = "") -> int:
    """Return the priority for a CaseType, warning on missing."""
    return case_type_priority_from_order(
        case_type,
        _CASE_TYPE_PRIORITY,
        entry_id=entry_id,
    )


def case_type_priority_from_order(
    case_type: CaseType,
    order: dict[CaseType, int],
    *,
    entry_id: str = "",
) -> int:
    """按给定映射计算 CaseType 优先级，供旧模块兼容入口复用。"""
    priority = order.get(case_type)
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


def _tokenize_situation(situation: str) -> set[str]:
    """将 key=value 风格的 situation 文本拆成可用于 tag overlap 的 token。"""
    if not situation:
        return set()
    tokens: set[str] = set()
    for chunk in situation.split():
        if "=" not in chunk:
            value = chunk
        else:
            key, _, value = chunk.partition("=")
            if not value.strip():
                tokens.add(key.lower())
                continue
        for piece in re.split(r"[\[\]\(\)\{\},'\"\s]+", value):
            piece = piece.strip().lower()
            if piece:
                tokens.add(piece)
    return tokens


def build_rerank_query(query: RAGQuery) -> str:
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


def filter_candidates(
    entries: list[RAGEntry],
    query: RAGQuery,
    *,
    quality_priority_fn: Callable[[QualityGrade], int],
    role_phase_matches_fn: Callable[[RAGQuery, Any], bool] = role_phase_matches,
) -> list[RAGEntry]:
    """Filter entries by hard criteria."""
    results: list[RAGEntry] = []
    for entry in entries:
        meta = entry.metadata

        if meta.visibility_boundary == VisibilityBoundary.GOD_VIEW:
            if not query.include_god_view:
                continue

        if query.ruleset_id and meta.ruleset_id:
            if meta.ruleset_id != query.ruleset_id:
                continue

        if query.quality_min:
            entry_priority = quality_priority_fn(
                meta.quality_grade,
                entry_id=entry.entry_id,
            )
            min_priority = quality_priority_fn(
                query.quality_min,
                entry_id=f"query:{query.quality_min.value}",
            )
            if entry_priority < min_priority:
                continue

        if query.source_types:
            if meta.source.source_type not in query.source_types:
                continue

        if query.case_types:
            if meta.case_type not in query.case_types:
                continue

        if not role_phase_matches_fn(query, meta):
            continue

        results.append(entry)
    return results


def score_entry(
    entry: RAGEntry,
    query: RAGQuery,
    *,
    case_type_priority_fn: Callable[[CaseType], int],
    quality_priority_fn: Callable[[QualityGrade], int],
    tokenize_situation_fn: Callable[[str], set[str]] = _tokenize_situation,
) -> float:
    """Compute relevance score [0..1] for an entry."""
    score = 0.0
    meta = entry.metadata

    score += case_type_priority_fn(
        meta.case_type,
        entry_id=entry.entry_id,
    ) * 0.075

    score += quality_priority_fn(
        meta.quality_grade,
        entry_id=entry.entry_id,
    ) / 20.0

    if query.role and meta.role_perspective:
        if query.role == meta.role_perspective:
            score += 0.15
        elif meta.role_perspective in ("general", "any"):
            score += 0.05

    if query.phase and meta.phase:
        if query.phase == meta.phase:
            score += 0.1
        elif meta.phase == "general":
            score += 0.03

    if query.situation:
        situation_words = tokenize_situation_fn(query.situation)
        tag_words = set(" ".join(meta.tags).lower().split())
        overlap = len(situation_words & tag_words)
        if overlap > 0:
            score += min(0.1, overlap * 0.03)

    return min(score, 1.0)


def merged_score(
    entry: RAGEntry,
    query: RAGQuery,
    *,
    vector_scores: dict[str, float],
    score_fn: Callable[[RAGEntry, RAGQuery], float],
) -> float:
    """Combine the rule-based score with the optional vector score."""
    rule = score_fn(entry, query)
    if not vector_scores or entry.entry_id not in vector_scores:
        return rule
    vec = max(0.0, min(1.0, float(vector_scores[entry.entry_id])))
    return max(0.0, min(1.0, max(rule, vec)))


def entry_to_hit(entry: RAGEntry, score: float, query: RAGQuery) -> RAGHit:
    """Convert an entry to a retrieval hit with display annotation."""
    meta = entry.metadata
    allowed_in_live = meta.visibility_boundary in (
        VisibilityBoundary.PUBLIC_ONLY,
        VisibilityBoundary.PLAYER_PERSPECTIVE,
    )

    source_label = _DISPLAY_SOURCE_LABELS.get(
        meta.source.source_type,
        meta.source.source_type.value,
    )
    quality_label = _DISPLAY_QUALITY_LABELS.get(
        meta.quality_grade,
        meta.quality_grade.value,
    )
    case_type_label = _DISPLAY_CASE_TYPE_LABELS.get(
        meta.case_type,
        meta.case_type.value,
    )
    annotation = f"[{source_label}|{quality_label}|{case_type_label}]"

    return RAGHit(
        entry_id=entry.entry_id,
        title=entry.title,
        summary=entry.summary[:800],
        tactical_frame=entry.tactical_frame,
        relevance_score=round(score, 3),
        quality_grade=meta.quality_grade,
        source_type=meta.source.source_type,
        visibility_boundary=meta.visibility_boundary,
        case_type=meta.case_type,
        role_perspective=meta.role_perspective,
        phase=meta.phase,
        key_decisions=entry.key_decisions[:5],
        short_quotes=entry.short_quotes,
        tags=meta.tags,
        allowed_in_live_context=allowed_in_live,
        display_annotation=annotation,
    )


def retrieve_ranked_hits(
    *,
    entries: list[RAGEntry],
    query: RAGQuery,
    reranker: Any,
    filter_candidates_fn: Callable[[RAGQuery], list[RAGEntry]],
    merged_score_fn: Callable[[RAGEntry, RAGQuery], float],
    entry_to_hit_fn: Callable[[RAGEntry, float, RAGQuery], RAGHit],
    case_type_priority_fn: Callable[[CaseType], int],
    quality_priority_fn: Callable[[QualityGrade], int],
) -> list[RAGHit]:
    """Retrieve, sort, optionally rerank, and shape final RAG hits."""
    candidates = filter_candidates_fn(query)
    scored = [
        (merged_score_fn(entry, query), entry)
        for entry in candidates
    ]
    scored.sort(
        key=lambda x: (
            case_type_priority_fn(
                x[1].metadata.case_type,
                entry_id=x[1].entry_id,
            ),
            quality_priority_fn(
                x[1].metadata.quality_grade,
                entry_id=x[1].entry_id,
            ),
            x[0],
        ),
        reverse=True,
    )

    if reranker and scored:
        rerank_pool_size = min(len(scored), query.max_results * 3)
        rerank_pool = scored[:rerank_pool_size]
        query_text = build_rerank_query(query)
        reranked = reranker.rerank_hits(
            query=query_text,
            documents=[
                {
                    "score": score,
                    "entry": entry,
                    "text": build_rag_retrieval_text(entry, max_chars=1500),
                }
                for score, entry in rerank_pool
            ],
            text_key="text",
            top_n=query.max_results,
        )
        results: list[RAGHit] = []
        for doc in reranked:
            entry = doc["entry"]
            raw_rerank = float(doc.get("rerank_score", 0.0))
            normalized_rerank = _sigmoid(raw_rerank)
            rule_score = float(doc.get("score", 0.0))
            combined_score = (normalized_rerank + rule_score) / 2.0
            combined_score = max(0.0, min(1.0, combined_score))
            results.append(entry_to_hit_fn(entry, round(combined_score, 3), query))
        return results

    return [
        entry_to_hit_fn(entry, score, query)
        for score, entry in scored[:query.max_results]
    ]


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

_DISPLAY_CASE_TYPE_LABELS: dict[CaseType, str] = {
    CaseType.EXTERNAL_HIGH_END_CASE: "高端案例",
    CaseType.EXTERNAL_TACTICS: "战术",
    CaseType.PROJECT_HISTORY: "历史",
    CaseType.PROJECT_REVIEW: "复盘",
    CaseType.ROLE_STRATEGY: "角色策略",
    CaseType.SPEECH_TEMPLATE: "模板",
}
