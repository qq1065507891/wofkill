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
from typing import Any

from werewolf_agent.rag.schemas import CaseType, QualityGrade, RAGQuery, SourceType


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
