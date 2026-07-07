# -*- coding: utf-8 -*-
"""
验证 RAG 查询处理拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> python -m pytest tests/rag/test_query_processing.py
"""

from __future__ import annotations

from werewolf_agent.rag.schemas import (
    CaseMetadata,
    CaseType,
    QualityGrade,
    RAGQuery,
    ReviewStatus,
    SourceMetadata,
    SourceType,
    VisibilityBoundary,
)


def _make_meta(*, role: str = "seer", phase: str = "speech") -> CaseMetadata:
    return CaseMetadata(
        case_type=CaseType.ROLE_STRATEGY,
        quality_grade=QualityGrade.COMMUNITY_CASE,
        review_status=ReviewStatus.APPROVED,
        phase=phase,
        role_perspective=role,
        visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
        source=SourceMetadata(source_type=SourceType.MANUAL_ENTRY),
    )


def test_tokenize_new_module_matches_legacy_vector_store_export() -> None:
    from werewolf_agent.rag.query_processing import _tokenize as new_tokenize
    from werewolf_agent.rag.vector_store import _tokenize as legacy_tokenize

    text = "Seer 查验 p01"

    assert new_tokenize(text) == ["seer", "查", "验", "p01"]
    assert legacy_tokenize(text) == new_tokenize(text)


def test_role_phase_matches_new_module_matches_legacy_retriever_export() -> None:
    from werewolf_agent.rag.retrieval_ranking import role_phase_matches as new_match
    from werewolf_agent.rag.retriever import role_phase_matches as legacy_match

    query = RAGQuery(role="seer", phase="speech")
    matching = _make_meta(role="general", phase="speech")
    mismatching = _make_meta(role="werewolf", phase="speech")

    assert new_match(query, matching) is True
    assert legacy_match(query, matching) == new_match(query, matching)
    assert new_match(query, mismatching) is False
    assert legacy_match(query, mismatching) == new_match(query, mismatching)


def test_legacy_retriever_quality_order_rebind_affects_priority() -> None:
    import werewolf_agent.rag.retriever as retriever

    original = retriever._QUALITY_ORDER
    try:
        retriever._QUALITY_ORDER = {}

        assert retriever._quality_priority(QualityGrade.PRO_MATCH, entry_id="q") == 0
    finally:
        retriever._QUALITY_ORDER = original


def test_legacy_retriever_case_type_priority_rebind_affects_priority() -> None:
    import werewolf_agent.rag.retriever as retriever

    original = retriever._CASE_TYPE_PRIORITY
    try:
        retriever._CASE_TYPE_PRIORITY = {}

        assert retriever._case_type_priority(CaseType.ROLE_STRATEGY, entry_id="c") == 0
    finally:
        retriever._CASE_TYPE_PRIORITY = original
