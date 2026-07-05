# -*- coding: utf-8 -*-
"""
功能描述：通过预定义黄金查询对 RAG 检索结果进行离线评估，计算 Recall/Precision 等指标。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from werewolf_agent.rag.schemas import RAGQuery


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    expected_entry_ids: list[str]
    role: str = ""
    phase: str = ""
    situation: str = ""
    ruleset_id: str = ""
    persona_style: str = ""
    forbidden_entry_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoldenQueryResult:
    query_id: str
    retrieved_entry_ids: list[str]
    expected_entry_ids: list[str]
    forbidden_entry_ids: list[str] = field(default_factory=list)
    first_relevant_rank: int | None = None
    forbidden_hit_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalEvalReport:
    total_queries: int
    recall_at: dict[int, float]
    mrr: float
    ndcg_at: dict[int, float]
    forbidden_hit_count: int
    results: list[GoldenQueryResult] = field(default_factory=list)


class RetrieverLike(Protocol):
    def retrieve(self, query: RAGQuery) -> list[Any]:
        ...


def load_golden_queries(path: str | Path) -> list[GoldenQuery]:
    """Load offline golden retrieval queries from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise TypeError("golden query file must contain a list")
    return [_golden_query_from_mapping(item) for item in raw]


def evaluate_golden_queries(
    retriever: RetrieverLike,
    golden_queries: list[GoldenQuery],
    *,
    k_values: tuple[int, ...] = (1, 3),
) -> RetrievalEvalReport:
    """Evaluate a retriever against golden query expectations."""
    normalized_k = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    ndcg_k_values = tuple(k for k in normalized_k if k == 3) or (3,)
    max_k = max((*normalized_k, *ndcg_k_values), default=3)

    results: list[GoldenQueryResult] = []
    recall_hits = {k: 0 for k in normalized_k}
    reciprocal_ranks: list[float] = []
    ndcg_values: dict[int, list[float]] = {k: [] for k in ndcg_k_values}
    forbidden_hit_count = 0

    for golden in golden_queries:
        query = RAGQuery(
            role=golden.role,
            phase=golden.phase,
            situation=golden.situation,
            ruleset_id=golden.ruleset_id,
            persona_style=golden.persona_style,
            max_results=max_k,
        )
        hits = retriever.retrieve(query)
        retrieved_ids = [_entry_id(hit) for hit in hits if _entry_id(hit)]
        expected = set(golden.expected_entry_ids)
        forbidden = set(golden.forbidden_entry_ids)

        first_rank = _first_relevant_rank(retrieved_ids, expected)
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)
        for k in normalized_k:
            if any(entry_id in expected for entry_id in retrieved_ids[:k]):
                recall_hits[k] += 1
        for k in ndcg_k_values:
            ndcg_values[k].append(_ndcg_at(retrieved_ids, expected, k))
        forbidden_hits = [
            entry_id for entry_id in retrieved_ids if entry_id in forbidden
        ]
        forbidden_hit_count += len(forbidden_hits)
        results.append(GoldenQueryResult(
            query_id=golden.query_id,
            retrieved_entry_ids=retrieved_ids,
            expected_entry_ids=list(golden.expected_entry_ids),
            forbidden_entry_ids=list(golden.forbidden_entry_ids),
            first_relevant_rank=first_rank,
            forbidden_hit_ids=forbidden_hits,
        ))

    total = len(golden_queries)
    recall_at = {
        k: (recall_hits[k] / total if total else 0.0)
        for k in normalized_k
    }
    ndcg_at = {
        k: (sum(values) / total if total else 0.0)
        for k, values in ndcg_values.items()
    }
    mrr = sum(reciprocal_ranks) / total if total else 0.0
    return RetrievalEvalReport(
        total_queries=total,
        recall_at=recall_at,
        mrr=mrr,
        ndcg_at=ndcg_at,
        forbidden_hit_count=forbidden_hit_count,
        results=results,
    )


def _golden_query_from_mapping(raw: Any) -> GoldenQuery:
    if not isinstance(raw, dict):
        raise TypeError("each golden query must be a mapping")
    return GoldenQuery(
        query_id=str(raw.get("query_id") or ""),
        role=str(raw.get("role") or ""),
        phase=str(raw.get("phase") or ""),
        situation=str(raw.get("situation") or ""),
        ruleset_id=str(raw.get("ruleset_id") or ""),
        persona_style=str(raw.get("persona_style") or ""),
        expected_entry_ids=[str(item) for item in raw.get("expected_entry_ids", []) or []],
        forbidden_entry_ids=[str(item) for item in raw.get("forbidden_entry_ids", []) or []],
        tags=[str(item) for item in raw.get("tags", []) or []],
    )


def _entry_id(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("entry_id") or "")
    return str(getattr(hit, "entry_id", "") or "")


def _first_relevant_rank(
    retrieved_entry_ids: list[str],
    expected_entry_ids: set[str],
) -> int | None:
    for index, entry_id in enumerate(retrieved_entry_ids, start=1):
        if entry_id in expected_entry_ids:
            return index
    return None


def _ndcg_at(
    retrieved_entry_ids: list[str],
    expected_entry_ids: set[str],
    k: int,
) -> float:
    if not expected_entry_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for index, entry_id in enumerate(retrieved_entry_ids[:k], start=1):
        if entry_id in expected_entry_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(expected_entry_ids), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
