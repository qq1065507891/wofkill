# -*- coding: utf-8 -*-
"""
统一构造和读取安全的赛后反思事件摘要。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-17

使用示例:
    >>> canonical_verified_reflections([])
    {}
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from werewolf_agent.memory.reflection_sanitization import (
    anonymize_player_ids_recursive,
)

_STATUSES = frozenset({
    "not_generated", "invalid_structured_draft", "verified", "agent_error",
})
_SAFE_FAILURE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


def safe_reflection_verification(
    candidate: Any,
    *,
    decision_id: str,
) -> dict[str, Any]:
    """从不可信 adapter 返回值按严格 allowlist 重建可持久化摘要。"""
    source = candidate if isinstance(candidate, dict) else {}
    status = source.get("status")
    safe_status = status if isinstance(status, str) and status in _STATUSES else "agent_error"
    candidate_decision_id = source.get("decision_id")
    safe_decision_id = (
        candidate_decision_id
        if isinstance(candidate_decision_id, str) and candidate_decision_id
        else decision_id
    )

    def count(name: str) -> int:
        value = source.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def identifiers(name: str) -> list[str]:
        value = source.get(name)
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(
            item for item in value if isinstance(item, str) and item
        ))

    lessons: list[dict[str, str]] = []
    labels: dict[str, str] = {}
    raw_lessons = source.get("verified_lessons")
    if isinstance(raw_lessons, list):
        for lesson in raw_lessons:
            if not isinstance(lesson, dict):
                continue
            lesson_id = lesson.get("lesson_id")
            abstraction = lesson.get("abstraction")
            if not isinstance(lesson_id, str) or not isinstance(abstraction, str):
                continue
            lessons.append({
                "lesson_id": lesson_id,
                "abstraction": anonymize_player_ids_recursive(abstraction, labels),
            })
    result = {
        "status": safe_status,
        "decision_id": safe_decision_id,
        "verified_fact_count": count("verified_fact_count"),
        "verified_claim_ids": identifiers("verified_claim_ids"),
        "rejected_claim_ids": identifiers("rejected_claim_ids"),
        "verified_lessons": lessons,
        "rejected_fact_count": count("rejected_fact_count"),
        "rejected_lesson_count": count("rejected_lesson_count"),
    }
    safe_failures: dict[str, str] = {}
    for field in ("failure_stage", "failure_code"):
        value = source.get(field)
        if isinstance(value, str) and _SAFE_FAILURE_IDENTIFIER.fullmatch(value):
            safe_failures[field] = value
    if set(safe_failures) == {"failure_stage", "failure_code"}:
        result.update(safe_failures)
    return result


def canonical_verified_reflections(events: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """每名玩家只保留最新 canonical decision，重复事件不重复累计。"""
    decisions: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    sequence = 0
    for event in events:
        if getattr(event, "type", "") != "reflection_complete":
            continue
        payload = getattr(event, "payload", {}) or {}
        for entry in payload.get("entries", []):
            if not isinstance(entry, dict):
                continue
            player_id = entry.get("player_id")
            verification = entry.get("verification")
            if not isinstance(player_id, str) or not player_id or not isinstance(verification, dict):
                continue
            decision_id = entry.get("decision_id") or verification.get("decision_id")
            canonical_id = str(decision_id) if decision_id else f"legacy:{sequence}"
            decisions[(player_id, canonical_id)] = (sequence, verification)
            sequence += 1

    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    for (player_id, _decision_id), item in decisions.items():
        if player_id not in latest or item[0] > latest[player_id][0]:
            latest[player_id] = item
    return {player_id: item[1] for player_id, item in latest.items()}


__all__ = [
    "canonical_verified_reflections",
    "safe_reflection_verification",
]
