# -*- coding: utf-8 -*-
"""Runtime audit events for prompt-visible module exposure.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from werewolf_agent.core.models import GameEvent
from werewolf_agent.evaluation.trace_identity import DecisionIdentity

_RAG_KEYS = frozenset({
    "entry_id",
    "rank",
    "relevance_score",
    "prompt_visible",
    "title",
    "situation_signature",
    "retrieval_reason",
})
_REFLECTION_KEYS = frozenset({
    "entry_id",
    "rank",
    "quality_score",
    "prompt_visible",
    "lesson_key",
    "quality_status",
})
_PERSONA_KEYS = frozenset({
    "profile_id",
    "prompt_visible",
    "policy_keys",
    "sanitized",
})


def _identity_payload(identity: DecisionIdentity) -> dict[str, Any]:
    return {
        "trace_id": identity.trace_id(),
        "game_id": identity.game_id,
        "player_id": identity.player_id,
        "phase": identity.phase,
        "day_number": identity.day_number,
        "night_number": identity.night_number,
        "task_type": identity.task_type,
        "action_index": identity.action_index,
        "visibility": "moderator_only",
    }


def _sanitize_allowed(value: Any, allowed_keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_allowed(item, allowed_keys)
            for key, item in value.items()
            if str(key) in allowed_keys
        }
    if isinstance(value, list):
        return [_sanitize_allowed(item, allowed_keys) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_allowed(item, allowed_keys) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summary_hash(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _skill_rows(analyses: Any) -> list[dict[str, Any]]:
    if isinstance(analyses, Mapping):
        items = [
            {
                "skill_name": str(name),
                "rank": index + 1,
                "prompt_visible": True,
                "summary_hash": _summary_hash(summary),
                "advice_type": "tactical",
            }
            for index, (name, summary) in enumerate(analyses.items())
            if summary
        ]
        return items

    if not isinstance(analyses, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(analyses):
        if not isinstance(item, Mapping):
            continue
        summary = (
            item.get("summary")
            or item.get("analysis")
            or item.get("advice")
            or item.get("prompt_visible")
            or ""
        )
        row: dict[str, Any] = {
            "skill_name": str(item.get("skill_name") or item.get("name") or item.get("skill") or ""),
            "rank": item.get("rank", index + 1),
            "prompt_visible": item.get("prompt_visible", True),
            "summary_hash": _summary_hash(summary),
        }
        advice_type = item.get("advice_type")
        if advice_type:
            row["advice_type"] = str(advice_type)
        rows.append(row)
    return rows


class ModuleExposureAuditCollector:
    """Collect sanitized module exposure audit events for one agent action."""

    def __init__(self) -> None:
        self._events: list[GameEvent] = []

    def record_rag(self, identity: DecisionIdentity, hits: list[dict[str, Any]] | None) -> None:
        if not hits:
            return
        sanitized = _sanitize_allowed(hits, _RAG_KEYS)
        if not any(row for row in sanitized if row):
            return
        self._append(
            "rag_exposure_audit",
            identity,
            {"hits": sanitized},
        )

    def record_reflection(
        self,
        identity: DecisionIdentity,
        cards: list[dict[str, Any]] | None,
    ) -> None:
        if not cards:
            return
        sanitized = _sanitize_allowed(cards, _REFLECTION_KEYS)
        if not any(row for row in sanitized if row):
            return
        self._append(
            "reflection_exposure_audit",
            identity,
            {"cards": sanitized},
        )

    def record_skill(self, identity: DecisionIdentity, analyses: Any) -> None:
        rows = _skill_rows(analyses)
        if not rows:
            return
        self._append("skill_exposure_audit", identity, {"analyses": rows})

    def record_persona(self, identity: DecisionIdentity, snapshot: Mapping[str, Any] | None) -> None:
        if not snapshot:
            return
        sanitized = _sanitize_allowed(snapshot, _PERSONA_KEYS)
        if not sanitized:
            return
        sanitized.setdefault("sanitized", True)
        self._append("persona_exposure_audit", identity, {"snapshot": sanitized})

    def flush_events(self) -> list[GameEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def _append(
        self,
        event_type: str,
        identity: DecisionIdentity,
        module_payload: dict[str, Any],
    ) -> None:
        payload = _identity_payload(identity)
        payload.update(module_payload)
        self._events.append(GameEvent(type=event_type, payload=payload))
