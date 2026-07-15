# -*- coding: utf-8 -*-
"""
为新游戏事件分配 V2 元数据，并提供唯一的存储边界序列化器。

作者: Project contributors
创建日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.core.models import GameEvent
    >>> stamp_new_events("g1", [], [GameEvent(type="enter_night")])[0].schema_version
    '2'
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("GameEvent.occurred_at must be timezone-aware")
    return value


def _is_complete_v2(event: GameEvent) -> bool:
    return (
        event.schema_version == "2"
        and event.event_id is not None
        and event.sequence_number is not None
        and event.occurred_at is not None
        and event.game_id is not None
        and event.visibility is not None
    )


def _stamp_event(
    game_id: str,
    event: GameEvent,
    sequence_number: int,
    occurred_at: datetime,
) -> GameEvent:
    payload = dict(event.payload)
    raw_visibility = payload.pop("visibility", "public")
    visibility = event.visibility or EventVisibility.from_legacy(raw_visibility)
    return replace(
        event,
        payload=payload,
        visibility=visibility,
        event_id=f"{game_id}:e{sequence_number:06d}",
        sequence_number=sequence_number,
        occurred_at=_require_aware(occurred_at),
        game_id=game_id,
        schema_version="2",
    )


def stamp_new_events(
    game_id: str,
    before: Sequence[GameEvent],
    after: Sequence[GameEvent],
    *,
    now: datetime | None = None,
) -> list[GameEvent]:
    """仅给 ``after`` 中本次新增且未盖章的事件分配 V2 元数据。"""
    occurred_at = _require_aware(now or datetime.now(timezone.utc))
    existing_sequences = [
        event.sequence_number
        for event in before
        if event.sequence_number is not None
    ]
    next_sequence = max([len(before), *(number + 1 for number in existing_sequences)])
    result = list(after[: len(before)])
    for event in after[len(before) :]:
        if _is_complete_v2(event):
            _require_aware(event.occurred_at)  # type: ignore[arg-type]
            result.append(event)
            next_sequence = max(next_sequence, event.sequence_number + 1)  # type: ignore[operator]
            continue
        result.append(_stamp_event(game_id, event, next_sequence, occurred_at))
        next_sequence += 1
    return result


def new_game_event(
    state: GameState,
    event_type: str,
    payload: Mapping[str, Any] | None = None,
    *,
    visibility: EventVisibility | None = None,
    trace_id: str | None = None,
    now: datetime | None = None,
) -> GameEvent:
    """立即创建带 ID 的 V2 事件，供同节点引用 source_event_id。"""
    event = GameEvent(
        type=event_type,
        payload=dict(payload or {}),
        visibility=visibility,
        trace_id=trace_id,
    )
    return stamp_new_events(state.game_id, state.events, [*state.events, event], now=now)[-1]


def serialize_game_event(event: GameEvent) -> dict[str, Any]:
    """把事件转换为 JSON/数据库可用字典。"""
    return {
        "type": event.type,
        "payload": dict(event.payload),
        "visibility": event.visibility.value if event.visibility is not None else None,
        "event_id": event.event_id,
        "sequence_number": event.sequence_number,
        "occurred_at": (
            _require_aware(event.occurred_at).isoformat()
            if event.occurred_at is not None
            else None
        ),
        "game_id": event.game_id,
        "trace_id": event.trace_id,
        "schema_version": event.schema_version,
    }


def deserialize_game_event(data: Mapping[str, Any]) -> GameEvent:
    """从完整 V2 JSON 或 V1 type/payload 字典读取事件。"""
    occurred_at_raw = data.get("occurred_at")
    occurred_at = (
        _require_aware(datetime.fromisoformat(str(occurred_at_raw)))
        if occurred_at_raw
        else None
    )
    visibility_raw = data.get("visibility")
    return GameEvent(
        type=str(data["type"]),
        payload=dict(data.get("payload") or {}),
        visibility=(
            EventVisibility.from_legacy(visibility_raw)
            if visibility_raw is not None
            else None
        ),
        event_id=data.get("event_id"),
        sequence_number=data.get("sequence_number"),
        occurred_at=occurred_at,
        game_id=data.get("game_id"),
        trace_id=data.get("trace_id"),
        schema_version=data.get("schema_version"),
    )


__all__ = [
    "deserialize_game_event",
    "new_game_event",
    "serialize_game_event",
    "stamp_new_events",
]
