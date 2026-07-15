# -*- coding: utf-8 -*-
"""
为新游戏事件分配 V2 元数据，并提供唯一的存储边界序列化器。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.core.models import GameEvent
    >>> stamp_new_events("g1", [], [GameEvent(type="enter_night")])[0].schema_version
    '2'
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.core.resolution_batches import (
    serialize_resolution_batches_in_value,
)


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


def _has_v2_identity_metadata(event: GameEvent) -> bool:
    return any((
        event.schema_version is not None,
        event.event_id is not None,
        event.sequence_number is not None,
        event.occurred_at is not None,
        event.game_id is not None,
    ))


def _validate_v2_event(game_id: str, event: GameEvent) -> bool:
    """验证已有 V2 身份字段，返回是否为完整 V2 事件。"""
    if not _has_v2_identity_metadata(event):
        return False
    if not _is_complete_v2(event):
        raise ValueError(f"partial V2 metadata on event: {event.type}")
    if event.game_id != game_id:
        raise ValueError(
            f"GameEvent.game_id mismatch: expected {game_id}, got {event.game_id}"
        )
    sequence_number = event.sequence_number
    if (
        not isinstance(sequence_number, int)
        or isinstance(sequence_number, bool)
        or sequence_number < 0
    ):
        raise ValueError(f"invalid GameEvent.sequence_number: {sequence_number}")
    expected_event_id = f"{game_id}:e{sequence_number:06d}"
    if event.event_id != expected_event_id:
        raise ValueError(
            f"GameEvent.event_id mismatch: expected {expected_event_id}, "
            f"got {event.event_id}"
        )
    _require_aware(event.occurred_at)  # type: ignore[arg-type]
    return True


def _collect_v2_identity(
    game_id: str,
    events: Sequence[GameEvent],
) -> tuple[set[str], set[int], int | None]:
    event_ids: set[str] = set()
    sequence_numbers: set[int] = set()
    last_sequence: int | None = None
    for event in events:
        if not _validate_v2_event(game_id, event):
            continue
        event_id = event.event_id
        sequence_number = event.sequence_number
        assert event_id is not None and sequence_number is not None
        if event_id in event_ids or sequence_number in sequence_numbers:
            raise ValueError(
                f"duplicate V2 event identity: {event_id}/{sequence_number}"
            )
        if last_sequence is not None and sequence_number <= last_sequence:
            raise ValueError(
                "V2 sequence_number must be strictly monotonic in event-log order"
            )
        event_ids.add(event_id)
        sequence_numbers.add(sequence_number)
        last_sequence = sequence_number
    return event_ids, sequence_numbers, last_sequence


def _matches_authoritative_prefix(
    authoritative: GameEvent,
    candidate: GameEvent,
) -> bool:
    """允许图状态携带未盖章逻辑副本，但拒绝任何已有 V2 身份冲突。"""
    if candidate == authoritative:
        return True
    if _has_v2_identity_metadata(candidate):
        return False
    authoritative_payload = deepcopy(authoritative.payload)
    candidate_payload = deepcopy(candidate.payload)
    authoritative_payload.pop("visibility", None)
    candidate_payload.pop("visibility", None)
    return (
        candidate.type == authoritative.type
        and candidate.trace_id == authoritative.trace_id
        and candidate_payload == authoritative_payload
        and event_visibility(candidate) is event_visibility(authoritative)
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
    if len(after) < len(before) or any(
        not _matches_authoritative_prefix(old_event, next_event)
        for old_event, next_event in zip(before, after)
    ):
        raise ValueError("after event prefix must preserve before events unchanged")
    event_ids, sequence_numbers, last_sequence = _collect_v2_identity(game_id, before)
    next_sequence = max(
        len(before),
        (max(sequence_numbers) + 1) if sequence_numbers else 0,
    )
    result = list(before)
    for event in after[len(before) :]:
        if _is_complete_v2(event):
            _validate_v2_event(game_id, event)
            assert event.event_id is not None and event.sequence_number is not None
            if (
                event.event_id in event_ids
                or event.sequence_number in sequence_numbers
            ):
                raise ValueError(
                    "duplicate V2 event identity: "
                    f"{event.event_id}/{event.sequence_number}"
                )
            if last_sequence is not None and event.sequence_number <= last_sequence:
                raise ValueError(
                    "V2 sequence_number must be strictly monotonic in event-log order"
                )
            result.append(event)
            event_ids.add(event.event_id)
            sequence_numbers.add(event.sequence_number)
            last_sequence = event.sequence_number
            next_sequence = max(next_sequence, event.sequence_number + 1)
            continue
        if _has_v2_identity_metadata(event):
            raise ValueError(f"partial V2 metadata on event: {event.type}")
        stamped = _stamp_event(game_id, event, next_sequence, occurred_at)
        result.append(stamped)
        assert stamped.event_id is not None and stamped.sequence_number is not None
        event_ids.add(stamped.event_id)
        sequence_numbers.add(stamped.sequence_number)
        last_sequence = stamped.sequence_number
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
    payload = serialize_resolution_batches_in_value(deepcopy(event.payload))
    if event.schema_version == "2" or event.visibility is not None:
        payload.pop("visibility", None)
    return {
        "type": event.type,
        "payload": payload,
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


def serialize_legacy_event_payload(event: GameEvent) -> dict[str, Any]:
    """为滚动升级期的旧 reader 生成带规范可见性的 payload。"""
    payload = serialize_resolution_batches_in_value(deepcopy(event.payload))
    payload["visibility"] = event_visibility(event).value
    return payload


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
        payload=deepcopy(data.get("payload") or {}),
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
    "serialize_legacy_event_payload",
    "stamp_new_events",
]
