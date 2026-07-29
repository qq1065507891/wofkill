# -*- coding: utf-8 -*-
"""
定义自主玩家 model/tool durable dispatch 的严格请求、状态与结果契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_serializer, field_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    _freeze_json_object,
    _thaw_json_value,
)


class DispatchOperationKind(StrEnum):
    """可持久化的外部请求类型。"""

    MODEL = "model"
    TOOL = "tool"


class DispatchRecoveryPolicy(StrEnum):
    """进程重启后用于判定是否可以安全恢复的策略。"""

    IDEMPOTENT_LOOKUP_OR_REISSUE = "idempotent_lookup_or_reissue"
    AT_MOST_ONCE_UNKNOWN = "at_most_once_unknown"


class DispatchStatus(StrEnum):
    """dispatch attempt 的持久化状态机。"""

    PENDING = "pending"
    DISPATCHING = "dispatching"
    DISPATCHED = "dispatched"
    RESULT_RECORDED = "result_recorded"
    CANCELLED = "cancelled"
    UNKNOWN_OUTCOME = "unknown_outcome"


class DispatchResultOutcome(StrEnum):
    """已脱敏结果的成功或失败结局。"""

    SUCCESS = "success"
    FAILURE = "failure"


class DispatchResultDisposition(StrEnum):
    """写入结果时对首次、重复和迟到结果的处置。"""

    RECORDED = "recorded"
    REPLAYED = "replayed"
    DISCARDED_LATE = "discarded_late"


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class DispatchAttempt(StrictFrozenModel):
    """网络 I/O 前写入的可恢复 dispatch 意图。"""

    dispatch_id: NonEmptyId
    game_id: NonEmptyId
    turn_id: NonEmptyId
    actor_id: NonEmptyId
    operation_kind: DispatchOperationKind
    executor_id: NonEmptyId
    provider_idempotency_key: NonEmptyId
    recovery_policy: DispatchRecoveryPolicy
    request_hash: ContentHash
    lease_hash: ContentHash
    view_fingerprint: ContentHash
    deadline: datetime
    created_at: datetime
    updated_at: datetime
    status: DispatchStatus
    state_version: int = Field(ge=0)
    reason_code: NonEmptyId | None = None

    @field_validator("deadline", "created_at", "updated_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, field_name=info.field_name)


class DispatchResultRecord(StrictFrozenModel):
    """与 attempt 绑定、已脱敏且只允许一次写入的结构化结果。"""

    result_id: NonEmptyId
    dispatch_id: NonEmptyId
    request_hash: ContentHash
    lease_hash: ContentHash
    result_hash: ContentHash
    result_kind: NonEmptyId
    outcome: DispatchResultOutcome
    payload: Mapping[str, Any] = Field(default_factory=dict)
    recorded_at: datetime

    @field_validator("payload")
    @classmethod
    def _immutable_payload(cls, value: Any) -> Mapping[str, Any]:
        return _freeze_json_object(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_value(value)

    @field_validator("recorded_at")
    @classmethod
    def _aware_recorded_at(cls, value: datetime) -> datetime:
        return _require_aware(value, field_name="recorded_at")


__all__ = [
    "DispatchAttempt",
    "DispatchOperationKind",
    "DispatchRecoveryPolicy",
    "DispatchResultDisposition",
    "DispatchResultOutcome",
    "DispatchResultRecord",
    "DispatchStatus",
]
