# -*- coding: utf-8 -*-
"""
定义自主玩家终端提交所需的事件、审计、outbox 请求和结果契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.proposals import SpeechProposalEnvelope
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.revisions import ReadReference


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({
            key: _freeze_json_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("value must be finite JSON data")


def _freeze_json_object(value: Any) -> Mapping[str, Any]:
    frozen = _freeze_json_value(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("value must be a JSON object")
    return frozen


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


class EventCandidate(StrictFrozenModel):
    """由 Host 在仓储事务中补齐权威身份的事件候选。"""

    type: NonEmptyId
    payload: Mapping[str, Any] = Field(default_factory=dict)
    visibility: NonEmptyId | None = None

    @field_validator("payload")
    @classmethod
    def _immutable_payload(cls, value: Any) -> Mapping[str, Any]:
        return _freeze_json_object(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_value(value)


class CriticalAuditRecord(StrictFrozenModel):
    """必须与 truth 一起落盘的结构化审计载荷。"""

    audit_id: NonEmptyId
    kind: NonEmptyId
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _immutable_payload(cls, value: Any) -> Mapping[str, Any]:
        return _freeze_json_object(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_value(value)


class ProjectionOutboxRecord(StrictFrozenModel):
    """提交后由投影 worker 消费的幂等请求。"""

    outbox_id: NonEmptyId
    kind: NonEmptyId
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _immutable_payload(cls, value: Any) -> Mapping[str, Any]:
        return _freeze_json_object(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_value(value)


class CommitTurnRequest(StrictFrozenModel):
    """绑定到一个 turn 的原子提交请求。"""

    game_id: NonEmptyId
    turn_id: NonEmptyId
    idempotency_key: NonEmptyId
    base_game_revision: int = Field(ge=0)
    read_set: tuple[ReadReference, ...] = ()
    proposal: SpeechProposalEnvelope
    rule_result: Mapping[str, Any] = Field(default_factory=dict)
    event: EventCandidate
    public_record: PublicSpeechRecord | None = None
    critical_audit_records: tuple[CriticalAuditRecord, ...] = ()
    projection_outbox_records: tuple[ProjectionOutboxRecord, ...] = ()

    @field_validator("rule_result")
    @classmethod
    def _immutable_rule_result(cls, value: Any) -> Mapping[str, Any]:
        return _freeze_json_object(value)

    @field_serializer("rule_result")
    def _serialize_rule_result(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _thaw_json_value(value)

    @model_validator(mode="after")
    def _bind_context(self) -> Self:
        if self.proposal.turn_id != self.turn_id:
            raise ValueError("proposal turn_id must match commit turn_id")
        if self.public_record is not None:
            if self.public_record.game_id != self.game_id:
                raise ValueError("public record game_id must match commit game_id")
            if self.public_record.turn_id != self.turn_id:
                raise ValueError("public record turn_id must match commit turn_id")
        require_unique(
            (reference.record_id for reference in self.read_set),
            field_name="read set record IDs",
        )
        require_unique(
            (record.audit_id for record in self.critical_audit_records),
            field_name="audit IDs",
        )
        require_unique(
            (record.outbox_id for record in self.projection_outbox_records),
            field_name="outbox IDs",
        )
        return self


class CommitResult(StrictFrozenModel):
    """一次成功提交或幂等重放的稳定结果。"""

    game_id: NonEmptyId
    turn_id: NonEmptyId
    idempotency_key: NonEmptyId
    committed_revision: int = Field(ge=1)
    event_id: NonEmptyId
    public_record_id: NonEmptyId | None = None
    audit_ids: tuple[NonEmptyId, ...] = ()
    outbox_ids: tuple[NonEmptyId, ...] = ()
    request_hash: ContentHash
    replayed: bool = False

    @model_validator(mode="after")
    def _unique_result_ids(self) -> Self:
        require_unique(self.audit_ids, field_name="result audit IDs")
        require_unique(self.outbox_ids, field_name="result outbox IDs")
        return self


__all__ = [
    "CommitResult",
    "CommitTurnRequest",
    "CriticalAuditRecord",
    "EventCandidate",
    "ProjectionOutboxRecord",
]
