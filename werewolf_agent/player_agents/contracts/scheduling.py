# -*- coding: utf-8 -*-
"""
定义串行公开调度、回合准入和持久化托管回合的严格契约。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from werewolf_agent.player_agents.contracts._base import (
    ContentHash,
    NonEmptyId,
    StrictFrozenModel,
    require_unique,
)
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)


class SerialPublicScheduleStatus(StrEnum):
    """串行公开调度的持久化状态。"""

    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TerminalDisposition(StrEnum):
    """活动回合终态对当前调度位置的处理方式。"""

    ADVANCE = "advance"
    REPLACE = "replace"
    CLOSE = "close"


class SerialPublicSlot(StrictFrozenModel):
    """绑定一个公开发言顺序与参与者。"""

    ordinal: int = Field(ge=0)
    player_id: NonEmptyId


def _require_aware(value: datetime, field_name: str) -> datetime:
    """拒绝缺少时区信息的时间戳。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class SerialPublicSchedule(StrictFrozenModel):
    """保存串行公开窗口的有序位置、活动回合和 CAS 版本。"""

    schedule_id: NonEmptyId
    game_id: NonEmptyId
    window: LegalActionWindow
    slots: tuple[SerialPublicSlot, ...] = Field(min_length=1)
    next_slot_ordinal: int = Field(ge=0)
    active_turn_id: NonEmptyId | None = None
    status: SerialPublicScheduleStatus
    state_version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @property
    def current_slot(self) -> SerialPublicSlot:
        """返回开放调度的当前发言位置。"""
        if self.status is not SerialPublicScheduleStatus.OPEN:
            raise ValueError("serial public schedule is not open")
        return self.slots[self.next_slot_ordinal]

    @model_validator(mode="after")
    def _consistent_schedule(self) -> Self:
        if self.window.conflict_class is not ConflictClass.SERIAL_PUBLIC:
            raise ValueError("schedule window must be serial_public")
        if self.game_id != self.window.game_id:
            raise ValueError("schedule game_id must match window game_id")

        ordinals = tuple(slot.ordinal for slot in self.slots)
        if ordinals != tuple(range(len(self.slots))):
            raise ValueError("slot ordinals must be contiguous from zero")

        player_ids = tuple(slot.player_id for slot in self.slots)
        require_unique(player_ids, field_name="slot player IDs")
        if set(player_ids) != set(self.window.participant_ids):
            raise ValueError("slot players must match window participants")

        if self.next_slot_ordinal > len(self.slots):
            raise ValueError("next_slot_ordinal exceeds slot count")
        if self.status is SerialPublicScheduleStatus.OPEN:
            if self.next_slot_ordinal >= len(self.slots):
                raise ValueError("open schedule must have a current slot")
        elif self.active_turn_id is not None:
            raise ValueError("terminal schedule cannot have an active turn")

        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class TurnAdmission(StrictFrozenModel):
    """携带当前调度位置创建新回合所需的调用方输入。"""

    turn_id: NonEmptyId
    player_id: NonEmptyId
    role_id: NonEmptyId
    phase: NonEmptyId
    revision: RevisionContext
    read_set: tuple[ReadReference, ...] = ()
    model_lease_hash: ContentHash
    budget: TurnBudget
    idempotency_key: NonEmptyId

    @model_validator(mode="after")
    def _unique_reads(self) -> Self:
        require_unique(
            (item.record_id for item in self.read_set),
            field_name="read_set record IDs",
        )
        return self


class ManagedAgentTurn(StrictFrozenModel):
    """封装持久化玩家回合及其独立 CAS 版本和审计时间。"""

    schedule_id: NonEmptyId
    turn: AgentTurn
    state_version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    terminal_reason: NonEmptyId | None = None

    @model_validator(mode="after")
    def _consistent_timestamps(self) -> Self:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self
