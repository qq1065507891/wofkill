# -*- coding: utf-8 -*-
"""
定义合法动作窗口、持久化玩家回合快照和允许的状态转换。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

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


class ConflictClass(StrEnum):
    SERIAL_PUBLIC = "serial_public"
    SERIAL_PRIVATE = "serial_private"
    COMMUTATIVE_PRIVATE = "commutative_private"
    TEAM_COORDINATOR = "team_coordinator"


class AgentTurnStatus(StrEnum):
    OPEN = "open"
    OBSERVING = "observing"
    THINKING = "thinking"
    WAITING_TOOL = "waiting_tool"
    COMPACTING = "compacting"
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class LegalActionWindow(StrictFrozenModel):
    window_id: NonEmptyId
    version: int = Field(ge=1)
    game_id: NonEmptyId
    task_type: NonEmptyId
    conflict_class: ConflictClass
    participant_ids: tuple[NonEmptyId, ...] = Field(min_length=1)
    legal_actions: tuple[NonEmptyId, ...] = Field(min_length=1)
    legal_target_ids: tuple[NonEmptyId, ...] = ()
    opened_revision: int = Field(ge=0)
    deadline: datetime

    @field_validator("deadline")
    @classmethod
    def _aware_deadline(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _unique_sets(self) -> LegalActionWindow:
        require_unique(self.participant_ids, field_name="participant_ids")
        require_unique(self.legal_actions, field_name="legal_actions")
        require_unique(self.legal_target_ids, field_name="legal_target_ids")
        return self


class TurnBudget(StrictFrozenModel):
    model_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    repairs: int = Field(ge=0)


class AgentTurn(StrictFrozenModel):
    turn_id: NonEmptyId
    game_id: NonEmptyId
    player_id: NonEmptyId
    role_id: NonEmptyId
    phase: NonEmptyId
    task_type: NonEmptyId
    revision: RevisionContext
    window: LegalActionWindow
    read_set: tuple[ReadReference, ...] = ()
    model_lease_hash: ContentHash
    budget: TurnBudget
    status: AgentTurnStatus
    idempotency_key: NonEmptyId

    @model_validator(mode="after")
    def _consistent_context(self) -> AgentTurn:
        if self.game_id != self.window.game_id:
            raise ValueError("turn game_id must match window game_id")
        if self.task_type != self.window.task_type:
            raise ValueError("turn task_type must match window task_type")
        if self.player_id not in self.window.participant_ids:
            raise ValueError("player must be a window participant")
        if self.revision.window_id != self.window.window_id:
            raise ValueError("revision window_id must match window")
        if self.revision.window_version != self.window.version:
            raise ValueError("revision window_version must match window")
        require_unique(
            (item.record_id for item in self.read_set),
            field_name="read_set record IDs",
        )
        return self


_TERMINAL = {
    AgentTurnStatus.COMMITTED,
    AgentTurnStatus.CANCELLED,
    AgentTurnStatus.EXPIRED,
}
_ALLOWED: dict[AgentTurnStatus, frozenset[AgentTurnStatus]] = {
    AgentTurnStatus.OPEN: frozenset({
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.OBSERVING: frozenset({
        AgentTurnStatus.THINKING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.THINKING: frozenset({
        AgentTurnStatus.WAITING_TOOL,
        AgentTurnStatus.COMPACTING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.WAITING_TOOL: frozenset({
        AgentTurnStatus.THINKING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.COMPACTING: frozenset({
        AgentTurnStatus.THINKING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.SUBMITTED: frozenset({
        AgentTurnStatus.VALIDATING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.VALIDATING: frozenset({
        AgentTurnStatus.COMMITTED,
        AgentTurnStatus.REPAIRING,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    AgentTurnStatus.REPAIRING: frozenset({
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }),
    **{status: frozenset() for status in _TERMINAL},
}


def transition_turn(turn: AgentTurn, next_status: AgentTurnStatus) -> AgentTurn:
    if next_status not in _ALLOWED[turn.status]:
        raise ValueError(
            f"illegal agent turn transition: {turn.status.value} -> {next_status.value}"
        )
    return AgentTurn.model_validate({**turn.model_dump(), "status": next_status})
