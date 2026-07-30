# -*- coding: utf-8 -*-
"""
定义自主玩家托管回合仓储能力及串行公开生命周期的纯准备逻辑。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Protocol, cast

from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    transition_turn,
)


class AutonomousTurnError(RuntimeError):
    """所有自主玩家托管回合错误的稳定基类。"""

    code: ClassVar[str] = "autonomous_turn_error"


class AutonomousTurnsUnsupported(AutonomousTurnError):
    """仓储未显式声明自主玩家托管回合能力。"""

    code = "autonomous_turns_unsupported"


class ScheduleNotFound(AutonomousTurnError):
    """请求的串行公开调度不存在。"""

    code = "schedule_not_found"


class ManagedTurnNotFound(AutonomousTurnError):
    """请求的托管玩家回合不存在。"""

    code = "managed_turn_not_found"


class ScheduleStateConflict(AutonomousTurnError):
    """调度 CAS 版本已过期。"""

    code = "schedule_state_conflict"


class TurnStateConflict(AutonomousTurnError):
    """托管回合 CAS 版本已过期。"""

    code = "turn_state_conflict"


class InvalidScheduleTransition(AutonomousTurnError):
    """调度或托管回合状态迁移不合法。"""

    code = "invalid_schedule_transition"


class InvalidTurnAdmission(AutonomousTurnError):
    """回合准入输入与当前公开调度不匹配。"""

    code = "invalid_turn_admission"


class AutonomousTurnTransactionError(AutonomousTurnError):
    """自主玩家托管回合事务失败。"""

    code = "autonomous_turn_transaction_error"


class AutonomousTurnRepository(Protocol):
    """实现托管回合的仓储必须显式满足的接口。"""

    def supports_autonomous_turns(self) -> bool: ...

    def create_serial_public_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule: ...

    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None: ...

    def load_active_serial_public_schedule(
        self,
        game_id: str,
    ) -> SerialPublicSchedule | None: ...

    def list_open_serial_public_schedules(
        self,
    ) -> tuple[SerialPublicSchedule, ...]: ...

    def admit_serial_public_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn: ...

    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn: ...

    def finish_active_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule: ...

    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None: ...


def require_autonomous_turn_repository(
    repository: object,
) -> AutonomousTurnRepository:
    """只接受具有可调用 capability 且明确返回真值的仓储。"""

    supports = getattr(repository, "supports_autonomous_turns", None)
    if not callable(supports):
        raise AutonomousTurnsUnsupported(
            "repository does not support autonomous turns",
        )
    try:
        supported = supports()
    except Exception as exc:
        raise AutonomousTurnsUnsupported(
            "repository does not support autonomous turns",
        ) from exc
    if not supported:
        raise AutonomousTurnsUnsupported(
            "repository does not support autonomous turns",
        )
    return cast(AutonomousTurnRepository, repository)


def _invalid_admission() -> InvalidTurnAdmission:
    """构造不含输入 payload 的稳定准入错误。"""

    return InvalidTurnAdmission("invalid autonomous turn admission")


def _invalid_transition() -> InvalidScheduleTransition:
    """构造不含内部状态详情的稳定迁移错误。"""

    return InvalidScheduleTransition("invalid autonomous turn transition")


def _require_active_identity(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> None:
    if schedule.active_turn_id != managed.turn.turn_id:
        raise _invalid_transition()
    if managed.schedule_id != schedule.schedule_id:
        raise _invalid_transition()


def prepare_serial_public_admission(
    schedule: SerialPublicSchedule,
    admission: TurnAdmission,
    now: datetime,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    """校验并准备一个绑定当前公开 slot 的新托管回合。"""

    try:
        if schedule.status is not SerialPublicScheduleStatus.OPEN:
            raise _invalid_admission()
        if schedule.active_turn_id is not None:
            raise _invalid_admission()
        # player-current-slot 绑定必须由主机在这里完成，避免调用方绕过顺序。
        if admission.player_id != schedule.current_slot.player_id:
            raise _invalid_admission()
        if admission.revision.window_id != schedule.window.window_id:
            raise _invalid_admission()
        if admission.revision.window_version != schedule.window.version:
            raise _invalid_admission()
        if admission.revision.base_revision < schedule.window.opened_revision:
            raise _invalid_admission()

        turn = AgentTurn(
            turn_id=admission.turn_id,
            game_id=schedule.game_id,
            player_id=admission.player_id,
            role_id=admission.role_id,
            phase=admission.phase,
            task_type=schedule.window.task_type,
            revision=admission.revision,
            window=schedule.window,
            read_set=admission.read_set,
            model_lease_hash=admission.model_lease_hash,
            budget=admission.budget,
            status=AgentTurnStatus.OPEN,
            idempotency_key=admission.idempotency_key,
        )
        managed = ManagedAgentTurn(
            schedule_id=schedule.schedule_id,
            turn=turn,
            state_version=0,
            created_at=now,
            updated_at=now,
        )
        updated_schedule = schedule.model_copy(
            update={
                "active_turn_id": turn.turn_id,
                "state_version": schedule.state_version + 1,
                "updated_at": now,
            },
        )
    except InvalidTurnAdmission:
        raise
    except (ValidationError, TypeError, ValueError, AttributeError) as exc:
        raise _invalid_admission() from exc
    return updated_schedule, managed


def prepare_active_transition(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    next_status: AgentTurnStatus,
) -> ManagedAgentTurn:
    """校验活动身份并准备一次非终态回合迁移。"""

    try:
        _require_active_identity(schedule, managed)
        if next_status in {
            AgentTurnStatus.COMMITTED,
            AgentTurnStatus.CANCELLED,
            AgentTurnStatus.EXPIRED,
        }:
            raise _invalid_transition()
        updated_turn = transition_turn(managed.turn, next_status)
        return managed.model_copy(
            update={
                "turn": updated_turn,
                "state_version": managed.state_version + 1,
            },
        )
    except InvalidScheduleTransition:
        raise
    except (ValidationError, TypeError, ValueError, AttributeError) as exc:
        raise _invalid_transition() from exc


def prepare_active_finish(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    terminal_status: AgentTurnStatus,
    disposition: TerminalDisposition,
    *,
    reason_code: str | None,
    now: datetime,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    """准备终态回合并原子地计算公开 slot 的后续调度状态。"""

    try:
        _require_active_identity(schedule, managed)
        if terminal_status not in {
            AgentTurnStatus.COMMITTED,
            AgentTurnStatus.CANCELLED,
            AgentTurnStatus.EXPIRED,
        }:
            raise _invalid_transition()
        updated_turn = transition_turn(managed.turn, terminal_status)

        if disposition is TerminalDisposition.ADVANCE:
            next_ordinal = schedule.next_slot_ordinal + 1
            next_status = (
                SerialPublicScheduleStatus.CLOSED
                if next_ordinal == len(schedule.slots)
                else SerialPublicScheduleStatus.OPEN
            )
        elif disposition is TerminalDisposition.REPLACE:
            next_ordinal = schedule.next_slot_ordinal
            next_status = SerialPublicScheduleStatus.OPEN
        elif disposition is TerminalDisposition.CLOSE:
            next_ordinal = schedule.next_slot_ordinal
            next_status = SerialPublicScheduleStatus.CANCELLED
        else:
            raise _invalid_transition()

        updated_managed = managed.model_copy(
            update={
                "turn": updated_turn,
                "state_version": managed.state_version + 1,
                "updated_at": now,
                "terminal_reason": reason_code,
            },
        )
        updated_schedule = schedule.model_copy(
            update={
                "active_turn_id": None,
                "next_slot_ordinal": next_ordinal,
                "status": next_status,
                "state_version": schedule.state_version + 1,
                "updated_at": now,
            },
        )
    except InvalidScheduleTransition:
        raise
    except (ValidationError, TypeError, ValueError, AttributeError) as exc:
        raise _invalid_transition() from exc
    return updated_schedule, updated_managed


__all__ = [
    "AutonomousTurnError",
    "AutonomousTurnRepository",
    "AutonomousTurnTransactionError",
    "AutonomousTurnsUnsupported",
    "InvalidScheduleTransition",
    "InvalidTurnAdmission",
    "ManagedTurnNotFound",
    "ScheduleNotFound",
    "ScheduleStateConflict",
    "TurnStateConflict",
    "prepare_active_finish",
    "prepare_active_transition",
    "prepare_serial_public_admission",
    "require_autonomous_turn_repository",
]
