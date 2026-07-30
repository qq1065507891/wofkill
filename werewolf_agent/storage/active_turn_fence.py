# -*- coding: utf-8 -*-
"""
定义活动回合 dispatch 围栏的显式仓储能力和纯准备逻辑。

作者: Project contributors
创建日期: 2026-07-31

使用示例:
    >>> updated_turn, fenced = prepare_active_turn_dispatch(
    ...     schedule, managed_turn, attempt, observed_at,
    ... )
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Protocol, cast

from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.dispatch import (
    ActiveTurnDispatchFence,
    DispatchAttempt,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    TerminalDisposition,
)
from werewolf_agent.player_agents.contracts.turns import AgentTurnStatus
from werewolf_agent.storage.autonomous_turns import prepare_active_finish
from werewolf_agent.storage.durable_dispatch import DispatchRecoveryBlocked


class ActiveTurnFenceError(RuntimeError):
    """所有活动回合 dispatch 围栏错误的稳定基类。"""

    code: ClassVar[str] = "active_turn_fence_error"


class ActiveTurnFenceUnsupported(ActiveTurnFenceError):
    """仓储未显式声明原子活动回合围栏能力。"""

    code = "active_turn_fence_unsupported"


class ActiveTurnFenceRejected(ActiveTurnFenceError):
    """围栏输入不再与当前活动回合身份一致。"""

    code = "active_turn_fence_rejected"


class ActiveTurnFenceTransactionError(ActiveTurnFenceError):
    """后端回滚活动回合围栏事务。"""

    code = "active_turn_fence_transaction_error"


class ActiveTurnFenceRepository(Protocol):
    """同一仓储对象提供的活动回合围栏原子操作。"""

    def supports_active_turn_fence(self) -> bool: ...

    def create_active_turn_dispatch(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        attempt: DispatchAttempt,
        observed_at: datetime,
    ) -> DispatchAttempt: ...

    def finish_active_turn_fenced(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule: ...


def require_active_turn_fence_repository(
    turn_repository: object,
    dispatch_repository: object,
) -> ActiveTurnFenceRepository:
    """要求三项 capability 同属一个显式支持的仓储。"""

    if turn_repository is not dispatch_repository:
        raise ActiveTurnFenceUnsupported(
            "repository does not support active turn fence",
        )
    for capability_name in (
        "supports_autonomous_turns",
        "supports_durable_dispatch",
        "supports_active_turn_fence",
    ):
        try:
            capability = getattr(turn_repository, capability_name)
            supported = capability() if callable(capability) else False
        except Exception:  # noqa: BLE001 - capability 异常必须映射为稳定边界。
            raise ActiveTurnFenceUnsupported(
                "repository does not support active turn fence",
            ) from None
        if not supported:
            raise ActiveTurnFenceUnsupported(
                "repository does not support active turn fence",
            )
    return cast(ActiveTurnFenceRepository, turn_repository)


def _rejected() -> ActiveTurnFenceRejected:
    """构造不包含运行时或私有上下文的稳定拒绝错误。"""

    return ActiveTurnFenceRejected("active turn fence rejected")


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _require_active_context(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> None:
    """校验调度和托管回合仍表示同一个可 dispatch 的活动身份。"""

    if schedule.status is not SerialPublicScheduleStatus.OPEN:
        raise _rejected()
    if schedule.active_turn_id != managed.turn.turn_id:
        raise _rejected()
    if managed.schedule_id != schedule.schedule_id:
        raise _rejected()
    if managed.turn.game_id != schedule.game_id:
        raise _rejected()
    if managed.turn.player_id != schedule.current_slot.player_id:
        raise _rejected()
    if managed.turn.window.window_id != schedule.window.window_id:
        raise _rejected()
    if managed.turn.window.version != schedule.window.version:
        raise _rejected()
    if managed.turn.window.deadline != schedule.window.deadline:
        raise _rejected()
    if managed.turn.revision.window_id != schedule.window.window_id:
        raise _rejected()
    if managed.turn.revision.window_version != schedule.window.version:
        raise _rejected()
    if managed.turn.revision.base_revision < schedule.window.opened_revision:
        raise _rejected()
    if managed.turn.status in {
        AgentTurnStatus.COMMITTED,
        AgentTurnStatus.CANCELLED,
        AgentTurnStatus.EXPIRED,
    }:
        raise _rejected()


def prepare_active_turn_dispatch(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    attempt: DispatchAttempt,
    observed_at: datetime,
) -> tuple[ManagedAgentTurn, DispatchAttempt]:
    """校验活动回合并准备其唯一的 dispatch 预约和持久化围栏。"""

    try:
        _require_active_context(schedule, managed)
        if (
            attempt.status is not DispatchStatus.PENDING
            or attempt.state_version != 0
            or attempt.reason_code is not None
            or attempt.active_turn_fence is not None
        ):
            raise _rejected()
        if attempt.game_id != schedule.game_id:
            raise _rejected()
        if attempt.turn_id != managed.turn.turn_id:
            raise _rejected()
        if attempt.actor_id != schedule.current_slot.player_id:
            raise _rejected()
        if attempt.lease_hash != managed.turn.model_lease_hash:
            raise _rejected()
        if attempt.view_fingerprint != managed.turn.revision.view_fingerprint:
            raise _rejected()
        if not _is_aware(observed_at):
            raise _rejected()
        if (
            attempt.deadline <= observed_at
            or managed.turn.window.deadline <= observed_at
        ):
            raise _rejected()
        if attempt.deadline > managed.turn.window.deadline:
            raise _rejected()

        updated_managed = managed.model_copy(
            update={
                "state_version": managed.state_version + 1,
                "updated_at": observed_at,
            },
        )
        fence = ActiveTurnDispatchFence(
            schedule_id=schedule.schedule_id,
            schedule_state_version=schedule.state_version,
            turn_state_version=updated_managed.state_version,
            window_id=schedule.window.window_id,
            window_version=schedule.window.version,
            base_game_revision=managed.turn.revision.base_revision,
        )
        fenced_attempt = attempt.model_copy(
            update={"active_turn_fence": fence},
        )
    except ActiveTurnFenceRejected:
        raise
    except (ValidationError, TypeError, ValueError, AttributeError):
        raise _rejected() from None
    return updated_managed, fenced_attempt


def prepare_fenced_active_finish(
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    attempts: tuple[DispatchAttempt, ...],
    terminal_status: AgentTurnStatus,
    disposition: TerminalDisposition,
    *,
    reason_code: str | None,
    now: datetime,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn, tuple[DispatchAttempt, ...]]:
    """准备围栏终态、可取消 dispatch 和公开调度的原子结果。"""

    try:
        _require_active_context(schedule, managed)
        if any(
            attempt.game_id != schedule.game_id
            or attempt.turn_id != managed.turn.turn_id
            for attempt in attempts
        ):
            raise _rejected()

        if terminal_status is AgentTurnStatus.COMMITTED and any(
            attempt.status in {
                DispatchStatus.PENDING,
                DispatchStatus.DISPATCHING,
                DispatchStatus.DISPATCHED,
            }
            for attempt in attempts
        ):
            raise DispatchRecoveryBlocked(schedule.game_id)

        updated_attempts = tuple(
            attempt.model_copy(
                update={
                    "status": DispatchStatus.CANCELLED,
                    "state_version": attempt.state_version + 1,
                    "reason_code": reason_code,
                    "updated_at": now,
                },
            )
            if terminal_status in {AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED}
            and attempt.status in {DispatchStatus.PENDING, DispatchStatus.DISPATCHING}
            else attempt
            for attempt in attempts
        )
        updated_schedule, updated_managed = prepare_active_finish(
            schedule,
            managed,
            terminal_status,
            disposition,
            reason_code=reason_code,
            now=now,
        )
    except (ActiveTurnFenceRejected, DispatchRecoveryBlocked):
        raise
    except (ValidationError, TypeError, ValueError, AttributeError):
        raise _rejected() from None
    return updated_schedule, updated_managed, updated_attempts


__all__ = [
    "ActiveTurnFenceError",
    "ActiveTurnFenceRejected",
    "ActiveTurnFenceRepository",
    "ActiveTurnFenceTransactionError",
    "ActiveTurnFenceUnsupported",
    "prepare_active_turn_dispatch",
    "prepare_fenced_active_finish",
    "require_active_turn_fence_repository",
]
