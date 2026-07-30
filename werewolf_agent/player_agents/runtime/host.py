# -*- coding: utf-8 -*-
"""
协调自主玩家托管回合的恢复门禁、生命周期、取消与确定性过期。

作者: Project contributors
创建日期: 2026-07-30
修改日期: 2026-07-30
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import ClassVar

from werewolf_agent.player_agents.contracts.dispatch import DispatchStatus
from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import AgentTurnStatus
from werewolf_agent.player_agents.runtime.serial_public import SerialPublicScheduler
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnRepository,
    InvalidScheduleTransition,
    InvalidTurnAdmission,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchReconciler,
    DispatchRecoveryBlocked,
    DurableDispatchRepository,
    RecoveryReport,
    require_durable_dispatch_repository,
)


class HostRuntimeError(RuntimeError):
    """主机托管回合协调错误的稳定基类。"""

    code: ClassVar[str] = "host_runtime_error"


class HostRecoveryRequired(HostRuntimeError):
    """当前进程尚未恢复该游戏。"""

    code = "host_recovery_required"


class HostRecoveryBlocked(HostRuntimeError):
    """该游戏仍有未决 durable dispatch，恢复屏障关闭。"""

    code = "host_recovery_blocked"


class HostRuntime:
    """持有进程内恢复状态并协调持久化托管回合。"""

    def __init__(
        self,
        turn_repository: AutonomousTurnRepository | object,
        dispatch_repository: DurableDispatchRepository | object,
        reconciler: DispatchReconciler,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._scheduler = SerialPublicScheduler(turn_repository)
        self._dispatch_repository = require_durable_dispatch_repository(
            dispatch_repository,
        )
        self._reconciler = reconciler
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._recovered_games: set[str] = set()
        self._blocked_games: set[str] = set()
        self._recovery_reports: dict[str, RecoveryReport] = {}

    def create_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule:
        """创建新调度，并仅为该新游戏打开当前进程的准入门。"""

        created = self._scheduler.create_schedule(schedule)
        self._blocked_games.discard(created.game_id)
        self._recovered_games.add(created.game_id)
        return created

    def recover_game(self, game_id: str) -> RecoveryReport:
        """协调未决 dispatch；只有报告与仓储屏障同时开放才准入。"""

        report = self._reconciler.reconcile_game(game_id)
        self._recovery_reports[game_id] = report
        if report.barrier_open:
            try:
                self._dispatch_repository.assert_dispatch_allowed(game_id)
            except DispatchRecoveryBlocked:
                self._recovered_games.discard(game_id)
                self._blocked_games.add(game_id)
            else:
                self._blocked_games.discard(game_id)
                self._recovered_games.add(game_id)
        else:
            self._recovered_games.discard(game_id)
            self._blocked_games.add(game_id)
        return report

    def admit_next_turn(
        self,
        schedule_id: str,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        """恢复通过且窗口未过期时准入当前公开 slot。"""

        schedule = self._scheduler.require_schedule(schedule_id)
        self._require_recovered(schedule.game_id)
        self._assert_dispatch_allowed(schedule.game_id)
        if schedule.window.deadline <= self._aware_clock_now():
            raise InvalidTurnAdmission("serial public admission deadline expired")
        return self._scheduler.admit_next_turn(schedule_id, admission)

    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn:
        """在恢复和 dispatch 屏障开放时推进一个活动回合。"""

        managed = self._scheduler.require_managed_turn(turn_id)
        self._require_recovered(managed.turn.game_id)
        self._assert_dispatch_allowed(managed.turn.game_id)
        return self._scheduler.transition_active_turn(
            turn_id,
            expected_turn_version,
            next_status,
        )

    def complete_active_turn(self, schedule_id: str) -> SerialPublicSchedule:
        """在外部权威提交成功后把 VALIDATING 回合标记为已提交。"""

        schedule, managed = self._require_active_schedule_turn(schedule_id)
        self._require_recovered(schedule.game_id)
        return self._scheduler.finish_active_turn(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            None,
        )

    def cancel_active_turn(
        self,
        schedule_id: str,
        reason_code: str,
        disposition: TerminalDisposition,
    ) -> SerialPublicSchedule:
        """先取消尚未完成的 dispatch，再按调用方处置终结活动回合。"""

        schedule, managed = self._require_active_schedule_turn(schedule_id)
        self._require_recovered(schedule.game_id)
        self._cancel_turn_dispatches(managed, reason_code)
        return self._scheduler.finish_active_turn(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.CANCELLED,
            disposition,
            reason_code,
        )

    def expire_due_turns(
        self,
        now: datetime,
    ) -> tuple[SerialPublicSchedule, ...]:
        """按游戏与调度 ID 顺序过期已到截止时间的已恢复活动回合。"""

        self._require_aware(now)
        schedules = sorted(
            self._list_open_schedules(),
            key=lambda item: (item.game_id, item.schedule_id),
        )
        changed: list[SerialPublicSchedule] = []
        for schedule in schedules:
            if (
                schedule.game_id not in self._recovered_games
                or schedule.active_turn_id is None
                or schedule.window.deadline > now
            ):
                continue
            managed = self._scheduler.require_managed_turn(schedule.active_turn_id)
            self._cancel_turn_dispatches(managed, "deadline_expired")
            changed.append(
                self._scheduler.finish_active_turn(
                    schedule.schedule_id,
                    schedule.state_version,
                    managed.turn.turn_id,
                    managed.state_version,
                    AgentTurnStatus.EXPIRED,
                    TerminalDisposition.ADVANCE,
                    "deadline_expired",
                ),
            )
        return tuple(changed)

    def load_active_turn(self, game_id: str) -> ManagedAgentTurn | None:
        """读取当前活动回合；无活动调度时返回空值。"""

        try:
            return self._scheduler.load_active_turn(game_id)
        except InvalidScheduleTransition:
            return None

    def _require_active_schedule_turn(
        self,
        schedule_id: str,
    ) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
        schedule = self._scheduler.require_schedule(schedule_id)
        if schedule.active_turn_id is None:
            raise InvalidScheduleTransition(
                "serial public schedule has no active turn",
            )
        managed = self._scheduler.require_managed_turn(schedule.active_turn_id)
        if managed.schedule_id != schedule.schedule_id:
            raise InvalidScheduleTransition(
                "serial public schedule has no active turn",
            )
        return schedule, managed

    def _list_open_schedules(self) -> tuple[SerialPublicSchedule, ...]:
        return self._scheduler.list_open_schedules()

    def _cancel_turn_dispatches(
        self,
        managed: ManagedAgentTurn,
        reason_code: str,
    ) -> None:
        attempts = self._dispatch_repository.list_dispatches_for_turn(
            managed.turn.game_id,
            managed.turn.turn_id,
        )
        for attempt in attempts:
            if attempt.status in {
                DispatchStatus.PENDING,
                DispatchStatus.DISPATCHING,
            }:
                self._dispatch_repository.cancel_dispatch(
                    attempt.dispatch_id,
                    attempt.state_version,
                    reason_code,
                )

    def _require_recovered(self, game_id: str) -> None:
        if game_id in self._recovered_games:
            return
        if game_id in self._blocked_games:
            raise HostRecoveryBlocked(f"host recovery blocked for {game_id}")
        report = self._recovery_reports.get(game_id)
        if report is not None and not report.barrier_open:
            raise HostRecoveryBlocked(f"host recovery blocked for {game_id}")
        raise HostRecoveryRequired(f"host recovery required for {game_id}")

    def _assert_dispatch_allowed(self, game_id: str) -> None:
        try:
            self._dispatch_repository.assert_dispatch_allowed(game_id)
        except DispatchRecoveryBlocked as exc:
            self._blocked_games.add(game_id)
            raise HostRecoveryBlocked(
                f"host recovery blocked for {game_id}",
            ) from exc

    def _aware_clock_now(self) -> datetime:
        now = self._clock()
        self._require_aware(now)
        return now

    @staticmethod
    def _require_aware(value: datetime) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("now must be timezone-aware")


__all__ = [
    "HostRecoveryBlocked",
    "HostRecoveryRequired",
    "HostRuntime",
    "HostRuntimeError",
]
