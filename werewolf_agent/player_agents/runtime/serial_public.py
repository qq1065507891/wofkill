# -*- coding: utf-8 -*-
"""
为串行公开调度提供不暴露仓储对象的窄生命周期门面。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import AgentTurnStatus
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnRepository,
    InvalidScheduleTransition,
    ManagedTurnNotFound,
    ScheduleNotFound,
    require_autonomous_turn_repository,
)


def _inactive_turn() -> InvalidScheduleTransition:
    """构造不泄露调度状态的稳定活动回合错误。"""

    return InvalidScheduleTransition("serial public schedule has no active turn")


class SerialPublicScheduler:
    """把串行公开调度操作约束为仓储支持的窄接口。"""

    def __init__(self, repository: AutonomousTurnRepository | object) -> None:
        self._repository = require_autonomous_turn_repository(repository)

    def create_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule:
        """持久化一个尚未准入回合的串行公开调度。"""

        return self._repository.create_serial_public_schedule(schedule)

    def require_schedule(self, schedule_id: str) -> SerialPublicSchedule:
        """加载调度；不存在时返回稳定仓储错误。"""

        schedule = self._repository.load_serial_public_schedule(schedule_id)
        if schedule is None:
            raise ScheduleNotFound("serial public schedule not found")
        return schedule

    def list_open_schedules(self) -> tuple[SerialPublicSchedule, ...]:
        """返回仓储中开放调度的不可变快照。"""

        return self._repository.list_open_serial_public_schedules()

    def require_managed_turn(self, turn_id: str) -> ManagedAgentTurn:
        """加载托管回合；不存在时返回稳定仓储错误。"""

        managed = self._repository.load_managed_turn(turn_id)
        if managed is None:
            raise ManagedTurnNotFound("managed agent turn not found")
        return managed

    def load_active_turn(self, game_id: str) -> ManagedAgentTurn:
        """加载指定游戏当前调度绑定的活动回合。"""

        schedule = self._repository.load_active_serial_public_schedule(game_id)
        if schedule is None or schedule.active_turn_id is None:
            raise _inactive_turn()
        managed = self.require_managed_turn(schedule.active_turn_id)
        if managed.schedule_id != schedule.schedule_id:
            raise _inactive_turn()
        return managed

    def admit_next_turn(
        self,
        schedule_id: str,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        """按当前调度版本准入唯一的当前 slot。"""

        schedule = self.require_schedule(schedule_id)
        return self._repository.admit_serial_public_turn(
            schedule_id,
            schedule.state_version,
            admission,
        )

    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn:
        """确认回合仍由活动调度持有后执行非终态迁移。"""

        managed = self.require_managed_turn(turn_id)
        schedule = self.require_schedule(managed.schedule_id)
        if schedule.active_turn_id != turn_id:
            raise _inactive_turn()
        return self._repository.transition_active_turn(
            turn_id,
            expected_turn_version,
            next_status,
        )

    def finish_active_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule:
        """使用 Host 首次捕获的 CAS 身份原子结束指定活动回合。"""

        return self._repository.finish_active_turn(
            schedule_id,
            expected_schedule_version,
            turn_id,
            expected_turn_version,
            terminal_status,
            disposition,
            reason_code,
        )


__all__ = ["SerialPublicScheduler"]
