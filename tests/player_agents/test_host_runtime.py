# -*- coding: utf-8 -*-
"""
验证串行公开调度门面与主机托管回合的恢复、围栏与生命周期语义。

作者: Project contributors
创建日期: 2026-07-30
修改日期: 2026-07-31
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

from werewolf_agent.core.models import GameState
from werewolf_agent.player_agents.contracts.dispatch import (
    ActiveTurnDispatchFence,
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    SerialPublicSlot,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)
from werewolf_agent.player_agents.runtime.host import (
    HostRecoveryBlocked,
    HostRecoveryRequired,
    HostRuntime,
    HostRuntimeError,
)
from werewolf_agent.player_agents.runtime.serial_public import SerialPublicScheduler
from werewolf_agent.storage import memory_store
from werewolf_agent.storage.active_turn_fence import (
    ActiveTurnFenceRejected,
    ActiveTurnFenceUnsupported,
)
from werewolf_agent.storage.autonomous_turns import (
    InvalidScheduleTransition,
    InvalidTurnAdmission,
    ManagedTurnNotFound,
    ScheduleNotFound,
    ScheduleStateConflict,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchReconciler,
    RecoveryResolution,
    RecoveryResolutionKind,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository

HASH = "a" * 64
NOW = datetime(2026, 7, 30, 10, tzinfo=timezone.utc)
DEADLINE = NOW + timedelta(hours=1)


class PendingResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        return RecoveryResolution(kind=RecoveryResolutionKind.PENDING)


class UnsafeResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        return RecoveryResolution(
            kind=RecoveryResolutionKind.UNSAFE,
            reason_code="unsafe_provider_binding",
        )


class ErrorResolver:
    def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
        raise RuntimeError("resolver failed")


class FenceSpyRepository(InMemoryGameRepository):
    """记录 Host 是否绕过活动回合围栏。"""

    def __init__(self) -> None:
        super().__init__()
        self.dispatch_scan_calls = 0
        self.fenced_finish_calls = 0
        self.unfenced_finish_calls = 0

    def list_dispatches_for_turn(
        self,
        game_id: str,
        turn_id: str,
    ) -> list[DispatchAttempt]:
        self.dispatch_scan_calls += 1
        raise AssertionError("HostRuntime must not pre-scan dispatches")

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
        self.unfenced_finish_calls += 1
        raise AssertionError("HostRuntime must use the fenced terminal call")

    def finish_active_turn_fenced(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule:
        self.fenced_finish_calls += 1
        return super().finish_active_turn_fenced(
            schedule_id,
            expected_schedule_version,
            turn_id,
            expected_turn_version,
            terminal_status,
            disposition,
            reason_code,
        )


def _window(
    *,
    game_id: str = "game-1",
    window_id: str = "speech-d1",
    deadline: datetime = DEADLINE,
) -> LegalActionWindow:
    return LegalActionWindow(
        window_id=window_id,
        version=1,
        game_id=game_id,
        task_type="day_speech",
        conflict_class=ConflictClass.SERIAL_PUBLIC,
        participant_ids=("p01", "p02"),
        legal_actions=("speech",),
        legal_target_ids=("p01", "p02"),
        opened_revision=4,
        deadline=deadline,
    )


def _schedule(
    *,
    game_id: str = "game-1",
    schedule_id: str = "schedule-1",
    deadline: datetime = DEADLINE,
) -> SerialPublicSchedule:
    return SerialPublicSchedule(
        schedule_id=schedule_id,
        game_id=game_id,
        window=_window(
            game_id=game_id,
            window_id=f"{schedule_id}:window",
            deadline=deadline,
        ),
        slots=(
            SerialPublicSlot(ordinal=0, player_id="p01"),
            SerialPublicSlot(ordinal=1, player_id="p02"),
        ),
        next_slot_ordinal=0,
        status=SerialPublicScheduleStatus.OPEN,
        state_version=0,
        created_at=NOW,
        updated_at=NOW,
    )


def _admission(
    *,
    schedule: SerialPublicSchedule | None = None,
    turn_id: str = "turn-1",
    player_id: str = "p01",
    idempotency_key: str | None = None,
) -> TurnAdmission:
    bound = schedule or _schedule()
    return TurnAdmission(
        turn_id=turn_id,
        player_id=player_id,
        role_id="villager",
        phase="day_discussion",
        revision=RevisionContext(
            base_revision=4,
            window_id=bound.window.window_id,
            window_version=bound.window.version,
            view_fingerprint=HASH,
        ),
        read_set=(
            ReadReference(record_id="public-4", revision=4, content_hash=HASH),
        ),
        model_lease_hash=HASH,
        budget=TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        idempotency_key=idempotency_key or f"{turn_id}:submit",
    )


def _attempt(
    managed: ManagedAgentTurn,
    *,
    dispatch_id: str = "dispatch-1",
    status: DispatchStatus = DispatchStatus.PENDING,
) -> DispatchAttempt:
    return DispatchAttempt(
        dispatch_id=dispatch_id,
        game_id=managed.turn.game_id,
        turn_id=managed.turn.turn_id,
        actor_id=managed.turn.player_id,
        operation_kind=DispatchOperationKind.MODEL,
        executor_id="mock-provider",
        provider_idempotency_key=f"provider:{dispatch_id}",
        recovery_policy=DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        request_hash=HASH,
        lease_hash=managed.turn.model_lease_hash,
        view_fingerprint=managed.turn.revision.view_fingerprint,
        deadline=managed.turn.window.deadline,
        created_at=NOW,
        updated_at=NOW,
        status=status,
        state_version=0,
    )


def _repository(*schedules: SerialPublicSchedule) -> InMemoryGameRepository:
    repository = InMemoryGameRepository()
    for schedule in schedules:
        repository.save_game(GameState(game_id=schedule.game_id))
        repository.create_serial_public_schedule(schedule)
    return repository


def _host(
    repository: InMemoryGameRepository,
    *,
    resolver: object | None = None,
    now: datetime = NOW,
) -> HostRuntime:
    reconciler = DispatchReconciler(
        repository,
        resolver=resolver or PendingResolver(),  # type: ignore[arg-type]
    )
    return HostRuntime(
        repository,
        repository,
        reconciler,
        clock=lambda: now,
    )


def _fresh_host_with_schedule() -> tuple[
    InMemoryGameRepository,
    HostRuntime,
    SerialPublicSchedule,
]:
    repository = InMemoryGameRepository()
    schedule = _schedule()
    repository.save_game(GameState(game_id=schedule.game_id))
    host = _host(repository)
    created = host.create_schedule(schedule)
    return repository, host, created


def _host_with_active_turn() -> tuple[
    InMemoryGameRepository,
    HostRuntime,
    ManagedAgentTurn,
]:
    repository, host, schedule = _fresh_host_with_schedule()
    managed = host.admit_next_turn(schedule.schedule_id, _admission(schedule=schedule))
    return repository, host, managed


def _host_with_fenceable_active_turn(
    *,
    repository_type: type[InMemoryGameRepository] = InMemoryGameRepository,
) -> tuple[
    InMemoryGameRepository,
    HostRuntime,
    ManagedAgentTurn,
]:
    repository = repository_type()
    opened_at = datetime.now(timezone.utc)
    schedule = _schedule(deadline=opened_at + timedelta(hours=1))
    repository.save_game(GameState(game_id=schedule.game_id))
    admission_host = _host(repository, now=opened_at)
    created = admission_host.create_schedule(schedule)
    managed = admission_host.admit_next_turn(
        created.schedule_id,
        _admission(schedule=created),
    )
    host = _host(repository, now=managed.updated_at)
    host.recover_game(managed.turn.game_id)
    return repository, host, managed


def _reserve(
    host: HostRuntime,
    managed: ManagedAgentTurn,
    *,
    dispatch_id: str = "dispatch-1",
) -> DispatchAttempt:
    """通过 Host 的生产围栏路径预约活动回合 dispatch。"""

    return host.create_active_turn_dispatch(
        managed.schedule_id,
        _attempt(managed, dispatch_id=dispatch_id),
    )


def _to_validating(host: HostRuntime, managed: ManagedAgentTurn) -> ManagedAgentTurn:
    for status in (
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.THINKING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.VALIDATING,
    ):
        managed = host.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            status,
        )
    return managed


def _replace_with_validating_turn(
    repository: InMemoryGameRepository,
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> ManagedAgentTurn:
    replaced = repository.finish_active_turn(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.REPLACE,
        "interleaved_replace",
    )
    replacement = repository.admit_serial_public_turn(
        replaced.schedule_id,
        replaced.state_version,
        _admission(
            schedule=replaced,
            turn_id="turn-2",
            idempotency_key="turn-2:submit",
        ),
    )
    for status in (
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.THINKING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.VALIDATING,
    ):
        replacement = repository.transition_active_turn(
            replacement.turn.turn_id,
            replacement.state_version,
            status,
        )
    return replacement


def test_host_errors_expose_stable_codes() -> None:
    assert HostRuntimeError.code == "host_runtime_error"
    assert HostRecoveryRequired.code == "host_recovery_required"
    assert HostRecoveryBlocked.code == "host_recovery_blocked"


def test_host_requires_one_physical_active_turn_fence_repository() -> None:
    turn_repository = _repository(_schedule())
    dispatch_repository = _repository(_schedule(game_id="other-game"))
    reconciler = DispatchReconciler(dispatch_repository, PendingResolver())

    with pytest.raises(ActiveTurnFenceUnsupported):
        HostRuntime(
            turn_repository,
            dispatch_repository,
            reconciler,
            clock=lambda: NOW,
        )


def test_host_creates_repository_generated_fenced_attempt() -> None:
    repository, host, managed = _host_with_fenceable_active_turn()

    attempt = host.create_active_turn_dispatch(
        managed.schedule_id,
        _attempt(managed),
    )

    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None
    assert attempt.active_turn_fence is not None
    assert attempt.active_turn_fence.turn_state_version == current.state_version


def test_host_load_active_turn_returns_defensive_repository_copies() -> None:
    """Host 的活动回合读取不暴露仓储持有的可变对象引用。"""

    _repository, host, managed = _host_with_active_turn()

    first = host.load_active_turn(managed.turn.game_id)
    second = host.load_active_turn(managed.turn.game_id)

    assert first == second == managed
    assert first is not None
    assert second is not None
    assert first is not second
    assert first.turn is not second.turn


def test_fence_fixture_uses_repository_lifecycle_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_datetime = datetime

    class LifecycleDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return real_datetime(2040, 1, 1, tzinfo=tz)

    monkeypatch.setattr(memory_store, "datetime", LifecycleDateTime)
    monkeypatch.setattr(sys.modules[__name__], "datetime", LifecycleDateTime)

    _repository, host, managed = _host_with_fenceable_active_turn()

    attempt = host.create_active_turn_dispatch(
        managed.schedule_id,
        _attempt(managed),
    )

    assert attempt.active_turn_fence is not None


def test_active_turn_dispatch_requires_recovered_game() -> None:
    repository, _active_host, managed = _host_with_fenceable_active_turn()
    restarted = _host(repository, now=managed.updated_at)

    with pytest.raises(HostRecoveryRequired):
        restarted.create_active_turn_dispatch(
            managed.schedule_id,
            _attempt(managed),
        )


def test_active_turn_dispatch_requires_open_recovery_barrier() -> None:
    repository, host, managed = _host_with_fenceable_active_turn()
    pending = repository.create_dispatch(_attempt(managed))
    repository.mark_dispatching(pending.dispatch_id, pending.state_version)
    host.recover_game(managed.turn.game_id)

    with pytest.raises(HostRecoveryBlocked):
        host.create_active_turn_dispatch(
            managed.schedule_id,
            _attempt(managed, dispatch_id="blocked-dispatch"),
        )


def test_active_turn_dispatch_rejects_expired_deadline() -> None:
    repository, _active_host, managed = _host_with_fenceable_active_turn()
    expired_host = _host(repository, now=managed.turn.window.deadline)
    expired_host.recover_game(managed.turn.game_id)

    with pytest.raises(ActiveTurnFenceRejected):
        expired_host.create_active_turn_dispatch(
            managed.schedule_id,
            _attempt(managed),
        )


def test_active_turn_dispatch_rejects_caller_supplied_fence() -> None:
    _repository, host, managed = _host_with_fenceable_active_turn()
    caller_fence = ActiveTurnDispatchFence(
        schedule_id=managed.schedule_id,
        schedule_state_version=1,
        turn_state_version=1,
        window_id=managed.turn.window.window_id,
        window_version=managed.turn.window.version,
        base_game_revision=managed.turn.revision.base_revision,
    )
    attempt = _attempt(managed).model_copy(
        update={"active_turn_fence": caller_fence},
    )

    with pytest.raises(ActiveTurnFenceRejected):
        host.create_active_turn_dispatch(managed.schedule_id, attempt)


def test_active_turn_dispatch_keeps_captured_identity_across_replace_interleaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, host, managed = _host_with_fenceable_active_turn()
    original_require = host._require_active_schedule_turn

    def interleave(
        schedule_id: str,
    ) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
        captured = original_require(schedule_id)
        _replace_with_validating_turn(repository, *captured)
        return captured

    monkeypatch.setattr(host, "_require_active_schedule_turn", interleave)

    with pytest.raises(ScheduleStateConflict):
        host.create_active_turn_dispatch(
            managed.schedule_id,
            _attempt(managed),
        )


def test_scheduler_raises_stable_errors_for_missing_or_inactive_turns() -> None:
    repository = _repository(_schedule())
    scheduler = SerialPublicScheduler(repository)

    with pytest.raises(ScheduleNotFound):
        scheduler.require_schedule("missing")
    with pytest.raises(InvalidScheduleTransition):
        scheduler.load_active_turn("game-1")
    with pytest.raises(ManagedTurnNotFound):
        scheduler.transition_active_turn(
            "missing-turn",
            0,
            AgentTurnStatus.OBSERVING,
        )


def test_fresh_schedule_can_admit_current_slot_without_recovery() -> None:
    repository, host, schedule = _fresh_host_with_schedule()

    managed = host.admit_next_turn(schedule.schedule_id, _admission(schedule=schedule))

    assert managed.turn.player_id == "p01"
    assert repository.load_managed_turn(managed.turn.turn_id) == managed


def test_admission_rejects_expired_window_using_injected_clock() -> None:
    schedule = _schedule(deadline=NOW)
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id=schedule.game_id))
    host = _host(repository, now=NOW)
    host.create_schedule(schedule)

    with pytest.raises(InvalidTurnAdmission):
        host.admit_next_turn(schedule.schedule_id, _admission(schedule=schedule))


def test_restarted_host_requires_recovery_before_admission() -> None:
    schedule = _schedule()
    repository = _repository(schedule)
    restarted = _host(repository)

    with pytest.raises(HostRecoveryRequired):
        restarted.admit_next_turn(schedule.schedule_id, _admission(schedule=schedule))


def test_restarted_host_requires_recovery_before_cancel() -> None:
    repository, _active_host, managed = _host_with_active_turn()
    restarted = _host(repository)

    with pytest.raises(HostRecoveryRequired):
        restarted.cancel_active_turn(
            managed.schedule_id,
            "operator_cancelled",
            TerminalDisposition.ADVANCE,
        )

    assert repository.load_managed_turn(managed.turn.turn_id) == managed


def test_recovery_without_pending_dispatch_opens_barrier() -> None:
    schedule = _schedule()
    repository = _repository(schedule)
    restarted = _host(repository)

    report = restarted.recover_game(schedule.game_id)
    managed = restarted.admit_next_turn(
        schedule.schedule_id,
        _admission(schedule=schedule),
    )

    assert report.barrier_open is True
    assert managed.turn.status is AgentTurnStatus.OPEN


@pytest.mark.parametrize("resolver", [PendingResolver(), ErrorResolver()])
def test_pending_or_errored_recovery_keeps_host_blocked(resolver: object) -> None:
    repository, _active_host, managed = _host_with_active_turn()
    attempt = repository.create_dispatch(_attempt(managed))
    repository.mark_dispatching(attempt.dispatch_id, attempt.state_version)
    restarted = _host(repository, resolver=resolver)

    report = restarted.recover_game(managed.turn.game_id)

    assert report.barrier_open is False
    with pytest.raises(HostRecoveryBlocked):
        restarted.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.OBSERVING,
        )


def test_failed_explicit_recovery_revokes_existing_schedule_qualification() -> None:
    repository, host, managed = _host_with_active_turn()
    attempt = repository.create_dispatch(_attempt(managed))
    repository.mark_dispatching(attempt.dispatch_id, attempt.state_version)

    report = host.recover_game(managed.turn.game_id)

    assert report.barrier_open is False
    with pytest.raises(HostRecoveryBlocked):
        host.cancel_active_turn(
            managed.schedule_id,
            "operator_cancelled",
            TerminalDisposition.ADVANCE,
        )
    assert repository.load_managed_turn(managed.turn.turn_id) == managed


def test_unsafe_recovery_marks_unknown_and_opens_barrier() -> None:
    repository, _active_host, managed = _host_with_active_turn()
    attempt = repository.create_dispatch(_attempt(managed))
    repository.mark_dispatching(attempt.dispatch_id, attempt.state_version)
    restarted = _host(repository, resolver=UnsafeResolver())

    report = restarted.recover_game(managed.turn.game_id)
    observed = restarted.transition_active_turn(
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.OBSERVING,
    )

    stored_attempt = repository.load_dispatch(attempt.dispatch_id)
    assert report.barrier_open is True
    assert report.unknown == 1
    assert report.budget_consumption_required is True
    assert stored_attempt is not None
    assert stored_attempt.status is DispatchStatus.UNKNOWN_OUTCOME
    assert observed.turn.status is AgentTurnStatus.OBSERVING


def test_complete_active_turn_advances_after_validating() -> None:
    repository, host, managed = _host_with_active_turn()
    managed = _to_validating(host, managed)

    schedule = host.complete_active_turn("schedule-1")

    assert schedule.next_slot_ordinal == 1
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    assert stored.turn.status is AgentTurnStatus.COMMITTED
    assert stored.terminal_reason is None


def test_complete_active_turn_does_not_bypass_turn_lifecycle() -> None:
    repository, host, managed = _host_with_active_turn()

    with pytest.raises(InvalidScheduleTransition):
        host.complete_active_turn("schedule-1")

    assert repository.load_managed_turn(managed.turn.turn_id) == managed


def test_host_complete_uses_one_fenced_terminal_call_without_prescan() -> None:
    repository, host, managed = _host_with_fenceable_active_turn(
        repository_type=FenceSpyRepository,
    )
    _to_validating(host, managed)

    host.complete_active_turn(managed.schedule_id)

    assert repository.fenced_finish_calls == 1
    assert repository.dispatch_scan_calls == 0
    assert repository.unfenced_finish_calls == 0


def test_complete_keeps_captured_turn_identity_across_replace_interleaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, host, managed = _host_with_active_turn()
    managed = _to_validating(host, managed)
    captured_schedule = repository.load_serial_public_schedule(managed.schedule_id)
    assert captured_schedule is not None
    original_require = host._require_active_schedule_turn
    replacements: list[ManagedAgentTurn] = []

    def interleave(
        schedule_id: str,
    ) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
        captured = original_require(schedule_id)
        replacements.append(
            _replace_with_validating_turn(repository, *captured),
        )
        return captured

    monkeypatch.setattr(host, "_require_active_schedule_turn", interleave)

    with pytest.raises(ScheduleStateConflict):
        host.complete_active_turn(captured_schedule.schedule_id)

    replacement = repository.load_managed_turn(replacements[0].turn.turn_id)
    assert replacement is not None
    assert replacement.turn.status is AgentTurnStatus.VALIDATING
    current = repository.load_serial_public_schedule(captured_schedule.schedule_id)
    assert current is not None
    assert current.active_turn_id == replacement.turn.turn_id


@pytest.mark.parametrize(
    ("disposition", "expected_status", "expected_ordinal"),
    [
        (TerminalDisposition.ADVANCE, SerialPublicScheduleStatus.OPEN, 1),
        (TerminalDisposition.REPLACE, SerialPublicScheduleStatus.OPEN, 0),
        (TerminalDisposition.CLOSE, SerialPublicScheduleStatus.CANCELLED, 0),
    ],
)
def test_cancel_active_turn_cancels_unresolved_dispatches_and_applies_disposition(
    disposition: TerminalDisposition,
    expected_status: SerialPublicScheduleStatus,
    expected_ordinal: int,
) -> None:
    repository, host, managed = _host_with_fenceable_active_turn()
    pending = _reserve(host, managed, dispatch_id="pending")
    dispatching = _reserve(host, managed, dispatch_id="dispatching")
    repository.mark_dispatching(dispatching.dispatch_id, dispatching.state_version)

    schedule = host.cancel_active_turn(
        managed.schedule_id,
        "operator_cancelled",
        disposition,
    )

    assert schedule.status is expected_status
    assert schedule.next_slot_ordinal == expected_ordinal
    assert repository.load_dispatch(pending.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
    assert repository.load_dispatch(dispatching.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    assert stored.turn.status is AgentTurnStatus.CANCELLED
    assert stored.terminal_reason == "operator_cancelled"


def test_host_cancel_uses_one_fenced_terminal_call_without_prescan() -> None:
    repository, host, managed = _host_with_fenceable_active_turn(
        repository_type=FenceSpyRepository,
    )
    _reserve(host, managed)

    host.cancel_active_turn(
        managed.schedule_id,
        "operator_cancelled",
        TerminalDisposition.ADVANCE,
    )

    assert repository.fenced_finish_calls == 1
    assert repository.dispatch_scan_calls == 0
    assert repository.unfenced_finish_calls == 0


def test_dispatch_block_after_recovery_still_allows_active_turn_cancel() -> None:
    repository, _active_host, managed = _host_with_fenceable_active_turn()
    restarted = _host(repository, now=managed.updated_at)
    report = restarted.recover_game(managed.turn.game_id)
    attempt = _reserve(restarted, managed)
    attempt = repository.mark_dispatching(
        attempt.dispatch_id,
        attempt.state_version,
    )

    with pytest.raises(HostRecoveryBlocked):
        restarted.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.OBSERVING,
        )

    schedule = restarted.cancel_active_turn(
        managed.schedule_id,
        "operator_cancelled",
        TerminalDisposition.ADVANCE,
    )

    assert report.barrier_open is True
    assert schedule.next_slot_ordinal == 1
    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    assert stored.turn.status is AgentTurnStatus.CANCELLED
    assert stored.terminal_reason == "operator_cancelled"


def test_cancel_keeps_captured_turn_identity_across_replace_interleaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, host, managed = _host_with_active_turn()
    managed = _to_validating(host, managed)
    original_require = host._require_active_schedule_turn
    replacements: list[ManagedAgentTurn] = []

    def interleave(
        schedule_id: str,
    ) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
        captured = original_require(schedule_id)
        replacements.append(
            _replace_with_validating_turn(repository, *captured),
        )
        return captured

    monkeypatch.setattr(host, "_require_active_schedule_turn", interleave)

    with pytest.raises(ScheduleStateConflict):
        host.cancel_active_turn(
            managed.schedule_id,
            "operator_cancelled",
            TerminalDisposition.ADVANCE,
        )

    replacement = repository.load_managed_turn(replacements[0].turn.turn_id)
    assert replacement is not None
    assert replacement.turn.status is AgentTurnStatus.VALIDATING
    current = repository.load_serial_public_schedule(managed.schedule_id)
    assert current is not None
    assert current.active_turn_id == replacement.turn.turn_id


def test_dispatched_attempt_survives_cancel_and_blocks_replacement_until_recovery() -> None:
    repository, host, managed = _host_with_fenceable_active_turn()
    attempt = _reserve(host, managed)
    attempt = repository.mark_dispatching(attempt.dispatch_id, attempt.state_version)
    attempt = repository.mark_dispatched(attempt.dispatch_id, attempt.state_version)
    replaced = host.cancel_active_turn(
        managed.schedule_id,
        "operator_cancelled",
        TerminalDisposition.REPLACE,
    )

    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.DISPATCHED  # type: ignore[union-attr]
    with pytest.raises(HostRecoveryBlocked):
        host.admit_next_turn(
            replaced.schedule_id,
            _admission(
                schedule=replaced,
                turn_id="turn-2",
                idempotency_key="turn-2:submit",
            ),
        )
    with pytest.raises(HostRecoveryBlocked):
        host.admit_next_turn(
            replaced.schedule_id,
            _admission(
                schedule=replaced,
                turn_id="turn-2",
                idempotency_key="turn-2:submit",
            ),
        )

    restarted = _host(repository, resolver=UnsafeResolver())
    report = restarted.recover_game(replaced.game_id)
    replacement = restarted.admit_next_turn(
        replaced.schedule_id,
        _admission(
            schedule=replaced,
            turn_id="turn-2",
            idempotency_key="turn-2:submit",
        ),
    )

    assert report.barrier_open is True
    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.UNKNOWN_OUTCOME  # type: ignore[union-attr]
    assert replacement.turn.turn_id != managed.turn.turn_id
    assert replacement.turn.idempotency_key != managed.turn.idempotency_key


def test_expire_due_turns_is_aware_recovered_and_deterministic() -> None:
    opened_at = datetime.now(timezone.utc)
    schedule_b = _schedule(
        game_id="game-b",
        schedule_id="schedule-b",
        deadline=opened_at + timedelta(hours=1),
    )
    schedule_a = _schedule(
        game_id="game-a",
        schedule_id="schedule-a",
        deadline=opened_at + timedelta(hours=1),
    )
    schedule_future = _schedule(
        game_id="game-c",
        schedule_id="schedule-c",
        deadline=opened_at + timedelta(hours=3),
    )
    repository = InMemoryGameRepository()
    admission_host = _host(repository, now=opened_at)
    managed_by_schedule: dict[str, ManagedAgentTurn] = {}
    for schedule in (schedule_b, schedule_a, schedule_future):
        repository.save_game(GameState(game_id=schedule.game_id))
        admission_host.create_schedule(schedule)
        managed_by_schedule[schedule.schedule_id] = admission_host.admit_next_turn(
            schedule.schedule_id,
            _admission(
                schedule=schedule,
                turn_id=f"turn-{schedule.game_id}",
            ),
        )
    host = _host(
        repository,
        now=max(item.updated_at for item in managed_by_schedule.values()),
    )
    for schedule in (schedule_a, schedule_b, schedule_future):
        host.recover_game(schedule.game_id)
    pending = _reserve(
        host,
        managed_by_schedule["schedule-a"],
        dispatch_id="expire-pending",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        host.expire_due_turns(datetime(2026, 7, 30, 10))  # noqa: DTZ001

    changed = host.expire_due_turns(schedule_a.window.deadline)

    assert tuple(schedule.schedule_id for schedule in changed) == (
        "schedule-a",
        "schedule-b",
    )
    assert repository.load_dispatch(pending.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
    for schedule_id in ("schedule-a", "schedule-b"):
        managed = repository.load_managed_turn(
            managed_by_schedule[schedule_id].turn.turn_id,
        )
        assert managed is not None
        assert managed.turn.status is AgentTurnStatus.EXPIRED
        assert managed.terminal_reason == "deadline_expired"
    future = repository.load_managed_turn(
        managed_by_schedule["schedule-c"].turn.turn_id,
    )
    assert future is not None
    assert future.turn.status is AgentTurnStatus.OPEN


def test_host_expiry_uses_one_fenced_terminal_call_without_prescan() -> None:
    repository, host, managed = _host_with_fenceable_active_turn(
        repository_type=FenceSpyRepository,
    )
    _reserve(host, managed)

    changed = host.expire_due_turns(managed.turn.window.deadline)

    assert tuple(schedule.schedule_id for schedule in changed) == (
        managed.schedule_id,
    )
    assert repository.fenced_finish_calls == 1
    assert repository.dispatch_scan_calls == 0
    assert repository.unfenced_finish_calls == 0


def test_dispatch_block_after_schedule_creation_still_allows_due_turn_expiry() -> None:
    repository, host, managed = _host_with_fenceable_active_turn()
    attempt = _reserve(host, managed)
    attempt = repository.mark_dispatching(
        attempt.dispatch_id,
        attempt.state_version,
    )

    with pytest.raises(HostRecoveryBlocked):
        host.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.OBSERVING,
        )

    changed = host.expire_due_turns(managed.turn.window.deadline)

    assert tuple(item.schedule_id for item in changed) == (managed.schedule_id,)
    assert repository.load_dispatch(attempt.dispatch_id).status is DispatchStatus.CANCELLED  # type: ignore[union-attr]
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    assert stored.turn.status is AgentTurnStatus.EXPIRED
    assert stored.terminal_reason == "deadline_expired"


def test_expire_due_turns_skips_unrecovered_persisted_game() -> None:
    schedule = _schedule(deadline=NOW)
    repository = _repository(schedule)
    managed = repository.admit_serial_public_turn(
        schedule.schedule_id,
        schedule.state_version,
        _admission(schedule=schedule),
    )
    restarted = _host(repository, now=NOW)

    assert restarted.expire_due_turns(NOW) == ()
    assert repository.load_managed_turn(managed.turn.turn_id) == managed
