# -*- coding: utf-8 -*-
"""
验证串行公开调度门面与主机托管回合的恢复和生命周期语义。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from werewolf_agent.core.models import GameState
from werewolf_agent.player_agents.contracts.dispatch import (
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
from werewolf_agent.storage.autonomous_turns import (
    InvalidScheduleTransition,
    InvalidTurnAdmission,
    ManagedTurnNotFound,
    ScheduleNotFound,
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


def test_host_errors_expose_stable_codes() -> None:
    assert HostRuntimeError.code == "host_runtime_error"
    assert HostRecoveryRequired.code == "host_recovery_required"
    assert HostRecoveryBlocked.code == "host_recovery_blocked"


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
    repository, host, managed = _host_with_active_turn()
    pending = repository.create_dispatch(_attempt(managed, dispatch_id="pending"))
    dispatching = repository.create_dispatch(_attempt(managed, dispatch_id="dispatching"))
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


def test_dispatched_attempt_survives_cancel_and_blocks_replacement_until_recovery() -> None:
    repository, host, managed = _host_with_active_turn()
    attempt = repository.create_dispatch(_attempt(managed))
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
    schedule_b = _schedule(
        game_id="game-b",
        schedule_id="schedule-b",
        deadline=NOW,
    )
    schedule_a = _schedule(
        game_id="game-a",
        schedule_id="schedule-a",
        deadline=NOW,
    )
    schedule_future = _schedule(
        game_id="game-c",
        schedule_id="schedule-c",
        deadline=NOW + timedelta(hours=2),
    )
    repository = InMemoryGameRepository()
    host = _host(repository, now=NOW - timedelta(minutes=1))
    managed_by_schedule: dict[str, ManagedAgentTurn] = {}
    for schedule in (schedule_b, schedule_a, schedule_future):
        repository.save_game(GameState(game_id=schedule.game_id))
        host.create_schedule(schedule)
        managed_by_schedule[schedule.schedule_id] = host.admit_next_turn(
            schedule.schedule_id,
            _admission(
                schedule=schedule,
                turn_id=f"turn-{schedule.game_id}",
            ),
        )
    pending = repository.create_dispatch(
        _attempt(managed_by_schedule["schedule-a"], dispatch_id="expire-pending"),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        host.expire_due_turns(datetime(2026, 7, 30, 10))  # noqa: DTZ001

    changed = host.expire_due_turns(NOW)

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
