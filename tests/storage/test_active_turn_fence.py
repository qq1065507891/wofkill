# -*- coding: utf-8 -*-
"""
验证活动回合 dispatch 围栏的纯准备逻辑与能力约束。

作者: Project contributors
创建日期: 2026-07-31
"""

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
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
from werewolf_agent.player_agents.contracts.revisions import RevisionContext
from werewolf_agent.player_agents.contracts.scheduling import (
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    SerialPublicSlot,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)
from werewolf_agent.storage.active_turn_fence import (
    ActiveTurnFenceError,
    ActiveTurnFenceRejected,
    ActiveTurnFenceTransactionError,
    ActiveTurnFenceUnsupported,
    prepare_active_turn_dispatch,
    prepare_fenced_active_finish,
    require_active_turn_fence_repository,
)
from werewolf_agent.storage.autonomous_turns import (
    InvalidScheduleTransition,
    ScheduleStateConflict,
    TurnStateConflict,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchIdempotencyConflict,
    DispatchInvalidTransition,
    DispatchRecoveryBlocked,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository

HASH = "a" * 64
NOW = datetime(2026, 7, 31, 10, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 7, 31, 11, tzinfo=timezone.utc)


class _PrivateDatetime(datetime):
    """在时间戳校验期间模拟不能泄漏的内部异常。"""

    def utcoffset(self) -> timedelta | None:
        raise ValueError("private payload")


def _formatted_traceback(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _window(**updates: object) -> LegalActionWindow:
    payload: dict[str, object] = {
        "window_id": "speech-d1",
        "version": 1,
        "game_id": "game-1",
        "task_type": "day_speech",
        "conflict_class": ConflictClass.SERIAL_PUBLIC,
        "participant_ids": ("p01", "p02"),
        "legal_actions": ("speech",),
        "opened_revision": 4,
        "deadline": DEADLINE,
    }
    payload.update(updates)
    return LegalActionWindow.model_validate(payload)


def _active_turn(
    *,
    status: AgentTurnStatus = AgentTurnStatus.THINKING,
    turn_version: int = 3,
    schedule_status: SerialPublicScheduleStatus = SerialPublicScheduleStatus.OPEN,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    window = _window()
    turn = AgentTurn(
        turn_id="turn-1",
        game_id="game-1",
        player_id="p01",
        role_id="villager",
        phase="day_discussion",
        task_type="day_speech",
        revision=RevisionContext(
            base_revision=4,
            window_id=window.window_id,
            window_version=window.version,
            view_fingerprint=HASH,
        ),
        window=window,
        model_lease_hash=HASH,
        budget=TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        status=status,
        idempotency_key="turn-1:submit",
    )
    schedule = SerialPublicSchedule(
        schedule_id="schedule-1",
        game_id="game-1",
        window=window,
        slots=(
            SerialPublicSlot(ordinal=0, player_id="p01"),
            SerialPublicSlot(ordinal=1, player_id="p02"),
        ),
        next_slot_ordinal=0,
        active_turn_id="turn-1" if schedule_status is SerialPublicScheduleStatus.OPEN else None,
        status=schedule_status,
        state_version=1,
        created_at=NOW,
        updated_at=NOW,
    )
    return schedule, ManagedAgentTurn(
        schedule_id=schedule.schedule_id,
        turn=turn,
        state_version=turn_version,
        created_at=NOW,
        updated_at=NOW,
    )


def _attempt_for(managed: ManagedAgentTurn, **updates: object) -> DispatchAttempt:
    payload: dict[str, object] = {
        "dispatch_id": "dispatch-1",
        "game_id": managed.turn.game_id,
        "turn_id": managed.turn.turn_id,
        "actor_id": managed.turn.player_id,
        "operation_kind": DispatchOperationKind.MODEL,
        "executor_id": "mock-provider",
        "provider_idempotency_key": "provider-key-1",
        "recovery_policy": DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        "request_hash": HASH,
        "lease_hash": managed.turn.model_lease_hash,
        "view_fingerprint": managed.turn.revision.view_fingerprint,
        "deadline": DEADLINE,
        "created_at": NOW,
        "updated_at": NOW,
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    payload.update(updates)
    return DispatchAttempt.model_validate(payload)


def _memory_active_turn(
    *,
    status: AgentTurnStatus = AgentTurnStatus.THINKING,
) -> tuple[InMemoryGameRepository, SerialPublicSchedule, ManagedAgentTurn]:
    """通过公开的调度 API 创建一个持久化活动回合。"""

    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="game-1"))
    schedule, managed = _active_turn()
    created = repository.create_serial_public_schedule(
        schedule.model_copy(update={"active_turn_id": None, "state_version": 0}),
    )
    admitted = repository.admit_serial_public_turn(
        created.schedule_id,
        created.state_version,
        TurnAdmission(
            turn_id=managed.turn.turn_id,
            player_id=managed.turn.player_id,
            role_id=managed.turn.role_id,
            phase=managed.turn.phase,
            revision=managed.turn.revision,
            read_set=(),
            model_lease_hash=managed.turn.model_lease_hash,
            budget=managed.turn.budget,
            idempotency_key=managed.turn.idempotency_key,
        ),
    )
    for next_status in (
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.THINKING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.VALIDATING,
    ):
        if admitted.turn.status is status:
            break
        admitted = repository.transition_active_turn(
            admitted.turn.turn_id,
            admitted.state_version,
            next_status,
        )
    return repository, repository.load_serial_public_schedule(created.schedule_id), admitted  # type: ignore[return-value]


def _reservation_state(
    repository: InMemoryGameRepository,
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
) -> tuple[ManagedAgentTurn | None, dict[str, DispatchAttempt], dict[tuple[str, str], str]]:
    """记录预约失败前后必须完全一致的内存持久化状态。"""

    return (
        repository.load_managed_turn(managed.turn.turn_id),
        dict(repository._dispatch_attempts),
        dict(repository._dispatch_key_index),
    )


def test_memory_fenced_create_persists_attempt_and_turn_version() -> None:
    repository, schedule, managed = _memory_active_turn()

    stored = repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _attempt_for(managed),
        NOW,
    )

    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None
    assert current.state_version == managed.state_version + 1
    assert stored.active_turn_fence is not None
    assert stored.active_turn_fence.turn_state_version == current.state_version
    assert repository.load_dispatch(stored.dispatch_id) == stored


def test_memory_plain_create_rejects_caller_supplied_fence() -> None:
    repository, schedule, managed = _memory_active_turn()
    fence = ActiveTurnDispatchFence(
        schedule_id=schedule.schedule_id,
        schedule_state_version=schedule.state_version,
        turn_state_version=managed.state_version,
        window_id=schedule.window.window_id,
        window_version=schedule.window.version,
        base_game_revision=managed.turn.revision.base_revision,
    )

    with pytest.raises(DispatchInvalidTransition):
        repository.create_dispatch(_attempt_for(managed, active_turn_fence=fence))


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("schedule_cas", ScheduleStateConflict),
        ("turn_cas", TurnStateConflict),
        ("recovery", DispatchRecoveryBlocked),
        ("dispatch_id", DispatchIdempotencyConflict),
        ("provider_key", DispatchIdempotencyConflict),
        ("context", ActiveTurnFenceRejected),
        ("expired", ActiveTurnFenceRejected),
    ],
)
def test_memory_fenced_create_failure_leaves_every_index_unchanged(
    case: str,
    expected_error: type[Exception],
) -> None:
    repository, schedule, managed = _memory_active_turn()
    attempt = _attempt_for(managed)
    expected_schedule_version = schedule.state_version
    expected_turn_version = managed.state_version
    observed_at = NOW

    if case == "schedule_cas":
        expected_schedule_version -= 1
    elif case == "turn_cas":
        expected_turn_version -= 1
    elif case == "recovery":
        blocker = _attempt_for(
            managed,
            dispatch_id="blocking",
            provider_idempotency_key="blocking-key",
        )
        repository.create_dispatch(blocker)
        repository.mark_dispatching(blocker.dispatch_id, blocker.state_version)
    elif case == "dispatch_id":
        repository.create_dispatch(attempt)
    elif case == "provider_key":
        repository.create_dispatch(
            _attempt_for(managed, dispatch_id="other-dispatch"),
        )
    elif case == "context":
        attempt = _attempt_for(managed, actor_id="p02")
    elif case == "expired":
        observed_at = DEADLINE

    before = _reservation_state(repository, schedule, managed)
    with pytest.raises(expected_error):
        repository.create_active_turn_dispatch(
            schedule.schedule_id,
            expected_schedule_version,
            managed.turn.turn_id,
            expected_turn_version,
            attempt,
            observed_at,
        )
    assert _reservation_state(repository, schedule, managed) == before


def test_memory_fenced_create_restores_all_records_after_publish_failure() -> None:
    class FailingDispatchDict(dict[str, DispatchAttempt]):
        def __setitem__(self, key: str, value: DispatchAttempt) -> None:
            raise OSError("injected dispatch write failure")

    repository, schedule, managed = _memory_active_turn()
    before = _reservation_state(repository, schedule, managed)
    repository._dispatch_attempts = FailingDispatchDict(repository._dispatch_attempts)

    with pytest.raises(ActiveTurnFenceTransactionError):
        _reserve(repository, schedule, managed)

    assert _reservation_state(repository, schedule, managed) == before


def _reserve(
    repository: InMemoryGameRepository,
    schedule: SerialPublicSchedule,
    managed: ManagedAgentTurn,
    *,
    dispatch_id: str = "dispatch-1",
) -> DispatchAttempt:
    """用当前 CAS 身份预约一个活动回合 dispatch。"""

    return repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _attempt_for(managed, dispatch_id=dispatch_id),
        NOW,
    )


def test_memory_fenced_cancel_rolls_attempts_and_turn_forward_together() -> None:
    repository, schedule, managed = _memory_active_turn()
    pending = _reserve(repository, schedule, managed, dispatch_id="pending")
    managed = repository.load_managed_turn(managed.turn.turn_id)
    assert managed is not None

    updated = repository.finish_active_turn_fenced(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.REPLACE,
        "operator_cancelled",
    )

    assert updated.active_turn_id is None
    stored = repository.load_dispatch(pending.dispatch_id)
    assert stored is not None
    assert stored.status is DispatchStatus.CANCELLED
    finished = repository.load_managed_turn(managed.turn.turn_id)
    assert finished is not None
    assert finished.turn.status is AgentTurnStatus.CANCELLED


def _race(*operations):
    """让两个操作从同一同步点竞争同一份已捕获的 CAS 身份。"""

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(operation) for operation in operations]
        results = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - 断言公开竞争结果。
                results.append(type(exc))
            else:
                results.append(None)
    return results


def test_memory_create_and_complete_have_one_cas_winner() -> None:
    repository, schedule, managed = _memory_active_turn(
        status=AgentTurnStatus.VALIDATING,
    )
    barrier = threading.Barrier(2)

    results = _race(
        lambda: (
            barrier.wait(),
            _reserve(repository, schedule, managed),
        ),
        lambda: (
            barrier.wait(),
            repository.finish_active_turn_fenced(
                schedule.schedule_id,
                schedule.state_version,
                managed.turn.turn_id,
                managed.state_version,
                AgentTurnStatus.COMMITTED,
                TerminalDisposition.ADVANCE,
                None,
            ),
        ),
    )

    assert results.count(None) == 1
    assert {result for result in results if result is not None} <= {
        ScheduleStateConflict,
        TurnStateConflict,
    }


@pytest.mark.parametrize(
    ("terminal_status", "reason_code"),
    [
        (AgentTurnStatus.CANCELLED, "operator_cancelled"),
        (AgentTurnStatus.EXPIRED, "window_expired"),
    ],
)
def test_memory_create_and_terminal_retry_leave_no_executable_attempt(
    terminal_status: AgentTurnStatus,
    reason_code: str,
) -> None:
    repository, schedule, managed = _memory_active_turn()
    barrier = threading.Barrier(2)

    results = _race(
        lambda: (
            barrier.wait(),
            _reserve(repository, schedule, managed),
        ),
        lambda: (
            barrier.wait(),
            repository.finish_active_turn_fenced(
                schedule.schedule_id,
                schedule.state_version,
                managed.turn.turn_id,
                managed.state_version,
                terminal_status,
                TerminalDisposition.REPLACE,
                reason_code,
            ),
        ),
    )

    assert results.count(None) == 1
    assert {result for result in results if result is not None} <= {
        ScheduleStateConflict,
        TurnStateConflict,
    }
    current_schedule = repository.load_serial_public_schedule(schedule.schedule_id)
    current_turn = repository.load_managed_turn(managed.turn.turn_id)
    assert current_schedule is not None
    assert current_turn is not None
    if current_schedule.active_turn_id == current_turn.turn.turn_id:
        repository.finish_active_turn_fenced(
            current_schedule.schedule_id,
            current_schedule.state_version,
            current_turn.turn.turn_id,
            current_turn.state_version,
            terminal_status,
            TerminalDisposition.REPLACE,
            reason_code,
        )
    assert all(
        attempt.status not in {DispatchStatus.PENDING, DispatchStatus.DISPATCHING}
        for attempt in repository.list_dispatches_for_turn(
            managed.turn.game_id,
            managed.turn.turn_id,
        )
    )


def test_memory_create_and_nonterminal_transition_have_one_cas_winner() -> None:
    repository, schedule, managed = _memory_active_turn()
    barrier = threading.Barrier(2)

    results = _race(
        lambda: (
            barrier.wait(),
            _reserve(repository, schedule, managed),
        ),
        lambda: (
            barrier.wait(),
            repository.transition_active_turn(
                managed.turn.turn_id,
                managed.state_version,
                AgentTurnStatus.SUBMITTED,
            ),
        ),
    )

    assert results.count(None) == 1
    assert {result for result in results if result is not None} == {
        TurnStateConflict,
    }


@pytest.mark.parametrize(
    "publication_error",
    [
        ActiveTurnFenceRejected("injected active fence rejection"),
        DispatchRecoveryBlocked("injected recovery block"),
        InvalidScheduleTransition("injected schedule transition"),
    ],
    ids=("active_fence", "recovery", "schedule_transition"),
)
def test_memory_fenced_finish_restores_and_sanitizes_publish_errors(
    publication_error: Exception,
) -> None:
    class FailingScheduleDict(dict[str, SerialPublicSchedule]):
        def __setitem__(self, key: str, value: SerialPublicSchedule) -> None:
            raise publication_error

    repository, schedule, managed = _memory_active_turn()
    pending = _reserve(repository, schedule, managed)
    managed = repository.load_managed_turn(managed.turn.turn_id)
    assert managed is not None
    before = (
        dict(repository._dispatch_attempts),
        dict(repository._managed_agent_turns),
        dict(repository._serial_public_schedules),
        dict(repository._active_schedule_by_game),
    )
    repository._serial_public_schedules = FailingScheduleDict(
        repository._serial_public_schedules,
    )

    with pytest.raises(ActiveTurnFenceTransactionError) as exc_info:
        repository.finish_active_turn_fenced(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.CANCELLED,
            TerminalDisposition.REPLACE,
            "operator_cancelled",
        )

    assert exc_info.value.__cause__ is None
    assert "injected" not in _formatted_traceback(exc_info.value)
    assert (
        dict(repository._dispatch_attempts),
        dict(repository._managed_agent_turns),
        dict(repository._serial_public_schedules),
        dict(repository._active_schedule_by_game),
    ) == before
    assert repository.load_dispatch(pending.dispatch_id) == pending


def test_prepare_active_turn_dispatch_reserves_turn_and_builds_fence() -> None:
    schedule, managed = _active_turn(turn_version=3)
    updated, attempt = prepare_active_turn_dispatch(
        schedule,
        managed,
        _attempt_for(managed),
        NOW,
    )
    assert updated.state_version == 4
    assert updated.turn == managed.turn
    assert updated.updated_at == NOW
    assert attempt.active_turn_fence == ActiveTurnDispatchFence(
        schedule_id=schedule.schedule_id,
        schedule_state_version=schedule.state_version,
        turn_state_version=4,
        window_id=managed.turn.window.window_id,
        window_version=managed.turn.window.version,
        base_game_revision=managed.turn.revision.base_revision,
    )


@pytest.mark.parametrize(
    "attempt_update",
    [
        {"game_id": "other-game"},
        {"turn_id": "other-turn"},
        {"actor_id": "p02"},
        {"lease_hash": "b" * 64},
        {"view_fingerprint": "b" * 64},
        {"deadline": DEADLINE + timedelta(seconds=1)},
    ],
)
def test_prepare_active_turn_dispatch_rejects_context_drift(
    attempt_update: dict[str, object],
) -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, **attempt_update),
            NOW,
        )


@pytest.mark.parametrize(
    "schedule_status, managed_status",
    [
        (SerialPublicScheduleStatus.CLOSED, AgentTurnStatus.THINKING),
        (SerialPublicScheduleStatus.CANCELLED, AgentTurnStatus.THINKING),
        (SerialPublicScheduleStatus.OPEN, AgentTurnStatus.COMMITTED),
        (SerialPublicScheduleStatus.OPEN, AgentTurnStatus.CANCELLED),
        (SerialPublicScheduleStatus.OPEN, AgentTurnStatus.EXPIRED),
    ],
)
def test_prepare_active_turn_dispatch_rejects_inactive_or_terminal_turn(
    schedule_status: SerialPublicScheduleStatus,
    managed_status: AgentTurnStatus,
) -> None:
    schedule, managed = _active_turn(
        schedule_status=schedule_status,
        status=managed_status,
    )
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(schedule, managed, _attempt_for(managed), NOW)


@pytest.mark.parametrize(
    "attempt_update",
    [
        {"status": DispatchStatus.DISPATCHING},
        {"state_version": 1},
        {"reason_code": "already_processed"},
    ],
)
def test_prepare_active_turn_dispatch_requires_new_pending_attempt(
    attempt_update: dict[str, object],
) -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, **attempt_update),
            NOW,
        )


def test_prepare_active_turn_dispatch_rejects_caller_supplied_fence() -> None:
    schedule, managed = _active_turn()
    fence = ActiveTurnDispatchFence(
        schedule_id=schedule.schedule_id,
        schedule_state_version=schedule.state_version,
        turn_state_version=managed.state_version,
        window_id=managed.turn.window.window_id,
        window_version=managed.turn.window.version,
        base_game_revision=managed.turn.revision.base_revision,
    )
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, active_turn_fence=fence),
            NOW,
        )


@pytest.mark.parametrize(
    "observed_at",
    [
        datetime(2026, 7, 31, 10),  # noqa: DTZ001 - 验证拒绝无时区输入。
        DEADLINE,
    ],
)
def test_prepare_active_turn_dispatch_rejects_invalid_observation_time(
    observed_at: datetime,
) -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed),
            observed_at,
        )


def test_prepare_active_turn_dispatch_rejects_inactive_schedule() -> None:
    schedule, managed = _active_turn()
    inactive = schedule.model_copy(update={"active_turn_id": None})
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(inactive, managed, _attempt_for(managed), NOW)


def test_prepare_active_turn_dispatch_rejects_window_deadline_at_observation() -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, deadline=DEADLINE - timedelta(seconds=1)),
            DEADLINE,
        )


def test_prepare_active_turn_dispatch_rejects_divergent_window_deadline() -> None:
    schedule, managed = _active_turn()
    schedule = schedule.model_copy(
        update={
            "window": schedule.window.model_copy(
                update={"deadline": DEADLINE + timedelta(hours=1)},
            ),
        },
    )
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed, deadline=DEADLINE + timedelta(minutes=30)),
            NOW,
        )


def test_prepare_active_turn_dispatch_sanitizes_private_time_failure() -> None:
    schedule, managed = _active_turn()
    private_now = _PrivateDatetime(2026, 7, 31, 10, tzinfo=timezone.utc)
    with pytest.raises(ActiveTurnFenceRejected) as exc_info:
        prepare_active_turn_dispatch(
            schedule,
            managed,
            _attempt_for(managed),
            private_now,
        )
    assert "private payload" not in _formatted_traceback(exc_info.value)


def test_prepare_fenced_cancel_cancels_only_cancellable_attempts() -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.THINKING)
    attempts = (
        _attempt_for(managed, dispatch_id="pending"),
        _attempt_for(
            managed,
            dispatch_id="dispatching",
            status=DispatchStatus.DISPATCHING,
            state_version=1,
        ),
        _attempt_for(
            managed,
            dispatch_id="dispatched",
            status=DispatchStatus.DISPATCHED,
            state_version=2,
        ),
    )
    updated_schedule, updated_turn, updated_attempts = prepare_fenced_active_finish(
        schedule,
        managed,
        attempts,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.REPLACE,
        reason_code="operator_cancelled",
        now=NOW,
    )
    assert updated_schedule.active_turn_id is None
    assert updated_turn.turn.status is AgentTurnStatus.CANCELLED
    assert tuple(item.status for item in updated_attempts) == (
        DispatchStatus.CANCELLED,
        DispatchStatus.CANCELLED,
        DispatchStatus.DISPATCHED,
    )
    assert tuple(item.state_version for item in updated_attempts) == (1, 2, 2)
    assert tuple(item.reason_code for item in updated_attempts) == (
        "operator_cancelled",
        "operator_cancelled",
        None,
    )


@pytest.mark.parametrize(
    "terminal_status",
    [AgentTurnStatus.CANCELLED, AgentTurnStatus.EXPIRED],
)
@pytest.mark.parametrize(
    "dispatch_status",
    [
        DispatchStatus.DISPATCHED,
        DispatchStatus.RESULT_RECORDED,
        DispatchStatus.UNKNOWN_OUTCOME,
        DispatchStatus.CANCELLED,
    ],
)
def test_prepare_fenced_terminal_keeps_noncancellable_attempt_immutable(
    terminal_status: AgentTurnStatus,
    dispatch_status: DispatchStatus,
) -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.THINKING)
    attempt = _attempt_for(
        managed,
        status=dispatch_status,
        state_version=2,
    )
    _, _, updated_attempts = prepare_fenced_active_finish(
        schedule,
        managed,
        (attempt,),
        terminal_status,
        TerminalDisposition.REPLACE,
        reason_code="operator_cancelled",
        now=NOW,
    )
    assert updated_attempts == (attempt,)


@pytest.mark.parametrize(
    "status",
    [DispatchStatus.PENDING, DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED],
)
def test_prepare_fenced_commit_rejects_unresolved_attempt(
    status: DispatchStatus,
) -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.VALIDATING)
    with pytest.raises(DispatchRecoveryBlocked):
        prepare_fenced_active_finish(
            schedule,
            managed,
            (_attempt_for(managed, status=status),),
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            reason_code=None,
            now=NOW,
        )


@pytest.mark.parametrize(
    "status",
    [
        DispatchStatus.RESULT_RECORDED,
        DispatchStatus.UNKNOWN_OUTCOME,
        DispatchStatus.CANCELLED,
    ],
)
def test_prepare_fenced_commit_accepts_resolved_attempt(status: DispatchStatus) -> None:
    schedule, managed = _active_turn(status=AgentTurnStatus.VALIDATING)
    updated_schedule, updated_turn, updated_attempts = prepare_fenced_active_finish(
        schedule,
        managed,
        (_attempt_for(managed, status=status),),
        AgentTurnStatus.COMMITTED,
        TerminalDisposition.ADVANCE,
        reason_code=None,
        now=NOW,
    )
    assert updated_schedule.active_turn_id is None
    assert updated_turn.turn.status is AgentTurnStatus.COMMITTED
    assert updated_attempts[0].status is status


def test_prepare_fenced_finish_rejects_attempt_for_another_turn() -> None:
    schedule, managed = _active_turn()
    with pytest.raises(ActiveTurnFenceRejected):
        prepare_fenced_active_finish(
            schedule,
            managed,
            (_attempt_for(managed, turn_id="other-turn"),),
            AgentTurnStatus.CANCELLED,
            TerminalDisposition.REPLACE,
            reason_code="operator_cancelled",
            now=NOW,
        )


def test_prepare_fenced_finish_sanitizes_private_time_failure() -> None:
    schedule, managed = _active_turn()
    private_now = _PrivateDatetime(2026, 7, 31, 10, tzinfo=timezone.utc)
    with pytest.raises(ActiveTurnFenceRejected) as exc_info:
        prepare_fenced_active_finish(
            schedule,
            managed,
            (_attempt_for(managed),),
            AgentTurnStatus.CANCELLED,
            TerminalDisposition.REPLACE,
            reason_code="operator_cancelled",
            now=private_now,
        )
    assert "private payload" not in _formatted_traceback(exc_info.value)


def test_fence_capability_guard_requires_one_explicit_repository() -> None:
    class CompleteRepository:
        def supports_autonomous_turns(self) -> bool:
            return True

        def supports_durable_dispatch(self) -> bool:
            return True

        def supports_active_turn_fence(self) -> bool:
            return True

    repository = CompleteRepository()
    assert require_active_turn_fence_repository(repository, repository) is repository
    with pytest.raises(ActiveTurnFenceUnsupported):
        require_active_turn_fence_repository(repository, CompleteRepository())


@pytest.mark.parametrize(
    ("capability_name", "failure_mode"),
    [
        ("supports_autonomous_turns", "missing"),
        ("supports_durable_dispatch", "false"),
        ("supports_active_turn_fence", "raising"),
    ],
)
def test_fence_capability_guard_rejects_missing_false_or_raising_capability(
    capability_name: str,
    failure_mode: str,
) -> None:
    class Repository:
        def __getattr__(self, name: str):
            if name != capability_name:
                return lambda: True
            if failure_mode == "missing":
                raise AttributeError(name)
            if failure_mode == "false":
                return lambda: False

            def _raising() -> bool:
                raise RuntimeError("private payload")

            return _raising

    repository = Repository()
    with pytest.raises(ActiveTurnFenceUnsupported) as exc_info:
        require_active_turn_fence_repository(repository, repository)
    assert "private payload" not in _formatted_traceback(exc_info.value)


def test_fence_errors_expose_stable_codes() -> None:
    assert ActiveTurnFenceError.code == "active_turn_fence_error"
    assert ActiveTurnFenceUnsupported.code == "active_turn_fence_unsupported"
    assert ActiveTurnFenceRejected.code == "active_turn_fence_rejected"
    assert (
        ActiveTurnFenceTransactionError.code
        == "active_turn_fence_transaction_error"
    )
