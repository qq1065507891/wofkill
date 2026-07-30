# -*- coding: utf-8 -*-
"""
验证活动回合 dispatch 围栏的纯准备逻辑与能力约束。

作者: Project contributors
创建日期: 2026-07-31
"""

from datetime import datetime, timedelta, timezone

import pytest

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
from werewolf_agent.storage.durable_dispatch import DispatchRecoveryBlocked

HASH = "a" * 64
NOW = datetime(2026, 7, 31, 10, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 7, 31, 11, tzinfo=timezone.utc)


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
    "capability_name",
    [
        "supports_autonomous_turns",
        "supports_durable_dispatch",
        "supports_active_turn_fence",
    ],
)
def test_fence_capability_guard_rejects_missing_false_or_raising_capability(
    capability_name: str,
) -> None:
    class Repository:
        def supports_autonomous_turns(self) -> bool:
            return capability_name != "supports_autonomous_turns"

        def supports_durable_dispatch(self) -> bool:
            return capability_name != "supports_durable_dispatch"

        def supports_active_turn_fence(self) -> bool:
            if capability_name == "supports_active_turn_fence":
                raise RuntimeError("private payload")
            return True

    repository = Repository()
    with pytest.raises(ActiveTurnFenceUnsupported) as exc_info:
        require_active_turn_fence_repository(repository, repository)
    assert "private payload" not in str(exc_info.value)


def test_fence_errors_expose_stable_codes() -> None:
    assert ActiveTurnFenceError.code == "active_turn_fence_error"
    assert ActiveTurnFenceUnsupported.code == "active_turn_fence_unsupported"
    assert ActiveTurnFenceRejected.code == "active_turn_fence_rejected"
    assert (
        ActiveTurnFenceTransactionError.code
        == "active_turn_fence_transaction_error"
    )
