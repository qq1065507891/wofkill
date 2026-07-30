# -*- coding: utf-8 -*-
"""
验证自主玩家托管回合能力协议和纯状态准备辅助函数。

作者: Project contributors
创建日期: 2026-07-30
"""

from datetime import datetime, timezone

import pytest

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
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnError,
    AutonomousTurnsUnsupported,
    AutonomousTurnTransactionError,
    InvalidScheduleTransition,
    InvalidTurnAdmission,
    ManagedTurnNotFound,
    ScheduleNotFound,
    ScheduleStateConflict,
    TurnStateConflict,
    prepare_active_finish,
    prepare_active_transition,
    prepare_serial_public_admission,
    require_autonomous_turn_repository,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 30, 10, tzinfo=timezone.utc)


def _window(**updates: object) -> LegalActionWindow:
    payload: dict[str, object] = {
        "window_id": "speech-d1",
        "version": 1,
        "game_id": "game-1",
        "task_type": "day_speech",
        "conflict_class": ConflictClass.SERIAL_PUBLIC,
        "participant_ids": ("p01", "p02"),
        "legal_actions": ("speech",),
        "legal_target_ids": ("p01", "p02"),
        "opened_revision": 4,
        "deadline": datetime(2026, 7, 30, 11, tzinfo=timezone.utc),
    }
    payload.update(updates)
    return LegalActionWindow.model_validate(payload)


def _schedule(**updates: object) -> SerialPublicSchedule:
    payload: dict[str, object] = {
        "schedule_id": "schedule-1",
        "game_id": "game-1",
        "window": _window(),
        "slots": (
            SerialPublicSlot(ordinal=0, player_id="p01"),
            SerialPublicSlot(ordinal=1, player_id="p02"),
        ),
        "next_slot_ordinal": 0,
        "active_turn_id": None,
        "status": SerialPublicScheduleStatus.OPEN,
        "state_version": 0,
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(updates)
    return SerialPublicSchedule.model_validate(payload)


def _admission(**updates: object) -> TurnAdmission:
    payload: dict[str, object] = {
        "turn_id": "turn-1",
        "player_id": "p01",
        "role_id": "villager",
        "phase": "day_discussion",
        "revision": RevisionContext(
            base_revision=4,
            window_id="speech-d1",
            window_version=1,
            view_fingerprint=HASH,
        ),
        "read_set": (
            ReadReference(record_id="public-4", revision=4, content_hash=HASH),
        ),
        "model_lease_hash": HASH,
        "budget": TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        "idempotency_key": "turn-1:submit",
    }
    payload.update(updates)
    return TurnAdmission.model_validate(payload)


def _admitted_validating() -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    schedule, managed = prepare_serial_public_admission(
        _schedule(),
        _admission(),
        NOW,
    )
    managed = prepare_active_transition(
        schedule,
        managed,
        AgentTurnStatus.OBSERVING,
    )
    managed = prepare_active_transition(
        schedule,
        managed,
        AgentTurnStatus.THINKING,
    )
    managed = prepare_active_transition(
        schedule,
        managed,
        AgentTurnStatus.SUBMITTED,
    )
    managed = prepare_active_transition(
        schedule,
        managed,
        AgentTurnStatus.VALIDATING,
    )
    return schedule, managed


def test_turn_capability_guard_rejects_legacy_repository() -> None:
    with pytest.raises(AutonomousTurnsUnsupported):
        require_autonomous_turn_repository(object())


def test_turn_capability_guard_requires_callable_true_result() -> None:
    class FalseCapability:
        def supports_autonomous_turns(self) -> bool:
            return False

    class RaisingCapability:
        def supports_autonomous_turns(self) -> bool:
            raise RuntimeError("private payload")

    with pytest.raises(AutonomousTurnsUnsupported):
        require_autonomous_turn_repository(FalseCapability())
    with pytest.raises(AutonomousTurnsUnsupported) as exc_info:
        require_autonomous_turn_repository(RaisingCapability())
    assert "private payload" not in str(exc_info.value)


def test_turn_errors_expose_stable_codes() -> None:
    assert AutonomousTurnError.code == "autonomous_turn_error"
    assert AutonomousTurnsUnsupported.code == "autonomous_turns_unsupported"
    assert ScheduleNotFound.code == "schedule_not_found"
    assert ManagedTurnNotFound.code == "managed_turn_not_found"
    assert ScheduleStateConflict.code == "schedule_state_conflict"
    assert TurnStateConflict.code == "turn_state_conflict"
    assert InvalidScheduleTransition.code == "invalid_schedule_transition"
    assert InvalidTurnAdmission.code == "invalid_turn_admission"
    assert AutonomousTurnTransactionError.code == "autonomous_turn_transaction_error"


def test_prepare_admission_binds_fresh_turn_to_current_slot() -> None:
    schedule, managed = prepare_serial_public_admission(
        _schedule(),
        _admission(),
        NOW,
    )

    assert schedule.active_turn_id == "turn-1"
    assert schedule.state_version == 1
    assert schedule.updated_at == NOW
    assert managed.schedule_id == schedule.schedule_id
    assert managed.turn.status is AgentTurnStatus.OPEN
    assert managed.turn.game_id == schedule.game_id
    assert managed.turn.task_type == schedule.window.task_type
    assert managed.turn.window == schedule.window
    assert managed.state_version == 0
    assert managed.created_at == NOW


@pytest.mark.parametrize(
    "schedule_updates, admission_updates",
    [
        ({"status": SerialPublicScheduleStatus.CLOSED, "next_slot_ordinal": 2}, {}),
        ({"active_turn_id": "existing-turn"}, {}),
        ({}, {"player_id": "p02"}),
        ({}, {"revision": RevisionContext(
            base_revision=4,
            window_id="stale-window",
            window_version=1,
            view_fingerprint=HASH,
        )}),
        ({}, {"revision": RevisionContext(
            base_revision=4,
            window_id="speech-d1",
            window_version=2,
            view_fingerprint=HASH,
        )}),
        ({}, {"revision": RevisionContext(
            base_revision=3,
            window_id="speech-d1",
            window_version=1,
            view_fingerprint=HASH,
        )}),
    ],
)
def test_prepare_admission_rejects_stale_or_ineligible_input(
    schedule_updates: dict[str, object],
    admission_updates: dict[str, object],
) -> None:
    with pytest.raises(InvalidTurnAdmission):
        prepare_serial_public_admission(
            _schedule(**schedule_updates),
            _admission(**admission_updates),
            NOW,
        )


def test_prepare_admission_maps_model_validation_without_payload() -> None:
    # Injecting an invalid role is not possible through the strict model, so
    # use a malformed object to verify the helper's stable error boundary.
    with pytest.raises(InvalidTurnAdmission) as exc_info:
        prepare_serial_public_admission(  # type: ignore[arg-type]
            object(),
            _admission(),
            NOW,
        )
    assert "object at" not in str(exc_info.value)


def test_prepare_transition_requires_matching_active_identity() -> None:
    schedule, managed = prepare_serial_public_admission(_schedule(), _admission(), NOW)

    with pytest.raises(InvalidScheduleTransition):
        prepare_active_transition(
            _schedule(active_turn_id="other-turn"),
            managed,
            AgentTurnStatus.OBSERVING,
        )
    with pytest.raises(InvalidScheduleTransition):
        prepare_active_transition(
            schedule,
            ManagedAgentTurn.model_validate(
                {**managed.model_dump(), "schedule_id": "other-schedule"},
            ),
            AgentTurnStatus.OBSERVING,
        )
    with pytest.raises(InvalidScheduleTransition):
        prepare_active_transition(schedule, managed, AgentTurnStatus.COMMITTED)


@pytest.mark.parametrize(
    "schedule_updates",
    [
        {
            "game_id": "game-2",
            "window": _window(game_id="game-2"),
        },
        {"window": _window(window_id="speech-d2")},
        {"window": _window(version=2)},
    ],
)
def test_prepare_transition_rejects_schedule_context_drift(
    schedule_updates: dict[str, object],
) -> None:
    _, managed = prepare_serial_public_admission(_schedule(), _admission(), NOW)
    drifted_schedule = _schedule(
        active_turn_id=managed.turn.turn_id,
        **schedule_updates,
    )

    with pytest.raises(InvalidScheduleTransition):
        prepare_active_transition(
            drifted_schedule,
            managed,
            AgentTurnStatus.OBSERVING,
        )


def test_prepare_transition_increments_only_turn_version() -> None:
    schedule, managed = prepare_serial_public_admission(_schedule(), _admission(), NOW)
    updated = prepare_active_transition(schedule, managed, AgentTurnStatus.OBSERVING)

    assert updated.turn.status is AgentTurnStatus.OBSERVING
    assert updated.state_version == 1
    assert schedule.state_version == 1
    assert schedule.active_turn_id == managed.turn.turn_id


def test_prepare_finish_advance_consumes_slot_atomically() -> None:
    schedule, managed = _admitted_validating()
    updated_schedule, updated_turn = prepare_active_finish(
        schedule,
        managed,
        AgentTurnStatus.COMMITTED,
        TerminalDisposition.ADVANCE,
        reason_code=None,
        now=NOW,
    )

    assert updated_turn.turn.status is AgentTurnStatus.COMMITTED
    assert updated_turn.state_version == managed.state_version + 1
    assert updated_schedule.active_turn_id is None
    assert updated_schedule.next_slot_ordinal == 1
    assert updated_schedule.status is SerialPublicScheduleStatus.OPEN
    assert updated_schedule.state_version == schedule.state_version + 1


def test_prepare_finish_last_advance_closes_schedule() -> None:
    schedule = _schedule(next_slot_ordinal=1)
    admission = _admission(player_id="p02", turn_id="turn-2", idempotency_key="turn-2:submit")
    schedule, managed = prepare_serial_public_admission(schedule, admission, NOW)
    updated_schedule, updated_turn = prepare_active_finish(
        schedule,
        managed,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.ADVANCE,
        reason_code="cancelled_by_host",
        now=NOW,
    )

    assert updated_turn.turn.status is AgentTurnStatus.CANCELLED
    assert updated_turn.terminal_reason == "cancelled_by_host"
    assert updated_schedule.status is SerialPublicScheduleStatus.CLOSED
    assert updated_schedule.next_slot_ordinal == 2
    assert updated_schedule.active_turn_id is None


def test_prepare_finish_replace_retains_current_slot() -> None:
    schedule, managed = _admitted_validating()
    updated_schedule, updated_turn = prepare_active_finish(
        schedule,
        managed,
        AgentTurnStatus.EXPIRED,
        TerminalDisposition.REPLACE,
        reason_code="expired",
        now=NOW,
    )

    assert updated_turn.turn.status is AgentTurnStatus.EXPIRED
    assert updated_schedule.next_slot_ordinal == schedule.next_slot_ordinal
    assert updated_schedule.status is SerialPublicScheduleStatus.OPEN
    assert updated_schedule.active_turn_id is None


def test_prepare_finish_close_cancels_schedule() -> None:
    schedule, managed = _admitted_validating()
    updated_schedule, _ = prepare_active_finish(
        schedule,
        managed,
        AgentTurnStatus.CANCELLED,
        TerminalDisposition.CLOSE,
        reason_code="game_over",
        now=NOW,
    )

    assert updated_schedule.status is SerialPublicScheduleStatus.CANCELLED
    assert updated_schedule.next_slot_ordinal == schedule.next_slot_ordinal
    assert updated_schedule.active_turn_id is None


def test_prepare_finish_rejects_wrong_identity_and_non_terminal_status() -> None:
    schedule, managed = _admitted_validating()
    with pytest.raises(InvalidScheduleTransition):
        prepare_active_finish(
            _schedule(active_turn_id="other-turn"),
            managed,
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            reason_code=None,
            now=NOW,
        )
    with pytest.raises(InvalidScheduleTransition):
        prepare_active_finish(
            schedule,
            managed,
            AgentTurnStatus.THINKING,
            TerminalDisposition.ADVANCE,
            reason_code=None,
            now=NOW,
        )
