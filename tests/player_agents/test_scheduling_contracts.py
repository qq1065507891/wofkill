# -*- coding: utf-8 -*-
"""
验证串行公开调度、回合准入和托管回合的严格契约。

作者: Project contributors
创建日期: 2026-07-30
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
)

HASH = "a" * 64
CREATED_AT = datetime(2026, 7, 30, 9, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 7, 30, 9, 5, tzinfo=timezone.utc)


def _window() -> LegalActionWindow:
    return LegalActionWindow(
        window_id="speech-d1",
        version=1,
        game_id="game-1",
        task_type="day_speech",
        conflict_class=ConflictClass.SERIAL_PUBLIC,
        participant_ids=("p01", "p02"),
        legal_actions=("speech",),
        legal_target_ids=("p01", "p02"),
        opened_revision=4,
        deadline=datetime(2026, 7, 30, 10, tzinfo=timezone.utc),
    )


def _schedule() -> SerialPublicSchedule:
    return SerialPublicSchedule(
        schedule_id="schedule-1",
        game_id="game-1",
        window=_window(),
        slots=(
            SerialPublicSlot(ordinal=0, player_id="p01"),
            SerialPublicSlot(ordinal=1, player_id="p02"),
        ),
        next_slot_ordinal=0,
        active_turn_id=None,
        status=SerialPublicScheduleStatus.OPEN,
        state_version=0,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )


def _admission() -> TurnAdmission:
    return TurnAdmission(
        turn_id="turn-1",
        player_id="p01",
        role_id="villager",
        phase="day_discussion",
        revision=RevisionContext(
            base_revision=4,
            window_id="speech-d1",
            window_version=1,
            view_fingerprint=HASH,
        ),
        read_set=(ReadReference(record_id="public-4", revision=4, content_hash=HASH),),
        model_lease_hash=HASH,
        budget=TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        idempotency_key="turn-1:submit",
    )


def _managed_turn() -> ManagedAgentTurn:
    admission = _admission()
    schedule = _schedule()
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
    return ManagedAgentTurn(
        schedule_id=schedule.schedule_id,
        turn=turn,
        state_version=0,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )


def test_schedule_requires_serial_public_ordered_unique_slots() -> None:
    schedule = _schedule()
    assert tuple(slot.ordinal for slot in schedule.slots) == (0, 1)
    assert schedule.current_slot.player_id == "p01"

    payload = schedule.model_dump()
    payload["slots"] = (
        {"ordinal": 0, "player_id": "p01"},
        {"ordinal": 2, "player_id": "p02"},
    )
    with pytest.raises(ValidationError, match="contiguous"):
        SerialPublicSchedule.model_validate(payload)

    payload["slots"] = (
        {"ordinal": 0, "player_id": "p01"},
        {"ordinal": 1, "player_id": "p01"},
    )
    with pytest.raises(ValidationError, match="duplicates"):
        SerialPublicSchedule.model_validate(payload)


def test_schedule_rejects_non_serial_window_and_participant_drift() -> None:
    schedule = _schedule()
    private_window = schedule.window.model_copy(
        update={"conflict_class": ConflictClass.SERIAL_PRIVATE},
    )
    with pytest.raises(ValidationError, match="serial_public"):
        SerialPublicSchedule.model_validate(
            {
                **schedule.model_dump(),
                "window": private_window,
            }
        )

    with pytest.raises(ValidationError, match="window participants"):
        SerialPublicSchedule.model_validate(
            {
                **schedule.model_dump(),
                "slots": ({"ordinal": 0, "player_id": "p01"},),
            }
        )


def test_managed_turn_and_admission_are_strict_frozen_models() -> None:
    admission = _admission()
    with pytest.raises(ValidationError):
        TurnAdmission.model_validate(
            {
                **admission.model_dump(),
                "unexpected": True,
            }
        )
    with pytest.raises((ValidationError, TypeError)):
        admission.player_id = "p09"  # type: ignore[misc]

    managed = _managed_turn()
    assert managed.turn.status is AgentTurnStatus.OPEN
    assert "status" not in type(managed).model_fields
    with pytest.raises(ValidationError):
        managed.state_version = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "updates",
    [
        {"next_slot_ordinal": 2},
        {
            "status": SerialPublicScheduleStatus.CLOSED,
            "active_turn_id": "turn-1",
        },
        {"updated_at": datetime(2026, 7, 30, 10)},  # noqa: DTZ001
        {"updated_at": datetime(2026, 7, 30, 8, tzinfo=timezone.utc)},
    ],
)
def test_schedule_rejects_inconsistent_state(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SerialPublicSchedule.model_validate(
            {
                **_schedule().model_dump(),
                **updates,
            }
        )


def test_schedule_rejects_wrong_game_and_current_slot_on_terminal_state() -> None:
    schedule = _schedule()
    with pytest.raises(ValidationError, match="game_id"):
        SerialPublicSchedule.model_validate(
            {
                **schedule.model_dump(),
                "game_id": "game-2",
            }
        )

    terminal = schedule.model_copy(
        update={"status": SerialPublicScheduleStatus.CLOSED},
    )
    with pytest.raises(ValueError, match="not open"):
        _ = terminal.current_slot


def test_admission_rejects_duplicate_read_references() -> None:
    admission = _admission()
    duplicate = admission.read_set[0].model_copy(update={"revision": 5})
    with pytest.raises(ValidationError, match="read_set record IDs"):
        TurnAdmission.model_validate(
            {
                **admission.model_dump(),
                "read_set": (*admission.read_set, duplicate),
            }
        )


@pytest.mark.parametrize("field_name", ["created_at", "updated_at"])
def test_managed_turn_rejects_naive_and_reversed_timestamps(
    field_name: str,
) -> None:
    managed = _managed_turn()
    with pytest.raises(ValidationError, match="timezone-aware"):
        ManagedAgentTurn.model_validate(
            {
                **managed.model_dump(),
                field_name: datetime(2026, 7, 30, 10),  # noqa: DTZ001
            }
        )

    with pytest.raises(ValidationError, match="must not precede"):
        ManagedAgentTurn.model_validate(
            {
                **managed.model_dump(),
                "updated_at": datetime(2026, 7, 30, 8, tzinfo=timezone.utc),
            }
        )


def test_scheduling_contracts_are_exported_from_public_package() -> None:
    from werewolf_agent.player_agents import contracts

    assert contracts.ManagedAgentTurn is ManagedAgentTurn
    assert contracts.SerialPublicSchedule is SerialPublicSchedule
    assert contracts.SerialPublicScheduleStatus is SerialPublicScheduleStatus
    assert contracts.SerialPublicSlot is SerialPublicSlot
    assert contracts.TerminalDisposition is TerminalDisposition
    assert contracts.TurnAdmission is TurnAdmission
