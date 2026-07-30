# -*- coding: utf-8 -*-
"""
验证自主玩家托管回合能力协议和纯状态准备辅助函数。

作者: Project contributors
创建日期: 2026-07-30
修改日期: 2026-07-30
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from werewolf_agent.core.models import GameState
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
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository

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


def _memory_repository() -> InMemoryGameRepository:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="game-1"))
    return repository


def _sqlite_repository(tmp_path):
    repository = SqliteGameRepository(str(tmp_path / "autonomous-turns.db"))
    repository.save_game(GameState(game_id="game-1"))
    return repository


def _memory_admit_validating(
    repository: InMemoryGameRepository,
    *,
    schedule_id: str = "schedule-1",
    turn_id: str = "turn-1",
    expected_schedule_version: int = 0,
    player_id: str = "p01",
) -> ManagedAgentTurn:
    admission = _admission(
        turn_id=turn_id,
        player_id=player_id,
        idempotency_key=f"{turn_id}:submit",
    )
    managed = repository.admit_serial_public_turn(
        schedule_id,
        expected_schedule_version,
        admission,
    )
    for next_status in (
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.THINKING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.VALIDATING,
    ):
        managed = repository.transition_active_turn(
            turn_id,
            managed.state_version,
            next_status,
        )
    return managed


def test_memory_schedule_creation_and_current_slot_admission() -> None:
    repository = _memory_repository()
    created = repository.create_serial_public_schedule(_schedule())

    assert repository.supports_autonomous_turns() is True
    assert created == _schedule()
    assert repository.load_active_serial_public_schedule("game-1") == created
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        expected_schedule_version=0,
        admission=_admission(),
    )

    assert managed.turn.player_id == "p01"
    stored = repository.load_serial_public_schedule(created.schedule_id)
    assert stored is not None
    assert stored.active_turn_id == managed.turn.turn_id
    assert stored.state_version == 1


def test_memory_schedule_creation_rejects_missing_game_and_duplicates() -> None:
    repository = InMemoryGameRepository()
    with pytest.raises(AutonomousTurnTransactionError):
        repository.create_serial_public_schedule(_schedule())

    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    with pytest.raises(AutonomousTurnTransactionError):
        repository.create_serial_public_schedule(_schedule())
    with pytest.raises(AutonomousTurnTransactionError):
        repository.create_serial_public_schedule(
            _schedule(schedule_id="schedule-2"),
        )


def test_memory_admission_rejects_wrong_current_slot_and_stale_schedule() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())

    with pytest.raises(InvalidTurnAdmission):
        repository.admit_serial_public_turn(
            "schedule-1",
            0,
            _admission(player_id="p02"),
        )
    with pytest.raises(ScheduleStateConflict):
        repository.admit_serial_public_turn(
            "schedule-1",
            4,
            _admission(),
        )
    with pytest.raises(ScheduleNotFound):
        repository.admit_serial_public_turn("missing", 0, _admission())
    assert repository.load_managed_turn("turn-1") is None


def test_memory_stale_admission_does_not_publish_partial_turn() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    repository.admit_serial_public_turn("schedule-1", 0, _admission())

    with pytest.raises(ScheduleStateConflict):
        repository.admit_serial_public_turn(
            "schedule-1",
            0,
            _admission(turn_id="turn-stale", idempotency_key="turn-stale:submit"),
        )
    assert repository.load_managed_turn("turn-stale") is None
    stored = repository.load_serial_public_schedule("schedule-1")
    assert stored is not None
    assert stored.active_turn_id == "turn-1"


def test_memory_transition_cas_and_missing_turn_errors() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    managed = repository.admit_serial_public_turn("schedule-1", 0, _admission())

    with pytest.raises(TurnStateConflict):
        repository.transition_active_turn(
            managed.turn.turn_id,
            expected_turn_version=99,
            next_status=AgentTurnStatus.OBSERVING,
        )
    with pytest.raises(ManagedTurnNotFound):
        repository.transition_active_turn(
            "missing-turn",
            expected_turn_version=0,
            next_status=AgentTurnStatus.OBSERVING,
        )

    updated = repository.transition_active_turn(
        managed.turn.turn_id,
        expected_turn_version=0,
        next_status=AgentTurnStatus.OBSERVING,
    )
    assert updated.turn.status is AgentTurnStatus.OBSERVING
    assert updated.state_version == 1


def test_memory_finish_advance_replace_and_close() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    managed = _memory_admit_validating(repository)
    advanced = repository.finish_active_turn(
        "schedule-1",
        expected_schedule_version=1,
        turn_id=managed.turn.turn_id,
        expected_turn_version=4,
        terminal_status=AgentTurnStatus.COMMITTED,
        disposition=TerminalDisposition.ADVANCE,
        reason_code=None,
    )
    assert advanced.status is SerialPublicScheduleStatus.OPEN
    assert advanced.next_slot_ordinal == 1
    assert repository.load_active_serial_public_schedule("game-1") == advanced
    assert repository.load_managed_turn("turn-1").turn.status is AgentTurnStatus.COMMITTED  # type: ignore[union-attr]

    validating = _memory_admit_validating(
        repository,
        turn_id="turn-2",
        expected_schedule_version=2,
        player_id="p02",
    )
    # A replacement must retain the current slot and permit a later admission.
    replaced = repository.finish_active_turn(
        "schedule-1",
        expected_schedule_version=3,
        turn_id=validating.turn.turn_id,
        expected_turn_version=4,
        terminal_status=AgentTurnStatus.EXPIRED,
        disposition=TerminalDisposition.REPLACE,
        reason_code="expired",
    )
    assert replaced.status is SerialPublicScheduleStatus.OPEN
    assert replaced.next_slot_ordinal == 1

    validating = _memory_admit_validating(
        repository,
        turn_id="turn-3",
        expected_schedule_version=4,
        player_id="p02",
    )

    closed = repository.finish_active_turn(
        "schedule-1",
        expected_schedule_version=5,
        turn_id=validating.turn.turn_id,
        expected_turn_version=4,
        terminal_status=AgentTurnStatus.CANCELLED,
        disposition=TerminalDisposition.CLOSE,
        reason_code="host_closed",
    )
    assert closed.status is SerialPublicScheduleStatus.CANCELLED
    assert repository.load_active_serial_public_schedule("game-1") is None


def test_memory_finish_rejects_stale_versions_and_wrong_identity() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    managed = _memory_admit_validating(repository)

    with pytest.raises(ScheduleStateConflict):
        repository.finish_active_turn(
            "schedule-1",
            expected_schedule_version=0,
            turn_id=managed.turn.turn_id,
            expected_turn_version=managed.state_version,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )
    with pytest.raises(TurnStateConflict):
        repository.finish_active_turn(
            "schedule-1",
            expected_schedule_version=1,
            turn_id=managed.turn.turn_id,
            expected_turn_version=0,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )
    with pytest.raises(ManagedTurnNotFound):
        repository.finish_active_turn(
            "schedule-1",
            expected_schedule_version=1,
            turn_id="missing-turn",
            expected_turn_version=0,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )
    with pytest.raises(ManagedTurnNotFound):
        repository.finish_active_turn(
            "schedule-1",
            expected_schedule_version=1,
            turn_id="other-turn",
            expected_turn_version=managed.state_version,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )


def test_memory_reads_are_defensive_and_delete_game_cleans_turn_state() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())
    managed = repository.admit_serial_public_turn("schedule-1", 0, _admission())

    schedule_one = repository.load_serial_public_schedule("schedule-1")
    schedule_two = repository.load_serial_public_schedule("schedule-1")
    managed_one = repository.load_managed_turn(managed.turn.turn_id)
    managed_two = repository.load_managed_turn(managed.turn.turn_id)
    assert schedule_one is not schedule_two
    assert managed_one is not managed_two
    assert schedule_one is not None and schedule_two is not None
    assert schedule_one.window is not schedule_two.window
    assert managed_one is not None and managed_two is not None
    assert managed_one.turn is not managed_two.turn

    repository.delete_game("game-1")
    assert repository.load_serial_public_schedule("schedule-1") is None
    assert repository.load_managed_turn(managed.turn.turn_id) is None
    assert repository.load_active_serial_public_schedule("game-1") is None
    assert repository.list_open_serial_public_schedules() == ()


def test_memory_prepare_then_publish_rolls_back_on_helper_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())

    def fail_prepare(*args: object, **kwargs: object) -> object:
        raise AutonomousTurnTransactionError("forced failure")

    monkeypatch.setattr(
        "werewolf_agent.storage.memory_store.prepare_serial_public_admission",
        fail_prepare,
    )
    with pytest.raises(AutonomousTurnTransactionError):
        repository.admit_serial_public_turn("schedule-1", 0, _admission())
    stored = repository.load_serial_public_schedule("schedule-1")
    assert stored is not None
    assert stored.active_turn_id is None
    assert stored.state_version == 0
    assert repository.load_managed_turn("turn-1") is None


def test_memory_concurrent_duplicate_admission_is_single_winner() -> None:
    repository = _memory_repository()
    repository.create_serial_public_schedule(_schedule())

    def attempt(index: int) -> tuple[str, type[BaseException] | None]:
        try:
            repository.admit_serial_public_turn(
                "schedule-1",
                expected_schedule_version=0,
                admission=_admission(
                    turn_id=f"turn-{index}",
                    idempotency_key=f"turn-{index}:submit",
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - classify stable repository errors
            return "error", type(exc)
        return "success", None

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(attempt, range(20)))

    assert sum(result == "success" for result, _ in results) == 1
    failures = [error for result, error in results if result == "error"]
    assert len(failures) == 19
    assert set(failures) <= {ScheduleStateConflict, InvalidTurnAdmission}
    managed_turns = [
        repository.load_managed_turn(f"turn-{index}")
        for index in range(20)
    ]
    assert sum(turn is not None for turn in managed_turns) == 1


def test_sqlite_schema_capability_and_schedule_lifecycle(tmp_path) -> None:
    repository = _sqlite_repository(tmp_path)
    try:
        assert repository.supports_autonomous_turns() is True
        tables = {
            row[0]
            for row in repository._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in repository._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'",
            ).fetchall()
        }
        assert {
            "autonomous_serial_public_schedules",
            "autonomous_managed_turns",
        } <= tables
        assert {
            "uq_open_serial_public_schedule",
            "idx_managed_turn_schedule_status",
        } <= indexes
        created = repository.create_serial_public_schedule(_schedule())
        assert repository.load_serial_public_schedule("schedule-1") == created
        assert repository.load_active_serial_public_schedule("game-1") == created
        assert repository.list_open_serial_public_schedules() == (created,)
    finally:
        repository.close()


def test_sqlite_admission_transition_finish_and_defensive_reads(tmp_path) -> None:
    repository = _sqlite_repository(tmp_path)
    try:
        repository.create_serial_public_schedule(_schedule())
        managed = repository.admit_serial_public_turn("schedule-1", 0, _admission())
        assert managed.turn.status is AgentTurnStatus.OPEN
        observed = repository.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.OBSERVING,
        )
        assert observed.state_version == 1
        assert repository.load_managed_turn("turn-1") is not observed
        repository.close()
        repository = SqliteGameRepository(str(tmp_path / "autonomous-turns.db"))
        current = repository.load_managed_turn("turn-1")
        assert current is not None
        assert current.turn.status is AgentTurnStatus.OBSERVING
        for next_status in (
            AgentTurnStatus.THINKING,
            AgentTurnStatus.SUBMITTED,
            AgentTurnStatus.VALIDATING,
        ):
            current = repository.transition_active_turn(
                "turn-1", current.state_version, next_status,
            )
        finished = repository.finish_active_turn(
            "schedule-1",
            expected_schedule_version=1,
            turn_id="turn-1",
            expected_turn_version=current.state_version,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )
        assert finished.next_slot_ordinal == 1
        assert finished.active_turn_id is None
        assert repository.load_active_serial_public_schedule("game-1") == finished
        assert repository.list_open_serial_public_schedules() == (finished,)
    finally:
        repository.close()


def test_sqlite_admission_conflict_rolls_back_schedule_pointer(tmp_path) -> None:
    repository = _sqlite_repository(tmp_path)
    try:
        repository.create_serial_public_schedule(_schedule())
        conflict_managed = ManagedAgentTurn(
            schedule_id="schedule-1",
            turn=AgentTurn(
                turn_id="turn-1",
                game_id="game-1",
                player_id="p09",
                role_id="villager",
                phase="day_discussion",
                task_type="day_speech",
                revision=_admission().revision,
                window=_window(participant_ids=("p01", "p02", "p09")),
                read_set=(),
                model_lease_hash=HASH,
                budget=_admission().budget,
                status=AgentTurnStatus.OPEN,
                idempotency_key="conflict",
            ),
            state_version=0,
            created_at=NOW,
            updated_at=NOW,
        )
        repository._conn.execute(
            "INSERT INTO autonomous_managed_turns "
            "(turn_id, schedule_id, game_id, player_id, status, state_version, "
            "turn_json, terminal_reason, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "turn-1", "schedule-1", "game-1", "p09", "open", 0,
                conflict_managed.model_dump_json(), None,
                NOW.isoformat(), NOW.isoformat(),
            ),
        )
        repository._conn.commit()
        with pytest.raises(AutonomousTurnTransactionError):
            repository.admit_serial_public_turn("schedule-1", 0, _admission())
        stored = repository.load_serial_public_schedule("schedule-1")
        assert stored is not None
        assert stored.active_turn_id is None
        assert stored.state_version == 0
    finally:
        repository.close()


def test_sqlite_concurrent_duplicate_admission_is_single_winner(tmp_path) -> None:
    db_path = tmp_path / "concurrent.db"
    seed = SqliteGameRepository(str(db_path))
    seed.save_game(GameState(game_id="game-1"))
    seed.create_serial_public_schedule(_schedule())
    seed.close()

    def attempt(index: int) -> tuple[str, type[BaseException] | None]:
        repository = SqliteGameRepository(str(db_path))
        try:
            repository.admit_serial_public_turn(
                "schedule-1", 0,
                _admission(
                    turn_id=f"turn-{index}",
                    idempotency_key=f"turn-{index}:submit",
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - classify repository errors
            return "error", type(exc)
        finally:
            repository.close()
        return "success", None

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(attempt, range(20)))
    assert sum(result == "success" for result, _ in results) == 1
    assert sum(result == "error" for result, _ in results) == 19
    repository = SqliteGameRepository(str(db_path))
    try:
        turns = [repository.load_managed_turn(f"turn-{index}") for index in range(20)]
        assert sum(turn is not None for turn in turns) == 1
    finally:
        repository.close()
