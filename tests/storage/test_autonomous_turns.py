# -*- coding: utf-8 -*-
"""
验证自主玩家托管回合契约、状态准备逻辑与 memory/SQLite 原子语义一致性。

作者: Project contributors
创建日期: 2026-07-30
修改日期: 2026-07-30
"""

import json
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


def test_sqlite_schedule_creation_rolls_back_when_final_read_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _sqlite_repository(tmp_path)
    original_load = repository._load_schedule_unlocked
    load_count = 0

    def fail_final_load(schedule_id: str) -> SerialPublicSchedule | None:
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise RuntimeError("forced final read failure")
        return original_load(schedule_id)

    monkeypatch.setattr(repository, "_load_schedule_unlocked", fail_final_load)
    try:
        with pytest.raises(AutonomousTurnTransactionError):
            repository.create_serial_public_schedule(_schedule())
        monkeypatch.setattr(repository, "_load_schedule_unlocked", original_load)
        assert repository.load_serial_public_schedule("schedule-1") is None
    finally:
        repository.close()


def test_sqlite_schedule_json_is_canonical_utf8(tmp_path) -> None:
    import json

    repository = SqliteGameRepository(str(tmp_path / "canonical.db"))
    repository.save_game(GameState(game_id="游戏-甲"))
    schedule = _schedule(
        schedule_id="日间-调度",
        game_id="游戏-甲",
        window=_window(
            window_id="发言窗口",
            game_id="游戏-甲",
            task_type="公开发言",
        ),
    )
    try:
        repository.create_serial_public_schedule(schedule)
        raw = repository._conn.execute(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = ?",
            (schedule.schedule_id,),
        ).fetchone()[0]
        assert "公开发言" in raw
        assert raw == json.dumps(
            json.loads(raw),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    finally:
        repository.close()


def test_sqlite_lists_open_schedules_in_deterministic_cross_game_order(
    tmp_path,
) -> None:
    repository = SqliteGameRepository(str(tmp_path / "ordered.db"))
    schedules = (
        _schedule(
            schedule_id="schedule-z",
            game_id="game-z",
            window=_window(game_id="game-z"),
        ),
        _schedule(
            schedule_id="schedule-a",
            game_id="game-a",
            window=_window(game_id="game-a"),
        ),
        _schedule(
            schedule_id="schedule-earlier",
            game_id="game-earlier",
            window=_window(game_id="game-earlier"),
            created_at=datetime(2026, 7, 30, 9, tzinfo=timezone.utc),
            updated_at=datetime(2026, 7, 30, 9, tzinfo=timezone.utc),
        ),
    )
    try:
        for schedule in schedules:
            repository.save_game(GameState(game_id=schedule.game_id))
            repository.create_serial_public_schedule(schedule)
        assert tuple(
            schedule.schedule_id
            for schedule in repository.list_open_serial_public_schedules()
        ) == ("schedule-earlier", "schedule-a", "schedule-z")
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


def test_sqlite_delete_game_cascades_schedule_and_managed_turn(tmp_path) -> None:
    repository = _sqlite_repository(tmp_path)
    try:
        repository.create_serial_public_schedule(_schedule())
        repository.admit_serial_public_turn("schedule-1", 0, _admission())
        repository.delete_game("game-1")
        assert repository.load_serial_public_schedule("schedule-1") is None
        assert repository.load_managed_turn("turn-1") is None
        assert repository.load_active_serial_public_schedule("game-1") is None
        assert repository.list_open_serial_public_schedules() == ()
    finally:
        repository.close()


def test_sqlite_finish_rolls_back_managed_update_when_schedule_update_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _sqlite_repository(tmp_path)
    try:
        repository.create_serial_public_schedule(_schedule())
        managed = repository.admit_serial_public_turn("schedule-1", 0, _admission())
        for next_status in (
            AgentTurnStatus.OBSERVING,
            AgentTurnStatus.THINKING,
            AgentTurnStatus.SUBMITTED,
            AgentTurnStatus.VALIDATING,
        ):
            managed = repository.transition_active_turn(
                managed.turn.turn_id,
                managed.state_version,
                next_status,
            )

        def fail_schedule_update(
            schedule: SerialPublicSchedule,
            expected_version: int,
        ) -> None:
            del schedule, expected_version
            raise ScheduleStateConflict("forced schedule conflict")

        original_update = repository._update_schedule_unlocked
        monkeypatch.setattr(
            repository,
            "_update_schedule_unlocked",
            fail_schedule_update,
        )
        with pytest.raises(ScheduleStateConflict):
            repository.finish_active_turn(
                "schedule-1",
                expected_schedule_version=1,
                turn_id="turn-1",
                expected_turn_version=managed.state_version,
                terminal_status=AgentTurnStatus.COMMITTED,
                disposition=TerminalDisposition.ADVANCE,
                reason_code=None,
            )
        monkeypatch.setattr(
            repository,
            "_update_schedule_unlocked",
            original_update,
        )
        stored_turn = repository.load_managed_turn("turn-1")
        stored_schedule = repository.load_serial_public_schedule("schedule-1")
        assert stored_turn is not None
        assert stored_turn.turn.status is AgentTurnStatus.VALIDATING
        assert stored_turn.state_version == managed.state_version
        assert stored_schedule is not None
        assert stored_schedule.active_turn_id == "turn-1"
        assert stored_schedule.state_version == 1
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
    failures = [error for result, error in results if result == "error"]
    assert len(failures) == 19
    assert set(failures) <= {ScheduleStateConflict, InvalidTurnAdmission}
    repository = SqliteGameRepository(str(db_path))
    try:
        turns = [repository.load_managed_turn(f"turn-{index}") for index in range(20)]
        assert sum(turn is not None for turn in turns) == 1
    finally:
        repository.close()


# ===========================================================================
# 最终 memory/SQLite 共享原子性矩阵
# ===========================================================================


SharedTurnRepository = InMemoryGameRepository | SqliteGameRepository


class _FixedRepositoryDateTime(datetime):
    """为两个仓储实现提供相同的持久化时间。"""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return NOW.replace(tzinfo=None)
        return NOW.astimezone(tz)


def _freeze_repository_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "werewolf_agent.storage.memory_store.datetime",
        _FixedRepositoryDateTime,
    )
    monkeypatch.setattr(
        "werewolf_agent.storage.sqlite_store.datetime",
        _FixedRepositoryDateTime,
    )


def _shared_repository(
    repository_kind: str,
    tmp_path,
    *,
    database_name: str = "shared-autonomous-turns.db",
) -> SharedTurnRepository:
    if repository_kind == "memory":
        repository: SharedTurnRepository = InMemoryGameRepository()
    else:
        assert repository_kind == "sqlite"
        repository = SqliteGameRepository(str(tmp_path / database_name))
    repository.save_game(GameState(game_id="game-1"))
    return repository


def _close_shared_repository(repository: SharedTurnRepository) -> None:
    if isinstance(repository, SqliteGameRepository):
        repository.close()


def _canonical_contract_bytes(
    value: SerialPublicSchedule | ManagedAgentTurn,
) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _shared_admit_to_validating(
    repository: SharedTurnRepository,
    *,
    schedule: SerialPublicSchedule | None = None,
    admission: TurnAdmission | None = None,
) -> tuple[SerialPublicSchedule, ManagedAgentTurn]:
    created = repository.create_serial_public_schedule(schedule or _schedule())
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        created.state_version,
        admission or _admission(),
    )
    for next_status in (
        AgentTurnStatus.OBSERVING,
        AgentTurnStatus.THINKING,
        AgentTurnStatus.SUBMITTED,
        AgentTurnStatus.VALIDATING,
    ):
        managed = repository.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            next_status,
        )
    active = repository.load_serial_public_schedule(created.schedule_id)
    assert active is not None
    return active, managed


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_schedule_creation_and_current_slot_admission(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    try:
        created = repository.create_serial_public_schedule(_schedule())
        assert _canonical_contract_bytes(created) == _canonical_contract_bytes(
            _schedule(),
        )
        assert repository.load_active_serial_public_schedule("game-1") == created

        with pytest.raises(InvalidTurnAdmission):
            repository.admit_serial_public_turn(
                created.schedule_id,
                created.state_version,
                _admission(player_id="p02"),
            )
        assert repository.load_managed_turn("turn-1") is None

        managed = repository.admit_serial_public_turn(
            created.schedule_id,
            created.state_version,
            _admission(),
        )
        active = repository.load_serial_public_schedule(created.schedule_id)
        assert active is not None
        assert managed.turn.player_id == created.current_slot.player_id
        assert managed.turn.status is AgentTurnStatus.OPEN
        assert active.active_turn_id == managed.turn.turn_id
        assert active.state_version == created.state_version + 1
    finally:
        _close_shared_repository(repository)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_stale_cas_never_publishes_partial_state(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    try:
        created = repository.create_serial_public_schedule(_schedule())
        with pytest.raises(ScheduleStateConflict):
            repository.admit_serial_public_turn(
                created.schedule_id,
                created.state_version + 1,
                _admission(),
            )
        assert repository.load_managed_turn("turn-1") is None

        managed = repository.admit_serial_public_turn(
            created.schedule_id,
            created.state_version,
            _admission(),
        )
        managed_before = _canonical_contract_bytes(managed)
        with pytest.raises(TurnStateConflict):
            repository.transition_active_turn(
                managed.turn.turn_id,
                managed.state_version + 1,
                AgentTurnStatus.OBSERVING,
            )
        stored_managed = repository.load_managed_turn(managed.turn.turn_id)
        assert stored_managed is not None
        assert _canonical_contract_bytes(stored_managed) == managed_before

        for next_status in (
            AgentTurnStatus.OBSERVING,
            AgentTurnStatus.THINKING,
            AgentTurnStatus.SUBMITTED,
            AgentTurnStatus.VALIDATING,
        ):
            managed = repository.transition_active_turn(
                managed.turn.turn_id,
                managed.state_version,
                next_status,
            )
        active = repository.load_serial_public_schedule(created.schedule_id)
        assert active is not None
        active_before = _canonical_contract_bytes(active)
        managed_before = _canonical_contract_bytes(managed)

        with pytest.raises(ScheduleStateConflict):
            repository.finish_active_turn(
                active.schedule_id,
                active.state_version - 1,
                managed.turn.turn_id,
                managed.state_version,
                AgentTurnStatus.COMMITTED,
                TerminalDisposition.ADVANCE,
                None,
            )
        with pytest.raises(TurnStateConflict):
            repository.finish_active_turn(
                active.schedule_id,
                active.state_version,
                managed.turn.turn_id,
                managed.state_version - 1,
                AgentTurnStatus.COMMITTED,
                TerminalDisposition.ADVANCE,
                None,
            )
        stored_schedule = repository.load_serial_public_schedule(active.schedule_id)
        stored_managed = repository.load_managed_turn(managed.turn.turn_id)
        assert stored_schedule is not None
        assert stored_managed is not None
        assert _canonical_contract_bytes(stored_schedule) == active_before
        assert _canonical_contract_bytes(stored_managed) == managed_before
    finally:
        _close_shared_repository(repository)


_HOST_NONTERMINAL_EDGES = (
    (AgentTurnStatus.OPEN, AgentTurnStatus.OBSERVING),
    (AgentTurnStatus.OBSERVING, AgentTurnStatus.THINKING),
    (AgentTurnStatus.THINKING, AgentTurnStatus.WAITING_TOOL),
    (AgentTurnStatus.WAITING_TOOL, AgentTurnStatus.THINKING),
    (AgentTurnStatus.THINKING, AgentTurnStatus.COMPACTING),
    (AgentTurnStatus.COMPACTING, AgentTurnStatus.THINKING),
    (AgentTurnStatus.THINKING, AgentTurnStatus.SUBMITTED),
    (AgentTurnStatus.SUBMITTED, AgentTurnStatus.VALIDATING),
    (AgentTurnStatus.VALIDATING, AgentTurnStatus.REPAIRING),
    (AgentTurnStatus.REPAIRING, AgentTurnStatus.SUBMITTED),
)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_host_nonterminal_transition_edges(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    try:
        created = repository.create_serial_public_schedule(_schedule())
        managed = repository.admit_serial_public_turn(
            created.schedule_id,
            created.state_version,
            _admission(),
        )
        for current_status, next_status in _HOST_NONTERMINAL_EDGES:
            assert managed.turn.status is current_status
            previous_version = managed.state_version
            managed = repository.transition_active_turn(
                managed.turn.turn_id,
                previous_version,
                next_status,
            )
            assert managed.turn.status is next_status
            assert managed.state_version == previous_version + 1
    finally:
        _close_shared_repository(repository)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
@pytest.mark.parametrize(
    (
        "terminal_status",
        "disposition",
        "reason_code",
        "expected_status",
        "expected_ordinal",
    ),
    (
        (
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            None,
            SerialPublicScheduleStatus.OPEN,
            1,
        ),
        (
            AgentTurnStatus.EXPIRED,
            TerminalDisposition.REPLACE,
            "deadline_expired",
            SerialPublicScheduleStatus.OPEN,
            0,
        ),
        (
            AgentTurnStatus.CANCELLED,
            TerminalDisposition.CLOSE,
            "operator_cancelled",
            SerialPublicScheduleStatus.CANCELLED,
            0,
        ),
    ),
)
def test_shared_atomic_finish_matrix(
    repository_kind: str,
    terminal_status: AgentTurnStatus,
    disposition: TerminalDisposition,
    reason_code: str | None,
    expected_status: SerialPublicScheduleStatus,
    expected_ordinal: int,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    try:
        schedule, managed = _shared_admit_to_validating(repository)
        updated = repository.finish_active_turn(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            terminal_status,
            disposition,
            reason_code,
        )
        stored = repository.load_managed_turn(managed.turn.turn_id)
        assert stored is not None
        assert stored.turn.status is terminal_status
        assert stored.terminal_reason == reason_code
        assert stored.state_version == managed.state_version + 1
        assert updated.status is expected_status
        assert updated.next_slot_ordinal == expected_ordinal
        assert updated.active_turn_id is None
        assert updated.state_version == schedule.state_version + 1
    finally:
        _close_shared_repository(repository)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_final_slot_advance_closes_schedule(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    final_schedule = _schedule(next_slot_ordinal=1)
    final_admission = _admission(
        turn_id="turn-2",
        player_id="p02",
        idempotency_key="turn-2:submit",
    )
    try:
        schedule, managed = _shared_admit_to_validating(
            repository,
            schedule=final_schedule,
            admission=final_admission,
        )
        closed = repository.finish_active_turn(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            None,
        )
        assert closed.status is SerialPublicScheduleStatus.CLOSED
        assert closed.next_slot_ordinal == len(closed.slots)
        assert closed.active_turn_id is None
        assert repository.load_active_serial_public_schedule("game-1") is None
    finally:
        _close_shared_repository(repository)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_twenty_duplicate_admissions_have_one_winner(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    database_name = "shared-concurrent.db"
    seed = _shared_repository(
        repository_kind,
        tmp_path,
        database_name=database_name,
    )
    seed.create_serial_public_schedule(_schedule())
    if isinstance(seed, SqliteGameRepository):
        seed.close()

    def attempt(index: int) -> tuple[str, type[Exception] | None]:
        repository = (
            seed
            if repository_kind == "memory"
            else SqliteGameRepository(str(tmp_path / database_name))
        )
        try:
            repository.admit_serial_public_turn(
                "schedule-1",
                0,
                _admission(
                    turn_id=f"shared-turn-{index}",
                    idempotency_key=f"shared-turn-{index}:submit",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - 验证稳定仓储错误分类
            return "error", type(exc)
        finally:
            if repository_kind == "sqlite":
                repository.close()
        return "success", None

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(attempt, range(20)))

    verifier = (
        seed
        if repository_kind == "memory"
        else SqliteGameRepository(str(tmp_path / database_name))
    )
    try:
        assert sum(result == "success" for result, _ in results) == 1
        failures = [error for result, error in results if result == "error"]
        assert len(failures) == 19
        assert set(failures) <= {ScheduleStateConflict, InvalidTurnAdmission}
        stored_turns = [
            verifier.load_managed_turn(f"shared-turn-{index}")
            for index in range(20)
        ]
        assert sum(turn is not None for turn in stored_turns) == 1
        active = verifier.load_serial_public_schedule("schedule-1")
        assert active is not None
        assert active.state_version == 1
        assert active.active_turn_id is not None
    finally:
        if repository_kind == "sqlite":
            verifier.close()


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_delete_game_cleans_schedule_and_turn_state(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    repository = _shared_repository(repository_kind, tmp_path)
    try:
        repository.create_serial_public_schedule(_schedule())
        repository.admit_serial_public_turn("schedule-1", 0, _admission())
        repository.delete_game("game-1")
        assert repository.load_serial_public_schedule("schedule-1") is None
        assert repository.load_managed_turn("turn-1") is None
        assert repository.load_active_serial_public_schedule("game-1") is None
        assert repository.list_open_serial_public_schedules() == ()
    finally:
        _close_shared_repository(repository)


@pytest.mark.parametrize("repository_kind", ("memory", "sqlite"))
def test_shared_forced_admission_failure_rolls_back_atomically(
    repository_kind: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    database_name = "shared-rollback.db"
    repository = _shared_repository(
        repository_kind,
        tmp_path,
        database_name=database_name,
    )
    created = repository.create_serial_public_schedule(_schedule())

    def fail_prepare(*args: object, **kwargs: object) -> object:
        raise AutonomousTurnTransactionError("forced shared failure")

    def fail_schedule_update(
        schedule: SerialPublicSchedule,
        expected_version: int,
    ) -> None:
        del schedule, expected_version
        raise ScheduleStateConflict("forced shared schedule conflict")

    def assert_admission_rolled_back(current: SharedTurnRepository) -> None:
        stored = current.load_serial_public_schedule(created.schedule_id)
        assert stored is not None
        assert _canonical_contract_bytes(stored) == _canonical_contract_bytes(created)
        assert current.load_managed_turn("turn-1") is None

    try:
        if repository_kind == "memory":
            monkeypatch.setattr(
                "werewolf_agent.storage.memory_store.prepare_serial_public_admission",
                fail_prepare,
            )
            expected_error = AutonomousTurnTransactionError
        else:
            assert isinstance(repository, SqliteGameRepository)
            monkeypatch.setattr(
                repository,
                "_update_schedule_unlocked",
                fail_schedule_update,
            )
            expected_error = ScheduleStateConflict

        with pytest.raises(expected_error):
            repository.admit_serial_public_turn(
                created.schedule_id,
                created.state_version,
                _admission(),
            )
        assert_admission_rolled_back(repository)

        if repository_kind == "sqlite":
            repository.close()
            reopened = SqliteGameRepository(str(tmp_path / database_name))
            try:
                assert_admission_rolled_back(reopened)
            finally:
                reopened.close()
    finally:
        _close_shared_repository(repository)


def _canonical_backend_trace(
    repository: SharedTurnRepository,
) -> tuple[bytes, ...]:
    trace: list[bytes] = []
    created = repository.create_serial_public_schedule(_schedule())
    trace.append(_canonical_contract_bytes(created))
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        created.state_version,
        _admission(),
    )
    active = repository.load_serial_public_schedule(created.schedule_id)
    assert active is not None
    trace.extend(
        (_canonical_contract_bytes(active), _canonical_contract_bytes(managed)),
    )
    for _current_status, next_status in _HOST_NONTERMINAL_EDGES:
        managed = repository.transition_active_turn(
            managed.turn.turn_id,
            managed.state_version,
            next_status,
        )
        trace.append(_canonical_contract_bytes(managed))
    managed = repository.transition_active_turn(
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.VALIDATING,
    )
    trace.append(_canonical_contract_bytes(managed))
    active = repository.load_serial_public_schedule(created.schedule_id)
    assert active is not None
    finished = repository.finish_active_turn(
        active.schedule_id,
        active.state_version,
        managed.turn.turn_id,
        managed.state_version,
        AgentTurnStatus.EXPIRED,
        TerminalDisposition.REPLACE,
        "canonical_replacement",
    )
    stored = repository.load_managed_turn(managed.turn.turn_id)
    assert stored is not None
    trace.extend(
        (_canonical_contract_bytes(finished), _canonical_contract_bytes(stored)),
    )
    return tuple(trace)


def test_memory_and_sqlite_contract_json_is_byte_equivalent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_repository_clocks(monkeypatch)
    memory = _shared_repository("memory", tmp_path)
    sqlite = _shared_repository(
        "sqlite",
        tmp_path,
        database_name="canonical-parity.db",
    )
    try:
        assert _canonical_backend_trace(memory) == _canonical_backend_trace(sqlite)
    finally:
        sqlite.close()
