# -*- coding: utf-8 -*-
"""
验证自主玩家 CommitTurn 仓储能力的共享辅助函数与基础错误。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from tests.player_agents.test_public_record_contracts import _record_payload
from tests.player_agents.test_transaction_contracts import _request
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultOutcome,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.errors import ValidationErrorCode
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.transactions import (
    EventCandidate,
    ProjectionOutboxRecord,
)
from werewolf_agent.storage.autonomous_commit import (
    AutonomousCommitUnsupported,
    CommitTransactionError,
    IdempotencyConflictError,
    StaleCommitError,
    build_committed_event,
    request_hash,
    require_autonomous_commit_repository,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchIdempotencyConflict,
    DispatchInvalidTransition,
    DispatchLeaseMismatch,
    DispatchNotFound,
    DispatchReconciler,
    DispatchRecoveryBlocked,
    DispatchResultConflict,
    DispatchStateConflict,
    DispatchTransactionError,
    RecoveryResolution,
    RecoveryResolutionKind,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository

DISPATCH_HASH = "a" * 64
DISPATCH_NOW = datetime(2026, 7, 29, 11, tzinfo=timezone.utc)


def _dispatch_attempt(**updates: object) -> DispatchAttempt:
    data: dict[str, object] = {
        "dispatch_id": "dispatch-1",
        "game_id": "g1",
        "turn_id": "turn-1",
        "actor_id": "p01",
        "operation_kind": DispatchOperationKind.MODEL,
        "executor_id": "mock-provider",
        "provider_idempotency_key": "provider-key-1",
        "recovery_policy": DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        "request_hash": DISPATCH_HASH,
        "lease_hash": DISPATCH_HASH,
        "view_fingerprint": DISPATCH_HASH,
        "deadline": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
        "created_at": DISPATCH_NOW,
        "updated_at": DISPATCH_NOW,
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    data.update(updates)
    return DispatchAttempt.model_validate(data)


def _dispatch_result(**updates: object) -> DispatchResultRecord:
    data: dict[str, object] = {
        "result_id": "result-1",
        "dispatch_id": "dispatch-1",
        "request_hash": DISPATCH_HASH,
        "lease_hash": DISPATCH_HASH,
        "result_hash": DISPATCH_HASH,
        "result_kind": "model_response",
        "outcome": DispatchResultOutcome.SUCCESS,
        "payload": {"accepted": True},
        "recorded_at": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    }
    data.update(updates)
    return DispatchResultRecord.model_validate(data)


def _memory_dispatch_repository() -> InMemoryGameRepository:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))
    return repository


def test_request_hash_is_sha256_hex() -> None:
    digest = request_hash(_request())
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_capability_guard_rejects_legacy_repository() -> None:
    with pytest.raises(AutonomousCommitUnsupported):
        require_autonomous_commit_repository(object())


def test_storage_errors_expose_stable_contract_codes() -> None:
    assert (
        AutonomousCommitUnsupported.code
        is ValidationErrorCode.UNKNOWN_CAPABILITY
    )
    assert StaleCommitError.code is ValidationErrorCode.STALE_READ_SET
    assert (
        IdempotencyConflictError.code
        is ValidationErrorCode.IDEMPOTENCY_CONFLICT
    )


def test_build_event_assigns_authoritative_identity() -> None:
    event = build_committed_event(
        "g1",
        EventCandidate(type="speech_submitted"),
        3,
    )
    assert event.event_id == "g1:e000003"
    assert event.sequence_number == 3
    assert event.game_id == "g1"
    assert event.schema_version == "2"


def test_memory_commit_advances_revision_and_publishes_records() -> None:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))

    result = repository.commit_turn(_request())

    assert result.committed_revision == 1
    assert result.replayed is False
    assert repository.supports_autonomous_commit() is True
    assert repository.load_game_revision("g1") == 1
    assert [event.event_id for event in repository.load_events("g1")] == ["g1:e000001"]
    assert [item.outbox_id for item in repository.load_outbox("g1")] == ["outbox-1"]


def test_memory_duplicate_submit_replays_without_second_event() -> None:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))
    request = _request()

    first = repository.commit_turn(request)
    replay = repository.commit_turn(request)

    assert replay.replayed is True
    assert replay.committed_revision == first.committed_revision
    assert len(repository.load_events("g1")) == 1
    assert repository.load_game_revision("g1") == 1


def test_memory_stale_submit_and_idempotency_conflict_do_not_mutate() -> None:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))
    repository.commit_turn(_request())

    with pytest.raises(StaleCommitError):
        repository.commit_turn(_request(turn_id="turn-2"))
    with pytest.raises(IdempotencyConflictError):
        repository.commit_turn(_request(event_type="different_event"))

    assert repository.load_game_revision("g1") == 1
    assert len(repository.load_events("g1")) == 1


def test_sqlite_commit_persists_revision_event_and_outbox(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "autonomous.db"))
    repository.save_game(GameState(game_id="g1"))

    result = repository.commit_turn(_request())

    assert result.committed_revision == 1
    assert repository.load_game_revision("g1") == 1
    assert repository.load_events("g1")[0].event_id == "g1:e000001"
    assert repository.load_outbox("g1")[0].outbox_id == "outbox-1"
    repository.close()


def test_sqlite_conflicting_outbox_rolls_back_every_write(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "rollback.db"))
    repository.save_game(GameState(game_id="g1"))
    repository._conn.execute(
        "INSERT INTO autonomous_projection_outbox "
        "(outbox_id, game_id, committed_revision, request_json) VALUES (?, ?, ?, ?)",
        ("outbox-1", "g1", 99, "{}"),
    )
    repository._conn.commit()

    with pytest.raises(CommitTransactionError):
        repository.commit_turn(_request())

    assert repository.load_game_revision("g1") == 0
    assert repository.load_events("g1") == []
    assert repository._conn.execute(
        "SELECT COUNT(*) FROM autonomous_turn_commits WHERE game_id = ?",
        ("g1",),
    ).fetchone()[0] == 0
    assert repository._conn.execute(
        "SELECT COUNT(*) FROM autonomous_audit_records WHERE game_id = ?",
        ("g1",),
    ).fetchone()[0] == 0
    repository.close()


@pytest.fixture(params=("memory", "sqlite"))
def autonomous_repository(request, tmp_path):
    if request.param == "memory":
        repository = InMemoryGameRepository()
    else:
        repository = SqliteGameRepository(str(tmp_path / "shared.db"))
    repository.save_game(GameState(game_id="g1"))
    yield repository
    close = getattr(repository, "close", None)
    if callable(close):
        close()


def test_concurrent_duplicate_submissions_commit_once(autonomous_repository) -> None:
    request = _request()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(autonomous_repository.commit_turn, [request] * 50))

    assert sum(not result.replayed for result in results) == 1
    assert sum(result.replayed for result in results) == 49
    assert autonomous_repository.load_game_revision("g1") == 1
    assert len(autonomous_repository.load_events("g1")) == 1
    assert len(autonomous_repository.load_outbox("g1")) == 1


def test_public_record_is_bound_to_allocated_revision(autonomous_repository) -> None:
    payload = _record_payload()
    payload["game_id"] = "g1"
    payload["committed_revision"] = 99
    record = PublicSpeechRecord.model_validate(payload)

    result = autonomous_repository.commit_turn(_request(public_record=record))

    assert result.public_record_id == record.record_id
    assert result.committed_revision == 1
    if isinstance(autonomous_repository, InMemoryGameRepository):
        stored = autonomous_repository._autonomous_public_records[record.record_id]
        assert stored.committed_revision == 1
    else:
        raw = autonomous_repository._conn.execute(
            "SELECT record_json FROM autonomous_public_records WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()[0]
        assert PublicSpeechRecord.model_validate_json(raw).committed_revision == 1


def test_existing_outbox_id_leaves_no_partial_memory_commit() -> None:
    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))
    repository._autonomous_outbox["outbox-1"] = ProjectionOutboxRecord(
        outbox_id="outbox-1",
        kind="existing",
    )
    repository._autonomous_outbox_game_ids["outbox-1"] = "g1"

    with pytest.raises(CommitTransactionError):
        repository.commit_turn(_request())

    assert repository.load_game_revision("g1") == 0
    assert repository.load_events("g1") == []
    assert repository._autonomous_commits == {}


def test_memory_write_failure_rolls_back_and_wraps_cause() -> None:
    class FailingAuditDict(dict):
        def __setitem__(self, key, value):
            raise OSError("injected audit write failure")

    repository = InMemoryGameRepository()
    repository.save_game(GameState(game_id="g1"))
    repository._autonomous_audits = FailingAuditDict()

    with pytest.raises(CommitTransactionError) as caught:
        repository.commit_turn(_request())

    assert isinstance(caught.value.__cause__, OSError)
    assert repository.load_game_revision("g1") == 0
    assert repository.load_events("g1") == []
    assert repository._autonomous_commits == {}
    assert repository._autonomous_public_records == {}
    assert repository._autonomous_outbox == {}


def test_sqlite_rejects_duplicate_legacy_event_sequences(tmp_path) -> None:
    db_path = tmp_path / "duplicate-events.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        INSERT INTO games (game_id, state_json) VALUES ('g1', '{}');
        INSERT INTO events (game_id, seq, event_type, payload_json)
        VALUES ('g1', 1, 'legacy-a', '{}');
        INSERT INTO events (game_id, seq, event_type, payload_json)
        VALUES ('g1', 1, 'legacy-b', '{}');
        """,
    )
    conn.close()

    with pytest.raises(RuntimeError, match="duplicate"):
        SqliteGameRepository(str(db_path))


def test_existing_legacy_event_sets_initial_revision(autonomous_repository) -> None:
    autonomous_repository.append_events(
        "g1",
        [GameEvent(type="legacy_event")],
    )

    result = autonomous_repository.commit_turn(_request(base_revision=1))

    assert result.committed_revision == 2
    assert result.event_id == "g1:e000002"


def test_mixed_legacy_write_after_stream_creation_is_rejected(
    autonomous_repository,
) -> None:
    autonomous_repository.commit_turn(_request())
    autonomous_repository.append_events(
        "g1",
        [GameEvent(type="legacy_event_after_stream")],
    )

    with pytest.raises(CommitTransactionError, match="stream head"):
        autonomous_repository.commit_turn(
            _request(
                turn_id="turn-2",
                base_revision=1,
                audit_ids=("audit-2",),
                outbox_ids=("outbox-2",),
            ),
        )


def test_delete_game_removes_autonomous_state(autonomous_repository) -> None:
    autonomous_repository.commit_turn(_request())

    autonomous_repository.delete_game("g1")
    autonomous_repository.save_game(GameState(game_id="g1"))

    assert autonomous_repository.load_game_revision("g1") == 0
    assert autonomous_repository.load_outbox("g1") == []
    assert autonomous_repository.commit_turn(_request()).committed_revision == 1


def test_memory_dispatch_valid_transitions_increment_version_and_use_cas() -> None:
    repository = _memory_dispatch_repository()

    created = repository.create_dispatch(_dispatch_attempt())
    assert created.status is DispatchStatus.PENDING
    assert created.state_version == 0
    assert repository.load_dispatch("dispatch-1") is not created

    with pytest.raises(DispatchInvalidTransition):
        repository.mark_dispatched("dispatch-1", expected_version=0)
    with pytest.raises(DispatchStateConflict):
        repository.mark_dispatching("dispatch-1", expected_version=99)

    dispatching = repository.mark_dispatching("dispatch-1", expected_version=0)
    assert dispatching.status is DispatchStatus.DISPATCHING
    assert dispatching.state_version == 1
    dispatched = repository.mark_dispatched("dispatch-1", expected_version=1)
    assert dispatched.status is DispatchStatus.DISPATCHED
    assert dispatched.state_version == 2

    with pytest.raises(DispatchInvalidTransition):
        repository.mark_dispatching("dispatch-1", expected_version=2)


def test_memory_dispatch_create_enforces_idempotency_and_recovery_barrier() -> None:
    repository = _memory_dispatch_repository()
    repository.create_dispatch(_dispatch_attempt())

    with pytest.raises(DispatchIdempotencyConflict):
        repository.create_dispatch(_dispatch_attempt())
    with pytest.raises(DispatchIdempotencyConflict):
        repository.create_dispatch(
            _dispatch_attempt(
                dispatch_id="dispatch-2",
                provider_idempotency_key="provider-key-1",
            )
        )
    repository.assert_dispatch_allowed("g1")

    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    with pytest.raises(DispatchRecoveryBlocked):
        repository.assert_dispatch_allowed("g1")
    with pytest.raises(DispatchRecoveryBlocked):
        repository.create_dispatch(
            _dispatch_attempt(
                dispatch_id="dispatch-3",
                provider_idempotency_key="provider-key-3",
            )
        )
    assert [
        attempt.dispatch_id
        for attempt in repository.list_recoverable_dispatches("g1")
    ] == ["dispatch-1"]


def test_memory_dispatch_records_result_replays_and_rejects_conflicts() -> None:
    repository = _memory_dispatch_repository()
    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    result = _dispatch_result()

    assert (
        repository.record_result("dispatch-1", expected_version=2, result=result)
        is DispatchResultDisposition.RECORDED
    )
    stored = repository.load_dispatch("dispatch-1")
    assert stored is not None
    assert stored.status is DispatchStatus.RESULT_RECORDED
    assert stored.state_version == 3

    assert (
        repository.record_result("dispatch-1", expected_version=3, result=result)
        is DispatchResultDisposition.REPLAYED
    )
    assert repository.load_dispatch("dispatch-1").state_version == 3  # type: ignore[union-attr]

    with pytest.raises(DispatchResultConflict):
        repository.record_result(
            "dispatch-1",
            expected_version=3,
            result=_dispatch_result(result_hash="b" * 64),
        )
    with pytest.raises(DispatchResultConflict):
        repository.record_result(
            "dispatch-1",
            expected_version=3,
            result=_dispatch_result(payload={"accepted": False}),
        )
    with pytest.raises(DispatchLeaseMismatch):
        repository.record_result(
            "dispatch-1",
            expected_version=3,
            result=_dispatch_result(lease_hash="b" * 64),
        )


def test_memory_dispatch_discards_late_results_after_cancel_and_unknown() -> None:
    cancelled = _memory_dispatch_repository()
    cancelled.create_dispatch(_dispatch_attempt())
    cancelled.cancel_dispatch("dispatch-1", expected_version=0, reason_code="cancelled")
    assert (
        cancelled.record_result(
            "dispatch-1", expected_version=1, result=_dispatch_result()
        )
        is DispatchResultDisposition.DISCARDED_LATE
    )
    assert cancelled.load_dispatch("dispatch-1").status is DispatchStatus.CANCELLED  # type: ignore[union-attr]

    unknown = _memory_dispatch_repository()
    unknown.create_dispatch(_dispatch_attempt())
    unknown.mark_dispatching("dispatch-1", expected_version=0)
    unknown.mark_unknown_outcome(
        "dispatch-1", expected_version=1, reason_code="provider_timeout"
    )
    assert (
        unknown.record_result(
            "dispatch-1", expected_version=2, result=_dispatch_result()
        )
        is DispatchResultDisposition.DISCARDED_LATE
    )
    assert unknown.load_dispatch("dispatch-1").status is DispatchStatus.UNKNOWN_OUTCOME  # type: ignore[union-attr]


def test_memory_dispatch_rejects_unknown_ids_and_cleans_up_on_delete() -> None:
    repository = _memory_dispatch_repository()
    with pytest.raises(DispatchNotFound):
        repository.mark_dispatching("missing", expected_version=0)
    with pytest.raises(DispatchNotFound):
        repository.record_result("missing", expected_version=0, result=_dispatch_result())

    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.delete_game("g1")
    assert repository.load_dispatch("dispatch-1") is None
    assert repository.list_recoverable_dispatches("g1") == []
    assert repository._dispatch_attempts == {}
    assert repository._dispatch_results == {}
    assert repository._dispatch_key_index == {}


def test_sqlite_dispatch_persists_across_reopen_and_enforces_barrier(tmp_path) -> None:
    db_path = tmp_path / "dispatch-restart.db"
    repository = SqliteGameRepository(str(db_path))
    repository.save_game(GameState(game_id="g1"))
    assert repository.supports_durable_dispatch() is True
    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    repository.close()

    reopened = SqliteGameRepository(str(db_path))
    loaded = reopened.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.DISPATCHED
    assert loaded.state_version == 2
    with pytest.raises(DispatchRecoveryBlocked):
        reopened.assert_dispatch_allowed("g1")
    reopened.close()


def test_sqlite_dispatch_cas_late_result_replay_and_conflict(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "dispatch-cas.db"))
    repository.save_game(GameState(game_id="g1"))
    repository.create_dispatch(_dispatch_attempt())
    with pytest.raises(DispatchInvalidTransition):
        repository.mark_dispatched("dispatch-1", expected_version=0)
    with pytest.raises(DispatchStateConflict):
        repository.mark_dispatching("dispatch-1", expected_version=99)
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    result = _dispatch_result()
    assert repository.record_result("dispatch-1", expected_version=2, result=result) is DispatchResultDisposition.RECORDED
    assert repository.record_result("dispatch-1", expected_version=3, result=result) is DispatchResultDisposition.REPLAYED
    with pytest.raises(DispatchResultConflict):
        repository.record_result(
            "dispatch-1", expected_version=3, result=_dispatch_result(result_hash="b" * 64)
        )

    cancelled = _dispatch_attempt(dispatch_id="dispatch-2", provider_idempotency_key="provider-key-2")
    repository.create_dispatch(cancelled)
    repository.cancel_dispatch("dispatch-2", expected_version=0, reason_code="cancelled")
    assert repository.record_result("dispatch-2", expected_version=1, result=_dispatch_result(dispatch_id="dispatch-2", result_id="result-2")) is DispatchResultDisposition.DISCARDED_LATE
    repository.close()


def test_sqlite_dispatch_result_rollback_preserves_attempt(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "dispatch-rollback.db"))
    repository.save_game(GameState(game_id="g1"))
    repository.create_dispatch(_dispatch_attempt())
    repository.create_dispatch(
        _dispatch_attempt(
            dispatch_id="dispatch-2",
            provider_idempotency_key="provider-key-2",
        )
    )
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    repository._conn.execute(
        "INSERT INTO autonomous_dispatch_results "
        "(result_id, dispatch_id, request_hash, lease_hash, result_hash, result_kind, outcome, result_json, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("result-1", "dispatch-2", DISPATCH_HASH, DISPATCH_HASH, DISPATCH_HASH, "model_response", "success", "{}", DISPATCH_NOW.isoformat()),
    )
    repository._conn.commit()

    with pytest.raises(DispatchResultConflict):
        repository.record_result("dispatch-1", expected_version=2, result=_dispatch_result())
    loaded = repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.DISPATCHED
    assert loaded.state_version == 2
    repository.close()


def test_sqlite_dispatch_result_json_is_canonical_utf8(tmp_path) -> None:
    def record_json(payload: dict[str, object], db_name: str) -> str:
        repository = SqliteGameRepository(str(tmp_path / db_name))
        repository.save_game(GameState(game_id="g1"))
        repository.create_dispatch(_dispatch_attempt())
        repository.mark_dispatching("dispatch-1", expected_version=0)
        repository.mark_dispatched("dispatch-1", expected_version=1)
        repository.record_result(
            "dispatch-1",
            expected_version=2,
            result=_dispatch_result(payload=payload),
        )
        raw = repository._conn.execute(
            "SELECT result_json FROM autonomous_dispatch_results "
            "WHERE dispatch_id = ?",
            ("dispatch-1",),
        ).fetchone()[0]
        repository.close()
        return raw

    first = record_json({"z": "终", "a": {"y": 2, "x": 1}}, "canonical-a.db")
    second = record_json({"a": {"x": 1, "y": 2}, "z": "终"}, "canonical-b.db")
    expected = json.dumps(
        {"a": {"x": 1, "y": 2}, "z": "终"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first == second
    assert first == expected
    assert "终" in first
    assert "\\u7ec8" not in first


def test_sqlite_dispatch_result_replay_preserves_payload(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "dispatch-replay.db"))
    repository.save_game(GameState(game_id="g1"))
    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    result = _dispatch_result(payload={"终": {"accepted": True, "score": 2}})

    assert (
        repository.record_result("dispatch-1", expected_version=2, result=result)
        is DispatchResultDisposition.RECORDED
    )
    assert (
        repository.record_result("dispatch-1", expected_version=3, result=result)
        is DispatchResultDisposition.REPLAYED
    )
    row = repository._conn.execute(
        "SELECT result_id, dispatch_id, request_hash, lease_hash, result_hash, "
        "result_kind, outcome, result_json, recorded_at "
        "FROM autonomous_dispatch_results WHERE dispatch_id = ?",
        ("dispatch-1",),
    ).fetchone()
    assert row is not None
    parsed = repository._result_from_row(row)
    assert parsed.payload == result.payload
    repository.close()


def test_sqlite_dispatch_payload_with_full_fields_is_not_unwrapped(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "dispatch-business-payload.db"))
    repository.save_game(GameState(game_id="g1"))
    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    payload = {
        "result_id": "result-1",
        "dispatch_id": "dispatch-1",
        "request_hash": DISPATCH_HASH,
        "lease_hash": DISPATCH_HASH,
        "result_hash": DISPATCH_HASH,
        "result_kind": "model_response",
        "outcome": "success",
        "recorded_at": "2026-07-29T12:00:00Z",
        "payload": {"accepted": True, "score": 2},
    }
    result = _dispatch_result(payload=payload)
    assert (
        repository.record_result("dispatch-1", expected_version=2, result=result)
        is DispatchResultDisposition.RECORDED
    )
    row = repository._conn.execute(
        "SELECT result_id, dispatch_id, request_hash, lease_hash, result_hash, "
        "result_kind, outcome, result_json, recorded_at "
        "FROM autonomous_dispatch_results WHERE dispatch_id = ?",
        ("dispatch-1",),
    ).fetchone()
    assert row is not None

    assert json.loads(row[7]) == payload
    parsed = repository._result_from_row(row)
    assert parsed == result
    repository.close()


def test_memory_create_dispatch_requires_existing_game() -> None:
    repository = InMemoryGameRepository()

    with pytest.raises(DispatchTransactionError, match="game does not exist: g1"):
        repository.create_dispatch(_dispatch_attempt())

    assert repository._dispatch_attempts == {}


def test_sqlite_dispatch_write_failures_roll_back_atomically(tmp_path) -> None:
    repository = SqliteGameRepository(str(tmp_path / "dispatch-faults.db"))
    repository.save_game(GameState(game_id="g1"))
    repository._conn.executescript(
        """
        CREATE TRIGGER fail_dispatch_insert
        BEFORE INSERT ON autonomous_dispatch_attempts
        BEGIN SELECT RAISE(ABORT, 'injected dispatch insert failure'); END;
        """,
    )
    with pytest.raises(DispatchTransactionError):
        repository.create_dispatch(_dispatch_attempt())
    assert repository.load_dispatch("dispatch-1") is None

    repository._conn.execute("DROP TRIGGER fail_dispatch_insert")
    repository._conn.commit()
    repository.create_dispatch(_dispatch_attempt())
    repository._conn.executescript(
        """
        CREATE TRIGGER fail_dispatch_update
        BEFORE UPDATE ON autonomous_dispatch_attempts
        WHEN NEW.status = 'dispatching'
        BEGIN SELECT RAISE(ABORT, 'injected dispatch update failure'); END;
        """,
    )
    with pytest.raises(DispatchTransactionError):
        repository.mark_dispatching("dispatch-1", expected_version=0)
    pending = repository.load_dispatch("dispatch-1")
    assert pending is not None
    assert pending.status is DispatchStatus.PENDING
    assert pending.state_version == 0

    repository._conn.execute("DROP TRIGGER fail_dispatch_update")
    repository._conn.commit()
    repository.mark_dispatching("dispatch-1", expected_version=0)
    repository.mark_dispatched("dispatch-1", expected_version=1)
    repository._conn.executescript(
        """
        CREATE TRIGGER fail_dispatch_result
        BEFORE INSERT ON autonomous_dispatch_results
        BEGIN SELECT RAISE(ABORT, 'injected dispatch result failure'); END;
        """,
    )
    with pytest.raises(DispatchTransactionError):
        repository.record_result(
            "dispatch-1", expected_version=2, result=_dispatch_result()
        )
    dispatched = repository.load_dispatch("dispatch-1")
    assert dispatched is not None
    assert dispatched.status is DispatchStatus.DISPATCHED
    assert dispatched.state_version == 2
    assert repository._conn.execute(
        "SELECT COUNT(*) FROM autonomous_dispatch_results"
    ).fetchone()[0] == 0
    repository.close()


@pytest.fixture(params=("memory", "sqlite"))
def dispatch_repository(request, tmp_path):
    """为 durable dispatch 合同提供统一的内存/SQLite 后端矩阵。"""
    if request.param == "memory":
        repository = InMemoryGameRepository()
    else:
        repository = SqliteGameRepository(str(tmp_path / "dispatch-matrix.db"))
    repository.save_game(GameState(game_id="g1"))
    yield repository
    close = getattr(repository, "close", None)
    if callable(close):
        close()


def test_dispatch_state_machine_is_identical_across_backends(
    dispatch_repository,
) -> None:
    created = dispatch_repository.create_dispatch(_dispatch_attempt())
    assert created.status is DispatchStatus.PENDING
    assert created.state_version == 0

    dispatching = dispatch_repository.mark_dispatching("dispatch-1", 0)
    assert dispatching.status is DispatchStatus.DISPATCHING
    assert dispatching.state_version == 1
    dispatched = dispatch_repository.mark_dispatched("dispatch-1", 1)
    assert dispatched.status is DispatchStatus.DISPATCHED
    assert dispatched.state_version == 2

    result = _dispatch_result()
    assert (
        dispatch_repository.record_result("dispatch-1", 2, result)
        is DispatchResultDisposition.RECORDED
    )
    assert (
        dispatch_repository.record_result("dispatch-1", 3, result)
        is DispatchResultDisposition.REPLAYED
    )
    stored = dispatch_repository.load_dispatch("dispatch-1")
    assert stored is not None
    assert stored.status is DispatchStatus.RESULT_RECORDED
    assert stored.state_version == 3

    # 内存后端保留字典快照模拟崩溃后恢复；SQLite 通过 close/reopen 恢复。
    if isinstance(dispatch_repository, InMemoryGameRepository):
        attempts = dict(dispatch_repository._dispatch_attempts)
        results = dict(dispatch_repository._dispatch_results)
        recovered = InMemoryGameRepository()
        recovered.save_game(GameState(game_id="g1"))
        recovered._dispatch_attempts = attempts
        recovered._dispatch_results = results
        recovered._dispatch_key_index = dict(dispatch_repository._dispatch_key_index)
    else:
        db_path = dispatch_repository._db_path
        dispatch_repository.close()
        recovered = SqliteGameRepository(db_path)
    try:
        loaded = recovered.load_dispatch("dispatch-1")
        assert loaded is not None
        assert loaded.status is DispatchStatus.RESULT_RECORDED
        assert loaded.state_version == 3
    finally:
        close = getattr(recovered, "close", None)
        if callable(close):
            close()


def test_dispatching_cas_race_has_one_winner(dispatch_repository) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())

    def transition() -> object:
        try:
            return dispatch_repository.mark_dispatching("dispatch-1", 0)
        except Exception as exc:  # noqa: BLE001 - race outcomes are asserted below
            return exc

    with ThreadPoolExecutor(max_workers=10) as executor:
        outcomes = list(executor.map(lambda _: transition(), range(10)))

    successes = [item for item in outcomes if isinstance(item, DispatchAttempt)]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(successes) == 1
    assert len(failures) == 9
    assert all(isinstance(item, DispatchStateConflict) for item in failures)
    current = dispatch_repository.load_dispatch("dispatch-1")
    assert current is not None
    assert current.status is DispatchStatus.DISPATCHING
    assert current.state_version == 1


def test_memory_result_write_failure_rolls_back_attempt_and_result() -> None:
    class FailingResultDict(dict):
        def __setitem__(self, key, value):
            raise OSError("injected result write failure")

    repository = _memory_dispatch_repository()
    repository.create_dispatch(_dispatch_attempt())
    repository.mark_dispatching("dispatch-1", 0)
    repository.mark_dispatched("dispatch-1", 1)
    repository._dispatch_results = FailingResultDict()

    with pytest.raises(DispatchTransactionError) as caught:
        repository.record_result("dispatch-1", 2, _dispatch_result())

    assert isinstance(caught.value.__cause__, OSError)
    current = repository.load_dispatch("dispatch-1")
    assert current is not None
    assert current.status is DispatchStatus.DISPATCHED
    assert current.state_version == 2
    assert repository._dispatch_results == {}


def test_dispatch_listing_is_sorted_and_returns_defensive_copies(dispatch_repository) -> None:
    for dispatch_id in ("dispatch-z", "dispatch-a", "dispatch-m"):
        dispatch_repository.create_dispatch(
            _dispatch_attempt(
                dispatch_id=dispatch_id,
                provider_idempotency_key=f"provider-{dispatch_id}",
            ),
        )
    for dispatch_id in ("dispatch-z", "dispatch-a", "dispatch-m"):
        dispatch_repository.mark_dispatching(dispatch_id, 0)

    listed = dispatch_repository.list_recoverable_dispatches("g1")
    assert [item.dispatch_id for item in listed] == [
        "dispatch-a",
        "dispatch-m",
        "dispatch-z",
    ]
    listed[0] = listed[0].model_copy(update={"reason_code": "mutated"})
    reloaded = dispatch_repository.load_dispatch("dispatch-a")
    assert reloaded is not None
    assert reloaded.reason_code is None


def test_dispatches_for_turn_filters_game_and_turn_and_includes_all_statuses(
    dispatch_repository,
) -> None:
    dispatch_repository.save_game(GameState(game_id="g2"))

    def create(
        dispatch_id: str,
        *,
        game_id: str = "g1",
        turn_id: str = "turn-1",
        hour: int,
    ) -> None:
        dispatch_repository.create_dispatch(
            _dispatch_attempt(
                dispatch_id=dispatch_id,
                game_id=game_id,
                turn_id=turn_id,
                provider_idempotency_key=f"provider-{dispatch_id}",
                created_at=datetime(2026, 7, 29, hour, tzinfo=timezone.utc),
                updated_at=datetime(2026, 7, 29, hour, tzinfo=timezone.utc),
            )
        )

    create("dispatch-pending", hour=12)
    create("dispatch-cancelled", hour=9)
    dispatch_repository.cancel_dispatch("dispatch-cancelled", 0, "expired")
    create("dispatch-result", hour=9)
    dispatch_repository.mark_dispatching("dispatch-result", 0)
    dispatch_repository.mark_dispatched("dispatch-result", 1)
    dispatch_repository.record_result(
        "dispatch-result",
        2,
        _dispatch_result(dispatch_id="dispatch-result", result_id="result-dispatch"),
    )
    create("dispatch-unknown", hour=8)
    dispatch_repository.mark_dispatching("dispatch-unknown", 0)
    dispatch_repository.mark_unknown_outcome("dispatch-unknown", 1, "timeout")
    create("dispatch-other-turn", turn_id="turn-2", hour=7)
    create("dispatch-other-game", game_id="g2", hour=6)

    listed = dispatch_repository.list_dispatches_for_turn("g1", "turn-1")

    assert [item.dispatch_id for item in listed] == [
        "dispatch-unknown",
        "dispatch-cancelled",
        "dispatch-result",
        "dispatch-pending",
    ]
    assert {item.status for item in listed} == {
        DispatchStatus.PENDING,
        DispatchStatus.CANCELLED,
        DispatchStatus.RESULT_RECORDED,
        DispatchStatus.UNKNOWN_OUTCOME,
    }
    assert listed[0] is not dispatch_repository.load_dispatch("dispatch-unknown")


def test_dispatches_for_turn_orders_offsets_identically_in_memory_and_sqlite(
    tmp_path,
) -> None:
    memory = _memory_dispatch_repository()
    sqlite = SqliteGameRepository(str(tmp_path / "dispatch-offset-order.db"))
    sqlite.save_game(GameState(game_id="g1"))
    repositories = (memory, sqlite)
    try:
        for repository in repositories:
            repository.create_dispatch(
                _dispatch_attempt(
                    dispatch_id="dispatch-earlier",
                    provider_idempotency_key="provider-earlier",
                    created_at=datetime(
                        2026, 7, 29, 10,
                        tzinfo=timezone(timedelta(hours=2)),
                    ),
                    updated_at=datetime(
                        2026, 7, 29, 10,
                        tzinfo=timezone(timedelta(hours=2)),
                    ),
                )
            )
            repository.create_dispatch(
                _dispatch_attempt(
                    dispatch_id="dispatch-later",
                    provider_idempotency_key="provider-later",
                    created_at=datetime(
                        2026, 7, 29, 9, tzinfo=timezone.utc,
                    ),
                    updated_at=datetime(
                        2026, 7, 29, 9, tzinfo=timezone.utc,
                    ),
                )
            )

        # 第一条记录是 08:00Z（10:00+02:00），第二条记录是 09:00Z。
        expected = ["dispatch-earlier", "dispatch-later"]
        assert [
            item.dispatch_id
            for item in memory.list_dispatches_for_turn("g1", "turn-1")
        ] == expected
        assert [
            item.dispatch_id
            for item in sqlite.list_dispatches_for_turn("g1", "turn-1")
        ] == expected
    finally:
        sqlite.close()


def test_sqlite_legacy_dispatch_offsets_are_backfilled_before_ordering(
    tmp_path,
) -> None:
    db_path = tmp_path / "dispatch-legacy-offset-order.db"
    sqlite = SqliteGameRepository(str(db_path))
    sqlite.save_game(GameState(game_id="g1"))
    sqlite.close()

    legacy_rows = (
        (
            "dispatch-earlier",
            "provider-earlier",
            "2026-07-29T10:00:00+02:00",
            "2026-07-29T12:00:00+02:00",
        ),
        (
            "dispatch-later",
            "provider-later",
            "2026-07-29T09:00:00+00:00",
            "2026-07-29T11:00:00+00:00",
        ),
    )
    with sqlite3.connect(str(db_path)) as connection:
        for dispatch_id, provider_key, created_at, deadline in legacy_rows:
            connection.execute(
                "INSERT INTO autonomous_dispatch_attempts ("
                "dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                "executor_id, provider_idempotency_key, recovery_policy, "
                "request_hash, lease_hash, view_fingerprint, deadline, status, "
                "state_version, reason_code, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dispatch_id,
                    "g1",
                    "turn-1",
                    "p01",
                    DispatchOperationKind.MODEL.value,
                    "mock-provider",
                    provider_key,
                    DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE.value,
                    DISPATCH_HASH,
                    DISPATCH_HASH,
                    DISPATCH_HASH,
                    deadline,
                    DispatchStatus.PENDING.value,
                    0,
                    None,
                    created_at,
                    created_at,
                ),
            )

    memory = _memory_dispatch_repository()
    for dispatch_id, provider_key, created_at, deadline in legacy_rows:
        memory.create_dispatch(
            _dispatch_attempt(
                dispatch_id=dispatch_id,
                provider_idempotency_key=provider_key,
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(created_at),
                deadline=datetime.fromisoformat(deadline),
            )
        )

    reopened = SqliteGameRepository(str(db_path))
    try:
        expected = ["dispatch-earlier", "dispatch-later"]
        assert [
            item.dispatch_id
            for item in memory.list_dispatches_for_turn("g1", "turn-1")
        ] == expected
        assert [
            item.dispatch_id
            for item in reopened.list_dispatches_for_turn("g1", "turn-1")
        ] == expected
        assert reopened._conn.execute(
            "SELECT dispatch_id, deadline, created_at, updated_at "
            "FROM autonomous_dispatch_attempts ORDER BY dispatch_id"
        ).fetchall() == [
            (
                "dispatch-earlier",
                "2026-07-29T10:00:00+00:00",
                "2026-07-29T08:00:00+00:00",
                "2026-07-29T08:00:00+00:00",
            ),
            (
                "dispatch-later",
                "2026-07-29T11:00:00+00:00",
                "2026-07-29T09:00:00+00:00",
                "2026-07-29T09:00:00+00:00",
            ),
        ]
    finally:
        reopened.close()


def test_found_recovery_records_result_from_dispatching_state(dispatch_repository) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.mark_dispatching("dispatch-1", 0)

    class Resolver:
        def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
            assert attempt.dispatch_id == "dispatch-1"
            return RecoveryResolution(
                kind=RecoveryResolutionKind.FOUND,
                result=_dispatch_result(),
            )

    report = DispatchReconciler(dispatch_repository, resolver=Resolver()).reconcile_game("g1")

    assert report.resolved == 1
    assert report.barrier_open is True
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.RESULT_RECORDED
    assert loaded.state_version == 3


def test_backends_reject_direct_result_from_dispatching_state(dispatch_repository) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.mark_dispatching("dispatch-1", 0)

    with pytest.raises(DispatchInvalidTransition) as caught:
        dispatch_repository.record_result("dispatch-1", 1, _dispatch_result())

    assert caught.value.code == "dispatch_invalid_transition"
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.DISPATCHING
    assert loaded.state_version == 1


@pytest.mark.parametrize(
    ("resolution_kind", "start_dispatched", "expected_status", "expected_version"),
    (
        (RecoveryResolutionKind.REISSUED, False, DispatchStatus.DISPATCHED, 2),
        (RecoveryResolutionKind.REISSUED, True, DispatchStatus.DISPATCHED, 2),
        (RecoveryResolutionKind.PENDING, False, DispatchStatus.DISPATCHING, 1),
        (RecoveryResolutionKind.UNAVAILABLE, False, DispatchStatus.DISPATCHING, 1),
    ),
)
def test_recovery_fault_barriers_are_consistent_across_backends(
    dispatch_repository,
    resolution_kind,
    start_dispatched,
    expected_status,
    expected_version,
) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.mark_dispatching("dispatch-1", 0)
    if start_dispatched:
        dispatch_repository.mark_dispatched("dispatch-1", 1)

    class Resolver:
        def __init__(self) -> None:
            self.seen_keys: list[str] = []

        def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
            self.seen_keys.append(attempt.provider_idempotency_key)
            return RecoveryResolution(kind=resolution_kind)

    resolver = Resolver()
    report = DispatchReconciler(dispatch_repository, resolver=resolver).reconcile_game("g1")

    assert report.pending == 1
    assert report.barrier_open is False
    assert resolver.seen_keys == ["provider-key-1"]
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is expected_status
    assert loaded.state_version == expected_version
    assert loaded.provider_idempotency_key == "provider-key-1"


def test_at_most_once_unknown_never_calls_resolver_across_backends(
    dispatch_repository,
) -> None:
    dispatch_repository.create_dispatch(
        _dispatch_attempt(recovery_policy=DispatchRecoveryPolicy.AT_MOST_ONCE_UNKNOWN),
    )
    dispatch_repository.mark_dispatching("dispatch-1", 0)

    class NeverCalled:
        def resolve(self, attempt: DispatchAttempt) -> RecoveryResolution:
            raise AssertionError("resolver must not run")

    report = DispatchReconciler(
        dispatch_repository,
        resolver=NeverCalled(),
    ).reconcile_game("g1")

    assert report.unknown == 1
    assert report.budget_consumption_required is True
    assert report.barrier_open is True
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.UNKNOWN_OUTCOME
    assert loaded.reason_code == "provider_not_idempotent"


def test_cancel_unknown_late_results_are_discarded_across_backends(
    dispatch_repository,
) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.cancel_dispatch("dispatch-1", 0, "cancelled")
    assert (
        dispatch_repository.record_result("dispatch-1", 1, _dispatch_result())
        is DispatchResultDisposition.DISCARDED_LATE
    )
    cancelled = dispatch_repository.load_dispatch("dispatch-1")
    assert cancelled is not None
    assert cancelled.status is DispatchStatus.CANCELLED
    assert cancelled.state_version == 1

    dispatch_repository.create_dispatch(
        _dispatch_attempt(
            dispatch_id="dispatch-2",
            provider_idempotency_key="provider-key-2",
        ),
    )
    dispatch_repository.mark_dispatching("dispatch-2", 0)
    dispatch_repository.mark_unknown_outcome("dispatch-2", 1, "provider_timeout")
    assert (
        dispatch_repository.record_result(
            "dispatch-2",
            2,
            _dispatch_result(dispatch_id="dispatch-2", result_id="result-2"),
        )
        is DispatchResultDisposition.DISCARDED_LATE
    )
    unknown = dispatch_repository.load_dispatch("dispatch-2")
    assert unknown is not None
    assert unknown.status is DispatchStatus.UNKNOWN_OUTCOME
    assert unknown.state_version == 2


def test_request_and_lease_mismatch_leave_attempt_unchanged_across_backends(
    dispatch_repository,
) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.mark_dispatching("dispatch-1", 0)
    dispatch_repository.mark_dispatched("dispatch-1", 1)

    with pytest.raises(DispatchResultConflict) as request_error:
        dispatch_repository.record_result(
            "dispatch-1",
            2,
            _dispatch_result(request_hash="b" * 64),
        )
    with pytest.raises(DispatchLeaseMismatch) as lease_error:
        dispatch_repository.record_result(
            "dispatch-1",
            2,
            _dispatch_result(lease_hash="b" * 64),
        )

    assert request_error.value.code == "dispatch_result_conflict"
    assert lease_error.value.code == "dispatch_lease_mismatch"
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.status is DispatchStatus.DISPATCHED
    assert loaded.state_version == 2


def test_recovery_report_is_frozen() -> None:
    report = DispatchReconciler(
        InMemoryGameRepository(),
        resolver=object(),
    )
    # Accessing the public report through an empty scan keeps this assertion
    # coupled to the frozen dataclass rather than implementation internals.
    result = report.reconcile_game("missing-game")
    with pytest.raises((AttributeError, TypeError)):
        result.errors = 99  # type: ignore[misc]


def test_result_hash_conflict_is_stable_across_backends(dispatch_repository) -> None:
    dispatch_repository.create_dispatch(_dispatch_attempt())
    dispatch_repository.mark_dispatching("dispatch-1", 0)
    dispatch_repository.mark_dispatched("dispatch-1", 1)
    result = _dispatch_result()
    dispatch_repository.record_result("dispatch-1", 2, result)

    with pytest.raises(DispatchResultConflict) as caught:
        dispatch_repository.record_result(
            "dispatch-1",
            3,
            result.model_copy(update={"result_hash": "b" * 64}),
        )

    assert caught.value.code == "dispatch_result_conflict"
    loaded = dispatch_repository.load_dispatch("dispatch-1")
    assert loaded is not None
    assert loaded.state_version == 3
