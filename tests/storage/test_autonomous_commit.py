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
from datetime import datetime, timezone

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
    DispatchRecoveryBlocked,
    DispatchResultConflict,
    DispatchStateConflict,
    DispatchTransactionError,
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
        json.loads(first),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first == second
    assert first == expected
    assert "终" in first
    assert "\\u7ec8" not in first


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
