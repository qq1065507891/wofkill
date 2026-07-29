# -*- coding: utf-8 -*-
"""
验证自主玩家 CommitTurn 仓储能力的共享辅助函数与基础错误。

作者: Project contributors
创建日期: 2026-07-29
"""

import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.player_agents.test_public_record_contracts import _record_payload
from tests.player_agents.test_transaction_contracts import _request
from werewolf_agent.core.models import GameEvent, GameState
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
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository


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
