# -*- coding: utf-8 -*-
"""
验证自主玩家 CommitTurn 仓储能力的共享辅助函数与基础错误。

作者: Project contributors
创建日期: 2026-07-29
"""

import re

import pytest

from tests.player_agents.test_transaction_contracts import _request
from werewolf_agent.core.models import GameState
from werewolf_agent.player_agents.contracts.transactions import EventCandidate
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
