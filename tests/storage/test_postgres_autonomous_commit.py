# -*- coding: utf-8 -*-
"""
验证 PostgreSQL 自主 CommitTurn capability 和 schema 定义。

作者: Project contributors
创建日期: 2026-07-29
"""

import threading
from unittest.mock import MagicMock

from tests.player_agents.test_transaction_contracts import _request


def _repository_without_connection():
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    repository = PostgresGameRepository.__new__(PostgresGameRepository)
    repository._dsn = "postgresql://unused"
    repository._conn = None
    repository._lock = threading.Lock()
    return repository


def _clean_schema_connection() -> MagicMock:
    connection = MagicMock()

    def execute(sql: str, _params=None):
        cursor = MagicMock()
        normalized = " ".join(sql.split())
        if "HAVING COUNT(*) > 1" in normalized:
            cursor.fetchall.return_value = []
        elif "SELECT to_regclass" in normalized:
            cursor.fetchone.return_value = (
                "uq_events_game_seq",
                "uq_events_game_event_id",
            )
        return cursor

    connection.execute.side_effect = execute
    return connection


class _CommitCursor:
    def __init__(self, row=None, rows=None) -> None:
        self._row = row
        self._rows = rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return list(self._rows)


class _CommitConnection:
    def __init__(self) -> None:
        self.revision = None
        self.commits = {}
        self.events = []
        self.outbox = []
        self.audits = []
        self.committed = 0
        self.rolled_back = 0

    def execute(self, sql: str, params=()):
        normalized = " ".join(sql.split())
        if "pg_advisory_xact_lock" in normalized:
            return _CommitCursor()
        if normalized.startswith("SELECT 1 FROM games"):
            return _CommitCursor((1,))
        if normalized.startswith("SELECT request_hash, result_json"):
            key = tuple(params)
            result = self.commits.get(key)
            return _CommitCursor(result)
        if normalized.startswith("SELECT game_revision FROM autonomous_game_streams"):
            return _CommitCursor((self.revision,) if self.revision is not None else None)
        if normalized.startswith("SELECT COALESCE(MAX(seq), 0)"):
            return _CommitCursor((len(self.events),))
        if normalized.startswith("INSERT INTO autonomous_game_streams"):
            self.revision = int(params[1])
            return _CommitCursor()
        if normalized.startswith("INSERT INTO events"):
            self.events.append(params)
            return _CommitCursor()
        if normalized.startswith("INSERT INTO autonomous_audit_records"):
            self.audits.append(params)
            return _CommitCursor()
        if normalized.startswith("INSERT INTO autonomous_projection_outbox"):
            self.outbox.append(params)
            return _CommitCursor()
        if normalized.startswith("UPDATE autonomous_game_streams"):
            self.revision = int(params[0])
            return _CommitCursor()
        if normalized.startswith("INSERT INTO autonomous_turn_commits"):
            key = tuple(params[:3])
            self.commits[key] = (params[3], params[4])
            return _CommitCursor()
        return _CommitCursor()

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


def test_uninitialized_postgres_reports_autonomous_commit_unsupported() -> None:
    repository = _repository_without_connection()

    assert repository.supports_autonomous_commit() is False


def test_postgres_schema_contains_all_autonomous_tables() -> None:
    repository = _repository_without_connection()
    connection = _clean_schema_connection()

    repository._ensure_schema_transaction(connection)

    sql = " ".join(call.args[0].lower() for call in connection.execute.call_args_list)
    for table in (
        "autonomous_game_streams",
        "autonomous_turn_commits",
        "autonomous_public_records",
        "autonomous_audit_records",
        "autonomous_projection_outbox",
    ):
        assert f"create table if not exists {table}" in sql
    assert "jsonb" in sql
    assert "%s" not in sql


def test_postgres_commit_smoke_replays_one_atomic_result() -> None:
    repository = _repository_without_connection()
    repository._conn = _CommitConnection()
    repository._autonomous_schema_ready = True

    first = repository.commit_turn(_request())
    replay = repository.commit_turn(_request())

    assert first.committed_revision == 1
    assert replay.replayed is True
    assert len(repository._conn.events) == 1
    assert repository._conn.committed == 1
