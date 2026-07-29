# -*- coding: utf-8 -*-
"""
验证 PostgreSQL 自主 CommitTurn、durable dispatch schema 与结果幂等契约。

作者: Project contributors
创建日期: 2026-07-29
"""

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.player_agents.test_transaction_contracts import _request
from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultOutcome,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchIdempotencyConflict,
    DispatchInvalidTransition,
    DispatchResultConflict,
    DispatchStateConflict,
)


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


def _dispatch_attempt(**updates: object) -> DispatchAttempt:
    data: dict[str, object] = {
        "dispatch_id": "dispatch-1",
        "game_id": "game-1",
        "turn_id": "turn-1",
        "actor_id": "p01",
        "operation_kind": DispatchOperationKind.MODEL,
        "executor_id": "mock-provider",
        "provider_idempotency_key": "provider-key-1",
        "recovery_policy": DispatchRecoveryPolicy.IDEMPOTENT_LOOKUP_OR_REISSUE,
        "request_hash": "a" * 64,
        "lease_hash": "b" * 64,
        "view_fingerprint": "c" * 64,
        "deadline": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
        "created_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        "status": DispatchStatus.PENDING,
        "state_version": 0,
    }
    data.update(updates)
    return DispatchAttempt.model_validate(data)


def _dispatch_result(**updates: object) -> DispatchResultRecord:
    data: dict[str, object] = {
        "result_id": "result-1",
        "dispatch_id": "dispatch-1",
        "request_hash": "a" * 64,
        "lease_hash": "b" * 64,
        "result_hash": "c" * 64,
        "result_kind": "model_response",
        "outcome": DispatchResultOutcome.SUCCESS,
        "payload": {"accepted": True},
        "recorded_at": datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    }
    data.update(updates)
    return DispatchResultRecord.model_validate(data)


def test_postgres_schema_contains_durable_dispatch_tables_and_indexes() -> None:
    repository = _repository_without_connection()
    connection = _clean_schema_connection()

    repository._ensure_schema_transaction(connection)

    sql = " ".join(call.args[0].lower() for call in connection.execute.call_args_list)
    assert "create table if not exists autonomous_dispatch_attempts" in sql
    assert "create table if not exists autonomous_dispatch_results" in sql
    assert "timestamptz" in sql
    assert "state_version bigint" in sql
    assert "references games(game_id) on delete cascade" in sql
    assert "jsonb" in sql
    assert "unique index" in sql
    assert "executor_id, provider_idempotency_key" in sql
    assert "game_id, status, created_at" in sql
    assert "on autonomous_dispatch_attempts (game_id, status, created_at)" in sql
    assert "created_at, dispatch_id" not in sql


def test_postgres_durable_capability_never_opens_connection() -> None:
    repository = _repository_without_connection()

    assert repository.supports_durable_dispatch() is False
    assert repository._conn is None


def test_postgres_dispatch_transition_locks_game_and_attempt_rows() -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "pending", 0,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )
    repository._conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=row)),
        MagicMock(fetchone=MagicMock(return_value=(1,))),
        MagicMock(fetchone=MagicMock(return_value=row)),
        MagicMock(rowcount=1),
    ]

    updated = repository.mark_dispatching("dispatch-1", 0)

    assert updated.status is DispatchStatus.DISPATCHING
    statements = " ".join(call.args[0].upper() for call in repository._conn.execute.call_args_list)
    assert "FROM AUTONOMOUS_DISPATCH_ATTEMPTS" in statements
    assert "FOR UPDATE" in statements
    assert "FROM GAMES" in statements
    repository._conn.commit.assert_called_once()


@pytest.mark.parametrize(
    ("current_row_updates", "expected_error"),
    (
        (
            {"state_version": 1, "status": "dispatching"},
            DispatchStateConflict,
        ),
        (
            {"state_version": 0, "status": "dispatched"},
            DispatchInvalidTransition,
        ),
    ),
)
def test_postgres_dispatch_transition_reloads_after_cas_miss(
    current_row_updates: dict[str, object],
    expected_error: type[Exception],
) -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    initial_row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "pending", 0,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )
    current_row = list(initial_row)
    current_row[12] = current_row_updates["status"]
    current_row[13] = current_row_updates["state_version"]
    repository._conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=initial_row)),
        MagicMock(fetchone=MagicMock(return_value=(1,))),
        MagicMock(fetchone=MagicMock(return_value=initial_row)),
        MagicMock(rowcount=0),
        MagicMock(fetchone=MagicMock(return_value=tuple(current_row))),
    ]

    with pytest.raises(expected_error):
        repository.mark_dispatching("dispatch-1", expected_version=0)

    repository._conn.rollback.assert_called_once()


class _UniqueViolation(RuntimeError):
    sqlstate = "23505"


class _NamedConstraintViolation(RuntimeError):
    def __init__(self, constraint_name: str = "uq_autonomous_dispatch_executor_provider_key") -> None:
        super().__init__("unique constraint raced")
        self.diag = SimpleNamespace(
            constraint_name=constraint_name,
        )


@pytest.mark.parametrize(
    "unique_error",
    (_UniqueViolation("provider key raced"), _NamedConstraintViolation()),
)
def test_postgres_create_dispatch_maps_unique_race_to_idempotency_conflict(
    unique_error: Exception,
) -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    repository._conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=(1,))),
        MagicMock(fetchone=MagicMock(return_value=None)),
        MagicMock(fetchone=MagicMock(return_value=None)),
        MagicMock(fetchone=MagicMock(return_value=None)),
        unique_error,
    ]

    with pytest.raises(DispatchIdempotencyConflict):
        repository.create_dispatch(_dispatch_attempt())

    repository._conn.rollback.assert_called_once()


@pytest.mark.parametrize(
    "unique_error",
    (
        _UniqueViolation("result row raced"),
        _NamedConstraintViolation("autonomous_dispatch_results_pkey"),
        _NamedConstraintViolation("autonomous_dispatch_results_dispatch_id_key"),
    ),
)
def test_postgres_record_result_maps_unique_race_to_result_conflict(
    unique_error: Exception,
) -> None:
    repository = _repository_without_connection()
    connection = MagicMock()
    repository._conn = connection
    repository._autonomous_schema_ready = True
    attempt_row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "dispatched", 2,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_row
        elif normalized.startswith("SELECT 1 FROM games"):
            cursor.fetchone.return_value = (1,)
        elif (
            "WHERE dispatch_id = %s" in normalized
            or "WHERE result_id = %s" in normalized
        ):
            cursor.fetchone.return_value = None
        elif normalized.startswith("INSERT INTO autonomous_dispatch_results"):
            raise unique_error
        else:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    connection.execute.side_effect = execute

    with pytest.raises(DispatchResultConflict):
        repository.record_result(
            "dispatch-1",
            expected_version=2,
            result=_dispatch_result(),
        )

    connection.rollback.assert_called_once()


def test_postgres_result_from_single_json_column_uses_json_validator() -> None:
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    result = _dispatch_result(outcome=DispatchResultOutcome.FAILURE)
    parsed = PostgresGameRepository._result_from_row((result.model_dump_json(),))

    assert parsed == result
    assert parsed.outcome is DispatchResultOutcome.FAILURE


def test_postgres_record_result_decodes_psycopg_outcome_for_replay() -> None:
    repository = _repository_without_connection()
    connection = MagicMock()
    repository._conn = connection
    repository._autonomous_schema_ready = True
    attempt_row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "dispatched", 2,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )
    result_row = (
        "result-1", "dispatch-1", "a" * 64, "b" * 64, "c" * 64,
        "model_response", "success", {"accepted": True},
        datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_row
        elif "SELECT 1 FROM games" in normalized:
            cursor.fetchone.return_value = (1,)
        elif "FROM autonomous_dispatch_results" in normalized:
            cursor.fetchone.return_value = result_row
        else:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    connection.execute.side_effect = execute
    result = _dispatch_result()

    assert (
        repository.record_result("dispatch-1", expected_version=2, result=result)
        is DispatchResultDisposition.REPLAYED
    )

    with pytest.raises(DispatchResultConflict):
        repository.record_result(
            "dispatch-1",
            expected_version=2,
            result=_dispatch_result(
                result_hash="d" * 64,
                payload={"accepted": False},
            ),
        )

    assert connection.rollback.call_count == 2


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
