# -*- coding: utf-8 -*-
"""
验证 PostgreSQL 串行公开调度、自主 CommitTurn、活动回合围栏与 durable dispatch 契约。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-31
"""

import copy
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.player_agents.test_transaction_contracts import _request
from tests.storage.test_active_turn_fence import (
    NOW as FENCE_NOW,
)
from tests.storage.test_active_turn_fence import (
    _active_turn as _fence_active_turn,
)
from tests.storage.test_active_turn_fence import (
    _attempt_for as _fence_attempt_for,
)
from tests.storage.test_autonomous_turns import (
    _admission,
    _admitted_validating,
    _schedule,
)
from werewolf_agent.player_agents.contracts.dispatch import (
    ActiveTurnDispatchFence,
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultOutcome,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.scheduling import (
    SerialPublicScheduleStatus,
    TerminalDisposition,
)
from werewolf_agent.player_agents.contracts.turns import AgentTurnStatus
from werewolf_agent.storage.active_turn_fence import (
    ActiveTurnFenceRejected,
    ActiveTurnFenceTransactionError,
)
from werewolf_agent.storage.autonomous_turns import (
    AutonomousTurnTransactionError,
    InvalidScheduleTransition,
    InvalidTurnAdmission,
    ScheduleStateConflict,
    TurnStateConflict,
)
from werewolf_agent.storage.durable_dispatch import (
    DispatchIdempotencyConflict,
    DispatchInvalidTransition,
    DispatchNotFound,
    DispatchRecoveryBlocked,
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


class _TurnCursor:
    def __init__(self, row=None, rows=None, *, rowcount: int = 1) -> None:
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return list(self._rows)


class _TurnConnection:
    """用内存快照模拟 PostgreSQL 事务，验证双记录原子发布。"""

    def __init__(
        self,
        *,
        schedules: dict[str, object] | None = None,
        turns: dict[str, object] | None = None,
        fail_on_schedule_update: bool = False,
        force_schedule_cas_miss: bool = False,
        force_turn_cas_miss: bool = False,
    ) -> None:
        self.games = {"game-1"}
        self.schedules = copy.deepcopy(schedules or {})
        self.turns = copy.deepcopy(turns or {})
        self.fail_on_schedule_update = fail_on_schedule_update
        self.force_schedule_cas_miss = force_schedule_cas_miss
        self.force_turn_cas_miss = force_turn_cas_miss
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.committed = 0
        self.rolled_back = 0
        self._snapshot: tuple[dict[str, object], dict[str, object]] | None = None

    @staticmethod
    def _payload(value: object) -> dict[str, object]:
        if isinstance(value, str):
            return json.loads(value)
        return copy.deepcopy(value)

    def _begin(self) -> None:
        if self._snapshot is None:
            self._snapshot = (
                copy.deepcopy(self.schedules),
                copy.deepcopy(self.turns),
            )

    def execute(self, sql: str, params=()):
        self._begin()
        normalized = " ".join(sql.split())
        bound = tuple(params)
        self.executed.append((normalized, bound))

        if "pg_advisory_xact_lock" in normalized:
            return _TurnCursor()
        if normalized.startswith("SELECT 1 FROM games"):
            return _TurnCursor((1,) if bound[0] in self.games else None)
        if normalized.startswith(
            "SELECT game_id FROM autonomous_serial_public_schedules"
        ):
            payload = self.schedules.get(str(bound[0]))
            row = None if payload is None else (self._payload(payload)["game_id"],)
            return _TurnCursor(row)
        if normalized.startswith(
            "SELECT schedule_id, game_id FROM autonomous_managed_turns"
        ):
            payload = self.turns.get(str(bound[0]))
            if payload is None:
                return _TurnCursor()
            data = self._payload(payload)
            return _TurnCursor((data["schedule_id"], data["turn"]["game_id"]))
        if normalized.startswith(
            "SELECT 1 FROM autonomous_managed_turns WHERE schedule_id = %s "
            "AND turn_json #>> '{turn,idempotency_key}' = %s"
        ):
            schedule_id, idempotency_key = bound
            for payload in self.turns.values():
                data = self._payload(payload)
                if (
                    data["schedule_id"] == schedule_id
                    and data["turn"]["idempotency_key"] == idempotency_key
                ):
                    return _TurnCursor((1,))
            return _TurnCursor()
        if normalized.startswith(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = %s"
        ):
            payload = self.schedules.get(str(bound[0]))
            return _TurnCursor(None if payload is None else (copy.deepcopy(payload),))
        if normalized.startswith(
            "SELECT turn_json FROM autonomous_managed_turns WHERE turn_id = %s"
        ):
            payload = self.turns.get(str(bound[0]))
            return _TurnCursor(None if payload is None else (copy.deepcopy(payload),))
        if normalized.startswith(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE game_id = %s AND status = %s"
        ):
            for payload in self.schedules.values():
                data = self._payload(payload)
                if data["game_id"] == bound[0] and data["status"] == bound[1]:
                    return _TurnCursor((copy.deepcopy(payload),))
            return _TurnCursor()
        if normalized.startswith(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE status = %s ORDER BY created_at, schedule_id"
        ):
            matching = [
                payload
                for payload in self.schedules.values()
                if self._payload(payload)["status"] == bound[0]
            ]
            matching.sort(
                key=lambda payload: (
                    self._payload(payload)["created_at"],
                    self._payload(payload)["schedule_id"],
                ),
            )
            return _TurnCursor(rows=[(copy.deepcopy(payload),) for payload in matching])
        if normalized.startswith(
            "INSERT INTO autonomous_serial_public_schedules"
        ):
            self.schedules[str(bound[0])] = copy.deepcopy(bound[7])
            return _TurnCursor()
        if normalized.startswith("INSERT INTO autonomous_managed_turns"):
            self.turns[str(bound[0])] = copy.deepcopy(bound[6])
            return _TurnCursor()
        if normalized.startswith("UPDATE autonomous_managed_turns SET"):
            if self.force_turn_cas_miss:
                return _TurnCursor(rowcount=0)
            turn_id = str(bound[9])
            current = self.turns.get(turn_id)
            if current is None or self._payload(current)["state_version"] != bound[10]:
                return _TurnCursor(rowcount=0)
            self.turns[turn_id] = copy.deepcopy(bound[5])
            return _TurnCursor()
        if normalized.startswith("UPDATE autonomous_serial_public_schedules SET"):
            if self.fail_on_schedule_update:
                raise RuntimeError("forced schedule update failure")
            if self.force_schedule_cas_miss:
                return _TurnCursor(rowcount=0)
            schedule_id = str(bound[9])
            current = self.schedules.get(schedule_id)
            if current is None or self._payload(current)["state_version"] != bound[10]:
                return _TurnCursor(rowcount=0)
            self.schedules[schedule_id] = copy.deepcopy(bound[6])
            return _TurnCursor()
        raise AssertionError(f"unexpected SQL: {normalized}")

    def commit(self) -> None:
        self.committed += 1
        self._snapshot = None

    def rollback(self) -> None:
        self.rolled_back += 1
        if self._snapshot is not None:
            self.schedules, self.turns = self._snapshot
        self._snapshot = None


class _InsertFailureConnection(_TurnConnection):
    """在托管回合写入时注入数据库异常，验证 admission 异常映射。"""

    def __init__(self, insert_error: BaseException, **kwargs) -> None:
        super().__init__(**kwargs)
        self._insert_error = insert_error

    def execute(self, sql: str, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("INSERT INTO autonomous_managed_turns"):
            self._begin()
            self.executed.append((normalized, tuple(params)))
            raise self._insert_error
        return super().execute(sql, params)


def _turn_repository(connection: _TurnConnection):
    repository = _repository_without_connection()
    repository._conn = connection
    repository._autonomous_schema_ready = True
    return repository


class _FenceConnection(_TurnConnection):
    """用可回滚的 attempt 快照模拟 PostgreSQL 围栏事务。"""

    def __init__(
        self,
        *,
        attempts: dict[str, object] | None = None,
        fail_on_attempt_insert: bool = False,
        attempt_insert_error: BaseException | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.attempts = copy.deepcopy(attempts or {})
        self.fail_on_attempt_insert = fail_on_attempt_insert
        self.attempt_insert_error = attempt_insert_error
        self._snapshot: tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ] | None = None

    @property
    def normalized_statements(self) -> list[str]:
        return [" ".join(statement.split()).lower() for statement, _ in self.executed]

    def _begin(self) -> None:
        if self._snapshot is None:
            self._snapshot = (
                copy.deepcopy(self.schedules),
                copy.deepcopy(self.turns),
                copy.deepcopy(self.attempts),
            )

    @classmethod
    def _attempt_row(cls, value: object) -> tuple[object, ...]:
        payload = cls._payload(value)
        return (
            payload["dispatch_id"],
            payload["game_id"],
            payload["turn_id"],
            payload["actor_id"],
            payload["operation_kind"],
            payload["executor_id"],
            payload["provider_idempotency_key"],
            payload["recovery_policy"],
            payload["request_hash"],
            payload["lease_hash"],
            payload["view_fingerprint"],
            payload["deadline"],
            payload["status"],
            payload["state_version"],
            payload.get("reason_code"),
            payload["created_at"],
            payload["updated_at"],
            payload.get("active_turn_fence"),
        )

    def execute(self, sql: str, params=()):
        normalized = " ".join(sql.split())
        bound = tuple(params)
        if normalized.startswith("SELECT ") and (
            "FROM autonomous_dispatch_attempts" in normalized
        ):
            self._begin()
            self.executed.append((normalized, bound))
            if "WHERE dispatch_id = %s" in normalized:
                value = self.attempts.get(str(bound[0]))
                return _TurnCursor(
                    None if value is None else self._attempt_row(value),
                )
            if "WHERE executor_id = %s" in normalized:
                for value in self.attempts.values():
                    payload = self._payload(value)
                    if (
                        payload["executor_id"],
                        payload["provider_idempotency_key"],
                    ) == bound:
                        return _TurnCursor((payload["dispatch_id"],))
                return _TurnCursor()
            if "WHERE game_id = %s AND turn_id = %s" in normalized:
                rows = [
                    self._attempt_row(value)
                    for value in self.attempts.values()
                    if (
                        self._payload(value)["game_id"],
                        self._payload(value)["turn_id"],
                    ) == bound
                ]
                return _TurnCursor(rows=rows)
            if "WHERE game_id = %s AND status IN" in normalized:
                statuses = set(bound[1:])
                for value in self.attempts.values():
                    payload = self._payload(value)
                    if (
                        payload["game_id"] == bound[0]
                        and payload["status"] in statuses
                    ):
                        return _TurnCursor((payload["dispatch_id"],))
                return _TurnCursor()
        if normalized.startswith("INSERT INTO autonomous_dispatch_attempts"):
            self._begin()
            self.executed.append((normalized, bound))
            if self.attempt_insert_error is not None:
                raise self.attempt_insert_error
            if self.fail_on_attempt_insert:
                raise RuntimeError("forced attempt insert failure")
            attempt = DispatchAttempt.model_validate(
                {
                    "dispatch_id": bound[0],
                    "game_id": bound[1],
                    "turn_id": bound[2],
                    "actor_id": bound[3],
                    "operation_kind": DispatchOperationKind(bound[4]),
                    "executor_id": bound[5],
                    "provider_idempotency_key": bound[6],
                    "recovery_policy": DispatchRecoveryPolicy(bound[7]),
                    "request_hash": bound[8],
                    "lease_hash": bound[9],
                    "view_fingerprint": bound[10],
                    "deadline": bound[11],
                    "status": DispatchStatus(bound[12]),
                    "state_version": bound[13],
                    "reason_code": bound[14],
                    "created_at": bound[15],
                    "updated_at": bound[16],
                    "active_turn_fence": (
                        None if bound[17] is None else json.loads(bound[17])
                    ),
                },
            )
            self.attempts[attempt.dispatch_id] = attempt.model_dump(mode="json")
            return _TurnCursor()
        if normalized.startswith(
            "UPDATE autonomous_dispatch_attempts SET status = %s, state_version = %s"
        ):
            self._begin()
            self.executed.append((normalized, bound))
            current = self.attempts.get(str(bound[4]))
            if current is None or self._payload(current)["state_version"] != bound[5]:
                return _TurnCursor(rowcount=0)
            updated = self._payload(current)
            updated.update(
                {
                    "status": bound[0],
                    "state_version": bound[1],
                    "reason_code": bound[2],
                    "updated_at": bound[3],
                },
            )
            self.attempts[str(bound[4])] = updated
            return _TurnCursor()
        return super().execute(sql, params)

    def rollback(self) -> None:
        self.rolled_back += 1
        if self._snapshot is not None:
            self.schedules, self.turns, self.attempts = self._snapshot
        self._snapshot = None


def _postgres_active_turn(
    *,
    fail_on_schedule_update: bool = False,
    fail_on_attempt_insert: bool = False,
    attempt_insert_error: BaseException | None = None,
) -> tuple[_FenceConnection, object, object]:
    schedule, managed = _fence_active_turn(turn_version=0)
    connection = _FenceConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        turns={managed.turn.turn_id: managed.model_dump(mode="json")},
        fail_on_schedule_update=fail_on_schedule_update,
        fail_on_attempt_insert=fail_on_attempt_insert,
        attempt_insert_error=attempt_insert_error,
    )
    return connection, schedule, managed


def test_postgres_fenced_create_commits_attempt_and_turn_version_once() -> None:
    connection, schedule, managed = _postgres_active_turn()
    repository = _turn_repository(connection)

    attempt = repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _fence_attempt_for(managed),
        FENCE_NOW,
    )

    assert repository.supports_active_turn_fence() is True
    assert connection.committed == 1
    assert connection.rolled_back == 0
    assert connection._payload(connection.turns[managed.turn.turn_id])["state_version"] == 1
    assert connection._payload(connection.attempts[attempt.dispatch_id])["active_turn_fence"] == {
        "schedule_id": schedule.schedule_id,
        "schedule_state_version": schedule.state_version,
        "turn_state_version": 1,
        "window_id": managed.turn.window.window_id,
        "window_version": managed.turn.window.version,
        "base_game_revision": managed.turn.revision.base_revision,
    }
    statements = connection.normalized_statements
    advisory_index = next(
        index
        for index, statement in enumerate(statements)
        if "pg_advisory_xact_lock" in statement
    )
    game_index = next(
        index
        for index, statement in enumerate(statements)
        if "from games" in statement and "for update" in statement
    )
    schedule_lock_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_serial_public_schedules" in statement
        and "for update" in statement
    )
    turn_lock_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_managed_turns" in statement and "for update" in statement
    )
    dispatch_lock_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_dispatch_attempts" in statement
        and "for update" in statement
    )
    assert advisory_index < game_index < schedule_lock_index < turn_lock_index
    assert turn_lock_index < dispatch_lock_index


def test_postgres_fenced_create_rolls_back_attempt_on_turn_cas_miss() -> None:
    connection, schedule, managed = _postgres_active_turn()
    connection.force_turn_cas_miss = True
    repository = _turn_repository(connection)

    with pytest.raises(TurnStateConflict):
        repository.create_active_turn_dispatch(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            _fence_attempt_for(managed),
            FENCE_NOW,
        )

    assert connection.attempts == {}
    assert connection._payload(connection.turns[managed.turn.turn_id]) == (
        managed.model_dump(mode="json")
    )
    assert connection.rolled_back == 1


def test_postgres_fenced_create_validates_context_before_idempotency_checks() -> None:
    connection, schedule, managed = _postgres_active_turn()
    existing = _fence_attempt_for(managed)
    connection.attempts[existing.dispatch_id] = existing.model_dump(mode="json")
    repository = _turn_repository(connection)

    with pytest.raises(ActiveTurnFenceRejected):
        repository.create_active_turn_dispatch(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            _fence_attempt_for(managed, actor_id="p02"),
            FENCE_NOW,
        )

    assert connection.attempts == {existing.dispatch_id: existing.model_dump(mode="json")}


class _ExactDispatchUniqueViolation(RuntimeError):
    sqlstate = "23505"

    def __init__(self, constraint_name: str) -> None:
        super().__init__("private postgres failure")
        self.diag = SimpleNamespace(constraint_name=constraint_name)


def test_postgres_fenced_create_maps_only_exact_dispatch_unique_constraint() -> None:
    known_error = _ExactDispatchUniqueViolation(
        "uq_autonomous_dispatch_executor_provider_key",
    )
    known_connection, schedule, managed = _postgres_active_turn(
        attempt_insert_error=known_error,
    )
    known_repository = _turn_repository(known_connection)

    with pytest.raises(DispatchIdempotencyConflict):
        known_repository.create_active_turn_dispatch(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            _fence_attempt_for(managed),
            FENCE_NOW,
        )

    unknown_error = _ExactDispatchUniqueViolation("unrelated_unique_constraint")
    unknown_connection, schedule, managed = _postgres_active_turn(
        attempt_insert_error=unknown_error,
    )
    unknown_repository = _turn_repository(unknown_connection)

    with pytest.raises(ActiveTurnFenceTransactionError) as exc_info:
        unknown_repository.create_active_turn_dispatch(
            schedule.schedule_id,
            schedule.state_version,
            managed.turn.turn_id,
            managed.state_version,
            _fence_attempt_for(managed),
            FENCE_NOW,
        )

    assert unknown_connection.rolled_back == 1
    assert exc_info.value.__cause__ is None
    assert "private postgres failure" not in str(exc_info.value)


def test_postgres_fenced_finish_rolls_back_cancel_and_turn_on_schedule_failure() -> None:
    connection, schedule, managed = _postgres_active_turn()
    repository = _turn_repository(connection)
    attempt = repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _fence_attempt_for(managed),
        FENCE_NOW,
    )
    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None
    connection.fail_on_schedule_update = True

    with pytest.raises(ActiveTurnFenceTransactionError):
        repository.finish_active_turn_fenced(
            schedule.schedule_id,
            schedule.state_version,
            current.turn.turn_id,
            current.state_version,
            AgentTurnStatus.CANCELLED,
            TerminalDisposition.ADVANCE,
            "operator_cancelled",
        )

    assert connection.rolled_back == 1
    assert connection._payload(connection.attempts[attempt.dispatch_id])["status"] == "pending"
    assert connection._payload(connection.turns[managed.turn.turn_id])["turn"]["status"] != "cancelled"


def test_postgres_fenced_completion_rejects_unresolved_attempt() -> None:
    schedule, managed = _fence_active_turn(
        status=AgentTurnStatus.VALIDATING,
        turn_version=0,
    )
    connection = _FenceConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        turns={managed.turn.turn_id: managed.model_dump(mode="json")},
    )
    repository = _turn_repository(connection)
    repository.create_active_turn_dispatch(
        schedule.schedule_id,
        schedule.state_version,
        managed.turn.turn_id,
        managed.state_version,
        _fence_attempt_for(managed),
        FENCE_NOW,
    )
    current = repository.load_managed_turn(managed.turn.turn_id)
    assert current is not None

    with pytest.raises(DispatchRecoveryBlocked):
        repository.finish_active_turn_fenced(
            schedule.schedule_id,
            schedule.state_version,
            current.turn.turn_id,
            current.state_version,
            AgentTurnStatus.COMMITTED,
            TerminalDisposition.ADVANCE,
            None,
        )

    assert connection._payload(connection.turns[managed.turn.turn_id])["turn"]["status"] == "validating"


def test_uninitialized_postgres_reports_autonomous_commit_unsupported() -> None:
    repository = _repository_without_connection()

    assert repository.supports_autonomous_commit() is False


def test_uninitialized_postgres_reports_autonomous_turns_unsupported() -> None:
    repository = _repository_without_connection()

    assert repository.supports_autonomous_turns() is False


def test_postgres_schema_contains_autonomous_turn_tables() -> None:
    repository = _repository_without_connection()
    connection = _clean_schema_connection()

    repository._ensure_schema_transaction(connection)

    sql = " ".join(call.args[0].lower() for call in connection.execute.call_args_list)
    assert "create table if not exists autonomous_serial_public_schedules" in sql
    assert "create table if not exists autonomous_managed_turns" in sql
    assert "schedule_json jsonb not null" in sql
    assert "turn_json jsonb not null" in sql
    assert "created_at timestamptz not null" in sql
    assert "updated_at timestamptz not null" in sql
    assert "state_version bigint not null" in sql
    assert "uq_open_serial_public_schedule" in sql
    assert "where status = 'open'" in sql
    assert "idx_managed_turn_schedule_status" in sql
    assert "uq_managed_turn_schedule_idempotency_key" in sql
    assert "turn_json #>> '{turn,idempotency_key}'" in sql


def test_postgres_rejects_duplicate_historical_idempotency_keys_during_initialization() -> None:
    from werewolf_agent.storage.postgres_store import PostgresSchemaMigrationError

    class FakeCursor:
        def __init__(self, rows=(), row=None) -> None:
            self._rows = list(rows)
            self._row = row

        def fetchall(self):
            return list(self._rows)

        def fetchone(self):
            return self._row

    class FakeConnection:
        def __init__(self) -> None:
            self.committed = 0
            self.rolled_back = 0
            self.executed: list[str] = []

        def execute(self, sql: str, params=()):
            del params
            normalized = " ".join(sql.split()).lower()
            self.executed.append(normalized)
            if (
                "from autonomous_managed_turns" in normalized
                and "group by schedule_id" in normalized
                and "having count(*) > 1" in normalized
            ):
                return FakeCursor(
                    rows=[
                        ("schedule-1", "historical-key", 2, ["turn-a", "turn-b"]),
                    ],
                )
            if "select to_regclass" in normalized:
                return FakeCursor(
                    row=("uq_events_game_seq", "uq_events_game_event_id"),
                )
            return FakeCursor()

        def commit(self) -> None:
            self.committed += 1

        def rollback(self) -> None:
            self.rolled_back += 1

    connection = FakeConnection()
    repository = _repository_without_connection()
    repository._conn = connection

    with pytest.raises(PostgresSchemaMigrationError) as exc_info:
        repository._ensure_schema()

    error_text = str(exc_info.value)
    assert "schedule-1" in error_text
    assert "historical-key" in error_text
    assert "turn-a" in str(exc_info.value)
    assert connection.rolled_back == 1
    assert connection.committed == 0
    assert repository._autonomous_schema_ready is False


def test_postgres_autonomous_turn_jsonb_decoders_accept_dicts_and_strings() -> None:
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    schedule = _schedule()
    _, managed = _admitted_validating()

    assert PostgresGameRepository._schedule_from_jsonb(
        schedule.model_dump(mode="json"),
    ) == schedule
    assert PostgresGameRepository._schedule_from_jsonb(
        schedule.model_dump_json(),
    ) == schedule
    assert PostgresGameRepository._managed_turn_from_jsonb(
        managed.model_dump(mode="json"),
    ) == managed
    assert PostgresGameRepository._managed_turn_from_jsonb(
        managed.model_dump_json(),
    ) == managed


def test_postgres_autonomous_turn_jsonb_is_canonical_utf8() -> None:
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    schedule = _schedule(
        window=_schedule().window.model_copy(update={"task_type": "公开发言"}),
    )

    payload = PostgresGameRepository._schedule_jsonb(schedule)

    assert payload == json.dumps(
        schedule.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "公开发言" in payload


def test_postgres_autonomous_turn_reads_are_ordered_and_defensive() -> None:
    later = _schedule(
        schedule_id="schedule-2",
        created_at=datetime(2026, 7, 30, 11, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 30, 11, tzinfo=timezone.utc),
    )
    first = _schedule()
    connection = _TurnConnection(
        schedules={
            later.schedule_id: later.model_dump(mode="json"),
            first.schedule_id: first.model_dump_json(),
        },
    )
    repository = _turn_repository(connection)

    loaded = repository.load_serial_public_schedule(first.schedule_id)
    active = repository.load_active_serial_public_schedule(first.game_id)
    open_schedules = repository.list_open_serial_public_schedules()

    assert loaded == first
    assert loaded is not first
    assert active in (first, later)
    assert tuple(item.schedule_id for item in open_schedules) == (
        "schedule-1",
        "schedule-2",
    )
    assert all(item is not first and item is not later for item in open_schedules)


@pytest.mark.parametrize(
    "schedule_updates",
    (
        {"status": SerialPublicScheduleStatus.CANCELLED},
        {"next_slot_ordinal": 1},
        {"active_turn_id": "existing-turn"},
        {"state_version": 7},
    ),
    ids=("terminal-status", "advanced-cursor", "active-turn", "advanced-version"),
)
def test_postgres_schedule_creation_rejects_non_fresh_initial_state(
    schedule_updates: dict[str, object],
) -> None:
    connection = _TurnConnection()
    repository = _turn_repository(connection)
    invalid_schedule = _schedule(**schedule_updates)

    with pytest.raises(InvalidScheduleTransition) as exc_info:
        repository.create_serial_public_schedule(invalid_schedule)

    insert_statements = [
        statement
        for statement, _ in connection.executed
        if statement.startswith("INSERT INTO autonomous_serial_public_schedules")
    ]
    assert exc_info.value.code == "invalid_schedule_transition"
    assert str(exc_info.value) == "invalid autonomous turn transition"
    assert insert_statements == []
    assert connection.schedules == {}
    assert connection.committed == 0


def test_postgres_autonomous_turn_lifecycle_persists_with_cas() -> None:
    connection = _TurnConnection()
    repository = _turn_repository(connection)

    assert repository.supports_autonomous_turns() is True
    created = repository.create_serial_public_schedule(_schedule())
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        created.state_version,
        _admission(),
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
    finished = repository.finish_active_turn(
        created.schedule_id,
        expected_schedule_version=1,
        turn_id=managed.turn.turn_id,
        expected_turn_version=managed.state_version,
        terminal_status=AgentTurnStatus.COMMITTED,
        disposition=TerminalDisposition.ADVANCE,
        reason_code=None,
    )

    persisted_turn = repository.load_managed_turn(managed.turn.turn_id)
    assert created == _schedule()
    assert finished.status is SerialPublicScheduleStatus.OPEN
    assert finished.next_slot_ordinal == 1
    assert finished.active_turn_id is None
    assert persisted_turn is not None
    assert persisted_turn.turn.status is AgentTurnStatus.COMMITTED
    assert persisted_turn.state_version == 5
    assert connection.committed == 7


def test_postgres_admission_preserves_schedule_version_conflict() -> None:
    schedule = _schedule()
    connection = _TurnConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
    )
    repository = _turn_repository(connection)

    with pytest.raises(ScheduleStateConflict):
        repository.admit_serial_public_turn(
            schedule.schedule_id,
            expected_schedule_version=99,
            admission=_admission(),
        )

    assert connection.rolled_back == 1
    assert connection.turns == {}


def test_postgres_admission_cas_miss_rolls_back_inserted_turn() -> None:
    schedule = _schedule()
    connection = _TurnConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        force_schedule_cas_miss=True,
    )
    repository = _turn_repository(connection)
    schedules_before = copy.deepcopy(connection.schedules)

    with pytest.raises(ScheduleStateConflict):
        repository.admit_serial_public_turn(
            schedule.schedule_id,
            expected_schedule_version=schedule.state_version,
            admission=_admission(),
        )

    assert connection.schedules == schedules_before
    assert connection.turns == {}
    assert connection.rolled_back == 1


def test_postgres_replacement_rejects_reused_idempotency_key() -> None:
    connection = _TurnConnection()
    repository = _turn_repository(connection)
    created = repository.create_serial_public_schedule(_schedule())
    managed = repository.admit_serial_public_turn(
        created.schedule_id,
        created.state_version,
        _admission(),
    )
    replaced = repository.finish_active_turn(
        created.schedule_id,
        expected_schedule_version=1,
        turn_id=managed.turn.turn_id,
        expected_turn_version=managed.state_version,
        terminal_status=AgentTurnStatus.CANCELLED,
        disposition=TerminalDisposition.REPLACE,
        reason_code="replace",
    )
    executed_before = len(connection.executed)
    commits_before = connection.committed

    with pytest.raises(InvalidTurnAdmission) as exc_info:
        repository.admit_serial_public_turn(
            created.schedule_id,
            replaced.state_version,
            _admission(turn_id="turn-2"),
        )

    assert exc_info.value.code == "invalid_turn_admission"
    assert str(exc_info.value) == "invalid autonomous turn admission"
    assert "INSERT INTO autonomous_managed_turns" not in " ".join(
        statement
        for statement, _ in connection.executed[executed_before:]
    )
    assert connection.committed == commits_before
    assert "turn-2" not in connection.turns
    assert connection._payload(connection.schedules["schedule-1"])["active_turn_id"] is None

    fresh = repository.admit_serial_public_turn(
        created.schedule_id,
        replaced.state_version,
        _admission(turn_id="turn-3", idempotency_key="turn-3:submit"),
    )
    assert fresh.turn.turn_id == "turn-3"


@pytest.mark.parametrize("code_attribute", ("sqlstate", "pgcode"))
def test_postgres_admission_maps_target_unique_violation(
    code_attribute: str,
) -> None:
    schedule = _schedule()
    insert_error = RuntimeError(
        "duplicate key value violates unique constraint "
        '"uq_managed_turn_schedule_idempotency_key"',
    )
    setattr(insert_error, code_attribute, "23505")
    insert_error.diag = SimpleNamespace(
        constraint_name="uq_managed_turn_schedule_idempotency_key",
    )
    connection = _InsertFailureConnection(
        insert_error,
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
    )
    repository = _turn_repository(connection)

    with pytest.raises(InvalidTurnAdmission) as exc_info:
        repository.admit_serial_public_turn(
            schedule.schedule_id,
            expected_schedule_version=schedule.state_version,
            admission=_admission(),
        )

    assert str(exc_info.value) == "invalid autonomous turn admission"
    assert connection.rolled_back == 1


def test_postgres_admission_does_not_map_constraint_text_without_unique_sqlstate() -> None:
    schedule = _schedule()
    insert_error = RuntimeError(
        "database connection failed near "
        '"uq_managed_turn_schedule_idempotency_key"',
    )
    insert_error.diag = SimpleNamespace(
        constraint_name="uq_managed_turn_schedule_idempotency_key",
    )
    connection = _InsertFailureConnection(
        insert_error,
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
    )
    repository = _turn_repository(connection)

    with pytest.raises(AutonomousTurnTransactionError) as exc_info:
        repository.admit_serial_public_turn(
            schedule.schedule_id,
            expected_schedule_version=schedule.state_version,
            admission=_admission(),
        )

    assert not isinstance(exc_info.value, InvalidTurnAdmission)
    assert connection.rolled_back == 1


def test_postgres_transition_cas_miss_preserves_managed_turn() -> None:
    schedule, managed = _admitted_validating()
    connection = _TurnConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        turns={managed.turn.turn_id: managed.model_dump(mode="json")},
        force_turn_cas_miss=True,
    )
    repository = _turn_repository(connection)
    turns_before = copy.deepcopy(connection.turns)

    with pytest.raises(TurnStateConflict):
        repository.transition_active_turn(
            managed.turn.turn_id,
            expected_turn_version=managed.state_version,
            next_status=AgentTurnStatus.REPAIRING,
        )

    assert connection.turns == turns_before
    assert connection.rolled_back == 1


def test_postgres_transition_locks_schedule_before_managed_turn() -> None:
    schedule, managed = _admitted_validating()
    connection = _TurnConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        turns={managed.turn.turn_id: managed.model_dump(mode="json")},
    )
    repository = _turn_repository(connection)

    repository.transition_active_turn(
        managed.turn.turn_id,
        expected_turn_version=managed.state_version,
        next_status=AgentTurnStatus.REPAIRING,
    )

    statements = [
        " ".join(statement.split()).lower()
        for statement, _ in connection.executed
    ]
    advisory_index = next(
        index
        for index, statement in enumerate(statements)
        if "pg_advisory_xact_lock" in statement
    )
    game_index = next(
        index
        for index, statement in enumerate(statements)
        if "from games" in statement and "for update" in statement
    )
    schedule_lock_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_serial_public_schedules" in statement
        and "for update" in statement
    )
    managed_turn_lock_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_managed_turns" in statement and "for update" in statement
    )

    assert advisory_index < game_index < schedule_lock_index < managed_turn_lock_index


def test_postgres_finish_rolls_back_both_records_when_second_write_fails() -> None:
    schedule, managed = _admitted_validating()
    connection = _TurnConnection(
        schedules={schedule.schedule_id: schedule.model_dump(mode="json")},
        turns={managed.turn.turn_id: managed.model_dump(mode="json")},
        fail_on_schedule_update=True,
    )
    repository = _turn_repository(connection)
    schedules_before = copy.deepcopy(connection.schedules)
    turns_before = copy.deepcopy(connection.turns)

    with pytest.raises(AutonomousTurnTransactionError):
        repository.finish_active_turn(
            schedule.schedule_id,
            expected_schedule_version=schedule.state_version,
            turn_id=managed.turn.turn_id,
            expected_turn_version=managed.state_version,
            terminal_status=AgentTurnStatus.COMMITTED,
            disposition=TerminalDisposition.ADVANCE,
            reason_code=None,
        )

    sql = " ".join(statement for statement, _ in connection.executed)
    assert connection.schedules == schedules_before
    assert connection.turns == turns_before
    assert connection.committed == 0
    assert connection.rolled_back == 1
    assert "pg_advisory_xact_lock" in sql
    assert "FOR UPDATE" in sql
    assert "state_version" in sql
    assert "%s" in sql


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


@pytest.mark.parametrize("status", tuple(DispatchStatus))
def test_postgres_dispatch_row_decodes_every_status_enum(status: DispatchStatus) -> None:
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), status.value, 0,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )

    parsed = PostgresGameRepository._dispatch_from_row(row)

    assert parsed.status is status


def test_postgres_dispatch_row_decodes_nullable_active_turn_fence() -> None:
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    fence = ActiveTurnDispatchFence(
        schedule_id="schedule-1",
        schedule_state_version=1,
        turn_state_version=2,
        window_id="speech-d1",
        window_version=1,
        base_game_revision=4,
    )
    row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "pending", 0,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        fence.model_dump(mode="json"),
    )

    parsed = PostgresGameRepository._dispatch_from_row(row)

    assert parsed.active_turn_fence == fence


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
    assert "idx_autonomous_dispatch_game_turn_created" in sql
    assert "on autonomous_dispatch_attempts (game_id, turn_id, created_at)" in sql


def test_postgres_schema_adds_nullable_active_turn_fence_jsonb() -> None:
    repository = _repository_without_connection()
    connection = _clean_schema_connection()

    repository._ensure_schema_transaction(connection)

    sql = " ".join(
        call.args[0].lower() for call in connection.execute.call_args_list
    )
    assert "active_turn_fence_json jsonb" in sql
    assert (
        "alter table autonomous_dispatch_attempts "
        "add column if not exists active_turn_fence_json jsonb"
    ) in sql


def test_postgres_dispatches_for_turn_binds_game_and_turn() -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "cancelled", 1,
        "expired", datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )
    repository._conn.execute.return_value.fetchall.return_value = [row]

    listed = repository.list_dispatches_for_turn("game-1", "turn-1")

    assert [item.dispatch_id for item in listed] == ["dispatch-1"]
    sql, params = repository._conn.execute.call_args.args
    normalized = " ".join(sql.split())
    assert "WHERE game_id = %s AND turn_id = %s ORDER BY created_at, dispatch_id" in normalized
    assert params == ("game-1", "turn-1")


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
    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = row
        elif "FROM games" in normalized:
            cursor.fetchone.return_value = (1,)
        elif normalized.startswith("UPDATE autonomous_dispatch_attempts"):
            cursor.rowcount = 1
        elif "pg_advisory_xact_lock" not in normalized:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    repository._conn.execute.side_effect = execute

    updated = repository.mark_dispatching("dispatch-1", 0)

    assert updated.status is DispatchStatus.DISPATCHING
    statements = " ".join(call.args[0].upper() for call in repository._conn.execute.call_args_list)
    assert "FROM AUTONOMOUS_DISPATCH_ATTEMPTS" in statements
    assert "FOR UPDATE" in statements
    assert "FROM GAMES" in statements
    repository._conn.commit.assert_called_once()


def test_postgres_fenced_dispatch_transition_locks_advisory_game_before_attempt() -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    fence = ActiveTurnDispatchFence(
        schedule_id="schedule-1",
        schedule_state_version=1,
        turn_state_version=1,
        window_id="speech-d1",
        window_version=1,
        base_game_revision=4,
    )
    row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "pending", 0,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        fence.model_dump(mode="json"),
    )
    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = row
        elif "FROM games" in normalized:
            cursor.fetchone.return_value = (1,)
        elif normalized.startswith("UPDATE autonomous_dispatch_attempts"):
            cursor.rowcount = 1
        elif "pg_advisory_xact_lock" not in normalized:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    repository._conn.execute.side_effect = execute

    repository.mark_dispatching("dispatch-1", 0)

    statements = [
        " ".join(call.args[0].split()).lower()
        for call in repository._conn.execute.call_args_list
    ]
    assert any("pg_advisory_xact_lock" in statement for statement in statements)
    advisory_index = next(
        index
        for index, statement in enumerate(statements)
        if "pg_advisory_xact_lock" in statement
    )
    locked_attempt_index = next(
        index
        for index, statement in enumerate(statements)
        if "from autonomous_dispatch_attempts" in statement and "for update" in statement
    )
    assert advisory_index < locked_attempt_index


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
    attempt_rows = [initial_row, initial_row, tuple(current_row)]

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_rows.pop(0)
        elif "FROM games" in normalized:
            cursor.fetchone.return_value = (1,)
        elif normalized.startswith("UPDATE autonomous_dispatch_attempts"):
            cursor.rowcount = 0
        elif "pg_advisory_xact_lock" not in normalized:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    repository._conn.execute.side_effect = execute

    with pytest.raises(expected_error):
        repository.mark_dispatching("dispatch-1", expected_version=0)

    repository._conn.rollback.assert_called_once()


class _UniqueViolation(RuntimeError):
    sqlstate = "23505"

    def __init__(self, constraint_name: str) -> None:
        super().__init__("unique constraint raced")
        self.diag = SimpleNamespace(
            constraint_name=constraint_name,
        )


@pytest.mark.parametrize(
    "unique_error",
    (
        _UniqueViolation("autonomous_dispatch_attempts_pkey"),
        _UniqueViolation("uq_autonomous_dispatch_executor_provider_key"),
    ),
)
def test_postgres_create_dispatch_maps_unique_race_to_idempotency_conflict(
    unique_error: Exception,
) -> None:
    repository = _repository_without_connection()
    repository._conn = MagicMock()
    repository._autonomous_schema_ready = True
    repository._conn.execute.side_effect = [
        MagicMock(),
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
        _UniqueViolation("autonomous_dispatch_results_pkey"),
        _UniqueViolation("autonomous_dispatch_results_dispatch_id_key"),
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
        elif "pg_advisory_xact_lock" in normalized:
            pass
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
        elif "pg_advisory_xact_lock" in normalized:
            pass
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


def test_postgres_backend_rejects_direct_dispatching_result() -> None:
    repository = _repository_without_connection()
    connection = MagicMock()
    repository._conn = connection
    repository._autonomous_schema_ready = True
    attempt_row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "dispatching", 1,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_row
        elif "pg_advisory_xact_lock" in normalized:
            pass
        elif normalized.startswith("SELECT 1 FROM games"):
            cursor.fetchone.return_value = (1,)
        elif "FROM autonomous_dispatch_results" in normalized:
            cursor.fetchone.return_value = None
        return cursor

    connection.execute.side_effect = execute

    with pytest.raises(DispatchInvalidTransition) as caught:
        repository.record_result(
            "dispatch-1",
            expected_version=1,
            result=_dispatch_result(),
        )

    assert caught.value.code == "dispatch_invalid_transition"
    connection.rollback.assert_called_once()


def test_postgres_record_result_replays_business_payload_with_result_fields() -> None:
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
    business_payload = {
        "result_id": "business-result-id",
        "dispatch_id": "business-dispatch-id",
        "request_hash": "business-request-hash",
        "lease_hash": "business-lease-hash",
        "result_hash": "business-result-hash",
        "result_kind": "business-result-kind",
        "outcome": "business-outcome",
        "payload": {"accepted": True},
        "recorded_at": "business-recorded-at",
    }
    result_row = (
        "result-1", "dispatch-1", "a" * 64, "b" * 64, "c" * 64,
        "model_response", "success", business_payload,
        datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_row
        elif "pg_advisory_xact_lock" in normalized:
            pass
        elif "SELECT 1 FROM games" in normalized:
            cursor.fetchone.return_value = (1,)
        elif "FROM autonomous_dispatch_results" in normalized:
            cursor.fetchone.return_value = result_row
        else:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    connection.execute.side_effect = execute
    result = _dispatch_result(payload=business_payload)

    assert (
        repository.record_result("dispatch-1", expected_version=2, result=result)
        is DispatchResultDisposition.REPLAYED
    )
    connection.rollback.assert_called_once()


@pytest.mark.parametrize(
    ("current_row_updates", "expected_error"),
    (
        (None, DispatchNotFound),
        ({"state_version": 3}, DispatchStateConflict),
        ({"status": "pending"}, DispatchInvalidTransition),
    ),
)
def test_postgres_record_result_reloads_after_cas_miss(
    current_row_updates: dict[str, object] | None,
    expected_error: type[Exception],
) -> None:
    repository = _repository_without_connection()
    connection = MagicMock()
    repository._conn = connection
    repository._autonomous_schema_ready = True
    initial_row = (
        "dispatch-1", "game-1", "turn-1", "p01", "model", "provider",
        "provider-key", "idempotent_lookup_or_reissue", "a" * 64, "b" * 64,
        "c" * 64, datetime(2026, 7, 29, 12, tzinfo=timezone.utc), "dispatched", 2,
        None, datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 11, tzinfo=timezone.utc),
    )
    current_row = None
    if current_row_updates is not None:
        current_values = list(initial_row)
        if "status" in current_row_updates:
            current_values[12] = current_row_updates["status"]
        if "state_version" in current_row_updates:
            current_values[13] = current_row_updates["state_version"]
        current_row = tuple(current_values)
    attempt_rows = [initial_row, initial_row, current_row]

    def execute(sql: str, _params=()):
        normalized = " ".join(sql.split())
        cursor = MagicMock()
        if "FROM autonomous_dispatch_attempts" in normalized:
            cursor.fetchone.return_value = attempt_rows.pop(0)
        elif "pg_advisory_xact_lock" in normalized:
            pass
        elif normalized.startswith("SELECT 1 FROM games"):
            cursor.fetchone.return_value = (1,)
        elif "FROM autonomous_dispatch_results" in normalized:
            cursor.fetchone.return_value = None
        elif normalized.startswith("INSERT INTO autonomous_dispatch_results"):
            pass
        elif normalized.startswith("UPDATE autonomous_dispatch_attempts"):
            cursor.rowcount = 0
        else:
            raise AssertionError(f"unexpected SQL: {normalized}")
        return cursor

    connection.execute.side_effect = execute

    with pytest.raises(expected_error):
        repository.record_result(
            "dispatch-1",
            expected_version=2,
            result=_dispatch_result(),
        )

    connection.rollback.assert_called_once()


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
