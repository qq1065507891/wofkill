# -*- coding: utf-8 -*-
"""
功能描述：SQLite 游戏仓库，支持事件兼容读写、自主玩家原子 CommitTurn 与 durable dispatch 状态机。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-29
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Self

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.core.resolution_batches import (
    normalize_resolution_batch_fields,
    serialize_resolution_batch_fields,
)
from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchOperationKind,
    DispatchRecoveryPolicy,
    DispatchResultDisposition,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.transactions import (
    CommitResult,
    CommitTurnRequest,
    ProjectionOutboxRecord,
)
from werewolf_agent.runtime.event_metadata import (
    deserialize_game_event,
    serialize_game_event,
    serialize_legacy_event_payload,
)
from werewolf_agent.runtime.game_termination import (
    validate_game_aborted_append,
    validate_game_state_save,
)
from werewolf_agent.storage.autonomous_commit import (
    CommitTransactionError,
    IdempotencyConflictError,
    StaleCommitError,
    bind_public_record,
    build_commit_result,
    build_committed_event,
    request_hash,
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


def _serialize_game_state(gs: GameState) -> str:
    data = asdict(gs)
    data["deaths"] = [
        serialize_resolution_batch_fields(asdict(death)) for death in gs.deaths
    ]
    data["events"] = [serialize_game_event(event) for event in gs.events]
    return json.dumps(data, ensure_ascii=False)


def _deserialize_game_state(raw: str) -> GameState:
    data = json.loads(raw)
    players = {
        pid: PlayerState(**pdata) for pid, pdata in data.pop("players", {}).items()
    }
    deaths = [
        Death(**normalize_resolution_batch_fields(d))
        for d in data.pop("deaths", [])
    ]
    events = [deserialize_game_event(e) for e in data.pop("events", [])]
    return GameState(
        players=players,
        deaths=deaths,
        events=events,
        **data,
    )


_AUTONOMOUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS autonomous_game_streams (
    game_id TEXT PRIMARY KEY,
    game_revision INTEGER NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_turn_commits (
    game_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    committed_revision INTEGER NOT NULL,
    PRIMARY KEY (game_id, turn_id, idempotency_key),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_public_records (
    record_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    committed_revision INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_audit_records (
    audit_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    committed_revision INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_projection_outbox (
    outbox_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    committed_revision INTEGER NOT NULL,
    request_json TEXT NOT NULL,
    delivered_at TEXT,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_autonomous_audit_revision
    ON autonomous_audit_records (game_id, committed_revision);
CREATE INDEX IF NOT EXISTS idx_autonomous_outbox_revision
    ON autonomous_projection_outbox (game_id, committed_revision);
"""


_AUTONOMOUS_DISPATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS autonomous_dispatch_attempts (
    dispatch_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    operation_kind TEXT NOT NULL,
    executor_id TEXT NOT NULL,
    provider_idempotency_key TEXT NOT NULL,
    recovery_policy TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    lease_hash TEXT NOT NULL,
    view_fingerprint TEXT NOT NULL,
    deadline TEXT NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    reason_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_dispatch_results (
    result_id TEXT PRIMARY KEY,
    dispatch_id TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL,
    lease_hash TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    result_kind TEXT NOT NULL,
    outcome TEXT NOT NULL,
    result_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (dispatch_id)
        REFERENCES autonomous_dispatch_attempts(dispatch_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dispatch_executor_key
    ON autonomous_dispatch_attempts (executor_id, provider_idempotency_key);
CREATE INDEX IF NOT EXISTS idx_dispatch_game_status_created
    ON autonomous_dispatch_attempts (game_id, status, created_at);
"""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    event_json TEXT,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deaths (
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    death_json TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    record_json TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    game_id TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS config_snapshots (
    game_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rag_entries (
    entry_id TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS custom_configs (
    config_id TEXT PRIMARY KEY,
    config_type TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
    entry_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    entry_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflections_game ON reflections (game_id);
CREATE INDEX IF NOT EXISTS idx_reflections_player ON reflections (player_id);
""" + _AUTONOMOUS_SCHEMA


class SqliteSchemaMigrationError(RuntimeError):
    """SQLite schema 升级因现存数据冲突而无法安全完成。"""


def _ensure_event_sequence_integrity(conn: sqlite3.Connection) -> None:
    """拒绝历史重复序号，并为后续 legacy/autonomous 混写建立约束。"""
    duplicates = conn.execute(
        """
        SELECT game_id, seq, COUNT(*), GROUP_CONCAT(id)
        FROM events
        GROUP BY game_id, seq
        HAVING COUNT(*) > 1
        ORDER BY game_id, seq
        LIMIT 20
        """,
    ).fetchall()
    if duplicates:
        details = " | ".join(
            f"game_id={game_id}, seq={seq}, count={count}, rows={row_ids}"
            for game_id, seq, count, row_ids in duplicates
        )
        raise SqliteSchemaMigrationError(
            "SQLite event integrity migration blocked by duplicate event "
            f"sequences; resolve or quarantine these rows before retrying: {details}",
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_game_seq "
        "ON events (game_id, seq)",
    )



def _ensure_event_schema_v2(conn: sqlite3.Connection) -> None:
    """为旧 repository 数据库补齐 nullable event_json，并记录版本。"""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(events)").fetchall()
    }
    if "event_json" not in columns:
        conn.execute("ALTER TABLE events ADD COLUMN event_json TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version, description) VALUES (?, ?)",
        (2, "Add nullable GameEvent V2 JSON storage"),
    )


class SqliteGameRepository:
    """SQLite implementation of GameRepository."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._durable_dispatch_schema_ready = False
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        try:
            self._conn.executescript(_SCHEMA)
            _ensure_event_schema_v2(self._conn)
            self._conn.executescript(_AUTONOMOUS_DISPATCH_SCHEMA)
            self._durable_dispatch_schema_ready = True
            _ensure_event_sequence_integrity(self._conn)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- Game state --------------------------------------------------------

    def save_game(self, state: GameState) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM games WHERE game_id = ?",
                (state.game_id,),
            ).fetchone()
            existing = _deserialize_game_state(row[0]) if row is not None else None
            validate_game_state_save(existing, state)
            data = _serialize_game_state(state)
            self._conn.execute(
                """
                INSERT INTO games (game_id, state_json) VALUES (?, ?)
                ON CONFLICT(game_id) DO UPDATE SET state_json = excluded.state_json
                """,
                (state.game_id, data),
            )
            self._conn.commit()

    def load_game(self, game_id: str) -> GameState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            return _deserialize_game_state(row[0])

    # -- Events -----------------------------------------------------------

    def append_events(self, game_id: str, events: list[GameEvent]) -> None:
        with self._lock:
            state_row = self._conn.execute(
                "SELECT state_json FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            saved_state = (
                _deserialize_game_state(state_row[0])
                if state_row is not None
                else None
            )
            rows = self._conn.execute(
                "SELECT event_type, payload_json, event_json FROM events "
                "WHERE game_id = ? ORDER BY seq",
                (game_id,),
            ).fetchall()
            existing = [
                (
                    deserialize_game_event(json.loads(row[2]))
                    if row[2] is not None
                    else GameEvent(type=row[0], payload=json.loads(row[1]))
                )
                for row in rows
            ]
            validate_game_aborted_append(
                game_id, saved_state, existing, events,
            )
            current_max = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = ?",
                (game_id,),
            ).fetchone()[0]
            for i, event in enumerate(events):
                self._conn.execute(
                    "INSERT INTO events (game_id, seq, event_type, payload_json, event_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        game_id,
                        current_max + i + 1,
                        event.type,
                        json.dumps(serialize_legacy_event_payload(event), ensure_ascii=False),
                        json.dumps(serialize_game_event(event), ensure_ascii=False),
                    ),
                )
            self._conn.commit()

    def load_events(self, game_id: str) -> list[GameEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_type, payload_json, event_json FROM events WHERE game_id = ? ORDER BY seq",
                (game_id,),
            ).fetchall()
            return [
                (
                    deserialize_game_event(json.loads(r[2]))
                    if r[2] is not None
                    else GameEvent(type=r[0], payload=json.loads(r[1]))
                )
                for r in rows
            ]

    # -- Autonomous CommitTurn --------------------------------------------

    def supports_autonomous_commit(self) -> bool:
        """声明 SQLite 仓储已初始化自主提交表。"""
        return True

    def load_game_revision(self, game_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT game_revision FROM autonomous_game_streams WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is not None:
                return int(row[0])
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            return int(row[0] or 0)

    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_json FROM autonomous_projection_outbox "
                "WHERE game_id = ? ORDER BY committed_revision, outbox_id",
                (game_id,),
            ).fetchall()
            return [
                ProjectionOutboxRecord.model_validate_json(row[0])
                for row in rows
            ]

    def commit_turn(self, request: CommitTurnRequest) -> CommitResult:
        with self._lock:
            digest = request_hash(request)
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                game_row = self._conn.execute(
                    "SELECT 1 FROM games WHERE game_id = ?",
                    (request.game_id,),
                ).fetchone()
                if game_row is None:
                    raise CommitTransactionError(
                        f"game does not exist: {request.game_id}",
                    )
                existing_row = self._conn.execute(
                    "SELECT request_hash, result_json FROM autonomous_turn_commits "
                    "WHERE game_id = ? AND turn_id = ? AND idempotency_key = ?",
                    (request.game_id, request.turn_id, request.idempotency_key),
                ).fetchone()
                if existing_row is not None:
                    if existing_row[0] != digest:
                        raise IdempotencyConflictError(
                            "idempotency key conflicts with an existing proposal",
                        )
                    existing = CommitResult.model_validate_json(existing_row[1])
                    self._conn.rollback()
                    return existing.model_copy(update={"replayed": True})

                stream_row = self._conn.execute(
                    "SELECT game_revision FROM autonomous_game_streams "
                    "WHERE game_id = ?",
                    (request.game_id,),
                ).fetchone()
                if stream_row is None:
                    current = int(self._conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = ?",
                        (request.game_id,),
                    ).fetchone()[0] or 0)
                    self._conn.execute(
                        "INSERT INTO autonomous_game_streams (game_id, game_revision) VALUES (?, ?)",
                        (request.game_id, current),
                    )
                else:
                    current = int(stream_row[0])
                    event_head = int(self._conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = ?",
                        (request.game_id,),
                    ).fetchone()[0] or 0)
                    if current != event_head:
                        raise CommitTransactionError(
                            "autonomous stream head "
                            f"{current} does not match event head {event_head}",
                        )
                if request.base_game_revision != current:
                    raise StaleCommitError(
                        f"expected revision {current}, got {request.base_game_revision}",
                    )

                next_revision = current + 1
                event = build_committed_event(
                    request.game_id, request.event, next_revision,
                )
                record = bind_public_record(request.public_record, next_revision)
                result = build_commit_result(
                    request, digest, next_revision, event, record,
                )
                self._conn.execute(
                    "INSERT INTO events "
                    "(game_id, seq, event_type, payload_json, event_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        request.game_id,
                        next_revision,
                        event.type,
                        json.dumps(
                            serialize_legacy_event_payload(event),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            serialize_game_event(event),
                            ensure_ascii=False,
                        ),
                    ),
                )
                if record is not None:
                    self._conn.execute(
                        "INSERT INTO autonomous_public_records "
                        "(record_id, game_id, turn_id, committed_revision, record_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            record.record_id,
                            request.game_id,
                            request.turn_id,
                            next_revision,
                            record.model_dump_json(),
                        ),
                    )
                for audit in request.critical_audit_records:
                    self._conn.execute(
                        "INSERT INTO autonomous_audit_records "
                        "(audit_id, game_id, committed_revision, record_json) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            audit.audit_id,
                            request.game_id,
                            next_revision,
                            audit.model_dump_json(),
                        ),
                    )
                for outbox in request.projection_outbox_records:
                    self._conn.execute(
                        "INSERT INTO autonomous_projection_outbox "
                        "(outbox_id, game_id, committed_revision, request_json) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            outbox.outbox_id,
                            request.game_id,
                            next_revision,
                            outbox.model_dump_json(),
                        ),
                    )
                self._conn.execute(
                    "UPDATE autonomous_game_streams SET game_revision = ? "
                    "WHERE game_id = ?",
                    (next_revision, request.game_id),
                )
                self._conn.execute(
                    "INSERT INTO autonomous_turn_commits "
                    "(game_id, turn_id, idempotency_key, request_hash, result_json, committed_revision) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request.game_id,
                        request.turn_id,
                        request.idempotency_key,
                        digest,
                        result.model_dump_json(),
                        next_revision,
                    ),
                )
                self._conn.commit()
                return result
            except (StaleCommitError, IdempotencyConflictError):
                self._conn.rollback()
                raise
            except CommitTransactionError:
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise CommitTransactionError(
                    "autonomous CommitTurn transaction failed",
                ) from exc

    # -- Durable dispatch --------------------------------------------------

    def supports_durable_dispatch(self) -> bool:
        """仅在 durable dispatch 表初始化完成后声明 capability。"""
        with self._lock:
            return self._durable_dispatch_schema_ready

    @staticmethod
    def _dispatch_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _dispatch_from_row(row: sqlite3.Row | tuple[Any, ...]) -> DispatchAttempt:
        return DispatchAttempt.model_validate(
            {
                "dispatch_id": row[0],
                "game_id": row[1],
                "turn_id": row[2],
                "actor_id": row[3],
                "operation_kind": DispatchOperationKind(row[4]),
                "executor_id": row[5],
                "provider_idempotency_key": row[6],
                "recovery_policy": DispatchRecoveryPolicy(row[7]),
                "request_hash": row[8],
                "lease_hash": row[9],
                "view_fingerprint": row[10],
                "deadline": datetime.fromisoformat(row[11]),
                "status": DispatchStatus(row[12]),
                "state_version": row[13],
                "reason_code": row[14],
                "created_at": datetime.fromisoformat(row[15]),
                "updated_at": datetime.fromisoformat(row[16]),
            },
        )

    def _load_dispatch_unlocked(self, dispatch_id: str) -> DispatchAttempt | None:
        row = self._conn.execute(
            "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
            "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
            "lease_hash, view_fingerprint, deadline, status, state_version, "
            "reason_code, created_at, updated_at "
            "FROM autonomous_dispatch_attempts WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        if row is None:
            return None
        return self._dispatch_from_row(row)

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
        """在外部 I/O 前原子持久化一个 PENDING dispatch 意图。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if (
                    attempt.status is not DispatchStatus.PENDING
                    or attempt.state_version != 0
                ):
                    raise DispatchInvalidTransition(
                        "new dispatch must start in PENDING at version 0",
                    )
                if self._conn.execute(
                    "SELECT 1 FROM games WHERE game_id = ?", (attempt.game_id,)
                ).fetchone() is None:
                    raise DispatchTransactionError(
                        f"game does not exist: {attempt.game_id}",
                    )
                if self._conn.execute(
                    "SELECT 1 FROM autonomous_dispatch_attempts "
                    "WHERE dispatch_id = ?", (attempt.dispatch_id,),
                ).fetchone() is not None:
                    raise DispatchIdempotencyConflict(attempt.dispatch_id)
                key = (attempt.executor_id, attempt.provider_idempotency_key)
                if self._conn.execute(
                    "SELECT 1 FROM autonomous_dispatch_attempts "
                    "WHERE executor_id = ? AND provider_idempotency_key = ?",
                    key,
                ).fetchone() is not None:
                    raise DispatchIdempotencyConflict(
                        f"provider idempotency key already exists: {key}",
                    )
                if self._conn.execute(
                    "SELECT 1 FROM autonomous_dispatch_attempts "
                    "WHERE game_id = ? AND status IN (?, ?)",
                    (
                        attempt.game_id,
                        DispatchStatus.DISPATCHING.value,
                        DispatchStatus.DISPATCHED.value,
                    ),
                ).fetchone() is not None:
                    raise DispatchRecoveryBlocked(attempt.game_id)
                self._conn.execute(
                    "INSERT INTO autonomous_dispatch_attempts ("
                    "dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                    "executor_id, provider_idempotency_key, recovery_policy, "
                    "request_hash, lease_hash, view_fingerprint, deadline, status, "
                    "state_version, reason_code, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        attempt.dispatch_id,
                        attempt.game_id,
                        attempt.turn_id,
                        attempt.actor_id,
                        attempt.operation_kind.value,
                        attempt.executor_id,
                        attempt.provider_idempotency_key,
                        attempt.recovery_policy.value,
                        attempt.request_hash,
                        attempt.lease_hash,
                        attempt.view_fingerprint,
                        attempt.deadline.isoformat(),
                        attempt.status.value,
                        attempt.state_version,
                        attempt.reason_code,
                        attempt.created_at.isoformat(),
                        attempt.updated_at.isoformat(),
                    ),
                )
                stored = attempt.model_copy(deep=True)
                self._conn.commit()
                return stored.model_copy(deep=True)
            except (
                DispatchInvalidTransition,
                DispatchIdempotencyConflict,
                DispatchRecoveryBlocked,
                DispatchTransactionError,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise DispatchTransactionError(
                    "durable dispatch create transaction failed",
                ) from exc

    def _transition_dispatch(
        self,
        dispatch_id: str,
        expected_version: int,
        allowed_statuses: tuple[DispatchStatus, ...],
        target_status: DispatchStatus,
        reason_code: str | None = None,
    ) -> DispatchAttempt:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._load_dispatch_unlocked(dispatch_id)
            if existing is None:
                raise DispatchNotFound(dispatch_id)
            placeholders = ", ".join("?" for _ in allowed_statuses)
            updated_at = datetime.now(timezone.utc)
            cur = self._conn.execute(
                "UPDATE autonomous_dispatch_attempts SET status = ?, "
                "state_version = state_version + 1, reason_code = ?, updated_at = ? "
                f"WHERE dispatch_id = ? AND state_version = ? AND status IN ({placeholders})",
                (
                    target_status.value,
                    reason_code,
                    updated_at.isoformat(),
                    dispatch_id,
                    expected_version,
                    *(status.value for status in allowed_statuses),
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                current = self._load_dispatch_unlocked(dispatch_id)
                if current is None:
                    raise DispatchNotFound(dispatch_id)
                if current.state_version != expected_version:
                    raise DispatchStateConflict(dispatch_id)
                raise DispatchInvalidTransition(dispatch_id)
            updated = existing.model_copy(
                deep=True,
                update={
                    "status": target_status,
                    "state_version": existing.state_version + 1,
                    "reason_code": reason_code,
                    "updated_at": updated_at,
                },
            )
            self._conn.commit()
            return updated.model_copy(deep=True)
        except (
            DispatchNotFound,
            DispatchStateConflict,
            DispatchInvalidTransition,
            DispatchTransactionError,
        ):
            self._conn.rollback()
            raise
        except Exception as exc:
            self._conn.rollback()
            raise DispatchTransactionError(
                "durable dispatch transition transaction failed",
            ) from exc

    def mark_dispatching(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt:
        with self._lock:
            return self._transition_dispatch(
                dispatch_id,
                expected_version,
                (DispatchStatus.PENDING,),
                DispatchStatus.DISPATCHING,
            )

    def mark_dispatched(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt:
        with self._lock:
            return self._transition_dispatch(
                dispatch_id,
                expected_version,
                (DispatchStatus.DISPATCHING,),
                DispatchStatus.DISPATCHED,
            )

    def cancel_dispatch(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt:
        with self._lock:
            return self._transition_dispatch(
                dispatch_id,
                expected_version,
                (DispatchStatus.PENDING, DispatchStatus.DISPATCHING),
                DispatchStatus.CANCELLED,
                reason_code,
            )

    def mark_unknown_outcome(
        self,
        dispatch_id: str,
        expected_version: int,
        reason_code: str,
    ) -> DispatchAttempt:
        with self._lock:
            return self._transition_dispatch(
                dispatch_id,
                expected_version,
                (DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED),
                DispatchStatus.UNKNOWN_OUTCOME,
                reason_code,
            )

    def record_result(
        self,
        dispatch_id: str,
        expected_version: int,
        result: DispatchResultRecord,
    ) -> DispatchResultDisposition:
        """在一个事务中写入结果并推进 attempt 状态。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                attempt = self._load_dispatch_unlocked(dispatch_id)
                if attempt is None:
                    raise DispatchNotFound(dispatch_id)
                if attempt.state_version != expected_version:
                    raise DispatchStateConflict(dispatch_id)
                if result.dispatch_id != dispatch_id:
                    raise DispatchResultConflict(dispatch_id)
                if result.request_hash != attempt.request_hash:
                    raise DispatchResultConflict(dispatch_id)
                if result.lease_hash != attempt.lease_hash:
                    raise DispatchLeaseMismatch(dispatch_id)

                prior_row = self._conn.execute(
                    "SELECT result_json FROM autonomous_dispatch_results "
                    "WHERE dispatch_id = ?",
                    (dispatch_id,),
                ).fetchone()
                if prior_row is not None:
                    prior = DispatchResultRecord.model_validate_json(prior_row[0])
                    if prior == result:
                        self._conn.rollback()
                        return DispatchResultDisposition.REPLAYED
                    raise DispatchResultConflict(dispatch_id)
                if attempt.status in {
                    DispatchStatus.CANCELLED,
                    DispatchStatus.UNKNOWN_OUTCOME,
                }:
                    self._conn.rollback()
                    return DispatchResultDisposition.DISCARDED_LATE
                if attempt.status is not DispatchStatus.DISPATCHED:
                    raise DispatchInvalidTransition(dispatch_id)
                if self._conn.execute(
                    "SELECT 1 FROM autonomous_dispatch_results WHERE result_id = ?",
                    (result.result_id,),
                ).fetchone() is not None:
                    raise DispatchResultConflict(dispatch_id)

                self._conn.execute(
                    "INSERT INTO autonomous_dispatch_results ("
                    "result_id, dispatch_id, request_hash, lease_hash, result_hash, "
                    "result_kind, outcome, result_json, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.result_id,
                        dispatch_id,
                        result.request_hash,
                        result.lease_hash,
                        result.result_hash,
                        result.result_kind,
                        result.outcome.value,
                        json.dumps(
                            result.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        result.recorded_at.isoformat(),
                    ),
                )
                cur = self._conn.execute(
                    "UPDATE autonomous_dispatch_attempts SET status = ?, "
                    "state_version = state_version + 1, reason_code = NULL, updated_at = ? "
                    "WHERE dispatch_id = ? AND state_version = ? AND status = ?",
                    (
                        DispatchStatus.RESULT_RECORDED.value,
                        self._dispatch_now(),
                        dispatch_id,
                        expected_version,
                        DispatchStatus.DISPATCHED.value,
                    ),
                )
                if cur.rowcount != 1:
                    self._conn.rollback()
                    current = self._load_dispatch_unlocked(dispatch_id)
                    if current is None:
                        raise DispatchNotFound(dispatch_id)
                    if current.state_version != expected_version:
                        raise DispatchStateConflict(dispatch_id)
                    raise DispatchInvalidTransition(dispatch_id)
                self._conn.commit()
                return DispatchResultDisposition.RECORDED
            except (
                DispatchNotFound,
                DispatchStateConflict,
                DispatchInvalidTransition,
                DispatchLeaseMismatch,
                DispatchResultConflict,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise DispatchTransactionError(
                    "durable dispatch result transaction failed",
                ) from exc

    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None:
        with self._lock:
            attempt = self._load_dispatch_unlocked(dispatch_id)
            return attempt.model_copy(deep=True) if attempt is not None else None

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
                "lease_hash, view_fingerprint, deadline, status, state_version, "
                "reason_code, created_at, updated_at "
                "FROM autonomous_dispatch_attempts "
                "WHERE game_id = ? AND status IN (?, ?) "
                "ORDER BY created_at, dispatch_id",
                (
                    game_id,
                    DispatchStatus.DISPATCHING.value,
                    DispatchStatus.DISPATCHED.value,
                ),
            ).fetchall()
            return [self._dispatch_from_row(row).model_copy(deep=True) for row in rows]

    def assert_dispatch_allowed(self, game_id: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM autonomous_dispatch_attempts "
                "WHERE game_id = ? AND status IN (?, ?) LIMIT 1",
                (
                    game_id,
                    DispatchStatus.DISPATCHING.value,
                    DispatchStatus.DISPATCHED.value,
                ),
            ).fetchone()
            if row is not None:
                raise DispatchRecoveryBlocked(game_id)

    # -- Deaths -----------------------------------------------------------

    def save_deaths(self, game_id: str, deaths: list[Death]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM deaths WHERE game_id = ?", (game_id,))
            for death in deaths:
                self._conn.execute(
                    "INSERT INTO deaths (game_id, player_id, death_json) VALUES (?, ?, ?)",
                    (
                        game_id,
                        death.player_id,
                        json.dumps(
                            serialize_resolution_batch_fields(asdict(death)),
                            ensure_ascii=False,
                        ),
                    ),
                )
            self._conn.commit()

    def load_deaths(self, game_id: str) -> list[Death]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT death_json FROM deaths WHERE game_id = ?",
                (game_id,),
            ).fetchall()
            return [
                Death(**normalize_resolution_batch_fields(json.loads(r[0])))
                for r in rows
            ]

    # -- Model usage -------------------------------------------------------

    def save_model_usage(self, game_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO model_usage (game_id, record_json) VALUES (?, ?)",
                (game_id, json.dumps(record, ensure_ascii=False)),
            )
            self._conn.commit()

    def load_model_usage(self, game_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_json FROM model_usage WHERE game_id = ? ORDER BY id",
                (game_id,),
            ).fetchall()
            return [json.loads(r[0]) for r in rows]

    # -- Evaluation --------------------------------------------------------

    def save_evaluation(self, game_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO evaluations (game_id, result_json) VALUES (?, ?)",
                (game_id, json.dumps(result, ensure_ascii=False)),
            )
            self._conn.commit()

    def load_evaluation(self, game_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM evaluations WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    # -- Config snapshots --------------------------------------------------

    def save_config_snapshot(self, game_id: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO config_snapshots (game_id, config_json) VALUES (?, ?)",
                (game_id, json.dumps(config, ensure_ascii=False)),
            )
            self._conn.commit()

    def load_config_snapshot(self, game_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT config_json FROM config_snapshots WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    # -- Customization configs --------------------------------------------

    def save_custom_config(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO custom_configs
                    (config_id, config_type, record_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record["config_id"],
                    record["config_type"],
                    json.dumps(record, ensure_ascii=False),
                    record.get("created_at", ""),
                    record.get("updated_at", ""),
                ),
            )
            self._conn.commit()

    def load_custom_config(self, config_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record_json FROM custom_configs WHERE config_id = ?",
                (config_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def list_custom_configs(self, config_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if config_type is None:
                rows = self._conn.execute(
                    "SELECT record_json FROM custom_configs ORDER BY created_at, config_id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT record_json FROM custom_configs WHERE config_type = ? ORDER BY created_at, config_id",
                    (config_type,),
                ).fetchall()
            return [json.loads(row[0]) for row in rows]

    # -- List / Delete -----------------------------------------------------

    def list_games(self) -> list[GameState]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state_json FROM games"
            ).fetchall()
            return [_deserialize_game_state(r[0]) for r in rows]

    def delete_game(self, game_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
            self._conn.commit()

    # -- RAG entries ---------------------------------------------------------

    def save_rag_entries(self, entries: list[dict[str, Any]]) -> None:
        """Persist RAG entries (list of serialized RAGEntry dicts)."""
        with self._lock:
            for entry in entries:
                entry_id = entry.get("entry_id", "")
                self._conn.execute(
                    "INSERT OR REPLACE INTO rag_entries (entry_id, entry_json) VALUES (?, ?)",
                    (entry_id, json.dumps(entry, ensure_ascii=False)),
                )
            self._conn.commit()

    def load_rag_entries(self) -> list[dict[str, Any]]:
        """Load all persisted RAG entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_json FROM rag_entries"
            ).fetchall()
            return [json.loads(r[0]) for r in rows]

    def delete_rag_entry(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM rag_entries WHERE entry_id = ?", (entry_id,))
            self._conn.commit()

    # -- Memory snapshots ----------------------------------------------------

    def save_memory_snapshot(self, snapshot_id: str, data: dict[str, Any]) -> None:
        """Persist a MemoryStore snapshot."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_snapshots (snapshot_id, snapshot_json) VALUES (?, ?)",
                (snapshot_id, json.dumps(data, ensure_ascii=False)),
            )
            self._conn.commit()

    def load_memory_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        """Load a MemoryStore snapshot by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot_json FROM memory_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def list_memory_snapshots(self) -> list[dict[str, Any]]:
        """List all memory snapshot metadata."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT snapshot_id, created_at FROM memory_snapshots ORDER BY created_at DESC"
        ).fetchall()
        return [{"snapshot_id": r[0], "created_at": r[1]} for r in rows]

    def delete_memory_snapshot(self, snapshot_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_snapshots WHERE snapshot_id = ?", (snapshot_id,)
            )
            self._conn.commit()

    # -- Reflections --------------------------------------------------------

    def save_reflection(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entry_id = str(entry.get("entry_id", ""))
            self._conn.execute(
                """
                INSERT OR REPLACE INTO reflections
                    (entry_id, game_id, player_id, entry_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    entry_id,
                    str(entry.get("game_id", "")),
                    str(entry.get("player_id", "")),
                    json.dumps(entry, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def load_reflection(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT entry_json FROM reflections WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def load_reflections_by_game(self, game_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_json FROM reflections WHERE game_id = ?",
                (game_id,),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def load_reflections_by_player(self, player_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_json FROM reflections WHERE player_id = ?",
                (player_id,),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def load_all_reflections(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_json FROM reflections ORDER BY entry_id"
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def delete_reflection(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM reflections WHERE entry_id = ?",
                (entry_id,),
            )
            self._conn.commit()
