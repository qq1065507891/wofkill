# -*- coding: utf-8 -*-
"""
功能描述：SQLite 游戏仓库，支持事件兼容读写、串行公开调度持久化、活动回合围栏、自主玩家原子 CommitTurn 与 durable dispatch 状态机。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-31
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
from werewolf_agent.player_agents.contracts._base import StrictFrozenModel
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
    ManagedAgentTurn,
    SerialPublicSchedule,
    SerialPublicScheduleStatus,
    TerminalDisposition,
    TurnAdmission,
)
from werewolf_agent.player_agents.contracts.transactions import (
    CommitResult,
    CommitTurnRequest,
    ProjectionOutboxRecord,
)
from werewolf_agent.player_agents.contracts.turns import AgentTurnStatus
from werewolf_agent.runtime.event_metadata import (
    deserialize_game_event,
    serialize_game_event,
    serialize_legacy_event_payload,
)
from werewolf_agent.runtime.game_termination import (
    validate_game_aborted_append,
    validate_game_state_save,
)
from werewolf_agent.storage.active_turn_fence import (
    ActiveTurnFenceRejected,
    ActiveTurnFenceTransactionError,
    prepare_active_turn_dispatch,
    prepare_fenced_active_finish,
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
from werewolf_agent.storage.autonomous_turns import (
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
    require_fresh_serial_public_schedule,
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
    active_turn_fence_json TEXT,
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
CREATE INDEX IF NOT EXISTS idx_dispatch_game_turn_created
    ON autonomous_dispatch_attempts (game_id, turn_id, created_at);
"""


_AUTONOMOUS_SCHEDULING_SCHEMA = """
CREATE TABLE IF NOT EXISTS autonomous_serial_public_schedules (
    schedule_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    status TEXT NOT NULL,
    next_slot_ordinal INTEGER NOT NULL,
    active_turn_id TEXT,
    state_version INTEGER NOT NULL,
    schedule_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS autonomous_managed_turns (
    turn_id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    status TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    turn_json TEXT NOT NULL,
    terminal_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (schedule_id)
        REFERENCES autonomous_serial_public_schedules(schedule_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_open_serial_public_schedule
    ON autonomous_serial_public_schedules (game_id)
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_managed_turn_schedule_status
    ON autonomous_managed_turns (schedule_id, status);
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


def _ensure_managed_turn_idempotency_integrity(
    conn: sqlite3.Connection,
) -> None:
    """预检历史重复 key 后创建托管回合唯一表达式索引。"""
    duplicates = conn.execute(
        """
        SELECT
            schedule_id,
            json_extract(turn_json, '$.turn.idempotency_key'),
            COUNT(*),
            GROUP_CONCAT(turn_id)
        FROM autonomous_managed_turns
        WHERE json_extract(turn_json, '$.turn.idempotency_key') IS NOT NULL
        GROUP BY schedule_id, json_extract(turn_json, '$.turn.idempotency_key')
        HAVING COUNT(*) > 1
        ORDER BY schedule_id, json_extract(turn_json, '$.turn.idempotency_key')
        LIMIT 20
        """,
    ).fetchall()
    if duplicates:
        details = " | ".join(
            f"schedule_id={schedule_id}, idempotency_key={idempotency_key}, "
            f"count={count}, rows={row_ids}"
            for schedule_id, idempotency_key, count, row_ids in duplicates
        )
        raise SqliteSchemaMigrationError(
            "SQLite managed turn idempotency migration blocked by duplicate "
            f"schedule keys; resolve or quarantine these rows before retrying: {details}",
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "uq_managed_turn_schedule_idempotency_key "
        "ON autonomous_managed_turns "
        "(schedule_id, json_extract(turn_json, '$.turn.idempotency_key'))",
    )


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


def _ensure_active_turn_fence_schema(conn: sqlite3.Connection) -> None:
    """为历史 durable dispatch 表补齐 nullable 活动回合围栏。"""

    columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(autonomous_dispatch_attempts)",
        ).fetchall()
    }
    if "active_turn_fence_json" not in columns:
        conn.execute(
            "ALTER TABLE autonomous_dispatch_attempts "
            "ADD COLUMN active_turn_fence_json TEXT",
        )


class SqliteGameRepository:
    """SQLite implementation of GameRepository."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._durable_dispatch_schema_ready = False
        self._autonomous_turn_schema_ready = False
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        try:
            self._conn.executescript(_SCHEMA)
            _ensure_event_schema_v2(self._conn)
            self._conn.executescript(_AUTONOMOUS_DISPATCH_SCHEMA)
            self._conn.executescript(_AUTONOMOUS_SCHEDULING_SCHEMA)
            _ensure_active_turn_fence_schema(self._conn)
            _ensure_managed_turn_idempotency_integrity(self._conn)
            self._normalize_dispatch_timestamps()
            self._durable_dispatch_schema_ready = True
            self._autonomous_turn_schema_ready = True
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

    # -- Autonomous serial-public turns -----------------------------------

    def supports_autonomous_turns(self) -> bool:
        """仅在串行公开回合表初始化成功后声明 capability。"""
        with self._lock:
            return self._autonomous_turn_schema_ready

    @staticmethod
    def _canonical_contract_json(value: StrictFrozenModel) -> str:
        """将严格契约模型编码为稳定、紧凑且 UTF-8 的 JSON。"""
        return json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _schedule_from_row(row: sqlite3.Row | tuple[Any, ...]) -> SerialPublicSchedule:
        payload = row[0]
        return SerialPublicSchedule.model_validate_json(payload)

    @staticmethod
    def _managed_turn_from_row(row: sqlite3.Row | tuple[Any, ...]) -> ManagedAgentTurn:
        payload = row[0]
        return ManagedAgentTurn.model_validate_json(payload)

    def _load_schedule_unlocked(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        row = self._conn.execute(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = ?",
            (schedule_id,),
        ).fetchone()
        return None if row is None else self._schedule_from_row(row)

    def _load_managed_turn_unlocked(
        self,
        turn_id: str,
    ) -> ManagedAgentTurn | None:
        row = self._conn.execute(
            "SELECT turn_json FROM autonomous_managed_turns "
            "WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        return None if row is None else self._managed_turn_from_row(row)

    def _update_schedule_unlocked(
        self,
        schedule: SerialPublicSchedule,
        expected_version: int,
    ) -> None:
        result = self._conn.execute(
            "UPDATE autonomous_serial_public_schedules SET "
            "game_id = ?, window_id = ?, status = ?, next_slot_ordinal = ?, "
            "active_turn_id = ?, state_version = ?, schedule_json = ?, "
            "created_at = ?, updated_at = ? "
            "WHERE schedule_id = ? AND state_version = ?",
            (
                schedule.game_id,
                schedule.window.window_id,
                schedule.status.value,
                schedule.next_slot_ordinal,
                schedule.active_turn_id,
                schedule.state_version,
                self._canonical_contract_json(schedule),
                self._utc_timestamp(schedule.created_at),
                self._utc_timestamp(schedule.updated_at),
                schedule.schedule_id,
                expected_version,
            ),
        )
        if result.rowcount != 1:
            raise ScheduleStateConflict("schedule state version conflict")

    def _update_managed_turn_unlocked(
        self,
        managed: ManagedAgentTurn,
        expected_version: int,
    ) -> None:
        result = self._conn.execute(
            "UPDATE autonomous_managed_turns SET "
            "schedule_id = ?, game_id = ?, player_id = ?, status = ?, "
            "state_version = ?, turn_json = ?, terminal_reason = ?, "
            "created_at = ?, updated_at = ? "
            "WHERE turn_id = ? AND state_version = ?",
            (
                managed.schedule_id,
                managed.turn.game_id,
                managed.turn.player_id,
                managed.turn.status.value,
                managed.state_version,
                self._canonical_contract_json(managed),
                managed.terminal_reason,
                self._utc_timestamp(managed.created_at),
                self._utc_timestamp(managed.updated_at),
                managed.turn.turn_id,
                expected_version,
            ),
        )
        if result.rowcount != 1:
            raise TurnStateConflict("managed turn state version conflict")

    def create_serial_public_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule:
        """创建一个公开调度并通过数据库唯一索引建立游戏级互斥。"""
        require_fresh_serial_public_schedule(schedule)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._conn.execute(
                    "SELECT 1 FROM games WHERE game_id = ?",
                    (schedule.game_id,),
                ).fetchone() is None:
                    raise AutonomousTurnTransactionError(
                        "game is required before creating a schedule",
                    )
                if self._load_schedule_unlocked(schedule.schedule_id) is not None:
                    raise AutonomousTurnTransactionError("schedule already exists")
                self._conn.execute(
                    "INSERT INTO autonomous_serial_public_schedules "
                    "(schedule_id, game_id, window_id, status, next_slot_ordinal, "
                    "active_turn_id, state_version, schedule_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        schedule.schedule_id,
                        schedule.game_id,
                        schedule.window.window_id,
                        schedule.status.value,
                        schedule.next_slot_ordinal,
                        schedule.active_turn_id,
                        schedule.state_version,
                        self._canonical_contract_json(schedule),
                        self._utc_timestamp(schedule.created_at),
                        self._utc_timestamp(schedule.updated_at),
                    ),
                )
                stored = self._load_schedule_unlocked(schedule.schedule_id)
                if stored is None:
                    raise AutonomousTurnTransactionError(
                        "created schedule could not be loaded",
                    )
                self._conn.commit()
                return stored
            except (
                AutonomousTurnTransactionError,
                InvalidScheduleTransition,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise AutonomousTurnTransactionError(
                    "autonomous schedule creation transaction failed",
                ) from exc

    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            schedule = self._load_schedule_unlocked(schedule_id)
            return None if schedule is None else schedule.model_copy(deep=True)

    def load_active_serial_public_schedule(
        self,
        game_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT schedule_json FROM autonomous_serial_public_schedules "
                "WHERE game_id = ? AND status = ?",
                (game_id, SerialPublicScheduleStatus.OPEN.value),
            ).fetchone()
            if row is None:
                return None
            return self._schedule_from_row(row).model_copy(deep=True)

    def list_open_serial_public_schedules(
        self,
    ) -> tuple[SerialPublicSchedule, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT schedule_json FROM autonomous_serial_public_schedules "
                "WHERE status = ? ORDER BY created_at, schedule_id",
                (SerialPublicScheduleStatus.OPEN.value,),
            ).fetchall()
            return tuple(
                self._schedule_from_row(row).model_copy(deep=True)
                for row in rows
            )

    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None:
        with self._lock:
            managed = self._load_managed_turn_unlocked(turn_id)
            return None if managed is None else managed.model_copy(deep=True)

    def admit_serial_public_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        """在一个 IMMEDIATE 事务中准入回合并更新调度指针。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                schedule = self._load_schedule_unlocked(schedule_id)
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                duplicate = self._conn.execute(
                    "SELECT 1 FROM autonomous_managed_turns "
                    "WHERE schedule_id = ? AND json_extract(turn_json, "
                    "'$.turn.idempotency_key') = ? LIMIT 1",
                    (schedule_id, admission.idempotency_key),
                ).fetchone()
                if duplicate is not None:
                    raise InvalidTurnAdmission(
                        "invalid autonomous turn admission",
                    )
                now = max(datetime.now(timezone.utc), schedule.updated_at)
                updated_schedule, managed = prepare_serial_public_admission(
                    schedule,
                    admission,
                    now,
                )
                self._conn.execute(
                    "INSERT INTO autonomous_managed_turns "
                    "(turn_id, schedule_id, game_id, player_id, status, state_version, "
                    "turn_json, terminal_reason, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        managed.turn.turn_id,
                        managed.schedule_id,
                        managed.turn.game_id,
                        managed.turn.player_id,
                        managed.turn.status.value,
                        managed.state_version,
                        self._canonical_contract_json(managed),
                        managed.terminal_reason,
                        self._utc_timestamp(managed.created_at),
                        self._utc_timestamp(managed.updated_at),
                    ),
                )
                self._update_schedule_unlocked(
                    updated_schedule,
                    expected_schedule_version,
                )
                self._conn.commit()
                return managed.model_copy(deep=True)
            except InvalidTurnAdmission:
                self._conn.rollback()
                raise
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                InvalidScheduleTransition,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                if (
                    isinstance(exc, sqlite3.IntegrityError)
                    and "uq_managed_turn_schedule_idempotency_key" in str(exc)
                ):
                    raise InvalidTurnAdmission(
                        "invalid autonomous turn admission",
                    ) from exc
                raise AutonomousTurnTransactionError(
                    "autonomous turn admission transaction failed",
                ) from exc

    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn:
        """在托管回合 CAS 成功后推进一个非终态状态。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                managed = self._load_managed_turn_unlocked(turn_id)
                if managed is None:
                    raise ManagedTurnNotFound("managed turn not found")
                if managed.state_version != expected_turn_version:
                    raise TurnStateConflict("managed turn state version conflict")
                schedule = self._load_schedule_unlocked(managed.schedule_id)
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                updated = prepare_active_transition(schedule, managed, next_status)
                self._update_managed_turn_unlocked(updated, expected_turn_version)
                self._conn.commit()
                return updated.model_copy(deep=True)
            except (
                ManagedTurnNotFound,
                TurnStateConflict,
                ScheduleNotFound,
                InvalidScheduleTransition,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise AutonomousTurnTransactionError(
                    "autonomous turn transition transaction failed",
                ) from exc

    def finish_active_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule:
        """在一个事务中终结托管回合并推进、替换或关闭调度。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                schedule = self._load_schedule_unlocked(schedule_id)
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                managed = self._load_managed_turn_unlocked(turn_id)
                if managed is None:
                    raise ManagedTurnNotFound("managed turn not found")
                if managed.state_version != expected_turn_version:
                    raise TurnStateConflict("managed turn state version conflict")
                now = max(datetime.now(timezone.utc), schedule.updated_at)
                updated_schedule, updated_turn = prepare_active_finish(
                    schedule,
                    managed,
                    terminal_status,
                    disposition,
                    reason_code=reason_code,
                    now=now,
                )
                self._update_managed_turn_unlocked(
                    updated_turn,
                    expected_turn_version,
                )
                self._update_schedule_unlocked(
                    updated_schedule,
                    expected_schedule_version,
                )
                self._conn.commit()
                return updated_schedule.model_copy(deep=True)
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                ManagedTurnNotFound,
                TurnStateConflict,
                InvalidScheduleTransition,
            ):
                self._conn.rollback()
                raise
            except Exception as exc:
                self._conn.rollback()
                raise AutonomousTurnTransactionError(
                    "autonomous turn finish transaction failed",
                ) from exc

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

    def supports_active_turn_fence(self) -> bool:
        """声明 SQLite 已完成活动回合围栏的 schema 初始化。"""

        with self._lock:
            return (
                self._durable_dispatch_schema_ready
                and self._autonomous_turn_schema_ready
            )

    @staticmethod
    def _dispatch_now() -> str:
        return SqliteGameRepository._utc_timestamp(datetime.now(timezone.utc))

    @staticmethod
    def _utc_timestamp(value: datetime) -> str:
        """将带时区时间统一为 UTC，确保文本排序等价于时间排序。"""
        return value.astimezone(timezone.utc).isoformat()

    @classmethod
    def _normalize_dispatch_timestamp(cls, value: str) -> str:
        """将可解析的旧版带时区字符串规范化为 UTC，异常值保持原样。"""
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return value
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return value
        return cls._utc_timestamp(parsed)

    def _normalize_dispatch_timestamps(self) -> None:
        """幂等回填历史 dispatch 时间，避免 offset 文本造成跨后端错序。"""
        attempt_rows = self._conn.execute(
            "SELECT dispatch_id, deadline, created_at, updated_at "
            "FROM autonomous_dispatch_attempts"
        ).fetchall()
        for dispatch_id, deadline, created_at, updated_at in attempt_rows:
            attempt_normalized = (
                self._normalize_dispatch_timestamp(deadline),
                self._normalize_dispatch_timestamp(created_at),
                self._normalize_dispatch_timestamp(updated_at),
            )
            if attempt_normalized != (deadline, created_at, updated_at):
                self._conn.execute(
                    "UPDATE autonomous_dispatch_attempts SET deadline = ?, "
                    "created_at = ?, updated_at = ? WHERE dispatch_id = ?",
                    (*attempt_normalized, dispatch_id),
                )

        result_rows = self._conn.execute(
            "SELECT result_id, recorded_at FROM autonomous_dispatch_results"
        ).fetchall()
        for result_id, recorded_at in result_rows:
            result_normalized = self._normalize_dispatch_timestamp(recorded_at)
            if result_normalized != recorded_at:
                self._conn.execute(
                    "UPDATE autonomous_dispatch_results SET recorded_at = ? "
                    "WHERE result_id = ?",
                    (result_normalized, result_id),
                )

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
                "active_turn_fence": (
                    None
                    if row[17] is None
                    else ActiveTurnDispatchFence.model_validate(json.loads(row[17]))
                ),
            },
        )

    @staticmethod
    def _result_from_row(row: sqlite3.Row | tuple[Any, ...]) -> DispatchResultRecord:
        """从元数据列和 payload-only JSON 重建完整结果记录。"""
        payload = row[7]
        if isinstance(payload, str):
            payload = json.loads(payload)
        recorded_at = row[8]
        if not isinstance(recorded_at, datetime):
            recorded_at = datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
        return DispatchResultRecord.model_validate(
            {
                "result_id": row[0],
                "dispatch_id": row[1],
                "request_hash": row[2],
                "lease_hash": row[3],
                "result_hash": row[4],
                "result_kind": row[5],
                "outcome": DispatchResultOutcome(row[6]),
                "payload": payload,
                "recorded_at": recorded_at,
            },
        )

    def _load_dispatch_unlocked(self, dispatch_id: str) -> DispatchAttempt | None:
        row = self._conn.execute(
            "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
            "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
            "lease_hash, view_fingerprint, deadline, status, state_version, "
            "reason_code, created_at, updated_at, active_turn_fence_json "
            "FROM autonomous_dispatch_attempts WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        if row is None:
            return None
        return self._dispatch_from_row(row)

    @staticmethod
    def _active_turn_fence_json(attempt: DispatchAttempt) -> str | None:
        """将围栏以稳定紧凑 JSON 存储，保持历史 NULL 记录可读。"""

        if attempt.active_turn_fence is None:
            return None
        return json.dumps(
            attempt.active_turn_fence.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _insert_dispatch_unlocked(self, attempt: DispatchAttempt) -> None:
        """插入一个已通过调用方状态校验的 durable dispatch。"""

        self._conn.execute(
            "INSERT INTO autonomous_dispatch_attempts ("
            "dispatch_id, game_id, turn_id, actor_id, operation_kind, "
            "executor_id, provider_idempotency_key, recovery_policy, "
            "request_hash, lease_hash, view_fingerprint, deadline, status, "
            "state_version, reason_code, created_at, updated_at, "
            "active_turn_fence_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                self._utc_timestamp(attempt.deadline),
                attempt.status.value,
                attempt.state_version,
                attempt.reason_code,
                self._utc_timestamp(attempt.created_at),
                self._utc_timestamp(attempt.updated_at),
                self._active_turn_fence_json(attempt),
            ),
        )

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
        """在外部 I/O 前原子持久化一个 PENDING dispatch 意图。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if attempt.active_turn_fence is not None:
                    raise DispatchInvalidTransition(
                        "plain dispatch cannot include an active turn fence",
                    )
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
                self._insert_dispatch_unlocked(attempt)
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

    def create_active_turn_dispatch(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        attempt: DispatchAttempt,
        observed_at: datetime,
    ) -> DispatchAttempt:
        """在同一 SQLite 事务中预约活动回合并写入其受围栏 dispatch。"""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                schedule = self._load_schedule_unlocked(schedule_id)
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                managed = self._load_managed_turn_unlocked(turn_id)
                if managed is None:
                    raise ManagedTurnNotFound("managed turn not found")
                if managed.state_version != expected_turn_version:
                    raise TurnStateConflict("managed turn state version conflict")
                if self._conn.execute(
                    "SELECT 1 FROM autonomous_dispatch_attempts "
                    "WHERE dispatch_id = ?",
                    (attempt.dispatch_id,),
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
                updated_managed, fenced_attempt = prepare_active_turn_dispatch(
                    schedule,
                    managed,
                    attempt,
                    observed_at,
                )
                self._insert_dispatch_unlocked(fenced_attempt)
                self._update_managed_turn_unlocked(
                    updated_managed,
                    expected_turn_version,
                )
                self._conn.commit()
                return fenced_attempt.model_copy(deep=True)
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                ManagedTurnNotFound,
                TurnStateConflict,
                ActiveTurnFenceRejected,
                DispatchIdempotencyConflict,
                DispatchRecoveryBlocked,
            ):
                self._conn.rollback()
                raise
            except Exception:  # noqa: BLE001 - 后端异常必须净化为稳定事务边界。
                self._conn.rollback()
                raise ActiveTurnFenceTransactionError(
                    "active turn fence transaction failed",
                ) from None

    def finish_active_turn_fenced(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        turn_id: str,
        expected_turn_version: int,
        terminal_status: AgentTurnStatus,
        disposition: TerminalDisposition,
        reason_code: str | None,
    ) -> SerialPublicSchedule:
        """在同一 SQLite 事务内终结回合并取消可撤销的受围栏 dispatch。"""

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                schedule = self._load_schedule_unlocked(schedule_id)
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                managed = self._load_managed_turn_unlocked(turn_id)
                if managed is None:
                    raise ManagedTurnNotFound("managed turn not found")
                if managed.state_version != expected_turn_version:
                    raise TurnStateConflict("managed turn state version conflict")
                rows = self._conn.execute(
                    "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                    "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
                    "lease_hash, view_fingerprint, deadline, status, state_version, "
                    "reason_code, created_at, updated_at, active_turn_fence_json "
                    "FROM autonomous_dispatch_attempts "
                    "WHERE game_id = ? AND turn_id = ? "
                    "ORDER BY created_at, dispatch_id",
                    (schedule.game_id, turn_id),
                ).fetchall()
                attempts = tuple(self._dispatch_from_row(row) for row in rows)
                now = max(datetime.now(timezone.utc), schedule.updated_at)
                updated_schedule, updated_managed, updated_attempts = (
                    prepare_fenced_active_finish(
                        schedule,
                        managed,
                        attempts,
                        terminal_status,
                        disposition,
                        reason_code=reason_code,
                        now=now,
                    )
                )
                for existing, updated in zip(attempts, updated_attempts):
                    if updated == existing:
                        continue
                    result = self._conn.execute(
                        "UPDATE autonomous_dispatch_attempts SET status = ?, "
                        "state_version = ?, reason_code = ?, updated_at = ? "
                        "WHERE dispatch_id = ? AND state_version = ?",
                        (
                            updated.status.value,
                            updated.state_version,
                            updated.reason_code,
                            self._utc_timestamp(updated.updated_at),
                            updated.dispatch_id,
                            existing.state_version,
                        ),
                    )
                    if result.rowcount != 1:
                        raise DispatchStateConflict(updated.dispatch_id)
                self._update_managed_turn_unlocked(
                    updated_managed,
                    expected_turn_version,
                )
                self._update_schedule_unlocked(
                    updated_schedule,
                    expected_schedule_version,
                )
                self._conn.commit()
                return updated_schedule.model_copy(deep=True)
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                ManagedTurnNotFound,
                TurnStateConflict,
                DispatchStateConflict,
                ActiveTurnFenceRejected,
                DispatchRecoveryBlocked,
                InvalidScheduleTransition,
            ):
                self._conn.rollback()
                raise
            except Exception:  # noqa: BLE001 - 后端异常必须净化为稳定事务边界。
                self._conn.rollback()
                raise ActiveTurnFenceTransactionError(
                    "active turn fence transaction failed",
                ) from None

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
                    self._utc_timestamp(updated_at),
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
                    "SELECT result_id, dispatch_id, request_hash, lease_hash, "
                    "result_hash, result_kind, outcome, result_json, recorded_at "
                    "FROM autonomous_dispatch_results "
                    "WHERE dispatch_id = ?",
                    (dispatch_id,),
                ).fetchone()
                if prior_row is not None:
                    prior = self._result_from_row(prior_row)
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
                            result.model_dump(mode="json")["payload"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        self._utc_timestamp(result.recorded_at),
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

    def list_dispatches_for_turn(
        self,
        game_id: str,
        turn_id: str,
    ) -> list[DispatchAttempt]:
        """按 game/turn 精确列出所有 dispatch，并保持稳定顺序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
                "lease_hash, view_fingerprint, deadline, status, state_version, "
                "reason_code, created_at, updated_at, active_turn_fence_json "
                "FROM autonomous_dispatch_attempts "
                "WHERE game_id = ? AND turn_id = ? "
                "ORDER BY created_at, dispatch_id",
                (game_id, turn_id),
            ).fetchall()
            return [self._dispatch_from_row(row).model_copy(deep=True) for row in rows]

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
                "lease_hash, view_fingerprint, deadline, status, state_version, "
                "reason_code, created_at, updated_at, active_turn_fence_json "
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
