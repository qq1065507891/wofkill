# -*- coding: utf-8 -*-
"""
功能描述：PostgreSQL 游戏仓库，支持事件兼容读写、串行公开调度持久化、自主玩家原子 CommitTurn 与 durable dispatch 状态机。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-30
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import (
    normalize_resolution_batch_fields,
    serialize_resolution_batch_fields,
)
from werewolf_agent.player_agents.contracts.dispatch import (
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
from werewolf_agent.storage.autonomous_commit import (
    AutonomousCommitUnsupported,
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
from werewolf_agent.storage.sqlite_store import (
    _deserialize_game_state,
    _serialize_game_state,
)


class PostgresSchemaMigrationError(RuntimeError):
    """PostgreSQL schema 升级因现存数据冲突而无法安全完成。"""


class PostgresGameRepository:
    """PostgreSQL implementation of GameRepository."""

    def __init__(self, dsn: str, *, initialize: bool = True) -> None:
        self._dsn = dsn
        self._lock = threading.Lock()
        self._conn: Any | None = None
        self._autonomous_schema_ready = False
        if initialize:
            self._ensure_connection()
            self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def save_game(self, state: GameState) -> None:
        with self._lock:
            conn = self._ensure_connection()
            try:
                self._lock_game_transaction(conn, state.game_id)
                row = conn.execute(
                    "SELECT state_json FROM games WHERE game_id = %s FOR UPDATE",
                    (state.game_id,),
                ).fetchone()
                existing = self._deserialize_state_row(row)
                validate_game_state_save(existing, state)
                conn.execute(
                    """
                    INSERT INTO games (game_id, state_json)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (game_id) DO UPDATE SET state_json = EXCLUDED.state_json
                    """,
                    (state.game_id, _serialize_game_state(state)),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def load_game(self, game_id: str) -> GameState | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT state_json FROM games WHERE game_id = %s",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            raw = row[0]
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            return _deserialize_game_state(raw)

    def append_events(self, game_id: str, events: list[GameEvent]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            try:
                self._lock_game_transaction(conn, game_id)
                state_row = conn.execute(
                    "SELECT state_json FROM games WHERE game_id = %s FOR UPDATE",
                    (game_id,),
                ).fetchone()
                saved_state = self._deserialize_state_row(state_row)
                rows = conn.execute(
                    "SELECT event_type, payload_json, event_json FROM events "
                    "WHERE game_id = %s ORDER BY seq",
                    (game_id,),
                ).fetchall()
                existing = [
                    (
                        deserialize_game_event(
                            row[2] if isinstance(row[2], dict) else json.loads(row[2])
                        )
                        if len(row) > 2 and row[2] is not None
                        else GameEvent(
                            type=row[0],
                            payload=(
                                row[1]
                                if isinstance(row[1], dict)
                                else json.loads(row[1])
                            ),
                        )
                    )
                    for row in rows
                ]
                validate_game_aborted_append(
                    game_id, saved_state, existing, events,
                )
                current = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = %s",
                    (game_id,),
                ).fetchone()[0]
                for i, event in enumerate(events):
                    conn.execute(
                        """
                        INSERT INTO events (game_id, seq, event_type, payload_json, event_json)
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                        """,
                        (
                            game_id,
                            current + i + 1,
                            event.type,
                            json.dumps(
                                serialize_legacy_event_payload(event), ensure_ascii=False,
                            ),
                            json.dumps(serialize_game_event(event), ensure_ascii=False),
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _lock_game_transaction(conn: Any, game_id: str) -> None:
        """使用事务级 advisory lock 串行化同一 game_id 的跨实例写入。"""
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (game_id,),
        )

    @staticmethod
    def _deserialize_state_row(row: Any) -> GameState | None:
        if row is None:
            return None
        raw = row[0]
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        return _deserialize_game_state(raw)

    def load_events(self, game_id: str) -> list[GameEvent]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT event_type, payload_json, event_json FROM events WHERE game_id = %s ORDER BY seq",
                (game_id,),
            ).fetchall()
        return [
                (
                    deserialize_game_event(
                        row[2] if isinstance(row[2], dict) else json.loads(row[2])
                    )
                    if len(row) > 2 and row[2] is not None
                    else GameEvent(
                        type=row[0],
                        payload=row[1] if isinstance(row[1], dict) else json.loads(row[1]),
                    )
                )
                for row in rows
        ]

    # -- Autonomous serial-public turns -----------------------------------

    def supports_autonomous_turns(self) -> bool:
        """仅在连接已完成自主 schema 初始化后报告支持。"""
        return self._conn is not None and bool(
            getattr(self, "_autonomous_schema_ready", False),
        )

    @staticmethod
    def _schedule_from_jsonb(payload: object) -> SerialPublicSchedule:
        """兼容 psycopg 的 JSONB 字典及测试/旧驱动返回的 JSON 字符串。"""
        if isinstance(payload, str):
            return SerialPublicSchedule.model_validate_json(payload)
        return SerialPublicSchedule.model_validate_json(
            json.dumps(payload, ensure_ascii=False),
        )

    @staticmethod
    def _managed_turn_from_jsonb(payload: object) -> ManagedAgentTurn:
        """兼容 psycopg 的 JSONB 字典及测试/旧驱动返回的 JSON 字符串。"""
        if isinstance(payload, str):
            return ManagedAgentTurn.model_validate_json(payload)
        return ManagedAgentTurn.model_validate_json(
            json.dumps(payload, ensure_ascii=False),
        )

    @staticmethod
    def _schedule_jsonb(schedule: SerialPublicSchedule) -> str:
        """将严格调度契约编码为 PostgreSQL JSONB 输入。"""
        return json.dumps(
            schedule.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _managed_turn_jsonb(managed: ManagedAgentTurn) -> str:
        """将严格托管回合契约编码为 PostgreSQL JSONB 输入。"""
        return json.dumps(
            managed.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _load_schedule_row(
        self,
        conn: Any,
        schedule_id: str,
        *,
        for_update: bool = False,
    ) -> SerialPublicSchedule | None:
        suffix = " FOR UPDATE" if for_update else ""
        row = conn.execute(
            "SELECT schedule_json FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = %s"
            f"{suffix}",
            (schedule_id,),
        ).fetchone()
        return None if row is None else self._schedule_from_jsonb(row[0])

    def _load_managed_turn_row(
        self,
        conn: Any,
        turn_id: str,
        *,
        for_update: bool = False,
    ) -> ManagedAgentTurn | None:
        suffix = " FOR UPDATE" if for_update else ""
        row = conn.execute(
            "SELECT turn_json FROM autonomous_managed_turns WHERE turn_id = %s"
            f"{suffix}",
            (turn_id,),
        ).fetchone()
        return None if row is None else self._managed_turn_from_jsonb(row[0])

    def _lock_schedule_transaction(self, conn: Any, schedule_id: str) -> str:
        """定位所属游戏并用现有 advisory lock 串行化跨实例写入。"""
        row = conn.execute(
            "SELECT game_id FROM autonomous_serial_public_schedules "
            "WHERE schedule_id = %s",
            (schedule_id,),
        ).fetchone()
        if row is None:
            raise ScheduleNotFound("schedule not found")
        game_id = str(row[0])
        self._lock_game_transaction(conn, game_id)
        return game_id

    def _lock_managed_turn_transaction(
        self,
        conn: Any,
        turn_id: str,
    ) -> tuple[str, str]:
        """定位托管回合的调度与游戏，并获取游戏级事务锁。"""
        row = conn.execute(
            "SELECT schedule_id, game_id FROM autonomous_managed_turns "
            "WHERE turn_id = %s",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise ManagedTurnNotFound("managed turn not found")
        schedule_id, game_id = str(row[0]), str(row[1])
        self._lock_game_transaction(conn, game_id)
        return schedule_id, game_id

    def _update_schedule_row(
        self,
        conn: Any,
        schedule: SerialPublicSchedule,
        expected_version: int,
    ) -> None:
        cursor = conn.execute(
            "UPDATE autonomous_serial_public_schedules SET "
            "game_id = %s, window_id = %s, status = %s, "
            "next_slot_ordinal = %s, active_turn_id = %s, state_version = %s, "
            "schedule_json = %s::jsonb, created_at = %s, updated_at = %s "
            "WHERE schedule_id = %s AND state_version = %s",
            (
                schedule.game_id,
                schedule.window.window_id,
                schedule.status.value,
                schedule.next_slot_ordinal,
                schedule.active_turn_id,
                schedule.state_version,
                self._schedule_jsonb(schedule),
                schedule.created_at,
                schedule.updated_at,
                schedule.schedule_id,
                expected_version,
            ),
        )
        if getattr(cursor, "rowcount", None) != 1:
            raise ScheduleStateConflict("schedule state version conflict")

    def _update_managed_turn_row(
        self,
        conn: Any,
        managed: ManagedAgentTurn,
        expected_version: int,
    ) -> None:
        cursor = conn.execute(
            "UPDATE autonomous_managed_turns SET "
            "schedule_id = %s, game_id = %s, player_id = %s, status = %s, "
            "state_version = %s, turn_json = %s::jsonb, terminal_reason = %s, "
            "created_at = %s, updated_at = %s "
            "WHERE turn_id = %s AND state_version = %s",
            (
                managed.schedule_id,
                managed.turn.game_id,
                managed.turn.player_id,
                managed.turn.status.value,
                managed.state_version,
                self._managed_turn_jsonb(managed),
                managed.terminal_reason,
                managed.created_at,
                managed.updated_at,
                managed.turn.turn_id,
                expected_version,
            ),
        )
        if getattr(cursor, "rowcount", None) != 1:
            raise TurnStateConflict("managed turn state version conflict")

    def create_serial_public_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule:
        """创建公开调度，并用游戏级锁与部分唯一索引维持单开放调度。"""
        require_fresh_serial_public_schedule(schedule)
        with self._lock:
            conn = self._ensure_connection()
            try:
                self._lock_game_transaction(conn, schedule.game_id)
                game_row = conn.execute(
                    "SELECT 1 FROM games WHERE game_id = %s FOR UPDATE",
                    (schedule.game_id,),
                ).fetchone()
                if game_row is None:
                    raise AutonomousTurnTransactionError(
                        "game is required before creating a schedule",
                    )
                if self._load_schedule_row(
                    conn,
                    schedule.schedule_id,
                    for_update=True,
                ) is not None:
                    raise AutonomousTurnTransactionError("schedule already exists")
                conn.execute(
                    "INSERT INTO autonomous_serial_public_schedules "
                    "(schedule_id, game_id, window_id, status, next_slot_ordinal, "
                    "active_turn_id, state_version, schedule_json, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)",
                    (
                        schedule.schedule_id,
                        schedule.game_id,
                        schedule.window.window_id,
                        schedule.status.value,
                        schedule.next_slot_ordinal,
                        schedule.active_turn_id,
                        schedule.state_version,
                        self._schedule_jsonb(schedule),
                        schedule.created_at,
                        schedule.updated_at,
                    ),
                )
                stored = self._load_schedule_row(conn, schedule.schedule_id)
                if stored is None:
                    raise AutonomousTurnTransactionError(
                        "created schedule could not be loaded",
                    )
                conn.commit()
                return stored.model_copy(deep=True)
            except (
                AutonomousTurnTransactionError,
                InvalidScheduleTransition,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                raise AutonomousTurnTransactionError(
                    "autonomous schedule creation transaction failed",
                ) from exc

    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            schedule = self._load_schedule_row(
                self._ensure_connection(),
                schedule_id,
            )
            return None if schedule is None else schedule.model_copy(deep=True)

    def load_active_serial_public_schedule(
        self,
        game_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT schedule_json FROM autonomous_serial_public_schedules "
                "WHERE game_id = %s AND status = %s",
                (game_id, SerialPublicScheduleStatus.OPEN.value),
            ).fetchone()
            if row is None:
                return None
            return self._schedule_from_jsonb(row[0]).model_copy(deep=True)

    def list_open_serial_public_schedules(
        self,
    ) -> tuple[SerialPublicSchedule, ...]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT schedule_json FROM autonomous_serial_public_schedules "
                "WHERE status = %s ORDER BY created_at, schedule_id",
                (SerialPublicScheduleStatus.OPEN.value,),
            ).fetchall()
            return tuple(
                self._schedule_from_jsonb(row[0]).model_copy(deep=True)
                for row in rows
            )

    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None:
        with self._lock:
            managed = self._load_managed_turn_row(
                self._ensure_connection(),
                turn_id,
            )
            return None if managed is None else managed.model_copy(deep=True)

    def admit_serial_public_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        """锁定游戏和调度后，以 CAS 原子准入当前公开位置。"""
        with self._lock:
            conn = self._ensure_connection()
            try:
                self._lock_schedule_transaction(conn, schedule_id)
                schedule = self._load_schedule_row(
                    conn,
                    schedule_id,
                    for_update=True,
                )
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                duplicate = conn.execute(
                    "SELECT 1 FROM autonomous_managed_turns "
                    "WHERE schedule_id = %s "
                    "AND turn_json #>> '{turn,idempotency_key}' = %s LIMIT 1",
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
                conn.execute(
                    "INSERT INTO autonomous_managed_turns "
                    "(turn_id, schedule_id, game_id, player_id, status, state_version, "
                    "turn_json, terminal_reason, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)",
                    (
                        managed.turn.turn_id,
                        managed.schedule_id,
                        managed.turn.game_id,
                        managed.turn.player_id,
                        managed.turn.status.value,
                        managed.state_version,
                        self._managed_turn_jsonb(managed),
                        managed.terminal_reason,
                        managed.created_at,
                        managed.updated_at,
                    ),
                )
                self._update_schedule_row(
                    conn,
                    updated_schedule,
                    expected_schedule_version,
                )
                conn.commit()
                return managed.model_copy(deep=True)
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                InvalidTurnAdmission,
                InvalidScheduleTransition,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                constraint_name = getattr(
                    getattr(exc, "diag", None),
                    "constraint_name",
                    None,
                )
                if (
                    constraint_name == "uq_managed_turn_schedule_idempotency_key"
                    or "uq_managed_turn_schedule_idempotency_key" in str(exc)
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
        """在游戏级锁和行锁下推进托管回合的非终态状态。"""
        with self._lock:
            conn = self._ensure_connection()
            try:
                schedule_id, _ = self._lock_managed_turn_transaction(conn, turn_id)
                managed = self._load_managed_turn_row(
                    conn,
                    turn_id,
                    for_update=True,
                )
                if managed is None:
                    raise ManagedTurnNotFound("managed turn not found")
                if managed.state_version != expected_turn_version:
                    raise TurnStateConflict("managed turn state version conflict")
                schedule = self._load_schedule_row(
                    conn,
                    schedule_id,
                    for_update=True,
                )
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                updated = prepare_active_transition(schedule, managed, next_status)
                self._update_managed_turn_row(conn, updated, expected_turn_version)
                conn.commit()
                return updated.model_copy(deep=True)
            except (
                ManagedTurnNotFound,
                TurnStateConflict,
                ScheduleNotFound,
                InvalidScheduleTransition,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
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
        """原子终结托管回合并推进、替换或关闭公开调度。"""
        with self._lock:
            conn = self._ensure_connection()
            try:
                self._lock_schedule_transaction(conn, schedule_id)
                schedule = self._load_schedule_row(
                    conn,
                    schedule_id,
                    for_update=True,
                )
                if schedule is None:
                    raise ScheduleNotFound("schedule not found")
                if schedule.state_version != expected_schedule_version:
                    raise ScheduleStateConflict("schedule state version conflict")
                managed = self._load_managed_turn_row(
                    conn,
                    turn_id,
                    for_update=True,
                )
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
                self._update_managed_turn_row(
                    conn,
                    updated_turn,
                    expected_turn_version,
                )
                self._update_schedule_row(
                    conn,
                    updated_schedule,
                    expected_schedule_version,
                )
                conn.commit()
                return updated_schedule.model_copy(deep=True)
            except (
                ScheduleNotFound,
                ScheduleStateConflict,
                ManagedTurnNotFound,
                TurnStateConflict,
                InvalidScheduleTransition,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                raise AutonomousTurnTransactionError(
                    "autonomous turn finish transaction failed",
                ) from exc

    # -- Autonomous CommitTurn --------------------------------------------

    def supports_autonomous_commit(self) -> bool:
        """仅在连接已完成自主事务 schema 初始化后报告支持。"""
        return self._conn is not None and getattr(
            self, "_autonomous_schema_ready", False,
        )

    def load_game_revision(self, game_id: str) -> int:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT game_revision FROM autonomous_game_streams WHERE game_id = %s",
                (game_id,),
            ).fetchone()
            if row is not None:
                return int(row[0])
            row = self._ensure_connection().execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = %s",
                (game_id,),
            ).fetchone()
            return int(row[0] or 0)

    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT request_json FROM autonomous_projection_outbox "
                "WHERE game_id = %s ORDER BY committed_revision, outbox_id",
                (game_id,),
            ).fetchall()
            return [
                ProjectionOutboxRecord.model_validate(
                    row[0] if isinstance(row[0], dict) else json.loads(row[0]),
                )
                for row in rows
            ]

    def commit_turn(self, request: CommitTurnRequest) -> CommitResult:
        with self._lock:
            if not self.supports_autonomous_commit():
                raise AutonomousCommitUnsupported(
                    "postgres repository schema is not ready for autonomous commits",
                )
            conn = self._ensure_connection()
            digest = request_hash(request)
            try:
                self._lock_game_transaction(conn, request.game_id)
                game_row = conn.execute(
                    "SELECT 1 FROM games WHERE game_id = %s FOR UPDATE",
                    (request.game_id,),
                ).fetchone()
                if game_row is None:
                    raise CommitTransactionError(
                        f"game does not exist: {request.game_id}",
                    )
                existing_row = conn.execute(
                    "SELECT request_hash, result_json FROM autonomous_turn_commits "
                    "WHERE game_id = %s AND turn_id = %s AND idempotency_key = %s",
                    (request.game_id, request.turn_id, request.idempotency_key),
                ).fetchone()
                if existing_row is not None:
                    if existing_row[0] != digest:
                        raise IdempotencyConflictError(
                            "idempotency key conflicts with an existing proposal",
                        )
                    raw_result = existing_row[1]
                    existing = CommitResult.model_validate_json(
                        raw_result
                        if isinstance(raw_result, str)
                        else json.dumps(raw_result, ensure_ascii=False),
                    )
                    conn.rollback()
                    return existing.model_copy(update={"replayed": True})

                stream_row = conn.execute(
                    "SELECT game_revision FROM autonomous_game_streams "
                    "WHERE game_id = %s FOR UPDATE",
                    (request.game_id,),
                ).fetchone()
                if stream_row is None:
                    current = int(conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = %s",
                        (request.game_id,),
                    ).fetchone()[0] or 0)
                    conn.execute(
                        "INSERT INTO autonomous_game_streams (game_id, game_revision) "
                        "VALUES (%s, %s)",
                        (request.game_id, current),
                    )
                else:
                    current = int(stream_row[0])
                    event_head = int(conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = %s",
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
                conn.execute(
                    "INSERT INTO events "
                    "(game_id, seq, event_type, payload_json, event_json) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)",
                    (
                        request.game_id,
                        next_revision,
                        event.type,
                        json.dumps(serialize_legacy_event_payload(event), ensure_ascii=False),
                        json.dumps(serialize_game_event(event), ensure_ascii=False),
                    ),
                )
                if record is not None:
                    conn.execute(
                        "INSERT INTO autonomous_public_records "
                        "(record_id, game_id, turn_id, committed_revision, record_json) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb)",
                        (
                            record.record_id,
                            request.game_id,
                            request.turn_id,
                            next_revision,
                            record.model_dump_json(),
                        ),
                    )
                for audit in request.critical_audit_records:
                    conn.execute(
                        "INSERT INTO autonomous_audit_records "
                        "(audit_id, game_id, committed_revision, record_json) "
                        "VALUES (%s, %s, %s, %s::jsonb)",
                        (
                            audit.audit_id,
                            request.game_id,
                            next_revision,
                            audit.model_dump_json(),
                        ),
                    )
                for outbox in request.projection_outbox_records:
                    conn.execute(
                        "INSERT INTO autonomous_projection_outbox "
                        "(outbox_id, game_id, committed_revision, request_json) "
                        "VALUES (%s, %s, %s, %s::jsonb)",
                        (
                            outbox.outbox_id,
                            request.game_id,
                            next_revision,
                            outbox.model_dump_json(),
                        ),
                    )
                conn.execute(
                    "UPDATE autonomous_game_streams SET game_revision = %s "
                    "WHERE game_id = %s",
                    (next_revision, request.game_id),
                )
                conn.execute(
                    "INSERT INTO autonomous_turn_commits "
                    "(game_id, turn_id, idempotency_key, request_hash, result_json, committed_revision) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                    (
                        request.game_id,
                        request.turn_id,
                        request.idempotency_key,
                        digest,
                        result.model_dump_json(),
                        next_revision,
                    ),
                )
                conn.commit()
                return result
            except (StaleCommitError, IdempotencyConflictError):
                conn.rollback()
                raise
            except CommitTransactionError:
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                raise CommitTransactionError(
                    "autonomous CommitTurn transaction failed",
                ) from exc

    # -- Durable dispatch --------------------------------------------------

    def supports_durable_dispatch(self) -> bool:
        """仅在连接已存在且 durable dispatch schema 完成后声明 capability。"""
        return self._conn is not None and bool(
            getattr(self, "_autonomous_schema_ready", False),
        )

    @staticmethod
    def _dispatch_copy(attempt: DispatchAttempt) -> DispatchAttempt:
        """从 ORM/数据库值重建防御性副本，避免泄露冻结映射内部对象。"""
        return DispatchAttempt.model_validate(attempt.model_dump(round_trip=True))

    @staticmethod
    def _dispatch_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    @classmethod
    def _dispatch_from_row(cls, row: Any) -> DispatchAttempt:
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
                "deadline": cls._dispatch_datetime(row[11]),
                "status": DispatchStatus(row[12]),
                "state_version": int(row[13]),
                "reason_code": row[14],
                "created_at": cls._dispatch_datetime(row[15]),
                "updated_at": cls._dispatch_datetime(row[16]),
            },
        )

    @staticmethod
    def _dispatch_select_columns() -> str:
        return (
            "dispatch_id, game_id, turn_id, actor_id, operation_kind, "
            "executor_id, provider_idempotency_key, recovery_policy, request_hash, "
            "lease_hash, view_fingerprint, deadline, status, state_version, "
            "reason_code, created_at, updated_at"
        )

    def _load_dispatch_row(
        self,
        conn: Any,
        dispatch_id: str,
        *,
        for_update: bool = False,
    ) -> DispatchAttempt | None:
        suffix = " FOR UPDATE" if for_update else ""
        row = conn.execute(
            "SELECT "
            f"{self._dispatch_select_columns()} "
            "FROM autonomous_dispatch_attempts WHERE dispatch_id = %s"
            f"{suffix}",
            (dispatch_id,),
        ).fetchone()
        return None if row is None else self._dispatch_from_row(row)

    @staticmethod
    def _lock_dispatch_game(conn: Any, game_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM games WHERE game_id = %s FOR UPDATE",
            (game_id,),
        ).fetchone()
        if row is None:
            raise DispatchTransactionError(f"game does not exist: {game_id}")

    @staticmethod
    def _rowcount_is_one(cursor: Any) -> bool:
        rowcount = getattr(cursor, "rowcount", 1)
        return not isinstance(rowcount, int) or rowcount == 1

    @staticmethod
    def _is_unique_violation(exc: BaseException) -> bool:
        """识别 psycopg 不同版本暴露的唯一约束冲突。"""
        if getattr(exc, "sqlstate", None) == "23505":
            return True
        if getattr(exc, "pgcode", None) == "23505":
            return True
        constraint_name = getattr(exc, "constraint_name", None)
        if constraint_name is None:
            constraint_name = getattr(
                getattr(exc, "diag", None),
                "constraint_name",
                None,
            )
        return constraint_name in {
            "autonomous_dispatch_attempts_pkey",
            "uq_autonomous_dispatch_executor_provider_key",
            "autonomous_dispatch_results_pkey",
            "autonomous_dispatch_results_dispatch_id_key",
        }

    @staticmethod
    def _dispatch_result_copy(result: DispatchResultRecord) -> DispatchResultRecord:
        return DispatchResultRecord.model_validate(result.model_dump(round_trip=True))

    @classmethod
    def _result_from_row(cls, row: Any) -> DispatchResultRecord:
        if len(row) == 1:
            raw = row[0]
            if isinstance(raw, str):
                return DispatchResultRecord.model_validate_json(raw)
            return DispatchResultRecord.model_validate(raw)
        payload = row[7]
        if isinstance(payload, str):
            payload = json.loads(payload)
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
                "recorded_at": cls._dispatch_datetime(row[8]),
            },
        )

    def _load_result_row(
        self,
        conn: Any,
        dispatch_id: str,
        *,
        for_update: bool = False,
    ) -> DispatchResultRecord | None:
        suffix = " FOR UPDATE" if for_update else ""
        row = conn.execute(
            "SELECT result_id, dispatch_id, request_hash, lease_hash, result_hash, "
            "result_kind, outcome, result_json, recorded_at "
            "FROM autonomous_dispatch_results WHERE dispatch_id = %s"
            f"{suffix}",
            (dispatch_id,),
        ).fetchone()
        return None if row is None else self._result_from_row(row)

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
        """在外部网络 I/O 前原子持久化一个 PENDING dispatch 意图。"""
        with self._lock:
            conn = self._ensure_connection()
            try:
                if (
                    attempt.status is not DispatchStatus.PENDING
                    or attempt.state_version != 0
                ):
                    raise DispatchInvalidTransition(
                        "new dispatch must start in PENDING at version 0",
                    )
                self._lock_dispatch_game(conn, attempt.game_id)
                if self._load_dispatch_row(conn, attempt.dispatch_id, for_update=True):
                    raise DispatchIdempotencyConflict(attempt.dispatch_id)
                key_row = conn.execute(
                    "SELECT dispatch_id FROM autonomous_dispatch_attempts "
                    "WHERE executor_id = %s AND provider_idempotency_key = %s "
                    "FOR UPDATE",
                    (attempt.executor_id, attempt.provider_idempotency_key),
                ).fetchone()
                if key_row is not None:
                    raise DispatchIdempotencyConflict(
                        "provider idempotency key already exists: "
                        f"{(attempt.executor_id, attempt.provider_idempotency_key)}",
                    )
                barrier_row = conn.execute(
                    "SELECT dispatch_id FROM autonomous_dispatch_attempts "
                    "WHERE game_id = %s AND status IN (%s, %s) FOR UPDATE",
                    (
                        attempt.game_id,
                        DispatchStatus.DISPATCHING.value,
                        DispatchStatus.DISPATCHED.value,
                    ),
                ).fetchone()
                if barrier_row is not None:
                    raise DispatchRecoveryBlocked(attempt.game_id)
                conn.execute(
                    "INSERT INTO autonomous_dispatch_attempts ("
                    "dispatch_id, game_id, turn_id, actor_id, operation_kind, "
                    "executor_id, provider_idempotency_key, recovery_policy, "
                    "request_hash, lease_hash, view_fingerprint, deadline, status, "
                    "state_version, reason_code, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s, %s)",
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
                        attempt.deadline,
                        attempt.status.value,
                        attempt.state_version,
                        attempt.reason_code,
                        attempt.created_at,
                        attempt.updated_at,
                    ),
                )
                conn.commit()
                return self._dispatch_copy(attempt)
            except (
                DispatchInvalidTransition,
                DispatchIdempotencyConflict,
                DispatchRecoveryBlocked,
                DispatchTransactionError,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                if self._is_unique_violation(exc):
                    raise DispatchIdempotencyConflict(attempt.dispatch_id) from exc
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
        conn = self._ensure_connection()
        try:
            # 先读出 game_id，再按 game -> dispatch 顺序加锁，避免与 create
            # 的同一顺序相反而形成跨事务死锁。
            existing = self._load_dispatch_row(conn, dispatch_id)
            if existing is None:
                raise DispatchNotFound(dispatch_id)
            self._lock_dispatch_game(conn, existing.game_id)
            existing = self._load_dispatch_row(conn, dispatch_id, for_update=True)
            if existing is None:
                raise DispatchNotFound(dispatch_id)
            if existing.state_version != expected_version:
                raise DispatchStateConflict(dispatch_id)
            if existing.status not in allowed_statuses:
                raise DispatchInvalidTransition(dispatch_id)
            updated_at = datetime.now(timezone.utc)
            status_placeholders = ", ".join("%s" for _ in allowed_statuses)
            cursor = conn.execute(
                "UPDATE autonomous_dispatch_attempts SET status = %s, "
                "state_version = state_version + 1, reason_code = %s, updated_at = %s "
                "WHERE dispatch_id = %s AND state_version = %s "
                f"AND status IN ({status_placeholders})",
                (
                    target_status.value,
                    reason_code,
                    updated_at,
                    dispatch_id,
                    expected_version,
                    *(status.value for status in allowed_statuses),
                ),
            )
            if not self._rowcount_is_one(cursor):
                current = self._load_dispatch_row(
                    conn,
                    dispatch_id,
                    for_update=True,
                )
                if current is None:
                    raise DispatchNotFound(dispatch_id)
                if current.state_version != expected_version:
                    raise DispatchStateConflict(dispatch_id)
                if current.status not in allowed_statuses:
                    raise DispatchInvalidTransition(dispatch_id)
                raise DispatchStateConflict(dispatch_id)
            updated = existing.model_copy(
                update={
                    "status": target_status,
                    "state_version": existing.state_version + 1,
                    "reason_code": reason_code,
                    "updated_at": updated_at,
                },
            )
            conn.commit()
            return self._dispatch_copy(updated)
        except (
            DispatchNotFound,
            DispatchStateConflict,
            DispatchInvalidTransition,
            DispatchTransactionError,
        ):
            conn.rollback()
            raise
        except Exception as exc:
            conn.rollback()
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
        """原子写入结果并使用 dispatch 状态版本 CAS 推进 attempt。"""
        with self._lock:
            conn = self._ensure_connection()
            try:
                attempt = self._load_dispatch_row(conn, dispatch_id)
                if attempt is None:
                    raise DispatchNotFound(dispatch_id)
                self._lock_dispatch_game(conn, attempt.game_id)
                attempt = self._load_dispatch_row(conn, dispatch_id, for_update=True)
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

                prior = self._load_result_row(conn, dispatch_id, for_update=True)
                if prior is not None:
                    if prior == result:
                        conn.rollback()
                        return DispatchResultDisposition.REPLAYED
                    raise DispatchResultConflict(dispatch_id)
                if attempt.status in {
                    DispatchStatus.CANCELLED,
                    DispatchStatus.UNKNOWN_OUTCOME,
                }:
                    conn.rollback()
                    return DispatchResultDisposition.DISCARDED_LATE
                if attempt.status is not DispatchStatus.DISPATCHED:
                    raise DispatchInvalidTransition(dispatch_id)
                result_id_row = conn.execute(
                    "SELECT dispatch_id FROM autonomous_dispatch_results "
                    "WHERE result_id = %s FOR UPDATE",
                    (result.result_id,),
                ).fetchone()
                if result_id_row is not None:
                    raise DispatchResultConflict(dispatch_id)

                payload = json.dumps(
                    result.model_dump(mode="json")["payload"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                conn.execute(
                    "INSERT INTO autonomous_dispatch_results ("
                    "result_id, dispatch_id, request_hash, lease_hash, result_hash, "
                    "result_kind, outcome, result_json, recorded_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                    (
                        result.result_id,
                        dispatch_id,
                        result.request_hash,
                        result.lease_hash,
                        result.result_hash,
                        result.result_kind,
                        result.outcome.value,
                        payload,
                        result.recorded_at,
                    ),
                )
                updated_at = datetime.now(timezone.utc)
                cursor = conn.execute(
                    "UPDATE autonomous_dispatch_attempts SET status = %s, "
                    "state_version = state_version + 1, reason_code = NULL, "
                    "updated_at = %s WHERE dispatch_id = %s "
                    "AND state_version = %s AND status = %s",
                    (
                        DispatchStatus.RESULT_RECORDED.value,
                        updated_at,
                        dispatch_id,
                        expected_version,
                        DispatchStatus.DISPATCHED.value,
                    ),
                )
                if not self._rowcount_is_one(cursor):
                    current = self._load_dispatch_row(
                        conn,
                        dispatch_id,
                        for_update=True,
                    )
                    if current is None:
                        raise DispatchNotFound(dispatch_id)
                    if current.state_version != expected_version:
                        raise DispatchStateConflict(dispatch_id)
                    if current.status is not DispatchStatus.DISPATCHED:
                        raise DispatchInvalidTransition(dispatch_id)
                    raise DispatchStateConflict(dispatch_id)
                conn.commit()
                return DispatchResultDisposition.RECORDED
            except (
                DispatchNotFound,
                DispatchStateConflict,
                DispatchInvalidTransition,
                DispatchLeaseMismatch,
                DispatchResultConflict,
                DispatchTransactionError,
            ):
                conn.rollback()
                raise
            except Exception as exc:
                conn.rollback()
                if self._is_unique_violation(exc):
                    raise DispatchResultConflict(dispatch_id) from exc
                raise DispatchTransactionError(
                    "durable dispatch result transaction failed",
                ) from exc

    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None:
        with self._lock:
            attempt = self._load_dispatch_row(
                self._ensure_connection(), dispatch_id,
            )
            return None if attempt is None else self._dispatch_copy(attempt)

    def list_dispatches_for_turn(
        self,
        game_id: str,
        turn_id: str,
    ) -> list[DispatchAttempt]:
        """按 game/turn 精确列出所有 dispatch，并保持稳定顺序。"""
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT "
                f"{self._dispatch_select_columns()} "
                "FROM autonomous_dispatch_attempts "
                "WHERE game_id = %s AND turn_id = %s "
                "ORDER BY created_at, dispatch_id",
                (game_id, turn_id),
            ).fetchall()
            return [
                self._dispatch_copy(self._dispatch_from_row(row))
                for row in rows
            ]

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT "
                f"{self._dispatch_select_columns()} "
                "FROM autonomous_dispatch_attempts "
                "WHERE game_id = %s AND status IN (%s, %s) "
                "ORDER BY created_at, dispatch_id",
                (
                    game_id,
                    DispatchStatus.DISPATCHING.value,
                    DispatchStatus.DISPATCHED.value,
                ),
            ).fetchall()
            return [
                self._dispatch_copy(self._dispatch_from_row(row))
                for row in rows
            ]

    def assert_dispatch_allowed(self, game_id: str) -> None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT dispatch_id FROM autonomous_dispatch_attempts "
                "WHERE game_id = %s AND status IN (%s, %s) LIMIT 1",
                (
                    game_id,
                    DispatchStatus.DISPATCHING.value,
                    DispatchStatus.DISPATCHED.value,
                ),
            ).fetchone()
            if row is not None:
                raise DispatchRecoveryBlocked(game_id)

    def save_deaths(self, game_id: str, deaths: list[Death]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("DELETE FROM deaths WHERE game_id = %s", (game_id,))
            for death in deaths:
                conn.execute(
                    """
                    INSERT INTO deaths (game_id, player_id, death_json)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    (
                        game_id,
                        death.player_id,
                        json.dumps(
                            serialize_resolution_batch_fields(asdict(death)),
                            ensure_ascii=False,
                        ),
                    ),
                )
            conn.commit()

    def load_deaths(self, game_id: str) -> list[Death]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT death_json FROM deaths WHERE game_id = %s",
                (game_id,),
            ).fetchall()
            return [
                Death(
                    **normalize_resolution_batch_fields(
                        row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    )
                )
                for row in rows
            ]

    def save_model_usage(self, game_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute(
                "INSERT INTO model_usage (game_id, record_json) VALUES (%s, %s::jsonb)",
                (game_id, json.dumps(record, ensure_ascii=False)),
            )
            conn.commit()

    def load_model_usage(self, game_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT record_json FROM model_usage WHERE game_id = %s ORDER BY id",
                (game_id,),
            ).fetchall()
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def save_evaluation(self, game_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute(
                """
                INSERT INTO evaluations (game_id, result_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (game_id) DO UPDATE SET result_json = EXCLUDED.result_json
                """,
                (game_id, json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()

    def load_evaluation(self, game_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT result_json FROM evaluations WHERE game_id = %s",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def save_config_snapshot(self, game_id: str, config: dict[str, Any]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute(
                """
                INSERT INTO config_snapshots (game_id, config_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (game_id) DO UPDATE SET config_json = EXCLUDED.config_json
                """,
                (game_id, json.dumps(config, ensure_ascii=False)),
            )
            conn.commit()

    def load_config_snapshot(self, game_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT config_json FROM config_snapshots WHERE game_id = %s",
                (game_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def save_rag_entries(self, entries: list[dict[str, Any]]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            for entry in entries:
                entry_id = entry.get("entry_id")
                if not entry_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO rag_entries (entry_id, entry_json)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (entry_id) DO UPDATE SET entry_json = EXCLUDED.entry_json
                    """,
                    (entry_id, json.dumps(entry, ensure_ascii=False)),
                )
            conn.commit()

    def load_rag_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT entry_json FROM rag_entries ORDER BY entry_id"
            ).fetchall()
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def delete_rag_entry(self, entry_id: str) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("DELETE FROM rag_entries WHERE entry_id = %s", (entry_id,))
            conn.commit()

    def list_games(self) -> list[GameState]:
        with self._lock:
            rows = self._ensure_connection().execute("SELECT state_json FROM games ORDER BY game_id").fetchall()
            states: list[GameState] = []
            for row in rows:
                raw = row[0]
                if not isinstance(raw, str):
                    raw = json.dumps(raw, ensure_ascii=False)
                states.append(_deserialize_game_state(raw))
            return states

    def delete_game(self, game_id: str) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("DELETE FROM games WHERE game_id = %s", (game_id,))
            conn.commit()

    # -- Memory snapshots ---------------------------------------------------

    def save_memory_snapshot(self, snapshot_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute(
                """
                INSERT INTO memory_snapshots (snapshot_id, snapshot_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (snapshot_id) DO UPDATE SET snapshot_json = EXCLUDED.snapshot_json
                """,
                (snapshot_id, json.dumps(data, ensure_ascii=False)),
            )
            conn.commit()

    def load_memory_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT snapshot_json FROM memory_snapshots WHERE snapshot_id = %s",
                (snapshot_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def list_memory_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT snapshot_id, created_at FROM memory_snapshots ORDER BY created_at DESC"
            ).fetchall()
            return [{"snapshot_id": r[0], "created_at": str(r[1]) if r[1] else None} for r in rows]

    def delete_memory_snapshot(self, snapshot_id: str) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("DELETE FROM memory_snapshots WHERE snapshot_id = %s", (snapshot_id,))
            conn.commit()

    # -- Reflections --------------------------------------------------------

    def save_reflection(self, entry: dict[str, Any]) -> None:
        with self._lock:
            conn = self._ensure_connection()
            entry_id = entry.get("entry_id", "")
            conn.execute(
                """
                INSERT INTO reflections (entry_id, game_id, player_id, entry_json)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (entry_id) DO UPDATE SET entry_json = EXCLUDED.entry_json
                """,
                (entry_id, entry.get("game_id", ""), entry.get("player_id", ""),
                 json.dumps(entry, ensure_ascii=False)),
            )
            conn.commit()

    def load_reflection(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._ensure_connection().execute(
                "SELECT entry_json FROM reflections WHERE entry_id = %s",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def load_reflections_by_game(self, game_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT entry_json FROM reflections WHERE game_id = %s",
                (game_id,),
            ).fetchall()
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def load_reflections_by_player(self, player_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT entry_json FROM reflections WHERE player_id = %s",
                (player_id,),
            ).fetchall()
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def load_all_reflections(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT entry_json FROM reflections ORDER BY entry_id"
            ).fetchall()
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]

    def delete_reflection(self, entry_id: str) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.execute("DELETE FROM reflections WHERE entry_id = %s", (entry_id,))
            conn.commit()

    # -- Customization configs ----------------------------------------------

    def save_custom_config(self, record: dict[str, Any]) -> None:
        """P-A1: persist a full custom-config record (upsert by config_id)."""
        with self._lock:
            conn = self._ensure_connection()
            conn.execute(
                """
                INSERT INTO custom_configs
                    (config_id, config_type, record_json, created_at, updated_at)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (config_id) DO UPDATE SET
                    config_type = EXCLUDED.config_type,
                    record_json = EXCLUDED.record_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    str(record["config_id"]),
                    str(record.get("config_type", "")),
                    json.dumps(record, ensure_ascii=False),
                    str(record.get("created_at", "")),
                    str(record.get("updated_at", "")),
                ),
            )
            conn.commit()

    def load_custom_config(self, config_id: str) -> dict[str, Any] | None:
        """P-A1: load custom-config record by id, None if missing."""
        with self._lock:
            conn = self._ensure_connection()
            row = conn.execute(
                "SELECT record_json FROM custom_configs WHERE config_id = %s",
                (config_id,),
            ).fetchone()
            if row is None:
                return None
            raw = row[0]
            return raw if isinstance(raw, dict) else json.loads(raw)

    def list_custom_configs(self, config_type: str | None = None) -> list[dict[str, Any]]:
        """P-A1: list all custom configs, optionally filtered by config_type."""
        with self._lock:
            conn = self._ensure_connection()
            if config_type is None:
                rows = conn.execute(
                    "SELECT record_json FROM custom_configs ORDER BY created_at, config_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT record_json FROM custom_configs "
                    "WHERE config_type = %s ORDER BY created_at, config_id",
                    (config_type,),
                ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                raw = row[0]
                results.append(raw if isinstance(raw, dict) else json.loads(raw))
            return results

    def _connect(self) -> Any:
        if self._conn is not None:
            return self._conn
        self._conn = self._new_connection()
        return self._conn

    def _ensure_connection(self) -> Any:
        """Return existing connection or establish a new one.

        Reconnects if the stored connection was previously marked dead
        (self._conn is None). Lightweight probe is deferred to the caller;
        if a query fails with a connection error, the caller should set
        self._conn = None and retry.
        """
        if self._conn is None:
            self._conn = self._new_connection()
        return self._conn

    def _new_connection(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required for postgres storage") from exc
        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        conn = self._ensure_connection()
        try:
            self._ensure_schema_transaction(conn)
            conn.commit()
            self._autonomous_schema_ready = True
        except Exception:
            conn.rollback()
            self._autonomous_schema_ready = False
            raise

    def _ensure_schema_transaction(self, conn: Any) -> None:
        """在单个事务中创建基础 schema 并完成事件完整性升级。"""
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                state_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id BIGSERIAL PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json JSONB NOT NULL
            )
        """)
        conn.execute(
            "ALTER TABLE events ADD COLUMN IF NOT EXISTS event_json JSONB"
        )
        self._upgrade_event_integrity_v3(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deaths (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                player_id TEXT NOT NULL,
                death_json JSONB NOT NULL,
                PRIMARY KEY (game_id, player_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_usage (
                id BIGSERIAL PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                record_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                game_id TEXT PRIMARY KEY REFERENCES games(game_id) ON DELETE CASCADE,
                result_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_snapshots (
                game_id TEXT PRIMARY KEY REFERENCES games(game_id) ON DELETE CASCADE,
                config_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_entries (
                entry_id TEXT PRIMARY KEY,
                entry_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                snapshot_json JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_configs (
                config_id TEXT PRIMARY KEY,
                config_type TEXT NOT NULL,
                record_json JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                entry_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                entry_json JSONB NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflections_game ON reflections (game_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflections_player ON reflections (player_id)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_game_streams (
                game_id TEXT PRIMARY KEY REFERENCES games(game_id) ON DELETE CASCADE,
                game_revision BIGINT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_turn_commits (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                turn_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                result_json JSONB NOT NULL,
                committed_revision BIGINT NOT NULL,
                PRIMARY KEY (game_id, turn_id, idempotency_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_public_records (
                record_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                turn_id TEXT NOT NULL,
                committed_revision BIGINT NOT NULL,
                record_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_audit_records (
                audit_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                committed_revision BIGINT NOT NULL,
                record_json JSONB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_projection_outbox (
                outbox_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                committed_revision BIGINT NOT NULL,
                request_json JSONB NOT NULL,
                delivered_at TIMESTAMP NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_autonomous_audit_revision "
            "ON autonomous_audit_records (game_id, committed_revision)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_autonomous_outbox_revision "
            "ON autonomous_projection_outbox (game_id, committed_revision)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_serial_public_schedules (
                schedule_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                window_id TEXT NOT NULL,
                status TEXT NOT NULL,
                next_slot_ordinal BIGINT NOT NULL,
                active_turn_id TEXT,
                state_version BIGINT NOT NULL,
                schedule_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_managed_turns (
                turn_id TEXT PRIMARY KEY,
                schedule_id TEXT NOT NULL
                    REFERENCES autonomous_serial_public_schedules(schedule_id)
                    ON DELETE CASCADE,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                player_id TEXT NOT NULL,
                status TEXT NOT NULL,
                state_version BIGINT NOT NULL,
                turn_json JSONB NOT NULL,
                terminal_reason TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_open_serial_public_schedule "
            "ON autonomous_serial_public_schedules (game_id) "
            "WHERE status = 'open'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_managed_turn_schedule_status "
            "ON autonomous_managed_turns (schedule_id, status)"
        )
        key_duplicates = conn.execute("""
            SELECT
                schedule_id,
                turn_json #>> '{turn,idempotency_key}' AS idempotency_key,
                COUNT(*),
                array_agg(turn_id ORDER BY turn_id)
            FROM autonomous_managed_turns
            WHERE turn_json #>> '{turn,idempotency_key}' IS NOT NULL
            GROUP BY schedule_id, turn_json #>> '{turn,idempotency_key}'
            HAVING COUNT(*) > 1
            ORDER BY schedule_id, turn_json #>> '{turn,idempotency_key}'
            LIMIT 20
        """).fetchall()
        if key_duplicates:
            details = " | ".join(
                f"schedule_id={schedule_id}, idempotency_key={idempotency_key}, "
                f"count={count}, rows={row_ids}"
                for schedule_id, idempotency_key, count, row_ids in key_duplicates
            )
            raise PostgresSchemaMigrationError(
                "PostgreSQL managed turn idempotency migration blocked by "
                "duplicate schedule keys; resolve or quarantine these rows "
                f"before retrying: {details}",
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_managed_turn_schedule_idempotency_key "
            "ON autonomous_managed_turns "
            "(schedule_id, (turn_json #>> '{turn,idempotency_key}'))"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_dispatch_attempts (
                dispatch_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                turn_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                operation_kind TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                provider_idempotency_key TEXT NOT NULL,
                recovery_policy TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                lease_hash TEXT NOT NULL,
                view_fingerprint TEXT NOT NULL,
                deadline TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                state_version BIGINT NOT NULL,
                reason_code TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_dispatch_results (
                result_id TEXT PRIMARY KEY,
                dispatch_id TEXT NOT NULL UNIQUE
                    REFERENCES autonomous_dispatch_attempts(dispatch_id)
                    ON DELETE CASCADE,
                request_hash TEXT NOT NULL,
                lease_hash TEXT NOT NULL,
                result_hash TEXT NOT NULL,
                result_kind TEXT NOT NULL,
                outcome TEXT NOT NULL,
                result_json JSONB NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_autonomous_dispatch_executor_provider_key "
            "ON autonomous_dispatch_attempts (executor_id, provider_idempotency_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_autonomous_dispatch_game_status_created "
            "ON autonomous_dispatch_attempts (game_id, status, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_autonomous_dispatch_game_turn_created "
            "ON autonomous_dispatch_attempts (game_id, turn_id, created_at)"
        )

    @staticmethod
    def _upgrade_event_integrity_v3(conn: Any) -> None:
        """预检历史重复项，安全创建事件唯一索引并最后记录 v3。"""
        seq_duplicates = conn.execute("""
            SELECT game_id, seq, COUNT(*), array_agg(id ORDER BY id)
            FROM events
            GROUP BY game_id, seq
            HAVING COUNT(*) > 1
            ORDER BY game_id, seq
            LIMIT 20
        """).fetchall()
        event_id_duplicates = conn.execute("""
            SELECT
                game_id,
                event_json->>'event_id',
                COUNT(*),
                array_agg(id ORDER BY id)
            FROM events
            WHERE event_json->>'event_id' IS NOT NULL
            GROUP BY game_id, event_json->>'event_id'
            HAVING COUNT(*) > 1
            ORDER BY game_id, event_json->>'event_id'
            LIMIT 20
        """).fetchall()
        if seq_duplicates or event_id_duplicates:
            details: list[str] = []
            details.extend(
                f"game_id={game_id}, seq={seq}, count={count}, rows={row_ids}"
                for game_id, seq, count, row_ids in seq_duplicates
            )
            details.extend(
                "game_id="
                f"{game_id}, event_id={event_id}, count={count}, rows={row_ids}"
                for game_id, event_id, count, row_ids in event_id_duplicates
            )
            raise PostgresSchemaMigrationError(
                "PostgreSQL event integrity v3 migration blocked by duplicate "
                "audit rows; resolve or quarantine these rows explicitly before "
                "retrying: " + " | ".join(details)
            )

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_game_seq "
            "ON events (game_id, seq)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_game_event_id "
            "ON events (game_id, (event_json->>'event_id')) "
            "WHERE event_json->>'event_id' IS NOT NULL"
        )
        indexes = conn.execute("""
            SELECT
                to_regclass('uq_events_game_seq'),
                to_regclass('uq_events_game_event_id')
        """).fetchone()
        if indexes is None or any(index is None for index in indexes):
            raise PostgresSchemaMigrationError(
                "PostgreSQL event integrity v3 indexes were not created"
            )
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING"
        )
