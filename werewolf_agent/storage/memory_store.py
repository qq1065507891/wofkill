# -*- coding: utf-8 -*-
"""
功能描述：内存游戏仓库，支持旧数据接口、自主玩家原子 CommitTurn 与 durable dispatch。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-29
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import normalize_resolution_batch_fields
from werewolf_agent.player_agents.contracts.dispatch import (
    DispatchAttempt,
    DispatchResultDisposition,
    DispatchResultRecord,
    DispatchStatus,
)
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
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


class InMemoryGameRepository:
    """Pure in-memory implementation of GameRepository.

    Data is lost on process exit. Used by default in API app
    when no persistent repository is provided.
    """

    def __init__(self) -> None:
        self._games: dict[str, GameState] = {}
        self._events: dict[str, list[GameEvent]] = {}
        self._deaths: dict[str, list[Death]] = {}
        self._usage: dict[str, list[dict[str, Any]]] = {}
        self._evaluations: dict[str, dict[str, Any]] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        self._custom_configs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._autonomous_revision_by_game: dict[str, int] = {}
        self._autonomous_commits: dict[tuple[str, str, str], CommitResult] = {}
        self._autonomous_public_records: dict[str, PublicSpeechRecord] = {}
        self._autonomous_audits: dict[str, tuple[str, dict[str, Any]]] = {}
        self._autonomous_outbox: dict[str, ProjectionOutboxRecord] = {}
        self._autonomous_outbox_game_ids: dict[str, str] = {}
        self._dispatch_attempts: dict[str, DispatchAttempt] = {}
        self._dispatch_results: dict[str, DispatchResultRecord] = {}
        self._dispatch_key_index: dict[tuple[str, str], str] = {}
        self._serial_public_schedules: dict[str, SerialPublicSchedule] = {}
        self._managed_agent_turns: dict[str, ManagedAgentTurn] = {}
        self._active_schedule_by_game: dict[str, str] = {}

    def save_game(self, state: GameState) -> None:
        with self._lock:
            validate_game_state_save(self._games.get(state.game_id), state)
            self._games[state.game_id] = state

    def load_game(self, game_id: str) -> GameState | None:
        with self._lock:
            return self._games.get(game_id)

    def append_events(self, game_id: str, events: list[GameEvent]) -> None:
        """保留事件的完整 V1/V2 数据类，不经过降级序列化。"""
        with self._lock:
            validate_game_aborted_append(
                game_id,
                self._games.get(game_id),
                self._events.get(game_id, []),
                events,
            )
            if game_id not in self._events:
                self._events[game_id] = []
            self._events[game_id].extend(events)

    def load_events(self, game_id: str) -> list[GameEvent]:
        with self._lock:
            return list(self._events.get(game_id, []))

    # -- Autonomous serial-public turns -----------------------------------

    def supports_autonomous_turns(self) -> bool:
        """声明内存仓储提供串行公开托管回合能力。"""
        return True

    def create_serial_public_schedule(
        self,
        schedule: SerialPublicSchedule,
    ) -> SerialPublicSchedule:
        """创建一个新的公开调度并建立游戏级活动索引。"""
        with self._lock:
            if schedule.game_id not in self._games:
                raise AutonomousTurnTransactionError(
                    "game is required before creating a schedule",
                )
            if schedule.schedule_id in self._serial_public_schedules:
                raise AutonomousTurnTransactionError(
                    "schedule already exists",
                )
            if (
                schedule.status is SerialPublicScheduleStatus.OPEN
                and schedule.game_id in self._active_schedule_by_game
            ):
                raise AutonomousTurnTransactionError(
                    "game already has an open schedule",
                )
            if (
                schedule.status is SerialPublicScheduleStatus.OPEN
                and schedule.active_turn_id is not None
            ):
                raise InvalidScheduleTransition(
                    "invalid autonomous turn transition",
                )
            stored = schedule.model_copy(deep=True)
            self._serial_public_schedules[stored.schedule_id] = stored
            if stored.status is SerialPublicScheduleStatus.OPEN:
                self._active_schedule_by_game[stored.game_id] = stored.schedule_id
            return stored.model_copy(deep=True)

    def load_serial_public_schedule(
        self,
        schedule_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            schedule = self._serial_public_schedules.get(schedule_id)
            return schedule.model_copy(deep=True) if schedule is not None else None

    def load_active_serial_public_schedule(
        self,
        game_id: str,
    ) -> SerialPublicSchedule | None:
        with self._lock:
            schedule_id = self._active_schedule_by_game.get(game_id)
            if schedule_id is None:
                return None
            schedule = self._serial_public_schedules.get(schedule_id)
            if schedule is None or schedule.status is not SerialPublicScheduleStatus.OPEN:
                return None
            return schedule.model_copy(deep=True)

    def list_open_serial_public_schedules(
        self,
    ) -> tuple[SerialPublicSchedule, ...]:
        with self._lock:
            schedules = [
                schedule
                for schedule in self._serial_public_schedules.values()
                if schedule.status is SerialPublicScheduleStatus.OPEN
            ]
            schedules.sort(key=lambda item: (item.created_at, item.schedule_id))
            return tuple(schedule.model_copy(deep=True) for schedule in schedules)

    def load_managed_turn(self, turn_id: str) -> ManagedAgentTurn | None:
        with self._lock:
            managed = self._managed_agent_turns.get(turn_id)
            return managed.model_copy(deep=True) if managed is not None else None

    def admit_serial_public_turn(
        self,
        schedule_id: str,
        expected_schedule_version: int,
        admission: TurnAdmission,
    ) -> ManagedAgentTurn:
        """在 schedule CAS 成功后原子发布一个新的活动托管回合。"""
        with self._lock:
            schedule = self._serial_public_schedules.get(schedule_id)
            if schedule is None:
                raise ScheduleNotFound("schedule not found")
            if schedule.state_version != expected_schedule_version:
                raise ScheduleStateConflict("schedule state version conflict")
            try:
                now = max(datetime.now(timezone.utc), schedule.updated_at)
                updated_schedule, managed = prepare_serial_public_admission(
                    schedule,
                    admission,
                    now,
                )
                if managed.turn.turn_id in self._managed_agent_turns:
                    raise AutonomousTurnTransactionError(
                        "managed turn already exists",
                    )
            except (
                InvalidTurnAdmission,
                InvalidScheduleTransition,
                AutonomousTurnTransactionError,
            ):
                raise
            except Exception as exc:
                raise AutonomousTurnTransactionError(
                    "autonomous turn admission transaction failed",
                ) from exc

            self._managed_agent_turns[managed.turn.turn_id] = managed.model_copy(
                deep=True,
            )
            self._serial_public_schedules[schedule_id] = updated_schedule.model_copy(
                deep=True,
            )
            self._active_schedule_by_game[schedule.game_id] = schedule_id
            return managed.model_copy(deep=True)

    def transition_active_turn(
        self,
        turn_id: str,
        expected_turn_version: int,
        next_status: AgentTurnStatus,
    ) -> ManagedAgentTurn:
        """使用托管回合 CAS 版本推进一个非终态活动回合。"""
        with self._lock:
            managed = self._managed_agent_turns.get(turn_id)
            if managed is None:
                raise ManagedTurnNotFound("managed turn not found")
            if managed.state_version != expected_turn_version:
                raise TurnStateConflict("managed turn state version conflict")
            schedule = self._serial_public_schedules.get(managed.schedule_id)
            if schedule is None:
                raise ScheduleNotFound("schedule not found")
            try:
                updated = prepare_active_transition(schedule, managed, next_status)
            except InvalidScheduleTransition:
                raise
            except Exception as exc:
                raise AutonomousTurnTransactionError(
                    "autonomous turn transition transaction failed",
                ) from exc
            self._managed_agent_turns[turn_id] = updated.model_copy(deep=True)
            return updated.model_copy(deep=True)

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
        """原子终结活动回合并推进、替换或关闭公开调度。"""
        with self._lock:
            schedule = self._serial_public_schedules.get(schedule_id)
            if schedule is None:
                raise ScheduleNotFound("schedule not found")
            if schedule.state_version != expected_schedule_version:
                raise ScheduleStateConflict("schedule state version conflict")
            managed = self._managed_agent_turns.get(turn_id)
            if managed is None:
                raise ManagedTurnNotFound("managed turn not found")
            if managed.state_version != expected_turn_version:
                raise TurnStateConflict("managed turn state version conflict")
            try:
                now = max(datetime.now(timezone.utc), schedule.updated_at)
                updated_schedule, updated_turn = prepare_active_finish(
                    schedule,
                    managed,
                    terminal_status,
                    disposition,
                    reason_code=reason_code,
                    now=now,
                )
            except InvalidScheduleTransition:
                raise
            except Exception as exc:
                raise AutonomousTurnTransactionError(
                    "autonomous turn finish transaction failed",
                ) from exc

            self._managed_agent_turns[turn_id] = updated_turn.model_copy(deep=True)
            self._serial_public_schedules[schedule_id] = updated_schedule.model_copy(
                deep=True,
            )
            if updated_schedule.status is SerialPublicScheduleStatus.OPEN:
                self._active_schedule_by_game[updated_schedule.game_id] = schedule_id
            else:
                self._active_schedule_by_game.pop(updated_schedule.game_id, None)
            return updated_schedule.model_copy(deep=True)

    # -- Autonomous CommitTurn --------------------------------------------

    def supports_autonomous_commit(self) -> bool:
        """声明内存仓储可以提供完整的自主提交语义。"""
        return True

    def load_game_revision(self, game_id: str) -> int:
        with self._lock:
            return self._autonomous_revision(game_id)

    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]:
        with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._autonomous_outbox.values()
                if self._autonomous_outbox_game_ids.get(record.outbox_id) == game_id
            ]

    def commit_turn(self, request: CommitTurnRequest) -> CommitResult:
        with self._lock:
            events_snapshot = {
                game_id: list(events) for game_id, events in self._events.items()
            }
            revision_snapshot = dict(self._autonomous_revision_by_game)
            commits_snapshot = dict(self._autonomous_commits)
            public_records_snapshot = dict(self._autonomous_public_records)
            audits_snapshot = dict(self._autonomous_audits)
            outbox_snapshot = dict(self._autonomous_outbox)
            outbox_game_ids_snapshot = dict(self._autonomous_outbox_game_ids)

            try:
                digest = request_hash(request)
                key = (request.game_id, request.turn_id, request.idempotency_key)
                existing = self._autonomous_commits.get(key)
                if existing is not None:
                    if existing.request_hash != digest:
                        raise IdempotencyConflictError(
                            "idempotency key conflicts with an existing proposal",
                        )
                    return existing.model_copy(update={"replayed": True})

                if request.game_id not in self._games:
                    raise CommitTransactionError(
                        f"game does not exist: {request.game_id}",
                    )
                current = self._autonomous_revision(request.game_id)
                event_head = self._autonomous_event_head(request.game_id)
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
                    request.game_id,
                    request.event,
                    next_revision,
                )
                record = bind_public_record(request.public_record, next_revision)
                result = build_commit_result(
                    request,
                    digest,
                    next_revision,
                    event,
                    record,
                )
                self._check_autonomous_ids(request, record)

                self._events.setdefault(request.game_id, []).append(event)
                self._autonomous_revision_by_game[request.game_id] = next_revision
                if record is not None:
                    self._autonomous_public_records[record.record_id] = record
                for audit in request.critical_audit_records:
                    self._autonomous_audits[audit.audit_id] = (
                        request.game_id,
                        audit.model_dump(mode="json"),
                    )
                for outbox in request.projection_outbox_records:
                    self._autonomous_outbox[outbox.outbox_id] = outbox
                    self._autonomous_outbox_game_ids[outbox.outbox_id] = (
                        request.game_id
                    )
                self._autonomous_commits[key] = result
                return result
            except Exception as exc:
                # 内存仓储通过整体恢复事务快照实现零部分发布。
                self._events = events_snapshot
                self._autonomous_revision_by_game = revision_snapshot
                self._autonomous_commits = commits_snapshot
                self._autonomous_public_records = public_records_snapshot
                self._autonomous_audits = audits_snapshot
                self._autonomous_outbox = outbox_snapshot
                self._autonomous_outbox_game_ids = outbox_game_ids_snapshot
                if isinstance(
                    exc,
                    (
                        StaleCommitError,
                        IdempotencyConflictError,
                        CommitTransactionError,
                    ),
                ):
                    raise
                raise CommitTransactionError(
                    "autonomous CommitTurn transaction failed",
                ) from exc

    def _autonomous_revision(self, game_id: str) -> int:
        current = self._autonomous_revision_by_game.get(game_id)
        if current is not None:
            return current
        return self._autonomous_event_head(game_id)

    def _autonomous_event_head(self, game_id: str) -> int:
        stored_events = self._events.get(game_id, [])
        state = self._games.get(game_id)
        state_events = state.events if state is not None else []
        explicit_revision = max(
            (
                event.sequence_number
                for event in [*stored_events, *state_events]
                if isinstance(event.sequence_number, int)
            ),
            default=0,
        )
        return max(explicit_revision, len(stored_events), len(state_events))

    def _check_autonomous_ids(
        self,
        request: CommitTurnRequest,
        record: PublicSpeechRecord | None,
    ) -> None:
        if record is not None and record.record_id in self._autonomous_public_records:
            raise CommitTransactionError(
                f"public record already exists: {record.record_id}",
            )
        if any(
            audit.audit_id in self._autonomous_audits
            for audit in request.critical_audit_records
        ):
            raise CommitTransactionError("audit record already exists")
        if any(
            outbox.outbox_id in self._autonomous_outbox
            for outbox in request.projection_outbox_records
        ):
            raise CommitTransactionError("projection outbox record already exists")

    # -- Durable dispatch ---------------------------------------------------

    def supports_durable_dispatch(self) -> bool:
        """声明内存仓储提供 durable dispatch 的完整状态机。"""
        with self._lock:
            return True

    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt:
        """在外部网络 I/O 前持久化一个新的 PENDING dispatch 意图。"""
        with self._lock:
            if (
                attempt.status is not DispatchStatus.PENDING
                or attempt.state_version != 0
            ):
                raise DispatchInvalidTransition(
                    "new dispatch must start in PENDING at version 0",
                )
            if attempt.game_id not in self._games:
                raise DispatchTransactionError(
                    f"game does not exist: {attempt.game_id}",
                )
            if attempt.dispatch_id in self._dispatch_attempts:
                raise DispatchIdempotencyConflict(attempt.dispatch_id)
            key = (attempt.executor_id, attempt.provider_idempotency_key)
            if key in self._dispatch_key_index:
                raise DispatchIdempotencyConflict(
                    f"provider idempotency key already exists: {key}",
                )
            self._assert_dispatch_allowed_unlocked(attempt.game_id)
            stored = attempt.model_copy(deep=True)
            self._dispatch_attempts[stored.dispatch_id] = stored
            self._dispatch_key_index[key] = stored.dispatch_id
            return stored.model_copy(deep=True)

    def mark_dispatching(
        self,
        dispatch_id: str,
        expected_version: int,
    ) -> DispatchAttempt:
        with self._lock:
            return self._transition_dispatch(
                dispatch_id,
                expected_version,
                {DispatchStatus.PENDING},
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
                {DispatchStatus.DISPATCHING},
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
                {DispatchStatus.PENDING, DispatchStatus.DISPATCHING},
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
                {DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED},
                DispatchStatus.UNKNOWN_OUTCOME,
                reason_code,
            )

    def _transition_dispatch(
        self,
        dispatch_id: str,
        expected_version: int,
        allowed_statuses: set[DispatchStatus],
        target_status: DispatchStatus,
        reason_code: str | None = None,
    ) -> DispatchAttempt:
        """在 RLock 保护下执行一次严格的 dispatch CAS 状态迁移。"""
        with self._lock:
            attempt = self._dispatch_attempts.get(dispatch_id)
            if attempt is None:
                raise DispatchNotFound(dispatch_id)
            if attempt.state_version != expected_version:
                raise DispatchStateConflict(dispatch_id)
            if attempt.status not in allowed_statuses:
                raise DispatchInvalidTransition(dispatch_id)
            updated = attempt.model_copy(
                deep=True,
                update={
                    "status": target_status,
                    "reason_code": reason_code,
                    "updated_at": datetime.now(timezone.utc),
                    "state_version": attempt.state_version + 1,
                },
            )
            self._dispatch_attempts[dispatch_id] = updated
            return updated.model_copy(deep=True)

    def record_result(
        self,
        dispatch_id: str,
        expected_version: int,
        result: DispatchResultRecord,
    ) -> DispatchResultDisposition:
        """原子写入一次 provider 结果并推进 attempt 状态。"""
        with self._lock:
            attempts_snapshot = dict(self._dispatch_attempts)
            results_snapshot = dict(self._dispatch_results)
            key_index_snapshot = dict(self._dispatch_key_index)
            try:
                attempt = self._dispatch_attempts.get(dispatch_id)
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

                prior = self._dispatch_results.get(dispatch_id)
                if prior is not None:
                    if prior == result:
                        return DispatchResultDisposition.REPLAYED
                    raise DispatchResultConflict(dispatch_id)
                if attempt.status in {
                    DispatchStatus.CANCELLED,
                    DispatchStatus.UNKNOWN_OUTCOME,
                }:
                    return DispatchResultDisposition.DISCARDED_LATE
                if attempt.status is not DispatchStatus.DISPATCHED:
                    raise DispatchInvalidTransition(dispatch_id)
                if any(
                    stored.result_id == result.result_id
                    and stored.dispatch_id != dispatch_id
                    for stored in self._dispatch_results.values()
                ):
                    raise DispatchResultConflict(dispatch_id)

                # DispatchResultRecord 的 payload 内部是 mappingproxy，Pydantic
                # 的原生 deep copy 无法 pickle；从 round-trip 数据重建同等防御副本。
                stored_result = DispatchResultRecord.model_validate(
                    result.model_dump(round_trip=True),
                )
                updated_attempt = attempt.model_copy(
                    deep=True,
                    update={
                        "status": DispatchStatus.RESULT_RECORDED,
                        "state_version": attempt.state_version + 1,
                        "reason_code": None,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                self._dispatch_results[dispatch_id] = stored_result
                self._dispatch_attempts[dispatch_id] = updated_attempt
                return DispatchResultDisposition.RECORDED
            except Exception as exc:
                # 两张表与幂等索引必须一起恢复，避免结果/状态只写入一半。
                self._dispatch_attempts = attempts_snapshot
                self._dispatch_results = results_snapshot
                self._dispatch_key_index = key_index_snapshot
                if isinstance(
                    exc,
                    (
                        DispatchNotFound,
                        DispatchStateConflict,
                        DispatchInvalidTransition,
                        DispatchResultConflict,
                        DispatchLeaseMismatch,
                    ),
                ):
                    raise
                raise DispatchTransactionError(
                    "durable dispatch result transaction failed",
                ) from exc

    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None:
        with self._lock:
            attempt = self._dispatch_attempts.get(dispatch_id)
            return attempt.model_copy(deep=True) if attempt is not None else None

    def list_dispatches_for_turn(
        self,
        game_id: str,
        turn_id: str,
    ) -> list[DispatchAttempt]:
        """按 game/turn 精确列出所有 dispatch，并保持稳定顺序。"""
        with self._lock:
            attempts = [
                attempt
                for attempt in self._dispatch_attempts.values()
                if attempt.game_id == game_id and attempt.turn_id == turn_id
            ]
            attempts.sort(key=lambda item: (item.created_at, item.dispatch_id))
            return [attempt.model_copy(deep=True) for attempt in attempts]

    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]:
        with self._lock:
            attempts = [
                attempt
                for attempt in self._dispatch_attempts.values()
                if attempt.game_id == game_id
                and attempt.status
                in {DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED}
            ]
            attempts.sort(key=lambda item: (item.created_at, item.dispatch_id))
            return [attempt.model_copy(deep=True) for attempt in attempts]

    def assert_dispatch_allowed(self, game_id: str) -> None:
        with self._lock:
            self._assert_dispatch_allowed_unlocked(game_id)

    def _assert_dispatch_allowed_unlocked(self, game_id: str) -> None:
        if any(
            attempt.game_id == game_id
            and attempt.status
            in {DispatchStatus.DISPATCHING, DispatchStatus.DISPATCHED}
            for attempt in self._dispatch_attempts.values()
        ):
            raise DispatchRecoveryBlocked(game_id)

    def save_deaths(self, game_id: str, deaths: list[Death]) -> None:
        self._deaths[game_id] = list(deaths)

    def load_deaths(self, game_id: str) -> list[Death]:
        return [
            Death(**normalize_resolution_batch_fields(death.__dict__))
            for death in self._deaths.get(game_id, [])
        ]

    def save_model_usage(self, game_id: str, record: dict[str, Any]) -> None:
        if game_id not in self._usage:
            self._usage[game_id] = []
        self._usage[game_id].append(dict(record))

    def load_model_usage(self, game_id: str) -> list[dict[str, Any]]:
        return list(self._usage.get(game_id, []))

    def save_evaluation(self, game_id: str, result: dict[str, Any]) -> None:
        self._evaluations[game_id] = dict(result)

    def load_evaluation(self, game_id: str) -> dict[str, Any] | None:
        return self._evaluations.get(game_id)

    def save_config_snapshot(self, game_id: str, config: dict[str, Any]) -> None:
        self._configs[game_id] = dict(config)

    def load_config_snapshot(self, game_id: str) -> dict[str, Any] | None:
        return self._configs.get(game_id)

    def save_custom_config(self, record: dict[str, Any]) -> None:
        self._custom_configs[str(record["config_id"])] = dict(record)

    def load_custom_config(self, config_id: str) -> dict[str, Any] | None:
        record = self._custom_configs.get(config_id)
        return dict(record) if record is not None else None

    def list_custom_configs(self, config_type: str | None = None) -> list[dict[str, Any]]:
        records = list(self._custom_configs.values())
        if config_type is not None:
            records = [record for record in records if record.get("config_type") == config_type]
        return [dict(record) for record in records]

    def list_games(self) -> list[GameState]:
        return list(self._games.values())

    def delete_game(self, game_id: str) -> None:
        """删除游戏及其 legacy 与自主提交关联数据。"""
        with self._lock:
            self._games.pop(game_id, None)
            self._events.pop(game_id, None)
            self._deaths.pop(game_id, None)
            self._usage.pop(game_id, None)
            self._evaluations.pop(game_id, None)
            self._configs.pop(game_id, None)
            self._autonomous_revision_by_game.pop(game_id, None)
            schedule_ids = {
                schedule_id
                for schedule_id, schedule in self._serial_public_schedules.items()
                if schedule.game_id == game_id
            }
            for schedule_id in schedule_ids:
                self._serial_public_schedules.pop(schedule_id, None)
            self._managed_agent_turns = {
                turn_id: managed
                for turn_id, managed in self._managed_agent_turns.items()
                if managed.turn.game_id != game_id
                and managed.schedule_id not in schedule_ids
            }
            self._active_schedule_by_game.pop(game_id, None)
            self._autonomous_commits = {
                key: result
                for key, result in self._autonomous_commits.items()
                if key[0] != game_id
            }
            self._autonomous_public_records = {
                record_id: record
                for record_id, record in self._autonomous_public_records.items()
                if record.game_id != game_id
            }
            self._autonomous_audits = {
                audit_id: stored
                for audit_id, stored in self._autonomous_audits.items()
                if stored[0] != game_id
            }
            for outbox_id, stored_game_id in tuple(
                self._autonomous_outbox_game_ids.items(),
            ):
                if stored_game_id == game_id:
                    self._autonomous_outbox_game_ids.pop(outbox_id, None)
                    self._autonomous_outbox.pop(outbox_id, None)
            dispatch_ids = {
                dispatch_id
                for dispatch_id, attempt in self._dispatch_attempts.items()
                if attempt.game_id == game_id
            }
            for dispatch_id in dispatch_ids:
                self._dispatch_attempts.pop(dispatch_id, None)
                self._dispatch_results.pop(dispatch_id, None)
            self._dispatch_key_index = {
                key: dispatch_id
                for key, dispatch_id in self._dispatch_key_index.items()
                if dispatch_id not in dispatch_ids
            }

    # -- RAG entries ---------------------------------------------------------

    def __init_rag(self) -> None:
        if not hasattr(self, "_rag_entries"):
            self._rag_entries: dict[str, dict[str, Any]] = {}

    def save_rag_entries(self, entries: list[dict[str, Any]]) -> None:
        self.__init_rag()
        for entry in entries:
            entry_id = entry.get("entry_id", "")
            self._rag_entries[entry_id] = dict(entry)

    def load_rag_entries(self) -> list[dict[str, Any]]:
        self.__init_rag()
        return list(self._rag_entries.values())

    def delete_rag_entry(self, entry_id: str) -> None:
        self.__init_rag()
        self._rag_entries.pop(entry_id, None)

    # -- Memory snapshots ----------------------------------------------------

    def __init_memory(self) -> None:
        if not hasattr(self, "_memory_snapshots"):
            self._memory_snapshots: dict[str, dict[str, Any]] = {}

    def save_memory_snapshot(self, snapshot_id: str, data: dict[str, Any]) -> None:
        self.__init_memory()
        self._memory_snapshots[snapshot_id] = dict(data)

    def load_memory_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        self.__init_memory()
        return self._memory_snapshots.get(snapshot_id)

    def list_memory_snapshots(self) -> list[dict[str, Any]]:
        self.__init_memory()
        return [{"snapshot_id": sid} for sid in self._memory_snapshots]

    def delete_memory_snapshot(self, snapshot_id: str) -> None:
        self.__init_memory()
        self._memory_snapshots.pop(snapshot_id, None)

    # -- Reflections (S4 post-review-v2) -------------------------------------

    def __init_reflections(self) -> None:
        if not hasattr(self, "_reflections"):
            self._reflections: dict[str, dict[str, Any]] = {}

    def save_reflection(self, entry: dict[str, Any]) -> None:
        self.__init_reflections()
        entry_id = str(entry.get("entry_id", ""))
        self._reflections[entry_id] = dict(entry)

    def load_reflection(self, entry_id: str) -> dict[str, Any] | None:
        self.__init_reflections()
        record = self._reflections.get(entry_id)
        return dict(record) if record is not None else None

    def load_reflections_by_game(self, game_id: str) -> list[dict[str, Any]]:
        self.__init_reflections()
        return [
            dict(entry)
            for entry in self._reflections.values()
            if entry.get("game_id") == game_id
        ]

    def load_reflections_by_player(self, player_id: str) -> list[dict[str, Any]]:
        self.__init_reflections()
        return [
            dict(entry)
            for entry in self._reflections.values()
            if entry.get("player_id") == player_id
        ]

    def load_all_reflections(self) -> list[dict[str, Any]]:
        self.__init_reflections()
        return [dict(entry) for entry in self._reflections.values()]

    def delete_reflection(self, entry_id: str) -> None:
        self.__init_reflections()
        self._reflections.pop(entry_id, None)
