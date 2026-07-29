# -*- coding: utf-8 -*-
"""
功能描述：内存游戏仓库，支持旧数据接口和自主玩家原子 CommitTurn。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-29
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import threading
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import normalize_resolution_batch_fields
from werewolf_agent.player_agents.contracts.records import PublicSpeechRecord
from werewolf_agent.player_agents.contracts.transactions import (
    CommitResult,
    CommitTurnRequest,
    ProjectionOutboxRecord,
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
                raise CommitTransactionError(f"game does not exist: {request.game_id}")
            current = self._autonomous_revision(request.game_id)
            event_head = self._autonomous_event_head(request.game_id)
            if current != event_head:
                raise CommitTransactionError(
                    f"autonomous stream head {current} does not match event head {event_head}",
                )
            if request.base_game_revision != current:
                raise StaleCommitError(
                    f"expected revision {current}, got {request.base_game_revision}",
                )
            next_revision = current + 1
            event = build_committed_event(request.game_id, request.event, next_revision)
            record = bind_public_record(request.public_record, next_revision)
            result = build_commit_result(request, digest, next_revision, event, record)
            self._check_autonomous_ids(request, record)

            # 所有唯一性和 revision 检查通过后才一次性发布内存记录。
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
                self._autonomous_outbox_game_ids[outbox.outbox_id] = request.game_id
            self._autonomous_commits[key] = result
            return result

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
