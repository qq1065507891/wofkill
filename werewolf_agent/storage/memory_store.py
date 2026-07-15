# -*- coding: utf-8 -*-
"""
功能描述：内存游戏仓库，保留完整 GameEvent 并规范化 Death 批次。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-15
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import normalize_resolution_batch_fields


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

    def save_game(self, state: GameState) -> None:
        self._games[state.game_id] = state

    def load_game(self, game_id: str) -> GameState | None:
        return self._games.get(game_id)

    def append_events(self, game_id: str, events: list[GameEvent]) -> None:
        """保留事件的完整 V1/V2 数据类，不经过降级序列化。"""
        if game_id not in self._events:
            self._events[game_id] = []
        self._events[game_id].extend(events)

    def load_events(self, game_id: str) -> list[GameEvent]:
        return list(self._events.get(game_id, []))

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
        """删除游戏及其所有关联数据，包括自定义配置。"""
        self._games.pop(game_id, None)
        self._events.pop(game_id, None)
        self._deaths.pop(game_id, None)
        self._usage.pop(game_id, None)
        self._evaluations.pop(game_id, None)
        self._configs.pop(game_id, None)

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
