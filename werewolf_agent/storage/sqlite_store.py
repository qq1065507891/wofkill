# -*- coding: utf-8 -*-
"""
功能描述：SQLite 游戏仓库，支持 GameEvent 与死亡批次 V2、V1 只读兼容。
作者: Project contributors
创建日期：2025-01-15
修改日期：2026-07-16
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
import threading
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.core.resolution_batches import (
    normalize_resolution_batch_fields,
    serialize_resolution_batch_fields,
)
from werewolf_agent.runtime.event_metadata import (
    deserialize_game_event,
    serialize_game_event,
    serialize_legacy_event_payload,
)
from werewolf_agent.runtime.game_termination import (
    validate_aborted_game,
    validate_game_aborted_append,
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
"""



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
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        _ensure_event_schema_v2(self._conn)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SqliteGameRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- Game state --------------------------------------------------------

    def save_game(self, state: GameState) -> None:
        if state.status == "aborted":
            validate_aborted_game(state)
        with self._lock:
            data = _serialize_game_state(state)
            self._conn.execute(
                "INSERT OR REPLACE INTO games (game_id, state_json) VALUES (?, ?)",
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
            validate_game_aborted_append(game_id, existing, events)
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
