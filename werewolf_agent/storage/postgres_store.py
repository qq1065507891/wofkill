"""PostgreSQL-backed game repository.

Uses JSONB storage for the current V1 game-state contract while keeping the
same repository interface as SQLite/InMemory. This is intended for Docker
Compose local production runs with the pgvector PostgreSQL image.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.storage.sqlite_store import _deserialize_game_state, _serialize_game_state


class PostgresGameRepository:
    """PostgreSQL implementation of GameRepository."""

    def __init__(self, dsn: str, *, initialize: bool = True) -> None:
        self._dsn = dsn
        self._lock = threading.Lock()
        self._conn: Any | None = None
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
            conn.execute(
                """
                INSERT INTO games (game_id, state_json)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (game_id) DO UPDATE SET state_json = EXCLUDED.state_json
                """,
                (state.game_id, _serialize_game_state(state)),
            )
            conn.commit()

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
            current = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE game_id = %s",
                (game_id,),
            ).fetchone()[0]
            for i, event in enumerate(events):
                conn.execute(
                    """
                    INSERT INTO events (game_id, seq, event_type, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (game_id, current + i + 1, event.type, json.dumps(event.payload, ensure_ascii=False)),
                )
            conn.commit()

    def load_events(self, game_id: str) -> list[GameEvent]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT event_type, payload_json FROM events WHERE game_id = %s ORDER BY seq",
                (game_id,),
            ).fetchall()
            return [
                GameEvent(type=row[0], payload=row[1] if isinstance(row[1], dict) else json.loads(row[1]))
                for row in rows
            ]

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
                    (game_id, death.player_id, json.dumps(asdict(death), ensure_ascii=False)),
                )
            conn.commit()

    def load_deaths(self, game_id: str) -> list[Death]:
        with self._lock:
            rows = self._ensure_connection().execute(
                "SELECT death_json FROM deaths WHERE game_id = %s",
                (game_id,),
            ).fetchall()
            return [Death(**(row[0] if isinstance(row[0], dict) else json.loads(row[0]))) for row in rows]

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
        conn.commit()
