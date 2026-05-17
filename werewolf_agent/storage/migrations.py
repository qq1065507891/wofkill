"""SQLite schema migration system.

Provides versioned migrations so that existing databases can be upgraded
without losing data. Each migration is a (version, sql) pair.
"""

from __future__ import annotations
import sqlite3


class Migration:
    def __init__(self, version: int, description: str, sql: str) -> None:
        self.version = version
        self.description = description
        self.sql = sql


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="Initial schema: games, events, deaths, model_usage, evaluations, config_snapshots, rag_entries, memory_snapshots, schema_version",
        sql="""
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
        """,
    ),
]


class MigrationManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def current_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] or 0
        except sqlite3.OperationalError:
            return 0

    def apply_all(self) -> None:
        current = self.current_version()
        for migration in MIGRATIONS:
            if migration.version > current:
                self._conn.executescript(migration.sql)
                self._conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (migration.version, migration.description),
                )
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()
