# -*- coding: utf-8 -*-
"""
功能描述：SQLite schema 迁移系统，版本化升级并为事件保留 V1 读取列。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-15
使用示例：内部模块，无对外接口
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
        CREATE TABLE IF NOT EXISTS reflections (
            entry_id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            entry_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reflections_game ON reflections (game_id);
        CREATE INDEX IF NOT EXISTS idx_reflections_player ON reflections (player_id);
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
        """,
    ),
    Migration(
        version=2,
        description="Add nullable GameEvent V2 JSON storage",
        sql="""
        ALTER TABLE events ADD COLUMN event_json TEXT;
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
        """按版本号顺序应用所有未执行的迁移。

        每条迁移的 SQL 被拆分为独立语句逐条执行，确保事务语义正确。
        不使用 executescript（它会隐式 commit，破坏事务边界）。
        """
        current = self.current_version()
        for migration in MIGRATIONS:
            if migration.version > current:
                # 将多语句 SQL 拆分后逐条执行，保持单个事务
                statements = [s.strip() for s in migration.sql.split(";") if s.strip()]
                for stmt in statements:
                    self._conn.execute(stmt)
                self._conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (migration.version, migration.description),
                )
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MigrationManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
