# -*- coding: utf-8 -*-
"""
验证 SQLite fresh schema、版本迁移与 repository 自动升级的一致性。

作者: Project contributors
修改日期: 2026-07-15
"""

from __future__ import annotations

import re


def test_sqlite_schema_matches_migration_v1():
    """审查 U10: SqliteGameRepository._SCHEMA 与 migrations.py v1 表集合必须一致。"""
    from werewolf_agent.storage.sqlite_store import _SCHEMA
    from werewolf_agent.storage.migrations import MIGRATIONS
    v1 = next((m for m in MIGRATIONS if m.version == 1), None)
    assert v1 is not None, "no v1 migration"
    schema_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA))
    migration_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", v1.sql))
    assert schema_tables == migration_tables, (
        f"schema drift:\n  in _SCHEMA only: {schema_tables - migration_tables}\n"
        f"  in migration v1 only: {migration_tables - schema_tables}"
    )


def test_sqlite_schema_has_reflections_table():
    """S3 (post-review-v2): SQLite _SCHEMA 应含 reflections 表。"""
    from werewolf_agent.storage.sqlite_store import _SCHEMA
    assert "CREATE TABLE IF NOT EXISTS reflections" in _SCHEMA, (
        "SQLite _SCHEMA missing reflections table"
    )


def test_fresh_sqlite_schema_includes_nullable_event_json() -> None:
    from werewolf_agent.storage.sqlite_store import _SCHEMA

    events_table = _SCHEMA.split("CREATE TABLE IF NOT EXISTS events", 1)[1].split(");", 1)[0]
    assert "event_json TEXT" in events_table
    assert "event_json TEXT NOT NULL" not in events_table


def test_repository_upgrades_legacy_events_table_and_round_trips_v2(tmp_path) -> None:
    import sqlite3

    from werewolf_agent.core.models import GameEvent
    from werewolf_agent.runtime.event_metadata import stamp_new_events
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        INSERT INTO games (game_id, state_json) VALUES ('g1', '{}');
        INSERT INTO events (game_id, seq, event_type, payload_json)
        VALUES ('g1', 1, 'legacy', '{"visibility":"moderator_only"}');
    """)
    conn.close()

    repository = SqliteGameRepository(str(db_path))
    legacy = repository.load_events("g1")[0]
    current = stamp_new_events(
        "g1",
        [legacy],
        [legacy, GameEvent(type="current")],
    )[1]
    repository.append_events("g1", [current])
    loaded = repository.load_events("g1")

    assert legacy.schema_version is None
    assert legacy.payload["visibility"] == "moderator_only"
    assert loaded[-1] == current
    columns = {
        row[1]
        for row in repository._conn.execute("PRAGMA table_info(events)").fetchall()
    }
    assert "event_json" in columns
    repository.close()
