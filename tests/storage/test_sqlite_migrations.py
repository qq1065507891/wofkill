# -*- coding: utf-8 -*-
"""
验证 SQLite fresh schema、版本迁移与 repository 自动升级的一致性。

作者: Project contributors
修改日期: 2026-07-29
"""

from __future__ import annotations

import re


def test_sqlite_schema_matches_migration_v1():
    """legacy schema 必须保持与 MigrationManager v1 一致。"""
    from werewolf_agent.storage.migrations import MIGRATIONS
    from werewolf_agent.storage.sqlite_store import _SCHEMA
    v1 = next((m for m in MIGRATIONS if m.version == 1), None)
    assert v1 is not None, "no v1 migration"
    schema_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA))
    migration_tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", v1.sql))
    autonomous_tables = {
        "autonomous_game_streams",
        "autonomous_turn_commits",
        "autonomous_public_records",
        "autonomous_audit_records",
        "autonomous_projection_outbox",
    }
    assert schema_tables - autonomous_tables == migration_tables, (
        "legacy schema drift:\n"
        f"  in _SCHEMA only: {schema_tables - autonomous_tables - migration_tables}\n"
        f"  in migration v1 only: {migration_tables - schema_tables}"
    )


def test_migration_manager_does_not_apply_autonomous_schema(tmp_path) -> None:
    from werewolf_agent.storage.migrations import MigrationManager

    with MigrationManager(str(tmp_path / "legacy-migrations.db")) as manager:
        manager.apply_all()
        tables = {
            row[0]
            for row in manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            ).fetchall()
        }

    assert not {
        "autonomous_game_streams",
        "autonomous_turn_commits",
        "autonomous_public_records",
        "autonomous_audit_records",
        "autonomous_projection_outbox",
    } & tables


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


def test_fresh_sqlite_schema_includes_autonomous_commit_tables(tmp_path) -> None:
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    repository = SqliteGameRepository(str(tmp_path / "autonomous-schema.db"))
    tables = {
        row[0]
        for row in repository._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        ).fetchall()
    }
    assert {
        "autonomous_game_streams",
        "autonomous_turn_commits",
        "autonomous_public_records",
        "autonomous_audit_records",
        "autonomous_projection_outbox",
    } <= tables
    repository.close()


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


def test_sqlite_dual_write_keeps_private_visibility_for_legacy_reader(tmp_path) -> None:
    import json

    from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
    from werewolf_agent.core.models import GameEvent, GameState
    from werewolf_agent.runtime.event_metadata import stamp_new_events
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    repository = SqliteGameRepository(str(tmp_path / "events.db"))
    event = stamp_new_events(
        "g1",
        [],
        [GameEvent(
            type="seer_check",
            payload={"target_id": "p02"},
            visibility=EventVisibility.SEER_PRIVATE,
        )],
    )[0]

    repository.save_game(GameState(game_id="g1"))
    repository.append_events("g1", [event])

    event_type, payload_json, event_json = repository._conn.execute(
        "SELECT event_type, payload_json, event_json FROM events WHERE game_id = ?",
        ("g1",),
    ).fetchone()
    legacy_event = GameEvent(type=event_type, payload=json.loads(payload_json))
    current_record = json.loads(event_json)

    assert event_visibility(legacy_event) is EventVisibility.SEER_PRIVATE
    assert current_record["visibility"] == "seer_private"
    assert "visibility" not in current_record["payload"]
    repository.close()
