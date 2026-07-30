# -*- coding: utf-8 -*-
"""
验证 SQLite fresh schema、版本迁移与 repository 自动升级的一致性。

作者: Project contributors
修改日期: 2026-07-31
"""

from __future__ import annotations

import re


def _create_pre_fence_database(path) -> None:
    """创建含完整历史 unfenced dispatch 的旧版 autonomous 表。"""

    import sqlite3

    from werewolf_agent.storage.sqlite_store import (
        _AUTONOMOUS_SCHEDULING_SCHEMA,
        _SCHEMA,
    )

    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.executescript(_AUTONOMOUS_SCHEDULING_SCHEMA)
    conn.execute(
        "CREATE TABLE autonomous_dispatch_attempts ("
        "dispatch_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, turn_id TEXT NOT NULL, "
        "actor_id TEXT NOT NULL, operation_kind TEXT NOT NULL, executor_id TEXT NOT NULL, "
        "provider_idempotency_key TEXT NOT NULL, recovery_policy TEXT NOT NULL, "
        "request_hash TEXT NOT NULL, lease_hash TEXT NOT NULL, "
        "view_fingerprint TEXT NOT NULL, deadline TEXT NOT NULL, status TEXT NOT NULL, "
        "state_version INTEGER NOT NULL, reason_code TEXT, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, FOREIGN KEY (game_id) REFERENCES games(game_id) "
        "ON DELETE CASCADE)",
    )
    conn.execute("INSERT INTO games (game_id, state_json) VALUES (?, ?)", ("game-1", "{}"))
    conn.execute(
        "INSERT INTO autonomous_dispatch_attempts ("
        "dispatch_id, game_id, turn_id, actor_id, operation_kind, executor_id, "
        "provider_idempotency_key, recovery_policy, request_hash, lease_hash, "
        "view_fingerprint, deadline, status, state_version, reason_code, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-dispatch",
            "game-1",
            "turn-1",
            "p01",
            "model",
            "legacy-provider",
            "legacy-key",
            "idempotent_lookup_or_reissue",
            "a" * 64,
            "a" * 64,
            "a" * 64,
            "2026-07-31T11:00:00+00:00",
            "pending",
            0,
            None,
            "2026-07-31T10:00:00+00:00",
            "2026-07-31T10:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


def test_sqlite_module_describes_serial_public_scheduling_responsibility() -> None:
    from werewolf_agent.storage import sqlite_store

    assert "串行公开调度" in (sqlite_store.__doc__ or "")


def test_sqlite_fresh_schema_has_nullable_active_turn_fence_column(tmp_path) -> None:
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    repository = SqliteGameRepository(str(tmp_path / "fresh.db"))
    columns = {
        row[1]: row
        for row in repository._conn.execute(
            "PRAGMA table_info(autonomous_dispatch_attempts)",
        ).fetchall()
    }

    assert "active_turn_fence_json" in columns
    assert columns["active_turn_fence_json"][3] == 0
    repository.close()


def test_sqlite_legacy_dispatch_schema_adds_nullable_fence_column(tmp_path) -> None:
    import sqlite3

    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    path = tmp_path / "legacy-dispatch.db"
    _create_pre_fence_database(path)
    with sqlite3.connect(path) as connection:
        columns_before = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(autonomous_dispatch_attempts)",
            ).fetchall()
        }
    assert "active_turn_fence_json" not in columns_before

    repository = SqliteGameRepository(str(path))

    columns_after = {
        row[1]: row
        for row in repository._conn.execute(
            "PRAGMA table_info(autonomous_dispatch_attempts)",
        ).fetchall()
    }

    assert repository.supports_active_turn_fence() is True
    assert "active_turn_fence_json" in columns_after
    assert columns_after["active_turn_fence_json"][3] == 0
    stored = repository.load_dispatch("legacy-dispatch")
    assert stored is not None
    assert stored.active_turn_fence is None
    repository.close()


def test_sqlite_uses_shared_utc_timestamp_serializer() -> None:
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    assert hasattr(SqliteGameRepository, "_utc_timestamp")
    assert not hasattr(SqliteGameRepository, "_dispatch_timestamp")


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
        "autonomous_dispatch_attempts",
        "autonomous_dispatch_results",
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


def test_fresh_sqlite_schema_includes_durable_dispatch_tables_and_indexes(tmp_path) -> None:
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    repository = SqliteGameRepository(str(tmp_path / "dispatch-schema.db"))
    tables = {
        row[0]
        for row in repository._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        ).fetchall()
    }
    indexes = {
        row[0]
        for row in repository._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'",
        ).fetchall()
    }
    assert {
        "autonomous_dispatch_attempts",
        "autonomous_dispatch_results",
    } <= tables
    assert {
        "uq_dispatch_executor_key",
        "idx_dispatch_game_status_created",
    } <= indexes
    repository.close()


def test_fresh_sqlite_schema_includes_autonomous_turn_tables_and_indexes(tmp_path) -> None:
    from werewolf_agent.storage.sqlite_store import SqliteGameRepository

    repository = SqliteGameRepository(str(tmp_path / "turn-schema.db"))
    tables = {
        row[0]
        for row in repository._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        ).fetchall()
    }
    indexes = {
        row[0]
        for row in repository._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'",
        ).fetchall()
    }
    assert {
        "autonomous_serial_public_schedules",
        "autonomous_managed_turns",
    } <= tables
    assert {
        "uq_open_serial_public_schedule",
        "idx_managed_turn_schedule_status",
        "uq_managed_turn_schedule_idempotency_key",
    } <= indexes
    repository.close()


def test_sqlite_rejects_duplicate_historical_idempotency_keys_during_initialization(
    tmp_path,
) -> None:
    """唯一索引升级前预检历史重复，避免暴露裸 IntegrityError。"""
    import sqlite3

    from tests.storage.test_autonomous_turns import _schedule
    from werewolf_agent.core.models import GameState
    from werewolf_agent.storage.sqlite_store import (
        SqliteGameRepository,
        SqliteSchemaMigrationError,
    )
    db_path = tmp_path / "duplicate-turn-history.db"
    repository = SqliteGameRepository(str(db_path))
    repository.save_game(GameState(game_id="game-1"))
    repository.create_serial_public_schedule(_schedule())
    repository.close()

    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX uq_managed_turn_schedule_idempotency_key")
    row = conn.execute(
        "SELECT schedule_json, created_at, updated_at FROM autonomous_serial_public_schedules",
    ).fetchone()
    assert row is not None
    _schedule_json, created_at, updated_at = row
    for turn_id in ("turn-history-1", "turn-history-2"):
        conn.execute(
            "INSERT INTO autonomous_managed_turns "
            "(turn_id, schedule_id, game_id, player_id, status, state_version, "
            "turn_json, terminal_reason, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id,
                "schedule-1",
                "game-1",
                "p01",
                "cancelled",
                1,
                '{"turn":{"idempotency_key":"historical-key"}}',
                None,
                created_at,
                updated_at,
            ),
        )
    conn.commit()
    conn.close()

    try:
        SqliteGameRepository(str(db_path))
    except SqliteSchemaMigrationError as exc:
        assert "schedule_id=schedule-1" in str(exc)
        assert "idempotency_key=historical-key" in str(exc)
        assert "turn-history-1" in str(exc)
    else:
        raise AssertionError("duplicate idempotency history was silently accepted")


def test_migration_manager_does_not_apply_durable_dispatch_schema(tmp_path) -> None:
    from werewolf_agent.storage.migrations import MIGRATIONS
    from werewolf_agent.storage.sqlite_store import _AUTONOMOUS_DISPATCH_SCHEMA

    assert "autonomous_dispatch_attempts" in _AUTONOMOUS_DISPATCH_SCHEMA
    assert all(
        "autonomous_dispatch_attempts" not in migration.sql
        and "autonomous_dispatch_results" not in migration.sql
        for migration in MIGRATIONS
    )


def test_migration_manager_does_not_apply_autonomous_turn_schema(tmp_path) -> None:
    from werewolf_agent.storage.migrations import MIGRATIONS, MigrationManager
    from werewolf_agent.storage.sqlite_store import _AUTONOMOUS_SCHEDULING_SCHEMA

    assert "autonomous_serial_public_schedules" in _AUTONOMOUS_SCHEDULING_SCHEMA
    assert all(
        "autonomous_serial_public_schedules" not in migration.sql
        and "autonomous_managed_turns" not in migration.sql
        for migration in MIGRATIONS
    )
    with MigrationManager(str(tmp_path / "legacy-turn-migrations.db")) as manager:
        manager.apply_all()
        tables = {
            row[0]
            for row in manager._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            ).fetchall()
        }
    assert not {
        "autonomous_serial_public_schedules",
        "autonomous_managed_turns",
    } & tables


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
