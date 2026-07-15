# -*- coding: utf-8 -*-
"""
验证 SQLite schema 迁移版本与 GameEvent V2 列升级。

作者: Project contributors
修改日期: 2026-07-15
"""

import os
import tempfile

from werewolf_agent.storage.migrations import MigrationManager


def test_migration_manager_tracks_version():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    assert mgr.current_version() == 0
    mgr.apply_all()
    assert mgr.current_version() >= 1
    mgr.close()
    os.unlink(db_path)


def test_migration_idempotent():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    mgr.apply_all()
    v1 = mgr.current_version()
    mgr.apply_all()
    assert mgr.current_version() == v1
    mgr.close()
    os.unlink(db_path)


def test_migration_adds_schema_version_table():
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    mgr.apply_all()
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "schema_version" in tables
    conn.close()
    mgr.close()
    os.unlink(db_path)


def test_migration_v2_adds_nullable_event_json_to_existing_events_table() -> None:
    db_path = tempfile.mktemp(suffix=".db")
    mgr = MigrationManager(db_path)
    mgr.apply_all()

    import sqlite3

    conn = sqlite3.connect(db_path)
    columns = {
        row[1]: row[3]
        for row in conn.execute("PRAGMA table_info(events)").fetchall()
    }
    assert "event_json" in columns
    assert columns["event_json"] == 0
    assert mgr.current_version() == 2
    conn.close()
    mgr.close()
    os.unlink(db_path)
