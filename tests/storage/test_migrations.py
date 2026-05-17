"""Tests for SQLite schema migration system."""
import tempfile, os
import pytest
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
