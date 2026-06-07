"""SQLite schema/migration consistency tests.

Catches drift between:
* ``werewolf_agent.storage.sqlite_store._SCHEMA`` — the legacy
  executescript path used by ``SqliteGameRepository.__init__``
* ``werewolf_agent.storage.migrations.MIGRATIONS[version=1]`` — the
  versioned migration applied by ``MigrationManager.apply_all``

Both must declare the same set of user tables so a database created
by one path is interoperable with the other (and a future migration
v2 can rely on the v1 baseline regardless of entry point).
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
        f"SQLite _SCHEMA missing reflections table"
    )
