"""Tests for durable customization config storage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository


def _record() -> dict:
    return {
        "config_id": "ruleset_abc123",
        "config_type": "ruleset",
        "raw_yaml": "ruleset_id: custom",
        "normalized": {"ruleset_id": "custom", "status": "playable"},
        "validation_result": {"valid": True},
        "content_hash": "abc123",
        "status": "playable",
        "version": "1",
        "maturity": "validated",
        "compatibility_matrix": {"status": "playable"},
        "diff_against_default": [],
        "creator_id": "mod1",
        "created_at": "2026-05-17T00:00:00+00:00",
        "updated_at": "2026-05-17T00:00:00+00:00",
    }


def test_inmemory_repository_round_trips_custom_config() -> None:
    repo = InMemoryGameRepository()
    record = _record()

    repo.save_custom_config(record)

    loaded = repo.load_custom_config("ruleset_abc123")
    assert loaded == record
    assert repo.list_custom_configs("ruleset") == [record]


def test_sqlite_repository_round_trips_custom_config_after_restart(tmp_path) -> None:
    db_path = str(tmp_path / "customization.db")
    repo = SqliteGameRepository(db_path)
    record = _record()

    repo.save_custom_config(record)
    repo.close()

    repo2 = SqliteGameRepository(db_path)
    loaded = repo2.load_custom_config("ruleset_abc123")
    assert loaded == record
    assert repo2.list_custom_configs("ruleset") == [record]
    repo2.close()


def test_customization_api_persists_saved_ruleset_to_game_repository() -> None:
    repo = InMemoryGameRepository()
    client = TestClient(create_app(repository=repo))
    template = client.get("/templates/ruleset").text

    response = client.post("/customization/rulesets?caller_id=mod1&caller_role=moderator", content=template)

    assert response.status_code == 200
    config_id = response.json()["config_id"]
    loaded = repo.load_custom_config(config_id)
    assert loaded is not None
    assert loaded["config_type"] == "ruleset"
    assert loaded["status"] == "playable"
