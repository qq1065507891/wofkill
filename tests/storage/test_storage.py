"""Storage tests: repository interface contract and SQLite implementation.

Covers:
1. Round-trip create/load/update game state
2. Append and load event log with ordering and metadata
3. Store and load deaths
4. Store and load model usage records
5. Store and load evaluation results (GameResult)
6. Store config snapshots
7. List games
8. Delete game
9. Restart-like reload: close and reopen SQLite, verify data survives
10. In-memory repository works for tests without file
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace

import pytest

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.storage.repository import GameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository
from werewolf_agent.storage.memory_store import InMemoryGameRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_game_state(game_id: str = "test_game") -> GameState:
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "seer": PlayerState(id="seer", role="seer"),
    }
    return GameState(
        game_id=game_id,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        players=players,
        phase="night",
        night_number=1,
    )


def _make_events() -> list[GameEvent]:
    return [
        GameEvent(type="roles_assigned", payload={"seed": 42}),
        GameEvent(type="enter_night", payload={"night": 1}),
        GameEvent(
            type="seer_check",
            payload={
                "seer_id": "seer",
                "target_id": "w1",
                "alignment": "werewolf",
                "night_number": 1,
                "visibility": "seer_only",
            },
        ),
        GameEvent(
            type="player_died",
            payload={
                "player_id": "v1",
                "reason": "wolf_kill",
                "timing": "night",
                "resolution_batch": "night_1",
            },
        ),
    ]


def _make_deaths() -> list[Death]:
    return [
        Death(
            player_id="v1",
            reason="wolf_kill",
            timing="night",
            resolution_batch="night_1",
        ),
    ]


# ---------------------------------------------------------------------------
# Parametrize: test against both SQLite and InMemory
# ---------------------------------------------------------------------------


def _repos() -> list[GameRepository]:
    """Create one InMemory and one SQLite (temp file) repository."""
    repos: list[GameRepository] = [InMemoryGameRepository()]
    tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpfile.close()
    repos.append(SqliteGameRepository(tmpfile.name))
    return repos


@pytest.fixture(params=["inmemory", "sqlite"])
def repo(request: pytest.FixtureRequest, tmp_path: object) -> GameRepository:
    if request.param == "inmemory":
        return InMemoryGameRepository()
    else:
        db_path = os.path.join(str(tmp_path), "test.db")
        return SqliteGameRepository(db_path)


# ---------------------------------------------------------------------------
# 1. Create / Load game state round-trip
# ---------------------------------------------------------------------------


class TestGameCreateLoad:
    def test_create_and_load_game(self, repo: GameRepository) -> None:
        gs = _make_game_state("g1")
        repo.save_game(gs)

        loaded = repo.load_game("g1")
        assert loaded is not None
        assert loaded.game_id == "g1"
        assert loaded.phase == "night"
        assert loaded.ruleset_id == "pre_witch_hunter_idiot_mixed"
        assert len(loaded.players) == 4
        assert loaded.players["w1"].role == "werewolf"
        assert loaded.night_number == 1

    def test_load_nonexistent_returns_none(self, repo: GameRepository) -> None:
        assert repo.load_game("nonexistent") is None

    def test_save_updates_existing_game(self, repo: GameRepository) -> None:
        gs = _make_game_state("g2")
        repo.save_game(gs)

        updated = replace(gs, phase="day", day_number=1)
        repo.save_game(updated)

        loaded = repo.load_game("g2")
        assert loaded is not None
        assert loaded.phase == "day"
        assert loaded.day_number == 1

    def test_save_game_preserves_all_fields(self, repo: GameRepository) -> None:
        players = {
            "w1": PlayerState(id="w1", role="werewolf", alive=False),
            "seer": PlayerState(id="seer", role="seer"),
            "witch": PlayerState(id="witch", role="witch"),
            "idiot": PlayerState(id="idiot", role="idiot", revealed_idiot=True, vote_enabled=False),
        }
        gs = GameState(
            game_id="full_fields",
            players=players,
            phase="finished",
            day_number=3,
            night_number=3,
            hybrid_master_id="w1",
            hybrid_master_faction="evil",
            sheriff_id="seer",
            sheriff_badge_state="active",
            antidote_used=True,
            poison_used=True,
            winning_faction="werewolf",
            hybrid_result="win",
            paused=True,
        )
        repo.save_game(gs)

        loaded = repo.load_game("full_fields")
        assert loaded is not None
        assert loaded.phase == "finished"
        assert loaded.hybrid_master_id == "w1"
        assert loaded.hybrid_master_faction == "evil"
        assert loaded.sheriff_id == "seer"
        assert loaded.sheriff_badge_state == "active"
        assert loaded.antidote_used is True
        assert loaded.poison_used is True
        assert loaded.winning_faction == "werewolf"
        assert loaded.hybrid_result == "win"
        assert loaded.paused is True
        assert not loaded.players["w1"].alive
        assert loaded.players["idiot"].revealed_idiot is True
        assert loaded.players["idiot"].vote_enabled is False


class TestProductionStorageBoundary:
    def test_sqlite_backend_factory_returns_sqlite_repo(self, tmp_path: object) -> None:
        from werewolf_agent.storage.production import (
            ProductionStorageConfig,
            create_game_repository,
        )

        db_path = os.path.join(str(tmp_path), "prod.db")
        repo = create_game_repository(ProductionStorageConfig(backend="sqlite", sqlite_path=db_path))

        assert isinstance(repo, SqliteGameRepository)

    def test_postgres_backend_requires_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.storage.production import (
            ProductionStorageConfig,
            ProductionStorageConfigError,
            create_game_repository,
        )

        monkeypatch.delenv("POSTGRES_DSN", raising=False)

        with pytest.raises(ProductionStorageConfigError, match="POSTGRES_DSN"):
            create_game_repository(ProductionStorageConfig(backend="postgres"))

    def test_redis_runtime_state_requires_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.storage.production import (
            ProductionStorageConfig,
            ProductionStorageConfigError,
        )

        monkeypatch.delenv("REDIS_URL", raising=False)

        with pytest.raises(ProductionStorageConfigError, match="REDIS_URL"):
            ProductionStorageConfig(backend="sqlite", redis_runtime_state=True).validate()

    def test_unknown_storage_backend_is_explicit_error(self) -> None:
        from werewolf_agent.storage.production import (
            ProductionStorageConfig,
            ProductionStorageConfigError,
            create_game_repository,
        )

        with pytest.raises(ProductionStorageConfigError, match="Unknown storage backend"):
            create_game_repository(ProductionStorageConfig(backend="mystery"))


# ---------------------------------------------------------------------------
# 2. Append and load event log
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_append_events_and_load(self, repo: GameRepository) -> None:
        gs = _make_game_state("e1")
        repo.save_game(gs)

        events = _make_events()
        repo.append_events("e1", events)

        loaded = repo.load_events("e1")
        assert len(loaded) == 4
        assert loaded[0].type == "roles_assigned"
        assert loaded[1].type == "enter_night"
        assert loaded[2].type == "seer_check"
        assert loaded[2].payload["alignment"] == "werewolf"
        assert loaded[3].type == "player_died"

    def test_events_preserve_ordering(self, repo: GameRepository) -> None:
        gs = _make_game_state("e2")
        repo.save_game(gs)

        repo.append_events("e2", [GameEvent(type="first", payload={})])
        repo.append_events("e2", [GameEvent(type="second", payload={})])
        repo.append_events("e2", [GameEvent(type="third", payload={})])

        loaded = repo.load_events("e2")
        assert [e.type for e in loaded] == ["first", "second", "third"]

    def test_events_preserve_payload(self, repo: GameRepository) -> None:
        gs = _make_game_state("e3")
        repo.save_game(gs)

        payload = {
            "player_id": "seer",
            "text": "我是预言家，查杀w1。",
            "nested": {"key": "value"},
            "number": 42,
            "flag": True,
        }
        repo.append_events("e3", [GameEvent(type="speech", payload=payload)])

        loaded = repo.load_events("e3")
        assert len(loaded) == 1
        assert loaded[0].payload["player_id"] == "seer"
        assert loaded[0].payload["text"] == "我是预言家，查杀w1。"
        assert loaded[0].payload["nested"]["key"] == "value"
        assert loaded[0].payload["number"] == 42
        assert loaded[0].payload["flag"] is True

    def test_load_events_empty_for_new_game(self, repo: GameRepository) -> None:
        gs = _make_game_state("e4")
        repo.save_game(gs)
        assert repo.load_events("e4") == []


# ---------------------------------------------------------------------------
# 3. Deaths
# ---------------------------------------------------------------------------


class TestDeaths:
    def test_save_and_load_deaths(self, repo: GameRepository) -> None:
        gs = _make_game_state("d1")
        repo.save_game(gs)

        deaths = _make_deaths()
        repo.save_deaths("d1", deaths)

        loaded = repo.load_deaths("d1")
        assert len(loaded) == 1
        assert loaded[0].player_id == "v1"
        assert loaded[0].reason == "wolf_kill"
        assert loaded[0].timing == "night"
        assert loaded[0].resolution_batch == "night_1"

    def test_deaths_with_triggered_skills(self, repo: GameRepository) -> None:
        gs = _make_game_state("d2")
        repo.save_game(gs)

        deaths = [
            Death(
                player_id="hunter",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_2",
                triggered_skills=["hunter_shot"],
            ),
        ]
        repo.save_deaths("d2", deaths)

        loaded = repo.load_deaths("d2")
        assert loaded[0].triggered_skills == ["hunter_shot"]

    def test_load_deaths_empty_for_new_game(self, repo: GameRepository) -> None:
        gs = _make_game_state("d3")
        repo.save_game(gs)
        assert repo.load_deaths("d3") == []


# ---------------------------------------------------------------------------
# 4. Model usage records
# ---------------------------------------------------------------------------


class TestModelUsage:
    def test_save_and_load_usage(self, repo: GameRepository) -> None:
        gs = _make_game_state("u1")
        repo.save_game(gs)

        record = {
            "agent_id": "p01",
            "task_type": "night_action",
            "provider": "openai",
            "model": "gpt-4",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "latency_ms": 1200,
            "estimated_cost": 0.015,
        }
        repo.save_model_usage("u1", record)

        loaded = repo.load_model_usage("u1")
        assert len(loaded) == 1
        assert loaded[0]["agent_id"] == "p01"
        assert loaded[0]["provider"] == "openai"
        assert loaded[0]["prompt_tokens"] == 100
        assert loaded[0]["latency_ms"] == 1200

    def test_multiple_usage_records(self, repo: GameRepository) -> None:
        gs = _make_game_state("u2")
        repo.save_game(gs)

        for i in range(5):
            repo.save_model_usage("u2", {
                "agent_id": f"p{i:02d}",
                "task_type": "speech",
                "provider": "mock",
                "model": "mock-v1",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            })

        loaded = repo.load_model_usage("u2")
        assert len(loaded) == 5


# ---------------------------------------------------------------------------
# 5. Evaluation results (GameResult)
# ---------------------------------------------------------------------------


class TestEvaluationResults:
    def test_save_and_load_evaluation(self, repo: GameRepository) -> None:
        gs = _make_game_state("ev1")
        repo.save_game(gs)

        result = {
            "game_id": "ev1",
            "initial_seed": 42,
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "winning_faction": "good",
            "total_days": 3,
            "total_nights": 3,
        }
        repo.save_evaluation("ev1", result)

        loaded = repo.load_evaluation("ev1")
        assert loaded is not None
        assert loaded["winning_faction"] == "good"
        assert loaded["total_days"] == 3

    def test_load_evaluation_nonexistent(self, repo: GameRepository) -> None:
        assert repo.load_evaluation("nonexistent") is None


# ---------------------------------------------------------------------------
# 6. Config snapshots
# ---------------------------------------------------------------------------


class TestConfigSnapshots:
    def test_save_and_load_config(self, repo: GameRepository) -> None:
        gs = _make_game_state("c1")
        repo.save_game(gs)

        config = {
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "models": {"p01": "gpt-4", "p02": "claude"},
            "seed": 42,
        }
        repo.save_config_snapshot("c1", config)

        loaded = repo.load_config_snapshot("c1")
        assert loaded is not None
        assert loaded["ruleset_id"] == "pre_witch_hunter_idiot_mixed"
        assert loaded["models"]["p01"] == "gpt-4"

    def test_load_config_nonexistent(self, repo: GameRepository) -> None:
        assert repo.load_config_snapshot("nonexistent") is None


# ---------------------------------------------------------------------------
# 7. List games
# ---------------------------------------------------------------------------


class TestListGames:
    def test_list_empty(self, repo: GameRepository) -> None:
        assert repo.list_games() == []

    def test_list_multiple_games(self, repo: GameRepository) -> None:
        for gid in ["g1", "g2", "g3"]:
            repo.save_game(_make_game_state(gid))

        games = repo.list_games()
        assert len(games) == 3
        ids = [g.game_id for g in games]
        assert "g1" in ids
        assert "g2" in ids
        assert "g3" in ids

    def test_list_reflects_phase(self, repo: GameRepository) -> None:
        repo.save_game(_make_game_state("active"))
        finished = replace(_make_game_state("done"), phase="finished", winning_faction="good")
        repo.save_game(finished)

        games = repo.list_games()
        by_id = {g.game_id: g for g in games}
        assert by_id["active"].phase == "night"
        assert by_id["done"].phase == "finished"


# ---------------------------------------------------------------------------
# 8. Delete game
# ---------------------------------------------------------------------------


class TestDeleteGame:
    def test_delete_removes_game(self, repo: GameRepository) -> None:
        repo.save_game(_make_game_state("del1"))
        assert repo.load_game("del1") is not None

        repo.delete_game("del1")
        assert repo.load_game("del1") is None

    def test_delete_removes_associated_data(self, repo: GameRepository) -> None:
        gs = _make_game_state("del2")
        repo.save_game(gs)
        repo.append_events("del2", _make_events())
        repo.save_deaths("del2", _make_deaths())

        repo.delete_game("del2")
        assert repo.load_events("del2") == []
        assert repo.load_deaths("del2") == []

    def test_delete_nonexistent_does_not_raise(self, repo: GameRepository) -> None:
        repo.delete_game("nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# 9. Restart-like reload (SQLite specific)
# ---------------------------------------------------------------------------


class TestSqliteRestartReload:
    def test_sqlite_data_survives_close_and_reopen(self, tmp_path: object) -> None:
        db_path = os.path.join(str(tmp_path), "restart.db")

        # Write data
        repo1 = SqliteGameRepository(db_path)
        gs = _make_game_state("survive")
        repo1.save_game(gs)
        repo1.append_events("survive", _make_events())
        repo1.save_deaths("survive", _make_deaths())
        repo1.close()

        # Reopen and read
        repo2 = SqliteGameRepository(db_path)
        loaded = repo2.load_game("survive")
        assert loaded is not None
        assert loaded.game_id == "survive"
        assert loaded.phase == "night"

        events = repo2.load_events("survive")
        assert len(events) == 4
        assert events[2].type == "seer_check"

        deaths = repo2.load_deaths("survive")
        assert len(deaths) == 1
        assert deaths[0].player_id == "v1"
        repo2.close()


# ---------------------------------------------------------------------------
# 10. API app wiring
# ---------------------------------------------------------------------------


class TestAppWiring:
    def test_create_app_with_inmemory_repo(self) -> None:
        from werewolf_agent.api.app import create_app

        repo = InMemoryGameRepository()
        app = create_app(repository=repo)
        assert app is not None

    def test_create_app_without_repo_uses_memory(self) -> None:
        from werewolf_agent.api.app import create_app

        app = create_app()
        assert app is not None
        assert hasattr(app.state, "games")

    def test_api_persists_through_repo(self) -> None:
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient

        repo = InMemoryGameRepository()
        app = create_app(repository=repo)
        client = TestClient(app)

        # Create a game
        resp = client.post("/games", json={
            "player_count": 12,
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
        })
        assert resp.status_code == 200
        game_id = resp.json()["game"]["game_id"]

        # Game should be in repo
        loaded = repo.load_game(game_id)
        assert loaded is not None
        assert loaded.phase == "setup"

    def test_api_start_persists_players(self) -> None:
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient

        repo = InMemoryGameRepository()
        app = create_app(repository=repo)
        client = TestClient(app)

        # Create and start
        resp = client.post("/games", json={
            "player_count": 12,
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 1,
        })
        game_id = resp.json()["game"]["game_id"]

        resp = client.post(f"/games/{game_id}/start", json={
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        assert resp.status_code == 200

        # Check repo has players
        loaded = repo.load_game(game_id)
        assert loaded is not None
        assert len(loaded.players) == 12


# ---------------------------------------------------------------------------
# RAG and Memory persistence
# ---------------------------------------------------------------------------


class TestRAGPersistence:
    """RAG entry save/load/delete through repository backends."""

    @pytest.fixture(params=["inmemory", "sqlite"])
    def repo(self, request, tmp_path):
        if request.param == "inmemory":
            return InMemoryGameRepository()
        return SqliteGameRepository(str(tmp_path / "test_rag.db"))

    def _make_rag_entries(self) -> list[dict]:
        return [
            {
                "entry_id": "rag_1",
                "title": "Seer Strategy",
                "summary": "Claim early as seer",
                "metadata": {
                    "case_type": "SPEECH_TEMPLATE",
                    "quality_grade": "RULE_DERIVED_SEED",
                    "visibility_boundary": "PLAYER_PERSPECTIVE",
                    "source": {"source_type": "RULE_DERIVED"},
                    "tags": ["seer", "claim"],
                    "review_status": "approved",
                },
            },
            {
                "entry_id": "rag_2",
                "title": "Wolf Tactics",
                "summary": "Deep hook strategy",
                "metadata": {
                    "case_type": "EXTERNAL_TACTICS",
                    "quality_grade": "EXPERT_REVIEW",
                    "visibility_boundary": "PLAYER_PERSPECTIVE",
                    "source": {"source_type": "EXPERT_COMMENTARY"},
                    "tags": ["wolf", "deception"],
                    "review_status": "approved",
                },
            },
        ]

    def test_save_and_load_rag_entries(self, repo) -> None:
        entries = self._make_rag_entries()
        repo.save_rag_entries(entries)
        loaded = repo.load_rag_entries()
        assert len(loaded) == 2
        ids = {e["entry_id"] for e in loaded}
        assert ids == {"rag_1", "rag_2"}

    def test_rag_entry_content_preserved(self, repo) -> None:
        entries = self._make_rag_entries()
        repo.save_rag_entries(entries)
        loaded = repo.load_rag_entries()
        seer = next(e for e in loaded if e["entry_id"] == "rag_1")
        assert seer["title"] == "Seer Strategy"
        assert seer["summary"] == "Claim early as seer"

    def test_delete_rag_entry(self, repo) -> None:
        entries = self._make_rag_entries()
        repo.save_rag_entries(entries)
        repo.delete_rag_entry("rag_1")
        loaded = repo.load_rag_entries()
        assert len(loaded) == 1
        assert loaded[0]["entry_id"] == "rag_2"

    def test_rag_overwrite_on_duplicate_id(self, repo) -> None:
        entries = [{"entry_id": "dup_1", "title": "V1", "summary": "First"}]
        repo.save_rag_entries(entries)
        entries2 = [{"entry_id": "dup_1", "title": "V2", "summary": "Second"}]
        repo.save_rag_entries(entries2)
        loaded = repo.load_rag_entries()
        assert len(loaded) == 1
        assert loaded[0]["title"] == "V2"

    def test_empty_rag_load(self, repo) -> None:
        loaded = repo.load_rag_entries()
        assert loaded == []


class TestMemorySnapshotPersistence:
    """Memory snapshot save/load/delete through repository backends."""

    @pytest.fixture(params=["inmemory", "sqlite"])
    def repo(self, request, tmp_path):
        if request.param == "inmemory":
            return InMemoryGameRepository()
        return SqliteGameRepository(str(tmp_path / "test_mem.db"))

    def _make_snapshot(self) -> dict:
        return {
            "cognition_matrices": {
                "p01": {
                    "viewer_id": "p01",
                    "player_states": {
                        "p02": {"role_prob": {"werewolf": 0.3, "villager": 0.5, "seer": 0.2}},
                    },
                },
            },
            "relation_graph": {"events": []},
            "reflections": [
                {
                    "entry_id": "ref_1",
                    "game_id": "g1",
                    "player_id": "p01",
                    "role": "werewolf",
                    "faction_won": False,
                    "text": "Need to deceive better",
                    "tags": ["wolf", "strategy"],
                },
            ],
            "profiles": [
                {
                    "player_id": "p01",
                    "logic": 5.0,
                    "deception": 3.0,
                    "leadership": 2.0,
                    "credibility": 4.0,
                    "learning_rate": 0.5,
                    "risk_preference": 0.3,
                    "games_played": 1,
                    "games_won": 0,
                },
            ],
        }

    def test_save_and_load_snapshot(self, repo) -> None:
        data = self._make_snapshot()
        repo.save_memory_snapshot("snap_1", data)
        loaded = repo.load_memory_snapshot("snap_1")
        assert loaded is not None
        assert "p01" in loaded["cognition_matrices"]

    def test_snapshot_content_preserved(self, repo) -> None:
        data = self._make_snapshot()
        repo.save_memory_snapshot("snap_1", data)
        loaded = repo.load_memory_snapshot("snap_1")
        assert loaded["reflections"][0]["text"] == "Need to deceive better"
        assert loaded["profiles"][0]["deception"] == 3.0

    def test_list_snapshots(self, repo) -> None:
        repo.save_memory_snapshot("snap_a", {"key": "a"})
        repo.save_memory_snapshot("snap_b", {"key": "b"})
        snapshots = repo.list_memory_snapshots()
        ids = {s["snapshot_id"] for s in snapshots}
        assert "snap_a" in ids
        assert "snap_b" in ids

    def test_delete_snapshot(self, repo) -> None:
        repo.save_memory_snapshot("snap_del", {"key": "del"})
        repo.delete_memory_snapshot("snap_del")
        assert repo.load_memory_snapshot("snap_del") is None

    def test_overwrite_snapshot(self, repo) -> None:
        repo.save_memory_snapshot("snap_over", {"version": 1})
        repo.save_memory_snapshot("snap_over", {"version": 2})
        loaded = repo.load_memory_snapshot("snap_over")
        assert loaded["version"] == 2

    def test_load_nonexistent_snapshot(self, repo) -> None:
        assert repo.load_memory_snapshot("nope") is None


class TestPersistentMemoryCoordinator:
    """Integration: PersistentMemoryCoordinator bridges MemoryStore/RAG to repository."""

    @pytest.fixture(params=["inmemory", "sqlite"])
    def repo(self, request, tmp_path):
        if request.param == "inmemory":
            return InMemoryGameRepository()
        return SqliteGameRepository(str(tmp_path / "test_coord.db"))

    def test_save_and_restore_rag(self, repo) -> None:
        from werewolf_agent.rag.schemas import RAGEntry, CaseMetadata, SourceMetadata, CaseType, QualityGrade, VisibilityBoundary, SourceType
        from werewolf_agent.rag.retriever import StrategyRetriever
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        retriever = StrategyRetriever()
        retriever.add_entry(RAGEntry(
            entry_id="coord_rag_1",
            title="Test Entry",
            summary="A test RAG entry for coordinator",
            metadata=CaseMetadata(
                case_type=CaseType.SPEECH_TEMPLATE,
                quality_grade=QualityGrade.RULE_DERIVED_SEED,
                visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                source=SourceMetadata(source_type=SourceType.RULE_DERIVED),
                tags=["test"],
            ),
        ))

        coord = PersistentMemoryCoordinator(repo)
        coord.save_rag(retriever)

        restored = coord.restore_rag()
        assert len(restored) == 1
        assert restored[0].entry_id == "coord_rag_1"
        assert restored[0].summary == "A test RAG entry for coordinator"

    def test_save_and_restore_memory(self, repo) -> None:
        from werewolf_agent.memory.store import MemoryStore
        from werewolf_agent.memory.schemas import ReflectionEntry
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        store = MemoryStore()
        store.init_matrix("p01", ["p01", "p02"])
        store.store_reflection(ReflectionEntry(
            entry_id="coord_ref_1",
            game_id="g1",
            player_id="p01",
            role="werewolf",
            faction_won=False,
            text="Test reflection from coordinator",
            tags=["test"],
        ))

        coord = PersistentMemoryCoordinator(repo)
        coord.save_memory(store, "snap_coord_1")

        restored = coord.restore_memory("snap_coord_1")
        assert restored is not None
        refs = restored.reflections_by_player("p01")
        assert len(refs) == 1
        assert refs[0].text == "Test reflection from coordinator"

    def test_save_all_and_restore_all(self, repo) -> None:
        from werewolf_agent.memory.store import MemoryStore
        from werewolf_agent.memory.schemas import ReflectionEntry
        from werewolf_agent.rag.schemas import RAGEntry, CaseMetadata, SourceMetadata, CaseType, QualityGrade, VisibilityBoundary, SourceType
        from werewolf_agent.rag.retriever import StrategyRetriever
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        store = MemoryStore()
        store.init_matrix("p01", ["p01", "p02"])
        store.store_reflection(ReflectionEntry(
            entry_id="all_ref_1",
            game_id="g1",
            player_id="p01",
            role="seer",
            faction_won=True,
            text="Won as seer",
            tags=["seer", "win"],
        ))

        retriever = StrategyRetriever()
        retriever.add_entry(RAGEntry(
            entry_id="all_rag_1",
            title="Combined Test",
            summary="Test entry for combined save",
            metadata=CaseMetadata(
                case_type=CaseType.ROLE_STRATEGY,
                quality_grade=QualityGrade.RULE_DERIVED_SEED,
                visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                source=SourceMetadata(source_type=SourceType.RULE_DERIVED),
                tags=["combined"],
            ),
        ))

        coord = PersistentMemoryCoordinator(repo)
        coord.save_all(store, retriever, "snap_all_1")

        mem, rag = coord.restore_all("snap_all_1")
        assert mem is not None
        assert len(mem.reflections_by_player("p01")) == 1
        assert len(rag) == 1
        assert rag[0].entry_id == "all_rag_1"

    def test_sqlite_survives_restart(self, tmp_path) -> None:
        """SQLite RAG and memory data survive close/reopen."""
        from werewolf_agent.rag.schemas import RAGEntry, CaseMetadata, SourceMetadata, CaseType, QualityGrade, VisibilityBoundary, SourceType
        from werewolf_agent.rag.retriever import StrategyRetriever
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        db_path = str(tmp_path / "restart_test.db")
        repo = SqliteGameRepository(db_path)

        retriever = StrategyRetriever()
        retriever.add_entry(RAGEntry(
            entry_id="restart_rag",
            title="Restart Test",
            summary="Should survive DB restart",
            metadata=CaseMetadata(
                case_type=CaseType.SPEECH_TEMPLATE,
                quality_grade=QualityGrade.RULE_DERIVED_SEED,
                visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                source=SourceMetadata(source_type=SourceType.RULE_DERIVED),
                tags=["restart"],
            ),
        ))

        coord = PersistentMemoryCoordinator(repo)
        coord.save_rag(retriever)

        # Close and reopen
        repo.close()
        repo2 = SqliteGameRepository(db_path)
        coord2 = PersistentMemoryCoordinator(repo2)
        restored = coord2.restore_rag()
        assert len(restored) == 1
        assert restored[0].summary == "Should survive DB restart"
        repo2.close()
