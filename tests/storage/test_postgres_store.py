"""PostgresGameRepository unit tests with mocked psycopg.

Covers:
1. Schema creation (_ensure_schema)
2. Game CRUD: save, load, delete, list
3. Events: append (with seq numbering) and load
4. Deaths: save (clear + insert) and load
5. Model usage: save and load
6. Evaluation: save (upsert) and load
7. Config snapshots: save (upsert) and load
8. Connection management: close, import error
9. JSONB handling: dict results from psycopg
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


# ---------------------------------------------------------------------------
# Helpers
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


def _make_mock_psycopg() -> types.ModuleType:
    """Build a fake psycopg module with mock connect()."""
    mock_mod = types.ModuleType("psycopg")
    mock_mod.connect = MagicMock()
    return mock_mod


def _make_repo_without_init() -> Any:
    """Create a PostgresGameRepository without connecting, for per-test setup."""
    from werewolf_agent.storage.postgres_store import PostgresGameRepository

    repo = PostgresGameRepository.__new__(PostgresGameRepository)
    repo._dsn = "postgresql://test:test@localhost:5432/test"
    repo._conn = None
    return repo


def _setup_repo_with_mock_conn() -> tuple[Any, MagicMock]:
    """Return (repo, mock_conn) with a pre-injected mock connection."""
    repo = _make_repo_without_init()
    mock_conn = MagicMock()
    repo._conn = mock_conn
    return repo, mock_conn


# ---------------------------------------------------------------------------
# Fixture: patched psycopg
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_psycopg():
    """Patch psycopg import inside postgres_store for every test."""
    mock_mod = _make_mock_psycopg()
    with patch.dict(sys.modules, {"psycopg": mock_mod}):
        yield mock_mod


# ===========================================================================
# 1. Schema creation
# ===========================================================================


class TestEnsureSchema:
    def test_creates_vector_extension(self) -> None:
        from werewolf_agent.storage.postgres_store import PostgresGameRepository

        mock_mod = _make_mock_psycopg()
        mock_conn = MagicMock()
        mock_mod.connect.return_value = mock_conn

        with patch.dict(sys.modules, {"psycopg": mock_mod}):
            repo = PostgresGameRepository.__new__(PostgresGameRepository)
            repo._dsn = "dsn"
            repo._conn = None
            repo._ensure_schema()

        # First execute call should be CREATE EXTENSION
        execute_calls = mock_conn.execute.call_args_list
        first_sql = execute_calls[0][0][0]
        assert "vector" in first_sql.lower()

    def test_creates_all_six_tables(self) -> None:
        from werewolf_agent.storage.postgres_store import PostgresGameRepository

        mock_mod = _make_mock_psycopg()
        mock_conn = MagicMock()
        mock_mod.connect.return_value = mock_conn

        with patch.dict(sys.modules, {"psycopg": mock_mod}):
            repo = PostgresGameRepository.__new__(PostgresGameRepository)
            repo._dsn = "dsn"
            repo._conn = None
            repo._ensure_schema()

        execute_calls = mock_conn.execute.call_args_list
        all_sql = " ".join(c[0][0].lower() for c in execute_calls)

        expected_tables = [
            "games",
            "events",
            "deaths",
            "model_usage",
            "evaluations",
            "config_snapshots",
        ]
        for table in expected_tables:
            assert table in all_sql, f"Missing table: {table}"

    def test_schema_ends_with_commit(self) -> None:
        from werewolf_agent.storage.postgres_store import PostgresGameRepository

        mock_mod = _make_mock_psycopg()
        mock_conn = MagicMock()
        mock_mod.connect.return_value = mock_conn

        with patch.dict(sys.modules, {"psycopg": mock_mod}):
            repo = PostgresGameRepository.__new__(PostgresGameRepository)
            repo._dsn = "dsn"
            repo._conn = None
            repo._ensure_schema()

        mock_conn.commit.assert_called_once()


# ===========================================================================
# 2. Game CRUD
# ===========================================================================


class TestSaveGame:
    def test_save_game_executes_upsert(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("g1")
        repo.save_game(gs)

        mock_conn.execute.assert_called_once()
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO games" in sql_arg
        assert "ON CONFLICT" in sql_arg
        mock_conn.commit.assert_called_once()

    def test_save_game_serializes_state(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("g2")
        repo.save_game(gs)

        args = mock_conn.execute.call_args[0][1]
        assert args[0] == "g2"
        # Second arg is JSON string
        parsed = json.loads(args[1])
        assert parsed["game_id"] == "g2"
        assert parsed["phase"] == "night"


class TestLoadGame:
    def test_load_game_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("g1")
        serialized = json.dumps(asdict(gs), ensure_ascii=False)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (serialized,)
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_game("g1")
        assert loaded is not None
        assert loaded.game_id == "g1"
        assert loaded.phase == "night"

    def test_load_game_not_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        result = repo.load_game("missing")
        assert result is None

    def test_load_game_handles_dict_jsonb(self) -> None:
        """psycopg returns JSONB columns as dicts, not strings."""
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("g_dict")
        state_dict = asdict(gs)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (state_dict,)
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_game("g_dict")
        assert loaded is not None
        assert loaded.game_id == "g_dict"


class TestDeleteGame:
    def test_delete_game_executes_delete(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        repo.delete_game("g1")

        mock_conn.execute.assert_called_once()
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "DELETE FROM games" in sql_arg
        assert mock_conn.execute.call_args[0][1][0] == "g1"
        mock_conn.commit.assert_called_once()


class TestListGames:
    def test_list_games_returns_states(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs1 = _make_game_state("g1")
        gs2 = _make_game_state("g2")

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (json.dumps(asdict(gs1), ensure_ascii=False),),
            (json.dumps(asdict(gs2), ensure_ascii=False),),
        ]
        mock_conn.execute.return_value = mock_cursor

        games = repo.list_games()
        assert len(games) == 2
        ids = {g.game_id for g in games}
        assert ids == {"g1", "g2"}

    def test_list_games_empty(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        assert repo.list_games() == []


# ===========================================================================
# 3. Events
# ===========================================================================


class TestAppendEvents:
    def test_append_events_with_seq_numbering(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        events = _make_events()  # 2 events

        # First execute call returns current max seq = 0
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_cursor

        repo.append_events("g1", events)

        # Should have 1 (max seq) + 2 (inserts) = 3 execute calls + 1 commit
        assert mock_conn.execute.call_count == 3

        # Check seq numbering: event 1 -> seq 1, event 2 -> seq 2
        insert_calls = mock_conn.execute.call_args_list[1:]
        assert insert_calls[0][0][1][1] == 1  # first event seq
        assert insert_calls[1][0][1][1] == 2  # second event seq

    def test_append_events_continues_from_existing_seq(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)  # max seq is 5
        mock_conn.execute.return_value = mock_cursor

        events = [GameEvent(type="test", payload={"k": "v"})]
        repo.append_events("g1", events)

        insert_calls = mock_conn.execute.call_args_list[1:]
        assert insert_calls[0][0][1][1] == 6  # continues from 5

    def test_append_events_commit(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_cursor

        repo.append_events("g1", _make_events())
        mock_conn.commit.assert_called_once()


class TestLoadEvents:
    def test_load_events_returns_list(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("roles_assigned", {"seed": 42}),
            ("enter_night", {"night": 1}),
        ]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_events("g1")
        assert len(loaded) == 2
        assert loaded[0].type == "roles_assigned"
        assert loaded[0].payload == {"seed": 42}
        assert loaded[1].type == "enter_night"

    def test_load_events_with_string_payload(self) -> None:
        """psycopg may return JSONB as string in some configurations."""
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("test_event", '{"key": "value"}'),
        ]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_events("g1")
        assert loaded[0].payload == {"key": "value"}

    def test_load_events_empty(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        assert repo.load_events("g1") == []


# ===========================================================================
# 4. Deaths
# ===========================================================================


class TestSaveDeaths:
    def test_save_deaths_clears_and_inserts(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        deaths = _make_deaths()  # 1 death

        repo.save_deaths("g1", deaths)

        # First call: DELETE, then one INSERT per death, then commit
        assert mock_conn.execute.call_count == 2
        delete_sql = mock_conn.execute.call_args_list[0][0][0]
        assert "DELETE FROM deaths" in delete_sql
        insert_sql = mock_conn.execute.call_args_list[1][0][0]
        assert "INSERT INTO deaths" in insert_sql

    def test_save_deaths_serializes_death_data(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        deaths = [
            Death(
                player_id="hunter",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_2",
                triggered_skills=["hunter_shot"],
            ),
        ]

        repo.save_deaths("g1", deaths)

        insert_call = mock_conn.execute.call_args_list[1]
        args = insert_call[0][1]
        assert args[0] == "g1"
        assert args[1] == "hunter"
        parsed = json.loads(args[2])
        assert parsed["triggered_skills"] == ["hunter_shot"]


class TestLoadDeaths:
    def test_load_deaths_returns_list(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        death_dict = {
            "player_id": "v1",
            "reason": "wolf_kill",
            "timing": "night",
            "resolution_batch": "night_1",
            "source_player_id": None,
            "can_leave_last_words": None,
            "triggered_skills": [],
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(death_dict,)]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_deaths("g1")
        assert len(loaded) == 1
        assert loaded[0].player_id == "v1"
        assert loaded[0].reason == "wolf_kill"

    def test_load_deaths_with_string_json(self) -> None:
        """JSONB column returned as string."""
        repo, mock_conn = _setup_repo_with_mock_conn()

        death_dict = {
            "player_id": "v1",
            "reason": "wolf_kill",
            "timing": "night",
            "resolution_batch": "night_1",
            "source_player_id": None,
            "can_leave_last_words": None,
            "triggered_skills": [],
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(json.dumps(death_dict),)]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_deaths("g1")
        assert loaded[0].player_id == "v1"

    def test_load_deaths_empty(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        assert repo.load_deaths("g1") == []


# ===========================================================================
# 5. Model usage
# ===========================================================================


class TestModelUsage:
    def test_save_model_usage(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        record = {"agent_id": "p01", "provider": "openai", "tokens": 100}

        repo.save_model_usage("g1", record)

        mock_conn.execute.assert_called_once()
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO model_usage" in sql_arg
        mock_conn.commit.assert_called_once()

    def test_load_model_usage_returns_dicts(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ({"agent_id": "p01", "provider": "openai", "tokens": 100},),
            ({"agent_id": "p02", "provider": "anthropic", "tokens": 200},),
        ]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_model_usage("g1")
        assert len(loaded) == 2
        assert loaded[0]["agent_id"] == "p01"
        assert loaded[1]["provider"] == "anthropic"

    def test_load_model_usage_with_string_json(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('{"agent_id": "p01", "tokens": 50}',),
        ]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_model_usage("g1")
        assert loaded[0]["agent_id"] == "p01"
        assert loaded[0]["tokens"] == 50

    def test_load_model_usage_empty(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        assert repo.load_model_usage("g1") == []


# ===========================================================================
# 6. Evaluation
# ===========================================================================


class TestEvaluation:
    def test_save_evaluation_upsert(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        result = {"game_id": "g1", "winning_faction": "good"}

        repo.save_evaluation("g1", result)

        sql_arg = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO evaluations" in sql_arg
        assert "ON CONFLICT" in sql_arg
        mock_conn.commit.assert_called_once()

    def test_load_evaluation_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            {"game_id": "g1", "winning_faction": "good"},
        )
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_evaluation("g1")
        assert loaded is not None
        assert loaded["winning_faction"] == "good"

    def test_load_evaluation_not_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        assert repo.load_evaluation("missing") is None

    def test_load_evaluation_string_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            '{"game_id": "g1", "winning_faction": "werewolf"}',
        )
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_evaluation("g1")
        assert loaded is not None
        assert loaded["winning_faction"] == "werewolf"


# ===========================================================================
# 7. Config snapshots
# ===========================================================================


class TestConfigSnapshots:
    def test_save_config_snapshot_upsert(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        config = {"ruleset_id": "test", "seed": 42}

        repo.save_config_snapshot("g1", config)

        sql_arg = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO config_snapshots" in sql_arg
        assert "ON CONFLICT" in sql_arg
        mock_conn.commit.assert_called_once()

    def test_load_config_snapshot_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            {"ruleset_id": "test", "seed": 42},
        )
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_config_snapshot("g1")
        assert loaded is not None
        assert loaded["ruleset_id"] == "test"
        assert loaded["seed"] == 42

    def test_load_config_snapshot_not_found(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor

        assert repo.load_config_snapshot("missing") is None

    def test_load_config_snapshot_string_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            '{"ruleset_id": "test", "models": {"p01": "gpt-4"}}',
        )
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_config_snapshot("g1")
        assert loaded is not None
        assert loaded["models"]["p01"] == "gpt-4"


# ===========================================================================
# 8. Connection management
# ===========================================================================


class TestConnectionManagement:
    def test_close_closes_connection(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        repo.close()
        mock_conn.close.assert_called_once()
        assert repo._conn is None

    def test_close_noop_when_not_connected(self) -> None:
        repo = _make_repo_without_init()
        repo.close()  # Should not raise
        assert repo._conn is None

    def test_connect_reuses_existing(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        result = repo._connect()
        assert result is mock_conn

    def test_psycopg_import_error_raises_runtime_error(self) -> None:
        """When psycopg is not installed, _connect raises RuntimeError."""
        repo = _make_repo_without_init()

        # Remove psycopg from importable modules to simulate missing install
        with patch.dict(sys.modules, {"psycopg": None}):
            # Also need to clear any cached import
            import builtins
            real_import = builtins.__import__

            def blocking_import(name, *args, **kwargs):
                if name == "psycopg":
                    raise ImportError("No module named 'psycopg'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=blocking_import):
                with pytest.raises(RuntimeError, match="psycopg is required"):
                    repo._connect()


# ===========================================================================
# 9. JSONB handling
# ===========================================================================


class TestJsonbHandling:
    """Verify that dict returns from psycopg (JSONB) are handled correctly."""

    def test_load_game_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("jsonb_test")
        state_dict = asdict(gs)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (state_dict,)  # dict, not string
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_game("jsonb_test")
        assert loaded is not None
        assert loaded.game_id == "jsonb_test"
        assert loaded.phase == "night"

    def test_list_games_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()
        gs = _make_game_state("list_jsonb")
        state_dict = asdict(gs)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(state_dict,)]
        mock_conn.execute.return_value = mock_cursor

        games = repo.list_games()
        assert len(games) == 1
        assert games[0].game_id == "list_jsonb"

    def test_load_events_dict_payload(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        payload = {"player_id": "seer", "target": "w1", "nested": {"key": "val"}}
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("seer_check", payload)]
        mock_conn.execute.return_value = mock_cursor

        events = repo.load_events("g1")
        assert events[0].payload == payload
        assert events[0].payload["nested"]["key"] == "val"

    def test_load_deaths_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        death_dict = {
            "player_id": "v1",
            "reason": "exile",
            "timing": "day",
            "resolution_batch": "day_2_vote",
            "source_player_id": None,
            "can_leave_last_words": None,
            "triggered_skills": [],
        }
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(death_dict,)]
        mock_conn.execute.return_value = mock_cursor

        deaths = repo.load_deaths("g1")
        assert deaths[0].reason == "exile"
        assert deaths[0].timing == "day"

    def test_load_model_usage_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ({"agent_id": "p01", "provider": "openai"},),
        ]
        mock_conn.execute.return_value = mock_cursor

        loaded = repo.load_model_usage("g1")
        assert isinstance(loaded[0], dict)
        assert loaded[0]["agent_id"] == "p01"

    def test_load_evaluation_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ({"winning_faction": "good"},)
        mock_conn.execute.return_value = mock_cursor

        result = repo.load_evaluation("g1")
        assert isinstance(result, dict)
        assert result["winning_faction"] == "good"

    def test_load_config_snapshot_dict_jsonb(self) -> None:
        repo, mock_conn = _setup_repo_with_mock_conn()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ({"seed": 42, "ruleset": "test"},)
        mock_conn.execute.return_value = mock_cursor

        config = repo.load_config_snapshot("g1")
        assert isinstance(config, dict)
        assert config["seed"] == 42
