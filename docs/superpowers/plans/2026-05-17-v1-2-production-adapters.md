# V1.2 Production Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete production adapter testing, add Redis distributed lock, and update project docs to reflect actual implementation state.

**Architecture:** PostgreSQL store and PgVector store are already fully implemented but lack tests. Redis adapter needs a new `RedisRuntimeExecutor` class. Production deployment docs need updating.

**Tech Stack:** pytest, psycopg (PostgreSQL), fakeredis/redis-py (Redis), unittest.mock

---

### Task 1: PostgreSQL Store Tests

**Files:**
- Create: `tests/storage/test_postgres_store.py`
- Reference: `werewolf_agent/storage/postgres_store.py`
- Reference: `werewolf_agent/storage/sqlite_store.py` (for `_serialize_game_state` / `_deserialize_game_state`)

- [ ] **Step 1: Write failing tests**

Create `tests/storage/test_postgres_store.py` with mocked psycopg connection tests:

```python
"""Tests for PostgresGameRepository using mocked psycopg connection."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.storage.postgres_store import PostgresGameRepository


@pytest.fixture
def mock_conn():
    """Create a mock psycopg connection."""
    conn = MagicMock()
    conn.execute.return_value = conn
    conn.fetchone.return_value = None
    conn.fetchall.return_value = []
    conn.commit.return_value = None
    conn.close.return_value = None
    return conn


@pytest.fixture
def mock_psycopg(mock_conn):
    """Patch psycopg.connect to return mock_conn."""
    with patch("werewolf_agent.storage.postgres_store.psycopg") as mock_mod:
        mock_mod.connect.return_value = mock_conn
        yield mock_mod, mock_conn


class TestPostgresStoreSchema:
    def test_ensure_schema_creates_tables(self, mock_psycopg):
        mock_mod, mock_conn = mock_psycopg
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        # Should have called execute for CREATE EXTENSION and 6 CREATE TABLE statements
        execute_calls = mock_conn.execute.call_args_list
        sqls = [str(c) for c in execute_calls]
        assert any("vector" in s for s in sqls)
        assert any("games" in s for s in sqls)
        assert any("events" in s for s in sqls)
        assert any("deaths" in s for s in sqls)
        assert any("model_usage" in s for s in sqls)
        assert any("evaluations" in s for s in sqls)
        assert any("config_snapshots" in s for s in sqls)
        repo.close()

    def test_close_closes_connection(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.close()
        mock_conn.close.assert_called()


class TestPostgresStoreGameCRUD:
    def test_save_game_upserts(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        state = GameState(game_id="g_test", ruleset_id="pre_witch_hunter_idiot_mixed")
        repo.save_game(state)
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()
        repo.close()

    def test_load_game_returns_state(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        state = GameState(game_id="g_test", ruleset_id="pre_witch_hunter_idiot_mixed")
        serialized = json.dumps({"game_id": "g_test", "ruleset_id": "pre_witch_hunter_idiot_mixed", "phase": "setup", "players": {}, "events": [], "deaths": []})
        mock_conn.execute.return_value.fetchone.return_value = (serialized,)
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        loaded = repo.load_game("g_test")
        assert loaded is not None
        assert loaded.game_id == "g_test"
        repo.close()

    def test_load_game_returns_none_when_missing(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        mock_conn.execute.return_value.fetchone.return_value = None
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        assert repo.load_game("nonexistent") is None
        repo.close()

    def test_delete_game(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.delete_game("g_test")
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()
        repo.close()

    def test_list_games(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        serialized = json.dumps({"game_id": "g_test", "ruleset_id": "mixed", "phase": "setup", "players": {}, "events": [], "deaths": []})
        mock_conn.execute.return_value.fetchall.return_value = [(serialized,)]
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        games = repo.list_games()
        assert len(games) == 1
        assert games[0].game_id == "g_test"
        repo.close()


class TestPostgresStoreEvents:
    def test_append_events(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        mock_conn.execute.return_value.fetchone.return_value = (0,)
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        events = [GameEvent(type="game_started", payload={"game_id": "g_test"})]
        repo.append_events("g_test", events)
        mock_conn.commit.assert_called()
        repo.close()

    def test_load_events(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        mock_conn.execute.return_value.fetchall.return_value = [
            ("game_started", {"game_id": "g_test"}),
        ]
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        events = repo.load_events("g_test")
        assert len(events) == 1
        assert events[0].type == "game_started"
        repo.close()


class TestPostgresStoreDeaths:
    def test_save_and_load_deaths(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        death = Death(player_id="p01", cause="wolf_kill", day=1, phase="night")
        mock_conn.execute.return_value.fetchall.return_value = [
            ({"player_id": "p01", "cause": "wolf_kill", "day": 1, "phase": "night", "triggered_skills": []}),
        ]
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.save_deaths("g_test", [death])
        loaded = repo.load_deaths("g_test")
        assert len(loaded) == 1
        assert loaded[0].player_id == "p01"
        repo.close()


class TestPostgresStoreModelUsage:
    def test_save_and_load_model_usage(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        record = {"provider": "anthropic", "tokens": 100}
        mock_conn.execute.return_value.fetchall.return_value = [(record,)]
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.save_model_usage("g_test", record)
        loaded = repo.load_model_usage("g_test")
        assert len(loaded) == 1
        assert loaded[0]["provider"] == "anthropic"
        repo.close()


class TestPostgresStoreEvaluation:
    def test_save_and_load_evaluation(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        result = {"winner": "werewolf"}
        mock_conn.execute.return_value.fetchone.return_value = (result,)
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.save_evaluation("g_test", result)
        loaded = repo.load_evaluation("g_test")
        assert loaded is not None
        assert loaded["winner"] == "werewolf"
        repo.close()


class TestPostgresStoreConfigSnapshot:
    def test_save_and_load_config(self, mock_psycopg):
        _, mock_conn = mock_psycopg
        config = {"ruleset_id": "mixed", "seed": 42}
        mock_conn.execute.return_value.fetchone.return_value = (config,)
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        repo.save_config_snapshot("g_test", config)
        loaded = repo.load_config_snapshot("g_test")
        assert loaded is not None
        assert loaded["ruleset_id"] == "mixed"
        repo.close()

    def test_load_config_returns_none_when_missing(self, mock_psycopg):
        _, mock_conn = mock_conn_missing = mock_psycopg
        mock_conn.execute.return_value.fetchone.return_value = None
        repo = PostgresGameRepository("postgresql://test", initialize=True)
        assert repo.load_config_snapshot("nonexistent") is None
        repo.close()


class TestPostgresStoreCustomConfig:
    def test_psycopg_import_error(self):
        """PostgresGameRepository should raise RuntimeError when psycopg is missing."""
        with patch.dict("sys.modules", {"psycopg": None}):
            with pytest.raises(RuntimeError, match="psycopg"):
                PostgresGameRepository("postgresql://test", initialize=True)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/storage/test_postgres_store.py -q`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/storage/test_postgres_store.py
git commit -m "test: add PostgresGameRepository unit tests with mocked psycopg"
```

---

### Task 2: PgVector Store Tests

**Files:**
- Modify: `tests/rag/test_rag_hardening.py`
- Reference: `werewolf_agent/rag/vector_store.py` (`PgVectorStore` class)

- [ ] **Step 1: Write PgVector tests with mocked psycopg**

Add a new test class to `tests/rag/test_rag_hardening.py`:

```python
class TestPgVectorStore:
    """Tests for PgVectorStore using mocked psycopg."""

    @pytest.fixture
    def mock_pg_conn(self):
        conn = MagicMock()
        conn.execute.return_value = conn
        conn.fetchone.return_value = (0,)
        conn.fetchall.return_value = []
        conn.commit.return_value = None
        conn.close.return_value = None
        return conn

    @pytest.fixture
    def pg_store(self, mock_pg_conn):
        with patch("werewolf_agent.rag.vector_store.psycopg") as mock_mod:
            mock_mod.connect.return_value = mock_pg_conn
            store = PgVectorStore("postgresql://test")
            yield store, mock_pg_conn

    def test_add_inserts_embedding(self, pg_store):
        store, mock_conn = pg_store
        store.add("entry_1", "test content")
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    def test_query_returns_results(self, pg_store):
        store, mock_conn = pg_store
        mock_conn.execute.return_value.fetchall.return_value = [
            ("entry_1", 0.95),
        ]
        results = store.query("test query", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "entry_1"

    def test_delete_removes_entry(self, pg_store):
        store, mock_conn = pg_store
        store.delete("entry_1")
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    def test_count_returns_number(self, pg_store):
        store, mock_conn = pg_store
        mock_conn.execute.return_value.fetchone.return_value = (42,)
        assert store.count() == 42

    def test_close_closes_connection(self, pg_store):
        store, mock_conn = pg_store
        store.close()
        mock_conn.close.assert_called()
```

- [ ] **Step 2: Run tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/rag/test_rag_hardening.py::TestPgVectorStore -q`

- [ ] **Step 3: Commit**

---

### Task 3: Redis Distributed Lock Adapter

**Files:**
- Create: `werewolf_agent/runtime/redis_executor.py`
- Create: `tests/runtime/test_redis_executor.py`
- Reference: `werewolf_agent/runtime/executor.py` (LocalRuntimeExecutor)

- [ ] **Step 1: Write failing tests**

Create `tests/runtime/test_redis_executor.py`:

```python
"""Tests for RedisRuntimeExecutor with mocked redis client."""

from __future__ import annotations

import pytest
import time
from unittest.mock import MagicMock, patch

from werewolf_agent.runtime.redis_executor import RedisRuntimeExecutor


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    client = MagicMock()
    client.set.return_value = True
    client.get.return_value = None
    client.delete.return_value = 1
    client.setnx.return_value = True
    client.expire.return_value = True
    return client


@pytest.fixture
def executor(mock_redis):
    with patch("werewolf_agent.runtime.redis_executor.redis") as mock_mod:
        mock_mod.Redis.from_url.return_value = mock_redis
        return RedisRuntimeExecutor(redis_url="redis://localhost:6379"), mock_redis


class TestRedisExecutorLock:
    def test_acquire_lock_succeeds(self, executor):
        ex, mock_redis = executor
        mock_redis.setnx.return_value = True
        assert ex.acquire_lock("game_1") is True

    def test_acquire_lock_fails_when_held(self, executor):
        ex, mock_redis = executor
        mock_redis.setnx.return_value = False
        assert ex.acquire_lock("game_1") is False

    def test_release_lock(self, executor):
        ex, mock_redis = executor
        ex.release_lock("game_1")
        mock_redis.delete.assert_called()

    def test_lock_has_ttl(self, executor):
        ex, mock_redis = executor
        ex.acquire_lock("game_1")
        mock_redis.expire.assert_called()


class TestRedisExecutorStatus:
    def test_set_and_get_status(self, executor):
        ex, mock_redis = executor
        mock_redis.get.return_value = b'{"state":"running","step_count":5}'
        status = ex.get_status("game_1")
        assert status is not None

    def test_status_returns_none_when_missing(self, executor):
        ex, mock_redis = executor
        mock_redis.get.return_value = None
        assert ex.get_status("game_1") is None

    def test_set_status(self, executor):
        ex, mock_redis = executor
        ex.set_status("game_1", {"state": "running", "step_count": 5})
        mock_redis.set.assert_called()


class TestRedisExecutorFallback:
    def test_redis_unavailable_returns_false_on_lock(self, mock_redis):
        mock_redis.setnx.side_effect = Exception("Connection refused")
        with patch("werewolf_agent.runtime.redis_executor.redis") as mock_mod:
            mock_mod.Redis.from_url.return_value = mock_redis
            ex = RedisRuntimeExecutor(redis_url="redis://localhost:6379")
            assert ex.acquire_lock("game_1") is False

    def test_redis_url_validation(self):
        with pytest.raises(ValueError, match="redis_url"):
            RedisRuntimeExecutor(redis_url="")
```

- [ ] **Step 2: Implement RedisRuntimeExecutor**

Create `werewolf_agent/runtime/redis_executor.py`:

```python
"""Redis-backed runtime executor for distributed game coordination.

Provides per-game distributed locks and status tracking via Redis.
Falls back gracefully when Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS = 300  # 5 minute TTL on game locks
LOCK_KEY_PREFIX = "werewolf:lock:"
STATUS_KEY_PREFIX = "werewolf:status:"


class RedisRuntimeExecutor:
    """Distributed runtime executor using Redis for locks and status."""

    def __init__(self, redis_url: str) -> None:
        if not redis_url:
            raise ValueError("redis_url must not be empty")
        self._redis_url = redis_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis.from_url(self._redis_url)
            except ImportError as exc:
                raise RuntimeError(
                    "redis-py is required for RedisRuntimeExecutor. "
                    "Install with: pip install redis"
                ) from exc
        return self._client

    def acquire_lock(self, game_id: str, ttl: int = LOCK_TTL_SECONDS) -> bool:
        key = f"{LOCK_KEY_PREFIX}{game_id}"
        try:
            acquired = self.client.setnx(key, "1")
            if acquired:
                self.client.expire(key, ttl)
            return bool(acquired)
        except Exception as exc:
            logger.warning("Redis lock acquire failed for %s: %s", game_id, exc)
            return False

    def release_lock(self, game_id: str) -> None:
        key = f"{LOCK_KEY_PREFIX}{game_id}"
        try:
            self.client.delete(key)
        except Exception as exc:
            logger.warning("Redis lock release failed for %s: %s", game_id, exc)

    def get_status(self, game_id: str) -> dict[str, Any] | None:
        key = f"{STATUS_KEY_PREFIX}{game_id}"
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis status get failed for %s: %s", game_id, exc)
            return None

    def set_status(self, game_id: str, status: dict[str, Any]) -> None:
        key = f"{STATUS_KEY_PREFIX}{game_id}"
        try:
            self.client.set(key, json.dumps(status, ensure_ascii=False))
        except Exception as exc:
            logger.warning("Redis status set failed for %s: %s", game_id, exc)
```

- [ ] **Step 3: Run tests**

Run: `D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_redis_executor.py -q`

- [ ] **Step 4: Commit**

---

### Task 4: Update PROGRESS.md and README.md

**Files:**
- Modify: `PROGRESS.md`
- Modify: `README.md`

- [ ] **Step 1: Update PROGRESS.md Next Step section**

Replace the outdated "Remaining production work beyond V1.1" with accurate status reflecting PostgreSQL store, PgVector store, Redis executor as done.

- [ ] **Step 2: Update README.md with Redis and production adapter info**

Add Redis executor documentation to README.md production section.

- [ ] **Step 3: Commit**
