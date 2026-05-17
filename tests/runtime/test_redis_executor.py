"""Tests for RedisRuntimeExecutor with mocked Redis client."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from werewolf_agent.runtime.redis_executor import RedisRuntimeExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_redis_module() -> types.ModuleType:
    """Create a fake ``redis`` module with a MagicMock Redis class."""
    mock_redis = types.ModuleType("redis")
    mock_redis.Redis = MagicMock()
    mock_redis.Redis.from_url = MagicMock()
    return mock_redis


def _make_executor() -> tuple[RedisRuntimeExecutor, MagicMock]:
    """Create a RedisRuntimeExecutor with an injected mock client.

    Returns (executor, mock_client).
    """
    executor = RedisRuntimeExecutor("redis://localhost:6379/0")
    mock_client = MagicMock()
    executor._client = mock_client
    return executor, mock_client


# ===================================================================
# 1. Lock tests
# ===================================================================


class TestRedisExecutorLock:
    """Per-game distributed lock operations."""

    def test_acquire_lock_succeeds(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.setnx.return_value = True

        result = executor.acquire_lock("game-1")

        assert result is True
        mock_client.setnx.assert_called_once_with("werewolf:lock:game-1", "1")

    def test_acquire_lock_fails_when_held(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.setnx.return_value = False

        result = executor.acquire_lock("game-1")

        assert result is False

    def test_release_lock(self) -> None:
        executor, mock_client = _make_executor()

        executor.release_lock("game-1")

        mock_client.delete.assert_called_once_with("werewolf:lock:game-1")

    def test_lock_has_ttl(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.setnx.return_value = True

        executor.acquire_lock("game-1", ttl=600)

        mock_client.expire.assert_called_once_with("werewolf:lock:game-1", 600)

    def test_refresh_lock(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.expire.return_value = True

        result = executor.refresh_lock("game-1", ttl=120)

        assert result is True
        mock_client.expire.assert_called_once_with("werewolf:lock:game-1", 120)

    def test_is_locked_true(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.exists.return_value = 1

        assert executor.is_locked("game-1") is True

    def test_is_locked_false(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.exists.return_value = 0

        assert executor.is_locked("game-1") is False


# ===================================================================
# 2. Status tests
# ===================================================================


class TestRedisExecutorStatus:
    """Game status get/set with JSON serialization."""

    def test_set_and_get_status(self) -> None:
        executor, mock_client = _make_executor()
        status = {"phase": "night", "step_count": 5}
        mock_client.get.return_value = json.dumps(status).encode("utf-8")

        executor.set_status("game-1", status)
        result = executor.get_status("game-1")

        mock_client.set.assert_called_once_with(
            "werewolf:status:game-1", json.dumps(status)
        )
        assert result == status

    def test_status_none_when_missing(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.get.return_value = None

        result = executor.get_status("game-1")

        assert result is None

    def test_set_status_stores_json(self) -> None:
        executor, mock_client = _make_executor()
        status = {"state": "running", "phase": "day"}

        executor.set_status("game-1", status)

        mock_client.set.assert_called_once_with(
            "werewolf:status:game-1", json.dumps(status)
        )

    def test_status_handles_bytes_return(self) -> None:
        executor, mock_client = _make_executor()
        data = {"phase": "night"}
        mock_client.get.return_value = json.dumps(data).encode("utf-8")

        result = executor.get_status("game-1")

        assert result == {"phase": "night"}


# ===================================================================
# 3. Error / resilience tests
# ===================================================================


class TestRedisExecutorErrors:
    """Graceful degradation when Redis is unavailable."""

    def test_redis_unavailable_returns_false_on_lock(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.setnx.side_effect = Exception("Connection refused")

        result = executor.acquire_lock("game-1")

        assert result is False

    def test_redis_unavailable_returns_none_on_status(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.get.side_effect = Exception("Connection refused")

        result = executor.get_status("game-1")

        assert result is None

    def test_redis_unavailable_no_raise_on_release(self) -> None:
        executor, mock_client = _make_executor()
        mock_client.delete.side_effect = Exception("Connection refused")

        # Should not raise
        executor.release_lock("game-1")

    def test_empty_redis_url_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="redis_url"):
            RedisRuntimeExecutor("")

        with pytest.raises(ValueError, match="redis_url"):
            RedisRuntimeExecutor("   ")


# ===================================================================
# 4. Connection lifecycle tests
# ===================================================================


class TestRedisExecutorConnection:
    """Client close / lifecycle."""

    def test_close_closes_client(self) -> None:
        executor, mock_client = _make_executor()

        executor.close()

        mock_client.close.assert_called_once()

    def test_close_without_init(self) -> None:
        executor = RedisRuntimeExecutor("redis://localhost:6379/0")
        # _client was never accessed, so close should be a no-op
        executor.close()  # should not raise


# ===================================================================
# 5. Lazy-init tests
# ===================================================================


class TestRedisExecutorClientLazy:
    """Lazy client initialization and import-guard."""

    def test_client_lazy_init(self) -> None:
        mock_redis_mod = _make_mock_redis_module()
        mock_client_instance = MagicMock()
        mock_redis_mod.Redis.from_url.return_value = mock_client_instance

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            executor = RedisRuntimeExecutor("redis://localhost:6379/0")
            # Client not yet created
            assert executor._client is None

            # Access property triggers lazy init
            client = executor.client
            assert client is mock_client_instance
            mock_redis_mod.Redis.from_url.assert_called_once_with(
                "redis://localhost:6379/0"
            )

            # Second access returns same cached client
            client2 = executor.client
            assert client2 is mock_client_instance
            assert mock_redis_mod.Redis.from_url.call_count == 1

    def test_missing_redis_py_raises_runtimeerror(self) -> None:
        with patch.dict(sys.modules, {"redis": None}):
            executor = RedisRuntimeExecutor("redis://localhost:6379/0")
            # Force lazy-init by removing cached client
            executor._client = None
            with pytest.raises(RuntimeError, match="redis-py"):
                _ = executor.client
