"""Redis-backed runtime executor for distributed game coordination."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisRuntimeExecutor:
    """Distributed runtime executor using Redis for locks and status.

    Provides per-game distributed locks with TTL and status tracking via Redis.
    Falls back gracefully (returns False/None) when Redis is unavailable.
    """

    LOCK_TTL_SECONDS: int = 300  # 5 minute TTL
    LOCK_KEY_PREFIX: str = "werewolf:lock:"
    STATUS_KEY_PREFIX: str = "werewolf:status:"

    def __init__(self, redis_url: str) -> None:
        if not redis_url or not redis_url.strip():
            raise ValueError("redis_url must be a non-empty string")
        self._redis_url = redis_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialize the Redis client on first access."""
        if self._client is None:
            try:
                import redis  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "redis-py is required for RedisRuntimeExecutor. "
                    "Install it with: pip install redis"
                ) from exc
            self._client = redis.Redis.from_url(self._redis_url)
        return self._client

    def acquire_lock(self, game_id: str, ttl: int = LOCK_TTL_SECONDS) -> bool:
        """Acquire a per-game distributed lock with TTL.

        Returns True if the lock was acquired, False if already held or on error.
        """
        key = f"{self.LOCK_KEY_PREFIX}{game_id}"
        try:
            acquired = self.client.setnx(key, "1")
            if acquired:
                self.client.expire(key, ttl)
            return bool(acquired)
        except Exception:
            logger.warning("Redis unavailable during acquire_lock for game %s", game_id)
            return False

    def release_lock(self, game_id: str) -> None:
        """Release a per-game distributed lock."""
        key = f"{self.LOCK_KEY_PREFIX}{game_id}"
        try:
            self.client.delete(key)
        except Exception:
            logger.warning("Redis unavailable during release_lock for game %s", game_id)

    def get_status(self, game_id: str) -> dict | None:
        """Retrieve game status from Redis.

        Returns the status dict or None if not found or on error.
        """
        key = f"{self.STATUS_KEY_PREFIX}{game_id}"
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            # Redis may return bytes; decode if necessary
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("Redis unavailable during get_status for game %s", game_id)
            return None

    def set_status(self, game_id: str, status: dict) -> None:
        """Store game status in Redis as JSON."""
        key = f"{self.STATUS_KEY_PREFIX}{game_id}"
        try:
            self.client.set(key, json.dumps(status))
        except Exception:
            logger.warning("Redis unavailable during set_status for game %s", game_id)

    def refresh_lock(self, game_id: str, ttl: int = LOCK_TTL_SECONDS) -> bool:
        """Refresh the TTL on an existing lock.

        Returns True if the key existed and was refreshed, False otherwise.
        """
        key = f"{self.LOCK_KEY_PREFIX}{game_id}"
        try:
            result = self.client.expire(key, ttl)
            return bool(result)
        except Exception:
            logger.warning("Redis unavailable during refresh_lock for game %s", game_id)
            return False

    def is_locked(self, game_id: str) -> bool:
        """Check whether a per-game lock is currently held."""
        key = f"{self.LOCK_KEY_PREFIX}{game_id}"
        try:
            return bool(self.client.exists(key))
        except Exception:
            logger.warning("Redis unavailable during is_locked for game %s", game_id)
            return False

    def close(self) -> None:
        """Close the Redis connection pool if the client was initialized."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.warning("Error closing Redis connection")
