"""Production storage adapter boundary.

V1.1 keeps SQLite as the implemented local repository. PostgreSQL and Redis
configuration is validated explicitly so production deployment work cannot be
mistaken for completed infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from werewolf_agent.storage.repository import GameRepository
from werewolf_agent.storage.sqlite_store import SqliteGameRepository


class ProductionStorageConfigError(RuntimeError):
    """Raised when production storage configuration is missing or unsupported."""


@dataclass
class ProductionStorageConfig:
    backend: str = "sqlite"
    sqlite_path: str = "data/wofkill.db"
    postgres_dsn: str = ""
    redis_url: str = ""
    redis_runtime_state: bool = False

    def validate(self) -> None:
        backend = self.backend.lower().strip()
        if backend == "postgres":
            dsn = self.postgres_dsn or os.getenv("POSTGRES_DSN", "")
            if not dsn:
                raise ProductionStorageConfigError("POSTGRES_DSN is required for postgres storage")
        if self.redis_runtime_state:
            redis_url = self.redis_url or os.getenv("REDIS_URL", "")
            if not redis_url:
                raise ProductionStorageConfigError("REDIS_URL is required for Redis runtime state")


def create_game_repository(config: ProductionStorageConfig) -> GameRepository:
    """Create a game repository from production-style configuration."""
    config.validate()
    backend = config.backend.lower().strip()
    if backend == "sqlite":
        return SqliteGameRepository(config.sqlite_path)
    if backend == "postgres":
        raise ProductionStorageConfigError("postgres storage adapter is not implemented in V1.1")
    raise ProductionStorageConfigError(f"Unknown storage backend: {config.backend}")
