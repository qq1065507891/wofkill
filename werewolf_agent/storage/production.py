# -*- coding: utf-8 -*-
"""
功能描述：生产存储适配器边界——V1.1 保持 SQLite 为已实现的本地仓库，显式校验 PostgreSQL/Redis 配置。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
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
    initialize: bool = True

    def validate(self) -> None:
        backend = self.backend.lower().strip()
        if backend not in ("sqlite", "postgres"):
            raise ProductionStorageConfigError(f"Unknown storage backend: {self.backend!r}")
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
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
        dsn = config.postgres_dsn or os.getenv("POSTGRES_DSN", "")
        return PostgresGameRepository(dsn, initialize=config.initialize)
    raise ProductionStorageConfigError(f"Unknown storage backend: {config.backend}")
