"""Environment variable loading for provider configuration."""

from __future__ import annotations

import os
from pathlib import Path

from werewolf_agent.model_gateway.providers.base import PROVIDER_DOTENV_KEYS

_ENV_OVERRIDES: dict[str, str] = {}


def get_env(key: str, default: str = "") -> str:
    """Read env var, preferring .env overrides loaded by load_local_dotenv."""
    return _ENV_OVERRIDES.get(key) or os.getenv(key, default)


def load_local_dotenv(
    path: str | Path = ".env",
    *,
    keys: set[str] | None = None,
) -> None:
    """Load provider KEY=VALUE lines without requiring python-dotenv at runtime."""
    env_path = Path(path)
    if not env_path.exists():
        return
    allowed_keys = keys or PROVIDER_DOTENV_KEYS
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed_keys:
            continue
        value = value.strip().strip('"').strip("'")
        if key and value:
            _ENV_OVERRIDES[key] = value
