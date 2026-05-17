"""Local development authentication for the Werewolf Agent API.

Uses HMAC-signed session tokens. Default mode is "local" which maps
known caller IDs to roles without any external auth service.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuthConfig:
    mode: str = "local"
    token_ttl_seconds: int = 3600
    local_users: dict[str, str] = field(default_factory=lambda: {
        "mod1": "moderator",
        "dbg1": "debugger",
        "spectator": "spectator",
    })
    secret_key: str = ""
    config_path: str = ""

    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = os.environ.get(
                "WEREWOLF_AUTH_SECRET", "wofkill-dev-key-change-me",
            )
        if self.config_path:
            self._load_from_file()

    def _load_from_file(self):
        p = Path(self.config_path)
        if p.exists():
            import yaml
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            self.mode = data.get("mode", self.mode)
            self.token_ttl_seconds = data.get(
                "token_ttl_seconds", self.token_ttl_seconds,
            )
            if "local_users" in data:
                self.local_users.update(data["local_users"])


@dataclass
class SessionToken:
    token: str
    caller_id: str
    role: str
    expires_at: float


class AuthManager:
    """Manages HMAC-signed session tokens for local development auth."""

    def __init__(self, config: AuthConfig | None = None) -> None:
        self._config = config or AuthConfig()
        self._sessions: dict[str, SessionToken] = {}
        self._revoked: set[str] = set()

    @property
    def config(self) -> AuthConfig:
        return self._config

    def create_session(self, caller_id: str, requested_role: str) -> str:
        """Create a new session token for *caller_id* with *requested_role*.

        In ``local`` mode the caller must be listed in ``local_users`` and
        the requested role must match the mapped role.
        """
        if self._config.mode == "local":
            allowed = self._config.local_users.get(caller_id)
            if allowed is None:
                raise PermissionError(f"Unknown user: {caller_id}")
            if allowed != requested_role:
                raise PermissionError(
                    f"User {caller_id} cannot assume role {requested_role}"
                )

        expires = time.time() + self._config.token_ttl_seconds
        raw = f"{caller_id}:{requested_role}:{expires:.0f}:{self._config.secret_key}"
        sig = hmac.new(
            self._config.secret_key.encode(),
            raw.encode(),
            hashlib.sha256,
        ).hexdigest()
        token_str = f"{caller_id}.{requested_role}.{expires:.0f}.{sig}"
        self._sessions[token_str] = SessionToken(
            token=token_str,
            caller_id=caller_id,
            role=requested_role,
            expires_at=expires,
        )
        return token_str

    def validate_session(self, token_str: str) -> str | None:
        """Validate a session token and return the role, or ``None``."""
        # Check revocation list first.
        if token_str in self._revoked:
            return None
        sess = self._sessions.get(token_str)
        if sess is None:
            # Token may have been created by another process — verify HMAC.
            parts = token_str.split(".")
            if len(parts) == 4:
                caller_id, role, exp_str, sig = parts
                raw = (
                    f"{caller_id}:{role}:{exp_str}:{self._config.secret_key}"
                )
                expected = hmac.new(
                    self._config.secret_key.encode(),
                    raw.encode(),
                    hashlib.sha256,
                ).hexdigest()
                if hmac.compare_digest(sig, expected) and float(exp_str) > time.time():
                    return role
            return None
        if sess.expires_at < time.time():
            del self._sessions[token_str]
            return None
        return sess.role

    def revoke_session(self, token_str: str) -> None:
        """Revoke a session token."""
        self._sessions.pop(token_str, None)
        self._revoked.add(token_str)
