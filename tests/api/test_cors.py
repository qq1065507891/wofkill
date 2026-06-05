"""Tests for the NEW-P2-7 CORS middleware fix.

api/app.py must mount a configurable CORSMiddleware. Origins come from
the WEREWOLF_CORS_ORIGINS env var (comma-separated).
"""

from __future__ import annotations

import pytest

from werewolf_agent.api.app import create_app
from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.storage.memory_store import InMemoryGameRepository

_TEST_SECRET = "test-secret-key-for-unit-tests-only"


def _build_auth():
    return AuthManager(AuthConfig(mode="local", secret_key=_TEST_SECRET))


def test_cors_middleware_configured(monkeypatch):
    """NEW-P2-7: ``create_app`` must mount a ``CORSMiddleware`` whose
    allowed origins are taken from ``WEREWOLF_CORS_ORIGINS``.

    The default (when the env var is unset) must still include at
    least localhost, and explicit env values must propagate through.
    """
    monkeypatch.setenv(
        "WEREWOLF_CORS_ORIGINS", "https://app.example.com,http://localhost:5173",
    )

    app = create_app(
        repository=InMemoryGameRepository(),
        auth_manager=_build_auth(),
    )

    # Starlette stores middleware in ``app.user_middleware`` (list of
    # middleware classes that have been *added*). We assert that
    # CORSMiddleware is among them, and inspect the kwargs passed to
    # the constructor for the configured origins.
    from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware

    found = False
    for mw in app.user_middleware:
        if mw.cls is StarletteCORSMiddleware:
            found = True
            origins = mw.kwargs.get("allow_origins") or []
            assert "https://app.example.com" in origins, (
                f"NEW-P2-7 not fixed: WEREWOLF_CORS_ORIGINS not propagated; "
                f"got allow_origins={origins!r}"
            )
            assert "http://localhost:5173" in origins
            break
    assert found, (
        "NEW-P2-7 not fixed: CORSMiddleware was not added to the app. "
        "Browser clients will be blocked by same-origin policy."
    )


def test_cors_middleware_default_origins(monkeypatch):
    """NEW-P2-7: when WEREWOLF_CORS_ORIGINS is unset, the default
    must still include localhost so dev parity is preserved.
    """
    monkeypatch.delenv("WEREWOLF_CORS_ORIGINS", raising=False)

    app = create_app(
        repository=InMemoryGameRepository(),
        auth_manager=_build_auth(),
    )

    from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware

    found = False
    for mw in app.user_middleware:
        if mw.cls is StarletteCORSMiddleware:
            found = True
            origins = mw.kwargs.get("allow_origins") or []
            assert len(origins) > 0, (
                "NEW-P2-7 not fixed: default CORS origins list is empty"
            )
            break
    assert found, "NEW-P2-7 not fixed: CORSMiddleware missing"
