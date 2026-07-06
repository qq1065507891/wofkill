# -*- coding: utf-8 -*-
"""
游戏路由鉴权 helper 的拆分兼容测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/api/test_game_auth_helpers.py
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException

from werewolf_agent.api.auth import AuthConfig, AuthManager
from werewolf_agent.api.permissions import PermissionChecker
from werewolf_agent.api.routes import game_auth
from werewolf_agent.api.routes import games
from werewolf_agent.api.schemas import CallerRole, CreateGameRequest, GameActionRequest


def test_auth_helpers_are_reexported_from_games_facade() -> None:
    assert games._resolve_caller_role is game_auth._resolve_caller_role
    assert games._enforce_moderator_only is not None
    assert games._enforce_create_game_auth is not None


def test_resolve_caller_role_keeps_production_player_agent_guard() -> None:
    auth = AuthManager(AuthConfig(mode="production", secret_key="test-secret"))

    with pytest.raises(HTTPException) as exc_info:
        game_auth._resolve_caller_role(
            {},
            "p01",
            CallerRole.PLAYER_AGENT,
            auth_manager=auth,
        )

    assert exc_info.value.status_code == 403


def test_create_game_auth_requires_moderator_role() -> None:
    auth = AuthManager(AuthConfig(mode="local", secret_key="test-secret"))
    checker = PermissionChecker()
    req = CreateGameRequest(caller_id="spectator", caller_role=CallerRole.SPECTATOR)

    with pytest.raises(HTTPException) as exc_info:
        game_auth._enforce_create_game_auth(req, auth, checker, {})

    assert exc_info.value.status_code == 403


def test_moderator_only_accepts_authorized_debugger() -> None:
    auth = AuthManager(AuthConfig(mode="local", secret_key="test-secret"))
    checker = PermissionChecker()
    req = GameActionRequest(caller_id="dbg1", caller_role=CallerRole.DEBUGGER)

    game_auth._enforce_moderator_only(
        req,
        auth,
        checker,
        {"dbg1": CallerRole.DEBUGGER},
        "g_auth",
        "step",
    )


def test_games_facade_enforce_create_game_auth_uses_patched_role_resolver(monkeypatch) -> None:
    auth = AuthManager(AuthConfig(mode="local", secret_key="test-secret"))
    checker = PermissionChecker()
    req = CreateGameRequest(caller_id="spectator", caller_role=CallerRole.SPECTATOR)

    def fake_resolver(*args, **kwargs) -> CallerRole:
        return CallerRole.MODERATOR

    monkeypatch.setattr(games, "_resolve_caller_role", fake_resolver)

    games._enforce_create_game_auth(req, auth, checker, {})


def test_legacy_elevated_auth_warning_keeps_games_logger_name(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="werewolf_agent.api.routes.games"):
        game_auth._resolve_caller_role(
            {"mod1": CallerRole.MODERATOR},
            "mod1",
            CallerRole.MODERATOR,
        )

    assert any(
        rec.name == "werewolf_agent.api.routes.games"
        and "Legacy query-param auth" in rec.getMessage()
        for rec in caplog.records
    )
