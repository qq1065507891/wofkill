"""Tests for concrete MCP transport connectors."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from werewolf_agent.core.models import GameState
from werewolf_agent.tools.mcp_transport import (
    HTTPTransport,
    PersonaQueryTransport,
    RepositoryHistoryTransport,
    TransportMCPProvider,
)
from werewolf_agent.tools.mcp_registry import MCPRegistry
from werewolf_agent.tools.schemas import ToolSource, ToolStatus


# -- RepositoryHistoryTransport --


class TestRepositoryHistoryTransport:
    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        gs1 = GameState(game_id="g001", ruleset_id="mixed", phase="ended")
        gs2 = GameState(game_id="g002", ruleset_id="mixed", phase="active")
        repo.list_games.return_value = [gs1, gs2]
        repo.load_game.return_value = gs1
        return repo

    def test_list_recent_games(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("list_recent_games", {"limit": 10})
        assert result["total"] == 2
        assert result["games"][0]["game_id"] == "g001"

    def test_list_recent_games_limits(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("list_recent_games", {"limit": 1})
        assert result["total"] == 1

    def test_get_game_summary(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("get_game_summary", {"game_id": "g001"})
        assert result["game_id"] == "g001"
        assert result["ruleset_id"] == "mixed"

    def test_get_game_summary_missing_id(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("get_game_summary", {})
        assert "error" in result

    def test_get_game_summary_not_found(self, mock_repo):
        mock_repo.load_game.return_value = None
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("get_game_summary", {"game_id": "nonexistent"})
        assert "error" in result

    def test_list_handles_repo_error(self, mock_repo):
        mock_repo.list_games.side_effect = RuntimeError("db down")
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("list_recent_games", {})
        assert result["games"] == []
        assert "error" in result

    def test_unknown_tool(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        result = t.call("bogus", {})
        assert "error" in result

    def test_via_registry_suggestion_only(self, mock_repo):
        t = RepositoryHistoryTransport(mock_repo)
        registry = MCPRegistry()
        registry.register(TransportMCPProvider(t))
        result = registry.call("repository_history", "list_recent_games", {})
        assert result.status == ToolStatus.SUCCESS
        assert result.data.get("_suggestion_only") is True


# -- PersonaQueryTransport --


class TestPersonaQueryTransport:
    @pytest.fixture
    def config_file(self, tmp_path):
        config = tmp_path / "personas.yaml"
        config.write_text(
            "persona_profiles:\n"
            "  logic_leader:\n"
            "    name: 'Logic Leader'\n"
            "    archetype: analytical\n"
            "  aggressive_bluffer:\n"
            "    name: 'Aggressive Bluffer'\n"
            "    archetype: aggressive\n",
            encoding="utf-8",
        )
        return config

    def test_list_profiles(self, config_file):
        t = PersonaQueryTransport(config_file)
        result = t.call("list_persona_profiles", {})
        assert result["total"] == 2
        ids = [p["persona_id"] for p in result["profiles"]]
        assert "logic_leader" in ids

    def test_get_profile(self, config_file):
        t = PersonaQueryTransport(config_file)
        result = t.call("get_persona_profile", {"persona_id": "logic_leader"})
        assert result["persona_id"] == "logic_leader"
        assert result["name"] == "Logic Leader"

    def test_get_profile_not_found(self, config_file):
        t = PersonaQueryTransport(config_file)
        result = t.call("get_persona_profile", {"persona_id": "missing"})
        assert "error" in result

    def test_get_profile_missing_id(self, config_file):
        t = PersonaQueryTransport(config_file)
        result = t.call("get_persona_profile", {})
        assert "error" in result

    def test_missing_config_file(self):
        t = PersonaQueryTransport("/nonexistent/path.yaml")
        result = t.call("list_persona_profiles", {})
        assert result["total"] == 0

    def test_unknown_tool(self, config_file):
        t = PersonaQueryTransport(config_file)
        result = t.call("bogus", {})
        assert "error" in result

    def test_via_registry_suggestion_only(self, config_file):
        t = PersonaQueryTransport(config_file)
        registry = MCPRegistry()
        registry.register(TransportMCPProvider(t))
        result = registry.call("persona_query", "list_persona_profiles", {})
        assert result.status == ToolStatus.SUCCESS
        assert result.data.get("_suggestion_only") is True


# -- HTTPTransport --


class TestHTTPTransport:
    def test_success_call(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com", api_key="test")
        result = t.call("my_tool", {"x": 1})
        assert result["result"] == "ok"
        assert "_latency_ms" in result

    def test_timeout_retries(self, monkeypatch):
        import httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com", max_retries=1)
        result = t.call("my_tool", {})
        assert "error" in result
        assert result["error"] == "timeout"
        assert mock_client.post.call_count == 2  # initial + 1 retry

    def test_client_error_no_retry(self, monkeypatch):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com", max_retries=3)
        result = t.call("my_tool", {})
        assert "404" in result["error"]
        assert mock_client.post.call_count == 1  # no retry for client errors

    def test_server_error_retries(self, monkeypatch):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com", max_retries=2)
        result = t.call("my_tool", {})
        assert "500" in result["error"]
        assert mock_client.post.call_count == 3  # initial + 2 retries

    def test_no_api_key_no_auth_header(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com")
        t.call("tool", {})
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1]["headers"] if call_kwargs[1] else call_kwargs.kwargs["headers"]
        assert "authorization" not in headers

    def test_api_key_adds_auth_header(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        import httpx
        monkeypatch.setattr(httpx, "Client", lambda **kw: mock_client)

        t = HTTPTransport(base_url="http://example.com", api_key="secret123")
        t.call("tool", {})
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1]["headers"] if call_kwargs[1] else call_kwargs.kwargs["headers"]
        assert headers["authorization"] == "Bearer secret123"

    def test_custom_name_and_description(self):
        t = HTTPTransport(base_url="http://example.com", name="custom", description="custom desc")
        assert t.name == "custom"
        assert t.description == "custom desc"
