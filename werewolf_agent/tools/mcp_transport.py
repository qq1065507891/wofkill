"""MCP transport adapter boundary.

Real MCP clients can be wrapped by ``TransportMCPProvider`` so the rest of the
codebase continues to consume the existing MCPProvider contract. Transport
failures are converted into ToolResult errors by the adapter/registry boundary.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from werewolf_agent.tools.schemas import ToolResult, ToolSource, ToolStatus

logger = logging.getLogger(__name__)


class MCPTransport(Protocol):
    name: str
    description: str

    def call(self, tool_name: str, params: dict[str, Any]) -> Any: ...


class TransportMCPProvider:
    """Adapter from a transport client to the MCPProvider interface."""

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self.name = transport.name
        self.description = transport.description

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        data = self._transport.call(tool_name, params)
        return ToolResult(
            tool_name=tool_name,
            source=ToolSource.MCP_EXTERNAL,
            status=ToolStatus.SUCCESS,
            data=data,
            metadata={"transport": self.name},
        )


# ---------------------------------------------------------------------------
# Concrete transports
# ---------------------------------------------------------------------------


class RepositoryHistoryTransport:
    """Queries game history from a GameRepository.

    Supported tools:
    - ``list_recent_games``: returns a list of recent game IDs and summaries.
    - ``get_game_summary``: returns basic info for one game (winner, player count, ruleset).
    """

    name = "repository_history"
    description = "查询本地游戏历史记录"

    def __init__(self, repository: Any, *, max_list: int = 50) -> None:
        self._repo = repository
        self._max_list = max_list

    def call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_recent_games":
            return self._list_recent_games(params)
        if tool_name == "get_game_summary":
            return self._get_game_summary(params)
        return {"error": f"unknown tool: {tool_name}"}

    def _list_recent_games(self, params: dict[str, Any]) -> dict[str, Any]:
        limit = min(params.get("limit", 20), self._max_list)
        try:
            games = self._repo.list_games()
        except Exception as exc:
            logger.warning("RepositoryHistoryTransport list_games failed: %s", exc)
            return {"games": [], "error": str(exc)}
        results = []
        for gs in games[:limit]:
            results.append({
                "game_id": gs.game_id,
                "ruleset_id": gs.ruleset_id,
                "phase": gs.phase,
                "player_count": len(gs.players),
            })
        return {"games": results, "total": len(results)}

    def _get_game_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        game_id = params.get("game_id", "")
        if not game_id:
            return {"error": "game_id is required"}
        try:
            gs = self._repo.load_game(game_id)
        except Exception as exc:
            logger.warning("RepositoryHistoryTransport load_game failed: %s", exc)
            return {"error": str(exc)}
        if gs is None:
            return {"error": "game not found"}
        return {
            "game_id": gs.game_id,
            "ruleset_id": gs.ruleset_id,
            "phase": gs.phase,
            "player_count": len(gs.players),
            "day_number": gs.day_number,
            "night_number": gs.night_number,
        }


class PersonaQueryTransport:
    _cache_ttl: float = 300.0  # 缓存 TTL（秒）
    """Queries persona configuration from YAML files.

    Supported tools:
    - ``list_persona_profiles``: returns all available profile IDs.
    - ``get_persona_profile``: returns details for one profile.
    """

    name = "persona_query"
    description = "查询人格配置档案"

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._data: dict[str, Any] | None = None
        self._cache_time: float = 0.0

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            # Check cache expiration
            if (time.monotonic() - self._cache_time) > self._cache_ttl:
                self._data = None
            else:
                return self._data
        if not self._config_path.exists():
            return {}
        import yaml
        self._data = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        self._cache_time = time.monotonic()
        return self._data

    def call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_persona_profiles":
            return self._list_profiles(params)
        if tool_name == "get_persona_profile":
            return self._get_profile(params)
        return {"error": f"unknown tool: {tool_name}"}

    def _list_profiles(self, params: dict[str, Any]) -> dict[str, Any]:
        data = self._load()
        personas = data.get("persona_profiles", {})
        results = []
        for pid, pdata in personas.items():
            results.append({
                "persona_id": pid,
                "name": pdata.get("name", pid),
                "archetype": pdata.get("archetype", ""),
            })
        return {"profiles": results, "total": len(results)}

    def _get_profile(self, params: dict[str, Any]) -> dict[str, Any]:
        persona_id = params.get("persona_id", "")
        if not persona_id:
            return {"error": "persona_id is required"}
        data = self._load()
        personas = data.get("persona_profiles", {})
        profile = personas.get(persona_id)
        if profile is None:
            return {"error": f"persona '{persona_id}' not found"}
        return {"persona_id": persona_id, **profile}


class HTTPTransport:
    """Generic HTTP transport for external MCP services.

    Sends tool calls as POST requests to a configured base URL.
    Includes retry logic and timeout handling.
    """

    name: str
    description: str

    def close(self) -> None:
        """关闭 HTTP 传输连接。"""
        if hasattr(self, '_client') and self._client is not None:
            self._client.close()

    def __enter__(self) -> "HTTPTransport":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __init__(
        self,
        *,
        base_url: str,
        name: str = "http_mcp",
        description: str = "HTTP MCP transport",
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.name = name
        self.description = description
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        payload = {"tool": tool_name, "params": params}
        last_error: str = ""

        for attempt in range(self._max_retries + 1):
            try:
                start = time.monotonic()
                resp = self._client.post(
                    f"{self._base_url}/tools/{tool_name}",
                    headers=headers,
                    json=payload,
                )
                latency = int((time.monotonic() - start) * 1000)
                resp.raise_for_status()
                data = resp.json()
                data["_latency_ms"] = latency
                return data
            except httpx.TimeoutException:
                last_error = "timeout"
                logger.warning("HTTPTransport timeout on attempt %d for %s", attempt + 1, tool_name)
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code}"
                if exc.response.status_code < 500:
                    break  # Don't retry client errors
                logger.warning("HTTPTransport %s on attempt %d for %s", last_error, attempt + 1, tool_name)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("HTTPTransport error on attempt %d for %s: %s", attempt + 1, tool_name, exc)

        return {"error": last_error, "tool": tool_name}
