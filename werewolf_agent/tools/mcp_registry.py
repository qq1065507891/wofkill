"""MCP Registry: external tool provider registration and dispatch.

Design doc §11.2 MCP positioning principles:
- Highly coupled local rule queries, state R/W, game progression do NOT go through MCP.
- MCP layer only provides external tool capabilities, does NOT own game state truth.
- All external MCP results must be annotated with source and treated as suggestions.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.tools.schemas import (
    MCPProvider,
    ToolCall,
    ToolResult,
    ToolSource,
    ToolStatus,
    annotate_mcp_result,
)
from werewolf_agent.tools.tool_logger import ToolCallLogger


class MCPRegistry:
    """Registry for external MCP tool providers.

    MCP tools are optional extensions — external game history, player
    profiles, custom prompt libraries, Notion/Slack integration, or
    player-written advisor plugins.
    """

    def __init__(self, logger: ToolCallLogger | None = None) -> None:
        self._providers: dict[str, MCPProvider] = {}
        self._logger = logger or ToolCallLogger()

    @property
    def logger(self) -> ToolCallLogger:
        return self._logger

    def register(self, provider: MCPProvider) -> None:
        self._providers[provider.name] = provider

    def unregister(self, provider_name: str) -> bool:
        if provider_name in self._providers:
            del self._providers[provider_name]
            return True
        return False

    def get(self, provider_name: str) -> MCPProvider | None:
        return self._providers.get(provider_name)

    def all_providers(self) -> list[MCPProvider]:
        return list(self._providers.values())

    def count(self) -> int:
        return len(self._providers)

    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def call(self, provider_name: str, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Call an external MCP tool. Result is always a suggestion."""
        provider = self._providers.get(provider_name)
        if provider is None:
            result = ToolResult(
                tool_name=tool_name,
                source=ToolSource.MCP_EXTERNAL,
                status=ToolStatus.NOT_FOUND,
                error_message=f"MCP provider '{provider_name}' not registered",
                is_suggestion=True,
                source_annotation=f"来源: {provider_name} | 错误: 未注册",
            )
            call_record = ToolCall(
                tool_name=tool_name,
                source=ToolSource.MCP_EXTERNAL,
                params=params,
            )
            self._logger.log(call_record, result)
            return result

        call_record = ToolCall(
            tool_name=tool_name,
            source=ToolSource.MCP_EXTERNAL,
            params=params,
        )

        try:
            result = provider.call(tool_name, params)
            result = annotate_mcp_result(result, provider_name)
        except Exception as exc:
            result = ToolResult(
                tool_name=tool_name,
                source=ToolSource.MCP_EXTERNAL,
                status=ToolStatus.ERROR,
                error_message=str(exc),
                is_suggestion=True,
                source_annotation=f"来源: {provider_name} | 错误: 调用失败",
            )

        self._logger.log(call_record, result)
        return result

    def call_all(self, tool_name: str, params: dict[str, Any]) -> list[ToolResult]:
        """Broadcast a tool call to all registered MCP providers."""
        results = []
        for name in list(self._providers.keys()):
            result = self.call(name, tool_name, params)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Built-in test MCP providers — for testing only
# ---------------------------------------------------------------------------

class MockMCPProvider(MCPProvider):
    """Mock MCP provider for testing."""

    def __init__(
        self,
        name: str = "mock_mcp",
        description: str = "Mock MCP provider",
        responses: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._responses = responses or {}

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        response = self._responses.get(tool_name)
        if response is None:
            return ToolResult(
                tool_name=tool_name,
                source=ToolSource.MCP_EXTERNAL,
                status=ToolStatus.NOT_FOUND,
                error_message=f"Tool '{tool_name}' not found in mock provider",
            )
        return ToolResult(
            tool_name=tool_name,
            source=ToolSource.MCP_EXTERNAL,
            status=ToolStatus.SUCCESS,
            data=response,
        )
