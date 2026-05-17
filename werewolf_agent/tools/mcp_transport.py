"""MCP transport adapter boundary.

Real MCP clients can be wrapped by ``TransportMCPProvider`` so the rest of the
codebase continues to consume the existing MCPProvider contract. Transport
failures are converted into ToolResult errors by the adapter/registry boundary.
"""

from __future__ import annotations

from typing import Any, Protocol

from werewolf_agent.tools.schemas import ToolResult, ToolSource, ToolStatus


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
