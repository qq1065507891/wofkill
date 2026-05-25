"""Tool call logger: records all tool invocations for observability.

Every tool call (local and MCP) is logged with caller, parameters,
result, duration, and source annotation. Logs are queryable for audit
and experiment analysis.
"""

from __future__ import annotations

import collections
from typing import Any

from werewolf_agent.tools.schemas import (
    ToolCall,
    ToolCallLogEntry,
    ToolResult,
    ToolSource,
    ToolStatus,
)


class ToolCallLogger:
    """Records and queries tool call history."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: collections.deque[ToolCallLogEntry] = collections.deque(maxlen=max_entries)
        self._max_entries = max_entries

    def log(self, call: ToolCall, result: ToolResult, duration_ms: float = 0.0) -> None:
        entry = ToolCallLogEntry(call=call, result=result, duration_ms=duration_ms)
        self._entries.append(entry)

    def all_entries(self) -> list[ToolCallLogEntry]:
        return list(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def by_tool(self, tool_name: str) -> list[ToolCallLogEntry]:
        return [e for e in self._entries if e.call.tool_name == tool_name]

    def by_caller(self, caller_id: str) -> list[ToolCallLogEntry]:
        return [e for e in self._entries if e.call.caller_id == caller_id]

    def by_source(self, source: ToolSource) -> list[ToolCallLogEntry]:
        return [e for e in self._entries if e.call.source == source]

    def errors(self) -> list[ToolCallLogEntry]:
        return [e for e in self._entries if e.result.status == ToolStatus.ERROR]

    def mcp_calls(self) -> list[ToolCallLogEntry]:
        return self.by_source(ToolSource.MCP_EXTERNAL)

    def local_calls(self) -> list[ToolCallLogEntry]:
        return self.by_source(ToolSource.LOCAL)

    def clear(self) -> None:
        self._entries.clear()

    def summary(self) -> dict[str, Any]:
        total = len(self._entries)
        local = len(self.local_calls())
        mcp = len(self.mcp_calls())
        errs = len(self.errors())
        return {
            "total_calls": total,
            "local_calls": local,
            "mcp_calls": mcp,
            "errors": errs,
            "tool_names": list({e.call.tool_name for e in self._entries}),
        }
