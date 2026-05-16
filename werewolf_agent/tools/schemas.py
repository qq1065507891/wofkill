"""Tool schemas: structured types for tool calls, results, and logging.

Design doc §11.2: internal tools query local state (no RPC/MCP).
External MCP results are annotated with source and treated as suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Tool source classification
# ---------------------------------------------------------------------------

class ToolSource(str, Enum):
    LOCAL = "local"            # Internal LangGraph tool — deterministic
    MCP_EXTERNAL = "mcp"       # External MCP tool — suggestion only


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"


# ---------------------------------------------------------------------------
# Internal tool names — per design doc §11.2
# ---------------------------------------------------------------------------

class InternalToolName(str, Enum):
    QUERY_LEGAL_ACTIONS = "query_legal_actions"
    QUERY_PUBLIC_STATE = "query_public_state"
    QUERY_PRIVATE_STATE = "query_private_state"
    QUERY_RELATION_GRAPH = "query_relation_graph"
    QUERY_COGNITION_MATRIX = "query_cognition_matrix"
    WRITE_REVIEW = "write_review"
    CALL_EVALUATOR = "call_evaluator"
    READ_EXPERIMENT_CONFIG = "read_experiment_config"
    GENERATE_GAME_REPORT = "generate_game_report"


# ---------------------------------------------------------------------------
# Tool call / result
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    source: ToolSource = ToolSource.LOCAL
    caller_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class ToolResult:
    tool_name: str
    source: ToolSource
    status: ToolStatus = ToolStatus.SUCCESS
    data: Any = None
    error_message: str = ""
    is_suggestion: bool = False  # True for MCP results
    source_annotation: str = ""  # Required for MCP results
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool call log entry
# ---------------------------------------------------------------------------

@dataclass
class ToolCallLogEntry:
    call: ToolCall
    result: ToolResult
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# MCP provider protocol
# ---------------------------------------------------------------------------

class MCPProvider:
    """Protocol for external MCP tool providers.

    MCP results must set is_suggestion=True and source_annotation.
    MCP providers never own game state truth.
    """

    name: str
    description: str

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MCP result annotation
# ---------------------------------------------------------------------------

MCP_SUGGESTION_PREFIX = "[MCP建议]"

MCP_SOURCE_TEMPLATE = (
    "来源: {provider_name} | 类型: 外部工具 | "
    "注意: 此结果为建议，非裁判事实，不覆盖本地规则引擎"
)


def annotate_mcp_result(result: ToolResult, provider_name: str) -> ToolResult:
    """Annotate an MCP result with source and suggestion markers."""
    result.is_suggestion = True
    result.source_annotation = MCP_SOURCE_TEMPLATE.format(provider_name=provider_name)
    return result
