# -*- coding: utf-8 -*-
"""
功能描述：工具 schema——工具调用、结果和日志的结构化类型（设计文档 §11.2）。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


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

@runtime_checkable
class MCPProvider(Protocol):
    """外部 MCP 工具提供者的协议接口。

    MCP 结果必须设置 is_suggestion=True 和 source_annotation。
    MCP 提供者永远不拥有游戏状态真相。
    """

    name: str
    description: str

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# MCP result annotation
# ---------------------------------------------------------------------------

MCP_SUGGESTION_PREFIX = "[MCP建议]"

MCP_SOURCE_TEMPLATE = (
    "来源: {provider_name} | 类型: 外部工具 | "
    "注意: 此结果为建议，非裁判事实，不覆盖本地规则引擎"
)


def annotate_mcp_result(result: ToolResult, provider_name: str) -> ToolResult:
    """Annotate an MCP result with source and suggestion markers.

    Adds ``_suggestion_only = True`` to result.data so downstream consumers
    can programmatically check that this is a suggestion, not rule truth.
    """
    result.is_suggestion = True
    result.source_annotation = MCP_SOURCE_TEMPLATE.format(provider_name=provider_name)
    if result.data is None:
        result.data = {}
    if isinstance(result.data, dict):
        result.data["_suggestion_only"] = True
    return result
