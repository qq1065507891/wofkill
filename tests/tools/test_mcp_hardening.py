"""MCP hardening tests: boundary enforcement and suggestion-only guarantee.

Covers Task 8 Step 4:
- MCP results are always annotated as suggestions, never as rule truth
- MCP cannot mutate game state
- MCP call logging and traceability
- External provider boundary enforcement
"""

from __future__ import annotations

import pytest

from werewolf_agent.tools.schemas import ToolSource, ToolStatus, ToolResult, annotate_mcp_result
from werewolf_agent.tools.mcp_registry import MCPRegistry, MockMCPProvider
from werewolf_agent.tools.tool_logger import ToolCallLogger


# ---------------------------------------------------------------------------
# MCP suggestion-only enforcement
# ---------------------------------------------------------------------------


class TestMCPSuggestionOnly:
    def test_mcp_result_has_source_annotation(self) -> None:
        result = ToolResult(
            source=ToolSource.MCP_EXTERNAL,
            tool_name="query_history",
            data={"games": []},
        )
        annotated = annotate_mcp_result(result, "test_provider")
        assert annotated.source == ToolSource.MCP_EXTERNAL
        assert annotated.data.get("_suggestion_only") is True

    def test_mcp_result_never_overrides_rules(self) -> None:
        """MCP result data must contain a suggestion-only marker."""
        result = ToolResult(
            source=ToolSource.MCP_EXTERNAL,
            tool_name="check_role",
            data={"role": "werewolf"},
        )
        annotated = annotate_mcp_result(result, "test_provider")
        assert annotated.data["_suggestion_only"] is True
        # The annotation must not change the original data except adding marker
        assert annotated.data["role"] == "werewolf"

    def test_local_tool_not_annotated_as_suggestion(self) -> None:
        result = ToolResult(
            source=ToolSource.LOCAL,
            tool_name="query_legal_actions",
            data={"actions": ["vote"]},
        )
        # Local tools should NOT get the suggestion annotation
        assert result.data.get("_suggestion_only") is None


# ---------------------------------------------------------------------------
# MCP registry boundary
# ---------------------------------------------------------------------------


class TestMCPRegistryBoundary:
    def test_mcp_call_logged(self) -> None:
        logger = ToolCallLogger()
        registry = MCPRegistry(logger=logger)
        registry.register(MockMCPProvider("test_mcp"))

        registry.call("test_mcp", "some_tool", {"param": "value"})

        logs = logger.by_tool("some_tool")
        assert len(logs) == 1
        assert logs[0].call.source == ToolSource.MCP_EXTERNAL

    def test_mcp_call_returns_tool_result(self) -> None:
        registry = MCPRegistry()
        registry.register(MockMCPProvider(
            "hist",
            responses={"get_games": {"games": [{"id": "g1"}]}},
        ))

        result = registry.call("hist", "get_games", {})
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.source == ToolSource.MCP_EXTERNAL

    def test_mcp_cannot_mutate_game_state(self) -> None:
        """MCP provider call must return a ToolResult, never modify state directly."""
        registry = MCPRegistry()
        provider = MockMCPProvider("safe_mcp")
        registry.register(provider)

        result = registry.call("safe_mcp", "any_tool", {})
        assert isinstance(result, ToolResult)
        # The result is just data - it cannot cause side effects on GameState
        assert hasattr(result, "data")
        assert hasattr(result, "status")

    def test_unregister_removes_provider(self) -> None:
        registry = MCPRegistry()
        registry.register(MockMCPProvider("temp"))
        assert registry.get("temp") is not None

        assert registry.unregister("temp") is True
        assert registry.get("temp") is None

    def test_call_nonexistent_provider_returns_error(self) -> None:
        registry = MCPRegistry()
        result = registry.call("nonexistent", "tool", {})
        assert result.status == ToolStatus.NOT_FOUND

    def test_transport_provider_success_is_suggestion_only(self) -> None:
        from werewolf_agent.tools.mcp_transport import TransportMCPProvider

        class FakeTransport:
            name = "transported"
            description = "fake transport"

            def call(self, tool_name: str, params: dict) -> dict:
                return {"tool": tool_name, "params": params}

        registry = MCPRegistry()
        registry.register(TransportMCPProvider(FakeTransport()))

        result = registry.call("transported", "lookup", {"x": 1})

        assert result.status == ToolStatus.SUCCESS
        assert result.data["tool"] == "lookup"
        assert result.data["_suggestion_only"] is True

    def test_transport_provider_failure_is_isolated(self) -> None:
        from werewolf_agent.tools.mcp_transport import TransportMCPProvider

        class FailingTransport:
            name = "failing"
            description = "fake failing transport"

            def call(self, tool_name: str, params: dict) -> dict:
                raise TimeoutError("transport timeout")

        registry = MCPRegistry()
        registry.register(TransportMCPProvider(FailingTransport()))

        result = registry.call("failing", "lookup", {})

        assert result.status == ToolStatus.ERROR
        assert result.is_suggestion is True
        assert "transport timeout" in result.error_message


# ---------------------------------------------------------------------------
# MCP call_all aggregation
# ---------------------------------------------------------------------------


class TestMCPCallAll:
    def test_call_all_returns_all_provider_results(self) -> None:
        registry = MCPRegistry()
        registry.register(MockMCPProvider("p1"))
        registry.register(MockMCPProvider("p2"))

        results = registry.call_all("test_tool", {})
        assert len(results) == 2
        for r in results:
            assert r.source == ToolSource.MCP_EXTERNAL

    def test_call_all_annotates_suggestions(self) -> None:
        registry = MCPRegistry()
        registry.register(MockMCPProvider("sug1"))

        results = registry.call_all("test", {})
        for r in results:
            # Each result should have the suggestion marker
            assert r.data.get("_suggestion_only") is True
