"""Tool tests: local tools, MCP registry, call logger, boundary enforcement."""

import pytest

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.tools.schemas import (
    InternalToolName,
    MCPProvider,
    ToolCall,
    ToolResult,
    ToolSource,
    ToolStatus,
    annotate_mcp_result,
    MCP_SOURCE_TEMPLATE,
)
from werewolf_agent.tools.local_tools import LocalToolExecutor
from werewolf_agent.tools.mcp_registry import (
    MCPRegistry,
    MockMCPProvider,
    ExternalHistoryProvider,
    ExternalProfileProvider,
)
from werewolf_agent.tools.tool_logger import ToolCallLogger


def _make_state(**overrides) -> GameState:
    defaults = {
        "game_id": "g1",
        "phase": "day_discussion",
        "day_number": 1,
        "players": {
            "p1": PlayerState(id="p1", role="seer", alive=True),
            "p2": PlayerState(id="p2", role="werewolf", alive=True),
            "p3": PlayerState(id="p3", role="villager", alive=True),
            "p4": PlayerState(id="p4", role="witch", alive=True),
            "p5": PlayerState(id="p5", role="hunter", alive=True, vote_enabled=False),
        },
    }
    defaults.update(overrides)
    return GameState(**defaults)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:

    def test_tool_call_creation(self):
        call = ToolCall(tool_name="test", source=ToolSource.LOCAL, caller_id="p1")
        assert call.tool_name == "test"
        assert call.source == ToolSource.LOCAL

    def test_tool_result_creation(self):
        result = ToolResult(tool_name="test", source=ToolSource.LOCAL, status=ToolStatus.SUCCESS, data={"a": 1})
        assert result.status == ToolStatus.SUCCESS
        assert result.data == {"a": 1}

    def test_mcp_result_is_suggestion(self):
        result = ToolResult(tool_name="test", source=ToolSource.MCP_EXTERNAL)
        assert result.is_suggestion is False  # Default false
        result = annotate_mcp_result(result, "test_provider")
        assert result.is_suggestion is True
        assert "test_provider" in result.source_annotation
        assert "建议" in result.source_annotation

    def test_internal_tool_names(self):
        assert InternalToolName.QUERY_LEGAL_ACTIONS.value == "query_legal_actions"
        assert InternalToolName.QUERY_PUBLIC_STATE.value == "query_public_state"
        assert InternalToolName.WRITE_REVIEW.value == "write_review"

    def test_mcp_source_template(self):
        annotated = MCP_SOURCE_TEMPLATE.format(provider_name="ext_hist")
        assert "ext_hist" in annotated
        assert "建议" in annotated


# ---------------------------------------------------------------------------
# ToolCallLogger
# ---------------------------------------------------------------------------

class TestToolCallLogger:

    def test_log_and_count(self):
        logger = ToolCallLogger()
        call = ToolCall(tool_name="test", source=ToolSource.LOCAL)
        result = ToolResult(tool_name="test", source=ToolSource.LOCAL, status=ToolStatus.SUCCESS)
        logger.log(call, result)
        assert logger.count() == 1

    def test_by_tool(self):
        logger = ToolCallLogger()
        for i in range(3):
            call = ToolCall(tool_name="tool_a", source=ToolSource.LOCAL)
            result = ToolResult(tool_name="tool_a", source=ToolSource.LOCAL)
            logger.log(call, result)
        call = ToolCall(tool_name="tool_b", source=ToolSource.LOCAL)
        result = ToolResult(tool_name="tool_b", source=ToolSource.LOCAL)
        logger.log(call, result)
        assert len(logger.by_tool("tool_a")) == 3
        assert len(logger.by_tool("tool_b")) == 1

    def test_by_caller(self):
        logger = ToolCallLogger()
        call = ToolCall(tool_name="t", source=ToolSource.LOCAL, caller_id="p1")
        result = ToolResult(tool_name="t", source=ToolSource.LOCAL)
        logger.log(call, result)
        assert len(logger.by_caller("p1")) == 1
        assert len(logger.by_caller("p2")) == 0

    def test_by_source(self):
        logger = ToolCallLogger()
        local_call = ToolCall(tool_name="t", source=ToolSource.LOCAL)
        mcp_call = ToolCall(tool_name="t", source=ToolSource.MCP_EXTERNAL)
        result = ToolResult(tool_name="t", source=ToolSource.LOCAL)
        logger.log(local_call, result)
        logger.log(mcp_call, result)
        assert len(logger.local_calls()) == 1
        assert len(logger.mcp_calls()) == 1

    def test_errors(self):
        logger = ToolCallLogger()
        call = ToolCall(tool_name="t", source=ToolSource.LOCAL)
        ok = ToolResult(tool_name="t", source=ToolSource.LOCAL, status=ToolStatus.SUCCESS)
        err = ToolResult(tool_name="t", source=ToolSource.LOCAL, status=ToolStatus.ERROR, error_message="fail")
        logger.log(call, ok)
        logger.log(call, err)
        assert len(logger.errors()) == 1

    def test_max_entries(self):
        logger = ToolCallLogger(max_entries=5)
        for i in range(10):
            call = ToolCall(tool_name=f"t{i}", source=ToolSource.LOCAL)
            result = ToolResult(tool_name=f"t{i}", source=ToolSource.LOCAL)
            logger.log(call, result)
        assert logger.count() == 5

    def test_summary(self):
        logger = ToolCallLogger()
        call = ToolCall(tool_name="t", source=ToolSource.LOCAL, caller_id="p1")
        result = ToolResult(tool_name="t", source=ToolSource.LOCAL, status=ToolStatus.SUCCESS)
        logger.log(call, result)
        s = logger.summary()
        assert s["total_calls"] == 1
        assert s["local_calls"] == 1
        assert s["mcp_calls"] == 0

    def test_clear(self):
        logger = ToolCallLogger()
        call = ToolCall(tool_name="t", source=ToolSource.LOCAL)
        result = ToolResult(tool_name="t", source=ToolSource.LOCAL)
        logger.log(call, result)
        logger.clear()
        assert logger.count() == 0


# ---------------------------------------------------------------------------
# LocalToolExecutor
# ---------------------------------------------------------------------------

class TestLocalToolExecutor:

    def test_query_legal_actions(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert "speech" in result.data["legal_actions"]

    def test_query_legal_actions_night_seer(self):
        executor = LocalToolExecutor()
        state = _make_state(phase="night")
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert "check_alignment" in result.data["legal_actions"]

    def test_query_legal_actions_night_witch(self):
        executor = LocalToolExecutor()
        state = _make_state(phase="night")
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p4",
            params={"player_id": "p4"},
        )
        result = executor.execute(call, state)
        assert "use_antidote" not in result.data["legal_actions"]
        assert "use_poison" in result.data["legal_actions"]

    def test_query_legal_actions_night_witch_sees_antidote_only_with_kill_target(self):
        executor = LocalToolExecutor()
        state = _make_state(
            phase="night",
            night_number=1,
            events=[GameEvent(type="wolf_kill_selected", payload={"night_number": 1, "target_id": "p3"})],
        )
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p4",
            params={"player_id": "p4"},
        )
        result = executor.execute(call, state)
        assert "use_antidote" in result.data["legal_actions"]
        assert "use_poison" in result.data["legal_actions"]
        assert result.data["legal_targets"]["use_antidote"] == ["p3"]
        assert "p3" in result.data["legal_targets"]["use_poison"]

    def test_query_legal_actions_night_wolf_can_no_kill(self):
        executor = LocalToolExecutor()
        state = _make_state(phase="night")
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p2",
            params={"player_id": "p2"},
        )
        result = executor.execute(call, state)
        assert "wolf_kill" in result.data["legal_actions"]
        assert "wolf_no_kill" in result.data["legal_actions"]

    def test_query_legal_actions_dead_player(self):
        executor = LocalToolExecutor()
        state = _make_state(players={
            "p1": PlayerState(id="p1", role="seer", alive=False),
        })
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.data["legal_actions"] == []

    def test_query_legal_actions_not_found(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p99",
            params={"player_id": "p99"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["legal_actions"] == []

    def test_query_legal_actions_other_player_denied(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p2"},
        )

        result = executor.execute(call, state)

        assert result.status == ToolStatus.UNAUTHORIZED
        assert result.data is None

    def test_query_public_state(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PUBLIC_STATE.value,
            source=ToolSource.LOCAL,
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["day_number"] == 1
        assert len(result.data["alive_players"]) == 5
        assert result.data["total_players"] == 5

    def test_query_private_state_seer(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["role"] == "seer"
        assert result.data["alive"] is True

    def test_query_private_state_other_player_denied(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p2"},
        )

        result = executor.execute(call, state)

        assert result.status == ToolStatus.UNAUTHORIZED
        assert result.data is None

    def test_query_private_state_witch_potions(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p4",
            params={"player_id": "p4"},
        )
        result = executor.execute(call, state)
        assert result.data["antidote_available"] is True
        assert result.data["poison_available"] is True
        assert result.data["current_wolf_kill_target_id"] is None

    def test_query_private_state_witch_used(self):
        executor = LocalToolExecutor()
        state = _make_state(antidote_used=True)
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p4",
            params={"player_id": "p4"},
        )
        result = executor.execute(call, state)
        assert result.data["antidote_available"] is False
        assert result.data["poison_available"] is True

    def test_query_private_state_witch_sees_current_kill_target_only_when_selected(self):
        executor = LocalToolExecutor()
        selected = _make_state(
            phase="night",
            night_number=1,
            events=[GameEvent(type="wolf_kill_selected", payload={"night_number": 1, "target_id": "p3"})],
        )
        no_kill = _make_state(
            phase="night",
            night_number=1,
            events=[GameEvent(type="wolf_no_kill_declared", payload={"night_number": 1})],
        )

        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p4",
            params={"player_id": "p4"},
        )

        assert executor.execute(call, selected).data["current_wolf_kill_target_id"] == "p3"
        assert executor.execute(call, no_kill).data["current_wolf_kill_target_id"] is None

    def test_query_private_state_not_found(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PRIVATE_STATE.value,
            source=ToolSource.LOCAL,
            params={"player_id": "p99"},
        )
        result = executor.execute(call, state)
        assert "error" in result.data

    def test_query_relation_graph(self):
        executor = LocalToolExecutor()
        state = _make_state()
        state.events.append(GameEvent(
            type="vote",
            payload={"voter": "p1", "target": "p2", "day_number": 1},
        ))
        call = ToolCall(
            tool_name=InternalToolName.QUERY_RELATION_GRAPH.value,
            source=ToolSource.LOCAL,
            params={"predicate": "voted"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["count"] == 1

    def test_query_relation_graph_filter_source(self):
        executor = LocalToolExecutor()
        state = _make_state()
        state.events.append(GameEvent(type="vote", payload={"voter": "p1", "target": "p2", "day_number": 1}))
        state.events.append(GameEvent(type="vote", payload={"voter": "p3", "target": "p2", "day_number": 1}))
        call = ToolCall(
            tool_name=InternalToolName.QUERY_RELATION_GRAPH.value,
            source=ToolSource.LOCAL,
            params={"source": "p1"},
        )
        result = executor.execute(call, state)
        assert result.data["count"] == 1

    def test_query_cognition_matrix(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_COGNITION_MATRIX.value,
            source=ToolSource.LOCAL,
            params={"viewer_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["viewer_id"] == "p1"

    def test_write_review_game_active(self):
        executor = LocalToolExecutor()
        state = _make_state()  # winning_faction = None
        call = ToolCall(
            tool_name=InternalToolName.WRITE_REVIEW.value,
            source=ToolSource.LOCAL,
            params={"player_id": "p1", "review_text": "test"},
        )
        result = executor.execute(call, state)
        assert result.status == ToolStatus.SUCCESS
        assert "error" in result.data

    def test_write_review_game_ended(self):
        executor = LocalToolExecutor()
        state = _make_state(winning_faction="good")
        call = ToolCall(
            tool_name=InternalToolName.WRITE_REVIEW.value,
            source=ToolSource.LOCAL,
            params={"player_id": "p1", "review_text": "复盘内容"},
        )
        result = executor.execute(call, state)
        assert result.data["status"] == "recorded"

    def test_call_evaluator(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.CALL_EVALUATOR.value,
            source=ToolSource.LOCAL,
        )
        result = executor.execute(call, state)
        assert result.data["alive_good"] == 4  # seer + villager + witch + hunter
        assert result.data["alive_wolves"] == 1

    def test_read_experiment_config(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.READ_EXPERIMENT_CONFIG.value,
            source=ToolSource.LOCAL,
        )
        result = executor.execute(call, state)
        assert result.data["ruleset_id"] == "pre_witch_hunter_idiot_mixed"
        assert result.data["game_id"] == "g1"

    def test_generate_game_report(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.GENERATE_GAME_REPORT.value,
            source=ToolSource.LOCAL,
        )
        result = executor.execute(call, state)
        assert result.data["game_id"] == "g1"
        assert result.data["day_number"] == 1

    def test_unknown_tool(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(tool_name="nonexistent_tool", source=ToolSource.LOCAL)
        result = executor.execute(call, state)
        assert result.status == ToolStatus.NOT_FOUND

    def test_all_calls_logged(self):
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PUBLIC_STATE.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
        )
        executor.execute(call, state)
        assert executor.logger.count() == 1
        assert executor.logger.by_caller("p1")[0].call.tool_name == "query_public_state"


# ---------------------------------------------------------------------------
# MCP Registry
# ---------------------------------------------------------------------------

class TestMCPRegistry:

    def test_register_and_count(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="mock1"))
        assert registry.count() == 1

    def test_register_multiple(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="a"))
        registry.register(MockMCPProvider(name="b"))
        assert registry.count() == 2

    def test_unregister(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="mock1"))
        assert registry.unregister("mock1")
        assert registry.count() == 0
        assert not registry.unregister("mock1")

    def test_get_provider(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="mock1"))
        p = registry.get("mock1")
        assert p is not None
        assert p.name == "mock1"

    def test_get_nonexistent(self):
        registry = MCPRegistry()
        assert registry.get("nonexistent") is None

    def test_call_success(self):
        registry = MCPRegistry()
        provider = MockMCPProvider(
            name="mock1",
            responses={"query_data": {"result": "ok"}},
        )
        registry.register(provider)
        result = registry.call("mock1", "query_data", {})
        assert result.status == ToolStatus.SUCCESS
        assert result.data == {"result": "ok", "_suggestion_only": True}
        assert result.is_suggestion is True
        assert "mock1" in result.source_annotation

    def test_call_provider_not_found(self):
        registry = MCPRegistry()
        result = registry.call("nonexistent", "test", {})
        assert result.status == ToolStatus.NOT_FOUND
        assert result.is_suggestion is True

    def test_call_tool_not_in_provider(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="mock1"))
        result = registry.call("mock1", "nonexistent_tool", {})
        assert result.status == ToolStatus.NOT_FOUND

    def test_call_all(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="a", responses={"t": {"from": "a"}}))
        registry.register(MockMCPProvider(name="b", responses={"t": {"from": "b"}}))
        results = registry.call_all("t", {})
        assert len(results) == 2
        assert all(r.is_suggestion for r in results)

    def test_call_logged(self):
        logger = ToolCallLogger()
        registry = MCPRegistry(logger=logger)
        registry.register(MockMCPProvider(name="mock1", responses={"test": {"ok": True}}))
        registry.call("mock1", "test", {"x": 1})
        assert logger.count() == 1
        assert logger.mcp_calls()[0].call.tool_name == "test"

    def test_provider_names(self):
        registry = MCPRegistry()
        registry.register(MockMCPProvider(name="a"))
        registry.register(MockMCPProvider(name="b"))
        names = registry.provider_names()
        assert "a" in names
        assert "b" in names

    def test_external_history_provider(self):
        registry = MCPRegistry()
        registry.register(ExternalHistoryProvider())
        result = registry.call("external_history", "query_external_games", {"player_id": "p1"})
        assert result.status == ToolStatus.SUCCESS
        assert result.is_suggestion is True
        assert "external_history" in result.source_annotation

    def test_external_profile_provider(self):
        registry = MCPRegistry()
        registry.register(ExternalProfileProvider())
        result = registry.call("external_profiles", "query_player_profile", {"player_id": "p1"})
        assert result.status == ToolStatus.SUCCESS
        assert result.is_suggestion is True

    def test_mcp_result_does_not_override_local(self):
        """MCP results are suggestions, not referee facts."""
        registry = MCPRegistry()
        registry.register(MockMCPProvider(
            name="fake_rules",
            responses={"check_role": {"role": "werewolf"}},
        ))
        result = registry.call("fake_rules", "check_role", {"player_id": "p1"})
        assert result.is_suggestion is True
        assert "非裁判事实" in result.source_annotation
        # The result data exists but must be treated as suggestion
        assert result.data["role"] == "werewolf"


# ---------------------------------------------------------------------------
# Boundary enforcement: local vs MCP
# ---------------------------------------------------------------------------

class TestToolBoundary:

    def test_local_tools_do_not_use_mcp(self):
        """Local tool results have source=LOCAL, not MCP."""
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_PUBLIC_STATE.value,
            source=ToolSource.LOCAL,
        )
        result = executor.execute(call, state)
        assert result.source == ToolSource.LOCAL
        assert result.is_suggestion is False

    def test_mcp_results_always_suggestions(self):
        """MCP results are always marked as suggestions."""
        registry = MCPRegistry()
        registry.register(MockMCPProvider(
            name="ext",
            responses={"t": {"data": 42}},
        ))
        result = registry.call("ext", "t", {})
        assert result.is_suggestion is True
        assert result.source == ToolSource.MCP_EXTERNAL

    def test_rule_engine_not_in_mcp(self):
        """Rule engine queries are local tools only, not MCP."""
        # Local tools include legal actions queries
        executor = LocalToolExecutor()
        state = _make_state()
        call = ToolCall(
            tool_name=InternalToolName.QUERY_LEGAL_ACTIONS.value,
            source=ToolSource.LOCAL,
            caller_id="p1",
            params={"player_id": "p1"},
        )
        result = executor.execute(call, state)
        assert result.source == ToolSource.LOCAL
        assert result.status == ToolStatus.SUCCESS
        # MCP registry has no such tool
        registry = MCPRegistry()
        assert registry.get("rule_engine") is None

    def test_game_state_mutation_local_only(self):
        """Game state writes (review) are local only."""
        executor = LocalToolExecutor()
        state = _make_state(winning_faction="good")
        call = ToolCall(
            tool_name=InternalToolName.WRITE_REVIEW.value,
            source=ToolSource.LOCAL,
            params={"player_id": "p1", "review_text": "test"},
        )
        result = executor.execute(call, state)
        assert result.source == ToolSource.LOCAL
        assert result.is_suggestion is False

    def test_logger_distinguishes_sources(self):
        """Logger correctly separates local and MCP calls."""
        logger = ToolCallLogger()

        # Local call
        executor = LocalToolExecutor(logger=logger)
        state = _make_state()
        local_call = ToolCall(
            tool_name=InternalToolName.QUERY_PUBLIC_STATE.value,
            source=ToolSource.LOCAL,
        )
        executor.execute(local_call, state)

        # MCP call
        registry = MCPRegistry(logger=logger)
        registry.register(MockMCPProvider(name="mock", responses={"t": {}}))
        registry.call("mock", "t", {})

        assert logger.count() == 2
        assert len(logger.local_calls()) == 1
        assert len(logger.mcp_calls()) == 1

    def test_core_rules_not_mcp_path(self):
        """Design doc §11.2: core rules don't go through MCP."""
        # Verify that internal tool names cover rule/state queries
        rule_tools = {
            InternalToolName.QUERY_LEGAL_ACTIONS,
            InternalToolName.QUERY_PUBLIC_STATE,
            InternalToolName.QUERY_PRIVATE_STATE,
        }
        for tool in rule_tools:
            executor = LocalToolExecutor()
            state = _make_state()
            call = ToolCall(tool_name=tool.value, source=ToolSource.LOCAL, params={"player_id": "p1"})
            result = executor.execute(call, state)
            assert result.source == ToolSource.LOCAL
            assert result.is_suggestion is False
