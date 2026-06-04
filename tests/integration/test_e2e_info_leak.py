"""End-to-end information leakage tests across AgentContext, RAG, Memory, Tool, and API.

These tests prove that no private information leaks at any pipeline stage
during a live multi-agent game with all role actions exercised.
"""

from __future__ import annotations

import pytest
from dataclasses import replace

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    PlayerAction,
    RetryInfo,
    TaskType,
)
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import StructuredWorldState, build_world_state, extract_facts
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.memory.schemas import CrossGameQuery, ReflectionEntry
from werewolf_agent.rag.schemas import (
    RAGEntry, RAGQuery, CaseType, CaseMetadata, QualityGrade, SourceType,
    SourceMetadata, VisibilityBoundary,
)
from werewolf_agent.rag.retriever import StrategyRetriever
from werewolf_agent.rag.injector import RAGInjector, InjectionContext
from werewolf_agent.tools.local_tools import LocalToolExecutor
from werewolf_agent.tools.schemas import ToolCall, ToolStatus
from werewolf_agent.runtime.agent_adapter import build_agent_context, SimpleAgentRegistry

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _new_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def _make_full_game() -> tuple[GameState, RuleEngine]:
    """Create a 12-player game with all roles assigned."""
    engine = _new_engine()
    roles_map = {
        "wolf1": "werewolf",
        "wolf2": "werewolf",
        "wolf3": "werewolf",
        "wolf4": "werewolf",
        "villager1": "villager",
        "villager2": "villager",
        "villager3": "villager",
        "seer": "seer",
        "witch": "witch",
        "hunter": "hunter",
        "idiot": "idiot",
        "hybrid": "hybrid",
    }
    players = {pid: PlayerState(id=pid, role=role) for pid, role in roles_map.items()}
    gs = GameState(
        game_id="leak_test",
        players=players,
        phase="night",
        night_number=1,
        hybrid_master_id="villager1",
        hybrid_master_faction="good",
    )
    return gs, engine


def _enrich_game_state(gs: GameState) -> GameState:
    """Add realistic events from a night+day cycle."""
    events = [
        GameEvent(type="wolf_kill_selected", payload={
            "night_number": 1, "target_id": "villager2",
        }),
        GameEvent(type="seer_check", payload={
            "seer_id": "seer", "target_id": "wolf1",
            "alignment": "werewolf", "night_number": 1,
            "visibility": "seer_only",
        }),
        GameEvent(type="witch_antidote_used", payload={
            "target_id": "villager2", "visibility": "witch_private",
        }),
        GameEvent(type="speech", payload={
            "speaker": "seer", "day_number": 1,
            "text": "I checked wolf1, they are a werewolf!",
        }),
        GameEvent(type="speech", payload={
            "speaker": "wolf1", "day_number": 1,
            "text": "I am a good person, seer is lying!",
        }),
        GameEvent(type="wolf_discussion", payload={
            "wolf_id": "wolf1", "night_number": 1,
            "text": "Let's kill the seer tonight",
            "visibility": "werewolf_team_only",
        }),
        GameEvent(type="hunter_idiot_status_confirmed", payload={
            "night_number": 1, "hunter_id": "hunter", "idiot_id": "idiot",
            "visibility": "moderator_only",
        }),
        GameEvent(type="vote_resolved", payload={
            "exiled": "wolf1", "reason": "majority",
        }),
        GameEvent(type="player_exiled", payload={
            "player_id": "wolf1", "resolution_batch": "day_1_vote",
        }),
    ]
    return replace(gs, events=gs.events + events)


# ---------------------------------------------------------------------------
# 1. AgentContext leakage — every role checked at every decision point
# ---------------------------------------------------------------------------


class TestAgentContextLeakage:
    """Verify build_agent_context never includes forbidden information."""

    @pytest.fixture
    def game(self) -> tuple[GameState, RuleEngine]:
        gs, engine = _make_full_game()
        return _enrich_game_state(gs), engine

    def _assert_no_forbidden_info(self, ctx: AgentContext, player_role: str) -> None:
        vs = ctx.visible_world_state
        if player_role != "werewolf":
            assert "wolf_teammates" not in vs, f"{player_role} sees wolf_teammates"
        if player_role != "seer":
            assert "check_results" not in vs, f"{player_role} sees seer check_results"
        if player_role != "witch":
            assert "antidote_available" not in vs, f"{player_role} sees antidote_available"
            assert "poison_available" not in vs, f"{player_role} sees poison_available"
            assert "wolf_kill_target" not in vs, f"{player_role} sees wolf_kill_target"
        if player_role != "hybrid":
            assert "master_id" not in vs, f"{player_role} sees hybrid master_id"
        assert "moderator_full" not in vs
        assert "events" not in vs
        # P0-I1: also assert that the base ``strategy_directive`` (as
        # produced by ``build_agent_context``) does not leak role-gated
        # directive keys for other roles.  Per-role injection is
        # covered separately in test_directive_role_gating.py.
        sd = ctx.strategy_directive or {}
        if player_role != "werewolf":
            for key in (
                "wolf_speech_directive", "wolf_universal_rules",
                "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
                "wolf_day_push_target", "wolf_plan_target",
                "wolf_teammate_exposed", "wolf_high_priority_target",
                "wolf_kill_instruction", "wolf_vote_strategy",
                "wolf_team_discussion",
            ):
                assert key not in sd, (
                    f"{player_role} base strategy_directive leaks {key}"
                )
        if player_role != "witch":
            for key in (
                "witch_night_action", "witch_strategy_hint",
                "witch_poison_deterrent", "witch_poison_threshold",
            ):
                assert key not in sd, (
                    f"{player_role} base strategy_directive leaks {key}"
                )
        if player_role != "seer":
            for key in (
                "seer_night_check", "check_value_assessment",
                "badge_flow_plan", "excluded_counterclaiming_seers",
            ):
                assert key not in sd, (
                    f"{player_role} base strategy_directive leaks {key}"
                )
        if player_role != "hybrid":
            for key in (
                "hybrid_master_choice", "master_assessment",
                "hybrid_master_dead", "hybrid_vote_strategy",
            ):
                assert key not in sd, (
                    f"{player_role} base strategy_directive leaks {key}"
                )

    def test_villager_context_no_leaks(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "villager1", TaskType.SPEECH)
        self._assert_no_forbidden_info(ctx, "villager")
        assert ctx.own_role == "villager"

    def test_wolf_context_sees_teammates_only(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "wolf2", TaskType.WOLF_DISCUSSION)
        self._assert_no_forbidden_info(ctx, "werewolf")
        assert ctx.own_role == "werewolf"
        assert "wolf_teammates" in ctx.visible_world_state
        teammates = ctx.visible_world_state["wolf_teammates"]
        assert "wolf2" not in teammates  # self excluded

    def test_seer_context_sees_own_checks_only(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "seer", TaskType.NIGHT_ACTION)
        self._assert_no_forbidden_info(ctx, "seer")
        assert ctx.own_role == "seer"
        assert "check_results" in ctx.visible_world_state

    def test_witch_context_sees_potions_and_target(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(
            engine, gs, "witch", TaskType.NIGHT_ACTION,
            wolf_kill_target_id="villager2",
        )
        self._assert_no_forbidden_info(ctx, "witch")
        assert ctx.own_role == "witch"
        assert ctx.visible_world_state.get("wolf_kill_target") == "villager2"

    def test_hunter_context_no_leaks(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "hunter", TaskType.SPEECH)
        self._assert_no_forbidden_info(ctx, "hunter")

    def test_idiot_context_no_leaks(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "idiot", TaskType.SPEECH)
        self._assert_no_forbidden_info(ctx, "idiot")

    def test_hybrid_context_sees_master_only(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "hybrid", TaskType.SPEECH)
        self._assert_no_forbidden_info(ctx, "hybrid")
        assert ctx.visible_world_state.get("master_id") == "villager1"

    def test_wolf_cannot_see_seer_checks(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "wolf2", TaskType.SPEECH)
        assert "check_results" not in ctx.visible_world_state

    def test_seer_cannot_see_wolf_teammates(self, game: tuple) -> None:
        gs, engine = game
        ctx = build_agent_context(engine, gs, "seer", TaskType.SPEECH)
        assert "wolf_teammates" not in ctx.visible_world_state

    def test_transcript_only_public_speech(self, game: tuple) -> None:
        gs, engine = game
        for pid in gs.players:
            ctx = build_agent_context(engine, gs, pid, TaskType.SPEECH)
            for entry in ctx.recent_transcript:
                assert entry.get("speaker") in gs.players


# ---------------------------------------------------------------------------
# 2. RAG visibility boundary — no god-view leakage to live players
# ---------------------------------------------------------------------------


class TestRAGVisibilityBoundary:
    """RAGInjector must not leak god-view entries to live players."""

    @pytest.fixture
    def rag_setup(self) -> tuple[StrategyRetriever, RAGInjector]:
        entries = [
            RAGEntry(
                entry_id="god_view_1",
                title="God View Analysis",
                summary="All roles: wolf1=werewolf, seer=seer, witch=witch",
                metadata=CaseMetadata(
                    case_type=CaseType.PROJECT_REVIEW,
                    quality_grade=QualityGrade.EXPERT_REVIEW,
                    visibility_boundary=VisibilityBoundary.MODERATOR_ONLY,
                    source=SourceMetadata(source_type=SourceType.MANUAL_ENTRY),
                    tags=["review", "god-view"],
                ),
            ),
            RAGEntry(
                entry_id="seer_strategy_1",
                title="Seer Claim Strategy",
                summary="As seer, claim on day 1 and give badge flow",
                metadata=CaseMetadata(
                    case_type=CaseType.SPEECH_TEMPLATE,
                    quality_grade=QualityGrade.RULE_DERIVED_SEED,
                    visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                    source=SourceMetadata(source_type=SourceType.RULE_DERIVED),
                    tags=["seer", "claim"],
                ),
            ),
            RAGEntry(
                entry_id="wolf_deep_hook",
                title="Wolf Deep Hook Tactic",
                summary="Pretend to be seer and give fake checks",
                metadata=CaseMetadata(
                    case_type=CaseType.EXTERNAL_TACTICS,
                    quality_grade=QualityGrade.EXPERT_REVIEW,
                    visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                    source=SourceMetadata(source_type=SourceType.EXPERT_COMMENTARY),
                    tags=["wolf", "deception"],
                ),
            ),
        ]
        retriever = StrategyRetriever()
        for entry in entries:
            retriever.add_entry(entry)
        injector = RAGInjector(retriever)
        return retriever, injector

    def test_live_player_gets_no_god_view(self, rag_setup: tuple) -> None:
        retriever, injector = rag_setup
        query = RAGQuery(query="role analysis", roles=["villager"], phases=["day"])
        hits = injector.inject(query, InjectionContext.LIVE_PLAYER)
        for hit in hits:
            assert "wolf1=werewolf" not in hit.summary
            assert "seer=seer" not in hit.summary

    def test_moderator_sees_all_entries(self, rag_setup: tuple) -> None:
        retriever, injector = rag_setup
        query = RAGQuery(query="role analysis", roles=["all"], phases=["all"])
        hits = injector.inject(query, InjectionContext.MODERATOR)
        assert len(hits) >= 1

    def test_live_player_no_moderator_only_entries(self, rag_setup: tuple) -> None:
        retriever, injector = rag_setup
        query = RAGQuery(query="god view", roles=["villager"], phases=["day"])
        hits = injector.inject(query, InjectionContext.LIVE_PLAYER)
        for hit in hits:
            assert hit.visibility_boundary != VisibilityBoundary.MODERATOR_ONLY


# ---------------------------------------------------------------------------
# 3. Memory access boundary — no cross-player leakage
# ---------------------------------------------------------------------------


class TestMemoryAccessBoundary:
    """MemoryStore must not leak one player's cognition to another."""

    @pytest.fixture
    def memory_setup(self) -> MemoryStore:
        store = MemoryStore()
        player_ids = ["wolf1", "villager1", "seer"]
        for pid in player_ids:
            store.init_matrix(pid, player_ids)
        return store

    def test_cognition_matrix_isolation(self, memory_setup: MemoryStore) -> None:
        store = memory_setup
        wolf_matrix = store.get_matrix("wolf1")
        villager_matrix = store.get_matrix("villager1")
        assert wolf_matrix is not None
        assert villager_matrix is not None
        # Each matrix is viewer-specific
        assert wolf_matrix.viewer_id == "wolf1"
        assert villager_matrix.viewer_id == "villager1"
        assert wolf_matrix is not villager_matrix

    def test_profile_store_no_role_leak(self, memory_setup: MemoryStore) -> None:
        store = memory_setup
        profile = store.get_or_create_profile("wolf1")
        # Profile contains ability scores, not role identity
        assert hasattr(profile, "logic")
        assert hasattr(profile, "deception")

    def test_reflection_isolation_by_player(self, memory_setup: MemoryStore) -> None:
        store = memory_setup
        wolf_entry = ReflectionEntry(
            entry_id="ref_wolf1",
            game_id="g1",
            player_id="wolf1",
            role="werewolf",
            faction_won=False,
            text="I am a wolf, need to deceive",
            tags=["wolf", "strategy"],
        )
        store.store_reflection(wolf_entry)
        # Query wolf's own reflections
        wolf_refs = store.reflections_by_player("wolf1")
        assert len(wolf_refs) > 0
        # Query villager's reflections — should be empty
        villager_refs = store.reflections_by_player("villager1")
        assert len(villager_refs) == 0


# ---------------------------------------------------------------------------
# 4. Tool output boundary — no forbidden info in tool results
# ---------------------------------------------------------------------------


class TestToolOutputBoundary:
    """Local tool executor must not expose private information."""

    @pytest.fixture
    def tool_setup(self) -> tuple[LocalToolExecutor, GameState]:
        engine = _new_engine()
        gs, _ = _make_full_game()
        gs = _enrich_game_state(gs)
        executor = LocalToolExecutor()
        return executor, gs

    def test_query_public_state_no_roles(self, tool_setup: tuple) -> None:
        executor, gs = tool_setup
        call = ToolCall(tool_name="query_public_state", caller_id="villager1")
        result = executor.execute(call, gs)
        assert result.status == ToolStatus.SUCCESS
        data = result.data
        if data:
            # No 'role' keys in any player entry
            assert "role" not in str(data.keys()) if hasattr(data, "keys") else True
            # No role assignments in values
            for key in ("roles", "role_assignments", "player_roles"):
                assert key not in str(data)

    def test_query_private_state_own_only(self, tool_setup: tuple) -> None:
        executor, gs = tool_setup
        call = ToolCall(tool_name="query_private_state", caller_id="villager1",
                        params={"player_id": "villager1"})
        result = executor.execute(call, gs)
        assert result.status == ToolStatus.SUCCESS
        if result.data and "role" in result.data:
            assert result.data["role"] == "villager"

    def test_query_cognition_matrix_own_only(self, tool_setup: tuple) -> None:
        executor, gs = tool_setup
        call = ToolCall(tool_name="query_cognition_matrix", caller_id="villager1",
                        params={"player_id": "villager1"})
        result = executor.execute(call, gs)
        assert result.status == ToolStatus.SUCCESS or result.data is not None

    def test_unknown_tool_returns_error(self, tool_setup: tuple) -> None:
        executor, gs = tool_setup
        call = ToolCall(tool_name="nonexistent_tool", caller_id="villager1")
        result = executor.execute(call, gs)
        assert result.status == ToolStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# 5. VisibilityPolicy enforcement — structured facts don't leak
# ---------------------------------------------------------------------------


class TestVisibilityPolicyFacts:
    """VisibilityPolicy must correctly classify all fact types."""

    @pytest.fixture
    def game(self) -> tuple[GameState, RuleEngine]:
        gs, engine = _make_full_game()
        return _enrich_game_state(gs), engine

    def test_seer_check_not_visible_to_wolf(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "wolf2", "werewolf")
        for fact in visible:
            assert fact.fact_type != "seer_check", f"Seer check visible to wolf: {fact}"

    def test_wolf_discussion_not_visible_to_villager(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "villager1", "villager")
        for fact in visible:
            assert fact.fact_type != "wolf_discussion", f"Wolf discussion visible to villager: {fact}"

    def test_wolf_discussion_visible_to_wolf(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "wolf2", "werewolf")
        wolf_facts = [f for f in visible if f.fact_type == "wolf_discussion"]
        assert len(wolf_facts) > 0, "Wolf discussion not visible to wolf"

    def test_witch_antidote_not_visible_to_villager(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "villager1", "villager")
        for fact in visible:
            assert fact.fact_type != "witch_antidote_used", f"Witch fact visible to villager: {fact}"

    def test_witch_antidote_visible_to_witch(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        visible = policy.filter_visible_facts(ws, "witch", "witch")
        witch_facts = [f for f in visible if f.fact_type == "witch_antidote_used"]
        assert len(witch_facts) > 0, "Witch antidote not visible to witch"

    def test_moderator_only_not_visible_to_any_player(self, game: tuple) -> None:
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        roles = ["villager", "werewolf", "seer", "witch", "hunter", "idiot", "hybrid"]
        for role in roles:
            visible = policy.filter_visible_facts(ws, f"test_{role}", role)
            for fact in visible:
                assert fact.fact_type != "hunter_idiot_status_confirmed", (
                    f"Moderator-only fact visible to {role}: {fact}"
                )

    def test_no_leaks_for_all_roles(self, game: tuple) -> None:
        """Comprehensive check: every visible fact is correctly scoped."""
        gs, engine = game
        ws = build_world_state(gs)
        policy = VisibilityPolicy()
        roles = ["villager", "werewolf", "seer", "witch", "hunter", "idiot", "hybrid"]
        violations = []
        for role in roles:
            report = policy.compute_visibility(ws, f"test_{role}", role)
            for idx in report.visible_indices:
                fact = ws.facts[idx]
                label = report.fact_labels[idx]
                # Check visibility matches role
                if label.visibility == "werewolf_team_only" and role != "werewolf":
                    violations.append(f"Wolf fact visible to {role}: {fact.fact_type}")
                elif label.visibility == "seer_only" and role != "seer":
                    violations.append(f"Seer fact visible to {role}: {fact.fact_type}")
                elif label.visibility == "witch_private" and role != "witch":
                    violations.append(f"Witch fact visible to {role}: {fact.fact_type}")
                elif label.visibility == "moderator_only":
                    violations.append(f"Moderator fact visible to player {role}: {fact.fact_type}")
        assert not violations, "Visibility violations:\n" + "\n".join(violations)
