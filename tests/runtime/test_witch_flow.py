from __future__ import annotations

import pytest
from dataclasses import replace

from typing import Any

from werewolf_agent.core.models import Death, GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo, FallbackAction,
    TaskType,
)
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    build_game_graph_with_checkpoint,
    _new_engine,
    _alive_wolves,
    _alive_non_wolves,
    _find_role,
    _stable_seed,
    check_victory,
    free_discussion,
    wolf_consensus,
    route_after_resolve_night,
    route_after_hunter_shot,
    route_after_post_exile,
    _sheriff_died_this_batch,
    _route_after_badge_transfer,
    _action_trace_event,
)
from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log
from werewolf_agent.runtime.checkpoints import make_checkpointer



RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


class _WitchMockAgent:
    """Mock PlayerAgent that returns a configured witch action."""
    def __init__(self, *, use_antidote: bool = False, poison_target_id: str | None = None,
                 no_action: bool = False) -> None:
        self._use_antidote = use_antidote
        self._poison_target_id = poison_target_id
        self._no_action = no_action
        self.last_context: Any = None

    def act(self, context: AgentContext) -> tuple[PlayerAction, RetryInfo]:
        self.last_context = context
        if self._no_action:
            return PlayerAction(action_type=ActionType.NO_ACTION, reason="witch_pass"), RetryInfo()
        if self._use_antidote:
            return PlayerAction(
                action_type=ActionType.USE_ANTIDOTE,
                target_id=context.visible_world_state.get("wolf_kill_target"),
                reason="save target",
                confidence=0.85,
            ), RetryInfo()
        if self._poison_target_id:
            return PlayerAction(
                action_type=ActionType.USE_POISON,
                target_id=self._poison_target_id,
                reason="poison suspect",
                confidence=0.7,
            ), RetryInfo()
        return PlayerAction(action_type=ActionType.NO_ACTION, reason="no potion"), RetryInfo()

class _WitchMockRegistry:
    """Mock AgentRegistry that serves _WitchMockAgent for the witch player."""
    def __init__(self, *, witch_id: str = "witch",
                 use_antidote: bool = False, poison_target_id: str | None = None,
                 no_action: bool = False) -> None:
        self._witch_id = witch_id
        self._agent = _WitchMockAgent(
            use_antidote=use_antidote,
            poison_target_id=poison_target_id,
            no_action=no_action,
        )
        self.witch_called: bool = False

    def get_agent(self, player_id: str) -> _WitchMockAgent | None:
        if player_id == self._witch_id:
            self.witch_called = True
            return self._agent
        return None

class TestWitchDecisionFlow:
    """Witch agent-driven night decisions with legal information boundary,
    private audit events, and correct potion constraints."""

    def _make_witch_state(
        self,
        *,
        wolf_kill_target_id: str | None = "v1",
        antidote_used: bool = False,
        poison_used: bool = False,
    ) -> tuple[RuntimeState, RuleEngine]:
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
            "w4": PlayerState(id="w4", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "v3": PlayerState(id="v3", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
            "seer": PlayerState(id="seer", role="seer"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "idiot": PlayerState(id="idiot", role="idiot"),
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
        }
        gs = GameState(
            game_id="witch_test",
            players=players,
            phase="night",
            night_number=1,
            antidote_used=antidote_used,
            poison_used=poison_used,
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": wolf_kill_target_id,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
            "hybrid_master_target_id": None,
            "self_destruct_wolf_id": None,
            "exile_votes": {},
            "revote": False,
            "sheriff_candidates": [],
            "sheriff_votes": {},
            "sheriff_withdrawing": [],
            "badge_decision": "tear",
            "badge_target_id": None,
            "hunter_shot_target_id": None,
        }
        return state, engine

    def test_witch_agent_sees_wolf_kill_target(self) -> None:
        """When wolves killed someone, witch agent context includes wolf_kill_target."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ctx is not None
        assert ctx.visible_world_state.get("wolf_kill_target") == "v1"
        assert ActionType.USE_ANTIDOTE in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_agent_no_kill_target_when_wolves_no_kill(self) -> None:
        """When wolves no-killed, witch sees no wolf_kill_target in context."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id=None)
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ctx.visible_world_state.get("wolf_kill_target") is None
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_cannot_self_save_antidote_not_offered(self) -> None:
        """When witch is the wolf kill target, antidote is NOT in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="witch")
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_antidote_already_used_no_option(self) -> None:
        """When antidote already used, USE_ANTIDOTE not in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(
            wolf_kill_target_id="v1", antidote_used=True,
        )
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE not in ctx.legal_actions
        assert ActionType.USE_POISON in ctx.legal_actions

    def test_witch_poison_already_used_no_option(self) -> None:
        """When poison already used, USE_POISON not in legal_actions."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine = self._make_witch_state(
            wolf_kill_target_id="v1", poison_used=True,
        )
        registry = _WitchMockRegistry(no_action=True)

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry._agent.last_context
        assert ActionType.USE_ANTIDOTE in ctx.legal_actions
        assert ActionType.USE_POISON not in ctx.legal_actions

    def test_witch_antidote_decision_produces_audit_event(self) -> None:
        """night_witch node produces witch_decision_audit event with witch_private visibility."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(use_antidote=True)

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("visibility") == "witch_private"
        assert audit_events[0].payload.get("action_taken") == "use_antidote"
        assert audit_events[0].payload.get("wolf_kill_target_id") == "v1"
        assert result["use_antidote"] is True
        assert result.get("poison_target_id") is None

    def test_witch_poison_decision_produces_audit_event(self) -> None:
        """night_witch node produces witch_decision_audit for poison."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(poison_target_id="w1")

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("action_taken") == "use_poison"
        assert audit_events[0].payload.get("poison_target_id") == "w1"
        assert result["use_antidote"] is False
        assert result.get("poison_target_id") == "w1"

    def test_witch_no_action_produces_audit_event(self) -> None:
        """Even witch no-action produces an audit event."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        registry = _WitchMockRegistry(no_action=True)

        result = night_witch({
            **state,
            "agent_registry": registry,
        })

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 1
        assert audit_events[0].payload.get("action_taken") == "no_action"

    def test_witch_scripted_no_audit_event(self) -> None:
        """Scripted fallback (no registry) should NOT produce audit event."""
        from werewolf_agent.runtime.graph import night_witch
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")

        result = night_witch(state)

        gs = result.get("game_state", state["game_state"])
        audit_events = [e for e in gs.events
                        if e.type == "witch_decision_audit"]
        assert len(audit_events) == 0

    def test_witch_resolve_night_events_have_visibility(self) -> None:
        """RuleEngine resolve_night witch events carry witch_private visibility."""
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        gs = state["game_state"]

        gs_out, events = engine.resolve_night(
            gs,
            night_number=1,
            wolf_kill_target_id="v1",
            use_antidote=True,
            poison_target_id=None,
        )
        witch_events = [e for e in events if e.type in ("witch_antidote_used", "witch_poison_used")]
        assert len(witch_events) == 1
        assert witch_events[0].payload.get("visibility") == "witch_private"

    def test_witch_poison_event_has_visibility(self) -> None:
        """RuleEngine resolve_night witch_poison_used event has witch_private visibility."""
        state, engine = self._make_witch_state(wolf_kill_target_id="v1")
        gs = state["game_state"]

        gs_out, events = engine.resolve_night(
            gs,
            night_number=1,
            wolf_kill_target_id="v1",
            use_antidote=False,
            poison_target_id="w1",
        )
        witch_events = [e for e in events if e.type in ("witch_antidote_used", "witch_poison_used")]
        assert len(witch_events) == 1
        assert witch_events[0].type == "witch_poison_used"
        assert witch_events[0].payload.get("visibility") == "witch_private"

class TestWitchPoisonPressureContext:
    """Witch night context must include pressure targets when they exist."""

    def test_witch_context_includes_poison_pressure_targets(self) -> None:
        """When public state has unresolved black claim, witch sees pressure targets."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=2,
            players=players,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "p05",
                        "day_number": 1,
                        "text": "我是预言家，p03查杀",
                    },
                ),
                GameEvent(
                    type="seer_check",
                    payload={
                        "seer_id": "p05",
                        "target_id": "p03",
                        "alignment": "wolf",
                        "night_number": 1,
                        "visibility": "seer_only",
                    },
                ),
            ],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION, ActionType.USE_POISON],
            legal_targets=["p03"],
        )

        # Witch context must include poison_pressure_targets
        assert "poison_pressure_targets" in context.visible_world_state
        targets = context.visible_world_state["poison_pressure_targets"]
        assert len(targets) >= 1
        black_claim_targets = [
            t for t in targets if t["pressure_type"] == "black_claim"
        ]
        assert len(black_claim_targets) == 1
        assert black_claim_targets[0]["player_id"] == "p03"
        assert black_claim_targets[0]["source"] == "p05"

    def test_witch_context_no_pressure_targets_without_claims(self) -> None:
        """When there are no public black claims, no pressure targets appear."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test_no_pressure",
            phase="night",
            night_number=1,
            players=players,
            events=[],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION, ActionType.USE_POISON],
            legal_targets=["p03"],
        )

        assert "poison_pressure_targets" not in context.visible_world_state

    def test_witch_context_no_pressure_when_poison_used(self) -> None:
        """When poison is already used, no pressure targets are added."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players: dict[str, PlayerState] = {}
        for i in range(1, 13):
            role = "werewolf" if i <= 4 else ("witch" if i == 9 else "villager")
            players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)

        gs = GameState(
            game_id="test_poison_used",
            phase="night",
            night_number=2,
            players=players,
            poison_used=True,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "p05",
                        "day_number": 1,
                        "text": "p03查杀",
                    },
                ),
            ],
        )
        engine = RuleEngine.from_yaml(RULESET_PATH)

        context = build_agent_context(
            engine, gs, "p09", TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.NO_ACTION],
            legal_targets=[],
        )

        # Poison already used, so no pressure targets added
        assert "poison_pressure_targets" not in context.visible_world_state

    def test_witch_pressure_strategy_directive_in_night_witch(self) -> None:
        """agent_night_witch adds strategy_directive when pressure targets exist."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch

        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "w3": PlayerState(id="w3", role="werewolf"),
            "w4": PlayerState(id="w4", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "v2": PlayerState(id="v2", role="villager"),
            "v3": PlayerState(id="v3", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
            "seer": PlayerState(id="seer", role="seer"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "idiot": PlayerState(id="idiot", role="idiot"),
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
        }
        gs = GameState(
            game_id="witch_pressure_directive",
            players=players,
            phase="night",
            night_number=2,
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "seer",
                        "day_number": 1,
                        "text": "我是预言家，w1查杀",
                    },
                ),
            ],
        )
        engine = _new_engine()

        class CaptureWitchAgent:
            """Agent that captures context and returns no_action."""
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(
                        action_type=ActionType.NO_ACTION,
                        reason="saving poison",
                    ),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureWitchAgent()

            def get_agent(self, player_id):
                if player_id == "witch":
                    return self.agent
                return None

        registry = CaptureRegistry()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "v1",
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
            "hybrid_master_target_id": None,
            "self_destruct_wolf_id": None,
            "exile_votes": {},
            "revote": False,
            "sheriff_candidates": [],
            "sheriff_votes": {},
            "sheriff_withdrawing": [],
            "badge_decision": "tear",
            "badge_target_id": None,
            "hunter_shot_target_id": None,
        }

        result = agent_night_witch(state, engine, registry)
        assert result is not None
        ctx = registry.agent.last_context
        assert ctx is not None
        # Strategy directive should mention pressure
        assert "witch_pressure" in ctx.strategy_directive
        assert "w1" in ctx.strategy_directive["witch_pressure"]
        assert "required_evaluation" in ctx.strategy_directive

    def test_build_agent_context_does_not_create_implicit_rag_service(self) -> None:
        """Context building stays pure unless runtime explicitly passes RAG service."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        players = {
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="werewolf"),
            "p03": PlayerState(id="p03", role="werewolf"),
            "p04": PlayerState(id="p04", role="werewolf"),
            "p05": PlayerState(id="p05", role="seer"),
            "p06": PlayerState(id="p06", role="witch"),
            "p07": PlayerState(id="p07", role="hunter"),
            "p08": PlayerState(id="p08", role="idiot"),
            "p09": PlayerState(id="p09", role="hybrid"),
            "p10": PlayerState(id="p10", role="villager"),
            "p11": PlayerState(id="p11", role="villager"),
            "p12": PlayerState(id="p12", role="villager"),
        }
        gs = GameState(
            game_id="rag_context",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            players=players,
            phase="night",
            night_number=2,
        )
        context = build_agent_context(
            _new_engine(),
            gs,
            "p01",
            TaskType.WOLF_DISCUSSION,
            legal_actions=[ActionType.SPEECH],
        )

        assert context.rag_hints == []
        assert [
            item for item in context.salience_items
            if item.get("type") == "rag_hit"
        ] == []

    def test_build_agent_context_uses_passed_rag_service(self) -> None:
        """Runtime can use a configured RAG service instead of seed fallback."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        class FakeRAGService:
            def __init__(self) -> None:
                self.last_query = None
                self.last_game_id = ""
                self.last_player_id = ""

            def retrieve_live_hints(self, query, *, game_id="", player_id=""):
                self.last_query = query
                self.last_game_id = game_id
                self.last_player_id = player_id
                return ["fake-hit"]

            def hits_to_context_items(self, hits, max_items=3):
                return [{
                    "type": "rag_hit",
                    "entry_id": "fake",
                    "title": "fake",
                    "allowed_in_live": True,
                }]

        players = {"p01": PlayerState(id="p01", role="werewolf")}
        gs = GameState(
            game_id="rag_service_context",
            ruleset_id="pre_witch_hunter_idiot_mixed",
            players=players,
            phase="night",
            night_number=1,
        )
        service = FakeRAGService()

        context = build_agent_context(
            _new_engine(),
            gs,
            "p01",
            TaskType.WOLF_DISCUSSION,
            legal_actions=[ActionType.SPEECH],
            rag_service=service,
        )

        assert service.last_query is not None
        assert service.last_query.role == "werewolf"
        assert service.last_query.phase == "night_discussion"
        assert service.last_game_id == "rag_service_context"
        assert service.last_player_id == "p01"
        assert any(item["entry_id"] == "fake" for item in context.rag_hints)
        assert not any(item.get("entry_id") == "fake" for item in context.salience_items)

    def test_good_player_cross_game_reflections_include_good_failure_lessons(self) -> None:
        """Good-side players should receive actionable good-side failure reflections."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        class FakeProfile:
            games_played = 3
            logic = 0.6
            deception = 0.2
            credibility = 0.7

        class FakeReflection:
            def __init__(
                self,
                entry_id: int,
                role: str,
                faction_won: bool,
                text: str,
                situation: str = "",
            ) -> None:
                self.entry_id = entry_id
                self.role = role
                self.faction_won = faction_won
                self.text = text
                self.situation = situation

        class FakeMemory:
            def get_profile(self, player_id):
                return FakeProfile()

            def reflections_by_player(self, player_id):
                return [
                    FakeReflection(
                        1,
                        "werewolf",
                        True,
                        "狼人赢在好人分票和女巫盲毒。",
                    ),
                    FakeReflection(
                        2,
                        "villager",
                        False,
                        "好人失败原因：没有把预言家查杀转成统一投票；改进：先按查验和票型排序。",
                        "D2放逐前",
                    ),
                ]

            def get_matrix(self, player_id):
                class Entry:
                    player_id = "p02"
                    faction_read = "wolf_lean"
                    trust = 0.25
                    key_evidence = ["连续跟票"]
                    open_questions = ["是否抱团"]

                class Matrix:
                    def all_entries(self):
                        return [Entry()]

                return Matrix()

        players = {
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        }
        gs = GameState(
            game_id="good_reflection_context",
            players=players,
            phase="day",
            day_number=2,
        )

        context = build_agent_context(
            _new_engine(),
            gs,
            "p01",
            TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            restored_memory=FakeMemory(),
        )

        assert "cross_game_reflections" not in context.strategy_directive
        reflections = context.reflection_memory_hints
        joined = str(reflections)
        assert "好人失败原因" in joined
        assert "改进" in joined
        assert joined.index("villager") < joined.index("werewolf")
        assert context.profile_memory_hint["games_played"] == 3
        assert context.cognition_matrix_hint["suspects"][0]["player"] == "p02"

    def test_skill_outputs_are_exposed_as_prompt_hints(self, monkeypatch) -> None:
        """Runtime skill analyses should populate the explicit prompt section."""
        import werewolf_agent.runtime.context as context_mod
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        def fake_inject_skill_output(
            strategy_directive,
            gs,
            player_id,
            world_state,
            belief_state,
            contradiction_alerts,
            phase,
            legal_targets=None,
            wolf_team_plan=None,
        ):
            return strategy_directive, {"skill_analyze_wolf_pit": "suspects: p02"}

        monkeypatch.setattr(context_mod, "_inject_skill_output", fake_inject_skill_output)

        players = {
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        }
        gs = GameState(game_id="skill_hint_context", players=players, phase="day")

        context = build_agent_context(
            _new_engine(),
            gs,
            "p01",
            TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
        )

        assert context.skill_analyses == {"skill_analyze_wolf_pit": "suspects: p02"}
        assert context.skill_analysis_hints == context.skill_analyses
