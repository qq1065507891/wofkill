# -*- coding: utf-8 -*-
"""
验证女巫行动、药剂约束与毒杀终局结算。

作者: Project contributors
修改日期: 2026-07-13
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo,
    TaskType,
)
from werewolf_agent.runtime.graph import (
    RuntimeState,
    _new_engine,
)



RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def test_witch_action_evidence_compares_multiple_legal_alternatives() -> None:
    from werewolf_agent.runtime.witch_night_directives import build_witch_action_evidence

    evidence = build_witch_action_evidence(
        legal_targets=["p02", "p03"],
        poison_candidates=[{"player_id": "p02", "reason": "公开查杀"}],
        wolf_kill_target_id=None,
    )

    assert evidence["alternative_comparison"]["legal_alternatives"] == ["p02", "p03"]
    assert evidence["alternative_comparison"]["no_legal_alternative"] is False
    assert evidence["retain_skill_evidence"]["available"] is True
    assert evidence["friendly_fire_risk"]["targets"] == ["p03"]


def test_witch_action_evidence_handles_single_and_zero_targets() -> None:
    from werewolf_agent.runtime.witch_night_directives import build_witch_action_evidence

    single = build_witch_action_evidence(
        legal_targets=["p02"], poison_candidates=[], wolf_kill_target_id=None
    )
    empty = build_witch_action_evidence(
        legal_targets=[], poison_candidates=[], wolf_kill_target_id=None
    )

    assert single["alternative_comparison"]["no_legal_alternative"] is True
    assert empty["alternative_comparison"]["legal_alternatives"] == []
    assert empty["alternative_comparison"]["no_legal_alternative"] is True


def test_witch_action_evidence_separates_action_specific_targets() -> None:
    import json

    from werewolf_agent.runtime.witch_night_directives import build_witch_action_evidence

    evidence = build_witch_action_evidence(
        legal_targets=["p02", "p03"],
        antidote_targets=["p02"],
        poison_targets=["p02", "p03"],
        poison_candidates=[{"player_id": "p03", "reason": "公开查杀"}],
        wolf_kill_target_id="p02",
    )

    assert evidence["antidote_targets"] == ["p02"]
    assert evidence["poison_targets"] == ["p02", "p03"]
    assert "p02" in evidence["friendly_fire_risk"]["targets"]
    assert evidence["retain_option"]["available"] is True
    assert evidence["retain_option"]["required"] is False
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert serialized.count('"antidote_targets"') == 1
    assert serialized.count('"poison_targets"') == 1
    assert serialized.count('"friendly_fire_risk"') == 1


def test_witch_only_no_action_requires_retain_option() -> None:
    from werewolf_agent.runtime.witch_night_directives import build_witch_action_evidence

    evidence = build_witch_action_evidence(
        legal_targets=[],
        antidote_targets=[],
        poison_targets=[],
        poison_candidates=[],
        wolf_kill_target_id=None,
    )

    assert evidence["retain_option"] == {
        "action": "no_action",
        "available": True,
        "required": True,
        "reason": "无合法用药目标",
    }


def test_poison_resolution_commits_victory_after_forced_death_reactions() -> None:
    from werewolf_agent.runtime.graph import resolve_night, route_after_resolve_night

    engine = _new_engine()
    gs = GameState(
        game_id="poison_terminal",
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "witch": PlayerState(id="witch", role="witch"),
            "good": PlayerState(id="good", role="villager"),
        },
        phase="night",
        night_number=1,
    )
    result = resolve_night({
        "game_state": gs,
        "engine": engine,
        "wolf_kill_target_id": "good",
        "use_antidote": False,
        "poison_target_id": "wolf",
        "seer_target_id": None,
    })

    assert result["game_state"].winning_faction == "good"
    assert len([e for e in result["game_state"].events if e.type == "victory"]) == 1
    assert route_after_resolve_night({**result, "engine": engine}) == "reflection"


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

            def hits_to_prompt_lines(self, hits, max_items=3):
                # P0-G1: live prompt path uses slim lines (title only,
                # no relevance/quality/source metadata).
                return [{
                    "title": "fake",
                    "summary": "fake-summary",
                    "key_decisions": ["fake-decision"],
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
        # P0-G1: live prompt now uses slim lines; only title/summary/
        # key_decisions are surfaced, no entry_id or audit metadata.
        assert any(item["title"] == "fake" for item in context.rag_hints)
        assert all(
            set(item.keys()) == {"title", "summary", "key_decisions"}
            for item in context.rag_hints
        )
        assert not any(item.get("entry_id") == "fake" for item in context.salience_items)

    def test_good_player_cross_game_reflections_include_good_failure_lessons(self) -> None:
        """Good-side players should receive actionable good-side failure reflections."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        from tests.memory.test_reflection_v2 import _v2_entry
        from werewolf_agent.memory.schemas import PlayerProfile

        class FakeReflectionMemory:
            def query_live(self, query):
                return [
                    _v2_entry(
                        entry_id="reflection_good_failure",
                        game_id="old_good_failure",
                        player_id="p01",
                        role="villager",
                        quality_status="approved",
                        quality_score=0.9,
                        prompt_card={
                            **_v2_entry().prompt_card.model_dump(),
                            "theme": "好人失败归票校准",
                            "lesson": "好人失败原因：没有把预言家查杀转成统一投票。",
                            "recommended_action": "改进：先按查验和票型排序。",
                        },
                    )
                ]

            def live_error_pattern(self, player_id, role=""):
                return {
                    "top_mistakes": [("vote_mistake", 1)],
                    "preserved_strength_count": 0,
                    "preserved_strength_labels": [],
                    "total_reflections": 1,
                    "same_role_reflections": 1,
                    "dominant_mistake_ratio": 1.0,
                    "current_role": role,
                }

        class FakeMemory:
            reflections = FakeReflectionMemory()

            def get_profile(self, player_id):
                return PlayerProfile(
                    player_id=player_id,
                    games_played=3,
                    logic=0.6,
                    deception=0.2,
                    credibility=0.7,
                )

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
        assert "werewolf" not in joined
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
        # NEW-S04-A: skill_analysis_hints is no longer populated. The
        # single source of truth is strategy_directive.skill_tactical_advice
        # (the structured render path inside the strategy_directive
        # section). The dual-render duplication is gone.
        assert context.skill_analysis_hints == {}
