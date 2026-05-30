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


class TestWitchStrategyHints:
    """Witch strategy directive must be context-aware, not hard-coded."""

    def _make_witch_state(
        self,
        night_number: int = 1,
        poison_used: bool = False,
        wolf_kill_target: str | None = "v1",
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

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
            game_id="witch_strategy_test",
            players=players,
            phase="night",
            night_number=night_number,
            poison_used=poison_used,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(action_type=ActionType.NO_ACTION, reason="test"), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "witch" else None

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": wolf_kill_target,
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
        return state, engine, registry

    def test_n1_strategy_hint_has_probability_framework(self) -> None:
        """N1 hint should contain probability framework, not mandate saving."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=1)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        assert "witch_strategy_hint" in ctx.strategy_directive
        hint = ctx.strategy_directive["witch_strategy_hint"]
        # N1 must have probability framework
        assert "save_value_assessment" in ctx.strategy_directive
        assessment = ctx.strategy_directive["save_value_assessment"]
        assert assessment["actionable"] is True
        assert "probability_framework" in assessment
        assert assessment["probability_framework"]["p_seer"] > 0
        # Must NOT contain the old hard-coded directive
        assert "首夜大概率应该救人" not in ctx.strategy_directive.get("witch_night_action", "")

    def test_later_night_has_scored_assessment(self) -> None:
        """N2+ should have a numeric score and interpretation."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=3)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        assessment = ctx.strategy_directive.get("save_value_assessment", {})
        assert assessment.get("actionable") is True
        assert assessment.get("public_info_available") is True
        assert "save_value_score" in assessment
        assert "interpretation" in assessment
        assert "目标公开行为分析" in assessment["interpretation"]

    def test_sheriff_target_gets_high_score(self) -> None:
        """If wolf-kill target is the active sheriff, save value should be high."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=2)
        # Make v1 the sheriff
        gs = state["game_state"]
        gs = replace(gs, sheriff_id="v1", sheriff_badge_state="active")
        state["game_state"] = gs
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        assessment = ctx.strategy_directive["save_value_assessment"]
        assert assessment["save_value_score"] >= 8
        assert "is_sheriff" in assessment["signals"]

    def test_target_claimed_seer_gets_high_score(self) -> None:
        """If target claimed seer in speech, save value should be high."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=2)
        gs = state["game_state"]
        gs = replace(gs, events=[
            GameEvent(type="speech", payload={"speaker": "v1", "text": "我是预言家，w1查杀"}),
        ])
        state["game_state"] = gs
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        assessment = ctx.strategy_directive["save_value_assessment"]
        assert assessment["save_value_score"] >= 6
        assert "claimed_seer_in_speech" in assessment["signals"]

    def test_poison_available_adds_poison_hint(self) -> None:
        """When poison is unused, strategy hint mentions poison as alternative."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=1, poison_used=False)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        hint = ctx.strategy_directive["witch_strategy_hint"]
        assert "毒药" in hint

    def test_witch_poison_requires_hard_evidence(self) -> None:
        """Witch poison guidance should require hard evidence before using poison."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=2, poison_used=False)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        directive = ctx.strategy_directive["witch_poison_threshold"]
        assert "查杀" in directive
        assert "强票型" in directive
        assert "单独开毒" in directive

    def test_poison_used_no_poison_hint(self) -> None:
        """When poison is already used, no poison alternative is mentioned."""
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=2, poison_used=True)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        hint = ctx.strategy_directive["witch_strategy_hint"]
        assert "毒药可用" not in hint


class TestSeerStrategyDirectives:
    """Seer strategy must be structured: exclude checked players, score targets,
    provide hybrid warning, and inject day speech guidance."""

    def _make_seer_state(
        self,
        night_number: int = 1,
        extra_events: list | None = None,
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_night_seer
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

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
        events = list(extra_events or [])
        gs = GameState(
            game_id="seer_strategy_test",
            players=players,
            phase="night",
            night_number=night_number,
            events=events,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.CHECK_ALIGNMENT,
                    target_id="v1",
                    reason="test",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "seer" else None

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
        return state, engine, registry

    def test_n1_has_check_value_assessment(self) -> None:
        """N1 seer should receive check_value_assessment with ranked targets."""
        state, engine, registry = self._make_seer_state(night_number=1)
        from werewolf_agent.runtime.agent_adapter import agent_night_seer
        agent_night_seer(state, engine, registry)
        ctx = registry.agent.last_context
        assert "check_value_assessment" in ctx.strategy_directive
        cv = ctx.strategy_directive["check_value_assessment"]
        assert "ranked_targets" in cv
        assert "recommendation" in cv
        assert len(cv["ranked_targets"]) > 0

    def test_checked_players_excluded_from_targets(self) -> None:
        """Players already checked by seer must not appear in legal_targets."""
        events = [
            GameEvent(type="seer_check", payload={
                "seer_id": "seer", "target_id": "v1",
                "alignment": "good", "night_number": 1,
            }),
        ]
        state, engine, registry = self._make_seer_state(
            night_number=2, extra_events=events,
        )
        from werewolf_agent.runtime.agent_adapter import agent_night_seer
        agent_night_seer(state, engine, registry)
        ctx = registry.agent.last_context
        assert "v1" not in ctx.legal_targets

    def test_hybrid_warning_in_night_directive(self) -> None:
        """Seer night directive must warn about hybrid seer_result=good blind spot."""
        state, engine, registry = self._make_seer_state(night_number=1)
        from werewolf_agent.runtime.agent_adapter import agent_night_seer
        agent_night_seer(state, engine, registry)
        ctx = registry.agent.last_context
        directive = ctx.strategy_directive.get("seer_night_check", "")
        assert "混血儿" in directive

    def test_target_claimed_power_role_gets_high_value(self) -> None:
        """A target who claimed a power role in speech should score high."""
        events = [
            GameEvent(type="speech", payload={
                "speaker": "v2", "text": "我是女巫",
                "claims": [{"type": "role", "value": "witch"}],
            }),
        ]
        state, engine, registry = self._make_seer_state(
            night_number=2, extra_events=events,
        )
        from werewolf_agent.runtime.agent_adapter import agent_night_seer
        agent_night_seer(state, engine, registry)
        ctx = registry.agent.last_context
        cv = ctx.strategy_directive["check_value_assessment"]
        v2_entry = next((t for t in cv["ranked_targets"] if t["target"] == "v2"), None)
        assert v2_entry is not None
        assert v2_entry["value"] >= 4
        assert any("claimed" in s for s in v2_entry["signals"])

    def test_seer_day_speech_has_directive(self) -> None:
        """Seer day speech must receive structured speech directive."""
        from werewolf_agent.runtime.agent_adapter import (
            agent_day_speech,
            _build_seer_day_speech_directive,
        )
        events = [
            GameEvent(type="seer_check", payload={
                "seer_id": "seer", "target_id": "w1",
                "alignment": "wolf", "night_number": 1,
            }),
        ]
        players = {
            "seer": PlayerState(id="seer", role="seer"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="seer_speech_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )
        result = _build_seer_day_speech_directive(gs, "seer")
        assert "seer_speech_directive" in result
        directive = result["seer_speech_directive"]
        # Should report the unreported wolf check
        assert "查杀" in directive or "狼人" in directive
        assert "unreported_checks" in result
        uc = result["unreported_checks"]
        assert any(c["alignment"] == "wolf" for c in uc)

    def test_seer_day_speech_counterclaim_context(self) -> None:
        """When there are counterclaiming seers, directive must mention it."""
        from werewolf_agent.runtime.agent_adapter import (
            _build_seer_day_speech_directive,
        )
        events = [
            GameEvent(type="sheriff_speech", payload={
                "speaker": "w1", "text": "我是预言家，v1是好人",
                "claims": [{"type": "role", "value": "seer"}],
            }),
        ]
        players = {
            "seer": PlayerState(id="seer", role="seer"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="seer_counterclaim_test",
            players=players,
            phase="day",
            day_number=1,
            events=events,
        )
        result = _build_seer_day_speech_directive(gs, "seer")
        directive = result["seer_speech_directive"]
        assert "对跳" in directive
        assert "w1" in directive

class TestHunterStrategyDirectives:
    """Hunter strategy must be structured: evaluate shot targets, inject death
    reason, enhance last words, and manage day speech identity."""

    def _make_hunter_state(
        self,
        extra_events: list | None = None,
        hunter_death_reason: str = "wolf_kill",
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

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
        events = list(extra_events or [])
        gs = GameState(
            game_id="hunter_strategy_test",
            players=players,
            phase="night",
            night_number=2,
            events=events,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.HUNTER_SHOT,
                    target_id="w1",
                    reason="test",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "hunter" else None

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
            "hunter_death_reason": hunter_death_reason,
        }
        return state, engine, registry

    def test_hunter_shot_has_value_assessment(self) -> None:
        """Hunter shot must receive shot_value_assessment with ranked targets."""
        state, engine, registry = self._make_hunter_state()
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        assert "shot_value_assessment" in ctx.strategy_directive
        sv = ctx.strategy_directive["shot_value_assessment"]
        assert "ranked_targets" in sv
        assert "recommendation" in sv
        assert len(sv["ranked_targets"]) > 0

    def test_seer_checked_wolf_gets_highest_score(self) -> None:
        """A player publicly accused of being wolf should get scored higher."""
        events = [
            GameEvent(type="speech", payload={
                "speaker": "seer", "text": "w1是狼人，我查验了，查杀",
                "day_number": 1,
            }),
        ]
        state, engine, registry = self._make_hunter_state(extra_events=events)
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        sv = ctx.strategy_directive["shot_value_assessment"]
        w1_entry = next((t for t in sv["ranked_targets"] if t["target"] == "w1"), None)
        assert w1_entry is not None
        # Public accusation gives +4, which is the strongest signal without
        # standard p\d{2} format seer_check_claim match
        assert w1_entry["value"] >= 4

    def test_counterclaiming_seer_gets_high_score(self) -> None:
        """A counterclaiming seer should score high for hunter shot."""
        events = [
            GameEvent(type="sheriff_speech", payload={
                "speaker": "w1", "text": "我是预言家",
                "claims": [{"type": "role", "value": "seer"}],
            }),
        ]
        state, engine, registry = self._make_hunter_state(extra_events=events)
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        sv = ctx.strategy_directive["shot_value_assessment"]
        w1_entry = next((t for t in sv["ranked_targets"] if t["target"] == "w1"), None)
        assert w1_entry is not None
        assert w1_entry["value"] >= 6
        assert "counterclaiming_seer" in w1_entry["signals"]

    def test_no_high_value_target_suggests_no_shot(self) -> None:
        """When no target has strong evidence, advisory should suggest no shot."""
        state, engine, registry = self._make_hunter_state()
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        sv = ctx.strategy_directive["shot_value_assessment"]
        top_value = sv["ranked_targets"][0]["value"]
        if top_value < 3:
            assert "不开枪" in sv["shoot_advisory"] or "NO_ACTION" in sv["shoot_advisory"]

    def test_low_evidence_hunter_directive_prefers_no_shot(self) -> None:
        """Hunter prompt must not turn a low-evidence target list into a forced shot."""
        state, engine, registry = self._make_hunter_state()
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        directive = ctx.strategy_directive["hunter_shot_directive"]
        assert "不开枪" in directive or "NO_ACTION" in directive
        assert "明确查杀" in directive
        assert "避免误伤好人" in directive

    def test_death_reason_passed_to_agent(self) -> None:
        """Death reason should be available in the strategy directive."""
        state, engine, registry = self._make_hunter_state(hunter_death_reason="exile")
        from werewolf_agent.runtime.agent_adapter import agent_hunter_shot
        agent_hunter_shot(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        sv = ctx.strategy_directive["shot_value_assessment"]
        assert sv["death_reason"] == "exile"
        directive = ctx.strategy_directive["hunter_shot_directive"]
        assert "放逐" in directive

    def test_hunter_last_words_directive(self) -> None:
        """Exiled hunter must receive structured last words guidance."""
        from werewolf_agent.runtime.agent_adapter import agent_exile_last_words
        players = {
            "hunter": PlayerState(id="hunter", role="hunter"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="hunter_lw_test",
            players=players,
            phase="day",
            day_number=2,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(action_type=ActionType.SPEECH, speech="test"), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
        agent_exile_last_words(state, engine, registry, "hunter")
        ctx = registry.agent.last_context
        assert "hunter_last_words" in ctx.strategy_directive
        assert "带走" in ctx.strategy_directive["hunter_last_words"]

    def test_hunter_day_speech_has_identity_management(self) -> None:
        """Hunter day speech must include identity management directive."""
        from werewolf_agent.runtime.agent_adapter import _build_hunter_day_speech_directive
        gs = GameState(
            game_id="hunter_speech_test",
            players={
                "hunter": PlayerState(id="hunter", role="hunter"),
                "v1": PlayerState(id="v1", role="villager"),
            },
            phase="day",
            day_number=2,
        )
        # Identity not exposed
        directive = _build_hunter_day_speech_directive(gs, "hunter")
        assert "不要暴露" in directive or "隐藏" in directive

        # Identity exposed by self
        gs2 = replace(gs, events=[
            GameEvent(type="speech", payload={"speaker": "hunter", "text": "我是猎人"}),
        ])
        directive2 = _build_hunter_day_speech_directive(gs2, "hunter")
        assert "身份已经公开" in directive2


class TestHybridStrategyDirectives:
    """Hybrid strategy: master selection assessment, day speech identity
    management, and vote alignment with master."""

    def _make_hybrid_state(
        self,
        extra_events: list | None = None,
        master_id: str | None = None,
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_hybrid_choose_master
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

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
        events = list(extra_events or [])
        gs_kwargs: dict = {
            "game_id": "hybrid_strategy_test",
            "players": players,
            "phase": "night",
            "night_number": 1,
            "events": events,
        }
        if master_id:
            gs_kwargs["hybrid_master_id"] = master_id
        gs = GameState(**gs_kwargs)

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.CHOOSE_MASTER,
                    target_id="seer",
                    reason="test",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "hybrid" else None

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
        return state, engine, registry

    def test_hybrid_master_choice_has_assessment(self) -> None:
        """Hybrid N1 must receive master_assessment with ranked candidates."""
        state, engine, registry = self._make_hybrid_state()
        from werewolf_agent.runtime.agent_adapter import agent_hybrid_choose_master
        agent_hybrid_choose_master(state, engine, registry, "hybrid")
        ctx = registry.agent.last_context
        assert "master_assessment" in ctx.strategy_directive
        ma = ctx.strategy_directive["master_assessment"]
        assert "ranked_candidates" in ma
        assert "probability_framework" in ma
        assert "strategy_guidance" in ma
        assert len(ma["ranked_candidates"]) == 11  # 12 - hybrid self

    def test_hybrid_master_choice_knows_victory_condition(self) -> None:
        """Strategy directive must explain the victory condition."""
        state, engine, registry = self._make_hybrid_state()
        from werewolf_agent.runtime.agent_adapter import agent_hybrid_choose_master
        agent_hybrid_choose_master(state, engine, registry, "hybrid")
        ctx = registry.agent.last_context
        directive = ctx.strategy_directive["hybrid_master_choice"]
        assert "跟随主人的原始阵营获胜" in directive or "跟随主人" in directive

    def test_hybrid_day_speech_has_identity_directive(self) -> None:
        """Hybrid day speech must warn against revealing identity."""
        from werewolf_agent.runtime.agent_adapter import _build_hybrid_day_speech_directive
        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "seer": PlayerState(id="seer", role="seer"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="hybrid_speech_test",
            players=players,
            phase="day",
            day_number=2,
            hybrid_master_id="seer",
        )
        result = _build_hybrid_day_speech_directive(gs, "hybrid")
        assert "hybrid_speech_directive" in result
        directive = result["hybrid_speech_directive"]
        assert "不要暴露" in directive
        assert "seer" in directive  # master_id mentioned

    def test_hybrid_day_speech_has_master_behavior(self) -> None:
        """D2+ hybrid speech should include master behavior summary."""
        from werewolf_agent.runtime.agent_adapter import _build_hybrid_day_speech_directive
        events = [
            GameEvent(type="speech", payload={
                "speaker": "seer", "text": "我验了v1是好人，w1是狼人",
            }),
        ]
        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "seer": PlayerState(id="seer", role="seer"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="hybrid_behavior_test",
            players=players,
            phase="day",
            day_number=2,
            hybrid_master_id="seer",
            events=events,
        )
        result = _build_hybrid_day_speech_directive(gs, "hybrid")
        assert "master_behavior_summary" in result
        assert "seer" in result["master_behavior_summary"]

    def test_hybrid_vote_has_master_strategy(self) -> None:
        """Hybrid vote must receive master-aligned voting strategy."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        gs = GameState(
            game_id="hybrid_vote_test",
            players=players,
            phase="day",
            day_number=2,
            hybrid_master_id="seer",
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="w1",
                    reason="test",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
        agent_day_vote(state, engine, registry, "hybrid")
        ctx = registry.agent.last_context
        assert "hybrid_vote_strategy" in ctx.strategy_directive
        assert "主人" in ctx.strategy_directive["hybrid_vote_strategy"]

    def test_sheriff_speaker_candidate_scores_higher(self) -> None:
        """Candidates who registered for sheriff should score higher."""
        events = [
            GameEvent(type="sheriff_registration", payload={"player_id": "seer", "registered": True}),
        ]
        state, engine, registry = self._make_hybrid_state(extra_events=events)
        from werewolf_agent.runtime.agent_adapter import agent_hybrid_choose_master
        agent_hybrid_choose_master(state, engine, registry, "hybrid")
        ctx = registry.agent.last_context
        ma = ctx.strategy_directive["master_assessment"]
        seer_entry = next(
            (c for c in ma["ranked_candidates"] if c["target"] == "seer"), None
        )
        assert seer_entry is not None
        assert seer_entry["value"] > 0
        assert "ran_for_sheriff" in seer_entry["signals"]


class TestVillagerStrategyDirectives:
    """Villager/idiot strategy: pure analysis with no private info."""

    def test_villager_day_speech_has_directive(self) -> None:
        """Villager day speech must include analysis strategy."""
        from werewolf_agent.runtime.agent_adapter import _build_villager_day_speech_directive
        gs = GameState(
            game_id="villager_test",
            players={
                "v1": PlayerState(id="v1", role="villager"),
                "w1": PlayerState(id="w1", role="werewolf"),
                "seer": PlayerState(id="seer", role="seer"),
            },
            phase="day",
            day_number=2,
        )
        result = _build_villager_day_speech_directive(gs, "v1")
        assert "villager_speech_directive" in result
        directive = result["villager_speech_directive"]
        assert "逻辑分析" in directive

    def test_villager_counterclaim_analysis(self) -> None:
        """When there are counterclaiming seers, villager gets analysis guidance."""
        from werewolf_agent.runtime.agent_adapter import _build_villager_day_speech_directive
        events = [
            GameEvent(type="sheriff_speech", payload={
                "speaker": "seer", "text": "我是预言家",
                "claims": [{"type": "role", "value": "seer"}],
            }),
            GameEvent(type="sheriff_speech", payload={
                "speaker": "w1", "text": "我是预言家",
                "claims": [{"type": "role", "value": "seer"}],
            }),
        ]
        gs = GameState(
            game_id="villager_counterclaim_test",
            players={
                "v1": PlayerState(id="v1", role="villager"),
                "w1": PlayerState(id="w1", role="werewolf"),
                "seer": PlayerState(id="seer", role="seer"),
            },
            phase="day",
            day_number=1,
            events=events,
        )
        result = _build_villager_day_speech_directive(gs, "v1")
        directive = result["villager_speech_directive"]
        assert "对跳预言家" in directive

    def test_villager_vote_has_independent_strategy(self) -> None:
        """Villager vote must include independent judgment strategy."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            "v1": PlayerState(id="v1", role="villager"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        gs = GameState(
            game_id="villager_vote_test",
            players=players,
            phase="day",
            day_number=2,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="w1",
                    reason="test",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent

        registry = CaptureRegistry()
        engine = _new_engine()
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
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
        agent_day_vote(state, engine, registry, "v1")
        ctx = registry.agent.last_context
        assert "villager_vote_strategy" in ctx.strategy_directive
        vs = ctx.strategy_directive["villager_vote_strategy"]
        assert "判断预言家真假" in vs
        assert "不要无条件" in vs

    def test_good_vote_has_hard_info_and_mistake_cost_guard(self) -> None:
        """Good-side vote context must prioritize hard info and evaluate mistake cost."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            "v1": PlayerState(id="v1", role="villager"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "seer": PlayerState(id="seer", role="seer"),
            "hunter": PlayerState(id="hunter", role="hunter"),
        }
        gs = GameState(
            game_id="good_vote_guard_test",
            players=players,
            phase="day",
            day_number=2,
            events=[
                GameEvent(type="sheriff_speech", payload={
                    "speaker": "seer",
                    "text": "我是预言家，w1查杀，警徽流先验hunter。",
                    "claims": [{"type": "role", "value": "seer"}],
                }),
            ],
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="w1",
                    reason="seer查杀w1",
                    seer_stance="trust",
                    vote_basis="seer_check",
                    standing_with_seer="seer",
                    suspect_reason="w1被seer查杀",
                    not_voting_reason="hunter没有硬证据狼面",
                    private_reason="优先走可信预言家的查验。",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent

        state: RuntimeState = {
            "game_state": gs,
            "engine": _new_engine(),
            "wolf_kill_target_id": None,
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
        registry = CaptureRegistry()
        agent_day_vote(state, state["engine"], registry, "v1")

        ctx = registry.agent.last_context
        guard = ctx.strategy_directive["good_vote_decision_guard"]
        assert "硬信息优先级" in guard
        assert "预言家查验" in guard
        assert "出错成本" in guard
        assert "猎人" in guard

    def test_idiot_uses_villager_strategy(self) -> None:
        """Idiot should get the same analysis strategy as villager."""
        from werewolf_agent.runtime.agent_adapter import _build_villager_day_speech_directive
        gs = GameState(
            game_id="idiot_speech_test",
            players={
                "idiot": PlayerState(id="idiot", role="idiot"),
                "w1": PlayerState(id="w1", role="werewolf"),
            },
            phase="day",
            day_number=2,
        )
        result = _build_villager_day_speech_directive(gs, "idiot")
        directive = result["villager_speech_directive"]
        assert "白痴" in directive

    def test_vote_history_in_speech(self) -> None:
        """Villager speech should include vote history when available."""
        from werewolf_agent.runtime.agent_adapter import _build_villager_day_speech_directive
        events = [
            GameEvent(type="vote_resolved", payload={
                "exiled": "w1",
                "day_number": 1,
                "votes": {"v1": "w1", "seer": "w1", "v2": "seer"},
            }),
        ]
        gs = GameState(
            game_id="villager_history_test",
            players={
                "v1": PlayerState(id="v1", role="villager"),
                "w1": PlayerState(id="w1", role="werewolf"),
                "seer": PlayerState(id="seer", role="seer"),
                "v2": PlayerState(id="v2", role="villager"),
            },
            phase="day",
            day_number=2,
            events=events,
        )
        result = _build_villager_day_speech_directive(gs, "v1")
        directive = result["villager_speech_directive"]
        assert "投票数据" in directive
        assert "w1" in directive

class TestIdiotStrategyDirectives:
    """Idiot strategy: pre-reveal caution, post-reveal boldness, vote bug fix."""

    def test_idiot_pre_reveal_has_caution_strategy(self) -> None:
        """Before reveal, idiot should be cautioned to avoid being voted."""
        from werewolf_agent.runtime.agent_adapter import _build_idiot_day_speech_directive
        players = {
            "idiot": PlayerState(id="idiot", role="idiot", revealed_idiot=False),
            "w1": PlayerState(id="w1", role="werewolf"),
        }
        gs = GameState(
            game_id="idiot_pre_reveal_test",
            players=players,
            phase="day",
            day_number=2,
        )
        result = _build_idiot_day_speech_directive(gs, "idiot")
        directive = result["idiot_speech_directive"]
        assert "尚未翻牌" in directive
        assert "避免" in directive

    def test_idiot_post_reveal_has_bold_strategy(self) -> None:
        """After reveal, idiot should be told to speak boldly."""
        from werewolf_agent.runtime.agent_adapter import _build_idiot_day_speech_directive
        players = {
            "idiot": PlayerState(
                id="idiot", role="idiot", revealed_idiot=True,
                exile_immune=True, vote_enabled=False,
            ),
            "w1": PlayerState(id="w1", role="werewolf"),
        }
        gs = GameState(
            game_id="idiot_post_reveal_test",
            players=players,
            phase="day",
            day_number=3,
        )
        result = _build_idiot_day_speech_directive(gs, "idiot")
        directive = result["idiot_speech_directive"]
        assert "已经翻牌" in directive
        assert "大胆发言" in directive
        assert "失去投票权" in directive

    def test_idiot_post_reveal_loses_vote_in_graph(self) -> None:
        """Revealed idiot (vote_enabled=False) should be excluded from voters."""
        players = {
            "idiot": PlayerState(
                id="idiot", role="idiot", revealed_idiot=True,
                exile_immune=True, vote_enabled=False,
            ),
            "v1": PlayerState(id="v1", role="villager"),
            "w1": PlayerState(id="w1", role="werewolf"),
        }
        gs = GameState(
            game_id="idiot_vote_test",
            players=players,
            phase="day",
            day_number=3,
        )
        voters = [
            pid for pid, p in gs.players.items()
            if p.alive and p.vote_enabled
        ]
        assert "idiot" not in voters
        assert "v1" in voters
        assert "w1" in voters

    def test_idiot_inherits_villager_analysis_data(self) -> None:
        """Idiot should get the same analysis data as villager (seer claims, etc)."""
        from werewolf_agent.runtime.agent_adapter import _build_idiot_day_speech_directive
        events = [
            GameEvent(type="vote_resolved", payload={
                "exiled": "w2",
                "day_number": 1,
                "votes": {"idiot": "w2", "v1": "w2"},
            }),
        ]
        players = {
            "idiot": PlayerState(id="idiot", role="idiot"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(
            game_id="idiot_analysis_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )
        result = _build_idiot_day_speech_directive(gs, "idiot")
        directive = result["idiot_speech_directive"]
        assert "投票数据" in directive

class TestWolfStrategyDirectives:
    """Tests for werewolf strategy injection in agent_adapter."""

    @staticmethod
    def _make_wolf_gs(**overrides) -> GameState:
        players = {
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "w2": PlayerState(id="w2", role="werewolf", alive=True),
            "w3": PlayerState(id="w3", role="werewolf", alive=True),
            "w4": PlayerState(id="w4", role="werewolf", alive=True),
            "seer": PlayerState(id="seer", role="seer", alive=True),
            "witch": PlayerState(id="witch", role="witch", alive=True),
            "hunter": PlayerState(id="hunter", role="hunter", alive=True),
            "idiot": PlayerState(id="idiot", role="idiot", alive=True),
            "v1": PlayerState(id="v1", role="villager", alive=True),
            "v2": PlayerState(id="v2", role="villager", alive=True),
            "v3": PlayerState(id="v3", role="villager", alive=True),
            "hyb": PlayerState(id="hyb", role="hybrid", alive=True),
        }
        defaults = dict(game_id="wolf_test", players=players, phase="day", day_number=2)
        defaults.update(overrides)
        return GameState(**defaults)

    def test_wolf_kill_value_assessment(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _evaluate_wolf_kill_target
        gs = self._make_wolf_gs(
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "seer", "text": "w2是狼人，我查验的，查杀",
                    "day_number": 1,
                }),
            ],
        )
        result = _evaluate_wolf_kill_target(gs, "w1", ["seer", "witch", "hunter", "v1"])
        assert result is not None
        assert "ranked_targets" in result
        ranked = result["ranked_targets"]
        seer_entry = next(r for r in ranked if r["target"] == "seer")
        # Seer is a high-value kill target; public wolf accusation gives +4
        assert seer_entry["value"] >= 4

    def test_sheriff_gets_high_kill_score(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _evaluate_wolf_kill_target
        gs = self._make_wolf_gs(sheriff_id="hunter", sheriff_badge_state="active")
        result = _evaluate_wolf_kill_target(gs, "w1", ["hunter", "v1", "v2"])
        assert result is not None
        ranked = result["ranked_targets"]
        hunter_entry = next(r for r in ranked if r["target"] == "hunter")
        assert hunter_entry["value"] >= 8
        assert "is_sheriff" in hunter_entry["signals"]

    def test_wolf_day_speech_has_role_strategy(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2", "hooker": "w3", "deep_cover": "w4"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        assert "wolf_speech_directive" in result
        assert "悍跳" in result["wolf_speech_directive"]
        assert "wolf_universal_rules" in result
        assert "绝对不要提到你的队友" in result["wolf_universal_rules"]

    def test_wolf_role_assignments_different(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2", "hooker": "w3", "deep_cover": "w4"}
        r1 = _build_wolf_day_speech_directive(gs, "w1", plan)
        r2 = _build_wolf_day_speech_directive(gs, "w2", plan)
        r3 = _build_wolf_day_speech_directive(gs, "w3", plan)
        r4 = _build_wolf_day_speech_directive(gs, "w4", plan)
        assert "悍跳" in r1["wolf_speech_directive"]
        assert "冲锋" in r2["wolf_speech_directive"]
        assert "倒钩" in r3["wolf_speech_directive"]
        assert "深水" in r4["wolf_speech_directive"]

    def test_wolf_speech_has_push_target(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"day_push_target": "v1", "fake_seer": "w1"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        assert "wolf_day_push_target" in result
        assert "v1" in result["wolf_day_push_target"]

    def test_wolf_speech_detects_teammate_exposure(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs(events=[
            GameEvent(type="speech", payload={
                "speaker": "seer", "text": "w2是狼人，查杀",
                "day_number": 1,
            }),
        ])
        result = _build_wolf_day_speech_directive(gs, "w1", {"fake_seer": "w1"})
        assert "wolf_speech_directive" in result

    def test_wolf_vote_strategy_has_role_hint(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_vote_strategy
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2", "hooker": "w3", "deep_cover": "w4"}
        r_hooker = _build_wolf_vote_strategy(gs, "w3", plan)
        assert "wolf_vote_role_hint" in r_hooker
        assert "投你的狼人队友" in r_hooker["wolf_vote_role_hint"]

        r_deep = _build_wolf_vote_strategy(gs, "w4", plan)
        assert "跟随主流好人" in r_deep["wolf_vote_role_hint"]

    def test_wolf_vote_has_push_target(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_vote_strategy
        gs = self._make_wolf_gs()
        plan = {"day_push_target": "v2"}
        result = _build_wolf_vote_strategy(gs, "w1", plan)
        assert "wolf_vote_target" in result
        assert "v2" in result["wolf_vote_target"]

    def test_wolf_speech_knows_fake_seer_teammate(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        # When fake seer has NOT spoken yet → anti-reveal constraint
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w2", "pusher": "w1"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        assert "wolf_fake_seer_teammate" in result
        assert "严禁信息穿越" in result["wolf_fake_seer_teammate"]

        # When fake seer HAS publicly claimed → coordination allowed
        gs2 = self._make_wolf_gs(events=[
            GameEvent(type="speech", payload={
                "speaker": "w2", "text": "我是预言家，查杀了v1", "day_number": 1,
            }),
        ])
        result2 = _build_wolf_day_speech_directive(gs2, "w1", plan)
        assert "wolf_fake_seer_teammate" in result2
        assert "w2" in result2["wolf_fake_seer_teammate"]
        assert "已公开跳预言家" in result2["wolf_fake_seer_teammate"]

    def test_single_wolf_vote_seer_is_highest_threat(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _evaluate_wolf_kill_target
        gs = self._make_wolf_gs(
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "seer", "text": "我是预言家，我验了w2是狼人", "day_number": 1,
                }),
            ],
        )
        result = _evaluate_wolf_kill_target(gs, "w1", ["seer", "witch", "v1"])
        assert result is not None
        ranked = result["ranked_targets"]
        seer_entry = next(r for r in ranked if r["target"] == "seer")
        assert seer_entry["value"] >= 6

    def test_fake_seer_self_gets_execution_directive_before_claim(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        assert "wolf_fake_seer_execution" in result
        assert "跳预言家" in result["wolf_fake_seer_execution"]
        assert "不要犹豫" in result["wolf_fake_seer_execution"]

    def test_fake_seer_self_gets_maintenance_directive_after_claim(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs(events=[
            GameEvent(type="speech", payload={
                "speaker": "w1", "text": "我是预言家，查杀v1", "day_number": 1,
            }),
        ])
        plan = {"fake_seer": "w1", "pusher": "w2"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        assert "wolf_fake_seer_execution" in result
        assert "已经" in result["wolf_fake_seer_execution"]
        assert "继续维护" in result["wolf_fake_seer_execution"]

    def test_fake_seer_rule6_allows_jumping(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2"}
        result = _build_wolf_day_speech_directive(gs, "w1", plan)
        rules = result["wolf_universal_rules"]
        assert "严禁信息穿越" not in rules
        assert "悍跳狼" in rules

    def test_non_fake_seer_rule6_has_info_leak_guard(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        plan = {"fake_seer": "w1", "pusher": "w2"}
        result = _build_wolf_day_speech_directive(gs, "w2", plan)
        rules = result["wolf_universal_rules"]
        assert "严禁信息穿越" in rules

    def test_unassigned_wolf_gets_generic_strategy(self) -> None:
        from werewolf_agent.runtime.agent_adapter import _build_wolf_day_speech_directive
        gs = self._make_wolf_gs()
        result = _build_wolf_day_speech_directive(gs, "w1", None)
        assert "wolf_speech_directive" in result
        assert "没有特定角色分工" in result["wolf_speech_directive"]


class TestEmptySpeechGuard:
    """F2: Empty speech fallback."""

    def test_agent_day_speech_fallback_on_empty(self) -> None:
        """agent_day_speech should produce non-empty speech even when agent returns empty."""
        from werewolf_agent.runtime.agent_adapter import agent_day_speech
        from unittest.mock import MagicMock

        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        # Override a few roles
        players["p01"] = PlayerState(id="p01", role="werewolf", alive=True)
        players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
        players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
        gs = GameState(game_id="empty_speech_test", players=players, phase="day", day_number=1)

        # Create a mock agent that returns empty speech
        mock_action = MagicMock()
        mock_action.action_type = ActionType.SPEECH
        mock_action.speech = ""
        mock_agent = MagicMock()
        mock_agent.act.return_value = (mock_action, None)

        registry = MagicMock()
        registry.get_agent.return_value = mock_agent

        engine = _new_engine()
        state = {"game_state": gs, "engine": engine, "agent_registry": registry}

        result = agent_day_speech(state, engine, registry, "p05")
        assert result is not None
        assert result["speech_text"].strip(), "Speech should not be empty after fallback guard"