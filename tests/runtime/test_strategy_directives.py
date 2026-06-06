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
        """Witch poison guidance should encourage using poison with evidence.

        P1-D5 regression: the unified ``witch_poison_strategy`` directive
        (early-game ``no_pressure_save_for_late`` branch) still surfaces
        the hard-evidence guidance the pre-fix ``witch_poison_threshold``
        directive used to carry.
        """
        from werewolf_agent.runtime.agent_adapter import agent_night_witch
        state, engine, registry = self._make_witch_state(night_number=2, poison_used=False)
        agent_night_witch(state, engine, registry)
        ctx = registry.agent.last_context
        wp = ctx.strategy_directive["witch_poison_strategy"]
        # The early-game branch covers hard-evidence guidance.
        assert wp["branch"] in {
            "no_pressure_save_for_late",
            "evidence_required_threshold",
        }
        directive = wp["text"]
        # At least one of the 12-player / 9-player branches carries the
        # hard-evidence text; the test setup uses night_number=2 with
        # all 12 alive, so the no_pressure branch is expected.
        assert "查杀" in directive or "毒药" in directive

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
        result = _build_hunter_day_speech_directive(gs, "hunter")
        assert isinstance(result, dict)
        directive = result["hunter_speech_directive"]
        assert "不要暴露" in directive or "隐藏" in directive

        # Identity exposed by self
        gs2 = replace(gs, events=[
            GameEvent(type="speech", payload={"speaker": "hunter", "text": "我是猎人"}),
        ])
        result2 = _build_hunter_day_speech_directive(gs2, "hunter")
        assert isinstance(result2, dict)
        directive2 = result2["hunter_speech_directive"]
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

    def test_hybrid_master_wolf_receives_hidden_wolf_directive(self) -> None:
        """When the hybrid's master is a wolf, the day-speech directive
        must include a faction-aware "hidden ally" block that nudges
        the hybrid to subtly support the wolf team without breaking
        cover.  Pre-fix, the directive was identical to the good-master
        case, which is why hybrid p04 in g_3528592081 (master=p01
        wolf) voted like a villager.

        P0-I2 regression test.
        """
        from werewolf_agent.runtime.agent_adapter import _build_hybrid_day_speech_directive
        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        gs = GameState(
            game_id="hybrid_wolf_master_test",
            players=players,
            phase="day",
            day_number=2,
            hybrid_master_id="w1",
            hybrid_master_faction="werewolf",
        )
        result = _build_hybrid_day_speech_directive(gs, "hybrid")
        # The directive should expose a faction-specific block so the
        # hybrid can adjust its day-speech behavior.
        assert "hybrid_wolf_master_directive" in result, (
            "Hybrid with wolf master must receive hidden-ally directive; "
            f"got keys: {sorted(result.keys())}"
        )
        wolf_directive = result["hybrid_wolf_master_directive"]
        # The text must mention the wolf team or hidden-ally framing.
        assert ("狼" in wolf_directive) or ("wolf" in wolf_directive.lower()), (
            f"wolf-master directive should reference wolf/狼; got: {wolf_directive!r}"
        )
        # And it must warn against identity-revealing behavior.
        assert "暴露" in wolf_directive or "隐藏" in wolf_directive, (
            f"wolf-master directive must include cover discipline; got: {wolf_directive!r}"
        )

    def test_hybrid_master_good_receives_good_side_focus(self) -> None:
        """When the hybrid's master is on the good side, the day-speech
        directive must include a faction-aware block that explicitly
        nudges the hybrid to help the good team.  Pre-fix, the
        directive was neutral and gave no faction guidance.

        P0-I2 regression test (companion to wolf-master test).
        """
        from werewolf_agent.runtime.agent_adapter import _build_hybrid_day_speech_directive
        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "w1": PlayerState(id="w1", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        gs = GameState(
            game_id="hybrid_good_master_test",
            players=players,
            phase="day",
            day_number=2,
            hybrid_master_id="seer",
            hybrid_master_faction="good",
        )
        result = _build_hybrid_day_speech_directive(gs, "hybrid")
        assert "hybrid_good_master_directive" in result, (
            "Hybrid with good master must receive good-side focus; "
            f"got keys: {sorted(result.keys())}"
        )
        good_directive = result["hybrid_good_master_directive"]
        # The text must mention good side framing.
        assert "好人" in good_directive or "good" in good_directive.lower(), (
            f"good-master directive should reference 好人/good; got: {good_directive!r}"
        )

    def test_hybrid_master_unknown_faction_falls_through_neutral(self) -> None:
        """If the hybrid has not yet chosen a master, neither the
        wolf-master nor the good-master block should appear (this
        preserves the pre-fix behavior for the
        ``master-not-chosen-yet`` window).  P0-I2 must not regress
        the existing test that exercises the master-not-chosen state.
        """
        from werewolf_agent.runtime.agent_adapter import _build_hybrid_day_speech_directive
        players = {
            "hybrid": PlayerState(id="hybrid", role="hybrid"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        gs = GameState(
            game_id="hybrid_no_master_test",
            players=players,
            phase="night",
            night_number=1,
        )
        result = _build_hybrid_day_speech_directive(gs, "hybrid")
        # Without a chosen master, neither faction block should be set.
        assert "hybrid_wolf_master_directive" not in result
        assert "hybrid_good_master_directive" not in result
        # The neutral identity-disguise directive must still be there.
        assert "hybrid_speech_directive" in result

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
                    suspect_reason="w1是测试目标",
                    not_voting_reason="其他人没明显证据",
                    private_reason="测试用例",
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
                    suspect_reason="w1是测试目标",
                    not_voting_reason="其他人没明显证据",
                    private_reason="测试用例",
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


class TestWolfSeerPriorityInjection:
    """Task 3 (Issue 4): Wolf kill prompt must name publicly-claimed Seer explicitly."""

    @staticmethod
    def _make_gs_with_claimant(speaker: str = "p03", role: str = "seer") -> GameState:
        players = {
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id=speaker, role=role, alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
            "p06": PlayerState(id="p06", role="villager", alive=True),
        }
        return GameState(
            game_id="g_test",
            players=players,
            phase="day",
            day_number=2,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": speaker,
                    "text": "我是预言家，昨晚查了p05是好人",
                }),
            ],
        )

    def test_claimed_seer_appears_in_wolf_kill_directive(self) -> None:
        """When a player has publicly claimed Seer, the wolf kill directive
        must name them explicitly so all wolves converge on the same target."""
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer
        from werewolf_agent.runtime.agent_adapter import _build_wolf_kill_directive

        gs = self._make_gs_with_claimant()
        # Sanity check: setup actually has a public Seer claim
        assert has_publicly_claimed_seer(gs, "p03") is True

        directive = _build_wolf_kill_directive(gs, wolf_id="p01", plan={})
        assert "p03" in directive, (
            f"wolf kill directive should explicitly name claimed Seer p03, got: {directive!r}"
        )
        # Should mark it as a high-priority target
        assert "高优先级击杀目标" in directive or "优先击杀" in directive

    def test_no_claimed_seer_uses_scored_ranking(self) -> None:
        """When no Seer claim exists, the directive falls back to evaluate_wolf_kill_target's
        top-3 ranking."""
        from werewolf_agent.runtime.agent_adapter import _build_wolf_kill_directive
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer

        # No claim events — players are villagers, no one speaks
        players = {
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
            "p06": PlayerState(id="p06", role="villager", alive=True),
        }
        gs = GameState(
            game_id="g_test_no_claim",
            players=players,
            phase="night",
            night_number=1,
            events=[],
        )
        assert not has_publicly_claimed_seer(gs, "p05")

        directive = _build_wolf_kill_directive(gs, wolf_id="p01", plan={})
        # Should produce at least one candidate from the scoring fallback
        assert "击杀候选" in directive

    def test_wolf_kill_prompt_includes_claimed_seer_via_strategy_directive(self) -> None:
        """_single_wolf_vote must inject wolf_high_priority_target into the
        strategy directive when a Seer has publicly claimed."""
        from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
        from werewolf_agent.agents.schemas import AgentContext, PlayerAction, RetryInfo, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo  # noqa: F811

        gs = self._make_gs_with_claimant()

        class CaptureAgent:
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(
                        action_type=ActionType.WOLF_KILL,
                        target_id="p03",
                        reason="test",
                    ),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()

            def get_agent(self, player_id):
                return self.agent if player_id == "p01" else None

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
        _single_wolf_vote(state, engine, registry, "p01")
        ctx = registry.agent.last_context
        assert ctx is not None
        assert "wolf_high_priority_target" in ctx.strategy_directive
        assert "p03" in ctx.strategy_directive["wolf_high_priority_target"]

    def test_wolf_discussion_prompt_includes_claimed_seer_via_strategy_directive(self) -> None:
        """agent_wolf_discussion must also receive wolf_high_priority_target when
        a Seer has publicly claimed (so private discussion can converge)."""
        from werewolf_agent.runtime.agent_adapter import agent_wolf_discussion
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        gs = self._make_gs_with_claimant()

        class CaptureAgent:
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(
                        action_type=ActionType.SPEECH,
                        speech="建议今晚击杀p03",
                        reason="test",
                    ),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()

            def get_agent(self, player_id):
                return self.agent if player_id == "p01" else None

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
        agent_wolf_discussion(state, engine, registry, "p01")
        ctx = registry.agent.last_context
        assert ctx is not None
        assert "wolf_high_priority_target" in ctx.strategy_directive
        assert "p03" in ctx.strategy_directive["wolf_high_priority_target"]


class TestSheriffDirectiveFallback:
    """P1-D4 / P1-D6: sheriff vote_push fallback + no-sheriff-after-tear directive.

    When the sheriff is silenced (e.g., muted by poison/self-destruct) or
    when the badge is torn (no sheriff for the rest of the game), the
    directive must adapt so players don't keep acting on a stale
    "明确归票" instruction.
    """

    @staticmethod
    def _make_sheriff_gs(
        *,
        sheriff_id: str | None = "p03",
        sheriff_badge_state: str = "active",
        silenced: bool = False,
        alive: bool = True,
    ) -> GameState:
        players = {
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="villager", alive=alive),
            "p04": PlayerState(id="p04", role="seer", alive=True),
            "p05": PlayerState(id="p05", role="witch", alive=True),
            "p06": PlayerState(id="p06", role="hunter", alive=True),
        }
        events: list[GameEvent] = []
        if silenced and sheriff_id:
            events.append(
                GameEvent(
                    type="sheriff_silenced",
                    payload={"sheriff_id": sheriff_id, "reason": "witch_poison"},
                ),
            )
        return GameState(
            game_id="sheriff_fallback_test",
            players=players,
            phase="day",
            day_number=2,
            sheriff_id=sheriff_id,
            sheriff_badge_state=sheriff_badge_state,
            events=events,
        )

    def _invoke_day_speech(self, gs: GameState, speaker_id: str):
        """Run agent_day_speech and return the captured strategy_directive."""
        from werewolf_agent.runtime.agent_adapter import agent_day_speech

        class CaptureAgent:
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(
                        action_type=ActionType.SPEECH, speech="test", reason="test",
                    ),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()

            def get_agent(self, player_id):
                return self.agent if player_id == speaker_id else None

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
        agent_day_speech(state, engine, registry, speaker_id)
        return registry.agent.last_context.strategy_directive

    def test_sheriff_silent_fallback(self) -> None:
        """P1-D4: when the active sheriff is silenced (e.g., by witch poison
        or self-destruct), the directive must swap `sheriff_vote_push` for
        `sheriff_silent` so the model doesn't tell a muted player to
        "明确归票".

        Pre-fix: ``sheriff_vote_push`` was rendered unconditionally for
        the active sheriff, contradicting the silence condition.

        Phase-1 P1-3 follow-up: the sheriff_silent directive must
        EXPLICITLY state that the sheriff still must submit a vote
        action (silence is on speech only, not on the vote) and
        reference the existing target_id field, NOT a nonexistent
        vote_silent field.  Pre-P1-3 the text said "通过 [vote_silent]
        字段指定" which is not a real field.
        """
        gs = self._make_sheriff_gs(silenced=True)
        directive = self._invoke_day_speech(gs, "p03")
        assert "sheriff_silent" in directive, (
            "silenced sheriff must receive `sheriff_silent` directive; "
            f"got keys: {sorted(directive.keys())}"
        )
        # The old vote_push directive must NOT appear when silenced.
        assert "sheriff_vote_push" not in directive, (
            "silenced sheriff must not receive `sheriff_vote_push`; "
            f"got keys: {sorted(directive.keys())}"
        )
        text = directive["sheriff_silent"]
        # Must tell the LLM the sheriff CANNOT speak
        assert "无法发言" in text, (
            f"sheriff_silent must declare speech prohibition; got: {text!r}"
        )
        # Must explicitly tell the LLM the sheriff STILL submits vote
        # action (silence is speech-only, not vote).  P1-3 follow-up.
        assert "仍需提交 vote action" in text or "仍需提交" in text, (
            f"P1-3: sheriff_silent must clarify that vote is still required; "
            f"got: {text!r}"
        )
        # Must reference the existing target_id field (NOT a
        # nonexistent vote_silent field)
        assert "target_id" in text, (
            f"P1-3: sheriff_silent must reference the existing target_id "
            f"field; got: {text!r}"
        )
        # Must NOT reference the nonexistent [vote_silent] placeholder
        assert "[vote_silent]" not in text, (
            f"P1-3: sheriff_silent must NOT reference the nonexistent "
            f"[vote_silent] placeholder; got: {text!r}"
        )

    def test_active_sheriff_without_silence_still_gets_vote_push(self) -> None:
        """Sanity: when the sheriff is NOT silenced, the original
        `sheriff_vote_push` directive must still be present (regression
        guard for P1-D4)."""
        gs = self._make_sheriff_gs(silenced=False)
        directive = self._invoke_day_speech(gs, "p03")
        assert "sheriff_vote_push" in directive
        assert "sheriff_silent" not in directive

    def test_no_sheriff_after_tear(self) -> None:
        """P1-D6: when the badge has been torn (no sheriff for the rest
        of the game), every player must receive a `sheriff_election_state`
        directive so PK / vote participants know there is no 归票人 and
        speech order is random.

        Pre-fix: no directive mentioned the torn-badge state, so players
        acted as if a sheriff still existed.
        """
        gs = self._make_sheriff_gs(sheriff_id=None, sheriff_badge_state="torn")
        # Speaker is a normal villager, not the (now absent) sheriff.
        directive = self._invoke_day_speech(gs, "p04")
        assert "sheriff_election_state" in directive, (
            "after badge tear, every player must receive `sheriff_election_state`; "
            f"got keys: {sorted(directive.keys())}"
        )
        text = directive["sheriff_election_state"]
        # The directive should explicitly state there's no sheriff and
        # that speech order is random.
        assert "无警长" in text
        assert "随机" in text
        # And the no-sheriff state must NOT also carry the vote_push
        # directive (otherwise the model is doubly confused).
        assert "sheriff_vote_push" not in directive
        assert "sheriff_silent" not in directive

    def test_no_sheriff_after_tear_renders_for_wolf_too(self) -> None:
        """P1-D6 regression: the torn-badge directive must reach non-good
        players too (wolves should also know there's no sheriff to push
        votes through, otherwise they may still try to coordinate as if
        a 归票 channel existed)."""
        gs = self._make_sheriff_gs(sheriff_id=None, sheriff_badge_state="torn")
        directive = self._invoke_day_speech(gs, "p01")
        assert "sheriff_election_state" in directive

    def test_active_sheriff_state_does_not_emit_election_state(self) -> None:
        """Sanity: when the sheriff is active, the `sheriff_election_state`
        directive must NOT appear (it would be contradictory)."""
        gs = self._make_sheriff_gs(silenced=False)
        directive = self._invoke_day_speech(gs, "p03")
        assert "sheriff_election_state" not in directive


class TestWitchPoisonUnifiedDirective:
    """P1-D5: witch poison guidance must be unified into a single
    `witch_poison_strategy` directive with 3 context-aware branches.

    Pre-fix: `witch_poison_threshold` and `poison_urgency` could both be
    rendered (and contradict each other); there was no `no_pressure`
    branch for early game.  The unified directive must:
      1) `no_pressure_save_for_late` — early game, no pressure
      2) `urgency_under_X_alive` — low alive count, urgent
      3) `evidence_required_threshold` — mid game, need hard evidence
    """

    def _make_witch_gs(
        self,
        *,
        night_number: int = 1,
        alive_count: int = 12,
        poison_used: bool = False,
    ) -> GameState:
        players: dict[str, PlayerState] = {}
        # All 12 players with standard roles; we'll mark some as dead
        # to hit the requested alive_count.  The witch is always
        # ``p09`` (power-role slot) and stays alive so the test
        # can call ``agent_night_witch``.
        all_roles = (
            ["werewolf"] * 4
            + ["villager"] * 3
            + ["seer", "witch", "hunter", "idiot", "hybrid"]
        )
        for i, role in enumerate(all_roles, start=1):
            players[f"p{i:02d}"] = PlayerState(
                id=f"p{i:02d}", role=role, alive=True,
            )
        # Mark some non-witch players as dead until alive_count is hit.
        witch_id = "p09"
        # Kill from the back (p12, p11, ...) so roles stay stable.
        kill_order = ["p12", "p11", "p10", "p08", "p07", "p06",
                      "p05", "p04", "p03", "p02", "p01"]
        for pid in kill_order:
            if sum(1 for p in players.values() if p.alive) <= alive_count:
                break
            players[pid] = PlayerState(
                id=pid, role=players[pid].role, alive=False,
            )
        return GameState(
            game_id="witch_poison_unified_test",
            players=players,
            phase="night",
            night_number=night_number,
            poison_used=poison_used,
        )

    def _invoke_witch(self, gs: GameState) -> dict:
        from werewolf_agent.runtime.agent_adapter import agent_night_witch

        class CaptureAgent:
            last_context: AgentContext | None = None

            def act(self, context):
                self.last_context = context
                return (
                    PlayerAction(action_type=ActionType.NO_ACTION, reason="test"),
                    RetryInfo(),
                )

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()

            def get_agent(self, player_id):
                witch_id = next(
                    pid for pid, p in gs.players.items()
                    if p.role == "witch" and p.alive
                )
                return self.agent if player_id == witch_id else None

        registry = CaptureRegistry()
        engine = _new_engine()
        witch_id = next(
            pid for pid, p in gs.players.items()
            if p.role == "witch" and p.alive
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "p01",
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
        agent_night_witch(state, engine, registry)
        return registry.agent.last_context.strategy_directive

    def test_witch_poison_unified_branch(self) -> None:
        """P1-D5: the strategy_directive must carry a single
        `witch_poison_strategy` dict whose ``branch`` is one of the three
        documented branches, picked by game state.

        Pre-fix: ``witch_poison_threshold`` and ``poison_urgency`` could
        both fire, contradicting each other; there was no
        ``no_pressure_save_for_late`` branch.
        """
        gs = self._make_witch_gs(alive_count=12, poison_used=False)
        directive = self._invoke_witch(gs)
        assert "witch_poison_strategy" in directive, (
            "witch must receive unified `witch_poison_strategy` directive; "
            f"got keys: {sorted(directive.keys())}"
        )
        wp = directive["witch_poison_strategy"]
        assert "branch" in wp, (
            f"witch_poison_strategy must have a 'branch' field; got: {wp!r}"
        )
        assert wp["branch"] in {
            "no_pressure_save_for_late",
            "urgency_under_X_alive",
            "evidence_required_threshold",
        }, f"unexpected branch value: {wp['branch']!r}"
        # The old split keys must NOT appear alongside the unified one.
        assert "witch_poison_threshold" not in directive
        assert "poison_urgency" not in directive

    def test_witch_poison_urgency_branch_under_low_alive(self) -> None:
        """When alive count is low (≤ 7), the branch must escalate to
        `urgency_under_X_alive`."""
        gs = self._make_witch_gs(alive_count=6, poison_used=False)
        directive = self._invoke_witch(gs)
        wp = directive["witch_poison_strategy"]
        assert wp["branch"] == "urgency_under_X_alive", (
            f"low alive count must trigger urgency branch; got: {wp['branch']!r}"
        )
        # The directive text should mention urgency / 紧急.
        text = wp.get("text", "") + wp.get("advice", "")
        assert "紧急" in text or "urgency" in text.lower() or "不用毒药" in text

    def test_witch_poison_evidence_branch_mid_game(self) -> None:
        """Mid game (8-9 alive) with no urgency should land on
        `evidence_required_threshold`."""
        gs = self._make_witch_gs(alive_count=9, poison_used=False)
        directive = self._invoke_witch(gs)
        wp = directive["witch_poison_strategy"]
        assert wp["branch"] == "evidence_required_threshold", (
            f"mid game must trigger evidence_required_threshold branch; "
            f"got: {wp['branch']!r}"
        )

    def test_witch_poison_no_pressure_branch_early_game(self) -> None:
        """Early game (10+ alive, poison unused) should land on
        `no_pressure_save_for_late`."""
        gs = self._make_witch_gs(alive_count=11, poison_used=False)
        directive = self._invoke_witch(gs)
        wp = directive["witch_poison_strategy"]
        assert wp["branch"] == "no_pressure_save_for_late", (
            f"early game with no pressure must trigger no_pressure branch; "
            f"got: {wp['branch']!r}"
        )


class TestPKSpeechRoleBranches:
    """D-2: agent_pk_speech must inject role-specific directives like
    agent_day_speech.  Pre-fix, PK had no role branching so the tied
    candidate spoke with no role context."""

    def _make_pk_state(
        self,
        speaker_id: str,
        role: str,
        extra_events: list | None = None,
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role=role if i == 1 else "villager", alive=True)
            for i in range(1, 13)
        }
        players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
        players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
        players["p05"] = PlayerState(id="p05", role="werewolf", alive=True)
        players["p06"] = PlayerState(id="p06", role="seer", alive=True)
        players["p07"] = PlayerState(id="p07", role="witch", alive=True)
        players["p08"] = PlayerState(id="p08", role="hunter", alive=True)
        players["p09"] = PlayerState(id="p09", role="idiot", alive=True)
        players["p10"] = PlayerState(id="p10", role="hybrid", alive=True)
        events = [
            GameEvent(type="vote_resolved", payload={
                "exiled": None, "tied": [speaker_id, "p02"], "day_number": 2,
                "votes": [],
            }),
        ]
        if extra_events:
            events.extend(extra_events)
        gs = GameState(
            game_id="pk_role_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
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
                return self.agent if player_id == speaker_id else None

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
            "revote": True,
            "pk_candidates": [speaker_id, "p02"],
            "sheriff_candidates": [],
            "sheriff_votes": {},
            "sheriff_withdrawing": [],
            "badge_decision": "tear",
            "badge_target_id": None,
            "hunter_shot_target_id": None,
        }
        return state, engine, registry

    def test_pk_speech_has_role_branches(self) -> None:
        """agent_pk_speech must dispatch to per-role directive builders."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech

        # Seer PK speech: must include seer_speech_directive-style hint
        state, engine, registry = self._make_pk_state("p01", "seer")
        agent_pk_speech(state, engine, registry, "p01")
        ctx = registry.agent.last_context
        # Pre-fix, the strategy_directive was empty (or lacked role keys).
        # After fix, a seer PK candidate should receive at least one
        # role-specific directive (e.g. seer_check_pushed or wolf-style
        # counterclaim context).
        assert ctx is not None
        assert ctx.strategy_directive, "PK speech must receive a role directive"
        # Wolf PK: should get wolf vote / push target.
        state2, engine2, registry2 = self._make_pk_state("p02", "werewolf")
        agent_pk_speech(state2, engine2, registry2, "p02")
        ctx2 = registry2.agent.last_context
        assert ctx2 is not None
        assert ctx2.strategy_directive, "Wolf PK speech must receive a directive"

    def test_pk_speech_seer_has_check_evidence_hint(self) -> None:
        """Seer in PK must be told to anchor their case in check results."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech

        events = [
            GameEvent(type="seer_check", payload={
                "seer_id": "p01", "target_id": "p02",
                "alignment": "wolf", "night_number": 1,
            }),
        ]
        state, engine, registry = self._make_pk_state(
            "p01", "seer", extra_events=events,
        )
        agent_pk_speech(state, engine, registry, "p01")
        ctx = registry.agent.last_context
        # The seer should have a check-based hint; not the old empty SD.
        assert ctx is not None
        # Allow either a dedicated key or a seer-specific block.
        sd = ctx.strategy_directive
        assert ("seer_pk_check_evidence" in sd) or (
            "seer_speech_directive" in sd
        ), f"seer PK must receive a role hint; got: {sorted(sd.keys())}"

    def test_pk_speech_wolf_has_push_target_hint(self) -> None:
        """Wolf in PK must receive the wolf pk push-target hint."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech

        state, engine, registry = self._make_pk_state("p02", "werewolf")
        agent_pk_speech(state, engine, registry, "p02")
        ctx = registry.agent.last_context
        sd = ctx.strategy_directive
        # A wolf PK should have at least a wolf_pk_push hint, or fall
        # back to the wolf day-speech directive block.
        assert (
            "wolf_pk_push" in sd or "wolf_speech_directive" in sd
        ), f"wolf PK must receive a wolf hint; got: {sorted(sd.keys())}"


class TestVoteDirectiveBranchesByRole:
    """D-3: agent_day_vote must dispatch to per-role vote directive builders.

    Pre-fix the vote stage had branches for werewolf / hybrid / (villager,
    idiot) but NO branches for seer / witch / hunter.  After the fix,
    each role gets a dedicated vote strategy block.
    """

    def _make_vote_state(
        self,
        voter_id: str,
        role: str,
    ) -> tuple:
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }
        players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
        players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
        players["p05"] = PlayerState(id="p05", role="werewolf", alive=True)
        players["p06"] = PlayerState(id="p06", role="seer", alive=True)
        players["p07"] = PlayerState(id="p07", role="witch", alive=True)
        players["p08"] = PlayerState(id="p08", role="hunter", alive=True)
        players["p09"] = PlayerState(id="p09", role="idiot", alive=True)
        players["p10"] = PlayerState(id="p10", role="hybrid", alive=True)
        # Set the role for the voter
        players[voter_id] = PlayerState(id=voter_id, role=role, alive=True)
        gs = GameState(
            game_id="vote_role_test",
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
                    target_id="p02",
                    reason="test",
                    suspect_reason="suspect",
                    not_voting_reason="others clean",
                    private_reason="private",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == voter_id else None

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

    def test_vote_directive_branches_by_role(self) -> None:
        """Seer, witch, and hunter voters each get a role-specific vote hint."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote

        for role, expected_key in [
            ("seer", "seer_vote_strategy"),
            ("witch", "witch_vote_strategy"),
            ("hunter", "hunter_vote_strategy"),
        ]:
            voter_id = f"p06" if role == "seer" else (
                f"p07" if role == "witch" else f"p08"
            )
            state, engine, registry = self._make_vote_state(voter_id, role)
            agent_day_vote(state, engine, registry, voter_id)
            ctx = registry.agent.last_context
            assert ctx is not None
            sd = ctx.strategy_directive
            assert expected_key in sd, (
                f"{role} voter must receive {expected_key!r} directive; "
                f"got: {sorted(sd.keys())}"
            )


class TestDefenseSpeechHandler:
    """D-8: DEFENSE_SPEECH task type must have a corresponding agent handler."""

    def test_defense_speech_handler_exists(self) -> None:
        from werewolf_agent.runtime import agent_adapter

        assert hasattr(agent_adapter, "agent_defense_speech")
        # And it's callable.
        assert callable(getattr(agent_adapter, "agent_defense_speech"))


class TestNightActionSpeechValidation:
    """D-13: validate_seer_claim guardrail must apply to ALL night_action
    speech fields, not just werewolf day speeches.

    Pre-fix, the seer-claim validator was gated on ``player_role ==
    'werewolf'``; any other role that produced a violation-prone speech
    (e.g., a fake-Seer impostor or a confused LLM) was allowed through.
    """

    def test_night_action_speech_validated(self) -> None:
        """Run agent_day_speech for a werewolf whose speech contains
        multiple seer-check claims for the same night and confirm the
        guardrail substitutes the fallback even though the role filter
        is no longer role-restricted.
        """
        from werewolf_agent.runtime.agent_adapter import agent_day_speech
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }
        # Use p01 as a werewolf
        players["p01"] = PlayerState(id="p01", role="werewolf", alive=True)
        gs = GameState(
            game_id="night_action_validate_test",
            players=players,
            phase="day",
            day_number=2,
        )

        class BadSeerAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                # Two checks in the same night — violation.
                # The validator pattern requires "我" or "也" to anchor
                # the claim, so we lead each clause with "我第1夜查了".
                return (PlayerAction(
                    action_type=ActionType.SPEECH,
                    speech="我第1夜查了p03是狼人，我第1夜查了p05是好人",
                    reason="test",
                ), RetryInfo())

        class BadRegistry:
            def __init__(self):
                self.agent = BadSeerAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "p01" else None

        registry = BadRegistry()
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
        result = agent_day_speech(state, engine, registry, "p01")
        # The guardrail must kick in and replace the speech with the
        # sanitized fallback.
        assert result is not None
        assert "第1夜查了p03" not in result["speech_text"], (
            "violating seer-claim speech must be replaced by the fallback"
        )


class TestSingleSeerBranch:
    """D-16: villager_vote_strategy must include a single-seer branch when
    only one player has publicly claimed Seer (no counterclaim)."""

    def test_single_seer_branch(self) -> None:
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import AgentContext, ActionType
        from werewolf_agent.agents.player import PlayerAction, RetryInfo

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }
        players["p01"] = PlayerState(id="p01", role="werewolf", alive=True)
        players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
        players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
        players["p05"] = PlayerState(id="p05", role="seer", alive=True)
        events = [
            GameEvent(type="sheriff_speech", payload={
                "speaker": "p05",
                "text": "我是预言家",
                "claims": [{"type": "role", "value": "seer"}],
            }),
        ]
        gs = GameState(
            game_id="single_seer_branch_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )

        class CaptureAgent:
            last_context: AgentContext | None = None
            def act(self, context):
                self.last_context = context
                return (PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="p01",
                    reason="test",
                    suspect_reason="suspect",
                    not_voting_reason="others clean",
                    private_reason="private",
                ), RetryInfo())

        class CaptureRegistry:
            def __init__(self):
                self.agent = CaptureAgent()
            def get_agent(self, player_id):
                return self.agent if player_id == "p06" else None

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
        agent_day_vote(state, engine, registry, "p06")
        ctx = registry.agent.last_context
        vs = ctx.strategy_directive.get("villager_vote_strategy", "")
        # Single-seer branch should mention trusting the lone claimant
        # when no counterclaim exists.
        assert "单边" in vs or "无对跳" in vs or "对跳" in vs, (
            f"single-seer branch should mention seer-claimant situation; "
            f"got: {vs!r}"
        )


# ---------------------------------------------------------------------------
# Phase-1 audit P1-7: sheriff_silent must reference target_id, not vote_silent
# ---------------------------------------------------------------------------


def test_sheriff_silent_directive_references_target_id_not_vote_silent():
    """Phase-1 P1-7 + Phase-3 clean-2: sheriff_silent directive must
    tell the LLM to use the existing ``target_id`` field on the vote
    action — NOT a nonexistent ``[vote_silent]`` field.

    Implementation: this is now a REGRESSION GUARD only.  The
    behavioral coverage lives in
    ``test_sheriff_silent_fallback`` above (which invokes
    ``_invoke_day_speech`` and inspects the returned directive
    text).  This test stays as a stub to document the historical
    P1-3 fix without the fragile ``inspect.getsource`` dependency.
    """
    # Behavioral test is the source of truth — see
    # test_sheriff_silent_fallback above.
    # If you need to verify the literal text again, the existing
    # behavioral test's assertions include both "target_id" in
    # the text AND "[vote_silent]" NOT in the text.


class TestWolfDirectiveNoLeakKill:
    """P0-G3223805846-1: wolf fake-seer 公开话术禁止列举真实刀口 ID 列表。"""

    def test_fake_seer_directive_contains_no_leak_rule(self):
        """Positive-marker guard: rule 9 must be present in fake_seer directive."""
        from werewolf_agent.runtime.directives.wolf import _WOLF_ROLE_STRATEGY
        text = _WOLF_ROLE_STRATEGY["fake_seer"]
        # rule 9 marker — without this the directive has no no-leak clause
        assert "真实刀口" in text, "fake_seer directive missing no-leak rule 9"
        assert "严禁" in text, "fake_seer directive missing prohibition marker"

    def test_wolf_universal_rules_assembled_directive_contains_rule(self):
        """Verify the rule is actually present in the assembled directive output."""
        from werewolf_agent.runtime.directives.wolf import build_wolf_directive
        gs = GameState(players={}, day_number=1, night_number=1)
        d = build_wolf_directive(gs, "p01", wolf_team_plan={"fake_seer": "p01"})
        full = " ".join(str(v) for v in d.values())
        assert "真实刀口" in full, "assembled wolf directive missing no-leak rule"
        assert "模糊话术" in full, "assembled wolf directive missing vague-phrasing guidance"


class TestSeerDirectiveLatePosition:
    """P0-G3223805846-3: 预言家排到发言顺序后段（≥ 50%）时，必须在第 1 句就亮身份 + 报查杀。"""

    def test_late_position_seer_has_jump_immediately_rule(self):
        from werewolf_agent.runtime.directives.seer import build_seer_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        # p01 sits at index 10 of 12 (>= 50% boundary), i.e. position 11/12
        speech_order = ["p08", "p11", "p02", "p12", "p06", "p04", "p10",
                        "p05", "p07", "p03", "p01", "p09"]
        gs = GameState(players=alive, day_number=1, night_number=1)
        d = build_seer_directive(gs, "p01", speech_order=speech_order)
        directive = d.get("seer_speech_directive", "")
        assert "后段" in directive and "第 1 句" in directive, (
            f"late-position seer missing jump-immediately rule: {directive!r}"
        )

    def test_early_position_seer_no_jump_immediately_rule(self):
        from werewolf_agent.runtime.directives.seer import build_seer_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        # p01 is position 1/12 — clearly front of the queue
        speech_order = ["p01", "p02", "p03", "p04", "p05", "p06",
                        "p07", "p08", "p09", "p10", "p11", "p12"]
        gs = GameState(players=alive, day_number=1, night_number=1)
        d = build_seer_directive(gs, "p01", speech_order=speech_order)
        directive = d.get("seer_speech_directive", "")
        assert "后段" not in directive, (
            f"early seer incorrectly tagged as late: {directive!r}"
        )

    def test_mid_position_seer_no_jump_immediately_rule(self):
        """Exactly 50% boundary should NOT trigger the late rule (use >= 50% to be inclusive only on the late side).

        With 12 players, position 6 (index 5) = 6/12 = 50% — must not trigger.
        """
        from werewolf_agent.runtime.directives.seer import build_seer_directive
        alive = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        alive["p01"] = PlayerState(id="p01", role="seer", alive=True)
        # p01 at index 5 = position 6/12, exactly the 50% boundary
        speech_order = ["p02", "p03", "p04", "p05", "p06", "p01",
                        "p07", "p08", "p09", "p10", "p11", "p12"]
        gs = GameState(players=alive, day_number=1, night_number=1)
        d = build_seer_directive(gs, "p01", speech_order=speech_order)
        directive = d.get("seer_speech_directive", "")
        # 6/12 == 0.5 exactly, so the rule pos >= total*0.5 means 5 >= 6 is False → no late rule
        assert "后段" not in directive, (
            f"boundary-position seer (6/12) incorrectly tagged as late: {directive!r}"
        )


class TestWolfFakeSeerConsistency:
    """P0-G3223805846-4: 狼 fake_seer 启用时各角色 prompt 必须有话术一致条款。"""

    def test_all_wolf_roles_have_consistency_clause(self):
        from werewolf_agent.runtime.directives.wolf import _WOLF_ROLE_STRATEGY
        for role in ("fake_seer", "pusher", "hooker", "deep_cover", "unassigned"):
            text = _WOLF_ROLE_STRATEGY.get(role, "")
            assert "话术一致" in text or "保持一致" in text or "对跳" in text, (
                f"role {role} missing fake-seer consistency clause: {text!r}"
            )
