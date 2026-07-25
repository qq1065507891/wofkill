from __future__ import annotations

import json

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



class TestJudgeControlsNightRoleSequence:
    """Night role sequence has judge broadcasts."""

    def test_game_start_has_public_judge_broadcast(self):
        from werewolf_agent.runtime.graph import setup_game

        gs = GameState(game_id="start_broadcast", phase="init")

        result = setup_game({"game_state": gs})

        broadcasts = [e for e in result["game_state"].events if e.type == "judge_broadcast"]
        assert any(
            b.payload.get("phase") == "game_start"
            and b.payload.get("visibility") == "public"
            and "游戏开始" in b.payload.get("message", "")
            for b in broadcasts
        )

    def test_judge_controls_night_role_sequence(self):
        """Event stream contains judge broadcasts before/after each night role stage."""
        from werewolf_agent.runtime.graph import enter_night

        gs = GameState(
            game_id="test",
            phase="setup",
            players={f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else ("seer" if i == 5 else ("witch" if i == 9 else "villager")),
                alive=True,
            ) for i in range(1, 13)},
        )

        result = enter_night({"game_state": gs, "agent_registry": None})
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "enter_night" in phases

    def test_night_witch_skips_when_witch_dead(self):
        """Dead witch must not receive wake/choose/sleep broadcasts or actions."""
        from werewolf_agent.runtime.graph import night_witch

        gs = GameState(
            game_id="dead_witch",
            phase="night",
            night_number=3,
            players={
                "witch": PlayerState(id="witch", role="witch", alive=False),
                "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
                "villager": PlayerState(id="villager", role="villager", alive=True),
            },
        )

        result = night_witch({"game_state": gs, "engine": _new_engine()})

        phases = [
            event.payload.get("phase")
            for event in result["game_state"].events
            if event.type == "judge_broadcast"
        ]
        assert "witch_wake" not in phases
        assert "witch_choose" not in phases
        assert "witch_sleep" not in phases
        assert result["use_antidote"] is False
        assert result["poison_target_id"] is None

    def test_night_seer_skips_when_seer_dead(self):
        """Dead seer must not receive wake/choose/sleep broadcasts or actions."""
        from werewolf_agent.runtime.graph import night_seer

        gs = GameState(
            game_id="dead_seer",
            phase="night",
            night_number=3,
            players={
                "seer": PlayerState(id="seer", role="seer", alive=False),
                "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
                "villager": PlayerState(id="villager", role="villager", alive=True),
            },
        )

        result = night_seer({"game_state": gs, "engine": _new_engine()})

        phases = [
            event.payload.get("phase")
            for event in result["game_state"].events
            if event.type == "judge_broadcast"
        ]
        assert "seer_wake" not in phases
        assert "seer_choose" not in phases
        assert "seer_sleep" not in phases
        assert result["seer_target_id"] is None

    def test_wolf_stages_have_broadcasts(self):
        """Wolf discussion and consensus emit judge broadcasts."""
        from werewolf_agent.runtime.graph import wolf_consensus, wolf_discussion

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=1,
            players={f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else "villager",
                alive=True,
            ) for i in range(1, 13)},
        )

        state = {"game_state": gs, "agent_registry": None}
        result = wolf_discussion(state)
        state = {**state, **result}
        result = wolf_consensus(state)
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "wolf_wake" in phases
        assert "wolf_discussion_start" in phases
        assert "wolf_discussion_end" in phases
        assert "wolf_kill_choice" in phases
        assert "wolf_sleep" in phases
        action_idx = next(i for i, e in enumerate(gs.events) if e.type in {
            "wolf_kill_selected",
            "wolf_no_kill_declared",
            "wolf_no_kill_timeout",
        })
        sleep_idx = next(i for i, e in enumerate(gs.events)
                         if e.type == "judge_broadcast" and e.payload.get("phase") == "wolf_sleep")
        assert action_idx < sleep_idx

    def test_night_roles_have_open_choose_result_and_close_broadcasts(self):
        from werewolf_agent.runtime.graph import (
            first_night_hybrid_master,
            night_hunter_idiot_status,
            night_seer,
            night_witch,
            resolve_night,
        )

        players = {
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="seer"),
            "p03": PlayerState(id="p03", role="witch"),
            "p04": PlayerState(id="p04", role="hunter"),
            "p05": PlayerState(id="p05", role="idiot"),
            "p06": PlayerState(id="p06", role="hybrid"),
            "p07": PlayerState(id="p07", role="villager"),
        }
        gs = GameState(game_id="night_broadcasts", phase="night", night_number=1, players=players)
        state = {
            "game_state": gs,
            "engine": _new_engine(),
            "agent_registry": None,
            "wolf_kill_target_id": "p07",
            "seer_target_id": "p01",
            "use_antidote": False,
            "poison_target_id": None,
            "hybrid_master_target_id": "p07",
        }

        for node in (night_seer, night_witch, night_hunter_idiot_status, first_night_hybrid_master, resolve_night):
            result = node(state)
            state = {**state, **result}

        broadcasts = [e for e in state["game_state"].events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        for phase in [
            "seer_wake", "seer_choose", "seer_result", "seer_sleep",
            "witch_wake", "witch_choose", "witch_sleep",
            "hunter_status", "idiot_status",
            "hybrid_wake", "hybrid_choose", "hybrid_sleep",
        ]:
            assert phase in phases
        seer_result = next(b for b in broadcasts if b.payload.get("phase") == "seer_result")
        assert seer_result.payload["visibility"] == "seer_private"
        phase_order = [e.payload.get("phase") for e in state["game_state"].events if e.type == "judge_broadcast"]
        assert phase_order.index("seer_result") < phase_order.index("seer_sleep")

    def test_public_death_announcement_does_not_reveal_death_reason(self):
        from werewolf_agent.runtime.graph import announce_deaths

        players = {
            "p01": PlayerState(id="p01", role="villager", alive=False),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        }
        gs = GameState(
            game_id="death_no_reason",
            phase="night",
            night_number=1,
            players=players,
            deaths=[Death(
                player_id="p01",
                reason="wolf_kill",
                timing="night",
                resolution_batch="night_1",
            )],
        )

        result = announce_deaths({"game_state": gs})

        broadcasts = [
            e for e in result["game_state"].events
            if e.type == "judge_broadcast" and e.payload.get("phase") == "death_announce"
        ]
        assert broadcasts
        message = broadcasts[-1].payload["message"]
        assert "p01" in message
        for forbidden in ["原因", "狼人", "女巫", "毒", "猎人", "枪", "wolf", "poison", "hunter"]:
            assert forbidden not in message


class TestJudgeControlsDaySequence:
    """Day flow broadcasts include all major transitions."""

    def test_day_announce_has_broadcast(self):
        from werewolf_agent.runtime.graph import announce_deaths

        gs = GameState(
            game_id="test",
            phase="night",
            night_number=1,
            day_number=0,
            players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)},
        )
        result = announce_deaths({"game_state": gs})
        gs = result["game_state"]

        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "day_announce" in phases

    def test_vote_phase_has_broadcast(self):
        """day_vote should emit a judge broadcast."""
        from werewolf_agent.runtime.graph import day_vote

        gs = GameState(
            game_id="test",
            phase="day",
            day_number=2,
            players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)},
        )
        state = {
            "game_state": gs,
            "agent_registry": None,
            "exile_votes": {"p01": "p03"},
            "exile_vote_day": 0,
            "revote": False,
        }
        result = day_vote(state)
        gs = result.get("game_state", gs)
        broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
        phases = [b.payload.get("phase") for b in broadcasts]
        assert "vote_start" in phases


class TestJudgeAgentWiredToGraph:
    """Layer 1: JudgeAgent is wired into the runtime graph."""

    def _make_state(self, *, judge_agent=None, judge_llm_enabled=False):
        """Build a minimal RuntimeState for judge broadcast testing."""
        gs = GameState(
            game_id="test", phase="setup",
            players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
                     for i in range(1, 13)},
        )
        state: dict[str, Any] = {"game_state": gs}
        if judge_agent is not None:
            state["judge_agent"] = judge_agent
            state["judge_llm_enabled"] = judge_llm_enabled
        return state

    def test_jb_uses_fallback_when_no_judge_agent(self):
        """When judge_agent is not in state, _jb uses hardcoded fallback message."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state()
        gs, _ = _jb(state, phase="test_phase", message="FALLBACK_UNIQUE_MSG")
        assert "FALLBACK_UNIQUE_MSG" in gs.events[-1].payload["message"]

    def test_jb_uses_fallback_when_llm_disabled(self):
        """When judge_llm_enabled=False, _jb uses hardcoded fallback even with agent present."""
        from werewolf_agent.runtime.nodes._shared import _jb
        from werewolf_agent.agents.judge import JudgeAgent
        judge = JudgeAgent(model_router=None)
        state = self._make_state(judge_agent=judge, judge_llm_enabled=False)
        gs, _ = _jb(state, phase="test_phase", message="FALLBACK_UNIQUE_MSG")
        assert "FALLBACK_UNIQUE_MSG" in gs.events[-1].payload["message"]

    def test_jb_delegates_to_judge_agent_when_llm_enabled(self):
        """With judge_agent + judge_llm_enabled=True, JudgeAgent template is used."""
        from werewolf_agent.runtime.nodes._shared import _jb
        from werewolf_agent.agents.judge import JudgeAgent
        judge = JudgeAgent(model_router=None)
        state = self._make_state(judge_agent=judge, judge_llm_enabled=True)
        # "night" phase — JudgeAgent has its own template for this
        gs, _ = _jb(state, phase="night", message="FALLBACK_UNIQUE_MSG", night_number=1)
        msg = gs.events[-1].payload["message"]
        # JudgeAgent's template produces a different message than the fallback
        assert "FALLBACK_UNIQUE_MSG" not in msg
        assert "天黑" in msg or "N1" in msg

    def test_judge_broadcast_event_structure_unchanged(self):
        """judge_broadcast events keep the same structure for backward compat."""
        from werewolf_agent.runtime.nodes._shared import _jb
        from werewolf_agent.agents.judge import JudgeAgent
        judge = JudgeAgent(model_router=None)
        state = self._make_state(judge_agent=judge, judge_llm_enabled=True)
        _, event = _jb(state, phase="vote", message="fallback", day_number=2,
                        visibility="public")
        assert event.type == "judge_broadcast"
        assert "phase" in event.payload
        assert "message" in event.payload
        assert "visibility" in event.payload

    def test_judge_broadcast_helpers_are_split_from_node_helpers(self):
        from werewolf_agent.runtime.nodes import judge_broadcast_helpers
        from werewolf_agent.runtime.nodes import node_helpers
        from werewolf_agent.runtime.nodes import _shared

        assert node_helpers._jb is judge_broadcast_helpers._jb
        assert node_helpers._judge_broadcast is judge_broadcast_helpers._judge_broadcast
        assert (
            node_helpers._generate_judge_message
            is judge_broadcast_helpers._generate_judge_message
        )
        assert _shared._jb is judge_broadcast_helpers._jb
        assert _shared._judge_broadcast is judge_broadcast_helpers._judge_broadcast

    def test_game_runner_injects_judge_agent_when_registry_exists(self):
        """GameRunner with use_agent_registry=True creates and injects JudgeAgent."""
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(
            seed=42,
            use_agent_registry=True,
            judge_llm_enabled=False,
            model_config_path="config/models.yaml",
        )
        runner = GameRunner(config)
        assert runner._judge_agent is not None
        rt = runner._build_runtime_state()
        assert "judge_agent" in rt
        assert rt["judge_llm_enabled"] is False
        assert rt["judge_agent"] is runner._judge_agent

    def test_game_runner_no_judge_agent_without_registry(self):
        """Without agent_registry, GameRunner does not create a JudgeAgent."""
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(seed=42, use_agent_registry=False)
        runner = GameRunner(config)
        assert runner._judge_agent is None
        rt = runner._build_runtime_state()
        assert "judge_agent" not in rt


class TestJudgeStructuredBroadcasts:
    """Layer 2: structured vote calling, skill guidance, vote tally, exile."""

    def _make_state_with_judge(self, *, judge_llm_enabled=True,
                                players=None, day_number=1, night_number=1):
        """Build RuntimeState with a JudgeAgent wired in."""
        from werewolf_agent.agents.judge import JudgeAgent
        ps = players or {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }
        gs = GameState(game_id="test", phase="day", day_number=day_number,
                       night_number=night_number, players=ps)
        judge = JudgeAgent(model_router=None)
        return {
            "game_state": gs,
            "judge_agent": judge,
            "judge_llm_enabled": judge_llm_enabled,
        }

    # -- Vote calling --

    def test_vote_calling_produces_per_voter_event(self):
        """Per-voter vote calling emits judge_broadcast with phase=vote_calling."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge()
        gs = state["game_state"]
        gs_now, ev = _jb(
            state, phase="vote_calling", message="fallback",
            judge_method="vote_calling",
            extra_payload={"voter_id": "p01", "voter_name": "测试", "position": 1, "total": 10},
        )
        assert ev.type == "judge_broadcast"
        assert ev.payload["phase"] == "vote_calling"

    def test_vote_calling_public_visibility(self):
        """Vote calling events have public visibility."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge()
        _, ev = _jb(
            state, phase="vote_calling", message="fallback",
            visibility="public", judge_method="vote_calling",
            extra_payload={"voter_id": "p01", "voter_name": "测试", "position": 1, "total": 10},
        )
        assert ev.payload["visibility"] == "public"

    # -- Skill guidance --

    def test_skill_guide_for_witch(self):
        """Skill guide for witch dispatches to guide_skill_use."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge(night_number=2)
        gs = state["game_state"]
        gs_now, ev = _jb(
            state, phase="witch_wake", message="女巫请睁眼",
            gs=gs, night_number=2, visibility="moderator_only",
            judge_method="skill_guide",
            extra_payload={
                "role": "witch", "player_id": "p09", "player_name": "女巫玩家",
                "available_actions": ["use_antidote", "use_poison", "no_action"],
            },
        )
        assert ev.type == "judge_broadcast"
        assert "女巫" in ev.payload["message"]

    def test_skill_guide_for_seer(self):
        """Skill guide for seer dispatches to guide_skill_use."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge(night_number=1)
        gs = state["game_state"]
        _, ev = _jb(
            state, phase="seer_wake", message="预言家请睁眼",
            gs=gs, night_number=1, visibility="moderator_only",
            judge_method="skill_guide",
            extra_payload={
                "role": "seer", "player_id": "p05", "player_name": "预言家玩家",
                "available_actions": ["check_alignment"],
            },
        )
        assert "预言家" in ev.payload["message"]

    def test_skill_guide_for_hunter(self):
        """Skill guide for hunter dispatches correctly."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge()
        gs = state["game_state"]
        _, ev = _jb(
            state, phase="hunter_shot_prompt", message="猎人开枪",
            gs=gs, visibility="public", judge_method="skill_guide",
            extra_payload={
                "role": "hunter", "player_id": "p08", "player_name": "猎人玩家",
                "available_actions": ["hunter_shot", "no_action"],
            },
        )
        assert "猎人" in ev.payload["message"]

    def test_skill_guide_fallback_when_no_llm(self):
        """Skill guide falls back to hardcoded message when no model_router."""
        from werewolf_agent.runtime.nodes._shared import _jb
        from werewolf_agent.agents.judge import JudgeAgent
        judge = JudgeAgent(model_router=None)  # No router → uses templates
        gs = GameState(game_id="test", phase="night",
                       players={f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
                                for i in range(1, 13)})
        state = {"game_state": gs, "judge_agent": judge, "judge_llm_enabled": True}
        _, ev = _jb(
            state, phase="seer_wake", message="FALLBACK_MSG",
            gs=gs, night_number=1, visibility="moderator_only",
            judge_method="skill_guide",
            extra_payload={
                "role": "seer", "player_id": "p05", "player_name": "预言家",
                "available_actions": ["check_alignment"],
            },
        )
        # With no model_router, guide_skill_use returns template output ≠ fallback
        assert "FALLBACK_MSG" not in ev.payload["message"]

    # -- Vote tally --

    def test_vote_tally_includes_weighted_counts(self):
        """Vote tally broadcast includes structured tally in extra_payload."""
        from werewolf_agent.runtime.graph import _new_engine
        from werewolf_agent.runtime.nodes.day_vote import _broadcast_vote_details

        players = {
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="werewolf", alive=True),
        }
        gs = GameState(
            game_id="vote_tally_v2",
            phase="day",
            day_number=2,
            players=players,
            sheriff_id="p01",
            sheriff_badge_state="active",
        )
        state = {
            "game_state": gs,
            "engine": _new_engine(),
            "judge_llm_enabled": False,
        }

        gs_now = _broadcast_vote_details(
            state,
            gs,
            {"p01": "p03", "p02": "p02"},
        )
        event = gs_now.events[-1]
        payload = event.payload

        assert event.type == "judge_broadcast"
        assert payload["vote_weight_format_version"] == 2
        assert payload["base_vote_weight"] == 2
        assert payload["tally"] == payload["tally_units"] == {"p03": 3, "p02": 2}
        assert payload["sheriff_weight"] == payload["sheriff_weight_units"] == 3
        assert payload["tally_display"] == {"p03": 1.5, "p02": 1}
        assert payload["sheriff_weight_display"] == 1.5
        assert type(payload["tally_display"]["p03"]) is float
        assert type(payload["tally_display"]["p02"]) is int
        assert type(payload["sheriff_weight_display"]) is float
        json.dumps(payload)
        assert "警长1.5票" in payload["message"]
        assert "警长3票" not in payload["message"]

    # -- Exile announcement --

    def test_exile_result_with_player(self):
        """Exile announcement with a valid exiled player."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge()
        gs = state["game_state"]
        _, ev = _jb(
            state, phase="exile", message="放逐",
            gs=gs, judge_method="exile",
            extra_payload={"exiled_player_id": "p03", "exiled_player_name": "玩家3",
                           "reason": "majority"},
        )
        assert "p03" in ev.payload["message"] or "玩家3" in ev.payload["message"]

    def test_exile_result_first_tie(self):
        """Exile announcement for first tie PK."""
        from werewolf_agent.runtime.nodes._shared import _jb
        state = self._make_state_with_judge()
        _, ev = _jb(
            state, phase="vote_tie_pk", message="平票",
            gs=state["game_state"], judge_method="exile",
            extra_payload={"reason": "first_tie_pk", "tied_player_ids": ["p03", "p07"]},
        )
        assert "PK" in ev.payload["message"] or "平票" in ev.payload["message"]


class TestJudgeProfileRouter:
    """Layer 3: JudgeProfileRouter loads and resolves persona profiles."""

    def test_load_from_yaml(self):
        """JudgeProfileRouter loads all 4 profiles from judge_profiles.yaml."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        profiles = router.list_profiles()
        assert "tournament_referee" in profiles
        assert "variety_show_host" in profiles
        assert "neutral_arbiter" in profiles
        assert "ancient_mystic" in profiles

    def test_resolve_by_profile_id(self):
        """Resolve a specific judge profile by ID."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        snap = router.resolve("variety_show_host", "judge_phase")
        assert snap.profile_id == "variety_show_host"
        assert snap.tone_variant == "variety_show"
        assert snap.display_name == "综艺节目主持人"
        assert "warmth" in snap.base_params
        assert snap.base_params["humor"] > 0.5
        assert len(snap.system_prompt) > 0

    def test_resolve_by_tone_variant(self):
        """Resolve by tone variant name."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        snap = router.resolve_by_tone("tournament")
        assert snap.tone_variant == "tournament"
        assert snap.base_params["authority"] > 0.8

    def test_resolve_fallback_to_neutral(self):
        """Unknown profile_id falls back to neutral_arbiter."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        snap = router.resolve("nonexistent_profile")
        assert snap.profile_id == "neutral_arbiter"

    def test_task_styles_per_profile(self):
        """Each profile has task_style mappings for judge task types."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        for pid in router.list_profiles():
            snap = router.resolve(pid, "judge_vote_calling")
            assert snap.task_styles, f"{pid} has no task_styles"
            assert snap.tone_variant, f"{pid} has no tone_variant"

    def test_broadcast_patterns_exist(self):
        """Each profile has broadcast_patterns with key phases."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        for pid in router.list_profiles():
            snap = router.resolve(pid)
            patterns = snap.broadcast_patterns
            for key in ("phase_open", "death_announce", "exile"):
                assert key in patterns, f"{pid} missing broadcast_pattern: {key}"


class TestJudgePersonaIntegration:
    """Layer 3: JudgeAgent persona injection and persona-aware broadcasts."""

    def test_judge_agent_accepts_profile_router(self):
        """JudgeAgent.__init__ accepts optional profile_router and profile_id."""
        from werewolf_agent.agents.judge import JudgeAgent
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        judge = JudgeAgent(model_router=None, profile_router=router,
                           profile_id="variety_show_host")
        assert judge._profile_router is router
        assert judge._profile_id == "variety_show_host"

    def test_judge_agent_resolves_persona(self):
        """JudgeAgent._resolve_persona returns a valid snapshot when router configured."""
        from werewolf_agent.agents.judge import JudgeAgent
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        judge = JudgeAgent(model_router=None, profile_router=router,
                           profile_id="tournament_referee")
        persona = judge._resolve_persona("judge_vote_calling")
        assert persona is not None
        assert persona.tone_variant == "tournament"

    def test_judge_agent_persona_none_without_router(self):
        """Without profile_router, _resolve_persona returns None."""
        from werewolf_agent.agents.judge import JudgeAgent
        judge = JudgeAgent(model_router=None)
        assert judge._resolve_persona() is None
        assert "不得推断平安夜原因" in judge._persona_system_prompt()

    def test_persona_inject_prepends_prompt(self):
        """_persona_inject returns (user_prompt, system_prompt) tuple.

        J-7: the function no longer concatenates the persona into the
        user prompt — it returns the user prompt unchanged as the first
        tuple element and the persona system_prompt as the second.
        """
        from werewolf_agent.agents.judge import JudgeAgent
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml("config/personas/judge_profiles.yaml")
        judge = JudgeAgent(model_router=None, profile_router=router,
                           profile_id="ancient_mystic")
        user_prompt, system_prompt = judge._persona_inject("请宣布天黑闭眼", "judge_phase")
        # user_prompt keeps the request and adds a fact-only boundary
        assert user_prompt.startswith("请宣布天黑闭眼")
        assert "不得补充未提供的身份、技能或夜间原因" in user_prompt
        # system_prompt carries the persona — ancient_mystic should match
        # one of the documented sentinel phrases.
        assert (
            "上古玄学" in system_prompt
            or "命运" in system_prompt
            or "神秘" in system_prompt
        )

    def test_game_runner_loads_judge_profile_router(self):
        """GameRunner with use_agent_registry loads JudgeProfileRouter."""
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(
            seed=42, use_agent_registry=True,
            judge_persona_profile_id="neutral_arbiter",
            model_config_path="config/models.yaml",
        )
        runner = GameRunner(config)
        assert runner._judge_agent is not None
        # Judge agent should have profile_router set
        assert runner._judge_agent._profile_router is not None
        profiles = runner._judge_agent._profile_router.list_profiles()
        assert "neutral_arbiter" in profiles
