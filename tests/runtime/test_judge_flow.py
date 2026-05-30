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
