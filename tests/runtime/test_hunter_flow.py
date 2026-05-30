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



class _HunterMockRegistry:
    """Mock registry for hunter shot tests."""

    def __init__(self, shot_target: str | None = None) -> None:
        self._target = shot_target
        self.hunter_called = False

    def get_agent(self, player_id: str):
        return _HunterMockAgent(self)



class _HunterMockAgent:
    def __init__(self, registry: _HunterMockRegistry) -> None:
        self._registry = registry

    def act(self, context):
        from werewolf_agent.agents.schemas import (
            ActionType, FallbackAction, PlayerAction, RetryInfo,
        )
        self._registry.hunter_called = True
        if self._registry._target and context.legal_actions and any(
            a.value == "hunter_shot" for a in context.legal_actions
        ):
            return PlayerAction(
                action_type=ActionType.HUNTER_SHOT,
                target_id=self._registry._target,
                speech="带走你",
                reason="shoot",
                confidence=0.9,
                private_intent=None,
            ), RetryInfo(attempts=0, errors=[])
        return FallbackAction(
            action_type=ActionType.NO_ACTION,
            target_id=None,
            reason="no shot",
        ), RetryInfo(attempts=0, errors=[])

class TestHunterShotTiming:
    """Design doc §3.2: hunter can shoot when killed by wolves or exiled,
    cannot shoot when poisoned by witch."""

    def test_hunter_killed_by_wolf_can_shoot(self) -> None:
        """Hunter killed by wolf kill at night must be able to shoot a target."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hunter_wolf", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "hunter",
            "use_antidote": False,
            "poison_target_id": None,
        })

        # Hunter should be dead from wolf kill
        assert result["game_state"].players["hunter"].alive is False
        # There should be a triggered skill indicating hunter can shoot
        hunter_deaths = [d for d in result["game_state"].deaths if d.player_id == "hunter"]
        assert len(hunter_deaths) == 1
        assert "hunter_shot" in hunter_deaths[0].triggered_skills, (
            "Hunter killed by wolf should have hunter_shot in triggered_skills"
        )

    def test_hunter_exiled_can_shoot_in_post_exile(self) -> None:
        """Hunter killed by exile must route to resolve_hunter_shot before victory."""
        from werewolf_agent.runtime.graph import (
            post_exile_skills,
            resolve_hunter_shot,
            route_after_post_exile,
        )
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        death = Death(
            player_id="hunter", reason="exile", timing="day",
            resolution_batch="day_1_exile", triggered_skills=["hunter_shot"],
        )
        gs = GameState(
            game_id="hunter_exile",
            players=players,
            deaths=[death],
        )

        result = post_exile_skills({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        pending_state = result["game_state"]
        assert pending_state.players["v1"].alive is True
        assert route_after_post_exile({"game_state": pending_state, "engine": engine}) == "resolve_hunter_shot"

        shot_result = resolve_hunter_shot({
            "game_state": pending_state,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        assert shot_result["game_state"].players["v1"].alive is False
        shot_deaths = [d for d in shot_result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1, "hunter_shot death should be recorded"

    def test_hunter_poisoned_cannot_shoot(self) -> None:
        """Hunter killed by witch poison must NOT be able to shoot."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
            "witch": PlayerState(id="witch", role="witch"),
        }
        gs = GameState(game_id="hunter_poison", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": "v1",
            "use_antidote": False,
            "poison_target_id": "hunter",
        })

        # Hunter should be dead from poison
        assert result["game_state"].players["hunter"].alive is False
        hunter_deaths = [d for d in result["game_state"].deaths if d.player_id == "hunter"]
        assert len(hunter_deaths) == 1
        assert "hunter_shot" not in hunter_deaths[0].triggered_skills, (
            "Hunter killed by witch poison must NOT have hunter_shot triggered"
        )

    def test_hunter_shot_replay_reconstructs_death(self) -> None:
        """Replay must reconstruct hunter_shot death from events."""
        engine = _new_engine()
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        initial = GameState(game_id="replay_hunter", players=players)
        events = [
            GameEvent(type="player_died", payload={
                "player_id": "hunter", "reason": "wolf_kill", "timing": "night",
                "resolution_batch": "night_1",
            }),
            GameEvent(type="player_died", payload={
                "player_id": "v1", "reason": "hunter_shot", "timing": "night",
                "resolution_batch": "night_1", "source_player_id": "hunter",
            }),
        ]
        replayed = replay_from_events(engine, initial, events)
        assert not replayed.players["hunter"].alive
        assert not replayed.players["v1"].alive
        shot_deaths = [d for d in replayed.deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1


class TestHunterShotOrdering:
    """Hunter shots triggered by exile must resolve before the next night."""

    def test_route_after_post_exile_routes_pending_hunter_shot_before_victory(self) -> None:
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter",
            reason="exile",
            timing="day_vote",
            resolution_batch="day_4_vote",
            triggered_skills=["hunter_shot"],
        )
        players = {
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
            "villager": PlayerState(id="villager", role="villager", alive=True),
            "idiot": PlayerState(id="idiot", role="idiot", alive=True),
        }
        gs = GameState(
            game_id="hunter_order",
            players=players,
            deaths=[hunter_death],
            day_number=4,
            phase="day",
            events=[
                GameEvent(type="player_died", payload={
                    "player_id": "hunter",
                    "reason": "exile",
                    "timing": "day_vote",
                    "resolution_batch": "day_4_vote",
                    "triggered_skills": ["hunter_shot"],
                }),
            ],
        )

        assert route_after_post_exile({"game_state": gs, "engine": engine}) == "resolve_hunter_shot"

    def test_post_exile_skills_does_not_resolve_scripted_hunter_shot_silently(self) -> None:
        from werewolf_agent.runtime.graph import post_exile_skills

        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter",
            reason="exile",
            timing="day_vote",
            resolution_batch="day_4_vote",
            triggered_skills=["hunter_shot"],
        )
        players = {
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
            "villager": PlayerState(id="villager", role="villager", alive=True),
        }
        gs = GameState(
            game_id="hunter_post_exile_pending",
            players=players,
            deaths=[hunter_death],
            day_number=4,
            phase="day",
        )

        result = post_exile_skills({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "wolf",
        })

        new_state = result["game_state"]
        assert new_state.players["wolf"].alive is True
        assert route_after_post_exile({"game_state": new_state, "engine": engine}) == "resolve_hunter_shot"

    def test_daytime_hunter_shot_returns_to_victory_check_not_night_announcement(self) -> None:
        engine = _new_engine()
        players = {
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=False),
            "villager": PlayerState(id="villager", role="villager", alive=True),
            "idiot": PlayerState(id="idiot", role="idiot", alive=True),
        }
        gs = GameState(
            game_id="hunter_day_route",
            players=players,
            deaths=[
                Death(
                    player_id="hunter",
                    reason="exile",
                    timing="day_vote",
                    resolution_batch="day_4_vote",
                    triggered_skills=["hunter_shot"],
                ),
                Death(
                    player_id="wolf",
                    reason="hunter_shot",
                    timing="day_vote",
                    resolution_batch="day_4_vote",
                    source_player_id="hunter",
                ),
            ],
            day_number=4,
            phase="day",
        )

        assert route_after_hunter_shot({"game_state": gs, "engine": engine}) == "check_victory"

    def test_resolve_hunter_shot_uses_target_declared_in_exile_last_words(self) -> None:
        from werewolf_agent.runtime.graph import resolve_hunter_shot

        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter",
            reason="exile",
            timing="day_vote",
            resolution_batch="day_3_vote",
            triggered_skills=["hunter_shot"],
        )
        players = {
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
            "villager": PlayerState(id="villager", role="villager", alive=True),
        }
        gs = GameState(
            game_id="hunter_last_words_target",
            players=players,
            deaths=[hunter_death],
            day_number=3,
            phase="day",
            events=[
                GameEvent(
                    type="exile_last_words",
                    payload={
                        "speaker": "hunter",
                        "day_number": 3,
                        "text": "我是猎人，被放逐出局时可以开枪。我选择带走wolf。",
                    },
                )
            ],
        )

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": None,
        })

        new_state = result["game_state"]
        assert not new_state.players["wolf"].alive
        shot_deaths = [d for d in new_state.deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1
        assert shot_deaths[0].player_id == "wolf"
        assert shot_deaths[0].source_player_id == "hunter"

    def test_resolve_hunter_shot_records_decline_when_no_target(self) -> None:
        from werewolf_agent.runtime.graph import resolve_hunter_shot

        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter",
            reason="exile",
            timing="day_vote",
            resolution_batch="day_3_vote",
            triggered_skills=["hunter_shot"],
        )
        players = {
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
            "villager": PlayerState(id="villager", role="villager", alive=True),
        }
        gs = GameState(
            game_id="hunter_declines_shot",
            players=players,
            deaths=[hunter_death],
            day_number=3,
            phase="day",
        )

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "agent_registry": _HunterMockRegistry(shot_target=None),
            "hunter_shot_target_id": None,
        })

        new_state = result["game_state"]
        assert new_state.players["wolf"].alive is True
        assert any(
            event.type == "judge_broadcast"
            and event.payload.get("phase") == "hunter_shot_prompt"
            for event in new_state.events
        )
        assert any(
            event.type == "hunter_shot_declined"
            and event.payload["hunter_id"] == "hunter"
            for event in new_state.events
        )


class TestHunterShotResolution:
    """Night hunter shot: when hunter is killed by wolves, the shot must be
    resolved before victory check. Agent-driven target selection when registry
    provided."""

    def test_resolve_hunter_shot_applies_shot_at_night(self) -> None:
        """resolve_hunter_shot node applies hunter_shot death when hunter is wolf-killed."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_night", players=players, deaths=[hunter_death])

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        assert result["game_state"].players["v1"].alive is False
        shot_deaths = [d for d in result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 1
        assert shot_deaths[0].source_player_id == "hunter"

    def test_resolve_hunter_shot_no_shot_when_poisoned(self) -> None:
        """Hunter killed by witch poison does NOT trigger a shot."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        poison_death = Death(
            player_id="hunter", reason="witch_poison", timing="night",
            resolution_batch="night_1", triggered_skills=[],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_poison", players=players, deaths=[poison_death])

        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "hunter_shot_target_id": "v1",
        })

        assert result["game_state"].players["v1"].alive is True
        shot_deaths = [d for d in result["game_state"].deaths if d.reason == "hunter_shot"]
        assert len(shot_deaths) == 0

    def test_resolve_hunter_shot_agent_selects_target(self) -> None:
        """Agent registry provides hunter shot target when no scripted target."""
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_agent", players=players, deaths=[hunter_death])

        registry = _HunterMockRegistry(shot_target="v1")
        result = resolve_hunter_shot({
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
            "hunter_shot_target_id": None,
        })

        assert registry.hunter_called
        assert result["game_state"].players["v1"].alive is False

    def test_route_after_resolve_night_routes_to_hunter_shot(self) -> None:
        """route_after_resolve_night routes to resolve_hunter_shot when hunter has pending shot."""
        from werewolf_agent.runtime.graph import route_after_resolve_night
        engine = _new_engine()
        hunter_death = Death(
            player_id="hunter", reason="wolf_kill", timing="night",
            resolution_batch="night_1", triggered_skills=["hunter_shot"],
        )
        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "v1": PlayerState(id="v1", role="villager"),
        }
        gs = GameState(game_id="hs_route", players=players, deaths=[hunter_death])

        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "resolve_hunter_shot"

    def test_route_after_resolve_night_no_hunter_shot_skips(self) -> None:
        """route_after_resolve_night skips hunter_shot when no pending shot."""
        from werewolf_agent.runtime.graph import route_after_resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
        gs = GameState(game_id="hs_skip", players=players)

        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result != "resolve_hunter_shot"

    def test_graph_contains_resolve_hunter_shot_node(self) -> None:
        """Graph must include resolve_hunter_shot node."""
        graph = build_game_graph()
        assert "resolve_hunter_shot" in graph.nodes, (
            "resolve_hunter_shot node is missing from the graph"
        )

class TestHunterShotPublicEvent:
    """F4: hunter_shot_public event should be emitted."""

    def test_hunter_shot_emits_public_event(self) -> None:
        from werewolf_agent.runtime.graph import resolve_hunter_shot
        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)
        }
        players["p01"] = PlayerState(id="p01", role="werewolf", alive=True)
        players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
        players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
        players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
        players["p05"] = PlayerState(id="p05", role="hunter", alive=False)
        gs = GameState(game_id="hunter_event_test", players=players, phase="day", day_number=1)
        # Add a hunter death
        death = Death(
            player_id="p05", reason="exile", timing="day_vote",
            resolution_batch="day_1_vote", can_leave_last_words=True,
            triggered_skills=["hunter_shot"],
        )
        state = {
            "game_state": replace(gs, deaths=[death]),
            "engine": _new_engine(),
            "agent_registry": None,
            "hunter_shot_target_id": "p01",
        }
        result = resolve_hunter_shot(state)
        new_gs = result.get("game_state", gs)
        event_types = [e.type for e in new_gs.events]
        assert "hunter_shot_public" in event_types, f"Missing hunter_shot_public event. Got: {event_types}"