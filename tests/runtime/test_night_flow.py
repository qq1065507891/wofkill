# -*- coding: utf-8 -*-
"""
验证夜间节点编排、行动结算与终局提交。

作者: Project contributors
修改日期: 2026-07-13
"""

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


def test_night_resolution_commits_victory_before_day_agents() -> None:
    from werewolf_agent.runtime.graph import resolve_night, route_after_resolve_night

    engine = _new_engine()
    gs = GameState(
        game_id="night_terminal",
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
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
        "poison_target_id": None,
        "seer_target_id": None,
    })

    assert result["game_state"].winning_faction == "werewolf"
    assert len([e for e in result["game_state"].events if e.type == "victory"]) == 1
    assert route_after_resolve_night({**result, "engine": engine}) == "reflection"


# Mock class
class _CapturingSeerRegistry:
    def __init__(self, seer_id: str) -> None:
        self.seer_id = seer_id
        self.context = None

    def get_agent(self, player_id: str):
        if player_id != self.seer_id:
            return None
        return self

    def act(self, context):
        from werewolf_agent.agents.schemas import ActionType, PlayerAction, RetryInfo

        self.context = context
        target = context.legal_targets[0] if context.legal_targets else None
        return PlayerAction(
            action_type=ActionType.CHECK_ALIGNMENT,
            target_id=target,
            speech="",
            reason="check first legal target",
            confidence=0.8,
        ), RetryInfo()

# Test classes
class TestNightHunterIdiotStatusNode:
    """Design doc §6.2 requires a night_hunter_idiot_status node between
    night_seer and first_night_hybrid_master."""

    def test_graph_contains_night_hunter_idiot_status_node(self) -> None:
        graph = build_game_graph()
        assert "night_hunter_idiot_status" in graph.nodes, (
            "Design doc §6.2 node 7 'night_hunter_idiot_status' is missing from the graph"
        )

    def test_night_edge_order_seer_to_hunter_idiot(self) -> None:
        """night_seer must route to night_hunter_idiot_status, not directly to first_night_hybrid_master."""
        graph = build_game_graph()
        assert "night_hunter_idiot_status" in graph.nodes
        # Verify the edge: night_seer -> night_hunter_idiot_status
        # We check by running a simple night flow and ensuring the node appears
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=99)
        gs = GameState(game_id="edge_order", players=players, phase="night", night_number=1)

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

        nodes_seen = []
        for chunk in graph.stream(state, {"recursion_limit": 50}):
            for node_name in chunk.keys():
                nodes_seen.append(node_name)
                if node_name == "resolve_night_node":
                    break
            else:
                continue
            break

        # The night flow must be: wolf_discussion, wolf_consensus, night_witch,
        # night_seer, night_hunter_idiot_status, first_night_hybrid_master, resolve_night_node
        night_nodes = [n for n in nodes_seen if n in {
            "wolf_discussion", "wolf_consensus", "night_witch", "night_seer",
            "night_hunter_idiot_status", "first_night_hybrid_master", "resolve_night_node",
        }]
        assert "night_hunter_idiot_status" in night_nodes, (
            f"night_hunter_idiot_status node was not visited during night flow. "
            f"Night nodes seen: {night_nodes}"
        )
        # Verify ordering
        if "night_hunter_idiot_status" in night_nodes and "first_night_hybrid_master" in night_nodes:
            assert night_nodes.index("night_hunter_idiot_status") < night_nodes.index("first_night_hybrid_master"), (
                "night_hunter_idiot_status must come before first_night_hybrid_master in night flow"
            )


class TestSeerNightResolution:
    """Design doc §3.3: seer checks alignment each night. resolve_night must
    produce a seer_check event with target_id, alignment, night_number."""

    def test_resolve_night_produces_seer_check_event(self) -> None:
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        seer_id = next(pid for pid, p in players.items() if p.role == "seer")
        # Pick a non-seer target
        target_id = next(pid for pid, p in players.items() if p.role != "seer" and p.alive)
        gs = GameState(game_id="seer_test", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": target_id,
        })

        events = result["game_state"].events
        seer_checks = [e for e in events if e.type == "seer_check"]
        assert len(seer_checks) == 1, f"Expected exactly 1 seer_check event, got {len(seer_checks)}"
        check = seer_checks[0]
        # seer_id 不再包含在 payload 中（H-5：防止通过事件泄漏预言家身份）
        assert "seer_id" not in check.payload
        assert check.payload["target_id"] == target_id
        assert check.payload["alignment"] in ("good", "werewolf")
        assert check.payload["night_number"] == 1

    def test_seer_check_hybrid_returns_good(self) -> None:
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        seer_id = next(pid for pid, p in players.items() if p.role == "seer")
        hybrid_id = next((pid for pid, p in players.items() if p.role == "hybrid"), None)
        if hybrid_id is None:
            pytest.skip("No hybrid in this seed")
        gs = GameState(game_id="seer_hybrid", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": hybrid_id,
        })

        seer_checks = [e for e in result["game_state"].events if e.type == "seer_check"]
        assert len(seer_checks) == 1
        assert seer_checks[0].payload["alignment"] == "good", (
            "Seer must see hybrid as good (design doc §3.2)"
        )

    def test_seer_check_is_not_in_public_timeline(self) -> None:
        """seer_check must be private; it must not appear in public events."""
        from werewolf_agent.runtime.graph import resolve_night
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        target_id = next(pid for pid, p in players.items() if p.role != "seer" and p.alive)
        gs = GameState(game_id="seer_private", players=players, night_number=1)

        result = resolve_night({
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": target_id,
        })

        seer_checks = [e for e in result["game_state"].events if e.type == "seer_check"]
        assert len(seer_checks) == 1
        # seer_check event must have a visibility marker indicating it's private
        check = seer_checks[0]
        assert check.payload.get("visibility") in ("private", "seer_only", "moderator_only"), (
            "seer_check must have visibility marker preventing public exposure"
        )



def test_seer_night_targets_exclude_counterclaiming_seers() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_night_seer

    players = {
        "seer": PlayerState(id="seer", role="seer"),
        "fake": PlayerState(id="fake", role="werewolf"),
        "p03": PlayerState(id="p03", role="villager"),
        "p04": PlayerState(id="p04", role="villager"),
    }
    gs = GameState(
        game_id="seer_counterclaim_filter",
        players=players,
        phase="night",
        day_number=1,
        night_number=2,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "fake",
                    "day_number": 1,
                    "text": "我是预言家，seer是假预言家。",
                },
            ),
        ],
    )
    registry = _CapturingSeerRegistry("seer")

    result = agent_night_seer({"game_state": gs, "engine": RuleEngine({})}, RuleEngine({}), registry)

    assert result is not None
    assert registry.context is not None
    assert "fake" not in registry.context.legal_targets
    assert result["seer_target_id"] != "fake"
    assert "p03" in registry.context.legal_targets
    assert "不要查验对跳预言家的玩家" in registry.context.strategy_directive["seer_night_check"]


def test_all_players_on_sheriff_announces_no_sheriff_before_speeches() -> None:
    from werewolf_agent.runtime.graph import route_after_sheriff_speech, sheriff_speech

    players = {
        "p01": PlayerState(id="p01", role="seer"),
        "p02": PlayerState(id="p02", role="werewolf"),
        "p03": PlayerState(id="p03", role="villager"),
    }
    gs = GameState(
        game_id="all_on_sheriff_vote",
        players=players,
        day_number=1,
        phase="day",
        sheriff_candidates=["p01", "p02", "p03"],
    )

    result = sheriff_speech({
        "game_state": gs,
        "engine": RuleEngine({}),
    })

    new_gs = result["game_state"]
    broadcasts = [
        event.payload
        for event in new_gs.events
        if event.type == "judge_broadcast"
    ]
    all_on_broadcast = next(
        payload for payload in broadcasts
        if payload.get("phase") == "sheriff_all_players_registered"
    )
    assert "全员上警" in all_on_broadcast["message"]
    assert "警徽流失" in all_on_broadcast["message"]
    assert "本局无警长" in all_on_broadcast["message"]
    assert "由" in all_on_broadcast["message"] and "开始发言" in all_on_broadcast["message"]
    assert all_on_broadcast["speech_order"][0] in players
    assert new_gs.sheriff_id is None
    # No death_announce broadcast yet → route to announce_deaths
    assert route_after_sheriff_speech({"game_state": new_gs}) == "announce_deaths"




# ---------------------------------------------------------------------------
# Night resolution standalone tests
# ---------------------------------------------------------------------------

def test_resolve_night_node_kills_target() -> None:
    from werewolf_agent.runtime.graph import resolve_night
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="rn1", players=players, night_number=1)
    result = resolve_night({
        "game_state": gs, "engine": engine,
        "wolf_kill_target_id": "p01", "use_antidote": False, "poison_target_id": None,
    })
    assert result["game_state"].players["p01"].alive is False

def test_resolve_night_node_keeps_witch_potion_events_in_timeline() -> None:
    from werewolf_agent.runtime.graph import resolve_night
    engine = _new_engine()
    players = {
        "w1": PlayerState(id="w1", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "witch": PlayerState(id="witch", role="witch"),
    }
    gs = GameState(game_id="rn_witch", players=players, night_number=1)

    result = resolve_night({
        "game_state": gs,
        "engine": engine,
        "wolf_kill_target_id": "v1",
        "use_antidote": True,
        "poison_target_id": None,
    })

    assert any(event.type == "witch_antidote_used" for event in result["game_state"].events)

def test_announce_deaths_increments_day() -> None:
    from werewolf_agent.runtime.graph import announce_deaths
    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=1)
    gs = GameState(game_id="d1", players=players, night_number=1, day_number=0)
    result = announce_deaths({"game_state": gs, "engine": engine})
    assert result["game_state"].day_number == 1
