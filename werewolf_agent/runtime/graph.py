"""LangGraph game graph: deterministic orchestration around RuleEngine.

Every node calls RuleEngine for rule decisions. No natural language adjudication.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from werewolf_agent.core.models import (
    Death,
    GameEvent,
    GameState,
    PlayerState,
    VictoryResult,
)
from werewolf_agent.engine.rule_engine import RuleEngine

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


# ---------------------------------------------------------------------------
# Runtime state — LangGraph operates on this dict, wraps immutable GameState
# ---------------------------------------------------------------------------

class RuntimeState(TypedDict, total=False):
    game_state: GameState
    engine: RuleEngine
    # Night action inputs (set by scripted/agent nodes before resolve)
    wolf_kill_target_id: str | None
    wolf_action: str
    wolf_action_reason: str
    use_antidote: bool
    poison_target_id: str | None
    seer_target_id: str | None
    hybrid_master_target_id: str | None
    # Day action inputs
    self_destruct_wolf_id: str | None
    current_speaker_id: str | None
    speech_order: list[str]
    speech_index: int
    speech_text: str
    speech_timed_out: bool
    speech_seconds_limit: int
    # Vote inputs
    exile_votes: dict[str, str]
    revote: bool
    # Sheriff election inputs
    sheriff_candidates: list[str]
    sheriff_votes: dict[str, str]
    sheriff_withdrawing: list[str]
    # Badge decision after sheriff death
    badge_decision: str  # "transfer" or "tear"
    badge_target_id: str | None
    # Hunter shot
    hunter_shot_target_id: str | None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _new_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


def _player_ids(gs: GameState) -> list[str]:
    return list(gs.players.keys())


def _alive_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]


def _alive_non_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]


def _find_role(gs: GameState, role: str) -> str | None:
    return next((pid for pid, p in gs.players.items() if p.role == role and p.alive), None)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def setup_game(state: RuntimeState) -> dict[str, Any]:
    gs = state.get("game_state")
    engine = state.get("engine") or _new_engine()
    if gs is None:
        gs = GameState(ruleset_id="pre_witch_hunter_idiot_mixed", game_id=uuid.uuid4().hex[:8])
    gs = replace(gs, phase="setup")
    return {"game_state": gs, "engine": engine}


def assign_roles(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    player_ids = [f"p{i:02d}" for i in range(1, 13)]
    players = engine.assign_roles(player_ids, seed=_stable_seed(gs.game_id, "roles"))
    gs = replace(gs, players=players, phase="roles_assigned",
                 events=gs.events + [GameEvent(type="roles_assigned", payload={})])
    return {"game_state": gs}


# -- Night nodes --

def enter_night(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    n = gs.night_number + 1
    gs = replace(gs, phase="night", night_number=n,
                 events=gs.events + [GameEvent(type="enter_night", payload={"night": n})])
    return {"game_state": gs}


def wolf_discussion(state: RuntimeState) -> dict[str, Any]:
    # Placeholder: in V2 agents will discuss here. For now, record event.
    gs: GameState = state["game_state"]
    gs = replace(gs, events=gs.events + [GameEvent(type="wolf_discussion", payload={})])
    return {"game_state": gs}


def wolf_consensus(state: RuntimeState) -> dict[str, Any]:
    """Determine wolf night action; timeout defaults to no-kill."""
    gs: GameState = state["game_state"]
    action = state.get("wolf_action")
    target = state.get("wolf_kill_target_id")

    if action == "no_kill":
        event = GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "night_number": gs.night_number,
                "reason": state.get("wolf_action_reason", ""),
            },
        )
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": None}

    if (action == "kill" or action is None) and target is not None:
        target_state = gs.players.get(target)
        if target_state is None or not target_state.alive:
            event = GameEvent(
                type="wolf_no_kill_timeout",
                payload={"night_number": gs.night_number},
            )
            gs = replace(gs, events=gs.events + [event])
            return {"game_state": gs, "wolf_kill_target_id": None}
        event = GameEvent(
            type="wolf_kill_selected",
            payload={"night_number": gs.night_number, "target_id": target},
        )
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": target}

    event = GameEvent(
        type="wolf_no_kill_timeout",
        payload={"night_number": gs.night_number},
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs, "wolf_kill_target_id": None}


def night_witch(state: RuntimeState) -> dict[str, Any]:
    # Scripted: no antidote, no poison by default
    return {"use_antidote": state.get("use_antidote", False),
            "poison_target_id": state.get("poison_target_id")}


def night_seer(state: RuntimeState) -> dict[str, Any]:
    # Scripted: no action by default, just pass through
    return {"seer_target_id": state.get("seer_target_id")}


def first_night_hybrid_master(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    if gs.night_number != 1 or gs.hybrid_master_id is not None:
        return {}
    engine: RuleEngine = state["engine"]
    master_target = state.get("hybrid_master_target_id")
    if master_target is None:
        import random
        candidates = [pid for pid in _player_ids(gs) if pid != _find_role(gs, "hybrid")]
        rng = random.Random(_stable_seed(gs.game_id, "hybrid_master"))
        master_target = rng.choice(candidates) if candidates else None
    if master_target is None:
        return {}
    hybrid_id = _find_role(gs, "hybrid")
    if hybrid_id is None:
        return {}
    gs, event = engine.choose_master(gs, hybrid_id=hybrid_id, master_id=master_target)
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs}


def resolve_night(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    gs, events = engine.resolve_night(
        gs,
        night_number=gs.night_number,
        wolf_kill_target_id=state.get("wolf_kill_target_id"),
        use_antidote=state.get("use_antidote", False),
        poison_target_id=state.get("poison_target_id"),
    )
    if events:
        gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


# -- Day nodes --

def announce_deaths(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    d = gs.day_number + 1
    gs = replace(gs, phase="day", day_number=d,
                 events=gs.events + [GameEvent(type="day_announce", payload={"day": d})])
    return {"game_state": gs}


def night_death_last_words(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    eligible = []
    for death in gs.deaths:
        if death.timing == "night" and engine.can_leave_last_words(
            death_reason=death.reason, timing=death.timing, night_number=gs.night_number
        ):
            eligible.append(death.player_id)
    gs = replace(gs, events=gs.events + [GameEvent(
        type="night_death_last_words", payload={"players": eligible}
    )])
    return {"game_state": gs}


# -- Sheriff election (first day only) --

def sheriff_registration(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    candidates = state.get("sheriff_candidates", [])
    if not candidates:
        candidates = [pid for pid, p in gs.players.items() if p.alive]
    gs = replace(gs, sheriff_candidates=candidates,
                 events=gs.events + [GameEvent(type="sheriff_registered", payload={"candidates": candidates})])
    return {"game_state": gs, "sheriff_candidates": candidates}


def sheriff_speech(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs = replace(gs, events=gs.events + [GameEvent(type="sheriff_speech", payload={})])
    return {"game_state": gs}


def sheriff_withdraw(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    candidates = gs.sheriff_candidates
    withdrawing = state.get("sheriff_withdrawing", [])
    gs, event = engine.sheriff_withdraw(gs, candidates=candidates, withdrawing=withdrawing)
    remaining = event.payload.get("remaining", candidates)
    gs = replace(gs, sheriff_candidates=remaining, events=gs.events + [event])
    return {"game_state": gs, "sheriff_candidates": remaining}


def sheriff_vote(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    votes = state.get("sheriff_votes", {})
    candidates = gs.sheriff_candidates
    gs, event = engine.resolve_sheriff_vote(gs, votes=votes, candidates=candidates)
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs}


# -- Day discussion & vote --

def free_discussion(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    speaker_id = state.get("current_speaker_id")
    if speaker_id is None and speech_index < len(speech_order):
        speaker_id = speech_order[speech_index]
    def advance_speaker() -> dict[str, Any]:
        next_index = speech_index + 1
        next_speaker = speech_order[next_index] if next_index < len(speech_order) else None
        return {
            "speech_index": next_index,
            "current_speaker_id": next_speaker,
            "speech_timed_out": False,
            "speech_text": "",
        }

    if speaker_id and state.get("speech_timed_out", False):
        gs = replace(gs, events=gs.events + [GameEvent(
            type="speech_timeout",
            payload={
                "player_id": speaker_id,
                "day_number": gs.day_number,
                "seconds_limit": state.get("speech_seconds_limit", 0),
            },
        )])
        return {"game_state": gs, **advance_speaker()}
    if speaker_id:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="speech",
            payload={
                "speaker": speaker_id,
                "day_number": gs.day_number,
                "text": state.get("speech_text", ""),
            },
        )])
        return {"game_state": gs, **advance_speaker()}
    gs = replace(gs, events=gs.events + [GameEvent(type="free_discussion", payload={})])
    return {"game_state": gs}


def day_vote(state: RuntimeState) -> dict[str, Any]:
    return {"exile_votes": state.get("exile_votes", {}),
            "revote": state.get("revote", False)}


def resolve_vote(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.resolve_vote(
        gs, votes=state.get("exile_votes", {}), revote=state.get("revote", False)
    )
    gs = replace(gs, votes=state.get("exile_votes", {}),
                 events=gs.events + [GameEvent(
                     type="vote_resolved",
                     payload={"exiled": result.exiled_player_id, "reason": result.reason},
                 )])
    return {"game_state": gs, "_vote_result": result}


def resolve_exile(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = state.get("_vote_result")
    if result is None or result.exiled_player_id is None:
        return {"game_state": gs}
    gs, events = engine.resolve_exile(gs, target_id=result.exiled_player_id)
    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def post_exile_skills(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    # Check if hunter died and can shoot
    for death in gs.deaths:
        if death.player_id in gs.players:
            player = gs.players[death.player_id]
            if player.role == "hunter" and not player.alive:
                if engine.can_hunter_shoot(gs, hunter_id=death.player_id, death_reason=death.reason):
                    target = state.get("hunter_shot_target_id")
                    if target:
                        shot_death = Death(
                            player_id=target, reason="hunter_shot",
                            timing="post_exile", resolution_batch=death.resolution_batch,
                            source_player_id=death.player_id,
                        )
                        gs = engine.apply_death(gs, shot_death)
    return {"game_state": gs}


def check_victory(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.check_victory(gs)
    checked_payload = {"winner": result.winner, "reason": result.reason}
    gs = replace(gs, events=gs.events + [GameEvent(type="victory_checked", payload=checked_payload)])

    if result.winner is not None:
        wf = result.winner
        hr = None
        if wf == "good" and gs.hybrid_master_faction == "good":
            hr = "win"
        elif wf == "good" and gs.hybrid_master_faction == "werewolf":
            hr = "lose"
        elif wf == "werewolf" and gs.hybrid_master_faction == "werewolf":
            hr = "win"
        elif wf == "werewolf" and gs.hybrid_master_faction == "good":
            hr = "lose"
        gs = replace(gs, winning_faction=wf, hybrid_result=hr,
                     events=gs.events + [GameEvent(
                         type="victory",
                         payload={
                             "winner": wf,
                             "winning_faction": wf,
                             "reason": result.reason,
                             "hybrid_master_id": gs.hybrid_master_id,
                             "hybrid_master_faction": gs.hybrid_master_faction,
                             "hybrid_result": hr,
                         },
                     )])
    return {"game_state": gs, "_victory_result": result}


def sheriff_badge_transfer(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return {"game_state": gs}
    sheriff = gs.players.get(gs.sheriff_id)
    if sheriff is None or sheriff.alive:
        return {"game_state": gs}
    # Sheriff is dead, find last death reason
    death_reason = "exile"
    for death in reversed(gs.deaths):
        if death.player_id == gs.sheriff_id:
            death_reason = death.reason
            break
    decision = state.get("badge_decision", "tear")
    target_id = state.get("badge_target_id")
    gs = engine.resolve_badge_decision(gs, decision=decision, target_id=target_id)
    event_type = "badge_torn" if decision == "tear" else "badge_transferred"
    gs = replace(gs, events=gs.events + [GameEvent(
        type=event_type,
        payload={"new_sheriff_id": target_id} if decision == "transfer" else {},
    )])
    return {"game_state": gs}


def finish_game(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs = replace(gs, phase="finished",
                 events=gs.events + [GameEvent(type="game_finished", payload={})])
    return {"game_state": gs}


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def route_after_resolve_night(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    result = engine.check_victory(gs)
    if result.winner is not None:
        return "check_victory"
    return "announce_deaths"


def route_after_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    # Check the last vote_resolved event for routing info
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            if event.payload.get("exiled") is not None:
                return "resolve_exile"
            if event.payload.get("reason") == "first_tie_pk":
                return "tie_pk_speech"
            break
    return "check_victory"


def route_after_exile(state: RuntimeState) -> str:
    return "post_exile_skills"


def route_after_post_exile(state: RuntimeState) -> str:
    return "check_victory"


def route_victory(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.winning_faction is not None:
        return "finish_game"
    # Sheriff badge transfer if sheriff died
    if gs.sheriff_id and gs.sheriff_badge_state == "active":
        sheriff = gs.players.get(gs.sheriff_id)
        if sheriff and not sheriff.alive:
            return "sheriff_badge_transfer"
    return "enter_night"


def route_after_announce(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.day_number == 1:
        return "sheriff_registration"
    return "free_discussion"


def route_after_sheriff_vote(state: RuntimeState) -> str:
    return "free_discussion"


def route_self_destruct_check(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    if speech_order and speech_index < len(speech_order):
        return "continue_discussion"
    return "day_vote"


def resolve_self_destruct_node(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id:
        gs, events = engine.resolve_self_destruct(gs, wolf_id=wolf_id, day_number=gs.day_number)
        gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def tie_pk_speech(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs = replace(gs, events=gs.events + [GameEvent(type="tie_pk_speech", payload={})])
    return {"game_state": gs}


def tie_revote(state: RuntimeState) -> dict[str, Any]:
    return {"exile_votes": state.get("exile_votes", {}),
            "revote": True}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_game_graph() -> CompiledStateGraph:
    graph = StateGraph(RuntimeState)
    _add_all_nodes(graph)
    _add_all_edges(graph)
    return graph.compile()


def build_game_graph_with_checkpoint(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build the graph with an external checkpointer for pause/resume."""
    graph = StateGraph(RuntimeState)
    _add_all_nodes(graph)
    _add_all_edges(graph)
    return graph.compile(checkpointer=checkpointer)


def _add_all_nodes(graph: StateGraph) -> None:
    graph.add_node("setup_game", setup_game)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("enter_night", enter_night)
    graph.add_node("wolf_discussion", wolf_discussion)
    graph.add_node("wolf_consensus", wolf_consensus)
    graph.add_node("night_witch", night_witch)
    graph.add_node("night_seer", night_seer)
    graph.add_node("first_night_hybrid_master", first_night_hybrid_master)
    graph.add_node("resolve_night_node", resolve_night)
    graph.add_node("announce_deaths", announce_deaths)
    graph.add_node("night_death_last_words", night_death_last_words)
    graph.add_node("sheriff_registration", sheriff_registration)
    graph.add_node("sheriff_speech", sheriff_speech)
    graph.add_node("sheriff_withdraw", sheriff_withdraw)
    graph.add_node("sheriff_vote", sheriff_vote)
    graph.add_node("free_discussion", free_discussion)
    graph.add_node("resolve_self_destruct", resolve_self_destruct_node)
    graph.add_node("day_vote", day_vote)
    graph.add_node("resolve_vote_node", resolve_vote)
    graph.add_node("tie_pk_speech", tie_pk_speech)
    graph.add_node("tie_revote", tie_revote)
    graph.add_node("resolve_exile", resolve_exile)
    graph.add_node("post_exile_skills", post_exile_skills)
    graph.add_node("check_victory", check_victory)
    graph.add_node("sheriff_badge_transfer", sheriff_badge_transfer)
    graph.add_node("finish_game", finish_game)


def _add_all_edges(graph: StateGraph) -> None:
    graph.set_entry_point("setup_game")
    graph.add_edge("setup_game", "assign_roles")
    graph.add_edge("assign_roles", "enter_night")
    graph.add_edge("enter_night", "wolf_discussion")
    graph.add_edge("wolf_discussion", "wolf_consensus")
    graph.add_edge("wolf_consensus", "night_witch")
    graph.add_edge("night_witch", "night_seer")
    graph.add_edge("night_seer", "first_night_hybrid_master")
    graph.add_edge("first_night_hybrid_master", "resolve_night_node")
    graph.add_conditional_edges("resolve_night_node", route_after_resolve_night, {
        "check_victory": "check_victory",
        "announce_deaths": "announce_deaths",
    })
    graph.add_edge("announce_deaths", "night_death_last_words")
    graph.add_conditional_edges("night_death_last_words", route_after_announce, {
        "sheriff_registration": "sheriff_registration",
        "free_discussion": "free_discussion",
    })
    graph.add_edge("sheriff_registration", "sheriff_speech")
    graph.add_edge("sheriff_speech", "sheriff_withdraw")
    graph.add_edge("sheriff_withdraw", "sheriff_vote")
    graph.add_conditional_edges("sheriff_vote", route_after_sheriff_vote, {
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("free_discussion", route_self_destruct_check, {
        "resolve_self_destruct": "resolve_self_destruct",
        "continue_discussion": "free_discussion",
        "day_vote": "day_vote",
    })
    graph.add_edge("resolve_self_destruct", "check_victory")
    graph.add_edge("day_vote", "resolve_vote_node")
    graph.add_conditional_edges("resolve_vote_node", route_after_vote, {
        "resolve_exile": "resolve_exile",
        "tie_pk_speech": "tie_pk_speech",
        "check_victory": "check_victory",
    })
    graph.add_edge("tie_pk_speech", "tie_revote")
    graph.add_edge("tie_revote", "resolve_vote_node")
    graph.add_edge("resolve_exile", "post_exile_skills")
    graph.add_edge("post_exile_skills", "check_victory")
    graph.add_conditional_edges("check_victory", route_victory, {
        "finish_game": "finish_game",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "enter_night": "enter_night",
    })
    graph.add_edge("sheriff_badge_transfer", "enter_night")
    graph.add_edge("finish_game", END)
