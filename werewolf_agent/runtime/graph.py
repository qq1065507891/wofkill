"""LangGraph game graph: deterministic orchestration around RuleEngine.

Every node calls RuleEngine for rule decisions. No natural language adjudication.

Node function implementations live in ``werewolf_agent.runtime.nodes``; this
module owns graph construction, conditional-edge routing, and re-exports.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.sheriff_policy import is_all_players_on_sheriff

# -- Re-export everything from nodes so external imports stay unchanged --
from werewolf_agent.runtime.nodes._shared import (  # noqa: F401
    RULESET_PATH,
    RuntimeState,
    logger,
    _action_trace_event,
    _alive_non_wolves,
    _alive_wolves,
    _agent_timeout,
    _build_wolf_team_plan,
    _call_agent,
    _deaths_already_announced,
    _dispatch_agent,
    _ensure_day_incremented,
    _find_role,
    _force_wolf_kill,
    _judge_broadcast,
    _needs_sheriff_before_deaths,
    _new_engine,
    _player_display,
    _player_ids,
    _planned_wolf_kill,
    _private_vote_audit_payload,
    _public_vote_reason,
    _sheriff_died_this_batch,
    _stable_seed,
    _timer_expired,
    _with_vote_target_in_trace,
)

from werewolf_agent.runtime.nodes.night import (  # noqa: F401
    enter_night,
    night_hunter_idiot_status,
    night_seer,
    night_witch,
    first_night_hybrid_master,
    resolve_night,
    wolf_consensus,
    wolf_discussion,
)

from werewolf_agent.runtime.nodes.day import (  # noqa: F401
    announce_deaths,
    announce_deaths_with_badge_loss,
    check_victory,
    day_vote,
    exile_last_words,
    finish_game,
    free_discussion,
    night_death_last_words,
    resolve_exile,
    resolve_vote,
)

from werewolf_agent.runtime.nodes.sheriff import (  # noqa: F401
    sheriff_first_day_entry,
    sheriff_registration,
    sheriff_speech,
    sheriff_vote,
    sheriff_withdraw,
)

from werewolf_agent.runtime.nodes.skills import (  # noqa: F401
    _hunter_shot_target_from_last_words,
    post_exile_skills,
    resolve_hunter_shot,
    resolve_self_destruct_node,
    sheriff_badge_transfer,
    tie_pk_speech,
    tie_revote,
)

from werewolf_agent.runtime.nodes.sheriff import (  # noqa: F401
    sheriff_endorse,
)

from werewolf_agent.runtime.nodes.summary import (  # noqa: F401
    _route_after_summarize,
    reflection,
    summarize_context,
    summarize_positions,
)


# ---------------------------------------------------------------------------
# Setup nodes (tiny, tightly coupled to graph init — stay here)
# ---------------------------------------------------------------------------

def setup_game(state: RuntimeState) -> dict[str, Any]:
    gs = state.get("game_state")
    engine = state.get("engine") or _new_engine()
    if gs is None:
        gs = GameState(ruleset_id="pre_witch_hunter_idiot_mixed", game_id=uuid.uuid4().hex[:8])
    gs = replace(gs, phase="setup")
    if not any(e.type == "judge_broadcast" and e.payload.get("phase") == "game_start" for e in gs.events):
        gs, _ = _judge_broadcast(
            phase="game_start",
            message="游戏开始，请所有玩家确认身份，准备进入首夜",
            gs=gs,
            visibility="public",
        )
    return {"game_state": gs, "engine": engine}


def assign_roles(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    player_ids = [f"p{i:02d}" for i in range(1, 13)]
    players = engine.assign_roles(player_ids, seed=_stable_seed(gs.game_id, "roles"))
    gs = replace(gs, players=players, phase="roles_assigned",
                 events=gs.events + [GameEvent(type="roles_assigned", payload={})])
    logger.debug(f"\n{'='*60}")
    logger.debug(f"  角色分配完成 (Game: {gs.game_id})")
    for pid, p in sorted(players.items()):
        logger.debug(f"    {_player_display(state, pid)}: {p.role}")
    logger.debug(f"{'='*60}")
    return {"game_state": gs}


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def route_after_resolve_night(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    for death in gs.deaths:
        if "hunter_shot" in (death.triggered_skills or []):
            if death.player_id in gs.players and not gs.players[death.player_id].alive:
                already_shot = any(
                    d.source_player_id == death.player_id and d.reason == "hunter_shot"
                    for d in gs.deaths
                )
                if not already_shot:
                    return "resolve_hunter_shot"
    engine: RuleEngine = state["engine"]
    result = engine.check_victory(gs)
    if result.winner is not None:
        return "check_victory"
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    if _needs_sheriff_before_deaths(gs):
        return "sheriff_first_day_entry"
    return "announce_deaths"


def route_after_hunter_shot(state: RuntimeState) -> str:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.check_victory(gs)
    if result.winner is not None:
        return "check_victory"
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    if gs.phase != "night":
        if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
            return "announce_deaths_with_badge_loss"
        if _needs_sheriff_before_deaths(gs):
            return "sheriff_first_day_entry"
        return "announce_deaths"
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    if _needs_sheriff_before_deaths(gs):
        return "sheriff_first_day_entry"
    return "announce_deaths"


def route_after_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
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
    gs: GameState = state["game_state"]
    for death in gs.deaths:
        if "hunter_shot" not in (death.triggered_skills or []):
            continue
        if death.player_id not in gs.players:
            continue
        if gs.players[death.player_id].alive:
            continue
        already_shot = any(
            d.source_player_id == death.player_id and d.reason == "hunter_shot"
            for d in gs.deaths
        )
        if not already_shot:
            return "resolve_hunter_shot"
    return "check_victory"


def _route_after_badge_transfer(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.phase == "night":
        if _needs_sheriff_before_deaths(gs):
            return "sheriff_first_day_entry"
        return "announce_deaths"
    return "enter_night"


def route_victory(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.winning_faction is not None:
        return "finish_game"
    if gs.sheriff_id and gs.sheriff_badge_state == "active":
        sheriff = gs.players.get(gs.sheriff_id)
        if sheriff and not sheriff.alive:
            return "sheriff_badge_transfer"
    return "enter_night"


def route_after_announce(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.sheriff_interrupt_count == 1 and gs.sheriff_id is None:
        return "sheriff_first_day_entry"
    return "free_discussion"


def route_after_sheriff_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    if not _deaths_already_announced(gs):
        return "announce_deaths"
    return "free_discussion"


def _route_after_sheriff_phase(state: RuntimeState, default_next: str) -> str:
    wolf_id = state.get("self_destruct_wolf_id")
    gs: GameState = state["game_state"]
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    return default_next


def route_after_sheriff_registration(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_speech")


def route_after_sheriff_speech(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if is_all_players_on_sheriff(gs, list(gs.sheriff_candidates or [])):
        if not _deaths_already_announced(gs):
            return _route_after_sheriff_phase(state, "announce_deaths")
        return _route_after_sheriff_phase(state, "free_discussion")
    return _route_after_sheriff_phase(state, "sheriff_withdraw")


def route_after_sheriff_withdraw(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_vote")


def route_self_destruct_check(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    if speech_order and speech_index < len(speech_order):
        return "continue_discussion"
    return "summarize_positions"


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
    graph.add_node("night_hunter_idiot_status", night_hunter_idiot_status)
    graph.add_node("first_night_hybrid_master", first_night_hybrid_master)
    graph.add_node("resolve_night_node", resolve_night)
    graph.add_node("sheriff_first_day_entry", sheriff_first_day_entry)
    graph.add_node("announce_deaths", announce_deaths)
    graph.add_node("announce_deaths_with_badge_loss", announce_deaths_with_badge_loss)
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
    graph.add_node("exile_last_words", exile_last_words)
    graph.add_node("post_exile_skills", post_exile_skills)
    graph.add_node("resolve_hunter_shot", resolve_hunter_shot)
    graph.add_node("check_victory", check_victory)
    graph.add_node("sheriff_badge_transfer", sheriff_badge_transfer)
    graph.add_node("summarize_positions", summarize_positions)
    graph.add_node("sheriff_endorse", sheriff_endorse)
    graph.add_node("summarize_context", summarize_context)
    graph.add_node("reflection", reflection)
    graph.add_node("finish_game", finish_game)


def _add_all_edges(graph: StateGraph) -> None:
    graph.set_entry_point("setup_game")
    graph.add_edge("setup_game", "assign_roles")
    graph.add_edge("assign_roles", "enter_night")
    graph.add_edge("enter_night", "wolf_discussion")
    graph.add_edge("wolf_discussion", "wolf_consensus")
    graph.add_edge("wolf_consensus", "night_witch")
    graph.add_edge("night_witch", "night_seer")
    graph.add_edge("night_seer", "night_hunter_idiot_status")
    graph.add_edge("night_hunter_idiot_status", "first_night_hybrid_master")
    graph.add_edge("first_night_hybrid_master", "resolve_night_node")
    graph.add_conditional_edges("resolve_night_node", route_after_resolve_night, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "check_victory": "check_victory",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    graph.add_conditional_edges("resolve_hunter_shot", route_after_hunter_shot, {
        "check_victory": "check_victory",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    graph.add_edge("sheriff_first_day_entry", "sheriff_registration")
    graph.add_conditional_edges("sheriff_registration", route_after_sheriff_registration, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_speech": "sheriff_speech",
    })
    graph.add_conditional_edges("sheriff_speech", route_after_sheriff_speech, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_withdraw": "sheriff_withdraw",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("sheriff_withdraw", route_after_sheriff_withdraw, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_vote": "sheriff_vote",
    })
    graph.add_conditional_edges("sheriff_vote", route_after_sheriff_vote, {
        "resolve_self_destruct": "resolve_self_destruct",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_edge("announce_deaths", "night_death_last_words")
    graph.add_edge("announce_deaths_with_badge_loss", "night_death_last_words")
    graph.add_conditional_edges("night_death_last_words", route_after_announce, {
        "free_discussion": "free_discussion",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
    })
    graph.add_conditional_edges("free_discussion", route_self_destruct_check, {
        "resolve_self_destruct": "resolve_self_destruct",
        "continue_discussion": "free_discussion",
        "summarize_positions": "summarize_positions",
    })
    graph.add_conditional_edges("summarize_positions", _route_after_summarize, {
        "sheriff_endorse": "sheriff_endorse",
        "day_vote": "day_vote",
    })
    graph.add_edge("sheriff_endorse", "day_vote")
    graph.add_edge("resolve_self_destruct", "check_victory")
    graph.add_edge("day_vote", "resolve_vote_node")
    graph.add_conditional_edges("resolve_vote_node", route_after_vote, {
        "resolve_exile": "resolve_exile",
        "tie_pk_speech": "tie_pk_speech",
        "check_victory": "check_victory",
    })
    graph.add_edge("tie_pk_speech", "tie_revote")
    graph.add_edge("tie_revote", "day_vote")
    graph.add_edge("resolve_exile", "exile_last_words")
    graph.add_edge("exile_last_words", "post_exile_skills")
    graph.add_conditional_edges("post_exile_skills", route_after_post_exile, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "check_victory": "check_victory",
    })
    graph.add_conditional_edges("check_victory", route_victory, {
        "finish_game": "reflection",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "enter_night": "summarize_context",
    })
    graph.add_conditional_edges("sheriff_badge_transfer", _route_after_badge_transfer, {
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "enter_night": "summarize_context",
    })
    graph.add_edge("summarize_context", "enter_night")
    graph.add_edge("reflection", "finish_game")
    graph.add_edge("finish_game", END)
