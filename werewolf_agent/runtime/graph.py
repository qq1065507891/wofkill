# -*- coding: utf-8 -*-
"""围绕 RuleEngine 编排确定性的 LangGraph 游戏流程与兼容导出。

作者: Project contributors
创建日期: 2025-01-15
修改日期: 2026-07-20
使用示例: 内部模块，无对外接口
Every node calls RuleEngine for rule decisions. No natural language adjudication.
Node function implementations live in ``werewolf_agent.runtime.nodes``; this
module owns graph factories, conditional-edge routing, and compatibility re-exports.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.core.resolution_batches import (
    carrier_matches_resolution_batch,
    same_resolution_batch,
)
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph_registration import (
    add_game_graph_edges as _add_all_edges,
    add_game_graph_nodes as _add_all_nodes,
)
from werewolf_agent.runtime.sheriff_policy import is_all_players_on_sheriff

# -- Re-export everything from nodes so external imports stay unchanged --
from werewolf_agent.runtime.nodes._shared import (  # noqa: F401
    RULESET_PATH,
    RuntimeState,
    logger,
    _action_trace_event,
    _alive_non_wolves,
    _alive_wolves,
    _build_wolf_team_plan,
    _deaths_already_announced,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _ensure_day_incremented,
    _find_role,
    _force_wolf_kill,
    _has_pending_hunter_shot,
    _judge_broadcast,
    _jb,
    _needs_sheriff_before_deaths,
    _new_engine,
    _player_display,
    _player_ids,
    _planned_wolf_kill,
    _private_vote_audit_payload,
    _public_vote_reason,
    _sheriff_died_this_batch,
    _stable_seed,
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
    wolf_team_plan_node,
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
    sheriff_endorse,
    sheriff_first_day_entry,
    sheriff_registration,
    sheriff_speech,
    sheriff_vote,
    sheriff_withdraw,
)

from werewolf_agent.runtime.nodes.sheriff_pk import (  # noqa: F401
    sheriff_pk_speech,
    sheriff_revote,
)

from werewolf_agent.runtime.nodes.skills import (  # noqa: F401
    _hunter_shot_target_from_last_words,
    resolve_hunter_shot,
    resolve_self_destruct_node,
    sheriff_badge_transfer,
    tie_pk_speech,
    tie_revote,
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
    _ensure_runtime_audit_state(state)
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
    return {
        "game_state": gs,
        "engine": engine,
        "action_index_by_game": state["action_index_by_game"],
        "pending_exposure_events_by_trace": state["pending_exposure_events_by_trace"],
    }


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

def _post_hunter_route(gs: GameState) -> str:
    """Shared routing logic for the post-night-resolve / post-hunter-shot
    night branch. Picks the next node based on sheriff interrupt count
    and whether the day needs sheriff election before death announcement.
    """
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    if _needs_sheriff_before_deaths(gs):
        return "sheriff_first_day_entry"
    return "announce_deaths"


def route_after_resolve_night(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if _has_pending_hunter_shot(gs):
        return "resolve_hunter_shot"
    if gs.winning_faction is not None:
        return "reflection"
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    # D1-flow-rewire: D1 N1 first resolve must go to sheriff_first_day_entry
    # BEFORE announcing deaths. V1 design: 天亮 → 警长竞选 → 死讯广播 →
    # 遗言 → 自由讨论. D2+ days skip the election block and go straight to
    # announce_deaths (preserves the prior post-sheriff announce_deaths path
    # via route_after_sheriff_vote).
    if _needs_sheriff_before_deaths(gs):
        return "sheriff_first_day_entry"
    # D1 with count>=2 (badge already torn) → badge_loss
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    return "announce_deaths"


def route_after_hunter_shot(state: RuntimeState) -> str:
    return "check_victory"


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


def route_after_post_exile(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if _has_pending_hunter_shot(gs):
        return "resolve_hunter_shot"
    if gs.winning_faction is not None:
        return "reflection"
    return "exile_last_words"


def route_after_exile_last_words(state: RuntimeState) -> str:
    """遗言完成后优先清理死亡警长的 active 警徽。"""
    gs: GameState = state["game_state"]
    if gs.winning_faction is not None:
        return "reflection"
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    return "summarize_context"


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
    latest_exile_index = -1
    latest_last_words_index = -1
    for index, event in enumerate(gs.events):
        if event.type == "vote_resolved" and event.payload.get("exiled") is not None:
            latest_exile_index = index
        if (
            event.type == "judge_broadcast"
            and event.payload.get("phase") == "exile_last_words"
        ):
            latest_last_words_index = index
    if gs.phase != "night" and latest_exile_index > latest_last_words_index:
        return "exile_last_words"
    current_night_batch = f"night_{gs.night_number}"
    night_hunter_reaction_complete = (
        gs.phase == "night"
        and (
            any(
                event.type == "hunter_shot_declined"
                and same_resolution_batch(
                    event.payload.get("resolution_batch", ""),
                    current_night_batch,
                    left_parse_failed=bool(
                        event.payload.get("resolution_batch_parse_failed", False)
                    ),
                )
                for event in gs.events
            )
            or any(
                death.reason == "hunter_shot"
                and carrier_matches_resolution_batch(death, current_night_batch)
                for death in gs.deaths
            )
        )
    )
    if gs.sheriff_id and gs.sheriff_badge_state == "active":
        sheriff = gs.players.get(gs.sheriff_id)
        if sheriff and not sheriff.alive:
            return "sheriff_badge_transfer"
    if night_hunter_reaction_complete:
        return _post_hunter_route(gs)
    return "enter_night"


def route_after_announce(state: RuntimeState) -> str:
    # D1-flow-rewire: V1 design moves the sheriff election BEFORE
    # announce_deaths / night_death_last_words, so after
    # night_death_last_words we always enter free_discussion. The
    # previous sheriff_first_day_entry branch (commits 2fb56a0 +
    # d156d3d) only existed because the legacy flow did
    # announce_deaths → last_words → sheriff.
    return "free_discussion"


def route_after_sheriff_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    # If first tie, route to PK speech
    if gs.sheriff_tie_count == 1 and gs.sheriff_pk_candidates:
        return "sheriff_pk_speech"
    if not _deaths_already_announced(gs):
        return "announce_deaths"
    return "free_discussion"


def route_after_sheriff_pk_speech(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_revote")


def route_after_sheriff_revote(state: RuntimeState) -> str:
    """After revote, go to next phase (deaths or free_discussion)."""
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
        # No sheriff election possible — badge lost. Sheriff phase ends here.
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


def route_after_self_destruct(state: RuntimeState) -> str:
    """D1-flow-rewire: after a wolf self-destructs, decide whether the
    night deaths still need a public broadcast before continuing.

    - D1 self-destruct during sheriff election: N1 deaths were NOT yet
      announced (sheriff election moved ahead of announce_deaths in the
      rewired flow), so the game must still run announce_deaths →
      night_death_last_words before the day continues. Legacy flow
      (commit 89b865b) avoided this by always running announce_deaths
      before the election, but that contradicts the V1 design that the
      election starts first.
    - D2+ self-destructs happen during free_discussion (handled by
      route_self_destruct_check above), so the deaths are already
      announced and the game can proceed to check_victory.
    """
    gs: GameState = state["game_state"]
    if gs.day_number == 1 and gs.sheriff_id is None and not _deaths_already_announced(gs):
        return "announce_deaths"
    return "check_victory"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_GAME_GRAPH_CACHE: CompiledStateGraph | None = None


def build_game_graph() -> CompiledStateGraph:
    """Build and compile the runtime game graph.

    Result is memoized at module level (test/perf optimization). The
    graph itself is stateless — tests stream events through it without
    mutating graph structure — so sharing one instance across all
    callers is safe. ``build_game_graph_with_checkpoint`` is NOT
    cached (it takes a parameter and is rarely called).
    """
    global _GAME_GRAPH_CACHE
    if _GAME_GRAPH_CACHE is None:
        graph = StateGraph(RuntimeState)
        _add_all_nodes(graph)
        _add_all_edges(graph)
        _GAME_GRAPH_CACHE = graph.compile()
    return _GAME_GRAPH_CACHE


def build_game_graph_with_checkpoint(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build the graph with an external checkpointer for pause/resume."""
    graph = StateGraph(RuntimeState)
    _add_all_nodes(graph)
    _add_all_edges(graph)
    return graph.compile(checkpointer=checkpointer)
