# -*- coding: utf-8 -*-
"""
集中注册运行时 LangGraph 的节点和边。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-13

使用示例:
    >>> from langgraph.graph import StateGraph
    >>> from werewolf_agent.runtime.graph_registration import add_game_graph_nodes
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from werewolf_agent.runtime.nodes.day import (
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
from werewolf_agent.runtime.nodes.night import (
    enter_night,
    first_night_hybrid_master,
    night_hunter_idiot_status,
    night_seer,
    night_witch,
    resolve_night,
    wolf_consensus,
    wolf_discussion,
    wolf_team_plan_node,
)
from werewolf_agent.runtime.nodes.sheriff import (
    sheriff_endorse,
    sheriff_first_day_entry,
    sheriff_registration,
    sheriff_speech,
    sheriff_vote,
    sheriff_withdraw,
)
from werewolf_agent.runtime.nodes.sheriff_pk import (
    sheriff_pk_speech,
    sheriff_revote,
)
from werewolf_agent.runtime.nodes.skills import (
    resolve_hunter_shot,
    resolve_self_destruct_node,
    sheriff_badge_transfer,
    tie_pk_speech,
    tie_revote,
)
from werewolf_agent.runtime.nodes.summary import (
    reflection,
    summarize_context,
    summarize_positions,
)


def _graph_facade():
    """延迟获取 graph facade，避免导入期循环依赖。"""
    from werewolf_agent.runtime import graph as graph_mod

    return graph_mod


def add_game_graph_nodes(graph: StateGraph) -> None:
    """注册运行时图的全部节点。"""
    graph_mod = _graph_facade()
    graph.add_node("setup_game", graph_mod.setup_game)
    graph.add_node("assign_roles", graph_mod.assign_roles)
    graph.add_node("enter_night", enter_night)
    graph.add_node("wolf_discussion", wolf_discussion)
    graph.add_node("wolf_team_plan", wolf_team_plan_node)
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
    graph.add_node("sheriff_pk_speech", sheriff_pk_speech)
    graph.add_node("sheriff_revote", sheriff_revote)
    graph.add_node("free_discussion", free_discussion)
    graph.add_node("resolve_self_destruct", resolve_self_destruct_node)
    graph.add_node("day_vote", day_vote)
    graph.add_node("resolve_vote_node", resolve_vote)
    graph.add_node("tie_pk_speech", tie_pk_speech)
    graph.add_node("tie_revote", tie_revote)
    graph.add_node("resolve_exile", resolve_exile)
    graph.add_node("exile_last_words", exile_last_words)
    graph.add_node("resolve_hunter_shot", resolve_hunter_shot)
    graph.add_node("check_victory", check_victory)
    graph.add_node("sheriff_badge_transfer", sheriff_badge_transfer)
    graph.add_node("summarize_positions", summarize_positions)
    graph.add_node("sheriff_endorse", sheriff_endorse)
    graph.add_node("summarize_context", summarize_context)
    graph.add_node("reflection", reflection)
    graph.add_node("finish_game", finish_game)


def add_game_graph_edges(graph: StateGraph) -> None:
    """注册运行时图的全部固定边和条件边。"""
    graph_mod = _graph_facade()
    graph.set_entry_point("setup_game")
    graph.add_edge("setup_game", "assign_roles")
    graph.add_edge("assign_roles", "enter_night")
    graph.add_edge("enter_night", "wolf_discussion")
    graph.add_edge("wolf_discussion", "wolf_team_plan")
    graph.add_edge("wolf_team_plan", "wolf_consensus")
    graph.add_edge("wolf_consensus", "night_witch")
    graph.add_edge("night_witch", "night_seer")
    graph.add_edge("night_seer", "night_hunter_idiot_status")
    graph.add_edge("night_hunter_idiot_status", "first_night_hybrid_master")
    graph.add_edge("first_night_hybrid_master", "resolve_night_node")
    graph.add_conditional_edges("resolve_night_node", graph_mod.route_after_resolve_night, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "reflection": "reflection",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    graph.add_conditional_edges("resolve_hunter_shot", graph_mod.route_after_hunter_shot, {
        "check_victory": "check_victory",
        "reflection": "reflection",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    graph.add_edge("sheriff_first_day_entry", "sheriff_registration")
    graph.add_conditional_edges("sheriff_registration", graph_mod.route_after_sheriff_registration, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_speech": "sheriff_speech",
    })
    graph.add_conditional_edges("sheriff_speech", graph_mod.route_after_sheriff_speech, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_withdraw": "sheriff_withdraw",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("sheriff_withdraw", graph_mod.route_after_sheriff_withdraw, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_vote": "sheriff_vote",
    })
    graph.add_conditional_edges("sheriff_vote", graph_mod.route_after_sheriff_vote, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_pk_speech": "sheriff_pk_speech",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("sheriff_pk_speech", graph_mod.route_after_sheriff_pk_speech, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_revote": "sheriff_revote",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("sheriff_revote", graph_mod.route_after_sheriff_revote, {
        "resolve_self_destruct": "resolve_self_destruct",
        "announce_deaths": "announce_deaths",
        "free_discussion": "free_discussion",
    })
    graph.add_edge("announce_deaths", "night_death_last_words")
    graph.add_edge("announce_deaths_with_badge_loss", "night_death_last_words")
    graph.add_conditional_edges("night_death_last_words", graph_mod.route_after_announce, {
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("free_discussion", graph_mod.route_self_destruct_check, {
        "resolve_self_destruct": "resolve_self_destruct",
        "continue_discussion": "free_discussion",
        "summarize_positions": "summarize_positions",
    })
    graph.add_conditional_edges("summarize_positions", graph_mod._route_after_summarize, {
        "sheriff_endorse": "sheriff_endorse",
        "day_vote": "day_vote",
    })
    graph.add_edge("sheriff_endorse", "day_vote")
    graph.add_conditional_edges("resolve_self_destruct", graph_mod.route_after_self_destruct, {
        "announce_deaths": "announce_deaths",
        "check_victory": "check_victory",
    })
    graph.add_edge("day_vote", "resolve_vote_node")
    graph.add_conditional_edges("resolve_vote_node", graph_mod.route_after_vote, {
        "resolve_exile": "resolve_exile",
        "tie_pk_speech": "tie_pk_speech",
        "check_victory": "check_victory",
    })
    graph.add_edge("tie_pk_speech", "tie_revote")
    graph.add_edge("tie_revote", "day_vote")
    graph.add_conditional_edges("resolve_exile", graph_mod.route_after_post_exile, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "reflection": "reflection",
        "exile_last_words": "exile_last_words",
    })
    graph.add_conditional_edges("exile_last_words", graph_mod.route_after_exile_last_words, {
        "reflection": "reflection",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "summarize_context": "summarize_context",
    })
    graph.add_conditional_edges("check_victory", graph_mod.route_victory, {
        "finish_game": "reflection",
        "exile_last_words": "exile_last_words",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
        "enter_night": "summarize_context",
    })
    graph.add_conditional_edges("sheriff_badge_transfer", graph_mod._route_after_badge_transfer, {
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "enter_night": "summarize_context",
    })
    graph.add_edge("summarize_context", "enter_night")
    graph.add_edge("reflection", "finish_game")
    graph.add_edge("finish_game", END)
