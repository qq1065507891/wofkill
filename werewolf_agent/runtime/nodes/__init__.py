# -*- coding: utf-8 -*-
"""Game graph nodes — re-export from sub-modules.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    RULESET_PATH,
    _new_engine,
    _stable_seed,
    _player_ids,
    _alive_wolves,
    _alive_non_wolves,
    _force_wolf_kill,
    _find_role,
    _timer_expired,
    _agent_timeout,
    _player_display,
    _call_agent,
    _dispatch_agent,
    _action_trace_event,
    _private_vote_audit_payload,
    _public_vote_reason,
    _with_vote_target_in_trace,
    _judge_broadcast,
    _jb,
    _hitl_checkpoint,
    _ensure_day_incremented,
    _build_wolf_team_plan,
    _first_alive_target,
    _planned_wolf_kill,
    _sheriff_died_this_batch,
    _needs_sheriff_before_deaths,
    _deaths_already_announced,
)

from werewolf_agent.runtime.nodes.night import (
    enter_night,
    _legacy_wolf_consensus,
    night_witch,
    night_seer,
    night_hunter_idiot_status,
    first_night_hybrid_master,
    resolve_night,
    wolf_discussion,
    wolf_team_plan_node,
    wolf_consensus,
)

from werewolf_agent.runtime.nodes.day import (
    announce_deaths,
    announce_deaths_with_badge_loss,
    night_death_last_words,
    free_discussion,
    day_vote,
    _broadcast_vote_details,
    resolve_vote,
    resolve_exile,
    exile_last_words,
    check_victory,
    finish_game,
)

from werewolf_agent.runtime.nodes.sheriff import (
    sheriff_first_day_entry,
    sheriff_registration,
    sheriff_withdraw,
    sheriff_vote,
    sheriff_speech,
    sheriff_endorse,
)

from werewolf_agent.runtime.nodes.skills import (
    resolve_hunter_shot,
    _hunter_shot_target_from_last_words,
    resolve_self_destruct_node,
    tie_pk_speech,
    tie_revote,
    sheriff_badge_transfer,
)

from werewolf_agent.runtime.nodes.sheriff_pk import (
    sheriff_pk_speech,
    sheriff_revote,
)

from werewolf_agent.runtime.nodes.summary import (
    summarize_positions,
    summarize_context,
    reflection,
    _route_after_summarize,
)

__all__ = [
    # _shared
    "RuntimeState",
    "RULESET_PATH",
    "_new_engine",
    "_stable_seed",
    "_player_ids",
    "_alive_wolves",
    "_alive_non_wolves",
    "_force_wolf_kill",
    "_find_role",
    "_timer_expired",
    "_agent_timeout",
    "_player_display",
    "_call_agent",
    "_dispatch_agent",
    "_action_trace_event",
    "_private_vote_audit_payload",
    "_public_vote_reason",
    "_with_vote_target_in_trace",
    "_judge_broadcast",
    "_jb",
    "_hitl_checkpoint",
    "_ensure_day_incremented",
    "_build_wolf_team_plan",
    "_first_alive_target",
    "_planned_wolf_kill",
    "_sheriff_died_this_batch",
    "_needs_sheriff_before_deaths",
    "_deaths_already_announced",
    # night
    "enter_night",
    "_legacy_wolf_consensus",
    "night_witch",
    "night_seer",
    "night_hunter_idiot_status",
    "first_night_hybrid_master",
    "resolve_night",
    "wolf_discussion",
    "wolf_team_plan_node",
    "wolf_consensus",
    # day
    "announce_deaths",
    "announce_deaths_with_badge_loss",
    "night_death_last_words",
    "free_discussion",
    "day_vote",
    "_broadcast_vote_details",
    "resolve_vote",
    "resolve_exile",
    "exile_last_words",
    "check_victory",
    "finish_game",
    # sheriff
    "sheriff_first_day_entry",
    "sheriff_registration",
    "sheriff_withdraw",
    "sheriff_vote",
    "sheriff_speech",
    "sheriff_endorse",
    # summary
    "summarize_positions",
    "summarize_context",
    "reflection",
    "_route_after_summarize",
    # skills
    "resolve_hunter_shot",
    "_hunter_shot_target_from_last_words",
    "resolve_self_destruct_node",
    "tie_pk_speech",
    "tie_revote",
    "sheriff_badge_transfer",
    # sheriff_pk
    "sheriff_pk_speech",
    "sheriff_revote",
]
