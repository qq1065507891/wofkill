# -*- coding: utf-8 -*-
"""
提供狼人夜晚讨论、队伍计划和统一击杀节点。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.wolf_night_nodes import wolf_discussion
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import (
    agent_wolf_consensus,
    agent_wolf_discussion,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _alive_wolves,
    _allocate_decision_identity,
    _build_wolf_team_plan,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _force_wolf_kill,
    _judge_broadcast,
    _planned_wolf_kill,
    _player_display,
    _timer_expired,
)


def _compat(name: str, fallback: Any) -> Any:
    """读取旧 facade 上的 monkeypatch，保持测试和外部补丁路径兼容。"""
    try:
        from werewolf_agent.runtime.nodes import night as night_mod
        return getattr(night_mod, name, fallback)
    except (ImportError, AttributeError):
        return fallback

def _legacy_wolf_consensus(state: RuntimeState) -> dict[str, Any]:

    """Determine wolf night action.



    Wolves may strategically skip a kill (空刀脏人), but consecutive

    no-kill nights are capped to prevent degenerate infinite loops.

    """

    gs: GameState = state["game_state"]

    max_consecutive_no_kill = 2  # After N straight no-kill nights, force a kill



    # Count consecutive no-kill nights so far

    consecutive_no_kill = 0

    for ev in reversed(gs.events):

        if ev.type == "wolf_no_kill_timeout" or ev.type == "wolf_no_kill_declared":

            consecutive_no_kill += 1

        elif ev.type in ("wolf_kill_selected",):

            break



    if _timer_expired(state, "wolf_discussion"):

        if consecutive_no_kill >= max_consecutive_no_kill:

            logger.debug(f"  [狼人决策] 连续{consecutive_no_kill}夜空刀，强制击杀")

            return _force_wolf_kill(gs, "timer_expired_forced_kill")

        logger.debug(f"  [狼人决策] 讨论超时，空刀")

        event = GameEvent(

            type="wolf_no_kill_timeout",

            payload={"night_number": gs.night_number, "reason": "timer_expired"},

        )

        gs = replace(gs, events=gs.events + [event])

        return {"game_state": gs, "wolf_kill_target_id": None}



    # Try agent-driven decision first
    if state.get("agent_registry") and not state.get("wolf_action"):
        wolves = _compat("_alive_wolves", _alive_wolves)(gs)
        decision_identities = {
            wolf_id: _allocate_decision_identity(
                state,
                player_id=wolf_id,
                phase="wolf_consensus",
                task_type="wolf_consensus",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            for wolf_id in wolves
        }
        exposure_collectors = {
            wolf_id: ModuleExposureAuditCollector()
            for wolf_id in wolves
        }
        result = _compat("_dispatch_agent", _dispatch_agent)(
            state,
            _compat("agent_wolf_consensus", agent_wolf_consensus),
            timeout_override=AGENT_TIMEOUTS.wolf_consensus,
            decision_identities=decision_identities,
            exposure_collectors=exposure_collectors,
        )
        if result is not None:
            action = result.get("wolf_action", "kill")
            target = result.get("wolf_kill_target_id")
            audit_events: list[GameEvent] = []
            for wolf_id, action_trace in (result.get("action_traces") or {}).items():
                audit_events.extend(_action_audit_events(
                    state=state,
                    player_id=wolf_id,
                    phase="wolf_consensus",
                    action_trace=action_trace,
                    decision_identity=(result.get("action_decision_identities") or {}).get(wolf_id),
                    exposure_collector=(result.get("action_exposure_collectors") or {}).get(wolf_id),
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            if action == "no_kill":
                if consecutive_no_kill >= max_consecutive_no_kill:
                    logger.debug(f"  [狼人决策] 连续{consecutive_no_kill}夜空刀，强制击杀")
                    return _force_wolf_kill(gs, "consecutive_no_kill_limit")
                logger.debug(f"  [狼人决策] 狼人选择空刀 (原因: {result.get('wolf_action_reason', 'agent decision')})")

                event = GameEvent(

                    type="wolf_no_kill_declared",

                    payload={

                        "night_number": gs.night_number,

                        "reason": result.get("wolf_action_reason", "agent decision"),

                        "action_traces": result.get("action_traces", {}),
                    },
                )
                gs = replace(gs, events=gs.events + audit_events + [event])
                return {"game_state": gs, "wolf_kill_target_id": None}
            if action == "kill" and target:
                target_state = gs.players.get(target)
                if target_state and target_state.alive:
                    logger.debug(f"  [狼人决策] 击杀目标: {_player_display(state, target)} (原因: {result.get('wolf_action_reason', '')})")

                    event = GameEvent(

                        type="wolf_kill_selected",

                        payload={

                            "night_number": gs.night_number,

                            "target_id": target,

                            "action_traces": result.get("action_traces", {}),
                        },
                    )
                    gs = replace(gs, events=gs.events + audit_events + [event])
                    return {"game_state": gs, "wolf_kill_target_id": target}
        else:

            # Agent call timed out entirely

            if consecutive_no_kill >= max_consecutive_no_kill:

                logger.debug(f"  [狼人决策] Agent超时，连续{consecutive_no_kill}夜空刀，强制击杀")

                return _force_wolf_kill(gs, "agent_timeout_forced_kill")

            logger.debug(f"  [狼人决策] Agent调用超时，空刀")

            event = GameEvent(

                type="wolf_no_kill_timeout",

                payload={"night_number": gs.night_number},

            )

            gs = replace(gs, events=gs.events + [event])

            return {"game_state": gs, "wolf_kill_target_id": None}



    # Scripted fallback

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







def wolf_discussion(state: RuntimeState) -> dict[str, Any]:

    """Run multi-round private wolf strategy and produce a team plan."""

    gs: GameState = state["game_state"]

    gs, _ = _judge_broadcast(

        phase="wolf_wake",

        message="狼人请睁眼",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    gs, _ = _judge_broadcast(

        phase="wolf_discussion_start",

        message="狼人开始讨论今晚的行动",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    registry = state.get("agent_registry")



    if not registry:

        gs, _ = _judge_broadcast(

            phase="wolf_discussion_end",

            message="狼人讨论完毕",

            gs=gs, night_number=gs.night_number,

            visibility="moderator_only",

        )

        gs = replace(gs, events=gs.events + [GameEvent(type="wolf_discussion", payload={})])

        return {"game_state": gs}



    wolves = _compat("_alive_wolves", _alive_wolves)(gs)
    events: list[GameEvent] = []

    round_count = 3 if gs.night_number == 1 else 2

    logger.debug(f"  [狼人密谈] 狼人: {[_player_display(state, w) for w in wolves]}，共{round_count}轮")

    discussion_start = time.monotonic()
    _ensure_runtime_audit_state(state)
    for round_number in range(1, round_count + 1):
        # Check total discussion timeout

        elapsed = time.monotonic() - discussion_start

        if elapsed >= AGENT_TIMEOUTS.wolf_discussion_total:

            logger.debug(f"  [狼人密谈] 讨论总超时({elapsed:.0f}s/{AGENT_TIMEOUTS.wolf_discussion_total:.0f}s)，跳过剩余轮次")

            break

        round_state = dict(state)

        round_state["wolf_discussion_round"] = round_number

        for wolf_id in wolves:

            round_state["game_state"] = gs  # Latest gs with accumulated speeches

            decision_identity = _allocate_decision_identity(
                round_state,
                player_id=wolf_id,
                phase=f"wolf_discussion_round_{round_number}",
                task_type="wolf_discussion",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector()
            result = _compat("_dispatch_agent", _dispatch_agent)(
                round_state,
                _compat("agent_wolf_discussion", agent_wolf_discussion),
                wolf_id,
                timeout_override=AGENT_TIMEOUTS.wolf_discussion_per_player,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
            )
            speech_text = result.get("speech_text", "") if result else ""

            logger.debug(

                f"    [第{round_number}轮] {_player_display(state, wolf_id)}(狼人): "

                f"{speech_text if speech_text else '(沉默)'}"

            )

            disc_event = GameEvent(

                type="wolf_discussion",

                payload={

                    "wolf_id": wolf_id,

                    "round": round_number,

                    "night_number": gs.night_number,

                    "text": speech_text,

                    "visibility": "werewolf_team_only",

                },

            )

            # Immediately merge into gs so next wolf sees this speech

            gs = replace(gs, events=gs.events + [disc_event])

            events.append(disc_event)
            if result and result.get("action_trace"):
                trace_events = _action_audit_events(
                    state=round_state,
                    player_id=wolf_id,
                    phase=f"wolf_discussion_round_{round_number}",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                )
                gs = replace(gs, events=gs.events + trace_events)
                events.extend(trace_events)
            else:
                exposure_collector.flush_events()


        # Check if wolves reached consensus after this round — end early if so

        if round_number < round_count:

            from werewolf_agent.runtime.wolf_strategy import (

                should_end_discussion_early,

                summarize_wolf_consensus,

            )

            mid_consensus = summarize_wolf_consensus(

                gs.events, wolves, night_number=gs.night_number

            )

            if should_end_discussion_early(mid_consensus, len(wolves)):

                logger.debug(f"  [狼人密谈] 第{round_number}轮已达成共识，提前结束讨论")

                break



    # Aggregate discussion into consensus plan, fallback to static plan

    # (events already merged incrementally into gs)

    #

    # NOTE: as of wolf-team-plan-llm-structured (2026-06-10), final

    # wolf_team_plan generation moved to the dedicated

    # `wolf_team_plan_node` (called after this node by the graph).

    # That node tries LLM captain decision first and falls back to

    # the legacy regex+static path implemented in

    # `_build_fallback_wolf_team_plan` below. This node no longer

    # emits `wolf_team_plan` events or returns the plan in state.

    gs, _ = _judge_broadcast(

        phase="wolf_discussion_end",

        message="狼人讨论完毕",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    return {"game_state": gs}





def _build_fallback_wolf_team_plan(

    state: RuntimeState,

    wolves: list[str],

) -> dict[str, Any]:

    """Legacy regex-extraction + static dedup fallback for wolf_team_plan.



    Called by `wolf_team_plan_node` when the LLM captain path fails

    (agent unavailable, retry exhausted, schema rejection). Keeps the

    pre-LLM behavior available so the game can continue even if the

    captain's provider is down or returns malformed plans.



    The legacy path's known weakness — regex extractor's keyword set

    misses LLM synonyms like '悍跳位' — is partially mitigated by the

    keyword-补丁 in wolf_strategy.py (T6) but is fundamentally why the

    LLM path exists as the primary route.

    """

    from werewolf_agent.runtime.wolf_strategy import (

        build_wolf_team_plan_from_discussion,

        summarize_wolf_consensus,

    )

    gs: GameState = state["game_state"]

    consensus = summarize_wolf_consensus(

        gs.events, wolves, night_number=gs.night_number

    )

    plan = build_wolf_team_plan_from_discussion(

        gs,

        previous_plan=state.get("wolf_team_plan"),

        consensus=consensus,

    )

    static_plan = _build_wolf_team_plan(

        gs, previous_plan=state.get("wolf_team_plan")

    )

    used_wolves = {

        plan[r] for r in ("fake_seer", "pusher", "hooker", "deep_cover")

        if plan.get(r)

    }

    for key in ("fake_seer", "pusher", "hooker", "deep_cover", "public_story"):

        if not plan.get(key) and static_plan.get(key):

            if key != "public_story" and static_plan[key] in used_wolves:

                continue

            plan[key] = static_plan[key]

            if key != "public_story":

                used_wolves.add(static_plan[key])

    return plan





def wolf_team_plan_node(state: RuntimeState) -> dict[str, Any]:

    """Produce structured WolfTeamPlan via LLM captain, fallback to legacy.



    Replaces the legacy regex-extraction path inside wolf_discussion as

    the primary plan source. Graph wires this node between

    `wolf_discussion` and `wolf_consensus`.



    On success: emits `wolf_team_plan` event with consensus_method="llm".

    On fallback: emits `wolf_team_plan_fallback` audit event with reason,

    then emits `wolf_team_plan` with consensus_method="fallback".

    """

    from werewolf_agent.runtime.agent_adapter import agent_wolf_team_plan



    gs: GameState = state["game_state"]

    wolves = _compat("_alive_wolves", _alive_wolves)(gs)

    if not wolves:

        # No wolves alive — nothing to plan.

        return {"game_state": gs}



    registry = state.get("agent_registry")

    plan: dict[str, Any] | None = None

    fallback_reason: str | None = None



    if registry is None:

        fallback_reason = "no_registry"

    else:

        try:

            plan = agent_wolf_team_plan(

                state, engine=state.get("engine"), registry=registry,

            )

        except Exception as e:  # noqa: BLE001

            logger.debug(

                "[wolf_team_plan_node] agent_wolf_team_plan raised: %s", e

            )

            plan = None

            fallback_reason = f"agent_exception: {e}"



    events: list[GameEvent] = []

    if plan is None:

        # Fallback to legacy regex + static path

        if fallback_reason is None:

            fallback_reason = "llm_failed_or_unavailable"

        plan = _build_fallback_wolf_team_plan(state, wolves)

        plan["consensus_method"] = "fallback"

        plan.setdefault("captain_id", wolves[0] if wolves else None)

        events.append(GameEvent(

            type="wolf_team_plan_fallback",

            payload={

                "night_number": gs.night_number,

                "reason": fallback_reason,

                "visibility": "werewolf_team_only",

            },

        ))

        logger.debug(

            "[wolf_team_plan_node] fallback path used, reason=%s", fallback_reason,

        )



    events.append(GameEvent(

        type="wolf_team_plan",

        payload={**plan, "visibility": "werewolf_team_only"},

    ))

    gs = replace(gs, events=gs.events + events)

    return {"game_state": gs, "wolf_team_plan": plan}





def wolf_consensus(state: RuntimeState) -> dict[str, Any]:

    """Determine wolf night action, preferring the private team plan."""

    gs: GameState = state["game_state"]

    gs, _ = _judge_broadcast(

        phase="wolf_kill_choice",

        message="狼人请统一选择今晚的行动",

        gs=gs, night_number=gs.night_number,

        visibility="moderator_only",

    )

    _ensure_runtime_audit_state(state)
    state = {**state, "game_state": gs}
    planned = _planned_wolf_kill(state)

    if planned is not None and not state.get("wolf_action"):

        result = planned

    else:

        result = _legacy_wolf_consensus(state)

    result_gs = result.get("game_state", gs)

    result_gs, _ = _judge_broadcast(

        phase="wolf_sleep",

        message="狼人请闭眼",

        gs=result_gs,

        night_number=result_gs.night_number,

        visibility="moderator_only",

    )

    return {**result, "game_state": result_gs}
