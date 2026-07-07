# -*- coding: utf-8 -*-
"""
提供狼人夜晚统一击杀和旧共识 fallback 节点。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.runtime.nodes.wolf_consensus import wolf_consensus
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import agent_wolf_consensus
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _alive_wolves,
    _allocate_decision_identity,
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
