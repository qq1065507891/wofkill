"""Night phase node functions."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState, VictoryResult
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_night_seer, agent_night_witch, agent_hybrid_choose_master,
    agent_wolf_consensus, agent_wolf_discussion,
)
from werewolf_agent.runtime.nodes._shared import (
    RULESET_PATH,
    logger,
    RuntimeState,
    _alive_non_wolves,
    _alive_wolves,
    _call_agent,
    _dispatch_agent,
    _find_role,
    _judge_broadcast,
    _jb,
    _hitl_checkpoint,
    _new_engine,
    _player_display,
    _player_ids,
    _stable_seed,
    _timer_expired,
    _agent_timeout,
    _action_trace_event,
    AGENT_TIMEOUTS,
    timed_call,
    _force_wolf_kill,
    _planned_wolf_kill,
    _build_wolf_team_plan,
)
from werewolf_agent.runtime.timeline import phase_label

def enter_night(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    n = gs.night_number + 1
    label = phase_label("night", n)
    gs, _ = _judge_broadcast(phase="enter_night", message=f"{label}：天黑请闭眼", gs=gs, night_number=n)
    gs = replace(gs, phase="night", night_number=n,
                 events=gs.events + [GameEvent(type="enter_night", payload={"night": n})])
    alive = [pid for pid, p in gs.players.items() if p.alive]
    logger.debug(f"\n{'='*60}")
    logger.debug(f"  【{label}】天黑请闭眼 (存活: {len(alive)}人)")
    logger.debug(f"{'='*60}")
    _hitl_checkpoint(state, "enter_night", "after")
    return {"game_state": gs}


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
        result = _dispatch_agent(
            state,
            agent_wolf_consensus,
            timeout_override=AGENT_TIMEOUTS.wolf_consensus,
        )
        if result is not None:
            action = result.get("wolf_action", "kill")
            target = result.get("wolf_kill_target_id")
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
                gs = replace(gs, events=gs.events + [event])
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
                    gs = replace(gs, events=gs.events + [event])
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


def night_witch(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    if _find_role(gs, "witch") is None:
        return {"game_state": gs, "use_antidote": False, "poison_target_id": None}
    gs, _ = _jb(
        state,
        phase="witch_wake",
        message="女巫请睁眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
        judge_method="skill_guide",
        extra_payload={
            "role": "witch",
            "player_id": _find_role(gs, "witch") or "",
            "player_name": _player_display(state, _find_role(gs, "witch") or ""),
            "available_actions": ["use_antidote", "use_poison", "no_action"],
        },
    )
    wolf_target = state.get("wolf_kill_target_id")
    if wolf_target:
        gs, _ = _jb(
            state,
            phase="witch_kill_info",
            message=f"今晚{_player_display(state, wolf_target)}被狼人杀害了",
            gs=gs, night_number=gs.night_number,
            visibility="witch_private",
            extra_payload={
                "wolf_kill_target_id": wolf_target,
                "role": "witch",
                "player_id": _find_role(gs, "witch") or "",
                "player_name": _player_display(state, _find_role(gs, "witch") or ""),
                "available_actions": ["use_antidote", "use_poison", "no_action"],
                "context_hints": {"wolf_kill_target": _player_display(state, wolf_target)},
            },
            judge_method="skill_guide",
        )
    gs, _ = _jb(
        state,
        phase="witch_choose",
        message="女巫请选择是否使用解药或毒药",
        gs=gs, night_number=gs.night_number,
        visibility="witch_private",
        judge_method="skill_guide",
        extra_payload={
            "role": "witch",
            "player_id": _find_role(gs, "witch") or "",
            "player_name": _player_display(state, _find_role(gs, "witch") or ""),
            "available_actions": ["use_antidote", "use_poison", "no_action"],
        },
    )
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    result = _dispatch_agent(
        state,
        agent_night_witch,
        timeout_override=AGENT_TIMEOUTS.witch_action,
    )
    if result is not None:
        use_antidote = result.get("use_antidote", False)
        poison_target_id = result.get("poison_target_id")
        action_taken = "no_action"
        if use_antidote:
            action_taken = "use_antidote"
        elif poison_target_id:
            action_taken = "use_poison"
        wolf_target = state.get("wolf_kill_target_id")
        if use_antidote:
            logger.debug(f"  [女巫] 使用解药救了 {_player_display(state, wolf_target)}")
        if poison_target_id:
            logger.debug(f"  [女巫] 使用毒药毒了 {_player_display(state, poison_target_id)}")
        if not use_antidote and not poison_target_id:
            logger.debug(f"  [女巫] 不使用药水 (解药{'已用' if gs.antidote_used else '可用'}, 毒药{'已用' if gs.poison_used else '可用'})")
        audit = GameEvent(
            type="witch_decision_audit",
            payload={
                "night_number": gs.night_number,
                "wolf_kill_target_id": state.get("wolf_kill_target_id"),
                "action_taken": action_taken,
                "poison_target_id": poison_target_id,
                "reason": "agent_decision",
                "visibility": "witch_private",
                "action_trace": result.get("witch_action_trace"),
            },
        )
        gs = replace(gs, events=gs.events + [audit])
        gs, _ = _judge_broadcast(
            phase="witch_sleep",
            message="女巫请闭眼",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        return {"game_state": gs, **result}

    # Scripted fallback
    gs, _ = _judge_broadcast(
        phase="witch_sleep",
        message="女巫请闭眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    return {"use_antidote": state.get("use_antidote", False),
            "poison_target_id": state.get("poison_target_id"),
            "game_state": gs}


def night_seer(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    if _find_role(gs, "seer") is None:
        return {"game_state": gs, "seer_target_id": None}
    gs, _ = _jb(
        state,
        phase="seer_wake",
        message="预言家请睁眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
        judge_method="skill_guide",
        extra_payload={
            "role": "seer",
            "player_id": _find_role(gs, "seer") or "",
            "player_name": _player_display(state, _find_role(gs, "seer") or ""),
            "available_actions": ["check_alignment"],
        },
    )
    gs, _ = _jb(
        state,
        phase="seer_choose",
        message="预言家请选择你要查验的玩家",
        gs=gs, night_number=gs.night_number,
        visibility="seer_private",
        judge_method="skill_guide",
        extra_payload={
            "role": "seer",
            "player_id": _find_role(gs, "seer") or "",
            "player_name": _player_display(state, _find_role(gs, "seer") or ""),
            "available_actions": ["check_alignment"],
        },
    )
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    result = _dispatch_agent(
        state,
        agent_night_seer,
        timeout_override=AGENT_TIMEOUTS.seer_check,
    )
    if result is not None:
        target = result.get("seer_target_id")
        if target:
            logger.debug(f"  [预言家] 查验目标: {_player_display(state, target)}")
        return {"game_state": gs, **result}

    # Scripted fallback
    return {"seer_target_id": state.get("seer_target_id"), "game_state": gs}


def night_hunter_idiot_status(state: RuntimeState) -> dict[str, Any]:
    """First night only: confirm hunter and idiot are alive for moderator audit.
    Produces no public output; event is moderator/private visibility only."""
    gs: GameState = state["game_state"]
    hunter_id = _find_role(gs, "hunter")
    idiot_id = _find_role(gs, "idiot")
    if hunter_id:
        gs, _ = _judge_broadcast(
            phase="hunter_status",
            message=f"猎人{_player_display(state, hunter_id)}请确认开枪状态",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        logger.debug(f"  [法官] 猎人{_player_display(state, hunter_id)}请确认开枪状态")
    if idiot_id and gs.night_number == 1:
        gs, _ = _judge_broadcast(
            phase="idiot_status",
            message=f"白痴{_player_display(state, idiot_id)}请确认身份",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        logger.debug(f"  [法官] 白痴{_player_display(state, idiot_id)}请确认身份")
    if gs.night_number != 1:
        return {}
    event = GameEvent(
        type="hunter_idiot_status_confirmed",
        payload={
            "night_number": 1,
            "hunter_id": hunter_id,
            "idiot_id": idiot_id,
            "visibility": "moderator_only",
        },
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs}


def first_night_hybrid_master(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    if gs.night_number != 1 or gs.hybrid_master_id is not None:
        return {}
    hybrid_id = _find_role(gs, "hybrid")
    if hybrid_id is None:
        return {}

    gs, _ = _judge_broadcast(
        phase="hybrid_wake",
        message=f"混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    gs, _ = _judge_broadcast(
        phase="hybrid_choose",
        message="混血儿请选择你的主人",
        gs=gs, night_number=gs.night_number,
        visibility="hybrid_private",
    )
    logger.debug(f"  [法官] 混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人")

    master_target = state.get("hybrid_master_target_id")

    # Agent-driven: ask hybrid player to choose master
    if master_target is None:
        result = _dispatch_agent(
            state,
            agent_hybrid_choose_master,
            hybrid_id,
            timeout_override=AGENT_TIMEOUTS.seer_check,
        )
        if result and result.get("master_target_id"):
            master_target = result["master_target_id"]

    # Fallback: random selection
    if master_target is None:
        import random
        candidates = [pid for pid in _player_ids(gs) if pid != hybrid_id]
        rng = random.Random(_stable_seed(gs.game_id, "hybrid_master"))
        master_target = rng.choice(candidates) if candidates else None

    if master_target is None:
        return {}
    gs, event = engine.choose_master(gs, hybrid_id=hybrid_id, master_id=master_target)
    gs = replace(gs, events=gs.events + [event])
    gs, _ = _judge_broadcast(
        phase="hybrid_sleep",
        message="混血儿请闭眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    master_role = gs.players[master_target].role if master_target in gs.players else "?"
    logger.debug(f"  [混血儿] {_player_display(state, hybrid_id)} 选择了 {_player_display(state, master_target)}({master_role}) 作为主人")
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
        seer_target_id=state.get("seer_target_id"),
    )
    seer_trace = state.get("seer_action_trace")
    if seer_trace:
        events = [
            replace(event, payload={**event.payload, "action_trace": seer_trace})
            if event.type == "seer_check" else event
            for event in events
        ]
    if events:
        gs = replace(gs, events=gs.events + events)
    # Log night resolution events
    seer_woke = any(
        event.type == "judge_broadcast" and event.payload.get("phase") == "seer_wake"
        for event in gs.events
    )
    for ev in events:
        if ev.type == "wolf_kill":
            target = ev.payload.get("player_id", "?")
            saved = ev.payload.get("saved_by_antidote", False)
            if saved:
                logger.debug(f"  [夜晚结算] {_player_display(state, target)} 被狼人袭击，但被女巫救活")
            else:
                logger.debug(f"  [夜晚结算] {_player_display(state, target)} 被狼人袭击身亡")
        elif ev.type == "poison_death":
            logger.debug(f"  [夜晚结算] {_player_display(state, ev.payload.get('player_id', '?'))} 被女巫毒杀")
        elif ev.type == "seer_check":
            target = ev.payload.get("target_id", "?")
            alignment = ev.payload.get("alignment", "?")
            gs, _ = _judge_broadcast(
                phase="seer_result",
                message=f"他的身份是{'好人' if alignment == 'good' else '狼人'}",
                gs=gs,
                night_number=gs.night_number,
                visibility="seer_private",
            )
            logger.debug(f"  [夜晚结算] 预言家查验 {_player_display(state, target)}: {'好人' if alignment == 'good' else '狼人'}")
        elif ev.type == "no_death":
            logger.debug(f"  [夜晚结算] 平安夜，无人死亡")
    if seer_woke:
        gs, _ = _judge_broadcast(
            phase="seer_sleep",
            message="预言家请闭眼",
            gs=gs,
            night_number=gs.night_number,
            visibility="moderator_only",
        )
    return {"game_state": gs}


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

    engine: RuleEngine = state["engine"]
    wolves = _alive_wolves(gs)
    events: list[GameEvent] = []
    round_count = 3 if gs.night_number == 1 else 2
    logger.debug(f"  [狼人密谈] 狼人: {[_player_display(state, w) for w in wolves]}，共{round_count}轮")
    discussion_start = time.monotonic()
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
            result = _dispatch_agent(
                round_state,
                agent_wolf_discussion,
                wolf_id,
                timeout_override=AGENT_TIMEOUTS.wolf_discussion_per_player,
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
                trace_event = _action_trace_event(
                    player_id=wolf_id,
                    phase=f"wolf_discussion_round_{round_number}",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                )
                gs = replace(gs, events=gs.events + [trace_event])
                events.append(trace_event)

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
    from werewolf_agent.runtime.wolf_strategy import (
        build_wolf_team_plan_from_discussion,
        summarize_wolf_consensus,
    )
    consensus = summarize_wolf_consensus(gs.events, wolves, night_number=gs.night_number)
    plan = build_wolf_team_plan_from_discussion(
        gs,
        previous_plan=state.get("wolf_team_plan"),
        consensus=consensus,
    )

    # Fallback to static plan when consensus lacks critical fields (dedup)
    static_plan = _build_wolf_team_plan(gs, previous_plan=state.get("wolf_team_plan"))
    used_wolves = {plan[r] for r in ("fake_seer", "pusher", "hooker", "deep_cover") if plan.get(r)}
    for key in ("fake_seer", "pusher", "hooker", "deep_cover", "public_story"):
        if not plan.get(key) and static_plan.get(key):
            if key != "public_story" and static_plan[key] in used_wolves:
                continue
            plan[key] = static_plan[key]
            if key != "public_story":
                used_wolves.add(static_plan[key])

    # Log consensus summary
    primary = plan.get("night_kill_primary")
    backup = plan.get("night_kill_backup")
    agreement = consensus.get("agreement_count", 0)
    total = consensus.get("total_wolves", len(wolves))
    if primary:
        logger.debug(f"  [狼队共识] 主目标: {_player_display(state, primary)}, 备选: {_player_display(state, backup) if backup else '无'}, "
              f"共识度: {agreement}/{total}")
    else:
        logger.debug(f"  [狼队共识] 未达成击杀共识 ({agreement}/{total} 同意)")

    events.append(GameEvent(
        type="wolf_team_plan",
        payload={**plan, "visibility": "werewolf_team_only"},
    ))
    gs = replace(gs, events=gs.events + events[-1:])  # Add plan event
    gs, _ = _judge_broadcast(
        phase="wolf_discussion_end",
        message="狼人讨论完毕",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
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
    state = {**state, "game_state": gs}
    _ = AGENT_TIMEOUTS.wolf_consensus  # referenced for timeout contract, wired in future
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

