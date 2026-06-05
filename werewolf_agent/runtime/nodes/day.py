"""Day phase node functions."""

from __future__ import annotations

import re
import uuid
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState, VictoryResult
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_day_speech, agent_day_vote, agent_exile_last_words,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_trace_event,
    _agent_timeout,
    _call_agent,
    _dispatch_agent,
    _judge_broadcast,
    _jb,
    _hitl_checkpoint,
    _ensure_day_incremented,
    _player_display,
    _public_vote_reason,
    _with_vote_target_in_trace,
    _timer_expired,
)
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.agent_adapter import agent_sheriff_pick_speech_order
from werewolf_agent.runtime.timeline import phase_label

_REASON_LABELS = {
    "wolf_kill": "狼杀",
    "witch_poison": "毒杀",
    "hunter_shot": "猎人开枪",
    "exile": "放逐",
    "self_destruct": "自爆",
    "unknown": "原因不明",
}


def _death_reason_label(reason: str) -> str:
    return _REASON_LABELS.get(reason, reason)


def announce_deaths(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    was_night = gs.phase != "day"
    gs, d = _ensure_day_incremented(state, gs)
    label = phase_label("day", d)

    # Announce last night's deaths
    night_deaths = [
        death for death in gs.deaths
        if death.timing == "night" and death.resolution_batch == f"night_{gs.night_number}"
    ]
    if night_deaths:
        dead_desc = "、".join(
            f"{_player_display(state, d.player_id)}({_death_reason_label(d.reason)})"
            for d in night_deaths
        )
        gs, _ = _judge_broadcast(
            phase="death_announce",
            message=f"昨夜死亡: {dead_desc}",
            gs=gs, day_number=d,
            visibility="public",
        )
    else:
        gs, _ = _judge_broadcast(
            phase="death_announce",
            message="昨夜是平安夜，无人死亡",
            gs=gs, day_number=d,
            visibility="public",
        )

    alive = [pid for pid, p in gs.players.items() if p.alive]
    if was_night:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"  【{label}】天亮了 (存活: {len(alive)}人: {alive})")
    if night_deaths:
        for death in night_deaths:
            logger.debug(f"  [死讯] {_player_display(state, death.player_id)} 死亡 (原因: {death.reason})")
    else:
        logger.debug(f"  [死讯] 平安夜，无人死亡")
    logger.debug(f"{'='*60}")
    _hitl_checkpoint(state, "announce_deaths", "after")
    return {"game_state": gs,
            "revote": False, "speech_index": 0,
            "speech_order": [],
            "current_speaker_id": None}


def announce_deaths_with_badge_loss(state: RuntimeState) -> dict[str, Any]:
    """Announce deaths AND declare badge permanently lost (after 2 sheriff interruptions)."""
    result = announce_deaths(state)
    gs: GameState = result["game_state"]
    gs, _ = _judge_broadcast(
        phase="badge_permanently_lost",
        message="本局警徽因竞选两度中断而永久流失，本局不再有警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    gs = replace(gs, sheriff_badge_state="torn",
                 events=gs.events + [GameEvent(
                     type="badge_permanently_lost",
                     payload={"reason": "sheriff_election_interrupted_twice"},
                 )])
    logger.debug(f"  [警徽] 本局警徽永久流失")
    result["game_state"] = gs
    return result


def night_death_last_words(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    registry = state.get("agent_registry")
    batch = f"night_{gs.night_number}"
    eligible = []
    for death in gs.deaths:
        if death.resolution_batch != batch:
            continue
        can_leave = death.can_leave_last_words
        if can_leave is None:
            can_leave = engine.can_leave_last_words(
            death_reason=death.reason, timing=death.timing, night_number=gs.night_number
            )
        if death.timing == "night" and can_leave:
            eligible.append(death.player_id)
    names = "、".join(_player_display(state, pid) for pid in eligible)
    message = f"请昨夜死亡玩家发表遗言: {names}" if eligible else "昨夜死亡玩家无遗言，遗言环节结束"
    gs, _ = _judge_broadcast(
        phase="night_death_last_words",
        message=message,
        gs=gs,
        day_number=gs.day_number,
        visibility="public",
        extra_payload={"players": eligible},
    )
    gs = replace(gs, events=gs.events + [GameEvent(
        type="night_death_last_words", payload={"players": eligible}
    )])

    # Let each eligible dead player actually speak their last words via agent
    if registry and eligible:
        for pid in eligible:
            call_state = {**state, "game_state": gs}
            result = _dispatch_agent(
                call_state,
                agent_exile_last_words,
                pid,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_text = result.get("speech_text", "") if result else ""
            logger.debug(f"  [夜死遗言] {_player_display(state, pid)}: {speech_text if speech_text else '(无遗言)'}")
            gs = replace(gs, events=gs.events + [GameEvent(
                type="night_death_last_words",
                payload={"speaker": pid, "day_number": gs.day_number, "text": speech_text},
            )])

    return {"game_state": gs}


# -- Day discussion & vote --

def free_discussion(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    speaker_id = state.get("current_speaker_id")

    # Announce discussion start on first entry (speech_index == 0)
    if speech_index == 0 and not speaker_id:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message=f"请警长{_player_display(state, gs.sheriff_id)}指定发言顺序，开始自由讨论",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )
        else:
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message="本局无警长，由法官随机指定发言顺序，开始自由讨论",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )

    # Auto-populate speech order based on sheriff status
    if not speech_order:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            # Sheriff agent picks first speaker; fallback to static order
            agent_order = _dispatch_agent(
                state,
                agent_sheriff_pick_speech_order,
                gs.sheriff_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_order = agent_order or choose_sheriff_led_speech_order(gs, gs.sheriff_id)
        else:
            speech_order = choose_no_sheriff_speech_order(gs)

    # Filter out dead players from any pre-existing or auto-generated speech_order
    speech_order = [pid for pid in speech_order if pid not in gs.players or gs.players[pid].alive]

    if speech_index == 0 and not speaker_id:
        order_names = "、".join(_player_display(state, pid) for pid in speech_order)
        gs, _ = _judge_broadcast(
            phase="speech_order",
            message=f"本轮发言顺序: {order_names}",
            gs=gs,
            day_number=gs.day_number,
            visibility="public",
            extra_payload={"speech_order": speech_order},
        )

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
            "speech_order": speech_order,
        }

    timed_out = state.get("speech_timed_out", False) or (
        speaker_id is not None and _timer_expired(state, f"speech:{speaker_id}")
    )

    if speaker_id and timed_out:
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
        # Judge announces current speaker
        gs, _ = _judge_broadcast(
            phase="speaker_turn",
            message=f"请{_player_display(state, speaker_id)}发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        speech_text = state.get("speech_text", "")
        action_trace = None
        if not speech_text:
            result = _dispatch_agent(
                state,
                agent_day_speech,
                speaker_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            if result is not None:
                if result.get("self_destruct"):
                    return {"game_state": gs, "self_destruct_wolf_id": speaker_id}
                speech_text = result.get("speech_text", "")
                action_trace = result.get("action_trace")
        player_role = gs.players[speaker_id].role if speaker_id in gs.players else "?"
        logger.debug(f"  [{_player_display(state, speaker_id)}({player_role})]: {speech_text if speech_text else '(未发言)'}")
        if not speech_text.strip():
            # Skip empty speeches — no event added to timeline
            advanced = advance_speaker()
            if advanced["current_speaker_id"] is None:
                gs, _ = _judge_broadcast(
                    phase="discussion_end",
                    message="所有玩家发言完毕，自由讨论结束",
                    gs=gs,
                    day_number=gs.day_number,
                    visibility="public",
                )
            return {"game_state": gs, **advanced}
        payload = {
            "speaker": speaker_id,
            "day_number": gs.day_number,
            "text": speech_text,
        }
        events = [GameEvent(type="speech", payload=payload)]
        if action_trace:
            events.append(_action_trace_event(
                player_id=speaker_id,
                phase="speech",
                action_trace=action_trace,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        gs = replace(gs, events=gs.events + events)
        advanced = advance_speaker()
        if advanced["current_speaker_id"] is None:
            gs, _ = _judge_broadcast(
                phase="discussion_end",
                message="所有玩家发言完毕，自由讨论结束",
                gs=gs,
                day_number=gs.day_number,
                visibility="public",
            )
        return {"game_state": gs, **advanced}
    gs = replace(gs, events=gs.events + [GameEvent(type="free_discussion", payload={})])
    return {"game_state": gs}


def day_vote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]

    # Sheriff endorse already announced vote_start when sheriff exists
    _already_announced = (
        gs.sheriff_id
        and gs.sheriff_badge_state == "active"
        and gs.players.get(gs.sheriff_id)
        and gs.players[gs.sheriff_id].alive
    )
    if not _already_announced:
        gs, _ = _jb(
            state,
            phase="vote_start",
            message="讨论结束，现在开始投票。所有人同时投票，投票时不能发言。",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )

    gs, _ = _jb(
        state,
        phase="vote_collect",
        message="请所有仍在场玩家同时投票",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    same_vote_window = (
        state.get("exile_vote_day") == gs.day_number
        and state.get("exile_vote_revote") == state.get("revote", False)
    )
    existing_votes = state.get("exile_votes", {}) if same_vote_window else {}
    registry = state.get("agent_registry")
    is_revote = state.get("revote", False)
    if is_revote:
        logger.debug(f"\n  --- PK重新投票 ---")
    else:
        logger.debug(f"\n  --- 投票开始 ---")
    votes: dict[str, str] = {}
    vote_traces: dict[str, Any] = {}
    has_agents = False
    # Count eligible voters for per-voter calling
    eligible_voter_ids = [pid for pid, p in gs.players.items() if p.alive and p.vote_enabled]
    total_voters = len(eligible_voter_ids)
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    if not existing_votes:
        for idx, pid in enumerate(eligible_voter_ids, start=1):
            player = gs.players[pid]
            # Per-voter judge broadcast (唱票)
            voter_name = _player_display(state, pid)
            gs, _ = _jb(
                state,
                phase="vote_calling",
                message=f"请{voter_name}投票，第{idx}/{total_voters}位",
                gs=gs, day_number=gs.day_number,
                visibility="public",
                judge_method="vote_calling",
                extra_payload={
                    "voter_id": pid,
                    "voter_name": voter_name,
                    "position": idx,
                    "total": total_voters,
                    "sheriff_weight": 1.5 if pid == sheriff_id else 1.0,
                },
            )
            result = _dispatch_agent(
                state,
                agent_day_vote,
                pid,
                timeout_override=AGENT_TIMEOUTS.day_vote,
            )
            if result is not None:
                has_agents = True
                if result.get("vote_target"):
                    votes[pid] = result["vote_target"]
                    logger.debug(f"    {_player_display(state, pid)} → {_player_display(state, result['vote_target'])}")
                else:
                    logger.debug(f"    {_player_display(state, pid)} 弃票")
                if result.get("action_trace"):
                    vote_traces[pid] = result["action_trace"]
            else:
                has_agents = True
                logger.warning(f"    {_player_display(state, pid)} 投票超时（视为弃票）")

        if has_agents:
            gs = _broadcast_vote_details(state, gs, votes)
            return {
                "game_state": gs,
                "exile_votes": votes,
                "vote_action_traces": vote_traces,
                "exile_vote_day": gs.day_number,
                "exile_vote_revote": state.get("revote", False),
                "revote": state.get("revote", False),
            }
    if existing_votes:
        gs = _broadcast_vote_details(state, gs, existing_votes)
    return {
        "game_state": gs,
        "exile_votes": existing_votes,
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": state.get("revote", False),
        "revote": state.get("revote", False),
    }


def _broadcast_vote_details(
    state: RuntimeState,
    gs: GameState,
    votes: dict[str, str],
) -> GameState:
    gs, _ = _jb(
        state,
        phase="vote_end",
        message="投票结束，开始统计票型",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    # Build vote tally for structured judge announcement
    tally: dict[str, float] = {}
    vote_lines = []
    for voter_id, target_id in votes.items():
        weight = 1.5 if voter_id == sheriff_id else 1.0
        tally[target_id] = tally.get(target_id, 0.0) + weight
        weight_label = " (警长1.5票)" if voter_id == sheriff_id else ""
        vote_lines.append(
            f"{_player_display(state, voter_id)}{weight_label} 投票给 {_player_display(state, target_id)}"
        )
    message = "投票结果："
    if vote_lines:
        message += "\n" + "\n".join(vote_lines)
    else:
        message += "无有效票"
    player_names = {pid: _player_display(state, pid) for pid in gs.players}
    gs, _ = _jb(
        state,
        phase="vote_result",
        message=message,
        gs=gs, day_number=gs.day_number,
        visibility="public",
        judge_method="vote_tally",
        extra_payload={
            "tally": tally,
            "player_names": player_names,
            "sheriff_id": sheriff_id,
            "sheriff_weight": 1.5,
        },
    )
    return gs


def resolve_vote(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    consecutive = state.get("consecutive_no_exile_days", 0)
    votes = state.get("exile_votes", {})
    result = engine.resolve_vote(
        gs, votes=votes,
        revote=state.get("revote", False),
        consecutive_no_exile_days=consecutive,
        pk_candidates=state.get("pk_candidates"),
        rng_seed=f"{gs.game_id}-vote-d{gs.day_number}",
    )
    # Log vote tally with weighted counts (read from ruleset config)
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    sheriff_weight = float(engine.ruleset.raw.get("sheriff", {}).get("vote_weight", 1.5))
    weighted_tally: dict[str, float] = {}
    vote_weights: dict[str, float] = {}
    if votes:
        for voter_id, target_id in votes.items():
            weight = sheriff_weight if voter_id == sheriff_id else 1.0
            vote_weights[voter_id] = weight
            weighted_tally[target_id] = weighted_tally.get(target_id, 0) + weight
        tally_items = sorted(weighted_tally.items(), key=lambda x: -x[1])
        tally_text = "  投票统计: " + ", ".join(
            f"{_player_display(state, t)}:{v}票" for t, v in tally_items
        )
    else:
        tally_text = "  投票统计: 无有效票"
    logger.debug(tally_text)
    if result.exiled_player_id:
        gs, _ = _judge_broadcast(
            phase="vote_result_announce",
            message=f"投票结果：{_player_display(state, result.exiled_player_id)} 以最高票被放逐出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] {_player_display(state, result.exiled_player_id)} 被放逐 (原因: {result.reason})")
    elif result.reason == "first_tie_pk":
        tied_names = "、".join(_player_display(state, t) for t in (result.tied_player_ids or []))
        gs, _ = _judge_broadcast(
            phase="vote_tie_pk",
            message=f"首次平票：{tied_names}进入PK发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 首次平票，进入PK: {[_player_display(state, t) for t in (result.tied_player_ids or [])]}")
    elif result.reason == "second_tie_no_exile":
        gs, _ = _judge_broadcast(
            phase="vote_second_tie",
            message="二次平票，无人出局，直接进入黑夜",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 二次平票，无人出局")
    elif result.reason == "anti_stall_empty_tally":
        gs, _ = _judge_broadcast(
            phase="vote_anti_stall",
            message=f"防死循环强制放逐：{_player_display(state, result.exiled_player_id)}出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [投票结果] 防死循环强制放逐: {_player_display(state, result.exiled_player_id)}")
    else:
        logger.debug(f"  [投票结果] {result.reason}")
    payload: dict[str, Any] = {
        "exiled": result.exiled_player_id,
        "reason": result.reason,
        "day_number": gs.day_number,
        "sheriff_id": sheriff_id,
        "sheriff_vote_weight": 1.5 if sheriff_id else 1.0,
        "weighted_tally": weighted_tally,
        "vote_weights": vote_weights,
        "votes": [
            {
                "voter": voter_id,
                "target": target_id,
                "reason": _public_vote_reason(
                    (state.get("vote_action_traces") or {}).get(voter_id)
                ),
            }
            for voter_id, target_id in sorted((state.get("exile_votes") or {}).items())
        ],
    }
    if result.tied_player_ids:
        payload["tied"] = result.tied_player_ids
    vote_trace_events = []
    for pid, trace in (state.get("vote_action_traces") or {}).items():
        audit_event = _action_trace_event(
            player_id=pid,
            phase="vote",
            action_trace=_with_vote_target_in_trace(trace, state.get("exile_votes", {}).get(pid, "")),
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        thought = audit_event.payload.get("private_vote_thought") or {}
        if thought:
            logger.debug(
                f"  [投票心理][仅主持人] {_player_display(state, pid)} -> "
                f"{_player_display(state, thought.get('target'))}: "
                f"站边={thought.get('standing_with_seer') or '未明确'}；"
                f"怀疑理由={thought.get('suspect_reason') or thought.get('public_reason') or '未说明'}；"
                f"排除理由={thought.get('not_voting_reason') or '未说明'}；"
                f"内心理由={thought.get('private_reason') or '未说明'}"
            )
        vote_trace_events.append(audit_event)
    gs = replace(gs, votes=state.get("exile_votes", {}),
                 events=gs.events + [GameEvent(
                     type="vote_resolved",
                     payload=payload,
                 )] + vote_trace_events)
    next_consecutive = (
        consecutive + 1 if result.reason == "second_tie_no_exile" else 0
    )
    next_state: dict[str, Any] = {
        "game_state": gs,
        "_vote_result": result,
        "consecutive_no_exile_days": next_consecutive,
    }
    if result.reason == "first_tie_pk":
        next_state["pk_candidates"] = result.tied_player_ids
    elif result.exiled_player_id is not None or result.reason == "second_tie_no_exile":
        next_state["pk_candidates"] = []
        next_state["exile_votes"] = {}
        next_state["vote_action_traces"] = {}
        next_state["exile_vote_revote"] = False
    return next_state


def resolve_exile(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    # Read exiled player from the last vote_resolved event (not _vote_result
    # which is not in RuntimeState and gets dropped by LangGraph channels).
    exiled_id = None
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            exiled_id = event.payload.get("exiled")
            break
    if exiled_id is None:
        return {"game_state": gs}
    exiled_role = gs.players.get(exiled_id, None)
    role_str = exiled_role.role if exiled_role else "?"
    gs, _ = _judge_broadcast(
        phase="exile",
        message=f"{_player_display(state, exiled_id)}被放逐出局",
        gs=gs, day_number=gs.day_number,
        extra_payload={"exiled": exiled_id},
        visibility="public",
    )
    gs, events = engine.resolve_exile(gs, target_id=exiled_id)
    gs = replace(gs, events=gs.events + events)
    logger.debug(f"  [放逐] {_player_display(state, exiled_id)}({role_str}) 被放逐出局")
    for ev in events:
        if ev.type == "idiot_revealed":
            logger.debug(f"  [白痴亮牌] {_player_display(state, exiled_id)} 是白痴，不会被放逐")
            gs, _ = _judge_broadcast(
                phase="idiot_revealed",
                message=f"{_player_display(state, exiled_id)}亮出白痴身份，不会被放逐，但失去投票权",
                gs=gs, day_number=gs.day_number,
                extra_payload={"player_id": exiled_id},
                visibility="public",
            )
    return {"game_state": gs}


def exile_last_words(state: RuntimeState) -> dict[str, Any]:
    """Exiled player gives last words before death effects resolve."""
    gs: GameState = state["game_state"]
    exiled_id = None
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            exiled_id = event.payload.get("exiled")
            break
    if exiled_id is None:
        return {"game_state": gs}
    # Idiot reveal: player stays alive, no last words needed
    player = gs.players.get(exiled_id)
    if player is None or player.alive:
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="exile_last_words",
        message=f"请{_player_display(state, exiled_id)}发表遗言",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    logger.debug(f"  [遗言] 请{_player_display(state, exiled_id)}发表遗言")

    registry = state.get("agent_registry")
    if registry:
        result = _dispatch_agent(
            state,
            agent_exile_last_words,
            exiled_id,
            timeout_override=AGENT_TIMEOUTS.day_speech,
        )
        speech_text = result.get("speech_text", "") if result else ""
        logger.debug(f"  [遗言] {_player_display(state, exiled_id)}: {speech_text if speech_text else '(无遗言)'}")
        gs = replace(gs, events=gs.events + [GameEvent(
            type="exile_last_words",
            payload={"speaker": exiled_id, "day_number": gs.day_number, "text": speech_text},
        )])

    return {"game_state": gs}


def check_victory(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.check_victory(gs)
    checked_payload = {"winner": result.winner, "reason": result.reason}
    gs = replace(gs, events=gs.events + [GameEvent(type="victory_checked", payload=checked_payload)])

    if result.winner is not None:
        wf = result.winner
        faction_label = "好人阵营" if wf == "good" else "狼人阵营"
        logger.debug(f"\n{'='*60}")
        logger.debug(f"  【游戏结束】胜利方: {faction_label} ({result.reason})")
        logger.debug(f"{'='*60}")

        # Public victory announcement
        identity_reveal = ", ".join(
            f"{pid}({p.role})" for pid, p in gs.players.items()
        )
        gs, _ = _judge_broadcast(
            phase="victory_announce",
            message=f"游戏结束，{faction_label}获胜！({result.reason})",
            gs=gs, day_number=gs.day_number,
            extra_payload={
                "winner": wf,
                "reason": result.reason,
                "identities": identity_reveal,
            },
            visibility="public",
        )
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


def finish_game(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]

    # Final identity reveal broadcast
    identity_lines = []
    for pid, p in gs.players.items():
        status = "存活" if p.alive else "死亡"
        identity_lines.append(f"{pid}({p.role}, {status})")
    gs, _ = _judge_broadcast(
        phase="game_end_reveal",
        message="游戏结束，公布所有玩家身份：" + "；".join(identity_lines),
        gs=gs, day_number=gs.day_number,
        extra_payload={"identities": {pid: p.role for pid, p in gs.players.items()}},
        visibility="public",
    )

    gs = replace(gs, phase="finished",
                 events=gs.events + [GameEvent(type="game_finished", payload={})])
    return {"game_state": gs}
