"""Special skill node functions (hunter, self-destruct, PK, badge transfer)."""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_badge_decision,
    agent_hunter_shot,
    agent_pk_speech,
)
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    _action_trace_event,
    _agent_timeout,
    _call_agent,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
    _sheriff_died_this_batch,
    _timer_expired,
    logger,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS


def post_exile_skills(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    # Skill effects are resolved by their dedicated nodes so broadcasts,
    # agent choices, and audit events cannot be skipped.
    return {"game_state": gs}


def resolve_hunter_shot(state: RuntimeState) -> dict[str, Any]:
    """Resolve pending hunter shot after night wolf-kill, before victory check."""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]

    for death in gs.deaths:
        if "hunter_shot" not in (death.triggered_skills or []):
            continue
        if death.player_id not in gs.players:
            continue
        player = gs.players[death.player_id]
        if player.role != "hunter" or player.alive:
            continue
        # Skip if already resolved (death from this hunter already applied)
        already_shot = any(
            d.source_player_id == death.player_id and d.reason == "hunter_shot"
            for d in gs.deaths
        )
        if already_shot:
            continue

        gs, _ = _judge_broadcast(
            phase="hunter_shot_prompt",
            message=f"猎人{_player_display(state, death.player_id)}发动技能，请选择是否开枪",
            gs=gs,
            day_number=gs.day_number,
            night_number=gs.night_number,
            visibility="public",
            extra_payload={"hunter_id": death.player_id},
        )
        logger.debug(f"  [猎人开枪] 请{_player_display(state, death.player_id)}选择是否开枪")

        # Get target: scripted, then agent, then explicit last-words declaration.
        target = state.get("hunter_shot_target_id")
        if target is None:
            shot_state = {**state, "hunter_death_reason": death.reason}
            shot_result = _dispatch_agent(
                shot_state,
                agent_hunter_shot,
                death.player_id,
                timeout_override=AGENT_TIMEOUTS.hunter_shot,
            )
            if isinstance(shot_result, dict):
                target = shot_result.get("hunter_shot_target_id")
            elif isinstance(shot_result, str):
                target = shot_result
            else:
                target = None
        if target is None:
            target = _hunter_shot_target_from_last_words(gs, death.player_id)
        if target and target in gs.players and gs.players[target].alive and target != death.player_id:
            gs, _ = _judge_broadcast(
                phase="hunter_shot_choice",
                message=f"猎人{_player_display(state, death.player_id)}选择带走{_player_display(state, target)}",
                gs=gs,
                day_number=gs.day_number,
                night_number=gs.night_number,
                visibility="public",
                extra_payload={"hunter_id": death.player_id, "target_id": target},
            )
            logger.debug(
                f"  [猎人开枪] {_player_display(state, death.player_id)} "
                f"选择带走{_player_display(state, target)}"
            )
            shot_death = Death(
                player_id=target, reason="hunter_shot",
                timing=death.timing, resolution_batch=death.resolution_batch,
                source_player_id=death.player_id,
            )
            gs = engine.apply_death(gs, shot_death)
            # Public death announcement for shot player
            gs, _ = _judge_broadcast(
                phase="hunter_shot_death",
                message=f"{_player_display(state, target)}被猎人开枪射杀，死亡",
                gs=gs, day_number=gs.day_number, night_number=gs.night_number,
                extra_payload={"target_id": target, "hunter_id": death.player_id},
                visibility="public",
            )
            # Emit public hunter_shot event for agent_adapter visibility
            gs = replace(gs, events=gs.events + [GameEvent(
                type="hunter_shot_public",
                payload={
                    "hunter_id": death.player_id,
                    "target_id": target,
                    "day_number": gs.day_number,
                    "night_number": gs.night_number,
                },
            )])
        else:
            gs, _ = _judge_broadcast(
                phase="hunter_shot_decline",
                message=f"猎人{_player_display(state, death.player_id)}选择不开枪",
                gs=gs,
                day_number=gs.day_number,
                night_number=gs.night_number,
                visibility="public",
                extra_payload={"hunter_id": death.player_id},
            )
            logger.debug(f"  [猎人开枪] {_player_display(state, death.player_id)} 选择不开枪")
            gs = replace(gs, events=gs.events + [GameEvent(
                type="hunter_shot_declined",
                payload={
                    "hunter_id": death.player_id,
                    "day_number": gs.day_number,
                    "night_number": gs.night_number,
                    "resolution_batch": death.resolution_batch,
                },
            )])
        break

    return {"game_state": gs}


def _hunter_shot_target_from_last_words(gs: GameState, hunter_id: str) -> str | None:
    """Extract an explicit hunter-shot target from the hunter's last words."""
    alive_targets = {
        pid for pid, player in gs.players.items()
        if player.alive and pid != hunter_id
    }
    if not alive_targets:
        return None
    for event in reversed(gs.events):
        if event.type not in {"exile_last_words", "night_death_last_words"}:
            continue
        payload = event.payload or {}
        if payload.get("speaker") != hunter_id:
            continue
        text = str(payload.get("text") or "")
        for match in re.finditer(r"(?:带走|开枪(?:带走|打)?|枪(?:带走|打)?|选择带走)\s*(p\d{2}|[A-Za-z]\w*)", text):
            candidate = match.group(1)
            if candidate in alive_targets:
                return candidate
    return None


def sheriff_badge_transfer(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return {"game_state": gs}
    sheriff = gs.players.get(gs.sheriff_id)
    if sheriff is None or sheriff.alive:
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="badge_decision",
        message=f"警长{_player_display(state, gs.sheriff_id)}死亡，请决定警徽去向",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    logger.debug(f"  [警徽] 警长{_player_display(state, gs.sheriff_id)}死亡，决定警徽去向")

    decision = state.get("badge_decision")
    target_id = state.get("badge_target_id")

    # Agent-driven: dying sheriff decides transfer or tear
    if decision is None:
        result = _dispatch_agent(
            state,
            agent_badge_decision,
            gs.sheriff_id,
            timeout_override=AGENT_TIMEOUTS.day_vote,
        )
        if result:
            decision = result.get("badge_decision", "tear")
            target_id = result.get("badge_target_id")

    if decision is None:
        decision = "tear"

    gs = engine.resolve_badge_decision(gs, decision=decision, target_id=target_id)
    event_type = "badge_torn" if decision == "tear" else "badge_transferred"

    if decision == "transfer" and target_id:
        gs, _ = _judge_broadcast(
            phase="badge_transferred",
            message=f"警长将警徽移交给{_player_display(state, target_id)}",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [警徽] 警长将警徽移交给 {_player_display(state, target_id)}")
    else:
        gs, _ = _judge_broadcast(
            phase="badge_torn",
            message="警长撕毁了警徽，本局不再有警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [警徽] 警长撕毁了警徽")

    gs = replace(gs, events=gs.events + [GameEvent(
        type=event_type,
        payload={"new_sheriff_id": target_id} if decision == "transfer" else {},
    )])
    return {"game_state": gs}


def resolve_self_destruct_node(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id:
        gs, events = engine.resolve_self_destruct(gs, wolf_id=wolf_id, day_number=gs.day_number)
        # If sheriff election was in progress (no sheriff yet), track interruption
        if gs.sheriff_id is None or gs.sheriff_badge_state == "none":
            gs = replace(gs,
                         sheriff_interrupt_count=gs.sheriff_interrupt_count + 1,
                         sheriff_candidates=[])
            count = gs.sheriff_interrupt_count
            gs, _ = _judge_broadcast(
                phase="sheriff_interrupted",
                message=f"竞选过程中有人自爆，警长竞选中断（第{count}次中断）",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )
        gs = replace(gs, events=gs.events + events)
    return {"game_state": gs, "self_destruct_wolf_id": None}


def tie_pk_speech(state: RuntimeState) -> dict[str, Any]:
    """After first exile tie, only PK candidates give speeches."""
    gs: GameState = state["game_state"]
    pk_candidates = state.get("pk_candidates", [])
    registry = state.get("agent_registry")
    events: list[GameEvent] = []

    # Judge announces PK speech phase
    pk_names = [_player_display(state, c) for c in pk_candidates]
    gs, _ = _judge_broadcast(
        phase="pk_speech_start",
        message=f"首次平票，{', '.join(pk_names)}进入PK发言环节，请依次发言",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    if registry and pk_candidates:
        for candidate_id in pk_candidates:
            result = _dispatch_agent(
                state,
                agent_pk_speech,
                candidate_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_text = result.get("speech_text", "") if result else ""
            logger.debug(f"  [PK发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
            events.append(GameEvent(
                type="tie_pk_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": speech_text,
                },
            ))
            if result and result.get("action_trace"):
                events.append(_action_trace_event(
                    player_id=candidate_id,
                    phase="pk_speech",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
    else:
        events.append(GameEvent(type="tie_pk_speech", payload={}))

    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def tie_revote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="pk_revote_start",
        message="PK发言结束，请重新投票",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    return {
        "game_state": gs,
        "exile_votes": {},
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": True,
        "revote": True,
    }
