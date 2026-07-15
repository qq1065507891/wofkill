# -*- coding: utf-8 -*-
"""
处理日间死亡公告、警徽流失公告和遗言节点。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-15

使用示例:
    内部运行时节点模块。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.core.resolution_batches import carrier_matches_resolution_batch
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_exile_last_words,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _hitl_checkpoint,
    _ensure_day_incremented,
    _ensure_runtime_audit_state,
    _player_display,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import phase_label
from werewolf_agent.evaluation.balance_public_claims import (
    public_speech_history,
    sanitize_public_text,
)


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

    # 公告昨夜死亡情况。
    night_deaths = [
        death for death in gs.deaths
        if death.timing == "night"
        and carrier_matches_resolution_batch(death, f"night_{gs.night_number}")
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
        logger.debug("  [死讯] 平安夜，无人死亡")
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
    logger.debug("  [警徽] 本局警徽永久流失")
    result["game_state"] = gs
    return result


def night_death_last_words(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    registry = state.get("agent_registry")
    batch = f"night_{gs.night_number}"
    eligible = []
    for death in gs.deaths:
        if not carrier_matches_resolution_batch(death, batch):
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

    # 让每个符合条件的夜死玩家通过 agent 发表遗言。
    if registry and eligible:
        _ensure_runtime_audit_state(state)
        for pid in eligible:
            call_state = {**state, "game_state": gs}
            decision_identity = _allocate_decision_identity(
                call_state,
                player_id=pid,
                phase="night_death_last_words",
                task_type="last_words",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector()
            result = _dispatch_agent(
                call_state,
                agent_exile_last_words,
                pid,
                timeout_override=AGENT_TIMEOUTS.day_speech,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
            )
            speech_text = result.get("speech_text", "") if result else ""
            speech_text, redacted_claims = sanitize_public_text(
                speech_text,
                public_speech_history(gs.events),
            )
            logger.debug(f"  [夜死遗言] {_player_display(state, pid)}: {speech_text if speech_text else '(无遗言)'}")
            new_events = [GameEvent(
                type="night_death_last_words",
                payload={
                    "speaker": pid,
                    "day_number": gs.day_number,
                    "text": speech_text,
                    **({"redacted_public_claims": redacted_claims} if redacted_claims else {}),
                },
            )]
            if result and result.get("action_trace"):
                new_events.extend(_action_audit_events(
                    state=call_state,
                    player_id=pid,
                    phase="night_death_last_words",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            gs = replace(gs, events=gs.events + new_events)

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
    player = gs.players.get(exiled_id)
    if player is None or player.alive:
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="exile_last_words",
        message=f"请{_player_display(state, exiled_id)}发表遗言",
        gs=gs, day_number=gs.day_number,
        extra_payload={"player_id": exiled_id},
        visibility="public",
    )
    logger.debug(f"  [遗言] 请{_player_display(state, exiled_id)}发表遗言")

    registry = state.get("agent_registry")
    if registry:
        decision_identity = _allocate_decision_identity(
            state,
            player_id=exiled_id,
            phase="exile_last_words",
            task_type="last_words",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector()
        result = _dispatch_agent(
            state,
            agent_exile_last_words,
            exiled_id,
            timeout_override=AGENT_TIMEOUTS.day_speech,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        speech_text = result.get("speech_text", "") if result else ""
        speech_text, redacted_claims = sanitize_public_text(
            speech_text,
            public_speech_history(gs.events),
        )
        logger.debug(f"  [遗言] {_player_display(state, exiled_id)}: {speech_text if speech_text else '(无遗言)'}")
        new_events = [GameEvent(
            type="exile_last_words",
            payload={
                "speaker": exiled_id,
                "day_number": gs.day_number,
                "text": speech_text,
                **({"redacted_public_claims": redacted_claims} if redacted_claims else {}),
            },
        )]
        if result and result.get("action_trace"):
            new_events.extend(_action_audit_events(
                state=state,
                player_id=exiled_id,
                phase="exile_last_words",
                action_trace=result["action_trace"],
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        else:
            exposure_collector.flush_events()
        gs = replace(gs, events=gs.events + new_events)

    return {"game_state": gs}
