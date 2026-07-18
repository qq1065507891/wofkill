# -*- coding: utf-8 -*-
"""
警长竞选入口、报名和退水节点。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-18

使用示例:
    >>> from werewolf_agent.runtime.nodes.sheriff_registration import sheriff_registration
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_sheriff_register,
    agent_sheriff_withdraw,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.skill_opportunity_events import append_private_skill_event
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _ensure_day_incremented,
    _judge_broadcast,
    _player_display,
)


def sheriff_first_day_entry(state: RuntimeState) -> dict[str, Any]:
    """警长竞选入口。"""
    gs: GameState = state["game_state"]
    gs, d = _ensure_day_incremented(state, gs)
    logger.debug(f"\n{'='*60}")
    logger.debug(f"  【警长竞选】开始D{d}警长竞选环节")
    logger.debug(f"{'='*60}")
    return {
        "game_state": gs,
        "revote": False,
        "speech_index": 0,
        "current_speaker_id": None,
        "speech_order": [],
    }


def sheriff_registration(state: RuntimeState) -> dict[str, Any]:
    """法官宣布警长竞选，每个存活玩家决定是否上警。"""
    gs: GameState = state["game_state"]

    gs, _ = _judge_broadcast(
        phase="sheriff_election",
        message="开始警上竞选环节，请想要竞选警长的玩家举手报名",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    candidates: list[str] = []
    has_agents = False
    for pid, p in gs.players.items():
        if not p.alive:
            continue
        self_destruct_available = p.role == "werewolf"
        if self_destruct_available:
            gs = append_private_skill_event(
                gs,
                "self_destruct_opportunity",
                actor_id=pid,
                day_number=gs.day_number,
                opportunity_phase="sheriff_registration",
            )
        decision_identity = _allocate_decision_identity(
            state,
            player_id=pid,
            phase="sheriff_registration",
            task_type="sheriff_register",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        result = _dispatch_agent(
            state,
            agent_sheriff_register,
            pid,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        if result is not None:
            has_agents = True
            if result.get("action_trace"):
                gs = replace(gs, events=gs.events + _action_audit_events(
                    state=state,
                    player_id=pid,
                    phase="sheriff_registration",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            if result.get("self_destruct"):
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_selected",
                    actor_id=pid,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_registration",
                )
                return {"game_state": gs, "self_destruct_wolf_id": pid}
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=pid,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_registration",
                    reason_code="registered_or_declined",
                )
            if result.get("registered"):
                candidates.append(pid)
                logger.debug(f"  [上警报名] {_player_display(state, pid)} 报名上警")
        else:
            exposure_collector.flush_events()
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=pid,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_registration",
                    reason_code="agent_unavailable",
                )
    if not has_agents:
        candidates = state.get("sheriff_candidates", [])
        if not candidates:
            candidates = [pid for pid, p in gs.players.items() if p.alive]

    if candidates:
        names = ", ".join(_player_display(state, c) for c in candidates)
        gs, _ = _judge_broadcast(
            phase="sheriff_registered",
            message=f"以下玩家报名上警: {names}",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
    else:
        gs, _ = _judge_broadcast(
            phase="sheriff_registered",
            message="无人报名上警，警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )

    gs = replace(gs, sheriff_candidates=candidates,
                 events=gs.events + [GameEvent(
                     type="sheriff_registered",
                     payload={"candidates": candidates},
                 )])
    return {"game_state": gs, "sheriff_candidates": candidates}


def sheriff_withdraw(state: RuntimeState) -> dict[str, Any]:
    """退水阶段：候选人决定留在警上或退水。"""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    candidates = list(gs.sheriff_candidates or [])

    if not candidates:
        return {"game_state": gs, "sheriff_candidates": []}

    gs, _ = _judge_broadcast(
        phase="sheriff_withdraw_start",
        message="退水环节开始，想要退出竞选的玩家可以退水",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    withdrawing: list[str] = []
    has_agents = False
    for candidate_id in candidates:
        candidate = gs.players.get(candidate_id)
        self_destruct_available = bool(candidate and candidate.alive and candidate.role == "werewolf")
        if self_destruct_available:
            gs = append_private_skill_event(
                gs,
                "self_destruct_opportunity",
                actor_id=candidate_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_withdraw",
            )
        decision_identity = _allocate_decision_identity(
            state,
            player_id=candidate_id,
            phase="sheriff_withdraw",
            task_type="sheriff_withdraw",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        result = _dispatch_agent(
            state,
            agent_sheriff_withdraw,
            candidate_id,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        if result is not None:
            has_agents = True
            if result.get("action_trace"):
                gs = replace(gs, events=gs.events + _action_audit_events(
                    state=state,
                    player_id=candidate_id,
                    phase="sheriff_withdraw",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            if result.get("self_destruct"):
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_selected",
                    actor_id=candidate_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_withdraw",
                )
                return {"game_state": gs, "self_destruct_wolf_id": candidate_id}
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=candidate_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_withdraw",
                    reason_code="withdrew_or_stayed",
                )
            if result.get("withdrew"):
                withdrawing.append(candidate_id)
                logger.debug(f"  [退水] {_player_display(state, candidate_id)} 退出竞选")
        else:
            exposure_collector.flush_events()
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=candidate_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_withdraw",
                    reason_code="agent_unavailable",
                )
    if not has_agents:
        withdrawing = state.get("sheriff_withdrawing", [])

    gs, event = engine.sheriff_withdraw(gs, candidates=candidates, withdrawing=withdrawing)
    remaining = event.payload.get("remaining", candidates)
    gs = replace(gs, sheriff_candidates=remaining, events=gs.events + [event])

    if remaining:
        stayed = ", ".join(_player_display(state, c) for c in remaining)
        gs, _ = _judge_broadcast(
            phase="sheriff_withdraw_result",
            message=f"退水结束，留在警上的玩家: {stayed}",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
    else:
        gs, _ = _judge_broadcast(
            phase="sheriff_withdraw_result",
            message="全部候选人退水，警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )

    return {
        "game_state": gs,
        "sheriff_candidates": remaining,
        "sheriff_withdrawing": withdrawing,
    }


__all__ = [
    "sheriff_first_day_entry",
    "sheriff_registration",
    "sheriff_withdraw",
]
