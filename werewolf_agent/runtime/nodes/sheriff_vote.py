# -*- coding: utf-8 -*-
"""
警长投票节点。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-18

使用示例:
    >>> from werewolf_agent.runtime.nodes.sheriff_vote import sheriff_vote
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import agent_sheriff_vote
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.skill_opportunity_events import append_private_skill_event
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
)
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
    eligible_sheriff_voters,
    filter_sheriff_votes_to_eligible,
    is_all_players_on_sheriff,
)


def sheriff_vote(state: RuntimeState) -> dict[str, Any]:
    """警下玩家投票选警长，并广播结果。"""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    candidates = list(gs.sheriff_candidates or [])

    if not candidates:
        event = GameEvent(type="sheriff_no_election", payload={"reason": "no_candidates"})
        gs = replace(gs, events=gs.events + [event])
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="无人竞选警长，警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        return {"game_state": gs}

    if len(candidates) == 1:
        winner = candidates[0]
        gs = replace(
            gs,
            sheriff_id=winner,
            sheriff_badge_state="active",
            events=gs.events + [
                GameEvent(type="sheriff_elected", payload={"sheriff_id": winner})
            ],
        )
        gs, _ = _judge_broadcast(
            phase="sheriff_elected",
            message=f"{_player_display(state, winner)} 当选警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [警长选举] {_player_display(state, winner)} 当选警长")
        speech_order = choose_sheriff_led_speech_order(gs, winner)
        return {"game_state": gs, "speech_order": speech_order}

    if is_all_players_on_sheriff(gs, candidates):
        reason = "all_players_on_sheriff"
        event = GameEvent(type="sheriff_no_election", payload={"reason": reason})
        gs = replace(gs, events=gs.events + [event])
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="全员上警，警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        return {"game_state": gs}

    withdrew = list(state.get("sheriff_withdrawing", []))
    voters = eligible_sheriff_voters(gs, candidates, withdrew)
    gs, _ = _judge_broadcast(
        phase="sheriff_vote_start",
        message="警下玩家开始投票选出警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    votes: dict[str, str] = {}
    has_agents = False
    vote_records: list[dict[str, Any]] = []
    audit_events: list[GameEvent] = []
    for voter_id in voters:
        voter = gs.players.get(voter_id)
        self_destruct_available = bool(voter and voter.role == "werewolf" and voter.alive)
        if self_destruct_available:
            gs = append_private_skill_event(
                gs,
                "self_destruct_opportunity",
                actor_id=voter_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_vote",
            )
        decision_identity = _allocate_decision_identity(
            state,
            player_id=voter_id,
            phase="sheriff_vote",
            task_type="sheriff_vote",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        result = _dispatch_agent(
            state,
            agent_sheriff_vote,
            voter_id,
            candidates,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        if result is not None:
            has_agents = True
            if result.get("action_trace"):
                audit_events.extend(_action_audit_events(
                    state=state,
                    player_id=voter_id,
                    phase="sheriff_vote",
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
                    actor_id=voter_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_vote",
                )
                if audit_events:
                    gs = replace(gs, events=gs.events + audit_events)
                return {"game_state": gs, "self_destruct_wolf_id": voter_id}
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=voter_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_vote",
                    reason_code="vote_or_abstain",
                )
            if result.get("vote_target"):
                votes[voter_id] = result["vote_target"]
                vote_records.append({"voter": voter_id, "target": result["vote_target"]})
                logger.debug(f"  [警长投票] {_player_display(state, voter_id)} 投票给 {_player_display(state, result['vote_target'])}")
            else:
                vote_records.append({"voter": voter_id, "target": None})
                logger.debug(f"  [警长投票] {_player_display(state, voter_id)} 弃票")
        else:
            exposure_collector.flush_events()
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=voter_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_vote",
                    reason_code="agent_unavailable",
                )
    if not has_agents:
        votes = state.get("sheriff_votes", {})

    votes = filter_sheriff_votes_to_eligible(
        gs,
        votes,
        candidates=candidates,
        withdrew=withdrew,
    )

    gs, event = engine.resolve_sheriff_vote(gs, votes=votes, candidates=candidates)
    gs = replace(gs, events=gs.events + [event])

    if vote_records:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_vote_record",
            payload={"votes": vote_records, "candidates": candidates},
        )] + audit_events)

    elected_id = event.payload.get("sheriff_id")
    if elected_id:
        gs, _ = _judge_broadcast(
            phase="sheriff_elected",
            message=f"{_player_display(state, elected_id)} 当选警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        logger.debug(f"  [警长选举] {_player_display(state, elected_id)} 当选警长")
        speech_order = choose_sheriff_led_speech_order(gs, elected_id)
        return {"game_state": gs, "speech_order": speech_order}

    tied = event.payload.get("tied", [])
    tie_count = gs.sheriff_tie_count
    if tie_count == 0 and tied:
        gs = replace(
            gs,
            sheriff_tie_count=1,
            sheriff_pk_candidates=tied,
            events=gs.events + [GameEvent(
                type="sheriff_vote_tie_first",
                payload={"tied": tied},
            )],
        )
        gs, _ = _judge_broadcast(
            phase="sheriff_vote_tie_first",
            message=f"警下投票首次平票，{', '.join(_player_display(state, c) for c in tied)} 进入 PK 发言环节",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="sheriff_no_election",
        message="投票未选出警长，警徽流失，本局无警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
    speech_order = choose_no_sheriff_speech_order(gs)
    return {"game_state": gs, "speech_order": speech_order}


__all__ = ["sheriff_vote"]
