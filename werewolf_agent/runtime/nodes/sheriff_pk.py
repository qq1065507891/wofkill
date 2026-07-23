# -*- coding: utf-8 -*-
"""Sheriff election PK speech and revote nodes (after first vote tie).
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-18
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import (
    agent_sheriff_election_speech,
    agent_sheriff_vote,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.skill_opportunity_events import (
    append_private_skill_event,
    append_self_destruct_declined,
    append_self_destruct_opportunity,
    append_self_destruct_selected,
    can_select_self_destruct,
    is_live_werewolf,
)
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
)
from werewolf_agent.evaluation.balance_public_claims import (
    public_speech_history,
    sanitize_public_text,
)


def sheriff_pk_speech(state: RuntimeState) -> dict[str, Any]:
    """Only sheriff_pk_candidates give speeches during PK phase."""
    gs: GameState = state["game_state"]
    pk_candidates = list(gs.sheriff_pk_candidates or [])
    events: list[GameEvent] = []

    if not pk_candidates:
        # Edge case: no candidates recorded — skip to no_election
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="警徽流失，本局无警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
        return {"game_state": gs}

    pk_names = ", ".join(_player_display(state, c) for c in pk_candidates)
    gs, _ = _judge_broadcast(
        phase="sheriff_pk_speech_start",
        message=f"首次平票，{pk_names} 进入 PK 发言环节，请依次发言",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    for candidate_id in pk_candidates:
        self_destruct_available = is_live_werewolf(gs, candidate_id)
        if self_destruct_available:
            gs, self_destruct_available = append_self_destruct_opportunity(
                gs,
                actor_id=candidate_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_pk_speech",
            )
        decision_identity = _allocate_decision_identity(
            state,
            player_id=candidate_id,
            phase="sheriff_pk_speech",
            task_type="sheriff_speech",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        result = _dispatch_agent(
            state,
            agent_sheriff_election_speech,
            candidate_id,
            pk_candidates,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        if result and result.get("self_destruct") and can_select_self_destruct(
            gs,
            actor_id=candidate_id,
            day_number=gs.day_number,
            opportunity_phase="sheriff_pk_speech",
        ):
            gs, _ = append_self_destruct_selected(
                gs,
                actor_id=candidate_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_pk_speech",
            )
            if result.get("action_trace"):
                events.extend(_action_audit_events(
                    state=state,
                    player_id=candidate_id,
                    phase="sheriff_pk_speech",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            return {
                "game_state": replace(gs, events=gs.events + events),
                "self_destruct_wolf_id": candidate_id,
            }
        if self_destruct_available:
            gs = append_self_destruct_declined(
                gs,
                actor_id=candidate_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_pk_speech",
                reason_code=("agent_unavailable" if result is None else "continued_speech"),
            )
        speech_text = result.get("speech_text", "") if result else ""
        speech_text, redacted_claims = sanitize_public_text(
            speech_text,
            public_speech_history(gs.events),
        )
        logger.debug(f"  [警长PK发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
        events.append(GameEvent(
            type="sheriff_pk_speech",
            payload={
                "speaker": candidate_id,
                "day_number": gs.day_number,
                "text": speech_text,
                **({"redacted_public_claims": redacted_claims} if redacted_claims else {}),
            },
        ))
        if result and result.get("action_trace"):
            events.extend(_action_audit_events(
                state=state,
                player_id=candidate_id,
                phase="sheriff_pk_speech",
                action_trace=result["action_trace"],
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        else:
            exposure_collector.flush_events()

    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def sheriff_revote(state: RuntimeState) -> dict[str, Any]:
    """Revote after sheriff PK — only sheriff_pk_candidates are eligible.

    Voters exclude all PK candidates (they cannot vote in their own PK).
    If revote also ties, emit sheriff_no_election.
    """
    from werewolf_agent.engine.rule_engine import RuleEngine

    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    pk_candidates = list(gs.sheriff_pk_candidates or [])

    if not pk_candidates:
        gs, _ = _judge_broadcast(
            phase="sheriff_no_election",
            message="无 PK 候选人，警徽流失",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="sheriff_revote_start",
        message=(
            f"PK 发言结束，警下玩家重新投票选出警长"
            f"（仅 {', '.join(_player_display(state, c) for c in pk_candidates)} 可选）"
        ),
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    # Voters exclude all PK candidates (they cannot vote in their own PK)
    voters = [
        pid for pid, p in gs.players.items()
        if p.alive and pid not in pk_candidates
    ]

    votes: dict[str, str] = {}
    vote_records: list[dict[str, Any]] = []
    audit_events: list[GameEvent] = []
    for voter_id in voters:
        self_destruct_available = is_live_werewolf(gs, voter_id)
        if self_destruct_available:
            gs = append_private_skill_event(
                gs,
                "self_destruct_opportunity",
                actor_id=voter_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_revote",
            )
        decision_identity = _allocate_decision_identity(
            state,
            player_id=voter_id,
            phase="sheriff_revote",
            task_type="sheriff_vote",
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
        result = _dispatch_agent(
            state,
            agent_sheriff_vote,
            voter_id,
            pk_candidates,
            decision_identity=decision_identity,
            exposure_collector=exposure_collector,
        )
        if result is not None:
            if result.get("action_trace"):
                audit_events.extend(_action_audit_events(
                    state=state,
                    player_id=voter_id,
                    phase="sheriff_revote",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            if result.get("self_destruct") and can_select_self_destruct(
                gs,
                actor_id=voter_id,
                day_number=gs.day_number,
                opportunity_phase="sheriff_revote",
            ):
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_selected",
                    actor_id=voter_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_revote",
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
                    opportunity_phase="sheriff_revote",
                    reason_code="vote_or_abstain",
                )
            if result.get("vote_target"):
                votes[voter_id] = result["vote_target"]
                vote_records.append({"voter": voter_id, "target": result["vote_target"]})
                logger.debug(
                    f"  [警长复投] {_player_display(state, voter_id)} "
                    f"投给 {_player_display(state, result['vote_target'])}"
                )
            else:
                vote_records.append({"voter": voter_id, "target": None})
                logger.debug(f"  [警长复投] {_player_display(state, voter_id)} 弃票")
        else:
            exposure_collector.flush_events()
            if self_destruct_available:
                gs = append_private_skill_event(
                    gs,
                    "self_destruct_declined",
                    actor_id=voter_id,
                    day_number=gs.day_number,
                    opportunity_phase="sheriff_revote",
                    reason_code="agent_unavailable",
                )

    if vote_records:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_vote_record",
            payload={"votes": vote_records, "candidates": pk_candidates, "revote": True},
        )] + audit_events)

    gs, event = engine.resolve_sheriff_vote(gs, votes=votes, candidates=pk_candidates)
    gs = replace(gs, events=gs.events + [event])

    elected_id = event.payload.get("sheriff_id")
    if elected_id:
        gs = replace(gs, sheriff_id=elected_id, sheriff_badge_state="active",
                     sheriff_tie_count=0, sheriff_pk_candidates=[])
        gs, _ = _judge_broadcast(
            phase="sheriff_elected",
            message=f"{_player_display(state, elected_id)} 当选警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        speech_order = choose_sheriff_led_speech_order(gs, elected_id)
        return {"game_state": gs, "speech_order": speech_order}

    # Revote also tied → no sheriff (second tie)
    gs, _ = _judge_broadcast(
        phase="sheriff_no_election",
        message="复投仍未选出警长，警徽流失，本局无警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    gs = replace(gs, sheriff_tie_count=0, sheriff_pk_candidates=[])
    speech_order = choose_no_sheriff_speech_order(gs)
    return {"game_state": gs, "speech_order": speech_order}
