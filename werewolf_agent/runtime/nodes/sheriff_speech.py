# -*- coding: utf-8 -*-
"""
警上竞选发言节点。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.sheriff_speech import sheriff_speech
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.agent_adapter import agent_sheriff_election_speech
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
    _stable_seed,
)
from werewolf_agent.runtime.sheriff_policy import is_all_players_on_sheriff
from werewolf_agent.evaluation.balance_public_claims import (
    public_speech_history,
    sanitize_public_text,
)


def sheriff_speech(state: RuntimeState) -> dict[str, Any]:
    """收集警上竞选发言。"""
    gs: GameState = state["game_state"]
    registry = state.get("agent_registry")
    candidates = list(gs.sheriff_candidates or state.get("sheriff_candidates", []))
    if not candidates:
        gs = replace(gs, events=gs.events + [GameEvent(type="sheriff_speech", payload={})])
        return {"game_state": gs}

    import random as _random

    seed = _stable_seed(gs.game_id, "sheriff_speech_order", gs.day_number)
    rng = _random.Random(seed)
    speech_order = list(candidates)
    rng.shuffle(speech_order)

    names = ", ".join(_player_display(state, p) for p in speech_order)
    first_speaker = speech_order[0] if speech_order else ""
    if is_all_players_on_sheriff(gs, candidates):
        no_election = GameEvent(
            type="sheriff_no_election",
            payload={"reason": "all_players_on_sheriff"},
        )
        gs = replace(gs, events=gs.events + [no_election])
        gs, _ = _judge_broadcast(
            phase="sheriff_all_players_registered",
            message=(
                "本局全员上警，警徽流失，本局无警长；"
                f"现在由{_player_display(state, first_speaker)}开始发言。警上发言顺序: {names}"
            ),
            gs=gs,
            day_number=gs.day_number,
            extra_payload={"speech_order": speech_order},
            visibility="public",
        )
    else:
        gs, _ = _judge_broadcast(
            phase="sheriff_speech_start",
            message=f"由法官随机指定警上发言顺序: {names}",
            gs=gs,
            day_number=gs.day_number,
            extra_payload={"speech_order": speech_order},
            visibility="public",
        )

    if registry:
        for candidate_id in speech_order:
            decision_identity = _allocate_decision_identity(
                state,
                player_id=candidate_id,
                phase="sheriff_speech",
                task_type="sheriff_speech",
                day_number=gs.day_number,
                night_number=gs.night_number,
            )
            exposure_collector = ModuleExposureAuditCollector(prompt_proof_key_provider=state.get("prompt_proof_key_provider"))
            result = _dispatch_agent(
                state,
                agent_sheriff_election_speech,
                candidate_id,
                candidates,
                timeout_override=AGENT_TIMEOUTS.day_speech,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
            )
            speech_text = result.get("speech_text", "") if result else ""
            speech_text, redacted_claims = sanitize_public_text(
                speech_text,
                public_speech_history(gs.events),
            )
            logger.debug(f"  [警上发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
            new_events: list[GameEvent] = []
            speech_event = GameEvent(
                type="sheriff_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": speech_text,
                    **({"redacted_public_claims": redacted_claims} if redacted_claims else {}),
                },
            )
            new_events.append(speech_event)
            if result and result.get("action_trace"):
                new_events.extend(_action_audit_events(
                    state=state,
                    player_id=candidate_id,
                    phase="sheriff_speech",
                    action_trace=result["action_trace"],
                    decision_identity=decision_identity,
                    exposure_collector=exposure_collector,
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            else:
                exposure_collector.flush_events()
            # 每个候选人发言后立即写回，让后续候选人能看到前序发言。
            gs = replace(gs, events=gs.events + new_events)
            state["game_state"] = gs
    else:
        gs = replace(gs, events=gs.events + [
            GameEvent(
                type="sheriff_speech",
                payload={"speech_order": speech_order},
            )
        ])

    return {"game_state": gs}


__all__ = ["sheriff_speech"]
