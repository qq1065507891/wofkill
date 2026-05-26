"""Sheriff election node functions."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_sheriff_election_speech,
    agent_sheriff_pick_speech_order,
    agent_sheriff_register,
    agent_sheriff_vote,
    agent_sheriff_withdraw,
)
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
    eligible_sheriff_voters,
    filter_sheriff_votes_to_eligible,
    is_all_players_on_sheriff,
)
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _action_trace_event,
    _agent_timeout,
    _call_agent,
    _dispatch_agent,
    _judge_broadcast,
    _ensure_day_incremented,
    _player_display,
    _timer_expired,
    AGENT_TIMEOUTS,
    _stable_seed,
)
from werewolf_agent.runtime.timeline import phase_label


def sheriff_first_day_entry(state: RuntimeState) -> dict[str, Any]:
    """Sheriff election entry.

    N1 (before deaths): increments day, broadcasts "天亮了".
    Interrupt re-entry (after deaths): day already set, skip increment.
    """
    gs: GameState = state["game_state"]
    gs, d = _ensure_day_incremented(gs)
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
    """Judge announces sheriff election; each alive player chooses to register or not."""
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    registry = state.get("agent_registry")

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
        result = _dispatch_agent(state, agent_sheriff_register, pid)
        if result is not None:
            has_agents = True
            if result.get("self_destruct"):
                return {"game_state": gs, "self_destruct_wolf_id": pid}
            if result.get("registered"):
                candidates.append(pid)
                logger.debug(f"  [上警报名] {_player_display(state, pid)} 报名上警")
    if not has_agents:
        # Scripted fallback: all alive players register
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
    """Withdrawal phase: candidates choose to stay or withdraw."""
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
        result = _dispatch_agent(state, agent_sheriff_withdraw, candidate_id)
        if result is not None:
            has_agents = True
            if result.get("self_destruct"):
                return {"game_state": gs, "self_destruct_wolf_id": candidate_id}
            if result.get("withdrew"):
                withdrawing.append(candidate_id)
                logger.debug(f"  [退水] {_player_display(state, candidate_id)} 退出竞选")
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


def sheriff_vote(state: RuntimeState) -> dict[str, Any]:
    """Off-sheriff players vote for sheriff. Announces result."""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    registry = state.get("agent_registry")
    candidates = list(gs.sheriff_candidates or [])

    # No candidates -> no sheriff
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

    # One remaining -> elect directly without vote
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
        # Set speech order for day discussion
        speech_order = choose_sheriff_led_speech_order(gs, winner)
        return {"game_state": gs, "speech_order": speech_order}

    # All alive players are candidates -> no sheriff vote, badge is lost.
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

    # Normal vote by off-sheriff voters
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
    for voter_id in voters:
        result = _dispatch_agent(
            state,
            agent_sheriff_vote,
            voter_id,
            candidates,
        )
        if result is not None:
            has_agents = True
            if result.get("self_destruct"):
                return {"game_state": gs, "self_destruct_wolf_id": voter_id}
            if result.get("vote_target"):
                votes[voter_id] = result["vote_target"]
                vote_records.append({"voter": voter_id, "target": result["vote_target"]})
                logger.debug(f"  [警长投票] {_player_display(state, voter_id)} 投票给 {_player_display(state, result['vote_target'])}")
            else:
                vote_records.append({"voter": voter_id, "target": None})
                logger.debug(f"  [警长投票] {_player_display(state, voter_id)} 弃票")
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

    # Record individual sheriff votes for game log visibility
    if vote_records:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_vote_record",
            payload={"votes": vote_records, "candidates": candidates},
        )])

    # Announce result
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

    # No election from vote tie
    gs, _ = _judge_broadcast(
        phase="sheriff_no_election",
        message="投票未选出警长，警徽流失，本局无警长",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    speech_order = choose_no_sheriff_speech_order(gs)
    return {"game_state": gs, "speech_order": speech_order}


def sheriff_speech(state: RuntimeState) -> dict[str, Any]:
    """Collect sheriff-election speeches from candidates."""
    gs: GameState = state["game_state"]
    registry = state.get("agent_registry")
    candidates = list(gs.sheriff_candidates or state.get("sheriff_candidates", []))
    events: list[GameEvent] = []
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
            result = _dispatch_agent(
                state,
                agent_sheriff_election_speech,
                candidate_id,
                candidates,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_text = result.get("speech_text", "") if result else ""
            logger.debug(f"  [警上发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
            new_events: list[GameEvent] = []
            speech_event = GameEvent(
                type="sheriff_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": speech_text,
                },
            )
            new_events.append(speech_event)
            if result and result.get("action_trace"):
                new_events.append(_action_trace_event(
                    player_id=candidate_id,
                    phase="sheriff_speech",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
            # Incrementally update gs so next candidate sees previous speeches
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


def sheriff_endorse(state: RuntimeState) -> dict[str, Any]:
    """Sheriff gives formal endorsement / guipiao before voting.

    Design doc §6.2 node 18: sheriff announces who should be voted out.
    Only runs when sheriff exists and badge is active.
    """
    gs: GameState = state["game_state"]
    sheriff_id = gs.sheriff_id
    if not sheriff_id or gs.sheriff_badge_state != "active":
        return {}

    gs, _ = _judge_broadcast(
        phase="sheriff_endorse",
        message=f"请警长{_player_display(state, sheriff_id)}进行归票",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    result = _dispatch_agent(
        state,
        _sheriff_endorse_adapter,
        sheriff_id,
        timeout_override=120,
    )
    if result:
        speech_text = result.get("speech_text", "")
        endorse_target = result.get("endorse_target", "")
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_endorse",
            payload={
                "speaker": sheriff_id,
                "day_number": gs.day_number,
                "text": speech_text,
                "endorse_target": endorse_target,
            },
        )])
        logger.debug(
            f"  [警长归票] {_player_display(state, sheriff_id)}: "
            f"{speech_text[:80] if speech_text else '(无)'}"
        )
    else:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_endorse",
            payload={"speaker": sheriff_id, "day_number": gs.day_number, "text": ""},
        )])

    return {"game_state": gs}


def _sheriff_endorse_adapter(
    state: RuntimeState,
    engine: RuleEngine,
    registry: Any,
    sheriff_id: str,
) -> dict[str, Any]:
    """Adapter: ask sheriff agent to give endorsement before vote."""
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return {}

    from werewolf_agent.agents.schemas import ActionType, TaskType
    gs: GameState = state["game_state"]
    legal_actions = [ActionType.SPEECH]
    context = None
    try:
        from werewolf_agent.runtime.agent_adapter import build_agent_context
        context = build_agent_context(
            engine, gs, sheriff_id, TaskType.SPEECH,
            legal_actions=legal_actions,
        )
    except Exception:
        pass

    system_prompt = (
        "你是警长，现在是归票环节。请总结今天的讨论，明确指出你认为应该投票出局的玩家。"
    )
    prompt = "请给出你的归票发言。"
    if context:
        prompt = (
            f"Visible state: {context.visible_state_summary}\n"
            f"请给出你的归票发言，明确指出投票目标。"
        )

    try:
        action = agent.act(
            prompt=prompt,
            system_prompt=system_prompt,
            task_type=TaskType.SPEECH,
            legal_actions=legal_actions,
        )
        speech_text = (action.speech_text or "").strip()
        if speech_text and len(speech_text) < 5:
            speech_text = "我归票给最可疑的玩家。"

        import re
        endorse_target = ""
        for m in re.finditer(r"(?:归票|出|投|票)\s*(p\d{2})", speech_text):
            endorse_target = m.group(1)
    except Exception:
        speech_text = ""
        endorse_target = ""

    return {"speech_text": speech_text, "endorse_target": endorse_target}
