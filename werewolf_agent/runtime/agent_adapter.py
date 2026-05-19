"""Agent runtime adapter: converts GameState into AgentContext for PlayerAgent.

When an AgentRegistry is provided to the runtime graph, night/day nodes will
delegate decisions to PlayerAgent instances. Without a registry, deterministic
scripted fallback is used (preserving existing test behavior).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    TaskType,
)
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS

logger = logging.getLogger(__name__)


class AgentRegistry(Protocol):
    """Maps player_id to PlayerAgent. Return None for scripted fallback."""

    def get_agent(self, player_id: str) -> PlayerAgent | None: ...


class SimpleAgentRegistry:
    """Concrete registry: maps player_id -> PlayerAgent."""

    def __init__(self, agents: dict[str, PlayerAgent] | None = None) -> None:
        self._agents: dict[str, PlayerAgent] = agents or {}

    def register(self, player_id: str, agent: PlayerAgent) -> None:
        self._agents[player_id] = agent

    def get_agent(self, player_id: str) -> PlayerAgent | None:
        return self._agents.get(player_id)


def _action_trace_payload(action: Any) -> dict[str, Any] | None:
    trace = getattr(action, "trace", None)
    return trace.model_dump() if trace else None


def _build_witch_pressure_targets(gs: GameState) -> list[dict[str, Any]]:
    """Build poison pressure targets from public state.

    Pressure sources:
    - Unresolved seer black claim (查杀 in public speech)
    - Player contradicted claimed role
    """
    import re as _re

    targets: dict[str, dict[str, Any]] = {}

    # Extract from public speeches: black claims (查杀)
    for event in gs.events:
        if event.type == "speech":
            text = event.payload.get("text", "")
            # Look for 查杀 claims targeting a player (support various ID formats)
            # Pattern: "PLAYER查杀" or "PLAYER...查杀"
            black_match = _re.search(r"([a-zA-Z]+\d+).*?查杀", text)
            if not black_match:
                # Also try reverse: "查杀...PLAYER"
                black_match = _re.search(r"查杀.*?([a-zA-Z]+\d+)", text)
            if black_match:
                target_id = black_match.group(1)
                if target_id not in targets:
                    targets[target_id] = {
                        "player_id": target_id,
                        "pressure_type": "black_claim",
                        "source": event.payload.get("speaker", ""),
                        "description": f"被{event.payload.get('speaker', '?')}查杀",
                    }

    return list(targets.values())


def build_agent_context(
    engine: RuleEngine,
    gs: GameState,
    player_id: str,
    task_type: TaskType,
    *,
    legal_actions: list[ActionType] | None = None,
    legal_targets: list[str] | None = None,
    wolf_kill_target_id: str | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
) -> AgentContext:
    """Build AgentContext for a player from current game state.

    Visibility rules:
    - Player only sees their own role.
    - No moderator_full, no other players' private state.
    - Wolf teammates visible only to wolves.
    - Seer sees own check results only.
    - Witch sees potion availability only.
    """
    player = gs.players.get(player_id)
    if player is None:
        return AgentContext(agent_id=player_id, task_type=task_type)

    # Build simplified visible state
    visible: dict[str, Any] = {
        "phase": gs.phase,
        "day": gs.day_number,
        "night": gs.night_number,
        "alive_players": [pid for pid, p in gs.players.items() if p.alive],
        "dead_players": [{"id": d.player_id, "reason": d.reason} for d in gs.deaths],
        "sheriff_id": gs.sheriff_id,
        "badge_state": gs.sheriff_badge_state,
    }

    # Role-specific private info
    if player.role == "werewolf":
        visible["wolf_teammates"] = [
            pid for pid, p in gs.players.items()
            if p.alive and p.role == "werewolf" and pid != player_id
        ]
        if wolf_team_plan:
            visible["wolf_team_plan"] = wolf_team_plan
    elif player.role == "seer":
        check_results = []
        for e in gs.events:
            if e.type == "seer_check" and e.payload.get("seer_id") == player_id:
                check_results.append({
                    "target_id": e.payload["target_id"],
                    "alignment": e.payload["alignment"],
                    "night_number": e.payload["night_number"],
                })
        visible["check_results"] = check_results
    elif player.role == "witch":
        visible["antidote_available"] = not gs.antidote_used
        visible["poison_available"] = not gs.poison_used
        if wolf_kill_target_id:
            visible["wolf_kill_target"] = wolf_kill_target_id
        # Poison pressure targets from public state
        if not gs.poison_used:
            pressure_targets = _build_witch_pressure_targets(gs)
            if pressure_targets:
                visible["poison_pressure_targets"] = pressure_targets
    elif player.role == "hybrid" and gs.hybrid_master_id:
        visible["master_id"] = gs.hybrid_master_id

    # Build recent transcript from public speech events
    # Include both day speeches and sheriff election speeches
    transcript: list[dict[str, Any]] = []
    for e in reversed(gs.events):
        if e.type in ("speech", "sheriff_speech") and len(transcript) < 8:
            transcript.insert(0, {
                "speaker": e.payload.get("speaker", ""),
                "text": e.payload.get("text", ""),
                "type": e.type,
            })

    # Build public summary: chronological timeline of key public events.
    # Strategy to stay within context budget (~2500 chars):
    #   - Structural events (deaths, votes, exiles): keep in full (compact)
    #   - Speeches from previous days: truncate to SPEECH_SNIPPET_LEN chars
    #   - If total exceeds SUMMARY_BUDGET, drop oldest lines first
    SUMMARY_BUDGET = 2500
    SPEECH_SNIPPET_LEN = 120
    summary_parts: list[str] = []
    current_day = gs.day_number
    for e in gs.events:
        if e.type == "day_announce":
            day = e.payload.get("day", "?")
            summary_parts.append(f"\n===== 第{day}天 =====")
        elif e.type == "judge_broadcast" and e.payload.get("visibility") == "public":
            msg = e.payload.get("message", "")
            phase = e.payload.get("phase", "")
            if phase in ("death_announce", "exile", "vote_result_announce",
                         "vote_tie_pk", "vote_second_tie",
                         "sheriff_elected", "sheriff_no_election"):
                summary_parts.append(f"[法官] {msg}")
        elif e.type in ("speech", "sheriff_speech"):
            speech_day = e.payload.get("day_number", 0)
            # Include speeches from previous days only (current day in transcript)
            if speech_day < current_day and speech_day > 0:
                speaker = e.payload.get("speaker", "?")
                text = e.payload.get("text", "")
                if text:
                    snippet = text[:SPEECH_SNIPPET_LEN]
                    if len(text) > SPEECH_SNIPPET_LEN:
                        snippet += "…"
                    summary_parts.append(f"[{speaker}] {snippet}")
        elif e.type == "vote_resolved":
            exiled = e.payload.get("exiled")
            reason = e.payload.get("reason", "")
            tied = e.payload.get("tied", [])
            if exiled:
                summary_parts.append(f"[投票结果] {exiled} 被放逐")
            elif reason == "second_tie_no_exile":
                summary_parts.append("[投票结果] 二次平票，无人出局")
            elif tied:
                summary_parts.append(f"[投票结果] 平票PK: {', '.join(tied)}")
        elif e.type == "idiot_reveal":
            summary_parts.append(f"[白痴亮牌] {e.payload.get('player_id', '?')} 亮出白痴身份")
        elif e.type == "hunter_shot_public":
            summary_parts.append(f"[猎人开枪] 猎人带走了 {e.payload.get('target_id', '?')}")

    # Trim from front to stay within budget
    total = sum(len(p) for p in summary_parts)
    while total > SUMMARY_BUDGET and len(summary_parts) > 1:
        dropped = summary_parts.pop(0)
        total -= len(dropped)
    public_summary = "\n".join(summary_parts) if summary_parts else ""

    # Build contradiction alerts from world state
    ctx_alerts: list[dict[str, Any]] = []
    must_address: list[dict[str, Any]] = []
    strategy_directive: dict[str, Any] = {}
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.contradiction import ContradictionEngine

        world_state = build_world_state(gs)
        contradiction_engine = ContradictionEngine()
        alerts = contradiction_engine.detect(world_state.facts, gs.day_number)

        # Filter to high-priority alerts
        for alert in alerts:
            if alert.priority == "high":
                alert_entry = {
                    "alert_type": alert.alert_type,
                    "player_id": alert.player_id,
                    "priority": alert.priority,
                    "description": alert.description,
                    "evidence": list(alert.evidence),
                }
                ctx_alerts.append(alert_entry)

        # Build must_address_alerts from high-priority alerts
        for alert in ctx_alerts:
            players = [p for p in alert["player_id"].split(",") if p]
            must_address.append({
                "alert_type": alert["alert_type"],
                "players": players,
                "public_evidence": alert["description"],
                "required_response": ["question", "side_with", "park"],
            })

        if must_address:
            strategy_directive = {
                "must_address_alerts": must_address,
                "directive": "你必须在发言中回应以下矛盾：选择站队、质疑、或明确表示暂不判断。",
            }
    except Exception:
        logger.debug("Contradiction alert building failed, skipping", exc_info=True)

    if legal_actions is None:
        legal_actions = []
    if legal_targets is None:
        legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != player_id]

    return AgentContext(
        agent_id=player_id,
        task_type=task_type,
        phase=gs.phase,
        day_number=gs.day_number,
        night_number=gs.day_number,
        own_role=player.role,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        public_summary=public_summary,
        visible_world_state=visible,
        recent_transcript=transcript,
        contradiction_alerts=ctx_alerts,
        strategy_directive=strategy_directive,
    )


def _action_result_to_dict(
    action: PlayerAction | FallbackAction,
) -> dict[str, Any]:
    """Convert a PlayerAction or FallbackAction to runtime state fields."""
    return {
        "action_type": action.action_type.value,
        "target_id": action.target_id,
        "speech": getattr(action, "speech", ""),
        "reason": getattr(action, "reason", ""),
    }


def agent_night_witch(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
) -> dict[str, Any] | None:
    """Try to get witch decision from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    witch_id = next(
        (pid for pid, p in gs.players.items() if p.role == "witch" and p.alive),
        None,
    )
    if witch_id is None:
        return None

    agent = registry.get_agent(witch_id)
    if agent is None:
        return None

    wolf_kill_target_id = state.get("wolf_kill_target_id")

    # Build legal actions for witch
    legal_actions = [ActionType.NO_ACTION]
    legal_targets: list[str] = []
    if wolf_kill_target_id and not gs.antidote_used:
        witch_cfg = engine.ruleset.raw["roles"]["witch"]["abilities"]
        if wolf_kill_target_id != witch_id or witch_cfg["antidote"].get("can_self_save", False):
            legal_actions.append(ActionType.USE_ANTIDOTE)
            legal_targets.append(wolf_kill_target_id)
    if not gs.poison_used:
        legal_actions.append(ActionType.USE_POISON)
        legal_targets.extend([
            pid for pid, p in gs.players.items()
            if p.alive and pid != witch_id
        ])

    context = build_agent_context(
        engine, gs, witch_id, TaskType.NIGHT_ACTION,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_kill_target_id=wolf_kill_target_id,
    )

    # Build witch strategy directive with clear action guidance
    witch_directive: dict[str, Any] = {
        "witch_night_action": (
            "你是女巫，现在是夜间行动阶段。你的选择：\n"
        ),
    }
    options = []
    if wolf_kill_target_id and not gs.antidote_used and ActionType.USE_ANTIDOTE in legal_actions:
        options.append(
            f"1) 使用解药救{wolf_kill_target_id}（他被狼人杀害了）—— action_type='use_antidote', target_id='{wolf_kill_target_id}'"
        )
    if not gs.poison_used and ActionType.USE_POISON in legal_actions:
        options.append(
            "2) 使用毒药毒杀某人 —— action_type='use_poison', target_id='目标玩家ID'"
        )
    options.append("3) 不使用药水 —— action_type='no_action'")
    witch_directive["witch_night_action"] += "\n".join(options)
    witch_directive["witch_night_action"] += (
        "\n\n重要规则：不能在同一夜同时使用解药和毒药。"
        "不能自救（解药不能救自己）。"
        "speech字段留空（夜间行动不需要发言）。"
    )

    # Add poison pressure targets if available
    poison_pressure = context.visible_world_state.get("poison_pressure_targets", [])
    if poison_pressure:
        pressure_desc = "; ".join(
            f"{p['player_id']}({p['pressure_type']}: {p['description']})"
            for p in poison_pressure
        )
        witch_directive["witch_pressure"] = f"存在毒药压力目标: {pressure_desc}"
        witch_directive["required_evaluation"] = "如果选择不用毒药，必须在reason中解释为什么不用。"

    context = context.model_copy(update={"strategy_directive": witch_directive})

    action, retry_info = agent.act(context)

    use_antidote = action.action_type == ActionType.USE_ANTIDOTE
    poison_target_id = action.target_id if action.action_type == ActionType.USE_POISON else None

    return {
        "use_antidote": use_antidote,
        "poison_target_id": poison_target_id,
        "witch_action_trace": _action_trace_payload(action),
    }


def agent_night_seer(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
) -> dict[str, Any] | None:
    """Try to get seer decision from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    seer_id = next(
        (pid for pid, p in gs.players.items() if p.role == "seer" and p.alive),
        None,
    )
    if seer_id is None:
        return None

    agent = registry.get_agent(seer_id)
    if agent is None:
        return None

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != seer_id]

    # Build seer strategy: follow badge flow plan from election speech
    badge_flow_next = None
    for e in gs.events:
        if e.type == "sheriff_speech" and e.payload.get("speaker") == seer_id:
            # Try to extract badge flow targets from the speech text
            text = e.payload.get("text", "")
            # Simple extraction: find player IDs mentioned near "验" or "警徽流"
            import re
            mentioned = re.findall(r'p\d+', text)
            if mentioned:
                badge_flow_next = [pid for pid in mentioned if pid in legal_targets]
            break

    night_num = gs.night_number
    if night_num == 1:
        seer_guidance = (
            "第一夜验人策略：选择你最怀疑的人，或者按照你上警时承诺的警徽流第一夜验人对象。"
            "如果上警时没有明确指定，优先验发言最少、最不透明的人。"
        )
    else:
        seer_guidance = (
            f"第{night_num}夜验人策略：根据白天讨论中你最怀疑的人选择查验目标。"
            "优先验：1) 发言前后矛盾的人；2) 站边不明确的人；3) 被多人怀疑但你不确定的人。"
            "不要验你已经确认的好人。"
        )

    strategy_directive = {
        "seer_night_check": (
            "你是预言家，现在是夜间验人阶段。你必须选择一名玩家查验其身份。"
            "验人结果（好人/狼人）将在明天白天得知。"
            f"\n\n{seer_guidance}"
            "\n\n注意：本局没有守卫，预言家无法被守护，必须谨慎选择。"
            "speech字段留空（夜间行动不需要发言）。"
        ),
    }
    if badge_flow_next:
        strategy_directive["badge_flow_plan"] = (
            f"你在警上承诺的警徽流计划中提到的验人对象: {badge_flow_next}，"
            "请优先按此计划验人以保持信息传递的一致性。"
        )

    context = build_agent_context(
        engine, gs, seer_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
    )
    context = context.model_copy(update={"strategy_directive": strategy_directive})

    action, retry_info = agent.act(context)

    seer_target_id = action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    return {
        "seer_target_id": seer_target_id,
        "seer_action_trace": _action_trace_payload(action),
    }


def agent_wolf_consensus(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
) -> dict[str, Any] | None:
    """Try to get wolf consensus from agents. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    wolves = [pid for pid, p in gs.players.items() if p.role == "werewolf" and p.alive]
    if not wolves:
        return None

    # Collect votes from ALL alive wolves
    kill_votes: dict[str, int] = {}
    no_kill_count = 0
    action_traces: dict[str, Any] = {}

    for wolf_id in wolves:
        vote = _single_wolf_vote(state, engine, registry, wolf_id)
        if vote is None:
            no_kill_count += 1
            continue
        if vote.get("action_trace"):
            action_traces[wolf_id] = vote["action_trace"]
        if vote.get("wolf_action") == "kill" and vote.get("wolf_kill_target_id"):
            target = vote["wolf_kill_target_id"]
            kill_votes[target] = kill_votes.get(target, 0) + 1
        else:
            no_kill_count += 1

    total_kill = sum(kill_votes.values())
    if total_kill > no_kill_count and kill_votes:
        best_target = max(kill_votes, key=kill_votes.get)
        return {"wolf_action": "kill", "wolf_kill_target_id": best_target,
                "wolf_action_reason": f"majority({total_kill}/{total_kill + no_kill_count})",
                "action_traces": action_traces}
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None,
            "wolf_action_reason": f"no_kill_majority({no_kill_count}/{total_kill + no_kill_count})",
            "action_traces": action_traces}


def _single_wolf_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
) -> dict[str, Any] | None:
    """Get a single wolf's kill/no_kill vote.

    Each wolf gets an individual timeout.  Unknown action types are treated
    as wolf_no_kill (the agent chose something unexpected) rather than
    silently swallowed as None which would distort consensus.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]
    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=legal_targets,
        wolf_team_plan=state.get("wolf_team_plan"),
    )

    timeout = float(state.get("wolf_vote_timeout") or AGENT_TIMEOUTS.wolf_consensus)
    if timeout > 0:
        from werewolf_agent.runtime.timers import timed_call
        action_result = timed_call(agent.act, context, timeout=timeout, fallback=None)
    else:
        try:
            action_result = agent.act(context)
        except Exception as exc:
            logger.warning("Wolf vote failed for %s: %s: %s", wolf_id, type(exc).__name__, exc)
            action_result = None

    if action_result is None:
        # Timeout or exception — count as no_kill (strategy: skip this vote)
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None}

    action, retry_info = action_result
    action_trace = _action_trace_payload(action)

    if action.action_type == ActionType.WOLF_NO_KILL:
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
    if action.action_type == ActionType.WOLF_KILL and action.target_id:
        return {"wolf_action": "kill", "wolf_kill_target_id": action.target_id, "action_trace": action_trace}
    # Unknown action type — treat as no_kill rather than silently returning None
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}


def agent_wolf_discussion(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
) -> dict[str, Any] | None:
    """Get wolf's private discussion speech. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    # Build round-specific requirements from wolf_strategy
    from werewolf_agent.runtime.wolf_strategy import round_requirements
    round_num = state.get("wolf_discussion_round", 1)
    requirements = round_requirements(gs.night_number, round_num)

    # Collect all wolf discussion speeches already in this night's events
    wolf_ids = [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]
    prev_speeches: list[dict[str, str]] = []
    for e in gs.events:
        if e.type == "wolf_discussion" and e.payload.get("wolf_id") in wolf_ids:
            prev_speeches.append({
                "wolf_id": e.payload.get("wolf_id", ""),
                "round": str(e.payload.get("round", "")),
                "text": e.payload.get("text", ""),
            })

    # Build readable transcript of teammate speeches for prompt injection
    wolf_teammates = [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf" and pid != wolf_id
    ]
    teammate_speeches = [s for s in prev_speeches if s["wolf_id"] != wolf_id]
    transcript_lines = []
    for s in teammate_speeches[-12:]:  # Last 12 teammate speeches
        transcript_lines.append(f"[第{s['round']}轮 {s['wolf_id']}]: {s['text']}")
    teammate_transcript = "\n".join(transcript_lines)

    # Build discussion instruction based on round and teammate content
    has_teammate_input = bool(teammate_speeches)
    discussion_instruction = (
        "这是狼队密谈，只有狼人队友能看到。你必须以狼人视角发言，讨论狼队策略。"
        "禁止假装好人视角发言，禁止质疑或试探队友身份——你清楚知道谁是队友。"
        "必须发言，不能沉默。必须提出具体的击杀目标或战术建议。"
    )
    if has_teammate_input:
        discussion_instruction += (
            "\n\n重要：你必须回应队友的发言！看看队友提出了什么建议，"
            "表示同意、反对或补充意见，形成真正的团队讨论，而不是自顾自发言。"
        )

    strategy_directive = {
        "wolf_team_discussion": discussion_instruction,
        "round_focus": requirements.get("required", "讨论狼队策略。"),
        "wolf_teammates": wolf_teammates,
        "previous_discussion": prev_speeches[-8:],
    }

    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
    )

    # Inject teammate transcript into recent_transcript for prompt visibility
    extra_transcript = []
    for s in teammate_speeches[-6:]:
        extra_transcript.append({
            "speaker": s["wolf_id"],
            "text": s["text"],
        })
    merged_transcript = extra_transcript + list(context.recent_transcript)
    context = context.model_copy(update={
        "strategy_directive": strategy_directive,
        "recent_transcript": merged_transcript[-8:],
    })

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Reject empty/silent wolf speeches — retry with fallback
    if not speech_text.strip():
        alive_non_wolves = [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]
        fallback_target = alive_non_wolves[0] if alive_non_wolves else ""
        speech_text = (
            f"我是{wolf_id}，本轮讨论我认为应该刀{fallback_target}。"
            f"{requirements.get('required', '')}请大家发表意见。"
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
) -> dict[str, Any] | None:
    """Try to get day speech from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, speaker_id, TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH, ActionType.NO_ACTION],
        wolf_team_plan=state.get("wolf_team_plan"),
    )

    # Sheriff gets 归票 (vote push) directive
    strategy_directive = context.strategy_directive or {}
    if gs.sheriff_id == speaker_id and gs.sheriff_badge_state == "active":
        alive_others = [pid for pid, p in gs.players.items() if p.alive and pid != speaker_id]
        strategy_directive["sheriff_vote_push"] = (
            "你是警长，你的发言需要归票：总结本轮讨论的关键信息点，"
            "明确表态你怀疑谁、要推谁，号召大家集中投票。"
            "警长归票是核心职责，不能含糊其辞。"
        )
        strategy_directive["sheriff_alive_others"] = alive_others

    # Include sheriff election speeches as salience items for day 1 discussion
    sheriff_speeches = []
    for e in gs.events:
        if e.type == "sheriff_speech" and e.payload.get("text"):
            sheriff_speeches.append({
                "speaker": e.payload.get("speaker", ""),
                "text": e.payload.get("text", ""),
            })
    if sheriff_speeches:
        strategy_directive["sheriff_election_record"] = (
            "以下是警上竞选环节各候选人的发言，请参考这些信息进行讨论：\n"
            + "\n".join(f"  [{s['speaker']}]: {s['text']}" for s in sheriff_speeches)
        )

    context = context.model_copy(update={"strategy_directive": strategy_directive})

    action, retry_info = agent.act(context)

    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_sheriff_pick_speech_order(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
) -> list[str] | None:
    """Ask the sheriff agent to choose the first speaker. Returns full speech order or None."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_players = [pid for pid, p in gs.players.items() if p.alive and pid != sheriff_id]
    if not alive_players:
        return None

    context = build_agent_context(
        engine, gs, sheriff_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=alive_players,
        wolf_team_plan=state.get("wolf_team_plan"),
    )
    strategy_directive = {
        "choose_speech_order": (
            "你是警长，需要选择发言顺序。请选择第一个发言的玩家（你将最后一个发言进行归票）。"
            "在speech字段中说明你的选择理由。"
        ),
        "alive_players": alive_players,
    }
    context = context.model_copy(update={"strategy_directive": strategy_directive})

    # Use VOTE action to pick a target (first speaker)
    context = context.model_copy(update={
        "legal_actions": [ActionType.VOTE],
        "legal_targets": alive_players,
    })

    action, retry_info = agent.act(context)
    first_speaker = action.target_id if action.action_type == ActionType.VOTE else None

    if first_speaker and first_speaker in alive_players:
        # Build order: first_speaker, then remaining in original order, sheriff last
        remaining = [pid for pid in alive_players if pid != first_speaker]
        return [first_speaker] + remaining + [sheriff_id]
    return None


def agent_pk_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
) -> dict[str, Any] | None:
    """Get PK speech from a tied candidate. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    # Build context with prior vote tally info
    prior_tally = {}
    for e in gs.events:
        if e.type == "vote_resolved":
            prior_tally = e.payload
            break

    context = build_agent_context(
        engine, gs, speaker_id, TaskType.PK_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
    )
    # Add prior tally to visible state
    if prior_tally:
        updated_visible = {**context.visible_world_state, "prior_vote_tally": prior_tally}
        context = context.model_copy(update={"visible_world_state": updated_visible})

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
) -> dict[str, Any] | None:
    """Try to get vote from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(voter_id)
    if agent is None:
        return None

    allow_abstain = engine.ruleset.raw["day_flow"]["vote"].get("allow_abstain", False)
    legal_actions = [ActionType.VOTE]
    if allow_abstain:
        legal_actions.append(ActionType.NO_ACTION)

    legal_targets = engine.legal_exile_targets(gs)
    if state.get("revote") and state.get("pk_candidates"):
        pk_candidates = set(state.get("pk_candidates") or [])
        legal_targets = [pid for pid in legal_targets if pid in pk_candidates]

    # Pass consecutive no-exile info as strategy directive
    consecutive_no_exile = state.get("consecutive_no_exile_days", 0)
    strategy_directive: dict[str, Any] = {
        "vote_silent": (
            "投票阶段不允许公开发言。speech字段必须留空。"
            "你只能内心选择要投谁，不能在投票时发表任何公开言论。"
            "请在reason字段中写下你的内心理由（只有你自己能看到）。"
        ),
        "vote_strategy": (
            "投票原则：\n"
            "1) 有查杀走查杀：如果被信任的预言家查杀了某人，优先投查杀对象。"
            "2) 跟预言家走：听完发言后，根据你信任的预言家的归票方向投票。"
            "3) 警上单边预言家可信度高，警下跳预言家的可信度很低。"
            "4) 如果没有明确查杀，投发言最可疑、逻辑最不通的人。"
        ),
    }
    if not allow_abstain:
        parts = ["必须投票选出一名玩家放逐，不能弃票。"]
        if consecutive_no_exile > 0:
            parts.append(f"已经连续{consecutive_no_exile}天无人出局，必须做出决定。")
        strategy_directive["vote_pressure"] = " ".join(parts)

    context = build_agent_context(
        engine, gs, voter_id, TaskType.VOTE,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_team_plan=state.get("wolf_team_plan"),
    )
    if strategy_directive:
        context = context.model_copy(update={"strategy_directive": strategy_directive})

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    # Fallback: if agent returned wrong action type but has legal targets,
    # pick the first one rather than abstaining silently
    if target is None and legal_targets:
        target = legal_targets[0]
    speech = getattr(action, "speech", "") or ""
    reason = getattr(action, "reason", "") or ""
    trace = getattr(action, "trace", None)
    return {
        "vote_target": target,
        "vote_speech": speech,
        "vote_reason": reason,
        "action_trace": trace.model_dump() if trace else None,
    }


def agent_hybrid_choose_master(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hybrid_id: str,
) -> dict[str, Any] | None:
    """Ask hybrid agent to choose their master. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hybrid_id)
    if agent is None:
        return None

    candidates = [pid for pid, p in gs.players.items() if p.alive and pid != hybrid_id]
    context = build_agent_context(
        engine, gs, hybrid_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHOOSE_MASTER],
        legal_targets=candidates,
    )
    strategy_directive = {
        "hybrid_master_choice": (
            "你是混血儿，第一夜需要选择一名玩家作为你的主人。"
            "你不知道主人的身份和阵营，但你将跟随主人的阵营获胜。"
            "你可以自由选择任何玩家作为主人——可以根据直觉、位置、或任何你喜欢的理由。"
            "选择后不能更改。speech字段留空。"
        ),
    }
    context = context.model_copy(update={"strategy_directive": strategy_directive})

    action, retry_info = agent.act(context)
    master_target_id = action.target_id if action.action_type == ActionType.CHOOSE_MASTER else None
    if master_target_id is None and candidates:
        master_target_id = candidates[0]

    return {"master_target_id": master_target_id}


def agent_hunter_shot(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hunter_id: str,
) -> str | None:
    """Get hunter shot target from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hunter_id)
    if agent is None:
        return None

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != hunter_id]
    context = build_agent_context(
        engine, gs, hunter_id, TaskType.HUNTER_SHOT,
        legal_actions=[ActionType.HUNTER_SHOT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
    )

    action, retry_info = agent.act(context)
    if action.action_type == ActionType.HUNTER_SHOT and action.target_id:
        return action.target_id
    return None


def agent_sheriff_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    candidates: list[str],
) -> dict[str, Any] | None:
    """Get sheriff vote from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(voter_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine, gs, voter_id, TaskType.VOTE,
        legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
        legal_targets=candidates,
    )

    action, retry_info = agent.act(context)
    target = action.target_id if action.action_type == ActionType.SHERIFF_VOTE else None
    return {
        "vote_target": target,
        "action_trace": _action_trace_payload(action),
    }


def agent_sheriff_register(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
) -> bool:
    """Ask a player whether they want to register for sheriff election.

    Returns True if the player registers, False otherwise.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return False

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    # Build role-specific registration guidance
    if player_role == "seer":
        role_hint = (
            "你是预言家，强烈建议上警！预言家几乎必须上警留警徽流，"
            "这是预言家的核心玩法——通过警徽流传递验人信息。"
        )
    elif player_role == "werewolf":
        # Check if any wolf is already planning to claim seer
        role_hint = (
            "你是狼人。如果团队安排你悍跳预言家，你必须上警与真预言家对抗。"
            "如果不悍跳，也可以上警发言获取信息或带节奏。"
        )
    else:
        # Good player (non-seer)
        role_hint = (
            "你是好人（非预言家），可以考虑上警发言表达观点、压制狼人发言空间。"
            "但注意：如果你不是预言家，不要在警上冒充预言家抢警徽，"
            "这会干扰真预言家的信息传递。上警主要目的是发言和表达立场。"
        )

    strategy_directive = {
        "sheriff_registration": (
            f"{role_hint}\n"
            "上警意味着你将在竞选环节发言，争取警长职位或表达观点。"
            "不上警则留在警下投票选出警长。"
        ),
    }

    context = build_agent_context(
        engine, gs, player_id, TaskType.SHERIFF_REGISTRATION,
        legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
    )
    context = context.model_copy(update={"strategy_directive": strategy_directive})

    try:
        action, retry_info = agent.act(context)
        return action.action_type == ActionType.SHERIFF_REGISTER
    except Exception:
        logger.warning("Sheriff registration failed for %s", player_id, exc_info=True)
        return False


def agent_sheriff_withdraw(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
) -> bool:
    """Ask a sheriff candidate whether they want to withdraw.

    Returns True if the candidate withdraws, False if they stay.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return False

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
    )

    try:
        action, retry_info = agent.act(context)
        return action.action_type == ActionType.SHERIFF_WITHDRAW
    except Exception:
        logger.warning("Sheriff withdrawal failed for %s", candidate_id, exc_info=True)
        return False


def agent_sheriff_election_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    all_candidates: list[str],
) -> dict[str, Any] | None:
    """Get sheriff election speech from a candidate.

    The speech must explain why the candidate is running for sheriff,
    their badge flow plan (警徽流), and their initial stance.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    other_candidates = [c for c in all_candidates if c != candidate_id]

    # Badge flow is seer-exclusive. Only seer (or wolf claiming seer) should mention it.
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    is_seer_or_claiming = player_role == "seer" or player_role == "werewolf"

    badge_flow_instruction = ""
    if is_seer_or_claiming:
        badge_flow_instruction = (
            "2) 你的警徽流：必须留两个晚上的验人对象！"
            "格式如'先验X，后验Y'。如果你验到好人，死后警徽给该好人；"
            "如果验到狼人，则不给警徽（撕徽或给之前验过的好人）。"
            "警徽流是预言家传递信息的核心机制，必须明确留出两夜验人计划。"
        )

    # Single-sided vs multi-seer context
    seer_count = sum(1 for c in all_candidates
                     if gs.players.get(c) and gs.players[c].role in ("seer", "werewolf"))
    seer_context = ""
    if seer_count >= 2:
        seer_context = (
            "场上有多人跳预言家，这是典型的悍跳局面。"
            "真预言家必须坚定立场，用逻辑和验人信息证明自己；"
            "悍跳预言家需要制造合理怀疑，攻击对方的逻辑漏洞。"
        )
    else:
        if player_role in ("seer", "werewolf"):
            seer_context = (
                "目前警上只有你跳预言家（单边预言家），你的可信度很高。"
                "要充分利用这一点，留下完整的警徽流，让好人信任你。"
            )

    strategy_directive = {
        "sheriff_election_speech": (
            "你正在竞选警长，必须解释上警原因。发言必须包含："
            "1) 为什么你适合当警长（身份声明、逻辑能力等）；"
            f"{badge_flow_instruction}"
            "3) 你对场上局势的初步判断。"
            "不能只说'我来上警'之类空洞的话，必须有实质内容。"
            "注意：只有预言家（或悍跳预言家）才能留警徽流，其他身份不要提警徽流。"
            "非预言家不要在警上冒充预言家抢警徽。"
            f"{seer_context}"
        ),
        "other_candidates": other_candidates,
    }

    context = build_agent_context(
        engine, gs, candidate_id, TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
    )
    context = context.model_copy(update={"strategy_directive": strategy_directive})

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Reject empty sheriff election speeches
    if not speech_text.strip() or len(speech_text.strip()) < 10:
        speech_text = (
            f"我上警是因为我需要通过警徽流传递关键信息。"
            f"我的警徽流暂定先看{other_candidates[0] if other_candidates else '待定'}。"
            f"希望大家支持我当选警长。"
        )

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}
