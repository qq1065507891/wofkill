"""Agent runtime adapter: converts GameState into AgentContext for PlayerAgent.

When an AgentRegistry is provided to the runtime graph, night/day nodes will
delegate decisions to PlayerAgent instances. Without a registry, deterministic
scripted fallback is used (preserving existing test behavior).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
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


def build_agent_context(
    engine: RuleEngine,
    gs: GameState,
    player_id: str,
    task_type: TaskType,
    *,
    legal_actions: list[ActionType] | None = None,
    legal_targets: list[str] | None = None,
    wolf_kill_target_id: str | None = None,
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
    elif player.role == "hybrid" and gs.hybrid_master_id:
        visible["master_id"] = gs.hybrid_master_id

    # Build recent transcript from public speech events
    transcript: list[dict[str, Any]] = []
    for e in reversed(gs.events):
        if e.type == "speech" and len(transcript) < 6:
            transcript.insert(0, {
                "speaker": e.payload.get("speaker", ""),
                "text": e.payload.get("text", ""),
            })

    if legal_actions is None:
        legal_actions = []
    if legal_targets is None:
        legal_targets = [pid for pid, p in gs.players.items() if p.alive and pid != player_id]

    return AgentContext(
        agent_id=player_id,
        task_type=task_type,
        phase=gs.phase,
        day_number=gs.day_number,
        night_number=gs.night_number,
        own_role=player.role,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        visible_world_state=visible,
        recent_transcript=transcript,
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

    action, retry_info = agent.act(context)

    use_antidote = action.action_type == ActionType.USE_ANTIDOTE
    poison_target_id = action.target_id if action.action_type == ActionType.USE_POISON else None

    return {"use_antidote": use_antidote, "poison_target_id": poison_target_id}


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
    context = build_agent_context(
        engine, gs, seer_id, TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
    )

    action, retry_info = agent.act(context)

    seer_target_id = action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    return {"seer_target_id": seer_target_id}


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

    for wolf_id in wolves:
        vote = _single_wolf_vote(state, engine, registry, wolf_id)
        if vote is None:
            no_kill_count += 1
            continue
        if vote.get("wolf_action") == "kill" and vote.get("wolf_kill_target_id"):
            target = vote["wolf_kill_target_id"]
            kill_votes[target] = kill_votes.get(target, 0) + 1
        else:
            no_kill_count += 1

    total_kill = sum(kill_votes.values())
    if total_kill > no_kill_count and kill_votes:
        best_target = max(kill_votes, key=kill_votes.get)
        return {"wolf_action": "kill", "wolf_kill_target_id": best_target,
                "wolf_action_reason": f"majority({total_kill}/{total_kill + no_kill_count})"}
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None,
            "wolf_action_reason": f"no_kill_majority({no_kill_count}/{total_kill + no_kill_count})"}


def _single_wolf_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
) -> dict[str, Any] | None:
    """Get a single wolf's kill/no_kill vote."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    legal_targets = [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]
    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=legal_targets,
    )

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.WOLF_NO_KILL:
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None}
    if action.action_type == ActionType.WOLF_KILL and action.target_id:
        return {"wolf_action": "kill", "wolf_kill_target_id": action.target_id}
    return None


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

    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH, ActionType.NO_ACTION],
    )

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text}


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
    )

    action, retry_info = agent.act(context)

    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text}


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

    legal_targets = engine.legal_exile_targets(gs)
    context = build_agent_context(
        engine, gs, voter_id, TaskType.VOTE,
        legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
        legal_targets=legal_targets,
    )

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    return {"vote_target": target}


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
