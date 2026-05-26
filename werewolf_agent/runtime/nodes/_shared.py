"""Shared utilities, types, and constants for game graph nodes."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import uuid
from dataclasses import replace
from typing import Any, TypedDict

from werewolf_agent.core.models import (
    Death,
    GameEvent,
    GameState,
    PlayerState,
    VictoryResult,
)
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    AgentRegistry,
    agent_badge_decision,
    agent_day_vote,
    agent_exile_last_words,
    agent_hunter_shot,
    agent_hybrid_choose_master,
    agent_night_seer,
    agent_night_witch,
    agent_pk_speech,
    agent_sheriff_election_speech,
    agent_sheriff_pick_speech_order,
    agent_sheriff_register,
    agent_sheriff_vote,
    agent_sheriff_withdraw,
    agent_wolf_consensus,
    agent_wolf_discussion,
)
from werewolf_agent.runtime.timers import timed_call
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import detect_timeline_confusion, phase_label

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime state — LangGraph operates on this dict, wraps immutable GameState
# ---------------------------------------------------------------------------

class RuntimeState(TypedDict, total=False):
    game_state: GameState
    engine: RuleEngine
    # Night action inputs (set by scripted/agent nodes before resolve)
    wolf_kill_target_id: str | None
    wolf_action: str
    wolf_action_reason: str
    use_antidote: bool
    poison_target_id: str | None
    seer_target_id: str | None
    seer_action_trace: dict[str, Any]
    witch_action_trace: dict[str, Any]
    hybrid_master_target_id: str | None
    # Day action inputs
    self_destruct_wolf_id: str | None
    current_speaker_id: str | None
    speech_order: list[str]
    speech_index: int
    speech_text: str
    speech_timed_out: bool
    speech_seconds_limit: int
    # Vote inputs
    exile_votes: dict[str, str]
    exile_vote_day: int
    exile_vote_revote: bool
    pk_candidates: list[str]
    vote_action_traces: dict[str, Any]
    revote: bool
    # Sheriff election inputs
    sheriff_candidates: list[str]
    sheriff_votes: dict[str, str]
    sheriff_withdrawing: list[str]
    # Badge decision after sheriff death
    badge_decision: str | None  # "transfer", "tear", or None to ask the agent
    badge_target_id: str | None
    # Hunter shot
    hunter_shot_target_id: str | None
    # Agent registry: when provided, nodes delegate to PlayerAgent
    agent_registry: Any  # AgentRegistry protocol, optional
    # RAG knowledge service: retrieves strategy hints for agent contexts
    rag_service: Any
    # Runtime flow-control timer; must not adjudicate RuleEngine truth
    runtime_timer: Any
    # Anti-stall: consecutive days with no exile from vote
    consecutive_no_exile_days: int
    # Per-call timeout (seconds) for agent provider calls; 0 = no timeout
    agent_call_timeout: float
    wolf_discussion_round: int
    wolf_team_plan: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _new_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


def _player_ids(gs: GameState) -> list[str]:
    return list(gs.players.keys())


def _alive_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]


def _alive_non_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]


def _force_wolf_kill(gs: GameState, reason: str) -> dict[str, Any]:
    """Force a wolf kill on a random alive non-wolf (consecutive no-kill cap)."""
    import random as _random
    non_wolves = _alive_non_wolves(gs)
    if not non_wolves:
        event = GameEvent(type="wolf_no_kill_timeout", payload={"night_number": gs.night_number})
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": None}
    rng = _random.Random(_stable_seed(gs.game_id, reason, gs.night_number))
    target = rng.choice(non_wolves)
    event = GameEvent(
        type="wolf_kill_selected",
        payload={"night_number": gs.night_number, "target_id": target, "reason": reason},
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs, "wolf_kill_target_id": target}


def _find_role(gs: GameState, role: str) -> str | None:
    return next((pid for pid, p in gs.players.items() if p.role == role and p.alive), None)


def _timer_expired(state: RuntimeState, key: str) -> bool:
    timer = state.get("runtime_timer")
    if timer is None:
        return False
    expired = getattr(timer, "expired", None)
    if expired is None:
        return False
    return bool(expired(key))


def _agent_timeout(state: RuntimeState) -> float:
    """Return per-call agent timeout in seconds; 0 means no wrapping."""
    return float(state.get("agent_call_timeout") or 0)


def _player_display(state: RuntimeState, player_id: str) -> str:
    """Return '名字(pid)' for display, e.g. '陈思远(p01)'."""
    registry = state.get("agent_registry")
    if registry:
        agent = registry.get_agent(player_id)
        if agent and hasattr(agent, "player_name") and agent.player_name != player_id:
            return f"{agent.player_name}({player_id})"
    return player_id


def _call_agent(fn, state: RuntimeState, *args, timeout_override: float | None = None):
    """Call an agent adapter function, optionally wrapped with a timeout."""
    timeout = timeout_override if timeout_override is not None else _agent_timeout(state)
    if timeout > 0:
        return timed_call(fn, *args, timeout=timeout)
    return fn(*args)


def _dispatch_agent(
    state: RuntimeState,
    fn,
    *extra_args,
    timeout_override: float | None = None,
) -> dict[str, Any] | None:
    """Helper to dispatch to an agent after checking registry existence.

    Consolidates registry validation and wraps standard parameter packing.
    """
    registry = state.get("agent_registry")
    if not registry:
        return None
    engine = state["engine"]
    return _call_agent(
        fn,
        state,
        state,
        engine,
        registry,
        *extra_args,
        timeout_override=timeout_override,
    )


def _action_trace_event(
    *,
    player_id: str,
    phase: str,
    action_trace: dict[str, Any],
    day_number: int = 0,
    night_number: int = 0,
) -> GameEvent:
    audit_text_parts: list[str] = []
    raw_text = action_trace.get("raw_text")
    if raw_text:
        audit_text_parts.append(str(raw_text))
    parsed_action = action_trace.get("parsed_action") or {}
    if isinstance(parsed_action, dict):
        for key in ("reason", "speech_text", "private_reason"):
            value = parsed_action.get(key)
            if value:
                audit_text_parts.append(str(value))
    timeline_confusion = detect_timeline_confusion("\n".join(audit_text_parts))
    payload = {
        "player_id": player_id,
        "phase": phase,
        "day_number": day_number,
        "night_number": night_number,
        "visibility": "moderator_only",
        "action_trace": action_trace,
        "timeline_confusion": timeline_confusion,
    }
    if phase == "vote":
        payload.update(_private_vote_audit_payload(action_trace))

    return GameEvent(type="action_trace_audit", payload=payload)


def _private_vote_audit_payload(action_trace: dict[str, Any]) -> dict[str, Any]:
    parsed = action_trace.get("parsed_action") or {}
    if not isinstance(parsed, dict):
        parsed = {}
    target = (
        parsed.get("target_id")
        or parsed.get("target")
        or action_trace.get("target_id")
        or action_trace.get("target")
    )
    thought = {
        "target": target,
        "public_reason": str(parsed.get("reason") or action_trace.get("reason") or "")[:300],
        "standing_with_seer": str(parsed.get("standing_with_seer") or "")[:100],
        "suspect_reason": str(parsed.get("suspect_reason") or "")[:300],
        "not_voting_reason": str(parsed.get("not_voting_reason") or "")[:300],
        "private_reason": str(parsed.get("private_reason") or "")[:500],
    }
    return {
        "vote_target": target,
        "private_vote_thought": thought,
    }


def _public_vote_reason(action_trace: dict[str, Any] | None) -> str:
    if not action_trace:
        return ""
    parsed = action_trace.get("parsed_action") or {}
    reason = (
        parsed.get("reason")
        or action_trace.get("reason")
        or action_trace.get("fallback_reason")
        or ""
    )
    return str(reason)[:200]


def _with_vote_target_in_trace(
    action_trace: dict[str, Any],
    target_id: str,
) -> dict[str, Any]:
    parsed = action_trace.get("parsed_action")
    if isinstance(parsed, dict):
        return {
            **action_trace,
            "target_id": target_id,
            "parsed_action": {**parsed, "target_id": parsed.get("target_id") or target_id},
        }
    return {**action_trace, "target_id": target_id}


def _judge_broadcast(
    *,
    phase: str,
    message: str,
    gs: GameState,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    visibility: str = "public",
) -> tuple[GameState, GameEvent]:
    """Create a judge broadcast event and append to game state."""
    payload: dict[str, Any] = {
        "phase": phase,
        "message": message,
        "day_number": day_number,
        "night_number": night_number,
        "visibility": visibility,
    }
    if night_number > 0:
        payload["phase_label"] = phase_label("night", night_number)
    elif day_number > 0:
        payload["phase_label"] = phase_label("day", day_number)
    if extra_payload:
        payload.update(extra_payload)
    event = GameEvent(type="judge_broadcast", payload=payload)
    gs = replace(gs, events=gs.events + [event])
    return gs, event


def _ensure_day_incremented(gs: GameState) -> tuple[GameState, int]:
    """Increment day_number and broadcast '天亮了' if not already done.

    Returns (updated_gs, current_day_number).
    """
    if gs.phase != "day":
        d = gs.day_number + 1
        label = phase_label("day", d)
        gs, _ = _judge_broadcast(phase="day_announce", message=f"{label}：天亮了", gs=gs, day_number=d)
        gs = replace(gs, phase="day", day_number=d,
                     events=gs.events + [GameEvent(type="day_announce", payload={"day": d})])
        return gs, d
    return gs, gs.day_number


def _build_wolf_team_plan(
    gs: GameState,
    *,
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wolves = _alive_wolves(gs)
    if not wolves:
        return {}

    assignments = list(wolves)
    roles = {
        "fake_seer": assignments[0] if len(assignments) > 0 else None,
        "pusher": assignments[1] if len(assignments) > 1 else None,
        "hooker": assignments[2] if len(assignments) > 2 else None,
        "deep_cover": assignments[3] if len(assignments) > 3 else None,
    }
    previous_plan = previous_plan or {}
    can_reuse_previous = previous_plan.get("evidence_quality") not in (None, "none")
    primary = _first_alive_target(gs, previous_plan.get("night_kill_primary")) if can_reuse_previous else None
    backup = _first_alive_target(gs, previous_plan.get("night_kill_backup")) if can_reuse_previous else None
    day_push = _first_alive_target(gs, previous_plan.get("day_push_target")) if can_reuse_previous else None

    return {
        "night_number": gs.night_number,
        **roles,
        "night_kill_primary": primary,
        "night_kill_backup": backup,
        "day_push_target": day_push,
        "evidence_from_discussion": previous_plan.get("evidence_from_discussion", []),
        "evidence_quality": previous_plan.get("evidence_quality", "none") if can_reuse_previous else "none",
        "public_story": "警上制造预言家对立，冲锋位打抗推目标，倒钩位保留质疑队友空间，深水位做中立复盘。",
        "hooking_intent": {
            "player_id": roles.get("hooker"),
            "policy": "可以轻踩或投票队友换取好人信任，但公开文本必须表现为独立逻辑判断。",
        },
    }


def _first_alive_target(gs: GameState, player_id: str | None) -> str | None:
    if player_id is None:
        return None
    player = gs.players.get(player_id)
    if player and player.alive and player.role != "werewolf":
        return player_id
    return None


def _planned_wolf_kill(state: RuntimeState) -> dict[str, Any] | None:
    gs: GameState = state["game_state"]
    plan = state.get("wolf_team_plan") or {}
    if plan.get("evidence_quality") == "none":
        return None
    evidence = plan.get("evidence_from_discussion") or []
    for key in ("night_kill_primary", "night_kill_backup"):
        target = _first_alive_target(gs, plan.get(key))
        if target:
            has_target_evidence = any(item.get("target") == target for item in evidence)
            if not has_target_evidence and plan.get("evidence_quality") != "strong":
                continue
            logger.debug(f"  [狼人决策] 按狼队计划击杀: {_player_display(state, target)}")
            event = GameEvent(
                type="wolf_kill_selected",
                payload={
                    "night_number": gs.night_number,
                    "target_id": target,
                    "reason": "wolf_team_plan",
                    "plan_key": key,
                },
            )
            gs = replace(gs, events=gs.events + [event])
            return {"game_state": gs, "wolf_kill_target_id": target}
    return None


def _sheriff_died_this_batch(gs: GameState) -> bool:
    """Check if the sheriff died in the current resolution batch."""
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return False
    sheriff = gs.players.get(gs.sheriff_id)
    return sheriff is not None and not sheriff.alive


def _needs_sheriff_before_deaths(gs: GameState) -> bool:
    """N1: sheriff election should run before death announcement."""
    return (
        gs.night_number == 1
        and gs.sheriff_interrupt_count == 0
        and gs.sheriff_id is None
        and gs.sheriff_badge_state not in ("torn", "active")
    )


def _deaths_already_announced(gs: GameState) -> bool:
    """Check if death_announce broadcast already exists for current day."""
    return any(
        e.type == "judge_broadcast"
        and e.payload.get("phase") == "death_announce"
        and e.payload.get("day_number") == gs.day_number
        for e in gs.events
    )
