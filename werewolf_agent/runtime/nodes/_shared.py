"""Shared utilities, types, and constants for game graph nodes."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
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

RULESET_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml")

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
    # Restored MemoryStore from previous game (cross-game learning)
    restored_memory: Any
    # Runtime flow-control timer; must not adjudicate RuleEngine truth
    runtime_timer: Any
    # Anti-stall: consecutive days with no exile from vote
    consecutive_no_exile_days: int
    # Per-call timeout (seconds) for agent provider calls; 0 = no timeout
    agent_call_timeout: float
    wolf_discussion_round: int
    wolf_team_plan: dict[str, Any]
    # Game repository for persistent storage (PostgresGameRepository etc.)
    repository: Any
    # Per-player discussion position summaries
    discussion_positions: list[dict[str, Any]]
    judge_agent: Any
    judge_llm_enabled: bool
    judge_hitl: Any
    judge_hitl_enabled: bool
    hitl_auto_pause_after: list[str]
    agent_call_delay_ms: int


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
    Introduces a small inter-call delay to avoid overwhelming the API.
    """
    registry = state.get("agent_registry")
    if not registry:
        return None
    # Inter-call delay: prevent hammering the API during sequential agent calls.
    # 0 = random 3000-6000ms; >0 = fixed delay in ms; <0 = no delay.
    delay_ms = state.get("agent_call_delay_ms", 0)
    if delay_ms == 0:
        delay_ms = random.randint(3000, 6000)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
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
    judge_agent: Any = None,
    judge_llm_enabled: bool = False,
    judge_method: str = "phase",
) -> tuple[GameState, GameEvent]:
    """Create a judge broadcast event and append to game state.

    When judge_agent is provided and judge_llm_enabled is True, dispatches to
    the appropriate JudgeAgent method based on ``judge_method``. The ``message``
    param serves as fallback on failure.
    """
    final_message = message
    if judge_agent is not None and judge_llm_enabled:
        try:
            llm_msg = _generate_judge_message(
                judge_agent, phase=phase, fallback=message,
                day_number=day_number, night_number=night_number,
                extra_payload=extra_payload, judge_method=judge_method,
            )
            if llm_msg:
                final_message = llm_msg
        except Exception:
            pass  # fallback to hardcoded message

    payload: dict[str, Any] = {
        "phase": phase,
        "message": final_message,
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


def _generate_judge_message(
    judge_agent: Any,
    *,
    phase: str,
    fallback: str,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    judge_method: str = "phase",
) -> str:
    """Dispatch to the appropriate JudgeAgent method, return empty string on failure."""
    ep = extra_payload or {}

    if judge_method == "vote_calling":
        result = judge_agent.broadcast_vote_calling(
            voter_id=ep.get("voter_id", ""),
            voter_name=ep.get("voter_name", ""),
            candidates=ep.get("candidates", []),
            position=ep.get("position", 1),
            total=ep.get("total", 1),
            day_number=day_number,
            sheriff_weight=ep.get("sheriff_weight", 1.0),
        )
        return result.message if result and result.message else ""

    if judge_method == "skill_guide":
        result = judge_agent.guide_skill_use(
            role=ep.get("role", ""),
            player_id=ep.get("player_id", ""),
            player_name=ep.get("player_name", ""),
            available_actions=ep.get("available_actions", []),
            context_hints=ep.get("context_hints"),
        )
        return result.message if result and result.message else ""

    if judge_method == "vote_tally":
        result = judge_agent.announce_vote_tally(
            tally=ep.get("tally", {}),
            player_names=ep.get("player_names", {}),
            sheriff_id=ep.get("sheriff_id"),
            sheriff_weight=ep.get("sheriff_weight", 1.5),
            day_number=day_number,
        )
        return result.message if result and result.message else ""

    if judge_method == "exile":
        result = judge_agent.announce_exile_result(
            exiled_player_id=ep.get("exiled_player_id"),
            exiled_player_name=ep.get("exiled_player_name", ""),
            reason=ep.get("reason", ""),
            tied_player_ids=ep.get("tied_player_ids"),
            day_number=day_number,
        )
        return result.message if result and result.message else ""

    if judge_method == "death":
        deaths = ep.get("deaths", [])
        result = judge_agent.broadcast_death_announcement(
            deaths=deaths, day_number=day_number,
        )
        return result.message if result and result.message else ""

    if judge_method == "sheriff":
        result = judge_agent.broadcast_sheriff_result(
            sheriff_id=ep.get("sheriff_id"),
            badge_state=ep.get("badge_state", "none"),
        )
        return result.message if result and result.message else ""

    # Default: broadcast_phase
    public_data: dict[str, Any] = dict(ep)
    if night_number > 0:
        public_data["night_number"] = night_number
    if day_number > 0:
        public_data["day_number"] = day_number
    result = judge_agent.broadcast_phase(
        phase=phase,
        day_number=day_number,
        night_number=night_number,
        public_data=public_data or None,
    )
    return result.message if result and result.message else ""


def _jb(
    state: RuntimeState,
    *,
    phase: str,
    message: str,
    gs: GameState | None = None,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    visibility: str = "public",
    judge_method: str = "phase",
) -> tuple[GameState, GameEvent]:
    """Shortcut: _judge_broadcast that extracts judge_agent from RuntimeState.

    Updates ``state["game_state"]`` in-place so subsequent ``_jb`` calls in
    the same node function accumulate events without needing explicit ``gs=gs``.
    This is safe for LangGraph checkpoint/replay because the node function's
    returned ``{"game_state": gs}`` dict is what gets checkpointed, not the
    mutated input state dict.
    """
    if gs is None:
        gs = state["game_state"]
    gs, event = _judge_broadcast(
        phase=phase,
        message=message,
        gs=gs,
        day_number=day_number,
        night_number=night_number,
        extra_payload=extra_payload,
        visibility=visibility,
        judge_agent=state.get("judge_agent"),
        judge_llm_enabled=state.get("judge_llm_enabled", False),
        judge_method=judge_method,
    )
    state["game_state"] = gs
    return gs, event


def _ensure_day_incremented(
    state: RuntimeState,
    gs: GameState | None = None,
) -> tuple[GameState, int]:
    """Increment day_number and broadcast '天亮了' if not already done.

    Returns (updated_gs, current_day_number).
    """
    if gs is None:
        gs = state["game_state"]
    if gs.phase != "day":
        d = gs.day_number + 1
        label = phase_label("day", d)
        gs, _ = _jb(
            state,
            phase="day_announce",
            message=f"{label}：天亮了",
            gs=gs,
            day_number=d,
            visibility="public",
        )
        gs = replace(gs, phase="day", day_number=d,
                     events=gs.events + [GameEvent(type="day_announce", payload={"day": d})])
        return gs, d
    return gs, gs.day_number


def _hitl_checkpoint(state: RuntimeState, phase: str, direction: str = "after") -> dict[str, Any]:
    """HITL checkpoint: pause execution if JudgeHITLInterface says so.

    Call at key phase transitions (enter_night, announce_deaths, day_vote,
    resolve_exile). When HITL is enabled and a pause is triggered, processes
    human commands and flushes audit events into game state.

    Returns a dict to merge into the node's return value.
    """
    if not state.get("judge_hitl_enabled"):
        return {}
    hitl = state.get("judge_hitl")
    if hitl is None:
        return {}
    if not hitl.should_pause(phase, direction):
        return {}
    gs = state["game_state"]
    # Wait for human input (non-blocking in simulation mode)
    cmd = hitl.wait_for_human(timeout=300)
    if cmd is not None:
        result = hitl.handle_command(cmd, gs)
        if "game_state" in result:
            state["game_state"] = result["game_state"]
    # Flush HITL audit events into game state
    hitl_events = hitl.flush_events()
    if hitl_events:
        gs = state["game_state"]
        state["game_state"] = replace(gs, events=gs.events + hitl_events)
    return {}


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
