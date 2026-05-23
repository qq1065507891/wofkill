"""LangGraph game graph: deterministic orchestration around RuleEngine.

Every node calls RuleEngine for rule decisions. No natural language adjudication.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import replace
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

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
    agent_day_speech,
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
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
    eligible_sheriff_voters,
    filter_sheriff_votes_to_eligible,
    is_all_players_on_sheriff,
)
from werewolf_agent.runtime.timers import timed_call
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import detect_timeline_confusion, phase_label

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


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
    # Day 1 sheriff-before-deaths flow control
    day_number_already_incremented: bool
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
    reason = parsed.get("reason") or action_trace.get("reason") or ""
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
            print(f"  [狼人决策] 按狼队计划击杀: {_player_display(state, target)}")
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


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def setup_game(state: RuntimeState) -> dict[str, Any]:
    gs = state.get("game_state")
    engine = state.get("engine") or _new_engine()
    if gs is None:
        gs = GameState(ruleset_id="pre_witch_hunter_idiot_mixed", game_id=uuid.uuid4().hex[:8])
    gs = replace(gs, phase="setup")
    if not any(e.type == "judge_broadcast" and e.payload.get("phase") == "game_start" for e in gs.events):
        gs, _ = _judge_broadcast(
            phase="game_start",
            message="游戏开始，请所有玩家确认身份，准备进入首夜",
            gs=gs,
            visibility="public",
        )
    return {"game_state": gs, "engine": engine}


def assign_roles(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    player_ids = [f"p{i:02d}" for i in range(1, 13)]
    players = engine.assign_roles(player_ids, seed=_stable_seed(gs.game_id, "roles"))
    gs = replace(gs, players=players, phase="roles_assigned",
                 events=gs.events + [GameEvent(type="roles_assigned", payload={})])
    print(f"\n{'='*60}")
    print(f"  角色分配完成 (Game: {gs.game_id})")
    for pid, p in sorted(players.items()):
        print(f"    {_player_display(state, pid)}: {p.role}")
    print(f"{'='*60}")
    return {"game_state": gs}


# -- Night nodes --

def enter_night(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    n = gs.night_number + 1
    label = phase_label("night", n)
    gs, _ = _judge_broadcast(phase="enter_night", message=f"{label}：天黑请闭眼", gs=gs, night_number=n)
    gs = replace(gs, phase="night", night_number=n,
                 events=gs.events + [GameEvent(type="enter_night", payload={"night": n})])
    alive = [pid for pid, p in gs.players.items() if p.alive]
    print(f"\n{'='*60}")
    print(f"  【{label}】天黑请闭眼 (存活: {len(alive)}人)")
    print(f"{'='*60}")
    return {"game_state": gs}


def _legacy_single_round_wolf_discussion(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    wolves = _alive_wolves(gs)
    events = []
    print(f"  [狼人密谈] 狼人: {[_player_display(state, w) for w in wolves]}")
    has_agents = False
    for wolf_id in wolves:
        result = _dispatch_agent(
            state,
            agent_wolf_discussion,
            wolf_id,
            timeout_override=AGENT_TIMEOUTS.wolf_discussion_per_player,
        )
        if result is not None:
            has_agents = True
            speech_text = result.get("speech_text", "")
            print(f"    {_player_display(state, wolf_id)}(狼人): {speech_text if speech_text else '(沉默)'}")
            payload = {
                "wolf_id": wolf_id,
                "night_number": gs.night_number,
                "text": speech_text,
                "visibility": "werewolf_team_only",
            }
            events.append(GameEvent(
                type="wolf_discussion",
                payload=payload,
            ))
            if result.get("action_trace"):
                events.append(_action_trace_event(
                    player_id=wolf_id,
                    phase="wolf_discussion",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
    if has_agents:
        gs = replace(gs, events=gs.events + events)
        return {"game_state": gs}

    # Scripted fallback
    gs = replace(gs, events=gs.events + [GameEvent(type="wolf_discussion", payload={})])
    return {"game_state": gs}


def _legacy_wolf_consensus(state: RuntimeState) -> dict[str, Any]:
    """Determine wolf night action.

    Wolves may strategically skip a kill (空刀脏人), but consecutive
    no-kill nights are capped to prevent degenerate infinite loops.
    """
    gs: GameState = state["game_state"]
    max_consecutive_no_kill = 2  # After N straight no-kill nights, force a kill

    # Count consecutive no-kill nights so far
    consecutive_no_kill = 0
    for ev in reversed(gs.events):
        if ev.type == "wolf_no_kill_timeout" or ev.type == "wolf_no_kill_declared":
            consecutive_no_kill += 1
        elif ev.type in ("wolf_kill_selected",):
            break
        elif ev.type == "enter_night":
            continue
        else:
            continue

    if _timer_expired(state, "wolf_discussion"):
        if consecutive_no_kill >= max_consecutive_no_kill:
            print(f"  [狼人决策] 连续{consecutive_no_kill}夜空刀，强制击杀")
            return _force_wolf_kill(gs, "timer_expired_forced_kill")
        print(f"  [狼人决策] 讨论超时，空刀")
        event = GameEvent(
            type="wolf_no_kill_timeout",
            payload={"night_number": gs.night_number, "reason": "timer_expired"},
        )
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": None}

    # Try agent-driven decision first
    if state.get("agent_registry") and not state.get("wolf_action"):
        result = _dispatch_agent(
            state,
            agent_wolf_consensus,
            timeout_override=AGENT_TIMEOUTS.wolf_consensus,
        )
        if result is not None:
            action = result.get("wolf_action", "kill")
            target = result.get("wolf_kill_target_id")
            if action == "no_kill":
                if consecutive_no_kill >= max_consecutive_no_kill:
                    print(f"  [狼人决策] 连续{consecutive_no_kill}夜空刀，强制击杀")
                    return _force_wolf_kill(gs, "consecutive_no_kill_limit")
                print(f"  [狼人决策] 狼人选择空刀 (原因: {result.get('wolf_action_reason', 'agent decision')})")
                event = GameEvent(
                    type="wolf_no_kill_declared",
                    payload={
                        "night_number": gs.night_number,
                        "reason": result.get("wolf_action_reason", "agent decision"),
                        "action_traces": result.get("action_traces", {}),
                    },
                )
                gs = replace(gs, events=gs.events + [event])
                return {"game_state": gs, "wolf_kill_target_id": None}
            if action == "kill" and target:
                target_state = gs.players.get(target)
                if target_state and target_state.alive:
                    print(f"  [狼人决策] 击杀目标: {_player_display(state, target)} (原因: {result.get('wolf_action_reason', '')})")
                    event = GameEvent(
                        type="wolf_kill_selected",
                        payload={
                            "night_number": gs.night_number,
                            "target_id": target,
                            "action_traces": result.get("action_traces", {}),
                        },
                    )
                    gs = replace(gs, events=gs.events + [event])
                    return {"game_state": gs, "wolf_kill_target_id": target}
        else:
            # Agent call timed out entirely
            if consecutive_no_kill >= max_consecutive_no_kill:
                print(f"  [狼人决策] Agent超时，连续{consecutive_no_kill}夜空刀，强制击杀")
                return _force_wolf_kill(gs, "agent_timeout_forced_kill")
            print(f"  [狼人决策] Agent调用超时，空刀")
            event = GameEvent(
                type="wolf_no_kill_timeout",
                payload={"night_number": gs.night_number},
            )
            gs = replace(gs, events=gs.events + [event])
            return {"game_state": gs, "wolf_kill_target_id": None}

    # Scripted fallback
    action = state.get("wolf_action")
    target = state.get("wolf_kill_target_id")

    if action == "no_kill":
        event = GameEvent(
            type="wolf_no_kill_declared",
            payload={
                "night_number": gs.night_number,
                "reason": state.get("wolf_action_reason", ""),
            },
        )
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": None}

    if (action == "kill" or action is None) and target is not None:
        target_state = gs.players.get(target)
        if target_state is None or not target_state.alive:
            event = GameEvent(
                type="wolf_no_kill_timeout",
                payload={"night_number": gs.night_number},
            )
            gs = replace(gs, events=gs.events + [event])
            return {"game_state": gs, "wolf_kill_target_id": None}
        event = GameEvent(
            type="wolf_kill_selected",
            payload={"night_number": gs.night_number, "target_id": target},
        )
        gs = replace(gs, events=gs.events + [event])
        return {"game_state": gs, "wolf_kill_target_id": target}

    event = GameEvent(
        type="wolf_no_kill_timeout",
        payload={"night_number": gs.night_number},
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs, "wolf_kill_target_id": None}


def night_witch(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="witch_wake",
        message="女巫请睁眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    gs, _ = _judge_broadcast(
        phase="witch_choose",
        message="女巫请选择是否使用解药或毒药",
        gs=gs, night_number=gs.night_number,
        visibility="witch_private",
    )
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    result = _dispatch_agent(
        state,
        agent_night_witch,
        timeout_override=AGENT_TIMEOUTS.witch,
    )
    if result is not None:
        use_antidote = result.get("use_antidote", False)
        poison_target_id = result.get("poison_target_id")
        action_taken = "no_action"
        if use_antidote:
            action_taken = "use_antidote"
        elif poison_target_id:
            action_taken = "use_poison"
        wolf_target = state.get("wolf_kill_target_id")
        if use_antidote:
            print(f"  [女巫] 使用解药救了 {_player_display(state, wolf_target)}")
        if poison_target_id:
            print(f"  [女巫] 使用毒药毒了 {_player_display(state, poison_target_id)}")
        if not use_antidote and not poison_target_id:
            print(f"  [女巫] 不使用药水 (解药{'已用' if gs.antidote_used else '可用'}, 毒药{'已用' if gs.poison_used else '可用'})")
        audit = GameEvent(
            type="witch_decision_audit",
            payload={
                "night_number": gs.night_number,
                "wolf_kill_target_id": state.get("wolf_kill_target_id"),
                "action_taken": action_taken,
                "poison_target_id": poison_target_id,
                "reason": "agent_decision",
                "visibility": "witch_private",
                "action_trace": result.get("witch_action_trace"),
            },
        )
        gs = replace(gs, events=gs.events + [audit])
        gs, _ = _judge_broadcast(
            phase="witch_sleep",
            message="女巫请闭眼",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        return {"game_state": gs, **result}

    # Scripted fallback
    gs, _ = _judge_broadcast(
        phase="witch_sleep",
        message="女巫请闭眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    return {"use_antidote": state.get("use_antidote", False),
            "poison_target_id": state.get("poison_target_id"),
            "game_state": gs}


def night_seer(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="seer_wake",
        message="预言家请睁眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    gs, _ = _judge_broadcast(
        phase="seer_choose",
        message="预言家请选择你要查验的玩家",
        gs=gs, night_number=gs.night_number,
        visibility="seer_private",
    )
    state = {**state, "game_state": gs}

    # Try agent-driven decision first
    result = _dispatch_agent(
        state,
        agent_night_seer,
        timeout_override=AGENT_TIMEOUTS.seer,
    )
    if result is not None:
        target = result.get("seer_target_id")
        if target:
            print(f"  [预言家] 查验目标: {_player_display(state, target)}")
        return {"game_state": gs, **result}

    # Scripted fallback
    return {"seer_target_id": state.get("seer_target_id"), "game_state": gs}


def night_hunter_idiot_status(state: RuntimeState) -> dict[str, Any]:
    """First night only: confirm hunter and idiot are alive for moderator audit.
    Produces no public output; event is moderator/private visibility only."""
    gs: GameState = state["game_state"]
    hunter_id = _find_role(gs, "hunter")
    idiot_id = _find_role(gs, "idiot")
    if hunter_id:
        gs, _ = _judge_broadcast(
            phase="hunter_status",
            message=f"猎人{_player_display(state, hunter_id)}请确认开枪状态",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        print(f"  [法官] 猎人{_player_display(state, hunter_id)}请确认开枪状态")
    if idiot_id and gs.night_number == 1:
        gs, _ = _judge_broadcast(
            phase="idiot_status",
            message=f"白痴{_player_display(state, idiot_id)}请确认身份",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        print(f"  [法官] 白痴{_player_display(state, idiot_id)}请确认身份")
    if gs.night_number != 1:
        return {}
    event = GameEvent(
        type="hunter_idiot_status_confirmed",
        payload={
            "night_number": 1,
            "hunter_id": hunter_id,
            "idiot_id": idiot_id,
            "visibility": "moderator_only",
        },
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs}


def first_night_hybrid_master(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    if gs.night_number != 1 or gs.hybrid_master_id is not None:
        return {}
    hybrid_id = _find_role(gs, "hybrid")
    if hybrid_id is None:
        return {}

    gs, _ = _judge_broadcast(
        phase="hybrid_wake",
        message=f"混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    gs, _ = _judge_broadcast(
        phase="hybrid_choose",
        message="混血儿请选择你的主人",
        gs=gs, night_number=gs.night_number,
        visibility="hybrid_private",
    )
    print(f"  [法官] 混血儿{_player_display(state, hybrid_id)}请睁眼，选择你的主人")

    master_target = state.get("hybrid_master_target_id")

    # Agent-driven: ask hybrid player to choose master
    if master_target is None:
        result = _dispatch_agent(
            state,
            agent_hybrid_choose_master,
            hybrid_id,
            timeout_override=AGENT_TIMEOUTS.seer,
        )
        if result and result.get("master_target_id"):
            master_target = result["master_target_id"]

    # Fallback: random selection
    if master_target is None:
        import random
        candidates = [pid for pid in _player_ids(gs) if pid != hybrid_id]
        rng = random.Random(_stable_seed(gs.game_id, "hybrid_master"))
        master_target = rng.choice(candidates) if candidates else None

    if master_target is None:
        return {}
    gs, event = engine.choose_master(gs, hybrid_id=hybrid_id, master_id=master_target)
    gs = replace(gs, events=gs.events + [event])
    gs, _ = _judge_broadcast(
        phase="hybrid_sleep",
        message="混血儿请闭眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    master_role = gs.players[master_target].role if master_target in gs.players else "?"
    print(f"  [混血儿] {_player_display(state, hybrid_id)} 选择了 {_player_display(state, master_target)}({master_role}) 作为主人")
    return {"game_state": gs}


def resolve_night(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    gs, events = engine.resolve_night(
        gs,
        night_number=gs.night_number,
        wolf_kill_target_id=state.get("wolf_kill_target_id"),
        use_antidote=state.get("use_antidote", False),
        poison_target_id=state.get("poison_target_id"),
        seer_target_id=state.get("seer_target_id"),
    )
    seer_trace = state.get("seer_action_trace")
    if seer_trace:
        events = [
            replace(event, payload={**event.payload, "action_trace": seer_trace})
            if event.type == "seer_check" else event
            for event in events
        ]
    if events:
        gs = replace(gs, events=gs.events + events)
    # Log night resolution events
    seer_woke = any(
        event.type == "judge_broadcast" and event.payload.get("phase") == "seer_wake"
        for event in gs.events
    )
    for ev in events:
        if ev.type == "wolf_kill":
            target = ev.payload.get("player_id", "?")
            saved = ev.payload.get("saved_by_antidote", False)
            if saved:
                print(f"  [夜晚结算] {_player_display(state, target)} 被狼人袭击，但被女巫救活")
            else:
                print(f"  [夜晚结算] {_player_display(state, target)} 被狼人袭击身亡")
        elif ev.type == "poison_death":
            print(f"  [夜晚结算] {_player_display(state, ev.payload.get('player_id', '?'))} 被女巫毒杀")
        elif ev.type == "seer_check":
            target = ev.payload.get("target_id", "?")
            alignment = ev.payload.get("alignment", "?")
            gs, _ = _judge_broadcast(
                phase="seer_result",
                message=f"他的身份是{'好人' if alignment == 'good' else '狼人'}",
                gs=gs,
                night_number=gs.night_number,
                visibility="seer_private",
            )
            print(f"  [夜晚结算] 预言家查验 {_player_display(state, target)}: {'好人' if alignment == 'good' else '狼人'}")
        elif ev.type == "no_death":
            print(f"  [夜晚结算] 平安夜，无人死亡")
    if seer_woke:
        gs, _ = _judge_broadcast(
            phase="seer_sleep",
            message="预言家请闭眼",
            gs=gs,
            night_number=gs.night_number,
            visibility="moderator_only",
        )
    return {"game_state": gs}


# -- Day nodes --

def sheriff_first_day_entry(state: RuntimeState) -> dict[str, Any]:
    """Day 1 (or resumed sheriff) entry: increment day, announce dawn WITHOUT deaths.
    Sheriff election follows, then deaths are announced after election completes.
    """
    gs: GameState = state["game_state"]
    d = gs.day_number + 1
    label = phase_label("day", d)
    gs, _ = _judge_broadcast(
        phase="day_announce", message=f"{label}：天亮了",
        gs=gs, day_number=d,
    )
    gs = replace(gs, phase="day", day_number=d,
                 events=gs.events + [GameEvent(type="day_announce", payload={"day": d})])
    alive = [pid for pid, p in gs.players.items() if p.alive]
    print(f"\n{'='*60}")
    print(f"  【{label}】天亮了 (存活: {len(alive)}人: {alive})")
    print(f"{'='*60}")
    return {
        "game_state": gs,
        "day_number_already_incremented": True,
        "revote": False,
        "speech_index": 0,
        "current_speaker_id": None,
        "speech_order": [],
    }


def announce_deaths(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    # Only increment day if not already done by sheriff_first_day_entry
    if not state.get("day_number_already_incremented"):
        d = gs.day_number + 1
        label = phase_label("day", d)
        gs, _ = _judge_broadcast(phase="day_announce", message=f"{label}：天亮了", gs=gs, day_number=d)
        gs = replace(gs, phase="day", day_number=d,
                     events=gs.events + [GameEvent(type="day_announce", payload={"day": d})])
    else:
        d = gs.day_number
        label = phase_label("day", d)

    # Announce last night's deaths
    night_deaths = [
        death for death in gs.deaths
        if death.timing == "night" and death.resolution_batch == f"night_{gs.night_number}"
    ]
    if night_deaths:
        dead_names = "、".join(_player_display(state, death.player_id) for death in night_deaths)
        gs, _ = _judge_broadcast(
            phase="death_announce",
            message=f"昨夜死亡: {dead_names}",
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
    if not state.get("day_number_already_incremented"):
        print(f"\n{'='*60}")
        print(f"  【{label}】天亮了 (存活: {len(alive)}人: {alive})")
    if night_deaths:
        for death in night_deaths:
            print(f"  [死讯] {_player_display(state, death.player_id)} 死亡 (原因: {death.reason})")
    else:
        print(f"  [死讯] 平安夜，无人死亡")
    print(f"{'='*60}")
    return {"game_state": gs, "day_number_already_incremented": False,
            "revote": False, "speech_index": 0,
            "current_speaker_id": None, "speech_order": []}


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
    print(f"  [警徽] 本局警徽永久流失")
    result["game_state"] = gs
    return result


def night_death_last_words(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    eligible = []
    for death in gs.deaths:
        if death.timing == "night" and engine.can_leave_last_words(
            death_reason=death.reason, timing=death.timing, night_number=gs.night_number
        ):
            eligible.append(death.player_id)
    gs = replace(gs, events=gs.events + [GameEvent(
        type="night_death_last_words", payload={"players": eligible}
    )])
    return {"game_state": gs}


# -- Sheriff election (first day only) --

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
                print(f"  [上警报名] {_player_display(state, pid)} 报名上警")
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


def _legacy_sheriff_speech(state: RuntimeState) -> dict[str, Any]:
    """Sheriff candidates speak in random order assigned by the judge."""
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]
    registry = state.get("agent_registry")
    candidates = list(gs.sheriff_candidates or state.get("sheriff_candidates", []))

    if not candidates:
        gs = replace(gs, events=gs.events + [GameEvent(type="sheriff_speech", payload={})])
        return {"game_state": gs}

    # Judge randomly assigns speaking order
    import random as _random
    seed = _stable_seed(gs.game_id, "sheriff_speech_order", gs.day_number)
    rng = _random.Random(seed)
    speech_order = list(candidates)
    rng.shuffle(speech_order)

    names = ", ".join(_player_display(state, p) for p in speech_order)
    gs, _ = _judge_broadcast(
        phase="sheriff_speech_start",
        message=f"警上发言顺序: {names}",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    events: list[GameEvent] = []
    has_agents = False
    for candidate_id in speech_order:
        result = _dispatch_agent(
            state,
            agent_day_speech,
            candidate_id,
            timeout_override=AGENT_TIMEOUTS.day_speech,
        )
        if result is not None:
            has_agents = True
            if result.get("self_destruct"):
                return {"game_state": gs, "self_destruct_wolf_id": candidate_id}
            speech_text = result.get("speech_text", "")
            print(f"  [警上发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
            events.append(GameEvent(
                type="sheriff_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": speech_text,
                },
            ))
            if result.get("action_trace"):
                events.append(_action_trace_event(
                    player_id=candidate_id,
                    phase="sheriff_speech",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
    if not has_agents:
        events.append(GameEvent(type="sheriff_speech", payload={}))

    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


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
                print(f"  [退水] {_player_display(state, candidate_id)} 退出竞选")
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
        print(f"  [警长选举] {_player_display(state, winner)} 当选警长")
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
                print(f"  [警长投票] {_player_display(state, voter_id)} 投票给 {_player_display(state, result['vote_target'])}")
            else:
                print(f"  [警长投票] {_player_display(state, voter_id)} 弃票")
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

    # Announce result
    elected_id = event.payload.get("sheriff_id")
    if elected_id:
        gs, _ = _judge_broadcast(
            phase="sheriff_elected",
            message=f"{_player_display(state, elected_id)} 当选警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [警长选举] {_player_display(state, elected_id)} 当选警长")
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


# -- Day discussion & vote --

def free_discussion(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    speaker_id = state.get("current_speaker_id")

    # Announce discussion start on first entry (speech_index == 0)
    if speech_index == 0 and not speaker_id:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message=f"请警长{_player_display(state, gs.sheriff_id)}安排发言顺序，开始自由讨论",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )
        else:
            gs, _ = _judge_broadcast(
                phase="discussion_start",
                message="自由讨论开始，随机选择发言起点",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )

    # Auto-populate speech order based on sheriff status
    if not speech_order:
        if gs.sheriff_id and gs.sheriff_badge_state == "active":
            # Sheriff agent picks first speaker; fallback to static order
            agent_order = _dispatch_agent(
                state,
                agent_sheriff_pick_speech_order,
                gs.sheriff_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_order = agent_order or choose_sheriff_led_speech_order(gs, gs.sheriff_id)
        else:
            speech_order = choose_no_sheriff_speech_order(gs)

    if speaker_id is None and speech_index < len(speech_order):
        speaker_id = speech_order[speech_index]

    def advance_speaker() -> dict[str, Any]:
        next_index = speech_index + 1
        next_speaker = speech_order[next_index] if next_index < len(speech_order) else None
        return {
            "speech_index": next_index,
            "current_speaker_id": next_speaker,
            "speech_timed_out": False,
            "speech_text": "",
            "speech_order": speech_order,
        }

    timed_out = state.get("speech_timed_out", False) or (
        speaker_id is not None and _timer_expired(state, f"speech:{speaker_id}")
    )

    if speaker_id and timed_out:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="speech_timeout",
            payload={
                "player_id": speaker_id,
                "day_number": gs.day_number,
                "seconds_limit": state.get("speech_seconds_limit", 0),
            },
        )])
        return {"game_state": gs, **advance_speaker()}
    if speaker_id:
        # Judge announces current speaker
        gs, _ = _judge_broadcast(
            phase="speaker_turn",
            message=f"请{_player_display(state, speaker_id)}发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        speech_text = state.get("speech_text", "")
        action_trace = None
        if not speech_text:
            result = _dispatch_agent(
                state,
                agent_day_speech,
                speaker_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            if result is not None:
                if result.get("self_destruct"):
                    return {"game_state": gs, "self_destruct_wolf_id": speaker_id}
                speech_text = result.get("speech_text", "")
                action_trace = result.get("action_trace")
        player_role = gs.players[speaker_id].role if speaker_id in gs.players else "?"
        print(f"  [{_player_display(state, speaker_id)}({player_role})]: {speech_text if speech_text else '(未发言)'}")
        payload = {
            "speaker": speaker_id,
            "day_number": gs.day_number,
            "text": speech_text,
        }
        events = [GameEvent(type="speech", payload=payload)]
        if action_trace:
            events.append(_action_trace_event(
                player_id=speaker_id,
                phase="speech",
                action_trace=action_trace,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        gs = replace(gs, events=gs.events + events)
        return {"game_state": gs, **advance_speaker()}
    gs = replace(gs, events=gs.events + [GameEvent(type="free_discussion", payload={})])
    return {"game_state": gs}


def day_vote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="vote_start",
        message="讨论结束，现在开始投票。所有人同时投票，投票时不能发言。",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    same_vote_window = (
        state.get("exile_vote_day") == gs.day_number
        and state.get("exile_vote_revote") == state.get("revote", False)
    )
    existing_votes = state.get("exile_votes", {}) if same_vote_window else {}
    registry = state.get("agent_registry")
    is_revote = state.get("revote", False)
    if is_revote:
        print(f"\n  --- PK重新投票 ---")
    else:
        print(f"\n  --- 投票开始 ---")
    votes: dict[str, str] = {}
    vote_traces: dict[str, Any] = {}
    has_agents = False
    if not existing_votes:
        for pid, player in gs.players.items():
            if player.alive:
                result = _dispatch_agent(
                    state,
                    agent_day_vote,
                    pid,
                    timeout_override=AGENT_TIMEOUTS.day_vote,
                )
                if result is not None:
                    has_agents = True
                    if result.get("vote_target"):
                        votes[pid] = result["vote_target"]
                        print(f"    {_player_display(state, pid)} → {_player_display(state, result['vote_target'])}")
                    else:
                        print(f"    {_player_display(state, pid)} 弃票")
                    if result.get("action_trace"):
                        vote_traces[pid] = result["action_trace"]

        if has_agents:
            # Judge announces each vote publicly
            sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
            vote_lines = []
            for voter_id, target_id in votes.items():
                weight_label = " (警长1.5票)" if voter_id == sheriff_id else ""
                vote_lines.append(f"{_player_display(state, voter_id)}{weight_label} 投票给 {_player_display(state, target_id)}")
            if vote_lines:
                gs, _ = _judge_broadcast(
                    phase="vote_result",
                    message="投票结果：\n" + "\n".join(vote_lines),
                    gs=gs, day_number=gs.day_number,
                    visibility="public",
                )

            return {
                "game_state": gs,
                "exile_votes": votes,
                "vote_action_traces": vote_traces,
                "exile_vote_day": gs.day_number,
                "exile_vote_revote": state.get("revote", False),
                "revote": state.get("revote", False),
            }
    return {
        "game_state": gs,
        "exile_votes": existing_votes,
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": state.get("revote", False),
        "revote": state.get("revote", False),
    }


def resolve_vote(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    consecutive = state.get("consecutive_no_exile_days", 0)
    votes = state.get("exile_votes", {})
    result = engine.resolve_vote(
        gs, votes=votes,
        revote=state.get("revote", False),
        consecutive_no_exile_days=consecutive,
        pk_candidates=state.get("pk_candidates"),
        rng_seed=f"{gs.game_id}-vote-d{gs.day_number}",
    )
    # Log vote tally with weighted counts (sheriff = 1.5, others = 1)
    sheriff_id = gs.sheriff_id if gs.sheriff_badge_state == "active" else None
    weighted_tally: dict[str, float] = {}
    vote_weights: dict[str, float] = {}
    if votes:
        for voter_id, target_id in votes.items():
            weight = 1.5 if voter_id == sheriff_id else 1.0
            vote_weights[voter_id] = weight
            weighted_tally[target_id] = weighted_tally.get(target_id, 0) + weight
        tally_items = sorted(weighted_tally.items(), key=lambda x: -x[1])
        tally_text = "  投票统计: " + ", ".join(
            f"{_player_display(state, t)}:{v}票" for t, v in tally_items
        )
    else:
        tally_text = "  投票统计: 无有效票"
    print(tally_text)
    if result.exiled_player_id:
        gs, _ = _judge_broadcast(
            phase="vote_result_announce",
            message=f"投票结果：{_player_display(state, result.exiled_player_id)} 以最高票被放逐出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [投票结果] {_player_display(state, result.exiled_player_id)} 被放逐 (原因: {result.reason})")
    elif result.reason == "first_tie_pk":
        tied_names = "、".join(_player_display(state, t) for t in (result.tied_player_ids or []))
        gs, _ = _judge_broadcast(
            phase="vote_tie_pk",
            message=f"首次平票：{tied_names}进入PK发言",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [投票结果] 首次平票，进入PK: {[_player_display(state, t) for t in (result.tied_player_ids or [])]}")
    elif result.reason == "second_tie_no_exile":
        gs, _ = _judge_broadcast(
            phase="vote_second_tie",
            message="二次平票，无人出局，直接进入黑夜",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [投票结果] 二次平票，无人出局")
    elif result.reason == "anti_stall_empty_tally":
        gs, _ = _judge_broadcast(
            phase="vote_anti_stall",
            message=f"防死循环强制放逐：{_player_display(state, result.exiled_player_id)}出局",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [投票结果] 防死循环强制放逐: {_player_display(state, result.exiled_player_id)}")
    else:
        print(f"  [投票结果] {result.reason}")
    payload: dict[str, Any] = {
        "exiled": result.exiled_player_id,
        "reason": result.reason,
        "day_number": gs.day_number,
        "sheriff_id": sheriff_id,
        "sheriff_vote_weight": 1.5 if sheriff_id else 1.0,
        "weighted_tally": weighted_tally,
        "vote_weights": vote_weights,
        "votes": [
            {
                "voter": voter_id,
                "target": target_id,
                "reason": _public_vote_reason(
                    (state.get("vote_action_traces") or {}).get(voter_id)
                ),
            }
            for voter_id, target_id in sorted((state.get("exile_votes") or {}).items())
        ],
    }
    if result.tied_player_ids:
        payload["tied"] = result.tied_player_ids
    vote_trace_events = []
    for pid, trace in (state.get("vote_action_traces") or {}).items():
        audit_event = _action_trace_event(
            player_id=pid,
            phase="vote",
            action_trace=_with_vote_target_in_trace(trace, state.get("exile_votes", {}).get(pid, "")),
            day_number=gs.day_number,
            night_number=gs.night_number,
        )
        thought = audit_event.payload.get("private_vote_thought") or {}
        if thought:
            print(
                f"  [投票心理][仅主持人] {_player_display(state, pid)} -> "
                f"{_player_display(state, thought.get('target'))}: "
                f"站边={thought.get('standing_with_seer') or '未明确'}；"
                f"怀疑理由={thought.get('suspect_reason') or thought.get('public_reason') or '未说明'}；"
                f"排除理由={thought.get('not_voting_reason') or '未说明'}；"
                f"内心理由={thought.get('private_reason') or '未说明'}"
            )
        vote_trace_events.append(audit_event)
    gs = replace(gs, votes=state.get("exile_votes", {}),
                 events=gs.events + [GameEvent(
                     type="vote_resolved",
                     payload=payload,
                 )] + vote_trace_events)
    next_consecutive = (
        consecutive + 1 if result.reason == "second_tie_no_exile" else 0
    )
    next_state: dict[str, Any] = {
        "game_state": gs,
        "_vote_result": result,
        "consecutive_no_exile_days": next_consecutive,
    }
    if result.reason == "first_tie_pk":
        next_state["pk_candidates"] = result.tied_player_ids
    elif result.exiled_player_id is not None or result.reason == "second_tie_no_exile":
        next_state["pk_candidates"] = []
        next_state["exile_votes"] = {}
        next_state["vote_action_traces"] = {}
        next_state["exile_vote_revote"] = False
    return next_state


def resolve_exile(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    # Read exiled player from the last vote_resolved event (not _vote_result
    # which is not in RuntimeState and gets dropped by LangGraph channels).
    exiled_id = None
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            exiled_id = event.payload.get("exiled")
            break
    if exiled_id is None:
        return {"game_state": gs}
    exiled_role = gs.players.get(exiled_id, None)
    role_str = exiled_role.role if exiled_role else "?"
    gs, _ = _judge_broadcast(
        phase="exile",
        message=f"{_player_display(state, exiled_id)}被放逐出局",
        gs=gs, day_number=gs.day_number,
        extra_payload={"exiled": exiled_id},
        visibility="public",
    )
    gs, events = engine.resolve_exile(gs, target_id=exiled_id)
    gs = replace(gs, events=gs.events + events)
    print(f"  [放逐] {_player_display(state, exiled_id)}({role_str}) 被放逐出局")
    for ev in events:
        if ev.type == "idiot_reveal":
            print(f"  [白痴亮牌] {_player_display(state, exiled_id)} 是白痴，不会被放逐")
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
    # Idiot reveal: player stays alive, no last words needed
    player = gs.players.get(exiled_id)
    if player is None or player.alive:
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="exile_last_words",
        message=f"请{_player_display(state, exiled_id)}发表遗言",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    print(f"  [遗言] 请{_player_display(state, exiled_id)}发表遗言")

    registry = state.get("agent_registry")
    if registry:
        result = _dispatch_agent(
            state,
            agent_exile_last_words,
            exiled_id,
            timeout_override=AGENT_TIMEOUTS.day_speech,
        )
        speech_text = result.get("speech_text", "") if result else ""
        print(f"  [遗言] {_player_display(state, exiled_id)}: {speech_text if speech_text else '(无遗言)'}")
        gs = replace(gs, events=gs.events + [GameEvent(
            type="exile_last_words",
            payload={"speaker": exiled_id, "day_number": gs.day_number, "text": speech_text},
        )])

    return {"game_state": gs}


def post_exile_skills(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    # Check if hunter died and can shoot
    for death in gs.deaths:
        if death.player_id in gs.players:
            player = gs.players[death.player_id]
            if player.role == "hunter" and not player.alive:
                if engine.can_hunter_shoot(gs, hunter_id=death.player_id, death_reason=death.reason):
                    target = state.get("hunter_shot_target_id")
                    if target:
                        shot_death = Death(
                            player_id=target, reason="hunter_shot",
                            timing="post_exile", resolution_batch=death.resolution_batch,
                            source_player_id=death.player_id,
                        )
                        gs = engine.apply_death(gs, shot_death)
    return {"game_state": gs}


def resolve_hunter_shot(state: RuntimeState) -> dict[str, Any]:
    """Resolve pending hunter shot after night wolf-kill, before victory check."""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]

    for death in gs.deaths:
        if "hunter_shot" not in (death.triggered_skills or []):
            continue
        if death.player_id not in gs.players:
            continue
        player = gs.players[death.player_id]
        if player.role != "hunter" or player.alive:
            continue
        # Skip if already resolved (death from this hunter already applied)
        already_shot = any(
            d.source_player_id == death.player_id and d.reason == "hunter_shot"
            for d in gs.deaths
        )
        if already_shot:
            continue

        # Get target: scripted, then agent
        target = state.get("hunter_shot_target_id")
        if target is None:
            target = _dispatch_agent(
                state,
                agent_hunter_shot,
                death.player_id,
                timeout_override=AGENT_TIMEOUTS.hunter_shot,
            )
        if target is None:
            target = _hunter_shot_target_from_last_words(gs, death.player_id)
        if target:
            shot_death = Death(
                player_id=target, reason="hunter_shot",
                timing=death.timing, resolution_batch=death.resolution_batch,
                source_player_id=death.player_id,
            )
            gs = engine.apply_death(gs, shot_death)
        break

    return {"game_state": gs}


def _hunter_shot_target_from_last_words(gs: GameState, hunter_id: str) -> str | None:
    """Extract an explicit hunter-shot target from the hunter's last words."""
    alive_targets = {
        pid for pid, player in gs.players.items()
        if player.alive and pid != hunter_id
    }
    if not alive_targets:
        return None
    for event in reversed(gs.events):
        if event.type not in {"exile_last_words", "night_death_last_words"}:
            continue
        payload = event.payload or {}
        if payload.get("speaker") != hunter_id:
            continue
        text = str(payload.get("text") or "")
        for match in re.finditer(r"(?:带走|开枪(?:带走|打)?|枪(?:带走|打)?|选择带走)\s*(p\d{2}|[A-Za-z]\w*)", text):
            candidate = match.group(1)
            if candidate in alive_targets:
                return candidate
    return None


def check_victory(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.check_victory(gs)
    checked_payload = {"winner": result.winner, "reason": result.reason}
    gs = replace(gs, events=gs.events + [GameEvent(type="victory_checked", payload=checked_payload)])

    if result.winner is not None:
        wf = result.winner
        print(f"\n{'='*60}")
        print(f"  【游戏结束】胜利方: {'好人阵营' if wf == 'good' else '狼人阵营'} ({result.reason})")
        print(f"{'='*60}")
        hr = None
        if wf == "good" and gs.hybrid_master_faction == "good":
            hr = "win"
        elif wf == "good" and gs.hybrid_master_faction == "werewolf":
            hr = "lose"
        elif wf == "werewolf" and gs.hybrid_master_faction == "werewolf":
            hr = "win"
        elif wf == "werewolf" and gs.hybrid_master_faction == "good":
            hr = "lose"
        gs = replace(gs, winning_faction=wf, hybrid_result=hr,
                     events=gs.events + [GameEvent(
                         type="victory",
                         payload={
                             "winner": wf,
                             "winning_faction": wf,
                             "reason": result.reason,
                             "hybrid_master_id": gs.hybrid_master_id,
                             "hybrid_master_faction": gs.hybrid_master_faction,
                             "hybrid_result": hr,
                         },
                     )])
    return {"game_state": gs, "_victory_result": result}


def sheriff_badge_transfer(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return {"game_state": gs}
    sheriff = gs.players.get(gs.sheriff_id)
    if sheriff is None or sheriff.alive:
        return {"game_state": gs}

    gs, _ = _judge_broadcast(
        phase="badge_decision",
        message=f"警长{_player_display(state, gs.sheriff_id)}死亡，请决定警徽去向",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )
    print(f"  [警徽] 警长{_player_display(state, gs.sheriff_id)}死亡，决定警徽去向")

    decision = state.get("badge_decision")
    target_id = state.get("badge_target_id")

    # Agent-driven: dying sheriff decides transfer or tear
    if decision is None:
        result = _dispatch_agent(
            state,
            agent_badge_decision,
            gs.sheriff_id,
            timeout_override=AGENT_TIMEOUTS.day_vote,
        )
        if result:
            decision = result.get("badge_decision", "tear")
            target_id = result.get("badge_target_id")

    if decision is None:
        decision = "tear"

    gs = engine.resolve_badge_decision(gs, decision=decision, target_id=target_id)
    event_type = "badge_torn" if decision == "tear" else "badge_transferred"

    if decision == "transfer" and target_id:
        gs, _ = _judge_broadcast(
            phase="badge_transferred",
            message=f"警长将警徽移交给{_player_display(state, target_id)}",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [警徽] 警长将警徽移交给 {_player_display(state, target_id)}")
    else:
        gs, _ = _judge_broadcast(
            phase="badge_torn",
            message="警长撕毁了警徽，本局不再有警长",
            gs=gs, day_number=gs.day_number,
            visibility="public",
        )
        print(f"  [警徽] 警长撕毁了警徽")

    gs = replace(gs, events=gs.events + [GameEvent(
        type=event_type,
        payload={"new_sheriff_id": target_id} if decision == "transfer" else {},
    )])
    return {"game_state": gs}


def finish_game(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    gs = replace(gs, phase="finished",
                 events=gs.events + [GameEvent(type="game_finished", payload={})])
    return {"game_state": gs}


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def _sheriff_died_this_batch(gs: GameState) -> bool:
    """Check if the sheriff died in the current resolution batch."""
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return False
    sheriff = gs.players.get(gs.sheriff_id)
    return sheriff is not None and not sheriff.alive


def route_after_resolve_night(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    # Check for pending hunter shot first (must resolve before victory check)
    for death in gs.deaths:
        if "hunter_shot" in (death.triggered_skills or []):
            if death.player_id in gs.players and not gs.players[death.player_id].alive:
                already_shot = any(
                    d.source_player_id == death.player_id and d.reason == "hunter_shot"
                    for d in gs.deaths
                )
                if not already_shot:
                    return "resolve_hunter_shot"
    engine: RuleEngine = state["engine"]
    result = engine.check_victory(gs)
    if result.winner is not None:
        return "check_victory"
    # If sheriff died at night and game continues, route to badge transfer
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    # Badge permanently lost after 2 interruptions
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    # Sheriff election needed: first night, or resumed after interruption
    needs_sheriff = False
    if gs.night_number == 1 and gs.sheriff_interrupt_count == 0 and gs.sheriff_id is None:
        needs_sheriff = True
    elif gs.sheriff_interrupt_count > 0 and gs.sheriff_interrupt_count < 2 and gs.sheriff_id is None:
        needs_sheriff = True
    if needs_sheriff:
        return "sheriff_first_day_entry"
    return "announce_deaths"


def route_after_hunter_shot(state: RuntimeState) -> str:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    result = engine.check_victory(gs)
    if result.winner is not None:
        return "check_victory"
    # If sheriff died at night and game continues, route to badge transfer
    if _sheriff_died_this_batch(gs):
        return "sheriff_badge_transfer"
    if gs.phase != "night":
        return "check_victory"
    # Badge permanently lost after 2 interruptions
    if gs.sheriff_interrupt_count >= 2 and gs.sheriff_id is None:
        return "announce_deaths_with_badge_loss"
    # Sheriff election needed: first night, or resumed after interruption
    needs_sheriff = False
    if gs.night_number == 1 and gs.sheriff_interrupt_count == 0 and gs.sheriff_id is None:
        needs_sheriff = True
    elif gs.sheriff_interrupt_count > 0 and gs.sheriff_interrupt_count < 2 and gs.sheriff_id is None:
        needs_sheriff = True
    if needs_sheriff:
        return "sheriff_first_day_entry"
    return "announce_deaths"


def route_after_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    # Check the last vote_resolved event for routing info
    for event in reversed(gs.events):
        if event.type == "vote_resolved":
            if event.payload.get("exiled") is not None:
                return "resolve_exile"
            if event.payload.get("reason") == "first_tie_pk":
                return "tie_pk_speech"
            break
    return "check_victory"


def route_after_exile(state: RuntimeState) -> str:
    return "post_exile_skills"


def route_after_post_exile(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    for death in gs.deaths:
        if "hunter_shot" not in (death.triggered_skills or []):
            continue
        if death.player_id not in gs.players:
            continue
        if gs.players[death.player_id].alive:
            continue
        already_shot = any(
            d.source_player_id == death.player_id and d.reason == "hunter_shot"
            for d in gs.deaths
        )
        if not already_shot:
            return "resolve_hunter_shot"
    return "check_victory"


def _route_after_badge_transfer(state: RuntimeState) -> str:
    """After badge transfer, route based on context.

    Night-death path: sheriff died at night -> badge transfer -> announce_deaths.
    Post-victory path: check_victory -> badge transfer -> enter_night.
    """
    gs: GameState = state["game_state"]
    # If phase is still "night", we're in the night-death path and need to
    # proceed to announce_deaths (which sets phase to "day").
    if gs.phase == "night":
        return "announce_deaths"
    return "enter_night"


def route_victory(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if gs.winning_faction is not None:
        return "finish_game"
    # Sheriff badge transfer if sheriff died
    if gs.sheriff_id and gs.sheriff_badge_state == "active":
        sheriff = gs.players.get(gs.sheriff_id)
        if sheriff and not sheriff.alive:
            return "sheriff_badge_transfer"
    return "enter_night"


def route_after_announce(state: RuntimeState) -> str:
    return "free_discussion"


def route_after_sheriff_vote(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    return "announce_deaths"


def _route_after_sheriff_phase(state: RuntimeState, default_next: str) -> str:
    wolf_id = state.get("self_destruct_wolf_id")
    gs: GameState = state["game_state"]
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    return default_next


def route_after_sheriff_registration(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_speech")


def route_after_sheriff_speech(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    if is_all_players_on_sheriff(gs, list(gs.sheriff_candidates or [])):
        return _route_after_sheriff_phase(state, "announce_deaths")
    return _route_after_sheriff_phase(state, "sheriff_withdraw")


def route_after_sheriff_withdraw(state: RuntimeState) -> str:
    return _route_after_sheriff_phase(state, "sheriff_vote")


def route_self_destruct_check(state: RuntimeState) -> str:
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id and wolf_id in gs.players and gs.players[wolf_id].alive and gs.players[wolf_id].role == "werewolf":
        return "resolve_self_destruct"
    speech_order = state.get("speech_order", [])
    speech_index = state.get("speech_index", 0)
    if speech_order and speech_index < len(speech_order):
        return "continue_discussion"
    return "day_vote"


def resolve_self_destruct_node(state: RuntimeState) -> dict[str, Any]:
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    wolf_id = state.get("self_destruct_wolf_id")
    if wolf_id:
        gs, events = engine.resolve_self_destruct(gs, wolf_id=wolf_id, day_number=gs.day_number)
        # If sheriff election was in progress (no sheriff yet), track interruption
        if gs.sheriff_id is None or gs.sheriff_badge_state == "none":
            gs = replace(gs,
                         sheriff_interrupt_count=gs.sheriff_interrupt_count + 1,
                         sheriff_candidates=[])
            count = gs.sheriff_interrupt_count
            gs, _ = _judge_broadcast(
                phase="sheriff_interrupted",
                message=f"竞选过程中有人自爆，警长竞选中断（第{count}次中断）",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )
        gs = replace(gs, events=gs.events + events)
    return {"game_state": gs, "self_destruct_wolf_id": None}


def tie_pk_speech(state: RuntimeState) -> dict[str, Any]:
    """After first exile tie, only PK candidates give speeches."""
    gs: GameState = state["game_state"]
    pk_candidates = state.get("pk_candidates", [])
    registry = state.get("agent_registry")
    events: list[GameEvent] = []

    if registry and pk_candidates:
        for candidate_id in pk_candidates:
            result = _dispatch_agent(
                state,
                agent_pk_speech,
                candidate_id,
                timeout_override=AGENT_TIMEOUTS.day_speech,
            )
            speech_text = result.get("speech_text", "") if result else ""
            print(f"  [PK发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
            events.append(GameEvent(
                type="tie_pk_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": speech_text,
                },
            ))
            if result and result.get("action_trace"):
                events.append(_action_trace_event(
                    player_id=candidate_id,
                    phase="pk_speech",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                ))
    else:
        events.append(GameEvent(type="tie_pk_speech", payload={}))

    gs = replace(gs, events=gs.events + events)
    return {"game_state": gs}


def tie_revote(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]
    return {
        "exile_votes": {},
        "exile_vote_day": gs.day_number,
        "exile_vote_revote": True,
        "revote": True,
    }


def wolf_discussion(state: RuntimeState) -> dict[str, Any]:
    """Run multi-round private wolf strategy and produce a team plan."""
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="wolf_wake",
        message="狼人请睁眼",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    gs, _ = _judge_broadcast(
        phase="wolf_discussion_start",
        message="狼人开始讨论今晚的行动",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    registry = state.get("agent_registry")

    if not registry:
        gs, _ = _judge_broadcast(
            phase="wolf_discussion_end",
            message="狼人讨论完毕",
            gs=gs, night_number=gs.night_number,
            visibility="moderator_only",
        )
        gs = replace(gs, events=gs.events + [GameEvent(type="wolf_discussion", payload={})])
        return {"game_state": gs}

    engine: RuleEngine = state["engine"]
    wolves = _alive_wolves(gs)
    events: list[GameEvent] = []
    round_count = 3 if gs.night_number == 1 else 2
    print(f"  [狼人密谈] 狼人: {[_player_display(state, w) for w in wolves]}，共{round_count}轮")
    for round_number in range(1, round_count + 1):
        round_state = dict(state)
        round_state["wolf_discussion_round"] = round_number
        for wolf_id in wolves:
            round_state["game_state"] = gs  # Latest gs with accumulated speeches
            result = _dispatch_agent(
                round_state,
                agent_wolf_discussion,
                wolf_id,
                timeout_override=AGENT_TIMEOUTS.wolf_discussion_per_player,
            )
            speech_text = result.get("speech_text", "") if result else ""
            print(
                f"    [第{round_number}轮] {_player_display(state, wolf_id)}(狼人): "
                f"{speech_text if speech_text else '(沉默)'}"
            )
            disc_event = GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": wolf_id,
                    "round": round_number,
                    "night_number": gs.night_number,
                    "text": speech_text,
                    "visibility": "werewolf_team_only",
                },
            )
            # Immediately merge into gs so next wolf sees this speech
            gs = replace(gs, events=gs.events + [disc_event])
            events.append(disc_event)
            if result and result.get("action_trace"):
                trace_event = _action_trace_event(
                    player_id=wolf_id,
                    phase=f"wolf_discussion_round_{round_number}",
                    action_trace=result["action_trace"],
                    day_number=gs.day_number,
                    night_number=gs.night_number,
                )
                gs = replace(gs, events=gs.events + [trace_event])
                events.append(trace_event)

        # Check if wolves reached consensus after this round — end early if so
        if round_number < round_count:
            from werewolf_agent.runtime.wolf_strategy import (
                should_end_discussion_early,
                summarize_wolf_consensus,
            )
            mid_consensus = summarize_wolf_consensus(gs.events, wolves)
            if should_end_discussion_early(mid_consensus, len(wolves)):
                print(f"  [狼人密谈] 第{round_number}轮已达成共识，提前结束讨论")
                break

    # Aggregate discussion into consensus plan, fallback to static plan
    # (events already merged incrementally into gs)
    from werewolf_agent.runtime.wolf_strategy import (
        build_wolf_team_plan_from_discussion,
        summarize_wolf_consensus,
    )
    consensus = summarize_wolf_consensus(gs.events, wolves)
    plan = build_wolf_team_plan_from_discussion(
        gs,
        previous_plan=state.get("wolf_team_plan"),
        consensus=consensus,
    )

    # Fallback to static plan when consensus lacks critical fields
    static_plan = _build_wolf_team_plan(gs, previous_plan=state.get("wolf_team_plan"))
    for key in ("fake_seer", "pusher", "hooker", "deep_cover", "public_story"):
        if not plan.get(key) and static_plan.get(key):
            plan[key] = static_plan[key]

    # Log consensus summary
    primary = plan.get("night_kill_primary")
    backup = plan.get("night_kill_backup")
    agreement = consensus.get("agreement_count", 0)
    total = consensus.get("total_wolves", len(wolves))
    if primary:
        print(f"  [狼队共识] 主目标: {_player_display(state, primary)}, 备选: {_player_display(state, backup) if backup else '无'}, "
              f"共识度: {agreement}/{total}")
    else:
        print(f"  [狼队共识] 未达成击杀共识 ({agreement}/{total} 同意)")

    events.append(GameEvent(
        type="wolf_team_plan",
        payload={**plan, "visibility": "werewolf_team_only"},
    ))
    gs = replace(gs, events=gs.events + events[-1:])  # Add plan event
    gs, _ = _judge_broadcast(
        phase="wolf_discussion_end",
        message="狼人讨论完毕",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    return {"game_state": gs, "wolf_team_plan": plan}


def wolf_consensus(state: RuntimeState) -> dict[str, Any]:
    """Determine wolf night action, preferring the private team plan."""
    gs: GameState = state["game_state"]
    gs, _ = _judge_broadcast(
        phase="wolf_kill_choice",
        message="狼人请统一选择今晚的行动",
        gs=gs, night_number=gs.night_number,
        visibility="moderator_only",
    )
    state = {**state, "game_state": gs}
    _timeout_contract = AGENT_TIMEOUTS.wolf_consensus
    planned = _planned_wolf_kill(state)
    if planned is not None and not state.get("wolf_action"):
        result = planned
    else:
        result = _legacy_wolf_consensus(state)
    result_gs = result.get("game_state", gs)
    result_gs, _ = _judge_broadcast(
        phase="wolf_sleep",
        message="狼人请闭眼",
        gs=result_gs,
        night_number=result_gs.night_number,
        visibility="moderator_only",
    )
    return {**result, "game_state": result_gs}


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
            message=f"警上发言顺序: {names}",
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
            print(f"  [警上发言] {_player_display(state, candidate_id)}: {speech_text if speech_text else '(未发言)'}")
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


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_game_graph() -> CompiledStateGraph:
    graph = StateGraph(RuntimeState)
    _add_all_nodes(graph)
    _add_all_edges(graph)
    return graph.compile()


def build_game_graph_with_checkpoint(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build the graph with an external checkpointer for pause/resume."""
    graph = StateGraph(RuntimeState)
    _add_all_nodes(graph)
    _add_all_edges(graph)
    return graph.compile(checkpointer=checkpointer)


def _add_all_nodes(graph: StateGraph) -> None:
    graph.add_node("setup_game", setup_game)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("enter_night", enter_night)
    graph.add_node("wolf_discussion", wolf_discussion)
    graph.add_node("wolf_consensus", wolf_consensus)
    graph.add_node("night_witch", night_witch)
    graph.add_node("night_seer", night_seer)
    graph.add_node("night_hunter_idiot_status", night_hunter_idiot_status)
    graph.add_node("first_night_hybrid_master", first_night_hybrid_master)
    graph.add_node("resolve_night_node", resolve_night)
    graph.add_node("sheriff_first_day_entry", sheriff_first_day_entry)
    graph.add_node("announce_deaths", announce_deaths)
    graph.add_node("announce_deaths_with_badge_loss", announce_deaths_with_badge_loss)
    graph.add_node("night_death_last_words", night_death_last_words)
    graph.add_node("sheriff_registration", sheriff_registration)
    graph.add_node("sheriff_speech", sheriff_speech)
    graph.add_node("sheriff_withdraw", sheriff_withdraw)
    graph.add_node("sheriff_vote", sheriff_vote)
    graph.add_node("free_discussion", free_discussion)
    graph.add_node("resolve_self_destruct", resolve_self_destruct_node)
    graph.add_node("day_vote", day_vote)
    graph.add_node("resolve_vote_node", resolve_vote)
    graph.add_node("tie_pk_speech", tie_pk_speech)
    graph.add_node("tie_revote", tie_revote)
    graph.add_node("resolve_exile", resolve_exile)
    graph.add_node("exile_last_words", exile_last_words)
    graph.add_node("post_exile_skills", post_exile_skills)
    graph.add_node("resolve_hunter_shot", resolve_hunter_shot)
    graph.add_node("check_victory", check_victory)
    graph.add_node("sheriff_badge_transfer", sheriff_badge_transfer)
    graph.add_node("finish_game", finish_game)


def _add_all_edges(graph: StateGraph) -> None:
    graph.set_entry_point("setup_game")
    graph.add_edge("setup_game", "assign_roles")
    graph.add_edge("assign_roles", "enter_night")
    graph.add_edge("enter_night", "wolf_discussion")
    graph.add_edge("wolf_discussion", "wolf_consensus")
    graph.add_edge("wolf_consensus", "night_witch")
    graph.add_edge("night_witch", "night_seer")
    graph.add_edge("night_seer", "night_hunter_idiot_status")
    graph.add_edge("night_hunter_idiot_status", "first_night_hybrid_master")
    graph.add_edge("first_night_hybrid_master", "resolve_night_node")
    graph.add_conditional_edges("resolve_night_node", route_after_resolve_night, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "check_victory": "check_victory",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    graph.add_conditional_edges("resolve_hunter_shot", route_after_hunter_shot, {
        "check_victory": "check_victory",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "sheriff_first_day_entry": "sheriff_first_day_entry",
        "announce_deaths": "announce_deaths",
        "announce_deaths_with_badge_loss": "announce_deaths_with_badge_loss",
    })
    # Day 1 / resumed sheriff: entry → election → announce_deaths
    graph.add_edge("sheriff_first_day_entry", "sheriff_registration")
    graph.add_conditional_edges("sheriff_registration", route_after_sheriff_registration, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_speech": "sheriff_speech",
    })
    graph.add_conditional_edges("sheriff_speech", route_after_sheriff_speech, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_withdraw": "sheriff_withdraw",
        "announce_deaths": "announce_deaths",
    })
    graph.add_conditional_edges("sheriff_withdraw", route_after_sheriff_withdraw, {
        "resolve_self_destruct": "resolve_self_destruct",
        "sheriff_vote": "sheriff_vote",
    })
    graph.add_conditional_edges("sheriff_vote", route_after_sheriff_vote, {
        "resolve_self_destruct": "resolve_self_destruct",
        "announce_deaths": "announce_deaths",
    })
    # Day 2+ normal: announce_deaths → last_words → free_discussion
    graph.add_edge("announce_deaths", "night_death_last_words")
    graph.add_edge("announce_deaths_with_badge_loss", "night_death_last_words")
    graph.add_conditional_edges("night_death_last_words", route_after_announce, {
        "free_discussion": "free_discussion",
    })
    graph.add_conditional_edges("free_discussion", route_self_destruct_check, {
        "resolve_self_destruct": "resolve_self_destruct",
        "continue_discussion": "free_discussion",
        "day_vote": "day_vote",
    })
    graph.add_edge("resolve_self_destruct", "check_victory")
    graph.add_edge("day_vote", "resolve_vote_node")
    graph.add_conditional_edges("resolve_vote_node", route_after_vote, {
        "resolve_exile": "resolve_exile",
        "tie_pk_speech": "tie_pk_speech",
        "check_victory": "check_victory",
    })
    graph.add_edge("tie_pk_speech", "tie_revote")
    graph.add_edge("tie_revote", "day_vote")
    graph.add_edge("resolve_exile", "exile_last_words")
    graph.add_edge("exile_last_words", "post_exile_skills")
    graph.add_conditional_edges("post_exile_skills", route_after_post_exile, {
        "resolve_hunter_shot": "resolve_hunter_shot",
        "check_victory": "check_victory",
    })
    graph.add_conditional_edges("check_victory", route_victory, {
        "finish_game": "finish_game",
        "sheriff_badge_transfer": "sheriff_badge_transfer",
        "enter_night": "enter_night",
    })
    graph.add_conditional_edges("sheriff_badge_transfer", _route_after_badge_transfer, {
        "announce_deaths": "announce_deaths",
        "enter_night": "enter_night",
    })
    graph.add_edge("finish_game", END)
