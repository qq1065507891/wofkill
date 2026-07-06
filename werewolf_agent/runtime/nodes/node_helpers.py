# -*- coding: utf-8 -*-
"""
运行时节点共享的通用调度、裁判广播、狼队计划和流程判断 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.node_helpers import _alive_wolves
    >>> _alive_wolves(game_state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.nodes.runtime_state import RuntimeState, _stable_seed
from werewolf_agent.runtime.timers import timed_call
from werewolf_agent.runtime.timeline import phase_label


logger = logging.getLogger("werewolf_agent.runtime.nodes._shared")


def _player_ids(gs: GameState) -> list[str]:
    return list(gs.players.keys())


def _alive_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role == "werewolf"]


def _alive_non_wolves(gs: GameState) -> list[str]:
    return [pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"]


def _force_wolf_kill(gs: GameState, reason: str) -> dict[str, Any]:
    """强制狼人击杀一个存活非狼人。"""
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
    """返回单次 agent 调用超时时间，0 表示不包装。"""
    return float(state.get("agent_call_timeout") or 0)


def _player_display(state: RuntimeState, player_id: str) -> str:
    """返回展示名，例如 '陈思远(p01)'。"""
    registry = state.get("agent_registry")
    if registry:
        agent = registry.get_agent(player_id)
        if agent and hasattr(agent, "player_name") and agent.player_name != player_id:
            return f"{agent.player_name}({player_id})"
    return player_id


def _call_agent(fn, state: RuntimeState, *args, timeout_override: float | None = None, **kwargs):
    """调用 agent adapter，可按配置包一层 timeout。"""
    timeout = timeout_override if timeout_override is not None else _agent_timeout(state)
    if timeout > 0:
        if kwargs:
            return timed_call(lambda *inner_args: fn(*inner_args, **kwargs), *args, timeout=timeout)
        return timed_call(fn, *args, timeout=timeout, **kwargs)
    return fn(*args, **kwargs)


def _dispatch_agent(
    state: RuntimeState,
    fn,
    *extra_args,
    timeout_override: float | None = None,
    **extra_kwargs,
) -> dict[str, Any] | None:
    """检查 registry 后调度 agent adapter。"""
    registry = state.get("agent_registry")
    if not registry:
        return None
    delay_ms = state.get("agent_call_delay_ms", -1)
    if delay_ms == 0:
        delay_ms = 10000
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
        **extra_kwargs,
        timeout_override=timeout_override,
    )


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
    """创建裁判广播事件并追加到 GameState。"""
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
            pass

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
    """调度 JudgeAgent 对应方法，失败时返回空字符串。"""
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
    """从 RuntimeState 提取 JudgeAgent 后创建裁判广播。"""
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
    """必要时递增 day_number 并广播天亮。"""
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
    """HITL checkpoint：需要时暂停并冲刷审计事件。"""
    if not state.get("judge_hitl_enabled"):
        return {}
    hitl = state.get("judge_hitl")
    if hitl is None:
        return {}
    if not hitl.should_pause(phase, direction):
        return {}
    gs = state["game_state"]
    cmd = hitl.wait_for_human(timeout=300)
    if cmd is not None:
        result = hitl.handle_command(cmd, gs)
        if "game_state" in result:
            state["game_state"] = result["game_state"]
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
    day_push = _first_alive_target(gs, previous_plan.get("day_push_target"))

    if len(wolves) == 1 and not primary:
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer
        claimed_seer_target: str | None = None
        for pid, p in gs.players.items():
            if p.alive and p.role != "werewolf" and has_publicly_claimed_seer(gs, pid):
                claimed_seer_target = pid
                break
        if claimed_seer_target:
            primary = claimed_seer_target
        elif day_push:
            primary = day_push

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
    evidence_quality = plan.get("evidence_quality")
    if evidence_quality == "none":
        return None
    if plan.get("consensus_method") == "fallback" and evidence_quality != "strong":
        return None
    evidence = plan.get("evidence_from_discussion") or []
    evidenced_targets = {
        item.get("target")
        for item in evidence
        if isinstance(item, dict) and item.get("target")
    }
    primary_alive = _first_alive_target(gs, plan.get("night_kill_primary"))
    primary_unavailable = primary_alive is None
    for key in ("night_kill_primary", "night_kill_backup"):
        target = _first_alive_target(gs, plan.get(key))
        if target is None:
            continue
        has_target_evidence = target in evidenced_targets
        if evidence_quality == "weak" and not has_target_evidence:
            continue
        if key == "night_kill_backup" and primary_unavailable and not has_target_evidence:
            continue
        if evidence_quality not in ("strong", "weak") and not has_target_evidence:
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
    """检查警长是否在当前结算批次死亡。"""
    if gs.sheriff_id is None or gs.sheriff_badge_state != "active":
        return False
    sheriff = gs.players.get(gs.sheriff_id)
    return sheriff is not None and not sheriff.alive


def _needs_sheriff_before_deaths(gs: GameState) -> bool:
    """首夜死亡公布前是否还需要警长竞选。"""
    return (
        gs.night_number == 1
        and gs.sheriff_interrupt_count == 0
        and gs.sheriff_id is None
        and gs.sheriff_badge_state not in ("torn", "active")
    )


def _deaths_already_announced(gs: GameState) -> bool:
    """检查当前白天是否已经广播过死亡公布。"""
    return any(
        e.type == "judge_broadcast"
        and e.payload.get("phase") == "death_announce"
        and e.payload.get("day_number") == gs.day_number
        for e in gs.events
    )


__all__ = [
    "_agent_timeout",
    "_alive_non_wolves",
    "_alive_wolves",
    "_build_wolf_team_plan",
    "_call_agent",
    "_deaths_already_announced",
    "_dispatch_agent",
    "_ensure_day_incremented",
    "_find_role",
    "_first_alive_target",
    "_force_wolf_kill",
    "_generate_judge_message",
    "_hitl_checkpoint",
    "_jb",
    "_judge_broadcast",
    "_needs_sheriff_before_deaths",
    "_planned_wolf_kill",
    "_player_display",
    "_player_ids",
    "_sheriff_died_this_batch",
    "_timer_expired",
]
