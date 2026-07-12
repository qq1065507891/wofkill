# -*- coding: utf-8 -*-
"""
运行时节点共享的通用调度、终局调用防御、裁判广播和流程判断 helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

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
from werewolf_agent.runtime.nodes.judge_broadcast_helpers import (
    _generate_judge_message,
    _jb,
    _judge_broadcast,
)
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
    post_game: bool = False,
    **extra_kwargs,
) -> dict[str, Any] | None:
    """检查 registry 后调度 agent adapter。"""
    gs = state.get("game_state")
    if gs is not None and gs.winning_faction is not None and not post_game:
        return None
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


def _hunter_reaction_resolved(
    gs: GameState,
    hunter_id: str,
    resolution_batch: str,
) -> bool:
    """按猎人和死亡批次判断开枪或放弃反应是否已经完成。"""
    if any(
        death.source_player_id == hunter_id
        and death.reason == "hunter_shot"
        and death.resolution_batch == resolution_batch
        for death in gs.deaths
    ):
        return True
    return any(
        event.type == "hunter_shot_declined"
        and event.payload.get("hunter_id") == hunter_id
        and event.payload.get("resolution_batch") == resolution_batch
        for event in gs.events
    )


def _has_pending_hunter_shot(gs: GameState) -> bool:
    """判断死亡批次中是否仍有必须先结算的猎人开枪。"""
    for death in gs.deaths:
        if "hunter_shot" not in (death.triggered_skills or []):
            continue
        player = gs.players.get(death.player_id)
        if player is None or player.alive:
            continue
        if not _hunter_reaction_resolved(gs, death.player_id, death.resolution_batch):
            return True
    return False


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
    if plan.get("consensus_method") == "fallback" and evidence_quality == "none":
        return None
    evidence = plan.get("evidence_from_discussion") or []
    # 弱证据计划必须在执行边界重新验证独立狼人多数。计划可能来自
    # LLM 或 fallback，不能仅凭生成阶段的 evidence_quality 放行。
    if evidence_quality == "weak":
        alive_wolves = set(_alive_wolves(gs))
        quorum = len(alive_wolves) // 2 + 1
        supporters_by_target: dict[str, set[str]] = {}
        for item in evidence:
            if not isinstance(item, dict):
                continue
            target_id = item.get("target")
            wolf_id = item.get("wolf_id")
            if target_id and wolf_id in alive_wolves:
                supporters_by_target.setdefault(target_id, set()).add(wolf_id)
        qualified = {
            target_id: supporters
            for target_id, supporters in supporters_by_target.items()
            if len(supporters) >= quorum
        }
        # 两个目标同票时保持安全空刀，避免按计划顺序隐式裁决平票。
        if qualified:
            max_support = max(len(supporters) for supporters in qualified.values())
            leaders = [
                target_id
                for target_id, supporters in qualified.items()
                if len(supporters) == max_support
            ]
            if len(leaders) != 1:
                event = GameEvent(
                    type="wolf_no_kill_timeout",
                    payload={
                        "night_number": gs.night_number,
                        "reason": "weak_plan_quorum_tie",
                        "quorum": quorum,
                        "supporters": {
                            target_id: sorted(supporters)
                            for target_id, supporters in supporters_by_target.items()
                        },
                    },
                )
                return {
                    "game_state": replace(gs, events=gs.events + [event]),
                    "wolf_kill_target_id": None,
                }
        else:
            # 完全没有讨论证据时保持旧调用契约，由上层继续走通用安全路径；
            # 只有存在但不足的证据才记录明确的 quorum 空刀。
            if not supporters_by_target:
                return None
            event = GameEvent(
                type="wolf_no_kill_timeout",
                payload={
                    "night_number": gs.night_number,
                    "reason": "weak_plan_quorum_not_met",
                    "quorum": quorum,
                    "supporters": {
                        target_id: sorted(supporters)
                        for target_id, supporters in supporters_by_target.items()
                    },
                },
            )
            return {
                "game_state": replace(gs, events=gs.events + [event]),
                "wolf_kill_target_id": None,
            }
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
        if plan.get("consensus_method") == "fallback" and not has_target_evidence:
            continue
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
    "_has_pending_hunter_shot",
    "_hunter_reaction_resolved",
    "_jb",
    "_judge_broadcast",
    "_needs_sheriff_before_deaths",
    "_planned_wolf_kill",
    "_player_display",
    "_player_ids",
    "_sheriff_died_this_batch",
    "_timer_expired",
]
