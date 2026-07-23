# -*- coding: utf-8 -*-
"""
运行时节点共享的通用调度、终局调用防御、裁判广播和流程判断 helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-20

使用示例:
    >>> from werewolf_agent.runtime.nodes.node_helpers import _alive_wolves
    >>> _alive_wolves(game_state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from werewolf_agent.agents.schemas import WolfTargetStance
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import (
    carrier_matches_resolution_batch,
    same_resolution_batch,
    valid_carrier_resolution_batch,
)
from werewolf_agent.runtime.nodes.judge_broadcast_helpers import (
    _generate_judge_message,
    _jb,
    _judge_broadcast,
)
from werewolf_agent.runtime.nodes.runtime_state import RuntimeState, _stable_seed
from werewolf_agent.runtime.event_metadata import (
    validate_v2_event_identity,
    validate_v2_event_log_identity,
)
from werewolf_agent.runtime.timeline import phase_label
from werewolf_agent.runtime.wolf_no_kill_policy import (
    NO_KILL_REASON_CODES,
    NoKillPolicy,
    NoKillReasonCode,
    no_kill_policy_for_state,
)
from werewolf_agent.runtime.wolf_decision_trace import (
    WOLF_KILL_FORCED_FALLBACK,
    new_wolf_decision_event,
    wolf_stance_kill_decision_kind,
)


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
        return NoKillPolicy().resolve(
            gs,
            reason_code="plan_generation_failed",
            extra_payload={"legacy_reason": reason},
        )
    rng = _random.Random(_stable_seed(gs.game_id, reason, gs.night_number))
    target = rng.choice(non_wolves)
    event = new_wolf_decision_event(
        gs,
        "wolf_kill_selected",
        {"night_number": gs.night_number, "target_id": target, "reason": reason},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        decision_kind=WOLF_KILL_FORCED_FALLBACK,
    )
    gs = replace(gs, events=gs.events + [event])
    return {"game_state": gs, "wolf_kill_target_id": target}


def _find_role(gs: GameState, role: str) -> str | None:
    return next((pid for pid, p in gs.players.items() if p.role == role and p.alive), None)


def _player_display(state: RuntimeState, player_id: str) -> str:
    """返回展示名，例如 '陈思远(p01)'。"""
    registry = state.get("agent_registry")
    if registry:
        agent = registry.get_agent(player_id)
        if agent and hasattr(agent, "player_name") and agent.player_name != player_id:
            return f"{agent.player_name}({player_id})"
    return player_id


def _dispatch_agent(
    state: RuntimeState,
    fn,
    *extra_args,
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
    try:
        return fn(
            state,
            engine,
            registry,
            *extra_args,
            **extra_kwargs,
        )
    except Exception as exc:
        logger.warning(
            "Agent dispatch failed for %s: %s: %s",
            getattr(fn, "__qualname__", repr(fn)),
            type(exc).__name__,
            exc,
        )
        return None


def _hunter_reaction_resolved(
    gs: GameState,
    hunter_death: Death,
) -> bool:
    """按猎人和死亡批次判断开枪或放弃反应是否已经完成。"""
    resolution_batch = valid_carrier_resolution_batch(hunter_death)
    if resolution_batch is None:
        # 损坏批次不能驱动技能链；视为无需继续结算，避免无限路由。
        return True
    if any(
        death.source_player_id == hunter_death.player_id
        and death.reason == "hunter_shot"
        and carrier_matches_resolution_batch(death, resolution_batch)
        for death in gs.deaths
    ):
        return True
    return any(
        event.type == "hunter_shot_declined"
        and event.payload.get("hunter_id") == hunter_death.player_id
        and same_resolution_batch(
            event.payload.get("resolution_batch", ""),
            resolution_batch,
            left_parse_failed=bool(
                event.payload.get("resolution_batch_parse_failed", False)
            ),
        )
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
        if not _hunter_reaction_resolved(gs, death):
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


def _trusted_wolf_plan_failure_reason(
    gs: GameState,
) -> NoKillReasonCode | None:
    """仅从本夜私有 V2 fallback 事件读取失败类别，不授予目标执行权。

    区分两类失败:
    - provider/captain/registry 不可用 → ``provider_unavailable`` (用于 no_kill
      reason 标签, 让审计看到这是工具栈问题, 不是协议失败)
    - LLM 输出了结构化内容但 schema/membership/parse 不通过 →
      原 ``reason`` 字符串直接透传 (例如 ``schema_validation_failed`` /
      ``json_parse_failed`` / ``membership_validation_failed`` /
      ``empty_response`` / ``generate_error``), 用于精确诊断 LLM 战术层问题。
    """
    if not _v2_event_log_identity_is_authoritative(gs):
        return None
    _PROVIDER_UNAVAILABLE_REASONS = frozenset({
        "llm_failed_or_unavailable",
        "provider_unavailable",
        "captain_agent_missing",
        "no_registry",
        "no_alive_wolves",
    })
    for event in reversed(gs.events):
        if event.type != "wolf_team_plan_fallback":
            continue
        try:
            validate_v2_event_identity(
                gs.game_id,
                event,
                required_visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            )
        except ValueError:
            continue
        if event.payload.get("night_number") != gs.night_number:
            continue
        raw_reason = event.payload.get("reason")
        if not isinstance(raw_reason, str):
            continue
        normalized = raw_reason.strip().lower()
        if (
            normalized in _PROVIDER_UNAVAILABLE_REASONS
            or normalized.startswith((
                "agent_exception:",
                "provider_exception:",
                "model_exception:",
            ))
        ):
            return "provider_unavailable"
        # 原样透传 agent_wolf_team_plan 的真实失败 reason
        return raw_reason
    return None


def _v2_event_log_identity_is_authoritative(gs: GameState) -> bool:
    """校验日志中所有 V2 身份唯一且按事件顺序严格递增。"""
    try:
        validate_v2_event_log_identity(gs.game_id, gs.events)
    except ValueError:
        return False
    return True


def _planned_wolf_kill(state: RuntimeState) -> dict[str, Any] | None:
    """只依据本夜结构化 stance 共识选择主刀，必要时再读取备刀。"""
    from werewolf_agent.runtime.wolf_consensus_evidence import (
        ConsensusInvariantViolation,
        WolfPriorityConsensus,
        derive_wolf_consensus_evidence,
    )
    from werewolf_agent.runtime.wolf_discussion_directives import (
        collect_current_wolf_target_stances,
    )

    gs: GameState = state["game_state"]
    alive_wolves = tuple(_alive_wolves(gs))
    logger.debug(
        "  [狼人决策] _planned_wolf_kill 入口: night=%d alive_wolves=%s",
        gs.night_number,
        list(alive_wolves),
    )
    if not alive_wolves:
        logger.debug("  [狼人决策] _planned_wolf_kill 无存活狼人, 返回 None")
        return None

    raw_stances = collect_current_wolf_target_stances(gs)
    stances = tuple(
        WolfTargetStance.model_validate(raw_stance)
        for raw_stance in raw_stances
    )
    logger.debug(
        "  [狼人决策] _planned_wolf_kill stance 数=%d",
        len(stances),
    )
    try:
        consensus = derive_wolf_consensus_evidence(
            gs.night_number,
            alive_wolves,
            stances,
        )
    except ConsensusInvariantViolation as exc:
        logger.debug(
            "  [狼人决策] _planned_wolf_kill 共识不变量违反: reason=%s priority=%s targets=%s",
            exc.reason_code,
            exc.priority,
            list(exc.targets),
        )
        event = GameEvent(
            type="wolf_consensus_invariant_violation",
            payload={
                "night_number": gs.night_number,
                "reason": exc.reason_code,
                "priority": exc.priority,
                "targets": list(exc.targets),
                "visibility": "moderator_only",
            },
        )
        invalid_gs = replace(gs, events=[*gs.events, event])
        return no_kill_policy_for_state(state).resolve(
            invalid_gs,
            reason_code="insufficient_quorum",
            extra_payload={
                "consensus_priority": exc.priority,
                "consensus_status": "invariant_violation",
            },
        )

    def positive_support(
        priority: WolfPriorityConsensus,
    ) -> dict[str, int]:
        return {
            target_id: len(supporters)
            for target_id, supporters in priority.supporters_by_target.items()
        }

    def no_kill(
        priority: WolfPriorityConsensus,
        reason: NoKillReasonCode,
        raw_reason: str | None = None,
    ) -> dict[str, Any]:
        logger.debug(
            "  [狼人决策] _planned_wolf_kill 空刀: reason=%s priority=%s status=%s quorum=%s supporters=%s",
            reason,
            priority.priority,
            priority.status,
            consensus.quorum,
            {
                target_id: list(supporters)
                for target_id, supporters
                in priority.supporters_by_target.items()
            },
        )
        extra_payload: dict[str, Any] = {
            "consensus_priority": priority.priority,
            "consensus_status": priority.status,
            "quorum": consensus.quorum,
            "supporters": {
                target_id: list(supporters)
                for target_id, supporters
                in priority.supporters_by_target.items()
            },
        }
        if raw_reason and raw_reason != reason:
            extra_payload["raw_reason"] = raw_reason
        return no_kill_policy_for_state(state).resolve(
            gs,
            reason_code=reason,
            primary_positive_support=positive_support(consensus.primary),
            backup_positive_support=positive_support(consensus.backup),
            extra_payload=extra_payload,
        )
    authorized_statuses = {"majority", "single_wolf"}
    # ``trusted_plan_failure`` 可能是 agent_wolf_team_plan 真实失败 reason
    # (如 schema_validation_failed / json_parse_failed / membership_validation_failed),
    # 这些不在 NoKillReasonCode 字面量枚举内, 必须规范化到 enum 才能让
    # no_kill_policy.resolve 接受, 同时把原 reason 透传到 extra_payload 用于诊断。
    _PLAN_GENERATION_FAILURE_FALLBACK: NoKillReasonCode = "plan_generation_failed"
    _NO_KILL_REASON_CODES_SET = NO_KILL_REASON_CODES
    raw_trusted = _trusted_wolf_plan_failure_reason(gs)
    if raw_trusted is not None and raw_trusted not in _NO_KILL_REASON_CODES_SET:
        trusted_plan_failure: NoKillReasonCode | None = _PLAN_GENERATION_FAILURE_FALLBACK
        raw_plan_failure_reason = raw_trusted
    else:
        trusted_plan_failure = raw_trusted  # type: ignore[assignment]
        raw_plan_failure_reason = None
    primary = consensus.primary
    logger.debug(
        "  [狼人决策] _planned_wolf_kill primary: status=%s target=%s supporters=%s quorum=%s",
        primary.status,
        primary.target_id,
        positive_support(primary),
        consensus.quorum,
    )
    if primary.status not in authorized_statuses or primary.target_id is None:
        reason_by_status: dict[str, NoKillReasonCode] = {
            "tie": "true_tie",
            "insufficient": "insufficient_quorum",
            "all_abstain": (
                trusted_plan_failure or "strategic_abstain"
            ),
        }
        return no_kill(
            primary, reason_by_status[primary.status],
            raw_reason=raw_plan_failure_reason,
        )

    primary_target = _first_alive_target(gs, primary.target_id)
    if primary_target is not None:
        selected_target = primary_target
        plan_key = "night_kill_primary"
        selected_consensus_status = primary.status
    else:
        backup = consensus.backup
        logger.debug(
            "  [狼人决策] _planned_wolf_kill primary 目标非法, 降级到 backup: backup_status=%s backup_target=%s",
            backup.status,
            backup.target_id,
        )
        if backup.status not in authorized_statuses or backup.target_id is None:
            reason_by_status = {
                "tie": "true_tie",
                "insufficient": "insufficient_quorum",
                "all_abstain": (
                    trusted_plan_failure or "strategic_abstain"
                ),
            }
            return no_kill(backup, reason_by_status[backup.status])
        selected_target = _first_alive_target(gs, backup.target_id)
        if selected_target is None:
            logger.debug(
                "  [狼人决策] _planned_wolf_kill backup 目标也非法, 空刀",
            )
            return no_kill(backup, "invalid_backup")
        plan_key = "night_kill_backup"
        selected_consensus_status = backup.status

    logger.debug(
        "  [狼人决策] 按结构化 stance 共识击杀: %s",
        _player_display(state, selected_target),
    )
    event = new_wolf_decision_event(
        gs,
        "wolf_kill_selected",
        {
            "night_number": gs.night_number,
            "target_id": selected_target,
            "reason": "wolf_stance_consensus",
            "plan_key": plan_key,
        },
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        decision_kind=wolf_stance_kill_decision_kind(
            consensus_status=selected_consensus_status,
            plan_key=plan_key,
        ),
    )
    gs = replace(gs, events=[*gs.events, event])
    return {"game_state": gs, "wolf_kill_target_id": selected_target}


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
    "_alive_non_wolves",
    "_alive_wolves",
    "_build_wolf_team_plan",
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
]
