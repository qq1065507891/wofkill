# -*- coding: utf-8 -*-
"""
提供狼人夜间击杀指令和单狼投票支持。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.wolf_kill_support import _build_wolf_kill_directive
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.agent_action_audit import _audit_context_kwargs
from werewolf_agent.runtime.agent_registry import AgentRegistry
from werewolf_agent.runtime.context import (
    _action_trace_payload,
    _merge_strategy_directive,
    build_agent_context,
)
from werewolf_agent.runtime.directives import (
    build_wolf_night_directive as _build_wolf_night_directive,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.strategy import (
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS

logger = logging.getLogger(__name__)


def _build_wolf_kill_directive(
    gs: GameState,
    *,
    wolf_id: str,
    plan: dict[str, Any] | None,
) -> str:
    """构造狼人击杀优先级提示。"""

    from werewolf_agent.runtime.strategy.wolf import (
        evaluate_wolf_kill_target,
        has_publicly_claimed_seer,
    )

    parts: list[str] = []

    claimed_seers: list[str] = []
    for pid, player in gs.players.items():
        if player.alive and player.role != "werewolf" and has_publicly_claimed_seer(gs, pid):
            claimed_seers.append(pid)

    if claimed_seers:
        names = ", ".join(claimed_seers)
        parts.append(
            f"高优先级击杀目标: {names} —— 该玩家已公开跳预言家，"
            "对狼队威胁最大，必须作为今晚的首选击杀目标。"
        )

    if plan and plan.get("night_kill_primary"):
        primary = plan["night_kill_primary"]
        if primary in gs.players and gs.players[primary].alive:
            if primary not in claimed_seers:
                parts.append(
                    f"狼队讨论主目标: {primary}（备选: {plan.get('night_kill_backup') or '无'}）"
                )

    if not parts or len(claimed_seers) == 0:
        scores = evaluate_wolf_kill_target(
            gs,
            wolf_id,
            [pid for pid, player in gs.players.items() if player.alive and player.role != "werewolf"],
        )
        if scores and scores.get("ranked_targets"):
            for entry in scores["ranked_targets"][:3]:
                parts.append(
                    f"击杀候选: {entry['target']}（威胁分={entry['value']}，"
                    f"信号: {', '.join(entry.get('signals', [])) or '无'}）"
                )

    if not parts:
        return "无明显优先目标，按战术需要自由选择击杀对象。"

    return "\n".join(parts)


def _single_wolf_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    wolf_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """获取单个狼人对击杀或空刀的投票。"""

    gs: GameState = state["game_state"]
    agent = registry.get_agent(wolf_id)
    if agent is None:
        return None

    legal_targets = [
        pid for pid, player in gs.players.items()
        if player.alive and player.role != "werewolf"
    ]

    kill_assessment = _evaluate_wolf_kill_target(gs, wolf_id, legal_targets)
    wolf_plan = state.get("wolf_team_plan")
    strategy_directive: dict[str, Any] = _build_wolf_night_directive(
        gs, wolf_id, wolf_plan,
    )
    strategy_directive["wolf_high_priority_target"] = _build_wolf_kill_directive(
        gs, wolf_id=wolf_id, plan=wolf_plan,
    )
    if kill_assessment:
        strategy_directive["kill_value_assessment"] = kill_assessment
    if wolf_plan and wolf_plan.get("night_kill_primary"):
        strategy_directive["wolf_plan_target"] = (
            f"狼队讨论确定的主目标: {wolf_plan['night_kill_primary']}"
            + (f"，备选: {wolf_plan['night_kill_backup']}" if wolf_plan.get("night_kill_backup") else "")
        )

    context = build_agent_context(
        engine, gs, wolf_id, TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=legal_targets,
        wolf_team_plan=wolf_plan,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(decision_identity, exposure_collector, decision_trace_sink),
    )
    context = _merge_strategy_directive(context, strategy_directive)

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
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None}

    action, _retry_info = action_result
    action_trace = _action_trace_payload(action)

    if action.action_type == ActionType.WOLF_NO_KILL:
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
    if action.action_type == ActionType.WOLF_KILL and action.target_id:
        target_player = gs.players.get(action.target_id)
        if target_player and target_player.alive and target_player.role != "werewolf":
            return {"wolf_action": "kill", "wolf_kill_target_id": action.target_id, "action_trace": action_trace}
        return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
    return {"wolf_action": "no_kill", "wolf_kill_target_id": None, "action_trace": action_trace}
