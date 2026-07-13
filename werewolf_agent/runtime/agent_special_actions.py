# -*- coding: utf-8 -*-
"""
运行时特殊角色和夜间行动的 agent 适配器。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.runtime.agent_special_actions import agent_night_witch
    >>> agent_night_witch(...)
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
from werewolf_agent.runtime.badge_decision_directives import (
    build_badge_decision_directive,
    build_badge_decision_result,
)
from werewolf_agent.runtime.context import (
    build_agent_context,
    _action_trace_payload,
    _merge_strategy_directive,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.hunter_shot_directives import (
    build_hunter_shot_directive,
    build_hunter_shot_result,
)
from werewolf_agent.runtime.hybrid_master_directives import (
    build_hybrid_master_candidates,
    build_hybrid_master_choice_directive,
    choose_hybrid_master_target,
)
from werewolf_agent.runtime.seer_night_directives import (
    build_badge_flow_next_targets,
    build_seer_legal_targets,
    build_seer_night_strategy_directive,
)
from werewolf_agent.runtime.sheriff_action_directives import living_non_sheriff_ids
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    evaluate_hunter_shot_target as _evaluate_hunter_shot_target,
    evaluate_hybrid_master_candidates as _evaluate_hybrid_master_candidates,
    evaluate_seer_check_value as _evaluate_seer_check_value,
)
from werewolf_agent.runtime.strategy.seer import (
    public_seer_claimants as _public_seer_claimants,
)
from werewolf_agent.runtime.witch_night_directives import (
    build_witch_action_evidence,
    build_witch_first_night_killed_directive,
    build_witch_legal_actions,
    build_witch_night_action_directive,
    build_witch_poison_candidates_directive,
    build_witch_poison_strategy,
    build_witch_pressure_directives,
    build_witch_strategy_hint,
)

logger = logging.getLogger(__name__)


def _state_context_dependencies(state: dict[str, Any]) -> dict[str, Any]:
    """集中透传 state 中的可选上下文依赖，避免各 adapter 重复拼装。"""
    return {
        "rag_service": state.get("rag_service"),
        "restored_memory": state.get("restored_memory"),
        "cognition_state_manager": state.get("cognition_state_manager"),
    }


def _context_audit_dependencies(
    decision_identity: DecisionIdentity | None,
    exposure_collector: ModuleExposureAuditCollector | None,
    decision_trace_sink: Any | None,
) -> dict[str, Any]:
    """集中透传审计参数，保持每个特殊行动 adapter 的调用形状一致。"""
    return _audit_context_kwargs(
        decision_identity,
        exposure_collector,
        decision_trace_sink,
    )


def _damage_decision_evidence(
    base: dict[str, Any],
    *,
    target_id: str,
) -> dict[str, Any]:
    """把提示期候选证据投影为带最终目标的伤害决策审计。"""
    evidence = dict(base)
    comparison = dict(evidence.get("alternative_comparison") or {})
    legal = [
        target for target in comparison.get("legal_alternatives", [])
        if target and target != target_id
    ]
    comparison.update({
        "legal_alternatives": legal,
        "alternative_target": legal[0] if legal else None,
        "no_legal_alternative": not legal,
    })
    evidence["target_id"] = target_id
    evidence["alternative_comparison"] = comparison
    return evidence


def agent_night_witch(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
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

    legal_actions, legal_targets = build_witch_legal_actions(
        gs,
        engine,
        witch_id=witch_id,
        wolf_kill_target_id=wolf_kill_target_id,
    )

    context = build_agent_context(
        engine,
        gs,
        witch_id,
        TaskType.NIGHT_ACTION,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_kill_target_id=wolf_kill_target_id,
        **_state_context_dependencies(state),
        **_context_audit_dependencies(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    witch_directive: dict[str, Any] = {
        "witch_night_action": build_witch_night_action_directive(
            wolf_kill_target_id=wolf_kill_target_id,
            witch_id=witch_id,
            antidote_used=gs.antidote_used,
            poison_used=gs.poison_used,
            can_use_antidote=ActionType.USE_ANTIDOTE in legal_actions,
            can_use_poison=ActionType.USE_POISON in legal_actions,
        ),
    }

    save_value = _estimate_witch_save_value(gs, wolf_kill_target_id)
    witch_directive["save_value_assessment"] = save_value
    witch_directive["witch_strategy_hint"] = build_witch_strategy_hint(
        save_value,
        poison_available=not gs.poison_used,
    )
    if not gs.poison_used:
        alive = sum(1 for p in gs.players.values() if p.alive)
        witch_directive["witch_poison_strategy"] = build_witch_poison_strategy(alive)

    cands: list[dict[str, Any]] = []
    if not gs.poison_used:
        try:
            from werewolf_agent.runtime.strategy.poison import (
                collect_witch_poison_candidates,
            )

            cands = collect_witch_poison_candidates(gs, witch_id)
        except Exception:
            cands = []
        alive = sum(1 for p in gs.players.values() if p.alive)
        witch_directive["witch_poison_candidates"] = (
            build_witch_poison_candidates_directive(
                cands,
                alive_count=alive,
            )
        )
    witch_evidence = build_witch_action_evidence(
        legal_targets=legal_targets,
        antidote_targets=(
            [wolf_kill_target_id]
            if ActionType.USE_ANTIDOTE in legal_actions and wolf_kill_target_id
            else []
        ),
        poison_targets=(
            [
                pid for pid, player in gs.players.items()
                if player.alive and pid != witch_id
            ]
            if ActionType.USE_POISON in legal_actions else []
        ),
        poison_candidates=cands,
        wolf_kill_target_id=wolf_kill_target_id,
    )
    witch_directive["witch_action_evidence"] = witch_evidence

    first_night_killed = build_witch_first_night_killed_directive(
        wolf_kill_target_id=wolf_kill_target_id,
        witch_id=witch_id,
        poison_used=gs.poison_used,
    )
    if first_night_killed is not None:
        witch_directive["first_night_killed"] = first_night_killed

    poison_pressure = context.visible_world_state.get("poison_pressure_targets", [])
    witch_directive.update(build_witch_pressure_directives(poison_pressure))

    context = _merge_strategy_directive(context, witch_directive)

    action, retry_info = agent.act(context)

    use_antidote = action.action_type == ActionType.USE_ANTIDOTE
    poison_target_id = (
        action.target_id if action.action_type == ActionType.USE_POISON else None
    )

    action_trace = _action_trace_payload(action) or {}
    if poison_target_id:
        action_trace.setdefault("final_action_type", ActionType.USE_POISON.value)
        action_trace["power_role_evidence"] = _damage_decision_evidence(
            witch_evidence,
            target_id=poison_target_id,
        )
    return {
        "use_antidote": use_antidote,
        "poison_target_id": poison_target_id,
        "witch_action_trace": action_trace,
    }


def agent_night_seer(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
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

    counterclaiming_seers = _public_seer_claimants(gs) - {seer_id}
    legal_targets = build_seer_legal_targets(
        gs,
        seer_id=seer_id,
        counterclaiming_seers=counterclaiming_seers,
    )
    badge_flow_next = build_badge_flow_next_targets(
        gs,
        seer_id=seer_id,
        legal_targets=legal_targets,
    )
    check_value = _evaluate_seer_check_value(gs, seer_id, legal_targets)

    strategy_directive = build_seer_night_strategy_directive(
        night_number=gs.night_number,
        check_value=check_value,
        badge_flow_next=badge_flow_next,
        counterclaiming_seers=counterclaiming_seers,
    )

    context = build_agent_context(
        engine,
        gs,
        seer_id,
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHECK_ALIGNMENT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        **_state_context_dependencies(state),
        **_context_audit_dependencies(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    seer_target_id = (
        action.target_id if action.action_type == ActionType.CHECK_ALIGNMENT else None
    )
    return {
        "seer_target_id": seer_target_id,
        "seer_action_trace": _action_trace_payload(action),
    }


def agent_hybrid_choose_master(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hybrid_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask hybrid agent to choose their master. Returns None if agent unavailable."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hybrid_id)
    if agent is None:
        return None

    candidates = build_hybrid_master_candidates(gs, hybrid_id)
    master_assessment = _evaluate_hybrid_master_candidates(gs, hybrid_id, candidates)
    strategy_directive = build_hybrid_master_choice_directive(master_assessment)

    context = build_agent_context(
        engine,
        gs,
        hybrid_id,
        TaskType.NIGHT_ACTION,
        legal_actions=[ActionType.CHOOSE_MASTER],
        legal_targets=candidates,
        **_state_context_dependencies(state),
        **_context_audit_dependencies(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    master_target_id = choose_hybrid_master_target(
        action_type=action.action_type,
        target_id=action.target_id,
        candidates=candidates,
    )

    return {
        "master_target_id": master_target_id,
        "action_trace": _action_trace_payload(action),
    }


def agent_badge_decision(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Dying sheriff decides to transfer badge or tear it."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = living_non_sheriff_ids(gs, sheriff_id)
    context = build_agent_context(
        engine,
        gs,
        sheriff_id,
        TaskType.LAST_WORDS,
        legal_actions=[ActionType.BADGE_TRANSFER, ActionType.BADGE_TEAR],
        legal_targets=alive_others,
        **_state_context_dependencies(state),
        **_context_audit_dependencies(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    player_role = gs.players[sheriff_id].role if sheriff_id in gs.players else ""
    strategy_directive = build_badge_decision_directive(player_role, alive_others)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    return build_badge_decision_result(
        action_type=action.action_type,
        target_id=action.target_id,
        action_trace=_action_trace_payload(action),
    )


def agent_hunter_shot(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    hunter_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | str | None:
    """Get hunter shot target from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(hunter_id)
    if agent is None:
        return None

    death_reason = state.get("hunter_death_reason", "unknown")
    legal_targets = [
        pid for pid, p in gs.players.items() if p.alive and pid != hunter_id
    ]

    shot_assessment = None
    if legal_targets:
        try:
            shot_assessment = _evaluate_hunter_shot_target(
                gs,
                hunter_id,
                legal_targets,
                death_reason,
            )
        except Exception:
            logger.warning("Hunter shot target evaluation failed", exc_info=True)

    strategy_directive = build_hunter_shot_directive(
        death_reason=death_reason,
        shot_assessment=shot_assessment,
    )

    context = build_agent_context(
        engine,
        gs,
        hunter_id,
        TaskType.HUNTER_SHOT,
        legal_actions=[ActionType.HUNTER_SHOT, ActionType.NO_ACTION],
        legal_targets=legal_targets,
        **_state_context_dependencies(state),
        **_context_audit_dependencies(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    action_trace = _action_trace_payload(action) or {}
    if action.action_type == ActionType.HUNTER_SHOT and action.target_id:
        action_trace.setdefault("final_action_type", ActionType.HUNTER_SHOT.value)
        action_trace["power_role_evidence"] = _damage_decision_evidence(
            {
                "alternative_comparison": strategy_directive["alternative_comparison"],
                "friendly_fire_risk": strategy_directive["friendly_fire_risk"],
                "retain_option": strategy_directive["retain_option"],
            },
            target_id=action.target_id,
        )
    return build_hunter_shot_result(
        action_type=action.action_type,
        target_id=action.target_id,
        action_trace=action_trace,
    )


__all__ = [
    "agent_night_witch",
    "agent_night_seer",
    "agent_hybrid_choose_master",
    "agent_badge_decision",
    "agent_hunter_shot",
]
