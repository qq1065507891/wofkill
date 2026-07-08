# -*- coding: utf-8 -*-
"""
运行时日间投票行动的 agent 适配器。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.agent_day_vote_actions import agent_day_vote
    >>> agent_day_vote(...)
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.agent_action_audit import (
    _audit_context_kwargs,
    _inject_vote_basis_hint,
)
from werewolf_agent.runtime.agent_registry import AgentRegistry
from werewolf_agent.runtime.context import (
    build_agent_context,
    _merge_strategy_directive,
)
from werewolf_agent.runtime.day_vote_directives import (
    build_day_vote_base_directive,
    build_fallback_seer_vote_strategy,
    build_hunter_vote_strategy,
    build_hybrid_vote_strategy,
    build_seer_vote_strategy,
    build_villager_vote_strategy,
    build_vote_anti_herd_directive,
    build_witch_vote_strategy,
)
from werewolf_agent.runtime.directives import (
    build_wolf_vote_directive as _build_wolf_vote_strategy,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.strategy.seer import (
    public_seer_claimants as _public_seer_claimants,
)

logger = logging.getLogger(__name__)


def agent_day_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
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

    legal_targets = [pid for pid in engine.legal_exile_targets(gs) if pid != voter_id]
    voter_role = gs.players[voter_id].role if voter_id in gs.players else ""
    if state.get("revote") and state.get("pk_candidates"):
        pk_candidates = set(state.get("pk_candidates") or [])
        legal_targets = [pid for pid in legal_targets if pid in pk_candidates]

    consecutive_no_exile = state.get("consecutive_no_exile_days", 0)
    strategy_directive = build_day_vote_base_directive(
        voter_role,
        allow_abstain=allow_abstain,
        consecutive_no_exile=consecutive_no_exile,
    )
    _inject_vote_basis_hint(strategy_directive, gs, voter_id)
    try:
        from werewolf_agent.runtime.vote_quality import (
            build_day_discussion_summary,
            build_vote_pressure_context,
        )

        strategy_directive["day_discussion_summary"] = build_day_discussion_summary(
            gs, gs.day_number
        )
        strategy_directive["vote_pressure_context"] = build_vote_pressure_context(
            gs, voter_id, pk_candidates=state.get("pk_candidates")
        )
        strategy_directive["anti_herd"] = build_vote_anti_herd_directive()
    except Exception:
        logger.debug("Vote quality context build failed, skipping", exc_info=True)

    # Role-specific vote strategy (voter_role computed above for legal_targets filtering)
    if voter_role == "werewolf":
        wolf_vote_parts = _build_wolf_vote_strategy(
            gs,
            voter_id,
            state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_vote_parts)
    elif voter_role == "hybrid" and gs.hybrid_master_id:
        strategy_directive["hybrid_vote_strategy"] = build_hybrid_vote_strategy(
            gs.hybrid_master_id
        )
    elif voter_role == "seer":
        try:
            strategy_directive["seer_vote_strategy"] = build_seer_vote_strategy(gs)
        except Exception:
            logger.debug("Failed to build seer vote strategy", exc_info=True)
            strategy_directive["seer_vote_strategy"] = (
                build_fallback_seer_vote_strategy()
            )
    elif voter_role == "witch":
        strategy_directive["witch_vote_strategy"] = build_witch_vote_strategy()
    elif voter_role == "hunter":
        strategy_directive["hunter_vote_strategy"] = build_hunter_vote_strategy()
    elif voter_role in ("villager", "idiot"):
        seer_claimants = _public_seer_claimants(gs)
        strategy_directive["villager_vote_strategy"] = build_villager_vote_strategy(
            seer_claimants
        )

    # Pre-compute evidence-based fallback target for structured failure
    non_self_legal = [t for t in legal_targets if t != voter_id]
    if non_self_legal:
        try:
            from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

            fb = choose_vote_fallback_target(
                gs,
                voter_id,
                non_self_legal,
                require_evidence=True,
            )
            if fb:
                strategy_directive["_vote_fallback_target"] = fb
        except Exception:
            logger.warning("Failed to compute vote fallback target", exc_info=True)

    context = build_agent_context(
        engine,
        gs,
        voter_id,
        TaskType.VOTE,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    target = action.target_id if action.action_type == ActionType.VOTE else None
    # Fallback: if agent returned wrong action type but has legal targets,
    # pick an evidence-aware target rather than abstaining silently.
    if target is None and legal_targets:
        target = choose_vote_fallback_target(
            gs,
            voter_id,
            legal_targets,
            require_evidence=True,
        )
    speech = getattr(action, "speech", "") or ""
    reason = getattr(action, "reason", "") or ""
    trace = getattr(action, "trace", None)
    return {
        "vote_target": target,
        "vote_speech": speech,
        "vote_reason": reason,
        "action_trace": trace.model_dump() if trace else None,
    }


__all__ = ["agent_day_vote"]