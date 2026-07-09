# -*- coding: utf-8 -*-
"""
警长相关 agent action 适配器。

作者: Mike
创建日期: 2026-07-07
修改日期: 2026-07-10

使用示例:
    >>> from werewolf_agent.runtime.agent_sheriff_actions import agent_sheriff_vote
    >>> agent_sheriff_vote(...)
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
)
from werewolf_agent.runtime.agent_registry import AgentRegistry
from werewolf_agent.runtime.context import (
    _action_trace_payload,
    _merge_strategy_directive,
    build_agent_context,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.agent_sheriff_speech_actions import (
    agent_sheriff_election_speech,
)
from werewolf_agent.runtime.sheriff_action_directives import (
    build_sheriff_endorse_directive,
    build_sheriff_endorse_result,
    build_sheriff_speech_order,
    build_sheriff_speech_order_directive,
    living_non_sheriff_ids,
)
from werewolf_agent.runtime.strategy import (
    get_wolf_role_assignment as _get_wolf_role_assignment,
)
from werewolf_agent.runtime.strategy.seer import public_seer_claimants
from werewolf_agent.runtime.vote_quality import validate_sheriff_vote_choice

logger = logging.getLogger(__name__)


def agent_sheriff_pick_speech_order(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
    include_action_trace: bool = False,
) -> list[str] | dict[str, Any] | None:
    """Ask the sheriff agent to choose the first speaker. Returns full speech order or None."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_players = living_non_sheriff_ids(gs, sheriff_id)
    if not alive_players:
        return None

    # 警长选首发言人属于 VOTE 风格动作，task_type 和 legal_actions 必须从一开始一致。
    context = build_agent_context(
        engine,
        gs,
        sheriff_id,
        TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=alive_players,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    strategy_directive = build_sheriff_speech_order_directive(alive_players)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    action_trace = _action_trace_payload(action)
    first_speaker = action.target_id if action.action_type == ActionType.VOTE else None

    speech_order = build_sheriff_speech_order(
        first_speaker=first_speaker,
        alive_players=alive_players,
        sheriff_id=sheriff_id,
    )
    if speech_order is not None:
        if include_action_trace:
            return {"speech_order": speech_order, "action_trace": action_trace}
        return speech_order
    if include_action_trace:
        return {"speech_order": None, "action_trace": action_trace}
    return None


def agent_sheriff_endorse(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    sheriff_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Sheriff privately decides endorsement target via VOTE action.

    Returns dict with endorse_target / private_reason / action_trace
    (or None for scripted fallback when no agent is registered).
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(sheriff_id)
    if agent is None:
        return None

    alive_others = living_non_sheriff_ids(gs, sheriff_id)

    strategy_directive = build_sheriff_endorse_directive(alive_others)

    context = build_agent_context(
        engine,
        gs,
        sheriff_id,
        TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=alive_others,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    target = action.target_id if action.action_type == ActionType.VOTE else None

    action_trace = (
        _action_trace_payload(action) if target and target in alive_others else None
    )
    return build_sheriff_endorse_result(
        target=target,
        alive_others=alive_others,
        private_reason=getattr(action, "reason", "") or "",
        action_trace=action_trace,
    )


def agent_sheriff_vote(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    voter_id: str,
    candidates: list[str],
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get sheriff vote from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(voter_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine,
        gs,
        voter_id,
        TaskType.VOTE,
        legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
        legal_targets=candidates,
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    strategy_directive = context.strategy_directive or {}
    voter_role = gs.players[voter_id].role if voter_id in gs.players else ""
    seer_claimants = sorted(public_seer_claimants(gs) & set(candidates))
    non_seer_candidates = [c for c in candidates if c not in seer_claimants]
    if seer_claimants and non_seer_candidates:
        strategy_directive["sheriff_vote_non_seer_candidate_rule"] = (
            "允许投给非预言家候选，但必须说明所有跳预言家的候选都不可信，"
            "再基于公开发言、验人矛盾、警徽流或票型解释该候选为什么更像好人。"
            f"当前跳预言家候选: {seer_claimants}; 非预言家候选: {non_seer_candidates}。"
        )
    # 警长投票是竞选投票，不是白天放逐投票，不注入放逐投票的 vote_basis/seer_stance。
    if voter_role == "werewolf":
        wolf_teammates = [
            pid
            for pid, p in gs.players.items()
            if p.alive and p.role == "werewolf" and pid != voter_id
        ]
        teammate_candidates = [c for c in candidates if c in wolf_teammates]
        if teammate_candidates:
            strategy_directive["wolf_sheriff_vote"] = (
                f"你是狼人。你的队友 {', '.join(teammate_candidates)} 也在候选人中。"
                "投票时不要明显全部投给队友——这样会暴露你们的关系。"
                "如果场上有多个候选人，你应该分散投票，表现得像一个独立判断的好人。"
            )
    if strategy_directive:
        context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"vote_target": None, "self_destruct": True}
    target = action.target_id if action.action_type == ActionType.SHERIFF_VOTE else None
    check = validate_sheriff_vote_choice(
        target_id=target,
        reason=getattr(action, "reason", "") or "",
        seer_claimants=seer_claimants,
        candidates=candidates,
    )
    vote_validation = {
        **check,
        "original_target": target,
        "final_target": target,
        "repaired": False,
    }
    if not check["valid"] and check["error_code"] == "weak_non_seer_sheriff_vote":
        target = next((pid for pid in seer_claimants if pid in candidates), None)
        vote_validation.update(
            {
                "final_target": target,
                "repaired": target is not None,
                "repair_reason": "weak_non_seer_sheriff_vote_retargeted_to_seer_claimant",
            }
        )
    elif not check["valid"] and check["error_code"] == "invalid_sheriff_vote_target":
        target = None
        vote_validation.update(
            {
                "final_target": None,
                "repaired": True,
                "repair_reason": "invalid_sheriff_vote_target_cleared",
            }
        )
    return {
        "vote_target": target,
        "action_trace": _action_trace_payload(action),
        "sheriff_vote_validation": vote_validation,
        "self_destruct": False,
    }


def agent_sheriff_register(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask a player whether they want to register for sheriff election.

    Returns dict with registration result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return None

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    wolf_plan = state.get("wolf_team_plan")
    if player_role == "seer":
        role_hint = (
            "你是预言家。上警通常有利于公开真实验人和建立警徽流，"
            "但应结合已有验人、发言顺序和场上声明决定；"
            "若上警，只能准确报告真实信息，不得为增强可信度编造结果。"
            "不上警是极低概率高阶战术，只有高玩画像且能说明隐忍收益、"
            "风险和后续补跳计划时才可使用；普通情况下必须上警。"
        )
    elif player_role == "werewolf":
        wolf_assignment = _get_wolf_role_assignment(wolf_plan, player_id)
        if wolf_assignment == "fake_seer":
            role_hint = (
                "【强制指令】你是团队安排的悍跳预言家！你必须上警！"
                "你需要在警上冒充预言家，报出假验人结果和警徽流，"
                "与真预言家争夺警徽。这是你的核心任务，必须上警。"
            )
        else:
            role_hint = (
                "你是狼人。如果团队安排你悍跳预言家，你必须上警与真预言家对抗。"
                "如果不悍跳，也可以上警发言获取信息或带节奏。"
            )
    else:
        role_hint = (
            "你是好人（非预言家），可以考虑上警发言表达观点、压制狼人发言空间。"
            "但注意：如果你不是预言家，不要在警上冒充预言家抢警徽，"
            "这会干扰真预言家的信息传递。上警主要目的是发言和表达立场。"
        )

    strategy_directive = {
        "sheriff_registration": (
            f"{role_hint}\n"
            "上警意味着你将在竞选环节发言，争取警长职位或表达观点。"
            "不上警则留在警下投票选出警长。"
        ),
    }

    context = build_agent_context(
        engine,
        gs,
        player_id,
        TaskType.SHERIFF_REGISTRATION,
        legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
        wolf_team_plan=wolf_plan,
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    try:
        action, retry_info = agent.act(context)
        action_trace = _action_trace_payload(action)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {
                "registered": False,
                "self_destruct": True,
                "action_trace": action_trace,
            }
        if (
            player_role == "seer"
            and action.action_type != ActionType.SHERIFF_REGISTER
            and not _seer_skip_sheriff_tactic_allowed(agent, context)
        ):
            return {
                "registered": True,
                "self_destruct": False,
                "action_trace": action_trace,
            }
        return {
            "registered": action.action_type == ActionType.SHERIFF_REGISTER,
            "self_destruct": False,
            "action_trace": action_trace,
        }
    except Exception:
        logger.warning("Sheriff registration failed for %s", player_id, exc_info=True)
        return {"registered": False, "self_destruct": False}


def _seer_skip_sheriff_tactic_allowed(agent: Any, context: Any) -> bool:
    """判断真预言家不上警是否达到高玩战术门槛。"""
    snapshot = dict(getattr(context, "persona_snapshot", {}) or {})
    if not snapshot:
        try:
            from dataclasses import asdict
            from werewolf_agent.persona_runtime.router import GameContext

            router = getattr(agent, "persona_router", None)
            if router is not None:
                resolved = router.resolve(
                    getattr(agent, "agent_id", context.agent_id),
                    context.task_type.value,
                    GameContext(
                        phase=context.phase,
                        day_number=context.day_number,
                        night_number=context.night_number,
                        own_role=context.own_role or "",
                    ),
                )
                snapshot = asdict(resolved)
        except Exception:
            snapshot = {}
    params = snapshot.get("effective_params") or {}
    logic = float(params.get("logic_skill", params.get("logic", 0.0)) or 0.0)
    credibility = float(params.get("credibility", 0.0) or 0.0)
    profile_id = str(snapshot.get("profile_id") or "").lower()
    expert_profile = any(marker in profile_id for marker in ("expert", "pro", "high", "高手", "高玩"))
    return expert_profile and logic >= 0.85 and credibility >= 0.75


def agent_sheriff_withdraw(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Ask a sheriff candidate whether they want to withdraw.

    Returns dict with withdrawal result and self_destruct flag.
    """
    gs: GameState = state["game_state"]
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    wolf_plan = state.get("wolf_team_plan")
    wolf_assignment = (
        _get_wolf_role_assignment(wolf_plan, candidate_id)
        if player_role == "werewolf"
        else ""
    )
    if player_role == "seer" or wolf_assignment == "fake_seer":
        return {"withdrew": False, "self_destruct": False}

    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine,
        gs,
        candidate_id,
        TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    try:
        action, retry_info = agent.act(context)
        action_trace = _action_trace_payload(action)
        if action.action_type == ActionType.SELF_DESTRUCT:
            return {
                "withdrew": False,
                "self_destruct": True,
                "action_trace": action_trace,
            }
        return {
            "withdrew": action.action_type == ActionType.SHERIFF_WITHDRAW,
            "self_destruct": False,
            "action_trace": action_trace,
        }
    except Exception:
        logger.warning("Sheriff withdrawal failed for %s", candidate_id, exc_info=True)
        return {"withdrew": False, "self_destruct": False}


__all__ = [
    "agent_sheriff_pick_speech_order",
    "agent_sheriff_endorse",
    "agent_sheriff_vote",
    "agent_sheriff_register",
    "agent_sheriff_withdraw",
    "agent_sheriff_election_speech",
]
