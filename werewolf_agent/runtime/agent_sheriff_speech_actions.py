# -*- coding: utf-8 -*-
"""
处理警长竞选发言类 agent action 适配。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.agent_sheriff_speech_actions import agent_sheriff_election_speech
"""

from __future__ import annotations

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
    _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _SPEECH_STYLE_HINTS,
    _TASK_STYLE_HINTS,
    _action_trace_payload,
    _get_persona_speech_style,
    _get_persona_task_style,
    _merge_strategy_directive,
    build_agent_context,
)
from werewolf_agent.runtime.directives import (
    build_wolf_day_directive as _build_wolf_day_speech_directive,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.sheriff_election_directives import (
    build_previous_sheriff_speech_instruction,
    build_seer_verification_rationale,
    build_sheriff_badge_flow_instruction,
    build_sheriff_election_speech_directive,
    build_sheriff_role_speech_hint,
    build_sheriff_seer_context,
    build_wolf_sheriff_election_directives,
    collect_previous_sheriff_speeches,
    sheriff_uses_seer_protocol,
)
from werewolf_agent.runtime.strategy import (
    get_wolf_role_assignment as _get_wolf_role_assignment,
    has_publicly_claimed_seer as _has_publicly_claimed_seer,
)


def agent_sheriff_election_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    candidate_id: str,
    all_candidates: list[str],
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """让警长候选人发表竞选发言。"""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(candidate_id)
    if agent is None:
        return None

    other_candidates = [c for c in all_candidates if c != candidate_id]

    # 警徽流只给真预言家或狼队明确安排的悍跳位。
    player_role = gs.players[candidate_id].role if candidate_id in gs.players else ""
    wolf_plan = state.get("wolf_team_plan")
    wolf_assignment = (
        _get_wolf_role_assignment(wolf_plan, candidate_id)
        if player_role == "werewolf"
        else ""
    )
    uses_seer_protocol = sheriff_uses_seer_protocol(player_role, wolf_assignment)
    badge_flow_instruction = build_sheriff_badge_flow_instruction(uses_seer_protocol)

    public_seer_claimers = {
        candidate
        for candidate in all_candidates
        if _has_publicly_claimed_seer(gs, candidate)
    }
    if uses_seer_protocol:
        public_seer_claimers.add(candidate_id)
    seer_context = build_sheriff_seer_context(
        public_seer_claimers,
        uses_seer_protocol=uses_seer_protocol,
    )
    prev_speeches = collect_previous_sheriff_speeches(gs, candidate_id)
    prev_speech_instruction = build_previous_sheriff_speech_instruction(prev_speeches)

    speech_style = _get_persona_speech_style(agent)
    task_style = _get_persona_task_style(agent, "sheriff_speech")

    merged_hints = {**_SPEECH_STYLE_HINTS, **_SHERIFF_SPEECH_STYLE_OVERRIDES}
    style_hint = merged_hints.get(speech_style, "从你自己的独特角度分析场上局势。")
    task_hint = _TASK_STYLE_HINTS.get(task_style, "")

    strategy_directive = build_sheriff_election_speech_directive(
        style_hint=style_hint,
        task_hint=task_hint,
        badge_flow_instruction=badge_flow_instruction,
        seer_context=seer_context,
        prev_speech_instruction=prev_speech_instruction,
        other_candidates=other_candidates,
    )
    _inject_vote_basis_hint(strategy_directive, gs, candidate_id)

    role_speech_hint = build_sheriff_role_speech_hint(player_role)
    if role_speech_hint:
        strategy_directive["role_speech_hint"] = role_speech_hint

    seer_verification_rationale = build_seer_verification_rationale(player_role)
    if seer_verification_rationale:
        strategy_directive["seer_verification_rationale"] = seer_verification_rationale

    if player_role == "werewolf":
        wolf_day_directive = _build_wolf_day_speech_directive(
            gs, candidate_id, wolf_plan
        )
        strategy_directive.update(wolf_day_directive)
        fake_seer_publicly_claimed = False
        if wolf_assignment != "fake_seer" and wolf_plan and wolf_plan.get("fake_seer"):
            fake_seer_publicly_claimed = _has_publicly_claimed_seer(
                gs,
                wolf_plan["fake_seer"],
            )
        strategy_directive.update(
            build_wolf_sheriff_election_directives(
                wolf_assignment=wolf_assignment,
                wolf_plan=wolf_plan,
                candidate_id=candidate_id,
                fake_seer_publicly_claimed=fake_seer_publicly_claimed,
            )
        )

    context = build_agent_context(
        engine,
        gs,
        candidate_id,
        TaskType.SHERIFF_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    context = _merge_strategy_directive(context, strategy_directive)

    action, _retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    if not speech_text.strip() or len(speech_text.strip()) < 10:
        if uses_seer_protocol:
            speech_text = (
                f"我上警是因为我需要通过警徽流传递关键信息。"
                f"我的警徽流暂定先看{other_candidates[0] if other_candidates else '待定'}。"
                f"希望大家支持我当选警长。"
            )
        else:
            speech_text = (
                "我上警是想先给出自己的观察视角。"
                f"我会重点听{other_candidates[0] if other_candidates else '后置位'}的发言，"
                "看站边和逻辑是否前后一致。"
            )

    return {
        "speech_text": speech_text,
        "action_trace": _action_trace_payload(action),
        "self_destruct": False,
    }


__all__ = ["agent_sheriff_election_speech"]
