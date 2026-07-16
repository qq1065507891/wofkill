# -*- coding: utf-8 -*-
"""
集中管理 Agent 决策审计和投票提示辅助函数。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.agent_action_audit import _audit_context_kwargs
    >>> _audit_context_kwargs(None, None)
    {}
"""

from __future__ import annotations

import hashlib
from typing import Any

from werewolf_agent.agents.player_failures import (
    terminal_failure_code_for_task_failure,
)
from werewolf_agent.agents.schemas import ActionTrace, AgentContext
from werewolf_agent.core.models import GameState
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    EvidenceKind,
    OpaqueRequestId,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)
from werewolf_agent.runtime.decision_outcomes import translate_decision_outcome
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector


def _audit_context_kwargs(
    decision_identity: DecisionIdentity | None,
    exposure_collector: ModuleExposureAuditCollector | None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any]:
    if decision_identity is None or exposure_collector is None:
        return {}
    return {
        "decision_identity": decision_identity,
        "exposure_collector": exposure_collector,
        "decision_trace_sink": decision_trace_sink,
    }


def build_runtime_terminal_fallback_trace(
    *,
    reason_code: str,
    failure_stage: str,
    fallback_kind: str,
    final_action_type: str,
    decision_key: str = "",
) -> dict[str, Any]:
    """为无模型调用的运行时终退构造完整且一致的 V2 trace。"""
    terminal_code = terminal_failure_code_for_task_failure(reason_code)
    entropy = hashlib.sha256(
        f"{reason_code}:{decision_key}".encode("utf-8")
    ).hexdigest()[:16]
    request_id = OpaqueRequestId.new("runtime", entropy)
    root_cause = _terminal_root_cause(terminal_code)
    attempts = tuple(
        AttemptExecutionRecord(
            opaque_request_id=request_id,
            ordinal=ordinal,
            provider="runtime",
            model="terminal-fallback",
            route_kind=route_kind,
            root_cause=root_cause,
            attempt_outcome=AttemptOutcome.FAILURE,
            requested_reasoning_level=ReasoningLevel.NONE,
            normalized_reasoning_status=ReasoningStatus.NOT_REQUESTED,
            reasoning_token_count=0,
            evidence_kind=EvidenceKind.NONE,
        )
        for ordinal, route_kind in enumerate(
            (RouteKind.PRIMARY, RouteKind.SAFE_FALLBACK), 1
        )
    )
    translated = translate_decision_outcome(
        attempts,
        structured_failure_reason=terminal_code,
    )
    return ActionTrace(
        final_action_type=final_action_type,
        retry={"error_code": terminal_code},
        fallback_reason=reason_code,
        parse_error=terminal_code,
        attempt_count=translated.attempt_count,
        retry_count=translated.retry_count,
        provider_fallback_count=translated.provider_fallback_count,
        generated_by=translated.generated_by.value,
        terminal_failure_code=terminal_code,
        original_failure_code=terminal_code,
        failure_stage=failure_stage,
        fallback_kind=fallback_kind,
        structured_failure_reason=terminal_code,
        structured_failure_stage=failure_stage,
        execution_attempts=attempts,
        decision_outcome=translated.outcome.value,
    ).model_dump()


def _terminal_root_cause(terminal_code: str) -> RootCause:
    """把稳定失败码压缩为执行记录支持的封闭根因。"""
    if terminal_code == "timeout":
        return RootCause.TIMEOUT
    if terminal_code in {"provider_error", "model_generation_failed"}:
        return RootCause.PROVIDER_ERROR
    if terminal_code in {
        "policy_rejection", "fallback_route_unavailable",
    }:
        return RootCause.POLICY_REJECTION
    return RootCause.INVALID_OUTPUT


# M2-2：投票/发言动作的单一 vote_basis 指引来源。该提示必须按回合注入，
# 避免夜间动作看到与任务无关的投票字段。
VOTE_BASIS_GUIDANCE = (
    "【投票时 vote_basis 选用 speech_logic / vote_pattern / "
    "seer_siding，不要用 seer_check。】"
)


def _inject_vote_basis_hint(
    strategy_directive: dict[str, Any],
    gs: GameState,
    player_id: str,
) -> None:
    """为非预言家注入 per-turn vote_basis 指引。"""

    role = gs.players[player_id].role if player_id in gs.players else ""
    if role != "seer":
        strategy_directive["vote_basis_hint"] = VOTE_BASIS_GUIDANCE


def _seer_credibility_audit_payload(
    context: AgentContext,
    day_number: int,
) -> dict[str, Any] | None:
    summary = context.seer_credibility or {}
    lines = summary.get("seer_lines")
    if not isinstance(lines, list) or not lines:
        return None
    safe_lines: list[dict[str, Any]] = []
    for item in lines[:3]:
        if not isinstance(item, dict):
            continue
        safe_lines.append({
            key: item[key]
            for key in (
                "claimant",
                "status",
                "score",
                "confidence",
                "checks",
                "evidence",
                "penalties",
            )
            if key in item
        })
    if not safe_lines:
        return None
    return {
        "day_number": day_number,
        "visibility": "moderator_only",
        "seer_lines": safe_lines,
    }


def _is_sheriff_silenced(gs: GameState, sheriff_id: str) -> bool:
    """判断当前警长是否处于禁言或冻结状态。"""

    for ev in gs.events:
        if ev.type == "sheriff_silenced" and ev.payload.get("sheriff_id") == sheriff_id:
            return True
    if gs.sheriff_badge_state in {"silenced", "frozen"}:
        return True
    return False
