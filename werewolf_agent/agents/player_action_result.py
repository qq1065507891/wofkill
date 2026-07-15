# -*- coding: utf-8 -*-
"""
封装 PlayerAgent 行动成功与 fallback 的最终结果收尾。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.agents.player_action_result import finalize_successful_player_action
    >>> final_action = finalize_successful_player_action(...)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import (
    AgentContext,
    FallbackAction,
    PlayerAction,
    RetryInfo,
)
from werewolf_agent.agents.trace_builder import build_action_trace as _build_action_trace
from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord
from werewolf_agent.runtime.decision_outcomes import summarize_attempt_counts


def finalize_successful_player_action(
    *,
    agent: Any,
    context: AgentContext,
    action: PlayerAction,
    retry: RetryInfo,
    raw_text: str,
    parsed_action: PlayerAction | dict[str, Any] | None,
    tool_call_required: bool,
    tool_call_received: bool,
    parse_success: bool,
    retry_count: int,
    structured_output_mode: str,
    execution_attempts: tuple[AttemptExecutionRecord, ...] = (),
    semantic_repair_audit: dict[str, Any] | None = None,
) -> PlayerAction:
    """为成功行动附加审计 trace，并记录成功 metrics。"""
    trace = _build_action_trace(
        context,
        raw_text=raw_text,
        parsed_action=parsed_action,
        final_action_type=action.action_type,
        retry=retry,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        parse_success=parse_success,
        retry_count=retry_count,
        structured_output_mode=structured_output_mode,
        execution_attempts=execution_attempts,
        semantic_repair_audit=semantic_repair_audit,
    )
    effective_retry_count = (
        summarize_attempt_counts(execution_attempts).retry_count
        if execution_attempts else retry_count
    )
    agent.metrics_collector.record(
        player_id=context.agent_id,
        task_type=context.task_type.value,
        error_code=None,
        fallback_used=False,
        retry_count=effective_retry_count,
    )
    return action.model_copy(update={"trace": trace})


def finalize_fallback_player_action(
    *,
    agent: Any,
    context: AgentContext,
    fallback: FallbackAction,
    retry: RetryInfo | None,
    raw_text: str,
    parsed_action: PlayerAction | dict[str, Any] | None,
    tool_call_required: bool,
    tool_call_received: bool,
    parse_success: bool,
    parse_error: str | None,
    retry_count: int,
    structured_failure_reason: str | None,
    structured_output_mode: str,
    structured_failure_stage: str | None,
    fallback_target_used: bool = False,
    metrics_error_code: str | None = None,
    execution_attempts: tuple[AttemptExecutionRecord, ...] = (),
    semantic_repair_audit: dict[str, Any] | None = None,
) -> FallbackAction:
    """为 fallback 行动附加审计 trace，并记录 fallback metrics。"""
    trace = _build_action_trace(
        context,
        raw_text=raw_text,
        parsed_action=parsed_action,
        final_action_type=fallback.action_type,
        retry=retry,
        fallback_reason=fallback.reason,
        fallback_target_used=fallback_target_used,
        fallback_target_id=fallback.target_id if fallback_target_used else None,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        parse_success=parse_success,
        parse_error=parse_error,
        retry_count=retry_count,
        structured_failure_reason=structured_failure_reason,
        structured_output_mode=structured_output_mode,
        structured_failure_stage=structured_failure_stage,
        execution_attempts=execution_attempts,
        semantic_repair_audit=semantic_repair_audit,
    )
    effective_retry_count = (
        summarize_attempt_counts(execution_attempts).retry_count
        if execution_attempts else retry_count
    )
    agent.metrics_collector.record(
        player_id=context.agent_id,
        task_type=context.task_type.value,
        error_code=metrics_error_code or (retry.error_code if retry else "exhausted_retries"),
        fallback_used=True,
        retry_count=effective_retry_count,
    )
    return fallback.model_copy(update={"trace": trace})
