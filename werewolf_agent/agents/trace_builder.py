# -*- coding: utf-8 -*-
"""
功能描述：从 player.py 拆出，将每次 LLM 调用经隐私净化的审计轨迹封装为 ActionTrace 对象。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-24
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import ActionTrace, AgentContext, PlayerAction, RetryInfo
from werewolf_agent.runtime.world_model_audit import build_world_model_audit_from_context
from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord
from werewolf_agent.runtime.decision_outcomes import (
    summarize_attempt_counts,
    translate_decision_outcome,
)


def build_action_trace(
    context: AgentContext,
    *,
    raw_text: str,
    parsed_action: PlayerAction | dict[str, Any] | None,
    final_action_type: Any,
    retry: RetryInfo | None,
    final_action: PlayerAction | dict[str, Any] | None = None,
    fallback_reason: str | None = None,
    fallback_target_used: bool = False,
    fallback_target_id: str | None = None,
    tool_call_required: bool = False,
    tool_call_received: bool = False,
    parse_success: bool = False,
    parse_error: str | None = None,
    retry_count: int = 0,
    structured_failure_reason: str | None = None,
    structured_output_mode: str = "",
    structured_failure_stage: str | None = None,
    execution_attempts: tuple[AttemptExecutionRecord, ...] = (),
    semantic_repair_audit: dict[str, Any] | None = None,
    fallback_kind: str | None = None,
) -> ActionTrace:
    """从当前尝试状态构造经隐私净化的 ActionTrace。

    ``final_action_type`` 统一转为字符串；仅供运行时重试使用的
    ``correction_hint`` 不会进入持久化 retry payload。
    """
    final_type_value = (
        final_action_type.value
        if hasattr(final_action_type, "value")
        else final_action_type
    )
    parsed_payload = (
        parsed_action.model_dump(exclude={"trace"})
        if isinstance(parsed_action, PlayerAction)
        else parsed_action
    )
    final_payload = (
        final_action.model_dump(exclude={"trace"})
        if isinstance(final_action, PlayerAction)
        else final_action
    )
    translated = (
        translate_decision_outcome(
            execution_attempts,
            structured_failure_reason=structured_failure_reason,
        )
        if execution_attempts else None
    )
    attempt_counts = (
        summarize_attempt_counts(execution_attempts) if execution_attempts else None
    )
    terminal_failure_code = (
        translated.terminal_failure_code
        if translated and translated.generated_by.value == "terminal_fallback"
        else None
    )
    # correction_hint 可包含被拒发言，只能留在当次行动内存中供重试。
    retry_payload = retry.model_dump(exclude={"correction_hint"}) if retry else None
    trace_raw_text = raw_text
    trace_parsed_payload = parsed_payload
    trace_parse_error = parse_error
    trace_failure_reason = structured_failure_reason
    if terminal_failure_code is not None:
        # 终退审计只导出稳定码，不保留 provider/schema 的原始错误正文。
        retry_payload = (
            retry.model_dump(exclude={"error_message", "correction_hint"})
            if retry
            else None
        )
        if retry_payload is not None:
            retry_payload["error_code"] = terminal_failure_code
        if retry and retry.reason_codes:
            # 语义拒绝的终退仅暴露安全原因码，拒绝输出本身不能进入审计轨迹。
            trace_raw_text = ""
            trace_parsed_payload = None
        trace_parse_error = terminal_failure_code
        trace_failure_reason = terminal_failure_code
    return ActionTrace(
        raw_text=trace_raw_text,
        parsed_action=trace_parsed_payload,
        final_action=final_payload,
        final_action_type=final_type_value,
        legal_actions=[action.value for action in context.legal_actions],
        legal_targets=list(context.legal_targets),
        retry=retry_payload,
        fallback_reason=fallback_reason,
        fallback_target_used=fallback_target_used,
        fallback_target_id=fallback_target_id,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        tool_call_name="submit_player_action" if tool_call_required else "",
        parse_success=parse_success,
        parse_error=trace_parse_error,
        attempt_count=attempt_counts.attempt_count if attempt_counts else 0,
        retry_count=attempt_counts.retry_count if attempt_counts else retry_count,
        provider_fallback_count=(
            attempt_counts.provider_fallback_count if attempt_counts else 0
        ),
        runtime_timeout_count=(
            attempt_counts.runtime_timeout_count if attempt_counts else 0
        ),
        generated_by=translated.generated_by.value if translated else None,
        terminal_failure_code=terminal_failure_code,
        original_failure_code=terminal_failure_code,
        failure_stage=(structured_failure_stage if terminal_failure_code else None),
        fallback_kind=(fallback_kind if terminal_failure_code else None),
        structured_failure_reason=trace_failure_reason,
        structured_output_mode=structured_output_mode,
        structured_failure_stage=structured_failure_stage,
        world_model_audit=build_world_model_audit_from_context(
            context,
            parsed_action=trace_parsed_payload,
        ),
        execution_attempts=execution_attempts,
        decision_outcome=translated.outcome.value if translated else None,
        semantic_repair_audit=semantic_repair_audit,
    )
