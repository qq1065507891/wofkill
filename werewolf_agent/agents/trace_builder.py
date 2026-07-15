# -*- coding: utf-8 -*-
"""
功能描述：**：从 player.py 拆出，将每次 LLM 调用的完整审计轨迹封装为 ActionTrace 对象。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-15
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.agents.schemas import ActionTrace, AgentContext, PlayerAction, RetryInfo
from werewolf_agent.runtime.world_model_audit import build_world_model_audit_from_context
from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord
from werewolf_agent.runtime.decision_outcomes import (
    TranslatedDecisionOutcome,
    translate_decision_outcome,
)


def build_action_trace(
    context: AgentContext,
    *,
    raw_text: str,
    parsed_action: PlayerAction | dict[str, Any] | None,
    final_action_type: Any,
    retry: RetryInfo | None,
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
) -> ActionTrace:
    """Build an ActionTrace from the current attempt's state.

    The ``final_action_type`` is normalized to its ``.value`` string when
    an ``ActionType`` enum is supplied so downstream audit code can
    treat the field uniformly.
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
    translated = (
        translate_decision_outcome(execution_attempts)
        if execution_attempts else None
    )
    return ActionTrace(
        raw_text=raw_text,
        parsed_action=parsed_payload,
        final_action_type=final_type_value,
        legal_actions=[action.value for action in context.legal_actions],
        legal_targets=list(context.legal_targets),
        retry=retry.model_dump() if retry else None,
        fallback_reason=fallback_reason,
        fallback_target_used=fallback_target_used,
        fallback_target_id=fallback_target_id,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        tool_call_name="submit_player_action" if tool_call_required else "",
        parse_success=parse_success,
        parse_error=parse_error,
        attempt_count=translated.attempt_count if translated else 0,
        retry_count=translated.retry_count if translated else retry_count,
        provider_fallback_count=(
            translated.provider_fallback_count if translated else 0
        ),
        generated_by=translated.generated_by.value if translated else None,
        terminal_failure_code=_terminal_failure_code(
            translated,
            structured_failure_reason,
        ),
        structured_failure_reason=structured_failure_reason,
        structured_output_mode=structured_output_mode,
        structured_failure_stage=structured_failure_stage,
        world_model_audit=build_world_model_audit_from_context(
            context,
            parsed_action=parsed_payload,
        ),
        execution_attempts=execution_attempts,
        decision_outcome=translated.outcome.value if translated else None,
        semantic_repair_audit=semantic_repair_audit,
    )


def _terminal_failure_code(
    translated: TranslatedDecisionOutcome | None,
    structured_failure_reason: str | None,
) -> str | None:
    """仅允许稳定错误码进入 trace，拒绝原始错误正文与身份信息。"""
    if not translated or translated.generated_by.value != "terminal_fallback":
        return None
    if structured_failure_reason and re.fullmatch(
        r"[a-z][a-z0-9_]{0,63}", structured_failure_reason
    ):
        return structured_failure_reason
    return translated.terminal_failure_code
