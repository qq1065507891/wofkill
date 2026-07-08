# -*- coding: utf-8 -*-
"""
构造 PlayerAgent 行动生成失败后的重试提示。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.player_retry_hints import build_empty_response_retry
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
)


def build_empty_response_retry(
    *,
    context: AgentContext,
    attempt: int,
    max_retries: int,
    failure_category: str,
    output_mode: OutputMode,
) -> RetryInfo:
    """为模型空响应构造下一轮重试提示。"""
    category_hint = f" (cause: {failure_category})" if failure_category else ""
    timeout_hint = ""
    if failure_category == "timeout":
        can_emit_no_action = (
            ActionType.NO_ACTION in context.legal_actions
            and output_mode == OutputMode.FULL_ACTION
        )
        if can_emit_no_action:
            timeout_hint = (
                " 如果超时，请直接返回 no_action 而非空响应"
                "（action_type='no_action', target_id=null,"
                "reason='timeout - safe no-op'）。"
            )
        elif context.legal_targets:
            first_target = context.legal_targets[0]
            timeout_hint = (
                f" 如果超时，请直接选择一个合法目标 "
                f"（例如 {first_target}）并提交结构化JSON。"
            )
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category=failure_category,
        correction_hint=(
            f"Please provide a valid JSON action{category_hint}. "
            f"If the model timed out, consider shorter reasoning."
            f"{timeout_hint}"
        ),
    )


def build_missing_tool_call_retry(
    *,
    attempt: int,
    max_retries: int,
    structured_failure_reason: str,
) -> RetryInfo:
    """为缺失 submit_player_action 工具调用构造重试提示。"""
    parse_error = "missing required tool call: submit_player_action"
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code=structured_failure_reason,
        error_message=parse_error,
        correction_hint=(
            "必须通过 submit_player_action 工具调用提交结构化参数；"
            "不要把JSON写在普通文本内容里。"
        ),
    )


__all__ = [
    "build_empty_response_retry",
    "build_missing_tool_call_retry",
]
