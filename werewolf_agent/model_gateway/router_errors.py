# -*- coding: utf-8 -*-
"""
模型路由器用量记录与失败结果构造。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> isinstance(_empty_result(config_provider="mock", config_model="mock", active_mode="native_tool"), object)
    True
"""

from __future__ import annotations

import threading

from werewolf_agent.model_gateway.retry_policy import (
    _http_status_from_exception,
    _raw_error_from_exception,
)
from werewolf_agent.model_gateway.usage_records import GenerateResult, UsageRecord


def _record_success_usage(
    *,
    usage_log: list[UsageRecord],
    usage_lock: threading.Lock,
    agent_id: str,
    task_type: str,
    result: GenerateResult,
    structured_output_mode: str,
    fallback_reason: str | None = None,
) -> None:
    """记录成功调用的用量，保持旧的截断策略。"""
    if not result.usage:
        return
    usage = UsageRecord(
        agent_id=agent_id,
        task_type=task_type,
        provider=result.provider,
        model=result.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        latency_ms=result.usage.latency_ms,
        fallback_reason=fallback_reason,
        success=True,
        structured_output_mode=structured_output_mode,
    )
    _append_usage(usage_log, usage_lock, usage)


def _record_failure_usage(
    *,
    usage_log: list[UsageRecord],
    usage_lock: threading.Lock,
    agent_id: str,
    task_type: str,
    provider: str,
    model: str,
    fallback_reason: str,
    structured_output_mode: str,
) -> None:
    """记录最终失败的路由用量。"""
    usage = UsageRecord(
        agent_id=agent_id,
        task_type=task_type,
        provider=provider,
        model=model,
        fallback_reason=fallback_reason,
        success=False,
        structured_output_mode=structured_output_mode,
    )
    _append_usage(usage_log, usage_lock, usage)


def _empty_result(
    *,
    config_provider: str,
    config_model: str,
    active_mode: str,
    primary_error: Exception | None = None,
    fallback_error: Exception | None = None,
    last_empty_result: GenerateResult | None = None,
) -> GenerateResult:
    """构造兼容旧字段的空响应结果。"""
    return GenerateResult(
        text="",
        provider=last_empty_result.provider if last_empty_result else config_provider,
        model=last_empty_result.model if last_empty_result else config_model,
        usage=last_empty_result.usage if last_empty_result else None,
        http_status=_http_status_from_exception(primary_error)
        or _http_status_from_exception(fallback_error),
        raw_error=_raw_error_from_exception(primary_error)
        or _raw_error_from_exception(fallback_error),
        structured_output_mode=(
            last_empty_result.structured_output_mode
            if last_empty_result
            else active_mode
        ),
    )


def _append_usage(
    usage_log: list[UsageRecord],
    usage_lock: threading.Lock,
    usage: UsageRecord,
) -> None:
    """追加记录并沿用旧的最大日志长度策略。"""
    with usage_lock:
        usage_log.append(usage)
        if len(usage_log) > 10000:
            usage_log[:] = usage_log[-5000:]


__all__ = ["_empty_result", "_record_failure_usage", "_record_success_usage"]
