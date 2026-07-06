# -*- coding: utf-8 -*-
"""
模型路由器拆分模块的兼容导入测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/model_gateway/test_router_split_helpers.py
"""

from __future__ import annotations

from werewolf_agent.model_gateway import provider_call
from werewolf_agent.model_gateway import retry_policy
from werewolf_agent.model_gateway import router
from werewolf_agent.model_gateway import usage_records


def test_usage_models_are_reexported_from_router_facade() -> None:
    assert router.ModelConfig is usage_records.ModelConfig
    assert router.UsageRecord is usage_records.UsageRecord
    assert router.GenerateResult is usage_records.GenerateResult
    assert router.EmptyModelResponseError is usage_records.EmptyModelResponseError
    assert router.LLMProvider is usage_records.LLMProvider
    assert router.MockProvider is usage_records.MockProvider


def test_provider_call_helpers_are_reexported_from_router_facade() -> None:
    assert router._call_provider_generate is provider_call._call_provider_generate
    assert router._normalize_tool_metadata is provider_call._normalize_tool_metadata


def test_retry_helpers_are_reexported_from_router_facade() -> None:
    assert router._format_exception is retry_policy._format_exception
    assert router._http_status_from_exception is retry_policy._http_status_from_exception
    assert router._raw_error_from_exception is retry_policy._raw_error_from_exception
    assert router._failure_reason is retry_policy._failure_reason
    assert router._is_retryable_exception is retry_policy._is_retryable_exception


def test_retry_delay_facade_uses_router_random_for_compatibility(monkeypatch) -> None:
    monkeypatch.setattr(router.random, "uniform", lambda low, high: high)

    delay = router._retry_delay_for_exception(RuntimeError("test"), 2)

    assert delay == 5.0
