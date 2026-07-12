# -*- coding: utf-8 -*-
"""
模型执行记录与旧版字段兼容契约测试。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from werewolf_agent.model_gateway.usage_records import GenerateResult, UsageRecord
from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    EvidenceKind,
    RootCause,
    RouteKind,
)


def _reasoned_attempt() -> AttemptExecutionRecord:
    return AttemptExecutionRecord(
        request_id="opaque-1",
        ordinal=1,
        provider="primary",
        model="model-a",
        route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.NONE,
        attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level="high",
        normalized_reasoning_status="confirmed",
        reasoning_token_count=17,
        evidence_kind=EvidenceKind.PROVIDER_METADATA,
    )


def test_usage_record_legacy_fields_remain_readable() -> None:
    usage = UsageRecord(
        agent_id="p01",
        task_type="day_vote",
        provider="primary",
        model="model-a",
        fallback_reason="timeout",
        retry_count=2,
        failure_category="provider_error",
        attempts=(_reasoned_attempt(),),
    )

    assert usage.fallback_reason == "timeout"
    assert usage.retry_count == 2
    assert usage.failure_category == "provider_error"
    assert usage.reasoning_level == "high"
    assert usage.reasoning_status == "confirmed"
    assert usage.reasoning_tokens == 17
    with pytest.raises(FrozenInstanceError):
        usage.retry_count = 3  # type: ignore[misc]


def test_generate_result_legacy_reasoning_fields_remain_readable() -> None:
    result = GenerateResult(
        text="ok",
        provider="primary",
        model="model-a",
        attempts=(_reasoned_attempt(),),
    )

    assert result.reasoning_level == "high"
    assert result.reasoning_status == "confirmed"
    assert result.reasoning_tokens == 17
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.reasoning_status = "drifted"  # type: ignore[misc]


def test_provider_package_exports_remain_stable() -> None:
    from werewolf_agent.model_gateway import providers

    expected = {
        "AnthropicProvider",
        "GLMProvider",
        "MiniMaxProvider",
        "OpenAIProvider",
        "PROVIDER_DOTENV_KEYS",
        "ProviderConfigError",
        "_BaseHttpProvider",
        "_ENV_OVERRIDES",
        "_anthropic_tool_name",
        "_extract_anthropic_text",
        "_has_anthropic_tool_use",
        "create_provider_from_env",
        "get_env",
        "load_local_dotenv",
    }

    assert set(providers.__all__) == expected
