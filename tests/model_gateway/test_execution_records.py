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
    OpaqueRequestId,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)


def _reasoned_attempt() -> AttemptExecutionRecord:
    return AttemptExecutionRecord(
        opaque_request_id=OpaqueRequestId.new("game", "abcd1234"),
        ordinal=1,
        provider="primary",
        model="model-a",
        route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.NONE,
        attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=17,
        evidence_kind=EvidenceKind.AUTHORITATIVE_PROVIDER_EXECUTION,
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

    assert usage.fallback_reason is None
    assert usage.retry_count == 0
    assert usage.failure_category is None
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


def test_raw_provider_request_id_and_invalid_reasoning_evidence_are_rejected() -> None:
    with pytest.raises(TypeError):
        OpaqueRequestId("run_openai_deadbeef")
    with pytest.raises(TypeError):
        OpaqueRequestId("provider-request-123")
    with pytest.raises(ValueError, match="confirmed reasoning"):
        AttemptExecutionRecord(
            opaque_request_id=OpaqueRequestId.new("game", "abcd1234"),
            ordinal=1,
            provider="primary",
            model="model-a",
            route_kind=RouteKind.PRIMARY,
            root_cause=RootCause.NONE,
            attempt_outcome=AttemptOutcome.SUCCESS,
            requested_reasoning_level=ReasoningLevel.HIGH,
            normalized_reasoning_status=ReasoningStatus.CONFIRMED,
            reasoning_token_count=0,
            evidence_kind=EvidenceKind.NORMALIZED_RESPONSE,
        )


def test_usage_projection_overrides_every_conflicting_legacy_route_field() -> None:
    usage = UsageRecord(
        agent_id="p01", task_type="vote", provider="wrong", model="wrong",
        primary_provider="wrong", primary_model="wrong", fallback_provider="wrong",
        fallback_model="wrong", success=False, fallback_reason="wrong", retry_count=99,
        failure_category="wrong", reasoning_level="none", reasoning_status="not_requested",
        attempts=(_reasoned_attempt(),),
    )
    assert (usage.primary_provider, usage.primary_model) == ("primary", "model-a")
    assert usage.fallback_provider is None and usage.fallback_model is None
    assert (usage.provider, usage.model, usage.success) == ("primary", "model-a", True)
    assert (usage.fallback_reason, usage.retry_count, usage.failure_category) == (None, 0, None)


def test_generate_result_rejects_mismatched_usage_evidence_chain() -> None:
    usage = UsageRecord(agent_id="p01", task_type="vote", provider="primary", model="model-a", attempts=(_reasoned_attempt(),))
    other = _reasoned_attempt().__class__(**{**_reasoned_attempt().__dict__, "opaque_request_id": OpaqueRequestId.new("game", "deadbeef")})
    with pytest.raises(ValueError, match="same evidence chain"):
        GenerateResult(text="ok", provider="primary", model="model-a", usage=usage, attempts=(other,))
