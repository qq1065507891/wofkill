# -*- coding: utf-8 -*-
"""
模型执行记录与旧版字段兼容契约测试。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from werewolf_agent.model_gateway.usage_records import (
    FailureDisposition,
    GenerateResult,
    UsageRecord,
)
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


def test_generate_result_defaults_to_no_failure_disposition() -> None:
    """旧 provider 构造结果时不必传入新的终态字段。"""
    result = GenerateResult(text="ok", provider="primary", model="model-a")

    assert result.failure_disposition is FailureDisposition.NONE
    assert FailureDisposition.TRANSPORT_EXHAUSTED.value == "transport_exhausted"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.reasoning_status = "drifted"  # type: ignore[misc]


def test_execution_record_preserves_skipped_provider_marker_across_mapping() -> None:
    """跳过 provider 的事实必须随跨轮次审计记录持久保存。"""
    skipped = replace(_reasoned_attempt(), provider_attempted=False)

    restored = AttemptExecutionRecord(**skipped.__dict__)

    assert restored == skipped
    assert restored.provider_attempted is False


def test_skipped_provider_error_cannot_masquerade_as_transport_failure() -> None:
    from werewolf_agent.model_gateway.router_errors import (
        _failure_disposition_from_attempts,
    )

    skipped = replace(
        _reasoned_attempt(),
        attempt_outcome=AttemptOutcome.FAILURE,
        root_cause=RootCause.PROVIDER_ERROR,
        provider_attempted=False,
    )

    assert _failure_disposition_from_attempts((skipped,)) is FailureDisposition.ROUTE_UNAVAILABLE


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


def test_execution_record_rejects_raw_string_enum_fields() -> None:
    base = _reasoned_attempt().__dict__
    raw_values = {
        "route_kind": "primary",
        "root_cause": "none",
        "attempt_outcome": "attempt_failure",
        "requested_reasoning_level": "high",
        "normalized_reasoning_status": "confirmed",
        "evidence_kind": "normalized_response",
    }
    for field, raw_value in raw_values.items():
        with pytest.raises(TypeError, match="enum fields"):
            AttemptExecutionRecord(**{**base, field: raw_value})


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


def test_generate_result_projects_attempts_into_legacy_usage_without_chain() -> None:
    legacy = UsageRecord(agent_id="p01", task_type="vote", provider="wrong", model="wrong")
    result = GenerateResult(text="ok", provider="wrong", model="wrong", usage=legacy, attempts=(_reasoned_attempt(),))
    assert result.usage is not None
    assert result.usage.attempts == result.attempts
    assert (result.usage.provider, result.usage.model) == ("primary", "model-a")


def test_usage_keeps_actual_fallback_route_when_final_attempt_is_retry() -> None:
    primary = replace(_reasoned_attempt(), attempt_outcome=AttemptOutcome.FAILURE, root_cause=RootCause.TIMEOUT)
    fallback = replace(primary, ordinal=2, provider="backup", route_kind=RouteKind.PROVIDER_FALLBACK)
    retry = replace(_reasoned_attempt(), ordinal=3, provider="backup", route_kind=RouteKind.RETRY)
    usage = UsageRecord(agent_id="p01", task_type="vote", provider="wrong", model="wrong", attempts=(primary, fallback, retry))
    assert (usage.fallback_provider, usage.fallback_model) == ("backup", "model-a")
