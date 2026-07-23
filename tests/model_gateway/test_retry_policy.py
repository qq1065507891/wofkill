# -*- coding: utf-8 -*-
"""
验证模型网关纯重试分类、预算与确定性等待策略。

作者: Project contributors
创建日期: 2026-07-23
"""

from datetime import datetime, timezone

import pytest

from werewolf_agent.model_gateway.execution_records import RouteKind
from werewolf_agent.model_gateway.retry_policy import (
    RetryBudget,
    RetryKind,
    _is_retryable_exception,
    _retry_delay_for_exception,
    retry_delay,
    retry_kind_for_exception,
)


NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def test_generic_retry_delays_are_route_specific_and_deterministic(monkeypatch) -> None:
    monkeypatch.setattr("random.uniform", lambda *_: pytest.fail("不应使用随机抖动"))

    assert [retry_delay(RetryKind.GENERIC, RouteKind.PRIMARY, index) for index in range(4)] == [2, 4, 8, 16]
    assert [
        retry_delay(RetryKind.GENERIC, RouteKind.PROVIDER_FALLBACK, index)
        for index in range(2)
    ] == [2, 4]


def test_rate_limit_retry_delays_follow_exponential_baseline() -> None:
    assert [retry_delay(RetryKind.RATE_LIMIT, RouteKind.PRIMARY, index) for index in range(3)] == [16, 32, 64]
    assert [
        retry_delay(RetryKind.RATE_LIMIT, RouteKind.PROVIDER_FALLBACK, index)
        for index in range(3)
    ] == [16, 32, 64]


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        ("45", 45),
        ("5", 16),
        ("999", 300),
        ("invalid", 16),
        ("-1", 16),
        ("Wed, 23 Jul 2026 12:00:45 GMT", 45),
        ("Wed, 23 Jul 2026 11:59:00 GMT", 16),
    ],
)
def test_rate_limit_retry_after_respects_baseline_and_cap(retry_after: str, expected: float) -> None:
    assert retry_delay(
        RetryKind.RATE_LIMIT,
        RouteKind.PRIMARY,
        0,
        retry_after=retry_after,
        now=NOW,
    ) == expected


def test_rate_limit_retry_after_uses_http_date_with_injected_clock() -> None:
    assert retry_delay(
        RetryKind.RATE_LIMIT,
        RouteKind.PRIMARY,
        1,
        retry_after="Wed, 23 Jul 2026 12:02:00 GMT",
        now=NOW,
    ) == 120


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _ProviderError(Exception):
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        message: str = "provider transport failed",
    ) -> None:
        self.response = _Response(status_code, headers)
        super().__init__(message)


class _StatusOnlyError(Exception):
    def __init__(self, status_code: int, message: str = "provider transport failed") -> None:
        self.status_code = status_code
        super().__init__(message)


def test_exception_classification_keeps_rate_limits_distinct_from_generic_retries() -> None:
    assert retry_kind_for_exception(_ProviderError(429)) is RetryKind.RATE_LIMIT
    assert retry_kind_for_exception(_ProviderError(503)) is RetryKind.GENERIC
    assert retry_kind_for_exception(_ProviderError(400)) is None


def test_legacy_delay_wrapper_extracts_provider_retry_after_header() -> None:
    assert _retry_delay_for_exception(_ProviderError(429, {"Retry-After": "45"}), 0) == 45


def test_status_only_provider_errors_are_classified_before_message_heuristics() -> None:
    direct_rate_limit = _StatusOnlyError(429)
    response_rate_limit = _ProviderError(429, {"Retry-After": "45"})
    server_error = _StatusOnlyError(500)
    client_error = _StatusOnlyError(400)

    assert retry_kind_for_exception(direct_rate_limit) is RetryKind.RATE_LIMIT
    assert retry_kind_for_exception(response_rate_limit) is RetryKind.RATE_LIMIT
    assert _retry_delay_for_exception(response_rate_limit, 0) == 45
    assert _is_retryable_exception(server_error)
    assert retry_kind_for_exception(server_error) is RetryKind.GENERIC
    assert not _is_retryable_exception(client_error)
    assert retry_kind_for_exception(client_error) is None


@pytest.mark.parametrize(
    "client_error",
    [
        _StatusOnlyError(400, "upstream mentioned 429 and 503"),
        _ProviderError(400, message="upstream mentioned 429 and 503"),
    ],
)
def test_explicit_ordinary_4xx_overrides_misleading_retryable_message(client_error: Exception) -> None:
    assert not _is_retryable_exception(client_error)
    assert retry_kind_for_exception(client_error) is None


def test_primary_generic_budget_has_four_retries() -> None:
    budget = RetryBudget(RouteKind.PRIMARY, config_retry_count=4)

    assert [budget.try_consume(RetryKind.GENERIC) for _ in range(5)] == [True, True, True, True, False]
    assert budget.total_retry_count == 4
    assert budget.generic_retry_count == 4
    assert budget.rate_limit_retry_count == 0


def test_fallback_generic_budget_has_two_retries() -> None:
    budget = RetryBudget(RouteKind.PROVIDER_FALLBACK, config_retry_count=4)

    assert [budget.try_consume(RetryKind.GENERIC) for _ in range(3)] == [True, True, False]
    assert budget.total_retry_count == 2
    assert budget.generic_retry_count == 2


@pytest.mark.parametrize("route_kind", [RouteKind.PRIMARY, RouteKind.PROVIDER_FALLBACK])
def test_each_route_kind_has_three_rate_limit_retries(route_kind: RouteKind) -> None:
    budget = RetryBudget(route_kind, config_retry_count=4)

    assert [budget.try_consume(RetryKind.RATE_LIMIT) for _ in range(4)] == [True, True, True, False]
    assert budget.total_retry_count == 3
    assert budget.rate_limit_retry_count == 3


def test_zero_configured_retries_disables_every_kind() -> None:
    budget = RetryBudget(RouteKind.PRIMARY, config_retry_count=0)

    assert not budget.try_consume(RetryKind.GENERIC)
    assert not budget.try_consume(RetryKind.RATE_LIMIT)
    assert budget.total_retry_count == 0


def test_mixed_retry_kinds_share_total_budget_without_resetting_category_counts() -> None:
    budget = RetryBudget(RouteKind.PROVIDER_FALLBACK, config_retry_count=4)

    assert [
        budget.try_consume(RetryKind.GENERIC),
        budget.try_consume(RetryKind.GENERIC),
        budget.try_consume(RetryKind.RATE_LIMIT),
        budget.try_consume(RetryKind.RATE_LIMIT),
        budget.try_consume(RetryKind.RATE_LIMIT),
    ] == [True, True, True, True, False]
    assert budget.total_retry_count == 4
    assert budget.generic_retry_count == 2
    assert budget.rate_limit_retry_count == 2
