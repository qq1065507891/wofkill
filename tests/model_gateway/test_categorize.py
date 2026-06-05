"""Tests for the empty-response failure categorizer and GenerateResult plumbing.

R3-MG-2: ensure 4xx/5xx responses are classified as ``provider_error`` rather
than ``unknown``. The hardcoded ``http_status=0`` on GenerateResult must be
replaced with the actual response status from the provider.
"""

import pytest


class TestFourXxClassifiedAsProviderError:
    def test_4xx_classified_as_provider_error(self) -> None:
        """R3-MG-2: an HTTP 4xx is provider_error, not unknown."""
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=400,
            raw_error=None,
        )
        assert category == "provider_error"

    def test_5xx_classified_as_provider_error(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=503,
            raw_error=None,
        )
        assert category == "provider_error"


class TestCategorizeHttpStatusTakesPrecedence:
    def test_4xx_with_token_in_error_classified_as_provider_error(self) -> None:
        """R3-MG-6: http_status check runs BEFORE the string-match fallback,
        so a 4xx whose raw_error body contains the word 'token' still
        classifies as ``provider_error`` (a token-limit message coming back
        from a 4xx is a quota / billing issue, not a model context overflow).
        """
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=429,
            raw_error="rate limit exceeded: per-minute token quota reached",
        )
        assert category == "provider_error"

    def test_timeout_threshold_is_configurable(self) -> None:
        """R3-MG-6: timeout_threshold_ms can be raised to ignore a slow call."""
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=200,
            raw_error=None,
            timeout_threshold_ms=500,  # raised: 1s is now "too slow"
        )
        assert category == "timeout"


class TestGenerateResultPlumbing:
    def test_generate_result_default_http_status_is_zero(self) -> None:
        """R3-MG-2: backwards compat — default 0 is preserved."""
        from werewolf_agent.model_gateway.router import GenerateResult

        result = GenerateResult(text="hello", provider="mock", model="x")
        assert result.http_status == 0
        assert result.raw_error is None

    def test_generate_result_accepts_http_status_and_raw_error(self) -> None:
        from werewolf_agent.model_gateway.router import GenerateResult

        result = GenerateResult(
            text="",
            provider="anthropic",
            model="claude",
            http_status=503,
            raw_error="HTTP 503 Service Unavailable",
        )
        assert result.http_status == 503
        assert result.raw_error == "HTTP 503 Service Unavailable"
