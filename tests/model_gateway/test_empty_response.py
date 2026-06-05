"""Tests for empty-response root-cause attribution in LLM providers.

Task 3 of the pipeline-optimization plan. Empty responses (e.g. g_3528592081)
were previously opaque; we now attribute them to a category so the retry
loop can give better correction hints.
"""

import pytest


class TestCategorizeEmptyResponse:
    def test_timeout_marker_classified_as_timeout(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=30001,
            http_status=200,
            raw_error=None,
        )
        assert category == "timeout"

    def test_token_limit_classified_as_token_limit(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        # R3-MG-6: http_status is inspected first. Use http_status=0 (no
        # HTTP failure signal) so the token-string match still applies.
        category = categorize_empty_response(
            response_text="",
            latency_ms=5000,
            http_status=0,
            raw_error="max_tokens exceeded",
        )
        assert category == "token_limit"

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

    def test_nonempty_response_returns_none(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text='{"action_type": "vote"}',
            latency_ms=1000,
            http_status=200,
            raw_error=None,
        )
        assert category is None

    def test_unknown_empty_classified_as_unknown(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=200,
            raw_error=None,
        )
        assert category == "unknown"

    def test_network_error_classified(self) -> None:
        from werewolf_agent.model_gateway.providers.base import (
            categorize_empty_response,
        )
        category = categorize_empty_response(
            response_text="",
            latency_ms=2000,
            http_status=200,
            raw_error="Connection refused",
        )
        assert category == "network_error"
