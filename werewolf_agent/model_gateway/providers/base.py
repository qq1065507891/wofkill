"""Base HTTP provider infrastructure."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from werewolf_agent.model_gateway.router import UsageRecord

# Failure categories for empty LLM responses (pipeline-optimization Task 3).
# These surface in RetryInfo so the retry loop can give better correction
# hints. The categorize_empty_response() helper below is the single
# source of truth for attribution.
FailureCategory = Literal[
    "timeout", "token_limit", "provider_error", "network_error", "unknown",
]


def categorize_empty_response(
    *,
    response_text: str,
    latency_ms: int,
    http_status: int,
    raw_error: str | None,
    timeout_threshold_ms: int = 30000,
) -> FailureCategory | None:
    """Categorize the cause of an empty LLM response.

    Returns None if the response is non-empty (no attribution needed).
    Order of checks:
      1. Non-empty response → None
      2. High latency → timeout
      3. Token-shaped error string → token_limit
      4. HTTP 5xx → provider_error
      5. HTTP 4xx → provider_error
      6. Network-shaped error string → network_error
      7. Otherwise → unknown
    """
    if response_text and response_text.strip():
        return None
    if latency_ms > timeout_threshold_ms:
        return "timeout"
    if raw_error and "token" in raw_error.lower():
        return "token_limit"
    if http_status >= 500:
        return "provider_error"
    if http_status >= 400:
        return "provider_error"
    if raw_error and (
        "network" in raw_error.lower() or "connection" in raw_error.lower()
    ):
        return "network_error"
    return "unknown"

PROVIDER_DOTENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_BASE_URL",
}


class ProviderConfigError(RuntimeError):
    """Raised when a provider cannot be configured safely."""


class _BaseHttpProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        http_client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderConfigError(f"{self.name} API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx.Client()
        self._owns_http = http_client is None

    def close(self) -> None:
        if self._owns_http:
            self._http_client.close()

    def __enter__(self) -> "_BaseHttpProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _usage(
        self,
        *,
        model: str,
        latency_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> UsageRecord:
        return UsageRecord(
            agent_id="",
            task_type="",
            provider=self.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
