"""Base HTTP provider infrastructure."""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.router import UsageRecord

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
        self._http_client = http_client or __import__("httpx").Client()
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
