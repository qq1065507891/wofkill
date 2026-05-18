"""HTTP LLM providers for ModelRouter.

API keys are read from environment variables or an untracked .env file.
The router still depends only on the LLMProvider protocol; provider SDKs are
not required for the V1 runtime.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig, UsageRecord


PROVIDER_DOTENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
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


class AnthropicProvider(_BaseHttpProvider):
    """Anthropic Messages API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.monotonic()
        response = self._http_client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        data = response.json()
        text = _extract_anthropic_text(data)
        usage = data.get("usage", {})
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            usage=self._usage(
                model=config.model,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
        )


class OpenAIProvider(_BaseHttpProvider):
    """OpenAI-compatible Chat Completions provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return _generate_openai_compatible(
            provider=self,
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=self._http_client,
            messages=messages,
            config=config,
        )


class GLMProvider(_BaseHttpProvider):
    """ZhipuAI/GLM OpenAI-compatible provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("GLM_API_KEY", ""),
            base_url=base_url or os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "glm"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return _generate_openai_compatible(
            provider=self,
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=self._http_client,
            messages=messages,
            config=config,
        )


def create_provider_from_env(provider_name: str):
    """Create a known provider only when its API key is present."""
    load_local_dotenv()
    normalized = provider_name.lower()
    if normalized == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if normalized == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if normalized == "glm" and os.getenv("GLM_API_KEY"):
        return GLMProvider()
    return None


def load_local_dotenv(
    path: str | Path = ".env",
    *,
    keys: set[str] | None = None,
) -> None:
    """Load provider KEY=VALUE lines without requiring python-dotenv at runtime."""
    env_path = Path(path)
    if not env_path.exists():
        return
    allowed_keys = keys or PROVIDER_DOTENV_KEYS
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed_keys:
            continue
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value


def _generate_openai_compatible(
    *,
    provider: _BaseHttpProvider,
    base_url: str,
    api_key: str,
    http_client: Any,
    messages: list[dict[str, str]],
    config: ModelConfig,
) -> GenerateResult:
    start = time.monotonic()
    response = http_client.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    latency_ms = int((time.monotonic() - start) * 1000)
    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return GenerateResult(
        text=text,
        provider=provider.name,
        model=config.model,
        usage=provider._usage(
            model=config.model,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
    )


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)
