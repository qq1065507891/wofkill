"""ZhipuAI/GLM OpenAI-compatible provider."""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.providers.openai import _generate_openai_compatible
from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig


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
            api_key=api_key or get_env("GLM_API_KEY"),
            base_url=base_url or get_env("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
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
            tools=tools,
            tool_choice=tool_choice,
        )
