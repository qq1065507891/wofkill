# -*- coding: utf-8 -*-
"""
功能描述：智谱 GLM OpenAI 兼容 Provider

作者：Mike
创建日期：2025-01-15
修改日期：2026-07-15

2026-07-15 新增：``config.base_url`` 覆盖 provider 实例默认 URL（透传给
``_generate_openai_compatible``）。``config.extra_body`` 在共享 helper
中已自动合并。

使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.providers.openai import _generate_openai_compatible
from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptObserver
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
        final_prompt_observer: FinalPromptObserver | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return _generate_openai_compatible(
            provider=self,
            base_url=config.base_url or self._base_url,
            api_key=self._api_key,
            http_client=self._http_client,
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            final_prompt_observer=final_prompt_observer,
        )
