# -*- coding: utf-8 -*-
"""
在模型请求发送前解析实际采样温度及覆盖原因。

作者: Project contributors
创建日期: 2026-07-27

使用示例:
    >>> from werewolf_agent.model_gateway.sampling_policy import resolve_sampling_audit
    >>> from werewolf_agent.model_gateway.usage_records import ModelConfig
    >>> resolve_sampling_audit(ModelConfig(provider="openai", model="x")).effective_temperature
    0.5
"""

from __future__ import annotations

from dataclasses import dataclass

from werewolf_agent.model_gateway.usage_records import ModelConfig


@dataclass(frozen=True)
class SamplingAudit:
    """记录单次 provider 调用最终采用的温度及覆盖原因。"""

    effective_temperature: float
    override_reason: str | None = None


def resolve_sampling_audit(config: ModelConfig) -> SamplingAudit:
    """在网络调用前解析 provider 特有的最终温度规则。"""
    if (
        config.provider.strip().lower() == "minimax"
        and config.reasoning_requested
    ):
        return SamplingAudit(
            effective_temperature=1.0,
            override_reason="thinking_requires_temperature_1",
        )
    return SamplingAudit(effective_temperature=config.temperature)
