# -*- coding: utf-8 -*-
"""
功能描述：Provider 工厂函数，根据环境变量按需创建各厂商 Provider 实例
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
from werewolf_agent.model_gateway.providers.glm import GLMProvider
from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
from werewolf_agent.model_gateway.providers.openai import OpenAIProvider


def create_provider_from_env(provider_name: str):
    """Create a known provider only when its API key is present."""
    # Lazy imports so monkeypatch can reach load_local_dotenv / get_env
    from werewolf_agent.model_gateway.providers.env import get_env, load_local_dotenv

    load_local_dotenv()
    normalized = provider_name.lower()
    if normalized == "anthropic" and get_env("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if normalized == "openai" and get_env("OPENAI_API_KEY"):
        return OpenAIProvider()
    if normalized == "glm" and get_env("GLM_API_KEY"):
        return GLMProvider()
    if normalized == "minimax" and (
        get_env("MINIMAX_API_KEY") or get_env("ANTHROPIC_API_KEY")
    ):
        return MiniMaxProvider()
    return None
