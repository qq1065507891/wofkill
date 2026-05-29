"""Provider factory: create providers from environment variables."""

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
