"""HTTP LLM providers for ModelRouter.

API keys are read from environment variables or an untracked .env file.
The router still depends only on the LLMProvider protocol; provider SDKs are
not required for the V1 runtime.

Sub-modules:
- base: _BaseHttpProvider, ProviderConfigError, PROVIDER_DOTENV_KEYS
- env: get_env, load_local_dotenv, _ENV_OVERRIDES
- anthropic: AnthropicProvider + response parsers
- openai: OpenAIProvider + shared OpenAI-compatible generation
- glm: GLMProvider
- minimax: MiniMaxProvider
- factory: create_provider_from_env
"""

from werewolf_agent.model_gateway.providers.anthropic import (
    AnthropicProvider,
    _anthropic_tool_name,
    _extract_anthropic_text,
    _has_anthropic_tool_use,
)
from werewolf_agent.model_gateway.providers.base import (
    PROVIDER_DOTENV_KEYS,
    ProviderConfigError,
    _BaseHttpProvider,
)
from werewolf_agent.model_gateway.providers.env import (
    _ENV_OVERRIDES,
    get_env,
    load_local_dotenv,
)
from werewolf_agent.model_gateway.providers.factory import create_provider_from_env
from werewolf_agent.model_gateway.providers.glm import GLMProvider
from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
from werewolf_agent.model_gateway.providers.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "GLMProvider",
    "MiniMaxProvider",
    "OpenAIProvider",
    "PROVIDER_DOTENV_KEYS",
    "ProviderConfigError",
    "create_provider_from_env",
    "get_env",
    "load_local_dotenv",
]
