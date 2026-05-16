"""Model Router Gateway: provider-agnostic LLM routing.

All agents call ModelRouter.generate() — never provider SDKs directly.
Supports per-player llm_profile, per-task model selection, fallback chains,
retry, timeout, and cost tracking. No hardcoded API keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Resolved model configuration for one call."""
    provider: str
    model: str
    temperature: float = 0.5
    max_tokens: int = 1024
    top_p: float = 0.9
    timeout: int = 30


@dataclass(frozen=True)
class UsageRecord:
    """Single model call usage record."""
    agent_id: str
    task_type: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0
    fallback_reason: str | None = None
    success: bool = True


@dataclass
class GenerateResult:
    """Result from a model generation call."""
    text: str
    provider: str
    model: str
    usage: UsageRecord | None = None


# ---------------------------------------------------------------------------
# Provider protocol — pluggable backend
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Protocol for LLM providers. Implementations wrap specific SDKs."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult: ...


class MockProvider:
    """Mock provider for testing — returns deterministic placeholder text."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        start = time.monotonic()
        text = f"[{self._name}:{config.model}] mock response"
        latency = int((time.monotonic() - start) * 1000)
        return GenerateResult(
            text=text,
            provider=self._name,
            model=config.model,
            usage=UsageRecord(
                agent_id="",
                task_type="",
                provider=self._name,
                model=config.model,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
                latency_ms=latency,
            ),
        )


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """Central LLM routing gateway.

    Routes by agent_id + task_type to the configured provider/model.
    Falls back through the fallback chain if primary fails.
    Records all usage for cost tracking and reproducibility.
    """

    def __init__(
        self,
        model_profiles: dict[str, dict[str, Any]],
        llm_profiles: dict[str, dict[str, Any]],
        player_assignments: dict[str, str],
        providers: dict[str, LLMProvider] | None = None,
    ) -> None:
        self._model_profiles = model_profiles
        self._llm_profiles = llm_profiles
        self._player_assignments = player_assignments
        self._providers: dict[str, LLMProvider] = providers or {}
        self._usage_log: list[UsageRecord] = []

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        register_env_providers: bool = False,
    ) -> "ModelRouter":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        model_profiles = data.get("model_profiles", {})
        llm_profiles = data.get("llm_profiles", {})
        players = data.get("players", {})
        assignments = {
            pid: cfg["llm_profile"]
            for pid, cfg in players.items()
            if "llm_profile" in cfg
        }
        router = cls(
            model_profiles=model_profiles,
            llm_profiles=llm_profiles,
            player_assignments=assignments,
        )
        if register_env_providers:
            router.register_env_providers()
        return router

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def register_env_providers(self) -> None:
        """Register configured providers that have API keys in env/.env."""
        from werewolf_agent.model_gateway.providers import create_provider_from_env

        provider_names = self._configured_provider_names()
        for provider_name in provider_names:
            provider = create_provider_from_env(provider_name)
            if provider is not None:
                self.register_provider(provider)

    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def resolve_config(
        self, agent_id: str, task_type: str
    ) -> tuple[ModelConfig, str | None]:
        """Resolve model config for agent+task. Returns (config, fallback_provider_name)."""
        llm_profile_id = self._player_assignments.get(agent_id, "")
        llm_profile = self._llm_profiles.get(llm_profile_id, {})

        # Try task-specific first, then default
        task_cfg = llm_profile.get("tasks", {}).get(task_type)
        default_cfg = llm_profile.get("default", {})
        source = task_cfg or default_cfg

        if not source:
            return ModelConfig(provider="mock", model="mock"), "mock"

        provider_name = source.get("provider", "mock")
        model_profile_id = source.get("model_profile", "")
        model_profile = self._model_profiles.get(model_profile_id, {})

        config = ModelConfig(
            provider=provider_name,
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.5),
            max_tokens=model_profile.get("max_tokens", 1024),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 30),
        )

        # Fallback config
        fallback_cfg = llm_profile.get("fallback")
        fallback_provider = None
        if fallback_cfg:
            fallback_provider = fallback_cfg.get("provider")

        return config, fallback_provider

    def generate(
        self,
        agent_id: str,
        task_type: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        """Generate via routed provider with fallback."""
        config, fallback_provider = self.resolve_config(agent_id, task_type)

        provider = self._providers.get(config.provider)
        if provider is None:
            provider = MockProvider(config.provider)
            self._providers[config.provider] = provider

        try:
            result = provider.generate(prompt, config, system_prompt)
            if result.usage:
                usage = UsageRecord(
                    agent_id=agent_id,
                    task_type=task_type,
                    provider=result.provider,
                    model=result.model,
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    latency_ms=result.usage.latency_ms,
                    success=True,
                )
                self._usage_log.append(usage)
            return result
        except Exception:
            # Try fallback
            if fallback_provider and fallback_provider in self._providers:
                fb_provider = self._providers[fallback_provider]
                fallback_model_profile = self._resolve_fallback_model(llm_profile_id=self._player_assignments.get(agent_id, ""))
                fb_config = fallback_model_profile or ModelConfig(
                    provider=fallback_provider, model="fallback"
                )
                try:
                    result = fb_provider.generate(prompt, fb_config, system_prompt)
                    if result.usage:
                        usage = UsageRecord(
                            agent_id=agent_id,
                            task_type=task_type,
                            provider=result.provider,
                            model=result.model,
                            prompt_tokens=result.usage.prompt_tokens,
                            completion_tokens=result.usage.completion_tokens,
                            latency_ms=result.usage.latency_ms,
                            fallback_reason="primary_failed",
                            success=True,
                        )
                        self._usage_log.append(usage)
                    return result
                except Exception:
                    pass

            # Record failure
            self._usage_log.append(UsageRecord(
                agent_id=agent_id,
                task_type=task_type,
                provider=config.provider,
                model=config.model,
                success=False,
            ))
            return GenerateResult(
                text="",
                provider=config.provider,
                model=config.model,
            )

    def _resolve_fallback_model(self, llm_profile_id: str) -> ModelConfig | None:
        llm_profile = self._llm_profiles.get(llm_profile_id, {})
        fallback_cfg = llm_profile.get("fallback", {})
        if not fallback_cfg:
            return None
        model_profile_id = fallback_cfg.get("model_profile", "")
        model_profile = self._model_profiles.get(model_profile_id, {})
        return ModelConfig(
            provider=fallback_cfg.get("provider", "mock"),
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.3),
            max_tokens=model_profile.get("max_tokens", 256),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 10),
        )

    def _configured_provider_names(self) -> set[str]:
        providers: set[str] = {
            str(cfg.get("provider", ""))
            for cfg in self._model_profiles.values()
            if cfg.get("provider")
        }
        for llm_profile in self._llm_profiles.values():
            default_cfg = llm_profile.get("default", {})
            if default_cfg.get("provider"):
                providers.add(str(default_cfg["provider"]))
            for task_cfg in llm_profile.get("tasks", {}).values():
                if task_cfg.get("provider"):
                    providers.add(str(task_cfg["provider"]))
            fallback_cfg = llm_profile.get("fallback", {})
            if fallback_cfg.get("provider"):
                providers.add(str(fallback_cfg["provider"]))
        return providers

    def get_usage_log(self) -> list[UsageRecord]:
        return list(self._usage_log)

    def get_llm_profile_for_agent(self, agent_id: str) -> str:
        return self._player_assignments.get(agent_id, "")

    def config_snapshot(self) -> dict[str, Any]:
        """Snapshot for experiment reproducibility."""
        return {
            "model_profiles": dict(self._model_profiles),
            "llm_profiles": dict(self._llm_profiles),
            "player_assignments": dict(self._player_assignments),
        }
