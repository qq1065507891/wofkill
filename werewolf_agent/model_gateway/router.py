# -*- coding: utf-8 -*-
"""
    功能描述：模型路由器网关 facade，负责配置解析、provider 路由、fallback 和用量追踪协调。
    作者：Mike
    创建日期：2025-01-15
    修改日期：2026-07-06
    使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from werewolf_agent.model_gateway.provider_call import (
    _call_provider_generate,
    _normalize_tool_metadata,
)
from werewolf_agent.model_gateway.retry_policy import (
    _failure_reason,
    _format_exception,
    _http_status_from_exception,
    _is_retryable_exception,
    _raw_error_from_exception,
)
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
    StructuredOutputPolicy,
    resolve_structured_output_mode,
)
from werewolf_agent.model_gateway.usage_records import (
    EmptyModelResponseError,
    GenerateResult,
    LLMProvider,
    MockProvider,
    ModelConfig,
    UsageRecord,
)

logger = logging.getLogger(__name__)


__all__ = [
    "EmptyModelResponseError",
    "GenerateResult",
    "LLMProvider",
    "MockProvider",
    "ModelConfig",
    "ModelRouter",
    "UsageRecord",
    "_call_provider_generate",
    "_failure_reason",
    "_format_exception",
    "_http_status_from_exception",
    "_is_retryable_exception",
    "_normalize_tool_metadata",
    "_raw_error_from_exception",
    "_retry_delay_for_exception",
]


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
        self._usage_lock = threading.Lock()

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
        router._validate_config()
        if register_env_providers:
            router.register_env_providers()
        return router

    def _validate_config(self) -> None:
        """Cross-check references in model_profiles / llm_profiles / players.

        Catches typos at load time rather than at first LLM call. Raises
        ``ProviderConfigError`` for any dangling reference.
        """
        from werewolf_agent.model_gateway.providers.base import (
            ProviderConfigError,
        )

        # 1. Every players[id].llm_profile ref must exist in llm_profiles.
        for pid, profile_id in self._player_assignments.items():
            if profile_id not in self._llm_profiles:
                raise ProviderConfigError(
                    f"player {pid!r} references unknown llm_profile {profile_id!r}"
                )

        # 2. Every llm_profile entry (default / tasks / fallback) that
        #    names a model_profile must point at a real model_profile.
        for profile_id, profile in self._llm_profiles.items():
            for block_name in ("default", "fallback"):
                block = profile.get(block_name) or {}
                if not block:
                    continue
                mp_id = block.get("model_profile")
                if mp_id and mp_id not in self._model_profiles:
                    raise ProviderConfigError(
                        f"llm_profile {profile_id!r}.{block_name} "
                        f"references unknown model_profile {mp_id!r}"
                    )
            for task_type, task_cfg in (profile.get("tasks") or {}).items():
                mp_id = task_cfg.get("model_profile")
                if mp_id and mp_id not in self._model_profiles:
                    raise ProviderConfigError(
                        f"llm_profile {profile_id!r}.tasks.{task_type} "
                        f"references unknown model_profile {mp_id!r}"
                    )

        # 3. Every provider name must be one the factory can build
        #    (so an unknown name fails at config load, not first LLM call).
        from werewolf_agent.model_gateway.providers.factory import (
            create_provider_from_env,
        )

        known_providers: set[str] = set()
        for cfg in self._model_profiles.values():
            pname = cfg.get("provider")
            if pname:
                known_providers.add(str(pname))
        for profile in self._llm_profiles.values():
            for block_name in ("default", "fallback"):
                block = profile.get(block_name) or {}
                pname = block.get("provider")
                if pname:
                    known_providers.add(str(pname))
            for task_cfg in (profile.get("tasks") or {}).values():
                pname = task_cfg.get("provider")
                if pname:
                    known_providers.add(str(pname))
        for pname in known_providers:
            # 'create_provider_from_env' returns None when the API key is
            # missing but does not validate the provider name. We probe
            # the factory's known list via its module rather than the
            # import side-effects.
            try:
                if pname.lower() not in {
                    "anthropic", "openai", "glm", "minimax", "mock",
                }:
                    raise ProviderConfigError(
                        f"provider {pname!r} is not registered "
                        "(known: anthropic, openai, glm, minimax, mock)"
                    )
            except ProviderConfigError:
                raise
            # Touch factory for import-side-effect parity.
            _ = create_provider_from_env

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def register_env_providers(self) -> None:
        """Register configured providers that have API keys in env/.env.

        R3-MG-8: log a WARNING for every configured provider name that
        resolves to ``None`` (i.e. its required API key is missing) so
        silent fallback is at least visible in the log. The pre-R3-MG-8
        behavior swallowed the None return value and only the
        downstream ``Provider not found`` error surfaced.
        """
        from werewolf_agent.model_gateway.providers import create_provider_from_env

        provider_names = self._configured_provider_names()
        missing: list[str] = []
        for provider_name in provider_names:
            provider = create_provider_from_env(provider_name)
            if provider is not None:
                self.register_provider(provider)
            else:
                missing.append(provider_name)
        if missing:
            logger.warning(
                "register_env_providers: %d configured provider(s) had no "
                "API key in env/.env and were skipped: %s",
                len(missing),
                sorted(missing),
            )

    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def probe_tool_call_support(self, agent_id: str, task_type: str) -> dict[str, Any]:
        """Probe whether the resolved provider returns an actual tool call."""
        config, _fallback_provider = self.resolve_config(agent_id, task_type)
        config = replace(
            config,
            structured_output_mode=StructuredOutputMode.NATIVE_TOOL.value,
        )
        provider = self._providers.get(config.provider)
        if provider is None:
            raise RuntimeError(f"Provider '{config.provider}' not found. Available: {list(self._providers.keys())}")

        tool = {
            "name": "submit_player_action",
            "description": "Probe structured action tool-call support.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action_type": {"type": "string", "enum": ["no_action"]},
                    "target_id": {"enum": [None]},
                    "speech": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["action_type", "target_id", "speech", "reason", "confidence"],
            },
        }
        try:
            result = _call_provider_generate(
                provider,
                "Call submit_player_action with no_action probe arguments.",
                config,
                "You are checking tool-call support. Use the tool.",
                tools=[tool],
                tool_choice={"type": "tool", "name": "submit_player_action"},
            )
            _normalize_tool_metadata(result, {"type": "tool", "name": "submit_player_action"})
        except Exception as exc:
            return {
                "supported": False,
                "provider": config.provider,
                "model": config.model,
                "failure_reason": _format_exception(exc),
                "tool_call_received": False,
                "text_fallback_used": False,
            }
        supported = bool(result.tool_call_received) and not result.structured_failure_reason
        return {
            "supported": supported,
            "provider": result.provider,
            "model": result.model,
            "failure_reason": result.structured_failure_reason,
            "tool_call_received": result.tool_call_received,
            "tool_call_name": result.tool_call_name,
            "text_fallback_used": result.text_fallback_used,
        }

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
        structured_policy = StructuredOutputPolicy.from_model_profile(
            provider=provider_name,
            model_profile=model_profile,
        )

        config = ModelConfig(
            provider=provider_name,
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.5),
            max_tokens=model_profile.get("max_tokens", 1024),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 30),
            allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
            retry_count=int(model_profile.get("retry_count", 2)),
            structured_output_mode=structured_policy.primary_mode.value,
            structured_output_fallback_modes=tuple(
                mode.value for mode in structured_policy.fallback_modes
            ),
        )

        # Fallback config
        fallback_cfg = llm_profile.get("fallback")
        fallback_provider = None
        if fallback_cfg:
            fallback_provider = fallback_cfg.get("provider")

        return config, fallback_provider

    def resolve_structured_output_policy(
        self,
        agent_id: str,
        task_type: str,
    ) -> StructuredOutputPolicy:
        config, _ = self.resolve_config(agent_id, task_type)
        return StructuredOutputPolicy.from_config(config)

    def generate(
        self,
        agent_id: str,
        task_type: str,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        structured_output_mode: str | None = None,
        jitter_seconds: tuple[float, float] = (0.0, 0.8),
    ) -> GenerateResult:
        """Generate via routed provider with fallback.

        ``jitter_seconds`` is the (low, high) range of a uniform random
        sleep applied before the FIRST attempt only. Defaults to
        ``(0, 0.8)`` to spread concurrent requests. Pass ``(0, 0)`` in
        tests to avoid 5-15s of cumulative wait across 12 players.
        """
        config, fallback_provider = self.resolve_config(agent_id, task_type)
        active_mode = resolve_structured_output_mode(
            provider=config.provider,
            configured_mode=structured_output_mode or config.structured_output_mode,
            allow_text_tool_fallback=config.allow_text_tool_fallback,
        )
        config = replace(config, structured_output_mode=active_mode.value)

        provider = self._providers.get(config.provider)
        if provider is None:
            raise RuntimeError(f"Provider '{config.provider}' not found. Available: {list(self._providers.keys())}")

        primary_error: Exception | None = None
        fallback_error: Exception | None = None
        last_empty_result: GenerateResult | None = None

        max_retries = getattr(config, "retry_count", 2) or 0
        for attempt in range(max_retries + 1):
            # Pre-call jitter: spread concurrent requests to avoid rate-limiting.
            # On the first attempt only — retries already have backoff.
            # R3-MG-3: skip entirely when jitter_seconds == (0, 0).
            if attempt == 0 and jitter_seconds != (0.0, 0.0):
                time.sleep(random.uniform(jitter_seconds[0], jitter_seconds[1]))
            try:
                effective_tool_choice = (
                    tool_choice
                    if active_mode == StructuredOutputMode.NATIVE_TOOL
                    else None
                )
                result = _call_provider_generate(
                    provider,
                    prompt,
                    config,
                    system_prompt,
                    tools=tools,
                    tool_choice=effective_tool_choice,
                )
                result.allow_text_tool_fallback = config.allow_text_tool_fallback
                result.structured_output_mode = active_mode.value
                _normalize_tool_metadata(result, effective_tool_choice)
                if not result.text:
                    last_empty_result = result
                    primary_error = EmptyModelResponseError("empty_response")
                    logger.warning(
                        "Model generation returned empty text for agent=%s task=%s provider=%s model=%s",
                        agent_id,
                        task_type,
                        config.provider,
                        config.model,
                    )
                    break
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
                        structured_output_mode=active_mode.value,
                    )
                    with self._usage_lock:
                        self._usage_log.append(usage)
                        if len(self._usage_log) > 10000:
                            self._usage_log = self._usage_log[-5000:]
                return result
            except Exception as exc:
                primary_error = exc
                if attempt < max_retries and _is_retryable_exception(exc):
                    delay = _retry_delay_for_exception(exc, attempt)
                    logger.warning(
                        "Retryable error for agent=%s task=%s provider=%s (attempt %d/%d, retry in %.1fs): %s",
                        agent_id, task_type, config.provider,
                        attempt + 1, max_retries + 1, delay,
                        _format_exception(exc),
                    )
                    time.sleep(delay)
                    continue
                logger.warning(
                    "Model generation failed for agent=%s task=%s provider=%s model=%s: %s",
                    agent_id,
                    task_type,
                    config.provider,
                    config.model,
                    _format_exception(exc),
                )
                break

        # Try fallback
        if fallback_provider and fallback_provider in self._providers:
            fb_provider = self._providers[fallback_provider]
            fallback_model_profile = self._resolve_fallback_model(llm_profile_id=self._player_assignments.get(agent_id, ""))
            fb_config = fallback_model_profile or ModelConfig(
                provider=fallback_provider, model="fallback"
            )
            fb_mode = resolve_structured_output_mode(
                provider=fb_config.provider,
                configured_mode=fb_config.structured_output_mode,
                allow_text_tool_fallback=fb_config.allow_text_tool_fallback,
            )
            fb_config = replace(
                fb_config,
                structured_output_mode=fb_mode.value,
            )
            fb_max_retries = getattr(fb_config, "retry_count", 1) or 0
            for fb_attempt in range(fb_max_retries + 1):
                try:
                    fb_effective_tool_choice = (
                        tool_choice
                        if fb_mode == StructuredOutputMode.NATIVE_TOOL
                        else None
                    )
                    result = _call_provider_generate(
                        fb_provider,
                        prompt,
                        fb_config,
                        system_prompt,
                        tools=tools,
                        tool_choice=fb_effective_tool_choice,
                    )
                    result.allow_text_tool_fallback = fb_config.allow_text_tool_fallback
                    result.structured_output_mode = fb_mode.value
                    _normalize_tool_metadata(result, fb_effective_tool_choice)
                    if not result.text:
                        last_empty_result = result
                        fallback_error = EmptyModelResponseError("empty_response")
                        logger.warning(
                            "Fallback model generation returned empty text for agent=%s task=%s provider=%s model=%s",
                            agent_id,
                            task_type,
                            fb_config.provider,
                            fb_config.model,
                        )
                        break
                    if result.usage:
                        usage = UsageRecord(
                            agent_id=agent_id,
                            task_type=task_type,
                            provider=result.provider,
                            model=result.model,
                            prompt_tokens=result.usage.prompt_tokens,
                            completion_tokens=result.usage.completion_tokens,
                            latency_ms=result.usage.latency_ms,
                            fallback_reason=f"primary_failed:{_format_exception(primary_error)}",
                            success=True,
                            structured_output_mode=fb_mode.value,
                        )
                        with self._usage_lock:
                            self._usage_log.append(usage)
                            if len(self._usage_log) > 10000:
                                self._usage_log = self._usage_log[-5000:]
                    return result
                except Exception as exc:
                    fallback_error = exc
                    if fb_attempt < fb_max_retries and _is_retryable_exception(exc):
                        delay = _retry_delay_for_exception(exc, fb_attempt)
                        logger.warning(
                            "Retryable fallback error for agent=%s task=%s (attempt %d/%d, retry in %.1fs): %s",
                            agent_id, task_type,
                            fb_attempt + 1, fb_max_retries + 1, delay,
                            _format_exception(exc),
                        )
                        # Rate-limit backoff: wait before retrying
                        time.sleep(delay)
                        continue
                    logger.warning(
                        "Fallback model generation failed for agent=%s task=%s provider=%s model=%s: %s",
                        agent_id,
                        task_type,
                        fb_config.provider,
                        fb_config.model,
                        _format_exception(exc),
                    )
                    break

        # Record failure
        failure_reason = _failure_reason(primary_error, fallback_error)
        with self._usage_lock:
            self._usage_log.append(UsageRecord(
                agent_id=agent_id,
                task_type=task_type,
                provider=config.provider,
                model=config.model,
                fallback_reason=failure_reason,
                success=False,
                structured_output_mode=active_mode.value,
            ))
            if len(self._usage_log) > 10000:
                self._usage_log = self._usage_log[-5000:]
        # R3-MG-2: surface the HTTP status / raw error from the most recent
        # exception so the categorizer can attribute 4xx/5xx to
        # ``provider_error`` rather than the silent ``unknown`` fallback.
        return GenerateResult(
            text="",
            provider=last_empty_result.provider if last_empty_result else config.provider,
            model=last_empty_result.model if last_empty_result else config.model,
            usage=last_empty_result.usage if last_empty_result else None,
            http_status=_http_status_from_exception(primary_error)
            or _http_status_from_exception(fallback_error),
            raw_error=_raw_error_from_exception(primary_error)
            or _raw_error_from_exception(fallback_error),
            structured_output_mode=(
                last_empty_result.structured_output_mode
                if last_empty_result
                else active_mode.value
            ),
        )

    def _resolve_fallback_model(self, llm_profile_id: str) -> ModelConfig | None:
        llm_profile = self._llm_profiles.get(llm_profile_id, {})
        fallback_cfg = llm_profile.get("fallback", {})
        if not fallback_cfg:
            return None
        model_profile_id = fallback_cfg.get("model_profile", "")
        # R3-MG-7: a fallback that references a missing model_profile
        # used to silently return ModelConfig(model="") at first
        # fallback invocation, which the LLM call would then explode
        # against. Raise at config load time instead.
        if not model_profile_id:
            from werewolf_agent.model_gateway.providers.base import (
                ProviderConfigError,
            )
            raise ProviderConfigError(
                f"llm_profile {llm_profile_id!r}.fallback has no model_profile"
            )
        if model_profile_id not in self._model_profiles:
            from werewolf_agent.model_gateway.providers.base import (
                ProviderConfigError,
            )
            raise ProviderConfigError(
                f"llm_profile {llm_profile_id!r}.fallback references "
                f"unknown model_profile {model_profile_id!r}"
            )
        model_profile = self._model_profiles.get(model_profile_id, {})
        structured_policy = StructuredOutputPolicy.from_model_profile(
            provider=fallback_cfg.get("provider", "mock"),
            model_profile=model_profile,
        )
        return ModelConfig(
            provider=fallback_cfg.get("provider", "mock"),
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.3),
            max_tokens=model_profile.get("max_tokens", 256),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 10),
            allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
            retry_count=int(model_profile.get("retry_count", 1)),
            structured_output_mode=structured_policy.primary_mode.value,
            structured_output_fallback_modes=tuple(
                mode.value for mode in structured_policy.fallback_modes
            ),
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
        with self._usage_lock:
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

def _retry_delay_for_exception(exc: Exception, attempt: int) -> float:
    from werewolf_agent.model_gateway.retry_policy import _retry_delay_for_exception as _impl

    return _impl(exc, attempt, uniform=random.uniform)
