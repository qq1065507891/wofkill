# -*- coding: utf-8 -*-
"""
    功能描述：模型路由器网关 facade，负责配置解析、provider 路由、fallback 和用量追踪协调。
    作者：Mike
    创建日期：2025-01-15
    修改日期：2026-07-13
    使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from werewolf_agent.model_gateway.provider_call import (
    _call_provider_generate,
    _normalize_tool_metadata,
)
from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    EvidenceKind,
    OpaqueRequestId,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)
from werewolf_agent.model_gateway.retry_policy import (
    _failure_reason,
    _format_exception,
    _http_status_from_exception,
    _is_retryable_exception,
    _raw_error_from_exception,
)
from werewolf_agent.model_gateway.reasoning_policy import (
    reasoning_capability_satisfies,
    validate_player_reasoning_profiles,
)
from werewolf_agent.model_gateway.router_config import (
    _configured_provider_names,
    _validate_config,
)
from werewolf_agent.model_gateway.router_errors import (
    _empty_result,
    _record_failure_usage,
    _record_success_usage,
)
from werewolf_agent.model_gateway.router_probe import probe_tool_call_support
from werewolf_agent.model_gateway.router_selection import (
    _resolve_config,
    _resolve_fallback_model,
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
    "probe_tool_call_support",
    "_raw_error_from_exception",
    "_configured_provider_names",
    "_empty_result",
    "_record_failure_usage",
    "_record_success_usage",
    "_resolve_config",
    "_resolve_fallback_model",
    "_retry_delay_for_exception",
    "_validate_config",
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
        validate_reasoning: bool = False,
    ) -> None:
        self._model_profiles = model_profiles
        self._llm_profiles = llm_profiles
        self._player_assignments = player_assignments
        self._providers: dict[str, LLMProvider] = providers or {}
        self._usage_log: list[UsageRecord] = []
        self._usage_lock = threading.Lock()
        if validate_reasoning:
            validate_player_reasoning_profiles(
                model_profiles=model_profiles,
                llm_profiles=llm_profiles,
                player_assignments={k: v for k, v in player_assignments.items() if k != "judge"},
            )

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
        validate_player_reasoning_profiles(
            model_profiles=model_profiles,
            llm_profiles=llm_profiles,
            player_assignments={k: v for k, v in assignments.items() if k != "judge"},
        )
        if register_env_providers:
            router.register_env_providers()
        return router

    def _validate_config(self) -> None:
        """Cross-check references in model_profiles / llm_profiles / players.

        Catches typos at load time rather than at first LLM call. Raises
        ``ProviderConfigError`` for any dangling reference.
        """
        _validate_config(
            model_profiles=self._model_profiles,
            llm_profiles=self._llm_profiles,
            player_assignments=self._player_assignments,
        )

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
        return probe_tool_call_support(self, agent_id, task_type)

    def resolve_config(
        self, agent_id: str, task_type: str
    ) -> tuple[ModelConfig, str | None]:
        """Resolve model config for agent+task. Returns (config, fallback_provider_name)."""
        return _resolve_config(
            model_profiles=self._model_profiles,
            llm_profiles=self._llm_profiles,
            player_assignments=self._player_assignments,
            agent_id=agent_id,
            task_type=task_type,
        )

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
        entropy = uuid.uuid4().hex[:16]
        run_scope = (
            agent_id.lower()
            if agent_id.lower().isalnum() and 4 <= len(agent_id) <= 32
            else "game"
        )
        opaque_request_id = OpaqueRequestId.new(run_scope, entropy)
        request_id = opaque_request_id.value
        attempts: list[AttemptExecutionRecord] = []
        active_mode = resolve_structured_output_mode(
            provider=config.provider,
            configured_mode=structured_output_mode or config.structured_output_mode,
            allow_text_tool_fallback=config.allow_text_tool_fallback,
        )
        config = replace(config, structured_output_mode=active_mode.value)

        primary_capable = reasoning_capability_satisfies(
            config.reasoning_capability,
            config.reasoning_level,
        )
        provider = self._providers.get(config.provider) if primary_capable else None
        primary_unavailable = primary_capable and provider is None

        primary_error: Exception | None = None
        fallback_error: Exception | None = None
        last_empty_result: GenerateResult | None = None
        primary_attempts = 0

        max_retries = (getattr(config, "retry_count", 2) or 0) if provider else -1
        for attempt in range(max_retries + 1):
            primary_attempts += 1
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
                result = replace(
                    result,
                    allow_text_tool_fallback=config.allow_text_tool_fallback,
                    structured_output_mode=active_mode.value,
                    reasoning_level=config.reasoning_level,
                    reasoning_status=(
                        result.reasoning_status
                        if result.reasoning_status == "confirmed"
                        else "requested_unconfirmed" if config.reasoning_requested
                        else "not_requested"
                    ),
                )
                result = _normalize_tool_metadata(result, effective_tool_choice)
                if not result.text:
                    attempts.append(_attempt_record(
                        opaque_request_id, len(attempts) + 1, config, result,
                        RouteKind.PRIMARY if attempt == 0 else RouteKind.RETRY,
                        AttemptOutcome.FAILURE, RootCause.INVALID_OUTPUT,
                    ))
                    result = replace(result, attempts=tuple(attempts))
                    last_empty_result = result
                    primary_error = EmptyModelResponseError("empty_response")
                    logger.warning(
                        "Model generation returned empty text for agent=%s task=%s provider=%s model=%s",
                        agent_id,
                        task_type,
                        config.provider,
                        config.model,
                    )
                    if attempt < max_retries:
                        continue
                    break
                attempts.append(_attempt_record(
                    opaque_request_id, len(attempts) + 1, config, result,
                    RouteKind.PRIMARY if attempt == 0 else RouteKind.RETRY,
                    AttemptOutcome.SUCCESS, RootCause.NONE,
                ))
                result = replace(result, attempts=tuple(attempts))
                if result.usage is None:
                    result = replace(
                        result,
                        usage=UsageRecord(
                            agent_id=agent_id,
                            task_type=task_type,
                            provider=result.provider,
                            model=result.model,
                            attempts=tuple(attempts),
                        ),
                    )
                if result.usage:
                    _record_success_usage(
                        usage_log=self._usage_log,
                        usage_lock=self._usage_lock,
                        agent_id=agent_id,
                        task_type=task_type,
                        result=result,
                        structured_output_mode=active_mode.value,
                        request_id=request_id,
                        primary_provider=config.provider,
                        primary_model=config.model,
                        fallback_provider=fallback_provider,
                        retry_count=primary_attempts - 1,
                        reasoning_level=config.reasoning_level,
                        reasoning_status=result.reasoning_status,
                        reasoning_tokens=result.reasoning_tokens,
                    )
                return result
            except Exception as exc:
                primary_error = exc
                attempts.append(_attempt_record(
                    opaque_request_id, len(attempts) + 1, config, None,
                    RouteKind.PRIMARY if attempt == 0 else RouteKind.RETRY,
                    AttemptOutcome.FAILURE, _root_cause(exc),
                ))
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
        chain_result = self._generate_fallback_chain(
            agent_id=agent_id,
            task_type=task_type,
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            primary_config=config,
            primary_error=primary_error,
            request_id=opaque_request_id,
            attempts=attempts,
        )
        if chain_result is not None:
            return chain_result
        fallback_provider = None
        if fallback_provider and fallback_provider in self._providers:
            fb_provider = self._providers[fallback_provider]
            fallback_model_profile = self._resolve_fallback_model(
                llm_profile_id=self._player_assignments.get(agent_id, ""),
                required_reasoning_level=config.reasoning_level,
            )
            fb_config = fallback_model_profile or ModelConfig(
                provider=fallback_provider, model="fallback"
            )
            fb_config = replace(
                fb_config,
                reasoning_level=config.reasoning_level,
                reasoning_requested=config.reasoning_requested,
            )
            if not reasoning_capability_satisfies(
                fb_config.reasoning_capability,
                fb_config.reasoning_level,
            ):
                fallback_provider = None
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
            fallback_attempts = 0
            for fb_attempt in range(fb_max_retries + 1):
                fallback_attempts += 1
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
                    result = replace(
                        result,
                        allow_text_tool_fallback=fb_config.allow_text_tool_fallback,
                        structured_output_mode=fb_mode.value,
                        reasoning_level=fb_config.reasoning_level,
                        reasoning_status=(
                            result.reasoning_status
                            if result.reasoning_status == "confirmed"
                            else "requested_unconfirmed" if fb_config.reasoning_requested
                            else "not_requested"
                        ),
                    )
                    result = _normalize_tool_metadata(result, fb_effective_tool_choice)
                    if not result.text:
                        attempts.append(_attempt_record(
                            opaque_request_id, len(attempts) + 1, fb_config, result,
                            RouteKind.PROVIDER_FALLBACK,
                            AttemptOutcome.FAILURE, RootCause.INVALID_OUTPUT,
                        ))
                        result = replace(result, attempts=tuple(attempts))
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
                    attempts.append(_attempt_record(
                        opaque_request_id, len(attempts) + 1, fb_config, result,
                        RouteKind.PROVIDER_FALLBACK,
                        AttemptOutcome.SUCCESS, RootCause.NONE,
                    ))
                    result = replace(result, attempts=tuple(attempts))
                    if result.usage is None:
                        result = replace(
                            result,
                            usage=UsageRecord(
                                agent_id=agent_id,
                                task_type=task_type,
                                provider=result.provider,
                                model=result.model,
                                attempts=tuple(attempts),
                            ),
                        )
                    if result.usage:
                        _record_success_usage(
                            usage_log=self._usage_log,
                            usage_lock=self._usage_lock,
                            agent_id=agent_id,
                            task_type=task_type,
                            result=result,
                            fallback_reason=f"primary_failed:{_format_exception(primary_error)}",
                            structured_output_mode=fb_mode.value,
                            request_id=request_id,
                            primary_provider=config.provider,
                            primary_model=config.model,
                            fallback_provider=fb_config.provider,
                            fallback_model=fb_config.model,
                            retry_count=primary_attempts - 1 + fallback_attempts - 1,
                            failure_category="unknown" if primary_error else None,
                            reasoning_level=fb_config.reasoning_level,
                            reasoning_status=result.reasoning_status,
                            reasoning_tokens=result.reasoning_tokens,
                        )
                    return result
                except Exception as exc:
                    fallback_error = exc
                    attempts.append(_attempt_record(
                        opaque_request_id, len(attempts) + 1, fb_config, None,
                        RouteKind.PROVIDER_FALLBACK,
                        AttemptOutcome.FAILURE, _root_cause(exc),
                    ))
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
        terminal_root = (
            RootCause.PROVIDER_ERROR
            if primary_unavailable and not attempts
            else RootCause.POLICY_REJECTION
        )
        attempts.append(_attempt_record(
            opaque_request_id, len(attempts) + 1, config, None,
            RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE,
            terminal_root,
            terminal=True,
        ))
        _record_failure_usage(
            usage_log=self._usage_log,
            usage_lock=self._usage_lock,
            agent_id=agent_id,
            task_type=task_type,
            provider=config.provider,
            model=config.model,
            fallback_reason=failure_reason,
            structured_output_mode=active_mode.value,
            request_id=request_id,
            primary_provider=config.provider,
            primary_model=config.model,
            fallback_provider=fallback_provider,
            retry_count=primary_attempts - 1,
            failure_category="unknown" if primary_error else None,
            reasoning_level=config.reasoning_level,
            reasoning_status=(
                "requested_unconfirmed" if config.reasoning_requested
                else "not_requested"
            ),
            attempts=tuple(attempts),
        )
        # R3-MG-2: surface the HTTP status / raw error from the most recent
        # exception so the categorizer can attribute 4xx/5xx to
        # ``provider_error`` rather than the silent ``unknown`` fallback.
        return _empty_result(
            config_provider=config.provider,
            config_model=config.model,
            active_mode=active_mode.value,
            primary_error=primary_error,
            fallback_error=fallback_error,
            last_empty_result=last_empty_result,
            attempts=tuple(attempts),
        )

    def _resolve_fallback_model(
        self,
        llm_profile_id: str,
        required_reasoning_level: str = "none",
        candidate_index: int = 0,
    ) -> ModelConfig | None:
        return _resolve_fallback_model(
            model_profiles=self._model_profiles,
            llm_profiles=self._llm_profiles,
            llm_profile_id=llm_profile_id,
            required_reasoning_level=required_reasoning_level,
            candidate_index=candidate_index,
        )

    def _generate_fallback_chain(
        self,
        *,
        agent_id: str,
        task_type: str,
        prompt: str,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        primary_config: ModelConfig,
        primary_error: Exception | None,
        request_id: OpaqueRequestId,
        attempts: list[AttemptExecutionRecord],
    ) -> GenerateResult | None:
        """按配置顺序尝试所有能力合格的 fallback 候选。"""
        profile_id = self._player_assignments.get(agent_id, "")
        index = 0
        while True:
            config = self._resolve_fallback_model(
                profile_id,
                primary_config.reasoning_level,
                index,
            )
            if config is None:
                return None
            index += 1
            provider = self._providers.get(config.provider)
            if provider is None:
                continue
            config = replace(
                config,
                reasoning_level=primary_config.reasoning_level,
                reasoning_requested=primary_config.reasoning_requested,
            )
            mode = resolve_structured_output_mode(
                provider=config.provider,
                configured_mode=config.structured_output_mode,
                allow_text_tool_fallback=config.allow_text_tool_fallback,
            )
            config = replace(config, structured_output_mode=mode.value)
            retries = getattr(config, "retry_count", 1) or 0
            for retry_index in range(retries + 1):
                try:
                    effective_choice = (
                        tool_choice if mode == StructuredOutputMode.NATIVE_TOOL else None
                    )
                    result = _call_provider_generate(
                        provider, prompt, config, system_prompt,
                        tools=tools, tool_choice=effective_choice,
                    )
                    result = replace(
                        result,
                        structured_output_mode=mode.value,
                        reasoning_level=config.reasoning_level,
                        reasoning_status=(
                            result.reasoning_status
                            if result.reasoning_status == "confirmed"
                            else "requested_unconfirmed"
                        ),
                    )
                    result = _normalize_tool_metadata(result, effective_choice)
                    if not result.text:
                        attempts.append(_attempt_record(
                            request_id, len(attempts) + 1, config, result,
                            RouteKind.PROVIDER_FALLBACK, AttemptOutcome.FAILURE,
                            RootCause.INVALID_OUTPUT,
                        ))
                        if retry_index < retries:
                            continue
                        break
                    attempts.append(_attempt_record(
                        request_id, len(attempts) + 1, config, result,
                        RouteKind.PROVIDER_FALLBACK, AttemptOutcome.SUCCESS,
                        RootCause.NONE,
                    ))
                    if result.usage is None:
                        result = replace(result, usage=UsageRecord(
                            agent_id=agent_id, task_type=task_type,
                            provider=result.provider, model=result.model,
                            attempts=tuple(attempts),
                        ))
                    result = replace(result, attempts=tuple(attempts))
                    _record_success_usage(
                        usage_log=self._usage_log,
                        usage_lock=self._usage_lock,
                        agent_id=agent_id,
                        task_type=task_type,
                        result=result,
                        fallback_reason="provider_error" if primary_error else None,
                        structured_output_mode=mode.value,
                        primary_provider=primary_config.provider,
                        primary_model=primary_config.model,
                        fallback_provider=config.provider,
                        fallback_model=config.model,
                        reasoning_level=config.reasoning_level,
                        reasoning_status=result.reasoning_status,
                        reasoning_tokens=result.reasoning_tokens,
                    )
                    return result
                except Exception as exc:
                    attempts.append(_attempt_record(
                        request_id, len(attempts) + 1, config, None,
                        RouteKind.PROVIDER_FALLBACK, AttemptOutcome.FAILURE,
                        _root_cause(exc),
                    ))
                    if retry_index < retries and _is_retryable_exception(exc):
                        continue
                    break

    def _configured_provider_names(self) -> set[str]:
        return _configured_provider_names(self._model_profiles, self._llm_profiles)

    def get_usage_log(self) -> list[UsageRecord]:
        with self._usage_lock:
            return list(self._usage_log)

    def _terminal_policy_failure(
        self,
        agent_id: str,
        task_type: str,
        config: ModelConfig,
        request_id: OpaqueRequestId,
        root_cause: RootCause = RootCause.POLICY_REJECTION,
    ) -> GenerateResult:
        attempt = _attempt_record(
            request_id, 1, config, None, RouteKind.SAFE_FALLBACK,
            AttemptOutcome.FAILURE, root_cause, terminal=True,
        )
        usage = UsageRecord(
            agent_id=agent_id, task_type=task_type, provider=config.provider,
            model=config.model, attempts=(attempt,),
        )
        with self._usage_lock:
            self._usage_log.append(usage)
        return GenerateResult(
            text="", provider=config.provider, model=config.model,
            usage=usage, attempts=(attempt,),
        )

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


def _root_cause(exc: Exception) -> RootCause:
    """仅把异常归一化为审计枚举，不保留原始错误文本。"""
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return RootCause.TIMEOUT
    if isinstance(exc, EmptyModelResponseError):
        return RootCause.INVALID_OUTPUT
    return RootCause.PROVIDER_ERROR


def _attempt_record(
    request_id: OpaqueRequestId,
    ordinal: int,
    config: ModelConfig,
    result: GenerateResult | None,
    route_kind: RouteKind,
    outcome: AttemptOutcome,
    root_cause: RootCause,
    *,
    terminal: bool = False,
) -> AttemptExecutionRecord:
    """把 provider 规范化结果翻译成 Task0 的唯一逐次证据类型。"""
    level = ReasoningLevel(config.reasoning_level)
    tokens = int(result.reasoning_tokens if result else 0)
    raw_status = result.reasoning_status if result else "requested_unconfirmed"
    if terminal and level is not ReasoningLevel.NONE:
        status = ReasoningStatus.FALLBACK_DISABLED
        evidence = EvidenceKind.FALLBACK_DISABLED
        tokens = 0
    elif level is ReasoningLevel.NONE:
        status = ReasoningStatus.NOT_REQUESTED
        evidence = EvidenceKind.NONE
        tokens = 0
    elif tokens > 0:
        status = ReasoningStatus.CONFIRMED
        evidence = EvidenceKind.TOKEN_COUNT
    elif raw_status == "confirmed":
        status = ReasoningStatus.CONFIRMED
        evidence = EvidenceKind.AUTHORITATIVE_PROVIDER_EXECUTION
    elif raw_status == "unsupported":
        status = ReasoningStatus.UNSUPPORTED
        evidence = EvidenceKind.UNSUPPORTED
    else:
        status = ReasoningStatus.REQUESTED_UNCONFIRMED
        evidence = EvidenceKind.NONE
    return AttemptExecutionRecord(
        opaque_request_id=request_id,
        ordinal=ordinal,
        provider=config.provider,
        model=config.model,
        route_kind=route_kind,
        root_cause=root_cause,
        attempt_outcome=outcome,
        requested_reasoning_level=level,
        normalized_reasoning_status=status,
        reasoning_token_count=tokens,
        evidence_kind=evidence,
    )
