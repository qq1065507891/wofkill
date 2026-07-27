# -*- coding: utf-8 -*-
"""
    功能描述：模型路由器网关 facade，负责配置解析、provider 路由、fallback 和用量追踪协调。
    作者：Mike
    创建日期：2025-01-15
    修改日期：2026-07-27
    使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

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
from werewolf_agent.model_gateway.fallback_policy import (
    FALLBACK_ROUTE_UNAVAILABLE,
    route_switch_is_valid,
)
from werewolf_agent.model_gateway.final_prompt_observer import (
    FinalPromptObserver,
    bind_attempt,
)
from werewolf_agent.model_gateway.generation_attempt_context import (
    GenerationAttemptContext,
)
from werewolf_agent.model_gateway.provider_call import (
    _call_provider_generate,
    _normalize_tool_metadata,
)
from werewolf_agent.model_gateway.reasoning_policy import (
    reasoning_capability_satisfies,
    validate_player_reasoning_profiles,
)
from werewolf_agent.model_gateway.retry_policy import (
    RetryBudget,
    RetryKind,
    _failure_reason,
    _format_exception,
    _http_status_from_exception,
    _is_retryable_exception,
    _raw_error_from_exception,
    _retry_after_from_exception,
    retry_delay,
    retry_kind_for_exception,
)
from werewolf_agent.model_gateway.router_config import (
    _canonical_provider_name,
    _configured_provider_names,
    _validate_config,
)
from werewolf_agent.model_gateway.router_errors import (
    _empty_result,
    _failure_disposition_from_attempts,
    _record_failure_usage,
    _record_success_usage,
)
from werewolf_agent.model_gateway.router_probe import probe_tool_call_support
from werewolf_agent.model_gateway.router_selection import (
    _resolve_config,
    _resolve_fallback_model,
    _resolve_fallback_routes,
)
from werewolf_agent.model_gateway.sampling_policy import (
    SamplingAudit,
    resolve_sampling_audit,
)
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
    StructuredOutputPolicy,
    resolve_structured_output_mode,
)
from werewolf_agent.model_gateway.usage_records import (
    EmptyModelResponseError,
    FailureDisposition,
    GenerateResult,
    LLMProvider,
    MockProvider,
    ModelConfig,
    StructuredOutputUnsupportedError,
    UsageRecord,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FallbackChainOutcome:
    """汇总备用链结果及最后一次真实 provider 尝试的审计信息。"""

    result: GenerateResult | None = None
    route_failure: str | None = None
    error: Exception | None = None
    sampling_audit: SamplingAudit | None = None
    last_empty_result: GenerateResult | None = None


__all__ = [
    "EmptyModelResponseError",
    "FailureDisposition",
    "GenerateResult",
    "LLMProvider",
    "MockProvider",
    "ModelConfig",
    "ModelRouter",
    "StructuredOutputUnsupportedError",
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
        allow_test_model_capability: bool = False,
    ) -> None:
        self._model_profiles = model_profiles
        self._llm_profiles = llm_profiles
        self._player_assignments = player_assignments
        self._providers: dict[str, LLMProvider] = {}
        for provider_key, provider in (providers or {}).items():
            canonical_key = _canonical_provider_name(provider_key)
            # 保留显式 canonical key，避免大小写别名覆盖已注册 provider。
            if canonical_key not in self._providers or provider_key == canonical_key:
                self._providers[canonical_key] = provider
        self._usage_log: list[UsageRecord] = []
        self._usage_lock = threading.Lock()
        self._allow_test_model_capability = allow_test_model_capability
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
        from werewolf_agent.model_gateway.providers.base import ProviderConfigError

        if not isinstance(data, dict):
            raise ProviderConfigError("router config root must be a mapping")
        model_profiles = data.get("model_profiles", {})
        llm_profiles = data.get("llm_profiles", {})
        players = data.get("players", {})
        for section_name, section in (
            ("model_profiles", model_profiles),
            ("llm_profiles", llm_profiles),
            ("players", players),
        ):
            if not isinstance(section, dict):
                raise ProviderConfigError(
                    f"router config {section_name} must be a mapping"
                )
        for pid, player in players.items():
            if not isinstance(player, dict):
                raise ProviderConfigError(f"player {pid!r} must be a mapping")
            profile_id = player.get("llm_profile")
            if profile_id is not None and (
                not isinstance(profile_id, str) or not profile_id.strip()
            ):
                raise ProviderConfigError(
                    f"player {pid!r}.llm_profile must be a nonblank string"
                )
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
        self._providers[_canonical_provider_name(provider.name)] = provider

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
        config, fallback = _resolve_config(
            model_profiles=self._model_profiles,
            llm_profiles=self._llm_profiles,
            player_assignments=self._player_assignments,
            agent_id=agent_id,
            task_type=task_type,
        )
        if (
            self._allow_test_model_capability
            and config.reasoning_capability == "none"
        ):
            config = replace(config, reasoning_capability="high")
        return config, fallback

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
        generation_attempt_context: GenerationAttemptContext | None = None,
        final_prompt_observer: FinalPromptObserver | None = None,
        max_provider_calls: int | None = None,
    ) -> GenerateResult:
        """Generate via routed provider with fallback.

        ``jitter_seconds`` is the (low, high) range of a uniform random
        sleep applied before the FIRST attempt only. Defaults to
        ``(0, 0.8)`` to spread concurrent requests. Pass ``(0, 0)`` in
        tests to avoid 5-15s of cumulative wait across 12 players.
        """
        if max_provider_calls is not None and max_provider_calls < 1:
            raise ValueError("max_provider_calls must be positive when provided")
        config, fallback_provider = self.resolve_config(agent_id, task_type)
        entropy = uuid.uuid4().hex[:16]
        run_scope = (
            agent_id.lower()
            if agent_id.lower().isalnum() and 4 <= len(agent_id) <= 32
            else "game"
        )
        opaque_request_id = (
            generation_attempt_context.opaque_request_id
            if generation_attempt_context else OpaqueRequestId.new(run_scope, entropy)
        )
        request_id = opaque_request_id.value
        attempts: list[AttemptExecutionRecord] = list(
            generation_attempt_context.attempts if generation_attempt_context else ()
        )
        first_route = (
            generation_attempt_context.next_route_kind
            if generation_attempt_context else RouteKind.PRIMARY
        )
        active_mode = resolve_structured_output_mode(
            provider=config.provider,
            configured_mode=structured_output_mode or config.structured_output_mode,
            allow_text_tool_fallback=config.allow_text_tool_fallback,
        )
        config = replace(config, structured_output_mode=active_mode.value)
        primary_sampling_audit = resolve_sampling_audit(config)

        primary_capable = reasoning_capability_satisfies(
            config.reasoning_capability,
            config.reasoning_level,
        )
        provider = self._providers.get(config.provider) if primary_capable else None
        primary_skip_reason = (
            "reasoning_unsupported"
            if not primary_capable
            else "provider_unavailable"
            if provider is None
            else None
        )

        primary_error: Exception | None = None
        fallback_error: Exception | None = None
        last_empty_result: GenerateResult | None = None
        primary_attempts = 0
        primary_audit_config = config
        primary_skip_root: RootCause | None = None

        if primary_skip_reason is not None:
            (
                primary_audit_config,
                primary_error,
                primary_skip_root,
            ) = _record_skipped_primary_attempt(
                attempts=attempts,
                request_id=opaque_request_id,
                config=config,
                route_kind=first_route,
                reason=primary_skip_reason,
            )

        primary_budget = RetryBudget(
            RouteKind.PRIMARY,
            int(getattr(config, "retry_count", 4) or 0),
        )
        attempt = 0
        while (
            provider is not None
            and (
                max_provider_calls is None
                or primary_attempts < max_provider_calls
            )
        ):
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
                    final_prompt_observer=bind_attempt(
                        final_prompt_observer,
                        attempt_kind=(first_route.value if attempt == 0 else RouteKind.RETRY.value),
                        attempt_ordinal=len(attempts) + 1,
                    ),
                )
                result = replace(
                    result,
                    allow_text_tool_fallback=config.allow_text_tool_fallback,
                    structured_output_mode=active_mode.value,
                    reasoning_level=config.reasoning_level,
                    failure_disposition=FailureDisposition.NONE,
                    reasoning_status=(
                        result.reasoning_status
                        if result.reasoning_status in {"confirmed", "unsupported"}
                        else "requested_unconfirmed" if config.reasoning_requested
                        else "not_requested"
                    ),
                )
                result = _normalize_tool_metadata(result, effective_tool_choice)
                if result.reasoning_status == "unsupported":
                    attempts.append(_attempt_record(
                        opaque_request_id, len(attempts) + 1, config, result,
                        first_route if attempt == 0 else RouteKind.RETRY,
                        AttemptOutcome.FAILURE, RootCause.POLICY_REJECTION,
                    ))
                    primary_error = RuntimeError("reasoning_unsupported")
                    break
                if not result.text:
                    attempts.append(_attempt_record(
                        opaque_request_id, len(attempts) + 1, config, result,
                        first_route if attempt == 0 else RouteKind.RETRY,
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
                    break
                attempts.append(_attempt_record(
                    opaque_request_id, len(attempts) + 1, config, result,
                    first_route if attempt == 0 else RouteKind.RETRY,
                    AttemptOutcome.SUCCESS, RootCause.NONE,
                ))
                result = replace(result, attempts=tuple(attempts))
                if generation_attempt_context:
                    generation_attempt_context.accept(tuple(attempts))
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
                structured_tool_policy_rejection = (
                    isinstance(exc, StructuredOutputUnsupportedError)
                    and effective_tool_choice is not None
                )
                attempts.append(_attempt_record(
                    opaque_request_id, len(attempts) + 1, config, None,
                    first_route if attempt == 0 else RouteKind.RETRY,
                    AttemptOutcome.FAILURE,
                    (
                        RootCause.POLICY_REJECTION
                        if structured_tool_policy_rejection else _root_cause(exc)
                    ),
                ))
                if structured_tool_policy_rejection:
                    primary_error = StructuredOutputUnsupportedError(
                        "structured_output_unsupported"
                    )
                    break
                if isinstance(exc, NotImplementedError):
                    if generation_attempt_context:
                        generation_attempt_context.accept(tuple(attempts))
                    raise
                retry_kind = retry_kind_for_exception(exc)
                retry_index = (
                    primary_budget.generic_retry_count
                    if retry_kind is RetryKind.GENERIC
                    else primary_budget.rate_limit_retry_count
                )
                if retry_kind is not None and primary_budget.try_consume(retry_kind):
                    delay = retry_delay(
                        retry_kind,
                        RouteKind.PRIMARY,
                        retry_index,
                        retry_after=_retry_after_from_exception(exc),
                    )
                    _log_retry(
                        agent_id=agent_id,
                        task_type=task_type,
                        config=config,
                        route_kind=RouteKind.PRIMARY,
                        budget=primary_budget,
                        retry_kind=retry_kind,
                        delay=delay,
                        exc=exc,
                    )
                    time.sleep(delay)
                    attempt += 1
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
        chain_outcome = self._generate_fallback_chain(
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
            generation_attempt_context=generation_attempt_context,
            final_prompt_observer=final_prompt_observer,
            max_provider_calls=(
                None
                if max_provider_calls is None
                else max(0, max_provider_calls - primary_attempts)
            ),
        )
        if chain_outcome.result is not None:
            return chain_outcome.result
        route_failure = chain_outcome.route_failure
        fallback_error = chain_outcome.error
        if chain_outcome.sampling_audit is not None:
            last_empty_result = chain_outcome.last_empty_result
        if isinstance(fallback_error, StructuredOutputUnsupportedError):
            route_failure = "structured_output_unsupported"
        fallback_provider = None
        terminal_sampling_audit = (
            chain_outcome.sampling_audit or primary_sampling_audit
        )
        effective_temperature = (
            last_empty_result.effective_temperature
            if last_empty_result and last_empty_result.effective_temperature is not None
            else terminal_sampling_audit.effective_temperature
        )
        temperature_override_reason = (
            last_empty_result.temperature_override_reason
            if last_empty_result
            and last_empty_result.temperature_override_reason is not None
            else terminal_sampling_audit.override_reason
        )

        # Record failure
        failure_reason = _failure_reason(primary_error, fallback_error)
        failure_exception = fallback_error or primary_error
        terminal_root = (
            primary_skip_root or RootCause.POLICY_REJECTION
        )
        if generation_attempt_context is None:
            attempts.append(_attempt_record(
                opaque_request_id, len(attempts) + 1, primary_audit_config, None,
                RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE,
                terminal_root,
                terminal=True,
            ))
        failure_disposition = _failure_disposition_from_attempts(tuple(attempts))
        failure_usage = _record_failure_usage(
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
            failure_category=(
                _root_cause(failure_exception).value
                if failure_exception else None
            ),
            reasoning_level=config.reasoning_level,
            reasoning_status=(
                "requested_unconfirmed" if config.reasoning_requested
                else "not_requested"
            ),
            effective_temperature=effective_temperature,
            temperature_override_reason=temperature_override_reason,
            attempts=tuple(attempts),
        )
        # R3-MG-2: surface the HTTP status / raw error from the most recent
        # exception so the categorizer can attribute 4xx/5xx to
        # ``provider_error`` rather than the silent ``unknown`` fallback.
        empty_result = _empty_result(
            config_provider=config.provider,
            config_model=config.model,
            active_mode=active_mode.value,
            primary_error=primary_error,
            fallback_error=fallback_error,
            last_empty_result=last_empty_result,
            effective_temperature=effective_temperature,
            temperature_override_reason=temperature_override_reason,
            attempts=tuple(attempts),
            failure_disposition=failure_disposition,
        )
        if route_failure is not None:
            empty_result = replace(
                empty_result,
                structured_failure_reason=route_failure,
            )
        if empty_result.usage is None:
            empty_result = replace(empty_result, usage=failure_usage)
        if generation_attempt_context:
            generation_attempt_context.accept(tuple(attempts))
            generation_attempt_context.terminal_failure_reason = route_failure
        return empty_result

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
        generation_attempt_context: GenerationAttemptContext | None,
        final_prompt_observer: FinalPromptObserver | None,
        max_provider_calls: int | None,
    ) -> _FallbackChainOutcome:
        """按配置顺序尝试所有能力合格的 fallback 候选。"""
        profile_id = self._player_assignments.get(agent_id, "")
        plan = _resolve_fallback_routes(
            model_profiles=self._model_profiles,
            llm_profiles=self._llm_profiles,
            llm_profile_id=profile_id,
            primary_config=primary_config,
            required_reasoning_level=primary_config.reasoning_level,
        )
        if not plan.routes:
            return _FallbackChainOutcome(
                route_failure=_fallback_route_failure_code(primary_error),
            )
        current_route = primary_config
        provider_route_attempted = False
        final_fallback_error: Exception | None = None
        final_sampling_audit: SamplingAudit | None = None
        last_empty_result: GenerateResult | None = None
        provider_calls = 0
        for config in plan.routes:
            if (
                max_provider_calls is not None
                and provider_calls >= max_provider_calls
            ):
                break
            # 热更新或共享映射污染不能绕过执行前的最后一道门禁。
            if not route_switch_is_valid(current_route, config):
                return _FallbackChainOutcome(
                    route_failure=FALLBACK_ROUTE_UNAVAILABLE,
                    error=final_fallback_error,
                    sampling_audit=final_sampling_audit,
                    last_empty_result=last_empty_result,
                )
            provider = self._providers.get(config.provider)
            if provider is None:
                continue
            provider_route_attempted = True
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
            sampling_audit = resolve_sampling_audit(config)
            retry_budget = RetryBudget(
                RouteKind.PROVIDER_FALLBACK,
                int(getattr(config, "retry_count", 4) or 0),
            )
            retry_index = 0
            while True:
                if (
                    max_provider_calls is not None
                    and provider_calls >= max_provider_calls
                ):
                    break
                provider_calls += 1
                final_sampling_audit = sampling_audit
                last_empty_result = None
                route_kind = (
                    RouteKind.PROVIDER_FALLBACK
                    if retry_index == 0 else RouteKind.RETRY
                )
                try:
                    effective_choice = (
                        tool_choice if mode == StructuredOutputMode.NATIVE_TOOL else None
                    )
                    result = _call_provider_generate(
                        provider, prompt, config, system_prompt,
                        tools=tools, tool_choice=effective_choice,
                        final_prompt_observer=bind_attempt(
                            final_prompt_observer,
                            attempt_kind=route_kind.value,
                            attempt_ordinal=len(attempts) + 1,
                        ),
                    )
                    result = replace(
                    result,
                    structured_output_mode=mode.value,
                    reasoning_level=config.reasoning_level,
                    failure_disposition=FailureDisposition.NONE,
                    reasoning_status=(
                            result.reasoning_status
                            if result.reasoning_status in {"confirmed", "unsupported"}
                            else "requested_unconfirmed"
                        ),
                    )
                    result = _normalize_tool_metadata(result, effective_choice)
                    if result.reasoning_status == "unsupported":
                        attempts.append(_attempt_record(
                            request_id, len(attempts) + 1, config, result,
                            route_kind, AttemptOutcome.FAILURE,
                            RootCause.POLICY_REJECTION,
                        ))
                        final_fallback_error = RuntimeError("reasoning_unsupported")
                        break
                    if not result.text:
                        empty_error = EmptyModelResponseError("empty_response")
                        final_fallback_error = empty_error
                        attempts.append(_attempt_record(
                            request_id, len(attempts) + 1, config, result,
                            route_kind, AttemptOutcome.FAILURE,
                            RootCause.INVALID_OUTPUT,
                        ))
                        last_empty_result = replace(
                            result,
                            attempts=tuple(attempts),
                        )
                        break
                    attempts.append(_attempt_record(
                        request_id, len(attempts) + 1, config, result,
                        route_kind, AttemptOutcome.SUCCESS,
                        RootCause.NONE,
                    ))
                    if result.usage is None:
                        result = replace(result, usage=UsageRecord(
                            agent_id=agent_id, task_type=task_type,
                            provider=result.provider, model=result.model,
                            attempts=tuple(attempts),
                        ))
                    result = replace(result, attempts=tuple(attempts))
                    if generation_attempt_context:
                        generation_attempt_context.accept(tuple(attempts))
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
                    return _FallbackChainOutcome(result=result)
                except Exception as exc:
                    final_fallback_error = exc
                    structured_tool_policy_rejection = (
                        isinstance(exc, StructuredOutputUnsupportedError)
                        and effective_choice is not None
                    )
                    attempts.append(_attempt_record(
                        request_id, len(attempts) + 1, config, None,
                        route_kind, AttemptOutcome.FAILURE,
                        (
                            RootCause.POLICY_REJECTION
                            if structured_tool_policy_rejection else _root_cause(exc)
                        ),
                    ))
                    if structured_tool_policy_rejection:
                        final_fallback_error = StructuredOutputUnsupportedError(
                            "structured_output_unsupported"
                        )
                        break
                    if isinstance(exc, NotImplementedError):
                        if generation_attempt_context:
                            generation_attempt_context.accept(tuple(attempts))
                        raise
                    retry_kind = retry_kind_for_exception(exc)
                    category_retry_index = (
                        retry_budget.generic_retry_count
                        if retry_kind is RetryKind.GENERIC
                        else retry_budget.rate_limit_retry_count
                    )
                    if retry_kind is not None and retry_budget.try_consume(retry_kind):
                        delay = retry_delay(
                            retry_kind,
                            RouteKind.PROVIDER_FALLBACK,
                            category_retry_index,
                            retry_after=_retry_after_from_exception(exc),
                        )
                        _log_retry(
                            agent_id=agent_id,
                            task_type=task_type,
                            config=config,
                            route_kind=RouteKind.PROVIDER_FALLBACK,
                            budget=retry_budget,
                            retry_kind=retry_kind,
                            delay=delay,
                            exc=exc,
                        )
                        time.sleep(delay)
                        retry_index += 1
                        continue
                    break
            current_route = config
        return _FallbackChainOutcome(
            route_failure=(
                None
                if provider_route_attempted
                else _fallback_route_failure_code(primary_error)
            ),
            error=final_fallback_error,
            sampling_audit=final_sampling_audit,
            last_empty_result=last_empty_result,
        )

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
            usage=usage,
            failure_disposition=FailureDisposition.POLICY_REJECTED,
            attempts=(attempt,),
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


def _log_retry(
    *,
    agent_id: str,
    task_type: str,
    config: ModelConfig,
    route_kind: RouteKind,
    budget: RetryBudget,
    retry_kind: RetryKind,
    delay: float,
    exc: Exception,
) -> None:
    """记录候选总预算与当前类别预算，避免混合类别时分母歧义。"""
    category_retry_count = (
        budget.generic_retry_count
        if retry_kind is RetryKind.GENERIC
        else budget.rate_limit_retry_count
    )
    logger.warning(
        "Retryable model error route=%s candidate=%s/%s agent=%s task=%s "
        "provider=%s model=%s total_retry=%d/%d category=%s "
        "category_retry=%d/%d delay_seconds=%.1f: %s",
        route_kind.value,
        config.provider,
        config.model,
        agent_id,
        task_type,
        config.provider,
        config.model,
        budget.total_retry_count,
        max(budget.config_retry_count, 0),
        retry_kind.value,
        category_retry_count,
        budget.max_retries_for(retry_kind),
        delay,
        _format_exception(exc),
    )


def _retry_delay_for_exception(exc: Exception, attempt: int) -> float:
    from werewolf_agent.model_gateway.retry_policy import (
        _retry_delay_for_exception as _impl,
    )

    return _impl(exc, attempt, uniform=random.uniform)


_PRIMARY_SKIP_ROOT_CAUSES = {
    "provider_unavailable": RootCause.PROVIDER_ERROR,
    "reasoning_unsupported": RootCause.POLICY_REJECTION,
}


def _record_skipped_primary_attempt(
    *,
    attempts: list[AttemptExecutionRecord],
    request_id: OpaqueRequestId,
    config: ModelConfig,
    route_kind: RouteKind,
    reason: str,
) -> tuple[ModelConfig, RuntimeError, RootCause]:
    """记录未发生 provider 调用的脱敏 primary 执行事实。"""
    try:
        root_cause = _PRIMARY_SKIP_ROOT_CAUSES[reason]
    except KeyError as exc:
        raise ValueError("unknown primary skip reason") from exc
    effective_route = RouteKind.PRIMARY if not attempts else route_kind
    audit_config = replace(
        config,
        provider="unavailable",
        model="unavailable",
        reasoning_level="none",
        reasoning_requested=False,
        reasoning_capability="none",
    )
    attempts.append(_attempt_record(
        request_id,
        len(attempts) + 1,
        audit_config,
        None,
        effective_route,
        AttemptOutcome.FAILURE,
        root_cause,
        provider_attempted=False,
    ))
    return audit_config, RuntimeError(reason), root_cause


def _root_cause(exc: Exception) -> RootCause:
    """仅把异常归一化为审计枚举，不保留原始错误文本。"""
    if isinstance(exc, TimeoutError):
        return RootCause.TIMEOUT
    try:
        import httpx
        if isinstance(exc, httpx.TimeoutException):
            return RootCause.TIMEOUT
    except ImportError:
        pass
    if isinstance(exc, EmptyModelResponseError):
        return RootCause.INVALID_OUTPUT
    return RootCause.PROVIDER_ERROR


def _fallback_route_failure_code(primary_error: Exception | None) -> str | None:
    """内容错误保留给结构化修复；仅切换型故障报告路由不可用。"""
    if isinstance(primary_error, StructuredOutputUnsupportedError):
        return "structured_output_unsupported"
    if isinstance(primary_error, EmptyModelResponseError):
        return None
    return FALLBACK_ROUTE_UNAVAILABLE


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
    provider_attempted: bool | None = None,
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
        provider_attempted=(not terminal if provider_attempted is None else provider_attempted),
    )
