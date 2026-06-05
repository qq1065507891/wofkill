"""Model Router Gateway: provider-agnostic LLM routing.

All agents call ModelRouter.generate() — never provider SDKs directly.
Supports per-player llm_profile, per-task model selection, fallback chains,
retry, timeout, and cost tracking. No hardcoded API keys.
"""

from __future__ import annotations

import logging
import inspect
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

logger = logging.getLogger(__name__)


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
    allow_text_tool_fallback: bool = False
    retry_count: int = 2


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
    tool_call_required: bool = False
    tool_call_received: bool = False
    tool_call_name: str = ""
    text_fallback_used: bool = False
    structured_failure_reason: str | None = None
    allow_text_tool_fallback: bool = False
    # R3-MG-2: surface raw HTTP status / error string from the provider so
    # the failure categorizer can attribute 4xx/5xx to ``provider_error``
    # instead of falling through to ``unknown``. Defaults preserve the
    # pre-R3-MG-2 behavior for callers that do not populate them.
    http_status: int = 0
    raw_error: str | None = None


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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
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
        """Register configured providers that have API keys in env/.env."""
        from werewolf_agent.model_gateway.providers import create_provider_from_env

        provider_names = self._configured_provider_names()
        for provider_name in provider_names:
            provider = create_provider_from_env(provider_name)
            if provider is not None:
                self.register_provider(provider)

    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def probe_tool_call_support(self, agent_id: str, task_type: str) -> dict[str, Any]:
        """Probe whether the resolved provider returns an actual tool call."""
        config, _fallback_provider = self.resolve_config(agent_id, task_type)
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

        config = ModelConfig(
            provider=provider_name,
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.5),
            max_tokens=model_profile.get("max_tokens", 1024),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 30),
            allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
            retry_count=int(model_profile.get("retry_count", 2)),
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        jitter_seconds: tuple[float, float] = (0.0, 0.8),
    ) -> GenerateResult:
        """Generate via routed provider with fallback.

        ``jitter_seconds`` is the (low, high) range of a uniform random
        sleep applied before the FIRST attempt only. Defaults to
        ``(0, 0.8)`` to spread concurrent requests. Pass ``(0, 0)`` in
        tests to avoid 5-15s of cumulative wait across 12 players.
        """
        config, fallback_provider = self.resolve_config(agent_id, task_type)

        provider = self._providers.get(config.provider)
        if provider is None:
            raise RuntimeError(f"Provider '{config.provider}' not found. Available: {list(self._providers.keys())}")

        primary_error: Exception | None = None
        fallback_error: Exception | None = None

        max_retries = getattr(config, "retry_count", 2) or 0
        for attempt in range(max_retries + 1):
            # Pre-call jitter: spread concurrent requests to avoid rate-limiting.
            # On the first attempt only — retries already have backoff.
            # R3-MG-3: skip entirely when jitter_seconds == (0, 0).
            if attempt == 0 and jitter_seconds != (0.0, 0.0):
                time.sleep(random.uniform(jitter_seconds[0], jitter_seconds[1]))
            try:
                # For text-fallback models, skip tool_choice on first attempt.
                # These models rarely return tool_calls reliably; forcing it
                # wastes retries. Let them respond with free-form text/JSON.
                effective_tool_choice = tool_choice
                if config.allow_text_tool_fallback and tool_choice and attempt == 0:
                    effective_tool_choice = None
                result = _call_provider_generate(
                    provider,
                    prompt,
                    config,
                    system_prompt,
                    tools=tools,
                    tool_choice=effective_tool_choice,
                )
                result.allow_text_tool_fallback = config.allow_text_tool_fallback
                _normalize_tool_metadata(result, tool_choice)
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
            fb_max_retries = getattr(fb_config, "retry_count", 1) or 0
            for fb_attempt in range(fb_max_retries + 1):
                try:
                    fb_effective_tool_choice = tool_choice
                    if fb_config.allow_text_tool_fallback and tool_choice and fb_attempt == 0:
                        fb_effective_tool_choice = None
                    result = _call_provider_generate(
                        fb_provider,
                        prompt,
                        fb_config,
                        system_prompt,
                        tools=tools,
                        tool_choice=fb_effective_tool_choice,
                    )
                    result.allow_text_tool_fallback = fb_config.allow_text_tool_fallback
                    _normalize_tool_metadata(result, tool_choice)
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
            ))
            if len(self._usage_log) > 10000:
                self._usage_log = self._usage_log[-5000:]
        # R3-MG-2: surface the HTTP status / raw error from the most recent
        # exception so the categorizer can attribute 4xx/5xx to
        # ``provider_error`` rather than the silent ``unknown`` fallback.
        return GenerateResult(
            text="",
            provider=config.provider,
            model=config.model,
            http_status=_http_status_from_exception(primary_error)
            or _http_status_from_exception(fallback_error),
            raw_error=_raw_error_from_exception(primary_error)
            or _raw_error_from_exception(fallback_error),
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
        return ModelConfig(
            provider=fallback_cfg.get("provider", "mock"),
            model=model_profile.get("model", model_profile_id),
            temperature=model_profile.get("temperature", 0.3),
            max_tokens=model_profile.get("max_tokens", 256),
            top_p=model_profile.get("top_p", 0.9),
            timeout=model_profile.get("timeout", 10),
            allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
            retry_count=int(model_profile.get("retry_count", 1)),
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


def _format_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown"
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _http_status_from_exception(exc: BaseException | None) -> int:
    """R3-MG-2: best-effort HTTP status extraction from an exception.

    Returns 0 when the exception does not carry an HTTP status (e.g. a
    bare ``RuntimeError`` from a SDK or a connection-refused ``OSError``).
    """
    if exc is None:
        return 0
    # httpx.HTTPStatusError and httpx.HTTPError variants expose .response
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return int(getattr(exc.response, "status_code", 0) or 0)
        response = getattr(exc, "response", None)
        if response is not None:
            return int(getattr(response, "status_code", 0) or 0)
    except ImportError:
        pass
    # Generic: look for an integer "code" attribute or HTTP NNN in str(exc).
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 100 <= code <= 599:
        return code
    import re
    m = re.search(r"\b([1-5]\d{2})\b", str(exc))
    if m:
        return int(m.group(1))
    return 0


def _raw_error_from_exception(exc: BaseException | None) -> str | None:
    """R3-MG-2: best-effort raw error string from an exception."""
    if exc is None:
        return None
    message = str(exc)
    return message or None


def _failure_reason(
    primary_error: BaseException | None,
    fallback_error: BaseException | None,
) -> str:
    reason = f"primary_failed:{_format_exception(primary_error)}"
    if fallback_error is not None:
        reason += f"; fallback_failed:{_format_exception(fallback_error)}"
    return reason


def _call_provider_generate(
    provider: LLMProvider,
    prompt: str,
    config: ModelConfig,
    system_prompt: str | None,
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | None,
) -> GenerateResult:
    signature = inspect.signature(provider.generate)
    if "tools" in signature.parameters:
        return provider.generate(
            prompt,
            config,
            system_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
    return provider.generate(prompt, config, system_prompt)


def _normalize_tool_metadata(
    result: GenerateResult,
    tool_choice: dict[str, Any] | str | None,
) -> None:
    if not tool_choice:
        return
    # If tool_choice is "auto", LLM may freely choose text-only response
    tc_type = tool_choice if isinstance(tool_choice, str) else tool_choice.get("type", "")
    if tc_type == "auto":
        return
    # If provider already signaled text fallback is acceptable, don't
    # override it — some providers (MiniMax, certain Baidu models) cannot
    # reliably produce tool_use blocks even when tool_choice is sent.
    if result.allow_text_tool_fallback and result.text:
        result.tool_call_required = True
        if not result.tool_call_name:
            result.tool_call_name = str(tool_choice.get("name") or "") if isinstance(tool_choice, dict) else ""
        result.text_fallback_used = True
        return
    result.tool_call_required = True
    if not result.tool_call_name:
        result.tool_call_name = str(tool_choice.get("name") or "") if isinstance(tool_choice, dict) else ""
    if not result.tool_call_received:
        result.text_fallback_used = bool(result.text)
        if result.structured_failure_reason is None:
            result.structured_failure_reason = "missing_tool_call"


def _is_retryable_exception(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    exc_str = type(exc).__name__.lower()
    if "connect" in exc_str or "timeout" in exc_str:
        return True
    # httpx exceptions
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500 or exc.response.status_code == 429
    except ImportError:
        pass
    # Generic checks via string matching for provider-agnostic errors
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg:
        return True
    if "503" in msg or "service unavailable" in msg:
        return True
    if "529" in msg or "overloaded" in msg:
        return True
    return False


def _retry_delay_for_exception(exc: Exception, attempt: int) -> float:
    """Exponential backoff with jitter and 429 Retry-After support.

    Base delay = 2^attempt seconds, capped at 60 s, with +-25 % random
    jitter to spread out concurrent retries and avoid thundering-herd
    rate-limiting.
    """
    base = 2.0
    # Check for Retry-After header on 429
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                return min(float(retry_after), 30.0)
    except (ImportError, ValueError, TypeError):
        pass
    # Exponential backoff with jitter: 1.0/2.0/4.0/8.0/... capped at 60 s
    raw = min(base ** attempt, 60.0)
    jitter = raw * random.uniform(-0.25, 0.25)
    return max(0.5, raw + jitter)
