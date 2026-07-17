# -*- coding: utf-8 -*-
"""
模型路由器拆分模块的兼容导入测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/model_gateway/test_router_split_helpers.py
"""

from __future__ import annotations

import inspect

import pytest

from werewolf_agent.model_gateway import provider_call
from werewolf_agent.model_gateway import retry_policy
from werewolf_agent.model_gateway import router
from werewolf_agent.model_gateway import router_config
from werewolf_agent.model_gateway import router_errors
from werewolf_agent.model_gateway import router_probe
from werewolf_agent.model_gateway import router_selection
from werewolf_agent.model_gateway import usage_records


def test_usage_record_legacy_fields_remain_readable() -> None:
    usage = usage_records.UsageRecord(
        agent_id="p01",
        task_type="vote",
        provider="mock",
        model="mock",
        fallback_reason="legacy",
        retry_count=1,
    )

    assert usage.fallback_reason == "legacy"
    assert usage.retry_count == 1


def test_usage_models_are_reexported_from_router_facade() -> None:
    assert router.ModelConfig is usage_records.ModelConfig
    assert router.UsageRecord is usage_records.UsageRecord
    assert router.GenerateResult is usage_records.GenerateResult
    assert router.EmptyModelResponseError is usage_records.EmptyModelResponseError
    assert router.LLMProvider is usage_records.LLMProvider
    assert router.MockProvider is usage_records.MockProvider


def test_provider_call_helpers_are_reexported_from_router_facade() -> None:
    assert router._call_provider_generate is provider_call._call_provider_generate
    assert router._normalize_tool_metadata is provider_call._normalize_tool_metadata


def test_tool_metadata_normalization_replaces_frozen_result() -> None:
    original = usage_records.GenerateResult(
        text="fallback text",
        provider="legacy",
        model="legacy-v1",
        allow_text_tool_fallback=True,
    )

    normalized = provider_call._normalize_tool_metadata(
        original,
        {"type": "tool", "name": "submit_player_action"},
    )

    assert original.tool_call_required is False
    assert normalized.tool_call_required is True
    assert normalized.tool_call_name == "submit_player_action"
    assert normalized.text_fallback_used is True


def test_retry_helpers_are_reexported_from_router_facade() -> None:
    assert router._format_exception is retry_policy._format_exception
    assert router._http_status_from_exception is retry_policy._http_status_from_exception
    assert router._raw_error_from_exception is retry_policy._raw_error_from_exception
    assert router._failure_reason is retry_policy._failure_reason
    assert router._is_retryable_exception is retry_policy._is_retryable_exception


def test_retry_delay_facade_uses_router_random_for_compatibility(monkeypatch) -> None:
    monkeypatch.setattr(router.random, "uniform", lambda low, high: high)

    delay = router._retry_delay_for_exception(RuntimeError("test"), 2)

    assert delay == 5.0


def test_config_helpers_are_reexported_from_router_facade() -> None:
    assert router._validate_config is router_config._validate_config
    assert router._configured_provider_names is router_config._configured_provider_names


def test_selection_helpers_are_reexported_from_router_facade() -> None:
    assert router._resolve_config is router_selection._resolve_config
    assert router._resolve_fallback_model is router_selection._resolve_fallback_model


def test_error_helpers_are_reexported_from_router_facade() -> None:
    assert router._record_success_usage is router_errors._record_success_usage
    assert router._record_failure_usage is router_errors._record_failure_usage
    assert router._empty_result is router_errors._empty_result


def test_probe_helper_is_reexported_from_router_facade() -> None:
    assert router.probe_tool_call_support is router_probe.probe_tool_call_support


def test_router_never_emits_deprecated_requested_not_confirmed_status() -> None:
    producers = (router, provider_call, usage_records, router_errors)
    assert all(
        "requested_not_confirmed" not in inspect.getsource(module)
        for module in producers
    )


@pytest.mark.parametrize(
    ("signature_kind", "expected_keys"),
    [
        ("legacy", set()),
        ("tools", {"tools", "tool_choice"}),
        ("observer", {"final_prompt_observer"}),
        ("both", {"tools", "tool_choice", "final_prompt_observer"}),
        ("kwargs", {"tools", "tool_choice", "final_prompt_observer"}),
    ],
)
def test_provider_call_adapts_tools_and_observer_independently(
    signature_kind: str,
    expected_keys: set[str],
) -> None:
    captured: dict[str, object] = {}

    class _LegacyProvider:
        def generate(self, prompt, config, system_prompt=None):
            return usage_records.GenerateResult(text="ok", provider="test", model="m")

    class _ToolsProvider:
        def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
            captured.update(tools=tools, tool_choice=tool_choice)
            return usage_records.GenerateResult(text="ok", provider="test", model="m")

    class _ObserverProvider:
        def generate(self, prompt, config, system_prompt=None, final_prompt_observer=None):
            captured["final_prompt_observer"] = final_prompt_observer
            return usage_records.GenerateResult(text="ok", provider="test", model="m")

    class _BothProvider:
        def generate(
            self, prompt, config, system_prompt=None, tools=None, tool_choice=None,
            final_prompt_observer=None,
        ):
            captured.update(
                tools=tools,
                tool_choice=tool_choice,
                final_prompt_observer=final_prompt_observer,
            )
            return usage_records.GenerateResult(text="ok", provider="test", model="m")

    class _KwargsProvider:
        def generate(self, prompt, config, system_prompt=None, **kwargs):
            captured.update(kwargs)
            return usage_records.GenerateResult(text="ok", provider="test", model="m")

    provider_types = {
        "legacy": _LegacyProvider,
        "tools": _ToolsProvider,
        "observer": _ObserverProvider,
        "both": _BothProvider,
        "kwargs": _KwargsProvider,
    }
    observer = (
        None if signature_kind in {"legacy", "tools"}
        else lambda _assembly: None
    )
    result = provider_call._call_provider_generate(
        provider_types[signature_kind](),
        "prompt",
        usage_records.ModelConfig(provider="test", model="m"),
        "system",
        tools=[{"name": "submit"}],
        tool_choice={"type": "tool", "name": "submit"},
        final_prompt_observer=observer,
    )

    assert result.text == "ok"
    assert set(captured) == expected_keys
    if "final_prompt_observer" in expected_keys:
        assert captured["final_prompt_observer"] is observer


def test_legacy_provider_without_observer_fails_closed_before_generate() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        RouterPromptContractCompatibilityError,
    )

    calls = 0

    class _LegacyProvider:
        def generate(self, prompt, config, system_prompt=None):
            nonlocal calls
            calls += 1
            return usage_records.GenerateResult(text="bad", provider="test", model="m")

    with pytest.raises(RouterPromptContractCompatibilityError):
        provider_call._call_provider_generate(
            _LegacyProvider(),
            "prompt",
            usage_records.ModelConfig(provider="test", model="m"),
            "system",
            tools=None,
            tool_choice=None,
            final_prompt_observer=lambda _assembly: None,
        )

    assert calls == 0
