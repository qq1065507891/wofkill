# -*- coding: utf-8 -*-
"""
验证 persona 证明来自 provider 最终 HTTP payload，而不是请求层预组装快照。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from typing import Any

import pytest


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload or {
            "choices": [{"message": {"content": "ok"}}],
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class _Client:
    def __init__(self, response_payload: dict[str, Any] | None = None) -> None:
        self.payload: dict[str, Any] | None = None
        self.post_calls = 0
        self.response_payload = response_payload

    def post(self, _url: str, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
        self.post_calls += 1
        self.payload = json
        return _Response(self.response_payload)


def test_openai_observer_receives_actual_final_messages_system_bytes() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    assemblies: list[FinalPromptAssembly] = []
    OpenAIProvider(api_key="k", base_url="https://example.test/v1", http_client=client).generate(
        "user", ModelConfig(provider="openai", model="m"),
        system_prompt="rules\npersona-final", final_prompt_observer=assemblies.append,
    )

    assert client.payload is not None
    actual_system = client.payload["messages"][0]["content"]
    assert len(assemblies) == 1
    assert assemblies[0].system_bytes == actual_system.encode("utf-8")
    assert assemblies[0].final_system_location == "messages"
    assert assemblies[0].final_system_message_index == 0


def test_openai_observer_receives_canonical_full_provider_payload() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        canonical_provider_payload,
    )
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    observed = []
    OpenAIProvider(api_key="k", base_url="https://example.test/v1", http_client=client).generate(
        "dynamic live_status and stance source_event_id=e-1",
        ModelConfig(provider="openai", model="m"),
        system_prompt="rules",
        final_prompt_observer=observed.append,
    )

    assert client.payload is not None
    assert observed[0].provider_payload_bytes == canonical_provider_payload(client.payload)
    assert b"dynamic live_status" in observed[0].provider_payload_bytes
    assert b"source_event_id=e-1" in observed[0].provider_payload_bytes


def test_anthropic_observer_reports_top_level_system_with_no_message_index() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    assemblies: list[FinalPromptAssembly] = []
    AnthropicProvider(api_key="k", base_url="https://example.test", http_client=client).generate(
        "user", ModelConfig(provider="anthropic", model="m"),
        system_prompt="rules\npersona-final", final_prompt_observer=assemblies.append,
    )

    assert client.payload is not None
    # 2026-07-21 R2: anthropic system 字段在 system_prompt 非空时升级为 list 形式
    # (cache_control), observer 把 list JSON 化成 system_bytes. str 形态走旧契约.
    payload_system = client.payload["system"]
    if isinstance(payload_system, list):
        import json
        expected = json.dumps(payload_system, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        expected = payload_system.encode("utf-8")
    assert assemblies[0].system_bytes == expected
    assert assemblies[0].final_system_location == "system"
    assert assemblies[0].final_system_message_index is None


def test_observer_exception_does_not_block_provider_http_request() -> None:
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()

    def broken_observer(_assembly: object) -> None:
        raise RuntimeError("private observer details")

    result = OpenAIProvider(
        api_key="k",
        base_url="https://example.test/v1",
        http_client=client,
    ).generate(
        "user",
        ModelConfig(provider="openai", model="m"),
        system_prompt="rules\npersona-final",
        final_prompt_observer=broken_observer,
    )

    assert result.text == "ok"
    assert client.post_calls == 1


def test_contract_failure_blocks_provider_before_http_request() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptContract,
        FinalPromptContractError,
        validate_final_prompt_contract,
    )
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    contract = FinalPromptContract(
        contract_id="player-system",
        version="2026-07-18",
        required_sections=(("wolf_semantics", b"WOLF-SEMANTICS"),),
    )

    with pytest.raises(FinalPromptContractError, match="wolf_semantics"):
        OpenAIProvider(
            api_key="k",
            base_url="https://example.test/v1",
            http_client=client,
        ).generate(
            "user",
            ModelConfig(provider="openai", model="m"),
            system_prompt="rules without required block",
            final_prompt_observer=lambda assembly: validate_final_prompt_contract(
                assembly,
                contract,
            ),
        )

    assert client.post_calls == 0


@pytest.mark.parametrize(
    "system_prompt",
    (
        "HEADER\nCLAUSE-A",
        "HEADER\nCLAUSE-A\nHEADER\nCLAUSE-B",
        "CLAUSE-A\nHEADER\nCLAUSE-B",
    ),
)
def test_contract_rejects_header_only_duplicate_or_reordered_sections_before_http(
    system_prompt: str,
) -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptContract,
        FinalPromptContractError,
        validate_final_prompt_contract,
    )
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    contract = FinalPromptContract(
        contract_id="strict",
        version="v1",
        required_sections=(
            ("header", b"HEADER"),
            ("critical_clause_a", b"CLAUSE-A"),
            ("critical_clause_b", b"CLAUSE-B"),
        ),
    )

    with pytest.raises(FinalPromptContractError):
        OpenAIProvider(
            api_key="k", base_url="https://example.test/v1", http_client=client,
        ).generate(
            "user", ModelConfig(provider="openai", model="m"),
            system_prompt=system_prompt,
            final_prompt_observer=lambda assembly: validate_final_prompt_contract(
                assembly, contract,
            ),
        )

    assert client.post_calls == 0


def test_router_does_not_record_observer_exception_as_provider_failure() -> None:
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelRouter

    client = _Client()
    router = ModelRouter(
        model_profiles={
            "primary": {
                "provider": "openai",
                "model": "p",
                "retry_count": 1,
                "reasoning": {"level": "high"},
            },
        },
        llm_profiles={
            "default": {
                "default": {"provider": "openai", "model_profile": "primary"},
            },
        },
        player_assignments={"p01": "default"},
        providers={
            "openai": OpenAIProvider(
                api_key="k",
                base_url="https://openai.test/v1",
                http_client=client,
            ),
        },
        allow_test_model_capability=True,
    )

    result = router.generate(
        "p01",
        "speech",
        "user",
        "rules\npersona-final",
        jitter_seconds=(0, 0),
        final_prompt_observer=lambda _assembly: (_ for _ in ()).throw(
            RuntimeError("private observer details")
        ),
    )

    assert result.text == "ok"
    assert client.post_calls == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].attempt_outcome.value == "attempt_success"
    assert result.attempts[0].root_cause.value == "none"
    assert result.usage is not None
    assert result.usage.retry_count == 0


def test_router_observes_each_real_provider_assembly_in_fallback_chain() -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelRouter

    primary_client = _Client({
        "choices": [{"message": {"content": ""}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    })
    fallback_client = _Client({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })

    router = ModelRouter(
        model_profiles={
            "primary": {
                "provider": "openai", "model": "p", "retry_count": 0,
                "reasoning": {"level": "high"},
            },
            "fallback": {
                "provider": "anthropic", "model": "f", "retry_count": 0,
                "reasoning": {"level": "high"},
            },
        },
        llm_profiles={"default": {
            "default": {"provider": "openai", "model_profile": "primary"},
            "fallback": {"provider": "anthropic", "model_profile": "fallback"},
        }},
        player_assignments={"p01": "default"},
        providers={
            "openai": OpenAIProvider(
                api_key="k", base_url="https://openai.test/v1", http_client=primary_client,
            ),
            "anthropic": AnthropicProvider(
                api_key="k", base_url="https://anthropic.test", http_client=fallback_client,
            ),
        },
        allow_test_model_capability=True,
    )
    assemblies: list[FinalPromptAssembly] = []

    result = router.generate(
        "p01", "speech", "user", "rules\npersona-final",
        jitter_seconds=(0, 0), final_prompt_observer=assemblies.append,
    )

    assert result.text == "ok"
    assert [item.provider for item in assemblies] == ["openai", "anthropic"]
    assert [item.attempt_kind for item in assemblies] == ["primary", "provider_fallback"]
    assert [item.attempt_ordinal for item in assemblies] == [1, 2]
    assert primary_client.payload is not None
    assert fallback_client.payload is not None
    assert assemblies[0].system_bytes == primary_client.payload["messages"][0]["content"].encode("utf-8")
    # 2026-07-21 R2: 见上一个测试的同款兼容：anthropic 在 system_prompt 非空时
    # 走 list 形态; fallback 链路下来时同样要按 list/str 自适应.
    fallback_system = fallback_client.payload["system"]
    if isinstance(fallback_system, list):
        import json
        expected = json.dumps(fallback_system, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        expected = fallback_system.encode("utf-8")
    assert assemblies[1].system_bytes == expected
    assert assemblies[1].final_system_location == "system"
    assert assemblies[1].final_system_message_index is None
