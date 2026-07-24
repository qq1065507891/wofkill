# -*- coding: utf-8 -*-
"""
验证 persona 证明来自 provider 最终 HTTP payload，而不是请求层预组装快照。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-24
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


def _assert_type_sensitive_equal(actual: object, expected: object) -> None:
    """递归比较值与容器类型，避免 bool 与 int 等宽松相等。"""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_type_sensitive_equal(actual[key], expected_value)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_type_sensitive_equal(actual_item, expected_item)
        return
    assert actual == expected


def _anthropic_compatible_provider(provider_name: str, client: _Client) -> Any:
    """用同一入口构造两种 Anthropic 兼容真实 provider。"""
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider

    provider_type = {
        "anthropic": AnthropicProvider,
        "minimax": MiniMaxProvider,
    }[provider_name]
    return provider_type(
        api_key="k",
        base_url="https://example.test",
        http_client=client,
    )


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
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptAssembly,
        canonical_provider_payload,
    )
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    assemblies: list[FinalPromptAssembly] = []
    system_prompt = '你是"守夜人"。\n保留路径 C:\\wolves\\alpha\n第二行含有"引号"与\\反斜杠。'
    AnthropicProvider(api_key="k", base_url="https://example.test", http_client=client).generate(
        "user", ModelConfig(provider="anthropic", model="m"),
        system_prompt=system_prompt, final_prompt_observer=assemblies.append,
    )

    assert client.payload is not None
    expected_system = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]
    _assert_type_sensitive_equal(client.payload["system"], expected_system)
    assert assemblies[0].system_bytes == system_prompt.encode("utf-8")
    assert assemblies[0].provider_payload_bytes == canonical_provider_payload(
        client.payload,
    )
    assert assemblies[0].final_system_location == "system"
    assert assemblies[0].final_system_message_index is None


def test_anthropic_system_block_text_is_concatenated_without_separator() -> None:
    from werewolf_agent.model_gateway.providers.anthropic import (
        _system_bytes_for_observer,
    )

    blocks = [
        {
            "type": "text",
            "text": '你是"守夜人"。\n路径 C:\\wolves\\alpha\n',
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": '继续保留\\与"引号"。'},
    ]

    assert _system_bytes_for_observer(blocks) == (
        '你是"守夜人"。\n路径 C:\\wolves\\alpha\n继续保留\\与"引号"。'
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("system_value", "expected"),
    (
        (None, b""),
        ("规则\n角色", "规则\n角色".encode("utf-8")),
    ),
)
def test_anthropic_system_scalar_observer_bytes(
    system_value: str | None,
    expected: bytes,
) -> None:
    from werewolf_agent.model_gateway.providers.anthropic import (
        _system_bytes_for_observer,
    )

    assert _system_bytes_for_observer(system_value) == expected


@pytest.mark.parametrize(
    "system_value",
    (
        ["not-a-block"],
        ({"type": "text", "text": "tuple block"},),
        {"type": "text", "text": "mapping container"},
        42,
        [{"text": "missing type"}],
        [{"type": "image", "text": "not text"}],
        [{"type": "text"}],
        [{"type": "text", "text": 1}],
        [
            {"type": "text", "text": "valid prefix"},
            {"type": "image", "text": "invalid suffix"},
        ],
    ),
)
def test_anthropic_system_blocks_fail_closed_as_a_whole(
    system_value: object,
) -> None:
    from werewolf_agent.model_gateway.providers.anthropic import (
        _system_bytes_for_observer,
    )

    assert _system_bytes_for_observer(system_value) == b""  # type: ignore[arg-type]


@pytest.mark.parametrize("provider_name", ("anthropic", "minimax"))
def test_anthropic_compatible_contract_accepts_exact_multiline_persona(
    provider_name: str,
) -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptAssembly,
        FinalPromptContract,
        canonical_provider_payload,
        validate_final_prompt_contract,
    )
    from werewolf_agent.model_gateway.router import ModelConfig

    persona = '【角色】\n你是"守夜人"。\n路径 C:\\wolves\\alpha\n保留\\与"引号"。'
    tools = [{
        "name": "submit_player_action",
        "input_schema": {"type": "object"},
    }]
    tool_choice = {"type": "tool", "name": "submit_player_action"}
    client = _Client()
    provider = _anthropic_compatible_provider(provider_name, client)
    assemblies: list[FinalPromptAssembly] = []
    contract = FinalPromptContract(
        contract_id="anthropic-compatible-persona",
        version="v1",
        required_sections=(("persona", persona.encode("utf-8")),),
    )

    def observe(assembly: FinalPromptAssembly) -> None:
        validate_final_prompt_contract(assembly, contract)
        assemblies.append(assembly)

    provider.generate(
        "user",
        ModelConfig(
            provider=provider_name,
            model="m",
            max_tokens=321,
            temperature=0.25,
            top_p=0.75,
        ),
        system_prompt=persona,
        tools=tools,
        tool_choice=tool_choice,
        final_prompt_observer=observe,
    )

    expected_payload: dict[str, Any] = {
        "model": "m",
        "temperature": 0.25,
        "top_p": 0.75,
        "messages": [{"role": "user", "content": "user"}],
        "max_tokens": 321,
        "system": [{
            "type": "text",
            "text": persona,
            "cache_control": {"type": "ephemeral"},
        }],
        "tools": [{
            "name": "submit_player_action",
            "input_schema": {"type": "object"},
        }],
    }
    if provider_name == "anthropic":
        expected_payload["tool_choice"] = {
            "type": "tool",
            "name": "submit_player_action",
        }

    assert client.payload is not None
    _assert_type_sensitive_equal(client.payload, expected_payload)
    assert len(assemblies) == 1
    assert assemblies[0].provider_payload_bytes == canonical_provider_payload(
        expected_payload,
    )
    assert client.post_calls == 1


@pytest.mark.parametrize("provider_name", ("anthropic", "minimax"))
def test_anthropic_compatible_contract_rejects_truly_missing_persona(
    provider_name: str,
) -> None:
    from werewolf_agent.model_gateway.final_prompt_observer import (
        FinalPromptContract,
        FinalPromptContractError,
        validate_final_prompt_contract,
    )
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    provider = _anthropic_compatible_provider(provider_name, client)
    contract = FinalPromptContract(
        contract_id="anthropic-compatible-persona",
        version="v1",
        required_sections=(("persona", "【角色】守夜人".encode("utf-8")),),
    )

    with pytest.raises(FinalPromptContractError, match="persona"):
        provider.generate(
            "user",
            ModelConfig(provider=provider_name, model="m"),
            system_prompt="只有通用规则，未包含角色。",
            final_prompt_observer=lambda assembly: validate_final_prompt_contract(
                assembly,
                contract,
            ),
        )

    assert client.post_calls == 0


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
    assert assemblies[1].system_bytes == b"rules\npersona-final"
    assert assemblies[1].final_system_location == "system"
    assert assemblies[1].final_system_message_index is None
