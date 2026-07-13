# -*- coding: utf-8 -*-
"""
验证 persona 证明来自 provider 最终 HTTP payload，而不是请求层预组装快照。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from typing import Any


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
        self.response_payload = response_payload

    def post(self, _url: str, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
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
    assert assemblies[0].system_bytes == client.payload["system"].encode("utf-8")
    assert assemblies[0].final_system_location == "system"
    assert assemblies[0].final_system_message_index is None


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
    assert assemblies[1].system_bytes == fallback_client.payload["system"].encode("utf-8")
    assert assemblies[1].final_system_location == "system"
    assert assemblies[1].final_system_message_index is None
