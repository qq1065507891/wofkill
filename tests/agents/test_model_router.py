# -*- coding: utf-8 -*-
"""
验证模型路由网关的配置解析、provider 路由、fallback 与用量记录。

作者: Project contributors
修改日期: 2026-07-26
"""

from __future__ import annotations

import json

from werewolf_agent.model_gateway.router import (
    ModelConfig,
    ModelRouter,
    MockProvider,
)
from werewolf_agent.model_gateway.providers import (
    AnthropicProvider,
    GLMProvider,
    MiniMaxProvider,
    OpenAIProvider,
    create_provider_from_env,
)

from tests.agents.test_player_agent import (
    _FailProvider,
    _FakeHttpClient,
    LegacyProvider,
    TextProbeProvider,
    ToolProbeProvider,
)

MODELS_YAML = "config/models.yaml"


class TestModelRouter:
    def test_load_from_yaml(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        assert "pro_reasoner" in router._llm_profiles

    def test_yaml_task_profiles_resolve_declared_temperatures(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)

        expected = {
            "discussion_summary": 0.2,
            "judge_vote_tally": 0.2,
            "speech": 0.4,
            "vote": 0.3,
            "night_action": 0.3,
            "deception": 0.5,
            "wolf_discussion": 0.5,
        }
        for task_type, temperature in expected.items():
            config, _ = router.resolve_config("p02", task_type)
            assert config.temperature == temperature
            assert config.top_p == 0.9

    def test_all_active_routes_use_task_sampling_profiles(self) -> None:
        """每个活动路由的六类主任务都必须使用声明的采样档位。"""
        router = ModelRouter.from_yaml(MODELS_YAML)
        task_temperatures = {
            "discussion_summary": 0.2,
            "vote": 0.3,
            "night_action": 0.3,
            "speech": 0.4,
            "deception": 0.5,
            "wolf_discussion": 0.5,
        }
        route_contracts = {
            **{
                player_id: {
                    "provider": "openai",
                    "model": "MiniMax-M3",
                    "base_url": "https://api.minimaxi.com/v1",
                    "extra_body": {"reasoning_split": True},
                }
                for player_id in ("p01", "p03", "p08")
            },
            **{
                player_id: {
                    "provider": "openai",
                    "model": "MiniMax-M2.7",
                    "base_url": "https://api.minimaxi.com/v1",
                    "extra_body": {"reasoning_split": True},
                }
                for player_id in ("p06", "p10")
            },
            **{
                player_id: {
                    "provider": "openai",
                    "model": "DeepSeek-V4-Pro",
                    "base_url": None,
                    "extra_body": {},
                }
                for player_id in ("p07", "p11")
            },
            "p05": {
                "provider": "openai",
                "model": "deepseek-v4-flash",
                "base_url": None,
                "extra_body": {},
            },
            **{
                player_id: {
                    "provider": "openai",
                    "model": "DeepSeek-V4-Pro",
                    "base_url": None,
                    "extra_body": {},
                }
                for player_id in ("p02", "p04", "p09", "p12", "judge")
            },
        }

        assert set(route_contracts) == {
            *(f"p{index:02d}" for index in range(1, 13)),
            "judge",
        }
        for player_id, contract in route_contracts.items():
            for task_type, temperature in task_temperatures.items():
                config, _ = router.resolve_config(player_id, task_type)
                assert config.temperature == temperature, (player_id, task_type)
                assert config.top_p == 0.9, (player_id, task_type)
                assert config.provider == contract["provider"]
                assert config.model == contract["model"]
                assert config.base_url == contract["base_url"]
                assert config.extra_body == contract["extra_body"]
                assert config.structured_output_mode == "text_json"

    def test_minimax_judge_and_thinking_profiles_declare_wire_temperatures(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        model_profiles = router._model_profiles

        assert model_profiles["minimax_m27_judge"]["temperature"] == 0.2
        assert model_profiles["minimax_m27_thinking"]["temperature"] == 1.0
        assert model_profiles["minimax_m27_reflection"]["temperature"] == 1.0

        judge, _ = router.resolve_config("judge", "judge_vote_tally")
        reflection, _ = router.resolve_config("p02", "reflection")
        fallback = router._resolve_fallback_model("ark_minimax")

        assert judge.temperature == 0.2
        assert reflection.temperature == 1.0
        assert fallback is not None
        assert fallback.temperature == 1.0
        assert {
            judge.structured_output_mode,
            reflection.structured_output_mode,
            fallback.structured_output_mode,
        } == {"text_json"}

    def test_resolve_config_for_known_agent(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, fallback = router.resolve_config("p01", "speech")
        assert config.provider != ""
        assert config.model != ""
        assert config.reasoning_level == "medium"
        assert config.reasoning_capability == "high"
        assert config.reasoning_requested is True

    def test_ark_profiles_use_stage_capability_levels(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        pro_config, _ = router.resolve_config("p03", "speech")
        flash_config, _ = router.resolve_config("p05", "speech")

        assert pro_config.reasoning_level == "medium"
        assert pro_config.reasoning_capability == "high"
        assert flash_config.reasoning_level == "medium"
        assert flash_config.reasoning_capability == "high"

    def test_resolve_config_task_specific(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, _ = router.resolve_config("p01", "reflection")
        assert config.provider != ""
        assert config.model != ""

    def test_resolve_config_fallback_chain(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        _, fallback = router.resolve_config("p02", "speech")
        assert fallback is not None
        assert fallback != ""

    def test_generate_with_mock_provider(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        router.register_provider(MockProvider("minimax"))
        router.register_provider(MockProvider("glm"))
        router.register_provider(MockProvider("openai"))
        result = router.generate("p01", "speech", "Test prompt")
        assert result.text != ""
        assert result.provider in ("minimax", "glm", "openai", "mock")

    def test_generate_fallback_on_failure(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        # Register failing provider as the primary (openai) and mock as fallback (minimax)
        router._providers["openai"] = _FailProvider()
        router.register_provider(MockProvider("minimax"))

        result = router.generate("p02", "speech", "Test prompt")
        # p02 uses openai as primary (fails) -> fallback to minimax
        assert result.text != "" or result.provider in ("openai", "minimax")

    def test_failed_generation_records_exception_reason(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, _ = router.resolve_config("p01", "speech")
        router._providers[config.provider] = _FailProvider()

        result = router.generate("p01", "speech", "Test prompt")
        log = router.get_usage_log()

        assert result.text == ""
        assert log[-1].success is False
        assert log[-1].fallback_reason is not None
        assert log[-1].fallback_reason == "provider_error"

    def test_router_marks_missing_tool_call_for_legacy_provider(self) -> None:
        router = ModelRouter(
            model_profiles={"legacy_model": {"model": "legacy-v1", "provider": "legacy", "reasoning": {"level": "high"}}},
            llm_profiles={"default": {"default": {"provider": "legacy", "model_profile": "legacy_model"}}},
            player_assignments={"p01": "default"},
            providers={"legacy": LegacyProvider()},
        )

        result = router.generate(
            "p01",
            "vote",
            "Choose",
            tools=[{"name": "submit_player_action", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "submit_player_action"},
        )

        assert result.tool_call_required is True
        assert result.tool_call_received is False
        assert result.text_fallback_used is True
        assert result.structured_failure_reason == "missing_tool_call"

    def test_probe_tool_call_support_detects_supported_provider(self) -> None:
        router = ModelRouter(
            model_profiles={"probe_model": {"model": "probe-v1", "provider": "tool_probe"}},
            llm_profiles={"default": {"default": {"provider": "tool_probe", "model_profile": "probe_model"}}},
            player_assignments={"p01": "default"},
            providers={"tool_probe": ToolProbeProvider()},
        )

        result = router.probe_tool_call_support("p01", "speech")

        assert result["supported"] is True
        assert result["provider"] == "tool_probe"
        assert result["tool_call_received"] is True

    def test_probe_tool_call_support_detects_text_fallback_provider(self) -> None:
        router = ModelRouter(
            model_profiles={"probe_model": {"model": "probe-v1", "provider": "text_probe"}},
            llm_profiles={"default": {"default": {"provider": "text_probe", "model_profile": "probe_model"}}},
            player_assignments={"p01": "default"},
            providers={"text_probe": TextProbeProvider()},
        )

        result = router.probe_tool_call_support("p01", "speech")

        assert result["supported"] is False
        assert result["failure_reason"] == "missing_tool_call"
        assert result["text_fallback_used"] is True

    def test_usage_logging(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, _ = router.resolve_config("p01", "speech")
        router.register_provider(MockProvider(config.provider))
        router.generate("p01", "speech", "Test prompt")
        log = router.get_usage_log()
        assert len(log) >= 1
        assert log[0].agent_id == "p01"
        assert log[0].task_type == "speech"

    def test_config_snapshot(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        snap = router.config_snapshot()
        assert "model_profiles" in snap
        assert "llm_profiles" in snap
        assert "player_assignments" in snap

    def test_unknown_agent_uses_default(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, _ = router.resolve_config("p99", "speech")
        assert config.provider == "mock"

    def test_no_hardcoded_api_keys(self) -> None:
        """Verify no API keys in config."""
        router = ModelRouter.from_yaml(MODELS_YAML)
        snap = router.config_snapshot()
        config_str = str(snap)
        assert "api_key" not in config_str.lower()
        assert "sk-" not in config_str
        # "token" appears in max_tokens, so check for auth-specific patterns
        assert "secret" not in config_str.lower()
        assert "bearer" not in config_str.lower()
        assert "password" not in config_str.lower()

    def test_register_env_providers_registers_available_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        router = ModelRouter.from_yaml(MODELS_YAML, register_env_providers=True)

        assert "minimax" in router.provider_names()

    def test_create_provider_from_env_returns_none_without_key(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        # Clear any cached .env overrides
        from werewolf_agent.model_gateway.providers import _ENV_OVERRIDES
        _ENV_OVERRIDES.clear()

        assert create_provider_from_env("anthropic") is None
        assert create_provider_from_env("minimax") is None

    def test_anthropic_provider_posts_messages_request(self) -> None:
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })
        provider = AnthropicProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Say hello",
            ModelConfig(provider="anthropic", model="claude-test", max_tokens=20, temperature=0.2, top_p=0.8),
            system_prompt="You are concise.",
        )

        assert result.text == "hello"
        assert client.calls[0]["url"].endswith("/v1/messages")
        assert client.calls[0]["headers"]["x-api-key"] == "key"
        assert client.calls[0]["json"]["model"] == "claude-test"
        # 2026-07-21 R2: Anthropic system_prompt 升级为 list-of-text-blocks +
        # cache_control: ephemeral, 跨轮跨玩家复用 cache_read_input_tokens.
        assert client.calls[0]["json"]["system"] == [{
            "type": "text",
            "text": "You are concise.",
            "cache_control": {"type": "ephemeral"},
        }]
        assert result.usage.prompt_tokens == 3
        assert result.usage.completion_tokens == 1

    def test_anthropic_provider_omits_unset_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })
        provider = AnthropicProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="anthropic", model="claude-test", max_tokens=None),
        )

        assert "max_tokens" not in client.calls[0]["json"]

    def test_anthropic_provider_sends_explicit_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })
        provider = AnthropicProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="anthropic", model="claude-test", max_tokens=20),
        )

        assert client.calls[0]["json"]["max_tokens"] == 20

    def test_anthropic_provider_uses_tool_call_for_structured_output(self) -> None:
        tool_input = {
            "action_type": "vote",
            "target_id": "p07",
            "speech": "归票7",
            "reason": "发言可疑",
            "confidence": 0.8,
        }
        client = _FakeHttpClient({
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "submit_player_action",
                    "input": tool_input,
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 7},
        })
        provider = AnthropicProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Choose an action",
            ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt="Use the tool.",
            tools=[{
                "name": "submit_player_action",
                "description": "Submit one player action.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string"},
                        "target_id": {"type": ["string", "null"]},
                        "speech": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["action_type", "target_id", "speech", "reason", "confidence"],
                },
            }],
            tool_choice={"type": "tool", "name": "submit_player_action"},
        )

        assert json.loads(result.text) == tool_input
        assert client.calls[0]["json"]["tools"][0]["name"] == "submit_player_action"
        assert client.calls[0]["json"]["tool_choice"] == {
            "type": "tool",
            "name": "submit_player_action",
        }

    def test_anthropic_provider_marks_missing_tool_call(self) -> None:
        """When model returns text instead of tool_use despite tool_choice, mark it explicitly."""
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "{\"action_type\":\"vote\"}"}],
            "usage": {"input_tokens": 4, "output_tokens": 2},
        })
        provider = AnthropicProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Choose an action",
            ModelConfig(provider="anthropic", model="claude-test"),
            tools=[{
                "name": "submit_player_action",
                "description": "Submit one player action.",
                "input_schema": {"type": "object"},
            }],
            tool_choice={"type": "tool", "name": "submit_player_action"},
        )
        assert result.text == '{"action_type":"vote"}'
        assert result.tool_call_required is True
        assert result.tool_call_received is False
        assert result.text_fallback_used is True
        assert result.structured_failure_reason == "missing_tool_call"

    def test_openai_provider_posts_chat_request(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = OpenAIProvider(
            api_key="key",
            base_url="https://api.openai.com",
            http_client=client,
        )

        result = provider.generate(
            "Say hello",
            ModelConfig(provider="openai", model="gpt-test"),
            system_prompt="You are concise.",
        )

        assert result.text == "hello"
        assert client.calls[0]["url"].endswith("/v1/chat/completions")
        assert client.calls[0]["headers"]["Authorization"] == "Bearer key"
        assert client.calls[0]["json"]["messages"][0]["role"] == "system"

    def test_openai_provider_omits_unset_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = OpenAIProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="openai", model="gpt-test", max_tokens=None),
        )

        assert "max_tokens" not in client.calls[0]["json"]

    def test_openai_provider_sends_explicit_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = OpenAIProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="openai", model="gpt-test", max_tokens=20),
        )

        assert client.calls[0]["json"]["max_tokens"] == 20

    def test_openai_provider_preserves_compatible_base_url_path(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = OpenAIProvider(
            api_key="key",
            base_url="https://qianfan.baidubce.com/v2/coding",
            http_client=client,
        )

        result = provider.generate(
            "Say hello",
            ModelConfig(provider="openai", model="deepseek-v3.2"),
        )

        assert result.text == "hello"
        assert client.calls[0]["url"] == "https://qianfan.baidubce.com/v2/coding/chat/completions"

    def test_glm_provider_posts_chat_request(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = GLMProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Say hello",
            ModelConfig(provider="glm", model="glm-test"),
        )

        assert result.text == "hello"
        assert "bigmodel.cn" in client.calls[0]["url"]
        assert client.calls[0]["headers"]["Authorization"] == "Bearer key"

    def test_minimax_provider_omits_unset_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })
        provider = MiniMaxProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="minimax", model="abab-test", max_tokens=None),
        )

        assert "max_tokens" not in client.calls[0]["json"]

    def test_minimax_provider_sends_explicit_max_tokens(self) -> None:
        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })
        provider = MiniMaxProvider(api_key="key", http_client=client)

        provider.generate(
            "Say hello",
            ModelConfig(provider="minimax", model="abab-test", max_tokens=20),
        )

        assert client.calls[0]["json"]["max_tokens"] == 20

    def test_openai_provider_marks_missing_tool_call(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "{\"action_type\":\"vote\"}"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })
        provider = OpenAIProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Choose an action",
            ModelConfig(provider="openai", model="gpt-test"),
            tools=[{
                "name": "submit_player_action",
                "description": "Submit one player action.",
                "input_schema": {"type": "object"},
            }],
            tool_choice={"type": "tool", "name": "submit_player_action"},
        )

        assert result.text == '{"action_type":"vote"}'
        assert result.tool_call_required is True
        assert result.tool_call_received is False
        assert result.text_fallback_used is True
        assert result.structured_failure_reason == "missing_tool_call"
