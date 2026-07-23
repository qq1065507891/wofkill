# -*- coding: utf-8 -*-
"""
每个模型配置的 base_url 与 extra_body 透传测试。

覆盖范围：
- ModelConfig 接受 base_url / extra_body 字段且 ``__hash__ = None`` 后仍可作为值对象传递。
- ``_resolve_config`` 与 ``_resolve_fallback_model`` 从 model_profile YAML 读这两个字段。
- ``OpenAIProvider``/``AnthropicProvider``/``MiniMaxProvider``/``GLMProvider``
  在 generate 时优先使用 ``config.base_url``，缺失时回退到 provider 实例默认 URL。
- ``_generate_openai_compatible`` 把 ``config.extra_body`` 合进 payload，不覆盖已有字段。
- ``OpenAIProvider`` 对 native MiniMax endpoint 按
  ``MINIMAX_NATIVE_API_KEY`` → ``MINIMAX_API_KEY`` → MiniMax-scoped
  ``ANTHROPIC_API_KEY`` 的优先级解析鉴权键；缺少合法键时 fail closed。

作者：Mike
创建日期：2026-07-15
修改日期：2026-07-23
"""

from __future__ import annotations

from dataclasses import fields

import pytest


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


class TestModelConfigFields:
    def test_base_url_defaults_to_none(self) -> None:
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        cfg = ModelConfig(provider="openai", model="x")
        assert cfg.base_url is None

    def test_extra_body_defaults_to_empty_dict(self) -> None:
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        cfg = ModelConfig(provider="openai", model="x")
        assert cfg.extra_body == {}

    def test_extra_body_default_is_independent_per_instance(self) -> None:
        """``field(default_factory=dict)`` 必须每个实例独立，否则共享可变默认是经典 bug。"""
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        cfg1 = ModelConfig(provider="openai", model="a")
        cfg2 = ModelConfig(provider="openai", model="b")
        cfg1.extra_body["reasoning_split"] = True
        assert cfg2.extra_body == {}

    def test_fields_include_base_url_and_extra_body(self) -> None:
        """显式锁住字段名，避免后续误删。"""
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        names = {f.name for f in fields(ModelConfig)}
        assert "base_url" in names
        assert "extra_body" in names

    def test_explicit_hash_is_none(self) -> None:
        """ModelConfig 含 dict 字段不可哈希，必须显式 ``__hash__ = None``。"""
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        assert ModelConfig.__hash__ is None


# ---------------------------------------------------------------------------
# _resolve_config / _resolve_fallback_model
# ---------------------------------------------------------------------------


class TestResolveConfigPlumbing:
    def test_resolve_config_picks_up_base_url(self) -> None:
        from werewolf_agent.model_gateway.router_selection import _resolve_config

        config, _fb = _resolve_config(
            model_profiles={
                "minimax_native_m3": {
                    "provider": "openai",
                    "model": "MiniMax-M3",
                    "base_url": "https://api.minimaxi.com/v1",
                },
            },
            llm_profiles={
                "native_m3": {
                    "default": {
                        "provider": "openai",
                        "model_profile": "minimax_native_m3",
                    },
                },
            },
            player_assignments={"p01": "native_m3"},
            agent_id="p01",
            task_type="speech",
        )
        assert config.base_url == "https://api.minimaxi.com/v1"
        assert config.model == "MiniMax-M3"

    def test_resolve_config_picks_up_extra_body(self) -> None:
        from werewolf_agent.model_gateway.router_selection import _resolve_config

        config, _fb = _resolve_config(
            model_profiles={
                "minimax_native_m3": {
                    "provider": "openai",
                    "model": "MiniMax-M3",
                    "extra_body": {"reasoning_split": True},
                },
            },
            llm_profiles={
                "native_m3": {
                    "default": {
                        "provider": "openai",
                        "model_profile": "minimax_native_m3",
                    },
                },
            },
            player_assignments={"p01": "native_m3"},
            agent_id="p01",
            task_type="speech",
        )
        assert config.extra_body == {"reasoning_split": True}

    def test_resolve_config_base_url_strips_trailing_slash(self) -> None:
        from werewolf_agent.model_gateway.router_selection import _resolve_config

        config, _fb = _resolve_config(
            model_profiles={
                "p": {
                    "provider": "openai",
                    "model": "x",
                    "base_url": "https://api.minimaxi.com/v1/",
                },
            },
            llm_profiles={
                "lp": {
                    "default": {"provider": "openai", "model_profile": "p"},
                },
            },
            player_assignments={"p01": "lp"},
            agent_id="p01",
            task_type="speech",
        )
        assert config.base_url == "https://api.minimaxi.com/v1"

    def test_resolve_config_base_url_non_string_falls_back_to_none(self) -> None:
        """防 YAML 误填数字 / 列表。"""
        from werewolf_agent.model_gateway.router_selection import _resolve_config

        config, _fb = _resolve_config(
            model_profiles={
                "p": {"provider": "openai", "model": "x", "base_url": 12345},
            },
            llm_profiles={
                "lp": {
                    "default": {"provider": "openai", "model_profile": "p"},
                },
            },
            player_assignments={"p01": "lp"},
            agent_id="p01",
            task_type="speech",
        )
        assert config.base_url is None

    def test_resolve_config_extra_body_non_dict_falls_back_to_empty(self) -> None:
        """防 YAML 误填字符串 / 列表。"""
        from werewolf_agent.model_gateway.router_selection import _resolve_config

        config, _fb = _resolve_config(
            model_profiles={
                "p": {
                    "provider": "openai",
                    "model": "x",
                    "extra_body": "reasoning_split",
                },
            },
            llm_profiles={
                "lp": {
                    "default": {"provider": "openai", "model_profile": "p"},
                },
            },
            player_assignments={"p01": "lp"},
            agent_id="p01",
            task_type="speech",
        )
        assert config.extra_body == {}

    def test_resolve_fallback_model_picks_up_both_fields(self) -> None:
        from werewolf_agent.model_gateway.router_selection import (
            _resolve_fallback_model,
        )

        config = _resolve_fallback_model(
            model_profiles={
                "fb": {
                    "provider": "openai",
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.minimaxi.com/v1",
                    "extra_body": {"reasoning_split": True},
                },
            },
            llm_profiles={
                "lp": {
                    "fallback": {
                        "provider": "openai",
                        "model_profile": "fb",
                    },
                },
            },
            llm_profile_id="lp",
        )
        assert config is not None
        assert config.base_url == "https://api.minimaxi.com/v1"
        assert config.extra_body == {"reasoning_split": True}


# ---------------------------------------------------------------------------
# OpenAI-compatible provider payload + URL behavior
# ---------------------------------------------------------------------------


class _CapturingClient:
    """记录最近一次 POST 的 URL、payload 与请求头。"""

    def __init__(self, response_data: dict | None = None) -> None:
        self.last_url: str | None = None
        self.last_payload: dict | None = None
        self.last_headers: dict | None = None
        self.last_timeout: int | None = None
        self._response = response_data or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def post(self, url, *, json, **kwargs):  # noqa: A002
        self.last_url = url
        self.last_payload = json
        self.last_headers = kwargs["headers"]
        self.last_timeout = kwargs["timeout"]
        return _FakeResponse(self._response)


class _FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


@pytest.mark.parametrize("provider_name", ["openai", "glm", "anthropic", "minimax"])
def test_all_provider_http_calls_receive_300_second_default_timeout(provider_name: str) -> None:
    """四类 provider 均须把统一的 ModelConfig 默认 timeout 传给 HTTP 客户端。"""
    from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
    from werewolf_agent.model_gateway.providers.glm import GLMProvider
    from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.usage_records import ModelConfig

    anthropic_compatible = provider_name in {"anthropic", "minimax"}
    client = _CapturingClient(
        {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        if anthropic_compatible else None,
    )
    provider_class = {
        "openai": OpenAIProvider,
        "glm": GLMProvider,
        "anthropic": AnthropicProvider,
        "minimax": MiniMaxProvider,
    }[provider_name]
    provider = provider_class(
        api_key="k", base_url="https://api.example/v1", http_client=client,
    )

    provider.generate("hello", ModelConfig(provider=provider_name, model="model"))

    assert client.last_timeout == 300


class TestOpenAIProviderPerProfile:
    @pytest.fixture(autouse=True)
    def _provide_native_minimax_key(self, monkeypatch: pytest.MonkeyPatch):
        from werewolf_agent.model_gateway.providers import env as env_mod

        env_mod._ENV_OVERRIDES.clear()
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "test-native-key")
        yield
        env_mod._ENV_OVERRIDES.clear()

    def test_config_base_url_overrides_provider_default(self) -> None:
        """config.base_url 应优先于 provider 实例默认 URL。"""
        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = OpenAIProvider(
            api_key="k",
            base_url="https://ark.example.com/api/coding/v3",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="openai",
                model="MiniMax-M3",
                base_url="https://api.minimaxi.com/v1",
            ),
        )
        assert client.last_url is not None
        assert client.last_url.startswith("https://api.minimaxi.com/v1")
        assert "ark.example.com" not in client.last_url

    def test_config_extra_body_merged_into_payload(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = OpenAIProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/v1",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="openai",
                model="MiniMax-M3",
                extra_body={"reasoning_split": True},
            ),
        )
        assert client.last_payload is not None
        assert client.last_payload.get("reasoning_split") is True

    def test_extra_body_setdefault_does_not_clobber_explicit_field(self) -> None:
        """reasoning_effort 已显式设置时，extra_body 里的同名 key 不应覆盖。"""
        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = OpenAIProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/v1",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="openai",
                model="MiniMax-M3",
                reasoning_level="high",
                reasoning_requested=True,
                extra_body={"reasoning_effort": "low"},
            ),
        )
        assert client.last_payload is not None
        assert client.last_payload["reasoning_effort"] == "high"

    def test_no_base_url_means_provider_default(self) -> None:
        """None base_url 时必须回退到 provider 实例默认 URL。"""
        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = OpenAIProvider(
            api_key="k",
            base_url="https://ark.example.com/api/coding/v3",
            http_client=client,
        )
        provider.generate("hi", ModelConfig(provider="openai", model="x"))
        assert client.last_url is not None
        assert "ark.example.com" in client.last_url


class TestGLMProviderPerProfile:
    """GLM 共用 _generate_openai_compatible，只验 base_url 透传。"""

    def test_config_base_url_overrides_glm_default(self) -> None:
        from werewolf_agent.model_gateway.providers.glm import GLMProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = GLMProvider(
            api_key="k",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="glm",
                model="glm-4",
                base_url="https://api.minimaxi.com/v1",
            ),
        )
        assert client.last_url is not None
        assert "api.minimaxi.com" in client.last_url
        assert "bigmodel.cn" not in client.last_url

    def test_extra_body_merged_into_payload_via_shared_helper(self) -> None:
        from werewolf_agent.model_gateway.providers.glm import GLMProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = GLMProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/v1",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="glm",
                model="glm-4",
                extra_body={"reasoning_split": True},
            ),
        )
        assert client.last_payload is not None
        assert client.last_payload.get("reasoning_split") is True


# ---------------------------------------------------------------------------
# Anthropic + MiniMax (Anthropic-compatible) providers
# ---------------------------------------------------------------------------


class TestAnthropicProviderPerProfile:
    def test_config_base_url_overrides_anthropic_default(self) -> None:
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient(
            response_data={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        provider = AnthropicProvider(
            api_key="k",
            base_url="https://api.anthropic.com",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",
                base_url="https://api.minimaxi.com/anthropic",
            ),
        )
        assert client.last_url is not None
        assert "api.minimaxi.com/anthropic" in client.last_url
        assert "api.anthropic.com" not in client.last_url

    def test_extra_body_merged_into_anthropic_payload(self) -> None:
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient(
            response_data={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        provider = AnthropicProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/anthropic",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",
                extra_body={"reasoning_split": True},
            ),
        )
        assert client.last_payload is not None
        assert client.last_payload.get("reasoning_split") is True


class TestMiniMaxAnthropicProviderPerProfile:
    def test_config_base_url_overrides_minimax_default(self) -> None:
        from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient(
            response_data={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        provider = MiniMaxProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/anthropic",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="minimax",
                model="abab-test",
                base_url="https://api.minimaxi.com/anthropic-v2",
            ),
        )
        assert client.last_url is not None
        assert "anthropic-v2" in client.last_url

    def test_extra_body_merged_into_minimax_payload(self) -> None:
        from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient(
            response_data={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        provider = MiniMaxProvider(
            api_key="k",
            base_url="https://api.minimaxi.com/anthropic",
            http_client=client,
        )
        provider.generate(
            "hi",
            ModelConfig(
                provider="minimax",
                model="abab-test",
                extra_body={"reasoning_split": True},
            ),
        )
        assert client.last_payload is not None
        assert client.last_payload.get("reasoning_split") is True


# ---------------------------------------------------------------------------
# Per-call API key routing (2026-07-16)
# ---------------------------------------------------------------------------


class TestNativeMiniMaxApiKeyRouting:
    """Native MiniMax 只接受专用、厂商或 scoped Anthropic 鉴权键。"""

    @pytest.fixture(autouse=True)
    def _isolate_minimax_key_environment(self, monkeypatch: pytest.MonkeyPatch):
        from werewolf_agent.model_gateway.providers import env as env_mod

        env_mod._ENV_OVERRIDES.clear()
        for key in (
            "MINIMAX_NATIVE_API_KEY",
            "MINIMAX_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
        ):
            monkeypatch.delenv(key, raising=False)
        yield
        env_mod._ENV_OVERRIDES.clear()

    @staticmethod
    def _generate(
        client: _CapturingClient,
        *,
        provider_base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
        config_base_url: str | None = "https://api.minimaxi.com/v1",
    ) -> None:
        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        OpenAIProvider(
            api_key="ark-key-should-not-leak",
            base_url=provider_base_url,
            http_client=client,
        ).generate(
            "hi",
            ModelConfig(
                provider="openai",
                model="MiniMax-M3",
                base_url=config_base_url,
            ),
        )

    def test_native_endpoint_uses_minimax_native_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "dedicated-key")
        monkeypatch.setenv("MINIMAX_API_KEY", "vendor-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
        client = _CapturingClient()
        self._generate(client)
        assert client.last_headers == {
            "Authorization": "Bearer dedicated-key",
            "content-type": "application/json",
        }

    def test_vendor_key_outranks_scoped_anthropic_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "vendor-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
        client = _CapturingClient()
        self._generate(client)
        assert client.last_headers is not None
        assert client.last_headers["Authorization"] == "Bearer vendor-key"

    def test_scoped_anthropic_key_is_reused_when_native_keys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
        client = _CapturingClient()
        self._generate(client)
        assert client.last_headers is not None
        assert client.last_headers["Authorization"] == "Bearer anthropic-key"

    def test_native_endpoint_without_minimax_key_fails_before_post(self) -> None:
        from werewolf_agent.model_gateway.providers.base import ProviderConfigError

        client = _CapturingClient()
        with pytest.raises(ProviderConfigError) as exc_info:
            self._generate(client)
        message = str(exc_info.value)
        assert "MINIMAX_NATIVE_API_KEY" in message
        assert "MINIMAX_API_KEY" in message
        assert "ANTHROPIC_API_KEY" in message
        assert "ANTHROPIC_BASE_URL" in message
        assert "api.minimaxi.com" in message
        assert client.last_url is None

    def test_provider_default_minimax_url_uses_native_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "dedicated-key")
        client = _CapturingClient()
        self._generate(
            client,
            provider_base_url="https://api.minimaxi.com/v1",
            config_base_url=None,
        )
        assert client.last_headers is not None
        assert client.last_headers["Authorization"] == "Bearer dedicated-key"

    def test_lookalike_hostname_uses_default_key(self) -> None:
        client = _CapturingClient()
        self._generate(client, config_base_url="https://api.minimaxi.com.evil.example/v1")
        assert client.last_headers is not None
        assert client.last_headers["Authorization"] == "Bearer ark-key-should-not-leak"

    def test_non_https_native_minimax_url_fails_before_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.model_gateway.providers.base import ProviderConfigError

        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "dedicated-key")
        client = _CapturingClient()
        with pytest.raises(ProviderConfigError):
            self._generate(client, config_base_url="http://api.minimaxi.com/v1")
        assert client.last_url is None

    def test_unscoped_anthropic_key_is_not_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.model_gateway.providers.base import ProviderConfigError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        client = _CapturingClient()
        with pytest.raises(ProviderConfigError):
            self._generate(client)
        assert client.last_url is None

    def test_ark_endpoint_uses_default_key_even_with_minimax_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "dedicated-key")
        client = _CapturingClient()
        self._generate(
            client,
            config_base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        )
        assert client.last_headers is not None
        assert client.last_headers["Authorization"] == "Bearer ark-key-should-not-leak"
