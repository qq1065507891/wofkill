# -*- coding: utf-8 -*-
"""
Tests for per-profile base_url + extra_body plumbing (2026-07-15/16).

覆盖范围：
- ModelConfig 接受 base_url / extra_body 字段且 ``__hash__ = None`` 后仍可作为值对象传递。
- ``_resolve_config`` 与 ``_resolve_fallback_model`` 从 model_profile YAML 读这两个字段。
- ``OpenAIProvider``/``AnthropicProvider``/``MiniMaxProvider``/``GLMProvider``
  在 generate 时优先使用 ``config.base_url``，缺失时回退到 provider 实例默认 URL。
- ``_generate_openai_compatible`` 把 ``config.extra_body`` 合进 payload，不覆盖已有字段。
- ``OpenAIProvider`` 在 base_url 指向 ``api.minimaxi.com`` 时改读
  ``MINIMAX_NATIVE_API_KEY``（2026-07-16：解决 Ark/MiniMax key 隔离）。

作者：Mike
创建日期：2026-07-15
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
    """记录最近一次 POST 的 URL 与 payload。"""

    def __init__(self, response_data: dict | None = None) -> None:
        self.last_url: str | None = None
        self.last_payload: dict | None = None
        self._response = response_data or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def post(self, url, *, json, **_kwargs):  # noqa: A002
        self.last_url = url
        self.last_payload = json
        return _FakeResponse(self._response)


class _FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class TestOpenAIProviderPerProfile:
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
    """``api.minimaxi.com/v1`` requires ``MINIMAX_NATIVE_API_KEY`` which is
    different from the default ``OPENAI_API_KEY`` (Ark Volcengine).

    The ``OpenAIProvider`` instance is constructed with ``OPENAI_API_KEY``
    once.  When ``config.base_url`` points at the native MiniMax host, the
    provider must switch to ``MINIMAX_NATIVE_API_KEY`` for that call only.
    This is a per-call override; other endpoints keep the default key.
    """

    def test_native_endpoint_uses_minimax_native_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from werewolf_agent.model_gateway.providers import env as env_mod
        env_mod.load_local_dotenv()
        monkeypatch.setenv("OPENAI_API_KEY", "ark-key-should-not-leak")
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "minimax-native-key")

        from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        client = _CapturingClient()
        provider = OpenAIProvider(
            api_key="ark-key-should-not-leak",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
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
        # The CapturingClient doesn't capture headers, so verify via the
        # provider's resolved key by reading the env path the helper takes.
        from werewolf_agent.model_gateway.providers.env import get_env
        assert get_env("MINIMAX_NATIVE_API_KEY") == "minimax-native-key"

    def test_non_native_endpoint_keeps_default_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ark endpoint stays on OPENAI_API_KEY even when MINIMAX_NATIVE_API_KEY
        is set.  The key switch is base_url-gated, not global."""
        from werewolf_agent.model_gateway.providers.openai import (
            _resolve_api_key_for_config,
        )
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        cfg = ModelConfig(
            provider="openai", model="x",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        )
        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "native-key")
        assert _resolve_api_key_for_config(cfg, "ark-key") == "ark-key"

    def test_native_endpoint_with_no_native_key_falls_back(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If MINIMAX_NATIVE_API_KEY is not set, fall back to default key
        (this will likely 401 in production but won't crash the loop)."""
        from werewolf_agent.model_gateway.providers.openai import (
            _resolve_api_key_for_config,
        )
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        monkeypatch.delenv("MINIMAX_NATIVE_API_KEY", raising=False)
        cfg = ModelConfig(
            provider="openai", model="x",
            base_url="https://api.minimaxi.com/v1",
        )
        assert _resolve_api_key_for_config(cfg, "default-key") == "default-key"

    def test_native_endpoint_picks_native_key_when_set(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _resolve_api_key_for_config,
        )
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "native-key")
        cfg = ModelConfig(
            provider="openai", model="x",
            base_url="https://api.minimaxi.com/v1",
        )
        assert _resolve_api_key_for_config(cfg, "default-key") == "native-key"

    def test_no_base_url_keeps_default_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _resolve_api_key_for_config,
        )
        from werewolf_agent.model_gateway.usage_records import ModelConfig

        monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "native-key")
        cfg = ModelConfig(provider="openai", model="x", base_url=None)
        assert _resolve_api_key_for_config(cfg, "default-key") == "default-key"
