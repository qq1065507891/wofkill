"""Tests for OpenAI-compatible URL normalization in OpenAIProvider.

R3-MG-10: the legacy helper accepted any URL whose last path segment
contained the substring ``/v`` (e.g. ``/v1beta``, ``/v2``, ``/v1.5``)
and appended ``/chat/completions`` directly, producing 404s for
real OpenAI-compatible gateways that use non-/v1 API versions.

The new rule whitelists the bare ``/v1`` (and integer ``/vN``)
endings, and falls back to ``/v1/chat/completions`` for everything
else.
"""

import pytest


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }


class _Client:
    def __init__(self):
        self.payload = None

    def post(self, _url, *, json, **_kwargs):
        self.payload = json
        return _Response()


def _build_url(base_url: str) -> str:
    """Mirror the production helper exactly."""
    from werewolf_agent.model_gateway.providers.openai import (
        _openai_chat_completions_url,
    )
    return _openai_chat_completions_url(base_url)


class TestOpenAIUrlNormalization:
    def test_default_openai_host_appends_v1_chat_completions(self) -> None:
        """The default api.openai.com base must end up at /v1/chat/completions."""
        url = _build_url("https://api.openai.com")
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_explicit_v1_base_appends_chat_completions(self) -> None:
        url = _build_url("https://example.com/v1")
        assert url == "https://example.com/v1/chat/completions"

    def test_v2_base_appends_chat_completions(self) -> None:
        """/v2 is whitelisted; the helper appends /chat/completions directly."""
        url = _build_url("https://example.com/v2")
        assert url == "https://example.com/v2/chat/completions"

    def test_url_normalization_v1beta_preserved(self) -> None:
        """N2 (post-review-v2): a base URL ending in /v1beta must be
        preserved as-is, not stripped and re-routed to /v1/chat/completions.

        Rationale: real OpenAI-compatible gateways (Zhipu, Baidu, etc.)
        use ``/v1beta`` as a stable API version path. The legacy
        "fall back to /v1" behavior was a 404 source for these
        gateways. The new rule preserves ANY ``/vN`` / ``/vNbeta`` /
        ``/vNalpha`` suffix the caller declared.
        """
        url = _build_url("https://example.com/v1beta")
        assert "v1beta" in url, f"v1beta lost: {url}"
        assert "/v1/chat/completions" not in url, f"v1beta misrouted: {url}"
        assert url == "https://example.com/v1beta/chat/completions"

    def test_url_normalization_v2beta_preserved(self) -> None:
        url = _build_url("https://example.com/v2beta")
        assert "v2beta" in url, f"v2beta lost: {url}"
        assert url == "https://example.com/v2beta/chat/completions"

    def test_v4beta_url_preserved(self) -> None:
        """N2 (post-review-v2): /v4beta /v5beta URLs must NOT be misrouted
        to /v1/chat/completions (the legacy fallback dropped the version
        and produced 404s on real OpenAI-compatible gateways that use
        versioned paths like /v4beta)."""
        url = _build_url("https://api.example.com/v4beta")
        assert "v4beta" in url, f"v4beta lost: {url}"
        assert "/v1/chat/completions" not in url, f"v4beta misrouted: {url}"
        assert url == "https://api.example.com/v4beta/chat/completions"

    def test_v5alpha_url_preserved(self) -> None:
        """N2 (post-review-v2): /v5alpha is also a valid versioned path."""
        url = _build_url("https://api.example.com/v5alpha")
        assert "v5alpha" in url, f"v5alpha lost: {url}"
        assert url == "https://api.example.com/v5alpha/chat/completions"

    def test_url_normalization_trailing_slash(self) -> None:
        url = _build_url("https://example.com/v1/")
        assert url == "https://example.com/v1/chat/completions"

    def test_url_normalization_zhipu_default_paas_v4(self) -> None:
        """GLMProvider's default base https://open.bigmodel.cn/api/paas/v4
        must continue to work. The new whitelister accepts /v4."""
        url = _build_url("https://open.bigmodel.cn/api/paas/v4")
        assert url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def test_openai_provider_sends_reasoning_effort_and_records_reasoning_tokens():
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _Client()
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://ark.example/v3",
        http_client=client,
    )
    result = provider.generate(
        "analyze",
        ModelConfig(
            provider="openai",
            model="deepseek-v4-pro",
            max_tokens=128,
            reasoning_level="high",
            reasoning_requested=True,
        ),
    )

    assert client.payload["reasoning_effort"] == "high"
    assert client.payload["max_completion_tokens"] == 128
    assert "max_tokens" not in client.payload
    assert result.reasoning_status == "confirmed"
    assert result.reasoning_tokens == 3


# 2026-07-21 R6: OpenAI Chat Completions prompt cache 解析.
# OpenAI 自动 cache prompt 前缀, 返回 usage.prompt_tokens_details.cached_tokens
# 表示命中 token 数. 不解析会导致 UsageRecord.cache_read_input_tokens 永远是 0
# (R2 anthropic 已经解析, OpenAI/GLM 没有).


class _CachedResponse:
    """2026-07-21 R6: 模拟 OpenAI 返回 cached_tokens."""

    status_code = 200

    def __init__(self, cached_tokens: int | None):
        self.cached_tokens = cached_tokens

    def raise_for_status(self):
        return None

    def json(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        if self.cached_tokens is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": usage,
        }


class _CachedClient:
    def __init__(self, cached_tokens: int | None):
        self.cached_tokens = cached_tokens
        self.last_json = None

    def post(self, url, *, json=None, **_):
        self.last_json = json
        return _CachedResponse(self.cached_tokens)


def test_openai_usage_parses_cached_tokens():
    """R6: cached_tokens 必须写入 UsageRecord.cache_read_input_tokens."""
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _CachedClient(cached_tokens=800)
    result = OpenAIProvider(
        api_key="k", base_url="https://api.example.com/v1", http_client=client,
    ).generate(
        prompt="hi",
        config=ModelConfig(provider="openai", model="gpt-test"),
    )

    assert result.usage is not None
    assert result.usage.cache_read_input_tokens == 800, (
        f"cached_tokens=800 应写入 UsageRecord, 实测 {result.usage.cache_read_input_tokens}"
    )
    # OpenAI 是 server-side auto cache, 没有 creation 概念.
    assert result.usage.cache_creation_input_tokens == 0


def test_openai_usage_missing_prompt_tokens_details_safe():
    """R6: 老 vendor 无 prompt_tokens_details 字段时, cached_tokens 默认 0 不崩."""
    from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _CachedClient(cached_tokens=None)
    result = OpenAIProvider(
        api_key="k", base_url="https://api.example.com/v1", http_client=client,
    ).generate(
        prompt="hi",
        config=ModelConfig(provider="openai", model="gpt-test"),
    )

    assert result.usage is not None
    assert result.usage.cache_read_input_tokens == 0
    assert result.usage.cache_creation_input_tokens == 0
