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
