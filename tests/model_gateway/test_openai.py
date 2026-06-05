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

    def test_url_normalization_v1beta_falls_back_to_v1(self) -> None:
        """R3-MG-10: a base URL ending in /v1beta must NOT trigger the
        /chat/completions append (which would 404). It should fall back
        to the default /v1/chat/completions.
        """
        url = _build_url("https://example.com/v1beta")
        assert url == "https://example.com/v1/chat/completions"

    def test_url_normalization_v2beta_falls_back_to_v1(self) -> None:
        url = _build_url("https://example.com/v2beta")
        assert url == "https://example.com/v1/chat/completions"

    def test_url_normalization_trailing_slash(self) -> None:
        url = _build_url("https://example.com/v1/")
        assert url == "https://example.com/v1/chat/completions"

    def test_url_normalization_zhipu_default_paas_v4(self) -> None:
        """GLMProvider's default base https://open.bigmodel.cn/api/paas/v4
        must continue to work. The new whitelister accepts /v4."""
        url = _build_url("https://open.bigmodel.cn/api/paas/v4")
        assert url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
