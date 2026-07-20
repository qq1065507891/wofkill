"""Unit tests for model gateway providers."""

import os
import tempfile
from pathlib import Path

import pytest


class TestGetEnv:
    def test_returns_os_environ_value_when_no_override(self) -> None:
        from werewolf_agent.model_gateway.providers import get_env
        os.environ["_TEST_KEY_"] = "from_os"
        try:
            assert get_env("_TEST_KEY_") == "from_os"
        finally:
            del os.environ["_TEST_KEY_"]

    def test_returns_default_when_not_set(self) -> None:
        from werewolf_agent.model_gateway.providers import get_env
        assert get_env("_NONEXISTENT_KEY_12345_", "fallback") == "fallback"

    def test_empty_default_when_not_set(self) -> None:
        from werewolf_agent.model_gateway.providers import get_env
        assert get_env("_NONEXISTENT_KEY_12345_") == ""


class TestLoadLocalDotenv:
    def test_loads_allowed_keys_from_dotenv_file(self) -> None:
        from werewolf_agent.model_gateway.providers import (
            load_local_dotenv,
            get_env,
            _ENV_OVERRIDES,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8",
        ) as f:
            f.write('ANTHROPIC_API_KEY="test-key-123"\n')
            f.write("# comment line\n")
            f.write('GLM_API_KEY=glm-key-456\n')
            f.write("INVALID_KEY=should-not-load\n")
            tmp = f.name

        try:
            _ENV_OVERRIDES.clear()
            load_local_dotenv(tmp)
            assert get_env("ANTHROPIC_API_KEY") == "test-key-123"
            assert get_env("GLM_API_KEY") == "glm-key-456"
            assert get_env("INVALID_KEY", "missing") == "missing"
        finally:
            _ENV_OVERRIDES.clear()
            Path(tmp).unlink()

    def test_skips_missing_file(self) -> None:
        from werewolf_agent.model_gateway.providers import (
            load_local_dotenv,
            _ENV_OVERRIDES,
        )
        _ENV_OVERRIDES.clear()
        load_local_dotenv("/nonexistent/path/.env")
        assert len(_ENV_OVERRIDES) == 0

    def test_skips_malformed_lines(self) -> None:
        from werewolf_agent.model_gateway.providers import (
            load_local_dotenv,
            _ENV_OVERRIDES,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8",
        ) as f:
            f.write("NO_EQUALS\n")
            f.write("=empty_key\n")
            f.write("  \n")
            tmp = f.name

        try:
            _ENV_OVERRIDES.clear()
            load_local_dotenv(tmp)
            assert len(_ENV_OVERRIDES) == 0
        finally:
            _ENV_OVERRIDES.clear()
            Path(tmp).unlink()

    def test_strips_quotes_from_value(self) -> None:
        from werewolf_agent.model_gateway.providers import (
            load_local_dotenv,
            get_env,
            _ENV_OVERRIDES,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8",
        ) as f:
            f.write('ANTHROPIC_API_KEY="double-quoted"\n')
            f.write("GLM_API_KEY='single-quoted'\n")
            tmp = f.name

        try:
            _ENV_OVERRIDES.clear()
            load_local_dotenv(tmp)
            assert get_env("ANTHROPIC_API_KEY") == "double-quoted"
            assert get_env("GLM_API_KEY") == "single-quoted"
        finally:
            _ENV_OVERRIDES.clear()
            Path(tmp).unlink()


class TestCreateProviderFromEnv:
    def test_returns_none_when_dotenv_disabled(self, monkeypatch) -> None:
        from werewolf_agent.model_gateway.providers import (
            create_provider_from_env,
            load_local_dotenv,
            _ENV_OVERRIDES,
        )
        _ENV_OVERRIDES.clear()
        monkeypatch.setattr(
            "werewolf_agent.model_gateway.providers.env.load_local_dotenv",
            lambda *a, **kw: None,
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        assert create_provider_from_env("anthropic") is None

    def test_creates_anthropic_with_key(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from werewolf_agent.model_gateway.providers import (
            create_provider_from_env,
            _ENV_OVERRIDES,
        )
        _ENV_OVERRIDES.clear()
        provider = create_provider_from_env("anthropic")
        assert provider is not None
        assert provider.name == "anthropic"

    def test_unknown_provider_returns_none(self) -> None:
        from werewolf_agent.model_gateway.providers import create_provider_from_env
        assert create_provider_from_env("nonexistent") is None


class TestProviderConstruction:
    def test_anthropic_provider_name(self) -> None:
        from werewolf_agent.model_gateway.providers import AnthropicProvider
        p = AnthropicProvider(api_key="k")
        assert p.name == "anthropic"

    def test_openai_provider_name(self) -> None:
        from werewolf_agent.model_gateway.providers import OpenAIProvider
        p = OpenAIProvider(api_key="k")
        assert p.name == "openai"

    def test_glm_provider_name(self) -> None:
        from werewolf_agent.model_gateway.providers import GLMProvider
        p = GLMProvider(api_key="k")
        assert p.name == "glm"

    def test_minimax_provider_name(self) -> None:
        from werewolf_agent.model_gateway.providers import MiniMaxProvider
        p = MiniMaxProvider(api_key="k")
        assert p.name == "minimax"


class TestResponseParsing:
    def test_extract_anthropic_text_finds_text_block(self) -> None:
        from werewolf_agent.model_gateway.providers import _extract_anthropic_text
        data = {"content": [{"type": "text", "text": "Hello werewolf"}]}
        assert "Hello werewolf" in _extract_anthropic_text(data)

    def test_extract_anthropic_text_includes_tool_use_input(self) -> None:
        from werewolf_agent.model_gateway.providers import _extract_anthropic_text
        data = {
            "content": [
                {"type": "tool_use", "name": "vote", "input": {"target": "p03"}},
            ],
        }
        text = _extract_anthropic_text(data)
        assert "p03" in text

    def test_extract_anthropic_text_returns_empty_for_empty_content(self) -> None:
        from werewolf_agent.model_gateway.providers import _extract_anthropic_text
        assert _extract_anthropic_text({"content": []}) == ""

    def test_has_anthropic_tool_use_detects_tool(self) -> None:
        from werewolf_agent.model_gateway.providers import _has_anthropic_tool_use
        assert _has_anthropic_tool_use({
            "content": [{"type": "tool_use", "name": "vote", "input": {}}],
        }) is True

    def test_has_anthropic_tool_use_no_tool_when_text_only(self) -> None:
        from werewolf_agent.model_gateway.providers import _has_anthropic_tool_use
        assert _has_anthropic_tool_use({
            "content": [{"type": "text", "text": "ok"}],
        }) is False

    def test_anthropic_tool_name(self) -> None:
        from werewolf_agent.model_gateway.providers import _anthropic_tool_name
        assert _anthropic_tool_name({
            "content": [{"type": "tool_use", "name": "vote", "input": {}}],
        }) == "vote"


class TestProviderConfigError:
    def test_provider_config_error_is_runtime_error(self) -> None:
        from werewolf_agent.model_gateway.providers import ProviderConfigError
        assert issubclass(ProviderConfigError, RuntimeError)


class TestStripThinkingPrefix:
    def test_removes_think_tags(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        content = "<think>The user asks 1+1=?</think>\n\n1+1=2。"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == "1+1=2。"
        assert "1+1=?" in thinking

    def test_no_think_tags_is_noop(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        content = "1+1=2。"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == "1+1=2。"
        assert thinking == ""

    def test_multiple_think_tags_concatenated(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        content = "<think>step1</think>\n<think>step2</think>\nanswer"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == "answer"
        assert "step1" in thinking
        assert "step2" in thinking

    def test_whitespace_inside_think_tag(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        content = "< think >reasoning< / think >\nanswer"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == "answer"
        assert "reasoning" in thinking

    def test_ark_models_already_clean(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        # Ark 端点: content 纯答案, reasoning_content 独立
        content = "1+1等于2。"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == "1+1等于2。"
        assert thinking == ""

    def test_empty_content_returns_empty(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        clean, thinking = _strip_thinking_prefix("")
        assert clean == ""
        assert thinking == ""

    def test_only_think_tag_returns_empty_clean(self) -> None:
        from werewolf_agent.model_gateway.providers.openai import (
            _strip_thinking_prefix,
        )
        content = "<think>just thinking, no output</think>"
        clean, thinking = _strip_thinking_prefix(content)
        assert clean == ""
        assert "just thinking" in thinking


class TestGenerateResultThinkingText:
    def test_thinking_text_defaults_to_empty_string(self) -> None:
        from werewolf_agent.model_gateway.usage_records import GenerateResult
        result = GenerateResult(text="{}", provider="openai", model="test")
        assert result.thinking_text == ""

    def test_thinking_text_preserves_explicit_value(self) -> None:
        from werewolf_agent.model_gateway.usage_records import GenerateResult
        result = GenerateResult(
            text="{}", provider="openai", model="test",
            thinking_text="推理链内容",
        )
        assert result.thinking_text == "推理链内容"
