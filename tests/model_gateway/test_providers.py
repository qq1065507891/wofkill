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
            "werewolf_agent.model_gateway.providers.load_local_dotenv",
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
