"""Unit tests for ModelRouter."""

import pytest


def _mock_provider(name: str = "anthropic"):
    from werewolf_agent.model_gateway.router import MockProvider
    return MockProvider(name=name)


def _make_router(*, providers: dict | None = None):
    from werewolf_agent.model_gateway.router import ModelRouter
    profiles = providers or {}
    if "anthropic" in profiles:
        pass  # caller provides
    return ModelRouter(
        model_profiles={
            "claude_default": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "temperature": 0.7,
            },
        },
        llm_profiles={
            "anthropic_default": {
                "default": {
                    "provider": "anthropic",
                    "model_profile": "claude_default",
                },
            },
        },
        player_assignments={
            f"p{i:02d}": "anthropic_default" for i in range(1, 13)
        },
        providers=profiles,
    )


class TestModelRouterConstruction:
    def test_empty_providers(self) -> None:
        router = _make_router()
        assert router.provider_names() == []

    def test_registered_provider_appears_in_names(self) -> None:
        router = _make_router()
        router.register_provider(_mock_provider("anthropic"))
        assert "anthropic" in router.provider_names()

    def test_multiple_providers(self) -> None:
        router = _make_router()
        router.register_provider(_mock_provider("anthropic"))
        router.register_provider(_mock_provider("openai"))
        names = router.provider_names()
        assert "anthropic" in names
        assert "openai" in names

    def test_register_env_providers_skips_when_no_keys(self, monkeypatch) -> None:
        from werewolf_agent.model_gateway.providers import _ENV_OVERRIDES
        _ENV_OVERRIDES.clear()
        monkeypatch.setattr(
            "werewolf_agent.model_gateway.providers.load_local_dotenv",
            lambda *a, **kw: None,
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        router = _make_router()
        router.register_env_providers()
        assert router.provider_names() == []


class TestResolveConfig:
    def test_resolves_config_for_known_player(self) -> None:
        router = _make_router()
        config, fallback = router.resolve_config(agent_id="p01", task_type="speech")
        assert config is not None
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-6"

    def test_returns_mock_config_for_unknown_player(self) -> None:
        router = _make_router()
        config, fallback = router.resolve_config(agent_id="ghost", task_type="speech")
        assert config.provider == "mock"


class TestGenerateWithMockProvider:
    def test_generate_returns_text_from_mock(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        result = router.generate(agent_id="p01", task_type="speech", prompt="Hello")
        assert result is not None
        assert "mock response" in result.text

    def test_generate_registers_usage(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        router.generate(agent_id="p01", task_type="speech", prompt="test")
        usage = router.get_usage_log()
        assert len(usage) == 1
        assert usage[0].agent_id == "p01"

    def test_probe_tool_call_support_detects_mock(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        result = router.probe_tool_call_support(agent_id="p01", task_type="speech")
        assert isinstance(result, dict)


class TestConfigSnapshot:
    def test_snapshot_includes_profiles(self) -> None:
        router = _make_router()
        snapshot = router.config_snapshot()
        for key in ("model_profiles", "llm_profiles", "player_assignments"):
            assert key in snapshot

    def test_snapshot_excludes_volatile_state(self) -> None:
        router = _make_router()
        snapshot = router.config_snapshot()
        for key in ("providers", "usage_log"):
            assert key not in snapshot


class TestRetryHelpers:
    def test_is_not_retryable_for_value_error(self) -> None:
        from werewolf_agent.model_gateway.router import _is_retryable_exception
        assert not _is_retryable_exception(ValueError("bad"))

    def test_retry_delay_increases_with_attempt(self) -> None:
        from werewolf_agent.model_gateway.router import _retry_delay_for_exception
        d1 = _retry_delay_for_exception(RuntimeError("test"), 0)
        d2 = _retry_delay_for_exception(RuntimeError("test"), 2)
        assert d2 > d1


class TestFormatException:
    def test_format_exception_returns_message(self) -> None:
        from werewolf_agent.model_gateway.router import _format_exception
        msg = _format_exception(ValueError("test error"))
        assert "test error" in msg

    def test_format_none_returns_unknown(self) -> None:
        from werewolf_agent.model_gateway.router import _format_exception
        assert _format_exception(None) == "unknown"
