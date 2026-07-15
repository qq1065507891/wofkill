# -*- coding: utf-8 -*-
"""
验证 ModelRouter 的配置解析、provider 路由、重试与 fallback 记录。

作者: Project contributors
修改日期: 2026-07-15
"""

import pytest


def _mock_provider(name: str = "anthropic"):
    from werewolf_agent.model_gateway.router import MockProvider
    return MockProvider(name=name)


class _EmptyTextProvider:
    def __init__(self, name: str = "primary") -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord

        self.calls += 1
        return GenerateResult(
            text="",
            provider=self.name,
            model=config.model,
            usage=UsageRecord(
                agent_id="",
                task_type="",
                provider=self.name,
                model=config.model,
                latency_ms=500,
            ),
        )


class _StaticTextProvider:
    def __init__(self, text: str, name: str = "fallback") -> None:
        self._text = text
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord

        self.calls += 1
        return GenerateResult(
            text=self._text,
            provider=self.name,
            model=config.model,
            usage=UsageRecord(
                agent_id="",
                task_type="",
                provider=self.name,
                model=config.model,
                latency_ms=500,
            ),
        )


class _ReasoningStatusProvider(_StaticTextProvider):
    def __init__(self, status: str, name: str) -> None:
        super().__init__("ok", name)
        self.status = status

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        result = super().generate(prompt, config, system_prompt, tools, tool_choice)
        from dataclasses import replace
        return replace(result, reasoning_status=self.status, reasoning_tokens=(3 if self.status == "confirmed" else 0))


class _SequenceProvider:
    def __init__(self, responses, name: str) -> None:
        self._responses = list(responses)
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord

        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GenerateResult(
            text=response,
            provider=self.name,
            model=config.model,
            tool_call_received=bool(tool_choice),
            usage=UsageRecord(
                agent_id="", task_type="", provider=self.name, model=config.model,
            ),
        )


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
                "reasoning": {"level": "high"},
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
            "werewolf_agent.model_gateway.providers.env.load_local_dotenv",
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

    def test_unspecified_primary_max_tokens_uses_provider_default(self) -> None:
        router = _make_router()

        config, _fallback = router.resolve_config(
            agent_id="p01",
            task_type="speech",
        )

        assert config.max_tokens is None

    def test_unspecified_fallback_max_tokens_uses_provider_default(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        router = ModelRouter(
            model_profiles={
                "primary_profile": {
                    "provider": "primary",
                    "model": "primary-model",
                },
                "fallback_profile": {
                    "provider": "fallback",
                    "model": "fallback-model",
                },
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "primary",
                        "model_profile": "primary_profile",
                    },
                    "fallback": {
                        "provider": "fallback",
                        "model_profile": "fallback_profile",
                    },
                },
            },
            player_assignments={"p01": "default"},
        )

        fallback_config = router._resolve_fallback_model("default")

        assert fallback_config is not None
        assert fallback_config.max_tokens is None

    def test_resolves_reasoning_request_without_claiming_provider_support(self) -> None:
        router = _make_router()
        router._model_profiles["claude_default"]["reasoning"] = {"level": "high"}

        config, _fallback = router.resolve_config("p01", "speech")

        assert config.reasoning_level == "medium"
        assert config.reasoning_capability == "high"
        assert config.reasoning_requested is True


class TestGenerateWithMockProvider:
    def test_success_exposes_one_primary_attempt(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        router._model_profiles["claude_default"]["reasoning"] = {"level": "medium"}

        result = router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))

        assert len(result.attempts) == 1
        assert result.attempts[0].ordinal == 1
        assert result.attempts[0].route_kind.value == "primary"
        assert result.attempts[0].normalized_reasoning_status.value == "requested_unconfirmed"
        assert result.usage is not None
        assert result.usage.attempts == result.attempts

    def test_fallback_retains_task_reasoning_minimum_and_earlier_failure(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        primary = _EmptyTextProvider("primary")
        fallback = _StaticTextProvider("ok", "fallback")
        router = ModelRouter(
            model_profiles={
                "primary": {"provider": "primary", "model": "p", "retry_count": 0, "reasoning": {"level": "medium"}},
                "fallback": {"provider": "fallback", "model": "f", "retry_count": 0, "reasoning": {"level": "high"}},
            },
            llm_profiles={"profile": {
                "default": {"provider": "primary", "model_profile": "primary"},
                "fallback": {"provider": "fallback", "model_profile": "fallback"},
            }},
            player_assignments={"p01": "profile"},
            providers={"primary": primary, "fallback": fallback},
        )

        result = router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))

        assert [item.ordinal for item in result.attempts] == [1, 2]
        assert result.attempts[0].attempt_outcome.value == "attempt_failure"
        assert result.attempts[1].route_kind.value == "provider_fallback"
        assert result.attempts[1].requested_reasoning_level.value == "medium"

    def test_high_task_skips_medium_primary_and_none_fallback_without_provider_calls(self) -> None:
        primary = _StaticTextProvider("must not run", "primary")
        fallback = _StaticTextProvider("must not run", "fallback")
        from werewolf_agent.model_gateway.router import ModelRouter

        router = ModelRouter(
            model_profiles={
                "primary": {"provider": "primary", "model": "p", "reasoning": {"level": "medium"}},
                "fallback": {"provider": "fallback", "model": "f", "reasoning": {"level": "none"}},
            },
            llm_profiles={"profile": {
                "default": {"provider": "primary", "model_profile": "primary"},
                "fallback": {"provider": "fallback", "model_profile": "fallback"},
            }},
            player_assignments={"p01": "profile"},
            providers={"primary": primary, "fallback": fallback},
            validate_reasoning=False,
        )

        result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))

        assert primary.calls == fallback.calls == 0
        assert result.text == ""
        assert result.attempts[-1].normalized_reasoning_status.value == "fallback_disabled"

    def test_empty_response_retries_before_fallback(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        empty = _EmptyTextProvider("primary")
        router = ModelRouter(
            model_profiles={"p": {"provider": "primary", "model": "p", "retry_count": 1, "reasoning": {"level": "high"}}},
            llm_profiles={"profile": {"default": {"provider": "primary", "model_profile": "p"}}},
            player_assignments={"p01": "profile"}, providers={"primary": empty},
            validate_reasoning=False,
        )
        result = router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))
        assert empty.calls == 2
        assert [attempt.route_kind.value for attempt in result.attempts[:2]] == ["primary", "retry"]

    def test_success_without_provider_usage_still_records_attempt_denominator(self) -> None:
        class NoUsageProvider(_StaticTextProvider):
            def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
                from werewolf_agent.model_gateway.router import GenerateResult
                self.calls += 1
                return GenerateResult(text="ok", provider=self.name, model=config.model)

        provider = NoUsageProvider("ok", "primary")
        router = _make_router(providers={"anthropic": provider})
        router._model_profiles["claude_default"]["reasoning"] = {"level": "high"}
        result = router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))
        assert result.usage is not None
        assert result.usage.attempts == result.attempts
        assert len(router.get_usage_log()) == 1

    def test_ordered_fallback_chain_skips_incapable_candidate(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        primary = _EmptyTextProvider("primary")
        weak = _StaticTextProvider("must not run", "weak")
        strong = _StaticTextProvider("ok", "strong")
        router = ModelRouter(
            model_profiles={
                "primary": {"provider": "primary", "model": "p", "retry_count": 0, "reasoning": {"level": "high"}},
                "weak": {"provider": "weak", "model": "w", "reasoning": {"level": "none"}},
                "strong": {"provider": "strong", "model": "s", "retry_count": 0, "reasoning": {"level": "high"}},
            },
            llm_profiles={"profile": {
                "default": {"provider": "primary", "model_profile": "primary"},
                "fallback": [
                    {"provider": "weak", "model_profile": "weak"},
                    {"provider": "strong", "model_profile": "strong"},
                ],
            }},
            player_assignments={"p01": "profile"},
            providers={"primary": primary, "weak": weak, "strong": strong},
            validate_reasoning=False,
        )
        result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))
        assert weak.calls == 0
        assert strong.calls == 1
        assert result.provider == "strong"

    def test_unregistered_primary_records_safe_failed_attempt_before_fallback(
        self,
    ) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter
        from werewolf_agent.runtime.decision_outcomes import (
            summarize_attempt_counts,
            translate_decision_outcome,
        )

        fallback = _StaticTextProvider("ok", "fallback")
        router = ModelRouter(
            model_profiles={
                "primary": {
                    "provider": "secret-provider-key",
                    "model": "secret-model-token",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
                "fallback": {
                    "provider": "fallback",
                    "model": "safe-model",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
            },
            llm_profiles={"profile": {
                "default": {"provider": "secret-provider-key", "model_profile": "primary"},
                "fallback": {"provider": "fallback", "model_profile": "fallback"},
            }},
            player_assignments={"p01": "profile"},
            providers={"fallback": fallback},
            validate_reasoning=False,
        )

        result = router.generate(
            "p01", "reflection", "hello", jitter_seconds=(0, 0)
        )

        assert fallback.calls == 1
        assert [item.route_kind.value for item in result.attempts] == [
            "primary",
            "provider_fallback",
        ]
        assert result.attempts[0].root_cause.value == "provider_error"
        assert result.attempts[0].attempt_outcome.value == "attempt_failure"
        assert "secret-provider-key" not in repr(result.attempts)
        assert "secret-model-token" not in repr(result.attempts)
        translated = translate_decision_outcome(result.attempts)
        counts = summarize_attempt_counts(result.attempts)
        assert translated.outcome.value == "provider_fallback_success"
        assert (counts.attempt_count, counts.retry_count, counts.provider_fallback_count) == (
            2,
            0,
            1,
        )
        assert result.usage is not None
        assert result.usage.retry_count == 0
        assert "secret-provider-key" not in repr(result.usage)
        assert "secret-model-token" not in repr(result.usage)

    def test_missing_provider_returns_auditable_terminal_failure(self) -> None:
        router = _make_router(providers={})
        router._model_profiles["claude_default"]["reasoning"] = {"level": "high"}
        result = router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))
        assert result.text == ""
        assert result.attempts[0].provider == "unavailable"
        assert result.attempts[0].model == "unavailable"
        assert result.attempts[-1].route_kind.value == "safe_fallback"
        assert result.attempts[-1].provider == "unavailable"
        assert result.attempts[-1].model == "unavailable"
        assert result.attempts[-1].root_cause.value == "provider_error"
        assert router.get_usage_log()[-1].attempts == result.attempts

    def test_ordered_fallback_continues_after_capable_candidate_failure(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        first = _EmptyTextProvider("first")
        second = _StaticTextProvider("ok", "second")
        router = ModelRouter(
            model_profiles={
                "primary": {"provider": "missing", "model": "p", "reasoning": {"level": "high"}},
                "first": {"provider": "first", "model": "f1", "retry_count": 0, "reasoning": {"level": "high"}},
                "second": {"provider": "second", "model": "f2", "retry_count": 0, "reasoning": {"level": "high"}},
            },
            llm_profiles={"profile": {
                "default": {"provider": "missing", "model_profile": "primary"},
                "fallback": [
                    {"provider": "first", "model_profile": "first"},
                    {"provider": "second", "model_profile": "second"},
                ],
            }},
            player_assignments={"p01": "profile"},
            providers={"first": first, "second": second}, validate_reasoning=False,
        )
        result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))
        assert first.calls == second.calls == 1
        assert [attempt.provider for attempt in result.attempts] == [
            "unavailable",
            "first",
            "second",
        ]
        assert result.text == "ok"

    @pytest.mark.parametrize(
        ("responses", "expected_root"),
        [
            (["", "ok"], "invalid_output"),
            ([RuntimeError("HTTP 503 service unavailable"), "ok"], "provider_error"),
        ],
    )
    def test_fallback_retry_uses_backoff_and_candidate_warning(
        self, responses, expected_root, monkeypatch, caplog,
    ) -> None:
        from werewolf_agent.model_gateway import router as router_module
        from werewolf_agent.model_gateway.router import ModelRouter

        provider = _SequenceProvider(responses, "fallback")
        sleeps: list[float] = []
        monkeypatch.setattr(router_module, "_retry_delay_for_exception", lambda exc, attempt: 1.75)
        monkeypatch.setattr(router_module.time, "sleep", sleeps.append)
        router = ModelRouter(
            model_profiles={
                "primary": {"provider": "missing", "model": "p", "reasoning": {"level": "high"}},
                "fallback": {"provider": "fallback", "model": "f", "retry_count": 1, "reasoning": {"level": "high"}},
            },
            llm_profiles={"profile": {
                "default": {"provider": "missing", "model_profile": "primary"},
                "fallback": {"provider": "fallback", "model_profile": "fallback"},
            }},
            player_assignments={"p01": "profile"},
            providers={"fallback": provider},
            validate_reasoning=False,
        )

        with caplog.at_level("WARNING"):
            result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))

        assert result.text == "ok"
        assert provider.calls == 2
        assert sleeps == [1.75]
        assert result.attempts[0].root_cause.value == "provider_error"
        assert result.attempts[1].root_cause.value == expected_root
        assert [item.ordinal for item in result.attempts] == [1, 2, 3]
        assert [item.route_kind.value for item in result.attempts] == [
            "primary",
            "provider_fallback",
            "retry",
        ]
        warning = "\n".join(record.getMessage() for record in caplog.records)
        assert "provider=fallback" in warning
        assert "model=f" in warning
        assert "attempt 1/2" in warning
        assert "retry in 1.8s" in warning

    def test_fallback_retry_records_one_transition_then_retry(
        self, monkeypatch,
    ) -> None:
        from werewolf_agent.model_gateway import router as router_module
        from werewolf_agent.model_gateway.router import ModelRouter
        from werewolf_agent.runtime.decision_outcomes import (
            summarize_attempt_counts,
            translate_decision_outcome,
        )

        primary = _SequenceProvider(
            [RuntimeError("HTTP 503 primary unavailable")],
            "primary",
        )
        fallback = _SequenceProvider(
            [RuntimeError("HTTP 503 fallback retry"), "ok"],
            "fallback",
        )
        monkeypatch.setattr(router_module.time, "sleep", lambda _: None)
        router = ModelRouter(
            model_profiles={
                "primary": {
                    "provider": "primary",
                    "model": "p",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
                "fallback": {
                    "provider": "fallback",
                    "model": "f",
                    "retry_count": 1,
                    "reasoning": {"level": "high"},
                },
            },
            llm_profiles={"profile": {
                "default": {"provider": "primary", "model_profile": "primary"},
                "fallback": {"provider": "fallback", "model_profile": "fallback"},
            }},
            player_assignments={"p01": "profile"},
            providers={"primary": primary, "fallback": fallback},
            validate_reasoning=False,
        )

        result = router.generate(
            "p01", "reflection", "hello", jitter_seconds=(0, 0)
        )

        assert [item.route_kind.value for item in result.attempts] == [
            "primary",
            "provider_fallback",
            "retry",
        ]
        translated = translate_decision_outcome(result.attempts)
        counts = summarize_attempt_counts(result.attempts)
        assert translated.provider_fallback_count == 1
        assert (counts.attempt_count, counts.retry_count) == (3, 1)

    @pytest.mark.parametrize(
        ("statuses", "expected_provider"),
        [
            (["unsupported", "confirmed"], "fallback1"),
            (["unsupported", "unsupported", "confirmed"], "fallback2"),
        ],
    )
    def test_unsupported_reasoning_continues_to_confirmed_candidate(self, statuses, expected_provider) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        names = ["primary", "fallback1", "fallback2"][:len(statuses)]
        profiles = {
            name: {"provider": name, "model": name, "retry_count": 0, "reasoning": {"level": "high"}}
            for name in names
        }
        providers = {
            name: _ReasoningStatusProvider(status, name)
            for name, status in zip(names, statuses)
        }
        router = ModelRouter(
            model_profiles=profiles,
            llm_profiles={"profile": {
                "default": {"provider": "primary", "model_profile": "primary"},
                "fallback": [
                    {"provider": name, "model_profile": name} for name in names[1:]
                ],
            }},
            player_assignments={"p01": "profile"}, providers=providers,
            validate_reasoning=False,
        )
        result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))
        assert result.provider == expected_provider
        assert result.attempts[-1].normalized_reasoning_status.value == "confirmed"
        assert result.attempts[0].normalized_reasoning_status.value == "unsupported"

    def test_all_unsupported_reasoning_fails_closed(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        primary = _ReasoningStatusProvider("unsupported", "primary")
        router = ModelRouter(
            model_profiles={"primary": {"provider": "primary", "model": "p", "retry_count": 0, "reasoning": {"level": "high"}}},
            llm_profiles={"profile": {"default": {"provider": "primary", "model_profile": "primary"}}},
            player_assignments={"p01": "profile"}, providers={"primary": primary}, validate_reasoning=False,
        )
        result = router.generate("p01", "reflection", "hello", jitter_seconds=(0, 0))
        assert result.text == ""
        assert result.attempts[0].normalized_reasoning_status.value == "unsupported"
        assert result.attempts[-1].route_kind.value == "safe_fallback"

    def test_generate_returns_text_from_mock(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        result = router.generate(agent_id="p01", task_type="speech", prompt="Hello")
        assert result is not None
        assert "mock response" in result.text

    def test_first_attempt_jitter_uses_short_default_window(self, monkeypatch) -> None:
        from werewolf_agent.model_gateway import router as router_module

        uniform_calls: list[tuple[float, float]] = []
        sleeps: list[float] = []

        def fake_uniform(low: float, high: float) -> float:
            uniform_calls.append((low, high))
            return high

        monkeypatch.setattr(router_module.random, "uniform", fake_uniform)
        monkeypatch.setattr(router_module.time, "sleep", lambda seconds: sleeps.append(seconds))

        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        router.generate(agent_id="p01", task_type="speech", prompt="Hello")

        assert uniform_calls[0] == (0, 0.8)
        assert sleeps[0] == 0.8

    def test_jitter_zero_in_test_mode(self, monkeypatch) -> None:
        """R3-MG-3: jitter_seconds=(0, 0) on a router call must skip the
        pre-call sleep entirely so test suites are not slowed by 5-15s of
        cumulative jitter across 12 players × many rounds.
        """
        from werewolf_agent.model_gateway import router as router_module

        uniform_calls: list[tuple[float, float]] = []
        sleeps: list[float] = []

        def fake_uniform(low: float, high: float) -> float:
            uniform_calls.append((low, high))
            return high

        monkeypatch.setattr(router_module.random, "uniform", fake_uniform)
        monkeypatch.setattr(router_module.time, "sleep", lambda seconds: sleeps.append(seconds))

        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        router.generate(
            agent_id="p01",
            task_type="speech",
            prompt="Hello",
            jitter_seconds=(0, 0),
        )

        # No sleep should have been issued because jitter_seconds is (0, 0).
        assert sleeps == []

    def test_generate_registers_usage(self) -> None:
        router = _make_router(providers={"anthropic": _mock_provider("anthropic")})
        router.generate(agent_id="p01", task_type="speech", prompt="test")
        usage = router.get_usage_log()
        assert len(usage) == 1
        assert usage[0].agent_id == "p01"

    def test_empty_primary_response_tries_fallback_provider(self) -> None:
        from werewolf_agent.model_gateway.router import ModelRouter

        primary = _EmptyTextProvider("primary")
        fallback = _StaticTextProvider(
            '{"action_type":"speech","target_id":null,'
            '"speech":"fallback ok","reason":"fallback","confidence":0.7}',
            "fallback",
        )
        router = ModelRouter(
            model_profiles={
                "primary_model": {
                    "provider": "primary",
                    "model": "primary-model",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
                "fallback_model": {
                    "provider": "fallback",
                    "model": "fallback-model",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "primary",
                        "model_profile": "primary_model",
                    },
                    "fallback": {
                        "provider": "fallback",
                        "model_profile": "fallback_model",
                    },
                },
            },
            player_assignments={"p01": "default"},
            providers={"primary": primary, "fallback": fallback},
        )

        result = router.generate(
            agent_id="p01",
            task_type="speech",
            prompt="hello",
            jitter_seconds=(0, 0),
        )

        assert result.provider == "fallback"
        assert "fallback ok" in result.text
        assert primary.calls == 1
        assert fallback.calls == 1

    def test_fallback_usage_records_request_and_model_transition(self) -> None:
        """fallback 元数据必须能还原主模型到备用模型的切换。"""
        from werewolf_agent.model_gateway.router import ModelRouter

        primary = _EmptyTextProvider("primary")
        fallback = _StaticTextProvider('{"action_type":"speech","speech":"ok"}', "fallback")
        router = ModelRouter(
            model_profiles={
                "primary_model": {
                    "provider": "primary",
                    "model": "primary-model",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
                "fallback_model": {
                    "provider": "fallback",
                    "model": "fallback-model",
                    "retry_count": 0,
                    "reasoning": {"level": "high"},
                },
            },
            llm_profiles={
                "default": {
                    "default": {"provider": "primary", "model_profile": "primary_model"},
                    "fallback": {"provider": "fallback", "model_profile": "fallback_model"},
                },
            },
            player_assignments={"p01": "default"},
            providers={"primary": primary, "fallback": fallback},
        )

        router.generate("p01", "speech", "hello", jitter_seconds=(0, 0))
        usage = router.get_usage_log()[-1]

        assert usage.request_id
        assert usage.primary_provider == "primary"
        assert usage.primary_model == "primary-model"
        assert usage.fallback_provider == "fallback"
        assert usage.fallback_model == "fallback-model"
        assert usage.retry_count == 0
        assert usage.failure_category == "invalid_output"

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


class TestFromYamlValidation:
    def test_from_yaml_raises_on_unknown_model_profile(self, tmp_path) -> None:
        """R3-MG-1: typos in model_profile references must surface at load time."""
        from werewolf_agent.model_gateway.router import ModelRouter

        yaml_path = tmp_path / "bad_models.yaml"
        yaml_path.write_text(
            "model_profiles:\n"
            "  real_profile:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet-4-6\n"
            "    reasoning:\n"
            "      level: high\n"
            "llm_profiles:\n"
            "  default:\n"
            "    default:\n"
            "      provider: anthropic\n"
            "      model_profile: real_profilo\n"  # typo
            "players:\n"
            "  p01:\n"
            "    llm_profile: default\n",
            encoding="utf-8",
        )
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            ModelRouter.from_yaml(yaml_path)

    def test_from_yaml_raises_on_unknown_llm_profile_ref(self, tmp_path) -> None:
        """R3-MG-1: players.<id>.llm_profile pointing to a missing profile raises."""
        from werewolf_agent.model_gateway.router import ModelRouter

        yaml_path = tmp_path / "bad_llm.yaml"
        yaml_path.write_text(
            "model_profiles:\n"
            "  real_profile:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet-4-6\n"
            "    reasoning:\n"
            "      level: high\n"
            "llm_profiles:\n"
            "  default:\n"
            "    default:\n"
            "      provider: anthropic\n"
            "      model_profile: real_profile\n"
            "players:\n"
            "  p01:\n"
            "    llm_profile: ghost_profile\n",
            encoding="utf-8",
        )
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            ModelRouter.from_yaml(yaml_path)

    def test_from_yaml_raises_on_unknown_provider(self, tmp_path) -> None:
        """R3-MG-1: model_profile.provider that the runtime does not know raises."""
        from werewolf_agent.model_gateway.router import ModelRouter

        yaml_path = tmp_path / "bad_provider.yaml"
        yaml_path.write_text(
            "model_profiles:\n"
            "  real_profile:\n"
            "    provider: chocochip_9000\n"  # not a known provider
            "    model: x\n"
            "llm_profiles:\n"
            "  default:\n"
            "    default:\n"
            "      provider: chocochip_9000\n"
            "      model_profile: real_profile\n"
            "players:\n"
            "  p01:\n"
            "    llm_profile: default\n",
            encoding="utf-8",
        )
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            ModelRouter.from_yaml(yaml_path)

    def test_from_yaml_accepts_valid_config(self, tmp_path) -> None:
        """R3-MG-1: a valid config does not raise."""
        from werewolf_agent.model_gateway.router import ModelRouter

        yaml_path = tmp_path / "good_models.yaml"
        yaml_path.write_text(
            "model_profiles:\n"
            "  real_profile:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet-4-6\n"
            "    reasoning:\n"
            "      level: high\n"
            "llm_profiles:\n"
            "  default:\n"
            "    default:\n"
            "      provider: anthropic\n"
            "      model_profile: real_profile\n"
            "players:\n"
            "  p01:\n"
            "    llm_profile: default\n",
            encoding="utf-8",
        )
        router = ModelRouter.from_yaml(yaml_path)
        assert router.get_llm_profile_for_agent("p01") == "default"


class TestFallbackModelProfile:
    def test_missing_fallback_model_profile_raises(self) -> None:
        """R3-MG-7: a fallback with an unknown model_profile must raise at
        config-validation time rather than silently returning
        ``ModelConfig(model="")`` at first fallback invocation.
        """
        from werewolf_agent.model_gateway.providers import ProviderConfigError
        from werewolf_agent.model_gateway.router import ModelRouter

        # Direct construction with a bad fallback model_profile id.
        router = ModelRouter(
            model_profiles={
                "primary_profile": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                },
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "anthropic",
                        "model_profile": "primary_profile",
                    },
                    "fallback": {
                        "provider": "mock",
                        "model_profile": "ghost_profile",  # typo
                    },
                },
            },
            player_assignments={"p01": "default"},
        )
        with pytest.raises(ProviderConfigError):
            router._resolve_fallback_model(llm_profile_id="default")

    def test_missing_fallback_model_profile_raises_from_yaml(self, tmp_path) -> None:
        """R3-MG-7: from_yaml should fail fast on a bad fallback ref."""
        from werewolf_agent.model_gateway.providers import ProviderConfigError
        from werewolf_agent.model_gateway.router import ModelRouter

        yaml_path = tmp_path / "bad_fallback.yaml"
        yaml_path.write_text(
            "model_profiles:\n"
            "  primary_profile:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet-4-6\n"
            "llm_profiles:\n"
            "  default:\n"
            "    default:\n"
            "      provider: anthropic\n"
            "      model_profile: primary_profile\n"
            "    fallback:\n"
            "      provider: mock\n"
            "      model_profile: ghost_profile\n"
            "players:\n"
            "  p01:\n"
            "    llm_profile: default\n",
            encoding="utf-8",
        )
        with pytest.raises(ProviderConfigError):
            ModelRouter.from_yaml(yaml_path)


class TestRegisterEnvProvidersLogging:
    def test_register_env_providers_logs_missing(self, caplog) -> None:
        """R3-MG-8: a configured provider whose API key is absent must
        produce a WARNING log naming the missing providers.
        """
        from unittest.mock import patch
        from werewolf_agent.model_gateway.router import ModelRouter

        # Build a router whose model_profiles requires a provider that
        # we know won't have a key in test env.
        router = ModelRouter(
            model_profiles={
                "needs_openai": {"provider": "openai", "model": "x"},
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "openai",
                        "model_profile": "needs_openai",
                    },
                },
            },
            player_assignments={"p01": "default"},
        )
        # Patch the re-exported name on the providers package, since
        # register_env_providers imports from there.
        with patch(
            "werewolf_agent.model_gateway.providers.create_provider_from_env",
            return_value=None,
        ):
            with caplog.at_level("WARNING"):
                router.register_env_providers()
        assert any(
            "register_env_providers" in rec.getMessage()
            and "openai" in rec.getMessage()
            for rec in caplog.records
        ), f"expected WARNING naming 'openai', got: {[r.getMessage() for r in caplog.records]}"


class TestFormatException:
    def test_format_exception_returns_message(self) -> None:
        from werewolf_agent.model_gateway.router import _format_exception
        msg = _format_exception(ValueError("test error"))
        assert "test error" in msg

    def test_format_none_returns_unknown(self) -> None:
        from werewolf_agent.model_gateway.router import _format_exception
        assert _format_exception(None) == "unknown"


class TestHttpStatusFromExceptionAccuracy:
    """N1 (post-review-v2): traceback 里的年份/端口不应被误抓为 HTTP 状态码。

    修复前: 正则 ``\\b([1-5]\\d{2})\\b`` 会从 ``"Failed at line 2024"`` 或
    ``"localhost:8080"`` 之类的字符串里抓到 808 当成 HTTP 状态码, 影响
    失败归因。
    修复后: 优先读 ``exc.status_code`` / ``exc.response.status_code``;
    fallback 只接受 ``HTTP NNN`` 前缀 (避免数字误抓)。
    """

    def test_year_2024_not_misclassified(self) -> None:
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("Failed at line 2024 of script")
        result = _http_status_from_exception(exc)
        assert result not in (2024, "2024"), f"year misclassified: {result}"

    def test_port_8080_not_misclassified(self) -> None:
        """``localhost:8080`` 里的 808 不应被误判为 HTTP 状态码。"""
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("Connection refused at localhost:8080")
        result = _http_status_from_exception(exc)
        assert result not in (808, "808", 8080, "8080"), f"port misclassified: {result}"

    def test_real_http_status_500_still_detected(self) -> None:
        from werewolf_agent.model_gateway.router import _http_status_from_exception
        exc = RuntimeError("HTTP 500 Internal Server Error")
        result = _http_status_from_exception(exc)
        assert result in (500, "500"), f"should detect 500: {result}"

    def test_explicit_status_code_attribute_takes_priority(self) -> None:
        """``exc.status_code`` (e.g. requests.HTTPError) 应被优先识别。"""
        from werewolf_agent.model_gateway.router import _http_status_from_exception

        class FakeResp:
            status_code = 429

        class FakeHTTPError(Exception):
            pass

        err = FakeHTTPError("Too many requests")
        err.status_code = 429  # type: ignore[attr-defined]
        result = _http_status_from_exception(err)
        assert result == 429, f"status_code attribute not honored: {result}"
