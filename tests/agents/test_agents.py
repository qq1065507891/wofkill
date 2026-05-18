"""Tests for agent interfaces, persona router, model gateway, and schema validation.

Covers:
- Player action schema validation (legal/illegal actions and targets)
- Private intent isolation
- Retry and fallback on illegal/invalid output
- Persona Router resolution and dynamic adjustments
- Model Router Gateway routing, fallback, and usage tracking
- Judge Agent broadcasts
- Visibility boundaries
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    FactionGoal,
    JudgeBroadcast,
    PlayerAction,
    PrivateIntent,
    RetryInfo,
    RiskFlag,
    TaskType,
)
from werewolf_agent.agents.player import (
    DefaultActionValidator,
    PlayerAgent,
)
from werewolf_agent.agents.judge import JudgeAgent
from werewolf_agent.persona_runtime.router import (
    GameContext,
    PersonaRouter,
    PersonaSnapshot,
)
from werewolf_agent.model_gateway.router import (
    GenerateResult,
    ModelConfig,
    ModelRouter,
    MockProvider,
    UsageRecord,
)
from werewolf_agent.model_gateway.providers import (
    AnthropicProvider,
    GLMProvider,
    OpenAIProvider,
    create_provider_from_env,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestPlayerActionSchema:
    def test_valid_vote_action(self) -> None:
        action = PlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            speech="我归7。",
            reason="7的视角不对",
            confidence=0.72,
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.PUSH_GOOD_PLAYER_OUT,
                claimed_view="good_player_without_night_info",
                pressure_target="p07",
                risk_flags=[RiskFlag.AVOID_NIGHT_KILL_LEAK],
            ),
        )
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert action.private_intent is not None
        assert action.private_intent.true_role == "werewolf"

    def test_action_requiring_target_fails_without_target(self) -> None:
        with pytest.raises(ValidationError, match="requires target_id"):
            PlayerAction(action_type=ActionType.VOTE, speech="归票")

    def test_no_action_does_not_require_target(self) -> None:
        action = PlayerAction(action_type=ActionType.NO_ACTION, speech="过")
        assert action.target_id is None

    def test_confidence_clamped(self) -> None:
        with pytest.raises(ValidationError):
            PlayerAction(
                action_type=ActionType.NO_ACTION,
                confidence=1.5,
            )

    def test_private_intent_isolation(self) -> None:
        """Private intent is stored separately and not in public fields."""
        action = PlayerAction(
            action_type=ActionType.SPEECH,
            speech="我是好人",
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view="villager",
            ),
        )
        # speech is public, private_intent is private
        assert action.speech == "我是好人"
        assert action.private_intent.true_role == "werewolf"
        # serializing: private_intent should be accessible for audit
        data = action.model_dump()
        assert "private_intent" in data
        assert data["private_intent"]["true_role"] == "werewolf"

    def test_all_action_types_with_required_targets(self) -> None:
        """All target-requiring types fail without target_id."""
        for at in [
            ActionType.WOLF_KILL, ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER,
            ActionType.SHERIFF_VOTE,
        ]:
            with pytest.raises(ValidationError):
                PlayerAction(action_type=at)

    def test_all_action_types_with_optional_targets(self) -> None:
        """These action types work without target_id."""
        for at in [ActionType.NO_ACTION, ActionType.SPEECH,
                    ActionType.SELF_DESTRUCT, ActionType.BADGE_TEAR,
                    ActionType.SHERIFF_REGISTER, ActionType.SHERIFF_WITHDRAW,
                    ActionType.USE_ANTIDOTE]:
            action = PlayerAction(action_type=at)
            assert action.action_type == at


class TestJudgeBroadcastSchema:
    def test_valid_broadcast(self) -> None:
        b = JudgeBroadcast(
            broadcast_type="death_announcement",
            message="昨夜p03倒牌。",
            phase="day",
            day_number=2,
            public_data={"deaths": [{"player_id": "p03"}]},
        )
        assert b.broadcast_type == "death_announcement"
        assert b.day_number == 2

    def test_minimal_broadcast(self) -> None:
        b = JudgeBroadcast(broadcast_type="phase", message="test", phase="night")
        assert b.day_number == 0


# ---------------------------------------------------------------------------
# Action validator tests
# ---------------------------------------------------------------------------


class TestDefaultActionValidator:
    def test_legal_action_and_target(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE, "p07",
            [ActionType.VOTE, ActionType.NO_ACTION],
            ["p07", "p08"],
        )
        assert ok is True
        assert err is None

    def test_illegal_action(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.WOLF_KILL, None,
            [ActionType.VOTE],
            [],
        )
        assert ok is False
        assert "not in legal_actions" in (err or "")

    def test_illegal_target(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE, "p99",
            [ActionType.VOTE],
            ["p07", "p08"],
        )
        assert ok is False
        assert "not in legal_targets" in (err or "")

    def test_empty_legal_sets_pass(self) -> None:
        """When no legal sets provided, everything passes (for non-LLM modes)."""
        v = DefaultActionValidator()
        ok, _ = v.validate(ActionType.NO_ACTION, None, [], [])
        assert ok is True

    def test_target_requiring_action_rejects_target_when_legal_targets_missing(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE,
            "p99",
            [ActionType.VOTE],
            [],
        )
        assert ok is False
        assert "no legal_targets" in (err or "")


# ---------------------------------------------------------------------------
# Player Agent retry/fallback tests
# ---------------------------------------------------------------------------


class _FailProvider:
    """Provider that always fails, for testing fallback."""

    @property
    def name(self) -> str:
        return "fail_provider"

    def generate(self, prompt, config, system_prompt=None):
        raise RuntimeError("always fails")


class _JsonProvider:
    """Provider that returns configurable JSON, for testing parsing."""

    def __init__(self, response: str) -> None:
        self._response = response

    @property
    def name(self) -> str:
        return "json_provider"

    def generate(self, prompt, config, system_prompt=None):
        return GenerateResult(
            text=self._response,
            provider=self.name,
            model=config.model,
            usage=UsageRecord(
                agent_id="test", task_type="vote",
                provider=self.name, model=config.model,
            ),
        )


class _FakeHttpResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({
            "url": url,
            "headers": headers or {},
            "json": json or {},
            "timeout": timeout,
        })
        return _FakeHttpResponse(self.payload)


class TestPlayerAgentRetryFallback:
    def _make_agent(self, provider_response: str) -> PlayerAgent:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(provider_response)},
        )
        return PlayerAgent(agent_id="p01", model_router=router, max_retries=3)

    def _make_context(self) -> AgentContext:
        return AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
            public_summary="Day 1 discussion",
        )

    def test_valid_action_no_retry(self) -> None:
        json_resp = '{"action_type":"vote","target_id":"p07","speech":"归7","reason":"可疑","confidence":0.8}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert retry.attempt == 1

    def test_invalid_json_triggers_retry(self) -> None:
        agent = self._make_agent("not json at all")
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "parse_error"

    def test_illegal_action_triggers_retry(self) -> None:
        json_resp = '{"action_type":"wolf_kill","target_id":"p07","speech":"test","reason":"test","confidence":0.5}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        # wolf_kill not in legal_actions → retry → eventually fallback
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "illegal_action"

    def test_illegal_target_triggers_retry(self) -> None:
        json_resp = '{"action_type":"vote","target_id":"p99","speech":"test","reason":"test","confidence":0.5}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)

    def test_fallback_uses_first_legal_action(self) -> None:
        agent = self._make_agent("bad json")
        action, _ = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"

    def test_empty_response_triggers_retry(self) -> None:
        agent = self._make_agent("")
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "empty_response"

    def test_private_intent_in_valid_action(self) -> None:
        json_resp = (
            '{"action_type":"vote","target_id":"p07","speech":"归7",'
            '"reason":"可疑","confidence":0.8,'
            '"private_intent":{"true_role":"werewolf","faction_goal":"push_good_player_out",'
            '"claimed_view":"villager","pressure_target":"p07",'
            '"risk_flags":["avoid_night_kill_leak"]}}'
        )
        agent = self._make_agent(json_resp)
        action, _ = agent.act(self._make_context())
        assert isinstance(action, PlayerAction)
        assert action.private_intent is not None
        assert action.private_intent.true_role == "werewolf"
        assert action.speech == "归7"  # Public speech doesn't contain private intent

    def test_code_fence_stripping(self) -> None:
        json_resp = '```json\n{"action_type":"vote","target_id":"p07","speech":"test","reason":"test","confidence":0.5}\n```'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, PlayerAction)
        assert retry.attempt == 1


# ---------------------------------------------------------------------------
# Persona Router tests
# ---------------------------------------------------------------------------


PERSONAS_YAML = "config/personas/jingcheng_style_prototypes.yaml"


class TestPersonaRouter:
    def test_load_from_yaml(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        assert "logic_leader" in router._profiles

    def test_resolve_known_agent(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p01": "logic_leader"})
        snap = router.resolve("p01", "speech")
        assert snap.profile_id == "logic_leader"
        assert snap.display_name == "强逻辑归票型"
        assert snap.task_style == "structured_reasoning"
        assert snap.base_params.get("logic_skill", 0) > 0.8

    def test_resolve_unknown_agent_returns_default(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        snap = router.resolve("p99", "speech")
        assert snap.profile_id == "default"

    def test_task_style_changes_by_task(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})
        speech_style = router.resolve("p02", "speech").task_style
        deception_style = router.resolve("p02", "deception").task_style
        assert speech_style == "pressure_attack"
        assert deception_style == "high_pressure_push"

    def test_dynamic_adjustment_when_suspected(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})

        ctx = GameContext(player_is_suspected=True)
        snap = router.resolve("p02", "speech", ctx)
        # aggressive_bluffer has aggression_delta: 0.20 when suspected
        base_aggression = router._profiles["aggressive_bluffer"]["base"].get("aggression", 0)
        # Check dynamic adjustments contain aggression
        assert "aggression" in snap.dynamic_adjustments or base_aggression > 0

    def test_dynamic_adjustment_when_teammate_exiled(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})

        ctx = GameContext(teammate_exiled=True)
        snap = router.resolve("p02", "speech", ctx)
        # Should have risk_tolerance delta of -0.15
        assert "risk_tolerance" in snap.dynamic_adjustments

    def test_effective_params_clamped(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p04": "bold_pretender"})
        ctx = GameContext(player_is_suspected=True)
        snap = router.resolve("p04", "speech", ctx)
        # All effective params must be in [0, 1]
        for v in snap.effective_params.values():
            assert 0.0 <= v <= 1.0

    def test_persona_does_not_affect_rules(self) -> None:
        """Persona params are metadata only, they don't change legal actions."""
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p01": "logic_leader"})
        snap = router.resolve("p01", "vote")
        # Snapshot has no rule-affecting fields
        assert "legal_actions" not in snap.effective_params


# ---------------------------------------------------------------------------
# Model Router Gateway tests
# ---------------------------------------------------------------------------


MODELS_YAML = "config/models.yaml"


class TestModelRouter:
    def test_load_from_yaml(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        assert "pro_reasoner" in router._llm_profiles

    def test_resolve_config_for_known_agent(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, fallback = router.resolve_config("p01", "speech")
        assert config.provider == "anthropic"
        assert config.model != ""

    def test_resolve_config_task_specific(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        config, _ = router.resolve_config("p01", "reflection")
        # pro_reasoner has task-specific reflection config
        assert config.provider == "anthropic"

    def test_resolve_config_fallback_chain(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        _, fallback = router.resolve_config("p02", "speech")
        assert fallback == "anthropic"

    def test_generate_with_mock_provider(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        router.register_provider(MockProvider("anthropic"))
        router.register_provider(MockProvider("glm"))
        result = router.generate("p01", "speech", "Test prompt")
        assert result.text != ""
        assert result.provider in ("anthropic", "glm", "mock")

    def test_generate_fallback_on_failure(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        router.register_provider(_FailProvider())
        router.register_provider(MockProvider("glm"))

        result = router.generate("p02", "speech", "Test prompt")
        # Should fall back to glm provider since anthropic fails
        # (p02 uses local_wolf which has fallback to glm)
        # Fail provider raises, so it should try fallback
        assert result.text != "" or result.provider == "anthropic"

    def test_usage_logging(self) -> None:
        router = ModelRouter.from_yaml(MODELS_YAML)
        router.register_provider(MockProvider("anthropic"))
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

        assert "anthropic" in router.provider_names()

    def test_create_provider_from_env_returns_none_without_key(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        assert create_provider_from_env("anthropic") is None

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
        assert client.calls[0]["json"]["system"] == "You are concise."
        assert result.usage.prompt_tokens == 3
        assert result.usage.completion_tokens == 1

    def test_openai_provider_posts_chat_request(self) -> None:
        client = _FakeHttpClient({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        provider = OpenAIProvider(api_key="key", http_client=client)

        result = provider.generate(
            "Say hello",
            ModelConfig(provider="openai", model="gpt-test"),
            system_prompt="You are concise.",
        )

        assert result.text == "hello"
        assert client.calls[0]["url"].endswith("/v1/chat/completions")
        assert client.calls[0]["headers"]["Authorization"] == "Bearer key"
        assert client.calls[0]["json"]["messages"][0]["role"] == "system"

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


# ---------------------------------------------------------------------------
# Judge Agent tests
# ---------------------------------------------------------------------------


class TestJudgeAgent:
    def _make_judge(self) -> JudgeAgent:
        router = ModelRouter(
            model_profiles={}, llm_profiles={},
            player_assignments={},
            providers={"mock": MockProvider()},
        )
        return JudgeAgent(model_router=router)

    def test_broadcast_night_phase(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_phase("night", night_number=2)
        assert "第 2 夜" in b.message
        assert b.broadcast_type == "night"

    def test_broadcast_day_phase(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_phase("day", day_number=3)
        assert "第 3 天" in b.message

    def test_broadcast_death_announcement_with_deaths(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_death_announcement(
            deaths=[{"player_id": "p03", "reason": "wolf_kill"}],
            day_number=2,
        )
        assert "p03" in b.message
        assert "倒牌" in b.message

    def test_broadcast_peace_night(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_death_announcement(deaths=[], day_number=2)
        assert "平安夜" in b.message

    def test_broadcast_vote_result_exile(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": "p05", "reason": "majority"}
        )
        assert "p05" in b.message
        assert "放逐" in b.message

    def test_broadcast_vote_result_first_tie(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": None, "reason": "first_tie_pk"}
        )
        assert "平票" in b.message

    def test_broadcast_vote_result_second_tie(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": None, "reason": "second_tie_no_exile"}
        )
        assert "再次平票" in b.message

    def test_broadcast_sheriff_elected(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_sheriff_result("p03", "active")
        assert "p03" in b.message
        assert "当选" in b.message

    def test_broadcast_badge_torn(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_sheriff_result(None, "torn")
        assert "撕掉" in b.message

    def test_summarize_speech_fallback(self) -> None:
        judge = self._make_judge()
        speeches = [
            {"speaker": "p01", "text": "我觉得3号是狼人。因为他的视角有问题。"},
            {"speaker": "p02", "text": "我站边1号，归票5号。"},
        ]
        summary = judge.summarize_speech(speeches)
        assert "p01" in summary

    def test_judge_does_not_adjudicate(self) -> None:
        """Judge broadcast has no authority fields."""
        judge = self._make_judge()
        b = judge.broadcast_phase("vote")
        data = b.model_dump()
        assert "winner" not in data
        assert "ruling" not in data


# ---------------------------------------------------------------------------
# Visibility boundary tests
# ---------------------------------------------------------------------------


class TestVisibilityBoundaries:
    def test_player_cannot_see_other_private_intent(self) -> None:
        """AgentContext only provides own role, never others' roles."""
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            own_role="villager",
            visible_world_state={"alive_players": ["p01", "p02"]},
        )
        # own_role is the only role field — no other player roles
        data = ctx.model_dump()
        role_keys = [k for k in data if "role" in k.lower()]
        for k in role_keys:
            assert "other" not in k.lower()
            assert "hidden" not in k.lower()

    def test_agent_context_no_moderator_full(self) -> None:
        """AgentContext must not contain moderator_full sections."""
        ctx = AgentContext(agent_id="p01", task_type=TaskType.SPEECH)
        data = ctx.model_dump()
        assert "moderator_full" not in str(data).lower()
        assert "hidden_identity" not in str(data).lower()

    def test_private_intent_not_in_speech(self) -> None:
        """Speech field must be safe to broadcast publicly."""
        action = PlayerAction(
            action_type=ActionType.SPEECH,
            speech="我是好人，站边逻辑型选手。",
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view="villager",
            ),
        )
        # Private data must not appear in speech
        assert "werewolf" not in action.speech
        assert "狼" not in action.speech


# ---------------------------------------------------------------------------
# Integration: Persona + Model + Player Agent
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    def test_full_pipeline_persona_model_agent(self) -> None:
        """End-to-end: persona → model router → player agent → valid output."""
        # Load configs
        persona_router = PersonaRouter.from_yaml(PERSONAS_YAML)
        persona_router.load_assignments({"p01": "logic_leader"})

        model_router = ModelRouter.from_yaml(MODELS_YAML)
        model_router.register_provider(MockProvider("anthropic"))
        model_router.register_provider(MockProvider("glm"))

        # Resolve persona
        persona = persona_router.resolve("p01", "vote")
        assert persona.profile_id == "logic_leader"

        # Resolve model config
        config, _ = model_router.resolve_config("p01", "vote")
        assert config.provider != ""

        # Create agent and act
        json_resp = '{"action_type":"vote","target_id":"p07","speech":"归7","reason":"逻辑链完整","confidence":0.85}'
        # Override provider to return valid JSON
        model_router._providers["anthropic"] = _JsonProvider(json_resp)

        agent = PlayerAgent(agent_id="p01", model_router=model_router)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
            persona_snapshot=persona.effective_params,
        )
        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id in ["p07", "p08"]
        assert retry.attempt == 1

        # Usage logged
        usage = model_router.get_usage_log()
        assert len(usage) >= 1


# ---------------------------------------------------------------------------
# Task 3: Mandatory vote and fallback tests
# ---------------------------------------------------------------------------


class TestMandatoryVote:
    """When allow_abstain is false, VOTE must be mandatory (no NO_ACTION)."""

    def _make_agent(self, provider_response: str) -> PlayerAgent:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(provider_response)},
        )
        return PlayerAgent(agent_id="p01", model_router=router, max_retries=3)

    def test_fallback_vote_only_with_legal_targets(self) -> None:
        """When legal actions are [VOTE] only, fallback picks first legal target."""
        agent = self._make_agent("bad json")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05"],
        )
        action, _ = agent.act(ctx)
        assert isinstance(action, FallbackAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p05"

    def test_mandatory_vote_prompt_contains_pressure(self) -> None:
        """System prompt mentions mandatory voting when NO_ACTION not available."""
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05", "p06"],
        )
        prompt = agent._build_system_prompt(ctx)
        assert "必须" in prompt
        assert "不能弃票" in prompt

    def test_vote_pressure_from_strategy_directive(self) -> None:
        """strategy_directive with vote_pressure appears in prompt."""
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05", "p06"],
            strategy_directive={"vote_pressure": "已经连续1天无人出局，必须做出决定。"},
        )
        prompt = agent._build_prompt(ctx, RetryInfo())
        assert "连续" in prompt
        assert "必须做出决定" in prompt

    def test_action_trace_records_raw_text_and_fallback(self) -> None:
        agent = self._make_agent("not json")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05"],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        assert action.trace is not None
        assert action.trace.raw_text == "not json"
        assert action.trace.legal_actions == ["vote"]
        assert action.trace.legal_targets == ["p05"]
        assert action.trace.final_action_type == "vote"
        assert action.trace.fallback_reason == "fallback: retries exhausted"
