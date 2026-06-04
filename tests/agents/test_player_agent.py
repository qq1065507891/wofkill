"""Tests for PlayerAgent retry/fallback, vote quality, speech quality, structured output, and skill handling."""

from __future__ import annotations

import json

from werewolf_agent.agents.schemas import (
    ActionTrace,
    ActionType,
    AgentContext,
    FallbackAction,
    FactionGoal,
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
from werewolf_agent.model_gateway.router import (
    GenerateResult,
    ModelConfig,
    ModelRouter,
    UsageRecord,
)


# ---------------------------------------------------------------------------
# Shared mock providers
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

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        return GenerateResult(
            text=self._response,
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=bool(tool_choice),
            tool_call_name=(tool_choice or {}).get("name", ""),
            usage=UsageRecord(
                agent_id="test", task_type="vote",
                provider=self.name, model=config.model,
            ),
        )


class _SequenceJsonProvider:
    """Provider that returns a sequence of structured tool-call payloads."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "sequence_json_provider"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        self.calls += 1
        self.prompts.append(prompt)
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return GenerateResult(
            text=response,
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=bool(tool_choice),
            tool_call_name=(tool_choice or {}).get("name", ""),
            usage=UsageRecord(
                agent_id="test", task_type="speech",
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


class ToolAwareProvider:
    @property
    def name(self) -> str:
        return "mock"

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        self.calls.append({"tools": tools, "tool_choice": tool_choice})
        return GenerateResult(
            text='{"action_type":"vote","target_id":"p07","speech":"归7","reason":"可疑","confidence":0.8}',
            provider=self.name,
            model=config.model,
            usage=UsageRecord(agent_id="", task_type="", provider=self.name, model=config.model),
        )


class EmptyFailureRouter:
    def __init__(self) -> None:
        self.calls = 0
        self._usage_log: list[UsageRecord] = []

    def resolve_config(self, agent_id: str, task_type: str):
        return ModelConfig(provider="minimax", model="MiniMax-M2.7", allow_text_tool_fallback=False), None

    def generate(self, *args, **kwargs):
        self.calls += 1
        self._usage_log.append(UsageRecord(
            agent_id="p01",
            task_type="speech",
            provider="minimax",
            model="MiniMax-M2.7",
            success=False,
            fallback_reason="primary_failed:ReadTimeout: The read operation timed out",
        ))
        return GenerateResult(text="", provider="minimax", model="MiniMax-M2.7")

    def get_usage_log(self):
        return list(self._usage_log)


class LegacyProvider:
    @property
    def name(self) -> str:
        return "legacy"

    def generate(self, prompt, config, system_prompt=None):
        return GenerateResult(
            text='{"action_type":"vote"}',
            provider=self.name,
            model=config.model,
        )


class ToolProbeProvider:
    @property
    def name(self) -> str:
        return "tool_probe"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        return GenerateResult(
            text='{"action_type":"no_action","target_id":null,"speech":"","reason":"probe","confidence":0.5}',
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=True,
            tool_call_name=(tool_choice or {}).get("name", ""),
        )


class TextProbeProvider:
    @property
    def name(self) -> str:
        return "text_probe"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        return GenerateResult(
            text='{"action_type":"no_action"}',
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=False,
            text_fallback_used=True,
            structured_failure_reason="missing_tool_call",
        )


class TextOnlyProvider:
    @property
    def name(self) -> str:
        return "text_only"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        return GenerateResult(
            text='{"action_type":"vote","target_id":"p07","speech":"","reason":"x","confidence":0.8}',
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=False,
            text_fallback_used=True,
            structured_failure_reason="missing_tool_call",
        )


class TextJsonProvider:
    @property
    def name(self) -> str:
        return "text_json"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        return GenerateResult(
            text='{"action_type":"speech","target_id":null,"speech":"我是好人阵营。我怀疑p02，他的站边没有说清楚，发言前后存在矛盾。我倾向投票p02，并继续对比他的票型。","reason":"补充视角","confidence":0.7}',
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=False,
            text_fallback_used=True,
            structured_failure_reason="missing_tool_call",
        )


class NoToolProvider:
    """Provider that doesn't support tool calls."""
    @property
    def name(self) -> str:
        return "notool"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None, **kwargs):
        # Returns None or raises an error for tool_choice
        if tool_choice:
            raise NotImplementedError("This provider does not support tool_choice")
        return GenerateResult(
            text="plain text response",
            provider=self.name,
            model=config.model,
            usage=UsageRecord(
                agent_id="", task_type="", provider=self.name, model=config.model,
            ),
        )


# ---------------------------------------------------------------------------
# Player Agent retry/fallback tests
# ---------------------------------------------------------------------------


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

    def test_player_agent_requests_tool_call_schema(self) -> None:
        provider = ToolAwareProvider()
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router)

        action, _ = agent.act(self._make_context())

        assert action.action_type == ActionType.VOTE
        assert provider.calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit_player_action",
        }
        tool = provider.calls[0]["tools"][0]
        assert tool["name"] == "submit_player_action"
        assert tool["input_schema"]["properties"]["action_type"]["enum"] == [
            "vote",
            "no_action",
        ]
        assert tool["input_schema"]["properties"]["target_id"]["enum"] == [
            "p07",
            "p08",
            None,
        ]
        assert tool["input_schema"]["properties"]["target_id"]["type"] == [
            "string",
            "null",
        ]

    def test_tool_call_schema_disallows_null_target_when_all_actions_require_target(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07", "p08"],
        )

        tool = agent._player_action_tool(ctx)

        assert tool["input_schema"]["properties"]["action_type"]["enum"] == ["vote"]
        assert tool["input_schema"]["properties"]["target_id"]["enum"] == ["p07", "p08"]

    def test_tool_call_schema_uses_plain_nullable_target_when_no_targets_are_legal(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SHERIFF_REGISTRATION,
            phase="sheriff_election",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
            legal_targets=[],
        )

        target_schema = agent._player_action_tool(ctx)["input_schema"]["properties"]["target_id"]

        assert target_schema["type"] == ["string", "null"]
        assert "enum" not in target_schema

    def test_vote_tool_schema_includes_private_audit_fields(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
        )

        props = agent._player_action_tool(ctx)["input_schema"]["properties"]

        assert "standing_with_seer" in props
        assert "suspect_reason" in props
        assert "not_voting_reason" in props
        assert "private_reason" in props
        assert props["seer_stance"]["enum"] == ["trust", "distrust", "undecided", "no_claim"]
        assert props["vote_basis"]["enum"] == [
            "seer_check",
            "seer_siding",
            "speech_logic",
            "vote_pattern",
            "pressure_test",
            "anti_herd",
            "fallback",
        ]

    def test_mandatory_vote_uses_choice_pipeline_and_assembles_action(self) -> None:
        json_resp = (
            '{"choice":"B","reason":"p08查杀p07后，p07回避核心问题",'
            '"seer_stance":"trust",'
            '"vote_basis":"seer_check",'
            '"standing_with_seer":"p08",'
            '"suspect_reason":"p07没有正面回应查杀逻辑",'
            '"not_voting_reason":"p08有查验信息，p06暂时没有明确狼面",'
            '"private_reason":"我更信p08的预言家线，所以投p07",'
            '"confidence":0.82}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p06", "p07"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p07", "result": "werewolf"},
            ],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert action.reason == "p08查杀p07后，p07回避核心问题"
        assert action.seer_stance == "trust"
        assert action.vote_basis == "seer_check"
        assert action.standing_with_seer == "p08"
        assert action.suspect_reason == "p07没有正面回应查杀逻辑"
        assert action.not_voting_reason
        assert action.private_reason
        assert action.trace is not None
        assert action.trace.parsed_action["choice"] == "B"
        assert retry.error_code is None

    def test_vote_choice_pipeline_repairs_mixed_text_json(self) -> None:
        json_resp = (
            "我先分析一下局势。"
            '{"choice":"A","reason":"p06连续两轮站边摇摆",'
            '"standing_with_seer":"",'
            '"suspect_reason":"p06票型和发言不一致",'
            '"not_voting_reason":"p07被查杀但已有回应，p08是报查验者",'
            '"private_reason":"先投票型更差的p06",'
            '"confidence":0.66}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p06", "p07", "p08"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.target_id == "p06"
        assert action.reason == "p06连续两轮站边摇摆"
        assert retry.error_code is None

    def test_vote_choice_pipeline_repairs_missing_reason_from_brief(self) -> None:
        json_resp = '{"choice":"A","confidence":0.51}'
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p05", "result": "werewolf"},
            ],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.target_id == "p05"
        assert action.reason
        assert action.reason != "未说明"
        assert "p05" in action.reason
        assert action.suspect_reason
        assert action.not_voting_reason
        assert action.private_reason
        assert action.seer_stance == "trust"
        assert action.vote_basis == "seer_check"
        assert retry.error_code is None

    def test_vote_quality_accepts_missing_basis_without_retry(self) -> None:
        """Task 2: Vote with no detectable basis pattern is accepted with fallback.

        Regression for Issue 5 (g_3528592081): 6/6 fallback votes stemmed from
        vote_quality retries caused by the strict basis regex. After relaxing
        ``validate_structured_vote_action`` to default ``vote_basis`` to
        "fallback" when no basis is detected, an "unexplained" vote is now
        accepted on the first attempt without retrying the LLM.
        """
        bad_resp = (
            '{"choice":"A","reason":"未说明",'
            '"seer_stance":"undecided","vote_basis":"fallback",'
            '"standing_with_seer":"","suspect_reason":"未说明",'
            '"not_voting_reason":"未说明","private_reason":"未说明",'
            '"confidence":0.5}'
        )
        # Provider has only one response — if the agent retries, the test
        # would crash with StopIteration. With the relaxed basis check, the
        # vote is accepted on the first call.
        provider = _SequenceJsonProvider([bad_resp])
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=2)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
            strategy_directive={"require_vote_quality": True},
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.target_id == "p07"
        # seer_stance and vote_basis kept from response (both already valid).
        assert action.seer_stance == "undecided"
        assert action.vote_basis == "fallback"
        # No retry: the relaxed basis check accepts the vote on attempt 1.
        assert retry.attempt == 1
        assert provider.calls == 1
        assert retry.error_code is None

    def test_target_choice_pipeline_assembles_wolf_kill(self) -> None:
        json_resp = '{"choice":"B","reason":"p08像预言家，夜里优先刀掉","confidence":0.76}'
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p02",
            task_type=TaskType.NIGHT_ACTION,
            phase="night",
            night_number=2,
            own_role="werewolf",
            legal_actions=[ActionType.WOLF_KILL],
            legal_targets=["p06", "p08"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p05", "result": "werewolf"},
            ],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.WOLF_KILL
        assert action.target_id == "p08"
        assert action.reason == "p08像预言家，夜里优先刀掉"
        assert action.trace is not None
        assert action.trace.parsed_action["choice"] == "B"
        assert retry.error_code is None

    def test_target_choice_pipeline_repairs_mixed_text_for_seer_check(self) -> None:
        json_resp = (
            "我会查验更关键的位置。"
            '{"choice":"A","reason":"p04站边摇摆，查验收益最高","confidence":0.69}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p08",
            task_type=TaskType.NIGHT_ACTION,
            phase="night",
            night_number=2,
            own_role="seer",
            legal_actions=[ActionType.CHECK_ALIGNMENT],
            legal_targets=["p04", "p06"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.CHECK_ALIGNMENT
        assert action.target_id == "p04"
        assert action.reason == "p04站边摇摆，查验收益最高"
        assert retry.error_code is None

    def test_target_choice_pipeline_repairs_missing_reason_for_poison(self) -> None:
        json_resp = '{"choice":"A"}'
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p12",
            task_type=TaskType.NIGHT_ACTION,
            phase="night",
            night_number=3,
            own_role="witch",
            legal_actions=[ActionType.USE_POISON],
            legal_targets=["p09"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p09", "result": "werewolf"},
            ],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.USE_POISON
        assert action.target_id == "p09"
        assert action.reason
        assert action.reason != "未说明"
        assert "p09" in action.reason
        assert retry.error_code is None

    def test_speech_intent_pipeline_assembles_speech_action(self) -> None:
        json_resp = (
            '{"intent":"question_target","target_id":"p07",'
            '"speech":"我想追问p07，为什么你昨天站边p02，今天又回避p08的查验？这个变化需要解释。",'
            '"reason":"围绕p07的站边变化施压","confidence":0.73}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07", "p08"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.SPEECH
        assert action.target_id == "p07"
        assert "p07" in action.speech
        assert action.reason == "围绕p07的站边变化施压"
        assert action.trace is not None
        assert action.trace.parsed_action["intent"] == "question_target"
        assert retry.error_code is None

    def test_speech_intent_pipeline_repairs_mixed_text_json(self) -> None:
        json_resp = (
            "我先组织一下发言。"
            '{"intent":"stand_with_seer","target_id":"p08",'
            '"speech":"我目前更站边p08，因为他的查验和昨天票型能对上；p02需要解释警徽流和查杀逻辑。",'
            '"reason":"表达站边并要求对跳方回应","confidence":0.68}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02", "p08"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.SPEECH
        assert action.target_id == "p08"
        assert "p08" in action.speech
        assert retry.error_code is None

    def test_speech_intent_pipeline_synthesizes_missing_speech(self) -> None:
        json_resp = '{"intent":"question_target","target_id":"p05","confidence":0.52}'
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p05"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p05", "result": "werewolf"},
            ],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.speech
        assert action.speech != "(未发言)"
        assert "p05" in action.speech
        assert action.reason
        assert retry.error_code is None

    def test_werewolf_speech_intent_quality_does_not_force_good_stance(self) -> None:
        json_resp = (
            '{"intent":"question_target","target_id":"p05",'
            '"speech":"我想追问p05，你昨天的站边和今天的投票目标没有对上。",'
            '"reason":"继续给p05压力","confidence":0.62}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p04",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="werewolf",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p05"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert "我是好人视角" not in action.speech
        assert "我是p04视角" in action.speech
        assert retry.error_code is None

    def test_speech_tool_schema_omits_private_audit_fields(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        props = agent._player_action_tool(ctx)["input_schema"]["properties"]

        assert "private_intent" not in props
        assert "standing_with_seer" not in props
        assert "suspect_reason" not in props
        assert "not_voting_reason" not in props
        assert "private_reason" not in props

    def test_night_action_tool_schema_omits_private_intent(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.CHECK_ALIGNMENT],
            legal_targets=["p07"],
        )

        props = agent._player_action_tool(ctx)["input_schema"]["properties"]

        assert "private_intent" not in props
        assert "action_type" in props
        assert "target_id" in props
        assert "reason" in props

    def test_vote_audit_fields_are_preserved_in_trace(self) -> None:
        json_resp = (
            '{"action_type":"vote","target_id":"p07","speech":"","reason":"投7",'
            '"confidence":0.8,"standing_with_seer":"p03",'
            '"suspect_reason":"p07站边摇摆",'
            '"not_voting_reason":"p08证据不足",'
            '"private_reason":"我更信p03，所以投p07"}'
        )
        agent = self._make_agent(json_resp)

        action, _ = agent.act(self._make_context())

        assert isinstance(action, PlayerAction)
        assert action.trace is not None
        assert action.trace.parsed_action["standing_with_seer"] == "p03"
        assert action.trace.parsed_action["suspect_reason"] == "p07站边摇摆"

    def test_weak_day_speech_retries_with_quality_hint(self) -> None:
        provider = _SequenceJsonProvider([
            '{"action_type":"speech","target_id":null,"speech":"再观察一下，先听后面",'
            '"reason":"先观察","confidence":0.4}',
            '{"action_type":"speech","target_id":null,'
            '"speech":"我是好人阵营。我怀疑p07，p07发言前后矛盾，票型也不合理。我倾向投p07。",'
            '"reason":"补充明确对象和依据","confidence":0.7}',
        ])
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=2)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert provider.calls == 2
        assert retry.attempt == 2
        assert action.speech.startswith("我是好人阵营")
        assert "发言过于空洞" in provider.prompts[1]

    def test_weak_sheriff_speech_retries_with_high_pressure_hint(self) -> None:
        provider = _SequenceJsonProvider([
            '{"action_type":"speech","target_id":null,"speech":"我竞选警长，大家听我发言。",'
            '"reason":"争取警徽","confidence":0.5}',
            '{"action_type":"speech","target_id":"p07",'
            '"speech":"我是预言家视角。我怀疑p07，p07发言前后矛盾，且警徽流安排不合理。我倾向投p07，并会对比他的票型。",'
            '"reason":"警上补充角色逻辑和攻击点","confidence":0.75}',
        ])
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=2)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SHERIFF_SPEECH,
            phase="sheriff_speech",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert provider.calls == 2
        assert retry.attempt == 2
        assert "警徽流" in action.speech
        assert "在警上/PK阶段需要包含" in provider.prompts[1]

    def test_invalid_json_triggers_retry(self) -> None:
        agent = self._make_agent("not json at all")
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "parse_error"

    def test_illegal_action_triggers_retry(self) -> None:
        json_resp = '{"action_type":"wolf_kill","target_id":"p07","speech":"test","reason":"test","confidence":0.5}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        # wolf_kill not in legal_actions -> retry -> eventually fallback
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

    def test_provider_failure_empty_response_does_not_retry_three_model_calls(self) -> None:
        router = EmptyFailureRouter()
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)

        action, retry = agent.act(self._make_context())

        assert isinstance(action, FallbackAction)
        assert router.calls == 1
        assert retry.error_code == "model_generation_failed"
        assert action.trace is not None
        assert action.trace.retry_count == 1

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

    def test_invalid_private_intent_risk_flags_do_not_fail_main_action(self) -> None:
        json_resp = (
            '{"action_type":"no_action","target_id":null,'
            '"speech":"我不上警，先听警上发言。",'
            '"reason":"村民没有额外信息，先观察警上格局","confidence":0.6,'
            '"private_intent":{"true_role":"villager","faction_goal":"find_wolves",'
            '"claimed_view":"普通村民","pressure_target":null,'
            '"risk_flags":["未参与竞选可能被视为不敢发言"]}}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SHERIFF_REGISTRATION,
            phase="sheriff_election",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.NO_ACTION
        assert action.private_intent is not None
        assert action.private_intent.risk_flags == []
        assert retry.error_code is None

    def test_extracts_nested_json_object_from_reasoning_text(self) -> None:
        json_resp = (
            "我先分析局势：当前没有公开信息，应该保守。\n"
            "{\n"
            '  "action_type": "speech",\n'
            '  "target_id": null,\n'
            '  "speech": "我是好人阵营。我怀疑p07，p07发言前后矛盾。我倾向投p07。",\n'
            '  "reason": "补充明确对象和依据",\n'
            '  "confidence": 0.7,\n'
            '  "private_intent": {\n'
            '    "true_role": "villager",\n'
            '    "faction_goal": "find_wolves",\n'
            '    "claimed_view": "好人视角",\n'
            '    "pressure_target": "p07",\n'
            '    "risk_flags": ["low_trust"]\n'
            "  }\n"
            "}\n"
            "以上是我的行动。"
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.SPEECH
        assert action.private_intent is not None
        assert action.private_intent.pressure_target == "p07"
        assert retry.error_code is None

    def test_extracts_action_json_when_text_contains_other_braces(self) -> None:
        json_resp = (
            '调试信息: {"note": "not an action"}\n'
            "我会输出行动，发言中可能引用符号{重点}。\n"
            "{\n"
            '  "action_type": "speech",\n'
            '  "target_id": null,\n'
            '  "speech": "我是好人阵营。我怀疑p07，p07发言前后矛盾。我倾向投p07。这里的{重点}是票型。",\n'
            '  "reason": "补充明确对象和依据",\n'
            '  "confidence": 0.7\n'
            "}"
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert "{重点}" in action.speech
        assert retry.error_code is None

    def test_code_fence_stripping(self) -> None:
        json_resp = '```json\n{"action_type":"vote","target_id":"p07","speech":"test","reason":"test","confidence":0.5}\n```'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, PlayerAction)
        assert retry.attempt == 1

    def test_minimax_tool_wrapper_parameters_parse(self) -> None:
        json_resp = (
            '<minimax:tool_call name="submit_player_action">'
            '<parameters>{"action_type":"vote","target_id":"p07",'
            '"speech":"归票p07","reason":"p07发言矛盾","confidence":0.72}'
            '</parameters></minimax:tool_call>'
        )
        agent = self._make_agent(json_resp)

        action, retry = agent.act(self._make_context())

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert retry.attempt == 1

    def test_invoke_tool_wrapper_arguments_parse(self) -> None:
        json_resp = (
            '<invoke name="submit_player_action">'
            '<tool_input>{"action_type":"vote","target_id":"p08",'
            '"speech":"我倾向投p08","reason":"p08票型不合理","confidence":0.66}'
            '</tool_input></invoke>'
        )
        agent = self._make_agent(json_resp)

        action, retry = agent.act(self._make_context())

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p08"
        assert retry.attempt == 1

    def test_minimax_parameter_tags_parse(self) -> None:
        json_resp = (
            '<minimax:tool_call>'
            '<invoke name="submit_player_action">'
            '<parameter name="action_type">vote</parameter>'
            '<parameter name="target_id">p07</parameter>'
            '<parameter name="speech"></parameter>'
            '<parameter name="reason">p07发言回避关键问题</parameter>'
            '<parameter name="confidence">0.71</parameter>'
            '<parameter name="standing_with_seer">p08</parameter>'
            '<parameter name="suspect_reason">p07没有回应查杀逻辑</parameter>'
            '<parameter name="not_voting_reason">p08有查验信息，p06发言更自洽</parameter>'
            '<parameter name="private_reason">综合查验和发言，投p07更合理</parameter>'
            '</invoke>'
            '</minimax:tool_call>'
        )
        agent = self._make_agent(json_resp)

        action, retry = agent.act(self._make_context())

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert action.reason == "p07发言回避关键问题"
        assert action.confidence == 0.71
        assert retry.attempt == 1

    def test_string_null_target_normalizes_to_none(self) -> None:
        json_resp = (
            '{"action_type":"no_action","target_id":"null",'
            '"speech":"我不上警，先听警上发言再判断。",'
            '"reason":"当前信息不足，先观察警上格局","confidence":0.63}'
        )
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SHERIFF_REGISTRATION,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
            legal_targets=[],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.target_id is None
        assert retry.error_code is None

    def test_wolf_examples_use_valid_private_intent_goals(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.NIGHT_ACTION,
            own_role="werewolf",
            legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
            legal_targets=["p05"],
        )

        prompt = agent._build_system_prompt(ctx)

        assert "eliminate_villager" not in prompt
        assert "frame_villager" not in prompt

    def test_sheriff_registration_prompt_prefers_tool_and_legal_examples(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SHERIFF_REGISTRATION,
            phase="sheriff_election",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
        )

        # s10: examples are in user_prompt (dynamic per-task context)
        prompt = agent._build_prompt(ctx, RetryInfo(max_retries=1))

        assert "submit_player_action" in prompt
        assert '"action_type": "sheriff_register"' in prompt
        assert '"action_type": "no_action"' in prompt
        assert "示例输出（投票场景）" not in prompt
        assert "示例输出（发言场景）" not in prompt

    def test_action_prompt_ends_with_strict_output_contract(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02"],
            recent_transcript=[{"speaker": "p02", "text": "我觉得p01可疑。"}],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "最终输出协议" in prompt
        assert "不要输出分析过程" in prompt
        assert prompt.rstrip().endswith("现在提交行动。")

    def test_system_prompt_defines_information_boundaries_and_skill_rules(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
        )

        prompt = agent._build_system_prompt(ctx)

        assert "【信息边界】" in prompt
        assert "知识库提示只是玩法经验，不是当前局发生的事" in prompt
        assert "跨局记忆只是历史经验" in prompt
        assert "【推理方法】" in prompt
        assert "盘狼坑时优先看" in prompt
        # P0-K1: tool-skill policy replaced with pre-injection policy
        assert "【技能与建议】" in prompt
        assert "技能分析不是裁判真相" in prompt

    def test_user_prompt_renders_dynamic_sources_as_separate_sections(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03"],
            public_summary="D2公开票型：p02被多人质疑。",
            visible_world_state={"alive_players": ["p01", "p02", "p03"]},
            salience_items=[{"type": "vote", "target_id": "p02"}],
            rag_hints=[{"type": "rag_hit", "entry_id": "basic_case_vote_bloc_001", "summary": "票型抱团识别"}],
            private_memory_hints={"vote_thoughts": [{"target": "p02", "private_reason": "上一轮跟票异常"}]},
            reflection_memory_hints=[{"role": "villager", "result": "负", "text": "好人失败原因：盲目跟票。"}],
            profile_memory_hint={"summary": "累计3局 · 逻辑6/10"},
            cognition_matrix_hint={"suspects": [{"player": "p02", "trust": 0.25, "faction_read": "wolf_lean"}]},
            strategy_directive={"vote_pressure": "必须投票"},
            skill_analysis_hints={"skill_analyze_wolf_pit": "嫌疑区：p02"},
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "当前局公开事实:" in prompt
        assert "可见状态:" in prompt
        # P0-M1: private_memory section uses "【本局·第N轮·私有记忆】" label.
        assert "【本局·第2轮·私有记忆】" in prompt
        assert "知识库提示:" in prompt
        assert "知识库提示不是当前局事实" in prompt
        assert "跨局反思记忆:" in prompt
        assert "长期能力画像:" in prompt
        assert "我的认知矩阵:" in prompt
        assert "本轮策略指令:" in prompt
        assert "技能分析结果:" in prompt
        assert prompt.index("知识库提示:") < prompt.index("跨局反思记忆:")
        assert prompt.index("跨局反思记忆:") < prompt.index("本轮策略指令:")
        assert "策略建议:" not in prompt

    def test_mandatory_vote_prompt_uses_choice_schema(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p06", "p07"],
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p07", "result": "werewolf"},
            ],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "投票候选枚举" in prompt
        assert "A = p06" in prompt
        assert "B = p07" in prompt
        assert '"choice"' in prompt
        assert "不要直接编写target_id" in prompt
        assert "必填字段：action_type、target_id、speech、reason、confidence" not in prompt
        assert (
            "最终输出字段：choice、reason、seer_stance、vote_basis、standing_with_seer、suspect_reason、"
            "not_voting_reason、private_reason、confidence"
        ) in prompt

    def test_target_action_prompt_uses_choice_schema(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p08",
            task_type=TaskType.NIGHT_ACTION,
            phase="night",
            night_number=2,
            own_role="seer",
            legal_actions=[ActionType.CHECK_ALIGNMENT],
            legal_targets=["p04", "p06"],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "目标候选枚举" in prompt
        assert "A = p04" in prompt
        assert "B = p06" in prompt
        assert '"choice"' in prompt
        assert "程序会把choice映射为target_id" in prompt
        assert "必填字段：action_type、target_id、speech、reason、confidence" not in prompt
        assert "最终输出字段：choice、reason、confidence" in prompt

    def test_speech_prompt_uses_intent_schema(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p05", "p08"],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "发言意图枚举" in prompt
        assert "question_target" in prompt
        assert "stand_with_seer" in prompt
        assert "info_synthesis" in prompt
        assert "anti_herd_call" in prompt
        assert '"intent"' in prompt
        assert '"speech"' in prompt
        assert "必填字段：action_type、target_id、speech、reason、confidence" not in prompt
        assert "最终输出字段：intent、target_id、speech、reason、confidence" in prompt

    def test_vote_prompt_warns_good_players_against_herd_voting(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p05", "p08"],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "反跟票警告" in prompt
        assert "不要无条件跟随任何人的归票" in prompt
        assert "狼人抱团" in prompt
        assert "独立判断优先级" in prompt

    def test_action_prompt_trims_long_context_for_json_stability(self) -> None:
        agent = self._make_agent("unused")
        long_text = "甲" * 500
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            visible_world_state={"huge": "乙" * 3000},
            salience_items=[{"type": "event", "idx": i, "detail": "丙" * 200} for i in range(8)],
            recent_transcript=[
                {"speaker": f"p{i:02d}", "text": long_text + str(i)}
                for i in range(8)
            ],
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "p00" not in prompt
        assert "p03" not in prompt
        assert "p04" in prompt
        assert "p07" in prompt
        assert long_text not in prompt
        assert "已截断" in prompt
        assert '"idx": 4' not in prompt


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
        """User prompt mentions mandatory voting when NO_ACTION not available (s10: dynamic)."""
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
        prompt = agent._build_prompt(ctx, RetryInfo(max_retries=1))
        assert "必须" in prompt
        assert "不能弃票" in prompt

    def test_system_prompt_forbids_peace_night_witch_fallacy(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
        )

        prompt = agent._build_system_prompt(ctx)

        assert "平安夜不等于无人被刀" in prompt
        assert "不能用「平安夜没人死」反驳女巫知道刀口" in prompt
        assert "不要跟风复述" in prompt

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
        assert action.trace.fallback_reason is not None
        assert action.trace.fallback_reason.startswith("fallback:")

    def test_vote_fallback_has_audit_reason(self) -> None:
        agent = self._make_agent("not json")
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=3,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p04"],
            visible_world_state={"sheriff_id": "p09"},
            salience_items=[
                {"type": "seer_claim", "speaker": "p08", "target": "p02", "result": "werewolf"},
            ],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        assert action.target_id == "p02"
        assert action.reason
        assert action.reason != "fallback: retries exhausted"
        # Task 1 fix: target must NOT be embedded in the reason string.
        # The audit log records the fallback target via the trace flags.
        assert "p02" not in action.reason
        assert action.trace is not None
        assert action.trace.fallback_target_used is True
        assert action.trace.fallback_target_id == "p02"

    def test_good_speech_fallback_marks_no_effective_public_speech(self) -> None:
        agent = self._make_agent("")
        ctx = AgentContext(
            agent_id="p06",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=3,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02"],
            visible_world_state={"sheriff_id": "p09"},
            salience_items=[
                {"type": "player_died", "player_id": "p08", "reason": "wolf_kill"},
                {"type": "vote_resolved", "exiled": "p01"},
            ],
            recent_transcript=[
                {"speaker": "p09", "text": "我建议今天重点看p02的身份。"},
            ],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        # SPEECH fallback now produces template content instead of "未发表有效言论"
        assert action.speech  # non-empty
        assert "[p06" not in action.speech  # no longer uses bracket marker
        assert "未发表有效言论" not in action.speech


class TestSpeechQualityAndWolfAssignments:
    def test_werewolf_speech_prompt_uses_team_assignment(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"w3": "default"},
            providers={"mock": _JsonProvider("unused")},
        )
        agent = PlayerAgent(agent_id="w3", model_router=router)
        ctx = AgentContext(
            agent_id="w3",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="werewolf",
            legal_actions=[ActionType.SPEECH],
            visible_world_state={
                "wolf_team_plan": {
                    "fake_seer": "w1",
                    "pusher": "w2",
                    "hooker": "w3",
                    "deep_cover": "w4",
                    "day_push_target": "p08",
                    "public_story": "倒钩位轻踩队友，冲锋位打p08。",
                }
            },
        )

        prompt = agent._build_prompt(ctx, RetryInfo())

        assert "hooker" in prompt
        assert "倒钩" in prompt
        assert "p08" in prompt

    def test_good_speech_fallback_contains_failure_marker(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02", "p03"],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        assert action.action_type == ActionType.SPEECH
        assert action.speech  # non-empty template fallback
        assert "未发表有效言论" not in action.speech
        assert action.reason

    def test_good_speech_fallback_varies_only_by_player_marker(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default", "p02": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p03", "p04"],
        )
        first, _ = PlayerAgent(agent_id="p01", model_router=router, max_retries=1).act(ctx)
        second, _ = PlayerAgent(agent_id="p02", model_router=router, max_retries=1).act(
            ctx.model_copy(update={"agent_id": "p02", "day_number": 3})
        )

        assert first.speech != second.speech
        # Both use template fallbacks with different content (varied by salt)
        assert first.speech  # non-empty
        assert second.speech  # non-empty
        assert "未发表有效言论" not in first.speech
        assert "未发表有效言论" not in second.speech

    def test_wolf_discussion_fallback_keeps_werewolf_private_perspective(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p04": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        agent = PlayerAgent(agent_id="p04", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p04",
            task_type=TaskType.WOLF_DISCUSSION,
            phase="night",
            night_number=1,
            own_role="werewolf",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p01", "p02"],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        assert action.action_type == ActionType.SPEECH
        assert "好人阵营" not in action.speech
        assert "狼队" in action.speech or "刀" in action.speech
        assert "[FALLBACK]" in action.speech
        # Hash-based target: any legal target may be selected, not just legal_targets[0]
        assert any(t in action.speech for t in ("p01", "p02"))

    def test_parse_error_retry_hint_is_actionable(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p03"],
        )

        _action, retry = agent.act(ctx)

        assert "只输出JSON" in retry.correction_hint
        assert "action_type" in retry.correction_hint

    def test_prompt_requires_public_record_grounding(self) -> None:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider("unused")},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02", "p03"],
        )

        system_prompt = agent._build_system_prompt(ctx)

        assert "公开记录" in system_prompt
        assert "不要编造" in system_prompt


# ---------------------------------------------------------------------------
# Task 7: Witch No-Poison Explanation
# ---------------------------------------------------------------------------


class TestWitchNoPoisonMustExplain:
    """Witch must explain why not poisoning when pressure targets exist."""

    def test_witch_no_poison_must_explain_pressure_targets(self) -> None:
        """If witch has poison and selects no_action while pressure targets exist,
        reason must explain. The validation function is reused -- empty reason is
        always invalid."""
        from werewolf_agent.runtime.vote_quality import validate_vote_reason

        action = {
            "action_type": "no_action",
            "target_id": None,
            "reason": "",
            "speech": "",
        }
        # Empty reason is invalid regardless of context
        result = validate_vote_reason(action, context={"has_pressure_targets": True})
        assert result["valid"] is False
        assert result.get("missing_basis") is True

    def test_witch_no_poison_with_explanation_passes(self) -> None:
        """Witch no_action with a clear reason passes validation."""
        from werewolf_agent.runtime.vote_quality import validate_vote_reason

        action = {
            "action_type": "no_action",
            "target_id": None,
            "reason": "查杀目标可能是假预言家，保留毒药观察一轮",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_pressure_targets": True})
        assert result["valid"] is True
        assert result.get("missing_basis") is False

    def test_witch_poison_action_with_target_passes(self) -> None:
        """Witch poison action targeting a suspect passes validation."""
        from werewolf_agent.runtime.vote_quality import validate_vote_reason

        action = {
            "action_type": "use_poison",
            "target_id": "p03",
            "reason": "被预言家查杀，毒掉确认狼人",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_pressure_targets": True})
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Task 9: Strict Tool-Call Structured Output
# ---------------------------------------------------------------------------


class TestPlainTextRejection:
    """When tool_choice is set, plain text output must be tracked as failure."""

    def test_plain_text_recorded_in_trace(self):
        """Provider output without tool/function call should be tracked in trace."""
        # _JsonProvider returns plain text JSON. In production the provider
        # would raise RuntimeError when tool_choice is set but no tool_call
        # arrives. Here we test the trace metadata is populated correctly
        # when the agent successfully parses a response.
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(
                '{"action_type": "vote", "target_id": "p07", '
                '"speech": "test", "reason": "test", "confidence": 0.8}'
            )},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)

        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
        )

        action, retry_info = agent.act(context)

        # Agent should produce a valid action
        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        # Trace should have structured output metadata
        assert action.trace is not None
        assert action.trace.tool_call_required is True
        assert action.trace.tool_call_received is True
        assert action.trace.parse_success is True
        assert action.trace.tool_call_name == "submit_player_action"

    def test_plain_text_parse_failure_recorded_in_trace(self):
        """When provider returns unparseable plain text, trace records failure."""
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider("not valid json")},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)

        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
        )

        action, retry_info = agent.act(context)

        # Should get fallback action
        assert isinstance(action, FallbackAction)
        assert action.trace is not None
        assert action.trace.tool_call_required is True
        assert action.trace.parse_success is False
        assert action.trace.parse_error is not None
        assert action.trace.retry_count == 1

    def test_missing_tool_call_does_not_parse_text_json(self):
        """If tool_choice was required but no tool call arrived, do not treat text JSON as success."""
        router = ModelRouter(
            model_profiles={"text": {"model": "text-v1", "provider": "text_only"}},
            llm_profiles={"default": {"default": {"provider": "text_only", "model_profile": "text"}}},
            player_assignments={"p01": "default"},
            providers={"text_only": TextOnlyProvider()},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
        )

        action, retry_info = agent.act(context)

        assert isinstance(action, FallbackAction)
        assert retry_info.error_code == "missing_tool_call"
        assert action.trace is not None
        assert action.trace.structured_failure_reason == "missing_tool_call"
        assert action.trace.parse_error == "missing required tool call: submit_player_action"
        assert action.trace.tool_call_received is False

    def test_configured_text_tool_fallback_parses_plain_json(self):
        """Some compatible gateways return JSON text instead of tool_calls."""
        router = ModelRouter(
            model_profiles={
                "text": {
                    "model": "text-v1",
                    "provider": "text_json",
                    "allow_text_tool_fallback": True,
                },
            },
            llm_profiles={"default": {"default": {"provider": "text_json", "model_profile": "text"}}},
            player_assignments={"p01": "default"},
            providers={"text_json": TextJsonProvider()},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02"],
        )

        action, retry_info = agent.act(context)

        assert isinstance(action, PlayerAction)
        assert "p02" in action.speech
        assert action.trace is not None
        assert action.trace.tool_call_received is False
        assert action.trace.parse_success is True
        assert retry_info.error_code is None


class TestProviderCapabilityFailure:
    """Provider without tool call support should fail explicitly."""

    def test_provider_without_tool_call_support_fails_explicitly(self):
        """When provider cannot enforce tool calls, action should fail with clear error."""
        router = ModelRouter(
            model_profiles={"notool_model": {"model": "notool-v1", "provider": "notool"}},
            llm_profiles={"default": {"default": {"provider": "notool", "model_profile": "notool_model"}}},
            player_assignments={"p01": "default"},
            providers={"notool": NoToolProvider()},
        )
        validator = DefaultActionValidator()
        agent = PlayerAgent(agent_id="p01", model_router=router, validator=validator)

        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
        )

        # Should get a fallback action, not an unhandled exception
        action, retry_info = agent.act(context)
        assert action is not None
        # Should be a fallback with structured_failure_reason
        assert isinstance(action, FallbackAction)
        assert action.trace is not None
        assert action.trace.structured_failure_reason == "structured_output_unsupported"
        assert action.trace.tool_call_required is True
        assert action.trace.tool_call_received is False


class TestStructuredOutputMetadata:
    """Action trace records structured output metadata."""

    def test_trace_records_tool_call_status(self):
        """ActionTrace should record whether tool_call was required and received."""
        trace = ActionTrace(
            raw_text="",
            parsed_action=None,
            final_action_type="vote",
            legal_actions=["vote"],
            legal_targets=["p07"],
            tool_call_required=True,
            tool_call_received=True,
            parse_success=True,
        )
        assert trace.final_action_type == "vote"
        assert trace.tool_call_required is True
        assert trace.tool_call_received is True
        assert trace.parse_success is True

    def test_trace_defaults_to_false(self):
        """New fields default to False/empty."""
        trace = ActionTrace()
        assert trace.tool_call_required is False
        assert trace.tool_call_received is False
        assert trace.tool_call_name == ""
        assert trace.parse_success is False
        assert trace.parse_error is None
        assert trace.retry_count == 0
        assert trace.structured_failure_reason is None

    def test_successful_action_has_complete_trace(self):
        """A successful action path populates all structured output fields."""
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(
                '{"action_type":"vote","target_id":"p07",'
                '"speech":"归7","reason":"可疑","confidence":0.8}'
            )},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.trace is not None
        assert action.trace.tool_call_required is True
        assert action.trace.tool_call_received is True
        assert action.trace.parse_success is True
        assert action.trace.tool_call_name == "submit_player_action"
        assert action.trace.retry_count == 1

    def test_fallback_action_has_failure_trace(self):
        """A fallback action records structured output failure metadata."""
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=2)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
        )

        action, _ = agent.act(ctx)

        assert isinstance(action, FallbackAction)
        assert action.trace is not None
        assert action.trace.tool_call_required is True
        assert action.trace.parse_success is False
        assert action.trace.parse_error is not None
        assert action.trace.retry_count == 2


# === Task 12: Speech Must Answer Contradiction ===

class TestSpeechMustAnswerVisibleContradictionAlert:
    """When contradiction alerts exist, speech must address them."""

    def test_speech_must_answer_visible_contradiction_alert(self):
        """Speech validation fails when high-priority alerts are ignored."""
        from werewolf_agent.runtime.speech_quality import validate_public_speech

        # Context has contradiction alerts
        context = {
            "must_address_alerts": [
                {
                    "alert_type": "claim_conflict",
                    "players": ["p01", "p05"],
                    "description": "p01和p05对跳预言家",
                    "required_response": ["question", "side_with", "park"],
                },
            ],
        }

        # Speech that ignores the contradiction (talks about p03 only)
        speech = "我觉得p03有问题，投p03。p03发言矛盾。"
        result = validate_public_speech(speech, phase="day_discussion", context=context)

        # Should fail because p01/p05 counterclaim is not addressed
        assert result["valid"] is False
        assert "contradiction_alert" in result.get("missing_fields", [])

    def test_speech_addressing_contradiction_passes(self):
        """Speech that addresses the contradiction should pass."""
        from werewolf_agent.runtime.speech_quality import validate_public_speech

        context = {
            "must_address_alerts": [
                {
                    "alert_type": "claim_conflict",
                    "players": ["p01", "p05"],
                    "description": "p01和p05对跳预言家",
                    "required_response": ["question", "side_with", "park"],
                },
            ],
        }

        speech = "p01和p05对跳预言家，我站p01这边。我怀疑p03是狼人，投p03。"
        result = validate_public_speech(speech, phase="day_discussion", context=context)
        assert result["valid"] is True


class TestVoteFallbackConsistency:
    """Vote fallback must not produce a reason mentioning a different target.

    Bug fixed by Task 1: ``_fallback_reason`` embedded the fallback target_id
    (e.g. "p07") into the reason string, while ``agent_day_vote`` later
    overwrote the actual ``vote_target`` with the LLM's choice. The audit log
    then showed a reason referencing a player that the action did not target.
    """

    def test_fallback_reason_does_not_embed_target(self) -> None:
        from werewolf_agent.agents.player import _fallback_reason
        from werewolf_agent.agents.schemas import FallbackAction

        action = FallbackAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            speech="",
            reason="结构化输出失败",
        )
        reason = _fallback_reason(action)
        import re
        assert not re.search(r"p\d{2}", reason), (
            f"fallback reason must not embed target_id, got: {reason!r}"
        )

    def test_action_trace_has_fallback_target_used_flag(self) -> None:
        """ActionTrace must expose fallback_target_used and fallback_target_id."""
        from werewolf_agent.agents.schemas import ActionTrace

        trace = ActionTrace()
        assert hasattr(trace, "fallback_target_used")
        assert hasattr(trace, "fallback_target_id")
        assert trace.fallback_target_used is False
        assert trace.fallback_target_id is None

        marked = ActionTrace(fallback_target_used=True, fallback_target_id="p07")
        assert marked.fallback_target_used is True
        assert marked.fallback_target_id == "p07"


# ---------------------------------------------------------------------------
# Pipeline Optimization Task 1: Smart retry — detect repeated failures
# ---------------------------------------------------------------------------


class TestSmartRetry:
    """Smart retry should early-exit when LLM repeats the same error signature."""

    def test_repeat_error_signature_triggers_early_exit(self) -> None:
        """Same error_code + same raw_text across 2 attempts -> skip remaining retries.

        Scenario: provider returns "not json at all" on every call. Attempt 1
        fails with parse_error; attempt 2 sees an identical (error_code,
        raw_text[:50]) signature and should short-circuit before attempt 3.
        """
        provider = _SequenceJsonProvider(["not json at all"])
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
        )

        action, retry = agent.act(ctx)

        # Early-exit should record the reason on the returned RetryInfo
        assert retry.early_exit_reason is not None
        assert "repeat" in retry.early_exit_reason.lower()
        # Should have stopped at attempt 2 instead of the configured max_retries=3
        assert provider.calls == 2
        # Fallback action is still returned to the caller
        assert isinstance(action, FallbackAction)

    def test_distinct_error_signatures_do_not_early_exit(self) -> None:
        """Different error_code/raw_text across attempts -> use all retries."""
        # 3 different broken responses: first two parse errors with distinct text,
        # third is also broken. All attempts have unique signatures, so no
        # early-exit; fallback fires after the full max_retries=3.
        provider = _SequenceJsonProvider([
            "not json at all",
            "still not json either",
            "completely different garbage",
        ])
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
        )

        action, retry = agent.act(ctx)

        assert retry.early_exit_reason is None
        assert provider.calls == 3
        assert isinstance(action, FallbackAction)
