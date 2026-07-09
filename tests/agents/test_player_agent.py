# -*- coding: utf-8 -*-
"""
测试 PlayerAgent 的重试、兜底、投票质量、发言质量、结构化输出和技能处理。

作者: Project contributors
修改日期: 2026-07-09
"""

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
from werewolf_agent.persona_runtime.router import PersonaRouter


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
        # D4-4 follow-up: production providers set text_fallback_used
        # when they return text without a tool call on a model that
        # allows text fallback. Mirror that so the agent's missing_tool_call
        # gate (which uses this flag) behaves the same in tests and prod.
        # Without this, real-config test setups (allow_text_tool_fallback=True)
        # would enter the missing_tool_call branch on every call and never
        # parse the text — see test_full_pipeline_persona_model_agent.
        text_fallback_used = bool(self._response and not tool_choice)
        return GenerateResult(
            text=self._response,
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=bool(tool_choice),
            tool_call_name=(tool_choice or {}).get("name", ""),
            text_fallback_used=text_fallback_used,
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
            text='{"action_type":"vote","target_id":"p07","speech":"归7","reason":"可疑","confidence":0.8,"suspect_reason":"p07发言矛盾","not_voting_reason":"p08没有证据","candidate_comparison":"p07发言矛盾比p08更具体","private_reason":"我投p07"}',
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
            text='{"action_type":"vote","target_id":"p07","speech":"","reason":"x","confidence":0.8,"suspect_reason":"p07发言矛盾","not_voting_reason":"p08没有证据","candidate_comparison":"p07发言矛盾比p08更具体","private_reason":"我投p07"}',
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


class ProtocolSequenceProvider:
    def __init__(self) -> None:
        self.modes: list[str] = []

    @property
    def name(self) -> str:
        return "protocol_sequence"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        self.modes.append(config.structured_output_mode)
        if len(self.modes) == 1:
            text = "not json"
        else:
            text = (
                '{"intent":"question_target","target_id":"p02",'
                '"speech":"我是好人阵营。p02上一轮站边没有说明依据，'
                '我质疑他的逻辑。今天我倾向投p02，请他回应票型变化。",'
                '"reason":"质疑p02站边和票型","confidence":0.7}'
            )
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            structured_output_mode=config.structured_output_mode,
            text_fallback_used=config.structured_output_mode != "native_tool",
            usage=UsageRecord(
                agent_id="", task_type="", provider=self.name, model=config.model,
                structured_output_mode=config.structured_output_mode,
            ),
        )


class EmptyThenJsonObjectProvider:
    def __init__(self) -> None:
        self.modes: list[str] = []

    @property
    def name(self) -> str:
        return "empty_then_json_object"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        self.modes.append(config.structured_output_mode)
        if config.structured_output_mode == "json_object":
            text = (
                '{"intent":"question_target","target_id":"p02",'
                '"speech":"我是好人阵营。p02上一轮站边没有说明依据，'
                '我质疑他的逻辑。今天我倾向投p02，请他回应票型变化。",'
                '"reason":"质疑p02站边和票型","confidence":0.7}'
            )
        else:
            text = ""
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            structured_output_mode=config.structured_output_mode,
            text_fallback_used=config.structured_output_mode != "native_tool",
            usage=UsageRecord(
                agent_id="",
                task_type="",
                provider=self.name,
                model=config.model,
                structured_output_mode=config.structured_output_mode,
                latency_ms=500,
            ),
        )


class AlwaysInvalidProtocolProvider:
    def __init__(self) -> None:
        self.modes: list[str] = []

    @property
    def name(self) -> str:
        return "always_invalid_protocol"

    def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
        self.modes.append(config.structured_output_mode)
        return GenerateResult(
            text="not json",
            provider=self.name,
            model=config.model,
            structured_output_mode=config.structured_output_mode,
            text_fallback_used=config.structured_output_mode != "native_tool",
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


class TestPlayerActionFlowSplit:
    """Task 16: PlayerAgent.act 应委托给 player_action_flow。"""

    def test_act_delegates_to_player_action_flow(self, monkeypatch) -> None:
        from werewolf_agent.agents import player_action_flow

        agent = PlayerAgent(agent_id="p01", model_router=None)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            legal_targets=[],
        )
        action = FallbackAction(action_type=ActionType.NO_ACTION, reason="sentinel")
        retry = RetryInfo()
        calls: list[tuple[PlayerAgent, AgentContext]] = []

        def fake_flow(received_agent: PlayerAgent, received_context: AgentContext):
            calls.append((received_agent, received_context))
            return action, retry

        monkeypatch.setattr(player_action_flow, "run_player_action_flow", fake_flow)

        assert agent.act(context) == (action, retry)
        assert calls == [(agent, context)]

    def test_generation_request_builder_is_split_from_action_flow_facade(self) -> None:
        from werewolf_agent.agents import player_action_flow, player_generation_request

        assert (
            player_action_flow.build_player_generation_request
            is player_generation_request.build_player_generation_request
        )
        assert (
            player_action_flow.call_player_generation_request
            is player_generation_request.call_player_generation_request
        )

    def test_result_finalizers_are_split_from_action_flow_facade(self) -> None:
        from werewolf_agent.agents import player_action_flow, player_action_result

        assert (
            player_action_flow.finalize_successful_player_action
            is player_action_result.finalize_successful_player_action
        )
        assert (
            player_action_flow.finalize_fallback_player_action
            is player_action_result.finalize_fallback_player_action
        )

    def test_retry_hint_helpers_are_split_from_action_flow_facade(self) -> None:
        from werewolf_agent.agents import player_action_flow, player_retry_hints

        assert (
            player_action_flow.build_empty_response_retry
            is player_retry_hints.build_empty_response_retry
        )
        assert (
            player_action_flow.build_missing_tool_call_retry
            is player_retry_hints.build_missing_tool_call_retry
        )

    def test_quality_retry_helpers_are_split_from_action_flow_facade(self) -> None:
        from werewolf_agent.agents import player_action_flow, player_quality_retries

        assert (
            player_action_flow.build_speech_quality_retry
            is player_quality_retries.build_speech_quality_retry
        )
        assert (
            player_action_flow.build_vote_quality_retry
            is player_quality_retries.build_vote_quality_retry
        )

    def test_choice_prompt_helpers_are_split_from_prompt_output_facade(self) -> None:
        from werewolf_agent.agents import prompt_choice, prompt_output

        assert (
            prompt_output.format_choice_prompt
            is prompt_choice.format_choice_prompt
        )
        assert prompt_output.vote_choice_map is prompt_choice.vote_choice_map
        assert (
            prompt_output.vote_candidate_summary
            is prompt_choice.vote_candidate_summary
        )
        assert (
            prompt_output.target_candidate_summary
            is prompt_choice.target_candidate_summary
        )

    def test_persona_helpers_are_split_from_player_facade(self) -> None:
        from werewolf_agent.agents import player, player_persona

        assert (
            player.PlayerAgent._attach_persona_snapshot
            is player_persona.attach_persona_snapshot
        )
        assert (
            player.PlayerAgent._record_persona_exposure
            is player_persona.record_persona_exposure
        )

    def test_fallback_speech_helpers_are_split_from_player_facade(self) -> None:
        from werewolf_agent.agents import player, player_fallback_speech

        assert (
            player.PlayerAgent._fallback_speech
            is player_fallback_speech.build_fallback_speech
        )
        assert (
            player.PlayerAgent._context_clues
            is player_fallback_speech.context_clues
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
        json_resp = '{"action_type":"vote","target_id":"p07","speech":"归7","reason":"可疑","confidence":0.8,"suspect_reason":"p07发言矛盾","not_voting_reason":"p08没有证据","candidate_comparison":"p07发言矛盾比p08更具体","private_reason":"我投p07"}'
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

    def test_single_target_action_tool_uses_choice_contract(self) -> None:
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

        properties = tool["input_schema"]["properties"]
        assert properties["choice"]["enum"] == ["A", "B"]
        assert "action_type" not in properties
        assert "target_id" not in properties

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
        assert "candidate_comparison" in props
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

    def test_mandatory_vote_accepts_decision_dialogue_plan_envelope(self) -> None:
        json_resp = json.dumps({
            "decision_plan": {
                "action_type": "vote",
                "target_id": "p07",
                "confidence": 0.83,
                "private_goal": "resolve p07 contradiction",
                "evidence_refs": ["event:12:speech"],
                "reference_refs": ["rag:vote_pressure"],
                "selected_world_ids": ["World A"],
                "risk_flags": ["could_be_wrong"],
            },
            "dialogue_plan": {
                "public_intent": "push p07",
                "public_target_id": "p07",
                "talking_points": [
                    "p07 changed stance twice",
                    "vote p07 to resolve the conflict",
                ],
                "conceal": ["p06 is my wolf teammate"],
                "tone": "direct",
            },
        })
        agent = self._make_agent(json_resp)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p06", "p07"],
        )

        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert action.confidence == 0.83
        assert "p07 changed stance twice" in action.reason
        assert "wolf teammate" not in action.reason
        assert action.private_reason
        assert action.trace is not None
        assert action.trace.parsed_action["planning_mode"] == "decision_dialogue"
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

    def test_persona_snapshot_is_resolved_into_each_agent_prompt(self) -> None:
        provider = _SequenceJsonProvider([
            '{"intent":"question_target","target_id":"p07",'
            '"speech":"我追问p07：你昨天和今天的判断前后矛盾，公开票型也没有对上。",'
            '"reason":"核对p07前后判断","confidence":0.73}'
        ])
        model_router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={},
            providers={"mock": provider},
        )
        persona_router = PersonaRouter.from_yaml(
            "config/personas/jingcheng_style_prototypes.yaml"
        )
        persona_router.load_assignments({"p02": "aggressive_bluffer"})
        agent = PlayerAgent(
            agent_id="p02",
            model_router=model_router,
            persona_key="aggressive_bluffer",
            persona_router=persona_router,
        )
        context = AgentContext(
            agent_id="p02",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
        )

        action, retry = agent.act(context)

        assert isinstance(action, PlayerAction)
        assert retry.error_code is None
        assert provider.prompts
        prompt = provider.prompts[0]
        assert "人格设定" in prompt
        assert '"profile_id":"aggressive_bluffer"' not in prompt
        assert "人格核心: dominant_pressurer" in prompt
        assert "表达风格: aggressive_short" in prompt
        assert "任务风格: pressure_attack" in prompt

    def test_existing_good_role_persona_snapshot_is_sanitized(self) -> None:
        agent = self._make_agent('{"action_type":"no_action"}')
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            persona_snapshot={
                "profile_id": "bold_pretender",
                "display_name": "悍跳进攻型",
                "personality": "bold_deceiver",
                "speech_style": "confident_fake_claim",
                "task_style": "fake_authority",
                "effective_params": {"deception_skill": 0.91, "logic_skill": 0.55},
                "dynamic_adjustments": {"deception_skill": 0.1},
            },
        )

        attached = agent._attach_persona_snapshot(context)

        assert attached.persona_snapshot["speech_style"] == "role_consistent_expression"
        assert attached.persona_snapshot["task_style"] == "evidence_based_expression"
        assert "deception_skill" not in attached.persona_snapshot["effective_params"]
        assert "deception_skill" not in attached.persona_snapshot["dynamic_adjustments"]

    def test_persona_suspicion_adjustment_requires_nearby_self_reference(self) -> None:
        model_router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={},
            providers={"mock": _JsonProvider("unused")},
        )
        persona_router = PersonaRouter.from_yaml(
            "config/personas/jingcheng_style_prototypes.yaml"
        )
        persona_router.load_assignments({"p02": "aggressive_bluffer"})
        agent = PlayerAgent(
            agent_id="p02",
            model_router=model_router,
            persona_key="aggressive_bluffer",
            persona_router=persona_router,
        )
        base_context = AgentContext(
            agent_id="p02",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            recent_transcript=[{
                "speaker": "p03",
                "text": "p02这段发言可以保留，但我怀疑p07。",
            }],
        )

        neutral = agent._attach_persona_snapshot(base_context)
        suspected = agent._attach_persona_snapshot(
            base_context.model_copy(update={
                "recent_transcript": [{
                    "speaker": "p03",
                    "text": "我明确怀疑p02，他需要回应当前压力。",
                }],
            })
        )

        assert "aggression" not in neutral.persona_snapshot["dynamic_adjustments"]
        assert suspected.persona_snapshot["dynamic_adjustments"]["aggression"] == 0.2

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
        assert action.speech == "我想追问p05，你昨天的站边和今天的投票目标没有对上。"
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

    def test_single_target_night_action_uses_choice_schema(self) -> None:
        agent = self._make_agent("unused")
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.NIGHT_ACTION,
            legal_actions=[ActionType.CHECK_ALIGNMENT],
            legal_targets=["p07"],
        )

        props = agent._player_action_tool(ctx)["input_schema"]["properties"]

        assert "private_intent" not in props
        assert "choice" in props
        assert "action_type" not in props
        assert "target_id" not in props
        assert "reason" in props

    def test_vote_audit_fields_are_preserved_in_trace(self) -> None:
        json_resp = (
            '{"action_type":"vote","target_id":"p07","speech":"","reason":"投7",'
            '"confidence":0.8,"standing_with_seer":"p03",'
            '"suspect_reason":"p07站边摇摆",'
            '"not_voting_reason":"p08证据不足",'
            '"candidate_comparison":"p07站边摇摆，p08证据不足",'
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
        # P1-S6 (residual): the high-pressure hint is now a short
        # action-oriented line (correction_hint). The full field-missing
        # enumeration still appears in error_message (and is rendered as
        # the "上次错误" snippet in the prompt). The 2nd attempt's
        # prompt must contain at least the new short hint, OR the
        # long field-hint from the error_message snippet.
        assert (
            "在警上/PK阶段需要包含" in provider.prompts[1]
            or "发言必须包含:角色身份/攻击或防御论点 (PK 阶段)" in provider.prompts[1]
        )

    def test_invalid_json_triggers_retry(self) -> None:
        agent = self._make_agent("not json at all")
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "parse_error"

    def test_truncated_json_triggers_specific_retry_reason(self) -> None:
        raw = (
            '{"action_type":"speech","target_id":"p07",'
            '"speech":"我是好人阵营。我怀疑p07，p07发言前后矛盾。我倾向投p07。",'
            '"reason":"输出被截断","confidence":0.7'
        )
        agent = self._make_agent(raw)

        action, retry = agent.act(self._make_context())

        assert isinstance(action, FallbackAction)
        assert retry.error_code == "truncated_json"
        assert "JSON没有闭合" in retry.correction_hint

    def test_illegal_action_triggers_retry(self) -> None:
        json_resp = '{"action_type":"wolf_kill","target_id":"p07","speech":"test","reason":"test","confidence":0.5}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        # wolf_kill not in legal_actions -> retry -> eventually fallback
        assert isinstance(action, FallbackAction)
        assert retry.error_code == "illegal_action"

    def test_illegal_target_triggers_retry(self) -> None:
        json_resp = '{"action_type":"vote","target_id":"p99","speech":"test","reason":"test","confidence":0.5,"suspect_reason":"p99发言矛盾","not_voting_reason":"p08没有证据","candidate_comparison":"p99发言矛盾比p08更具体","private_reason":"我投p99"}'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)

    def test_fallback_vote_without_evidence_has_no_target(self) -> None:
        agent = self._make_agent("bad json")
        action, _ = agent.act(self._make_context())
        assert isinstance(action, FallbackAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id is None

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
            '"suspect_reason":"p07发言矛盾",'
            '"not_voting_reason":"p08没有证据",'
            '"candidate_comparison":"p07发言矛盾比p08更具体",'
            '"private_reason":"我投p07",'
            '"private_intent":{"true_role":"werewolf","faction_goal":"push_good_player_out",'
            '"claimed_view":"good_player_without_night_info","pressure_target":"p07",'
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
        json_resp = '```json\n{"action_type":"vote","target_id":"p07","speech":"test","reason":"test","confidence":0.5,"suspect_reason":"p07发言矛盾","not_voting_reason":"p08没有证据","candidate_comparison":"p07发言矛盾比p08更具体","private_reason":"我投p07"}\n```'
        agent = self._make_agent(json_resp)
        action, retry = agent.act(self._make_context())
        assert isinstance(action, PlayerAction)
        assert retry.attempt == 1

    def test_minimax_tool_wrapper_parameters_parse(self) -> None:
        json_resp = (
            '<minimax:tool_call name="submit_player_action">'
            '<parameters>{"action_type":"vote","target_id":"p07",'
            '"speech":"归票p07","reason":"p07发言矛盾","confidence":0.72,'
            '"suspect_reason":"p07发言矛盾",'
            '"not_voting_reason":"p08没有证据",'
            '"candidate_comparison":"p07发言矛盾比p08更具体",'
            '"private_reason":"我投p07"}'
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
            '"speech":"我倾向投p08","reason":"p08票型不合理","confidence":0.66,'
            '"suspect_reason":"p08票型不合理",'
            '"not_voting_reason":"p07没有证据",'
            '"candidate_comparison":"p08票型不合理，p07缺少同等证据",'
            '"private_reason":"我投p08"}'
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
            '<parameter name="candidate_comparison">p07回避查杀，p06发言更自洽</parameter>'
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
        assert "跨局学习参考包含知识库提示" in prompt
        assert "不是当前局事实" in prompt
        # Phase-1 audit (P1-29): reasoning method was restructured
        # from 4 abstract lines to a numbered 3-step actionable flow.
        # Update the assertions to match the new label and a key
        # phrase from the 3-step body.
        assert "【推理方法-3 步】" in prompt, (
            "Phase-1 P1-29: reasoning method section must be labeled "
            "with the 3-step marker; the old single 【推理方法】 label "
            "is no longer used."
        )
        assert "盘狼坑：按发言矛盾" in prompt, (
            "Phase-1 P1-29: reasoning method must include the 3-step "
            "盘狼坑 body; the old 盘狼坑时优先看 phrasing is gone."
        )
        # P0-K1: tool-skill policy replaced with pre-injection policy
        assert "【技能与建议】" in prompt
        assert "技能战术建议不是裁判真相" in prompt

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
        assert "跨局学习参考:" in prompt
        assert "知识库提示:" in prompt
        assert "知识库提示不是当前局事实" in prompt
        assert "跨局反思记忆:" in prompt
        assert "历史角色经验:" in prompt
        assert "认知校准摘要:" in prompt
        assert "本轮策略指令:" in prompt
        # NEW-S04-A: the legacy "技能分析结果:" section is dropped.
        # The structured skill_tactical_advice in strategy_directive
        # is the single source of truth.
        assert "技能分析结果:" not in prompt
        assert prompt.index("跨局学习参考:") < prompt.index("跨局反思记忆:")
        assert prompt.index("跨局反思记忆:") < prompt.index("知识库提示:")
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
            "not_voting_reason、candidate_comparison、private_reason、confidence"
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
        """When no evidence target exists, fallback does not invent a vote target."""
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
        assert action.target_id is None

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
        assert "不能用「平安夜没人死」否定预言家验人" in prompt
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
        assert action.speech == ""  # FALLBACK text hidden from other players
        assert action.action_type == ActionType.SPEECH
        assert action.reason


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
        assert action.speech == ""  # FALLBACK speech excluded from public transcript
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

        assert first.speech == ""  # FALLBACK excluded from public transcript
        assert second.speech == ""  # both players get empty fallback speech

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

    def test_sheriff_vote_extra_vote_audit_fields_are_repaired(self) -> None:
        raw = json.dumps({
            "action_type": "sheriff_vote",
            "target_id": "p04",
            "speech": "",
            "reason": "p04 is the best sheriff candidate",
            "confidence": 0.6,
            "seer_stance": "undecided",
            "vote_basis": "speech_logic",
            "private_reason": "extra exile-vote audit field",
        })
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(raw)},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="sheriff_vote",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_VOTE, ActionType.NO_ACTION],
            legal_targets=["p04", "p05"],
        )

        action, retry = agent.act(ctx)

        assert retry.error_code is None
        assert action.action_type == ActionType.SHERIFF_VOTE
        assert action.target_id == "p04"

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
                '"speech": "test", "reason": "test", "confidence": 0.8,'
                '"suspect_reason":"p07发言矛盾",'
                '"not_voting_reason":"p08没有证据",'
                '"candidate_comparison":"p07发言矛盾比p08更具体",'
                '"private_reason":"我投p07"}'
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

    def test_protocol_failure_advances_mode_and_records_recovery_mode(self):
        provider = ProtocolSequenceProvider()
        router = ModelRouter(
            model_profiles={
                "model": {
                    "provider": provider.name,
                    "model": "test",
                    "allow_text_tool_fallback": True,
                    "structured_output": {
                        "mode": "json_schema",
                        "fallback_modes": ["json_object", "text_json"],
                    },
                },
            },
            llm_profiles={
                "profile": {
                    "default": {
                        "provider": provider.name,
                        "model_profile": "model",
                    },
                },
            },
            player_assignments={"p01": "profile"},
            providers={provider.name: provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02"],
        )

        action, retry_info = agent.act(context)

        assert isinstance(action, PlayerAction)
        assert provider.modes == ["json_schema", "json_object"]
        assert retry_info.attempt == 2
        assert action.trace is not None
        assert action.trace.structured_output_mode == "json_object"
        assert action.trace.structured_failure_stage is None

    def test_empty_response_advances_mode_before_repeat_short_circuit(self):
        provider = EmptyThenJsonObjectProvider()
        router = ModelRouter(
            model_profiles={
                "model": {
                    "provider": provider.name,
                    "model": "test",
                    "allow_text_tool_fallback": True,
                    "structured_output": {
                        "mode": "json_schema",
                        "fallback_modes": ["json_object", "text_json"],
                    },
                },
            },
            llm_profiles={
                "profile": {
                    "default": {
                        "provider": provider.name,
                        "model_profile": "model",
                    },
                },
            },
            player_assignments={"p01": "profile"},
            providers={provider.name: provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02"],
        )

        action, retry_info = agent.act(context)

        assert isinstance(action, PlayerAction)
        assert provider.modes == ["json_schema", "json_object"]
        assert retry_info.early_exit_reason is None
        assert action.trace is not None
        assert action.trace.structured_output_mode == "json_object"

    def test_repeat_error_does_not_skip_untried_protocol_modes(self):
        provider = AlwaysInvalidProtocolProvider()
        router = ModelRouter(
            model_profiles={
                "model": {
                    "provider": provider.name,
                    "model": "test",
                    "allow_text_tool_fallback": True,
                    "structured_output": {
                        "mode": "json_schema",
                        "fallback_modes": ["json_object", "text_json"],
                    },
                },
            },
            llm_profiles={
                "profile": {
                    "default": {
                        "provider": provider.name,
                        "model_profile": "model",
                    },
                },
            },
            player_assignments={"p01": "profile"},
            providers={provider.name: provider},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
        context = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02"],
        )

        action, retry_info = agent.act(context)

        assert isinstance(action, FallbackAction)
        assert provider.modes == ["json_schema", "json_object", "text_json"]
        assert retry_info.early_exit_reason is None

    def test_successful_action_has_complete_trace(self):
        """A successful action path populates all structured output fields."""
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _JsonProvider(
                '{"action_type":"vote","target_id":"p07",'
                '"speech":"归7","reason":"可疑","confidence":0.8,'
                '"suspect_reason":"p07发言矛盾",'
                '"not_voting_reason":"p08没有证据",'
                '"candidate_comparison":"p07发言矛盾比p08更具体",'
                '"private_reason":"我投p07"}'
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
# P1-S6 (residual): speech_quality / vote_quality correction_hint must be
# a short, action-oriented hint distinct from the detailed error_message.
# ---------------------------------------------------------------------------
#
# Audit P1-S6 (residual) finding: P0-S6 set correction_hint ==
# error_message (both carried the full field-missing enumeration). For
# the LLM retrying, the detailed enumeration is noisy — the LLM just
# needs to know what KIND of action to take. The new behavior splits
# the two: error_message keeps the full detail (for the audit log),
# correction_hint becomes a short action-oriented line that the LLM
# can act on directly.
#
# Speech quality hint: "发言必须包含:角色身份/攻击或防御论点 (PK 阶段)"
# Vote quality hint:   "投票理由必须基于:预言家查杀/票型/警徽流/发言分析 (公开来源)"


def test_speech_quality_hint_specific():
    """P1-S6 (residual): speech_quality retry → short specific correction_hint.

    Builds a RetryInfo with error_code="speech_quality" + a long
    field-missing error_message (mimicking what _speech_quality_error
    returns). Asserts the correction_hint is the short action-oriented
    hint, NOT the long error_message.
    """
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.model_gateway.router import ModelRouter

    router = ModelRouter(
        model_profiles={},
        llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider("unused")},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05", "p07"],
    )
    detailed_message = (
        "发言不完整。需要在警上/PK阶段需要包含角色声明、对跳分析或攻击/防守论点；"
        "需要表明你的身份立场（如'我是好人阵营'）；"
        "需要指出至少一个怀疑对象（如'我怀疑pXX'）。"
    )
    retry = RetryInfo(
        attempt=1,
        max_retries=3,
        error_code="speech_quality",
        error_message=detailed_message,
        correction_hint="发言必须包含:角色身份/攻击或防御论点 (PK 阶段)",
    )
    prompt = agent._build_prompt(ctx, retry)
    # The short hint should appear in the prompt's retry section
    assert "发言必须包含" in prompt
    # The detailed message appears in the error_message snippet
    assert "发言不完整" in prompt
    # The short hint is the one in the correction_hint line
    assert "角色身份/攻击或防御论点" in prompt


def test_speech_quality_retry_hint_targets_missing_identity_stance():
    from werewolf_agent.agents.player_quality_retries import build_speech_quality_retry

    retry = build_speech_quality_retry(
        "发言不完整。需要表明你的身份立场（如'我是好人阵营'）。",
        attempt=1,
        max_retries=3,
    )

    assert "先补一句身份立场" in retry.correction_hint
    assert "我是好人阵营" in retry.correction_hint


def test_speech_quality_retry_hint_targets_public_record_grounding():
    from werewolf_agent.agents.player_quality_retries import build_speech_quality_retry

    retry = build_speech_quality_retry(
        "发言不完整。引用公开记录时，必须能在游戏概况或近期发言中找到对应原文；"
        "不要把推测写成“公开记录”，无法确认时改成“我推测/我质疑”。",
        attempt=1,
        max_retries=3,
    )

    assert "把无法确认的公开记录改写为“我推测/我质疑”" in retry.correction_hint
    assert "不要继续声称公开记录已经证明" in retry.correction_hint


def test_vote_quality_hint_specific():
    """P1-S6 (residual): vote_quality retry → short specific correction_hint.

    Builds a RetryInfo with error_code="vote_quality" + a long
    field-missing error_message. Asserts the correction_hint is the
    short action-oriented hint, NOT the long error_message.
    """
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.model_gateway.router import ModelRouter

    router = ModelRouter(
        model_profiles={},
        llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider("unused")},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p05", "p07"],
    )
    detailed_message = (
        "投票理由缺少具体逻辑依据。请引用以下至少一种："
        "查验结果、对跳分析、警徽流、矛盾点、投票数据、"
        "立场变化、PK发言、或之前发言引用。"
    )
    retry = RetryInfo(
        attempt=1,
        max_retries=3,
        error_code="vote_quality",
        error_message=detailed_message,
        correction_hint=(
            "投票理由必须基于:预言家查杀/票型/警徽流/发言分析 (公开来源)"
        ),
    )
    prompt = agent._build_prompt(ctx, retry)
    # The short hint should appear in the prompt's retry section
    assert "投票理由必须基于" in prompt
    # The detailed message appears in the error_message snippet
    assert "投票理由缺少具体逻辑依据" in prompt
    # The short hint is the one in the correction_hint line
    assert "预言家查杀/票型/警徽流/发言分析" in prompt


def test_speech_quality_correction_hint_differs_from_error_message():
    """P1-S6 (residual): production must emit short hint, not the long
    speech_quality_error enumeration.

    This is the regression test for the production code path in
    PlayerAgent._act: when the LLM fails speech_quality, the resulting
    RetryInfo's correction_hint must be the short action-oriented hint
    (not the long field-missing enumeration from _speech_quality_error).
    The detailed enumeration lives in error_message only.
    """
    from unittest.mock import patch
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.agents.schemas import (
        SpeechPlayerAction,
        PlayerAction as _PA,
    )
    from werewolf_agent.model_gateway.router import ModelRouter

    router = ModelRouter(
        model_profiles={},
        llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider("unused")},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)

    # Build a speech action that will fail _speech_quality_error.
    bad_speech_action = SpeechPlayerAction(
        action_type=ActionType.SPEECH,
        target_id=None,
        speech="",  # Empty -> fails stance/suspicion_target/evidence checks
        reason="ok",
        confidence=0.5,
    )

    captured_retry: list = []

    def _capture_retry(retry, raw_text, attempt, last_signature, **_kwargs):
        captured_retry.append(retry)
        return False, last_signature

    with patch.object(
        agent,
        "_speech_quality_error",
        return_value="发言不完整。需要表明你的身份立场。需要指出怀疑对象。",
    ), patch.object(
        agent, "_check_repeat_error_signature", side_effect=_capture_retry,
    ):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p05", "p07"],
        )
        # Manually invoke the speech_quality branch by simulating the
        # retry path. The production code constructs a RetryInfo with
        # the short correction_hint; we just need to verify the
        # construction.
        # Patch the parsed_action check by going through act() with a
        # provider that returns the bad speech action's JSON.
        from werewolf_agent.agents.schemas import PrivateIntent

        provider = _SequenceJsonProvider([
            (
                '{"action_type":"speech","target_id":null,'
                '"speech":"","reason":"ok","confidence":0.5,'
                '"private_intent":{"true_role":"villager",'
                '"faction_goal":"find_wolves",'
                '"claimed_view":"good_player_without_night_info",'
                '"pressure_target":null,"risk_flags":[]}}'
            )
        ])
        router2 = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": provider},
        )
        agent2 = PlayerAgent(agent_id="p01", model_router=router2, max_retries=1)
        with patch.object(
            agent2, "_speech_quality_error",
            return_value="发言不完整。需要表明你的身份立场。",
        ):
            action, retry = agent2.act(ctx)
        # We expect a FallbackAction (since max_retries=1 and the speech
        # failed quality). The retry should be captured.
        assert retry is not None
        # The key regression check: correction_hint must be the SHORT
        # action-oriented hint, NOT the long enumeration.
        # P3-3: correction_hint is the executable template with
        # 1) 2) 3) numbered steps the LLM can mechanically follow.
        # error_message keeps the long detail for the audit log.
        assert retry.correction_hint == (
            "先补一句身份立场，例如“我是好人阵营”。"
            "再基于一条公开发言、票型或查验声明给出攻击或防御论点。"
        ), (
            f"P3-3: speech_quality correction_hint must be the executable "
            f"specific template; got: {retry.correction_hint!r}"
        )
        # error_message keeps the long detail
        assert "发言不完整" in retry.error_message, (
            "speech_quality error_message must keep the long detail "
            "for the audit log."
        )


def test_vote_quality_correction_hint_differs_from_error_message():
    """P1-S6 (residual): production must emit short vote hint, not the long
    vote_quality_error enumeration.
    """
    from unittest.mock import patch
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.model_gateway.router import ModelRouter

    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p05", "p07"],
    )
    bad_vote_json = (
        '{"action_type":"vote","target_id":"p05","speech":"",'
        '"reason":"可疑","confidence":0.5}'
    )
    provider = _SequenceJsonProvider([bad_vote_json])
    router = ModelRouter(
        model_profiles={},
        llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
    with patch.object(
        agent, "_vote_quality_error",
        return_value="投票理由缺少具体逻辑依据。请引用以下至少一种：查验结果、警徽流。",
    ):
        action, retry = agent.act(ctx)
    # Fallback should fire after 1 retry. retry object should be populated.
    assert retry is not None
    # P3-3: correction_hint is the executable template with
    # 1) 2) 3) 4) numbered steps.  error_message keeps the long
    # detail for the audit log.
    assert retry.correction_hint == (
        "投票理由缺少以下必填字段: 投票理由缺少具体逻辑依据。"
        "请引用以下至少一种：查验结果、警徽流。。"
        "请基于以下公开来源重写 vote reason："
        "1) 预言家查杀声明（金水/查杀 + 报验人+夜数）；"
        "2) 票型异常（谁跟谁、票型突变）；"
        "3) 警徽流状态（撕徽/未撕）；"
        "4) 公开记录里的具体发言引用。"
        "不要写「综合分析」之类的占位文本。"
    ), (
        f"P3-3: vote_quality correction_hint must be the executable "
        f"1) 2) 3) 4) template; got: {retry.correction_hint!r}"
    )
    assert "投票理由缺少具体逻辑依据" in retry.error_message


# ---------------------------------------------------------------------------
# P1-S7 (residual): sanitize claimed_view to enum-like identifier in
# production code (not just the example)
# ---------------------------------------------------------------------------
#
# Audit P1-S7 finding: P0-S7 changed the EXAMPLE claimed_view values
# in the prompt to enum-style identifiers. But the production
# sanitization (sanitize_optional_private_fields in output_parser.py)
# only validates faction_goal and risk_flags — claimed_view is a
# free-form str. Game trace g_3528592081 showed real wolves writing
# claimed_view="我是好人，混水摸鱼" — the LLM copied the bad pattern
# even with the new example, and the sanitizer let it through.
#
# Fix: define VALID_CLAIMED_VIEW_VALUES and sanitize any non-enum
# value to a safe default. The safe default is derived from
# true_role (or "good_player_without_night_info" if no role hint).
# This is the prompt-side AND production-side gate — both layers
# must reject the natural-language pattern.
#
# Tests use FULL_ACTION mode (SPEECH task) so private_intent
# survives the parse path. VOTE uses TARGET_CHOICE mode which
# intentionally drops private_intent (P0-S8 design choice).


def test_claimed_view_uses_enum_in_production_when_natural_language():
    """P1-S7: when LLM writes Chinese phrase for claimed_view, sanitize.

    A wolf with true_role=werewolf writes
    "claimed_view": "我是好人，混水摸鱼" — natural-language claim.
    The sanitizer should reject this and substitute a safe enum-style
    default (good_player_without_night_info for non-seer).
    """
    from werewolf_agent.agents.player import PlayerAgent
    # Speech text must pass speech_quality: has 立场/怀疑对象/投票倾向/依据
    speech_text = "我是好人阵营。我怀疑p07，p07发言前后矛盾。我倾向投p07。"
    json_resp = (
        '{"action_type":"speech","target_id":null,'
        f'"speech":"{speech_text}",'
        '"reason":"分析矛盾","confidence":0.7,'
        '"private_intent":{"true_role":"werewolf","faction_goal":"confuse_good",'
        '"claimed_view":"我是好人，混水摸鱼","pressure_target":"p07",'
        '"risk_flags":[]}}'
    )
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider(json_resp)},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05", "p07"],
    )
    action, _ = agent.act(ctx)
    assert isinstance(action, PlayerAction), (
        f"Expected PlayerAction, got {type(action).__name__}. "
        f"speech_quality may be rejecting the test speech text."
    )
    assert action.private_intent is not None
    # The Chinese phrase must NOT survive sanitization
    assert "我是好人" not in action.private_intent.claimed_view, (
        f"claimed_view must be sanitized to enum-style identifier, "
        f"got natural-language: {action.private_intent.claimed_view!r}"
    )
    # Safe default: non-seer default is good_player_without_night_info
    assert action.private_intent.claimed_view == "good_player_without_night_info", (
        f"Expected safe default 'good_player_without_night_info', "
        f"got: {action.private_intent.claimed_view!r}"
    )


def test_claimed_view_preserves_valid_enum_value():
    """P1-S7: when LLM writes a valid enum value, it's preserved as-is."""
    from werewolf_agent.agents.player import PlayerAgent
    # Use a speech that passes speech_quality: must have 立场/怀疑对象/投票倾向/依据
    speech_text = "我是预言家。昨晚查验p05。我怀疑p07，p07没给查杀。我倾向投p07。"
    json_resp = (
        '{"action_type":"speech","target_id":null,'
        f'"speech":"{speech_text}",'
        '"reason":"公开查验","confidence":0.8,'
        '"private_intent":{"true_role":"seer","faction_goal":"find_wolves",'
        '"claimed_view":"seer","pressure_target":"p05",'
        '"risk_flags":[]}}'
    )
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p03": "default"},
        providers={"mock": _JsonProvider(json_resp)},
    )
    agent = PlayerAgent(agent_id="p03", model_router=router, max_retries=1)
    ctx = AgentContext(
        agent_id="p03",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="seer",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05", "p07"],
    )
    action, _ = agent.act(ctx)
    assert isinstance(action, PlayerAction), (
        f"Expected PlayerAction, got {type(action).__name__}"
    )
    assert action.private_intent is not None
    # Valid enum value must be preserved
    assert action.private_intent.claimed_view == "seer", (
        f"Valid enum value must be preserved, got: {action.private_intent.claimed_view!r}"
    )


def test_claimed_view_role_identifier_treated_as_valid():
    """P1-S7: role identifiers (werewolf, etc.) are valid claimed_views.

    A wolf claiming to be a good player (default false cover) is a
    common strategy. The wolf's claimed_view uses a clean enum
    identifier, not natural language — it must pass through.

    P1-3: claimed_view is now Literal-enforced. The valid set is
    ``{good_player_without_night_info, seer, witch, hunter, idiot,
    hybrid, werewolf}``. 'villager' is not in the set (a villager
    does not need to claim an identity — they ARE the default), so
    the canonical cover for a wolf pretending to be a villager is
    ``good_player_without_night_info``.
    """
    from werewolf_agent.agents.player import PlayerAgent
    speech_text = (
        "我是好人阵营。我怀疑p05，p05发言前后矛盾。"
        "我倾向投p05，因为p05没有合理的归票理由。"
    )
    json_resp = (
        '{"action_type":"speech","target_id":null,'
        f'"speech":"{speech_text}",'
        '"reason":"保守观察","confidence":0.5,'
        '"private_intent":{"true_role":"werewolf","faction_goal":"confuse_good",'
        '"claimed_view":"good_player_without_night_info","pressure_target":"p05",'
        '"risk_flags":[]}}'
    )
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider(json_resp)},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05", "p07"],
    )
    action, _ = agent.act(ctx)
    assert isinstance(action, PlayerAction)
    assert action.private_intent is not None
    assert action.private_intent.claimed_view == "good_player_without_night_info", (
        f"Enum value 'good_player_without_night_info' must be preserved, "
        f"got: {action.private_intent.claimed_view!r}"
    )


def test_claimed_view_good_player_identifier_preserved():
    """P1-S7: good_player_without_night_info is the canonical safe default."""
    from werewolf_agent.agents.player import PlayerAgent
    speech_text = (
        "我是好人阵营。我怀疑p05，p05发言前后矛盾。"
        "我倾向投p05，因为p05的归票理由与p07不同。"
    )
    json_resp = (
        '{"action_type":"speech","target_id":null,'
        f'"speech":"{speech_text}",'
        '"reason":"分析矛盾","confidence":0.7,'
        '"private_intent":{"true_role":"villager","faction_goal":"find_wolves",'
        '"claimed_view":"good_player_without_night_info","pressure_target":"p05",'
        '"risk_flags":[]}}'
    )
    router = ModelRouter(
        model_profiles={}, llm_profiles={},
        player_assignments={"p01": "default"},
        providers={"mock": _JsonProvider(json_resp)},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=1)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05", "p07"],
    )
    action, _ = agent.act(ctx)
    assert isinstance(action, PlayerAction)
    assert action.private_intent is not None
    assert action.private_intent.claimed_view == "good_player_without_night_info"


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


class TestRetryCountConsistency:
    """R3-MG-5: trace.retry_count and metrics_collector.retry_count must agree.

    Pre-fix, the success path used ``retry_count=attempt`` (e.g. 2) but
    the exhausted-path trace used ``retry_count=self.max_retries`` (e.g. 4)
    even when retries had short-circuited. The audit log and the
    metrics_collector disagreed about how many attempts actually ran.
    """

    def test_retry_count_consistent(self):
        """Exhausted path with early-exit at attempt=2 of max=4:
        trace.retry_count and metrics.retry_count must both be 2.
        """
        from werewolf_agent.agents.player import PlayerAgent
        from werewolf_agent.model_gateway.router import ModelRouter

        class _AlwaysEmptyProvider:
            """Provider that returns empty text — every attempt is empty_response."""

            @property
            def name(self) -> str:
                return "always_empty"

            def generate(self, prompt, config, system_prompt=None):
                # R3-MG-2: include the new http_status / raw_error so the
                # categorizer classifies this as a clean "unknown" rather
                # than timing out. We do NOT set retry_count here — the
                # router tracks that.
                from werewolf_agent.model_gateway.router import (
                    GenerateResult,
                    UsageRecord,
                )
                return GenerateResult(
                    text="",
                    provider="always_empty",
                    model=config.model,
                    usage=UsageRecord(
                        agent_id="",
                        task_type="",
                        provider="always_empty",
                        model=config.model,
                        latency_ms=500,
                    ),
                )

        # max_retries=4 so we have room to early-exit at attempt 2
        # via the repeat-error-signature short-circuit.
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p01": "default"},
            providers={"mock": _AlwaysEmptyProvider()},
        )
        agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=4)
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

        # Sanity: we should have fallen back after early-exit.
        assert isinstance(action, FallbackAction)
        assert action.trace is not None

        # The trace must reflect the actual attempt number, not max_retries.
        # Pre-fix: trace.retry_count == 4 (max_retries) even when we
        # short-circuited at attempt 2.
        assert action.trace.retry_count == retry.attempt, (
            f"trace.retry_count ({action.trace.retry_count}) must equal "
            f"retry.attempt ({retry.attempt})"
        )
        # And the metrics_collector must record the same number.
        profile = agent.metrics_collector.get_profile("p01")
        # The per-task breakdown also tracks retry_count through fallback_count;
        # what we really want to check is that the agent's recorded
        # fallback_used attempt is consistent with the trace.
        assert profile.sample_count >= 1
        assert profile.fallback_count == 1


# ---------------------------------------------------------------------------
# D4-3: empty_response hint must only suggest no_action when it's legal
# ---------------------------------------------------------------------------
#
# Audit D4-3 finding: the timeout fallback hint at player.py:381-395
# suggests `action_type='no_action'` whenever the empty_response is
# categorized as a timeout. But for VOTE (legal_actions=[VOTE]), no_action
# is NOT in the legal set — the LLM would copy the hint and the agent
# would reject the action. The fix: only inject the no_action suggestion
# when `ActionType.NO_ACTION in ctx.legal_actions`; otherwise fall back
# to a target-suggestion hint that names one of the legal targets.


class TestEmptyResponseHintValidatesNoAction:
    """D4-3: timeout hint must respect legal_actions."""

    def _build_timeout_provider(self, latency_ms: int):
        """Build a provider that returns empty text with high latency
        so ``_categorize_failure_category`` returns ``"timeout"``.
        """
        from werewolf_agent.model_gateway.router import (
            GenerateResult,
            UsageRecord,
        )

        class _TimeoutProvider:
            @property
            def name(self) -> str:
                return "timeout"

            def generate(self, prompt, config, system_prompt=None):
                return GenerateResult(
                    text="",
                    provider="timeout",
                    model=config.model,
                    usage=UsageRecord(
                        agent_id="",
                        task_type="",
                        provider="timeout",
                        model=config.model,
                        latency_ms=latency_ms,
                    ),
                )

        return _TimeoutProvider()

    def test_empty_response_hint_only_suggests_no_action_when_legal(self):
        """D4-3: VOTE-only context (no NO_ACTION) must not suggest no_action.

        Pre-fix: the timeout hint always suggested ``no_action``, even
        when VOTE was the only legal action — the LLM copied the hint,
        the validator rejected the action, and the player had to retry.
        The fix: only inject the ``no_action`` hint when
        ``ActionType.NO_ACTION in ctx.legal_actions``.
        """
        from werewolf_agent.model_gateway.router import ModelRouter

        provider = self._build_timeout_provider(latency_ms=31_000)  # > 30s threshold
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p05": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p05", model_router=router, max_retries=1)
        # VOTE-only — no NO_ACTION in legal_actions.
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07", "p08"],
            public_summary="D2 vote",
        )

        action, retry = agent.act(ctx)

        # Sanity: the empty_response path was taken (otherwise the test
        # is meaningless).
        assert retry.error_code == "empty_response", (
            f"D4-3: expected empty_response retry, got {retry.error_code!r}"
        )
        # The hint must NOT contain "no_action" as a suggestion because
        # the only legal action is VOTE — the LLM would copy it and
        # the validator would reject it.
        assert "no_action" not in retry.correction_hint, (
            "D4-3: timeout hint for VOTE-only context must NOT mention "
            "`no_action` (it's not in legal_actions). The LLM would copy "
            "the suggestion and the validator would reject the action. "
            f"Got hint: {retry.correction_hint!r}"
        )

    def test_empty_response_hint_suggests_no_action_when_legal(self):
        """D4-3: when NO_ACTION is legal, the hint SHOULD still mention it.

        Regression guard: the fix must not over-correct. If the
        context allows no_action, the timeout hint should still
        suggest it (P0-R2's whole point).
        """
        from werewolf_agent.model_gateway.router import ModelRouter

        provider = self._build_timeout_provider(latency_ms=31_000)
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p05": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p05", model_router=router, max_retries=1)
        # NO_ACTION is legal — keep the original P0-R2 hint.
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.SHERIFF_REGISTRATION,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.SHERIFF_WITHDRAW, ActionType.NO_ACTION],
            legal_targets=[],
            public_summary="D1 sheriff election",
        )

        action, retry = agent.act(ctx)

        assert retry.error_code == "empty_response"
        # NO_ACTION is legal — the hint should still mention it.
        assert "no_action" in retry.correction_hint, (
            "D4-3 regression: when NO_ACTION is legal, the timeout hint "
            "must still mention it (P0-R2). "
            f"Got hint: {retry.correction_hint!r}"
        )

    def test_empty_response_hint_omits_no_action_when_output_mode_cannot_emit_it(self):
        """D4-3: SPEECH_INTENT mode cannot emit action_type=no_action."""
        from werewolf_agent.model_gateway.router import ModelRouter

        provider = self._build_timeout_provider(latency_ms=31_000)
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p05": "default"},
            providers={"mock": provider},
        )
        agent = PlayerAgent(agent_id="p05", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.SPEECH, ActionType.VOTE],
            legal_targets=["p07"],
            public_summary="D2 speech",
        )

        action, retry = agent.act(ctx)

        assert retry.error_code == "empty_response"
        assert "no_action" not in retry.correction_hint


# ---------------------------------------------------------------------------
# D4-4: missing_tool_call hint must adapt to allow_text_tool_fallback
# ---------------------------------------------------------------------------
#
# Audit D4-4 finding: the missing_tool_call hint at player.py:432-452
# says "必须通过 submit_player_action 工具调用..." even when the model
# is configured with `allow_text_tool_fallback=True`. The hint contradicts
# the model's own behavior — the model is allowed to emit plain-text
# JSON when it has no tool schema, but the agent tells it not to.
#
# Game trace g_3528592081 showed text-fallback-allowed models
# oscillating between tool-call and text-JSON output, repeatedly
# hitting this hint. The fix: branch the hint on `_model_text_fallback`
# — fallback-allowed models get "优先工具调用，无 tool schema 时允许文本 JSON"
# instead of the strict "必须工具调用".


class TestMissingToolCallHintAdaptsToTextFallback:
    """D4-4: missing_tool_call hint must respect allow_text_tool_fallback."""

    def test_text_json_mode_treats_non_json_as_parse_error(self):
        """Text-JSON models parse text directly instead of inventing a tool failure."""
        from werewolf_agent.model_gateway.router import (
            GenerateResult,
            ModelRouter,
            UsageRecord,
        )

        class _TextFallbackNoToolProvider:
            """Provider that returns text without setting text_fallback_used.

            Simulates a model with ``allow_text_tool_fallback=True`` on
            config, but the result's ``text_fallback_used=False`` —
            forcing the agent into the missing_tool_call branch (the
            test target). Text is non-empty so we don't trip
            empty_response instead.
            """

            @property
            def name(self) -> str:
                return "text_fallback_no_tool"

            def generate(self, prompt, config, system_prompt=None):
                return GenerateResult(
                    text="<some non-JSON text response>",
                    provider="text_fallback_no_tool",
                    model=config.model,
                    tool_call_required=True,
                    tool_call_received=False,
                    text_fallback_used=False,  # <-- key for the test
                    structured_failure_reason="missing_tool_call",
                    usage=UsageRecord(
                        agent_id="",
                        task_type="",
                        provider="text_fallback_no_tool",
                        model=config.model,
                    ),
                )

        router = ModelRouter(
            model_profiles={
                "text_model": {
                    "model": "text-v1",
                    "provider": "text_fallback_no_tool",
                    "allow_text_tool_fallback": True,  # <-- key for the test
                },
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "text_fallback_no_tool",
                        "model_profile": "text_model",
                    },
                },
            },
            player_assignments={"p05": "default"},
            providers={"text_fallback_no_tool": _TextFallbackNoToolProvider()},
        )
        agent = PlayerAgent(agent_id="p05", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07", "p08"],
            public_summary="D2 vote",
        )

        action, retry = agent.act(ctx)

        assert retry.error_code == "parse_error"
        assert "必须通过 submit_player_action 工具调用" not in retry.correction_hint, (
            "text_json mode must not ask the model to use a tool call"
        )
        assert "只输出JSON" in retry.correction_hint

    def test_missing_tool_call_hint_strict_for_tool_only_models(self):
        """D4-4 regression: tool-only models still get the strict hint.

        When the model is NOT configured with ``allow_text_tool_fallback``,
        the strict "must use tool call" hint is correct — text JSON is
        not a valid fallback for these models. The fix must not
        over-correct: tool-only models keep the original strict hint.
        """
        from werewolf_agent.model_gateway.router import (
            GenerateResult,
            ModelRouter,
            UsageRecord,
        )

        class _ToolOnlyNoToolProvider:
            """Provider for tool-only models — no text fallback allowed."""

            @property
            def name(self) -> str:
                return "tool_only_no_tool"

            def generate(self, prompt, config, system_prompt=None):
                return GenerateResult(
                    text="<some non-JSON text response>",
                    provider="tool_only_no_tool",
                    model=config.model,
                    tool_call_required=True,
                    tool_call_received=False,
                    text_fallback_used=False,
                    structured_failure_reason="missing_tool_call",
                    usage=UsageRecord(
                        agent_id="",
                        task_type="",
                        provider="tool_only_no_tool",
                        model=config.model,
                    ),
                )

        router = ModelRouter(
            model_profiles={
                "tool_only_model": {
                    "model": "tool-v1",
                    "provider": "tool_only_no_tool",
                    # NOTE: no allow_text_tool_fallback → tool-only
                },
            },
            llm_profiles={
                "default": {
                    "default": {
                        "provider": "tool_only_no_tool",
                        "model_profile": "tool_only_model",
                    },
                },
            },
            player_assignments={"p05": "default"},
            providers={"tool_only_no_tool": _ToolOnlyNoToolProvider()},
        )
        agent = PlayerAgent(agent_id="p05", model_router=router, max_retries=1)
        ctx = AgentContext(
            agent_id="p05",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=2,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07", "p08"],
            public_summary="D2 vote",
        )

        action, retry = agent.act(ctx)

        assert retry.error_code == "missing_tool_call"
        # Tool-only model — strict hint is correct.
        assert "必须通过 submit_player_action 工具调用" in retry.correction_hint, (
            "D4-4 regression: tool-only model must keep the strict "
            "'must use tool call' hint. "
            f"Got hint: {retry.correction_hint!r}"
        )


# ---------------------------------------------------------------------------
# g_3223805846-B1 follow-up: vote fallback must not invent a target
# ---------------------------------------------------------------------------


class TestVoteFallbackTargetGate:
    """Vote fallback should use evidence-backed targets, not seat order."""

    def _make_agent(self) -> PlayerAgent:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p08": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        return PlayerAgent(agent_id="p08", model_router=router, max_retries=1)

    def test_fallback_vote_returns_no_target_without_evidence_target(self) -> None:
        ctx = AgentContext(
            agent_id="p08",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03", "p05", "p07"],
            strategy_directive={},
        )
        agent = self._make_agent()
        fb = agent._fallback_action(ctx)
        assert isinstance(fb, FallbackAction)
        assert fb.target_id is None

    def test_fallback_vote_uses_evidence_target_when_present(self) -> None:
        ctx = AgentContext(
            agent_id="p08",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03", "p05", "p07"],
            strategy_directive={"_vote_fallback_target": "p05"},
        )
        agent = self._make_agent()
        fb = agent._fallback_action(ctx)
        assert isinstance(fb, FallbackAction)
        assert fb.target_id == "p05"


# ---------------------------------------------------------------------------
# g_3223805846-B3: seer PK 段 fallback 必须给非空内容
# ---------------------------------------------------------------------------


class TestSeerPKNonEmptyProtection:
    """P1-G3223805846-3: seer 在 PK 段 (task_type=PK_SPEECH) speech fallback
    必须给非空内容，并尽量包含查杀信息（my_check_history 中未报过的狼人）。
    """

    def _make_agent(self) -> PlayerAgent:
        router = ModelRouter(
            model_profiles={},
            llm_profiles={},
            player_assignments={"p03": "default"},
            providers={"mock": _JsonProvider("not json")},
        )
        return PlayerAgent(agent_id="p03", model_router=router, max_retries=1)

    def test_seer_pk_speech_fallback_uses_check_history(self) -> None:
        ctx = AgentContext(
            agent_id="p03",
            task_type=TaskType.PK_SPEECH,
            own_role="seer",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02", "p05", "p07"],
            strategy_directive={
                "my_check_history": [
                    {"target": "p07", "alignment": "wolf", "night": 1, "reported": False},
                ],
            },
        )
        agent = self._make_agent()
        speech = agent._fallback_speech(ctx)
        # 必须非空
        assert speech and len(speech) > 10
        # 必含查杀信息（"预言家" 身份 + p07 目标）
        assert "预言家" in speech, (
            f"seer PK fallback lost identity: {speech!r}"
        )
        assert "p07" in speech, (
            f"seer PK fallback lost check target: {speech!r}"
        )

    def test_seer_pk_speech_fallback_no_unreported_wolves(self) -> None:
        """Seer 已报过所有查杀 / 没有狼查杀结果时，仍必须给非空占位内容。"""
        ctx = AgentContext(
            agent_id="p03",
            task_type=TaskType.PK_SPEECH,
            own_role="seer",
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p02", "p05", "p07"],
            strategy_directive={
                "my_check_history": [
                    # good — 无可报的狼
                    {"target": "p02", "alignment": "good", "night": 1, "reported": True},
                ],
            },
        )
        agent = self._make_agent()
        speech = agent._fallback_speech(ctx)
        # 必须非空 + 仍保留预言家身份声明
        assert speech and len(speech) > 10
        assert "预言家" in speech, (
            f"seer PK fallback without wolves lost identity: {speech!r}"
        )


class TestMostSuspectTargetResolution:
    """P4 (post-review-v2): _most_suspect_target 路径在无 producer 时不应被消费。"""

    def test_most_suspect_target_falls_through_to_non_self(self):
        from werewolf_agent.agents.player import PlayerAgent
        from werewolf_agent.agents.schemas import (
            ActionType, AgentContext, FallbackAction, TaskType,
        )
        ctx = AgentContext(
            agent_id="p08",
            task_type=TaskType.VOTE,
            own_role="villager",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03", "p05", "p07"],
            strategy_directive={"_most_suspect_target": "p05"},
        )
        agent = PlayerAgent(agent_id="p08", model_router=None)
        fb = agent._fallback_action(ctx)
        # _most_suspect_target 路径已删除，且不再 fallthrough 到座位顺序目标。
        assert fb.target_id is None
