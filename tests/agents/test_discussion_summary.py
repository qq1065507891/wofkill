# -*- coding: utf-8 -*-
"""
验证讨论摘要 V2 Schema、旧 checkpoint 迁移和文本兼容投影。

作者: Project contributors
创建日期: 2026-07-25
修改日期: 2026-07-27

使用示例:
    >>> python -m pytest tests/agents/test_discussion_summary.py -q
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.discussion_summary import (
    DiscussionSummary,
    DiscussionSummaryGenerationError,
    discussion_summary_for_player,
    discussion_summary_text,
    parse_discussion_summary_text,
)
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.model_gateway.usage_records import GenerateResult


def _v2_payload(summary: str = "我怀疑p03") -> dict[str, object]:
    return {
        "summary": summary,
        "suspected_players": ["p03"],
        "trusted_players": ["p02"],
        "vote_target": "p03",
        "evidence_refs": ["event-7"],
    }


def test_discussion_summary_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DiscussionSummary.model_validate({
            **_v2_payload(),
            "private_reason": "我是狼人，所以要推动p03。",
        })


@pytest.mark.parametrize(
    ("field", "coercible_value"),
    [
        ("suspected_players", ("p03",)),
        ("trusted_players", {"p02"}),
        ("evidence_refs", ("event-7",)),
    ],
)
def test_discussion_summary_rejects_coercible_container_types(
    field: str,
    coercible_value: object,
) -> None:
    with pytest.raises(ValidationError):
        DiscussionSummary.model_validate({
            **_v2_payload(),
            field: coercible_value,
        })


def test_unversioned_coercible_mapping_does_not_upgrade() -> None:
    payload = _v2_payload()
    payload["suspected_players"] = ("p03",)
    state = {"discussion_positions": {"p01": payload}}

    assert discussion_summary_for_player(state, "p01") is None
    assert "discussion_positions_version" not in state
    assert state["discussion_positions"]["p01"]["suspected_players"] == (
        "p03",
    )


def test_legacy_summary_string_upgrades_to_v2() -> None:
    state = {
        "discussion_positions": {
            "p01": "我怀疑p03",
            "p02": "我信任p04",
        },
    }

    summary = discussion_summary_for_player(state, "p01")

    assert summary == DiscussionSummary(summary="我怀疑p03")
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"] == {
        "p01": DiscussionSummary(summary="我怀疑p03").model_dump(),
        "p02": DiscussionSummary(summary="我信任p04").model_dump(),
    }


def test_unversioned_v2_mapping_upgrades_in_memory() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": _v2_payload("我信任p03"),
        },
    }

    summary = discussion_summary_for_player(state, "p02")

    assert summary == DiscussionSummary.model_validate(_v2_payload("我信任p03"))
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"]["p02"] == _v2_payload("我信任p03")


def test_explicit_v2_payload_is_read_without_coercion() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())


def test_v2_version_schema_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": "legacy string"},
    }

    assert discussion_summary_for_player(state, "p01") is None
    assert state["discussion_positions"]["p01"] == "legacy string"


def test_explicit_v2_global_mapping_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": "legacy string conflicts with explicit V2",
        },
    }

    assert discussion_summary_for_player(state, "p01") is None
    assert state["discussion_positions"]["p01"] == _v2_payload()
    assert state["discussion_positions"]["p02"] == (
        "legacy string conflicts with explicit V2"
    )


def test_unknown_explicit_version_fails_closed() -> None:
    state = {
        "discussion_positions_version": 3,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(state, "p01") is None


def test_mixed_unversioned_entries_validate_per_player_without_upgrade() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": {"summary": "字段冲突", "unexpected": True},
        },
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())
    assert discussion_summary_for_player(state, "p02") is None
    assert "discussion_positions_version" not in state


def test_missing_player_returns_none() -> None:
    assert discussion_summary_for_player(
        {"discussion_positions": {}},
        "p09",
    ) is None


def test_text_projection_is_deterministic_and_excludes_private_fields() -> None:
    summary = DiscussionSummary.model_validate(_v2_payload())

    first = discussion_summary_text(summary)
    second = discussion_summary_text(summary)

    assert first == second
    assert first == (
        "我怀疑p03\n"
        "怀疑玩家: p03\n"
        "信任玩家: p02\n"
        "投票目标: p03\n"
        "证据引用: event-7"
    )
    assert "private" not in first.lower()
    assert "我是狼人" not in first


def test_text_json_provider_request_contains_exact_narrow_schema() -> None:
    class _CapturingProvider:
        name = "capture"

        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def generate(
            self,
            prompt,
            config,
            system_prompt=None,
            tools=None,
            tool_choice=None,
            final_prompt_observer=None,
        ):
            self.requests.append({
                "prompt": prompt,
                "mode": config.structured_output_mode,
                "tools": tools,
                "tool_choice": tool_choice,
            })
            return GenerateResult(
                text=DiscussionSummary(
                    summary="p02发言前后矛盾。",
                    suspected_players=["p02"],
                    vote_target="p02",
                    evidence_refs=["speech-1"],
                ).model_dump_json(),
                provider=self.name,
                model=config.model,
            )

    provider = _CapturingProvider()
    router = ModelRouter(
        model_profiles={
            "summary_model": {
                "model": "summary-model",
                "structured_output": {"mode": "text_json"},
                "reasoning": {"level": "medium"},
                "retry_count": 0,
            },
        },
        llm_profiles={
            "player": {
                "tasks": {
                    "discussion_summary": {
                        "provider": "capture",
                        "model_profile": "summary_model",
                    },
                },
            },
        },
        player_assignments={"p01": "player"},
        providers={"capture": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router)

    summary = agent.summarize_discussion(AgentContext(
        agent_id="p01",
        task_type=TaskType.DISCUSSION_SUMMARY,
        strategy_directive={"transcript_text": "[p02]: 我怀疑p03。"},
    ))

    request = provider.requests[0]
    prompt = str(request["prompt"])
    assert summary.vote_target == "p02"
    assert request["mode"] == "json_object"
    assert request["tool_choice"] is None
    assert '"additionalProperties": false' in prompt
    assert '"required": ["summary"]' in prompt
    for field in DiscussionSummary.model_fields:
        assert f'"{field}"' in prompt
    for generic_field in ("action_type", "speech", "reason", "private_intent"):
        assert f'"{generic_field}"' not in prompt


@pytest.mark.parametrize(
    "raw_text",
    [
        '```json\n{"summary":"怀疑p03"}\n```',
        '\ufeff前置说明：\n{"summary":"怀疑p03"}\n以上是摘要。',
    ],
)
def test_parse_discussion_summary_text_accepts_fenced_bom_and_mixed_prose(
    raw_text: str,
) -> None:
    assert parse_discussion_summary_text(raw_text) == DiscussionSummary(
        summary="怀疑p03",
    )


def test_parse_discussion_summary_text_preserves_urls_in_summary() -> None:
    assert parse_discussion_summary_text(
        '{"summary":"查看 https://example.com/a"}'
    ).summary == "查看 https://example.com/a"


def test_parse_discussion_summary_text_accepts_mixed_prose_url() -> None:
    assert parse_discussion_summary_text(
        'See https://example.com before {"summary":"ok"}'
    ) == DiscussionSummary(summary="ok")


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"summary":"ok","unknown":true}',
        '{"suspected_players":[]}',
        '{"summary":"ok","suspected_players":"p03"}',
        '[{"summary":"ok"}]',
        '```json\n[{"summary":"ok"}]\n```',
        '前置说明：[{"summary":"ok"}]\n以上是结果。',
    ],
)
def test_parse_discussion_summary_text_rejects_invalid_schema(raw_text: str) -> None:
    with pytest.raises(ValidationError):
        parse_discussion_summary_text(raw_text)


def test_parse_discussion_summary_text_rejects_ambiguous_multiple_valid_objects() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        parse_discussion_summary_text(
            '{"summary":"第一个"} 中间 {"summary":"第二个"}'
        )


class _SummaryTextProvider:
    name = "summary-text"

    def __init__(
        self,
        texts: str | list[str],
        *,
        tool_call_received: bool = False,
        failure: Exception | None = None,
    ) -> None:
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        self.tool_call_received = tool_call_received
        self.failure = failure
        self.requests: list[dict[str, object]] = []

    def generate(
        self,
        prompt,
        config,
        system_prompt=None,
        tools=None,
        tool_choice=None,
        final_prompt_observer=None,
    ):
        self.requests.append({
            "prompt": prompt,
            "mode": config.structured_output_mode,
            "tool_choice": tool_choice,
        })
        if self.failure is not None:
            raise self.failure
        text = self.texts[min(len(self.requests) - 1, len(self.texts) - 1)]
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            tool_call_received=self.tool_call_received,
        )


def _summary_agent_for_provider(
    provider: _SummaryTextProvider,
    *,
    structured_output_mode: str = "text_json",
    retry_count: int = 0,
) -> PlayerAgent:
    router = ModelRouter(
        model_profiles={
            "summary_model": {
                "model": "summary-model",
                "structured_output": {"mode": structured_output_mode},
                "reasoning": {"level": "medium"},
                "retry_count": retry_count,
            },
        },
        llm_profiles={
            "player": {
                "tasks": {
                    "discussion_summary": {
                        "provider": provider.name,
                        "model_profile": "summary_model",
                    },
                },
            },
        },
        player_assignments={"p01": "player"},
        providers={provider.name: provider},
    )
    return PlayerAgent(agent_id="p01", model_router=router)


def _summary_agent_for_text(text: str) -> PlayerAgent:
    return _summary_agent_for_provider(_SummaryTextProvider(text))


def _summary_context() -> AgentContext:
    return AgentContext(
        agent_id="p01",
        task_type=TaskType.DISCUSSION_SUMMARY,
        strategy_directive={"transcript_text": "[p02]: 我怀疑p03。"},
    )


def test_player_summary_uses_repaired_text_parser() -> None:
    summary = _summary_agent_for_text(
        '```json\n{"summary":"怀疑p03"}\n```'
    ).summarize_discussion(_summary_context())

    assert summary.summary == "怀疑p03"


@pytest.mark.parametrize(
    ("raw_text", "failure_code"),
    [
        ("not json", "invalid_json"),
        ('{"summary":"ok","unknown":true}', "schema_validation_failed"),
        ('{"suspected_players":"p03"}', "schema_validation_failed"),
        ('{"summary":"one"} {"summary":"two"}', "invalid_json"),
    ],
)
def test_player_summary_maps_parser_failures_to_safe_codes(
    raw_text: str,
    failure_code: str,
) -> None:
    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_text(raw_text).summarize_discussion(_summary_context())

    assert exc_info.value.failure_code == failure_code


def test_player_summary_invalid_json_exposes_only_sanitized_audit_shape() -> None:
    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_text("not json").summarize_discussion(_summary_context())

    error = exc_info.value
    assert error.failure_code == "invalid_json"
    assert error.audit == {
        "failure_code": "invalid_json",
        "structured_output_mode": "json_object",
        "tool_call_required": False,
        "tool_call_received": False,
        "response_shape": "text",
        "json_candidate_count": 0,
        "failure_stage": "protocol",
    }
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "not json" not in repr(error.audit)


@pytest.mark.parametrize(
    ("raw_text", "failure_code", "failure_stage", "candidate_count"),
    [
        ('{"summary":"one"} {"summary":"two"}', "invalid_json", "protocol", 2),
        ('{"summary":"ok","unknown":true}', "schema_validation_failed", "schema", 1),
    ],
)
def test_player_summary_audit_classifies_parser_failure_stage(
    raw_text: str,
    failure_code: str,
    failure_stage: str,
    candidate_count: int,
) -> None:
    provider = _SummaryTextProvider([raw_text, raw_text])

    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_provider(
            provider,
            retry_count=9,
        ).summarize_discussion(_summary_context())

    assert len(provider.requests) == 2
    assert exc_info.value.audit["failure_code"] == failure_code
    assert exc_info.value.audit["failure_stage"] == failure_stage
    assert exc_info.value.audit["json_candidate_count"] == candidate_count


def test_player_summary_parses_native_tool_arguments_and_audits_metadata() -> None:
    provider = _SummaryTextProvider(
        '{"summary":"工具参数"}',
        tool_call_received=True,
    )

    summary = _summary_agent_for_provider(
        provider,
        structured_output_mode="native_tool",
    ).summarize_discussion(_summary_context())

    assert summary.summary == "工具参数"
    assert provider.requests[0]["mode"] == "native_tool"
    assert provider.requests[0]["tool_choice"] == {
        "type": "tool",
        "name": "submit_discussion_summary",
    }


def test_player_summary_repairs_once_with_shared_attempt_context(monkeypatch) -> None:
    from werewolf_agent.model_gateway.execution_records import (
        AttemptOutcome,
        RouteKind,
    )

    rejected_marker = "REJECTED_PRIVATE_RESPONSE"
    provider = _SummaryTextProvider([
        rejected_marker,
        '```json\n{"summary":"修复成功"}\n```',
    ])
    agent = _summary_agent_for_provider(provider)
    attempt_contexts: list[object] = []
    original_generate = agent.model_router.generate

    def capture_attempt_context(**kwargs):
        attempt_contexts.append(kwargs["generation_attempt_context"])
        return original_generate(**kwargs)

    monkeypatch.setattr(agent.model_router, "generate", capture_attempt_context)

    summary = agent.summarize_discussion(_summary_context())

    assert summary.summary == "修复成功"
    assert len(provider.requests) == 2
    assert len({id(context) for context in attempt_contexts}) == 1
    attempts = attempt_contexts[-1].attempts
    assert [item.route_kind for item in attempts] == [
        RouteKind.PRIMARY,
        RouteKind.REPAIR,
    ]
    assert [item.attempt_outcome for item in attempts] == [
        AttemptOutcome.FAILURE,
        AttemptOutcome.SUCCESS,
    ]
    assert provider.requests[1]["mode"] == "json_object"
    assert str(provider.requests[1]["prompt"]).endswith(
        "\n只输出一个符合 submit_discussion_summary Schema 的 JSON 对象；"
        "不要输出解释、数组或多个对象。"
    )
    assert rejected_marker not in str(provider.requests[1]["prompt"])


def test_player_summary_second_parse_failure_stops_after_two_calls() -> None:
    provider = _SummaryTextProvider(["not json", "still not json"])

    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_provider(
            provider,
            retry_count=9,
        ).summarize_discussion(_summary_context())

    assert len(provider.requests) == 2
    assert exc_info.value.failure_code == "invalid_json"


def test_player_summary_missing_native_tool_call_repairs_once_then_stops() -> None:
    provider = _SummaryTextProvider([
        '{"summary":"first text fallback"}',
        '{"summary":"second text fallback"}',
    ])

    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_provider(
            provider,
            structured_output_mode="native_tool",
            retry_count=9,
        ).summarize_discussion(_summary_context())

    assert len(provider.requests) == 2
    assert exc_info.value.audit == {
        "failure_code": "missing_tool_call",
        "structured_output_mode": "native_tool",
        "tool_call_required": True,
        "tool_call_received": False,
        "response_shape": "json_object",
        "json_candidate_count": 1,
        "failure_stage": "protocol",
    }


def test_player_summary_supports_legacy_provider_signature_through_router() -> None:
    class _LegacyProvider:
        name = "legacy-summary"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, config, system_prompt=None):
            self.calls += 1
            return GenerateResult(
                text='{"summary":"legacy provider ok"}',
                provider=self.name,
                model=config.model,
            )

    provider = _LegacyProvider()
    router = ModelRouter(
        model_profiles={
            "summary_model": {
                "model": "summary-model",
                "structured_output": {"mode": "text_json"},
                "reasoning": {"level": "medium"},
                "retry_count": 0,
            },
        },
        llm_profiles={
            "player": {
                "tasks": {
                    "discussion_summary": {
                        "provider": provider.name,
                        "model_profile": "summary_model",
                    },
                },
            },
        },
        player_assignments={"p01": "player"},
        providers={provider.name: provider},
    )

    summary = PlayerAgent("p01", router).summarize_discussion(_summary_context())

    assert summary.summary == "legacy provider ok"
    assert provider.calls == 1


def test_player_summary_empty_response_is_not_repaired() -> None:
    provider = _SummaryTextProvider("")

    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_provider(
            provider,
            retry_count=9,
        ).summarize_discussion(_summary_context())

    assert len(provider.requests) == 1
    assert exc_info.value.audit == {
        "failure_code": "empty_response",
        "structured_output_mode": "json_object",
        "tool_call_required": False,
        "tool_call_received": False,
        "response_shape": "empty",
        "json_candidate_count": 0,
        "failure_stage": "provider",
    }


def test_player_summary_provider_failure_is_not_repaired(monkeypatch) -> None:
    provider = _SummaryTextProvider(
        "unused",
        failure=TimeoutError("PRIVATE_PROVIDER_FAILURE"),
    )
    monkeypatch.setattr(
        "werewolf_agent.model_gateway.router.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
        _summary_agent_for_provider(
            provider,
            retry_count=9,
        ).summarize_discussion(_summary_context())

    assert len(provider.requests) == 1
    assert exc_info.value.failure_code == "model_generation_failed"
    assert exc_info.value.audit["failure_stage"] == "provider"
    assert "PRIVATE_PROVIDER_FAILURE" not in repr(exc_info.value.audit)


def test_summary_error_discards_unsafe_audit_fields_and_values() -> None:
    error = DiscussionSummaryGenerationError(
        "invalid_json",
        audit={
            "response_shape": "text",
            "json_candidate_count": 0,
            "raw_text": "PRIVATE_RAW_TEXT",
            "prompt": "PRIVATE_PROMPT",
            "exception": RuntimeError("PRIVATE_EXCEPTION"),
        },
    )

    assert str(error) == "invalid_json"
    assert error.audit == {
        "failure_code": "invalid_json",
        "response_shape": "text",
        "json_candidate_count": 0,
    }
    assert "PRIVATE" not in repr(error.audit)
